from typing import Dict, Optional
import boto3
import json
import pandas as pd
from io import StringIO
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from modal_controller.modal_runner import ModalRunner, Operation
from modal_controller.constants import *


def get_s3_result(uuid: str, operation: Operation, bucket_name: Optional[str] = None) -> dict:
    """Fetch the JSON result from S3 for a given UUID and operation type."""
    s3_client = boto3.client('s3')
    bucket_name = bucket_name or S3_BUCKET_NAME
    result_type = {
        Operation.PLAN: "plan",
        Operation.VALIDATE: "validate",
        Operation.EXECUTE: "execute",
        Operation.EVALUATE: "evaluation",
        Operation.FETCH_SCHEMA: "fetch_schema",
    }[operation]
    key = f"{result_type}-results/{uuid}.json"
    
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    return json.loads(response['Body'].read())


def read_uuid_result(operation: Operation, uuid: Optional[str] = None, bucket_name: Optional[str] = None) -> dict:
    """Read a result from S3 for the given *uuid*."""
    result = {}
    if uuid:
        try:
            result = get_s3_result(uuid, operation, bucket_name)
        except Exception as e:
            result = {"error": f"Error retrieving result from S3 for UUID {uuid}: {e}"}
    else:
        result = {"error": "No UUID found in output."}
    return result

def compare_result_sets(baseline_data: str, model_data: str, rounding: int = 6) -> bool:
    """Compare two JSON result sets for equality (order-independent).

    Parses both as DataFrames, sorts rows, rounds floats to *rounding*
    decimals, and checks element-wise equality (treating NaN == NaN).

    Returns True if result sets match, False otherwise.
    """
    try:
        baseline_df = pd.read_json(StringIO(baseline_data))
        model_df = pd.read_json(StringIO(model_data))
        
        if sorted(baseline_df.columns.tolist()) != sorted(model_df.columns.tolist()):
            return False
        
        model_df = model_df[baseline_df.columns]
        
        if len(baseline_df) != len(model_df):
            return False
        
        try:
            baseline_sorted = baseline_df.sort_values(by=list(baseline_df.columns)).reset_index(drop=True)
            model_sorted = model_df.sort_values(by=list(model_df.columns)).reset_index(drop=True)
        except Exception:
            baseline_sorted = baseline_df.reset_index(drop=True)
            model_sorted = model_df.reset_index(drop=True)
        
        float_cols = baseline_sorted.select_dtypes(include=["floating"]).columns
        baseline_sorted[float_cols] = baseline_sorted[float_cols].round(rounding)
        model_sorted[float_cols] = model_sorted[float_cols].round(rounding)

        eq = baseline_sorted.eq(model_sorted)
        both_na = baseline_sorted.isna() & model_sorted.isna()
        diff_mask = ~(eq | both_na)
        all_equal = not diff_mask.to_numpy().any()

        return all_equal
        
    except Exception:
        return False


def validate_plan_result_set(
    candidate_plan_str: str,
    baseline_plan_str: str,
    data_folder: str,
    n_retry: int = 5,
    n_determinism_retries: int = 3,
    **kwargs,
) -> Optional[str]:
    """Execute *candidate_plan_str* and *baseline_plan_str* and compare result sets.

    Submits 1 baseline EXECUTE + ``n_determinism_retries`` candidate EXECUTEs
    in parallel via a per-call ThreadPoolExecutor, then stream-compares the
    results as each S3 fetch returns (compare-as-you-fetch). Holds at most
    ~2-3 result_data string references in memory at any time.

    Compare semantics:
      - The first candidate run to arrive becomes the "primary candidate"
        that defines the determinism reference.
      - baseline ↔ primary candidate  → correctness check.
      - primary candidate ↔ each other candidate run → determinism check.

    Returns None on success, or an error string on execution failure,
    result-set mismatch, or non-determinism. n_retry defaults to 5 because
    these EXECUTEs are one-shot (no R-redundancy below them); eval-phase
    EXECUTEs use the submit_run_operation default of 2 (R=n_runs absorbs
    transient flakes there).
    """
    n_total = 1 + n_determinism_retries  # 1 baseline + N candidate EXECUTEs

    def _submit_one(kind: str, plan_str: str):
        r = submit_run_operation(
            Operation.EXECUTE, plan_str, data_folder, n_retry=n_retry, **kwargs,
        )
        return kind, r

    baseline_data: Optional[str] = None       # set once baseline EXECUTE arrives
    primary_cand_data: Optional[str] = None   # set by the FIRST candidate to arrive
    error_to_return: Optional[str] = None

    pool = ThreadPoolExecutor(max_workers=n_total)
    futures = []
    try:
        futures.append(pool.submit(_submit_one, "base", baseline_plan_str))
        for _ in range(n_determinism_retries):
            futures.append(pool.submit(_submit_one, "cand", candidate_plan_str))

        for fut in as_completed(futures):
            try:
                kind, r = fut.result()
            except Exception as exc:
                error_to_return = f"Validation failed: {exc}"
                break

            if not r or r.get("error"):
                if kind == "base":
                    error_to_return = (
                        f"Baseline execution failed: {(r or {}).get('error', 'Unknown error')}"
                    )
                else:
                    error_to_return = (
                        f"Candidate execution failed: {(r or {}).get('error', 'Unknown error')}"
                    )
                break

            if kind == "base":
                if primary_cand_data is not None:
                    # A candidate already became primary -> correctness compare now.
                    if not compare_result_sets(r["result_data"], primary_cand_data):
                        error_to_return = (
                            "Result set mismatch: candidate plan output does not match "
                            "baseline plan output."
                        )
                        r.pop("result_data", None)
                        break
                    r.pop("result_data", None)
                else:
                    # Stash baseline; correctness compare deferred to first cand arrival.
                    baseline_data = r["result_data"]
                    r.pop("result_data", None)
            else:  # kind == "cand"
                if primary_cand_data is None:
                    # First candidate to arrive becomes the determinism reference.
                    primary_cand_data = r["result_data"]
                    r.pop("result_data", None)
                    if baseline_data is not None:
                        if not compare_result_sets(baseline_data, primary_cand_data):
                            error_to_return = (
                                "Result set mismatch: candidate plan output does not match "
                                "baseline plan output."
                            )
                            baseline_data = None
                            break
                        baseline_data = None  # drop baseline ref after correctness check
                else:
                    # Subsequent candidates: determinism check vs the primary.
                    if not compare_result_sets(primary_cand_data, r["result_data"]):
                        error_to_return = (
                            "Non-deterministic plan: candidate plan produced different "
                            "results across executions."
                        )
                        r.pop("result_data", None)
                        break
                    r.pop("result_data", None)

        return error_to_return

    finally:
        # Drop any remaining payload refs, and best-effort cancel pending
        # futures. In-flight Modal sandboxes finish in the background and
        # self-clean via their sandbox_timeout; we don't block on them.
        baseline_data = None
        primary_cand_data = None
        for fut in futures:
            fut.cancel()
        pool.shutdown(wait=False, cancel_futures=True)


# ModalRunner cache — avoids repeated modal.App.lookup() and image definition
# construction. Keyed by (app_name, scale_factor). Thread-safe.
_runner_cache: dict[tuple, ModalRunner] = {}
_runner_cache_lock = threading.Lock()

def _get_runner(app_name: str, runner_kwargs: dict) -> ModalRunner:
    """Return a cached ModalRunner, creating one if needed.

    Bypasses the cache when ``rebuild_image`` is True in *runner_kwargs*.
    """
    scale_factor = runner_kwargs.get("scale_factor", DEFAULT_SCALE_FACTOR)
    rebuild_image = runner_kwargs.get("rebuild_image", False)
    key = (app_name, scale_factor)
    with _runner_cache_lock:
        if rebuild_image or key not in _runner_cache:
            _runner_cache[key] = ModalRunner(app_name, **runner_kwargs)
        return _runner_cache[key]

# Protects the scheduling state
_REMOTE_SCHED_LOCK = threading.Lock()
_SANDBOX_SEM = threading.Semaphore(MAX_CONCURRENT_SANDBOXES)
_latest_scheduled_start = 0.0  # monotonic time

def _gate_modal_start():
    """Rate-limit Modal sandbox starts to at most RATE_LIMIT_PER_SEC per second."""
    global _latest_scheduled_start

    now0 = time.monotonic()
    with _REMOTE_SCHED_LOCK:
        scheduled = max(_latest_scheduled_start, now0)
        WAIT_TIME = 1.0 / RATE_LIMIT_PER_SEC
        scheduled += random.uniform(0.0, 0.2 * WAIT_TIME) # add some jitter to ensure max safety
        _latest_scheduled_start = scheduled + WAIT_TIME

    now1 = time.monotonic()
    time.sleep(max(scheduled - now1, 0.0))

def submit_run_operation(
    operation: Operation,
    input_str: str,
    data_folder: str,
    n_retry: int = 2,
    runner_kwargs: dict | None = None,
    verbose: bool = True,
    **kwargs
):
    """Run an operation on Modal with rate limiting, retries, and error mapping.

    Creates a ModalRunner sandbox, executes the operation, retrieves the
    result from S3, and retries on transient errors (up to *n_retry* times).

    Args:
        operation: Operation type (PLAN, VALIDATE, EXECUTE, EVALUATE, FETCH_SCHEMA).
        input_str: Input string (SQL query or plan JSON).
        data_folder: Modal-side data directory.
        n_retry: Max retry attempts on transient errors.
        runner_kwargs: Extra kwargs for ModalRunner constructor.
        verbose: Print retry messages.
        **kwargs: Forwarded to ``ModalRunner.run_operation``.

    Returns:
        Result dict from S3, or ``{"error": ...}`` on failure.
    """
    if runner_kwargs is None:
        runner_kwargs = {}
        
    def run_attempt():
        """Run sandbox and return UUID string on success, or error dict on failure."""
        runner = _get_runner(SANDBOX_NAME, runner_kwargs or {})
        try:
            uuid = runner.run_operation(
                operation=operation,
                input_str=input_str,
                data_folder=data_folder,
                **kwargs
            )
        except Exception as exc:
            return {"error": str(exc)}

        if uuid is None:
            return {"error": "No UUID found in output."}
        return uuid

    if n_retry < 0:
        raise ValueError("n_retry must be non-negative")
    result = None

    for i in range(n_retry + 1):
        if i > 0:
            time.sleep(min(2 ** i, 10))  # ~exponential backoff

        # Semaphore caps concurrent Modal sandboxes; rate gate inside it
        # spaces consecutive starts by >= 1/RATE_LIMIT_PER_SEC.
        with _SANDBOX_SEM:
            _gate_modal_start()
            uuid_or_error = run_attempt()

        # S3 result fetch happens OUTSIDE the semaphore — no need to hold
        # a sandbox slot while waiting on an HTTP call to AWS.
        if isinstance(uuid_or_error, str):
            result = read_uuid_result(operation, uuid=uuid_or_error)
        else:
            result = uuid_or_error

        if result and not result.get("error", None):
            break

        if not result:
            if i == n_retry:
                return {"error": "No result returned from operation after retries."}
            continue

        if i == n_retry: # final attempt, don't retry again
            break

        error_kw = next((kw for kw in RETRY_DEFAULT_ERROR_KWS if kw in result.get("error", "")), None)
        if error_kw and (not error_kw in RETRY_ONLY_ONCE or i == 0):
            continue

        break
    
    if result and result.get("error", None):
        error_kw = next((kw for kw in DEFAULT_ERROR_KWS_TO_MESSAGES.keys() if kw in result["error"]), None)
        if error_kw:
            result["error"] = DEFAULT_ERROR_KWS_TO_MESSAGES.get(error_kw, result["error"])  # map if known

    return result


METRIC_STAT_KEYS = ("min", "p10", "p25", "p50", "p75", "p90", "max", "mean", "std")

def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linearly interpolate the *p*-th percentile from pre-sorted *sorted_vals*."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    if p <= 0:
        return sorted_vals[0]
    if p >= 1:
        return sorted_vals[-1]
    rank = p * (len(sorted_vals) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_vals) - 1)
    if low == high:
        return sorted_vals[low]
    weight = rank - low
    return sorted_vals[low] * (1 - weight) + sorted_vals[high] * weight


def compute_metric_stats(values: list[float]) -> dict:
    """Compute summary statistics (min/max/mean/std/percentiles) for *values*.

    Returns a dict with keys from ``METRIC_STAT_KEYS``.
    """
    values = [v for v in values if v is not None]
    if not values:
        return {k: None for k in METRIC_STAT_KEYS}
    values_sorted = sorted(values)
    mean_value = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((t - mean_value) ** 2 for t in values) / len(values)
        std_value = variance ** 0.5
    else:
        std_value = 0.0
    return {
        "min": values_sorted[0],
        "p10": _percentile(values_sorted, 0.10),
        "p25": _percentile(values_sorted, 0.25),
        "p50": _percentile(values_sorted, 0.50),
        "p75": _percentile(values_sorted, 0.75),
        "p90": _percentile(values_sorted, 0.90),
        "max": values_sorted[-1],
        "mean": mean_value,
        "std": std_value,
    }


def evaluate_plan_n_runs(
    operation: Operation,
    plan_json: str,
    data_folder: str,
    n_runs: int = 1,
    max_eval_workers: int = 8,
    runner_kwargs: dict | None = None,
    **kwargs
) -> dict:
    """Execute a plan *n_runs* times concurrently and aggregate metrics.

    Each run is submitted via ``submit_run_operation``. Metric values from
    all successful runs are aggregated into summary statistics.

    Args:
        operation: Operation type (typically EVALUATE).
        plan_json: Plan JSON string to execute.
        data_folder: Modal-side data directory.
        n_runs: Number of evaluation runs (must be >= 1).
        max_eval_workers: Max concurrent evaluation workers.
        runner_kwargs: Extra kwargs for ModalRunner.
        **kwargs: Forwarded to ``submit_run_operation``.

    Returns:
        Dict with ``n_runs``, ``benchmark_stats`` (per-metric summaries),
        and ``error`` (if all runs failed).
    """

    if runner_kwargs is None:
        runner_kwargs = {}

    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")

    benchmark_stats: dict[str, object] = {}

    # Keep "one of the errors" around in case everything fails.
    best_error = None
    best_error_rank = -1

    def _rank_error(err: str | None) -> int:
        if not err:
            return -1
        # Prefer concrete platform/model errors over "no result", and prefer both over exceptions.
        if "Exception during evaluation process!" in err:
            return 0
        if "No Result returned from AWS (S3)" in err:
            return 1
        return 2

    def run_single_evaluation(run_id: int) -> tuple[int, dict | None, str | None]:
        try:
            # Eval/timing path uses the submit_run_operation default (n_retry=2)
            # because R=n_runs redundancy absorbs transient flakes. The
            # validation/determinism path overrides this to n_retry=5 because
            # those EXECUTEs are one-shot with no replay below them.
            result = submit_run_operation(
                operation=operation,
                input_str=plan_json,
                data_folder=data_folder,
                runner_kwargs=runner_kwargs,
                **kwargs,
            )

            if not result:
                return (run_id, None, "Failed to evaluate plan: No Result returned from AWS (S3)!")

            if result.get("error", None):
                return (run_id, None, f"Failed to evaluate plan: {result['error']}")

            return (run_id, result, None)

        except Exception:
            return (run_id, None, "Failed to evaluate plan: Exception during evaluation process!")

    total_successful_runs: int = 0

    with ThreadPoolExecutor(max_workers=max_eval_workers) as executor:
        future_to_id = {executor.submit(run_single_evaluation, i + 1): i + 1 for i in range(n_runs)}

        for future in as_completed(future_to_id):
            run_id, evaluation_results, err = future.result()

            if err is not None:
                r = _rank_error(err)
                if r > best_error_rank:
                    best_error_rank = r
                    best_error = err

            if evaluation_results is not None:
                total_successful_runs += 1
                for k, v in evaluation_results.items():
                    if k not in benchmark_stats:
                        benchmark_stats[k] = {'all_runs': []}
                    benchmark_stats[k]['all_runs'].append(v)

    # If nothing succeeded, return one of the errors we observed.
    if total_successful_runs == 0:
        return {
            "n_runs": 0,
            "benchmark_stats": {},
            "error": best_error or "Failed to evaluate plan: All evaluation attempts failed!",
        }

    return {
        "n_runs": total_successful_runs,
        "benchmark_stats": {
            k: {
                'all_runs': v['all_runs'],
                **compute_metric_stats(v['all_runs'])
            }
            for k, v in benchmark_stats.items()
        }
    }
