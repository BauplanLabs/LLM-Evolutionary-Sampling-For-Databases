from __future__ import annotations

from dotenv import load_dotenv
load_dotenv(override=False)

import copy
from typing import Any, Dict, List, Optional, Sequence, Union, Literal, Tuple
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import jsonpatch
import tempfile
from modal_controller.constants import DEFAULT_SCALE_FACTOR
from modal_controller.modal_runner import Operation
from modal_controller.utils import submit_run_operation, validate_plan_result_set, compute_metric_stats
from sampling.plan_scaler import transfer_plan
from dbplanbench_types import (
    QueryGenerationResult,
    PatchedPlan,
    OptimizationResult,
    BenchmarkResult,
    ScaleResult,
    PlanningResult,
    QueryValidationResult,
    Patch,
)
from dbplanbench_utils import (
    write_json,
    resolve_run_dir,
    get_evaluation_stats,
    get_metric_value,
    log_line,
    data_folder_for_dataset,
    ensure_local_data,
    plan_to_json,
    build_validation_stats,
    format_metric_stats,
)


def generate_queries(
    *,
    dataset: str,
    complexity_distribution: Dict[int, Union[int, float]],
    n_queries: int,
    run_dir: Optional[str] = None,
    include_sample_rows_in_prompt: bool = False,
    max_concurrent_generations: int = 50,
    max_concurrent_validators: int = 160,
    scale_factor: int = DEFAULT_SCALE_FACTOR,
    max_steps: int = 25,
    oversample_cap: float = 3.0,
    resume: bool = True,
    validate_kwargs: Optional[Dict[str, Any]] = None,
    exec_local: bool = False,
    verbose: bool = True,
) -> QueryGenerationResult:
    """
    Generate and validate SQL queries using LLM generation.

    Iteratively generates queries via an LLM to satisfy the requested complexity
    distribution, then validates each query for determinism and correctness on the
    target dataset. Queries that produce empty result sets or non-deterministic
    outputs (checked via multiple executions) are rejected. Schema fetch and
    validation run on Modal by default, or locally when ``exec_local=True``.

    Args:
        dataset: Dataset name (e.g., "tpch", "tpcds").
        complexity_distribution: Mapping of complexity level to count or weight.
        n_queries: Total number of valid queries to produce.
        run_dir: Output directory. Defaults to a timestamped dir under outputs/.
        include_sample_rows_in_prompt: Include sample rows in the LLM prompt.
        max_concurrent_generations: Max concurrent LLM generation workers.
        max_concurrent_validators: Max concurrent validation workers.
        scale_factor: Scale factor for the Modal runner during validation.
        max_steps: Maximum generation/validation iterations before stopping.
        oversample_cap: Upper bound on step-wise oversampling multiplier.
        resume: Reuse existing run-dir files when possible.
        validate_kwargs: Extra kwargs forwarded to the validation backend.
        exec_local: Fetch the schema and validate generated queries locally
            (via LocalRunner) instead of on Modal. Requires the ``local`` extra;
            missing data is generated under ``data/`` on first use.
        verbose: Print progress and step summaries.

    Returns:
        QueryGenerationResult with run_dir, summary (overall + by_complexity
        counts), queries (list of SQL strings), and per-query metadata.
    """
    from query_gen.orchestrator import run_query_generation
    if dataset.lower() not in {"tpch", "tpcds"}:
        raise ValueError("dataset must be one of: tpch, tpcds")
    if n_queries <= 0:
        raise ValueError("n_queries must be > 0")
    if max_concurrent_generations < 1:
        raise ValueError("max_concurrent_generations must be >= 1")
    if max_concurrent_validators < 1:
        raise ValueError("max_concurrent_validators must be >= 1")
    if scale_factor < 1:
        raise ValueError("scale_factor must be >= 1")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if oversample_cap < 1.0:
        raise ValueError("oversample_cap must be >= 1.0")
    if validate_kwargs is not None and not isinstance(validate_kwargs, dict):
        raise ValueError("validate_kwargs must be a dict or None")

    normalized_distribution: Dict[int, float] = {}
    for raw_c, raw_v in complexity_distribution.items():
        try:
            c = int(raw_c)
        except Exception as exc:
            raise ValueError(f"Invalid complexity key: {raw_c}") from exc
        try:
            v = float(raw_v)
        except Exception as exc:
            raise ValueError(
                f"Invalid complexity weight/count for complexity {raw_c}: {raw_v}"
            ) from exc
        if v < 0:
            raise ValueError("complexity_distribution values must be non-negative")
        normalized_distribution[c] = v

    if not normalized_distribution:
        raise ValueError("complexity_distribution cannot be empty")
    if sum(normalized_distribution.values()) <= 0:
        raise ValueError("complexity_distribution must sum to > 0")

    if exec_local:
        ensure_local_data(dataset.lower(), int(scale_factor))

    return run_query_generation(
        dataset=dataset.lower(),
        complexity_distribution=normalized_distribution,
        n_queries=int(n_queries),
        run_dir=run_dir,
        include_sample_rows_in_prompt=bool(include_sample_rows_in_prompt),
        max_concurrent_generations=int(max_concurrent_generations),
        max_concurrent_validators=int(max_concurrent_validators),
        scale_factor=int(scale_factor),
        max_steps=int(max_steps),
        oversample_cap=float(oversample_cap),
        resume=bool(resume),
        validate_kwargs=dict(validate_kwargs or {}),
        exec_local=bool(exec_local),
        verbose=bool(verbose),
    )


def _build_early_exit_result(
    message: str,
    failures: List[Optional[str]],
    config_current: Dict[str, Any],
    result_path: Path,
    queries_list: List[str],
    run_dir_path: Path,
    verbose: bool,
) -> OptimizationResult:
    """Build an OptimizationResult for early exit due to validation/evaluation failure.

    Writes a partial result.json (with ``optimization_outcome=None``) and returns
    an ``OptimizationResult`` whose ``optimization_outcome`` and ``metadata`` are None.
    """
    log_line(verbose, message)
    stats = build_validation_stats(failures)
    run_summary = {**config_current, "n_sampled_plans": 0}
    summary = {
        "validation_stats": stats,
        "run": run_summary,
        "improvement_x_stats": {},
        "outcome_rates": {},
    }
    write_json(
        result_path,
        {"queries": queries_list, "optimization_outcome": None, "summary": summary, "metadata": None},
    )
    return OptimizationResult(
        run_dir=str(run_dir_path),
        summary=summary,
        queries=list(queries_list),
        optimization_outcome=None,
        metadata=None,
        validation_failures=failures,
    )


def _plan_queries_with_dedup(
    queries_list: List[str],
    dataset: str,
    scale_factor: int,
    max_workers: int,
    validate_kwargs: Dict[str, Any],
) -> Tuple[List[Optional[Dict[str, Any]]], List[Optional[str]]]:
    """Obtain engine plans for *queries_list*, rejecting all occurrences of duplicate queries.

    Returns ``(plans, failures)`` aligned to *queries_list*. Duplicate queries
    receive ``failures[i] = "duplicate"`` and ``plans[i] = None``.
    """
    seen: set = set()
    duplicates: set = set()
    for q in queries_list:
        (duplicates if q in seen else seen).add(q)

    non_dup_indices = [i for i, q in enumerate(queries_list) if q not in duplicates]
    non_dup_queries = [queries_list[i] for i in non_dup_indices]

    plans: List[Optional[Dict[str, Any]]] = [None] * len(queries_list)
    failures: List[Optional[str]] = [None] * len(queries_list)

    for i, q in enumerate(queries_list):
        if q in duplicates:
            failures[i] = "duplicate"

    if non_dup_queries:
        planning_result = get_engine_plans(
            non_dup_queries,
            dataset=dataset,
            scale_factor=scale_factor,
            max_workers=max_workers,
            verbose=False,
            **validate_kwargs,
        )
        for j, orig_idx in enumerate(non_dup_indices):
            plans[orig_idx] = planning_result.plans[j]
            failures[orig_idx] = planning_result.errors[j]

    return plans, failures


def _validate_and_plan_queries(
    queries_list: List[str],
    dataset: str,
    scale_factor: int,
    max_workers: int,
    validate_kwargs: Dict[str, Any],
    skip_validation: bool,
    base_sampling_data: Optional[List[Dict[str, Any]]],
    run_dir_path: Path,
    verbose: bool,
) -> Tuple[List[Optional[Dict[str, Any]]], List[Optional[str]]]:
    """Validate queries and obtain their engine execution plans.

    When *base_sampling_data* is provided (resume path), verifies query
    consistency and returns all-None plans/failures.  Otherwise validates
    via ``_plan_queries_with_dedup`` (skip_validation) or the full
    ``validate_queries`` pipeline.

    Returns ``(engine_plans, validation_failures)`` aligned to *queries_list*.
    """
    if base_sampling_data is not None:
        existing_queries = [entry.get("query") for entry in base_sampling_data]
        if existing_queries != list(queries_list):
            raise ValueError("queries do not match existing base_sampling.json for resume=True")
        return [None] * len(queries_list), [None] * len(queries_list)

    from query_gen.query_validator import validate_queries as _validate_queries_fn

    with tempfile.TemporaryDirectory(dir=run_dir_path, prefix="tmp_validate_") as tmpdir:
        tmp_path = Path(tmpdir)
        input_queries_path = tmp_path / "input_queries.json"
        validated_queries_path = tmp_path / "validated_queries.json"
        dead_letter_path = tmp_path / "validation_dead_letter.json"

        if skip_validation:
            log_line(verbose, "Warning: skip_validation=True; queries will not be validated for determinism or correctness.")
            return _plan_queries_with_dedup(queries_list, dataset, scale_factor, max_workers, validate_kwargs)

        input_payload = [
            {"id": i, "query": q, "complexity": 0}
            for i, q in enumerate(queries_list)
        ]
        write_json(input_queries_path, input_payload)

        _validate_queries_fn(
            input_file=str(input_queries_path),
            output_file=str(validated_queries_path),
            dead_letter_file=str(dead_letter_path),
            dataset=dataset,
            max_workers=max_workers,
            output_format="plans",
            verbose=verbose,
            log_context="optimization",
            **validate_kwargs,
        )

        validation_failures: List[Optional[str]] = [None] * len(queries_list)
        error_map: Dict[str, str] = {}
        if dead_letter_path.exists():
            dead = json.loads(dead_letter_path.read_text())
            for err_type, queries_with_err in dead.items():
                for q in queries_with_err:
                    error_map[q] = err_type

        validated_data = []
        if validated_queries_path.exists():
            validated_data = json.loads(validated_queries_path.read_text())
        validated_by_id = {int(item["id"]): item for item in validated_data}

        for i, q in enumerate(queries_list):
            if q in error_map:
                validation_failures[i] = error_map[q]
            elif i not in validated_by_id:
                validation_failures[i] = "validation_missing"
            else:
                validation_failures[i] = None

        engine_plans = [
            validated_by_id[i]["plan"] if i in validated_by_id else None
            for i in range(len(queries_list))
        ]

    return engine_plans, validation_failures


def _integrate_user_base_plans(
    user_base_plans: Optional[List[Optional[Dict[str, Any]]]],
    engine_plans: List[Optional[Dict[str, Any]]],
    queries_list: List[str],
    dataset: str,
    validate_kwargs: Dict[str, Any],
    max_workers: int,
    skip_validation: bool,
    verbose: bool,
) -> Tuple[List[Optional[Dict[str, Any]]], List[str], Optional[List[Optional[str]]]]:
    """Validate and integrate user-provided base plans into *engine_plans*.

    When *user_base_plans* is None, returns the engine plans unchanged with
    ``base_plan_sources = ["engine", ...]``.  Otherwise validates each user plan
    against the corresponding engine plan via result-set comparison, and
    overwrites engine plans where user plans are provided.

    Returns ``(plans, base_plan_sources, user_plan_failures)``.
    ``user_plan_failures`` is None on success, or a per-query error list on failure.
    """
    n = len(queries_list)
    if user_base_plans is None:
        return engine_plans, ["engine"] * n, None

    if not skip_validation:
        log_line(verbose, "Validating user-provided base plans...")
        data_folder = data_folder_for_dataset(dataset)
        user_plan_failures: List[Optional[str]] = [None] * n

        def _validate_one(idx: int) -> Tuple[int, Optional[str]]:
            user_plan = user_base_plans[idx]
            if user_plan is None:
                return idx, None
            engine_plan = engine_plans[idx]
            if engine_plan is None:
                return idx, "Cannot validate: engine plan is None"
            return idx, validate_plan_result_set(
                plan_to_json(user_plan),
                plan_to_json(engine_plan),
                data_folder,
                n_retry=5,
                **validate_kwargs,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_validate_one, i): i for i in range(n)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Validating user plans", disable=not verbose):
                idx, err = future.result()
                user_plan_failures[idx] = err

        if any(err is not None for err in user_plan_failures):
            return engine_plans, ["engine"] * n, user_plan_failures

        log_line(verbose, "User base plans validated successfully.")
    else:
        log_line(verbose, "Warning: skip_validation=True; user-provided base plans will not be validated.")

    for i in range(n):
        if user_base_plans[i] is not None:
            engine_plans[i] = user_base_plans[i]

    base_plan_sources = [
        "custom" if user_base_plans[i] is not None else "engine"
        for i in range(n)
    ]
    return engine_plans, base_plan_sources, None


def _evaluate_and_curate_base_plans(
    queries_list: List[str],
    engine_plans: List[Optional[Dict[str, Any]]],
    curated_path: Path,
    dataset: str,
    scale_factor: int,
    n_runs: int,
    max_workers: int,
    evaluation_kwargs: Dict[str, Any],
    base_sampling_data: Optional[List[Dict[str, Any]]],
    resume: bool,
    verbose: bool,
) -> Optional[List[Optional[str]]]:
    """Benchmark base engine plans and write a curated dataset for sampling.

    Tries to reuse existing curated data on disk when resuming.  Otherwise
    benchmarks each base plan, checks for evaluation failures, and writes the
    curated dataset to *curated_path* (the curated data is consumed from disk by
    the sampling phase, so it is not returned).

    Returns the per-query ``evaluation_failures`` list (``None`` entry = OK), or
    ``None`` when no fresh evaluation was needed (data reused from disk or
    *base_sampling_data* exists).
    """
    curated_data: Optional[List[Dict[str, Any]]] = None
    if base_sampling_data is None and resume and curated_path.exists():
        curated_data = json.loads(curated_path.read_text())
        curated_by_id = {int(item["id"]): item for item in curated_data if "id" in item}
        if len(curated_by_id) != len(queries_list):
            curated_data = None
        else:
            for i, q in enumerate(queries_list):
                if i not in curated_by_id or curated_by_id[i].get("query") != q:
                    curated_data = None
                    break

    if base_sampling_data is not None or curated_data is not None:
        return None

    base_plan_entries = [
        PatchedPlan(base_plan=engine_plans[i], patch=[[]])
        for i in range(len(queries_list))
    ]

    base_benchmark = benchmark_plans(
        base_plan_entries,
        dataset=dataset,
        scale_factor=scale_factor,
        n_runs=n_runs,
        max_workers=max_workers,
        verbose=verbose,
        **evaluation_kwargs,
    )

    evaluation_failures: List[Optional[str]] = [None] * len(queries_list)
    for i in range(len(queries_list)):
        evaluation_result = (
            base_benchmark.results[i][0]
            if base_benchmark.results and i < len(base_benchmark.results) and base_benchmark.results[i]
            else {}
        )
        if not evaluation_result or evaluation_result.get("error"):
            evaluation_failures[i] = evaluation_result.get("error", "evaluation_failed") if isinstance(evaluation_result, dict) else "evaluation_failed"

    if any(err is not None for err in evaluation_failures):
        return evaluation_failures

    curated_data = []
    for i, q in enumerate(queries_list):
        plan = engine_plans[i]
        evaluation_stats = base_benchmark.results[i][0] if base_benchmark.results else {}
        entry = {
            "id": i,
            "query": q,
            "plan": plan,
            "complexity": 0,
            "row_count": 0,
            "evaluation_stats": evaluation_stats,
        }
        curated_data.append(entry)
    write_json(curated_path, curated_data)
    return None


def _run_optimization_sampling(
    base_sampling_data: Optional[List[Dict[str, Any]]],
    curated_path: Path,
    base_sampling_path: Path,
    final_sampling_path: Path,
    sampling_dir: Path,
    queries_list: List[str],
    dataset: str,
    model: str,
    strategy: str,
    n_steps: int,
    n_samples_per_step: int,
    max_sampling_workers: int,
    max_workers: int,
    optimization_metric: str,
    resume: bool,
    n_runs: int,
    sampling_kwargs: Dict[str, Any],
    evaluation_kwargs: Dict[str, Any],
    verbose: bool,
) -> Tuple[Dict[int, Any], bool, Dict[str, Any]]:
    """Prepare base sampling data and run the iterative optimization loop.

    Calls ``prepare_sampling`` (if needed) and ``run_sampling_steps``, then
    loads the final sampling result and builds lookup indices.

    Returns ``(sampling_by_id, id_indexed, query_to_entry)`` for downstream
    result processing.
    """
    from sampling.prepare_sampling import prepare_sampling
    from sampling.orchestrator import run_sampling_steps

    if base_sampling_data is None:
        if (not resume) or (not base_sampling_path.exists()):
            prepare_sampling(
                input_file=str(curated_path),
                output_file=str(base_sampling_path),
                model=model,
                verbose=False,
            )

    run_sampling_steps(
        sampling_dir=sampling_dir,
        base_sampling_path=base_sampling_path,
        final_sampling_path=final_sampling_path,
        dataset=dataset,
        model=model,
        strategy=strategy,
        n_steps=n_steps,
        n_samples_per_step=n_samples_per_step,
        max_sampling_workers=max_sampling_workers,
        max_workers=max_workers,
        optimization_metric=optimization_metric,
        resume=resume,
        sampling_kwargs=sampling_kwargs,
        evaluation_kwargs={"n_runs": n_runs, **evaluation_kwargs},
        verbose=verbose,
    )

    if not final_sampling_path.exists():
        raise RuntimeError("final_sampling.json missing after optimization steps")

    sampling_data = json.loads(final_sampling_path.read_text())
    sampling_by_id = {int(item["id"]): item for item in sampling_data if "id" in item}
    id_indexed = set(sampling_by_id.keys()) == set(range(len(queries_list)))
    if not id_indexed:
        query_to_entry: Dict[str, Any] = {}
        for item in sampling_data:
            q = item.get("query")
            if q is None or q in query_to_entry:
                query_to_entry = {}
                break
            query_to_entry[q] = item
    else:
        query_to_entry = {}

    return sampling_by_id, id_indexed, query_to_entry


def _process_sampling_results(
    queries_list: List[str],
    sampling_by_id: Dict[int, Any],
    id_indexed: bool,
    query_to_entry: Dict[str, Any],
    optimization_metric: str,
    top_k_patches: int,
) -> Tuple[List[PatchedPlan], List[Dict[str, Any]], List[float], Dict[str, int], int]:
    """Process sampling results: categorize outcomes, rank candidates, select top-k.

    For each query, categorizes sampled plan outcomes (success, error type, etc.),
    filters valid candidates, ranks them by *optimization_metric* (lower is better),
    selects the top-*top_k_patches*, and computes improvement ratios vs the base.

    Returns ``(optimization_outcome, metadata, improvement_values,
    outcome_counts, total_outcomes)``.
    """
    from sampling.sample_plans import get_upstream_patches

    optimization_outcome: List[PatchedPlan] = []
    metadata: List[Dict[str, Any]] = []
    improvement_values: List[float] = []
    outcome_counts = {
        "Invalid Patch": 0,
        "Plan Deserialization Error": 0,
        "Plan Execution Error": 0,
        "Execution Output Mismatch": 0,
        "Server-side Execution Error": 0,
        "LLM Failure": 0,
        "Empty Patch": 0,
        "Successful Execution": 0,
    }
    total_outcomes = 0

    for i, query in enumerate(queries_list):
        if id_indexed:
            if i not in sampling_by_id:
                raise RuntimeError(f"Missing sampling data for query id {i}")
            query_data = sampling_by_id[i]
        else:
            if query_to_entry and query in query_to_entry:
                query_data = query_to_entry[query]
            else:
                raise RuntimeError(f"Missing sampling data for query at index {i}")
        sampleid2sampleidx = {
            plan_info["sample_id"]: idx
            for idx, plan_info in enumerate(query_data["sampled_plans"])
        }

        base_evaluation = None
        for plan_info in query_data["sampled_plans"]:
            if plan_info["sample_id"] is None:
                base_evaluation = get_evaluation_stats(plan_info)
                break

        candidates = [
            plan_info
            for plan_info in query_data["sampled_plans"]
            if plan_info.get("sample_id") is not None
        ]

        for plan_info in candidates:
            total_outcomes += 1
            if plan_info.get("is_valid") and plan_info.get("sampled_patches") == []:
                outcome_counts["Empty Patch"] += 1
                continue
            if plan_info.get("is_valid"):
                outcome_counts["Successful Execution"] += 1
                continue
            msg = (plan_info.get("error_message") or "").lower()
            error_types = [
                ("failed to apply patches to plan:", "Invalid Patch"),
                ("failed to convert from succinct json: failed to deserialize plan from json:", "Plan Deserialization Error"),
                ("result set mismatch:", "Execution Output Mismatch"),
                ("expected results to have at least one row, instead got 0.", "Execution Output Mismatch"),
                ("execution error: error executing query:", "Plan Execution Error"),
            ]
            matched = False
            if any(tok in msg for tok in ["litellm", "openai", "connection error"]):
                outcome_counts["LLM Failure"] += 1
                matched = True
            for pattern, label in error_types:
                if pattern and pattern in msg:
                    outcome_counts[label] += 1
                    matched = True
                    break
            if not matched:
                outcome_counts["Server-side Execution Error"] += 1

        valid_candidates = [
            c for c in candidates
            if c.get("is_valid") and get_metric_value(get_evaluation_stats(c), optimization_metric) is not None
        ]
        valid_candidates.sort(key=lambda c: get_metric_value(get_evaluation_stats(c), optimization_metric))
        base_metric = get_metric_value(base_evaluation, optimization_metric)
        ranked_candidates: List[Dict[str, Any]] = [
            {
                "patch_ops": [],
                "evaluation_stats": base_evaluation or {},
                "metric_value": base_metric,
            }
        ]
        for plan_info in valid_candidates:
            eval_stats = get_evaluation_stats(plan_info) or {}
            ranked_candidates.append(
                {
                    "patch_ops": get_upstream_patches(plan_info, query_data, sampleid2sampleidx),
                    "evaluation_stats": eval_stats,
                    "metric_value": get_metric_value(eval_stats, optimization_metric),
                }
            )

        ranked_candidates.sort(key=lambda x: x["metric_value"] if x["metric_value"] is not None else float("inf"))
        chosen = ranked_candidates[:top_k_patches]
        patch_list: List[Optional[Patch]] = []
        patch_stats: List[Optional[Dict[str, Any]]] = []
        improvement_x: List[Optional[float]] = []

        for candidate in chosen:
            patch_ops = candidate["patch_ops"]
            patch_metric = candidate["metric_value"]
            patch_list.append(patch_ops)
            patch_stats.append(candidate["evaluation_stats"])
            if base_metric is not None and patch_metric is not None and patch_metric > 0:
                improvement_x.append(base_metric / patch_metric)
            else:
                improvement_x.append(None)

        while len(patch_list) < top_k_patches:
            patch_list.append(None)
            patch_stats.append(None)
            improvement_x.append(None)

        improvement_values.extend(
            [float(v) for v in improvement_x if isinstance(v, (int, float))]
        )

        optimization_outcome.append(
            PatchedPlan(
                base_plan=query_data["plan"],
                patch=patch_list,
            )
        )
        metadata.append(
            {
                "improvement_x": improvement_x,
                "benchmark_stats": {
                    "base": base_evaluation or {},
                    "patch": patch_stats,
                },
            }
        )

    return optimization_outcome, metadata, improvement_values, outcome_counts, total_outcomes


def optimize_queries(
    queries: Optional[Sequence[str]] = None,
    base_plans: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    run_dir: Optional[str] = None,
    dataset: str = "tpch",
    scale_factor: int = DEFAULT_SCALE_FACTOR,
    *,
    strategy: Literal["bol_evol", "pst_evol", "best_of"] = "bol_evol",
    n_steps: int = 4,
    n_samples_per_step: int = 5,
    top_k_patches: int = 1,
    n_runs: int = 5,
    model: str = "gpt-5",
    optimization_metric: str = "execution_time.min",
    max_sampling_workers: int = 50,
    max_workers: int = 20,
    max_eval_workers: int = 8,
    resume: bool = False,
    skip_validation: bool = False,
    verbose: bool = True,
    validate_kwargs: Optional[Dict[str, Any]] = None,
    sampling_kwargs: Optional[Dict[str, Any]] = None,
    evaluation_kwargs: Optional[Dict[str, Any]] = None,
    get_full_metrics: bool = False,
    exec_local: bool = False,
) -> OptimizationResult:
    """
    Optimize SQL queries using LLM-driven evolutionary sampling of execution plans.

    Validates queries, obtains base execution plans, then iteratively samples plan
    patches via an LLM, validates them for correctness (result-set comparison), and
    evaluates performance. Returns the top-k best patches per query ranked by
    optimization_metric. Writes result.json and algorithm artifacts to run_dir.

    Args:
        queries: SQL strings to optimize. Required for new runs; omit to resume.
        base_plans: Alternative base plans aligned to queries. Each element is None
            (use engine plan) or a plan dict (validated against the engine plan's
            result set before use). Requires queries.
        run_dir: Output directory. Defaults to a timestamped dir under outputs/.
        dataset: Dataset name (e.g., "tpch", "tpcds").
        scale_factor: Scale factor for the Modal runner.
        strategy: Sampling strategy ("bol_evol", "pst_evol", "best_of").
        n_steps: Number of optimization steps (sampling rounds).
        n_samples_per_step: LLM samples per step.
        top_k_patches: Number of best patches to keep per query.
        n_runs: Benchmark runs per candidate plan.
        model: LLM model name.
        optimization_metric: Metric for ranking patches (lower is better),
            e.g. "execution_time.min".
        max_sampling_workers: Max concurrent LLM sampling workers.
        max_workers: Max concurrent plan-level workers (validation/evaluation).
        max_eval_workers: Max concurrent evaluation runs per plan. Total
            concurrent Modal sandboxes is approximately
            ``max_workers * max_eval_workers``.
        resume: Resume from an existing run_dir instead of restarting.
        skip_validation: Skip query validation. When True, queries are not
            checked for empty results or non-determinism (prints a warning).
        verbose: Print progress and summaries.
        validate_kwargs: Extra kwargs forwarded to validation.
        sampling_kwargs: Extra kwargs forwarded to sampling.
        evaluation_kwargs: Extra kwargs forwarded to evaluation.
        get_full_metrics: Collect detailed execution metrics (bytes scanned,
            join stats, memory usage) in addition to execution_time.
        exec_local: Run validation and evaluation locally (via LocalRunner)
            instead of on Modal. Requires the ``local`` extra; missing data is
            generated under ``data/`` on first use.

    Returns:
        OptimizationResult with run_dir, summary (validation_stats, base_plan_sources,
        run config, improvement_x_stats, outcome_rates), queries, optimization_outcome
        (list of PatchedPlan), metadata (per-query benchmark stats), and
        validation_failures. optimization_outcome/metadata are None on early exit.

    Run-dir behavior:
        queries + run_dir: resume must be False; if run_dir exists, raises
            FileExistsError.
        queries only: creates a new timestamped run_dir.
        run_dir only: resume must be True; loads queries from result.json.
        neither: raises ValueError.
    """

    # --- A: Parameter validation ---
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if n_samples_per_step < 1:
        raise ValueError("n_samples_per_step must be >= 1")
    if top_k_patches < 1:
        raise ValueError("top_k_patches must be >= 1")
    if strategy == "best_of" and n_steps > 1:
        raise ValueError("best_of only supports n_steps <= 1")

    if isinstance(queries, str):
        raise ValueError("queries must be a sequence of strings, not a single string")

    if base_plans is not None:
        if queries is None:
            raise ValueError("base_plans requires queries to be provided")
        if len(base_plans) != len(queries):
            raise ValueError(
                f"base_plans length ({len(base_plans)}) must match queries length ({len(queries)})"
            )

    user_base_plans: Optional[List[Optional[Dict[str, Any]]]] = (
        list(base_plans) if base_plans is not None else None
    )

    validate_kwargs_local = dict(validate_kwargs or {})
    sampling_kwargs_local = dict(sampling_kwargs or {})
    evaluation_kwargs_local = dict(evaluation_kwargs or {})
    validate_kwargs_local.pop("dataset", None)
    evaluation_kwargs_local.pop("dataset", None)
    evaluation_kwargs_local.pop("n_runs", None)

    # Thread exec_local into validation and evaluation kwargs.
    if exec_local:
        validate_kwargs_local["exec_local"] = True
        evaluation_kwargs_local["exec_local"] = True

    if scale_factor is None:
        raise ValueError("scale_factor cannot be None")

    if exec_local:
        ensure_local_data(dataset, scale_factor)

    validate_scale = validate_kwargs_local.pop("scale_factor", None)
    evaluation_scale = evaluation_kwargs_local.pop("scale_factor", None)
    if validate_scale is not None and validate_scale != scale_factor:
        raise ValueError("scale_factor conflicts with validate_kwargs['scale_factor']")
    if evaluation_scale is not None and evaluation_scale != scale_factor:
        raise ValueError("scale_factor conflicts with evaluation_kwargs['scale_factor']")
    effective_scale_factor = scale_factor

    def _attach_runner_scale(kwargs_local: Dict[str, Any], kwargs_name: str) -> Dict[str, Any]:
        updated = dict(kwargs_local)
        runner_kwargs = updated.pop("runner_kwargs", {}) or {}
        if not isinstance(runner_kwargs, dict):
            raise ValueError(f"{kwargs_name}['runner_kwargs'] must be a dict")
        runner_kwargs = dict(runner_kwargs)
        if (
            "scale_factor" in runner_kwargs
            and runner_kwargs["scale_factor"] != effective_scale_factor
        ):
            raise ValueError(f"scale_factor conflicts with {kwargs_name}['runner_kwargs']['scale_factor']")
        runner_kwargs["scale_factor"] = effective_scale_factor
        updated["runner_kwargs"] = runner_kwargs
        return updated

    validate_kwargs_local = _attach_runner_scale(validate_kwargs_local, "validate_kwargs")
    evaluation_kwargs_local = _attach_runner_scale(evaluation_kwargs_local, "evaluation_kwargs")

    if get_full_metrics:
        evaluation_kwargs_local.setdefault("sandbox_placeholders", {})
        evaluation_kwargs_local["sandbox_placeholders"]["FULL_METRICS"] = True

    evaluation_kwargs_local["max_eval_workers"] = max_eval_workers

    # --- B: Run directory setup ---
    if run_dir is None:
        run_dir_path = None
    else:
        run_dir_path = resolve_run_dir(run_dir, "optimize_queries")

    queries_list: Optional[List[str]] = list(queries) if queries is not None else None

    if queries_list is not None and run_dir_path is not None:
        if resume:
            raise ValueError("queries + run_dir requires resume=False")
        if run_dir_path.exists():
            raise FileExistsError(
                f"Run dir already exists at {run_dir_path}. "
                "Set resume=True to continue from existing state, "
                "or delete the directory to start fresh."
            )
        run_dir_path.mkdir(parents=True, exist_ok=True)
    elif queries_list is not None and run_dir_path is None:
        run_dir_path = resolve_run_dir(None, "optimize_queries")
        run_dir_path.mkdir(parents=True, exist_ok=True)
    elif queries_list is None and run_dir_path is not None:
        if not resume:
            raise ValueError("queries missing: resume must be True")
        if not run_dir_path.exists():
            raise ValueError("run_dir does not exist for resume=True")
    else:
        raise ValueError("queries and run_dir cannot both be None")

    result_path = run_dir_path / "result.json"
    sampling_dir = run_dir_path / "sampling"
    sampling_dir.mkdir(parents=True, exist_ok=True)
    curated_path = sampling_dir / "curated_queries.json"
    base_sampling_path = sampling_dir / "base_sampling.json"
    final_sampling_path = sampling_dir / "final_sampling.json"

    existing_result = None
    if resume and result_path.exists():
        existing_result = json.loads(result_path.read_text())

    if queries_list is None:
        if existing_result and existing_result.get("queries"):
            queries_list = existing_result.get("queries")
        else:
            raise ValueError("queries missing and no existing result.json to resume from")
    log_line(verbose, f"Starting optimization of {len(queries_list)} {'query' if len(queries_list) == 1 else 'queries'} on {dataset} (scale_factor={effective_scale_factor})")
    log_line(verbose, f"  Strategy: {strategy}, {n_steps} steps x {n_samples_per_step} samples/step, model: {model}")
    log_line(verbose, f"  Run directory: {run_dir_path}")

    # --- C: Config & resume validation ---
    config_current = {
        "strategy": strategy,
        "n_steps": n_steps,
        "n_samples_per_step": n_samples_per_step,
        "top_k_patches": top_k_patches,
        "model": model,
        "optimization_metric": optimization_metric,
        "n_runs": n_runs,
        "max_sampling_workers": max_sampling_workers,
        "max_workers": max_workers,
        "dataset": dataset,
        "scale_factor": effective_scale_factor,
    }

    if resume and existing_result:
        existing_run = existing_result.get("summary", {}).get("run", {})
        allowed_diff = {"n_steps", "top_k_patches"}
        if strategy == "best_of":
            allowed_diff.add("n_samples_per_step")
        for key, value in config_current.items():
            if key in allowed_diff:
                continue
            if key in existing_run and existing_run[key] != value:
                raise ValueError(f"Config mismatch for optimize_queries and resume=True: {key}")

    base_sampling_data: Optional[List[Dict[str, Any]]] = None
    if resume and base_sampling_path.exists():
        try:
            base_sampling_data = json.loads(base_sampling_path.read_text())
        except Exception:
            base_sampling_data = None

    early_exit_args = dict(
        config_current=config_current,
        result_path=result_path,
        queries_list=queries_list,
        run_dir_path=run_dir_path,
        verbose=verbose,
    )

    # --- D: Validate and plan queries ---
    n_phases = 3 if n_steps > 0 else 2
    log_line(verbose, f"\n[Phase 1/{n_phases}] Validating queries...")
    engine_plans, validation_failures = _validate_and_plan_queries(
        queries_list=queries_list,
        dataset=dataset,
        scale_factor=effective_scale_factor,
        max_workers=max_workers,
        validate_kwargs=validate_kwargs_local,
        skip_validation=skip_validation,
        base_sampling_data=base_sampling_data,
        run_dir_path=run_dir_path,
        verbose=verbose,
    )
    validation_stats = build_validation_stats(validation_failures)
    log_line(verbose, f"  Validation complete: {validation_stats['n_valid']}/{validation_stats['n_queries']} queries valid")
    if validation_stats["n_valid"] < validation_stats["n_queries"]:
        return _build_early_exit_result(
            "Warning: some queries failed validation; optimization will not proceed.",
            validation_failures,
            **early_exit_args,
        )

    # --- E: Integrate user base plans ---
    engine_plans, base_plan_sources, user_plan_failures = _integrate_user_base_plans(
        user_base_plans=user_base_plans,
        engine_plans=engine_plans,
        queries_list=queries_list,
        dataset=dataset,
        validate_kwargs=validate_kwargs_local,
        max_workers=max_workers,
        skip_validation=skip_validation,
        verbose=verbose,
    )
    if user_plan_failures and any(err is not None for err in user_plan_failures):
        return _build_early_exit_result(
            "Warning: some user-provided base plans failed validation; optimization will not proceed.",
            user_plan_failures,
            **early_exit_args,
        )

    # --- F: Evaluate base plans ---
    log_line(verbose, f"\n[Phase 2/{n_phases}] Benchmarking base plans...")
    evaluation_failures = _evaluate_and_curate_base_plans(
        queries_list=queries_list,
        engine_plans=engine_plans,
        curated_path=curated_path,
        dataset=dataset,
        scale_factor=effective_scale_factor,
        n_runs=n_runs,
        max_workers=max_workers,
        evaluation_kwargs=evaluation_kwargs_local,
        base_sampling_data=base_sampling_data,
        resume=resume,
        verbose=verbose,
    )
    if evaluation_failures and any(err is not None for err in evaluation_failures):
        return _build_early_exit_result(
            "Warning: some base plans failed evaluation; optimization will not proceed.",
            evaluation_failures,
            **early_exit_args,
        )

    # --- G: Run sampling ---
    if n_steps > 0:
        log_line(verbose, f"\n[Phase 3/{n_phases}] Running optimization ({n_steps} steps)...")
    sampling_by_id, id_indexed, query_to_entry = _run_optimization_sampling(
        base_sampling_data=base_sampling_data,
        curated_path=curated_path,
        base_sampling_path=base_sampling_path,
        final_sampling_path=final_sampling_path,
        sampling_dir=sampling_dir,
        queries_list=queries_list,
        dataset=dataset,
        model=model,
        strategy=strategy,
        n_steps=n_steps,
        n_samples_per_step=n_samples_per_step,
        max_sampling_workers=max_sampling_workers,
        max_workers=max_workers,
        optimization_metric=optimization_metric,
        resume=resume,
        n_runs=n_runs,
        sampling_kwargs=sampling_kwargs_local,
        evaluation_kwargs=evaluation_kwargs_local,
        verbose=verbose,
    )

    # --- H: Process results ---
    optimization_outcome, metadata, improvement_values, outcome_counts, total_outcomes = (
        _process_sampling_results(
            queries_list=queries_list,
            sampling_by_id=sampling_by_id,
            id_indexed=id_indexed,
            query_to_entry=query_to_entry,
            optimization_metric=optimization_metric,
            top_k_patches=top_k_patches,
        )
    )

    # --- I: Summary & return ---
    summary = {
        "validation_stats": validation_stats,
        "base_plan_sources": base_plan_sources,
        "run": {
            **config_current,
            "n_sampled_plans": total_outcomes,
        },
        "improvement_x_stats": compute_metric_stats(improvement_values) if improvement_values else {},
        "outcome_rates": {
            k: (outcome_counts[k] / total_outcomes) if total_outcomes else 0.0
            for k in outcome_counts
        },
    }

    write_json(
        result_path,
        {
            "queries": queries_list,
            "optimization_outcome": [
                {
                    "base": plan.base_plan,
                    "patch": plan.patch,
                }
                for plan in optimization_outcome
            ],
            "summary": summary,
            "metadata": metadata,
        },
    )

    # --- J: Completion message ---
    if verbose:
        log_line(verbose, "\nOptimization complete.")
        log_line(verbose, f"  Results saved to: {result_path}")
        if improvement_values:
            imp_stats = summary["improvement_x_stats"]
            log_line(verbose, f"  improvement_x per query:\n    {format_metric_stats(imp_stats)}")
        else:
            log_line(verbose, "  No valid optimizations found.")
        n_sampled = summary["run"]["n_sampled_plans"]
        success_rate = summary["outcome_rates"].get("Successful Execution", 0)
        log_line(verbose, f"  Sampled plans: {n_sampled}, success rate: {success_rate:.0%}")

    return OptimizationResult(
        run_dir=str(run_dir_path),
        summary=summary,
        queries=list(queries_list),
        optimization_outcome=optimization_outcome,
        metadata=metadata,
        validation_failures=validation_failures,
    )


def benchmark_plans(
    plans: Sequence[Union[PatchedPlan, Dict[str, Any]]],
    *,
    dataset: str,
    scale_factor: int = DEFAULT_SCALE_FACTOR,
    n_runs: int = 3,
    output_file: Optional[str] = None,
    max_workers: int = 1,
    max_eval_workers: int = 8,
    get_full_metrics: bool = False,
    exec_local: bool = False,
    verbose: bool = True,
    **kwargs,
) -> BenchmarkResult:
    """
    Benchmark a list of plans against a dataset.

    Expands each PatchedPlan's patches, evaluates them concurrently, and
    returns per-plan, per-patch measurements.

    Args:
        plans: PatchedPlan instances or dicts with ``base_plan`` and
            ``patch`` keys. Dicts are normalized to PatchedPlan internally.
        dataset: Dataset name (e.g., "tpch", "tpcds").
        scale_factor: Scale factor for the Modal runner.
        n_runs: Number of evaluation runs per plan.
        output_file: Optional path to write results JSON.
        max_workers: Max concurrent plan-level workers.
        max_eval_workers: Max concurrent evaluation runs per plan. Total
            concurrent Modal sandboxes is approximately
            ``max_workers * max_eval_workers``.
        get_full_metrics: Collect detailed execution metrics (bytes scanned,
            join stats, memory usage) in addition to execution_time.
        exec_local: Evaluate locally (via LocalRunner) instead of on Modal.
            Requires the ``local`` extra; missing data is generated under
            ``data/`` on first use.
        verbose: Print progress messages.
        **kwargs: Passed through to the evaluation backend. Supports
            ``runner_kwargs`` (dict for ModalRunner constructor),
            ``cpu``, ``memory``, ``timeout``, ``sandbox_timeout``.

    Returns:
        BenchmarkResult with results (list-of-lists aligned to plans and
        their patches) and optional output_file path.
    """
    from sampling.utils import apply_patches_to_plan
    from modal_controller.utils import evaluate_plan_n_runs

    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")

    # Ensure runner_kwargs has scale_factor
    if "runner_kwargs" not in kwargs or kwargs["runner_kwargs"] is None:
        kwargs["runner_kwargs"] = {}
    kwargs["runner_kwargs"] = dict(kwargs["runner_kwargs"])
    if (
        "scale_factor" in kwargs["runner_kwargs"]
        and kwargs["runner_kwargs"]["scale_factor"] != scale_factor
    ):
        raise ValueError("scale_factor conflicts with runner_kwargs['scale_factor']")
    kwargs["runner_kwargs"]["scale_factor"] = scale_factor

    if get_full_metrics:
        kwargs.setdefault("sandbox_placeholders", {})
        kwargs["sandbox_placeholders"]["FULL_METRICS"] = True

    # Allow max_eval_workers from kwargs (for evaluation_kwargs pass-through)
    max_eval_workers = kwargs.pop("max_eval_workers", max_eval_workers)

    # Normalize dicts to PatchedPlan
    normalized_plans: List[PatchedPlan] = []
    for p in plans:
        if isinstance(p, PatchedPlan):
            normalized_plans.append(p)
        elif isinstance(p, dict):
            normalized_plans.append(PatchedPlan(
                base_plan=p["base_plan"],
                patch=p.get("patch"),
            ))
        else:
            raise TypeError(f"Expected PatchedPlan or dict, got {type(p).__name__}")

    if exec_local:
        ensure_local_data(dataset, scale_factor)
    data_folder = data_folder_for_dataset(dataset, exec_local=exec_local, scale_factor=scale_factor)

    # Inject exec_local into kwargs so evaluate_plan_n_runs receives it.
    kwargs["exec_local"] = exec_local

    total_evals = sum(len(p.patch) for p in normalized_plans)
    log_line(verbose, f"Benchmarking {len(normalized_plans)} plans ({total_evals} evaluations, {n_runs} runs each, scale_factor={scale_factor})")

    results: List[List[Dict[str, Any]]] = [
        [] for _ in range(len(normalized_plans))
    ]

    def worker(plan_idx: int, patch_idx: int, base_plan: Dict[str, Any], patch_ops: Optional[Patch]):
        try:
            if base_plan is None:
                return plan_idx, patch_idx, {"error": "base_plan is None"}
            if patch_ops is None:
                return plan_idx, patch_idx, {"error": "patch_missing", "skipped": True}
            if patch_ops:
                plan = apply_patches_to_plan(base_plan, patch_ops)
            else:
                plan = base_plan
            plan_json = plan_to_json(plan)
            del plan
            measurement = evaluate_plan_n_runs(
                operation=Operation.EVALUATE,
                plan_json=plan_json,
                data_folder=data_folder,
                n_runs=n_runs,
                max_eval_workers=max_eval_workers,
                **kwargs,
            )
            return plan_idx, patch_idx, measurement
        except Exception as exc:
            return plan_idx, patch_idx, {"error": str(exc)}

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for plan_idx, plan in enumerate(normalized_plans):
            patch_list = plan.patch
            for patch_idx, patch_ops in enumerate(patch_list):
                futures.append(
                    executor.submit(worker, plan_idx, patch_idx, plan.base_plan, patch_ops)
                )
        for future in tqdm(as_completed(futures), total=len(futures), desc="Benchmarking plans", disable=not verbose):
            plan_idx, patch_idx, measurement = future.result()
            while len(results[plan_idx]) <= patch_idx:
                results[plan_idx].append({})
            results[plan_idx][patch_idx] = measurement

    n_errors = sum(1 for plan_results in results for r in plan_results if r.get("error"))
    log_line(verbose, f"Benchmarking complete: {total_evals} evaluations across {len(normalized_plans)} plans ({n_errors} errors)")

    if output_file:
        write_json(Path(output_file), results)

    return BenchmarkResult(results=results, output_file=output_file)


def get_engine_plans(
    queries: Sequence[str],
    *,
    dataset: str,
    scale_factor: int = DEFAULT_SCALE_FACTOR,
    max_workers: int = 10,
    exec_local: bool = False,
    verbose: bool = True,
    **kwargs,
) -> PlanningResult:
    """
    Obtain engine execution plans for a list of SQL queries.

    Plans each query concurrently via the execution engine at the given
    scale factor. Useful for inspecting plans, modifying them, and then
    benchmarking with ``benchmark_plans``.

    Args:
        queries: SQL query strings to plan.
        dataset: Dataset name (e.g., "tpch", "tpcds").
        scale_factor: Scale factor for the Modal runner.
        max_workers: Max concurrent planning workers.
        exec_local: Plan locally (via LocalRunner) instead of on Modal.
            Requires the ``local`` extra; missing data is generated under
            ``data/`` on first use.
        verbose: Print progress messages.
        **kwargs: Passed through to the planning backend. Supports
            ``runner_kwargs`` (dict for ModalRunner constructor),
            ``cpu``, ``memory``, ``timeout``, ``sandbox_timeout``.

    Returns:
        PlanningResult with per-query plans and errors aligned to inputs.
    """
    if isinstance(queries, str):
        raise ValueError("queries must be a sequence of strings, not a single string")
    queries_list: List[str] = list(queries)
    if not queries_list:
        raise ValueError("queries must not be empty")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    # Ensure runner_kwargs has scale_factor
    if "runner_kwargs" not in kwargs or kwargs["runner_kwargs"] is None:
        kwargs["runner_kwargs"] = {}
    kwargs["runner_kwargs"] = dict(kwargs["runner_kwargs"])
    if (
        "scale_factor" in kwargs["runner_kwargs"]
        and kwargs["runner_kwargs"]["scale_factor"] != scale_factor
    ):
        raise ValueError("scale_factor conflicts with runner_kwargs['scale_factor']")
    kwargs["runner_kwargs"]["scale_factor"] = scale_factor

    if exec_local:
        ensure_local_data(dataset, scale_factor)
    kwargs["exec_local"] = exec_local  # thread to the planning backend
    data_folder = data_folder_for_dataset(dataset, exec_local=exec_local, scale_factor=scale_factor)
    plans: List[Optional[Dict[str, Any]]] = [None] * len(queries_list)
    errors: List[Optional[str]] = [None] * len(queries_list)

    log_line(verbose, f"Fetching execution plans for {len(queries_list)} queries on {dataset} (scale_factor={scale_factor})")

    def _plan_worker(idx: int, query: str):
        result = submit_run_operation(
            operation=Operation.PLAN,
            input_str=query,
            data_folder=data_folder,
            **kwargs,
        )
        if not result:
            return idx, None, "plan_error"
        if result.get("error"):
            return idx, None, f"plan_error: {result['error']}"
        plan = result.get("plan")
        del result
        if not plan:
            return idx, None, "no_plan"
        if isinstance(plan, str):
            plan = json.loads(plan)
        return idx, plan, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_plan_worker, i, q): i
            for i, q in enumerate(queries_list)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Planning queries", disable=not verbose):
            idx, plan, err = future.result()
            plans[idx] = plan
            errors[idx] = err

    n_planned = sum(1 for p in plans if p is not None)
    n_failed = len(queries_list) - n_planned
    if n_failed > 0:
        log_line(verbose, f"Planning complete: {n_planned}/{len(queries_list)} queries planned ({n_failed} failed)")
    else:
        log_line(verbose, f"Planning complete: all {n_planned} queries planned successfully")

    return PlanningResult(plans=plans, errors=errors)


def benchmark_queries(
    queries: Sequence[str],
    *,
    dataset: str,
    scale_factor: int = DEFAULT_SCALE_FACTOR,
    n_runs: int = 3,
    output_file: Optional[str] = None,
    max_workers: int = 1,
    max_eval_workers: int = 8,
    get_full_metrics: bool = False,
    exec_local: bool = False,
    verbose: bool = True,
    **kwargs,
) -> BenchmarkResult:
    """
    Benchmark raw SQL queries against a dataset.

    Plans each query via the execution engine, then evaluates the resulting
    plans concurrently. Planning failures are recorded as error dicts in the
    output rather than raising exceptions.

    Args:
        queries: SQL query strings to benchmark.
        dataset: Dataset name (e.g., "tpch", "tpcds").
        scale_factor: Scale factor for the Modal runner used during
            planning and evaluation.
        n_runs: Number of evaluation runs per plan.
        output_file: Optional path to write results JSON.
        max_workers: Max concurrent plan-level workers.
        max_eval_workers: Max concurrent evaluation runs per plan. Total
            concurrent Modal sandboxes is approximately
            ``max_workers * max_eval_workers``.
        get_full_metrics: Collect detailed execution metrics (bytes scanned,
            join stats, memory usage) in addition to execution_time.
        exec_local: Plan and evaluate locally (via LocalRunner) instead of on
            Modal. Requires the ``local`` extra; missing data is generated under
            ``data/`` on first use.
        verbose: Print progress messages.
        **kwargs: Passed through to planning and evaluation backends.
            Supports ``runner_kwargs`` (dict for ModalRunner constructor),
            ``cpu``, ``memory``, ``timeout``, ``sandbox_timeout``.

    Returns:
        BenchmarkResult with results aligned to input queries. Each
        ``results[i]`` is a single-element list since no patches are
        involved. Planning failures appear as error dicts.
    """
    if isinstance(queries, str):
        raise ValueError("queries must be a sequence of strings, not a single string")
    queries_list: List[str] = list(queries)
    if not queries_list:
        raise ValueError("queries must not be empty")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")

    # Ensure runner_kwargs has scale_factor
    if "runner_kwargs" not in kwargs or kwargs["runner_kwargs"] is None:
        kwargs["runner_kwargs"] = {}
    kwargs["runner_kwargs"] = dict(kwargs["runner_kwargs"])
    if (
        "scale_factor" in kwargs["runner_kwargs"]
        and kwargs["runner_kwargs"]["scale_factor"] != scale_factor
    ):
        raise ValueError("scale_factor conflicts with runner_kwargs['scale_factor']")
    kwargs["runner_kwargs"]["scale_factor"] = scale_factor

    log_line(verbose, f"Benchmarking {len(queries_list)} queries on {dataset} (scale_factor={scale_factor}, {n_runs} runs each)")

    # Phase 1: Plan queries
    planning_result = get_engine_plans(
        queries_list,
        dataset=dataset,
        scale_factor=scale_factor,
        max_workers=max_workers,
        exec_local=exec_local,
        verbose=verbose,
        **kwargs,
    )

    # Phase 2: Separate successes from failures, delegate to benchmark_plans
    successful_indices: List[int] = []
    plan_entries: List[PatchedPlan] = []
    for i in range(len(queries_list)):
        if planning_result.plans[i] is not None and planning_result.errors[i] is None:
            successful_indices.append(i)
            plan_entries.append(PatchedPlan(base_plan=planning_result.plans[i], patch=[[]]))

    results: List[List[Dict[str, Any]]] = [[] for _ in range(len(queries_list))]
    for i in range(len(queries_list)):
        if planning_result.errors[i] is not None:
            results[i] = [{"error": f"planning_failed: {planning_result.errors[i]}"}]

    if plan_entries:
        n_failed = len(queries_list) - len(plan_entries)
        if n_failed > 0:
            log_line(verbose, f"Planning failed for {n_failed} queries; benchmarking {len(plan_entries)} remaining...")
        benchmark_result = benchmark_plans(
            plan_entries,
            dataset=dataset,
            scale_factor=scale_factor,
            n_runs=n_runs,
            output_file=None,
            max_workers=max_workers,
            max_eval_workers=max_eval_workers,
            get_full_metrics=get_full_metrics,
            exec_local=exec_local,
            verbose=verbose,
            **kwargs,
        )
        for delegate_idx, original_idx in enumerate(successful_indices):
            results[original_idx] = benchmark_result.results[delegate_idx]

    n_errors = sum(1 for r in results if r and r[0].get("error"))
    n_ok = len(queries_list) - n_errors
    log_line(verbose, f"Benchmarking complete: {n_ok}/{len(queries_list)} queries benchmarked successfully")

    if output_file:
        write_json(Path(output_file), results)

    return BenchmarkResult(results=results, output_file=output_file)


def scale_optimizations(
    queries: Sequence[str],
    plans: Sequence[PatchedPlan],
    *,
    source_scale_factor: int,
    target_scale_factor: int,
    dataset: str = "tpch",
    run_dir: Optional[str] = None,
    skip_validation: bool = False,
    max_workers: int = 20,
    modal_resources: Optional[Dict[str, Any]] = None,
    validate_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> ScaleResult:
    """
    Scale optimized plans from a smaller to a larger dataset scale factor.

    For each query, plans it at target_scale_factor to obtain the SF2 baseline,
    then transfers both the SF1 base plan and each optimization patch using
    scan-signature matching and ID remapping. Computes clean transferred patches
    as the diff between the transferred base and transferred optimization.
    Optionally validates transferred plans for correctness via result-set
    comparison against the SF2 engine plan. Writes results to run_dir.

    Args:
        queries: SQL query strings (aligned to plans).
        plans: PatchedPlan instances from a previous optimize_queries run at
            source_scale_factor. Each must have base_plan with ``structure`` and
            ``succinct_table_info`` keys.
        source_scale_factor: Scale factor the input plans were optimized at.
        target_scale_factor: Target scale factor to transfer optimizations to.
        dataset: Dataset name (e.g., "tpch", "tpcds").
        run_dir: Output directory. Defaults to a timestamped dir under outputs/.
        skip_validation: Skip result-set validation of transferred plans.
            When False, each transferred plan is executed multiple times and
            compared against the target-scale engine plan for correctness, and
            across runs for determinism. Empty results are accepted (a query
            may legitimately return no rows at a different scale), but
            mismatched or non-deterministic results cause the transfer to be
            marked as failed.
        max_workers: Max concurrent workers for planning, transfer,
            and validation.
        modal_resources: Common Modal resource overrides. Supported keys: ``cpu``,
            ``memory``, ``timeout``, ``sandbox_timeout``. These are forwarded to
            ``ModalRunner.run_operation`` alongside the scale factor.
        validate_kwargs: Extra kwargs forwarded to validation/planning backends.
        verbose: Print progress.

    Returns:
        ScaleResult with run_dir, summary, queries, scaled_plans (PatchedPlan
        list at target_scale_factor), metadata, and transfer_failures.
    """
    # ── Input validation ──────────────────────────────────────────────
    if isinstance(queries, str):
        raise ValueError("queries must be a sequence of strings, not a single string")
    queries_list: List[str] = list(queries)
    plans_list: List[PatchedPlan] = list(plans)
    if len(queries_list) != len(plans_list):
        raise ValueError(
            f"queries length ({len(queries_list)}) must match plans length ({len(plans_list)})"
        )
    if not queries_list:
        raise ValueError("queries must not be empty")
    if source_scale_factor < 1:
        raise ValueError("source_scale_factor must be >= 1")
    if target_scale_factor < 1:
        raise ValueError("target_scale_factor must be >= 1")
    if target_scale_factor <= source_scale_factor:
        raise ValueError("target_scale_factor must be > source_scale_factor")
    for i, plan in enumerate(plans_list):
        bp = plan.base_plan
        if not isinstance(bp, dict) or "structure" not in bp or "succinct_table_info" not in bp:
            raise ValueError(
                f"plans[{i}].base_plan must be a dict with 'structure' and 'succinct_table_info' keys"
            )

    # ── Runner kwargs assembly ────────────────────────────────────────
    validate_kwargs_local = dict(validate_kwargs or {})
    modal_resources_local = dict(modal_resources or {})
    for key in ("cpu", "memory", "timeout", "sandbox_timeout"):
        if key in modal_resources_local and key not in validate_kwargs_local:
            validate_kwargs_local[key] = modal_resources_local[key]

    rk = validate_kwargs_local.pop("runner_kwargs", {}) or {}
    rk = dict(rk)
    rk["scale_factor"] = target_scale_factor
    validate_kwargs_local["runner_kwargs"] = rk

    # ── Run directory setup ───────────────────────────────────────────
    if run_dir is not None:
        run_dir_path = resolve_run_dir(run_dir, "scale_optimizations")
        if run_dir_path.exists():
            raise FileExistsError(
                f"Run dir already exists at {run_dir_path}. "
                "Delete the directory to start fresh, or use a different run_dir."
            )
    else:
        run_dir_path = resolve_run_dir(None, "scale_optimizations")
    run_dir_path.mkdir(parents=True, exist_ok=True)
    result_path = run_dir_path / "result.json"

    log_line(verbose, f"Scaling optimizations for {len(queries_list)} queries from scale_factor={source_scale_factor} to scale_factor={target_scale_factor} ({dataset})")
    log_line(verbose, f"  Run directory: {run_dir_path}")

    data_folder = data_folder_for_dataset(dataset)

    # ── Phase 1: Plan queries at SF_large ─────────────────────────────
    log_line(verbose, "Phase 1: Planning queries at target_scale_factor...")
    sf2_planning_result = get_engine_plans(
        queries_list,
        dataset=dataset,
        scale_factor=target_scale_factor,
        max_workers=max_workers,
        verbose=False,
        **validate_kwargs_local,
    )
    sf2_base_plans = sf2_planning_result.plans
    sf2_planning_errors = sf2_planning_result.errors

    n_planned = sum(1 for p in sf2_base_plans if p is not None)
    log_line(verbose, f"  {n_planned}/{len(queries_list)} queries planned at scale_factor={target_scale_factor}")

    # ── Phase 2: Transfer + validate + compute patches ────────────────
    log_line(verbose, "Phase 2: Transferring and validating plans...")

    # Outcome tracking (denominator = total individual patches)
    outcome_counts = {
        "Successful Transfer": 0,
        "Planning Failed": 0,
        "Base Transfer Failed": 0,
        "Base Validation Failed": 0,
        "Optimization Transfer Failed": 0,
        "Optimization Validation Failed": 0,
        "Skipped (None Patch)": 0,
    }
    total_patches = 0

    # Per-query results
    scaled_plans: List[PatchedPlan] = []
    metadata: List[Dict[str, Any]] = []
    transfer_failures: List[Optional[str]] = [None] * len(queries_list)

    def _process_query(query_idx: int):
        """Transfer + validate all patches for a single query. Returns
        (query_idx, PatchedPlan, metadata_dict, transfer_failure, patch_outcomes)
        where patch_outcomes is a list of outcome category strings."""
        query = queries_list[query_idx]
        plan = plans_list[query_idx]
        sf2_plan = sf2_base_plans[query_idx]

        patch_list = plan.patch
        n_patches = len(patch_list)
        patch_outcomes: List[str] = []

        # SF2 planning failed → all patches fail
        if sf2_plan is None:
            err_msg = sf2_planning_errors[query_idx] or "SF2 planning failed"
            patch_outcomes = ["Planning Failed"] * n_patches
            per_patch = [{"status": "planning_failed", "error": err_msg} for _ in patch_list]
            meta = {
                "query_status": "sf2_planning_failed",
                "error_message": err_msg,
                "per_patch": per_patch,
            }
            out_plan = PatchedPlan(base_plan=plan.base_plan, patch=[None])
            return query_idx, out_plan, meta, err_msg, patch_outcomes

        sf1_base = plan.base_plan

        # Step 1: Transfer SF1 base (with empty patch)
        try:
            transferred_base_structure = transfer_plan(
                query,
                sf1_base["structure"],
                sf1_base["succinct_table_info"],
                sf2_plan["structure"],
                sf2_plan["succinct_table_info"],
                [],
            )
        except Exception as exc:
            err_msg = f"Base transfer failed: {exc}"
            patch_outcomes = ["Base Transfer Failed"] * n_patches
            per_patch = [{"status": "base_transfer_failed", "error": err_msg} for _ in patch_list]
            meta = {
                "query_status": "base_transfer_failed",
                "error_message": err_msg,
                "per_patch": per_patch,
            }
            out_plan = PatchedPlan(base_plan=plan.base_plan, patch=[None])
            return query_idx, out_plan, meta, err_msg, patch_outcomes

        # Step 2: Validate transferred base (if not skip_validation)
        if not skip_validation:
            transferred_base_full = copy.deepcopy(sf2_plan)
            transferred_base_full["structure"] = transferred_base_structure
            base_val_err = validate_plan_result_set(
                plan_to_json(transferred_base_full),
                plan_to_json(sf2_plan),
                data_folder,
                n_retry=5,
                **validate_kwargs_local,
            )
            if base_val_err is not None:
                err_msg = f"Base validation failed: {base_val_err}"
                patch_outcomes = ["Base Validation Failed"] * n_patches
                per_patch = [{"status": "base_validation_failed", "error": err_msg} for _ in patch_list]
                meta = {
                    "query_status": "base_validation_failed",
                    "error_message": err_msg,
                    "per_patch": per_patch,
                }
                out_plan = PatchedPlan(base_plan=plan.base_plan, patch=[None])
                return query_idx, out_plan, meta, err_msg, patch_outcomes

        # Build output base_plan: SF2 plan with transferred base structure
        output_base = copy.deepcopy(sf2_plan)
        output_base["structure"] = transferred_base_structure

        # Step 3: Transfer each patch
        transferred_normalized: List[Optional[Patch]] = []
        per_patch: List[Dict[str, Any]] = []

        for patch_ops in patch_list:
            if patch_ops is None:
                transferred_normalized.append(None)
                per_patch.append({"status": "skipped_none", "error": None})
                patch_outcomes.append("Skipped (None Patch)")
                continue

            # Transfer the optimization (including empty patches)
            try:
                transferred_opt_structure = transfer_plan(
                    query,
                    sf1_base["structure"],
                    sf1_base["succinct_table_info"],
                    sf2_plan["structure"],
                    sf2_plan["succinct_table_info"],
                    patch_ops,
                )
            except Exception as exc:
                transferred_normalized.append(None)
                per_patch.append({"status": "transfer_failed", "error": str(exc)})
                patch_outcomes.append("Optimization Transfer Failed")
                continue

            # Compute transferred patch as diff between transferred base and transferred optimization
            transferred_patch_ops = jsonpatch.make_patch(
                transferred_base_structure, transferred_opt_structure
            ).patch

            # Validate transferred optimization (if not skip_validation)
            if not skip_validation:
                transferred_opt_full = copy.deepcopy(sf2_plan)
                transferred_opt_full["structure"] = transferred_opt_structure
                opt_val_err = validate_plan_result_set(
                    plan_to_json(transferred_opt_full),
                    plan_to_json(sf2_plan),
                    data_folder,
                    n_retry=5,
                    **validate_kwargs_local,
                )
                if opt_val_err is not None:
                    transferred_normalized.append(None)
                    per_patch.append({"status": "validation_failed", "error": opt_val_err})
                    patch_outcomes.append("Optimization Validation Failed")
                    continue

            transferred_normalized.append(transferred_patch_ops)
            per_patch.append({"status": "transferred", "error": None})
            patch_outcomes.append("Successful Transfer")

        # Determine query-level failure
        first_patch_err = next(
            (p["error"] for p in per_patch if p["error"] is not None), None
        )
        meta = {
            "query_status": "ok",
            "error_message": None,
            "per_patch": per_patch,
        }
        out_plan = PatchedPlan(base_plan=output_base, patch=transferred_normalized)
        return query_idx, out_plan, meta, first_patch_err, patch_outcomes

    # Execute Phase 2 concurrently per query
    query_results: Dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_process_query, i): i
            for i in range(len(queries_list))
        }
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Scaling optimizations", disable=not verbose):
            query_results[future_to_idx[future]] = future.result()

    # Assemble results in order
    for i in range(len(queries_list)):
        query_idx, out_plan, meta, failure, patch_outcomes = query_results[i]
        scaled_plans.append(out_plan)
        metadata.append(meta)
        transfer_failures[i] = failure
        for outcome in patch_outcomes:
            outcome_counts[outcome] += 1
            total_patches += 1

    n_successful = sum(1 for f in transfer_failures if f is None)
    log_line(verbose, f"Phase 2 complete: {n_successful}/{len(queries_list)} queries fully transferred")

    # ── Summary ───────────────────────────────────────────────────────
    summary = {
        "config": {
            "source_scale_factor": source_scale_factor,
            "target_scale_factor": target_scale_factor,
            "dataset": dataset,
            "skip_validation": skip_validation,
        },
        "outcome_rates": {
            k: (outcome_counts[k] / total_patches) if total_patches else 0.0
            for k in outcome_counts
        },
    }

    # ── Write results ─────────────────────────────────────────────────
    write_json(
        result_path,
        {
            "queries": queries_list,
            "scaled_plans": [
                {"base_plan": p.base_plan, "patch": p.patch}
                for p in scaled_plans
            ],
            "summary": summary,
            "metadata": metadata,
        },
    )
    success_rate = summary["outcome_rates"].get("Successful Transfer", 0)
    log_line(verbose, f"\nScaling complete: {n_successful}/{len(queries_list)} queries transferred successfully")
    log_line(verbose, f"  Results saved to: {result_path}")
    log_line(verbose, f"  Transfer success rate: {success_rate:.0%}")

    return ScaleResult(
        run_dir=str(run_dir_path),
        summary=summary,
        queries=queries_list,
        scaled_plans=scaled_plans,
        metadata=metadata,
        transfer_failures=transfer_failures,
    )


def validate_queries(
    queries: Sequence[str],
    *,
    dataset: str,
    scale_factor: int = DEFAULT_SCALE_FACTOR,
    n_determinism_retries: int = 3,
    max_workers: int = 10,
    exec_local: bool = False,
    verbose: bool = True,
    **kwargs,
) -> QueryValidationResult:
    """
    Validate SQL queries for syntax, executability, and determinism.

    Each query is planned and executed on the target dataset. Queries that
    parse, execute successfully, return non-empty results, and produce
    identical outputs across *n_determinism_retries* executions are
    considered valid. Use this as a pre-flight check before
    ``optimize_queries`` or ``benchmark_queries``.

    Args:
        queries: SQL query strings to validate.
        dataset: Dataset name (e.g., "tpch", "tpcds").
        scale_factor: Scale factor for the Modal runner.
        n_determinism_retries: Number of executions for determinism
            checking. Higher values increase confidence but cost more.
        max_workers: Max concurrent validation workers.
        exec_local: Validate locally (via LocalRunner) instead of on Modal.
            Requires the ``local`` extra; missing data is generated under
            ``data/`` on first use.
        verbose: Print progress and summary.
        **kwargs: Passed through to the validation backend. Supports
            ``runner_kwargs`` (dict for ModalRunner constructor),
            ``cpu``, ``memory``, ``timeout``, ``sandbox_timeout``.

    Returns:
        QueryValidationResult with per-query plans, errors, and summary.
    """
    from query_gen.query_validator import check_query_validity

    if isinstance(queries, str):
        raise ValueError("queries must be a sequence of strings, not a single string")
    queries_list: List[str] = list(queries)
    if not queries_list:
        raise ValueError("queries must not be empty")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    # Ensure runner_kwargs has scale_factor
    if "runner_kwargs" not in kwargs or kwargs["runner_kwargs"] is None:
        kwargs["runner_kwargs"] = {}
    kwargs["runner_kwargs"] = dict(kwargs["runner_kwargs"])
    if (
        "scale_factor" in kwargs["runner_kwargs"]
        and kwargs["runner_kwargs"]["scale_factor"] != scale_factor
    ):
        raise ValueError("scale_factor conflicts with runner_kwargs['scale_factor']")
    kwargs["runner_kwargs"]["scale_factor"] = scale_factor

    if exec_local:
        ensure_local_data(dataset, scale_factor)

    plans: List[Optional[Dict[str, Any]]] = [None] * len(queries_list)
    errors: List[Optional[str]] = [None] * len(queries_list)

    _ERROR_CATEGORIES = (
        "syntax_error", "no_plan", "cannot_run",
        "empty_result", "nondeterministic", "validation_error",
    )

    def _categorize(vr):
        if vr.error is not None:
            return "validation_error"
        if not vr.is_syntax_valid:
            return "syntax_error"
        if not vr.plan:
            return "no_plan"
        if not vr.can_run:
            return "cannot_run"
        if vr.is_empty:
            return "empty_result"
        if vr.is_nondeterministic:
            return "nondeterministic"
        return None

    def _worker(idx: int, query: str):
        vr = check_query_validity(
            query=query,
            dataset=dataset,
            n_determinism_retries=n_determinism_retries,
            verbose=False,
            exec_local=exec_local,
            **kwargs,
        )
        category = _categorize(vr)
        if category is None:
            plan = vr.plan
            if isinstance(plan, str):
                plan = json.loads(plan)
            return idx, plan, None
        return idx, None, category

    log_line(verbose, f"Validating {len(queries_list)} queries on {dataset} (scale_factor={scale_factor})...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker, i, q): i
            for i, q in enumerate(queries_list)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating queries", disable=not verbose):
            idx, plan, err = future.result()
            plans[idx] = plan
            errors[idx] = err

    counts = {cat: 0 for cat in _ERROR_CATEGORIES}
    for err in errors:
        if err is not None and err in counts:
            counts[err] += 1
    n_valid = sum(1 for e in errors if e is None)
    summary = {
        "n_queries": len(queries_list),
        "n_valid": n_valid,
        **{k: v for k, v in counts.items() if v > 0},
    }

    if verbose:
        log_line(verbose, f"Validation complete: {n_valid}/{len(queries_list)} queries valid")
        failures = {k: v for k, v in counts.items() if v > 0}
        if failures:
            for k, v in failures.items():
                log_line(verbose, f"  {k}: {v}")

    return QueryValidationResult(plans=plans, errors=errors, summary=summary)


__all__ = [
    "QueryGenerationResult",
    "PatchedPlan",
    "OptimizationResult",
    "BenchmarkResult",
    "ScaleResult",
    "PlanningResult",
    "QueryValidationResult",
    "generate_queries",
    "optimize_queries",
    "validate_queries",
    "benchmark_plans",
    "get_engine_plans",
    "benchmark_queries",
    "scale_optimizations",
]
