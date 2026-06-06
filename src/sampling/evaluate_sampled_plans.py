"""Evaluate sampled plans using Modal infrastructure."""

import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm

from modal_controller.modal_runner import Operation
from modal_controller.constants import DEFAULT_SCALE_FACTOR
from modal_controller.utils import evaluate_plan_n_runs, validate_plan_result_set
from dbplanbench_utils import write_json, log_line, data_folder_for_dataset, plan_to_json
from sampling.utils import apply_patches_to_plan

def evaluate_sampled_plans(
    input_file: str,
    output_file: str,
    dataset: str = "tpch",
    n_runs: int = 3,
    max_workers: int = 10,
    verbose: bool = True,
    exec_local: bool = False,
    **kwargs
) -> str:
    """Validate and benchmark each sampled plan, writing results back in place.

    For every sampled plan with non-empty patches, validates correctness via
    result-set comparison against the original plan, then benchmarks with
    *n_runs* evaluation runs. Plans that were already invalid or have empty
    patches are handled without remote calls.

    Args:
        input_file: Path to sampled plans JSON (output of ``sample_plans_from_file``).
        output_file: Destination for the annotated sampled plans JSON.
        dataset: Dataset to evaluate against ("tpch" or "tpcds").
        n_runs: Number of evaluation runs per valid plan.
        max_workers: Maximum concurrent evaluation workers.
        verbose: Print progress.
        exec_local: If True, validate/evaluate locally instead of on Modal.
        **kwargs: Forwarded to validation and evaluation backends.

    Returns:
        Path to *output_file*.
    """

    # Load sampled plans
    with open(input_file, 'r') as f:
        sampled_data = json.load(f)

    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    # Local data folders are scale-specific; the scale comes from runner_kwargs
    # (always set by callers, same source the Modal path uses).
    scale_factor = (kwargs.get("runner_kwargs") or {}).get("scale_factor", DEFAULT_SCALE_FACTOR)

    def invalid_result(error_message: str) -> Dict[str, Any]:
        return {
            "is_valid": False,
            "error_message": error_message,
            "evaluation_stats": None
        }

    def valid_result(evaluation_stats: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "is_valid": True,
            "error_message": None,
            "evaluation_stats": evaluation_stats,
        }

    def validate_evaluate_plan_wrapper(
        plan_str: str,
        original_plan_str: str
    ) -> Dict[str, Any]:

        data_folder = data_folder_for_dataset(dataset, exec_local=exec_local, scale_factor=scale_factor)

        # Validation
        validation_error = validate_plan_result_set(
            plan_str, original_plan_str, data_folder, n_retry=5, verbose=verbose,
            exec_local=exec_local, **kwargs
        )
        if validation_error is not None:
            return invalid_result(validation_error)

        # Evaluation
        try:
            evaluation_stats = evaluate_plan_n_runs(
                Operation.EVALUATE,
                plan_str,
                data_folder,
                n_runs,
                verbose=verbose,
                exec_local=exec_local,
                **kwargs
            )
            if evaluation_stats.get("error", None):
                return invalid_result(f"Evaluation failed: {evaluation_stats.get('error', 'Unknown error')}")

            return valid_result(evaluation_stats)

        except Exception as exc:
            return invalid_result(f"Evaluation failed: {exc}")


    plan_results: Dict[Tuple[int, int], Dict[str, Any]] = {}
    errors: Dict[Tuple[int, int], str] = {}
    future_to_metadata: Dict[Any, Tuple[int, int]] = {}

    max_inflight = max(1, max_workers * 2)  # tweak: 2x is a good default
    submitted = 0

    def harvest(done_futures, progress):
        nonlocal submitted
        for future in done_futures:
            query_index, sample_index = future_to_metadata.pop(future)
            try:
                plan_result = future.result()
                plan_results[(query_index, sample_index)] = plan_result
            except Exception as exc:
                log_line(verbose, f"Warning: Failed to get evaluation result for query {query_index} sample {sample_index}: {exc}")
                errors[(query_index, sample_index)] = f"Failed to get evaluation result: {exc}"
            if progress:
                progress.update(1)

    inflight = set()
    progress = tqdm(total=0, desc="Evaluating plans", disable=not verbose)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for query_index, query_data in random.sample(list(enumerate(sampled_data)), k=len(sampled_data)):
            plans = query_data["sampled_plans"]

            # Precompute baseline JSON string once per query (same behavior, less memory/cpu)
            original_plan_str = plan_to_json(query_data["plan"])

            for sample_index, sampled_plan in random.sample(list(enumerate(plans)), k=len(plans)):
                if sampled_plan["is_valid"] is False:
                    sampled_plan.update(
                        invalid_result(sampled_plan["error_message"])
                    )
                    continue

                try:
                    if sampled_plan["sampled_patches"] is None:
                        raise ValueError("sampled_patches is None")

                    if len(sampled_plan["sampled_patches"]) == 0:
                        plan_results[(query_index, sample_index)] = valid_result(
                            sampled_plan.get("upstream_evaluation_stats")
                        )
                        continue

                    plan = apply_patches_to_plan(
                        query_data["plan"],
                        sampled_plan["upstream_patches"] + sampled_plan["sampled_patches"]
                    )

                    # Convert candidate plan to JSON string BEFORE submit to avoid keeping big dict alive
                    plan_str = plan_to_json(plan)
                    del plan  # important: drop big dict now

                except Exception as exc:
                    error_msg = f"Failed to apply patches to plan: {exc}"
                    errors[(query_index, sample_index)] = error_msg
                    continue

                # Bound the number of queued futures
                while len(inflight) >= max_inflight:
                    done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    harvest(done, progress)

                future = executor.submit(
                    validate_evaluate_plan_wrapper,
                    plan_str,
                    original_plan_str
                )
                inflight.add(future)
                future_to_metadata[future] = (query_index, sample_index)

                submitted += 1
                progress.total = submitted

        # Drain remaining futures
        while inflight:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            harvest(done, progress)

    progress.close()

    for (query_index, sample_index), plan_result in plan_results.items():
        sampled_plan = sampled_data[query_index]["sampled_plans"][sample_index]
        sampled_plan.update(plan_result)
    
    for (query_index, sample_index), error_msg in errors.items():
        sampled_plan = sampled_data[query_index]["sampled_plans"][sample_index]
        sampled_plan.update(
            invalid_result(error_msg)
        )

    # Save results
    write_json(Path(output_file), sampled_data)

    return output_file
