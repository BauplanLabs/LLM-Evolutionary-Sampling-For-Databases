from datetime import datetime
import json
from typing import Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from modal_controller.modal_runner import Operation
from modal_controller.utils import evaluate_plan_n_runs
from pathlib import Path
from dbplanbench_utils import log_line, data_folder_for_dataset, plan_to_json, write_json, get_metric_value


def evaluate_validated_queries(
    input_file: str,
    output_file: str,
    dead_letter_file: Optional[str] = None,
    dataset: str = "tpch",
    max_workers: int = 3,
    n_runs: int = 1,
    optimization_metric: Optional[str] = None,
    verbose: bool = True,
    **kwargs,
):
    """Benchmark validated queries by executing their plans on Modal.

    Runs each query plan *n_runs* times concurrently and records per-query
    evaluation statistics. Optionally prints per-complexity summaries.

    Args:
        input_file: JSON file with validated queries (must include ``plan``).
        output_file: Destination for evaluation results JSON.
        dead_letter_file: Optional file for queries that failed evaluation.
        dataset: Dataset to evaluate against ("tpch" or "tpcds").
        max_workers: Concurrent evaluation workers.
        n_runs: Evaluation runs per plan.
        optimization_metric: Metric path for per-complexity summaries (e.g.
            ``"execution_time.min"``). None to skip.
        verbose: Print progress.
        **kwargs: Forwarded to ``evaluate_plan_n_runs``.

    Returns:
        List of query dicts augmented with ``evaluation_stats``.
    """
    
    log_line(verbose, f"Evaluating validated queries from {input_file} against {dataset.upper()} dataset at {datetime.now()}")
    
    # Load validated queries with plans
    with open(input_file, 'r') as file:
        validated_data = json.load(file)
    
    log_line(verbose, f"Loaded {len(validated_data)} validated queries for evaluation")

    log_per_query = verbose and max_workers <= 1
    
    def evaluate_single_query(data):
        """Evaluate execution of a single query plan."""
        query = data['query']
        plan = data['plan']
        complexity = data['complexity']
        
        if log_per_query:
            log_line(verbose, f"Evaluating query (complexity {complexity})")
        data_folder = data_folder_for_dataset(dataset)

        plan_json = plan_to_json(plan)

        try:
            evaluation_summary = evaluate_plan_n_runs(
                operation=Operation.EVALUATE,
                plan_json=plan_json,
                data_folder=data_folder,
                n_runs=n_runs,
                **kwargs,
            )

            error = evaluation_summary.get('error', None)
            if error:
                log_line(verbose, f"Failed to get evaluation result for query: {error}")

            result_payload = {
                **data,
                'evaluation_stats': evaluation_summary,
            }

            return result_payload

        except Exception as e:
            log_line(verbose, f"Error evaluating query: {e}")
            return {
                **data,
                'evaluation_stats': {
                    'error': str(e)
                },
            }
    
    # Time queries concurrently
    log_line(verbose, f"Evaluating queries with {max_workers} workers...")
    evaluation_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all evaluation tasks
        future_to_data = {executor.submit(evaluate_single_query, data): data for data in random.sample(validated_data, len(validated_data))}
        
        # Process completed tasks
        for future in tqdm(as_completed(future_to_data), total=len(validated_data), desc="Evaluating queries"):
            result = future.result()
            evaluation_results.append(result)
    
    # Filter successful results and calculate statistics
    successful_results = [r for r in evaluation_results if 'error' not in r.get('evaluation_stats', {})]
    failed_results = [r for r in evaluation_results if 'error' in r.get('evaluation_stats', {})]

    log_line(verbose, f"Successfully evaluated {len(successful_results)} queries out of {len(validated_data)}")
    if failed_results:
        log_line(verbose, f"Failed to evaluate {len(failed_results)} queries.")
    
    # Print evaluation statistics
    if successful_results and optimization_metric:
        # Evaluation by complexity
        complexity_times = {}
        for result in successful_results:
            exec_time = get_metric_value(result.get('evaluation_stats', {}), optimization_metric)
            if exec_time is None:
                continue
            complexity = result['complexity']
            if complexity not in complexity_times:
                complexity_times[complexity] = []
            complexity_times[complexity].append(exec_time)
        
        if complexity_times:
            log_line(verbose, f"\nEvaluation by complexity:")
            for complexity, times in sorted(complexity_times.items()):
                avg_complexity_time = sum(times) / len(times)
                log_line(verbose, f"  Complexity {complexity}: {avg_complexity_time:.4f}s avg ({len(times)} queries)")

    # Save evaluation results
    write_json(Path(output_file), evaluation_results)
    log_line(verbose, f"Evaluation results saved to {output_file}")

    if dead_letter_file:
        write_json(Path(dead_letter_file), {'failed_evaluations': failed_results})
        log_line(verbose, f"Dead letter file saved to {dead_letter_file}")

    log_line(verbose, f"Evaluation completed at {datetime.now()}")
    
    return evaluation_results
