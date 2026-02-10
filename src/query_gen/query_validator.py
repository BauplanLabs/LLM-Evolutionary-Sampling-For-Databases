from dataclasses import dataclass
import json
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Literal, Dict, Any, List, Set, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from modal_controller.modal_runner import Operation
from modal_controller.constants import DEFAULT_SCALE_FACTOR
from modal_controller.utils import submit_run_operation, validate_plan_result_set
from dbplanbench_utils import log_line, data_folder_for_dataset, plan_to_json, write_json

@dataclass
class ValidationResult:
    """Result of validating a single SQL query on Modal.

    Attributes:
        error: Infrastructure/runtime error string, else None. This captures also the
            failures in the validation flow itself (for example Modal/S3
            failures or determinism-check execution failures).
        is_syntax_valid: Query parses successfully.
        plan: Physical execution plan JSON when planning succeeds. May be
            None for syntax-invalid queries or planning failures.
        can_run: Query executes without errors.
        is_empty: Query returns zero rows.
        row_count: Number of result rows.
        execution_time: Execution wall time in seconds.
        is_nondeterministic: Result set differs across repeated executions.
    """
    error: Optional[str]
    is_syntax_valid: bool
    plan: Optional[str]
    can_run: bool
    is_empty: bool
    row_count: int
    execution_time: float
    is_nondeterministic: bool

def check_query_validity(
    query: str,
    dataset: str = "tpch",
    n_determinism_retries: int = 3,
    verbose: bool = True,
    **kwargs
) -> ValidationResult:
    """Validate a SQL query on Modal for syntax, executability, and determinism.

    Sends the query to Modal for planning/execution, then checks determinism
    via ``validate_plan_result_set`` (executes the plan multiple times and
    compares result sets).

    Args:
        query: SQL query string.
        dataset: Dataset to validate against.
        n_determinism_retries: Number of executions for determinism check.
        verbose: Print validation errors.
        **kwargs: Forwarded to ``submit_run_operation``.

    Returns:
        ValidationResult with all check outcomes.
    """

    def bad_result(error_msg: str, result: Optional[dict] = None) -> ValidationResult:
        if result is None:
            result = {}
        log_line(verbose, f"Validation error: {error_msg}")
        return ValidationResult(
            error=error_msg,
            is_syntax_valid=result.get('is_syntax_valid', False),
            plan=result.get('plan', None),
            can_run=result.get('can_run', False),
            is_empty=result.get('is_empty', True),
            row_count=result.get('row_count', 0),
            execution_time=result.get('execution_time', 0.0),
            is_nondeterministic=False
        )

    data_folder = data_folder_for_dataset(dataset)

    result = submit_run_operation(
        operation=Operation.VALIDATE,
        input_str=query,
        data_folder=data_folder,
        n_retry=1,
        **kwargs
    )

    if not result:
        return bad_result("Failed to get validation result for query")

    if 'error' in result:
        return bad_result(result['error'])

    non_deterministic = False
    if result['is_syntax_valid'] and result['plan'] is not None and result['can_run'] and not result['is_empty']:
        plan_json = plan_to_json(result["plan"])
        validation_err = validate_plan_result_set(
            plan_json, plan_json, data_folder,
            n_retry=1,
            n_determinism_retries=n_determinism_retries,
            **kwargs,
        )
        if validation_err is not None:
            if "mismatch" in validation_err or "Non-deterministic" in validation_err:
                log_line(verbose, "Detected non-deterministic behavior in query results")
                non_deterministic = True
            else:
                return bad_result(
                    f"Failed to time query plan for nondeterminism check. Reason: {validation_err}",
                    result,
                )

    return ValidationResult(
        error=None,
        is_syntax_valid=result["is_syntax_valid"],
        plan=result["plan"],
        can_run=result["can_run"],
        is_empty=result["is_empty"],
        row_count=result["row_count"],
        execution_time=result["execution_time"],
        is_nondeterministic=non_deterministic
    )

def save_dead_letter_queries_to_file(
    dead_letters: Dict[str, Set[str]],
    output_file: str,
):
    """Write failed queries grouped by failure category to *output_file*."""
    write_json(Path(output_file), {k: list(v) for k, v in dead_letters.items()})


def save_mapping_to_json_with_complexity(
    query_to_data: dict,
    output_file: str,
):
    """Write validated queries with plans, complexity, and row_count to *output_file*."""
    data = []
    for query_data in query_to_data.values():
        try:
            parsed_plan = json.loads(query_data['plan']) if isinstance(query_data['plan'], str) else query_data['plan']
        except (json.JSONDecodeError, TypeError):
            parsed_plan = query_data['plan']

        data.append({
            'id': query_data['id'],
            "query": query_data['query'],
            "plan": parsed_plan,
            "complexity": query_data['complexity'],
            "row_count": query_data['row_count'],
        })

    write_json(Path(output_file), data)


def validate_queries(
    input_file: str,
    output_file: str,
    dead_letter_file: Optional[str],
    dataset: str = "tpch",
    max_workers: int = 5,
    output_format: Literal["plans", "validation"] = "plans",
    scale_factor: int = DEFAULT_SCALE_FACTOR,
    seen_queries: Optional[Set[str]] = None,
    verbose: bool = True,
    log_context: str = "generation",
    **kwargs
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Validate generated queries concurrently on Modal and save results.

    Each query is checked for syntax, executability, empty result, and
    determinism. Duplicates (against *seen_queries*) are rejected.

    Args:
        input_file: JSON file with generated queries (id, query, complexity).
        output_file: Destination for validation results.
        dead_letter_file: Optional file for failed queries by category.
        dataset: Dataset to validate against ("tpch" or "tpcds").
        max_workers: Concurrent validation workers.
        output_format: ``"plans"`` saves valid queries with plans;
            ``"validation"`` saves per-query validation metadata.
        scale_factor: Scale factor for Modal runner.
        seen_queries: Set of already-accepted queries (updated in place).
        verbose: Print progress.
        log_context: Label for summary messages ("generation" or "optimization").
        **kwargs: Forwarded to Modal calls (``runner_kwargs`` merged with
            *scale_factor*).

    Returns:
        ``query_to_data`` dict (format=plans) or list of validation dicts.
    """

    with open(input_file, 'r') as f:
        generated_queries = json.load(f)
    log_line(verbose, f"Validation batch: n={len(generated_queries)} dataset={dataset.upper()}")
    
    # map queries (avoid duplicates) to their plans and metadata
    query_to_data = dict()
    seen = seen_queries if seen_queries is not None else set()
    validation_results_by_id: Dict[int, Dict[str, Any]] = {}

    kwargs_local = dict(kwargs)
    runner_kwargs = kwargs_local.pop("runner_kwargs", {}) or {}
    if not isinstance(runner_kwargs, dict):
        raise ValueError("runner_kwargs must be a dict")
    if scale_factor is None:
        raise ValueError("scale_factor cannot be None")
    runner_kwargs = dict(runner_kwargs)
    if "scale_factor" in runner_kwargs and runner_kwargs["scale_factor"] != scale_factor:
        raise ValueError("scale_factor conflicts with runner_kwargs['scale_factor']")
    runner_kwargs["scale_factor"] = scale_factor
    _DEAD_LETTER_CATEGORIES = (
        "validation_error", "syntax_error", "no_plan",
        "cannot_run", "empty_result", "nondeterministic", "duplicate",
    )
    dead_letters: Dict[str, Set[str]] = {cat: set() for cat in _DEAD_LETTER_CATEGORIES}
    
    log_per_query = verbose and max_workers <= 1

    def validate_single_query(result):
        """Validate a single query and return the result."""
        new_query = result['query']
        complexity = result['complexity']
        query_id = result['id']
        if log_per_query:
            log_line(verbose, f"Query {query_id+1} (complexity {complexity})")
        
        # validate if the query is valid by sending it to the Modal endpoint
        validation_result = check_query_validity(
            query=new_query,
            dataset=dataset,
            runner_kwargs=runner_kwargs,
            verbose=log_per_query,
            **kwargs_local
        )
        
        if validation_result.error is not None:
            if log_per_query:
                log_line(verbose, f"Query {query_id+1} failed validation: {validation_result.error}")
            return {'type': 'validation_error', 
                    'id': query_id,
                    'query': new_query,
                    'complexity': complexity}
        elif not validation_result.is_syntax_valid:
            if log_per_query:
                log_line(verbose, f"Query {query_id+1} invalid syntax")
            return {'type': 'syntax_error', 
                    'id': query_id,
                    'query': new_query,
                    'complexity': complexity}
        elif not validation_result.plan:
            if log_per_query:
                log_line(verbose, f"Query {query_id+1} has no physical plan")
            return {'type': 'no_plan', 
                    'id': query_id,
                    'query': new_query,
                    'complexity': complexity}
        elif not validation_result.can_run:
            if log_per_query:
                log_line(verbose, f"Query {query_id+1} cannot run")
            return {'type': 'cannot_run', 
                    'id': query_id,
                    'query': new_query,
                    'complexity': complexity}
        elif validation_result.is_empty:
            if log_per_query:
                log_line(verbose, f"Query {query_id+1} empty result ({validation_result.row_count} rows)")
            return {'type': 'empty_result', 
                    'id': query_id,
                    'query': new_query,
                    'complexity': complexity}
        elif validation_result.is_nondeterministic:
            if log_per_query:
                log_line(verbose, f"Query {query_id+1} non-deterministic")
            return {'type': 'nondeterministic', 
                    'id': query_id,
                    'query': new_query,
                    'complexity': complexity}

        # Return valid query data
        return {
            'type': 'valid',
            'id': query_id,
            'query': new_query,
            'plan': validation_result.plan,
            'complexity': complexity,
            'row_count': validation_result.row_count
        }
    
    # Validate queries concurrently
    log_line(verbose, f"Validating with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all validation tasks
        future_to_result = {executor.submit(validate_single_query, result): result for result in random.sample(generated_queries, len(generated_queries))}
        
        # Process completed tasks
        for future in tqdm(
            as_completed(future_to_result),
            total=len(generated_queries),
            desc="Validating queries",
            disable=not verbose,
        ):
            result = future.result()
            
            result_type = result['type']
            if result_type == 'valid':
                if result['query'] in seen:
                    result_type = 'duplicate'
                else:
                    seen.add(result['query'])
                    if output_format == "plans":
                        query_to_data[result['query']] = {
                            'id': result['id'],
                            'query': result['query'],
                            'plan': result['plan'],
                            'complexity': result['complexity'],
                            'row_count': result['row_count']
                        }

            if result_type != 'valid':
                dead_letters[result_type].add(result['query'])

            validation_results_by_id[result['id']] = {
                'id': result['id'],
                'query': result['query'],
                'complexity': result.get('complexity'),
                'is_valid': result_type == 'valid',
                'error': None if result_type == 'valid' else result_type,
            }
    
    if output_format == "plans":
        valid_count = len(query_to_data)
    else:
        valid_count = sum(1 for v in validation_results_by_id.values() if v.get("is_valid"))
    if verbose:
        log_line(verbose, f"Valid: {valid_count}/{len(generated_queries)}")
        nonzero = {k: len(v) for k, v in dead_letters.items() if v}
        if nonzero:
            log_line(verbose, "Failures:")
            for k, v in nonzero.items():
                log_line(verbose, f"  {k}: {v}")
    
    if output_format == "plans":
        # save the mapping to a json file
        save_mapping_to_json_with_complexity(
            query_to_data=query_to_data,
            output_file=output_file
        )
    elif output_format == "validation":
        ordered = [validation_results_by_id[k] for k in sorted(validation_results_by_id)]
        write_json(Path(output_file), ordered)
    else:
        raise ValueError(f"Unknown output_format: {output_format}")
    if dead_letter_file:
        save_dead_letter_queries_to_file(dead_letters, dead_letter_file)
    
    # Print summary statistics
    if output_format == "plans":
        actual_complexities = [data['complexity'] for data in query_to_data.values()]
    else:
        actual_complexities = [
            data['complexity']
            for data in validation_results_by_id.values()
            if data['is_valid']
        ]
    complexity_counts = {}
    for c in actual_complexities:
        complexity_counts[c] = complexity_counts.get(c, 0) + 1
    
    if verbose:
        if complexity_counts and not (len(complexity_counts) == 1 and 0 in complexity_counts):
            if log_context == "generation":
                log_line(verbose, "Valid SQL by complexity:")
            else:
                log_line(verbose, "Valid by complexity:")
            for complexity, count in sorted(complexity_counts.items()):
                percentage = (count / valid_count) * 100 if valid_count else 0
                log_line(verbose, f"  {complexity}: {count} ({percentage:.1f}%)")
    
    if output_format == "plans":
        return query_to_data
    return [validation_results_by_id[k] for k in sorted(validation_results_by_id)]
