"""Generate multiple optimization samples per query."""

import json
from pathlib import Path
from typing import List, Dict, Any, Literal, Optional
import copy
from dataclasses import dataclass, field

from sampling.gpt_plan_optimizer import GPTPlanOptimizer
from sampling.utils import apply_patches_to_plan
from dbplanbench_utils import write_json, log_line, get_metric_value

@dataclass
class SamplingStrategy:
    """Controls which existing plans serve as base candidates for new LLM sampling.

    Attributes:
        starting_selector: Which plans to consider ("original", "so_far",
            "of_last", "of_leafs").
        final_selector: How to narrow them ("all", "best_1", "best_n").
        from_valid_only: If True, only include plans that passed validation.
        kwargs: Extra strategy params (e.g. ``optimization_metric``, ``best_n``).
    """
    starting_selector: Literal["original", "so_far", "of_last", "of_leafs"]
    final_selector: Literal["all", "best_1", "best_n"]
    from_valid_only: bool
    kwargs: Dict[str, Any] = field(default_factory=dict)

# Sample predefined strategies
ORIGINAL_STRATEGY = SamplingStrategy(
    starting_selector="original",
    final_selector="all",
    from_valid_only=False
)

def get_upstream_patches(
    plan_info: Dict[str, Any],
    query_data: Dict[str, Any],
    sampleid2sampleidx: Dict[int, int],
    must_apply_current_patches: bool = False
) -> List[Dict[str, Any]]:
    """Collect the flattened chain of patches from root to *plan_info*.

    Walks the parent chain and gathers valid patches (or all patches for
    the first node when *must_apply_current_patches* is True).

    Args:
        plan_info: Sampled plan node to trace back from.
        query_data: Full query data containing all sampled_plans.
        sampleid2sampleidx: Mapping from sample_id to index in sampled_plans.
        must_apply_current_patches: Include patches at *plan_info* even if invalid.

    Returns:
        Flat list of patch operations ordered root-to-current.
    """
    
    patch_stack = []
    while plan_info["sample_id"] is not None:
        if must_apply_current_patches or plan_info["is_valid"]:
            patch_stack.append(plan_info["sampled_patches"])
        must_apply_current_patches = False # only usable for the first iteration
        plan_info = query_data["sampled_plans"][
            sampleid2sampleidx[plan_info["parent_sample_id"]]
        ]
    upstream_patch_stack = patch_stack[::-1] # reverse to get from root to current
    upstream_patches = [
        patch
        for patches in upstream_patch_stack
        for patch in patches
    ]
    return upstream_patches

def get_upstream_evaluation_stats(
    plan_info: Dict[str, Any],
    query_data: Dict[str, Any],
    sampleid2sampleidx: Dict[int, int],
):
    """Walk the parent chain and return the evaluation_stats of the nearest valid ancestor."""
    while plan_info["sample_id"] is not None:
        if plan_info["is_valid"]:
            return plan_info.get("evaluation_stats")
        plan_info = query_data["sampled_plans"][
            sampleid2sampleidx[plan_info["parent_sample_id"]]
        ]
    if not plan_info["is_valid"]:
        raise RuntimeError("Root plan must be valid!")
    return plan_info.get("evaluation_stats")

def get_base_plans_for_sampling(
    query_data: Dict[str, Any],
    sampling_strategy: SamplingStrategy,
) -> List[Dict[str, Any]]:
    """Select base plans to seed new LLM sampling from, per *sampling_strategy*.

    Filters and ranks the existing sample tree according to
    ``starting_selector``, ``from_valid_only``, and ``final_selector``.

    Args:
        query_data: Query data with ``plan`` and ``sampled_plans``.
        sampling_strategy: Strategy controlling selection/ranking.

    Returns:
        List of dicts with ``parent_sample_id``, ``upstream_patches``,
        and ``upstream_evaluation_stats``.
    """

    optimization_metric = sampling_strategy.kwargs.get('optimization_metric', None)
    best_n = sampling_strategy.kwargs.get('best_n', None)

    sampleid2sampleidx = {
        plan_info["sample_id"]: idx
        for idx, plan_info in enumerate(query_data["sampled_plans"])
    }

    if sampling_strategy.starting_selector == "original":
        return [
            {
                'parent_sample_id': None,
                'upstream_patches': [],
                'upstream_evaluation_stats': (
                    query_data["sampled_plans"][sampleid2sampleidx[None]].get("evaluation_stats")
                ),
            }
        ]

    def get_best_plan_infos(plan_infos: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        if optimization_metric is None:
            raise ValueError("optimization_metric must be provided to determine best plans!")
        try:
            key_fn = lambda x: (
                get_metric_value(x.get('evaluation_stats'), optimization_metric)
                if x.get('evaluation_stats') is not None else None
            ) or float('inf')
            return sorted(plan_infos, key=key_fn)[:n]
        except Exception as e:
            raise RuntimeError(f"Failed to determine best plan for {query_data}: {e}")

    base_plans = query_data["sampled_plans"]

    if sampling_strategy.starting_selector == "of_last":
        base_plans = [
            plan_info
            for plan_info in query_data["sampled_plans"]
            if plan_info["is_last"]
        ]

    elif sampling_strategy.starting_selector == "of_leafs":
        base_plans = [
            plan_info
            for plan_info in query_data["sampled_plans"]
            if plan_info["is_leaf"]
        ]

    if sampling_strategy.from_valid_only:
        base_plans = [
            plan_info
            for plan_info in base_plans
            if plan_info["is_valid"]
        ]

    if sampling_strategy.final_selector == "best_1":
        base_plans = get_best_plan_infos(base_plans, 1)

    elif sampling_strategy.final_selector == "best_n":
        if best_n is None:
            raise ValueError("best_n must be provided for BEST_N_* sampling!")
        base_plans = get_best_plan_infos(base_plans, best_n)

    return [
        {
            'parent_sample_id': plan_info["sample_id"],
            'upstream_patches': get_upstream_patches(plan_info, query_data, sampleid2sampleidx),
            'upstream_evaluation_stats': get_upstream_evaluation_stats(plan_info, query_data, sampleid2sampleidx),
        }
        for plan_info in base_plans
    ]


def sample_plans_from_file(
    input_file: str,
    output_file: str,
    sampling_strategy: SamplingStrategy = ORIGINAL_STRATEGY,
    n_samples: int = 5,
    model: str = "gpt-5",
    verbose: bool = False,
    completion_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    """Sample N LLM optimization attempts per query from an existing plan collection.

    Selects base candidates via *sampling_strategy*, generates patches
    through ``GPTPlanOptimizer``, and writes the results to *output_file*.

    Args:
        input_file: Path to accumulated sampling-state JSON file.
        output_file: Output file path for the new sampled plans.
        sampling_strategy: Controls which plans serve as LLM input.
        n_samples: Number of optimization attempts per base plan.
        model: LLM model name for generation.
        verbose: Print progress.
        **kwargs: Forwarded to ``optimize_plans_batch``.

    Returns:
        Path to *output_file*.
    """
    
    # Load queries
    with open(input_file, 'r') as f:
        queries = json.load(f)

    sampling_results = []
    for query_data in queries:
        base_plans = get_base_plans_for_sampling(
            query_data,
            sampling_strategy,
        )
        sampling_results.append({
            "id": query_data["id"],
            "query": query_data["query"],
            "plan": query_data["plan"],
            "model": model,
            "sampled_plans": base_plans,
        })

    optimizer = GPTPlanOptimizer(model=model, completion_kwargs=completion_kwargs)

    queries_batch = [
        (
            sampling_result["query"], 
            apply_patches_to_plan(
                sampling_result["plan"],
                sampled_plan["upstream_patches"]
            )
        )
        for sampling_result in sampling_results
        for sampled_plan in sampling_result["sampled_plans"]
    ]

    all_generation_results = optimizer.optimize_plans_batch(queries_batch, n_samples=n_samples, verbose=verbose, **kwargs)

    it = iter(all_generation_results)
    for sampling_result in sampling_results:
        final_sampled_plans = []
        for sampled_plan in sampling_result["sampled_plans"]:
            generation_results_list = next(it)
            for generation_results in generation_results_list:
                new_sampled_plan = copy.deepcopy(sampled_plan)
                
                new_sampled_plan["sampled_patches"] = generation_results.sampled_patches
                if generation_results.error_message is not None:
                    new_sampled_plan["is_valid"] = False
                    new_sampled_plan["error_message"] = generation_results.error_message
                else:
                    new_sampled_plan["is_valid"] = True
                    new_sampled_plan["error_message"] = None
                new_sampled_plan["model_response"] = generation_results.model_response
                new_sampled_plan["reasoning_content"] = generation_results.reasoning_content
                new_sampled_plan["prompt_tokens"] = generation_results.prompt_tokens
                new_sampled_plan["completion_tokens"] = generation_results.completion_tokens
                new_sampled_plan["total_tokens"] = generation_results.total_tokens

                final_sampled_plans.append(new_sampled_plan)
        sampling_result["sampled_plans"] = final_sampled_plans
    
    # Save results
    write_json(Path(output_file), sampling_results)

    return output_file
