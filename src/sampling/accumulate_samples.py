import json
from pathlib import Path
from api_utils import write_json, log_line

def accumulate_samples(
    input_file: str,
    output_file: str,
    verbose: bool = True,
) -> str:
    """Merge newly evaluated sampled plans into the accumulated sample tree.

    Reads *input_file* (step output) and *output_file* (accumulated state),
    assigns fresh sample IDs, updates ``is_last`` / ``is_leaf`` flags on
    existing nodes, appends the new samples, and writes back to *output_file*.

    Args:
        input_file: Evaluated sampled plans from the latest step.
        output_file: Accumulated sampling state file (read and overwritten).
        verbose: Print progress.

    Returns:
        Path to *output_file*.
    """

    # Load sampled plans with evaluation stats
    with open(input_file, 'r') as f:
        sampled_data = json.load(f)

    # Load the destination file to accumulate into
    with open(output_file, 'r') as f:
        final_data = json.load(f)

    # Create a mapping from query id to necessary metadata in final_data
    id2metadata = {}
    for query_info in final_data:
        query_id = query_info["id"]
        id2metadata[query_id] = {
            "new_sample_id": max(
                [
                    plan["sample_id"] if plan["sample_id"] is not None else 0
                    for plan in query_info["sampled_plans"]
                ],
                default=0
            ) + 1
        }

    # Accumulate new samples into final_data structure
    id2samples = {}
    leaf_parents = set()
    for query_sampling in sampled_data:
        query_id = query_sampling["id"]
        id2samples[query_id] = []
        new_sample_id = id2metadata[query_id]["new_sample_id"]
        for sampled_plan in query_sampling["sampled_plans"]:
            finalized_plan = {
                "plan_type": "optimized",
                "sample_id": new_sample_id,
                "parent_sample_id": sampled_plan["parent_sample_id"],
                "is_last": True,
                "is_leaf": True,
                "sampled_patches": sampled_plan["sampled_patches"],
                "is_valid": sampled_plan["is_valid"],
                "error_message": sampled_plan["error_message"],
                "evaluation_stats": sampled_plan.get("evaluation_stats"),
            }
            id2samples[query_id].append(finalized_plan)
            leaf_parents.add((query_id, sampled_plan["parent_sample_id"]))
            new_sample_id += 1

    # Update final_data with new samples and is_last / is_leaf flags
    for query_info in final_data:
        query_id = query_info["id"]
        new_samples = id2samples.get(query_id, [])
        for plan in query_info["sampled_plans"]:
            if len(new_samples) > 0:
                plan["is_last"] = False # not the last if new samples are added
            if (query_id, plan["sample_id"]) in leaf_parents:
                plan["is_leaf"] = False
        query_info["sampled_plans"].extend(new_samples)

    # Save accumulated results
    write_json(Path(output_file), final_data)
    
    log_line(verbose, "Updated sampling state.")

    return output_file
