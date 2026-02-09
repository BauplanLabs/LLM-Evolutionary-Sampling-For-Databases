import json
from pathlib import Path
from api_utils import log_line, write_json

def prepare_sampling(
    input_file: str,
    output_file: str,
    model: str,
    verbose: bool = True,
):
    """Convert a curated dataset into the initial sampling-state format.

    Creates one root ``sampled_plans`` entry (the original plan) per query,
    ready for the iterative sample → evaluate → accumulate loop.

    Args:
        input_file: Curated dataset JSON (validated queries with plans).
        output_file: Destination for the prepared sampling-state JSON.
        model: LLM model name recorded in each entry.
        verbose: Print progress.
    """
    log_line(verbose, f"Prepare sampling: model={model}")

    # Load curated dataset
    with open(input_file, 'r') as f:
        curated_data = json.load(f)

    log_line(verbose, f"Curated queries: {len(curated_data)}")

    # Prepare data for sampling
    prepared_data = []
    for entry in curated_data:
        prepared_entry = {
            "id": entry["id"],
            "query": entry["query"],
            "plan": entry["plan"],
            "complexity": entry["complexity"],
            "row_count": entry["row_count"],
            "model": model,
            "sampled_plans": [
                {
                    "plan_type": "original",
                    "sample_id": None,
                    "parent_sample_id": None,
                    "is_last": True,
                    "is_leaf": True,
                    "sampled_patches": [],
                    "is_valid": True,
                    "error_message": None,
                    "evaluation_stats": entry.get("evaluation_stats")
                }
            ]
        }
        prepared_data.append(prepared_entry)

    # Save prepared dataset
    write_json(Path(output_file), prepared_data)

    log_line(verbose, "Prepare sampling: done")
