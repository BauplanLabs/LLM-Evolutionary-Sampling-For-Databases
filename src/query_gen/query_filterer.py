from typing import Optional
from pathlib import Path
import json
from datetime import datetime
from dbplanbench_utils import log_line, write_json, get_metric_value

def filter_evaluation_queries(
    input_file: str,
    output_file: str,
    dead_letter_file: Optional[str],
    lowerbound_seconds: float,
    upperbound_seconds: float,
    runtime_metric: str = "execution_time.min",
    dataset: str = "tpch",
    verbose: bool = True,
):
    """Filter queries whose runtime metric falls within a given range.

    Keeps queries where ``lowerbound_seconds <= metric <= upperbound_seconds``
    and writes the rest to *dead_letter_file* (if provided).

    Args:
        input_file: Evaluated queries JSON file.
        output_file: Destination for filtered (kept) queries.
        dead_letter_file: Optional destination for dropped queries.
        lowerbound_seconds: Minimum runtime metric value (inclusive).
        upperbound_seconds: Maximum runtime metric value (inclusive).
        runtime_metric: Metric path to filter on (e.g. ``"execution_time.min"``).
        dataset: Dataset context for logging.
        verbose: Print progress.
    """
    log_line(verbose, f"Filtering queries ({lowerbound_seconds}s <= x <= {upperbound_seconds}s) from {input_file} for {dataset.upper()} dataset at {datetime.now()}")

    # Load queries
    with open(input_file, 'r') as f:
        query_data = json.load(f)

    log_line(verbose, f"Loaded {len(query_data)} queries for filtering")

    # Apply filter function
    def _in_range(q):
        val = get_metric_value(q.get('evaluation_stats', {}), runtime_metric)
        return val is not None and lowerbound_seconds <= val <= upperbound_seconds

    keep = [_in_range(q) for q in query_data]
    filtered_data = [q for q, k in zip(query_data, keep) if k]
    dropped_data = [q for q, k in zip(query_data, keep) if not k]

    log_line(verbose, f"Filtered down to {len(filtered_data)} queries after applying metric filter")
    log_line(verbose, f"Dropped {len(dropped_data)} queries after applying metric filter")

    # Save filtered and dead-letter queries
    write_json(Path(output_file), filtered_data)
    log_line(verbose, f"Filtered queries saved to {output_file}")

    if dead_letter_file is not None:
        write_json(Path(dead_letter_file), dropped_data)
        log_line(verbose, f"Dead letter queries saved to {dead_letter_file}")
