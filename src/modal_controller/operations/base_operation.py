"""Shared operation logic for both execution transports.

Holds the per-operation result builders (``op_*``) and plan-metric parsing.
Each ``op_*`` takes a ``DataFusionDB`` handle (or, for fetch-schema, a data
folder) and returns the exact result dict that the Modal operation scripts
persist to S3 and that ``LocalRunner`` returns in-process — the transports
differ only in where input comes from and where the result goes.

Like ``db_base.py``, this module is BOTH imported in-process (by
``local_runner.py``) and concatenated as text by ``modal_runner.py`` ahead of
an ``operations/<op>.py`` script. It is engine-agnostic (it never imports
``datafusion``); the ``DataFusionDB`` instances it operates on are supplied by
the caller. Do not add ``from __future__ import annotations`` here — it would
land mid-file in the Modal concatenation and raise SyntaxError.
"""

import os
import random
import re
import time
from typing import TYPE_CHECKING, Any, Dict

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from modal_controller.db_base import DataFusionDB


# --- Plan-metric parsing -----------------------------------------------------

def _parse_duration_to_seconds(val: str) -> float:
    """Parse a DataFusion metric duration string (e.g. ``1.2ms``) to seconds."""
    val = val.strip()
    if val.endswith("ns"):
        return float(val[:-2]) * 1e-9
    if val.endswith("µs"):
        return float(val[:-2]) * 1e-6
    if val.endswith("ms"):
        return float(val[:-2]) * 1e-3
    if val.endswith("s"):
        return float(val[:-1])
    return float(val)


def _parse_int(val: str) -> int:
    return int(val.replace(",", "").strip())


def parse_plan_metrics(plan: Any) -> Dict[str, Any]:
    """Extract structural metrics from an executed plan's ``display_with_metrics``."""
    try:
        display_text = plan.display_with_metrics()
    except Exception:
        return {}

    parsed = []
    for i, line in enumerate(display_text.splitlines()):
        m = re.search(r"metrics=\[([^\]]+)\]", line)
        if not m:
            continue
        metrics = {}
        for metric_str in m.group(1).split(", "):
            if "=" in metric_str:
                k, v = metric_str.split("=", 1)
                metrics[k] = v
        parsed.append({"line": i, "text": line, "metrics": metrics})

    def is_join(text: str) -> bool:
        return bool(re.search(r"join", text, re.IGNORECASE))

    def safe_sum_int(key: str, join_only: bool = False) -> int | None:
        try:
            return sum(
                _parse_int(p["metrics"][key])
                for p in parsed
                if key in p["metrics"] and (not join_only or is_join(p["text"]))
            )
        except Exception:
            return None

    def safe_sum_duration(key: str, join_only: bool = False) -> float | None:
        try:
            return sum(
                _parse_duration_to_seconds(p["metrics"][key])
                for p in parsed
                if key in p["metrics"] and (not join_only or is_join(p["text"]))
            )
        except Exception:
            return None

    def safe_max_int(key: str, join_only: bool = False) -> int | None:
        try:
            vals = [
                _parse_int(p["metrics"][key])
                for p in parsed
                if key in p["metrics"] and (not join_only or is_join(p["text"]))
            ]
            return max(vals) if vals else 0
        except Exception:
            return None

    return {
        "bytes_scanned": safe_sum_int("bytes_scanned"),
        "output_rows_sum": safe_sum_int("output_rows", join_only=True),
        "input_rows_sum": safe_sum_int("input_rows", join_only=True),
        "join_time_s_sum": safe_sum_duration("join_time", join_only=True),
        "build_time_s_sum": safe_sum_duration("build_time", join_only=True),
        "build_mem_used_sum": safe_sum_int("build_mem_used", join_only=True),
        "build_mem_used_max": safe_max_int("build_mem_used", join_only=True),
        "peak_mem_used_max": safe_max_int("peak_mem_used"),
    }


# --- Operation result builders -----------------------------------------------

def op_plan(db: "DataFusionDB", query: str) -> Dict[str, Any]:
    """Serialize a SQL query to a succinct physical plan."""
    plan = db.serialize_query_to_physical_plan(query)
    return {"plan": plan, "query": query}


def op_execute(db: "DataFusionDB", plan_json: str) -> Dict[str, Any]:
    """Execute a serialized plan; return execution_time + result_data + schema."""
    start = time.perf_counter()
    results = db.execute_serialized_physical_plan(plan_json)
    execution_time = time.perf_counter() - start

    if results.is_exec_error:
        return {"error": f"Execution error: {results.error_message}"}
    if not isinstance(results.data, pa.Table):
        return {"error": f"Expected results to be a pyarrow Table, instead got {type(results.data)}."}

    return {
        "execution_time": execution_time,
        "result_data": results.data.to_pandas().to_json(),
        "schema": str(results.data.schema),
    }


def op_evaluate(db: "DataFusionDB", plan_json: str, full_metrics: bool = False) -> Dict[str, Any]:
    """Execute a plan and return execution_time plus optional structural metrics."""
    start = time.perf_counter()
    results = db.execute_serialized_physical_plan(plan_json)
    execution_time = time.perf_counter() - start

    if results.is_exec_error:
        return {"error": f"Execution error: {results.error_message}"}
    if not isinstance(results.data, pa.Table):
        return {"error": f"Expected results to be a pyarrow Table, instead got {type(results.data)}."}

    composed: Dict[str, Any] = {"execution_time": execution_time}
    if full_metrics and getattr(db, "_last_executed_plan", None) is not None:
        extra = parse_plan_metrics(db._last_executed_plan)
        composed.update({k: v for k, v in extra.items() if v is not None})
    return composed


def op_validate(db: "DataFusionDB", query: str) -> Dict[str, Any]:
    """Validate a SQL query: syntax, planability, executability."""
    if not db.validate_syntax(query):
        return {
            "is_syntax_valid": False, "plan": None, "can_run": False,
            "row_count": 0, "is_empty": True, "execution_time": 0.0,
        }

    plan = db.serialize_query_to_physical_plan(query)
    if not plan:
        return {
            "is_syntax_valid": True, "plan": None, "can_run": False,
            "row_count": 0, "is_empty": True, "execution_time": 0.0,
        }

    result = db.execute_serialized_physical_plan(plan)
    can_run = not result.is_exec_error
    if can_run and result.data is not None:
        row_count = result.data.num_rows
        is_empty = row_count == 0
        execution_time = result.time
    else:
        row_count = 0
        is_empty = True
        execution_time = result.time if result else 0.0

    return {
        "is_syntax_valid": True, "plan": plan, "can_run": can_run,
        "row_count": row_count, "is_empty": is_empty, "execution_time": execution_time,
    }


def op_fetch_schema(data_folder: str, include_sample_rows: bool = False) -> Dict[str, Any]:
    """Fetch table schemas (and optional sample rows) from a folder of parquet files."""
    parquet_files = [f for f in os.listdir(data_folder) if f.endswith(".parquet")]
    table_to_schema = {}
    for parquet_file in parquet_files:
        file_path = os.path.join(data_folder, parquet_file)
        table_name = parquet_file.replace(".parquet", "")

        if include_sample_rows:
            table = pq.read_table(file_path)
            schema_list = [{"name": field.name, "type": str(field.type)} for field in table.schema]
            num_rows = table.num_rows
            sample_size = min(2, num_rows)
            if sample_size > 0:
                random_indices = sorted(random.sample(range(num_rows), sample_size))
                sample_table = table.take(random_indices)
                sample_data = sample_table.to_pydict()
            else:
                sample_data = {name: [] for name in table.schema.names}
        else:
            schema = pq.read_schema(file_path)
            schema_list = [{"name": field.name, "type": str(field.type)} for field in schema]
            sample_data = None

        table_to_schema[table_name] = {
            "schema": schema_list,
            "sample_rows": sample_data,
        }

    return {"tables": table_to_schema}
