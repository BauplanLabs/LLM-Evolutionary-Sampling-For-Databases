import boto3
import json
import re

CPU_LIMIT = 'CPU_LIMIT_HERE'
FULL_METRICS = 'FULL_METRICS_HERE' == 'True'


def _parse_duration_to_seconds(val):
    val = val.strip()
    if val.endswith("ns"):
        return float(val[:-2]) * 1e-9
    if val.endswith("\u00b5s"):
        return float(val[:-2]) * 1e-6
    if val.endswith("ms"):
        return float(val[:-2]) * 1e-3
    if val.endswith("s"):
        return float(val[:-1])
    return float(val)


def _parse_int(val):
    return int(val.replace(",", "").strip())


def parse_plan_metrics(plan):
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

    def is_join(text):
        return bool(re.search(r"join", text, re.IGNORECASE))

    def safe_sum_int(key, join_only=False):
        try:
            return sum(
                _parse_int(p["metrics"][key])
                for p in parsed
                if key in p["metrics"] and (not join_only or is_join(p["text"]))
            )
        except Exception:
            return None

    def safe_sum_duration(key, join_only=False):
        try:
            return sum(
                _parse_duration_to_seconds(p["metrics"][key])
                for p in parsed
                if key in p["metrics"] and (not join_only or is_join(p["text"]))
            )
        except Exception:
            return None

    def safe_max_int(key, join_only=False):
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


with open('/tmp/input_data.txt', 'r') as f:
    plan_json = f.read()

try:
    print(f"Executing plan ({len(plan_json)} characters)...")

    db_client = DataFusionDB(data_folder=DATA_FOLDER, cpu_limit=CPU_LIMIT)
    # Execute plan (timed)
    start = time.perf_counter()
    results = db_client.execute_serialized_physical_plan(plan_json)
    execution_time = time.perf_counter() - start
    print(f"Plan executed in {execution_time:.4f} seconds")

    if results.is_exec_error:
        raise Exception(f"Execution error: {results.error_message}")

    assert type(results.data) is pa.Table, f"Expected results to be a pyarrow Table, instead got {type(results.data)}."

    composed_data = {
        "execution_time": execution_time,
    }

    if FULL_METRICS and hasattr(db_client, '_last_executed_plan') and db_client._last_executed_plan is not None:
        extra = parse_plan_metrics(db_client._last_executed_plan)
        composed_data.update({k: v for k, v in extra.items() if v is not None})

except Exception as e:
    composed_data = {
        "error": str(e)
    }
    print(f"!!!!!!ERROR!!!!! {e}")

s3_client = boto3.client('s3')
bucket_name = S3_BUCKET_NAME
key = f"evaluation-results/{UUID}.json"

s3_client.put_object(
    Bucket=bucket_name,
    Key=key,
    Body=json.dumps(composed_data),
    ContentType='application/json'
)

print(f"UUID: {UUID}")
