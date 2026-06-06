"""Fetch dataset table schemas (and optional sample rows) and persist to S3.

Scans parquet files under ``DATA_FOLDER`` and writes either ``{"tables": ...}``
or an ``error`` payload under ``fetch_schema-results/<UUID>.json``.
"""

import boto3
import json

# Context constants overriden by modal_runner.py
INCLUDE_SAMPLE_ROWS = 'INCLUDE_SAMPLE_ROWS_HERE'
include_sample_rows = (INCLUDE_SAMPLE_ROWS == "True")

try:
    composed_data = op_fetch_schema(DATA_FOLDER, include_sample_rows=include_sample_rows)
except Exception as e:
    composed_data = {"error": str(e)}
    print(f"!!!!!!ERROR!!!!! {e}")

s3_client = boto3.client("s3")
bucket_name = S3_BUCKET_NAME
key = f"fetch_schema-results/{UUID}.json"

s3_client.put_object(
    Bucket=bucket_name,
    Key=key,
    Body=json.dumps(composed_data),
    ContentType="application/json",
)

print(f"UUID: {UUID}")
