"""Execute a serialized physical plan and persist execution output to S3.

Reads plan JSON from ``/tmp/input_data.txt``, executes it via ``DataFusionDB``,
and writes either execution time + schema of the results + result data or an 
``error`` payload under ``execute-results/<UUID>.json``.
"""

import boto3
import json

# Context constants overriden by modal_runner.py
CPU_LIMIT = 'CPU_LIMIT_HERE'

with open('/tmp/input_data.txt', 'r') as f:
    plan_json = f.read()

try:
    db_client = DataFusionDB(data_folder=DATA_FOLDER, cpu_limit=CPU_LIMIT)
    composed_data = op_execute(db_client, plan_json)
except Exception as e:
    composed_data = {
        "error": str(e)
    }
    print(f"!!!!!!ERROR!!!!! {e}")

s3_client = boto3.client('s3')
bucket_name = S3_BUCKET_NAME
key = f"execute-results/{UUID}.json"

s3_client.put_object(
    Bucket=bucket_name,
    Key=key,
    Body=json.dumps(composed_data),
    ContentType='application/json'
)

print(f"UUID: {UUID}")
