import boto3
import json

# Context constants overriden by modal_runner.py
CPU_LIMIT = 'CPU_LIMIT_HERE'

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
        "result_data": results.data.to_pandas().to_json(), 
        "schema": str(results.data.schema),
        "plan_size_chars": len(plan_json)
    }

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
