import boto3
import json

CPU_LIMIT = 'CPU_LIMIT_HERE'

with open('/tmp/input_data.txt', 'r') as f:
    query = f.read()

try:
    db_client = DataFusionDB(data_folder=DATA_FOLDER, cpu_limit=CPU_LIMIT)

    # Check syntax validity
    is_syntax_valid = db_client.validate_syntax(query)
    
    if not is_syntax_valid:
        composed_data = {
            "is_syntax_valid": False,
            "plan": None,
            "can_run": False,
            "row_count": 0,
            "is_empty": True,
            "execution_time": 0.0
        }
    else:
        # Try to generate plan
        plan = db_client.serialize_query_to_physical_plan(query)
        
        if not plan:
            composed_data = {
                "is_syntax_valid": True,
                "plan": None,
                "can_run": False,
                "row_count": 0,
                "is_empty": True,
                "execution_time": 0.0
            }
        else:
            # Try to execute the plan to see if it can run
            result = db_client.execute_serialized_physical_plan(plan)
            can_run = not result.is_exec_error
            
            # Capture additional metadata from execution
            if can_run and result.data is not None:
                row_count = result.data.num_rows
                is_empty = row_count == 0
                execution_time = result.time
            else:
                row_count = 0
                is_empty = True
                execution_time = result.time if result else 0.0
            
            composed_data = {
                "is_syntax_valid": True,
                "plan": plan,
                "can_run": can_run,
                "row_count": row_count,
                "is_empty": is_empty,
                "execution_time": execution_time
            }

except Exception as e:
    composed_data = {
        "error": str(e)
    }
    print(f"!!!!!!ERROR!!!!! {e}")

s3_client = boto3.client('s3')
bucket_name = S3_BUCKET_NAME
key = f"validate-results/{UUID}.json"

s3_client.put_object(
    Bucket=bucket_name,
    Key=key,
    Body=json.dumps(composed_data),
    ContentType='application/json'
)

print(f"UUID: {UUID}")