import boto3
import json

CPU_LIMIT = 'CPU_LIMIT_HERE'

with open('/tmp/input_data.txt', 'r') as f:
    query = f.read()

try:
    db_client = DataFusionDB(data_folder=DATA_FOLDER, cpu_limit=CPU_LIMIT)
    plan = db_client.serialize_query_to_physical_plan(query)
    
    composed_data = {
        "plan": plan,
        "query": query
    }
    
except Exception as e:
    composed_data = {
        "error": str(e)
    }
    print(f"!!!!!!ERROR!!!!! {e}")


s3_client = boto3.client('s3')
bucket_name = S3_BUCKET_NAME
key = f"plan-results/{UUID}.json"

s3_client.put_object(
    Bucket=bucket_name,
    Key=key,
    Body=json.dumps(composed_data),
    ContentType='application/json'
)

print(f"UUID: {UUID}")
