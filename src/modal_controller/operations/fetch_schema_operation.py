import os
import pyarrow.parquet as pq
import boto3
import json
import random

# Context constants overriden by modal_runner.py
INCLUDE_SAMPLE_ROWS = 'INCLUDE_SAMPLE_ROWS_HERE'
include_sample_rows = (INCLUDE_SAMPLE_ROWS == "True")

try:
    parquet_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".parquet")]
    table_to_schema = {}
    for parquet_file in parquet_files:
        file_path = os.path.join(DATA_FOLDER, parquet_file)
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

    composed_data = {"tables": table_to_schema}
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
