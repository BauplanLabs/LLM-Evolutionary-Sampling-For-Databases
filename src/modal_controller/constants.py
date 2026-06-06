import os
from pathlib import Path

# Modal sandbox name
SANDBOX_NAME = "modal-faster-dbs-with-llms"

# AWS S3 bucket name for storing results (must be set via environment variable)
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "")
AWS_SECRET_NAME = 's3-aws-credentials'

# Default scale factor for data generation / Modal image build.
DEFAULT_SCALE_FACTOR = 3

# Local data directory for exec_local mode — where generate_benchmark_data.py
# writes parquet tables (<repo_root>/data).
LOCAL_DATA_DIR = str(Path(__file__).resolve().parents[2] / "data")

# Recommended error keyword portions for retrying operations
RETRY_DEFAULT_ERROR_KWS = [
    "Status.RESOURCE_EXHAUSTED",
    "Status.UNAVAILABLE",
    "Protocol error",
    "No UUID found in output.", # if modal fails
    "Error retrieving result from S3 for UUID", # if s3 or modal (any) fails
    "internal server error",
    "Failed to convert from succinct JSON:",
]

# Special cases where only one retry is given
RETRY_ONLY_ONCE = [ # special errors observed sometimes due to server-side issues, giving only one retry may help resolving
    "Failed to convert from succinct JSON:",
]

# Mapping of known error keywords to user-friendly messages
DEFAULT_ERROR_KWS_TO_MESSAGES = {
    "Status.RESOURCE_EXHAUSTED": "Error in our execution platform (Modal/S3): Status.RESOURCE_EXHAUSTED.",
    "Status.UNAVAILABLE": "Error in our execution platform (Modal/S3): Status.UNAVAILABLE.",
    "Protocol error": "Error in our execution platform (Modal/S3): Protocol error.",
    "No UUID found in output.": "Execution or result-saving failed/terminated unexpectedly in our execution platform (Modal/S3), so no usable output was produced.",
    "Error retrieving result from S3 for UUID": "Execution finished, but the saved output could not be retrieved from storage (S3).",
    "internal server error": "Error in our execution platform (Modal/S3): internal server error.",
}

# How many sandbox creation requests per second to allow
RATE_LIMIT_PER_SEC = 4

# How many concurrent sandboxes to allow
MAX_CONCURRENT_SANDBOXES = 100
