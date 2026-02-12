import uuid
import modal
import os
from pathlib import Path
from enum import Enum
from typing import Optional
from modal.stream_type import StreamType
from modal_controller.constants import S3_BUCKET_NAME, AWS_SECRET_NAME, DEFAULT_SCALE_FACTOR

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parents[1]

class Operation(Enum):
    """Modal sandbox operation types, each mapping to a Python script."""
    EVALUATE = "operations/evaluate_operation.py"
    EXECUTE = "operations/execute_operation.py"
    PLAN = "operations/plan_operation.py"
    VALIDATE = "operations/validate_operation.py"
    FETCH_SCHEMA = "operations/fetch_schema_operation.py"

class ModalRunner:
    """Manages a Modal sandbox for running DataFusion operations.

    Builds a container image with the patched DataFusion engine and
    TPC-H/TPC-DS data, then executes operations by concatenating
    ``db_base.py`` with an operation script and running them in a sandbox.
    """

    def __init__(self, app_name: str, scale_factor: int = DEFAULT_SCALE_FACTOR, rebuild_image: bool = False):
        """Initialize a ModalRunner, looking up the Modal app and building the container image."""
        self.scale_factor = scale_factor
        self.app = modal.App.lookup(app_name, create_if_missing=True)
        self.image = self._get_base_image(rebuild_image)
    
    def _get_base_image(self, rebuild_image: bool = False):
        """Build the Modal container image with DataFusion, dependencies, and TPC data."""
        datafusion_local = (_REPO_ROOT / "datafusion_patched").resolve()
        gen_tpch_file_local = _MODULE_DIR / "generate_tpch_files.py"
        return (modal.Image.debian_slim(force_build=rebuild_image)
                .apt_install("build-essential", force_build=rebuild_image)
                .apt_install("protobuf-compiler", force_build=rebuild_image)
                .apt_install("curl", force_build=rebuild_image)
                .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y", force_build=rebuild_image)
                .add_local_dir(
                    local_path=str(datafusion_local),
                    remote_path="/app/datafusion",
                    copy=True, 
                    ignore=[".git", "**/__pycache__", ".venv", "target"],
                )
                .run_commands(
                    "bash -lc 'source $HOME/.cargo/env && pip install -e /app/datafusion'",
                    force_build=rebuild_image,
                )
                .pip_install('boto3', 'pyarrow', 'duckdb', 'pandas', force_build=rebuild_image)
                .add_local_file(
                    local_path=str(gen_tpch_file_local),
                    remote_path="/app/generate_tpch_files.py", 
                    copy=True
                )
                .run_commands(f"python /app/generate_tpch_files.py --scale-factor {self.scale_factor} --benchmark both --data-dir /tmp/data", force_build=rebuild_image)
        )
    
    def _load_file(self, filepath: str) -> str:
        """Read and return the text content of a local file."""
        with open(filepath, 'r') as f:
            return f.read()
    
    def run_operation(self, operation: Operation, input_str: str, data_folder: str = '/tmp/data/data_tpch',
            cpu: tuple = (4, 4), memory: tuple = (4 * 1024, 4 * 1024),
            env: Optional[dict] = None, sandbox_kwargs: Optional[dict] = None,
            sandbox_placeholders: Optional[dict] = None, modal_output: bool = False,
            **kwargs) -> Optional[str]:
        """Execute an operation in a Modal sandbox and return the result UUID.

        Concatenates ``db_base.py`` with the operation script, injects
        placeholders (data folder, UUID, S3 bucket), writes *input_str*
        to the sandbox, and runs the code. Results are stored in S3
        under the returned UUID.

        Returns:
            UUID string for retrieving the result from S3, or None on failure.
        """
        
        if not S3_BUCKET_NAME:
            raise EnvironmentError(
                "S3_BUCKET_NAME environment variable is not set. "
                "Copy local.env to .env and set your S3 bucket name. "
                "See README for setup instructions."
            )

        operation_file = operation.value
        
        # Load base and operation files
        base_code = self._load_file(os.path.join(os.path.dirname(__file__), "db_base.py"))
        operation_code = self._load_file(os.path.join(os.path.dirname(__file__), operation_file))
        
        # Concatenate code (no more embedding large input_str)
        full_code = base_code + "\n\n" + operation_code

        # Derive CPU count from allocation (use max value from the tuple)
        cpu_count = str(cpu[1]) if isinstance(cpu, tuple) else str(cpu)

        # Default CPU_LIMIT placeholder and RAYON_NUM_THREADS env var
        # from the cpu allocation, unless the caller overrides them.
        placeholders = dict(sandbox_placeholders or {})
        placeholders.setdefault("CPU_LIMIT", cpu_count)
        env_dict = dict(env or {})
        env_dict.setdefault("RAYON_NUM_THREADS", cpu_count)

        # Override placeholders
        for key, value in placeholders.items():
            placeholder = f"{key.upper()}_HERE"
            full_code = full_code.replace(placeholder, str(value))

        full_code = full_code.replace("DATA_FOLDER_HERE", data_folder)
        
        return_uuid = str(uuid.uuid4())
        full_code = full_code.replace("UUID_HERE", return_uuid)
        
        full_code = full_code.replace("S3_BUCKET_NAME_HERE", S3_BUCKET_NAME)
        
        # Execute in Modal
        sb = modal.Sandbox.create(
            image=self.image,
            app=self.app,
            cpu=cpu,
            memory=memory,
            timeout=kwargs.get("sandbox_timeout", 120), # lets not keep it too long if hangs
            env=env_dict,
            secrets=[modal.Secret.from_name(AWS_SECRET_NAME)],
            **(sandbox_kwargs or {}),
        )
        try:
            # Write input data to file in sandbox using Modal's file API
            with sb.open("/tmp/input_data.txt", "w") as f:
                f.write(input_str)
            
            # Execute the operation code
            stream_type = StreamType.PIPE if modal_output else StreamType.DEVNULL
            p = sb.exec(
                "python", "-c", full_code, 
                timeout=kwargs.get("timeout", 45),
                stdout=stream_type,
                stderr=stream_type,
            )
            
            rc = p.wait()
            if modal_output:
                stdout_text = "" if not p.stdout else p.stdout.read()
                stderr_text = "" if not p.stderr else p.stderr.read()
                if isinstance(stdout_text, bytes):
                    stdout_text = stdout_text.decode("utf-8", errors="replace")
                if isinstance(stderr_text, bytes):
                    stderr_text = stderr_text.decode("utf-8", errors="replace")
                print("== MODAL_SANDBOX_OUTPUT_BEGIN ==")
                if stdout_text:
                    print(stdout_text, end="" if stdout_text.endswith("\n") else "\n")
                if stderr_text:
                    print("==== MODAL_SANDBOX_OUTPUT_STDERR_BEGIN ====")
                    print(stderr_text, end="" if stderr_text.endswith("\n") else "\n")
                    print("==== MODAL_SANDBOX_OUTPUT_STDERR_END ====")
                print("== MODAL_SANDBOX_OUTPUT_END ==")
            if rc != 0:
                return None
            return return_uuid
        finally:
            sb.terminate()
