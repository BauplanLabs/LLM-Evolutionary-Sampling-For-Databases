# DataFusion engine wrapper, shared by two execution transports:
#   - Modal: modal_runner.py reads this file as TEXT and concatenates it (ahead
#     of operations/base_operation.py and an operations/<op>.py script) into one
#     script run inside a sandbox. So every top-level import here must exist in
#     the Modal image (datafusion + pyarrow do), and the placeholder constants
#     below are string-replaced by modal_runner before execution.
#   - Local: local_runner.py imports this module in-process. The datafusion
#     import therefore requires the optional ``local`` extra, which is why local
#     code is only imported behind exec_local (see utils._get_local_runner).
from collections import namedtuple
import glob
from datafusion import ExecutionPlan, SessionContext, SessionConfig
import os
import time
import pyarrow as pa

## Context constants overriden by modal_runner.py
DATA_FOLDER = 'DATA_FOLDER_HERE'
UUID = 'UUID_HERE'
S3_BUCKET_NAME = 'S3_BUCKET_NAME_HERE'


QueryResult = namedtuple("QueryResult", ["data", "time", "is_exec_error", "error_message"])
"""Result of a query execution: (data: pa.Table|None, time: float, is_exec_error: bool, error_message: str|None)."""

class DataFusionDB:
    """Wrapper around DataFusion's SessionContext for executing SQL and physical plans.

    Registers parquet files from *data_folder* as tables and provides
    methods for validation, direct SQL execution, plan serialization,
    and serialized plan execution.
    """

    def __init__(self, data_folder: str, cpu_limit: str, verbose: bool = True) -> None:
        """Initialize the DB wrapper, registering parquet tables from *data_folder*."""
        self.data_folder = data_folder
        self.parquet_files = glob.glob(f"{data_folder}/*.parquet")
        self.tables = [file.split("/")[-1].split(".")[0] for file in self.parquet_files]
        if not self.tables:
            raise RuntimeError("No tables found in the data folder. Please check the path and files.")
        self.cpu_limit = cpu_limit
        self.verbose = verbose
        self.ctx = self.get_new_embedded_db()

    def get_new_embedded_db(self) -> SessionContext:
        """Create a fresh SessionContext with registered parquet tables."""
        config = SessionConfig().set("datafusion.sql_parser.dialect", "postgresql")
        
        try:
            int_cpu_limit = int(self.cpu_limit)
        except:
            int_cpu_limit = None
        if int_cpu_limit and int_cpu_limit > 0:
            config = config.set("datafusion.execution.target_partitions", str(int_cpu_limit))
        
        config = config.set("datafusion.execution.parquet.schema_force_view_types", "False")
        config = config.set("datafusion.optimizer.expand_views_at_output", "True")
        ctx = SessionContext(config)

        for table in self.tables:
            ctx.register_parquet(
                table, os.path.join(self.data_folder, f"{table}.parquet")
            )

        return ctx

    def validate_syntax(self, query: str) -> bool:
        """Return True if *query* parses and plans successfully."""
        try:
            df = self.ctx.sql(query)
            _lp = df.logical_plan()
            return True
        except Exception as e:
            if self.verbose:
                print(f"SQL parse/planning failed: {e}")

        return False

    def execute_sql_query_directly(self, query: str) -> QueryResult:
        """Execute *query* via DataFusion SQL and return a QueryResult."""
        arrow_result = None
        is_error = False
        error_message = None
        elapsed_time = 0
        try:
            start = time.perf_counter()
            arrow_result = self.ctx.sql(query).collect()
            elapsed_time = time.perf_counter() - start
            if self.verbose:
                print(f"Query executed directly in {elapsed_time:.2f} seconds")
        except Exception as e:
            error_message = f"Error executing query: {e}"
            if self.verbose:
                print(error_message)
            is_error = True

        if is_error or arrow_result is None:
            data = None
        elif len(arrow_result) == 0:
            try:
                df = self.ctx.sql(query)
                schema = df.schema()
                empty_arrays = [pa.array([], type=field.type) for field in schema]
                data = pa.Table.from_arrays(empty_arrays, schema=schema)
            except Exception as e:
                if self.verbose:
                    print(f"Error creating empty table: {e}")
                data = None
        else:
            data = pa.Table.from_batches(arrow_result)

        return QueryResult(data=data, time=elapsed_time, is_exec_error=is_error, error_message=error_message)

    def execute_sql_query_through_serialization(self, query: str) -> QueryResult:
        """Serialize *query* to a physical plan and execute the plan."""
        plan = self.serialize_query_to_physical_plan(query)
        if not plan:
            error_message = "❌ Failed to serialize query to physical plan."
            if self.verbose:
                print(error_message)
            return QueryResult(data=None, time=1000, is_exec_error=True, error_message=error_message)

        return self.execute_serialized_physical_plan(plan)
        
    def serialize_query_to_physical_plan(self, query: str) -> str | None:
        """Return the succinct JSON representation of *query*'s physical plan, or None."""
        try:
            return self.ctx.sql(query).execution_plan().to_succinct_json()
        except Exception as e:
            if self.verbose:
                print(f"Error serializing query to physical plan: {e}")

        return None

    def execute_serialized_physical_plan(self, plan_as_json_string: str) -> QueryResult:
        """Deserialize and execute a succinct JSON physical plan."""
        result = []
        elapsed_time = 0.0
        is_error = False
        error_message = None
        self._last_executed_plan = None
        try:
            start = time.perf_counter()
            plan = ExecutionPlan.from_succinct_json(self.ctx, plan_as_json_string)
            result = self.ctx.collect(plan)
            self._last_executed_plan = plan
            elapsed_time = time.perf_counter() - start
            if self.verbose:
                print(f"Query executed through serialization in {elapsed_time:.2f} seconds")
        except Exception as e:
            error_message = f"Error executing query: {e}"
            if self.verbose:
                print(error_message)
            is_error = True
            
        if is_error or result is None:
            data = None
        elif len(result) == 0:
            data = pa.Table.from_arrays([])
        else:
            data = pa.Table.from_batches(result)

        return QueryResult(data=data, time=elapsed_time, is_exec_error=is_error, error_message=error_message)
