"""Local execution runner — runs DataFusion operations in-process (no Modal/S3).

This is the local sibling of the Modal operation scripts (``operations/*.py``).
Both transports execute the same shared ``op_*`` builders from :mod:`db_base`;
the only difference is that the Modal scripts read their input from a file and
persist the result to S3, while ``LocalRunner`` takes the input as an argument
and returns the result dict in-process.
"""

from typing import Optional

from modal_controller.modal_runner import Operation
from modal_controller.db_base import DataFusionDB
from modal_controller.operations.base_operation import (
    op_execute,
    op_evaluate,
    op_plan,
    op_validate,
    op_fetch_schema,
)


class LocalRunner:
    """Runs DataFusion operations in-process, replacing the Modal sandbox + S3.

    One shared instance suffices: like ``ModalRunner`` takes ``cpu`` per
    ``run_operation`` call (and derives the sandbox CPU from it), ``LocalRunner``
    derives DataFusion's ``target_partitions`` from the per-call ``cpu`` and
    holds no run state of its own. Each call builds a fresh ``DataFusionDB``
    (its own ``SessionContext``), so the instance is safe to share across
    threads.
    """

    def run_operation(
        self,
        operation: Operation,
        input_str: str,
        data_folder: str,
        cpu: tuple = (4, 4),
        sandbox_placeholders: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """Execute an operation locally and return the same result dict the
        corresponding Modal operation script would have persisted to S3.

        Mirrors ``ModalRunner.run_operation``: ``cpu`` sets DataFusion
        parallelism (derived from the second tuple element, like the Modal CPU
        count), and the ``full_metrics`` / ``include_sample_rows`` flags are read
        from *sandbox_placeholders* — the same dict the Modal path injects into
        the operation script as ``FULL_METRICS_HERE`` / ``INCLUDE_SAMPLE_ROWS_HERE``.
        Resolving the placeholders here is the local analog of the Modal script's
        ``== 'True'`` check, so callers forward kwargs to both runners uniformly.
        Modal-only kwargs (memory, env, sandbox_*, ...) are accepted and ignored.
        """
        cpu_limit = str(cpu[1] if isinstance(cpu, tuple) else cpu)
        flags = {str(k).upper(): v for k, v in (sandbox_placeholders or {}).items()}
        full_metrics = str(flags.get("FULL_METRICS")) == "True"
        include_sample_rows = str(flags.get("INCLUDE_SAMPLE_ROWS")) == "True"

        def db() -> DataFusionDB:
            return DataFusionDB(data_folder=data_folder, cpu_limit=cpu_limit, verbose=False)

        try:
            if operation == Operation.PLAN:
                return op_plan(db(), input_str)
            if operation == Operation.EXECUTE:
                return op_execute(db(), input_str)
            if operation == Operation.EVALUATE:
                return op_evaluate(db(), input_str, full_metrics=full_metrics)
            if operation == Operation.VALIDATE:
                return op_validate(db(), input_str)
            if operation == Operation.FETCH_SCHEMA:
                return op_fetch_schema(data_folder, include_sample_rows=include_sample_rows)
            return {"error": f"Unsupported operation: {operation}"}
        except Exception as e:
            return {"error": str(e)}
