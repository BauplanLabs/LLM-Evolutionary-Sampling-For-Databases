"""
Tests for ExecutionPlan summary serialization functionality.

The summary format provides a flattened representation of execution plans
with separate arrays for nodes and their relationships.

This test uses real parquet files from the parquet/data directory to create
realistic execution plans with actual data sources.
"""

import json
import pytest
from datafusion import SessionContext, ExecutionPlan
import os


class TestExecutionPlanSummary:
    """Test the new to_succinct_json and from_succinct_json functionality with real parquet data."""

    @pytest.fixture
    def ctx_with_tables(self):
        """Create a SessionContext with two registered parquet tables."""
        ctx = SessionContext()

        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        parquet_data_dir = os.path.join(project_root, "parquet", "data")

        # Register only two parquet files as tables
        ctx.register_parquet(
            "table_a", os.path.join(parquet_data_dir, "alltypes_plain.parquet")
        )
        ctx.register_parquet(
            "table_b", os.path.join(parquet_data_dir, "alltypes_dictionary.parquet")
        )

        return ctx


if __name__ == "__main__":
    pytest.main([__file__])
