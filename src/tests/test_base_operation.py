"""Tests for the shared operation builders in operations/base_operation.py.

``op_plan`` / ``op_execute`` / ``op_evaluate`` / ``op_validate`` /
``op_fetch_schema`` are the single source of truth shared by the Modal
operation scripts (operations/*.py) and the local runner. These tests exercise
them in-process against local TPC-H data, so they require the patched
DataFusion engine (``uv sync --extra local``) and a generated dataset.
"""

from __future__ import annotations

import os

import pytest

# Shared op builders operate on a DataFusionDB, which requires the engine.
pytest.importorskip("datafusion")

from modal_controller.db_base import DataFusionDB
from modal_controller.operations.base_operation import (
    op_plan,
    op_execute,
    op_evaluate,
    op_validate,
    op_fetch_schema,
)
from dbplanbench_utils import data_folder_for_dataset

TPCH_DATA = data_folder_for_dataset("tpch", exec_local=True, scale_factor=1)
SIMPLE_QUERY = "SELECT l_orderkey, SUM(l_quantity) FROM lineitem GROUP BY l_orderkey LIMIT 10"
JOIN_QUERY = (
    "SELECT n.n_name, COUNT(*) FROM nation n "
    "JOIN supplier s ON n.n_nationkey = s.s_nationkey GROUP BY n.n_name"
)

_needs_local_data = pytest.mark.skipif(
    not os.path.isdir(TPCH_DATA),
    reason=f"Local TPC-H data not found at {TPCH_DATA}",
)


@pytest.fixture(scope="module")
def db():
    return DataFusionDB(data_folder=TPCH_DATA, cpu_limit="4", verbose=False)


@_needs_local_data
class TestOpPlan:
    def test_valid_query_serializes(self, db):
        result = op_plan(db, SIMPLE_QUERY)
        assert result["query"] == SIMPLE_QUERY
        assert result["plan"] is not None

    def test_invalid_query_returns_none_plan(self, db):
        result = op_plan(db, "SELECT FROM nonexistent")
        assert result["plan"] is None


@_needs_local_data
class TestOpExecute:
    def test_execute_returns_payload(self, db):
        plan = op_plan(db, SIMPLE_QUERY)["plan"]
        result = op_execute(db, plan)
        assert "error" not in result
        assert result["execution_time"] > 0
        assert "result_data" in result
        assert "schema" in result

    def test_execute_invalid_plan_errors(self, db):
        result = op_execute(db, '{"invalid": true}')
        assert "error" in result


@_needs_local_data
class TestOpEvaluate:
    def test_evaluate_time_only_without_full_metrics(self, db):
        plan = op_plan(db, JOIN_QUERY)["plan"]
        result = op_evaluate(db, plan, full_metrics=False)
        assert "error" not in result
        assert result["execution_time"] > 0
        assert "result_data" not in result
        # No structural metrics requested.
        assert "build_mem_used_sum" not in result

    def test_evaluate_full_metrics_include_structural(self, db):
        plan = op_plan(db, JOIN_QUERY)["plan"]
        result = op_evaluate(db, plan, full_metrics=True)
        assert "error" not in result
        # parse_plan_metrics keys for a join query.
        for key in ("bytes_scanned", "output_rows_sum", "build_mem_used_sum", "build_mem_used_max"):
            assert key in result, f"missing structural metric {key}"


@_needs_local_data
class TestOpValidate:
    def test_valid_query(self, db):
        result = op_validate(db, SIMPLE_QUERY)
        assert result["is_syntax_valid"] is True
        assert result["plan"] is not None
        assert result["can_run"] is True
        assert result["row_count"] > 0

    def test_invalid_syntax(self, db):
        result = op_validate(db, "SELECTT BADQUERY")
        assert result["is_syntax_valid"] is False
        assert result["can_run"] is False


@_needs_local_data
class TestOpFetchSchema:
    def test_schema_lists_tables(self):
        result = op_fetch_schema(TPCH_DATA)
        tables = result["tables"]
        assert "lineitem" in tables and "orders" in tables
        assert isinstance(tables["lineitem"]["schema"], list)
        assert tables["lineitem"]["sample_rows"] is None

    def test_schema_with_sample_rows(self):
        result = op_fetch_schema(TPCH_DATA, include_sample_rows=True)
        nation = result["tables"]["nation"]
        assert nation["sample_rows"] is not None
        assert isinstance(nation["sample_rows"], dict)
