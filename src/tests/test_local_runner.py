"""Tests for LocalRunner — local DataFusion execution without Modal/S3."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Local execution requires the (patched) DataFusion engine. Skip the whole
# module when it isn't installed — local runs aren't possible without it.
pytest.importorskip("datafusion")

from modal_controller.local_runner import LocalRunner
from modal_controller.modal_runner import Operation
from dbplanbench_utils import data_folder_for_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Local tests run at scale factor 1 (folders are scale-keyed: data_<dataset>_sf<N>).
TPCH_DATA = data_folder_for_dataset("tpch", exec_local=True, scale_factor=1)
TPCDS_DATA = data_folder_for_dataset("tpcds", exec_local=True, scale_factor=1)
SIMPLE_QUERY = "SELECT l_orderkey, SUM(l_quantity) FROM lineitem GROUP BY l_orderkey LIMIT 10"

_needs_local_data = pytest.mark.skipif(
    not os.path.isdir(TPCH_DATA),
    reason=f"Local TPC-H data not found at {TPCH_DATA}",
)


@pytest.fixture(scope="module")
def runner():
    return LocalRunner()


@pytest.fixture(scope="module")
def tpch_plan(runner):
    """Get an engine plan for a simple TPC-H query."""
    result = runner.run_operation(Operation.PLAN, SIMPLE_QUERY, TPCH_DATA)
    assert "error" not in result, f"Plan failed: {result.get('error')}"
    assert result["plan"] is not None
    return result["plan"]


# ---------------------------------------------------------------------------
# LocalRunner operations
# ---------------------------------------------------------------------------

@_needs_local_data
class TestLocalRunnerPlan:
    def test_plan_returns_plan_and_query(self, runner):
        result = runner.run_operation(Operation.PLAN, SIMPLE_QUERY, TPCH_DATA)
        assert "plan" in result
        assert "query" in result
        assert result["query"] == SIMPLE_QUERY
        assert result["plan"] is not None
        # Plan should be valid JSON
        plan_dict = json.loads(result["plan"])
        assert "structure" in plan_dict or isinstance(plan_dict, dict)

    def test_plan_invalid_query_returns_none_plan(self, runner):
        result = runner.run_operation(Operation.PLAN, "SELECT FROM nonexistent", TPCH_DATA)
        # Should not crash, plan may be None
        assert "plan" in result


@_needs_local_data
class TestLocalRunnerExecute:
    def test_execute_returns_result_data(self, runner, tpch_plan):
        result = runner.run_operation(Operation.EXECUTE, tpch_plan, TPCH_DATA)
        assert "error" not in result, f"Execute failed: {result.get('error')}"
        assert "execution_time" in result
        assert "result_data" in result
        assert "schema" in result
        assert result["execution_time"] > 0

    def test_execute_invalid_plan_returns_error(self, runner):
        result = runner.run_operation(Operation.EXECUTE, '{"invalid": true}', TPCH_DATA)
        assert "error" in result


@_needs_local_data
class TestLocalRunnerEvaluate:
    def test_evaluate_returns_execution_time(self, runner, tpch_plan):
        result = runner.run_operation(Operation.EVALUATE, tpch_plan, TPCH_DATA)
        assert "error" not in result, f"Evaluate failed: {result.get('error')}"
        assert "execution_time" in result
        assert result["execution_time"] > 0
        # Without full_metrics, should still have execution_time
        assert "result_data" not in result  # evaluate doesn't return result data

    def test_evaluate_with_full_metrics(self, runner, tpch_plan):
        result = runner.run_operation(
            Operation.EVALUATE, tpch_plan, TPCH_DATA,
            sandbox_placeholders={"FULL_METRICS": True},
        )
        assert "error" not in result, f"Evaluate failed: {result.get('error')}"
        assert "execution_time" in result
        # Full metrics should include structural metrics
        metric_keys = {"bytes_scanned", "output_rows_sum", "build_mem_used_sum", "build_mem_used_max"}
        assert metric_keys.issubset(result.keys()), f"Missing metrics: {metric_keys - set(result.keys())}"


@_needs_local_data
class TestLocalRunnerValidate:
    def test_validate_valid_query(self, runner):
        result = runner.run_operation(Operation.VALIDATE, SIMPLE_QUERY, TPCH_DATA)
        assert "error" not in result
        assert result["is_syntax_valid"] is True
        assert result["plan"] is not None
        assert result["can_run"] is True
        assert result["row_count"] > 0

    def test_validate_invalid_syntax(self, runner):
        result = runner.run_operation(Operation.VALIDATE, "SELECTT BADQUERY", TPCH_DATA)
        assert result["is_syntax_valid"] is False
        assert result["can_run"] is False


@_needs_local_data
class TestLocalRunnerFetchSchema:
    def test_fetch_schema_returns_tables(self, runner):
        result = runner.run_operation(Operation.FETCH_SCHEMA, "", TPCH_DATA)
        assert "tables" in result
        tables = result["tables"]
        assert "lineitem" in tables
        assert "orders" in tables
        assert "schema" in tables["lineitem"]
        assert isinstance(tables["lineitem"]["schema"], list)
        assert len(tables["lineitem"]["schema"]) > 0


@_needs_local_data
class TestLocalRunnerPlaceholderFlags:
    """LocalRunner reads full_metrics / include_sample_rows from the Modal-style
    sandbox_placeholders dict (the local analog of the operation scripts), so the
    pipeline can forward kwargs to both runners uniformly."""

    def test_evaluate_full_metrics_via_placeholders(self, runner, tpch_plan):
        result = runner.run_operation(
            Operation.EVALUATE, tpch_plan, TPCH_DATA,
            sandbox_placeholders={"FULL_METRICS": True},
        )
        assert "error" not in result
        assert "build_mem_used_sum" in result  # structural metrics collected

    def test_evaluate_no_placeholder_skips_metrics(self, runner, tpch_plan):
        result = runner.run_operation(Operation.EVALUATE, tpch_plan, TPCH_DATA)
        assert "build_mem_used_sum" not in result

    def test_fetch_schema_sample_rows_via_lowercase_placeholder(self, runner):
        # query_gen sets the key lowercase with a string "True".
        result = runner.run_operation(
            Operation.FETCH_SCHEMA, "", TPCH_DATA,
            sandbox_placeholders={"include_sample_rows": "True"},
        )
        assert result["tables"]["nation"]["sample_rows"] is not None


# ---------------------------------------------------------------------------
# Integration: submit_run_operation with exec_local
# ---------------------------------------------------------------------------

@_needs_local_data
class TestSubmitRunOperationLocal:
    def test_plan_via_submit(self):
        from modal_controller.utils import submit_run_operation

        result = submit_run_operation(
            Operation.PLAN,
            SIMPLE_QUERY,
            TPCH_DATA,
            exec_local=True,
        )
        assert "error" not in result
        assert result["plan"] is not None

    def test_evaluate_via_submit(self):
        from modal_controller.utils import submit_run_operation

        # First get a plan
        plan_result = submit_run_operation(
            Operation.PLAN, SIMPLE_QUERY, TPCH_DATA, exec_local=True,
        )
        assert plan_result["plan"] is not None

        # Then evaluate it
        eval_result = submit_run_operation(
            Operation.EVALUATE,
            plan_result["plan"],
            TPCH_DATA,
            exec_local=True,
        )
        assert "error" not in eval_result
        assert "execution_time" in eval_result


# ---------------------------------------------------------------------------
# Integration: validate_plan_result_set with exec_local
# ---------------------------------------------------------------------------

@_needs_local_data
class TestValidatePlanResultSetLocal:
    def test_identical_plans_pass_validation(self):
        from modal_controller.utils import submit_run_operation, validate_plan_result_set

        plan_result = submit_run_operation(
            Operation.PLAN, SIMPLE_QUERY, TPCH_DATA, exec_local=True,
        )
        plan_json = plan_result["plan"]

        error = validate_plan_result_set(
            candidate_plan_str=plan_json,
            baseline_plan_str=plan_json,
            data_folder=TPCH_DATA,
            n_retry=1,
            n_determinism_retries=2,
            exec_local=True,
        )
        assert error is None, f"Validation failed: {error}"


# ---------------------------------------------------------------------------
# Integration: get_full_metrics flows to local op_evaluate via the public API.
# (get_full_metrics is encoded as a Modal sandbox_placeholder; the local path
# must translate it to LocalRunner's full_metrics param.)
# ---------------------------------------------------------------------------

_JOIN_QUERY = (
    "SELECT n.n_name, COUNT(*) FROM nation n "
    "JOIN supplier s ON n.n_nationkey = s.s_nationkey GROUP BY n.n_name"
)


@_needs_local_data
class TestLocalFullMetricsPipeline:
    def test_benchmark_plans_local_collects_structural_metrics(self):
        from dbplanbench import get_engine_plans, benchmark_plans
        from dbplanbench_types import PatchedPlan

        planning = get_engine_plans(
            [_JOIN_QUERY], dataset="tpch", scale_factor=1, exec_local=True, verbose=False
        )
        plan = planning.plans[0]
        assert plan is not None, f"Planning failed: {planning.errors[0]}"

        result = benchmark_plans(
            [PatchedPlan(base_plan=plan, patch=[[]])],
            dataset="tpch",
            scale_factor=1,
            n_runs=2,
            get_full_metrics=True,
            exec_local=True,
            verbose=False,
        )
        stats = result.results[0][0].get("benchmark_stats", {})
        # Structural metrics must be present (not just execution_time) — this is
        # the point of get_full_metrics and the main local-execution use case.
        for key in ("build_mem_used_sum", "build_mem_used_max", "output_rows_sum"):
            assert key in stats, f"missing structural metric {key}; got {sorted(stats)}"


# ---------------------------------------------------------------------------
# Integration: validate_queries honors exec_local (the VALIDATE path that
# query generation also uses for validating generated queries).
# ---------------------------------------------------------------------------

@_needs_local_data
class TestValidateQueriesLocal:
    def test_valid_query_validates_locally(self):
        from dbplanbench import validate_queries

        result = validate_queries(
            [SIMPLE_QUERY], dataset="tpch", scale_factor=1,
            n_determinism_retries=2, exec_local=True, verbose=False,
        )
        assert result.errors[0] is None, f"Local validation errored: {result.errors[0]}"
        assert result.plans[0] is not None

    def test_invalid_query_reports_error_locally(self):
        from dbplanbench import validate_queries

        result = validate_queries(
            ["SELECT * FROM nonexistent_table_xyz"], dataset="tpch", scale_factor=1,
            n_determinism_retries=2, exec_local=True, verbose=False,
        )
        assert result.plans[0] is None


@_needs_local_data
class TestBenchmarkQueriesLocal:
    def test_benchmark_queries_local(self):
        from dbplanbench import benchmark_queries

        result = benchmark_queries(
            [SIMPLE_QUERY], dataset="tpch", scale_factor=1, n_runs=2,
            exec_local=True, verbose=False,
        )
        stats = result.results[0][0].get("benchmark_stats", {})
        assert "execution_time" in stats, f"got: {result.results[0][0]}"


# ---------------------------------------------------------------------------
# Heavy: real local data generation (DuckDB), no Modal and no LLM. These write
# TPC-H data to disk into a temp LOCAL_DATA_DIR and are slow, so they carry the
# `heavy` marker and are skipped by the default suite (run with `-m heavy`).
# ---------------------------------------------------------------------------

@pytest.mark.heavy
class TestMultiScaleLocalDataIsolation:
    def test_distinct_scales_generate_distinct_data(self, tmp_path, monkeypatch):
        """Generating tpch at sf=1 then sf=2 must produce two distinct on-disk
        folders, and the larger scale must hold strictly more lineitem rows.

        Regression test for the silent multi-scale reuse bug: before folders were
        scale-keyed, the sf=2 request would have found the existing sf=1 folder
        and silently returned wrong-sized data.
        """
        duckdb = pytest.importorskip("duckdb")
        from dbplanbench_utils import ensure_local_data

        monkeypatch.setattr("modal_controller.constants.LOCAL_DATA_DIR", str(tmp_path))

        folder_sf1 = ensure_local_data("tpch", 1)
        folder_sf2 = ensure_local_data("tpch", 2)

        assert folder_sf1 == str(tmp_path / "data_tpch_sf1")
        assert folder_sf2 == str(tmp_path / "data_tpch_sf2")
        assert os.path.isdir(folder_sf1) and os.path.isdir(folder_sf2)

        def lineitem_rows(folder: str) -> int:
            path = os.path.join(folder, "lineitem.parquet")
            return duckdb.sql(f"SELECT COUNT(*) FROM '{path}'").fetchone()[0]

        rows_sf1 = lineitem_rows(folder_sf1)
        rows_sf2 = lineitem_rows(folder_sf2)
        assert rows_sf1 > 0
        # sf=2 holds ~2x the data of sf=1; at minimum it must be strictly larger.
        assert rows_sf2 > rows_sf1, f"sf=2 ({rows_sf2}) not larger than sf=1 ({rows_sf1})"


@pytest.mark.heavy
class TestFullLocalPipeline:
    def test_benchmark_queries_auto_generates_and_benchmarks(self, tmp_path, monkeypatch):
        """End-to-end local path with no Modal and no LLM: benchmark_queries at a
        fresh scale_factor auto-generates the data, plans each query, and
        evaluates it locally — returning execution times, not errors.
        """
        pytest.importorskip("duckdb")
        monkeypatch.setattr("modal_controller.constants.LOCAL_DATA_DIR", str(tmp_path))
        from dbplanbench import benchmark_queries

        queries = [
            "SELECT l_returnflag, COUNT(*) AS c FROM lineitem "
            "GROUP BY l_returnflag ORDER BY l_returnflag",
            "SELECT COUNT(*) FROM orders WHERE o_orderstatus = 'F'",
        ]
        result = benchmark_queries(
            queries, dataset="tpch", scale_factor=2, n_runs=2,
            exec_local=True, verbose=False,
        )

        # Data was auto-generated once at the scale-keyed folder for sf=2.
        assert os.path.isdir(tmp_path / "data_tpch_sf2")

        assert len(result.results) == len(queries)
        for i, entry in enumerate(result.results):
            assert len(entry) == 1, f"query {i}: expected one result entry, got {entry}"
            stats = entry[0]
            assert "error" not in stats, f"query {i} failed: {stats}"
            assert "execution_time" in stats.get("benchmark_stats", {}), stats
