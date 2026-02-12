"""Tests for the extracted helper functions in dbplanbench.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from dbplanbench_types import PatchedPlan, OptimizationResult, PlanningResult


# ---------------------------------------------------------------------------
# _build_early_exit_result
# ---------------------------------------------------------------------------

class TestBuildEarlyExitResult:
    def _call(self, tmp_path: Path, failures: List[Optional[str]], **kwargs):
        from dbplanbench import _build_early_exit_result

        defaults = dict(
            message="Validation failed",
            failures=failures,
            config_current={"dataset": "tpch"},
            result_path=tmp_path / "result.json",
            queries_list=["SELECT 1", "SELECT 2"],
            run_dir_path=tmp_path,
            verbose=False,
        )
        defaults.update(kwargs)
        return _build_early_exit_result(**defaults)

    def test_returns_optimization_result(self, tmp_path: Path):
        result = self._call(tmp_path, [None, "error"])
        assert isinstance(result, OptimizationResult)

    def test_optimization_outcome_is_none(self, tmp_path: Path):
        result = self._call(tmp_path, [None, "error"])
        assert result.optimization_outcome is None
        assert result.metadata is None

    def test_writes_result_json(self, tmp_path: Path):
        result_path = tmp_path / "result.json"
        self._call(tmp_path, ["err"], result_path=result_path)
        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert data["optimization_outcome"] is None
        assert "summary" in data

    def test_summary_includes_validation_stats(self, tmp_path: Path):
        result = self._call(tmp_path, [None, "timeout", None])
        stats = result.summary["validation_stats"]
        assert stats["n_queries"] == 3
        assert stats["n_valid"] == 2

    def test_summary_has_zero_sampled_plans(self, tmp_path: Path):
        result = self._call(tmp_path, ["err"])
        assert result.summary["run"]["n_sampled_plans"] == 0

    def test_queries_preserved(self, tmp_path: Path):
        result = self._call(tmp_path, [None], queries_list=["SELECT 42"])
        assert result.queries == ["SELECT 42"]

    def test_validation_failures_preserved(self, tmp_path: Path):
        failures = [None, "duplicate", "timeout"]
        result = self._call(tmp_path, failures)
        assert result.validation_failures == failures

    def test_run_dir_in_result(self, tmp_path: Path):
        result = self._call(tmp_path, [None])
        assert result.run_dir == str(tmp_path)

    def test_config_merged_into_run(self, tmp_path: Path):
        result = self._call(tmp_path, [None], config_current={"model": "gpt-5", "k": 3})
        assert result.summary["run"]["model"] == "gpt-5"
        assert result.summary["run"]["k"] == 3


# ---------------------------------------------------------------------------
# _plan_queries_with_dedup
# ---------------------------------------------------------------------------

class TestPlanQueriesWithDedup:
    @patch("dbplanbench.get_engine_plans")
    def test_no_duplicates(self, mock_gep):
        from dbplanbench import _plan_queries_with_dedup

        mock_gep.return_value = PlanningResult(
            plans=[{"plan": "p1"}, {"plan": "p2"}],
            errors=[None, None],
        )
        plans, failures = _plan_queries_with_dedup(
            ["SELECT 1", "SELECT 2"], "tpch", 1, 4, {}
        )
        assert plans == [{"plan": "p1"}, {"plan": "p2"}]
        assert failures == [None, None]
        mock_gep.assert_called_once()

    @patch("dbplanbench.get_engine_plans")
    def test_all_duplicates(self, mock_gep):
        from dbplanbench import _plan_queries_with_dedup

        plans, failures = _plan_queries_with_dedup(
            ["SELECT 1", "SELECT 1"], "tpch", 1, 4, {}
        )
        assert plans == [None, None]
        assert failures == ["duplicate", "duplicate"]
        mock_gep.assert_not_called()

    @patch("dbplanbench.get_engine_plans")
    def test_mixed_duplicates(self, mock_gep):
        from dbplanbench import _plan_queries_with_dedup

        mock_gep.return_value = PlanningResult(
            plans=[{"plan": "unique"}],
            errors=[None],
        )
        queries = ["SELECT 1", "SELECT 2", "SELECT 1", "SELECT 2", "SELECT 3"]
        plans, failures = _plan_queries_with_dedup(queries, "tpch", 1, 4, {})
        # SELECT 1 and SELECT 2 are duplicates (appear more than once)
        assert failures[0] == "duplicate"
        assert failures[1] == "duplicate"
        assert failures[2] == "duplicate"
        assert failures[3] == "duplicate"
        # SELECT 3 is unique
        assert failures[4] is None
        assert plans[4] == {"plan": "unique"}

    @patch("dbplanbench.get_engine_plans")
    def test_engine_error_propagated(self, mock_gep):
        from dbplanbench import _plan_queries_with_dedup

        mock_gep.return_value = PlanningResult(
            plans=[None, {"plan": "ok"}],
            errors=["parse_error", None],
        )
        plans, failures = _plan_queries_with_dedup(
            ["bad query", "SELECT 1"], "tpch", 1, 4, {}
        )
        assert plans[0] is None
        assert failures[0] == "parse_error"
        assert plans[1] == {"plan": "ok"}
        assert failures[1] is None

    @patch("dbplanbench.get_engine_plans")
    def test_empty_queries(self, mock_gep):
        from dbplanbench import _plan_queries_with_dedup

        plans, failures = _plan_queries_with_dedup([], "tpch", 1, 4, {})
        assert plans == []
        assert failures == []
        mock_gep.assert_not_called()


# ---------------------------------------------------------------------------
# _integrate_user_base_plans
# ---------------------------------------------------------------------------

class TestIntegrateUserBasePlans:
    def test_none_user_plans_passthrough(self):
        from dbplanbench import _integrate_user_base_plans

        engine = [{"plan": "e1"}, {"plan": "e2"}]
        plans, sources, failures = _integrate_user_base_plans(
            user_base_plans=None,
            engine_plans=engine,
            queries_list=["q1", "q2"],
            dataset="tpch",
            validate_kwargs={},
            max_workers=4,
            skip_validation=False,
            verbose=False,
        )
        assert plans is engine
        assert sources == ["engine", "engine"]
        assert failures is None

    def test_skip_validation_overwrites(self):
        from dbplanbench import _integrate_user_base_plans

        engine = [{"plan": "e1"}, {"plan": "e2"}]
        user = [{"plan": "u1"}, None]
        plans, sources, failures = _integrate_user_base_plans(
            user_base_plans=user,
            engine_plans=engine,
            queries_list=["q1", "q2"],
            dataset="tpch",
            validate_kwargs={},
            max_workers=4,
            skip_validation=True,
            verbose=False,
        )
        assert plans[0] == {"plan": "u1"}
        assert plans[1] == {"plan": "e2"}
        assert sources == ["custom", "engine"]
        assert failures is None

    def test_all_none_user_plans_keeps_engine(self):
        from dbplanbench import _integrate_user_base_plans

        engine = [{"plan": "e1"}]
        user = [None]
        plans, sources, failures = _integrate_user_base_plans(
            user_base_plans=user,
            engine_plans=engine,
            queries_list=["q1"],
            dataset="tpch",
            validate_kwargs={},
            max_workers=4,
            skip_validation=True,
            verbose=False,
        )
        assert plans[0] == {"plan": "e1"}
        assert sources == ["engine"]


# ---------------------------------------------------------------------------
# _validate_and_plan_queries  (resume path only — no Modal)
# ---------------------------------------------------------------------------

class TestValidateAndPlanQueriesResume:
    def test_resume_matching_queries(self, tmp_path: Path):
        from dbplanbench import _validate_and_plan_queries

        base_data = [{"query": "SELECT 1"}, {"query": "SELECT 2"}]
        plans, failures = _validate_and_plan_queries(
            queries_list=["SELECT 1", "SELECT 2"],
            dataset="tpch",
            scale_factor=1,
            max_workers=4,
            validate_kwargs={},
            skip_validation=False,
            base_sampling_data=base_data,
            run_dir_path=tmp_path,
            verbose=False,
        )
        assert plans == [None, None]
        assert failures == [None, None]

    def test_resume_mismatched_queries(self, tmp_path: Path):
        from dbplanbench import _validate_and_plan_queries

        base_data = [{"query": "SELECT 1"}]
        with pytest.raises(ValueError, match="queries do not match"):
            _validate_and_plan_queries(
                queries_list=["SELECT 2"],
                dataset="tpch",
                scale_factor=1,
                max_workers=4,
                validate_kwargs={},
                skip_validation=False,
                base_sampling_data=base_data,
                run_dir_path=tmp_path,
                verbose=False,
            )


# ---------------------------------------------------------------------------
# _process_sampling_results
# ---------------------------------------------------------------------------

def _make_sampled_plan(
    sample_id,
    is_valid=True,
    error_message=None,
    metric_value=1.0,
    sampled_patches=None,
):
    """Helper to build a sampled_plan entry for testing."""
    entry = {
        "sample_id": sample_id,
        "is_valid": is_valid,
        "sampled_patches": sampled_patches if sampled_patches is not None else [{"op": "replace"}],
    }
    if error_message:
        entry["error_message"] = error_message
        entry["is_valid"] = False
    if is_valid:
        entry["evaluation_stats"] = {"execution_time": {"min": metric_value}}
    return entry


class TestProcessSamplingResults:
    def _build_query_data(self, query, base_metric, candidates):
        """Build query_data dict with a base plan and candidate plans."""
        base = {
            "sample_id": None,
            "is_valid": True,
            "evaluation_stats": {"execution_time": {"min": base_metric}},
            "sampled_patches": [],
        }
        return {
            "id": 0,
            "query": query,
            "plan": {"op": "scan"},
            "sampled_plans": [base] + candidates,
        }

    @patch("sampling.sample_plans.get_upstream_patches", return_value=[{"op": "replace"}])
    def test_single_query_single_candidate(self, mock_gup):
        from dbplanbench import _process_sampling_results

        candidates = [_make_sampled_plan("s1", metric_value=0.5)]
        query_data = self._build_query_data("SELECT 1", base_metric=1.0, candidates=candidates)

        outcome, meta, improvements, counts, total = _process_sampling_results(
            queries_list=["SELECT 1"],
            sampling_by_id={0: query_data},
            id_indexed=True,
            query_to_entry={},
            optimization_metric="execution_time.min",
            top_k_patches=1,
        )
        assert len(outcome) == 1
        assert isinstance(outcome[0], PatchedPlan)
        assert outcome[0].base_plan == {"op": "scan"}
        assert counts["Successful Execution"] == 1
        assert total == 1

    @patch("sampling.sample_plans.get_upstream_patches", return_value=[])
    def test_improvement_ratio(self, mock_gup):
        from dbplanbench import _process_sampling_results

        candidates = [_make_sampled_plan("s1", metric_value=0.5)]
        query_data = self._build_query_data("SELECT 1", base_metric=2.0, candidates=candidates)

        outcome, meta, improvements, counts, total = _process_sampling_results(
            queries_list=["SELECT 1"],
            sampling_by_id={0: query_data},
            id_indexed=True,
            query_to_entry={},
            optimization_metric="execution_time.min",
            top_k_patches=1,
        )
        # Best candidate has metric 0.5, base is 2.0  =>  improvement = 2.0/0.5 = 4.0
        # But the base (patch=[]) also competes — base metric=2.0, candidate metric=0.5
        # Ranked: candidate (0.5) then base (2.0)  =>  chosen = candidate (top_k=1)
        # improvement_x = 2.0 / 0.5 = 4.0
        assert any(v == pytest.approx(4.0) for v in improvements)

    @patch("sampling.sample_plans.get_upstream_patches", return_value=[])
    def test_top_k_padding(self, mock_gup):
        from dbplanbench import _process_sampling_results

        # Only 1 valid candidate but top_k=3
        candidates = [_make_sampled_plan("s1", metric_value=0.5)]
        query_data = self._build_query_data("SELECT 1", base_metric=1.0, candidates=candidates)

        outcome, meta, improvements, counts, total = _process_sampling_results(
            queries_list=["SELECT 1"],
            sampling_by_id={0: query_data},
            id_indexed=True,
            query_to_entry={},
            optimization_metric="execution_time.min",
            top_k_patches=3,
        )
        # Should have 3 patch slots (padded with None)
        assert len(outcome[0].patch) == 3

    @patch("sampling.sample_plans.get_upstream_patches", return_value=[])
    def test_error_categorization(self, mock_gup):
        from dbplanbench import _process_sampling_results

        candidates = [
            _make_sampled_plan("s1", error_message="Failed to apply patches to plan: bad patch"),
            _make_sampled_plan("s2", error_message="Result set mismatch: expected X got Y"),
            _make_sampled_plan("s3", error_message="LiteLLM timeout"),
            _make_sampled_plan("s4", error_message="Some unknown server error"),
        ]
        query_data = self._build_query_data("SELECT 1", base_metric=1.0, candidates=candidates)

        outcome, meta, improvements, counts, total = _process_sampling_results(
            queries_list=["SELECT 1"],
            sampling_by_id={0: query_data},
            id_indexed=True,
            query_to_entry={},
            optimization_metric="execution_time.min",
            top_k_patches=1,
        )
        assert total == 4
        assert counts["Invalid Patch"] == 1
        assert counts["Execution Output Mismatch"] == 1
        assert counts["LLM Failure"] == 1
        assert counts["Server-side Execution Error"] == 1

    @patch("sampling.sample_plans.get_upstream_patches", return_value=[])
    def test_empty_patch_counted(self, mock_gup):
        from dbplanbench import _process_sampling_results

        candidates = [_make_sampled_plan("s1", is_valid=True, sampled_patches=[])]
        query_data = self._build_query_data("SELECT 1", base_metric=1.0, candidates=candidates)

        outcome, meta, improvements, counts, total = _process_sampling_results(
            queries_list=["SELECT 1"],
            sampling_by_id={0: query_data},
            id_indexed=True,
            query_to_entry={},
            optimization_metric="execution_time.min",
            top_k_patches=1,
        )
        assert counts["Empty Patch"] == 1
        assert counts["Successful Execution"] == 0

    @patch("sampling.sample_plans.get_upstream_patches", return_value=[])
    def test_query_to_entry_fallback(self, mock_gup):
        from dbplanbench import _process_sampling_results

        candidates = [_make_sampled_plan("s1", metric_value=0.5)]
        query_data = self._build_query_data("SELECT 1", base_metric=1.0, candidates=candidates)

        outcome, meta, improvements, counts, total = _process_sampling_results(
            queries_list=["SELECT 1"],
            sampling_by_id={},
            id_indexed=False,
            query_to_entry={"SELECT 1": query_data},
            optimization_metric="execution_time.min",
            top_k_patches=1,
        )
        assert len(outcome) == 1
        assert total == 1

    @patch("sampling.sample_plans.get_upstream_patches", return_value=[])
    def test_missing_query_raises(self, mock_gup):
        from dbplanbench import _process_sampling_results

        with pytest.raises(RuntimeError, match="Missing sampling data"):
            _process_sampling_results(
                queries_list=["SELECT 1"],
                sampling_by_id={},
                id_indexed=False,
                query_to_entry={},
                optimization_metric="execution_time.min",
                top_k_patches=1,
            )

    @patch("sampling.sample_plans.get_upstream_patches", return_value=[])
    def test_metadata_structure(self, mock_gup):
        from dbplanbench import _process_sampling_results

        candidates = [_make_sampled_plan("s1", metric_value=0.8)]
        query_data = self._build_query_data("SELECT 1", base_metric=1.0, candidates=candidates)

        outcome, meta, improvements, counts, total = _process_sampling_results(
            queries_list=["SELECT 1"],
            sampling_by_id={0: query_data},
            id_indexed=True,
            query_to_entry={},
            optimization_metric="execution_time.min",
            top_k_patches=1,
        )
        assert "improvement_x" in meta[0]
        assert "benchmark_stats" in meta[0]
        assert "base" in meta[0]["benchmark_stats"]
        assert "patch" in meta[0]["benchmark_stats"]

    @patch("sampling.sample_plans.get_upstream_patches", return_value=[])
    def test_multiple_queries(self, mock_gup):
        from dbplanbench import _process_sampling_results

        qd0 = self._build_query_data("q0", base_metric=2.0, candidates=[
            _make_sampled_plan("s1", metric_value=1.0),
        ])
        qd0["id"] = 0
        qd1 = self._build_query_data("q1", base_metric=3.0, candidates=[
            _make_sampled_plan("s2", metric_value=1.5),
        ])
        qd1["id"] = 1

        outcome, meta, improvements, counts, total = _process_sampling_results(
            queries_list=["q0", "q1"],
            sampling_by_id={0: qd0, 1: qd1},
            id_indexed=True,
            query_to_entry={},
            optimization_metric="execution_time.min",
            top_k_patches=1,
        )
        assert len(outcome) == 2
        assert len(meta) == 2
        assert total == 2
