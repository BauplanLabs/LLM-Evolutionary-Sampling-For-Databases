"""Full-pipeline tests for the public API.

Tests that hit real Modal sandboxes are marked ``@modal`` (skipped by default,
and they cost money); the remaining tests are mocked and run in the default
fast suite.

Run the real-Modal tests:  pytest -m modal -v
Run the fast suite only:   pytest -m "not modal and not heavy" -v
Run everything:            pytest -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from dbplanbench import get_engine_plans, benchmark_plans, benchmark_queries, validate_queries
from dbplanbench_types import PatchedPlan, PlanningResult, BenchmarkResult, QueryValidationResult
from modal_controller.utils import METRIC_STAT_KEYS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET = "tpch"
SCALE_FACTOR = 1

# Tiny queries — nation (25 rows) and region (5 rows) are the smallest TPC-H tables.
SIMPLE_QUERY = "SELECT n_nationkey, n_name FROM nation LIMIT 5"
FILTER_QUERY = "SELECT n_nationkey, n_name FROM nation WHERE n_regionkey = 1"
JOIN_QUERY = (
    "SELECT n.n_name, r.r_name "
    "FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey"
)
AGG_QUERY = "SELECT n_regionkey, COUNT(*) AS cnt FROM nation GROUP BY n_regionkey"
INVALID_QUERY = "SELECT * FROM nonexistent_table_xyz_abc"

# All Modal-hitting tests below are tagged with this; mocked tests are left
# unmarked so they run in the default fast suite.
modal = pytest.mark.modal


# ---------------------------------------------------------------------------
# Module-scoped fixtures (avoid re-planning the same query per test run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def simple_plan():
    result = get_engine_plans(
        [SIMPLE_QUERY], dataset=DATASET, scale_factor=SCALE_FACTOR, verbose=False,
    )
    assert result.plans[0] is not None, f"Planning failed: {result.errors[0]}"
    return result.plans[0]


@pytest.fixture(scope="module")
def join_plan():
    result = get_engine_plans(
        [JOIN_QUERY], dataset=DATASET, scale_factor=SCALE_FACTOR, verbose=False,
    )
    assert result.plans[0] is not None, f"Planning failed: {result.errors[0]}"
    return result.plans[0]


# ===================================================================
# get_engine_plans
# ===================================================================

class TestGetEnginePlansIntegration:
    @modal
    def test_single_valid_query(self):
        result = get_engine_plans(
            [SIMPLE_QUERY], dataset=DATASET, scale_factor=SCALE_FACTOR, verbose=False,
        )
        assert isinstance(result, PlanningResult)
        assert len(result.plans) == 1
        assert len(result.errors) == 1
        assert result.plans[0] is not None
        assert result.errors[0] is None

    @modal
    def test_plan_is_dict_with_structure(self):
        result = get_engine_plans(
            [SIMPLE_QUERY], dataset=DATASET, scale_factor=SCALE_FACTOR, verbose=False,
        )
        plan = result.plans[0]
        assert isinstance(plan, dict)
        # Engine plans have a "structure" key used by patch application
        assert "structure" in plan

    @modal
    def test_invalid_query_gives_error(self):
        result = get_engine_plans(
            [INVALID_QUERY], dataset=DATASET, scale_factor=SCALE_FACTOR, verbose=False,
        )
        # Either plans[0] is None or errors[0] is set
        assert result.plans[0] is None or result.errors[0] is not None

    @modal
    def test_multiple_queries(self):
        result = get_engine_plans(
            [SIMPLE_QUERY, JOIN_QUERY, AGG_QUERY],
            dataset=DATASET, scale_factor=SCALE_FACTOR, verbose=False,
        )
        assert len(result.plans) == 3
        assert len(result.errors) == 3
        assert all(p is not None for p in result.plans)
        assert all(e is None for e in result.errors)

    @modal
    def test_mixed_valid_invalid(self):
        result = get_engine_plans(
            [SIMPLE_QUERY, INVALID_QUERY],
            dataset=DATASET, scale_factor=SCALE_FACTOR, verbose=False,
        )
        assert len(result.plans) == 2
        # Valid query succeeds
        assert result.plans[0] is not None
        assert result.errors[0] is None
        # Invalid query fails
        assert result.plans[1] is None or result.errors[1] is not None

    @modal
    def test_result_lengths_match_input(self):
        queries = [SIMPLE_QUERY, FILTER_QUERY, JOIN_QUERY, AGG_QUERY]
        result = get_engine_plans(
            queries, dataset=DATASET, scale_factor=SCALE_FACTOR, verbose=False,
        )
        assert len(result.plans) == len(queries)
        assert len(result.errors) == len(queries)


# ===================================================================
# benchmark_plans
# ===================================================================

class TestBenchmarkPlansIntegration:
    @modal
    def test_basic_benchmark(self, simple_plan):
        pp = PatchedPlan(base_plan=simple_plan, patch=[[]])
        result = benchmark_plans(
            [pp], dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_runs=1, verbose=False,
        )
        assert isinstance(result, BenchmarkResult)
        assert len(result.results) == 1
        assert len(result.results[0]) == 1

    @modal
    def test_measurement_has_benchmark_stats(self, simple_plan):
        pp = PatchedPlan(base_plan=simple_plan, patch=[[]])
        result = benchmark_plans(
            [pp], dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_runs=1, verbose=False,
        )
        m = result.results[0][0]
        assert "benchmark_stats" in m
        assert "n_runs" in m
        assert m["n_runs"] >= 1

    @modal
    def test_execution_time_stats_structure(self, simple_plan):
        pp = PatchedPlan(base_plan=simple_plan, patch=[[]])
        result = benchmark_plans(
            [pp], dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_runs=2, verbose=False,
        )
        m = result.results[0][0]
        et = m["benchmark_stats"]["execution_time"]
        # Must have all stat keys
        for key in METRIC_STAT_KEYS:
            assert key in et, f"Missing stat key: {key}"
        assert "all_runs" in et
        assert len(et["all_runs"]) == 2
        # execution_time must be positive
        assert et["min"] > 0
        assert et["max"] >= et["min"]

    @modal
    def test_no_get_full_metrics_by_default(self, simple_plan):
        pp = PatchedPlan(base_plan=simple_plan, patch=[[]])
        result = benchmark_plans(
            [pp], dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_runs=1, verbose=False,
        )
        bs = result.results[0][0]["benchmark_stats"]
        assert "execution_time" in bs
        assert "bytes_scanned" not in bs
        assert "join_time_s_sum" not in bs

    @modal
    def test_get_full_metrics_on_join(self, join_plan):
        pp = PatchedPlan(base_plan=join_plan, patch=[[]])
        result = benchmark_plans(
            [pp], dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_runs=1, get_full_metrics=True, verbose=False,
        )
        bs = result.results[0][0]["benchmark_stats"]
        assert "execution_time" in bs
        # Full metrics keys should be present for a join query
        for key in ("bytes_scanned", "output_rows_sum", "input_rows_sum",
                     "join_time_s_sum", "build_time_s_sum"):
            assert key in bs, f"Missing get_full_metrics key: {key}"
        # Each metric should have the standard stats structure
        for key in bs:
            assert "all_runs" in bs[key]
            assert "min" in bs[key]

    @modal
    def test_get_full_metrics_on_simple_query(self, simple_plan):
        pp = PatchedPlan(base_plan=simple_plan, patch=[[]])
        result = benchmark_plans(
            [pp], dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_runs=1, get_full_metrics=True, verbose=False,
        )
        bs = result.results[0][0]["benchmark_stats"]
        assert "execution_time" in bs
        # Even simple queries should have bytes_scanned
        assert "bytes_scanned" in bs

    @modal
    def test_none_patch_skipped(self, simple_plan):
        pp = PatchedPlan(base_plan=simple_plan, patch=[[], None])
        result = benchmark_plans(
            [pp], dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_runs=1, verbose=False,
        )
        assert len(result.results[0]) == 2
        # First patch (no-op) should succeed
        assert "benchmark_stats" in result.results[0][0]
        # Second patch (None) should be skipped
        skipped = result.results[0][1]
        assert "error" in skipped or "skipped" in skipped

    @modal
    def test_multiple_plans(self, simple_plan, join_plan):
        pp1 = PatchedPlan(base_plan=simple_plan, patch=[[]])
        pp2 = PatchedPlan(base_plan=join_plan, patch=[[]])
        result = benchmark_plans(
            [pp1, pp2], dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_runs=1, verbose=False,
        )
        assert len(result.results) == 2
        assert all("benchmark_stats" in r[0] for r in result.results)

    @modal
    def test_output_file(self, simple_plan, tmp_path):
        pp = PatchedPlan(base_plan=simple_plan, patch=[[]])
        outfile = str(tmp_path / "benchmark_result.json")
        result = benchmark_plans(
            [pp], dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_runs=1, output_file=outfile, verbose=False,
        )
        assert result.output_file == outfile
        assert Path(outfile).exists()
        data = json.loads(Path(outfile).read_text())
        assert isinstance(data, list)
        assert len(data) == 1


# ===================================================================
# benchmark_queries
# ===================================================================

class TestBenchmarkQueriesIntegration:
    @modal
    def test_valid_query(self):
        result = benchmark_queries(
            [SIMPLE_QUERY], dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_runs=1, verbose=False,
        )
        assert isinstance(result, BenchmarkResult)
        assert len(result.results) == 1
        assert len(result.results[0]) == 1
        m = result.results[0][0]
        assert "benchmark_stats" in m
        assert "execution_time" in m["benchmark_stats"]

    @modal
    def test_invalid_query_reports_planning_error(self):
        result = benchmark_queries(
            [INVALID_QUERY], dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_runs=1, verbose=False,
        )
        assert len(result.results) == 1
        err_entry = result.results[0][0]
        assert "error" in err_entry
        assert "planning_failed" in err_entry["error"]

    @modal
    def test_mixed_valid_invalid(self):
        result = benchmark_queries(
            [SIMPLE_QUERY, INVALID_QUERY],
            dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_runs=1, verbose=False,
        )
        assert len(result.results) == 2
        # Valid query has benchmark_stats
        assert "benchmark_stats" in result.results[0][0]
        # Invalid query has error
        assert "error" in result.results[1][0]

    @modal
    def test_get_full_metrics(self):
        result = benchmark_queries(
            [JOIN_QUERY], dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_runs=1, get_full_metrics=True, verbose=False,
        )
        bs = result.results[0][0]["benchmark_stats"]
        assert "execution_time" in bs
        assert "bytes_scanned" in bs

    @modal
    def test_output_file(self, tmp_path):
        outfile = str(tmp_path / "bq_result.json")
        result = benchmark_queries(
            [SIMPLE_QUERY], dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_runs=1, output_file=outfile, verbose=False,
        )
        assert result.output_file == outfile
        assert Path(outfile).exists()
        data = json.loads(Path(outfile).read_text())
        assert isinstance(data, list) and len(data) == 1

    @modal
    def test_multiple_queries(self):
        result = benchmark_queries(
            [SIMPLE_QUERY, JOIN_QUERY, AGG_QUERY],
            dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_runs=1, verbose=False,
        )
        assert len(result.results) == 3
        assert all("benchmark_stats" in r[0] for r in result.results)


# ===================================================================
# Plan structure and patch application (no Modal)
# ===================================================================

class TestPlanPatchApplication:
    def test_noop_patch_preserves_plan(self, simple_plan):
        from sampling.utils import apply_patches_to_plan
        result = apply_patches_to_plan(simple_plan, [])
        assert result == simple_plan
        # Must be a deep copy
        assert result is not simple_plan

    def test_patch_does_not_mutate_original(self, simple_plan):
        from sampling.utils import apply_patches_to_plan
        import copy
        original = copy.deepcopy(simple_plan)
        # Apply some patch (may fail if path doesn't exist, but
        # at least verify original isn't mutated)
        try:
            apply_patches_to_plan(simple_plan, [{"op": "add", "path": "/structure/_test_key", "value": 99}])
        except Exception:
            pass
        assert simple_plan == original

    def test_plan_has_structure_key(self, simple_plan):
        assert "structure" in simple_plan
        assert isinstance(simple_plan["structure"], dict)


# ===================================================================
# extract_patches_from_response (no Modal)
# ===================================================================

class TestExtractPatchesFromResponse:
    def test_valid_patch_block(self):
        from sampling.utils import extract_patches_from_response
        content = 'Some text <patch>[{"op":"replace","path":"/x","value":1}]</patch> more text'
        patches = extract_patches_from_response(content, verbose=False)
        assert patches is not None
        assert len(patches) == 1
        assert patches[0]["op"] == "replace"

    def test_multiple_ops(self):
        from sampling.utils import extract_patches_from_response
        content = '<patch>[{"op":"add","path":"/a","value":1},{"op":"remove","path":"/b"}]</patch>'
        patches = extract_patches_from_response(content, verbose=False)
        assert patches is not None
        assert len(patches) == 2

    def test_empty_patch_list(self):
        from sampling.utils import extract_patches_from_response
        content = '<patch>[]</patch>'
        patches = extract_patches_from_response(content, verbose=False)
        assert patches is not None
        assert patches == []

    def test_no_patch_tag(self):
        from sampling.utils import extract_patches_from_response
        content = 'No patch tags here at all.'
        patches = extract_patches_from_response(content, verbose=False)
        assert patches is None

    def test_invalid_json(self):
        from sampling.utils import extract_patches_from_response
        content = '<patch>[not valid json}</patch>'
        patches = extract_patches_from_response(content, verbose=False)
        assert patches is None

    def test_whitespace_tolerance(self):
        from sampling.utils import extract_patches_from_response
        content = '<patch>\n  [\n    {"op":"add","path":"/x","value":1}\n  ]\n</patch>'
        patches = extract_patches_from_response(content, verbose=False)
        assert patches is not None
        assert len(patches) == 1


# ===================================================================
# compute_metric_stats None handling (no Modal)
# ===================================================================

class TestComputeMetricStatsNoneHandling:
    def test_all_none(self):
        from modal_controller.utils import compute_metric_stats, METRIC_STAT_KEYS
        stats = compute_metric_stats([None, None, None])
        assert all(stats[k] is None for k in METRIC_STAT_KEYS)

    def test_mixed_none_and_values(self):
        from modal_controller.utils import compute_metric_stats
        stats = compute_metric_stats([1.0, None, 3.0, None, 5.0])
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0

    def test_single_value_among_nones(self):
        from modal_controller.utils import compute_metric_stats
        stats = compute_metric_stats([None, 42.0, None])
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0
        assert stats["std"] == 0.0


# ===================================================================
# API input validation (no Modal)
# ===================================================================

class TestAPIInputValidation:
    def test_benchmark_plans_string_raises(self):
        with pytest.raises((ValueError, TypeError)):
            benchmark_plans(
                "not a list", dataset=DATASET, scale_factor=SCALE_FACTOR,
            )

    def test_benchmark_queries_string_raises(self):
        with pytest.raises(ValueError, match="sequence of strings"):
            benchmark_queries(
                "SELECT 1", dataset=DATASET, scale_factor=SCALE_FACTOR,
            )

    def test_benchmark_queries_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            benchmark_queries(
                [], dataset=DATASET, scale_factor=SCALE_FACTOR,
            )

    def test_benchmark_plans_bad_n_runs(self):
        with pytest.raises(ValueError, match="n_runs"):
            benchmark_plans(
                [], dataset=DATASET, scale_factor=SCALE_FACTOR, n_runs=0,
            )

    def test_benchmark_plans_bad_workers(self):
        with pytest.raises(ValueError, match="max_workers"):
            benchmark_plans(
                [], dataset=DATASET, scale_factor=SCALE_FACTOR,
                max_workers=0,
            )

    def test_benchmark_plans_scale_conflict(self):
        with pytest.raises(ValueError, match="scale_factor conflicts"):
            benchmark_plans(
                [], dataset=DATASET, scale_factor=1,
                runner_kwargs={"scale_factor": 2},
            )


# ===================================================================
# optimize_queries input validation (no Modal)
# ===================================================================

class TestOptimizeQueriesValidation:
    def test_string_queries_raises(self):
        from dbplanbench import optimize_queries
        with pytest.raises(ValueError, match="sequence of strings"):
            optimize_queries(queries="SELECT 1", dataset=DATASET)

    def test_no_queries_no_run_dir_raises(self):
        from dbplanbench import optimize_queries
        with pytest.raises(ValueError):
            optimize_queries(queries=None, run_dir=None, dataset=DATASET)

    def test_negative_n_steps_raises(self):
        from dbplanbench import optimize_queries
        with pytest.raises(ValueError, match="n_steps"):
            optimize_queries(queries=["SELECT 1"], dataset=DATASET, n_steps=-1)

    def test_base_plans_length_mismatch_raises(self):
        from dbplanbench import optimize_queries
        with pytest.raises(ValueError, match="base_plans length"):
            optimize_queries(
                queries=["SELECT 1", "SELECT 2"],
                base_plans=[None],
                dataset=DATASET,
            )

    def test_best_of_multi_step_raises(self):
        from dbplanbench import optimize_queries
        with pytest.raises(ValueError, match="best_of"):
            optimize_queries(
                queries=["SELECT 1"], dataset=DATASET,
                strategy="best_of", n_steps=2,
            )

    def test_existing_run_dir_no_resume_raises(self, tmp_path: Path):
        from dbplanbench import optimize_queries
        run_dir = tmp_path / "existing_run"
        run_dir.mkdir()
        with pytest.raises(FileExistsError, match="already exists"):
            optimize_queries(
                queries=["SELECT 1"], dataset=DATASET,
                run_dir=str(run_dir),
            )


# ===================================================================
# scale_optimizations input validation (no Modal)
# ===================================================================

class TestScaleOptimizationsValidation:
    def test_existing_run_dir_raises(self, tmp_path: Path):
        from dbplanbench import scale_optimizations
        from dbplanbench_types import PatchedPlan
        run_dir = tmp_path / "existing_scale_run"
        run_dir.mkdir()
        dummy_plan = PatchedPlan(
            base_plan={"structure": {}, "succinct_table_info": {}},
            patch=[[]],
        )
        with pytest.raises(FileExistsError, match="already exists"):
            scale_optimizations(
                queries=["SELECT 1"], plans=[dummy_plan],
                source_scale_factor=1, target_scale_factor=3,
                dataset=DATASET, run_dir=str(run_dir),
            )


# ===================================================================
# get_metric_value extended coverage (no Modal)
# ===================================================================

class TestGetMetricValueExtended:
    def test_new_metric_keys(self):
        from dbplanbench_utils import get_metric_value
        stats = {
            "benchmark_stats": {
                "bytes_scanned": {"min": 100, "max": 200},
                "join_time_s_sum": {"min": 0.01, "max": 0.05},
            }
        }
        assert get_metric_value(stats, "bytes_scanned.min") == 100
        assert get_metric_value(stats, "join_time_s_sum.max") == 0.05

    def test_deeply_nested_missing(self):
        from dbplanbench_utils import get_metric_value
        stats = {"benchmark_stats": {"execution_time": {"min": 1.0}}}
        assert get_metric_value(stats, "execution_time.p50") is None
        assert get_metric_value(stats, "nonexistent.min") is None


# ===================================================================
# validate_queries input validation (no Modal)
# ===================================================================

class TestValidateQueriesInputValidation:
    def test_string_raises(self):
        with pytest.raises(ValueError, match="sequence of strings"):
            validate_queries("SELECT 1", dataset=DATASET)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_queries([], dataset=DATASET)

    def test_bad_workers(self):
        with pytest.raises(ValueError, match="max_workers"):
            validate_queries(["SELECT 1"], dataset=DATASET, max_workers=0)

    def test_scale_conflict(self):
        with pytest.raises(ValueError, match="scale_factor conflicts"):
            validate_queries(
                ["SELECT 1"], dataset=DATASET, scale_factor=1,
                runner_kwargs={"scale_factor": 2},
            )


# ===================================================================
# validate_queries end-to-end (requires Modal)
# ===================================================================

@modal
class TestValidateQueriesEndToEnd:
    def test_valid_query(self):
        result = validate_queries(
            [SIMPLE_QUERY], dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_determinism_retries=1, verbose=False,
        )
        assert isinstance(result, QueryValidationResult)
        assert result.summary["n_queries"] == 1
        assert result.summary["n_valid"] == 1
        assert result.errors[0] is None
        assert result.plans[0] is not None
        assert isinstance(result.plans[0], dict)

    def test_invalid_query(self):
        result = validate_queries(
            [INVALID_QUERY], dataset=DATASET, scale_factor=SCALE_FACTOR,
            n_determinism_retries=1, verbose=False,
        )
        assert result.summary["n_valid"] == 0
        assert result.errors[0] is not None
        assert result.plans[0] is None

    def test_mixed_valid_invalid(self):
        result = validate_queries(
            [SIMPLE_QUERY, INVALID_QUERY], dataset=DATASET,
            scale_factor=SCALE_FACTOR, n_determinism_retries=1, verbose=False,
        )
        assert result.summary["n_queries"] == 2
        assert result.summary["n_valid"] == 1
        assert result.errors[0] is None
        assert result.plans[0] is not None
        assert result.errors[1] is not None
        assert result.plans[1] is None
