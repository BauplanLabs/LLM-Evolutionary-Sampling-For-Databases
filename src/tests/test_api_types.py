"""Tests for api_types module."""

from __future__ import annotations

from api_types import (
    Patch,
    PatchSet,
    PatchedPlan,
    QueryGenerationResult,
    OptimizationResult,
    BenchmarkResult,
    ScaleResult,
    PlanningResult,
)


# ---------------------------------------------------------------------------
# PatchedPlan
# ---------------------------------------------------------------------------

class TestPatchedPlan:
    def test_construction_with_patch(self):
        base = {"op": "scan", "table": "t1"}
        patch: Patch = [{"op": "replace", "path": "/table", "value": "t2"}]
        pp = PatchedPlan(base_plan=base, patch=[patch])
        assert pp.base_plan == base
        assert pp.patch == [patch]

    def test_construction_base_only(self):
        pp = PatchedPlan(base_plan={"op": "scan"}, patch=[[]])
        assert pp.patch == [[]]
        assert pp.patch[0] == []

    def test_construction_unavailable_patch(self):
        pp = PatchedPlan(base_plan={"op": "scan"}, patch=[None])
        assert pp.patch == [None]
        assert pp.patch[0] is None

    def test_top_k_patches(self):
        p1: Patch = [{"op": "replace", "path": "/a", "value": 1}]
        p2: Patch = [{"op": "replace", "path": "/a", "value": 2}]
        pp = PatchedPlan(base_plan={}, patch=[p1, p2])
        assert len(pp.patch) == 2


# ---------------------------------------------------------------------------
# PatchSet semantics
# ---------------------------------------------------------------------------

class TestPatchSetSemantics:
    def test_single_patch(self):
        patch: Patch = [{"op": "add", "path": "/x", "value": 1}]
        ps: PatchSet = [patch]
        assert len(ps) == 1
        assert ps[0] is not None
        assert len(ps[0]) == 1

    def test_no_op(self):
        ps: PatchSet = [[]]
        assert len(ps) == 1
        assert ps[0] == []

    def test_unavailable(self):
        ps: PatchSet = [None]
        assert ps[0] is None

    def test_mixed(self):
        patch: Patch = [{"op": "add", "path": "/x", "value": 1}]
        ps: PatchSet = [patch, [], None]
        assert len(ps) == 3
        assert ps[0] is not None and len(ps[0]) == 1
        assert ps[1] == []
        assert ps[2] is None


# ---------------------------------------------------------------------------
# QueryGenerationResult
# ---------------------------------------------------------------------------

class TestQueryGenerationResult:
    def test_construction(self):
        r = QueryGenerationResult(
            run_dir="/tmp/run",
            summary={"total": 10},
            queries=["SELECT 1", "SELECT 2"],
            metadata=[{"id": 1}, {"id": 2}],
        )
        assert r.run_dir == "/tmp/run"
        assert len(r.queries) == 2
        assert len(r.metadata) == 2


# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------

class TestOptimizationResult:
    def test_successful_result(self):
        pp = PatchedPlan(base_plan={"op": "scan"}, patch=[[]])
        r = OptimizationResult(
            run_dir="/tmp/run",
            summary={"status": "ok"},
            queries=["SELECT 1"],
            optimization_outcome=[pp],
            metadata=[{"stats": {}}],
            validation_failures=[None],
        )
        assert r.optimization_outcome is not None
        assert len(r.optimization_outcome) == 1

    def test_early_exit(self):
        r = OptimizationResult(
            run_dir=None,
            summary={"status": "validation_failed"},
            queries=["SELECT 1"],
            optimization_outcome=None,
            metadata=None,
            validation_failures=["invalid syntax"],
        )
        assert r.optimization_outcome is None
        assert r.validation_failures == ["invalid syntax"]


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    def test_construction(self):
        r = BenchmarkResult(
            results=[[{"execution_time": 1.0}]],
            output_file="/tmp/out.json",
        )
        assert len(r.results) == 1
        assert r.output_file == "/tmp/out.json"

    def test_default_output_file(self):
        r = BenchmarkResult(results=[])
        assert r.output_file is None


# ---------------------------------------------------------------------------
# ScaleResult
# ---------------------------------------------------------------------------

class TestScaleResult:
    def test_construction(self):
        pp = PatchedPlan(base_plan={}, patch=[None])
        r = ScaleResult(
            run_dir="/tmp/run",
            summary={},
            queries=["SELECT 1"],
            scaled_plans=[pp],
            metadata=[{}],
            transfer_failures=["transfer error"],
        )
        assert r.transfer_failures == ["transfer error"]
        assert r.scaled_plans[0].patch == [None]


# ---------------------------------------------------------------------------
# PlanningResult
# ---------------------------------------------------------------------------

class TestPlanningResult:
    def test_construction(self):
        r = PlanningResult(
            plans=[{"op": "scan"}, None],
            errors=[None, "planning failed"],
        )
        assert r.plans[0] == {"op": "scan"}
        assert r.plans[1] is None
        assert r.errors[1] == "planning failed"
