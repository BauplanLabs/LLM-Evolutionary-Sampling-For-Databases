"""Tests for sampling.sample_plans — pure plan-selection utilities (no Modal/LLM)."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from sampling.sample_plans import (
    SamplingStrategy,
    get_upstream_patches,
    get_upstream_evaluation_stats,
    get_base_plans_for_sampling,
)


# ---------------------------------------------------------------------------
# Helpers to build sample trees
# ---------------------------------------------------------------------------

def _make_plan(sample_id, parent_sample_id, patches, is_valid=True,
               metric_value=None, is_leaf=False, is_last=False):
    """Build a sampled_plan entry for testing."""
    entry = {
        "sample_id": sample_id,
        "parent_sample_id": parent_sample_id,
        "sampled_patches": patches,
        "is_valid": is_valid,
        "is_leaf": is_leaf,
        "is_last": is_last,
    }
    if is_valid and metric_value is not None:
        entry["evaluation_stats"] = {"execution_time": {"min": metric_value}}
    return entry


def _make_sampleid2idx(sampled_plans):
    return {p["sample_id"]: idx for idx, p in enumerate(sampled_plans)}


def _linear_chain():
    """Root -> A -> B  (linear chain, all valid).

    root (id=None, metric=2.0) -> A (id=1, metric=1.5) -> B (id=2, metric=1.0)
    """
    plans = [
        _make_plan(None, None, [], is_valid=True, metric_value=2.0),
        _make_plan(1, None, [{"op": "replace", "path": "/a", "value": 1}],
                   is_valid=True, metric_value=1.5, is_last=False, is_leaf=False),
        _make_plan(2, 1, [{"op": "replace", "path": "/b", "value": 2}],
                   is_valid=True, metric_value=1.0, is_last=True, is_leaf=True),
    ]
    return plans


def _branching_tree():
    """Root -> A, Root -> B, A -> C.

    root (None, 3.0) -> A (1, 2.0) -> C (3, 1.0)
                      -> B (2, 2.5)
    B and C are leaves. A and B are last-step nodes.
    """
    plans = [
        _make_plan(None, None, [], is_valid=True, metric_value=3.0),
        _make_plan(1, None, [{"op": "add", "path": "/x", "value": 1}],
                   is_valid=True, metric_value=2.0, is_last=True, is_leaf=False),
        _make_plan(2, None, [{"op": "add", "path": "/y", "value": 2}],
                   is_valid=True, metric_value=2.5, is_last=True, is_leaf=True),
        _make_plan(3, 1, [{"op": "add", "path": "/z", "value": 3}],
                   is_valid=True, metric_value=1.0, is_last=False, is_leaf=True),
    ]
    return plans


# ===================================================================
# get_upstream_patches
# ===================================================================

class TestGetUpstreamPatches:
    def test_root_has_no_patches(self):
        plans = _linear_chain()
        idx = _make_sampleid2idx(plans)
        result = get_upstream_patches(plans[0], {"sampled_plans": plans}, idx)
        assert result == []

    def test_single_hop(self):
        plans = _linear_chain()
        idx = _make_sampleid2idx(plans)
        result = get_upstream_patches(plans[1], {"sampled_plans": plans}, idx)
        assert result == [{"op": "replace", "path": "/a", "value": 1}]

    def test_two_hops_flattened(self):
        plans = _linear_chain()
        idx = _make_sampleid2idx(plans)
        result = get_upstream_patches(plans[2], {"sampled_plans": plans}, idx)
        assert result == [
            {"op": "replace", "path": "/a", "value": 1},
            {"op": "replace", "path": "/b", "value": 2},
        ]

    def test_invalid_node_skipped(self):
        """Invalid intermediate nodes have their patches skipped."""
        plans = [
            _make_plan(None, None, [], is_valid=True, metric_value=2.0),
            _make_plan(1, None, [{"op": "add", "path": "/x", "value": 1}],
                       is_valid=False),
            _make_plan(2, 1, [{"op": "add", "path": "/y", "value": 2}],
                       is_valid=True, metric_value=1.0),
        ]
        idx = _make_sampleid2idx(plans)
        result = get_upstream_patches(plans[2], {"sampled_plans": plans}, idx)
        # Node 1 is invalid, so its patches are skipped
        assert result == [{"op": "add", "path": "/y", "value": 2}]

    def test_must_apply_current_patches(self):
        """When must_apply_current_patches=True, even invalid current node's patches are included."""
        plans = [
            _make_plan(None, None, [], is_valid=True, metric_value=2.0),
            _make_plan(1, None, [{"op": "add", "path": "/x", "value": 1}],
                       is_valid=False),
        ]
        idx = _make_sampleid2idx(plans)
        result = get_upstream_patches(
            plans[1], {"sampled_plans": plans}, idx, must_apply_current_patches=True
        )
        assert result == [{"op": "add", "path": "/x", "value": 1}]

    def test_must_apply_only_affects_first_node(self):
        """must_apply_current_patches only applies to the first (current) node."""
        plans = [
            _make_plan(None, None, [], is_valid=True, metric_value=2.0),
            _make_plan(1, None, [{"op": "add", "path": "/a", "value": 1}],
                       is_valid=False),
            _make_plan(2, 1, [{"op": "add", "path": "/b", "value": 2}],
                       is_valid=False),
        ]
        idx = _make_sampleid2idx(plans)
        result = get_upstream_patches(
            plans[2], {"sampled_plans": plans}, idx, must_apply_current_patches=True
        )
        # Node 2 (current): invalid, but must_apply -> included
        # Node 1 (parent): invalid, must_apply=False now -> skipped
        assert result == [{"op": "add", "path": "/b", "value": 2}]

    def test_branching_path(self):
        """Upstream patches from a branch only include that branch's ancestors."""
        plans = _branching_tree()
        idx = _make_sampleid2idx(plans)
        # C (id=3) -> parent A (id=1) -> root
        result = get_upstream_patches(plans[3], {"sampled_plans": plans}, idx)
        assert result == [
            {"op": "add", "path": "/x", "value": 1},
            {"op": "add", "path": "/z", "value": 3},
        ]
        # B (id=2) -> root
        result_b = get_upstream_patches(plans[2], {"sampled_plans": plans}, idx)
        assert result_b == [{"op": "add", "path": "/y", "value": 2}]


# ===================================================================
# get_upstream_evaluation_stats
# ===================================================================

class TestGetUpstreamEvaluationStats:
    def test_valid_node_returns_own_stats(self):
        plans = _linear_chain()
        idx = _make_sampleid2idx(plans)
        result = get_upstream_evaluation_stats(plans[2], {"sampled_plans": plans}, idx)
        assert result == {"execution_time": {"min": 1.0}}

    def test_invalid_node_returns_parent_stats(self):
        plans = [
            _make_plan(None, None, [], is_valid=True, metric_value=2.0),
            _make_plan(1, None, [], is_valid=False),
        ]
        idx = _make_sampleid2idx(plans)
        result = get_upstream_evaluation_stats(plans[1], {"sampled_plans": plans}, idx)
        assert result == {"execution_time": {"min": 2.0}}

    def test_invalid_root_raises(self):
        plans = [
            _make_plan(None, None, [], is_valid=False),
        ]
        idx = _make_sampleid2idx(plans)
        with pytest.raises(RuntimeError, match="Root plan must be valid"):
            get_upstream_evaluation_stats(plans[0], {"sampled_plans": plans}, idx)

    def test_skips_multiple_invalid_ancestors(self):
        plans = [
            _make_plan(None, None, [], is_valid=True, metric_value=5.0),
            _make_plan(1, None, [], is_valid=False),
            _make_plan(2, 1, [], is_valid=False),
        ]
        idx = _make_sampleid2idx(plans)
        result = get_upstream_evaluation_stats(plans[2], {"sampled_plans": plans}, idx)
        assert result == {"execution_time": {"min": 5.0}}


# ===================================================================
# get_base_plans_for_sampling
# ===================================================================

class TestGetBasePlansForSampling:
    def _query_data(self, plans):
        return {"plan": {"op": "scan"}, "sampled_plans": plans}

    # --- "original" selector ---

    def test_original_returns_root(self):
        plans = _linear_chain()
        qd = self._query_data(plans)
        strategy = SamplingStrategy("original", "all", False)
        result = get_base_plans_for_sampling(qd, strategy)
        assert len(result) == 1
        assert result[0]["parent_sample_id"] is None
        assert result[0]["upstream_patches"] == []

    # --- "so_far" selector ---

    def test_so_far_all_returns_all_valid(self):
        plans = _linear_chain()
        qd = self._query_data(plans)
        strategy = SamplingStrategy("so_far", "all", from_valid_only=True,
                                    kwargs={"optimization_metric": "execution_time.min"})
        result = get_base_plans_for_sampling(qd, strategy)
        # All 3 plans are valid, so all 3 are returned
        assert len(result) == 3

    def test_so_far_best_1(self):
        plans = _linear_chain()
        qd = self._query_data(plans)
        strategy = SamplingStrategy("so_far", "best_1", from_valid_only=True,
                                    kwargs={"optimization_metric": "execution_time.min"})
        result = get_base_plans_for_sampling(qd, strategy)
        assert len(result) == 1
        # Best metric is 1.0 (plan id=2)
        assert result[0]["parent_sample_id"] == 2

    def test_so_far_best_n(self):
        plans = _branching_tree()
        qd = self._query_data(plans)
        strategy = SamplingStrategy("so_far", "best_n", from_valid_only=True,
                                    kwargs={"optimization_metric": "execution_time.min", "best_n": 2})
        result = get_base_plans_for_sampling(qd, strategy)
        assert len(result) == 2
        # Best 2: C (1.0) and A (2.0)
        ids = [r["parent_sample_id"] for r in result]
        assert ids == [3, 1]

    def test_best_n_missing_raises(self):
        plans = _linear_chain()
        qd = self._query_data(plans)
        strategy = SamplingStrategy("so_far", "best_n", from_valid_only=True,
                                    kwargs={"optimization_metric": "execution_time.min"})
        with pytest.raises(ValueError, match="best_n"):
            get_base_plans_for_sampling(qd, strategy)

    def test_best_1_missing_metric_raises(self):
        plans = _linear_chain()
        qd = self._query_data(plans)
        strategy = SamplingStrategy("so_far", "best_1", from_valid_only=True, kwargs={})
        with pytest.raises(ValueError, match="optimization_metric"):
            get_base_plans_for_sampling(qd, strategy)

    # --- "of_last" selector ---

    def test_of_last_filters(self):
        plans = _branching_tree()
        qd = self._query_data(plans)
        strategy = SamplingStrategy("of_last", "all", from_valid_only=False)
        result = get_base_plans_for_sampling(qd, strategy)
        # A (is_last=True) and B (is_last=True) are the "last" nodes
        ids = sorted(r["parent_sample_id"] for r in result)
        assert ids == [1, 2]

    # --- "of_leafs" selector ---

    def test_of_leafs_filters(self):
        plans = _branching_tree()
        qd = self._query_data(plans)
        strategy = SamplingStrategy("of_leafs", "all", from_valid_only=False)
        result = get_base_plans_for_sampling(qd, strategy)
        # B (is_leaf=True) and C (is_leaf=True)
        ids = sorted(r["parent_sample_id"] for r in result)
        assert ids == [2, 3]

    # --- from_valid_only ---

    def test_from_valid_only_filters_invalid(self):
        plans = [
            _make_plan(None, None, [], is_valid=True, metric_value=3.0),
            _make_plan(1, None, [], is_valid=False, is_leaf=True, is_last=True),
            _make_plan(2, None, [], is_valid=True, metric_value=1.0, is_leaf=True, is_last=True),
        ]
        qd = self._query_data(plans)
        strategy = SamplingStrategy("of_leafs", "all", from_valid_only=True)
        result = get_base_plans_for_sampling(qd, strategy)
        assert len(result) == 1
        assert result[0]["parent_sample_id"] == 2

    # --- upstream_patches correctness ---

    def test_upstream_patches_populated(self):
        plans = _linear_chain()
        qd = self._query_data(plans)
        strategy = SamplingStrategy("of_leafs", "all", from_valid_only=True)
        result = get_base_plans_for_sampling(qd, strategy)
        assert len(result) == 1
        # Leaf is B (id=2): patches from A + B
        assert len(result[0]["upstream_patches"]) == 2

    # --- upstream_evaluation_stats correctness ---

    def test_upstream_evaluation_stats_populated(self):
        plans = _linear_chain()
        qd = self._query_data(plans)
        strategy = SamplingStrategy("so_far", "best_1", from_valid_only=True,
                                    kwargs={"optimization_metric": "execution_time.min"})
        result = get_base_plans_for_sampling(qd, strategy)
        assert result[0]["upstream_evaluation_stats"] == {"execution_time": {"min": 1.0}}

    # --- determinism ---

    def test_deterministic(self):
        plans = _branching_tree()
        qd = self._query_data(plans)
        strategy = SamplingStrategy("so_far", "best_n", from_valid_only=True,
                                    kwargs={"optimization_metric": "execution_time.min", "best_n": 2})
        r1 = get_base_plans_for_sampling(qd, strategy)
        r2 = get_base_plans_for_sampling(qd, strategy)
        assert r1 == r2
