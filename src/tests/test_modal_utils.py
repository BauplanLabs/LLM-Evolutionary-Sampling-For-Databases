"""Tests for modal_controller.utils — pure functions (no Modal/S3 calls)."""

from __future__ import annotations

import json
import math

import pytest

from modal_controller.utils import (
    compare_result_sets,
    compute_metric_stats,
    _percentile,
    METRIC_STAT_KEYS,
)


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_single_element(self):
        assert _percentile([5.0], 0.5) == 5.0

    def test_min_percentile(self):
        assert _percentile([1.0, 2.0, 3.0], 0.0) == 1.0

    def test_max_percentile(self):
        assert _percentile([1.0, 2.0, 3.0], 1.0) == 3.0

    def test_median_odd(self):
        assert _percentile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_median_even(self):
        assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5

    def test_p25(self):
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = _percentile(vals, 0.25)
        assert result == 20.0

    def test_empty_list(self):
        assert math.isnan(_percentile([], 0.5))


# ---------------------------------------------------------------------------
# compute_metric_stats
# ---------------------------------------------------------------------------

class TestComputeMetricStats:
    def test_normal_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_metric_stats(values)
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["p50"] == 3.0
        assert all(k in stats for k in METRIC_STAT_KEYS)

    def test_single_value(self):
        stats = compute_metric_stats([42.0])
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0
        assert stats["mean"] == 42.0
        assert stats["std"] == 0.0

    def test_empty_list(self):
        stats = compute_metric_stats([])
        assert all(stats[k] is None for k in METRIC_STAT_KEYS)

    def test_std_positive(self):
        stats = compute_metric_stats([1.0, 3.0])
        assert stats["std"] > 0

    def test_all_same(self):
        stats = compute_metric_stats([7.0, 7.0, 7.0])
        assert stats["std"] == 0.0
        assert stats["min"] == stats["max"] == 7.0


# ---------------------------------------------------------------------------
# compare_result_sets
# ---------------------------------------------------------------------------

class TestCompareResultSets:
    def _to_json(self, rows, columns=None):
        """Helper to create JSON string from list of dicts."""
        return json.dumps(rows)

    def test_identical_sets(self):
        data = self._to_json([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        assert compare_result_sets(data, data) is True

    def test_different_row_order(self):
        baseline = self._to_json([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        model = self._to_json([{"a": 3, "b": 4}, {"a": 1, "b": 2}])
        assert compare_result_sets(baseline, model) is True

    def test_different_values(self):
        baseline = self._to_json([{"a": 1}])
        model = self._to_json([{"a": 2}])
        assert compare_result_sets(baseline, model) is False

    def test_different_columns(self):
        baseline = self._to_json([{"a": 1}])
        model = self._to_json([{"b": 1}])
        assert compare_result_sets(baseline, model) is False

    def test_different_row_count(self):
        baseline = self._to_json([{"a": 1}])
        model = self._to_json([{"a": 1}, {"a": 2}])
        assert compare_result_sets(baseline, model) is False

    def test_float_rounding(self):
        baseline = self._to_json([{"val": 1.00000001}])
        model = self._to_json([{"val": 1.00000002}])
        assert compare_result_sets(baseline, model, rounding=6) is True

    def test_float_rounding_fails_on_larger_diff(self):
        baseline = self._to_json([{"val": 1.0}])
        model = self._to_json([{"val": 1.1}])
        assert compare_result_sets(baseline, model, rounding=6) is False

    def test_nan_equality(self):
        baseline = self._to_json([{"val": None}])
        model = self._to_json([{"val": None}])
        assert compare_result_sets(baseline, model) is True

    def test_invalid_json_returns_false(self):
        assert compare_result_sets("not json", "[1]") is False

    def test_empty_result_sets(self):
        data = self._to_json([])
        assert compare_result_sets(data, data) is True

    def test_column_order_independent(self):
        baseline = self._to_json([{"a": 1, "b": 2}])
        model = self._to_json([{"b": 2, "a": 1}])
        assert compare_result_sets(baseline, model) is True
