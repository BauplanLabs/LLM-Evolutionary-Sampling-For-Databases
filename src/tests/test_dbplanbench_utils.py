"""Tests for dbplanbench_utils module."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from unittest.mock import patch

from dbplanbench_utils import (
    normalize_config,
    write_json,
    resolve_run_dir,
    get_evaluation_stats,
    log_line,
    data_folder_for_dataset,
    ensure_local_data,
    plan_to_json,
    get_metric_value,
    build_validation_stats,
    repo_root,
)


# ---------------------------------------------------------------------------
# normalize_config
# ---------------------------------------------------------------------------

class TestNormalizeConfig:
    def test_sorts_keys(self):
        cfg = {"z": 1, "a": 2, "m": 3}
        result = normalize_config(cfg)
        assert list(result.keys()) == ["a", "m", "z"]

    def test_round_trip_stability(self):
        cfg = {"b": [3, 2, 1], "a": {"y": 10, "x": 20}}
        first = normalize_config(cfg)
        second = normalize_config(first)
        assert first == second

    def test_preserves_values(self):
        cfg = {"key": [1, "two", None, True, 3.14]}
        assert normalize_config(cfg) == cfg

    def test_empty_dict(self):
        assert normalize_config({}) == {}


# ---------------------------------------------------------------------------
# write_json
# ---------------------------------------------------------------------------

class TestWriteJson:
    def test_creates_file_and_parent_dirs(self, tmp_path: Path):
        target = tmp_path / "sub" / "dir" / "out.json"
        payload = {"hello": "world"}
        write_json(target, payload)
        assert target.exists()
        loaded = json.loads(target.read_text())
        assert loaded == payload

    def test_sorts_keys(self, tmp_path: Path):
        target = tmp_path / "sorted.json"
        write_json(target, {"z": 1, "a": 2})
        text = target.read_text()
        assert text.index('"a"') < text.index('"z"')

    def test_overwrites_existing(self, tmp_path: Path):
        target = tmp_path / "overwrite.json"
        write_json(target, {"v": 1})
        write_json(target, {"v": 2})
        assert json.loads(target.read_text()) == {"v": 2}


# ---------------------------------------------------------------------------
# resolve_run_dir
# ---------------------------------------------------------------------------

class TestResolveRunDir:
    def test_none_creates_timestamped_dir(self):
        result = resolve_run_dir(None, "optimize")
        assert result.is_absolute()
        assert "outputs" in result.parts
        assert "optimize" in result.parts
        # Timestamp component matches YY.MM.DD.HH.MM pattern
        assert re.match(r"\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}", result.name)

    def test_absolute_path_unchanged(self):
        result = resolve_run_dir("/tmp/my_run", "optimize")
        assert result == Path("/tmp/my_run")

    def test_relative_with_outputs_prefix(self):
        result = resolve_run_dir("outputs/my_run", "optimize")
        root = repo_root()
        assert result == root / "outputs" / "my_run"

    def test_relative_without_outputs_prefix(self):
        result = resolve_run_dir("my_run", "optimize")
        root = repo_root()
        assert result == root / "outputs" / "my_run"


# ---------------------------------------------------------------------------
# get_evaluation_stats
# ---------------------------------------------------------------------------

class TestGetEvaluationStats:
    def test_extracts_stats(self):
        plan_info = {"evaluation_stats": {"time": 1.5}}
        assert get_evaluation_stats(plan_info) == {"time": 1.5}

    def test_returns_none_for_missing_key(self):
        assert get_evaluation_stats({"other": 1}) is None

    def test_returns_none_for_none(self):
        assert get_evaluation_stats(None) is None

    def test_returns_none_for_non_dict(self):
        assert get_evaluation_stats("not a dict") is None
        assert get_evaluation_stats(42) is None


# ---------------------------------------------------------------------------
# log_line
# ---------------------------------------------------------------------------

class TestLogLine:
    def test_prints_when_verbose(self, capsys):
        log_line(True, "hello")
        assert capsys.readouterr().out.strip() == "hello"

    def test_silent_when_not_verbose(self, capsys):
        log_line(False, "hello")
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# data_folder_for_dataset
# ---------------------------------------------------------------------------

class TestDataFolderForDataset:
    def test_tpch(self):
        assert data_folder_for_dataset("tpch") == "/tmp/data/data_tpch"

    def test_tpcds(self):
        assert data_folder_for_dataset("tpcds") == "/tmp/data/data_tpcds"

    def test_arbitrary(self):
        assert data_folder_for_dataset("custom") == "/tmp/data/data_custom"

    def test_modal_path_ignores_scale_factor(self):
        # The scale factor lives in the Modal image, not the path, so encoding
        # it here would needlessly change the path (and the image).
        assert data_folder_for_dataset("tpch", scale_factor=3) == "/tmp/data/data_tpch"

    def test_local_scaleless_has_no_suffix(self):
        # A scaleless dataset (e.g. a future fixed dataset like IMDB/JOB) passes
        # scale_factor=None and gets a plain folder name.
        local = data_folder_for_dataset("tpch", exec_local=True)
        assert local != "/tmp/data/data_tpch"
        assert os.path.isabs(local)
        assert local.endswith(os.path.join("data", "data_tpch"))

    def test_local_scaled_gets_sf_suffix(self):
        # A scaled dataset gets a _sf<N> suffix so every scale coexists locally.
        local = data_folder_for_dataset("tpch", exec_local=True, scale_factor=3)
        assert os.path.isabs(local)
        assert local.endswith(os.path.join("data", "data_tpch_sf3"))

    def test_local_distinct_folder_per_scale(self):
        sf1 = data_folder_for_dataset("tpch", exec_local=True, scale_factor=1)
        sf3 = data_folder_for_dataset("tpch", exec_local=True, scale_factor=3)
        assert sf1 != sf3
        assert sf1.endswith("data_tpch_sf1")
        assert sf3.endswith("data_tpch_sf3")


# ---------------------------------------------------------------------------
# ensure_local_data
# ---------------------------------------------------------------------------

def _fake_generate(dataset, factor=None, base_data_dir=None, **kwargs):
    """Stand-in for generate_benchmark_data: writes a dummy parquet table."""
    d = os.path.join(base_data_dir, f"data_{dataset}")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "lineitem.parquet"), "w").close()


class TestEnsureLocalData:
    def test_noop_when_data_present(self, tmp_path):
        """Existing scale-keyed folder -> return it, do not regenerate."""
        (tmp_path / "data_tpch_sf3").mkdir()
        (tmp_path / "data_tpch_sf3" / "lineitem.parquet").touch()
        with patch("modal_controller.constants.LOCAL_DATA_DIR", str(tmp_path)), \
             patch("modal_controller.generate_tpch_files.generate_benchmark_data") as gen:
            folder = ensure_local_data("tpch", 3)
        gen.assert_not_called()
        assert folder == str(tmp_path / "data_tpch_sf3")

    def test_generates_in_use_benchmark_at_scale_when_missing(self, tmp_path):
        """No data folder -> generate exactly that dataset at the given scale,
        then atomically place it at the scale-keyed folder under LOCAL_DATA_DIR."""
        with patch("modal_controller.constants.LOCAL_DATA_DIR", str(tmp_path)), \
             patch("modal_controller.generate_tpch_files.generate_benchmark_data",
                   side_effect=_fake_generate) as gen:
            folder = ensure_local_data("tpcds", 7)
        gen.assert_called_once()
        assert gen.call_args.args[0] == "tpcds"
        assert gen.call_args.kwargs.get("factor") == 7
        # Landed at the scale-keyed folder; no leftover temp generation dirs.
        assert folder == str(tmp_path / "data_tpcds_sf7")
        assert os.path.exists(os.path.join(folder, "lineitem.parquet"))
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".tpcds_gen_")]

    def test_different_scales_do_not_collide(self, tmp_path):
        """Generating sf=1 then sf=2 yields two distinct folders; the second
        generation is not short-circuited by the first."""
        with patch("modal_controller.constants.LOCAL_DATA_DIR", str(tmp_path)), \
             patch("modal_controller.generate_tpch_files.generate_benchmark_data",
                   side_effect=_fake_generate) as gen:
            folder_sf1 = ensure_local_data("tpch", 1)
            folder_sf2 = ensure_local_data("tpch", 2)
        assert folder_sf1 != folder_sf2
        assert folder_sf1 == str(tmp_path / "data_tpch_sf1")
        assert folder_sf2 == str(tmp_path / "data_tpch_sf2")
        assert gen.call_count == 2
        assert {c.kwargs.get("factor") for c in gen.call_args_list} == {1, 2}

    def test_rejects_unsupported_dataset(self, tmp_path):
        """Unsupported dataset -> clear ValueError before any generation."""
        with patch("modal_controller.constants.LOCAL_DATA_DIR", str(tmp_path)), \
             patch("modal_controller.generate_tpch_files.generate_benchmark_data") as gen:
            with pytest.raises(ValueError, match="tpch"):
                ensure_local_data("imdb", 3)
        gen.assert_not_called()


# ---------------------------------------------------------------------------
# plan_to_json
# ---------------------------------------------------------------------------

class TestPlanToJson:
    def test_dict_to_json(self):
        plan = {"op": "scan", "table": "t1"}
        result = plan_to_json(plan)
        assert isinstance(result, str)
        assert json.loads(result) == plan

    def test_string_passthrough(self):
        already_json = '{"op": "scan"}'
        assert plan_to_json(already_json) is already_json

    def test_list_to_json(self):
        plan = [{"op": "a"}, {"op": "b"}]
        assert json.loads(plan_to_json(plan)) == plan


# ---------------------------------------------------------------------------
# get_metric_value
# ---------------------------------------------------------------------------

class TestGetMetricValue:
    def test_simple_key(self):
        stats = {"execution_time": 1.5, "memory": 256}
        assert get_metric_value(stats, "execution_time") == 1.5

    def test_dotted_path_with_benchmark_stats(self):
        stats = {
            "benchmark_stats": {
                "execution_time": {"min": 0.5, "max": 2.0}
            }
        }
        assert get_metric_value(stats, "execution_time.min") == 0.5

    def test_dotted_path_without_benchmark_stats(self):
        stats = {"execution_time": {"min": 0.5}}
        assert get_metric_value(stats, "execution_time.min") == 0.5

    def test_missing_key(self):
        assert get_metric_value({"a": 1}, "b") is None

    def test_missing_dotted_path(self):
        stats = {"benchmark_stats": {"x": {"y": 1}}}
        assert get_metric_value(stats, "x.z") is None

    def test_non_numeric_value(self):
        assert get_metric_value({"key": "string"}, "key") is None

    def test_none_stats(self):
        assert get_metric_value(None, "key") is None

    def test_empty_stats(self):
        assert get_metric_value({}, "key") is None

    def test_int_value(self):
        assert get_metric_value({"count": 42}, "count") == 42


# ---------------------------------------------------------------------------
# build_validation_stats
# ---------------------------------------------------------------------------

class TestBuildValidationStats:
    def test_all_valid(self):
        stats = build_validation_stats([None, None, None])
        assert stats["n_queries"] == 3
        assert stats["n_valid"] == 3
        assert stats["random_3_error_messages"] == []

    def test_all_failed(self):
        stats = build_validation_stats(["err1", "err2"])
        assert stats["n_queries"] == 2
        assert stats["n_valid"] == 0
        assert set(stats["random_3_error_messages"]) == {"err1", "err2"}

    def test_mixed(self):
        stats = build_validation_stats([None, "timeout", None, "duplicate"])
        assert stats["n_queries"] == 4
        assert stats["n_valid"] == 2
        assert len(stats["random_3_error_messages"]) == 2

    def test_empty_list(self):
        stats = build_validation_stats([])
        assert stats["n_queries"] == 0
        assert stats["n_valid"] == 0
        assert stats["random_3_error_messages"] == []

    def test_caps_at_three_errors(self):
        failures = ["e1", "e2", "e3", "e4", "e5"]
        stats = build_validation_stats(failures)
        assert len(stats["random_3_error_messages"]) == 3
        assert all(e in failures for e in stats["random_3_error_messages"])

    def test_single_failure(self):
        stats = build_validation_stats(["only_error"])
        assert stats["n_queries"] == 1
        assert stats["n_valid"] == 0
        assert stats["random_3_error_messages"] == ["only_error"]
