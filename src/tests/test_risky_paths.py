"""Tests targeting risky code paths in the sampling + local-execution layers.

Adapted to main's structure: operation builders live in
``modal_controller/operations/base_operation.py`` (extracted from the runners),
and there is no full-plan sampling mode. Covers metric/number parsing,
``validate_plan_result_set``, ``submit_run_operation`` (local path),
``evaluate_sampled_plans`` branching, the ``op_*`` operation contracts,
``get_upstream_patches`` chain reconstruction, and ``GenerationResult`` types.
"""

from __future__ import annotations

import json
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Metric / number parsing (base_operation)
# ===========================================================================

class TestParseDurationToSeconds:
    def test_nanoseconds(self):
        from modal_controller.operations.base_operation import _parse_duration_to_seconds
        assert _parse_duration_to_seconds("500ns") == pytest.approx(500e-9)

    def test_microseconds_unicode(self):
        from modal_controller.operations.base_operation import _parse_duration_to_seconds
        assert _parse_duration_to_seconds("100µs") == pytest.approx(100e-6)

    def test_milliseconds(self):
        from modal_controller.operations.base_operation import _parse_duration_to_seconds
        assert _parse_duration_to_seconds("250ms") == pytest.approx(0.25)

    def test_seconds(self):
        from modal_controller.operations.base_operation import _parse_duration_to_seconds
        assert _parse_duration_to_seconds("3.5s") == pytest.approx(3.5)

    def test_plain_number(self):
        from modal_controller.operations.base_operation import _parse_duration_to_seconds
        assert _parse_duration_to_seconds("42") == pytest.approx(42.0)

    def test_whitespace_stripped(self):
        from modal_controller.operations.base_operation import _parse_duration_to_seconds
        assert _parse_duration_to_seconds("  100ms  ") == pytest.approx(0.1)

    def test_zero(self):
        from modal_controller.operations.base_operation import _parse_duration_to_seconds
        assert _parse_duration_to_seconds("0ns") == pytest.approx(0.0)


class TestParseInt:
    def test_plain_int(self):
        from modal_controller.operations.base_operation import _parse_int
        assert _parse_int("42") == 42

    def test_comma_separated(self):
        from modal_controller.operations.base_operation import _parse_int
        assert _parse_int("1,234,567") == 1234567

    def test_whitespace(self):
        from modal_controller.operations.base_operation import _parse_int
        assert _parse_int("  99  ") == 99

    def test_zero(self):
        from modal_controller.operations.base_operation import _parse_int
        assert _parse_int("0") == 0


class TestParsePlanMetrics:
    def test_basic_metric_extraction(self):
        from modal_controller.operations.base_operation import parse_plan_metrics

        mock_plan = MagicMock()
        mock_plan.display_with_metrics.return_value = (
            "HashJoinExec: metrics=[output_rows=1000, bytes_scanned=4096, "
            "build_mem_used=2048, peak_mem_used=3072]\n"
            "ParquetExec: metrics=[output_rows=5000, bytes_scanned=8192]\n"
        )

        result = parse_plan_metrics(mock_plan)
        assert result["bytes_scanned"] == 4096 + 8192
        assert result["output_rows_sum"] == 1000  # join_only
        assert result["build_mem_used_sum"] == 2048
        assert result["build_mem_used_max"] == 2048
        assert result["peak_mem_used_max"] == 3072

    def test_display_with_metrics_exception(self):
        from modal_controller.operations.base_operation import parse_plan_metrics

        mock_plan = MagicMock()
        mock_plan.display_with_metrics.side_effect = RuntimeError("no metrics")

        result = parse_plan_metrics(mock_plan)
        assert result == {}

    def test_no_metrics_lines(self):
        from modal_controller.operations.base_operation import parse_plan_metrics

        mock_plan = MagicMock()
        mock_plan.display_with_metrics.return_value = (
            "SortExec: no metrics available\n"
            "ProjectionExec: columns=[a, b]\n"
        )

        result = parse_plan_metrics(mock_plan)
        assert result["bytes_scanned"] is None or result["bytes_scanned"] == 0

    def test_join_only_metrics_filter(self):
        from modal_controller.operations.base_operation import parse_plan_metrics

        mock_plan = MagicMock()
        mock_plan.display_with_metrics.return_value = (
            "HashJoinExec: metrics=[output_rows=500]\n"
            "ParquetExec: metrics=[output_rows=10000]\n"
            "FilterExec: metrics=[output_rows=3000]\n"
        )

        result = parse_plan_metrics(mock_plan)
        assert result["output_rows_sum"] == 500

    def test_multiple_joins(self):
        from modal_controller.operations.base_operation import parse_plan_metrics

        mock_plan = MagicMock()
        mock_plan.display_with_metrics.return_value = (
            "HashJoinExec: metrics=[build_mem_used=1024, output_rows=100]\n"
            "HashJoinExec: metrics=[build_mem_used=2048, output_rows=200]\n"
        )

        result = parse_plan_metrics(mock_plan)
        assert result["build_mem_used_sum"] == 1024 + 2048
        assert result["build_mem_used_max"] == 2048
        assert result["output_rows_sum"] == 100 + 200

    def test_commas_in_metric_values(self):
        from modal_controller.operations.base_operation import parse_plan_metrics

        mock_plan = MagicMock()
        mock_plan.display_with_metrics.return_value = (
            "HashJoinExec: metrics=[output_rows=1,234, bytes_scanned=5,678,901]\n"
        )

        result = parse_plan_metrics(mock_plan)
        assert result["output_rows_sum"] == 1234
        assert result["bytes_scanned"] == 5678901


# ===========================================================================
# get_upstream_patches — flattening + storage/rebench consistency
# ===========================================================================

class TestGetUpstreamPatches:
    def test_list_patches_flatten_correctly(self):
        from sampling.sample_plans import get_upstream_patches

        query_data = {
            "sampled_plans": [
                {"sample_id": None, "is_valid": True, "sampled_patches": []},
                {"sample_id": 1, "parent_sample_id": None, "is_valid": True,
                 "sampled_patches": [{"op": "replace", "path": "/0/x", "value": 1}]},
                {"sample_id": 2, "parent_sample_id": 1, "is_valid": True,
                 "sampled_patches": [{"op": "replace", "path": "/0/y", "value": 2}]},
            ]
        }
        sid2idx = {None: 0, 1: 1, 2: 2}
        result = get_upstream_patches(query_data["sampled_plans"][2], query_data, sid2idx)
        assert [p["path"] for p in result] == ["/0/x", "/0/y"]

    def test_empty_list_patches_produces_empty_result(self):
        from sampling.sample_plans import get_upstream_patches

        query_data = {
            "sampled_plans": [
                {"sample_id": None, "is_valid": True, "sampled_patches": []},
                {"sample_id": 1, "parent_sample_id": None, "is_valid": True, "sampled_patches": []},
            ]
        }
        sid2idx = {None: 0, 1: 1}
        assert get_upstream_patches(query_data["sampled_plans"][1], query_data, sid2idx) == []

    def test_rebench_skips_invalid_leaf_own_patches(self):
        """By design: get_upstream_patches (must_apply=False) excludes an INVALID
        leaf's own patches. Such candidates are never benchmarked by rebench, so
        the storage-vs-rebench difference on invalid leaves is expected + harmless."""
        from sampling.sample_plans import get_upstream_patches

        query_data = {
            "sampled_plans": [
                {"sample_id": None, "is_valid": True, "sampled_patches": []},
                {"sample_id": 1, "parent_sample_id": None, "is_valid": False,
                 "sampled_patches": [{"op": "replace", "path": "/0/x", "value": 1}]},
            ]
        }
        sid2idx = {None: 0, 1: 1}
        assert get_upstream_patches(query_data["sampled_plans"][1], query_data, sid2idx) == []

    def test_storage_chain_equals_rebench_for_valid_samples(self):
        """The chain stored at sampling time (get_base_plans_for_sampling) equals the
        chain recomputed at rebench time (get_upstream_patches over the final tree)
        for every VALID sample. Regression guard: there is no chain-reconstruction
        bug for valid samples (only the by-design invalid-leaf skip above)."""
        from sampling.sample_plans import (
            get_base_plans_for_sampling, get_upstream_patches, SamplingStrategy,
        )

        strat = SamplingStrategy("of_leafs", "all", from_valid_only=False)
        tree = [{
            "plan_type": "original", "sample_id": None, "parent_sample_id": None,
            "is_last": True, "is_leaf": True, "sampled_patches": [], "is_valid": True,
            "evaluation_stats": {"execution_time": {"min": 100.0}},
        }]
        stored = {}
        nid = 1
        for _ in range(2):  # two deepening steps
            qd = {"id": 0, "query": "q", "plan": {"structure": {}}, "sampled_plans": tree}
            selections = get_base_plans_for_sampling(qd, strat)
            new_nodes = []
            for sel in selections:
                child = {
                    "plan_type": "optimized", "sample_id": nid,
                    "parent_sample_id": sel["parent_sample_id"], "is_last": True,
                    "is_leaf": True, "is_valid": True,
                    "sampled_patches": [{"op": "replace", "path": f"/{nid}/x", "value": nid}],
                    "evaluation_stats": {"execution_time": {"min": 90.0 - nid}},
                }
                stored[nid] = list(sel["upstream_patches"])
                new_nodes.append(child)
                nid += 1
            gained = {n["parent_sample_id"] for n in new_nodes}
            for p in tree:
                if p["sample_id"] in gained:
                    p["is_leaf"] = False
            tree.extend(new_nodes)

        sid2idx = {p["sample_id"]: i for i, p in enumerate(tree)}
        final_qd = {"sampled_plans": tree}
        for node in tree:
            if node["sample_id"] is None:
                continue
            stored_chain = stored[node["sample_id"]] + node["sampled_patches"]
            rebench_chain = get_upstream_patches(node, final_qd, sid2idx)
            assert stored_chain == rebench_chain, f"sid={node['sample_id']}"


# ===========================================================================
# validate_plan_result_set — edge cases (submit_run_operation mocked)
# ===========================================================================

class TestValidatePlanResultSet:
    def test_baseline_execution_failure(self):
        from modal_controller.utils import validate_plan_result_set
        with patch("modal_controller.utils.submit_run_operation",
                   return_value={"error": "Execution error: bad plan"}):
            result = validate_plan_result_set(
                '{"plan": "candidate"}', '{"plan": "baseline"}', "/data", exec_local=True,
            )
        assert result is not None and "Baseline execution failed" in result

    def test_candidate_execution_failure(self):
        from modal_controller.utils import validate_plan_result_set
        side_effects = [
            {"result_data": '[{"a": 1}]', "execution_time": 0.1},
            {"error": "Execution error: bad candidate"},
        ]
        with patch("modal_controller.utils.submit_run_operation", side_effect=side_effects):
            result = validate_plan_result_set(
                '{"plan": "candidate"}', '{"plan": "baseline"}', "/data",
                n_determinism_retries=1, exec_local=True,
            )
        assert result is not None and "Candidate execution failed" in result

    def test_result_set_mismatch(self):
        from modal_controller.utils import validate_plan_result_set
        side_effects = [
            {"result_data": '[{"a": 1}]', "execution_time": 0.1},
            {"result_data": '[{"a": 999}]', "execution_time": 0.1},
        ]
        with patch("modal_controller.utils.submit_run_operation", side_effect=side_effects):
            result = validate_plan_result_set(
                '{"plan": "candidate"}', '{"plan": "baseline"}', "/data",
                n_determinism_retries=1, exec_local=True,
            )
        assert result is not None and "Result set mismatch" in result

    def test_non_determinism_detected(self):
        from modal_controller.utils import validate_plan_result_set
        side_effects = [
            {"result_data": '[{"a": 1}]', "execution_time": 0.1},
            {"result_data": '[{"a": 1}]', "execution_time": 0.1},
            {"result_data": '[{"a": 2}]', "execution_time": 0.1},
        ]
        with patch("modal_controller.utils.submit_run_operation", side_effect=side_effects):
            result = validate_plan_result_set(
                '{"plan": "candidate"}', '{"plan": "baseline"}', "/data",
                n_determinism_retries=2, exec_local=True,
            )
        assert result is not None and "Non-deterministic" in result

    def test_all_matching_returns_none(self):
        from modal_controller.utils import validate_plan_result_set
        data = '[{"a": 1}]'
        side_effects = [
            {"result_data": data, "execution_time": 0.1},
            {"result_data": data, "execution_time": 0.1},
            {"result_data": data, "execution_time": 0.1},
        ]
        with patch("modal_controller.utils.submit_run_operation", side_effect=side_effects):
            result = validate_plan_result_set(
                '{"plan": "candidate"}', '{"plan": "baseline"}', "/data",
                n_determinism_retries=2, exec_local=True,
            )
        assert result is None

    def test_exception_during_validation_caught(self):
        from modal_controller.utils import validate_plan_result_set
        with patch("modal_controller.utils.submit_run_operation",
                   side_effect=RuntimeError("connection lost")):
            result = validate_plan_result_set(
                '{"plan": "candidate"}', '{"plan": "baseline"}', "/data", exec_local=True,
            )
        assert result is not None and "Validation failed" in result

    def test_none_result_from_submit(self):
        from modal_controller.utils import validate_plan_result_set
        with patch("modal_controller.utils.submit_run_operation", return_value=None):
            result = validate_plan_result_set(
                '{"plan": "candidate"}', '{"plan": "baseline"}', "/data", exec_local=True,
            )
        assert result is not None and "failed" in result.lower()


# ===========================================================================
# submit_run_operation — local path (LocalRunner mocked)
# ===========================================================================

class TestSubmitRunOperationLocal:
    """The local path runs once (no retry — local execution is deterministic) and
    returns the LocalRunner result verbatim, forwarding kwargs as-is."""

    def test_runs_once_no_retry(self):
        from modal_controller.utils import submit_run_operation
        from modal_controller.modal_runner import Operation
        mock_runner = MagicMock()
        mock_runner.run_operation.return_value = {"error": "boom"}
        with patch("modal_controller.utils._get_local_runner", return_value=mock_runner):
            result = submit_run_operation(
                Operation.EVALUATE, '{"plan": "test"}', "/data", exec_local=True, n_retry=5,
            )
        assert result == {"error": "boom"}
        assert mock_runner.run_operation.call_count == 1  # n_retry ignored locally

    def test_returns_success_verbatim(self):
        from modal_controller.utils import submit_run_operation
        from modal_controller.modal_runner import Operation
        mock_runner = MagicMock()
        mock_runner.run_operation.return_value = {"execution_time": 0.1}
        with patch("modal_controller.utils._get_local_runner", return_value=mock_runner):
            result = submit_run_operation(
                Operation.EVALUATE, '{"plan": "test"}', "/data", exec_local=True,
            )
        assert result == {"execution_time": 0.1}
        assert mock_runner.run_operation.call_count == 1

    def test_forwards_kwargs_to_runner(self):
        from modal_controller.utils import submit_run_operation
        from modal_controller.modal_runner import Operation
        mock_runner = MagicMock()
        mock_runner.run_operation.return_value = {"execution_time": 0.1}
        with patch("modal_controller.utils._get_local_runner", return_value=mock_runner):
            submit_run_operation(
                Operation.EVALUATE, '{"plan": "test"}', "/data", exec_local=True,
                full_metrics=True, sandbox_placeholders={"FULL_METRICS": "True"},
            )
        call = mock_runner.run_operation.call_args
        assert call.kwargs["operation"] == Operation.EVALUATE
        assert call.kwargs["input_str"] == '{"plan": "test"}'
        assert call.kwargs["data_folder"] == "/data"
        assert call.kwargs["full_metrics"] is True
        assert call.kwargs["sandbox_placeholders"] == {"FULL_METRICS": "True"}


# ===========================================================================
# evaluate_sampled_plans — patch-mode branching (remote calls mocked)
# ===========================================================================

class TestEvaluateSampledPlansBranching:
    def _write(self, tmp_path, sampled_plan):
        input_data = [{
            "id": 0, "query": "SELECT 1",
            "plan": {"structure": {"0": {"scan": {}}}, "succinct_table_info": {}, "full_table_info": {}},
            "sampled_plans": [sampled_plan],
        }]
        input_file = tmp_path / "input.json"
        output_file = tmp_path / "output.json"
        input_file.write_text(json.dumps(input_data))
        return str(input_file), str(output_file)

    def test_empty_patches_skipped(self, tmp_path):
        from sampling.evaluate_sampled_plans import evaluate_sampled_plans
        inp, out = self._write(tmp_path, {
            "sample_id": 1, "parent_sample_id": None, "upstream_patches": [],
            "sampled_patches": [], "upstream_evaluation_stats": {"execution_time": 0.5},
            "is_valid": True, "error_message": None,
        })
        with patch("sampling.evaluate_sampled_plans.validate_plan_result_set") as mock_val, \
             patch("sampling.evaluate_sampled_plans.evaluate_plan_n_runs") as mock_eval:
            evaluate_sampled_plans(inp, out, "tpch", 1, 1, False, exec_local=True)
            mock_val.assert_not_called()
            mock_eval.assert_not_called()
        plan = json.loads(open(out).read())[0]["sampled_plans"][0]
        assert plan["is_valid"] is True
        assert plan["evaluation_stats"] == {"execution_time": 0.5}

    def test_already_invalid_skipped(self, tmp_path):
        from sampling.evaluate_sampled_plans import evaluate_sampled_plans
        inp, out = self._write(tmp_path, {
            "sample_id": 1, "parent_sample_id": None, "upstream_patches": [],
            "sampled_patches": [{"op": "replace", "path": "/0/scan", "value": {}}],
            "is_valid": False, "error_message": "LLM failure",
        })
        with patch("sampling.evaluate_sampled_plans.validate_plan_result_set") as mock_val, \
             patch("sampling.evaluate_sampled_plans.evaluate_plan_n_runs") as mock_eval:
            evaluate_sampled_plans(inp, out, "tpch", 1, 1, False, exec_local=True)
            mock_val.assert_not_called()
            mock_eval.assert_not_called()

    def test_validation_failure_marks_invalid(self, tmp_path):
        from sampling.evaluate_sampled_plans import evaluate_sampled_plans
        inp, out = self._write(tmp_path, {
            "sample_id": 1, "parent_sample_id": None, "upstream_patches": [],
            "sampled_patches": [{"op": "replace", "path": "/0/scan", "value": {"t": 1}}],
            "is_valid": True, "error_message": None,
        })
        with patch("sampling.evaluate_sampled_plans.validate_plan_result_set",
                   return_value="Result set mismatch: candidate differs"), \
             patch("sampling.evaluate_sampled_plans.evaluate_plan_n_runs") as mock_eval:
            evaluate_sampled_plans(inp, out, "tpch", 1, 1, False, exec_local=True)
            mock_eval.assert_not_called()
        plan = json.loads(open(out).read())[0]["sampled_plans"][0]
        assert plan["is_valid"] is False
        assert "Result set mismatch" in plan["error_message"]

    def test_patch_apply_failure_recorded(self, tmp_path):
        from sampling.evaluate_sampled_plans import evaluate_sampled_plans
        inp, out = self._write(tmp_path, {
            "sample_id": 1, "parent_sample_id": None, "upstream_patches": [],
            "sampled_patches": [{"op": "replace", "path": "/nonexistent/deep/path", "value": 1}],
            "is_valid": True, "error_message": None,
        })
        with patch("sampling.evaluate_sampled_plans.validate_plan_result_set") as mock_val:
            evaluate_sampled_plans(inp, out, "tpch", 1, 1, False, exec_local=True)
            mock_val.assert_not_called()  # fails before validation
        plan = json.loads(open(out).read())[0]["sampled_plans"][0]
        assert plan["is_valid"] is False
        assert "Failed to apply patches" in plan["error_message"]

    def test_null_sampled_patches_caught(self, tmp_path):
        from sampling.evaluate_sampled_plans import evaluate_sampled_plans
        inp, out = self._write(tmp_path, {
            "sample_id": 1, "parent_sample_id": None, "upstream_patches": [],
            "sampled_patches": None, "is_valid": True, "error_message": None,
        })
        evaluate_sampled_plans(inp, out, "tpch", 1, 1, False, exec_local=True)
        plan = json.loads(open(out).read())[0]["sampled_plans"][0]
        assert plan["is_valid"] is False
        assert "sampled_patches is None" in plan["error_message"]


# ===========================================================================
# op_* operation contracts (base_operation, mocked DB)
# ===========================================================================

def _mock_query_result(data=None, is_error=False, error_msg=""):
    import pyarrow as pa
    QueryResult = namedtuple("QueryResult", ["data", "is_exec_error", "error_message"])
    if data is None:
        data = pa.table({"col": [1, 2, 3]})
    return QueryResult(data=data, is_exec_error=is_error, error_message=error_msg)


class TestOpContracts:
    def test_execute_returns_result_data(self):
        from modal_controller.operations.base_operation import op_execute
        db = MagicMock()
        db.execute_serialized_physical_plan.return_value = _mock_query_result()
        result = op_execute(db, '{"plan": "test"}')
        assert "result_data" in result and "execution_time" in result and "schema" in result
        assert isinstance(result["result_data"], str)

    def test_evaluate_does_not_return_result_data(self):
        from modal_controller.operations.base_operation import op_evaluate
        db = MagicMock()
        db.execute_serialized_physical_plan.return_value = _mock_query_result()
        result = op_evaluate(db, '{"plan": "test"}')
        assert "result_data" not in result and "execution_time" in result

    def test_execute_error_propagation(self):
        from modal_controller.operations.base_operation import op_execute
        db = MagicMock()
        db.execute_serialized_physical_plan.return_value = _mock_query_result(
            is_error=True, error_msg="column not found")
        result = op_execute(db, '{"plan": "test"}')
        assert "error" in result and "column not found" in result["error"]


# ===========================================================================
# LocalRunner.run_operation — dispatch + error handling (needs datafusion)
# ===========================================================================

@pytest.mark.local
class TestLocalRunnerRunOperation:
    def test_run_operation_unsupported(self):
        from modal_controller.local_runner import LocalRunner
        result = LocalRunner().run_operation("UNKNOWN_OP", "", "/data")
        assert "error" in result and "Unsupported" in result["error"]

    def test_run_operation_exception_caught(self):
        from modal_controller.local_runner import LocalRunner
        from modal_controller.modal_runner import Operation
        with patch("modal_controller.local_runner.DataFusionDB", side_effect=RuntimeError("boom")):
            result = LocalRunner().run_operation(Operation.PLAN, "SELECT 1", "/data")
        assert "error" in result and "boom" in result["error"]


# ===========================================================================
# GenerationResult — sampled_patches type flexibility
# ===========================================================================

class TestGenerationResultTypeFlexibility:
    def test_list_type(self):
        from sampling.gpt_plan_optimizer import GenerationResult
        gr = GenerationResult(
            model_response="", reasoning_content="", error_message=None,
            sampled_patches=[{"op": "replace", "path": "/0/x", "value": 1}],
        )
        assert isinstance(gr.sampled_patches, list)

    def test_none_type(self):
        from sampling.gpt_plan_optimizer import GenerationResult
        gr = GenerationResult(
            model_response="", reasoning_content="", error_message="LLM failed",
            sampled_patches=None,
        )
        assert gr.sampled_patches is None
