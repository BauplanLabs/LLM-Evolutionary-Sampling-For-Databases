"""Kwarg/flag propagation contract for submit_run_operation (both transports).

These are the regression tests for the class of bug where a flag silently fails
to reach one runner. They mock the runner factories so no Modal sandbox, S3, or
DataFusion engine is needed — they assert only that submit_run_operation hands
the right arguments to ModalRunner.run_operation / LocalRunner.run_operation.

Background: full_metrics / include_sample_rows reach the Modal operation scripts
as injected ``sandbox_placeholders`` string tokens (FULL_METRICS_HERE, ...),
whereas LocalRunner takes them as bool params. submit_run_operation's local
branch translates the placeholder encoding (and any direct kwarg) to those
params; the Modal branch forwards the placeholders dict unchanged.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from modal_controller.modal_runner import Operation


# ---------------------------------------------------------------------------
# Local path: flags must reach LocalRunner.run_operation as bool params.
# ---------------------------------------------------------------------------

def _run_local(**call_kwargs):
    """Invoke submit_run_operation(exec_local=True) with a mocked LocalRunner.

    Returns the runner mock for assertions on what run_operation received.
    """
    from modal_controller import utils
    runner = MagicMock()
    runner.run_operation.return_value = {"ok": True}
    with patch.object(utils, "_get_local_runner", return_value=runner):
        utils.submit_run_operation(
            Operation.EVALUATE, "plan-json", "/data", exec_local=True, **call_kwargs
        )
    return runner


class TestLocalDispatch:
    """submit_run_operation forwards kwargs to LocalRunner.run_operation as-is
    (symmetric with the Modal path); LocalRunner itself derives cpu and reads
    the flags from sandbox_placeholders (covered in test_local_runner.py)."""

    def test_sandbox_placeholders_forwarded(self):
        runner = _run_local(sandbox_placeholders={"FULL_METRICS": True})
        assert runner.run_operation.call_args.kwargs["sandbox_placeholders"] == {"FULL_METRICS": True}

    def test_cpu_forwarded(self):
        runner = _run_local(cpu=(8, 8))
        assert runner.run_operation.call_args.kwargs["cpu"] == (8, 8)

    def test_modal_only_kwargs_forwarded_without_error(self):
        # memory / sandbox_timeout etc. apply only to Modal; forwarding is fine
        # because LocalRunner.run_operation ignores what it doesn't use.
        runner = _run_local(memory=(8 * 1024, 8 * 1024), sandbox_timeout=600)
        assert runner.run_operation.called


# ---------------------------------------------------------------------------
# Modal path: kwargs (placeholders + cpu) must reach ModalRunner.run_operation.
# ---------------------------------------------------------------------------

def _run_modal(**call_kwargs):
    from modal_controller import utils
    runner = MagicMock()
    runner.run_operation.return_value = "uuid-123"  # run_attempt expects a uuid str
    with patch.object(utils, "_get_runner", return_value=runner), \
         patch.object(utils, "_gate_modal_start"), \
         patch.object(utils, "read_uuid_result", return_value={"ok": True}):
        utils.submit_run_operation(
            Operation.EVALUATE, "plan-json", "/tmp/data/data_tpch",
            exec_local=False, **call_kwargs,
        )
    return runner


class TestModalFlagPropagation:
    def test_sandbox_placeholders_forwarded_unchanged(self):
        runner = _run_modal(sandbox_placeholders={"FULL_METRICS": True})
        assert runner.run_operation.call_args.kwargs["sandbox_placeholders"] == {"FULL_METRICS": True}

    def test_cpu_forwarded_unchanged(self):
        runner = _run_modal(cpu=(8, 8))
        assert runner.run_operation.call_args.kwargs["cpu"] == (8, 8)
