"""Tests for the parallelized ``validate_plan_result_set`` (Fix 1).

The implementation submits 1 baseline EXECUTE + N candidate EXECUTEs in
parallel via a per-call ThreadPoolExecutor and stream-compares results as
they arrive. These tests verify:

  - Behavior contract preserved: same return semantics as the prior
    sequential implementation across success / mismatch / error paths.
  - Compare-as-you-fetch correctness under both arrival orders
    (baseline-first and baseline-last).
  - Real parallelism: wall-clock < N × per-call latency.
  - Argument propagation (n_retry, kwargs, plan strings).

All mocks key off the *plan_str* argument so tests are insensitive to the
nondeterministic thread-arrival order of the parallel implementation.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional
from unittest.mock import patch

import pytest

CAND = '{"plan": "candidate"}'
BASE = '{"plan": "baseline"}'

DATA_OK = '[{"a": 1, "b": 2.0}]'
DATA_ALT = '[{"a": 999, "b": 2.0}]'


def _by_plan(side_effects: Dict[str, Any]) -> Callable[..., Any]:
    """Return a side_effect callable that dispatches on the plan_str arg.

    *side_effects* maps plan_str -> either a single dict (returned every
    call) or a list/iterator of dicts (consumed in order of calls for that
    plan_str). Thread-safe via a lock.
    """
    lock = threading.Lock()
    iters: Dict[str, Any] = {}
    for k, v in side_effects.items():
        if isinstance(v, list):
            iters[k] = iter(v)
        else:
            iters[k] = v

    def _impl(*args, **kwargs):
        # submit_run_operation(operation, input_str, data_folder, ...)
        plan_str = args[1] if len(args) > 1 else kwargs.get("input_str")
        with lock:
            entry = iters[plan_str]
            if isinstance(entry, dict):
                # Return a fresh shallow copy so result_data pops don't poison
                # later calls.
                return dict(entry)
            return dict(next(entry))

    return _impl


# ---------------------------------------------------------------------------
# Behavior contract
# ---------------------------------------------------------------------------

class TestBehaviorContract:
    def test_all_match_returns_none(self):
        """Baseline + 3 identical candidate runs => None (success)."""
        from modal_controller.utils import validate_plan_result_set
        with patch("modal_controller.utils.submit_run_operation",
                   side_effect=_by_plan({
                       BASE: {"result_data": DATA_OK},
                       CAND: {"result_data": DATA_OK},
                   })):
            result = validate_plan_result_set(
                CAND, BASE, "/data", n_determinism_retries=3,
            )
            assert result is None

    def test_baseline_failure_message(self):
        """Baseline EXECUTE fails => 'Baseline execution failed' in error."""
        from modal_controller.utils import validate_plan_result_set
        with patch("modal_controller.utils.submit_run_operation",
                   side_effect=_by_plan({
                       BASE: {"error": "boom: bad plan"},
                       CAND: {"result_data": DATA_OK},
                   })):
            result = validate_plan_result_set(
                CAND, BASE, "/data", n_determinism_retries=3,
            )
            assert result is not None and "Baseline execution failed" in result

    def test_candidate_failure_message(self):
        """Any candidate EXECUTE fails => 'Candidate execution failed' in error."""
        from modal_controller.utils import validate_plan_result_set
        with patch("modal_controller.utils.submit_run_operation",
                   side_effect=_by_plan({
                       BASE: {"result_data": DATA_OK},
                       CAND: {"error": "exec mismatch"},
                   })):
            result = validate_plan_result_set(
                CAND, BASE, "/data", n_determinism_retries=3,
            )
            assert result is not None and "Candidate execution failed" in result

    def test_result_set_mismatch(self):
        """Baseline data != candidate data => 'Result set mismatch' in error."""
        from modal_controller.utils import validate_plan_result_set
        with patch("modal_controller.utils.submit_run_operation",
                   side_effect=_by_plan({
                       BASE: {"result_data": DATA_OK},
                       CAND: {"result_data": DATA_ALT},
                   })):
            result = validate_plan_result_set(
                CAND, BASE, "/data", n_determinism_retries=3,
            )
            assert result is not None and "Result set mismatch" in result

    def test_non_determinism_one_cand_differs(self):
        """One of the candidate runs differs from the others => 'Non-deterministic'."""
        from modal_controller.utils import validate_plan_result_set
        # Two cand runs return DATA_OK, one returns DATA_ALT. Whichever pair
        # is compared, we should detect non-determinism (the result is
        # 'Non-deterministic' if the differing one is NOT the primary, or
        # 'Result set mismatch' if it IS the primary — both are valid 'fail'
        # outcomes; we accept either error string here).
        cand_seq = [
            {"result_data": DATA_OK},
            {"result_data": DATA_OK},
            {"result_data": DATA_ALT},
        ]
        with patch("modal_controller.utils.submit_run_operation",
                   side_effect=_by_plan({
                       BASE: {"result_data": DATA_OK},
                       CAND: cand_seq,
                   })):
            result = validate_plan_result_set(
                CAND, BASE, "/data", n_determinism_retries=3,
            )
            assert result is not None
            assert ("Non-deterministic" in result) or ("Result set mismatch" in result)

    def test_exception_caught_returns_validation_failed(self):
        """Exception inside submit_run_operation surfaces as 'Validation failed'."""
        from modal_controller.utils import validate_plan_result_set
        with patch("modal_controller.utils.submit_run_operation",
                   side_effect=RuntimeError("connection lost")):
            result = validate_plan_result_set(
                CAND, BASE, "/data", n_determinism_retries=3,
            )
            assert result is not None and "Validation failed" in result

    def test_none_result_treated_as_failure(self):
        """submit_run_operation returning None is reported as a failure."""
        from modal_controller.utils import validate_plan_result_set
        with patch("modal_controller.utils.submit_run_operation", return_value=None):
            result = validate_plan_result_set(
                CAND, BASE, "/data", n_determinism_retries=3,
            )
            assert result is not None and "failed" in result.lower()


# ---------------------------------------------------------------------------
# Compare-as-you-fetch ordering — both extreme arrival orders
# ---------------------------------------------------------------------------

class TestArrivalOrder:
    def test_baseline_arrives_last_success(self):
        """Slow baseline + fast cands. Baseline arrives last; correctness
        compare must still happen (against the already-stashed primary)."""
        from modal_controller.utils import validate_plan_result_set
        cand_done = threading.Event()
        cand_count = {"n": 0}
        cand_lock = threading.Lock()

        def side(*args, **kwargs):
            plan_str = args[1] if len(args) > 1 else kwargs.get("input_str")
            if plan_str == BASE:
                # Wait until at least 2 cand calls have completed
                cand_done.wait(timeout=2.0)
                return {"result_data": DATA_OK}
            else:
                with cand_lock:
                    cand_count["n"] += 1
                    n = cand_count["n"]
                if n >= 2:
                    cand_done.set()
                return {"result_data": DATA_OK}

        with patch("modal_controller.utils.submit_run_operation", side_effect=side):
            result = validate_plan_result_set(
                CAND, BASE, "/data", n_determinism_retries=3,
            )
            assert result is None

    def test_baseline_arrives_first_success(self):
        """Fast baseline + slow cands. Baseline gets stashed; first arriving
        cand triggers the correctness compare."""
        from modal_controller.utils import validate_plan_result_set
        base_done = threading.Event()

        def side(*args, **kwargs):
            plan_str = args[1] if len(args) > 1 else kwargs.get("input_str")
            if plan_str == BASE:
                base_done.set()
                return {"result_data": DATA_OK}
            else:
                # Cands wait until baseline is done
                base_done.wait(timeout=2.0)
                return {"result_data": DATA_OK}

        with patch("modal_controller.utils.submit_run_operation", side_effect=side):
            result = validate_plan_result_set(
                CAND, BASE, "/data", n_determinism_retries=3,
            )
            assert result is None

    def test_baseline_arrives_last_mismatch(self):
        """Baseline arrives last AND data differs => mismatch must still be
        detected (the deferred correctness compare must fire)."""
        from modal_controller.utils import validate_plan_result_set
        cand_done = threading.Event()
        cand_count = {"n": 0}
        cand_lock = threading.Lock()

        def side(*args, **kwargs):
            plan_str = args[1] if len(args) > 1 else kwargs.get("input_str")
            if plan_str == BASE:
                cand_done.wait(timeout=2.0)
                return {"result_data": DATA_OK}
            else:
                with cand_lock:
                    cand_count["n"] += 1
                    n = cand_count["n"]
                if n >= 2:
                    cand_done.set()
                return {"result_data": DATA_ALT}

        with patch("modal_controller.utils.submit_run_operation", side_effect=side):
            result = validate_plan_result_set(
                CAND, BASE, "/data", n_determinism_retries=3,
            )
            assert result is not None and "Result set mismatch" in result


# ---------------------------------------------------------------------------
# Real parallelism — wall-clock contract
# ---------------------------------------------------------------------------

class TestParallelism:
    def test_wall_clock_proves_parallel_execution(self):
        """4 sandboxes each sleeping 0.4s should finish in < 1.0s wall-clock,
        not the ~1.6s a sequential implementation would produce."""
        from modal_controller.utils import validate_plan_result_set

        SLEEP = 0.4

        def side(*args, **kwargs):
            time.sleep(SLEEP)
            return {"result_data": DATA_OK}

        with patch("modal_controller.utils.submit_run_operation", side_effect=side):
            t0 = time.perf_counter()
            result = validate_plan_result_set(
                CAND, BASE, "/data", n_determinism_retries=3,
            )
            elapsed = time.perf_counter() - t0
            assert result is None
            # Sequential lower bound: 4*SLEEP = 1.6s. Parallel upper bound:
            # ~1*SLEEP + thread/scheduling overhead. 1.0s leaves comfy margin.
            assert elapsed < 1.0, f"validate ran in {elapsed:.3f}s; expected <1.0s"


# ---------------------------------------------------------------------------
# Argument propagation
# ---------------------------------------------------------------------------

class TestArgumentPropagation:
    def test_default_n_retry_is_5(self):
        """validate_plan_result_set must pass n_retry=5 to submit_run_operation
        by default (Fix 3 documentation contract)."""
        from modal_controller.utils import validate_plan_result_set
        seen_n_retries = []
        lock = threading.Lock()

        def side(*args, **kwargs):
            with lock:
                seen_n_retries.append(kwargs.get("n_retry"))
            return {"result_data": DATA_OK}

        with patch("modal_controller.utils.submit_run_operation", side_effect=side):
            validate_plan_result_set(
                CAND, BASE, "/data", n_determinism_retries=3,
            )
            assert all(n == 5 for n in seen_n_retries), seen_n_retries
            assert len(seen_n_retries) == 4  # 1 baseline + 3 cands

    def test_override_n_retry_propagated(self):
        """Caller-provided n_retry must override the default."""
        from modal_controller.utils import validate_plan_result_set
        seen = []
        lock = threading.Lock()

        def side(*args, **kwargs):
            with lock:
                seen.append(kwargs.get("n_retry"))
            return {"result_data": DATA_OK}

        with patch("modal_controller.utils.submit_run_operation", side_effect=side):
            validate_plan_result_set(
                CAND, BASE, "/data", n_retry=2, n_determinism_retries=3,
            )
            assert all(n == 2 for n in seen)

    def test_correct_call_count_per_plan(self):
        """Exactly 1 baseline call + n_determinism_retries candidate calls."""
        from modal_controller.utils import validate_plan_result_set
        plan_calls: Dict[str, int] = {BASE: 0, CAND: 0}
        lock = threading.Lock()

        def side(*args, **kwargs):
            plan_str = args[1] if len(args) > 1 else kwargs.get("input_str")
            with lock:
                plan_calls[plan_str] += 1
            return {"result_data": DATA_OK}

        with patch("modal_controller.utils.submit_run_operation", side_effect=side):
            validate_plan_result_set(
                CAND, BASE, "/data", n_determinism_retries=3,
            )
            assert plan_calls[BASE] == 1
            assert plan_calls[CAND] == 3

    def test_n_determinism_retries_one(self):
        """Minimal config: 1 cand run only. Baseline + 1 cand = 2 sandboxes."""
        from modal_controller.utils import validate_plan_result_set
        with patch("modal_controller.utils.submit_run_operation",
                   side_effect=_by_plan({
                       BASE: {"result_data": DATA_OK},
                       CAND: {"result_data": DATA_OK},
                   })):
            result = validate_plan_result_set(
                CAND, BASE, "/data", n_determinism_retries=1,
            )
            assert result is None
