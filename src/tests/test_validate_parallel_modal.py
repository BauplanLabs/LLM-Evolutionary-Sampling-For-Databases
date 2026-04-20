"""Real-Modal integration tests for the parallelized ``validate_plan_result_set``.

Unlike ``test_validate_parallel.py`` (mocks ``submit_run_operation``), these
tests run the full production stack:

  - real Modal Sandbox creates (1 baseline EXECUTE + N candidate EXECUTEs)
  - real S3 result_data fetch
  - real ``compare_result_sets`` (float-aware pandas comparator)

Each test is marked ``@pytest.mark.modal`` and is SKIPPED by default. Run:

    pytest src/tests/test_validate_parallel_modal.py -m modal -v -s

Prerequisites: ``.env`` with AWS creds + ``S3_BUCKET_NAME`` + Modal auth.

Fixture plans are generated on the fly against TPC-H at scale_factor=1
(same pattern as ``test_integration.py``), so no external artifacts are
required.
"""
from __future__ import annotations

import pytest

_DATASET = "tpch"
_SCALE_FACTOR = 1
_DATA_FOLDER = "/tmp/data/data_tpch"
_RUNNER_KWARGS = {"scale_factor": _SCALE_FACTOR}

# Tiny queries on the smallest TPC-H tables (nation=25 rows, region=5 rows).
_Q0_SQL = "SELECT n_nationkey, n_name FROM nation LIMIT 5"
_Q1_SQL = "SELECT r_regionkey, r_name FROM region"

# Deliberately malformed plan_str — valid JSON, invalid as a DataFusion plan.
# EXECUTE will fail inside the sandbox and surface as an error result.
_BROKEN_PLAN_STR = '{"this": "is not a valid datafusion plan"}'


pytestmark = pytest.mark.modal


# ---------------------------------------------------------------------------
# Fixtures: real plan strings generated from tiny TPC-H queries.
# ---------------------------------------------------------------------------

def _plan_str_for(query: str) -> str:
    from dbplanbench import get_engine_plans
    from dbplanbench_utils import plan_to_json
    result = get_engine_plans(
        [query], dataset=_DATASET, scale_factor=_SCALE_FACTOR, verbose=False,
    )
    assert result.plans[0] is not None, f"Planning failed: {result.errors[0]}"
    return plan_to_json(result.plans[0].base_plan)


@pytest.fixture(scope="module")
def q0_baseline_plan_str():
    return _plan_str_for(_Q0_SQL)


@pytest.fixture(scope="module")
def q1_baseline_plan_str():
    return _plan_str_for(_Q1_SQL)


# ---------------------------------------------------------------------------
# Tests (all hit real Modal).
# ---------------------------------------------------------------------------

def test_identical_plans_all_retries_match(q0_baseline_plan_str):
    """baseline == candidate (same plan string). Real 4 EXECUTEs on Modal, real
    S3 fetch, real compare_result_sets. Expected: None (success).

    Exercises:
      - UUID plumbing through submit_run_operation
      - S3 round-trip for 4 result_data payloads
      - baseline↔primary-candidate correctness compare
      - 2× primary-candidate↔other-candidate determinism compares
    """
    from modal_controller.utils import validate_plan_result_set
    result = validate_plan_result_set(
        q0_baseline_plan_str,
        q0_baseline_plan_str,
        _DATA_FOLDER,
        n_determinism_retries=3,
        runner_kwargs=_RUNNER_KWARGS,
    )
    assert result is None, f"Expected None, got: {result!r}"


def test_identical_plans_n_retries_5(q0_baseline_plan_str):
    """Same as above but with n_determinism_retries=5 (6 sandboxes total).

    Stresses the parallel pool beyond the default and confirms the
    streaming compare correctly handles 4 determinism compares + 1
    correctness compare in any arrival order.
    """
    from modal_controller.utils import validate_plan_result_set
    result = validate_plan_result_set(
        q0_baseline_plan_str,
        q0_baseline_plan_str,
        _DATA_FOLDER,
        n_determinism_retries=5,
        runner_kwargs=_RUNNER_KWARGS,
    )
    assert result is None, f"Expected None, got: {result!r}"


def test_different_queries_mismatch(q0_baseline_plan_str, q1_baseline_plan_str):
    """baseline = q0 plan, candidate = q1 plan. Both execute successfully on
    Modal but produce different result sets. compare_result_sets must flag
    it as 'Result set mismatch'.
    """
    from modal_controller.utils import validate_plan_result_set
    result = validate_plan_result_set(
        q1_baseline_plan_str,       # candidate_plan_str
        q0_baseline_plan_str,       # baseline_plan_str
        _DATA_FOLDER,
        n_determinism_retries=3,
        runner_kwargs=_RUNNER_KWARGS,
    )
    assert result is not None, "Expected mismatch error, got None"
    assert "Result set mismatch" in result, f"Expected mismatch, got: {result!r}"


def test_candidate_execution_failure(q0_baseline_plan_str):
    """Valid baseline, malformed candidate plan. Candidate EXECUTE must fail
    inside the sandbox and surface as 'Candidate execution failed'.
    """
    from modal_controller.utils import validate_plan_result_set
    result = validate_plan_result_set(
        _BROKEN_PLAN_STR,           # candidate_plan_str
        q0_baseline_plan_str,       # baseline_plan_str
        _DATA_FOLDER,
        n_determinism_retries=3,
        runner_kwargs=_RUNNER_KWARGS,
    )
    assert result is not None, "Expected failure, got None"
    assert "Candidate execution failed" in result, (
        f"Expected 'Candidate execution failed', got: {result!r}"
    )


def test_baseline_execution_failure(q0_baseline_plan_str):
    """Malformed baseline, valid candidate. Baseline EXECUTE must fail and
    surface as 'Baseline execution failed'. Since the baseline fails, the
    fact that the candidate succeeds is irrelevant — the validator must
    still classify the whole call as a baseline failure.
    """
    from modal_controller.utils import validate_plan_result_set
    result = validate_plan_result_set(
        q0_baseline_plan_str,       # candidate_plan_str
        _BROKEN_PLAN_STR,           # baseline_plan_str
        _DATA_FOLDER,
        n_determinism_retries=3,
        runner_kwargs=_RUNNER_KWARGS,
    )
    assert result is not None, "Expected failure, got None"
    assert "Baseline execution failed" in result, (
        f"Expected 'Baseline execution failed', got: {result!r}"
    )
