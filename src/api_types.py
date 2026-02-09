from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


Patch = List[Dict[str, Any]]
"""A single JSON Patch: list of RFC 6902 operations."""

PatchSet = List[Optional[Patch]]
"""List of candidate patches. Each element is either a Patch (list of ops),
an empty list ``[]`` (no-op / base plan), or ``None`` (patch unavailable).
A single-patch result is ``[patch]``, top-k results are ``[p1, p2, ...]``."""


@dataclass
class QueryGenerationResult:
    """Returned by ``generate_queries``.

    Attributes:
        run_dir: Directory containing outcome files and config.
        summary: Overall and per-complexity generation/validation counts.
        queries: Final list of valid SQL query strings.
        metadata: Per-query metadata aligned to *queries* (same length).
    """

    run_dir: str
    summary: Dict[str, Any]
    queries: List[str]
    metadata: List[Dict[str, Any]]


@dataclass
class PatchedPlan:
    """A base execution plan paired with one or more candidate patches.

    Attributes:
        base_plan: The base physical plan dict.
        patch: List of candidate patches. Each element is a Patch (list of
            ops), ``[]`` (no-op), or ``None`` (unavailable). For example:
            ``[[]]`` for a base plan only, ``[patch]`` for one optimization,
            ``[p1, p2]`` for top-2 candidates.
    """

    base_plan: Dict[str, Any]
    patch: PatchSet


@dataclass
class OptimizationResult:
    """Returned by ``optimize_queries``.

    Attributes:
        run_dir: Directory containing result.json and sampling artifacts.
        summary: Validation stats, run config, improvement/outcome metrics.
        queries: SQL queries optimized (same order as inputs).
        optimization_outcome: Per-query PatchedPlan list (None on early exit).
        metadata: Per-query benchmark stats (None on early exit).
        validation_failures: Per-query error strings (None for valid queries).
            When any entry is non-None, optimization does not proceed.
    """

    run_dir: Optional[str]
    summary: Dict[str, Any]
    queries: Optional[List[str]]
    optimization_outcome: Optional[List[PatchedPlan]]
    metadata: Optional[List[Dict[str, Any]]]
    validation_failures: Optional[List[Optional[str]]]


@dataclass
class BenchmarkResult:
    """Returned by ``benchmark_plans``.

    Attributes:
        results: List-of-lists aligned to input PatchedPlans and their
            patches. Each inner dict contains evaluation metrics.
        output_file: Path where results were written (if requested).
    """

    results: List[List[Dict[str, Any]]]
    output_file: Optional[str] = None


@dataclass
class ScaleResult:
    """Returned by ``scale_optimizations``.

    Attributes:
        run_dir: Directory containing result.json.
        summary: Config, outcome rates, per-query statuses.
        queries: SQL queries (same order as inputs).
        scaled_plans: Per-query PatchedPlan at target_scale_factor. base_plan
            holds the transferred SF1 base (with SF2 succinct_table_info),
            patch holds the transferred optimization delta. For failed
            queries, base_plan is the SF1 base and patch is ``[None]``.
        metadata: Per-query transfer details (per-patch statuses/errors).
        transfer_failures: Per-query error string (None = all patches OK).
    """

    run_dir: Optional[str]
    summary: Dict[str, Any]
    queries: List[str]
    scaled_plans: List[PatchedPlan]
    metadata: List[Dict[str, Any]]
    transfer_failures: List[Optional[str]]


@dataclass
class PlanningResult:
    """Returned by ``get_engine_plans``.

    Attributes:
        plans: Per-query engine plan dict (None on planning failure).
        errors: Per-query error string (None on success).
    """

    plans: List[Optional[Dict[str, Any]]]
    errors: List[Optional[str]]


@dataclass
class QueryValidationResult:
    """Returned by ``validate_queries``.

    Attributes:
        plans: Per-query engine plan dict (None for invalid queries).
            Valid plans can be fed directly to ``benchmark_plans``.
        errors: Per-query error category (None for valid queries).
            Categories: ``syntax_error``, ``no_plan``, ``cannot_run``,
            ``empty_result``, ``nondeterministic``, or a descriptive message.
        summary: Aggregate validation counts (``n_queries``, ``n_valid``,
            and per-category failure counts).
    """

    plans: List[Optional[Dict[str, Any]]]
    errors: List[Optional[str]]
    summary: Dict[str, Any]
