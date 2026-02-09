from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from sampling.prepare_sampling import prepare_sampling
from sampling.sample_plans import sample_plans_from_file, SamplingStrategy, ORIGINAL_STRATEGY
from sampling.evaluate_sampled_plans import evaluate_sampled_plans
from sampling.accumulate_samples import accumulate_samples
from api_utils import log_line, get_metric_value, format_metric_stats
from modal_controller.utils import compute_metric_stats


def _log_step_improvements(
    final_sampling_path: Path,
    optimization_metric: str,
    step_idx: int,
    n_steps: int,
    verbose: bool,
) -> None:
    """Compute and print per-query best improvement_x after a step."""
    if not verbose or not final_sampling_path.exists():
        return

    try:
        sampling_data = json.loads(final_sampling_path.read_text())
    except Exception:
        return

    improvement_values: List[float] = []
    for entry in sampling_data:
        base_metric = None
        best_metric = None
        for plan_info in entry.get("sampled_plans", []):
            eval_stats = plan_info.get("evaluation_stats")
            metric_val = get_metric_value(eval_stats, optimization_metric)
            if plan_info.get("plan_type") == "original":
                base_metric = metric_val
            if metric_val is not None and plan_info.get("is_valid", False):
                if best_metric is None or metric_val < best_metric:
                    best_metric = metric_val
        if base_metric is not None and best_metric is not None and best_metric > 0:
            improvement_values.append(base_metric / best_metric)

    if improvement_values:
        stats = compute_metric_stats(improvement_values)
        log_line(verbose, f"  [Step {step_idx}/{n_steps}] Best improvement_x per query so far:")
        log_line(verbose, f"    {format_metric_stats(stats)}")


def _step_evaluated_path(step_dir: Path) -> Path:
    """Return the path to the evaluated sampled-plans file for a step directory."""
    return step_dir / "evaluated_sampled_plans.json"


def _list_completed_steps(sampling_dir: Path) -> List[int]:
    """Return sorted list of step indices that have evaluated output files."""
    steps = []
    for path in sampling_dir.glob("step_*"):
        if _step_evaluated_path(path).exists():
            try:
                steps.append(int(path.name.split("_")[-1]))
            except Exception:
                continue
    return sorted(set(steps))


def _select_strategy(
    strategy: str,
    step_idx: int,
    n_samples_per_step: int,
    optimization_metric: str,
) -> tuple[SamplingStrategy, int]:
    """Pick the SamplingStrategy and effective sample count for a given step.

    Args:
        strategy: High-level strategy name ("bol_evol", "pst_evol", or "best_of").
        step_idx: 1-based step index.
        n_samples_per_step: Requested samples per step.
        optimization_metric: Metric path for ranking (used by bol_evol).

    Returns:
        Tuple of (SamplingStrategy, effective_n_samples).
    """
    if strategy == "pst_evol":
        if step_idx == 1:
            return ORIGINAL_STRATEGY, n_samples_per_step
        return (
            SamplingStrategy(
                starting_selector="of_last",
                final_selector="all",
                from_valid_only=False,
            ),
            1,
        )
    if strategy == "bol_evol":
        return (
            SamplingStrategy(
                starting_selector="of_last",
                final_selector="best_1",
                from_valid_only=False,
                kwargs={"optimization_metric": optimization_metric},
            ),
            n_samples_per_step,
        )
    return ORIGINAL_STRATEGY, n_samples_per_step


def run_sampling_steps(
    *,
    sampling_dir: Path,
    base_sampling_path: Path,
    final_sampling_path: Path,
    dataset: str,
    model: str,
    strategy: str,
    n_steps: int,
    n_samples_per_step: int,
    max_sampling_workers: int,
    max_workers: int,
    optimization_metric: str,
    resume: bool,
    sampling_kwargs: Dict[str, Any],
    evaluation_kwargs: Dict[str, Any],
    verbose: bool,
) -> None:
    """Execute the multi-step sampling loop (sample → evaluate → accumulate).

    Handles resume by rebuilding the accumulated state from prior step
    artifacts, then runs remaining steps. For ``best_of`` with resume,
    top-ups extra samples when the current count is below target.

    Args:
        sampling_dir: Root directory for per-step artifacts.
        base_sampling_path: Initial (prepared) sampling state file.
        final_sampling_path: Accumulated sampling state file (updated in place).
        dataset: Dataset name for plan evaluation.
        model: LLM model name for sampling.
        strategy: Strategy name ("bol_evol", "pst_evol", "best_of").
        n_steps: Total number of sampling steps to run.
        n_samples_per_step: LLM samples per step.
        max_sampling_workers: Max concurrent LLM workers.
        max_workers: Max concurrent evaluation workers.
        optimization_metric: Metric for ranking (lower is better).
        resume: Whether to resume from existing step artifacts.
        sampling_kwargs: Extra kwargs forwarded to sample_plans_from_file.
        evaluation_kwargs: Extra kwargs forwarded to evaluate_sampled_plans.
        verbose: Print progress messages.
    """
    completed_steps = _list_completed_steps(sampling_dir)
    max_completed = max(completed_steps) if completed_steps else 0
    rebuild_limit = min(max_completed, n_steps) if n_steps >= 0 else max_completed

    if not resume:
        final_sampling_path.write_text(base_sampling_path.read_text())
    else:
        if (not final_sampling_path.exists()) or (n_steps < max_completed):
            final_sampling_path.write_text(base_sampling_path.read_text())
            for step_idx in range(1, rebuild_limit + 1):
                step_dir = sampling_dir / f"step_{step_idx}"
                evaluated_path = _step_evaluated_path(step_dir)
                if not evaluated_path.exists():
                    raise RuntimeError(f"Missing evaluated_sampled_plans.json for step {step_idx}")
                accumulate_samples(
                    input_file=str(evaluated_path),
                    output_file=str(final_sampling_path),
                    verbose=verbose,
                )

    def _run_step(step_idx: int, n_samples: int, sampling_strategy: SamplingStrategy) -> None:
        step_dir = sampling_dir / f"step_{step_idx}"
        step_dir.mkdir(parents=True, exist_ok=True)
        sampled_path = step_dir / "sampled_plans.json"
        evaluated_path = step_dir / "evaluated_sampled_plans.json"

        log_line(verbose, f"\n  [Step {step_idx}/{n_steps}] Generating {n_samples} optimization samples...")

        if not (resume and sampled_path.exists()):
            sample_plans_from_file(
                input_file=str(final_sampling_path),
                output_file=str(sampled_path),
                sampling_strategy=sampling_strategy,
                n_samples=n_samples,
                model=model,
                max_workers=max_sampling_workers,
                verbose=verbose,
                **sampling_kwargs,
            )

        eval_exists = evaluated_path.exists()
        if not (resume and eval_exists):
            log_line(verbose, f"  [Step {step_idx}/{n_steps}] Validating and evaluating candidates...")
            evaluate_sampled_plans(
                input_file=str(sampled_path),
                output_file=str(evaluated_path),
                dataset=dataset,
                n_runs=evaluation_kwargs.get("n_runs"),
                max_workers=max_workers,
                verbose=verbose,
                **{k: v for k, v in evaluation_kwargs.items() if k != "n_runs"},
            )

        accumulate_samples(
            input_file=str(evaluated_path),
            output_file=str(final_sampling_path),
            verbose=False,
        )

        _log_step_improvements(final_sampling_path, optimization_metric, step_idx, n_steps, verbose)

    if n_steps > 0 and rebuild_limit < n_steps:
        for step_idx in range(rebuild_limit + 1, n_steps + 1):
            sampling_strategy, n_samples = _select_strategy(
                strategy,
                step_idx,
                n_samples_per_step,
                optimization_metric,
            )
            _run_step(step_idx, n_samples, sampling_strategy)

    if strategy == "best_of" and resume and n_steps == 1:
        if final_sampling_path.exists():
            sampling_state = json.loads(final_sampling_path.read_text())
            if sampling_state:
                current_samples = sum(
                    1
                    for p in sampling_state[0].get("sampled_plans", [])
                    if p.get("sample_id") is not None
                )
                if current_samples < n_samples_per_step:
                    extra = n_samples_per_step - current_samples
                    extra_step = max_completed + 1 if max_completed >= 1 else 1
                    _run_step(extra_step, extra, ORIGINAL_STRATEGY)
