from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json
import random

def repo_root() -> Path:
    """Return the repository root (one level above src/)."""
    return Path(__file__).resolve().parents[1]


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-round-tripped copy of *cfg* with sorted keys, for stable comparison."""
    return json.loads(json.dumps(cfg, sort_keys=True))


def write_json(path: Path, payload: Any) -> None:
    """Write *payload* as pretty-printed JSON to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def resolve_run_dir(run_dir: Optional[str], default_subdir: str) -> Path:
    """Resolve *run_dir* to an absolute path under ``<repo>/outputs/``.

    Args:
        run_dir: User-supplied path or None. When None a timestamped dir is
            created under ``outputs/<default_subdir>/``. Relative paths are
            anchored to the repo outputs directory.
        default_subdir: Subdirectory name used when *run_dir* is None.

    Returns:
        Absolute Path for the run directory.
    """
    root = repo_root()
    if run_dir is None:
        ts = datetime.now().strftime("%y.%m.%d.%H.%M")
        return root / "outputs" / default_subdir / ts
    run_dir_path = Path(run_dir)
    if not run_dir_path.is_absolute():
        if run_dir_path.parts and run_dir_path.parts[0] == "outputs":
            run_dir_path = root / run_dir_path
        else:
            run_dir_path = root / "outputs" / run_dir_path
    return run_dir_path


def get_evaluation_stats(plan_info: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract the ``evaluation_stats`` dict from *plan_info*, or return None."""
    if not isinstance(plan_info, dict):
        return None
    return plan_info.get("evaluation_stats")


def log_line(verbose: bool, message: str) -> None:
    """Print *message* only when *verbose* is True."""
    if verbose:
        print(message)


def data_folder_for_dataset(dataset: str) -> str:
    """Return the Modal-side data folder path for a given dataset name."""
    return f"/tmp/data/data_{dataset}"


def plan_to_json(plan: Any) -> str:
    """Serialize *plan* to a JSON string (no-op if already a string)."""
    return json.dumps(plan) if not isinstance(plan, str) else plan


def build_validation_stats(failures: List[Optional[str]]) -> Dict[str, Any]:
    """Compute validation summary from a per-query failures list.

    Args:
        failures: Per-query list where each element is an error string
            (query failed) or None (query valid).

    Returns:
        Dict with ``n_queries``, ``n_valid``, and ``random_3_error_messages``.
    """
    errors = [err for err in failures if err]
    return {
        "n_queries": len(failures),
        "n_valid": len(failures) - len(errors),
        "random_3_error_messages": random.sample(errors, k=min(3, len(errors))) if errors else [],
    }


def format_metric_stats(stats: Dict[str, Any]) -> str:
    """Format a stats dict (from ``compute_metric_stats``) as a compact ``key=value`` string.

    Only includes keys present in ``METRIC_STAT_KEYS`` whose values are not None.
    """
    from modal_controller.utils import METRIC_STAT_KEYS
    return "  ".join(
        f"{k}={stats[k]:.2f}"
        for k in METRIC_STAT_KEYS
        if stats.get(k) is not None
    )


def get_metric_value(stats: Optional[Dict[str, Any]], metric_path: str) -> Optional[float]:
    """Look up a numeric metric from *stats* using a dot-separated path.

    Args:
        stats: Evaluation stats dict. May contain a nested ``benchmark_stats``
            sub-dict for dotted paths.
        metric_path: Dot-separated key (e.g. ``"execution_time.min"``).
            A simple key (no dots) is looked up directly on *stats*.

    Returns:
        The metric value as a float, or None if not found / not numeric.
    """
    if not stats:
        return None
    if "." in metric_path:
        current: Any = stats.get("benchmark_stats") if isinstance(stats, dict) and "benchmark_stats" in stats else stats
        for part in metric_path.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current if isinstance(current, (int, float)) else None
    value = stats.get(metric_path)
    return value if isinstance(value, (int, float)) else None
