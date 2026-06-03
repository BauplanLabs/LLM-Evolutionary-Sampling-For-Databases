from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple
import json
import os
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


def data_folder_for_dataset(dataset: str, exec_local: bool = False,
                            scale_factor: Optional[int] = None) -> str:
    """Return the data folder path for a given dataset name.

    Modal returns a fixed path (``/tmp/data/data_<dataset>``); the scale factor
    lives in the scale-specific image, so encoding it in the path would change
    the image and trigger rebuilds — it is deliberately omitted here.

    Locally there is no image, so every scale factor coexists under
    ``LOCAL_DATA_DIR`` and a *scaled* dataset gets a ``_sf<N>`` suffix (pass its
    *scale_factor*). A *scaleless* dataset — e.g. a fixed dataset like IMDB/JOB —
    passes ``scale_factor=None`` and gets no suffix.
    """
    if exec_local:
        from modal_controller.constants import LOCAL_DATA_DIR
        suffix = f"_sf{scale_factor}" if scale_factor is not None else ""
        return os.path.join(LOCAL_DATA_DIR, f"data_{dataset}{suffix}")
    return f"/tmp/data/data_{dataset}"


@contextmanager
def _file_lock(lock_path: str) -> Iterator[None]:
    """Best-effort cross-process exclusive lock via ``fcntl`` (POSIX).

    On platforms without ``fcntl`` (e.g. Windows, where local execution is
    unsupported anyway) this degrades to a no-op; the atomic temp-rename in
    :func:`ensure_local_data` still prevents half-populated data folders.
    """
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def ensure_local_data(dataset: str, scale_factor: int) -> str:
    """Ensure local data for *dataset* at *scale_factor* exists.

    Used by the ``exec_local`` paths: local execution mirrors Modal, which
    generates its data at image-build time. When the local data folder for this
    (dataset, scale_factor) doesn't exist yet, this generates it into
    ``LOCAL_DATA_DIR`` — every scale factor coexists in its own ``_sf<N>`` folder,
    so switching scale never regenerates. The generation is always announced
    because it can take a few minutes. Returns the data folder.

    Generation is guarded by a cross-process lock (so concurrent runs generate
    once, not N times) and written to a temp directory that is atomically
    renamed into place. Because of that atomic rename, the folder existing means
    generation completed — so a plain ``isdir`` check is the success signal.
    """
    folder = data_folder_for_dataset(dataset, exec_local=True, scale_factor=scale_factor)
    if os.path.isdir(folder):
        return folder

    if dataset not in ("tpch", "tpcds"):
        raise ValueError(
            f"Local data generation is supported for 'tpch' and 'tpcds', not '{dataset}'."
        )

    from modal_controller.constants import LOCAL_DATA_DIR
    from modal_controller.generate_tpch_files import generate_benchmark_data
    import shutil
    import tempfile

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    with _file_lock(os.path.join(LOCAL_DATA_DIR, f".{os.path.basename(folder)}.gen.lock")):
        # Re-check under the lock: another process may have generated it while
        # we waited.
        if os.path.isdir(folder):
            return folder

        print(
            f"[exec_local] No local data for '{dataset}' (sf={scale_factor}) found at {folder}. "
            f"Generating {dataset} at scale_factor={scale_factor} (this may take a few minutes)..."
        )
        tmp_base = tempfile.mkdtemp(prefix=f".{dataset}_gen_", dir=LOCAL_DATA_DIR)
        try:
            generate_benchmark_data(dataset, factor=scale_factor, base_data_dir=tmp_base)
            tmp_folder = os.path.join(tmp_base, f"data_{dataset}")
            if not (os.path.isdir(tmp_folder) and os.listdir(tmp_folder)):
                raise RuntimeError(f"Data generation produced no files for '{dataset}'.")
            os.replace(tmp_folder, folder)  # atomic; folder is absent (checked under lock)
        finally:
            shutil.rmtree(tmp_base, ignore_errors=True)

    return folder


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
