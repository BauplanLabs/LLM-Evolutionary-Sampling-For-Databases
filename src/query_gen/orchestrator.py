from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from dbplanbench_types import QueryGenerationResult
from dbplanbench_utils import log_line, normalize_config, resolve_run_dir, write_json, data_folder_for_dataset
from modal_controller.modal_runner import Operation
from modal_controller.utils import submit_run_operation
from query_gen import prompts as query_prompts
from query_gen.query_generator import generate_queries as _generate_queries
from query_gen.query_validator import validate_queries as _validate_queries


def run_query_generation(
    *,
    dataset: str,
    complexity_distribution: Dict[int, Union[int, float]],
    n_queries: int,
    run_dir: Optional[str],
    include_sample_rows_in_prompt: bool,
    max_concurrent_generations: int,
    max_concurrent_validators: int,
    scale_factor: int,
    max_steps: int,
    oversample_cap: float,
    resume: bool,
    validate_kwargs: Optional[Dict[str, Any]],
    verbose: bool,
) -> QueryGenerationResult:
    """Orchestrate iterative query generation + validation and persist results.

    Generates SQL queries via LLM, validates them on Modal (syntax, determinism,
    non-empty result), and collects valid queries until the target complexity
    distribution is met or *max_steps* is reached. Supports resume from
    existing run-dir artifacts.

    Args:
        dataset: Dataset name ("tpch" or "tpcds").
        complexity_distribution: Target distribution of complexity levels.
        n_queries: Total number of valid queries to produce.
        run_dir: Output directory (None = timestamped default).
        include_sample_rows_in_prompt: Include sample rows in the LLM prompt.
        max_concurrent_generations: Max concurrent LLM workers.
        max_concurrent_validators: Max concurrent Modal validation workers.
        scale_factor: Scale factor for Modal runner.
        max_steps: Maximum generate/validate iterations.
        oversample_cap: Upper bound on per-step oversampling multiplier.
        resume: Resume from existing run-dir state.
        validate_kwargs: Extra kwargs forwarded to validation.
        verbose: Print progress.

    Returns:
        QueryGenerationResult with run_dir, summary, queries, and metadata.
    """
    def _allocate_targets(
        dist: Dict[int, Union[int, float]],
        total: int,
    ) -> Dict[int, int]:
        weights = {int(k): float(v) for k, v in dist.items()}
        if total <= 0:
            raise ValueError("n_queries must be > 0")
        if any(v < 0 for v in weights.values()):
            raise ValueError("complexity_distribution values must be non-negative")
        wsum = sum(weights.values())
        if wsum <= 0:
            raise ValueError("complexity_distribution must sum to > 0")

        raw = {k: total * (v / wsum) for k, v in weights.items()}
        floors = {k: int(raw[k]) for k in raw}
        remainder = total - sum(floors.values())
        if remainder > 0:
            frac = sorted(
                ((k, raw[k] - floors[k]) for k in raw),
                key=lambda x: x[1],
                reverse=True,
            )
            for i in range(remainder):
                floors[frac[i % len(frac)][0]] += 1
        return floors

    # Resolve run_dir
    run_dir_path = resolve_run_dir(run_dir, "generate_queries")
    run_dir_path.mkdir(parents=True, exist_ok=True)

    # Config handling
    config = {
        "dataset": dataset,
        "complexity_distribution": complexity_distribution,
        "n_queries": n_queries,
        "include_sample_rows_in_prompt": include_sample_rows_in_prompt,
        "max_concurrent_generations": max_concurrent_generations,
        "max_concurrent_validators": max_concurrent_validators,
        "max_steps": max_steps,
        "oversample_cap": oversample_cap,
        "scale_factor": scale_factor,
    }
    config_path = run_dir_path / "generate_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if normalize_config(existing) != normalize_config(config):
            if resume:
                raise ValueError("Config mismatch for generate_queries and resume=True")
            write_json(config_path, config)
    else:
        write_json(config_path, config)

    # If restart requested, clear previous outputs
    if not resume:
        for path in run_dir_path.glob("outcome_*.json"):
            path.unlink(missing_ok=True)
        final_path = run_dir_path / "final_queries.json"
        if final_path.exists():
            final_path.unlink()

    # Load existing state if resuming
    final_path = run_dir_path / "final_queries.json"
    existing_queries: List[str] = []
    existing_meta: List[Dict[str, Any]] = []
    payload: Optional[Dict[str, Any]] = None
    if resume and final_path.exists():
        payload = json.loads(final_path.read_text())
        existing_queries = payload.get("queries", [])
        existing_meta = payload.get("metadata", [])

    target_counts = _allocate_targets(complexity_distribution, n_queries)
    current_counts: Dict[int, int] = {k: 0 for k in target_counts}
    for m in existing_meta:
        c = int(m["complexity"])
        current_counts[c] = current_counts.get(c, 0) + 1

    for c in target_counts:
        if current_counts.get(c, 0) > target_counts[c]:
            raise ValueError("Existing final_queries exceeds target complexity distribution")

    # Determine starting step index and existing totals
    step_files = list(run_dir_path.glob("outcome_*.json"))
    if step_files:
        step_nums = []
        for p in step_files:
            try:
                step_nums.append(int(p.stem.split("_")[-1]))
            except Exception:
                pass
        step = (max(step_nums) + 1) if step_nums else 1
    else:
        step = 1

    total_generated = 0
    total_generated_by_c: Dict[int, int] = {k: 0 for k in target_counts}
    if step_files:
        for p in step_files:
            try:
                data = json.loads(p.read_text())
                total_generated += int(data.get("summary", {}).get("overall", {}).get("n_generated", 0))
                by_c = data.get("summary", {}).get("by_complexity", {})
                for k, v in by_c.items():
                    if "n_generated" in v:
                        total_generated_by_c[int(k)] = total_generated_by_c.get(int(k), 0) + int(v["n_generated"])
            except Exception:
                continue

    # If already complete, return
    if sum(current_counts.values()) >= n_queries:
        summary = payload.get("summary") if payload else None
        if summary is None:
            fallback_generated = total_generated if total_generated > 0 else len(existing_queries)
            summary = {
                "overall": {
                    "n_requested": n_queries,
                    "n_generated": fallback_generated,
                    "n_final": len(existing_queries),
                },
                "by_complexity": {
                    str(k): {
                        "n_requested": target_counts[k],
                        "n_generated": total_generated_by_c.get(k, 0),
                        "n_final": current_counts.get(k, 0),
                    }
                    for k in sorted(target_counts)
                },
            }
        return QueryGenerationResult(
            run_dir=str(run_dir_path),
            summary=summary,
            queries=existing_queries,
            metadata=existing_meta,
        )

    remaining_counts = {
        k: max(target_counts[k] - current_counts.get(k, 0), 0)
        for k in target_counts
    }

    # Initialize final containers
    final_queries = list(existing_queries)
    final_meta = list(existing_meta)
    seen_queries = set(final_queries)

    log_line(verbose, f"Run dir: {run_dir_path}")
    log_line(verbose, f"Target counts: {target_counts}")
    if sum(current_counts.values()) > 0:
        log_line(verbose, f"Already collected: {current_counts}")

    oai_client = OpenAI()

    validate_kwargs_local = dict(validate_kwargs or {})
    if (
        "scale_factor" in validate_kwargs_local
        and validate_kwargs_local["scale_factor"] != scale_factor
    ):
        raise ValueError("scale_factor conflicts with validate_kwargs['scale_factor']")
    validate_kwargs_local["scale_factor"] = scale_factor

    fetch_kwargs = dict(validate_kwargs_local)
    runner_kwargs = fetch_kwargs.pop("runner_kwargs", {}) or {}
    if not isinstance(runner_kwargs, dict):
        raise ValueError("runner_kwargs must be a dict")
    runner_kwargs = dict(runner_kwargs)
    if "scale_factor" in runner_kwargs and runner_kwargs["scale_factor"] != scale_factor:
        raise ValueError("scale_factor conflicts with runner_kwargs['scale_factor']")
    runner_kwargs["scale_factor"] = scale_factor
    sandbox_placeholders = fetch_kwargs.pop("sandbox_placeholders", {}) or {}
    if not isinstance(sandbox_placeholders, dict):
        raise ValueError("sandbox_placeholders must be a dict")
    sandbox_placeholders = dict(sandbox_placeholders)
    sandbox_placeholders["include_sample_rows"] = (
        "True" if include_sample_rows_in_prompt else "False"
    )
    fetch_kwargs["sandbox_placeholders"] = sandbox_placeholders
    schema_result = submit_run_operation(
        operation=Operation.FETCH_SCHEMA,
        input_str="",
        data_folder=data_folder_for_dataset(dataset),
        runner_kwargs=runner_kwargs,
        **fetch_kwargs,
    )
    if not schema_result or schema_result.get("error"):
        raise RuntimeError(f"Schema fetch failed: {schema_result.get('error') if schema_result else 'no result'}")
    table_to_schema = schema_result.get("tables")
    if not isinstance(table_to_schema, dict) or not table_to_schema:
        raise RuntimeError("Schema fetch returned no tables")
    log_line(verbose, f"Schema tables: {sorted(table_to_schema.keys())}")

    while sum(remaining_counts.values()) > 0:
        if step > max_steps:
            raise RuntimeError("Exceeded maximum generation steps without meeting target distribution")

        remaining_total = sum(remaining_counts.values())
        multiplier = min(oversample_cap, 1.5 ** (step - 1))
        need_n = max(1, int(remaining_total * multiplier))
        generation_request_counts = _allocate_targets(remaining_counts, need_n)
        log_line(
            verbose,
            f"\n[Step {step}] remaining={remaining_counts} request={generation_request_counts} oversample={multiplier:.2f}",
        )

        with tempfile.TemporaryDirectory(dir=run_dir_path, prefix=f"tmp_step_{step}_") as tmpdir:
            tmp_path = Path(tmpdir)
            gen_file = tmp_path / "generated_queries.json"
            validated_file = tmp_path / "validated_queries.json"

            _generate_queries(
                oai_client=oai_client,
                system_prompt=query_prompts.SYSTEM_PROMPT,
                user_prompt=query_prompts.USER_PROMPT,
                complexity_counts=generation_request_counts,
                dataset=dataset,
                table_to_schema=table_to_schema,
                output_file=str(gen_file),
                max_workers=max_concurrent_generations,
                include_sample_rows=include_sample_rows_in_prompt,
                verbose=verbose,
            )

            generated_data = json.loads(gen_file.read_text()) if gen_file.exists() else []
            generated_queries = [d["query"] for d in generated_data]
            total_generated += len(generated_queries)

            _validate_queries(
                input_file=str(gen_file),
                output_file=str(validated_file),
                dead_letter_file=None,
                dataset=dataset,
                max_workers=max_concurrent_validators,
                output_format="validation",
                seen_queries=seen_queries,
                verbose=verbose,
                **validate_kwargs_local,
            )

            validation_data = json.loads(validated_file.read_text()) if validated_file.exists() else []

        validation_by_id = {d["id"]: d for d in validation_data if "id" in d}

        step_meta: List[Dict[str, Any]] = []
        valid_entries: List[Dict[str, Any]] = []
        for item in generated_data:
            q = item["query"]
            c = int(item["complexity"])
            v = validation_by_id.get(item["id"])
            if v is None:
                is_valid = False
                err = "validation_missing"
            else:
                is_valid = bool(v.get("is_valid"))
                err = v.get("error")
            step_meta.append({"complexity": c, "is_valid": is_valid, "error": err})
            if is_valid:
                valid_entries.append({"query": q, "complexity": c})

        step_summary_by_c: Dict[str, Dict[str, int]] = {}
        for c in remaining_counts:
            step_summary_by_c[str(c)] = {
                "n_requested": remaining_counts[c],
                "n_generated": 0,
                "n_valid": 0,
                "n_invalid": 0,
            }
        for item in generated_data:
            c = str(item["complexity"])
            step_summary_by_c[c]["n_generated"] += 1
        for item in valid_entries:
            c = str(item["complexity"])
            step_summary_by_c[c]["n_valid"] += 1
        for c in step_summary_by_c:
            step_summary_by_c[c]["n_invalid"] = (
                step_summary_by_c[c]["n_generated"] - step_summary_by_c[c]["n_valid"]
            )

        outcome = {
            "summary": {
                "overall": {
                    "n_requested": remaining_total,
                    "n_generated": len(generated_queries),
                    "n_valid": len(valid_entries),
                    "n_invalid": len(generated_queries) - len(valid_entries),
                },
                "by_complexity": step_summary_by_c,
            },
            "generated_queries": generated_queries,
            "metadata": step_meta,
        }
        write_json(run_dir_path / f"outcome_{step}.json", outcome)
        for c in step_summary_by_c:
            total_generated_by_c[int(c)] = total_generated_by_c.get(int(c), 0) + step_summary_by_c[c]["n_generated"]

        valid_by_c: Dict[int, List[str]] = {}
        for item in valid_entries:
            valid_by_c.setdefault(item["complexity"], []).append(item["query"])

        accepted_this_step = 0
        for c in sorted(remaining_counts):
            need = remaining_counts[c]
            if need <= 0:
                continue
            candidates = valid_by_c.get(c, [])
            if not candidates:
                continue
            if len(candidates) <= need:
                chosen = candidates
            else:
                chosen = random.sample(candidates, need)
            for q in chosen:
                final_queries.append(q)
                final_meta.append({"complexity": c})
            accepted_count = len(chosen)
            remaining_counts[c] -= accepted_count
            accepted_this_step += accepted_count

        final_summary = {
            "overall": {
                "n_requested": n_queries,
                "n_generated": total_generated,
                "n_final": len(final_queries),
            },
            "by_complexity": {
                str(c): {
                    "n_requested": target_counts[c],
                    "n_generated": total_generated_by_c.get(c, 0),
                    "n_final": target_counts[c] - remaining_counts[c],
                }
                for c in sorted(target_counts)
            },
        }
        write_json(
            final_path,
            {
                "summary": final_summary,
                "queries": final_queries,
                "metadata": final_meta,
            },
        )
        log_line(verbose, f"[Step {step}] accepted={accepted_this_step} total_final={len(final_queries)} remaining={remaining_counts}")

        if accepted_this_step == 0:
            if step >= max_steps:
                raise RuntimeError("No progress in query generation; try increasing samples.")

        step += 1

    final_payload = json.loads(final_path.read_text())
    log_line(
        verbose,
        f"Generation run complete: final={len(final_payload['queries'])}/{n_queries} saved to {final_path}",
    )
    return QueryGenerationResult(
        run_dir=str(run_dir_path),
        summary=final_payload["summary"],
        queries=final_payload["queries"],
        metadata=final_payload["metadata"],
    )
