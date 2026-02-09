# API Reference

All public functions are exported from `api.py`. Return types are dataclasses defined in `api_types.py`.

---

## Functions

### `generate_queries`

Generate and validate SQL queries using LLM generation and Modal-based execution. The function synthesizes queries at varying complexity levels, validates each one on Modal for syntax, executability, non-empty results, and determinism, then returns only the queries that pass every check. If the first batch doesn't yield enough valid queries, additional rounds are generated automatically (up to `max_steps`).

```python
generate_queries(
    *,
    dataset: str,
    complexity_distribution: Dict[int, Union[int, float]],
    n_queries: int,
    run_dir: Optional[str] = None,
    include_sample_rows_in_prompt: bool = False,
    max_concurrent_generations: int = 50,
    max_concurrent_validators: int = 160,
    scale_factor: int = 3,
    max_steps: int = 25,
    oversample_cap: float = 3.0,
    resume: bool = True,
    validate_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> QueryGenerationResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `str` | *required* | Dataset name (e.g., `"tpch"`, `"tpcds"`). |
| `complexity_distribution` | `Dict[int, Union[int, float]]` | *required* | Maps complexity level to count or fraction. E.g., `{5: 2, 6: 1}` requests 2 queries at complexity 5 and 1 at complexity 6. |
| `n_queries` | `int` | *required* | Total number of valid queries to return. |
| `run_dir` | `Optional[str]` | `None` | Output directory. Defaults to a timestamped dir under `outputs/`. |
| `include_sample_rows_in_prompt` | `bool` | `False` | Include sample data rows in the LLM prompt. |
| `max_concurrent_generations` | `int` | `50` | Max concurrent LLM generation workers. |
| `max_concurrent_validators` | `int` | `160` | Max concurrent Modal validation workers. |
| `scale_factor` | `int` | `3` | Dataset scale factor for the Modal sandbox. |
| `max_steps` | `int` | `25` | Max generation rounds before stopping. |
| `oversample_cap` | `float` | `3.0` | Multiplier cap for oversampling to account for validation failures. |
| `resume` | `bool` | `True` | Resume from existing `run_dir` if available. |
| `validate_kwargs` | `Optional[Dict]` | `None` | Extra kwargs forwarded to the validation backend. |
| `verbose` | `bool` | `True` | Print progress. |

Returns `QueryGenerationResult`:

| Field | Type | Description |
|-------|------|-------------|
| `run_dir` | `str` | Output directory path. |
| `summary` | `Dict` | Overall + per-complexity generation/validation counts. |
| `queries` | `List[str]` | Validated SQL query strings. |
| `metadata` | `List[Dict]` | Per-query metadata (aligned to `queries`). |

---

### `optimize_queries`

Optimize SQL queries by iteratively sampling execution-plan patches with an LLM and evaluating them on Modal. The optimizer evolves plan modifications over multiple steps, benchmarks each candidate, and keeps the top-performing patches ranked by `optimization_metric` (lower is better). Queries are validated for correctness and determinism before optimization begins (unless `skip_validation=True`).

```python
optimize_queries(
    queries: Optional[Sequence[str]] = None,
    base_plans: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    run_dir: Optional[str] = None,
    dataset: str = "tpch",
    scale_factor: int = 3,
    *,
    strategy: Literal["bol_evol", "pst_evol", "best_of"] = "bol_evol",
    n_steps: int = 4,
    n_samples_per_step: int = 5,
    top_k_patches: int = 1,
    n_runs: int = 5,
    model: str = "gpt-5",
    optimization_metric: str = "execution_time.min",
    max_sampling_workers: int = 50,
    max_workers: int = 20,
    max_eval_workers: int = 8,
    resume: bool = False,
    skip_validation: bool = False,
    verbose: bool = True,
    validate_kwargs: Optional[Dict[str, Any]] = None,
    sampling_kwargs: Optional[Dict[str, Any]] = None,
    evaluation_kwargs: Optional[Dict[str, Any]] = None,
    get_full_metrics: bool = False,
) -> OptimizationResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `queries` | `Optional[Sequence[str]]` | `None` | SQL queries to optimize. Required unless resuming. |
| `base_plans` | `Optional[Sequence[Optional[Dict]]]` | `None` | Custom base engine plans (aligned to `queries`). When provided, these are validated against engine plans for correctness. |
| `run_dir` | `Optional[str]` | `None` | Output directory. Defaults to a timestamped dir under `outputs/`. |
| `dataset` | `str` | `"tpch"` | Dataset name. |
| `scale_factor` | `int` | `3` | Dataset scale factor. |
| `strategy` | `Literal` | `"bol_evol"` | Sampling strategy: `bol_evol` (evolutionary), `pst_evol` (post-evaluation evolutionary), or `best_of` (single-step). |
| `n_steps` | `int` | `4` | Number of evolutionary sampling steps. |
| `n_samples_per_step` | `int` | `5` | Number of LLM samples per step per query. |
| `top_k_patches` | `int` | `1` | Number of top patches to keep per query. |
| `n_runs` | `int` | `5` | Number of benchmark runs for evaluating each candidate. |
| `model` | `str` | `"gpt-5"` | LLM model for sampling plan patches. |
| `optimization_metric` | `str` | `"execution_time.min"` | Metric for ranking patches (lower is better). Format: `<metric>.<stat>`. |
| `max_sampling_workers` | `int` | `50` | Max concurrent LLM sampling workers. |
| `max_workers` | `int` | `20` | Max concurrent query-level workers. |
| `max_eval_workers` | `int` | `8` | Max concurrent benchmark evaluation workers. |
| `resume` | `bool` | `False` | Resume from existing `run_dir`. |
| `skip_validation` | `bool` | `False` | Skip query validation for determinism and correctness. Prints a warning when enabled. |
| `verbose` | `bool` | `True` | Print progress. |
| `validate_kwargs` | `Optional[Dict]` | `None` | Extra kwargs for the validation backend. |
| `sampling_kwargs` | `Optional[Dict]` | `None` | Extra kwargs for the sampling backend. |
| `evaluation_kwargs` | `Optional[Dict]` | `None` | Extra kwargs for the evaluation backend. |
| `get_full_metrics` | `bool` | `False` | Collect detailed execution metrics (bytes scanned, join times, etc.) in addition to execution time. |

Returns `OptimizationResult`:

| Field | Type | Description |
|-------|------|-------------|
| `run_dir` | `Optional[str]` | Output directory path. |
| `summary` | `Dict` | Validation stats, run config, improvement stats, outcome rates. |
| `queries` | `Optional[List[str]]` | SQL queries (same order as input). |
| `optimization_outcome` | `Optional[List[PatchedPlan]]` | Per-query best plan + patch(es). `None` on early exit (e.g., all queries fail validation). |
| `metadata` | `Optional[List[Dict]]` | Per-query benchmark stats. `None` on early exit. |
| `validation_failures` | `Optional[List[Optional[str]]]` | Per-query validation error (`None` = valid). |

---

### `validate_queries`

Validate SQL queries for syntax, executability, and determinism. Each query is planned and executed on the target dataset. Queries that parse, execute successfully, return non-empty results, and produce identical outputs across `n_determinism_retries` executions are considered valid. Use this as a pre-flight check before `optimize_queries` or `benchmark_queries` to avoid wasted compute on invalid queries.

```python
validate_queries(
    queries: Sequence[str],
    *,
    dataset: str,
    scale_factor: int = 3,
    n_determinism_retries: int = 3,
    max_workers: int = 10,
    verbose: bool = True,
    **kwargs,
) -> QueryValidationResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `queries` | `Sequence[str]` | *required* | SQL query strings to validate. |
| `dataset` | `str` | *required* | Dataset name. |
| `scale_factor` | `int` | `3` | Dataset scale factor. |
| `n_determinism_retries` | `int` | `3` | Number of executions for the determinism check. Higher values increase confidence but cost more. |
| `max_workers` | `int` | `10` | Max concurrent validation workers. |
| `verbose` | `bool` | `True` | Print progress and summary. |
| `**kwargs` | | | Passed through to the Modal backend. Supports `runner_kwargs`, `cpu`, `memory`, `timeout`, `sandbox_timeout`. |

Returns `QueryValidationResult`:

| Field | Type | Description |
|-------|------|-------------|
| `plans` | `List[Optional[Dict]]` | Per-query engine plan (`None` for invalid queries). Valid plans can be fed directly to `benchmark_plans`. |
| `errors` | `List[Optional[str]]` | Per-query error category (`None` for valid queries). |
| `summary` | `Dict` | Aggregate counts: `n_queries`, `n_valid`, and per-category failure counts. |

Error categories: `syntax_error`, `no_plan`, `cannot_run`, `empty_result`, `nondeterministic`, `validation_error`.

---

### `benchmark_plans`

Benchmark execution plans (with optional patches) against a dataset. Each `PatchedPlan` can carry multiple patches (a multi-patch `PatchSet`), and the function evaluates every patch variant independently, so `results[i]` is a list of measurements aligned to the plan's patches.

```python
benchmark_plans(
    plans: Sequence[Union[PatchedPlan, Dict[str, Any]]],
    *,
    dataset: str,
    scale_factor: int = 3,
    n_runs: int = 3,
    output_file: Optional[str] = None,
    max_workers: int = 1,
    max_eval_workers: int = 8,
    get_full_metrics: bool = False,
    verbose: bool = True,
    **kwargs,
) -> BenchmarkResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plans` | `Sequence[Union[PatchedPlan, Dict]]` | *required* | Plans to benchmark. Each can be a `PatchedPlan` or a raw dict with `base_plan` and `patch` keys. |
| `dataset` | `str` | *required* | Dataset name. |
| `scale_factor` | `int` | `3` | Dataset scale factor. |
| `n_runs` | `int` | `3` | Number of benchmark runs per plan variant. |
| `output_file` | `Optional[str]` | `None` | Path to write results JSON. |
| `max_workers` | `int` | `1` | Max concurrent plan-level workers. |
| `max_eval_workers` | `int` | `8` | Max concurrent evaluation workers within each plan. |
| `get_full_metrics` | `bool` | `False` | Collect detailed execution metrics beyond execution time. |
| `verbose` | `bool` | `True` | Print progress. |
| `**kwargs` | | | Passed through to the Modal backend. |

Returns `BenchmarkResult`:

| Field | Type | Description |
|-------|------|-------------|
| `results` | `List[List[Dict]]` | `results[plan_idx][patch_idx]` = measurement dict with `benchmark_stats`. |
| `output_file` | `Optional[str]` | Path where results were written (if requested). |

---

### `benchmark_queries`

Benchmark raw SQL queries — plans them automatically via the engine, then evaluates. This is a convenience wrapper around `get_engine_plans` + `benchmark_plans` for users who just want to measure query performance without manual plan construction.

```python
benchmark_queries(
    queries: Sequence[str],
    *,
    dataset: str,
    scale_factor: int = 3,
    n_runs: int = 3,
    output_file: Optional[str] = None,
    max_workers: int = 1,
    max_eval_workers: int = 8,
    get_full_metrics: bool = False,
    verbose: bool = True,
    **kwargs,
) -> BenchmarkResult
```

Parameters are the same as `benchmark_plans`, except `queries` replaces `plans`. Returns `BenchmarkResult` where each `results[i]` is a single-element list with one measurement dict (since there are no patches to compare against).

---

### `get_engine_plans`

Obtain engine execution plans for SQL queries. Useful for inspecting what the query engine produces, or for manually modifying plans before feeding them to `benchmark_plans`.

```python
get_engine_plans(
    queries: Sequence[str],
    *,
    dataset: str,
    scale_factor: int = 3,
    max_workers: int = 10,
    verbose: bool = True,
    **kwargs,
) -> PlanningResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `queries` | `Sequence[str]` | *required* | SQL query strings. |
| `dataset` | `str` | *required* | Dataset name. |
| `scale_factor` | `int` | `3` | Dataset scale factor. |
| `max_workers` | `int` | `10` | Max concurrent planning workers. |
| `verbose` | `bool` | `True` | Print progress. |
| `**kwargs` | | | Passed through to the Modal backend. |

Returns `PlanningResult`:

| Field | Type | Description |
|-------|------|-------------|
| `plans` | `List[Optional[Dict]]` | Per-query engine plan (`None` on failure). |
| `errors` | `List[Optional[str]]` | Per-query error string (`None` on success). |

---

### `scale_optimizations`

Transfer optimized plans from a smaller to a larger dataset scale factor. For each query, the function plans it at the target scale to obtain the new baseline, then transfers both the base plan and each optimization patch using scan-signature matching and ID remapping. Optionally validates transferred plans for correctness and determinism via result-set comparison against the target-scale engine plan.

```python
scale_optimizations(
    queries: Sequence[str],
    plans: Sequence[PatchedPlan],
    *,
    source_scale_factor: int,
    target_scale_factor: int,
    dataset: str = "tpch",
    run_dir: Optional[str] = None,
    skip_validation: bool = False,
    max_workers: int = 20,
    modal_resources: Optional[Dict[str, Any]] = None,
    validate_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> ScaleResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `queries` | `Sequence[str]` | *required* | SQL queries (aligned to `plans`). |
| `plans` | `Sequence[PatchedPlan]` | *required* | `PatchedPlan` instances from a previous `optimize_queries` run. |
| `source_scale_factor` | `int` | *required* | Scale factor the input plans were optimized at. |
| `target_scale_factor` | `int` | *required* | Target scale factor to transfer optimizations to. |
| `dataset` | `str` | `"tpch"` | Dataset name. |
| `run_dir` | `Optional[str]` | `None` | Output directory. |
| `skip_validation` | `bool` | `False` | Skip result-set validation of transferred plans. |
| `max_workers` | `int` | `20` | Max concurrent workers. |
| `modal_resources` | `Optional[Dict]` | `None` | Modal sandbox resource configuration. Supported keys: `cpu`, `memory`, `timeout`, `sandbox_timeout`. These are forwarded to `ModalRunner.run_operation` alongside the scale factor. Example: `{"memory": (8192, 8192)}` for 8 GB RAM. |
| `validate_kwargs` | `Optional[Dict]` | `None` | Extra kwargs forwarded to validation/planning backends. |
| `verbose` | `bool` | `True` | Print progress. |

Returns `ScaleResult`:

| Field | Type | Description |
|-------|------|-------------|
| `run_dir` | `Optional[str]` | Output directory path. |
| `summary` | `Dict` | Config + outcome rates (per-patch). |
| `queries` | `List[str]` | SQL queries (same order as input). |
| `scaled_plans` | `List[PatchedPlan]` | Transferred plans at `target_scale_factor`. |
| `metadata` | `List[Dict]` | Per-query transfer details and per-patch statuses. |
| `transfer_failures` | `List[Optional[str]]` | Per-query error (`None` = all patches OK). |

---

## Core Type: `PatchedPlan`

A `PatchedPlan` pairs a base execution plan with one or more optimization patches:

```python
@dataclass
class PatchedPlan:
    base_plan: Dict[str, Any]    # plan dict with "structure" and "succinct_table_info"
    patch: PatchSet              # list of candidate patches
```

`PatchSet` is always a `List[Optional[Patch]]`. Each element is:
- A `Patch` (list of JSON Patch ops): `[{"op": "replace", "path": "/...", "value": ...}, ...]`
- `[]`: empty patch (no modifications to the base plan)
- `None`: patch unavailable

Examples:
- `[[]]`: base plan only (one no-op patch)
- `[patch]`: single optimization
- `[patch_1, patch_2]`: top-2 candidates
- `[None]`: patch unavailable

---

## Error Handling & Troubleshooting

### How functions report errors

| Function | On invalid input | On execution failure |
|---|---|---|
| `generate_queries` | Raises `ValueError` | Returns partial `QueryGenerationResult` (fewer queries than requested) |
| `validate_queries` | Raises `ValueError` | `QueryValidationResult.errors[i]` is non-None for failed queries |
| `optimize_queries` | Raises `ValueError` | Returns `OptimizationResult` with `optimization_outcome=None` and `validation_failures` populated |
| `benchmark_plans` | Raises `ValueError` | Per-plan result dicts contain `{"error": "..."}` |
| `get_engine_plans` | N/A | `PlanningResult.errors[i]` is non-None for failed queries |
| `benchmark_queries` | Raises `ValueError` | Per-query result dicts contain `{"error": "..."}` |
| `scale_optimizations` | Raises `ValueError` | `ScaleResult.transfer_failures[i]` is non-None for failed queries; `scaled_plans[i].patch = [None]` |

### Common Modal/S3 error messages

| Error message | Cause | Resolution |
|---|---|---|
| `Status.RESOURCE_EXHAUSTED` | Modal cluster at capacity | Retry automatically; reduce `max_workers` if persistent |
| `Status.UNAVAILABLE` | Modal service temporarily down | Retry automatically; wait and re-run |
| `Protocol error` | Network issue between client and Modal | Retry automatically; check internet connection |
| `No UUID found in output` | Sandbox crashed or timed out before producing output | Check that your query isn't too complex for the timeout; increase `sandbox_timeout` in kwargs |
| `Error retrieving result from S3 for UUID` | Sandbox ran but result wasn't saved to S3 | May indicate an AWS credentials issue or sandbox crash during S3 upload |
| `internal server error` | Modal platform error | Retry automatically; contact Modal support if persistent |

### Troubleshooting

**"ModuleNotFoundError: No module named 'api'"**
Make sure you installed dependencies with `uv sync` from the repo root (see the main [README](../README.md)) and are running from the correct environment (`source .venv/bin/activate`).

**Sandbox timeouts**
The default `sandbox_timeout` is 120 seconds and execution `timeout` is 45 seconds. For complex queries or large scale factors, pass higher values:
```python
result = optimize_queries(..., evaluation_kwargs={"sandbox_timeout": 300, "timeout": 120})
```

**Low sandbox concurrency**
If you observe fewer concurrent Modal sandboxes than expected, note that sandbox creation is rate-limited to `RATE_LIMIT_PER_SEC = 4` starts per second (configurable in `src/modal_controller/constants.py`). The total concurrent sandbox cap is `MAX_CONCURRENT_SANDBOXES = 100`. Effective concurrency depends on `max_workers * max_eval_workers`.

**LiteLLM / OpenAI errors**
Ensure `OPENAI_API_KEY` (or the appropriate provider key) is set in your `.env` file. LiteLLM supports multiple providers — see [LiteLLM docs](https://docs.litellm.ai/) for configuration.

**AWS credential errors**
If S3 operations fail, verify:
1. `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set in `.env`
2. The IAM user has the required S3 permissions (see Setup section in the root README)
3. The Modal secret `s3-aws-credentials` is configured with your AWS keys
