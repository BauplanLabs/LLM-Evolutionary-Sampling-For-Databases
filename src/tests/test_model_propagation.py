"""Tests for sampling observability + model propagation.

Covers the feat/sampling-observability surface:
  - the model name flowing through GPTPlanOptimizer / sample_plans_from_file,
  - completion_kwargs being forwarded to litellm.completion,
  - reasoning_content + token usage captured on GenerationResult,
  - those fields surfaced into the sampled-plan output, and
  - carried through accumulate_samples.

All tests mock litellm — no real LLM, Modal, or DataFusion.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from sampling.gpt_plan_optimizer import GPTPlanOptimizer, GenerationResult
from sampling.sample_plans import sample_plans_from_file
from sampling.accumulate_samples import accumulate_samples


PATCH_CONTENT = '<patch>[{"op": "replace", "path": "/0/x", "value": 1}]</patch>'
PLAN = {"structure": {}, "succinct_table_info": {}}


def _mock_response(content=PATCH_CONTENT, reasoning="reasoning", usage=None):
    """Build a litellm-like response mock. usage=None -> no usage object."""
    msg = MagicMock()
    msg.content = content
    msg.reasoning_content = reasoning
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    if usage is None:
        resp.usage = None
    else:
        u = MagicMock()
        u.prompt_tokens, u.completion_tokens, u.total_tokens = usage
        resp.usage = u
    return resp


def _one_sample(opt, **kwargs):
    return opt.optimize_plan("SELECT 1", PLAN, n_samples=1, verbose=False, **kwargs)[0]


# ---------------------------------------------------------------------------
# Model propagation
# ---------------------------------------------------------------------------

class TestModelPropagation:
    def test_optimizer_passes_model_to_litellm(self):
        custom = "together_ai/meta-llama/Meta-Llama-3-70B"
        opt = GPTPlanOptimizer(model=custom)
        assert opt.model == custom
        with patch("litellm.completion", return_value=_mock_response()) as mc:
            result = _one_sample(opt)
        mc.assert_called_once()
        assert mc.call_args.kwargs["model"] == custom
        assert result.sampled_patches is not None

    def test_sample_plans_from_file_passes_model(self, tmp_path):
        custom = "openrouter/anthropic/claude-3-opus"
        out = _run_sample_plans(tmp_path, model=custom)
        out["mock"].assert_called_once_with(model=custom, completion_kwargs=None)

    def test_default_model_is_gpt5(self):
        assert GPTPlanOptimizer().model == "gpt-5"


# ---------------------------------------------------------------------------
# completion_kwargs forwarding
# ---------------------------------------------------------------------------

class TestCompletionKwargs:
    def test_forwarded_to_litellm(self):
        opt = GPTPlanOptimizer(completion_kwargs={"reasoning_effort": "high", "max_tokens": 4096})
        with patch("litellm.completion", return_value=_mock_response()) as mc:
            _one_sample(opt)
        kw = mc.call_args.kwargs
        assert kw["reasoning_effort"] == "high"
        assert kw["max_tokens"] == 4096

    def test_default_is_empty_dict(self):
        assert GPTPlanOptimizer().completion_kwargs == {}

    def test_none_is_safe(self):
        opt = GPTPlanOptimizer(completion_kwargs=None)
        with patch("litellm.completion", return_value=_mock_response()) as mc:
            _one_sample(opt)
        # No extra keys injected beyond the standard call.
        assert {"model", "messages", "temperature"} <= set(mc.call_args.kwargs)

    def test_sample_plans_forwards_completion_kwargs(self, tmp_path):
        ck = {"reasoning_effort": "low"}
        out = _run_sample_plans(tmp_path, completion_kwargs=ck)
        out["mock"].assert_called_once_with(model="gpt-5", completion_kwargs=ck)


# ---------------------------------------------------------------------------
# Observability capture on GenerationResult
# ---------------------------------------------------------------------------

class TestObservabilityCapture:
    def test_reasoning_and_tokens_captured(self):
        opt = GPTPlanOptimizer()
        with patch("litellm.completion", return_value=_mock_response(reasoning="because X", usage=(11, 22, 33))):
            r = _one_sample(opt)
        assert r.reasoning_content == "because X"
        assert (r.prompt_tokens, r.completion_tokens, r.total_tokens) == (11, 22, 33)

    def test_model_response_is_raw_content(self):
        # Regression: model_response is now the raw content, not a
        # "Response:\n--\n...Reasoning:\n--\n..." blob.
        opt = GPTPlanOptimizer()
        with patch("litellm.completion", return_value=_mock_response()):
            r = _one_sample(opt)
        assert r.model_response == PATCH_CONTENT

    def test_missing_usage_yields_none_tokens(self):
        opt = GPTPlanOptimizer()
        with patch("litellm.completion", return_value=_mock_response(usage=None)):
            r = _one_sample(opt)
        assert r.prompt_tokens is None
        assert r.completion_tokens is None
        assert r.total_tokens is None

    def test_missing_reasoning_attribute_uses_placeholder(self):
        # A provider whose message has no reasoning_content attribute at all.
        msg = MagicMock()
        msg.content = PATCH_CONTENT
        del msg.reasoning_content
        resp = MagicMock()
        resp.choices = [MagicMock(message=msg)]
        resp.usage = None
        opt = GPTPlanOptimizer()
        with patch("litellm.completion", return_value=resp):
            r = _one_sample(opt)
        assert r.reasoning_content == "<No reasoning content available>"

    def test_error_path_populates_reasoning_field(self):
        opt = GPTPlanOptimizer()
        with patch("litellm.completion", side_effect=RuntimeError("boom")):
            r = _one_sample(opt, n_retry=0)
        assert r.error_message is not None
        assert r.sampled_patches is None
        assert r.reasoning_content == ""


# ---------------------------------------------------------------------------
# Fields surfaced into the sampled-plan output, then through accumulation
# ---------------------------------------------------------------------------

class TestObservabilitySurfacedToOutput:
    def test_sample_plans_writes_observability_fields(self, tmp_path):
        result = GenerationResult(
            model_response="raw response",
            reasoning_content="the trace",
            error_message=None,
            sampled_patches=[{"op": "replace", "path": "/0/x", "value": 1}],
            prompt_tokens=5,
            completion_tokens=7,
            total_tokens=12,
        )
        out = _run_sample_plans(tmp_path, gen_result=result)
        sampled = json.loads(out["output_file"].read_text())[0]["sampled_plans"][0]
        assert sampled["model_response"] == "raw response"
        assert sampled["reasoning_content"] == "the trace"
        assert sampled["prompt_tokens"] == 5
        assert sampled["completion_tokens"] == 7
        assert sampled["total_tokens"] == 12

    def test_accumulate_propagates_observability_fields(self, tmp_path):
        accumulated = [
            {
                "id": 0,
                "sampled_plans": [
                    {"sample_id": 0, "parent_sample_id": None, "is_last": True, "is_leaf": True},
                ],
            }
        ]
        step = [
            {
                "id": 0,
                "sampled_plans": [
                    {
                        "parent_sample_id": 0,
                        "sampled_patches": [{"op": "replace", "path": "/0/x", "value": 1}],
                        "is_valid": True,
                        "error_message": None,
                        "model_response": "resp",
                        "reasoning_content": "why",
                        "prompt_tokens": 5,
                        "completion_tokens": 7,
                        "total_tokens": 12,
                    }
                ],
            }
        ]
        output_file = tmp_path / "accumulated.json"
        input_file = tmp_path / "step.json"
        output_file.write_text(json.dumps(accumulated))
        input_file.write_text(json.dumps(step))

        accumulate_samples(str(input_file), str(output_file), verbose=False)

        plans = json.loads(output_file.read_text())[0]["sampled_plans"]
        new_plan = next(p for p in plans if p.get("plan_type") == "optimized")
        assert new_plan["model_response"] == "resp"
        assert new_plan["reasoning_content"] == "why"
        assert new_plan["prompt_tokens"] == 5
        assert new_plan["completion_tokens"] == 7
        assert new_plan["total_tokens"] == 12


# ---------------------------------------------------------------------------
# Shared helper: drive sample_plans_from_file with a mocked optimizer.
# ---------------------------------------------------------------------------

def _run_sample_plans(tmp_path, *, model="gpt-5", completion_kwargs=None, gen_result=None):
    input_data = [
        {
            "id": 0,
            "query": "SELECT 1",
            "plan": {"structure": {"0": {"scan": {}}}, "succinct_table_info": []},
            "sampled_plans": [
                {
                    "sample_id": None,
                    "parent_sample_id": None,
                    "sampled_patches": [],
                    "is_valid": True,
                    "is_leaf": True,
                    "is_last": True,
                    "evaluation_stats": {"execution_time": {"min": 1.0}},
                }
            ],
        }
    ]
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"
    input_file.write_text(json.dumps(input_data))

    if gen_result is None:
        gen_result = GenerationResult(
            model_response="test",
            reasoning_content="",
            error_message=None,
            sampled_patches=[{"op": "replace", "path": "/0/scan/x", "value": 1}],
        )

    with patch("sampling.sample_plans.GPTPlanOptimizer") as MockOptimizer:
        MockOptimizer.return_value.optimize_plans_batch.return_value = [[gen_result]]
        sample_plans_from_file(
            input_file=str(input_file),
            output_file=str(output_file),
            model=model,
            n_samples=1,
            verbose=False,
            completion_kwargs=completion_kwargs,
        )
    return {"mock": MockOptimizer, "output_file": output_file}


# ---------------------------------------------------------------------------
# Opt-in: REAL LLM calls (cost credits). Carry the `llm` marker, so they are
# excluded from the default suite; they also auto-skip without an API key.
# Run with:  pytest src/tests/test_model_propagation.py -m llm
# ---------------------------------------------------------------------------

@pytest.mark.llm
class TestRealLLMSampling:
    # A tiny hand-written plan — enough for the model to respond to. We only
    # assert on what WE capture (usage, reasoning, raw response), never on the
    # quality of the optimization.
    PLAN = {
        "structure": {"0": {"parquetScan": {"table": "nation", "projection": [0, 1]}}},
        "succinct_table_info": {"nation": {"row_count": 25, "columns": ["n_nationkey", "n_name"]}},
    }
    QUERY = "SELECT n_nationkey, n_name FROM nation"

    @pytest.fixture(autouse=True)
    def _require_key(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set — real-LLM test skipped")

    def test_real_call_captures_usage_and_reasoning(self):
        """A real sampling call populates token usage, reasoning, and raw response."""
        model = os.getenv("LLM_TEST_MODEL", "gpt-5")
        opt = GPTPlanOptimizer(model=model)
        r = opt.optimize_plan(self.QUERY, self.PLAN, n_samples=1, verbose=False)[0]
        assert r.error_message is None, r.error_message
        assert isinstance(r.prompt_tokens, int) and r.prompt_tokens > 0
        assert isinstance(r.completion_tokens, int) and r.completion_tokens > 0
        assert isinstance(r.total_tokens, int) and r.total_tokens > 0
        assert isinstance(r.reasoning_content, str)  # real trace, or the placeholder
        assert isinstance(r.model_response, str) and r.model_response

    def test_completion_kwargs_bounds_output(self):
        """A small max_tokens cap must bound completion_tokens, proving the
        completion_kwarg actually reached the provider. Uses a cheap non-reasoning
        model for clean max_tokens semantics."""
        model = os.getenv("LLM_TEST_CHEAP_MODEL", "gpt-4o-mini")
        opt = GPTPlanOptimizer(model=model, completion_kwargs={"max_tokens": 16})
        r = opt.optimize_plan(self.QUERY, self.PLAN, n_samples=1, verbose=False)[0]
        assert r.error_message is None, r.error_message
        assert isinstance(r.completion_tokens, int)
        assert r.completion_tokens <= 16
