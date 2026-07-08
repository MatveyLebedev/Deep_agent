"""Phase-1 hardening: token budget enforcement and salvage of aborted runs."""
from types import SimpleNamespace

import pytest

try:
    from main import Agent, BudgetExceeded, _FixedToolIdModel
except Exception as e:  # pragma: no cover - env-dependent
    pytest.skip(f"main.py not importable here: {e}", allow_module_level=True)

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult


def _model(budget: int) -> _FixedToolIdModel:
    return _FixedToolIdModel(model="test/model", openai_api_key="k", token_budget=budget)


def _result(total_tokens: int | None, llm_output: dict | None = None) -> LLMResult:
    usage = None
    if total_tokens is not None:
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": total_tokens}
    msg = AIMessage(content="", usage_metadata=usage)
    return LLMResult(generations=[[ChatGeneration(message=msg)]], llm_output=llm_output)


class TestTokenBudget:
    def test_under_budget_passes(self):
        m = _model(100)
        m._register_usage(_result(60))
        m._check_budget()  # no raise

    def test_spent_budget_raises_before_next_call(self):
        m = _model(100)
        m._register_usage(_result(60))
        m._register_usage(_result(60))
        with pytest.raises(BudgetExceeded, match="120/100"):
            m._check_budget()

    def test_zero_budget_means_unlimited(self):
        m = _model(0)
        m._register_usage(_result(10**9))
        m._check_budget()  # no raise

    def test_usage_accumulates_across_calls(self):
        m = _model(1000)
        m._register_usage(_result(15))
        m._register_usage(_result(25))
        assert m.tokens_used == 40

    def test_fallback_to_llm_output_aggregate(self):
        m = _model(1000)
        m._register_usage(_result(None, llm_output={"token_usage": {"total_tokens": 42}}))
        assert m.tokens_used == 42

    def test_no_usage_reported_counts_nothing(self):
        m = _model(1000)
        m._register_usage(_result(None))
        assert m.tokens_used == 0

    def test_budget_liftable_for_finalization(self):
        m = _model(10)
        m._register_usage(_result(50))
        m.token_budget = 0  # what run() does before _finalize_structured
        m._check_budget()  # no raise


class TestSalvageState:
    def test_reads_checkpointed_messages(self, tmp_path):
        msgs = [AIMessage(content="findings")]
        fake_agent = SimpleNamespace(
            get_state=lambda cfg: SimpleNamespace(values={"messages": msgs})
        )
        out = Agent(name="t", root=tmp_path)._salvage_state(fake_agent, {})
        assert out["messages"] == msgs

    def test_survives_broken_checkpointer(self, tmp_path):
        def boom(cfg):
            raise RuntimeError("db locked")

        fake_agent = SimpleNamespace(get_state=boom)
        out = Agent(name="t", root=tmp_path)._salvage_state(fake_agent, {})
        assert out["messages"] == []

    def test_survives_empty_state(self, tmp_path):
        fake_agent = SimpleNamespace(get_state=lambda cfg: None)
        out = Agent(name="t", root=tmp_path)._salvage_state(fake_agent, {})
        assert out["messages"] == []
