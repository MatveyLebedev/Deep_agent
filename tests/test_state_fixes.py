"""Phase-6 state fixes: prompt slimming (no dead /memories/ step, no todo
ceremony) and the HITL-gated checkpointer."""
import pytest

try:
    import main
    from main import _build_prompt, build_agent
except Exception as e:  # pragma: no cover - env-dependent
    pytest.skip(f"main.py not importable here: {e}", allow_module_level=True)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


class TestPromptSlimming:
    def test_no_dead_memories_step(self):
        p = _build_prompt("BUSINESS RULES HERE")
        assert "/memories/" not in p
        assert "STEP 5" not in p

    def test_no_todo_ceremony(self):
        p = _build_prompt("rules")
        assert "write_todos" not in p

    def test_steps_renumbered_and_complete(self):
        p = _build_prompt("rules")
        assert "STEP 1 - STARTUP" in p
        assert "STEP 2 - DELEGATE" in p
        assert "STEP 3 - AGGREGATE" in p
        assert "STEP 4" not in p

    def test_core_content_still_present(self):
        p = _build_prompt("MARKER-RULES")
        assert "MARKER-RULES" in p
        assert "=== REQUIRED OUTPUT SCHEMA ===" in p
        assert "extract-major-transactions" in p


class TestCheckpointerGating:
    def _build(self, tmp_path):
        (tmp_path / "instructions").mkdir(exist_ok=True)
        agent, _model = build_agent(
            instructions_root=tmp_path / "instructions",
            skills_root=tmp_path / "skills",
            business_rules="rules",
            agent_root=tmp_path,
        )
        return agent

    def test_memory_saver_without_hitl(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "HITL_ENABLED", False)
        agent = self._build(tmp_path)
        assert isinstance(agent.checkpointer, MemorySaver)
        assert not (tmp_path / "checkpoint.db").exists()

    def test_sqlite_saver_with_hitl(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "HITL_ENABLED", True)
        agent = self._build(tmp_path)
        assert isinstance(agent.checkpointer, SqliteSaver)
        assert (tmp_path / "checkpoint.db").exists()
