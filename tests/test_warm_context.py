"""Phase-5: per-field retrieval warm-up baked into subagent prompts."""
import pytest

try:
    import main
    from main import _build_doc_context, _build_field_subagents, _build_subagent_prompt
except Exception as e:  # pragma: no cover - env-dependent
    pytest.skip(f"main.py not importable here: {e}", allow_module_level=True)

CHARTER_MD = """## Статья 11. ОБЩЕЕ СОБРАНИЕ АКЦИОНЕРОВ

11.1. Компетенция Общего собрания акционеров

## Статья 12. НАБЛЮДАТЕЛЬНЫЙ СОВЕТ

12.1.4. одобрение крупных сделок стоимостью от 25 до 50 процентов
балансовой стоимости активов

## Статья 13. ИСПОЛНИТЕЛЬНЫЕ ОРГАНЫ

13.1. Генеральный директор осуществляет руководство
"""


class TestSubagentPrompt:
    def test_without_preload_keeps_cold_workflow(self):
        p = _build_subagent_prompt("major_transaction_clauses", "Крупные сделки")
        assert "read_pdf('/input/<file>')" in p
        assert "search_hybrid" in p
        assert "PRELOADED CONTEXT" not in p

    def test_with_preload_swaps_workflow_and_embeds_context(self):
        p = _build_subagent_prompt(
            "major_transaction_clauses", "Крупные сделки",
            preload="[Статья 12]\n12.1.4. одобрение крупных сделок",
        )
        assert "PRELOADED CONTEXT" in p
        assert "12.1.4. одобрение крупных сделок" in p
        assert "read_section" in p                      # verify-in-full step stays
        assert "read_pdf('/input/<file>')" not in p     # cold workflow replaced

    def test_output_format_tail_preserved(self):
        name = _build_subagent_prompt("supreme_governing_body", "Высший орган",
                                      style="name", preload="ctx")
        clause = _build_subagent_prompt("major_transaction_clauses", "Сделки",
                                        preload="ctx")
        assert "canonical ORGAN NAME" in name
        assert "MOST SPECIFIC sub-clause number" in clause


class TestBuildDocContext:
    @pytest.fixture
    def work_root(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORK_ROOT", str(tmp_path))
        monkeypatch.setenv("EXTRACTION_HYBRID", "0")  # no network in tests
        (tmp_path / "scratch").mkdir()
        (tmp_path / "scratch" / "doc.md").write_text(CHARTER_MD, encoding="utf-8")
        return tmp_path

    def test_returns_relevant_context_per_field(self, work_root):
        ctx = _build_doc_context("doc")
        assert ctx is not None
        assert "12.1.4" in ctx["major_transaction_clauses"]
        assert "Генеральный директор" in ctx["sole_executive_bodies"]

    def test_missing_markdown_returns_none(self, work_root):
        assert _build_doc_context("no-such-stem") is None

    def test_context_capped_by_env(self, work_root, monkeypatch):
        monkeypatch.setenv("SUBAGENT_PRELOAD_MAX_CHARS", "50")
        ctx = _build_doc_context("doc")
        assert ctx and all(len(v) <= 50 for v in ctx.values())


class TestBuildFieldSubagents:
    def test_preload_lands_in_matching_subagent_only(self, monkeypatch):
        monkeypatch.setenv("EXTRACTION_HYBRID", "0")
        doc_context = {"major_transaction_clauses": "WARM-CTX-MAJOR"}
        subs = _build_field_subagents(doc_context)
        by_name = {s["name"]: s for s in subs}
        assert "WARM-CTX-MAJOR" in by_name["extract-major-transactions"]["system_prompt"]
        assert "WARM-CTX-MAJOR" not in by_name["extract-supreme-body"]["system_prompt"]
        assert "PRELOADED CONTEXT" not in by_name["extract-supreme-body"]["system_prompt"]

    def test_no_context_builds_cold_prompts(self):
        subs = _build_field_subagents(None)
        assert all("PRELOADED CONTEXT" not in s["system_prompt"] for s in subs)

    def test_name_fields_get_narrow_toolset(self):
        subs = _build_field_subagents(None)
        by_name = {s["name"]: s for s in subs}
        name_tools = {t.name for t in by_name["extract-supreme-body"]["tools"]}
        clause_tools = {t.name for t in by_name["extract-major-transactions"]["tools"]}
        assert "search_examples" not in name_tools
        assert "search_examples" in clause_tools
        assert "search_hybrid" in name_tools
