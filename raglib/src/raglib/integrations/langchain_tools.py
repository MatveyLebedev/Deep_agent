"""Optional LangChain @tool wrappers (extra [langchain]) so an agent framework
can drive a RagIndex. Import only when langchain-core is installed."""
from __future__ import annotations

from raglib.api import RagIndex


def _location(h) -> str:
    """Provenance header for a hit: document name + titled path chapter→paragraph,
    e.g. 'charter.md • Статья 12. Наблюдательный совет › 12.1. … › 12.1.4'."""
    where = h.breadcrumb or f"пункт {h.clause_number or '(без номера)'}"
    return f"{h.doc_name or h.doc_id} • {where}"


def make_tools(index: RagIndex, llm=None) -> list:
    try:
        from langchain_core.tools import tool
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "langchain-core is required: pip install raglib[langchain]") from e

    @tool
    def rag_search(query: str, method: str = "hybrid", top_k: int = 5,
                   section: str = "", doc: str = "") -> str:
        """Search the indexed documents. Returns whole numbered clauses.
        method: bm25 | vector | hybrid. Optional filters: section key, doc id."""
        hits = index.search(query, method=method, top_k=top_k,
                            section=section or None, doc=doc or None)
        if not hits:
            return "No matches."
        return "\n\n".join(
            f"--- {_location(h)} (score={h.score:.3f}) ---\n{h.text}" for h in hits)

    @tool
    def rag_toc(doc: str = "") -> str:
        """Show the section outline (table of contents) of the indexed documents."""
        return index.toc(doc=doc or None)

    @tool
    def rag_read_section(doc_id: str, section: str) -> str:
        """Read ONE whole section (with sub-sections) by key like '12.1' or title."""
        return index.read_section(doc_id, section)

    @tool
    def rag_agentic_search(prompt: str, top_k: int = 8) -> str:
        """Agentic search: plan -> multi-tool retrieve -> LLM relevance
        reflection -> refine. Returns whole clauses with numbers."""
        res = index.agentic_search(prompt, llm=llm, top_k=top_k)
        lines = [f"degraded={res.degraded} iterations={res.iterations}"]
        for h in res.hits:
            lines.append(f"--- {_location(h)} [{h.verdict or '-'}] ---\n{h.text}")
        return "\n\n".join(lines)

    return [rag_search, rag_toc, rag_read_section, rag_agentic_search]
