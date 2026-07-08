"""Agentic search: prompt in -> whole clauses out, via a CODE-DRIVEN loop.

The model never orchestrates tools (that fails on weak models — the Deep agent
lesson). Control flow is Python; the LLM does three narrow jobs:

    PLAN    prompt -> search queries + aspects to cover
    REFLECT batch relevance verdicts for candidate clauses
    REFINE  reformulate queries from what's still missing

Tool routing is code heuristics; budgets are hard; any LLM failure degrades
gracefully to plain hybrid search (never worse than the non-agentic path).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from raglib.agent import prompts
from raglib.agent.llm import ChatLLM
from raglib.models import SearchHit
from raglib.search.engine import SearchEngine
from raglib.search.toc import find_section

_NUM_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%?")
_SECTION_WORDS_RE = re.compile(r"\b(раздел|статья|глава|оглавлени)", re.IGNORECASE)


@dataclass
class AgenticResult:
    hits: list[SearchHit]
    trace: list[dict] = field(default_factory=list)
    degraded: bool = False
    iterations: int = 0
    llm_calls: int = 0


class AgenticSearcher:
    def __init__(self, engine: SearchEngine, llm: ChatLLM | None, *,
                 top_k: int = 8, max_iters: int = 3, max_llm_calls: int = 8,
                 per_iter_candidates: int = 30, reflect_batch: int = 10,
                 snippet_chars: int = 800):
        self.engine = engine
        self.llm = llm
        self.top_k = top_k
        self.max_iters = max(1, max_iters)
        self.max_llm_calls = max_llm_calls
        self.per_iter_candidates = per_iter_candidates
        self.reflect_batch = reflect_batch
        self.snippet_chars = snippet_chars

    # ------------------------------------------------------------ plumbing
    def _base_method(self) -> str:
        return "hybrid" if (self.engine.data.has_vectors
                            and self.engine.embeddings is not None) else "bm25"

    def _plain(self, prompt: str) -> list[SearchHit]:
        return self.engine.search(prompt, method=self._base_method(),
                                  top_k=self.top_k)

    def _llm_json(self, purpose: str, user_text: str, state: AgenticResult):
        """One budgeted LLM call -> parsed JSON (or None on any failure)."""
        if state.llm_calls >= self.max_llm_calls:
            state.trace.append({"step": purpose, "error": "llm budget exhausted"})
            return None
        state.llm_calls += 1
        try:
            raw = self.llm.complete([  # type: ignore[union-attr]
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": user_text},
            ])
        except Exception as e:
            state.trace.append({"step": purpose, "error": f"llm call failed: {e}"})
            return None
        parsed = prompts.parse_json(raw)
        state.trace.append({"step": purpose, "raw": raw[:500],
                            "parsed": parsed is not None})
        return parsed

    def _degrade(self, prompt: str, state: AgenticResult, reason: str) -> AgenticResult:
        state.trace.append({"step": "degrade", "reason": reason})
        state.hits = self._plain(prompt)
        for h in state.hits:
            h.method = "agentic"
        state.degraded = True
        return state

    # ------------------------------------------------------------ retrieval
    def _retrieve(self, queries: list[str], state: AgenticResult
                  ) -> dict[int, SearchHit]:
        """Multi-tool retrieval; candidates keyed by clause_id, best score wins."""
        method = self._base_method()
        candidates: dict[int, SearchHit] = {}

        def add(hits: list[SearchHit], tool: str) -> None:
            for h in hits:
                cur = candidates.get(h.clause_id)
                if cur is None or h.score > cur.score:
                    candidates[h.clause_id] = h
            state.trace.append({"step": "retrieve", "tool": tool,
                                "found": len(hits)})

        for q in queries:
            add(self.engine.search(q, method=method, top_k=10), f"{method}:{q}")
            # heuristic: exact numbers / percents in the query -> grep them too
            for tok in _NUM_TOKEN_RE.findall(q)[:3]:
                pattern = re.escape(tok.strip())
                try:
                    add(self.engine.grep(pattern, top_k=5), f"grep:{tok.strip()}")
                except re.error:
                    pass
            # heuristic: the user asks about a section -> semantic TOC routing
            if (_SECTION_WORDS_RE.search(q) and self.engine.data.has_vectors
                    and self.engine.embeddings is not None):
                for ref in find_section(self.engine.data, q, semantic=True,
                                        engine=self.engine, top_k=1):
                    if ref.key:
                        add(self.engine.search(q, method=method, top_k=5,
                                               doc=ref.doc_id, section=ref.key),
                            f"toc:{ref.key}")

        top = sorted(candidates.values(), key=lambda h: h.score,
                     reverse=True)[:self.per_iter_candidates]
        return {h.clause_id: h for h in top}

    # ------------------------------------------------------------ reflection
    def _reflect(self, prompt: str, aspects: list[str],
                 batch: list[SearchHit], state: AgenticResult) -> list[dict] | None:
        listing = []
        for i, h in enumerate(batch, start=1):
            snippet = h.text[:self.snippet_chars]  # truncation INSIDE the prompt only
            listing.append(f'C{i} (пункт {h.clause_number or "без номера"}, '
                           f'документ {h.doc_name or h.doc_id}; путь: {h.breadcrumb}):'
                           f'\n{snippet}')
        text = prompts.REFLECT_TEMPLATE.format(
            prompt=prompt,
            aspects="\n".join(f"{i}. {a}" for i, a in enumerate(aspects, start=1)),
            candidates="\n\n".join(listing),
        )
        parsed = self._llm_json("reflect", text, state)
        if not isinstance(parsed, list):
            return None
        verdicts: list[dict] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            m = re.search(r"\d+", str(item.get("id", "")))
            if not m:
                continue
            idx = int(m.group()) - 1
            if not (0 <= idx < len(batch)):
                continue
            verdict = str(item.get("verdict", "")).strip().lower()
            if verdict not in ("relevant", "partial", "irrelevant"):
                continue
            covered = [int(a) for a in item.get("aspects", [])
                       if str(a).strip().lstrip("-").isdigit()]
            verdicts.append({"clause_id": batch[idx].clause_id, "verdict": verdict,
                             "aspects": covered,
                             "missing": str(item.get("missing", "")).strip()})
        return verdicts or None

    # ------------------------------------------------------------ main loop
    def run(self, prompt: str) -> AgenticResult:
        state = AgenticResult(hits=[])
        if self.llm is None:  # legal mode: plain search, no reflection
            state.trace.append({"step": "plan", "note": "llm=None -> plain "
                                + self._base_method()})
            state.hits = self._plain(prompt)
            return state

        plan = self._llm_json("plan", prompts.PLAN_TEMPLATE.format(prompt=prompt),
                              state)
        if not isinstance(plan, dict):
            return self._degrade(prompt, state, "plan failed")
        queries = prompts.str_list(plan.get("queries"), 3) or [prompt]
        aspects = prompts.str_list(plan.get("aspects"), 6) or [prompt]
        state.trace.append({"step": "plan", "queries": queries, "aspects": aspects})

        graded: dict[int, dict] = {}          # clause_id -> verdict record
        best_hit: dict[int, SearchHit] = {}
        covered: set[int] = set()
        missing_notes: list[str] = []

        for iteration in range(1, self.max_iters + 1):
            state.iterations = iteration
            candidates = self._retrieve(queries, state)
            best_hit.update({cid: h for cid, h in candidates.items()
                             if cid not in best_hit or h.score > best_hit[cid].score})

            batch = [h for cid, h in sorted(candidates.items(),
                                            key=lambda kv: kv[1].score, reverse=True)
                     if cid not in graded][:self.reflect_batch]
            if batch:
                verdicts = self._reflect(prompt, aspects, batch, state)
                if verdicts is None:
                    return self._degrade(prompt, state, "reflect failed")
                for v in verdicts:
                    graded[v["clause_id"]] = v
                    covered.update(a - 1 for a in v["aspects"]
                                   if v["verdict"] in ("relevant", "partial"))
                    if v["missing"]:
                        missing_notes.append(v["missing"])
                state.trace.append({"step": "graded",
                                    "verdicts": {str(v["clause_id"]): v["verdict"]
                                                 for v in verdicts}})

            relevant = [cid for cid, v in graded.items() if v["verdict"] == "relevant"]
            all_covered = covered >= set(range(len(aspects)))
            if len(relevant) >= self.top_k or all_covered:
                state.trace.append({"step": "decide", "stop": True,
                                    "relevant": len(relevant),
                                    "aspects_covered": sorted(covered)})
                break
            if iteration == self.max_iters or state.llm_calls >= self.max_llm_calls:
                break

            uncovered = [a for i, a in enumerate(aspects) if i not in covered]
            refine = self._llm_json("refine", prompts.REFINE_TEMPLATE.format(
                prompt=prompt,
                uncovered="\n".join(f"- {a}" for a in uncovered) or "- (нет)",
                missing="\n".join(f"- {n}" for n in missing_notes[-5:]) or "- (нет)",
            ), state)
            if not isinstance(refine, dict):
                state.trace.append({"step": "refine", "note": "failed; stopping "
                                    "with current results"})
                break
            new_queries = prompts.str_list(refine.get("queries"), 3)
            if not new_queries:
                break
            queries = new_queries
            state.trace.append({"step": "refine", "queries": queries})

        order = {"relevant": 0, "partial": 1}
        picked = [(graded[cid]["verdict"], best_hit[cid])
                  for cid in graded
                  if graded[cid]["verdict"] in order and cid in best_hit]
        picked.sort(key=lambda vh: (order[vh[0]], -vh[1].score))
        if not picked:  # LLM worked but accepted nothing: still никогда не хуже hybrid
            state.trace.append({"step": "final",
                                "note": "reflection accepted nothing; "
                                        "returning plain results"})
            state.hits = self._plain(prompt)
            for h in state.hits:
                h.method = "agentic"
            return state

        hits: list[SearchHit] = []
        for verdict, hit in picked[:self.top_k]:
            hit.method = "agentic"
            hit.verdict = verdict
            hits.append(hit)
        state.hits = hits
        return state
