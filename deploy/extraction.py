"""Deterministic charter extraction pipeline.

Why this exists
---------------
The agentic path (orchestrator → 7 field subagents → verify → structured output)
relies on the model reliably emitting tool calls and following a multi-step plan.
Weak / quirky tool-calling models (e.g. minimax-m3) fail at this: they skip
delegation and emit an empty structured response that echoes the schema's type
words ("list", "string").

This module removes the model's freedom to go wrong. Orchestration is done in
Python: for each output field we (1) retrieve the most relevant chunks of the
charter with BM25, then (2) make ONE small, single-purpose LLM call that returns
just that field as JSON. Small context + one task per call + tolerant parsing =
stable results even on a weak model. Field calls are independent, so they run
concurrently.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage

from tools import _chunk_text  # reuse the same chunker as the search tools


# One spec per CharterStructuredOutput field.
#   key      : schema field name
#   ru       : human label (Russian) used in prompt + report
#   kind     : "list" -> list[str]; "str" -> str
#   keywords : BM25 query to retrieve the relevant charter section
#   topic    : what to look for, passed to the model
FIELD_SPECS: list[dict] = [
    {
        "key": "supreme_governing_body", "kind": "str", "style": "name",
        "ru": "Высший орган управления",
        "keywords": "общее собрание участников высший орган управления компетенция",
        "topic": "Высший орган управления (обычно Общее собрание участников).",
    },
    {
        "key": "collegial_governing_bodies", "kind": "list", "style": "name",
        "ru": "Коллегиальные органы управления",
        "keywords": "совет директоров наблюдательный совет правление коллегиальный орган",
        "topic": "Коллегиальные органы управления: Совет директоров, Наблюдательный совет, Правление.",
    },
    {
        "key": "sole_executive_bodies", "kind": "list", "style": "name",
        "ru": "Единоличные исполнительные органы",
        "keywords": "генеральный директор единоличный исполнительный орган директор управляющий президент",
        "topic": ("Только ЕДИНОЛИЧНЫЕ исполнительные органы: Генеральный директор, Директор, "
                  "Управляющий, Президент. НЕ включай Правление/Дирекцию (это коллегиальные органы)."),
    },
    {
        "key": "major_transaction_clauses", "kind": "list",
        "ru": "Пункты о крупных сделках",
        "keywords": ("крупная сделка крупные сделки процент балансовой стоимости активов "
                     "одобрение порог компетенция общее собрание акционеров участников "
                     "наблюдательный совет совет директоров статья 79"),
        "topic": ("Пункты устава о крупных сделках (пороги, % от активов, кто одобряет). "
                  "Собери из компетенции ВСЕХ органов: и Общего собрания, и Наблюдательного "
                  "совета / Совета директоров — не останавливайся на первом разделе."),
    },
    {
        "key": "related_party_transaction_clauses", "kind": "list",
        "ru": "Пункты о сделках с заинтересованностью",
        "keywords": ("сделка с заинтересованностью заинтересованные лица одобрение статья 45 "
                     "статья 83 компетенция общее собрание акционеров участников "
                     "наблюдательный совет совет директоров"),
        "topic": ("Пункты устава о сделках с заинтересованностью. Собери из компетенции ВСЕХ "
                  "органов: и Общего собрания, и Наблюдательного совета / Совета директоров — "
                  "не останавливайся на первом разделе."),
    },
    {
        "key": "general_meeting_minutes_protocol", "kind": "str",
        "ru": "Протокол общего собрания (способ удостоверения)",
        "keywords": "протокол общего собрания удостоверение нотариус способ подтверждение решений",
        "topic": "Протокол ОСУ: номер пункта + способ удостоверения решений (нотариус / иной способ).",
    },
    {
        "key": "sole_executive_body_restrictions", "kind": "list",
        "ru": "Уставные ограничения единоличного ИО",
        "keywords": "ограничения полномочий генерального директора предварительное согласие одобрение совершение сделок",
        "topic": "Уставные ограничения полномочий единоличного исполнительного органа.",
    },
]

# Values the model emits when it gives up / echoes the schema — never real data.
_PLACEHOLDERS = {"", "list", "string", "str", "list[str]", "[]", "{}",
                 "не указано", "нет", "none", "null", "n/a"}


# --------------------------------------------------------------------- parsing
# Dict keys the model uses for the clause number vs. the clause content. Used to
# flatten an object like {"пункт": "7.1", "значение": "..."} into one string.
_NUM_KEYS = ("пункт", "номер", "clause", "clause_number", "number", "статья")
_TXT_KEYS = ("значение", "текст", "value", "text", "содержание", "описание",
             "способ_удостоверения", "способ", "наименование", "название")


def _strip_fences(text: str) -> str:
    """Remove markdown code fences of ANY backtick count (``` or `````` …),
    with or without a language tag, anywhere in the text."""
    s = re.sub(r"`{3,}[ \t]*[a-zA-Z]*[ \t]*\n?", "", text)
    s = s.replace("`", "")
    return s.strip()


def _flatten_value(v) -> str:
    """Turn a dict/list/scalar into one human-readable string, clause-number first."""
    if isinstance(v, dict):
        nums = [str(v[k]).strip() for k in _NUM_KEYS if k in v and str(v[k]).strip()]
        txts = [str(v[k]).strip() for k in _TXT_KEYS if k in v and str(v[k]).strip()]
        parts = nums + txts
        if not parts:  # unknown keys → join scalar values in order
            parts = [str(x).strip() for x in v.values()
                     if isinstance(x, (str, int, float)) and str(x).strip()]
        return ". ".join(parts)
    if isinstance(v, list):
        return "; ".join(_flatten_value(x) for x in v if x not in (None, ""))
    return str(v).strip()


def _match_bracket(s: str, start: int) -> int:
    """Index of the bracket that closes the one at `start`, or -1. String-aware."""
    open_ch = s[start]
    close_ch = "]" if open_ch == "[" else "}"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _find_json_values(s: str) -> list:
    """Scan for top-level JSON arrays/objects and return each parsed value, in
    order. Tolerates repeated blocks and surrounding prose."""
    out, i = [], 0
    while i < len(s):
        if s[i] in "[{":
            end = _match_bracket(s, i)
            if end != -1:
                try:
                    out.append(json.loads(s[i:end + 1]))
                    i = end + 1
                    continue
                except (ValueError, TypeError):
                    pass
        i += 1
    return out


def _normalize_list(values: list) -> list:
    """Flatten parsed values into a flat list of strings, unwrapping any element
    that is itself a stringified JSON array."""
    items: list[str] = []
    for v in values:
        if isinstance(v, list):
            items.extend(_normalize_list(v))
        elif isinstance(v, dict):
            s = _flatten_value(v)
            if s:
                items.append(s)
        else:
            s = str(v).strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    inner = json.loads(s)
                    if isinstance(inner, list):
                        items.extend(_normalize_list(inner))
                        continue
                except (ValueError, TypeError):
                    pass
            if s:
                items.append(s)
    return items


def _tolerant_json(raw: str, kind: str):
    """Parse a model response into list[str] (kind='list') or str (kind='str'),
    tolerating prose, code fences, JSON objects, repeated/duplicated blocks, and
    bullet lists."""
    if not raw or not raw.strip():
        return [] if kind == "list" else ""
    s = _strip_fences(raw)
    values = _find_json_values(s)

    if kind == "list":
        if values:
            return _normalize_list(values)
        # No JSON found → split lines, stripping only bullet markers (NOT digits,
        # which are clause numbers).
        items = []
        for ln in s.splitlines():
            ln = re.sub(r"^[\-\*•\s]+", "", ln).strip().strip(",")
            if ln:
                items.append(ln)
        return items

    # kind == "str"
    if values:
        return _flatten_value(values[0])
    return s.strip().strip('"').strip()


def _coerce(val, kind: str):
    """Normalize to the field's type, drop placeholder/schema-echo values, and
    de-duplicate list items (the model sometimes repeats the whole answer)."""
    if kind == "list":
        if not isinstance(val, list):
            val = [] if val is None else [str(val)]
        out, seen = [], set()
        for x in val:
            xs = str(x).strip()
            low = xs.lower()
            if xs and low not in _PLACEHOLDERS and low not in seen:
                seen.add(low)
                out.append(xs)
        return out
    s = "" if val is None else str(val).strip()
    return s if s and s.lower() not in _PLACEHOLDERS else "не указано"


# ------------------------------------------------------------------ retrieval
def _build_bm25(chunks: list[str]):
    from rank_bm25 import BM25Okapi
    tokenized = [re.findall(r"\w+", c.lower()) for c in chunks]
    return BM25Okapi(tokenized)


def _hybrid_enabled() -> bool:
    return os.getenv("EXTRACTION_HYBRID", "1").strip().lower() in ("1", "true", "yes", "on")


def _rrf(rank_lists: list[tuple[list, float]], k: int = 60) -> dict:
    """Weighted Reciprocal Rank Fusion. rank_lists = [(ranked_indices, weight), ...].
    A doc highly ranked in either list gets a boost; weights tilt the balance."""
    scores: dict[int, float] = {}
    for ranked, weight in rank_lists:
        for rank, idx in enumerate(ranked):
            i = int(idx)
            scores[i] = scores.get(i, 0.0) + weight * (1.0 / (k + rank + 1))
    return scores


class _Retriever:
    """Hybrid retriever: BM25 (lexical, precise on clause numbers/terms) fused
    with dense vectors (semantic recall) via RRF. Vectors are best-effort — if
    embeddings can't be built (offline / no key / error) it transparently runs
    BM25-only, so the pipeline always works in a closed network."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.bm25 = _build_bm25(chunks)
        self.vecs = None      # normalized (n, d) matrix, or None
        self.embedder = None
        if _hybrid_enabled():
            self._build_vectors()
        else:
            print("[extraction] retrieval: BM25 only (hybrid disabled)")

    def _build_vectors(self) -> None:
        try:
            import numpy as np
            from tools import _get_embeddings
            emb = _get_embeddings()
            mat = np.asarray(emb.embed_documents(self.chunks), dtype=np.float32)
            if mat.ndim != 2 or mat.shape[0] != len(self.chunks):
                raise ValueError("unexpected embedding shape")
            self.vecs = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
            self.embedder = emb
            print(f"[extraction] retrieval: hybrid BM25+vector (RRF) over {len(self.chunks)} chunks")
        except Exception as e:
            self.vecs = None
            reason = (str(e).splitlines() or [type(e).__name__])[0][:140]
            print(f"[extraction] retrieval: BM25 only — embeddings unavailable ({reason})")

    def _bm25_ranked(self, keywords: str, cand: int) -> list[int]:
        q = re.findall(r"\w+", keywords.lower())
        scores = self.bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked if scores[i] > 0][:cand]

    def _vec_ranked(self, keywords: str, cand: int) -> list[int]:
        if self.vecs is None:
            return []
        try:
            import numpy as np
            qv = np.asarray(self.embedder.embed_query(keywords), dtype=np.float32)
            qv = qv / (np.linalg.norm(qv) + 1e-9)
            sims = self.vecs @ qv
            return [int(i) for i in np.argsort(-sims)[:cand]]
        except Exception:
            return []

    def context(self, keywords: str, top_k: int, bm25_weight: float = 0.7) -> str:
        cand = max(top_k * 3, 15)
        bm_ranked = self._bm25_ranked(keywords, cand)
        vec_ranked = self._vec_ranked(keywords, cand)

        if not vec_ranked:                       # BM25-only path
            chosen = bm_ranked[:top_k]
        else:
            fused = _rrf([(bm_ranked, bm25_weight), (vec_ranked, 1.0 - bm25_weight)])
            chosen = [i for i, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)][:top_k]
            chosen = chosen or bm_ranked[:top_k]

        if not chosen:
            chosen = [0] if self.chunks else []
        chosen = sorted(set(chosen))             # restore document order
        return "\n\n".join(self.chunks[i] for i in chosen)


# -------------------------------------------------------------------- the LLM
def _invoke_text(model, system: str, user: str) -> str:
    try:
        resp = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    except Exception:
        return ""
    content = getattr(resp, "content", resp)
    return content if isinstance(content, str) else str(content)


def _extract_field(model, spec: dict, context: str):
    kind = spec["kind"]
    # Two field classes need different output shapes:
    #   style="name"   → a canonical organ NAME (normalize, do NOT quote a sentence
    #                     or paste a clause number) — e.g. "Наблюдательный совет".
    #   style="clause" → a verbatim clause "<number>. <text>" (the default).
    style = spec.get("style", "clause")
    if style == "name":
        if kind == "list":
            fmt = ('Верни ТОЛЬКО JSON-массив КОРОТКИХ НАЗВАНИЙ органов, например '
                   '["Наблюдательный совет", "Правление"]. Каждый элемент — каноническое '
                   'НАЗВАНИЕ органа, БЕЗ номера пункта/статьи и БЕЗ целого предложения из '
                   'устава. НЕ копируй определительные предложения дословно — выдели только '
                   'название. НЕ используй объекты {{...}}, НЕ оборачивай в markdown/```. '
                   'Если органа нет — верни [].')
        else:
            fmt = ('Верни ТОЛЬКО ОДНУ строку — каноническое НАЗВАНИЕ органа, например '
                   '"Общее собрание акционеров". БЕЗ номера пункта/статьи и БЕЗ целого '
                   'предложения. Если нет — "не указано".')
    elif kind == "list":
        fmt = ('Верни ТОЛЬКО JSON-массив строк, например ["7.1. текст", "7.2. текст"]. '
               'Каждый элемент — ОДНА строка вида "<номер пункта>. <дословный текст>". '
               'Указывай САМЫЙ КОНКРЕТНЫЙ (нижний) номер подпункта, например 12.1.4(3), '
               'а не родительский пункт целиком. '
               'НЕ используй объекты {{...}} с ключами, НЕ оборачивай в markdown/```, '
               'НЕ повторяй массив дважды. Если темы нет в ОТРЫВКАХ — верни [].')
    else:
        fmt = ('Верни ТОЛЬКО ОДНУ строку в двойных кавычках, например '
               '"7.1. Общее собрание участников". НЕ используй объект {{...}} с ключами, '
               'НЕ выводи JSON-объект. Если темы нет в ОТРЫВКАХ — верни "не указано".')

    system = (
        "Ты извлекаешь одно поле из устава ООО. Работай СТРОГО по приведённым "
        "ОТРЫВКАМ — не используй внешние знания и не придумывай номера пунктов. "
        "Никогда не выводи слова 'list' или 'string'. " + fmt
    )
    user = f"ПОЛЕ: {spec['ru']}\nЧТО ИСКАТЬ: {spec['topic']}\n\nОТРЫВКИ:\n{context}"

    raw = _invoke_text(model, system, user)
    val = _tolerant_json(raw, kind)
    if val in (None, [], "") or (kind == "str" and str(val).lower() in _PLACEHOLDERS):
        # one focused retry
        raw = _invoke_text(model, system, user + "\n\nВАЖНО: ответ — ТОЛЬКО валидный JSON, без пояснений.")
        retried = _tolerant_json(raw, kind)
        if retried not in (None, [], ""):
            val = retried
    return _coerce(val, kind)


# ----------------------------------------------------------------- public API
# ------------------------------------------------------------------- chunking
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")                       # ## 9. Совет директоров
_CLAUSE_RE = re.compile(r"^\s*(?:[-*]\s*)?\d+(?:\.\d+)*\.?\s+\S")      # 9 / 9.1 / 9.7.1
_NOISE_RE = re.compile(r"^\s*(?:---\s*Page\s+\d+.*|Total pages:.*|<!--.*-->)\s*$", re.IGNORECASE)


def _size_split(text: str, size: int, overlap: int = 150) -> list[str]:
    size = max(200, size)
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, len(text), step)] or [text]


def _split_clause_units(lines: list[str]) -> list[str]:
    """Start a new unit at each top-level clause-number line; keep sub-lines with it."""
    units, cur = [], []
    for ln in lines:
        if _CLAUSE_RE.match(ln) and cur:
            units.append("\n".join(cur))
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        units.append("\n".join(cur))
    return units


def _merge_small(units: list[str], target: int) -> list[str]:
    """Merge adjacent clause units up to `target` chars so chunks aren't tiny."""
    merged, buf = [], ""
    for u in units:
        if not buf:
            buf = u
        elif len(buf) + len(u) + 1 <= target:
            buf = buf + "\n" + u
        else:
            merged.append(buf)
            buf = u
    if buf:
        merged.append(buf)
    return merged


def hierarchical_chunks(text: str, max_chars: int = 1800, overlap: int = 150) -> list[str]:
    """Structure-aware chunking for charter markdown: group by markdown headings,
    split each section body at clause-number boundaries, merge tiny clauses, and
    prepend the section heading to every chunk (small-to-large parent context).
    Falls back to fixed-size splitting for unstructured text / oversized clauses."""
    # Drop page-marker/image noise and collapse the long whitespace runs docling
    # emits for table-cell padding (pure token waste for extraction).
    lines = []
    for ln in text.splitlines():
        if _NOISE_RE.match(ln):
            continue
        lines.append(re.sub(r"[ \t]{2,}", " ", ln).rstrip())

    sections: list[tuple[str, list[str]]] = []
    heading, body = "", []
    for ln in lines:
        if _HEADING_RE.match(ln):
            if heading or body:
                sections.append((heading, body))
            heading, body = ln.strip().lstrip("#").strip(), []
        else:
            body.append(ln)
    if heading or body:
        sections.append((heading, body))

    chunks: list[str] = []
    for heading, body in sections:
        for unit in _merge_small(_split_clause_units(body), max_chars - 80):
            unit = unit.strip()
            if not unit:
                continue
            prefix = f"[{heading}]\n" if heading else ""
            if len(prefix) + len(unit) <= max_chars:
                chunks.append(prefix + unit)
            else:
                chunks.extend(prefix + piece for piece in _size_split(unit, max_chars - len(prefix), overlap))

    chunks = [c for c in chunks if c.strip()]
    return chunks or _size_split(text, max_chars, overlap)


def extract_charter(model, md_texts: list[str], specs: list[dict] | None = None,
                    top_k: int | None = None, workers: int | None = None) -> dict:
    """Extract every field deterministically from the charter markdown.

    md_texts: one or more markdown strings (e.g. scratch/*.md). Returns a plain
    dict keyed by CharterStructuredOutput field names.
    """
    specs = specs or FIELD_SPECS
    combined = "\n\n".join(t for t in md_texts if t)
    if not combined:
        return {s["key"]: ([] if s["kind"] == "list" else "не указано") for s in specs}

    chunk_mode = os.getenv("EXTRACTION_CHUNKING", "hierarchical").strip().lower()
    if chunk_mode == "fixed":
        chunks = _chunk_text(combined) or [combined]
    else:
        chunks = hierarchical_chunks(combined) or [combined]

    if top_k is None:
        top_k = int(os.getenv("EXTRACTION_TOP_K", "12"))
    bm25_weight = float(os.getenv("EXTRACTION_BM25_WEIGHT", "0.7"))
    retriever = _Retriever(chunks)

    def _work(spec: dict):
        ctx = retriever.context(spec["keywords"], top_k, bm25_weight)
        return spec["key"], _extract_field(model, spec, ctx)

    if workers is None:
        workers = int(os.getenv("EXTRACTION_WORKERS", "4"))

    results: dict = {}
    if workers and workers > 1 and len(specs) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(specs))) as ex:
            for key, value in ex.map(_work, specs):
                results[key] = value
    else:
        for spec in specs:
            key, value = _work(spec)
            results[key] = value

    for spec in specs:  # guarantee every field is present
        results.setdefault(spec["key"], [] if spec["kind"] == "list" else "не указано")
    return results


def markdown_for_inputs(input_dir) -> list[str]:
    """Convert each PDF under input_dir to markdown (via the cached read_pdf) and
    return the markdown texts. Non-PDFs are read as-is."""
    from pathlib import Path
    from tools import read_pdf, _work_root

    texts: list[str] = []
    for f in sorted(Path(input_dir).iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue
        if f.suffix.lower() == ".pdf":
            read_pdf.func(f"/input/{f.name}")  # populates /scratch/<stem>.md (cached)
            md_path = _work_root() / "scratch" / f"{f.stem}.md"
            if md_path.exists():
                texts.append(md_path.read_text(encoding="utf-8"))
        else:
            try:
                texts.append(f.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    return texts


def render_markdown_report(structured: dict, specs: list[dict] | None = None) -> str:
    """Human-readable report built from the structured result."""
    specs = specs or FIELD_SPECS
    lines = ["## Извлечённые поля\n"]
    for spec in specs:
        v = structured.get(spec["key"])
        lines.append(f"### {spec['ru']} (`{spec['key']}`)")
        if isinstance(v, list):
            lines.extend(f"- {item}" for item in v) if v else lines.append("- (не найдено)")
        else:
            lines.append(str(v) if v else "(не найдено)")
        lines.append("")
    return "\n".join(lines)
