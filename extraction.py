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

from tools import _chunk_text  # legacy fixed-size chunker (EXTRACTION_CHUNKING=fixed)
from retrieval import Retriever, hierarchical_chunks, _size_split  # noqa: F401  (re-exported)


# One spec per CharterStructuredOutput field — the shared single-source list
# (field_specs.py). This pipeline uses: key, kind, style, ru, keywords, topic.
from field_specs import FIELD_SPECS

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


# Retrieval (hierarchical chunking + hybrid BM25/vector RRF) lives in
# retrieval.py, shared with the agent path's search tools.


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
    val = _coerce(val, kind)
    if style == "name":  # belt & suspenders: the prompt already forbids prefixes
        val = strip_clause_prefix(val)
    return val


def extract_field_from_context(model, spec: dict, context: str):
    """One small, single-purpose LLM call: extract the field described by
    `spec` from `context`, with tolerant parsing and placeholder stripping.

    Public entry point shared by the deterministic pipeline (context = retrieved
    chunks) and the agent path's finalization (context = subagent reports)."""
    return _extract_field(model, spec, context)


# --------------------------------------------------------------- verification
# Leading clause number of an extracted item: "12.1.4. текст" -> "12.1.4".
_LEADING_CLAUSE_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)")


def leading_clause_number(item: str) -> str:
    m = _LEADING_CLAUSE_NUM_RE.match(str(item))
    return m.group(1) if m else ""


def strip_clause_prefix(value):
    """Remove a leading clause-number prefix from organ NAME value(s):
    '12.1. Наблюдательный совет' -> 'Наблюдательный совет'. Never empties a
    value: if nothing remains after stripping, the original is kept."""
    def _one(s: str) -> str:
        stripped = re.sub(r"^\s*\d+(?:\.\d+)*[.)\s:—–-]*", "", s).strip()
        return stripped or s
    if isinstance(value, list):
        return [_one(str(x)) for x in value]
    return _one(str(value))


def clause_in_document(num: str, doc_text: str) -> bool:
    """True if `num` occurs in the document as a clause label — i.e. not as a
    digit-run inside a longer number ('12.1' does NOT match inside '12.1.4',
    and '2.1' does not match inside '12.1')."""
    if not num or not doc_text:
        return False
    pattern = rf"(?<![\d.]){re.escape(num)}(?!\.?\d)"
    return re.search(pattern, doc_text) is not None


def verify_extracted_fields(structured: dict, doc_text: str,
                            specs: list[dict] | None = None) -> list[str]:
    """Flag-only integrity check of extracted values against the source text.
    Returns human-readable warnings; values are never mutated or dropped here
    (an LLM-independent second opinion, recorded next to the result).

    Checks:
      * CLAUSE items whose leading clause number never appears in the document
        (the classic weak-model failure: clause IDs copied from few-shot
        examples instead of /input/);
      * NAME values that still look like clause quotes (start with a number).
    """
    specs = specs or FIELD_SPECS
    warnings: list[str] = []
    for spec in specs:
        v = structured.get(spec["key"])
        if isinstance(v, list):
            items = v
        elif v and str(v).strip().lower() != "не указано":
            items = [v]
        else:
            items = []
        style = spec.get("style", "clause")
        for item in items:
            s = str(item)
            num = leading_clause_number(s)
            if style == "name":
                if num:
                    warnings.append(
                        f"{spec['key']}: название органа содержит номер пункта: {s[:80]!r}")
            elif num and doc_text and not clause_in_document(num, doc_text):
                warnings.append(
                    f"{spec['key']}: пункт {num} не найден в документе: {s[:80]!r}")
    return warnings


# ----------------------------------------------------------------- public API
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
    retriever = Retriever(chunks)

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
