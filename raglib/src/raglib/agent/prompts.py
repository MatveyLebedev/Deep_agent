"""Prompts for the agentic loop + tolerant JSON parsing.

Designed for a weak model (lesson from Deep agent / extraction.py): each call
is one small task, output is strict JSON, and the parser tolerates code fences,
chatter around the JSON and truncated tails.
"""
from __future__ import annotations

import json
import re
from typing import Any

SYSTEM = (
    "Ты — помощник по поиску в юридических документах (уставы, регламенты). "
    "Отвечай ТОЛЬКО валидным JSON, без пояснений, без markdown."
)

PLAN_TEMPLATE = (
    "Запрос пользователя: «{prompt}»\n\n"
    "Составь план поиска по документам. Верни JSON вида:\n"
    '{{"queries": ["до 3 коротких поисковых запросов (ключевые слова, синонимы)"], '
    '"aspects": ["какие аспекты запроса должны быть покрыты найденными пунктами"]}}'
)

REFLECT_TEMPLATE = (
    "Запрос пользователя: «{prompt}»\n"
    "Аспекты, которые нужно покрыть:\n{aspects}\n\n"
    "Найденные пункты-кандидаты:\n{candidates}\n\n"
    "Оцени КАЖДОГО кандидата. Верни JSON-массив вида:\n"
    '[{{"id": "C1", "verdict": "relevant" | "partial" | "irrelevant", '
    '"aspects": [номера покрытых аспектов], "missing": "чего не хватает (или пусто)"}}]'
)

REFINE_TEMPLATE = (
    "Запрос пользователя: «{prompt}»\n"
    "Непокрытые аспекты:\n{uncovered}\n"
    "Замечания о том, чего не хватает в найденном:\n{missing}\n\n"
    "Предложи новые поисковые запросы (другие формулировки, синонимы, номера "
    "статей). Верни JSON вида:\n"
    '{{"queries": ["до 3 новых запросов"]}}'
)


def _balanced_slice(text: str, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_json(text: str) -> Any | None:
    """Tolerant JSON extraction: strips <think> reasoning blocks and code
    fences, finds the first balanced {...} or [...] block; returns None when
    nothing parseable is found."""
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()
    for candidate in (cleaned,
                      _balanced_slice(cleaned, "[", "]"),
                      _balanced_slice(cleaned, "{", "}")):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return None


def str_list(value, limit: int) -> list[str]:
    """Coerce an LLM-provided value into a bounded list of non-empty strings."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = [str(v).strip() for v in value if str(v).strip()]
    return out[:limit]
