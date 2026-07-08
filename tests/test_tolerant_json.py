"""Parsing helpers in extraction.py — the layer that makes a weak model's
free-form answers usable. Every case here is a real output shape observed
from minimax/deepseek runs."""
import pytest

from extraction import (
    _coerce,
    _find_json_values,
    _flatten_value,
    _strip_fences,
    _tolerant_json,
)


class TestStripFences:
    def test_triple_fence_with_lang(self):
        assert _strip_fences('```json\n["a"]\n```') == '["a"]'

    def test_many_backticks(self):
        assert _strip_fences('``````\n["a"]\n``````') == '["a"]'

    def test_inline_backticks_removed(self):
        assert _strip_fences("пункт `7.1` устава") == "пункт 7.1 устава"


class TestFindJsonValues:
    def test_json_inside_prose(self):
        vals = _find_json_values('Вот ответ: ["7.1. текст"] — готово.')
        assert vals == [["7.1. текст"]]

    def test_repeated_blocks_both_found(self):
        vals = _find_json_values('["a"]\n["a"]')
        assert vals == [["a"], ["a"]]

    def test_string_aware_bracket_matching(self):
        vals = _find_json_values('["скобка ] внутри строки"]')
        assert vals == [["скобка ] внутри строки"]]

    def test_unclosed_bracket_ignored(self):
        assert _find_json_values("[1, 2") == []

    def test_object_value(self):
        assert _find_json_values('{"пункт": "7.1"}') == [{"пункт": "7.1"}]


class TestFlattenValue:
    def test_dict_number_first_then_text(self):
        v = {"текст": "решения удостоверяются нотариусом", "пункт": "7.1"}
        assert _flatten_value(v) == "7.1. решения удостоверяются нотариусом"

    def test_dict_unknown_keys_joins_scalars(self):
        assert _flatten_value({"foo": "a", "bar": 3}) == "a. 3"

    def test_nested_list(self):
        assert _flatten_value(["a", ["b", "c"]]) == "a; b; c"


class TestTolerantJsonList:
    def test_clean_array(self):
        assert _tolerant_json('["7.1. текст", "7.2. текст"]', "list") == [
            "7.1. текст", "7.2. текст",
        ]

    def test_fenced_array(self):
        assert _tolerant_json('```json\n["7.1. т"]\n```', "list") == ["7.1. т"]

    def test_array_of_objects_flattened(self):
        raw = '[{"пункт": "12.1.4", "текст": "одобрение крупных сделок"}]'
        assert _tolerant_json(raw, "list") == ["12.1.4. одобрение крупных сделок"]

    def test_duplicated_array_concatenated(self):
        # dedup is _coerce's job, not the parser's
        assert _tolerant_json('["a"]\n["a"]', "list") == ["a", "a"]

    def test_stringified_inner_array_unwrapped(self):
        assert _tolerant_json('["[\\"a\\", \\"b\\"]"]', "list") == ["a", "b"]

    def test_bullet_list_keeps_clause_digits(self):
        raw = "- 7.1. первый пункт\n* 7.2. второй пункт\n"
        assert _tolerant_json(raw, "list") == ["7.1. первый пункт", "7.2. второй пункт"]

    def test_empty_input(self):
        assert _tolerant_json("", "list") == []


class TestTolerantJsonStr:
    def test_quoted_string(self):
        assert _tolerant_json('"7.1. Общее собрание"', "str") == "7.1. Общее собрание"

    def test_object_flattened(self):
        raw = '{"пункт": "13.4", "способ_удостоверения": "нотариус"}'
        assert _tolerant_json(raw, "str") == "13.4. нотариус"

    def test_prose_passthrough(self):
        assert _tolerant_json("не указано", "str") == "не указано"

    def test_empty_input(self):
        assert _tolerant_json("", "str") == ""


class TestCoerce:
    def test_placeholders_dropped_from_list(self):
        val = ["list", "string", "не указано", "7.1. реальный текст", ""]
        assert _coerce(val, "list") == ["7.1. реальный текст"]

    def test_dedup_case_insensitive(self):
        assert _coerce(["Правление", "правление"], "list") == ["Правление"]

    def test_scalar_wrapped_into_list(self):
        assert _coerce("7.1. текст", "list") == ["7.1. текст"]

    def test_none_list(self):
        assert _coerce(None, "list") == []

    def test_str_placeholder_becomes_not_specified(self):
        assert _coerce("string", "str") == "не указано"
        assert _coerce(None, "str") == "не указано"

    def test_str_real_value_kept(self):
        assert _coerce("  7.1. нотариус ", "str") == "7.1. нотариус"
