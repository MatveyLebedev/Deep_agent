"""_FixedToolIdModel message repair (main.py): minimax XML → tool_calls,
empty tool-id patching, binary-block stripping.

main.py imports deepagents/langgraph at module level; if the venv build of
those diverges, skip the module instead of erroring the whole suite.
"""
import pytest

try:
    from main import _FixedToolIdModel
except Exception as e:  # pragma: no cover - env-dependent
    pytest.skip(f"main.py not importable here: {e}", allow_module_level=True)

from langchain_core.messages import AIMessage, ToolMessage


def _parse(content: str) -> AIMessage:
    msg = AIMessage(content=content)
    _FixedToolIdModel._parse_minimax_xml(msg)
    return msg


class TestParseMinimaxXml:
    def test_clean_block_single_invoke(self):
        msg = _parse(
            "Думаю...\n"
            "<minimax:tool_call>\n"
            '<invoke name="read_section">\n'
            '<parameter name="path">/scratch/doc.md</parameter>\n'
            '<parameter name="max_chars">4000</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>\n"
            "готово"
        )
        assert len(msg.tool_calls) == 1
        tc = msg.tool_calls[0]
        assert tc["name"] == "read_section"
        assert tc["args"] == {"path": "/scratch/doc.md", "max_chars": 4000}
        assert tc["id"].startswith("call_")
        # XML stripped from content, surrounding text kept
        assert "<minimax:tool_call>" not in msg.content
        assert "Думаю" in msg.content and "готово" in msg.content

    def test_json_param_parsed_string_param_kept_raw(self):
        msg = _parse(
            "<minimax:tool_call>"
            '<invoke name="search_bm25">'
            '<parameter name="paths">["/scratch/a.md"]</parameter>'
            '<parameter name="query">крупные сделки</parameter>'
            "</invoke>"
            "</minimax:tool_call>"
        )
        args = msg.tool_calls[0]["args"]
        assert args["paths"] == ["/scratch/a.md"]
        assert args["query"] == "крупные сделки"

    def test_two_invokes_in_one_block(self):
        msg = _parse(
            "<minimax:tool_call>"
            '<invoke name="a"><parameter name="x">1</parameter></invoke>'
            '<invoke name="b"><parameter name="y">2</parameter></invoke>'
            "</minimax:tool_call>"
        )
        assert [tc["name"] for tc in msg.tool_calls] == ["a", "b"]

    def test_truncated_by_max_tokens_still_yields_call(self):
        # response cut mid-parameter: no closing tags at all
        msg = _parse(
            "<minimax:tool_call>"
            '<invoke name="read_pdf">'
            '<parameter name="path">/input/Устав'
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "read_pdf"
        assert msg.tool_calls[0]["args"]["path"].startswith("/input/")

    def test_no_block_no_change(self):
        msg = _parse("обычный текст без tool-call")
        assert msg.tool_calls == []
        assert msg.content == "обычный текст без tool-call"

    def test_additional_kwargs_mirrored(self):
        msg = _parse(
            "<minimax:tool_call>"
            '<invoke name="t"><parameter name="k">v</parameter></invoke>'
            "</minimax:tool_call>"
        )
        ak = msg.additional_kwargs["tool_calls"]
        assert len(ak) == 1
        assert ak[0]["function"]["name"] == "t"
        assert ak[0]["id"] == msg.tool_calls[0]["id"]


class TestPatchAndSanitize:
    def test_patch_ai_fills_empty_ids(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "t", "args": {}, "id": "", "type": "tool_call"}],
        )
        _FixedToolIdModel._patch_ai(msg)
        assert msg.tool_calls[0]["id"].startswith("call_")

    def test_sanitize_matches_orphan_tool_message_fifo(self):
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "t", "args": {}, "id": "", "type": "tool_call"}],
        )
        tm = ToolMessage(content="result", tool_call_id="")
        _FixedToolIdModel._sanitize([[ai, tm]])
        assert ai.tool_calls[0]["id"]
        assert tm.tool_call_id == ai.tool_calls[0]["id"]

    def test_as_conversations_normalizes_flat_list(self):
        ai = AIMessage(content="x")
        assert _FixedToolIdModel._as_conversations([ai]) == [[ai]]
        assert _FixedToolIdModel._as_conversations([[ai]]) == [[ai]]


class TestStripBinaryBlocks:
    def test_binary_block_removed_text_kept(self):
        msg = AIMessage(content=[
            {"type": "file", "base64": "AAAA"},
            {"type": "text", "text": "привет"},
        ])
        _FixedToolIdModel._strip_binary_blocks([[msg]])
        assert msg.content == [{"type": "text", "text": "привет"}]

    def test_only_binary_gets_hint(self):
        msg = AIMessage(content=[{"type": "image", "image_url": "data:..."}])
        _FixedToolIdModel._strip_binary_blocks([[msg]])
        assert len(msg.content) == 1
        assert msg.content[0]["type"] == "text"
        assert "read_pdf" in msg.content[0]["text"]

    def test_plain_string_content_untouched(self):
        msg = AIMessage(content="просто текст")
        _FixedToolIdModel._strip_binary_blocks([[msg]])
        assert msg.content == "просто текст"
