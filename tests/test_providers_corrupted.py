"""Phase-7: corrupted minimax dialect — the ]<]minimax[>[ delimiter with
direct <param>value</param> tags, as captured in the LangSmith trace of a
no-tools OpenRouter provider. The shim must translate it (and its truncated
variants) into real tool_calls, so agent mode survives an unpinnable stack
(e.g. self-hosted vLLM without the minimax tool-call parser)."""
from langchain_core.messages import AIMessage

from providers import FixedToolIdModel


def _parse(content: str) -> AIMessage:
    msg = AIMessage(content=content)
    FixedToolIdModel._parse_minimax_xml(msg)
    return msg


class TestCorruptedDialect:
    def test_broken_delimiter_and_direct_tags(self):
        msg = _parse(
            ']<]minimax[>[<invoke name="read_file">'
            "<file_path>/instructions/process.md</file_path>"
            "</invoke>]<]/minimax[>["
        )
        assert len(msg.tool_calls) == 1
        tc = msg.tool_calls[0]
        assert tc["name"] == "read_file"
        assert tc["args"] == {"file_path": "/instructions/process.md"}
        assert "]<]minimax[>[" not in msg.content

    def test_truncated_by_max_tokens(self):
        msg = _parse(']<]minimax[>[<invoke name="ls"><path>/skills</path')
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "ls"
        assert msg.tool_calls[0]["args"]["path"] == "/skills"

    def test_json_value_in_direct_tag(self):
        msg = _parse(
            ']<]minimax[>[<invoke name="search_hybrid">'
            '<paths>["/scratch/doc.md"]</paths><query>крупные сделки</query>'
            "</invoke>]<]/minimax[>["
        )
        args = msg.tool_calls[0]["args"]
        assert args["paths"] == ["/scratch/doc.md"]
        assert args["query"] == "крупные сделки"

    def test_mm_think_markers_stripped_with_block(self):
        msg = _parse(
            "<mm:think>рассуждение остаётся</mm:think>"
            ']<]minimax[>[<invoke name="ls"><path>/</path></invoke>]<]/minimax[>['
        )
        assert len(msg.tool_calls) == 1
        assert "<mm:think>" not in msg.content and "</mm:think>" not in msg.content
        assert "рассуждение остаётся" in msg.content

    def test_mixed_clean_and_corrupted_blocks(self):
        msg = _parse(
            "<minimax:tool_call>"
            '<invoke name="a"><parameter name="x">1</parameter></invoke>'
            "</minimax:tool_call>\n"
            ']<]minimax[>[<invoke name="b"><y>2</y></invoke>]<]/minimax[>['
        )
        assert [tc["name"] for tc in msg.tool_calls] == ["a", "b"]
        assert msg.tool_calls[0]["args"] == {"x": 1}
        assert msg.tool_calls[1]["args"] == {"y": 2}

    def test_plain_text_untouched(self):
        msg = _parse("обычный ответ без XML")
        assert msg.tool_calls == []
        assert msg.content == "обычный ответ без XML"
