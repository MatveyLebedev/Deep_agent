"""Phase-2 finalization: mapping task-ToolMessages to schema fields."""
import pytest

try:
    from main import Agent
except Exception as e:  # pragma: no cover - env-dependent
    pytest.skip(f"main.py not importable here: {e}", allow_module_level=True)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def _task_call(subagent: str, call_id: str, arg_key: str = "subagent_type") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "task",
            "args": {"description": "extract the field", arg_key: subagent},
            "id": call_id,
            "type": "tool_call",
        }],
    )


class TestSubagentReports:
    def test_maps_report_to_schema_field(self):
        messages = [
            HumanMessage(content="Analyze /input/doc.pdf"),
            _task_call("extract-major-transactions", "call_1"),
            ToolMessage(content="12.1.4. одобрение крупных сделок …", tool_call_id="call_1"),
        ]
        reports = Agent._subagent_reports(messages)
        assert reports == {"major_transaction_clauses": "12.1.4. одобрение крупных сделок …"}

    def test_arg_key_name_does_not_matter(self):
        messages = [
            _task_call("extract-supreme-body", "call_2", arg_key="agent"),
            ToolMessage(content="Общее собрание акционеров", tool_call_id="call_2"),
        ]
        reports = Agent._subagent_reports(messages)
        assert reports == {"supreme_governing_body": "Общее собрание акционеров"}

    def test_non_task_and_unknown_subagents_ignored(self):
        messages = [
            AIMessage(content="", tool_calls=[{
                "name": "read_section", "args": {"path": "/scratch/doc.md"},
                "id": "call_3", "type": "tool_call",
            }]),
            ToolMessage(content="section body", tool_call_id="call_3"),
            _task_call("some-other-agent", "call_4"),
            ToolMessage(content="whatever", tool_call_id="call_4"),
        ]
        assert Agent._subagent_reports(messages) == {}

    def test_repeated_delegation_concatenates(self):
        messages = [
            _task_call("extract-collegial-bodies", "call_5"),
            ToolMessage(content="Наблюдательный совет", tool_call_id="call_5"),
            _task_call("extract-collegial-bodies", "call_6"),
            ToolMessage(content="Правление", tool_call_id="call_6"),
        ]
        reports = Agent._subagent_reports(messages)
        assert reports["collegial_governing_bodies"] == "Наблюдательный совет\n\nПравление"

    def test_empty_history(self):
        assert Agent._subagent_reports([]) == {}
        assert Agent._subagent_reports(None) == {}


class TestMsgText:
    def test_string_content(self):
        assert Agent._msg_text(AIMessage(content=" x ")) == "x"

    def test_block_list_content(self):
        msg = AIMessage(content=[{"type": "text", "text": "a"}, "b"])
        assert Agent._msg_text(msg) == "a b"
