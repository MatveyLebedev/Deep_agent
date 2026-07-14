import json
import re


import pytest

from raglib.agent import MockLLM
from raglib.agent.llm import content_to_text
from raglib.agent.prompts import parse_json, str_list

# parses the candidate listing rendered by AgenticSearcher._reflect
CAND_RE = re.compile(
    r"C(\d+) \(пункт (.+?), документ (.+?)\):\n(.*?)(?=\n\nC\d+ \(пункт|\Z)", re.S)


def reflect_by(pred):
    """Scripted REFLECT: verdict = pred(clause_number, snippet) per candidate."""
    def responder(messages):
        items = []
        for m in CAND_RE.finditer(messages[-1]["content"]):
            verdict = pred(m.group(2).strip(), m.group(4))
            items.append({
                "id": f"C{m.group(1)}",
                "verdict": verdict,
                "aspects": [1] if verdict != "irrelevant" else [],
                "missing": "" if verdict != "irrelevant" else "нужен порог в процентах",
            })
        return json.dumps(items, ensure_ascii=False)
    return responder


def test_parse_json_tolerant():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    # reasoning models wrap chatter in <think> before the JSON
    assert parse_json('<think>прикину {варианты}…</think>\n{"a": 1}') == {"a": 1}
    assert parse_json('Вот ответ: [{"id": "C1"}] — готово') == [{"id": "C1"}]
    assert parse_json('{"s": "скобка ] в строке"}') == {"s": "скобка ] в строке"}
    assert parse_json("это вообще не JSON") is None
    assert parse_json("") is None
    assert str_list(["a", "", "b", 3], 2) == ["a", "b"]
    assert str_list("одна строка", 3) == ["одна строка"]
    assert str_list(None, 3) == []


def test_happy_path_one_iteration(bm25_index):
    plan = json.dumps({"queries": ["крупной сделки", "балансовой стоимости активов"],
                       "aspects": ["пороги крупных сделок"]}, ensure_ascii=False)
    llm = MockLLM([plan, reflect_by(lambda num, snip: "relevant")])
    res = bm25_index.agentic_search("Какие пороги одобрения крупных сделок?",
                                    llm=llm, top_k=8)
    assert not res.degraded
    assert res.iterations == 1 and res.llm_calls == 2
    assert res.hits and all(h.verdict == "relevant" for h in res.hits)
    assert all(h.method == "agentic" for h in res.hits)
    assert {"13.1", "13.2"} <= {h.clause_number for h in res.hits}


def test_refine_recovers_missed_clause(bm25_index):
    """Iteration 1 searches the wrong thing; reflection rejects it and reports
    what is missing; REFINE reformulates; iteration 2 finds the real clauses."""
    plan = json.dumps({"queries": ["фирменное наименование"],
                       "aspects": ["порог 25 процентов для крупной сделки"]},
                      ensure_ascii=False)
    content_aware = reflect_by(
        lambda num, snip: "relevant" if "25" in snip else "irrelevant")
    refine = json.dumps({"queries": ["25 процентов балансовой стоимости"]},
                        ensure_ascii=False)
    llm = MockLLM([plan, content_aware, refine, content_aware])

    res = bm25_index.agentic_search("Каков порог крупной сделки?", llm=llm, top_k=8)
    assert not res.degraded
    assert res.iterations == 2 and res.llm_calls == 4
    numbers = {h.clause_number for h in res.hits}
    assert {"13.1", "13.2"} <= numbers
    assert all(h.verdict in ("relevant", "partial") for h in res.hits)
    # grade cache: the clause rejected in iteration 1 is not re-graded in iteration 2
    second_reflect = llm.calls[3][-1]["content"]
    assert "фирменное" not in second_reflect.lower()


def test_unparseable_llm_degrades_to_plain_search(bm25_index):
    prompt = "нотариальное удостоверение протокола"
    llm = MockLLM(["это вообще не JSON, извините"])
    res = bm25_index.agentic_search(prompt, llm=llm, top_k=8)
    assert res.degraded and res.llm_calls == 1
    plain = bm25_index.search(prompt, method="bm25", top_k=8)
    assert [h.clause_id for h in res.hits] == [h.clause_id for h in plain]
    assert all(h.method == "agentic" for h in res.hits)


def test_llm_exception_degrades_gracefully(bm25_index):
    class BoomLLM:
        def invoke(self, messages):
            raise TimeoutError("corporate gateway timeout")

    res = bm25_index.agentic_search("пороги сделок", llm=BoomLLM(), top_k=5)
    assert res.degraded
    assert any("failed" in str(step.get("error", "")) for step in res.trace)


def test_llm_none_equals_plain_search(vec_index):
    prompt = "крупные сделки"
    res = vec_index.agentic_search(prompt, llm=None, top_k=5)
    assert not res.degraded and res.llm_calls == 0
    plain = vec_index.search(prompt, method="hybrid", top_k=5)
    assert [h.clause_id for h in res.hits] == [h.clause_id for h in plain]


def test_llm_call_budget_is_enforced(bm25_index):
    plan = json.dumps({"queries": ["балансовой стоимости"], "aspects": ["пороги"]},
                      ensure_ascii=False)
    llm = MockLLM([plan])  # only the plan fits into the budget
    res = bm25_index.agentic_search("пороги сделок", llm=llm,
                                    top_k=5, max_llm_calls=1)
    assert res.degraded and res.llm_calls == 1
    assert any("budget" in str(step) for step in res.trace)


def test_content_to_text_flattens_blocks():
    assert content_to_text("abc") == "abc"
    # some LangChain models return a list of content blocks instead of a string
    assert content_to_text(
        [{"type": "text", "text": "abc"}, {"text": "de"}]) == "abcde"
    assert content_to_text(None) == ""


@pytest.mark.parametrize("as_blocks", [False, True])
def test_langchain_model_drives_agentic_search(bm25_index, as_blocks):
    """A LangChain chat model is used directly (no adapter): raglib calls
    .invoke() with OpenAI-style dicts and flattens the AIMessage content —
    whether it comes back as a string or as content blocks."""
    plan = json.dumps({"queries": ["крупной сделки", "балансовой стоимости"],
                       "aspects": ["пороги крупных сделок"]}, ensure_ascii=False)
    llm = MockLLM([plan, reflect_by(lambda num, snip: "relevant")],
                  as_blocks=as_blocks)

    res = bm25_index.agentic_search("Какие пороги одобрения крупных сделок?",
                                    llm=llm, top_k=6)
    assert not res.degraded and res.llm_calls == 2
    assert {"13.1", "13.2"} <= {h.clause_number for h in res.hits}
    assert all(h.method == "agentic" for h in res.hits)
    # raglib passes OpenAI-style dict messages straight into .invoke()
    assert llm.calls[0][0]["role"] == "system"


def test_agentic_hits_keep_output_invariant(bm25_index):
    plan = json.dumps({"queries": ["утверждение независимого аудитора"],
                       "aspects": ["аудитор"]}, ensure_ascii=False)
    llm = MockLLM([plan, reflect_by(lambda num, snip: "relevant")])
    res = bm25_index.agentic_search("Кто утверждает аудитора?", llm=llm, top_k=5)
    hit = next(h for h in res.hits if h.clause_number == "12.1.4")
    # snippet truncation happens ONLY inside the reflection prompt:
    assert len(hit.text) > 1500
    assert "н) утверждение независимого аудитора" in hit.text
