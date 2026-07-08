"""Phase-3 refactor guard: generating prompts/subagents from field_specs.py
must be byte-identical to the historical literals (snapshotted before the
merge), and the specs must line up with the default Pydantic schema."""
import json
from pathlib import Path

import pytest

import extraction
import field_specs
from schemas import CharterStructuredOutput

try:
    import main
except Exception as e:  # pragma: no cover - env-dependent
    main = None
    _main_err = e

SNAP = Path(__file__).parent / "snapshots"

needs_main = pytest.mark.skipif(main is None, reason="main.py not importable here")


@needs_main
def test_output_schema_prompt_identical_to_snapshot():
    expected = (SNAP / "output_schema.txt").read_text(encoding="utf-8")
    assert main._OUTPUT_SCHEMA == expected


@needs_main
def test_field_subagents_identical_to_snapshot():
    expected = json.loads((SNAP / "field_subagents.json").read_text(encoding="utf-8"))
    assert main._FIELD_SUBAGENTS == expected


def test_extraction_specs_cover_legacy_specs():
    legacy = json.loads((SNAP / "field_specs_extraction.json").read_text(encoding="utf-8"))
    by_key = {s["key"]: s for s in extraction.FIELD_SPECS}
    assert [s["key"] for s in legacy] == [s["key"] for s in extraction.FIELD_SPECS]
    for old in legacy:
        new = by_key[old["key"]]
        for k, v in old.items():
            assert new[k] == v, f"{old['key']}.{k} diverged from legacy spec"


def test_spec_keys_match_default_schema_fields():
    spec_keys = [s["key"] for s in field_specs.FIELD_SPECS]
    assert len(spec_keys) == len(set(spec_keys)), "duplicate field keys"
    assert set(spec_keys) == set(CharterStructuredOutput.model_fields)


def test_every_spec_has_the_required_keys():
    required = {"key", "kind", "ru", "keywords", "topic",
                "subagent_name", "subagent_description", "agent_topic", "schema_entry"}
    for s in field_specs.FIELD_SPECS:
        missing = required - set(s)
        assert not missing, f"{s['key']} lacks {missing}"
        assert s["kind"] in ("str", "list")
        assert s.get("style", "clause") in ("name", "clause")
