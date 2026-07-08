import json
import os
import re
import uuid
import shutil
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel
from tracing import get_run_config
from langchain.agents import create_agent

from schemas import (
    ScoringResult,
    FieldEvaluation,
    DifferenceItem,
    InstructionEditItem,
    ReflectionResult,
    SkillEditItem,
)


class _FieldJudgeOutput(BaseModel):
    per_field: list[FieldEvaluation]


_SCORE_WEIGHTS = {
    "correct": 1.0,
    "not_significant_error": 0.7,
    "significant_error": 0.0,
}


def _tool_steps_from_messages(messages) -> list[dict]:
    """Pair AIMessage tool_calls with following ToolMessages in order."""
    records: list[dict] = []
    id_to_idx: dict[str, int] = {}
    for msg in messages or []:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls or []:
                name = tc.get("name", "")
                args = tc.get("args") or {}
                records.append({"name": name, "inputs": args, "outputs": {}, "error": None})
                tid = tc.get("id")
                if tid:
                    id_to_idx[tid] = len(records) - 1
        elif isinstance(msg, ToolMessage):
            idx = id_to_idx.get(msg.tool_call_id)
            if idx is not None:
                records[idx]["outputs"] = {"output": msg.content}
            else:
                records.append({
                    "name": "tool",
                    "inputs": {},
                    "outputs": {"output": msg.content},
                    "error": None,
                })
    return records


def fetch_trace_summary_from_agent(agent, config: dict, max_chars: int = 6000) -> str:
    """Compact one-line-per-tool-call log from checkpointer message history."""
    try:
        state = agent.get_state(config)
        messages = (state.values or {}).get("messages", []) if state else []
    except Exception:
        messages = []
    runs = _tool_steps_from_messages(messages)
    lines = []
    for i, r in enumerate(runs):
        args = ", ".join((r.get("inputs") or {}).keys())
        out = str((r.get("outputs") or {}).get("output", ""))[:120].replace("\n", " ")
        lines.append(f"[{i}] {r['name']}({args}) → {out}")
    text = "\n".join(lines)
    return text[:max_chars] + ("\n... (truncated)" if len(text) > max_chars else "")


def _make_stage_detail_tool(agent, config: dict):
    @tool
    def get_stage_detail(step_index: int) -> str:
        """Get full inputs/outputs for a specific tool call from the agent run.
        Use when the trace summary is unclear and you need to inspect what one step actually did."""
        try:
            state = agent.get_state(config)
            messages = (state.values or {}).get("messages", []) if state else []
        except Exception as e:
            return f"Could not read agent state: {e}"
        runs = _tool_steps_from_messages(messages)
        if not (0 <= step_index < len(runs)):
            return f"Invalid index. Valid range: 0-{len(runs) - 1}"
        r = runs[step_index]
        return (
            f"Step [{step_index}] {r['name']}\n"
            f"INPUTS: {r.get('inputs')}\n\n"
            f"OUTPUTS: {r.get('outputs')}\n\n"
            f"ERROR: {r.get('error')}"
        )
    return get_stage_detail

DATA_DIR = Path(os.environ.get("DATA_DIR", "/workspace/agent_init/data"))


@dataclass
class TrainingSample:
    name: str
    input_dir: Path
    input_files: list[str]
    expected_output: str
    comments: str


@dataclass
class TrainingResult:
    sample_name: str
    agent_output: str
    differences: list[DifferenceItem]
    edits_proposed: list[InstructionEditItem]
    edits_accepted: list[InstructionEditItem]
    edits_rolled_back: list[InstructionEditItem]
    skills_proposed: list[SkillEditItem]
    skills_created: list[SkillEditItem]
    skills_rolled_back: list[SkillEditItem]
    score_before: ScoringResult
    score_after: ScoringResult


def load_sample(sample) -> TrainingSample:
    """Accept either a sample name (resolved under DATA_DIR) or a folder path."""
    candidate = Path(sample)
    base = candidate if candidate.exists() else (DATA_DIR / str(sample))
    if not base.exists():
        raise FileNotFoundError(f"Sample not found: {base}")

    input_dir = base / "input"
    input_files = []
    if input_dir.exists():
        input_files = [
            f.name for f in sorted(input_dir.iterdir())
            if f.is_file() and not f.name.startswith(".")
        ]

    expected = ""
    reference_path = base / "output" / "reference.json"
    output_path = base / "output" / "res.txt"
    if reference_path.exists():
        import json as _json
        try:
            expected = _json.dumps(
                _json.loads(reference_path.read_text(encoding="utf-8")),
                ensure_ascii=False, indent=2,
            )
        except Exception:
            pass
    if not expected and output_path.exists():
        expected = output_path.read_text(encoding="utf-8", errors="replace")

    comments = ""
    comments_path = base / "comments" / "comments.md"
    if comments_path.exists():
        comments = comments_path.read_text(encoding="utf-8", errors="replace")

    return TrainingSample(
        name=base.name,
        input_dir=input_dir,
        input_files=input_files,
        expected_output=expected,
        comments=comments,
    )


def _try_parse_json(s: str) -> dict | None:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _aggregate(per_field: list[FieldEvaluation]) -> tuple[float, str]:
    if not per_field:
        return 0.0, "No fields to score."
    overall = sum(_SCORE_WEIGHTS.get(f.status, 0.0) for f in per_field) / len(per_field)
    counts = {k: sum(1 for f in per_field if f.status == k) for k in _SCORE_WEIGHTS}
    reasoning = (
        f"{counts['correct']}/{len(per_field)} correct, "
        f"{counts['not_significant_error']} extras-only, "
        f"{counts['significant_error']} significant"
    )
    return overall, reasoning


def score_output(model, actual: str, expected: str) -> ScoringResult:
    """LLM-judge per-field comparison of structured JSON outputs.

    Categories: correct | not_significant_error (list with extras only) | significant_error.
    Overall is the weighted average over fields.
    """
    expected_obj = _try_parse_json(expected)
    actual_obj = _try_parse_json(actual)
    if expected_obj is None:
        return ScoringResult(
            overall=0.0,
            reasoning="EXPECTED is not valid JSON — check sample/output/reference.json",
            per_field=[],
        )
    if actual_obj is None:
        fields = [
            FieldEvaluation(
                field=k, status="significant_error",
                missing=[json.dumps(v, ensure_ascii=False)] if v else [],
                reasoning="Agent did not return structured JSON output.",
            )
            for k, v in expected_obj.items()
        ]
        overall, summary = _aggregate(fields)
        return ScoringResult(
            overall=overall,
            reasoning="ACTUAL is not valid JSON (no structured_response). " + summary,
            per_field=fields,
        )

    judge = model.with_structured_output(_FieldJudgeOutput)
    prompt = (
        "You compare two JSON objects extracted from a Russian LLC charter.\n"
        "For EACH key in EXPECTED, return one FieldEvaluation with status:\n"
        "  - correct: ACTUAL conveys the same content as EXPECTED — same органы / "
        "same пункты / same протокол. Paraphrasing, clause-number prefixes, "
        "and wording differences are OK as long as it is the same пункт текста.\n"
        "  - not_significant_error: ONLY for list fields where every EXPECTED item "
        "is covered by some ACTUAL item AND ACTUAL has extra items.\n"
        "  - significant_error: at least one EXPECTED item/value is missing or "
        "refers to a different пункт/орган.\n"
        "Fill `missing` with EXPECTED items not covered (verbatim from EXPECTED), "
        "`extras` with ACTUAL list items not matching any EXPECTED item. "
        "Keep `reasoning` ≤ 1 sentence per field.\n\n"
        f"EXPECTED:\n{json.dumps(expected_obj, ensure_ascii=False, indent=2)}\n\n"
        f"ACTUAL:\n{json.dumps(actual_obj, ensure_ascii=False, indent=2)}"
    )
    try:
        verdict: _FieldJudgeOutput = judge.invoke(prompt)
        fields = verdict.per_field
    except Exception as e:
        return ScoringResult(
            overall=0.0,
            reasoning=f"Judge call failed: {e}",
            per_field=[],
        )
    overall, summary = _aggregate(fields)
    return ScoringResult(overall=overall, reasoning=summary, per_field=fields)


def _skill_slug(name: str) -> str:
    s = name.strip().lower().replace(" ", "-")
    s = re.sub(r"[^a-z0-9_-]+", "-", s)
    s = "-".join(x for x in s.split("-") if x).strip("-")
    return s or "skill"


def _skill_frontmatter(slug: str, description: str) -> str:
    desc = " ".join(description.split())
    desc = desc.replace("\\", "\\\\").replace('"', '\\"')
    return f'---\nname: {slug}\ndescription: "{desc}"\n---\n\n'


def list_existing_skills_summary(skills_dir: Path, max_each: int = 280) -> str:
    """Brief listing of skills under ``skills_dir`` for reflection prompts."""
    if not skills_dir.exists():
        return "(none — directory missing)"
    lines: list[str] = []
    for d in sorted(skills_dir.iterdir()):
        if not d.is_dir() or d.name.startswith(".backup_"):
            continue
        sm = d / "SKILL.md"
        if sm.exists():
            preview = sm.read_text(encoding="utf-8", errors="replace")[:max_each]
            lines.append(f"- `{d.name}`:\n```\n{preview}\n```")
        else:
            lines.append(f"- `{d.name}`: (no SKILL.md)")
    return "\n".join(lines) if lines else "(none)"


def apply_skills(items: list[SkillEditItem], skills_dir: Path) -> None:
    """Write ``skills/<slug>/SKILL.md`` for CREATE/MODIFY; DELETE removes the skill folder."""
    skills_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        slug = _skill_slug(it.name)
        path_dir = skills_dir / slug
        if it.operation.upper() == "DELETE":
            if path_dir.exists():
                shutil.rmtree(path_dir)
            continue
        if it.operation.upper() in ("CREATE", "MODIFY"):
            path_dir.mkdir(parents=True, exist_ok=True)
            body = it.content or ""
            fm = _skill_frontmatter(slug, it.description or "")
            (path_dir / "SKILL.md").write_text(fm + body, encoding="utf-8")


def reflect(model, agent_output: str, expected: str, comments: str,
            process_md: str, tool_tips_md: str, agent, config: dict,
            existing_skills_summary: str = "") -> ReflectionResult:
    summary = fetch_trace_summary_from_agent(agent, config)
    detail_tool = _make_stage_detail_tool(agent, config)

    system_prompt = (
        "You review an agent run and propose instruction edits.\n"
        "Use get_stage_detail(step_index) ONLY for steps that look suspicious "
        "in the summary (errors, repeated calls, wrong tools).\n\n"

        "=== RULES FOR EDITING process.md / tool_tips.md ===\n"
        "Rules in these files must work for ANY charter — not just this one sample.\n"
        "FORBIDDEN in proposed edits:\n"
        "  - Hardcoded article/clause numbers (e.g. «12.1.1», «11.1.1.7», «Статья 13.1»)\n"
        "  - Section lists like «found in charter sections X, Y, Z»\n"
        "  - Sample-specific wording presented as a universal rule\n"
        "REQUIRED in proposed edits:\n"
        "  - Describe what to find by TOPIC and KEYWORDS (e.g. крупная сделка, единоличный орган, % от активов)\n"
        "  - Describe HOW to search (scan headings, grep terms, check all governing body sections)\n"
        "  - If an example is helpful, quote text from ACTUAL AGENT OUTPUT and label it «example from this doc:»\n"
        "SELF-CHECK before finalising each proposed edit:\n"
        "  → Does the new text contain any article/section numbers? If yes, replace with keyword description.\n"
        "  → Would this rule work on a charter with completely different numbering? If no, rewrite.\n\n"

        "=== SKILLS (skills/<slug>/SKILL.md) ===\n"
        "Propose proposed_skills when reusable domain HOWTO fits progressive disclosure — "
        "e.g. multi-step extraction for one topic, reusable verification checklist — "
        "NOT when a short process/tool_tips tweak is enough.\n"
        "Same numbering/forbidden rules as edits: no hardcoded charter article IDs as universal rules.\n"
        "Each SkillEditItem: name = slug (ASCII, hyphens); operation CREATE | MODIFY | DELETE;\n"
        "description = one-line YAML description for frontmatter; content = markdown BODY only "
        "(after frontmatter), starting with # Title, include ## When to Use and ## Instructions.\n"
        "MODIFY = replace entire SKILL.md body + description; DELETE removes the skill folder.\n\n"

        "OUTPUT: ReflectionResult — list every difference (MISSING/WRONG/EXTRA/"
        "FORMAT/TOOL_MISUSE), rate severity (critical/major/minor), propose "
        "INSERT/MODIFY/DELETE edits to process.md or tool_tips.md, and propose_skills as needed. "
        "TOOL_MISUSE → tool_tips.md edit. Process flow problems → process.md edit.\n\n"
        f"AGENT TRACE SUMMARY ({len(summary.splitlines())} steps):\n{summary}"
    )
    user_prompt = (
        f"EXPECTED OUTPUT:\n{expected[:6000]}\n\n"
        f"ACTUAL AGENT OUTPUT:\n{agent_output[:6000]}\n\n"
        f"HUMAN COMMENTS:\n{comments[:3000]}\n\n"
        f"CURRENT PROCESS.MD:\n{process_md[:4000]}\n\n"
        f"CURRENT TOOL_TIPS.MD:\n{tool_tips_md[:2000]}\n\n"
        f"EXISTING SKILLS (under /skills/):\n{existing_skills_summary[:3500]}"
    )

    agent = create_agent(
        model=model,
        tools=[detail_tool],
        response_format=ReflectionResult,
        system_prompt=system_prompt,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    structured = result.get("structured_response")
    return structured or ReflectionResult(
        differences=[], proposed_edits=[], proposed_skills=[],
    )


def _skill_leaf_dirs(skills_dir: Path) -> list[Path]:
    if not skills_dir.exists():
        return []
    return sorted(
        d for d in skills_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".backup_")
    )


def _backup_skills_tree(skills_dir: Path, version: int) -> None:
    skills_dir.mkdir(parents=True, exist_ok=True)
    dest = skills_dir / f".backup_v{version}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for d in _skill_leaf_dirs(skills_dir):
        shutil.copytree(d, dest / d.name)


def _rollback_skills_tree(skills_dir: Path, version: int) -> None:
    backup = skills_dir / f".backup_v{version}"
    if not backup.exists():
        return
    for d in _skill_leaf_dirs(skills_dir):
        shutil.rmtree(d)
    for item in backup.iterdir():
        if item.is_dir():
            shutil.copytree(item, skills_dir / item.name)


def backup_instructions(instructions_dir: Path, skills_dir: Path | None = None) -> int:
    version = 1
    while (instructions_dir / f"process.md.v{version}").exists():
        version += 1
    for fname in ("process.md", "tool_tips.md"):
        src = instructions_dir / fname
        if src.exists():
            shutil.copy2(src, instructions_dir / f"{fname}.v{version}")
    if skills_dir is not None:
        _backup_skills_tree(skills_dir, version)
    return version


def apply_edits(edits: list[InstructionEditItem], instructions_dir: Path):
    for edit in edits:
        if edit.target not in ("process.md", "tool_tips.md"):
            continue
        path = instructions_dir / edit.target
        if not path.exists():
            path.write_text("", encoding="utf-8")
        content = path.read_text(encoding="utf-8")

        if edit.operation == "INSERT" and edit.new_text:
            content = content + "\n" + edit.new_text
        elif edit.operation == "MODIFY" and edit.old_text and edit.new_text:
            if edit.old_text in content:
                content = content.replace(edit.old_text, edit.new_text, 1)
        elif edit.operation == "DELETE" and edit.old_text:
            content = content.replace(edit.old_text, "", 1)

        path.write_text(content, encoding="utf-8")


def rollback(version: int, instructions_dir: Path, skills_dir: Path | None = None):
    for fname in ("process.md", "tool_tips.md"):
        backup = instructions_dir / f"{fname}.v{version}"
        if backup.exists():
            shutil.copy2(backup, instructions_dir / fname)
    if skills_dir is not None:
        _rollback_skills_tree(skills_dir, version)


def format_training_report(results: list, timestamp: str = "") -> str:
    """Return a detailed markdown training report from a list of TrainingResult."""
    ts_line = f"  \n_Run: {timestamp}_" if timestamp else ""
    lines = [f"# Training Report{ts_line}\n"]

    for r in results:
        delta = r.score_after.overall - r.score_before.overall
        sign = "+" if delta >= 0 else ""
        if r.edits_accepted or r.skills_created:
            verdict = "✅ ACCEPTED"
        elif r.edits_rolled_back or r.skills_rolled_back:
            verdict = "❌ ROLLED BACK"
        else:
            verdict = "⏭ NO EDITS"

        lines.append(f"## Sample: `{r.sample_name}`\n")
        lines.append(
            f"| | Score |\n|---|---|\n"
            f"| Before | **{r.score_before.overall:.2f}** |\n"
            f"| After  | **{r.score_after.overall:.2f}** |\n"
            f"| Delta  | **{sign}{delta:.2f}** |\n"
            f"| Result | {verdict} |\n"
        )
        lines.append(f"\n**Scoring note:** {r.score_before.reasoning[:300]}\n")

        if r.score_before.per_field:
            icon = {
                "correct": "🟢",
                "not_significant_error": "🟡",
                "significant_error": "🔴",
            }
            lines.append("\n### Per-field verdict (before)\n")
            lines.append("| Field | Status | Missing | Extras |")
            lines.append("|---|---|---|---|")
            for f in r.score_before.per_field:
                miss = "; ".join(m[:80] for m in f.missing) or "—"
                extra = "; ".join(x[:80] for x in f.extras) or "—"
                lines.append(
                    f"| `{f.field}` | {icon.get(f.status, '⚪')} {f.status} | {miss} | {extra} |"
                )
            lines.append("")

        if r.differences:
            lines.append(f"\n### Differences found ({len(r.differences)})\n")
            for d in r.differences:
                icon = {"critical": "🔴", "major": "🟠", "minor": "🟡"}.get(d.severity, "⚪")
                lines.append(f"- {icon} **{d.type}** — {d.description}")
                if d.expected_fragment:
                    lines.append(f"  - _Expected:_ `{d.expected_fragment[:120]}`")
                if d.actual_fragment:
                    lines.append(f"  - _Actual:_   `{d.actual_fragment[:120]}`")
            lines.append("")

        if r.edits_accepted:
            lines.append(f"\n### ✅ Edits that WORKED ({len(r.edits_accepted)})\n")
            for e in r.edits_accepted:
                lines.append(f"**[{e.operation}] `{e.target}`** — {e.reason}")
                if e.new_text:
                    lines.append(f"```\n{e.new_text[:400]}\n```")
            lines.append("")

        if r.edits_rolled_back:
            lines.append(f"\n### ❌ Edits that DID NOT WORK ({len(r.edits_rolled_back)})\n")
            for e in r.edits_rolled_back:
                lines.append(f"**[{e.operation}] `{e.target}`** — {e.reason}")
                if e.new_text:
                    lines.append(f"```\n{e.new_text[:400]}\n```")
            lines.append("")

        if r.skills_proposed:
            lines.append(f"\n### Skills proposed ({len(r.skills_proposed)})\n")
            for s in r.skills_proposed:
                lines.append(
                    f"**[{s.operation}] `{s.name}`** — {s.reason}  \n"
                    f"_description:_ {s.description[:200]}\n",
                )
                if s.content:
                    lines.append(f"```\n{s.content[:400]}\n```")
            lines.append("")

        if r.skills_created:
            lines.append(f"\n### ✅ Skills accepted ({len(r.skills_created)})\n")
            for s in r.skills_created:
                lines.append(f"**[{s.operation}] `{s.name}`** — {s.reason}\n")
            lines.append("")

        if r.skills_rolled_back:
            lines.append(f"\n### ❌ Skills rolled back ({len(r.skills_rolled_back)})\n")
            for s in r.skills_rolled_back:
                lines.append(f"**[{s.operation}] `{s.name}`** — {s.reason}\n")
            lines.append("")

        if not r.edits_proposed and not r.skills_proposed:
            lines.append("_No instruction or skill changes were proposed for this sample._\n")

        lines.append("---\n")

    return "\n".join(lines)


def _format_agent_output(result: dict) -> str:
    """Prefer structured_response JSON; fall back to final message text."""
    structured = result.get("structured_response")
    if structured is not None:
        if hasattr(structured, "model_dump"):
            return json.dumps(structured.model_dump(), ensure_ascii=False, indent=2)
        return json.dumps(structured, ensure_ascii=False, indent=2)
    messages = result.get("messages") or []
    if messages:
        return str(messages[-1].content)
    return "No output"


def run_agent_on_input(agent, sample: TrainingSample) -> tuple[str, dict]:
    """Stage the sample input and invoke the agent. Returns (output, langgraph config)."""
    from main import prepare_run_workspace
    prepare_run_workspace(sample.input_dir)

    if not sample.input_files:
        return "Error: no input files", {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    config = get_run_config(
        {"configurable": {"thread_id": f"train-{timestamp}"}},
        run_id=run_id,
    )
    virtual_path = f"/input/{sample.input_files[0]}"

    user_message = (
        f"Analyze the document at: {virtual_path}\n"
        f"Follow /instructions/process.md, /instructions/tool_tips.md, and any relevant /skills/*/SKILL.md. "
        f"Extract clause IDs and legal text only from **this** input file — not from examples. "
        f"Use search_examples for format hints. "
        f"Delegate each OUTPUT SCHEMA field to the matching named subagent via task(). "
        f"Verify every subagent result before proceeding."
    )

    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_message}]
        }, config=config)
    except ValueError as e:
        msg = str(e)
        if "tool" in msg.lower() and "id" in msg.lower():
            raise RuntimeError(
                f"Model tool-call error (likely Minimax bug with tool IDs). "
                f"Set MODEL_NAME in .env to a model that supports tools reliably "
                f"(e.g. openai/gpt-4o or anthropic/claude-3.5-sonnet).\n"
                f"Original error: {e}"
            ) from e
        raise

    return _format_agent_output(result), config


def train_on_sample(
    agent,
    model,
    sample: TrainingSample,
    instructions_dir: Path,
    skills_dir: Path,
) -> TrainingResult:
    os.environ["ACTIVE_SAMPLE"] = sample.name
    try:
        print("Phase 1: Running agent on input...")
        agent_output, run_config = run_agent_on_input(agent, sample)

        print("Phase 2: Scoring output vs expected...")
        score_before = score_output(model, agent_output, sample.expected_output)
        print(f"  Score before: {score_before.overall:.2f} ({score_before.reasoning[:100]})")

        print("Phase 3: Reflecting on differences (with trace)...")

        def _read_instruction(name: str) -> str:
            p = instructions_dir / name
            return p.read_text(encoding="utf-8") if p.exists() else ""

        process_md = _read_instruction("process.md")
        tool_tips_md = _read_instruction("tool_tips.md")
        skills_dir.mkdir(parents=True, exist_ok=True)
        existing_skills = list_existing_skills_summary(skills_dir)

        reflection = reflect(
            model,
            agent_output,
            sample.expected_output,
            sample.comments,
            process_md,
            tool_tips_md,
            agent,
            run_config,
            existing_skills_summary=existing_skills,
        )
        print(f"  Differences found: {len(reflection.differences)}")
        print(f"  Edits proposed: {len(reflection.proposed_edits)}")
        print(f"  Skills proposed: {len(reflection.proposed_skills)}")

        if not reflection.proposed_edits and not reflection.proposed_skills:
            print("  No instruction or skill changes proposed. Skipping edit/verify phases.")
            return TrainingResult(
                sample_name=sample.name,
                agent_output=agent_output,
                differences=reflection.differences,
                edits_proposed=[],
                edits_accepted=[],
                edits_rolled_back=[],
                skills_proposed=[],
                skills_created=[],
                skills_rolled_back=[],
                score_before=score_before,
                score_after=score_before,
            )

        print("Phase 4: Applying edits (with backup)...")
        version = backup_instructions(instructions_dir, skills_dir)
        print(f"  Backup created: v{version}")
        apply_edits(reflection.proposed_edits, instructions_dir)
        apply_skills(reflection.proposed_skills, skills_dir)
        print(
            f"  {len(reflection.proposed_edits)} instruction edits, "
            f"{len(reflection.proposed_skills)} skill change(s) applied",
        )

        print("Phase 5: Verifying -- re-running agent with edited instructions/skills...")
        new_output, _run_config = run_agent_on_input(agent, sample)
        score_after = score_output(model, new_output, sample.expected_output)
        print(f"  Score after: {score_after.overall:.2f}")

        if score_after.overall >= score_before.overall:
            print("  Score improved or equal -> ACCEPTING changes")
            edits_accepted = reflection.proposed_edits
            edits_rolled_back = []
            skills_created = reflection.proposed_skills
            skills_rolled_back = []
        else:
            print("  Score decreased -> ROLLING BACK changes")
            rollback(version, instructions_dir, skills_dir)
            edits_accepted = []
            edits_rolled_back = reflection.proposed_edits
            skills_created = []
            skills_rolled_back = reflection.proposed_skills

        return TrainingResult(
            sample_name=sample.name,
            agent_output=agent_output,
            differences=reflection.differences,
            edits_proposed=reflection.proposed_edits,
            edits_accepted=edits_accepted,
            edits_rolled_back=edits_rolled_back,
            skills_proposed=reflection.proposed_skills,
            skills_created=skills_created,
            skills_rolled_back=skills_rolled_back,
            score_before=score_before,
            score_after=score_after,
        )
    finally:
        os.environ.pop("ACTIVE_SAMPLE", None)
