# SDK Usage Guide

Drive the Deep Agent from Python on your host. The SDK wraps `docker compose run` — every ML dependency lives inside the container.

## Requirements

- Docker + `docker compose`
- Python 3.10+ on the host (stdlib only for `SDK.py`)
- Image built once: `docker compose build`

## Setup

### 1. Environment

Copy `.env.example` to `.env` and configure:

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `openai`, `openrouter`, or `custom` |
| `MODEL_NAME` | Model id on your endpoint |
| `OPENROUTER_API_KEY` / `OPENAI_API_KEY` / `CUSTOM_LLM_*` | Per-provider credentials |
| `TRACING_PROVIDER` | `langsmith`, `langfuse`, or `none` |
| `HITL_ENABLED` | `true` to interrupt on `write_file` / `edit_file` (default `false`) |
| `MAX_TOKENS_PER_RUN` / `MAX_SUBAGENTS_PER_RUN` | Per-run budget hints |

### 2. Point the SDK at the project

```python
import os
os.environ["DEEP_AGENT_PROJECT"] = "/absolute/path/to/Deep agent"
```

### 3. Build the image

```bash
cd "$DEEP_AGENT_PROJECT"
docker compose build
```

## Agent architecture

`build_agent()` configures `create_deep_agent` with:

| Component | Source |
|-----------|--------|
| Structured output | `response_format=<your schema>` — populated in `result.structured` |
| Orchestrator tools | `search_examples` only — the main agent delegates everything else |
| Field subagents | One per OUTPUT SCHEMA field (`extract-supreme-body`, `extract-collegial-bodies`, `extract-sole-executive`, `extract-major-transactions`, `extract-related-party-transactions`, `extract-meeting-protocol`, `extract-executive-restrictions`) — each with `read_pdf`, `extract_tables`, `search_bm25`, `search_vector`, `search_examples` |
| `/input/`, `/scratch/` | `LocalShellBackend` on disk under `/workspace/work/current/` |
| `/instructions/`, `/skills/` | `FilesystemBackend` (read-only, per-agent on disk) |
| `/memories/` | `StoreBackend` backed by `InMemoryStore` (cross-thread persistent memory) |
| Default ephemeral paths | `StateBackend` (per-thread, auto-wiped) |
| Planning | Built-in `write_todos` (`TodoListMiddleware`) |
| Checkpointer | `SqliteSaver` at `agents/<name>/checkpoint.db` (required for HITL resume) |
| HITL | `interrupt_on={"write_file": ..., "edit_file": ...}` when `HITL_ENABLED=true` |

## API Reference

Import from `SDK.py` in the project root (or copy `SDK.py` elsewhere and set `DEEP_AGENT_PROJECT`).

### `create_agent(name, ...) -> Agent`

Creates `agents/<name>/` on disk and seeds instructions.

| Argument | Type | Description |
|----------|------|-------------|
| `name` | `str` | Agent directory name |
| `business_rules` | path or str | `agent_init/buisness_rules.md` |
| `process` | path or str | `instructions/process.md` |
| `tool_tips` | path or str | `instructions/tool_tips.md` |
| `tools_file` | path | Copied to `custom_tools.py`; overrides the orchestrator tool set |
| `schema_file` | path | Copied to `custom_schema.py`; one Pydantic `BaseModel` becomes `response_format` |
| `overwrite` | `bool` | Recreate if agent exists |

### `load_agent(name) -> Agent`

Loads an existing agent from `agents/<name>/`.

### `Agent.run(sample) -> RunResult`

Runs analysis on one file or directory.

| Field | Type | Description |
|-------|------|-------------|
| `output` | `str` | Markdown report (work summary + structured JSON + final message) |
| `output_dir` | `Path` | `output/<name>/<timestamp>/` |
| `structured` | `dict \| None` | JSON object matching your schema |
| `interrupt` | `list[dict] \| None` | Pending HITL interrupts (None when the run finished) |
| `thread_id` | `str \| None` | Conversation id; pass to `resume()` |

### `Agent.resume(thread_id, decisions) -> RunResult`

Resume an interrupted HITL run. `decisions` is a `dict` like `{"type": "approve"}` or a list of such dicts; valid types are `approve`, `edit` (`edited_action: {...}`), `reject` (`message: "..."`).

### `Agent.train(samples) -> None`

Reflection-based training. Mutates `instructions/` and `skills/` only when scores improve. Reads tool-call history from the checkpointer message log (`agent.get_state(config).values["messages"]`), so tracing is optional.

### `Agent.test(samples) -> None`

Scores the agent on samples without changing instructions.

## End-to-end example

```python
import os

os.environ["DEEP_AGENT_PROJECT"] = "/path/to/Deep agent"

from SDK import create_agent, load_agent

agent = create_agent(
    name="charter_v1",
    business_rules="agent_init/buisness_rules.md",
    schema_file="custom_schema.py",
    overwrite=True,
)

agent = load_agent("charter_v1")

result = agent.run("input/charter.pdf")

print(result.structured)
print(result.output_dir)
print(result.output[:500])
```

## HITL example

```python
import os
os.environ["DEEP_AGENT_PROJECT"] = "/path/to/Deep agent"

from SDK import load_agent

agent = load_agent("charter_v1")

# Set HITL_ENABLED=true in .env before this call
result = agent.run("input/charter.pdf")

if result.interrupt:
    print("Pending interrupts:", result.interrupt)
    result = agent.resume(result.thread_id, {"type": "approve"})
    # or: agent.resume(result.thread_id, {"type": "reject", "message": "Wrong path"})

print(result.structured)
```

Checkpointer state lives in `agents/<name>/checkpoint.db` and persists across docker invocations, so `resume()` always picks up where the previous `run()` paused.

## Output layout

```
output/<agent_name>/<timestamp>/
├── result.md          # Markdown report
├── structured.json    # Parsed schema fields
├── interrupt.json     # Only present when the run paused for HITL
└── work/              # Snapshot of /input/ and /scratch/
```

Training reports: `output/<agent_name>/training/<timestamp>/training_report.md`.

## CLI (without SDK)

```bash
docker compose run --rm agent create  --name charter_v1 --overwrite
docker compose run --rm agent run     --name charter_v1 --input input/charter.pdf
docker compose run --rm agent resume  --name charter_v1 --thread-id <id> --decisions '{"type":"approve"}'
docker compose run --rm agent train   --name charter_v1 --samples sample1
docker compose run --rm agent test    --name charter_v1 --samples sample1
```

## Corporate LLM example

`.env`:

```
LLM_PROVIDER=custom
MODEL_NAME=gpt-4o
CUSTOM_LLM_BASE_URL=https://llm-gateway.mycorp.local/v1
CUSTOM_LLM_API_KEY=...
TRACING_PROVIDER=langfuse
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://langfuse.mycorp.local
HITL_ENABLED=true
```

## Notes

- Input paths in `agent.run()` are resolved inside the container relative to `/workspace/` (mounted via `docker-compose.yml`).
- `/memories/` lives in an in-process `InMemoryStore`; for cross-container persistence wire a `PostgresStore` in `build_agent()`.
- Rebuild after code changes: `docker compose build`.
