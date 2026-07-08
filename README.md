# Deep Agent — LLC Charter Extraction

Deep Agent extracts a fixed set of **corporate-governance facts from Russian LLC
charters** (уставы ООО) — governing bodies, transaction-approval clauses, and
executive-power restrictions — and returns them as structured JSON.

It is built to run in a **closed corporate network** against a **weak,
quirky-tool-calling LLM** (e.g. `minimax-m3`) served through an OpenAI-compatible
gateway, with embeddings from **GigaChat** (Sber). Everything runs offline inside
one Docker image; no data has to leave the perimeter.

> **Language note:** the target documents, prompts, and extracted values are in
> Russian. The code, configuration, and this guide are in English.

---

## What it extracts

Seven fields (single source of truth: [`field_specs.py`](field_specs.py),
schema in [`schemas.py`](schemas.py)):

| Field | Meaning |
|---|---|
| `supreme_governing_body` | Высший орган управления (обычно Общее собрание участников) |
| `collegial_governing_bodies` | Коллегиальные органы (Совет директоров, Наблюдательный совет, Правление) |
| `sole_executive_bodies` | Единоличные исполнительные органы (Генеральный директор, …) |
| `major_transaction_clauses` | Пункты устава о крупных сделках (пороги, кто одобряет) |
| `related_party_transaction_clauses` | Пункты о сделках с заинтересованностью |
| `general_meeting_minutes_protocol` | Способ удостоверения протокола общего собрания |
| `sole_executive_body_restrictions` | Уставные ограничения полномочий ЕИО |

Clause fields return `"<номер пункта>. <дословный текст>"`; name fields return a
canonical organ name.

---

## How it works

There are **two run modes**, selected by the `EXTRACTION_MODE` environment variable.

### `deterministic` — recommended (robust on weak models)

Orchestration happens in Python, not in the model. For each of the seven fields
the pipeline ([`extraction.py`](extraction.py)):

1. Converts the PDF to Markdown once (Docling, via [`tools.py`](tools.py) `read_pdf`).
2. Splits it into hierarchical chunks ([`retrieval.py`](retrieval.py)).
3. Retrieves the most relevant chunks for that field (BM25 / hybrid BM25+vector RRF).
4. Makes **one small, single-purpose LLM call** returning just that field as JSON.
5. Parses tolerantly (prose, code fences, duplicated blocks, truncation) and
   drops schema-echo placeholders.

Field calls are independent, so they run concurrently. A final
[verification pass](extraction.py) flags any clause number that does not actually
appear in the source document (the classic weak-model failure of copying clause
IDs from few-shot examples). This mode does not depend on the model reliably
emitting tool calls or following a multi-step plan, so results are stable.

### `agent` — legacy multi-agent orchestration

An LLM orchestrator ([`main.py`](main.py), built on
[`deepagents`](https://pypi.org/project/deepagents/)) delegates each field to a
dedicated subagent, verifies each result, then finalizes structured output. It is
more flexible but fragile on weak tool-calling models — it exists for capable
models and for comparison. Internals are documented in
[DOCUMENTATION.md](DOCUMENTATION.md) (which describes this mode specifically).

### Module map

| Module | Responsibility |
|---|---|
| [`extraction.py`](extraction.py) | Deterministic per-field extraction + verification |
| [`retrieval.py`](retrieval.py) | Hierarchical chunking, hybrid BM25/vector retrieval |
| [`field_specs.py`](field_specs.py) | The 7 fields (keywords, prompts, styles) — single source of truth |
| [`schemas.py`](schemas.py) | Pydantic output models |
| [`tools.py`](tools.py) | `read_pdf` (Docling) and search tools |
| [`providers.py`](providers.py) | Model shims: MiniMax XML tool-call parsing, per-run token budget, binary-block stripping |
| [`netguard.py`](netguard.py) | Opt-in strict egress allowlist for closed networks |
| [`gigachat_embeddings.py`](gigachat_embeddings.py) | GigaChat OAuth2 + embeddings client |
| [`tracing.py`](tracing.py) | Optional Langfuse / LangSmith tracing |
| [`main.py`](main.py) | CLI, agent-mode orchestration, run/train/test entry points |
| [`SDK.py`](SDK.py) | Host- and in-container Python API ([SDK_USAGE.md](SDK_USAGE.md)) |
| [`training.py`](training.py) | Legacy self-improvement loop (agent mode) |
| [`raglib/`](raglib/) | Standalone RAG library used for retrieval experiments ([raglib/README.md](raglib/README.md)) |

---

## Requirements

- Docker + `docker compose`
- Access to an OpenAI-compatible LLM endpoint (corporate gateway, OpenRouter, or
  OpenAI) and an embeddings endpoint (GigaChat or any OpenAI-compatible one)

All ML dependencies live inside the image. Python on the host is only needed if
you want to call the SDK as a host-side library.

## Setup

```bash
cp .env.example .env      # then fill in endpoints + keys (see below)
docker compose build      # bakes Docling models into the image for offline use
```

Minimum configuration in `.env`:

```dotenv
LLM_PROVIDER=custom
CUSTOM_LLM_BASE_URL=https://your-gateway.example/v1
CUSTOM_LLM_API_KEY=...
MODEL_NAME=minimax-m3

EMBED_PROVIDER=gigachat
GIGACHAT_CREDENTIALS=...            # base64(client_id:client_secret)

EXTRACTION_MODE=deterministic       # recommended
```

See [`.env.example`](.env.example) for every option (rate limits, retrieval
tuning, budgets, tracing, the network guard). **Never commit your `.env`** — it is
gitignored.

---

## Usage

### CLI (inside the container)

```bash
# Extract from a single document (place the PDF under ./input/ first)
docker compose run --rm agent run \
    --name charter \
    --input /workspace/input/charter.pdf
```

Outputs land in `output/<name>/<timestamp>/`:

```
result.md          # human-readable report
structured.json    # the seven-field JSON
work/              # extracted markdown, retrieval scratch
```

`create`, `train`, and `test` subcommands also exist and belong to the legacy
agent mode; run `docker compose run --rm agent --help`.

### SDK (from the host or inside the container)

```python
from SDK import load_agent

agent = load_agent("charter")            # or create_agent("charter", ...)
result = agent.run("input/charter.pdf")

print(result.structured)                 # dict of the 7 fields
print(result.output)                     # full result.md
```

The same `SDK.py` shells out to `docker compose` from the host and runs
in-process inside the container — detection is automatic. Details:
[SDK_USAGE.md](SDK_USAGE.md).

---

## Closed-network / offline deployment

- **`NETWORK_GUARD=strict`** ([`netguard.py`](netguard.py)) wraps DNS resolution
  and refuses any outbound connection except the endpoints you explicitly
  configured (LLM gateway, GigaChat) plus loopback — a second, in-process line of
  defense so a stray default to a public cloud can never leak keys or document
  text.
- **Docling models are pre-baked** into the image at build time, so the first run
  never tries to download them.
- **GigaChat** embeddings ([`gigachat_embeddings.py`](gigachat_embeddings.py))
  handle the OAuth2 flow and support a corporate CA bundle.
- The [`deploy/`](deploy/) folder contains a self-contained bundle for moving the
  service into an air-gapped environment (see `deploy/README_DEPLOY.md`).

---

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest                 # top-level suite; no network required
```

The tests use synthetic fixtures and stub PDFs — **no real documents are needed
or included** in this repository. `raglib/` has its own fully offline suite
(`cd raglib && python -m pytest`).

---

## Project layout

```
.
├── extraction.py          # deterministic pipeline (recommended path)
├── retrieval.py           # chunking + hybrid retrieval
├── field_specs.py         # the 7 fields — single source of truth
├── schemas.py             # pydantic models
├── tools.py               # read_pdf (Docling) + search tools
├── providers.py           # weak-model shims + token budget
├── netguard.py            # strict egress guard
├── gigachat_embeddings.py # GigaChat embeddings
├── main.py                # CLI + legacy agent orchestration
├── SDK.py                 # host/in-container Python API
├── tracing.py, training.py
├── agent_init/            # seed business rules (data/ is gitignored)
├── instructions/          # default agent instructions (agent mode)
├── tests/                 # offline test suite
├── raglib/                # standalone RAG library (own README + tests)
├── deploy/                # air-gapped deployment bundle
├── Dockerfile, docker-compose.yml, requirements*.txt
├── DOCUMENTATION.md       # deep internals (agent mode)
└── .env.example
```

Runtime folders (`input/`, `agents/`, `output/`, `work/`, `models/`) are created
as needed and are gitignored so that **documents and run artifacts are never
committed**.

---

## License

Released under the [MIT License](LICENSE).
