import hashlib
import os
import re
import shutil
from pathlib import Path
from langchain.tools import tool

DATA_DIR = Path(os.environ.get("DATA_DIR", "/workspace/agent_init/data"))
DEFAULT_WORK_ROOT = "/workspace/work/current"

# One converter per table-structure setting. read_pdf only exports markdown, so
# it uses a converter with table-structure detection OFF (skips the slow
# TableFormer model load + inference); extract_tables uses one with it ON.
_converters: dict[bool, object] = {}
# In-process cache of docling ConversionResult keyed by (path, mtime, with_tables)
# so repeated calls within one run never re-convert the same PDF.
_conversion_cache: dict[tuple, object] = {}
_CONVERSION_CACHE_MAX = 16


def _work_root() -> Path:
    return Path(os.environ.get("WORK_ROOT", DEFAULT_WORK_ROOT))


def _cache_root() -> Path:
    """Persistent, content-addressed cache for document-derived artifacts
    (markdown, embeddings). It lives OUTSIDE the per-run /scratch (which is wiped
    each run), so the heavy docling conversion + embedding of a given file are
    computed ONCE and reused across runs. Override with CACHE_ROOT."""
    return Path(os.environ.get("CACHE_ROOT", str(_work_root().parent / "cache")))


def _file_sha(real: Path) -> str:
    """Stable content hash of a file (first 16 hex of sha256) for cache keys."""
    h = hashlib.sha256()
    with open(real, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _resolve_virtual(path: str) -> Path:
    """Resolve a virtual /input/* or /scratch/* path to a real OS path under WORK_ROOT.
    Rejects everything else (including /memories/* and /instructions/* which are
    handled by the agent's built-in FS tools, and any host paths)."""
    if path.startswith("/input/"):
        return (_work_root() / "input" / path[len("/input/"):]).resolve()
    if path.startswith("/scratch/"):
        return (_work_root() / "scratch" / path[len("/scratch/"):]).resolve()
    raise ValueError(
        f"Path must start with /input/ or /scratch/. Got: {path!r}"
    )


def _get_converter(with_tables: bool):
    conv = _converters.get(with_tables)
    if conv is None:
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

        # Use explicit artifacts_path only when the directory is non-empty.
        # Otherwise pass None so Docling auto-downloads to ~/.cache/docling.
        from pathlib import Path as _Path
        _ap_str = os.environ.get("DOCLING_ARTIFACTS_PATH", "/workspace/models/docling")
        _ap = _Path(_ap_str)
        _artifacts_path = _ap if (_ap.is_dir() and any(_ap.iterdir())) else None

        pipeline_options = PdfPipelineOptions(
            do_table_structure=with_tables,
            do_ocr=False,
            artifacts_path=_artifacts_path,
        )
        conv = DocumentConverter(
            format_options={
                InputFormat.PDF: FormatOption(
                    pipeline_options=pipeline_options,
                    pipeline_cls=StandardPdfPipeline,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )
        _converters[with_tables] = conv
    return conv


def _convert(real: Path, with_tables: bool):
    """Convert a PDF once and cache the ConversionResult for the process.
    Repeated read_pdf/extract_tables calls on the same file reuse this."""
    try:
        mtime = real.stat().st_mtime
    except OSError:
        mtime = 0.0
    key = (str(real), mtime, with_tables)
    result = _conversion_cache.get(key)
    if result is None:
        result = _get_converter(with_tables).convert(str(real))
        if len(_conversion_cache) >= _CONVERSION_CACHE_MAX:
            _conversion_cache.pop(next(iter(_conversion_cache)))
        _conversion_cache[key] = result
    return result


@tool
def read_pdf(path: str, pages: str = "") -> str:
    """Parse a PDF and save extracted text as a markdown file in /scratch/.
    Returns the saved path (e.g. 'Saved to /scratch/file.md (Total pages: N)').
    After calling this tool, use read_file('/scratch/file.md') to read the text.
    Do NOT call read_pdf again on the same file — reuse the saved scratch file.
    Args:
        path: virtual path under /input/ or /scratch/ (e.g. '/input/report.pdf').
        pages: optional page selector — single page "5", range "3-7",
               comma-separated "1,3,5", or empty for all pages.
               Use pages="count" to only return total page count.
    """
    try:
        real = _resolve_virtual(path)
    except ValueError as e:
        return f"Error: {e}"
    if not real.exists():
        return f"Error: file not found: {path}"

    pages_spec = pages.strip()
    is_count = pages_spec.lower() == "count"

    stem = real.stem
    suffix = f"_p{pages_spec}" if (pages_spec and not is_count) else ""
    out_name = f"{stem}{suffix}.md"
    scratch_real = _work_root() / "scratch"
    out_path = scratch_real / out_name

    # In-run cache: same PDF already converted in this run's /scratch — reuse.
    if out_path.exists():
        first_line = out_path.read_text(encoding="utf-8").split("\n", 1)[0]
        if is_count:
            return first_line if first_line.startswith("Total pages:") else "Total pages: (unknown)"
        return f"Saved to /scratch/{out_name} (cached; {first_line})"

    # Persistent cache (full-document conversions only): survives the per-run
    # /scratch wipe, so docling runs once per unique file across all runs.
    persist_md = (_cache_root() / _file_sha(real) / "text.md") if not pages_spec else None
    if persist_md is not None and persist_md.exists():
        scratch_real.mkdir(parents=True, exist_ok=True)
        shutil.copy2(persist_md, out_path)
        first_line = persist_md.read_text(encoding="utf-8").split("\n", 1)[0]
        if is_count:
            return first_line if first_line.startswith("Total pages:") else "Total pages: (unknown)"
        return f"Saved to /scratch/{out_name} (cached; {first_line})"

    result = _convert(real, with_tables=False)
    doc = result.document
    total_pages = len(doc.pages)

    if is_count:
        return f"Total pages: {total_pages}"

    selected = _parse_pages(pages, total_pages) if pages_spec else list(range(1, total_pages + 1))

    parts = []
    for pg in selected:
        md = doc.export_to_markdown(page_no=pg)
        if md.strip():
            parts.append(f"--- Page {pg}/{total_pages} ---\n{md}")
    if not parts:
        return f"No content found on selected pages. Total pages: {total_pages}"
    output = f"Total pages: {total_pages}\n\n" + "\n\n".join(parts)

    scratch_real.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    if persist_md is not None:  # populate the cross-run cache
        persist_md.parent.mkdir(parents=True, exist_ok=True)
        persist_md.write_text(output, encoding="utf-8")
    return f"Saved to /scratch/{out_name} (Total pages: {total_pages})"


def _parse_pages(spec: str, total: int) -> list[int]:
    """Parse a page spec like '1,3,5-8' into sorted list of valid page numbers."""
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            bounds = part.split("-", 1)
            try:
                lo, hi = int(bounds[0]), int(bounds[1])
                pages.update(range(max(1, lo), min(total, hi) + 1))
            except ValueError:
                continue
        else:
            try:
                p = int(part)
                if 1 <= p <= total:
                    pages.add(p)
            except ValueError:
                continue
    return sorted(pages)


def _read_files(paths: list[str]) -> list[tuple[str, str]]:
    """Resolve and read multiple virtual paths. Returns [(vpath, text_or_error), ...]."""
    out = []
    for p in paths:
        try:
            real = _resolve_virtual(p)
        except ValueError as e:
            out.append((p, f"[ERROR: {e}]"))
            continue
        if not real.exists():
            out.append((p, "[ERROR: not found]"))
            continue
        out.append((p, real.read_text(encoding="utf-8", errors="replace")))
    return out


def _chunk_text(text: str, split_pattern: str = "", chunk_size: int = 1500, overlap: int = 150) -> list[str]:
    if split_pattern:
        return [c.strip() for c in re.split(split_pattern, text) if c.strip()]
    chunks, i, step = [], 0, max(1, chunk_size - overlap)
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += step
    return chunks


@tool
def search_bm25(paths: list[str], query: str, top_k: int = 5, split_pattern: str = "") -> str:
    """BM25 lexical ranked search across chunks of one or more files.
    Args:
        paths: list of virtual paths.
        query: natural-language keyword query.
        top_k: number of top chunks (default 5).
        split_pattern: optional regex to split text into chunks; default = ~1500-char chunks with 150-char overlap.
    """
    from rank_bm25 import BM25Okapi

    chunks: list[tuple[str, str]] = []
    for vpath, text in _read_files(paths):
        if text.startswith("[ERROR"):
            continue
        for c in _chunk_text(text, split_pattern):
            chunks.append((vpath, c))
    if not chunks:
        return "No content to search."

    tokenized = [re.findall(r"\w+", c.lower()) for _, c in chunks]
    bm25 = BM25Okapi(tokenized)
    q_tok = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(q_tok)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    out = []
    for idx, score in ranked:
        if score <= 0:
            continue
        vpath, c = chunks[idx]
        out.append(f"--- {vpath} (BM25={score:.2f}) ---\n{c}")
    return "\n\n".join(out) if out else "No matches."


def _get_embeddings():
    provider = os.environ.get("EMBED_PROVIDER", "openai").lower()
    if provider == "gigachat":
        from gigachat_embeddings import GigaChatEmbeddings
        return GigaChatEmbeddings.from_env()
    # default: any OpenAI-compatible embeddings endpoint (OpenRouter, internal vLLM, ...)
    api_key = os.environ.get("EMBED_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # Without credentials the POST would still transmit the document text to
        # the (external by default) endpoint before failing auth — refuse instead.
        # Callers treat this like any embeddings failure and fall back to BM25.
        raise ValueError(
            "EMBED_PROVIDER=openai needs EMBED_API_KEY (or OPENROUTER_API_KEY); "
            "refusing to send document text to an external embeddings endpoint "
            "without explicit credentials."
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=os.environ.get("EMBED_MODEL", "mistralai/mistral-embed-2312"),
        openai_api_base=os.environ.get("EMBED_API_BASE", "https://openrouter.ai/api/v1"),
        openai_api_key=api_key,
    )


@tool
def search_vector(paths: list[str], query: str, top_k: int = 5, split_pattern: str = "") -> str:
    """RAG vector search using mistralai/mistral-embed-2312 (via OpenRouter by default).
    Embeddings are cached per-(content,split) in the persistent cache (survives the
    per-run /scratch wipe), so identical text is embedded only once across runs.
    Args:
        paths: list of virtual paths.
        query: search query.
        top_k: number of top chunks (default 5).
        split_pattern: optional regex split; default = ~1500-char chunks with 150-char overlap.
    """
    import numpy as np

    cache_dir = _cache_root() / "embed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedder = None

    all_chunks: list[tuple[str, str]] = []
    all_vecs: list = []
    for vpath, text in _read_files(paths):
        if text.startswith("[ERROR"):
            continue
        chs = _chunk_text(text, split_pattern)
        if not chs:
            continue
        key = hashlib.sha1((text + "||" + split_pattern).encode("utf-8")).hexdigest()[:16]
        cache_path = cache_dir / f"{key}.npz"
        if cache_path.exists():
            vecs = np.load(cache_path)["vecs"]
        else:
            if embedder is None:
                embedder = _get_embeddings()
            vecs = np.array(embedder.embed_documents(chs), dtype=np.float32)
            np.savez(cache_path, vecs=vecs)
        for c in chs:
            all_chunks.append((vpath, c))
        all_vecs.append(vecs)
    if not all_chunks:
        return "No content to search."

    if embedder is None:
        embedder = _get_embeddings()
    qv = np.array(embedder.embed_query(query), dtype=np.float32)
    mat = np.vstack(all_vecs)
    mat_n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    qv_n = qv / (np.linalg.norm(qv) + 1e-9)
    scores = mat_n @ qv_n
    top = np.argsort(-scores)[:top_k]
    return "\n\n".join(
        f"--- {all_chunks[i][0]} (cos={scores[i]:.3f}) ---\n{all_chunks[i][1]}" for i in top
    )


@tool
def extract_tables(path: str) -> str:
    """Extract all tables from a PDF and save each as a CSV file in /scratch/.
    Returns the list of saved virtual paths (e.g. ['/scratch/file_table1.csv', ...]).
    After calling this tool, use read_file('/scratch/file_table1.csv') to read each table.
    Do NOT call extract_tables again on the same file — reuse the saved scratch files.
    Args:
        path: virtual path under /input/ or /scratch/ (e.g. '/input/report.pdf').
    """
    try:
        real = _resolve_virtual(path)
    except ValueError as e:
        return f"Error: {e}"
    if not real.exists():
        return f"Error: file not found: {path}"

    scratch_real = _work_root() / "scratch"
    stem = real.stem

    # Disk cache: reuse previously-extracted CSVs instead of re-converting.
    existing = sorted(scratch_real.glob(f"{stem}_table*.csv")) if scratch_real.exists() else []
    if existing:
        return (f"Extracted {len(existing)} table(s) (cached):\n"
                + "\n".join(f"/scratch/{p.name}" for p in existing))

    import pandas as pd  # noqa: F401  (used by docling export_to_dataframe)
    result = _convert(real, with_tables=True)
    tables = result.document.tables
    if not tables:
        return "No tables found in the document."

    scratch_real.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, table in enumerate(tables, start=1):
        out_name = f"{stem}_table{i}.csv"
        try:
            df = table.export_to_dataframe(doc=result.document)
            (scratch_real / out_name).write_text(df.to_csv(index=False), encoding="utf-8")
            saved.append(f"/scratch/{out_name}")
        except Exception as e:
            saved.append(f"Table {i}: error extracting — {e}")
    return f"Extracted {len(tables)} table(s):\n" + "\n".join(saved)


# --------------------------------------------------------------- section tools
# Docling flattens a charter's nesting to flat `##` headings, so section scope is
# derived from the CLAUSE NUMBER (12 ⊃ 12.1 ⊃ 12.1.4), not the heading level.
_SECTION_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(\S.*)$")


def _section_key(title: str) -> str:
    """Normalized section key from a heading title.
    'Статья 12 . НАБЛЮДАТЕЛЬНЫЙ СОВЕТ' -> '12'; '12.1.4. Решение…' -> '12.1.4';
    '11.1 . Компетенция…' -> '11.1'; 'СОДЕРЖАНИЕ' -> '' (no number)."""
    m = re.match(r"\s*(?:статья|глава|раздел)\s+(\d+)", title, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"\s*(\d+(?:\s*\.\s*\d+)*)", title)
    if m:
        return re.sub(r"\s+", "", m.group(1)).rstrip(".")
    return ""


def _is_descendant(key: str, parent: str) -> bool:
    """True if `key` is `parent` itself or a sub-section of it (dotted prefix)."""
    return bool(parent) and bool(key) and (key == parent or key.startswith(parent + "."))


def _iter_headings(lines: list[str]):
    """Yield (line_index, level, title, key) for each markdown heading line."""
    for i, ln in enumerate(lines):
        m = _SECTION_HEADING_RE.match(ln)
        if m:
            title = m.group(2).strip()
            yield i, len(m.group(1)), title, _section_key(title)


def _load_md(path: str) -> tuple[Path | None, str]:
    """Resolve a virtual /scratch|/input path and read it as text (markdown)."""
    try:
        real = _resolve_virtual(path)
    except ValueError as e:
        return None, f"Error: {e}"
    if not real.exists():
        return None, f"Error: file not found: {path}"
    if real.suffix.lower() == ".pdf":
        return None, (f"Error: {path} is a PDF. Call read_pdf('{path}') first, then use "
                      "the /scratch/<file>.md it produces.")
    return real, real.read_text(encoding="utf-8", errors="replace")


@tool
def list_sections(path: str) -> str:
    """List the section outline (table of contents) of a markdown file.

    Use this FIRST to see a document's structure, then read_section to pull the
    full text of a relevant section. Each line shows the section key you pass to
    read_section (e.g. '12', '12.1.4') and its title.
    Args:
        path: virtual path to a markdown file under /scratch/ (produced by
              read_pdf) or /input/.
    """
    real, text = _load_md(path)
    if real is None:
        return text
    out = []
    for _, level, title, key in _iter_headings(text.splitlines()):
        indent = "  " * max(0, level - 1)
        tag = f"[{key}] " if key else ""
        out.append(f"{indent}{tag}{title}")
    if not out:
        return f"No headings found in {path}. Use search_bm25/search_vector instead."
    return f"Sections of {path} ({len(out)}):\n" + "\n".join(out)


@tool
def read_section(path: str, section: str, max_chars: int = 8000) -> str:
    """Read ONE full section of a markdown file by key, number, or title.

    Returns the heading plus its entire body, INCLUDING nested sub-sections —
    e.g. read_section(path, '12') returns all of 'Статья 12' together with 12.1,
    12.1.4, … Use this when a clause spans many lines and snippets aren't enough.
    Args:
        path: virtual markdown path under /scratch/ or /input/.
        section: a section key/number ('12', '12.1.4'), 'Статья 12', or a word
                 from the heading title (e.g. 'Наблюдательный совет').
        max_chars: truncate the returned section to this many characters.
    """
    real, text = _load_md(path)
    if real is None:
        return text
    lines = text.splitlines()
    headings = list(_iter_headings(lines))
    if not headings:
        return f"No headings found in {path}."

    q = section.strip()
    q_key = _section_key(q) or q
    start = None
    for idx, (_li, _lvl, _title, key) in enumerate(headings):   # 1) exact key match
        if key and key == q_key:
            start = idx
            break
    if start is None:                                           # 2) title substring
        ql = q.lower()
        for idx, (_li, _lvl, title, _key) in enumerate(headings):
            if ql in title.lower():
                start = idx
                break
    if start is None:
        keys = ", ".join(k for _, _, _, k in headings if k)[:600]
        return (f"Section {section!r} not found in {path}. Available keys: {keys}. "
                f"Try list_sections('{path}').")

    start_li, _lvl, _title, start_key = headings[start]
    end_li = len(lines)
    for li, _l, _t, key in headings[start + 1:]:
        # Stop at the first later heading that is NOT inside this section.
        if start_key:
            if key and not _is_descendant(key, start_key):
                end_li = li
                break
        else:                              # title-matched, no number → next heading
            end_li = li
            break
    body = "\n".join(lines[start_li:end_li]).strip()
    if len(body) > max_chars:
        body = body[:max_chars] + (f"\n…[section truncated at {max_chars} chars; "
                                   f"narrow with a sub-key like '{start_key}.1']")
    return body


@tool
def search_examples(task_description: str, step_hint: str = "") -> str:
    """Find relevant training examples from agent_init/data/.
    The active sample (env ACTIVE_SAMPLE) is always excluded to prevent leakage during training.
    If step_hint is provided, extract only the section of the example output
    matching that step (for per-step few-shot injection)."""
    if not DATA_DIR.exists():
        return "No training data directory found."

    active = os.environ.get("ACTIVE_SAMPLE", "").strip()
    samples = [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name != active]
    if not samples:
        return "No training samples found."

    results = []
    for sample_dir in sorted(samples):
        name = sample_dir.name
        comments_path = sample_dir / "comments" / "comments.md"
        output_path = sample_dir / "output" / "res.txt"
        reference_path = sample_dir / "output" / "reference.json"

        comments = ""
        if comments_path.exists():
            comments = comments_path.read_text(encoding="utf-8", errors="replace")

        output_text = ""
        if reference_path.exists():
            import json as _json
            try:
                ref = _json.loads(reference_path.read_text(encoding="utf-8"))
                output_text = _json.dumps(ref, ensure_ascii=False, indent=2)
            except Exception:
                pass
        if not output_text and output_path.exists():
            output_text = output_path.read_text(encoding="utf-8", errors="replace")

        if step_hint and output_text:
            fragment = _extract_section(output_text, step_hint)
            if fragment:
                results.append(
                    f"=== Sample: {name} (section: {step_hint}) ===\n{fragment}"
                )
            else:
                results.append(
                    f"=== Sample: {name} ===\nNo section matching '{step_hint}' found."
                )
        else:
            preview = output_text[:500] + "..." if len(output_text) > 500 else output_text
            entry = f"=== Sample: {name} ===\n"
            if comments:
                entry += f"Comments:\n{comments}\n\n"
            if preview:
                entry += f"Output preview:\n{preview}"
            results.append(entry)

    return "\n\n".join(results)


def _extract_section(text: str, hint: str) -> str:
    """Extract a section from text that matches the hint keyword."""
    lines = text.split("\n")
    hint_lower = hint.lower()
    best_start = -1
    best_score = 0

    for i, line in enumerate(lines):
        line_lower = line.lower()
        score = 0
        for word in hint_lower.split():
            if word in line_lower:
                score += 1
        if score > best_score and (line.startswith("#") or line.startswith("**") or line.strip().endswith(":")):
            best_score = score
            best_start = i

    if best_start < 0:
        return ""

    end = len(lines)
    start_prefix = ""
    for ch in lines[best_start]:
        if ch == "#":
            start_prefix += "#"
        else:
            break

    if start_prefix:
        for j in range(best_start + 1, len(lines)):
            if lines[j].startswith(start_prefix) and not lines[j].startswith(start_prefix + "#"):
                end = j
                break

    section = "\n".join(lines[best_start:min(end, best_start + 50)])
    return section
