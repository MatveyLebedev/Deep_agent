FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ripgrep \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake docling models into the image so the first run does NOT try to
# download them (auto-download fails in a closed/offline network). At runtime
# _get_converter() picks up this non-empty dir via DOCLING_ARTIFACTS_PATH.
RUN python -c "from pathlib import Path; from docling.utils.model_downloader import download_models; download_models(output_dir=Path('/workspace/models/docling'))"

COPY main.py tools.py schemas.py field_specs.py retrieval.py providers.py training.py tracing.py gigachat_embeddings.py extraction.py netguard.py ./
COPY agent_init/ ./agent_init/

RUN mkdir -p /workspace/work/current/input /workspace/work/current/scratch \
             /workspace/agent_init/data \
             /workspace/agents \
             /workspace/models/docling \
             /workspace/output

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
