# Deep Agent — образ для внедрения в закрытой корпоративной сети.
#
# ВАЖНО: сборка (pip install + пребейк docling-моделей) требует доступа в сеть.
# В закрытом контуре собирайте образ ОДНИМ из способов:
#   1) на хосте-сборщике с интернетом → `docker save` → перенос → `docker load`;
#   2) внутри контура с зеркалами PyPI и HuggingFace (PIP_INDEX_URL / HF_ENDPOINT).
# Готовый образ не требует сети для docling — модели уже внутри него.

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

# Пребейк docling-моделей в образ, чтобы первый запуск НЕ пытался скачать их
# (авто-загрузка падает в закрытой/офлайн-сети). В рантайме _get_converter()
# подхватывает непустой каталог через DOCLING_ARTIFACTS_PATH.
RUN python -c "from pathlib import Path; from docling.utils.model_downloader import download_models; download_models(output_dir=Path('/workspace/models/docling'))"

# Рантайм никогда не ходит в HuggingFace: модели уже в образе, а эти флаги
# запрещают huggingface_hub любые сетевые попытки даже при пустом каталоге.
ENV DOCLING_ARTIFACTS_PATH=/workspace/models/docling \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Исходники рантайма (детерминированный путь извлечения + общие модули).
COPY main.py tools.py schemas.py custom_schema.py training.py tracing.py \
     gigachat_embeddings.py extraction.py netguard.py ./
COPY agent_init/ ./agent_init/
COPY instructions/ ./instructions/

RUN mkdir -p /workspace/work/current/input /workspace/work/current/scratch \
             /workspace/agent_init/data \
             /workspace/agents \
             /workspace/models/docling \
             /workspace/certs \
             /workspace/input \
             /workspace/output

# Непривилегированный пользователь: процесс в контейнере работает НЕ от root.
# Если тома на хосте принадлежат другому uid и запись падает с "Permission
# denied" — раскомментируйте `user:` в docker-compose.yml (см. комментарий там).
RUN useradd --create-home --shell /usr/sbin/nologin --uid 1000 agent \
 && chown -R agent:agent /workspace
USER agent

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
