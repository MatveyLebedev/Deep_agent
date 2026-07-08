# Deep Agent — пакет для внедрения в закрытой корпоративной сети

Этот каталог — самодостаточный пакет для запуска извлечения структурированных
данных из уставов ООО в **закрытом корпоративном контуре**. Собран под:

- **слабую модель `minimax-m3`** за корпоративным OpenAI-совместимым шлюзом →
  используется **детерминированный** режим извлечения (не агент-оркестратор);
- **эмбеддинги GigaChat (Sber)** для RAG-поиска (с откатом на BM25);
- **офлайн-работу**: docling-модели вшиваются в образ при сборке, трассировка
  в облако отключена.

---

## 1. Что внутри

```
deploy/
├── README_DEPLOY.md        ← этот файл
├── .env                    ← КОНФИГ контура (заполнить плейсхолдеры <...>)
├── .env.example            ← полный справочник всех переменных
├── Dockerfile              ← образ с вшитыми docling-моделями
├── docker-compose.yml      ← запуск + тома данных
├── .dockerignore / .gitignore
├── requirements.txt
├── main.py tools.py extraction.py schemas.py custom_schema.py
│   tracing.py training.py gigachat_embeddings.py   ← исходники рантайма
├── netguard.py             ← сетевой предохранитель (NETWORK_GUARD=strict)
├── SDK.py                  ← host-side Python SDK (обёртка над docker compose)
├── example.ipynb           ← готовый ноутбук с примером (раздел 7)
├── agent_init/buisness_rules.md        ← бизнес-правила (сид агента)
├── instructions/process.md, tool_tips.md
├── certs/   ← сюда положить russian_trusted_root_ca.pem (см. certs/README.md)
├── models/  ← заглушка; docling-модели живут в образе
├── input/   ← входные PDF (том, ro)
├── output/  ← результаты прогонов (том, rw)
├── work/    ← рабочий каталог + кэш ./work/cache (том, rw)
└── agents/  ← созданные агенты (том, rw)
```

---

## 2. Перед отправкой / запуском — заполнить `.env`

Откройте `.env` и подставьте реальные значения вместо `<...>`:

| Переменная | Что указать |
|---|---|
| `CUSTOM_LLM_BASE_URL` | URL корпоративного OpenAI-совместимого шлюза (`.../v1`) |
| `CUSTOM_LLM_API_KEY`  | ключ доступа к шлюзу |
| `MODEL_NAME`          | имя модели в шлюзе (по умолчанию `minimax-m3`) |
| `GIGACHAT_CREDENTIALS`| base64 от `client_id:client_secret` из кабинета GigaChat |
| `GIGACHAT_SCOPE`      | `GIGACHAT_API_CORP` / `_B2B` / `_PERS` по вашему тарифу |

Уже выставлено правильно для контура (менять не нужно):
`EMBED_PROVIDER=gigachat`, `TRACING_PROVIDER=none`, `LLM_PROVIDER=custom`.

Режим извлечения: **`EXTRACTION_MODE=agent`** — полный оркестратор с полевыми
субагентами. Он требователен к tool-calling модели: если на `minimax-m3`
прогоны стопорятся или в `structured.json` приходят пустые поля — переключитесь
на стабильный фолбэк `EXTRACTION_MODE=deterministic` (одна строка в `.env`)
или переопределите на один прогон:
`docker compose run --rm -e EXTRACTION_MODE=deterministic agent run ...`.

Опционально: `NETWORK_GUARD=strict` — программный предохранитель, который
отклоняет любое сетевое соединение, кроме настроенных эндпоинтов (LLM-шлюз,
GigaChat) и loopback. Сейчас выключен (`NETWORK_GUARD=off`, изоляцию
обеспечивает сетевой периметр); включается одной строкой в `.env`, если
политика ИБ потребует программную гарантию.

Положите корневой сертификат Минцифры в `certs/russian_trusted_root_ca.pem`
(см. `certs/README.md`).

---

## 3. Сборка образа (нужен доступ в сеть!)

`pip install` и пребейк docling-моделей требуют интернета. В закрытом контуре —
один из двух путей:

**А. Собрать снаружи и перенести образ (рекомендуется):**
```bash
# на хосте-сборщике с интернетом, из каталога deploy/
docker compose build
docker save deep-agent-app:latest | gzip > deep-agent-app.tar.gz
# перенести deep-agent-app.tar.gz + каталог deploy/ в закрытый контур, затем:
docker load < deep-agent-app.tar.gz
```

**Б. Собрать внутри контура** — если есть внутренние зеркала PyPI и HuggingFace:
```bash
# прокинуть в сборку PIP_INDEX_URL / HF_ENDPOINT через корпоративные зеркала
docker compose build
```

Готовый образ docling-модели уже содержит — в рантайме сеть для конвертации
PDF не нужна.

---

## 4. Запуск

Все операции — через `docker compose run --rm agent <команда>`.

**Шаг 1. Создать агента** (один раз; сидит инструкции и бизнес-правила):
```bash
docker compose run --rm agent create \
  --name charter \
  --process /workspace/instructions/process.md \
  --tool-tips /workspace/instructions/tool_tips.md
```

**Шаг 2. Положить PDF в `input/` и запустить извлечение:**
```bash
# файл лежит в ./input/устав.pdf  →  внутри контейнера /workspace/input/устав.pdf
docker compose run --rm agent run \
  --name charter \
  --input /workspace/input/устав.pdf
```

Результат печатается в stdout (markdown-отчёт) и сохраняется в `output/`.
Первый прогон по файлу считает docling+эмбеддинги; повторные берут готовое из
кэша `./work/cache` (по хэшу файла).

---

## 5. Проверка после установки (smoke-тест)

1. `docker compose run --rm agent --help` — образ поднимается, выводит справку.
2. Создать агента (шаг 1) — появляется каталог `agents/charter/`.
3. Прогнать тестовый PDF — в `output/` появляется результат, в логе
   `mode=agent`.
4. Если GigaChat недоступен — векторный поиск выпадает, но извлечение
   продолжает работать на лексическом поиске: в agent-режиме через инструмент
   `search_bm25`, в deterministic — автооткат BM25-only (`EXTRACTION_HYBRID`).

---

## 6. Руководство пользователя

### 6.1. Что подаётся на вход
- Один PDF: `--input /workspace/input/устав.pdf` (файл лежит в `./input/`).
- **Папка целиком (пакетный режим):** `--input /workspace/input` — обработаются
  все файлы из каталога за один прогон, результат сводится в один отчёт.

### 6.2. Что получается на выходе
Каждый прогон создаёт каталог `output/<имя_агента>/<timestamp>/`:

| Файл | Назначение |
|---|---|
| `structured.json` | **главный результат** — 7 полей в JSON (см. ниже) |
| `result.md` | тот же результат человекочитаемым markdown-отчётом |
| `work/` | снимок рабочего каталога (извлечённый markdown, scratch) для аудита |

Поля `structured.json`:

| Поле | Смысл |
|---|---|
| `supreme_governing_body` | высший орган управления |
| `collegial_governing_bodies` | коллегиальные органы (список) |
| `sole_executive_bodies` | единоличные исполнительные органы (список) |
| `major_transaction_clauses` | пункты о крупных сделках (с номерами) |
| `related_party_transaction_clauses` | пункты о сделках с заинтересованностью |
| `general_meeting_minutes_protocol` | протокол ОСУ (пункт + способ удостоверения) |
| `sole_executive_body_restrictions` | уставные ограничения ЕИО |

### 6.3. Кэш и повторные прогоны
Первый прогон по файлу считает docling + эмбеддинги; повторные берут готовое из
`./work/cache` (ключ — хэш файла). Чтобы форсировать пересчёт — удалите каталог
`./work/cache`.

### 6.4. Типовые проблемы

| Симптом | Причина / решение |
|---|---|
| В логе `BM25-only` / нет векторного поиска | GigaChat недоступен или нет CA. Извлечение продолжает работать на лексическом поиске. Проверьте `certs/` и `GIGACHAT_*` в `.env`. |
| Ошибки `429` от LLM-шлюза | Понизьте `LLM_REQUESTS_PER_SECOND` в `.env` (напр. `0.25`). |
| `download_models ... failed` при сборке | Нет доступа к HuggingFace. Собирайте образ снаружи и переносите через `docker save` (раздел 3А). |
| Пустые поля в `structured.json` / прогон стопорится | Известное ограничение режима `agent` на слабой `minimax-m3` (нужен сильный tool-calling). Переключитесь на фолбэк `EXTRACTION_MODE=deterministic` — он стабилен на слабых моделях. |
| `Agent not found` | Сначала выполните `create` (раздел 4, шаг 1). |
| TLS-ошибка к GigaChat | Положите `russian_trusted_root_ca.pem` в `certs/` (см. `certs/README.md`) или временно `GIGACHAT_VERIFY_SSL=false`. |
| `BlockedEgressError: [netguard] blocked connection to '<хост>'` | Сработал сетевой предохранитель: код попытался соединиться с ненастроенным хостом. Если это ваш легитимный эндпоинт/прокси — добавьте его в `NETWORK_EXTRA_ALLOWED_HOSTS`; если нет — так и задумано, данные наружу не ушли. |
| `RuntimeError: LLM_PROVIDER=custom requires CUSTOM_LLM_BASE_URL` | В `.env` не заполнен адрес корпоративного шлюза (остался плейсхолдер `<...>`). Заполните раздел 2. |
| `Permission denied` при записи в `output/`/`work/` | Контейнер работает от uid 1000, а каталоги на хосте принадлежат другому пользователю. Раскомментируйте `user:` в `docker-compose.yml` и подставьте uid:gid владельца каталогов. |

---

## 7. Python SDK (как в `r.ipynb`)

`SDK.py` — host-side обёртка над `docker compose run` (только stdlib, ML-зависимости
живут в контейнере). Сам прогон идёт в Docker; SDK лишь запускает контейнер и
читает результат из `output/`.

### 7.1. Предпосылки
- Образ собран (раздел 3), `.env` заполнен (раздел 2).
- На хосте: Python 3.10+ и `docker compose`.
- `DEEP_AGENT_PROJECT` указывает на **этот каталог** (`deploy/`, где лежат
  `docker-compose.yml` и `.env`).

### 7.2. Пример (режим извлечения берётся из `.env`)
```python
import os
os.environ["DEEP_AGENT_PROJECT"] = "/абсолютный/путь/к/deploy"

from SDK import create_agent, load_agent

# создать агента один раз (инструкции сидятся из instructions/)
agent = create_agent(
    name="charter",
    business_rules="agent_init/buisness_rules.md",
    process="instructions/process.md",
    tool_tips="instructions/tool_tips.md",
    overwrite=True,
)
agent = load_agent("charter")            # повторно открыть позже

# положить PDF в ./input/, затем прогнать (путь относителен /workspace в контейнере)
result = agent.run("input/устав.pdf")

print(result.structured)                 # dict — 7 полей (главный результат)
print(result.output_dir)                 # ./output/charter/<timestamp>/
print(result.output[:500])               # markdown-отчёт (str)
```

`result` (тип `RunResult`): `.structured` (dict), `.output` (markdown-строка),
`.output_dir` (Path к каталогу прогона), `.thread_id`.

### 7.3. Отличие от `r.ipynb`
В исходном ноутбуке передавались `schema_file="custom_schema.py"` и
`tools_file="tools.py"`. Эти аргументы **опциональны**: без них agent-режим
использует встроенную схему (`schemas.py`) и штатный набор инструментов, а
режим `deterministic` их игнорирует (у него всё зашито в `extraction.py`).
Передавайте их только для осознанной кастомизации. Готовый пример — в
`example.ipynb`.

> Пакетный режим из SDK: `agent.run("input")` (вся папка).
> `train()` / `test()` относятся к агент-режиму самообучения и для внедрения
> не используются.

---

## 8. Замечания по безопасности

**Гарантия «данные не уходят в интернет»** (несколько независимых слоёв):

- Доступен программный предохранитель `NETWORK_GUARD=strict` (сейчас выключен,
  см. раздел 2): процесс сам отклоняет любое соединение, кроме настроенных
  эндпоинтов (LLM-шлюз, GigaChat) и loopback, ДО отправки байтов.
- Трассировка в облако отключена (`TRACING_PROVIDER=none`), переменные
  `LANGCHAIN_*` не заданы — данные прогонов наружу не уходят.
- При `LLM_PROVIDER=custom` и пустом `CUSTOM_LLM_BASE_URL` процесс падает с
  понятной ошибкой, а не откатывается молча на `api.openai.com`.
- Эмбеддинги без ключа не отправляются: код отказывается слать текст документа
  на внешний эндпоинт без явных учётных данных (и уходит в BM25-фолбэк).
- Docling-модели вшиты в образ; `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
  запрещают huggingface_hub любые сетевые попытки в рантайме.

**Контейнер и секреты:**

- `.env` и сертификаты в git не коммитятся (`.gitignore`).
- Образ собирается без секретов (`.dockerignore` исключает `.env`); ключи
  подаются только в рантайме через `env_file`.
- Контейнер работает от непривилегированного пользователя (uid 1000), все
  capabilities сброшены (`cap_drop: ALL`), эскалация прав запрещена
  (`no-new-privileges`), порты наружу не публикуются, `input/` и `certs/`
  смонтированы только на чтение.

## 9. Что модель/агент может исполнять внутри контейнера

**Режим `deterministic` (фолбэк): модель не исполняет ничего.**
Пайплайн — фиксированный код (`extraction.py`): docling-конвертация PDF →
чанкинг → BM25/эмбеддинги → серия LLM-вызовов «вопрос → текст ответа» →
валидация в JSON-схему. Модель возвращает только текст полей; у неё нет
инструментов, она не может выполнять команды, читать/писать файлы или
открывать соединения. Инъекция во входном PDF в худшем случае искажает
значения извлечённых полей (для аудита каждый прогон сохраняет снимок
`work/` рядом с результатом).

**Режим `agent` (текущий рабочий режим):**
модель получает инструменты, но исполнение жёстко ограничено:

- **Shell-доступа нет.** Файловые маршруты работают на `FilesystemBackend`
  (только файловые операции); ни один бэкенд не реализует выполнение команд,
  поэтому инструмент `execute` фреймворка deepagents недоступен модели.
- **Файловая песочница:** запись — только в `/scratch/` (виртуальный корень
  `work/current`, наружу не выйти) и в память диалога; `/input/`,
  `/instructions/`, `/skills/` — только чтение (запрещено политиками).
- **Python-инструменты фиксированы кодом:** чтение PDF, таблицы, BM25/векторный
  поиск, чтение секций. Произвольный код появляется только если оператор сам
  положил его при `create --tools-file` (файл `custom_tools.py` в каталоге
  агента) — не передавайте этот флаг в промышленном контуре.
- **Сеть из инструментов**: только клиенты LLM/эмбеддингов (шлюз + GigaChat);
  при включённом `NETWORK_GUARD=strict` это дополнительно гарантируется кодом.

Границы контейнера (non-root, cap_drop, ro-тома, без портов) — раздел 8.
