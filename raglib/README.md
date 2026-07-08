# raglib

RAG-библиотека: файлы → «распознавание текста» (в проде — целевая система,
здесь — мок) → персистентный иерархический индекс → поиск несколькими
инструментами. **Инвариант выдачи: результат — всегда целый пункт документа с
его номером**, пригодный для программной обработки. План и архитектура —
в [PLAN.md](PLAN.md).

## Установка

```bash
pip install -e .                    # ядро: numpy, rank-bm25, faiss-cpu
pip install -e '.[stem]'            # + snowballstemmer  (BM25 stem — лучшее качество RU)
pip install -e '.[ru]'              # + pymorphy3        (BM25 lemma)
pip install -e '.[llm]'             # + requests         (OpenAI-compatible / GigaChat)
pip install -e '.[gigachat]'        # + langchain-gigachat (эмбеддинги + chat GigaChat)
pip install -e '.[langchain]'       # + langchain-core   (@tool-обёртки, LangChainChatLLM)
pip install -e '.[langchain-openai]'# + langchain-openai (LangChainChatLLM.from_openai)
pip install -e '.[dev]'             # + pytest, ruff, build, все нормализаторы
```

Extra комбинируются: `pip install -e '.[stem,llm,langchain]'`. Ядро зависит
только от `numpy`, `rank-bm25`, `faiss-cpu` — всё остальное опционально.
RU-нормализация BM25 (`bm25_normalizer="auto"`, дефолт) сама берёт лучший
доступный бэкенд: `stem` → `lemma` → `none`.

## Сборка wheel (для переноса в закрытый контур)

Ядро — чистый Python, поэтому колесо получается универсальное
(`py3-none-any`). Для сборки нужен только `setuptools>=68` (build-backend в
`pyproject.toml`) — отдельный пакет `wheel` НЕ требуется (современный
`setuptools.build_meta` сам умеет собирать `.whl`), `build`-фронтенд тоже
опционален.

**На машине с интернетом** — удобнее через `build` (сам разрешит зависимости):

```bash
python -m pip install build
python -m build --wheel              # → dist/raglib-<версия>-py3-none-any.whl
```

**«На месте», офлайн, в контуре** — здесь есть нюанс: и `python -m build`, и
голый `pip wheel` по умолчанию создают ИЗОЛИРОВАННОЕ окружение для сборки и
пытаются СКАЧАТЬ `setuptools` из PyPI в него, даже если нужная версия уже
стоит в системе. Без сети это упадёт. Решение — флаг `--no-build-isolation`,
который заставляет использовать уже установленный в окружении `setuptools`
(в контуре это `80.9.0` — с запасом выше требуемых `>=68`):

```bash
pip wheel . --no-deps --no-build-isolation -w dist
```

Проверено: собирает колесо в venv, где кроме `setuptools==80.9.0` ничего
нет (ни `wheel`, ни `build`, ни сети) — то есть эта команда работает именно
в условиях контура.

Проверка колеса в чистом окружении:

```bash
python -m venv /tmp/check && /tmp/check/bin/pip install dist/raglib-*.whl
/tmp/check/bin/python -c "import raglib, faiss; print(raglib.__version__)"
```

Установка в закрытом контуре (без интернета) — заранее скачайте зависимости
там, где сеть есть, и перенесите вместе с колесом raglib:

```bash
# на машине с интернетом: собрать колёса всех зависимостей
pip wheel raglib[llm] -w wheelhouse            # или: pip download raglib -d wheelhouse
# в контуре: поставить только из локальной папки, без обращения к PyPI
pip install --no-index --find-links wheelhouse raglib[llm]
```

Версия задаётся в `pyproject.toml` (`project.version`) — поднимите её перед
сборкой нового колеса. Артефакты (`dist/`, `build/`) в git не коммитятся
(см. `.gitignore`).

## Деплой в закрытый контур (офлайн)

raglib совместим с окружением контура **как есть** — его зависимости уже стоят
там нужных версий (сверено с `requirements.txt` целевого сервиса):

| raglib | требует | в контуре | |
|---|---|---|---|
| `numpy` | ≥1.24 | 2.3.3 | ✅ |
| `faiss-cpu` | ≥1.7.4 | 1.12.0 | ✅ |
| `rank_bm25` | ≥0.2.2 | 0.2.2 | ✅ |
| `pymorphy3` (BM25 lemma) | ≥1.3 | 2.0.5 | ✅ |
| `langchain-gigachat` (LLM+эмбеддинги) | — | 0.5.0 | ✅ |
| `requests` | ≥2.28 | 2.33.0 | ✅ |
| `snowballstemmer` (BM25 stem) | ≥2.2 | — | опц. |
| `langchain-openai` (`from_openai`) | — | — | не нужен |

**Установка офлайн** (raglib — из колеса, зависимости фиксированы под контур —
см. [`requirements-contour.txt`](requirements-contour.txt)):

```bash
pip install --no-index --find-links wheelhouse \
    -c requirements-contour.txt 'raglib[gigachat,langchain]'
```

**BM25-нормализация**: дефолт `bm25_normalizer="auto"` в контуре сам выбирает
`lemma` (pymorphy3 есть, snowballstemmer нет) и пишет конкретный выбор в
manifest — менять `requirements` не нужно. Для лучшего качества (`stem`,
MRR 0.938 против 0.854) добавьте `snowballstemmer` (чистый Python, без
зависимостей) — `auto` подхватит его сам.

**GigaChat через LangChain** (эмбеддинги без адаптера, LLM — через `LangChainChatLLM`):

```python
from langchain_gigachat import GigaChatEmbeddings, GigaChat
from raglib import RagIndex
from raglib.agent import LangChainChatLLM

emb = GigaChatEmbeddings(credentials="...", scope="GIGACHAT_API_CORP", verify_ssl_certs=False)
index = RagIndex.build(inputs="docs/", index_dir="./idx", embeddings=emb)  # эмбеддер — напрямую

llm = LangChainChatLLM(GigaChat(credentials="...", scope="GIGACHAT_API_CORP", verify_ssl_certs=False))
res = index.agentic_search("Какие сделки требуют одобрения совета?", llm=llm, top_k=8)
```

Проверено: в venv с точными пакетами контура (numpy 2.3.3, faiss, rank-bm25,
pymorphy3; **без** snowballstemmer и langchain-openai) весь конвейер —
сборка (`auto`→`lemma`) → BM25 → навигация → перезагрузка — работает, все 65
офлайн-тестов зелёные.

## Быстрый старт

```python
from raglib import RagIndex
from raglib.recognition import MockRecognizer
from raglib.embeddings import HashingEmbeddings   # офлайн; в проде GigaChatEmbeddings

# построить индекс (вход — .md, как отдаёт целевая система распознавания)
index = RagIndex.build(
    inputs=["docs/charter.md"],            # файл / список / директория
    index_dir="./charter_index",
    recognizer=MockRecognizer(),           # шов для реальной системы распознавания
    embeddings=HashingEmbeddings(),        # None → BM25-only индекс
)

# загрузить готовый
index = RagIndex.load("./charter_index", embeddings=HashingEmbeddings())

# поиск: bm25 | vector | hybrid; strategy="tree" — сначала разделы, потом пункты
# BM25-нормализация RU: bm25_normalizer="auto" (дефолт) берёт лучший доступный
# бэкенд stem→lemma→none («сделки»≈«сделкой»); можно задать явно "stem"/"lemma"/"none".
# Конкретный выбор пишется в manifest; при load() переопределяется без пересборки.
hits = index.search("крупные сделки", method="hybrid", top_k=5)
for h in hits:
    print(h.clause_number, h.doc_id, h.score)   # "13.1", ...
    print(h.text)                               # ПОЛНЫЙ текст пункта

# оглавление и разделы: заголовки обогащаются текстом («## 7.» → первое
# предложение тела раздела), артефакты распознавания в превью не попадают
print(index.toc())                       # ключи + заголовки
print(index.toc(preview=True))           # + превью-предложение у каждого раздела
print(index.toc(clauses=True))           # + номера пунктов под каждым разделом
                                         #   (полнота сегментации видна сразу)
entries = index.toc_entries(doc="charter")   # структурно: key/title/preview/level
print(index.read_section("charter", "12.1"))  # раздел целиком, с подразделами

# find_section: по ключу, заголовку И превью содержимого; semantic=True — по смыслу
refs = index.find_section("наблюдательный совет")
refs = index.find_section("подтверждение решений собрания", semantic=True)

# навигационный поиск: раздел по смыслу → ranked-поиск внутри него
ref = refs[0]
hits = index.search("нотариальное удостоверение", method="bm25",
                    doc=ref.doc_id, section=ref.key)

# regex-поиск (выдача — те же целые пункты)
hits = index.grep(r"\d+\s*процент")

# агентский поиск: план → мульти-инструментальный поиск → LLM-рефлексия → дообыск
from raglib.agent import OpenAICompatChatLLM
llm = OpenAICompatChatLLM(base_url="https://gw.corp/v1", api_key="...", model="minimax-m3")
res = index.agentic_search("Какие сделки требуют одобрения наблюдательного совета?",
                           llm=llm, top_k=8)
print(res.degraded, [h.clause_number for h in res.hits])

# удаление (валидирует, что папка — индекс raglib)
index.delete()                     # или RagIndex.delete_index("./charter_index")
```

## Инструменты поиска

Все методы работают над одним индексом и возвращают `SearchHit` — **целый
пункт с номером** (кроме `toc_*`, отдающих структуру оглавления).

| Метод | Что делает |
| --- | --- |
| `search(q, method="bm25")` | лексический BM25 (нормализация RU: stem/lemma/none) |
| `search(q, method="vector")` | векторный (FAISS, косинус); `strategy="tree"` — разделы→пункты |
| `search(q, method="hybrid")` | BM25 + вектор через RRF |
| `search(..., doc=…, section=…)` | фильтры: документ / префикс раздела по нумерации |
| `grep(pattern)` | regex по пунктам (выдача — те же целые пункты) |
| `toc(preview=…, clauses=…)` | оглавление; `toc_entries()` — структурно |
| `find_section(q, semantic=…)` | раздел по ключу/заголовку/превью или по смыслу |
| `read_section(doc, key)` | раздел целиком, с подразделами, без усечения |
| `agentic_search(q, llm=…)` | промт → план → поиск → LLM-рефлексия → дообыск |

**Формат выдачи — `SearchHit`:**

```python
h.clause_number   # "13.1"  — номер пункта ("" у ненумерованных)
h.text            # полный текст пункта — точный срез исходного markdown, не режется
h.score           # ранг метода (BM25 / косинус / RRF / число совпадений grep)
h.doc_id          # идентификатор документа
h.section_path    # цепочка предков: ["13", "13.1"]
h.method          # bm25 | vector | hybrid | grep | agentic
h.verdict         # relevant | partial — только для agentic_search
```

`agentic_search` возвращает `AgenticResult`: `.hits`, `.trace` (журнал шагов),
`.degraded` (True = откат в hybrid), `.iterations`, `.llm_calls`.

## LLM и эмбеддинги через LangChain

Тот же стек, что в LangChain / deepagents, работает и здесь — extra `[langchain]`.

**LLM** для агентского поиска через OpenAI-адаптер LangChain
(`langchain_openai.ChatOpenAI`, extra `[langchain-openai]`) — одной строкой:

```python
from raglib.agent import LangChainChatLLM

llm = LangChainChatLLM.from_openai(model="deepseek/deepseek-v4-flash",
                                   base_url="https://gw.corp/v1", api_key="...")
res = index.agentic_search("…", llm=llm, top_k=8)
```

Или оберните уже готовую LangChain chat-модель (`.invoke → AIMessage`) —
GigaChat в контуре, кастомный подкласс `ChatOpenAI` из deepagents и т.п.
(тогда `langchain-openai` не нужен — достаточно `[langchain]`):

```python
from langchain_openai import ChatOpenAI
from raglib.agent import LangChainChatLLM

chat = ChatOpenAI(model="deepseek/deepseek-v4-flash",
                  base_url="https://gw.corp/v1", api_key="...", temperature=0)
res = index.agentic_search("…", llm=LangChainChatLLM(chat), top_k=8)
```

**Эмбеддинги** — адаптер НЕ нужен: протокол raglib (`embed_documents` /
`embed_query`) уже совместим с LangChain, передавайте объект напрямую:

```python
from langchain_gigachat import GigaChatEmbeddings          # прод в контуре
RagIndex.build(inputs=..., index_dir=..., embeddings=GigaChatEmbeddings(...))
```

> Примечание: `langchain_openai.OpenAIEmbeddings` против OpenRouter даёт
> «No embedding data received» (эта LangChain-обёртка шлёт base64 и tiktoken-
> батчинг, несовместимые с чужими шлюзами). Для OpenRouter берите родной
> `raglib.embeddings.OpenAICompatEmbeddings`; для контура — GigaChat. LLM
> через LangChain этой проблемы не имеет — `LangChainChatLLM` проверен вживую.

## Эмбеддинги

| Класс | Когда |
| --- | --- |
| `GigaChatEmbeddings.from_env()` | прод в закрытом контуре (Sber cloud / on-prem) |
| `OpenAICompatEmbeddings(...)` | любой OpenAI-совместимый /embeddings endpoint |
| любой LangChain-эмбеддер | напрямую (протокол совместим), extra `[langchain]` |
| `HashingEmbeddings()` | тесты/CI: детерминированный, без сети |
| `None` | BM25-only индекс (vector/hybrid дают понятную ошибку) |

## Результаты тестирования

**Офлайн-набор:** 65 unit/integration-тестов (фикстуры, `HashingEmbeddings`,
`MockLLM` — сеть в CI не нужна): парсинг разделов и пунктов, все артефакты
распознавания, roundtrip хранилища, все методы поиска, инварианты выдачи,
навигация, RU-нормализация (в т.ч. авто-выбор бэкенда), агентский цикл
(happy-path / refine / деградация / бюджеты). Отдельно проверен запуск в
venv, имитирующем контур (без snowballstemmer / langchain-openai).

**Боевой корпус:** два распознанных устава (ООО, 18 стр. + АО, 35 стр.,
markdown из docling), эмбеддинги `google/gemini-embedding-001`
через OpenRouter (dim 3072). Итог: 447 пунктов / 494 юнита, сборка ~30 с.

**Полнота сегментации** (после обработки артефактов распознавания: пункты-списки
`- 1.1 …`, пробелы в номерах `7 . 2.`, склейки `7.3.1.текст`, пункты-строки
таблиц, таблицы, сплющенные в одну строку, OCR-склейки `9. 21.2.`):
**ноль дыр в нумерации и ноль дубликатов номеров** на обоих уставах;
самый длинный пункт сжался с 19,6 тыс. до 3 тыс. символов.
Проверка своего корпуса: `index.toc(clauses=True)` — дыры видны сразу.

**Качество поиска** (8 перефразированных юридических запросов, релевантность —
автоматически по паттернам в тексте пункта):

| Конфигурация | MRR | Σ релевантных@5 |
|---|---|---|
| bm25, normalizer="none" | 0.719 | 24 |
| **bm25, normalizer="stem"** (дефолт) | **0.938** | **34** |
| bm25, normalizer="lemma" | 0.854 | 30 |
| hybrid (stem) | 0.854 | 32 |
| vector (gemini-embedding-001) | 0.833 | 27 |

Контрольный случай: пункт о неприменении ст. 45 (сделки с заинтересованностью)
по перефразированному запросу — вне топ-10 без нормализации → ранг 1 со stem.

**Агентский поиск вживую** на слабой модели (`deepseek/deepseek-v4-flash`,
та же, что в целевом контуре): 3 вопроса по уставам — все `degraded=False`,
по 1 итерации / 2 LLM-вызова / 25–35 с; PLAN и REFLECT стабильно возвращают
парсибельный JSON; выдача — целые пункты с вердиктами relevant/partial.

## Как выбирать инструмент (рекомендации по итогам замеров)

| Задача | Инструмент |
|---|---|
| Точные термины, номера статей, проценты | `search(method="bm25")` или `grep()` |
| Перефразированный смысловой вопрос | `search(method="vector")` или `"hybrid"` |
| «Найди раздел про X и прочитай целиком» | `find_section(semantic=True)` → `read_section()` |
| Сложный вопрос без готовой формулировки | `agentic_search()` (фильтрует шум рефлексией) |
| Большой корпус / известна область | `strategy="tree"` и/или фильтры `doc=`, `section=` |

Практические советы:

- **Нормализатор BM25**: `stem` (дефолт) — лучший по замеру; `lemma` не окупает
  зависимость pymorphy3; `none` — только если нужны точные словоформы. Менять
  режим можно при `load(bm25_normalizer=...)` — пересборка и повторные
  эмбеддинги не нужны.
- **BM25-only режим** (`embeddings=None`) — законный: BM25 + TOC + grep работают
  полностью офлайн; это же деградация при недоступности эмбеддера.
- **`find_section` по подстроке** — для случаев «примерно знаю название раздела»;
  одиночные частотные слова («протокол») цепляют преамбулы. Для смысла —
  `semantic=True`.
- **Агентский поиск**: проверяйте `res.degraded` (True = честный откат в hybrid)
  и держите `res.trace` в логах — там весь план/вердикты для разбора качества.
  A/B против обычного поиска: тот же вызов с `llm=None`.
- **Выдача любого метода** — целые пункты с номерами (`clause_number`, полный
  `text` — точный срез распознанного markdown): можно парсить программно и
  цитировать без сверки с оригиналом.

## Тесты

```bash
python -m pytest      # полностью офлайн: HashingEmbeddings + MockLLM + фикстуры
```
