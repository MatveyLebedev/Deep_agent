# Deep agent — деплой в закрытый контур

Заметки по запуску основного проекта в закрытом контуре компании. Пины
зависимостей — в [`requirements-contour.txt`](requirements-contour.txt).

## 1. `langgraph-checkpoint-sqlite` — некритично

Sqlite-checkpointer используется **только для HITL-resume** (продолжить
прерванный на человеке прогон между перезапусками docker):

```python
# main.py:485
checkpointer = _checkpointer_for(agent_root) if HITL_ENABLED else MemorySaver()
```

Если HITL не нужен (детерминированный прогон — не нужен): ставим
`HITL_ENABLED=false`, и берётся `MemorySaver` — он в **ядре langgraph**
(`langgraph.checkpoint.memory`), отдельный пакет не требуется.

**Вывод:** `langgraph-checkpoint-sqlite` из зависимостей можно убрать.
Достаточно базового `langgraph-checkpoint` (есть в контуре, 4.0.3).
Единственная потеря — resume прерванного HITL-прогона после перезапуска
контейнера; в контуре это не сценарий.

## 2. Версии зависимостей

Пины — в [`requirements-contour.txt`](requirements-contour.txt), сверены с
`requirements.txt` сервиса `SBF.ML.AssistantUnified` и реестром PIP. Строки с
`(?)` (например `langsmith`, где в контуре встречались разные версии) —
**точную версию подтвердить при запуске в контуре** и зафиксировать.

Три зависимости корневого `requirements.txt` требуют решения (не просто пин):

| Зависимость | Проблема | Решение в контуре |
|---|---|---|
| `langchain-openai` | в контуре нет | LLM+эмбеддинги → `langchain-gigachat` (правка `providers.py`, `retrieval.py`, `_build_model`) |
| `docling`, `langchain-docling` | в контуре нет, тяжёлые | распознавание внешнее → вход `.md`; docling-код в `tools.py` не вызывается |
| `langgraph-checkpoint-sqlite` | sqlite-варианта нет | `MemorySaver` (см. §1) |

## 3. Не-Python пакеты в Docker — можно убрать почти все

Текущий [`Dockerfile`](Dockerfile) ставит через apt:

```
ripgrep  libgl1  libglib2.0-0  libxcb1  libsm6  libxext6  libxrender1
```

- `libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1` — все это
  графические/OpenCV-зависимости **docling** (его layout-моделей).
  Уходят вместе с docling.
- Пре-загрузка моделей docling (`Dockerfile:21`, `download_models`) — тоже
  уходит.
- `ripgrep` — нужен только встроенному fs-grep инструменту deepagents.
  В детерминированном пути (`EXTRACTION_MODE=deterministic`) / при поиске
  через raglib — не нужен.

**Вывод:** после отказа от docling образу **не нужны системные пакеты вообще** —
только Python + wheels. `faiss-cpu` и `numpy` ставятся бинарными manylinux-
колёсами (свои `.so` внутри, системные libgl/… им не требуются).

## 4. Стандартный образ компании — что в нём должно быть

Да, можно перейти на стандартный корпоративный образ вместо `python:3.12-slim`
+ apt. Требования к нему:

- **Python ≥ 3.10** (проект писался под 3.12; raglib требует ≥3.10). Лучше 3.12.
- **pip** (для установки колёс; офлайн — из локального `wheelhouse`).
- **glibc-дистрибутив** (Debian/Ubuntu/RHEL/Alt и т.п.). **НЕ Alpine/musl** —
  manylinux-колёса `faiss-cpu` и `numpy` требуют glibc и в musl-образ не
  встанут (либо придётся собирать из исходников с toolchain'ом).
- Больше **ничего системного не нужно** (после удаления docling).

Чего в образе быть НЕ должно / не обязательно: apt-libs из §3, `ripgrep`
(если без deepagents-fs-grep), CUDA/GPU-драйверы (всё на CPU).

Установка в таком образе (офлайн):

```bash
pip install --no-index --find-links wheelhouse -r requirements-contour.txt
```

## 5. Прогон под слабую модель

Оркестрация deepagents на слабой модели контура нестабильна (ради этого и
существует `providers.py` с парсингом битых tool-call'ов). Для контура —
`EXTRACTION_MODE=deterministic`: извлечение по полям кодом, без агентской
оркестрации (`main.py:_run_deterministic`, логика в `extraction.py`).
Ретрив — BM25/вектор; его можно взять из уже адаптированного под контур
`raglib/` (см. `raglib/README.md` → «Деплой в закрытый контур»).
