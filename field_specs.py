"""Single source of truth for the 7 charter output fields.

Every consumer derives its per-field texts from FIELD_SPECS:

  * main._OUTPUT_SCHEMA prompt block      ← `schema_entry`
  * main._FIELD_SUBAGENTS (agent path)    ← `subagent_name` / `subagent_description`
                                            / `agent_topic` / `style`
  * extraction.py deterministic pipeline  ← `kind` / `style` / `ru` / `keywords`
                                            / `topic`
  * schemas.CharterStructuredOutput       ← field names must match `key`s
                                            (pinned by tests/test_field_specs.py)

Editing a field = editing exactly one record here. The `topic` (deterministic
prompt) and `agent_topic` (subagent prompt) wordings differ historically and
are kept verbatim so the refactor is behavior-neutral; unify them deliberately,
not accidentally.
"""

FIELD_SPECS: list[dict] = [
    {
        "key": "supreme_governing_body",
        "kind": "str",
        "style": "name",
        "ru": "Высший орган управления",
        "keywords": "общее собрание участников высший орган управления компетенция",
        "topic": "Высший орган управления (обычно Общее собрание участников).",
        "subagent_name": "extract-supreme-body",
        "subagent_description": "Extract supreme governing body (высший орган управления).",
        "agent_topic": "Высший орган управления (Общее собрание участников)",
        "schema_entry": (
            "  supreme_governing_body — Высший орган управления   [NAME]\n"
            "        (one line, name only; default: Общее собрание участников, always present)"
        ),
    },
    {
        "key": "collegial_governing_bodies",
        "kind": "list",
        "style": "name",
        "ru": "Коллегиальные органы управления",
        "keywords": "совет директоров наблюдательный совет правление коллегиальный орган",
        "topic": "Коллегиальные органы управления: Совет директоров, Наблюдательный совет, Правление.",
        "subagent_name": "extract-collegial-bodies",
        "subagent_description": ("Extract collegial governing bodies "
                                 "(Совет директоров, Наблюдательный совет, Правление)."),
        "agent_topic": "Коллегиальные органы управления",
        "schema_entry": (
            "  collegial_governing_bodies — Коллегиальные органы управления   [NAME]\n"
            "        (one name per body: Совет директоров, Наблюдательный совет, Правление, …)"
        ),
    },
    {
        "key": "sole_executive_bodies",
        "kind": "list",
        "style": "name",
        "ru": "Единоличные исполнительные органы",
        "keywords": ("генеральный директор единоличный исполнительный орган "
                     "директор управляющий президент"),
        "topic": ("Только ЕДИНОЛИЧНЫЕ исполнительные органы: Генеральный директор, Директор, "
                  "Управляющий, Президент. НЕ включай Правление/Дирекцию (это коллегиальные органы)."),
        "subagent_name": "extract-sole-executive",
        "subagent_description": ("Extract sole executive bodies "
                                 "(Генеральный директор, Директор, Управляющий)."),
        "agent_topic": ("Только ЕДИНОЛИЧНЫЕ исполнительные органы (Генеральный директор, "
                        "Директор, Управляющий). НЕ Правление"),
        "schema_entry": (
            "  sole_executive_bodies — Единоличные органы управления   [NAME]\n"
            "        (one name per SOLE body: Генеральный директор, Директор, Управляющий, … —\n"
            "         do NOT include Правление/Дирекция, those are collegial)"
        ),
    },
    {
        "key": "major_transaction_clauses",
        "kind": "list",
        "ru": "Пункты о крупных сделках",
        "keywords": ("крупная сделка крупные сделки процент балансовой стоимости активов "
                     "одобрение порог компетенция общее собрание акционеров участников "
                     "наблюдательный совет совет директоров статья 79"),
        "topic": ("Пункты устава о крупных сделках (пороги, % от активов, кто одобряет). "
                  "Собери из компетенции ВСЕХ органов: и Общего собрания, и Наблюдательного "
                  "совета / Совета директоров — не останавливайся на первом разделе."),
        "subagent_name": "extract-major-transactions",
        "subagent_description": "Extract clauses about major transactions (крупные сделки).",
        "agent_topic": ("Крупные сделки, % от активов, пороги одобрения — собери из компетенции "
                        "ВСЕХ органов (и Общего собрания, и Наблюдательного совета / Совета директоров)"),
        "schema_entry": (
            "  major_transaction_clauses — Пункты о крупных сделках   [CLAUSE]\n"
            "        (one item per clause: \"<clause_number>. <full clause text>\";\n"
            "         collect from the competence of EVERY organ that has it)"
        ),
    },
    {
        "key": "related_party_transaction_clauses",
        "kind": "list",
        "ru": "Пункты о сделках с заинтересованностью",
        "keywords": ("сделка с заинтересованностью заинтересованные лица одобрение статья 45 "
                     "статья 83 компетенция общее собрание акционеров участников "
                     "наблюдательный совет совет директоров"),
        "topic": ("Пункты устава о сделках с заинтересованностью. Собери из компетенции ВСЕХ "
                  "органов: и Общего собрания, и Наблюдательного совета / Совета директоров — "
                  "не останавливайся на первом разделе."),
        "subagent_name": "extract-related-party-transactions",
        "subagent_description": ("Extract clauses about related-party transactions "
                                 "(сделки с заинтересованностью)."),
        "agent_topic": ("Сделки с заинтересованностью — собери из компетенции ВСЕХ органов "
                        "(и Общего собрания, и Наблюдательного совета / Совета директоров)"),
        "schema_entry": (
            "  related_party_transaction_clauses — Пункты о сделках с заинтересованностью   [CLAUSE]\n"
            "        (one item per clause: \"<clause_number>. <full clause text>\";\n"
            "         collect from the competence of EVERY organ that has it)"
        ),
    },
    {
        "key": "general_meeting_minutes_protocol",
        "kind": "str",
        "ru": "Протокол общего собрания (способ удостоверения)",
        "keywords": "протокол общего собрания удостоверение нотариус способ подтверждение решений",
        "topic": "Протокол ОСУ: номер пункта + способ удостоверения решений (нотариус / иной способ).",
        "subagent_name": "extract-meeting-protocol",
        "subagent_description": "Extract general-meeting minutes certification (протокол ОСУ).",
        "agent_topic": "Протокол общего собрания, способ удостоверения",
        "schema_entry": (
            "  general_meeting_minutes_protocol — Протокол общего собрания   [CLAUSE]\n"
            "        (one line: clause number + method of certification)"
        ),
    },
    {
        "key": "sole_executive_body_restrictions",
        "kind": "list",
        "ru": "Уставные ограничения единоличного ИО",
        "keywords": ("ограничения полномочий генерального директора предварительное согласие "
                     "одобрение совершение сделок"),
        "topic": "Уставные ограничения полномочий единоличного исполнительного органа.",
        "subagent_name": "extract-executive-restrictions",
        "subagent_description": "Extract charter restrictions on the sole executive body.",
        "agent_topic": "Уставные ограничения единоличного исполнительного органа",
        "schema_entry": (
            "  sole_executive_body_restrictions — Уставные ограничения единоличного ИО   [CLAUSE]\n"
            "        (one item per restriction, the most specific sub-clause + its text)"
        ),
    },
]
