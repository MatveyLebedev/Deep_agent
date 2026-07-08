
### Extract general_meeting_minutes_protocol

1. Search for '67\.1' or '67.1' in the charter to find direct references to Civil Code Article 67.1
2. If not found, search for 'Гражданск.*кодекс.*собрани' or 'удостовер.*собрани'
3. Check the Общее собрание акционеров article for procedural sections
4. Read the relevant paragraph(s) and construct the answer referencing Article 67.1 ГК РФ
5. If the charter does not contain a specific reference, check whether the applicable law (Federal Law on Joint-Stock Companies) is cited instead

**Note**: Do NOT assume the charter will use the exact phrase 'протокол.*удостовер' — this pattern is too restrictive.
### Extract sole_executive_body_restrictions

1. Search for transaction-size thresholds under the Наблюдательный совет competence section — look for clauses mentioning 'более X% активов', 'более X% баланса', 'МСФО', or 'не в порядке обычной хозяйственной деятельности'
2. Read all items in section 12.1.x (Наблюдательный совет competence) to find transaction-related restrictions
3. Search for 'единоличн.*исполнительн.*не вправе', 'Генеральн.*директор.*не вправе', 'требуется согласие.*единоличн' as supplementary patterns
4. Extract the clause text and number (e.g., '12.1.4 3)') for the transaction size threshold

**Note**: Restrictions on the sole executive body are usually found in the Наблюдательный совет competence section (requiring the supervisory board's consent for large transactions), not in a direct prohibition clause.