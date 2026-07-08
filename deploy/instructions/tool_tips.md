
#### Searching for specific clause restrictions (e.g., CEO cannot do X without approval)

Many Russian charters express restrictions NOT as prohibitions on the sole executive body (e.g. 'Генеральный директор не вправе'), but as **competence items requiring consent from the Наблюдательный совет** (e.g., 'требуется согласие Наблюдательного совета на сделки выше X'). When searching for sole_executive_body_restrictions:
- **DO NOT** search only for 'не вправе' + 'Генеральн' — this pattern is rare in practice.
- **INSTEAD**, search broadly for: 'единоличн', 'Генеральн.*директор', 'требуется согласие', 'не вправе совершать', 'ограничение' and then **scan all competence/approval sections of the Наблюдательный совет** (look for language like 'сделки в размере более X%' or 'не в порядке обычной хозяйственной деятельности').
- Read the full relevant sections to identify the specific threshold clause (e.g., 'более 2% активов баланса Компании по МСФО').

**Example**: The clause `12.1.4 3) сделки В размере (балансовая, рыночная или кадастровая цена) более 2% активов баланса Компании по МСФО, осуществляемые не в порядке обычной хозяйственной деятельности;` would NOT match a search for 'директор.*не вправе'. It is found by searching for transaction-size thresholds under Наблюдательный совет competence.
#### Searching for general meeting minutes / protocol certification

When extracting **general_meeting_minutes_protocol**, the relevant text often:
- Does NOT contain the word 'протокол' (minutes) in the charter itself
- References federal law (Гражданский кодекс РФ Article 67.1 or the Federal Law on Joint-Stock Companies) governing how meeting minutes are certified/verified

**Search strategy**:
1. First grep for '67\.1' or '67.1' to find direct Civil Code citations
2. If not found, grep for 'удостоверение', 'удостовер', 'Гражданск.*кодекс' combined with 'собрани'
3. Also check the Общее собрание акционеров article sections on meeting procedures (the section heading may be 'Порядок проведения собрания' or similar)
4. Read surrounding paragraphs to confirm the charter's reference to Article 67.1 ГК РФ

**Important**: Do NOT limit the search to patterns like 'протокол.*удостовер' — this will fail if the charter says 'протокол общего собрания подлежит удостоверению в порядке, установленном статьей 67.1 ГК РФ' without those two words appearing adjacent.
#### Verifying clause numbers when extracting transaction clauses

When extracting **major_transaction_clauses** or **related_party_transaction_clauses**, multiple similar clauses may appear at different locations in the document (e.g., both under Наблюдательный совет competence and under Общее собрание акционеров competence). The output must cite the **correct clause reference number** as it appears in the document.


**Critical verification step**:
- After finding a match, **read the surrounding text** to confirm the exact clause number (item number in parentheses) and the parent section reference
- Example: A search for 'крупн' returns matches at lines where the item is numbered '(7)', '(15)', etc. — the agent must not assume item '(2)' exists if the actual document shows '(15)'
- If the charter uses numbered items like '(1)', '(2)', '(3)', always verify the number visually before writing it to the output

**Example from this document**: Searching for 'крупн' found two relevant items: '(7)' referencing п.3 ст. 79 (line 356) and '(15)' referencing п.2 ст. 79 (line 396). The clause number '(2)' does NOT appear in the document for this topic.