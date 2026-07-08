"""Phase-2 code-side verification (flag-only) and NAME normalization."""
from extraction import (
    clause_in_document,
    leading_clause_number,
    strip_clause_prefix,
    verify_extracted_fields,
)

DOC = """## Статья 12. НАБЛЮДАТЕЛЬНЫЙ СОВЕТ

12.1. Компетенция Наблюдательного совета:

12.1.4. одобрение крупных сделок от 25 до 50 процентов

## Статья 13. ИСПОЛНИТЕЛЬНЫЕ ОРГАНЫ

13.4. Протокол удостоверяется нотариусом
"""


class TestLeadingClauseNumber:
    def test_extracts_dotted_number(self):
        assert leading_clause_number("12.1.4. одобрение сделок") == "12.1.4"

    def test_no_number(self):
        assert leading_clause_number("Наблюдательный совет") == ""


class TestClauseInDocument:
    def test_present(self):
        assert clause_in_document("12.1.4", DOC)
        assert clause_in_document("13.4", DOC)

    def test_absent(self):
        assert not clause_in_document("99.9", DOC)

    def test_prefix_of_longer_number_does_not_count(self):
        # DOC has 12.1. as its own clause AND inside 12.1.4 — both real.
        # But 2.1 must not match inside 12.1, and 1.4 not inside 12.1.4 / 13.4.
        assert not clause_in_document("2.1", DOC)
        assert not clause_in_document("1.4", DOC)

    def test_empty_inputs(self):
        assert not clause_in_document("", DOC)
        assert not clause_in_document("12.1", "")


class TestStripClausePrefix:
    def test_strips_number_and_separators(self):
        assert strip_clause_prefix("12.1. Наблюдательный совет") == "Наблюдательный совет"
        assert strip_clause_prefix("13.1) Генеральный директор") == "Генеральный директор"

    def test_never_empties_value(self):
        assert strip_clause_prefix("12.1.4") == "12.1.4"

    def test_list_handled(self):
        assert strip_clause_prefix(["9. Правление", "Совет директоров"]) == [
            "Правление", "Совет директоров",
        ]

    def test_plain_name_untouched(self):
        assert strip_clause_prefix("Общее собрание участников") == "Общее собрание участников"


class TestVerifyExtractedFields:
    def test_clean_result_no_warnings(self):
        structured = {
            "supreme_governing_body": "Общее собрание акционеров",
            "collegial_governing_bodies": ["Наблюдательный совет"],
            "sole_executive_bodies": ["Генеральный директор"],
            "major_transaction_clauses": ["12.1.4. одобрение крупных сделок"],
            "related_party_transaction_clauses": [],
            "general_meeting_minutes_protocol": "13.4. нотариусом",
            "sole_executive_body_restrictions": [],
        }
        assert verify_extracted_fields(structured, DOC) == []

    def test_fabricated_clause_number_flagged(self):
        structured = {"major_transaction_clauses": ["79.3. текст из примера"]}
        warnings = verify_extracted_fields(structured, DOC)
        assert len(warnings) == 1
        assert "79.3" in warnings[0] and "major_transaction_clauses" in warnings[0]

    def test_name_with_clause_number_flagged(self):
        structured = {"collegial_governing_bodies": ["12.1. Наблюдательный совет"]}
        warnings = verify_extracted_fields(structured, DOC)
        assert len(warnings) == 1
        assert "название органа" in warnings[0]

    def test_ne_ukazano_and_empty_skipped(self):
        structured = {
            "general_meeting_minutes_protocol": "не указано",
            "major_transaction_clauses": [],
        }
        assert verify_extracted_fields(structured, DOC) == []

    def test_items_without_leading_number_not_checked(self):
        structured = {"sole_executive_body_restrictions": ["ограничение без номера"]}
        assert verify_extracted_fields(structured, DOC) == []
