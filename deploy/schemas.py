from pydantic import BaseModel, Field


class CharterStructuredOutput(BaseModel):
    """Seven OUTPUT SCHEMA fields extracted from the agent's final report."""

    supreme_governing_body: str = Field(default="", description="Высший орган управления")
    collegial_governing_bodies: list[str] = Field(default_factory=list, description="Коллегиальные органы")
    sole_executive_bodies: list[str] = Field(default_factory=list, description="Единоличные органы")
    major_transaction_clauses: list[str] = Field(default_factory=list, description="Пункты о крупных сделках")
    related_party_transaction_clauses: list[str] = Field(
        default_factory=list, description="Пункты о сделках с заинтересованностью",
    )
    general_meeting_minutes_protocol: str = Field(default="", description="Протокол ОСУ")
    sole_executive_body_restrictions: list[str] = Field(
        default_factory=list, description="Ограничения единоличного ИО",
    )


class SubagentResult(BaseModel):
    """Structured result from any subagent step."""
    step_name: str = Field(description="Which pipeline step was executed")
    data: str = Field(description="The extracted/computed data")
    confidence: float = Field(ge=0, le=1, description="Self-assessed confidence 0-1")
    issues: list[str] = Field(default_factory=list, description="Any problems encountered")


class VerificationResult(BaseModel):
    """Result of verifying a subagent's output."""
    passed: bool = Field(description="Whether the result passes verification")
    problems: list[str] = Field(default_factory=list, description="Problems found")
    suggestion: str = Field(default="", description="How to fix if failed")


class FieldEvaluation(BaseModel):
    """Per-field verdict from the structured-output judge."""
    field: str = Field(description="Name of the schema field being judged")
    status: str = Field(
        description=(
            "correct — actual conveys the same content as expected. "
            "not_significant_error — list field where every expected item is covered AND actual has extras. "
            "significant_error — missing or wrong content."
        ),
    )
    missing: list[str] = Field(
        default_factory=list,
        description="Expected items / value not covered by actual.",
    )
    extras: list[str] = Field(
        default_factory=list,
        description="Actual items not matching any expected item (lists only).",
    )
    reasoning: str = Field(default="", description="Short justification.")


class ScoringResult(BaseModel):
    """Training mode: per-field scoring of actual vs expected structured output."""
    overall: float = Field(ge=0, le=1, description="Weighted average across fields.")
    reasoning: str = Field(default="", description="Aggregate summary.")
    per_field: list[FieldEvaluation] = Field(default_factory=list)


class DifferenceItem(BaseModel):
    """A single difference between expected and actual output."""
    type: str = Field(description="MISSING | WRONG | EXTRA | FORMAT | TOOL_MISUSE")
    description: str = Field(description="What is different")
    expected_fragment: str = Field(default="", description="What should have been")
    actual_fragment: str = Field(default="", description="What agent produced")
    severity: str = Field(description="critical | major | minor")


class InstructionEditItem(BaseModel):
    """A proposed edit to process.md or tool_tips.md."""
    target: str = Field(description="'process.md' or 'tool_tips.md'")
    operation: str = Field(description="INSERT | MODIFY | DELETE")
    section: str = Field(default="", description="Which section to edit")
    old_text: str = Field(default="", description="For MODIFY/DELETE")
    new_text: str = Field(default="", description="For INSERT/MODIFY")
    reason: str = Field(description="Why this edit is needed")


class SkillEditItem(BaseModel):
    """A proposed create/modify/delete for a skill under skills/<name>/SKILL.md."""
    name: str = Field(description="Slug, e.g. major-transactions")
    operation: str = Field(description="CREATE | MODIFY | DELETE")
    description: str = Field(
        default="",
        description="YAML frontmatter description (for CREATE/MODIFY)",
    )
    content: str = Field(
        default="",
        description="Markdown body after frontmatter (# title, When to Use, Instructions…)",
    )
    reason: str = Field(description="Why this skill change is needed")


class ReflectionResult(BaseModel):
    """Training mode: reflection output with typed edits."""
    differences: list[DifferenceItem] = Field(description="All differences found")
    proposed_edits: list[InstructionEditItem] = Field(description="Proposed instruction edits")
    proposed_skills: list[SkillEditItem] = Field(
        default_factory=list,
        description="Proposed skill files under skills/<name>/SKILL.md",
    )
