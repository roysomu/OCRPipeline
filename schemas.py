"""Pydantic v2 contracts for Mistral structured outputs and local cross-field checks."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from config import TOPICS

# Literal makes the 17 permitted categories an ENUM in the JSON schema sent to Mistral.
TopicLabel = Literal[*TOPICS]
DocumentType = Literal[
    "policy or procedure",
    "business profile",
    "market or economic analysis",
    "human-interest narrative",
    "incident or emergency report",
    "event-centered news report",
    "domain analysis or explainer",
    "cultural or lifestyle feature",
    "other",
]

class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str = Field(min_length=1)
    is_answerable: bool
    short_answer: str = Field(min_length=1)
    long_answer: str = Field(min_length=1)
    evidence_quote: str
    evidence_page: int | None

    @model_validator(mode="after")
    def coherent_abstention(self):
        if not self.is_answerable:
            if (self.short_answer.strip() != "UNANSWERABLE"
                    or self.evidence_quote.strip() or self.evidence_page is not None):
                raise ValueError("Abstention must use UNANSWERABLE, empty quote, null page")
        elif (self.short_answer.strip() == "UNANSWERABLE"
              or not self.evidence_quote.strip()
              or self.evidence_page is None or self.evidence_page < 1):
            raise ValueError("Answerable item requires a short answer and evidence quote/page")
        return self


class TopicClassification(BaseModel):
    """Dedicated topic-only response contract with an auditable decision trace."""
    model_config = ConfigDict(extra="forbid")
    primary_subject: str = Field(min_length=3, max_length=180)
    topic: TopicLabel
    runner_up: TopicLabel
    # Recorded only after topic selection; diagnostic and never evaluated as a label.
    document_type: DocumentType
    decisive_reason: str = Field(min_length=5, max_length=320)

    @field_validator("topic", "runner_up")
    @classmethod
    def valid_topic(cls, value):
        if value not in TOPICS:
            raise ValueError(f"Topic must equal one of {len(TOPICS)} RepLiQA canonical labels")
        return value

    @model_validator(mode="after")
    def distinct_candidates(self):
        if self.topic == self.runner_up:
            raise ValueError("Topic and runner-up must be different labels")
        return self


class QuestionAnswers(BaseModel):
    """Dedicated question-answering response contract, intentionally without topic."""
    model_config = ConfigDict(extra="forbid")
    answers: list[Answer]

    def ensure_questions(self, question_ids):
        expected = list(question_ids)
        given = [a.question_id for a in self.answers]
        if (len(expected) != len(set(expected)) or len(given) != len(set(given))
                or set(given) != set(expected)):
            raise ValueError("Missing, extra, or duplicated question IDs in structured response")
        return self


class DocumentAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: TopicLabel
    answers: list[Answer]

    @field_validator("topic")
    @classmethod
    def valid_topic(cls, value):
        if value not in TOPICS:
            raise ValueError(f"Topic must equal one of {len(TOPICS)} RepLiQA canonical labels")
        return value

    def ensure_questions(self, question_ids):
        expected = list(question_ids)
        given = [a.question_id for a in self.answers]
        if (len(expected) != len(set(expected)) or len(given) != len(set(given))
                or set(given) != set(expected)):
            raise ValueError("Missing, extra, or duplicated question IDs in structured response")
        return self

class JudgeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str = Field(min_length=1)
    semantic_alignment_score: int = Field(ge=1, le=5)
    reference_supported: bool
    rationale: str = Field(min_length=1, max_length=400)

class JudgeBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assessments: list[JudgeItem]

    def ensure_questions(self, ids):
        given = [x.question_id for x in self.assessments]
        if len(given) != len(set(given)) or set(given) != set(ids):
            raise ValueError("Judge returned missing, extra or duplicate IDs")
        return self
