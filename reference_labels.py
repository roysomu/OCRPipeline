"""Canonical handling of RepLiQA answerability labels.

RepLiQA revisions have represented an unanswerable reference with both the
literal ``UNANSWERABLE`` and the sentence ``The answer is not found in the
document.``.  Some revisions additionally use ``NA`` for the evidence
paragraph.  Centralizing these aliases prevents evaluation drift.
"""
import re


CANONICAL_UNANSWERABLE = "UNANSWERABLE"

_ANSWER_MARKERS = {
    "unanswerable",
    "the answer is not found in the document",
}
_EVIDENCE_MARKERS = {"na", "n/a", "not applicable"}


def _normalise_marker(value) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
    return text.rstrip(". ")


def is_unanswerable_reference(answer, long_answer=None) -> bool:
    """Return whether a RepLiQA reference denotes an unanswerable question."""
    if _normalise_marker(answer) in _ANSWER_MARKERS:
        return True
    return long_answer is not None and _normalise_marker(long_answer) in _EVIDENCE_MARKERS


def normalise_reference_answer(answer, long_answer=None) -> str:
    """Map known dataset aliases to the pipeline's canonical sentinel."""
    raw = str(answer or "")
    return CANONICAL_UNANSWERABLE if is_unanswerable_reference(raw, long_answer) else raw
