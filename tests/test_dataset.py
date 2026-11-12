from collections import Counter
import pytest
import dataset
from dataset import group_records, pdf_download, proportional_sample, selection_digest


def test_five_questions_grouped_one_pdf():
    rows = [{"document_id": "docA", "document_topic": "Company Policies",
             "document_path": "pdfs/repliqa_0/docA.pdf", "document_extracted": "abc",
             "question_id": f"docA-q{i}", "question": f"Question {i}",
             "answer": "answer", "long_answer": "passage"} for i in range(5)]
    groups = group_records(rows, "repliqa_0")
    assert len(groups) == 1
    assert len(groups[0]["questions"]) == 5


def test_repliqa_unanswerable_alias_is_canonicalized_and_raw_value_preserved():
    rows = [{"document_id": "docA", "document_topic": "Company Policies",
             "document_path": "pdfs/repliqa_0/docA.pdf", "document_extracted": "abc",
             "question_id": "docA-q1", "question": "Missing fact?",
             "answer": "The answer is not found in the document.", "long_answer": "NA"}]

    question = group_records(rows, "repliqa_0")[0]["questions"][0]

    assert question["answer"] == "UNANSWERABLE"
    assert question["long_answer"] == ""
    assert question["answer_raw"] == "The answer is not found in the document."
    assert question["long_answer_raw"] == "NA"


def test_proportional_selection_is_reproducible():
    groups = ([{"document_id": f"A-{i:04d}", "topic": "A"} for i in range(90)] +
              [{"document_id": f"B-{i:04d}", "topic": "B"} for i in range(10)])
    s1 = proportional_sample(groups, 20, 42)
    s2 = proportional_sample(groups, 20, 42)
    assert [d["document_id"] for d in s1] == [d["document_id"] for d in s2]
    assert Counter(x["topic"] for x in s1) == {"A": 18, "B": 2}
    assert len({x["document_id"] for x in s1}) == 20
    assert len(proportional_sample(groups, 100, 42)) == 100


def test_pdf_download_uses_hugging_face_token_without_putting_it_in_settings(tmp_path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(pdf)

    monkeypatch.setenv("HF_TOKEN", "hf_test_read_token")
    monkeypatch.setattr(dataset, "hf_hub_download", fake_download)
    group = {"document_id": "doc", "split": "repliqa_0",
             "pdf_path": "pdfs/repliqa_0/doc.pdf", "dataset_revision": "commit123"}

    assert pdf_download(group) == pdf
    assert captured["token"] == "hf_test_read_token"
    assert captured["revision"] == "commit123"
