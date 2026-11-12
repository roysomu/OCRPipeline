import json
import pytest
from pydantic import ValidationError

import pipeline
from config import Settings, TOPICS
from schemas import Answer, DocumentAnalysis, TopicClassification


def good_answer(qid="q1"):
    return {"question_id": qid, "is_answerable": True,
            "short_answer": "20 days", "long_answer": "The handbook specifies twenty days of annual leave.",
            "evidence_quote": "20 days", "evidence_page": 1}


def good_topic(topic=TOPICS[0], runner_up=TOPICS[1]):
    return {"document_type": "policy or procedure",
            "primary_subject": "employee annual leave policy",
            "topic": topic, "runner_up": runner_up,
            "decisive_reason": "The document establishes an internal employee rule."}


def test_pydantic_rejects_invalid_taxonomy_and_abstention():
    with pytest.raises(ValidationError):
        DocumentAnalysis.model_validate({"topic": "Corporate Policies", "answers": [good_answer()]})
    with pytest.raises(ValidationError):
        Answer.model_validate({"question_id": "q1", "is_answerable": False,
            "short_answer": "Guess", "long_answer": "Not sure", "evidence_quote": "", "evidence_page": None})
    with pytest.raises(ValueError):
        DocumentAnalysis.model_validate({"topic": TOPICS[0], "answers": [good_answer()]})\
            .ensure_questions(["q1", "q2"])
    with pytest.raises(ValidationError):
        TopicClassification.model_validate(good_topic(TOPICS[0], TOPICS[0]))


def test_only_markdown_pages_are_extracted():
    resp = {"model": "ocr-test", "pages": [
        {"index": 0, "markdown": "# Policy", "dimensions": {"width": 100, "height": 200},
         "confidence_scores": {"average_page_confidence_score": 0.97},
         "blocks": [{"type": "title", "top_left_x": 5, "top_left_y": 10,
                     "bottom_right_x": 95, "bottom_right_y": 30,
                     "content": "DO_NOT_SCORE",
                     "confidence_scores": {"average_content_confidence_score": 0.96}}]},
        {"markdown": "20 days", "images": [{"id": "image-no"}]}]}
    res = pipeline.ocr_visible_text(resp)
    assert res["markdown_text"] == "# Policy\n\n20 days"
    assert "DO_NOT_SCORE" not in res["text"]
    assert "[PAGE 2]" in res["text"]
    assert res["page_details"][0]["blocks"][0]["type"] == "title"
    assert res["page_details"][0]["confidence_scores"][
        "average_page_confidence_score"] == 0.97


def test_ocr_tracks_only_ocr_post_elapsed_time_per_page(tmp_path, monkeypatch):
    pdf = tmp_path / "timed.pdf"
    pdf.write_bytes(b"%PDF-1.4 timed OCR")
    api = pipeline.MistralAPI(key="test")
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(api.session, "delete", lambda *args, **kwargs: None)

    def fake_json(method, url, **kwargs):
        if url.endswith("/files"):
            return {"id": "file-1"}
        if url.endswith("/files/file-1/url"):
            return {"url": "https://signed.example.test/document"}
        assert url.endswith("/ocr")
        assert kwargs["json"]["include_blocks"] is True
        assert kwargs["json"]["confidence_scores_granularity"] == "block"
        return {"model": "ocr-test", "pages": [
            {"markdown": "Page one"}, {"markdown": "Page two"}]}

    monkeypatch.setattr(api, "_json", fake_json)
    result = api.ocr_pdf(pdf)

    assert result["processing_seconds"] == 2.5
    assert result["processing_seconds_per_page"] == 1.25
    assert result["processing_timing_scope"] == "ocr_post_request_elapsed"


def test_topic_input_extracts_title_and_two_lead_paragraphs():
    text = ("[PAGE 1]\n# River Guardians\n\nResidents restored the river.\n\n"
            "Their personal stories anchor the article.\n\nA later detail.\n\n"
            "[PAGE 2]\nMore background.")
    sections = pipeline.topic_input_sections(text)
    assert sections["title"] == "River Guardians"
    assert "Residents restored the river." in sections["lead"]
    assert "Their personal stories anchor the article." in sections["lead"]
    assert "A later detail." not in sections["lead"]
    assert "[PAGE" not in sections["lead"]


def test_topic_prompt_has_definition_and_development_example_for_every_label():
    assert set(pipeline.TOPIC_GUIDANCE) == set(TOPICS)
    for label in TOPICS:
        guidance = pipeline.TOPIC_GUIDANCE[label]
        assert guidance["definition"] in pipeline.TOPIC_SYSTEM_PROMPT
        assert guidance["example"] in pipeline.TOPIC_SYSTEM_PROMPT
        assert "repliqa_0/" in guidance["example"]
    assert "Tie-breaking rules:" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "Local News is a residual label" not in pipeline.TOPIC_SYSTEM_PROMPT
    assert "Local News and News Stories are viable editorial categories" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "general, speculative, future-oriented, or society-wide" in pipeline.TOPIC_SYSTEM_PROMPT.lower()
    assert "community business fairs" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "technology is a tool, proposed solution, setting" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "ecological benefit or theme inside community news" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "lived experiences drive the narrative" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "Hard-negative boundaries from development errors:" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "Incident Report vs Cybersecurity News" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "Local Economy and Market vs Small and Medium Enterprises" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "Local News vs Local Environmental Issues" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "Neighborhood Stories vs Local Environmental Issues" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "News Stories vs Local Arts and Culture" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "News Stories vs Local Technology and Innovation" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "Local News vs Small and Medium Enterprises" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "Local News vs Neighborhood Stories" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "Neighborhood Stories vs Local Arts and Culture" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "Local Sports and Activities vs Local Health and Wellness" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "News Stories vs Local Education Systems" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "Incident Report vs Local Economy and Market or News Stories" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "Use the supplied title and lead as initial evidence" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "apply the demonstrated dataset-label" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "conventions when a specialized semantic interpretation conflicts" in pipeline.TOPIC_SYSTEM_PROMPT
    choose_topic = pipeline.TOPIC_SYSTEM_PROMPT.index("Select the best topic")
    record_type = pipeline.TOPIC_SYSTEM_PROMPT.index("record document_type")
    assert choose_topic < record_type
    assert "Do not assign a document type yet" in pipeline.TOPIC_SYSTEM_PROMPT
    assert "Never revise the topic merely to make it agree with the document type" in pipeline.TOPIC_SYSTEM_PROMPT


def test_qa_prompt_requires_evidence_before_answerability():
    assert "EVIDENCE-FIRST" in pipeline.QA_SYSTEM_PROMPT
    assert "Only after selecting valid evidence" in pipeline.QA_SYSTEM_PROMPT
    assert "unsupported name/date/number/title/technical term" in pipeline.QA_SYSTEM_PROMPT
    assert "abstain unless the document supports every requested part" in pipeline.QA_SYSTEM_PROMPT


def test_prompt_never_contains_reference_answers_or_topic():
    api = pipeline.MistralAPI(key="test")
    calls = []
    def fake_request(method, url, **kwargs):
        payload = kwargs["json"]
        content = payload["messages"][1]["content"]
        schema_name = payload["response_format"]["json_schema"]["name"]
        calls.append(schema_name)
        assert "SECRET_GOLD_ANSWER" not in content
        assert "SECRET_GOLD_TOPIC" not in content
        assert payload["temperature"] == 0
        assert payload["top_p"] == 1
        assert payload["response_format"]["type"] == "json_schema"
        assert "schema" in payload["response_format"]["json_schema"]
        if schema_name == "TopicClassification":
            assert "Which leave entitlement?" not in content
            assert "<title>\n20 days\n</title>" in content
            assert "<lead>\n20 days\n</lead>" in content
            assert "<full_ocr_document>" in content
            assert payload["model"] == api.settings.topic_model
            assert payload["reasoning_effort"] == "none"
            assert payload["max_tokens"] == pipeline.TOPIC_REQUEST_CONFIG["max_tokens"]
            body = good_topic()
            usage = {"prompt_tokens": 180, "completion_tokens": 45}
        else:
            assert schema_name == "QuestionAnswers"
            assert payload["model"] == api.settings.chat_model
            assert "Which leave entitlement?" in content
            assert "Allowed primary topic labels" not in content
            body = {"answers": [good_answer()]}
            usage = {"prompt_tokens": 300, "completion_tokens": 70}
        return {"choices": [{"message": {"content": json.dumps(body)}}],
                "usage": usage, "model": "mistral-small-test"}
    api._json = fake_request
    analysis, usage = api.analyse("[PAGE 1]\n20 days", [{"question_id": "q1",
        "question": "Which leave entitlement?", "answer": "SECRET_GOLD_ANSWER",
        "topic": "SECRET_GOLD_TOPIC"}])
    assert analysis["topic"] == TOPICS[0]
    assert usage["prompt_tokens"] == 480
    assert calls == ["TopicClassification", "QuestionAnswers"]


def test_structured_response_extracts_final_text_after_thinking_blocks():
    api = pipeline.MistralAPI(key="test")
    api._json = lambda *args, **kwargs: {
        "choices": [{"message": {"content": [
            {"type": "thinking", "thinking": [{"type": "text", "text": "analysis"}],
             "closed": True},
            {"type": "text", "text": json.dumps(good_topic())},
        ]}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 200, "completion_tokens": 40},
        "model": "mistral-large-4",
    }

    topic, usage = api.classify_topic("[PAGE 1]\nEmployee leave policy")

    assert topic == good_topic()
    assert usage["completion_tokens"] == 40


def test_embedding_candidates_are_cosine_ranked():
    document = [1.0, 0.0]
    label_vectors = [[1.0, 0.0], [0.8, 0.2]] + [
        [-1.0, 0.1 + index * 0.01] for index in range(len(TOPICS) - 2)
    ]
    ranked = pipeline.rank_topic_candidates(document, label_vectors, 3)
    assert [item["label"] for item in ranked[:2]] == [TOPICS[0], TOPICS[1]]


def test_advisory_embedding_candidates_are_visible_but_non_binding(monkeypatch):
    configured = Settings(topic_embedding_mode="advisory", topic_embedding_top_k=3)
    api = pipeline.MistralAPI(settings=configured, key="test")
    routing = {"mode": "advisory", "model": "embed-test", "top_k": 3,
               "candidates": [{"label": TOPICS[1], "score": 0.91},
                              {"label": TOPICS[0], "score": 0.88},
                              {"label": TOPICS[2], "score": 0.80}]}
    embedding_usage = {"prompt_tokens": 30, "fresh_prompt_tokens": 30,
                       "label_setup_fresh_prompt_tokens": 0, "model": "embed-test"}
    monkeypatch.setattr(api, "_embedding_candidates",
                        lambda text: (routing, embedding_usage))

    def fake_request(method, url, **kwargs):
        content = kwargs["json"]["messages"][1]["content"]
        assert "<embedding_candidate_hints>" in content
        assert "non-binding retrieval hints" in content
        assert "Candidate shortlist (unordered" in content
        assert "similarity ranks and scores are intentionally omitted" in content
        assert "Do not assume the first candidate" in content
        assert "supporting evidence and contradicting evidence" in content
        assert "cosine similarity" not in content
        assert "0.9100" not in content
        assert TOPICS[1] in content
        assert content.index(f"- {TOPICS[0]}") < content.index(f"- {TOPICS[1]}")
        return {"choices": [{"message": {"content": json.dumps(good_topic())}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30},
                "model": "small-test"}

    monkeypatch.setattr(api, "_json", fake_request)
    topic, usage = api.classify_topic("[PAGE 1]\nEmployee leave policy")
    assert topic == good_topic()
    assert usage["embedding_routing"]["candidates"][0]["label"] == TOPICS[1]
    assert usage["embedding_routing"]["advisory_prompt_version"] == \
        pipeline.TOPIC_ADVISORY_PROMPT_VERSION
    assert usage["embedding_call"]["prompt_tokens"] == 30


def test_embedding_vectors_are_cached_without_storing_document_text(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
    configured = Settings(topic_embedding_mode="shadow", topic_embedding_top_k=3)
    api = pipeline.MistralAPI(settings=configured, key="test")
    calls = []

    def fake_embed(texts):
        calls.append(len(texts))
        if len(texts) == len(TOPICS):
            vectors = [[1.0, 0.0], [0.8, 0.2]] + [
                [-1.0, 0.1 + index * 0.01] for index in range(len(TOPICS) - 2)
            ]
        else:
            vectors = [[1.0, 0.0]]
        return vectors, {"prompt_tokens": len(texts) * 10, "model": "embed-test"}

    monkeypatch.setattr(api, "_embed_texts", fake_embed)
    first, first_usage = api._embedding_candidates("[PAGE 1]\nEmployee leave policy")
    second, second_usage = api._embedding_candidates("[PAGE 1]\nEmployee leave policy")

    assert calls == [len(TOPICS), 1]
    assert first["candidates"] == second["candidates"]
    assert first_usage["fresh_prompt_tokens"] > 0
    assert second_usage["fresh_prompt_tokens"] == 0
    document_cache = next((tmp_path / "cache" / "embeddings" / "documents").glob("*.json"))
    assert "Employee leave policy" not in document_cache.read_text()


def test_embedding_failure_falls_back_to_original_topic_prompt(monkeypatch):
    configured = Settings(topic_embedding_mode="advisory")
    api = pipeline.MistralAPI(settings=configured, key="test")
    monkeypatch.setattr(api, "_embedding_candidates",
                        lambda text: (_ for _ in ()).throw(RuntimeError("temporary outage")))

    def fake_request(method, url, **kwargs):
        content = kwargs["json"]["messages"][1]["content"]
        assert "<embedding_candidate_hints>" not in content
        return {"choices": [{"message": {"content": json.dumps(good_topic())}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30},
                "model": "small-test"}

    monkeypatch.setattr(api, "_json", fake_request)
    topic, usage = api.classify_topic("[PAGE 1]\nEmployee leave policy")
    assert topic == good_topic()
    assert usage["embedding_routing"]["status"] == "fallback_to_original_prompt"
    assert usage["embedding_routing"]["candidates"] == []


def test_topic_model_changes_only_topic_cache_key():
    group = {"document_id": "d1", "split": "repliqa_0",
             "questions": [{"question_id": "q1", "question": "Why?"}]}
    first = Settings(topic_model="large-a", chat_model="small-a")
    second = Settings(topic_model="large-b", chat_model="small-a")

    assert pipeline._topic_key("ocr-key", first) != pipeline._topic_key("ocr-key", second)
    assert pipeline._qa_key(group, "ocr-key", first) == pipeline._qa_key(group, "ocr-key", second)


def test_topic_reasoning_effort_changes_only_topic_cache_key():
    group = {"document_id": "d1", "split": "repliqa_0",
             "questions": [{"question_id": "q1", "question": "Why?"}]}
    none = Settings(topic_reasoning_effort="none")
    medium = Settings(topic_reasoning_effort="medium")

    assert pipeline._topic_key("ocr-key", none) != pipeline._topic_key("ocr-key", medium)
    assert pipeline._qa_key(group, "ocr-key", none) == pipeline._qa_key(group, "ocr-key", medium)
    assert pipeline.topic_request_config(none)["max_tokens"] == \
        pipeline.TOPIC_REQUEST_CONFIG["max_tokens"]
    assert pipeline.topic_request_config(medium)["max_tokens"] == \
        pipeline.TOPIC_REASONING_MAX_TOKENS


def test_topic_classifier_uses_configured_reasoning_effort():
    configured = Settings(topic_model="large-test", topic_reasoning_effort="medium",
                          topic_embedding_mode="off")
    api = pipeline.MistralAPI(settings=configured, key="unit-test")

    def fake_request(method, url, **kwargs):
        payload = kwargs["json"]
        assert payload["model"] == "large-test"
        assert payload["reasoning_effort"] == "medium"
        assert payload["max_tokens"] == pipeline.TOPIC_REASONING_MAX_TOKENS
        return {"choices": [{"message": {"content": json.dumps(good_topic())}}],
                "usage": {"prompt_tokens": 180, "completion_tokens": 45},
                "model": "large-test"}

    api._json = fake_request
    topic, usage = api.classify_topic("[PAGE 1]\nEmployee leave policy")

    assert topic == good_topic()
    assert usage["model"] == "large-test"


def test_runner_up_adjudicator_uses_configured_reasoning_effort():
    configured = Settings(
        topic_refinement_model="large-test",
        topic_refinement_reasoning_effort="medium",
    )
    api = pipeline.MistralAPI(settings=configured, key="unit-test")

    def fake_request(method, url, **kwargs):
        payload = kwargs["json"]
        assert payload["model"] == "large-test"
        assert payload["reasoning_effort"] == "medium"
        assert payload["max_tokens"] == pipeline.TOPIC_REASONING_MAX_TOKENS
        return {"choices": [{"message": {"content": json.dumps(good_topic())}}],
                "usage": {"prompt_tokens": 180, "completion_tokens": 45},
                "model": "large-test"}

    api._json = fake_request
    topic, usage = api.refine_topic_runner_up(
        "[PAGE 1]\nEmployee leave policy", good_topic())

    assert topic == good_topic()
    assert usage["model"] == "large-test"


def test_embedding_mode_changes_only_topic_cache_key():
    group = {"document_id": "d1", "split": "repliqa_0",
             "questions": [{"question_id": "q1", "question": "Why?"}]}
    off = Settings(topic_embedding_mode="off")
    advisory = Settings(topic_embedding_mode="advisory")
    assert pipeline._topic_key("ocr-key", off) != pipeline._topic_key("ocr-key", advisory)
    assert pipeline._qa_key(group, "ocr-key", off) == pipeline._qa_key(group, "ocr-key", advisory)


def test_disabling_runner_up_large_experiment_restores_stable_topic_key():
    disabled_a = Settings(topic_runner_up_large_enabled=False,
                          topic_refinement_model="large-a")
    disabled_b = Settings(topic_runner_up_large_enabled=False,
                          topic_refinement_model="large-b")
    enabled_a = Settings(topic_runner_up_large_enabled=True,
                         topic_refinement_model="large-a")
    enabled_b = Settings(topic_runner_up_large_enabled=True,
                         topic_refinement_model="large-b")
    enabled_medium = Settings(topic_runner_up_large_enabled=True,
                              topic_refinement_model="large-a",
                              topic_refinement_reasoning_effort="medium")

    assert pipeline._topic_key("ocr-key", disabled_a) == \
        pipeline._topic_key("ocr-key", disabled_b)
    assert pipeline._topic_key("ocr-key", enabled_a) != \
        pipeline._topic_key("ocr-key", enabled_b)
    assert pipeline._topic_key("ocr-key", enabled_a) != \
        pipeline._topic_key("ocr-key", enabled_medium)
    assert pipeline._topic_key("ocr-key", enabled_a, TOPICS[0]) != \
        pipeline._topic_key("ocr-key", enabled_a, TOPICS[1])


def test_cache_invalidates_when_prompt_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
    file = tmp_path / "sample.pdf"
    file.write_bytes(b"%PDF-1.4 test document contents")
    monkeypatch.setattr(pipeline, "pdf_download", lambda group, settings: file)
    group = {"document_id": "d1", "split": "repliqa_0", "pdf_path": "pdfs/repliqa_0/d1.pdf",
            "questions": [{"question_id": "q1", "question": "How much leave?",
                           "answer": "secret gold"}]}
    class FakeClient:
        def __init__(self): self.ocr_calls=0; self.topic_calls=0; self.qa_calls=0
        def ocr_pdf(self, pdf):
            self.ocr_calls+=1
            return {"text": "[PAGE 1]\n20 days", "markdown_pages": ["20 days"],
                    "markdown_text": "20 days", "pages": 1, "model": "ocr-test"}
        def classify_topic(self, text):
            self.topic_calls += 1
            return good_topic(), {
                "prompt_tokens": 60, "completion_tokens": 5, "model": "small-test",
                "token_usage_reported": True}
        def answer_questions(self, text, questions):
            self.qa_calls+=1
            return {"answers": [good_answer()]}, {
                "prompt_tokens": 100, "completion_tokens": 20, "model": "small-test",
                "token_usage_reported": True}
    api = FakeClient()
    first = pipeline.process_document(group, client=api)
    second = pipeline.process_document(group, client=api)
    assert api.ocr_calls == 1 and api.topic_calls == 1 and api.qa_calls == 1
    assert first["cost"]["this_run_fresh_api_estimated_usd"] > 0
    assert second["cost"]["this_run_fresh_api_estimated_usd"] == 0
    original = pipeline.QA_SYSTEM_PROMPT
    monkeypatch.setattr(pipeline, "QA_SYSTEM_PROMPT", original + " Additional revision")
    third = pipeline.process_document(group, client=api)
    assert api.ocr_calls == 1 and api.topic_calls == 1 and api.qa_calls == 2
    assert third["ocr_cached"] and third["topic_cached"] and not third["qa_cached"]
    assert first["cache_keys"]["qa"] != third["cache_keys"]["qa"]

    topic_original = pipeline.TOPIC_SYSTEM_PROMPT
    monkeypatch.setattr(pipeline, "TOPIC_SYSTEM_PROMPT", topic_original + " Topic revision")
    fourth = pipeline.process_document(group, client=api)
    assert api.ocr_calls == 1 and api.topic_calls == 2 and api.qa_calls == 2
    assert fourth["ocr_cached"] and not fourth["topic_cached"] and fourth["qa_cached"]
    assert first["cache_keys"]["topic"] != fourth["cache_keys"]["topic"]


def test_process_document_retains_embedding_routing_and_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
    file = tmp_path / "sample.pdf"
    file.write_bytes(b"%PDF-1.4 embedding test")
    monkeypatch.setattr(pipeline, "pdf_download", lambda group, settings: file)
    configured = Settings(topic_embedding_mode="advisory",
                          embedding_input_per_million=0.10)
    group = {"document_id": "embed-doc", "split": "repliqa_0",
             "pdf_path": "pdfs/repliqa_0/embed-doc.pdf",
             "questions": [{"question_id": "q1", "question": "How much leave?",
                            "answer": "20 days"}]}
    routing = {"mode": "advisory", "model": "mistral-embed", "top_k": 3,
               "candidates": [{"label": TOPICS[0], "score": 0.9}]}

    class FakeClient:
        def ocr_pdf(self, pdf):
            return {"text": "[PAGE 1]\n20 days", "markdown_pages": ["20 days"],
                    "markdown_text": "20 days", "pages": 1, "model": "ocr-test"}
        def classify_topic(self, text):
            return good_topic(), {
                "prompt_tokens": 60, "completion_tokens": 5, "model": "small-test",
                "token_usage_reported": True, "embedding_routing": routing,
                "embedding_call": {"prompt_tokens": 25, "fresh_prompt_tokens": 30,
                                   "model": "mistral-embed"}}
        def answer_questions(self, text, questions):
            return {"answers": [good_answer()]}, {
                "prompt_tokens": 100, "completion_tokens": 20,
                "model": "small-test", "token_usage_reported": True}

    first = pipeline.process_document(group, settings=configured, client=FakeClient())
    second = pipeline.process_document(group, settings=configured, client=FakeClient())
    assert first["topic_embedding"] == routing
    assert first["cost"]["embedding_usd"] > 0
    assert first["cost"]["this_run_fresh_api_estimated_usd"] > 0
    assert second["topic_cached"] is True
    assert second["cost"]["this_run_fresh_api_estimated_usd"] == 0


def test_runner_up_correct_uses_large_and_keeps_small_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
    file = tmp_path / "sample.pdf"
    file.write_bytes(b"%PDF-1.4 runner-up experiment")
    monkeypatch.setattr(pipeline, "pdf_download", lambda group, settings: file)
    configured = Settings(topic_runner_up_large_enabled=True,
                          topic_refinement_model="large-test")
    group = {"document_id": "runner-up-doc", "split": "repliqa_0",
             "topic": TOPICS[0], "pdf_path": "unused.pdf",
             "questions": [{"question_id": "q1", "question": "How much leave?",
                            "answer": "20 days"}]}

    class FakeClient:
        def __init__(self):
            self.refinement_calls = 0

        def ocr_pdf(self, pdf):
            return {"text": "[PAGE 1]\n20 days", "markdown_pages": ["20 days"],
                    "markdown_text": "20 days", "pages": 1, "model": "ocr-test"}

        def classify_topic(self, text):
            return good_topic(TOPICS[1], TOPICS[0]), {
                "prompt_tokens": 60, "completion_tokens": 5, "model": "small-test",
                "token_usage_reported": True}

        def refine_topic_runner_up(self, text, small_classification):
            self.refinement_calls += 1
            assert small_classification["topic"] == TOPICS[1]
            assert small_classification["runner_up"] == TOPICS[0]
            return good_topic(TOPICS[0], TOPICS[1]), {
                "prompt_tokens": 80, "completion_tokens": 10, "model": "large-test",
                "token_usage_reported": True}

        def answer_questions(self, text, questions):
            return {"answers": [good_answer()]}, {
                "prompt_tokens": 100, "completion_tokens": 20,
                "model": "small-test", "token_usage_reported": True}

    client = FakeClient()
    result = pipeline.process_document(group, settings=configured, client=client)

    assert client.refinement_calls == 1
    assert result["analysis"]["topic"] == TOPICS[0]
    assert result["topic_small_classification"]["topic"] == TOPICS[1]
    assert result["topic_refinement"]["status"] == "applied"
    assert result["topic_refinement"]["final_source"] == "large"
    assert result["cost"]["topic_refinement_usd"] > 0


def test_runner_up_large_failure_always_falls_back_to_small(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
    file = tmp_path / "sample.pdf"
    file.write_bytes(b"%PDF-1.4 runner-up fallback")
    monkeypatch.setattr(pipeline, "pdf_download", lambda group, settings: file)
    configured = Settings(topic_runner_up_large_enabled=True)
    group = {"document_id": "runner-up-fallback", "split": "repliqa_0",
             "topic": TOPICS[0], "pdf_path": "unused.pdf",
             "questions": [{"question_id": "q1", "question": "How much leave?",
                            "answer": "20 days"}]}

    class FakeClient:
        def ocr_pdf(self, pdf):
            return {"text": "[PAGE 1]\n20 days", "markdown_pages": ["20 days"],
                    "markdown_text": "20 days", "pages": 1, "model": "ocr-test"}

        def classify_topic(self, text):
            return good_topic(TOPICS[1], TOPICS[0]), {
                "prompt_tokens": 60, "completion_tokens": 5, "model": "small-test",
                "token_usage_reported": True}

        def refine_topic_runner_up(self, text, small_classification):
            raise RuntimeError("temporary Large outage")

        def answer_questions(self, text, questions):
            return {"answers": [good_answer()]}, {
                "prompt_tokens": 100, "completion_tokens": 20,
                "model": "small-test", "token_usage_reported": True}

    result = pipeline.process_document(group, settings=configured, client=FakeClient())

    assert result["analysis"]["topic"] == TOPICS[1]
    assert result["topic_classification"] == result["topic_small_classification"]
    assert result["topic_refinement"]["status"] == "fallback_to_small"
    assert result["topic_refinement"]["fallback_used"] is True
    assert "temporary Large outage" in result["topic_refinement"]["error"]
    assert result["cost"]["topic_refinement_usd"] == 0


def test_judge_truth_is_separate_only_after_inference():
    group = {"questions": [{"question_id": "q1", "question": "How much?",
                            "answer": "REFERENCE_SECRET", "long_answer": "Supporting paragraph"}]}
    result = {"analysis": {"answers": [good_answer()]}}
    api = pipeline.MistralAPI(key="unit-test")
    def fake(method, url, **kwargs):
        payload = kwargs["json"]
        assert "REFERENCE_SECRET" in payload["messages"][1]["content"]
        return {"choices": [{"message": {"content": json.dumps({"assessments": [
            {"question_id": "q1", "semantic_alignment_score": 4,
             "reference_supported": True, "rationale": "Largely correct"}]})}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20}}
    api._json=fake
    judge, _ = api.judge(group, result)
    assert judge["assessments"][0]["semantic_alignment_score"] == 4


def test_optional_judge_cache_tracks_actual_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
    group = {"document_id": "doc-judge", "split": "repliqa_0", "questions": [
        {"question_id": "q1", "question": "How much?", "answer": "20 days",
         "long_answer": "Twenty days were awarded"}]}
    result = {"cache_keys": {"qa": "same-qa-key"},
              "analysis": {"topic": TOPICS[0], "answers": [good_answer()]}}
    class Fake:
        def __init__(self): self.calls = 0
        def judge(self, group, result):
            self.calls += 1
            return {"assessments": [{"question_id": "q1", "semantic_alignment_score": 5,
                   "reference_supported": True, "rationale": "Correct"}]}, {
                   "prompt_tokens": 25, "completion_tokens": 20}
    client = Fake()
    first = pipeline.evaluate_semantic(group, result, client=client)
    cached = pipeline.evaluate_semantic(group, result, client=client)
    assert client.calls == 1
    assert cached["cached"]
    result["analysis"]["answers"][0]["long_answer"] = "Changed prediction"
    pipeline.evaluate_semantic(group, result, client=client)
    assert client.calls == 2
