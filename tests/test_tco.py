import pytest
from tco import model_tco


def test_batch_projection_halves_api_not_labor():
    a = model_tco(monthly_pages=100000, batch_mode=False)
    b = model_tco(monthly_pages=100000, batch_mode=True)
    assert b["mistral_monthly_api_usd"] == pytest.approx(a["mistral_monthly_api_usd"] / 2)
    assert b["mistral_monthly_review_usd"] == a["mistral_monthly_review_usd"]
    assert a["incumbent_monthly_api_usd"] == 75000


def test_review_rates_independent_of_undefined_85_percent():
    a = model_tco(incumbent_review_rate_pct=20, mistral_review_rate_pct=3)
    assert a["incumbent_monthly_review_usd"] == pytest.approx(20000)
    assert a["mistral_monthly_review_usd"] == pytest.approx(3000)


def test_question_count_changes_tokens_not_ocr_page_count():
    a = model_tco(questions_per_doc=1)
    b = model_tco(questions_per_doc=5)
    assert a["ocr_cost_mistral_usd"] == b["ocr_cost_mistral_usd"]
    assert a["topic_cost_mistral_usd"] == b["topic_cost_mistral_usd"]
    assert a["qa_cost_mistral_usd"] < b["qa_cost_mistral_usd"]
    assert a["chat_cost_mistral_usd"] < b["chat_cost_mistral_usd"]
