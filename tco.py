"""Transparent scenario calculator. Not a claim of achieved accuracy or billed savings."""
from config import Settings


def model_tco(monthly_pages=100000, avg_pages_per_doc=2, questions_per_doc=5,
              batch_mode=False, incumbent_review_rate_pct=15.0,
              mistral_review_rate_pct=5.0, review_cost_per_page=1.0,
              tokens_per_page=800, fixed_prompt_tokens_per_doc=300,
              prompt_tokens_per_question=65, output_tokens_per_question=95,
              topic_prompt_tokens_per_doc=900, topic_output_tokens_per_doc=20,
              settings=None):
    settings = settings or Settings()
    if monthly_pages <= 0 or avg_pages_per_doc <= 0 or questions_per_doc <= 0:
        raise ValueError("All volume and document-size values must be positive")
    if not (0 <= incumbent_review_rate_pct <= 100 and 0 <= mistral_review_rate_pct <= 100):
        raise ValueError("Review rates must lie between 0 and 100")
    docs_per_month = monthly_pages / avg_pages_per_doc
    document_tokens = tokens_per_page * avg_pages_per_doc
    topic_token_in = topic_prompt_tokens_per_doc + document_tokens
    qa_token_in = fixed_prompt_tokens_per_doc + document_tokens + prompt_tokens_per_question * questions_per_doc
    token_in = topic_token_in + qa_token_in
    topic_token_out = topic_output_tokens_per_doc
    qa_token_out = 60 + output_tokens_per_question * questions_per_doc
    token_out = topic_token_out + qa_token_out
    batch_multiplier = 0.5 if batch_mode else 1.0
    ocr = monthly_pages * settings.ocr_per_1000_pages / 1000 * batch_multiplier
    topic = docs_per_month * (
        topic_token_in * settings.topic_input_per_million / 1_000_000
        + topic_token_out * settings.topic_output_per_million / 1_000_000
    ) * batch_multiplier
    qa = docs_per_month * (
        qa_token_in * settings.chat_input_per_million / 1_000_000
        + qa_token_out * settings.chat_output_per_million / 1_000_000
    ) * batch_multiplier
    chat = topic + qa
    incumbent_api = monthly_pages * settings.incumbent_per_page
    mistral_api = ocr + chat
    incumbent_reviews = monthly_pages * incumbent_review_rate_pct / 100
    mistral_reviews = monthly_pages * mistral_review_rate_pct / 100
    incumbent_labor = incumbent_reviews * review_cost_per_page
    mistral_labor = mistral_reviews * review_cost_per_page
    return {
        "monthly_pages": int(monthly_pages), "documents_per_month_approx": docs_per_month,
        "questions_per_doc": questions_per_doc, "pages_per_doc": avg_pages_per_doc,
        "batch_scenario_only_not_executed": bool(batch_mode),
        "batch_multiplier_assumed": batch_multiplier,
        "unit_token_assumptions": {"input_per_document": token_in, "output_per_document": token_out,
             "topic_call": {"input": topic_token_in, "output": topic_token_out},
             "qa_call": {"input": qa_token_in, "output": qa_token_out}},
        "ocr_cost_mistral_usd": ocr, "topic_cost_mistral_usd": topic,
        "qa_cost_mistral_usd": qa, "chat_cost_mistral_usd": chat,
        "mistral_api_per_page_usd": mistral_api / monthly_pages,
        "incumbent_monthly_api_usd": incumbent_api,
        "mistral_monthly_api_usd": mistral_api,
        "incumbent_monthly_review_usd": incumbent_labor,
        "mistral_monthly_review_usd": mistral_labor,
        "incumbent_monthly_tco_usd": incumbent_api + incumbent_labor,
        "mistral_monthly_tco_usd": mistral_api + mistral_labor,
        "monthly_tco_difference_usd": incumbent_api + incumbent_labor - mistral_api - mistral_labor,
        "annualized_tco_difference_usd": 12 * (incumbent_api + incumbent_labor - mistral_api - mistral_labor),
        "price_as_of": settings.price_as_of,
        "disclaimer": (
            "Illustrative model ONLY. Incumbent 85% has no defined metric; the review rates "
            "are independently entered assumptions, not derived from benchmark accuracy. "
            "Batch discount is scenario only (not executed by this code). Cost excludes "
            "infrastructure, egress, retries, errors and enterprise contracts."
        ),
    }


def tco_markdown(data):
    def money(x): return f"${x:,.2f}"
    return f"""### Commercial scenario (NOT measured outcome)

| Metric | Incumbent | Mistral scenario |
|---|---:|---:|
| Monthly API costs | {money(data['incumbent_monthly_api_usd'])} | {money(data['mistral_monthly_api_usd'])} |
| Monthly review costs | {money(data['incumbent_monthly_review_usd'])} | {money(data['mistral_monthly_review_usd'])} |
| **Monthly TCO** | **{money(data['incumbent_monthly_tco_usd'])}** | **{money(data['mistral_monthly_tco_usd'])}** |

**Illustrative annualized TCO difference:** {money(data['annualized_tco_difference_usd'])}<br>
**Mistral modeled API cost/page:** ${data['mistral_api_per_page_usd']:.5f}<br>
**Token assumptions per document:** {data['unit_token_assumptions']['input_per_document']:,.0f} input / {data['unit_token_assumptions']['output_per_document']:,.0f} output<br>
**Price assumption date:** {data['price_as_of']}

**Important:** {data['disclaimer']}
"""
