# Mistral PSA Gradio V2 — video and panel narrative

## Five-minute video (record screen only)

| Time | Screen | Narration |
|---|---|---|
| 0:00–0:35 | Gradio introduction | Customer reports $0.75/page, ~85% accuracy of unspecified definition. Objective is a measurable, apples-to-apples pilot, not an invented accuracy-lift number. |
| 0:35–1:20 | Inspector, dev split | One PDF, five questions, one OCR + a dedicated Small topic call + one evidence-first Small QA call. Explicit abstention when no direct evidence supports every requested part. |
| 1:20–2:00 | OCR comparison | WER runs only on visible page Markdown; the reference is auto-extracted text, so it is a proxy rather than human-certified OCR truth. |
| 2:00–2:45 | Evidence and schema | Demonstrate quote/page verification and explain why it checks quote location but not semantic entailment. Mention optional offline judge. |
| 2:45–3:35 | Holdout evaluator / precomputed results | Show `repliqa_3` evaluation, reproducible sampling, per-topic metrics, PDF-clustered confidence intervals and counts of failed documents. **Do not pretend cached results are live or show unmeasured numbers.** |
| 3:35–4:25 | Commercial TCO | Change page volume and manually entered exception rates. The 50% batch toggle is a *pricing projection*, not an executed Batch API job. |
| 4:25–5:00 | Conclusions | Proposed pilot: evaluate incumbent and Mistral on identical documents, specify KPI thresholds, then consider hosted production and partner rollout. |

## 60-minute panel structure

- 0–5: Customer discovery and definition of success (what is "85%"?).
- 5–13: Architecture and Mistral model decisions; short documents, five-question batching, no vector database by default.
- 13–27: Live Gradio walkthrough; separately show dev inference and frozen holdout results.
- 27–38: Evaluation metrics, failure analysis, confidence intervals, reference limitations.
- 38–46: Unit economics and human-review scenario with clear assumptions.
- 46–53: Scaling: async jobs, Batch API architecture, error recovery, data sovereignty, partner enablement.
- 53–60: Panel challenges and close on agreed proof-of-value acceptance criteria.

## Defensible statements

- "The holdout measures generalization of a frozen pipeline, not a competitor's accuracy."
- "My QA metric is question-level but uncertainty is clustered at document level."
- "I use OCR Markdown-only text for WER; image references and block JSON never become tokens."
- "The 1-to-5 LLM judge is a secondary reference-alignment rubric. I don't convert it to an accuracy percentage."
- "The cost model starts from measured token usage and configurable public list prices. Human-review savings remain a scenario until observed in a customer pilot."
- "For high throughput I would adopt asynchronous Batch OCR and batch inference separately; the present Gradio implementation is synchronous and resumable, not a Batch API deployment."

## Pilot exit criteria (illustrative, to agree with customer)

1. Same blinded customer documents and question sets on both incumbent and Mistral; agreed definition of correctness and manual adjudication for disputes.
2. Confidence intervals and topic-level error analysis, with high-risk question types and false answers included.
3. Per-page API and end-to-end workflow economics including human review and exception handling.
4. Document retention, regional processing, tenant isolation, rate limits, observability and partner-operated support model verified before production.
