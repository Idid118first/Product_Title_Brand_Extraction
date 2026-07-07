# Brand extraction comparison

Generated: 2026-07-07 10:44 UTC
Config: `/Users/arnavroy/PycharmProjects/Product_Title_Brand_Extraction/configs/benchmark.csv`

Grading: normalized prediction must exactly or fuzzily match any comma-separated variant in `brands_norm` (same length-tier RapidFuzz thresholds as the extractor) on recoverable segments; abstain on segment 3.

LLM pricing basis: `gpt-4o-mini` at $0.15/1M input and $0.60/1M output tokens (OpenAI standard API rates).

## Comparison

| Metric | KnownListExtractor | LLMExtractor | HybridExtractor |
| --- | --- | --- | --- |
| Rows evaluated | 892 | 892 | 892 |
| Brands returned | 636 | 522 | 652 |
| LLM fallback calls | — | 892/892 | 283/892 |
| Accuracy (overall) | 0.981 | 0.776 | 0.953 |
| Accuracy — segment 1 | 0.994 (n=476) | 0.739 (n=476) | 0.998 (n=476) |
| Accuracy — segment 2 | 0.973 (n=150) | 0.647 (n=150) | 0.913 (n=150) |
| Accuracy — segment 3 | 0.962 (n=266) | 0.914 (n=266) | 0.895 (n=266) |
| Precision | 0.973 | 0.860 | 0.939 |
| Recall (coverage) | 0.989 | 0.717 | 0.978 |
| Abstention rate (seg 3) | 0.962 | 0.914 | 0.895 |

### Token usage & cost (LLM / hybrid fallback)

| Metric | KnownListExtractor | LLMExtractor | HybridExtractor |
| --- | --- | --- | --- |
| Model | N/A | gpt-4o-mini | gpt-4o-mini |
| Prompt tokens (run) | N/A | 411,800 | 1,008,926 |
| Completion tokens (run) | N/A | 1,774 | 387 |
| Total tokens (run) | N/A | 413,574 | 1,009,313 |
| Estimated cost (run) | N/A | $0.0628 | $0.1516 |
| Tokens per 1,000 titles | N/A | 463,648 | 1,131,517 |
| Cost per 1,000 titles | N/A | $0.0704 | $0.1699 |

## Notes

- **HybridExtractor:** LLM fallback calls: 283/892.
