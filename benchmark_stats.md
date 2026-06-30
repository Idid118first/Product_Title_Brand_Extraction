# Brand extraction benchmark comparison

Generated: 2026-06-28 12:03 UTC
Benchmark: `benchmark.csv`

Grading: normalized prediction must match any comma-separated variant in `brands_norm` (recoverable segments) or abstain on segment 3.

LLM pricing basis: `gpt-4o-mini` at $0.15/1M input and $0.60/1M output tokens (OpenAI standard API rates).

## Comparison

| Metric | KnownListExtractor | LLMExtractor |
| --- | --- | --- |
| Benchmark rows | 892 | 892 |
| Brands returned | 636 | 614 |
| Accuracy (overall) | 0.961 | 0.889 |
| Accuracy — segment 1 | 0.994 (n=476) | 0.943 (n=476) |
| Accuracy — segment 2 | 0.853 (n=150) | 0.647 (n=150) |
| Accuracy — segment 3 | 0.962 (n=266) | 0.929 (n=266) |
| Precision | 0.945 | 0.889 |
| Recall (coverage) | 0.960 | 0.872 |
| Abstention rate (seg 3) | 0.962 | 0.929 |

### Token usage & cost (LLM only)

| Metric | KnownListExtractor | LLMExtractor |
| --- | --- | --- |
| Model | N/A | gpt-4o-mini |
| Prompt tokens (run) | N/A | 3,298,306 |
| Completion tokens (run) | N/A | 1,968 |
| Total tokens (run) | N/A | 3,300,274 |
| Estimated cost (run) | N/A | $0.4959 |
| Tokens per 1,000 titles | N/A | 3,699,859 |
| Cost per 1,000 titles | N/A | $0.5560 |
