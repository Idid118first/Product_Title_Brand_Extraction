import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from brand_extractor import BrandExtractor, KnownListExtractor, LLMExtractor

BENCHMARK_PATH = "benchmark.csv"
STATS_PATH = "benchmark_stats.md"
STATS_CACHE_PATH = "benchmark_stats_cache.json"
RECOVERABLE_SEGMENTS = (1, 2)  # 1 = exact in title, 2 = inferable via fuzzy; 3 = brand absent


def gold_variants(brands_norm: str) -> set[str]:
    """Acceptable gold brands for a row: each comma-separated variant, normalized."""
    return {
        BrandExtractor.normalize(part)
        for part in str(brands_norm).split(",")
        if BrandExtractor.normalize(part)
    }


@dataclass
class ExtractorStats:
    name: str
    rows: int
    brands_returned: int
    accuracy_overall: float
    accuracy_by_segment: dict[int, float]
    segment_counts: dict[int, int]
    precision: float
    recall: float
    abstention_rate: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    api_requests: int | None = None
    cost_usd: float | None = None
    tokens_per_1k_titles: float | None = None
    cost_per_1k_titles_usd: float | None = None
    model: str | None = None
    notes: str = ""


def evaluate_extractor(
    name: str,
    extractor: BrandExtractor,
    benchmark: pd.DataFrame,
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    rows = []
    total = len(benchmark)
    for i, (_, row) in enumerate(benchmark.iterrows(), start=1):
        segment = int(row["match_category"])
        recoverable = segment in RECOVERABLE_SEGMENTS
        gold = gold_variants(row["brands_norm"])

        prediction = extractor.predict(row["product_name"])
        predicted = prediction is not None
        predicted_norm = BrandExtractor.normalize(prediction) if predicted else None

        if recoverable:
            correct = predicted and predicted_norm in gold
        else:
            correct = not predicted

        rows.append(
            {"segment": segment, "recoverable": recoverable, "predicted": predicted, "correct": correct}
        )
        if show_progress and i % 25 == 0:
            print(f"  [{name}] {i}/{total} rows...", file=sys.stderr)

    return pd.DataFrame(rows)


def results_to_stats(
    name: str,
    results: pd.DataFrame,
    *,
    llm_extractor: LLMExtractor | None = None,
    notes: str = "",
) -> ExtractorStats:
    total = len(results)
    returned = results[results["predicted"]]
    recoverable = results[results["recoverable"]]
    absent = results[~results["recoverable"]]

    per_segment = results.groupby("segment")["correct"].mean().to_dict()
    segment_counts = results["segment"].value_counts().to_dict()

    stats = ExtractorStats(
        name=name,
        rows=total,
        brands_returned=int(results["predicted"].sum()),
        accuracy_overall=float(results["correct"].mean()),
        accuracy_by_segment={int(k): float(v) for k, v in per_segment.items()},
        segment_counts={int(k): int(v) for k, v in segment_counts.items()},
        precision=float(returned["correct"].mean()) if len(returned) else float("nan"),
        recall=float(recoverable["correct"].mean()) if len(recoverable) else float("nan"),
        abstention_rate=float(absent["correct"].mean()) if len(absent) else float("nan"),
        notes=notes,
    )

    if llm_extractor is not None:
        n = stats.rows
        scale = 1000 / n if n else 0
        stats.model = llm_extractor.model
        stats.prompt_tokens = llm_extractor.usage["prompt_tokens"]
        stats.completion_tokens = llm_extractor.usage["completion_tokens"]
        stats.total_tokens = llm_extractor.usage["total_tokens"]
        stats.api_requests = llm_extractor.usage["requests"]
        stats.cost_usd = llm_extractor.estimated_cost_usd()
        stats.tokens_per_1k_titles = stats.total_tokens * scale
        stats.cost_per_1k_titles_usd = stats.cost_usd * scale

    return stats


def _stats_key(name: str) -> str:
    return "known_list" if name.startswith("KnownList") else "llm"


def load_stats_cache() -> dict[str, ExtractorStats]:
    if not os.path.isfile(STATS_CACHE_PATH):
        return {}
    with open(STATS_CACHE_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, ExtractorStats] = {}
    for key, payload in raw.items():
        payload["accuracy_by_segment"] = {int(k): v for k, v in payload["accuracy_by_segment"].items()}
        payload["segment_counts"] = {int(k): v for k, v in payload["segment_counts"].items()}
        out[key] = ExtractorStats(**payload)
    return out


def save_stats_cache(cache: dict[str, ExtractorStats]) -> None:
    serializable = {key: asdict(stats) for key, stats in cache.items()}
    with open(STATS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def print_stats(stats: ExtractorStats) -> None:
    print(f"=== {stats.name} ===")
    if stats.notes:
        print(stats.notes)
    print(f"Benchmark rows: {stats.rows}")
    print(f"Brands returned (non-null): {stats.brands_returned}")
    print()
    print(f"Accuracy (overall exact-match): {stats.accuracy_overall:.3f}")
    for segment in sorted(stats.accuracy_by_segment):
        count = stats.segment_counts.get(segment, 0)
        print(f"  segment {segment} (n={count}): {stats.accuracy_by_segment[segment]:.3f}")
    print()
    returned_n = stats.brands_returned
    recoverable_n = sum(
        stats.segment_counts.get(seg, 0) for seg in stats.accuracy_by_segment if seg in RECOVERABLE_SEGMENTS
    )
    absent_n = stats.segment_counts.get(3, 0)
    print(f"Precision (correct / brand returned, n={returned_n}): {stats.precision:.3f}")
    print(f"Recall    (correct / recoverable, n={recoverable_n}): {stats.recall:.3f}")
    print(f"Abstention rate (correct None / truly absent, n={absent_n}): {stats.abstention_rate:.3f}")

    if stats.total_tokens is not None:
        print()
        print(f"Model: {stats.model}")
        print(f"API requests: {stats.api_requests}")
        print(f"Token usage — prompt: {stats.prompt_tokens:,}, completion: {stats.completion_tokens:,}, total: {stats.total_tokens:,}")
        print(f"Estimated API cost (run): ${stats.cost_usd:.4f}")
        print(f"Token usage per 1,000 titles: {stats.tokens_per_1k_titles:,.0f}")
        print(f"Estimated cost per 1,000 titles: ${stats.cost_per_1k_titles_usd:.4f}")
    print()


def _fmt_pct(value: float) -> str:
    return "—" if value != value else f"{value:.3f}"


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:.4f}"


def _fmt_tokens(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.0f}"


def write_stats_markdown(
    path: str,
    known: ExtractorStats | None,
    llm: ExtractorStats | None,
    *,
    benchmark_path: str,
) -> None:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Brand extraction benchmark comparison",
        "",
        f"Generated: {generated}",
        f"Benchmark: `{benchmark_path}`",
        "",
        "Grading: normalized prediction must match any comma-separated variant in `brands_norm` "
        "(recoverable segments) or abstain on segment 3.",
        "",
    ]

    if llm and llm.model:
        lines.extend(
            [
                f"LLM pricing basis: `{llm.model}` at "
                f"${LLMExtractor.INPUT_USD_PER_1M_TOKENS:.2f}/1M input and "
                f"${LLMExtractor.OUTPUT_USD_PER_1M_TOKENS:.2f}/1M output tokens "
                "(OpenAI standard API rates).",
                "",
            ]
        )

    def row(metric: str, known_val: str, llm_val: str) -> None:
        lines.append(f"| {metric} | {known_val} | {llm_val} |")

    lines.extend(
        [
            "## Comparison",
            "",
            "| Metric | KnownListExtractor | LLMExtractor |",
            "| --- | --- | --- |",
        ]
    )

    row("Benchmark rows", str(known.rows if known else "—"), str(llm.rows if llm else "—"))
    row("Brands returned", str(known.brands_returned if known else "—"), str(llm.brands_returned if llm else "—"))
    row(
        "Accuracy (overall)",
        _fmt_pct(known.accuracy_overall) if known else "—",
        _fmt_pct(llm.accuracy_overall) if llm else "—",
    )

    for segment in (1, 2, 3):
        k_acc = known.accuracy_by_segment.get(segment) if known else None
        l_acc = llm.accuracy_by_segment.get(segment) if llm else None
        k_n = known.segment_counts.get(segment, 0) if known else 0
        l_n = llm.segment_counts.get(segment, 0) if llm else 0
        row(
            f"Accuracy — segment {segment}",
            f"{_fmt_pct(k_acc)} (n={k_n})" if known else "—",
            f"{_fmt_pct(l_acc)} (n={l_n})" if llm else "—",
        )

    row(
        "Precision",
        _fmt_pct(known.precision) if known else "—",
        _fmt_pct(llm.precision) if llm else "—",
    )
    row(
        "Recall (coverage)",
        _fmt_pct(known.recall) if known else "—",
        _fmt_pct(llm.recall) if llm else "—",
    )
    row(
        "Abstention rate (seg 3)",
        _fmt_pct(known.abstention_rate) if known else "—",
        _fmt_pct(llm.abstention_rate) if llm else "—",
    )

    lines.extend(["", "### Token usage & cost (LLM only)", ""])
    lines.extend(
        [
            "| Metric | KnownListExtractor | LLMExtractor |",
            "| --- | --- | --- |",
        ]
    )

    row("Model", "N/A", llm.model if llm and llm.model else "—")
    row("Prompt tokens (run)", "N/A", _fmt_tokens(float(llm.prompt_tokens)) if llm and llm.prompt_tokens is not None else "—")
    row("Completion tokens (run)", "N/A", _fmt_tokens(float(llm.completion_tokens)) if llm and llm.completion_tokens is not None else "—")
    row("Total tokens (run)", "N/A", _fmt_tokens(float(llm.total_tokens)) if llm and llm.total_tokens is not None else "—")
    row("Estimated cost (run)", "N/A", _fmt_cost(llm.cost_usd) if llm else "—")
    row("Tokens per 1,000 titles", "N/A", _fmt_tokens(llm.tokens_per_1k_titles) if llm else "—")
    row("Cost per 1,000 titles", "N/A", _fmt_cost(llm.cost_per_1k_titles_usd) if llm else "—")

    if (known and known.notes) or (llm and llm.notes):
        lines.extend(["", "## Notes", ""])
        if known and known.notes:
            lines.append(f"- **KnownListExtractor:** {known.notes}")
        if llm and llm.notes:
            lines.append(f"- **LLMExtractor:** {llm.notes}")

    lines.append("")
    Path_write = path
    with open(Path_write, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate brand extractors on the frozen benchmark.")
    parser.add_argument(
        "--known-list-only",
        action="store_true",
        help="Evaluate only KnownListExtractor (skip LLM).",
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Evaluate only LLMExtractor.",
    )
    parser.add_argument(
        "--llm-limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate LLMExtractor on the first N benchmark rows only (saves API cost).",
    )
    parser.add_argument(
        "--stats-output",
        default=STATS_PATH,
        help=f"Markdown comparison output path (default: {STATS_PATH}).",
    )
    parser.add_argument(
        "--no-stats-file",
        action="store_true",
        help="Skip writing the markdown stats file.",
    )
    args = parser.parse_args()

    load_dotenv()
    benchmark = pd.read_csv(BENCHMARK_PATH, index_col=0)
    llm_benchmark = benchmark if args.llm_limit is None else benchmark.head(args.llm_limit)

    run_known = not args.llm_only
    run_llm = not args.known_list_only

    known_stats: ExtractorStats | None = None
    llm_stats: ExtractorStats | None = None

    if run_known:
        known_extractor = KnownListExtractor(config_path=BENCHMARK_PATH)
        known_results = evaluate_extractor("KnownListExtractor", known_extractor, benchmark)
        known_stats = results_to_stats("KnownListExtractor", known_results)
        print_stats(known_stats)

    if run_llm:
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            print("Skipping LLMExtractor: OPENAI_API_KEY not set in .env", file=sys.stderr)
        elif not api_key.isascii():
            print(
                "Skipping LLMExtractor: OPENAI_API_KEY contains non-ASCII characters — re-copy from OpenAI",
                file=sys.stderr,
            )
        else:
            llm_extractor = LLMExtractor(config_path=BENCHMARK_PATH)
            llm_results = evaluate_extractor(
                "LLMExtractor",
                llm_extractor,
                llm_benchmark,
                show_progress=True,
            )
            llm_notes = ""
            llm_name = "LLMExtractor"
            if args.llm_limit is not None:
                llm_name = f"LLMExtractor (first {len(llm_benchmark)} rows)"
                llm_notes = f"Evaluated on first {len(llm_benchmark)} of {len(benchmark)} benchmark rows."
            llm_stats = results_to_stats(
                llm_name,
                llm_results,
                llm_extractor=llm_extractor,
                notes=llm_notes,
            )
            print_stats(llm_stats)

    if not args.no_stats_file and (known_stats or llm_stats):
        cache = load_stats_cache()
        if known_stats:
            cache["known_list"] = known_stats
        if llm_stats:
            cache["llm"] = llm_stats
        save_stats_cache(cache)
        write_stats_markdown(
            args.stats_output,
            cache.get("known_list"),
            cache.get("llm"),
            benchmark_path=BENCHMARK_PATH,
        )
        print(f"Wrote comparison stats to {args.stats_output}")


if __name__ == "__main__":
    main()
