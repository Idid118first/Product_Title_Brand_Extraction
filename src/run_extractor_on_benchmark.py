import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from brand_extractor import (
    CONFIGS_DIR,
    BrandExtractor,
    HybridExtractor,
    KnownListExtractor,
    LLMExtractor,
    resolve_config_path,
)
STATS_PATH = "benchmark_stats.md"
STATS_CACHE_PATH = "benchmark_stats_cache.json"
RECOVERABLE_SEGMENTS = (1, 2)  # 1 = exact in title, 2 = inferable via fuzzy; 3 = brand absent
# Food-domain generic-word/ingredient lists used to reject spurious fuzzy matches on this
# benchmark. Override with --rejection-list when evaluating a different domain.
DEFAULT_REJECTION_LISTS = (
    CONFIGS_DIR / "1-1000.txt",
    CONFIGS_DIR / "10000_common_food_ingredients.txt",
)


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
    config_df: pd.DataFrame,
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    rows = []
    total = len(config_df)
    for i, (_, row) in enumerate(config_df.iterrows(), start=1):
        segment = int(row["match_category"])
        recoverable = segment in RECOVERABLE_SEGMENTS
        gold = gold_variants(row["brands_norm"])

        prediction = extractor.predict(row["product_name"])
        predicted = prediction is not None

        if recoverable:
            correct = bool(prediction) and BrandExtractor.matches_any_gold_variant(prediction, gold)
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
    llm_fallback_requests: int | None = None,
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
        if llm_fallback_requests is not None:
            stats.notes = (
                f"{notes} LLM fallback calls: {llm_fallback_requests}/{stats.rows}."
                if notes
                else f"LLM fallback calls: {llm_fallback_requests}/{stats.rows}."
            )

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
    print(f"Rows evaluated: {stats.rows}")
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
        if stats.notes and "LLM fallback calls" in stats.notes:
            print(stats.notes)
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
    hybrid: ExtractorStats | None = None,
    *,
    config_path: str,
) -> None:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Brand extraction comparison",
        "",
        f"Generated: {generated}",
        f"Config: `{config_path}`",
        "",
        "Grading: normalized prediction must exactly or fuzzily match any comma-separated variant "
        "in `brands_norm` (same length-tier RapidFuzz thresholds as the extractor) on recoverable "
        "segments; abstain on segment 3.",
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

    def row(metric: str, known_val: str, llm_val: str, hybrid_val: str = "—") -> None:
        lines.append(f"| {metric} | {known_val} | {llm_val} | {hybrid_val} |")

    lines.extend(
        [
            "## Comparison",
            "",
            "| Metric | KnownListExtractor | LLMExtractor | HybridExtractor |",
            "| --- | --- | --- | --- |",
        ]
    )

    row("Rows evaluated", str(known.rows if known else "—"), str(llm.rows if llm else "—"), str(hybrid.rows if hybrid else "—"))
    row(
        "Brands returned",
        str(known.brands_returned if known else "—"),
        str(llm.brands_returned if llm else "—"),
        str(hybrid.brands_returned if hybrid else "—"),
    )
    row(
        "LLM fallback calls",
        "—",
        f"{llm.api_requests}/{llm.rows}" if llm and llm.api_requests is not None else "—",
        f"{hybrid.api_requests}/{hybrid.rows}" if hybrid and hybrid.api_requests is not None else "—",
    )
    row(
        "Accuracy (overall)",
        _fmt_pct(known.accuracy_overall) if known else "—",
        _fmt_pct(llm.accuracy_overall) if llm else "—",
        _fmt_pct(hybrid.accuracy_overall) if hybrid else "—",
    )

    for segment in (1, 2, 3):
        k_acc = known.accuracy_by_segment.get(segment) if known else None
        l_acc = llm.accuracy_by_segment.get(segment) if llm else None
        h_acc = hybrid.accuracy_by_segment.get(segment) if hybrid else None
        k_n = known.segment_counts.get(segment, 0) if known else 0
        l_n = llm.segment_counts.get(segment, 0) if llm else 0
        h_n = hybrid.segment_counts.get(segment, 0) if hybrid else 0
        row(
            f"Accuracy — segment {segment}",
            f"{_fmt_pct(k_acc)} (n={k_n})" if known else "—",
            f"{_fmt_pct(l_acc)} (n={l_n})" if llm else "—",
            f"{_fmt_pct(h_acc)} (n={h_n})" if hybrid else "—",
        )

    row(
        "Precision",
        _fmt_pct(known.precision) if known else "—",
        _fmt_pct(llm.precision) if llm else "—",
        _fmt_pct(hybrid.precision) if hybrid else "—",
    )
    row(
        "Recall (coverage)",
        _fmt_pct(known.recall) if known else "—",
        _fmt_pct(llm.recall) if llm else "—",
        _fmt_pct(hybrid.recall) if hybrid else "—",
    )
    row(
        "Abstention rate (seg 3)",
        _fmt_pct(known.abstention_rate) if known else "—",
        _fmt_pct(llm.abstention_rate) if llm else "—",
        _fmt_pct(hybrid.abstention_rate) if hybrid else "—",
    )

    lines.extend(["", "### Token usage & cost (LLM / hybrid fallback)", ""])
    lines.extend(
        [
            "| Metric | KnownListExtractor | LLMExtractor | HybridExtractor |",
            "| --- | --- | --- | --- |",
        ]
    )

    row("Model", "N/A", llm.model if llm and llm.model else "—", hybrid.model if hybrid and hybrid.model else "—")
    row(
        "Prompt tokens (run)",
        "N/A",
        _fmt_tokens(float(llm.prompt_tokens)) if llm and llm.prompt_tokens is not None else "—",
        _fmt_tokens(float(hybrid.prompt_tokens)) if hybrid and hybrid.prompt_tokens is not None else "—",
    )
    row(
        "Completion tokens (run)",
        "N/A",
        _fmt_tokens(float(llm.completion_tokens)) if llm and llm.completion_tokens is not None else "—",
        _fmt_tokens(float(hybrid.completion_tokens)) if hybrid and hybrid.completion_tokens is not None else "—",
    )
    row(
        "Total tokens (run)",
        "N/A",
        _fmt_tokens(float(llm.total_tokens)) if llm and llm.total_tokens is not None else "—",
        _fmt_tokens(float(hybrid.total_tokens)) if hybrid and hybrid.total_tokens is not None else "—",
    )
    row(
        "Estimated cost (run)",
        "N/A",
        _fmt_cost(llm.cost_usd) if llm else "—",
        _fmt_cost(hybrid.cost_usd) if hybrid else "—",
    )
    row(
        "Tokens per 1,000 titles",
        "N/A",
        _fmt_tokens(llm.tokens_per_1k_titles) if llm else "—",
        _fmt_tokens(hybrid.tokens_per_1k_titles) if hybrid else "—",
    )
    row(
        "Cost per 1,000 titles",
        "N/A",
        _fmt_cost(llm.cost_per_1k_titles_usd) if llm else "—",
        _fmt_cost(hybrid.cost_per_1k_titles_usd) if hybrid else "—",
    )

    if (known and known.notes) or (llm and llm.notes) or (hybrid and hybrid.notes):
        lines.extend(["", "## Notes", ""])
        if known and known.notes:
            lines.append(f"- **KnownListExtractor:** {known.notes}")
        if llm and llm.notes:
            lines.append(f"- **LLMExtractor:** {llm.notes}")
        if hybrid and hybrid.notes:
            lines.append(f"- **HybridExtractor:** {hybrid.notes}")

    lines.append("")
    Path_write = path
    with open(Path_write, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate brand extractors on a labeled config CSV.")
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to config CSV (brand list + labeled rows for evaluation).",
    )
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
        "--hybrid-only",
        action="store_true",
        help="Evaluate only HybridExtractor (known-list first, LLM fallback).",
    )
    parser.add_argument(
        "--llm-limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate LLM/Hybrid extractors on the first N config rows only (saves API cost).",
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
    parser.add_argument(
        "--rejection-list",
        action="append",
        metavar="PATH",
        help="Generic word/phrase list file used to reject spurious fuzzy matches (repeatable). "
        "Defaults to the bundled food word + ingredient lists.",
    )
    parser.add_argument(
        "--domain",
        default="food",
        metavar="NAME",
        help="Domain label substituted into the LLM prompt (default: food).",
    )
    args = parser.parse_args()

    load_dotenv()
    config_path = resolve_config_path(args.config)
    config_df = pd.read_csv(config_path, index_col=0)
    rejection_lists = args.rejection_list if args.rejection_list else list(DEFAULT_REJECTION_LISTS)
    limited_df = config_df if args.llm_limit is None else config_df.head(args.llm_limit)

    run_known = not args.llm_only and not args.hybrid_only
    run_llm = not args.known_list_only and not args.hybrid_only
    run_hybrid = args.hybrid_only

    known_stats: ExtractorStats | None = None
    llm_stats: ExtractorStats | None = None
    hybrid_stats: ExtractorStats | None = None

    if run_known:
        known_extractor = KnownListExtractor(
            config_path=config_path, rejection_list_paths=rejection_lists
        )
        known_results = evaluate_extractor("KnownListExtractor", known_extractor, config_df)
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
            llm_extractor = LLMExtractor(config_path=config_path, domain=args.domain)
            llm_results = evaluate_extractor(
                "LLMExtractor",
                llm_extractor,
                limited_df,
                show_progress=True,
            )
            llm_notes = ""
            llm_name = "LLMExtractor"
            if args.llm_limit is not None:
                llm_name = f"LLMExtractor (first {len(limited_df)} rows)"
                llm_notes = f"Evaluated on first {len(limited_df)} of {len(config_df)} config rows."
            llm_stats = results_to_stats(
                llm_name,
                llm_results,
                llm_extractor=llm_extractor,
                notes=llm_notes,
            )
            print_stats(llm_stats)

    if run_hybrid:
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            print("Skipping HybridExtractor: OPENAI_API_KEY not set in .env", file=sys.stderr)
        elif not api_key.isascii():
            print(
                "Skipping HybridExtractor: OPENAI_API_KEY contains non-ASCII characters — re-copy from OpenAI",
                file=sys.stderr,
            )
        else:
            hybrid_extractor = HybridExtractor(
                config_path=config_path,
                rejection_list_paths=rejection_lists,
                domain=args.domain,
            )
            hybrid_results = evaluate_extractor(
                "HybridExtractor",
                hybrid_extractor,
                limited_df,
                show_progress=True,
            )
            hybrid_notes = ""
            hybrid_name = "HybridExtractor"
            if args.llm_limit is not None:
                hybrid_name = f"HybridExtractor (first {len(limited_df)} rows)"
                hybrid_notes = f"Evaluated on first {len(limited_df)} of {len(config_df)} config rows."
            hybrid_stats = results_to_stats(
                hybrid_name,
                hybrid_results,
                llm_extractor=hybrid_extractor.llm,
                notes=hybrid_notes,
                llm_fallback_requests=hybrid_extractor.usage["requests"],
            )
            print_stats(hybrid_stats)

    if not args.no_stats_file and (known_stats or llm_stats or hybrid_stats):
        cache = load_stats_cache()
        if known_stats:
            cache["known_list"] = known_stats
        if llm_stats:
            cache["llm"] = llm_stats
        if hybrid_stats:
            cache["hybrid"] = hybrid_stats
        save_stats_cache(cache)
        if known_stats or llm_stats or hybrid_stats:
            write_stats_markdown(
                args.stats_output,
                cache.get("known_list"),
                cache.get("llm"),
                cache.get("hybrid"),
                config_path=str(config_path),
            )
            print(f"Wrote comparison stats to {args.stats_output}")


if __name__ == "__main__":
    main()
