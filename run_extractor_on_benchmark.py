import pandas as pd

from brand_extractor import KnownListExtractor, normalize

BENCHMARK_PATH = "benchmark.csv"
RECOVERABLE_SEGMENTS = (1, 2) # 1 = exact in title, 2 = inferable via fuzzy; 3 = brand absent


def gold_variants(brands_norm: str) -> set[str]:
    """Acceptable gold brands for a row: each comma-separated variant, normalized."""
    return {normalize(part) for part in str(brands_norm).split(",") if normalize(part)}


def evaluate(benchmark_path: str = BENCHMARK_PATH) -> None:
    benchmark = pd.read_csv(benchmark_path, index_col=0)
    extractor = KnownListExtractor(config_path=benchmark_path)

    rows = []
    for _, row in benchmark.iterrows():
        segment = int(row["match_category"])
        recoverable = segment in RECOVERABLE_SEGMENTS
        gold = gold_variants(row["brands_norm"])

        prediction = extractor.predict(row["product_name"])
        predicted = prediction is not None
        predicted_norm = normalize(prediction) if predicted else None

        if recoverable:
            correct = predicted and predicted_norm in gold # right brand recovered
        else:
            correct = not predicted # segment 3: correct means abstaining

        rows.append(
            {"segment": segment, "recoverable": recoverable, "predicted": predicted, "correct": correct}
        )

    results = pd.DataFrame(rows)
    total = len(results)

    # Accuracy: exact-match fraction overall and per segment
    overall_accuracy = results["correct"].mean()
    per_segment_accuracy = results.groupby("segment")["correct"].mean()

    # Precision: of items where a brand was returned, how many were right
    returned = results[results["predicted"]]
    precision = returned["correct"].mean() if len(returned) else float("nan")

    # Recall (coverage): of recoverable items, how many were correctly found
    recoverable = results[results["recoverable"]]
    recall = recoverable["correct"].mean() if len(recoverable) else float("nan")

    # Abstention rate: of truly-absent items, how often the extractor correctly returned nothing
    absent = results[~results["recoverable"]]
    abstention_rate = absent["correct"].mean() if len(absent) else float("nan")

    print(f"Benchmark rows: {total}")
    print(f"Brands returned (non-null): {int(results['predicted'].sum())}")
    print()
    print(f"Accuracy (overall exact-match): {overall_accuracy:.3f}")
    for segment, accuracy in per_segment_accuracy.items():
        count = int((results["segment"] == segment).sum())
        print(f"  segment {segment} (n={count}): {accuracy:.3f}")
    print()
    print(f"Precision (correct / brand returned, n={len(returned)}): {precision:.3f}")
    print(f"Recall    (correct / recoverable, n={len(recoverable)}): {recall:.3f}")
    print(f"Abstention rate (correct None / truly absent, n={len(absent)}): {abstention_rate:.3f}")


if __name__ == "__main__":
    evaluate()
