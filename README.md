# Product Title → Brand Extraction

Extract brand names from Open Food Facts product titles using two interchangeable
implementations in `brand_extractor.py`:

- **Implementation A** — `KnownListExtractor`: rule-based matching against a known brand list
- **Implementation B** — `LLMExtractor`: OpenAI chat completion with the same brand list as context

Both share the `BrandExtractor` base class and the same `extract()` / `predict()` API. See
[Progress log](#progress-log) at the bottom for weekly milestones.

## Quick start

```python
from brand_extractor import BrandExtractor, KnownListExtractor, LLMExtractor

# Implementation A — no API key required
known = KnownListExtractor(config_path="benchmark.csv")
brand = known.predict("Lay's Classic Potato Chips 8oz")  # predict() aliases extract()

# Implementation B — requires OPENAI_API_KEY in .env (or pass api_key=...)
llm = LLMExtractor(config_path="benchmark.csv")
brand = llm.extract("Lay's Classic Potato Chips 8oz")
```

**Import convention:** only classes are exported from `brand_extractor.py`. Normalization,
fuzzy thresholds, generic-word rejection, and tie-break helpers live as methods on
`BrandExtractor` so subclasses and the benchmark harness can call `BrandExtractor.normalize()`
without importing module-level functions.

Run the benchmark harness (both extractors by default):

```bash
python run_extractor_on_benchmark.py
```

Requires `OPENAI_API_KEY` in `.env` for the LLM pass. Use `--known-list-only` to skip API calls.

---

## Architecture

### `BrandExtractor` (abstract base class)

`BrandExtractor` defines the extraction contract **and** the shared matching toolkit used by
Implementation A (and referenced in Implementation B's prompt tie-break rules):

```python
class BrandExtractor(ABC):
    @abstractmethod
    def extract(self, product_name: str) -> str | None:
        pass
```

| Shared helper | Role |
|---|---|
| `normalize(text)` | Lowercase, HTML unescape, accent fold, apostrophe fold, whitespace collapse |
| `fuzzy_threshold_for_brand(brand)` | Length-tier minimum RapidFuzz ratio (0–100) |
| `is_generic_match_window(window)` | Reject common English words and food-ingredient phrases |
| `_drop_ungrounded_at_tied_positions(...)` | Prefer brands whose tokens all appear in the title |
| `_pick_best(matches)` | Earliest position wins; longer brand wins ties |

**Design intent:** keep the prediction API stable so Implementation A and B can be swapped or
compared behind the same `extract(product_name) -> str | None` contract. Callers grade on
normalized output; implementations decide how to produce it. Subclasses implement `extract()`;
callers may use `predict()` as an alias (both implementations provide it).

### `KnownListExtractor` (Implementation A)

`KnownListExtractor(BrandExtractor)` loads a known brand universe, normalizes each entry, and
searches the normalized title for the best brand match.

| Piece | Role |
|---|---|
| `create_brand_list(path)` | Load brands from a CSV or text file; normalize and store as `frozenset[str]`. |
| `_display_by_normalized` | Map normalized brand → first-seen display form (e.g. `lay's` → `Lay's`). |
| `_exact_matches` | Consecutive token-window exact match (comma-stripped tokens). |
| `_fuzzy_matches` | RapidFuzz consecutive-window near match with length thresholds and generic rejection. |
| `_split_token_presence_matches` | Multi-word brands whose tokens all appear non-adjacently in the title. |
| `extract` / `predict` | Run the pipeline, apply tie-breakers, return display name or `None`. |

**Brand list sources**

- **Text file** — one brand per line and/or comma-separated brands per line.
- **CSV** — reads the first available column among `brands`, `gold_brand`, `brands_norm`.
  Comma-separated cells expand to multiple brands. `benchmark.csv` is a typical config path
  when evaluating against the frozen benchmark.

**Return value:** the canonical display string from the brand list when a match wins; `None` when
no candidate passes the matching rules. The extractor does **not** read the `brands` field on
the input row — only the known list and the product title.

### `LLMExtractor` (Implementation B)

`LLMExtractor(BrandExtractor)` sends each product title to an OpenAI chat model with a fixed
system-style user prompt: extraction guidelines, tie-break rules (mirroring Implementation A),
and a comma-separated sample of known brands loaded from the same config path.

```python
from brand_extractor import LLMExtractor

extractor = LLMExtractor(
    config_path="benchmark.csv",  # loads brand list via KnownListExtractor helper
    model="gpt-4o-mini",          # default
    brand_list_limit=1000,        # first N sorted brands embedded in the prompt
)
brand = extractor.predict(product_name)
```

| Piece | Role |
|---|---|
| `_load_brand_list` | Reuses `KnownListExtractor.create_brand_list`, sorts brands, truncates to `brand_list_limit` |
| `extract` / `predict` | Build prompt, call OpenAI, parse brand name or `None` |
| `_parse_response` | Map `none` / `n/a` / etc. → `None`; strip quotes from model output |
| `usage` | Accumulated `prompt_tokens`, `completion_tokens`, `total_tokens`, `requests` per instance |
| `estimated_cost_usd()` | Run cost from accumulated usage at standard `gpt-4o-mini` rates |

**API setup:** set `OPENAI_API_KEY` in `.env` (loaded via `python-dotenv`). The key must be
ASCII-only (re-copy from OpenAI if paste introduced stray Unicode). A `0.6s` delay between
requests avoids TPM rate limits during full-benchmark runs.

**Prompt behavior:** the model is instructed to return only the brand name (in list format) or
`none` when no brand is present. Tie-break guidance in the prompt matches Implementation A:
earliest position, longer brand at same position, grounded over ungrounded, all match types
compete equally.

**Dependencies:** `openai`, `python-dotenv` (in addition to Implementation A deps).

---

## Known-Brand-List Matching (Implementation A — detail)

Rule-based brand extraction. Given a product title and a list of known brands, return the best
matching brand or `None` when nothing matches confidently enough.

### Normalization

Both titles and brands pass through `BrandExtractor.normalize()` before matching:

- Decode HTML entities (`html.unescape`)
- Lowercase
- Fold apostrophe variants (`'`, `'`, `ʼ`, `` ` ``, `´`) to `'`
- Strip accents (NFKD + drop combining marks)
- Collapse whitespace

Raw product names are never mutated; matching uses the normalized form only. Titles are
tokenized by splitting on whitespace and stripping attached commas per token (`co.,` → `co.`).

### Matching pipeline

Search is **brand-driven**: loop over the known brand list and test each brand against the
title — not title-word lookup in the brand set.

**1. Exact consecutive-token matching** (always runs)

- For each brand, slide a window of `N` tokens (`N` = brand word count) over the title.
- A hit requires token-for-token equality after comma-stripping (e.g. `co.,` matches `co.`).
- Multi-word brands must appear as one consecutive phrase (`pg tips`, not `pg … tips`).

**2. Fuzzy consecutive-window matching** (conditional)

Runs when exact finds nothing, **or** when exact finds a brand but the title has multiple words
(single-word titles with an exact hit skip fuzzy).

- Same sliding window as exact, scored with **RapidFuzz** `fuzz.ratio` (0–100).
- **Pre-filters:** character-length difference ≤ 3; same first letter.
- **Generic rejection:** skip windows that are a common English word (`1-1000.txt`), a food
  ingredient phrase (`10000_common_food_ingredients.txt`), or a single ingredient token.
- **Length-based thresholds** (shorter brands need higher scores):

  | Brand length (chars, no spaces) | Min `fuzz.ratio` |
  |---|---|
  | ≤ 3 | 97 |
  | 4–5 | 93 |
  | 6–8 | 92 |
  | 9–12 | 85 |
  | 13+ | 82 |

**3. Split-token presence** (multi-word brands only, conditional)

For multi-word brands **not** already matched in steps 1–2:

- Every brand token must appear somewhere in the title (duplicate counts respected).
- Tokens need not be adjacent or in brand order.
- Tie-break position = earliest title token that is any of the brand's tokens.

**4. Grounded tie-break filter** (universe guard)

When multiple candidates share the same start position, prefer brands whose tokens all appear
literally in the title over ungrounded ones (e.g. `giant eagle` over fuzzy `giant eagle inc.`
matched to `giant eagle thin`). Misspelled tokens (`putti` vs `puti`) are still allowed when
no grounded alternative exists at that position.

### Tie-breakers

All surviving candidates compete with the same rule:

1. **Earliest character position** in the normalized title wins.
2. **Longer canonical brand** wins ties (e.g. `tesco finest` over `tesco`).

Position uses character offset in the joined token string. Length is the brand from the list,
not the misspelled title window.

### Dependencies

- **rapidfuzz** — fuzzy similarity in the extractor (`fuzz.ratio`, 0–100). Project standard;
  do not substitute `difflib` or other fuzzy libraries in extraction code.
- **Word lists** — `1-1000.txt`, `10000_common_food_ingredients.txt` (generic-match rejection)
- **stdlib** — `csv`, `re`, `html`, `unicodedata`, `abc`, `collections`

Listed in `requirements.txt`.

---

## Benchmark evaluation (`run_extractor_on_benchmark.py`)

The harness evaluates **both** extractors on the frozen `benchmark.csv` (892 rows), using that
file as the brand-list config (`config_path`) and as the labeled test set. Results print to
stdout and are written to `benchmark_stats.md` as a side-by-side comparison table. Partial runs
merge into `benchmark_stats_cache.json` so `--known-list-only` and `--llm-only` do not wipe the
other extractor's cached numbers.

```bash
# Both extractors (KnownList is instant; LLM ~20 min + API cost for 892 rows)
python run_extractor_on_benchmark.py

# Rule-based only — no API key needed
python run_extractor_on_benchmark.py --known-list-only

# LLM only (merges with cached KnownList stats if present)
python run_extractor_on_benchmark.py --llm-only

# Cheaper LLM smoke test on first N rows
python run_extractor_on_benchmark.py --llm-only --llm-limit 50
```

The harness calls `extractor.predict()` on each row and grades via `BrandExtractor.normalize()`.
For LLM runs it also reports token usage, estimated API cost for the run, and extrapolated
**cost per 1,000 titles** (KnownList shows `N/A` for cost columns).

### Grading rules

- **Gold label:** each comma-separated variant in `brands_norm`, normalized. A prediction is
  correct if its normalized form equals **any** gold variant.
- **Segments 1 & 2 (recoverable):** correct = returned a brand and it matches gold.
- **Segment 3 (brand absent):** correct = returned `None` (abstained).

### Metrics reported

| Metric | Definition |
|---|---|
| **Accuracy (overall)** | Fraction of all rows graded correct. |
| **Accuracy per segment** | Same, broken out by `match_category` (1 = exact in title, 2 = fuzzy/inferable, 3 = absent). |
| **Precision** | Of rows where a brand was returned, fraction that were correct. |
| **Recall (coverage)** | Of recoverable rows (segments 1 & 2), fraction correctly found. |
| **Abstention rate** | Of truly-absent rows (segment 3), fraction where the extractor correctly returned `None`. |
| **Token usage** (LLM only) | Prompt, completion, and total tokens accumulated across API calls. |
| **Estimated cost** (LLM only) | Run cost and per-1,000-title extrapolation at `gpt-4o-mini` standard rates ($0.15/1M input, $0.60/1M output). |

### Latest results (892-row benchmark)

Full comparison table: [`benchmark_stats.md`](benchmark_stats.md).

| Metric | KnownListExtractor | LLMExtractor (`gpt-4o-mini`) |
|---|---|---|
| Brands returned | 636 | 614 |
| **Accuracy (overall)** | **0.961** | **0.889** |
| Accuracy — segment 1 (n=476) | 0.994 | 0.943 |
| Accuracy — segment 2 (n=150) | 0.853 | 0.647 |
| Accuracy — segment 3 (n=266) | 0.962 | 0.929 |
| **Precision** | **0.945** | **0.889** |
| **Recall** (n=626 recoverable) | **0.960** | **0.872** |
| **Abstention rate** (n=266 absent) | **0.962** | **0.929** |
| Total tokens (run) | N/A | 3,300,274 |
| **Cost per 1,000 titles** | N/A | **~$0.56** |

KnownListExtractor leads on overall accuracy and especially segment 2 (fuzzy/inferable cases).
LLMExtractor is competitive on segment 1 and abstention but weaker on fuzzy recovery and costs
~$0.50 per full benchmark run at current prompt size (brand list embedded per request).

Re-run the script after extractor or benchmark changes to refresh these numbers.

---

## Design choices

### Data & cleaning

- **Open Food Facts ingestion:** tab-separated CSV (`sep="\t"`, `on_bad_lines="skip"`) for the
  ~12 GB dump; keep only `product_name`, `brands`, `brands_tags` early.
- **Drop unusable rows:** no `product_name` → nothing to extract; no `brands` → no ground truth.
- **Deduplicate after normalization** on `(product_name_norm, brands_norm)` so case/accent/spacing
  variants collapse in one pass.
- **Keep raw + normalized columns:** raw for display and future LLM use; normalized for matching.

### Ground-truth segmentation (notebook)

Every branded product is categorized by how recoverable its brand is from the title:

| Segment | Meaning |
|---|---|
| **1 — exact** | Brand appears verbatim (after normalization) as consecutive tokens in the title. |
| **2 — inferable** | Brand recoverable only via fuzzy near-match (misspelling, transliteration, punctuation noise). |
| **3 — absent** | Brand not present in the title at all. |

The labeler may use the `brands` field; the extractor must not. Segment 3 rows test abstention
behavior on real "nothing to extract" cases.

Notebook fuzzy labeling uses `difflib.SequenceMatcher` (0–1 scale) with equivalent length-tier
thresholds and the same generic-word/ingredient rejection. The **extractor** uses RapidFuzz
(0–100 scale) with the same rules ported — not a library swap.

### Benchmark

- **Frozen sample** in `benchmark.csv` (~892 rows): original 500 recoverable rows (segments 1 & 2,
  `random_state=10`) + 250 segment-3 rows + **150 segment-2 oversample** (category 2 is only
  ~0.4% of the full dataset; proportional sampling yielded too few fuzzy test cases).
- **Stable indices:** saved with `index=True` so each row traces back to its position in the
  source dump.
- **Reload on rerun:** `try: pd.read_csv("benchmark.csv")` preserves row stability; segment
  columns refresh from freshly categorized data each notebook run.
- **`FORCE_ABSENT_INDICES`:** manual overrides for rows where fuzzy matching falsely mapped an
  ingredient/common word onto a brand (e.g. `aloe` → `alo`); forced to segment 3 every run.
- **Decouple curated examples:** edge-case demos are looked up from `brands_known_products`, not
  only from the sampled subset.

### Edge-case taxonomy

Labels are comma-joined on the benchmark sample; `NaN` = ordinary entry.

| Tag | When flagged |
|---|---|
| **vague** | Brand is a common English word or food ingredient/product name. |
| **misspelled** | `match_category == 2` — brand in title only via fuzzy match. |
| **brand_conflict** | Multiple brands in `brands`; every one appears as a whole word in the title. |
| **spurious_substring** | Naive substring matcher would pick the wrong embedded universe brand. |
| **unique_chars** | Non-Latin/non-ASCII after accent normalization, or a space inside one brand name. |
| **noisy** | Title longer than 60 characters (~95th percentile). |

### Extraction & library choices

- **Class-only imports** — `BrandExtractor`, `KnownListExtractor`, `LLMExtractor`; shared logic
  lives on the base class, not as module-level helpers.
- **Brand-driven search** (Implementation A) over a known list — not per-title-word dictionary lookup.
- **RapidFuzz** for all extractor fuzzy matching; faster and the project standard.
- **Display names preserved** via `_display_by_normalized` so output reads `Lay's` not `lay's`.
- **Abstention via `None`** — no brand returned when nothing matches; segment 3 grading depends on this.
- **LLM prompt mirrors A's tie-breakers** — earliest position, longer brand, grounded preference.
- **Tunable constants** (`fuzzy_threshold_for_brand`, `NOISY_LENGTH_THRESHOLD`, etc.) kept as
  named values for easy adjustment.

### Project layout

| File | Purpose |
|---|---|
| `food_data.ipynb` | Data cleaning, segmentation, edge-case labeling, benchmark creation |
| `brand_extractor.py` | `BrandExtractor` ABC + `KnownListExtractor` (A) + `LLMExtractor` (B) |
| `run_extractor_on_benchmark.py` | Dual-extractor benchmark harness and stats writer |
| `benchmark.csv` | Frozen labeled sample |
| `benchmark_stats.md` | Side-by-side accuracy and cost comparison (generated) |
| `benchmark_stats_cache.json` | Cached stats for partial benchmark runs (generated) |
| `requirements.txt` | `pandas`, `rapidfuzz`, `openai`, `python-dotenv` |
| `.env` | `OPENAI_API_KEY` for Implementation B (gitignored) |

---

## Progress log

### Week of 06/01/2026 – 06/05/2026

- Onboarding completed
- Python refresher course completed
- Pandas refresher course completed
- Environment partially set up
- Open Food Facts Dataset website visited and explored

### Week of 06/08/2026

- Dataset downloaded
- Jupyter Notebook set up
- Dataset loaded in using Pandas
- Dataset explored / preliminary cleaning done
- Preliminary sample of 500 obtained — first PR

### Week of 06/15/2026

- Update segmenting logic
- Improve edge case detection
- Report statistics regarding data (edge case occurrence proportions, segment split, etc.)
- Begin Implementation A (Known-Brand-List Matching)

### Week of 06/22/2026 – 06/26/2026

- `BrandExtractor` ABC and `KnownListExtractor` in `brand_extractor.py`
- Benchmark evaluation harness (`run_extractor_on_benchmark.py`)
- Segment-2 oversample (150 fuzzy/inferable rows) for reliable fuzzy metrics
- RapidFuzz fuzzy pass with generic/ingredient rejection and grounded tie-break
- Refactored `BrandExtractor` to hold shared normalization, fuzzy thresholds, and tie-break helpers (class-only imports)
- `LLMExtractor` (Implementation B) with OpenAI `gpt-4o-mini`, token usage tracking, and cost reporting
- Dual-extractor benchmark comparison (`benchmark_stats.md`, `benchmark_stats_cache.json`)
