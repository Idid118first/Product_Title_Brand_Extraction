# Product Title → Brand Extraction

Extract brand names from Open Food Facts product titles using a known-brand list
(Implementation A: `KnownListExtractor`). See [Progress log](#progress-log) at the bottom for
weekly milestones.

## Known-Brand-List Matching (Implementation A)

Rule-based brand extraction in `brand_extractor.py`. Given a product title and a list of known
brands, return the best matching brand or `None` when nothing matches confidently enough.

### Architecture

#### `BrandExtractor` (abstract base class)

`BrandExtractor` defines the interface every extraction implementation must satisfy. It is a
thin ABC with one required method:

```python
class BrandExtractor(ABC):
    @abstractmethod
    def extract(self, product_name: str) -> str | None:
        pass
```

**Design intent:** keep the prediction API stable so Implementation A (known-brand-list
matching) and future approaches (e.g. an LLM-based Implementation B) can be swapped or compared
behind the same `extract(product_name) -> str | None` contract. Callers grade on normalized
output; implementations decide how to produce it.

#### `KnownListExtractor`

`KnownListExtractor(BrandExtractor)` is Implementation A. It loads a known brand universe,
normalizes each entry, and searches the normalized title for the best brand match.

```python
from brand_extractor import KnownListExtractor

extractor = KnownListExtractor(config_path="benchmark.csv")
brand = extractor.predict("Lay's Classic Potato Chips 8oz")  # predict() aliases extract()
```

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

### Normalization

Both titles and brands pass through `normalize()` before matching:

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

The harness runs `KnownListExtractor.predict()` on every row of the frozen `benchmark.csv`,
using that file both as the brand list (`config_path`) and as the labeled test set.

```bash
python run_extractor_on_benchmark.py
```

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

### Latest results (892-row benchmark)

| Metric | Value |
|---|---|
| Benchmark rows | 892 |
| Brands returned (non-null) | 636 |
| **Accuracy (overall)** | **0.961** |
| Accuracy — segment 1 (n=476) | 0.994 |
| Accuracy — segment 2 (n=150) | 0.853 |
| Accuracy — segment 3 (n=266) | 0.962 |
| **Precision** (n=636 returned) | **0.945** |
| **Recall** (n=626 recoverable) | **0.960** |
| **Abstention rate** (n=266 absent) | **0.962** |

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

- **Brand-driven search** over a known list — not per-title-word dictionary lookup.
- **RapidFuzz** for all extractor fuzzy matching; faster and the project standard.
- **Display names preserved** via `_display_by_normalized` so output reads `Lay's` not `lay's`.
- **Abstention via `None`** — no brand returned when nothing matches; segment 3 grading depends on this.
- **Tunable constants** (`fuzzy_threshold_for_brand`, `NOISY_LENGTH_THRESHOLD`, etc.) kept as
  named values for easy adjustment.

### Project layout

| File | Purpose |
|---|---|
| `food_data.ipynb` | Data cleaning, segmentation, edge-case labeling, benchmark creation |
| `brand_extractor.py` | `BrandExtractor` ABC + `KnownListExtractor` (Implementation A) |
| `run_extractor_on_benchmark.py` | Full-benchmark evaluation harness |
| `benchmark.csv` | Frozen labeled sample |

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
