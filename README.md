# Product Title → Brand Extraction

**What it does:** given a product title (e.g. `"Lay's Classic Potato Chips 8oz"`), return the
brand (`"Lay's"`), or `None` when no brand is present. Although it is developed and benchmarked on
the Open Food Facts dataset, the extractors are **domain-agnostic** — point them at any product
category (electronics, pharma, cosmetics, …) by supplying your own brand list; see
[Using it for another product category](#using-it-for-another-product-category).

Three interchangeable extractors live in `src/brand_extractor.py`, all sharing the
`BrandExtractor` base class and the same `extract()` / `predict()` API:

- `KnownListExtractor` — rule-based matching (exact → fuzzy → split-token) against a known
brand list. Fast, deterministic, no API key.
- `LLMExtractor` — OpenAI chat completion using extraction guidelines only (no brand list in
the prompt), so it generalizes without a curated list.
- `HybridExtractor` — known-list first, falling back to the LLM (with the brand list in its
prompt) only when the rule-based pass abstains or is low-confidence (cheaper than pure LLM).

See [Progress log](#progress-log) at the bottom for weekly milestones.

## Setup

**Prerequisites:** Python 3.10+ and `git`. An OpenAI API key is only needed for the LLM and hybrid
extractors — the rule-based `KnownListExtractor` runs with no key.

```bash
# 1. Clone
git clone <your-fork-or-repo-url> Product_Title_Brand_Extraction
cd Product_Title_Brand_Extraction

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install dependencies (extraction + tests)
pip install -r requirements.txt

# 4. (Optional — only for LLM / hybrid) add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env
```

Verify the install (no API key required):

```bash
pytest                                                       # run the test suite
python src/run_extractor_on_benchmark.py --config configs/benchmark.csv --known-list-only
```

Then use it in your own code:

```python
import sys; sys.path.insert(0, "src")   # or install the package / set PYTHONPATH=src
from brand_extractor import KnownListExtractor

known = KnownListExtractor(config_path="configs/benchmark.csv")
print(known.predict("Lay's Classic Potato Chips 8oz"))       # -> "Lay's"
```



## Quick start

```python
import sys; sys.path.insert(0, "src")   # make `brand_extractor` importable
from brand_extractor import KnownListExtractor, LLMExtractor, HybridExtractor

# KnownListExtractor — no API key required
known = KnownListExtractor(config_path="configs/benchmark.csv")
brand = known.predict("Lay's Classic Potato Chips 8oz")  # predict() aliases extract()

# LLMExtractor — requires OPENAI_API_KEY in .env (or pass api_key=...)
llm = LLMExtractor(config_path="configs/benchmark.csv")
brand = llm.extract("Lay's Classic Potato Chips 8oz")

# HybridExtractor — known-list first, LLM only on abstention / low confidence
hybrid = HybridExtractor(config_path="configs/benchmark.csv")
brand = hybrid.predict("Lay's Classic Potato Chips 8oz")
```

**Import convention:** only classes are exported from `brand_extractor.py`. Normalization,
fuzzy thresholds, generic-word rejection, and tie-break helpers live as methods on
`BrandExtractor` so subclasses and the benchmark harness can call `BrandExtractor.normalize()`
without importing module-level functions.

### Domain-agnostic usage

The extractors ship with **no hardcoded food/dataset assumptions** in their matching logic. Point
them at any domain by supplying the appropriate config:

```python
# Any catalog — brands from a custom CSV column, domain-specific rejection lists, domain label
known = KnownListExtractor(
    config_path="electronics_brands.csv",
    brand_column="maker",                       # default: first of brands / gold_brand / brands_norm
    rejection_list_paths=["stopwords.txt"],     # default: [] (no fuzzy rejection at all)
)

llm = LLMExtractor(
    config_path="electronics_brands.csv",
    brand_column="maker",
    domain="electronics",                       # replaces "food" in the prompt; default "food"
)

hybrid = HybridExtractor(
    config_path="electronics_brands.csv",
    brand_column="maker",
    rejection_list_paths=["stopwords.txt"],
    domain="electronics",
)
```

- `rejection_list_paths` — generic word/phrase files whose entries are never treated as brands
during fuzzy matching. **Defaults to empty**, so fuzzy rejection is opt-in per domain. The
bundled food lists (`configs/1-1000.txt`, `configs/10000_common_food_ingredients.txt`) are just
one possible choice; the benchmark harness passes them by default (override with
`--rejection-list PATH`, repeatable).
- `brand_column` — the CSV column to read brands from. Defaults to the first available of
`brands`, `gold_brand`, `brands_norm` (or use a plain text file, which is fully generic).
- `domain` — substituted wherever the LLM prompt refers to the dataset domain. Defaults to
`"food"` for backward compatibility.

**Argument validation (pydantic):** all extractor constructor arguments are validated with
pydantic models (`KnownListConfig`, `LLMConfig`, `HybridConfig`) before any work runs — bad types
or values (e.g. a negative `brand_list_limit`, a blank `domain`) raise a `ValidationError`.
`KnownListExtractor` and `HybridExtractor` additionally verify that a **CSV** config has a usable
brand column (the explicit `brand_column`, or one of `brands` / `gold_brand` / `brands_norm`).

### Using it for another product category

Extending the module to a new domain requires **no code changes** — you only edit/supply config
files and pass a couple of arguments:

1. **Provide a brand list** (`config_path`). Either:
  - a **text file**, one brand per line and/or comma-separated (fully generic), or
  - a **CSV** with a brand column named `brands`, `gold_brand`, or `brands_norm` — or any header
  name if you pass `brand_column="your_column"`.
2. **(Optional) provide a rejection list** (`rejection_list_paths`) — generic words that should
  never be treated as brands during fuzzy matching (the domain equivalent of the bundled food
   word/ingredient lists). Defaults to empty (no rejection).
3. **Set the domain label** (`domain`) so the LLM prompt reads naturally (e.g. "from the
  electronics dataset"). Only affects `LLMExtractor` / `HybridExtractor`.
4. **Call the extractor** with those config paths:
  ```python
   from brand_extractor import KnownListExtractor

   known = KnownListExtractor(
       config_path="electronics_brands.txt",
       rejection_list_paths=["electronics_stopwords.txt"],
   )
   known.predict("Anker PowerCore 10000 Portable Charger")   # -> "Anker"
  ```

To benchmark a new domain, point the harness at your labeled CSV (see
[Benchmark evaluation](#benchmark-evaluation-run_extractor_on_benchmarkpy)):

```bash
python src/run_extractor_on_benchmark.py \
  --config electronics_benchmark.csv \
  --hybrid-only --domain electronics --rejection-list electronics_stopwords.txt
```

That's the entire extension path: **new brand list in, correct brands out** — the matching logic,
normalization, thresholds, and prompt are all reused unchanged.

Run the benchmark harness (both extractors by default):

```bash
python src/run_extractor_on_benchmark.py --config configs/benchmark.csv
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


| Shared helper                             | Role                                                                        |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| `normalize(text)`                         | Lowercase, HTML unescape, accent fold, apostrophe fold, whitespace collapse |
| `fuzzy_threshold_for_brand(brand)`        | Length-tier minimum RapidFuzz ratio (0–100)                                 |
| `is_generic_match_window(window)`         | Reject common English words and food-ingredient phrases                     |
| `_drop_ungrounded_at_tied_positions(...)` | Prefer brands whose tokens all appear in the title                          |
| `_pick_best(matches)`                     | Earliest position wins; longer brand wins ties                              |


**Design intent:** keep the prediction API stable so Implementation A and B can be swapped or
compared behind the same `extract(product_name) -> str | None` contract. Callers grade on
normalized output; implementations decide how to produce it. Subclasses implement `extract()`;
callers may use `predict()` as an alias (both implementations provide it).

### `KnownListExtractor` (Implementation A)

`KnownListExtractor(BrandExtractor)` loads a known brand universe, normalizes each entry, and
searches the normalized title for the best brand match.


| Piece                                        | Role                                                                                                      |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `create_brand_list(path, brand_column=None)` | Load brands from a CSV (optionally a named column) or text file; normalize and store as `frozenset[str]`. |
| `_display_by_normalized`                     | Map normalized brand → first-seen display form (e.g. `lay's` → `Lay's`).                                  |
| `_exact_matches`                             | Consecutive token-window exact match (comma-stripped tokens).                                             |
| `_fuzzy_matches`                             | RapidFuzz consecutive-window near match with length thresholds and generic rejection.                     |
| `_split_token_presence_matches`              | Multi-word brands whose tokens all appear non-adjacently in the title.                                    |
| `extract` / `predict`                        | Run the pipeline, apply tie-breakers, return display name or `None`.                                      |


**Brand list sources**

- **Text file** — one brand per line and/or comma-separated brands per line.
- **CSV** — reads the column named by `brand_column`, or the first available among `brands`,
`gold_brand`, `brands_norm` when `brand_column` is omitted. Comma-separated cells expand to
multiple brands. `benchmark.csv` is a typical config path when evaluating against the frozen
benchmark.

**Generic-word rejection** is opt-in via `rejection_list_paths` (defaults to none), so the
extractor makes no domain assumptions unless you supply word/phrase lists.

**Return value:** the canonical display string from the brand list when a match wins; `None` when
no candidate passes the matching rules. The extractor does **not** read the `brands` field on
the input row — only the known list and the product title.

### `LLMExtractor` (Implementation B)

`LLMExtractor(BrandExtractor)` sends each product title to an OpenAI chat model with a fixed
system-style user prompt: extraction guidelines and tie-break rules (mirroring Implementation A).
By default the brand list is **not** embedded in the prompt (`include_brand_list_in_prompt=False`),
so the standalone LLM relies purely on the guidelines; `HybridExtractor` constructs its internal
`LLMExtractor` with `include_brand_list_in_prompt=True` to add the brand list as fallback context.

```python
from brand_extractor import LLMExtractor

extractor = LLMExtractor(
    config_path="benchmark.csv",           # loads brand list via KnownListExtractor helper
    model="gpt-4o-mini",                   # default
    brand_list_limit=1000,                 # first N sorted brands (only used if list embedded)
    include_brand_list_in_prompt=False,    # default; set True to embed the brand list
)
brand = extractor.predict(product_name)
```


| Piece                  | Role                                                                                                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_load_brand_list`     | Reuses `KnownListExtractor.create_brand_list`, sorts brands, truncates to `brand_list_limit` (embedded in the prompt only when `include_brand_list_in_prompt=True`) |
| `extract` / `predict`  | Build prompt, call OpenAI, parse brand name or `None`                                                                                                               |
| `_parse_response`      | Map `none` / `n/a` / etc. → `None`; strip quotes from model output                                                                                                  |
| `usage`                | Accumulated `prompt_tokens`, `completion_tokens`, `total_tokens`, `requests` per instance                                                                           |
| `estimated_cost_usd()` | Run cost from accumulated usage at standard `gpt-4o-mini` rates                                                                                                     |


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
- Fold apostrophe variants (`'`, `'`, `ʼ`, ```, `´`) to `'`
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
- **Generic rejection (opt-in):** skip windows that match any supplied `rejection_list_paths`
entry (a full phrase, or a single token of one). Disabled by default; the food benchmark passes
`configs/1-1000.txt` (common English words) and `configs/10000_common_food_ingredients.txt`.
- **Length-based thresholds** (shorter brands need higher scores):

  | Brand length (chars, no spaces) | Min `fuzz.ratio` |
  | ------------------------------- | ---------------- |
  | ≤ 3                             | 97               |
  | 4–5                             | 93               |
  | 6–8                             | 92               |
  | 9–12                            | 85               |
  | 13+                             | 82               |


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
- **Word lists (optional)** — `1-1000.txt`, `10000_common_food_ingredients.txt` for food
generic-match rejection; supplied per instance via `rejection_list_paths`, not hardcoded
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
python src/run_extractor_on_benchmark.py --config configs/benchmark.csv

# Rule-based only — no API key needed
python src/run_extractor_on_benchmark.py --config configs/benchmark.csv --known-list-only

# LLM only (merges with cached KnownList stats if present)
python src/run_extractor_on_benchmark.py --config configs/benchmark.csv --llm-only

# Hybrid only (known-list first, LLM fallback)
python src/run_extractor_on_benchmark.py --config configs/benchmark.csv --hybrid-only

# Cheaper LLM smoke test on first N rows
python src/run_extractor_on_benchmark.py --config configs/benchmark.csv --llm-only --llm-limit 50
```

The harness calls `extractor.predict()` on each row and grades via `BrandExtractor.normalize()`.
For LLM runs it also reports token usage, estimated API cost for the run, and extrapolated
**cost per 1,000 titles** (KnownList shows `N/A` for cost columns).

### Grading rules

- **Gold label:** each comma-separated variant in `brands_norm`, normalized. A prediction is
correct if it **exactly or fuzzily** matches **any** gold variant (same length-tier RapidFuzz
thresholds as the extractor).
- **Segments 1 & 2 (recoverable):** correct = returned a brand and it matches gold.
- **Segment 3 (brand absent):** correct = returned `None` (abstained).



### Metrics reported


| Metric                        | Definition                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Accuracy (overall)**        | Fraction of all rows graded correct.                                                                          |
| **Accuracy per segment**      | Same, broken out by `match_category` (1 = exact in title, 2 = fuzzy/inferable, 3 = absent).                   |
| **Precision**                 | Of rows where a brand was returned, fraction that were correct.                                               |
| **Recall (coverage)**         | Of recoverable rows (segments 1 & 2), fraction correctly found.                                               |
| **Abstention rate**           | Of truly-absent rows (segment 3), fraction where the extractor correctly returned `None`.                     |
| **Token usage** (LLM only)    | Prompt, completion, and total tokens accumulated across API calls.                                            |
| **Estimated cost** (LLM only) | Run cost and per-1,000-title extrapolation at `gpt-4o-mini` standard rates ($0.15/1M input, $0.60/1M output). |




### Latest results (892-row benchmark)

Full comparison table: `[benchmark_stats.md](benchmark_stats.md)`.


| Metric                             | KnownListExtractor | LLMExtractor (`gpt-4o-mini`) | HybridExtractor (`gpt-4o-mini`) |
| ---------------------------------- | ------------------ | ---------------------------- | ------------------------------- |
| Brands returned                    | 636                | 522                          | 652                             |
| LLM fallback calls                 | —                  | 892 / 892                    | 283 / 892                       |
| **Accuracy (overall)**             | **0.981**          | **0.776**                    | **0.953**                       |
| Accuracy — segment 1 (n=476)       | 0.994              | 0.739                        | 0.998                           |
| Accuracy — segment 2 (n=150)       | 0.973              | 0.647                        | 0.913                           |
| Accuracy — segment 3 (n=266)       | 0.962              | 0.914                        | 0.895                           |
| **Precision**                      | **0.973**          | **0.860**                    | **0.939**                       |
| **Recall** (n=626 recoverable)     | **0.989**          | **0.717**                    | **0.978**                       |
| **Abstention rate** (n=266 absent) | **0.962**          | **0.914**                    | **0.895**                       |
| Total tokens (run)                 | N/A                | 413,574                      | 1,009,313                       |
| **Cost per 1,000 titles**          | N/A                | **~$0.07**                   | **~$0.17**                      |


KnownListExtractor leads on overall accuracy and especially segment 2 (fuzzy/inferable cases).
LLMExtractor now runs **without the brand list in its prompt** — this makes each request ~~7x
cheaper (~~$0.07 per full run) but drops accuracy on segments 1–2 since the model has no brand
grounding; it stays strong on abstention (segment 3). HybridExtractor keeps the brand list in its
fallback prompt, so it recovers most of that accuracy while only paying for the 283 titles the
rule-based pass could not resolve.

**IMPORTANT:** The reason for the unexpected lower accuracy, precision, recall, and abstension rate for the hybrid approach when compared to the "Known list" approach is due to the design of the stat gathering itself. The stats were collected on a benchmark from which we built our brand universe (gold brands) which served as our known list of brands, hence all stats stand to bias towards this approach. the reality is, in production we will not have such a clean and comprehensive list of brands, so this is a bit of a one off scenario. What we should focus on is bringing the hybrid approach as close to the known list approach as possible, because this is the approach which will work well both in a control environment like ours and in prodcution.

Re-run the script after extractor or benchmark changes to refresh these numbers.

---



## Testing

Unit tests live in `tests/` and use **pytest** with **pytest-cov**. Configuration is in
`pytest.ini` (`pythonpath = src`, coverage enabled by default).

```bash
pip install -r requirements.txt      # includes pytest + pytest-cov
pytest                               # runs the suite with a coverage summary
```

The suite is organized one class per unit — `TestKnownListMatchResult`, `TestBrandExtractor`,
`TestKnownListExtractor`, `TestLLMExtractor`, `TestHybridExtractor`, `TestConfigLoading` — and
covers ~97% of `src/brand_extractor.py`. The OpenAI client is fully stubbed (see
`tests/conftest.py`), so tests run offline with no API key or network calls; hybrid routing is
tested by mocking the known-list result and the LLM.

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


| Segment           | Meaning                                                                                        |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| **1 — exact**     | Brand appears verbatim (after normalization) as consecutive tokens in the title.               |
| **2 — inferable** | Brand recoverable only via fuzzy near-match (misspelling, transliteration, punctuation noise). |
| **3 — absent**    | Brand not present in the title at all.                                                         |


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
- `FORCE_ABSENT_INDICES`**:** manual overrides for rows where fuzzy matching falsely mapped an
ingredient/common word onto a brand (e.g. `aloe` → `alo`); forced to segment 3 every run.
- **Decouple curated examples:** edge-case demos are looked up from `brands_known_products`, not
only from the sampled subset.



### Edge-case taxonomy

Labels are comma-joined on the benchmark sample; `NaN` = ordinary entry.


| Tag                    | When flagged                                                                      |
| ---------------------- | --------------------------------------------------------------------------------- |
| **vague**              | Brand is a common English word or food ingredient/product name.                   |
| **misspelled**         | `match_category == 2` — brand in title only via fuzzy match.                      |
| **brand_conflict**     | Multiple brands in `brands`; every one appears as a whole word in the title.      |
| **spurious_substring** | Naive substring matcher would pick the wrong embedded universe brand.             |
| **unique_chars**       | Non-Latin/non-ASCII after accent normalization, or a space inside one brand name. |
| **noisy**              | Title longer than 60 characters (~95th percentile).                               |




### Extraction & library choices

- **Class-only imports** — `BrandExtractor`, `KnownListExtractor`, `LLMExtractor`; shared logic
lives on the base class, not as module-level helpers.
- **Brand-driven search** (Implementation A) over a known list — not per-title-word dictionary lookup.
- **RapidFuzz** for all extractor fuzzy matching; faster and the project standard.
- **Display names preserved** via `_display_by_normalized` so output reads `Lay's` not `lay's`.
- **Abstention via** `None` — no brand returned when nothing matches; segment 3 grading depends on this.
- **LLM prompt mirrors A's tie-breakers** — earliest position, longer brand, grounded preference.
- **Tunable constants** (`fuzzy_threshold_for_brand`, `NOISY_LENGTH_THRESHOLD`, etc.) kept as
named values for easy adjustment.



### Project layout


| Path                                                              | Purpose                                                                                  |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `src/brand_extractor.py`                                          | `BrandExtractor` ABC + `KnownListExtractor` (A) + `LLMExtractor` (B) + `HybridExtractor` |
| `src/run_extractor_on_benchmark.py`                               | Multi-extractor benchmark harness and stats writer                                       |
| `tests/`                                                          | Test suite (WIP)                                                                         |
| `notebooks/food_data.ipynb`                                       | Data cleaning, segmentation, edge-case labeling, benchmark creation                      |
| `configs/benchmark.csv`                                           | Frozen labeled sample + brand universe                                                   |
| `configs/1-1000.txt`, `configs/10000_common_food_ingredients.txt` | Optional food generic-word rejection lists                                               |
| `benchmark_stats.md`                                              | Side-by-side accuracy and cost comparison (generated)                                    |
| `benchmark_stats_cache.json`                                      | Cached stats for partial benchmark runs (generated)                                      |
| `requirements.txt`                                                | `pandas`, `rapidfuzz`, `openai`, `python-dotenv`, `pydantic`, `pytest`, `pytest-cov`     |
| `.env`                                                            | `OPENAI_API_KEY` for Implementation B (gitignored)                                       |




## API



When calling the extractors, because I was not sure what was meant by tunable para,eters or how to ingest and utilize cofnig specific normalization methods, I decided to use a common normalization function for all config files. Instead of now taking a single config argument for the extractors, I take the known brand list as a txt or CSV file, alias for brand column, and any common domain specific terms (to be used to counter vague results from extraction) when it comes to Known Lists extraction. For the LLM extraction, I take the brand list file, API key, AI model, brand list cap for the prompt, brand column name, domain (to aid the LLM in context), and inclusion of the brand list in the LLM prompt (default is exclusion). For hybrid I take  the brand list file, API key, AI model, brand list cap for the prompt, brand column name, domain (to aid the LLM in context), and any common domain specific terms.

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

