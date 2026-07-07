from __future__ import annotations

import csv
import html
import os
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
# Fuzzy wins within this many points above the minimum threshold are treated as low-confidence.
FUZZY_LOW_CONFIDENCE_MARGIN = 5.0
# Only these minimum threshold tiers (9-12 and 13+ char brands) may fuzzy-fallback to the LLM.
FUZZY_LLM_FALLBACK_THRESHOLDS = frozenset({85.0, 82.0})


@dataclass(frozen=True)
class KnownListMatchResult:
    """Outcome of a known-list pass, including metadata for hybrid fallback decisions."""

    brand: str | None
    match_type: str  # exact, fuzzy, split_token, none
    fuzzy_score: float | None = None
    fuzzy_threshold: float | None = None

    @property
    def is_low_confidence(self) -> bool:
        if self.brand is None:
            return False
        if self.match_type == "split_token":
            return True
        if (
            self.match_type == "fuzzy"
            and self.fuzzy_score is not None
            and self.fuzzy_threshold is not None
            and self.fuzzy_score < self.fuzzy_threshold + FUZZY_LOW_CONFIDENCE_MARGIN
        ):
            return True
        return False

    def triggers_hybrid_llm_fuzzy_fallback(self) -> bool:
        """True for weak fuzzy wins on the two lowest threshold tiers only (85, 82)."""
        return (
            self.match_type == "fuzzy"
            and self.is_low_confidence
            and self.fuzzy_threshold in FUZZY_LLM_FALLBACK_THRESHOLDS
        )


@dataclass(frozen=True)
class _Candidate:
    start: int
    neg_len: int
    brand: str
    match_type: str
    fuzzy_score: float | None = None
    fuzzy_threshold: float | None = None

    @property
    def sort_key(self) -> tuple[int, int, str]:
        return (self.start, self.neg_len, self.brand)


def resolve_config_path(path: str | Path) -> Path:
    """Resolve a config path relative to cwd, then the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    from_root = PROJECT_ROOT / candidate
    if from_root.exists():
        return from_root
    return candidate


# Default CSV columns searched for brands when no explicit brand_column is given.
BRAND_COLUMN_CANDIDATES = ("brands", "gold_brand", "brands_norm")


def _require_brand_column(config_path: Path, brand_column: str | None) -> None:
    """Validate that a CSV config has a usable brand column.

    Only enforced for existing CSV files: missing files are left to raise `FileNotFoundError`
    downstream, and text brand lists have no columns to check. Raises `ValueError` (surfaced as a
    pydantic validation error) when the required brand column is absent.
    """
    path = resolve_config_path(config_path)
    if not path.is_file() or path.suffix.lower() != ".csv":
        return
    with path.open(encoding="utf-8", newline="") as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    if brand_column is not None:
        if brand_column not in fieldnames:
            raise ValueError(f"CSV missing brand column {brand_column!r}; found {fieldnames}")
    elif not any(col in fieldnames for col in BRAND_COLUMN_CANDIDATES):
        raise ValueError(
            f"CSV must contain one of {BRAND_COLUMN_CANDIDATES} (or pass brand_column); "
            f"found {fieldnames}"
        )


class KnownListConfig(BaseModel):
    """Validated arguments for KnownListExtractor (also enforces a brand column on CSV configs)."""

    model_config = ConfigDict(protected_namespaces=())

    config_path: Path
    brand_column: str | None = None
    rejection_list_paths: list[Path] | None = None

    @model_validator(mode="after")
    def _check_brand_column(self) -> "KnownListConfig":
        _require_brand_column(self.config_path, self.brand_column)
        return self


class LLMConfig(BaseModel):
    """Validated arguments for LLMExtractor."""

    model_config = ConfigDict(protected_namespaces=())

    config_path: Path
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    brand_list_limit: int = Field(default=1000, ge=0)
    brand_column: str | None = None
    domain: str = "food"
    include_brand_list_in_prompt: bool = False

    @field_validator("model", "domain")
    @classmethod
    def _reject_blank(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return value


class HybridConfig(BaseModel):
    """Validated arguments for HybridExtractor (also enforces a brand column on CSV configs)."""

    model_config = ConfigDict(protected_namespaces=())

    config_path: Path
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    brand_list_limit: int = Field(default=1000, ge=0)
    brand_column: str | None = None
    rejection_list_paths: list[Path] | None = None
    domain: str = "food"

    @field_validator("model", "domain")
    @classmethod
    def _reject_blank(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return value

    @model_validator(mode="after")
    def _check_brand_column(self) -> "HybridConfig":
        _require_brand_column(self.config_path, self.brand_column)
        return self


class BrandExtractor(ABC):
    """Base class for product-title → brand extractors.

    Shared normalization, tokenization, fuzzy thresholds, and matching helpers live here so
    any subclass (e.g. KnownListExtractor) exposes the full toolkit via import of the class alone.
    """

    Match = tuple[int, int, str]  # (char start in title, -len(brand), brand) for tie-breaking

    # Generic-match rejection sets are domain-specific and default to empty (no rejection).
    # Populate them per instance via the `rejection_list_paths` constructor argument.
    _rejection_phrases: frozenset[str] = frozenset()
    _rejection_tokens: frozenset[str] = frozenset()

    @staticmethod
    def normalize(text: str) -> str:
        """Lowercase, decode HTML entities, strip accents, and collapse whitespace."""
        text = html.unescape(text)
        text = text.lower()
        text = re.sub(r"[\u2018\u2019\u02bc\u0060\u00b4]", "'", text)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in text if not unicodedata.combining(char))
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _build_rejection_sets(
        paths: Sequence[str | Path] | None,
    ) -> tuple[frozenset[str], frozenset[str]]:
        """Load rejection phrases (and their individual tokens) from one or more list files.

        Each file holds one phrase per line (blank lines ignored), normalized the same way as
        titles and brands. Returns (phrases, tokens); tokens are the individual words of every
        phrase and reject single-token windows. Domain-agnostic: pass any word/phrase lists
        appropriate to the catalog (e.g. common words, ingredient names), or none to disable.
        """
        if not paths:
            return frozenset(), frozenset()
        phrases: set[str] = set()
        tokens: set[str] = set()
        for path in paths:
            resolved = resolve_config_path(path)
            if not resolved.is_file():
                raise FileNotFoundError(f"rejection list not found: {resolved}")
            with resolved.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    phrase = BrandExtractor.normalize(line)
                    if not phrase:
                        continue
                    phrases.add(phrase)
                    tokens.update(phrase.split())
        return frozenset(phrases), frozenset(tokens)

    def is_generic_match_window(self, window: str) -> bool:
        """True when a fuzzy window is a configured generic phrase/token, not a real brand.

        Always False when no rejection lists were supplied (the default), keeping the extractor
        domain-agnostic until a caller opts into rejection lists via `rejection_list_paths`.
        """
        if window in self._rejection_phrases:
            return True
        if len(window.split()) == 1 and window in self._rejection_tokens:
            return True
        return False

    @staticmethod
    def fuzzy_threshold_for_brand(brand: str) -> float:
        """Minimum rapidfuzz ratio (0-100) for a near match; shorter brands require a higher threshold."""
        length = len(brand.replace(" ", ""))
        if length <= 3:
            return 97.0
        if length <= 5:
            return 93.0
        if length <= 8:
            return 92.0
        if length <= 12:
            return 85.0
        return 82.0

    @classmethod
    def matches_gold_variant(cls, prediction: str, gold_variant: str) -> bool:
        """True when prediction equals or fuzzy-matches a single gold variant (grading)."""
        pred = cls.normalize(prediction)
        gold = cls.normalize(gold_variant)
        if not pred or not gold:
            return False
        if pred == gold:
            return True
        if pred[0] != gold[0] or abs(len(pred) - len(gold)) > 3:
            return False
        threshold = cls.fuzzy_threshold_for_brand(gold)
        return fuzz.ratio(pred, gold) >= threshold

    @classmethod
    def matches_any_gold_variant(cls, prediction: str, gold_variants: set[str]) -> bool:
        """True when prediction exactly or fuzzily matches any normalized gold variant."""
        return any(cls.matches_gold_variant(prediction, gold) for gold in gold_variants)

    @staticmethod
    def _tokenize(product: str) -> list[str]:
        """Split a normalized title on whitespace and strip attached commas per token."""
        tokens: list[str] = []
        for piece in product.split():
            token = piece.strip(",")
            if token:
                tokens.append(token)
        return tokens

    @staticmethod
    def _drop_ungrounded_at_tied_positions(
        candidates: list[Match], tokens: list[str]
    ) -> list[Match]:
        """When candidates share a start position, drop ungrounded brands if a grounded one exists."""
        if not candidates:
            return candidates
        title_tokens = set(tokens)
        by_position: dict[int, list[BrandExtractor.Match]] = {}
        for match in candidates:
            by_position.setdefault(match[0], []).append(match)

        kept: list[BrandExtractor.Match] = []
        for group in by_position.values():
            grounded = [
                match for match in group if all(token in title_tokens for token in match[2].split())
            ]
            kept.extend(grounded if grounded else group)
        return kept

    @staticmethod
    def _brand_tokens_present(brand: str, tokens: list[str]) -> bool:
        """True when every token of a multi-token brand appears in the title token list."""
        title_counts = Counter(tokens)
        brand_token_counts = Counter(brand.split())
        return all(title_counts[token] >= count for token, count in brand_token_counts.items())

    @staticmethod
    def _token_char_offset(tokens: list[str], index: int) -> int:
        """Character offset in ' '.join(tokens) where the token window at index begins."""
        if index == 0:
            return 0
        return len(" ".join(tokens[:index])) + 1

    @staticmethod
    def _pick_best(matches: list[Match]) -> str | None:
        """Earliest position in the title wins; ties go to the longer brand name."""
        return min(matches)[2] if matches else None

    @abstractmethod
    def extract(self, product_name: str) -> str | None:
        pass


class KnownListExtractor(BrandExtractor):
    def __init__(
        self,
        config_path: str | Path,
        brand_column: str | None = None,
        rejection_list_paths: Sequence[str | Path] | None = None,
    ) -> None:
        cfg = KnownListConfig(
            config_path=config_path,
            brand_column=brand_column,
            rejection_list_paths=rejection_list_paths,
        )
        self.config_path: Path | None = None
        self.brands: frozenset[str] | None = None
        self._display_by_normalized: dict[str, str] = {}
        self._rejection_phrases, self._rejection_tokens = self._build_rejection_sets(
            cfg.rejection_list_paths
        )
        self.create_brand_list(cfg.config_path, cfg.brand_column)

    def create_brand_list(
        self, brands: str | Path, brand_column: str | None = None
    ) -> frozenset[str]:
        """Load brands from a CSV (e.g. configs/benchmark.csv) or text file.

        CSV: reads `brand_column` when given, otherwise the first available among the defaults
        (`brands`, then `gold_brand`, `brands_norm`). Pass `brand_column` for any other header.
        Text: one brand per line and/or comma-separated per line.
        """
        path = resolve_config_path(brands)
        if not path.is_file():
            raise FileNotFoundError(f"brand config not found: {path}")
        self.config_path = path
        normalized: set[str] = set()
        display_by_normalized: dict[str, str] = {}
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV has no header: {path}")
                if brand_column is not None:
                    if brand_column not in reader.fieldnames:
                        raise ValueError(
                            f"CSV missing brand column {brand_column!r}; found {reader.fieldnames}"
                        )
                    column = brand_column
                else:
                    column = next(
                        (col for col in BRAND_COLUMN_CANDIDATES if col in reader.fieldnames), None
                    )
                    if column is None:
                        raise ValueError(
                            f"CSV must contain one of {BRAND_COLUMN_CANDIDATES} (or pass brand_column); "
                            f"found {reader.fieldnames}"
                        )
                for row in reader:
                    cell = row.get(column)
                    if not cell:
                        continue
                    for raw_brand in cell.split(","):
                        raw_brand = raw_brand.strip()
                        brand = self.normalize(raw_brand)
                        if brand:
                            normalized.add(brand)
                            display_by_normalized.setdefault(brand, raw_brand)
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                for raw_brand in line.split(","):
                    raw_brand = raw_brand.strip()
                    brand = self.normalize(raw_brand)
                    if brand:
                        normalized.add(brand)
                        display_by_normalized.setdefault(brand, raw_brand)
        self.brands = frozenset(normalized)
        self._display_by_normalized = display_by_normalized
        return self.brands

    def _fuzzy_matches(self, tokens: list[str]) -> list[_Candidate]:
        matches: list[_Candidate] = []
        for brand in self.brands:
            threshold = self.fuzzy_threshold_for_brand(brand)
            width = len(brand.split())
            for i in range(len(tokens) - width + 1):
                window = " ".join(tokens[i : i + width])
                if abs(len(window) - len(brand)) > 3 or window[0] != brand[0]:
                    continue
                if self.is_generic_match_window(window):
                    continue
                score = fuzz.ratio(brand, window)
                if score >= threshold:
                    matches.append(
                        _Candidate(
                            self._token_char_offset(tokens, i),
                            -len(brand),
                            brand,
                            "fuzzy",
                            fuzzy_score=score,
                            fuzzy_threshold=threshold,
                        )
                    )
                    break
        return matches

    def _split_token_presence_matches(
        self, tokens: list[str], window_matched: set[str]
    ) -> list[_Candidate]:
        """Multi-word brands not found as one consecutive token window: accept when every brand
        token appears separately in the title. Tie-break position is the earliest title token
        among the brand's tokens (whichever appears first in the title, not brand order)."""
        matches: list[_Candidate] = []
        for brand in self.brands:
            if brand in window_matched:
                continue
            if len(brand.split()) < 2:
                continue
            if not self._brand_tokens_present(brand, tokens):
                continue
            brand_token_set = set(brand.split())
            for i, token in enumerate(tokens):
                if token in brand_token_set:
                    matches.append(
                        _Candidate(
                            self._token_char_offset(tokens, i),
                            -len(brand),
                            brand,
                            "split_token",
                        )
                    )
                    break
        return matches

    def _exact_match_candidates(self, tokens: list[str]) -> list[_Candidate]:
        matches: list[_Candidate] = []
        for brand in self.brands:
            brand_tokens = brand.split()
            width = len(brand_tokens)
            for i in range(len(tokens) - width + 1):
                if tokens[i : i + width] == brand_tokens:
                    matches.append(
                        _Candidate(
                            self._token_char_offset(tokens, i),
                            -len(brand),
                            brand,
                            "exact",
                        )
                    )
                    break
        return matches

    @staticmethod
    def _pick_best_candidate(candidates: list[_Candidate]) -> _Candidate | None:
        return min(candidates, key=lambda candidate: candidate.sort_key) if candidates else None

    @staticmethod
    def _drop_ungrounded_candidates(
        candidates: list[_Candidate], tokens: list[str]
    ) -> list[_Candidate]:
        if not candidates:
            return candidates
        title_tokens = set(tokens)
        by_position: dict[int, list[_Candidate]] = {}
        for candidate in candidates:
            by_position.setdefault(candidate.start, []).append(candidate)

        kept: list[_Candidate] = []
        for group in by_position.values():
            grounded = [
                candidate
                for candidate in group
                if all(token in title_tokens for token in candidate.brand.split())
            ]
            kept.extend(grounded if grounded else group)
        return kept

    def match(self, product_name: str) -> KnownListMatchResult:
        """Run the known-list pipeline and return the best match with confidence metadata."""
        if self.brands is None:
            raise ValueError("call create_brand_list before match")
        product = self.normalize(product_name)
        tokens = self._tokenize(product)
        candidates = self._exact_match_candidates(tokens)
        if not candidates or len(tokens) > 1:
            window_matched = {candidate.brand for candidate in candidates}
            fuzzy = self._fuzzy_matches(tokens)
            window_matched |= {candidate.brand for candidate in fuzzy}
            candidates = candidates + fuzzy + self._split_token_presence_matches(tokens, window_matched)
        candidates = self._drop_ungrounded_candidates(candidates, tokens)
        best = self._pick_best_candidate(candidates)
        if best is None:
            return KnownListMatchResult(brand=None, match_type="none")
        display = self._display_by_normalized.get(best.brand, best.brand)
        return KnownListMatchResult(
            brand=display,
            match_type=best.match_type,
            fuzzy_score=best.fuzzy_score,
            fuzzy_threshold=best.fuzzy_threshold,
        )

    def extract(self, product_name: str) -> str | None:
        """Return the best known brand in the normalized title, or None if no match.

        Exact whole-word matches are tried first. When exact finds nothing, or when it finds a
        brand but the title has multiple words, fuzzy consecutive-window matching and (for
        multi-word brands only) split-token presence matching also run. All candidates compete
        with the same tie-breakers: earliest position, then longest brand.
        """
        return self.match(product_name).brand

    def predict(self, product_name: str) -> str | None:
        """Alias for extract(), matching expected inference-style API."""
        return self.extract(product_name)


class LLMExtractor(BrandExtractor):
    _NONE_RESPONSES = frozenset({"none", "n/a", "na", "null", "unknown"})
    _REQUEST_DELAY_SEC = 0.6  # stay under OpenAI TPM limits when benchmarking many rows
    INPUT_USD_PER_1M_TOKENS = 0.15  # gpt-4o-mini standard API (see OpenAI pricing)
    OUTPUT_USD_PER_1M_TOKENS = 0.60

    def __init__(
        self,
        config_path: str | Path,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        brand_list_limit: int = 1000,
        brand_column: str | None = None,
        domain: str = "food",
        include_brand_list_in_prompt: bool = False,
    ) -> None:
        cfg = LLMConfig(
            config_path=config_path,
            api_key=api_key,
            model=model,
            brand_list_limit=brand_list_limit,
            brand_column=brand_column,
            domain=domain,
            include_brand_list_in_prompt=include_brand_list_in_prompt,
        )
        load_dotenv()
        key = (cfg.api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        if not key.isascii():
            raise ValueError("OPENAI_API_KEY contains non-ASCII characters — re-copy from OpenAI")
        self.client = OpenAI(api_key=key)
        self.model = cfg.model
        self.domain = cfg.domain
        self._include_brand_list = cfg.include_brand_list_in_prompt
        self.config_path: Path | None = None
        self._brand_list_text = ""
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }
        self._load_brand_list(cfg.config_path, cfg.brand_list_limit, cfg.brand_column)

    def estimated_cost_usd(self) -> float:
        """Estimated API spend from accumulated usage (standard gpt-4o-mini rates)."""
        return (
            self.usage["prompt_tokens"] / 1_000_000 * self.INPUT_USD_PER_1M_TOKENS
            + self.usage["completion_tokens"] / 1_000_000 * self.OUTPUT_USD_PER_1M_TOKENS
        )

    def _load_brand_list(
        self, path: str | Path, limit: int, brand_column: str | None = None
    ) -> None:
        resolved = resolve_config_path(path)
        self.config_path = resolved
        helper = KnownListExtractor(config_path=resolved, brand_column=brand_column)
        brands = sorted(helper.brands or ())
        if limit > 0:
            brands = brands[:limit]
        self._brand_list_text = ", ".join(brands)

    def _parse_response(self, text: str | None) -> str | None:
        if not text:
            return None
        cleaned = text.strip().strip("\"'")
        if self.normalize(cleaned) in self._NONE_RESPONSES:
            return None
        return cleaned

    def _build_prompt(self, product_name: str, fallback_context: str | None = None) -> str:
        lines = [
            f"Your task is to extract the brand from the following {self.domain} product title: {product_name}.",
            "There are a few guidelines to follow:",
            "- Be careful with generic words, items, products, or ingredients present in the title. "
            "Do not assume a common word is a brand just because a similar brand name exists "
            "(for example, chocolate is an ingredient, not necessarily a brand).",
            "- Misspellings are okay, to some extent. If you cannot pick up a direct match, "
            "you can try a misspelling of a potential brand name. For shorter brand names, be "
            "stricter about how much misspelling is acceptable; for longer brand names, be less "
            "strict as there is more room for error.",
        ]
        if self._include_brand_list:
            lines.append(
                f"- Consult this brand list for possible brand names: {self._brand_list_text}. "
                f"This is a sample list of known brand names from the {self.domain} dataset."
            )
        lines.extend(
            [
                "- Brand names in the list are normalized: special characters stripped, lowercased, "
                "and simplified. If a title variant looks slightly different for those reasons, it "
                "may still be the same brand.",
                "- Not every product title contains a brand. If you do not see a brand name or "
                "something close to it in the title, do not invent one; reply with only none.",
                "- Reply with only the brand name, or none if no brand is present in the title.",
                "- When comparing or extracting, normalize the title the same way: lowercase, strip "
                "special characters (smart apostrophes, accent marks, HTML entities like &amp;, extra "
                "whitespace, commas attached to words like Co.,), but keep meaningful brand "
                "punctuation such as ampersands and hyphens (M&M, Coca-Cola).",
                "- Return the brand name in the same format as it appears in the brand list.",
                "",
                "If multiple brands could apply, use these tie-breakers:",
                "- Earliest position wins: prefer the brand that starts leftmost in the title.",
                "- Longer brand wins ties at the same position (e.g. tesco finest over tesco).",
                "- Grounded over ungrounded at the same position: every word of the brand must appear "
                "in the title (e.g. prefer giant eagle over giant eagle inc. when inc is missing).",
                "- Prefer an exact match over a fuzzy near-match.",
                "",
                "Position means:",
                "- Exact or fuzzy consecutive match: where the brand word sequence starts in the title.",
                "- Split-token match: the earliest title word that belongs to that brand.",
            ]
        )
        if fallback_context:
            lines.extend(
                [
                    "",
                    "Why you are being called (hybrid fallback):",
                    fallback_context,
                ]
            )
        return "\n".join(lines)

    def extract(
        self, product_name: str, *, fallback_context: str | None = None
    ) -> str | None:
        prompt = self._build_prompt(product_name, fallback_context)
        time.sleep(self._REQUEST_DELAY_SEC)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        if response.usage is not None:
            self.usage["prompt_tokens"] += response.usage.prompt_tokens
            self.usage["completion_tokens"] += response.usage.completion_tokens
            self.usage["total_tokens"] += response.usage.total_tokens
            self.usage["requests"] += 1
        return self._parse_response(response.choices[0].message.content)

    def predict(
        self, product_name: str, *, fallback_context: str | None = None
    ) -> str | None:
        return self.extract(product_name, fallback_context=fallback_context)


class HybridExtractor(BrandExtractor):
    """Known-list first, LLM fallback on abstention or low-confidence matches.

    Calls the LLM when the rule-based pass abstains, when the winning match used split-token
    presence, or when a fuzzy win is weak and on a bottom-tier threshold (85 or 82 only).
    Short-brand tiers (97/93/92) keep their match without LLM fallback.
    """

    def __init__(
        self,
        config_path: str | Path,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        brand_list_limit: int = 1000,
        brand_column: str | None = None,
        rejection_list_paths: Sequence[str | Path] | None = None,
        domain: str = "food",
    ) -> None:
        cfg = HybridConfig(
            config_path=config_path,
            api_key=api_key,
            model=model,
            brand_list_limit=brand_list_limit,
            brand_column=brand_column,
            rejection_list_paths=rejection_list_paths,
            domain=domain,
        )
        self.config_path = resolve_config_path(cfg.config_path)
        self.known_list = KnownListExtractor(
            config_path=self.config_path,
            brand_column=cfg.brand_column,
            rejection_list_paths=cfg.rejection_list_paths,
        )
        self.llm = LLMExtractor(
            api_key=cfg.api_key,
            config_path=self.config_path,
            model=cfg.model,
            brand_list_limit=cfg.brand_list_limit,
            brand_column=cfg.brand_column,
            domain=cfg.domain,
            include_brand_list_in_prompt=True,
        )

    @staticmethod
    def _fallback_context(result: KnownListMatchResult) -> str | None:
        if result.brand is None:
            return (
                "Rule-based matching found no brand in this title. The configured brand list "
                "may be incomplete in production, so look carefully for any brand present in "
                "the title. If no brand is clearly present, reply none."
            )
        if result.match_type == "split_token":
            return (
                f"Rule-based matching proposed {result.brand!r} using split-token matching "
                "(the brand words appear in the title but not as one consecutive phrase). "
                "Confirm this brand, correct it, or reply none if no brand is truly present."
            )
        if result.triggers_hybrid_llm_fuzzy_fallback():
            return (
                f"Rule-based matching proposed {result.brand!r} using a weak fuzzy near-match "
                f"(score {result.fuzzy_score:.0f}, minimum {result.fuzzy_threshold:.0f}). "
                "Confirm this brand, correct it, or reply none if no brand is truly present."
            )
        return None

    @staticmethod
    def _needs_llm(result: KnownListMatchResult) -> bool:
        if result.brand is None:
            return True
        if result.match_type == "split_token":
            return True
        return result.triggers_hybrid_llm_fuzzy_fallback()

    def extract(self, product_name: str) -> str | None:
        result = self.known_list.match(product_name)
        if not self._needs_llm(result):
            return result.brand
        return self.llm.extract(
            product_name,
            fallback_context=self._fallback_context(result),
        )

    def predict(self, product_name: str) -> str | None:
        return self.extract(product_name)

    @property
    def usage(self) -> dict[str, int]:
        """LLM token usage (only incremented on hybrid fallback calls)."""
        return self.llm.usage

    def estimated_cost_usd(self) -> float:
        return self.llm.estimated_cost_usd()