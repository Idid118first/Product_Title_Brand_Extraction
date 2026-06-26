from __future__ import annotations

import csv
import html
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz

Match = tuple[int, int, str] # (char start in title, -len(brand), brand) for tie-breaking

_COMMON_WORDS_PATH = Path("1-1000.txt")
_FOOD_INGREDIENTS_PATH = Path("10000_common_food_ingredients.txt")
_generic_rejection_cache: tuple[frozenset[str], frozenset[str], frozenset[str]] | None = None


def normalize(text: str) -> str:
    """Lowercase, decode HTML entities, strip accents, and collapse whitespace."""
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"[\u2018\u2019\u02bc\u0060\u00b4]", "'", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def _generic_rejection_sets() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Common English words and food-ingredient phrases used to reject spurious fuzzy hits."""
    global _generic_rejection_cache
    if _generic_rejection_cache is None:
        with _COMMON_WORDS_PATH.open(encoding="utf-8") as f:
            common_words = frozenset(normalize(line) for line in f if line.strip())
        food_ingredients: set[str] = set()
        food_ingredient_tokens: set[str] = set()
        with _FOOD_INGREDIENTS_PATH.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                phrase = normalize(line)
                food_ingredients.add(phrase)
                food_ingredient_tokens.update(phrase.split())
        _generic_rejection_cache = (
            common_words,
            frozenset(food_ingredients),
            frozenset(food_ingredient_tokens),
        )
    return _generic_rejection_cache


def is_generic_match_window(window: str) -> bool:
    """True when the title window is a common word or food ingredient, not a real brand occurrence."""
    common_words, food_ingredients, food_ingredient_tokens = _generic_rejection_sets()
    if window in common_words or window in food_ingredients:
        return True
    if len(window.split()) == 1 and window in food_ingredient_tokens:
        return True
    return False


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


def _tokenize(product: str) -> list[str]:
    """Split a normalized title on whitespace and strip attached commas per token."""
    tokens: list[str] = []
    for piece in product.split():
        token = piece.strip(",")
        if token:
            tokens.append(token)
    return tokens


def _drop_ungrounded_at_tied_positions(candidates: list[Match], tokens: list[str]) -> list[Match]:
    """When candidates share a start position, drop ungrounded brands if a grounded one exists."""
    if not candidates:
        return candidates
    title_tokens = set(tokens)
    by_position: dict[int, list[Match]] = {}
    for match in candidates:
        by_position.setdefault(match[0], []).append(match)

    kept: list[Match] = []
    for group in by_position.values():
        grounded = [match for match in group if all(token in title_tokens for token in match[2].split())]
        kept.extend(grounded if grounded else group)
    return kept


def _brand_tokens_present(brand: str, tokens: list[str]) -> bool:
    """True when every token of a multi-token brand appears in the title token list."""
    title_counts = Counter(tokens)
    brand_token_counts = Counter(brand.split())
    return all(title_counts[token] >= count for token, count in brand_token_counts.items())


def _token_char_offset(tokens: list[str], index: int) -> int:
    """Character offset in ' '.join(tokens) where the token window at index begins."""
    if index == 0:
        return 0
    return len(" ".join(tokens[:index])) + 1


def _pick_best(matches: list[Match]) -> str | None:
    """Earliest position in the title wins; ties go to the longer brand name."""
    return min(matches)[2] if matches else None


class BrandExtractor(ABC):

    @abstractmethod
    def extract(self, product_name: str) -> str | None:
        pass


class KnownListExtractor(BrandExtractor):
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.brands: frozenset[str] | None = None
        self._display_by_normalized: dict[str, str] = {}
        if config_path is not None:
            self.create_brand_list(config_path)

    def create_brand_list(self, brands: str | Path) -> frozenset[str]:
        """Load brands from a CSV (e.g. benchmark.csv) or text file.

        CSV: expects a brands-like column (`brands` preferred, then `gold_brand`, `brands_norm`).
        Text: one brand per line and/or comma-separated per line.
        """
        path = Path(brands)
        normalized: set[str] = set()
        display_by_normalized: dict[str, str] = {}
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV has no header: {path}")
                candidate_columns = ("brands", "gold_brand", "brands_norm")
                column = next((col for col in candidate_columns if col in reader.fieldnames), None)
                if column is None:
                    raise ValueError(
                        f"CSV must contain one of {candidate_columns}; found {reader.fieldnames}"
                    )
                for row in reader:
                    cell = row.get(column)
                    if not cell:
                        continue
                    for raw_brand in cell.split(","):
                        raw_brand = raw_brand.strip()
                        brand = normalize(raw_brand)
                        if brand:
                            normalized.add(brand)
                            display_by_normalized.setdefault(brand, raw_brand)
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                for raw_brand in line.split(","):
                    raw_brand = raw_brand.strip()
                    brand = normalize(raw_brand)
                    if brand:
                        normalized.add(brand)
                        display_by_normalized.setdefault(brand, raw_brand)
        self.brands = frozenset(normalized)
        self._display_by_normalized = display_by_normalized
        return self.brands

    def _exact_matches(self, tokens: list[str]) -> list[Match]:
        """Exact match on consecutive punctuation-stripped title tokens (e.g. co., -> co.)."""
        matches: list[Match] = []
        for brand in self.brands:
            brand_tokens = brand.split()
            width = len(brand_tokens)
            for i in range(len(tokens) - width + 1):
                if tokens[i : i + width] == brand_tokens:
                    matches.append((_token_char_offset(tokens, i), -len(brand), brand))
                    break
        return matches

    def _fuzzy_matches(self, tokens: list[str]) -> list[Match]:
        matches: list[Match] = []
        for brand in self.brands:
            threshold = fuzzy_threshold_for_brand(brand)
            width = len(brand.split())
            for i in range(len(tokens) - width + 1):
                window = " ".join(tokens[i : i + width])
                if abs(len(window) - len(brand)) > 3 or window[0] != brand[0]:
                    continue
                if is_generic_match_window(window):
                    continue
                if fuzz.ratio(brand, window) >= threshold:
                    matches.append((_token_char_offset(tokens, i), -len(brand), brand))
                    break # earliest window for this brand
        return matches

    def _split_token_presence_matches(self, tokens: list[str], window_matched: set[str]) -> list[Match]:
        """Multi-word brands not found as one consecutive token window: accept when every brand
        token appears separately in the title. Tie-break position is the earliest title token
        among the brand's tokens (whichever appears first in the title, not brand order)."""
        matches: list[Match] = []
        for brand in self.brands:
            if brand in window_matched:
                continue
            if len(brand.split()) < 2:
                continue
            if not _brand_tokens_present(brand, tokens):
                continue
            brand_token_set = set(brand.split())
            for i, token in enumerate(tokens):
                if token in brand_token_set:
                    matches.append((_token_char_offset(tokens, i), -len(brand), brand))
                    break
        return matches

    def extract(self, product_name: str) -> str | None:
        """Return the best known brand in the normalized title, or None if no match.

        Exact whole-word matches are tried first. When exact finds nothing, or when it finds a
        brand but the title has multiple words, fuzzy consecutive-window matching and (for
        multi-word brands only) split-token presence matching also run. All candidates compete
        with the same tie-breakers: earliest position, then longest brand.
        """
        if self.brands is None:
            raise ValueError("call create_brand_list before extract")
        product = normalize(product_name)
        tokens = _tokenize(product)
        candidates = self._exact_matches(tokens)
        if not candidates or len(tokens) > 1:
            window_matched = {brand for _, _, brand in candidates}
            fuzzy = self._fuzzy_matches(tokens)
            window_matched |= {brand for _, _, brand in fuzzy}
            candidates = candidates + fuzzy + self._split_token_presence_matches(tokens, window_matched)
        candidates = _drop_ungrounded_at_tied_positions(candidates, tokens)
        best = _pick_best(candidates)
        if best is None:
            return None
        return self._display_by_normalized.get(best, best)

    def predict(self, product_name: str) -> str | None:
        """Alias for extract(), matching expected inference-style API."""
        return self.extract(product_name)
