from __future__ import annotations

import csv
import html
import os
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz


class BrandExtractor(ABC):
    """Base class for product-title → brand extractors.

    Shared normalization, tokenization, fuzzy thresholds, and matching helpers live here so
    any subclass (e.g. KnownListExtractor) exposes the full toolkit via import of the class alone.
    """

    Match = tuple[int, int, str]  # (char start in title, -len(brand), brand) for tie-breaking

    _common_words_path = Path("1-1000.txt")
    _food_ingredients_path = Path("10000_common_food_ingredients.txt")
    _generic_rejection_cache: tuple[frozenset[str], frozenset[str], frozenset[str]] | None = None

    @staticmethod
    def normalize(text: str) -> str:
        """Lowercase, decode HTML entities, strip accents, and collapse whitespace."""
        text = html.unescape(text)
        text = text.lower()
        text = re.sub(r"[\u2018\u2019\u02bc\u0060\u00b4]", "'", text)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in text if not unicodedata.combining(char))
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _generic_rejection_sets(cls) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
        """Common English words and food-ingredient phrases used to reject spurious fuzzy hits."""
        if cls._generic_rejection_cache is None:
            with cls._common_words_path.open(encoding="utf-8") as f:
                common_words = frozenset(cls.normalize(line) for line in f if line.strip())
            food_ingredients: set[str] = set()
            food_ingredient_tokens: set[str] = set()
            with cls._food_ingredients_path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    phrase = cls.normalize(line)
                    food_ingredients.add(phrase)
                    food_ingredient_tokens.update(phrase.split())
            cls._generic_rejection_cache = (
                common_words,
                frozenset(food_ingredients),
                frozenset(food_ingredient_tokens),
            )
        return cls._generic_rejection_cache

    @classmethod
    def is_generic_match_window(cls, window: str) -> bool:
        """True when the title window is a common word or food ingredient, not a real brand occurrence."""
        common_words, food_ingredients, food_ingredient_tokens = cls._generic_rejection_sets()
        if window in common_words or window in food_ingredients:
            return True
        if len(window.split()) == 1 and window in food_ingredient_tokens:
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

    def _exact_matches(self, tokens: list[str]) -> list[BrandExtractor.Match]:
        """Exact match on consecutive punctuation-stripped title tokens (e.g. co., -> co.)."""
        matches: list[BrandExtractor.Match] = []
        for brand in self.brands:
            brand_tokens = brand.split()
            width = len(brand_tokens)
            for i in range(len(tokens) - width + 1):
                if tokens[i : i + width] == brand_tokens:
                    matches.append((self._token_char_offset(tokens, i), -len(brand), brand))
                    break
        return matches

    def _fuzzy_matches(self, tokens: list[str]) -> list[BrandExtractor.Match]:
        matches: list[BrandExtractor.Match] = []
        for brand in self.brands:
            threshold = self.fuzzy_threshold_for_brand(brand)
            width = len(brand.split())
            for i in range(len(tokens) - width + 1):
                window = " ".join(tokens[i : i + width])
                if abs(len(window) - len(brand)) > 3 or window[0] != brand[0]:
                    continue
                if self.is_generic_match_window(window):
                    continue
                if fuzz.ratio(brand, window) >= threshold:
                    matches.append((self._token_char_offset(tokens, i), -len(brand), brand))
                    break  # earliest window for this brand
        return matches

    def _split_token_presence_matches(
        self, tokens: list[str], window_matched: set[str]
    ) -> list[BrandExtractor.Match]:
        """Multi-word brands not found as one consecutive token window: accept when every brand
        token appears separately in the title. Tie-break position is the earliest title token
        among the brand's tokens (whichever appears first in the title, not brand order)."""
        matches: list[BrandExtractor.Match] = []
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
                    matches.append((self._token_char_offset(tokens, i), -len(brand), brand))
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
        product = self.normalize(product_name)
        tokens = self._tokenize(product)
        candidates = self._exact_matches(tokens)
        if not candidates or len(tokens) > 1:
            window_matched = {brand for _, _, brand in candidates}
            fuzzy = self._fuzzy_matches(tokens)
            window_matched |= {brand for _, _, brand in fuzzy}
            candidates = candidates + fuzzy + self._split_token_presence_matches(tokens, window_matched)
        candidates = self._drop_ungrounded_at_tied_positions(candidates, tokens)
        best = self._pick_best(candidates)
        if best is None:
            return None
        return self._display_by_normalized.get(best, best)

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
        api_key: str | None = None,
        config_path: str | Path | None = None,
        model: str = "gpt-4o-mini",
        brand_list_limit: int = 1000,
    ) -> None:
        load_dotenv()
        key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        if not key.isascii():
            raise ValueError("OPENAI_API_KEY contains non-ASCII characters — re-copy from OpenAI")
        self.client = OpenAI(api_key=key)
        self.model = model
        self._brand_list_text = ""
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }
        if config_path is not None:
            self._load_brand_list(config_path, brand_list_limit)

    def estimated_cost_usd(self) -> float:
        """Estimated API spend from accumulated usage (standard gpt-4o-mini rates)."""
        return (
            self.usage["prompt_tokens"] / 1_000_000 * self.INPUT_USD_PER_1M_TOKENS
            + self.usage["completion_tokens"] / 1_000_000 * self.OUTPUT_USD_PER_1M_TOKENS
        )

    def _load_brand_list(self, path: str | Path, limit: int) -> None:
        helper = KnownListExtractor(config_path=path)
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

    def extract(self, product_name: str) -> str | None:
        prompt = (
            f"Your task is to extract the brand from the following product title: {product_name}. "
            "There are a few guidelines to follow:\n"
            "- Be careful with generic words, items, products, or ingredients present in the given "
            "product's title, do not always assume that some common word as such is an indication "
            "of a brand with a similar name.\n"
            "- Misspellings are okay, to some extent. If you are not able to pick up a direct match, "
            "you can try to pick up a misspelling of a potential brand name. However, with shorter "
            "brand names, be stricter with how much misspelling is acceptable, for longer brand names "
            "be less stict as there is more room for error there.\n"
            f"- Consult this brand list for possible brand names: {self._brand_list_text}. This is a "
            "list of sample of known brand names that are present in the Open Food Facts dataset. "
            "You can use this list to help you extract the brand from the product title.\n"
            "- The brand names in the earlier given list are normalized in such a way that they are "
            "stripped of special characters, lower cased, and just overall simplified. Do not let this "
            "confuse you, if a brand name in the product title seems a little different because of "
            "these small details, you may overlook that as long as it is essentially the same brand name.\n"
            "- Not every product title will contain a brand in it, if you do not see a brand name or "
            "something close to it in the title, do not give a fake brand name, just say 'none'.\n"
            "- Just to make sure you understand, I want you to either reply with only the brand name, or 'none' if you do not see a brand name in the title.\n"
             "- When you are extracting a brand name, or even comparing brand names to what is present in the product title, normalize the product title as the brand name in the given list is. Lower case it, strip special characters, strip extra spaces, etc.\n"
            "- When you return the brand name, return it in the same format as it is in the brand list.\n\n\n"
            "If at any point you are confused regarding the presence of multiple brands in a product title, Here are some tie breaking criteria:\n"
            "Earliest position wins — Prefer the brand that starts leftmost in the product title (earlier in the title beats later).\n"
            "Longer brand wins ties — If two candidates start at the same position, prefer the longer brand name (e.g. tesco finest over tesco, pg tips over tips)."
            "Grounded over ungrounded (same position) — If two brands tie on position, prefer the one whose every word actually appears in the title. Reject a longer brand if it needs words that aren’t in the title (e.g. prefer giant eagle over giant eagle inc. when inc is not in the title)."
            "All match types compete equally — Exact matches, fuzzy near-matches, and split-token matches are all candidates; the rules above pick the single best brand."
            "\n\nTo clear up any confusion about the term 'position':\n"
            "Exact / fuzzy consecutive match: position = where that brand’s word sequence starts in the title.\n"
            "Split-token match (multi-word brand, words not adjacent): position = the earliest title word that belongs to that brand (not the order of words in the brand name)."
        )
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

    def predict(self, product_name: str) -> str | None:
        return self.extract(product_name)