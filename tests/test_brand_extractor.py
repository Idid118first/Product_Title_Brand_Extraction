"""Test suite for the brand-extraction core logic.

Organized into one class per unit under test:

- TestKnownListMatchResult  — confidence flags and hybrid-fallback triggering
- TestBrandExtractor        — shared pure helpers on the base class
- TestKnownListExtractor    — rule-based matching pipeline
- TestLLMExtractor          — prompt building, response parsing, cost, stubbed API call
- TestHybridExtractor       — known-list-first routing with a stubbed LLM
- TestConfigLoading         — brand-list / rejection-list file loading and path resolution

The OpenAI client is fully stubbed (see conftest.py), so no API key or network is used.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic import ValidationError

import brand_extractor
from brand_extractor import (
    BrandExtractor,
    HybridExtractor,
    KnownListExtractor,
    KnownListMatchResult,
    LLMExtractor,
    resolve_config_path,
)


class _Concrete(BrandExtractor):
    """Minimal concrete subclass so base-class instance methods can be exercised."""

    def extract(self, product_name: str) -> str | None:  # pragma: no cover - trivial
        return None


def _result(**kwargs) -> KnownListMatchResult:
    """Build a KnownListMatchResult with sensible defaults for the confidence tests."""
    base = {"brand": "acme", "match_type": "exact", "fuzzy_score": None, "fuzzy_threshold": None}
    base.update(kwargs)
    return KnownListMatchResult(**base)


# ======================================================================================
# General: KnownListMatchResult
# ======================================================================================
class TestKnownListMatchResult:
    def test_abstention_is_not_low_confidence(self):
        assert _result(brand=None, match_type="none").is_low_confidence is False

    def test_exact_is_not_low_confidence(self):
        assert _result(match_type="exact").is_low_confidence is False

    def test_split_token_is_low_confidence(self):
        assert _result(match_type="split_token").is_low_confidence is True

    def test_weak_fuzzy_is_low_confidence(self):
        # score within margin (5) of the threshold -> weak
        assert _result(match_type="fuzzy", fuzzy_score=86.0, fuzzy_threshold=85.0).is_low_confidence

    def test_strong_fuzzy_is_not_low_confidence(self):
        assert not _result(match_type="fuzzy", fuzzy_score=95.0, fuzzy_threshold=85.0).is_low_confidence

    def test_fuzzy_missing_metadata_is_not_low_confidence(self):
        assert _result(match_type="fuzzy").is_low_confidence is False

    def test_trigger_true_for_weak_fuzzy_on_bottom_tier(self):
        for threshold in (85.0, 82.0):
            r = _result(match_type="fuzzy", fuzzy_score=threshold + 1, fuzzy_threshold=threshold)
            assert r.triggers_hybrid_llm_fuzzy_fallback() is True

    def test_trigger_false_for_weak_fuzzy_on_upper_tier(self):
        # weak but threshold 92 is not in the fallback set {85, 82}
        r = _result(match_type="fuzzy", fuzzy_score=93.0, fuzzy_threshold=92.0)
        assert r.is_low_confidence is True
        assert r.triggers_hybrid_llm_fuzzy_fallback() is False

    def test_trigger_false_for_strong_fuzzy(self):
        r = _result(match_type="fuzzy", fuzzy_score=99.0, fuzzy_threshold=85.0)
        assert r.triggers_hybrid_llm_fuzzy_fallback() is False

    def test_trigger_false_for_split_token(self):
        assert _result(match_type="split_token").triggers_hybrid_llm_fuzzy_fallback() is False


# ======================================================================================
# BrandExtractor — shared helpers
# ======================================================================================
class TestBrandExtractor:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Merck & Co.", "merck & co."),
            ("Merck &amp; Co.", "merck & co."),
            ("Merck   &   Co.", "merck & co."),
            ("Café", "cafe"),
            ("Welch\u2019s", "welch's"),
            ("  MiXeD   Case  ", "mixed case"),
            ("", ""),
        ],
    )
    def test_normalize(self, raw, expected):
        assert BrandExtractor.normalize(raw) == expected

    @pytest.mark.parametrize(
        "brand, expected",
        [
            ("abc", 97.0),        # <= 3
            ("abcd", 93.0),       # 4-5
            ("abcde", 93.0),
            ("abcdef", 92.0),     # 6-8
            ("abcdefgh", 92.0),
            ("abcdefghi", 85.0),  # 9-12
            ("abcdefghijkl", 85.0),
            ("abcdefghijklm", 82.0),  # 13+
            ("a b c", 97.0),      # spaces excluded from length
        ],
    )
    def test_fuzzy_threshold_for_brand(self, brand, expected):
        assert BrandExtractor.fuzzy_threshold_for_brand(brand) == expected

    def test_matches_gold_variant_exact(self):
        assert BrandExtractor.matches_gold_variant("Pepsi", "pepsi") is True

    def test_matches_gold_variant_accent_folds_to_exact(self):
        assert BrandExtractor.matches_gold_variant("nestle", "Nestlé") is True

    def test_matches_gold_variant_fuzzy_within_tier(self):
        assert BrandExtractor.matches_gold_variant("doritos", "doritoss") is True

    def test_matches_gold_variant_first_letter_mismatch(self):
        assert BrandExtractor.matches_gold_variant("pepsi", "tepsi") is False

    def test_matches_gold_variant_length_gap_too_large(self):
        assert BrandExtractor.matches_gold_variant("cat", "category") is False

    def test_matches_gold_variant_empty(self):
        assert BrandExtractor.matches_gold_variant("", "pepsi") is False

    def test_matches_any_gold_variant(self):
        assert BrandExtractor.matches_any_gold_variant("pepsi", {"coke", "pepsi"}) is True
        assert BrandExtractor.matches_any_gold_variant("sprite", {"coke", "pepsi"}) is False
        assert BrandExtractor.matches_any_gold_variant("pepsi", set()) is False

    @pytest.mark.parametrize(
        "product, expected",
        [
            ("merck & co., inc", ["merck", "&", "co.", "inc"]),
            ("a   b\tc", ["a", "b", "c"]),
            (",,,", []),
            ("", []),
        ],
    )
    def test_tokenize(self, product, expected):
        assert BrandExtractor._tokenize(product) == expected

    def test_token_char_offset(self):
        tokens = ["a", "bb", "ccc"]
        assert BrandExtractor._token_char_offset(tokens, 0) == 0
        assert BrandExtractor._token_char_offset(tokens, 1) == 2
        assert BrandExtractor._token_char_offset(tokens, 2) == 5

    def test_pick_best_earliest_then_longest(self):
        # position wins first; at equal position the longer (more negative neg_len) brand wins
        matches = [(5, -3, "abc"), (0, -2, "xy"), (0, -5, "xyzzy")]
        assert BrandExtractor._pick_best(matches) == "xyzzy"

    def test_pick_best_empty(self):
        assert BrandExtractor._pick_best([]) is None

    def test_brand_tokens_present(self):
        assert BrandExtractor._brand_tokens_present("giant eagle", ["big", "eagle", "giant"]) is True
        assert BrandExtractor._brand_tokens_present("giant eagle", ["giant", "store"]) is False

    def test_brand_tokens_present_respects_counts(self):
        assert BrandExtractor._brand_tokens_present("a a", ["a", "b"]) is False
        assert BrandExtractor._brand_tokens_present("a a", ["a", "a", "b"]) is True

    def test_drop_ungrounded_at_tied_positions(self):
        tokens = ["giant", "eagle", "thin"]
        grounded = (0, -11, "giant eagle")
        ungrounded = (0, -15, "giant eagle inc")
        kept = BrandExtractor._drop_ungrounded_at_tied_positions([grounded, ungrounded], tokens)
        assert kept == [grounded]

    def test_drop_ungrounded_keeps_all_when_none_grounded(self):
        tokens = ["something", "else"]
        cands = [(0, -5, "brand x"), (0, -7, "brand y")]
        kept = BrandExtractor._drop_ungrounded_at_tied_positions(cands, tokens)
        assert set(kept) == set(cands)

    def test_drop_ungrounded_empty(self):
        assert BrandExtractor._drop_ungrounded_at_tied_positions([], ["a"]) == []

    def test_is_generic_match_window_default_blank(self):
        # No rejection lists supplied -> never generic (domain-agnostic default)
        assert _Concrete().is_generic_match_window("chocolate") is False

    def test_is_generic_match_window_phrase_and_token(self):
        ext = _Concrete()
        ext._rejection_phrases = frozenset({"extra strength", "chocolate"})
        ext._rejection_tokens = frozenset({"extra", "strength", "chocolate"})
        assert ext.is_generic_match_window("chocolate") is True       # phrase (single token)
        assert ext.is_generic_match_window("extra strength") is True  # multi-word phrase
        assert ext.is_generic_match_window("strength") is True        # single-token match
        assert ext.is_generic_match_window("extra bold") is False     # multi-word, not a phrase

    def test_build_rejection_sets_none_is_empty(self):
        phrases, tokens = BrandExtractor._build_rejection_sets(None)
        assert phrases == frozenset() and tokens == frozenset()

    def test_build_rejection_sets_loads(self, rejection_file):
        phrases, tokens = BrandExtractor._build_rejection_sets([rejection_file])
        assert "chocolate" in phrases
        assert "extra strength" in phrases
        assert {"extra", "strength", "chocolate"} <= tokens

    def test_build_rejection_sets_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BrandExtractor._build_rejection_sets([tmp_path / "nope.txt"])


# ======================================================================================
# KnownListExtractor
# ======================================================================================
class TestKnownListExtractor:
    def test_exact_match(self, brand_text_file):
        k = KnownListExtractor(config_path=brand_text_file)
        assert k.predict("Doritos Nacho Cheese") == "Doritos"

    def test_display_form_preserved(self, brand_text_file):
        k = KnownListExtractor(config_path=brand_text_file)
        assert k.predict("Coca-Cola 12 pack") == "Coca-Cola"

    def test_abstains_when_no_brand(self, brand_text_file):
        k = KnownListExtractor(config_path=brand_text_file)
        assert k.predict("Generic Potato Chips") is None

    def test_split_token_match(self, brand_text_file):
        k = KnownListExtractor(config_path=brand_text_file)
        r = k.match("Giant purple Eagle store")
        assert r.brand == "Giant Eagle"
        assert r.match_type == "split_token"

    def test_fuzzy_match_metadata(self, brand_text_file):
        k = KnownListExtractor(config_path=brand_text_file)
        r = k.match("Doritoss Nacho Cheese")
        assert r.brand == "Doritos"
        assert r.match_type == "fuzzy"
        assert r.fuzzy_score is not None and r.fuzzy_threshold == 92.0

    def test_match_none_result(self, brand_text_file):
        k = KnownListExtractor(config_path=brand_text_file)
        r = k.match("nothing here")
        assert r.brand is None and r.match_type == "none"

    def test_match_requires_loaded_brands(self):
        k = KnownListExtractor.__new__(KnownListExtractor)  # bypass __init__
        k.brands = None
        with pytest.raises(ValueError):
            k.match("anything")

    def test_exact_match_candidates(self, brand_text_file):
        k = KnownListExtractor(config_path=brand_text_file)
        tokens = k._tokenize(k.normalize("Doritos chips"))
        cands = k._exact_match_candidates(tokens)
        assert any(c.brand == "doritos" and c.match_type == "exact" for c in cands)

    def test_fuzzy_matches_returns_candidate(self, brand_text_file):
        k = KnownListExtractor(config_path=brand_text_file)
        tokens = k._tokenize(k.normalize("Doritoss chips"))
        cands = k._fuzzy_matches(tokens)
        assert any(c.brand == "doritos" and c.match_type == "fuzzy" for c in cands)

    def test_fuzzy_matches_suppressed_by_rejection_list(self, tmp_path, rejection_file):
        brand_file = tmp_path / "b.txt"
        brand_file.write_text("Chocolatier\n", encoding="utf-8")
        k = KnownListExtractor(config_path=brand_file, rejection_list_paths=[rejection_file])
        tokens = k._tokenize(k.normalize("chocolate bar"))
        # "chocolate" window would fuzzy-match "chocolatier" but is on the rejection list
        assert k._fuzzy_matches(tokens) == []

    def test_split_token_presence_matches(self, brand_text_file):
        k = KnownListExtractor(config_path=brand_text_file)
        tokens = k._tokenize(k.normalize("Giant purple Eagle"))
        cands = k._split_token_presence_matches(tokens, set())
        assert any(c.brand == "giant eagle" and c.match_type == "split_token" for c in cands)

    def test_pick_best_candidate_and_empty(self, brand_text_file):
        from brand_extractor import _Candidate

        k = KnownListExtractor(config_path=brand_text_file)
        a = _Candidate(0, -3, "abc", "exact")
        b = _Candidate(5, -9, "abcdefghi", "exact")
        assert k._pick_best_candidate([a, b]) is a
        assert k._pick_best_candidate([]) is None

    def test_drop_ungrounded_candidates(self, brand_text_file):
        from brand_extractor import _Candidate

        k = KnownListExtractor(config_path=brand_text_file)
        tokens = ["giant", "eagle", "thin"]
        grounded = _Candidate(0, -11, "giant eagle", "exact")
        ungrounded = _Candidate(0, -15, "giant eagle inc", "fuzzy")
        kept = k._drop_ungrounded_candidates([grounded, ungrounded], tokens)
        assert kept == [grounded]

    def test_extract_predict_alias(self, brand_text_file):
        k = KnownListExtractor(config_path=brand_text_file)
        assert k.extract("Pepsi Max") == k.predict("Pepsi Max") == "Pepsi"


# ======================================================================================
# LLMExtractor
# ======================================================================================
class TestLLMExtractor:
    def test_missing_key_raises(self, monkeypatch, brand_text_file):
        monkeypatch.setattr(brand_extractor, "load_dotenv", lambda *a, **k: None)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="not set"):
            LLMExtractor(config_path=brand_text_file, api_key=None)

    def test_non_ascii_key_raises(self, monkeypatch, brand_text_file):
        monkeypatch.setattr(brand_extractor, "load_dotenv", lambda *a, **k: None)
        with pytest.raises(ValueError, match="non-ASCII"):
            LLMExtractor(config_path=brand_text_file, api_key="sk-café")

    def test_construction_wires_stub(self, llm_extractor):
        assert llm_extractor.domain == "food"
        assert isinstance(llm_extractor.client, brand_extractor.OpenAI)
        assert llm_extractor.client.api_key == "sk-test-dummy-key"
        assert llm_extractor._brand_list_text  # brand list populated

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("none", None),
            ("N/A", None),
            ("null", None),
            ("unknown", None),
            ("", None),
            ('"Pepsi"', "Pepsi"),
            ("'Doritos'", "Doritos"),
            ("Coca-Cola", "Coca-Cola"),
        ],
    )
    def test_parse_response(self, llm_extractor, raw, expected):
        assert llm_extractor._parse_response(raw) == expected

    def test_parse_response_none_input(self, llm_extractor):
        assert llm_extractor._parse_response(None) is None

    def test_parse_response_whitespace_only_quirk(self, llm_extractor):
        # Current behavior: a whitespace-only reply is truthy, so it passes the empty guard
        # and returns "" rather than None. Documented here as a known edge case.
        assert llm_extractor._parse_response("   ") == ""

    def test_build_prompt_contents(self, llm_extractor):
        prompt = llm_extractor._build_prompt("Doritos Nacho Cheese")
        assert "Doritos Nacho Cheese" in prompt
        # standalone LLM prompt puts the domain in the opening line
        assert "following food product title" in prompt
        # standalone LLM prompt does NOT embed the brand list
        assert "Consult this brand list" not in prompt
        assert "hybrid fallback" not in prompt

    def test_build_prompt_domain_substitution(self, stub_openai, brand_text_file):
        llm = LLMExtractor(config_path=brand_text_file, domain="electronics")
        assert "following electronics product title" in llm._build_prompt("x")

    def test_build_prompt_omits_brand_list_by_default(self, llm_extractor):
        # brand list text is loaded but not injected into the standalone LLM prompt
        assert llm_extractor._brand_list_text  # loaded
        assert llm_extractor._brand_list_text not in llm_extractor._build_prompt("x")

    def test_build_prompt_includes_brand_list_when_enabled(self, stub_openai, brand_text_file):
        llm = LLMExtractor(config_path=brand_text_file, include_brand_list_in_prompt=True)
        prompt = llm._build_prompt("x")
        assert "Consult this brand list" in prompt
        assert llm._brand_list_text in prompt

    def test_build_prompt_appends_fallback_context(self, llm_extractor):
        prompt = llm_extractor._build_prompt("x", fallback_context="WHY-CALLED")
        assert "hybrid fallback" in prompt
        assert "WHY-CALLED" in prompt

    def test_estimated_cost_usd(self, llm_extractor):
        llm_extractor.usage["prompt_tokens"] = 1_000_000
        llm_extractor.usage["completion_tokens"] = 500_000
        assert llm_extractor.estimated_cost_usd() == pytest.approx(0.15 + 0.30)

    def test_load_brand_list_limit(self, stub_openai, brand_text_file):
        llm = LLMExtractor(config_path=brand_text_file, brand_list_limit=2)
        assert llm._brand_list_text == "coca-cola, doritos"

    def test_load_brand_list_brand_column(self, stub_openai, brand_csv_file):
        llm = LLMExtractor(config_path=brand_csv_file, brand_column="brands")
        assert "doritos" in llm._brand_list_text

    def test_extract_returns_parsed_brand_and_tracks_usage(self, llm_extractor):
        llm_extractor.client.chat.completions.content = "Doritos"
        result = llm_extractor.extract("Doritos Nacho Cheese")
        assert result == "Doritos"
        assert llm_extractor.usage["requests"] == 1
        assert llm_extractor.usage["prompt_tokens"] == 10
        assert llm_extractor.usage["total_tokens"] == 12
        # the request used the configured model and a single user message
        kwargs = llm_extractor.client.chat.completions.last_kwargs
        assert kwargs["model"] == llm_extractor.model
        assert kwargs["messages"][0]["role"] == "user"

    def test_extract_parses_none_response(self, llm_extractor):
        llm_extractor.client.chat.completions.content = "none"
        assert llm_extractor.extract("Generic Chips") is None

    def test_extract_passes_fallback_context_into_prompt(self, llm_extractor):
        llm_extractor.client.chat.completions.content = "Pepsi"
        llm_extractor.extract("x", fallback_context="SENTINEL-CTX")
        sent_prompt = llm_extractor.client.chat.completions.last_kwargs["messages"][0]["content"]
        assert "SENTINEL-CTX" in sent_prompt

    def test_predict_alias(self, llm_extractor):
        llm_extractor.client.chat.completions.content = "Pepsi"
        assert llm_extractor.predict("Pepsi Max") == "Pepsi"


# ======================================================================================
# HybridExtractor
# ======================================================================================
class _LLMRecorder:
    """Records calls to a stubbed llm.extract and returns a fixed sentinel."""

    def __init__(self, return_value="LLM_BRAND"):
        self.return_value = return_value
        self.calls: list[dict] = []

    def __call__(self, product_name, *, fallback_context=None):
        self.calls.append({"product_name": product_name, "fallback_context": fallback_context})
        return self.return_value


class TestHybridExtractor:
    def test_init_passes_through(self, stub_openai, brand_text_file, rejection_file):
        hybrid = HybridExtractor(
            config_path=brand_text_file,
            rejection_list_paths=[rejection_file],
            domain="pharma",
        )
        assert isinstance(hybrid.known_list, KnownListExtractor)
        assert "chocolate" in hybrid.known_list._rejection_phrases
        assert hybrid.llm.domain == "pharma"

    def test_hybrid_llm_prompt_includes_brand_list(self, hybrid_extractor):
        # unlike the standalone LLM, the hybrid's LLM embeds the brand list in its prompt
        assert hybrid_extractor.llm._include_brand_list is True
        prompt = hybrid_extractor.llm._build_prompt("x")
        assert "Consult this brand list" in prompt
        assert hybrid_extractor.llm._brand_list_text in prompt

    @pytest.mark.parametrize(
        "result, expected",
        [
            (_result(brand=None, match_type="none"), True),
            (_result(brand="x", match_type="split_token"), True),
            (_result(brand="x", match_type="fuzzy", fuzzy_score=86.0, fuzzy_threshold=85.0), True),
            (_result(brand="x", match_type="fuzzy", fuzzy_score=95.0, fuzzy_threshold=85.0), False),
            (_result(brand="x", match_type="exact"), False),
        ],
    )
    def test_needs_llm(self, result, expected):
        assert HybridExtractor._needs_llm(result) is expected

    def test_fallback_context_abstention(self):
        ctx = HybridExtractor._fallback_context(_result(brand=None, match_type="none"))
        assert ctx is not None and "no brand" in ctx.lower()

    def test_fallback_context_split_token(self):
        ctx = HybridExtractor._fallback_context(_result(brand="Giant Eagle", match_type="split_token"))
        assert ctx is not None and "split-token" in ctx and "Giant Eagle" in ctx

    def test_fallback_context_weak_fuzzy(self):
        ctx = HybridExtractor._fallback_context(
            _result(brand="Acme", match_type="fuzzy", fuzzy_score=86.0, fuzzy_threshold=85.0)
        )
        assert ctx is not None and "weak fuzzy" in ctx

    def test_fallback_context_none_for_confident(self):
        assert HybridExtractor._fallback_context(_result(match_type="exact")) is None

    def test_extract_confident_skips_llm(self, hybrid_extractor, monkeypatch):
        recorder = _LLMRecorder()
        monkeypatch.setattr(
            hybrid_extractor.known_list, "match",
            lambda t: _result(brand="Doritos", match_type="exact"),
        )
        monkeypatch.setattr(hybrid_extractor.llm, "extract", recorder)
        assert hybrid_extractor.extract("Doritos Nacho") == "Doritos"
        assert recorder.calls == []

    def test_extract_abstention_calls_llm(self, hybrid_extractor, monkeypatch):
        recorder = _LLMRecorder()
        monkeypatch.setattr(
            hybrid_extractor.known_list, "match",
            lambda t: _result(brand=None, match_type="none"),
        )
        monkeypatch.setattr(hybrid_extractor.llm, "extract", recorder)
        assert hybrid_extractor.extract("mystery product") == "LLM_BRAND"
        assert len(recorder.calls) == 1
        assert recorder.calls[0]["fallback_context"] is not None

    def test_extract_split_token_calls_llm(self, hybrid_extractor, monkeypatch):
        recorder = _LLMRecorder()
        monkeypatch.setattr(
            hybrid_extractor.known_list, "match",
            lambda t: _result(brand="Giant Eagle", match_type="split_token"),
        )
        monkeypatch.setattr(hybrid_extractor.llm, "extract", recorder)
        assert hybrid_extractor.extract("x") == "LLM_BRAND"
        assert "split-token" in recorder.calls[0]["fallback_context"]

    def test_extract_weak_fuzzy_calls_llm(self, hybrid_extractor, monkeypatch):
        recorder = _LLMRecorder()
        monkeypatch.setattr(
            hybrid_extractor.known_list, "match",
            lambda t: _result(brand="Acme", match_type="fuzzy", fuzzy_score=83.0, fuzzy_threshold=82.0),
        )
        monkeypatch.setattr(hybrid_extractor.llm, "extract", recorder)
        assert hybrid_extractor.extract("x") == "LLM_BRAND"
        assert len(recorder.calls) == 1

    def test_predict_alias(self, hybrid_extractor, monkeypatch):
        monkeypatch.setattr(
            hybrid_extractor.known_list, "match",
            lambda t: _result(brand="Pepsi", match_type="exact"),
        )
        assert hybrid_extractor.predict("Pepsi Max") == "Pepsi"

    def test_usage_property_proxies_llm(self, hybrid_extractor):
        assert hybrid_extractor.usage is hybrid_extractor.llm.usage

    def test_estimated_cost_proxies_llm(self, hybrid_extractor):
        hybrid_extractor.llm.usage["prompt_tokens"] = 2_000_000
        assert hybrid_extractor.estimated_cost_usd() == pytest.approx(0.30)


# ======================================================================================
# Config / file loading
# ======================================================================================
class TestConfigLoading:
    def test_load_text_file_dedup_and_comma_expand(self, tmp_path):
        path = tmp_path / "b.txt"
        path.write_text("Pepsi, Mountain Dew\npepsi\n\nDoritos\n", encoding="utf-8")
        k = KnownListExtractor(config_path=path)
        assert k.brands == frozenset({"pepsi", "mountain dew", "doritos"})

    def test_display_map_first_seen(self, tmp_path):
        path = tmp_path / "b.txt"
        path.write_text("Lay's\nLAY'S\n", encoding="utf-8")
        k = KnownListExtractor(config_path=path)
        assert k._display_by_normalized["lay's"] == "Lay's"

    @pytest.mark.parametrize("column", ["brands", "gold_brand", "brands_norm"])
    def test_csv_default_column_detection(self, tmp_path, column):
        path = tmp_path / "b.csv"
        path.write_text(f"{column},other\nDoritos,x\nPepsi,y\n", encoding="utf-8")
        k = KnownListExtractor(config_path=path)
        assert k.brands == frozenset({"doritos", "pepsi"})

    def test_csv_explicit_brand_column(self, tmp_path):
        path = tmp_path / "b.csv"
        path.write_text("maker,sku\nSony,1\nAnker,2\n", encoding="utf-8")
        k = KnownListExtractor(config_path=path, brand_column="maker")
        assert k.brands == frozenset({"sony", "anker"})

    def test_csv_missing_explicit_column_raises(self, tmp_path):
        path = tmp_path / "b.csv"
        path.write_text("maker,sku\nSony,1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing brand column"):
            KnownListExtractor(config_path=path, brand_column="nope")

    def test_csv_no_recognized_column_raises(self, tmp_path):
        path = tmp_path / "b.csv"
        path.write_text("foo,bar\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain one of"):
            KnownListExtractor(config_path=path)

    def test_missing_config_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            KnownListExtractor(config_path=tmp_path / "does_not_exist.txt")

    def test_resolve_config_path_absolute(self, tmp_path):
        path = tmp_path / "f.txt"
        path.write_text("x", encoding="utf-8")
        assert resolve_config_path(path) == path

    def test_resolve_config_path_project_root_fallback(self):
        resolved = resolve_config_path("configs/benchmark.csv")
        assert resolved.is_file()

    def test_resolve_config_path_unresolved_returns_candidate(self):
        assert resolve_config_path("definitely_missing_xyz.txt") == Path("definitely_missing_xyz.txt")


# ======================================================================================
# Pydantic argument / config-file validation
# ======================================================================================
class TestConfigValidation:
    # --- KnownListExtractor argument validation -----------------------------------------
    def test_known_list_requires_config_path(self):
        with pytest.raises(ValidationError):
            KnownListExtractor(config_path=None)

    def test_known_list_rejection_list_wrong_type(self, brand_text_file):
        with pytest.raises(ValidationError):
            KnownListExtractor(config_path=brand_text_file, rejection_list_paths=123)

    def test_known_list_csv_requires_brand_column(self, tmp_path):
        path = tmp_path / "b.csv"
        path.write_text("foo,bar\n1,2\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="must contain one of"):
            KnownListExtractor(config_path=path)

    def test_known_list_csv_explicit_missing_column(self, tmp_path):
        path = tmp_path / "b.csv"
        path.write_text("maker,sku\nSony,1\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="missing brand column"):
            KnownListExtractor(config_path=path, brand_column="nope")

    def test_known_list_csv_with_brand_column_ok(self, brand_csv_file):
        k = KnownListExtractor(config_path=brand_csv_file)
        assert "doritos" in k.brands

    def test_known_list_text_file_skips_column_check(self, brand_text_file):
        # text files have no columns; construction should succeed
        assert KnownListExtractor(config_path=brand_text_file).brands

    # --- LLMExtractor argument validation ----------------------------------------------
    def test_llm_negative_brand_limit(self, brand_text_file):
        with pytest.raises(ValidationError):
            LLMExtractor(config_path=brand_text_file, brand_list_limit=-1)

    def test_llm_non_int_brand_limit(self, brand_text_file):
        with pytest.raises(ValidationError):
            LLMExtractor(config_path=brand_text_file, brand_list_limit="abc")

    def test_llm_blank_domain(self, brand_text_file):
        with pytest.raises(ValidationError, match="domain"):
            LLMExtractor(config_path=brand_text_file, domain="   ")

    def test_llm_blank_model(self, brand_text_file):
        with pytest.raises(ValidationError, match="model"):
            LLMExtractor(config_path=brand_text_file, model="")

    # --- HybridExtractor argument + brand-column validation ----------------------------
    def test_hybrid_csv_requires_brand_column(self, stub_openai, tmp_path):
        path = tmp_path / "b.csv"
        path.write_text("foo,bar\n1,2\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="must contain one of"):
            HybridExtractor(config_path=path)

    def test_hybrid_csv_explicit_missing_column(self, stub_openai, tmp_path):
        path = tmp_path / "b.csv"
        path.write_text("maker,sku\nSony,1\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="missing brand column"):
            HybridExtractor(config_path=path, brand_column="nope")

    def test_hybrid_negative_brand_limit(self, stub_openai, brand_text_file):
        with pytest.raises(ValidationError):
            HybridExtractor(config_path=brand_text_file, brand_list_limit=-1)

    def test_hybrid_blank_domain(self, stub_openai, brand_text_file):
        with pytest.raises(ValidationError, match="domain"):
            HybridExtractor(config_path=brand_text_file, domain="")

    def test_hybrid_csv_with_brand_column_ok(self, stub_openai, brand_csv_file):
        hybrid = HybridExtractor(config_path=brand_csv_file)
        assert "doritos" in hybrid.known_list.brands
