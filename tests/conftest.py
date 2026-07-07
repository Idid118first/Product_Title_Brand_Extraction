"""Shared pytest fixtures and path setup for the brand-extraction test suite.

Adds `src/` to `sys.path` so tests can `import brand_extractor` directly, and provides
fixtures for building small brand lists and constructing an `LLMExtractor`/`HybridExtractor`
with the OpenAI client fully stubbed out (no API key, no network).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import brand_extractor  # noqa: E402  (import after sys.path tweak)


# --------------------------------------------------------------------------------------
# Brand-list config fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture
def brand_text_file(tmp_path: Path) -> Path:
    """A small text brand list (one brand per line, plus a comma-separated line)."""
    path = tmp_path / "brands.txt"
    path.write_text(
        "Lay's\nDoritos\nCoca-Cola\nTrader Joe's\nGiant Eagle\nPepsi, Mountain Dew\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def brand_csv_file(tmp_path: Path) -> Path:
    """A CSV whose brand column is the default-detected `brands` header."""
    path = tmp_path / "brands.csv"
    path.write_text(
        "brands,extra\nLay's,x\nDoritos,y\nCoca-Cola,z\nGiant Eagle,w\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def rejection_file(tmp_path: Path) -> Path:
    """A tiny generic-word rejection list."""
    path = tmp_path / "reject.txt"
    path.write_text("chocolate\nextra strength\n\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# OpenAI stubbing so LLM/Hybrid extractors construct without a key or network
# --------------------------------------------------------------------------------------
class FakeCompletions:
    """Stub for client.chat.completions with a scriptable single response."""

    def __init__(self, content: str = "none", usage: SimpleNamespace | None = None) -> None:
        self.content = content
        self.usage = usage or SimpleNamespace(
            prompt_tokens=10, completion_tokens=2, total_tokens=12
        )
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice], usage=self.usage)


class FakeOpenAI:
    """Stub replacing `openai.OpenAI`; records the api_key it was built with."""

    last_instance: "FakeOpenAI | None" = None

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        FakeOpenAI.last_instance = self


@pytest.fixture
def stub_openai(monkeypatch):
    """Replace the OpenAI client, silence the request delay, and set a dummy ASCII key."""
    monkeypatch.setattr(brand_extractor, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(brand_extractor.time, "sleep", lambda *_a, **_k: None)
    # Prevent load_dotenv() from pulling in a real key; setenv wins because dotenv
    # does not override already-present env vars.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")
    return FakeOpenAI


@pytest.fixture
def llm_extractor(stub_openai, brand_text_file):
    """A constructed LLMExtractor backed by the stubbed OpenAI client."""
    return brand_extractor.LLMExtractor(config_path=brand_text_file, domain="food")


@pytest.fixture
def hybrid_extractor(stub_openai, brand_text_file):
    """A constructed HybridExtractor backed by the stubbed OpenAI client."""
    return brand_extractor.HybridExtractor(config_path=brand_text_file)
