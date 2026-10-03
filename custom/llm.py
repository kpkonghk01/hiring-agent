"""Which LLM each pipeline stage talks to.

Upstream modules (pdf.py, github.py, evaluator.py) read a module-level
``DEFAULT_MODEL`` and call a module-level ``initialize_llm_provider``. Rather than
editing them (docs/adr/0001), ``bound_to`` temporarily points one module at a
chosen provider and model, so extraction and judging can use different models.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Iterator, Optional

from config import provider_for
from models import OpenAICompatibleProvider


@dataclass(frozen=True)
class LLMChoice:
    """A model as served by one provider; ``label`` identifies a judge (ADR 0002)."""

    provider: str
    model: str
    params: dict

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def resolve_choice(model: str, provider: Optional[str] = None) -> LLMChoice:
    """Validate ``model`` against providers.json/providers.custom.json.

    Raises ValueError (unknown provider/model, missing API key) so a bad choice
    fails before any Resume is processed.
    """
    cfg = provider_for(model, provider)
    return LLMChoice(provider=cfg["name"], model=model, params=cfg["params"])


def build_client(choice: LLMChoice) -> OpenAICompatibleProvider:
    cfg = provider_for(choice.model, choice.provider)
    return OpenAICompatibleProvider(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        structured_output=cfg["structured_output"],
        extra_body=cfg["extra_body"],
    )


@contextmanager
def bound_to(
    module: ModuleType, choice: LLMChoice, client: OpenAICompatibleProvider
) -> Iterator[None]:
    """Make ``module`` use ``choice`` and ``client`` for the duration of the block."""
    saved_model = getattr(module, "DEFAULT_MODEL")
    saved_init = getattr(module, "initialize_llm_provider")
    setattr(module, "DEFAULT_MODEL", choice.model)
    setattr(module, "initialize_llm_provider", lambda _model_name: client)
    try:
        yield
    finally:
        setattr(module, "DEFAULT_MODEL", saved_model)
        setattr(module, "initialize_llm_provider", saved_init)
