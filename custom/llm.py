"""Which LLM each pipeline stage talks to.

Providers come from providers.json overlaid with providers.custom.json (merged by
config.py). Resolution for batch.py lives here, not in config.py, so the upstream
file keeps only a load hook (docs/adr/0001): an explicitly chosen provider
(argument or env ``LLM_PROVIDER``) may serve any model name; without one, the
provider is the one that lists the model, as upstream infers it.

Upstream modules (pdf.py, github.py, evaluator.py) read a module-level
``DEFAULT_MODEL`` and call a module-level ``initialize_llm_provider``. Rather than
editing them, ``bound_to`` temporarily points one module at a chosen provider and
model, so extraction and judging can use different models.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Iterator, Optional

import config
from models import OpenAICompatibleProvider

ANY_MODEL = "*"
# The same fallback pdf.py and github.py use for models providers.json does not list.
_DEFAULT_PARAMS = {"temperature": 0.1, "top_p": 0.9}
_PARAM_KEYS = ("temperature", "top_p")


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
    """Pick and validate the provider for ``model``.

    Raises ValueError (unknown provider, unlisted model without a provider, malformed
    provider config, missing API key) so a bad choice fails before any Resume is processed.
    """
    name = provider or os.getenv("LLM_PROVIDER") or _provider_listing(model)
    spec = _provider_spec(name)
    _api_key(name, spec)
    listed = _listed_config(spec, model)
    params = {
        **_DEFAULT_PARAMS,
        **{k: v for k, v in listed.items() if k in _PARAM_KEYS},
    }
    return LLMChoice(provider=name, model=model, params=params)


def build_client(choice: LLMChoice) -> OpenAICompatibleProvider:
    spec = _provider_spec(choice.provider)
    base_url_env = spec.get("base_url_env")
    base_url = (os.getenv(base_url_env) if base_url_env else None) or spec["base_url"]
    return OpenAICompatibleProvider(
        base_url=base_url.rstrip("/"),
        api_key=_api_key(choice.provider, spec),
        structured_output=spec.get("structured_output", "json_schema"),
        extra_body={
            **spec.get("extra_body", {}),
            **_listed_config(spec, choice.model).get("extra_body", {}),
        },
    )


@contextmanager
def bound_to(
    module: ModuleType, choice: LLMChoice, client: OpenAICompatibleProvider
) -> Iterator[None]:
    """Make ``module`` use ``choice`` and ``client`` for the duration of the block."""
    with patched(
        module,
        DEFAULT_MODEL=choice.model,
        initialize_llm_provider=lambda _model_name: client,
    ):
        yield


@contextmanager
def patched(module: ModuleType, **attributes) -> Iterator[None]:
    """Temporarily replace module-level attributes, restoring them afterwards."""
    saved = {name: getattr(module, name) for name in attributes}
    for name, value in attributes.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def _providers() -> dict:
    return config._config["providers"]  # providers.json + providers.custom.json


def _provider_listing(model: str) -> str:
    for name, spec in _providers().items():
        if isinstance(spec.get("models"), dict) and model in spec["models"]:
            return name
    raise ValueError(
        f"No provider lists model '{model}'. Choose one with --provider or LLM_PROVIDER."
    )


def _provider_spec(name: str) -> dict:
    providers = _providers()
    if name not in providers:
        raise ValueError(
            f"Unknown provider '{name}'. Available providers: {', '.join(sorted(providers))}"
        )
    spec = providers[name]
    models = spec.get("models")
    if not spec.get("base_url") or not (
        models == ANY_MODEL or isinstance(models, dict)
    ):
        raise ValueError(
            f"Provider '{name}' needs a 'base_url' and 'models' set to \"*\" or an object "
            "(providers.json / providers.custom.json)."
        )
    return spec


def _listed_config(spec: dict, model: str) -> dict:
    models = spec["models"]
    return models.get(model, {}) if isinstance(models, dict) else {}


def _api_key(name: str, spec: dict) -> Optional[str]:
    env = spec.get("api_key_env")
    if not env:
        return None
    key = os.getenv(env)
    if not key:
        raise ValueError(
            f"Provider '{name}' requires env var '{env}', but it is unset."
        )
    return key
