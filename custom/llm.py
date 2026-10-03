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

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Iterator, List, Optional, Union

import requests

import config
from models import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

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


class LLMAuthError(Exception):
    """The provider rejected the credentials; every later call would fail the same way,
    so the whole batch stops instead of failing (and OCR-ing) Resume after Resume."""


# HTTP statuses that mean the API key itself is wrong, revoked or not permitted.
_AUTH_STATUSES = {401, 403}
# Wall-clock limit for one LLM call, retries included. Upstream's requests timeout only
# bounds the gap between bytes, so a provider that holds the connection open (as an
# OpenRouter upstream did in a trial run) would otherwise hang the batch forever.
_DEFAULT_LLM_TIMEOUT_SECONDS = 90.0
# EXTRACTION_REASONING / JUDGE_REASONING = off add OpenRouter's switch for a model's
# thinking phase to that stage's calls. In trials, reasoning models at our low
# temperature sometimes thought without end (160k characters in 150 s, no answer),
# which looked like a hung provider; with thinking off the same calls took ~2 s.
# Opt-in, because OpenRouter finds no endpoint at all when a model without
# reasoning support receives the switch.
_REASONING_OFF = {"reasoning": {"enabled": False}}
REASONING_SETTINGS = ("EXTRACTION_REASONING", "JUDGE_REASONING")


def llm_timeout_seconds() -> float:
    """``LLM_TIMEOUT`` from the environment (seconds), else the default."""
    raw = os.getenv("LLM_TIMEOUT")
    if not raw:
        return _DEFAULT_LLM_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"LLM_TIMEOUT must be a number of seconds, got '{raw}'"
        ) from None
    if value <= 0:
        raise ValueError(f"LLM_TIMEOUT must be positive, got '{raw}'")
    return value


def _with_deadline(fn, timeout: float, *args, **kwargs):
    """Run ``fn`` but give up after ``timeout`` seconds.

    A blocked socket read cannot be interrupted, so the call runs on a daemon thread
    that is abandoned on timeout; it cannot keep the process alive at exit.
    """
    outcome = {}

    def target():
        try:
            outcome["value"] = fn(*args, **kwargs)
        except BaseException as err:
            outcome["error"] = err

    worker = threading.Thread(target=target, daemon=True, name="llm-call")
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"no response within {timeout:.0f}s (LLM_TIMEOUT)")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


class ObservedClient:
    """Wraps a provider client for one stage of one Resume.

    Upstream pdf.py and github.py catch every LLM error and carry on, so failures are
    recorded here: ``check()`` re-raises a rejected key, and ``first_failure`` explains
    an extraction that came back empty.
    """

    def __init__(self, client: OpenAICompatibleProvider, choice: LLMChoice):
        self._client = client
        self._choice = choice
        self._timeout = llm_timeout_seconds()
        self._auth_error: Optional[LLMAuthError] = None
        self._failures: List[str] = []

    def chat(self, *args, **kwargs):
        if self._auth_error:
            raise self._auth_error  # no point calling again with a rejected key
        try:
            started = time.monotonic()
            reply = _with_deadline(self._client.chat, self._timeout, *args, **kwargs)
            logger.info(
                "%s answered in %.1fs", self._choice.label, time.monotonic() - started
            )
            return reply
        except requests.HTTPError as err:
            status = getattr(err.response, "status_code", None)
            if status in _AUTH_STATUSES:
                self._auth_error = LLMAuthError(
                    f"{self._choice.provider} rejected the API key (HTTP {status}) for "
                    f"model '{self._choice.model}'; check its API key in .env"
                )
                raise self._auth_error from err
            self._failures.append(f"{type(err).__name__}: {err}")
            raise
        except Exception as err:
            self._failures.append(f"{type(err).__name__}: {err}")
            raise

    @property
    def first_failure(self) -> Optional[str]:
        return self._failures[0] if self._failures else None

    def check(self) -> None:
        """Re-raise a rejected key that upstream code swallowed."""
        if self._auth_error:
            raise self._auth_error


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


def reasoning_extra_body(setting_name: str) -> dict:
    """Extra request fields for one stage, from env ``setting_name`` (see
    ``REASONING_SETTINGS``): ``off`` disables thinking, ``default`` sends nothing."""
    setting = (os.getenv(setting_name) or "default").strip().lower()
    if setting == "default":
        return {}
    if setting == "off":
        return _REASONING_OFF
    raise ValueError(f"{setting_name} must be 'off' or 'default', got '{setting}'")


def build_client(
    choice: LLMChoice, extra_body: Optional[dict] = None
) -> OpenAICompatibleProvider:
    """A client for ``choice``; ``extra_body`` adds stage-specific request fields."""
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
            **(extra_body or {}),
        },
    )


@contextmanager
def bound_to(
    module: ModuleType,
    choice: LLMChoice,
    client: Union[OpenAICompatibleProvider, ObservedClient],
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
