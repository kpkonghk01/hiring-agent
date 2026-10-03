"""Loads providers.json and exposes provider/model resolution."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Global development mode flag. Preserved here because score.py and github.py
# import it from this module.
DEVELOPMENT_MODE = True

# Load .env before any os.getenv below, so values apply regardless of import order.
load_dotenv(Path(__file__).parent / ".env")

_CONFIG_PATH = Path(__file__).parent / "providers.json"
# Fork-specific providers live in their own file so providers.json stays as upstream ships it.
_CUSTOM_CONFIG_PATH = Path(__file__).parent / "providers.custom.json"

# A provider whose "models" is this string accepts any model name.
_ANY_MODEL = "*"
_DEFAULT_MODEL_PARAMS = {"temperature": 0.1, "top_p": 0.9}

with open(_CONFIG_PATH) as _f:
    _config = json.load(_f)

if _CUSTOM_CONFIG_PATH.exists():
    _custom_providers = json.loads(_CUSTOM_CONFIG_PATH.read_text(encoding="utf-8"))
    _config = {
        **_config,
        "providers": {**_config["providers"], **_custom_providers.get("providers", {})},
    }

# Default model, overridable by env.
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", _config["default_model"])

# Flat model -> {temperature, top_p} map. Preserves the contract that
# prompt.MODEL_PARAMETERS exposed to evaluator.py / pdf.py / github.py / score.py.
# Providers that accept any model ("models": "*") have no entries here; their models
# fall back to each caller's default parameters.
MODEL_PARAMETERS = {
    model: {k: v for k, v in params.items() if k in ("temperature", "top_p")}
    for provider in _config["providers"].values()
    if provider["models"] != _ANY_MODEL
    for model, params in provider["models"].items()
}


def _serves(provider: dict, model_name: str) -> bool:
    return provider["models"] == _ANY_MODEL or model_name in provider["models"]


def provider_for(model_name: str, provider_name: str = None) -> dict:
    """Resolve provider config for a model.

    The provider is ``provider_name``, else env ``LLM_PROVIDER``, else inferred from
    which provider lists the model. Only an explicitly chosen provider may serve a
    model it does not list (when its "models" is "*").

    Returns {name, base_url, api_key, structured_output, extra_body, params}.
    Raises ValueError if the provider or model is unknown or its required key is unset.
    """
    providers = _config["providers"]
    provider_name = provider_name or os.getenv("LLM_PROVIDER")
    if provider_name:
        if provider_name not in providers:
            raise ValueError(
                f"Unknown provider '{provider_name}'. "
                f"Available providers: {', '.join(sorted(providers))}"
            )
        if not _serves(providers[provider_name], model_name):
            raise ValueError(
                f"Provider '{provider_name}' does not list model '{model_name}'."
            )
        return _resolve(provider_name, model_name)

    for name, prov in providers.items():
        if prov["models"] != _ANY_MODEL and model_name in prov["models"]:
            return _resolve(name, model_name)

    available = ", ".join(sorted(MODEL_PARAMETERS))
    raise ValueError(
        f"Unknown model '{model_name}'. Available models: {available}. "
        f"To use an unlisted model, set LLM_PROVIDER to a provider that accepts any model."
    )


def _resolve(name: str, model_name: str) -> dict:
    prov = _config["providers"][name]
    api_key_env = prov.get("api_key_env")
    api_key = os.getenv(api_key_env) if api_key_env else None
    if api_key_env and not api_key:
        raise ValueError(
            f"Model '{model_name}' uses provider '{name}', which requires "
            f"env var '{api_key_env}', but it is unset."
        )
    model_config = {} if prov["models"] == _ANY_MODEL else prov["models"].get(model_name, {})
    extra_body = {
        **prov.get("extra_body", {}),
        **model_config.get("extra_body", {}),
    }
    base_url_env = prov.get("base_url_env")
    base_url = (os.getenv(base_url_env) if base_url_env else None) or prov["base_url"]
    return {
        "name": name,
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "structured_output": prov.get("structured_output", "json_schema"),
        "extra_body": extra_body,
        "params": {
            **_DEFAULT_MODEL_PARAMS,
            **{k: v for k, v in model_config.items() if k in ("temperature", "top_p")},
        },
    }
