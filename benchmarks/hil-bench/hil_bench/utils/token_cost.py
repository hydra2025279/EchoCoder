from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _default_cost_config_path() -> Path:
    return Path(__file__).resolve().parents[4] / "cost.yaml"


def load_cost_config() -> dict[str, Any]:
    path = Path(os.environ.get("ECHOCODER_COST_CONFIG", _default_cost_config_path()))
    if not path.exists():
        return {
            "path": str(path),
            "sha256": None,
            "currency": "USD",
            "token_unit": 1_000_000,
            "models": {},
            "error": "cost configuration file does not exist",
        }
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        payload = yaml.safe_load(text)
    except ModuleNotFoundError:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            payload = {}
            error = f"PyYAML is required for non-JSON YAML: {exc}"
        else:
            error = None
    except Exception as exc:
        payload = {}
        error = str(exc)
    else:
        error = None
    if not isinstance(payload, dict):
        payload = {}
    models = payload.get("models")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "currency": str(payload.get("currency", "USD")),
        "token_unit": int(payload.get("token_unit", 1_000_000)),
        "models": models if isinstance(models, dict) else {},
        "error": error,
    }


def _resolve_price(model: str, models: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    if model in models and isinstance(models[model], dict):
        return model, models[model]
    lowered = model.lower()
    for configured_model, value in models.items():
        name = str(configured_model)
        normalized = name.lower()
        if isinstance(value, dict) and (
            lowered == normalized
            or lowered.endswith(f"/{normalized}")
            or lowered.endswith(f"_{normalized}")
        ):
            return name, value
    return None, None


def price_usage(model: str, input_tokens: int, output_tokens: int) -> dict[str, Any]:
    config = load_cost_config()
    pricing_model, price = _resolve_price(model, config["models"])
    input_price = price.get("input_price") if price else None
    output_price = price.get("output_price") if price else None
    cost_available = (
        isinstance(input_price, (int, float))
        and not isinstance(input_price, bool)
        and isinstance(output_price, (int, float))
        and not isinstance(output_price, bool)
        and input_price >= 0
        and output_price >= 0
    )
    input_cost = (
        input_tokens * float(input_price) / config["token_unit"] if cost_available else None
    )
    output_cost = (
        output_tokens * float(output_price) / config["token_unit"] if cost_available else None
    )
    return {
        "model": model,
        "pricing_model": pricing_model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(input_tokens) + int(output_tokens),
        "cost_available": cost_available,
        "currency": config["currency"],
        "token_unit": config["token_unit"],
        "input_price": float(input_price) if cost_available else input_price,
        "output_price": float(output_price) if cost_available else output_price,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost if cost_available else None,
        "pricing_config": config["path"],
        "pricing_config_sha256": config["sha256"],
        "pricing_error": config["error"],
    }


def combine_usage(*usages: dict[str, Any] | None) -> dict[str, Any]:
    present = [usage for usage in usages if usage]
    input_tokens = sum(int(usage.get("input_tokens") or 0) for usage in present)
    output_tokens = sum(int(usage.get("output_tokens") or 0) for usage in present)
    priced = bool(present) and all(bool(usage.get("cost_available")) for usage in present)
    input_cost = (
        sum(float(usage.get("input_cost") or 0.0) for usage in present) if priced else None
    )
    output_cost = (
        sum(float(usage.get("output_cost") or 0.0) for usage in present) if priced else None
    )
    config = load_cost_config()
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_available": priced,
        "currency": config["currency"],
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost if priced else None,
        "pricing_config": config["path"],
        "pricing_config_sha256": config["sha256"],
        "pricing_error": config["error"],
    }


def response_token_counts(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        return 0, 0
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    return int(input_tokens or 0), int(output_tokens or 0)
