"""
src/utils/config.py
-------------------
Load and validate YAML config; return as nested SimpleNamespace.
Supports CLI overrides: --federated.rounds=30
"""

import argparse
import sys
import yaml
from types import SimpleNamespace


# ─── Required top-level keys ────────────────────────────────────────────────
REQUIRED_SECTIONS = ["data", "model", "ssl", "federated", "finetuning", "evaluation", "logging"]


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict into a SimpleNamespace."""
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def _namespace_to_dict(ns) -> dict:
    """Recursively convert a SimpleNamespace back to a dict."""
    if not isinstance(ns, SimpleNamespace):
        return ns
    return {k: _namespace_to_dict(v) for k, v in vars(ns).items()}


def _apply_override(config_dict: dict, key_path: str, value: str) -> None:
    """
    Apply a dotted key override to a nested dict.
    e.g. key_path='federated.rounds', value='30'
    """
    keys = key_path.split(".")
    d = config_dict
    for k in keys[:-1]:
        if k not in d:
            raise KeyError(f"Config override key '{k}' not found in config.")
        d = d[k]

    # Attempt type coercion: bool → int → float → str
    raw = value
    if raw.lower() in ("true", "false"):
        coerced = raw.lower() == "true"
    else:
        try:
            coerced = int(raw)
        except ValueError:
            try:
                coerced = float(raw)
            except ValueError:
                coerced = raw

    d[keys[-1]] = coerced


def _validate(config_dict: dict) -> None:
    """Validate that all required top-level sections exist."""
    missing = [s for s in REQUIRED_SECTIONS if s not in config_dict]
    if missing:
        raise ValueError(
            f"Config is missing required sections: {missing}. "
            f"Check your YAML file."
        )


def load_config(path: str = "configs/default.yaml") -> SimpleNamespace:
    """
    Load YAML config from `path`, apply any CLI `--key.subkey=value` overrides,
    validate required fields, and return as a nested SimpleNamespace.

    CLI usage:
        python simulation.py --config configs/default.yaml --federated.rounds=30
    """
    # Parse known args: --config and any --section.key=value overrides
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=path)
    known, unknown = parser.parse_known_args()

    config_path = known.config

    # Load YAML
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Apply CLI overrides (format: --section.key=value)
    for arg in unknown:
        if arg.startswith("--"):
            arg = arg[2:]  # strip leading --
            if "=" in arg:
                key_path, value = arg.split("=", 1)
            else:
                # flag without value (treat as True)
                key_path, value = arg, "true"
            try:
                _apply_override(config_dict, key_path, value)
            except KeyError as e:
                print(f"[WARNING] CLI override ignored: {e}")

    # Validate
    _validate(config_dict)

    return _dict_to_namespace(config_dict)
