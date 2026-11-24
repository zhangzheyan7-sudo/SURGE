from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _merge_mappings(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mappings(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path, loading: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in loading:
        chain = " -> ".join(str(item) for item in (*loading, path))
        raise ValueError(f"Circular configuration inheritance: {chain}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = Path(parent).expanduser()
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _merge_mappings(_load_yaml(parent_path, (*loading, path)), config)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = _load_yaml(path)
    config["_config_path"] = str(path)
    config["_project_root"] = str(Path(__file__).resolve().parents[2])
    _resolve_paths(config)
    from .paper_rgcm import validate_paper_config
    validate_paper_config(config)
    return config


def _resolve_path(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def _resolve_paths(config: dict[str, Any]) -> None:
    project_root = Path(config["_project_root"])
    for dataset in config.get("data", {}).get("datasets", []):
        dataset["root"] = _resolve_path(dataset["root"], project_root)
    for key in ("split_dir",):
        if key in config.get("data", {}):
            config["data"][key] = _resolve_path(config["data"][key], project_root)
    for key in ("repo_dir", "weights"):
        if key in config.get("model", {}).get("dino", {}):
            config["model"]["dino"][key] = _resolve_path(config["model"]["dino"][key], project_root)
    for section, key in (("training", "output_dir"), ("evaluation", "output_dir"), ("visualization", "output_dir")):
        if key in config.get(section, {}):
            config[section][key] = _resolve_path(config[section][key], project_root)
    for section in ("evaluation", "visualization"):
        if "checkpoint" in config.get(section, {}):
            config[section]["checkpoint"] = _resolve_path(config[section]["checkpoint"], project_root)
    if config.get("training", {}).get("resume"):
        config["training"]["resume"] = _resolve_path(config["training"]["resume"], project_root)


def deep_update(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
