"""Centralised config loader — finds config.yaml relative to this file."""
from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)
