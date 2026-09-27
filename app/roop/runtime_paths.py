"""Machine-local runtime paths shared by launchers and model loaders.

The checked-in example documents the schema; runtime.local.json is deliberately
ignored by Git because absolute paths belong to one machine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "app" / "models"
LOCAL_CONFIG = PROJECT_ROOT / "runtime.local.json"


def runtime_config() -> dict:
    if not LOCAL_CONFIG.is_file():
        return {}
    try:
        data = json.loads(LOCAL_CONFIG.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid runtime configuration {LOCAL_CONFIG}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Runtime configuration must be a JSON object: {LOCAL_CONFIG}")
    return data


def models_directory() -> Path:
    """Use the explicit directory, otherwise the normal app/models cache.

    ROOP_MODELS_DIR allows a launcher to override runtime.local.json. An
    explicit path must already exist so a typo cannot silently create a second
    model library or trigger a large download into the wrong location.
    """
    configured = os.environ.get("ROOP_MODELS_DIR") or runtime_config().get("models_dir")
    if not configured:
        return DEFAULT_MODELS_DIR
    if not isinstance(configured, str):
        raise ValueError("models_dir must be a path string")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Configured model directory does not exist: {path}")
    return path
