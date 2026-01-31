# -*- coding: utf-8 -*-
"""Style guide JSON helpers."""
from __future__ import annotations
import json
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def default_style_guide() -> Dict[str, Any]:
    return {
        "notes": "",
        "tone": "neutral",
        "glossary": [],
        "required_terms": [],
        "forbidden_terms": [],
        "characters": [], # List of {name, gender, details}
    }


def load_style_guide(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load style guide from {path}: {e}")
        # Return default to avoid crashing app for IO error
        return default_style_guide()


def save_style_guide(path: str, data: Dict[str, Any]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save style guide to {path}: {e}")
        raise
