# -*- coding: utf-8 -*-
"""Style guide JSON helpers."""
from __future__ import annotations
import json
from typing import Any, Dict


def default_style_guide() -> Dict[str, Any]:
    return {
        "notes": "",
        "tone": "neutral",
        "glossary": [],
        "required_terms": [],
        "forbidden_terms": [],
    }


def load_style_guide(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_style_guide(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
