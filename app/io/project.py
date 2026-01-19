# -*- coding: utf-8 -*-
"""Project IO helpers."""
from __future__ import annotations
import json
from typing import Any, Dict


def default_project_dict() -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "project": {
            "name": "",
            "language": {"source": "ja", "target": "zh-Hans"},
            "created_at": "",
            "model": {"detector": "PaddleOCR", "ocr": "MangaOCR", "translator": "ollama:auto"},
            "style_guide": "",
        },
        "pages": [],
    }


def save_project(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_project(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
