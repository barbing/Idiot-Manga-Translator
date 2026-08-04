# -*- coding: utf-8 -*-
"""Project IO helpers."""
from __future__ import annotations
import json
import os
import tempfile
from typing import Any, Dict


def default_project_dict() -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "project": {
            "name": "",
            "language": {"source": "ja", "target": "zh-Hans"},
            "created_at": "",
            "model": {"detector": "ComicTextDetector", "ocr": "PaddleOCR-VL", "translator": "ollama:auto"},
            "style_guide": "",
        },
        "pages": [],
    }


def save_project(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_project_atomic(
    path: str,
    data: Dict[str, Any],
    *,
    compact: bool = False,
) -> None:
    """Write one complete project view without exposing a partial JSON file."""

    absolute_path = os.path.abspath(path)
    parent = os.path.dirname(absolute_path) or os.getcwd()
    os.makedirs(parent, exist_ok=True)
    handle = -1
    temp_path = ""
    try:
        handle, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(absolute_path)}.",
            suffix=".tmp",
            dir=parent,
        )
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            handle = -1
            if compact:
                json.dump(
                    data,
                    stream,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            else:
                json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, absolute_path)
        temp_path = ""
    finally:
        if handle >= 0:
            os.close(handle)
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def save_project_bytes_atomic(path: str, payload: bytes) -> None:
    """Atomically publish an already-serialized complete project payload."""

    if not isinstance(payload, bytes):
        raise TypeError("project payload must be bytes")
    absolute_path = os.path.abspath(path)
    parent = os.path.dirname(absolute_path) or os.getcwd()
    os.makedirs(parent, exist_ok=True)
    handle = -1
    temp_path = ""
    try:
        handle, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(absolute_path)}.",
            suffix=".tmp",
            dir=parent,
        )
        with os.fdopen(handle, "wb") as stream:
            handle = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, absolute_path)
        temp_path = ""
    finally:
        if handle >= 0:
            os.close(handle)
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def load_project(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        project = json.load(f)
    from app.io.project_checkpoint import (
        is_project_checkpoint_descriptor,
        recover_project_from_descriptor,
    )

    if is_project_checkpoint_descriptor(project):
        return recover_project_from_descriptor(path, project)
    return project
