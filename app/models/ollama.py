# -*- coding: utf-8 -*-
"""Ollama model discovery."""
from __future__ import annotations
import subprocess
from typing import List


def list_models() -> List[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    lines = result.stdout.strip().splitlines()
    if not lines:
        return []

    models = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models
