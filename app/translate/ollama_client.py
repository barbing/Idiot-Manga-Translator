# -*- coding: utf-8 -*-
"""Ollama API client."""
from __future__ import annotations
import requests
from typing import Optional

import logging

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url.rstrip("/")

    def is_available(self, timeout: int = 5) -> bool:
        url = f"{self._base_url}/api/tags"
        try:
            response = requests.get(url, timeout=timeout)
            available = response.status_code == 200
            if not available:
                logger.warning(f"Ollama check failed. Status: {response.status_code}")
            return available
        except requests.RequestException as e:
            logger.warning(f"Ollama unavailable: {e}")
            return False

    def generate(self, model: str, prompt: str, timeout: int = 600, options: Optional[dict] = None) -> str:
        url = f"{self._base_url}/api/generate"
        default_options = {"temperature": 0.2}
        if options:
            default_options.update(options)
        payload = {"model": model, "prompt": prompt, "stream": False, "options": default_options}
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Ollama generation success for model {model}")
        return str(data.get("response", "")).strip()
