# -*- coding: utf-8 -*-
"""GGUF translation backend using llama-cpp-python."""
from __future__ import annotations
from functools import lru_cache
import os
from typing import Optional


def _normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


@lru_cache(maxsize=2)
def _load_model(
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    n_threads: int,
    n_batch: int,
):
    try:
        from llama_cpp import Llama
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("llama-cpp-python is not installed") from exc
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        n_batch=n_batch,
        verbose=False,
    )


def _wrap_prompt(prompt: str, style: str) -> str:
    style = (style or "plain").lower()
    if style == "sakura":
        system = (
            "你是一个日本二次元领域的日语翻译模型，可以流畅通顺地以日本轻小说/漫画/Galgame的风格将日文翻译成简体中文，"
            "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
            "只输出译文或JSON，不要附加说明。"
        )
        return (
            "<|im_start|>system\n"
            f"{system}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if style == "qwen":
        system = "You are a translation engine. Output only the translation or JSON."
        return (
            "<|im_start|>system\n"
            f"{system}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    return prompt


class GGUFClient:
    def __init__(
        self,
        model_path: str,
        prompt_style: str = "qwen",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int = 8,
        n_batch: int = 512,
    ) -> None:
        self._model_path = _normalize_path(model_path)
        if not os.path.isfile(self._model_path):
            raise RuntimeError(f"GGUF model not found: {self._model_path}")
        self._prompt_style = prompt_style
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._n_threads = n_threads
        self._n_batch = n_batch
        self.gpu_offload = False
        try:
            from llama_cpp import llama_cpp
            if hasattr(llama_cpp, "llama_supports_gpu_offload"):
                has_gpu = bool(llama_cpp.llama_supports_gpu_offload())
            else:
                has_gpu = hasattr(llama_cpp, "llama_cuda_init") or hasattr(llama_cpp, "llama_gpu_init")
            self.gpu_offload = has_gpu and self._n_gpu_layers != 0
        except Exception:
            self.gpu_offload = False
        if self._n_gpu_layers == -1:
             self._n_gpu_layers = 100 if self.gpu_offload else 0
             
        self._llama = _load_model(
            self._model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            n_threads=self._n_threads,
            n_batch=self._n_batch,
        )

    def is_available(self) -> bool:
        return os.path.isfile(self._model_path)
        
    def is_gpu_enabled(self) -> bool:
        return self.gpu_offload and self._n_gpu_layers > 0

    def generate(self, model: str, prompt: str, timeout: int = 600, options: Optional[dict] = None) -> str:
        opts = {"temperature": 0.2, "top_p": 0.95}
        if options:
            opts.update(options)
        max_tokens = int(opts.pop("num_predict", 256))
        stop = opts.pop("stop", None)
        if stop is None:
            stop = ["<|im_end|>", "<|im_start|>", "###", "Instruction:", "System:", "User:"]
        prompt_wrapped = _wrap_prompt(prompt, self._prompt_style)
        result = self._llama(
            prompt_wrapped,
            max_tokens=max_tokens,
            stop=stop,
            **opts,
        )
        choices = result.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("text", "")).strip()
