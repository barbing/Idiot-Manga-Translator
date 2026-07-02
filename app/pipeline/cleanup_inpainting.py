# -*- coding: utf-8 -*-
"""Compatibility adapter for the standalone SimpleLama backend.

Cleanup contracts live in ``app.pipeline``. SimpleLama model resolution,
loading, warmup, and inference live in ``app.inpaint.simple_lama_engine`` so the
cleanup pipeline does not embed a model-processing engine.
"""

from __future__ import annotations

from app.inpaint.simple_lama_engine import (
    FIXED_CLEANUP_INPAINT_MODEL_ID,
    FIXED_CLEANUP_INPAINT_MODEL_NAME,
    FIXED_CLEANUP_INPAINT_MODEL_RELATIVE_PATH,
    FIXED_CLEANUP_INPAINT_SELECTION_POLICY,
    ai_inpaint,
    ai_inpaint_cleanup,
    ai_inpaint_cleanup_crop,
    clear_model_cache,
    resolve_cleanup_inpaint_model,
    run_simple_lama_model_crop,
    warm_cleanup_inpaint_model,
)

__all__ = [
    "FIXED_CLEANUP_INPAINT_MODEL_ID",
    "FIXED_CLEANUP_INPAINT_MODEL_NAME",
    "FIXED_CLEANUP_INPAINT_MODEL_RELATIVE_PATH",
    "FIXED_CLEANUP_INPAINT_SELECTION_POLICY",
    "ai_inpaint",
    "ai_inpaint_cleanup",
    "ai_inpaint_cleanup_crop",
    "clear_model_cache",
    "resolve_cleanup_inpaint_model",
    "run_simple_lama_model_crop",
    "warm_cleanup_inpaint_model",
]
