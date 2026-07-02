# -*- coding: utf-8 -*-
"""Deprecated renderer-package compatibility wrapper for fixed inpainting."""

from __future__ import annotations

from app.inpaint.simple_lama_engine import (
    FIXED_CLEANUP_INPAINT_MODEL_ID,
    ai_inpaint_cleanup,
    clear_model_cache,
)


def ai_inpaint(image, mask, use_gpu: bool = True, model_id: str = FIXED_CLEANUP_INPAINT_MODEL_ID):
    """Compatibility wrapper; model execution lives in app.inpaint."""

    return ai_inpaint_cleanup(image, mask, use_gpu=use_gpu, model_id=model_id)
