# -*- coding: utf-8 -*-
"""Central opt-in runtime controls for pipeline debug artifacts."""
from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_ALL_STAGE = "all"


DEBUG_STAGE_ALIASES: dict[str, str] = {
    "all": _ALL_STAGE,
    "audit": "audit",
    "region_audit": "audit",
    "timing": "timing",
    "overlay": "overlay",
    "text_area": "text_area_plan",
    "text_area_plan": "text_area_plan",
    "tap": "text_area_plan",
    "text_area_diagnostics": "text_area_diagnostics",
    "diagnostics": "text_area_diagnostics",
    "root": "root_parent_child",
    "parent": "root_parent_child",
    "root_parent": "root_parent_child",
    "root_parent_child": "root_parent_child",
    "rpc": "root_parent_child",
    "source_glyph": "source_glyph",
    "glyph": "source_glyph",
    "translation": "translation_review",
    "translation_review": "translation_review",
    "ocr": "ocr",
    "renderer": "renderer",
    "render": "renderer",
    "cleanup": "cleanup_runtime",
    "cleanup_process": "cleanup_process",
    "cleanup_diagnostics": "cleanup_process",
    "cleanup_auth": "cleanup_process",
    "cleanup_mask": "cleanup_process",
    "cleanup_masks": "cleanup_process",
    "cleanup_runtime": "cleanup_runtime",
    "cleanup_commit": "cleanup_commit",
    "cleanup_trace": "cleanup_trace",
    "checkpoint": "diagnostic_checkpoints",
    "checkpoints": "diagnostic_checkpoints",
    "diagnostic_checkpoints": "diagnostic_checkpoints",
}


def env_bool(name: str) -> bool | None:
    value = os.getenv(name, "").strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return None


def debug_enabled(settings: Any = None) -> bool:
    value = env_bool("MT_DEBUG_ARTIFACTS")
    if value is not None:
        return value
    return bool(getattr(settings, "debug_artifacts", False))


def _source_value(source: Any, key: str, default: Any = "") -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def normalize_debug_stage(value: object) -> str:
    token = safe_trace_token(value, "").strip().lower()
    if not token:
        return ""
    return DEBUG_STAGE_ALIASES.get(token, token)


def parse_debug_stage_tokens(raw: object) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        parts = re.split(r"[,;\s]+", raw)
    elif isinstance(raw, Mapping):
        parts = raw.keys()
    elif isinstance(raw, (list, tuple, set)):
        parts = raw
    else:
        parts = [raw]
    tokens: set[str] = set()
    for part in parts:
        token = normalize_debug_stage(part)
        if token:
            tokens.add(token)
    return tokens


def debug_stages(settings: Any = None) -> set[str]:
    raw = os.getenv("MT_DEBUG_STAGES", "").strip()
    if not raw:
        raw = _source_value(settings, "debug_stages", "")
    return parse_debug_stage_tokens(raw)


def debug_disabled_stages(settings: Any = None) -> set[str]:
    raw = os.getenv("MT_DEBUG_DISABLE_STAGES", "").strip()
    if not raw:
        raw = _source_value(settings, "debug_disabled_stages", "")
    return parse_debug_stage_tokens(raw)


def debug_stage_enabled(source: Any, stage: str, *, default: bool = True) -> bool:
    """Return whether one debug artifact stage may write.

    The master debug toggle enables the debug context; stage filters decide
    which artifact families can write. If MT_DEBUG_STAGES is unset, all stages
    remain enabled for backwards-compatible explicit debug runs. If it is set,
    only named stages (or "all") write. MT_DEBUG_DISABLE_STAGES always wins.
    """

    norm = normalize_debug_stage(stage)
    if not norm:
        return False
    if isinstance(source, Mapping):
        if not bool(source.get("enabled")):
            return False
        level = str(source.get("debug_artifact_level") or debug_artifact_level()).strip().lower()
        enabled = parse_debug_stage_tokens(source.get("debug_stages"))
        disabled = parse_debug_stage_tokens(source.get("debug_disabled_stages"))
    else:
        if not debug_enabled(source):
            return False
        level = debug_artifact_level(source)
        enabled = debug_stages(source)
        disabled = debug_disabled_stages(source)
    if level == "off":
        return False
    if _ALL_STAGE in disabled or norm in disabled:
        return False
    if enabled:
        return _ALL_STAGE in enabled or norm in enabled
    return bool(default)


def diagnostic_enabled(env_name: str) -> bool:
    """Return whether a focused diagnostic stream may write debug files.

    MT_DEBUG_DIR only selects an output directory. It never enables writes by
    itself. Focused diagnostic flags select extra streams only after the master
    MT_DEBUG_ARTIFACTS toggle is enabled.
    """

    if not debug_enabled(None):
        return False
    if not debug_stage_enabled(None, "diagnostic_checkpoints"):
        return False
    return env_bool(env_name) is True


def debug_artifact_level(settings: Any = None) -> str:
    value = os.getenv("MT_DEBUG_ARTIFACT_LEVEL", "").strip().lower()
    if not value:
        value = str(getattr(settings, "debug_artifact_level", "") or "").strip().lower()
    if value in {"full", "lite", "off"}:
        return value
    if value in _TRUE_VALUES:
        return "full"
    if value in _FALSE_VALUES:
        return "off"
    return "full"


def debug_pages(settings: Any = None) -> set[str]:
    raw = os.getenv("MT_DEBUG_PAGES", "").strip()
    if not raw:
        raw = str(getattr(settings, "debug_pages", "") or "").strip()
    pages: set[str] = set()
    for part in re.split(r"[,;\s]+", raw):
        token = part.strip()
        if not token:
            continue
        stem = os.path.splitext(os.path.basename(token))[0]
        pages.add(stem)
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            pages.add(digits)
            pages.add(digits.zfill(3))
    return pages


def debug_root(settings: Any = None) -> str:
    configured = os.getenv("MT_DEBUG_DIR", "").strip() or str(getattr(settings, "debug_dir", "") or "").strip()
    if configured:
        return configured
    export_dir = str(getattr(settings, "export_dir", "") or "").strip()
    if export_dir:
        return os.path.join(export_dir, "debug_artifacts")
    return os.path.join(os.getcwd(), "debug_artifacts")


def page_debug_dir(context: Mapping[str, Any] | None) -> str:
    if not context:
        return ""
    root_dir = str(context.get("debug_dir") or "").strip()
    page_id = str(context.get("page_id") or "page").strip() or "page"
    if not root_dir:
        return ""
    page_dir = os.path.join(root_dir, page_id)
    os.makedirs(page_dir, exist_ok=True)
    return page_dir


def stage_artifact_dir(
    context: Mapping[str, Any] | None,
    stage: str,
    *parts: str,
    default: bool = True,
) -> str:
    if not debug_stage_enabled(context, stage, default=default):
        return ""
    page_dir = page_debug_dir(context)
    if not page_dir:
        return ""
    safe_parts = [safe_trace_token(part, "debug") for part in parts if str(part or "").strip()]
    target_dir = os.path.join(page_dir, *safe_parts) if safe_parts else page_dir
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def stage_artifact_path(
    context: Mapping[str, Any] | None,
    stage: str,
    filename: str,
    *parts: str,
    default: bool = True,
) -> str:
    target_dir = stage_artifact_dir(context, stage, *parts, default=default)
    if not target_dir:
        return ""
    return os.path.join(target_dir, safe_trace_token(filename, "debug"))


def write_image_path(
    context: Mapping[str, Any] | None,
    stage: str,
    path: str,
    image: Any,
    *,
    quality: int | None = None,
) -> tuple[str, bool, str]:
    if not debug_stage_enabled(context, stage):
        return "", False, "debug_stage_disabled"
    if not path:
        return "", False, "path_missing"
    try:
        target_dir = os.path.dirname(path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        if not hasattr(image, "save"):
            return path, False, "image_has_no_save_method"
        if quality is not None:
            image.save(path, quality=int(quality))
        else:
            image.save(path)
        return path, True, ""
    except Exception as exc:
        return path, False, f"{type(exc).__name__}: {exc}"


def save_context_image(
    context: Mapping[str, Any] | None,
    *,
    subdir: str,
    filename: str,
    image: Any,
    stage: str = "ocr",
) -> tuple[str, bool, str]:
    path = stage_artifact_path(context, stage, filename, subdir)
    if not path:
        return "", False, "debug_stage_disabled"
    return write_image_path(context, stage, path, image)


def perf_telemetry_enabled(settings: Any = None) -> bool:
    value = env_bool("MT_PERF_TELEMETRY")
    if value is not None:
        return value
    return bool(getattr(settings, "perf_telemetry", False))


def perf_telemetry_root(settings: Any = None) -> str:
    configured = os.getenv("MT_PERF_TELEMETRY_DIR", "").strip()
    if configured:
        return configured
    export_dir = str(getattr(settings, "export_dir", "") or "").strip()
    if export_dir:
        return os.path.join(export_dir, "performance_timing")
    return os.path.join(os.getcwd(), "performance_timing")


def safe_trace_token(value: object, fallback: str = "item") -> str:
    text = str(value or "").strip() or fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:96] or fallback


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in list(value)[:80]]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"shape": [int(item) for item in tuple(shape)]}
    return str(value)


def write_diagnostic_checkpoint(
    filename: str,
    *,
    module: str,
    stage: str,
    event: str,
    fields: Mapping[str, Any] | None = None,
    debug_dir: str | None = None,
    include_monotonic: bool = True,
) -> str:
    root_dir = str(debug_dir or "").strip() or debug_root()
    os.makedirs(root_dir, exist_ok=True)
    path = os.path.join(root_dir, filename)
    payload: dict[str, Any] = {
        "ts": time.time(),
        "module": module,
        "stage": stage,
        "event": event,
    }
    if include_monotonic:
        payload["monotonic"] = time.monotonic()
    payload.update(json_safe(dict(fields or {})))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return path


def pipeline_diagnostic_checkpoint(
    *,
    module: str,
    stage: str,
    event: str,
    fields: Mapping[str, Any] | None = None,
    debug_dir: str | None = None,
) -> str:
    """Write one generic opt-in pipeline checkpoint.

    Checkpoints are diagnostic artifacts only. They are disabled unless the
    normal debug controls and MT_PIPELINE_DIAGNOSTIC_CHECKPOINTS are enabled.
    """

    if not diagnostic_enabled("MT_PIPELINE_DIAGNOSTIC_CHECKPOINTS"):
        return ""
    try:
        return write_diagnostic_checkpoint(
            "pipeline_diagnostic_checkpoints.jsonl",
            module=module,
            stage=stage,
            event=event,
            fields=fields,
            debug_dir=debug_dir,
            include_monotonic=False,
        )
    except Exception:
        return ""
