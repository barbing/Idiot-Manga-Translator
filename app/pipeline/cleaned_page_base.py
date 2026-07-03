# -*- coding: utf-8 -*-
"""Cleaned page base contract for renderer input and review rerendering.

The cleanup pipeline owns source erasure. Rendering should draw onto the
immutable cleaned page image produced by cleanup, not reconstruct or rerun
cleanup from a page review action.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Mapping, Sequence


CLEANED_PAGE_BASE_VERSION = "cleaned_page_base_v1"
CLEANED_PAGE_BASE_DIRNAME = "cleaned_page_base"


def file_sha256(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def persist_cleaned_page_base(
    *,
    page_id: str,
    source_path: str,
    export_dir: str,
    cleanup_upstream_commit_result: Any,
    parent_execution_bundles: Sequence[Any] | None = None,
    cleanup_required: bool = False,
) -> dict[str, Any]:
    """Persist the renderer's cleaned page input as a cacheable page contract."""

    page_id_text = str(page_id or "")
    parent_signature_payload = _parent_signature_payload(parent_execution_bundles or [])
    commit_identity_payload = _cleanup_commit_identity_payload(cleanup_upstream_commit_result)
    commit_records = list(getattr(cleanup_upstream_commit_result, "commit_records", []) or [])
    blocked_records = list(getattr(cleanup_upstream_commit_result, "blocked_records", []) or [])
    errors = [str(item) for item in getattr(cleanup_upstream_commit_result, "errors", []) or []]
    source_hash = file_sha256(source_path)
    record: dict[str, Any] = {
        "cleaned_page_base_version": CLEANED_PAGE_BASE_VERSION,
        "page_id": page_id_text,
        "state": "unavailable",
        "valid": False,
        "image_path": "",
        "cache_path": "",
        "source_image_path": str(source_path or ""),
        "source_sha256": source_hash,
        "cleanup_required": bool(cleanup_required),
        "cleanup_commit_version": str(getattr(cleanup_upstream_commit_result, "version", "") or ""),
        "cleanup_committed_count": len(commit_records),
        "cleanup_blocked_count": len(blocked_records),
        "cleanup_committed_region_ids": _committed_region_ids(cleanup_upstream_commit_result, commit_records),
        "cleanup_blocked_region_ids": _blocked_region_ids(blocked_records),
        "cleanup_commit_record_ids": _cleanup_record_ids(commit_records),
        "cleanup_proof_ids": _cleanup_proof_ids(commit_records),
        "parent_execution_bundle_ids": [
            str(item.get("bundle_id") or "")
            for item in parent_signature_payload
            if str(item.get("bundle_id") or "")
        ],
        "parent_execution_signature": _stable_digest(parent_signature_payload),
        "cleanup_identity_signature": _stable_digest(commit_identity_payload),
        "errors": errors,
        "invalidation": {
            "valid": False,
            "reason": "",
        },
    }

    if commit_records:
        cache_path = cleaned_page_base_cache_path(export_dir, page_id_text, source_path)
        save_error = _save_cleaned_image(cleanup_upstream_commit_result, cache_path)
        if save_error:
            record.update(
                {
                    "state": "cache_write_failed",
                    "valid": False,
                    "image_path": str(source_path or ""),
                    "cache_path": cache_path,
                    "invalidation": {
                        "valid": False,
                        "reason": save_error,
                    },
                }
            )
            return record
        record.update(
            {
                "state": "committed",
                "valid": True,
                "image_path": cache_path,
                "cache_path": cache_path,
                "cleaned_page_base_sha256": file_sha256(cache_path),
                "invalidation": {
                    "valid": True,
                    "reason": "",
                },
            }
        )
        return record

    if not cleanup_required and not blocked_records and not errors:
        record.update(
            {
                "state": "source_noop",
                "valid": bool(source_path and os.path.isfile(source_path)),
                "image_path": str(source_path or ""),
                "cache_path": "",
                "cleaned_page_base_sha256": source_hash,
                "invalidation": {
                    "valid": bool(source_path and os.path.isfile(source_path)),
                    "reason": "" if source_path and os.path.isfile(source_path) else "source_image_missing",
                },
            }
        )
        return record

    reason = "cleanup_blocked" if blocked_records else "cleanup_required_but_not_committed"
    if errors:
        reason = "cleanup_commit_errors"
    record.update(
        {
            "state": "blocked_or_partial",
            "valid": False,
            "image_path": str(source_path or ""),
            "cache_path": "",
            "cleaned_page_base_sha256": source_hash,
            "invalidation": {
                "valid": False,
                "reason": reason,
            },
        }
    )
    return record


def cleaned_page_base_cache_path(export_dir: str, page_id: str, source_path: str) -> str:
    base_dir = os.path.join(str(export_dir or ""), CLEANED_PAGE_BASE_DIRNAME)
    safe_page_id = _safe_filename(page_id or "page")
    source_stem = _safe_filename(os.path.splitext(os.path.basename(str(source_path or "")))[0] or "source")
    return os.path.join(base_dir, f"{safe_page_id}_{source_stem}_cleaned_base.png")


def resolve_cleaned_page_base_for_rerender(page_record: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the image path page review should render onto plus audit status."""

    source_path = str(page_record.get("image_path") or "")
    record = page_record.get("cleaned_page_base") or {}
    if isinstance(record, Mapping):
        image_path = str(record.get("image_path") or "")
        state = str(record.get("state") or "")
        valid = bool(record.get("valid"))
        if valid and image_path and os.path.isfile(image_path):
            return image_path, {
                "cleaned_page_base_version": CLEANED_PAGE_BASE_VERSION,
                "status": "cache_hit",
                "state": state,
                "image_path": image_path,
                "fallback_image_path": "",
                "reason": "",
            }
        if valid and image_path:
            return source_path, {
                "cleaned_page_base_version": CLEANED_PAGE_BASE_VERSION,
                "status": "fallback_original_source",
                "state": state,
                "image_path": image_path,
                "fallback_image_path": source_path,
                "reason": "cleaned_page_base_file_missing",
            }
        if record:
            invalidation = record.get("invalidation") or {}
            reason = ""
            if isinstance(invalidation, Mapping):
                reason = str(invalidation.get("reason") or "")
            return source_path, {
                "cleaned_page_base_version": CLEANED_PAGE_BASE_VERSION,
                "status": "fallback_original_source",
                "state": state or "invalid",
                "image_path": image_path,
                "fallback_image_path": source_path,
                "reason": reason or "cleaned_page_base_invalid",
            }
    return source_path, {
        "cleaned_page_base_version": CLEANED_PAGE_BASE_VERSION,
        "status": "fallback_original_source",
        "state": "missing",
        "image_path": "",
        "fallback_image_path": source_path,
        "reason": "cleaned_page_base_missing",
    }


def _save_cleaned_image(cleanup_upstream_commit_result: Any, cache_path: str) -> str:
    cleaned_image = getattr(cleanup_upstream_commit_result, "cleaned_image", None)
    if cleaned_image is None or not hasattr(cleaned_image, "save"):
        return "cleaned_image_unavailable"
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cleaned_image.save(cache_path)
    except Exception as exc:
        return f"cleaned_page_base_write_error:{type(exc).__name__}: {exc}"
    if not os.path.isfile(cache_path):
        return "cleaned_page_base_write_missing"
    return ""


def _parent_signature_payload(parent_execution_bundles: Sequence[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for bundle in parent_execution_bundles or []:
        if isinstance(bundle, Mapping):
            record = bundle
        elif hasattr(bundle, "to_audit_dict"):
            record = bundle.to_audit_dict()
        else:
            record = {
                "bundle_id": getattr(bundle, "bundle_id", ""),
                "parent_id": getattr(bundle, "parent_id", ""),
                "root_id": getattr(bundle, "root_id", ""),
            }
        payload.append(
            {
                "bundle_id": str(record.get("bundle_id") or ""),
                "parent_id": str(record.get("parent_id") or ""),
                "root_id": str(record.get("root_id") or ""),
                "state": str(record.get("state") or ""),
                "role": str(record.get("role") or ""),
                "cleanup_required": bool(record.get("cleanup_required")),
                "source_text": str(record.get("source_text") or ""),
                "source_region_ids": _list_strings(record.get("source_region_ids")),
                "source_contract_region_id": str(record.get("source_contract_region_id") or ""),
                "source_contract_bbox": _list_ints(record.get("source_contract_bbox")),
                "parent_bbox": _list_ints(record.get("parent_bbox")),
                "cleanup_target_bbox": _list_ints(record.get("cleanup_target_bbox")),
                "render_allowed_area": _list_ints(record.get("render_allowed_area")),
                "source_glyph_mask_ids": _list_strings(record.get("source_glyph_mask_ids")),
                "cleanup_job_ids": _list_strings(record.get("cleanup_job_ids")),
                "cleanup_mask_ids": _list_strings(record.get("cleanup_mask_ids")),
                "render_decision_id": str(record.get("render_decision_id") or ""),
                "semantic_class": str(record.get("semantic_class") or ""),
                "route_intent": str(record.get("route_intent") or ""),
                "reading_order_index": int(record.get("reading_order_index") or 0),
            }
        )
    return sorted(payload, key=lambda item: (item.get("reading_order_index", 0), item.get("bundle_id", "")))


def _cleanup_commit_identity_payload(cleanup_upstream_commit_result: Any) -> dict[str, Any]:
    commit_records = list(getattr(cleanup_upstream_commit_result, "commit_records", []) or [])
    blocked_records = list(getattr(cleanup_upstream_commit_result, "blocked_records", []) or [])
    return {
        "version": str(getattr(cleanup_upstream_commit_result, "version", "") or ""),
        "commits": [
            {
                "region_id": str(record.get("region_id") or ""),
                "cleanup_result_id": str(record.get("cleanup_result_id") or ""),
                "cleanup_plan_id": str(record.get("cleanup_plan_id") or ""),
                "cleanup_mask_id": str(record.get("cleanup_mask_id") or ""),
                "cleanup_proof_id": str(record.get("cleanup_proof_id") or ""),
                "cleanup_committed_to_working_image": bool(record.get("cleanup_committed_to_working_image")),
                "failure_reason": str(record.get("failure_reason") or ""),
            }
            for record in commit_records
            if isinstance(record, Mapping)
        ],
        "blocked": [
            {
                "region_id": str(record.get("region_id") or ""),
                "cleanup_result_id": str(record.get("cleanup_result_id") or ""),
                "cleanup_plan_id": str(record.get("cleanup_plan_id") or ""),
                "cleanup_mask_id": str(record.get("cleanup_mask_id") or ""),
                "cleanup_proof_id": str(record.get("cleanup_proof_id") or ""),
                "failure_reason": str(record.get("failure_reason") or record.get("block_reason") or ""),
            }
            for record in blocked_records
            if isinstance(record, Mapping)
        ],
        "errors": [str(item) for item in getattr(cleanup_upstream_commit_result, "errors", []) or []],
    }


def _committed_region_ids(cleanup_upstream_commit_result: Any, commit_records: Sequence[Mapping[str, Any]]) -> list[str]:
    committed_region_ids = getattr(cleanup_upstream_commit_result, "committed_region_ids", None)
    if committed_region_ids is not None:
        return sorted(_list_strings(committed_region_ids))
    return sorted(
        {
            str(record.get("region_id") or "")
            for record in commit_records
            if isinstance(record, Mapping)
            and record.get("cleanup_committed_to_working_image")
            and str(record.get("region_id") or "")
        }
    )


def _blocked_region_ids(blocked_records: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(
        {
            str(record.get("region_id") or "")
            for record in blocked_records or []
            if isinstance(record, Mapping) and str(record.get("region_id") or "")
        }
    )


def _cleanup_record_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    ids: set[str] = set()
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        for key in ("cleanup_result_id", "cleanup_plan_id", "cleanup_mask_id"):
            value = str(record.get(key) or "")
            if value:
                ids.add(value)
    return sorted(ids)


def _cleanup_proof_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(
        {
            str(record.get("cleanup_proof_id") or "")
            for record in records or []
            if isinstance(record, Mapping) and str(record.get("cleanup_proof_id") or "")
        }
    )


def _stable_digest(payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return text.strip("._") or "page"


def _list_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    try:
        return [str(item) for item in value if str(item)]
    except TypeError:
        return [str(value)] if str(value) else []


def _list_ints(value: Any) -> list[int]:
    result: list[int] = []
    if value is None:
        return result
    try:
        iterable = list(value)
    except TypeError:
        return result
    for item in iterable:
        try:
            result.append(int(round(float(item))))
        except (TypeError, ValueError):
            continue
    return result
