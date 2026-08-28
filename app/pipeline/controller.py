# -*- coding: utf-8 -*-
"""Pipeline controller placeholder."""
from __future__ import annotations
import difflib
import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping
from app.pipeline.filters import TextFilter
from PySide6 import QtCore
from app.io.project import default_project_dict, load_project
from app.io.project_checkpoint import ProjectCheckpointSession
from app.io.style_guide import default_style_guide, load_style_guide
from app.pipeline.text_block_root_graph import (
    annotate_parent_candidate_visual_group,
    parent_candidate_contract,
    visual_parent_group_analysis,
)
from app.pipeline.parent_execution_bundle import (
    ParentExecutionBundle,
    build_parent_execution_bundles,
    parent_execution_region_records,
    sync_bundles_from_region_records,
)
from app.pipeline.debug_runtime import (
    diagnostic_enabled,
    pipeline_diagnostic_checkpoint,
    safe_trace_token,
    save_context_image,
    write_diagnostic_checkpoint,
)
from app.pipeline.style_context_cache import (
    StyleContextPageIdentity,
    build_style_context_policy_identity,
    build_style_context_run_identity,
    journal_with_committed_delta,
    load_style_context_journal,
    prepare_style_context_delta,
    style_context_snapshot_before,
)
from app.pipeline.status_contracts import (
    PipelineErrorReceipt,
    PipelineLifecycleEvent,
    PipelineProgressSnapshot,
    PipelineRetryAction,
    PipelineRunState,
    PipelineStage,
    PipelineStageEvent,
    PipelineStageOutcome,
    PipelineStageOutcomeState,
    PipelineStageTechnicalError,
    RuntimeBackendEvent,
    new_error_id,
    new_run_id,
)
from app.pipeline.steps import build_output_path, build_page_record
from app.models.ollama import list_models
from app.platform_services.compute import release_torch_memory
from app.translate.prompts import build_translation_prompt, build_batch_translation_prompt, build_entity_extraction_prompt
import tempfile
import re

import logging

logger = logging.getLogger(__name__)
_GLOSSARY_DEBUG = os.getenv("MT_DEBUG_GLOSSARY") == "1"
_TERMINAL_EMPHASIS_SYMBOL_EXPANSIONS = {
    "!": "!",
    "！": "!",
    "︕": "!",
    "?": "?",
    "？": "?",
    "︖": "?",
    "‼": "!!",
    "⁇": "??",
    "⁉": "!?",
    "⁈": "?!",
}


@dataclass(frozen=True)
class TranslationAssignment:
    assignment_id: str
    parent_id: str
    source_text: str
    cache_key: str
    region_ids: list[str]
    source_contract_owner: str = ""
    source_contract_region_id: str = ""
    source_contract_bbox: tuple[int, ...] = ()
    source_contract_scope: str = ""
    source_contract_stage: str = ""
    source_contract_ocr_confidence: float | None = None
    ocr_backend: str = ""
    ocr_model_path: str = ""
    ocr_mmproj_path: str = ""
    ocr_endpoint: str = ""
    ocr_prompt_version: str = ""
    source_quality_state: str = ""
    source_quality_action: str = ""
    source_quality_reason_codes: tuple[str, ...] = ()


@dataclass
class PageProcessingResult:
    regions: list[dict[str, Any]]
    execution_regions: list[dict[str, Any]]
    parent_execution_bundles: list[ParentExecutionBundle]
    page_class: str
    text_area_plan: Any | None = None
    ctd_segmentation_result: Any | None = None


def _stage_parent_bundle_records(
    bundles: Iterable[ParentExecutionBundle | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return durable bundle facts without debug-heavy pixel evidence."""

    records: list[dict[str, Any]] = []
    for bundle in bundles or ():
        if isinstance(bundle, ParentExecutionBundle):
            record = bundle.to_audit_dict()
        elif isinstance(bundle, Mapping):
            record = dict(bundle)
        else:
            continue
        record.pop("execution_region", None)
        record.pop("style_evidence_summary", None)
        record.pop("source_candidates", None)
        records.append(record)
    return records


def _compact_stage_outcome_artifact_summary(
    stage: PipelineStage,
    artifact_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    summary = dict(artifact_summary or {})
    bundles = summary.get("parent_execution_bundles")
    if isinstance(bundles, (list, tuple)):
        summary["parent_execution_bundles"] = _stage_parent_bundle_records(bundles)
    style_evidence = summary.get("style_evidence")
    if stage is PipelineStage.STYLE and isinstance(style_evidence, Mapping):
        compact_evidence = {
            str(key): value
            for key, value in style_evidence.items()
            if str(key) != "evidence"
        }
        evidence = style_evidence.get("evidence")
        compact_evidence["evidence_count"] = (
            len(evidence) if isinstance(evidence, (list, tuple)) else 0
        )
        summary["style_evidence"] = compact_evidence
    return summary


def resolve_parent_style_for_page(
    *,
    page_id: str,
    parent_execution_bundles: Iterable[ParentExecutionBundle],
    evidence: Iterable[Any],
    font_manager: Any,
    style_context_snapshot: Any | None = None,
) -> Any:
    """Sequence the existing v3 style owners for exactly one current page."""

    from app.pipeline import parent_font_detection as font_detection

    normalized_page_id = str(page_id or "")
    if not normalized_page_id:
        raise ValueError("current-page style resolution requires page identity")
    render_bundles = tuple(
        bundle
        for bundle in tuple(parent_execution_bundles or ())
        if bool(getattr(bundle, "render_required", False))
    )
    evidence_items = tuple(evidence or ())

    bundles_by_id: dict[str, ParentExecutionBundle] = {}
    for bundle in render_bundles:
        bundle_id = str(getattr(bundle, "bundle_id", "") or "")
        if not bundle_id:
            raise ValueError(
                f"render-required bundle identity is empty on {normalized_page_id}"
            )
        if bundle_id in bundles_by_id:
            raise ValueError(f"duplicate current-page bundle identity {bundle_id}")
        bundles_by_id[bundle_id] = bundle

    evidence_by_id: dict[str, Any] = {}
    for item in evidence_items:
        bundle_id = str(getattr(item, "bundle_id", "") or "")
        if not bundle_id:
            raise ValueError(
                f"style evidence identity is empty on {normalized_page_id}"
            )
        if bundle_id in evidence_by_id:
            raise ValueError(f"duplicate style evidence for {bundle_id}")
        evidence_by_id[bundle_id] = item

    expected_ids = set(bundles_by_id)
    actual_ids = set(evidence_by_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            "style evidence does not conserve render-required parents: "
            f"page={normalized_page_id} missing={missing} extra={extra}"
        )

    for bundle_id, bundle in bundles_by_id.items():
        item = evidence_by_id[bundle_id]
        bundle_identity = (
            str(getattr(bundle, "page_id", "") or ""),
            str(getattr(bundle, "parent_id", "") or ""),
            str(getattr(bundle, "root_id", "") or ""),
        )
        evidence_identity = (
            str(getattr(item, "page_id", "") or ""),
            str(getattr(item, "parent_id", "") or ""),
            str(getattr(item, "root_id", "") or ""),
        )
        if (
            bundle_identity[0] != normalized_page_id
            or bundle_identity != evidence_identity
        ):
            raise ValueError(
                f"style evidence identity mismatch for current-page bundle {bundle_id}"
            )

    decision_ledger = font_detection.resolve_parent_style_decision_ledger_v3(
        parent_execution_bundles=render_bundles,
        evidence=evidence_items,
        style_context_snapshot=style_context_snapshot,
    )
    style_ledger = font_detection.realize_parent_render_styles_v3(
        parent_execution_bundles=render_bundles,
        decision_ledger=decision_ledger,
        font_manager=font_manager,
    )
    return font_detection.activate_parent_render_style_ledger_v3(
        parent_execution_bundles=render_bundles,
        evidence=evidence_items,
        style_ledger=style_ledger,
    )


_STYLE_CONTEXT_CACHE_UNSET = object()


def _commit_page_project_checkpoint(
    *,
    checkpoint_session: ProjectCheckpointSession,
    project: dict[str, Any],
    committed_pages: list[dict[str, Any]],
    page_record: dict[str, Any],
    style_context_delta: Mapping[str, Any] | None = None,
    style_context_cache: Any = _STYLE_CONTEXT_CACHE_UNSET,
) -> Any:
    """Persist the next page prefix before publishing it to the GUI."""

    style_context_cache_supplied = (
        style_context_cache is not _STYLE_CONTEXT_CACHE_UNSET
    )
    if not style_context_cache_supplied:
        style_context_cache = project.get("style_context_cache")
    journal_id = (
        str(style_context_cache.get("journal_id") or "")
        if isinstance(style_context_cache, Mapping)
        else ""
    )
    receipt = checkpoint_session.commit_page(
        page_record=page_record,
        style_delta=style_context_delta,
        style_cache_journal_id=journal_id,
    )
    committed_pages.append(page_record)
    project["pages"] = list(committed_pages)
    if style_context_cache_supplied:
        project["style_context_cache"] = style_context_cache
    return receipt


def _cleanup_perf_contract_diag_enabled() -> bool:
    return diagnostic_enabled("MT_CLEANUP_PERF_CONTRACT_DIAGNOSTIC")


def _cleanup_perf_contract_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _cleanup_perf_contract_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_cleanup_perf_contract_json_safe(item) for item in list(value)[:80]]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"shape": [int(item) for item in tuple(shape)]}
    return str(value)


def _cleanup_perf_contract_checkpoint(stage: str, event: str, **fields: Any) -> None:
    if not _cleanup_perf_contract_diag_enabled():
        return
    try:
        debug_dir = str(fields.pop("debug_dir", "") or "")
        write_diagnostic_checkpoint(
            "cleanup_perf_contract_checkpoints.jsonl",
            module="app.pipeline.controller",
            stage=stage,
            event=event,
            fields=_cleanup_perf_contract_json_safe(fields),
            debug_dir=debug_dir,
        )
    except Exception:
        return


def _numeric_or_zero(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _compact_enum_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value or "")


def _merge_numeric_perf_tree(target: dict[str, Any], source: Any) -> None:
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        if isinstance(value, dict):
            child = target.setdefault(str(key), {})
            if isinstance(child, dict):
                _merge_numeric_perf_tree(child, value)
        elif (
            str(key).endswith("_ms")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            target[str(key)] = round(float(target.get(str(key)) or 0.0) + float(value), 3)


def _attach_cleanup_runtime_perf_summary(
    debug_context: dict[str, Any] | None,
    cleanup_runtime_contract_result: Any,
    *,
    runtime_elapsed_seconds: float,
    cleanup_runtime_perf: dict[str, Any] | None = None,
) -> None:
    if not isinstance(debug_context, dict) or cleanup_runtime_contract_result is None:
        return

    proof_by_result_id = {
        str(getattr(proof, "cleanup_result_id", "") or ""): proof
        for proof in getattr(cleanup_runtime_contract_result, "proof_records", []) or []
        if str(getattr(proof, "cleanup_result_id", "") or "")
    }
    perf_jobs = list((cleanup_runtime_perf or {}).get("jobs") or [])
    perf_by_result_id = {
        str(record.get("cleanup_result_id") or ""): record
        for record in perf_jobs
        if isinstance(record, dict) and str(record.get("cleanup_result_id") or "")
    }
    job_records: list[dict[str, Any]] = []
    backend_elapsed_ms = 0.0
    model_call_elapsed_ms = 0.0
    proof_elapsed_ms = 0.0
    result_runtime_observed_ms = 0.0
    substage_totals: dict[str, Any] = {}
    cleanup_devices: set[str] = set()
    crop_area_pixels = 0
    ai_backend_calls = 0

    for result in getattr(cleanup_runtime_contract_result, "result_records", []) or []:
        params = getattr(result, "backend_parameters", {}) or {}
        proof = proof_by_result_id.get(str(getattr(result, "cleanup_result_id", "") or ""))
        proof_metrics = getattr(proof, "metrics", {}) or {}
        result_id = str(getattr(result, "cleanup_result_id", "") or "")
        perf_record = perf_by_result_id.get(result_id, {})
        perf_stages = perf_record.get("stages") if isinstance(perf_record, dict) else {}
        if not isinstance(perf_stages, dict):
            perf_stages = {}
        backend_perf = perf_stages.get("backend") if isinstance(perf_stages.get("backend"), dict) else {}
        crop_local_perf = (
            perf_stages.get("crop_local_backend")
            if isinstance(perf_stages.get("crop_local_backend"), dict)
            else {}
        )
        runner_perf = backend_perf.get("runner") if isinstance(backend_perf.get("runner"), dict) else {}
        backend_ms = _numeric_or_zero(backend_perf.get("engine_total_ms"))
        backend_ms += _numeric_or_zero(crop_local_perf.get("elapsed_ms"))
        if backend_ms <= 0:
            backend_ms = _numeric_or_zero(params.get("backend_elapsed_ms"))
        model_ms = _numeric_or_zero(runner_perf.get("cuda_model_event_ms"))
        if model_ms <= 0:
            model_ms = _numeric_or_zero(runner_perf.get("model_and_output_ms"))
        if model_ms <= 0:
            model_ms = _numeric_or_zero(params.get("model_call_elapsed_ms"))
        proof_ms = _numeric_or_zero(params.get("proof_elapsed_ms") or proof_metrics.get("proof_elapsed_ms"))
        runtime_ms = _numeric_or_zero(getattr(result, "runtime_ms", 0.0))
        backend_kind = str(getattr(result, "backend_kind", "") or params.get("backend_kind") or "")
        model_attempted = bool(getattr(result, "model_invocation_attempted", False) or params.get("model_invocation_attempted"))
        crop_width = int(getattr(result, "crop_width", None) or params.get("backend_crop_width") or 0)
        crop_height = int(getattr(result, "crop_height", None) or params.get("backend_crop_height") or 0)
        crop_area = int(params.get("backend_crop_area") or getattr(result, "crop_area", None) or (crop_width * crop_height) or 0)
        if backend_kind == "model_inpaint" or model_attempted:
            ai_backend_calls += 1
            crop_area_pixels += max(0, crop_area)
            backend_elapsed_ms += backend_ms
            model_call_elapsed_ms += model_ms
        proof_elapsed_ms += proof_ms
        runtime_observed_ms = _numeric_or_zero(
            ((perf_stages.get("runtime") or {}).get("total_observed_ms"))
            if isinstance(perf_stages.get("runtime"), dict)
            else 0.0
        )
        result_runtime_observed_ms += runtime_observed_ms
        _merge_numeric_perf_tree(substage_totals, perf_stages)
        cleanup_device = str(backend_perf.get("device") or runner_perf.get("device") or "")
        if cleanup_device:
            cleanup_devices.add(cleanup_device)
        job_records.append(
            {
                "cleanup_result_id": result_id,
                "cleanup_plan_id": str(getattr(result, "cleanup_plan_id", "") or ""),
                "cleanup_job_id": str(getattr(result, "cleanup_job_id", "") or ""),
                "cleanup_mask_id": str(getattr(result, "cleanup_mask_id", "") or ""),
                "parent_execution_bundle_id": str(getattr(result, "parent_execution_bundle_id", "") or ""),
                "region_id": str(getattr(result, "region_id", "") or ""),
                "backend_name": str(getattr(result, "backend_name", "") or getattr(result, "execution_backend", "") or ""),
                "backend_kind": backend_kind,
                "backend_method": str(getattr(result, "backend_method", "") or params.get("backend_method") or ""),
                "model_invocation_attempted": model_attempted,
                "model_invocation_succeeded": bool(getattr(result, "model_invocation_succeeded", False) or params.get("model_invocation_succeeded")),
                "runtime_ms": round(runtime_ms, 3),
                "backend_elapsed_ms": round(backend_ms, 3),
                "model_call_elapsed_ms": round(model_ms, 3),
                "proof_elapsed_ms": round(proof_ms, 3),
                "crop_bbox": list(getattr(result, "crop_bbox", None) or getattr(result, "operation_bbox", None) or []),
                "crop_width": crop_width,
                "crop_height": crop_height,
                "crop_area": crop_area,
                "mask_pixels": int((getattr(result, "mask_stats", {}) or {}).get("pixels", 0) or 0),
                "proof_status": _compact_enum_value(getattr(proof, "proof_status", "")) if proof else "",
                "execution_status": str(getattr(result, "execution_status", "") or ""),
                "failure_reason": str(getattr(result, "failure_reason", "") or ""),
                "cleanup_substage_ms": perf_stages,
            }
        )

    runtime_elapsed_ms = max(0.0, float(runtime_elapsed_seconds or 0.0) * 1000.0)
    debug_context["cleanup_job_timings"] = job_records
    contract_perf = dict((cleanup_runtime_perf or {}).get("contract") or {})
    contract_total_ms = _numeric_or_zero(contract_perf.get("contract_total_ms"))
    debug_context["cleanup_runtime_summary"] = {
        "result_count": len(getattr(cleanup_runtime_contract_result, "result_records", []) or []),
        "proof_count": len(getattr(cleanup_runtime_contract_result, "proof_records", []) or []),
        "ai_backend_calls": ai_backend_calls,
        "ai_backend_crop_area_pixels": crop_area_pixels,
        "backend_elapsed_ms": round(backend_elapsed_ms, 3),
        "model_call_elapsed_ms": round(model_call_elapsed_ms, 3),
        "proof_elapsed_ms": round(proof_elapsed_ms, 3),
        "runtime_elapsed_ms": round(runtime_elapsed_ms, 3),
        "runtime_overhead_ms": round(max(0.0, runtime_elapsed_ms - backend_elapsed_ms - proof_elapsed_ms), 3),
        "result_runtime_observed_ms": round(result_runtime_observed_ms, 3),
        "contract_perf": contract_perf,
        "contract_wrapper_unattributed_ms": round(max(0.0, runtime_elapsed_ms - contract_total_ms), 3),
        "contract_non_result_proof_ms": round(
            max(0.0, contract_total_ms - result_runtime_observed_ms - proof_elapsed_ms),
            3,
        ),
        "cleanup_devices": sorted(cleanup_devices),
        "substage_totals_ms": substage_totals,
    }
    debug_context.setdefault("counts", {})["cleanup_result_records"] = len(job_records)
    debug_context.setdefault("counts", {})["cleanup_proof_records"] = len(getattr(cleanup_runtime_contract_result, "proof_records", []) or [])
    debug_context.setdefault("counts", {})["cleanup_ai_backend_calls"] = ai_backend_calls
    debug_context.setdefault("counts", {})["inpaint_calls"] = ai_backend_calls
    timing = debug_context.setdefault("timing", {})
    timing["cleanup_backend_time"] = round(backend_elapsed_ms / 1000.0, 6)
    timing["cleanup_model_call_time"] = round(model_call_elapsed_ms / 1000.0, 6)
    timing["cleanup_proof_time"] = round(proof_elapsed_ms / 1000.0, 6)
    timing["cleanup_runtime_overhead_time"] = round(
        max(0.0, runtime_elapsed_ms - backend_elapsed_ms - proof_elapsed_ms) / 1000.0,
        6,
    )
    timing["cleanup_runtime_result_observed_time"] = round(result_runtime_observed_ms / 1000.0, 6)
    timing["cleanup_contract_wrapper_unattributed_time"] = round(
        max(0.0, runtime_elapsed_ms - contract_total_ms) / 1000.0,
        6,
    )
    timing["cleanup_contract_non_result_proof_time"] = round(
        max(0.0, contract_total_ms - result_runtime_observed_ms - proof_elapsed_ms) / 1000.0,
        6,
    )


def _cleanup_mask_region_records_with_protection(
    regions: Iterable[dict[str, Any]] | None,
    debug_context: dict[str, Any] | None,
    text_area_plan: Any | None = None,
) -> list[dict[str, Any]]:
    """Expose canonical cleanup authorization/protection evidence to CleanupMask.

    Pre-OCR TextAreaPlan records are the semantic authority. Region-enriched
    records are included for diagnostics and linkage only; this helper must not
    strengthen weak pre-OCR authorization into cleanup ownership.
    """

    def iter_plan_sources() -> Iterable[tuple[str, dict[str, Any]]]:
        plan_payloads: list[tuple[str, Any]] = []
        if text_area_plan is not None:
            payload = text_area_plan
            if hasattr(payload, "to_dict"):
                try:
                    payload = payload.to_dict()
                except Exception:
                    payload = None
            plan_payloads.append(("text_area_plan", payload))
        if isinstance(debug_context, dict):
            plan_payloads.extend(
                (
                    (plan_key, debug_context.get(plan_key))
                    for plan_key in ("text_area_plan_pre_ocr", "text_area_plan")
                )
            )
        seen_plan_ids: set[int] = set()
        for plan_key, plan in plan_payloads:
            if not isinstance(plan, dict):
                continue
            plan_identity = id(plan)
            if plan_identity in seen_plan_ids:
                continue
            seen_plan_ids.add(plan_identity)
            yield plan_key, plan

    auth_by_container: dict[str, dict[str, Any]] = {}
    for _plan_key, plan in iter_plan_sources():
        for container in plan.get("containers") or []:
            if not isinstance(container, dict):
                continue
            container_id = str(container.get("container_id") or container.get("id") or "")
            if container_id and container_id not in auth_by_container:
                auth_by_container[container_id] = container

    records: list[dict[str, Any]] = []
    for region in (regions or []):
        if not isinstance(region, dict):
            continue
        record = dict(region)
        container_id = str(record.get("text_area_container_id") or record.get("container_id") or "")
        container = auth_by_container.get(container_id)
        if container:
            for src, dst in (
                ("cleanup_authorization", "cleanup_authorization"),
                ("semantic_unit_id", "semantic_unit_id"),
                ("semantic_kind", "semantic_kind"),
                ("must_not_mutate", "must_not_mutate"),
                ("protection_reason", "protection_reason"),
                ("pre_ocr_authority", "pre_ocr_authority"),
                ("source_stage", "source_stage"),
                ("authorization_source_stage", "authorization_source_stage"),
                ("authorization_basis", "authorization_basis"),
                ("authorization_explicit", "authorization_explicit"),
                ("authorization_field_origin", "authorization_field_origin"),
                ("semantic_authorization_state", "semantic_authorization_state"),
                ("parent_source_evidence", "parent_source_evidence"),
            ):
                if src in container and dst not in record:
                    record[dst] = container.get(src)
        records.append(record)

    seen_ids = {str(record.get("region_id") or record.get("id") or "") for record in records}

    def add_record(record_id: str, source: dict[str, Any], reason_hint: str = "") -> None:
        if not record_id or record_id in seen_ids:
            return
        bbox = source.get("bbox") or source.get("xyxy") or source.get("bounds")
        if not bbox:
            return
        route_intent = str(source.get("route_intent") or source.get("intent") or "")
        container_type = str(source.get("container_type") or source.get("type") or source.get("role") or "")
        cleanup_mode = str(source.get("cleanup_mode") or "")
        cleanup_authorization = str(
            source.get("cleanup_authorization")
            or source.get("text_area_cleanup_authorization")
            or ""
        )
        must_not_mutate = bool(source.get("must_not_mutate") or source.get("text_area_must_not_mutate"))
        protection_reason = str(
            source.get("protection_reason")
            or source.get("text_area_protection_reason")
            or ""
        )
        authorization_source_stage = (
            source.get("authorization_source_stage")
            or source.get("text_area_authorization_source_stage")
            or source.get("source_stage")
            or reason_hint
            or "controller_cleanup_mask_authorization_handoff"
        )
        authorization_explicit_value = source.get("authorization_explicit")
        if authorization_explicit_value is None:
            authorization_explicit_value = source.get("text_area_authorization_explicit")
        authorization_explicit = bool(authorization_explicit_value)
        authorization_field_origin = str(
            source.get("authorization_field_origin")
            or source.get("text_area_authorization_field_origin")
            or ""
        )
        if cleanup_authorization and authorization_explicit and not authorization_field_origin:
            authorization_field_origin = "fresh_text_area_plan"
        field_origins = dict(source.get("field_origins") or {})
        if cleanup_authorization:
            field_origins.setdefault("cleanup_authorization", authorization_field_origin or "unlabeled_source")
        if protection_reason:
            field_origins.setdefault("protection_reason", authorization_field_origin or "unlabeled_source")
        records.append(
            {
                "region_id": record_id,
                "container_id": source.get("container_id") or source.get("text_area_container_id") or record_id,
                "semantic_unit_id": source.get("semantic_unit_id") or source.get("text_area_semantic_unit_id") or source.get("container_id") or source.get("text_area_container_id") or record_id,
                "semantic_kind": source.get("semantic_kind") or source.get("text_area_semantic_kind") or source.get("semantic_class") or container_type,
                "bbox": bbox,
                "container_type": container_type,
                "semantic_class": source.get("semantic_class") or container_type,
                "route_intent": route_intent,
                "cleanup_authorization": cleanup_authorization,
                "must_not_mutate": must_not_mutate,
                "protection_reason": protection_reason,
                "pre_ocr_authority": bool(
                    source.get("pre_ocr_authority", source.get("text_area_pre_ocr_authority", reason_hint == "text_area_plan_pre_ocr"))
                ),
                "source_stage": authorization_source_stage,
                "authorization_source_stage": authorization_source_stage,
                "authorization_basis": source.get("authorization_basis") or source.get("text_area_authorization_basis") or "",
                "authorization_explicit": authorization_explicit,
                "authorization_field_origin": authorization_field_origin,
                "semantic_authorization_state": source.get("semantic_authorization_state")
                or source.get("text_area_semantic_authorization_state")
                or cleanup_authorization,
                "field_origins": field_origins,
                "cleanup_mode": cleanup_mode,
                "classification_reason": source.get("classification_reason") or reason_hint,
                "protection_source": reason_hint or "cleanup_mask_region_records_with_protection",
                "parent_source_evidence": source.get("parent_source_evidence") or {
                    "source_model_ids": list(source.get("source_model_ids") or []),
                    "evidence_reason_codes": list(source.get("evidence_reason_codes") or source.get("text_area_reason_codes") or []),
                    "conflict_flags": list(source.get("conflict_flags") or source.get("text_area_conflict_flags") or []),
                },
            }
        )
        seen_ids.add(record_id)

    for plan_key, plan in iter_plan_sources():
        for index, container in enumerate(plan.get("containers") or []):
            if isinstance(container, dict):
                add_record(
                    str(container.get("container_id") or container.get("id") or f"{plan_key}_container_{index:04d}"),
                    container,
                    plan_key,
                )

    if isinstance(debug_context, dict):
        for key in ("blocked_text_area_candidates", "caption_localization_candidates"):
            for index, candidate in enumerate(debug_context.get(key) or []):
                if isinstance(candidate, dict):
                    add_record(str(candidate.get("candidate_id") or candidate.get("region_id") or f"{key}_{index:04d}"), candidate, key)

    return records


def _pipeline_runtime_checkpoint(stage: str, event: str, **fields: Any) -> None:
    _cleanup_perf_contract_checkpoint(stage, event, **fields)
    debug_dir = str(fields.pop("debug_dir", "") or "")
    pipeline_diagnostic_checkpoint(
        module="app.pipeline.controller",
        stage=stage,
        event=event,
        fields=fields,
        debug_dir=debug_dir,
    )


def _cleanup_runtime_backend_event(
    run_id: str,
    warmup_record: Mapping[str, Any],
) -> RuntimeBackendEvent:
    selected = str(warmup_record.get("device") or "cpu").strip()
    requested = str(
        warmup_record.get("requested_device") or selected
    ).strip()
    return RuntimeBackendEvent(
        run_id=str(run_id),
        module_id="cleanup",
        requested_backend=requested,
        selected_backend=selected,
        fallback_reason=str(warmup_record.get("fallback_reason") or "").strip(),
    )

class PipelineStatus(QtCore.QObject):
    # GUI-5 typed status seam.  Historical signals below remain intact for the
    # compatibility shell during migration.
    lifecycle_changed = QtCore.Signal(object)
    stage_changed = QtCore.Signal(object)
    stage_outcome = QtCore.Signal(object)
    progress_snapshot = QtCore.Signal(object)
    structured_error = QtCore.Signal(object)
    runtime_backend_selected = QtCore.Signal(object)
    progress_changed = QtCore.Signal(int)
    eta_changed = QtCore.Signal(str)
    page_changed = QtCore.Signal(int, int)
    message = QtCore.Signal(str)
    queue_reset = QtCore.Signal(list)
    queue_item = QtCore.Signal(int, str)
    total_time_changed = QtCore.Signal(str)
    page_time_changed = QtCore.Signal(str)
    page_ready = QtCore.Signal(int, dict)
    consistency_issue = QtCore.Signal(list)  # Pages needing glossary update
    # Two-Pass Pipeline signals
    prescan_started = QtCore.Signal()
    prescan_progress = QtCore.Signal(int)
    prescan_finished = QtCore.Signal()


@dataclass
class PipelineSettings:
    import_dir: str
    export_dir: str
    json_path: str
    output_suffix: str
    source_lang: str
    target_lang: str
    ollama_model: str
    ollama_base_url: str
    style_guide_path: str
    font_name: str
    use_gpu: bool
    filter_background: bool
    filter_strength: str
    detector_engine: str
    ocr_engine: str
    inpaint_mode: str
    font_detection: str
    translator_backend: str
    deepseek_model: str
    deepseek_base_url: str
    # Generation Options
    ollama_temperature: float
    ollama_top_p: float
    ollama_context: int
    gguf_temperature: float
    gguf_top_p: float
    gguf_model_path: str
    gguf_prompt_style: str
    gguf_n_ctx: int
    gguf_n_gpu_layers: int
    gguf_n_threads: int
    gguf_n_batch: int
    fast_mode: bool
    auto_glossary: bool
    # New settings
    detector_input_size: int
    inpaint_model_id: str
    use_ollama_discovery: bool = False
    files_whitelist: List[str] | None = None
    discovery_model: str | None = None # Model to use for discovery (None=Auto)
    discovery_backend: str = "Ollama" # "Ollama" or "GGUF"
    discovery_base_url: str = "http://localhost:11434"
    discovery_context: int = 4096
    prescan_enabled: bool = False  # Run pre-scan to build glossary before translation
    prescan_use_ner: bool = False  # Optional heavy NER enhancement for pre-scan
    debug_ocr: bool = False  # Save OCR crop images for debugging
    prescan_only: bool = False  # Build glossary only, then stop without page translation
    gguf_cross_page_context: bool = False
    debug_artifacts: bool = False
    debug_pages: str = ""
    debug_stages: str = ""
    debug_disabled_stages: str = ""
    debug_dir: str = ""
    private_cleanup_validation_stop_after_cleanup: bool = False


class PipelineRuntimeBinding:
    """One-run provider credential carrier that is never serialized.

    Typed GUI runs resolve an opaque credential reference immediately before
    Start and place the resolved value only in this redacted, memory-only
    carrier.  Legacy callers may omit the binding and keep the historical
    provider lookup behavior.
    """

    __slots__ = ("_provider_kind", "_resolved_credential")

    def __init__(
        self,
        *,
        provider_kind: object,
        resolved_credential: str | None,
    ) -> None:
        raw_kind = getattr(provider_kind, "value", provider_kind)
        normalized_kind = str(raw_kind or "").strip().casefold().replace("_", "-")
        if not normalized_kind:
            raise ValueError("provider_kind must not be empty")
        if resolved_credential is not None and not isinstance(resolved_credential, str):
            raise TypeError("resolved_credential must be a string or None")
        self._provider_kind = normalized_kind
        self._resolved_credential = (
            resolved_credential.strip() if resolved_credential is not None else None
        )

    @property
    def provider_kind(self) -> str:
        return self._provider_kind

    @property
    def has_resolved_credential(self) -> bool:
        return bool(self._resolved_credential)

    def credential_for_backend(self, backend: str) -> str:
        normalized_backend = str(backend or "").strip().casefold().replace("_", "-")
        if normalized_backend != self._provider_kind:
            raise ValueError(
                "runtime provider binding does not match the selected translator backend"
            )
        if not self._resolved_credential:
            raise ValueError("resolved runtime credential is unavailable")
        return self._resolved_credential

    def __repr__(self) -> str:
        state = "<redacted>" if self._resolved_credential else "<unavailable>"
        return (
            "PipelineRuntimeBinding("
            f"provider_kind={self._provider_kind!r}, resolved_credential={state})"
        )

    __str__ = __repr__


def _missing_required_gguf_model_path(settings: object) -> bool:
    """Return whether the selected translation backend lacks its GGUF path."""

    backend = str(getattr(settings, "translator_backend", "") or "").strip()
    model_path = str(getattr(settings, "gguf_model_path", "") or "").strip()
    return backend == "GGUF" and not model_path


def _deepseek_unavailable_message(
    runtime_binding: PipelineRuntimeBinding | None,
) -> str:
    if runtime_binding is not None:
        return (
            "DeepSeek API is not available. Verify the selected provider "
            "credential and network access."
        )
    return (
        "DeepSeek API is not available. Set DEEPSEEK_API_KEY or api/API_KEY "
        "and verify network access."
    )


OCR_ENGINE_PADDLE_VL = "PaddleOCR-VL"
OCR_ENGINE_MANGA = "MangaOCR"
OCR_ENGINE_CHOICES = (OCR_ENGINE_PADDLE_VL, OCR_ENGINE_MANGA)


def _normalize_ocr_engine_name(value: str) -> str:
    text = str(value or "").strip()
    normalized = text.replace("_", "-").replace(" ", "").lower()
    if normalized in {"paddleocr", "paddleocrvl", "paddleocr-vl", "paddleocr-v1.6", "paddleocrvl1.6"}:
        return OCR_ENGINE_PADDLE_VL
    if normalized in {"mangaocr", "manga-ocr"}:
        return OCR_ENGINE_MANGA
    return OCR_ENGINE_PADDLE_VL


def _create_selected_ocr_engine(settings: PipelineSettings, message_callback=None):
    selected = _normalize_ocr_engine_name(settings.ocr_engine)
    if selected != settings.ocr_engine and message_callback:
        message_callback(f"OCR Engine '{settings.ocr_engine}' is no longer available; using {selected}.")
    if selected == OCR_ENGINE_PADDLE_VL:
        from app.ocr.paddle_ocr_vl_engine import PaddleOcrVlEngine

        if message_callback:
            message_callback("OCR Engine: PaddleOCR-VL.")
        return PaddleOcrVlEngine(use_gpu=settings.use_gpu)
    if selected == OCR_ENGINE_MANGA:
        from app.ocr.manga_ocr_engine import MangaOcrEngine, ensure_torch_runtime_ready

        try:
            ensure_torch_runtime_ready()
        except Exception:
            pass
        force_worker = os.getenv("MT_FORCE_MANGA_OCR_WORKER") == "1"
        try:
            if force_worker:
                raise RuntimeError("Forced MangaOCR worker mode.")
            if message_callback:
                message_callback("OCR Engine: MangaOCR.")
            return MangaOcrEngine(settings.use_gpu)
        except Exception as exc:
            if _is_torch_missing(exc):
                raise
            try:
                from app.ocr.manga_ocr_worker import MangaOcrWorker

                if message_callback:
                    message_callback("MangaOCR in-process failed; using MangaOCR worker process.")
                return MangaOcrWorker(use_gpu=settings.use_gpu)
            except Exception as worker_exc:
                raise RuntimeError(f"MangaOCR failed to initialize: {worker_exc}") from worker_exc
    raise RuntimeError(f"Unsupported OCR engine: {settings.ocr_engine}")


class PipelineWorker(QtCore.QThread):
    lifecycle_changed = QtCore.Signal(object)
    stage_changed = QtCore.Signal(object)
    stage_outcome = QtCore.Signal(object)
    progress_snapshot = QtCore.Signal(object)
    structured_error = QtCore.Signal(object)
    runtime_backend_selected = QtCore.Signal(object)
    progress_changed = QtCore.Signal(int)
    eta_changed = QtCore.Signal(str)
    page_changed = QtCore.Signal(int, int)
    message = QtCore.Signal(str)
    queue_reset = QtCore.Signal(list)
    queue_item = QtCore.Signal(int, str)
    total_time_changed = QtCore.Signal(str)
    page_time_changed = QtCore.Signal(str)
    page_ready = QtCore.Signal(int, dict)
    consistency_issue = QtCore.Signal(list)
    # Two-Pass Pipeline signals
    prescan_started = QtCore.Signal()
    prescan_progress = QtCore.Signal(int)
    prescan_finished = QtCore.Signal()

    def __init__(
        self,
        settings: PipelineSettings,
        parent=None,
        *,
        runtime_binding: PipelineRuntimeBinding | None = None,
        run_id: str | None = None,
    ) -> None:
        super().__init__(parent)
        if runtime_binding is not None and not isinstance(
            runtime_binding, PipelineRuntimeBinding
        ):
            raise TypeError("runtime_binding must be PipelineRuntimeBinding or None")
        self._settings = settings
        self._runtime_binding = runtime_binding
        self._run_id = str(run_id or new_run_id())
        self._current_stage = PipelineStage.IDLE
        self._terminal_lifecycle_emitted = False
        self._terminal_error_emitted = False
        self._pending_terminal_state: PipelineRunState | None = None
        self._pending_terminal_message = ""
        self._stop_requested = False
        self._checkpoint_session: ProjectCheckpointSession | None = None
        self._stage_outcomes_by_page: dict[str, list[PipelineStageOutcome]] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    def _emit_lifecycle(self, state: PipelineRunState, message: str = "") -> None:
        event = PipelineLifecycleEvent(
            run_id=self._run_id,
            state=state,
            message=message,
        )
        self.lifecycle_changed.emit(event)
        if state in {
            PipelineRunState.STOPPED,
            PipelineRunState.COMPLETED,
            PipelineRunState.FAILED,
        }:
            self._terminal_lifecycle_emitted = True

    def _queue_terminal_lifecycle(
        self,
        state: PipelineRunState,
        message: str = "",
    ) -> None:
        if state not in {
            PipelineRunState.STOPPED,
            PipelineRunState.COMPLETED,
            PipelineRunState.FAILED,
        }:
            raise ValueError("only terminal lifecycle states may be queued")
        self._pending_terminal_state = state
        self._pending_terminal_message = message

    def _flush_terminal_lifecycle(self) -> None:
        if self._terminal_lifecycle_emitted:
            return
        if self._pending_terminal_state is not None:
            self._emit_lifecycle(
                self._pending_terminal_state,
                self._pending_terminal_message,
            )
            return
        self._emit_failed_terminal_if_needed()

    def _emit_stage(
        self,
        stage: PipelineStage,
        detail: str = "",
        *,
        page_id: str = "",
        parent_id: str = "",
    ) -> None:
        self._current_stage = stage
        self.stage_changed.emit(
            PipelineStageEvent(
                run_id=self._run_id,
                stage=stage,
                page_id=page_id,
                parent_id=parent_id,
                detail=detail,
            )
        )

    def _record_stage_outcome(
        self,
        *,
        page_id: str,
        page_index: int,
        page_name: str,
        source_path: str,
        output_path: str,
        stage: PipelineStage,
        state: PipelineStageOutcomeState,
        parent_ids: Iterable[str] = (),
        artifact_kind: str = "",
        artifact_summary: Mapping[str, Any] | None = None,
        diagnostics: Iterable[str] = (),
        error_code: str = "",
        message: str = "",
        detail: str = "",
    ) -> PipelineStageOutcome:
        outcome = PipelineStageOutcome(
            run_id=self._run_id,
            page_id=str(page_id or ""),
            page_index=int(page_index),
            page_name=str(page_name or ""),
            source_path=str(source_path or ""),
            output_path=str(output_path or ""),
            stage=stage,
            state=state,
            parent_ids=tuple(str(value) for value in parent_ids if str(value)),
            artifact_kind=str(artifact_kind or ""),
            artifact_summary=_compact_stage_outcome_artifact_summary(
                stage,
                artifact_summary,
            ),
            diagnostics=tuple(str(value) for value in diagnostics if str(value)),
            error_code=str(error_code or ""),
            message=str(message or ""),
            detail=str(detail or ""),
        )
        self._stage_outcomes_by_page.setdefault(outcome.page_id, []).append(outcome)
        if self._checkpoint_session is not None:
            self._checkpoint_session.record_stage_outcome(outcome.to_dict())
        self.stage_outcome.emit(outcome)
        return outcome

    def _emit_progress_snapshot(
        self,
        *,
        completed_pages: int,
        total_pages: int,
        percent: int,
        eta_seconds: float | None,
        current_page_id: str = "",
        current_parent_id: str = "",
    ) -> None:
        self.progress_snapshot.emit(
            PipelineProgressSnapshot(
                run_id=self._run_id,
                completed_pages=completed_pages,
                total_pages=total_pages,
                percent=percent,
                stage=self._current_stage,
                eta_seconds=eta_seconds,
                current_page_id=current_page_id,
                current_parent_id=current_parent_id,
            )
        )

    def _emit_structured_error(
        self,
        *,
        code: str,
        owner_stage: PipelineStage,
        message: str,
        detail: str = "",
        page_id: str = "",
        parent_id: str = "",
        recoverable: bool,
        retry_action: PipelineRetryAction,
        operation: str = "",
        prior_state_safe: bool = True,
        terminal: bool = False,
    ) -> PipelineErrorReceipt:
        receipt = PipelineErrorReceipt(
            error_id=new_error_id(),
            run_id=self._run_id,
            code=code,
            owner_stage=owner_stage,
            message=message,
            detail=detail,
            page_id=page_id,
            parent_id=parent_id,
            recoverable=recoverable,
            retry_action=retry_action,
            operation=operation,
            prior_state_safe=prior_state_safe,
        )
        if terminal:
            self._terminal_error_emitted = True
        self.structured_error.emit(receipt)
        return receipt

    def _emit_failed_terminal_if_needed(self) -> None:
        if self._terminal_lifecycle_emitted:
            return
        if not self._terminal_error_emitted:
            self._emit_structured_error(
                code="pipeline_terminated",
                owner_stage=self._current_stage,
                message="The pipeline ended before completion.",
                detail="No typed terminal receipt was produced by the owning stage.",
                recoverable=True,
                retry_action=PipelineRetryAction.RETRY_RUN,
                operation="run",
                terminal=True,
            )
        self._emit_lifecycle(
            PipelineRunState.FAILED,
            "The pipeline ended before completion.",
        )

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        self._emit_stage(PipelineStage.VALIDATION, "Validating run inputs")
        images = _list_images(self._settings.import_dir)

        # Filter by whitelist if provided (for re-translation)
        if self._settings.files_whitelist:
            whitelist_names = set(os.path.basename(f) for f in self._settings.files_whitelist)
            # Find matching images in the import dir
            images = [img for img in images if os.path.basename(img) in whitelist_names]

        total = len(images)
        self.queue_reset.emit(images)
        if total == 0:
            message = "No images found in import folder."
            self._emit_structured_error(
                code="no_input_pages",
                owner_stage=PipelineStage.VALIDATION,
                message=message,
                recoverable=True,
                retry_action=PipelineRetryAction.RELINK,
                operation="enumerate_inputs",
                terminal=True,
            )
            self.message.emit(message)
            self._emit_lifecycle(PipelineRunState.FAILED, message)
            return
        if self._settings.fast_mode:
            self._settings.detector_engine = "ComicTextDetector"
            self._settings.inpaint_mode = "fast"
            self._settings.font_detection = "off"
            self._settings.filter_strength = "normal"
            self.message.emit("Fast Mode: detector=ComicTextDetector, inpaint=fast, font detection=off.")
        if not os.path.isdir(self._settings.export_dir):
            try:
                os.makedirs(self._settings.export_dir, exist_ok=True)
            except OSError as exc:
                message = "Failed to create export folder."
                self._emit_structured_error(
                    code="export_directory_unavailable",
                    owner_stage=PipelineStage.VALIDATION,
                    message=message,
                    detail=f"{type(exc).__name__}: {exc}",
                    recoverable=True,
                    retry_action=PipelineRetryAction.RELINK,
                    operation="create_export_directory",
                    terminal=True,
                )
                self.message.emit(message)
                self._emit_lifecycle(PipelineRunState.FAILED, message)
                return

        self._emit_stage(PipelineStage.INITIALIZATION, "Initializing selected runtime")
        start_time = time.time()
        from app.translate.ollama_client import DeepSeekClient, OllamaClient
        from app.render.renderer import render_parent_execution_bundles
        from app.render.font_manager import FontManager
        from app.pipeline.cleaned_page_base import persist_cleaned_page_base
        from app.pipeline.cleanup_contracts import build_cleanup_job_candidates_for_parent_bundles
        from app.pipeline.cleanup_masks import build_cleanup_masks
        from app.pipeline.cleanup_planning import (
            build_cleanup_plans,
            collect_cleanup_runtime_perf,
            commit_cleanup_runtime_results_to_working_image,
            run_cleanup_runtime_contract,
        )
        from app.pipeline.render_eligibility import build_render_eligibility_decisions_for_parent_bundles
        from app.pipeline.source_glyph_masks import generate_source_glyph_masks_for_parent_bundles
        from app.pipeline.text_area_plan import build_text_area_component_authorization_map
        from app.pipeline.debug_artifacts import (
            append_perf_timing_overhead_artifact,
            debug_enabled,
            debug_pages,
            debug_root,
            debug_stage_artifact_dir,
            mark_render_region,
            new_page_context,
            new_perf_page_context,
            page_matches,
            perf_telemetry_enabled,
            perf_telemetry_root,
            set_count,
            set_timing,
            write_cleanup_process_debug_artifacts,
            write_perf_timing_artifact,
            write_page_artifacts,
        )

        ocr_engine = None
        ollama = None
        auto_glossary_state = None
        pages = []
        checkpoint_session: ProjectCheckpointSession | None = None
        debug_artifacts_enabled = debug_enabled(self._settings)
        perf_telemetry_is_enabled = perf_telemetry_enabled(self._settings)
        debug_page_filter = debug_pages(self._settings) if debug_artifacts_enabled else set()
        debug_artifacts_root = debug_root(self._settings) if debug_artifacts_enabled else ""
        perf_telemetry_output_root = perf_telemetry_root(self._settings) if perf_telemetry_is_enabled else ""
        worker_initialization: dict[str, Any] = {}
        if debug_artifacts_enabled:
            self.message.emit(f"Debug artifacts enabled: {debug_artifacts_root}")
        if perf_telemetry_is_enabled:
            self.message.emit(f"Performance telemetry enabled: {perf_telemetry_output_root}")
        try:
            try:
                ocr_initialization_start = time.perf_counter()
                ocr_engine = _create_selected_ocr_engine(self._settings, self.message.emit)
                self._settings.ocr_engine = _normalize_ocr_engine_name(self._settings.ocr_engine)
                if perf_telemetry_is_enabled:
                    worker_initialization["ocr_engine_initialization_time"] = (
                        time.perf_counter() - ocr_initialization_start
                    )
            except Exception as inner_exc:
                message = _friendly_model_error(inner_exc)
                self._emit_structured_error(
                    code="ocr_initialization_failed",
                    owner_stage=PipelineStage.INITIALIZATION,
                    message=message,
                    detail=f"{type(inner_exc).__name__}: {inner_exc}",
                    recoverable=True,
                    retry_action=PipelineRetryAction.RETRY_RUN,
                    operation="initialize_ocr",
                    terminal=True,
                )
                self.message.emit(message)
                return

            try:
                detector_initialization_start = time.perf_counter()
                if self._settings.detector_engine != "ComicTextDetector":
                    self.message.emit(
                        f"Detector '{self._settings.detector_engine}' is no longer available; using ComicTextDetector."
                    )
                    self._settings.detector_engine = "ComicTextDetector"
                from app.detect.comic_text_detector import ComicTextDetector
                detector = ComicTextDetector(self._settings.use_gpu)
                if perf_telemetry_is_enabled:
                    worker_initialization["detector_initialization_time"] = (
                        time.perf_counter() - detector_initialization_start
                    )
            except Exception as exc:
                message = _friendly_model_error(exc)
                self._emit_structured_error(
                    code="detector_initialization_failed",
                    owner_stage=PipelineStage.INITIALIZATION,
                    message=message,
                    detail=f"{type(exc).__name__}: {exc}",
                    recoverable=True,
                    retry_action=PipelineRetryAction.RETRY_RUN,
                    operation="initialize_detector",
                    terminal=True,
                )
                self.message.emit(message)
                return
            background_detector = detector if not self._settings.filter_background else None

            try:
                translator_initialization_start = time.perf_counter()
                if self._settings.translator_backend == "GGUF":
                    from app.translate.gguf_client import GGUFClient
                    n_gpu_layers = self._settings.gguf_n_gpu_layers
                    # Auto-detect prompt style from filename if generic settings used
                    prompt_style = self._settings.gguf_prompt_style
                    if "sakura" in self._settings.gguf_model_path.lower() and prompt_style == "qwen":
                        prompt_style = "sakura"

                    ollama = GGUFClient(
                        model_path=self._settings.gguf_model_path,
                        prompt_style=prompt_style,
                        n_ctx=self._settings.gguf_n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        n_threads=self._settings.gguf_n_threads,
                        n_batch=self._settings.gguf_n_batch,
                    )
                    if n_gpu_layers != 0 and not getattr(ollama, "gpu_offload", True):
                        self.message.emit(
                            "GGUF is running in CPU mode. For speed, install a CUDA-enabled llama-cpp-python "
                            "build or switch to Ollama."
                        )
                elif self._settings.translator_backend == "DeepSeek":
                    runtime_api_key = (
                        self._runtime_binding.credential_for_backend("DeepSeek")
                        if self._runtime_binding is not None
                        else None
                    )
                    ollama = DeepSeekClient(
                        base_url=self._settings.deepseek_base_url,
                        model_name=self._settings.deepseek_model,
                        api_key=runtime_api_key,
                    )
                    if not ollama.is_available():
                        message = _deepseek_unavailable_message(self._runtime_binding)
                        self._emit_structured_error(
                            code="translation_provider_unavailable",
                            owner_stage=PipelineStage.INITIALIZATION,
                            message=message,
                            recoverable=True,
                            retry_action=PipelineRetryAction.RETRY_RUN,
                            operation="initialize_translation_provider",
                            terminal=True,
                        )
                        self.message.emit(message)
                        return
                else:
                    ollama = OllamaClient(
                        base_url=self._settings.ollama_base_url,
                        context_tokens=self._settings.ollama_context,
                    )
                    if not ollama.is_available():
                        message = "Ollama server is not running. Start it with: ollama serve"
                        self._emit_structured_error(
                            code="translation_provider_unavailable",
                            owner_stage=PipelineStage.INITIALIZATION,
                            message=message,
                            recoverable=True,
                            retry_action=PipelineRetryAction.RETRY_RUN,
                            operation="initialize_translation_provider",
                            terminal=True,
                        )
                        self.message.emit(message)
                        return
                if perf_telemetry_is_enabled:
                    worker_initialization["translator_initialization_time"] = (
                        time.perf_counter() - translator_initialization_start
                    )
            except Exception as exc:
                message = _friendly_model_error(exc)
                self._emit_structured_error(
                    code="translator_initialization_failed",
                    owner_stage=PipelineStage.INITIALIZATION,
                    message=message,
                    detail=f"{type(exc).__name__}: {exc}",
                    recoverable=True,
                    retry_action=PipelineRetryAction.RETRY_RUN,
                    operation="initialize_translator",
                    terminal=True,
                )
                self.message.emit(message)
                return
            if self._settings.translator_backend == "GGUF":
                model_name = self._settings.gguf_model_path
                resolved_model = model_name
            elif self._settings.translator_backend == "DeepSeek":
                model_name = self._settings.deepseek_model
                resolved_model = model_name
            else:
                model_name = self._settings.ollama_model
                resolved_model = _resolve_model(self._settings.ollama_model)
            if self._settings.translator_backend == "Ollama":
                if resolved_model and self._settings.ollama_model != "auto-detect":
                    available = list_models()
                    if available and resolved_model not in available:
                        message = f"Ollama model not found: {resolved_model}"
                        self._emit_structured_error(
                            code="translation_model_missing",
                            owner_stage=PipelineStage.INITIALIZATION,
                            message=message,
                            recoverable=True,
                            retry_action=PipelineRetryAction.RELINK,
                            operation="resolve_translation_model",
                            terminal=True,
                        )
                        self.message.emit(message)
                        return
            elif _missing_required_gguf_model_path(self._settings):
                message = "GGUF model path is required for GGUF backend."
                self._emit_structured_error(
                    code="translation_model_missing",
                    owner_stage=PipelineStage.INITIALIZATION,
                    message=message,
                    recoverable=True,
                    retry_action=PipelineRetryAction.RELINK,
                    operation="resolve_translation_model",
                    terminal=True,
                )
                self.message.emit(message)
                return

            # Ensure model name is set on the client for glossary translation
            if hasattr(ollama, "translate_glossary"):
                 # Use resolved model for Ollama, or path for GGUF (though GGUF uses internal path)
                 setattr(ollama, "model_name", resolved_model or model_name)

            if self._settings.auto_glossary and not self._settings.style_guide_path:
                self._settings.style_guide_path = os.path.join(self._settings.export_dir, "style_guide.json")
            style_guide = _load_style_guide(self._settings.style_guide_path, self._settings.target_lang)
            if self._settings.auto_glossary and self._settings.style_guide_path and not os.path.isfile(self._settings.style_guide_path):
                try:
                    from app.io.style_guide import save_style_guide
                    save_style_guide(self._settings.style_guide_path, style_guide)
                except Exception:
                    pass
            context_window = []
            translation_cache: dict[str, str] = {}
            project = default_project_dict()
            project["project"]["name"] = os.path.basename(self._settings.import_dir.rstrip("\\/"))
            project["project"]["language"]["source"] = _lang_code(self._settings.source_lang)
            project["project"]["language"]["target"] = _lang_code(self._settings.target_lang)
            project["project"]["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            project["project"]["model"]["detector"] = self._settings.detector_engine
            project["project"]["model"]["ocr"] = self._settings.ocr_engine
            if self._settings.translator_backend == "GGUF":
                project["project"]["model"]["translator"] = f"gguf:{self._settings.gguf_model_path}"
            elif self._settings.translator_backend == "DeepSeek":
                project["project"]["model"]["translator"] = f"deepseek:{self._settings.deepseek_model}"
            else:
                project["project"]["model"]["translator"] = f"ollama:{self._settings.ollama_model}"
            project["project"]["style_guide"] = self._settings.style_guide_path or ""
            json_path = self._settings.json_path or os.path.join(
                self._settings.export_dir,
                "project.json",
            )
            from app.pipeline.parent_font_detection import (
                PARENT_STYLE_DECISION_LEDGER_VERSION,
                YUZUMARKER_PROVIDER_MODEL,
                YuzuMarkerOnnxFontDetector,
                resolve_yuzumarker_font_labels_file,
                resolve_yuzumarker_font_onnx_file,
            )
            from app.pipeline.parent_style_evidence import (
                AUTHORIZED_SOURCE_STYLE_VIEW_VERSION,
                SOURCE_STYLE_AXIS_EVIDENCE_VERSION,
            )
            from app.render.font_manager import FONT_MANAGER_VERSION

            font_detection_mode = str(
                self._settings.font_detection or "off"
            ).strip().lower()
            parent_style_detector = None
            parent_style_detector_initialization_attempted = False
            style_context_policy_identity = build_style_context_policy_identity(
                {
                    "observer_version": AUTHORIZED_SOURCE_STYLE_VIEW_VERSION,
                    "axis_evidence_version": SOURCE_STYLE_AXIS_EVIDENCE_VERSION,
                    "arbitrator_version": PARENT_STYLE_DECISION_LEDGER_VERSION,
                    "font_registry_version": FONT_MANAGER_VERSION,
                    "font_detection_mode": font_detection_mode,
                    "model_identity": (
                        YUZUMARKER_PROVIDER_MODEL
                        if font_detection_mode == "yuzumarker"
                        else font_detection_mode
                    ),
                }
            )
            style_context_run_identity = build_style_context_run_identity(
                import_dir=self._settings.import_dir,
                page_names=images,
                source_language=_lang_code(self._settings.source_lang),
                target_language=_lang_code(self._settings.target_lang),
            )
            persisted_style_context_cache = None
            if os.path.isfile(json_path):
                try:
                    persisted_project = load_project(json_path)
                    if isinstance(persisted_project, Mapping):
                        persisted_style_context_cache = persisted_project.get(
                            "style_context_cache"
                        )
                except Exception as exc:
                    self.message.emit(
                        "Prior style context is unreadable; continuing with "
                        f"an empty prefix ({type(exc).__name__})."
                    )
            style_context_load = load_style_context_journal(
                persisted_style_context_cache,
                run_identity=style_context_run_identity,
                policy_identity=style_context_policy_identity,
            )
            style_context_journal = style_context_load.journal
            project["style_context_cache"] = (
                style_context_journal.to_project_dict()
            )
            auto_glossary_state = None
            if self._settings.auto_glossary:
                auto_glossary_state = {"counts": {}, "map": {}}

            # Pre-Scan Mode: Build complete glossary before translation
            if self._settings.prescan_enabled and self._settings.auto_glossary:
                self._emit_stage(PipelineStage.PRESCAN, "Building the run glossary")
                self.prescan_started.emit()
                self.message.emit("Pre-Scan Mode: Building glossary before translation...")
                try:
                    from app.pipeline.prescan import prescan_for_glossary
                    style_guide = prescan_for_glossary(
                        import_dir=self._settings.import_dir,
                        images=images,
                        style_guide=style_guide,
                        settings=self._settings,
                        progress_callback=lambda p: self.prescan_progress.emit(p),
                        message_callback=lambda m: self.message.emit(f"[Pre-Scan] {m}"),
                        stop_check=lambda: self._stop_requested,
                        translator=ollama,
                        detector=detector,
                        ocr_engine=ocr_engine,
                    )
                    # Save the updated style guide
                    if self._settings.style_guide_path:
                        from app.io.style_guide import save_style_guide
                        save_style_guide(self._settings.style_guide_path, style_guide)
                    self.message.emit(f"Pre-Scan complete: {len(style_guide.get('glossary', []))} glossary entries.")
                except Exception as e:
                    message = f"Pre-Scan failed: {e}. Continuing with normal translation."
                    self._emit_structured_error(
                        code="prescan_failed",
                        owner_stage=PipelineStage.PRESCAN,
                        message=message,
                        detail=f"{type(e).__name__}: {e}",
                        recoverable=False,
                        retry_action=PipelineRetryAction.NONE,
                        operation="prescan",
                    )
                    self.message.emit(message)
                    import logging
                    logging.getLogger(__name__).exception("Pre-scan error")
                finally:
                    self.prescan_finished.emit()
            if self._settings.prescan_only:
                self.message.emit("Pre-Scan only mode complete.")
                self._queue_terminal_lifecycle(
                    PipelineRunState.COMPLETED,
                    "Pre-Scan only mode complete.",
                )
                return
            cleanup_model_prewarmed = False
            cleanup_model_warmup_record: dict[str, Any] = {}
            cleanup_model_warmup_elapsed = 0.0
            if self._settings.use_gpu and str(self._settings.inpaint_mode or "").strip().lower() != "off":
                warmup_started = time.time()
                try:
                    from app.inpaint.simple_lama_engine import warm_cleanup_inpaint_model

                    cleanup_model_warmup_record = warm_cleanup_inpaint_model(
                        use_gpu=self._settings.use_gpu,
                        model_id=self._settings.inpaint_model_id,
                    )
                except Exception as exc:
                    cleanup_model_warmup_record = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                cleanup_model_warmup_elapsed = time.time() - warmup_started
                cleanup_model_warmup_record = {
                    "stage": "pre_page_loop_cleanup_model_warmup",
                    "elapsed_ms": round(cleanup_model_warmup_elapsed * 1000.0, 3),
                    **cleanup_model_warmup_record,
                }
                cleanup_model_prewarmed = str(cleanup_model_warmup_record.get("status") or "") in {
                    "warmed",
                    "already_warmed",
                }
                _pipeline_runtime_checkpoint(
                    "cleanup_model_pre_page_warmup",
                    "end",
                    status=str(cleanup_model_warmup_record.get("status") or ""),
                    elapsed_ms=cleanup_model_warmup_record.get("elapsed_ms"),
                )
                if perf_telemetry_is_enabled and perf_telemetry_output_root:
                    try:
                        os.makedirs(perf_telemetry_output_root, exist_ok=True)
                        with open(
                            os.path.join(perf_telemetry_output_root, "cleanup_model_warmup.json"),
                            "w",
                            encoding="utf-8",
                        ) as handle:
                            json.dump(cleanup_model_warmup_record, handle, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                if cleanup_model_prewarmed:
                    backend_event = _cleanup_runtime_backend_event(
                        self._run_id,
                        cleanup_model_warmup_record,
                    )
                    self.runtime_backend_selected.emit(backend_event)
                    if backend_event.fallback_reason:
                        self.message.emit(
                            "Cleanup selected CPU after the requested MPS backend "
                            f"failed ({backend_event.fallback_reason})."
                        )
                    else:
                        self.message.emit(
                            "Cleanup model warmup completed on "
                            f"{backend_event.selected_backend.upper()} before first "
                            "page "
                            f"({cleanup_model_warmup_record.get('elapsed_ms')} ms)."
                        )
            if perf_telemetry_is_enabled:
                worker_initialization["pre_page_setup_time"] = time.time() - start_time
                worker_initialization["cleanup_model_warmup_time"] = cleanup_model_warmup_elapsed
            try:
                checkpoint_session = ProjectCheckpointSession(
                    json_path=json_path,
                    base_project=project,
                )
            except Exception as exc:
                message = (
                    "Failed to initialize incremental project persistence: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._emit_structured_error(
                    code="checkpoint_initialization_failed",
                    owner_stage=PipelineStage.PERSISTENCE,
                    message=message,
                    detail=f"{type(exc).__name__}: {exc}",
                    recoverable=True,
                    retry_action=PipelineRetryAction.RETRY_RUN,
                    operation="initialize_checkpoint",
                    terminal=True,
                )
                self.message.emit(message)
                return
            self._checkpoint_session = checkpoint_session
            style_font_manager = None
            for index, name in enumerate(images, start=1):
                if self._stop_requested:
                    self.message.emit("Stopped")
                    self._queue_terminal_lifecycle(
                        PipelineRunState.STOPPED,
                        "Stopped at the next safe page boundary.",
                    )
                    return

                page_start = time.time()
                page_process_cpu_start = time.process_time() if perf_telemetry_is_enabled else 0.0
                self.queue_item.emit(index - 1, "processing")
                self.page_changed.emit(index, total)

                source_path = os.path.join(self._settings.import_dir, name)
                output_path = build_output_path(self._settings.export_dir, name, self._settings.output_suffix)
                page_id = os.path.splitext(name)[0]
                self._emit_stage(
                    PipelineStage.DETECTION,
                    "Preparing page analysis",
                    page_id=page_id,
                )
                self._emit_progress_snapshot(
                    completed_pages=index - 1,
                    total_pages=total,
                    percent=int((index - 1) / total * 100),
                    eta_seconds=None,
                    current_page_id=page_id,
                )
                style_context_snapshot = style_context_snapshot_before(
                    style_context_journal,
                    page_index=index - 1,
                )
                style_context_delta = None
                style_context_candidate_journal = style_context_journal
                debug_context = None
                if debug_artifacts_enabled and page_matches(name, debug_page_filter):
                    debug_context = new_page_context(
                        name,
                        source_path,
                        output_path,
                        debug_artifacts_root,
                        settings=self._settings,
                    )
                elif perf_telemetry_is_enabled:
                    debug_context = new_perf_page_context(name, source_path, output_path, perf_telemetry_output_root)
                if debug_context is not None and perf_telemetry_is_enabled and index == 1:
                    debug_context["worker_initialization"] = dict(worker_initialization)
                if debug_context is not None and cleanup_model_warmup_record:
                    debug_context["cleanup_model_warmup"] = cleanup_model_warmup_record
                    set_timing(
                        debug_context,
                        "cleanup_model_pre_page_warmup_time",
                        cleanup_model_warmup_elapsed if index == 1 else 0.0,
                    )

                try:
                    _pipeline_runtime_checkpoint(
                        "controller_process_page",
                        "start",
                        page_name=name,
                        source_path=source_path,
                        output_path=output_path,
                    )
                    process_page_start = time.time()
                    page_result = _process_page(
                        source_path,
                        detector,
                        ocr_engine,
                        ollama,
                        model_name,
                        style_guide,
                        context_window,
                        self._settings.target_lang,
                        self._settings.source_lang,
                        self._settings.font_name,
                        self._settings.filter_background,
                        self._settings.filter_strength,
                        translation_cache,
                        background_detector,

                        auto_glossary_state,
                        image_input_size=self._settings.detector_input_size,
                        style_guide_path=self._settings.style_guide_path,
                        allow_ollama_discovery=self._settings.use_ollama_discovery,
                        discovery_model=self._settings.discovery_model,
                        settings=self._settings,
                        debug_context=debug_context,
                        stage_callback=lambda stage, detail: self._emit_stage(
                            stage,
                            detail,
                            page_id=page_id,
                        ),
                        stage_outcome_callback=lambda **fields: self._record_stage_outcome(
                            page_id=page_id,
                            page_index=index - 1,
                            page_name=name,
                            source_path=source_path,
                            output_path=output_path,
                            **fields,
                        ),
                    )
                    regions = page_result.regions
                    execution_regions = page_result.execution_regions
                    parent_execution_bundles = page_result.parent_execution_bundles
                    page_class = page_result.page_class
                    text_area_plan = page_result.text_area_plan
                    ctd_segmentation_result = page_result.ctd_segmentation_result
                    _pipeline_runtime_checkpoint(
                        "controller_process_page",
                        "end",
                        page_name=name,
                        region_count=len(regions) if regions is not None else 0,
                        execution_region_count=len(execution_regions) if execution_regions is not None else 0,
                        parent_execution_bundle_count=len(parent_execution_bundles),
                        page_class=page_class,
                        elapsed_ms=round((time.time() - process_page_start) * 1000.0, 3),
                    )
                    _pipeline_runtime_checkpoint(
                        "post_ocr_detection_hierarchy",
                        "end",
                        page_name=name,
                        page_id=os.path.splitext(name)[0],
                        region_count=len(regions) if regions is not None else 0,
                        execution_region_count=len(execution_regions) if execution_regions is not None else 0,
                        page_class=page_class,
                    )
                    if debug_context is not None:
                        debug_context["page_class"] = page_class
                    if auto_glossary_state is not None:
                        new_client = auto_glossary_state.pop("translation_client", None)
                        if new_client is not None and new_client is not ollama:
                            ollama = new_client
                except Exception as exc:
                    _pipeline_runtime_checkpoint(
                        "controller_process_page",
                        "error",
                        page_name=name,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    page_elapsed = time.time() - page_start
                    self.queue_item.emit(index - 1, f"error ({_format_seconds(page_elapsed)}): {exc}")
                    technical = (
                        exc
                        if isinstance(exc, PipelineStageTechnicalError)
                        else PipelineStageTechnicalError(
                            stage=self._current_stage,
                            code=f"{self._current_stage.value}_technical_failure",
                            message=f"{self._current_stage.value} could not produce a valid artifact.",
                            detail=f"{type(exc).__name__}: {exc}",
                            page_id=page_id,
                            operation="process_page",
                        )
                    )
                    try:
                        self._record_stage_outcome(
                            page_id=page_id,
                            page_index=index - 1,
                            page_name=name,
                            source_path=source_path,
                            output_path=output_path,
                            stage=technical.stage,
                            state=PipelineStageOutcomeState.TECHNICAL_FAILURE,
                            parent_ids=([technical.parent_id] if technical.parent_id else ()),
                            artifact_kind="technical_failure_evidence",
                            artifact_summary=technical.artifact_summary,
                            diagnostics=technical.diagnostics,
                            error_code=technical.code,
                            message=technical.message,
                            detail=technical.detail,
                        )
                    except Exception as persistence_exc:
                        message = (
                            "Failed to persist the owning stage failure for "
                            f"{name}: {type(persistence_exc).__name__}: {persistence_exc}"
                        )
                        self._emit_structured_error(
                            code="stage_outcome_persistence_failed",
                            owner_stage=PipelineStage.PERSISTENCE,
                            message=message,
                            detail=f"{type(persistence_exc).__name__}: {persistence_exc}",
                            page_id=page_id,
                            recoverable=True,
                            retry_action=PipelineRetryAction.RETRY_RUN,
                            operation="record_stage_outcome",
                            terminal=True,
                        )
                        self.message.emit(message)
                        return
                    message = f"Failed to process {name}: {technical.message}"
                    self._emit_structured_error(
                        code=technical.code,
                        owner_stage=technical.stage,
                        message=message,
                        detail=technical.detail,
                        page_id=page_id,
                        parent_id=technical.parent_id,
                        recoverable=True,
                        retry_action=PipelineRetryAction.RETRY_PAGE,
                        operation=technical.operation or "process_page",
                        terminal=True,
                    )
                    self.message.emit(message)
                    return

                source_glyph_mask_result = None
                self._emit_stage(
                    PipelineStage.SOURCE_GLYPH,
                    "Building source-glyph evidence",
                    page_id=page_id,
                )
                try:
                    source_glyph_start = time.time()
                    _pipeline_runtime_checkpoint("sourceglyph_generation", "start", page_id=page_id)
                    source_glyph_mask_result = generate_source_glyph_masks_for_parent_bundles(
                        page_id=page_id,
                        image_path=source_path,
                        parent_execution_bundles=parent_execution_bundles,
                    )
                    _pipeline_runtime_checkpoint(
                        "sourceglyph_generation",
                        "end",
                        page_id=page_id,
                        mask_count=len(source_glyph_mask_result.masks_by_region),
                        elapsed_ms=round((time.time() - source_glyph_start) * 1000.0, 3),
                    )
                    if debug_context is not None:
                        if not debug_context.get("perf_telemetry_only"):
                            debug_context["source_glyph_masks"] = source_glyph_mask_result.to_audit_dict()
                        set_timing(debug_context, "source_glyph_mask_time", time.time() - source_glyph_start)
                        set_count(debug_context, "source_glyph_masks", len(source_glyph_mask_result.masks_by_region))
                        if not debug_context.get("perf_telemetry_only"):
                            for rid, fields in source_glyph_mask_result.region_audit_fields().items():
                                render_fields = dict(fields)
                                render_fields.pop("region_id", None)
                                render_fields.pop("page_id", None)
                                mark_render_region(debug_context, rid, **render_fields)
                    self._record_stage_outcome(
                        page_id=page_id,
                        page_index=index - 1,
                        page_name=name,
                        source_path=source_path,
                        output_path=output_path,
                        stage=PipelineStage.SOURCE_GLYPH,
                        state=PipelineStageOutcomeState.VALID,
                        parent_ids=(bundle.parent_id for bundle in parent_execution_bundles),
                        artifact_kind="source_glyph_evidence",
                        artifact_summary=source_glyph_mask_result.to_audit_dict(),
                    )
                except Exception as exc:
                    _pipeline_runtime_checkpoint(
                        "sourceglyph_generation",
                        "error",
                        page_id=page_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    if debug_context is not None:
                        debug_context["source_glyph_masks"] = {
                            "source_glyph_mask_version": "source_glyph_masks_v1",
                            "source_glyph_mask_generated": False,
                            "source_glyph_mask_errors": [f"{type(exc).__name__}: {exc}"],
                            "source_glyph_masks": [],
                        }
                    self._record_stage_outcome(
                        page_id=page_id,
                        page_index=index - 1,
                        page_name=name,
                        source_path=source_path,
                        output_path=output_path,
                        stage=PipelineStage.SOURCE_GLYPH,
                        state=PipelineStageOutcomeState.VALID_WITH_DIAGNOSTICS,
                        parent_ids=(bundle.parent_id for bundle in parent_execution_bundles),
                        artifact_kind="source_glyph_unavailable_evidence",
                        artifact_summary=dict(debug_context.get("source_glyph_masks") or {})
                        if isinstance(debug_context, dict)
                        else {},
                        diagnostics=(f"source_glyph_unavailable:{type(exc).__name__}",),
                    )
                    self.message.emit(
                        f"Source-glyph evidence is unavailable for {name}; cleanup will use its parent-owned fallback."
                    )

                cleanup_job_contract_result = None
                cleanup_mask_contract_result = None
                cleanup_plan_contract_result = None
                cleanup_runtime_contract_result = None
                cleanup_upstream_commit_result = None
                render_eligibility_contract_result = None
                render_input_path = source_path
                cleaned_page_base_record = {}
                cleanup_upstream_temp_path = ""
                source_image_size = None
                self._emit_stage(
                    PipelineStage.CLEANUP,
                    "Building and applying cleanup contracts",
                    page_id=page_id,
                )
                try:
                    cleanup_contract_start = time.time()
                    _pipeline_runtime_checkpoint("cleanup_contract_chain", "start", page_id=page_id)
                    cleanup_job_start = time.time()
                    _pipeline_runtime_checkpoint(
                        "cleanup_job_build",
                        "start",
                        page_id=page_id,
                        parent_execution_bundle_count=len(parent_execution_bundles),
                    )
                    cleanup_job_contract_result = build_cleanup_job_candidates_for_parent_bundles(
                        page_id=page_id,
                        parent_execution_bundles=parent_execution_bundles,
                        source_glyph_masks=source_glyph_mask_result,
                    )
                    _pipeline_runtime_checkpoint(
                        "cleanup_job_build",
                        "end",
                        page_id=page_id,
                        job_count=len(cleanup_job_contract_result.jobs),
                        elapsed_ms=round((time.time() - cleanup_job_start) * 1000.0, 3),
                    )
                    source_image_size = _get_image_size(source_path)
                    if not source_image_size or source_image_size[0] <= 0 or source_image_size[1] <= 0:
                        source_image_size = None
                    segmentation_start = time.time()
                    _pipeline_runtime_checkpoint("text_foreground_segmentation", "start", page_id=page_id)
                    text_foreground_segmentation_mask = _build_text_foreground_segmentation_mask(
                        detector=detector,
                        source_path=source_path,
                        image_size=source_image_size,
                        input_size=int(getattr(self._settings, "detector_input_size", 1024) or 1024),
                        page_id=page_id,
                        debug_context=debug_context,
                        text_area_plan=text_area_plan,
                        segmentation_result=ctd_segmentation_result,
                    )
                    _pipeline_runtime_checkpoint(
                        "text_foreground_segmentation",
                        "end",
                        page_id=page_id,
                        text_pixel_count=getattr(text_foreground_segmentation_mask, "text_pixel_count", 0),
                        elapsed_ms=round((time.time() - segmentation_start) * 1000.0, 3),
                    )
                    cleanup_mask_start = time.time()
                    _pipeline_runtime_checkpoint(
                        "cleanup_mask_build",
                        "start",
                        page_id=page_id,
                        job_count=len(getattr(cleanup_job_contract_result, "jobs", []) or []),
                        source_glyph_record_count=len(getattr(source_glyph_mask_result, "masks_by_region", {}) or {}),
                    )
                    cleanup_mask_region_records = _cleanup_mask_region_records_with_protection(
                        execution_regions,
                        debug_context,
                        text_area_plan=text_area_plan,
                    )
                    component_authorization_start = time.time()
                    component_authorization_map = build_text_area_component_authorization_map(
                        page_id=page_id,
                        text_foreground_segmentation=text_foreground_segmentation_mask,
                        text_area_plan=text_area_plan,
                        page_region_records=cleanup_mask_region_records,
                        cleanup_jobs=cleanup_job_contract_result.jobs,
                    )
                    component_authorization_elapsed = time.time() - component_authorization_start
                    _pipeline_runtime_checkpoint(
                        "text_area_component_authorization",
                        "end",
                        page_id=page_id,
                        component_count=len(component_authorization_map.components),
                        elapsed_ms=round(component_authorization_elapsed * 1000.0, 3),
                    )
                    if debug_context is not None:
                        if not debug_context.get("perf_telemetry_only"):
                            debug_context["text_area_component_authorization_map"] = component_authorization_map.to_audit_dict()
                        set_timing(
                            debug_context,
                            "text_area_component_authorization_time",
                            component_authorization_elapsed,
                        )
                        set_count(
                            debug_context,
                            "text_area_component_authorization_components",
                            len(component_authorization_map.components),
                        )
                    cleanup_mask_module_start = time.time()
                    cleanup_mask_contract_result = build_cleanup_masks(
                        page_id=page_id,
                        job_candidates=cleanup_job_contract_result.jobs,
                        source_glyph_masks=source_glyph_mask_result,
                        image_size=source_image_size,
                        source_image_path=source_path,
                        text_foreground_segmentation=text_foreground_segmentation_mask,
                        page_region_records=cleanup_mask_region_records,
                        component_authorization_map=component_authorization_map,
                    )
                    cleanup_mask_module_elapsed = time.time() - cleanup_mask_module_start
                    if debug_context is not None:
                        set_timing(debug_context, "cleanup_mask_build_time", cleanup_mask_module_elapsed)
                        if not debug_context.get("perf_telemetry_only"):
                            write_cleanup_process_debug_artifacts(
                                debug_context,
                                source_image_path=source_path,
                                image_size=source_image_size,
                                text_foreground_segmentation=text_foreground_segmentation_mask,
                                component_authorization_map=component_authorization_map,
                                source_glyph_masks=source_glyph_mask_result,
                                cleanup_job_contracts=cleanup_job_contract_result,
                                cleanup_mask_contracts=cleanup_mask_contract_result,
                            )
                    _pipeline_runtime_checkpoint(
                        "cleanup_mask_build",
                        "end",
                        page_id=page_id,
                        mask_count=len(getattr(cleanup_mask_contract_result, "masks", []) or []),
                        rejected_count=len(getattr(cleanup_mask_contract_result, "rejected_records", []) or []),
                        elapsed_ms=round((time.time() - cleanup_mask_start) * 1000.0, 3),
                    )
                    render_eligibility_start = time.time()
                    _pipeline_runtime_checkpoint("render_eligibility_build", "start", page_id=page_id)
                    render_eligibility_contract_result = build_render_eligibility_decisions_for_parent_bundles(
                        page_id=page_id,
                        parent_execution_bundles=parent_execution_bundles,
                        source_glyph_masks=source_glyph_mask_result,
                        cleanup_job_contracts=cleanup_job_contract_result,
                        cleanup_mask_contracts=cleanup_mask_contract_result,
                        source_image_path=source_path,
                        image_size=source_image_size,
                    )
                    render_eligibility_elapsed = time.time() - render_eligibility_start
                    _pipeline_runtime_checkpoint(
                        "render_eligibility_build",
                        "end",
                        page_id=page_id,
                        decision_count=len(getattr(render_eligibility_contract_result, "decisions", []) or []),
                        diagnostic_count=len(getattr(render_eligibility_contract_result, "diagnostic_records", []) or []),
                        elapsed_ms=round(render_eligibility_elapsed * 1000.0, 3),
                    )
                    if debug_context is not None:
                        set_timing(debug_context, "render_eligibility_contract_time", render_eligibility_elapsed)
                    cleanup_plan_start = time.time()
                    cleanup_plan_mask_contracts = cleanup_mask_contract_result
                    _pipeline_runtime_checkpoint(
                        "cleanup_plan_build",
                        "start",
                        page_id=page_id,
                        job_count=len(getattr(cleanup_job_contract_result, "jobs", []) or []),
                        mask_count=len(getattr(cleanup_plan_mask_contracts, "masks", []) or []),
                    )
                    cleanup_plan_contract_result = build_cleanup_plans(
                        page_id=page_id,
                        job_candidates=cleanup_job_contract_result.jobs,
                        mask_contracts=cleanup_plan_mask_contracts,
                        image_size=source_image_size,
                        source_image_path=source_path,
                        render_eligibility=render_eligibility_contract_result,
                        inpaint_mode=self._settings.inpaint_mode,
                    )
                    cleanup_plan_elapsed = time.time() - cleanup_plan_start
                    _pipeline_runtime_checkpoint(
                        "cleanup_plan_build",
                        "end",
                        page_id=page_id,
                        plan_count=len(getattr(cleanup_plan_contract_result, "plans", []) or []),
                        elapsed_ms=round(cleanup_plan_elapsed * 1000.0, 3),
                    )
                    if debug_context is not None:
                        set_timing(debug_context, "cleanup_plan_build_time", cleanup_plan_elapsed)
                    cleanup_runtime_start = time.time()
                    runtime_artifact_dir = None
                    upstream_commit_artifact_dir = None
                    if debug_context is not None and not debug_context.get("perf_telemetry_only"):
                        runtime_artifact_dir = debug_stage_artifact_dir(
                            debug_context,
                            "cleanup_runtime",
                            "cleanup_runtime_contracts",
                        )
                        upstream_commit_artifact_dir = debug_stage_artifact_dir(
                            debug_context,
                            "cleanup_commit",
                            "cleanup_upstream_commit",
                        )
                    try:
                        from PIL import Image
                        with Image.open(source_path) as runtime_source:
                            runtime_source_image = runtime_source.convert("RGB")
                        _pipeline_runtime_checkpoint("cleanup_runtime_contract", "start", page_id=page_id)
                        with collect_cleanup_runtime_perf(
                            bool(debug_context and debug_context.get("perf_telemetry_only"))
                        ) as cleanup_runtime_perf:
                            cleanup_runtime_contract_result = run_cleanup_runtime_contract(
                                page_id=page_id,
                                image=runtime_source_image.copy(),
                                source_image=runtime_source_image.copy(),
                                job_candidates=cleanup_job_contract_result.jobs,
                                mask_contracts=cleanup_mask_contract_result,
                                plan_contracts=cleanup_plan_contract_result,
                                render_eligibility=render_eligibility_contract_result,
                                use_gpu=self._settings.use_gpu,
                                model_id=self._settings.inpaint_model_id,
                                artifact_dir=runtime_artifact_dir,
                                inpaint_mode=self._settings.inpaint_mode,
                                prewarmed_cleanup_model=cleanup_model_prewarmed,
                            )
                        cleanup_runtime_elapsed = time.time() - cleanup_runtime_start
                        _pipeline_runtime_checkpoint(
                            "cleanup_runtime_contract",
                            "end",
                            page_id=page_id,
                            status_count=len(cleanup_runtime_contract_result.status_records),
                            result_count=len(cleanup_runtime_contract_result.result_records),
                            proof_count=len(cleanup_runtime_contract_result.proof_records),
                            elapsed_ms=round(cleanup_runtime_elapsed * 1000.0, 3),
                        )
                        if debug_context is not None:
                            set_timing(debug_context, "cleanup_runtime_contract_time", cleanup_runtime_elapsed)
                            _attach_cleanup_runtime_perf_summary(
                                debug_context,
                                cleanup_runtime_contract_result,
                                runtime_elapsed_seconds=cleanup_runtime_elapsed,
                                cleanup_runtime_perf=cleanup_runtime_perf,
                            )
                        render_eligibility_contract_result = _apply_cleanup_runtime_render_blocks(
                            render_eligibility_contract_result,
                            cleanup_runtime_contract_result,
                            debug_context,
                        )
                        commit_start = time.time()
                        _pipeline_runtime_checkpoint("cleanup_upstream_commit", "start", page_id=page_id)
                        cleanup_upstream_commit_result = commit_cleanup_runtime_results_to_working_image(
                            page_id=page_id,
                            source_image=runtime_source_image.copy(),
                            runtime_contract=cleanup_runtime_contract_result,
                            artifact_dir=upstream_commit_artifact_dir,
                            excluded_region_ids=_phase5_upstream_protected_region_ids(page_id),
                        )
                        commit_elapsed = time.time() - commit_start
                        _pipeline_runtime_checkpoint(
                            "cleanup_upstream_commit",
                            "end",
                            page_id=page_id,
                            committed_count=len(cleanup_upstream_commit_result.commit_records),
                            blocked_count=len(cleanup_upstream_commit_result.blocked_records),
                            elapsed_ms=round(commit_elapsed * 1000.0, 3),
                        )
                        if debug_context is not None:
                            set_timing(debug_context, "cleanup_upstream_commit_time", commit_elapsed)
                        render_eligibility_contract_result = _apply_cleanup_upstream_commit_render_blocks(
                            render_eligibility_contract_result,
                            cleanup_upstream_commit_result,
                            debug_context,
                        )
                        cleanup_diagnostics = _validate_cleanup_parent_conservation(
                            page_id=page_id,
                            bundles=parent_execution_bundles,
                            cleanup_jobs=cleanup_job_contract_result,
                            cleanup_masks=cleanup_mask_contract_result,
                            cleanup_plans=cleanup_plan_contract_result,
                            cleanup_runtime=cleanup_runtime_contract_result,
                            cleanup_commit=cleanup_upstream_commit_result,
                        )
                        cleanup_required_for_cleaned_base = bool(
                            getattr(cleanup_job_contract_result, "jobs", []) or []
                        ) or bool(getattr(cleanup_mask_contract_result, "masks", []) or []) or bool(
                            getattr(cleanup_plan_contract_result, "plans", []) or []
                        ) or bool(getattr(cleanup_runtime_contract_result, "status_records", []) or []) or any(
                            bool(getattr(bundle, "cleanup_required", False))
                            for bundle in parent_execution_bundles or []
                        )
                        cleaned_page_base_record = persist_cleaned_page_base(
                            page_id=page_id,
                            source_path=source_path,
                            export_dir=self._settings.export_dir,
                            cleanup_upstream_commit_result=cleanup_upstream_commit_result,
                            parent_execution_bundles=parent_execution_bundles,
                            cleanup_required=cleanup_required_for_cleaned_base,
                        )
                        if not bool(cleaned_page_base_record.get("valid")):
                            raise PipelineStageTechnicalError(
                                stage=PipelineStage.CLEANUP,
                                code="cleaned_page_base_invalid",
                                message="Cleanup did not publish a valid CleanedPageBase artifact.",
                                detail=str(
                                    (cleaned_page_base_record.get("invalidation") or {}).get("reason")
                                    or cleaned_page_base_record.get("state")
                                    or "invalid cleaned page base"
                                ),
                                page_id=page_id,
                                operation="persist_cleaned_page_base",
                                artifact_summary=cleaned_page_base_record,
                            )
                        cleaned_base_path = str(cleaned_page_base_record.get("image_path") or "")
                        if bool(cleaned_page_base_record.get("valid")) and cleaned_base_path and os.path.isfile(cleaned_base_path):
                            render_input_path = cleaned_base_path
                        elif cleanup_upstream_commit_result.commit_records:
                            with tempfile.NamedTemporaryFile(
                                prefix=f"phase5_upstream_{page_id}_",
                                suffix=".png",
                                delete=False,
                            ) as temp_file:
                                cleanup_upstream_temp_path = temp_file.name
                            cleanup_upstream_commit_result.cleaned_image.save(cleanup_upstream_temp_path)
                            render_input_path = cleanup_upstream_temp_path
                            cleaned_page_base_record["runtime_render_input_fallback_path"] = cleanup_upstream_temp_path
                            cleaned_page_base_record["runtime_render_input_fallback_reason"] = "cleaned_page_base_cache_unavailable"
                        if debug_context is not None:
                            debug_context["cleaned_page_base"] = dict(cleaned_page_base_record)
                            if not debug_context.get("perf_telemetry_only"):
                                write_cleanup_process_debug_artifacts(
                                    debug_context,
                                    source_image_path=source_path,
                                    image_size=source_image_size,
                                    text_foreground_segmentation=text_foreground_segmentation_mask,
                                    component_authorization_map=component_authorization_map,
                                    source_glyph_masks=source_glyph_mask_result,
                                    cleanup_job_contracts=cleanup_job_contract_result,
                                    cleanup_mask_contracts=cleanup_mask_contract_result,
                                    cleanup_plan_contracts=cleanup_plan_contract_result,
                                    cleanup_runtime_contracts=cleanup_runtime_contract_result,
                                    cleanup_upstream_commit_result=cleanup_upstream_commit_result,
                                    cleaned_page_base=cleaned_page_base_record,
                                )
                        if debug_context is not None and not debug_context.get("perf_telemetry_only"):
                            runtime_audit = cleanup_runtime_contract_result.to_audit_dict()
                            commit_audit = cleanup_upstream_commit_result.to_audit_dict()
                            debug_context["cleanup_runtime_status"] = runtime_audit
                            debug_context["cleanup_runtime_result_contracts"] = {
                                "version": runtime_audit.get("version"),
                                "page_id": page_id,
                                "renderer_consumed": False,
                                "results": runtime_audit.get("results", []),
                                "summary": {
                                    "result_count": len(cleanup_runtime_contract_result.result_records),
                                    "renderer_consumed": False,
                                },
                            }
                            debug_context["cleanup_runtime_proof_contracts"] = {
                                "version": runtime_audit.get("version"),
                                "page_id": page_id,
                                "renderer_consumed": False,
                                "proofs": runtime_audit.get("proofs", []),
                                "summary": {
                                    "proof_count": len(cleanup_runtime_contract_result.proof_records),
                                    "renderer_consumed": False,
                                },
                            }
                            debug_context["cleanup_upstream_commit_contracts"] = commit_audit
                            for status_record in cleanup_runtime_contract_result.status_records:
                                for rid in status_record.get("target_region_ids", []):
                                    mark_render_region(
                                        debug_context,
                                        str(rid),
                                        cleanup_runtime_class=status_record.get("cleanup_class"),
                                        cleanup_runtime_status=status_record.get("runtime_status"),
                                        cleanup_runtime_failure_reason=status_record.get("failure_reason"),
                                        cleanup_runtime_plan_id=status_record.get("cleanup_plan_id"),
                                        cleanup_runtime_result_id=status_record.get("cleanup_result_id"),
                                        cleanup_runtime_proof_id=status_record.get("cleanup_proof_id"),
                                        cleanup_runtime_renderer_consumed=False,
                                        cleanup_runtime_render_consumption_decision_if_consumed=status_record.get(
                                            "render_consumption_decision_if_consumed"
                                        ),
                                    )
                            for commit_record in cleanup_upstream_commit_result.commit_records:
                                mark_render_region(
                                    debug_context,
                                    str(commit_record.get("region_id") or ""),
                                    cleanup_applied_upstream=True,
                                    cleanup_committed_to_working_image=True,
                                    cleanup_upstream_commit_status="committed",
                                    cleanup_upstream_committed_pixel_count=commit_record.get("committed_pixel_count"),
                                    cleanup_upstream_cleaned_image_ref=commit_record.get(
                                        "cleanup_upstream_cleaned_image_ref"
                                    ),
                                    cleanup_upstream_diff_ref=commit_record.get("cleanup_upstream_diff_ref"),
                                    cleanup_upstream_mask_ref=commit_record.get("cleanup_upstream_mask_ref"),
                                    cleanup_runtime_renderer_consumed=False,
                                )
                            for blocked_record in cleanup_upstream_commit_result.blocked_records:
                                rid = str(blocked_record.get("region_id") or "")
                                if rid:
                                    mark_render_region(
                                        debug_context,
                                        rid,
                                        cleanup_applied_upstream=False,
                                        cleanup_committed_to_working_image=False,
                                        cleanup_upstream_commit_status="blocked",
                                        cleanup_upstream_commit_failure_reason=blocked_record.get("failure_reason"),
                                        cleanup_runtime_renderer_consumed=False,
                                    )
                            set_count(
                                debug_context,
                                "cleanup_runtime_result_count",
                                len(cleanup_runtime_contract_result.result_records),
                            )
                            set_count(
                                debug_context,
                                "cleanup_runtime_proof_count",
                                len(cleanup_runtime_contract_result.proof_records),
                            )
                            set_count(
                                debug_context,
                                "cleanup_upstream_commit_count",
                                len(cleanup_upstream_commit_result.commit_records),
                            )
                    except Exception as exc:
                        _pipeline_runtime_checkpoint(
                            "cleanup_runtime_or_commit",
                            "error",
                            page_id=page_id,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        if debug_context is not None and not debug_context.get("perf_telemetry_only"):
                            debug_context["cleanup_runtime_status"] = {
                                "version": "cleanup_runtime_phase5_speech_flat_bubble_result_proof",
                                "page_id": page_id,
                                "renderer_consumed": False,
                                "status_records": [],
                                "results": [],
                                "proofs": [],
                                "errors": [f"{type(exc).__name__}: {exc}"],
                                "summary": {
                                    "status_count": 0,
                                    "result_count": 0,
                                    "proof_count": 0,
                                    "renderer_consumed": False,
                                },
                            }
                            debug_context["cleanup_runtime_result_contracts"] = {
                                "version": "cleanup_runtime_phase5_speech_flat_bubble_result_proof",
                                "page_id": page_id,
                                "renderer_consumed": False,
                                "results": [],
                                "summary": {"result_count": 0, "renderer_consumed": False},
                            }
                            debug_context["cleanup_runtime_proof_contracts"] = {
                                "version": "cleanup_runtime_phase5_speech_flat_bubble_result_proof",
                                "page_id": page_id,
                                "renderer_consumed": False,
                                "proofs": [],
                                "summary": {"proof_count": 0, "renderer_consumed": False},
                            }
                            debug_context["cleanup_upstream_commit_contracts"] = {
                                "version": "cleanup_upstream_commit_phase5_pre_render_working_image",
                                "page_id": page_id,
                                "renderer_consumed": False,
                                "cleanup_applied_upstream": False,
                                "cleanup_committed_to_working_image": False,
                                "commit_records": [],
                                "blocked_records": [],
                                "errors": [f"{type(exc).__name__}: {exc}"],
                                "summary": {
                                    "committed_count": 0,
                                    "blocked_count": 0,
                                    "error_count": 1,
                                    "renderer_consumed": False,
                                },
                            }
                        raise
                    if debug_context is not None:
                        if debug_context.get("perf_telemetry_only"):
                            render_eligibility_audit = render_eligibility_contract_result.to_audit_dict()
                            debug_context["render_eligibility_contracts"] = {
                                "summary": render_eligibility_audit.get("summary", {}),
                                "diagnostic_records": render_eligibility_audit.get("diagnostic_records", []),
                            }
                        else:
                            debug_context["cleanup_job_contracts"] = cleanup_job_contract_result.to_audit_dict()
                            debug_context["cleanup_mask_contracts"] = cleanup_mask_contract_result.to_audit_dict()
                            debug_context["render_eligibility_contracts"] = render_eligibility_contract_result.to_audit_dict()
                            debug_context["cleanup_plan_contracts"] = cleanup_plan_contract_result.to_audit_dict()
                            debug_context["cleanup_backend_inventory"] = cleanup_plan_contract_result.backend_inventory
                        set_timing(debug_context, "cleanup_mask_contract_time", time.time() - cleanup_contract_start)
                        set_timing(debug_context, "render_eligibility_contract_time", render_eligibility_elapsed)
                        set_count(debug_context, "cleanup_job_contract_count", len(cleanup_job_contract_result.jobs))
                        set_count(debug_context, "cleanup_mask_contract_count", len(cleanup_mask_contract_result.masks))
                        set_count(debug_context, "render_eligibility_diagnostic_count", len(render_eligibility_contract_result.diagnostic_records))
                        set_count(debug_context, "cleanup_plan_contract_count", len(cleanup_plan_contract_result.plans))
                        set_count(debug_context, "cleanup_mask_rejected_count", len(cleanup_mask_contract_result.rejected_records))
                        set_count(debug_context, "cleanup_mask_protected_count", len(cleanup_mask_contract_result.protected_records))
                    _pipeline_runtime_checkpoint(
                        "cleanup_contract_chain",
                        "end",
                        page_id=page_id,
                        elapsed_ms=round((time.time() - cleanup_contract_start) * 1000.0, 3),
                    )
                except Exception as exc:
                    technical = (
                        exc
                        if isinstance(exc, PipelineStageTechnicalError)
                        else PipelineStageTechnicalError(
                            stage=PipelineStage.CLEANUP,
                            code="cleanup_technical_failure",
                            message="Cleanup could not produce a valid page artifact.",
                            detail=f"{type(exc).__name__}: {exc}",
                            page_id=page_id,
                            operation="run_cleanup",
                        )
                    )
                    _pipeline_runtime_checkpoint(
                        "cleanup_contract_chain",
                        "error",
                        page_id=page_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    if debug_context is not None:
                        debug_context["cleanup_mask_contracts"] = {
                            "version": "cleanup_masks_phase2",
                            "page_id": page_id,
                            "renderer_consumed": False,
                            "errors": [f"{type(exc).__name__}: {exc}"],
                            "summary": {
                                "mask_count": 0,
                                "rejected_record_count": 0,
                                "protected_record_count": 0,
                                "skipped_record_count": 0,
                                "error_count": 1,
                                "renderer_consumed": False,
                            },
                        }
                        debug_context["cleanup_plan_contracts"] = {
                            "version": "cleanup_plans_phase5_cleanup_mask_obligations",
                            "page_id": page_id,
                            "renderer_consumed": False,
                            "errors": [f"{type(exc).__name__}: {exc}"],
                            "summary": {
                                "plan_count": 0,
                                "rejected_record_count": 0,
                                "protected_record_count": 0,
                                "skipped_record_count": 0,
                                "error_count": 1,
                                "renderer_consumed": False,
                            },
                        }
                        debug_context["render_eligibility_contracts"] = {
                            "version": "render_readiness_diagnostics_v2",
                            "page_id": page_id,
                            "renderer_consumed": False,
                            "decisions": [],
                            "errors": [f"{type(exc).__name__}: {exc}"],
                            "summary": {
                                "decision_count": 0,
                                "diagnostic_count": 0,
                                "eligible_count": 0,
                                "error_count": 1,
                                "renderer_consumed": False,
                                },
                            }
                    self._record_stage_outcome(
                        page_id=page_id,
                        page_index=index - 1,
                        page_name=name,
                        source_path=source_path,
                        output_path=output_path,
                        stage=PipelineStage.CLEANUP,
                        state=PipelineStageOutcomeState.TECHNICAL_FAILURE,
                        parent_ids=([technical.parent_id] if technical.parent_id else ()),
                        artifact_kind="cleanup_failure_evidence",
                        artifact_summary=technical.artifact_summary,
                        diagnostics=technical.diagnostics,
                        error_code=technical.code,
                        message=technical.message,
                        detail=technical.detail,
                    )
                    self._emit_structured_error(
                        code=technical.code,
                        owner_stage=PipelineStage.CLEANUP,
                        message=f"Cleanup processing failed for {name}: {technical.message}",
                        detail=technical.detail,
                        page_id=page_id,
                        parent_id=technical.parent_id,
                        recoverable=True,
                        retry_action=PipelineRetryAction.RETRY_PAGE,
                        operation=technical.operation or "run_cleanup",
                        terminal=True,
                    )
                    self.message.emit(f"Cleanup processing failed for {name}: {technical.message}")
                    return

                _sync_parent_execution_downstream_contracts(
                    parent_execution_bundles,
                    execution_regions,
                    source_glyph_masks=source_glyph_mask_result,
                    cleanup_jobs=cleanup_job_contract_result,
                    cleanup_masks=cleanup_mask_contract_result,
                    render_eligibility=render_eligibility_contract_result,
                )
                self._record_stage_outcome(
                    page_id=page_id,
                    page_index=index - 1,
                    page_name=name,
                    source_path=source_path,
                    output_path=output_path,
                    stage=PipelineStage.CLEANUP,
                    state=(
                        PipelineStageOutcomeState.VALID_WITH_DIAGNOSTICS
                        if cleanup_diagnostics
                        else PipelineStageOutcomeState.VALID
                    ),
                    parent_ids=(bundle.parent_id for bundle in parent_execution_bundles),
                    artifact_kind="cleaned_page_base",
                    artifact_summary={
                        "cleaned_page_base": cleaned_page_base_record,
                        "parent_execution_bundles": [
                            bundle.to_audit_dict() for bundle in parent_execution_bundles
                        ],
                    },
                    diagnostics=cleanup_diagnostics,
                )

                if getattr(self._settings, "private_cleanup_validation_stop_after_cleanup", False):
                    try:
                        if os.path.isfile(render_input_path):
                            shutil.copyfile(render_input_path, output_path)
                        if debug_context is not None:
                            set_timing(debug_context, "rendering_time", 0.0)
                            debug_context["private_cleanup_validation_stop_after_cleanup"] = True
                            debug_context["render_translations_called"] = False
                            debug_context["final_translated_text_drawn"] = False
                            debug_context["cleanup_upstream_renderer_input_path"] = render_input_path
                        _pipeline_runtime_checkpoint(
                            "renderer_entry",
                            "skipped_private_cleanup_validation",
                            page_id=page_id,
                            render_input_path=render_input_path,
                        )
                    except Exception as exc:
                        _pipeline_runtime_checkpoint(
                            "renderer_entry",
                            "error",
                            page_id=page_id,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        page_elapsed = time.time() - page_start
                        self.queue_item.emit(index - 1, f"error ({_format_seconds(page_elapsed)}): {exc}")
                        message = f"Failed to write cleanup validation output for {name}: {exc}"
                        self._emit_structured_error(
                            code="cleanup_validation_output_failed",
                            owner_stage=PipelineStage.CLEANUP,
                            message=message,
                            detail=f"{type(exc).__name__}: {exc}",
                            page_id=page_id,
                            recoverable=True,
                            retry_action=PipelineRetryAction.RETRY_PAGE,
                            operation="write_cleanup_validation_output",
                        )
                        self.message.emit(message)
                        continue
                else:
                    self._emit_stage(
                        PipelineStage.STYLE,
                        "Resolving current-page source style",
                        page_id=page_id,
                    )
                    parent_font_mode = str(self._settings.font_detection or "off").strip()
                    parent_style_observation = None
                    style_view_result = None
                    source_punctuation_geometry_observation = None
                    if parent_execution_bundles:
                        parent_font_start = time.time()
                        try:
                            _pipeline_runtime_checkpoint(
                                "parent_style_after_cleanup",
                                "start",
                                page_id=page_id,
                                mode=parent_font_mode,
                                parent_execution_bundle_count=len(parent_execution_bundles),
                            )
                            if (
                                font_detection_mode == "yuzumarker"
                                and parent_style_detector is None
                                and not parent_style_detector_initialization_attempted
                            ):
                                parent_style_detector_initialization_attempted = True
                                try:
                                    models_dir = os.path.join(os.getcwd(), "models")
                                    parent_style_detector = YuzuMarkerOnnxFontDetector(
                                        model_path=resolve_yuzumarker_font_onnx_file(models_dir),
                                        labels_path=resolve_yuzumarker_font_labels_file(models_dir),
                                        use_gpu=self._settings.use_gpu,
                                    )
                                except Exception:
                                    # Preserve the existing observer-owned
                                    # unavailable/fallback and page-local retry
                                    # contract after one worker-owned attempt.
                                    parent_style_detector = None
                            style_view_result, parent_style_observation = (
                                _observe_parent_style_after_cleanup(
                                    page_id=page_id,
                                    image_path=source_path,
                                    parent_execution_bundles=parent_execution_bundles,
                                    cleanup_masks=cleanup_mask_contract_result,
                                    image_size=source_image_size,
                                    mode=parent_font_mode,
                                    use_gpu=self._settings.use_gpu,
                                    models_dir=os.path.join(os.getcwd(), "models"),
                                    detector=parent_style_detector,
                                )
                            )
                            _pipeline_runtime_checkpoint(
                                "parent_style_after_cleanup",
                                "observed_current_page",
                                page_id=page_id,
                                evidence_count=len(parent_style_observation.evidence),
                                observed_count=sum(
                                    1
                                    for item in parent_style_observation.evidence
                                    if item.status == "observed" and item.vote_eligible
                                ),
                                requested_execution_provider=(
                                    parent_style_observation.requested_execution_provider
                                ),
                                primary_execution_provider=(
                                    parent_style_observation.primary_execution_provider
                                ),
                                provider_fallback_reason=(
                                    parent_style_observation.provider_fallback_reason
                                ),
                                elapsed_ms=round((time.time() - parent_font_start) * 1000.0, 3),
                            )
                            if parent_style_observation.provider_fallback_reason:
                                self.message.emit(
                                    "Font detection GPU fallback: "
                                    f"{parent_style_observation.provider_fallback_reason}; "
                                    "continuing with CPUExecutionProvider."
                                )
                            if debug_context is not None:
                                if not debug_context.get("perf_telemetry_only"):
                                    from app.pipeline.parent_style_evidence import (
                                        write_authorized_source_style_view_debug_artifacts,
                                    )

                                    debug_context["authorized_source_style_views"] = (
                                        style_view_result.to_audit_dict()
                                    )
                                    debug_context["authorized_source_style_view_artifacts"] = (
                                        write_authorized_source_style_view_debug_artifacts(
                                            debug_context,
                                            image_path=source_path,
                                            result=style_view_result,
                                        )
                                    )
                                    debug_context["parent_style_evidence"] = (
                                        parent_style_observation.to_audit_dict()
                                    )
                                set_timing(
                                    debug_context,
                                    "parent_font_detection_time",
                                    time.time() - parent_font_start,
                                )
                                set_count(
                                    debug_context,
                                    "authorized_source_style_views_ready",
                                    sum(1 for view in style_view_result.views if view.available),
                                )
                                set_count(
                                    debug_context,
                                    "parent_style_evidence_observed",
                                    sum(
                                        1
                                        for item in parent_style_observation.evidence
                                        if item.status == "observed"
                                        and item.vote_eligible
                                    ),
                                )
                        except Exception as exc:
                            _pipeline_runtime_checkpoint(
                                "parent_style_after_cleanup",
                                "error",
                                page_id=page_id,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                            try:
                                from app.pipeline.parent_font_detection import (
                                    ParentStyleEvidenceRunResult,
                                    StyleEvidence,
                                )

                                parent_style_observation = ParentStyleEvidenceRunResult(
                                    page_id=page_id,
                                    mode=parent_font_mode,
                                    evidence=[
                                        StyleEvidence.unavailable(
                                            page_id=page_id,
                                            bundle_id=str(
                                                getattr(bundle, "bundle_id", "") or ""
                                            ),
                                            parent_id=str(
                                                getattr(bundle, "parent_id", "") or ""
                                            ),
                                            root_id=str(
                                                getattr(bundle, "root_id", "") or ""
                                            ),
                                            reason_codes=(
                                                "authorized_style_stage_failed",
                                                "authorized_style_stage_failed_"
                                                f"{type(exc).__name__}",
                                            ),
                                        )
                                        for bundle in parent_execution_bundles
                                        if bool(getattr(bundle, "render_required", False))
                                    ],
                                    errors=(
                                        f"authorized_style_stage_failed:{type(exc).__name__}:{exc}",
                                    ),
                                )
                                message = (
                                    f"Parent style evidence failed for {name}; "
                                    "current-page arbitration will use explicit unavailable evidence."
                                )
                                self.message.emit(message)
                            except Exception as fallback_exc:
                                page_elapsed = time.time() - page_start
                                self.queue_item.emit(
                                    index - 1,
                                    f"error ({_format_seconds(page_elapsed)}): {fallback_exc}",
                                )
                                message = (
                                    f"Failed to record unavailable parent style evidence for "
                                    f"{name}: {fallback_exc}"
                                )
                                technical = PipelineStageTechnicalError(
                                    stage=PipelineStage.STYLE,
                                    code="style_evidence_fallback_failed",
                                    message="Style could not produce valid unavailable-evidence fallback records.",
                                    detail=f"{type(fallback_exc).__name__}: {fallback_exc}",
                                    page_id=page_id,
                                    operation="record_unavailable_style_evidence",
                                )
                                self._record_stage_outcome(
                                    page_id=page_id,
                                    page_index=index - 1,
                                    page_name=name,
                                    source_path=source_path,
                                    output_path=output_path,
                                    stage=PipelineStage.STYLE,
                                    state=PipelineStageOutcomeState.TECHNICAL_FAILURE,
                                    artifact_kind="style_failure_evidence",
                                    error_code=technical.code,
                                    message=technical.message,
                                    detail=technical.detail,
                                )
                                self._emit_structured_error(
                                    code=technical.code,
                                    owner_stage=PipelineStage.STYLE,
                                    message=message,
                                    detail=technical.detail,
                                    page_id=page_id,
                                    recoverable=True,
                                    retry_action=PipelineRetryAction.RETRY_PAGE,
                                    operation=technical.operation,
                                    terminal=True,
                                )
                                self.message.emit(message)
                                return
                            if debug_context is not None and not debug_context.get(
                                "perf_telemetry_only"
                            ):
                                debug_context["authorized_source_style_views"] = {
                                    "authorized_source_style_view_version": (
                                        "authorized_source_style_view_v1"
                                    ),
                                    "page_id": page_id,
                                    "views": [],
                                    "errors": [f"{type(exc).__name__}: {exc}"],
                                }
                                debug_context["parent_style_evidence"] = (
                                    parent_style_observation.to_audit_dict()
                                )
                        source_punctuation_geometry_observation = (
                            _observe_parent_punctuation_geometry_after_cleanup(
                                page_id=page_id,
                                image_path=source_path,
                                parent_execution_bundles=parent_execution_bundles,
                                cleanup_masks=cleanup_mask_contract_result,
                                image_size=source_image_size,
                                style_views=style_view_result,
                                style_evidence=parent_style_observation,
                            )
                        )
                        if debug_context is not None:
                            if not debug_context.get("perf_telemetry_only"):
                                debug_context["source_punctuation_geometry"] = (
                                    source_punctuation_geometry_observation.to_audit_dict()
                                )
                            set_count(
                                debug_context,
                                "source_punctuation_geometry_occurrences",
                                sum(
                                    len(item.occurrences)
                                    for item in source_punctuation_geometry_observation.evidence
                                ),
                            )
                    else:
                        from app.pipeline.parent_font_detection import (
                            ParentStyleEvidenceRunResult,
                        )
                        from app.render.source_punctuation_hints import (
                            SourcePunctuationGeometryRunResult,
                        )

                        parent_style_observation = ParentStyleEvidenceRunResult(
                            page_id=page_id,
                            mode=parent_font_mode,
                        )
                        source_punctuation_geometry_observation = (
                            SourcePunctuationGeometryRunResult(page_id=page_id)
                        )
                        if debug_context is not None and not debug_context.get(
                            "perf_telemetry_only"
                        ):
                            debug_context["parent_style_evidence"] = {
                                **parent_style_observation.to_audit_dict(),
                                "reason": "no_parent_execution_bundles",
                            }
                            debug_context["source_punctuation_geometry"] = {
                                **source_punctuation_geometry_observation.to_audit_dict(),
                                "reason": "no_parent_execution_bundles",
                            }

                    render_eligibility_audit = (
                        render_eligibility_contract_result.to_audit_dict()
                        if hasattr(render_eligibility_contract_result, "to_audit_dict")
                        else dict(render_eligibility_contract_result or {})
                    )
                    try:
                        style_context_delta = prepare_style_context_delta(
                            snapshot=style_context_snapshot,
                            page_identity=StyleContextPageIdentity(
                                page_index=index - 1,
                                page_id=page_id,
                                page_name=name,
                                source_sha256=str(
                                    cleaned_page_base_record.get(
                                        "source_sha256"
                                    )
                                    or ""
                                ),
                            ),
                            parent_execution_bundles=parent_execution_bundles,
                            evidence=parent_style_observation.evidence,
                        )
                    except Exception as exc:
                        style_context_delta = None
                        if isinstance(debug_context, dict):
                            debug_context["style_context_transport"] = {
                                "contract_version": "style_context_transport_v1",
                                "status": "delta_unavailable",
                                "reason": f"{type(exc).__name__}: {exc}",
                                "incoming_snapshot_id": (
                                    style_context_snapshot.snapshot_id
                                ),
                                "incoming_prefix_page_ids": list(
                                    style_context_snapshot.prefix_page_ids
                                ),
                                "arbitration_consumer_enabled": False,
                            }
                    try:
                        if style_font_manager is None:
                            style_font_manager = FontManager(
                                base_dir=os.path.join(os.getcwd(), "models")
                            )
                        page_arbitration = resolve_parent_style_for_page(
                            page_id=page_id,
                            parent_execution_bundles=parent_execution_bundles,
                            evidence=parent_style_observation.evidence,
                            font_manager=style_font_manager,
                            style_context_snapshot=style_context_snapshot,
                        )
                    except Exception as exc:
                        if (
                            cleanup_upstream_temp_path
                            and os.path.isfile(cleanup_upstream_temp_path)
                        ):
                            try:
                                os.unlink(cleanup_upstream_temp_path)
                            except OSError:
                                pass
                        page_elapsed = time.time() - page_start
                        self.queue_item.emit(
                            index - 1,
                            f"error ({_format_seconds(page_elapsed)}): {exc}",
                        )
                        message = (
                            "Failed to resolve current-page parent styles for "
                            f"{name}: {type(exc).__name__}: {exc}"
                        )
                        self._record_stage_outcome(
                            page_id=page_id,
                            page_index=index - 1,
                            page_name=name,
                            source_path=source_path,
                            output_path=output_path,
                            stage=PipelineStage.STYLE,
                            state=PipelineStageOutcomeState.TECHNICAL_FAILURE,
                            parent_ids=(bundle.parent_id for bundle in parent_execution_bundles),
                            artifact_kind="style_failure_evidence",
                            error_code="style_resolution_failed",
                            message="Style arbitration could not produce valid parent styles.",
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                        self._emit_structured_error(
                            code="style_resolution_failed",
                            owner_stage=PipelineStage.STYLE,
                            message=message,
                            detail=f"{type(exc).__name__}: {exc}",
                            page_id=page_id,
                            recoverable=True,
                            retry_action=PipelineRetryAction.RETRY_PAGE,
                            operation="resolve_parent_styles",
                            terminal=True,
                        )
                        self.message.emit(message)
                        return

                    observation_audit = parent_style_observation.to_audit_dict()
                    records = [dict(record) for record in page_arbitration.records]
                    prior_page_cache_used = any(
                        any(
                            str(reason).startswith("prior_page_")
                            or str(reason)
                            == "compatible_prior_prefix_target_affinity"
                            for reason in (
                                axis_record.get("reason_codes") or ()
                            )
                        )
                        for record in records
                        for axis_record in (
                            (
                                record.get("render_style", {})
                                .get("axis_authority", {})
                                .values()
                            )
                            if isinstance(record.get("render_style"), dict)
                            and isinstance(
                                record.get("render_style", {}).get(
                                    "axis_authority"
                                ),
                                dict,
                            )
                            else ()
                        )
                        if isinstance(axis_record, dict)
                    )
                    applied_count = sum(
                        1
                        for record in records
                        if str(record.get("status") or "") == "applied"
                    )
                    skipped_count = sum(
                        1
                        for record in records
                        if str(record.get("status") or "") == "skipped"
                    )
                    fallback_count = len(records) - applied_count - skipped_count
                    parent_font_audit = {
                        "parent_font_detection_version": "parent_font_detection_v2",
                        "page_id": page_id,
                        "mode": str(observation_audit.get("mode") or "off"),
                        "enabled": bool(observation_audit.get("enabled")),
                        "applied_count": applied_count,
                        "fallback_count": fallback_count,
                        "skipped_count": skipped_count,
                        "model_path": str(observation_audit.get("model_path") or ""),
                        "labels_path": str(observation_audit.get("labels_path") or ""),
                        "gpu_requested": bool(
                            observation_audit.get("gpu_requested")
                        ),
                        "requested_execution_provider": str(
                            observation_audit.get("requested_execution_provider")
                            or ""
                        ),
                        "available_execution_providers": list(
                            observation_audit.get("available_execution_providers")
                            or []
                        ),
                        "active_execution_providers": list(
                            observation_audit.get("active_execution_providers")
                            or []
                        ),
                        "primary_execution_provider": str(
                            observation_audit.get("primary_execution_provider")
                            or ""
                        ),
                        "provider_fallback_reason": str(
                            observation_audit.get("provider_fallback_reason") or ""
                        ),
                        "provider_preload_error": str(
                            observation_audit.get("provider_preload_error") or ""
                        ),
                        "errors": list(observation_audit.get("errors") or []),
                        "records": records,
                    }
                    if isinstance(debug_context, dict):
                        if not debug_context.get("perf_telemetry_only"):
                            debug_context["parent_font_detection"] = parent_font_audit
                        debug_context["parent_style_page_resolution"] = {
                            "contract_version": "parent_style_page_resolution_v1",
                            "page_id": page_id,
                            "current_page_only": True,
                            "prior_page_cache_used": prior_page_cache_used,
                            "future_page_evidence_used": False,
                        }
                        if "style_context_transport" not in debug_context:
                            debug_context["style_context_transport"] = {
                                "contract_version": "style_context_transport_v1",
                                "status": "delta_prepared",
                                "incoming_snapshot_id": (
                                    style_context_snapshot.snapshot_id
                                ),
                                "incoming_prefix_page_ids": list(
                                    style_context_snapshot.prefix_page_ids
                                ),
                                "prepared_delta_id": (
                                    style_context_delta.delta_id
                                    if style_context_delta is not None
                                    else ""
                                ),
                                "prepared_record_count": (
                                    len(style_context_delta.records)
                                    if style_context_delta is not None
                                    else 0
                                ),
                                "arbitration_consumer_enabled": True,
                            }
                        else:
                            debug_context["style_context_transport"][
                                "arbitration_consumer_enabled"
                            ] = True
                        set_count(
                            debug_context,
                            "parent_font_detection_applied",
                            applied_count,
                        )
                        set_count(
                            debug_context,
                            "parent_font_detection_fallback",
                            fallback_count,
                        )
                        for record in records:
                            bundle_id = str(record.get("bundle_id") or "")
                            if bundle_id:
                                mark_render_region(
                                    debug_context,
                                    bundle_id,
                                    parent_font_detection=record,
                                )

                    style_diagnostics = tuple(
                        str(value)
                        for value in getattr(parent_style_observation, "errors", ()) or ()
                        if str(value)
                    )
                    self._record_stage_outcome(
                        page_id=page_id,
                        page_index=index - 1,
                        page_name=name,
                        source_path=source_path,
                        output_path=output_path,
                        stage=PipelineStage.STYLE,
                        state=(
                            PipelineStageOutcomeState.VALID_WITH_DIAGNOSTICS
                            if style_diagnostics
                            else PipelineStageOutcomeState.VALID
                        ),
                        parent_ids=(bundle.parent_id for bundle in parent_execution_bundles),
                        artifact_kind="resolved_parent_styles",
                        artifact_summary={
                            "parent_execution_bundles": [
                                bundle.to_audit_dict() for bundle in parent_execution_bundles
                            ],
                            "style_evidence": observation_audit,
                        },
                        diagnostics=style_diagnostics,
                    )
                    self._emit_stage(
                        PipelineStage.RENDERING,
                        "Rendering the committed page output",
                        page_id=page_id,
                    )
                    render_start = time.time()
                    _pipeline_runtime_checkpoint(
                        "renderer_entry",
                        "start_after_current_page_style",
                        page_id=page_id,
                        render_input_path=render_input_path,
                    )
                    try:
                        if parent_execution_bundles:
                            render_parent_execution_bundles(
                                render_input_path,
                                output_path,
                                parent_execution_bundles,
                                self._settings.font_name,
                                inpaint_mode=self._settings.inpaint_mode,
                                use_gpu=self._settings.use_gpu,
                                model_id=self._settings.inpaint_model_id,
                                debug_context=(
                                    debug_context
                                    if debug_artifacts_enabled
                                    else None
                                ),
                                render_eligibility=render_eligibility_audit,
                                perf_telemetry_context=(
                                    debug_context
                                    if perf_telemetry_is_enabled
                                    else None
                                ),
                                cleaned_page_base=cleaned_page_base_record,
                            )
                            execution_regions = parent_execution_region_records(
                                parent_execution_bundles
                            )
                        else:
                            _write_no_layer_render_output(
                                render_input_path,
                                output_path,
                                debug_context=debug_context,
                            )
                        if isinstance(debug_context, dict):
                            set_timing(
                                debug_context,
                                "rendering_time",
                                time.time() - render_start,
                            )
                            if render_input_path != source_path:
                                debug_context[
                                    "cleanup_upstream_renderer_input_path"
                                ] = render_input_path
                        _pipeline_runtime_checkpoint(
                            "renderer_entry",
                            "end_after_current_page_style",
                            page_id=page_id,
                            elapsed_ms=round(
                                (time.time() - render_start) * 1000.0,
                                3,
                            ),
                        )
                        self._record_stage_outcome(
                            page_id=page_id,
                            page_index=index - 1,
                            page_name=name,
                            source_path=source_path,
                            output_path=output_path,
                            stage=PipelineStage.RENDERING,
                            state=PipelineStageOutcomeState.VALID,
                            parent_ids=(bundle.parent_id for bundle in parent_execution_bundles),
                            artifact_kind="rendered_page",
                            artifact_summary={
                                "output_path": output_path,
                                "parent_execution_bundles": [
                                    bundle.to_audit_dict() for bundle in parent_execution_bundles
                                ],
                            },
                        )
                    except Exception as exc:
                        _pipeline_runtime_checkpoint(
                            "renderer_entry",
                            "error_after_current_page_style",
                            page_id=page_id,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        if (
                            cleanup_upstream_temp_path
                            and os.path.isfile(cleanup_upstream_temp_path)
                        ):
                            try:
                                os.unlink(cleanup_upstream_temp_path)
                            except OSError:
                                pass
                        self.queue_item.emit(index - 1, f"error: {exc}")
                        message = f"Failed to render {name}: {exc}"
                        self._record_stage_outcome(
                            page_id=page_id,
                            page_index=index - 1,
                            page_name=name,
                            source_path=source_path,
                            output_path=output_path,
                            stage=PipelineStage.RENDERING,
                            state=PipelineStageOutcomeState.TECHNICAL_FAILURE,
                            parent_ids=(bundle.parent_id for bundle in parent_execution_bundles),
                            artifact_kind="render_failure_evidence",
                            error_code="render_failed",
                            message="Renderer could not produce a valid page artifact.",
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                        self._emit_structured_error(
                            code="render_failed",
                            owner_stage=PipelineStage.RENDERING,
                            message=message,
                            detail=f"{type(exc).__name__}: {exc}",
                            page_id=page_id,
                            recoverable=True,
                            retry_action=PipelineRetryAction.RETRY_PAGE,
                            operation="render_page",
                            terminal=True,
                        )
                        self.message.emit(message)
                        return
                    if style_context_delta is not None:
                        try:
                            style_context_candidate_journal = (
                                journal_with_committed_delta(
                                    style_context_journal,
                                    style_context_delta,
                                )
                            )
                        except Exception as exc:
                            style_context_candidate_journal = (
                                style_context_journal
                            )
                            if isinstance(debug_context, dict):
                                debug_context["style_context_transport"] = {
                                    "contract_version": (
                                        "style_context_transport_v1"
                                    ),
                                    "status": "delta_commit_unavailable",
                                    "reason": f"{type(exc).__name__}: {exc}",
                                    "incoming_snapshot_id": (
                                        style_context_snapshot.snapshot_id
                                    ),
                                    "arbitration_consumer_enabled": True,
                                }
                if cleanup_upstream_temp_path:
                    try:
                        os.unlink(cleanup_upstream_temp_path)
                    except OSError:
                        pass

                self._emit_stage(
                    PipelineStage.PERSISTENCE,
                    "Committing the page checkpoint",
                    page_id=page_id,
                )
                page_record = build_page_record(
                    source_path,
                    page_id,
                    execution_regions,
                    output_path,
                    page_class=page_class,
                )
                page_record["file_name"] = name
                page_record["processing_state"] = "completed"
                if cleaned_page_base_record:
                    page_record["cleaned_page_base"] = cleaned_page_base_record
                if parent_execution_bundles:
                    page_record["source_regions"] = regions
                    page_record["parent_execution_bundles"] = [
                        bundle.to_audit_dict() for bundle in parent_execution_bundles
                    ]
                try:
                    checkpoint_receipt = _commit_page_project_checkpoint(
                        checkpoint_session=checkpoint_session,
                        project=project,
                        committed_pages=pages,
                        page_record=page_record,
                        style_context_delta=(
                            style_context_delta.to_project_dict()
                            if style_context_delta is not None
                            else None
                        ),
                        style_context_cache=(
                            style_context_candidate_journal.to_project_dict()
                        ),
                    )
                except Exception as exc:
                    page_elapsed = time.time() - page_start
                    self.queue_item.emit(
                        index - 1,
                        f"error ({_format_seconds(page_elapsed)}): {exc}",
                    )
                    message = (
                        f"Failed to checkpoint project after {name}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    try:
                        self._record_stage_outcome(
                            page_id=page_id,
                            page_index=index - 1,
                            page_name=name,
                            source_path=source_path,
                            output_path=output_path,
                            stage=PipelineStage.PERSISTENCE,
                            state=PipelineStageOutcomeState.TECHNICAL_FAILURE,
                            parent_ids=(bundle.parent_id for bundle in parent_execution_bundles),
                            artifact_kind="checkpoint_failure_evidence",
                            error_code="checkpoint_commit_failed",
                            message="The page checkpoint could not be committed.",
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                    except Exception:
                        pass
                    self._emit_structured_error(
                        code="checkpoint_commit_failed",
                        owner_stage=PipelineStage.PERSISTENCE,
                        message=message,
                        detail=f"{type(exc).__name__}: {exc}",
                        page_id=page_id,
                        recoverable=True,
                        retry_action=PipelineRetryAction.RETRY_PAGE,
                        operation="commit_page_checkpoint",
                        terminal=True,
                    )
                    self.message.emit(message)
                    return
                self._record_stage_outcome(
                    page_id=page_id,
                    page_index=index - 1,
                    page_name=name,
                    source_path=source_path,
                    output_path=output_path,
                    stage=PipelineStage.PERSISTENCE,
                    state=PipelineStageOutcomeState.VALID,
                    parent_ids=(bundle.parent_id for bundle in parent_execution_bundles),
                    artifact_kind="page_checkpoint",
                    artifact_summary={
                        "page_id": page_id,
                        "commit_sha256": str(checkpoint_receipt.commit_sha256 or ""),
                    },
                )
                if isinstance(debug_context, dict):
                    set_timing(
                        debug_context,
                        "project_checkpoint_encode_time",
                        checkpoint_receipt.encode_seconds,
                    )
                    set_timing(
                        debug_context,
                        "project_checkpoint_transaction_time",
                        checkpoint_receipt.transaction_seconds,
                    )
                    set_timing(
                        debug_context,
                        "project_checkpoint_descriptor_time",
                        checkpoint_receipt.descriptor_seconds,
                    )
                    set_timing(
                        debug_context,
                        "project_checkpoint_total_time",
                        checkpoint_receipt.total_seconds,
                    )
                    set_count(
                        debug_context,
                        "project_checkpoint_page_bytes",
                        checkpoint_receipt.page_bytes,
                    )
                    set_count(
                        debug_context,
                        "project_checkpoint_style_delta_bytes",
                        checkpoint_receipt.style_delta_bytes,
                    )
                style_context_journal = style_context_candidate_journal
                self.page_ready.emit(index - 1, page_record)

                # Track glossary size at this page for consistency checking
                if auto_glossary_state is not None:
                    with _glossary_lock:
                        current_glossary_size = len(auto_glossary_state.get("map", {}))
                        snapshots = auto_glossary_state.setdefault("page_snapshots", {})
                        snapshots[index - 1] = current_glossary_size

                page_elapsed = time.time() - page_start
                if debug_context is not None:
                    set_timing(debug_context, "total_page_time", page_elapsed)
                    set_timing(debug_context, "page_functional_time", page_elapsed)
                if debug_artifacts_enabled and debug_context is not None:
                    try:
                        artifact_start = time.time()
                        _pipeline_runtime_checkpoint("debug_artifact_write", "start", page_id=page_id)
                        write_page_artifacts(
                            debug_context,
                            execution_regions if parent_execution_bundles else regions,
                        )
                        _pipeline_runtime_checkpoint(
                            "debug_artifact_write",
                            "end",
                            page_id=page_id,
                            elapsed_ms=round((time.time() - artifact_start) * 1000.0, 3),
                        )
                        self.message.emit(f"Debug artifacts written for {name}")
                    except Exception as exc:
                        _pipeline_runtime_checkpoint(
                            "debug_artifact_write",
                            "error",
                            page_id=page_id,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        self.message.emit(f"Failed to write debug artifacts for {name}: {exc}")
                self.page_time_changed.emit(f"Page: {_format_seconds(page_elapsed)}")
                self.queue_item.emit(index - 1, f"done ({_format_seconds(page_elapsed)})")
                progress = int(index / total * 100)
                self.progress_changed.emit(progress)

                elapsed = time.time() - start_time
                self.total_time_changed.emit(f"Total: {_format_seconds(elapsed)}")
                avg = elapsed / index
                remaining = avg * (total - index)
                self.eta_changed.emit(_format_eta(remaining))
                self._emit_progress_snapshot(
                    completed_pages=index,
                    total_pages=total,
                    percent=progress,
                    eta_seconds=remaining,
                    current_page_id=page_id,
                )

                # --- PER-PAGE MEMORY CLEANUP ---
                # Prevent memory accumulation over long chapters (fixes 2GB+ leak)
                memory_maintenance_start = time.perf_counter() if perf_telemetry_is_enabled else 0.0
                try:
                    del regions
                except NameError:
                    pass
                import gc
                python_gc_start = time.perf_counter() if perf_telemetry_is_enabled else 0.0
                gc.collect()
                python_gc_time = (
                    time.perf_counter() - python_gc_start
                    if perf_telemetry_is_enabled
                    else 0.0
                )

                # Clear the active Torch accelerator cache every 5 pages.
                accelerator_cache_clear_time = 0.0
                if self._settings.use_gpu and index % 5 == 0:
                    try:
                        cache_clear_start = (
                            time.perf_counter()
                            if perf_telemetry_is_enabled
                            else 0.0
                        )
                        release_torch_memory()
                        if perf_telemetry_is_enabled:
                            accelerator_cache_clear_time = (
                                time.perf_counter() - cache_clear_start
                            )
                    except Exception:
                        pass

                if perf_telemetry_is_enabled and debug_context is not None:
                    memory_maintenance_time = time.perf_counter() - memory_maintenance_start
                    page_cycle_time = time.time() - page_start
                    set_timing(debug_context, "python_gc_time", python_gc_time)
                    set_timing(
                        debug_context,
                        "accelerator_cache_clear_time",
                        accelerator_cache_clear_time,
                    )
                    set_timing(debug_context, "memory_maintenance_time", memory_maintenance_time)
                    set_timing(debug_context, "page_cycle_time", page_cycle_time)
                    set_timing(
                        debug_context,
                        "page_postprocess_time",
                        max(0.0, page_cycle_time - page_elapsed),
                    )
                    set_timing(
                        debug_context,
                        "page_process_cpu_time",
                        max(0.0, time.process_time() - page_process_cpu_start),
                    )
                    try:
                        telemetry_write_start = time.perf_counter()
                        write_perf_timing_artifact(
                            debug_context,
                            execution_regions if parent_execution_bundles else page_record.get("regions", []),
                        )
                        telemetry_write_time = time.perf_counter() - telemetry_write_start
                        append_perf_timing_overhead_artifact(
                            debug_context,
                            telemetry_artifact_write_time=telemetry_write_time,
                            observed_page_cycle_with_telemetry=time.time() - page_start,
                        )
                    except Exception as exc:
                        self.message.emit(f"Failed to write performance telemetry for {name}: {exc}")

            self._emit_stage(
                PipelineStage.FINALIZING,
                "Finalizing the durable project",
            )
            project["pages"] = list(pages)
            project["stage_outcomes"] = checkpoint_session.stage_outcomes()
            project["style_context_cache"] = (
                style_context_journal.to_project_dict()
            )
            try:
                checkpoint_finalize_receipt = checkpoint_session.finalize(
                    expected_project=project,
                )
            except Exception as exc:
                message = (
                    "Failed to finalize project JSON from the durable page "
                    f"checkpoint: {type(exc).__name__}: {exc}"
                )
                self._emit_structured_error(
                    code="checkpoint_finalize_failed",
                    owner_stage=PipelineStage.FINALIZING,
                    message=message,
                    detail=f"{type(exc).__name__}: {exc}",
                    recoverable=True,
                    retry_action=PipelineRetryAction.RETRY_RUN,
                    operation="finalize_checkpoint",
                    terminal=True,
                )
                self.message.emit(message)
                return
            if perf_telemetry_is_enabled and perf_telemetry_output_root:
                try:
                    os.makedirs(perf_telemetry_output_root, exist_ok=True)
                    checkpoint_summary = checkpoint_session.summary()
                    checkpoint_summary["finalize"] = (
                        checkpoint_finalize_receipt.to_dict()
                    )
                    with open(
                        os.path.join(
                            perf_telemetry_output_root,
                            "project_checkpoint_summary.json",
                        ),
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        json.dump(
                            checkpoint_summary,
                            handle,
                            ensure_ascii=False,
                            indent=2,
                        )
                except Exception as exc:
                    self.message.emit(
                        "Failed to write project checkpoint telemetry: "
                        f"{type(exc).__name__}: {exc}"
                    )

            # --- MEMORY CLEANUP START ---
            # Flush Python Garbage Collector
            import gc
            gc.collect()

            # Flush the selected PyTorch accelerator cache (if used).
            if self._settings.use_gpu:
                release_torch_memory()
            # --- MEMORY CLEANUP END ---

            total_elapsed = time.time() - start_time
            self.total_time_changed.emit(f"Total: {_format_seconds(total_elapsed)}")
            self.message.emit("Completed")
            self._queue_terminal_lifecycle(PipelineRunState.COMPLETED, "Completed")
        finally:
            if checkpoint_session is not None:
                try:
                    checkpoint_session.close()
                except Exception:
                    pass
            should_finalize_auto_glossary = (
                auto_glossary_state
                and self._settings.style_guide_path
                and not self._settings.prescan_enabled
                and not self._settings.files_whitelist
            )
            if should_finalize_auto_glossary:
                try:
                    # Force final discovery if buffer has remaining text
                    with _glossary_lock:
                        remaining_buffer = auto_glossary_state.get("buffer", [])
                        is_running = auto_glossary_state.get("is_running", False)

                    if remaining_buffer and not is_running:
                        self.message.emit("Running final Auto-Glossary discovery...")
                        # Run synchronously (not in thread) to ensure completion
                        use_deep_scan = bool(self._settings.use_ollama_discovery)
                        discovery_client = ollama
                        created_client = None
                        discovery_model = self._settings.discovery_model
                        if use_deep_scan:
                            backend = self._settings.discovery_backend
                            if backend == "GGUF" or (discovery_model and ".gguf" in discovery_model.lower()):
                                target_path = str(discovery_model or "").strip()
                                if target_path and os.path.isfile(target_path):
                                    if hasattr(ollama, "_model_path") and os.path.abspath(target_path) == os.path.abspath(getattr(ollama, "_model_path", "")):
                                        discovery_client = ollama
                                    else:
                                        from app.translate.gguf_client import GGUFClient
                                        n_gpu_layers = self._settings.gguf_n_gpu_layers
                                        created_client = GGUFClient(
                                            model_path=target_path,
                                            prompt_style="extract",
                                            n_ctx=2048,
                                            n_gpu_layers=n_gpu_layers,
                                            n_threads=max(1, self._settings.gguf_n_threads),
                                            n_batch=min(128, self._settings.gguf_n_batch),
                                        )
                                        discovery_client = created_client
                                else:
                                    self.message.emit("Deep Scan GGUF model path is invalid for final discovery.")
                                    use_deep_scan = False
                            elif backend == "Ollama":
                                if hasattr(ollama, "list_models"):
                                    discovery_client = ollama
                                elif self._settings.use_ollama_discovery:
                                    try:
                                        from app.translate.ollama_client import OllamaClient
                                        new_client = OllamaClient(
                                            base_url=self._settings.discovery_base_url,
                                            context_tokens=self._settings.discovery_context,
                                        )
                                        if new_client.is_available():
                                            discovery_client = new_client
                                            created_client = new_client
                                        else:
                                            use_deep_scan = False
                                    except Exception:
                                        use_deep_scan = False
                        if use_deep_scan and discovery_client:
                            _run_sakura_discovery(
                                discovery_client,
                                model_name,
                                self._settings.source_lang,
                                self._settings.target_lang,
                                auto_glossary_state,
                                style_guide,
                                self._settings.style_guide_path,
                                discovery_model,
                            )
                        else:
                            _run_discovery(
                                ollama,
                                model_name,
                                self._settings.source_lang,
                                self._settings.target_lang,
                                auto_glossary_state,
                                style_guide,
                                self._settings.style_guide_path,
                                bool(ollama and hasattr(ollama, "generate")),
                            )
                        if created_client is not None and hasattr(created_client, "close"):
                            try:
                                created_client.close()
                            except Exception:
                                pass

                    # Ensure we have the latest data from background threads
                    with _glossary_lock:
                        learned_map = auto_glossary_state.get("map", {})
                        learned_chars = auto_glossary_state.get("characters", [])

                    if learned_map or learned_chars:
                        from app.io.style_guide import save_style_guide, load_style_guide
                        # Re-load to avoid overwriting external edits
                        current_sg = _load_style_guide(self._settings.style_guide_path, self._settings.target_lang)
                        updated_sg = _merge_glossary(current_sg, learned_map, learned_chars)
                        updated_sg = _sanitize_style_guide(updated_sg, self._settings.target_lang)
                        save_style_guide(self._settings.style_guide_path, updated_sg)
                        self.message.emit(
                            f"Auto-Glossary: Saved {len(learned_map)} terms, {len(learned_chars)} characters."
                        )
                except Exception as e:
                    self.message.emit(f"Failed to save Auto-Glossary data: {e}")

            # Consistency Check: Compare early pages vs final glossary
            # SKIP if running in re-translation mode (files_whitelist is set)
            # to prevent infinite loop: re-translate → consistency check → dialog → re-translate...
            if auto_glossary_state is None:
                pass
            elif self._settings.files_whitelist:
                self.message.emit("Skipping consistency check (re-translation mode).")
            elif self._settings.prescan_enabled:
                self.message.emit("Skipping consistency check (Pre-Scan enabled).")
            else:
                try:
                    final_style = _load_style_guide(self._settings.style_guide_path, self._settings.target_lang)
                    cleaned_style = _sanitize_style_guide(final_style, self._settings.target_lang)
                    if cleaned_style is not final_style and self._settings.style_guide_path:
                        from app.io.style_guide import save_style_guide
                        save_style_guide(self._settings.style_guide_path, cleaned_style)
                    inconsistent_pages = _find_inconsistent_pages(pages, cleaned_style)
                    if inconsistent_pages:
                        self.message.emit(
                            f"Consistency check: {len(inconsistent_pages)} pages may have "
                            f"outdated name translations."
                        )
                        # Emit signal for UI to handle
                        self.consistency_issue.emit(inconsistent_pages)
                except Exception as e:
                    self.message.emit(f"Consistency check failed: {e}")

            try:
                if hasattr(ocr_engine, "close"):
                    ocr_engine.close()
            except Exception:
                pass

            glossary_client_in_use = False
            if auto_glossary_state is not None:
                with _glossary_lock:
                    glossary_client_in_use = bool(auto_glossary_state.get("is_running", False))
            if (
                not glossary_client_in_use
                and getattr(ollama, "owns_http_sessions", False)
                and hasattr(ollama, "close")
            ):
                try:
                    ollama.close()
                except Exception:
                    pass
            self._checkpoint_session = None
            self._flush_terminal_lifecycle()


class PipelineController(QtCore.QObject):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.status = PipelineStatus()
        self._running = False
        self._worker: PipelineWorker | None = None
        self._current_run_id = ""

    def start(
        self,
        settings: PipelineSettings,
        *,
        runtime_binding: PipelineRuntimeBinding | None = None,
    ) -> bool:
        if self._running:
            return False
        if runtime_binding is not None and not isinstance(
            runtime_binding, PipelineRuntimeBinding
        ):
            raise TypeError("runtime_binding must be PipelineRuntimeBinding or None")
        run_id = new_run_id()
        self._current_run_id = run_id
        self.status.lifecycle_changed.emit(
            PipelineLifecycleEvent(
                run_id=run_id,
                state=PipelineRunState.VALIDATING,
                message="Validating run settings",
            )
        )
        self.status.stage_changed.emit(
            PipelineStageEvent(
                run_id=run_id,
                stage=PipelineStage.VALIDATION,
                detail="Validating run settings",
            )
        )
        if not settings.import_dir:
            message = "Import folder is required."
            self.status.structured_error.emit(
                PipelineErrorReceipt(
                    error_id=new_error_id(),
                    run_id=run_id,
                    code="import_directory_required",
                    owner_stage=PipelineStage.VALIDATION,
                    message=message,
                    recoverable=True,
                    retry_action=PipelineRetryAction.RELINK,
                    operation="validate_run_settings",
                )
            )
            self.status.message.emit(message)
            self.status.lifecycle_changed.emit(
                PipelineLifecycleEvent(
                    run_id=run_id,
                    state=PipelineRunState.FAILED,
                    message=message,
                )
            )
            self._current_run_id = ""
            return False
        if not settings.export_dir:
            message = "Export folder is required."
            self.status.structured_error.emit(
                PipelineErrorReceipt(
                    error_id=new_error_id(),
                    run_id=run_id,
                    code="export_directory_required",
                    owner_stage=PipelineStage.VALIDATION,
                    message=message,
                    recoverable=True,
                    retry_action=PipelineRetryAction.RELINK,
                    operation="validate_run_settings",
                )
            )
            self.status.message.emit(message)
            self.status.lifecycle_changed.emit(
                PipelineLifecycleEvent(
                    run_id=run_id,
                    state=PipelineRunState.FAILED,
                    message=message,
                )
            )
            self._current_run_id = ""
            return False
        self._running = True
        self._worker = PipelineWorker(
            settings,
            self,
            runtime_binding=runtime_binding,
            run_id=run_id,
        )
        self._worker.lifecycle_changed.connect(self.status.lifecycle_changed.emit)
        self._worker.stage_changed.connect(self.status.stage_changed.emit)
        self._worker.stage_outcome.connect(self.status.stage_outcome.emit)
        self._worker.progress_snapshot.connect(self.status.progress_snapshot.emit)
        self._worker.structured_error.connect(self.status.structured_error.emit)
        self._worker.runtime_backend_selected.connect(
            self.status.runtime_backend_selected.emit
        )
        self._worker.progress_changed.connect(self.status.progress_changed.emit)
        self._worker.eta_changed.connect(self.status.eta_changed.emit)
        self._worker.page_changed.connect(self.status.page_changed.emit)
        self._worker.total_time_changed.connect(self.status.total_time_changed.emit)
        self._worker.page_time_changed.connect(self.status.page_time_changed.emit)
        self._worker.message.connect(self.status.message.emit)
        self._worker.queue_reset.connect(self.status.queue_reset.emit)
        self._worker.queue_item.connect(self.status.queue_item.emit)
        self._worker.page_ready.connect(self.status.page_ready.emit)
        self._worker.consistency_issue.connect(self.status.consistency_issue.emit)
        # Two-Pass Pipeline signals
        self._worker.prescan_started.connect(self.status.prescan_started.emit)
        self._worker.prescan_progress.connect(self.status.prescan_progress.emit)
        self._worker.prescan_finished.connect(self.status.prescan_finished.emit)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self.status.message.emit("Started")
        self.status.lifecycle_changed.emit(
            PipelineLifecycleEvent(
                run_id=run_id,
                state=PipelineRunState.RUNNING,
                message="Started",
            )
        )
        return True

    def stop(self) -> None:
        if not self._running:
            return
        if self._worker:
            self._worker.request_stop()
        self.status.message.emit("Stopping...")
        if self._current_run_id:
            self.status.lifecycle_changed.emit(
                PipelineLifecycleEvent(
                    run_id=self._current_run_id,
                    state=PipelineRunState.STOP_REQUESTED,
                    message="Stop requested; waiting for the next safe page boundary.",
                )
            )

    def _on_finished(self):
        self._running = False
        self._worker = None
        self._current_run_id = ""

    def start_deep_scan(self, settings: PipelineSettings):
        """Start deep scan worker."""
        if self._running:
            return

        self.deep_scan_worker = DeepScanWorker(settings)
        # Relay signals? For now just simple finished
        self.deep_scan_worker.finished.connect(self._on_deep_scan_finished)
        self.deep_scan_worker.start()

    def _on_deep_scan_finished(self):
        self.status.message.emit("Deep scan completed. Glossary updated.")
        self.status.consistency_issue.emit([]) # Signal to maybe refresh?
        # Actually Main Window handles the dialog logic, it waits for this worker to finish?
        # We'll rely on the worker reference in MainWindow if we want to block interaction.


class DeepScanWorker(QtCore.QThread):
    finished = QtCore.Signal()

    def __init__(self, settings: PipelineSettings, parent=None):
        super().__init__(parent)
        self.settings = settings

    def run(self):
        try:
            # Load project pages to get text
            # We assume the project is located at settings.json_path
            if not os.path.exists(self.settings.json_path):
                return

            from app.translate.ollama_client import OllamaClient
            from app.models.ollama import list_models

            project = load_project(self.settings.json_path)

            pages = project.get("pages", [])
            accumulated = []
            if isinstance(pages, dict):
                sorted_keys = sorted(pages.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
                page_items = [pages[k] for k in sorted_keys]
            else:
                page_items = pages
            for page in page_items:
                if not isinstance(page, dict):
                    continue
                blocks = page.get("regions", []) or page.get("blocks", [])
                for b in blocks:
                    if isinstance(b, dict) and b.get("ocr_text"):
                        t = str(b["ocr_text"]).replace("\n", "").strip()
                        if t:
                            accumulated.append(t)

            if not accumulated:
                return

            # Hybrid Strategy:
            # Even if user checks "GGUF" for translation speed, we can attempt
            # to use a smart Ollama model (like Qwen) for discovery if available.

            backend = getattr(self.settings, "discovery_backend", "Ollama")
            discovery_model = getattr(self.settings, "discovery_model", None)
            model_to_use = self.settings.ollama_model

            ollama = None
            if backend == "GGUF" or (discovery_model and ".gguf" in str(discovery_model).lower()):
                if discovery_model and "sakura" in str(discovery_model).lower():
                    print("DeepScan: Sakura GGUF is translation-only; skipping Deep Scan.")
                    return
                if discovery_model and os.path.isfile(discovery_model):
                    from app.translate.gguf_client import GGUFClient
                    n_gpu_layers = self.settings.gguf_n_gpu_layers
                    ollama = GGUFClient(
                        model_path=discovery_model,
                        prompt_style="extract",
                        n_ctx=2048,
                        n_gpu_layers=n_gpu_layers,
                        n_threads=max(1, self.settings.gguf_n_threads),
                        n_batch=min(128, self.settings.gguf_n_batch),
                    )
                    model_to_use = "gguf_model"
                else:
                    print("DeepScan: GGUF backend selected but model path is invalid")
                    return
            else:
                ollama = OllamaClient(
                    base_url=self.settings.discovery_base_url,
                    context_tokens=self.settings.discovery_context,
                )
                if not ollama.is_available():
                    print("DeepScan: Ollama server is not running")
                    return
                if discovery_model and str(discovery_model).strip() and "auto" not in str(discovery_model).lower():
                    model_to_use = str(discovery_model).strip()
                if model_to_use and "sakura" in model_to_use.lower():
                    model_to_use = ""
                if not model_to_use or "auto" in model_to_use.lower():
                    available_models = list_models()
                    qwen = next((m for m in available_models if "qwen" in m.lower()), None)
                    non_sakura = next((m for m in available_models if "sakura" not in m.lower()), None)
                    model_to_use = qwen if qwen else (non_sakura if non_sakura else "")

            if not model_to_use:
                # No model found
                print("DeepScan: No Ollama model found")
                return

            print(f"DeepScan: using model {model_to_use}")

            if not ollama:
                print("DeepScan: No discovery client available")
                return
            if not model_to_use:
                print("DeepScan: No model found")
                return
            # Load style guide
            base_style = _load_style_guide(self.settings.style_guide_path, self.settings.target_lang)

            # Run discovery
            # Mock state
            state = {"buffer": accumulated}

            _run_sakura_discovery(
                ollama=ollama,
                model=model_to_use,
                source_lang=self.settings.source_lang,
                target_lang=self.settings.target_lang,
                state=state,
                base_style=base_style,
                style_guide_path=self.settings.style_guide_path
            )

        except Exception as e:
            print(f"Deep scan error: {e}")
        finally:
            if "ollama" in locals() and hasattr(ollama, "close"):
                try:
                    ollama.close()
                except Exception:
                    pass
            self.finished.emit()


def _list_images(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    names = []
    for entry in os.listdir(folder):
        _, ext = os.path.splitext(entry)
        if ext.lower() in allowed:
            names.append(entry)
    names.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)])
    return names


def _format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "00:00"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_seconds(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _lang_code(label: str) -> str:
    mapping = {
        "Japanese": "ja",
        "Simplified Chinese": "zh-Hans",
        "English": "en",
    }
    return mapping.get(label, label)


def _friendly_model_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "paddleocr-vl" in lowered or "paddleocr_vl" in lowered or "paddle ocr-vl" in lowered:
        return f"PaddleOCR-VL failed to initialize: {text}"
    if "llama-server" in lowered:
        return f"PaddleOCR-VL runtime failed: {text}"
    if "failed to load torch" in lowered:
        return (
            "Torch failed to load (DLL dependency error). Restart the app after installing conda PyTorch. "
            "If it persists, reboot Windows to refresh DLL search paths."
        )
    if "no module named 'torch'" in lowered:
        return (
            "PyTorch is not installed in the current environment. Install it or switch OCR Engine to PaddleOCR-VL. "
            "Suggested: pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )
    if "cve-2025-32434" in lowered or "upgrade torch to at least v2.6" in lowered:
        return (
            "MangaOCR hit the new torch.load safety restriction. YomiFrame will try a local safetensors "
            "compatibility copy first; if that fails, upgrade torch to 2.6+ or switch OCR Engine to PaddleOCR-VL."
        )
    if "manga-ocr" in lowered or "manga_ocr" in lowered:
        return f"MangaOCR failed to load: {text}"
    if "comictextdetector" in lowered or "comic-text-detector" in lowered or "utils.general" in lowered:
        return (
            "ComicTextDetector is not ready. Download comictextdetector.pt.onnx (CPU) or "
            "comictextdetector.pt (GPU) from https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1 "
            "and place it under models/comic-text-detector."
        )
    if "llama-cpp-python is not installed" in lowered or "no module named 'llama_cpp'" in lowered:
        return (
            "GGUF backend failed: llama-cpp-python is missing in the current environment. "
            "Install it with: pip install llama-cpp-python, or switch Translator backend to Ollama."
        )
    if "gguf model not found:" in lowered:
        return f"GGUF backend failed: {text}"
    if "gguf" in lowered or "llama-cpp-python" in lowered or "llama_cpp" in lowered:
        return f"GGUF backend failed: {text}"
    if "yuzumarker" in lowered or "font detection" in lowered:
        return (
            "Font detection failed to initialize. Ensure the font model checkpoint is set and dependencies are installed."
        )
    if "numpy" in lowered and "abi" in lowered:
        return (
            "NumPy ABI mismatch. Reinstall numpy and the OCR deps. "
            "Suggested: pip install -U numpy==1.26.4 manga-ocr"
        )
    if "shm.dll" in lowered or "winerror 127" in lowered:
        return (
            "PyTorch DLL load failed. Reinstall torch in the conda env. "
            "Suggested: pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )
    return f"Failed to initialize models: {text}"


def _phase5_upstream_protected_region_ids(page_id: str) -> set[str]:
    """Return cleanup-commit exclusions.

    Phase 5 cleanup commit safety is now owned by CleanupProof and allowed-area
    containment. Keeping page/row-specific production exclusions here would
    turn cleanup accounting into a hidden allowlist and can block proof-passed
    cleanup results from becoming the renderer input image.
    """

    return set()


def _apply_cleanup_runtime_render_blocks(
    render_eligibility_result: Any,
    cleanup_runtime_contract_result: Any,
    debug_context: dict[str, Any] | None,
) -> Any:
    """Annotate cleanup runtime blockers without taking render permission.

    Cleanup proof/runtime may block unsafe pixel mutation and must remain visible
    in debug artifacts, but it is not a text-admission or render-entry owner for
    already accepted translated text.
    """

    if render_eligibility_result is None or cleanup_runtime_contract_result is None:
        return render_eligibility_result

    warning_records: list[dict[str, Any]] = []
    for status_record in getattr(cleanup_runtime_contract_result, "status_records", []) or []:
        if not isinstance(status_record, dict):
            continue
        runtime_status = str(status_record.get("runtime_status") or "")
        if runtime_status not in {"blocked", "failed", "inconclusive"}:
            continue
        cleanup_class = str(status_record.get("cleanup_class") or status_record.get("runtime_class") or "")
        cleanup_owned_hard_block = (
            runtime_status == "blocked"
            and cleanup_class in {"speech_flat_bubble", "speech_complex_bubble", "small_reaction"}
            and str(status_record.get("render_consumption_decision_if_consumed") or "")
            == "block_future_renderer_consumption"
            and str(status_record.get("failure_reason") or "")
        )
        proof_backed_failure = (
            runtime_status in {"failed", "inconclusive"}
            and bool(status_record.get("cleanup_result_id") or status_record.get("cleanup_proof_id"))
        )
        if not (cleanup_owned_hard_block or proof_backed_failure):
            continue
        for rid in status_record.get("target_region_ids", []) or []:
            region_id = str(rid or "")
            if region_id:
                warning_reason = (
                    "cleanup_runtime_hard_block_warning_before_renderer"
                    if cleanup_owned_hard_block
                    else "cleanup_runtime_proof_failed_warning_before_renderer"
                )
                warning_records.append(
                    {**status_record, "region_id": region_id, "phase5_cleanup_warning_reason": warning_reason}
                )
    if not warning_records:
        return render_eligibility_result

    for record in warning_records:
        region_id = str(record.get("region_id") or "")
        warning_reason = str(record.get("phase5_cleanup_warning_reason") or "cleanup_runtime_warning_before_renderer")
        if debug_context is not None:
            try:
                from app.pipeline.debug_artifacts import mark_render_region
                mark_render_region(
                    debug_context,
                    region_id,
                    cleanup_runtime_warning_before_renderer=True,
                    cleanup_runtime_warning_reason=warning_reason,
                    cleanup_runtime_status=record.get("runtime_status"),
                    cleanup_runtime_failure_reason=record.get("failure_reason"),
                    cleanup_result_id=record.get("cleanup_result_id"),
                    cleanup_proof_id=record.get("cleanup_proof_id"),
                    cleanup_render_permission_gate_released=True,
                    cleanup_render_permission_gate_release_reason="diagnostic_only_renderer_unaffected",
                    renderer_policy_changed=False,
                )
            except Exception:
                pass

    return render_eligibility_result


def _apply_cleanup_upstream_commit_render_blocks(
    render_eligibility_result: Any,
    cleanup_upstream_commit_result: Any,
    debug_context: dict[str, Any] | None,
) -> Any:
    """Annotate cleanup commit blockers without taking render permission."""

    if render_eligibility_result is None or cleanup_upstream_commit_result is None:
        return render_eligibility_result

    blocked_records: list[dict[str, Any]] = []
    for record in getattr(cleanup_upstream_commit_result, "blocked_records", []) or []:
        if not isinstance(record, dict):
            continue
        region_id = str(record.get("region_id") or "")
        if not region_id:
            continue
        failure_reason = str(record.get("failure_reason") or "")
        if not failure_reason:
            continue
        blocked_records.append(record)
    if not blocked_records:
        return render_eligibility_result

    for record in blocked_records:
        region_id = str(record.get("region_id") or "")
        if debug_context is not None:
            try:
                from app.pipeline.debug_artifacts import mark_render_region
                mark_render_region(
                    debug_context,
                    region_id,
                    cleanup_upstream_commit_status="blocked",
                    cleanup_upstream_commit_failure_reason=record.get("failure_reason"),
                    cleanup_upstream_commit_warning_before_renderer=True,
                    cleanup_upstream_commit_warning_reason="cleanup_upstream_commit_blocked_warning_before_renderer",
                    cleanup_result_id=record.get("cleanup_result_id"),
                    cleanup_proof_id=record.get("cleanup_proof_id"),
                    cleanup_render_permission_gate_released=True,
                    cleanup_render_permission_gate_release_reason="diagnostic_only_renderer_unaffected",
                    renderer_policy_changed=False,
                )
            except Exception:
                pass

    return render_eligibility_result


def _is_torch_missing(exc: Exception | None) -> bool:
    if exc is None:
        return False
    text = str(exc).lower()
    return (
        "no module named 'torch'" in text
        or "upgrade torch to at least v2.6" in text
        or "cve-2025-32434" in text
    )


def _load_style_guide(path: str, target_lang: str = ""):
    if path and os.path.isfile(path):
        try:
            # Handle empty or corrupt files gracefully
            if os.path.getsize(path) == 0:
                return default_style_guide()

            guide = load_style_guide(path)

            if target_lang:
                guide = _sanitize_style_guide(guide, target_lang)
            return guide
        except Exception:
            # Return default if file is corrupt (prevent crash)
            return default_style_guide()
    return default_style_guide()


_ocr_debug_counter = 0


def _safe_trace_token(value: object, fallback: str = "item") -> str:
    return safe_trace_token(value, fallback)


def _debug_page_dir(debug_context: dict | None) -> str:
    from app.pipeline.debug_artifacts import debug_stage_artifact_dir

    return debug_stage_artifact_dir(debug_context, "ocr")


def _build_text_foreground_segmentation_mask(
    *,
    detector,
    source_path: str,
    image_size: tuple[int, int] | None,
    input_size: int,
    page_id: str,
    debug_context: dict | None,
    text_area_plan=None,
    segmentation_result=None,
):
    from app.pipeline.cleanup_contracts import TextForegroundSegmentationMask

    if detector is None or not hasattr(detector, "detect_with_segmentation"):
        return TextForegroundSegmentationMask(
            page_id=page_id,
            image_size=image_size,
            provider=getattr(detector, "__class__", type("", (), {})).__name__ if detector is not None else "",
            provenance={"status": "segmentation_api_unavailable"},
        )
    result = segmentation_result
    if result is None:
        try:
            refinement_scopes = _text_area_ctd_refinement_scope_geometry(
                text_area_plan,
                image_size=image_size,
            )
            try:
                result = detector.detect_with_segmentation(
                    source_path,
                    input_size=input_size,
                    keep_undetected_mask=True,
                    refinement_scopes=refinement_scopes,
                )
            except TypeError:
                result = detector.detect_with_segmentation(source_path)
        except Exception as exc:
            return TextForegroundSegmentationMask(
                page_id=page_id,
                image_size=image_size,
                provider=detector.__class__.__name__,
                provenance={"status": "segmentation_failed", "error": f"{type(exc).__name__}: {exc}"},
            )

    raw_ref = ""
    refined_ref = ""
    page_dir = _debug_page_dir(debug_context)
    if page_dir:
        raw_ref = _save_segmentation_mask_ref(
            getattr(result, "raw_mask", None),
            debug_context,
            "text_foreground_segmentation",
            f"{page_id}_ctd_raw_mask.png",
        )
        refined_ref = _save_segmentation_mask_ref(
            getattr(result, "refined_mask", None),
            debug_context,
            "text_foreground_segmentation",
            f"{page_id}_ctd_refined_mask.png",
        )
    contract = TextForegroundSegmentationMask(
        page_id=page_id,
        image_size=getattr(result, "image_size", None) or image_size,
        raw_mask_ref=raw_ref,
        refined_mask_ref=refined_ref,
        threshold_used=getattr(result, "threshold_used", None),
        provider=getattr(result, "provider", "") or detector.__class__.__name__,
        backend=getattr(result, "backend", ""),
        runtime_ms=getattr(result, "runtime_ms", None),
        text_pixel_count=int(getattr(result, "text_pixel_count", 0) or 0),
        connected_component_stats=dict(getattr(result, "connected_component_stats", {}) or {}),
        block_associations=list(getattr(result, "blocks", []) or []),
        keep_undetected_mask=bool(getattr(result, "keep_undetected_mask", False)),
        confidence=dict(getattr(result, "confidence", {}) or {}),
        provenance=dict(getattr(result, "provenance", {}) or {}),
        raw_mask=getattr(result, "raw_mask", None),
        refined_mask=getattr(result, "refined_mask", None),
    )
    if debug_context is not None:
        debug_context["text_foreground_segmentation_mask"] = contract.to_audit_dict()
    return contract


def _text_area_ctd_refinement_scope_geometry(
    text_area_plan,
    *,
    image_size: tuple[int, int] | None,
) -> list[tuple[int, int, int, int]]:
    if text_area_plan is None:
        return []
    scopes = (
        text_area_plan.get("scopes", [])
        if isinstance(text_area_plan, dict)
        else getattr(text_area_plan, "scopes", [])
    )
    page_width = int(image_size[0]) if image_size else 0
    page_height = int(image_size[1]) if image_size else 0
    rectangles: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for scope in scopes or []:
        if isinstance(scope, dict):
            eligible = bool(scope.get("comic_text_detector_scope_eligible", False))
            bbox = scope.get("bbox") or []
        else:
            eligible = bool(getattr(scope, "comic_text_detector_scope_eligible", False))
            bbox = getattr(scope, "bbox", None) or []
        if not eligible or len(bbox) < 4:
            continue
        try:
            x, y, width, height = [int(round(float(value))) for value in bbox[:4]]
        except (TypeError, ValueError):
            continue
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = x + max(0, width)
        y1 = y + max(0, height)
        if page_width > 0:
            x0 = min(page_width, x0)
            x1 = min(page_width, x1)
        if page_height > 0:
            y0 = min(page_height, y0)
            y1 = min(page_height, y1)
        rectangle = (x0, y0, x1, y1)
        if x1 <= x0 or y1 <= y0 or rectangle in seen:
            continue
        seen.add(rectangle)
        rectangles.append(rectangle)
    return rectangles


def _save_segmentation_mask_ref(mask, debug_context: dict | None, subdir: str, filename: str) -> str:
    if mask is None:
        return ""
    try:
        import numpy as np
        from PIL import Image

        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = np.any(arr > 0, axis=2)
        elif arr.ndim != 2:
            return ""
        out = (arr > 0).astype("uint8") * 255
        path, saved, _error = save_context_image(
            debug_context,
            subdir=subdir,
            filename=filename,
            image=Image.fromarray(out, mode="L"),
            stage="ocr",
        )
        return path if saved else ""
    except Exception:
        return ""


def _ocr_trace_outcome(text: str, confidence: float, route_intent: str) -> tuple[str, str, str]:
    route = str(route_intent or "").strip()
    if route in _TEXT_AREA_TRANSLATABLE_ROUTES:
        state, reason = _ocr_transaction_state_for_text_area_route(text, confidence, route)
    else:
        cleaned = _clean_ocr_text(text)
        if not cleaned:
            state, reason = "ocr_empty_blocker", "empty_ocr"
        elif _is_punct_only(cleaned) or not _non_punct_chars(cleaned):
            state, reason = "ocr_punctuation_only_blocker", "punctuation_or_placeholder_ocr"
        elif _is_valid_japanese(cleaned) < 0.35:
            state, reason = "ocr_malformed_blocker", "ocr_not_japanese_cjk_or_kana"
        elif float(confidence or 0.0) < 0.45:
            state, reason = _OCR_LOW_CONFIDENCE_WARNING_STATE, "low_confidence_scoped_ocr_warning"
        else:
            state, reason = _OCR_TRANSLATION_READY_STATE, "ocr_sane"
    if state == _OCR_TRANSLATION_READY_STATE:
        outcome = "recognized_meaningful"
    elif state == _OCR_LOW_CONFIDENCE_WARNING_STATE:
        outcome = "low_confidence_meaningful"
    elif state == _OCR_PUNCTUATION_IDENTITY_STATE:
        outcome = "punctuation_identity"
    elif state == "ocr_empty_blocker":
        outcome = "empty"
    elif state == "ocr_punctuation_only_blocker":
        outcome = "punctuation_only"
    elif state == "ocr_malformed_blocker":
        outcome = "malformed"
    else:
        outcome = state or "unknown"
    return state, reason, outcome


def _begin_scoped_ocr_trace(
    debug_context: dict | None,
    crop,
    bbox,
    trace_context: dict | None,
) -> dict | None:
    if not debug_context:
        return None
    trace_context = dict(trace_context or {})
    if not _debug_page_dir(debug_context):
        return None
    counter = int(debug_context.get("_scoped_ocr_trace_counter") or 0)
    debug_context["_scoped_ocr_trace_counter"] = counter + 1
    page_id = str(trace_context.get("page_id") or debug_context.get("page_id") or "page")
    attempt_id = f"{page_id}_ocr_{counter:04d}"
    crop_filename = (
        f"{attempt_id}_"
        f"{_safe_trace_token(trace_context.get('attempt_kind') or trace_context.get('text_area_container_id') or 'scoped')}.png"
    )
    crop_path, crop_saved, crop_error = save_context_image(
        debug_context,
        subdir="scoped_ocr_crops",
        filename=crop_filename,
        image=crop,
    )
    record = {
        "page_id": page_id,
        "ocr_trace_attempt_id": attempt_id,
        "attempt_index": counter,
        "attempt_kind": trace_context.get("attempt_kind") or "scoped_ocr",
        "region_id": trace_context.get("region_id"),
        "root_id": trace_context.get("root_id"),
        "parent_id": trace_context.get("parent_id"),
        "logical_block_id": trace_context.get("logical_block_id"),
        "text_area_container_id": trace_context.get("text_area_container_id"),
        "route_intent": trace_context.get("route_intent"),
        "ocr_eligible": trace_context.get("ocr_eligible"),
        "source_bbox": trace_context.get("source_bbox") or bbox,
        "container_bbox": trace_context.get("container_bbox"),
        "actual_crop_bbox": trace_context.get("actual_crop_bbox") or bbox,
        "crop_image_path": crop_path if crop_saved else "",
        "crop_saved": crop_saved,
        "crop_save_error": crop_error,
        "ocr_raw_text": "",
        "ocr_text": "",
        "ocr_confidence": None,
        "ocr_backend": "",
        "ocr_model_path": "",
        "ocr_mmproj_path": "",
        "ocr_endpoint": "",
        "ocr_prompt_version": "",
        "ocr_finish_reason": "",
        "ocr_prompt_tokens": None,
        "ocr_completion_tokens": None,
        "ocr_total_tokens": None,
        "ocr_max_tokens": None,
        "ocr_response_complete": None,
        "ocr_response_authoritative": None,
        "ocr_response_rejection_reason": "",
        "ocr_transaction_state": "",
        "ocr_transaction_reason": "",
        "ocr_outcome_class": "",
        "downstream_parent_id": trace_context.get("parent_id") or "",
        "translation_unit_id": "",
        "render_unit_id": "",
    }
    debug_context.setdefault("scoped_ocr_trace", []).append(record)
    return record


def _finish_scoped_ocr_trace(
    debug_context: dict | None,
    record: dict | None,
    text: str,
    confidence: float,
) -> None:
    if not debug_context or not record:
        return
    state, reason, outcome = _ocr_trace_outcome(text, confidence, str(record.get("route_intent") or ""))
    record["ocr_text"] = text
    record["ocr_confidence"] = float(confidence or 0.0)
    record["ocr_transaction_state"] = state
    record["ocr_transaction_reason"] = reason
    record["ocr_outcome_class"] = outcome


def _ocr_trace_context_from_assignment(
    *,
    page_id: str,
    region_id: str | None,
    bbox,
    assignment: dict | None,
    attempt_kind: str,
) -> dict[str, object]:
    assignment = assignment or {}
    return {
        "page_id": page_id,
        "region_id": region_id,
        "attempt_kind": attempt_kind,
        "text_area_container_id": assignment.get("text_area_container_id"),
        "route_intent": assignment.get("text_area_route_intent"),
        "ocr_eligible": assignment.get("text_area_ocr_eligible"),
        "source_bbox": list(bbox or []),
        "actual_crop_bbox": list(bbox or []),
        "container_bbox": assignment.get("text_area_container_bbox") or [],
    }


def _is_valid_japanese(text: str) -> float:
    """
    Score how likely text is valid Japanese (0.0 to 1.0).
    Higher score = more valid Japanese characters.
    """
    if not text:
        return 0.0
    valid = 0
    for c in text:
        code = ord(c)
        # Hiragana, Katakana, Kanji, punctuation
        if (0x3040 <= code <= 0x30FF or  # Hiragana + Katakana
            0x4E00 <= code <= 0x9FFF or  # Kanji
            0x3000 <= code <= 0x303F or  # Japanese punctuation
            c in '!?。、…・「」『』（）'):
            valid += 1
    return valid / len(text) if text else 0.0


def _record_perf_ocr_request(
    debug_context: dict | None,
    *,
    trace_context: dict | None,
    crop,
    elapsed_seconds: float,
    failed: bool,
) -> None:
    if not debug_context or not debug_context.get("perf_telemetry_only"):
        return
    trace = trace_context or {}
    attempt_kind = str(trace.get("attempt_kind") or "unclassified_ocr")
    width = 0
    height = 0
    try:
        width, height = [int(value) for value in crop.size]
    except Exception:
        pass
    elapsed = max(0.0, float(elapsed_seconds or 0.0))
    summary = debug_context.setdefault(
        "ocr_request_summary",
        {
            "request_count": 0,
            "failed_count": 0,
            "total_latency_sec": 0.0,
            "max_latency_sec": 0.0,
            "attempt_kinds": {},
        },
    )
    summary["request_count"] = int(summary.get("request_count") or 0) + 1
    summary["failed_count"] = int(summary.get("failed_count") or 0) + (1 if failed else 0)
    summary["total_latency_sec"] = float(summary.get("total_latency_sec") or 0.0) + elapsed
    summary["max_latency_sec"] = max(float(summary.get("max_latency_sec") or 0.0), elapsed)
    kinds = summary.setdefault("attempt_kinds", {})
    entry = kinds.setdefault(
        attempt_kind,
        {
            "request_count": 0,
            "failed_count": 0,
            "total_latency_sec": 0.0,
            "max_latency_sec": 0.0,
            "total_crop_area": 0,
            "max_crop_area": 0,
        },
    )
    area = max(0, width * height)
    entry["request_count"] = int(entry.get("request_count") or 0) + 1
    entry["failed_count"] = int(entry.get("failed_count") or 0) + (1 if failed else 0)
    entry["total_latency_sec"] = float(entry.get("total_latency_sec") or 0.0) + elapsed
    entry["max_latency_sec"] = max(float(entry.get("max_latency_sec") or 0.0), elapsed)
    entry["total_crop_area"] = int(entry.get("total_crop_area") or 0) + area
    entry["max_crop_area"] = max(int(entry.get("max_crop_area") or 0), area)
    debug_context.setdefault("counts", {})["ocr_request_count"] = int(summary["request_count"])


def _recognize_with_fallback(
    ocr_engine,
    crop,
    settings,
    bbox=None,
    *,
    debug_context: dict | None = None,
    trace_context: dict | None = None,
) -> tuple[str, float]:
    """OCR recognition using the selected engine only."""
    global _ocr_debug_counter
    text = ""
    conf = 1.0
    trace_record = _begin_scoped_ocr_trace(debug_context, crop, bbox, trace_context)
    if trace_record is not None:
        if hasattr(ocr_engine, "backend_metadata"):
            try:
                trace_record.update(ocr_engine.backend_metadata())
            except Exception:
                trace_record["ocr_backend"] = ocr_engine.__class__.__name__
        else:
            trace_record["ocr_backend"] = ocr_engine.__class__.__name__

    # DEBUG: Save crop images
    if settings and getattr(settings, 'debug_ocr', False):
        ocr_debug_context = debug_context or {
            "debug_dir": os.path.join(str(getattr(settings, "export_dir", "") or os.getcwd()), "debug_artifacts"),
            "page_id": "ocr_debug",
        }
        crop_path, crop_saved, crop_error = save_context_image(
            ocr_debug_context,
            subdir="ocr_debug_crops",
            filename=f"crop_{_ocr_debug_counter:04d}_bbox_{_safe_trace_token(bbox, 'bbox')}.png",
            image=crop,
        )
        if crop_saved:
            print(f"[OCR DEBUG] Saved crop: {crop_path}")
        elif crop_error:
            print(f"[OCR DEBUG] Failed to save crop: {crop_error}")
        _ocr_debug_counter += 1

    perf_ocr_start = (
        time.perf_counter()
        if debug_context and debug_context.get("perf_telemetry_only")
        else None
    )
    ocr_failed = False
    recognition_metadata: dict[str, Any] = {}
    try:
        if hasattr(ocr_engine, "recognize_with_confidence"):
            text, conf = ocr_engine.recognize_with_confidence(crop)
        else:
            text = ocr_engine.recognize(crop)
            conf = 1.0
    except Exception:
        ocr_failed = True
        raise
    finally:
        if perf_ocr_start is not None:
            _record_perf_ocr_request(
                debug_context,
                trace_context=trace_context,
                crop=crop,
                elapsed_seconds=time.perf_counter() - perf_ocr_start,
                failed=ocr_failed,
            )

    raw_text = _clean_ocr_text(text)
    if hasattr(ocr_engine, "last_recognition_metadata"):
        try:
            recognition_metadata = dict(ocr_engine.last_recognition_metadata() or {})
        except Exception:
            recognition_metadata = {}
    if trace_record is not None:
        trace_record["ocr_raw_text"] = raw_text
        for key in (
            "ocr_finish_reason",
            "ocr_prompt_tokens",
            "ocr_completion_tokens",
            "ocr_total_tokens",
            "ocr_max_tokens",
            "ocr_response_complete",
            "ocr_response_authoritative",
            "ocr_response_rejection_reason",
        ):
            if key in recognition_metadata:
                trace_record[key] = recognition_metadata[key]
    if recognition_metadata.get("ocr_response_authoritative") is False:
        text = ""
        conf = 0.0

    if settings and getattr(settings, 'debug_ocr', False):
         backend = getattr(ocr_engine, "backend_name", ocr_engine.__class__.__name__)
         print(f"[OCR DEBUG] bbox={bbox} backend={backend} text='{text}' conf={conf:.3f}")

    cleaned_text = _clean_ocr_text(text)
    _finish_scoped_ocr_trace(debug_context, trace_record, cleaned_text, conf)
    return cleaned_text, conf


def _record_text_area_fallback_decision(
    debug_context: dict | None,
    page_id: str,
    bbox: list,
    assignment: dict,
    reason: str,
) -> None:
    if debug_context is None:
        return
    decisions = debug_context.setdefault("fallback_decisions", [])
    decisions.append(
        {
            "page_id": page_id,
            "bbox": bbox,
            "text_area_container_id": assignment.get("text_area_container_id"),
            "container_type": assignment.get("text_area_container_type"),
            "route_intent": assignment.get("text_area_route_intent"),
            "detection_source": assignment.get("text_area_detection_source"),
            "fallback_reason": assignment.get("text_area_fallback_reason") or reason,
            "ocr_eligibility_reason": assignment.get("text_area_ocr_eligibility_reason"),
            "reason_codes": assignment.get("text_area_reason_codes") or [],
            "conflict_flags": assignment.get("text_area_conflict_flags") or [],
            "would_change_behavior": False,
        }
    )
    if reason.startswith("text_area_plan_blocked") or assignment.get("text_area_detection_source") == "blocked_by_text_area_plan":
        debug_context.setdefault("blocked_text_area_candidates", []).append(
            {
                "page_id": page_id,
                "bbox": bbox,
                "text_area_container_id": assignment.get("text_area_container_id"),
                "container_type": assignment.get("text_area_container_type"),
                "route_intent": assignment.get("text_area_route_intent"),
                "ocr_eligible": bool(assignment.get("text_area_ocr_eligible")),
                "fallback_reason": assignment.get("text_area_fallback_reason") or reason,
                "ocr_eligibility_reason": assignment.get("text_area_ocr_eligibility_reason"),
                "reason_codes": assignment.get("text_area_reason_codes") or [],
                "conflict_flags": assignment.get("text_area_conflict_flags") or [],
                "would_change_behavior": False,
            }
        )


_TEXT_AREA_TRANSLATABLE_ROUTES = {"translate_speech", "translate_caption_background"}
_TEXT_AREA_TRANSLATABLE_AUTHORIZATION_STATES = {
    "cleanup_translate_speech",
    "cleanup_translate_background",
    "cleanup_translate_caption",
}
_TEXT_AREA_ASSIGNMENT_FIELD_KEYS = (
    "text_area_container_id",
    "text_area_semantic_unit_id",
    "text_area_semantic_kind",
    "text_area_container_type",
    "text_area_route_intent",
    "text_area_cleanup_authorization",
    "text_area_must_not_mutate",
    "text_area_protection_reason",
    "text_area_authorization_source_stage",
    "text_area_authorization_basis",
    "text_area_authorization_explicit",
    "text_area_authorization_field_origin",
    "text_area_semantic_authorization_state",
    "text_area_ctd_scope_eligible",
    "text_area_comic_text_detector_scope_eligible",
    "text_area_ocr_eligible",
    "text_area_translation_eligible",
    "text_area_render_eligible",
    "text_area_cleanup_executable",
    "text_area_detection_source",
    "text_area_fallback_reason",
    "text_area_confidence_tier",
    "text_area_container_bbox",
    "text_area_reason_codes",
    "text_area_conflict_flags",
    "text_area_pre_ocr_authority",
    "text_area_enriched_from_region",
    "text_area_ocr_eligibility_reason",
    "text_area_overlap_ratio",
)
_OCR_TRANSLATION_READY_STATE = "recognized_for_translation"
_OCR_LOW_CONFIDENCE_WARNING_STATE = "recognized_low_confidence_warning"
_OCR_PUNCTUATION_IDENTITY_STATE = "recognized_punctuation_identity"
_OCR_TRANSLATION_QUEUED_STATES = {
    _OCR_TRANSLATION_READY_STATE,
    _OCR_LOW_CONFIDENCE_WARNING_STATE,
    _OCR_PUNCTUATION_IDENTITY_STATE,
}
_OCR_BLOCKER_STATES = {
    "ocr_empty_blocker",
    "ocr_punctuation_only_blocker",
    "ocr_malformed_blocker",
}
_NONLEXICAL_OCR_PROXY_CHARS = frozenset({"*", "＊", "∗", "☀"})


def _is_text_area_translatable_assignment(assignment: dict | None) -> bool:
    if not isinstance(assignment, dict):
        return False
    if assignment.get("text_area_ocr_eligible") is not True:
        return False
    if assignment.get("text_area_translation_eligible") is not True:
        return False
    if assignment.get("text_area_cleanup_executable") is not True:
        return False
    route = str(assignment.get("text_area_route_intent") or "").strip()
    if route not in _TEXT_AREA_TRANSLATABLE_ROUTES:
        return False
    if not bool(assignment.get("text_area_authorization_explicit")):
        return False
    state = str(
        assignment.get("text_area_semantic_authorization_state")
        or assignment.get("text_area_cleanup_authorization")
        or ""
    ).strip()
    return state in _TEXT_AREA_TRANSLATABLE_AUTHORIZATION_STATES


def _region_has_translatable_text_area_route(region: dict | None) -> bool:
    return _is_text_area_translatable_assignment(_region_text_area_assignment(region or {}))


def _ocr_transaction_state_for_text_area_route(
    ocr_text: str,
    ocr_conf: float,
    route_intent: str,
) -> tuple[str, str]:
    cleaned = _clean_ocr_text(ocr_text)
    if not cleaned:
        return "ocr_empty_blocker", "empty_ocr"
    if _is_nonlexical_ocr_proxy_run(cleaned):
        return _OCR_LOW_CONFIDENCE_WARNING_STATE, "nonlexical_ocr_proxy_warning"
    if _is_punct_only(cleaned):
        return _OCR_PUNCTUATION_IDENTITY_STATE, "punctuation_identity_ocr"
    body = _non_punct_chars(cleaned)
    if not body:
        return _OCR_PUNCTUATION_IDENTITY_STATE, "punctuation_identity_ocr"
    if _placeholder_ratio(cleaned) >= 0.18:
        return _OCR_LOW_CONFIDENCE_WARNING_STATE, "placeholder_heavy_ocr_warning"
    has_cjk_or_kana = any(_is_kana(ch) or 0x4E00 <= ord(ch) <= 0x9FFF for ch in body)
    if not has_cjk_or_kana:
        return _OCR_LOW_CONFIDENCE_WARNING_STATE, "ocr_not_japanese_cjk_or_kana_warning"
    if route_intent == "translate_caption_background":
        kana_only = all(_is_kana(ch) or ch in {"ー", "～"} for ch in body)
        katakana_count = sum(1 for ch in body if 0x30A0 <= ord(ch) <= 0x30FF)
        if kana_only and len(body) <= 5 and katakana_count >= max(1, len(body) - 1):
            return _OCR_LOW_CONFIDENCE_WARNING_STATE, "short_katakana_caption_ocr_warning"
    if float(ocr_conf or 0.0) < 0.45:
        return _OCR_LOW_CONFIDENCE_WARNING_STATE, "low_confidence_scoped_ocr_warning"
    return _OCR_TRANSLATION_READY_STATE, "text_area_route_ocr_sane"


def _is_nonlexical_ocr_proxy_run(text: str) -> bool:
    symbols = [char for char in str(text or "") if not char.isspace()]
    return bool(symbols) and all(char in _NONLEXICAL_OCR_PROXY_CHARS for char in symbols)


def _ocr_transaction_state_queues_translation(state: str) -> bool:
    return str(state or "").strip() in _OCR_TRANSLATION_QUEUED_STATES


def _apply_text_area_route_authority(
    region: dict,
    assignment: dict,
    ocr_text: str,
    ocr_conf: float,
    *,
    attempted_demote_reason: str = "",
) -> str:
    if not _is_text_area_translatable_assignment(assignment):
        return ""
    route = str(assignment.get("text_area_route_intent") or "").strip()
    state, reason = _ocr_transaction_state_for_text_area_route(ocr_text, ocr_conf, route)
    render = region.setdefault("render", {})
    flags = region.setdefault("flags", {})
    previous_type = str(region.get("type") or "")
    previous_cleanup = str(render.get("cleanup_mode") or "")
    previous_ignore = bool(flags.get("ignore"))

    region["text_area_original_route_intent"] = route
    region["text_area_ocr_transaction_state"] = state
    region["text_area_ocr_warning_reason"] = reason if state == _OCR_LOW_CONFIDENCE_WARNING_STATE else ""
    region["text_area_ocr_blocker_reason"] = "" if _ocr_transaction_state_queues_translation(state) else reason
    render["text_area_original_route_intent"] = route
    render["text_area_ocr_transaction_state"] = state
    render["text_area_ocr_warning_reason"] = region["text_area_ocr_warning_reason"]
    render["text_area_ocr_blocker_reason"] = region["text_area_ocr_blocker_reason"]

    if route == "translate_caption_background":
        region["type"] = "background_text"
        region["semantic_class"] = "background_text"
        flags["bg_text"] = True
        render["cleanup_mode"] = "local_text_mask"
        render["classification_reason"] = "text_area_route_authority_caption_background"
    else:
        region["type"] = "speech_bubble"
        region["semantic_class"] = "speech_bubble"
        flags["bg_text"] = False
        render["cleanup_mode"] = "bubble"
        render["classification_reason"] = "text_area_route_authority_speech"

    blocked_demote = (
        previous_ignore
        or previous_type in {"decorative_text", "sfx"}
        or previous_cleanup == "preserve"
        or bool(attempted_demote_reason)
    )
    if blocked_demote:
        region["text_area_route_authority_blocked_demote"] = True
        region["text_area_downstream_attempted_demote_reason"] = (
            attempted_demote_reason
            or str(render.get("classification_reason") or previous_cleanup or previous_type or "unknown_demote")
        )
        render["text_area_route_authority_blocked_demote"] = True
        render["text_area_downstream_attempted_demote_reason"] = region["text_area_downstream_attempted_demote_reason"]

    flags["ignore"] = False
    if _ocr_transaction_state_queues_translation(state):
        flags["needs_review"] = bool(flags.get("needs_review")) or state == _OCR_LOW_CONFIDENCE_WARNING_STATE
        flags.pop("hard_fail", None)
        region.pop("translation_blocked_by_ocr_transaction", None)
        render.pop("translation_blocked_by_ocr_transaction", None)
        render["text_area_route_authority_status"] = state
        region["route_owned_translation_queued"] = True
        render["route_owned_translation_queued"] = True
        region["render_activation_state"] = "eligible_after_translation"
        region["cleanup_activation_state"] = "eligible_after_translation"
        render["render_activation_state"] = region["render_activation_state"]
        render["cleanup_activation_state"] = region["cleanup_activation_state"]
    else:
        flags["needs_review"] = True
        flags["hard_fail"] = True
        region["translation"] = ""
        region["translated_text"] = ""
        region["translation_blocked_by_ocr_transaction"] = True
        region["route_owned_translation_queued"] = False
        region["logical_text_block_translation_unit"] = False
        region["active_translation_unit_id"] = None
        region["source_text_represented_by_block_id"] = None
        region["render_activation_state"] = "blocked_before_translation"
        region["cleanup_activation_state"] = "blocked_before_translation"
        render["text_area_route_authority_status"] = state
        render["translation_blocked_by_ocr_transaction"] = True
        render["route_owned_translation_queued"] = False
        render["logical_text_block_translation_unit"] = False
        render["active_translation_unit_id"] = None
        render["source_text_represented_by_block_id"] = None
        render["render_activation_state"] = region["render_activation_state"]
        render["cleanup_activation_state"] = region["cleanup_activation_state"]
        render["cleanup_mode"] = "ocr_blocked_no_cleanup"
        render.pop("final_render_bbox", None)
        render.pop("wrapped_lines", None)
        region.pop("final_render_bbox", None)
        region.pop("wrapped_lines", None)
    return state


def _region_translation_blocked_by_ocr_transaction(region: dict | None) -> bool:
    if not isinstance(region, dict):
        return False
    state = str(
        region.get("text_area_ocr_transaction_state")
        or (region.get("render") or {}).get("text_area_ocr_transaction_state")
        or ""
    ).strip()
    return state in _OCR_BLOCKER_STATES or bool(region.get("translation_blocked_by_ocr_transaction"))


def _route_owned_retry_candidate_bbox(
    bbox: list,
    assignment: dict,
    image_size: tuple[int, int] | None,
) -> list[int]:
    if not _is_text_area_translatable_assignment(assignment):
        return []
    current = _clip_controller_bbox([int(round(float(v or 0))) for v in (bbox or [])[:4]], image_size)
    container = _clip_controller_bbox(
        [int(round(float(v or 0))) for v in (assignment.get("text_area_container_bbox") or [])[:4]],
        image_size,
    )
    if not current or not container:
        return []
    if _bbox_inside_ratio_controller(current, container) <= 0.0:
        return []
    current_area = max(1, int(current[2]) * int(current[3]))
    container_area = max(1, int(container[2]) * int(container[3]))
    same_geometry = (
        abs(current[0] - container[0]) <= 2
        and abs(current[1] - container[1]) <= 2
        and abs(current[2] - container[2]) <= 4
        and abs(current[3] - container[3]) <= 4
    )
    if same_geometry or container_area <= current_area * 1.20:
        return []
    return container


def _record_route_owned_ocr_retry(
    debug_context: dict | None,
    page_id: str,
    assignment: dict,
    original_bbox: list,
    retry_bbox: list,
    original_text: str,
    original_conf: float,
    original_state: str,
    original_reason: str,
    retry_text: str = "",
    retry_conf: float = 0.0,
    retry_state: str = "",
    retry_reason: str = "",
    *,
    status: str,
) -> dict[str, Any]:
    record = {
        "page_id": page_id,
        "text_area_container_id": assignment.get("text_area_container_id"),
        "route_intent": assignment.get("text_area_route_intent"),
        "original_bbox": list(original_bbox or []),
        "retry_bbox": list(retry_bbox or []),
        "original_ocr_text": original_text,
        "original_ocr_confidence": float(original_conf or 0.0),
        "original_ocr_transaction_state": original_state,
        "original_ocr_transaction_reason": original_reason,
        "retry_ocr_text": retry_text,
        "retry_ocr_confidence": float(retry_conf or 0.0),
        "retry_ocr_transaction_state": retry_state,
        "retry_ocr_transaction_reason": retry_reason,
        "status": status,
        "final_ocr_transaction_state": retry_state if status == "accepted_retry_for_translation" else original_state,
        "failure_reason": "" if status == "accepted_retry_for_translation" else (retry_reason or original_reason),
    }
    if debug_context is not None:
        debug_context.setdefault("route_owned_ocr_retry_attempts", []).append(record)
    return record


def _stamp_route_owned_ocr_retry(region: dict, retry_info: dict | None) -> None:
    if not retry_info:
        return
    render = region.setdefault("render", {})
    fields = {
        "route_owned_ocr_retry_attempted": True,
        "route_owned_ocr_retry_status": retry_info.get("status"),
        "route_owned_ocr_retry_original_bbox": retry_info.get("original_bbox") or [],
        "route_owned_ocr_retry_bbox": retry_info.get("retry_bbox") or [],
        "route_owned_ocr_retry_original_text": retry_info.get("original_ocr_text") or "",
        "route_owned_ocr_retry_original_confidence": retry_info.get("original_ocr_confidence"),
        "route_owned_ocr_retry_original_state": retry_info.get("original_ocr_transaction_state") or "",
        "route_owned_ocr_retry_text": retry_info.get("retry_ocr_text") or "",
        "route_owned_ocr_retry_confidence": retry_info.get("retry_ocr_confidence"),
        "route_owned_ocr_retry_state": retry_info.get("retry_ocr_transaction_state") or "",
        "route_owned_ocr_retry_failure_reason": retry_info.get("failure_reason") or "",
    }
    for key, value in fields.items():
        region[key] = value
        render[key] = value


def _try_route_owned_scoped_ocr_retry(
    *,
    image_path: str,
    page_image,
    image_size: tuple[int, int] | None,
    bbox: list,
    assignment: dict,
    ocr_text: str,
    ocr_conf: float,
    ocr_engine,
    settings,
    debug_context: dict | None,
    page_id: str,
    region_id: str,
    attempt_kind: str,
) -> tuple[str, float, list, dict | None]:
    if not _is_text_area_translatable_assignment(assignment):
        return ocr_text, ocr_conf, bbox, None
    route = str(assignment.get("text_area_route_intent") or "").strip()
    original_state, original_reason = _ocr_transaction_state_for_text_area_route(ocr_text, ocr_conf, route)
    if original_state not in {"ocr_empty_blocker", "ocr_punctuation_only_blocker"}:
        return ocr_text, ocr_conf, bbox, None
    retry_bbox = _route_owned_retry_candidate_bbox(bbox, assignment, image_size)
    if not retry_bbox:
        info = _record_route_owned_ocr_retry(
            debug_context,
            page_id,
            assignment,
            bbox,
            [],
            ocr_text,
            ocr_conf,
            original_state,
            original_reason,
            status="skipped_no_bounded_route_local_retry_bbox",
        )
        return ocr_text, ocr_conf, bbox, info
    crop = _crop_image(image_path, retry_bbox, expand_wide=False, image_obj=page_image)
    if crop is None:
        info = _record_route_owned_ocr_retry(
            debug_context,
            page_id,
            assignment,
            bbox,
            retry_bbox,
            ocr_text,
            ocr_conf,
            original_state,
            original_reason,
            status="retry_crop_failed",
        )
        return ocr_text, ocr_conf, bbox, info
    retry_text, retry_conf = _recognize_with_fallback(
        ocr_engine,
        crop,
        settings,
        retry_bbox,
        debug_context=debug_context,
        trace_context={
            "page_id": page_id,
            "region_id": region_id,
            "attempt_kind": f"{attempt_kind}_route_owned_retry",
            "text_area_container_id": assignment.get("text_area_container_id"),
            "route_intent": assignment.get("text_area_route_intent"),
            "ocr_eligible": assignment.get("text_area_ocr_eligible"),
            "source_bbox": list(bbox or []),
            "actual_crop_bbox": list(retry_bbox or []),
            "container_bbox": assignment.get("text_area_container_bbox") or [],
        },
    )
    retry_text = _clean_ocr_text(str(retry_text or ""))
    retry_state, retry_reason = _ocr_transaction_state_for_text_area_route(retry_text, retry_conf, route)
    status = "accepted_retry_for_translation" if _ocr_transaction_state_queues_translation(retry_state) else "retry_failed_ocr_blocker"
    info = _record_route_owned_ocr_retry(
        debug_context,
        page_id,
        assignment,
        bbox,
        retry_bbox,
        ocr_text,
        ocr_conf,
        original_state,
        original_reason,
        retry_text,
        retry_conf,
        retry_state,
        retry_reason,
        status=status,
    )
    if status == "accepted_retry_for_translation":
        return retry_text, float(retry_conf or 0.0), retry_bbox, info
    return ocr_text, ocr_conf, bbox, info


def _attach_text_area_assignment(
    region: dict,
    assignment: dict,
    debug_context: dict | None,
    page_id: str,
    ocr_text: str,
    ocr_conf: float,
    *,
    accepted: bool,
    apply_text_area_assignment_to_region,
    build_scoped_ocr_candidate,
) -> None:
    apply_text_area_assignment_to_region(region, assignment)
    if _should_restore_text_area_speech_assignment(assignment, region, ocr_text):
        region["type"] = "speech_bubble"
        region["semantic_class"] = "speech_bubble"
        region["skip_reason"] = ""
        flags = dict(region.get("flags", {}))
        flags["ignore"] = False
        flags["bg_text"] = False
        flags["needs_review"] = False
        region["flags"] = flags
        render = dict(region.get("render", {}))
        render["semantic_class"] = "speech_bubble"
        render["cleanup_mode"] = "bubble"
        render["classification_reason"] = "text_area_speech_container_override"
        render["logical_text_speech_container_override_applied"] = True
        region["render"] = render
        region["logical_text_speech_container_override_applied"] = True
        accepted = True
    elif _should_preserve_review_only_text_area_region(assignment, region, ocr_text, ocr_conf):
        region["type"] = "decorative_text"
        flags = dict(region.get("flags", {}))
        flags["ignore"] = True
        flags["bg_text"] = False
        flags["needs_review"] = True
        region["flags"] = flags
        render = dict(region.get("render", {}))
        render["cleanup_mode"] = "preserve"
        render["classification_reason"] = "text_area_review_only_unknown_not_auto_translated"
        region["render"] = render
        region["translation"] = ""
        accepted = False
    elif _should_preserve_compatibility_unknown_text_area_region(assignment, region, ocr_text, ocr_conf):
        region["type"] = "decorative_text"
        flags = dict(region.get("flags", {}))
        flags["ignore"] = True
        flags["bg_text"] = True
        flags["needs_review"] = True
        region["flags"] = flags
        render = dict(region.get("render", {}))
        render["cleanup_mode"] = "preserve"
        render["classification_reason"] = "text_area_compatibility_unknown_not_auto_translated"
        region["render"] = render
        region["translation"] = ""
        accepted = False
    route_state = _apply_text_area_route_authority(region, assignment, ocr_text, ocr_conf)
    if route_state:
        accepted = _ocr_transaction_state_queues_translation(route_state)
    if debug_context is None:
        return
    candidate = build_scoped_ocr_candidate(
        page_id=page_id,
        region_id=str(region.get("region_id") or ""),
        bbox=region.get("bbox") or [0, 0, 0, 0],
        assignment=assignment,
        ocr_text=ocr_text,
        ocr_confidence=ocr_conf,
        accepted=accepted,
    )
    debug_context.setdefault("scoped_ocr_candidates", []).append(candidate)
    meta = debug_context.setdefault("regions", {}).setdefault(str(region.get("region_id") or ""), {})
    for key, value in region.items():
        if key.startswith("text_area_"):
            meta[key] = value


def _append_empty_ocr_child_evidence(
    regions: list[dict],
    *,
    idx: int,
    polygons: list,
    bbox: list[int],
    det_conf: float,
    ocr_conf: float,
    assignment: dict,
    debug_context: dict | None,
    page_id: str,
    retry_info: dict | None,
    semantic_bg: bool,
    region_type: str,
    apply_text_area_assignment_to_region,
    build_scoped_ocr_candidate,
) -> dict:
    """Retain empty child evidence until parent-boundary OCR owns the source."""

    region = _region_record(
        idx,
        polygons,
        bbox,
        "",
        "",
        det_conf,
        bg_text=semantic_bg,
        needs_review=True,
        ignore=True,
        region_type=region_type,
        ocr_conf=ocr_conf,
        render_updates={
            "cleanup_mode": "preserve",
            "classification_reason": "empty_child_ocr_deferred_to_parent_source_contract",
        },
    )
    _attach_text_area_assignment(
        region,
        assignment,
        debug_context,
        page_id,
        "",
        ocr_conf,
        accepted=False,
        apply_text_area_assignment_to_region=apply_text_area_assignment_to_region,
        build_scoped_ocr_candidate=build_scoped_ocr_candidate,
    )
    _stamp_route_owned_ocr_retry(region, retry_info)
    flags = region.setdefault("flags", {})
    flags["ignore"] = True
    flags["needs_review"] = True
    region["translation"] = ""
    region["ocr_source_deferred_to_parent_boundary"] = True
    region["ocr_source_contract_state"] = "pending_parent_boundary_ocr"
    region["ocr_source_diagnostic"] = "empty_child_ocr"
    region.setdefault("render", {})[
        "ocr_source_deferred_to_parent_boundary"
    ] = True
    regions.append(region)
    return region


def _append_parent_boundary_ocr_source_regions(
    *,
    regions: list[dict],
    text_area_plan,
    page_id: str,
    image_path: str,
    page_image,
    image_size: tuple[int, int] | None,
    ocr_engine,
    settings,
    debug_context: dict | None,
    assign_bbox_to_text_area_plan,
    apply_text_area_assignment_to_region,
    build_scoped_ocr_candidate,
    existing_parent_units: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Attach parent-boundary OCR as source evidence for finalized graph parents."""

    from app.pipeline.debug_artifacts import add_timing, set_count

    # Initial hierarchy sources are provisional until this parent-owned OCR pass.
    _ = existing_parent_units

    plan = _plan_to_dict_for_text_area(text_area_plan)
    graph_plan = plan.get("root_parent_child_plan") if isinstance(plan, dict) else {}
    parent_nodes = graph_plan.get("parent_nodes") if isinstance(graph_plan, dict) else []
    if not parent_nodes:
        return {"attempted": 0, "created": 0, "skipped": 0, "records": []}
    root_nodes = graph_plan.get("root_nodes") if isinstance(graph_plan, dict) else []
    root_nodes_by_id = {
        str(root.get("root_node_id") or ""): root
        for root in root_nodes or []
        if isinstance(root, Mapping) and str(root.get("root_node_id") or "")
    }
    parent_count_by_root: dict[str, int] = {}
    for node in parent_nodes:
        if not isinstance(node, Mapping):
            continue
        root_id = str(node.get("root_node_id") or "")
        if root_id:
            parent_count_by_root[root_id] = parent_count_by_root.get(root_id, 0) + 1

    existing_parent_source_ids = {
        str(region.get("parent_ocr_source_parent_id") or "")
        for region in regions
        if isinstance(region, dict) and bool(region.get("parent_boundary_ocr_source_contract"))
    }
    records: list[dict[str, Any]] = []
    created = 0
    skipped = 0
    reused = 0
    next_index = len(regions)
    for parent_node in parent_nodes:
        if not isinstance(parent_node, Mapping):
            skipped += 1
            continue
        parent_id = str(parent_node.get("parent_node_id") or "")
        if not parent_id or parent_id in existing_parent_source_ids:
            skipped += 1
            continue
        if not bool(parent_node.get("is_explicit_parent_obligation", True)):
            skipped += 1
            continue
        bbox = _clip_controller_bbox([int(round(float(v or 0))) for v in (parent_node.get("bbox") or [])[:4]], image_size)
        if not bbox:
            skipped += 1
            records.append(
                {
                    "parent_id": parent_id,
                    "status": "skipped_invalid_parent_bbox",
                    "bbox": list(parent_node.get("bbox") or []),
                }
            )
            continue
        route = _parent_boundary_ocr_route(parent_node)
        region_id = _parent_boundary_ocr_region_id(parent_id, regions)
        source_scope = "parent_boundary"
        source_stage = "controller_parent_boundary_ocr"
        status = "created"
        ocr_backend_meta: dict[str, Any] = {}
        context_retry: dict[str, Any] = {}
        crop = _crop_image(image_path, bbox, expand_wide=False, image_obj=page_image)
        if crop is None:
            skipped += 1
            records.append({"parent_id": parent_id, "status": "skipped_crop_failed", "bbox": bbox})
            continue
        ocr_start = time.time()
        ocr_text, ocr_conf = _recognize_with_fallback(
            ocr_engine,
            crop,
            settings,
            bbox,
            debug_context=debug_context,
            trace_context={
                "page_id": page_id,
                "region_id": region_id,
                "parent_id": parent_id,
                "root_id": parent_node.get("root_node_id"),
                "attempt_kind": "parent_boundary_ocr_source_contract",
                "text_area_container_id": parent_node.get("container_id"),
                "route_intent": route,
                "ocr_eligible": True,
                "source_bbox": list(bbox),
                "actual_crop_bbox": list(bbox),
                "container_bbox": list(bbox),
            },
        )
        add_timing(debug_context, "ocr_time", time.time() - ocr_start)
        if hasattr(ocr_engine, "backend_metadata"):
            try:
                ocr_backend_meta = dict(ocr_engine.backend_metadata())
            except Exception:
                ocr_backend_meta = {}
        if not ocr_backend_meta:
            ocr_backend_meta = {"ocr_backend": ocr_engine.__class__.__name__}
        ocr_text = _clean_ocr_text(str(ocr_text or ""))
        state, reason = _ocr_transaction_state_for_text_area_route(ocr_text, ocr_conf, route)
        retry_reason = _parent_boundary_ocr_context_retry_reason(ocr_text, state, reason)
        if retry_reason:
                retry_bbox, retry_eligibility = _parent_boundary_ocr_context_bbox(
                    parent_node,
                    root_nodes_by_id=root_nodes_by_id,
                    parent_count_by_root=parent_count_by_root,
                    parent_bbox=bbox,
                    image_size=image_size,
                )
                context_retry = {
                    "trigger_reason": retry_reason,
                    "eligibility": retry_eligibility,
                    "attempted": False,
                    "selected": False,
                    "initial_text": ocr_text,
                    "initial_confidence": float(ocr_conf or 0.0),
                    "initial_state": state,
                    "initial_reason": reason,
                    "parent_bbox": list(bbox),
                    "context_bbox": list(retry_bbox),
                }
                if retry_bbox:
                    retry_crop = _crop_image(
                        image_path,
                        retry_bbox,
                        expand_wide=False,
                        image_obj=page_image,
                    )
                    if retry_crop is not None:
                        retry_start = time.time()
                        retry_text, retry_conf = _recognize_with_fallback(
                            ocr_engine,
                            retry_crop,
                            settings,
                            retry_bbox,
                            debug_context=debug_context,
                            trace_context={
                                "page_id": page_id,
                                "region_id": region_id,
                                "parent_id": parent_id,
                                "root_id": parent_node.get("root_node_id"),
                                "attempt_kind": "parent_boundary_ocr_source_contract_context_retry",
                                "text_area_container_id": parent_node.get("container_id"),
                                "route_intent": route,
                                "ocr_eligible": True,
                                "source_bbox": list(bbox),
                                "actual_crop_bbox": list(retry_bbox),
                                "container_bbox": list(retry_bbox),
                            },
                        )
                        add_timing(debug_context, "ocr_time", time.time() - retry_start)
                        retry_text = _clean_ocr_text(str(retry_text or ""))
                        retry_state, retry_state_reason = _ocr_transaction_state_for_text_area_route(
                            retry_text,
                            retry_conf,
                            route,
                        )
                        selected = _parent_boundary_ocr_candidate_rank(
                            retry_text,
                            retry_state,
                            retry_conf,
                        ) > _parent_boundary_ocr_candidate_rank(ocr_text, state, ocr_conf)
                        context_retry.update(
                            {
                                "attempted": True,
                                "selected": selected,
                                "retry_text": retry_text,
                                "retry_confidence": float(retry_conf or 0.0),
                                "retry_state": retry_state,
                                "retry_reason": retry_state_reason,
                            }
                        )
                        if selected:
                            ocr_text = retry_text
                            ocr_conf = retry_conf
                            state = retry_state
                            reason = retry_state_reason
                            source_stage = "controller_parent_boundary_ocr_context_retry"
                            status = "created_from_parent_owned_context_retry"
                    else:
                        context_retry["eligibility"] = "context_crop_failed"
        terminal_symbol_evidence = {
            "text": ocr_text,
            "raw_text": ocr_text,
            "applied": False,
            "reason": "no_qualifying_terminal_symbol_evidence",
        }
        terminal_symbol_evidence = _reconcile_parent_terminal_symbol_multiplicity(
            ocr_text,
            parent_node=parent_node,
            parent_bbox=bbox,
            regions=regions,
        )
        ocr_text = str(terminal_symbol_evidence.get("text") or ocr_text)
        if terminal_symbol_evidence.get("applied"):
            status = "created_with_terminal_symbol_multiplicity_evidence"
        state, reason = _ocr_transaction_state_for_text_area_route(ocr_text, ocr_conf, route)
        assignment = _parent_boundary_ocr_assignment(
            parent_node,
            assign_bbox_to_text_area_plan(text_area_plan, bbox, detection_source="parent_boundary_ocr_source_contract"),
            bbox,
            route,
        )
        region_type = "background_text" if route == "translate_caption_background" else "speech_bubble"
        quality_state, quality_action, quality_reasons = _parent_boundary_ocr_source_quality(
            state,
            reason,
            ocr_text,
        )
        region = _region_record(
            next_index,
            [_bbox_to_polygon(bbox)],
            bbox,
            ocr_text,
            "",
            1.0,
            bg_text=route == "translate_caption_background",
            needs_review=quality_action != "translate",
            ignore=False,
            region_type=region_type,
            ocr_conf=ocr_conf,
            render_updates={
                "classification_reason": status,
                "cleanup_mode": "local_text_mask" if route == "translate_caption_background" else "bubble",
            },
        )
        next_index += 1
        region["region_id"] = region_id
        apply_text_area_assignment_to_region(region, assignment)
        _stamp_parent_boundary_ocr_source_contract(
            region,
            parent_node=parent_node,
            route=route,
            state=state,
            reason=reason,
            quality_state=quality_state,
            quality_action=quality_action,
            quality_reasons=quality_reasons,
            source_contract_scope=source_scope,
            source_contract_stage=source_stage,
        )
        _stamp_parent_terminal_symbol_evidence(region, terminal_symbol_evidence)
        if context_retry:
            region["parent_ocr_context_retry"] = dict(context_retry)
            region.setdefault("render", {})["parent_ocr_context_retry"] = dict(context_retry)
        for meta_key, meta_value in ocr_backend_meta.items():
            region[meta_key] = meta_value
            region.setdefault("render", {})[meta_key] = meta_value
        if debug_context is not None:
            debug_context.setdefault("scoped_ocr_candidates", []).append(
                build_scoped_ocr_candidate(
                    page_id=page_id,
                    region_id=region_id,
                    bbox=bbox,
                    assignment=assignment,
                    ocr_text=ocr_text,
                    ocr_confidence=ocr_conf,
                    accepted=bool(ocr_text),
                )
            )
            meta = debug_context.setdefault("regions", {}).setdefault(region_id, {})
            meta.update(
                {
                    "parent_boundary_ocr_source_contract": True,
                    "parent_ocr_source_parent_id": parent_id,
                    "parent_ocr_source_quality_state": quality_state,
                    "parent_ocr_source_quality_action": quality_action,
                    "parent_ocr_source_quality_reason_codes": list(quality_reasons),
                    "source_contract_bbox": list(bbox),
                    "source_contract_scope": source_scope,
                    "source_contract_stage": source_stage,
                    "source_contract_ocr_confidence": float(ocr_conf or 0.0),
                    "parent_ocr_context_retry": dict(context_retry),
                    **_parent_terminal_symbol_audit_fields(terminal_symbol_evidence),
                    **ocr_backend_meta,
                }
            )
        regions.append(region)
        created += 1
        records.append(
            {
                "parent_id": parent_id,
                "region_id": region_id,
                "status": status,
                "bbox": bbox,
                "ocr_text": ocr_text,
                "ocr_confidence": float(ocr_conf or 0.0),
                "ocr_transaction_state": state,
                "ocr_transaction_reason": reason,
                "source_quality_state": quality_state,
                "source_quality_action": quality_action,
                "source_quality_reason_codes": list(quality_reasons),
                "source_contract_bbox": list(bbox),
                "source_contract_scope": source_scope,
                "source_contract_stage": source_stage,
                "source_contract_ocr_confidence": float(ocr_conf or 0.0),
                "parent_ocr_context_retry": dict(context_retry),
                **_parent_terminal_symbol_audit_fields(terminal_symbol_evidence),
                "ocr_backend": ocr_backend_meta.get("ocr_backend", ""),
                "ocr_model_path": ocr_backend_meta.get("ocr_model_path", ""),
                "ocr_mmproj_path": ocr_backend_meta.get("ocr_mmproj_path", ""),
                "ocr_endpoint": ocr_backend_meta.get("ocr_endpoint", ""),
                "ocr_prompt_version": ocr_backend_meta.get("ocr_prompt_version", ""),
            }
        )
    result = {"attempted": len(parent_nodes), "created": created, "skipped": skipped, "reused": reused, "records": records}
    if debug_context is not None:
        debug_context["parent_boundary_ocr_source_contract"] = result
        set_count(debug_context, "parent_boundary_ocr_source_contract_created", created)
        set_count(debug_context, "parent_boundary_ocr_source_contract_skipped", skipped)
        set_count(debug_context, "parent_boundary_ocr_source_contract_reused", reused)
    return result


def _parent_boundary_ocr_region_id(parent_id: str, regions: list[dict]) -> str:
    base = re.sub(r"[^A-Za-z0-9_]+", "_", f"parent_ocr_{parent_id}").strip("_")
    existing = {str(region.get("region_id") or "") for region in regions if isinstance(region, dict)}
    if base not in existing:
        return base
    suffix = 2
    while f"{base}_{suffix}" in existing:
        suffix += 1
    return f"{base}_{suffix}"


_PARENT_OCR_IDENTITY_PUNCTUATION = set(
    "。．.、，,！？!?：:；;…‥・･ー〜～~—―－-︙︴〰⋯⋮"
    "「」『』（）()［］[]【】《》〈〉☆★♡♥♪♫♬"
)


def _parent_boundary_ocr_context_retry_reason(text: str, state: str, reason: str) -> str:
    cleaned = _clean_ocr_text(text)
    if not cleaned:
        return "empty_parent_boundary_ocr"
    if state in _OCR_BLOCKER_STATES:
        return reason or "blocked_parent_boundary_ocr"
    body = _non_punct_chars(cleaned)
    has_japanese = any(_is_kana(ch) or 0x4E00 <= ord(ch) <= 0x9FFF for ch in body)
    if state == _OCR_LOW_CONFIDENCE_WARNING_STATE and not has_japanese:
        return reason or "non_japanese_parent_boundary_ocr"
    if state == _OCR_PUNCTUATION_IDENTITY_STATE:
        punctuation = [ch for ch in cleaned if not ch.isspace()]
        if len(punctuation) == 1:
            return "single_punctuation_parent_source_requires_context_verification"
        if punctuation and not all(ch in _PARENT_OCR_IDENTITY_PUNCTUATION for ch in punctuation):
            return "unsupported_parent_boundary_punctuation_placeholder"
    return ""


def _parent_boundary_ocr_context_bbox(
    parent_node: Mapping[str, Any],
    *,
    root_nodes_by_id: Mapping[str, Mapping[str, Any]],
    parent_count_by_root: Mapping[str, int],
    parent_bbox: list[int],
    image_size: tuple[int, int] | None,
) -> tuple[list[int], str]:
    root_id = str(parent_node.get("root_node_id") or "")
    if not root_id or int(parent_count_by_root.get(root_id, 0)) != 1:
        return [], "root_context_requires_single_parent"
    root_node = root_nodes_by_id.get(root_id)
    if not isinstance(root_node, Mapping):
        return [], "root_context_missing"
    root_bbox = _clip_controller_bbox(
        [int(round(float(value or 0))) for value in (root_node.get("bbox") or [])[:4]],
        image_size,
    )
    if not root_bbox or root_bbox == parent_bbox:
        return [], "root_context_not_larger_than_parent"
    if _bbox_inside_ratio_controller(parent_bbox, root_bbox) < 0.95:
        return [], "root_context_does_not_contain_parent"
    parent_area = max(1, int(parent_bbox[2]) * int(parent_bbox[3]))
    root_area = max(1, int(root_bbox[2]) * int(root_bbox[3]))
    if root_area > parent_area * 6:
        return [], "root_context_excessively_broad"
    return root_bbox, "eligible_single_parent_root_context"


def _parent_boundary_ocr_candidate_rank(text: str, state: str, confidence: float) -> tuple[int, int, float]:
    cleaned = _clean_ocr_text(text)
    if not cleaned:
        return (0, 0, 0.0)
    body = _non_punct_chars(cleaned)
    has_japanese = any(_is_kana(ch) or 0x4E00 <= ord(ch) <= 0x9FFF for ch in body)
    if state == _OCR_TRANSLATION_READY_STATE:
        quality = 5
    elif state == _OCR_LOW_CONFIDENCE_WARNING_STATE and has_japanese:
        quality = 4
    elif state == _OCR_PUNCTUATION_IDENTITY_STATE and not _parent_boundary_ocr_context_retry_reason(
        cleaned,
        state,
        "",
    ):
        quality = 3
    elif state == _OCR_LOW_CONFIDENCE_WARNING_STATE:
        quality = 2
    else:
        quality = 1
    return (quality, len(body), float(confidence or 0.0))


def _terminal_emphasis_symbol_run(text: str) -> str:
    value = str(text or "")
    index = len(value)
    while index > 0 and value[index - 1] in _TERMINAL_EMPHASIS_SYMBOL_EXPANSIONS:
        index -= 1
    return value[index:]


def _expanded_emphasis_symbols(text: str) -> str:
    return "".join(
        _TERMINAL_EMPHASIS_SYMBOL_EXPANSIONS.get(char, "")
        for char in str(text or "")
    )


def _parent_terminal_symbol_evidence_rank(
    region: Mapping[str, Any],
    *,
    parent_id: str,
    container_id: str,
    parent_bbox: list[int],
) -> int:
    if bool(region.get("parent_boundary_ocr_source_contract")):
        return 0
    render = region.get("render") if isinstance(region.get("render"), Mapping) else {}
    route = str(
        region.get("text_area_route_intent")
        or render.get("text_area_route_intent")
        or ""
    ).strip()
    if route and route not in {"translate_speech", "translate_caption", "translate_caption_background"}:
        return 0
    region_parent_id = str(
        region.get("parent_logical_text_unit_id")
        or render.get("parent_logical_text_unit_id")
        or region.get("source_text_represented_by_block_id")
        or render.get("source_text_represented_by_block_id")
        or ""
    ).strip()
    region_container_id = str(
        region.get("text_area_container_id")
        or render.get("text_area_container_id")
        or ""
    ).strip()
    region_bbox = _clip_controller_bbox(list(region.get("bbox") or []), None)
    covers_parent = bool(
        region_bbox
        and _bbox_inside_ratio_controller(parent_bbox, region_bbox) >= 0.80
        and _bbox_inside_ratio_controller(region_bbox, parent_bbox) >= 0.80
    )
    same_parent = bool(region_parent_id and region_parent_id == parent_id)
    same_container = bool(container_id and region_container_id == container_id)
    if same_parent and covers_parent:
        return 3
    if same_parent or (same_container and covers_parent):
        return 2
    return 0


def _reconcile_parent_terminal_symbol_multiplicity(
    parent_text: str,
    *,
    parent_node: Mapping[str, Any],
    parent_bbox: list[int],
    regions: list[dict],
) -> dict[str, Any]:
    raw_text = _clean_ocr_text(parent_text)
    parent_run = _terminal_emphasis_symbol_run(raw_text)
    parent_expanded = _expanded_emphasis_symbols(parent_run)
    result: dict[str, Any] = {
        "text": raw_text,
        "raw_text": raw_text,
        "applied": False,
        "reason": "parent_terminal_emphasis_run_missing",
        "parent_run": parent_run,
        "parent_expanded": parent_expanded,
    }
    if not parent_run or not parent_expanded:
        return result

    parent_id = str(parent_node.get("parent_node_id") or "").strip()
    container_id = str(parent_node.get("container_id") or "").strip()
    candidates: list[dict[str, Any]] = []
    for region in regions:
        if not isinstance(region, Mapping):
            continue
        rank = _parent_terminal_symbol_evidence_rank(
            region,
            parent_id=parent_id,
            container_id=container_id,
            parent_bbox=parent_bbox,
        )
        if rank <= 0:
            continue
        evidence_text = _clean_ocr_text(str(region.get("ocr_text") or ""))
        evidence_run = _terminal_emphasis_symbol_run(evidence_text)
        evidence_expanded = _expanded_emphasis_symbols(evidence_run)
        if len(evidence_expanded) <= len(parent_expanded):
            continue
        if not evidence_expanded.startswith(parent_expanded):
            continue
        candidates.append(
            {
                "rank": rank,
                "region_id": str(region.get("region_id") or ""),
                "run": evidence_run,
                "expanded": evidence_expanded,
            }
        )
    if not candidates:
        result["reason"] = "no_longer_compatible_terminal_symbol_evidence"
        return result

    best_rank = max(int(item["rank"]) for item in candidates)
    best = [item for item in candidates if int(item["rank"]) == best_rank]
    exact_runs = {str(item["run"]) for item in best}
    expanded_runs = {str(item["expanded"]) for item in best}
    if len(exact_runs) != 1 or len(expanded_runs) != 1:
        result["reason"] = "conflicting_terminal_symbol_evidence"
        result["conflicting_region_ids"] = sorted(
            str(item["region_id"]) for item in best if str(item["region_id"])
        )
        return result

    evidence_run = next(iter(exact_runs))
    corrected = raw_text[: len(raw_text) - len(parent_run)] + evidence_run
    result.update(
        {
            "text": corrected,
            "applied": True,
            "reason": "longer_exact_terminal_symbol_run_from_parent_attached_evidence",
            "evidence_run": evidence_run,
            "evidence_expanded": next(iter(expanded_runs)),
            "evidence_region_ids": sorted(
                str(item["region_id"]) for item in best if str(item["region_id"])
            ),
            "evidence_rank": best_rank,
        }
    )
    return result


def _parent_terminal_symbol_audit_fields(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    item = dict(evidence or {})
    if not item.get("applied"):
        return {}
    return {
        "parent_ocr_raw_text": str(item.get("raw_text") or ""),
        "parent_terminal_symbol_multiplicity_restored": True,
        "parent_terminal_symbol_before": str(item.get("parent_run") or ""),
        "parent_terminal_symbol_after": str(item.get("evidence_run") or ""),
        "parent_terminal_symbol_expanded_before": str(item.get("parent_expanded") or ""),
        "parent_terminal_symbol_expanded_after": str(item.get("evidence_expanded") or ""),
        "parent_terminal_symbol_evidence_region_ids": list(item.get("evidence_region_ids") or []),
        "parent_terminal_symbol_evidence_reason": str(item.get("reason") or ""),
    }


def _stamp_parent_terminal_symbol_evidence(
    region: dict[str, Any],
    evidence: Mapping[str, Any] | None,
) -> None:
    fields = _parent_terminal_symbol_audit_fields(evidence)
    if not fields:
        return
    region.update(fields)
    render = region.setdefault("render", {})
    render.update(fields)


def _parent_boundary_ocr_route(parent_node: Mapping[str, Any]) -> str:
    kind = str(parent_node.get("parent_kind") or "").strip().lower()
    if kind in {"caption", "caption_background", "background", "background_narration"}:
        return "translate_caption_background"
    return "translate_speech"


def _parent_boundary_ocr_assignment(
    parent_node: Mapping[str, Any],
    assignment: Mapping[str, Any] | None,
    bbox: list[int],
    route: str,
) -> dict[str, Any]:
    result = dict(assignment or {})
    speech = route == "translate_speech"
    cleanup_auth = "cleanup_translate_speech" if speech else "cleanup_translate_background"
    result.update(
        {
            "text_area_container_id": str(parent_node.get("container_id") or result.get("text_area_container_id") or ""),
            "text_area_semantic_unit_id": str(parent_node.get("root_node_id") or result.get("text_area_semantic_unit_id") or ""),
            "text_area_semantic_kind": "speech" if speech else "background_narration",
            "text_area_container_type": "speech_bubble" if speech else "caption_background",
            "text_area_route_intent": route,
            "text_area_cleanup_authorization": cleanup_auth,
            "text_area_authorization_source_stage": "parent_logical_text_unit_ocr_source_contract",
            "text_area_authorization_basis": "finalized_parent_boundary_ocr_source_contract",
            "text_area_authorization_explicit": True,
            "text_area_authorization_field_origin": "parent_logical_text_unit",
            "text_area_semantic_authorization_state": cleanup_auth,
            "text_area_ctd_scope_eligible": True,
            "text_area_comic_text_detector_scope_eligible": True,
            "text_area_ocr_eligible": True,
            "text_area_translation_eligible": True,
            "text_area_render_eligible": True,
            "text_area_cleanup_executable": True,
            "text_area_detection_source": "parent_boundary_ocr_source_contract",
            "text_area_container_bbox": list(bbox),
            "text_area_reason_codes": list(result.get("text_area_reason_codes") or [])
            + ["parent_boundary_ocr_source_contract"],
            "text_area_pre_ocr_authority": True,
        }
    )
    return result


def _parent_boundary_ocr_source_quality(
    state: str,
    reason: str,
    ocr_text: str,
) -> tuple[str, str, list[str]]:
    if state == _OCR_TRANSLATION_READY_STATE:
        return "verified_source", "translate", []
    if state == _OCR_LOW_CONFIDENCE_WARNING_STATE:
        return "usable_source_with_warning", "translate_with_review", [reason or "low_confidence_parent_ocr"]
    if state in {_OCR_PUNCTUATION_IDENTITY_STATE, "ocr_punctuation_only_blocker"}:
        return "punctuation_identity_source", "identity_punctuation", [reason or "parent_ocr_punctuation_identity"]
    body = _non_punct_chars(ocr_text)
    has_japanese = any(_is_kana(ch) or 0x4E00 <= ord(ch) <= 0x9FFF for ch in body)
    if body and has_japanese:
        return "uncertain_source_translated_for_review", "translate_with_review", [reason or "parent_ocr_uncertain"]
    if ocr_text and not body:
        return "punctuation_identity_source", "identity_punctuation", [reason or "parent_ocr_punctuation_identity"]
    return "unusable_source_blocked", "block_review_only", [reason or "empty_parent_ocr"]


def _stamp_parent_boundary_ocr_source_contract(
    region: dict[str, Any],
    *,
    parent_node: Mapping[str, Any],
    route: str,
    state: str,
    reason: str,
    quality_state: str,
    quality_action: str,
    quality_reasons: list[str],
    source_contract_scope: str = "parent_boundary",
    source_contract_stage: str = "controller_parent_boundary_ocr",
) -> None:
    parent_id = str(parent_node.get("parent_node_id") or "")
    root_id = str(parent_node.get("root_node_id") or "")
    region["parent_boundary_ocr_source_contract"] = True
    region["source_region_evidence_only"] = True
    region["source_contract_owner"] = "parent_logical_text_unit_ocr_source_contract"
    region["source_contract_region_id"] = str(region.get("region_id") or "")
    region["parent_ocr_source_contract_owner"] = "parent_logical_text_unit_ocr_source_contract"
    region["parent_ocr_source_parent_id"] = parent_id
    region["parent_ocr_source_root_id"] = root_id
    region["source_contract_bbox"] = list(region.get("bbox") or [])
    region["source_contract_scope"] = source_contract_scope
    region["source_contract_stage"] = source_contract_stage
    confidence = region.get("confidence") if isinstance(region.get("confidence"), dict) else {}
    region["source_contract_ocr_confidence"] = confidence.get("ocr")
    region["parent_source_candidate_scope"] = source_contract_scope
    region["parent_source_candidate_stage"] = source_contract_stage
    region["parent_ocr_source_quality_state"] = quality_state
    region["parent_ocr_source_quality_action"] = quality_action
    region["parent_ocr_source_quality_reason_codes"] = list(quality_reasons)
    region["text_area_ocr_transaction_state"] = state
    region["text_area_ocr_warning_reason"] = reason if quality_action != "translate" else ""
    region["text_area_ocr_blocker_reason"] = reason if quality_state == "unusable_source_blocked" else ""
    region["parent_logical_text_unit_id"] = parent_id
    region["text_block_root_id"] = root_id
    region["logical_text_source_quality_action"] = quality_action
    region["source_conservation_status"] = "complete" if str(region.get("ocr_text") or "").strip() else "unresolved"
    render = region.setdefault("render", {})
    for key in (
        "parent_boundary_ocr_source_contract",
        "source_region_evidence_only",
        "source_contract_owner",
        "source_contract_region_id",
        "parent_ocr_source_contract_owner",
        "parent_ocr_source_parent_id",
        "parent_ocr_source_root_id",
        "source_contract_bbox",
        "source_contract_scope",
        "source_contract_stage",
        "source_contract_ocr_confidence",
        "parent_source_candidate_scope",
        "parent_source_candidate_stage",
        "parent_ocr_source_quality_state",
        "parent_ocr_source_quality_action",
        "parent_ocr_source_quality_reason_codes",
        "text_area_ocr_transaction_state",
        "text_area_ocr_warning_reason",
        "text_area_ocr_blocker_reason",
        "parent_logical_text_unit_id",
        "text_block_root_id",
        "logical_text_source_quality_action",
        "source_conservation_status",
    ):
        render[key] = region.get(key)
    render["source_text"] = str(region.get("ocr_text") or "")
    render["parent_logical_text_unit_source_text"] = str(region.get("ocr_text") or "")


def _should_restore_text_area_speech_assignment(
    assignment: dict,
    region: dict,
    ocr_text: str,
) -> bool:
    if str(assignment.get("text_area_container_type") or "").strip() != "speech_bubble":
        return False
    if str(assignment.get("text_area_route_intent") or "").strip() != "translate_speech":
        return False
    if not _is_text_area_translatable_assignment(assignment):
        return False
    state = str(
        assignment.get("text_area_semantic_authorization_state")
        or assignment.get("text_area_cleanup_authorization")
        or ""
    ).strip()
    if state != "cleanup_translate_speech":
        return False
    if any(str(flag).strip() for flag in assignment.get("text_area_conflict_flags") or []):
        return False
    semantic = str(region.get("type") or region.get("semantic_class") or "").strip().lower()
    if semantic in {"caption", "narration_box"}:
        return False
    text = str(ocr_text or "").strip()
    if not _has_meaningful_japanese_fragment(text):
        return False
    reason_text = " ".join(str(v) for v in assignment.get("text_area_reason_codes") or []).lower()
    if "sfx" in reason_text or "decorative" in reason_text:
        return False
    inside_ratio = _bbox_inside_ratio_controller(region.get("bbox") or [], assignment.get("text_area_container_bbox") or [])
    if inside_ratio <= 0.0 and assignment.get("text_area_scoped_candidate_speech"):
        return True
    return inside_ratio >= 0.70


def _region_text_area_assignment(region: dict) -> dict:
    return {key: region.get(key) for key in _TEXT_AREA_ASSIGNMENT_FIELD_KEYS}


def _restore_text_area_speech_fragments_after_assignment(
    regions: list[dict],
    debug_context: dict | None = None,
) -> dict[str, object]:
    candidate_assignments: dict[str, dict] = {}
    if debug_context is not None:
        for candidate in debug_context.get("scoped_ocr_candidates") or []:
            rid = str(candidate.get("region_id") or "")
            if not rid:
                continue
            candidate_assignments[rid] = {
                "text_area_container_id": candidate.get("text_area_container_id"),
                "text_area_semantic_unit_id": candidate.get("semantic_unit_id") or candidate.get("text_area_container_id"),
                "text_area_semantic_kind": candidate.get("semantic_kind") or "",
                "text_area_container_type": candidate.get("container_type"),
                "text_area_route_intent": candidate.get("route_intent"),
                "text_area_cleanup_authorization": candidate.get("cleanup_authorization") or "",
                "text_area_must_not_mutate": bool(candidate.get("must_not_mutate", False)),
                "text_area_protection_reason": candidate.get("protection_reason") or "",
                "text_area_authorization_source_stage": candidate.get("authorization_source_stage") or candidate.get("source_stage") or "",
                "text_area_authorization_basis": candidate.get("authorization_basis") or "",
                "text_area_authorization_explicit": bool(candidate.get("authorization_explicit", False)),
                "text_area_authorization_field_origin": candidate.get("authorization_field_origin") or "",
                "text_area_semantic_authorization_state": candidate.get("semantic_authorization_state") or "",
                "text_area_ctd_scope_eligible": bool(candidate.get("ctd_scope_eligible", False)),
                "text_area_comic_text_detector_scope_eligible": bool(candidate.get("ctd_scope_eligible", False)),
                "text_area_ocr_eligible": bool(candidate.get("ocr_eligible", True)),
                "text_area_translation_eligible": bool(candidate.get("translation_eligible", False)),
                "text_area_render_eligible": bool(candidate.get("render_eligible", False)),
                "text_area_cleanup_executable": bool(candidate.get("cleanup_executable", False)),
                "text_area_confidence_tier": candidate.get("text_area_confidence_tier")
                or candidate.get("confidence_tier")
                or "strong_model_container",
                "text_area_container_bbox": candidate.get("text_area_container_bbox") or [],
                "text_area_reason_codes": candidate.get("reason_codes") or [],
                "text_area_conflict_flags": candidate.get("conflict_flags") or [],
                "text_area_pre_ocr_authority": bool(candidate.get("text_area_pre_ocr_authority", True)),
                "text_area_enriched_from_region": bool(candidate.get("text_area_enriched_from_region", False)),
                "text_area_scoped_candidate_speech": True,
            }
    restored: list[str] = []
    for region in regions:
        rid = str(region.get("region_id") or "")
        if not rid:
            continue
        flags = region.get("flags") or {}
        semantic = str(region.get("type") or region.get("semantic_class") or "").strip().lower()
        suppressed = bool(flags.get("ignore") or str(region.get("skip_reason") or "").strip() or semantic in {"decorative_text", "sfx"})
        if not suppressed:
            continue
        assignment = _region_text_area_assignment(region)
        candidate_assignment = candidate_assignments.get(rid)
        if candidate_assignment:
            for key, value in candidate_assignment.items():
                if key == "text_area_container_bbox" and not value:
                    continue
                if value not in (None, "", []):
                    assignment[key] = value
                    region[key] = value
        candidate_allows_restore = bool(
            candidate_assignment
            and _should_restore_text_area_speech_assignment(
                candidate_assignment,
                region,
                str(region.get("ocr_text") or ""),
            )
        )
        if not candidate_allows_restore and not _should_restore_text_area_speech_assignment(
            assignment,
            region,
            str(region.get("ocr_text") or ""),
        ):
            continue
        region["type"] = "speech_bubble"
        region["semantic_class"] = "speech_bubble"
        region["skip_reason"] = ""
        flags = region.setdefault("flags", {})
        flags["ignore"] = False
        flags["bg_text"] = False
        flags["needs_review"] = False
        region["logical_text_speech_container_override_applied"] = True
        render = region.setdefault("render", {})
        render["semantic_class"] = "speech_bubble"
        render["cleanup_mode"] = "bubble"
        render["classification_reason"] = "text_area_speech_container_override"
        render["logical_text_speech_container_override_applied"] = True
        restored.append(rid)
    return {
        "logical_text_speech_container_override_count": len(restored),
        "logical_text_speech_container_override_region_ids": restored,
    }


def _has_meaningful_japanese_fragment(text: str) -> bool:
    body = re.sub(r"[\s　。、．,.!?！？…・･ー~〜\\-]+", "", str(text or ""))
    if not body:
        return False
    return any("\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in body)


def _bbox_inside_ratio_controller(inner: list, outer: list) -> float:
    try:
        ix, iy, iw, ih = [float(v) for v in (inner or [0, 0, 0, 0])[:4]]
        ox, oy, ow, oh = [float(v) for v in (outer or [0, 0, 0, 0])[:4]]
    except Exception:
        return 0.0
    if iw <= 0 or ih <= 0 or ow <= 0 or oh <= 0:
        return 0.0
    ix1, iy1 = ix + iw, iy + ih
    ox1, oy1 = ox + ow, oy + oh
    overlap_w = max(0.0, min(ix1, ox1) - max(ix, ox))
    overlap_h = max(0.0, min(iy1, oy1) - max(iy, oy))
    return (overlap_w * overlap_h) / max(1.0, iw * ih)












































































































































def _clip_controller_bbox(bbox: list[int], image_size: tuple[int, int] | None) -> list[int]:
    try:
        x, y, w, h = [int(round(float(v or 0))) for v in bbox[:4]]
    except Exception:
        return []
    if w <= 0 or h <= 0:
        return []
    if image_size:
        img_w = max(1, int(image_size[0] or 1))
        img_h = max(1, int(image_size[1] or 1))
        x0 = max(0, min(img_w - 1, x))
        y0 = max(0, min(img_h - 1, y))
        x1 = max(x0 + 1, min(img_w, x + w))
        y1 = max(y0 + 1, min(img_h, y + h))
        return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]
    return [max(0, x), max(0, y), max(1, w), max(1, h)]


def _should_preserve_review_only_text_area_region(
    assignment: dict,
    region: dict,
    ocr_text: str,
    ocr_conf: float,
) -> bool:
    if not isinstance(assignment, dict):
        return False
    if assignment.get("text_area_container_type") != "unknown_fallback":
        return False
    if assignment.get("text_area_route_intent") != "review_or_fallback":
        return False
    tier = str(assignment.get("text_area_confidence_tier") or "")
    if tier not in {"text_bubble_review_container", "text_free_review_only", "mask_primary_container"}:
        return False
    if str(region.get("type") or "") not in {"speech_bubble", "background_text", "decorative_text"}:
        return False
    cleaned = str(ocr_text or "").strip()
    if not cleaned:
        return True
    body = _non_punct_chars(cleaned)
    has_japanese = any(_is_kana(ch) or 0x4E00 <= ord(ch) <= 0x9FFF for ch in cleaned)
    if not has_japanese:
        return True
    if _is_punct_only(cleaned) or _placeholder_ratio(cleaned) >= 0.2:
        return True
    # Review-only model evidence can request OCR for inspection, but short
    # fragments must not become normal translated speech without stronger
    # container ownership.
    if len(body) < 8:
        return True
    return False


def _should_preserve_compatibility_unknown_text_area_region(
    assignment: dict,
    region: dict,
    ocr_text: str,
    ocr_conf: float,
) -> bool:
    if not isinstance(assignment, dict):
        return False
    if assignment.get("text_area_container_type") != "unknown_fallback":
        return False
    if assignment.get("text_area_detection_source") != "compatibility_fallback":
        return False
    cleaned = str(ocr_text or "").strip()
    if not cleaned:
        return True
    if _is_punct_only(cleaned) or _placeholder_ratio(cleaned) >= 0.18:
        return True
    body = _non_punct_chars(cleaned)
    if len(body) < 4:
        return True
    if float(ocr_conf or 0.0) < 0.70 and not _is_meaningful_background_caption_source(cleaned):
        return True
    return False


def _plan_to_dict_for_text_area(plan) -> dict:
    if plan is None:
        return {}
    if hasattr(plan, "to_dict"):
        try:
            return plan.to_dict()
        except Exception:
            return {}
    return plan if isinstance(plan, dict) else {}


def _caption_recovery_text_is_acceptable(text: str, ocr_conf: float) -> bool:
    cleaned = _clean_ocr_text(text)
    if not cleaned:
        return False
    if _is_punct_only(cleaned) or _placeholder_ratio(cleaned) >= 0.18:
        return False
    body = _non_punct_chars(cleaned)
    if len(body) < 3:
        return False
    contains_kanji = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in body)
    contains_kana = any(_is_kana(ch) for ch in body)
    has_digits = any(ch.isdigit() for ch in cleaned)
    has_caption_marker = any(marker in cleaned for marker in ("日目", "回目", "生活", "最終日", "無人島"))
    if _caption_recovery_text_looks_like_sfx(cleaned, body):
        return False
    if float(ocr_conf or 0.0) < 0.45 and not (has_caption_marker or has_digits):
        return False
    if has_caption_marker and (contains_kanji or has_digits):
        return True
    if has_digits and contains_kanji:
        return True
    if contains_kana and contains_kanji and len(body) >= 3:
        return True
    # Some caption/background narration in manga is kana-only. Accept it only
    # when it has enough body to be a phrase rather than a short impact sound.
    if contains_kana and len(body) >= 6:
        return True
    return False


def _caption_recovery_text_looks_like_sfx(cleaned: str, body: str) -> bool:
    if not body:
        return True
    kana_count = sum(1 for ch in body if _is_kana(ch))
    katakana_count = sum(1 for ch in body if 0x30A0 <= ord(ch) <= 0x30FF)
    unique_body = {ch for ch in body}
    if katakana_count == len(body) and len(body) <= 5:
        return True
    if len(body) <= 4 and kana_count == len(body):
        return True
    if len(body) >= 3 and len(unique_body) <= 2 and kana_count >= max(2, len(body) - 1):
        return True
    if re.fullmatch(r"[\u3040-\u30ffっッー～]+[!?！？ッっー～]*", cleaned) and len(body) <= 5:
        return True
    return False


def _caption_recovery_rejection_reason(text: str, ocr_conf: float) -> str:
    cleaned = _clean_ocr_text(text)
    if not cleaned or _is_punct_only(cleaned) or _placeholder_ratio(cleaned) >= 0.18:
        return "punctuation_or_noise"
    body = _non_punct_chars(cleaned)
    if len(body) < 3:
        return "punctuation_or_noise"
    if _caption_recovery_text_looks_like_sfx(cleaned, body):
        return "rejected_sfx_decorative_art"
    if float(ocr_conf or 0.0) < 0.45:
        return "unsafe_ocr_evidence"
    return "unsafe_ocr_evidence"


def _caption_container_recovery_scope_is_safe(scope: dict, image_size: tuple[int, int]) -> bool:
    if not isinstance(scope, dict):
        return False
    if scope.get("container_type") != "caption_background":
        return False
    if scope.get("route_intent") != "translate_caption_background":
        return False
    if not bool(scope.get("ocr_eligible")) or not bool(scope.get("comic_text_detector_scope_eligible")):
        return False
    if scope.get("conflict_flags"):
        return False
    bbox = scope.get("bbox") or []
    if len(bbox) < 4:
        return False
    x, y, w, h = [int(round(float(v or 0))) for v in bbox[:4]]
    img_w, img_h = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w < 24 or h < 80:
        return False
    reason_text = " ".join(
        str(item)
        for item in list(scope.get("reason_codes") or [])
        + [scope.get("fallback_reason"), scope.get("ocr_eligibility_reason")]
    )
    side_caption = "deterministic_vertical_side_caption_search" in reason_text
    if side_caption:
        if x < img_w * 0.70 or y < img_h * 0.18 or y > img_h * 0.72:
            return False
        if w > img_w * 0.26 or h > img_h * 0.62:
            return False
    else:
        if y > img_h * 0.24:
            return False
        if w > img_w * 0.22 or h > img_h * 0.18:
            return False
    if "caption_background" not in reason_text and "top_caption" not in reason_text and not side_caption:
        return False
    return True


def _caption_component_v4_candidate_groups(
    scope: dict,
    caption_components: list[dict[str, object]],
    component_records: list[dict[str, object]],
    *,
    root_bbox: list[int],
    root_area: int,
    image_size: tuple[int, int],
    page_id: str,
    debug_context: dict | None = None,
) -> list[dict]:
    if len(caption_components) < 3:
        return []
    x, y, w, h = [int(v) for v in root_bbox[:4]]
    art_components = [
        comp for comp in component_records
        if str(comp.get("component_role") or "") == "sfx_decorative_art_like"
    ]
    candidates: list[dict[str, object]] = []
    by_polarity: dict[str, list[dict[str, object]]] = {}
    for comp in caption_components:
        try:
            area = int(comp.get("component_area") or 0)
            bbox = [int(v) for v in (comp.get("bbox") or [])[:4]]
        except Exception:
            continue
        if len(bbox) < 4 or area < 10:
            continue
        by_polarity.setdefault(str(comp.get("component_polarity") or "dark"), []).append(comp)

    def _add_cluster(axis: str, polarity: str, comps: list[dict[str, object]], cluster_index: int) -> None:
        if len(comps) < 2:
            return
        xs = [int(comp["bbox"][0]) for comp in comps]
        ys = [int(comp["bbox"][1]) for comp in comps]
        xe = [int(comp["bbox"][0]) + int(comp["bbox"][2]) for comp in comps]
        ye = [int(comp["bbox"][1]) + int(comp["bbox"][3]) for comp in comps]
        ux0, uy0, ux1, uy1 = min(xs), min(ys), max(xe), max(ye)
        pad = max(3, min(12, int(max(ux1 - ux0, uy1 - uy0) * 0.05)))
        candidate = _clip_controller_bbox([ux0 - pad, uy0 - pad, (ux1 - ux0) + pad * 2, (uy1 - uy0) + pad * 2], image_size)
        if not candidate:
            return
        cx, cy, cw, ch = [int(v) for v in candidate[:4]]
        area = max(1, cw * ch)
        area_ratio = float(area) / float(max(1, root_area))
        text_area = sum(int(comp.get("component_area") or 0) for comp in comps)
        stroke_density = float(text_area) / float(max(1, area))
        width_ratio = float(cw) / float(max(1, w))
        height_ratio = float(ch) / float(max(1, h))
        art_overlap = 0.0
        for art in art_components:
            art_box = art.get("bbox") or []
            if len(art_box) < 4:
                continue
            art_xyxy = [int(art_box[0]), int(art_box[1]), int(art_box[0]) + int(art_box[2]), int(art_box[1]) + int(art_box[3])]
            cand_xyxy = [cx, cy, cx + cw, cy + ch]
            ix0 = max(cand_xyxy[0], art_xyxy[0])
            iy0 = max(cand_xyxy[1], art_xyxy[1])
            ix1 = min(cand_xyxy[2], art_xyxy[2])
            iy1 = min(cand_xyxy[3], art_xyxy[3])
            overlap = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            art_overlap = max(art_overlap, float(overlap) / float(max(1, area)))
        vertical_column = axis == "x" and width_ratio <= 0.58 and height_ratio >= 0.14
        horizontal_band = axis == "y" and height_ratio <= 0.50 and width_ratio >= 0.16
        reasons: list[str] = []
        if area_ratio > 0.48:
            reasons.append("caption_component_v4_candidate_too_large")
        if stroke_density < 0.010:
            reasons.append("caption_component_v4_stroke_density_too_low")
        if stroke_density > 0.52:
            reasons.append("caption_component_v4_stroke_density_artlike")
        if art_overlap > 0.42:
            reasons.append("caption_component_v4_overlaps_large_art_component")
        if not (vertical_column or horizontal_band):
            reasons.append("caption_component_v4_no_column_or_band_alignment")
        if len(comps) < 3 and area_ratio > 0.18:
            reasons.append("caption_component_v4_insufficient_textlike_members")
        reading_order = [
            str(comp.get("component_id") or "")
            for comp in sorted(
                comps,
                key=(
                    (lambda item: (-int(item["bbox"][0]), int(item["bbox"][1])))
                    if vertical_column
                    else (lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))
                ),
            )
        ]
        status = "scheduled_component_v4_ocr" if not reasons else "rejected_component_v4_candidate"
        candidate_id = f"{scope.get('container_id')}_v4_{polarity}_{axis}_{cluster_index}"
        record = {
            "page_id": page_id,
            "text_area_container_id": scope.get("container_id"),
            "parent_root_id": f"tbr_{page_id}_{scope.get('container_id')}",
            "component_id": candidate_id,
            "bbox": candidate,
            "component_bbox": candidate,
            "component_role": "caption_like" if not reasons else "unsafe",
            "component_polarity": polarity,
            "status": status,
            "caption_component_v4_candidate": True,
            "caption_component_v4_candidate_id": candidate_id,
            "caption_component_v4_axis": axis,
            "caption_component_v4_reading_order": reading_order,
            "caption_component_v4_score": round(float((len(comps) * 0.12) + min(0.35, height_ratio if vertical_column else width_ratio) - art_overlap - max(0.0, area_ratio - 0.30)), 3),
            "caption_component_v4_stroke_density": round(float(stroke_density), 3),
            "caption_component_v4_area_ratio": round(float(area_ratio), 3),
            "caption_component_v4_art_overlap_ratio": round(float(art_overlap), 3),
            "rejection_reason": ",".join(reasons),
            "member_component_ids": reading_order,
            "would_change_behavior": False,
        }
        candidates.append(record)

    for polarity, comps in by_polarity.items():
        if len(comps) < 3:
            continue
        for axis, root_span in (("x", w), ("y", h)):
            threshold = max(18, min(46, int(root_span * (0.16 if axis == "x" else 0.13))))
            ordered = sorted(
                comps,
                key=lambda item: int(item["bbox"][0]) + int(item["bbox"][2]) // 2
                if axis == "x"
                else int(item["bbox"][1]) + int(item["bbox"][3]) // 2,
            )
            clusters: list[list[dict[str, object]]] = []
            current_cluster: list[dict[str, object]] = []
            last_center: int | None = None
            for comp in ordered:
                bbox = comp.get("bbox") or []
                if len(bbox) < 4:
                    continue
                center = int(bbox[0]) + int(bbox[2]) // 2 if axis == "x" else int(bbox[1]) + int(bbox[3]) // 2
                if last_center is None or abs(center - last_center) <= threshold:
                    current_cluster.append(comp)
                else:
                    if current_cluster:
                        clusters.append(current_cluster)
                    current_cluster = [comp]
                last_center = center
            if current_cluster:
                clusters.append(current_cluster)
            for idx, cluster in enumerate(clusters):
                _add_cluster(axis, polarity, cluster, idx)

    # Deduplicate nested candidates, preferring the safer, denser subgroup.
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for candidate in candidates:
        if str(candidate.get("status") or "").startswith("rejected"):
            rejected.append(candidate)
            continue
        duplicate = False
        cbox = candidate.get("bbox") or []
        for existing in accepted:
            ebox = existing.get("bbox") or []
            if len(cbox) >= 4 and len(ebox) >= 4 and _overlap_ratio(cbox, ebox) > 0.74:
                duplicate = True
                if float(candidate.get("caption_component_v4_score") or 0.0) > float(existing.get("caption_component_v4_score") or 0.0):
                    existing.update(candidate)
                break
        if not duplicate:
            accepted.append(candidate)

    accepted.sort(key=lambda item: float(item.get("caption_component_v4_score") or 0.0), reverse=True)
    scheduled = accepted[:4]
    overflow = accepted[4:]
    for candidate in overflow:
        candidate = dict(candidate)
        candidate["status"] = "rejected_component_v4_candidate"
        candidate["rejection_reason"] = "caption_component_v4_lower_ranked_candidate"
        rejected.append(candidate)

    if debug_context is not None:
        debug_context.setdefault("caption_component_recovery_candidates", []).extend(scheduled + rejected)

    groups: list[dict] = []
    for candidate in scheduled:
        bbox = [int(v) for v in (candidate.get("bbox") or [])[:4]]
        polygon = _bbox_to_polygon(bbox)
        groups.append(
            {
                "bbox": bbox,
                "polygons": [polygon],
                "conf": 0.58,
                "bg_text": True,
                "text_area_detection_source": "caption_container_text_instance_recovery",
                "caption_component_detection_source": "caption_component_v4_recovery",
                "text_area_caption_recovery": True,
                "text_area_caption_component_recovery": True,
                "text_area_caption_component_v4_recovery": True,
                "caption_component_id": candidate.get("caption_component_v4_candidate_id"),
                "caption_component_role": "caption_like",
                "caption_component_source_polarity": candidate.get("component_polarity"),
                "caption_component_v4_candidate_id": candidate.get("caption_component_v4_candidate_id"),
                "caption_component_v4_reading_order": candidate.get("caption_component_v4_reading_order") or [],
                "caption_component_v4_candidate_bbox": bbox,
            }
        )
    return groups


def _caption_component_recovery_groups_for_scope(
    scope: dict,
    *,
    image_path: str,
    page_image,
    page_id: str,
    image_size: tuple[int, int],
    debug_context: dict | None = None,
) -> list[dict]:
    if not _caption_container_recovery_scope_is_safe(scope, image_size):
        return []
    if not _caption_component_split_scope_should_run(scope):
        return []
    try:
        import numpy as np
        import cv2
        from PIL import Image
    except Exception:
        return []
    bbox = _clip_controller_bbox([int(round(float(v or 0))) for v in (scope.get("bbox") or [])[:4]], image_size)
    if not bbox:
        return []
    x, y, w, h = bbox
    if w < 24 or h < 48:
        return []
    try:
        if page_image is not None:
            crop_img = page_image.crop((x, y, x + w, y + h)).convert("L")
        else:
            with Image.open(image_path) as img:
                crop_img = img.crop((x, y, x + w, y + h)).convert("L")
    except Exception:
        return []
    gray = np.asarray(crop_img, dtype=np.uint8)
    if gray.size <= 0:
        return []

    dark_ratio = float((gray < 110).sum()) / float(max(1, gray.size))
    masks = [("dark", gray < 110)]
    if dark_ratio >= 0.30:
        masks.append(("light", gray > 185))
    component_records: list[dict[str, object]] = []
    caption_components: list[dict[str, object]] = []
    root_area = max(1, w * h)
    for polarity, mask in masks:
        try:
            count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
        except Exception:
            continue
        for idx in range(1, count):
            cx, cy, cw, ch, area = [int(v) for v in stats[idx]]
            if area < 8:
                role = "punctuation_or_noise"
                reason = "component_area_too_small"
            else:
                aspect = max(cw / max(1, ch), ch / max(1, cw))
                area_ratio = float(area) / float(root_area)
                spans_root = cw > w * 0.72 or ch > h * 0.72
                line_like = aspect >= 8.0
                impact_like = area_ratio >= 0.055 or spans_root or (line_like and area >= 60)
                if impact_like:
                    role = "sfx_decorative_art_like"
                    reason = "large_or_line_like_component"
                elif aspect <= 7.0 and area <= max(1400, int(root_area * 0.040)):
                    role = "caption_like"
                    reason = "compact_textlike_component"
                else:
                    role = "unsafe"
                    reason = "component_shape_ambiguous"
            abs_box = [x + cx, y + cy, cw, ch]
            record = {
                "page_id": page_id,
                "text_area_container_id": scope.get("container_id"),
                "parent_root_id": f"tbr_{page_id}_{scope.get('container_id')}",
                "component_id": f"{scope.get('container_id')}_comp_{len(component_records)}",
                "bbox": abs_box,
                "component_bbox": abs_box,
                "component_role": role,
                "component_polarity": polarity,
                "component_area": int(area),
                "component_reason": reason,
                "status": "classified",
                "would_change_behavior": False,
            }
            component_records.append(record)
            if role == "caption_like":
                caption_components.append(record)

    if debug_context is not None and component_records:
        debug_context.setdefault("caption_component_recovery_candidates", []).extend(component_records)
    if len(caption_components) < 3:
        if debug_context is not None:
            debug_context.setdefault("caption_component_recovery_candidates", []).append(
                {
                    "page_id": page_id,
                    "text_area_container_id": scope.get("container_id"),
                    "parent_root_id": f"tbr_{page_id}_{scope.get('container_id')}",
                    "component_id": f"{scope.get('container_id')}_component_split",
                    "bbox": bbox,
                    "component_bbox": bbox,
                    "component_role": "unsafe",
                    "status": "rejected_no_caption_like_component_cluster",
                    "rejection_reason": "fewer_than_three_caption_like_components",
                    "would_change_behavior": False,
                }
            )
        return []

    v4_groups = _caption_component_v4_candidate_groups(
        scope,
        caption_components,
        component_records,
        root_bbox=bbox,
        root_area=root_area,
        image_size=image_size,
        page_id=page_id,
        debug_context=debug_context,
    )
    if v4_groups:
        return v4_groups

    # Prefer one compact root-owned component crop per polarity. This keeps
    # recovery bounded by TextAreaPlan while excluding rain strokes, impact
    # lines, and large art fills from the OCR crop.
    groups: list[dict] = []
    by_polarity: dict[str, list[dict[str, object]]] = {}
    for comp in caption_components:
        by_polarity.setdefault(str(comp.get("component_polarity") or ""), []).append(comp)
    for polarity, comps in by_polarity.items():
        if len(comps) < 3:
            continue
        xs = [int(comp["bbox"][0]) for comp in comps]
        ys = [int(comp["bbox"][1]) for comp in comps]
        xe = [int(comp["bbox"][0]) + int(comp["bbox"][2]) for comp in comps]
        ye = [int(comp["bbox"][1]) + int(comp["bbox"][3]) for comp in comps]
        ux0, uy0, ux1, uy1 = min(xs), min(ys), max(xe), max(ye)
        pad = max(3, min(10, int(max(ux1 - ux0, uy1 - uy0) * 0.04)))
        candidate = _clip_controller_bbox([ux0 - pad, uy0 - pad, (ux1 - ux0) + pad * 2, (uy1 - uy0) + pad * 2], image_size)
        if not candidate:
            continue
        _, _, cw, ch = candidate
        if cw * ch > root_area * 0.72:
            status = "rejected_component_cluster_too_large"
            if debug_context is not None:
                debug_context.setdefault("caption_component_recovery_candidates", []).append(
                    {
                        "page_id": page_id,
                        "text_area_container_id": scope.get("container_id"),
                        "parent_root_id": f"tbr_{page_id}_{scope.get('container_id')}",
                        "component_id": f"{scope.get('container_id')}_{polarity}_cluster",
                        "bbox": candidate,
                        "component_bbox": candidate,
                        "component_role": "unsafe",
                        "status": status,
                        "rejection_reason": "component_cluster_covers_most_of_root",
                        "would_change_behavior": False,
                    }
                )
            continue
        polygon = [
            [float(candidate[0]), float(candidate[1])],
            [float(candidate[0] + candidate[2]), float(candidate[1])],
            [float(candidate[0] + candidate[2]), float(candidate[1] + candidate[3])],
            [float(candidate[0]), float(candidate[1] + candidate[3])],
        ]
        component_id = f"{scope.get('container_id')}_{polarity}_cluster"
        groups.append(
            {
                "bbox": candidate,
                "polygons": [polygon],
                "conf": 0.56,
                "bg_text": True,
                "text_area_detection_source": "caption_container_text_instance_recovery",
                "caption_component_detection_source": "caption_component_recovery",
                "text_area_caption_recovery": True,
                "text_area_caption_component_recovery": True,
                "caption_component_id": component_id,
                "caption_component_role": "caption_like",
                "caption_component_source_polarity": polarity,
                "caption_component_member_ids": [str(comp.get("component_id") or "") for comp in comps],
            }
        )
        if debug_context is not None:
            debug_context.setdefault("caption_component_recovery_candidates", []).append(
                {
                    "page_id": page_id,
                    "text_area_container_id": scope.get("container_id"),
                    "parent_root_id": f"tbr_{page_id}_{scope.get('container_id')}",
                    "component_id": component_id,
                    "bbox": candidate,
                    "component_bbox": candidate,
                    "component_role": "caption_like",
                    "component_polarity": polarity,
                    "status": "scheduled_component_ocr",
                    "member_component_ids": [str(comp.get("component_id") or "") for comp in comps],
                    "would_change_behavior": False,
                }
            )
    return groups


def _caption_component_split_scope_should_run(scope: dict) -> bool:
    reason_text = " ".join(
        str(item)
        for item in list(scope.get("reason_codes") or [])
        + [scope.get("fallback_reason"), scope.get("confidence_tier"), scope.get("ocr_eligibility_reason")]
    ).lower()
    return any(
        token in reason_text
        for token in (
            "caption_background_model_candidate_review",
            "text_free_review_only",
            "ogkalu_text_free_without_kitsumed_mask",
            "root_source_coherence_requires_reconstruction",
            "mixed",
            "review",
        )
    )


def _caption_text_area_ocr_requires_quality_gate(assignment: dict, group: dict) -> bool:
    if group.get("text_area_caption_recovery"):
        return True
    if not isinstance(assignment, dict):
        return False
    if assignment.get("text_area_route_intent") != "translate_caption_background":
        return False
    reason_text = " ".join(
        str(item)
        for item in list(assignment.get("text_area_reason_codes") or [])
        + [
            assignment.get("text_area_fallback_reason"),
            assignment.get("text_area_ocr_eligibility_reason"),
        ]
    )
    return (
        "deterministic_top_band_caption_search" in reason_text
        or "deterministic_vertical_side_caption_search" in reason_text
    )


def _append_caption_container_recovery_groups(
    groups: list,
    text_area_plan,
    *,
    page_id: str,
    image_size: tuple[int, int],
    image_path: str = "",
    page_image=None,
    debug_context: dict | None = None,
) -> list:
    from app.pipeline.text_area_plan import (
        DETECTION_CAPTION_RECOVERY,
        ROUTE_TRANSLATE_CAPTION,
        assign_bbox_to_text_area_plan,
    )

    plan_dict = _plan_to_dict_for_text_area(text_area_plan)
    scopes = plan_dict.get("scopes") or []
    if not scopes:
        return []
    existing_boxes = [list(group.get("bbox") or [0, 0, 0, 0]) for group in groups if isinstance(group.get("bbox"), list)]
    added: list[dict] = []
    recovery_records = debug_context.setdefault("caption_container_recovery_candidates", []) if debug_context is not None else None
    for scope in scopes:
        if not _caption_container_recovery_scope_is_safe(scope, image_size):
            continue
        component_groups = _caption_component_recovery_groups_for_scope(
            scope,
            image_path=image_path,
            page_image=page_image,
            page_id=page_id,
            image_size=image_size,
            debug_context=debug_context,
        )
        for component_group in component_groups:
            component_bbox = [int(round(float(v or 0))) for v in (component_group.get("bbox") or [])[:4]]
            if any(_overlap_ratio(component_bbox, other) > 0.72 for other in existing_boxes):
                continue
            assignment = assign_bbox_to_text_area_plan(
                text_area_plan,
                component_bbox,
                detection_source=DETECTION_CAPTION_RECOVERY,
            )
            if assignment.get("text_area_route_intent") != ROUTE_TRANSLATE_CAPTION:
                continue
            component_group["text_area_assignment"] = assignment
            added.append(component_group)
            existing_boxes.append(component_bbox)
            if recovery_records is not None:
                recovery_records.append(
                    {
                        "page_id": page_id,
                        "text_area_container_id": assignment.get("text_area_container_id") or scope.get("container_id"),
                        "bbox": component_bbox,
                        "status": "scheduled_component_ocr",
                        "reason": "caption_component_split_recovery",
                        "detection_source": "caption_component_recovery",
                        "caption_component_id": component_group.get("caption_component_id"),
                        "caption_component_role": component_group.get("caption_component_role"),
                        "caption_component_v4_candidate_id": component_group.get("caption_component_v4_candidate_id"),
                        "caption_component_v4_reading_order": component_group.get("caption_component_v4_reading_order") or [],
                        "would_change_behavior": False,
                    }
                )
            if debug_context is not None:
                debug_context.setdefault("scoped_detection_candidates", []).append(
                    {
                        "detection_id": f"caption_component_recovery_{component_group.get('caption_component_id') or len(added)}",
                        "page_id": page_id,
                        "bbox": component_bbox,
                        "polygon": component_group.get("polygons", [[]])[0],
                        "confidence": component_group.get("conf"),
                        "text_area_container_id": assignment.get("text_area_container_id"),
                        "container_type": assignment.get("text_area_container_type"),
                        "route_intent": assignment.get("text_area_route_intent"),
                        "ocr_eligible": bool(assignment.get("text_area_ocr_eligible")),
                        "detection_source": "caption_component_recovery",
                        "fallback_reason": assignment.get("text_area_fallback_reason"),
                        "reason_codes": list(assignment.get("text_area_reason_codes") or []) + ["caption_component_text_instance_recovery"],
                        "conflict_flags": list(assignment.get("text_area_conflict_flags") or []),
                        "caption_component_id": component_group.get("caption_component_id"),
                        "caption_component_role": component_group.get("caption_component_role"),
                        "caption_component_v4_candidate_id": component_group.get("caption_component_v4_candidate_id"),
                        "caption_component_v4_reading_order": component_group.get("caption_component_v4_reading_order") or [],
                        "text_area_pre_ocr_authority": bool(assignment.get("text_area_pre_ocr_authority", True)),
                        "text_area_enriched_from_region": bool(assignment.get("text_area_enriched_from_region", False)),
                        "would_change_behavior": False,
                    }
                )
        if component_groups:
            # Component recovery is more precise than the full mixed root crop.
            # If component OCR later fails, the root remains an explicit blocker
            # with component-level evidence instead of accepting the whole root.
            continue
        bbox = [int(round(float(v or 0))) for v in (scope.get("bbox") or [])[:4]]
        if any(_overlap_ratio(bbox, other) > 0.18 for other in existing_boxes):
            continue
        assignment = assign_bbox_to_text_area_plan(
            text_area_plan,
            bbox,
            detection_source=DETECTION_CAPTION_RECOVERY,
        )
        if assignment.get("text_area_route_intent") != ROUTE_TRANSLATE_CAPTION:
            continue
        polygon = [
            [float(bbox[0]), float(bbox[1])],
            [float(bbox[0] + bbox[2]), float(bbox[1])],
            [float(bbox[0] + bbox[2]), float(bbox[1] + bbox[3])],
            [float(bbox[0]), float(bbox[1] + bbox[3])],
        ]
        group = {
            "bbox": bbox,
            "polygons": [polygon],
            "conf": 0.5,
            "bg_text": True,
            "text_area_detection_source": DETECTION_CAPTION_RECOVERY,
            "text_area_assignment": assignment,
            "text_area_caption_recovery": True,
        }
        added.append(group)
        existing_boxes.append(bbox)
        if recovery_records is not None:
            recovery_records.append(
                {
                    "page_id": page_id,
                    "text_area_container_id": assignment.get("text_area_container_id") or scope.get("container_id"),
                    "bbox": bbox,
                    "status": "scheduled",
                    "reason": "caption_container_scoped_ctd_miss",
                    "would_change_behavior": False,
                }
            )
        if debug_context is not None:
            debug_context.setdefault("scoped_detection_candidates", []).append(
                {
                    "detection_id": f"caption_recovery_{assignment.get('text_area_container_id') or len(added)}",
                    "page_id": page_id,
                    "bbox": bbox,
                    "polygon": polygon,
                    "confidence": 0.5,
                    "text_area_container_id": assignment.get("text_area_container_id"),
                    "container_type": assignment.get("text_area_container_type"),
                    "route_intent": assignment.get("text_area_route_intent"),
                    "ocr_eligible": bool(assignment.get("text_area_ocr_eligible")),
                    "detection_source": DETECTION_CAPTION_RECOVERY,
                    "fallback_reason": assignment.get("text_area_fallback_reason"),
                    "reason_codes": list(assignment.get("text_area_reason_codes") or []) + ["caption_container_text_instance_recovery"],
                    "conflict_flags": list(assignment.get("text_area_conflict_flags") or []),
                    "text_area_pre_ocr_authority": bool(assignment.get("text_area_pre_ocr_authority", True)),
                    "text_area_enriched_from_region": bool(assignment.get("text_area_enriched_from_region", False)),
                    "would_change_behavior": False,
                }
            )
    if added:
        groups.extend(added)
    return added


def _append_text_area_activation_completeness_groups(
    groups: list,
    text_area_plan,
    *,
    page_id: str,
    image_size: tuple[int, int],
    debug_context: dict | None = None,
) -> list:
    from app.pipeline.text_area_plan import (
        DETECTION_SCOPED,
        ROUTE_TRANSLATE_CAPTION,
        ROUTE_TRANSLATE_SPEECH,
        assign_bbox_to_text_area_plan,
    )

    plan_dict = _plan_to_dict_for_text_area(text_area_plan)
    scopes = plan_dict.get("scopes") or []
    if not scopes:
        return []
    existing_boxes = [list(group.get("bbox") or [0, 0, 0, 0]) for group in groups if isinstance(group.get("bbox"), list)]
    records = debug_context.setdefault("text_area_activation_completeness_candidates", []) if debug_context is not None else None
    added: list[dict] = []

    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        bbox = [int(round(float(v or 0))) for v in (scope.get("bbox") or [])[:4]]
        if len(bbox) < 4 or bbox[2] <= 2 or bbox[3] <= 2:
            continue
        route = str(scope.get("route_intent") or "")
        ctype = str(scope.get("container_type") or "")
        reason_text = " ".join(
            str(item)
            for item in [
                scope.get("fallback_reason"),
                scope.get("ocr_eligibility_reason"),
                ctype,
                route,
            ]
        ).lower()
        if any(token in reason_text for token in ("sfx", "decorative", "preserve")):
            if records is not None:
                records.append(
                    {
                        "page_id": page_id,
                        "text_area_container_id": scope.get("container_id"),
                        "bbox": bbox,
                        "status": "rejected",
                        "reason": "rejected_sfx_decorative_art",
                    }
                )
            continue

        recovery_kind = ""
        if route == ROUTE_TRANSLATE_SPEECH and ctype == "speech_bubble":
            if str(scope.get("ocr_eligibility_reason") or "") == "speech_activation_completeness_scope_required":
                recovery_kind = "speech"
        elif route == ROUTE_TRANSLATE_CAPTION and ctype == "caption_background":
            if scope.get("fallback_reason") == "caption_background_model_candidate_review":
                top_band_caption = bool(image_size and bbox[1] <= int(image_size[1]) * 0.08)
                if top_band_caption and _caption_container_recovery_scope_is_safe(scope, image_size):
                    recovery_kind = "caption"
                elif records is not None:
                    records.append(
                        {
                            "page_id": page_id,
                            "text_area_container_id": scope.get("container_id"),
                            "bbox": bbox,
                            "status": "rejected",
                            "reason": "route_policy_reject",
                        }
                    )
        if not recovery_kind:
            continue
        if any(_overlap_ratio(bbox, other) > 0.72 for other in existing_boxes):
            if records is not None:
                records.append(
                    {
                        "page_id": page_id,
                        "text_area_container_id": scope.get("container_id"),
                        "bbox": bbox,
                        "status": "skipped",
                        "reason": "existing_text_region_coverage",
                    }
                )
            continue

        assignment = assign_bbox_to_text_area_plan(
            text_area_plan,
            bbox,
            detection_source=DETECTION_SCOPED,
        )
        if recovery_kind == "speech" and assignment.get("text_area_route_intent") != ROUTE_TRANSLATE_SPEECH:
            reject_reason = "route_policy_reject"
        elif recovery_kind == "caption" and assignment.get("text_area_route_intent") != ROUTE_TRANSLATE_CAPTION:
            reject_reason = "route_policy_reject"
        else:
            reject_reason = ""
        if reject_reason:
            if records is not None:
                records.append(
                    {
                        "page_id": page_id,
                        "text_area_container_id": scope.get("container_id"),
                        "bbox": bbox,
                        "status": "rejected",
                        "reason": reject_reason,
                    }
                )
            continue

        polygon = [
            [float(bbox[0]), float(bbox[1])],
            [float(bbox[0] + bbox[2]), float(bbox[1])],
            [float(bbox[0] + bbox[2]), float(bbox[1] + bbox[3])],
            [float(bbox[0]), float(bbox[1] + bbox[3])],
        ]
        group = {
            "bbox": bbox,
            "polygons": [polygon],
            "conf": 0.52 if recovery_kind == "speech" else 0.50,
            "bg_text": recovery_kind == "caption",
            "text_area_detection_source": DETECTION_SCOPED,
            "text_area_assignment": assignment,
            "text_area_activation_completeness_recovery": True,
            "text_area_caption_recovery": recovery_kind == "caption",
        }
        added.append(group)
        existing_boxes.append(bbox)
        if records is not None:
            records.append(
                {
                    "page_id": page_id,
                    "text_area_container_id": assignment.get("text_area_container_id") or scope.get("container_id"),
                    "bbox": bbox,
                    "status": "scheduled",
                    "reason": f"{recovery_kind}_root_activation_completeness_scoped_ocr",
                    "route_intent": assignment.get("text_area_route_intent"),
                    "container_type": assignment.get("text_area_container_type"),
                }
            )
        if debug_context is not None:
            debug_context.setdefault("scoped_detection_candidates", []).append(
                {
                    "detection_id": f"activation_completeness_{assignment.get('text_area_container_id') or len(added)}",
                    "page_id": page_id,
                    "bbox": bbox,
                    "polygon": polygon,
                    "confidence": group["conf"],
                    "text_area_container_id": assignment.get("text_area_container_id"),
                    "container_type": assignment.get("text_area_container_type"),
                    "route_intent": assignment.get("text_area_route_intent"),
                    "ocr_eligible": bool(assignment.get("text_area_ocr_eligible")),
                    "detection_source": "activation_completeness_scoped_ocr",
                    "fallback_reason": assignment.get("text_area_fallback_reason"),
                    "reason_codes": list(assignment.get("text_area_reason_codes") or []) + ["text_area_activation_completeness_recovery"],
                    "conflict_flags": list(assignment.get("text_area_conflict_flags") or []),
                    "text_area_pre_ocr_authority": bool(assignment.get("text_area_pre_ocr_authority", True)),
                    "text_area_enriched_from_region": bool(assignment.get("text_area_enriched_from_region", False)),
                    "would_change_behavior": True,
                }
            )
    if added:
        groups.extend(added)
    return added


def _consolidate_deterministic_caption_groups(
    groups: list,
    text_area_plan,
    *,
    page_id: str,
    image_size: tuple[int, int],
    debug_context: dict | None = None,
) -> list:
    """Merge adjacent scoped caption columns inside deterministic top-band caption containers."""
    from app.pipeline.text_area_plan import (
        DETECTION_SCOPED,
        ROUTE_TRANSLATE_CAPTION,
        assign_bbox_to_text_area_plan,
    )

    plan_dict = _plan_to_dict_for_text_area(text_area_plan)
    scopes_by_id: dict[str, dict] = {}
    for scope in plan_dict.get("scopes") or []:
        cid = str(scope.get("container_id") or "")
        if not cid:
            continue
        if scope.get("route_intent") != ROUTE_TRANSLATE_CAPTION:
            continue
        if not _caption_text_area_ocr_requires_quality_gate(
            {
                "text_area_route_intent": scope.get("route_intent"),
                "text_area_reason_codes": scope.get("reason_codes") or [],
                "text_area_fallback_reason": scope.get("fallback_reason"),
                "text_area_ocr_eligibility_reason": scope.get("ocr_eligibility_reason"),
            },
            {},
        ):
            continue
        scopes_by_id[cid] = scope
    if not scopes_by_id or not groups:
        return groups

    indexed_by_container: dict[str, list[int]] = {}
    assignments_by_index: dict[int, dict] = {}
    for idx, group in enumerate(groups):
        bbox = group.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        detection_source = group.get("text_area_detection_source") or DETECTION_SCOPED
        assignment = assign_bbox_to_text_area_plan(text_area_plan, bbox, detection_source=detection_source)
        cid = str(assignment.get("text_area_container_id") or "")
        if cid not in scopes_by_id:
            continue
        indexed_by_container.setdefault(cid, []).append(idx)
        assignments_by_index[idx] = assignment

    replace_indexes: set[int] = set()
    replacements: list[dict] = []
    records = debug_context.setdefault("caption_container_recovery_candidates", []) if debug_context is not None else None
    for cid, indexes in indexed_by_container.items():
        if len(indexes) < 2:
            continue
        scope_bbox = [int(round(float(v or 0))) for v in (scopes_by_id[cid].get("bbox") or [])[:4]]
        if len(scope_bbox) < 4:
            continue
        boxes = [groups[i].get("bbox") for i in indexes if isinstance(groups[i].get("bbox"), list)]
        if len(boxes) < 2:
            continue
        union = [int(v) for v in boxes[0][:4]]
        for box in boxes[1:]:
            union = _union_box(union, [int(v) for v in box[:4]])
        # Keep this as a caption text-instance consolidation, not a full scope OCR.
        img_w, img_h = image_size
        pad_x = max(4, int(union[2] * 0.04))
        pad_y = max(4, int(union[3] * 0.03))
        x0 = max(scope_bbox[0], union[0] - pad_x)
        y0 = max(scope_bbox[1], union[1] - pad_y)
        x1 = min(scope_bbox[0] + scope_bbox[2], union[0] + union[2] + pad_x)
        y1 = min(scope_bbox[1] + scope_bbox[3], union[1] + union[3] + pad_y)
        x0 = max(0, min(img_w, x0))
        y0 = max(0, min(img_h, y0))
        x1 = max(x0, min(img_w, x1))
        y1 = max(y0, min(img_h, y1))
        consolidated_bbox = [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]
        scope_area = max(1, int(scope_bbox[2]) * int(scope_bbox[3]))
        consolidated_area = int(consolidated_bbox[2]) * int(consolidated_bbox[3])
        if consolidated_area <= 0 or consolidated_area > scope_area * 0.75:
            continue
        # Avoid replacing two distant unrelated marks.
        if consolidated_bbox[2] < 18 or consolidated_bbox[3] < 40:
            continue

        assignment = assignments_by_index.get(indexes[0]) or assign_bbox_to_text_area_plan(
            text_area_plan,
            consolidated_bbox,
            detection_source=DETECTION_SCOPED,
        )
        polygons: list = []
        conf = 0.0
        for i in indexes:
            polygons.extend(groups[i].get("polygons") or [])
            conf = max(conf, float(groups[i].get("conf") or 0.0))
        if not polygons:
            polygons = [_bbox_to_polygon(consolidated_bbox)]
        replacement = {
            "bbox": consolidated_bbox,
            "polygons": polygons,
            "conf": conf or 0.5,
            "bg_text": True,
            "text_area_detection_source": DETECTION_SCOPED,
            "text_area_assignment": assignment,
            "text_area_caption_column_consolidation": True,
        }
        replacements.append(replacement)
        replace_indexes.update(indexes)
        if records is not None:
            records.append(
                {
                    "page_id": page_id,
                    "text_area_container_id": cid,
                    "bbox": consolidated_bbox,
                    "source_bboxes": [list(box[:4]) for box in boxes],
                    "status": "scheduled_scoped_column_consolidation",
                    "reason": "caption_container_scoped_column_consolidation",
                    "would_change_behavior": False,
                }
            )
        if debug_context is not None:
            debug_context.setdefault("scoped_detection_candidates", []).append(
                {
                    "detection_id": f"caption_consolidated_{cid}",
                    "page_id": page_id,
                    "bbox": consolidated_bbox,
                    "polygon": _bbox_to_polygon(consolidated_bbox),
                    "confidence": conf or 0.5,
                    "text_area_container_id": cid,
                    "container_type": assignment.get("text_area_container_type"),
                    "route_intent": assignment.get("text_area_route_intent"),
                    "ocr_eligible": bool(assignment.get("text_area_ocr_eligible")),
                    "detection_source": "scoped_caption_column_consolidation",
                    "fallback_reason": assignment.get("text_area_fallback_reason"),
                    "reason_codes": list(assignment.get("text_area_reason_codes") or []) + [
                        "caption_container_scoped_column_consolidation"
                    ],
                    "conflict_flags": list(assignment.get("text_area_conflict_flags") or []),
                    "text_area_pre_ocr_authority": bool(assignment.get("text_area_pre_ocr_authority", True)),
                    "text_area_enriched_from_region": bool(assignment.get("text_area_enriched_from_region", False)),
                    "would_change_behavior": False,
                }
            )

    if not replace_indexes:
        return groups
    consolidated_groups = [group for idx, group in enumerate(groups) if idx not in replace_indexes]
    consolidated_groups.extend(replacements)
    return consolidated_groups


def _detect_regions_scoped_by_text_area_plan(
    detector,
    image_path: str,
    image_size: tuple[int, int],
    text_area_plan,
    *,
    page_id: str,
    input_size: int = 1024,
    use_gpu: bool = False,
    debug_context: dict | None = None,
):
    from app.pipeline.text_area_plan import (
        DETECTION_BLOCKED,
        DETECTION_SCOPED,
        build_scoped_detection_candidates,
    )

    def _plan_to_dict(plan):
        if plan is None:
            return {}
        if hasattr(plan, "to_dict"):
            try:
                return plan.to_dict()
            except Exception:
                return {}
        return plan if isinstance(plan, dict) else {}

    plan_dict = _plan_to_dict(text_area_plan)
    scopes = [
        scope
        for scope in (plan_dict.get("scopes") or [])
        if bool(scope.get("ocr_eligible", True)) and bool(scope.get("comic_text_detector_scope_eligible", True))
    ]
    if not plan_dict.get("generated"):
        raise PipelineStageTechnicalError(
            stage=PipelineStage.DETECTION,
            code="text_area_plan_artifact_unavailable",
            message="Scoped detection requires a generated TextAreaPlan artifact.",
            page_id=str(page_id or ""),
            operation="detect_regions_scoped_by_text_area_plan",
        )
    if not scopes:
        _record_text_area_fallback_decision(
            debug_context,
            page_id,
            [0, 0, int(image_size[0] or 0), int(image_size[1] or 0)],
            {"text_area_detection_source": DETECTION_BLOCKED, "text_area_ocr_eligible": False},
            "text_area_plan_all_scopes_blocked",
        )
        return [], [], DETECTION_BLOCKED

    image = _read_image_cv(image_path)
    if image is None or not hasattr(detector, "detect_image"):
        raise PipelineStageTechnicalError(
            stage=PipelineStage.DETECTION,
            code="scoped_detector_api_unavailable",
            message="Scoped ComicTextDetector is unavailable.",
            page_id=str(page_id or ""),
            operation="detect_regions_scoped_by_text_area_plan",
        )

    detections: list[tuple[list[list[float]], float]] = []
    img_h, img_w = image.shape[:2]
    try:
        for scope in scopes:
            bbox = scope.get("bbox") or []
            if len(bbox) < 4:
                _record_text_area_fallback_decision(debug_context, page_id, [0, 0, 0, 0], scope, "invalid_scope_bbox")
                continue
            x, y, w, h = [int(round(float(v or 0))) for v in bbox[:4]]
            pad_x = max(8, int(max(0, w) * 0.08))
            pad_y = max(8, int(max(0, h) * 0.08))
            x0 = max(0, min(img_w, x - pad_x))
            y0 = max(0, min(img_h, y - pad_y))
            x1 = max(x0, min(img_w, x + max(0, w) + pad_x))
            y1 = max(y0, min(img_h, y + max(0, h) + pad_y))
            if (x1 - x0) < 2 or (y1 - y0) < 2:
                _record_text_area_fallback_decision(
                    debug_context,
                    page_id,
                    [x0, y0, max(0, x1 - x0), max(0, y1 - y0)],
                    scope,
                    "scope_too_small_for_detector",
                )
                continue
            crop = image[y0:y1, x0:x1]
            try:
                try:
                    scoped = detector.detect_image(crop, input_size=input_size)
                except TypeError:
                    scoped = detector.detect_image(crop)
            except Exception as exc:
                raise PipelineStageTechnicalError(
                    stage=PipelineStage.DETECTION,
                    code="scoped_comic_text_detector_failed",
                    message="Scoped ComicTextDetector could not produce a valid detection artifact.",
                    detail=f"{type(exc).__name__}: {exc}",
                    page_id=str(page_id or ""),
                    operation="detect_regions_scoped_by_text_area_plan",
                    diagnostics=(str(scope.get("scope_id") or ""),),
                ) from exc
            for polygon, conf in scoped or []:
                shifted: list[list[float]] = []
                for point in polygon or []:
                    if point is None or len(point) < 2:
                        continue
                    shifted.append([float(point[0]) + float(x0), float(point[1]) + float(y0)])
                if len(shifted) >= 2:
                    detections.append((shifted, float(conf or 0.0)))
    except Exception:
        raise

    if not detections:
        # A generated TextAreaPlan is the page-area owner. If scoped CTD finds
        # no text in accepted scopes, do not promote the whole page back into
        # normal CTD/OCR; that reopens decorative/title/art areas as speech.
        _record_text_area_fallback_decision(
            debug_context,
            page_id,
            [0, 0, int(image_size[0] or 0), int(image_size[1] or 0)],
            {"text_area_detection_source": DETECTION_BLOCKED, "text_area_ocr_eligible": False},
            "scoped_detector_returned_no_candidates_blocked_no_full_page_fallback",
        )
        return [], [], DETECTION_BLOCKED

    candidates = build_scoped_detection_candidates(
        page_id,
        detections,
        text_area_plan,
        detection_source=DETECTION_SCOPED,
    )
    return detections, candidates, DETECTION_SCOPED


def _reuse_identical_scoped_detection_result(
    detector,
    background_detector,
    detections,
    scoped_detection_candidates,
    text_area_detection_source,
):
    """Reuse a scoped result only when both roles share one detector instance."""
    if background_detector is not detector:
        return None
    return (
        list(detections),
        list(scoped_detection_candidates),
        text_area_detection_source,
    )


def _bubble_acceleration_allowed(settings: object) -> bool:
    value = getattr(settings, "use_gpu", None)
    if type(value) is not bool:
        raise TypeError("settings.use_gpu must be a boolean")
    return value


def _process_page(
    image_path: str,
    detector,
    ocr_engine,
    ollama,
    model: str,
    style_guide: dict,
    context_window: list,
    target_lang: str,
    source_lang: str,
    font_name: str,
    filter_background: bool,
    filter_strength: str,
    translation_cache: dict[str, str],
    background_detector,
    auto_glossary_state,
    image_input_size: int = 1024,
    style_guide_path: str = "",
    allow_ollama_discovery: bool = False,
    discovery_model: str | None = None,
    settings: PipelineSettings | None = None,
    debug_context: dict | None = None,
    stage_callback: Callable[[PipelineStage, str], None] | None = None,
    stage_outcome_callback: Callable[..., None] | None = None,
) -> PageProcessingResult:
    from app.pipeline.debug_artifacts import add_count, add_timing, mark_render_region, mark_translation_plan, set_count
    from app.pipeline.bubble_detection import BubbleDetectionInput, run_bubble_detection
    from app.pipeline.text_block_hierarchy import build_text_block_hierarchy
    from app.pipeline.text_area_plan import (
        DETECTION_COMPATIBILITY_FALLBACK,
        DETECTION_SCOPED,
        ROUTE_TRANSLATE_CAPTION,
        apply_text_area_assignment_to_region,
        assign_bbox_to_text_area_plan,
        build_scoped_ocr_candidate,
        build_scoped_detection_candidates,
        build_text_area_plan,
        enrich_text_area_plan_with_region_records,
        finalize_text_area_plan_with_ctd_boundary_evidence,
    )

    def notify_stage(stage: PipelineStage, detail: str) -> None:
        if stage_callback is None:
            return
        try:
            stage_callback(stage, detail)
        except Exception:
            # Presentation status must never become a pipeline dependency.
            logger.debug("Pipeline stage callback failed", exc_info=True)

    def publish_stage_outcome(
        *,
        stage: PipelineStage,
        state: PipelineStageOutcomeState,
        parent_ids: Iterable[str] = (),
        artifact_kind: str = "",
        artifact_summary: Mapping[str, Any] | None = None,
        diagnostics: Iterable[str] = (),
    ) -> None:
        if stage_outcome_callback is None:
            return
        stage_outcome_callback(
            stage=stage,
            state=state,
            parent_ids=tuple(parent_ids),
            artifact_kind=artifact_kind,
            artifact_summary=dict(artifact_summary or {}),
            diagnostics=tuple(diagnostics),
        )

    def fail_required_ocr(region_id: str, bbox: Iterable[Any], reason: str) -> None:
        raise PipelineStageTechnicalError(
            stage=PipelineStage.OCR,
            code="ocr_required_source_unavailable",
            message="OCR could not produce source text for an admitted workflow region.",
            detail=str(reason or "ocr source unavailable"),
            page_id=str(page_id or ""),
            parent_id=str(region_id or ""),
            operation="recognize_parent_source",
            artifact_summary={"bbox": list(bbox or [])},
        )

    # Initialize Filter
    text_filter = TextFilter(settings)

    if not image_path or not os.path.exists(image_path):
        raise PipelineStageTechnicalError(
            stage=PipelineStage.DETECTION,
            code="source_page_missing",
            message="The source page is unavailable.",
            detail=str(image_path or ""),
            operation="load_source_page",
        )
    image_load_start = time.time()
    image_size = _get_image_size(image_path)
    page_image = _load_image_for_crop(image_path)
    add_timing(debug_context, "image_loading_time", time.time() - image_load_start)
    page_id = os.path.splitext(os.path.basename(image_path))[0]
    text_area_plan = None
    ctd_segmentation_result = None
    text_area_plan_start = time.time()
    notify_stage(PipelineStage.DETECTION, "Detecting page text areas")
    try:
        bubble_detection_start = time.perf_counter()
        bubble_detection_result = run_bubble_detection(
            BubbleDetectionInput(
                page_id=page_id,
                image_path=image_path,
                image_size=image_size,
                regions=[],
                mode="default_text_area_plan",
                allow_acceleration=_bubble_acceleration_allowed(settings),
            )
        )
        add_timing(debug_context, "bubble_detection_time", time.perf_counter() - bubble_detection_start)
        text_area_plan_build_start = time.perf_counter()
        text_area_plan = build_text_area_plan(
            page_id,
            image_path,
            image_size,
            bubble_detection_result,
            current_region_records=None,
            finalize_graph=False,
        )
        add_timing(
            debug_context,
            "text_area_plan_build_time",
            time.perf_counter() - text_area_plan_build_start,
        )
        if debug_context is not None:
            debug_context["bubble_detection_pre_ocr"] = bubble_detection_result.to_dict()
            debug_context["text_area_plan_semantic_pre_ctd"] = text_area_plan.to_dict()
            set_count(debug_context, "text_area_plan_containers", len(text_area_plan.containers))
            set_count(debug_context, "text_area_plan_scopes", len(text_area_plan.scopes))
    except Exception as exc:
        if debug_context is not None:
            debug_context["text_area_plan_error"] = f"{type(exc).__name__}: {exc}"
            debug_context["text_area_plan"] = None
        raise
    add_timing(debug_context, "text_area_plan_time", time.time() - text_area_plan_start)
    detect_start = time.time()
    detections, scoped_detection_candidates, text_area_detection_source = _detect_regions_scoped_by_text_area_plan(
        detector,
        image_path,
        image_size,
        text_area_plan,
        page_id=page_id,
        input_size=image_input_size,
        use_gpu=bool(settings and settings.use_gpu),
        debug_context=debug_context,
    )
    primary_detection_elapsed = time.time() - detect_start
    add_timing(debug_context, "detection_time", primary_detection_elapsed)
    add_timing(debug_context, "detection_primary_time", primary_detection_elapsed)
    ctd_parent_boundary_start = time.perf_counter()
    full_page_parent_boundary_candidates: list[dict[str, Any]] = []
    if hasattr(detector, "detect_with_segmentation"):
        refinement_scopes = _text_area_ctd_refinement_scope_geometry(
            text_area_plan,
            image_size=image_size,
        )
        try:
            try:
                ctd_segmentation_result = detector.detect_with_segmentation(
                    image_path,
                    input_size=image_input_size,
                    keep_undetected_mask=True,
                    refinement_scopes=refinement_scopes,
                )
            except TypeError:
                ctd_segmentation_result = detector.detect_with_segmentation(image_path)
        except Exception as exc:
            raise PipelineStageTechnicalError(
                stage=PipelineStage.DETECTION,
                code="ctd_parent_boundary_provider_failed",
                message="ComicTextDetector could not provide parent-boundary geometry.",
                detail=f"{type(exc).__name__}: {exc}",
                page_id=page_id,
                operation="detect_ctd_parent_boundaries",
            ) from exc
        full_page_parent_boundary_candidates = build_scoped_detection_candidates(
            page_id,
            list(getattr(ctd_segmentation_result, "detections", []) or []),
            text_area_plan,
            detection_source="comic_text_detector_parent_boundary_evidence",
        )
        for index, candidate in enumerate(full_page_parent_boundary_candidates):
            candidate["detection_id"] = "ctd_full_{:04d}".format(index)
    text_area_plan = finalize_text_area_plan_with_ctd_boundary_evidence(
        text_area_plan,
        scoped_detection_candidates=scoped_detection_candidates,
        full_page_detection_candidates=full_page_parent_boundary_candidates,
    )
    add_timing(
        debug_context,
        "ctd_parent_boundary_graph_finalization_time",
        time.perf_counter() - ctd_parent_boundary_start,
    )
    if debug_context is not None:
        debug_context["ctd_parent_boundary_candidates"] = full_page_parent_boundary_candidates
        debug_context["text_area_plan"] = text_area_plan.to_dict()
        debug_context["text_area_plan_pre_ocr"] = text_area_plan.to_dict()
    if text_area_plan is not None and hasattr(text_area_plan, "runtime"):
        try:
            text_area_plan.runtime.true_scoped_detector_available = text_area_detection_source == DETECTION_SCOPED
            text_area_plan.runtime.compatibility_mode = (
                "scoped_detector_by_text_area_plan"
                if text_area_detection_source == DETECTION_SCOPED
                else "scoped_detector_returned_no_candidates"
            )
        except Exception:
            pass
    if debug_context is not None:
        if text_area_plan is not None and hasattr(text_area_plan, "to_dict"):
            debug_context["text_area_plan"] = text_area_plan.to_dict()
            debug_context.setdefault("text_area_plan_pre_ocr", text_area_plan.to_dict())
        debug_context["scoped_detection_candidates"] = scoped_detection_candidates
        debug_context["text_area_detection_source"] = text_area_detection_source

    merge = getattr(detector, "merge_mode", "auto") != "none"
    grouping_start = time.time()
    groups = _merge_detections(detections, image_size, merge=merge)
    groups = _sort_groups(groups)
    if not groups:
        groups = [{"bbox": _polygon_to_bbox(p), "polygons": [p], "conf": float(c or 0.0)} for p, c in detections]
    bubble_boxes = [g["bbox"] for g in groups]
    add_timing(debug_context, "grouping_time", time.time() - grouping_start)
    if background_detector is not None:
        bg_detect_start = time.time()
        reused_background_result = _reuse_identical_scoped_detection_result(
            detector,
            background_detector,
            detections,
            scoped_detection_candidates,
            text_area_detection_source,
        )
        if reused_background_result is None:
            bg_detections, bg_scoped_candidates, bg_detection_source = _detect_regions_scoped_by_text_area_plan(
                background_detector,
                image_path,
                image_size,
                text_area_plan,
                page_id=page_id,
                input_size=image_input_size,
                use_gpu=bool(settings and settings.use_gpu),
                debug_context=debug_context,
            )
        else:
            bg_detections, bg_scoped_candidates, bg_detection_source = reused_background_result
            add_count(debug_context, "detection_background_reuse_count", 1)
            if debug_context is not None:
                debug_context["background_scoped_detection_reused"] = True
        background_detection_elapsed = time.time() - bg_detect_start
        add_timing(debug_context, "detection_time", background_detection_elapsed)
        add_timing(debug_context, "detection_background_time", background_detection_elapsed)
        if debug_context is not None and bg_scoped_candidates:
            debug_context.setdefault("scoped_detection_candidates", []).extend(bg_scoped_candidates)
        grouping_start = time.time()
        for polygon, conf in bg_detections:
            try:
                bbox = _polygon_to_bbox(polygon)
            except Exception:
                continue
            bg_assignment = assign_bbox_to_text_area_plan(
                text_area_plan,
                bbox,
                detection_source=bg_detection_source,
            )
            caption_column_candidate = _caption_text_area_ocr_requires_quality_gate(bg_assignment, {})
            if any(_overlap_ratio(bbox, bb) > 0.2 for bb in bubble_boxes) and not caption_column_candidate:
                continue
            groups.append(
                {
                    "bbox": bbox,
                    "polygons": [polygon],
                    "conf": float(conf or 0.0),
                    "bg_text": True,
                    "text_area_detection_source": bg_detection_source,
                    "text_area_assignment": bg_assignment,
                }
            )
        add_timing(debug_context, "grouping_time", time.time() - grouping_start)
    grouping_start = time.time()
    groups = _dedupe_groups(groups)
    groups = _sort_groups(groups)
    groups = _consolidate_deterministic_caption_groups(
        groups,
        text_area_plan,
        page_id=page_id,
        image_size=image_size,
        debug_context=debug_context,
    )
    groups = _dedupe_groups(groups)
    groups = _sort_groups(groups)
    caption_recovery_groups = _append_caption_container_recovery_groups(
        groups,
        text_area_plan,
        page_id=page_id,
        image_size=image_size,
        image_path=image_path,
        page_image=page_image,
        debug_context=debug_context,
    )
    if caption_recovery_groups:
        groups = _sort_groups(groups)
    activation_completeness_groups = _append_text_area_activation_completeness_groups(
        groups,
        text_area_plan,
        page_id=page_id,
        image_size=image_size,
        debug_context=debug_context,
    )
    set_count(debug_context, "text_area_activation_completeness_groups", len(activation_completeness_groups))
    if activation_completeness_groups:
        groups = _sort_groups(groups)
    for group in groups:
        group_detection_source = group.get("text_area_detection_source") or text_area_detection_source
        assignment = assign_bbox_to_text_area_plan(
            text_area_plan,
            group.get("bbox") or [0, 0, 0, 0],
            detection_source=group_detection_source,
        )
        group["text_area_assignment"] = assignment
        if assignment.get("text_area_detection_source") == DETECTION_COMPATIBILITY_FALLBACK:
            _record_text_area_fallback_decision(
                debug_context,
                page_id,
                group.get("bbox") or [0, 0, 0, 0],
                assignment,
                "compatibility_detector_fallback_after_scoped_detector_unavailable_or_unsafe",
            )
    add_timing(debug_context, "grouping_time", time.time() - grouping_start)
    set_count(debug_context, "detected_regions", len(groups))
    detection_diagnostics: list[str] = []
    if bool(getattr(bubble_detection_result, "provider_fallback_used", False)):
        detection_diagnostics.append("bubble_detection_provider_fallback")
    detection_diagnostics.extend(
        str(getattr(reason, "reason", "") or "")
        for reason in getattr(text_area_plan, "fallback_reasons", []) or []
        if str(getattr(reason, "reason", "") or "")
    )
    publish_stage_outcome(
        stage=PipelineStage.DETECTION,
        state=(
            PipelineStageOutcomeState.VALID_WITH_DIAGNOSTICS
            if detection_diagnostics
            else PipelineStageOutcomeState.VALID
        ),
        artifact_kind="text_area_plan_and_scoped_detections",
        artifact_summary={
            "width": int(image_size[0] or 0),
            "height": int(image_size[1] or 0),
            "detection_count": len(groups),
            "text_area_plan": text_area_plan.to_dict(),
        },
        diagnostics=detection_diagnostics,
    )
    notify_stage(PipelineStage.OCR, "Recognizing source text")
    regions = []
    pending_texts: dict[str, list[str]] = {}
    glossary_texts: list[str] = []
    for idx, group in enumerate(groups):
        bbox = group["bbox"]
        polygons = group["polygons"]
        det_conf = group["conf"]
        text_area_assignment = group.get("text_area_assignment") or assign_bbox_to_text_area_plan(text_area_plan, bbox)
        route_authority = _is_text_area_translatable_assignment(text_area_assignment)
        if not bool(text_area_assignment.get("text_area_ocr_eligible", True)):
            _record_text_area_fallback_decision(
                debug_context,
                page_id,
                bbox,
                text_area_assignment,
                "text_area_plan_blocked_normal_ocr",
            )
            continue
        is_bg_group = bool(group.get("bg_text")) or text_area_assignment.get("text_area_route_intent") == ROUTE_TRANSLATE_CAPTION
        if is_bg_group:
            crop = _crop_image(image_path, bbox, image_obj=page_image)
            if crop is None:
                if route_authority:
                    fail_required_ocr(f"r{idx:03d}", bbox, "ocr crop unavailable")
                continue
            ocr_start = time.time()
            ocr_text, ocr_conf = _recognize_with_fallback(
                ocr_engine,
                crop,
                settings,
                bbox,
                debug_context=debug_context,
                trace_context=_ocr_trace_context_from_assignment(
                    page_id=page_id,
                    region_id=f"r{idx:03d}",
                    bbox=bbox,
                    assignment=text_area_assignment,
                    attempt_kind="caption_background_scoped_ocr",
                ),
            )
            add_timing(debug_context, "ocr_time", time.time() - ocr_start)
            retry_info = None
            ocr_text, ocr_conf, retry_bbox, retry_info = _try_route_owned_scoped_ocr_retry(
                image_path=image_path,
                page_image=page_image,
                image_size=image_size,
                bbox=bbox,
                assignment=text_area_assignment,
                ocr_text=ocr_text,
                ocr_conf=ocr_conf,
                ocr_engine=ocr_engine,
                settings=settings,
                debug_context=debug_context,
                page_id=page_id,
                region_id=f"r{idx:03d}",
                attempt_kind="caption_background_scoped_ocr",
            )
            if retry_info and retry_info.get("status") == "accepted_retry_for_translation":
                bbox = list(retry_bbox)
                polygons = [_bbox_to_polygon(bbox)]
                group["bbox"] = bbox
                group["polygons"] = polygons
                group["route_owned_ocr_retry"] = retry_info
            caption_quality_gate = _caption_text_area_ocr_requires_quality_gate(text_area_assignment, group)
            if not ocr_text:
                if route_authority:
                    _record_text_area_fallback_decision(
                        debug_context,
                        page_id,
                        bbox,
                        text_area_assignment,
                        "ocr_empty_blocker",
                    )
                    _append_empty_ocr_child_evidence(
                        regions,
                        idx=idx,
                        polygons=polygons,
                        bbox=bbox,
                        det_conf=det_conf,
                        ocr_conf=ocr_conf,
                        assignment=text_area_assignment,
                        debug_context=debug_context,
                        page_id=page_id,
                        retry_info=retry_info,
                        semantic_bg=True,
                        region_type="background_text",
                        apply_text_area_assignment_to_region=apply_text_area_assignment_to_region,
                        build_scoped_ocr_candidate=build_scoped_ocr_candidate,
                    )
                if caption_quality_gate and debug_context is not None:
                    debug_context.setdefault("caption_container_recovery_candidates", []).append(
                        {
                            "page_id": page_id,
                            "text_area_container_id": text_area_assignment.get("text_area_container_id"),
                            "bbox": bbox,
                            "status": "rejected_no_ocr_text",
                            "detection_source": group.get("text_area_detection_source"),
                            "caption_component_id": group.get("caption_component_id"),
                            "caption_component_role": group.get("caption_component_role"),
                            "caption_component_source_polarity": group.get("caption_component_source_polarity"),
                            "caption_component_v4_candidate_id": group.get("caption_component_v4_candidate_id"),
                            "caption_component_v4_reading_order": group.get("caption_component_v4_reading_order") or [],
                            "ocr_transaction_state": "ocr_empty_blocker" if route_authority else "",
                            "would_change_behavior": False,
                        }
                    )
                continue
            if caption_quality_gate and not route_authority and not _caption_recovery_text_is_acceptable(ocr_text, ocr_conf):
                rejection_reason = _caption_recovery_rejection_reason(ocr_text, ocr_conf)
                _record_text_area_fallback_decision(
                    debug_context,
                    page_id,
                    bbox,
                    text_area_assignment,
                    rejection_reason
                    if group.get("text_area_activation_completeness_recovery")
                    else (
                        "caption_container_text_instance_recovery_rejected_ocr_quality"
                        if group.get("text_area_caption_recovery")
                        else "caption_container_scoped_ocr_rejected_ocr_quality"
                    ),
                )
                if debug_context is not None:
                    debug_context.setdefault("caption_container_recovery_candidates", []).append(
                        {
                            "page_id": page_id,
                            "text_area_container_id": text_area_assignment.get("text_area_container_id"),
                            "bbox": bbox,
                            "status": (
                                "rejected_ocr_quality"
                                if group.get("text_area_caption_recovery")
                                else "rejected_scoped_ocr_quality"
                            ),
                            "detection_source": group.get("text_area_detection_source"),
                            "caption_component_id": group.get("caption_component_id"),
                            "caption_component_role": group.get("caption_component_role"),
                            "caption_component_source_polarity": group.get("caption_component_source_polarity"),
                            "caption_component_v4_candidate_id": group.get("caption_component_v4_candidate_id"),
                            "caption_component_v4_reading_order": group.get("caption_component_v4_reading_order") or [],
                            "ocr_text": ocr_text,
                            "ocr_confidence": float(ocr_conf or 0.0),
                            "rejection_reason": rejection_reason,
                            "would_change_behavior": False,
                        }
                    )
                continue
            if not route_authority:
                _record_text_area_fallback_decision(
                    debug_context,
                    page_id,
                    bbox,
                    text_area_assignment,
                    "caption_container_scoped_ocr_rejected_without_route_authority",
                )
                if debug_context is not None:
                    debug_context.setdefault("caption_container_recovery_candidates", []).append(
                        {
                            "page_id": page_id,
                            "text_area_container_id": text_area_assignment.get("text_area_container_id"),
                            "bbox": bbox,
                            "status": "rejected_without_route_authority",
                            "detection_source": group.get("text_area_detection_source"),
                            "caption_component_id": group.get("caption_component_id"),
                            "caption_component_role": group.get("caption_component_role"),
                            "caption_component_source_polarity": group.get("caption_component_source_polarity"),
                            "caption_component_v4_candidate_id": group.get("caption_component_v4_candidate_id"),
                            "caption_component_v4_reading_order": group.get("caption_component_v4_reading_order") or [],
                            "ocr_text": ocr_text,
                            "ocr_confidence": float(ocr_conf or 0.0),
                            "rejection_reason": "text_area_assignment_not_translation_authorized",
                            "would_change_behavior": False,
                        }
                    )
                continue
            if caption_quality_gate and debug_context is not None:
                debug_context.setdefault("caption_container_recovery_candidates", []).append(
                    {
                        "page_id": page_id,
                        "text_area_container_id": text_area_assignment.get("text_area_container_id"),
                        "bbox": bbox,
                        "status": (
                            "accepted_ocr_quality"
                            if group.get("text_area_caption_recovery")
                            else "accepted_scoped_ocr_quality"
                        ),
                        "detection_source": group.get("text_area_detection_source"),
                        "caption_component_id": group.get("caption_component_id"),
                        "caption_component_role": group.get("caption_component_role"),
                        "caption_component_source_polarity": group.get("caption_component_source_polarity"),
                        "caption_component_v4_candidate_id": group.get("caption_component_v4_candidate_id"),
                        "caption_component_v4_reading_order": group.get("caption_component_v4_reading_order") or [],
                        "ocr_text": ocr_text,
                        "ocr_confidence": float(ocr_conf or 0.0),
                        "would_change_behavior": False,
                    }
                )
            add_count(debug_context, "ocr_results")
            region_type, semantic_bg, semantic_ignore, semantic_review, render_updates = _classify_semantic_region(
                ocr_text,
                bbox,
                image_size,
                det_conf,
                ocr_conf,
                page_image,
                text_filter,
                initial_bg=True,
                text_area_assignment=text_area_assignment,
            )
            if text_area_assignment.get("text_area_route_intent") == ROUTE_TRANSLATE_CAPTION:
                region_type = "background_text"
                semantic_bg = True
                semantic_ignore = False
                semantic_review = bool(semantic_review)
                render_updates = dict(render_updates or {})
                render_updates["cleanup_mode"] = "local_text_mask"
                render_updates["classification_reason"] = "caption_background_ownership_accepted"
                render_updates["caption_background_ownership_status"] = "accepted_caption_background"
                render_updates["caption_background_ownership_reason"] = "text_area_plan_caption_route_scoped_ocr"
            skip_text = _should_skip_text(ocr_text, bbox, image_size) if semantic_bg else False
            region = _region_record(
                idx,
                polygons,
                bbox,
                ocr_text,
                "",
                det_conf,
                bg_text=semantic_bg,
                needs_review=semantic_review or skip_text,
                ignore=semantic_ignore or skip_text,
                region_type=region_type,
                ocr_conf=ocr_conf,
                render_updates=render_updates,
            )
            _attach_text_area_assignment(
                region,
                text_area_assignment,
                debug_context,
                page_id,
                ocr_text,
                ocr_conf,
                accepted=not region.get("ignore"),
                apply_text_area_assignment_to_region=apply_text_area_assignment_to_region,
                build_scoped_ocr_candidate=build_scoped_ocr_candidate,
            )
            _stamp_route_owned_ocr_retry(region, retry_info)
            if text_area_assignment.get("text_area_route_intent") == ROUTE_TRANSLATE_CAPTION:
                region["type"] = "background_text"
                flags = region.setdefault("flags", {})
                flags["bg_text"] = True
                render = region.setdefault("render", {})
                render["cleanup_mode"] = "local_text_mask"
                render["classification_reason"] = "caption_background_ownership_accepted"
                render["caption_background_ownership_status"] = "accepted_caption_background"
                render["caption_background_ownership_reason"] = "text_area_plan_caption_route_scoped_ocr"
            if group.get("text_area_caption_component_recovery"):
                region["caption_component_id"] = group.get("caption_component_id")
                region["caption_component_role"] = group.get("caption_component_role")
                region["caption_component_source_polarity"] = group.get("caption_component_source_polarity")
                region["caption_component_v4_candidate_id"] = group.get("caption_component_v4_candidate_id")
                region["caption_component_v4_reading_order"] = group.get("caption_component_v4_reading_order") or []
                region.setdefault("render", {})["caption_component_recovery"] = True
            regions.append(region)
            if _region_translation_blocked_by_ocr_transaction(region):
                continue
            if region.get("flags", {}).get("ignore"):
                continue
            glossary_texts.append(ocr_text)
            cached = translation_cache.get(ocr_text)
            if cached is not None:
                region["translation"] = cached
            else:
                pending_texts.setdefault(ocr_text, []).append(region["region_id"])
            continue
        bg_text, needs_review = _classify_region(
            bbox,
            image_size,
            det_conf,
            filter_background,
            filter_strength,
        )
        if bg_text:
            crop = _crop_image(image_path, bbox, image_obj=page_image)
            if crop is None:
                if route_authority:
                    fail_required_ocr(f"r{idx:03d}", bbox, "ocr crop unavailable")
                continue
            ocr_start = time.time()
            ocr_text, ocr_conf = _recognize_with_fallback(
                ocr_engine,
                crop,
                settings,
                bbox,
                debug_context=debug_context,
                trace_context=_ocr_trace_context_from_assignment(
                    page_id=page_id,
                    region_id=f"r{idx:03d}",
                    bbox=bbox,
                    assignment=text_area_assignment,
                    attempt_kind="background_scoped_ocr",
                ),
            )
            add_timing(debug_context, "ocr_time", time.time() - ocr_start)
            retry_info = None
            ocr_text, ocr_conf, retry_bbox, retry_info = _try_route_owned_scoped_ocr_retry(
                image_path=image_path,
                page_image=page_image,
                image_size=image_size,
                bbox=bbox,
                assignment=text_area_assignment,
                ocr_text=ocr_text,
                ocr_conf=ocr_conf,
                ocr_engine=ocr_engine,
                settings=settings,
                debug_context=debug_context,
                page_id=page_id,
                region_id=f"r{idx:03d}",
                attempt_kind="background_scoped_ocr",
            )
            if retry_info and retry_info.get("status") == "accepted_retry_for_translation":
                bbox = list(retry_bbox)
                polygons = [_bbox_to_polygon(bbox)]
                group["bbox"] = bbox
                group["polygons"] = polygons
                group["route_owned_ocr_retry"] = retry_info
            if not ocr_text:
                if route_authority:
                    _append_empty_ocr_child_evidence(
                        regions,
                        idx=idx,
                        polygons=polygons,
                        bbox=bbox,
                        det_conf=det_conf,
                        ocr_conf=ocr_conf,
                        assignment=text_area_assignment,
                        debug_context=debug_context,
                        page_id=page_id,
                        retry_info=retry_info,
                        semantic_bg=True,
                        region_type="background_text",
                        apply_text_area_assignment_to_region=apply_text_area_assignment_to_region,
                        build_scoped_ocr_candidate=build_scoped_ocr_candidate,
                    )
                continue
            add_count(debug_context, "ocr_results")
            region_type, semantic_bg, semantic_ignore, semantic_review, render_updates = _classify_semantic_region(
                ocr_text,
                bbox,
                image_size,
                det_conf,
                ocr_conf,
                page_image,
                text_filter,
                initial_bg=bg_text,
                text_area_assignment=text_area_assignment,
            )
            skip_text = _should_skip_text(ocr_text, bbox, image_size)
            ignore = semantic_ignore or bool(filter_background and skip_text and semantic_bg)

            region = _region_record(
                idx,
                polygons,
                bbox,
                ocr_text,
                "",
                det_conf,
                bg_text=semantic_bg,
                needs_review=needs_review or semantic_review or skip_text,
                ignore=ignore,
                region_type=region_type,
                ocr_conf=ocr_conf,
                render_updates=render_updates,
            )
            _attach_text_area_assignment(
                region,
                text_area_assignment,
                debug_context,
                page_id,
                ocr_text,
                ocr_conf,
                accepted=not ignore,
                apply_text_area_assignment_to_region=apply_text_area_assignment_to_region,
                build_scoped_ocr_candidate=build_scoped_ocr_candidate,
            )
            _stamp_route_owned_ocr_retry(region, retry_info)
            regions.append(region)
            if _region_translation_blocked_by_ocr_transaction(region):
                continue
            if ignore or region.get("flags", {}).get("ignore"):
                continue
            glossary_texts.append(ocr_text)
            cached = translation_cache.get(ocr_text)
            if cached is not None:
                region["translation"] = cached
            else:
                pending_texts.setdefault(ocr_text, []).append(region["region_id"])
            continue
        crop = _crop_image(image_path, bbox, image_obj=page_image)
        if crop is None:
            if route_authority:
                fail_required_ocr(f"r{idx:03d}", bbox, "ocr crop unavailable")
            continue
        ocr_start = time.time()
        ocr_text, ocr_conf = _recognize_with_fallback(
            ocr_engine,
            crop,
            settings,
            bbox,
            debug_context=debug_context,
            trace_context=_ocr_trace_context_from_assignment(
                page_id=page_id,
                region_id=f"r{idx:03d}",
                bbox=bbox,
                assignment=text_area_assignment,
                attempt_kind="speech_scoped_ocr",
            ),
        )
        add_timing(debug_context, "ocr_time", time.time() - ocr_start)
        retry_info = None
        ocr_text, ocr_conf, retry_bbox, retry_info = _try_route_owned_scoped_ocr_retry(
            image_path=image_path,
            page_image=page_image,
            image_size=image_size,
            bbox=bbox,
            assignment=text_area_assignment,
            ocr_text=ocr_text,
            ocr_conf=ocr_conf,
            ocr_engine=ocr_engine,
            settings=settings,
            debug_context=debug_context,
            page_id=page_id,
            region_id=f"r{idx:03d}",
            attempt_kind="speech_scoped_ocr",
        )
        if retry_info and retry_info.get("status") == "accepted_retry_for_translation":
            bbox = list(retry_bbox)
            polygons = [_bbox_to_polygon(bbox)]
            group["bbox"] = bbox
            group["polygons"] = polygons
            group["route_owned_ocr_retry"] = retry_info
            crop = _crop_image(image_path, bbox, image_obj=page_image) or crop
        if not ocr_text:
            if route_authority:
                _append_empty_ocr_child_evidence(
                    regions,
                    idx=idx,
                    polygons=polygons,
                    bbox=bbox,
                    det_conf=det_conf,
                    ocr_conf=ocr_conf,
                    assignment=text_area_assignment,
                    debug_context=debug_context,
                    page_id=page_id,
                    retry_info=retry_info,
                    semantic_bg=False,
                    region_type="speech_bubble",
                    apply_text_area_assignment_to_region=apply_text_area_assignment_to_region,
                    build_scoped_ocr_candidate=build_scoped_ocr_candidate,
                )
            continue
        add_count(debug_context, "ocr_results")
        region_type, semantic_bg, semantic_ignore, semantic_review, render_updates = _classify_semantic_region(
            ocr_text,
            bbox,
            image_size,
            det_conf,
            ocr_conf,
            page_image,
            text_filter,
            initial_bg=False,
            text_area_assignment=text_area_assignment,
        )
        if (
            region_type == "speech_bubble"
            and not route_authority
            and _should_ignore_speech_fragment(ocr_text, bbox, image_size, ocr_conf)
        ):
            semantic_ignore = True
            semantic_review = True
        if semantic_ignore:
            region = _region_record(
                idx,
                polygons,
                bbox,
                ocr_text,
                "",
                det_conf,
                bg_text=semantic_bg,
                needs_review=True,
                ignore=True,
                region_type=region_type,
                ocr_conf=ocr_conf,
                render_updates=render_updates,
            )
            _attach_text_area_assignment(
                region,
                text_area_assignment,
                debug_context,
                page_id,
                ocr_text,
                ocr_conf,
                accepted=False,
                apply_text_area_assignment_to_region=apply_text_area_assignment_to_region,
                build_scoped_ocr_candidate=build_scoped_ocr_candidate,
            )
            _stamp_route_owned_ocr_retry(region, retry_info)
            regions.append(region)
            continue
        glossary_texts.append(ocr_text)
        # REMOVED: _should_skip_text filter for speech bubbles
        # Speech bubbles detected by the detector should NEVER be filtered
        # They are legitimate dialogue that must always be translated
        # REMOVED: TextFilter check for speech bubbles
        # Speech bubbles detected by the detector should NEVER be filtered
        # They are legitimate dialogue - always translate them

        region = _region_record(
            idx,
            polygons,
            bbox,
            ocr_text,
            "",
            det_conf,
            bg_text=semantic_bg,
            needs_review=needs_review or semantic_review,
            ignore=False,
            region_type=region_type,
            ocr_conf=ocr_conf,
            render_updates=render_updates,
        )
        _attach_text_area_assignment(
            region,
            text_area_assignment,
            debug_context,
            page_id,
            ocr_text,
            ocr_conf,
            accepted=True,
            apply_text_area_assignment_to_region=apply_text_area_assignment_to_region,
            build_scoped_ocr_candidate=build_scoped_ocr_candidate,
        )
        _stamp_route_owned_ocr_retry(region, retry_info)
        regions.append(region)
        if _region_translation_blocked_by_ocr_transaction(region):
            if glossary_texts and glossary_texts[-1] == ocr_text:
                glossary_texts.pop()
            continue
        if region.get("flags", {}).get("ignore"):
            if glossary_texts and glossary_texts[-1] == ocr_text:
                glossary_texts.pop()
            continue
        cached = translation_cache.get(ocr_text)
        if cached is not None:
            region["translation"] = cached
        elif region.get("ignore") and text_filter.should_ignore(ocr_text, "background_text"):
            # Skip background text if the filter agrees it's skippable (SFX)
            # This allows Plot Descriptions (which don't look like SFX) to pass through.
            pass
        else:
            pending_texts.setdefault(ocr_text, []).append(region["region_id"])

    page_class = _classify_page(regions, page_image)
    if debug_context is not None:
        debug_context["controller_semantic_mutation_status"] = {
            "status": "disabled_text_area_plan_is_semantic_authority",
            "page_class": page_class,
            "disabled_paths": [
                "page_class_region_suppression",
                "top_row_caption_ocr_rescue",
                "low_confidence_sfx_reroute",
                "missed_speech_region_creation",
                "adjacent_vertical_region_merge",
                "bubble_local_region_merge",
            ],
        }

    if text_area_plan is not None:
        try:
            logical_assignment_plan = enrich_text_area_plan_with_region_records(text_area_plan, regions)
            if debug_context is not None:
                debug_context["text_area_plan_logical_assignment_enriched"] = True
                debug_context["text_area_plan_logical_assignment_enriched_audit_only"] = logical_assignment_plan
        except Exception as exc:
            if debug_context is not None:
                debug_context["text_area_plan_logical_assignment_error"] = f"{type(exc).__name__}: {exc}"

    logical_text_area_plan = text_area_plan
    if text_area_plan is not None:
        try:
            enriched_plan = enrich_text_area_plan_with_region_records(text_area_plan, regions)
            if debug_context is not None:
                debug_context["text_area_plan"] = enriched_plan
                debug_context.setdefault("text_area_plan_pre_ocr", text_area_plan.to_dict() if hasattr(text_area_plan, "to_dict") else text_area_plan)
                set_count(debug_context, "text_area_plan_enriched_containers", len(enriched_plan.get("containers") or []))
                set_count(debug_context, "text_area_plan_enriched_from_region", int((enriched_plan.get("summary") or {}).get("enriched_from_region_count") or 0))
        except Exception as exc:
            if debug_context is not None:
                debug_context["text_area_plan_enrichment_error"] = f"{type(exc).__name__}: {exc}"

    hierarchy_start = time.time()
    initial_hierarchy_start = time.perf_counter()
    ocr_diagnostics = [
        "ocr_review_recommended"
        for region in regions
        if bool((region.get("flags") or {}).get("needs_review"))
    ]
    publish_stage_outcome(
        stage=PipelineStage.OCR,
        state=(
            PipelineStageOutcomeState.VALID_WITH_DIAGNOSTICS
            if ocr_diagnostics
            else PipelineStageOutcomeState.VALID
        ),
        parent_ids=(str(region.get("region_id") or "") for region in regions),
        artifact_kind="ocr_region_records",
        artifact_summary={
            "width": int(image_size[0] or 0),
            "height": int(image_size[1] or 0),
            "page_class": str(page_class or "normal"),
            "regions": regions,
        },
        diagnostics=ocr_diagnostics,
    )
    notify_stage(PipelineStage.HIERARCHY, "Building the effective text hierarchy")
    initial_text_block_hierarchy = build_text_block_hierarchy(
        page_id=page_id,
        regions=regions,
        text_area_plan=logical_text_area_plan,
        mutate_regions=False,
    )
    if not initial_text_block_hierarchy.generated:
        raise PipelineStageTechnicalError(
            stage=PipelineStage.HIERARCHY,
            code="root_parent_child_plan_required",
            message="TextAreaPlan did not provide a canonical root-parent-child graph.",
            detail=str(initial_text_block_hierarchy.error or "hierarchy_generation_failed"),
            page_id=str(page_id or ""),
            operation="build_initial_text_block_hierarchy",
            artifact_summary=initial_text_block_hierarchy.to_audit_dict(),
        )
    add_timing(
        debug_context,
        "hierarchy_initial_build_time",
        time.perf_counter() - initial_hierarchy_start,
    )
    parent_ocr_start = time.perf_counter()
    parent_ocr_source_status = _append_parent_boundary_ocr_source_regions(
        regions=regions,
        text_area_plan=logical_text_area_plan,
        page_id=page_id,
        image_path=image_path,
        page_image=page_image,
        image_size=image_size,
        ocr_engine=ocr_engine,
        settings=settings,
        debug_context=debug_context,
        assign_bbox_to_text_area_plan=assign_bbox_to_text_area_plan,
        apply_text_area_assignment_to_region=apply_text_area_assignment_to_region,
        build_scoped_ocr_candidate=build_scoped_ocr_candidate,
        existing_parent_units=initial_text_block_hierarchy.parent_units,
    )
    add_timing(debug_context, "parent_ocr_time", time.perf_counter() - parent_ocr_start)
    final_hierarchy_start = time.perf_counter()
    text_block_hierarchy = build_text_block_hierarchy(
        page_id=page_id,
        regions=regions,
        text_area_plan=logical_text_area_plan,
        mutate_regions=True,
    )
    if not text_block_hierarchy.generated:
        raise PipelineStageTechnicalError(
            stage=PipelineStage.HIERARCHY,
            code="root_parent_child_plan_required",
            message="TextAreaPlan did not provide a canonical root-parent-child graph.",
            detail=str(text_block_hierarchy.error or "hierarchy_generation_failed"),
            page_id=str(page_id or ""),
            operation="build_final_text_block_hierarchy",
            artifact_summary=text_block_hierarchy.to_audit_dict(),
        )
    add_timing(
        debug_context,
        "hierarchy_final_build_time",
        time.perf_counter() - final_hierarchy_start,
    )
    add_timing(debug_context, "text_block_hierarchy_time", time.time() - hierarchy_start)
    if debug_context is not None:
        hierarchy_payload = text_block_hierarchy.to_audit_dict()
        debug_context["text_block_hierarchy"] = hierarchy_payload
        debug_context["parent_boundary_ocr_source_contract"] = parent_ocr_source_status
        set_count(debug_context, "text_block_hierarchy_roots", len(text_block_hierarchy.roots))
        set_count(debug_context, "text_block_hierarchy_parent_units", len(text_block_hierarchy.parent_units))
        set_count(debug_context, "text_block_hierarchy_child_segments", len(text_block_hierarchy.child_segments))
        set_count(debug_context, "text_block_hierarchy_unresolved_children", len(text_block_hierarchy.unresolved_children))
    parent_execution_bundle_result = build_parent_execution_bundles(
        page_id=page_id,
        hierarchy_result=text_block_hierarchy,
        regions=regions,
    )
    parent_execution_bundles = parent_execution_bundle_result.executable_bundles()
    if debug_context is not None:
        debug_context["parent_execution_bundles"] = parent_execution_bundle_result.to_audit_dict()
        set_count(debug_context, "parent_execution_bundle_count", len(parent_execution_bundles))
        set_count(debug_context, "parent_execution_blocked_bundle_count", len(parent_execution_bundle_result.blocked_bundles))
        set_count(debug_context, "parent_execution_bundle_error_count", len(parent_execution_bundle_result.errors))
    if parent_execution_bundle_result.errors or parent_execution_bundle_result.blocked_bundles:
        blocked_ids = [
            str(bundle.parent_id or bundle.bundle_id or "")
            for bundle in parent_execution_bundle_result.blocked_bundles
        ]
        raise PipelineStageTechnicalError(
            stage=PipelineStage.HIERARCHY,
            code="parent_execution_bundle_contract_failed",
            message="Hierarchy could not finalize every generated parent as executable.",
            detail=",".join(
                [
                    *list(parent_execution_bundle_result.errors),
                    *(f"blocked_parent:{value}" for value in blocked_ids if value),
                ]
            ),
            page_id=str(page_id or ""),
            parent_id=blocked_ids[0] if blocked_ids else "",
            operation="build_parent_execution_bundles",
            artifact_summary=parent_execution_bundle_result.to_audit_dict(),
        )
    if not parent_execution_bundles:
        _enforce_no_bundle_parent_contract(
            page_id=page_id,
            regions=regions,
            debug_context=debug_context,
        )
        pending_texts = {}
        glossary_texts = []
    parent_translation_plan: dict[str, list[str]] = {}
    execution_regions = parent_execution_region_records(parent_execution_bundles)
    publish_stage_outcome(
        stage=PipelineStage.HIERARCHY,
        state=PipelineStageOutcomeState.VALID,
        parent_ids=(bundle.parent_id for bundle in parent_execution_bundles),
        artifact_kind="parent_execution_bundles",
        artifact_summary={
            "page_class": str(page_class or "normal"),
            "regions": execution_regions,
            "parent_execution_bundles": [
                bundle.to_audit_dict() for bundle in parent_execution_bundles
            ],
        },
    )
    if parent_execution_bundles:
        parent_translation_plan, glossary_texts = _rebuild_translation_inputs_from_parent_execution_bundles(
            parent_execution_bundles
        )
        pending_texts = {}

    active_style_guide = style_guide
    use_context_lines = bool(
        settings
        and settings.translator_backend == "GGUF"
        and target_lang == "Simplified Chinese"
        and bool(getattr(settings, "gguf_cross_page_context", False))
    )
    context_lines = _recent_context_lines(context_window, max_lines=4) if use_context_lines else []

    # Skip runtime discovery if Pre-Scan is enabled (glossary is already built)
    should_run_discovery = (
        auto_glossary_state is not None
        and glossary_texts
        and not (settings and settings.prescan_enabled)
    )

    glossary_start = time.time()
    if should_run_discovery:
        if _GLOSSARY_DEBUG:
            import tempfile
            log_path = os.path.join(tempfile.gettempdir(), "auto_glossary_debug.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"  -> Calling _apply_auto_glossary with {len(glossary_texts)} texts\n")
        active_style_guide = _apply_auto_glossary(
            style_guide,
            auto_glossary_state,
            glossary_texts,
            ollama,
            model,
            source_lang,
            target_lang,
            style_guide_path=style_guide_path,
            allow_ollama=allow_ollama_discovery,
            discovery_model=discovery_model,
            settings=settings,
            mecab_only=not allow_ollama_discovery,
        )
        if auto_glossary_state is not None:
            new_client = auto_glossary_state.get("translation_client")
            if new_client is not None:
                ollama = new_client
    elif auto_glossary_state is not None:
        if _GLOSSARY_DEBUG:
            import tempfile
            log_path = os.path.join(tempfile.gettempdir(), "auto_glossary_debug.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("  -> SKIPPED: glossary_texts is EMPTY\n")
    add_timing(debug_context, "glossary_time", time.time() - glossary_start)
    for region in execution_regions:
        ocr_text = str(region.get("ocr_text", "") or "")
        terms = _matched_glossary_terms(ocr_text, active_style_guide)
        if terms:
            mark_render_region(
                debug_context,
                str(region.get("region_id", "") or ""),
                glossary_terms_available=_debug_glossary_terms(terms),
            )
    mark_translation_plan(
        debug_context,
        execution_regions,
        parent_translation_plan if parent_execution_bundles else pending_texts,
    )
    translation_start = time.time()
    notify_stage(PipelineStage.TRANSLATION, "Translating parent assignments")
    translation_assignments = _translation_assignments_from_parent_execution_bundles(
        parent_execution_bundles
    )
    translation_touched = bool(translation_assignments)
    if translation_assignments:
        perf_pending_texts = {
            assignment_id: list(assignment.region_ids)
            for assignment_id, assignment in translation_assignments.items()
        }
        perf_source_text_by_key = {
            assignment_id: assignment.source_text
            for assignment_id, assignment in translation_assignments.items()
        }
        translation_perf_records = _translation_perf_records_for_page(
            debug_context,
            perf_pending_texts,
            execution_regions,
            source_lang=source_lang,
            target_lang=target_lang,
            settings=settings,
            source_text_by_key=perf_source_text_by_key,
        )
        assignment_source_texts = [assignment.source_text for assignment in translation_assignments.values()]
        prompt_style_guide = _build_page_style_guide(
            active_style_guide,
            assignment_source_texts,
        )
        prompt_has_glossary = bool((prompt_style_guide.get("glossary") or []) or (prompt_style_guide.get("characters") or []))
        batch_items = []
        short_batch_items_by_context: dict[bool, list[dict[str, str]]] = {
            False: [],
            True: [],
        }
        id_to_assignment_id: dict[str, str] = {}
        id_to_context_lines: dict[str, list[str]] = {}
        assignment_to_translation: dict[str, str] = {}
        single_assignment_ids: list[str] = []
        for idx, (assignment_id, assignment) in enumerate(translation_assignments.items()):
            text = assignment.source_text
            region_ids = assignment.region_ids
            available_terms = _matched_glossary_terms(text, active_style_guide)
            prompt_terms = _matched_glossary_terms(text, prompt_style_guide)
            _translation_perf_set_glossary_context(
                translation_perf_records.get(assignment_id),
                available_terms,
            )
            prompt_sources = {str(item.get("source", "")).strip() for item in prompt_terms}
            ignored_terms = [
                item for item in available_terms
                if str(item.get("source", "")).strip() not in prompt_sources
            ]
            for rid in region_ids:
                mark_render_region(
                    debug_context,
                    rid,
                    glossary_terms_available=_debug_glossary_terms(available_terms),
                    glossary_terms_ignored=_debug_glossary_terms(ignored_terms),
                    prompt_glossary_section_included=prompt_has_glossary,
                )
            if _should_single_translate_text(text, region_ids, execution_regions):
                short_batch_context_lane = _deepseek_short_batch_context_lane(
                    text,
                    region_ids,
                    execution_regions,
                    target_lang=target_lang,
                    settings=settings,
                )
                if short_batch_context_lane is None:
                    single_assignment_ids.append(assignment_id)
                    continue
                item_id = f"t{idx:03d}"
                short_batch_items_by_context[short_batch_context_lane].append(
                    {"id": item_id, "text": text}
                )
                id_to_assignment_id[item_id] = assignment_id
                id_to_context_lines[item_id] = (
                    list(context_lines or []) if short_batch_context_lane else []
                )
                continue
            item_id = f"t{idx:03d}"
            batch_items.append({"id": item_id, "text": text})
            id_to_assignment_id[item_id] = assignment_id
            id_to_context_lines[item_id] = list(context_lines or [])
        translations = {}
        translation_perf_records_by_item_id = {
            item_id: translation_perf_records.get(assignment_id)
            for item_id, assignment_id in id_to_assignment_id.items()
            if translation_perf_records.get(assignment_id) is not None
        }
        if batch_items:
            translations.update(
                _batch_translate(
                    ollama,
                    model,
                    source_lang,
                    target_lang,
                    prompt_style_guide,
                    batch_items,
                    context_lines=context_lines,
                    settings=settings,
                    debug_records_by_text=translation_perf_records_by_item_id,
                )
            )
        for use_context, short_batch_items in short_batch_items_by_context.items():
            if len(short_batch_items) < 2:
                for item in short_batch_items:
                    assignment_id = id_to_assignment_id.get(str(item.get("id") or ""))
                    if assignment_id:
                        single_assignment_ids.append(assignment_id)
                continue
            translations.update(
                _batch_translate(
                    ollama,
                    model,
                    source_lang,
                    target_lang,
                    prompt_style_guide,
                    short_batch_items,
                    context_lines=context_lines if use_context else [],
                    settings=settings,
                    debug_records_by_text=translation_perf_records_by_item_id,
                )
            )
        if translations:
            for item_id, translation in translations.items():
                assignment_id = id_to_assignment_id.get(item_id)
                assignment = translation_assignments.get(assignment_id or "")
                if assignment is not None:
                    text = assignment.source_text
                    # Apply glossary enforcement to ensure consistent name translations
                    enforced = _enforce_glossary(translation, text, active_style_guide)
                    if _has_glossary_count_mismatch(text, enforced, active_style_guide):
                        protected = _translate_with_glossary_placeholders(
                            ollama,
                            model,
                            source_lang,
                            target_lang,
                            text,
                            _matched_glossary_terms(text, active_style_guide),
                            debug_record=translation_perf_records.get(assignment_id or ""),
                            debug_phase="batch_glossary_placeholder",
                        )
                        record = translation_perf_records.get(assignment_id or "")
                        if record:
                            _translation_perf_add_path(record, "glossary_placeholder_repair")
                            record.setdefault("json_repair_fallback_status", []).append(
                                "glossary_placeholder_repair_after_batch"
                            )
                        if protected:
                            enforced = _enforce_glossary(protected, text, active_style_guide)
                    item_context_lines = id_to_context_lines.get(
                        item_id,
                        list(context_lines or []),
                    )
                    if not _translation_reuses_recent_context(
                        enforced,
                        text,
                        item_context_lines,
                    ):
                        assignment_to_translation[assignment.assignment_id] = enforced
        missing_assignment_ids = list(single_assignment_ids)
        for assignment_id in translation_assignments.keys():
            if assignment_id not in assignment_to_translation and assignment_id not in missing_assignment_ids:
                missing_assignment_ids.append(assignment_id)
        for assignment_id in missing_assignment_ids:
            assignment = translation_assignments.get(assignment_id)
            if assignment is None:
                continue
            text = assignment.source_text
            region_ids = assignment.region_ids
            text_context_lines = context_lines if _should_use_context_for_text(text, region_ids, execution_regions) else []
            unit_record = translation_perf_records.get(assignment_id)
            raw_trans = _translate_single(
                ollama,
                model,
                source_lang,
                target_lang,
                prompt_style_guide,
                text,
                context_lines=text_context_lines,
                settings=settings,
                debug_record=unit_record,
            )
            if _translation_reuses_recent_context(raw_trans, text, text_context_lines):
                _translation_perf_add_path(unit_record, "context_reuse_retry_no_context")
                if unit_record:
                    unit_record.setdefault("failure_retry_reason", []).append("translation_reused_recent_context")
                raw_trans = _translate_single(
                    ollama,
                    model,
                    source_lang,
                    target_lang,
                    prompt_style_guide,
                    text,
                    context_lines=[],
                    settings=settings,
                    debug_record=unit_record,
                )
            # Apply glossary enforcement
            enforced = _enforce_glossary(raw_trans, text, active_style_guide)
            if _has_glossary_count_mismatch(text, enforced, active_style_guide):
                _translation_perf_add_path(unit_record, "glossary_placeholder_repair")
                if unit_record:
                    unit_record.setdefault("json_repair_fallback_status", []).append(
                        "glossary_placeholder_repair_after_single"
                    )
                protected = _translate_with_glossary_placeholders(
                    ollama,
                    model,
                    source_lang,
                    target_lang,
                    text,
                    _matched_glossary_terms(text, active_style_guide),
                    debug_record=unit_record,
                    debug_phase="single_glossary_placeholder",
                )
                if protected:
                    enforced = _enforce_glossary(protected, text, active_style_guide)
            assignment_to_translation[assignment_id] = enforced
        for assignment_id, assignment in translation_assignments.items():
            text = assignment.source_text
            region_ids = assignment.region_ids
            is_bubble = False
            for region in execution_regions:
                if region["region_id"] in region_ids and region.get("type") == "speech_bubble":
                    is_bubble = True
                    break

            translation, lang_ok = _ensure_target_language(
                ollama,
                _resolve_model(model),
                source_lang,
                target_lang,
                text,
                assignment_to_translation.get(assignment_id, ""),
                is_bubble=is_bubble,
                debug_record=translation_perf_records.get(assignment_id),
            )
            if translation:
                translation = _enforce_glossary(translation, text, active_style_guide)
                pre_repair_translation = translation
                unit_record = translation_perf_records.get(assignment_id)
                if _matched_glossary_terms(text, active_style_guide):
                    _translation_perf_add_path(unit_record, "glossary_repair_check")
                translation = _repair_translation_with_glossary(
                    ollama,
                    model,
                    source_lang,
                    target_lang,
                    text,
                    translation,
                    active_style_guide,
                    debug_record=unit_record,
                )
                if _translation_is_unsafe_for_output(translation, text):
                    translation = pre_repair_translation
            if target_lang == "Simplified Chinese" and _is_short_reaction_source(text):
                forced_short = _translate_short_reaction_fallback(text, target_lang)
                if forced_short:
                    translation = forced_short
                    lang_ok = True
            if translation:
                translation = _apply_source_level_semantic_corrections(text, translation)
                translation = _normalize_translation_format_for_record(
                    target_lang,
                    translation,
                    translation_perf_records.get(assignment_id),
                    stage="final_translation_assignment",
                )
                unit_record = translation_perf_records.get(assignment_id)
                _translation_perf_set_final(unit_record, translation=translation)
                translation, terminal_symbol_evidence = _preserve_repeated_terminal_emphasis_symbols(
                    text,
                    translation,
                )
                if terminal_symbol_evidence.get("changed"):
                    _translation_perf_record_terminal_symbol_conservation(
                        unit_record,
                        terminal_symbol_evidence,
                    )
                    _translation_perf_set_final(
                        unit_record,
                        translation=translation,
                        status="terminal_symbol_multiplicity_repaired",
                    )
                translation_cache[assignment.cache_key] = translation
            matched_terms = _matched_glossary_terms(text, active_style_guide)
            applied_terms = []
            ignored_terms = []
            warnings = []
            for item in matched_terms:
                source = str(item.get("source", "")).strip()
                target = str(item.get("target", "")).strip()
                if target and target in str(translation or ""):
                    applied_terms.append(item)
                else:
                    ignored_terms.append(item)
                    if source and target:
                        warnings.append(f"missing_glossary_target:{source}->{target}")
            unit_record = translation_perf_records.get(assignment_id)
            _translation_perf_set_glossary_status(
                unit_record,
                applied_terms=applied_terms,
                ignored_terms=ignored_terms,
                warnings=warnings,
            )
            for region in execution_regions:
                if region["region_id"] in region_ids:
                    mark_render_region(
                        debug_context,
                        str(region.get("region_id", "") or ""),
                        glossary_terms_applied=_debug_glossary_terms(applied_terms),
                        glossary_terms_ignored=_debug_glossary_terms(ignored_terms),
                        terminology_consistency_warnings=warnings,
                    )
                    recover_candidate = _looks_like_recoverable_speech_region(region, page_class)
                    if _should_preserve_decorative_fragment_translation(text, region, active_style_guide) and not recover_candidate:
                        region["translation"] = ""
                        region["flags"]["ignore"] = True
                        region["flags"]["bg_text"] = True
                        region["flags"]["needs_review"] = False
                        region.setdefault("render", {})["cleanup_mode"] = "preserve"
                        continue
                    if recover_candidate:
                        region["type"] = "speech_bubble"
                        region.setdefault("flags", {})["bg_text"] = False
                        region["flags"]["ignore"] = False
                        region["flags"].pop("hard_fail", None)
                        render = region.setdefault("render", {})
                        render["cleanup_mode"] = "bubble"
                    final_translation = translation
                    region["translation"] = final_translation
                    _translation_perf_mark_region_consumed(
                        unit_record,
                        region,
                        final_translation,
                        consumed_path="region.translation",
                    )
                    if not lang_ok or _translation_is_unsafe_for_output(final_translation, text):
                        region["flags"]["needs_review"] = True

    if translation_touched:
        add_timing(debug_context, "translation_time", time.time() - translation_start)
    _summarize_translation_requests(debug_context)

    sync_bundles_from_region_records(parent_execution_bundles, execution_regions)
    missing_translation_parent_ids = [
        str(bundle.parent_id or bundle.bundle_id or "")
        for bundle in parent_execution_bundles
        if bool(bundle.translation_required)
        and not str(bundle.translated_text or "").strip()
    ]
    if missing_translation_parent_ids:
        raise PipelineStageTechnicalError(
            stage=PipelineStage.TRANSLATION,
            code="parent_translation_output_missing",
            message="Translation did not produce output for every required parent.",
            detail=",".join(missing_translation_parent_ids),
            page_id=str(page_id or ""),
            parent_id=missing_translation_parent_ids[0],
            operation="translate_parent_assignments",
            artifact_summary={
                "regions": execution_regions,
                "parent_execution_bundles": [
                    bundle.to_audit_dict() for bundle in parent_execution_bundles
                ],
            },
        )
    publish_stage_outcome(
        stage=PipelineStage.TRANSLATION,
        state=PipelineStageOutcomeState.VALID,
        parent_ids=(bundle.parent_id for bundle in parent_execution_bundles),
        artifact_kind="translated_parent_execution_bundles",
        artifact_summary={
            "page_class": str(page_class or "normal"),
            "regions": execution_regions,
            "parent_execution_bundles": [
                bundle.to_audit_dict() for bundle in parent_execution_bundles
            ],
        },
    )
    if debug_context is not None and parent_execution_bundles:
        debug_context["parent_execution_bundles"] = parent_execution_bundle_result.to_audit_dict()

    page_context = []
    for region in execution_regions:
        if not _region_can_feed_context(region, page_class):
            continue
        trans = region.get("translation", "").strip()
        if trans:
             page_context.append(trans)

    # Keep last 10 lines of context to avoid overflow
    if page_context:
        context_window.extend(page_context)
        while len(context_window) > 4:
            context_window.pop(0)

    return PageProcessingResult(
        regions,
        execution_regions,
        parent_execution_bundles,
        page_class,
        text_area_plan,
        ctd_segmentation_result,
    )


def _enforce_no_bundle_parent_contract(
    *,
    page_id: str,
    regions: list[dict],
    debug_context: dict | None = None,
) -> None:
    """Reject authorized workflow regions that lost their canonical parent unit."""

    authorized_region_ids: list[str] = []
    for index, region in enumerate(regions or []):
        if not isinstance(region, dict):
            continue
        if not _region_has_translatable_text_area_route(region):
            continue
        region_id = str(region.get("region_id") or f"region_{index}")
        if region_id not in authorized_region_ids:
            authorized_region_ids.append(region_id)
    if debug_context is not None:
        debug_context["parent_execution_no_bundle_contract"] = {
            "page_id": str(page_id or ""),
            "authorized_region_ids": list(authorized_region_ids),
            "status": (
                "contract_error_authorized_regions_without_parent_bundle"
                if authorized_region_ids
                else "no_render_layers"
            ),
            "legacy_region_fallback_used": False,
        }
    if authorized_region_ids:
        joined = ",".join(authorized_region_ids)
        raise PipelineStageTechnicalError(
            stage=PipelineStage.HIERARCHY,
            code="parent_execution_bundle_missing",
            message="Authorized workflow regions have no executable parent bundle.",
            detail=f"{page_id}:{joined}",
            page_id=str(page_id or ""),
            parent_id=authorized_region_ids[0],
            operation="enforce_no_bundle_parent_contract",
            artifact_summary={"authorized_region_ids": authorized_region_ids},
        )


def _write_no_layer_render_output(
    render_input_path: str,
    output_path: str,
    *,
    debug_context: dict | None = None,
) -> None:
    """Preserve CleanedPageBase when the canonical graph has no render layers."""

    source = os.path.abspath(str(render_input_path or ""))
    target = os.path.abspath(str(output_path or ""))
    if not source or not os.path.isfile(source):
        raise FileNotFoundError(
            f"no_layer_render_input_missing:{render_input_path}"
        )
    if source != target:
        shutil.copyfile(source, target)
    if debug_context is not None:
        debug_context["stage5_renderer_compositor_active"] = False
        debug_context["render_translations_called"] = False
        debug_context["final_translated_text_drawn"] = False
        debug_context["legacy_region_rendering_used"] = False
        debug_context["renderer_no_layer_result"] = {
            "status": "cleaned_page_base_copied_without_render_layers",
            "render_input_path": source,
            "output_path": target,
        }


def _logical_text_region_blocks_independent_render(region: dict) -> bool:
    """Compatibility probe; finalized parent regions can never be vetoed here."""

    _ = region
    return False


def _sync_parent_execution_downstream_contracts(
    bundles: list[ParentExecutionBundle],
    execution_regions: list[dict[str, Any]],
    *,
    source_glyph_masks: Any = None,
    cleanup_jobs: Any = None,
    cleanup_masks: Any = None,
    render_eligibility: Any = None,
) -> None:
    if not bundles:
        return
    bundle_by_id = {bundle.bundle_id: bundle for bundle in bundles}
    region_by_id = {
        str(region.get("region_id") or ""): region
        for region in execution_regions
        if isinstance(region, dict) and str(region.get("region_id") or "")
    }
    cleanup_job_parent_by_id: dict[str, str] = {}

    def _parent_bundle_id_from_record(record: Any, *, fallback_job_id: str = "") -> str:
        parent_id = str(
            getattr(record, "parent_execution_bundle_id", "")
            or getattr(record, "parent_logical_text_unit_id", "")
            or ""
        )
        if not parent_id and isinstance(record, dict):
            parent_id = str(
                record.get("parent_execution_bundle_id")
                or record.get("parent_logical_text_unit_id")
                or record.get("parent_id")
                or ""
            )
        if not parent_id:
            target_ids = getattr(record, "target_region_ids", None)
            if target_ids is None and isinstance(record, dict):
                target_ids = record.get("target_region_ids")
            parent_id = str((target_ids or [""])[0] or "")
        if not parent_id and fallback_job_id:
            parent_id = cleanup_job_parent_by_id.get(fallback_job_id, "")
        return parent_id

    for region_id, fields in _safe_region_audit_fields(source_glyph_masks).items():
        parent_id = str(
            fields.get("parent_execution_bundle_id")
            or fields.get("parent_logical_text_unit_id")
            or region_id
        )
        bundle = bundle_by_id.get(parent_id)
        if not bundle:
            continue
        mask_ids = _unique_strings(
            bundle.source_glyph_mask_ids
            + [
                fields.get("source_glyph_mask_id"),
                fields.get("mask_ref"),
            ]
        )
        bundle.source_glyph_mask_ids = mask_ids
        record = region_by_id.get(parent_id)
        if record is not None:
            record["source_glyph_mask_ids"] = mask_ids

    for job in getattr(cleanup_jobs, "jobs", []) or []:
        parent_id = _parent_bundle_id_from_record(job)
        job_id = str(getattr(job, "cleanup_job_id", "") or "")
        if job_id and parent_id:
            cleanup_job_parent_by_id[job_id] = parent_id
        bundle = bundle_by_id.get(parent_id)
        if not bundle:
            continue
        bundle.cleanup_job_ids = _unique_strings(bundle.cleanup_job_ids + [job_id])
        record = region_by_id.get(parent_id)
        if record is not None:
            record["cleanup_job_ids"] = list(bundle.cleanup_job_ids)

    for mask in getattr(cleanup_masks, "masks", []) or []:
        cleanup_job_id = str(getattr(mask, "cleanup_job_id", "") or "")
        parent_id = _parent_bundle_id_from_record(mask, fallback_job_id=cleanup_job_id)
        bundle = bundle_by_id.get(parent_id)
        if not bundle:
            continue
        mask_id = str(
            getattr(mask, "cleanup_mask_id", "")
            or getattr(mask, "mask_id", "")
            or ""
        )
        bundle.cleanup_mask_ids = _unique_strings(bundle.cleanup_mask_ids + [mask_id])
        record = region_by_id.get(parent_id)
        if record is not None:
            record["cleanup_mask_ids"] = list(bundle.cleanup_mask_ids)

    for decision in getattr(render_eligibility, "decisions", []) or []:
        parent_id = _parent_bundle_id_from_record(decision)
        if not parent_id:
            parent_id = str(getattr(decision, "region_id", "") or "")
        bundle = bundle_by_id.get(parent_id)
        if not bundle:
            continue
        bundle.render_decision_id = parent_id
        record = region_by_id.get(parent_id)
        if record is not None:
            record["render_decision_id"] = parent_id

    sync_bundles_from_region_records(bundles, execution_regions)


def _validate_cleanup_parent_conservation(
    *,
    page_id: str,
    bundles: Iterable[ParentExecutionBundle],
    cleanup_jobs: Any,
    cleanup_masks: Any,
    cleanup_plans: Any,
    cleanup_runtime: Any,
    cleanup_commit: Any,
) -> tuple[str, ...]:
    expected = {
        str(bundle.bundle_id or bundle.parent_id or "")
        for bundle in bundles or []
        if bool(bundle.cleanup_required)
        and str(bundle.bundle_id or bundle.parent_id or "")
    }
    if not expected:
        return ()

    job_parent_by_id: dict[str, str] = {}
    job_parents: set[str] = set()
    for job in getattr(cleanup_jobs, "jobs", []) or []:
        parent_id = str(
            getattr(job, "parent_execution_bundle_id", "")
            or getattr(job, "parent_logical_text_unit_id", "")
            or (getattr(job, "target_region_ids", []) or [""])[0]
            or ""
        )
        job_id = str(getattr(job, "cleanup_job_id", "") or "")
        if job_id and parent_id:
            job_parent_by_id[job_id] = parent_id
        if parent_id:
            job_parents.add(parent_id)

    mask_parents: set[str] = set()
    for mask in getattr(cleanup_masks, "masks", []) or []:
        parent_id = str(
            getattr(mask, "parent_execution_bundle_id", "")
            or getattr(mask, "parent_logical_text_unit_id", "")
            or job_parent_by_id.get(str(getattr(mask, "cleanup_job_id", "") or ""), "")
        )
        if parent_id:
            mask_parents.add(parent_id)

    plan_parents: set[str] = set()
    for plan in getattr(cleanup_plans, "plans", []) or []:
        parent_id = str(
            getattr(plan, "parent_execution_bundle_id", "")
            or getattr(plan, "parent_logical_text_unit_id", "")
            or job_parent_by_id.get(str(getattr(plan, "cleanup_job_id", "") or ""), "")
        )
        if parent_id:
            plan_parents.add(parent_id)

    runtime_parents: set[str] = set()
    warning_parents: set[str] = set()
    invalid_runtime: list[str] = []
    for status in getattr(cleanup_runtime, "status_records", []) or []:
        if not isinstance(status, Mapping):
            continue
        parent_id = str(
            status.get("parent_execution_bundle_id")
            or status.get("parent_logical_text_unit_id")
            or status.get("region_id")
            or ""
        )
        runtime_status = str(status.get("runtime_status") or "")
        if runtime_status in {"passed", "warning"} and parent_id:
            runtime_parents.add(parent_id)
            if runtime_status == "warning":
                warning_parents.add(parent_id)
        elif parent_id:
            invalid_runtime.append(f"{parent_id}:{runtime_status or 'missing_status'}")

    commit_parents = {
        str(
            record.get("parent_execution_bundle_id")
            or record.get("parent_logical_text_unit_id")
            or record.get("region_id")
            or ""
        )
        for record in getattr(cleanup_commit, "commit_records", []) or []
        if isinstance(record, Mapping)
    }
    commit_parents.discard("")
    blocked = [
        str(record.get("parent_execution_bundle_id") or record.get("region_id") or "")
        for record in getattr(cleanup_commit, "blocked_records", []) or []
        if isinstance(record, Mapping)
    ]
    errors = [str(value) for value in getattr(cleanup_commit, "errors", []) or []]

    failures: list[str] = []
    for label, observed in (
        ("job", job_parents),
        ("mask", mask_parents),
        ("plan", plan_parents),
        ("runtime", runtime_parents),
        ("commit", commit_parents),
    ):
        missing = sorted(expected - observed)
        if missing:
            failures.append(f"missing_{label}:" + ",".join(missing))
    failures.extend(f"invalid_runtime:{value}" for value in invalid_runtime)
    failures.extend(f"blocked_commit:{value}" for value in blocked if value)
    failures.extend(f"commit_error:{value}" for value in errors if value)
    if failures:
        raise PipelineStageTechnicalError(
            stage=PipelineStage.CLEANUP,
            code="cleanup_parent_conservation_failed",
            message="Cleanup did not produce a valid artifact for every required parent.",
            detail=";".join(failures),
            page_id=str(page_id or ""),
            parent_id=sorted(expected)[0],
            operation="validate_cleanup_parent_conservation",
            artifact_summary={
                "expected_parent_ids": sorted(expected),
                "failures": failures,
            },
        )
    return tuple(f"cleanup_quality_warning:{value}" for value in sorted(warning_parents))


def _observe_parent_style_after_cleanup(
    *,
    page_id: str,
    image_path: str,
    parent_execution_bundles: list[ParentExecutionBundle],
    cleanup_masks: Any,
    image_size: tuple[int, int] | None,
    mode: str,
    use_gpu: bool,
    models_dir: str,
    detector: Any = None,
) -> tuple[Any, Any]:
    """Observe typed style evidence after parent cleanup authority exists."""

    from app.pipeline.parent_font_detection import observe_parent_style_evidence
    from app.pipeline.parent_style_evidence import build_authorized_source_style_views

    style_views = build_authorized_source_style_views(
        page_id=page_id,
        parent_execution_bundles=parent_execution_bundles,
        cleanup_masks=cleanup_masks,
        image_size=image_size,
    )
    observed = observe_parent_style_evidence(
        page_id=page_id,
        image_path=image_path,
        parent_execution_bundles=parent_execution_bundles,
        authorized_style_views=style_views.views_by_bundle_id,
        mode=mode,
        use_gpu=use_gpu,
        models_dir=models_dir,
        detector=detector,
    )
    return style_views, observed


def _observe_parent_punctuation_geometry_after_cleanup(
    *,
    page_id: str,
    image_path: str,
    parent_execution_bundles: list[ParentExecutionBundle],
    cleanup_masks: Any,
    image_size: tuple[int, int] | None,
    style_views: Any = None,
    style_evidence: Any = None,
) -> Any:
    """Attach degradable source-pixel punctuation facts to parent bundles.

    The accepted source-style view is reused only as a parent-bound pixel
    aperture.  Observation failure records unavailable evidence and never
    changes render admission or the translated text carried by the bundle.
    """

    from app.pipeline.parent_style_evidence import (
        build_authorized_source_style_views,
    )
    from app.render.source_punctuation_hints import (
        SourcePunctuationGeometryEvidence,
        SourcePunctuationGeometryRunResult,
        observe_source_punctuation_geometry,
    )

    render_bundles = [
        bundle
        for bundle in list(parent_execution_bundles or [])
        if bool(getattr(bundle, "render_required", False))
    ]
    errors: list[str] = []
    try:
        if hasattr(style_views, "views_by_bundle_id"):
            views_by_bundle_id = dict(style_views.views_by_bundle_id)
        elif isinstance(style_views, Mapping):
            views_by_bundle_id = dict(style_views)
        else:
            rebuilt = build_authorized_source_style_views(
                page_id=page_id,
                parent_execution_bundles=parent_execution_bundles,
                cleanup_masks=cleanup_masks,
                image_size=image_size,
            )
            views_by_bundle_id = dict(rebuilt.views_by_bundle_id)
        if hasattr(style_evidence, "evidence"):
            style_evidence_items = list(style_evidence.evidence or ())
        elif isinstance(style_evidence, Mapping):
            style_evidence_items = list(style_evidence.values())
        else:
            style_evidence_items = list(style_evidence or ())
        style_evidence_by_bundle_id = {
            str(getattr(item, "bundle_id", "") or ""): item
            for item in style_evidence_items
            if str(getattr(item, "bundle_id", "") or "")
        }
        observed = observe_source_punctuation_geometry(
            page_id=page_id,
            source_image_path=image_path,
            parent_execution_bundles=parent_execution_bundles,
            authorized_style_views=views_by_bundle_id,
            source_style_evidence_by_bundle_id=style_evidence_by_bundle_id,
        )
        observed_by_bundle_id = observed.evidence_by_bundle_id
        errors.extend(observed.errors)
    except Exception as exc:
        views_by_bundle_id = {}
        observed_by_bundle_id = {}
        errors.append(
            "source_punctuation_geometry_stage_failed:"
            f"{type(exc).__name__}:{exc}"
        )

    evidence: list[SourcePunctuationGeometryEvidence] = []
    for bundle in render_bundles:
        bundle_id = str(getattr(bundle, "bundle_id", "") or "")
        item = observed_by_bundle_id.get(bundle_id)
        if item is None:
            view = views_by_bundle_id.get(bundle_id)
            item = SourcePunctuationGeometryEvidence.unavailable(
                page_id=str(page_id or ""),
                bundle_id=bundle_id,
                parent_id=str(getattr(bundle, "parent_id", "") or ""),
                root_id=str(getattr(bundle, "root_id", "") or ""),
                view_id=str(getattr(view, "view_id", "") or ""),
                reason="source_punctuation_geometry_stage_unavailable",
                reason_codes=(
                    "source_punctuation_geometry_stage_unavailable",
                ),
            )
        bundle.source_punctuation_geometry = item.to_audit_dict()
        bundle.to_region_record()
        evidence.append(item)

    return SourcePunctuationGeometryRunResult(
        page_id=str(page_id or ""),
        evidence=tuple(evidence),
        errors=tuple(_unique_strings(errors)),
    )


def _safe_region_audit_fields(source_glyph_masks: Any) -> dict[str, dict[str, Any]]:
    if source_glyph_masks is None:
        return {}
    try:
        fields = source_glyph_masks.region_audit_fields()
    except Exception:
        return {}
    if not isinstance(fields, dict):
        return {}
    return {
        str(region_id): dict(value)
        for region_id, value in fields.items()
        if str(region_id) and isinstance(value, dict)
    }


def _unique_strings(values: list[Any]) -> list[str]:
    output: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in output:
            output.append(text)
    return output


def _logical_text_region_failed_closed(region: dict) -> bool:
    render = region.get("render", {}) if isinstance(region.get("render"), dict) else {}
    state = str(
        region.get("root_final_state")
        or render.get("root_final_state")
        or region.get("text_block_root_final_state")
        or render.get("text_block_root_final_state")
        or ""
    ).strip()
    reason = str(
        region.get("root_final_state_reason")
        or render.get("root_final_state_reason")
        or region.get("text_block_root_final_state_reason")
        or render.get("text_block_root_final_state_reason")
        or ""
    ).strip()
    skip = str(region.get("skip_reason") or render.get("skip_reason") or "").strip()
    action = str(
        region.get("source_coherence_action")
        or render.get("source_coherence_action")
        or region.get("root_source_coherence_action")
        or render.get("root_source_coherence_action")
        or ""
    ).strip()
    combined = " ".join(value for value in (state, reason, skip, action) if value).lower()
    if state == "unresolved_meaningful_blocker":
        return True
    failed_closed_markers = {
        "root_transaction_failed_closed",
        "ocr_punctuation_only_blocker",
        "root_source_coherence_rejected_parent",
        "block_review_only",
        "unresolved_review_only",
        "no_accepted_parent_unit",
    }
    return any(marker in combined for marker in failed_closed_markers)






def _resolve_model(model: str) -> str:
    if model == "auto-detect":
        models = list_models()
        if models:
            preferred = [
                "aya:35b",
                "huihui_ai/qwen3-abliterated:32b",
                "huihui_ai/qwen3-abliterated:14b",
                "qwen3-coder:30b",
                "dolphin3:8b",
            ]
            for name in preferred:
                if name in models:
                    return name
            return models[0]
        return "aya:35b"
    return model


def _recent_context_lines(context_window: list, max_lines: int = 6) -> list[str]:
    if not context_window:
        return []
    return [str(line).strip() for line in context_window[-max_lines:] if str(line).strip()]


def _translation_reuses_recent_context(translation: str, source_text: str, context_lines: list[str]) -> bool:
    cleaned = str(translation or "").strip()
    if not cleaned or not context_lines:
        return False
    if _is_short_reaction_source(source_text) or _is_ellipsis_like(source_text):
        return False
    body = _non_punct_chars(cleaned)
    if len(body) < 6:
        return False
    normalized = re.sub(r"\s+", "", cleaned)
    normalized_body = "".join(_non_punct_chars(normalized))
    source_body = "".join(_non_punct_chars(source_text))
    for line in context_lines:
        candidate = str(line or "").strip()
        if not candidate:
            continue
        candidate_normalized = re.sub(r"\s+", "", candidate)
        if candidate_normalized == normalized:
            return True
        candidate_body = "".join(_non_punct_chars(candidate_normalized))
        if len(normalized_body) >= 8 and len(candidate_body) >= 8:
            similarity = difflib.SequenceMatcher(None, normalized_body, candidate_body).ratio()
            source_similarity = (
                difflib.SequenceMatcher(None, source_body, candidate_body).ratio()
                if source_body and candidate_body
                else 0.0
            )
            if similarity >= 0.78 and source_similarity <= 0.45:
                return True
    return False


def _classify_page(regions: list[dict], page_image) -> str:
    if _looks_like_decorative_cover_page(regions, page_image):
        return "cover"
    if _looks_like_contents_page(regions, page_image):
        return "contents"
    if _looks_like_chapter_title_page(regions, page_image):
        return "chapter_title"
    return "normal"


def _is_meaningful_speech_source(text: str) -> bool:
    cleaned = _clean_ocr_text(text)
    if not cleaned:
        return False
    if _is_short_reaction_source(cleaned):
        return True
    if _is_ellipsis_like(cleaned):
        return True
    body = _non_punct_chars(cleaned)
    if not body:
        return False
    if len(body) >= 2:
        return True
    return any(_is_cjk_char(ch) for ch in cleaned)


def _is_meaningful_background_caption_source(text: str) -> bool:
    cleaned = _clean_ocr_text(text)
    if not cleaned:
        return False
    body = _non_punct_chars(cleaned)
    if not body:
        return False
    contains_kanji = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in body)
    contains_kana = any(_is_kana(ch) for ch in body)
    has_digits = any(ch.isdigit() for ch in cleaned)
    if contains_kanji and len(body) >= 4:
        return True
    if contains_kanji and has_digits:
        return True
    if contains_kanji and contains_kana and len(body) >= 3:
        return True
    if any(marker in cleaned for marker in ("日目", "回目", "生活", "最終日", "無人島")) and (contains_kanji or has_digits):
        return True
    return False


def _is_probable_short_vertical_dialogue_box(
    text: str,
    bbox: list,
    stats_mean: float | None = None,
    image_size: tuple[int, int] | None = None,
) -> bool:
    cleaned = _clean_ocr_text(text)
    body = _non_punct_chars(cleaned)
    reaction_like = _is_ellipsis_like(cleaned) or _is_short_reaction_source(cleaned)
    if (not body and not reaction_like) or len(body) > 6:
        return False
    if any(ch.isdigit() for ch in cleaned):
        return False
    if body and not any(_is_cjk_char(ch) for ch in body):
        return False
    _, _, w, h = bbox or [0, 0, 0, 0]
    w = max(1, int(w or 1))
    h = max(1, int(h or 1))
    if reaction_like:
        if w < 10 or h < 54 or h < w * 1.55:
            return False
    elif w < 18 or h < 70 or h < w * 1.85:
        return False
    if stats_mean is not None and stats_mean < (160.0 if reaction_like else 150.0):
        return False
    if image_size:
        img_w, img_h = image_size
        if img_w > 0 and img_h > 0:
            cy = bbox[1] + (h / 2.0)
            if cy <= img_h * 0.28:
                return False
    return True


def _is_short_kana_laugh_source(text: str) -> bool:
    cleaned = _clean_ocr_text(text)
    body = _non_punct_chars(cleaned)
    if len(body) < 2 or len(body) > 4:
        return False
    if not all(_is_kana(ch) or ch == "ー" for ch in body):
        return False
    seed = [ch for ch in body if ch != "ー"]
    if not seed or len(set(seed)) != 1:
        return False
    return seed[0] in {"フ", "ふ", "ハ", "は", "ヘ", "へ", "ヒ", "ひ"}


def _has_bright_bubble_context_pil(image_obj, bbox: list) -> bool:
    if image_obj is None or not bbox:
        return False
    try:
        from PIL import ImageStat
    except Exception:
        return False
    try:
        img_w, img_h = image_obj.size
        x, y, w, h = [int(v) for v in bbox[:4]]
        w = max(1, w)
        h = max(1, h)
        pad = max(10, min(24, int(round(max(w, h) * 0.22))))
        x0 = max(0, min(x - pad, img_w - 1))
        y0 = max(0, min(y - pad, img_h - 1))
        x1 = max(x0 + 1, min(x + w + pad, img_w))
        y1 = max(y0 + 1, min(y + h + pad, img_h))
        crop = image_obj.crop((x0, y0, x1, y1)).convert("L")
        stat = ImageStat.Stat(crop)
        if not stat.mean:
            return False
        hist = crop.histogram()
        total = max(1, sum(hist))
        bright_ratio = sum(hist[230:]) / total
        dark_ratio = sum(hist[:80]) / total
        return float(stat.mean[0]) >= 225.0 and bright_ratio >= 0.82 and dark_ratio <= 0.10
    except Exception:
        return False


def _is_bubble_contained_short_laugh_speech_candidate(
    text: str,
    bbox: list,
    image_size: tuple[int, int],
    det_conf: float,
    ocr_conf: float,
    image_obj,
    stats_mean: float | None,
) -> bool:
    if not _is_short_kana_laugh_source(text):
        return False
    if det_conf < 0.80 or ocr_conf < 0.95:
        return False
    if not bbox or len(bbox) < 4:
        return False
    _, _, w, h = bbox
    w = max(1, int(w or 1))
    h = max(1, int(h or 1))
    page_area = max(1, int(image_size[0]) * int(image_size[1])) if image_size else 1
    area_ratio = (w * h) / page_area
    if w > 64 or h < 54 or h < w * 1.5 or area_ratio > 0.004:
        return False
    if stats_mean is not None and stats_mean < 200.0:
        return False
    return _has_bright_bubble_context_pil(image_obj, bbox)


_LOW_CONF_DARK_SHORT_ART_SFX_REASON = "low_conf_dark_short_art_sfx_candidate"
_MEDIUM_LARGE_KATAKANA_SFX_REASON = "medium_large_katakana_sfx_candidate"
_NONBUBBLE_SHORT_KANA_ART_TEXT_REASON = "nonbubble_short_kana_art_text_candidate"
_NONBUBBLE_SHORT_REACTION_ART_TEXT_REASON = "nonbubble_short_reaction_art_text_candidate"
_NONBUBBLE_SHORT_REACTION_ART_SFX_REASON = "nonbubble_short_reaction_art_sfx_candidate"
_SHORT_REACTION_WITHOUT_VISUAL_SPEECH_OWNERSHIP_REASON = "short_reaction_without_visual_speech_ownership"
_NONBUBBLE_BREATH_SFX_ART_TEXT_REASON = "nonbubble_breath_sfx_art_text_candidate"
_LARGE_LOW_CONFIDENCE_NONBUBBLE_SFX_REASON = "large_low_confidence_nonbubble_sfx_candidate"
_BUBBLE_CONTAINED_SHORT_LAUGH_SPEECH_REASON = "bubble_contained_short_laugh_speech"
_TOP_ROW_BACKGROUND_CAPTION_REASON = "top_row_background_caption_candidate"
_TOP_ROW_CAPTION_FRAGMENT_REASON = "top_row_caption_fragment_candidate"


def _nonbubble_short_kana_art_text_reason(
    text: str,
    bbox: list,
    image_size: tuple[int, int],
    det_conf: float,
    ocr_conf: float,
    image_obj,
    stats_mean: float | None = None,
) -> str:
    cleaned = str(text or "").strip()
    if not cleaned or any(ch.isdigit() for ch in cleaned):
        return ""
    body = _non_punct_chars(cleaned)
    if len(body) < 2 or len(body) > 3:
        return ""
    if any(0x4E00 <= ord(ch) <= 0x9FFF for ch in body):
        return ""
    if not all(_is_kana(ch) or ch == "ー" for ch in body):
        return ""
    if _is_short_reaction_source(cleaned) or _is_meaningful_background_caption_source(cleaned):
        return ""
    if _has_latin_text(cleaned) or _has_bright_bubble_context_pil(image_obj, bbox):
        return ""
    if stats_mean is None:
        stats = _box_luma_stats_pil(image_obj, bbox)
        stats_mean = float(stats[0]) if stats else None
    if stats_mean is None or stats_mean >= 180.0:
        return ""
    if _is_probable_short_vertical_dialogue_box(
        cleaned,
        bbox,
        stats_mean=stats_mean,
        image_size=image_size,
    ):
        return ""
    try:
        x, y, w, h = [int(v) for v in bbox[:4]]
        img_w, img_h = int(image_size[0]), int(image_size[1])
        page_area = max(1, img_w * img_h)
        w = max(1, w)
        h = max(1, h)
        area_ratio = (w * h) / page_area
    except Exception:
        return ""
    if det_conf > 0.65 or ocr_conf >= 0.90:
        return ""
    if area_ratio < 0.0035 or area_ratio > 0.012:
        return ""
    if h < 90 or h < w * 1.45:
        return ""
    if (y + (h / 2.0)) < img_h * 0.35:
        return ""
    try:
        surround_stats = _box_luma_stats_pil(
            image_obj,
            [
                x - 45,
                y - 45,
                w + 90,
                h + 90,
            ],
        )
        surround_mean = float(surround_stats[0]) if surround_stats else None
    except Exception:
        surround_mean = None
    if surround_mean is not None and surround_mean >= 210.0:
        return ""
    return _NONBUBBLE_SHORT_KANA_ART_TEXT_REASON


def _nonbubble_short_reaction_art_text_reason(
    text: str,
    bbox: list,
    image_size: tuple[int, int],
    det_conf: float,
    ocr_conf: float,
    image_obj,
    stats_mean: float | None = None,
) -> str:
    cleaned = str(text or "").strip()
    if not cleaned or any(ch.isdigit() for ch in cleaned):
        return ""
    has_ellipsis_marker = any(ch in cleaned for ch in ".．…‥・･")
    body = _non_punct_chars(cleaned)
    if len(body) < 2 or len(body) > 4:
        return ""
    if any(0x4E00 <= ord(ch) <= 0x9FFF for ch in body):
        return ""
    if not all(_is_kana(ch) or ch == "ー" for ch in body):
        return ""
    if not _is_short_reaction_source(cleaned):
        return ""
    if _has_latin_text(cleaned) or _has_bright_bubble_context_pil(image_obj, bbox):
        return ""
    if stats_mean is None:
        stats = _box_luma_stats_pil(image_obj, bbox)
        stats_mean = float(stats[0]) if stats else None
    if stats_mean is None or stats_mean < 170.0 or stats_mean >= 215.0:
        return ""
    if _is_probable_short_vertical_dialogue_box(
        cleaned,
        bbox,
        stats_mean=stats_mean,
        image_size=image_size,
    ):
        return ""
    try:
        x, y, w, h = [int(v) for v in bbox[:4]]
        img_w, img_h = int(image_size[0]), int(image_size[1])
        page_area = max(1, img_w * img_h)
        w = max(1, w)
        h = max(1, h)
        area_ratio = (w * h) / page_area
    except Exception:
        return ""
    if det_conf > 0.65 or ocr_conf >= 0.75:
        return ""
    if area_ratio < 0.006 or area_ratio > 0.020:
        return ""
    if w < 90 or h < 90:
        return ""
    aspect = w / max(1, h)
    if aspect < 0.70 or aspect > 1.60:
        return ""
    if (y + (h / 2.0)) < img_h * 0.35:
        return ""
    try:
        surround_stats = _box_luma_stats_pil(
            image_obj,
            [
                x - 45,
                y - 45,
                w + 90,
                h + 90,
            ],
        )
        surround_mean = float(surround_stats[0]) if surround_stats else None
    except Exception:
        surround_mean = None
    if surround_mean is not None and surround_mean >= 205.0:
        return ""
    if has_ellipsis_marker:
        return _SHORT_REACTION_WITHOUT_VISUAL_SPEECH_OWNERSHIP_REASON
    return _NONBUBBLE_SHORT_REACTION_ART_TEXT_REASON


def _nonbubble_breath_sfx_art_text_reason(
    text: str,
    bbox: list,
    image_size: tuple[int, int],
    det_conf: float,
    ocr_conf: float,
    image_obj,
    stats_mean: float | None = None,
) -> str:
    cleaned = str(text or "").strip()
    if not cleaned or any(ch.isdigit() for ch in cleaned) or _has_latin_text(cleaned):
        return ""
    body = _non_punct_chars(cleaned)
    if any(0x4E00 <= ord(ch) <= 0x9FFF for ch in body):
        return ""
    if _normalized_kana_body(cleaned) not in {"はあ", "はぁ"}:
        return ""
    if det_conf > 0.65 or ocr_conf < 0.75:
        return ""
    if _has_bright_bubble_context_pil(image_obj, bbox):
        return ""
    if stats_mean is None:
        stats = _box_luma_stats_pil(image_obj, bbox)
        stats_mean = float(stats[0]) if stats else None
    if stats_mean is None or stats_mean >= 225.0:
        return ""
    try:
        x, y, w, h = [int(v) for v in bbox[:4]]
        img_w, img_h = int(image_size[0]), int(image_size[1])
        page_area = max(1, img_w * img_h)
        w = max(1, w)
        h = max(1, h)
        area_ratio = (w * h) / page_area
    except Exception:
        return ""
    if area_ratio < 0.006 or area_ratio > 0.016:
        return ""
    if w < 80 or h < 120 or (y + (h / 2.0)) < img_h * 0.35:
        return ""
    try:
        surround_stats = _box_luma_stats_pil(
            image_obj,
            [
                x - 45,
                y - 45,
                w + 90,
                h + 90,
            ],
        )
        surround_mean = float(surround_stats[0]) if surround_stats else None
    except Exception:
        surround_mean = None
    if surround_mean is not None and surround_mean >= 235.0:
        return ""
    return _NONBUBBLE_BREATH_SFX_ART_TEXT_REASON




















def _rebuild_translation_inputs_from_parent_execution_bundles(
    bundles: list[ParentExecutionBundle],
) -> tuple[dict[str, list[str]], list[str]]:
    """Build a parent-keyed plan while retaining source text as glossary input."""

    pending: dict[str, list[str]] = {}
    glossary: list[str] = []
    for bundle in bundles or []:
        parent_id = str(bundle.parent_id or bundle.bundle_id or "").strip()
        execution_region_id = str(bundle.bundle_id or parent_id).strip()
        text = str(bundle.source_text or "").strip()
        if not parent_id or not execution_region_id or not text:
            continue
        if not _parent_execution_bundle_is_translatable(bundle):
            continue
        glossary.append(text)
        if not str(bundle.translated_text or "").strip():
            pending[parent_id] = [execution_region_id]
    return pending, glossary


def _translation_assignments_from_parent_execution_bundles(
    bundles: list[ParentExecutionBundle],
) -> dict[str, TranslationAssignment]:
    assignments: dict[str, TranslationAssignment] = {}
    for bundle in bundles or []:
        parent_id = str(bundle.parent_id or bundle.bundle_id or "").strip()
        text = str(bundle.source_text or "").strip()
        if not parent_id or not text:
            continue
        if not _parent_execution_bundle_is_translatable(bundle):
            continue
        if str(bundle.translated_text or "").strip():
            continue
        assignments[parent_id] = TranslationAssignment(
            assignment_id=parent_id,
            parent_id=parent_id,
            source_text=text,
            cache_key=text,
            region_ids=[str(bundle.bundle_id or parent_id)],
            source_contract_owner=str(bundle.source_contract_owner or ""),
            source_contract_region_id=str(bundle.source_contract_region_id or ""),
            source_contract_bbox=tuple(int(v) for v in (bundle.source_contract_bbox or [])[:4]),
            source_contract_scope=str(bundle.source_contract_scope or ""),
            source_contract_stage=str(bundle.source_contract_stage or ""),
            source_contract_ocr_confidence=bundle.source_contract_ocr_confidence,
            ocr_backend=str(bundle.ocr_backend or ""),
            ocr_model_path=str(bundle.ocr_model_path or ""),
            ocr_mmproj_path=str(bundle.ocr_mmproj_path or ""),
            ocr_endpoint=str(bundle.ocr_endpoint or ""),
            ocr_prompt_version=str(bundle.ocr_prompt_version or ""),
            source_quality_state=str(bundle.source_quality_state or ""),
            source_quality_action=str(bundle.source_quality_action or ""),
            source_quality_reason_codes=tuple(str(reason) for reason in (bundle.source_quality_reason_codes or [])),
        )
    return assignments


def _parent_execution_bundle_is_translatable(bundle: ParentExecutionBundle) -> bool:
    return bool(bundle.translation_required)












































































def _looks_like_recoverable_speech_region(region: dict, page_class: str = "normal") -> bool:
    if _is_authoritative_parent_execution_region(region):
        return False
    if str(page_class or "").strip().lower() in {"cover", "contents", "chapter_title"}:
        return False
    region_type = str(region.get("type", "") or "").strip().lower()
    if region_type not in {"background_text", "narration_box", "decorative_text"}:
        return False
    text = str(region.get("ocr_text", "") or "").strip()
    if not _is_meaningful_speech_source(text):
        return False
    render = region.get("render", {}) or {}
    route = str(render.get("text_area_route_intent") or region.get("text_area_route_intent") or "").strip()
    container_type = str(render.get("text_area_container_type") or region.get("text_area_container_type") or "").strip()
    if route in {"translate_caption", "translate_caption_background"} or container_type == "caption_background":
        return False
    cleanup_mode = str(render.get("cleanup_mode", "") or "").strip().lower()
    ellipsis_or_reaction = _is_ellipsis_like(text) or _is_short_reaction_source(text)
    bbox = region.get("bbox", [0, 0, 0, 0]) or [0, 0, 0, 0]
    box_w = max(1, int(bbox[2] or 1))
    box_h = max(1, int(bbox[3] or 1))
    source_orientation = str(render.get("source_orientation", "") or "").strip().lower()
    wrap_mode = str(render.get("wrap_mode", "") or "").strip().lower()
    body = _non_punct_chars(text)
    probable_short_vertical = _is_probable_short_vertical_dialogue_box(text, bbox)
    classification_reason = str(render.get("classification_reason", "") or "").strip().lower()
    flags = region.get("flags", {}) or {}
    if (
        classification_reason in {
            _TOP_ROW_BACKGROUND_CAPTION_REASON,
            _TOP_ROW_CAPTION_FRAGMENT_REASON,
        }
        and flags.get("bg_text")
        and not region.get("bubble_id")
    ):
        return False
    if cleanup_mode == "preserve" and classification_reason in {
        "large_short_decorative_sfx_candidate",
        _LOW_CONF_DARK_SHORT_ART_SFX_REASON,
        _MEDIUM_LARGE_KATAKANA_SFX_REASON,
        _NONBUBBLE_SHORT_KANA_ART_TEXT_REASON,
        _NONBUBBLE_SHORT_REACTION_ART_TEXT_REASON,
        _SHORT_REACTION_WITHOUT_VISUAL_SPEECH_OWNERSHIP_REASON,
        _NONBUBBLE_SHORT_REACTION_ART_SFX_REASON,
        _NONBUBBLE_BREATH_SFX_ART_TEXT_REASON,
        _LARGE_LOW_CONFIDENCE_NONBUBBLE_SFX_REASON,
    }:
        return False
    if len(body) > 24:
        return False
    if cleanup_mode == "preserve" and not ellipsis_or_reaction and not probable_short_vertical:
        return False
    if region_type == "decorative_text" and not ellipsis_or_reaction and not probable_short_vertical:
        return False
    if probable_short_vertical:
        return True
    if source_orientation == "vertical" or wrap_mode == "vertical":
        return True
    return box_h > box_w * (0.80 if ellipsis_or_reaction else 0.92)


def _region_can_feed_context(region: dict, page_class: str) -> bool:
    if page_class in {"cover", "contents", "chapter_title"}:
        return False
    flags = region.get("flags", {}) or {}
    if flags.get("ignore") or flags.get("needs_review") or flags.get("hard_fail"):
        return False
    if str(region.get("type", "") or "") not in {"speech_bubble", "narration_box"}:
        return False
    original = str(region.get("ocr_text", "") or "").strip()
    trans = str(region.get("translation", "") or "").strip()
    if not original or not trans:
        return False
    if _translation_is_unsafe_for_output(trans, original):
        return False
    body = _non_punct_chars(trans)
    if len(body) > 24:
        return False
    return True


def _should_use_context_for_text(text: str, region_ids: list[str], regions: list[dict]) -> bool:
    matched = [r for r in regions if r.get("region_id") in region_ids]
    if not matched:
        return False
    region_types = {str(r.get("type", "") or "") for r in matched}
    if not region_types.issubset({"speech_bubble", "narration_box"}):
        return False
    cleaned = _clean_ocr_text(text)
    if not cleaned or _is_short_reaction_source(cleaned) or _is_ellipsis_like(cleaned):
        return False
    body_len = len(_non_punct_chars(cleaned))
    if "narration_box" in region_types:
        return 4 <= body_len <= 16
    return 3 <= body_len <= 9


def _iter_character_sources(entry: dict) -> Iterable[str]:
    if not isinstance(entry, dict):
        return []
    values = []
    for key in ("original", "canonical", "name"):
        value = str(entry.get(key, "")).strip()
        if value:
            values.append(value)
    for alias in entry.get("aliases", []) or []:
        if isinstance(alias, dict):
            value = str(alias.get("source", "")).strip()
        else:
            value = str(alias).strip()
        if value:
            values.append(value)
    return values


def _match_count(texts: list[str], term: str) -> int:
    if not term:
        return 0
    return sum(1 for text in texts if _contains_term(text, term))


def _build_page_style_guide(
    style_guide: dict,
    source_texts: Iterable[str],
    max_glossary: int = 24,
    max_characters: int = 10,
) -> dict:
    if not isinstance(style_guide, dict):
        return default_style_guide()

    texts = [str(text).strip() for text in source_texts if str(text).strip()]
    if not texts:
        return style_guide

    glossary = style_guide.get("glossary", []) or []
    characters = style_guide.get("characters", []) or []
    if len(glossary) <= max_glossary and len(characters) <= max_characters:
        return style_guide

    selected_glossary = list(glossary) if len(glossary) <= max_glossary else []
    glossary_candidates = []
    if len(glossary) > max_glossary:
        for item in glossary:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            if not source or not target:
                continue
            match_count = _match_count(texts, source)
            if match_count <= 0:
                continue
            priority = str(item.get("priority", "")).strip().lower()
            score = (1000 if priority == "hard" else 0) + (match_count * 100) + len(source)
            glossary_candidates.append((score, item))
        glossary_candidates.sort(key=lambda pair: pair[0], reverse=True)
        seen_sources = set()
        for _, item in glossary_candidates:
            source = str(item.get("source", "")).strip()
            if source and source not in seen_sources:
                selected_glossary.append(item)
                seen_sources.add(source)
            if len(selected_glossary) >= max_glossary:
                break

    selected_characters = list(characters) if len(characters) <= max_characters else []
    character_candidates = []
    if len(characters) > max_characters:
        for raw_entry in characters:
            entry = _normalize_character_entry(raw_entry)
            if not entry:
                continue
            score = 0
            for source in _iter_character_sources(entry):
                score += _match_count(texts, source) * 100
                score += len(source)
            if score <= 0:
                continue
            character_candidates.append((score, entry))
        character_candidates.sort(key=lambda pair: pair[0], reverse=True)
        for _, entry in character_candidates[:max_characters]:
            selected_characters.append(entry)

    filtered = dict(style_guide)
    filtered["glossary"] = selected_glossary
    filtered["characters"] = selected_characters
    return filtered


def _polygon_to_bbox(polygon: list) -> list:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _bbox_to_polygon(bbox: list) -> list:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _merge_detections(detections: list, image_size: tuple[int, int], merge: bool = True) -> list:
    if not detections:
        return []
    groups = []
    for polygon, conf in detections:
        try:
            bbox = _polygon_to_bbox(polygon)
        except Exception:
            continue
        groups.append({"bbox": bbox, "polygons": [polygon], "conf": float(conf or 0.0)})
    if not groups or not merge:
        return []
    changed = True
    while changed:
        changed = False
        result = []
        while groups:
            current = groups.pop(0)
            merged = False
            for i, other in enumerate(groups):
                if _should_merge(current["bbox"], other["bbox"], image_size):
                    current["bbox"] = _union_box(current["bbox"], other["bbox"])
                    current["polygons"].extend(other["polygons"])
                    current["conf"] = max(current["conf"], other["conf"])
                    groups.pop(i)
                    merged = True
                    changed = True
                    break
            result.append(current)
            if merged:
                groups = result + groups
                result = []
                break
        if not changed:
            groups = result
    return groups


def _sort_groups(groups: list) -> list:
    """Sort groups in manga reading order (Right-to-Left, Top-to-Bottom)."""
    # Simply sort by Y then -X? No, manga is columns. Vertical columns from right to left.
    # Actually, R-to-L is primary. Top-to-Bottom is secondary within column.
    # But often checking Y first then -X is better for "standard" text detection sorts.
    # Standard "Manga" order:
    # 1. Top-Right quadrant
    # 2. Bottom-Right quadrant
    # ...
    # A simple robust heuristic: Sort by -RightX + Y*0.1? No.
    # Let's use: Top-to-Bottom as primary, Right-to-Left as secondary?
    # No, Manga is Right-to-Left *Pages*, but bubbles?
    # Usually: Top Right -> Bottom Right -> Top Left -> Bottom Left.
    # So we sort primarily by -CenterX, but we need to group vertical lines.
    # Let's try a simple sort: (Y // 100, -X). Rough banding.
    if not groups:
        return []

    def sort_key(g):
        bbox = g["bbox"]
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        # Use simple banding logic to handle slight misalignments
        return (int(cy / 300), -cx)
        # This is very rough.
        # Better: recursively partition?
        # Let's stick to standard reading order logic: Vertical columns starting from right.
        # But 'ComicTextDetector' usually gives them unsorted.
        # A clearer sort:  - (Right Edge), then Top.
        # But top bubbles in right col come before bottom bubbles in right col.
        # So: Band by X?

    # Let's use a simpler heuristic common in OCR:
    # Sort by -X (Right to Left).
    # Then for items with similar X, sort by Y.
    # But if a bubble is far top-left vs near top-right...
    # Correct order: 1 (Top Right), 2 (Bottom Right), 3 (Top Left).
    # So Primary: -X (Right). Secondary: Y (Top).
    # But pure -X is bad because slight X difference overrides massive Y difference.
    # It should be: Sort by columns.

    # Revised Logic:
    # 1. Sort all by -X.
    # 2. Group into "Right", "Center", "Left" columns?
    # Too complex.

    # Let's assume standard R-L, T-B:
    # Just sort by -RightX is usually decent for columns.
    # Let's do: Sort by (sum of X+Y?) No.

    # Let's use the logic found in existing manga-ocr tools:
    # Sort by Y-coordinate first? No, that's English/Webtoon (Top to Bottom).
    # Manga is R-L.
    # Actually, most sophisticated tools use a graph or precise column detection.
    # For now, let's implement a robust "Top-Right to Bottom-Left" sort:
    # Score = - (X + (ImageHeight - Y))?

    # Let's keep it simple and robust for now:
    # Sort by -RightX. (Rightmost first).
    # If X is within a threshold (e.g. 50px), consider them same column, then sort by Y.

    return sorted(groups, key=lambda g: (- (g["bbox"][0] + g["bbox"][2]), g["bbox"][1]))


def _dedupe_groups(groups: list, overlap_threshold: float = 0.85) -> list:
    if not groups:
        return []
    deduped = []
    for group in groups:
        bbox = group.get("bbox")
        if not bbox:
            continue
        if any(_overlap_ratio(bbox, existing.get("bbox", bbox)) >= overlap_threshold for existing in deduped):
            continue
        deduped.append(group)
    return deduped


def _load_image_for_crop(image_path: str):
    """Load an RGB image once for repeated region crops."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(image_path) as img:
            return img.convert("RGB")
    except Exception:
        return None


def _crop_image(image_path: str, bbox: list, expand_wide: bool = True, image_obj=None):
    """Crop image at bbox. Optionally expands wide regions to capture clipped text."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        img = image_obj if image_obj is not None else _load_image_for_crop(image_path)
        if img is None:
            return None
        img_w, img_h = img.size
        x, y, w, h = [int(v) for v in bbox]

        # Expand wide regions (likely impact text with clipped edges)
        # Detection often clips sides of stylized horizontal text
        if expand_wide and h > 0 and w > h * 2:
            # Expand by 15% of width on each side for wide text
            expand = int(w * 0.15)
            x = max(0, x - expand)
            # Recalculate width to reach original right edge + expansion
            x_right = min(img_w, int(bbox[0]) + int(bbox[2]) + expand)
            w = x_right - x

        return img.crop((x, y, x + w, y + h))
    except Exception:
        return None


def _merge_bboxes(bboxes: list, image_size: tuple[int, int]) -> list:
    if not bboxes:
        return []
    boxes = [_expand_box(b, 8, image_size) for b in bboxes]
    changed = True
    while changed:
        changed = False
        result = []
        while boxes:
            current = boxes.pop(0)
            merged = False
            for i, other in enumerate(boxes):
                if _should_merge(current, other, image_size):
                    current = _union_box(current, other)
                    boxes.pop(i)
                    merged = True
                    changed = True
                    break
            result.append(current)
            if merged:
                boxes = result + boxes
                result = []
                break
        if not changed:
            boxes = result
    return boxes


def _should_merge(a: list, b: list, image_size: tuple[int, int]) -> bool:
    if _boxes_overlap(a, b):
        return _overlap_ratio(a, b) >= 0.25
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    x_overlap = not (ax2 < bx or bx2 < ax)
    y_overlap = not (ay2 < by or by2 < ay)
    v_gap = min(abs(by - ay2), abs(ay - by2))
    h_gap = min(abs(bx - ax2), abs(ax - bx2))
    if x_overlap and v_gap <= max(6, min(ah, bh) * 0.25):
        return _union_area_ratio(a, b, image_size) <= 0.03
    if y_overlap and h_gap <= max(6, min(aw, bw) * 0.2):
        return _union_area_ratio(a, b, image_size) <= 0.03
    return False


def _boxes_overlap(a: list, b: list) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay)


def _union_box(a: list, b: list) -> list:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return [x0, y0, x1 - x0, y1 - y0]


def _expand_box(box: list, padding: int, image_size: tuple[int, int]) -> list:
    img_w, img_h = image_size
    x, y, w, h = box
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(img_w, x + w + padding) if img_w else x + w + padding
    y1 = min(img_h, y + h + padding) if img_h else y + h + padding
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _union_area_ratio(a: list, b: list, image_size: tuple[int, int]) -> float:
    img_w, img_h = image_size
    if img_w <= 0 or img_h <= 0:
        return 0.0
    area = img_w * img_h
    union = _union_box(a, b)
    return (union[2] * union[3]) / area


def _overlap_ratio(a: list, b: list) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    min_area = min(aw * ah, bw * bh)
    return inter / max(1, min_area)


def _clean_translation(text: str) -> str:
    cleaned = text.strip()
    lowered = cleaned.lower()
    if lowered.startswith("translation:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    if cleaned.startswith("文字："):
        cleaned = cleaned.split("：", 1)[1].strip()
    if cleaned.startswith("文本："):
        cleaned = cleaned.split("：", 1)[1].strip()
    if cleaned.startswith("原文："):
        cleaned = cleaned.split("：", 1)[1].strip()
    if cleaned.startswith("翻译："):
        cleaned = cleaned.split("：", 1)[1].strip()
    if cleaned.startswith("翻译："):
        cleaned = cleaned.split("：", 1)[1].strip()
    if cleaned.startswith("译文："):
        cleaned = cleaned.split("：", 1)[1].strip()
    if "translates to" in lowered:
        parts = cleaned.split("translates to", 1)
        cleaned = parts[1].strip() if len(parts) > 1 else cleaned
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    cleaned = re.sub(r"<[^>]*>", "", cleaned)
    cleaned = re.sub(r"<\s*e=\d+\s*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\be=\d+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"e=\d+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("□", "")
    cleaned = re.sub(r"(?:口|□){2,}", "", cleaned)
    if _placeholder_ratio(cleaned) >= 0.15:
        cleaned = cleaned.replace("口", "")
    if _placeholder_ratio(cleaned) >= 0.25:
        return ""
    lines = [line for line in cleaned.splitlines() if line.strip()]
    filtered = []
    strip_phrases = [
        "文本：",
        "文本:",
        "仅需翻译",
        "只需翻译",
        "只翻译",
        "不要任何标签",
        "不要任何引号",
        "不要任何解释",
        "不要任何说明",
        "不要任何注释",
        "不要任何多余",
        "不要标签",
        "不要引号",
        "不要解释",
        "不要说明",
        "不要注释",
        "不要多余",
        "只输出译文",
        "仅输出译文",
        "输出译文",
        "只输出翻译",
        "译文如下",
        "翻译如下",
        # Kana-related prompt phrases (from retry prompts)
        "重要：",
        "重要:",
        "你的回答中",
        "绝对不能包含",
        "不能包含",
        "日语假名",
        "ひらがな",
        "カタカナ",
        "只能使用",
        "纯中文",
        "汉字进行翻译",
        "进行翻译",
        "将下面的日语翻译成",
        "翻译成简体中文",
        "翻译成中文",
        "翻譯成中文",
        "翻译成中文是",
        "翻譯成中文是",
        "只输出简体中文",
        "不要片假名",
        "不要平假名",
        "罗马音或英文",
        "日语原文",
        "请将日语",
        "翻译成中文。",
        "翻譯成中文。",
        "输出时只保留",
        "输出只包含",
        "修改后的简体中文",
        "修改后的繁體中文",
        "修改后",
        "修改後",
        "必须原样保留这些占位符",
        "必須原樣保留這些佔位符",
        "原样保留这些占位符",
        "原樣保留這些佔位符",
        "不要翻译、不要删除、不要新增",
        "不要翻譯、不要刪除、不要新增",
        "不要删除、不要新增",
        "不要刪除、不要新增",
        "占位符",
        "佔位符",
        "标记：",
        "標記：",
    ]
    for line in lines:
        head = line.strip()
        lower = head.lower()
        if (
            lower.startswith("text:")
            or lower.startswith("文本:")
            or lower.startswith("文本：")
            or lower.startswith("context:")
            or lower.startswith("input:")
            or lower.startswith("重要：")
            or lower.startswith("重要:")
            or "return only the translation" in lower
            or "output only the translation" in lower
            or "no labels" in lower
            or "no quotes" in lower
            or "no explanations" in lower
            or "<<text>>" in lower
            or "<</text>>" in lower
            # Chinese/Japanese prompt leak patterns
            or "ひらがな" in head
            or "カタカナ" in head
            or "日语假名" in head
            or "绝对不能包含" in head
            or "纯中文汉字" in head
            or "只能使用" in head
            or "进行翻译" in head
            or "翻译成简体中文" in head
            or "翻译成中文" in head
            or "翻譯成中文" in head
            or "修改后的简体中文" in head
            or "修改后的繁體中文" in head
            or "输出时只保留" in head
            or "输出只包含" in head
            or "将下面的日语翻译成" in head
            or "请将日语" in head
            or "日语原文" in head
            or "占位符" in head
            or "佔位符" in head
            or "原样保留这些" in head
            or "原樣保留這些" in head
            or "不要删除" in head
            or "不要刪除" in head
            or "不要新增" in head
        ):
            continue
        head = head.replace("文本：", "").replace("文本:", "")
        for phrase in strip_phrases:
            head = head.replace(phrase, "")
        if not head.strip():
            continue
        filtered.append(head)
    if len(filtered) >= 2 and all(_cjk_ratio(line) >= 0.45 for line in filtered):
        first = filtered[0].strip()
        rest = "".join(line.strip() for line in filtered[1:] if line.strip())
        if first and rest:
            first_body = re.sub(r"[，。！？：；、…\s]", "", first)
            if len(first_body) <= 6 and first[-1] not in "，。！？：；、…,.!?;:":
                cleaned = f"{first}，{rest}"
            else:
                cleaned = f"{first}{rest}"
        else:
            cleaned = "".join(filtered).strip()
    else:
        cleaned = "\n".join(filtered).strip()
    if cleaned.startswith("\"") and cleaned.endswith("\""):
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("`") and cleaned.endswith("`"):
        cleaned = cleaned[1:-1].strip()
    if "Return only the translation" in cleaned:
        cleaned = cleaned.split("Return only the translation", 1)[0].strip()
    cleaned = re.sub(r"[\"'“”]*(?:翻译成中文是[:：]?|翻譯成中文是[:：]?|翻译成中文[:：]?|翻譯成中文[:：]?).*$", "", cleaned).strip()
    cleaned = cleaned.strip("<> ")
    return cleaned


def _sanitize_glossary_target(target: str, source: str, target_lang: str) -> str:
    if not target:
        return ""
    cleaned = _clean_translation(target)
    if "\n" in cleaned:
        cleaned = cleaned.splitlines()[0].strip()
    cleaned = cleaned.strip().strip("“”\"' ").rstrip("。.，,")
    if not cleaned:
        return ""
    if target_lang == "Simplified Chinese":
        cleaned = _normalize_simplified_name_target(cleaned)
    leak_markers = [
        "回复格式",
        "回復格式",
        "回复格式：",
        "回復格式：",
        "不要标点",
        "不要標點",
        "只输出",
        "只輸出",
        "只输出译文",
        "只輸出譯文",
        "traceback",
        "unicodeencodeerror",
        "<stdin>",
        "gbk",
    ]
    if _looks_like_prompt_leak(cleaned) or any(m in cleaned for m in leak_markers):
        return ""
    if target_lang in ["Simplified Chinese", "Traditional Chinese"]:
        if not _language_ok(target_lang, cleaned):
            return ""
        if _is_cjk_term(source) and _is_cjk_term(cleaned):
            digit_chars = set("0123456789０１２３４５６７８９一二三四五六七八九十百千万亿兩两")
            if not any(ch in digit_chars for ch in source) and any(ch in digit_chars for ch in cleaned):
                return ""
            extra_len = len(cleaned) - len(source)
            if len(source) <= 3 and extra_len >= 3:
                expansion_markers = (
                    "这里",
                    "那边",
                    "这个",
                    "那个",
                    "这些",
                    "那些",
                    "二楼",
                    "一楼",
                    "三楼",
                    "四楼",
                    "楼",
                    "习惯",
                    "地方",
                    "浴场",
                    "学园",
                    "学生",
                    "少女",
                    "休息",
                    "休息场",
                    "场",
                    "家",
                )
                if any(marker in cleaned for marker in expansion_markers):
                    return ""
    return cleaned


def _estimate_single_num_predict(text: str, target_lang: str = "") -> int:
    text = str(text or "").strip()
    if not text:
        return 24
    base = len(text)
    if target_lang in {"Simplified Chinese", "Traditional Chinese"}:
        return max(16, min(72, base * 2 + 12))
    return max(24, min(96, base * 3 + 16))


def _max_char_run(text: str) -> int:
    text = str(text or "")
    if not text:
        return 0
    best = 1
    current = 1
    prev = text[0]
    for ch in text[1:]:
        if ch == prev:
            current += 1
            if current > best:
                best = current
        else:
            prev = ch
            current = 1
    return best


def _non_punct_chars(text: str) -> list[str]:
    chars = []
    punct = set("。．，、！？：；….,!?:;·・—～\"'`()[]{}<>-")
    for ch in str(text or ""):
        if ch.isspace() or ch in punct:
            continue
        chars.append(ch)
    return chars


def _leading_char_run(text: str) -> int:
    text = str(text or "")
    if not text:
        return 0
    first = text[0]
    run = 1
    for ch in text[1:]:
        if ch != first:
            break
        run += 1
    return run


def _source_has_stutter_prefix(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return False
    if re.search(r"([ぁ-んァ-ンー])\1+", text):
        return True
    return bool(re.match(r"^[ぁ-んァ-ンー]{2,}[一-龯々ァ-ヶぁ-ゖA-Za-z0-9！？!?…]", text))


def _looks_like_short_repeat_loop(translation: str, source_text: str = "") -> bool:
    body = "".join(_non_punct_chars(translation))
    if len(body) < 3 or len(body) > 10:
        return False
    longest = _max_char_run(body)
    if longest >= 4:
        return True
    lead = _leading_char_run(body)
    if lead >= 3:
        return True
    if _source_has_stutter_prefix(source_text) and longest >= 3:
        return True
    counts = {}
    for ch in body:
        counts[ch] = counts.get(ch, 0) + 1
    dominant = max(counts.values(), default=0)
    return dominant >= max(3, (len(body) * 2 + 2) // 3)


def _looks_like_repetition_loop(translation: str, source_text: str = "") -> bool:
    translation = str(translation or "").strip()
    if not translation:
        return False
    body = _non_punct_chars(translation)
    if _looks_like_short_repeat_loop(translation, source_text):
        return True
    joined = "".join(body)
    if _leading_char_run(joined) >= 4 and len(joined) <= 16:
        return True
    if len(body) < 12:
        return False
    unique = len(set(body))
    longest = _max_char_run(joined)
    if longest >= 10:
        return True
    if unique <= 3 and len(body) >= max(18, len(str(source_text or "").strip()) * 2):
        return True
    if longest >= 6 and unique <= 2 and len(body) >= 16:
        return True
    return False


def _normalize_retry_source(text: str) -> str:
    return str(text or "").strip()


def _is_ellipsis_like(text: str) -> bool:
    stripped = "".join(ch for ch in str(text or "") if ch.strip())
    if not stripped:
        return False
    ellipsis_chars = ".．…‥・･"
    allowed_chars = ellipsis_chars + "—―－-ー〜～?？!！"
    return any(ch in ellipsis_chars for ch in stripped) and all(ch in allowed_chars for ch in stripped)


def _short_reaction_key(text: str) -> str:
    cleaned = _clean_ocr_text(text)
    normalized = _normalize_retry_source(cleaned)
    if not normalized:
        return ""
    if "いいえ" in cleaned:
        return "いいえ"
    normalized = normalized.strip()
    normalized = re.sub(r"[.．…‥・･]+", "", normalized)
    normalized = re.sub(r"[!！?？〜～♡❤♥「」『』（）()]+", "", normalized)
    normalized = normalized.rstrip("ー-—―－")
    return normalized


_SHORT_REACTION_TERMINAL_SYMBOLS = frozenset(".．…‥・･ー—―－-〜～~〰!?！？♡❤♥")


def _split_short_reaction_terminal_symbols(text: str) -> tuple[str, str]:
    index = len(text)
    while index > 0 and text[index - 1] in _SHORT_REACTION_TERMINAL_SYMBOLS:
        index -= 1
    return text[:index], text[index:]


def _normalize_short_reaction_terminal_symbols(symbols: str) -> str:
    return re.sub(r"ー+", "——", symbols)


def _is_short_reaction_source(text: str) -> bool:
    cleaned = _clean_ocr_text(text)
    if not cleaned:
        return False
    if _is_ellipsis_like(cleaned):
        return True
    body = _non_punct_chars(cleaned)
    if not body:
        return False
    key = _short_reaction_key(cleaned)
    if key in {
        "あ",
        "あっ",
        "ああ",
        "あら",
        "え",
        "えっ",
        "えー",
        "ええ",
        "う",
        "うっ",
        "わ",
        "わっ",
        "ま",
        "きゃ",
        "ぎゃ",
        "ふん",
        "フン",
        "ふふ",
        "ほら",
        "まあ",
        "はい",
        "いいえ",
        "ううん",
        "すいません",
        "はっ",
        "はあ",
        "やん",
    }:
        return True
    if (
        len(body) <= 4
        and all(_is_kana(ch) or ch == "ー" for ch in body)
        and key.endswith("はい")
    ):
        return True
    if len(body) <= 4 and all(_is_kana(ch) or ch == "ー" for ch in body):
        seed = [ch for ch in body if ch != "ー"]
        if seed and len(set(seed)) == 1:
            return True
    if len(body) <= 2 and all(_is_kana(ch) for ch in body):
        return True
    return False












def _region_is_sfx_or_decorative_preserve(region: dict) -> bool:
    flags = region.get("flags", {}) or {}
    render = region.get("render", {}) if isinstance(region.get("render"), dict) else {}
    cleanup = str(render.get("cleanup_mode") or "").strip().lower()
    route = str(render.get("text_area_route_intent") or region.get("text_area_route_intent") or "").strip().lower()
    region_type = str(region.get("type") or "").strip().lower()
    if flags.get("sfx") or flags.get("sign"):
        return True
    if route == "preserve_sfx_decorative":
        return True
    if cleanup == "preserve" and region_type in {"decorative_text", "sfx", "sign"}:
        return True
    return region_type in {"sfx", "sign"}


def _translate_short_reaction_fallback(text: str, target_lang: str) -> str:
    if target_lang != "Simplified Chinese":
        return ""
    cleaned = _clean_ocr_text(text)
    if not cleaned:
        return ""
    stripped = "".join(ch for ch in cleaned if ch.strip())
    if _is_ellipsis_like(stripped) or _is_punct_only(stripped):
        return _normalize_short_reaction_terminal_symbols(stripped)
    return ""


def _translation_bad_shape_reasons(translation: str, source_text: str = "") -> list[str]:
    reasons: list[str] = []
    if _looks_like_prompt_leak(translation):
        reasons.append("prompt_leak")
    if _looks_like_repetition_loop(translation, source_text):
        reasons.append("repetition_loop")
    src_body = _non_punct_chars(source_text)
    dst_body = _non_punct_chars(translation)
    if src_body and len(src_body) <= 4 and len(dst_body) >= max(8, len(src_body) * 3):
        reasons.append("short_source_overexpanded")
    if src_body and len(src_body) <= 3:
        punct_count = sum(1 for ch in str(translation or "") if ch in "，,。！？!?；;")
        if punct_count >= 2 and not _is_short_reaction_source(source_text):
            reasons.append("short_source_punctuation_heavy")
    return reasons


def _translation_has_bad_shape(translation: str, source_text: str = "") -> bool:
    return bool(_translation_bad_shape_reasons(translation, source_text))


def _translation_format_artifact_reasons(text: str) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    reasons: list[str] = []
    pairs = {
        "「": "」",
        "『": "』",
        "“": "”",
        "（": "）",
        "(": ")",
        "［": "］",
        "[": "]",
        "【": "】",
        "〈": "〉",
        '"': '"',
        "'": "'",
    }
    reverse_pairs = {v: k for k, v in pairs.items()}
    if len(text) >= 2 and text[0] in pairs and text[-1] == pairs[text[0]]:
        reasons.append(f"outer_wrapper_{text[0]}{text[-1]}")
    if text[0] in set(pairs.keys()) and pairs.get(text[0], "") not in text[1:]:
        reasons.append(f"leading_wrapper_{text[0]}")
    if text[-1] in set(pairs.values()) and reverse_pairs.get(text[-1], "") not in text[:-1]:
        reasons.append(f"trailing_wrapper_{text[-1]}")
    return sorted(dict.fromkeys(reasons))


def _normalize_translation_format(
    target_lang: str,
    translation: str,
) -> tuple[str, list[str]]:
    text = str(translation or "").strip()
    if target_lang != "Simplified Chinese" or not text:
        return text, []
    original = text
    reasons: list[str] = []
    pairs = {
        "「": "」",
        "『": "』",
        "“": "”",
        "（": "）",
        "(": ")",
        "［": "］",
        "[": "]",
        "【": "】",
        "〈": "〉",
        '"': '"',
        "'": "'",
    }
    reverse_pairs = {v: k for k, v in pairs.items()}
    leading = set(pairs.keys())
    trailing = set(pairs.values())
    for _ in range(4):
        changed = False
        if len(text) >= 2 and text[0] in pairs and text[-1] == pairs[text[0]]:
            reasons.append(f"removed_outer_wrapper_{text[0]}{text[-1]}")
            text = text[1:-1].strip()
            changed = True
        if text and text[0] in leading and pairs.get(text[0], "") not in text[1:]:
            reasons.append(f"removed_leading_wrapper_{text[0]}")
            text = text[1:].strip()
            changed = True
        if text and text[-1] in trailing and reverse_pairs.get(text[-1], "") not in text[:-1]:
            reasons.append(f"removed_trailing_wrapper_{text[-1]}")
            text = text[:-1].strip()
            changed = True
        if not changed:
            break
    if text == original:
        return original, []
    if not text:
        return original, []
    return text, sorted(dict.fromkeys(reasons))


def _translation_perf_record_format_normalization(
    record: dict[str, Any] | None,
    *,
    before: str,
    after: str,
    reasons: list[str],
    stage: str,
) -> None:
    if not record or before == after or not reasons:
        return
    record["translation_format_normalized"] = True
    record.setdefault("translation_before_format_normalization", before)
    record["translation_after_format_normalization"] = after
    current = record.setdefault("translation_format_normalization_reasons", [])
    for reason in reasons:
        if reason not in current:
            current.append(reason)
    stages = record.setdefault("translation_format_normalization_stages", [])
    if stage not in stages:
        stages.append(stage)


def _normalize_translation_format_for_record(
    target_lang: str,
    translation: str,
    record: dict[str, Any] | None,
    *,
    stage: str,
) -> str:
    normalized, reasons = _normalize_translation_format(target_lang, translation)
    if reasons and normalized != translation:
        _translation_perf_record_format_normalization(
            record,
            before=str(translation or ""),
            after=normalized,
            reasons=reasons,
            stage=stage,
        )
    return normalized


def _preserve_repeated_terminal_emphasis_symbols(
    source_text: str,
    translation: str,
) -> tuple[str, dict[str, Any]]:
    source = str(source_text or "").strip()
    translated = str(translation or "").strip()
    source_run = _terminal_emphasis_symbol_run(source)
    translation_run = _terminal_emphasis_symbol_run(translated)
    expanded_source = _expanded_emphasis_symbols(source_run)
    expanded_translation = _expanded_emphasis_symbols(translation_run)
    evidence: dict[str, Any] = {
        "changed": False,
        "source_run": source_run,
        "translation_run_before": translation_run,
        "expanded_source_symbols": expanded_source,
        "expanded_translation_symbols_before": expanded_translation,
        "reason": "source_has_no_repeated_terminal_emphasis_run",
    }
    if not translated or len(expanded_source) < 2:
        return translated, evidence
    if expanded_translation == expanded_source:
        evidence["reason"] = "terminal_symbol_multiplicity_already_equal"
        return translated, evidence

    if translation_run:
        body = translated[: len(translated) - len(translation_run)]
    else:
        body = translated.rstrip()
        while body.endswith(("。", "．", ".")):
            body = body[:-1].rstrip()
    corrected = body + source_run
    if not body or corrected == translated:
        evidence["reason"] = "terminal_symbol_conservation_not_applied"
        return translated, evidence
    evidence.update(
        {
            "changed": True,
            "translation_after": corrected,
            "translation_run_after": source_run,
            "expanded_translation_symbols_after": expanded_source,
            "reason": "repeated_terminal_emphasis_count_and_order_conserved",
        }
    )
    return corrected, evidence


def _translation_perf_record_terminal_symbol_conservation(
    record: dict[str, Any] | None,
    evidence: Mapping[str, Any],
) -> None:
    if not record or not evidence.get("changed"):
        return
    record["translation_terminal_symbol_multiplicity_repaired"] = True
    record["translation_terminal_symbol_source_run"] = str(evidence.get("source_run") or "")
    record["translation_terminal_symbol_before"] = str(evidence.get("translation_run_before") or "")
    record["translation_terminal_symbol_after"] = str(evidence.get("translation_run_after") or "")
    record["translation_terminal_symbol_expanded_source"] = str(
        evidence.get("expanded_source_symbols") or ""
    )
    record["translation_terminal_symbol_expanded_before"] = str(
        evidence.get("expanded_translation_symbols_before") or ""
    )
    record["translation_terminal_symbol_reason"] = str(evidence.get("reason") or "")


def _looks_like_runtime_failure(text: str) -> bool:
    text = str(text or "")
    if not text:
        return False
    lowered = text.lower()
    markers = (
        "traceback",
        "unicodeencodeerror",
        "file \"<stdin>\"",
        "<stdin>",
        "gbk codec",
        "the above exception",
    )
    return any(marker in lowered for marker in markers)


def _has_placeholder_token(text: str) -> bool:
    text = str(text or "")
    if not text:
        return False
    return bool(re.search(r"(?:\[\[?N\d+\]?\]|@@N\d+@@)", text))


def _translation_is_unsafe_for_output(text: str, source_text: str = "") -> bool:
    text = str(text or "").strip()
    if not text:
        return False
    if _looks_like_prompt_leak(text):
        return True
    if _looks_like_runtime_failure(text):
        return True
    if _has_placeholder_token(text):
        return True
    if _looks_like_repetition_loop(text, source_text):
        return True
    return False


def _normalized_kana_body(source_text: str) -> str:
    source = str(source_text or "").strip()
    if not source:
        return ""
    normalized_chars: list[str] = []
    for ch in source:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            normalized_chars.append(chr(code - 0x60))
        else:
            normalized_chars.append(ch)
    return "".join(_non_punct_chars("".join(normalized_chars)))


def _apply_source_level_semantic_corrections(source_text: str, translation: str) -> str:
    _ = source_text
    return str(translation or "").strip()


def _repair_bubble_local_nested_speech_translation(
    source_text: str,
    translation: str,
    target_lang: str,
) -> tuple[str, list[str]]:
    """Compatibility probe with no authority to rewrite parent translation."""

    _ = source_text, target_lang
    return str(translation or "").strip(), []


def _should_preserve_decorative_fragment_translation(
    source_text: str,
    region: dict,
    style_guide: dict,
) -> bool:
    if _is_authoritative_parent_execution_region(region):
        return False
    source = str(source_text or "").strip()
    region_type = str(region.get("type", "") or "").strip().lower()
    if region_type not in {"background_text", "decorative_text", "sfx"}:
        return False
    body = "".join(_non_punct_chars(source))
    if not body:
        return False
    if _matched_glossary_terms(source, style_guide):
        return False
    render = region.get("render", {}) or {}
    flags = region.get("flags", {}) or {}
    confidence = region.get("confidence", {}) or {}
    try:
        ocr_conf = float(confidence.get("ocr", 0.0) or 0.0)
    except Exception:
        ocr_conf = 0.0
    classification_reason = str(render.get("classification_reason", "") or "").strip().lower()
    cleanup_mode = str(render.get("cleanup_mode", "") or "").strip().lower()
    if (
        classification_reason == _TOP_ROW_BACKGROUND_CAPTION_REASON
        and cleanup_mode == "local_text_mask"
        and flags.get("bg_text")
        and not region.get("bubble_id")
        and ocr_conf >= 0.90
        and _is_meaningful_background_caption_source(source)
    ):
        return False
    contains_kanji = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in body)
    contains_kana = any(_is_kana(ch) for ch in body)
    if (
        len(body) <= 4
        and contains_kanji
        and contains_kana
        and not any(ch.isdigit() for ch in source)
    ):
        return True
    return False


def _is_authoritative_parent_execution_region(region: dict | None) -> bool:
    if not isinstance(region, dict):
        return False
    render = region.get("render") if isinstance(region.get("render"), dict) else {}
    return bool(
        region.get("parent_execution_authoritative")
        or render.get("parent_execution_authoritative")
        or str(region.get("execution_region_authority") or "")
        == "parent_execution_bundle"
        or str(render.get("execution_region_authority") or "")
        == "parent_execution_bundle"
    )


def _strip_name_suffixes(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    suffixes = ("ちゃん", "さん", "くん", "様", "さま", "先生", "先輩", "殿", "君", "氏", "っち", "ッチ")
    changed = True
    while changed and text:
        changed = False
        for suffix in suffixes:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                changed = True
                break
    return text


def _glossary_target_for_source(style_guide: dict, source: str) -> str:
    if not isinstance(style_guide, dict):
        return ""
    source = str(source or "").strip()
    if not source:
        return ""
    for item in style_guide.get("glossary", []) or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("source", "")).strip() == source:
            return str(item.get("target", "")).strip()
    for char in style_guide.get("characters", []) or []:
        if not isinstance(char, dict):
            continue
        if source in {
            str(char.get("canonical", "")).strip(),
            str(char.get("original", "")).strip(),
        }:
            return str(char.get("translation", "") or char.get("name", "")).strip()
        for alias in char.get("aliases", []) or []:
            if not isinstance(alias, dict):
                continue
            if str(alias.get("source", "")).strip() == source:
                return str(alias.get("target", "") or alias.get("translation", "")).strip()
    return ""


def _replace_omitted_honorific_glossary_target(
    translation: str,
    source: str,
    correct_target: str,
    style_guide: dict,
    expected_count: int = 1,
) -> str:
    if not translation or not source or not correct_target:
        return translation
    base_source = _strip_name_suffixes(source)
    if not base_source or base_source == source:
        return translation
    base_target = _glossary_target_for_source(style_guide, base_source)
    if not base_target or base_target == correct_target:
        return translation
    if not correct_target.startswith(base_target) or base_target not in translation:
        return translation
    replace_count = max(1, expected_count)
    name_boundary = (
        r"$|[，。！？、,.!?]"
        r"|[要也呢吗嘛啊哦呀吧的都就会能想去来回说问看跟和与一]"
    )
    pattern = re.compile(rf"{re.escape(base_target)}(?=(?:{name_boundary}))")
    if pattern.search(translation):
        return pattern.sub(correct_target, translation, count=replace_count)
    if translation.count(base_target) <= replace_count:
        return translation.replace(base_target, correct_target, replace_count)
    return translation


def _romanize_kana_name(text: str) -> str:
    text = _strip_name_suffixes(text)
    if not text:
        return ""
    chars = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            chars.append(chr(code - 0x60))
        else:
            chars.append(ch)
    hira = "".join(chars)
    digraphs = {
        "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
        "しゃ": "sha", "しゅ": "shu", "しょ": "sho",
        "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho",
        "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
        "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
        "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
        "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
        "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
        "じゃ": "ja", "じゅ": "ju", "じょ": "jo",
        "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
        "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    }
    singles = {
        "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
        "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
        "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
        "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
        "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
        "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
        "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
        "や": "ya", "ゆ": "yu", "よ": "yo",
        "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
        "わ": "wa", "を": "o", "ん": "n",
        "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
        "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
        "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
        "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
        "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
        "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
        "ゃ": "ya", "ゅ": "yu", "ょ": "yo",
        "ゔ": "vu", "ー": "-", "っ": "",
    }
    result = []
    i = 0
    geminate = False
    while i < len(hira):
        ch = hira[i]
        if ch == "っ":
            geminate = True
            i += 1
            continue
        pair = hira[i : i + 2]
        romaji = digraphs.get(pair)
        if romaji:
            i += 2
        else:
            romaji = singles.get(ch, ch if ch.isascii() else "")
            i += 1
        if not romaji:
            continue
        if romaji == "-" and result:
            result[-1] = result[-1] + result[-1][-1:]
            continue
        if geminate and romaji[0].isalpha():
            romaji = romaji[0] + romaji
            geminate = False
        result.append(romaji)
    return "".join(result).lower()


def _replace_romanized_glossary_names(translation: str, item: dict) -> str:
    if not translation or not isinstance(item, dict):
        return translation
    target = str(item.get("target", "")).strip()
    source = str(item.get("source", "")).strip()
    reading = str(item.get("reading", "")).strip()
    if not target:
        return translation
    variants = set()
    for seed in (reading, source):
        romaji = _romanize_kana_name(seed)
        if romaji and len(romaji) >= 3:
            variants.add(romaji)
            base = re.sub(r"(chan|san|kun|sama|shi|cchi)$", "", romaji)
            if len(base) >= 3:
                variants.add(base)
    if not variants:
        return translation
    names = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
    pattern = re.compile(
        rf"(?i)\b(?:{names})(?:[-· ]?(?:chan|san|kun|sama|shi|cchi))?(?:酱|醬|桑)?\b"
    )
    return pattern.sub(target, translation)


_CODE_NAME_VARIANTS = {
    "阿尔法": {"阿法", "阿尔发", "亞爾法"},
    "贝塔": {"倍塔", "貝塔", "贝达"},
    "伽玛": {"加玛", "伽馬"},
    "德尔塔": {"戴尔塔", "德塔", "戴塔", "德爾塔", "戴爾塔"},
    "伊普西龙": {"伊普西隆", "伊普西龍", "伊普西隆"},
    "泽塔": {"洁塔", "澤塔", "泽达", "潔塔"},
    "伊塔": {"伊藤", "伊他", "伊达", "伊特", "伊塔兒", "伊塔尔", "伊塔兒"},
    "西塔": {"希塔", "西达"},
    "拉姆达": {"拉姆塔", "拉姆妲"},
    "欧米伽": {"欧米加", "欧米咖"},
}


def _replace_glossary_drift_variants(translation: str, item: dict) -> str:
    if not translation or not isinstance(item, dict):
        return translation
    target = str(item.get("target", "")).strip()
    source = str(item.get("source", "")).strip()
    if not target or not source:
        return translation
    variants = set(_CODE_NAME_VARIANTS.get(target, set()))
    if not variants:
        return translation
    result = translation
    for variant in sorted(variants, key=len, reverse=True):
        if not variant or variant == target:
            continue
        result = result.replace(variant, target)
    return result


def _matched_glossary_terms(source_text: str, style_guide: dict) -> list[dict]:
    if not source_text or not isinstance(style_guide, dict):
        return []
    glossary = style_guide.get("glossary", []) or []
    matched: list[dict] = []
    for item in glossary:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if not source or not target or source not in source_text:
            continue
        matched.append(item)
    matched.sort(
        key=lambda entry: (
            1 if str(entry.get("priority", "")).strip().lower() == "hard" else 0,
            len(str(entry.get("source", ""))),
        ),
        reverse=True,
    )
    selected: list[dict] = []
    for item in matched:
        source = str(item.get("source", "")).strip()
        if any(source in str(existing.get("source", "")).strip() for existing in selected):
            continue
        selected.append(item)
    return selected


def _debug_glossary_terms(terms: Iterable[dict]) -> list[dict]:
    debug_terms = []
    for item in terms or []:
        if not isinstance(item, dict):
            continue
        debug_terms.append(
            {
                "source": str(item.get("source", "")).strip(),
                "target": str(item.get("target", "")).strip(),
                "type": str(item.get("type", "")).strip(),
                "priority": str(item.get("priority", "")).strip(),
            }
        )
    return debug_terms


def _glossary_target_counts(source_text: str, translation: str, style_guide: dict) -> tuple[dict[str, int], dict[str, int]]:
    expected: dict[str, int] = {}
    actual: dict[str, int] = {}
    for item in _matched_glossary_terms(source_text, style_guide):
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if not source or not target:
            continue
        expected[target] = expected.get(target, 0) + source_text.count(source)
    for target in expected:
        actual[target] = translation.count(target)
    return expected, actual


def _collapse_target_overuse(translation: str, target: str, expected_count: int) -> str:
    if not translation or not target or expected_count < 0:
        return translation
    if expected_count <= 1:
        translation = re.sub(rf"(?:{re.escape(target)}){{2,}}", target, translation)
    actual_count = translation.count(target)
    if actual_count <= expected_count:
        return translation
    pieces = translation.split(target)
    rebuilt = []
    used = 0
    for idx, piece in enumerate(pieces[:-1]):
        rebuilt.append(piece)
        if used < expected_count:
            rebuilt.append(target)
            used += 1
    rebuilt.append(pieces[-1])
    return "".join(rebuilt)


def _has_glossary_count_mismatch(source_text: str, translation: str, style_guide: dict) -> bool:
    expected, actual = _glossary_target_counts(source_text, translation, style_guide)
    if not expected:
        return False
    for target, expected_count in expected.items():
        if actual.get(target, 0) != expected_count:
            return True
    return False


def _enforce_glossary(
    translation: str,
    source_text: str,
    style_guide: dict,
) -> str:
    """
    Post-process translation to enforce glossary term consistency.

    If source text contains a glossary source term, ensure the translation
    uses the correct target term. This fixes LLM inconsistency issues.

    Args:
        translation: The LLM translation output
        source_text: The original Japanese text
        style_guide: The style guide containing glossary entries

    Returns:
        Translation with glossary terms enforced
    """
    if not translation or not source_text:
        return translation

    terms_to_enforce = _matched_glossary_terms(source_text, style_guide)
    if not terms_to_enforce:
        return translation

    # For each term that should be in the translation, check and fix
    result = translation

    for item in terms_to_enforce:
        source = str(item.get("source", "")).strip()
        correct_target = str(item.get("target", "")).strip()
        if not source or not correct_target:
            continue
        result = _replace_glossary_drift_variants(result, item)
        updated = _replace_romanized_glossary_names(result, item)
        if updated != result:
            result = updated
        # Skip if target is already present
        if correct_target in result:
            continue

        result = _replace_omitted_honorific_glossary_target(
            result,
            source,
            correct_target,
            style_guide,
            source_text.count(source),
        )
        if correct_target in result:
            continue

        if source_text.startswith(source):
            remainder = source_text[len(source):].lstrip()
            emphatic_remainder = remainder.lstrip("ー〜～っッ ")
            if remainder.startswith(("!", "！")) or emphatic_remainder.startswith(("!", "！")):
                lead = f"{correct_target}！"
            elif remainder.startswith(("?", "？")) or emphatic_remainder.startswith(("?", "？")):
                lead = f"{correct_target}？"
            elif not emphatic_remainder and any(ch in remainder for ch in "ー〜～っッ"):
                lead = f"{correct_target}！"
            elif remainder.startswith(("。", "、", ",", "，")):
                lead = f"{correct_target}，"
            else:
                lead = f"{correct_target}，"
            body = result.lstrip("，。！？!?,、 ")
            body = re.sub(r"^[\u4e00-\u9fff]{1,4}(?:[！!？?，,、]\s*)", "", body)
            result = lead if not body else f"{lead}{body}"
            continue

        # Calculate expected length of the name translation (in characters)
        target_len = len(correct_target)

        # For kana-based names (like まゆ), the model might have produced
        # a different Chinese transliteration (like 真由 instead of 麻由)
        # Look for Chinese character sequences of similar length to replace

        # Find all Chinese character sequences in the result
        # We look for sequences of length target_len
        chinese_sequences = set(re.findall(r'[\u4e00-\u9fff]{' + str(target_len) + '}', result))

        for seq in chinese_sequences:
            if seq == correct_target:
                continue

            # Check context to see if this sequence looks like a name
            # We use regex to ensure we only replace instances that look like names
            name_patterns = [
                (r'(' + re.escape(seq) + r')([酱桑君小姐先生老师])', 1),   # Name + honorific
                (r'(' + re.escape(seq) + r')((的|吗|呢|啊|吧|呀|哦|哇))', 1), # Name + particle
                (r'((是|叫|找|给|对|跟|和|与|爱|恨))(' + re.escape(seq) + r')', 3), # Verb + name
                (r'^(' + re.escape(seq) + r')($|[，。！？])', 1), # Start/End or standalone
                (r'([，。！？])(' + re.escape(seq) + r')([，。！？]|$)', 2), # Surrounded by punct
            ]

            replaced = False
            for pattern, group_idx in name_patterns:
                # If pattern matches, replace ONLY that instance
                if re.search(pattern, result):
                    # We found a context match. Now safely replace.
                    # Note: simple replace() is still risky regarding multiple occurrences of same word used differently
                    # But if we found "seq小姐", it's likely a name.
                    # We'll replace all occurrences if we find strong evidence it's a name anywhere.
                    # This is a compromise.
                    result = result.replace(seq, correct_target)
                    replaced = True
                    break

            if replaced:
                pass

    return result


def _repair_translation_with_glossary(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation: str,
    style_guide: dict,
    debug_record: dict[str, Any] | None = None,
) -> str:
    matched_terms = _matched_glossary_terms(source_text, style_guide)
    if not matched_terms:
        return translation
    _translation_perf_add_path(debug_record, "glossary_repair")
    base_translation = translation if not _translation_is_unsafe_for_output(translation, source_text) else ""
    masked_primary = _translate_with_glossary_placeholders(
        ollama,
        model,
        source_lang,
        target_lang,
        source_text,
        matched_terms,
        debug_record=debug_record,
        debug_phase="glossary_repair_placeholder",
    )
    if masked_primary:
        expected, actual = _glossary_target_counts(source_text, masked_primary, style_guide)
        if expected and all(actual.get(target, 0) == expected_count for target, expected_count in expected.items()):
            return masked_primary
    revised = _enforce_glossary(base_translation, source_text, style_guide) if base_translation else ""
    expected, actual = _glossary_target_counts(source_text, revised, style_guide)
    for target, expected_count in expected.items():
        revised = _collapse_target_overuse(revised, target, expected_count)
    expected, actual = _glossary_target_counts(source_text, revised, style_guide)
    if revised and not _translation_is_unsafe_for_output(revised, source_text):
        if not expected or all(actual.get(target, 0) == expected_count for target, expected_count in expected.items()):
            return revised
    return base_translation


def _translate_with_glossary_placeholders(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    source_text: str,
    matched_terms: list[dict],
    debug_record: dict[str, Any] | None = None,
    debug_phase: str = "glossary_placeholder",
) -> str:
    placeholders: list[tuple[str, str]] = []
    masked_source = source_text
    for idx, item in enumerate(matched_terms):
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if not source or not target or source not in masked_source:
            continue
        token = f"@@N{idx}@@"
        masked_source = masked_source.replace(source, token)
        placeholders.append((token, target))
    if not placeholders:
        return ""
    token_list = " ".join(token for token, _ in placeholders)
    if target_lang == "Simplified Chinese":
        prompt = f"把下面日语译成简体中文，保留这些标记不变：{token_list}\n{masked_source}"
    else:
        prompt = f"Translate to {target_lang}. Keep these tokens unchanged: {token_list}\n{masked_source}"
    try:
        token_limit = _estimate_single_num_predict(source_text, target_lang)
        call_start = time.time()
        raw = ollama.generate(
                _resolve_model(model),
                prompt,
                timeout=30,
                options={
                    "num_predict": token_limit,
                    "temperature": 0.05,
                    "top_p": 0.9,
                },
            )
        _translation_perf_record_llm_call(
            debug_record,
            phase=debug_phase,
            prompt=prompt,
            latency_sec=time.time() - call_start,
            output=raw,
            token_limit=token_limit,
        )
        translated = _clean_translation(raw)
    except Exception:
        if debug_record:
            debug_record.setdefault("failure_retry_reason", []).append(f"{debug_phase}_exception")
        return ""
    if not translated:
        return ""
    if _translation_is_unsafe_for_output(raw, source_text) or _translation_is_unsafe_for_output(translated, source_text):
        return ""
    for token, target in placeholders:
        if token not in translated:
            return ""
        translated = translated.replace(token, target)
    if _translation_is_unsafe_for_output(translated, source_text):
        return ""
    return translated


import threading
import json
import re
from app.io.style_guide import save_style_guide

_glossary_lock = threading.Lock()


def _extract_names_heuristic(texts: list[str]) -> list[str]:
    """
    DEPRECATED: Old heuristic extraction, kept as fallback if MeCab unavailable.
    Looks for repeated katakana sequences (common for character names in manga).
    """
    from collections import Counter

    # Katakana pattern (2+ chars, common for names)
    katakana_pattern = re.compile(r'[\u30A0-\u30FF]{2,}')

    all_katakana = []
    for text in texts:
        matches = katakana_pattern.findall(text)
        all_katakana.extend(matches)

    # Count occurrences - names appear multiple times
    counts = Counter(all_katakana)

    # Filter: names should appear at least 2 times and be 2-8 chars (typical name length)
    potential_names = [
        name for name, count in counts.items()
        if count >= 2 and 2 <= len(name) <= 8
    ]

    # Also look for common suffixes that indicate names
    name_suffixes = ['さん', 'ちゃん', 'くん', '君', '様', '先生', '先輩', '殿']
    for text in texts:
        for suffix in name_suffixes:
            # Pattern: word + suffix
            pattern = re.compile(rf'([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]{{1,6}}){re.escape(suffix)}')
            matches = pattern.findall(text)
            potential_names.extend(matches)

    # Filter common stopwords
    blacklist = {
        "学校", "先生", "同級生", "委員長", "部長", "会長", "社長", "校長",
        "今日", "明日", "昨日", "今年", "来年", "先輩", "後輩", "毎日", "毎朝",
        "日本", "東京", "大阪", "中国", "全国", "本当", "本当に", "嘘",
        "時間", "場所", "気持ち", "問題", "事情", "理由", "意味",
        "能力", "危険", "危機", "戦争", "世界", "宇宙", "地球", "人間",
        "私", "僕", "俺", "自分", "貴様", "お前", "あなた", "アンタ", "君", "我",
        "彼", "彼女", "あいつ", "こいつ", "そいつ", "誰", "何", "何処",
        "男", "女", "人", "奴", "子供", "大人", "生徒", "教師", "医者", "刑事",
        "教室", "部屋", "家", "町", "都市", "国", "王", "城", "村",
    }

    # Deduplicate and filter
    unique_names = set(potential_names)
    return [n for n in unique_names if n not in blacklist]


def _extract_kanji_name_heuristic(text: str) -> list[str]:
    """Fallback: extract likely Kanji names from honorifics and repetition."""
    if not text:
        return []
    from collections import Counter

    honorifics = ["さん", "くん", "ちゃん", "様", "先生", "先輩", "殿", "君", "氏"]
    honorific_pattern = re.compile(
        rf"([\u4E00-\u9FFF]{{2,6}})(?:{'|'.join(honorifics)})"
    )
    names = set(m.group(1) for m in honorific_pattern.finditer(text))

    # Repetition fallback (3+ Kanji, appears 3+ times)
    pattern = re.compile(r"[\u4E00-\u9FFF]{3,6}")
    matches = pattern.findall(text)
    counts = Counter(matches)
    blacklist = {
        "学校", "先生", "同級生", "委員長", "部長", "会長", "社長", "校長",
        "今日", "明日", "昨日", "今年", "来年", "先輩", "後輩", "毎日", "毎朝",
        "日本", "東京", "大阪", "中国", "全国", "本当", "本当に", "嘘",
        "時間", "場所", "気持ち", "問題", "事情", "理由", "意味",
        "能力", "危険", "危機", "戦争", "世界", "宇宙", "地球", "人間",
        "私", "僕", "俺", "自分", "貴様", "お前", "あなた", "アンタ", "君", "我",
        "彼", "彼女", "あいつ", "こいつ", "そいつ", "誰", "何", "何処",
        "男", "女", "人", "奴", "子供", "大人", "生徒", "教師", "医者", "刑事",
        "教室", "部屋", "家", "町", "都市", "国", "王", "城", "村",
    }
    for name, count in counts.items():
        if count >= 3 and name not in blacklist:
            names.add(name)
    return list(names)


def _translate_name(ollama, model: str, name: str, target_lang: str) -> str:
    """Translate a proper noun using a simple, focused prompt."""
    if target_lang == "Simplified Chinese":
        prompt = f"把日语人名'{name}'翻译成中文。\n回复格式：只输出翻译后的名字，不要标点、不要解释。"
    elif target_lang == "Traditional Chinese":
        prompt = f"把日語人名'{name}'翻譯成繁體中文。\n回復格式：只輸出翻譯後的名字，不要標點、不要解釋。"
    else:
        prompt = f"Translate the Japanese name '{name}' to {target_lang}.\nFormat: Output ONLY the translated name, nothing else."

    try:
        result = ollama.generate(model, prompt, timeout=30, options={"num_predict": 30, "temperature": 0.1})
        if result:
            cleaned = _sanitize_glossary_target(result.strip(), name, target_lang)
            if cleaned:
                return cleaned
    except Exception:
        pass
    return ""


def _translate_alias(ollama, model: str, alias: str, hint: str, base_trans: str, target_lang: str) -> str:
    """
    Translate an alias with pattern context.
    The 'hint' comes from MeCab suffix detection (e.g., "亲昵的称呼" for -chan).
    """
    if target_lang == "Simplified Chinese":
        if hint:
            # For names with suffixes like -chan, -san
            prompt = f"'{alias}'是'{base_trans}'的{hint}。把'{alias}'翻译成中文名。\n回复格式：只输出翻译后的名字，不要其他内容。"
        else:
            # For plain aliases
            prompt = f"'{alias}'是人名'{base_trans}'的简称或别称。把'{alias}'翻译成中文。\n回复格式：只输出翻译后的名字，不要标点、不要解释。"
    elif target_lang == "Traditional Chinese":
        if hint:
            prompt = f"'{alias}'是'{base_trans}'的{hint}。把'{alias}'翻譯成繁體中文名。\n回復格式：只輸出翻譯後的名字，不要其他內容。"
        else:
            prompt = f"'{alias}'是人名'{base_trans}'的簡稱或別稱。把'{alias}'翻譯成繁體中文。\n回復格式：只輸出翻譯後的名字，不要標點、不要解釋。"
    else:
        prompt = f"'{alias}' is a nickname for '{base_trans}'. Translate '{alias}' to {target_lang}.\nFormat: Output ONLY the translated name, nothing else."

    try:
        result = ollama.generate(model, prompt, timeout=30, options={"num_predict": 30, "temperature": 0.1})
        if result:
            cleaned = _sanitize_glossary_target(result.strip(), alias, target_lang)
            if cleaned:
                return cleaned
    except Exception:
        pass
    return ""

def _parse_json_list(text: str) -> list:
    """Robustly parse a JSON list from LLM output."""
    if not text:
        return []
    def _list_from_json_value(value: Any) -> list:
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for key in ("translations", "items", "results", "data"):
                nested = value.get(key)
                if isinstance(nested, list):
                    return nested
        return []
    try:
        data = json.loads(text)
        parsed = _list_from_json_value(data)
        if parsed:
            return parsed
    except:
        pass

    # Try finding list pattern
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except:
            pass
    # DeepSeek JSON Output uses a JSON object response_format. Accept a
    # wrapper object when the model obeys that API contract.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = _list_from_json_value(json.loads(match.group()))
            if parsed:
                return parsed
        except:
            pass
    return []


def _parsed_batch_item_id(item: dict) -> str:
    for key in ("id", "region_id", "unit_id"):
        value = str(item.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _parsed_batch_translation_value(item: dict) -> str:
    for key in ("translation", "translated_text", "target", "target_text", "cn", "zh", "chinese", "译文", "翻译", "中文"):
        if key in item:
            cleaned = _clean_translation(str(item.get(key, "") or "").strip())
            if cleaned:
                return cleaned
    return ""


def _build_compact_batch_retry_prompt(
    source_lang: str,
    target_lang: str,
    items: list[dict],
    json_object_wrapper: bool = False,
) -> str:
    lines = []
    if target_lang == "Simplified Chinese":
        if json_object_wrapper:
            lines.extend(
                [
                    "将下面每条日语分别翻译成简体中文。",
                    "只输出有效的 json 对象，格式为{\"translations\":[{\"id\":\"...\",\"translation\":\"...\"}]}。",
                    "不要输出顶层JSON数组，不要合并条目，不要解释，不要保留日语假名。",
                ]
            )
        else:
            lines.extend(
                [
                    "将下面每条日语分别翻译成简体中文。",
                    "只输出JSON数组，格式为[{\"id\":\"...\",\"translation\":\"...\"}]。",
                    "不要合并条目，不要解释，不要保留日语假名。",
                ]
            )
    else:
        if json_object_wrapper:
            lines.extend(
                [
                    f"Translate each {source_lang} item to {target_lang}.",
                    "Output only a valid json object: {\"translations\":[{\"id\":\"...\",\"translation\":\"...\"}]}.",
                    "Do not output a top-level JSON array. Do not merge items or add explanations.",
                ]
            )
        else:
            lines.extend(
                [
                    f"Translate each {source_lang} item to {target_lang}.",
                    "Output only JSON: [{\"id\":\"...\",\"translation\":\"...\"}].",
                    "Do not merge items or add explanations.",
                ]
            )
    payload = [
        {"id": str(item.get("id", "") or ""), "text": str(item.get("text", "") or "")}
        for item in items
        if isinstance(item, dict)
    ]
    lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines)


def _parse_compact_batch_retry_output(
    raw: str,
    items: list[dict],
    target_lang: str,
) -> dict:
    parsed = _parse_json_list(raw)
    parsed_items = [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []
    by_id = {str(item.get("id") or ""): item for item in items if isinstance(item, dict)}
    translations: dict[str, str] = {}
    if parsed_items:
        for item in parsed_items:
            region_id = _parsed_batch_item_id(item)
            source_item = by_id.get(region_id)
            if not region_id or not source_item:
                continue
            source_text = str(source_item.get("text", "") or "")
            translation = _parsed_batch_translation_value(item)
            if not translation:
                continue
            translation, _ = _normalize_translation_format(target_lang, translation)
            if _translation_postcheck_assessment(target_lang, translation, source_text)["hard_failure_reasons"]:
                continue
            translations[region_id] = translation
    if translations:
        return translations

    lines = [line.strip() for line in str(raw or "").splitlines() if line.strip()]
    if len(lines) != len(items):
        return {}
    line_translations: dict[str, str] = {}
    for line, item in zip(lines, items):
        if not isinstance(item, dict):
            return {}
        region_id = str(item.get("id", "") or "").strip()
        source_text = str(item.get("text", "") or "")
        if not region_id:
            return {}
        match = re.match(r"^\s*(?:[-*+]\s*)?(?:\"?([A-Za-z]?\d{3,}|t\d{3})\"?\s*[:：]\s*)?(.*?)\s*$", line)
        if not match:
            return {}
        if match.group(1) and match.group(1) != region_id:
            return {}
        cleaned = _clean_translation(match.group(2))
        if not cleaned:
            return {}
        cleaned, _ = _normalize_translation_format(target_lang, cleaned)
        if _translation_postcheck_assessment(target_lang, cleaned, source_text)["hard_failure_reasons"]:
            return {}
        line_translations[region_id] = cleaned
    return line_translations


def _parse_plain_line_batch_fallback(
    raw: str,
    chunk: list,
    target_lang: str,
    settings: PipelineSettings | None = None,
) -> dict:
    """Map strict one-line-per-item GGUF batch output back to chunk ids."""
    if not (
        settings
        and settings.translator_backend == "GGUF"
        and target_lang == "Simplified Chinese"
    ):
        return {}
    if not raw or not chunk:
        return {}
    if any(marker in raw for marker in ("```", "{", "}", "[", "]")):
        return {}
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    if len(lines) != len(chunk):
        return {}

    translations: dict[str, str] = {}
    for line, item in zip(lines, chunk):
        if not isinstance(item, dict):
            return {}
        source_text = str(item.get("text", "") or "").strip()
        region_id = str(item.get("id", "") or "").strip()
        if not region_id:
            return {}
        cleaned = _clean_translation(line)
        cleaned, _ = _normalize_translation_format(target_lang, cleaned)
        if not _is_safe_plain_batch_line(line, cleaned, source_text, target_lang):
            return {}
        translations[region_id] = cleaned
    return translations


def _is_safe_plain_batch_line(
    raw_line: str,
    cleaned: str,
    source_text: str,
    target_lang: str,
) -> bool:
    raw_line = str(raw_line or "").strip()
    cleaned = str(cleaned or "").strip()
    if not raw_line or not cleaned:
        return False
    if raw_line != cleaned and _looks_like_prompt_leak(raw_line):
        return False
    if _looks_like_prompt_leak(raw_line) or _looks_like_prompt_leak(cleaned):
        return False
    lowered = raw_line.lower()
    if any(marker in lowered for marker in ("system:", "user:", "assistant:", "json", "translation", "source:", "text:")):
        return False
    if any(marker in raw_line for marker in ("系统：", "用户：", "助手：", "原文：", "文本：", "输入：", "输出：", "格式", "解释", "说明")):
        return False
    if re.match(r"^\s*(?:[-*+]\s+|\d+[\.)、]\s+)", raw_line):
        return False
    if re.match(r"^\s*(?:t\d{3}|id|translation|text|source|译文|翻译)\s*[:：]", raw_line, re.IGNORECASE):
        return False
    if any(marker in raw_line for marker in ("{", "}", "[", "]", "```")):
        return False
    if not _language_ok(target_lang, cleaned):
        return False
    if _kana_ratio(cleaned) > 0.02:
        return False
    if _translation_is_unsafe_for_output(cleaned, source_text):
        return False
    if _looks_like_merged_batch_output(cleaned, source_text):
        return False
    source_body = "".join(_non_punct_chars(source_text))
    cleaned_body = "".join(_non_punct_chars(cleaned))
    if source_body and cleaned_body == source_body and _kana_ratio(source_text) > 0.0:
        return False
    if len(cleaned_body) > max(80, len(source_body) * 5 + 20):
        return False
    return True

def _is_garbage(text: str) -> bool:
    """Check if text is likely OCR noise."""
    if not text or len(text.strip()) < 2:
        return True
    # Check if all symbols/numbers (no letters/cjk)
    # Using a simple heuristic: must have at least one CJK or letter
    if not re.search(r"[a-zA-Z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text):
        return True
    return False

def _accumulate_text(state: dict, text: str):
    """Accumulate text for batched analysis."""
    if not text or _is_garbage(text):
        return
    with _glossary_lock:
        buffer = state.setdefault("buffer", [])
        buffer.append(text)
        if len(buffer) > 300:
            buffer.pop(0)



def _trigger_discovery_if_needed(
    state: dict,
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    base_style: dict,
    style_guide_path: str,
    allow_ollama: bool = False,
    discovery_model: str | None = None,
    settings: PipelineSettings | None = None,
):
    """Check buffer size and trigger background discovery if threshold met."""
    import tempfile
    log_path = os.path.join(tempfile.gettempdir(), "auto_glossary_debug.log")

    if not state:
        return

    # User choice: If not allowed to use Ollama for discovery, specifically check if we are using GGUF
    # If users use Ollama for translation, 'ollama' object is valid.
    # If users use GGUF, 'ollama' passed here might be None or a dummy?
    # Actually _process_page logic: if GGUF, ollama might be None.

    # If allow_ollama is False, and we are not using Ollama for translation (model is gguf?), skip.
    # But wait, if we are using Ollama for translation, then 'ollama' is valid and we SHOULD use it?
    # User said: "users can decide whether to use Ollama for our Auto-Glossary system"
    # This implies a global switch.

    if not allow_ollama and (not ollama or not hasattr(ollama, 'generate')):
         # Only allow if we are ALREADY using ollama for translation?
         # Or stricter: if use_ollama_discovery is False, NEVER do background discovery?
         # Let's assume the latter for safety/conflict avoidance.
         return

    # If we don't have an ollama client at all, we can't do it anyway
    if not ollama:
        return

    # Strategy for Hybrid Discovery:
    # 1. If we are already using Ollama (has list_models), use it.
    # 2. If allow_ollama is True: Instantiate a temporary OllamaClient.
    # 3. Else: Fall back to MeCab-only mode (using GGUF or whatever available for simple translation).

    # Logic for Deep Scan Client Resolution
    discovery_client = ollama
    is_real_ollama = hasattr(ollama, "list_models")
    use_deep_scan = False

    # Resolve backend preference.
    # MeCab-only mode must never invoke LLM discovery.
    backend = getattr(settings, "discovery_backend", "Ollama") if settings else "Ollama"
    if not allow_ollama:
        backend = "MeCab"

    # 1. GGUF Backend (LLM discovery path)
    if allow_ollama and (backend == "GGUF" or (discovery_model and ".gguf" in discovery_model.lower())):
        target_path = str(discovery_model or "").strip()
        translation_path = getattr(settings, "gguf_model_path", "") if settings else ""
        needs_swap = (
            settings
            and settings.translator_backend == "GGUF"
            and target_path
            and translation_path
            and os.path.abspath(target_path) != os.path.abspath(translation_path)
        )
        # Reuse existing GGUF client if it matches the target model (avoids double-load)
        if hasattr(ollama, "_model_path"):
            existing_path = getattr(ollama, "_model_path", "")
            if not target_path or (existing_path and os.path.abspath(target_path) == os.path.abspath(existing_path)):
                discovery_client = ollama
                use_deep_scan = True
                needs_swap = False
                logger.info("Deep Scan: Reusing current GGUF client for discovery.")

        if not use_deep_scan and target_path and os.path.isfile(target_path):
            try:
                from app.translate.gguf_client import GGUFClient
                if needs_swap and hasattr(ollama, "close"):
                    logger.info("Deep Scan: Swapping GGUF models to avoid dual load.")
                    ollama.close()
                logger.info(f"Deep Scan: Loading specialized GGUF model: {target_path}")
                n_gpu_layers = settings.gguf_n_gpu_layers if settings else 0
                discovery_client = GGUFClient(
                    model_path=target_path,
                    prompt_style="extract",
                    n_ctx=2048,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=max(1, settings.gguf_n_threads) if settings else 4,
                    n_batch=min(128, settings.gguf_n_batch) if settings else 64,
                )
                use_deep_scan = True
                logger.info("Deep Scan: GGUF enabled via Backend Selection.")
            except Exception as e:
                logger.error(f"Failed to load Deep Scan GGUF model: {e}")
                return
        elif not use_deep_scan:
            logger.warning("Deep Scan: GGUF backend selected but invalid path string.")

    # 2. Ollama Backend
    elif allow_ollama and backend == "Ollama":
        # If user explicitly wants Deep Scan via Ollama (allowed)
        if (discovery_model and discovery_model.lower() not in ["auto-detect", "none", ""]) or allow_ollama:
            if is_real_ollama:
                use_deep_scan = True
            else:
                try:
                    from app.translate.ollama_client import OllamaClient
                    new_client = OllamaClient(
                        base_url=settings.discovery_base_url
                        if settings is not None
                        else "http://localhost:11434",
                        context_tokens=settings.discovery_context
                        if settings is not None
                        else 4096,
                    )
                    if new_client.is_available():
                        discovery_client = new_client
                        use_deep_scan = True
                except Exception:
                    pass

    # Check buffer length
    with _glossary_lock:
        buffer = state.get("buffer", [])
        total_len = sum(len(s) for s in buffer)
        is_running = state.get("is_running", False)

    logger.debug(f"TRIGGER CHECK: total_len={total_len}, is_running={is_running}, deep_scan={use_deep_scan}")

    # Threshold: ~6000 chars to reduce LLM invocations and memory churn
    if total_len >= 6000 and not is_running:
        logger.info(f"TRIGGER: Starting discovery thread! (Deep Scan: {use_deep_scan})")

        with _glossary_lock:
            state["is_running"] = True
            state["had_live_discovery"] = True

        # Choose the correct worker function
        target_func = _run_sakura_discovery if use_deep_scan else _run_discovery

        if use_deep_scan:
            # Synchronous: Pause pipeline to prevent VRAM thrashing with LLM
            logger.info(f"STARTING DISCOVERY SYNCHRONOUSLY (Deep Scan Safe Mode)")
            try:
                 target_func(discovery_client, model, source_lang, target_lang, state, base_style, style_guide_path, discovery_model)
            except Exception as e:
                 logger.error(f"Discovery crashed: {e}")
            if discovery_client is not ollama and hasattr(discovery_client, "close"):
                 discovery_client.close()
            if settings and settings.translator_backend == "GGUF" and hasattr(ollama, "_model_path"):
                 target_path = str(getattr(settings, "gguf_model_path", "")).strip()
                 if target_path and os.path.isfile(target_path):
                     try:
                         from app.translate.gguf_client import GGUFClient
                         n_gpu_layers = settings.gguf_n_gpu_layers
                         state["translation_client"] = GGUFClient(
                             model_path=target_path,
                             prompt_style=settings.gguf_prompt_style,
                             n_ctx=settings.gguf_n_ctx,
                             n_gpu_layers=n_gpu_layers,
                             n_threads=settings.gguf_n_threads,
                             n_batch=settings.gguf_n_batch,
                         )
                         logger.info("Deep Scan: Reloaded translation GGUF client after swap.")
                     except Exception as e:
                         logger.error(f"Deep Scan: Failed to reload translation GGUF model: {e}")
            with _glossary_lock:
                 state["is_running"] = False
        else:
            # Asynchronous: Run MeCab in background (CPU only, safe for concurrency)
            logger.info(f"STARTING DISCOVERY IN BACKGROUND (MeCab Mode)")
            t = threading.Thread(
                target=target_func,
                args=(
                    discovery_client,
                    model,
                    source_lang,
                    target_lang,
                    state,
                    base_style,
                    style_guide_path,
                    bool(discovery_client and hasattr(discovery_client, "generate")),
                )
            )
            t.daemon = True
            t.start()


def _run_sakura_discovery(
    ollama,
    main_model: str,  # The model currently used by the main translation pipeline
    source_lang: str,
    target_lang: str,
    state: dict,
    base_style: dict,
    style_guide_path: str,
    target_model: str | None = None, # User-selected discovery model (None = Auto)
):
    """
    Background worker for Advanced Auto-Glossary discovery.
    """
    accumulated_text = list(state.get("buffer", []))
    if not accumulated_text:
        return

    with _glossary_lock:
         state["buffer"] = []

    # 1. Resolve Best Model for Extraction
    extraction_model = None

    extraction_model = None
    is_gguf_client = hasattr(ollama, "is_available") # Duck typing check for GGUFClient

    # Debug logging for model resolution
    available_models: list[str] = []
    if not is_gguf_client:
        try:
            available_models = list_models()
            logger.debug(f"Available Models: {available_models}")
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
    else:
        # GGUF Client doesn't list models, it HAS a model
        # The 'extraction_model' string is ignored by GGUF generate() usually, but strictly speaking
        # we treat the client as the model.
        extraction_model = "gguf_model"
        logger.info("Deep Scan: Using GGUF Client.")

    try:
        main_model = str(main_model or "")
    # Check if main_model is a valid Ollama model (not a path)
        is_gguf_path = (
            os.path.sep in main_model
            or "/" in main_model
            or "\\" in main_model
            or main_model.lower().endswith(".gguf")
            or os.path.isfile(main_model)
        )
        if not is_gguf_path and "sakura" not in main_model.lower():
             extraction_model = main_model

        # Priority 1: Manual Override
        if target_model and target_model.lower() != "auto-detect" and "sakura" not in target_model.lower():
             extraction_model = target_model

        # Priority 2: Use Main Model if it's in Ollama list
        elif extraction_model and extraction_model in available_models:
             pass # extraction_model is already set to main_model

        # Priority 3: Smart Selection from Available
        elif not extraction_model:
            qwen_candidates = [m for m in available_models if "qwen" in m.lower() and "sakura" not in m.lower()]
            non_sakura_candidates = [m for m in available_models if "sakura" not in m.lower()]

            if qwen_candidates:
                extraction_model = qwen_candidates[0]
            elif non_sakura_candidates:
                extraction_model = non_sakura_candidates[0]

    except Exception:
        pass

    if extraction_model and "sakura" in extraction_model.lower() and not is_gguf_client:
        logger.warning("Deep Scan: Sakura is translation-only; skipping Deep Scan.")
        return

    # FORCE FALLBACK (Only for Ollama)
    if not extraction_model and not is_gguf_client:
        extraction_model = "huihui_ai/qwen3-abliterated:14b"
        logger.warning(f"No model matched. Forcing default '{extraction_model}'")

    # Final check
    if not extraction_model and not is_gguf_client:
         pass

    # Join text into chunks
    full_text = "\n".join(accumulated_text)
    # Join text into chunks
    full_text = "\n".join(accumulated_text)
    chunk_size = 800 # Reduced from 1500 to prevent timeouts
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

    logger.info(f"Starting Discovery on {len(chunks)} chunks using {extraction_model}...")

    for i, chunk in enumerate(chunks):
        glossary_map = {}
        # Build prompt - simple line based is safer for weaker models
        # Build prompt using the shared robust prompt builder
        # This ensures we get JSON output and "Canonical" fields for nickname support
        prompt = build_entity_extraction_prompt(
            text_block=chunk,
            source_lang=source_lang,
            target_lang=target_lang
        )

        # Override for extracting model if it's very dumb (optional, but Qwen 14b is smart enough)
        # If extraction_model is explicitly "sakura", maybe fallback?
        # But we assume Qwen/Smart model is used for Deep Scan as per design.

        # If using Qwen, we can try JSON for better structure, but line-based is universally robust.
        # Let's stick to line-based to be safe for all models including Sakura.

        try:
            # Increase timeout to 600s (10min) for very slow GPUs
            # Reduce num_predict to 1024 to save time
            result = ollama.generate(extraction_model, prompt, timeout=600, options={"num_predict": 1024, "temperature": 0.1})
            if not result:
                continue

            if not result:
                continue

            logger.debug(f"Chunk {i+1} Output:\n{result}\n---")

            # Parse JSON output
            # We use the robust parser from controller (already defined) or local logic
            current_extracted = _parse_json_list(result)

            # Post-process: Resolve Canonical Names (Nicknames -> Full Name Translation)
            # 1. First pass: Collect all "Canonical" -> "Translation" mappings
            #    e.g. Canonical: "Mayuzumi" -> Translation: "Xiao Dai"
            canonical_map = {}
            for item in current_extracted:
                if not isinstance(item, dict): continue
                canon = item.get("canonical", "").strip()
                raw_trans = item.get("translation", "").strip() or item.get("target", "").strip()
                source = item.get("text", "").strip() or item.get("source", "").strip()
                trans = _sanitize_glossary_target(raw_trans, canon or source, target_lang)

                # If this item IS the canonical form (source == canonical), save its translation
                if canon and trans and source == canon:
                    canonical_map[canon] = trans

            # 2. Second pass: Build the Glossary Map
            for item in current_extracted:
                if not isinstance(item, dict): continue

                source = item.get("text", "").strip() or item.get("source", "").strip()
                # Try finding translation in 'target' or 'translation' keys (prompts vary slightly)
                translation = item.get("translation", "").strip() or item.get("target", "").strip()
                type_ = item.get("type", "proper_noun")
                canon = item.get("canonical", "").strip()

                if not source or len(source) < 2:
                    continue
                if source in ["...", "、", "。"]:
                    continue

                # MAGIC: Canonical Name Logic
                # If we have a canonical name (e.g. Mayuzumi -> Xiao Dai)
                # And the current term is a nickname (e.g. Mayu-Mayu),
                # resolving it is tricky.

                # Case 1: If the LLM was lazy and just copied the source (Target="Mayu-Mayu"),
                # we SHOULD overwrite with canonical (Target="Xiao Dai") to be safe.
                # Case 2: If the LLM gave a specific variation (Target="Xiao Dai Dai"),
                # we should PRESERVE it.

                if canon and canon in canonical_map:
                    # Only overwrite if current translation is likely trash (same as source)
                    # or if it's completely empty.
                    is_lazy = (translation == source) or (not translation)
                    if is_lazy:
                        translation = canonical_map[canon]
                translation = _sanitize_glossary_target(translation, source, target_lang)

                if source and translation:
                    glossary_map[source] = {
                        "target": translation,
                        "type": type_,
                        "info": item.get("info", "") or f"Canon: {canon}" if canon else ""
                    }

            # Update global glossary securely
            if glossary_map:
                # Re-load style guide inside lock to prevent race conditions with PipelineWorker
                with _glossary_lock:
                    # Update in-memory state for other components
                    state_map = state.setdefault("map", {})
                    state_map.update(glossary_map)

                    # Update file on disk
                    try:
                        current_sg = _load_style_guide(style_guide_path, target_lang)
                        # We pass None for characters because we only extracted glossary terms here
                        # (Actually we extracted Names as glossary terms, so putting them in glossary map is fine for now)
                        updated_sg = _merge_glossary(current_sg, glossary_map, None)
                        updated_sg = _sanitize_style_guide(updated_sg, target_lang)
                        save_style_guide(style_guide_path, updated_sg)
                    except Exception as io_err:
                        logger.error(f"Glossary Save Error: {str(io_err)}")

        except Exception as e:
            logger.error(f"LLM Error in chunk {i+1}: {str(e)}")


def _run_discovery(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    state: dict,
    base_style: dict,
    style_guide_path: str,
    translate_entities: bool = False,
):
    """
    Background worker for Auto-Glossary discovery using MeCab.

    This function:
    1. Extracts proper nouns using MeCab (Japanese NLP)
    2. Groups names into canonical + aliases by reading matching
    3. Translates each name using focused prompts
    4. Saves results to style_guide.json
    """
    with _glossary_lock:
        buffer = list(state.get("buffer", []))
        state["buffer"] = []

    if not buffer:
        with _glossary_lock:
            state["is_running"] = False
        return

    full_text = "\n".join(buffer)

    # Debug log file (optional)
    log_path = None
    if _GLOSSARY_DEBUG:
        import tempfile
        log_path = os.path.join(tempfile.gettempdir(), "auto_glossary_debug.log")

    try:
        # Determine model to use
        resolved_model = _resolve_model(model)

        logger.info(f"--- MECAB DISCOVERY ---\nBuffer size: {len(full_text)} chars\nModel: {resolved_model}")

        # Try MeCab-based extraction first
        try:
            from app.nlp.mecab_extractor import MeCabExtractor, ExtractedName

            # Load user suffix config if available
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "suffixes.json")
            extractor = MeCabExtractor(config_path=config_path)

            if extractor.is_available:
                # Extract proper nouns
                names = extractor.extract_proper_nouns(full_text)

                if log_path:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"MeCab extracted {len(names)} proper nouns\n")
                        for name in names[:10]:
                            f.write(f"  - {name.surface} (reading: {name.reading}, pos: {name.pos})\n")

                # Fallback: add Kanji name-like chunks if MeCab misses full names
                if source_lang == "Japanese":
                    extra_names = _extract_kanji_name_heuristic(full_text)
                    if extra_names:
                        existing = {n.surface for n in names}
                        for surface in extra_names:
                            if surface not in existing:
                                names.append(ExtractedName(surface=surface, reading=surface, pos="固有名詞"))

                # Group into canonical + aliases
                groups = extractor.group_aliases(names)
                # Translate each group
                for group in groups:
                    # Translate canonical name first
                    canonical_trans = ""
                    if translate_entities:
                        canonical_trans = _translate_name(ollama, resolved_model, group.canonical, target_lang)
                        if not canonical_trans:
                            continue

                    with _glossary_lock:
                        glossary_map = state.setdefault("map", {})
                        characters_list = state.setdefault("characters", [])

                        if canonical_trans:
                            glossary_map[group.canonical] = {
                                "target": canonical_trans,
                                "reading": group.canonical_reading,
                                "pattern": "canonical",
                                "type": "proper_noun"
                            }

                        # Track this as a character
                        char_entry = {
                            "name": canonical_trans or group.canonical,
                            "translation": canonical_trans,
                            "original": group.canonical,
                            "reading": group.canonical_reading,
                            "gender": "unknown",
                            "aliases": []
                        }

                    # Translate each alias
                    for alias in group.aliases:
                        alias_source = alias["source"]
                        alias_hint = alias.get("hint", "")

                        alias_trans = ""
                        if translate_entities:
                            alias_trans = _translate_alias(
                                ollama, resolved_model,
                                alias_source, alias_hint,
                                canonical_trans, target_lang
                            )
                            if not alias_trans:
                                continue

                        with _glossary_lock:
                            if alias_trans:
                                glossary_map[alias_source] = {
                                    "target": alias_trans,
                                    "reading": alias.get("reading", ""),
                                    "pattern": alias.get("pattern", ""),
                                    "hint": alias.get("hint", ""),
                                    "type": "proper_noun"
                                }

                            # Store full alias object with translation
                            alias_obj = dict(alias)
                            alias_obj["target"] = alias_trans
                            char_entry["aliases"].append(alias_obj)

                    # Add character entry
                    with _glossary_lock:
                        # Check if character already exists
                        found = False
                        for existing in characters_list:
                            if existing.get("original") == group.canonical or existing.get("name") == canonical_trans:
                                # Merge aliases
                                existing_aliases = existing.setdefault("aliases", [])
                                for a in char_entry["aliases"]:
                                    if a not in existing_aliases:
                                        existing_aliases.append(a)
                                found = True
                                break

                        if not found:
                            characters_list.append(char_entry)

                # Also translate standalone names (not in groups)
                with _glossary_lock:
                    glossary_map = state.setdefault("map", {})

                for name in names:
                    # Skip if already in glossary
                    if name.surface in glossary_map:
                        continue
                    if translate_entities:
                        trans = _translate_name(ollama, resolved_model, name.surface, target_lang)
                        if not trans:
                            continue
                        with _glossary_lock:
                            glossary_map[name.surface] = {
                                "target": trans,
                                "reading": name.reading,
                                "pattern": "standalone",
                                "type": "proper_noun"
                            }
            else:
                pass

        except ImportError:
            # Fallback to old heuristic method
            if translate_entities:
                heuristic_names = _extract_names_heuristic(buffer)
                for name in heuristic_names:
                    trans = _translate_name(ollama, resolved_model, name, target_lang)
                    if not trans:
                        continue
                    with _glossary_lock:
                        glossary_map = state.setdefault("map", {})
                        if name not in glossary_map:
                            glossary_map[name] = trans

        # Save to disk
        with _glossary_lock:
            chars = list(state.get("characters", []))
            g_map = dict(state.get("map", {}))

        merged_for_save = _merge_glossary(base_style, g_map, chars)
        merged_for_save = _sanitize_style_guide(merged_for_save, target_lang)
        if style_guide_path:
            try:
                save_style_guide(style_guide_path, merged_for_save)
            except Exception as e:
                print(f"Failed to save auto-glossary: {e}")

    except Exception as e:
        print(f"Discovery failed: {e}")
    finally:
        with _glossary_lock:
            state["is_running"] = False


def _apply_auto_glossary(
    base_style: dict,
    state: dict,
    texts: list[str],
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    style_guide_path: str = "",
    allow_ollama: bool = False,
    discovery_model: str | None = None,
    settings: PipelineSettings | None = None,
    mecab_only: bool = True,
) -> dict:
    # 1. Accumulate texts
    if texts:
        for t in texts:
             _accumulate_text(state, t)

    # 2. Trigger discovery
    if mecab_only:
        allow_ollama = False
        discovery_model = None
        settings = None
    _trigger_discovery_if_needed(
        state,
        ollama,
        model,
        source_lang,
        target_lang,
        base_style,
        style_guide_path,
        allow_ollama,
        discovery_model=discovery_model,
        settings=settings,
    )

    # 3. Read current state to merge
    with _glossary_lock:
         chars = list(state.get("characters", []))
         g_map = dict(state.get("map", {}))

    return _merge_glossary(base_style, g_map, chars)


def _translation_perf_records_for_page(
    debug_context: dict | None,
    pending_texts: dict[str, list[str]],
    regions: list[dict],
    *,
    source_lang: str = "",
    target_lang: str = "",
    settings: PipelineSettings | None = None,
    source_text_by_key: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    if not debug_context:
        return {}
    records: dict[str, dict[str, Any]] = {}
    existing_list = debug_context.setdefault("translation_unit_timings", [])
    if not isinstance(existing_list, list):
        existing_list = []
        debug_context["translation_unit_timings"] = existing_list
    existing_by_text = {
        str(record.get("source_text") or ""): record
        for record in existing_list
        if isinstance(record, dict) and str(record.get("source_text") or "")
    }
    region_by_id = {
        str(region.get("region_id") or ""): region
        for region in regions
        if str(region.get("region_id") or "")
    }
    hierarchy = debug_context.get("text_block_hierarchy") or {}
    roots_by_id = {
        str(root.get("root_id") or ""): root
        for root in (hierarchy.get("text_area_root_blocks") or [])
        if isinstance(root, dict) and str(root.get("root_id") or "")
    }
    parents_by_id = {
        str(parent.get("parent_id") or ""): parent
        for parent in (hierarchy.get("parent_logical_text_units") or [])
        if isinstance(parent, dict) and str(parent.get("parent_id") or "")
    }
    for work_key, region_ids in pending_texts.items():
        text = str((source_text_by_key or {}).get(str(work_key), work_key) or "")
        rid_list = [str(rid) for rid in (region_ids or []) if str(rid)]
        source_regions = [region_by_id[rid] for rid in rid_list if rid in region_by_id]
        primary = source_regions[0] if source_regions else {}
        render = primary.get("render") or {}
        root_id = str(
            primary.get("text_block_root_id")
            or render.get("text_block_root_id")
            or ""
        )
        parent_id = str(
            primary.get("parent_logical_text_unit_id")
            or render.get("parent_logical_text_unit_id")
            or primary.get("active_translation_unit_id")
            or render.get("active_translation_unit_id")
            or ""
        )
        root = roots_by_id.get(root_id, {})
        parent = parents_by_id.get(parent_id, {})
        child_ids = []
        for region in source_regions:
            child = str(
                region.get("child_recognized_text_segment_id")
                or (region.get("render") or {}).get("child_recognized_text_segment_id")
                or ""
            )
            if child and child not in child_ids:
                child_ids.append(child)
        if not child_ids and isinstance(parent, dict):
            child_ids = [str(cid) for cid in (parent.get("child_segment_ids") or []) if str(cid)]
        metrics = _translation_perf_source_metrics(text)
        if source_text_by_key is None and text in existing_by_text:
            existing = existing_by_text[text]
            _translation_perf_ensure_contract_fields(
                existing,
                page_id=str(debug_context.get("page_id") or ""),
                source_text=text,
                source_region_ids=rid_list,
                root_id=root_id,
                parent_id=parent_id,
                source_lang=source_lang,
                target_lang=target_lang,
                settings=settings,
                source_adequacy_status=parent.get("source_reconstruction_status") or root.get("root_reconstruction_status"),
                source_regions=source_regions,
                parent=parent,
                root=root,
            )
            records[str(work_key)] = existing
            continue
        translation_unit_id = _translation_perf_unit_id(
            page_id=str(debug_context.get("page_id") or ""),
            source_text=text,
            region_ids=rid_list,
            root_id=root_id,
            parent_id=parent_id,
        )
        source_adequacy_status = parent.get("source_reconstruction_status") or root.get("root_reconstruction_status")
        source_confidence, source_confidence_available = _translation_perf_source_confidence(
            source_regions,
            parent,
            root,
        )
        logical_block_id = _translation_perf_logical_block_id(
            translation_unit_id=translation_unit_id,
            root_id=root_id,
            parent_id=parent_id,
        )
        model_backend = _translation_perf_backend_name(settings)
        prompt_style = _translation_perf_prompt_style(settings)
        record = {
            "page_id": str(debug_context.get("page_id") or ""),
            "root_id": root_id,
            "parent_logical_text_unit_id": parent_id,
            "source_region_ids": rid_list,
            "child_ids": child_ids,
            "root_transaction_status": root.get("root_transaction_status"),
            "parent_acceptance_status": "translation_unit" if bool(parent.get("translation_unit")) else "unknown_or_region_unit",
            "source_text": str(text or ""),
            "normalized_source_text": _normalize_retry_source(text) or str(text or ""),
            "source_text_length": metrics["source_text_length"],
            "japanese_char_count": metrics["japanese_char_count"],
            "punctuation_ellipsis_ratio": metrics["punctuation_ellipsis_ratio"],
            "translated_text_length": 0,
            "translation_path": "pending",
            "translation_paths": [],
            "llm_call_count": 0,
            "llm_calls": [],
            "per_call_latency_sec": [],
            "total_unit_latency_sec": 0.0,
            "prompt_char_count": 0,
            "max_prompt_char_count": 0,
            "output_length": 0,
            "cache_status": "miss_pending_translation",
            "json_repair_fallback_status": [],
            "failure_retry_reason": [],
            "unit_origin": _translation_perf_unit_origin(debug_context, root_id, parent_id, source_regions, parent, root),
            "source_reconstruction_status": parent.get("source_reconstruction_status") or root.get("root_reconstruction_status"),
            "evidence_scopes": _translation_perf_evidence_scopes(debug_context, root_id, parent_id),
            "translation_contract_version": "translation_contract_v1",
            "translation_unit_id": translation_unit_id,
            "logical_block_id": logical_block_id,
            "translation_unit_contract_status": "pending_translation",
            "source_text_confidence": source_confidence,
            "source_text_confidence_available": bool(source_confidence_available),
            "source_language": str(source_lang or "unknown"),
            "target_language": str(target_lang or "unknown"),
            "glossary_context_ids": [],
            "recent_context_ids": [],
            "translation_mode": "batch_or_single",
            "retry_policy": "bounded_postcheck_retry",
            "source_adequacy_status": str(source_adequacy_status or "accepted_for_translation"),
            "failure_flags": [],
            "translation_unit_source_language": str(source_lang or ""),
            "translation_unit_target_language": str(target_lang or ""),
            "translation_unit_source_text": str(text or ""),
            "translation_unit_source_region_ids": rid_list,
            "translation_unit_source_adequacy_status": str(source_adequacy_status or "accepted_for_translation"),
            "translation_unit_translation_mode": "batch_or_single",
            "translation_unit_retry_policy": "bounded_postcheck_retry",
            "translation_result_id": f"tr_{translation_unit_id}",
            "translation_result_contract_status": "pending_translation",
            "translation_result_translated_text": "",
            "model_backend": model_backend or "unknown",
            "prompt_style": prompt_style or "unknown",
            "glossary_applied": [],
            "language_check_status": "pending",
            "format_check_status": "pending",
            "meaning_review_status": "pending",
            "retry_count": 0,
            "runtime_ms": 0,
            "failure_reason": "none",
            "translation_result_model_backend": model_backend,
            "translation_result_prompt_style": prompt_style,
            "translation_language_check_status": "pending",
            "translation_prompt_leak_status": "pending",
            "translation_format_check_status": "pending",
            "translation_glossary_check_status": "not_evaluated",
            "translation_retry_count": 0,
            "translation_runtime_ms": 0,
            "translation_failure_reason": "",
            "translation_result_consumed_text": "",
            "translation_result_consumed_path": "",
            "translation_result_consumed_region_ids": [],
        }
        existing_list.append(record)
        records[str(work_key)] = record
    return records


def _translation_perf_unit_id(
    *,
    page_id: str,
    source_text: str,
    region_ids: list[str],
    root_id: str,
    parent_id: str,
) -> str:
    if parent_id:
        return str(parent_id)
    if root_id:
        return f"tu_{page_id}_{root_id}" if page_id else f"tu_{root_id}"
    if region_ids:
        joined = "_".join(str(rid) for rid in region_ids if str(rid))
        return f"tu_{page_id}_{joined}" if page_id else f"tu_{joined}"
    digest = hashlib.sha1(str(source_text or "").encode("utf-8", "ignore")).hexdigest()[:12]
    return f"tu_{page_id}_{digest}" if page_id else f"tu_{digest}"


def _translation_perf_logical_block_id(
    *,
    translation_unit_id: str,
    root_id: str,
    parent_id: str,
) -> str:
    if parent_id:
        return str(parent_id)
    if root_id:
        return str(root_id)
    if translation_unit_id:
        return str(translation_unit_id)
    return "not_available"


def _translation_perf_source_confidence(
    source_regions: list[dict],
    parent: dict,
    root: dict,
) -> tuple[float | None, bool]:
    values: list[float] = []
    for region in source_regions:
        render = region.get("render") or {}
        for key in (
            "ocr_confidence",
            "logical_text_source_reconstruction_ocr_confidence",
            "diagnostic_ownership_confidence",
        ):
            raw = region.get(key)
            if raw is None:
                raw = render.get(key)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.append(value)
                break
    if not values:
        for source in (parent, root):
            for key in ("ocr_confidence", "source_text_confidence", "confidence"):
                try:
                    value = float(source.get(key))
                except (AttributeError, TypeError, ValueError):
                    continue
                if value > 0:
                    values.append(value)
                    break
            if values:
                break
    if not values:
        return None, False
    return round(sum(values) / len(values), 4), True


def _translation_perf_backend_name(settings: PipelineSettings | None) -> str:
    if not settings:
        return ""
    return str(getattr(settings, "translator_backend", "") or "")


def _translation_perf_prompt_style(settings: PipelineSettings | None) -> str:
    if not settings:
        return ""
    backend = str(getattr(settings, "translator_backend", "") or "")
    if backend == "GGUF":
        return str(getattr(settings, "gguf_prompt_style", "") or "")
    if backend == "DeepSeek":
        return "deepseek_chat_completions"
    return "ollama_batch_single"


def _translation_perf_ensure_contract_fields(
    record: dict[str, Any],
    *,
    page_id: str,
    source_text: str,
    source_region_ids: list[str],
    root_id: str,
    parent_id: str,
    source_lang: str,
    target_lang: str,
    settings: PipelineSettings | None,
    source_adequacy_status: Any,
    source_regions: list[dict] | None = None,
    parent: dict | None = None,
    root: dict | None = None,
) -> None:
    unit_id = str(record.get("translation_unit_id") or "").strip()
    if not unit_id:
        unit_id = _translation_perf_unit_id(
            page_id=page_id,
            source_text=source_text,
            region_ids=source_region_ids,
            root_id=root_id,
            parent_id=parent_id,
        )
    record.setdefault("translation_contract_version", "translation_contract_v1")
    record["translation_unit_id"] = unit_id
    record.setdefault(
        "logical_block_id",
        _translation_perf_logical_block_id(
            translation_unit_id=unit_id,
            root_id=root_id,
            parent_id=parent_id,
        ),
    )
    source_confidence, confidence_available = _translation_perf_source_confidence(
        source_regions or [],
        parent or {},
        root or {},
    )
    record.setdefault("source_text_confidence", source_confidence)
    record.setdefault("source_text_confidence_available", bool(confidence_available))
    record.setdefault("translation_unit_contract_status", "pending_translation")
    record.setdefault("source_language", str(source_lang or "unknown"))
    record.setdefault("target_language", str(target_lang or "unknown"))
    record.setdefault("glossary_context_ids", [])
    record.setdefault("recent_context_ids", [])
    record.setdefault("translation_mode", "batch_or_single")
    record.setdefault("retry_policy", "bounded_postcheck_retry")
    record.setdefault("source_adequacy_status", str(source_adequacy_status or "accepted_for_translation"))
    record.setdefault("failure_flags", [])
    record.setdefault("translation_unit_source_language", str(source_lang or ""))
    record.setdefault("translation_unit_target_language", str(target_lang or ""))
    record.setdefault("translation_unit_source_text", str(source_text or record.get("source_text") or ""))
    record.setdefault("translation_unit_source_region_ids", source_region_ids)
    record.setdefault(
        "translation_unit_source_adequacy_status",
        str(source_adequacy_status or "accepted_for_translation"),
    )
    record.setdefault("translation_unit_translation_mode", "batch_or_single")
    record.setdefault("translation_unit_retry_policy", "bounded_postcheck_retry")
    record.setdefault("translation_result_id", f"tr_{unit_id}")
    record.setdefault("translation_result_contract_status", "pending_translation")
    record.setdefault("translation_result_translated_text", "")
    record.setdefault("model_backend", _translation_perf_backend_name(settings) or "unknown")
    record.setdefault("prompt_style", _translation_perf_prompt_style(settings) or "unknown")
    record.setdefault("glossary_applied", [])
    record.setdefault("language_check_status", "pending")
    record.setdefault("format_check_status", "pending")
    record.setdefault("meaning_review_status", "pending")
    record.setdefault("retry_count", 0)
    record.setdefault("runtime_ms", 0)
    record.setdefault("failure_reason", "none")
    record.setdefault("translation_result_model_backend", _translation_perf_backend_name(settings))
    record.setdefault("translation_result_prompt_style", _translation_perf_prompt_style(settings))
    record.setdefault("translation_language_check_status", "pending")
    record.setdefault("translation_prompt_leak_status", "pending")
    record.setdefault("translation_format_check_status", "pending")
    record.setdefault("translation_glossary_check_status", "not_evaluated")
    record.setdefault("translation_retry_count", 0)
    record.setdefault("translation_runtime_ms", 0)
    record.setdefault("translation_failure_reason", "")
    record.setdefault("translation_result_consumed_text", "")
    record.setdefault("translation_result_consumed_path", "")
    record.setdefault("translation_result_consumed_region_ids", [])


def _translation_perf_source_metrics(text: str) -> dict[str, Any]:
    source = str(text or "")
    jp = sum(1 for ch in source if "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff")
    punct = sum(1 for ch in source if ch in "。、，,.!?！？…・･ー—-~〜「」『』（）()[]［］ 　\n\t")
    return {
        "source_text_length": len(source),
        "japanese_char_count": jp,
        "punctuation_ellipsis_ratio": round(punct / max(1, len(source)), 4),
    }


def _translation_perf_unit_origin(
    debug_context: dict | None,
    root_id: str,
    parent_id: str,
    source_regions: list[dict],
    parent: dict,
    root: dict,
) -> list[str]:
    origins: set[str] = set()
    if str(parent.get("source_reconstruction_status") or "") == "applied":
        origins.add("parent_source_reconstruction")
    if str(root.get("root_reconstruction_status") or "") == "applied":
        origins.add("root_reconstruction")
    for region in source_regions:
        render = region.get("render") or {}
        for key in (
            "text_area_detection_source",
            "logical_text_source_reconstruction_status",
            "source_reconstruction_status",
        ):
            value = str(region.get(key) or render.get(key) or "")
            if value:
                origins.add(value)
        reason_text = " ".join(
            str(item)
            for item in (
                list(region.get("logical_text_source_reconstruction_reason_codes") or [])
                + list(render.get("logical_text_source_reconstruction_reason_codes") or [])
                + list(region.get("hierarchy_reason_codes") or [])
                + list(render.get("hierarchy_reason_codes") or [])
            )
        )
        if "full_page_ctd" in reason_text:
            origins.add("full_page_ctd_evidence")
        if "scoped" in reason_text:
            origins.add("scoped_ctd_or_ocr")
    for scope in _translation_perf_evidence_scopes(debug_context, root_id, parent_id):
        origins.add(scope)
    return sorted(origins) or ["region_pending_text"]


def _translation_perf_evidence_scopes(
    debug_context: dict | None,
    root_id: str,
    parent_id: str,
) -> list[str]:
    if not debug_context or not root_id:
        return []
    scopes: set[str] = set()
    executor = debug_context.get("root_reconstruction_executor") or {}
    attempts = executor.get("attempts") or []
    for attempt in attempts:
        if not isinstance(attempt, dict) or str(attempt.get("root_id") or "") != root_id:
            continue
        evidence = attempt.get("multi_scope_ctd_evidence") or {}
        for scope in evidence.get("source_scopes") or []:
            if scope:
                scopes.add(str(scope))
        for candidate in evidence.get("parent_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            if parent_id and parent_id not in {
                str(candidate.get("parent_candidate_id") or ""),
                str(candidate.get("new_block_id") or ""),
                str(attempt.get("new_block_id") or ""),
            }:
                # Keep root-level scopes too when no direct candidate->parent id
                # exists; otherwise this remains a root-owned evidence summary.
                pass
            for child in candidate.get("child_candidates") or []:
                scope = str(child.get("source_scope") or "")
                if scope:
                    scopes.add(scope)
    return sorted(scopes)


def _translation_perf_add_path(record: dict[str, Any] | None, path: str) -> None:
    if not record or not path:
        return
    paths = record.setdefault("translation_paths", [])
    if path not in paths:
        paths.append(path)
    record["translation_path"] = "+".join(paths)


def _translation_perf_context_ids(context_lines: list[str] | None) -> list[str]:
    ids: list[str] = []
    for line in context_lines or []:
        text = str(line or "").strip()
        if not text:
            continue
        digest = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:12]
        cid = f"recent_context:{digest}"
        if cid not in ids:
            ids.append(cid)
    return ids


def _translation_perf_set_recent_context(
    record: dict[str, Any] | None,
    context_lines: list[str] | None,
) -> None:
    if not record:
        return
    ids = _translation_perf_context_ids(context_lines)
    record["recent_context_ids"] = ids
    record["recent_context_available"] = bool(ids)


def _translation_perf_glossary_context_id(term: dict) -> str:
    source = str(term.get("source") or term.get("canonical") or term.get("original") or "").strip()
    target = str(term.get("target") or term.get("translation") or term.get("name") or "").strip()
    pattern = str(term.get("pattern") or term.get("type") or "").strip()
    base = "|".join([source, target, pattern])
    digest = hashlib.sha1(base.encode("utf-8", "ignore")).hexdigest()[:12]
    return f"style_guide:{digest}"


def _translation_perf_set_glossary_context(
    record: dict[str, Any] | None,
    terms: list[dict] | None,
) -> None:
    if not record:
        return
    ids: list[str] = []
    contexts: list[dict[str, Any]] = []
    for term in terms or []:
        if not isinstance(term, dict):
            continue
        cid = _translation_perf_glossary_context_id(term)
        if cid not in ids:
            ids.append(cid)
            contexts.append({"glossary_context_id": cid, **_debug_glossary_terms([term])[0]})
    record["glossary_context_ids"] = ids
    record["glossary_context_terms"] = contexts


def _translation_perf_update_failure_flags(record: dict[str, Any]) -> None:
    flags: list[str] = []
    for key in (
        "ensure_retry_hard_failure_reasons",
        "failure_retry_reason",
        "translation_glossary_warning_reasons",
    ):
        for item in record.get(key) or []:
            text = str(item or "").strip()
            if text and text not in flags:
                flags.append(text)
    record["failure_flags"] = flags


def _translation_perf_record_llm_call(
    record: dict[str, Any] | None,
    *,
    phase: str,
    prompt: str,
    latency_sec: float,
    output: str,
    token_limit: int | None,
    status: str = "ok",
    shared_batch_size: int | None = None,
    error: str | None = None,
    request_id: str | None = None,
) -> None:
    if not record:
        return
    prompt_len = len(str(prompt or ""))
    output_len = len(str(output or ""))
    call = {
        "request_id": str(request_id or _translation_perf_request_id(phase)),
        "phase": phase,
        "latency_sec": round(max(0.0, float(latency_sec or 0.0)), 4),
        "prompt_char_count": prompt_len,
        "output_length": output_len,
        "token_limit": token_limit,
        "status": status,
    }
    if shared_batch_size is not None:
        call["shared_batch_size"] = int(shared_batch_size)
    if error:
        call["error"] = str(error)
    record.setdefault("llm_calls", []).append(call)
    record.setdefault("per_call_latency_sec", []).append(call["latency_sec"])
    record["llm_call_count"] = int(record.get("llm_call_count") or 0) + 1
    record["prompt_char_count"] = int(record.get("prompt_char_count") or 0) + prompt_len
    record["max_prompt_char_count"] = max(int(record.get("max_prompt_char_count") or 0), prompt_len)
    record["output_length"] = max(int(record.get("output_length") or 0), output_len)
    record["total_unit_latency_sec"] = round(
        float(record.get("total_unit_latency_sec") or 0.0) + call["latency_sec"],
        4,
    )


def _translation_perf_request_id(phase: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(phase or "request")).strip("_") or "request"
    return f"{token}_{time.time_ns()}"


def _summarize_translation_requests(debug_context: dict[str, Any] | None) -> None:
    if not isinstance(debug_context, dict):
        return
    requests_by_id: dict[str, dict[str, Any]] = {}
    for unit in debug_context.get("translation_unit_timings") or []:
        if not isinstance(unit, dict):
            continue
        unit_id = str(unit.get("translation_unit_id") or "")
        backend = str(unit.get("model_backend") or unit.get("translation_result_model_backend") or "unknown")
        for index, call in enumerate(unit.get("llm_calls") or []):
            if not isinstance(call, dict):
                continue
            request_id = str(call.get("request_id") or f"{unit_id or 'unit'}:{index}")
            record = requests_by_id.setdefault(
                request_id,
                {
                    "request_id": request_id,
                    "phase": str(call.get("phase") or "unknown"),
                    "backend": backend,
                    "status": str(call.get("status") or "unknown"),
                    "latency_sec": round(_numeric_or_zero(call.get("latency_sec")), 4),
                    "prompt_char_count": int(call.get("prompt_char_count") or 0),
                    "output_length": int(call.get("output_length") or 0),
                    "token_limit": call.get("token_limit"),
                    "payload_items": int(call.get("shared_batch_size") or 1),
                    "translation_unit_ids": [],
                },
            )
            if unit_id and unit_id not in record["translation_unit_ids"]:
                record["translation_unit_ids"].append(unit_id)

    request_records = list(requests_by_id.values())
    phase_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    backend_counts: dict[str, int] = {}
    for record in request_records:
        phase = str(record.get("phase") or "unknown")
        status = str(record.get("status") or "unknown")
        backend = str(record.get("backend") or "unknown")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
    total_latency = sum(_numeric_or_zero(record.get("latency_sec")) for record in request_records)
    payload_items = sum(int(record.get("payload_items") or 0) for record in request_records)
    summary = {
        "request_count": len(request_records),
        "payload_item_count": payload_items,
        "total_request_latency_sec": round(total_latency, 4),
        "max_request_latency_sec": round(
            max([_numeric_or_zero(record.get("latency_sec")) for record in request_records] or [0.0]),
            4,
        ),
        "phase_counts": phase_counts,
        "status_counts": status_counts,
        "backend_counts": backend_counts,
        "requests": request_records,
    }
    debug_context["translation_request_summary"] = summary
    debug_context.setdefault("counts", {})["translation_request_count"] = len(request_records)
    debug_context.setdefault("counts", {})["translation_request_payload_items"] = payload_items


def _translation_perf_set_final(
    record: dict[str, Any] | None,
    *,
    translation: str,
    status: str | None = None,
) -> None:
    if not record:
        return
    final_text = str(translation or "")
    record["translated_text_length"] = len(final_text)
    record["translated_text"] = final_text
    record["output_length"] = max(int(record.get("output_length") or 0), len(final_text))
    record["translation_unit_contract_status"] = "accepted_for_translation"
    record["translation_result_translated_text"] = final_text
    record["translation_result_contract_status"] = "complete" if final_text.strip() else "empty_result"
    record["translation_runtime_ms"] = int(round(float(record.get("total_unit_latency_sec") or 0.0) * 1000))
    llm_calls = record.get("llm_calls") if isinstance(record.get("llm_calls"), list) else []
    record["translation_retry_count"] = max(0, len(llm_calls) - 1)
    record["retry_count"] = int(record["translation_retry_count"])
    record["runtime_ms"] = int(record["translation_runtime_ms"])
    hard_reasons = list(record.get("ensure_retry_hard_failure_reasons") or [])
    failure_reasons = list(record.get("failure_retry_reason") or [])
    if hard_reasons or failure_reasons:
        failure_reason = ",".join(
            str(reason)
            for reason in dict.fromkeys(hard_reasons + failure_reasons)
            if str(reason)
        )
    else:
        failure_reason = "none"
    record["translation_failure_reason"] = failure_reason
    record["failure_reason"] = failure_reason
    record["meaning_review_status"] = "needs_review" if failure_reason else "pass"
    _translation_perf_update_failure_flags(record)
    if status:
        statuses = record.setdefault("json_repair_fallback_status", [])
        if status not in statuses:
            statuses.append(status)


def _translation_perf_set_glossary_status(
    record: dict[str, Any] | None,
    *,
    applied_terms: list[dict] | None,
    ignored_terms: list[dict] | None,
    warnings: list[str] | None,
) -> None:
    if not record:
        return
    warning_list = [str(item) for item in (warnings or []) if str(item)]
    record["translation_glossary_terms_applied"] = _debug_glossary_terms(applied_terms or [])
    record["translation_glossary_terms_ignored"] = _debug_glossary_terms(ignored_terms or [])
    record["translation_glossary_warning_reasons"] = warning_list
    record["glossary_applied"] = _debug_glossary_terms(applied_terms or [])
    if warning_list:
        record["translation_glossary_check_status"] = "warning"
        soft = record.setdefault("ensure_retry_soft_warning_reasons", [])
        for warning in warning_list:
            if warning not in soft:
                soft.append(warning)
        record["meaning_review_status"] = "needs_review"
    elif (applied_terms or ignored_terms):
        record["translation_glossary_check_status"] = "checked"
        record.setdefault("meaning_review_status", "pass")
    else:
        record["translation_glossary_check_status"] = "not_applicable"
        record.setdefault("meaning_review_status", "pass")
    _translation_perf_update_failure_flags(record)


def _translation_perf_mark_region_consumed(
    record: dict[str, Any] | None,
    region: dict,
    translation: str,
    *,
    consumed_path: str,
) -> None:
    if not record or not isinstance(region, dict):
        return
    unit_id = str(record.get("translation_unit_id") or "")
    result_id = str(record.get("translation_result_id") or "")
    result_text = str(record.get("translation_result_translated_text") or translation or "")
    rid = str(region.get("region_id") or "")
    if unit_id:
        region["translation_unit_id"] = unit_id
    if result_id:
        region["translation_result_id"] = result_id
    region["translation_result_translated_text"] = result_text
    region["translation_result_consumed_text"] = str(translation or "")
    region["translation_result_consumed_path"] = consumed_path
    render = region.setdefault("render", {})
    if isinstance(render, dict):
        if unit_id:
            render["translation_unit_id"] = unit_id
        if result_id:
            render["translation_result_id"] = result_id
        render["translation_result_translated_text"] = result_text
        render["translation_result_consumed_text"] = str(translation or "")
        render["translation_result_consumed_path"] = consumed_path
    record["translation_result_consumed_text"] = str(translation or "")
    record["translation_result_consumed_path"] = consumed_path
    consumed = record.setdefault("translation_result_consumed_region_ids", [])
    if rid and rid not in consumed:
        consumed.append(rid)


def _translation_reuses_source_text(translation: str, source_text: str, target_lang: str) -> bool:
    if target_lang != "Simplified Chinese":
        return False
    cleaned_translation = re.sub(r"\s+", "", str(translation or ""))
    cleaned_source = re.sub(r"\s+", "", str(source_text or ""))
    if not cleaned_translation or not cleaned_source:
        return False
    translation_body = "".join(_non_punct_chars(cleaned_translation))
    source_body = "".join(_non_punct_chars(cleaned_source))
    if not translation_body or not source_body:
        return False
    if translation_body == source_body and _kana_ratio(source_body) > 0:
        return True
    if len(source_body) >= 4 and source_body in translation_body and _kana_ratio(source_body) > 0:
        return True
    return False


def _translation_postcheck_assessment(
    target_lang: str,
    translation: str,
    source_text: str,
) -> dict[str, Any]:
    text = str(translation or "").strip()
    language_ok = _language_ok(target_lang, text)
    bad_shape_reasons = _translation_bad_shape_reasons(text, source_text)
    prompt_leak = _looks_like_prompt_leak(text)
    repetition_loop = _looks_like_repetition_loop(text, source_text)
    merged_batch_output = _looks_like_merged_batch_output(text, source_text)
    source_reuse = _translation_reuses_source_text(text, source_text, target_lang)
    kana_ratio = _kana_ratio(text)
    chinese_ratio = _cjk_ratio(text)
    hard: list[str] = []
    soft: list[str] = []
    if not text:
        hard.append("empty_output")
    if merged_batch_output:
        hard.append("merged_batch_output")
    if prompt_leak:
        hard.append("prompt_leak")
    if repetition_loop:
        hard.append("repetition_loop")
    if source_reuse:
        hard.append("source_reuse")
    if target_lang == "Simplified Chinese" and text:
        if kana_ratio > 0.1:
            hard.append("meaningful_japanese_kana")
        elif kana_ratio > 0:
            soft.append("minor_kana_trace")
        if chinese_ratio < 0.3:
            hard.append("low_chinese_ratio")
    elif not language_ok:
        hard.append("language_check_failed")
    for reason in bad_shape_reasons:
        if reason in {"prompt_leak", "repetition_loop"}:
            continue
        if reason == "short_source_overexpanded":
            src_len = max(1, len(_non_punct_chars(source_text)))
            dst_len = len(_non_punct_chars(text))
            if dst_len >= max(18, src_len * 6):
                hard.append("severe_short_source_overexpanded")
            else:
                soft.append(reason)
        elif reason == "short_source_punctuation_heavy":
            soft.append(reason)
        else:
            soft.append(reason)
    format_reasons = _translation_format_artifact_reasons(text)
    if format_reasons:
        soft.append("quote_or_bracket_punctuation")
    hard = sorted(dict.fromkeys(hard))
    soft = sorted(dict.fromkeys(reason for reason in soft if reason not in hard))
    return {
        "translation": text,
        "language_ok": language_ok,
        "bad_shape": bool(bad_shape_reasons),
        "bad_shape_reasons": bad_shape_reasons,
        "kana_ratio": round(kana_ratio, 4),
        "chinese_ratio": round(chinese_ratio, 4),
        "prompt_leak": prompt_leak,
        "repetition_loop": repetition_loop,
        "merged_batch_output": merged_batch_output,
        "source_reuse": source_reuse,
        "format_artifact_reasons": format_reasons,
        "hard_failure_reasons": hard,
        "soft_warning_reasons": soft,
        "retry_required": bool(hard),
    }


def _translation_perf_record_pre_ensure(
    record: dict[str, Any] | None,
    assessment: dict[str, Any],
) -> None:
    if not record:
        return
    record["pre_ensure_translation"] = assessment.get("translation", "")
    record["pre_ensure_language_ok"] = bool(assessment.get("language_ok"))
    record["pre_ensure_bad_shape"] = bool(assessment.get("bad_shape"))
    record["pre_ensure_bad_shape_reasons"] = list(assessment.get("bad_shape_reasons") or [])
    record["pre_ensure_kana_ratio"] = assessment.get("kana_ratio", 0.0)
    record["pre_ensure_chinese_ratio"] = assessment.get("chinese_ratio", 0.0)
    record["pre_ensure_prompt_leak"] = bool(assessment.get("prompt_leak"))
    record["pre_ensure_repetition_loop"] = bool(assessment.get("repetition_loop"))
    record["pre_ensure_merged_batch_output"] = bool(assessment.get("merged_batch_output"))
    record["pre_ensure_source_reuse"] = bool(assessment.get("source_reuse"))
    record["pre_ensure_format_artifact_reasons"] = list(assessment.get("format_artifact_reasons") or [])
    record["ensure_retry_required"] = bool(assessment.get("retry_required"))
    record["ensure_retry_required_reason"] = ",".join(assessment.get("hard_failure_reasons") or [])
    record["ensure_retry_hard_failure_reasons"] = list(assessment.get("hard_failure_reasons") or [])
    record["ensure_retry_soft_warning_reasons"] = list(assessment.get("soft_warning_reasons") or [])
    record["translation_language_check_status"] = (
        "pass"
        if bool(assessment.get("language_ok")) and not assessment.get("hard_failure_reasons")
        else "retry_required"
    )
    record["language_check_status"] = record["translation_language_check_status"]
    record["translation_prompt_leak_status"] = "fail" if assessment.get("prompt_leak") else "pass"
    record["translation_format_check_status"] = (
        "warning"
        if assessment.get("format_artifact_reasons") or assessment.get("soft_warning_reasons")
        else "pass"
    )
    record["format_check_status"] = record["translation_format_check_status"]
    record["meaning_review_status"] = "retry_required" if assessment.get("hard_failure_reasons") else "pass"
    _translation_perf_update_failure_flags(record)


def _translation_perf_record_post_ensure(
    record: dict[str, Any] | None,
    translation: str,
    *,
    acceptance_status: str,
    retry_skipped_reason: str = "",
) -> None:
    if not record:
        return
    record["post_ensure_translation"] = str(translation or "")
    record["batch_output_acceptance_status"] = acceptance_status
    record["translation_language_check_status"] = "pass" if str(translation or "").strip() else "empty"
    record["language_check_status"] = record["translation_language_check_status"]
    if acceptance_status:
        record["translation_result_contract_status"] = str(acceptance_status)
        if str(acceptance_status).startswith("deterministic_short_reaction"):
            record["meaning_review_status"] = "deterministic_short_reaction"
    if retry_skipped_reason:
        record["ensure_retry_skipped_reason"] = retry_skipped_reason


def _is_deepseek_translation_backend(settings: PipelineSettings | None) -> bool:
    return str(getattr(settings, "translator_backend", "") or "") == "DeepSeek"


def _deepseek_request_timeout(default_timeout: int = 600) -> int:
    return min(int(default_timeout), 35)


def _sanitized_llm_output_snippet(text: str, limit: int = 240) -> str:
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > limit:
        return cleaned[:limit] + "..."
    return cleaned


def _batch_translate(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    style_guide: dict,
    items: list,
    context_lines: list[str] | None = None,
    settings: PipelineSettings | None = None,
    debug_records_by_text: dict[str, dict[str, Any]] | None = None,
) -> dict:
    resolved = _resolve_model(model)
    translations: dict = {}

    # Defaults
    temp = 0.2
    top_p = 0.9

    if settings:
        if settings.translator_backend == "GGUF":
             temp = settings.gguf_temperature
             top_p = settings.gguf_top_p
        else:
             temp = settings.ollama_temperature
             top_p = settings.ollama_top_p

    deepseek_backend = _is_deepseek_translation_backend(settings)
    batch_size = 16
    if settings and settings.translator_backend == "GGUF":
        # Smaller GGUF batches are more stable for Sakura-style JSON output
        # and avoid pathological long generations on dense pages.
        batch_size = 6
    elif deepseek_backend:
        batch_size = 8
    initial_batch_size = batch_size
    effective_batch_size = batch_size
    start = 0
    chunk_index = 0
    while start < len(items):
        chunk = items[start : start + effective_batch_size]
        start += len(chunk)
        chunk_index += 1
        prompt = build_batch_translation_prompt(
            source_lang,
            target_lang,
            style_guide,
            chunk,
            context_lines=context_lines,
            json_object_wrapper=deepseek_backend,
        )
        token_limit = _estimate_num_predict(chunk)
        if settings and settings.translator_backend == "GGUF":
            token_limit = min(token_limit, 224 if target_lang == "Simplified Chinese" else 256)
        elif deepseek_backend:
            token_limit = max(token_limit, 256)
        request_options = {"num_predict": token_limit, "temperature": temp, "top_p": top_p}
        request_timeout = 600
        if deepseek_backend:
            request_options["response_format"] = {"type": "json_object"}
            request_options["thinking"] = {"type": "disabled"}
            request_timeout = _deepseek_request_timeout(600)
        chunk_records = [
            (debug_records_by_text or {}).get(str(item.get("id") or ""))
            or (debug_records_by_text or {}).get(str(item.get("text") or ""))
            for item in chunk
            if isinstance(item, dict)
        ]
        for record in chunk_records:
            _translation_perf_add_path(record, "batch")
            _translation_perf_set_recent_context(record, context_lines)
            if record:
                record["batch_initial_size"] = initial_batch_size
                record["batch_effective_size"] = len(chunk)
                record["batch_chunk_index"] = chunk_index
        batch_request_id = _translation_perf_request_id("batch_chunk")
        call_start = time.time()
        try:
            raw = ollama.generate(
                resolved,
                prompt,
                timeout=request_timeout,
                options=request_options,
            )
            call_latency = time.time() - call_start
            for record in chunk_records:
                if record:
                    record["batch_latency_sec"] = round(call_latency, 4)
                _translation_perf_record_llm_call(
                    record,
                    phase="batch_chunk",
                    prompt=prompt,
                    latency_sec=call_latency,
                    output=raw,
                    token_limit=token_limit,
                    shared_batch_size=len(chunk),
                    request_id=batch_request_id,
                )
        except Exception as exc:
            call_latency = time.time() - call_start
            for record in chunk_records:
                _translation_perf_record_llm_call(
                    record,
                    phase="batch_chunk",
                    prompt=prompt,
                    latency_sec=call_latency,
                    output="",
                    token_limit=token_limit,
                    status="exception",
                    shared_batch_size=len(chunk),
                    error=f"{type(exc).__name__}: {exc}",
                    request_id=batch_request_id,
                )
                record.setdefault("failure_retry_reason", []).append("batch_chunk_exception")
                if record:
                    record["adaptive_batch_split_trigger"] = "batch_chunk_exception"
            if settings and settings.translator_backend == "GGUF" and effective_batch_size > 2:
                effective_batch_size = max(2, math.ceil(effective_batch_size / 2))
            logger.warning("Batch translation chunk failed; falling back to single translation for this chunk.", exc_info=True)
            continue
        parsed = _parse_json_list(raw)
        parsed_items = [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []
        if not parsed_items:
            line_fallback = _parse_plain_line_batch_fallback(raw, chunk, target_lang, settings)
            if line_fallback:
                for item in chunk:
                    if not isinstance(item, dict):
                        continue
                    record = (
                        (debug_records_by_text or {}).get(str(item.get("id") or ""))
                        or (debug_records_by_text or {}).get(str(item.get("text") or ""))
                    )
                    if record:
                        record.setdefault("json_repair_fallback_status", []).append("batch_plain_line_fallback")
                logger.warning(
                    "batch_plain_line_fallback accepted %d batch translations for ids: %s",
                    len(line_fallback),
                    ", ".join(line_fallback.keys()),
                )
                translations.update(line_fallback)
                continue
            for record in chunk_records:
                if record:
                    record.setdefault("json_repair_fallback_status", []).append("batch_no_usable_json")
                    record.setdefault("failure_retry_reason", []).append("batch_no_usable_json_single_fallback")
                    record["batch_empty_translation_ratio"] = 1.0
                    record["adaptive_batch_split_trigger"] = "batch_no_usable_json"
            if settings and settings.translator_backend == "GGUF" and effective_batch_size > 2:
                effective_batch_size = max(2, math.ceil(effective_batch_size / 2))
            logger.warning(
                "Batch translation chunk returned no usable JSON output; falling back to single translation for this chunk. "
                "raw_response_snippet=%s",
                _sanitized_llm_output_snippet(raw),
            )
            continue
        for record in chunk_records:
            if record:
                record.setdefault("json_repair_fallback_status", []).append("batch_json_parsed")
        empty_chunk_items: list[dict] = []
        for item in parsed_items:
            region_id = _parsed_batch_item_id(item)
            translation = _parsed_batch_translation_value(item)
            if region_id:
                if translation:
                    translations[region_id] = translation
                else:
                    for chunk_item in chunk:
                        if isinstance(chunk_item, dict) and str(chunk_item.get("id") or "") == region_id:
                            empty_chunk_items.append(chunk_item)
                            break
        empty_ratio = len(empty_chunk_items) / max(1, len(chunk))
        split_reasons: list[str] = []
        if settings and settings.translator_backend == "GGUF":
            if call_latency > 20.0:
                split_reasons.append("batch_latency_gt_20s")
            if empty_ratio >= 0.5:
                split_reasons.append("empty_translation_ratio_ge_0_5")
            if empty_chunk_items and empty_ratio >= 0.5:
                split_reasons.append("compact_repair_required_for_most_chunk")
        for record in chunk_records:
            if record:
                record["batch_empty_translation_ratio"] = round(empty_ratio, 4)
                if split_reasons:
                    record["adaptive_batch_split_trigger"] = ",".join(split_reasons)
        if split_reasons and settings and settings.translator_backend == "GGUF" and effective_batch_size > 2:
            effective_batch_size = max(2, math.ceil(effective_batch_size / 2))
            for record in chunk_records:
                if record:
                    record["adaptive_batch_next_effective_size"] = effective_batch_size
        if empty_chunk_items:
            repair_group_size = 3 if settings and settings.translator_backend == "GGUF" else len(empty_chunk_items)
            repair_group_size = max(1, min(repair_group_size, len(empty_chunk_items)))
            repair_groups = [
                empty_chunk_items[index : index + repair_group_size]
                for index in range(0, len(empty_chunk_items), repair_group_size)
            ]
            all_empty_ids = [str(item.get("id") or "") for item in empty_chunk_items if isinstance(item, dict)]
            for group_index, repair_items in enumerate(repair_groups, start=1):
                empty_records = [
                    (debug_records_by_text or {}).get(str(item.get("id") or ""))
                    or (debug_records_by_text or {}).get(str(item.get("text") or ""))
                    for item in repair_items
                    if isinstance(item, dict)
                ]
                for record in empty_records:
                    if record:
                        record.setdefault("json_repair_fallback_status", []).append("batch_empty_translation_compact_repair")
                        _translation_perf_add_path(record, "batch_empty_repair")
                        record["compact_repair_group_size"] = len(repair_items)
                        record["compact_repair_group_count"] = len(repair_groups)
                        record["compact_repair_group_index"] = group_index
                repair_prompt = _build_compact_batch_retry_prompt(source_lang, target_lang, repair_items)
                repair_token_limit = max(48, min(128, _estimate_num_predict(repair_items)))
                if settings and settings.translator_backend == "GGUF":
                    repair_token_limit = min(repair_token_limit, 128 if target_lang == "Simplified Chinese" else 160)
                elif deepseek_backend:
                    repair_token_limit = max(repair_token_limit, 128)
                    repair_prompt = _build_compact_batch_retry_prompt(
                        source_lang,
                        target_lang,
                        repair_items,
                        json_object_wrapper=True,
                    )
                compact_request_id = _translation_perf_request_id(
                    "batch_empty_translation_compact_repair"
                )
                repair_start = time.time()
                try:
                    repair_options = {"num_predict": repair_token_limit, "temperature": temp, "top_p": top_p}
                    repair_timeout = 600
                    if deepseek_backend:
                        repair_options["response_format"] = {"type": "json_object"}
                        repair_options["thinking"] = {"type": "disabled"}
                        repair_timeout = _deepseek_request_timeout(600)
                    repair_raw = ollama.generate(
                        resolved,
                        repair_prompt,
                        timeout=repair_timeout,
                        options=repair_options,
                    )
                    repair_latency = time.time() - repair_start
                    for record in empty_records:
                        _translation_perf_record_llm_call(
                            record,
                            phase="batch_empty_translation_compact_repair",
                            prompt=repair_prompt,
                            latency_sec=repair_latency,
                            output=repair_raw,
                            token_limit=repair_token_limit,
                            shared_batch_size=len(repair_items),
                            request_id=compact_request_id,
                        )
                    repair_translations = _parse_compact_batch_retry_output(repair_raw, repair_items, target_lang)
                    translations.update(repair_translations)
                    accepted_ids = set(repair_translations.keys())
                    failed_ids = [
                        str(item.get("id") or "")
                        for item in repair_items
                        if isinstance(item, dict) and str(item.get("id") or "") not in accepted_ids
                    ]
                    for item in repair_items:
                        if not isinstance(item, dict):
                            continue
                        record = (
                            (debug_records_by_text or {}).get(str(item.get("id") or ""))
                            or (debug_records_by_text or {}).get(str(item.get("text") or ""))
                        )
                        if not record:
                            continue
                        record["compact_repair_accepted_count"] = len(accepted_ids)
                        record["compact_repair_failed_ids"] = failed_ids
                        record["compact_repair_all_empty_ids"] = all_empty_ids
                        if str(item.get("id") or "") in accepted_ids:
                            record.setdefault("json_repair_fallback_status", []).append(
                                "batch_empty_translation_compact_repair_accepted"
                            )
                        else:
                            record.setdefault("json_repair_fallback_status", []).append(
                                "batch_empty_translation_single_fallback"
                            )
                            record["compact_repair_single_fallback_count"] = len(failed_ids)
                            record.setdefault("failure_retry_reason", []).append("batch_empty_translation")
                except Exception as exc:
                    repair_latency = time.time() - repair_start
                    failed_ids = [str(item.get("id") or "") for item in repair_items if isinstance(item, dict)]
                    for record in empty_records:
                        _translation_perf_record_llm_call(
                            record,
                            phase="batch_empty_translation_compact_repair",
                            prompt=repair_prompt,
                            latency_sec=repair_latency,
                            output="",
                            token_limit=repair_token_limit,
                            status="exception",
                            shared_batch_size=len(repair_items),
                            error=f"{type(exc).__name__}: {exc}",
                            request_id=compact_request_id,
                        )
                        if record:
                            record.setdefault("json_repair_fallback_status", []).append(
                                "batch_empty_translation_single_fallback"
                            )
                            record["compact_repair_failed_ids"] = failed_ids
                            record["compact_repair_single_fallback_count"] = len(failed_ids)
                            record.setdefault("failure_retry_reason", []).append("batch_empty_translation_rebatch_exception")
    return translations


def _estimate_num_predict(items: list) -> int:
    if not items:
        return 128
    lengths = [len(str(item.get("text", ""))) for item in items if isinstance(item, dict)]
    total_len = sum(lengths)
    estimate = int(max(128, min(512, total_len * 3 + len(lengths) * 12)))
    return estimate


def _translate_single(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    style_guide: dict,
    text: str,
    context_lines: list[str] | None = None,
    settings: PipelineSettings | None = None,
    debug_record: dict[str, Any] | None = None,
) -> str:
    body = _non_punct_chars(text)
    short_reaction = target_lang == "Simplified Chinese" and _is_short_reaction_source(text)
    prompt_context = [] if short_reaction else (context_lines or [])
    _translation_perf_set_recent_context(debug_record, prompt_context)
    deterministic = _translate_short_reaction_fallback(text, target_lang) if short_reaction else ""
    if deterministic:
        _translation_perf_add_path(debug_record, "deterministic_short_reaction")
        _translation_perf_set_final(debug_record, translation=deterministic, status="deterministic_short_reaction")
        return deterministic
    if short_reaction:
        prompt = (
            f"将下面的{source_lang}短句翻译成自然的简体中文漫画对白，只输出短短的译文。\n"
            "不要结合上下文扩写，不要补充主语、称呼或说明。\n"
            f"原文：{text}"
        )
    else:
        prompt = build_translation_prompt(
            source_lang,
            target_lang,
            style_guide,
            prompt_context,
            text,
        )

    # Defaults
    temp = 0.2
    top_p = 0.9

    if settings:
        if settings.translator_backend == "GGUF":
             temp = settings.gguf_temperature
             top_p = settings.gguf_top_p
        else:
             temp = settings.ollama_temperature
             top_p = settings.ollama_top_p

    token_limit = _estimate_single_num_predict(text, target_lang)
    deepseek_backend = _is_deepseek_translation_backend(settings)
    single_timeout = _deepseek_request_timeout(300) if deepseek_backend else 300
    single_options = {"num_predict": token_limit, "temperature": temp, "top_p": top_p}
    if deepseek_backend:
        single_options["thinking"] = {"type": "disabled"}
    _translation_perf_add_path(debug_record, "single")
    call_start = time.time()
    result = ollama.generate(
        _resolve_model(model),
        prompt,
        timeout=single_timeout,
        options=single_options,
    )
    _translation_perf_record_llm_call(
        debug_record,
        phase="single_initial",
        prompt=prompt,
        latency_sec=time.time() - call_start,
        output=result,
        token_limit=token_limit,
    )
    cleaned = _clean_translation(result)
    cleaned = _normalize_translation_format_for_record(
        target_lang,
        cleaned,
        debug_record,
        stage="single_initial",
    )
    if short_reaction and _translation_has_bad_shape(cleaned, text):
        fallback = _translate_short_reaction_fallback(text, target_lang)
        if fallback:
            _translation_perf_add_path(debug_record, "deterministic_short_reaction_after_bad_shape")
            _translation_perf_set_final(debug_record, translation=fallback, status="deterministic_short_reaction_after_bad_shape")
            return fallback
    if (_translation_has_bad_shape(cleaned, text) or not cleaned) and text.strip():
        # Fallback: Force translation
        _translation_perf_add_path(debug_record, "single_retry")
        if debug_record:
            debug_record.setdefault("failure_retry_reason", []).append("single_bad_shape_or_empty")
        retry_text = _normalize_retry_source(text) or text
        retry_prompt = (
            f"Translate to {target_lang}. Translate exactly, do not skip. Output only the translation.\n"
            "Do not repeat a single character or syllable in a loop.\n"
            f"Text: {retry_text}"
        )
        retry_start = time.time()
        result = ollama.generate(
            _resolve_model(model),
            retry_prompt,
            timeout=single_timeout,
            options={
                **single_options,
                "temperature": min(temp, 0.1),
            },
        )
        _translation_perf_record_llm_call(
            debug_record,
            phase="single_retry_bad_shape_or_empty",
            prompt=retry_prompt,
            latency_sec=time.time() - retry_start,
            output=result,
            token_limit=token_limit,
        )
        cleaned = _clean_translation(result)
        cleaned = _normalize_translation_format_for_record(
            target_lang,
            cleaned,
            debug_record,
            stage="single_retry_bad_shape_or_empty",
        )
        if short_reaction and (_translation_has_bad_shape(cleaned, text) or not cleaned):
            fallback = _translate_short_reaction_fallback(text, target_lang)
            if fallback:
                _translation_perf_add_path(debug_record, "deterministic_short_reaction_after_retry")
                _translation_perf_set_final(debug_record, translation=fallback, status="deterministic_short_reaction_after_retry")
                return fallback
    _translation_perf_set_final(debug_record, translation=cleaned)
    return cleaned


def _ensure_target_language(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    ocr_text: str,
    translation: str,
    is_bubble: bool = False,
    debug_record: dict[str, Any] | None = None,
) -> tuple[str, bool]:
    retry_source = _normalize_retry_source(ocr_text) or ocr_text
    short_reaction = target_lang == "Simplified Chinese" and _is_short_reaction_source(ocr_text)
    deterministic = _translate_short_reaction_fallback(ocr_text, target_lang) if short_reaction else ""
    initial_assessment = _translation_postcheck_assessment(target_lang, translation, ocr_text)
    _translation_perf_record_pre_ensure(debug_record, initial_assessment)
    if short_reaction and deterministic:
        _translation_perf_add_path(debug_record, "ensure_deterministic_short_reaction")
        _translation_perf_set_final(debug_record, translation=deterministic, status="ensure_deterministic_short_reaction")
        _translation_perf_record_post_ensure(
            debug_record,
            deterministic,
            acceptance_status="deterministic_short_reaction",
            retry_skipped_reason="deterministic_short_reaction",
        )
        return deterministic, True
    if initial_assessment["merged_batch_output"]:
        _translation_perf_add_path(debug_record, "ensure_strict_merged_batch")
        if debug_record:
            debug_record.setdefault("failure_retry_reason", []).append("merged_batch_output")
        translation = _translate_strict(
            ollama,
            model,
            source_lang,
            target_lang,
            retry_source,
            debug_record=debug_record,
            debug_phase="ensure_strict_merged_batch",
        )
    elif initial_assessment["repetition_loop"]:
        _translation_perf_add_path(debug_record, "ensure_strict_repetition_loop")
        if debug_record:
            debug_record.setdefault("failure_retry_reason", []).append("repetition_loop")
        translation = _translate_strict(
            ollama,
            model,
            source_lang,
            target_lang,
            retry_source,
            debug_record=debug_record,
            debug_phase="ensure_strict_repetition_loop",
        )
    if _looks_like_prompt_leak(translation):
        _translation_perf_add_path(debug_record, "ensure_strict_prompt_leak")
        if debug_record:
            debug_record.setdefault("failure_retry_reason", []).append("prompt_leak")
        translation = _translate_strict(
            ollama,
            model,
            source_lang,
            target_lang,
            retry_source,
            debug_record=debug_record,
            debug_phase="ensure_strict_prompt_leak",
        )
    translation = _normalize_translation_format_for_record(
        target_lang,
        translation,
        debug_record,
        stage="pre_ensure_acceptance",
    )

    if short_reaction and deterministic:
        if not translation:
            _translation_perf_set_final(debug_record, translation=deterministic, status="ensure_short_reaction_empty")
            _translation_perf_record_post_ensure(
                debug_record,
                deterministic,
                acceptance_status="deterministic_short_reaction_empty",
                retry_skipped_reason="deterministic_short_reaction",
            )
            return deterministic, True
        if not _language_ok(target_lang, translation) or _translation_has_bad_shape(translation, ocr_text):
            _translation_perf_set_final(debug_record, translation=deterministic, status="ensure_short_reaction_bad_shape")
            _translation_perf_record_post_ensure(
                debug_record,
                deterministic,
                acceptance_status="deterministic_short_reaction_bad_shape",
                retry_skipped_reason="deterministic_short_reaction",
            )
            return deterministic, True

    # Only silence SFX/Empty if it's NOT a speech bubble.
    if not translation and TextFilter(None).should_ignore(ocr_text, "background_text") and not is_bubble:
        _translation_perf_record_post_ensure(
            debug_record,
            "",
            acceptance_status="empty_nonbubble_ignored",
            retry_skipped_reason="ignored_nonbubble_empty",
        )
        return "", True

    assessment = _translation_postcheck_assessment(target_lang, translation, ocr_text)
    if not assessment["hard_failure_reasons"]:
        soft = assessment["soft_warning_reasons"]
        status = "ensure_soft_warning_no_retry" if soft else "ensure_language_ok"
        if "quote_or_bracket_punctuation" in soft:
            acceptance_status = "accepted_with_unresolved_format_warning"
            retry_skipped_reason = "unresolved_format_warning_no_retry"
            if debug_record:
                debug_record["ensure_retry_unresolved_format_warning_reasons"] = list(soft)
        else:
            acceptance_status = "accepted_with_soft_warnings" if soft else "accepted_without_retry"
            retry_skipped_reason = "soft_warning_no_retry" if soft else "no_retry_needed"
        _translation_perf_set_final(debug_record, translation=translation, status=status)
        _translation_perf_record_post_ensure(
            debug_record,
            translation,
            acceptance_status=acceptance_status,
            retry_skipped_reason=retry_skipped_reason,
        )
        _translation_perf_set_final(debug_record, translation=translation, status="ensure_language_ok")
        return translation, True

    # Build retry prompt - be explicit about language requirements
    if target_lang == "Simplified Chinese":
        retry_prompt = (
            f"将下面的日语翻译成简体中文。\n"
            f"只输出简体中文译文，不要片假名、平假名、罗马音或英文。\n"
            "不要把同一个字重复很多次。\n"
            f"日语原文: {retry_source}\n"
        )
    else:
        retry_prompt = (
            f"Translate {source_lang} to {target_lang}.\n"
            "No English, no romaji, no explanations.\n"
            "Do not repeat a single character or syllable in a loop.\n"
            f"Text: {retry_source}\n"
        )
    _translation_perf_add_path(debug_record, "ensure_language_retry")
    if debug_record:
        for reason in assessment["hard_failure_reasons"]:
            debug_record.setdefault("failure_retry_reason", []).append(reason)
    retry_token_limit = _estimate_single_num_predict(retry_source, target_lang)
    retry_start = time.time()
    retry_raw = ollama.generate(
        model,
        retry_prompt,
        timeout=30,
        options={"num_predict": retry_token_limit, "temperature": 0.1, "top_p": 0.9},
    )
    _translation_perf_record_llm_call(
        debug_record,
        phase="ensure_language_retry",
        prompt=retry_prompt,
        latency_sec=time.time() - retry_start,
        output=retry_raw,
        token_limit=retry_token_limit,
    )
    retry = _clean_translation(retry_raw)
    retry = _normalize_translation_format_for_record(
        target_lang,
        retry,
        debug_record,
        stage="ensure_language_retry",
    )
    retry_assessment = _translation_postcheck_assessment(target_lang, retry, ocr_text)
    if "repetition_loop" in retry_assessment["hard_failure_reasons"] or "prompt_leak" in retry_assessment["hard_failure_reasons"]:
        _translation_perf_add_path(debug_record, "ensure_strict_after_retry_bad_shape")
        if debug_record:
            debug_record.setdefault("failure_retry_reason", []).append("ensure_retry_bad_shape")
        retry = _translate_strict(
            ollama,
            model,
            source_lang,
            target_lang,
            retry_source,
            debug_record=debug_record,
            debug_phase="ensure_strict_after_retry_bad_shape",
        )
        retry_assessment = _translation_postcheck_assessment(target_lang, retry, ocr_text)
    if not retry_assessment["hard_failure_reasons"]:
        _translation_perf_set_final(debug_record, translation=retry, status="ensure_retry_language_ok")
        _translation_perf_record_post_ensure(
            debug_record,
            retry,
            acceptance_status="retry_accepted",
        )
        return retry, True

    # Second retry for Chinese if still has Kana - be even more explicit
    if target_lang == "Simplified Chinese" and "meaningful_japanese_kana" in retry_assessment["hard_failure_reasons"]:
        final_prompt = (
            f"请将日语'{retry_source}'翻译成中文。\n"
            f"重要：你的回答中绝对不能包含日语假名（ひらがな/カタカナ）。\n"
            f"只能使用纯中文汉字进行翻译。\n"
            "不要把同一个字重复很多次。\n"
        )
        _translation_perf_add_path(debug_record, "ensure_final_kana_retry")
        if debug_record:
            debug_record.setdefault("failure_retry_reason", []).append("ensure_retry_still_contains_kana")
        final_token_limit = _estimate_single_num_predict(retry_source, target_lang)
        final_start = time.time()
        final_raw = ollama.generate(
            model,
            final_prompt,
            timeout=30,
            options={"num_predict": final_token_limit, "temperature": 0.05, "top_p": 0.9},
        )
        _translation_perf_record_llm_call(
            debug_record,
            phase="ensure_final_kana_retry",
            prompt=final_prompt,
            latency_sec=time.time() - final_start,
            output=final_raw,
            token_limit=final_token_limit,
        )
        final = _clean_translation(final_raw)
        final = _normalize_translation_format_for_record(
            target_lang,
            final,
            debug_record,
            stage="ensure_final_kana_retry",
        )
        final_assessment = _translation_postcheck_assessment(target_lang, final, ocr_text)
        if not final_assessment["hard_failure_reasons"]:
            _translation_perf_set_final(debug_record, translation=final, status="ensure_final_retry_language_ok")
            _translation_perf_record_post_ensure(
                debug_record,
                final,
                acceptance_status="final_retry_accepted",
            )
            return final, True
        retry = final if final else retry
        retry_assessment = final_assessment if final else retry_assessment

    if retry_assessment["hard_failure_reasons"]:
        flagged_translation = retry or translation
        if flagged_translation:
            _translation_perf_set_final(
                debug_record,
                translation=flagged_translation,
                status="ensure_failed_quality_flagged_output_preserved",
            )
            _translation_perf_record_post_ensure(
                debug_record,
                flagged_translation,
                acceptance_status="retry_failed_quality_flagged_output_preserved",
                retry_skipped_reason="postcheck_quality_failure_advisory",
            )
            return flagged_translation, False
        _translation_perf_set_final(debug_record, translation="", status="ensure_failed_bad_shape")
        _translation_perf_record_post_ensure(
            debug_record,
            "",
            acceptance_status="retry_failed_hard_failure",
        )
        return "", False
    _translation_perf_set_final(debug_record, translation=retry or translation, status="ensure_returned_unverified")
    _translation_perf_record_post_ensure(
        debug_record,
        retry or translation,
        acceptance_status="returned_without_hard_failure",
    )
    return retry or translation, False


def _too_long_translation(translation: str, ocr_text: str) -> bool:
    if not translation or not ocr_text:
        return False
    if "\n" in translation:
        return True
    t_len = len(translation)
    o_len = len(ocr_text)
    if o_len <= 4:
        return t_len > max(12, o_len * 3)
    return t_len > o_len * 2.2


def _looks_like_merged_batch_output(translation: str, ocr_text: str) -> bool:
    if not translation:
        return False
    lines = [line.strip() for line in str(translation).splitlines() if line.strip()]
    if len(lines) >= 2:
        return True
    src_punct = sum(1 for ch in ocr_text if ch in "。！？!?…")
    dst_punct = sum(1 for ch in translation if ch in "。！？!?…")
    if dst_punct >= max(3, src_punct + 3) and len(translation) > max(24, len(ocr_text) * 1.6):
        return True
    return False


def _language_ok(target_lang: str, text: str) -> bool:
    if not text:
        return False
    if target_lang == "Simplified Chinese":
        return _cjk_ratio(text) >= 0.3 and _kana_ratio(text) <= 0.1
    if target_lang == "English":
        return _cjk_ratio(text) < 0.2
    return True


def _looks_like_prompt_leak(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    markers = [
        "return only",
        "output only",
        "output only the translation",
        "no labels",
        "no quotes",
        "no explanations",
        "text to translate",
        "<<text>>",
        "<</text>>",
        "translation:",
    ]
    chinese_markers = [
        "只需翻译",
        "仅需翻译",
        "只翻译",
        "不要任何",
        "不要标签",
        "不要引号",
        "不要解释",
        "不要多余",
        "不要说明",
        "不要注释",
        "上下文",
        "译文",
        "只输出",
        "输出译文",
        "只输出翻译",
        "翻译如下",
        "文字：",
        "文本：",
        "原文：",
        "翻译：",
        "不要英文",
        "不要罗马音",
        "不要羅馬音",
        # Additional patterns seen in user-reported prompt leaks
        "不要用英语",
        "不要用罗马音",
        "不要加解释",
        "没有英语",
        "没有罗马音",
        "没有解释",
        "必须原样保留这些占位符",
        "原样保留这些占位符",
        "不要翻译、不要删除、不要新增",
        "不要删除、不要新增",
        "占位符",
        "标记：",
    ]
    if any(m in lowered for m in markers):
        return True
    for marker in chinese_markers:
        if marker in text:
            return True
    return False


def _translate_strict(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    text: str,
    debug_record: dict[str, Any] | None = None,
    debug_phase: str = "strict",
) -> str:
    if target_lang == "Simplified Chinese":
        prompt = f"将以下{source_lang}翻译成简体中文，不要把同一个字重复很多次：{text}"
    elif target_lang == "English":
        prompt = f"Translate the following {source_lang} into English: {text}"
    else:
        prompt = f"Translate the following {source_lang} into {target_lang}: {text}"
    token_limit = _estimate_single_num_predict(text, target_lang)
    call_start = time.time()
    result = ollama.generate(
        _resolve_model(model),
        prompt,
        timeout=180,
        options={"num_predict": token_limit, "temperature": 0.1, "top_p": 0.9},
    )
    _translation_perf_record_llm_call(
        debug_record,
        phase=debug_phase,
        prompt=prompt,
        latency_sec=time.time() - call_start,
        output=result,
        token_limit=token_limit,
    )
    return _clean_translation(result)


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    body = _non_punct_chars(text)
    denominator = len(body) if body else len(text)
    cjk = sum(1 for ch in body if _is_japanese(ch)) if body else sum(1 for ch in text if _is_japanese(ch))
    return cjk / max(1, denominator)


def _kana_ratio(text: str) -> float:
    if not text:
        return 0.0
    kana = sum(1 for ch in text if _is_kana(ch))
    return kana / max(1, len(text))




def _should_skip_text(text: str, bbox: list, image_size: tuple[int, int]) -> bool:
    if not text:
        return True
    if _is_punct_only(text):
        return True
    if _placeholder_ratio(text) >= 0.15:
        return True

    # CRITICAL FIX: If text is strongly valid Japanese, NEVER skip it
    # This ensures short dialogue like "フ…", "そ", "え?" are always translated
    if _is_valid_japanese(text) >= 0.6:
        return False

    x, y, w, h = bbox
    area = w * h
    img_w, img_h = image_size
    page_area = img_w * img_h if img_w and img_h else 1
    ratio = area / page_area
    length = len(text)
    jp_ratio = _japanese_ratio(text)
    if length <= 2 and ratio < 0.003:
        # Check aspect ratio for very small boxes
        aspect = w / h if h else 0

        # FIX: "そ" and vertical text (tall/narrow) are often skipped by current aspect ratio check (0.3 < aspect < 3.5)
        # If it's strongly Japanese, KEEP IT regardless of aspect ratio
        if _is_valid_japanese(text) >= 0.5:
             return False

        if jp_ratio >= 0.6 and 0.3 < aspect < 3.5:
            return False
        return True
    if jp_ratio < 0.3 and length < 6:
        return True
    if jp_ratio < 0.2 and ratio < 0.006:
        return True
    return False


def _should_ignore_speech_fragment(
    text: str,
    bbox: list,
    image_size: tuple[int, int],
    ocr_conf: float,
) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return True
    if _is_punct_only(cleaned):
        if _is_ellipsis_like(cleaned):
            return False
        return True
    if _placeholder_ratio(cleaned) >= 0.2:
        return True
    img_w, img_h = image_size
    page_area = max(1, img_w * img_h)
    _, _, w, h = bbox
    area_ratio = (max(1, w) * max(1, h)) / page_area
    kana_only = all(_is_kana(ch) or ch in {"ー", "・"} for ch in cleaned)
    narrow_box = min(max(1, w), max(1, h)) <= 42
    has_japanese = any(_is_kana(ch) or 0x4E00 <= ord(ch) <= 0x9FFF for ch in cleaned)
    if kana_only and _is_short_reaction_source(cleaned):
        return False
    if not has_japanese:
        if len(cleaned) <= 6 and area_ratio < 0.003:
            return True
        if narrow_box and re.fullmatch(r"[A-Za-z0-9+\-_.:/…]+", cleaned):
            return True
    if len(cleaned) == 1:
        if kana_only and area_ratio < 0.0035 and ocr_conf < 0.985:
            return True
        if cleaned in {"っ", "ッ", "ー", "・"}:
            return True
    if len(cleaned) == 2 and kana_only and area_ratio < 0.0025 and ocr_conf < 0.96:
        return True
    if len(cleaned) <= 2 and kana_only and not _is_short_reaction_source(cleaned) and ocr_conf < 0.96:
        return True
    if len(cleaned) == 3 and kana_only and narrow_box and area_ratio < 0.0015 and ocr_conf < 0.93:
        return True
    if len(cleaned) <= 3 and kana_only and narrow_box and area_ratio < 0.0009 and ocr_conf < 0.985:
        return True
    return False


def _should_single_translate_text(
    text: str,
    region_ids: list[str],
    regions: list[dict],
) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    matched = [r for r in regions if r.get("region_id") in region_ids]
    if not matched:
        return False
    if _is_punct_only(cleaned) or _is_ellipsis_like(cleaned):
        return True
    if _is_short_reaction_source(cleaned):
        return True

    semantic_len = len(_non_punct_chars(cleaned))
    region_types = {
        str(region.get("type", "") or "").strip().lower()
        for region in matched
    }
    speech_like = "speech_bubble" in region_types or any(
        _looks_like_recoverable_speech_region(region) for region in matched
    )
    has_background = "background_text" in region_types
    has_decorative = "decorative_text" in region_types or "sfx" in region_types

    if semantic_len <= 2:
        return True
    if speech_like and semantic_len <= 4:
        return True
    if has_decorative:
        return True
    if has_background and semantic_len <= 4:
        return True

    return False


def _deepseek_short_batch_context_lane(
    text: str,
    region_ids: list[str],
    regions: list[dict],
    *,
    target_lang: str,
    settings: PipelineSettings | None,
) -> bool | None:
    """Select a safe DeepSeek batch lane for an existing short single unit.

    The function does not rewrite, mask, split, or normalize source text.  It
    only consolidates LLM requests when the existing single-unit policy would
    otherwise call DeepSeek and when all units in the request share the same
    existing context policy.  Punctuation, deterministic reactions, and
    decorative/SFX material retain their established individual path.
    """

    if not _is_deepseek_translation_backend(settings):
        return None
    if not _should_single_translate_text(text, region_ids, regions):
        return None
    cleaned = str(text or "").strip()
    if not cleaned or _is_punct_only(cleaned) or _is_ellipsis_like(cleaned):
        return None
    if len(_non_punct_chars(cleaned)) < 3 or _is_short_reaction_source(cleaned):
        return None
    if _translate_short_reaction_fallback(cleaned, target_lang):
        return None
    matched = [region for region in regions if region.get("region_id") in region_ids]
    if not matched or any(
        str(region.get("type") or "").strip().lower()
        in {"decorative_text", "sfx", "sign"}
        or _region_is_sfx_or_decorative_preserve(region)
        for region in matched
    ):
        return None
    return bool(_should_use_context_for_text(cleaned, region_ids, regions))


def _classify_semantic_region(
    text: str,
    bbox: list,
    image_size: tuple[int, int],
    det_conf: float,
    ocr_conf: float,
    image_obj,
    text_filter: TextFilter,
    initial_bg: bool = False,
    text_area_assignment: dict | None = None,
) -> tuple[str, bool, bool, bool, dict]:
    cleaned = str(text or "").strip()
    region_type = "background_text" if initial_bg else "speech_bubble"
    bg_text = bool(initial_bg)
    needs_review = det_conf < 0.6
    render_updates: dict[str, object] = {"cleanup_mode": "bubble" if not initial_bg else "local_text_mask"}

    if _is_text_area_translatable_assignment(text_area_assignment):
        route = str(text_area_assignment.get("text_area_route_intent") or "").strip()
        state, reason = _ocr_transaction_state_for_text_area_route(cleaned, ocr_conf, route)
        render_updates = {
            "cleanup_mode": "local_text_mask" if route == "translate_caption_background" else "bubble",
            "classification_reason": (
                "text_area_route_authority_caption_background"
                if route == "translate_caption_background"
                else "text_area_route_authority_speech"
            ),
            "text_area_ocr_transaction_state": state,
            "text_area_ocr_warning_reason": reason if state == _OCR_LOW_CONFIDENCE_WARNING_STATE else "",
            "text_area_ocr_blocker_reason": "" if _ocr_transaction_state_queues_translation(state) else reason,
        }
        return (
            "background_text" if route == "translate_caption_background" else "speech_bubble",
            route == "translate_caption_background",
            False,
            needs_review or state != _OCR_TRANSLATION_READY_STATE,
            render_updates,
        )

    if isinstance(text_area_assignment, dict):
        authorization = str(
            text_area_assignment.get("text_area_semantic_authorization_state")
            or text_area_assignment.get("text_area_cleanup_authorization")
            or ""
        ).strip()
        container_type = str(text_area_assignment.get("text_area_container_type") or "").strip()
        if authorization in {"protect_sfx_decorative", "protect_art_or_non_text"}:
            semantic_type = "sfx" if container_type == "sfx_decorative_art" else "decorative_text"
            return semantic_type, True, True, False, {
                "cleanup_mode": "preserve",
                "classification_reason": "text_area_plan_protected_nonworkflow_authority",
            }
        if authorization in {
            "review_unknown_not_cleanup",
            "outside_cleanup_scope",
            "ambiguous_component_owner",
        }:
            return "unknown", True, True, True, {
                "cleanup_mode": "preserve",
                "classification_reason": "text_area_plan_nonworkflow_review_authority",
            }

    return "unknown", True, True, True, {
        "cleanup_mode": "preserve",
        "classification_reason": "missing_text_area_semantic_authority",
    }

    if not cleaned:
        return region_type, bg_text, True, True, render_updates
    stats = _box_luma_stats_pil(image_obj, bbox)
    _, _, w, h = bbox
    aspect = w / max(1, h)
    thin_strip = h <= 28 and aspect >= 3.0
    tall_narrow = h >= max(110, w * 2.2)
    slim_vertical = h >= 70 and w <= 32 and h >= max(70, w * 1.8)
    topish = (bbox[1] + (h / 2.0)) <= image_size[1] * 0.28
    page_area = max(1, image_size[0] * image_size[1])
    area_ratio = (max(1, w) * max(1, h)) / page_area
    katakana_ratio = _katakana_ratio_text(cleaned)
    contains_kanji = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in cleaned)
    contains_kana = any(_is_kana(ch) for ch in cleaned)
    mixed_scripts = _has_mixed_scripts(cleaned)
    has_latin = _has_latin_text(cleaned)
    body = _non_punct_chars(cleaned)
    stats_mean = float(stats[0]) if stats else None
    meaningful_caption_source = _is_meaningful_background_caption_source(cleaned)
    probable_short_vertical_dialogue = _is_probable_short_vertical_dialogue_box(
        cleaned,
        bbox,
        stats_mean=stats_mean,
        image_size=image_size,
    )

    if _looks_like_decorative_title_artifact(
        cleaned,
        bbox,
        image_size,
        det_conf,
        ocr_conf,
        mixed_scripts,
        has_latin,
    ):
        return "decorative_text", True, True, False, {
            "cleanup_mode": "preserve",
            "classification_reason": "low_conf_dark_short_art_sfx_candidate",
        }

    if _is_bubble_contained_short_laugh_speech_candidate(
        cleaned,
        bbox,
        image_size,
        det_conf,
        ocr_conf,
        image_obj,
        stats_mean,
    ):
        return "speech_bubble", False, False, False, {
            "cleanup_mode": "bubble",
            "classification_reason": _BUBBLE_CONTAINED_SHORT_LAUGH_SPEECH_REASON,
        }

    nonbubble_breath_sfx_art = _nonbubble_breath_sfx_art_text_reason(
        cleaned,
        bbox,
        image_size,
        det_conf,
        ocr_conf,
        image_obj,
        stats_mean,
    )
    if nonbubble_breath_sfx_art:
        return "decorative_text", True, True, False, {
            "cleanup_mode": "preserve",
            "classification_reason": nonbubble_breath_sfx_art,
        }

    nonbubble_short_reaction_art = _nonbubble_short_reaction_art_text_reason(
        cleaned,
        bbox,
        image_size,
        det_conf,
        ocr_conf,
        image_obj,
        stats_mean,
    )
    if not bg_text and nonbubble_short_reaction_art:
        return "decorative_text", True, True, False, {
            "cleanup_mode": "preserve",
            "classification_reason": nonbubble_short_reaction_art,
        }

    if (
        not bg_text
        and stats_mean is not None
        and stats_mean < 170.0
        and area_ratio >= 0.0035
        and len(body) <= 6
        and ocr_conf < 0.80
        and det_conf < 0.88
        and (not meaningful_caption_source or ocr_conf < 0.55)
        and not any(ch.isdigit() for ch in cleaned)
        and (
            ocr_conf < 0.50
            or (
                len(body) <= 1
                and not probable_short_vertical_dialogue
            )
        )
    ):
        return "decorative_text", True, True, False, {
            "cleanup_mode": "preserve",
            "classification_reason": "low_conf_dark_short_art_sfx_candidate",
        }

    if (
        not bg_text
        and stats_mean is not None
        and stats_mean < 205.0
        and area_ratio >= 0.02
        and len(body) <= 6
    ):
        return "decorative_text", True, True, False, {
            "cleanup_mode": "preserve",
            "classification_reason": "large_short_decorative_sfx_candidate",
        }

    if (
        not bg_text
        and stats_mean is not None
        and stats_mean < 215.0
        and katakana_ratio >= 0.6
        and len(body) <= 4
        and area_ratio <= 0.012
    ):
        return "decorative_text", True, True, False, {"cleanup_mode": "preserve"}

    if _is_dark_caption_box(stats, cleaned):
        region_type = "narration_box"
        bg_text = True
        render_updates = {
            "cleanup_mode": "caption_box",
        }

    if (
        bg_text
        and not meaningful_caption_source
        and det_conf <= 0.65
        and ocr_conf <= 0.72
        and stats_mean is not None
        and stats_mean < 210.0
        and area_ratio >= 0.020
        and len(body) <= 6
        and not contains_kanji
        and not has_latin
        and not any(ch.isdigit() for ch in cleaned)
        and (
            max(1, h) >= max(1, w) * 1.35
            or min(max(1, w), max(1, h)) >= 120
        )
    ):
        return "decorative_text", True, True, False, {
            "cleanup_mode": "preserve",
            "classification_reason": _LARGE_LOW_CONFIDENCE_NONBUBBLE_SFX_REASON,
        }

    if (
        bg_text
        and probable_short_vertical_dialogue
        and not topish
    ):
        return "speech_bubble", False, False, needs_review, {"cleanup_mode": "bubble"}

    if (
        bg_text
        and tall_narrow
        and not thin_strip
        and len(body) <= 8
        and area_ratio <= 0.015
        and not has_latin
        and stats_mean is not None
        and stats_mean >= 220.0
        and (contains_kana or contains_kanji)
    ):
        region_type = "speech_bubble"
        bg_text = False
        render_updates = {"cleanup_mode": "bubble"}
    elif (
        bg_text
        and tall_narrow
        and not thin_strip
        and len(body) <= 4
        and contains_kana
        and not contains_kanji
        and area_ratio <= 0.004
        and stats_mean is not None
        and stats_mean >= 180.0
    ):
        region_type = "speech_bubble"
        bg_text = False
        render_updates = {"cleanup_mode": "bubble"}

    if (
        not bg_text
        and contains_kanji
        and contains_kana
        and len(body) <= 4
        and not probable_short_vertical_dialogue
        and ocr_conf < 0.78
        and det_conf < 0.85
        and area_ratio <= 0.006
        and stats_mean is not None
        and stats_mean < 215.0
        and not _is_short_reaction_source(cleaned)
    ):
        return "decorative_text", True, True, False, {"cleanup_mode": "preserve"}

    if (
        not bg_text
        and tall_narrow
        and contains_kanji
        and contains_kana
        and len(body) <= 4
        and not probable_short_vertical_dialogue
        and ocr_conf < 0.75
        and det_conf < 0.8
        and area_ratio <= 0.015
        and not _is_short_reaction_source(cleaned)
    ):
        return "decorative_text", True, True, False, {"cleanup_mode": "preserve"}

    if (
        not bg_text
        and contains_kana
        and not contains_kanji
        and len(body) <= 8
        and not probable_short_vertical_dialogue
        and ocr_conf < 0.55
        and det_conf < 0.7
        and area_ratio <= 0.02
        and stats_mean is not None
        and stats_mean < 215.0
        and not _is_short_reaction_source(cleaned)
        and not any(ch in cleaned for ch in "。！？!?…")
    ):
        return "decorative_text", True, True, False, {"cleanup_mode": "preserve"}

    if (
        not bg_text
        and topish
        and (tall_narrow or slim_vertical)
        and len(body) <= 4
        and det_conf < 0.8
        and area_ratio <= 0.0045
        and any(ch.isdigit() for ch in cleaned)
    ):
        return "background_text", True, False, needs_review, {"cleanup_mode": "local_text_mask"}

    if thin_strip and not bg_text:
        region_type = "background_text"
        bg_text = True
        render_updates = {"cleanup_mode": "local_text_mask"}

    # Decorative vertical page furniture near the top of the page is a major false-positive
    # source on contents / splash / narrative montage pages. Route these away from speech
    # bubble handling unless the crop is obviously bubble-like (bright, uniform interior).
    if (
        not bg_text
        and topish
        and tall_narrow
        and len(body) <= 18
        and area_ratio <= 0.02
        and stats_mean is not None
        and stats_mean < 205.0
    ):
        if meaningful_caption_source and ocr_conf >= 0.90:
            return "background_text", True, False, needs_review, {
                "cleanup_mode": "local_text_mask",
                "classification_reason": _TOP_ROW_BACKGROUND_CAPTION_REASON,
            }
        region_type = "background_text"
        bg_text = True
        render_updates = {"cleanup_mode": "preserve"}

    if bg_text:
        if meaningful_caption_source:
            render_updates = {"cleanup_mode": "local_text_mask"}
            if topish and tall_narrow:
                render_updates["classification_reason"] = _TOP_ROW_BACKGROUND_CAPTION_REASON
            return "background_text", bg_text, False, needs_review, render_updates
        kana_only = bool(body) and all(_is_kana(ch) for ch in body)
        if (
            len(body) <= 4
            and contains_kanji
            and contains_kana
            and not any(ch.isdigit() for ch in cleaned)
            and area_ratio <= 0.008
            and stats_mean is not None
            and stats_mean < 225.0
        ):
            render_updates = {"cleanup_mode": "preserve"}
            if topish and tall_narrow:
                render_updates["classification_reason"] = _TOP_ROW_CAPTION_FRAGMENT_REASON
            return "decorative_text", bg_text, True, False, render_updates
        if (
            len(body) <= 4
            and tall_narrow
            and not thin_strip
            and area_ratio <= 0.006
            and (any(_is_kana(ch) for ch in body) or _is_ellipsis_like(cleaned))
            and stats_mean is not None
            and stats_mean >= 165.0
        ):
            return "speech_bubble", False, False, needs_review, {"cleanup_mode": "bubble"}
        if (
            _is_ellipsis_like(cleaned)
            and tall_narrow
            and not thin_strip
            and area_ratio <= 0.012
        ):
            return "speech_bubble", False, False, needs_review, {"cleanup_mode": "bubble"}
        if len(body) <= 4 and not contains_kanji:
            return "decorative_text", bg_text, True, False, {"cleanup_mode": "preserve"}
        if len(body) <= 5 and area_ratio < 0.0035 and (ocr_conf < 0.997 or det_conf < 0.75):
            return "decorative_text", bg_text, True, False, {"cleanup_mode": "preserve"}
        if len(body) <= 2 and kana_only and ocr_conf < 0.999:
            return "decorative_text", bg_text, True, False, {"cleanup_mode": "preserve"}
        if len(body) <= 4 and area_ratio >= 0.006:
            ignore_type = "sfx" if katakana_ratio >= 0.45 or len(cleaned) <= 4 else "decorative_text"
            return ignore_type, bg_text, True, False, {"cleanup_mode": "preserve"}
        if area_ratio >= 0.018 and len(body) <= 6:
            ignore_type = "sfx" if katakana_ratio >= 0.45 or len(cleaned) <= 4 else "decorative_text"
            return ignore_type, bg_text, True, False, {"cleanup_mode": "preserve"}
        if (
            topish
            and tall_narrow
            and len(body) <= 18
            and area_ratio <= 0.03
            and stats_mean is not None
            and stats_mean < 210.0
        ):
            if contains_kanji and len(body) >= 4 and ocr_conf >= 0.95:
                return "background_text", bg_text, False, needs_review, {"cleanup_mode": "local_text_mask"}
            return "decorative_text", bg_text, True, False, {"cleanup_mode": "preserve"}
        if thin_strip and area_ratio < 0.02 and len(body) <= 10:
            if mixed_scripts or has_latin or ocr_conf < 0.985 or det_conf < 0.9:
                return "decorative_text", bg_text, True, False, {"cleanup_mode": "preserve"}
        if len(body) <= 3 and not contains_kanji and any(ch in cleaned for ch in "「」『』（）()") and ocr_conf < 0.995:
            return "decorative_text", bg_text, True, False, {"cleanup_mode": "preserve"}
        if len(body) <= 2 and area_ratio < 0.003 and ocr_conf < 0.995 and det_conf < 0.95:
            return "decorative_text", bg_text, True, False, {"cleanup_mode": "preserve"}
        if len(body) <= 4 and area_ratio < 0.002 and mixed_scripts and ocr_conf < 0.985:
            return "decorative_text", bg_text, True, False, {"cleanup_mode": "preserve"}
        if len(body) <= 5 and area_ratio < 0.0045 and not contains_kanji and (ocr_conf < 0.99 or det_conf < 0.92):
            ignore_type = "sfx" if katakana_ratio >= 0.45 or len(cleaned) <= 4 else "decorative_text"
            return ignore_type, bg_text, True, False, {"cleanup_mode": "preserve"}
        if text_filter.should_ignore(cleaned, "background_text"):
            if not contains_kanji or katakana_ratio >= 0.6 or len(cleaned) <= 6:
                ignore_type = "sfx" if katakana_ratio >= 0.45 or len(cleaned) <= 4 else "decorative_text"
                return ignore_type, bg_text, True, False, {"cleanup_mode": "preserve"}
        if _looks_like_background_artifact(cleaned, bbox, image_size, det_conf, ocr_conf, mixed_scripts):
            return "decorative_text", bg_text, True, False, {"cleanup_mode": "preserve"}
        return region_type, bg_text, False, needs_review, render_updates

    if (
        not bg_text
        and _nonbubble_short_kana_art_text_reason(
            cleaned,
            bbox,
            image_size,
            det_conf,
            ocr_conf,
            image_obj,
            stats_mean=stats_mean,
        )
    ):
        return "decorative_text", True, True, False, {
            "cleanup_mode": "preserve",
            "classification_reason": _NONBUBBLE_SHORT_KANA_ART_TEXT_REASON,
        }

    if text_filter.should_ignore(cleaned, "speech_bubble") and _likely_sfx_effect_box(
        cleaned, bbox, image_size, ocr_conf
    ):
        return "sfx", bg_text, True, False, {"cleanup_mode": "preserve"}

    return region_type, bg_text, False, needs_review, render_updates


def _cover_page_saturation(image_obj) -> float:
    if image_obj is None:
        return 0.0
    try:
        hsv = image_obj.convert("HSV")
        from PIL import ImageStat

        stats = ImageStat.Stat(hsv)
        if not stats.mean or len(stats.mean) < 2:
            return 0.0
        return float(stats.mean[1])
    except Exception:
        return 0.0


def _looks_like_decorative_cover_page(regions: list[dict], image_obj) -> bool:
    active = [r for r in regions if not r.get("ignore")]
    if not active or len(active) > 8:
        return False
    if _cover_page_saturation(image_obj) < 28.0:
        return False
    texts = [str(r.get("ocr_text", "")).strip() for r in active]
    if not texts:
        return False
    sentence_like = 0
    mixed_or_latin = 0
    shortish = 0
    tall_title_like = 0
    page_area = max(1, image_obj.size[0] * image_obj.size[1]) if image_obj is not None else 1
    total_area_ratio = 0.0
    for region, text in zip(active, texts):
        body = _non_punct_chars(text)
        if any(ch in text for ch in "。！？!?") or len(body) >= 10:
            sentence_like += 1
        if _has_latin_text(text) or _has_mixed_scripts(text):
            mixed_or_latin += 1
        if len(body) <= 6:
            shortish += 1
        x, y, w, h = region.get("bbox", [0, 0, 0, 0])
        total_area_ratio += (max(1, int(w)) * max(1, int(h))) / page_area
        if max(1, int(h)) > max(1, int(w)) * 2.2 and len(body) <= 8:
            tall_title_like += 1
    if sentence_like > 0:
        return False
    if shortish < max(3, len(active) - 1):
        return False
    if total_area_ratio > 0.18:
        return False
    if mixed_or_latin > 0:
        return True
    return tall_title_like >= 2


def _looks_like_contents_page(regions: list[dict], image_obj) -> bool:
    active = [r for r in regions if not r.get("ignore")]
    if len(active) < 6 or image_obj is None:
        return False
    thin_rows = 0
    numeric_rows = 0
    marker_rows = 0
    wide_rows = 0
    tall_bubbles = 0
    page_area = max(1, image_obj.size[0] * image_obj.size[1])
    total_area_ratio = 0.0
    for region in active:
        text = str(region.get("ocr_text", "")).strip()
        body = _non_punct_chars(text)
        x, y, w, h = region.get("bbox", [0, 0, 0, 0])
        w = max(1, int(w))
        h = max(1, int(h))
        total_area_ratio += (w * h) / page_area
        if h <= 180 and w >= h * 2.0:
            thin_rows += 1
        if w >= h * 3.5:
            wide_rows += 1
        if h >= w * 1.5:
            tall_bubbles += 1
        if any(ch.isdigit() for ch in text):
            numeric_rows += 1
        if any(marker in text for marker in ("第", "話", "话", "CONTENTS", "目次")):
            marker_rows += 1
        if any(ch in text for ch in "。！？!?") and len(body) >= 16:
            return False
    if total_area_ratio > 0.35:
        return False
    if tall_bubbles > max(2, len(active) // 3):
        return False
    return (thin_rows >= 6 or wide_rows >= 5) and (numeric_rows >= 4 or marker_rows >= 2)


def _looks_like_chapter_title_page(regions: list[dict], image_obj) -> bool:
    active = [r for r in regions if not r.get("ignore")]
    if not active or len(active) > 3 or image_obj is None:
        return False
    page_w, page_h = image_obj.size
    page_area = max(1, page_w * page_h)
    total_area_ratio = 0.0
    wide_strips = 0
    title_markers = 0
    for region in active:
        text = str(region.get("ocr_text", "")).strip()
        body = _non_punct_chars(text)
        x, y, w, h = region.get("bbox", [0, 0, 0, 0])
        w = max(1, int(w))
        h = max(1, int(h))
        total_area_ratio += (w * h) / page_area
        bottomish = (y + h) >= page_h * 0.60
        if w >= page_w * 0.35 and h <= page_h * 0.15 and bottomish:
            wide_strips += 1
        if any(marker in text for marker in ("第", "話", "话", "章", "編", "篇")):
            title_markers += 1
        if any(ch in text for ch in "。！？!?") and len(body) >= 10:
            return False
    if total_area_ratio > 0.12:
        return False
    return title_markers > 0 or wide_strips > 0


def _should_preserve_region_on_page_class(
    region: dict,
    page_class: str,
    image_size: tuple[int, int] | None,
) -> bool:
    page_class = str(page_class or "").strip().lower()
    if page_class in {"cover", "contents"}:
        return True
    if page_class != "chapter_title":
        return False
    text = str(region.get("ocr_text", "") or "").strip()
    body = _non_punct_chars(text)
    bbox = region.get("bbox", [0, 0, 0, 0]) or [0, 0, 0, 0]
    x, y, w, h = [int(v) for v in bbox[:4]]
    w = max(1, w)
    h = max(1, h)
    page_area = 1
    page_w = 1
    page_h = 1
    if image_size:
        page_w = max(1, int(image_size[0]))
        page_h = max(1, int(image_size[1]))
        page_area = page_w * page_h
    area_ratio = (w * h) / max(1, page_area)
    topish = y <= page_h * 0.55
    wide_strip = w >= page_w * 0.28 and h <= page_h * 0.16
    chapter_marker = any(marker in text for marker in ("第", "話", "话", "章", "編", "篇"))
    if chapter_marker or wide_strip:
        return True
    if any(ch.isdigit() for ch in text) or _has_latin_text(text):
        return True
    if len(body) <= 18 and area_ratio <= 0.08 and topish:
        return True
    return False


def _box_luma_stats_pil(image_obj, bbox: list):
    if image_obj is None or not bbox:
        return None
    try:
        from PIL import ImageStat
    except Exception:
        return None
    try:
        img_w, img_h = image_obj.size
        x, y, w, h = [int(v) for v in bbox[:4]]
        x0 = max(0, min(x, img_w - 1))
        y0 = max(0, min(y, img_h - 1))
        x1 = max(x0 + 1, min(x + max(1, w), img_w))
        y1 = max(y0 + 1, min(y + max(1, h), img_h))
        crop = image_obj.crop((x0, y0, x1, y1)).convert("L")
        stat = ImageStat.Stat(crop)
        extrema = crop.getextrema()
        if not stat.mean or extrema is None:
            return None
        return float(stat.mean[0]), int(extrema[0]), int(extrema[1])
    except Exception:
        return None


def _is_dark_caption_box(stats, text: str) -> bool:
    if not stats or len(text) < 2:
        return False
    mean, low, high = stats
    if high >= 190:
        return False
    return mean < 125 and low < 110


def _katakana_ratio_text(text: str) -> float:
    if not text:
        return 0.0
    count = sum(1 for ch in text if 0x30A0 <= ord(ch) <= 0x30FF)
    return count / max(1, len(text))


def _has_mixed_scripts(text: str) -> bool:
    has_hira = any(0x3040 <= ord(ch) <= 0x309F for ch in text)
    has_kata = any(0x30A0 <= ord(ch) <= 0x30FF for ch in text)
    has_kanji = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in text)
    return sum(1 for flag in (has_hira, has_kata, has_kanji) if flag) >= 3


def _has_latin_text(text: str) -> bool:
    return any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in str(text or ""))


def _looks_like_decorative_title_artifact(
    text: str,
    bbox: list,
    image_size: tuple[int, int],
    det_conf: float,
    ocr_conf: float,
    mixed_scripts: bool,
    has_latin: bool,
) -> bool:
    if not text:
        return False
    if any(ch in text for ch in "。！？!?"):
        return False
    has_cjk = any(_is_cjk_char(ch) for ch in str(text or ""))
    _, _, w, h = bbox
    page_area = max(1, image_size[0] * image_size[1])
    area_ratio = (max(1, w) * max(1, h)) / page_area
    cx = (bbox[0] + (w / 2.0)) / max(1, image_size[0])
    cy = (bbox[1] + (h / 2.0)) / max(1, image_size[1])
    centered = 0.22 <= cx <= 0.78 and 0.18 <= cy <= 0.82
    thin_strip = h <= 80 and w >= h * 3.0
    large_box = area_ratio >= 0.012 or (max(w, h) >= min(image_size) * 0.22)
    if has_latin and has_cjk and large_box:
        return True
    if has_latin and mixed_scripts and large_box:
        return True
    if has_latin and area_ratio >= 0.006 and ocr_conf < 0.995 and det_conf >= 0.85:
        return True
    if centered and thin_strip and any(marker in text for marker in ("第", "話", "章", "編", "列伝", "列傳", "伝", "傳", "〜", "~", "「", "」", "『", "』", "・", "【", "】")):
        if ocr_conf < 0.92 or det_conf < 0.92 or mixed_scripts or has_latin:
            return True
    return False


def _looks_like_background_artifact(
    text: str,
    bbox: list,
    image_size: tuple[int, int],
    det_conf: float,
    ocr_conf: float,
    mixed_scripts: bool,
) -> bool:
    _, _, w, h = bbox
    page_area = max(1, image_size[0] * image_size[1])
    area_ratio = (max(1, w) * max(1, h)) / page_area
    thin_strip = h <= 28 and w >= h * 3.0
    cx = (bbox[0] + (w / 2.0)) / max(1, image_size[0])
    cy = (bbox[1] + (h / 2.0)) / max(1, image_size[1])
    centered = 0.22 <= cx <= 0.78 and 0.18 <= cy <= 0.82
    body = _non_punct_chars(text)
    if thin_strip and mixed_scripts and ocr_conf < 0.95:
        return True
    if thin_strip and len(text) <= 8 and det_conf >= 0.95 and ocr_conf < 0.92:
        return True
    if centered and h <= 84 and w >= h * 3.0 and area_ratio >= 0.003:
        if (det_conf < 0.8 and ocr_conf < 0.8) or any(marker in text for marker in ("第", "話", "章", "編", "列伝", "列傳", "伝", "傳", "〜", "~", "「", "」", "『", "』", "・", "【", "】")):
            return True
    if len(body) <= 3 and area_ratio <= 0.0045 and det_conf < 0.8 and ocr_conf < 0.8:
        return True
    if area_ratio < 0.001 and _placeholder_ratio(text) > 0.0:
        return True
    return False


def _likely_sfx_effect_box(
    text: str,
    bbox: list,
    image_size: tuple[int, int],
    ocr_conf: float,
) -> bool:
    if any(ch in text for ch in "、。！？!?…"):
        return False
    _, _, w, h = bbox
    if h >= max(90, w * 2.2):
        return False
    page_area = max(1, image_size[0] * image_size[1])
    area_ratio = (max(1, w) * max(1, h)) / page_area
    short = len(text) <= 6
    mostly_katakana = _katakana_ratio_text(text) >= 0.6
    return short and mostly_katakana and (min(w, h) <= 60 or area_ratio < 0.003) and ocr_conf < 0.995


def _japanese_ratio(text: str) -> float:
    if not text:
        return 0.0
    jp = sum(1 for ch in text if _is_japanese(ch))
    return jp / max(1, len(text))


def _placeholder_ratio(text: str) -> float:
    if not text:
        return 0.0
    placeholders = {"□", "口", "�"}
    count = sum(1 for ch in text if ch in placeholders)
    return count / max(1, len(text))


def _is_punct_only(text: str) -> bool:
    stripped = "".join(ch for ch in text if ch.strip())
    if not stripped:
        return True
    letters = sum(1 for ch in stripped if ch.isalnum() or _is_japanese(ch))
    return letters == 0


_SLASH_LIKE_WAVE_OCR_MARK = "〳"
_VERTICAL_KANA_REPEAT_FOLLOWERS = {"〵", "〴"}
_WAVE_OCR_TERMINAL_FOLLOWERS = set("。．.、，,！!？?…︙ー-—―－～〜~〰︴」』）)]｝》】")


def _normalize_slash_like_wave_ocr_marks(text: str) -> str:
    if _SLASH_LIKE_WAVE_OCR_MARK not in text:
        return text
    chars = list(text)
    for index, char in enumerate(chars):
        if char != _SLASH_LIKE_WAVE_OCR_MARK:
            continue
        previous = _previous_nonspace(chars, index)
        following = _next_nonspace(chars, index)
        if following in _VERTICAL_KANA_REPEAT_FOLLOWERS:
            continue
        if previous and _is_cjk_char(previous) and (not following or following in _WAVE_OCR_TERMINAL_FOLLOWERS):
            chars[index] = "〜"
    return "".join(chars)


def _previous_nonspace(chars: list[str], index: int) -> str:
    for pos in range(index - 1, -1, -1):
        if str(chars[pos]).strip():
            return chars[pos]
    return ""


def _next_nonspace(chars: list[str], index: int) -> str:
    for pos in range(index + 1, len(chars)):
        if str(chars[pos]).strip():
            return chars[pos]
    return ""


def _clean_ocr_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("□", "").replace("�", "")
    if _placeholder_ratio(cleaned) >= 0.2:
        cleaned = cleaned.replace("口", "")
    cleaned = _normalize_slash_like_wave_ocr_marks(cleaned)

    # For CJK text, remove ALL spaces (Japanese/Chinese don't use word spaces)
    # Use _is_valid_japanese score which correctly includes punctuation
    # If score > 0.4, it's likely Japanese/Chinese text
    stripped = cleaned.replace(" ", "")
    if stripped and _is_valid_japanese(stripped) > 0.4:
        # Remove all spaces from Japanese-dominant text
        cleaned = stripped

    # For non-CJK text, just normalize whitespace
    if " " in cleaned:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def _is_cjk_char(ch: str) -> bool:
    """Check if character is CJK (Chinese/Japanese/Korean)."""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF       # CJK Unified Ideographs
        or 0x3040 <= code <= 0x30FF    # Hiragana + Katakana
        or 0x3400 <= code <= 0x4DBF    # CJK Extension A
    )




def _is_japanese(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF
        or 0x4E00 <= code <= 0x9FFF
    )


def _is_kana(ch: str) -> bool:
    code = ord(ch)
    return 0x3040 <= code <= 0x30FF




def _region_record(
    idx: int,
    polygon: list,
    bbox: list,
    ocr_text: str,
    translation: str,
    det_conf: float,
    bg_text: bool,
    needs_review: bool,
    ignore: bool,
    region_type: str = "speech_bubble",
    ocr_conf: float = 1.0,
    render_updates: dict | None = None,
) -> dict:
    render = {
        "cleanup_mode": (
            "bubble"
            if region_type == "speech_bubble"
            else ("local_text_mask" if region_type in {"background_text", "narration_box"} else "background_box")
        ),
    }
    if isinstance(render_updates, dict):
        legacy_style_keys = {
            "font",
            "font_size",
            "source_size_hint",
            "source_size_min",
            "source_size_max",
            "font_size_locked",
            "font_size_policy",
            "font_size_fallback_policy",
            "source_orientation",
            "wrap_mode",
            "line_height",
            "align",
            "color",
            "stroke",
            "stroke_width",
            "font_style",
            "font_weight",
            "spacing_profile",
            "render_style",
            "render_style_owner",
            "render_style_version",
            "render_style_source",
            "render_style_provider",
            "render_style_provider_model",
            "render_style_confidence",
        }
        render.update(
            {
                key: value
                for key, value in render_updates.items()
                if value is not None and key not in legacy_style_keys
            }
        )
    return {
        "region_id": f"r{idx:03d}",
        "bbox": bbox,
        "polygon": polygon,
        "type": region_type,
        "ocr_text": ocr_text,
        "translation": translation,
        "confidence": {"det": det_conf, "ocr": ocr_conf, "trans": 1.0},
        "render": render,
        "flags": {"ignore": ignore, "bg_text": bg_text, "needs_review": needs_review},
    }

def _get_image_size(image_path: str) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError:
        return (0, 0)
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return (0, 0)


def _read_image_cv(image_path: str):
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    image = cv2.imread(image_path)
    if image is None:
        try:
            data = np.fromfile(image_path, dtype=np.uint8)
            if data.size:
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            image = None
    return image


def _scale_polygon(polygon: list, scale: float) -> list:
    scaled = []
    for point in polygon:
        if point is None or len(point) < 2:
            continue
        scaled.append([float(point[0]) * scale, float(point[1]) * scale])
    return scaled


def _detect_with_scale(detector, image_path: str, image_size: tuple[int, int], target_long: int = 1280):
    image = _read_image_cv(image_path)
    if image is None or not hasattr(detector, "detect_image"):
        return detector.detect(image_path)
    try:
        import cv2
    except Exception:
        return detector.detect(image_path)
    h, w = image.shape[:2]
    long_edge = max(w, h)
    scale = 1.0
    if long_edge > target_long:
        scale = target_long / float(long_edge)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    detections = detector.detect_image(image)
    if scale != 1.0:
        inv = 1.0 / scale
        scaled = []
        for polygon, conf in detections:
            scaled.append((_scale_polygon(polygon, inv), conf))
        return scaled
    return detections


def _detect_regions(
    detector,
    image_path: str,
    image_size: tuple[int, int],
    input_size: int = 1024,
    use_gpu: bool = False,
    message_callback=None,
):
    try:
        if hasattr(detector, "detect"):
            try:
                return detector.detect(image_path, input_size=input_size)
            except TypeError:
                return _detect_with_scale(detector, image_path, image_size, target_long=input_size)
        return _detect_with_scale(detector, image_path, image_size, target_long=input_size)
    except Exception as exc:
        detector_name = detector.__class__.__name__
        if message_callback is not None:
            try:
                message_callback(f"{detector_name} failed on {os.path.basename(image_path)}: {exc}")
            except Exception:
                pass
        raise


def _classify_region(
    bbox: list,
    image_size: tuple[int, int],
    det_conf: float,
    filter_background: bool,
    filter_strength: str,
) -> tuple[bool, bool]:
    img_w, img_h = image_size
    if img_w <= 0 or img_h <= 0:
        return False, det_conf < 0.6
    x, y, w, h = bbox
    area = w * h
    page_area = img_w * img_h
    if page_area <= 0:
        return False, det_conf < 0.6
    ratio = area / page_area
    aspect = w / h if h else 0
    margin_x = img_w * 0.02
    margin_y = img_h * 0.02
    near_edge = x < margin_x or y < margin_y or (x + w) > (img_w - margin_x) or (y + h) > (img_h - margin_y)

    aggressive = filter_strength == "aggressive"
    large_ratio = 0.12 if not aggressive else 0.09
    strip_ratio = 0.05 if not aggressive else 0.03
    edge_ratio = 0.03 if not aggressive else 0.02

    bg_text = False
    if ratio > large_ratio and (near_edge or aspect > 4):
        bg_text = True
    elif aspect > 5 and ratio > strip_ratio:
        bg_text = True
    elif near_edge and ratio > edge_ratio:
        bg_text = True

    if not filter_background:
        bg_text = False

    needs_review = det_conf < 0.6 or (bg_text and aggressive)
    return bg_text, needs_review


def _is_cjk_term(term: str) -> bool:
    for ch in term:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            return True
        if 0x3040 <= code <= 0x30FF:
            return True
        if 0xAC00 <= code <= 0xD7AF:
            return True
    return False


def _contains_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    if _is_cjk_term(term):
        return term in text
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _normalize_character_entry(entry: dict) -> dict:
    """Normalize character schema to a stable structure for all pipeline consumers."""
    if not isinstance(entry, dict):
        return {}
    original = str(entry.get("original") or entry.get("canonical") or "").strip()
    reading = str(entry.get("reading") or entry.get("canonical_reading") or "").strip()
    translation = str(entry.get("translation") or "").strip()
    name = str(entry.get("name") or "").strip()
    if not original and name:
        original = name
    if not name:
        name = translation or original
    aliases_raw = entry.get("aliases", []) or []
    aliases = []
    for alias in aliases_raw:
        if isinstance(alias, dict):
            source = str(alias.get("source", "")).strip()
            target = str(alias.get("target", "") or alias.get("translation", "")).strip()
            if source:
                aliases.append(
                    {
                        "source": source,
                        "target": target,
                        "reading": str(alias.get("reading", "")).strip(),
                        "pattern": str(alias.get("pattern", "")).strip(),
                        "hint": str(alias.get("hint", "")).strip(),
                    }
                )
        else:
            source = str(alias).strip()
            if source:
                aliases.append(
                    {
                        "source": source,
                        "target": translation,
                        "reading": "",
                        "pattern": "",
                        "hint": "",
                    }
                )
    return {
        "canonical": original,
        "original": original,
        "name": name,
        "translation": translation,
        "reading": reading,
        "gender": str(entry.get("gender", "")).strip(),
        "info": str(entry.get("info", "")).strip(),
        "aliases": aliases,
    }


def _find_inconsistent_pages(pages: list, style_guide: dict) -> list[int]:
    if not pages or not isinstance(style_guide, dict):
        return []
    term_targets: dict[str, set[str]] = {}
    glossary = style_guide.get("glossary", [])
    for item in glossary:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source", "")).strip()
        tgt = str(item.get("target", "")).strip()
        if len(src) < 2 or not tgt:
            continue
        term_targets.setdefault(src, set()).add(tgt)
    characters = style_guide.get("characters", [])
    if isinstance(characters, list):
        for raw_char in characters:
            char = _normalize_character_entry(raw_char)
            if not char:
                continue
            original = str(char.get("original", "")).strip()
            translation = str(char.get("translation", "")).strip()
            canonical_target = translation
            if original and canonical_target and canonical_target != original:
                term_targets.setdefault(original, set()).add(canonical_target)
            aliases = char.get("aliases", []) or []
            for alias in aliases:
                alias_source = str(alias.get("source", "")).strip()
                alias_target = str(alias.get("target", "")).strip()
                if not alias_source:
                    continue
                alias_targets = set()
                if alias_target and alias_target != alias_source:
                    alias_targets.add(alias_target)
                if canonical_target and canonical_target != alias_source:
                    alias_targets.add(canonical_target)
                if alias_targets:
                    term_targets.setdefault(alias_source, set()).update(alias_targets)
    if not term_targets:
        return []
    terms = list(term_targets.items())
    inconsistent_pages = []
    for page_idx, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        regions = page.get("regions", []) or page.get("blocks", [])
        for region in regions:
            if not isinstance(region, dict):
                continue
            flags = region.get("flags", {}) or {}
            if flags.get("ignore"):
                continue
            source_text = str(region.get("ocr_text", "")).strip()
            translation = str(region.get("translation", "")).strip()
            if not source_text or not translation:
                continue
            for src, targets in terms:
                if _contains_term(source_text, src):
                    if not any(_contains_term(translation, tgt) for tgt in targets):
                        inconsistent_pages.append(page_idx)
                        break
            if inconsistent_pages and inconsistent_pages[-1] == page_idx:
                break
    return inconsistent_pages


def _is_supported_name_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF
        or 0x4E00 <= code <= 0x9FFF
        or ch in {"ー", "・", "々", "ヶ", "ケ", "ヴ"}
    )


def _is_pure_katakana(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return False
    for ch in text:
        code = ord(ch)
        if not (0x30A0 <= code <= 0x30FF or ch in {"ー", "・"}):
            return False
    return True


def _looks_like_clean_name_surface(text: str) -> bool:
    text = str(text or "").strip()
    if not text or len(text) > 12:
        return False
    if not all(_is_supported_name_char(ch) for ch in text):
        return False
    if all(0x4E00 <= ord(ch) <= 0x9FFF for ch in text) and len(text) > 6:
        return False
    for honorific in ("さん", "くん", "ちゃん", "様", "先生", "先輩", "殿", "君", "氏"):
        pos = text.find(honorific)
        if pos >= 0 and pos + len(honorific) < len(text):
            return False
    return _is_cjk_term(text)


def _looks_like_clean_cjk_target(text: str, target_lang: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return False
    if target_lang not in {"Simplified Chinese", "Traditional Chinese"}:
        return True
    if len(text) > 12:
        return False
    allowed_punct = set("·・0123456789０１２３４５６７８９ ")
    saw_han = False
    for ch in text:
        code = ord(ch)
        if ch in allowed_punct:
            continue
        if 0x4E00 <= code <= 0x9FFF:
            saw_han = True
            continue
        return False
    return saw_han


_JP_TO_SIMPLIFIED_NAME_CHARS = str.maketrans(
    {
        "亜": "亚",
        "亞": "亚",
        "偉": "伟",
        "傳": "传",
        "伝": "传",
        "優": "优",
        "兒": "儿",
        "児": "儿",
        "劍": "剑",
        "剣": "剑",
        "勝": "胜",
        "國": "国",
        "園": "园",
        "廣": "广",
        "広": "广",
        "恆": "恒",
        "櫻": "樱",
        "桜": "樱",
        "澤": "泽",
        "沢": "泽",
        "濱": "滨",
        "浜": "滨",
        "瀧": "泷",
        "滝": "泷",
        "瑤": "瑶",
        "發": "发",
        "穂": "穗",
        "繪": "绘",
        "絵": "绘",
        "聖": "圣",
        "與": "与",
        "葉": "叶",
        "藝": "艺",
        "藏": "藏",
        "蔵": "藏",
        "衛": "卫",
        "謙": "谦",
        "貴": "贵",
        "賢": "贤",
        "輔": "辅",
        "輝": "辉",
        "邊": "边",
        "辺": "边",
        "鄉": "乡",
        "郷": "乡",
        "關": "关",
        "関": "关",
        "陽": "阳",
        "隱": "隐",
        "隠": "隐",
        "靜": "静",
        "須": "须",
        "顯": "显",
        "顕": "显",
        "馬": "马",
        "島": "岛",
        "鳥": "鸟",
        "豐": "丰",
        "豊": "丰",
        "齋": "斋",
        "斎": "斋",
        "齊": "齐",
        "斉": "齐",
        "龍": "龙",
        "竜": "龙",
    }
)


def _normalize_simplified_name_target(text: str) -> str:
    return str(text or "").translate(_JP_TO_SIMPLIFIED_NAME_CHARS)


def _dedupe_repeated_cjk_phrase(text: str) -> str:
    text = str(text or "").strip()
    if not text or len(text) < 4:
        return text
    for unit_len in range(2, (len(text) // 2) + 1):
        if len(text) % unit_len != 0:
            continue
        unit = text[:unit_len]
        repeats = len(text) // unit_len
        if repeats >= 2 and unit * repeats == text:
            return unit
    return text


def _sanitize_style_guide(style_guide: dict, target_lang: str) -> dict:
    if not isinstance(style_guide, dict):
        return style_guide
    glossary = style_guide.get("glossary", [])
    cleaned_glossary = []
    changed = False
    # Normalize characters to a single schema.
    normalized_chars = []
    raw_chars = style_guide.get("characters", []) or []
    for raw_char in raw_chars:
        norm = _normalize_character_entry(raw_char)
        if not norm:
            continue
        original = str(norm.get("original", "")).strip()
        reading = str(norm.get("reading", "")).strip()
        translation = str(norm.get("translation", "")).strip()
        if translation and target_lang == "Simplified Chinese":
            cleaned_translation = _sanitize_glossary_target(translation, original, target_lang)
            if cleaned_translation and cleaned_translation != translation:
                norm = dict(norm)
                norm["translation"] = cleaned_translation
                norm["name"] = cleaned_translation
                translation = cleaned_translation
                changed = True
        if not original or not _looks_like_clean_name_surface(original):
            changed = True
            continue
        if all(_is_kana(ch) for ch in original) and not _is_pure_katakana(original):
            changed = True
            continue
        if _is_cjk_term(original) and (not reading or not all(_is_kana(ch) for ch in reading)):
            changed = True
            continue
        if translation and not _looks_like_clean_cjk_target(translation, target_lang):
            changed = True
            continue
        if norm and norm.get("original"):
            normalized_chars.append(norm)
    if raw_chars != normalized_chars:
        style_guide = dict(style_guide)
        style_guide["characters"] = normalized_chars
        changed = True

    alias_target_map: dict[str, str] = {}
    alias_owner_map: dict[str, str] = {}
    for char in normalized_chars:
        canonical = str(char.get("original", "")).strip()
        for alias in char.get("aliases", []) or []:
            if not isinstance(alias, dict):
                continue
            src = str(alias.get("source", "")).strip()
            tgt = str(alias.get("target", "")).strip()
            if not src or src == canonical:
                continue
            alias_owner_map.setdefault(src, canonical)
            if tgt:
                alias_target_map.setdefault(src, tgt)

    deduped_chars = []
    for char in normalized_chars:
        original = str(char.get("original", "")).strip()
        owner = alias_owner_map.get(original, "")
        if owner and owner != original:
            changed = True
            continue
        deduped_chars.append(char)
    if deduped_chars != normalized_chars:
        normalized_chars = deduped_chars
        style_guide = dict(style_guide)
        style_guide["characters"] = normalized_chars

    alias_sources = set()
    alias_target_map = {}
    for char in normalized_chars:
        original = str(char.get("original", "")).strip()
        translation = str(char.get("translation", "")).strip()
        if original:
            alias_sources.add(original)
        if original and translation and translation != original:
            alias_target_map.setdefault(original, translation)
        for alias in char.get("aliases", []) or []:
            src = str(alias.get("source", "")).strip()
            tgt = str(alias.get("target", "")).strip()
            if src:
                alias_sources.add(src)
            if src and tgt:
                alias_target_map[src] = tgt

    # Collect aliases for name validation.
    honorifics = ("さん", "くん", "ちゃん", "様", "先生", "先輩", "殿", "君", "氏")
    standalone_honorifics = set(honorifics)
    for item in glossary:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        reading = str(item.get("reading", "")).strip()
        pattern = str(item.get("pattern", "")).strip()
        preferred_target = alias_target_map.get(source, "")
        target_to_check = preferred_target or target
        cleaned_target = _sanitize_glossary_target(target_to_check, source, target_lang)
        cleaned_target = _dedupe_repeated_cjk_phrase(cleaned_target)
        if (
            cleaned_target
            and _is_cjk_term(source)
            and _is_cjk_term(cleaned_target)
            and len(source) <= 2
            and len(cleaned_target) - len(source) >= 2
            and cleaned_target.startswith(source)
        ):
            cleaned_target = source

        if not source:
            changed = True
            continue

        if len(source) > 30 or "处理用户" in source or "Need to" in source or "require" in source:
            changed = True
            continue

        if cleaned_target and not _looks_like_clean_cjk_target(cleaned_target, target_lang):
            changed = True
            continue

        if (
            cleaned_target
            and _is_cjk_term(source)
            and _is_cjk_term(cleaned_target)
            and len(source) <= 3
            and len(cleaned_target) - len(source) >= (2 if len(source) <= 2 else 3)
            and cleaned_target.startswith(source)
            and source not in alias_sources
            and not reading
        ):
            changed = True
            continue

        if item.get("auto"):
            if not _looks_like_clean_name_surface(source):
                changed = True
                continue
            has_honorific = any(h in source for h in honorifics)
            reading_is_kana = bool(reading) and all(_is_kana(ch) for ch in reading)
            source_is_kana = bool(source) and all(_is_kana(ch) for ch in source)
            source_is_pure_katakana = _is_pure_katakana(source)
            if source in standalone_honorifics:
                changed = True
                continue
            if source_is_kana and len(source) <= 2 and not has_honorific and not source_is_pure_katakana:
                changed = True
                continue
            if source_is_kana and not source_is_pure_katakana and not has_honorific and source not in alias_sources:
                changed = True
                continue
            if source_is_kana and not has_honorific and source not in alias_sources:
                if not (source_is_pure_katakana and len(source) >= 3):
                    changed = True
                    continue
            if pattern == "standalone" and _is_pure_katakana(source) and source not in alias_sources:
                changed = True
                continue
            if not source_is_kana and not has_honorific and not reading_is_kana and source not in alias_sources:
                changed = True
                continue
            if not (has_honorific or reading_is_kana or source_is_kana or source in alias_sources):
                if _is_cjk_term(source) and (not reading or reading == source) and len(source) <= 3:
                    changed = True
                    continue
                if not reading or reading == source:
                    changed = True
                    continue
            if not cleaned_target:
                changed = True
                continue
            if cleaned_target != target:
                new_item = dict(item)
                new_item["target"] = cleaned_target
                cleaned_glossary.append(new_item)
                changed = True
                continue
        elif cleaned_target and cleaned_target != target:
            new_item = dict(item)
            new_item["target"] = cleaned_target
            cleaned_glossary.append(new_item)
            changed = True
            continue
        cleaned_glossary.append(item)
    final_glossary = []
    for item in cleaned_glossary:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        if not source:
            changed = True
            continue
        if item.get("auto"):
            source_is_kana = bool(source) and all(_is_kana(ch) for ch in source)
            source_is_pure_katakana = _is_pure_katakana(source)
            has_honorific = any(h in source for h in honorifics)
            if source in standalone_honorifics:
                changed = True
                continue
            if source_is_kana and not source_is_pure_katakana and not has_honorific and source not in alias_sources:
                changed = True
                continue
        final_glossary.append(item)
    cleaned_glossary = final_glossary
    if changed:
        style_guide = dict(style_guide)
        style_guide["glossary"] = cleaned_glossary
    return style_guide


def _merge_glossary(style_guide: dict, new_map: dict, new_chars: list) -> dict:
    """Merge new glossary items into style guide."""
    # Ensure glossary list exists
    sg_glossary = style_guide.setdefault("glossary", [])

    # Map existing entries by source for quick lookup
    existing_map = {item["source"]: item for item in sg_glossary if "source" in item}

    for src, val in new_map.items():
        # Handle rich dict vs simple string
        if isinstance(val, dict):
            target = val.get("target", "")
            reading = val.get("reading", "")
            pattern = val.get("pattern", "")
            hint = val.get("hint", "")
            entry_type = val.get("type", "term")
        else:
            target = val
            reading = ""
            pattern = ""
            hint = ""
            entry_type = "term"

        if src not in existing_map:
            # Create new entry
            entry = {
                "source": src,
                "target": target,
                "priority": "hard",
                "auto": True
            }
            if reading: entry["reading"] = reading
            if pattern: entry["pattern"] = pattern
            if hint: entry["hint"] = hint
            if entry_type: entry["type"] = entry_type

            sg_glossary.append(entry)
            existing_map[src] = entry
        else:
            # Update existing if needed (e.g. add metadata)
            entry = existing_map[src]
            if entry.get("auto"):
                 if target and target != entry.get("target", ""):
                     entry["target"] = target
                 if reading and "reading" not in entry:
                     entry["reading"] = reading
                 if pattern and "pattern" not in entry:
                     entry["pattern"] = pattern
                 if hint and "hint" not in entry:
                     entry["hint"] = hint

    # Merge characters with normalized schema.
    sg_chars_raw = style_guide.setdefault("characters", [])
    sg_chars = []
    existing_chars = {}
    for c in sg_chars_raw:
        norm = _normalize_character_entry(c)
        if not norm or not norm.get("original"):
            continue
        key = norm.get("original")
        sg_chars.append(norm)
        existing_chars[key] = norm
    style_guide["characters"] = sg_chars

    if new_chars:
        for char in new_chars:
            norm_char = _normalize_character_entry(char)
            if not norm_char:
                continue
            original = norm_char.get("original", "").strip()
            if len(original) > 20 or "处理用户" in original or "需要" in original:
                continue
            if not original:
                continue

            existing = existing_chars.get(original)
            if existing is None:
                sg_chars.append(norm_char)
                existing_chars[original] = norm_char
                continue

            new_aliases = norm_char.get("aliases", [])
            # Fill canonical fields if the existing entry is incomplete.
            if not existing.get("translation") and norm_char.get("translation"):
                existing["translation"] = norm_char.get("translation")
            if (not existing.get("name") or existing.get("name") == original) and norm_char.get("name"):
                existing["name"] = norm_char.get("name")
            if not existing.get("reading") and norm_char.get("reading"):
                existing["reading"] = norm_char.get("reading")
            if not existing.get("gender") and norm_char.get("gender"):
                existing["gender"] = norm_char.get("gender")
            if not existing.get("info") and norm_char.get("info"):
                existing["info"] = norm_char.get("info")
            existing_aliases = existing.setdefault("aliases", [])
            existing_alias_sources = set()
            for a in existing_aliases:
                s = a.get("source") if isinstance(a, dict) else str(a)
                if s:
                    existing_alias_sources.add(s)
            for alias in new_aliases:
                src = alias.get("source") if isinstance(alias, dict) else str(alias)
                if src and src not in existing_alias_sources:
                    existing_aliases.append(alias)
                    existing_alias_sources.add(src)

    return style_guide
