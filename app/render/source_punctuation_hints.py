# -*- coding: utf-8 -*-
"""Typed source-visual punctuation geometry for renderer presentation.

The observer consumes original pixels only through an accepted, parent-bound
``AuthorizedSourceStyleView``.  It never reads OCR punctuation to decide
whether an occurrence exists, never rewrites translated text, and never owns
render admission.  Only compact immutable evidence leaves this module.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from functools import lru_cache
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from app.pipeline.parent_style_evidence import (
    AUTHORIZED_SOURCE_STYLE_VIEW_VERSION,
)
from app.render.typesetting_contracts import bbox_from_value, copy_jsonish


SOURCE_PUNCTUATION_GEOMETRY_EVIDENCE_VERSION = (
    "source_punctuation_geometry_evidence_v1"
)
SOURCE_PUNCTUATION_GEOMETRY_OBSERVER_VERSION = (
    "source_punctuation_geometry_observer_v2"
)
SOURCE_PUNCTUATION_GEOMETRY_SUPPORT_VERSION = (
    "source_punctuation_geometry_support_v2"
)
SOURCE_PUNCTUATION_CELL_CALIBRATION_VERSION = (
    "source_punctuation_cell_calibration_v1"
)
_SUPPORTED_KINDS = frozenset({"dash", "ellipsis", "wave"})
_SUPPORTED_INLINE_AXES = frozenset({"ttb", "ltr"})


@dataclass(frozen=True)
class SourcePunctuationGeometryOccurrence:
    """One source-pixel punctuation occurrence in visual reading order."""

    occurrence_id: str
    kind: str
    visual_reading_order_ordinal: int
    kind_ordinal: int
    inline_axis: str
    component_bboxes_local_xywh: tuple[tuple[int, int, int, int], ...]
    component_bboxes_page_xywh: tuple[tuple[int, int, int, int], ...]
    group_bbox_local_xywh: tuple[int, int, int, int]
    group_bbox_page_xywh: tuple[int, int, int, int]
    span_px: float
    pitch_px: float
    source_cell_px: float
    normalized_span: float
    normalized_pitch: float
    confidence: float
    reason_codes: tuple[str, ...] = ()

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "occurrence_id": self.occurrence_id,
            "kind": self.kind,
            "visual_reading_order_ordinal": int(
                self.visual_reading_order_ordinal
            ),
            "kind_ordinal": int(self.kind_ordinal),
            "inline_axis": self.inline_axis,
            "component_bboxes_local_xywh": [
                list(item) for item in self.component_bboxes_local_xywh
            ],
            "component_bboxes_page_xywh": [
                list(item) for item in self.component_bboxes_page_xywh
            ],
            "group_bbox_local_xywh": list(self.group_bbox_local_xywh),
            "group_bbox_page_xywh": list(self.group_bbox_page_xywh),
            "span_px": round(float(self.span_px), 6),
            "pitch_px": round(float(self.pitch_px), 6),
            "source_cell_px": round(float(self.source_cell_px), 6),
            "normalized_span": round(float(self.normalized_span), 6),
            "normalized_pitch": round(float(self.normalized_pitch), 6),
            "confidence": round(float(self.confidence), 6),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class SourcePunctuationGeometryEvidence:
    """Immutable compact geometry evidence for one parent execution bundle."""

    page_id: str
    bundle_id: str
    parent_id: str
    root_id: str
    status: str
    view_id: str = ""
    support_identity: Mapping[str, Any] = field(default_factory=dict)
    occurrences: tuple[SourcePunctuationGeometryOccurrence, ...] = ()
    abstention_reason: str = ""
    reason_codes: tuple[str, ...] = ()
    fact_set_id: str = ""

    def __post_init__(self) -> None:
        support = MappingProxyType(
            {
                str(key): _freeze_jsonish(value)
                for key, value in sorted(
                    dict(self.support_identity or {}).items(),
                    key=lambda item: str(item[0]),
                )
            }
        )
        object.__setattr__(self, "support_identity", support)
        object.__setattr__(self, "occurrences", tuple(self.occurrences or ()))
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_unique_strings(self.reason_codes)),
        )
        if not self.fact_set_id:
            object.__setattr__(
                self,
                "fact_set_id",
                source_punctuation_geometry_fact_set_id(
                    self._payload_without_fact_set()
                ),
            )

    @classmethod
    def unavailable(
        cls,
        *,
        page_id: str,
        bundle_id: str,
        parent_id: str,
        root_id: str,
        view_id: str = "",
        support_identity: Mapping[str, Any] | None = None,
        reason: str,
        reason_codes: Sequence[str] = (),
    ) -> "SourcePunctuationGeometryEvidence":
        reasons = _unique_strings([reason, *list(reason_codes or ())])
        return cls(
            page_id=str(page_id or ""),
            bundle_id=str(bundle_id or ""),
            parent_id=str(parent_id or ""),
            root_id=str(root_id or ""),
            status="unavailable",
            view_id=str(view_id or ""),
            support_identity=dict(support_identity or {}),
            occurrences=(),
            abstention_reason=str(reason or "source_geometry_unavailable"),
            reason_codes=tuple(reasons),
        )

    def _payload_without_fact_set(self) -> dict[str, Any]:
        return {
            "contract_version": SOURCE_PUNCTUATION_GEOMETRY_EVIDENCE_VERSION,
            "observer_version": SOURCE_PUNCTUATION_GEOMETRY_OBSERVER_VERSION,
            "source_identity": {
                "page_id": self.page_id,
                "bundle_id": self.bundle_id,
                "parent_id": self.parent_id,
                "root_id": self.root_id,
            },
            "status": self.status,
            "view_id": self.view_id,
            "support_identity": _thaw_jsonish(self.support_identity),
            "occurrences": [
                item.to_audit_dict() for item in self.occurrences
            ],
            "abstention_reason": self.abstention_reason,
            "reason_codes": list(self.reason_codes),
            "text_identity_authority": "translated_lossless_tokens",
            "geometry_authority": "parent_authorized_source_pixels",
            "render_admission_authority": False,
        }

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_fact_set(),
            "fact_set_id": self.fact_set_id,
        }


@dataclass(frozen=True)
class SourcePunctuationGeometryRunResult:
    page_id: str
    evidence: tuple[SourcePunctuationGeometryEvidence, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def evidence_by_bundle_id(
        self,
    ) -> dict[str, SourcePunctuationGeometryEvidence]:
        return {
            item.bundle_id: item
            for item in self.evidence
            if item.bundle_id
        }

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "source_punctuation_geometry_run_version": (
                "source_punctuation_geometry_run_v1"
            ),
            "page_id": self.page_id,
            "evidence": [item.to_audit_dict() for item in self.evidence],
            "errors": list(self.errors),
            "summary": {
                "parent_count": len(self.evidence),
                "observed_parent_count": sum(
                    1 for item in self.evidence if item.status == "observed"
                ),
                "unavailable_parent_count": sum(
                    1 for item in self.evidence if item.status == "unavailable"
                ),
                "occurrence_count": sum(
                    len(item.occurrences) for item in self.evidence
                ),
            },
        }


def observe_source_punctuation_geometry(
    *,
    page_id: str,
    source_image_path: str,
    parent_execution_bundles: Sequence[Any],
    authorized_style_views: Mapping[str, Any] | None,
    source_style_evidence_by_bundle_id: Mapping[str, Any] | None = None,
) -> SourcePunctuationGeometryRunResult:
    """Observe parent-local source punctuation without consulting OCR text."""

    page_value = str(page_id or "")
    bundles = [
        bundle
        for bundle in list(parent_execution_bundles or [])
        if bool(_value(bundle, "render_required", False))
    ]
    views = dict(authorized_style_views or {})
    style_evidence_by_bundle_id = dict(
        source_style_evidence_by_bundle_id or {}
    )
    image = _cached_grayscale_source_image(str(source_image_path or ""))
    image_error = "" if image is not None else "source_image_unavailable"
    evidence: list[SourcePunctuationGeometryEvidence] = []
    errors: list[str] = []
    seen_bundle_ids: set[str] = set()

    for bundle in bundles:
        bundle_id = str(_value(bundle, "bundle_id") or "")
        parent_id = str(_value(bundle, "parent_id") or "")
        root_id = str(_value(bundle, "root_id") or "")
        bundle_page_id = str(_value(bundle, "page_id") or "")
        view = views.get(bundle_id)
        source_style_evidence = style_evidence_by_bundle_id.get(bundle_id)
        if not bundle_id or bundle_id in seen_bundle_ids:
            errors.append(
                "source_punctuation_geometry_bundle_identity_missing_or_duplicate"
            )
            continue
        seen_bundle_ids.add(bundle_id)
        source_cell_calibration = _source_cell_calibration(
            page_id=page_value,
            bundle_id=bundle_id,
            parent_id=parent_id,
            root_id=root_id,
            view=view,
            style_evidence=source_style_evidence,
        )
        base_support = _base_support_identity(
            page_id=page_value,
            bundle_id=bundle_id,
            parent_id=parent_id,
            root_id=root_id,
            view=view,
            source_cell_calibration=source_cell_calibration,
        )
        identity_reasons: list[str] = []
        if not page_value:
            identity_reasons.append("source_geometry_page_identity_missing")
        if bundle_page_id != page_value:
            identity_reasons.append(
                "source_geometry_bundle_page_identity_mismatch"
            )
        if not parent_id:
            identity_reasons.append("source_geometry_parent_identity_missing")
        if not root_id:
            identity_reasons.append("source_geometry_root_identity_missing")
        if image_error:
            identity_reasons.append(image_error)
        identity_reasons.extend(
            _authorized_view_rejection_reasons(
                view,
                page_id=page_value,
                bundle_id=bundle_id,
                parent_id=parent_id,
                root_id=root_id,
                image_shape=image.shape if image is not None else (),
            )
        )
        if identity_reasons:
            evidence.append(
                SourcePunctuationGeometryEvidence.unavailable(
                    page_id=page_value,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    view_id=str(_value(view, "view_id") or ""),
                    support_identity=base_support,
                    reason=identity_reasons[0],
                    reason_codes=tuple(identity_reasons),
                )
            )
            continue
        try:
            evidence.append(
                _observe_parent_geometry(
                    image=image,
                    page_id=page_value,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    view=view,
                    source_cell_calibration=source_cell_calibration,
                )
            )
        except Exception as exc:
            reason = f"source_geometry_observer_error_{type(exc).__name__}"
            evidence.append(
                SourcePunctuationGeometryEvidence.unavailable(
                    page_id=page_value,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    view_id=str(_value(view, "view_id") or ""),
                    support_identity=base_support,
                    reason=reason,
                    reason_codes=(reason,),
                )
            )
            errors.append(f"{bundle_id}:{type(exc).__name__}:{exc}")

    return SourcePunctuationGeometryRunResult(
        page_id=page_value,
        evidence=tuple(evidence),
        errors=tuple(_unique_strings(errors)),
    )


def source_punctuation_geometry_fact_set_id(
    record: Mapping[str, Any],
) -> str:
    """Return the deterministic fact-set id for a record without its id."""

    payload = {
        str(key): copy_jsonish(value)
        for key, value in dict(record or {}).items()
        if str(key) != "fact_set_id"
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"{SOURCE_PUNCTUATION_GEOMETRY_EVIDENCE_VERSION}:{digest}"


def source_punctuation_cell_calibration_sha256(
    record: Mapping[str, Any],
) -> str:
    """Hash one source-cell calibration payload without its stored hash."""

    payload = {
        str(key): copy_jsonish(value)
        for key, value in dict(record or {}).items()
        if str(key) != "calibration_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _source_cell_calibration(
    *,
    page_id: str,
    bundle_id: str,
    parent_id: str,
    root_id: str,
    view: Any,
    style_evidence: Any,
) -> dict[str, Any]:
    """Bind supported raw scale evidence to the same parent-authorized view."""

    identity = {
        "page_id": str(_value(style_evidence, "page_id") or ""),
        "bundle_id": str(_value(style_evidence, "bundle_id") or ""),
        "parent_id": str(_value(style_evidence, "parent_id") or ""),
        "root_id": str(_value(style_evidence, "root_id") or ""),
        "view_id": str(_value(style_evidence, "view_id") or ""),
        "cleanup_mask_ids": _strings(
            _value(style_evidence, "cleanup_mask_ids", ())
        ),
        "owned_component_ids": _strings(
            _value(style_evidence, "owned_component_ids", ())
        ),
        "content_bbox_xywh": list(
            bbox_from_value(_value(style_evidence, "content_bbox", ()))
        ),
        "analysis_bbox_xywh": list(
            bbox_from_value(_value(style_evidence, "analysis_bbox", ()))
        ),
        "style_evidence_status": str(
            _value(style_evidence, "status") or ""
        ),
    }
    expected_identity = {
        "page_id": str(page_id or ""),
        "bundle_id": str(bundle_id or ""),
        "parent_id": str(parent_id or ""),
        "root_id": str(root_id or ""),
        "view_id": str(_value(view, "view_id") or ""),
        "cleanup_mask_ids": _strings(
            _value(view, "cleanup_mask_ids", ())
        ),
        "owned_component_ids": _strings(
            _value(view, "owned_component_ids", ())
        ),
        "content_bbox_xywh": list(
            bbox_from_value(_value(view, "content_bbox", ()))
        ),
        "analysis_bbox_xywh": list(
            bbox_from_value(_value(view, "analysis_bbox", ()))
        ),
        "style_evidence_status": "observed",
    }
    reasons: list[str] = []
    if style_evidence is None:
        reasons.append("source_style_evidence_missing")
    if not bool(_value(view, "available", False)):
        reasons.append("authorized_source_style_view_unavailable_for_calibration")
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            reasons.append(f"source_style_evidence_identity_mismatch:{key}")

    scale_axis = next(
        (
            item
            for item in tuple(_value(style_evidence, "axis_evidence", ()) or ())
            if str(_value(item, "axis") or "").strip().lower() == "scale"
        ),
        None,
    )
    scale_payload: dict[str, Any] = {}
    if scale_axis is not None:
        to_audit_dict = getattr(scale_axis, "to_audit_dict", None)
        if callable(to_audit_dict):
            scale_payload = copy_jsonish(to_audit_dict())
        elif isinstance(scale_axis, Mapping):
            scale_payload = copy_jsonish(scale_axis)
    scale_sha256 = (
        hashlib.sha256(
            json.dumps(
                scale_payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if scale_payload
        else ""
    )
    if scale_axis is None:
        reasons.append("source_scale_axis_missing")
    elif str(_value(scale_axis, "status") or "") != "supported":
        reasons.append("source_scale_axis_not_supported")

    scale_support_identity = _value(
        scale_axis,
        "support_identity",
        {},
    )
    if not isinstance(scale_support_identity, Mapping):
        scale_support_identity = {}
    for key in (
        "page_id",
        "bundle_id",
        "parent_id",
        "root_id",
        "view_id",
        "cleanup_mask_ids",
        "owned_component_ids",
    ):
        expected = expected_identity[key]
        actual = (
            _strings(scale_support_identity.get(key, ()))
            if key in {"cleanup_mask_ids", "owned_component_ids"}
            else str(scale_support_identity.get(key) or "")
        )
        if scale_axis is not None and actual != expected:
            reasons.append(f"source_scale_axis_identity_mismatch:{key}")

    values = _value(scale_axis, "value", {})
    if not isinstance(values, Mapping):
        values = {}
    identity_valid = not reasons
    axes: dict[str, dict[str, Any]] = {}
    axis_reasons: list[str] = []
    for inline_axis, direction in (("ttb", "vertical"), ("ltr", "horizontal")):
        cell = _finite_nonnegative(values.get(f"{direction}_px"))
        confidence = _finite_unit_interval(
            values.get(
                f"{direction}_confidence",
                _value(scale_axis, "confidence", 0.0),
            )
        )
        support_status = str(
            values.get(f"{direction}_support") or ""
        )
        supported = (
            identity_valid
            and cell > 0.0
            and confidence > 0.0
            and support_status.startswith("supported_")
        )
        axis_reason = "" if supported else f"source_{direction}_cell_unavailable"
        axes[inline_axis] = {
            "source_axis": "scale",
            "source_direction": direction,
            "status": "supported" if supported else "unavailable",
            "source_cell_px": round(cell if supported else 0.0, 6),
            "confidence": round(confidence if supported else 0.0, 8),
            "support_status": support_status if supported else "",
            "reason": axis_reason,
        }
        if axis_reason:
            axis_reasons.append(axis_reason)

    payload = {
        "contract_version": SOURCE_PUNCTUATION_CELL_CALIBRATION_VERSION,
        "status": (
            "supported"
            if any(item["status"] == "supported" for item in axes.values())
            else "unavailable"
        ),
        "style_evidence_identity": identity,
        "source_scale_axis_sha256": scale_sha256,
        "axes": axes,
        "reason_codes": _unique_strings([*reasons, *axis_reasons]),
    }
    payload["calibration_sha256"] = (
        source_punctuation_cell_calibration_sha256(payload)
    )
    return payload


def _supported_source_cells(
    calibration: Mapping[str, Any],
) -> dict[str, float]:
    if str(_value(calibration, "status") or "") != "supported":
        return {}
    axes = _value(calibration, "axes", {})
    if not isinstance(axes, Mapping):
        return {}
    result: dict[str, float] = {}
    for axis in _SUPPORTED_INLINE_AXES:
        record = axes.get(axis)
        if not isinstance(record, Mapping):
            continue
        if str(record.get("status") or "") != "supported":
            continue
        cell = _finite_nonnegative(record.get("source_cell_px"))
        if cell > 0.0:
            result[axis] = cell
    return result


def _finite_nonnegative(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number >= 0.0 else 0.0


def _finite_unit_interval(value: Any) -> float:
    number = _finite_nonnegative(value)
    return number if number <= 1.0 else 0.0


def _observe_parent_geometry(
    *,
    image: np.ndarray,
    page_id: str,
    bundle_id: str,
    parent_id: str,
    root_id: str,
    view: Any,
    source_cell_calibration: Mapping[str, Any],
) -> SourcePunctuationGeometryEvidence:
    analysis_bbox = bbox_from_value(_value(view, "analysis_bbox", ()))
    x, y, width, height = analysis_bbox
    mask = _foreground_array(_value(view, "foreground_mask"))
    source_crop = np.ascontiguousarray(
        image[y : y + height, x : x + width], dtype=np.uint8
    )
    mask_crop = np.ascontiguousarray(
        mask[y : y + height, x : x + width] > 0
    )
    contrast_ink = _contrast_ink(source_crop, mask_crop)
    support_identity = {
        **_base_support_identity(
            page_id=page_id,
            bundle_id=bundle_id,
            parent_id=parent_id,
            root_id=root_id,
            view=view,
            source_cell_calibration=source_cell_calibration,
        ),
        "analysis_bbox_page_xywh": list(analysis_bbox),
        "authorized_foreground_mask_sha256": _array_sha256(
            np.ascontiguousarray(mask_crop, dtype=np.uint8)
        ),
        "source_pixel_crop_sha256": _array_sha256(source_crop),
        "contrast_ink_sha256": _array_sha256(
            np.ascontiguousarray(contrast_ink, dtype=np.uint8)
        ),
    }
    components = _connected_components(contrast_ink)
    mask_components = _connected_components(mask_crop)
    source_cells = _supported_source_cells(source_cell_calibration)
    occurrences: list[SourcePunctuationGeometryOccurrence] = []
    occurrences.extend(
        _straight_stroke_occurrences(
            contrast_ink,
            components=components,
            analysis_bbox=analysis_bbox,
            source_cells=source_cells,
        )
    )
    occurrences.extend(
        _wave_occurrences(
            contrast_ink,
            components=components,
            mask_components=mask_components,
            analysis_bbox=analysis_bbox,
            source_cells=source_cells,
            excluded_boxes=[
                item.group_bbox_local_xywh
                for item in occurrences
                if item.kind == "dash"
            ],
        )
    )
    occurrences.extend(
        _ellipsis_occurrences(
            components,
            analysis_bbox=analysis_bbox,
            source_cells=source_cells,
        )
    )
    occurrences = _assign_visual_ordinals(_deduplicate_occurrences(occurrences))
    status = "observed" if occurrences else "abstained"
    abstention = (
        ""
        if occurrences
        else "source_cell_calibration_unavailable"
        if not source_cells
        else "no_unambiguous_visual_punctuation"
    )
    reasons = (
        "parent_authorized_source_punctuation_observed",
        "ocr_punctuation_not_consulted",
        "native_source_cell_calibration_applied",
    )
    if not occurrences:
        reasons = (*reasons, abstention)
    return SourcePunctuationGeometryEvidence(
        page_id=page_id,
        bundle_id=bundle_id,
        parent_id=parent_id,
        root_id=root_id,
        status=status,
        view_id=str(_value(view, "view_id") or ""),
        support_identity=support_identity,
        occurrences=tuple(occurrences),
        abstention_reason=abstention,
        reason_codes=tuple(reasons),
    )


def _straight_stroke_occurrences(
    ink: np.ndarray,
    *,
    components: Sequence[Mapping[str, Any]],
    analysis_bbox: Sequence[int],
    source_cells: Mapping[str, float],
) -> list[SourcePunctuationGeometryOccurrence]:
    results: list[SourcePunctuationGeometryOccurrence] = []
    for axis in ("ttb", "ltr"):
        cell = float(source_cells.get(axis) or 0.0)
        if cell <= 0.0:
            continue
        runs = _longest_axis_runs(ink, axis)
        qualifying = [
            run
            for run in runs
            if float(run[2] - run[1]) >= cell * 1.50
        ]
        for group in _group_adjacent_runs(qualifying):
            cross_values = [item[0] for item in group]
            starts = [item[1] for item in group]
            ends = [item[2] for item in group]
            cross_start = min(cross_values)
            cross_end = max(cross_values) + 1
            inline_start = int(round(float(np.median(starts))))
            inline_end = int(round(float(np.median(ends))))
            if inline_end <= inline_start:
                continue
            span = float(inline_end - inline_start)
            cross_span = float(cross_end - cross_start)
            if cross_span > max(1.0, cell * 0.36):
                continue
            endpoint_spread = float(max(ends) - min(ends) + max(starts) - min(starts))
            if endpoint_spread > max(2.0, cell * 0.30):
                continue
            local_box = (
                (cross_start, inline_start, cross_end - cross_start, inline_end - inline_start)
                if axis == "ttb"
                else (inline_start, cross_start, inline_end - inline_start, cross_end - cross_start)
            )
            if _has_aligned_terminal_dot(
                local_box,
                components,
                axis=axis,
                source_cell=cell,
            ):
                continue
            normalized_span = span / cell
            confidence = min(
                0.995,
                0.72
                + min(0.18, max(0.0, normalized_span - 1.25) * 0.055)
                + min(0.08, len(group) * 0.015),
            )
            page_box = _page_box(local_box, analysis_bbox)
            results.append(
                SourcePunctuationGeometryOccurrence(
                    occurrence_id="",
                    kind="dash",
                    visual_reading_order_ordinal=-1,
                    kind_ordinal=-1,
                    inline_axis=axis,
                    component_bboxes_local_xywh=(tuple(local_box),),
                    component_bboxes_page_xywh=(tuple(page_box),),
                    group_bbox_local_xywh=tuple(local_box),
                    group_bbox_page_xywh=tuple(page_box),
                    span_px=span,
                    pitch_px=span,
                    source_cell_px=cell,
                    normalized_span=normalized_span,
                    normalized_pitch=normalized_span,
                    confidence=confidence,
                    reason_codes=(
                        "parallel_source_pixel_runs",
                        "source_span_independent_of_ocr_count",
                    ),
                )
            )
    return _prefer_axis_specific_nonoverlap(results)


def _wave_occurrences(
    ink: np.ndarray,
    *,
    components: Sequence[Mapping[str, Any]],
    mask_components: Sequence[Mapping[str, Any]],
    analysis_bbox: Sequence[int],
    source_cells: Mapping[str, float],
    excluded_boxes: Sequence[Sequence[int]],
) -> list[SourcePunctuationGeometryOccurrence]:
    results: list[SourcePunctuationGeometryOccurrence] = []
    candidates = list(components or [])
    if not candidates:
        candidates = list(mask_components or [])
    for component in candidates:
        box = tuple(int(value) for value in component.get("bbox_xywh", ()))
        if len(box) != 4 or any(_boxes_overlap(box, other) for other in excluded_boxes):
            continue
        x, y, width, height = box
        for axis in ("ttb", "ltr"):
            inline_span = float(height if axis == "ttb" else width)
            cross_span = float(width if axis == "ttb" else height)
            cell = float(source_cells.get(axis) or 0.0)
            if cell <= 0.0 or inline_span < cell * 1.25:
                continue
            if cross_span > cell * 0.80 or cross_span < 2.0:
                continue
            component_mask = np.zeros_like(ink, dtype=bool)
            pixels = list(component.get("pixels") or [])
            for px, py in pixels:
                if 0 <= int(py) < component_mask.shape[0] and 0 <= int(px) < component_mask.shape[1]:
                    component_mask[int(py), int(px)] = True
            track = _component_center_track(component_mask, box, axis)
            if len(track) < max(5, int(round(inline_span * 0.45))):
                continue
            centers = np.asarray([item[1] for item in track], dtype=float)
            smoothed = _smooth_track(centers)
            center_range = float(np.max(smoothed) - np.min(smoothed)) if len(smoothed) else 0.0
            turns = _direction_change_count(smoothed)
            if center_range < max(0.75, cell * 0.020) or turns < 2:
                continue
            pitch = inline_span / float(max(1, turns // 2 + 1))
            page_box = _page_box(box, analysis_bbox)
            results.append(
                SourcePunctuationGeometryOccurrence(
                    occurrence_id="",
                    kind="wave",
                    visual_reading_order_ordinal=-1,
                    kind_ordinal=-1,
                    inline_axis=axis,
                    component_bboxes_local_xywh=(box,),
                    component_bboxes_page_xywh=(tuple(page_box),),
                    group_bbox_local_xywh=box,
                    group_bbox_page_xywh=tuple(page_box),
                    span_px=inline_span,
                    pitch_px=pitch,
                    source_cell_px=cell,
                    normalized_span=inline_span / cell,
                    normalized_pitch=pitch / cell,
                    confidence=min(0.96, 0.74 + min(0.16, turns * 0.025)),
                    reason_codes=(
                        "continuous_source_centerline_oscillation",
                        "source_span_independent_of_ocr_count",
                    ),
                )
            )
            break
    return _prefer_axis_specific_nonoverlap(results)


def _ellipsis_occurrences(
    components: Sequence[Mapping[str, Any]],
    *,
    analysis_bbox: Sequence[int],
    source_cells: Mapping[str, float],
) -> list[SourcePunctuationGeometryOccurrence]:
    results: list[SourcePunctuationGeometryOccurrence] = []
    for axis in ("ttb", "ltr"):
        cell = float(source_cells.get(axis) or 0.0)
        if cell <= 0.0:
            continue
        dot_components: list[Mapping[str, Any]] = []
        for component in components:
            box = tuple(int(value) for value in component.get("bbox_xywh", ()))
            if len(box) != 4:
                continue
            _x, _y, comp_w, comp_h = box
            major = float(max(comp_w, comp_h))
            minor = float(min(comp_w, comp_h))
            fill_ratio = float(component.get("area") or 0) / float(max(1, comp_w * comp_h))
            if (
                major > cell * 0.44
                or minor < 1.0
                or major > minor * 2.5
                or fill_ratio < 0.20
            ):
                continue
            dot_components.append(component)
        if len(dot_components) < 3:
            continue
        groups = _collinear_dot_groups(dot_components, axis=axis)
        for group in groups:
            if len(group) < 3:
                continue
            boxes = [tuple(int(value) for value in item["bbox_xywh"]) for item in group]
            group_box = _union_boxes(boxes)
            if not group_box:
                continue
            if any(
                _dot_has_aligned_stem(box, components, axis=axis, source_cell=cell)
                for box in boxes
            ):
                continue
            inline_centers = [
                (box[1] + box[3] / 2.0) if axis == "ttb" else (box[0] + box[2] / 2.0)
                for box in boxes
            ]
            inline_centers.sort()
            pitches = [
                inline_centers[index + 1] - inline_centers[index]
                for index in range(len(inline_centers) - 1)
            ]
            if not pitches or min(pitches) <= 0.0:
                continue
            pitch = float(np.median(pitches))
            pitch_spread = float(max(pitches) - min(pitches))
            if pitch_spread > max(2.0, pitch * 0.38):
                continue
            span = float(group_box[3] if axis == "ttb" else group_box[2])
            page_boxes = tuple(
                tuple(_page_box(box, analysis_bbox)) for box in boxes
            )
            page_group = tuple(_page_box(group_box, analysis_bbox))
            results.append(
                SourcePunctuationGeometryOccurrence(
                    occurrence_id="",
                    kind="ellipsis",
                    visual_reading_order_ordinal=-1,
                    kind_ordinal=-1,
                    inline_axis=axis,
                    component_bboxes_local_xywh=tuple(boxes),
                    component_bboxes_page_xywh=page_boxes,
                    group_bbox_local_xywh=tuple(group_box),
                    group_bbox_page_xywh=page_group,
                    span_px=span,
                    pitch_px=pitch,
                    source_cell_px=cell,
                    normalized_span=span / cell,
                    normalized_pitch=pitch / cell,
                    confidence=max(
                        0.72,
                        min(0.98, 0.90 - pitch_spread / max(1.0, pitch) * 0.20),
                    ),
                    reason_codes=(
                        "collinear_source_dot_group",
                        "uniform_source_dot_pitch",
                        "source_dot_count_not_target_identity_authority",
                    ),
                )
            )
    return _prefer_axis_specific_nonoverlap(results)


def _authorized_view_rejection_reasons(
    view: Any,
    *,
    page_id: str,
    bundle_id: str,
    parent_id: str,
    root_id: str,
    image_shape: Sequence[int],
) -> list[str]:
    reasons: list[str] = []
    if view is None:
        return ["authorized_source_style_view_missing"]
    if str(_value(view, "page_id") or "") != page_id:
        reasons.append("authorized_source_style_view_page_identity_mismatch")
    if str(_value(view, "bundle_id") or "") != bundle_id:
        reasons.append("authorized_source_style_view_bundle_identity_mismatch")
    if str(_value(view, "parent_id") or "") != parent_id:
        reasons.append("authorized_source_style_view_parent_identity_mismatch")
    if str(_value(view, "root_id") or "") != root_id:
        reasons.append("authorized_source_style_view_root_identity_mismatch")
    if not bool(_value(view, "available", False)):
        reasons.append("authorized_source_style_view_unavailable")
    analysis_bbox = bbox_from_value(_value(view, "analysis_bbox", ()))
    if not analysis_bbox:
        reasons.append("authorized_source_style_view_analysis_bbox_invalid")
    mask = _foreground_array(_value(view, "foreground_mask"))
    if mask is None:
        reasons.append("authorized_source_style_view_foreground_invalid")
    elif tuple(mask.shape[:2]) != tuple(image_shape[:2]):
        reasons.append("authorized_source_style_view_foreground_shape_mismatch")
    elif int(np.count_nonzero(mask)) <= 0:
        reasons.append("authorized_source_style_view_foreground_empty")
    if analysis_bbox and image_shape:
        x, y, width, height = analysis_bbox
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > int(image_shape[1])
            or y + height > int(image_shape[0])
        ):
            reasons.append("authorized_source_style_view_analysis_bbox_outside_source")
    return _unique_strings(reasons)


def _base_support_identity(
    *,
    page_id: str,
    bundle_id: str,
    parent_id: str,
    root_id: str,
    view: Any,
    source_cell_calibration: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract_version": SOURCE_PUNCTUATION_GEOMETRY_SUPPORT_VERSION,
        "authorized_source_style_view_version": AUTHORIZED_SOURCE_STYLE_VIEW_VERSION,
        "page_id": str(page_id or ""),
        "bundle_id": str(bundle_id or ""),
        "parent_id": str(parent_id or ""),
        "root_id": str(root_id or ""),
        "view_id": str(_value(view, "view_id") or ""),
        "cleanup_mask_ids": _strings(_value(view, "cleanup_mask_ids", ())),
        "owned_component_ids": _strings(
            _value(view, "owned_component_ids", ())
        ),
        "pixel_projection_owner": "CleanupMask",
        "geometry_observer_owner": "SourcePunctuationGeometryEvidence",
        "source_cell_calibration": copy_jsonish(source_cell_calibration),
    }


@lru_cache(maxsize=4)
def _cached_grayscale_source_image(path: str) -> np.ndarray | None:
    if not path:
        return None
    try:
        from PIL import Image

        with Image.open(path) as source:
            array = np.ascontiguousarray(source.convert("L"), dtype=np.uint8)
        array.setflags(write=False)
        return array
    except Exception:
        return None


def _contrast_ink(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    foreground = np.asarray(mask, dtype=bool)
    if source.ndim != 2 or foreground.shape != source.shape or not np.any(foreground):
        return np.zeros_like(foreground, dtype=bool)
    exterior_values = source[~foreground]
    if exterior_values.size:
        background = float(np.median(exterior_values))
    else:
        masked_values = source[foreground]
        background = 255.0 if float(np.median(masked_values)) < 127.5 else 0.0
    contrast = np.abs(source.astype(np.float32) - background)
    supported = contrast[foreground]
    high = float(np.percentile(supported, 90)) if supported.size else 0.0
    threshold = max(8.0, min(64.0, high * 0.25))
    result = foreground & (contrast >= threshold)
    if int(np.count_nonzero(result)) <= 0:
        return foreground.copy()
    return result


def _foreground_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.ndim == 3:
        array = np.max(array, axis=2)
    if array.ndim != 2 or array.size <= 0:
        return None
    return np.ascontiguousarray(array > 0, dtype=bool)


def _connected_components(mask: np.ndarray) -> list[dict[str, Any]]:
    ink = np.asarray(mask, dtype=bool)
    height, width = ink.shape
    seen = np.zeros_like(ink, dtype=bool)
    components: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if not ink[y, x] or seen[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(x, y)])
            seen[y, x] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                current_x, current_y = queue.popleft()
                pixels.append((current_x, current_y))
                for next_y in range(
                    max(0, current_y - 1), min(height, current_y + 2)
                ):
                    for next_x in range(
                        max(0, current_x - 1), min(width, current_x + 2)
                    ):
                        if (
                            ink[next_y, next_x]
                            and not seen[next_y, next_x]
                        ):
                            seen[next_y, next_x] = True
                            queue.append((next_x, next_y))
            xs = [item[0] for item in pixels]
            ys = [item[1] for item in pixels]
            box = (
                min(xs),
                min(ys),
                max(xs) - min(xs) + 1,
                max(ys) - min(ys) + 1,
            )
            components.append(
                {
                    "bbox_xywh": box,
                    "area": len(pixels),
                    "pixels": tuple(pixels),
                }
            )
    return sorted(
        components,
        key=lambda item: (
            int(item["bbox_xywh"][1]),
            int(item["bbox_xywh"][0]),
        ),
    )


def _longest_axis_runs(
    mask: np.ndarray,
    axis: str,
) -> list[tuple[int, int, int]]:
    values = np.asarray(mask, dtype=bool)
    cross_limit = values.shape[1] if axis == "ttb" else values.shape[0]
    runs: list[tuple[int, int, int]] = []
    for cross in range(cross_limit):
        line = values[:, cross] if axis == "ttb" else values[cross, :]
        indices = np.flatnonzero(line)
        if not len(indices):
            continue
        start = previous = int(indices[0])
        best = (start, previous + 1)
        for value in indices[1:]:
            value = int(value)
            if value > previous + 1:
                if previous + 1 - start > best[1] - best[0]:
                    best = (start, previous + 1)
                start = value
            previous = value
        if previous + 1 - start > best[1] - best[0]:
            best = (start, previous + 1)
        runs.append((cross, best[0], best[1]))
    return runs


def _group_adjacent_runs(
    runs: Sequence[tuple[int, int, int]],
) -> list[list[tuple[int, int, int]]]:
    groups: list[list[tuple[int, int, int]]] = []
    for run in sorted(runs, key=lambda item: item[0]):
        if not groups or run[0] > groups[-1][-1][0] + 1:
            groups.append([run])
        else:
            groups[-1].append(run)
    return groups


def _has_aligned_terminal_dot(
    line_box: Sequence[int],
    components: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    source_cell: float,
) -> bool:
    x, y, width, height = [int(value) for value in line_box]
    line_cross_center = (
        float(x) + float(width) / 2.0
        if axis == "ttb"
        else float(y) + float(height) / 2.0
    )
    inline_start = y if axis == "ttb" else x
    inline_end = y + height if axis == "ttb" else x + width
    for component in components:
        box = tuple(int(value) for value in component.get("bbox_xywh", ()))
        if len(box) != 4 or _boxes_overlap(line_box, box):
            continue
        bx, by, bw, bh = box
        if max(bw, bh) > source_cell * 0.46:
            continue
        cross_center = (
            float(bx) + float(bw) / 2.0
            if axis == "ttb"
            else float(by) + float(bh) / 2.0
        )
        if abs(cross_center - line_cross_center) > source_cell * 0.28:
            continue
        dot_start = by if axis == "ttb" else bx
        dot_end = by + bh if axis == "ttb" else bx + bw
        gap = min(abs(dot_start - inline_end), abs(inline_start - dot_end))
        if gap <= source_cell * 0.80:
            return True
    return False


def _component_center_track(
    component_mask: np.ndarray,
    box: Sequence[int],
    axis: str,
) -> list[tuple[float, float]]:
    x, y, width, height = [int(value) for value in box]
    track: list[tuple[float, float]] = []
    if axis == "ttb":
        for inline in range(y, y + height):
            crosses = np.flatnonzero(component_mask[inline, x : x + width])
            if len(crosses):
                track.append((float(inline), float(x) + float(np.mean(crosses))))
    else:
        for inline in range(x, x + width):
            crosses = np.flatnonzero(component_mask[y : y + height, inline])
            if len(crosses):
                track.append((float(inline), float(y) + float(np.mean(crosses))))
    return track


def _smooth_track(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values
    window = min(5, len(values))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="valid")


def _direction_change_count(values: np.ndarray) -> int:
    if len(values) < 3:
        return 0
    differences = np.diff(values)
    tolerance = max(0.15, float(np.max(values) - np.min(values)) * 0.025)
    signs: list[int] = []
    for value in differences:
        if abs(float(value)) <= tolerance:
            continue
        sign = 1 if value > 0 else -1
        if not signs or signs[-1] != sign:
            signs.append(sign)
    return max(0, len(signs) - 1)


def _collinear_dot_groups(
    components: Sequence[Mapping[str, Any]],
    *,
    axis: str,
) -> list[list[Mapping[str, Any]]]:
    records = []
    for component in components:
        x, y, width, height = [int(value) for value in component["bbox_xywh"]]
        cross = x + width / 2.0 if axis == "ttb" else y + height / 2.0
        inline = y + height / 2.0 if axis == "ttb" else x + width / 2.0
        size = max(width, height)
        records.append((cross, inline, float(size), component))
    if not records:
        return []
    typical = max(1.0, float(np.median([item[2] for item in records])))
    groups: list[list[tuple[float, float, float, Mapping[str, Any]]]] = []
    for record in sorted(records, key=lambda item: (item[0], item[1])):
        target = None
        for group in groups:
            center = float(np.median([item[0] for item in group]))
            if abs(record[0] - center) <= typical * 0.65:
                target = group
                break
        if target is None:
            groups.append([record])
        else:
            target.append(record)
    return [
        [item[3] for item in sorted(group, key=lambda item: item[1])]
        for group in groups
        if len(group) >= 3
    ]


def _dot_has_aligned_stem(
    dot_box: Sequence[int],
    components: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    source_cell: float,
) -> bool:
    x, y, width, height = [int(value) for value in dot_box]
    cross_center = (
        x + width / 2.0 if axis == "ttb" else y + height / 2.0
    )
    inline_center = (
        y + height / 2.0 if axis == "ttb" else x + width / 2.0
    )
    inline_start = y if axis == "ttb" else x
    inline_end = y + height if axis == "ttb" else x + width
    cross_start = x if axis == "ttb" else y
    cross_end = x + width if axis == "ttb" else y + height
    for component in components:
        box = tuple(int(value) for value in component.get("bbox_xywh", ()))
        if len(box) != 4 or tuple(dot_box) == box:
            continue
        bx, by, bw, bh = box
        component_inline = float(bh if axis == "ttb" else bw)
        component_cross = float(bw if axis == "ttb" else bh)
        component_cross_center = (
            bx + bw / 2.0 if axis == "ttb" else by + bh / 2.0
        )
        component_inline_center = (
            by + bh / 2.0 if axis == "ttb" else bx + bw / 2.0
        )
        component_start = by if axis == "ttb" else bx
        component_end = by + bh if axis == "ttb" else bx + bw
        if (
            component_inline
            >= max(source_cell * 0.75, component_cross * 2.0)
            and abs(component_cross_center - cross_center)
            <= source_cell * 0.28
        ):
            gap = min(
                abs(component_start - inline_end),
                abs(inline_start - component_end),
            )
            if gap <= source_cell * 0.90:
                return True

        component_cross_start = bx if axis == "ttb" else by
        component_cross_end = bx + bw if axis == "ttb" else by + bh
        if (
            component_cross
            >= max(source_cell * 0.75, component_inline * 2.0)
            and abs(component_inline_center - inline_center)
            <= source_cell * 0.28
        ):
            cross_gap = min(
                abs(component_cross_start - cross_end),
                abs(cross_start - component_cross_end),
            )
            if cross_gap <= source_cell * 0.90:
                return True
    return False


def _assign_visual_ordinals(
    values: Sequence[SourcePunctuationGeometryOccurrence],
) -> list[SourcePunctuationGeometryOccurrence]:
    ordered = sorted(values, key=_visual_reading_key)
    kind_counts: dict[str, int] = {}
    result: list[SourcePunctuationGeometryOccurrence] = []
    for ordinal, item in enumerate(ordered):
        kind_ordinal = kind_counts.get(item.kind, 0)
        kind_counts[item.kind] = kind_ordinal + 1
        result.append(
            replace(
                item,
                occurrence_id=f"source_{item.kind}_{kind_ordinal:04d}",
                visual_reading_order_ordinal=ordinal,
                kind_ordinal=kind_ordinal,
            )
        )
    return result


def _visual_reading_key(
    item: SourcePunctuationGeometryOccurrence,
) -> tuple[float, ...]:
    x, y, width, height = item.group_bbox_page_xywh
    center_x = float(x) + float(width) / 2.0
    center_y = float(y) + float(height) / 2.0
    if item.inline_axis == "ttb":
        return (0.0, -center_x, center_y, item.kind)
    return (1.0, center_y, center_x, item.kind)


def _deduplicate_occurrences(
    values: Sequence[SourcePunctuationGeometryOccurrence],
) -> list[SourcePunctuationGeometryOccurrence]:
    kept: list[SourcePunctuationGeometryOccurrence] = []
    for item in sorted(values, key=lambda value: value.confidence, reverse=True):
        duplicate = False
        for current in kept:
            if (
                item.kind == current.kind
                and item.inline_axis == current.inline_axis
                and _box_iou(
                    item.group_bbox_local_xywh,
                    current.group_bbox_local_xywh,
                )
                >= 0.62
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def _prefer_axis_specific_nonoverlap(
    values: Sequence[SourcePunctuationGeometryOccurrence],
) -> list[SourcePunctuationGeometryOccurrence]:
    return _deduplicate_occurrences(values)


def _page_box(
    local_box: Sequence[int],
    analysis_bbox: Sequence[int],
) -> tuple[int, int, int, int]:
    x, y, width, height = [int(value) for value in local_box]
    page_x, page_y = int(analysis_bbox[0]), int(analysis_bbox[1])
    return (page_x + x, page_y + y, width, height)


def _union_boxes(
    values: Sequence[Sequence[int]],
) -> tuple[int, int, int, int] | tuple[()]:
    boxes = [bbox_from_value(value) for value in values]
    boxes = [box for box in boxes if box]
    if not boxes:
        return ()
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return (left, top, right - left, bottom - top)


def _boxes_overlap(first: Sequence[int], second: Sequence[int]) -> bool:
    return _intersection_area(first, second) > 0.0


def _box_iou(first: Sequence[int], second: Sequence[int]) -> float:
    intersection = _intersection_area(first, second)
    if intersection <= 0.0:
        return 0.0
    first_area = float(max(0, int(first[2])) * max(0, int(first[3])))
    second_area = float(max(0, int(second[2])) * max(0, int(second[3])))
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _intersection_area(first: Sequence[int], second: Sequence[int]) -> float:
    if len(first) != 4 or len(second) != 4:
        return 0.0
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(
        float(first[0]) + float(first[2]),
        float(second[0]) + float(second[2]),
    )
    bottom = min(
        float(first[1]) + float(first[3]),
        float(second[1]) + float(second[3]),
    )
    return max(0.0, right - left) * max(0.0, bottom - top)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype=np.uint8).tobytes()
    ).hexdigest()


def _freeze_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_jsonish(item)
                for key, item in sorted(
                    value.items(), key=lambda pair: str(pair[0])
                )
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_jsonish(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _thaw_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_jsonish(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_jsonish(item) for item in value]
    return value


def _value(source: Any, key: str, default: Any = "") -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _strings(values: Any) -> list[str]:
    if isinstance(values, (str, bytes)):
        values = [values]
    if not isinstance(values, Sequence):
        return []
    return _unique_strings(str(value or "") for value in values)


def _unique_strings(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result
