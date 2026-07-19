# -*- coding: utf-8 -*-
"""Render-layer data contracts for the staged typesetting engine.

These contracts are inert records. They do not perform layout, draw text, run
cleanup, or reinterpret parent execution identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


RENDER_LAYER_PLAN_VERSION = "render_layer_plan_v2"
LOSSLESS_TEXT_TOKEN_VERSION = "lossless_text_token_v1"
PUNCTUATION_TOKEN_VERSION = "punctuation_token_v1"
DRAWING_PRIMITIVE_VERSION = "drawing_primitive_v1"
TYPESET_LAYOUT_VERSION = "typeset_layout_v3"
FIT_REPORT_VERSION = "fit_report_v2"


JsonDict = dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class TextToken:
    """One immutable translated-text token before or after presentation.

    ``original_text`` and its translated-text span are identity. Presentation
    fields may change by returning a new frozen record, but never replace or
    reinterpret that identity.
    """

    token_id: str
    original_text: str
    translated_start: int
    translated_end: int
    grapheme_start: int
    grapheme_end: int
    kind: str
    original_codepoints: tuple[str, ...] = ()
    writing_mode: str = "unresolved"
    presentation_text: str = ""
    presentation_codepoints: tuple[str, ...] = ()
    presentation_grapheme_start: int = 0
    presentation_grapheme_end: int = 0
    atomic_break: bool = False
    orientation_policy: str = "unresolved"
    render_policy: str = "resolved_font_glyph"
    presentation_reason: str = "identity"

    def to_audit_dict(self) -> JsonDict:
        return {
            "lossless_text_token_version": LOSSLESS_TEXT_TOKEN_VERSION,
            "token_type": "text",
            "token_id": self.token_id,
            "original_text": self.original_text,
            "translated_start": int(self.translated_start),
            "translated_end": int(self.translated_end),
            "grapheme_start": int(self.grapheme_start),
            "grapheme_end": int(self.grapheme_end),
            "kind": self.kind,
            "original_codepoints": list(self.original_codepoints),
            "writing_mode": self.writing_mode,
            "presentation_text": self.presentation_text,
            "presentation_codepoints": list(self.presentation_codepoints),
            "presentation_grapheme_start": int(self.presentation_grapheme_start),
            "presentation_grapheme_end": int(self.presentation_grapheme_end),
            "atomic_break": bool(self.atomic_break),
            "orientation_policy": self.orientation_policy,
            "render_policy": self.render_policy,
            "presentation_reason": self.presentation_reason,
        }


@dataclass(frozen=True, kw_only=True)
class PunctuationToken(TextToken):
    """Lossless punctuation identity plus an attached presentation decision."""

    punctuation_kind: str
    source_class: str
    exact_multiplicity: int = 1
    unit_count: int = 1
    dot_count: int = 0
    sequence_group_count: int = 1
    candidate_forms: tuple[str, ...] = ()
    supported_forms: tuple[str, ...] = ()
    supported: bool = True

    def to_audit_dict(self) -> JsonDict:
        return {
            **super().to_audit_dict(),
            "punctuation_token_version": PUNCTUATION_TOKEN_VERSION,
            "token_type": "punctuation",
            "punctuation_kind": self.punctuation_kind,
            "source_class": self.source_class,
            "exact_multiplicity": int(self.exact_multiplicity),
            "unit_count": int(self.unit_count),
            "dot_count": int(self.dot_count),
            "sequence_group_count": int(self.sequence_group_count),
            "candidate_forms": list(self.candidate_forms),
            "supported_forms": list(self.supported_forms),
            "supported": bool(self.supported),
        }


@dataclass
class RenderLayerPlan:
    """Editable render-layer unit derived from one parent execution bundle."""

    page_id: str
    layer_id: str
    bundle_id: str
    parent_id: str
    root_id: str
    translated_text: str
    source_text_summary: str = ""
    source_provenance_ref: JsonDict = field(default_factory=dict)
    target_box: list[int] = field(default_factory=list)
    hard_bounds: list[int] = field(default_factory=list)
    clipping_region_ref: JsonDict = field(default_factory=dict)
    resolved_render_style: JsonDict = field(default_factory=dict)
    writing_mode: str = "auto"
    draw_order: int = 0
    editable: bool = True
    editability_flags: list[str] = field(default_factory=list)
    cleaned_page_base_ref: JsonDict = field(default_factory=dict)
    parent_execution_bundle_ref: JsonDict = field(default_factory=dict)
    role: str = ""
    state: str = ""
    render_required: bool = True
    metadata: JsonDict = field(default_factory=dict)

    def to_audit_dict(self) -> JsonDict:
        return {
            "render_layer_plan_version": RENDER_LAYER_PLAN_VERSION,
            "page_id": self.page_id,
            "layer_id": self.layer_id,
            "bundle_id": self.bundle_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "translated_text": self.translated_text,
            "source_text_summary": self.source_text_summary,
            "source_provenance_ref": copy_jsonish(self.source_provenance_ref),
            "target_box": list(self.target_box),
            "hard_bounds": list(self.hard_bounds),
            "clipping_region_ref": copy_jsonish(self.clipping_region_ref),
            "resolved_render_style": copy_jsonish(self.resolved_render_style),
            "writing_mode": self.writing_mode,
            "draw_order": int(self.draw_order),
            "editable": bool(self.editable),
            "editability_flags": list(self.editability_flags),
            "cleaned_page_base_ref": copy_jsonish(self.cleaned_page_base_ref),
            "parent_execution_bundle_ref": copy_jsonish(self.parent_execution_bundle_ref),
            "role": self.role,
            "state": self.state,
            "render_required": bool(self.render_required),
            "metadata": copy_jsonish(self.metadata),
        }


@dataclass
class GlyphPlacement:
    """One measured glyph or text run in a completed typeset layout."""

    text: str
    bbox: list[int] = field(default_factory=list)
    position: list[float] = field(default_factory=list)
    font_family: str = ""
    font_size: float | None = None
    advance: float | None = None
    writing_mode: str = "auto"
    metadata: JsonDict = field(default_factory=dict)

    def to_audit_dict(self) -> JsonDict:
        return {
            "text": self.text,
            "bbox": list(self.bbox),
            "position": list(self.position),
            "font_family": self.font_family,
            "font_size": self.font_size,
            "advance": self.advance,
            "writing_mode": self.writing_mode,
            "metadata": copy_jsonish(self.metadata),
        }


@dataclass(frozen=True, kw_only=True)
class DrawingPrimitive:
    """Final non-glyph geometry emitted by the typesetting owner.

    The compositor may paint this geometry but must not recompute its bounds,
    centers, pitch, diameter, line width, or path.
    """

    primitive_id: str
    kind: str
    source_text: str
    token_ids: list[str] = field(default_factory=list)
    orientation: str = "vertical"
    bounds: list[int] = field(default_factory=list)
    centers: list[list[float]] = field(default_factory=list)
    points: list[list[float]] = field(default_factory=list)
    diameter_px: float = 0.0
    line_width_px: float = 0.0
    pitch_px: float = 0.0
    unit_count: int = 1
    visible_count: int = 1
    sequence_group_count: int = 1
    metadata: JsonDict = field(default_factory=dict)

    def to_audit_dict(self) -> JsonDict:
        return {
            "drawing_primitive_version": DRAWING_PRIMITIVE_VERSION,
            "primitive_id": self.primitive_id,
            "kind": self.kind,
            "source_text": self.source_text,
            "token_ids": list(self.token_ids),
            "orientation": self.orientation,
            "bounds": list(self.bounds),
            "centers": copy_jsonish(self.centers),
            "points": copy_jsonish(self.points),
            "diameter_px": float(self.diameter_px),
            "line_width_px": float(self.line_width_px),
            "pitch_px": float(self.pitch_px),
            "unit_count": int(self.unit_count),
            "visible_count": int(self.visible_count),
            "sequence_group_count": int(self.sequence_group_count),
            "metadata": copy_jsonish(self.metadata),
        }


@dataclass
class TypesetLayout:
    """Deterministic layout output for one render layer."""

    page_id: str
    layer_id: str
    bundle_id: str
    parent_id: str
    root_id: str
    selected_font_face: str
    selected_font_size: float
    writing_mode: str
    lines: list[JsonDict] = field(default_factory=list)
    columns: list[JsonDict] = field(default_factory=list)
    glyphs: list[GlyphPlacement | JsonDict] = field(default_factory=list)
    drawing_primitives: list[DrawingPrimitive | JsonDict] = field(default_factory=list)
    punctuation_placements: list[JsonDict] = field(default_factory=list)
    symbol_placements: list[JsonDict] = field(default_factory=list)
    measured_bounds: list[int] = field(default_factory=list)
    visual_center: list[float] = field(default_factory=list)
    fit_status: str = "not_typeset"
    normalized_text: str = ""
    original_text: str = ""
    lossless_text_tokens: list[TextToken | PunctuationToken | JsonDict] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_audit_dict(self) -> JsonDict:
        return {
            "typeset_layout_version": TYPESET_LAYOUT_VERSION,
            "page_id": self.page_id,
            "layer_id": self.layer_id,
            "bundle_id": self.bundle_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "selected_font_face": self.selected_font_face,
            "selected_font_size": float(self.selected_font_size),
            "writing_mode": self.writing_mode,
            "lines": copy_jsonish(self.lines),
            "columns": copy_jsonish(self.columns),
            "glyphs": [_glyph_to_dict(item) for item in self.glyphs],
            "drawing_primitives": [
                _drawing_primitive_to_dict(item)
                for item in self.drawing_primitives
            ],
            "punctuation_placements": copy_jsonish(self.punctuation_placements),
            "symbol_placements": copy_jsonish(self.symbol_placements),
            "measured_bounds": list(self.measured_bounds),
            "visual_center": list(self.visual_center),
            "fit_status": self.fit_status,
            "normalized_text": self.normalized_text,
            "original_text": self.original_text,
            "lossless_text_tokens": [
                _text_token_to_dict(item) for item in self.lossless_text_tokens
            ],
            "metadata": copy_jsonish(self.metadata),
        }


@dataclass
class FitReport:
    """Fit and fallback evidence for one render layer."""

    page_id: str
    layer_id: str
    bundle_id: str
    parent_id: str
    root_id: str
    natural_fit_success: bool = False
    fallback_used: bool = False
    fallback_reason: str = ""
    scaling_used: float = 1.0
    overflow_risk: bool = False
    clipping_risk: bool = False
    clipped_region: list[int] = field(default_factory=list)
    full_text_placed: bool = False
    punctuation_normalization_applied: list[JsonDict] = field(default_factory=list)
    symbol_fallbacks: list[JsonDict] = field(default_factory=list)
    user_review_recommended: bool = False
    fit_status: str = "not_typeset"
    issues: list[str] = field(default_factory=list)
    lossless_text_tokens: list[TextToken | PunctuationToken | JsonDict] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_audit_dict(self) -> JsonDict:
        return {
            "fit_report_version": FIT_REPORT_VERSION,
            "page_id": self.page_id,
            "layer_id": self.layer_id,
            "bundle_id": self.bundle_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "natural_fit_success": bool(self.natural_fit_success),
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": self.fallback_reason,
            "scaling_used": float(self.scaling_used),
            "overflow_risk": bool(self.overflow_risk),
            "clipping_risk": bool(self.clipping_risk),
            "clipped_region": list(self.clipped_region),
            "full_text_placed": bool(self.full_text_placed),
            "punctuation_normalization_applied": copy_jsonish(self.punctuation_normalization_applied),
            "symbol_fallbacks": copy_jsonish(self.symbol_fallbacks),
            "user_review_recommended": bool(self.user_review_recommended),
            "fit_status": self.fit_status,
            "issues": list(self.issues),
            "lossless_text_tokens": [
                _text_token_to_dict(item) for item in self.lossless_text_tokens
            ],
            "metadata": copy_jsonish(self.metadata),
        }


def render_layer_plans_to_audit_dict(plans: Sequence[RenderLayerPlan]) -> list[JsonDict]:
    return [plan.to_audit_dict() for plan in plans or []]


def typeset_layouts_to_audit_dict(layouts: Sequence[TypesetLayout]) -> list[JsonDict]:
    return [layout.to_audit_dict() for layout in layouts or []]


def fit_reports_to_audit_dict(reports: Sequence[FitReport]) -> list[JsonDict]:
    return [report.to_audit_dict() for report in reports or []]


def validated_source_text_footprint_ref(
    plan: RenderLayerPlan,
) -> JsonDict:
    """Return the adapter-validated footprint bound to this exact plan.

    This is a consumer trust check, not a second footprint validator. The
    adapter remains responsible for schema, hash, and geometry validation.
    """

    provenance = (
        plan.source_provenance_ref
        if isinstance(plan.source_provenance_ref, Mapping)
        else {}
    )
    validation = provenance.get("source_text_footprint_validation")
    footprint = provenance.get("source_text_footprint")
    if not isinstance(validation, Mapping) or not isinstance(footprint, Mapping):
        return {}
    if str(validation.get("status") or "") != "accepted":
        return {}
    contract_version = str(footprint.get("contract_version") or "")
    fact_set_id = str(footprint.get("fact_set_id") or "")
    if (
        not contract_version
        or contract_version != str(validation.get("contract_version") or "")
        or not fact_set_id
        or fact_set_id != str(validation.get("fact_set_id") or "")
        or not fact_set_id.startswith(f"{contract_version}:")
    ):
        return {}
    identity = footprint.get("source_identity")
    if not isinstance(identity, Mapping):
        return {}
    expected_identity = {
        "page_id": plan.page_id,
        "bundle_id": plan.bundle_id,
        "parent_id": plan.parent_id,
        "root_id": plan.root_id,
    }
    if any(
        str(identity.get(key) or "") != str(expected or "")
        for key, expected in expected_identity.items()
    ):
        return {}
    return copy_jsonish(footprint)


def bbox_from_value(value: Any) -> list[int]:
    """Return a four-int bbox-like list when present; never invent a box."""

    if isinstance(value, Mapping):
        for key in ("bbox", "box", "target_box", "hard_bounds", "render_allowed_area"):
            bbox = bbox_from_value(value.get(key))
            if bbox:
                return bbox
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values: list[int] = []
        for item in list(value)[:4]:
            try:
                values.append(int(round(float(item))))
            except (TypeError, ValueError):
                return []
        if len(values) == 4:
            return values
    return []


def copy_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): copy_jsonish(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [copy_jsonish(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_audit_dict"):
        return copy_jsonish(value.to_audit_dict())
    return str(value)


def _glyph_to_dict(value: GlyphPlacement | JsonDict) -> JsonDict:
    if isinstance(value, GlyphPlacement):
        return value.to_audit_dict()
    if isinstance(value, Mapping):
        return copy_jsonish(value)
    return {"text": str(value)}


def _drawing_primitive_to_dict(
    value: DrawingPrimitive | JsonDict,
) -> JsonDict:
    if isinstance(value, DrawingPrimitive):
        return value.to_audit_dict()
    if isinstance(value, Mapping):
        return copy_jsonish(value)
    return {"primitive_id": "", "kind": str(value)}


def _text_token_to_dict(
    value: TextToken | PunctuationToken | JsonDict,
) -> JsonDict:
    if isinstance(value, (TextToken, PunctuationToken)):
        return value.to_audit_dict()
    if isinstance(value, Mapping):
        return copy_jsonish(value)
    return {"token_id": "", "original_text": str(value)}
