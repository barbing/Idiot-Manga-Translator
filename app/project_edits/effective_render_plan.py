# -*- coding: utf-8 -*-
"""Mechanical projection from effective edits to renderer layer plans.

This module does not observe style, choose typography, or mutate an automatic
``ParentExecutionBundle``.  It clones the renderer's accepted automatic plan
and applies only the explicitly supported user-owned fields.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from app.render.parent_layer_effects import PARENT_LAYER_EFFECTS_VERSION
from app.render.render_layer_adapter import (
    build_render_layer_plans_from_parent_bundles,
)
from app.render.typesetting_contracts import RenderLayerPlan

from .contracts import (
    REGISTERED_RENDER_FONT_ROLES,
    canonical_render_box,
    canonical_render_fill_color,
    canonical_render_font_weight_tier,
    canonical_render_outline_color,
    canonical_render_outline_width,
    canonical_render_preferred_size,
    canonical_render_shadow_blur,
    canonical_render_shadow_color,
    canonical_render_shadow_offset,
    freeze_json,
    thaw_json,
)
from .fingerprints import canonical_sha256
from .projection import (
    EffectivePageSnapshot,
    EffectiveParentSnapshot,
    TargetFreshness,
    automatic_ordered_parent_ids_for_page,
    automatic_parent_fingerprint,
    cleaned_base_erasure_membership,
)


EFFECTIVE_RENDER_PLAN_VERSION = "effective_render_layer_plan_v2"

_REGISTERED_FONT_ROLES = REGISTERED_RENDER_FONT_ROLES
_UNSUPPORTED_STYLE_FIELDS = frozenset(
    {
        # The current renderer selects registered roles, not arbitrary face IDs.
        "font_face",
        # The v3 interval is source-quality evidence, not a hard user fit band.
        "minimum_size",
        "maximum_size",
        "shadow_spread",
    }
)
_UNSUPPORTED_LAYOUT_FIELDS = frozenset(
    {
        # Alignment is currently audit-only in the renderer.
        "alignment",
        "letter_spacing",
        "column_spacing",
        "run_spacing",
        "break_hints",
    }
)


class EffectiveRenderPlanError(ValueError):
    """An effective page cannot form a renderer-consumable plan."""


class MissingCleanedPageBaseError(EffectiveRenderPlanError):
    """The selected page has no valid immutable render substrate."""


@dataclass(frozen=True, slots=True)
class EffectiveRenderLayerPlan:
    page_id: str
    parent_id: str
    layer_id: str
    base_plan_fingerprint: str
    override_fingerprint: str
    effective_plan_fingerprint: str
    plan_payload: Any
    field_authority: tuple[tuple[str, str], ...]
    applied_override_ids: tuple[str, ...]
    applied_edit_ids: tuple[str, ...]
    validation_issues: tuple[str, ...] = ()

    def to_render_layer_plan(self) -> RenderLayerPlan:
        return render_layer_plan_from_payload(thaw_json(self.plan_payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_render_plan_version": EFFECTIVE_RENDER_PLAN_VERSION,
            "page_id": self.page_id,
            "parent_id": self.parent_id,
            "layer_id": self.layer_id,
            "base_plan_fingerprint": self.base_plan_fingerprint,
            "override_fingerprint": self.override_fingerprint,
            "effective_plan_fingerprint": self.effective_plan_fingerprint,
            "plan": thaw_json(self.plan_payload),
            "field_authority": dict(self.field_authority),
            "applied_override_ids": list(self.applied_override_ids),
            "applied_edit_ids": list(self.applied_edit_ids),
            "validation_issues": list(self.validation_issues),
        }


def render_layer_plan_payload(plan: RenderLayerPlan) -> dict[str, Any]:
    payload = plan.to_audit_dict()
    payload.pop("render_layer_plan_version", None)
    return payload


def render_layer_plan_from_payload(value: Mapping[str, Any]) -> RenderLayerPlan:
    payload = dict(value)
    payload.pop("render_layer_plan_version", None)
    return RenderLayerPlan(**payload)


def project_effective_render_layers(
    snapshot: EffectivePageSnapshot,
    automatic_parent_bundles: Sequence[Any],
) -> tuple[EffectiveRenderLayerPlan, ...]:
    """Compile one immutable effective layer set without changing base data."""

    cleaned = thaw_json(snapshot.cleaned_page_base)
    if (
        not snapshot.cleaned_base_revision_id
        or not isinstance(cleaned, Mapping)
        or not bool(cleaned.get("valid"))
        or not str(cleaned.get("asset") or "").strip()
        or not _is_sha256(cleaned.get("content_sha256"))
    ):
        raise MissingCleanedPageBaseError(
            "a valid selected CleanedPageBase revision is required"
        )

    if snapshot.issues:
        kinds = sorted({issue.kind.value for issue in snapshot.issues})
        raise EffectiveRenderPlanError(
            "effective page has unresolved projection issues: " + ",".join(kinds)
        )

    bundles: dict[str, Any] = {}
    bundle_records: dict[str, Mapping[str, Any]] = {}
    for bundle in automatic_parent_bundles or ():
        record = _bundle_record(bundle)
        page_id = str(record.get("page_id") or "").strip()
        parent_id = str(record.get("parent_id") or "").strip()
        if page_id != snapshot.page_id or not parent_id:
            raise EffectiveRenderPlanError(
                "automatic parent identity does not match the effective page"
            )
        if parent_id in bundles:
            raise EffectiveRenderPlanError(
                f"automatic parent identity is duplicated: {parent_id}"
            )
        bundles[parent_id] = record
        bundle_records[parent_id] = record

    source_parent_ids_by_effective_parent: dict[str, tuple[str, ...]] = {}
    claimed_automatic_parent_ids: set[str] = set()
    for parent in snapshot.parents:
        mapping = parent.source_evidence_mapping
        source_parent_ids = (
            mapping.source_parent_ids if mapping is not None else (parent.parent_id,)
        )
        if claimed_automatic_parent_ids.intersection(source_parent_ids):
            raise EffectiveRenderPlanError(
                "effective parents contain overlapping automatic source mappings"
            )
        claimed_automatic_parent_ids.update(source_parent_ids)
        source_parent_ids_by_effective_parent[parent.parent_id] = source_parent_ids
    if set(bundle_records) != claimed_automatic_parent_ids:
        missing = sorted(claimed_automatic_parent_ids - set(bundle_records))
        extra = sorted(set(bundle_records) - claimed_automatic_parent_ids)
        raise EffectiveRenderPlanError(
            "automatic parent set does not match the effective snapshot"
            f"; missing={missing}; extra={extra}"
        )

    effective_ordered_parent_ids = tuple(
        parent.parent_id for parent in snapshot.parents
    )
    if effective_ordered_parent_ids != snapshot.hierarchy.ordered_parent_ids:
        raise EffectiveRenderPlanError(
            "effective parent order does not match the hierarchy snapshot"
        )
    try:
        automatic_ordered_parent_ids = automatic_ordered_parent_ids_for_page(
            {
                "page_id": snapshot.page_id,
                "parent_execution_bundles": tuple(bundle_records.values()),
            }
        )
    except (TypeError, ValueError) as exc:
        raise EffectiveRenderPlanError(
            "automatic parent reading order is invalid"
        ) from exc
    automatic_position = {
        parent_id: index
        for index, parent_id in enumerate(automatic_ordered_parent_ids)
    }
    natural_effective_order = tuple(
        sorted(
            effective_ordered_parent_ids,
            key=lambda parent_id: min(
                automatic_position[source_parent_id]
                for source_parent_id in source_parent_ids_by_effective_parent[
                    parent_id
                ]
            ),
        )
    )
    natural_draw_slots = tuple(
        _automatic_reading_order_value(
            bundle_records[
                min(
                    source_parent_ids_by_effective_parent[parent_id],
                    key=lambda source_parent_id: automatic_position[source_parent_id],
                )
            ]
        )
        for parent_id in natural_effective_order
    )
    effective_draw_order = {
        parent_id: natural_draw_slots[index]
        for index, parent_id in enumerate(effective_ordered_parent_ids)
    }
    reading_order_affected_parent_ids = frozenset(
        parent_id
        for index, parent_id in enumerate(effective_ordered_parent_ids)
        if natural_effective_order[index] != parent_id
    )
    if any(
        parent.reading_order
        != effective_ordered_parent_ids.index(parent.parent_id)
        for parent in snapshot.parents
    ):
        raise EffectiveRenderPlanError(
            "effective reading-order ordinals do not match the full permutation"
        )

    for parent in snapshot.parents:
        if not parent.excluded:
            continue
        for source_parent_id in source_parent_ids_by_effective_parent[parent.parent_id]:
            actual_erasure = cleaned_base_erasure_membership(
                cleaned,
                bundle_records[source_parent_id],
            )
            if actual_erasure is None or actual_erasure:
                raise EffectiveRenderPlanError(
                    "excluded parent is incompatible with the selected "
                    f"CleanedPageBase: {parent.parent_id}"
                )

    for parent in snapshot.parents:
        if parent.source_evidence_mapping is not None:
            if parent.source_evidence_mapping.page_id != snapshot.page_id:
                raise EffectiveRenderPlanError("mapped parent page identity changed")
            _require_exact_source_mapping(parent, bundle_records)
            if not parent.excluded:
                for source_parent_id in parent.source_evidence_mapping.source_parent_ids:
                    if cleaned_base_erasure_membership(
                        cleaned,
                        bundle_records[source_parent_id],
                    ) is not True:
                        raise EffectiveRenderPlanError(
                            "mapped source is not erased by the selected "
                            f"CleanedPageBase: {source_parent_id}"
                        )
        else:
            bundle = bundles[parent.parent_id]
            record = bundle_records[parent.parent_id]
            if automatic_parent_fingerprint(record) != parent.automatic_fingerprint:
                raise EffectiveRenderPlanError(
                    f"automatic parent fingerprint changed: {parent.parent_id}"
                )
            _require_supported_structural_state(parent, bundle)

    renderable_source_parent_ids = {
        source_parent_id
        for parent in snapshot.parents
        if not parent.excluded
        for source_parent_id in source_parent_ids_by_effective_parent[parent.parent_id]
    }
    renderable_bundles = tuple(
        bundles[parent_id]
        for parent_id in automatic_ordered_parent_ids
        if parent_id in renderable_source_parent_ids
    )
    automatic_plans = build_render_layer_plans_from_parent_bundles(
        page_id=snapshot.page_id,
        parent_execution_bundles=renderable_bundles,
        cleaned_page_base=_renderer_cleaned_base_ref(snapshot, cleaned),
    )
    plans_by_parent: dict[str, RenderLayerPlan] = {}
    for plan in automatic_plans:
        if plan.parent_id in plans_by_parent:
            raise EffectiveRenderPlanError(
                f"automatic render layer identity is duplicated: {plan.parent_id}"
            )
        if plan.parent_id not in bundles:
            raise EffectiveRenderPlanError(
                f"automatic render layer has no parent bundle: {plan.parent_id}"
            )
        plans_by_parent[plan.parent_id] = plan

    layers: list[EffectiveRenderLayerPlan] = []
    for parent in snapshot.parents:
        if parent.excluded:
            continue
        if parent.target_freshness is not TargetFreshness.CURRENT:
            raise EffectiveRenderPlanError(
                f"effective target text is not current: {parent.parent_id}"
            )
        if parent.source_evidence_mapping is not None:
            automatic_plan = _mapped_render_layer_plan(parent, plans_by_parent)
        else:
            bundle = bundles.get(parent.parent_id)
            if bundle is None:
                raise EffectiveRenderPlanError(
                    f"automatic parent is unavailable: {parent.parent_id}"
                )
            if not bool(_field(bundle, "render_required")):
                # A non-render-required automatic parent is not a render layer.
                continue
            automatic_plan = plans_by_parent.get(parent.parent_id)
            if automatic_plan is None:
                raise EffectiveRenderPlanError(
                    f"render-required automatic parent has no layer: {parent.parent_id}"
                )
        automatic_plan = replace(
            automatic_plan,
            cleaned_page_base_ref=_renderer_cleaned_base_ref(snapshot, cleaned),
        )
        base_payload = render_layer_plan_payload(automatic_plan)
        base_fingerprint = canonical_sha256(base_payload)
        reading_order_is_user_for_parent = (
            parent.parent_id in reading_order_affected_parent_ids
        )
        effective_plan, authorities = _apply_effective_parent(
            automatic_plan,
            parent,
            effective_draw_order=(
                effective_draw_order[parent.parent_id]
                if reading_order_is_user_for_parent
                else None
            ),
        )
        effective_payload = render_layer_plan_payload(effective_plan)
        override_body = {
            "style": dict(parent.render_style_overrides),
            "layout": dict(parent.render_layout_overrides),
            "structural": (
                {"draw_order": effective_draw_order[parent.parent_id]}
                if reading_order_is_user_for_parent
                else {}
            ),
        }
        override_fingerprint = canonical_sha256(override_body)
        effective_body = {
            "version": EFFECTIVE_RENDER_PLAN_VERSION,
            "base_plan_fingerprint": base_fingerprint,
            "override_fingerprint": override_fingerprint,
            "plan": effective_payload,
        }
        layers.append(
            EffectiveRenderLayerPlan(
                page_id=snapshot.page_id,
                parent_id=parent.parent_id,
                layer_id=effective_plan.layer_id,
                base_plan_fingerprint=base_fingerprint,
                override_fingerprint=override_fingerprint,
                effective_plan_fingerprint=canonical_sha256(effective_body),
                plan_payload=freeze_json(
                    effective_payload,
                    field_name="effective_render_plan",
                ),
                field_authority=tuple(sorted(authorities.items())),
                applied_override_ids=tuple(parent.render_override_edit_ids),
                applied_edit_ids=tuple(parent.applied_edit_ids),
            )
        )
    return tuple(layers)


def _apply_effective_parent(
    automatic_plan: RenderLayerPlan,
    parent: EffectiveParentSnapshot,
    *,
    effective_draw_order: int | None = None,
) -> tuple[RenderLayerPlan, dict[str, str]]:
    style_overrides = dict(parent.render_style_overrides)
    layout_overrides = dict(parent.render_layout_overrides)
    unsupported = sorted(
        (_UNSUPPORTED_STYLE_FIELDS & set(style_overrides))
        | (_UNSUPPORTED_LAYOUT_FIELDS & set(layout_overrides))
    )
    if unsupported:
        raise EffectiveRenderPlanError(
            "renderer does not support effective override fields: "
            + ",".join(unsupported)
        )

    style = deepcopy(dict(automatic_plan.resolved_render_style or {}))
    metadata = deepcopy(dict(automatic_plan.metadata or {}))
    authorities: dict[str, str] = {
        "translated_text": parent.target_authority,
        "target_box": "automatic",
        "writing_mode": "automatic",
        "draw_order": "user" if effective_draw_order is not None else "automatic",
        "role": "automatic",
    }
    for field in style:
        authorities[f"resolved_render_style.{field}"] = "automatic"

    if {
        "font_family",
        "font_role",
        "font_weight",
        "font_weight_tier",
    } & set(style_overrides):
        for field in (
            "font_family_role",
            "font_weight_tier",
            "primary_font_role",
            "primary_font_role_status",
        ):
            authorities[f"resolved_render_style.{field}"] = "user"
    if "preferred_size" in style_overrides:
        authorities["resolved_render_style.target_preferred_em_px"] = "user"
        authorities["resolved_render_style.target_fit_start_em_px"] = "user"
    if "fill_color" in style_overrides:
        authorities["resolved_render_style.fill"] = "user"
    if {"outline_color", "outline_width"} & set(style_overrides):
        authorities["resolved_render_style.outline"] = "user"
    if {
        "shadow_enabled",
        "shadow_color",
        "shadow_offset",
        "shadow_blur",
        "shadow_opacity",
    } & set(style_overrides) or "rotation" in layout_overrides:
        authorities["resolved_render_style.parent_layer_effects"] = "user"

    _apply_font_overrides(style, style_overrides)
    _apply_size_overrides(style, style_overrides)
    _apply_paint_overrides(style, style_overrides)
    _apply_effect_overrides(style, style_overrides, layout_overrides)

    target_box = list(automatic_plan.target_box)
    explicit_box = layout_overrides.get("render_box")
    if explicit_box is not None:
        target_box = list(
            canonical_render_box(explicit_box, field_name="effective render_box")
        )
    else:
        for index, field in enumerate(("x", "y", "width", "height")):
            if field in layout_overrides:
                target_box[index] = int(round(float(layout_overrides[field])))
    if {"render_box", "x", "y", "width", "height"} & set(layout_overrides):
        authorities["target_box"] = "user"
    if len(target_box) != 4 or target_box[2] <= 0 or target_box[3] <= 0:
        raise EffectiveRenderPlanError("effective render box is invalid")
    box_overridden = bool(
        {"render_box", "x", "y", "width", "height"} & set(layout_overrides)
    )
    editable_hard_bounds = list(
        metadata.get("editable_hard_bounds")
        or automatic_plan.hard_bounds
    )
    effective_hard_bounds = (
        editable_hard_bounds
        if box_overridden
        else list(automatic_plan.hard_bounds)
    )
    if effective_hard_bounds and not _contains_xywh(
        effective_hard_bounds,
        target_box,
    ):
        raise EffectiveRenderPlanError(
            "effective render box exceeds the editable hard bounds"
        )

    writing_mode = automatic_plan.writing_mode
    if "writing_mode" in layout_overrides:
        writing_mode = _renderer_writing_mode(layout_overrides["writing_mode"])
        style["writing_mode"] = writing_mode
        authorities["writing_mode"] = "user"
        authorities["resolved_render_style.writing_mode"] = "user"
    if "alignment" in layout_overrides:
        raise EffectiveRenderPlanError(
            "renderer does not support effective override field: alignment"
        )
    if "line_height" in layout_overrides:
        line_height = float(layout_overrides["line_height"])
        if line_height < 0.5:
            raise EffectiveRenderPlanError(
                "renderer line_height must be at least 0.5"
            )
        style["line_height"] = line_height
        authorities["resolved_render_style.line_height"] = "user"

    metadata["effective_render_plan"] = {
        "version": EFFECTIVE_RENDER_PLAN_VERSION,
        "base_plan_authority": "automatic_parent_execution_bundle",
        "target_authority": parent.target_authority,
        "field_authority": dict(sorted(authorities.items())),
        "applied_override_ids": list(parent.render_override_edit_ids),
        "automatic_style_mutated": False,
    }
    return (
        replace(
            automatic_plan,
            translated_text=parent.target_text,
            target_box=target_box,
            hard_bounds=effective_hard_bounds,
            resolved_render_style=style,
            writing_mode=writing_mode,
            draw_order=(
                effective_draw_order
                if effective_draw_order is not None
                else automatic_plan.draw_order
            ),
            metadata=metadata,
        ),
        authorities,
    )


def _apply_font_overrides(style: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    if not {
        "font_family",
        "font_role",
        "font_weight",
        "font_weight_tier",
    }.intersection(overrides):
        return
    family = str(style.get("font_family_role") or "sans")
    weight_tier = str(style.get("font_weight_tier") or "base")
    explicit_role = ""
    if "font_family" in overrides:
        value = str(overrides["font_family"]).strip()
        if value in {"sans", "serif"}:
            family = value
        else:
            raise EffectiveRenderPlanError(
                f"effective font family is not registered: {value}"
            )
    if "font_role" in overrides:
        explicit_role = str(overrides["font_role"]).strip()
    if "font_weight" in overrides:
        weight_tier = _weight_tier(int(overrides["font_weight"]))
    if "font_weight_tier" in overrides:
        try:
            weight_tier = canonical_render_font_weight_tier(
                overrides["font_weight_tier"]
            )
        except (TypeError, ValueError) as exc:
            raise EffectiveRenderPlanError(str(exc)) from exc
    requested_family = family
    requested_weight_tier = weight_tier
    role = explicit_role or _role_for(family, weight_tier)
    if role:
        if role not in _REGISTERED_FONT_ROLES:
            raise EffectiveRenderPlanError(
                f"effective font role is not registered: {role}"
            )
        selected_family, selected_weight_tier = _role_axes(role)
        if explicit_role and "font_family" not in overrides:
            requested_family = selected_family
        if explicit_role and not {
            "font_weight",
            "font_weight_tier",
        }.intersection(overrides):
            requested_weight_tier = selected_weight_tier
        style["font_family_role"] = requested_family
        style["font_weight_tier"] = requested_weight_tier
        style["primary_font_role"] = role
        style["primary_font_role_status"] = (
            "registered_role"
            if (
                selected_family == requested_family
                and selected_weight_tier == requested_weight_tier
            )
            else "degraded_registered_role"
        )


def _apply_size_overrides(style: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    if "preferred_size" in overrides:
        try:
            preferred = canonical_render_preferred_size(overrides["preferred_size"])
        except (TypeError, ValueError) as exc:
            raise EffectiveRenderPlanError(
                "preferred_size must be a finite number between 0.1 and 2048.0 pixels"
            ) from exc
        style["target_preferred_em_px"] = preferred
        style["target_fit_start_em_px"] = preferred


def _apply_paint_overrides(style: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    if "fill_color" in overrides:
        try:
            color = canonical_render_fill_color(overrides["fill_color"])
        except (TypeError, ValueError) as exc:
            raise EffectiveRenderPlanError(
                "fill_color must use the GUI's canonical opaque #RRGGBB contract"
            ) from exc
        style["fill"] = {
            **dict(style.get("fill") or {}),
            "color": color,
            "polarity": _color_polarity(color),
        }
    outline = dict(style.get("outline") or {})
    if "outline_color" in overrides:
        try:
            color = canonical_render_outline_color(overrides["outline_color"])
        except (TypeError, ValueError) as exc:
            raise EffectiveRenderPlanError(
                "outline_color must use the GUI's canonical opaque #RRGGBB contract"
            ) from exc
        outline["color"] = color
    if "outline_width" in overrides:
        try:
            width = canonical_render_outline_width(overrides["outline_width"])
        except (TypeError, ValueError) as exc:
            raise EffectiveRenderPlanError(
                "outline_width must be a finite number between 0.0 and 128.0 pixels"
            ) from exc
        outline["target_width_px"] = width
        outline["present"] = width > 0.0
        source_cell = style.get("source_visual_cell")
        source_median = (
            float(source_cell.get("median_px") or 0.0)
            if isinstance(source_cell, Mapping)
            else 0.0
        )
        outline["source_width_to_cell_ratio"] = (
            width / source_median if source_median > 0.0 else 0.0
        )
    if {"outline_color", "outline_width"} & set(overrides):
        style["outline"] = outline


def _apply_effect_overrides(
    style: dict[str, Any],
    style_overrides: Mapping[str, Any],
    layout_overrides: Mapping[str, Any],
) -> None:
    effect_fields = {
        "shadow_enabled",
        "shadow_color",
        "shadow_offset",
        "shadow_blur",
        "shadow_opacity",
    }
    effects = deepcopy(dict(style.get("parent_layer_effects") or {}))
    effects["contract_version"] = PARENT_LAYER_EFFECTS_VERSION
    if "rotation" in layout_overrides:
        effects["rotation"] = {
            "availability": "resolved",
            "degrees_clockwise": float(layout_overrides["rotation"]),
            "pivot": "visual_center",
        }
    if effect_fields & set(style_overrides):
        if (
            "shadow_enabled" in style_overrides
            and style_overrides["shadow_enabled"] is not False
        ):
            raise EffectiveRenderPlanError(
                "shadow_enabled supports only false; automatic shadows are restored by removing the override"
            )
        enabled = bool(style_overrides.get("shadow_enabled", True))
        if not enabled:
            effects["shadow"] = {"availability": "unavailable"}
        else:
            existing = dict(effects.get("shadow") or {})
            try:
                color = (
                    canonical_render_shadow_color(style_overrides["shadow_color"])
                    if "shadow_color" in style_overrides
                    else str(existing.get("color") or "#00000080")
                )
            except (TypeError, ValueError) as exc:
                raise EffectiveRenderPlanError(str(exc)) from exc
            if "shadow_opacity" in style_overrides:
                alpha = int(round(float(style_overrides["shadow_opacity"]) * 255.0))
                color = f"{color[:7]}{alpha:02X}"
            try:
                shadow_blur = canonical_render_shadow_blur(
                    style_overrides.get(
                        "shadow_blur",
                        existing.get("blur_radius_px") or 0.0,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise EffectiveRenderPlanError(str(exc)) from exc
            effects["shadow"] = {
                "availability": "resolved",
                "color": color,
                "offset_px": list(
                    canonical_render_shadow_offset(
                        style_overrides.get(
                            "shadow_offset",
                            existing.get("offset_px") or [0.0, 0.0],
                        )
                    )
                ),
                "blur_radius_px": shadow_blur,
            }
    if "rotation" in layout_overrides or effect_fields & set(style_overrides):
        effects.setdefault("rotation", {"availability": "unavailable"})
        effects.setdefault("shadow", {"availability": "unavailable"})
        style["parent_layer_effects"] = effects


def _role_for(family: str, tier: str) -> str:
    matrix = {
        ("sans", "slender"): "sans_regular",
        ("sans", "base"): "sans_medium",
        ("sans", "emphasis"): "sans_bold",
        ("sans", "heavy"): "sans_black",
        ("serif", "slender"): "serif_regular",
        ("serif", "base"): "serif_semibold",
        ("serif", "emphasis"): "serif_bold",
        ("serif", "heavy"): "serif_bold",
    }
    return matrix.get((family, tier), "")


def _role_axes(role: str) -> tuple[str, str]:
    values = {
        "sans_regular": ("sans", "slender"),
        "sans_medium": ("sans", "base"),
        "sans_bold": ("sans", "emphasis"),
        "sans_black": ("sans", "heavy"),
        "serif_regular": ("serif", "slender"),
        "serif_semibold": ("serif", "base"),
        "serif_bold": ("serif", "emphasis"),
    }
    return values[role]


def _weight_tier(weight: int) -> str:
    if weight <= 350:
        return "slender"
    if weight <= 550:
        return "base"
    if weight <= 750:
        return "emphasis"
    return "heavy"


def _renderer_writing_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"horizontal", "horizontal-tb"}:
        return "horizontal"
    if normalized in {"vertical", "vertical-rl"}:
        return "vertical"
    if normalized == "vertical-lr":
        raise EffectiveRenderPlanError(
            "renderer does not support vertical-lr column direction"
        )
    raise EffectiveRenderPlanError(f"unsupported writing mode: {normalized}")


def _color_polarity(value: str) -> str:
    raw = str(value or "#000000")[1:7]
    red, green, blue = (int(raw[index : index + 2], 16) for index in (0, 2, 4))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "light" if luminance >= 128.0 else "dark"


def _require_renderer_color(value: Any, field_name: str) -> str:
    color = str(value or "")
    if (
        len(color) not in {7, 9}
        or not color.startswith("#")
        or any(character not in "0123456789abcdefABCDEF" for character in color[1:])
    ):
        raise EffectiveRenderPlanError(
            f"{field_name} must use the renderer's #RRGGBB or #RRGGBBAA contract"
        )
    return color


def _renderer_cleaned_base_ref(
    snapshot: EffectivePageSnapshot,
    cleaned: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "revision_id": snapshot.cleaned_base_revision_id,
        "page_id": snapshot.page_id,
        "asset": str(cleaned.get("asset") or ""),
        "content_sha256": str(cleaned.get("content_sha256") or "").lower(),
        "provenance": snapshot.cleaned_base_provenance,
        "valid": True,
    }


def _require_exact_source_mapping(
    parent: EffectiveParentSnapshot,
    bundle_records: Mapping[str, Mapping[str, Any]],
) -> None:
    mapping = parent.source_evidence_mapping
    if mapping is None:
        raise EffectiveRenderPlanError("mapped parent has no source evidence")
    if parent.role not in set(mapping.source_roles):
        raise EffectiveRenderPlanError("mapped parent role changed")
    for index, source_parent_id in enumerate(mapping.source_parent_ids):
        record = bundle_records.get(source_parent_id)
        if record is None:
            raise EffectiveRenderPlanError(
                f"mapped automatic parent is unavailable: {source_parent_id}"
            )
        try:
            source_bbox = canonical_render_box(
                record.get("parent_bbox"),
                field_name="mapped source parent_bbox",
            )
        except (TypeError, ValueError) as exc:
            raise EffectiveRenderPlanError(
                f"mapped automatic parent geometry is invalid: {source_parent_id}"
            ) from exc
        if (
            str(record.get("bundle_id") or "")
            != mapping.source_bundle_ids[index]
            or str(record.get("root_id") or "")
            != mapping.source_root_ids[index]
            or automatic_parent_fingerprint(record)
            != mapping.source_automatic_fingerprints[index]
            or source_bbox != mapping.source_bboxes[index]
            or str(record.get("source_text") or "")
            != mapping.source_texts[index]
            or str(record.get("translated_text") or "")
            != mapping.source_target_texts[index]
            or canonical_sha256(
                {
                    "parent_id": source_parent_id,
                    "target_text": str(record.get("translated_text") or ""),
                }
            )
            != mapping.source_target_text_fingerprints[index]
            or _automatic_reading_order_value(record)
            != mapping.source_reading_orders[index]
            or str(record.get("role") or record.get("region_type") or "")
            != mapping.source_roles[index]
        ):
            raise EffectiveRenderPlanError(
                f"mapped automatic source evidence changed: {source_parent_id}"
            )


def _mapped_render_layer_plan(
    parent: EffectiveParentSnapshot,
    plans_by_parent: Mapping[str, RenderLayerPlan],
) -> RenderLayerPlan:
    mapping = parent.source_evidence_mapping
    if mapping is None:
        raise EffectiveRenderPlanError("mapped parent has no source evidence")
    source_plans = tuple(
        plans_by_parent.get(parent_id) for parent_id in mapping.source_parent_ids
    )
    if any(plan is None for plan in source_plans):
        raise EffectiveRenderPlanError(
            f"mapped render-required source has no layer: {parent.parent_id}"
        )
    exact_source_plans = tuple(plan for plan in source_plans if plan is not None)
    primary_index = mapping.source_parent_ids.index(
        mapping.primary_source_parent_id
    )
    primary = exact_source_plans[primary_index]
    if any(
        plan.role != primary.role
        or plan.writing_mode != primary.writing_mode
        or plan.resolved_render_style != primary.resolved_render_style
        or not plan.render_required
        for plan in exact_source_plans
    ):
        raise EffectiveRenderPlanError(
            "mapped automatic sources do not share one render style and writing mode"
        )
    try:
        hard_bounds = canonical_render_box(
            thaw_json(parent.workflow_area_bbox),
            field_name="mapped parent workflow_area_bbox",
        )
        target_box = canonical_render_box(
            thaw_json(parent.geometry)
            if thaw_json(parent.geometry) is not None
            else hard_bounds,
            field_name="mapped parent geometry",
        )
    except (TypeError, ValueError) as exc:
        raise EffectiveRenderPlanError(
            f"mapped parent geometry is invalid: {parent.parent_id}"
        ) from exc
    if not _contains_xywh(hard_bounds, target_box):
        raise EffectiveRenderPlanError(
            f"mapped parent geometry exceeds its evidence-backed scope: {parent.parent_id}"
        )
    mapping_record = mapping.to_dict()
    metadata = deepcopy(dict(primary.metadata or {}))
    metadata["effective_source_evidence_mapping"] = mapping_record
    metadata["source_layer_ids"] = [plan.layer_id for plan in exact_source_plans]
    clipping = {
        "authority": "effective_source_evidence_mapping",
        "render_allowed_area": list(hard_bounds),
        "root_bbox": list(hard_bounds),
        "source_clipping_region_refs": [
            deepcopy(dict(plan.clipping_region_ref or {}))
            for plan in exact_source_plans
        ],
    }
    source_text = " ".join(str(parent.source_text or "").split())
    if len(source_text) > 80:
        source_text = source_text[:77] + "..."
    layer_fingerprint = canonical_sha256(
        {
            "page_id": mapping.page_id,
            "parent_id": parent.parent_id,
            "mapping_fingerprint": mapping.fingerprint,
        }
    )
    return replace(
        primary,
        layer_id=f"effective_{layer_fingerprint[:32]}",
        parent_id=parent.parent_id,
        root_id=parent.root_id,
        translated_text="",
        source_text_summary=source_text,
        source_provenance_ref={
            "authority": "effective_source_evidence_mapping",
            "mapping": mapping_record,
        },
        target_box=list(target_box),
        hard_bounds=list(hard_bounds),
        clipping_region_ref=clipping,
        parent_execution_bundle_ref={
            "authority": "mapped_automatic_parent_execution_bundles",
            "primary_bundle_id": primary.bundle_id,
            "mapping": mapping_record,
            "source_bundle_refs": [
                deepcopy(dict(plan.parent_execution_bundle_ref or {}))
                for plan in exact_source_plans
            ],
        },
        role=parent.role,
        state="mapped",
        metadata=metadata,
    )


def _require_supported_structural_state(
    parent: EffectiveParentSnapshot,
    bundle: Any,
) -> None:
    if thaw_json(parent.geometry) != thaw_json(parent.automatic_geometry):
        raise EffectiveRenderPlanError(
            f"structural geometry requires upstream revalidation: {parent.parent_id}"
        )
    automatic_role = str(_field(bundle, "role") or _field(bundle, "region_type") or "")
    if parent.role != automatic_role:
        raise EffectiveRenderPlanError(
            f"structural role requires upstream revalidation: {parent.parent_id}"
        )


def _contains_xywh(outer: Sequence[Any], inner: Sequence[Any]) -> bool:
    if len(outer) != 4 or len(inner) != 4:
        return False
    ox, oy, ow, oh = (float(value) for value in outer)
    ix, iy, iw, ih = (float(value) for value in inner)
    return (
        ow > 0
        and oh > 0
        and iw > 0
        and ih > 0
        and ix >= ox
        and iy >= oy
        and ix + iw <= ox + ow
        and iy + ih <= oy + oh
    )


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _automatic_reading_order_value(record: Mapping[str, Any]) -> int:
    value = record.get("reading_order_index", record.get("reading_order"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EffectiveRenderPlanError(
            "automatic parent reading-order ordinal must be a non-negative integer"
        )
    return value


def _field(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _bundle_record(bundle: Any) -> Mapping[str, Any]:
    if isinstance(bundle, Mapping):
        return deepcopy(dict(bundle))
    # ParentExecutionBundle.to_audit_dict() normalizes its execution region in
    # place.  Hash a deep copy so effective projection never mutates immutable
    # automatic evidence supplied by the caller.
    bundle = deepcopy(bundle)
    converter = getattr(bundle, "to_audit_dict", None)
    if not callable(converter):
        converter = getattr(bundle, "to_dict", None)
    if not callable(converter):
        raise EffectiveRenderPlanError(
            "automatic parent bundle is not serializable"
        )
    record = converter()
    if not isinstance(record, Mapping):
        raise EffectiveRenderPlanError(
            "automatic parent bundle serialization is invalid"
        )
    return record
