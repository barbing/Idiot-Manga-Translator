# -*- coding: utf-8 -*-
"""Canonical page execution for parent-bundle render layers.

This module sequences existing owners. RenderLayoutPlanner owns page slots,
TypesettingEngine owns text layout, InkBoundLayoutFitter owns bounded shaped-ink
translation, and RendererCompositor owns raster/composition. No legacy renderer
or upstream pipeline module is imported here.
"""
from __future__ import annotations

import os
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.render.compositor import RENDERER_COMPOSITOR_VERSION, RendererCompositor
from app.render.font_manager import FontManager
from app.render.glyph_rasterizer import GLYPH_RASTER_AUTHORITY
from app.render.ink_bound_layout_fitter import InkBoundFitResult, InkBoundLayoutFitter
from app.render.layout_planner import RENDER_LAYOUT_PLANNER_VERSION, RenderLayoutPlanner
from app.render.parent_layer_effects import (
    PARENT_LAYER_EFFECTS_VERSION,
    resolve_parent_layer_effects,
)
from app.render.typesetting_contracts import (
    FitReport,
    RenderLayerPlan,
    TypesetLayout,
    copy_jsonish,
    fit_reports_to_audit_dict,
    render_layer_plans_to_audit_dict,
    typeset_layouts_to_audit_dict,
)
from app.render.typesetting_engine import TypesettingEngine

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None


PAGE_RENDER_EXECUTOR_VERSION = "page_render_executor_v4"


class PageRenderTransactionError(RuntimeError):
    """Raised by the public renderer when an atomic page render is rejected."""

    def __init__(self, result: "PageRenderResult") -> None:
        self.result = result
        reason = str(result.failure_reason or "page_render_transaction_failed")
        failed = ",".join(result.failed_layer_ids or [])
        detail = f" ({failed})" if failed else ""
        super().__init__(f"{reason}{detail}")


@dataclass
class PageRenderResult:
    """Page-level execution result and audit payload."""

    cleaned_page_base_path: str
    output_path: str
    plans: list[RenderLayerPlan]
    layouts: list[TypesetLayout]
    fit_reports: list[FitReport]
    layer_audits: list[dict[str, Any]]
    elapsed_ms: float = 0.0
    status: str = "not_started"
    issues: list[str] | None = None
    canvas_size: list[int] | None = None
    page_slot_owner: str = "RenderLayoutPlanner"
    render_layout_planner_version: str = RENDER_LAYOUT_PLANNER_VERSION
    output_committed: bool = False
    requested_layer_count: int = 0
    failed_layer_ids: list[str] | None = None
    failure_reason: str = ""

    def to_audit_dict(self) -> dict[str, Any]:
        issues = list(self.issues or [])
        drawn_layers = [item for item in self.layer_audits if item.get("drawn")]
        committed_layers = [
            item
            for item in self.layer_audits
            if isinstance(item.get("page_transaction"), Mapping)
            and item.get("page_transaction", {}).get("status") == "committed"
        ]
        failed_layer_ids = list(self.failed_layer_ids or [])
        return {
            "page_render_executor_version": PAGE_RENDER_EXECUTOR_VERSION,
            "renderer_compositor_version": RENDERER_COMPOSITOR_VERSION,
            "page_slot_owner": self.page_slot_owner,
            "render_layout_planner_version": self.render_layout_planner_version,
            "status": self.status,
            "page_transaction_status": (
                "committed" if self.output_committed else "rejected"
            ),
            "output_committed": bool(self.output_committed),
            "requested_layer_count": int(self.requested_layer_count),
            "failed_layer_ids": failed_layer_ids,
            "failure_reason": str(self.failure_reason or ""),
            "cleaned_page_base_path": self.cleaned_page_base_path,
            "output_path": self.output_path,
            "elapsed_ms": round(float(self.elapsed_ms), 3),
            "drawing_authority": "typeset_glyph_placements",
            "raster_authority": GLYPH_RASTER_AUTHORITY,
            "pillow_string_raster_used": False,
            "input_authority": "parent_execution_bundle_render_layer_plan",
            "cleanup_mutation_allowed": False,
            "renderer_cleanup_mutation_applied": False,
            "legacy_region_rendering_used": False,
            "canvas_size": list(self.canvas_size or []),
            "page_orientation": _page_orientation(self.canvas_size),
            "page_aspect_ratio_sets_writing_mode": False,
            "layer_count": len(self.plans),
            "layout_count": len(self.layouts),
            "fit_report_count": len(self.fit_reports),
            "drawn_layer_count": len(drawn_layers),
            "committed_layer_count": len(committed_layers),
            "issues": issues,
            "layers": copy_jsonish(self.layer_audits),
            "render_layer_plans": render_layer_plans_to_audit_dict(self.plans),
            "typeset_layouts": typeset_layouts_to_audit_dict(self.layouts),
            "fit_reports": fit_reports_to_audit_dict(self.fit_reports),
        }


class PageRenderExecutor:
    """Sequence page-slot planning, typesetting, fit, and draw-only composition."""

    version = PAGE_RENDER_EXECUTOR_VERSION

    def __init__(
        self,
        *,
        font_manager: FontManager | None = None,
        typesetting_engine: TypesettingEngine | None = None,
        layout_planner: RenderLayoutPlanner | None = None,
        glyph_rasterizer: Any | None = None,
        ink_bound_fitter: InkBoundLayoutFitter | None = None,
        compositor: RendererCompositor | None = None,
    ) -> None:
        compositor_font_manager = getattr(compositor, "font_manager", None)
        self.font_manager = font_manager or compositor_font_manager or FontManager()
        self.typesetting_engine = typesetting_engine or TypesettingEngine(self.font_manager)
        self.layout_planner = layout_planner or RenderLayoutPlanner(self.typesetting_engine)
        self.ink_bound_fitter = ink_bound_fitter or InkBoundLayoutFitter()
        self.compositor = compositor or RendererCompositor(
            font_manager=self.font_manager,
            glyph_rasterizer=glyph_rasterizer,
        )

    def compose(
        self,
        cleaned_page_base_path: str,
        output_path: str,
        plans: Sequence[RenderLayerPlan],
    ) -> PageRenderResult:
        """Execute all layers against one immutable CleanedPageBase image."""

        if Image is None:
            raise RuntimeError("Pillow is not installed.")
        if not cleaned_page_base_path:
            raise ValueError("cleaned_page_base_path is required")
        if not output_path:
            raise ValueError("output_path is required")

        start = time.perf_counter()
        ordered_plans = _ordered_plans(plans)
        page_slotted_plans = self.layout_planner.plan_page_slots(ordered_plans)
        layouts: list[TypesetLayout] = []
        reports: list[FitReport] = []
        layer_audits: list[dict[str, Any]] = []
        issues: list[str] = []

        with Image.open(cleaned_page_base_path) as source:
            immutable_cleaned_page = source.convert("RGBA")
        output_page = immutable_cleaned_page.copy()
        canvas_size = [
            int(immutable_cleaned_page.size[0]),
            int(immutable_cleaned_page.size[1]),
        ]
        adjusted_plans: list[RenderLayerPlan] = []
        occupied_bounds: list[dict[str, Any]] = []
        failed_layer_ids: list[str] = []
        for plan in page_slotted_plans:
            adjusted_plan = self.layout_planner.plan_layer(
                immutable_cleaned_page,
                plan,
                occupied_bounds=occupied_bounds,
            )
            adjusted_plans.append(adjusted_plan)
            layout, report, effect_degradation = _typeset_with_optional_effect_fallback(
                self.typesetting_engine,
                adjusted_plan,
            )
            parent_effects = resolve_parent_layer_effects(adjusted_plan.resolved_render_style)
            if parent_effects.active and not effect_degradation:
                fit_result = InkBoundFitResult(
                    layout=layout,
                    report=report,
                    audit={
                        "ink_bound_layout_fitter_version": self.ink_bound_fitter.version,
                        "policy": "effect_envelope_owned_by_typesetting_engine_v1",
                        "policy_owner": "typesetting_engine_parent_effect_envelope",
                        "status": "not_required",
                        "selected_shift": [0, 0],
                        "relative_geometry_preserved": True,
                        "font_size_changed": False,
                        "breaks_changed": False,
                        "writing_mode_changed": False,
                        "reason": "active_parent_effect_envelope_already_fitted",
                        "issues": [],
                    },
                )
            else:
                fit_result = self.ink_bound_fitter.fit(
                    adjusted_plan,
                    layout,
                    report,
                    _layout_fit_evidence(adjusted_plan, layout),
                )
            if fit_result.applied:
                layout = fit_result.layout
                report = fit_result.report
            candidate_page = output_page.copy()
            audit = self.compositor.compose_layer(
                candidate_page,
                adjusted_plan,
                layout,
                report,
            )
            if effect_degradation:
                audit["optional_effect_degradation"] = copy_jsonish(
                    effect_degradation
                )
            ink_fit_audit = copy_jsonish(fit_result.audit)
            ink_fit_audit["post_fit_failed_raster_placement_count"] = int(
                audit.get("failed_raster_placement_count") or 0
            )
            ink_fit_audit["post_fit_hard_bound_containment_failure_count"] = int(
                audit.get("hard_bound_containment_failure_count") or 0
            )
            audit["ink_bound_fit"] = ink_fit_audit
            audit["issues"] = _unique_strings(
                [
                    *(audit.get("issues") or []),
                    *(fit_result.audit.get("issues") or []),
                ]
            )
            layouts.append(layout)
            reports.append(report)
            layer_audits.append(audit)
            issues.extend(str(item) for item in audit.get("issues", []) or [])
            transaction_failures = _parent_layer_rejection_reasons(
                adjusted_plan,
                layout,
                report,
                audit,
            )
            if transaction_failures:
                audit["page_transaction"] = {
                    "status": "parent_rejected_page_continues",
                    "required_parent": True,
                    "reasons": list(transaction_failures),
                    "output_committed": False,
                }
                audit["issues"] = _unique_strings(
                    [
                        *(audit.get("issues") or []),
                        *transaction_failures,
                        "parent_render_rejected_page_continues",
                    ]
                )
                issues.extend(transaction_failures)
                issues.append("parent_render_rejected_page_continues")
                failed_layer_ids.append(str(adjusted_plan.layer_id or adjusted_plan.parent_id))
                continue
            if audit.get("drawn"):
                audit["page_transaction"] = {
                    "status": "staged",
                    "required_parent": bool(adjusted_plan.render_required),
                    "reasons": [],
                    "output_committed": False,
                }
                output_page = candidate_page
            else:
                audit["page_transaction"] = {
                    "status": "skipped_nonrequired_or_empty",
                    "required_parent": bool(adjusted_plan.render_required),
                    "reasons": _unique_strings(audit.get("issues") or []),
                    "output_committed": False,
                }
            if layout.measured_bounds and audit.get("drawn"):
                composition = (
                    audit.get("parent_layer_composition")
                    if isinstance(audit.get("parent_layer_composition"), Mapping)
                    else {}
                )
                occupied_box = _xyxy_to_xywh(
                    composition.get("final_alpha_bounds") or []
                ) or list(layout.measured_bounds)
                occupied_bounds.append(
                    {
                        "root_id": str(adjusted_plan.root_id or ""),
                        "parent_id": str(adjusted_plan.parent_id or ""),
                        "box": occupied_box,
                    }
                )

        elapsed = (time.perf_counter() - start) * 1000.0
        try:
            _save_image_atomic(output_page, output_path)
        except Exception as exc:
            issues.extend(
                [
                    "page_output_atomic_commit_failed",
                    f"page_output_atomic_commit_error:{type(exc).__name__}",
                ]
            )
            _finalize_page_transaction_audits(
                layer_audits,
                committed=False,
                failure_reason="page_output_atomic_commit_failed",
            )
            return PageRenderResult(
                cleaned_page_base_path=cleaned_page_base_path,
                output_path=output_path,
                plans=adjusted_plans,
                layouts=layouts,
                fit_reports=reports,
                layer_audits=layer_audits,
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                status="failed",
                issues=_unique_strings(issues),
                canvas_size=canvas_size,
                page_slot_owner=type(self.layout_planner).__name__,
                render_layout_planner_version=str(
                    getattr(self.layout_planner, "version", "") or "unversioned"
                ),
                output_committed=False,
                requested_layer_count=len(ordered_plans),
                failed_layer_ids=failed_layer_ids,
                failure_reason="page_output_atomic_commit_failed",
            )
        if failed_layer_ids:
            issues.append("page_committed_with_parent_diagnostics")
        _finalize_page_transaction_audits(layer_audits, committed=True)
        return PageRenderResult(
            cleaned_page_base_path=cleaned_page_base_path,
            output_path=output_path,
            plans=adjusted_plans,
            layouts=layouts,
            fit_reports=reports,
            layer_audits=layer_audits,
            elapsed_ms=elapsed,
            status="completed",
            issues=_unique_strings(issues),
            canvas_size=canvas_size,
            page_slot_owner=type(self.layout_planner).__name__,
            render_layout_planner_version=str(
                getattr(self.layout_planner, "version", "") or "unversioned"
            ),
            output_committed=True,
            requested_layer_count=len(ordered_plans),
            failed_layer_ids=failed_layer_ids,
            failure_reason="",
        )


def _ordered_plans(plans: Sequence[RenderLayerPlan]) -> list[RenderLayerPlan]:
    return sorted(
        [plan for plan in plans or [] if isinstance(plan, RenderLayerPlan)],
        key=lambda plan: (int(plan.draw_order), str(plan.layer_id)),
    )


def _xyxy_to_xywh(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    try:
        items = [int(round(float(item))) for item in list(value)[:4]]
    except (TypeError, ValueError):
        return []
    if len(items) != 4 or items[2] <= items[0] or items[3] <= items[1]:
        return []
    return [items[0], items[1], items[2] - items[0], items[3] - items[1]]


def _xywh_to_xyxy(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    try:
        items = [int(round(float(item))) for item in list(value)[:4]]
    except (TypeError, ValueError):
        return []
    if len(items) != 4 or items[2] <= 0 or items[3] <= 0:
        return []
    return [items[0], items[1], items[0] + items[2], items[1] + items[3]]


def _contains_xyxy(outer: Sequence[int], inner: Sequence[int]) -> bool:
    return bool(
        len(outer) == 4
        and len(inner) == 4
        and inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _layout_fit_evidence(
    plan: RenderLayerPlan,
    layout: TypesetLayout,
) -> list[dict[str, Any]]:
    """Build pre-raster fit evidence from finalized layout geometry."""

    hard_bounds = _xywh_to_xyxy(plan.hard_bounds or plan.target_box)
    records: list[dict[str, Any]] = []
    for value in list(layout.glyphs or []):
        if hasattr(value, "to_audit_dict"):
            item = value.to_audit_dict()
        elif isinstance(value, Mapping):
            item = dict(value)
        else:
            continue
        text = str(item.get("text") or "")
        box = _xywh_to_xyxy(item.get("bbox"))
        if not box or not text or text.isspace():
            continue
        accepted = bool(hard_bounds and _contains_xyxy(hard_bounds, box))
        records.append(
            {
                "status": "drawn",
                "placement_text": text,
                "placement_bbox": list(box),
                "composite_bounds": list(box),
                "hard_bound_containment": {
                    "accepted": accepted,
                    "reason": "layout_bounds_inside_parent_hard_bounds"
                    if accepted
                    else "layout_bounds_exceed_parent_hard_bounds",
                    "parent_hard_bounds": list(hard_bounds),
                    "raster_alpha_bounds": list(box),
                },
                "issues": [] if accepted else ["raster_ink_exceeds_parent_hard_bounds"],
                "evidence_source": "finalized_layout_geometry_pre_raster",
            }
        )
    return records


def _typeset_with_optional_effect_fallback(
    typesetting_engine: TypesettingEngine,
    plan: RenderLayerPlan,
) -> tuple[TypesetLayout, FitReport, dict[str, Any]]:
    effects = resolve_parent_layer_effects(plan.resolved_render_style)
    if effects.status == "invalid":
        base_plan = _plan_without_optional_effects(plan)
        layout, report = typesetting_engine.typeset_layer(base_plan)
        degradation = {
            "status": "degraded_to_base",
            "reason": "parent_layer_effect_contract_invalid",
            "issues": _unique_strings(
                ["parent_layer_effect_contract_invalid", *effects.issues]
            ),
            "typeset_retry_used": True,
        }
        _annotate_optional_effect_fallback(layout, report, degradation)
        return layout, report, degradation

    layout, report = typesetting_engine.typeset_layer(plan)
    if effects.active and _effect_envelope_blocked_base_layout(report):
        base_plan = _plan_without_optional_effects(plan)
        base_layout, base_report = typesetting_engine.typeset_layer(base_plan)
        degradation = {
            "status": "degraded_to_base",
            "reason": "parent_layer_effect_envelope_exceeds_hard_bounds",
            "issues": _unique_strings(
                [
                    "parent_layer_effect_envelope_exceeds_hard_bounds",
                    *list(report.issues or []),
                ]
            ),
            "typeset_retry_used": True,
        }
        _annotate_optional_effect_fallback(base_layout, base_report, degradation)
        return base_layout, base_report, degradation
    return layout, report, {}


def _effect_envelope_blocked_base_layout(report: FitReport) -> bool:
    return "parent_layer_effect_envelope_exceeds_hard_bounds" in {
        str(issue) for issue in list(report.issues or [])
    }


def _plan_without_optional_effects(plan: RenderLayerPlan) -> RenderLayerPlan:
    base_plan = deepcopy(plan)
    style = deepcopy(dict(plan.resolved_render_style or {}))
    style["parent_layer_effects"] = {
        "contract_version": PARENT_LAYER_EFFECTS_VERSION,
        "rotation": {"availability": "unavailable"},
        "shadow": {"availability": "unavailable"},
    }
    base_plan.resolved_render_style = style
    metadata = deepcopy(dict(base_plan.metadata or {}))
    metadata["optional_effect_fallback"] = {
        "policy": "visible_base_text_before_optional_effects_v1",
        "effects_removed_for_typesetting": True,
    }
    base_plan.metadata = metadata
    return base_plan


def _annotate_optional_effect_fallback(
    layout: TypesetLayout,
    report: FitReport,
    degradation: Mapping[str, Any],
) -> None:
    layout.metadata = deepcopy(dict(layout.metadata or {}))
    layout.metadata["optional_effect_degradation"] = copy_jsonish(degradation)
    report.metadata = deepcopy(dict(report.metadata or {}))
    report.metadata["optional_effect_degradation"] = copy_jsonish(degradation)
    report.fallback_used = True
    report.fallback_reason = "optional_effect_removed_for_visible_base_text"
    report.user_review_recommended = True
    report.issues = _unique_strings(
        [*(report.issues or []), *(degradation.get("issues") or [])]
    )


def _parent_layer_rejection_reasons(
    plan: RenderLayerPlan,
    layout: TypesetLayout,
    report: FitReport,
    audit: Mapping[str, Any],
) -> list[str]:
    if not bool(plan.render_required):
        return []
    composition = (
        audit.get("parent_layer_composition")
        if isinstance(audit.get("parent_layer_composition"), Mapping)
        else {}
    )
    final_containment = (
        composition.get("final_alpha_containment")
        if isinstance(composition.get("final_alpha_containment"), Mapping)
        else {}
    )
    failures: list[str] = []
    if not str(plan.translated_text or "").strip():
        failures.append("required_parent_translated_text_empty")
    if (
        not bool(report.text_placement_complete)
        or not bool(layout.text_placement_complete)
    ):
        failures.append("required_parent_layout_not_composable")
    if not bool(audit.get("drawn")):
        failures.append("required_parent_base_text_not_drawn")
    if not bool(audit.get("glyph_text_matches_layout")):
        failures.append("required_parent_glyph_text_not_conserved")
    if int(audit.get("failed_raster_placement_count") or 0) > 0:
        failures.append("required_parent_raster_placement_failed")
    if str(composition.get("status") or "") != "committed":
        failures.append("required_parent_layer_not_committed")
    if int(composition.get("page_composite_count") or 0) != 1:
        failures.append("required_parent_atomic_commit_count_invalid")
    if (
        final_containment.get("raster_alpha_bounds")
        and not bool(final_containment.get("inside_page_bounds"))
    ):
        failures.append("required_parent_final_alpha_outside_page_canvas")
    if int(composition.get("final_alpha_sum") or 0) <= 0:
        failures.append("required_parent_visible_alpha_empty")
    return _unique_strings(failures)


def _save_image(image, output_path: str) -> None:
    ext = os.path.splitext(output_path)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        image.convert("RGB").save(output_path, quality=95)
    else:
        image.save(output_path)


def _save_image_atomic(image, output_path: str) -> None:
    absolute_output = os.path.abspath(output_path)
    out_dir = os.path.dirname(absolute_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    extension = os.path.splitext(absolute_output)[1].lower() or ".png"
    handle, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(absolute_output)}.",
        suffix=extension,
        dir=out_dir or None,
    )
    os.close(handle)
    try:
        _save_image(image, temp_path)
        os.replace(temp_path, absolute_output)
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _finalize_page_transaction_audits(
    layer_audits: Sequence[dict[str, Any]],
    *,
    committed: bool,
    failure_reason: str = "",
) -> None:
    """Finalize staged layer evidence only after the page transaction resolves."""

    for audit in layer_audits:
        existing = (
            dict(audit.get("page_transaction") or {})
            if isinstance(audit.get("page_transaction"), Mapping)
            else {}
        )
        staged = str(existing.get("status") or "") in {"staged", "accepted"}
        if committed and staged:
            existing.update(
                {
                    "status": "committed",
                    "reasons": [],
                    "output_committed": True,
                }
            )
        elif not committed:
            existing.update(
                {
                    "status": "rolled_back" if staged else "rejected",
                    "reasons": _unique_strings(
                        [*(existing.get("reasons") or []), failure_reason]
                    ),
                    "output_committed": False,
                }
            )
        audit["page_transaction"] = existing


def _page_orientation(value: Sequence[int] | None) -> str:
    if not value or len(value) < 2:
        return "unknown"
    width = int(value[0])
    height = int(value[1])
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


def _unique_strings(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in output:
            output.append(text)
    return output
