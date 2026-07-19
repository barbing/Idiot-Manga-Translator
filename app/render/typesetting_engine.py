# -*- coding: utf-8 -*-
"""Pure Stage 4 typesetting engine.

The engine consumes RenderLayerPlan records and emits TypesetLayout/FitReport
records. It does not draw final text, mutate cleanup, or reinterpret parent
identity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_CEILING
from typing import Any, Mapping, Sequence

from app.render.font_manager import (
    FontManager,
    FontResolution,
    FontSpanResolution,
    RunFontResolution,
)
from app.render.line_break_planner import LineBreakPlanner
from app.render.parent_layer_effects import (
    fit_effect_envelope,
    outward_int_xywh,
    resolve_parent_layer_effects,
    shift_layout_geometry,
)
from app.render.text_shaper import HarfBuzzShaper, ShapedRun
from app.render.typesetting_contracts import (
    DrawingPrimitive,
    FitReport,
    GlyphPlacement,
    PunctuationToken,
    RenderLayerPlan,
    TypesetLayout,
    bbox_from_value,
    validated_source_text_footprint_ref,
)
from app.render.typesetting_text import (
    BreakOpportunity,
    VERTICAL_CENTERED_PUNCTUATION_CHARS,
    InlineTextRun,
    build_lossless_text_tokens,
    compute_break_opportunities,
    grapheme_clusters,
    presentation_notes_from_tokens,
    punctuation_occurrences_from_tokens,
    resolve_writing_mode_presentations,
    segment_inline_runs,
    source_char_requires_visible_glyph,
    source_text_requires_visible_glyph,
    symbol_occurrences_from_tokens,
    tokens_original_text,
    tokens_presentation_text,
)


_VERTICAL_NO_COLUMN_START = {
    ")",
    "）",
    "]",
    "］",
    "}",
    "｝",
    "」",
    "』",
    "】",
    "〉",
    "》",
    "”",
    "’",
    "。",
    "，",
    "、",
    "．",
    ".",
    ",",
    "!",
    "?",
    "！",
    "？",
    "︕",
    "︖",
    "‼",
    "⁇",
    "⁉",
    "⁈",
    "︙",
    "︐",
    "︑",
    "︒",
    "︓",
    "︔",
}


@dataclass(frozen=True)
class TypesettingPolicy:
    min_font_size: int = 8
    min_readable_font_size: int = 12
    vertical_inner_padding_ratio_x: float = 0.04
    vertical_inner_padding_ratio_y: float = 0.04
    horizontal_inner_padding_ratio_x: float = 0.05
    horizontal_inner_padding_ratio_y: float = 0.05
    max_binary_fit_steps: int = 8
    default_japanese_manga_writing_mode: str = "vertical"
    allow_block_writing_mode_auto_flip: bool = False
    max_tate_chu_yoko_graphemes: int = 4
    max_vertical_compact_latin_graphemes: int = 4
    allow_latin_letter_stacking: bool = False
    allow_emergency_word_break: bool = True
    enable_japanese_line_break_rules: bool = True
    allow_punctuation_hanging: bool = False


class TypesettingEngine:
    def __init__(
        self,
        font_manager: FontManager | None = None,
        shaper: HarfBuzzShaper | None = None,
        policy: TypesettingPolicy | None = None,
        break_planner: LineBreakPlanner | None = None,
    ) -> None:
        self.font_manager = font_manager or FontManager()
        self.shaper = shaper or HarfBuzzShaper(self.font_manager)
        self.policy = policy or TypesettingPolicy()
        self.break_planner = break_planner or LineBreakPlanner()

    def typeset_layer(self, plan: RenderLayerPlan) -> tuple[TypesetLayout, FitReport]:
        missing = _missing_identity(plan)
        if missing:
            return self._failed(plan, "missing_layer_identity", missing)
        target_box = bbox_from_value(plan.target_box)
        hard_bounds = bbox_from_value(plan.hard_bounds) or target_box
        if not target_box or not hard_bounds:
            return self._failed(plan, "missing_hard_bounds", ["missing_hard_bounds"])
        if not str(plan.translated_text or ""):
            return self._failed(plan, "empty_text", ["empty_text"], hard_bounds=hard_bounds)

        parent_layer_effects = resolve_parent_layer_effects(plan.resolved_render_style)
        if parent_layer_effects.status == "invalid":
            return self._failed(
                plan,
                "invalid_parent_layer_effect_contract",
                ["parent_layer_effect_contract_invalid", *parent_layer_effects.issues],
                hard_bounds=hard_bounds,
            )

        writing_mode, writing_policy = self._resolve_writing_mode(plan)
        preferred_font_size = _font_size_from_style(plan.resolved_render_style)
        if preferred_font_size <= 0:
            return self._failed(
                plan,
                "invalid_resolved_render_style_font_size",
                ["invalid_resolved_render_style_font_size"],
                hard_bounds=hard_bounds,
            )
        resolved = self.font_manager.resolve_font(
            plan.resolved_render_style,
            fallback_chain_key=str(
                plan.resolved_render_style.get("fallback_font_chain_key") or ""
            ),
            writing_mode=writing_mode,
            text=plan.translated_text,
        )
        if not resolved.usable or resolved.primary_face is None:
            return self._failed(plan, "missing_font", list(resolved.issues or ["missing_font"]), hard_bounds=hard_bounds)
        face = resolved.primary_face
        identity_tokens = build_lossless_text_tokens(plan.translated_text)
        text_tokens = resolve_writing_mode_presentations(
            identity_tokens,
            writing_mode=writing_mode,
            font_manager=self.font_manager,
            face=face,
        )
        token_text_conserved = tokens_original_text(text_tokens) == str(
            plan.translated_text or ""
        )
        if not token_text_conserved:
            return self._failed(
                plan,
                "lossless_text_token_identity_not_conserved",
                ["lossless_text_token_identity_not_conserved"],
                hard_bounds=hard_bounds,
            )
        normalized = tokens_presentation_text(text_tokens)
        punctuation = punctuation_occurrences_from_tokens(text_tokens)
        symbols = symbol_occurrences_from_tokens(
            text_tokens,
            font_manager=self.font_manager,
            face=face,
        )
        normalization_notes = presentation_notes_from_tokens(text_tokens)
        logical_runs = segment_inline_runs(
            text_tokens,
            writing_mode=writing_mode,
            language_hint=_language_hint(plan),
        )
        try:
            runs, font_span_resolutions = self._expand_runs_for_font_spans(
                logical_runs,
                resolved,
                writing_mode=writing_mode,
            )
        except RuntimeError as exc:
            issue = str(exc)
            if issue not in {
                "uax29_grapheme_segmenter_unavailable",
                "font_span_token_provenance_missing",
                "font_span_token_provenance_not_conserved",
            }:
                raise
            return self._font_span_failure(
                plan,
                normalized=normalized,
                hard_bounds=hard_bounds,
                resolved=resolved,
                logical_runs=logical_runs,
                expanded_runs=logical_runs,
                spans=[],
                issues=[issue],
            )
        font_span_text_conserved = (
            "".join(run.normalized_text for run in runs) == normalized
        )
        unresolved_span_issues = _unique(
            [
                issue
                for span in font_span_resolutions
                if not span.usable
                for issue in span.issues
            ]
        )
        if not font_span_text_conserved:
            unresolved_span_issues.append("font_span_text_not_conserved")
        if unresolved_span_issues:
            return self._font_span_failure(
                plan,
                normalized=normalized,
                hard_bounds=hard_bounds,
                resolved=resolved,
                logical_runs=logical_runs,
                expanded_runs=runs,
                spans=font_span_resolutions,
                issues=unresolved_span_issues,
            )
        font_spans_by_id = {
            span.span_id: span for span in font_span_resolutions
        }
        breaks = compute_break_opportunities(runs, writing_mode=writing_mode)
        breaks = [
            replace(
                item,
                allowed=False,
                strength="forbidden",
                reason="font_span_internal_boundary",
                metadata={
                    **dict(item.metadata),
                    "shaping_safe_boundary_only": True,
                },
            )
            if _logical_run_id_for(runs[index])
            == _logical_run_id_for(runs[index + 1])
            else item
            for index, item in enumerate(breaks)
        ]
        style_issues = _style_issues(runs, writing_mode, self.policy)
        script_policy = _script_policy(runs, writing_mode, self.policy)
        attempts: list[dict[str, Any]] = []
        selected_attempt: dict[str, Any] | None = None
        for font_size in _font_size_candidates(
            preferred_font_size,
            plan.resolved_render_style,
            self.policy,
            target_box,
            plan.metadata,
        ):
            shaped_runs, run_font_resolutions = self._shape_runs(
                runs,
                resolved,
                font_size,
                writing_mode,
                font_spans_by_id,
            )
            notdef_run_ids = _visible_notdef_run_ids(shaped_runs)
            if notdef_run_ids:
                return self._font_span_failure(
                    plan,
                    normalized=normalized,
                    hard_bounds=hard_bounds,
                    resolved=resolved,
                    logical_runs=logical_runs,
                    expanded_runs=runs,
                    spans=font_span_resolutions,
                    issues=["notdef_glyph_forbidden", "shaping_missing_glyph"],
                    shaped_runs=shaped_runs,
                    notdef_run_ids=notdef_run_ids,
                )
            metrics_by_face: dict[str, dict[str, Any]] = {}
            metric_faces = [face]
            metric_faces.extend(
                item.selected_face
                for item in run_font_resolutions
                if item.selected_face is not None and item.selected_face.face_id != face.face_id
            )
            for metric_face in metric_faces:
                if metric_face.face_id in metrics_by_face:
                    continue
                metrics_by_face[metric_face.face_id] = self.font_manager.open_type_metrics(
                    metric_face,
                    size=font_size,
                ).to_audit_dict()
            layout_intent_box, box_model, reason_codes = self._layout_intent_box(
                plan=plan,
                hard_bounds=hard_bounds,
                target_box=target_box,
                writing_mode=writing_mode,
                font_size=font_size,
                text=normalized,
                runs=runs,
                shaped_runs=shaped_runs,
            )
            if writing_mode == "horizontal":
                placements, lines, columns, measured_bounds, fit_status, fit_issues, break_plan = self._layout_horizontal(
                    normalized,
                    runs,
                    shaped_runs,
                    breaks,
                    layout_intent_box,
                    font_size,
                    plan.resolved_render_style,
                )
            else:
                placements, lines, columns, measured_bounds, fit_status, fit_issues, break_plan = self._layout_vertical(
                    normalized,
                    runs,
                    shaped_runs,
                    breaks,
                    layout_intent_box,
                    font_size,
                    plan.resolved_render_style,
                    plan,
                    candidate_capacity_box=hard_bounds,
                )
                retry = None
                if fit_status != "fits":
                    retry = self._retry_vertical_fit_within_hard_bounds(
                        normalized=normalized,
                        runs=runs,
                        shaped_runs=shaped_runs,
                        breaks=breaks,
                        plan=plan,
                        style=plan.resolved_render_style,
                        font_size=font_size,
                        hard_bounds=hard_bounds,
                        layout_intent_box=layout_intent_box,
                        box_model=box_model,
                        break_plan=break_plan,
                    )
                if retry is not None:
                    placements = retry["placements"]
                    lines = retry["lines"]
                    columns = retry["columns"]
                    measured_bounds = retry["measured_bounds"]
                    fit_status = retry["fit_status"]
                    fit_issues = retry["fit_issues"]
                    break_plan = retry["break_plan"]
                    layout_intent_box = retry["layout_intent_box"]
                    box_model = retry["box_model"]
                    reason_codes.extend(retry["reason_codes"])
                if columns and isinstance(
                    columns[0].get("layout_profile"),
                    Mapping,
                ):
                    box_model = dict(box_model)
                    box_model["vertical_layout_profile"] = dict(
                        columns[0]["layout_profile"]
                    )
            base_measured_bounds = list(measured_bounds)
            base_visual_center = _center_of(base_measured_bounds)
            parent_effect_envelope = fit_effect_envelope(
                base_measured_bounds,
                hard_bounds,
                parent_layer_effects,
                raster_guard_px=_parent_effect_raster_guard(plan.resolved_render_style),
            )
            if parent_layer_effects.active and fit_status == "fits":
                if not parent_effect_envelope.contained:
                    fit_status = "overflow"
                    fit_issues = _unique(
                        [
                            *fit_issues,
                            *parent_effect_envelope.issues,
                            "parent_layer_effect_envelope_exceeds_hard_bounds",
                        ]
                    )
                else:
                    shift_x, shift_y = parent_effect_envelope.translation
                    placements, lines, columns, base_measured_bounds = shift_layout_geometry(
                        placements,
                        lines,
                        columns,
                        base_measured_bounds,
                        shift_x,
                        shift_y,
                    )
                    base_visual_center = _center_of(base_measured_bounds)
                    measured_bounds = outward_int_xywh(parent_effect_envelope.final_bounds)
            attempt = {
                "font_size": int(font_size),
                "fit_status": fit_status,
                "layout_intent_box": list(layout_intent_box),
                "measured_bounds": list(measured_bounds),
                "base_measured_bounds": list(base_measured_bounds),
                "base_visual_center": list(base_visual_center),
                "parent_layer_effect_envelope": parent_effect_envelope.to_audit_dict(),
                "issues": list(fit_issues),
                "placements": placements,
                "lines": lines,
                "columns": columns,
                "shaped_runs": shaped_runs,
                "box_model": box_model,
                "reason_codes": reason_codes,
                "break_plan": break_plan,
                "run_font_resolutions": run_font_resolutions,
                "open_type_metrics_by_face": metrics_by_face,
                "effective_line_height": (
                    float(box_model.get("effective_line_height"))
                    if box_model.get("effective_line_height") is not None
                    else _line_height(plan.resolved_render_style)
                ),
            }
            attempts.append(attempt)
            selected_attempt = attempt
            if fit_status == "fits":
                break
        if selected_attempt is None:
            return self._failed(plan, "layout_attempt_failed", ["layout_attempt_failed"], hard_bounds=hard_bounds)
        font_size = int(selected_attempt["font_size"])
        shaped_runs = list(selected_attempt["shaped_runs"])
        placements = list(selected_attempt["placements"])
        lines = list(selected_attempt["lines"])
        columns = list(selected_attempt["columns"])
        measured_bounds = list(selected_attempt["measured_bounds"])
        base_measured_bounds = list(selected_attempt["base_measured_bounds"])
        base_visual_center = list(selected_attempt["base_visual_center"])
        parent_effect_envelope = dict(selected_attempt["parent_layer_effect_envelope"])
        fit_status = str(selected_attempt["fit_status"])
        fit_issues = list(selected_attempt["issues"])
        box_model = dict(selected_attempt["box_model"])
        reason_codes = list(selected_attempt["reason_codes"])
        break_plan = dict(selected_attempt["break_plan"])
        run_font_resolutions = list(selected_attempt["run_font_resolutions"])
        open_type_metrics_by_face = dict(selected_attempt["open_type_metrics_by_face"])
        font_span_cluster_map = _font_span_cluster_ledger(runs, shaped_runs)
        effective_line_height = float(selected_attempt.get("effective_line_height") or _line_height(plan.resolved_render_style))
        if font_size < preferred_font_size:
            reason_codes.append("font_size_reduced_to_fit_translated_text")
            if parent_layer_effects.active and any(
                not bool(item["parent_layer_effect_envelope"].get("contained"))
                for item in attempts[:-1]
            ):
                reason_codes.append("font_size_reduced_to_fit_parent_effect_envelope")
            if bool(plan.resolved_render_style.get("font_size_locked")):
                reason_codes.append("font_size_lock_relaxed_for_layout_fit")
        kinsoku_adjustments = [item.to_audit_dict() for item in breaks if not item.allowed and item.reason.startswith("kinsoku")]
        if fit_status != "fits" and kinsoku_adjustments and "kinsoku_fit_conflict" not in fit_issues:
            fit_issues.append("kinsoku_fit_conflict")
        if fit_status != "fits" and writing_mode == "vertical":
            fit_issues.append("writing_mode_fit_failure")

        placements, drawing_primitives = _finalize_drawing_primitives(
            placements,
            plan.resolved_render_style,
        )

        run_font_issues = [
            issue
            for item in run_font_resolutions
            for issue in item.issues
        ]
        all_issues = _unique([*resolved.issues, *run_font_issues, *style_issues, *fit_issues])
        full_text_placed = fit_status == "fits"
        token_audit = [item.to_audit_dict() for item in text_tokens]
        placed_token_ids = _unique(
            [
                token_id
                for placement in placements
                for token_id in list(placement.metadata.get("token_ids") or [])
            ]
        )
        visible_token_ids = [
            item.token_id
            for item in text_tokens
            if source_text_requires_visible_glyph(item.presentation_text)
        ]
        token_conservation = {
            "exact_input_reconstructed": bool(token_text_conserved),
            "translated_text": str(plan.translated_text or ""),
            "token_original_text": tokens_original_text(text_tokens),
            "presentation_text": tokens_presentation_text(text_tokens),
            "logical_run_original_text": "".join(
                item.original_text for item in logical_runs
            ),
            "token_count": len(text_tokens),
            "atomic_token_ids": [
                item.token_id for item in text_tokens if item.atomic_break
            ],
            "visible_token_ids": list(visible_token_ids),
            "placed_token_ids": list(placed_token_ids),
            "visible_tokens_placed": set(visible_token_ids).issubset(
                set(placed_token_ids)
            ),
        }
        metadata = {
            "typesetting_engine_version": "typesetting_engine_stage4_v1",
            "lossless_text_tokens": token_audit,
            "text_token_conservation": token_conservation,
            "box_model": box_model,
            "writing_mode_policy": writing_policy,
            "inline_runs": [_run_audit(run, writing_mode, self.policy) for run in runs],
            "break_opportunities": [item.to_audit_dict() for item in breaks],
            "chosen_breaks": list(break_plan.get("selected_breaks") or []),
            "break_plan": break_plan,
            "kinsoku_adjustments": kinsoku_adjustments,
            "line_break_policy": {
                "policy_version": "line_break_policy_stage4_v2",
                "locale_hint": _language_hint(plan),
                "writing_mode": writing_mode,
                "selection_authority": "explicit_break_opportunities",
                "planner_version": self.break_planner.version,
                "accepted_tailoring_rules": ["space_word_boundary", "cjk_grapheme_boundary"],
                "rejected_fallback_rules": ["raw_character_count_splitting"],
            },
            "normalization_notes": normalization_notes,
            "bidi_runs": [
                run.to_audit_dict()
                for run in runs
                if run.direction == "rtl" or run.metadata.get("bidi_visual_text")
            ],
            "script_policy": script_policy,
            "shaped_runs": [run.to_audit_dict() for run in shaped_runs],
            "font_resolution": resolved.to_audit_dict(),
            "run_font_resolutions": [item.to_audit_dict() for item in run_font_resolutions],
            "font_span_resolutions": [
                item.to_audit_dict() for item in font_span_resolutions
            ],
            "font_span_text_conserved": bool(font_span_text_conserved),
            "font_span_cluster_map": font_span_cluster_map,
            "notdef_run_ids": [],
            "parent_layer_effects": parent_layer_effects.to_audit_dict(),
            "parent_layer_effect_envelope": parent_effect_envelope,
            "base_measured_bounds": list(base_measured_bounds),
            "base_visual_center": list(base_visual_center),
            "drawing_primitives": [
                item.to_audit_dict() for item in drawing_primitives
            ],
            "drawing_primitive_geometry_final": True,
            "open_type_metrics_by_face": open_type_metrics_by_face,
            "font_size_selection": {
                "preferred_font_size": int(preferred_font_size),
                "selected_font_size": int(font_size),
                "scaling_used": round(float(font_size) / float(max(1, preferred_font_size)), 4),
                "fallback_used": font_size != preferred_font_size,
                "candidate_count": len(attempts),
                "attempts": [
                    {
                        "font_size": int(item["font_size"]),
                        "fit_status": str(item["fit_status"]),
                        "layout_intent_box": list(item["layout_intent_box"]),
                        "measured_bounds": list(item["measured_bounds"]),
                        "issues": list(item["issues"]),
                        "parent_layer_effect_envelope": dict(
                            item["parent_layer_effect_envelope"]
                        ),
                    }
                    for item in attempts
                ],
            },
            "render_style": {
                "line_height": plan.resolved_render_style.get("line_height"),
                "effective_line_height": round(effective_line_height, 4),
                "align": plan.resolved_render_style.get("align"),
                "fill_color": plan.resolved_render_style.get("fill_color"),
                "stroke_color": plan.resolved_render_style.get("stroke_color"),
                "stroke_width": plan.resolved_render_style.get("stroke_width"),
            },
        }
        layout = TypesetLayout(
            page_id=plan.page_id,
            layer_id=plan.layer_id,
            bundle_id=plan.bundle_id,
            parent_id=plan.parent_id,
            root_id=plan.root_id,
            selected_font_face=face.face_id,
            selected_font_size=float(font_size),
            writing_mode=writing_mode,
            lines=lines,
            columns=columns,
            glyphs=placements,
            drawing_primitives=drawing_primitives,
            punctuation_placements=punctuation,
            symbol_placements=symbols,
            measured_bounds=measured_bounds,
            visual_center=(
                list(base_visual_center)
                if parent_layer_effects.active
                else _center_of(measured_bounds)
            ),
            fit_status=fit_status,
            normalized_text=normalized,
            original_text=plan.translated_text,
            lossless_text_tokens=list(text_tokens),
            metadata=metadata,
        )
        report = FitReport(
            page_id=plan.page_id,
            layer_id=plan.layer_id,
            bundle_id=plan.bundle_id,
            parent_id=plan.parent_id,
            root_id=plan.root_id,
            natural_fit_success=full_text_placed,
            fallback_used=font_size != preferred_font_size,
            scaling_used=float(font_size) / float(max(1, preferred_font_size)),
            overflow_risk=fit_status != "fits",
            clipping_risk=fit_status != "fits",
            clipped_region=[],
            full_text_placed=full_text_placed,
            punctuation_normalization_applied=punctuation,
            symbol_fallbacks=[item for item in symbols if not item.get("supported")],
            user_review_recommended=bool(all_issues),
            fit_status=fit_status,
            issues=all_issues,
            lossless_text_tokens=list(text_tokens),
            metadata={
                "typesetting_engine_version": "typesetting_engine_stage4_v1",
                "lossless_text_tokens": token_audit,
                "text_token_conservation": token_conservation,
                "reason_codes": reason_codes,
                "writing_mode_policy": {
                    **writing_policy,
                    "block_writing_mode_flip_forbidden": True,
                },
                "script_policy": script_policy,
                "line_break_policy": metadata["line_break_policy"],
                "break_opportunities": metadata["break_opportunities"],
                "chosen_breaks": metadata["chosen_breaks"],
                "break_plan": break_plan,
                "kinsoku_adjustments": kinsoku_adjustments,
                "normalization_notes": normalization_notes,
                "inline_runs": metadata["inline_runs"],
                "shaped_runs": metadata["shaped_runs"],
                "run_font_resolutions": metadata["run_font_resolutions"],
                "font_span_resolutions": metadata["font_span_resolutions"],
                "font_span_text_conserved": metadata["font_span_text_conserved"],
                "font_span_cluster_map": metadata["font_span_cluster_map"],
                "notdef_run_ids": metadata["notdef_run_ids"],
                "parent_layer_effects": metadata["parent_layer_effects"],
                "parent_layer_effect_envelope": parent_effect_envelope,
                "base_measured_bounds": list(base_measured_bounds),
                "base_visual_center": list(base_visual_center),
                "open_type_metrics_by_face": open_type_metrics_by_face,
                "box_model": box_model,
            },
        )
        return layout, report

    def typeset_layers(self, plans: Sequence[RenderLayerPlan]) -> tuple[list[TypesetLayout], list[FitReport]]:
        layouts: list[TypesetLayout] = []
        reports: list[FitReport] = []
        for plan in plans:
            layout, report = self.typeset_layer(plan)
            layouts.append(layout)
            reports.append(report)
        return layouts, reports

    def _resolve_writing_mode(self, plan: RenderLayerPlan) -> tuple[str, dict[str, Any]]:
        candidates = [
            ("render_layer_plan", plan.writing_mode),
            ("resolved_render_style.wrap_mode", plan.resolved_render_style.get("wrap_mode")),
            ("resolved_render_style.source_orientation", plan.resolved_render_style.get("source_orientation")),
        ]
        for source, value in candidates:
            mode = _normalize_mode(value)
            if mode in {"vertical", "horizontal"}:
                return mode, {
                    "block_mode_source": source,
                    "block_writing_mode": mode,
                    "allow_block_writing_mode_auto_flip": self.policy.allow_block_writing_mode_auto_flip,
                }
        return self.policy.default_japanese_manga_writing_mode, {
            "block_mode_source": "policy.default_japanese_manga_writing_mode",
            "block_writing_mode": self.policy.default_japanese_manga_writing_mode,
            "allow_block_writing_mode_auto_flip": self.policy.allow_block_writing_mode_auto_flip,
        }

    def _shape_runs(
        self,
        runs: Sequence[InlineTextRun],
        resolution: FontResolution,
        font_size: int,
        writing_mode: str,
        font_spans_by_id: Mapping[str, FontSpanResolution],
    ) -> tuple[list[ShapedRun], list[RunFontResolution]]:
        shaped: list[ShapedRun] = []
        run_resolutions: list[RunFontResolution] = []
        for run in runs:
            if not run.normalized_text or run.role == "space":
                continue
            existing_resolution = self.font_manager.resolve_run_font(
                resolution,
                run.normalized_text,
                run_id=run.run_id,
            )
            span = font_spans_by_id.get(run.run_id)
            use_span_resolution = bool(
                span is not None
                and span.selected_face is not None
                and (
                    span.span_id != span.logical_run_id
                    or existing_resolution.selected_face is None
                    or existing_resolution.selected_face.face_id
                    != span.selected_face.face_id
                )
            )
            if use_span_resolution and span is not None:
                run_resolution = RunFontResolution(
                    run_id=run.run_id,
                    text=run.normalized_text,
                    selected_face=span.selected_face,
                    coverage=span.coverage,
                    fallback_used=span.fallback_used,
                    fallback_index=span.fallback_index,
                    selection_reason=span.selection_reason,
                    missing_glyphs=list(span.coverage.missing_chars),
                    issues=list(span.issues),
                )
            else:
                run_resolution = existing_resolution
            run_resolutions.append(run_resolution)
            face = run_resolution.selected_face or resolution.primary_face
            if face is None:
                continue
            placement_mode = _run_audit(run, writing_mode, self.policy)["placement_mode"]
            shape_writing_mode = _shape_writing_mode_for_run(run, writing_mode, self.policy)
            shaped_run = self.shaper.shape_text(
                run.normalized_text,
                face=face,
                font_size=font_size,
                writing_mode=shape_writing_mode,
                language=run.language,
                script=run.script,
                direction=run.direction if run.direction == "rtl" else "",
            )
            shaped_metadata = {
                **dict(shaped_run.metadata),
                "run_id": run.run_id,
                "run_role": run.role,
                "grapheme_start": run.grapheme_start,
                "grapheme_end": run.grapheme_end,
                "block_writing_mode": writing_mode,
                "shape_writing_mode": shape_writing_mode,
                "inline_placement_mode": placement_mode,
                "punctuation_occurrences": list(run.metadata.get("punctuation_occurrences") or []),
                "symbol_occurrences": list(run.metadata.get("symbol_occurrences") or []),
                "token_ids": list(run.token_ids),
                "original_text": run.original_text,
                "translated_start": int(run.translated_start),
                "translated_end": int(run.translated_end),
                "lossless_tokens": list(run.metadata.get("lossless_tokens") or []),
                "run_font_resolution": run_resolution.to_audit_dict(),
                "font_fallback_used": bool(run_resolution.fallback_used),
            }
            if run.metadata.get("font_span_id"):
                shaped_metadata.update(
                    {
                        "logical_run_id": _logical_run_id_for(run),
                        "font_span_id": str(run.metadata.get("font_span_id")),
                        "source_cluster_map": _shaped_source_cluster_map(
                            run,
                            shaped_run,
                        ),
                    }
                )
            elif run.role in {"complex_script", "symbol"}:
                shaped_metadata["source_cluster_map"] = _shaped_source_cluster_map(
                    run,
                    shaped_run,
                )
            shaped.append(replace(shaped_run, metadata=shaped_metadata))
        return shaped, run_resolutions

    def _expand_runs_for_font_spans(
        self,
        runs: Sequence[InlineTextRun],
        resolution: FontResolution,
        *,
        writing_mode: str,
    ) -> tuple[list[InlineTextRun], list[FontSpanResolution]]:
        expanded: list[InlineTextRun] = []
        span_resolutions: list[FontSpanResolution] = []
        for run in runs:
            spans = self.font_manager.resolve_run_font_spans(
                resolution,
                run.normalized_text,
                run_id=run.run_id,
                script=run.script,
                direction=run.direction,
                role=run.role,
                writing_mode=writing_mode,
            )
            span_resolutions.extend(spans)
            if (
                len(spans) == 1
                and spans[0].span_id == run.run_id
                and spans[0].text == run.normalized_text
            ):
                expanded.append(run)
                continue
            for span in spans:
                provenance = _font_span_token_provenance(run, span)
                metadata = {
                    **dict(run.metadata),
                    **provenance["metadata"],
                    "logical_run_id": run.run_id,
                    "font_span_id": span.span_id,
                    "font_span_source_grapheme_start": span.source_grapheme_start,
                    "font_span_source_grapheme_end": span.source_grapheme_end,
                    "font_span_source_codepoint_start": span.source_codepoint_start,
                    "font_span_source_codepoint_end": span.source_codepoint_end,
                    "font_span_selection_reason": span.selection_reason,
                    "font_span_missing_clusters": list(span.missing_clusters),
                }
                expanded.append(
                    replace(
                        run,
                        run_id=span.span_id,
                        text=span.text,
                        normalized_text=span.text,
                        original_text=provenance["original_text"],
                        translated_start=provenance["translated_start"],
                        translated_end=provenance["translated_end"],
                        token_start=provenance["token_start"],
                        token_end=provenance["token_end"],
                        token_ids=provenance["token_ids"],
                        grapheme_start=(
                            run.grapheme_start + span.source_grapheme_start
                        ),
                        grapheme_end=(
                            run.grapheme_start + span.source_grapheme_end
                        ),
                        metadata=metadata,
                    )
                )
        return expanded, span_resolutions

    def _font_span_failure(
        self,
        plan: RenderLayerPlan,
        *,
        normalized: str,
        hard_bounds: list[int],
        resolved: FontResolution,
        logical_runs: Sequence[InlineTextRun],
        expanded_runs: Sequence[InlineTextRun],
        spans: Sequence[FontSpanResolution],
        issues: Sequence[str],
        shaped_runs: Sequence[ShapedRun] | None = None,
        notdef_run_ids: Sequence[str] | None = None,
    ) -> tuple[TypesetLayout, FitReport]:
        failure_issues = _unique(
            ["font_span_resolution_failed", *list(issues or [])]
        )
        layout, report = self._failed(
            plan,
            "font_span_resolution_failed",
            failure_issues,
            hard_bounds=hard_bounds,
        )
        span_audit = [item.to_audit_dict() for item in spans]
        metadata = {
            "typesetting_engine_version": "typesetting_engine_stage4_v1",
            "failure_status": "font_span_resolution_failed",
            "font_resolution": resolved.to_audit_dict(),
            "logical_inline_runs": [item.to_audit_dict() for item in logical_runs],
            "inline_runs": [item.to_audit_dict() for item in expanded_runs],
            "font_span_resolutions": span_audit,
            "font_span_text_conserved": (
                "".join(item.normalized_text for item in expanded_runs)
                == normalized
            ),
            "shaped_runs": [item.to_audit_dict() for item in list(shaped_runs or [])],
            "font_span_cluster_map": _font_span_cluster_ledger(
                expanded_runs,
                list(shaped_runs or []),
            ),
            "notdef_run_ids": _unique(list(notdef_run_ids or [])),
        }
        layout.normalized_text = normalized
        layout.original_text = str(plan.translated_text or "")
        layout.metadata = metadata
        report.issues = failure_issues
        report.metadata = dict(metadata)
        return layout, report

    def _layout_intent_box(
        self,
        *,
        plan: RenderLayerPlan,
        hard_bounds: list[int],
        target_box: list[int],
        writing_mode: str,
        font_size: int,
        text: str,
        runs: Sequence[InlineTextRun],
        shaped_runs: Sequence[ShapedRun],
    ) -> tuple[list[int], dict[str, Any], list[str]]:
        x, y, w, h = target_box
        hard_x, hard_y, hard_w, hard_h = hard_bounds
        source_box = bbox_from_value(plan.source_provenance_ref.get("source_contract_bbox") if isinstance(plan.source_provenance_ref, dict) else [])
        footprint_profile_selection = _source_text_footprint_profile_selection(
            plan,
            writing_mode,
        )
        reason_codes: list[str] = []
        evidence = "translated_text_natural_box"
        vertical_profile: dict[str, Any] = {}
        if writing_mode == "vertical":
            profile_items = _vertical_layout_items(
                runs,
                shaped_runs,
                self.policy,
                font_size,
                plan.resolved_render_style,
                plan=plan,
            )
            profile_content_width = max(
                (float(item.get("width", 0.0)) for item in profile_items),
                default=0.0,
            )
            profile_compact_only = _vertical_items_are_compact_sequences(
                profile_items
            )
            profile_column_width = _vertical_column_pitch(
                font_size,
                profile_content_width,
                compact_sequence_only=profile_compact_only,
            )
            profile_cell_height = max(
                _dominant_vertical_advance(shaped_runs)
                * _line_height(plan.resolved_render_style),
                max(
                    (
                        _vertical_item_cell_height(item)
                        for item in profile_items
                    ),
                    default=0.0,
                ),
                1.0,
            )
            vertical_profile = _vertical_layout_profile(
                plan=plan,
                box=target_box,
                font_size=font_size,
                text=text,
                shaped_runs=shaped_runs,
                policy=self.policy,
                item_count=len(profile_items),
                item_row_units=_vertical_items_row_units(profile_items),
                column_width=profile_column_width,
                cell_height=profile_cell_height,
            )
            natural_w, natural_h = _natural_box_size(
                text,
                writing_mode,
                font_size,
                plan.resolved_render_style,
                shaped_runs,
                desired_columns=int(vertical_profile.get("desired_columns") or 1),
                item_row_units=_vertical_items_row_units(profile_items),
                column_width=profile_column_width,
            )
            if vertical_profile.get("source_text_footprint_used"):
                reason_codes.append(
                    "source_text_footprint_used_for_vertical_layout"
                )
        else:
            natural_w, natural_h = _natural_box_size(text, writing_mode, font_size, plan.resolved_render_style, shaped_runs)
            if natural_w > max(1, w):
                estimated_lines = max(1, int(math.ceil(float(natural_w) / float(max(1, w)))))
                natural_w = max(1, w)
                natural_h = max(natural_h, int(math.ceil(float(font_size) * _line_height(plan.resolved_render_style) * estimated_lines)))
                reason_codes.append("horizontal_wrap_capacity_reserved_from_measured_runs")
        if w > max(natural_w * 2, natural_w + font_size) or h > max(natural_h * 2, natural_h + font_size):
            reason_codes.append("oversized_parent_bbox_not_used_as_fill_box")
        visual_alignment_center: list[float] = []
        metadata = plan.metadata if isinstance(plan.metadata, Mapping) else {}
        visual_center_candidate = _point_from_value(metadata.get("visual_alignment_center"))
        if visual_center_candidate and _point_inside_box(visual_center_candidate, hard_bounds):
            visual_alignment_center = visual_center_candidate
            reason_codes.append("visual_alignment_center_used_as_layout_intent")
        elif visual_center_candidate:
            reason_codes.append("visual_alignment_center_outside_hard_bounds_ignored")

        source_center: list[float] = []
        source_center_candidate = _center_of(source_box)
        if source_box and source_center_candidate and _point_inside_box(source_center_candidate, hard_bounds):
            source_center = source_center_candidate
            if source_box != target_box and not visual_alignment_center:
                reason_codes.append("source_contract_bbox_used_as_layout_alignment_prior")
            elif source_box == target_box:
                reason_codes.append("source_contract_bbox_equals_target_box")
        natural_w = min(max(1, natural_w), max(1, w))
        natural_h = min(max(1, natural_h), max(1, h))
        if visual_alignment_center:
            intent = [
                int(round(float(visual_alignment_center[0]) - float(natural_w) / 2.0)),
                int(round(float(visual_alignment_center[1]) - float(natural_h) / 2.0)),
                int(natural_w),
                int(natural_h),
            ]
            reason_codes.append("layout_intent_centered_on_visual_alignment_center")
        elif source_center:
            intent = [
                int(round(float(source_center[0]) - float(natural_w) / 2.0)),
                int(round(float(source_center[1]) - float(natural_h) / 2.0)),
                int(natural_w),
                int(natural_h),
            ]
            reason_codes.append("layout_intent_centered_on_source_contract_bbox")
        else:
            intent = [
                x + max(0, int(round((w - natural_w) / 2))),
                y + max(0, int(round((h - natural_h) / 2))),
                int(natural_w),
                int(natural_h),
            ]
        intent = _clamp_box(intent, [hard_x, hard_y, hard_w, hard_h])
        return intent, {
            "hard_bounds": list(hard_bounds),
            "target_box": list(target_box),
            "layout_intent_box": list(intent),
            "layout_intent_evidence": evidence,
            "source_contract_bbox": list(source_box),
            "source_contract_bbox_is_layout_box": False,
            "source_contract_bbox_is_alignment_prior": bool(source_center and not visual_alignment_center),
            "visual_alignment_center": list(visual_alignment_center),
            "visual_alignment_center_is_layout_prior": bool(visual_alignment_center),
            "source_text_footprint_profile_selection": footprint_profile_selection,
            "vertical_layout_profile": vertical_profile,
            "rejected_box_candidates": [],
        }, reason_codes

    def _layout_vertical(
        self,
        normalized: str,
        runs: Sequence[InlineTextRun],
        shaped_runs: Sequence[ShapedRun],
        breaks: Sequence[BreakOpportunity],
        box: list[int],
        font_size: int,
        style: dict[str, Any],
        plan: RenderLayerPlan,
        *,
        candidate_capacity_box: Sequence[int] | None = None,
    ) -> tuple[
        list[GlyphPlacement],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[int],
        str,
        list[str],
        dict[str, Any],
    ]:
        x, y, w, h = box
        line_height = _line_height(style)
        items = _vertical_layout_items(
            runs,
            shaped_runs,
            self.policy,
            font_size,
            style,
            plan=plan,
        )
        total_row_units = _vertical_items_row_units(items)
        shaped_cell_h = _dominant_vertical_advance(shaped_runs)
        cell_h = max(1.0, shaped_cell_h * line_height)
        cell_h = max(cell_h, max((_vertical_item_cell_height(item) for item in items), default=0.0))
        column_w = _vertical_column_pitch(
            font_size,
            max(
                (float(item.get("width", 0.0)) for item in items),
                default=0.0,
            ),
            compact_sequence_only=_vertical_items_are_compact_sequences(items),
        )
        profile = _vertical_layout_profile(
            plan=plan,
            box=box,
            font_size=font_size,
            text=normalized,
            shaped_runs=shaped_runs,
            policy=self.policy,
            item_count=len(items),
            item_row_units=total_row_units,
            column_width=column_w,
            cell_height=cell_h,
        )
        profile = dict(profile)
        columns_needed = max(1, int(profile.get("desired_columns") or 1)) if items else 1
        max_rows = max(1, int(math.floor(h / cell_h)))
        capacity_box = bbox_from_value(candidate_capacity_box) or list(box)
        capacity_width = int(capacity_box[2]) if capacity_box else int(w)
        layout_box_max_columns = max(1, int(math.floor(w / column_w)))
        physical_capacity_max_columns = max(
            1,
            min(
                max(1, len(items)),
                int(math.floor(capacity_width / column_w)),
            ),
        )
        source_group_upper_bound = int(
            profile.get("source_text_footprint_cross_axis_group_upper_bound") or 0
        )
        source_group_reliable = bool(
            profile.get("source_text_footprint_cross_axis_group_reliable")
        )
        source_group_clamped = bool(
            source_group_reliable and source_group_upper_bound > 0
        )
        source_preferred_max_columns = physical_capacity_max_columns
        if source_group_clamped:
            source_preferred_max_columns = min(
                physical_capacity_max_columns,
                source_group_upper_bound,
            )
        minimum_hard_fit_columns = max(
            1,
            int(math.ceil(total_row_units / float(max_rows))),
        )
        max_columns = physical_capacity_max_columns
        profile.update(
            {
                "layout_box_max_columns": int(layout_box_max_columns),
                "candidate_capacity_box": list(capacity_box),
                "candidate_physical_max_columns": int(
                    physical_capacity_max_columns
                ),
                "candidate_source_preferred_max_columns": int(
                    source_preferred_max_columns
                ),
                "candidate_minimum_hard_fit_columns": int(
                    minimum_hard_fit_columns
                ),
                "candidate_capacity_max_columns": int(max_columns),
                "candidate_capacity_source_group_clamped": False,
                "candidate_capacity_reason": "physical_target_hard_capacity",
            }
        )
        columns_needed = min(max(1, columns_needed), max(1, len(items)), max_columns)
        rows = max(1, int(math.ceil(total_row_units / columns_needed))) if items else 1
        while rows > max_rows and columns_needed < min(max_columns, max(1, len(items))):
            columns_needed += 1
            rows = max(1, int(math.ceil(total_row_units / columns_needed)))
        break_result = self.break_planner.plan_vertical(
            items,
            breaks,
            desired_columns=columns_needed,
            max_columns=min(max_columns, max(1, len(items))),
            max_rows=max_rows,
            profile=profile,
        )
        column_groups = break_result.groups
        grouping_meta = break_result.to_audit_dict()
        selected_columns = max(1, len(column_groups))
        source_preference_escape = bool(
            source_group_clamped
            and selected_columns > source_preferred_max_columns
        )
        hard_fit_required = bool(
            source_preference_escape
            and minimum_hard_fit_columns > source_preferred_max_columns
        )
        phrase_driven_expansion = bool(
            source_preference_escape and not hard_fit_required
        )
        if hard_fit_required:
            source_preference_escape_reason = (
                "hard_fit_required_beyond_source_group_preference"
            )
        elif phrase_driven_expansion:
            source_preference_escape_reason = (
                "legal_break_quality_expansion_beyond_source_group_preference"
            )
        elif source_group_clamped:
            source_preference_escape_reason = "within_source_group_preference"
        else:
            source_preference_escape_reason = "source_group_preference_unavailable"
        profile.update(
            {
                "candidate_capacity_selected_columns": int(selected_columns),
                "candidate_capacity_source_preference_escape": (
                    source_preference_escape
                ),
                "candidate_capacity_hard_fit_escape": hard_fit_required,
                "candidate_capacity_phrase_driven_expansion": (
                    phrase_driven_expansion
                ),
                "candidate_capacity_source_preference_escape_reason": (
                    source_preference_escape_reason
                ),
            }
        )
        grouping_meta.update(
            {
                "source_group_preferred_max_columns": int(
                    source_preferred_max_columns
                ),
                "physical_candidate_max_columns": int(max_columns),
                "source_group_preference_escape": source_preference_escape,
                "source_group_hard_fit_escape": hard_fit_required,
                "source_group_phrase_driven_expansion": (
                    phrase_driven_expansion
                ),
                "source_group_preference_escape_reason": (
                    source_preference_escape_reason
                ),
            }
        )
        rows = max((int(math.ceil(_vertical_items_row_units(group))) for group in column_groups), default=rows)
        columns_needed = max(1, len(column_groups))
        required_w = int(math.ceil(columns_needed * column_w))
        required_h = int(math.ceil(rows * cell_h))
        fit_status = "fits" if required_w <= w and required_h <= h and items and not break_result.issues else "overflow"
        issues: list[str] = list(break_result.issues)
        if fit_status != "fits":
            issues.append("layout_overflow")
            issues.append("line_break_fit_failure")
        run_modes = _run_modes(runs, self.policy, "vertical")
        if any(run.script == "Latn" and len(grapheme_clusters(run.text)) > self.policy.max_vertical_compact_latin_graphemes for run in runs):
            issues.append("long_latin_vertical_fit_review")
        block_w = float(columns_needed * column_w)
        block_h = float(rows * cell_h)
        base_x = x + max(0.0, (float(w) - block_w) / 2.0)
        base_y = y + max(0.0, (float(h) - block_h) / 2.0)
        placements: list[GlyphPlacement] = []
        cursor = 0
        for col, group in enumerate(column_groups):
            row_cursor = 0.0
            for item_index, item in enumerate(group):
                index = cursor + item_index
                row_units = _vertical_item_row_units(item)
                item_w = int(max(1, min(float(item.get("width") or column_w), max(1, w))))
                item_h = int(max(1, min(float(item.get("height") or (cell_h * row_units)), max(1, h))))
                slot_h = max(1.0, cell_h * row_units)
                raw_column_x = base_x + block_w - float((col + 1) * column_w)
                raw_px = int(round(raw_column_x + max(0.0, (column_w - item_w) / 2.0)))
                raw_py = int(round(base_y + row_cursor * cell_h + max(0.0, (slot_h - float(item_h)) / 2.0)))
                px = min(max(x, raw_px), x + w - item_w)
                py = min(max(y, raw_py), y + h - item_h)
                shaped_glyph = item.get("shaped_glyph")
                shaped_glyphs = list(item.get("shaped_glyphs") or ([] if shaped_glyph is None else [shaped_glyph]))
                placements.append(
                    GlyphPlacement(
                        text=str(item["text"]),
                        bbox=[px, py, item_w, item_h],
                        position=[float(px), float(py)],
                        font_family=str(item.get("font_face_id") or ""),
                        font_size=float(font_size),
                        advance=slot_h,
                        writing_mode="vertical",
                        metadata={
                            "cluster_index": index,
                            "column": col,
                            "row": int(math.floor(row_cursor)),
                            "row_units": float(row_units),
                            "row_span": int(math.ceil(row_units)),
                            "run_id": item.get("run_id", ""),
                            "token_ids": list(item.get("token_ids") or []),
                            "original_text": str(item.get("original_text") or ""),
                            "translated_start": int(item.get("translated_start") or 0),
                            "translated_end": int(item.get("translated_end") or 0),
                            "atomic_break": bool(item.get("atomic_break")),
                            **(
                                {
                                    "logical_run_id": item.get("logical_run_id", ""),
                                    "font_span_id": item.get("font_span_id", ""),
                                }
                                if item.get("font_span_id")
                                else {}
                            ),
                            "placement_mode": item.get("placement_mode", "vertical_glyph"),
                            "placement_source": "stage4_vertical_layout",
                            "shaped_glyph_id": int(shaped_glyph.glyph_id) if shaped_glyph else None,
                            "shaped_glyph_name": shaped_glyph.glyph_name if shaped_glyph else "",
                            "shaped_glyph_ids": [int(glyph.glyph_id) for glyph in shaped_glyphs],
                            "shaped_x_advance_total": float(item.get("x_advance", 0.0)),
                            "shaped_y_advance": float(shaped_glyph.y_advance) if shaped_glyph else 0.0,
                            "shaped_position_authority": bool(shaped_glyph),
                            "punctuation_occurrences": list(item.get("punctuation_occurrences") or []),
                            "symbol_occurrences": list(item.get("symbol_occurrences") or []),
                            "ellipsis_unit_count": int(item.get("ellipsis_unit_count") or 0),
                            "ellipsis_dot_count": int(item.get("ellipsis_dot_count") or 0),
                            "ellipsis_sequence_group_count": int(item.get("ellipsis_sequence_group_count") or 0),
                            "wave_unit_count": int(item.get("wave_unit_count") or 0),
                            "dash_unit_count": int(item.get("dash_unit_count") or 0),
                            "emphasis_symbol_count": int(item.get("emphasis_symbol_count") or 0),
                            "emphasis_sequence_group_count": int(
                                item.get("emphasis_sequence_group_count") or 0
                            ),
                            "font_face_id": str(item.get("font_face_id") or ""),
                            "font_path": str(item.get("font_path") or ""),
                            "font_fallback_used": bool(item.get("font_fallback_used")),
                            **(
                                {
                                    "source_punctuation_footprint_status": str(
                                        item.get(
                                            "source_punctuation_footprint_status"
                                        )
                                        or ""
                                    ),
                                    "source_punctuation_footprint_box": list(
                                        item.get(
                                            "source_punctuation_footprint_box"
                                        )
                                        or []
                                    ),
                                    "source_punctuation_footprint_fact_set_id": str(
                                        item.get(
                                            "source_punctuation_footprint_fact_set_id"
                                        )
                                        or ""
                                    ),
                                }
                                if item.get(
                                    "source_punctuation_footprint_status"
                                )
                                else {}
                            ),
                        },
                    )
                )
                row_cursor += row_units
            cursor += len(group)
        columns: list[dict[str, Any]] = []
        for idx in range(columns_needed):
            column_group = column_groups[idx] if idx < len(column_groups) else []
            raw_x = int(round(base_x + block_w - float((idx + 1) * column_w)))
            raw_y = int(round(base_y))
            raw_box = [raw_x, raw_y, int(max(1, math.ceil(column_w))), int(max(1, math.ceil(rows * cell_h)))]
            clipped = _clamp_box(raw_box, [x, y, w, h])
            columns.append(
                {
                    "column_index": idx,
                    "text": "".join(str(item.get("text") or "") for item in column_group),
                    "run_ids": [str(item.get("run_id") or "") for item in column_group],
                    "item_count": len(column_group),
                    "row_units": round(_vertical_items_row_units(column_group), 3),
                    "x": clipped[0],
                    "y": clipped[1],
                    "width": clipped[2],
                    "height": clipped[3],
                    "writing_mode": "vertical",
                    "raw_x": raw_x,
                    "raw_width": raw_box[2],
                    "layout_profile": profile,
                    "column_grouping": grouping_meta,
                    "centered_block_box": [int(round(base_x)), int(round(base_y)), int(max(1, math.ceil(block_w))), int(max(1, math.ceil(block_h)))],
                    "clipped_to_hard_bounds": clipped != raw_box,
                    "overflow_column": fit_status != "fits" and (raw_x < x or raw_x + raw_box[2] > x + w),
                }
            )
        measured = _union_bounds([item.bbox for item in placements]) or [x, y, min(w, required_w), min(h, int(rows * cell_h))]
        source_punctuation_boxes = [
            bbox_from_value(item.get("source_punctuation_footprint_box"))
            for item in items
            if str(item.get("source_punctuation_footprint_status") or "")
            == "applied_unambiguous_standalone_occurrence"
        ]
        source_punctuation_boxes = [
            item for item in source_punctuation_boxes if item
        ]
        alignment_bounds = (
            list(capacity_box)
            if len(source_punctuation_boxes) == 1
            else [x, y, w, h]
        )
        alignment_center, alignment_source = _layout_alignment_center(
            plan,
            alignment_bounds,
            source_punctuation_box=(
                source_punctuation_boxes[0]
                if len(source_punctuation_boxes) == 1
                else None
            ),
        )
        if alignment_center and measured:
            dx, dy = _measured_alignment_shift(
                measured,
                alignment_center,
                alignment_bounds,
            )
            if dx or dy:
                placements = [_shift_glyph_placement(item, dx, dy) for item in placements]
                columns = _shift_column_records(
                    columns,
                    dx,
                    dy,
                    alignment_bounds,
                )
                measured = _union_bounds([item.bbox for item in placements]) or measured
            if columns:
                for column in columns:
                    column["layout_visual_alignment"] = {
                        "source": alignment_source,
                        "alignment_center": [round(float(alignment_center[0]), 3), round(float(alignment_center[1]), 3)],
                        "shift": [int(dx), int(dy)],
                        "measured_bounds": list(measured),
                    }
        lines: list[dict[str, Any]] = []
        return placements, lines, columns, measured, fit_status, _unique(issues), grouping_meta

    def _retry_vertical_fit_within_hard_bounds(
        self,
        *,
        normalized: str,
        runs: Sequence[InlineTextRun],
        shaped_runs: Sequence[ShapedRun],
        breaks: Sequence[BreakOpportunity],
        plan: RenderLayerPlan,
        style: Mapping[str, Any],
        font_size: int,
        hard_bounds: Sequence[int],
        layout_intent_box: Sequence[int],
        box_model: Mapping[str, Any],
        break_plan: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        hard = bbox_from_value(hard_bounds)
        current = bbox_from_value(layout_intent_box)
        if not hard or not current:
            return None

        row_units = [float(value) for value in (break_plan.get("column_row_units") or [])]
        selected_columns = int(break_plan.get("selected_columns") or len(row_units) or 1)
        if not row_units or selected_columns <= 0:
            return None

        items = _vertical_layout_items(
            runs,
            shaped_runs,
            self.policy,
            font_size,
            style,
            plan=plan,
        )
        if not items:
            return None
        shaped_advance = max(1.0, _dominant_vertical_advance(shaped_runs))
        item_cell_height = max((_vertical_item_cell_height(item) for item in items), default=0.0)
        original_line_height = _line_height(dict(style or {}))
        effective_line_height = original_line_height
        max_row_units = max(row_units)
        hard_w = int(hard[2])
        hard_h = int(hard[3])

        cell_height = max(shaped_advance * effective_line_height, item_cell_height)
        needed_height = int(math.ceil(max_row_units * cell_height))
        reason_codes = ["layout_intent_expanded_for_legal_vertical_partition"]
        if (
            needed_height > hard_h
            and original_line_height > 1.0
            and _font_size_is_locked(style)
        ):
            compacted = min(
                original_line_height,
                float(hard_h) / max(1.0, max_row_units * shaped_advance),
            )
            if compacted >= 1.0:
                effective_line_height = max(1.0, compacted - 1e-6)
                cell_height = max(shaped_advance * effective_line_height, item_cell_height)
                needed_height = int(math.ceil(max_row_units * cell_height))
                reason_codes.append("line_height_compacted_for_hard_bound_fit")
                if _font_size_is_locked(style):
                    reason_codes.append("line_height_compacted_for_locked_size_fit")

        content_width = max((float(item.get("width", 0.0)) for item in items), default=0.0)
        column_width = _vertical_column_pitch(
            font_size,
            content_width,
            compact_sequence_only=_vertical_items_are_compact_sequences(items),
        )
        needed_width = int(math.ceil(float(selected_columns) * column_width))
        if needed_width > hard_w or needed_height > hard_h:
            return None

        retry_box = _expanded_box_within_hard_bounds(
            current,
            required_width=max(int(current[2]), needed_width),
            required_height=max(int(current[3]), needed_height),
            hard_bounds=hard,
        )
        retry_style = dict(style or {})
        retry_style["line_height"] = effective_line_height
        (
            placements,
            lines,
            columns,
            measured_bounds,
            fit_status,
            fit_issues,
            retry_break_plan,
        ) = self._layout_vertical(
            normalized,
            runs,
            shaped_runs,
            breaks,
            retry_box,
            font_size,
            retry_style,
            plan,
            candidate_capacity_box=hard,
        )
        if fit_status != "fits":
            return None

        updated_box_model = dict(box_model or {})
        updated_box_model["layout_intent_box"] = list(retry_box)
        updated_box_model["layout_intent_evidence"] = (
            "translated_text_natural_box_expanded_for_legal_vertical_partition"
        )
        updated_box_model["effective_line_height"] = round(effective_line_height, 6)
        if columns and isinstance(columns[0].get("layout_profile"), Mapping):
            updated_box_model["vertical_layout_profile"] = dict(columns[0]["layout_profile"])
        return {
            "placements": placements,
            "lines": lines,
            "columns": columns,
            "measured_bounds": measured_bounds,
            "fit_status": fit_status,
            "fit_issues": fit_issues,
            "break_plan": retry_break_plan,
            "layout_intent_box": list(retry_box),
            "box_model": updated_box_model,
            "reason_codes": reason_codes,
        }

    def _layout_horizontal(
        self,
        normalized: str,
        runs: Sequence[InlineTextRun],
        shaped_runs: Sequence[ShapedRun],
        breaks: Sequence[BreakOpportunity],
        box: list[int],
        font_size: int,
        style: dict[str, Any],
    ) -> tuple[
        list[GlyphPlacement],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[int],
        str,
        list[str],
        dict[str, Any],
    ]:
        x, y, w, h = box
        line_height_ratio = _line_height(style)
        line_height = max(1.0, font_size * line_height_ratio)
        line_pixel_extent = _horizontal_line_pixel_extent(
            font_size,
            line_height_ratio,
        )
        max_lines = max(1, int(math.floor(float(h) / line_height)))
        shaped_by_run = {
            str(item.metadata.get("run_id") or ""): item
            for item in shaped_runs
            if str(item.metadata.get("run_id") or "")
        }
        layout_items: list[dict[str, Any]] = []
        for run in runs:
            shaped = shaped_by_run.get(run.run_id)
            if run.role == "space":
                advance = font_size * 0.35
            else:
                advance = sum(max(0.0, glyph.x_advance) for glyph in shaped.glyphs) if shaped else font_size * len(run.text)
            layout_items.append(
                {
                    "text": run.normalized_text,
                    "run_id": run.run_id,
                    "advance": max(0.0, float(advance)),
                    "role": run.role,
                    "script": run.script,
                    "run": run,
                    "shaped_run": shaped,
                }
            )
        break_result = self.break_planner.plan_horizontal(
            layout_items,
            breaks,
            max_width=float(w),
            max_lines=max_lines,
        )
        line_groups = break_result.groups
        placements: list[GlyphPlacement] = []
        issues: list[str] = list(break_result.issues)
        lines: list[dict[str, Any]] = []
        selected_breaks = list(break_result.selected_breaks)
        for line_index, group in enumerate(line_groups):
            cursor_x = float(x)
            raw_y = float(y) + float(line_index) * line_height
            placement_height = max(1, min(int(h), line_pixel_extent))
            py = min(max(y, int(round(raw_y))), y + h - placement_height)
            for item in group:
                run = item.get("run")
                shaped = item.get("shaped_run")
                text = str(item.get("text") or "")
                advance = max(0.0, float(item.get("advance") or 0.0))
                if isinstance(run, InlineTextRun) and run.script == "Latn" and advance > w:
                    issues.append("word_overflow_break_applied")
                px = min(max(x, int(round(cursor_x))), x + w - 1)
                remaining_width = max(1.0, float(x + w - px))
                visible_width = int(max(1, min(max(1.0, advance), remaining_width)))
                placements.append(
                    GlyphPlacement(
                        text=text,
                        bbox=[px, py, visible_width, placement_height],
                        position=[float(px), float(py)],
                        font_family=shaped.font_face_id if isinstance(shaped, ShapedRun) else "",
                        font_size=float(font_size),
                        advance=advance,
                        writing_mode="horizontal",
                        metadata={
                            "run_id": str(item.get("run_id") or ""),
                            **(
                                _placement_font_span_metadata(run)
                                if isinstance(run, InlineTextRun)
                                else {}
                            ),
                            "line_index": line_index,
                            "space_run": bool(isinstance(run, InlineTextRun) and run.role == "space"),
                            "placement_source": "authoritative_break_opportunity_partition",
                            "font_face_id": shaped.font_face_id if isinstance(shaped, ShapedRun) else "",
                            "font_path": shaped.font_path if isinstance(shaped, ShapedRun) else "",
                            "font_fallback_used": bool((shaped.metadata or {}).get("font_fallback_used")) if isinstance(shaped, ShapedRun) else False,
                            "punctuation_occurrences": list((run.metadata or {}).get("punctuation_occurrences") or []) if isinstance(run, InlineTextRun) else [],
                            "symbol_occurrences": list((run.metadata or {}).get("symbol_occurrences") or []) if isinstance(run, InlineTextRun) else [],
                        },
                    )
                )
                cursor_x += advance
            line_width = sum(max(0.0, float(item.get("advance") or 0.0)) for item in group)
            line_record: dict[str, Any] = {
                "line_index": line_index,
                "text": "".join(str(item.get("text") or "") for item in group),
                "writing_mode": "horizontal",
                "run_ids": [str(item.get("run_id") or "") for item in group],
                "measured_advance": round(line_width, 3),
            }
            if line_index < len(selected_breaks):
                line_record["selected_break"] = dict(selected_breaks[line_index])
            lines.append(line_record)
        measured = _union_bounds([item.bbox for item in placements]) or [x, y, 1, 1]
        required_width = max(
            (sum(max(0.0, float(item.get("advance") or 0.0)) for item in group) for group in line_groups),
            default=0.0,
        )
        required_height = _horizontal_line_pixel_extent(
            font_size,
            line_height_ratio,
            line_count=len(line_groups),
        )
        overflow = required_width > float(w) or required_height > float(h) or bool(break_result.issues)
        if overflow:
            issues.append("layout_overflow")
        break_plan = break_result.to_audit_dict()
        return placements, lines, [], measured, "overflow" if overflow else "fits", _unique(issues), break_plan

    def _failed(
        self,
        plan: RenderLayerPlan,
        status: str,
        issues: list[str],
        *,
        hard_bounds: list[int] | None = None,
    ) -> tuple[TypesetLayout, FitReport]:
        layout = TypesetLayout(
            page_id=str(getattr(plan, "page_id", "")),
            layer_id=str(getattr(plan, "layer_id", "")),
            bundle_id=str(getattr(plan, "bundle_id", "")),
            parent_id=str(getattr(plan, "parent_id", "")),
            root_id=str(getattr(plan, "root_id", "")),
            selected_font_face="",
            selected_font_size=0,
            writing_mode=_normalize_mode(getattr(plan, "writing_mode", "")) or "auto",
            measured_bounds=list(hard_bounds or []),
            fit_status="failed",
            normalized_text="",
            original_text=str(getattr(plan, "translated_text", "") or ""),
            metadata={"typesetting_engine_version": "typesetting_engine_stage4_v1", "failure_status": status},
        )
        report = FitReport(
            page_id=layout.page_id,
            layer_id=layout.layer_id,
            bundle_id=layout.bundle_id,
            parent_id=layout.parent_id,
            root_id=layout.root_id,
            full_text_placed=False,
            fit_status="failed",
            issues=_unique([status, *issues]),
            metadata={"typesetting_engine_version": "typesetting_engine_stage4_v1", "failure_status": status},
        )
        return layout, report


def _missing_identity(plan: RenderLayerPlan) -> list[str]:
    return [
        name
        for name in ("page_id", "layer_id", "bundle_id", "parent_id", "root_id")
        if not str(getattr(plan, name, "") or "").strip()
    ]


def _font_size_from_style(style: dict[str, Any]) -> int:
    value = style.get("font_size") if isinstance(style, dict) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(number) or number <= 0.0:
        return 0
    return max(1, int(round(number)))


def _font_size_candidates(
    preferred: int,
    style: dict[str, Any],
    policy: TypesettingPolicy,
    target_box: Sequence[int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[int]:
    preferred = max(1, int(preferred))
    if _font_size_is_locked(style):
        return [preferred]
    minimum = _minimum_fit_font_size(preferred, style, policy)
    if minimum >= preferred:
        return [preferred]
    max_steps = max(1, int(policy.max_binary_fit_steps))
    step = max(1, int(math.ceil((preferred - minimum) / max_steps)))
    values = list(range(preferred, minimum - 1, -step))
    if values[-1] != minimum:
        values.append(minimum)
    return sorted(_unique_ints(values), reverse=True)


def _font_size_is_locked(style: Mapping[str, Any] | None) -> bool:
    values = dict(style or {})
    authority = str(values.get("font_size_authority") or "")
    if authority == "automated_style_arbitrator" or str(
        values.get("font_size_policy") or ""
    ) == "source_ink_preferred":
        return False
    return authority == "user_override" and (
        bool(values.get("font_size_locked"))
        or str(values.get("font_size_fallback_policy") or "")
        == "layout_failure_audit_only"
    )


def _minimum_fit_font_size(preferred: int, style: dict[str, Any], policy: TypesettingPolicy) -> int:
    preferred = max(1, int(preferred))
    readable = int(policy.min_readable_font_size)
    if isinstance(style, dict):
        profile = style.get("spacing_profile") if isinstance(style.get("spacing_profile"), dict) else {}
        for value in (
            style.get("minimum_readable_font_size"),
            profile.get("minimum_readable_font_size") if isinstance(profile, dict) else None,
        ):
            try:
                if value is not None and float(value) > 0:
                    readable = max(readable, int(round(float(value))))
            except (TypeError, ValueError):
                continue
    proportional_floor = int(round(preferred * 0.72))
    return min(preferred, max(int(policy.min_font_size), readable, proportional_floor))


def _normalize_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"vertical", "vert", "v", "ttb"}:
        return "vertical"
    if text in {"horizontal", "horiz", "h", "ltr", "rtl"}:
        return "horizontal"
    return text


def _language_hint(plan: RenderLayerPlan) -> str:
    style = plan.resolved_render_style if isinstance(plan.resolved_render_style, dict) else {}
    return str(style.get("language") or style.get("target_language") or "zh")


def _line_height(style: dict[str, Any]) -> float:
    try:
        return max(0.5, float(style.get("line_height", 1.0)))
    except (TypeError, ValueError):
        return 1.0


def _horizontal_line_pixel_extent(
    font_size: int,
    line_height: float,
    *,
    line_count: int = 1,
) -> int:
    """Return the outward integer extent for horizontal line-box geometry."""

    count = max(0, int(line_count))
    if count <= 0:
        return 0
    pitch = Decimal(str(max(1, int(font_size)))) * Decimal(str(line_height))
    pitch = max(Decimal("1"), pitch)
    return int(
        (pitch * Decimal(count)).to_integral_value(rounding=ROUND_CEILING)
    )


def _parent_effect_raster_guard(style: Mapping[str, Any] | None) -> float:
    values = dict(style or {})
    stroke_width = 0.0
    value = values.get("stroke_width")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            stroke_width = abs(number)
    return float(max(2, int(math.ceil(stroke_width)) + 2))


def _style_issues(runs: Sequence[InlineTextRun], writing_mode: str, policy: TypesettingPolicy) -> list[str]:
    issues: list[str] = []
    if writing_mode == "vertical":
        for run in runs:
            if run.script == "Latn" and len(grapheme_clusters(run.text)) > policy.max_vertical_compact_latin_graphemes:
                issues.append("long_latin_vertical_fit_review")
    return _unique(issues)


def _script_policy(runs: Sequence[InlineTextRun], writing_mode: str, policy: TypesettingPolicy) -> list[str]:
    policies = ["latin_letter_stack_forbidden"]
    for run in runs:
        if run.script in {"Hani", "Hira", "Kana", "Hang"} and writing_mode == "vertical":
            policies.append("cjk_vertical_grapheme_flow")
        if run.script == "Latn":
            policies.append("latin_word_run")
        if run.direction == "rtl" or run.role == "complex_script":
            policies.append("complex_script_shaped")
    return _unique(policies)


def _run_audit(run: InlineTextRun, writing_mode: str, policy: TypesettingPolicy) -> dict[str, Any]:
    record = run.to_audit_dict()
    placement = "standard_run"
    letter_stack = False
    if writing_mode == "vertical" and _is_vertical_emphasis_sequence(run):
        placement = "vertical_emphasis_sequence"
    elif writing_mode == "vertical" and run.script == "Latn":
        count = len(grapheme_clusters(run.text))
        if count <= policy.max_tate_chu_yoko_graphemes:
            placement = "tate_chu_yoko"
        else:
            placement = "rotated_latin_run"
    elif writing_mode == "vertical" and run.role in {"numeric_token", "symbol"}:
        placement = "vertical_inline_run"
    record.update({"placement_mode": placement, "letter_stack": letter_stack})
    return record


def _run_modes(runs: Sequence[InlineTextRun], policy: TypesettingPolicy, writing_mode: str) -> dict[str, str]:
    return {run.run_id: _run_audit(run, writing_mode, policy)["placement_mode"] for run in runs}


def _shape_writing_mode_for_run(run: InlineTextRun, block_writing_mode: str, policy: TypesettingPolicy) -> str:
    if block_writing_mode != "vertical":
        return block_writing_mode
    placement = _run_audit(run, block_writing_mode, policy)["placement_mode"]
    if placement in {
        "tate_chu_yoko",
        "rotated_latin_run",
        "vertical_inline_run",
        "vertical_emphasis_sequence",
    }:
        return "horizontal"
    if run.role in {"complex_script", "symbol"}:
        return "horizontal"
    return "vertical"


def _natural_box_size(
    text: str,
    writing_mode: str,
    font_size: int,
    style: dict[str, Any],
    shaped_runs: Sequence[ShapedRun] | None = None,
    *,
    desired_columns: int | None = None,
    item_row_units: float | None = None,
    column_width: float | None = None,
) -> tuple[int, int]:
    count = max(1, len(grapheme_clusters(text)))
    try:
        vertical_units = float(
            item_row_units if item_row_units is not None else count
        )
    except (TypeError, ValueError):
        vertical_units = float(count)
    if not math.isfinite(vertical_units) or vertical_units <= 0.0:
        vertical_units = float(count)
    line_height = _line_height(style)
    if writing_mode == "horizontal":
        shaped_width = 0.0
        for run in shaped_runs or []:
            shaped_width += sum(max(0.0, float(glyph.x_advance)) for glyph in run.glyphs)
        if shaped_width > 0.0:
            shaped_width += sum(1 for cluster in grapheme_clusters(text) if cluster.isspace()) * font_size * 0.35
            return int(math.ceil(shaped_width)), _horizontal_line_pixel_extent(
                font_size,
                line_height,
            )
        return int(max(1, min(count, 12) * font_size * 0.75)), _horizontal_line_pixel_extent(
            font_size,
            line_height,
        )
    inline_width = 0.0
    for run in shaped_runs or []:
        if (run.metadata or {}).get("shape_writing_mode") == "horizontal":
            inline_width = max(inline_width, sum(max(0.0, abs(float(glyph.x_advance))) for glyph in run.glyphs))
    resolved_column_width = float(
        column_width or _vertical_column_pitch(font_size, inline_width)
    )
    cell_height = max(font_size * line_height, _dominant_vertical_advance(shaped_runs or []) * line_height)
    columns = max(1, min(count, int(desired_columns or 1)))
    rows = max(1, int(math.ceil(vertical_units / columns)))
    if columns > 1 and _text_has_no_column_start_punctuation(text):
        rows += 1
    return int(max(1, math.ceil(columns * resolved_column_width))), int(max(1, math.ceil(rows * cell_height)))


def _vertical_layout_profile(
    *,
    plan: RenderLayerPlan,
    box: Sequence[int],
    font_size: int,
    text: str,
    shaped_runs: Sequence[ShapedRun],
    policy: TypesettingPolicy,
    item_count: int | None = None,
    item_row_units: float | None = None,
    column_width: float | None = None,
    cell_height: float | None = None,
) -> dict[str, Any]:
    target = bbox_from_value(box)
    count = max(1, int(item_count if item_count is not None else len(grapheme_clusters(text))))
    try:
        translated_row_units = float(
            item_row_units if item_row_units is not None else count
        )
    except (TypeError, ValueError):
        translated_row_units = float(count)
    if not math.isfinite(translated_row_units) or translated_row_units <= 0.0:
        translated_row_units = float(count)
    if not target:
        return {
            "desired_columns": 1,
            "max_columns": 1,
            "max_rows": 1,
            "item_count": int(count),
            "translated_row_units": round(translated_row_units, 3),
            "source_text_footprint_available": False,
            "source_text_footprint_used": False,
            "reason": "missing_target_box",
        }
    _x, _y, w, h = target
    inline_width = 0.0
    for run in shaped_runs or []:
        if (run.metadata or {}).get("shape_writing_mode") == "horizontal":
            inline_width = max(inline_width, sum(max(0.0, abs(float(glyph.x_advance))) for glyph in run.glyphs))
    column_w = float(column_width or _vertical_column_pitch(font_size, inline_width))
    cell_h = float(cell_height or max(_dominant_vertical_advance(shaped_runs) * _line_height(plan.resolved_render_style), font_size * _line_height(plan.resolved_render_style), 1.0))
    max_columns = max(1, min(count, int(math.floor(max(1, w) / max(1.0, column_w)))))
    max_rows = max(1, int(math.floor(max(1, h) / max(1.0, cell_h))))
    style = dict(plan.resolved_render_style or {})
    footprint, axis_profile = _source_text_footprint_axis_profile(plan, "ttb")
    footprint_available = bool(footprint)
    source_group_reliable = bool(
        axis_profile.get("cross_axis_group_count_reliable")
    )
    source_inline_reliable = bool(
        axis_profile.get("inline_capacity_reliable")
    )
    try:
        source_group_upper_bound = (
            max(0, int(axis_profile.get("cross_axis_group_count") or 0))
            if source_group_reliable
            else 0
        )
    except (TypeError, ValueError):
        source_group_upper_bound = 0
        source_group_reliable = False
    try:
        source_inline_capacity = (
            max(0, int(axis_profile.get("inline_capacity") or 0))
            if source_inline_reliable
            else 0
        )
    except (TypeError, ValueError):
        source_inline_capacity = 0
        source_inline_reliable = False
    if source_group_reliable and source_group_upper_bound <= 0:
        source_group_reliable = False
    if source_inline_reliable and source_inline_capacity <= 0:
        source_inline_reliable = False
    semantic_class = str(style.get("semantic_class") or style.get("source_role") or plan.role or "")
    source_role = str(style.get("source_role") or plan.role or "")
    footprint_used = False
    if translated_row_units <= 3.0:
        desired = 1
        reason = "short_vertical_text_single_column"
    elif source_inline_reliable and source_group_reliable:
        desired = int(
            math.ceil(
                translated_row_units
                / float(max(1, source_inline_capacity))
            )
        )
        if source_group_reliable:
            desired = min(desired, source_group_upper_bound)
        reason = "source_text_footprint_inline_capacity"
        footprint_used = True
    elif source_group_reliable:
        target_estimate = max(
            1,
            int(
                round(
                    math.sqrt(
                        float(count)
                        * max(1.0, float(w))
                        / max(1.0, float(h))
                    )
                )
            ),
        )
        desired = min(source_group_upper_bound, target_estimate)
        reason = "target_aspect_source_text_footprint_group_upper_bound"
        footprint_used = True
    else:
        desired = max(
            1,
            int(
                round(
                    math.sqrt(
                        float(count)
                        * max(1.0, float(w))
                        / max(1.0, float(h))
                    )
                )
            ),
        )
        reason = "target_aspect_column_estimate"
    desired = min(max(1, desired), max_columns)
    initial_desired = desired
    while (
        desired < max_columns
        and math.ceil(translated_row_units / float(desired)) > max_rows
    ):
        desired += 1
    expanded_for_height_fit = desired > initial_desired
    if expanded_for_height_fit:
        reason = f"{reason}_expanded_for_height_fit"
    return {
        "desired_columns": int(max(1, desired)),
        "initial_desired_columns": int(max(1, initial_desired)),
        "max_columns": int(max_columns),
        "max_rows": int(max_rows),
        "item_count": int(count),
        "translated_row_units": round(translated_row_units, 3),
        "column_width": round(column_w, 3),
        "cell_height": round(cell_h, 3),
        "source_text_footprint_available": footprint_available,
        "source_text_footprint_used": footprint_used,
        "source_text_footprint_profile_direction": "ttb",
        "source_text_footprint_profile_available": bool(axis_profile),
        "source_text_footprint_cross_axis_group_upper_bound": int(
            source_group_upper_bound
        ),
        "source_text_footprint_cross_axis_group_reliable": (
            source_group_reliable
        ),
        "source_text_footprint_inline_capacity": int(source_inline_capacity),
        "source_text_footprint_inline_capacity_reliable": (
            source_inline_reliable
        ),
        "source_text_footprint_inline_capacity_provenance": str(
            axis_profile.get("inline_capacity_provenance") or ""
        ),
        "source_text_footprint_fact_set_id": str(
            footprint.get("fact_set_id") or ""
        ),
        "source_text_footprint_confidence": float(
            axis_profile.get("confidence") or 0.0
        ),
        "source_text_footprint_reason": str(axis_profile.get("reason") or ""),
        "expanded_for_height_fit": expanded_for_height_fit,
        "semantic_class": semantic_class,
        "source_role": source_role,
        "reason": reason,
    }


def _source_text_footprint_axis_profile(
    plan: RenderLayerPlan,
    direction: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    footprint = validated_source_text_footprint_ref(plan)
    profiles = footprint.get("axis_profiles")
    profile = profiles.get(direction) if isinstance(profiles, Mapping) else {}
    if not isinstance(profile, Mapping):
        profile = {}
    if str(profile.get("writing_direction") or "") != direction:
        return footprint, {}
    return footprint, dict(profile)


def _source_text_footprint_profile_selection(
    plan: RenderLayerPlan,
    writing_mode: str,
) -> dict[str, Any]:
    direction = {
        "vertical": "ttb",
        "horizontal": "ltr",
    }.get(str(writing_mode or ""), "")
    footprint, profile = _source_text_footprint_axis_profile(plan, direction)
    direction_evidence = footprint.get("writing_direction_evidence")
    selection_authority = (
        str(direction_evidence.get("selection_authority") or "")
        if isinstance(direction_evidence, Mapping)
        else ""
    )
    if not footprint:
        application_status = "validated_source_text_footprint_unavailable"
    elif not profile:
        application_status = "resolved_direction_profile_unavailable"
    elif writing_mode == "vertical":
        application_status = "vertical_initial_group_preference_available"
    else:
        application_status = (
            "horizontal_profile_transported_line_preference_hook_unavailable"
        )
    return {
        "resolved_writing_mode": str(writing_mode or ""),
        "selected_profile_direction": direction,
        "selection_authority": selection_authority,
        "footprint_available": bool(footprint),
        "profile_available": bool(profile),
        "fact_set_id": str(footprint.get("fact_set_id") or ""),
        "cross_axis_group_count": int(
            profile.get("cross_axis_group_count") or 0
        ),
        "cross_axis_group_count_reliable": bool(
            profile.get("cross_axis_group_count_reliable")
        ),
        "inline_capacity": int(profile.get("inline_capacity") or 0),
        "inline_capacity_reliable": bool(
            profile.get("inline_capacity_reliable")
        ),
        "application_status": application_status,
        "footprint_does_not_select_writing_mode": True,
    }


def _expanded_box_within_hard_bounds(
    box: Sequence[int],
    *,
    required_width: int,
    required_height: int,
    hard_bounds: Sequence[int],
) -> list[int]:
    current = bbox_from_value(box)
    hard = bbox_from_value(hard_bounds)
    if not current or not hard:
        return list(current or hard or [])
    hx, hy, hw, hh = hard
    width = min(hw, max(int(current[2]), int(required_width)))
    height = min(hh, max(int(current[3]), int(required_height)))
    center_x = float(current[0]) + float(current[2]) / 2.0
    center_y = float(current[1]) + float(current[3]) / 2.0
    x = int(round(center_x - float(width) / 2.0))
    y = int(round(center_y - float(height) / 2.0))
    x = min(max(hx, x), hx + hw - width)
    y = min(max(hy, y), hy + hh - height)
    return [x, y, width, height]


def _vertical_column_pitch(
    font_size: int,
    content_width: float,
    *,
    compact_sequence_only: bool = False,
) -> float:
    base = max(1.0, float(font_size))
    width = max(1.0, float(content_width or 0.0))
    if compact_sequence_only:
        return max(base * 0.38, width + base * 0.10)
    return max(base * 1.24, width + base * 0.2)


def _vertical_item_row_units(item: Mapping[str, Any]) -> float:
    try:
        value = float(item.get("row_units", 1.0))
    except Exception:
        value = 1.0
    return max(1.0, value)


def _vertical_items_row_units(items: Sequence[Mapping[str, Any]]) -> float:
    return sum(_vertical_item_row_units(item) for item in (items or []))


def _vertical_items_are_compact_sequences(
    items: Sequence[Mapping[str, Any]],
) -> bool:
    return bool(items) and all(
        bool(item.get("compact_vertical_sequence")) for item in items
    )


def _vertical_item_cell_height(item: Mapping[str, Any]) -> float:
    row_units = _vertical_item_row_units(item)
    try:
        height = float(item.get("height", 0.0))
    except Exception:
        height = 0.0
    if height <= 0.0:
        return 0.0
    return height / row_units


def _text_has_no_column_start_punctuation(text: str) -> bool:
    return any(cluster[:1] in _VERTICAL_NO_COLUMN_START for cluster in grapheme_clusters(str(text or "")))


def _vertical_item_is_centered_punctuation(text: str) -> bool:
    value = str(text or "")
    return value in VERTICAL_CENTERED_PUNCTUATION_CHARS


def _box_inside(box: Sequence[int], container: Sequence[int]) -> bool:
    bx, by, bw, bh = [int(v) for v in box[:4]]
    cx, cy, cw, ch = [int(v) for v in container[:4]]
    return bx >= cx and by >= cy and bx + bw <= cx + cw and by + bh <= cy + ch


def _clamp_box(box: Sequence[int], container: Sequence[int]) -> list[int]:
    bx, by, bw, bh = [int(v) for v in box[:4]]
    cx, cy, cw, ch = [int(v) for v in container[:4]]
    bw = max(1, min(bw, cw))
    bh = max(1, min(bh, ch))
    bx = min(max(cx, bx), cx + cw - bw)
    by = min(max(cy, by), cy + ch - bh)
    return [int(bx), int(by), int(bw), int(bh)]


def _union_bounds(boxes: Sequence[Sequence[int]]) -> list[int]:
    clean = [bbox_from_value(box) for box in boxes]
    clean = [box for box in clean if box]
    if not clean:
        return []
    left = min(box[0] for box in clean)
    top = min(box[1] for box in clean)
    right = max(box[0] + box[2] for box in clean)
    bottom = max(box[1] + box[3] for box in clean)
    return [left, top, max(1, right - left), max(1, bottom - top)]


def _center_of(box: Sequence[int]) -> list[float]:
    bbox = bbox_from_value(box)
    if not bbox:
        return []
    return [float(bbox[0] + bbox[2] / 2), float(bbox[1] + bbox[3] / 2)]


def _point_inside_box(point: Sequence[float], box: Sequence[int]) -> bool:
    if (
        not isinstance(point, Sequence)
        or isinstance(point, (str, bytes, bytearray))
        or len(point) < 2
    ):
        return False
    bbox = bbox_from_value(box)
    if not bbox:
        return False
    px, py = float(point[0]), float(point[1])
    bx, by, bw, bh = bbox
    return float(bx) <= px <= float(bx + bw) and float(by) <= py <= float(by + bh)


def _point_from_value(value: Any) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) < 2
    ):
        return []
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return []


def _layout_alignment_center(
    plan: RenderLayerPlan,
    box: Sequence[int],
    *,
    source_punctuation_box: Sequence[int] | None = None,
) -> tuple[list[float], str]:
    metadata = plan.metadata if isinstance(plan.metadata, Mapping) else {}
    explicit_center = _point_from_value(metadata.get("visual_alignment_center"))
    if explicit_center and _point_inside_box(explicit_center, box):
        return explicit_center, "visual_alignment_center"

    punctuation_center = _center_of(source_punctuation_box or [])
    if punctuation_center and _point_inside_box(punctuation_center, box):
        return punctuation_center, "validated_source_punctuation_footprint_center"

    source_box = bbox_from_value(plan.source_provenance_ref.get("source_contract_bbox") if isinstance(plan.source_provenance_ref, dict) else [])
    source_center = _center_of(source_box)
    if source_center and _point_inside_box(source_center, box):
        return source_center, "source_contract_bbox_center"
    return [], ""


def _measured_alignment_shift(measured: Sequence[int], center: Sequence[float], bounds: Sequence[int]) -> tuple[int, int]:
    measured_box = bbox_from_value(measured)
    bounds_box = bbox_from_value(bounds)
    alignment_center = _point_from_value(center)
    if not measured_box or not bounds_box or not alignment_center:
        return 0, 0
    mx, my = _center_of(measured_box)
    dx = int(round(float(alignment_center[0]) - float(mx)))
    dy = int(round(float(alignment_center[1]) - float(my)))
    bx, by, bw, bh = bounds_box
    shifted_x = measured_box[0] + dx
    shifted_y = measured_box[1] + dy
    if shifted_x < bx:
        dx += bx - shifted_x
    if shifted_x + measured_box[2] > bx + bw:
        dx -= shifted_x + measured_box[2] - (bx + bw)
    if shifted_y < by:
        dy += by - shifted_y
    if shifted_y + measured_box[3] > by + bh:
        dy -= shifted_y + measured_box[3] - (by + bh)
    return int(dx), int(dy)


def _shift_glyph_placement(placement: GlyphPlacement, dx: int, dy: int) -> GlyphPlacement:
    bbox = bbox_from_value(placement.bbox)
    shifted_bbox = [bbox[0] + int(dx), bbox[1] + int(dy), bbox[2], bbox[3]] if bbox else list(placement.bbox)
    position = list(placement.position or [])
    if len(position) >= 2:
        position = [float(position[0]) + float(dx), float(position[1]) + float(dy), *position[2:]]
    metadata = dict(placement.metadata or {})
    metadata["layout_visual_alignment_shift"] = [int(dx), int(dy)]
    return replace(placement, bbox=shifted_bbox, position=position, metadata=metadata)


def _shift_column_records(
    columns: Sequence[dict[str, Any]],
    dx: int,
    dy: int,
    bounds_box: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    shifted: list[dict[str, Any]] = []
    for column in columns:
        item = dict(column)
        for key in ("x", "raw_x"):
            if key in item:
                item[key] = int(item[key]) + int(dx)
        for key in ("y", "raw_y"):
            if key in item:
                item[key] = int(item[key]) + int(dy)
        centered = bbox_from_value(item.get("centered_block_box"))
        if centered:
            item["centered_block_box"] = [centered[0] + int(dx), centered[1] + int(dy), centered[2], centered[3]]
        if bounds_box:
            display_box = bbox_from_value([item.get("x"), item.get("y"), item.get("width"), item.get("height")])
            if display_box:
                clipped = _clamp_box(display_box, bounds_box)
                item["x"], item["y"], item["width"], item["height"] = clipped
                item["clipped_to_hard_bounds"] = bool(item.get("clipped_to_hard_bounds")) or clipped != display_box
        shifted.append(item)
    return shifted


def _source_standalone_punctuation_footprint_hint(
    plan: RenderLayerPlan,
    runs: Sequence[InlineTextRun],
    run: InlineTextRun,
) -> dict[str, Any]:
    """Match one compact translated occurrence to one source occurrence.

    The adapter owns footprint validation and the arbitrator owns style. This
    helper supplies only an advisory inline presentation span when occurrence
    identity is unambiguous; every other case stays on font metrics.
    """

    expected_kind = {
        "ellipsis_sequence": "ellipsis",
        "dash_sequence": "dash",
        "wave_sequence": "wave",
    }.get(str(run.role or ""), "")
    if not expected_kind:
        return {}

    visible_runs = [
        candidate
        for candidate in list(runs or [])
        if candidate.role != "space" and str(candidate.text or "").strip()
    ]
    if (
        len(visible_runs) != 1
        or str(visible_runs[0].run_id or "") != str(run.run_id or "")
    ):
        return {"status": "fallback_ambiguous_translated_occurrence"}

    translated_occurrences = [
        dict(item)
        for item in list(run.metadata.get("punctuation_occurrences") or [])
        if isinstance(item, Mapping)
    ]
    if (
        len(translated_occurrences) != 1
        or str(translated_occurrences[0].get("kind") or "") != expected_kind
    ):
        return {"status": "fallback_ambiguous_translated_occurrence"}

    source_tokens = [
        token
        for token in build_lossless_text_tokens(plan.source_text_summary)
        if str(token.original_text or "").strip()
    ]
    if len(source_tokens) != 1:
        return {"status": "fallback_ambiguous_source_occurrence"}
    source_token = source_tokens[0]
    if not isinstance(source_token, PunctuationToken):
        return {"status": "fallback_incompatible_source_occurrence"}
    if str(source_token.punctuation_kind or "") != expected_kind:
        return {"status": "fallback_incompatible_source_occurrence"}

    footprint = validated_source_text_footprint_ref(plan)
    if not footprint:
        return {"status": "fallback_validated_source_footprint_unavailable"}
    source_box = bbox_from_value(footprint.get("union_bbox_page_xywh"))
    hard_bounds = bbox_from_value(plan.hard_bounds)
    if not source_box or not hard_bounds:
        return {"status": "fallback_source_footprint_geometry_unavailable"}

    source_left, source_top, source_width, source_height = source_box
    hard_left, hard_top, hard_width, hard_height = hard_bounds
    bounded_left = max(source_left, hard_left)
    bounded_top = max(source_top, hard_top)
    bounded_right = min(
        source_left + source_width,
        hard_left + hard_width,
    )
    bounded_bottom = min(
        source_top + source_height,
        hard_top + hard_height,
    )
    if bounded_right <= bounded_left or bounded_bottom <= bounded_top:
        return {"status": "fallback_source_footprint_outside_hard_bounds"}
    bounded_box = [
        int(bounded_left),
        int(bounded_top),
        int(bounded_right - bounded_left),
        int(bounded_bottom - bounded_top),
    ]
    return {
        "status": "applied_unambiguous_standalone_occurrence",
        "box": bounded_box,
        "inline_span_px": int(bounded_box[3]),
        "fact_set_id": str(footprint.get("fact_set_id") or ""),
    }


def _vertical_layout_items(
    runs: Sequence[InlineTextRun],
    shaped_runs: Sequence[ShapedRun],
    policy: TypesettingPolicy,
    font_size: int,
    style: Mapping[str, Any] | None = None,
    *,
    plan: RenderLayerPlan,
) -> list[dict[str, Any]]:
    shaped_by_run = {
        str(run.metadata.get("run_id") or ""): run
        for run in shaped_runs
        if str(run.metadata.get("run_id") or "")
    }
    items: list[dict[str, Any]] = []
    for run in runs:
        if run.role == "space":
            continue
        shaped = shaped_by_run.get(run.run_id)
        glyphs = list(shaped.glyphs if shaped else [])
        if run.role in {"ellipsis_sequence", "dash_sequence", "wave_sequence", "punctuation_sequence"}:
            item_width, item_height, x_advance = _vertical_atomic_size(
                run,
                glyphs,
                policy,
                font_size,
                style,
            )
            source_footprint_hint = _source_standalone_punctuation_footprint_hint(
                plan,
                runs,
                run,
            )
            if source_footprint_hint.get("status") == (
                "applied_unambiguous_standalone_occurrence"
            ):
                item_height = max(
                    float(item_height),
                    float(source_footprint_hint.get("inline_span_px") or 0.0),
                )
            row_units = _vertical_sequence_row_units(run)
            items.append(
                {
                    "text": run.text,
                    "run_id": run.run_id,
                    **_placement_font_span_metadata(run),
                    "placement_mode": (
                        "vertical_emphasis_sequence"
                        if _is_vertical_emphasis_sequence(run)
                        else f"vertical_{run.role}"
                    ),
                    "shaped_glyph": glyphs[0] if glyphs else None,
                    "shaped_glyphs": glyphs,
                    "width": item_width,
                    "height": item_height,
                    "row_units": row_units,
                    "compact_vertical_sequence": bool(
                        run.role
                        in {"ellipsis_sequence", "dash_sequence", "wave_sequence"}
                    ),
                    "x_advance": x_advance,
                    "punctuation_occurrences": list(run.metadata.get("punctuation_occurrences") or []),
                    "symbol_occurrences": list(run.metadata.get("symbol_occurrences") or []),
                    "ellipsis_unit_count": _occurrence_unit_count(run, "ellipsis"),
                    "ellipsis_dot_count": _occurrence_dot_count(run, "ellipsis"),
                    "ellipsis_sequence_group_count": _occurrence_sequence_group_count(run, "ellipsis"),
                    "wave_unit_count": _occurrence_unit_count(run, "wave"),
                    "dash_unit_count": _occurrence_unit_count(run, "dash"),
                    "emphasis_symbol_count": _occurrence_unit_count(run, "emphasis_punctuation"),
                    "emphasis_sequence_group_count": _occurrence_sequence_group_count(
                        run,
                        "emphasis_punctuation",
                    ),
                    "font_face_id": shaped.font_face_id if shaped else "",
                    "font_path": shaped.font_path if shaped else "",
                    "font_fallback_used": bool((shaped.metadata or {}).get("font_fallback_used")) if shaped else False,
                    **(
                        {
                            "source_punctuation_footprint_status": str(
                                source_footprint_hint.get("status") or ""
                            ),
                            "source_punctuation_footprint_box": list(
                                source_footprint_hint.get("box") or []
                            ),
                            "source_punctuation_footprint_fact_set_id": str(
                                source_footprint_hint.get("fact_set_id") or ""
                            ),
                        }
                        if source_footprint_hint.get("status")
                        else {}
                    ),
                }
            )
            continue
        if _vertical_run_is_atomic(run, policy):
            item_width, item_height, x_advance = _vertical_atomic_size(
                run,
                glyphs,
                policy,
                font_size,
                style,
            )
            items.append(
                {
                    "text": run.text,
                    "run_id": run.run_id,
                    **_placement_font_span_metadata(run),
                    "placement_mode": _run_audit(run, "vertical", policy)["placement_mode"],
                    "shaped_glyph": glyphs[0] if glyphs else None,
                    "shaped_glyphs": glyphs,
                    "width": item_width,
                    "height": item_height,
                    "row_units": 1.0,
                    "x_advance": x_advance,
                    "punctuation_occurrences": list(run.metadata.get("punctuation_occurrences") or []),
                    "symbol_occurrences": list(run.metadata.get("symbol_occurrences") or []),
                    "font_face_id": shaped.font_face_id if shaped else "",
                    "font_path": shaped.font_path if shaped else "",
                    "font_fallback_used": bool((shaped.metadata or {}).get("font_fallback_used")) if shaped else False,
                }
            )
            continue
        clusters = grapheme_clusters(run.text)
        for offset, cluster in enumerate(clusters):
            glyph = glyphs[offset] if offset < len(glyphs) else (glyphs[0] if glyphs else None)
            glyph_w = abs(float(glyph.x_advance)) if glyph else 0.0
            glyph_h = abs(float(glyph.y_advance)) if glyph else 0.0
            if glyph_w <= 0.0:
                glyph_w = glyph_h if glyph_h > 0.0 else float(font_size)
            if glyph_h <= 0.0:
                glyph_h = float(font_size)
            placement_mode = "vertical_glyph"
            if _vertical_item_is_centered_punctuation(cluster):
                glyph_w = float(font_size)
                glyph_h = float(font_size)
                placement_mode = "vertical_punctuation"
            items.append(
                {
                    "text": cluster,
                    "run_id": run.run_id,
                    **_placement_font_span_metadata(run),
                    "placement_mode": placement_mode,
                    "shaped_glyph": glyph,
                    "shaped_glyphs": [glyph] if glyph else [],
                    "width": max(1.0, glyph_w),
                    "height": max(1.0, glyph_h),
                    "row_units": 1.0,
                    "x_advance": abs(float(glyph.x_advance)) if glyph else 0.0,
                    "punctuation_occurrences": list(run.metadata.get("punctuation_occurrences") or []),
                    "symbol_occurrences": list(run.metadata.get("symbol_occurrences") or []),
                    "font_face_id": shaped.font_face_id if shaped else "",
                    "font_path": shaped.font_path if shaped else "",
                    "font_fallback_used": bool((shaped.metadata or {}).get("font_fallback_used")) if shaped else False,
                }
            )
    return items


def _vertical_sequence_row_units(run: InlineTextRun) -> float:
    occurrence_kind = {
        "ellipsis_sequence": "ellipsis",
        "dash_sequence": "dash",
        "wave_sequence": "wave",
    }.get(run.role)
    if occurrence_kind:
        return float(max(1, _occurrence_unit_count(run, occurrence_kind)))
    return 1.0


def _vertical_atomic_size(
    run: InlineTextRun,
    glyphs: Sequence[Any],
    policy: TypesettingPolicy,
    font_size: int,
    style: Mapping[str, Any] | None = None,
) -> tuple[float, float, float]:
    if run.role in {"ellipsis_sequence", "dash_sequence", "wave_sequence", "punctuation_sequence"}:
        count = max(1, len(grapheme_clusters(run.text)))
        width = float(font_size)
        height = float(font_size)
        if _is_vertical_emphasis_sequence(run):
            return max(1.0, width), max(1.0, height), width
        if run.role in {"ellipsis_sequence", "dash_sequence", "wave_sequence"}:
            row_units = _vertical_sequence_row_units(run)
            width = _vertical_primitive_cross_axis_extent(
                run,
                font_size,
                style,
            )
            height = float(font_size) * float(row_units)
        elif count > 1:
            height = max(height, min(float(font_size) * 1.15, float(font_size) * (0.58 * count)))
        return max(1.0, width), max(1.0, height), width
    if not glyphs:
        count = max(1, len(grapheme_clusters(run.text)))
        return float(max(font_size, count)), float(font_size), 0.0
    if run.script == "Latn" or run.role in {"numeric_token", "complex_script", "symbol"}:
        width = sum(max(0.0, abs(float(glyph.x_advance))) for glyph in glyphs)
        height = max(float(font_size), max((abs(float(glyph.y_advance)) for glyph in glyphs), default=0.0))
        if height <= 0.0:
            height = max((abs(float(glyph.x_advance)) for glyph in glyphs), default=1.0)
        return max(1.0, width), max(1.0, height), width
    width = max((abs(float(glyph.x_advance)) for glyph in glyphs), default=1.0)
    height = max((abs(float(glyph.y_advance)) for glyph in glyphs), default=1.0)
    if width <= 0.0:
        width = height if height > 0.0 else float(font_size)
    if height <= 0.0:
        height = float(font_size)
    return max(1.0, width), max(1.0, height), width


def _vertical_primitive_cross_axis_extent(
    run: InlineTextRun,
    font_size: int,
    style: Mapping[str, Any] | None,
) -> float:
    try:
        stroke_width = max(0.0, float((style or {}).get("stroke_width") or 0.0))
    except (TypeError, ValueError):
        stroke_width = 0.0
    raster_guard = 1.0
    size = max(1.0, float(font_size))
    if run.role == "ellipsis_sequence":
        diameter = max(2.0, float(round(size * 0.12)))
        return diameter + 2.0 * stroke_width + 2.0 * raster_guard
    if run.role == "dash_sequence":
        line_width = max(2.0, float(round(size * 0.09)))
        return line_width + 2.0 * stroke_width + 2.0 * raster_guard
    amplitude = max(2.0, size * 0.14)
    line_width = max(2.0, float(round(size * 0.08)))
    return 2.0 * amplitude + line_width + 2.0 * stroke_width + 2.0 * raster_guard


def _finalize_drawing_primitives(
    placements: Sequence[GlyphPlacement],
    style: Mapping[str, Any] | None,
) -> tuple[list[GlyphPlacement], list[DrawingPrimitive]]:
    finalized: list[GlyphPlacement] = []
    primitives: list[DrawingPrimitive] = []
    for index, placement in enumerate(placements):
        metadata = dict(placement.metadata or {})
        mode = str(metadata.get("placement_mode") or "")
        if mode not in {
            "vertical_ellipsis_sequence",
            "vertical_dash_sequence",
            "vertical_wave_sequence",
        }:
            finalized.append(placement)
            continue
        primitive_id = (
            f"drawing_primitive_{index:04d}_"
            f"{str(metadata.get('run_id') or 'run')}"
        )
        primitive = _drawing_primitive_for_placement(
            primitive_id=primitive_id,
            placement=placement,
            style=style,
        )
        metadata["drawing_primitive_id"] = primitive_id
        metadata["primitive_geometry_final"] = True
        metadata["primitive_bounds"] = list(primitive.bounds)
        finalized.append(replace(placement, metadata=metadata))
        primitives.append(primitive)
    return finalized, primitives


def _drawing_primitive_for_placement(
    *,
    primitive_id: str,
    placement: GlyphPlacement,
    style: Mapping[str, Any] | None,
) -> DrawingPrimitive:
    metadata = dict(placement.metadata or {})
    mode = str(metadata.get("placement_mode") or "")
    box = bbox_from_value(placement.bbox)
    if not box:
        raise ValueError("drawing_primitive_missing_placement_bounds")
    x, y, width, height = box
    font_size = max(1.0, float(placement.font_size or 1.0))
    try:
        outline_width = max(
            0.0,
            float((style or {}).get("stroke_width") or 0.0),
        )
    except (TypeError, ValueError):
        outline_width = 0.0
    token_ids = [str(value) for value in list(metadata.get("token_ids") or [])]
    occurrences = [
        dict(value)
        for value in list(metadata.get("punctuation_occurrences") or [])
        if isinstance(value, Mapping)
    ]
    centers: list[list[float]] = []
    points: list[list[float]] = []
    diameter = 0.0
    line_width = 0.0
    pitch = 0.0
    unit_count = 1
    visible_count = 1
    sequence_group_count = 1
    primitive_metadata: dict[str, Any] = {
        "geometry_owner": "TypesettingEngine",
        "geometry_status": "final",
        "placement_mode": mode,
        "font_size": float(font_size),
        "outline_width_px": float(outline_width),
        "punctuation_occurrences": occurrences,
        "relative_geometry_recomputation_allowed": False,
    }
    source_punctuation_status = str(
        metadata.get("source_punctuation_footprint_status") or ""
    )
    if source_punctuation_status:
        primitive_metadata.update(
            {
                "source_punctuation_footprint_status": (
                    source_punctuation_status
                ),
                "source_punctuation_footprint_box": list(
                    metadata.get("source_punctuation_footprint_box") or []
                ),
                "source_punctuation_footprint_fact_set_id": str(
                    metadata.get("source_punctuation_footprint_fact_set_id")
                    or ""
                ),
            }
        )

    if mode == "vertical_ellipsis_sequence":
        unit_count = max(1, int(metadata.get("ellipsis_unit_count") or 1))
        visible_count = max(
            1,
            int(metadata.get("ellipsis_dot_count") or unit_count * 3),
        )
        sequence_group_count = max(
            1,
            int(metadata.get("ellipsis_sequence_group_count") or 1),
        )
        desired_diameter = max(2.0, float(round(font_size * 0.12)))
        maximum_diameter = max(1.0, float(width) - 2.0 * outline_width)
        diameter = min(desired_diameter, maximum_diameter)
        radius = diameter / 2.0
        outer_radius = radius + outline_width
        center_x = float(x) + (float(width) - 1.0) / 2.0
        requested_inset = float(height) * (0.25 / float(unit_count))
        edge_inset = min(
            max(outer_radius, requested_inset),
            max(outer_radius, float(height) / 2.0),
        )
        first_y = float(y) + edge_inset
        last_y = float(y + height) - edge_inset
        if visible_count > 1:
            pitch = max(0.0, (last_y - first_y) / float(visible_count - 1))
        for dot_index in range(visible_count):
            center_y = (
                first_y + pitch * float(dot_index)
                if visible_count > 1
                else float(y) + float(height) / 2.0
            )
            centers.append([round(center_x, 4), round(center_y, 4)])
        primitive_metadata.update(
            {
                "ellipsis_policy": "one_continuous_uniform_dot_sequence",
                "dot_column_count": 1,
                "continuous_sequence": True,
            }
        )
    elif mode == "vertical_dash_sequence":
        unit_count = max(1, int(metadata.get("dash_unit_count") or 1))
        line_width = max(2.0, float(round(font_size * 0.09)))
        center_x = float(x) + (float(width) - 1.0) / 2.0
        endpoint_radius = line_width / 2.0 + outline_width
        pad_y = max(
            1.0,
            float(round(font_size * 0.04)),
            endpoint_radius,
        )
        first_y = min(float(y + height - 1), float(y) + pad_y)
        last_y = max(float(y), float(y + height - 1) - pad_y)
        points = [
            [round(center_x, 4), round(first_y, 4)],
            [round(center_x, 4), round(last_y, 4)],
        ]
        pitch = max(0.0, last_y - first_y)
        primitive_metadata.update(
            {
                "continuous_segment_count": 1,
                "continuous_multi_cell_dash": True,
                "endpoint_effect_inset_px": float(pad_y),
            }
        )
    else:
        unit_count = max(1, int(metadata.get("wave_unit_count") or 1))
        line_width = max(2.0, float(round(font_size * 0.08)))
        amplitude = max(
            1.0,
            min(float(width) * 0.24, font_size * 0.14),
        )
        endpoint_radius = line_width / 2.0 + outline_width
        pad_y = max(
            1.0,
            float(round(font_size * 0.04)),
            endpoint_radius,
        )
        first_y = min(float(y + height - 1), float(y) + pad_y)
        last_y = max(float(y), float(y + height - 1) - pad_y)
        span = max(1.0, last_y - first_y)
        center_x = float(x) + (float(width) - 1.0) / 2.0
        sample_step = max(1, int(math.ceil(span / 128.0)))
        sample_ys = list(range(int(round(first_y)), int(round(last_y)) + 1, sample_step))
        if not sample_ys or sample_ys[-1] != int(round(last_y)):
            sample_ys.append(int(round(last_y)))
        for sample_y in sample_ys:
            t = max(0.0, min(1.0, (float(sample_y) - first_y) / span))
            sample_x = center_x + math.sin(t * math.tau * float(unit_count)) * amplitude
            points.append([round(sample_x, 4), round(float(sample_y), 4)])
        pitch = span / float(unit_count)
        visible_count = unit_count
        primitive_metadata.update(
            {
                "wave_cycle_count": float(unit_count),
                "path_sample_step_px": int(sample_step),
                "continuous_multi_cell_wave": True,
                "endpoint_effect_inset_px": float(pad_y),
            }
        )

    return DrawingPrimitive(
        primitive_id=primitive_id,
        kind=mode,
        source_text=str(placement.text or ""),
        token_ids=token_ids,
        orientation="vertical",
        bounds=list(box),
        centers=centers,
        points=points,
        diameter_px=float(diameter),
        line_width_px=float(line_width),
        pitch_px=float(pitch),
        unit_count=int(unit_count),
        visible_count=int(visible_count),
        sequence_group_count=int(sequence_group_count),
        metadata=primitive_metadata,
    )


def _vertical_run_is_atomic(run: InlineTextRun, policy: TypesettingPolicy) -> bool:
    if run.script == "Latn":
        return True
    if run.role in {"numeric_token", "complex_script", "symbol", "ellipsis_sequence", "dash_sequence", "wave_sequence", "punctuation_sequence"}:
        return True
    return False


def _is_vertical_emphasis_sequence(run: InlineTextRun) -> bool:
    return bool(
        run.role == "punctuation_sequence"
        and _occurrence_unit_count(run, "emphasis_punctuation") >= 3
    )


def _occurrence_unit_count(run: InlineTextRun, kind: str) -> int:
    occurrences = list(run.metadata.get("punctuation_occurrences") or [])
    count = sum(
        int(item.get("unit_count") or 0)
        for item in occurrences
        if isinstance(item, Mapping) and str(item.get("kind") or "") == kind
    )
    if count > 0:
        return count
    if (
        (kind == "ellipsis" and run.role == "ellipsis_sequence")
        or (kind == "wave" and run.role == "wave_sequence")
        or (kind == "dash" and run.role == "dash_sequence")
    ):
        return max(1, len(grapheme_clusters(run.text)))
    return 0


def _occurrence_dot_count(run: InlineTextRun, kind: str) -> int:
    occurrences = [
        item
        for item in list(run.metadata.get("punctuation_occurrences") or [])
        if isinstance(item, Mapping) and str(item.get("kind") or "") == kind
    ]
    count = sum(int(item.get("dot_count") or 0) for item in occurrences)
    if count > 0:
        return count
    units = _occurrence_unit_count(run, kind)
    return units * 3 if kind == "ellipsis" and units > 0 else 0


def _occurrence_sequence_group_count(run: InlineTextRun, kind: str) -> int:
    occurrences = [
        item
        for item in list(run.metadata.get("punctuation_occurrences") or [])
        if isinstance(item, Mapping) and str(item.get("kind") or "") == kind
    ]
    if not occurrences:
        return 0
    return sum(max(1, int(item.get("sequence_group_count") or 1)) for item in occurrences)


def _dominant_vertical_advance(shaped_runs: Sequence[ShapedRun]) -> float:
    advances: list[float] = []
    for run in shaped_runs:
        for glyph in run.glyphs:
            if abs(float(glyph.y_advance)) > 0.0:
                advances.append(abs(float(glyph.y_advance)))
    if not advances:
        return 1.0
    advances.sort()
    return max(1.0, advances[len(advances) // 2])


def _logical_run_id_for(run: InlineTextRun) -> str:
    return str(run.metadata.get("logical_run_id") or run.run_id)


def _placement_font_span_metadata(run: InlineTextRun) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "token_ids": list(run.token_ids),
        "original_text": run.original_text,
        "translated_start": int(run.translated_start),
        "translated_end": int(run.translated_end),
        "atomic_break": bool(run.metadata.get("atomic_break")),
    }
    span_id = str(run.metadata.get("font_span_id") or "")
    if not span_id:
        return metadata
    metadata.update(
        {
            "logical_run_id": _logical_run_id_for(run),
            "font_span_id": span_id,
        }
    )
    return metadata


def _visible_notdef_run_ids(shaped_runs: Sequence[ShapedRun]) -> list[str]:
    run_ids: list[str] = []
    for shaped in shaped_runs:
        if not shaped.glyphs and source_text_requires_visible_glyph(shaped.normalized_text):
            run_ids.append(str(shaped.metadata.get("run_id") or ""))
            continue
        for glyph in shaped.glyphs:
            if int(glyph.glyph_id) != 0:
                continue
            if not any(
                source_char_requires_visible_glyph(char)
                for char in str(glyph.text or "")
            ):
                continue
            run_ids.append(str(shaped.metadata.get("run_id") or ""))
            break
    return _unique(run_ids)


def _shaped_source_cluster_map(
    run: InlineTextRun,
    shaped: ShapedRun,
) -> list[dict[str, Any]]:
    clusters = grapheme_clusters(run.normalized_text)
    codepoint_offsets = [0]
    for cluster in clusters:
        codepoint_offsets.append(codepoint_offsets[-1] + len(cluster))
    hb_clusters = sorted(
        {
            int(glyph.cluster)
            for glyph in shaped.glyphs
            if 0 <= int(glyph.cluster) <= len(run.normalized_text)
        }
    )
    hb_ranges: list[tuple[int, int]] = []
    for index, start in enumerate(hb_clusters):
        end = (
            hb_clusters[index + 1]
            if index + 1 < len(hb_clusters)
            else len(run.normalized_text)
        )
        hb_ranges.append((start, max(start, end)))
    output: list[dict[str, Any]] = []
    for grapheme_index, cluster in enumerate(clusters):
        codepoint_start = codepoint_offsets[grapheme_index]
        codepoint_end = codepoint_offsets[grapheme_index + 1]
        anchor = hb_clusters[0] if hb_clusters else None
        for hb_start, hb_end in hb_ranges:
            if hb_start <= codepoint_start < max(hb_start + 1, hb_end):
                anchor = hb_start
                break
            if hb_start <= codepoint_start:
                anchor = hb_start
        glyph_indices = (
            [
                index
                for index, glyph in enumerate(shaped.glyphs)
                if int(glyph.cluster) == int(anchor)
            ]
            if anchor is not None
            else []
        )
        visible_glyph_required = any(
            source_char_requires_visible_glyph(char) for char in cluster
        )
        output.append(
            {
                "logical_run_id": _logical_run_id_for(run),
                "font_span_id": str(run.metadata.get("font_span_id") or run.run_id),
                "span_grapheme_index": grapheme_index,
                "source_grapheme_start": run.grapheme_start + grapheme_index,
                "source_grapheme_end": run.grapheme_start + grapheme_index + 1,
                "span_codepoint_start": codepoint_start,
                "span_codepoint_end": codepoint_end,
                "text": cluster,
                "harfbuzz_cluster": int(anchor) if anchor is not None else None,
                "shaped_glyph_indices": glyph_indices,
                "visible_glyph_required": visible_glyph_required,
                "no_ink_default_ignorable": not visible_glyph_required,
            }
        )
    return output


def _font_span_cluster_ledger(
    runs: Sequence[InlineTextRun],
    shaped_runs: Sequence[ShapedRun],
) -> list[dict[str, Any]]:
    shaped_by_run = {
        str(item.metadata.get("run_id") or ""): item
        for item in shaped_runs
        if str(item.metadata.get("run_id") or "")
    }
    output: list[dict[str, Any]] = []
    for run in runs:
        shaped = shaped_by_run.get(run.run_id)
        if shaped is not None:
            output.extend(_shaped_source_cluster_map(run, shaped))
            continue
        if run.role != "space":
            continue
        for index, cluster in enumerate(grapheme_clusters(run.normalized_text)):
            output.append(
                {
                    "logical_run_id": _logical_run_id_for(run),
                    "font_span_id": str(run.metadata.get("font_span_id") or run.run_id),
                    "span_grapheme_index": index,
                    "source_grapheme_start": run.grapheme_start + index,
                    "source_grapheme_end": run.grapheme_start + index + 1,
                    "text": cluster,
                    "harfbuzz_cluster": None,
                    "shaped_glyph_indices": [],
                    "visible_glyph_required": False,
                    "no_ink_space": True,
                }
            )
    return output


def _font_span_token_provenance(
    run: InlineTextRun,
    span: FontSpanResolution,
) -> dict[str, Any]:
    raw_tokens = list(run.metadata.get("lossless_tokens") or [])
    span_start = int(run.grapheme_start) + int(span.source_grapheme_start)
    span_end = int(run.grapheme_start) + int(span.source_grapheme_end)
    selected: list[tuple[int, Mapping[str, Any]]] = []
    for index, raw_token in enumerate(raw_tokens):
        if not isinstance(raw_token, Mapping):
            continue
        token_start = int(raw_token.get("presentation_grapheme_start") or 0)
        token_end = int(
            raw_token.get("presentation_grapheme_end") or token_start
        )
        if token_start < span_end and token_end > span_start:
            selected.append((index, raw_token))
    if not selected:
        raise RuntimeError("font_span_token_provenance_missing")

    token_records = [dict(item) for _, item in selected]
    presentation_text = "".join(
        str(item.get("presentation_text") or "") for item in token_records
    )
    if presentation_text != str(span.text or ""):
        raise RuntimeError("font_span_token_provenance_not_conserved")

    token_ids = tuple(
        str(item.get("token_id") or "") for item in token_records
    )
    if not all(token_ids):
        raise RuntimeError("font_span_token_provenance_missing")
    original_text = "".join(
        str(item.get("original_text") or "") for item in token_records
    )
    translated_start = min(
        int(item.get("translated_start") or 0) for item in token_records
    )
    translated_end = max(
        int(item.get("translated_end") or 0) for item in token_records
    )
    first_token_index = selected[0][0]
    last_token_index = selected[-1][0]
    token_start = int(run.token_start) + first_token_index
    token_end = int(run.token_start) + last_token_index + 1
    return {
        "original_text": original_text,
        "translated_start": translated_start,
        "translated_end": translated_end,
        "token_start": token_start,
        "token_end": token_end,
        "token_ids": token_ids,
        "metadata": {
            "token_ids": list(token_ids),
            "lossless_tokens": token_records,
            "original_text": original_text,
            "translated_start": translated_start,
            "translated_end": translated_end,
        },
    }


def _unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _unique_ints(values: Sequence[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
