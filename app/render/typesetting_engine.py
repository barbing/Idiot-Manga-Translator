# -*- coding: utf-8 -*-
"""Pure Stage 4 typesetting engine.

The engine consumes RenderLayerPlan records and emits TypesetLayout/FitReport
records. It does not draw final text, mutate cleanup, or reinterpret parent
identity.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_CEILING
from typing import Any, Mapping, Sequence

from app.render.font_manager import (
    FontManager,
    FontResolution,
    FontSpanResolution,
    RunFontResolution,
)
from app.render.line_break_planner import (
    LineBreakPlanner,
    canonical_break_quality_key,
    canonical_break_quality_summary,
)
from app.render.parent_layer_effects import (
    fit_effect_envelope,
    outward_int_xywh,
    resolve_parent_layer_effects,
    shift_layout_geometry,
)
from app.render.source_punctuation_hints import (
    SOURCE_PUNCTUATION_MEASUREMENT_BASIS_ABSOLUTE_STROKE,
)
from app.render.text_shaper import HarfBuzzShaper, ShapedRun
from app.render.target_lexical_segmenter import (
    TargetLexicalSegmenter,
    default_target_lexical_segmenter,
)
from app.render.typesetting_contracts import (
    DrawingPrimitive,
    FitReport,
    GlyphPlacement,
    RenderLayerPlan,
    TypesetLayout,
    bbox_from_value,
    validated_source_punctuation_geometry_ref,
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

BASE_TEXT_INK_ENVELOPE_VERSION = "base_text_ink_envelope_v1"
BASE_TEXT_HINTED_DIMENSION_GUARD_PX = 1.0


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
        lexical_segmenter: TargetLexicalSegmenter | None = None,
    ) -> None:
        self.font_manager = font_manager or FontManager()
        self.shaper = shaper or HarfBuzzShaper(self.font_manager)
        self.policy = policy or TypesettingPolicy()
        self.break_planner = break_planner or LineBreakPlanner()
        self.lexical_segmenter = (
            lexical_segmenter or default_target_lexical_segmenter()
        )

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
        target_lexical_segmentation = self.lexical_segmenter.segment(
            str(plan.translated_text or ""),
            identity_tokens,
        )
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
        breaks = compute_break_opportunities(
            runs,
            writing_mode=writing_mode,
            language_hint=_language_hint(plan),
            target_lexical_spans=(
                target_lexical_segmentation.spans
                if target_lexical_segmentation.available
                else None
            ),
            target_lexical_boundaries=(
                target_lexical_segmentation.boundaries
                if target_lexical_segmentation.available
                else None
            ),
        )
        runs_by_id = {run.run_id: run for run in runs}
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
            if (
                runs_by_id.get(item.before_run_id) is not None
                and runs_by_id.get(item.after_run_id) is not None
                and _logical_run_id_for(runs_by_id[item.before_run_id])
                == _logical_run_id_for(runs_by_id[item.after_run_id])
            )
            else item
            for item in breaks
        ]
        style_issues = _unique(
            [
                *_style_issues(runs, writing_mode, self.policy),
                *target_lexical_segmentation.issues,
            ]
        )
        script_policy = _script_policy(runs, writing_mode, self.policy)
        attempts: list[dict[str, Any]] = []
        selected_attempt: dict[str, Any] | None = None
        source_preferred_interval_floor = _source_preferred_interval_floor(
            preferred_font_size,
            plan.resolved_render_style,
        )
        lexical_candidate_selection_reason = ""
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
            break_quality_retry = None
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
                if (
                    fit_status == "fits"
                    and _vertical_break_quality_probe_is_relevant(break_plan)
                ):
                    break_quality_retry = (
                        self._retry_vertical_break_quality_within_hard_bounds(
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
                    )
                if break_quality_retry is not None:
                    placements = break_quality_retry["placements"]
                    lines = break_quality_retry["lines"]
                    columns = break_quality_retry["columns"]
                    measured_bounds = break_quality_retry["measured_bounds"]
                    fit_status = break_quality_retry["fit_status"]
                    fit_issues = break_quality_retry["fit_issues"]
                    break_plan = break_quality_retry["break_plan"]
                    layout_intent_box = break_quality_retry["layout_intent_box"]
                    box_model = break_quality_retry["box_model"]
                    reason_codes.extend(break_quality_retry["reason_codes"])
                if columns and isinstance(
                    columns[0].get("layout_profile"),
                    Mapping,
                ):
                    box_model = dict(box_model)
                    box_model["vertical_layout_profile"] = dict(
                        columns[0]["layout_profile"]
                    )
            base_ink_fit = _fit_base_text_ink_envelope(
                placements=placements,
                lines=lines,
                columns=columns,
                shaped_runs=shaped_runs,
                hard_bounds=hard_bounds,
                style=plan.resolved_render_style,
                layout_fit_status=fit_status,
            )
            placements = list(base_ink_fit["placements"])
            lines = list(base_ink_fit["lines"])
            columns = list(base_ink_fit["columns"])
            measured_bounds = list(base_ink_fit["measured_bounds"])
            base_text_ink_envelope = dict(base_ink_fit["audit"])
            if fit_status == "fits" and not bool(
                base_text_ink_envelope.get("contained")
            ):
                fit_status = "overflow"
                fit_issues = _unique(
                    [
                        *fit_issues,
                        "base_text_ink_envelope_exceeds_hard_bounds",
                    ]
                )
            elif base_text_ink_envelope.get("status") == "translated":
                reason_codes.append(
                    "base_text_ink_envelope_translated_within_hard_bounds"
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
                    fit_issues = _unique(
                        [
                            *fit_issues,
                            *parent_effect_envelope.issues,
                            "parent_layer_effect_envelope_exceeds_hard_bounds",
                            "optional_parent_effect_requires_base_text_fallback",
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
                "base_text_ink_envelope": base_text_ink_envelope,
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
                "same_size_break_quality_layout_expansion_used": bool(
                    break_quality_retry is not None
                ),
                "selected_confirmed_lexical_break_count": (
                    _selected_lexical_break_count(
                        break_plan,
                        "confirmed_lexical_break_rank",
                    )
                ),
                "selected_weak_lexical_break_count": (
                    _selected_lexical_break_count(
                        break_plan,
                        "weak_lexical_break_rank",
                    )
                ),
                "canonical_break_quality": canonical_break_quality_summary(
                    break_plan
                ),
            }
            attempts.append(attempt)
            selected_attempt = attempt
            if fit_status != "fits":
                continue
            lexical_candidate_selection_reason = (
                "same_size_authorized_layout_expansion"
                if bool(
                    attempt["same_size_break_quality_layout_expansion_used"]
                )
                else "first_technical_fit"
            )
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
        base_text_ink_envelope = dict(selected_attempt["base_text_ink_envelope"])
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
                for item in attempts
                if item is not selected_attempt
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
        placement_text = "".join(item.text for item in placements)
        text_placement_complete = bool(
            token_text_conserved
            and font_span_text_conserved
            and token_conservation["visible_tokens_placed"]
            and placement_text == normalized
        )
        hard_bounds_contained = bool(
            base_measured_bounds
            and _box_inside(base_measured_bounds, hard_bounds)
        )
        final_scale = float(font_size) / float(max(1, preferred_font_size))
        candidate_rank = next(
            (
                index
                for index, item in enumerate(attempts)
                if item is selected_attempt
            ),
            max(0, len(attempts) - 1),
        )
        break_count = len(list(break_plan.get("selected_breaks") or []))
        if final_scale < 0.5 or fit_status != "fits":
            fit_quality = "severe_scale_risk"
        elif font_size < preferred_font_size:
            fit_quality = "scaled"
        elif break_count:
            fit_quality = "reflowed"
        else:
            fit_quality = "preferred"
        fit_trigger = (
            "preferred_size_break_quality_layout_expansion"
            if (
                font_size == preferred_font_size
                and fit_status == "fits"
                and bool(
                    selected_attempt.get(
                        "same_size_break_quality_layout_expansion_used"
                    )
                )
            )
            else "lexical_integrity_within_source_preferred_interval"
            if (
                font_size < preferred_font_size
                and fit_status == "fits"
                and lexical_candidate_selection_reason
                == "source_preferred_interval_size_adjustment"
            )
            else "preferred_layout"
            if candidate_rank == 0 and fit_status == "fits"
            else "layout_fit_pressure"
            if fit_status == "fits"
            else "complete_layout_scale_to_box_fallback"
        )
        full_text_placed = text_placement_complete
        typesetting_candidate_quality = _typesetting_candidate_quality_summary(
            preferred_font_size=preferred_font_size,
            selected_font_size=font_size,
            fit_status=fit_status,
            text_placement_complete=text_placement_complete,
            hard_bounds_contained=hard_bounds_contained,
            break_plan=break_plan,
        )
        if not text_placement_complete:
            all_issues = _unique(
                [*all_issues, "required_text_placement_incomplete"]
            )
        if not hard_bounds_contained:
            all_issues = _unique(
                [*all_issues, "base_layout_hard_bounds_not_contained"]
            )
        if fit_quality == "severe_scale_risk":
            all_issues = _unique([*all_issues, "severe_scale_readability_risk"])
        metadata = {
            "typesetting_engine_version": "typesetting_engine_stage4_v1",
            "lossless_text_tokens": token_audit,
            "text_token_conservation": token_conservation,
            "box_model": box_model,
            "writing_mode_policy": writing_policy,
            "target_lexical_segmentation": (
                target_lexical_segmentation.to_audit_dict()
            ),
            "inline_runs": [_run_audit(run, writing_mode, self.policy) for run in runs],
            "break_opportunities": [item.to_audit_dict() for item in breaks],
            "chosen_breaks": list(break_plan.get("selected_breaks") or []),
            "break_plan": break_plan,
            "typesetting_candidate_quality": typesetting_candidate_quality,
            "kinsoku_adjustments": kinsoku_adjustments,
            "line_break_policy": {
                "policy_version": "line_break_policy_stage4_target_lexical_v2",
                "locale_hint": _language_hint(plan),
                "writing_mode": writing_mode,
                "selection_authority": "explicit_break_opportunities",
                "planner_version": self.break_planner.version,
                "accepted_tailoring_rules": [
                    "space_word_boundary",
                    "target_lexical_confirmed_keep_boundary",
                    "target_lexical_weak_keep_boundary",
                    "target_lexical_conflict_abstention",
                    "cjk_grapheme_boundary_degraded_fallback",
                ],
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
            "base_text_ink_envelope": base_text_ink_envelope,
            "drawing_primitives": [
                item.to_audit_dict() for item in drawing_primitives
            ],
            "drawing_primitive_geometry_final": True,
            "open_type_metrics_by_face": open_type_metrics_by_face,
            "font_size_selection": {
                "preferred_font_size": int(preferred_font_size),
                "selected_pre_scale_font_size": int(font_size),
                "selected_font_size": int(font_size),
                "final_effective_font_size": float(font_size),
                "final_scale": round(final_scale, 6),
                "fit_trigger": fit_trigger,
                "candidate_rank": int(candidate_rank),
                "fit_quality": fit_quality,
                "text_placement_complete": text_placement_complete,
                "hard_bounds_contained": hard_bounds_contained,
                "scaling_used": round(final_scale, 4),
                "fallback_used": font_size != preferred_font_size,
                "candidate_count": len(attempts),
                "lexical_candidate_selection": {
                    "status": "selected",
                    "selection_reason": lexical_candidate_selection_reason,
                    "segmenter_available": bool(
                        target_lexical_segmentation.available
                    ),
                    "comparison_triggered": bool(
                        len(attempts) > 1
                        or any(
                            bool(
                                item.get(
                                    "same_size_break_quality_layout_expansion_used"
                                )
                            )
                            for item in attempts
                        )
                    ),
                    "source_preferred_interval_px": list(
                        plan.resolved_render_style.get(
                            "target_preferred_em_interval_px"
                        )
                        or []
                    ),
                    "rounded_source_preferred_interval_floor_px": (
                        source_preferred_interval_floor
                    ),
                    "source_preferred_interval_is_render_admission": False,
                    "technical_fit_continues_below_interval": True,
                    "second_break_selector_added": False,
                    "selected_confirmed_lexical_break_count": int(
                        selected_attempt.get(
                            "selected_confirmed_lexical_break_count"
                        )
                        or 0
                    ),
                    "selected_weak_lexical_break_count": int(
                        selected_attempt.get(
                            "selected_weak_lexical_break_count"
                        )
                        or 0
                    ),
                    "same_size_authorized_expansion_used": bool(
                        selected_attempt.get(
                            "same_size_break_quality_layout_expansion_used"
                        )
                    ),
                },
                "attempts": [
                    {
                        "selected": item is selected_attempt,
                        "font_size": int(item["font_size"]),
                        "fit_status": str(item["fit_status"]),
                        "layout_intent_box": list(item["layout_intent_box"]),
                        "measured_bounds": list(item["measured_bounds"]),
                        "issues": list(item["issues"]),
                        "selected_confirmed_lexical_break_count": int(
                            item.get("selected_confirmed_lexical_break_count")
                            or 0
                        ),
                        "selected_weak_lexical_break_count": int(
                            item.get("selected_weak_lexical_break_count")
                            or 0.0
                        ),
                        "canonical_break_quality": dict(
                            item.get("canonical_break_quality") or {}
                        ),
                        "same_size_break_quality_layout_expansion_used": bool(
                            item.get(
                                "same_size_break_quality_layout_expansion_used"
                            )
                        ),
                        "parent_layer_effect_envelope": dict(
                            item["parent_layer_effect_envelope"]
                        ),
                        "base_text_ink_envelope": dict(
                            item["base_text_ink_envelope"]
                        ),
                    }
                    for item in attempts
                ],
            },
            "render_style": {
                "line_height": plan.resolved_render_style.get("line_height"),
                "effective_line_height": round(effective_line_height, 4),
                "align": plan.resolved_render_style.get("align"),
                "fill_color": _style_fill_color(plan.resolved_render_style),
                "stroke_color": _style_outline_color(plan.resolved_render_style),
                "stroke_width": _style_outline_width(plan.resolved_render_style),
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
            text_placement_complete=text_placement_complete,
            hard_bounds_contained=hard_bounds_contained,
            fit_quality=fit_quality,
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
            text_placement_complete=text_placement_complete,
            hard_bounds_contained=hard_bounds_contained,
            fit_quality=fit_quality,
            preferred_font_size=float(preferred_font_size),
            selected_pre_scale_font_size=float(font_size),
            final_effective_font_size=float(font_size),
            final_scale=final_scale,
            fit_trigger=fit_trigger,
            candidate_rank=candidate_rank,
            natural_fit_success=(
                candidate_rank == 0
                and fit_status == "fits"
                and not bool(
                    selected_attempt.get(
                        "same_size_break_quality_layout_expansion_used"
                    )
                )
            ),
            fallback_used=font_size != preferred_font_size,
            scaling_used=final_scale,
            overflow_risk=not hard_bounds_contained,
            clipping_risk=not hard_bounds_contained,
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
                "target_lexical_segmentation": metadata[
                    "target_lexical_segmentation"
                ],
                "line_break_policy": metadata["line_break_policy"],
                "break_opportunities": metadata["break_opportunities"],
                "chosen_breaks": metadata["chosen_breaks"],
                "break_plan": break_plan,
                "typesetting_candidate_quality": typesetting_candidate_quality,
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
                "base_text_ink_envelope": base_text_ink_envelope,
                "open_type_metrics_by_face": open_type_metrics_by_face,
                "box_model": box_model,
                "fit_contract": {
                    "text_placement_complete": text_placement_complete,
                    "hard_bounds_contained": hard_bounds_contained,
                    "fit_quality": fit_quality,
                    "preferred_font_size": float(preferred_font_size),
                    "selected_pre_scale_font_size": float(font_size),
                    "final_effective_font_size": float(font_size),
                    "final_scale": round(final_scale, 6),
                    "fit_trigger": fit_trigger,
                    "candidate_rank": int(candidate_rank),
                    "readability_is_render_admission": False,
                },
            },
        )
        return layout, report

    def candidate_quality_summary(
        self,
        plan: RenderLayerPlan,
        layout: TypesetLayout,
        report: FitReport,
    ) -> dict[str, Any]:
        """Expose opaque typography quality to the mechanical slot owner."""

        existing = (
            layout.metadata.get("typesetting_candidate_quality")
            if isinstance(layout.metadata, Mapping)
            else None
        )
        if isinstance(existing, Mapping) and existing.get("sort_key"):
            return deepcopy(dict(existing))
        return _typesetting_candidate_quality_summary(
            preferred_font_size=_font_size_from_style(
                dict(plan.resolved_render_style or {})
            ),
            selected_font_size=float(layout.selected_font_size or 0.0),
            fit_status=str(report.fit_status or layout.fit_status or ""),
            text_placement_complete=bool(
                report.full_text_placed and layout.text_placement_complete
            ),
            hard_bounds_contained=bool(
                report.hard_bounds_contained and layout.hard_bounds_contained
            ),
            break_plan=(
                layout.metadata.get("break_plan")
                if isinstance(layout.metadata, Mapping)
                else {}
            ),
        )

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
                            **{
                                str(key): value
                                for key, value in item.items()
                                if str(key).startswith(
                                    "source_punctuation_geometry_"
                                )
                            },
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
        alignment_bounds = [x, y, w, h]
        alignment_center, alignment_source = _layout_alignment_center(
            plan,
            alignment_bounds,
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
        # ``_layout_vertical`` exposes capacity to the break planner as a
        # whole-row count (``floor(box_height / cell_height)``).  Punctuation
        # primitives may consume fractional row units, so sizing a retry box
        # from the raw fractional total can reopen a 6.145-row partition with
        # only six planner rows and force a different, lexically worse split.
        # Round the selected partition up to the same whole-row capacity
        # contract before converting it back to pixels.
        required_row_capacity = max(
            1,
            int(math.ceil(max_row_units - 1e-9)),
        )
        hard_w = int(hard[2])
        hard_h = int(hard[3])

        cell_height = max(shaped_advance * effective_line_height, item_cell_height)
        needed_height = int(math.ceil(required_row_capacity * cell_height))
        reason_codes = ["layout_intent_expanded_for_legal_vertical_partition"]
        if (
            needed_height > hard_h
            and original_line_height > 1.0
            and _font_size_is_locked(style)
        ):
            compacted = min(
                original_line_height,
                float(hard_h)
                / max(1.0, required_row_capacity * shaped_advance),
            )
            if compacted >= 1.0:
                effective_line_height = max(1.0, compacted - 1e-6)
                cell_height = max(shaped_advance * effective_line_height, item_cell_height)
                needed_height = int(
                    math.ceil(required_row_capacity * cell_height)
                )
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

    def _retry_vertical_break_quality_within_hard_bounds(
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
        """Try a minimally larger intent for a strictly better target plan.

        This remains orchestration around the sole ``LineBreakPlanner``.  The
        full hard box is used only to compare the planner's canonical quality
        record at the same font size. Source-footprint evidence remains the
        final advisory field in that record. A better executable partition is then
        reduced to the smallest centered expansion that can hold it; the
        upstream hard box never changes.
        """

        initial_quality = canonical_break_quality_key(break_plan)
        if not _vertical_break_quality_probe_is_relevant(break_plan):
            return None
        hard = bbox_from_value(hard_bounds)
        current = bbox_from_value(layout_intent_box)
        if not hard or not current or hard == current:
            return None
        (
            _probe_placements,
            _probe_lines,
            _probe_columns,
            _probe_measured_bounds,
            probe_fit_status,
            _probe_fit_issues,
            probe_break_plan,
        ) = self._layout_vertical(
            normalized,
            runs,
            shaped_runs,
            breaks,
            hard,
            font_size,
            dict(style or {}),
            plan,
            candidate_capacity_box=hard,
        )
        probe_quality = canonical_break_quality_key(probe_break_plan)
        if probe_fit_status != "fits" or not probe_quality < initial_quality:
            return None

        retry = self._retry_vertical_fit_within_hard_bounds(
            normalized=normalized,
            runs=runs,
            shaped_runs=shaped_runs,
            breaks=breaks,
            plan=plan,
            style=style,
            font_size=font_size,
            hard_bounds=hard,
            layout_intent_box=current,
            box_model=box_model,
            break_plan=probe_break_plan,
        )
        if retry is None:
            return None
        selected_quality = canonical_break_quality_key(retry["break_plan"])
        if not selected_quality < initial_quality:
            return None
        retry["reason_codes"] = _unique(
            [
                *list(retry.get("reason_codes") or []),
                "layout_intent_expanded_for_target_break_quality",
                "same_size_candidate_selected_before_font_reduction",
            ]
        )
        retry_box_model = dict(retry.get("box_model") or {})
        retry_box_model["break_quality_candidate"] = {
            "status": "selected",
            "selection_authority": "TypesettingEngine",
            "break_selector": self.break_planner.version,
            "second_break_selector_added": False,
            "comparison_order": [
                "canonical_line_break_quality_v1",
            ],
            "source_footprint_used_in_comparison": True,
            "initial_target_quality_key": list(initial_quality),
            "full_hard_bounds_target_quality_key": list(probe_quality),
            "selected_target_quality_key": list(selected_quality),
            "special_case_boundary_guard": False,
            "initial_confirmed_lexical_break_count": (
                _selected_lexical_break_count(
                    break_plan,
                    "confirmed_lexical_break_rank",
                )
            ),
            "selected_confirmed_lexical_break_count": (
                _selected_lexical_break_count(
                    retry["break_plan"],
                    "confirmed_lexical_break_rank",
                )
            ),
            "hard_bounds_unchanged": True,
            "full_hard_bounds_probe_is_executable_box": False,
            "minimal_authorized_expansion": list(
                retry.get("layout_intent_box") or []
            ),
        }
        retry["box_model"] = retry_box_model
        return retry

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
    value = style.get("target_preferred_em_px") if isinstance(style, dict) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(number) or number <= 0.0:
        return 0
    return max(1, int(round(number)))


def _typesetting_candidate_quality_summary(
    *,
    preferred_font_size: float,
    selected_font_size: float,
    fit_status: str,
    text_placement_complete: bool,
    hard_bounds_contained: bool,
    break_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the sole typography summary consumed by slot selection."""

    preferred = max(1.0, float(preferred_font_size or 1.0))
    selected = max(0.0, float(selected_font_size or 0.0))
    size_loss = max(0.0, preferred - selected)
    break_quality = canonical_break_quality_summary(break_plan)
    break_key = canonical_break_quality_key(break_plan)
    sort_key = (
        int(not bool(text_placement_complete)),
        int(str(fit_status or "") != "fits"),
        int(not bool(hard_bounds_contained)),
        round(size_loss, 6),
        *break_key[5:],
    )
    return {
        "quality_version": "typesetting_candidate_quality_v1",
        "selection_authority": "TypesettingEngine",
        "full_text_placed": bool(text_placement_complete),
        "fit_status": str(fit_status or ""),
        "hard_bounds_contained": bool(hard_bounds_contained),
        "preferred_font_size": round(preferred, 6),
        "selected_font_size": round(selected, 6),
        "preferred_size_loss": round(size_loss, 6),
        "break_quality": break_quality,
        "sort_key": list(sort_key),
        "readability_is_render_admission": False,
    }


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
    return list(range(preferred, minimum - 1, -1))


def _source_preferred_interval_floor(
    preferred: int,
    style: Mapping[str, Any] | None,
) -> int | None:
    """Return a rounded quality-comparison floor, never a fit/admission floor."""

    interval = list(dict(style or {}).get("target_preferred_em_interval_px") or [])
    if len(interval) < 2:
        return None
    try:
        lower = float(interval[0])
        upper = float(interval[1])
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower <= 0.0
        or upper < lower
    ):
        return None
    rounded_lower = int(math.floor(lower + 0.5))
    return max(1, min(int(preferred), rounded_lower))


def _selected_lexical_break_count(
    break_plan: Mapping[str, Any] | None,
    rank_field: str,
) -> int:
    count = 0
    for item in list(dict(break_plan or {}).get("selected_breaks") or []):
        try:
            rank = int(
                (item.get("opportunity_metadata") or {}).get(rank_field)
                or 0
            )
        except (TypeError, ValueError):
            continue
        count += int(rank > 0)
    return int(count)


def _vertical_break_plan_target_quality_key(
    break_plan: Mapping[str, Any] | None,
) -> tuple[Any, ...]:
    return canonical_break_quality_key(break_plan)


def _vertical_break_quality_probe_is_relevant(
    break_plan: Mapping[str, Any] | None,
) -> bool:
    quality = canonical_break_quality_summary(break_plan)
    evidence_defect = bool(
        any(quality.get("confirmed_lexical_integrity") or [])
        or float(quality.get("punctuation_attachment") or 0.0) != 0.0
        or any(quality.get("row_unit_segment_quality") or [])
        or any(quality.get("weak_lexical_evidence") or [])
    )
    if evidence_defect:
        return True

    # A topology-only retry may remove columns that the constrained intent box
    # added beyond the planner's source-aware desired topology. It may not
    # collapse an otherwise clean desired composition merely because the full
    # hard box can hold fewer columns.
    selected_topology = int(quality.get("target_topology_economy") or 1)
    desired_topology = max(
        1,
        int(dict(break_plan or {}).get("desired_columns") or selected_topology),
    )
    return bool(selected_topology > desired_topology)


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
    """Return the technical raster minimum; readability is diagnostic only."""

    return 1


def _style_fill_color(style: Mapping[str, Any] | None) -> str:
    values = dict(style or {})
    fill = values.get("fill")
    if isinstance(fill, Mapping):
        value = str(fill.get("color") or "")
        if value:
            return value
    return "#000000"


def _style_outline_color(style: Mapping[str, Any] | None) -> str:
    values = dict(style or {})
    outline = values.get("outline")
    if isinstance(outline, Mapping):
        value = str(outline.get("color") or "")
        if value:
            return value
    return "#FFFFFF"


def _style_outline_width(style: Mapping[str, Any] | None) -> float:
    values = dict(style or {})
    outline = values.get("outline")
    if not isinstance(outline, Mapping) or outline.get("present") is not True:
        return 0.0
    try:
        value = float(outline.get("target_width_px") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0.0 else 0.0


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
    stroke_width = _style_outline_width(style)
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


def _fit_base_text_ink_envelope(
    *,
    placements: Sequence[GlyphPlacement],
    lines: Sequence[Mapping[str, Any]],
    columns: Sequence[Mapping[str, Any]],
    shaped_runs: Sequence[ShapedRun],
    hard_bounds: Sequence[int],
    style: Mapping[str, Any] | None,
    layout_fit_status: str,
) -> dict[str, Any]:
    """Fit one conservative shaped/base-outline envelope as typeset geometry.

    HarfBuzz supplies outline dimensions; the single extra dimension pixel
    covers integer raster-grid enclosure. The calculation mirrors the
    rasterizer's crop-and-center policy but never draws or changes shaping.
    """

    hard = bbox_from_value(hard_bounds)
    logical_bounds = _union_bounds([item.bbox for item in placements])
    outline_width = _base_raster_outline_width(style)
    shaped_by_run = {
        str(item.metadata.get("font_span_id") or item.metadata.get("run_id") or ""): item
        for item in shaped_runs
        if str(item.metadata.get("font_span_id") or item.metadata.get("run_id") or "")
    }
    placement_records: list[dict[str, Any]] = []
    envelope_boxes: list[list[int]] = []
    for placement in placements:
        record = _placement_base_text_ink_envelope(
            placement,
            shaped_by_run,
            outline_width=outline_width,
        )
        if record:
            placement_records.append(record)
            envelope = bbox_from_value(record.get("envelope_bounds"))
            if envelope:
                envelope_boxes.append(envelope)
    input_envelope = _union_bounds(envelope_boxes) or list(logical_bounds)
    audit: dict[str, Any] = {
        "base_text_ink_envelope_version": BASE_TEXT_INK_ENVELOPE_VERSION,
        "policy": "harfbuzz_outline_base_paint_envelope_v1",
        "policy_owner": "TypesettingEngine",
        "status": "contained",
        "contained": bool(hard and input_envelope and _box_inside(input_envelope, hard)),
        "layout_fit_status_before_envelope": str(layout_fit_status or ""),
        "parent_hard_bounds": list(hard),
        "logical_placement_bounds": list(logical_bounds),
        "input_envelope_bounds": list(input_envelope),
        "output_envelope_bounds": list(input_envelope),
        "base_outline_width_px": int(outline_width),
        "hinted_dimension_guard_px": float(BASE_TEXT_HINTED_DIMENSION_GUARD_PX),
        "hinted_dimension_guard_policy": "one_total_pixel_after_outward_outline_span",
        "allowed_shift_x": [],
        "allowed_shift_y": [],
        "selected_shift": [0, 0],
        "relative_geometry_preserved": True,
        "font_size_changed": False,
        "breaks_changed": False,
        "writing_mode_changed": False,
        "placement_evidence": placement_records,
        "issues": [],
    }
    if not hard or not input_envelope:
        audit.update(
            {
                "status": "invalid",
                "contained": False,
                "issues": ["base_text_ink_envelope_bounds_invalid"],
            }
        )
        return {
            "placements": list(placements),
            "lines": [deepcopy(dict(item)) for item in lines],
            "columns": [deepcopy(dict(item)) for item in columns],
            "measured_bounds": list(logical_bounds),
            "audit": audit,
        }

    hard_right = hard[0] + hard[2]
    hard_bottom = hard[1] + hard[3]
    envelope_right = input_envelope[0] + input_envelope[2]
    envelope_bottom = input_envelope[1] + input_envelope[3]
    min_dx = int(hard[0] - input_envelope[0])
    max_dx = int(hard_right - envelope_right)
    min_dy = int(hard[1] - input_envelope[1])
    max_dy = int(hard_bottom - envelope_bottom)
    audit["allowed_shift_x"] = [min_dx, max_dx]
    audit["allowed_shift_y"] = [min_dy, max_dy]
    if min_dx > max_dx or min_dy > max_dy:
        audit.update(
            {
                "status": "exceeds_hard_bounds",
                "contained": False,
                "issues": ["complete_base_text_ink_envelope_exceeds_hard_bounds"],
            }
        )
        return {
            "placements": list(placements),
            "lines": [deepcopy(dict(item)) for item in lines],
            "columns": [deepcopy(dict(item)) for item in columns],
            "measured_bounds": list(input_envelope),
            "audit": audit,
        }

    if str(layout_fit_status or "") != "fits":
        audit["status"] = "deferred_existing_layout_overflow"
        audit["contained"] = _box_inside(input_envelope, hard)
        return {
            "placements": list(placements),
            "lines": [deepcopy(dict(item)) for item in lines],
            "columns": [deepcopy(dict(item)) for item in columns],
            "measured_bounds": list(input_envelope),
            "audit": audit,
        }

    dx = _closest_integer_to_zero(min_dx, max_dx)
    dy = _closest_integer_to_zero(min_dy, max_dy)
    audit["selected_shift"] = [dx, dy]
    output_envelope = [
        input_envelope[0] + dx,
        input_envelope[1] + dy,
        input_envelope[2],
        input_envelope[3],
    ]
    audit["output_envelope_bounds"] = output_envelope
    audit["contained"] = _box_inside(output_envelope, hard)
    if not audit["contained"]:
        audit.update(
            {
                "status": "invalid",
                "issues": ["selected_base_text_ink_shift_not_contained"],
            }
        )
        return {
            "placements": list(placements),
            "lines": [deepcopy(dict(item)) for item in lines],
            "columns": [deepcopy(dict(item)) for item in columns],
            "measured_bounds": list(input_envelope),
            "audit": audit,
        }
    if dx == 0 and dy == 0:
        return {
            "placements": list(placements),
            "lines": [deepcopy(dict(item)) for item in lines],
            "columns": [deepcopy(dict(item)) for item in columns],
            "measured_bounds": output_envelope,
            "audit": audit,
        }

    audit["status"] = "translated"
    shifted_placements = [
        _shift_glyph_placement(
            item,
            dx,
            dy,
            metadata_key="base_text_ink_fit_shift",
        )
        for item in placements
    ]
    return {
        "placements": shifted_placements,
        "lines": _shift_base_ink_coordinate_records(lines, dx, dy),
        "columns": _shift_base_ink_coordinate_records(columns, dx, dy),
        "measured_bounds": output_envelope,
        "audit": audit,
    }


def _placement_base_text_ink_envelope(
    placement: GlyphPlacement,
    shaped_by_run: Mapping[str, ShapedRun],
    *,
    outline_width: int,
) -> dict[str, Any]:
    box = bbox_from_value(placement.bbox)
    text = str(placement.text or "")
    if not box or not text or text.isspace() or not source_text_requires_visible_glyph(text):
        return {}
    metadata = dict(placement.metadata or {})
    mode = str(metadata.get("placement_mode") or "")
    if mode in {
        "vertical_ellipsis_sequence",
        "vertical_dash_sequence",
        "vertical_wave_sequence",
    }:
        return {
            "text": text,
            "run_id": str(metadata.get("run_id") or ""),
            "placement_mode": mode,
            "placement_bounds": list(box),
            "envelope_bounds": list(box),
            "evidence_status": "typesetting_drawing_primitive_bounds",
        }

    run_id = str(metadata.get("font_span_id") or metadata.get("run_id") or "")
    shaped = shaped_by_run.get(run_id)
    requested = [
        int(value)
        for value in list(metadata.get("shaped_glyph_ids") or [])
    ]
    position_policy = (
        "compact_horizontal_sequence_preserved"
        if mode == "vertical_emphasis_sequence"
        else "harfbuzz"
    )
    core_width, core_height, evidence_status = _predicted_shaped_core_size(
        shaped,
        requested,
        position_policy=position_policy,
        target_size=(box[2], box[3]),
    )
    natural_width = max(1, int(core_width) + int(outline_width) * 2)
    natural_height = max(1, int(core_height) + int(outline_width) * 2)
    dest_x = int(round((float(box[2]) - float(core_width)) / 2.0))
    dest_y = int(round((float(box[3]) - float(core_height)) / 2.0))
    natural_box = [
        box[0] + dest_x - int(outline_width),
        box[1] + dest_y - int(outline_width),
        natural_width,
        natural_height,
    ]
    envelope = _union_bounds([box, natural_box])
    return {
        "text": text,
        "run_id": run_id,
        "placement_mode": mode,
        "placement_bounds": list(box),
        "predicted_core_size": [int(core_width), int(core_height)],
        "predicted_natural_raster_size": [natural_width, natural_height],
        "predicted_natural_raster_bounds": natural_box,
        "envelope_bounds": envelope,
        "evidence_status": evidence_status,
    }


def _predicted_shaped_core_size(
    shaped: ShapedRun | None,
    requested_glyph_ids: Sequence[int],
    *,
    position_policy: str,
    target_size: tuple[int, int],
) -> tuple[int, int, str]:
    if shaped is None:
        return max(1, int(target_size[0])), max(1, int(target_size[1])), "fallback_logical_cell"
    glyphs = _select_shaped_glyphs(shaped, requested_glyph_ids)
    if not glyphs:
        return max(1, int(target_size[0])), max(1, int(target_size[1])), "fallback_logical_cell"
    boxes = [_shaped_glyph_outline_box(item, 0.0, 0.0) for item in glyphs]
    if any(not item for item in boxes):
        return max(1, int(target_size[0])), max(1, int(target_size[1])), "fallback_logical_cell"

    if position_policy == "compact_horizontal_sequence_preserved":
        widths = [_hinted_dimension_upper(item[2] - item[0]) for item in boxes]
        heights = [_hinted_dimension_upper(item[3] - item[1]) for item in boxes]
        available_gap = max(0.0, float(target_size[0] - sum(widths))) / float(max(1, len(widths) - 1))
        preferred_gap = max(1.0, round(float(shaped.font_size) * 0.06))
        gap = 0.0 if len(widths) <= 1 else max(0.0, min(preferred_gap, available_gap))
        return (
            max(1, int(round(sum(widths) + gap * max(0, len(widths) - 1)))),
            max(1, max(heights)),
            "harfbuzz_outline_extents_compact_horizontal",
        )

    pen_x = 0.0
    pen_y = 0.0
    positioned: list[list[float]] = []
    for glyph in glyphs:
        outline = _shaped_glyph_outline_box(glyph, pen_x, pen_y)
        if not outline:
            return max(1, int(target_size[0])), max(1, int(target_size[1])), "fallback_logical_cell"
        positioned.append(outline)
        pen_x += float(glyph.x_advance)
        pen_y += float(glyph.y_advance)
    left = min(item[0] for item in positioned)
    top = min(item[1] for item in positioned)
    right = max(item[2] for item in positioned)
    bottom = max(item[3] for item in positioned)
    return (
        _hinted_dimension_upper(right - left),
        _hinted_dimension_upper(bottom - top),
        "harfbuzz_outline_extents",
    )


def _select_shaped_glyphs(
    shaped: ShapedRun,
    requested_glyph_ids: Sequence[int],
) -> list[Any]:
    glyphs = list(shaped.glyphs or [])
    requested = [int(value) for value in requested_glyph_ids]
    if not requested:
        return glyphs
    available = [int(item.glyph_id) for item in glyphs]
    if requested == available:
        return glyphs
    width = len(requested)
    for start in range(0, len(available) - width + 1):
        if available[start : start + width] == requested:
            return glyphs[start : start + width]
    return []


def _shaped_glyph_outline_box(
    glyph: Any,
    pen_x: float,
    pen_y: float,
) -> list[float]:
    metadata = dict(getattr(glyph, "metadata", {}) or {})
    extents = metadata.get("outline_extents_px")
    if not isinstance(extents, Mapping):
        return []
    try:
        x_bearing = float(extents["x_bearing"])
        y_bearing = float(extents["y_bearing"])
        width = float(extents["width"])
        height = float(extents["height"])
        x_offset = float(glyph.x_offset)
        y_offset = float(glyph.y_offset)
    except (KeyError, TypeError, ValueError):
        return []
    x0 = float(pen_x) + x_offset + x_bearing
    x1 = x0 + width
    y0 = -(float(pen_y) + y_offset + y_bearing)
    y1 = -(float(pen_y) + y_offset + y_bearing + height)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _hinted_dimension_upper(value: float) -> int:
    return max(
        1,
        int(
            math.ceil(
                max(0.0, float(value))
                + BASE_TEXT_HINTED_DIMENSION_GUARD_PX
            )
        ),
    )


def _base_raster_outline_width(style: Mapping[str, Any] | None) -> int:
    return max(0, int(round(_style_outline_width(style))))


def _closest_integer_to_zero(lower: int, upper: int) -> int:
    if lower <= 0 <= upper:
        return 0
    return int(lower if lower > 0 else upper)


def _shift_base_ink_coordinate_records(
    values: Sequence[Mapping[str, Any]],
    dx: int,
    dy: int,
) -> list[dict[str, Any]]:
    shifted: list[dict[str, Any]] = []
    for value in values:
        item = deepcopy(dict(value))
        for key in ("x", "raw_x"):
            if item.get(key) is not None:
                item[key] = item[key] + dx
        for key in ("y", "raw_y"):
            if item.get(key) is not None:
                item[key] = item[key] + dy
        for key in (
            "bbox",
            "box",
            "display_box",
            "raw_box",
            "centered_block_box",
            "measured_bounds",
        ):
            box = bbox_from_value(item.get(key))
            if box:
                item[key] = [box[0] + dx, box[1] + dy, box[2], box[3]]
        alignment = item.get("layout_visual_alignment")
        if isinstance(alignment, Mapping):
            adjusted = deepcopy(dict(alignment))
            box = bbox_from_value(adjusted.get("measured_bounds"))
            if box:
                adjusted["measured_bounds"] = [
                    box[0] + dx,
                    box[1] + dy,
                    box[2],
                    box[3],
                ]
            adjusted["base_text_ink_fit_shift"] = [dx, dy]
            item["layout_visual_alignment"] = adjusted
        item["base_text_ink_fit_shift"] = [dx, dy]
        shifted.append(item)
    return shifted


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
) -> tuple[list[float], str]:
    metadata = plan.metadata if isinstance(plan.metadata, Mapping) else {}
    explicit_center = _point_from_value(metadata.get("visual_alignment_center"))
    if explicit_center and _point_inside_box(explicit_center, box):
        return explicit_center, "visual_alignment_center"

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


def _shift_glyph_placement(
    placement: GlyphPlacement,
    dx: int,
    dy: int,
    *,
    metadata_key: str = "layout_visual_alignment_shift",
) -> GlyphPlacement:
    bbox = bbox_from_value(placement.bbox)
    shifted_bbox = [bbox[0] + int(dx), bbox[1] + int(dy), bbox[2], bbox[3]] if bbox else list(placement.bbox)
    position = list(placement.position or [])
    if len(position) >= 2:
        position = [float(position[0]) + float(dx), float(position[1]) + float(dy), *position[2:]]
    metadata = dict(placement.metadata or {})
    metadata[str(metadata_key)] = [int(dx), int(dy)]
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


def _source_punctuation_geometry_match(
    plan: RenderLayerPlan,
    runs: Sequence[InlineTextRun],
    run: InlineTextRun,
    writing_mode: str,
) -> dict[str, Any]:
    """Match a lossless target run to one validated source visual occurrence."""

    expected_kind = {
        "dash_sequence": "dash",
        "wave_sequence": "wave",
    }.get(str(run.role or ""), "")
    if not expected_kind:
        return {}
    inline_axis = {"vertical": "ttb", "horizontal": "ltr"}.get(
        str(writing_mode or ""),
        "",
    )
    if not inline_axis:
        return {"status": "fallback_incompatible_writing_mode"}

    evidence = validated_source_punctuation_geometry_ref(plan)
    if not evidence:
        return {
            "status": "fallback_validated_source_punctuation_geometry_unavailable"
        }
    source_occurrences = [
        dict(item)
        for item in list(evidence.get("occurrences") or [])
        if isinstance(item, Mapping)
        and str(item.get("kind") or "") == expected_kind
        and str(item.get("inline_axis") or "") == inline_axis
    ]
    source_occurrences.sort(
        key=lambda item: (
            int(item.get("kind_ordinal") or 0),
            int(item.get("visual_reading_order_ordinal") or 0),
        )
    )

    translated_runs: list[InlineTextRun] = []
    for candidate in list(runs or []):
        compatible = [
            item
            for item in list(candidate.metadata.get("punctuation_occurrences") or [])
            if isinstance(item, Mapping)
            and str(item.get("kind") or "") == expected_kind
        ]
        if len(compatible) > 1:
            return {
                "status": "fallback_ambiguous_translated_occurrence_count"
            }
        if compatible:
            translated_runs.append(candidate)

    if len(source_occurrences) != len(translated_runs) or not source_occurrences:
        return {
            "status": "fallback_ambiguous_source_occurrence_count",
            "source_occurrence_count": len(source_occurrences),
            "translated_occurrence_count": len(translated_runs),
            "fact_set_id": str(evidence.get("fact_set_id") or ""),
        }
    target_index = next(
        (
            index
            for index, candidate in enumerate(translated_runs)
            if str(candidate.run_id or "") == str(run.run_id or "")
        ),
        -1,
    )
    if target_index < 0:
        return {"status": "fallback_translated_occurrence_not_found"}
    occurrence = source_occurrences[target_index]
    return {
        "status": "applied",
        "kind": expected_kind,
        "inline_axis": inline_axis,
        "occurrence_id": str(occurrence.get("occurrence_id") or ""),
        "visual_reading_order_ordinal": int(
            occurrence.get("visual_reading_order_ordinal") or 0
        ),
        "kind_ordinal": int(occurrence.get("kind_ordinal") or 0),
        "source_span_px": float(occurrence.get("span_px") or 0.0),
        "source_pitch_px": float(occurrence.get("pitch_px") or 0.0),
        "measurement_basis": str(
            occurrence.get("measurement_basis") or ""
        ),
        "source_cell_px": float(occurrence.get("source_cell_px") or 0.0),
        "normalized_span": float(occurrence.get("normalized_span") or 0.0),
        "normalized_pitch": float(occurrence.get("normalized_pitch") or 0.0),
        "source_group_bbox_page_xywh": list(
            occurrence.get("group_bbox_page_xywh") or []
        ),
        "confidence": float(occurrence.get("confidence") or 0.0),
        "fact_set_id": str(evidence.get("fact_set_id") or ""),
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
            source_geometry_match = (
                _source_punctuation_geometry_match(
                    plan,
                    runs,
                    run,
                    "vertical",
                )
                if run.role in {"dash_sequence", "wave_sequence"}
                else {}
            )
            row_units = _vertical_sequence_row_units(run)
            if source_geometry_match.get("status") == "applied":
                preferred_em = max(
                    1.0,
                    float(
                        (style or {}).get("target_preferred_em_px")
                        or font_size
                    ),
                )
                candidate_scale = min(
                    1.0,
                    max(0.0, float(font_size) / preferred_em),
                )
                requested_source_span = max(
                    1.0,
                    float(source_geometry_match.get("source_span_px") or 0.0),
                )
                item_height = requested_source_span * candidate_scale
                if source_geometry_match.get("measurement_basis") == (
                    SOURCE_PUNCTUATION_MEASUREMENT_BASIS_ABSOLUTE_STROKE
                ):
                    row_units = max(
                        1.0,
                        float(item_height) / max(1.0, float(font_size)),
                    )
                else:
                    row_units = max(
                        1.0,
                        float(
                            source_geometry_match.get("normalized_span")
                            or 0.0
                        ),
                    )
                source_geometry_match = {
                    **source_geometry_match,
                    "requested_span_px": requested_source_span,
                    "candidate_span_px": float(item_height),
                    "candidate_scale": candidate_scale,
                }
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
                    **{
                        (
                            "source_punctuation_geometry_match_status"
                            if key == "status"
                            else f"source_punctuation_geometry_{key}"
                        ): value
                        for key, value in source_geometry_match.items()
                    },
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
        stroke_width = _style_outline_width(style)
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
            _style_outline_width(style),
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
    primitive_metadata.update(
        {
            str(key): value
            for key, value in metadata.items()
            if str(key).startswith("source_punctuation_geometry_")
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
                "ellipsis_geometry_basis": (
                    "immutable_target_token_and_resolved_font"
                ),
                "ellipsis_count_basis": "immutable_target_token",
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
        sample_ys = [first_y]
        sample_y = first_y + float(sample_step)
        while sample_y < last_y:
            sample_ys.append(sample_y)
            sample_y += float(sample_step)
        if last_y > first_y:
            sample_ys.append(last_y)
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

    requested_source_span = float(
        primitive_metadata.get(
            "source_punctuation_geometry_requested_span_px"
        )
        or 0.0
    )
    if mode == "vertical_ellipsis_sequence" and centers:
        applied_source_span = (
            float(centers[-1][1]) - float(centers[0][1]) + float(diameter)
        )
    elif points:
        applied_source_span = float(points[-1][1]) - float(points[0][1])
    else:
        applied_source_span = 0.0
    if requested_source_span > 0.0:
        primitive_metadata.update(
            {
                "source_punctuation_geometry_applied_span_px": round(
                    applied_source_span,
                    6,
                ),
                "source_punctuation_geometry_source_span_miss_ratio": round(
                    abs(applied_source_span - requested_source_span)
                    / requested_source_span,
                    6,
                ),
                "source_punctuation_geometry_fit_downscaled": bool(
                    float(
                        primitive_metadata.get(
                            "source_punctuation_geometry_candidate_scale"
                        )
                        or 1.0
                    )
                    < 0.999999
                ),
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
