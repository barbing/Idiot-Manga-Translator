# -*- coding: utf-8 -*-
"""Pure Stage 4 typesetting engine.

The engine consumes RenderLayerPlan records and emits TypesetLayout/FitReport
records. It does not draw final text, mutate cleanup, or reinterpret parent
identity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from app.render.font_manager import FontManager
from app.render.text_shaper import HarfBuzzShaper, ShapedRun
from app.render.typesetting_contracts import FitReport, GlyphPlacement, RenderLayerPlan, TypesetLayout, bbox_from_value
from app.render.typesetting_text import (
    BreakOpportunity,
    VERTICAL_CENTERED_PUNCTUATION_CHARS,
    InlineTextRun,
    classify_grapheme,
    compute_break_opportunities,
    grapheme_clusters,
    normalize_for_writing_mode,
    segment_inline_runs,
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
    ) -> None:
        self.font_manager = font_manager or FontManager()
        self.shaper = shaper or HarfBuzzShaper(self.font_manager)
        self.policy = policy or TypesettingPolicy()

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

        writing_mode, writing_policy = self._resolve_writing_mode(plan)
        preferred_font_size = _font_size_from_style(plan.resolved_render_style)
        resolved = self.font_manager.resolve_font(
            plan.resolved_render_style,
            writing_mode=writing_mode,
            text=plan.translated_text,
        )
        if not resolved.usable or resolved.primary_face is None:
            return self._failed(plan, "missing_font", list(resolved.issues or ["missing_font"]), hard_bounds=hard_bounds)
        face = resolved.primary_face
        normalized, punctuation, symbols, normalization_notes = normalize_for_writing_mode(
            plan.translated_text,
            writing_mode,
            self.font_manager,
            face,
        )
        runs = segment_inline_runs(normalized, writing_mode=writing_mode, language_hint=_language_hint(plan))
        breaks = compute_break_opportunities(runs, writing_mode=writing_mode)
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
            shaped_runs = self._shape_runs(runs, face, font_size, writing_mode)
            layout_intent_box, box_model, reason_codes = self._layout_intent_box(
                plan=plan,
                hard_bounds=hard_bounds,
                target_box=target_box,
                writing_mode=writing_mode,
                font_size=font_size,
                text=normalized,
                shaped_runs=shaped_runs,
            )
            if writing_mode == "horizontal":
                placements, lines, columns, measured_bounds, fit_status, fit_issues = self._layout_horizontal(
                    normalized,
                    runs,
                    shaped_runs,
                    layout_intent_box,
                    font_size,
                    plan.resolved_render_style,
                )
            else:
                placements, lines, columns, measured_bounds, fit_status, fit_issues = self._layout_vertical(
                    normalized,
                    runs,
                    shaped_runs,
                    layout_intent_box,
                    font_size,
                    plan.resolved_render_style,
                    plan,
                )
            attempt = {
                "font_size": int(font_size),
                "fit_status": fit_status,
                "layout_intent_box": list(layout_intent_box),
                "measured_bounds": list(measured_bounds),
                "issues": list(fit_issues),
                "placements": placements,
                "lines": lines,
                "columns": columns,
                "shaped_runs": shaped_runs,
                "box_model": box_model,
                "reason_codes": reason_codes,
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
        fit_status = str(selected_attempt["fit_status"])
        fit_issues = list(selected_attempt["issues"])
        box_model = dict(selected_attempt["box_model"])
        reason_codes = list(selected_attempt["reason_codes"])
        if font_size < preferred_font_size:
            reason_codes.append("font_size_reduced_to_fit_translated_text")
            if bool(plan.resolved_render_style.get("font_size_locked")):
                reason_codes.append("font_size_lock_relaxed_for_layout_fit")
        kinsoku_adjustments = [item.to_audit_dict() for item in breaks if not item.allowed and item.reason.startswith("kinsoku")]
        if fit_status != "fits" and kinsoku_adjustments and "kinsoku_fit_conflict" not in fit_issues:
            fit_issues.append("kinsoku_fit_conflict")
        if fit_status != "fits" and writing_mode == "vertical":
            fit_issues.append("writing_mode_fit_failure")

        all_issues = _unique([*resolved.issues, *style_issues, *fit_issues])
        full_text_placed = fit_status == "fits"
        metadata = {
            "typesetting_engine_version": "typesetting_engine_stage4_v1",
            "box_model": box_model,
            "writing_mode_policy": writing_policy,
            "inline_runs": [_run_audit(run, writing_mode, self.policy) for run in runs],
            "break_opportunities": [item.to_audit_dict() for item in breaks],
            "chosen_breaks": _chosen_breaks(lines, columns, writing_mode),
            "kinsoku_adjustments": kinsoku_adjustments,
            "line_break_policy": {
                "policy_version": "line_break_policy_stage4_v1",
                "locale_hint": _language_hint(plan),
                "writing_mode": writing_mode,
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
                    }
                    for item in attempts
                ],
            },
            "render_style": {
                "line_height": plan.resolved_render_style.get("line_height"),
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
            punctuation_placements=punctuation,
            symbol_placements=symbols,
            measured_bounds=measured_bounds,
            visual_center=_center_of(measured_bounds),
            fit_status=fit_status,
            normalized_text=normalized,
            original_text=plan.translated_text,
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
            metadata={
                "typesetting_engine_version": "typesetting_engine_stage4_v1",
                "reason_codes": reason_codes,
                "writing_mode_policy": {
                    **writing_policy,
                    "block_writing_mode_flip_forbidden": True,
                },
                "script_policy": script_policy,
                "line_break_policy": metadata["line_break_policy"],
                "break_opportunities": metadata["break_opportunities"],
                "chosen_breaks": metadata["chosen_breaks"],
                "kinsoku_adjustments": kinsoku_adjustments,
                "normalization_notes": normalization_notes,
                "inline_runs": metadata["inline_runs"],
                "shaped_runs": metadata["shaped_runs"],
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

    def _shape_runs(self, runs: Sequence[InlineTextRun], face, font_size: int, writing_mode: str) -> list[ShapedRun]:
        shaped: list[ShapedRun] = []
        for run in runs:
            if not run.normalized_text or run.role == "space":
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
            shaped.append(
                replace(
                    shaped_run,
                    metadata={
                        **dict(shaped_run.metadata),
                        "run_id": run.run_id,
                        "run_role": run.role,
                        "grapheme_start": run.grapheme_start,
                        "grapheme_end": run.grapheme_end,
                        "block_writing_mode": writing_mode,
                        "shape_writing_mode": shape_writing_mode,
                        "inline_placement_mode": placement_mode,
                    },
                )
            )
        return shaped

    def _layout_intent_box(
        self,
        *,
        plan: RenderLayerPlan,
        hard_bounds: list[int],
        target_box: list[int],
        writing_mode: str,
        font_size: int,
        text: str,
        shaped_runs: Sequence[ShapedRun],
    ) -> tuple[list[int], dict[str, Any], list[str]]:
        x, y, w, h = target_box
        hard_x, hard_y, hard_w, hard_h = hard_bounds
        source_box = bbox_from_value(plan.source_provenance_ref.get("source_contract_bbox") if isinstance(plan.source_provenance_ref, dict) else [])
        reason_codes: list[str] = []
        evidence = "translated_text_natural_box"
        vertical_profile: dict[str, Any] = {}
        if writing_mode == "vertical":
            vertical_profile = _vertical_layout_profile(
                plan=plan,
                box=target_box,
                font_size=font_size,
                text=text,
                shaped_runs=shaped_runs,
                policy=self.policy,
            )
            natural_w, natural_h = _natural_box_size(
                text,
                writing_mode,
                font_size,
                plan.resolved_render_style,
                shaped_runs,
                desired_columns=int(vertical_profile.get("desired_columns") or 1),
            )
            if vertical_profile.get("source_columns"):
                reason_codes.append("source_column_structure_used_for_vertical_layout")
        else:
            natural_w, natural_h = _natural_box_size(text, writing_mode, font_size, plan.resolved_render_style, shaped_runs)
        if writing_mode == "vertical" and vertical_profile.get("source_columns"):
            column_w = float(vertical_profile.get("column_width") or _vertical_column_pitch(font_size, font_size))
            source_columns = int(vertical_profile.get("source_columns") or 0)
            max_target_columns = max(1, int(math.floor(max(1, w) / max(1.0, column_w))))
            reserved_columns = max(1, min(source_columns, max_target_columns))
            reserved_w = int(math.ceil(reserved_columns * column_w))
            if reserved_w > natural_w:
                natural_w = min(max(1, w), reserved_w)
                reason_codes.append("source_column_capacity_reserved_for_vertical_layout_quality")
        if w > max(natural_w * 2, natural_w + font_size) or h > max(natural_h * 2, natural_h + font_size):
            reason_codes.append("oversized_parent_bbox_not_used_as_fill_box")
        source_center: list[float] = []
        source_center_candidate = _center_of(source_box)
        if source_box and source_center_candidate and _point_inside_box(source_center_candidate, hard_bounds):
            source_center = source_center_candidate
            if source_box != target_box:
                reason_codes.append("source_contract_bbox_used_as_layout_alignment_prior")
            else:
                reason_codes.append("source_contract_bbox_equals_target_box")
        natural_w = min(max(1, natural_w), max(1, w))
        natural_h = min(max(1, natural_h), max(1, h))
        if source_center:
            intent = [
                int(round(float(source_center[0]) - float(natural_w) / 2.0)),
                int(round(float(source_center[1]) - float(natural_h) / 2.0)),
                int(natural_w),
                int(natural_h),
            ]
            reason_codes.append("layout_intent_centered_on_source_footprint")
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
            "source_contract_bbox_is_alignment_prior": bool(source_center),
            "vertical_layout_profile": vertical_profile,
            "rejected_box_candidates": [],
        }, reason_codes

    def _layout_vertical(
        self,
        normalized: str,
        runs: Sequence[InlineTextRun],
        shaped_runs: Sequence[ShapedRun],
        box: list[int],
        font_size: int,
        style: dict[str, Any],
        plan: RenderLayerPlan,
    ) -> tuple[list[GlyphPlacement], list[dict[str, Any]], list[dict[str, Any]], list[int], str, list[str]]:
        x, y, w, h = box
        line_height = _line_height(style)
        items = _vertical_layout_items(runs, shaped_runs, self.policy, font_size)
        shaped_cell_h = _dominant_vertical_advance(shaped_runs)
        cell_h = max(1.0, shaped_cell_h * line_height)
        cell_h = max(cell_h, max((_vertical_item_cell_height(item) for item in items), default=0.0))
        column_w = _vertical_column_pitch(font_size, max((float(item.get("width", 0.0)) for item in items), default=0.0))
        profile = _vertical_layout_profile(
            plan=plan,
            box=box,
            font_size=font_size,
            text=normalized,
            shaped_runs=shaped_runs,
            policy=self.policy,
            item_count=len(items),
            column_width=column_w,
            cell_height=cell_h,
        )
        columns_needed = max(1, int(profile.get("desired_columns") or 1)) if items else 1
        total_row_units = _vertical_items_row_units(items)
        max_rows = max(1, int(math.floor(h / cell_h)))
        max_columns = max(1, int(math.floor(w / column_w)))
        columns_needed = min(max(1, columns_needed), max(1, len(items)), max_columns)
        rows = max(1, int(math.ceil(total_row_units / columns_needed))) if items else 1
        while rows > max_rows and columns_needed < min(max_columns, max(1, len(items))):
            columns_needed += 1
            rows = max(1, int(math.ceil(total_row_units / columns_needed)))
        column_groups, grouping_meta = _choose_vertical_column_groups(
            items,
            desired_columns=columns_needed,
            max_columns=min(max_columns, max(1, len(items))),
            max_rows=max_rows,
            profile=profile,
        )
        rows = max((int(math.ceil(_vertical_items_row_units(group))) for group in column_groups), default=rows)
        columns_needed = max(1, len(column_groups))
        required_w = int(math.ceil(columns_needed * column_w))
        required_h = int(math.ceil(rows * cell_h))
        fit_status = "fits" if required_w <= w and required_h <= h and items else "overflow"
        issues: list[str] = []
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
                        font_family=shaped_runs[0].font_face_id if shaped_runs else "",
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
                            "placement_mode": item.get("placement_mode", "vertical_glyph"),
                            "placement_source": "stage4_vertical_layout",
                            "shaped_glyph_id": int(shaped_glyph.glyph_id) if shaped_glyph else None,
                            "shaped_glyph_name": shaped_glyph.glyph_name if shaped_glyph else "",
                            "shaped_glyph_ids": [int(glyph.glyph_id) for glyph in shaped_glyphs],
                            "shaped_x_advance_total": float(item.get("x_advance", 0.0)),
                            "shaped_y_advance": float(shaped_glyph.y_advance) if shaped_glyph else 0.0,
                            "shaped_position_authority": bool(shaped_glyph),
                        },
                    )
                )
                row_cursor += row_units
            cursor += len(group)
        columns: list[dict[str, Any]] = []
        for idx in range(columns_needed):
            raw_x = int(round(base_x + block_w - float((idx + 1) * column_w)))
            raw_y = int(round(base_y))
            raw_box = [raw_x, raw_y, int(max(1, math.ceil(column_w))), int(max(1, math.ceil(rows * cell_h)))]
            clipped = _clamp_box(raw_box, [x, y, w, h])
            columns.append(
                {
                    "column_index": idx,
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
        alignment_center, alignment_source = _layout_alignment_center(plan, [x, y, w, h])
        if alignment_center and measured:
            dx, dy = _measured_alignment_shift(measured, alignment_center, [x, y, w, h])
            if dx or dy:
                placements = [_shift_glyph_placement(item, dx, dy) for item in placements]
                columns = _shift_column_records(columns, dx, dy)
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
        return placements, lines, columns, measured, fit_status, _unique(issues)

    def _layout_horizontal(
        self,
        normalized: str,
        runs: Sequence[InlineTextRun],
        shaped_runs: Sequence[ShapedRun],
        box: list[int],
        font_size: int,
        style: dict[str, Any],
    ) -> tuple[list[GlyphPlacement], list[dict[str, Any]], list[dict[str, Any]], list[int], str, list[str]]:
        x, y, w, h = box
        line_height = max(1.0, font_size * _line_height(style))
        placements: list[GlyphPlacement] = []
        cursor_x = float(x)
        cursor_y = float(y)
        line_index = 0
        issues: list[str] = []
        for run in runs:
            text = run.normalized_text
            if run.role == "space":
                advance = font_size * 0.35
                if cursor_x > x and cursor_x + advance > x + w:
                    cursor_x = float(x)
                    cursor_y += line_height
                    line_index += 1
                placements.append(
                    GlyphPlacement(
                        text=text,
                        bbox=[int(cursor_x), int(cursor_y), int(max(1, advance)), int(max(1, line_height))],
                        position=[cursor_x, cursor_y],
                        font_family=shaped_runs[0].font_face_id if shaped_runs else "",
                        font_size=float(font_size),
                        advance=advance,
                        writing_mode="horizontal",
                        metadata={"run_id": run.run_id, "line_index": line_index, "space_run": True},
                    )
                )
                cursor_x += advance
                continue
            shaped = next((item for item in shaped_runs if item.text == text), None)
            advance = sum(max(0.0, glyph.x_advance) for glyph in shaped.glyphs) if shaped else font_size * len(text)
            if cursor_x > x and cursor_x + advance > x + w:
                cursor_x = float(x)
                cursor_y += line_height
                line_index += 1
            if run.script == "Latn" and advance > w:
                issues.append("word_overflow_break_applied")
            visible_width = min(max(1.0, advance), max(1.0, float(w)))
            placements.append(
                GlyphPlacement(
                    text=text,
                    bbox=[int(cursor_x), int(cursor_y), int(visible_width), int(max(1, line_height))],
                    position=[cursor_x, cursor_y],
                    font_family=shaped.font_face_id if shaped else "",
                    font_size=float(font_size),
                    advance=advance,
                    writing_mode="horizontal",
                    metadata={"run_id": run.run_id, "line_index": line_index},
                )
            )
            cursor_x += advance
        measured = _union_bounds([item.bbox for item in placements]) or [x, y, 1, 1]
        overflow = measured[1] + measured[3] > y + h or measured[0] + measured[2] > x + w
        if overflow:
            issues.append("layout_overflow")
        lines = [{"line_index": idx, "writing_mode": "horizontal"} for idx in range(line_index + 1)]
        return placements, lines, [], measured, "overflow" if overflow else "fits", _unique(issues)

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
    for key in ("font_size", "font_size_hint", "font_size_px", "size"):
        value = style.get(key) if isinstance(style, dict) else None
        try:
            if value is not None and float(value) > 0:
                return max(1, int(round(float(value))))
        except (TypeError, ValueError):
            continue
    return 24


def _font_size_candidates(
    preferred: int,
    style: dict[str, Any],
    policy: TypesettingPolicy,
    target_box: Sequence[int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[int]:
    preferred = max(1, int(preferred))
    minimum = _minimum_fit_font_size(preferred, style, policy)
    if minimum >= preferred:
        return [preferred]
    max_steps = max(1, int(policy.max_binary_fit_steps))
    step = max(1, int(math.ceil((preferred - minimum) / max_steps)))
    values = list(range(preferred, minimum - 1, -step))
    if values[-1] != minimum:
        values.append(minimum)
    return sorted(_unique_ints(values), reverse=True)


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
    if writing_mode == "vertical" and run.script == "Latn":
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
    if placement in {"tate_chu_yoko", "rotated_latin_run", "vertical_inline_run"}:
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
) -> tuple[int, int]:
    count = max(1, len(grapheme_clusters(text)))
    line_height = _line_height(style)
    if writing_mode == "horizontal":
        shaped_width = 0.0
        for run in shaped_runs or []:
            shaped_width += sum(max(0.0, float(glyph.x_advance)) for glyph in run.glyphs)
        if shaped_width > 0.0:
            shaped_width += sum(1 for cluster in grapheme_clusters(text) if cluster.isspace()) * font_size * 0.35
            return int(math.ceil(shaped_width)), int(max(1, font_size * line_height))
        return int(max(1, min(count, 12) * font_size * 0.75)), int(max(1, font_size * line_height))
    inline_width = 0.0
    for run in shaped_runs or []:
        if (run.metadata or {}).get("shape_writing_mode") == "horizontal":
            inline_width = max(inline_width, sum(max(0.0, abs(float(glyph.x_advance))) for glyph in run.glyphs))
    column_width = _vertical_column_pitch(font_size, inline_width)
    cell_height = max(font_size * line_height, _dominant_vertical_advance(shaped_runs or []) * line_height)
    columns = max(1, min(count, int(desired_columns or 1)))
    rows = max(1, int(math.ceil(count / columns)))
    if columns > 1 and _text_has_no_column_start_punctuation(text):
        rows += 1
    return int(max(1, math.ceil(columns * column_width))), int(max(1, math.ceil(rows * cell_height)))


def _vertical_layout_profile(
    *,
    plan: RenderLayerPlan,
    box: Sequence[int],
    font_size: int,
    text: str,
    shaped_runs: Sequence[ShapedRun],
    policy: TypesettingPolicy,
    item_count: int | None = None,
    column_width: float | None = None,
    cell_height: float | None = None,
) -> dict[str, Any]:
    target = bbox_from_value(box)
    count = max(1, int(item_count if item_count is not None else len(grapheme_clusters(text))))
    if not target:
        return {"desired_columns": 1, "source_columns": 0, "max_columns": 1, "reason": "missing_target_box"}
    _x, _y, w, h = target
    inline_width = 0.0
    for run in shaped_runs or []:
        if (run.metadata or {}).get("shape_writing_mode") == "horizontal":
            inline_width = max(inline_width, sum(max(0.0, abs(float(glyph.x_advance))) for glyph in run.glyphs))
    column_w = float(column_width or _vertical_column_pitch(font_size, inline_width))
    cell_h = float(cell_height or max(_dominant_vertical_advance(shaped_runs) * _line_height(plan.resolved_render_style), font_size * _line_height(plan.resolved_render_style), 1.0))
    max_columns = max(1, min(count, int(math.floor(max(1, w) / max(1.0, column_w)))))
    max_rows = max(1, int(math.floor(max(1, h) / max(1.0, cell_h))))
    source_box = bbox_from_value(plan.source_provenance_ref.get("source_contract_bbox") if isinstance(plan.source_provenance_ref, dict) else [])
    source_columns = _source_vertical_column_count(source_box, font_size)
    source_rows = _source_vertical_row_capacity(source_box, font_size, _line_height(plan.resolved_render_style))
    style = dict(plan.resolved_render_style or {})
    semantic_class = str(style.get("semantic_class") or style.get("source_role") or plan.role or "")
    source_role = str(style.get("source_role") or plan.role or "")
    if count <= 3:
        desired = 1
        reason = "short_vertical_text_single_column"
    elif source_rows:
        desired = int(math.ceil(float(count) / float(max(1, source_rows))))
        if source_columns:
            desired = min(desired, source_columns)
        reason = "source_row_capacity_column_structure"
    elif source_columns:
        desired = min(source_columns, max(1, int(math.ceil(float(count) / 5.0))))
        reason = "source_column_upper_bound"
    else:
        desired = max(1, int(round(math.sqrt(count * max(1.0, float(w)) / max(1.0, float(h))))))
        reason = "target_aspect_column_estimate"
    desired = min(max(1, desired), max_columns)
    while desired < max_columns and math.ceil(count / desired) > max_rows:
        desired += 1
        reason = f"{reason}_expanded_for_height_fit"
    return {
        "desired_columns": int(max(1, desired)),
        "source_columns": int(source_columns),
        "source_rows": int(source_rows),
        "max_columns": int(max_columns),
        "max_rows": int(max_rows),
        "item_count": int(count),
        "column_width": round(column_w, 3),
        "cell_height": round(cell_h, 3),
        "source_contract_bbox": list(source_box),
        "semantic_class": semantic_class,
        "source_role": source_role,
        "reason": reason,
    }


def _source_vertical_column_count(source_box: Sequence[int], font_size: int) -> int:
    source = bbox_from_value(source_box)
    if not source:
        return 0
    _x, _y, w, h = source
    if w <= 0 or h <= 0:
        return 0
    if w / max(1.0, float(h)) < 0.45:
        return 1
    pitch = _vertical_column_pitch(font_size, font_size)
    return max(1, min(8, int(round(float(w) / pitch))))


def _source_vertical_row_capacity(source_box: Sequence[int], font_size: int, line_height: float) -> int:
    source = bbox_from_value(source_box)
    if not source:
        return 0
    _x, _y, _w, h = source
    if h <= 0:
        return 0
    pitch = max(1.0, float(font_size) * max(0.75, float(line_height)))
    return max(1, min(16, int(round(float(h) / pitch))))


def _vertical_column_pitch(font_size: int, content_width: float) -> float:
    base = max(1.0, float(font_size))
    width = max(1.0, float(content_width or 0.0))
    return max(base * 1.24, width + base * 0.2)


def _choose_vertical_column_groups(
    items: Sequence[dict[str, Any]],
    *,
    desired_columns: int,
    max_columns: int,
    max_rows: int,
    profile: dict[str, Any],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    values = list(items or [])
    if not values:
        return [[]], {"strategy": "empty"}
    desired = max(1, min(int(desired_columns or 1), len(values)))
    limit = max(1, min(int(max_columns or desired), len(values)))
    row_limit = max(1, int(max_rows or len(values)))
    total_row_units = _vertical_items_row_units(values)
    source_columns = max(0, int(profile.get("source_columns") or 0)) if isinstance(profile, dict) else 0
    candidates = set(range(1, limit + 1))
    candidates.add(desired)
    if source_columns:
        candidates.add(min(limit, source_columns))
    min_fit_columns = int(math.ceil(float(total_row_units) / float(row_limit)))
    candidates.add(max(1, min(limit, min_fit_columns)))
    best: tuple[float, list[list[dict[str, Any]]], dict[str, Any]] | None = None
    for columns in sorted(value for value in candidates if 1 <= value <= limit):
        if math.ceil(total_row_units / columns) > row_limit:
            continue
        result = _best_vertical_partition(
            values,
            columns,
            row_limit,
            desired,
            source_columns,
            penalize_non_phrase_extra_columns=_vertical_profile_needs_speech_column_conservation(profile),
        )
        if result is None:
            continue
        score, groups, meta = result
        if best is None or score < best[0]:
            best = (score, groups, meta)
    if best is None:
        groups = _balanced_vertical_column_groups(values, min(limit, max(1, min_fit_columns)), row_limit)
        return groups, {
            "strategy": "balanced_fallback",
            "desired_columns": desired,
            "selected_columns": len(groups),
            "max_rows": row_limit,
        }
    score, groups, meta = best
    meta.update(
        {
            "strategy": "quality_scored_vertical_columns",
            "desired_columns": desired,
            "selected_columns": len(groups),
            "max_columns": limit,
            "max_rows": row_limit,
            "source_columns": source_columns,
            "score": round(float(score), 3),
        }
    )
    return groups, meta


def _best_vertical_partition(
    items: Sequence[dict[str, Any]],
    columns: int,
    max_rows: int,
    desired_columns: int,
    source_columns: int,
    *,
    penalize_non_phrase_extra_columns: bool,
) -> tuple[float, list[list[dict[str, Any]]], dict[str, Any]] | None:
    values = list(items or [])
    count = len(values)
    if columns <= 0 or count <= 0 or columns > count:
        return None
    total_row_units = _vertical_items_row_units(values)
    if math.ceil(total_row_units / columns) > max_rows:
        return None
    ideal = float(total_row_units) / float(columns)
    prefix_units = [0.0]
    for item in values:
        prefix_units.append(prefix_units[-1] + _vertical_item_row_units(item))

    def units_between(start: int, end: int) -> float:
        return max(0.0, prefix_units[end] - prefix_units[start])

    dp: dict[tuple[int, int], tuple[float, list[int]]] = {(0, 0): (0.0, [])}
    for col in range(columns):
        next_dp: dict[tuple[int, int], tuple[float, list[int]]] = {}
        for (used_cols, start), (score, breaks) in dp.items():
            if used_cols != col:
                continue
            remaining_cols = columns - col - 1
            min_end = start + 1
            max_end = min(count - remaining_cols, count)
            for end in range(min_end, max_end + 1):
                segment_units = units_between(start, end)
                if segment_units > float(max_rows):
                    break
                remaining = count - end
                remaining_units = units_between(end, count)
                if remaining_cols and math.ceil(remaining_units / remaining_cols) > max_rows:
                    continue
                if remaining < remaining_cols:
                    continue
                segment = values[start:end]
                segment_score = _vertical_segment_score(values, start, end, ideal)
                total = score + segment_score
                key = (col + 1, end)
                prior = next_dp.get(key)
                new_breaks = breaks + [end]
                if prior is None or total < prior[0]:
                    next_dp[key] = (total, new_breaks)
        dp = next_dp
        if not dp:
            return None
    final = dp.get((columns, count))
    if final is None:
        return None
    score, breaks = final
    score += abs(columns - max(1, desired_columns)) * 2.25
    if source_columns:
        score += max(0, columns - source_columns) * 4.0
        if columns < min(source_columns, count) and _has_strong_vertical_phrase_break(values):
            score += max(0, min(source_columns, count) - columns) * 0.75
    cursor = 0
    groups: list[list[dict[str, Any]]] = []
    split_points: list[int] = []
    for end in breaks:
        groups.append(values[cursor:end])
        if end < count:
            split_points.append(end)
        cursor = end
    extra_columns = max(0, columns - max(1, desired_columns, int(math.ceil(float(total_row_units) / float(max(1, max_rows))))))
    non_phrase_extra_break_penalty = 0.0
    if penalize_non_phrase_extra_columns and extra_columns:
        non_phrase_extra_break_penalty = sum(
            max(0.0, float(_vertical_break_penalty(values, split)))
            for split in split_points
        )
        score += non_phrase_extra_break_penalty * 2.0
    meta = {
        "split_points": split_points,
        "column_lengths": [len(group) for group in groups],
        "column_row_units": [round(float(_vertical_items_row_units(group)), 3) for group in groups],
        "break_penalties": [
            {
                "split_after": split,
                "previous": str(values[split - 1].get("text") or "") if split > 0 else "",
                "next": str(values[split].get("text") or "") if split < count else "",
                "penalty": round(float(_vertical_break_penalty(values, split)), 3),
            }
            for split in split_points
        ],
    }
    if extra_columns:
        meta["extra_columns_beyond_desired"] = extra_columns
        meta["non_phrase_extra_break_penalty"] = round(float(non_phrase_extra_break_penalty), 3)
        meta["non_phrase_extra_break_penalty_applied"] = bool(penalize_non_phrase_extra_columns)
    return score, groups, meta


def _vertical_profile_needs_speech_column_conservation(profile: dict[str, Any]) -> bool:
    if not isinstance(profile, dict):
        return False
    semantic = str(profile.get("semantic_class") or "").lower()
    role = str(profile.get("source_role") or "").lower()
    return (
        "speech" in semantic
        or "bubble" in semantic
        or role in {"speech", "speech_bubble", "dialogue"}
    )


def _vertical_item_row_units(item: Mapping[str, Any]) -> float:
    try:
        value = float(item.get("row_units", 1.0))
    except Exception:
        value = 1.0
    return max(1.0, value)


def _vertical_items_row_units(items: Sequence[Mapping[str, Any]]) -> float:
    return sum(_vertical_item_row_units(item) for item in (items or []))


def _vertical_item_cell_height(item: Mapping[str, Any]) -> float:
    row_units = _vertical_item_row_units(item)
    try:
        height = float(item.get("height", 0.0))
    except Exception:
        height = 0.0
    if height <= 0.0:
        return 0.0
    return height / row_units


def _balanced_vertical_column_groups(items: Sequence[dict[str, Any]], columns_needed: int, max_rows: int) -> list[list[dict[str, Any]]]:
    values = list(items or [])
    if not values:
        return [[]]
    columns = max(1, min(int(columns_needed), len(values)))
    groups: list[list[dict[str, Any]]] = []
    cursor = 0
    for col in range(columns):
        remaining_items = len(values) - cursor
        remaining_columns = max(1, columns - col)
        if col == columns - 1:
            take = remaining_items
        else:
            remaining_units = _vertical_items_row_units(values[cursor:])
            target_units = min(float(max_rows), max(1.0, remaining_units / float(remaining_columns)))
            take = 0
            taken_units = 0.0
            while cursor + take < len(values):
                items_left_after = len(values) - (cursor + take + 1)
                if items_left_after < remaining_columns - 1:
                    break
                next_units = _vertical_item_row_units(values[cursor + take])
                if take > 0 and taken_units + next_units > target_units and taken_units >= 1.0:
                    break
                take += 1
                taken_units += next_units
                if taken_units >= target_units:
                    break
            if take <= 0 and remaining_items > 0:
                take = 1
        groups.append(values[cursor: cursor + take])
        cursor += take
    if cursor < len(values) and groups:
        groups[-1].extend(values[cursor:])
    return [group for group in groups if group]


def _vertical_segment_score(items: Sequence[dict[str, Any]], start: int, end: int, ideal: float) -> float:
    segment = list(items[start:end])
    if not segment:
        return 1000.0
    segment_units = _vertical_items_row_units(segment)
    score = abs(float(segment_units) - ideal) * 1.1
    if start > 0:
        first = str(segment[0].get("text") or "")
        if _item_must_not_start_vertical_column(segment[0]):
            score += 80.0
        elif _vertical_item_is_weak_column_start(first):
            score += 10.0
    last = str(segment[-1].get("text") or "")
    if _vertical_item_is_open_punctuation(last):
        score += 30.0
    if len(segment) == 1 and len(items) > 3:
        score += 9.0
    if _vertical_segment_is_punctuation_only(segment) and not _vertical_segment_is_punctuation_only(items):
        score += 45.0
    if end < len(items):
        score += _vertical_break_penalty(items, end)
    return score


def _vertical_break_penalty(items: Sequence[dict[str, Any]], split: int) -> float:
    if split <= 0 or split >= len(items):
        return 0.0
    prev_text = str(items[split - 1].get("text") or "")
    next_text = str(items[split].get("text") or "")
    if _vertical_item_is_continuation_punctuation(next_text):
        return 90.0
    if _vertical_item_is_open_punctuation(prev_text):
        return 40.0
    if _vertical_item_is_sequence_punctuation(prev_text) and _vertical_item_is_sequence_punctuation(next_text):
        return 85.0
    if _vertical_item_is_strong_phrase_end(prev_text) and not _vertical_item_is_continuation_punctuation(next_text):
        return -8.0
    if _vertical_item_is_cjk(prev_text) and _vertical_item_is_cjk(next_text):
        return 4.5
    return 0.0


def _item_must_not_start_vertical_column(item: dict[str, Any]) -> bool:
    text = str(item.get("text") or "")
    return bool(text) and text[:1] in _VERTICAL_NO_COLUMN_START


def _text_has_no_column_start_punctuation(text: str) -> bool:
    return any(cluster[:1] in _VERTICAL_NO_COLUMN_START for cluster in grapheme_clusters(str(text or "")))


def _vertical_item_is_open_punctuation(text: str) -> bool:
    return str(text or "")[:1] in {"(", "（", "[", "［", "{", "｛", "「", "『", "【", "〈", "《", "“", "‘"}


def _vertical_item_is_continuation_punctuation(text: str) -> bool:
    return str(text or "")[:1] in _VERTICAL_NO_COLUMN_START


def _vertical_item_is_sequence_punctuation(text: str) -> bool:
    return str(text or "")[:1] in {"︙", "︱", "…", "-", "—", "―", "─", "︕", "︖", "‼", "⁇", "⁉", "⁈"}


def _vertical_item_is_strong_phrase_end(text: str) -> bool:
    return str(text or "")[:1] in {
        "︙",
        "︱",
        "。",
        "，",
        "、",
        "︐",
        "︑",
        "︒",
        "！",
        "？",
        "︕",
        "︖",
        "‼",
        "⁇",
        "⁉",
        "⁈",
        "!",
        "?",
        "～",
        "〜",
        "~",
    }


def _vertical_item_is_weak_column_start(text: str) -> bool:
    return str(text or "")[:1] in {"个", "的", "了", "吗", "呢", "吧", "啊", "呀", "嘛", "啦", "，", "、"}


def _vertical_item_is_cjk(text: str) -> bool:
    if not text:
        return False
    kind = classify_grapheme(str(text)[0])
    return kind == "cjk"


def _vertical_item_is_centered_punctuation(text: str) -> bool:
    value = str(text or "")
    return value in VERTICAL_CENTERED_PUNCTUATION_CHARS


def _vertical_segment_is_punctuation_only(items: Sequence[dict[str, Any]]) -> bool:
    values = [str(item.get("text") or "") for item in items or [] if str(item.get("text") or "")]
    if not values:
        return False
    return all(
        _vertical_item_is_continuation_punctuation(text)
        or _vertical_item_is_open_punctuation(text)
        or _vertical_item_is_sequence_punctuation(text)
        or _vertical_item_is_strong_phrase_end(text)
        for text in values
    )


def _has_strong_vertical_phrase_break(items: Sequence[dict[str, Any]]) -> bool:
    return any(_vertical_break_penalty(items, index) < 0 for index in range(1, len(items)))


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


def _layout_alignment_center(plan: RenderLayerPlan, box: Sequence[int]) -> tuple[list[float], str]:
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


def _shift_glyph_placement(placement: GlyphPlacement, dx: int, dy: int) -> GlyphPlacement:
    bbox = bbox_from_value(placement.bbox)
    shifted_bbox = [bbox[0] + int(dx), bbox[1] + int(dy), bbox[2], bbox[3]] if bbox else list(placement.bbox)
    position = list(placement.position or [])
    if len(position) >= 2:
        position = [float(position[0]) + float(dx), float(position[1]) + float(dy), *position[2:]]
    metadata = dict(placement.metadata or {})
    metadata["layout_visual_alignment_shift"] = [int(dx), int(dy)]
    return replace(placement, bbox=shifted_bbox, position=position, metadata=metadata)


def _shift_column_records(columns: Sequence[dict[str, Any]], dx: int, dy: int) -> list[dict[str, Any]]:
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
        shifted.append(item)
    return shifted


def _chosen_breaks(lines: Sequence[dict[str, Any]], columns: Sequence[dict[str, Any]], writing_mode: str) -> list[dict[str, Any]]:
    items = columns if writing_mode == "vertical" else lines
    return [
        {
            "index": int(item.get("column_index", item.get("line_index", idx))),
            "writing_mode": writing_mode,
            "reason": "measured_fit",
        }
        for idx, item in enumerate(items)
    ]


def _vertical_layout_items(runs: Sequence[InlineTextRun], shaped_runs: Sequence[ShapedRun], policy: TypesettingPolicy, font_size: int) -> list[dict[str, Any]]:
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
        if run.role in {"ellipsis_sequence", "dash_sequence", "punctuation_sequence"}:
            item_width, item_height, x_advance = _vertical_atomic_size(run, glyphs, policy, font_size)
            row_units = _vertical_sequence_row_units(run)
            items.append(
                {
                    "text": run.text,
                    "run_id": run.run_id,
                    "placement_mode": f"vertical_{run.role}",
                    "shaped_glyph": glyphs[0] if glyphs else None,
                    "shaped_glyphs": glyphs,
                    "width": item_width,
                    "height": item_height,
                    "row_units": row_units,
                    "x_advance": x_advance,
                }
            )
            continue
        if _vertical_run_is_atomic(run, policy):
            item_width, item_height, x_advance = _vertical_atomic_size(run, glyphs, policy, font_size)
            items.append(
                {
                    "text": run.text,
                    "run_id": run.run_id,
                    "placement_mode": _run_audit(run, "vertical", policy)["placement_mode"],
                    "shaped_glyph": glyphs[0] if glyphs else None,
                    "shaped_glyphs": glyphs,
                    "width": item_width,
                    "height": item_height,
                    "row_units": 1.0,
                    "x_advance": x_advance,
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
                    "placement_mode": placement_mode,
                    "shaped_glyph": glyph,
                    "shaped_glyphs": [glyph] if glyph else [],
                    "width": max(1.0, glyph_w),
                    "height": max(1.0, glyph_h),
                    "row_units": 1.0,
                    "x_advance": abs(float(glyph.x_advance)) if glyph else 0.0,
                }
            )
    return items


def _vertical_sequence_row_units(run: InlineTextRun) -> float:
    if run.role == "dash_sequence":
        return float(max(1, len(grapheme_clusters(run.text))))
    return 1.0


def _vertical_atomic_size(run: InlineTextRun, glyphs: Sequence[Any], policy: TypesettingPolicy, font_size: int) -> tuple[float, float, float]:
    if run.role in {"ellipsis_sequence", "dash_sequence", "punctuation_sequence"}:
        count = max(1, len(grapheme_clusters(run.text)))
        width = float(font_size)
        height = float(font_size)
        if run.role == "dash_sequence" and count > 1:
            height = float(font_size) * float(count)
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


def _vertical_run_is_atomic(run: InlineTextRun, policy: TypesettingPolicy) -> bool:
    if run.script == "Latn":
        return True
    if run.role in {"numeric_token", "complex_script", "symbol", "ellipsis_sequence", "dash_sequence", "punctuation_sequence"}:
        return True
    return False


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
