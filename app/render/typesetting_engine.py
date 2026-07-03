# -*- coding: utf-8 -*-
"""Pure Stage 4 typesetting engine.

The engine consumes RenderLayerPlan records and emits TypesetLayout/FitReport
records. It does not draw final text, mutate cleanup, or reinterpret parent
identity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

from app.render.font_manager import FontManager
from app.render.text_shaper import HarfBuzzShaper, ShapedRun
from app.render.typesetting_contracts import FitReport, GlyphPlacement, RenderLayerPlan, TypesetLayout, bbox_from_value
from app.render.typesetting_text import (
    BreakOpportunity,
    InlineTextRun,
    compute_break_opportunities,
    grapheme_clusters,
    normalize_for_writing_mode,
    segment_inline_runs,
)


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
        font_size = _font_size_from_style(plan.resolved_render_style)
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
        shaped_runs = self._shape_runs(runs, face, font_size, writing_mode)
        style_issues = _style_issues(runs, writing_mode, self.policy)
        script_policy = _script_policy(runs, writing_mode, self.policy)
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
            )
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
            fallback_used=False,
            scaling_used=1.0,
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
        evidence = "natural_text_box"
        if source_box and _box_inside(source_box, hard_bounds):
            intent = source_box
            evidence = "source_contract_bbox"
        else:
            natural_w, natural_h = _natural_box_size(text, writing_mode, font_size, plan.resolved_render_style, shaped_runs)
            if w > max(natural_w * 2, natural_w + font_size) or h > max(natural_h * 2, natural_h + font_size):
                reason_codes.append("oversized_parent_bbox_not_used_as_fill_box")
            natural_w = min(max(1, natural_w), max(1, w))
            natural_h = min(max(1, natural_h), max(1, h))
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
    ) -> tuple[list[GlyphPlacement], list[dict[str, Any]], list[dict[str, Any]], list[int], str, list[str]]:
        x, y, w, h = box
        line_height = _line_height(style)
        items = _vertical_layout_items(runs, shaped_runs, self.policy, font_size)
        shaped_cell_h = _dominant_vertical_advance(shaped_runs)
        cell_h = max(1.0, shaped_cell_h * line_height)
        cell_h = max(cell_h, max((float(item.get("height", 0.0)) for item in items), default=0.0))
        column_w = max(1.0, font_size * 1.05, max((float(item.get("width", 0.0)) for item in items), default=0.0))
        rows = max(1, int(math.floor(h / cell_h)))
        columns_needed = max(1, int(math.ceil(len(items) / rows))) if items else 1
        required_w = int(math.ceil(columns_needed * column_w))
        fit_status = "fits" if required_w <= w and items else "overflow"
        issues: list[str] = []
        if fit_status != "fits":
            issues.append("layout_overflow")
            issues.append("line_break_fit_failure")
        run_modes = _run_modes(runs, self.policy, "vertical")
        if any(run.script == "Latn" and len(grapheme_clusters(run.text)) > self.policy.max_vertical_compact_latin_graphemes for run in runs):
            issues.append("long_latin_vertical_fit_review")
        placements: list[GlyphPlacement] = []
        for index, item in enumerate(items):
            col = index // rows
            row = index % rows
            item_w = int(max(1, min(float(item.get("width") or column_w), max(1, w))))
            item_h = int(max(1, min(float(item.get("height") or cell_h), max(1, h))))
            raw_column_x = x + w - float((col + 1) * column_w)
            raw_px = int(round(raw_column_x + max(0.0, (column_w - item_w) / 2.0)))
            raw_py = y + int(round(row * cell_h))
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
                    advance=cell_h,
                    writing_mode="vertical",
                    metadata={
                        "cluster_index": index,
                        "column": col,
                        "row": row,
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
        columns: list[dict[str, Any]] = []
        for idx in range(columns_needed):
            raw_x = x + w - int(round((idx + 1) * column_w))
            raw_box = [raw_x, y, int(max(1, math.ceil(column_w))), h]
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
                    "clipped_to_hard_bounds": clipped != raw_box,
                    "overflow_column": fit_status != "fits" and (raw_x < x or raw_x + raw_box[2] > x + w),
                }
            )
        measured = _union_bounds([item.bbox for item in placements]) or [x, y, min(w, required_w), min(h, int(rows * cell_h))]
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


def _natural_box_size(text: str, writing_mode: str, font_size: int, style: dict[str, Any], shaped_runs: Sequence[ShapedRun] | None = None) -> tuple[int, int]:
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
    rows = min(max(1, count), 8)
    columns = max(1, math.ceil(count / rows))
    inline_width = 0.0
    for run in shaped_runs or []:
        if (run.metadata or {}).get("shape_writing_mode") == "horizontal":
            inline_width = max(inline_width, sum(max(0.0, abs(float(glyph.x_advance))) for glyph in run.glyphs))
    column_width = max(font_size * 1.1, inline_width)
    return int(max(1, math.ceil(columns * column_width))), int(max(1, rows * font_size * line_height))


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
            items.append(
                {
                    "text": cluster,
                    "run_id": run.run_id,
                    "placement_mode": "vertical_glyph",
                    "shaped_glyph": glyph,
                    "shaped_glyphs": [glyph] if glyph else [],
                    "width": max(1.0, glyph_w),
                    "height": max(1.0, glyph_h),
                    "x_advance": abs(float(glyph.x_advance)) if glyph else 0.0,
                }
            )
    return items


def _vertical_atomic_size(run: InlineTextRun, glyphs: Sequence[Any], policy: TypesettingPolicy, font_size: int) -> tuple[float, float, float]:
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
    if run.role in {"numeric_token", "complex_script", "symbol", "ellipsis_sequence", "dash_sequence"}:
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
