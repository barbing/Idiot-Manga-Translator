# -*- coding: utf-8 -*-
"""Authoritative line and column partition planning.

The planner consumes the explicit ``BreakOpportunity`` records produced by
``typesetting_text``.  A non-terminal split that cannot be traced to an allowed
record is never considered a legal candidate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.render.typesetting_text import BreakOpportunity, classify_grapheme


LINE_BREAK_PLANNER_VERSION = "line_break_planner_v2_target_lexical"
LEXICOGRAPHIC_VERTICAL_ORDER = [
    "hard_legality",
    "fit",
    "source_relative_preferred_size",
    "lexical_span_integrity",
    "punctuation_attachment",
    "optical_balance",
    "advisory_source_footprint_distribution",
    "rtl_frontload_fallback",
]
LEXICOGRAPHIC_HORIZONTAL_ORDER = LEXICOGRAPHIC_VERTICAL_ORDER[:-1]

_OPEN_PUNCTUATION = {"(", "（", "[", "［", "{", "｛", "「", "『", "【", "〈", "《", "“", "‘"}
_CONTINUATION_PUNCTUATION = {
    ")", "）", "]", "］", "}", "｝", "」", "』", "】", "〉", "》", "”", "’",
    "。", "，", "、", "．", ".", ",", "!", "?", "！", "？", "︕", "︖", "‼",
    "⁇", "⁉", "⁈", "︙", "︐", "︑", "︒", "︓", "︔",
}
_SEQUENCE_PUNCTUATION = {
    "︙", "︱", "︴", "…", "-", "—", "―", "─", "～", "〜", "~", "〰",
    "︕", "︖", "‼", "⁇", "⁉", "⁈",
}
_STRONG_PHRASE_END = {
    "︙", "︱", "︴", "。", "，", "、", "︐", "︑", "︒", "！", "？", "︕",
    "︖", "‼", "⁇", "⁉", "⁈", "!", "?", "～", "〜", "~", "〰",
}
@dataclass
class BreakPlanResult:
    groups: list[list[Any]]
    selected_breaks: list[dict[str, Any]] = field(default_factory=list)
    boundary_candidates: list[dict[str, Any]] = field(default_factory=list)
    rejected_breaks: list[dict[str, Any]] = field(default_factory=list)
    candidate_partitions: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            **dict(self.metadata),
            "selected_breaks": [dict(item) for item in self.selected_breaks],
            "boundary_candidates": [dict(item) for item in self.boundary_candidates],
            "rejected_breaks": [dict(item) for item in self.rejected_breaks],
            "candidate_partitions": [dict(item) for item in self.candidate_partitions],
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class _VerticalPath:
    breaks: tuple[int, ...]
    previous_units: float
    fit_overflow_count: int
    fit_overflow_units: float
    lexical_penalty: float
    punctuation_penalty: float
    balance_penalty: float
    source_layout_error: float
    frontload_penalty: float


@dataclass(frozen=True)
class _HorizontalPath:
    breaks: tuple[int, ...]
    width_overflow_count: int
    width_overflow_amount: float
    lexical_penalty: float
    punctuation_penalty: float
    raggedness: float


class LineBreakPlanner:
    version = LINE_BREAK_PLANNER_VERSION

    def plan_vertical(
        self,
        items: Sequence[Mapping[str, Any]],
        opportunities: Sequence[BreakOpportunity],
        *,
        desired_columns: int,
        max_columns: int,
        max_rows: int,
        profile: Mapping[str, Any] | None = None,
    ) -> BreakPlanResult:
        values = [dict(item) for item in (items or [])]
        boundary_candidates, boundary_by_split = _boundary_catalog(values, opportunities)
        rejected = [
            {**item, "rejection_reason": str(item.get("reason") or "missing_break_opportunity")}
            for item in boundary_candidates
            if not bool(item.get("allowed"))
        ]
        if not values:
            return BreakPlanResult(
                groups=[[]],
                boundary_candidates=boundary_candidates,
                rejected_breaks=rejected,
                metadata={
                    "line_break_planner_version": self.version,
                    "selection_authority": "explicit_break_opportunities",
                    "strategy": "empty",
                    "lexicographic_decision_order": list(LEXICOGRAPHIC_VERTICAL_ORDER),
                },
            )

        desired = max(1, min(int(desired_columns or 1), len(values)))
        limit = max(1, min(int(max_columns or desired), len(values)))
        desired = min(desired, limit)
        row_limit = max(1, int(max_rows or len(values)))
        total_units = _items_row_units(values)
        minimum_fit_columns = max(1, int(math.ceil(total_units / float(row_limit))))
        minimum_columns = min(limit, max(desired, minimum_fit_columns))
        column_counts = list(range(minimum_columns, limit + 1))
        if not column_counts:
            column_counts = [limit]

        profile_value = dict(profile or {})
        candidate_records: list[dict[str, Any]] = []
        candidates: list[tuple[tuple[Any, ...], _VerticalPath, list[list[dict[str, Any]]], dict[str, Any]]] = []
        for columns in column_counts:
            path, evaluated = _best_vertical_path(
                values,
                boundary_by_split,
                columns=columns,
                max_rows=row_limit,
                profile=profile_value,
            )
            if path is None:
                candidate_records.append(
                    {
                        "columns": columns,
                        "legal_candidate_count": evaluated,
                        "selected": False,
                        "rejection_reason": "no_complete_legal_partition",
                    }
                )
                continue
            groups = _groups_from_breaks(values, path.breaks)
            structural_penalty = _vertical_structure_penalty(
                profile_value,
                desired,
                columns,
                values,
                path.breaks[:-1],
            )
            key = _vertical_candidate_key(path, structural_penalty, desired, columns)
            record = _vertical_candidate_audit(
                values,
                groups,
                path,
                key,
                columns=columns,
                desired_columns=desired,
                legal_candidate_count=evaluated,
                structural_penalty=structural_penalty,
            )
            candidate_records.append(record)
            candidates.append((key, path, groups, record))

        if not candidates:
            groups = [values]
            return BreakPlanResult(
                groups=groups,
                boundary_candidates=boundary_candidates,
                rejected_breaks=rejected,
                candidate_partitions=candidate_records,
                issues=["no_legal_break_partition", "line_break_fit_failure"],
                metadata={
                    "line_break_planner_version": self.version,
                    "selection_authority": "explicit_break_opportunities",
                    "strategy": "unpartitioned_explicit_failure",
                    "desired_columns": desired,
                    "selected_columns": 1,
                    "max_columns": limit,
                    "max_rows": row_limit,
                    "lexicographic_decision_order": list(LEXICOGRAPHIC_VERTICAL_ORDER),
                },
            )

        candidates.sort(key=lambda item: item[0])
        _key, selected_path, groups, selected_record = candidates[0]
        for record in candidate_records:
            record["selected"] = record is selected_record
        selected_breaks = [
            _selected_break_record(boundary_by_split[split], split, ordinal)
            for ordinal, split in enumerate(selected_path.breaks[:-1])
        ]
        fit_overflow = selected_path.fit_overflow_count > 0
        extra_columns = max(0, len(groups) - desired)
        punctuation_adjustments = _partition_quality_adjustments(groups)
        metadata = {
            "line_break_planner_version": self.version,
            "selection_authority": "explicit_break_opportunities",
            "strategy": "authoritative_break_opportunity_partition",
            "lexicographic_decision_order": list(LEXICOGRAPHIC_VERTICAL_ORDER),
            "desired_columns": desired,
            "selected_columns": len(groups),
            "max_columns": limit,
            "max_rows": row_limit,
            "source_columns": int(profile_value.get("source_columns") or 0),
            "split_points": list(selected_path.breaks[:-1]),
            "column_lengths": [len(group) for group in groups],
            "column_row_units": [round(_items_row_units(group), 3) for group in groups],
            "right_to_left_frontload_penalty": round(selected_path.frontload_penalty, 3),
            "vertical_break_quality_rules": ["terminal_cjk_widow_before_punctuation_tail"],
            "segment_quality_adjustments": punctuation_adjustments,
            "break_penalties": [
                {
                    "split_after": split,
                    "previous": str(values[split - 1].get("text") or ""),
                    "next": str(values[split].get("text") or ""),
                    "lexical_integrity_penalty": round(
                        _boundary_lexical_penalty(boundary_by_split.get(split)),
                        3,
                    ),
                    "punctuation_attachment_penalty": round(
                        _boundary_punctuation_penalty(values, split),
                        3,
                    ),
                }
                for split in selected_path.breaks[:-1]
            ],
            "extra_columns_beyond_desired": extra_columns,
            "non_phrase_extra_break_penalty": float(
                _vertical_structure_penalty(
                    profile_value,
                    desired,
                    len(groups),
                    values,
                    selected_path.breaks[:-1],
                )
            ),
            "non_phrase_extra_break_penalty_applied": bool(
                extra_columns and _profile_needs_speech_column_conservation(profile_value)
            ),
            "selected_lexicographic_key": selected_record.get("lexicographic_key", {}),
        }
        issues = ["line_break_fit_failure"] if fit_overflow else []
        return BreakPlanResult(
            groups=groups,
            selected_breaks=selected_breaks,
            boundary_candidates=boundary_candidates,
            rejected_breaks=rejected,
            candidate_partitions=candidate_records,
            issues=issues,
            metadata=metadata,
        )

    def plan_horizontal(
        self,
        items: Sequence[Mapping[str, Any]],
        opportunities: Sequence[BreakOpportunity],
        *,
        max_width: float,
        max_lines: int,
    ) -> BreakPlanResult:
        values = [dict(item) for item in (items or [])]
        boundary_candidates, boundary_by_split = _boundary_catalog(values, opportunities)
        rejected = [
            {**item, "rejection_reason": str(item.get("reason") or "missing_break_opportunity")}
            for item in boundary_candidates
            if not bool(item.get("allowed"))
        ]
        if not values:
            return BreakPlanResult(
                groups=[[]],
                boundary_candidates=boundary_candidates,
                rejected_breaks=rejected,
                metadata={
                    "line_break_planner_version": self.version,
                    "selection_authority": "explicit_break_opportunities",
                    "strategy": "empty",
                    "lexicographic_decision_order": list(
                        LEXICOGRAPHIC_HORIZONTAL_ORDER
                    ),
                },
            )

        width_limit = max(1.0, float(max_width or 1.0))
        line_limit = max(1, int(max_lines or 1))
        candidate_records: list[dict[str, Any]] = []
        candidates: list[tuple[tuple[Any, ...], _HorizontalPath, list[list[dict[str, Any]]], dict[str, Any]]] = []
        for line_count in range(1, len(values) + 1):
            path, evaluated = _best_horizontal_path(
                values,
                boundary_by_split,
                line_count=line_count,
                max_width=width_limit,
            )
            if path is None:
                candidate_records.append(
                    {
                        "lines": line_count,
                        "legal_candidate_count": evaluated,
                        "selected": False,
                        "rejection_reason": "no_complete_legal_partition",
                    }
                )
                continue
            groups = _groups_from_breaks(values, path.breaks)
            height_overflow = max(0, line_count - line_limit)
            key = (
                path.width_overflow_count,
                round(path.width_overflow_amount, 6),
                height_overflow,
                0,
                round(path.lexical_penalty, 6),
                round(path.punctuation_penalty, 6),
                round(path.raggedness, 6),
                line_count,
                0,
                path.breaks,
            )
            record = {
                "lines": line_count,
                "line_widths": [round(_items_advance(group), 3) for group in groups],
                "split_points": list(path.breaks[:-1]),
                "legal_candidate_count": evaluated,
                "selected": False,
                "lexicographic_key": {
                    "hard_legality": 0,
                    "fit": [path.width_overflow_count, round(path.width_overflow_amount, 3), height_overflow],
                    "source_relative_preferred_size": 0,
                    "lexical_span_integrity": round(path.lexical_penalty, 3),
                    "punctuation_attachment": round(path.punctuation_penalty, 3),
                    "optical_balance": [round(path.raggedness, 3), line_count],
                    "advisory_source_footprint_distribution": 0,
                },
            }
            candidate_records.append(record)
            candidates.append((key, path, groups, record))

        if not candidates:
            return BreakPlanResult(
                groups=[values],
                boundary_candidates=boundary_candidates,
                rejected_breaks=rejected,
                candidate_partitions=candidate_records,
                issues=["no_legal_break_partition", "line_break_fit_failure"],
                metadata={
                    "line_break_planner_version": self.version,
                    "selection_authority": "explicit_break_opportunities",
                    "strategy": "unpartitioned_explicit_failure",
                    "max_width": round(width_limit, 3),
                    "max_lines": line_limit,
                    "lexicographic_decision_order": list(
                        LEXICOGRAPHIC_HORIZONTAL_ORDER
                    ),
                },
            )

        candidates.sort(key=lambda item: item[0])
        _key, selected_path, groups, selected_record = candidates[0]
        for record in candidate_records:
            record["selected"] = record is selected_record
        selected_breaks = [
            _selected_break_record(boundary_by_split[split], split, ordinal)
            for ordinal, split in enumerate(selected_path.breaks[:-1])
        ]
        selected_lines = len(groups)
        issues: list[str] = []
        if selected_path.width_overflow_count:
            issues.append("line_break_fit_failure")
            if any(
                len(group) == 1 and float(group[0].get("advance") or 0.0) > width_limit
                for group in groups
            ):
                issues.append("atomic_run_overflow")
        if selected_lines > line_limit:
            issues.extend(["layout_overflow", "line_break_fit_failure"])
        metadata = {
            "line_break_planner_version": self.version,
            "selection_authority": "explicit_break_opportunities",
            "strategy": "authoritative_break_opportunity_partition",
            "lexicographic_decision_order": list(
                LEXICOGRAPHIC_HORIZONTAL_ORDER
            ),
            "max_width": round(width_limit, 3),
            "max_lines": line_limit,
            "selected_lines": selected_lines,
            "split_points": list(selected_path.breaks[:-1]),
            "line_widths": [round(_items_advance(group), 3) for group in groups],
            "selected_lexicographic_key": selected_record.get("lexicographic_key", {}),
        }
        return BreakPlanResult(
            groups=groups,
            selected_breaks=selected_breaks,
            boundary_candidates=boundary_candidates,
            rejected_breaks=rejected,
            candidate_partitions=candidate_records,
            issues=_unique(issues),
            metadata=metadata,
        )


def _boundary_catalog(
    items: Sequence[Mapping[str, Any]],
    opportunities: Sequence[BreakOpportunity],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    by_pair = {
        (str(item.before_run_id), str(item.after_run_id)): item
        for item in (opportunities or [])
    }
    records: list[dict[str, Any]] = []
    by_split: dict[int, dict[str, Any]] = {}
    for split in range(1, len(items)):
        before = items[split - 1]
        after = items[split]
        before_id = str(before.get("run_id") or "")
        after_id = str(after.get("run_id") or "")
        opportunity = by_pair.get((before_id, after_id))
        if opportunity is None:
            record = {
                "split_after_item_index": split,
                "before_run_id": before_id,
                "after_run_id": after_id,
                "before_text": str(before.get("text") or ""),
                "after_text": str(after.get("text") or ""),
                "position": None,
                "strength": "forbidden",
                "reason": "atomic_run_internal_boundary" if before_id and before_id == after_id else "missing_break_opportunity",
                "allowed": False,
            }
        else:
            record = {
                "split_after_item_index": split,
                "before_run_id": str(opportunity.before_run_id),
                "after_run_id": str(opportunity.after_run_id),
                "before_text": str(before.get("text") or ""),
                "after_text": str(after.get("text") or ""),
                "position": int(opportunity.position),
                "strength": str(opportunity.strength),
                "reason": str(opportunity.reason),
                "allowed": bool(opportunity.allowed),
                "opportunity_metadata": dict(opportunity.metadata or {}),
            }
        records.append(record)
        by_split[split] = record
    return records, by_split


def _best_vertical_path(
    items: Sequence[Mapping[str, Any]],
    boundary_by_split: Mapping[int, Mapping[str, Any]],
    *,
    columns: int,
    max_rows: int,
    profile: Mapping[str, Any],
) -> tuple[_VerticalPath | None, int]:
    count = len(items)
    if columns <= 0 or columns > count:
        return None, 0
    prefix = [0.0]
    for item in items:
        prefix.append(prefix[-1] + _item_row_units(item))
    ideal = prefix[-1] / float(columns)
    source_distribution = _reliable_source_distribution(profile, columns)
    states: dict[tuple[int, int], list[_VerticalPath]] = {
        (0, 0): [
            _VerticalPath(
                breaks=(),
                previous_units=0.0,
                fit_overflow_count=0,
                fit_overflow_units=0.0,
                lexical_penalty=0.0,
                punctuation_penalty=0.0,
                balance_penalty=0.0,
                source_layout_error=0.0,
                frontload_penalty=0.0,
            )
        ]
    }
    evaluated = 0
    for column in range(columns):
        next_states: dict[tuple[int, int], list[_VerticalPath]] = {}
        for (used, start), paths in states.items():
            if used != column:
                continue
            remaining_columns = columns - column - 1
            min_end = start + 1
            max_end = count - remaining_columns
            for end in range(min_end, max_end + 1):
                if end < count and not bool((boundary_by_split.get(end) or {}).get("allowed")):
                    continue
                evaluated += 1
                units = max(0.0, prefix[end] - prefix[start])
                overflow = max(0.0, units - float(max_rows))
                source_error = 0.0
                if source_distribution:
                    source_error = abs(units - float(source_distribution[column]))
                boundary = boundary_by_split.get(end) if end < count else None
                lexical = _boundary_lexical_penalty(boundary)
                punctuation = _segment_punctuation_penalty(items, start, end)
                if end < count:
                    punctuation += _boundary_punctuation_penalty(items, end)
                balance = abs(units - ideal) + _segment_optical_penalty(
                    items,
                    start,
                    end,
                )
                for path in paths:
                    frontload = path.frontload_penalty
                    if path.breaks:
                        frontload += max(0.0, units - path.previous_units)
                    candidate = _VerticalPath(
                        breaks=(*path.breaks, end),
                        previous_units=units,
                        fit_overflow_count=path.fit_overflow_count + int(overflow > 1e-9),
                        fit_overflow_units=path.fit_overflow_units + overflow,
                        lexical_penalty=path.lexical_penalty + lexical,
                        punctuation_penalty=path.punctuation_penalty + punctuation,
                        balance_penalty=path.balance_penalty + balance,
                        source_layout_error=path.source_layout_error + source_error,
                        frontload_penalty=frontload,
                    )
                    key = (column + 1, end)
                    next_states.setdefault(key, []).append(candidate)
        states = {key: _prune_vertical_paths(paths) for key, paths in next_states.items()}
        if not states:
            return None, evaluated
    final_paths = states.get((columns, count), [])
    if not final_paths:
        return None, evaluated
    final_paths.sort(key=_vertical_path_core_key)
    return final_paths[0], evaluated


def _prune_vertical_paths(paths: Sequence[_VerticalPath]) -> list[_VerticalPath]:
    if not paths:
        return []
    core = min(_vertical_path_prefix_key(path) for path in paths)
    matching = [path for path in paths if _vertical_path_prefix_key(path) == core]
    by_previous_units: dict[float, _VerticalPath] = {}
    for path in matching:
        unit_key = round(path.previous_units, 6)
        prior = by_previous_units.get(unit_key)
        if prior is None or (path.frontload_penalty, path.breaks) < (prior.frontload_penalty, prior.breaks):
            by_previous_units[unit_key] = path
    return list(by_previous_units.values())


def _vertical_path_prefix_key(path: _VerticalPath) -> tuple[Any, ...]:
    return (
        path.fit_overflow_count,
        round(path.fit_overflow_units, 6),
        round(path.lexical_penalty, 6),
        round(path.punctuation_penalty, 6),
        round(path.balance_penalty, 6),
        round(path.source_layout_error, 6),
    )


def _vertical_path_core_key(path: _VerticalPath) -> tuple[Any, ...]:
    return (*_vertical_path_prefix_key(path), round(path.frontload_penalty, 6), path.breaks)


def _vertical_candidate_key(
    path: _VerticalPath,
    structural_penalty: float,
    desired_columns: int,
    columns: int,
) -> tuple[Any, ...]:
    return (
        0,
        path.fit_overflow_count,
        round(path.fit_overflow_units, 6),
        0,
        round(path.lexical_penalty, 6),
        round(path.punctuation_penalty, 6),
        round(path.balance_penalty, 6),
        max(0, columns - desired_columns),
        round(path.source_layout_error + structural_penalty, 6),
        round(path.frontload_penalty, 6),
        path.breaks,
    )


def _vertical_candidate_audit(
    items: Sequence[Mapping[str, Any]],
    groups: Sequence[Sequence[Mapping[str, Any]]],
    path: _VerticalPath,
    key: tuple[Any, ...],
    *,
    columns: int,
    desired_columns: int,
    legal_candidate_count: int,
    structural_penalty: float,
) -> dict[str, Any]:
    return {
        "columns": columns,
        "desired_columns": desired_columns,
        "split_points": list(path.breaks[:-1]),
        "column_lengths": [len(group) for group in groups],
        "column_row_units": [round(_items_row_units(group), 3) for group in groups],
        "legal_candidate_count": legal_candidate_count,
        "selected": False,
        "lexicographic_key": {
            "hard_legality": 0,
            "fit": [path.fit_overflow_count, round(path.fit_overflow_units, 3)],
            "source_relative_preferred_size": 0,
            "lexical_span_integrity": round(path.lexical_penalty, 3),
            "punctuation_attachment": round(path.punctuation_penalty, 3),
            "optical_balance": [round(path.balance_penalty, 3), max(0, columns - desired_columns)],
            "advisory_source_footprint_distribution": round(
                path.source_layout_error + structural_penalty,
                3,
            ),
            "rtl_frontload_fallback": round(path.frontload_penalty, 3),
        },
        "sort_key": [item if not isinstance(item, tuple) else list(item) for item in key[:-1]],
        "segment_quality_adjustments": _partition_quality_adjustments(groups),
    }


def _best_horizontal_path(
    items: Sequence[Mapping[str, Any]],
    boundary_by_split: Mapping[int, Mapping[str, Any]],
    *,
    line_count: int,
    max_width: float,
) -> tuple[_HorizontalPath | None, int]:
    count = len(items)
    if line_count <= 0 or line_count > count:
        return None, 0
    prefix = [0.0]
    for item in items:
        prefix.append(prefix[-1] + max(0.0, float(item.get("advance") or 0.0)))
    states: dict[tuple[int, int], _HorizontalPath] = {
        (0, 0): _HorizontalPath((), 0, 0.0, 0.0, 0.0, 0.0)
    }
    evaluated = 0
    for line in range(line_count):
        next_states: dict[tuple[int, int], _HorizontalPath] = {}
        for (used, start), path in states.items():
            if used != line:
                continue
            remaining_lines = line_count - line - 1
            for end in range(start + 1, count - remaining_lines + 1):
                boundary = boundary_by_split.get(end) if end < count else None
                if end < count and not bool((boundary or {}).get("allowed")):
                    continue
                evaluated += 1
                width = max(0.0, prefix[end] - prefix[start])
                overflow = max(0.0, width - max_width)
                segment = items[start:end]
                lexical = _boundary_lexical_penalty(boundary)
                punctuation = _horizontal_whitespace_penalty(segment)
                if end < count:
                    punctuation += _break_strength_penalty(boundary)
                raggedness = max(0.0, max_width - min(max_width, width))
                candidate = _HorizontalPath(
                    breaks=(*path.breaks, end),
                    width_overflow_count=path.width_overflow_count + int(overflow > 1e-9),
                    width_overflow_amount=path.width_overflow_amount + overflow,
                    lexical_penalty=path.lexical_penalty + lexical,
                    punctuation_penalty=path.punctuation_penalty + punctuation,
                    raggedness=path.raggedness + raggedness,
                )
                key = (line + 1, end)
                prior = next_states.get(key)
                if prior is None or _horizontal_path_key(candidate) < _horizontal_path_key(prior):
                    next_states[key] = candidate
        states = next_states
        if not states:
            return None, evaluated
    return states.get((line_count, count)), evaluated


def _horizontal_path_key(path: _HorizontalPath) -> tuple[Any, ...]:
    return (
        path.width_overflow_count,
        round(path.width_overflow_amount, 6),
        round(path.lexical_penalty, 6),
        round(path.punctuation_penalty, 6),
        round(path.raggedness, 6),
        path.breaks,
    )


def _selected_break_record(boundary: Mapping[str, Any], split: int, ordinal: int) -> dict[str, Any]:
    return {
        **dict(boundary),
        "split_after_item_index": int(split),
        "selected_break_index": int(ordinal),
        "selection_authority": "explicit_break_opportunities",
    }


def _groups_from_breaks(items: Sequence[Any], breaks: Sequence[int]) -> list[list[Any]]:
    groups: list[list[Any]] = []
    start = 0
    for end in breaks:
        groups.append(list(items[start:end]))
        start = end
    return groups


def _segment_punctuation_penalty(
    items: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> float:
    segment = list(items[start:end])
    if not segment:
        return 1000.0
    penalty = sum(float(item.get("penalty") or 0.0) for item in _segment_quality_adjustments(segment))
    if _segment_is_punctuation_only(segment) and not _segment_is_punctuation_only(items):
        penalty += 45.0
    return penalty


def _segment_optical_penalty(
    items: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> float:
    segment = list(items[start:end])
    if len(segment) == 1 and len(items) > 3:
        return 9.0
    return 0.0


def _boundary_lexical_penalty(boundary: Mapping[str, Any] | None) -> float:
    metadata = dict((boundary or {}).get("opportunity_metadata") or {})
    try:
        return max(0.0, float(metadata.get("lexical_integrity_penalty") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _boundary_punctuation_penalty(
    items: Sequence[Mapping[str, Any]],
    split: int,
) -> float:
    if split <= 0 or split >= len(items):
        return 0.0
    previous = str(items[split - 1].get("text") or "")
    following = str(items[split].get("text") or "")
    if _is_strong_phrase_end(previous) and not _is_continuation_punctuation(following):
        return -8.0
    return 0.0


def _segment_quality_adjustments(segment: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if _segment_has_terminal_cjk_widow_before_punctuation_tail(segment):
        return [{"reason": "terminal_cjk_widow_before_punctuation_tail", "penalty": 12.0}]
    return []


def _partition_quality_adjustments(groups: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    start = 0
    for column, group in enumerate(groups):
        for item in _segment_quality_adjustments(group):
            records.append(
                {
                    **item,
                    "column_index": column,
                    "start": start,
                    "end": start + len(group),
                    "text": "".join(str(value.get("text") or "") for value in group),
                }
            )
        start += len(group)
    return records


def _segment_has_terminal_cjk_widow_before_punctuation_tail(segment: Sequence[Mapping[str, Any]]) -> bool:
    values = [item for item in segment if str(item.get("text") or "")]
    if len(values) < 2:
        return False
    tail_count = 0
    for item in reversed(values):
        text = str(item.get("text") or "")
        if _is_sequence_punctuation(text) or _is_continuation_punctuation(text):
            tail_count += 1
            continue
        break
    if tail_count <= 0 or tail_count >= len(values):
        return False
    content = values[:-tail_count]
    lexical = [
        item
        for item in content
        if not _is_sequence_punctuation(str(item.get("text") or ""))
        and not _is_continuation_punctuation(str(item.get("text") or ""))
        and not _is_open_punctuation(str(item.get("text") or ""))
    ]
    return len(lexical) == 1 and _is_cjk(str(lexical[0].get("text") or ""))


def _vertical_structure_penalty(
    profile: Mapping[str, Any],
    desired: int,
    columns: int,
    items: Sequence[Mapping[str, Any]],
    split_points: Sequence[int],
) -> float:
    if not _profile_needs_speech_column_conservation(profile) or columns <= desired:
        return 0.0
    return sum(
        max(0.0, _boundary_punctuation_penalty(items, int(split)))
        for split in split_points
    )


def _profile_needs_speech_column_conservation(profile: Mapping[str, Any]) -> bool:
    semantic = str(profile.get("semantic_class") or "").lower()
    role = str(profile.get("source_role") or "").lower()
    return "speech" in semantic or "bubble" in semantic or role in {"speech", "speech_bubble", "dialogue"}


def _reliable_source_distribution(profile: Mapping[str, Any], columns: int) -> list[float]:
    raw = profile.get("source_column_row_distribution")
    try:
        confidence = float(profile.get("source_column_distribution_confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.75 or not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    try:
        values = [max(0.0, float(item)) for item in raw]
    except (TypeError, ValueError):
        return []
    return values if len(values) == columns and any(values) else []


def _break_strength_penalty(boundary: Mapping[str, Any] | None) -> float:
    metadata = dict((boundary or {}).get("opportunity_metadata") or {})
    if metadata.get("target_lexical_boundary") in {"intra_span", "between_spans"}:
        return 0.0
    strength = str((boundary or {}).get("strength") or "normal")
    return {"preferred": 0.0, "normal": 1.0, "weak": 2.0}.get(strength, 3.0)


def _horizontal_whitespace_penalty(segment: Sequence[Mapping[str, Any]]) -> float:
    texts = [str(item.get("text") or "") for item in segment]
    if texts and all(text.isspace() for text in texts):
        return 50.0
    penalty = 0.0
    if texts and texts[0].isspace():
        penalty += 2.0
    return penalty


def _item_row_units(item: Mapping[str, Any]) -> float:
    try:
        return max(1.0, float(item.get("row_units", 1.0)))
    except (TypeError, ValueError):
        return 1.0


def _items_row_units(items: Sequence[Mapping[str, Any]]) -> float:
    return sum(_item_row_units(item) for item in items)


def _items_advance(items: Sequence[Mapping[str, Any]]) -> float:
    return sum(max(0.0, float(item.get("advance") or 0.0)) for item in items)


def _is_open_punctuation(text: str) -> bool:
    return str(text or "")[:1] in _OPEN_PUNCTUATION


def _is_continuation_punctuation(text: str) -> bool:
    return str(text or "")[:1] in _CONTINUATION_PUNCTUATION


def _is_sequence_punctuation(text: str) -> bool:
    return str(text or "")[:1] in _SEQUENCE_PUNCTUATION


def _is_strong_phrase_end(text: str) -> bool:
    return str(text or "")[:1] in _STRONG_PHRASE_END


def _is_cjk(text: str) -> bool:
    return bool(text) and classify_grapheme(str(text)[0]) == "cjk"


def _segment_is_punctuation_only(items: Sequence[Mapping[str, Any]]) -> bool:
    values = [str(item.get("text") or "") for item in items if str(item.get("text") or "")]
    return bool(values) and all(
        _is_continuation_punctuation(text)
        or _is_open_punctuation(text)
        or _is_sequence_punctuation(text)
        or _is_strong_phrase_end(text)
        for text in values
    )


def _unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result
