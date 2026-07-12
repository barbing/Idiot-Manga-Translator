# -*- coding: utf-8 -*-
"""Text segmentation and writing-mode normalization for Stage 4 layout.

This module is intentionally inert: it does not draw, inspect page pixels, or
make parent ownership decisions. It only prepares text evidence for the
TypesettingEngine.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Sequence

try:
    import regex as _regex
except Exception:  # pragma: no cover - dependency is declared for Stage 4
    _regex = None

try:
    from bidi.algorithm import get_display as _bidi_get_display
except Exception:  # pragma: no cover - optional until dependency install
    _bidi_get_display = None


SYMBOL_CHARS = {"☆", "★", "♡", "❤", "♪"}
LTR_COMPLEX_SCRIPTS = {"Deva", "Beng", "Guru", "Gujr", "Orya", "Taml", "Telu", "Knda", "Mlym", "Sinh", "Thai"}
ELLIPSIS_CHARS = {"…", "︙"}
DASH_CHARS = {"-", "―", "—", "─", "︱", "ー"}
_TYPESETTING_DASH_CHARS = DASH_CHARS.difference({"ー"})
WAVE_DASH_CHARS = {"~", "～", "〜", "〰", "︴"}
COMPACT_VERTICAL_PUNCTUATION_CHARS = {"!", "?", "！", "？", "︕", "︖", "‼", "⁇", "⁉", "⁈"}
VERTICAL_CENTERED_PUNCTUATION_CHARS = {
    ",",
    ".",
    ":",
    ";",
    "，",
    "、",
    "。",
    "：",
    "；",
    "︐",
    "︑",
    "︒",
    "︓",
    "︔",
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
    "︱",
    "︴",
}
OPEN_PUNCTUATION = {"(", "（", "[", "［", "{", "｛", "「", "『", "【", "〈", "《", "“", "‘"}
CLOSE_PUNCTUATION = {
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
class InlineTextRun:
    run_id: str
    text: str
    normalized_text: str
    grapheme_start: int
    grapheme_end: int
    script: str
    direction: str
    language: str
    role: str
    break_class: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "grapheme_start": int(self.grapheme_start),
            "grapheme_end": int(self.grapheme_end),
            "script": self.script,
            "direction": self.direction,
            "language": self.language,
            "role": self.role,
            "break_class": self.break_class,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BreakOpportunity:
    before_run_id: str
    after_run_id: str
    position: int
    strength: str
    reason: str
    allowed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "before_run_id": self.before_run_id,
            "after_run_id": self.after_run_id,
            "position": int(self.position),
            "strength": self.strength,
            "reason": self.reason,
            "allowed": bool(self.allowed),
            "metadata": dict(self.metadata),
        }


def grapheme_clusters(text: str) -> list[str]:
    value = str(text or "")
    if not value:
        return []
    if _regex is not None:
        return [cluster for cluster in _regex.findall(r"\X", value) if cluster]
    return [char for char in value if char]


def normalize_for_writing_mode(text: str, writing_mode: str, font_manager, face) -> tuple[str, list[dict], list[dict], list[dict]]:
    value = str(text or "")
    mode = str(writing_mode or "").lower()
    punctuation: list[dict[str, Any]] = []
    symbols: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    if not mode.startswith("vert"):
        out: list[str] = []
        source_index = 0
        normalized_grapheme_index = 0
        while source_index < len(value):
            ellipsis_source = _match_ellipsis_source_run(value, source_index)
            if ellipsis_source:
                normalized = _canonical_ellipsis_text(ellipsis_source)
                punctuation.append(
                    _punctuation_occurrence(
                        ellipsis_source,
                        normalized,
                        source_index=source_index,
                        normalized_grapheme_start=normalized_grapheme_index,
                        occurrence_index=len(punctuation),
                        supported=_symbol_supported(font_manager, face, normalized),
                        candidate_forms=[normalized],
                        supported_forms=[normalized] if _symbol_supported(font_manager, face, normalized) else [],
                        writing_mode="horizontal",
                    )
                )
                out.append(normalized)
                normalized_grapheme_index += len(grapheme_clusters(normalized))
                source_index += len(ellipsis_source)
                continue
            cluster = grapheme_clusters(value[source_index:])[0]
            if cluster in SYMBOL_CHARS:
                symbols.append(
                    _symbol_occurrence(
                        cluster,
                        source_index=source_index,
                        normalized_grapheme_start=normalized_grapheme_index,
                        occurrence_index=len(symbols),
                        font_manager=font_manager,
                        face=face,
                        writing_mode="horizontal",
                    )
                )
            elif _is_punctuation_cluster(cluster):
                punctuation.append(
                    _punctuation_occurrence(
                        cluster,
                        cluster,
                        source_index=source_index,
                        normalized_grapheme_start=normalized_grapheme_index,
                        occurrence_index=len(punctuation),
                        supported=True,
                        candidate_forms=[cluster],
                        supported_forms=[cluster],
                        writing_mode="horizontal",
                    )
                )
            out.append(cluster)
            source_index += len(cluster)
            normalized_grapheme_index += 1
        return "".join(out), punctuation, symbols, notes

    replacements = (
        "！！",
        "!!",
        "？？",
        "??",
        "！？",
        "!?",
        "？！",
        "?!",
        "——",
        "--",
        "—",
        "―",
        "－",
        "-",
        "～",
        "〜",
        "〰",
        "~",
        "…",
        "！",
        "!",
        "？",
        "?",
        "，",
        ",",
        "、",
        "。",
        "：",
        ":",
        "；",
        ";",
        "．",
        ".",
    )
    out: list[str] = []
    index = 0
    normalized_grapheme_index = 0
    while index < len(value):
        cluster = grapheme_clusters(value[index:])[0]
        if cluster.isspace():
            notes.append(
                {
                    "type": "vertical_space_collapsed",
                    "source": cluster,
                    "source_index": index,
                    "reason": "layout_insignificant_vertical_space",
                }
            )
            index += len(cluster)
            continue
        ellipsis_source = _match_ellipsis_source_run(value, index)
        if ellipsis_source:
            canonical = _canonical_ellipsis_text(ellipsis_source)
            support = font_manager.vertical_punctuation_support(face, canonical)
            normalized = support.selected_form or canonical
            punctuation.append(
                _punctuation_occurrence(
                    ellipsis_source,
                    normalized,
                    source_index=index,
                    normalized_grapheme_start=normalized_grapheme_index,
                    occurrence_index=len(punctuation),
                    supported=bool(support.supported),
                    candidate_forms=list(support.candidate_forms),
                    supported_forms=list(support.supported_forms),
                    writing_mode="vertical",
                )
            )
            out.append(normalized)
            normalized_grapheme_index += len(grapheme_clusters(normalized))
            index += len(ellipsis_source)
            continue
        matched = ""
        for source in replacements:
            if value.startswith(source, index):
                matched = source
                break
        if matched:
            support = font_manager.vertical_punctuation_support(face, matched)
            normalized = support.selected_form or matched
            punctuation.append(
                _punctuation_occurrence(
                    matched,
                    normalized,
                    source_index=index,
                    normalized_grapheme_start=normalized_grapheme_index,
                    occurrence_index=len(punctuation),
                    supported=bool(support.supported),
                    candidate_forms=list(support.candidate_forms),
                    supported_forms=list(support.supported_forms),
                    writing_mode="vertical",
                )
            )
            out.append(normalized)
            normalized_grapheme_index += len(grapheme_clusters(normalized))
            index += len(matched)
            continue
        if cluster in SYMBOL_CHARS:
            symbols.append(
                _symbol_occurrence(
                    cluster,
                    source_index=index,
                    normalized_grapheme_start=normalized_grapheme_index,
                    occurrence_index=len(symbols),
                    font_manager=font_manager,
                    face=face,
                    writing_mode="vertical",
                )
            )
        elif _is_punctuation_cluster(cluster):
            punctuation.append(
                _punctuation_occurrence(
                    cluster,
                    cluster,
                    source_index=index,
                    normalized_grapheme_start=normalized_grapheme_index,
                    occurrence_index=len(punctuation),
                    supported=_symbol_supported(font_manager, face, cluster),
                    candidate_forms=[cluster],
                    supported_forms=[cluster] if _symbol_supported(font_manager, face, cluster) else [],
                    writing_mode="vertical",
                )
            )
        out.append(cluster)
        normalized_grapheme_index += 1
        index += len(cluster)
    return "".join(out), punctuation, symbols, notes


def classify_grapheme(text: str) -> str:
    cluster = str(text or "")
    if not cluster:
        return "empty"
    if cluster.isspace():
        return "space"
    if cluster in SYMBOL_CHARS or _is_emoji(cluster):
        return "symbol"
    if _is_ellipsis_sequence(cluster):
        return "ellipsis"
    if _is_dash_sequence(cluster):
        return "dash"
    if _is_wave_sequence(cluster):
        return "wave"
    if all(_is_latin_char(char) for char in cluster):
        return "latin"
    if all(char.isdigit() for char in cluster):
        return "number"
    if any(_script_for_char(char) in {"Arab", "Hebr"} for char in cluster):
        return "rtl"
    if any(_script_for_char(char) in LTR_COMPLEX_SCRIPTS for char in cluster):
        return "complex"
    script = _script_for_char(cluster[0])
    if script in {"Hani", "Hira", "Kana", "Hang"}:
        return "cjk"
    if cluster in OPEN_PUNCTUATION:
        return "open_punctuation"
    if cluster in CLOSE_PUNCTUATION:
        return "close_punctuation"
    cat = unicodedata.category(cluster[0])
    if cat.startswith("P"):
        return "punctuation"
    if cat.startswith("M"):
        return "mark"
    return "other"


def segment_inline_runs(
    text: str,
    *,
    writing_mode: str,
    language_hint: str = "",
    punctuation_occurrences: Sequence[dict[str, Any]] | None = None,
    symbol_occurrences: Sequence[dict[str, Any]] | None = None,
) -> list[InlineTextRun]:
    clusters = grapheme_clusters(text)
    runs: list[InlineTextRun] = []
    index = 0
    while index < len(clusters):
        cluster = clusters[index]
        kind = classify_grapheme(cluster)
        start = index
        group = [cluster]
        text_value = "".join(group)
        if kind in {"latin", "number"}:
            while index + 1 < len(clusters):
                next_kind = classify_grapheme(clusters[index + 1])
                if next_kind != kind:
                    break
                group.append(clusters[index + 1])
                index += 1
        elif kind in {"ellipsis", "dash", "wave"}:
            while index + 1 < len(clusters) and classify_grapheme(clusters[index + 1]) == kind:
                group.append(clusters[index + 1])
                index += 1
        elif _is_compact_vertical_punctuation_char(cluster):
            while index + 1 < len(clusters) and _is_compact_vertical_punctuation_char(clusters[index + 1]):
                group.append(clusters[index + 1])
                index += 1
        elif kind in {"rtl", "complex"}:
            script = _script_for_run(text_value, kind)
            while index + 1 < len(clusters):
                next_kind = classify_grapheme(clusters[index + 1])
                next_script = _script_for_run(clusters[index + 1], next_kind)
                if next_kind != kind or next_script != script:
                    break
                group.append(clusters[index + 1])
                text_value = "".join(group)
                index += 1
        text_value = "".join(group)
        script = _script_for_run(text_value, kind)
        direction = _direction_for_script(script)
        role = _role_for_kind(kind, text_value)
        metadata: dict[str, Any] = {}
        if role == "latin_word":
            metadata["letter_stacking_allowed"] = False
        if direction == "rtl" and _bidi_get_display is not None:
            metadata["bidi_visual_text"] = _bidi_get_display(text_value)
        punctuation_evidence = _occurrences_for_span(
            punctuation_occurrences,
            start,
            index + 1,
        )
        symbol_evidence = _occurrences_for_span(
            symbol_occurrences,
            start,
            index + 1,
        )
        if punctuation_evidence:
            metadata["punctuation_occurrences"] = punctuation_evidence
        if symbol_evidence:
            metadata["symbol_occurrences"] = symbol_evidence
        runs.append(
            InlineTextRun(
                run_id=f"run_{len(runs):04d}",
                text=text_value,
                normalized_text=text_value,
                grapheme_start=start,
                grapheme_end=index + 1,
                script=script,
                direction=direction,
                language=_language_for_script(script, language_hint),
                role=role,
                break_class=kind,
                metadata=metadata,
            )
        )
        index += 1
    return runs


def compute_break_opportunities(runs: Sequence[InlineTextRun], *, writing_mode: str) -> list[BreakOpportunity]:
    items = list(runs or [])
    opportunities: list[BreakOpportunity] = []
    for index in range(1, len(items)):
        before = items[index - 1]
        after = items[index]
        reason = "generic_run_boundary"
        strength = "normal"
        allowed = True
        if after.text in CLOSE_PUNCTUATION or after.role in {"ellipsis_sequence", "dash_sequence", "punctuation_sequence"} and after.text[:1] in CLOSE_PUNCTUATION:
            reason = "kinsoku_rejected_before_closing_punctuation"
            strength = "forbidden"
            allowed = False
        elif before.text in OPEN_PUNCTUATION:
            reason = "kinsoku_rejected_after_opening_punctuation"
            strength = "forbidden"
            allowed = False
        elif before.break_class == "space" or after.break_class == "space":
            reason = "space_word_boundary"
            strength = "preferred"
        elif before.script in {"Hani", "Hira", "Kana", "Hang"} and after.script in {"Hani", "Hira", "Kana", "Hang"}:
            reason = "cjk_grapheme_boundary"
            strength = "normal"
        elif before.script == "Latn" and after.script == "Latn":
            reason = "latin_run_boundary"
            strength = "weak"
        opportunities.append(
            BreakOpportunity(
                before_run_id=before.run_id,
                after_run_id=after.run_id,
                position=after.grapheme_start,
                strength=strength,
                reason=reason,
                allowed=allowed,
                metadata={
                    "writing_mode": writing_mode,
                    "before_text": before.text,
                    "after_text": after.text,
                },
            )
        )
    return opportunities


def _symbol_supported(font_manager, face, symbol: str) -> bool:
    try:
        return bool(font_manager.coverage_for_text(face, symbol).supports_text)
    except Exception:
        return False


def _symbol_occurrence(
    symbol: str,
    *,
    source_index: int,
    normalized_grapheme_start: int,
    occurrence_index: int,
    font_manager,
    face,
    writing_mode: str,
) -> dict[str, Any]:
    supported = _symbol_supported(font_manager, face, symbol)
    return {
        "occurrence_id": f"symbol_{occurrence_index:04d}",
        "symbol": symbol,
        "source": symbol,
        "normalized": symbol,
        "source_codepoints": _codepoints(symbol),
        "normalized_codepoints": _codepoints(symbol),
        "source_start": int(source_index),
        "source_end": int(source_index + len(symbol)),
        "normalized_grapheme_start": int(normalized_grapheme_start),
        "normalized_grapheme_end": int(normalized_grapheme_start + 1),
        "writing_mode": writing_mode,
        "kind": "symbol",
        "source_class": _symbol_class(symbol),
        "orientation_policy": "upright_resolved_face" if writing_mode == "vertical" else "horizontal",
        "render_policy": "resolved_font_glyph",
        "supported": bool(supported),
    }


def _punctuation_occurrence(
    source: str,
    normalized: str,
    *,
    source_index: int,
    normalized_grapheme_start: int,
    occurrence_index: int,
    supported: bool,
    candidate_forms: Sequence[str],
    supported_forms: Sequence[str],
    writing_mode: str,
) -> dict[str, Any]:
    kind = _punctuation_kind(source)
    normalized_count = len(grapheme_clusters(normalized))
    unit_count = _punctuation_unit_count(source, normalized, kind)
    record = {
        "occurrence_id": f"punctuation_{occurrence_index:04d}",
        "source": source,
        "normalized": normalized,
        "source_codepoints": _codepoints(source),
        "normalized_codepoints": _codepoints(normalized),
        "source_start": int(source_index),
        "source_end": int(source_index + len(source)),
        "normalized_grapheme_start": int(normalized_grapheme_start),
        "normalized_grapheme_end": int(normalized_grapheme_start + normalized_count),
        "writing_mode": writing_mode,
        "kind": kind,
        "source_class": _punctuation_source_class(source, kind),
        "orientation_policy": _punctuation_orientation_policy(kind, writing_mode),
        "render_policy": _punctuation_render_policy(kind, writing_mode),
        "unit_count": unit_count,
        "supported": bool(supported),
        "candidate_forms": list(candidate_forms),
        "supported_forms": list(supported_forms),
    }
    if kind == "ellipsis":
        record["dot_count"] = int(unit_count * 3)
        record["sequence_group_count"] = 1
    return record


def _occurrences_for_span(
    occurrences: Sequence[dict[str, Any]] | None,
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in list(occurrences or []):
        if not isinstance(item, dict):
            continue
        item_start = int(item.get("normalized_grapheme_start") or 0)
        item_end = int(item.get("normalized_grapheme_end") or item_start)
        if item_start < end and item_end > start:
            result.append(dict(item))
    return result


def _is_punctuation_cluster(cluster: str) -> bool:
    return bool(cluster) and (
        classify_grapheme(cluster) in {
            "ellipsis",
            "dash",
            "wave",
            "punctuation",
            "open_punctuation",
            "close_punctuation",
        }
        or unicodedata.category(cluster[0]).startswith("P")
    )


def _punctuation_kind(source: str) -> str:
    if _ellipsis_source_unit_count(source) > 0:
        return "ellipsis"
    if _is_dash_sequence(source):
        return "dash"
    if _is_wave_sequence(source):
        return "wave"
    if source and all(char in COMPACT_VERTICAL_PUNCTUATION_CHARS for char in source):
        return "emphasis_punctuation"
    if source in OPEN_PUNCTUATION:
        return "open_punctuation"
    if source in CLOSE_PUNCTUATION:
        return "close_punctuation"
    return "punctuation"


def _punctuation_source_class(source: str, kind: str) -> str:
    if kind == "wave":
        return {
            "~": "ascii_tilde",
            "～": "fullwidth_tilde",
            "〜": "wave_dash",
            "〰": "wavy_dash",
            "︴": "vertical_wavy_line",
        }.get(source[:1], "mixed_wave_sequence")
    if kind == "ellipsis":
        if source and all(char == "." for char in source):
            units = _ellipsis_source_unit_count(source)
            if units == 1:
                return "ascii_three_dot_ellipsis"
            if units == 2:
                return "ascii_six_dot_ellipsis"
            return "ascii_multi_unit_ellipsis"
        if _is_ellipsis_sequence(source):
            return "unicode_ellipsis"
        return "mixed_ellipsis_sequence"
    if kind == "dash":
        return "dash_sequence"
    if kind in {"open_punctuation", "close_punctuation"}:
        return kind
    return "punctuation"


def _punctuation_orientation_policy(kind: str, writing_mode: str) -> str:
    if writing_mode != "vertical":
        return "horizontal"
    if kind in {"ellipsis", "wave", "dash"}:
        return "vertical_inline_axis"
    if kind in {"open_punctuation", "close_punctuation"}:
        return "font_vertical_alternate"
    return "upright_vertical_form"


def _punctuation_render_policy(kind: str, writing_mode: str) -> str:
    if writing_mode != "vertical":
        return "resolved_font_glyph"
    if kind == "ellipsis":
        return "vertical_ellipsis_primitive"
    if kind == "wave":
        return "vertical_wave_primitive"
    if kind == "dash":
        return "vertical_dash_primitive"
    if kind in {"open_punctuation", "close_punctuation"}:
        return "font_vertical_alternate"
    return "resolved_vertical_form_glyph"


def _punctuation_unit_count(source: str, normalized: str, kind: str) -> int:
    if kind == "ellipsis":
        source_units = _ellipsis_source_unit_count(source)
        if source_units > 0:
            return source_units
        return max(1, len([char for char in normalized if char in ELLIPSIS_CHARS]))
    if kind in {"wave", "dash"}:
        return max(1, len(grapheme_clusters(normalized)))
    return 1


def _codepoints(text: str) -> list[str]:
    return [f"U+{ord(char):04X}" for char in str(text or "")]


def _symbol_class(symbol: str) -> str:
    return {
        "☆": "white_star",
        "★": "black_star",
        "♡": "white_heart",
        "❤": "heart",
        "♪": "music_note",
    }.get(symbol, "symbol")


def _is_latin_char(char: str) -> bool:
    if not char:
        return False
    codepoint = ord(char)
    return (0x41 <= codepoint <= 0x5A) or (0x61 <= codepoint <= 0x7A)


def _is_emoji(cluster: str) -> bool:
    return any(ord(char) >= 0x1F000 for char in cluster)


def _is_ellipsis_sequence(text: str) -> bool:
    return bool(text) and all(char in ELLIPSIS_CHARS for char in text)


def _ellipsis_source_unit_count(text: str) -> int:
    value = str(text or "")
    if not value:
        return 0
    units = 0
    index = 0
    while index < len(value):
        char = value[index]
        if char in ELLIPSIS_CHARS:
            units += 1
            index += 1
            continue
        if value.startswith("...", index):
            units += 1
            index += 3
            continue
        return 0
    return units


def _canonical_ellipsis_text(source: str) -> str:
    return "…" * max(1, _ellipsis_source_unit_count(source))


def _match_ellipsis_source_run(text: str, start: int) -> str:
    value = str(text or "")
    index = max(0, int(start))
    end = index
    units = 0
    while end < len(value):
        if value[end] in ELLIPSIS_CHARS:
            units += 1
            end += 1
            continue
        if value[end] == ".":
            dot_end = end
            while dot_end < len(value) and value[dot_end] == ".":
                dot_end += 1
            complete_length = ((dot_end - end) // 3) * 3
            if complete_length <= 0:
                break
            units += complete_length // 3
            end += complete_length
            if end < dot_end:
                break
            continue
        break
    if units <= 0:
        return ""
    return value[index:end]


def _is_dash_sequence(text: str) -> bool:
    return bool(text) and all(char in _TYPESETTING_DASH_CHARS for char in text)


def _is_wave_sequence(text: str) -> bool:
    return bool(text) and all(char in WAVE_DASH_CHARS for char in text)


def _is_compact_vertical_punctuation_char(text: str) -> bool:
    return bool(text) and text in COMPACT_VERTICAL_PUNCTUATION_CHARS


def _is_compact_vertical_punctuation_sequence(text: str) -> bool:
    clusters = grapheme_clusters(text)
    return len(clusters) > 1 and all(_is_compact_vertical_punctuation_char(cluster) for cluster in clusters)


def _script_for_run(text: str, kind: str) -> str:
    if kind == "latin":
        return "Latn"
    if kind == "number":
        return "Zyyy"
    if kind == "rtl":
        return _script_for_char(text[0])
    if kind == "complex":
        for char in text:
            script = _script_for_char(char)
            if script in LTR_COMPLEX_SCRIPTS:
                return script
    if kind in {"symbol", "ellipsis", "dash", "wave", "punctuation", "open_punctuation", "close_punctuation", "space"}:
        return "Zyyy"
    for char in text:
        script = _script_for_char(char)
        if script != "Zyyy":
            return script
    return "Zyyy"


def _script_for_char(char: str) -> str:
    codepoint = ord(char)
    if 0x4E00 <= codepoint <= 0x9FFF or 0x3400 <= codepoint <= 0x4DBF:
        return "Hani"
    if 0x3040 <= codepoint <= 0x309F:
        return "Hira"
    if 0x30A0 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF:
        return "Kana"
    if 0xAC00 <= codepoint <= 0xD7AF:
        return "Hang"
    if 0x0600 <= codepoint <= 0x06FF:
        return "Arab"
    if 0x0590 <= codepoint <= 0x05FF:
        return "Hebr"
    if 0x0900 <= codepoint <= 0x097F:
        return "Deva"
    if 0x0980 <= codepoint <= 0x09FF:
        return "Beng"
    if 0x0A00 <= codepoint <= 0x0A7F:
        return "Guru"
    if 0x0A80 <= codepoint <= 0x0AFF:
        return "Gujr"
    if 0x0B00 <= codepoint <= 0x0B7F:
        return "Orya"
    if 0x0B80 <= codepoint <= 0x0BFF:
        return "Taml"
    if 0x0C00 <= codepoint <= 0x0C7F:
        return "Telu"
    if 0x0C80 <= codepoint <= 0x0CFF:
        return "Knda"
    if 0x0D00 <= codepoint <= 0x0D7F:
        return "Mlym"
    if 0x0D80 <= codepoint <= 0x0DFF:
        return "Sinh"
    if 0x0E00 <= codepoint <= 0x0E7F:
        return "Thai"
    if _is_latin_char(char):
        return "Latn"
    return "Zyyy"


def _direction_for_script(script: str) -> str:
    if script in {"Arab", "Hebr"}:
        return "rtl"
    return "ltr"


def _language_for_script(script: str, language_hint: str) -> str:
    hint = str(language_hint or "").strip()
    if hint:
        return hint
    if script == "Arab":
        return "ar"
    if script == "Hebr":
        return "he"
    if script == "Deva":
        return "hi"
    if script == "Thai":
        return "th"
    if script in {"Hani", "Hira", "Kana"}:
        return "zh"
    if script == "Hang":
        return "ko"
    if script == "Latn":
        return "en"
    return ""


def _role_for_kind(kind: str, text: str) -> str:
    if kind == "latin":
        return "latin_word"
    if kind == "number":
        return "numeric_token"
    if kind == "ellipsis":
        return "ellipsis_sequence"
    if kind == "dash":
        return "dash_sequence"
    if kind == "wave":
        return "wave_sequence"
    if _is_compact_vertical_punctuation_sequence(text):
        return "punctuation_sequence"
    if kind == "symbol":
        return "symbol"
    if kind == "space":
        return "space"
    if kind == "rtl":
        return "complex_script"
    if kind == "complex":
        return "complex_script"
    if kind == "cjk":
        return "cjk_grapheme"
    if "punctuation" in kind:
        return kind
    return classify_grapheme(text)
