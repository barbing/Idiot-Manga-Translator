# -*- coding: utf-8 -*-
"""Text segmentation and writing-mode normalization for Stage 4 layout.

This module is intentionally inert: it does not draw, inspect page pixels, or
make parent ownership decisions. It only prepares text evidence for the
TypesettingEngine.
"""
from __future__ import annotations

import threading
import unicodedata
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from app.render.typesetting_contracts import PunctuationToken, TextToken

if TYPE_CHECKING:
    from app.render.target_lexical_segmenter import (
        TargetLexicalBoundary,
        TargetLexicalSpan,
    )

try:
    import regex as _regex
except Exception:  # pragma: no cover - dependency is declared for Stage 4
    _regex = None

_DEFAULT_IGNORABLE_RE = (
    _regex.compile(r"\A\p{Default_Ignorable_Code_Point}\Z")
    if _regex is not None
    else None
)
_LATIN_SCRIPT_RE = (
    _regex.compile(r"\A\p{Script=Latin}\Z")
    if _regex is not None
    else None
)
_EXTENDED_PICTOGRAPHIC_RE = (
    _regex.compile(r"\p{Extended_Pictographic}")
    if _regex is not None
    else None
)
_REGIONAL_INDICATOR_SEQUENCE_RE = (
    _regex.compile(r"\A\p{Regional_Indicator}{2,}\Z")
    if _regex is not None
    else None
)
_EMOJI_KEYCAP_RE = (
    _regex.compile(r"\A[0-9#*]\ufe0f?\u20e3\Z")
    if _regex is not None
    else None
)

try:
    from bidi.algorithm import get_display as _bidi_get_display
except Exception:  # pragma: no cover - optional until dependency install
    _bidi_get_display = None


SYMBOL_CHARS = {"☆", "★", "♡", "❤", "♪"}
LTR_COMPLEX_SCRIPTS = {"Deva", "Beng", "Guru", "Gujr", "Orya", "Taml", "Telu", "Knda", "Mlym", "Sinh", "Thai"}
ELLIPSIS_DOT_WEIGHTS = {
    "…": 3,
    "︙": 3,
    "⋯": 3,
    "‥": 2,
    "︰": 2,
}
ELLIPSIS_CHARS = set(ELLIPSIS_DOT_WEIGHTS)
MIDDLE_DOT_CHARS = {"・", "･"}
DASH_CHARS = {"-", "―", "—", "─", "︱", "ー"}
_TYPESETTING_DASH_CHARS = DASH_CHARS.difference({"ー"})
WAVE_DASH_CHARS = {"~", "～", "〜", "〰", "︴"}
COMPACT_VERTICAL_PUNCTUATION_CHARS = {"!", "?", "！", "？", "︕", "︖", "‼", "⁇", "⁉", "⁈"}
EMPHASIS_PUNCTUATION_UNIT_COUNTS = {
    "!": 1,
    "?": 1,
    "！": 1,
    "？": 1,
    "︕": 1,
    "︖": 1,
    "‼": 2,
    "⁇": 2,
    "⁉": 2,
    "⁈": 2,
}
EMPHASIS_PUNCTUATION_EXPANSIONS = {
    "!": "!",
    "?": "?",
    "！": "!",
    "？": "?",
    "︕": "!",
    "︖": "?",
    "‼": "!!",
    "⁇": "??",
    "⁉": "!?",
    "⁈": "?!",
}
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
    original_text: str = ""
    translated_start: int = 0
    translated_end: int = 0
    token_start: int = 0
    token_end: int = 0
    token_ids: tuple[str, ...] = ()
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
            "original_text": self.original_text,
            "translated_start": int(self.translated_start),
            "translated_end": int(self.translated_end),
            "token_start": int(self.token_start),
            "token_end": int(self.token_end),
            "token_ids": list(self.token_ids),
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


def strict_grapheme_clusters(text: str) -> list[str]:
    """Return UAX #29 extended grapheme clusters or fail closed.

    The compatibility helper above retains its historical codepoint fallback.
    Font-span construction cannot use that fallback because it would permit a
    combining sequence, variation sequence, or ZWJ sequence to be split across
    faces.
    """

    value = str(text or "")
    if not value:
        return []
    if _regex is None:
        raise RuntimeError("uax29_grapheme_segmenter_unavailable")
    return [cluster for cluster in _regex.findall(r"\X", value) if cluster]


def is_default_ignorable_codepoint(char: str) -> bool:
    """Return the Unicode Default_Ignorable_Code_Point property."""

    value = str(char or "")
    if len(value) != 1:
        return False
    if _DEFAULT_IGNORABLE_RE is not None:
        return bool(_DEFAULT_IGNORABLE_RE.fullmatch(value))
    codepoint = ord(value)
    if unicodedata.category(value) == "Cf":
        return True
    return (
        0xFE00 <= codepoint <= 0xFE0F
        or 0xE0100 <= codepoint <= 0xE01EF
        or 0xE0000 <= codepoint <= 0xE0FFF
        or codepoint in {0x034F, 0x061C, 0x115F, 0x1160, 0x17B4, 0x17B5}
        or 0x180B <= codepoint <= 0x180F
        or 0x1BCA0 <= codepoint <= 0x1BCA3
    )


def source_char_requires_visible_glyph(char: str) -> bool:
    value = str(char or "")
    return bool(
        len(value) == 1
        and not value.isspace()
        and not is_default_ignorable_codepoint(value)
    )


def source_text_requires_visible_glyph(text: str) -> bool:
    return any(source_char_requires_visible_glyph(char) for char in str(text or ""))


def is_emoji_grapheme_cluster(cluster: str) -> bool:
    """Return whether a cluster carries genuine emoji-presentation evidence.

    ZWJ and variation selectors are format controls used by many scripts.  They
    refine an emoji base but never establish emoji ownership by themselves.
    Plain BMP pictographs remain symbol-font candidates unless an emoji
    presentation selector or an emoji ZWJ sequence makes that intent explicit.
    """

    value = str(cluster or "")
    if not value:
        return False
    if _EMOJI_KEYCAP_RE is not None and _EMOJI_KEYCAP_RE.fullmatch(value):
        return True
    if _REGIONAL_INDICATOR_SEQUENCE_RE is not None and _REGIONAL_INDICATOR_SEQUENCE_RE.fullmatch(value):
        return True
    contains_extended_pictographic = bool(
        _EXTENDED_PICTOGRAPHIC_RE is not None
        and _EXTENDED_PICTOGRAPHIC_RE.search(value)
    )
    if any(0x1F000 <= ord(char) <= 0x1FAFF for char in value):
        return True
    if contains_extended_pictographic and ("\ufe0f" in value or "\u200d" in value):
        return True
    codepoints = [ord(char) for char in value]
    if (
        len(codepoints) in {2, 3}
        and value[0] in "0123456789#*"
        and codepoints[-1] == 0x20E3
    ):
        return True
    if len(codepoints) >= 2 and all(0x1F1E6 <= item <= 0x1F1FF for item in codepoints):
        return True
    return bool(
        any(0x1F000 <= codepoint <= 0x1FAFF for codepoint in codepoints)
        or (
            any(0x2600 <= codepoint <= 0x27BF for codepoint in codepoints)
            and ("\ufe0f" in value or "\u200d" in value)
        )
    )


def build_lossless_text_tokens(
    text: str,
) -> list[TextToken | PunctuationToken]:
    """Tokenize exact translated code points before presentation policy.

    Sequence recognition is lossless: every input code point belongs to one
    immutable token, and concatenating ``original_text`` reconstructs the
    exact translated input.
    """

    value = str(text or "")
    clusters = _grapheme_records(value)
    tokens: list[TextToken | PunctuationToken] = []
    punctuation_index = 0
    text_index = 0
    index = 0
    while index < len(clusters):
        end, punctuation_kind = _punctuation_sequence_at(clusters, index)
        if punctuation_kind:
            selected = clusters[index:end]
            original = "".join(item["text"] for item in selected)
            dot_count = _punctuation_dot_count(original, punctuation_kind)
            exact_multiplicity = _exact_punctuation_multiplicity(
                original,
                punctuation_kind,
            )
            unit_count = _presentation_unit_count(
                original,
                punctuation_kind,
                dot_count,
            )
            tokens.append(
                PunctuationToken(
                    token_id=f"punctuation_{punctuation_index:04d}",
                    original_text=original,
                    translated_start=int(selected[0]["start"]),
                    translated_end=int(selected[-1]["end"]),
                    grapheme_start=int(selected[0]["grapheme_index"]),
                    grapheme_end=int(selected[-1]["grapheme_index"] + 1),
                    kind="punctuation",
                    original_codepoints=tuple(_codepoints(original)),
                    writing_mode="unresolved",
                    presentation_text=original,
                    presentation_codepoints=tuple(_codepoints(original)),
                    presentation_grapheme_start=int(selected[0]["grapheme_index"]),
                    presentation_grapheme_end=int(selected[-1]["grapheme_index"] + 1),
                    atomic_break=True,
                    orientation_policy="unresolved",
                    render_policy="unresolved",
                    presentation_reason="identity_frozen_before_presentation",
                    punctuation_kind=punctuation_kind,
                    source_class=_lossless_punctuation_source_class(
                        original,
                        punctuation_kind,
                    ),
                    exact_multiplicity=exact_multiplicity,
                    unit_count=unit_count,
                    dot_count=dot_count,
                    sequence_group_count=1,
                    candidate_forms=(original,),
                    supported_forms=(),
                    supported=True,
                )
            )
            punctuation_index += 1
            index = end
            continue

        item = clusters[index]
        original = str(item["text"])
        tokens.append(
            TextToken(
                token_id=f"text_{text_index:04d}",
                original_text=original,
                translated_start=int(item["start"]),
                translated_end=int(item["end"]),
                grapheme_start=int(item["grapheme_index"]),
                grapheme_end=int(item["grapheme_index"] + 1),
                kind=classify_grapheme(original),
                original_codepoints=tuple(_codepoints(original)),
                writing_mode="unresolved",
                presentation_text=original,
                presentation_codepoints=tuple(_codepoints(original)),
                presentation_grapheme_start=int(item["grapheme_index"]),
                presentation_grapheme_end=int(item["grapheme_index"] + 1),
                atomic_break=False,
                orientation_policy="unresolved",
                render_policy="resolved_font_glyph",
                presentation_reason="identity_frozen_before_presentation",
            )
        )
        text_index += 1
        index += 1
    if tokens_original_text(tokens) != value:
        raise RuntimeError("lossless_text_token_identity_not_conserved")
    return tokens


def resolve_writing_mode_presentations(
    tokens: Sequence[TextToken | PunctuationToken],
    *,
    writing_mode: str,
    font_manager,
    face,
) -> list[TextToken | PunctuationToken]:
    """Attach presentation choices without mutating translated identity."""

    mode = "vertical" if str(writing_mode or "").lower().startswith("vert") else "horizontal"
    resolved: list[TextToken | PunctuationToken] = []
    presentation_index = 0
    for token in list(tokens or []):
        presentation = token.original_text
        candidate_forms = (presentation,)
        supported_forms: tuple[str, ...] = ()
        supported = True
        orientation_policy = "horizontal"
        render_policy = "resolved_font_glyph"
        reason = "identity_presentation"

        if isinstance(token, PunctuationToken):
            presentation = _punctuation_presentation_base(token)
            orientation_policy = _punctuation_orientation_policy(
                token.punctuation_kind,
                mode,
            )
            render_policy = _punctuation_render_policy(
                token.punctuation_kind,
                mode,
            )
            if mode == "vertical":
                if token.punctuation_kind == "emphasis_punctuation":
                    if (
                        token.exact_multiplicity == 2
                        and len(grapheme_clusters(token.original_text)) == 2
                    ):
                        presentation, candidate_forms, supported_forms, supported = (
                            _resolve_vertical_punctuation_form(
                                token.original_text,
                                font_manager=font_manager,
                                face=face,
                            )
                        )
                    elif len(grapheme_clusters(token.original_text)) == 1:
                        presentation, candidate_forms, supported_forms, supported = (
                            _resolve_vertical_punctuation_form(
                                token.original_text,
                                font_manager=font_manager,
                                face=face,
                            )
                        )
                    else:
                        presentation, candidate_forms, supported_forms, supported = (
                            _resolve_each_punctuation_grapheme(
                                _expand_emphasis_punctuation(token.original_text),
                                font_manager=font_manager,
                                face=face,
                            )
                        )
                elif token.punctuation_kind in {"wave", "dash"}:
                    presentation, candidate_forms, supported_forms, supported = (
                        _resolve_each_punctuation_grapheme(
                            token.original_text,
                            font_manager=font_manager,
                            face=face,
                        )
                    )
                else:
                    presentation, candidate_forms, supported_forms, supported = (
                        _resolve_vertical_punctuation_form(
                            presentation,
                            font_manager=font_manager,
                            face=face,
                        )
                    )
                reason = "vertical_presentation_selected_after_identity"
            else:
                supported = _symbol_supported(font_manager, face, presentation)
                supported_forms = (presentation,) if supported else ()
                reason = "horizontal_presentation_selected_after_identity"
            if token.punctuation_kind == "emphasis_punctuation" and token.unit_count > 1:
                orientation_policy = "compact_horizontal_inline_axis" if mode == "vertical" else "horizontal"
                render_policy = "shaped_compact_horizontal_sequence" if mode == "vertical" else "resolved_font_glyph"
        elif mode == "vertical" and token.kind == "space":
            presentation = ""
            supported_forms = ()
            orientation_policy = "layout_insignificant"
            render_policy = "no_ink_space"
            reason = "vertical_space_collapsed_after_identity"
        elif token.kind == "symbol":
            supported = _symbol_supported(font_manager, face, presentation)
            supported_forms = (presentation,) if supported else ()
            orientation_policy = "upright_resolved_face" if mode == "vertical" else "horizontal"

        presentation_clusters = grapheme_clusters(presentation)
        common = {
            "writing_mode": mode,
            "presentation_text": presentation,
            "presentation_codepoints": tuple(_codepoints(presentation)),
            "presentation_grapheme_start": presentation_index,
            "presentation_grapheme_end": presentation_index + len(presentation_clusters),
            "orientation_policy": orientation_policy,
            "render_policy": render_policy,
            "presentation_reason": reason,
        }
        if isinstance(token, PunctuationToken):
            resolved.append(
                replace(
                    token,
                    **common,
                    candidate_forms=tuple(candidate_forms),
                    supported_forms=tuple(supported_forms),
                    supported=bool(supported),
                )
            )
        else:
            resolved.append(replace(token, **common))
        presentation_index += len(presentation_clusters)
    if tokens_original_text(resolved) != tokens_original_text(tokens):
        raise RuntimeError("lossless_text_token_identity_changed_by_presentation")
    return resolved


def tokens_original_text(
    tokens: Sequence[TextToken | PunctuationToken],
) -> str:
    return "".join(str(item.original_text) for item in list(tokens or []))


def tokens_presentation_text(
    tokens: Sequence[TextToken | PunctuationToken],
) -> str:
    return "".join(str(item.presentation_text) for item in list(tokens or []))


def punctuation_occurrences_from_tokens(
    tokens: Sequence[TextToken | PunctuationToken],
) -> list[dict[str, Any]]:
    return [
        _punctuation_token_occurrence(item)
        for item in list(tokens or [])
        if isinstance(item, PunctuationToken)
    ]


def symbol_occurrences_from_tokens(
    tokens: Sequence[TextToken | PunctuationToken],
    *,
    font_manager,
    face,
) -> list[dict[str, Any]]:
    symbols: list[dict[str, Any]] = []
    for token in list(tokens or []):
        if isinstance(token, PunctuationToken) or token.kind != "symbol":
            continue
        symbols.append(
            {
                "occurrence_id": f"symbol_{len(symbols):04d}",
                "token_id": token.token_id,
                "symbol": token.original_text,
                "source": token.original_text,
                "normalized": token.presentation_text,
                "source_codepoints": list(token.original_codepoints),
                "normalized_codepoints": list(token.presentation_codepoints),
                "source_start": int(token.translated_start),
                "source_end": int(token.translated_end),
                "normalized_grapheme_start": int(token.presentation_grapheme_start),
                "normalized_grapheme_end": int(token.presentation_grapheme_end),
                "writing_mode": token.writing_mode,
                "kind": "symbol",
                "source_class": _symbol_class(token.original_text),
                "orientation_policy": token.orientation_policy,
                "render_policy": token.render_policy,
                "supported": _symbol_supported(font_manager, face, token.presentation_text),
            }
        )
    return symbols


def presentation_notes_from_tokens(
    tokens: Sequence[TextToken | PunctuationToken],
) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for token in list(tokens or []):
        if token.presentation_reason == "vertical_space_collapsed_after_identity":
            notes.append(
                {
                    "type": "vertical_space_collapsed",
                    "token_id": token.token_id,
                    "source": token.original_text,
                    "source_index": int(token.translated_start),
                    "reason": "layout_insignificant_vertical_space",
                }
            )
        if (
            isinstance(token, PunctuationToken)
            and token.punctuation_kind == "emphasis_punctuation"
            and token.writing_mode == "vertical"
            and token.unit_count >= 3
        ):
            notes.append(
                {
                    "type": "vertical_emphasis_run_normalized",
                    "token_id": token.token_id,
                    "source": token.original_text,
                    "normalized": token.presentation_text,
                    "source_index": int(token.translated_start),
                    "unit_count": int(token.unit_count),
                    "sequence_group_count": int(token.sequence_group_count),
                    "reason": "preserve_complete_emphasis_run_for_atomic_inline_composition",
                }
            )
    return notes


def normalize_for_writing_mode(
    text: str,
    writing_mode: str,
    font_manager,
    face,
) -> tuple[str, list[dict], list[dict], list[dict]]:
    """Compatibility view derived from the sole lossless token path."""

    tokens = resolve_writing_mode_presentations(
        build_lossless_text_tokens(text),
        writing_mode=writing_mode,
        font_manager=font_manager,
        face=face,
    )
    return (
        tokens_presentation_text(tokens),
        punctuation_occurrences_from_tokens(tokens),
        symbol_occurrences_from_tokens(tokens, font_manager=font_manager, face=face),
        presentation_notes_from_tokens(tokens),
    )


def _grapheme_records(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    for grapheme_index, cluster in enumerate(grapheme_clusters(text)):
        records.append(
            {
                "text": cluster,
                "start": offset,
                "end": offset + len(cluster),
                "grapheme_index": grapheme_index,
            }
        )
        offset += len(cluster)
    return records


def _punctuation_sequence_at(
    records: Sequence[Mapping[str, Any]],
    index: int,
) -> tuple[int, str]:
    if index < 0 or index >= len(records):
        return index, ""
    cluster = str(records[index].get("text") or "")
    if cluster == "\u00ad":
        return index + 1, "soft_hyphen"
    if cluster == ".":
        end = index
        while end < len(records) and str(records[end].get("text") or "") == ".":
            end += 1
        if end - index >= 3:
            return end, "ellipsis"
    if cluster in ELLIPSIS_DOT_WEIGHTS:
        end = index + 1
        while end < len(records) and str(records[end].get("text") or "") in ELLIPSIS_DOT_WEIGHTS:
            end += 1
        return end, "ellipsis"
    if cluster in MIDDLE_DOT_CHARS:
        end = index + 1
        while end < len(records) and str(records[end].get("text") or "") in MIDDLE_DOT_CHARS:
            end += 1
        return (end, "ellipsis") if end - index > 1 else (end, "middle_dot")
    if cluster in COMPACT_VERTICAL_PUNCTUATION_CHARS:
        end = index + 1
        while end < len(records) and str(records[end].get("text") or "") in COMPACT_VERTICAL_PUNCTUATION_CHARS:
            end += 1
        return end, "emphasis_punctuation"
    if cluster in _TYPESETTING_DASH_CHARS:
        end = index + 1
        while end < len(records) and str(records[end].get("text") or "") == cluster:
            end += 1
        return end, "dash"
    if cluster in WAVE_DASH_CHARS:
        end = index + 1
        while end < len(records) and str(records[end].get("text") or "") == cluster:
            end += 1
        return end, "wave"
    kind = classify_grapheme(cluster)
    if kind in {"open_punctuation", "close_punctuation", "punctuation"}:
        return index + 1, kind
    return index, ""


def _punctuation_dot_count(source: str, punctuation_kind: str) -> int:
    if punctuation_kind != "ellipsis":
        return 0
    value = str(source or "")
    if value and all(char == "." for char in value):
        return len(value)
    if value and all(char in MIDDLE_DOT_CHARS for char in value):
        return len(grapheme_clusters(value))
    return sum(ELLIPSIS_DOT_WEIGHTS.get(char, 0) for char in value)


def _exact_punctuation_multiplicity(source: str, punctuation_kind: str) -> int:
    if punctuation_kind == "emphasis_punctuation":
        return max(
            1,
            sum(
                EMPHASIS_PUNCTUATION_UNIT_COUNTS.get(cluster, 1)
                for cluster in grapheme_clusters(source)
            ),
        )
    return max(1, len(grapheme_clusters(source)))


def _presentation_unit_count(
    source: str,
    punctuation_kind: str,
    dot_count: int,
) -> int:
    if punctuation_kind == "ellipsis":
        return max(1, (int(dot_count) + 2) // 3)
    if punctuation_kind == "emphasis_punctuation":
        return _exact_punctuation_multiplicity(source, punctuation_kind)
    return max(1, len(grapheme_clusters(source)))


def _lossless_punctuation_source_class(source: str, punctuation_kind: str) -> str:
    value = str(source or "")
    if punctuation_kind == "ellipsis":
        if value and all(char == "." for char in value):
            return "ascii_dot_ellipsis_sequence"
        if value and all(char in MIDDLE_DOT_CHARS for char in value):
            return "middle_dot_ellipsis_sequence"
        if value and all(char in ELLIPSIS_DOT_WEIGHTS for char in value):
            return "unicode_ellipsis_sequence"
        return "mixed_ellipsis_sequence"
    if punctuation_kind == "middle_dot":
        return "fullwidth_middle_dot" if value == "・" else "halfwidth_middle_dot"
    if punctuation_kind == "wave":
        return {
            "~": "ascii_tilde",
            "～": "fullwidth_tilde",
            "〜": "wave_dash",
            "〰": "wavy_dash",
            "︴": "vertical_wavy_line",
        }.get(value[:1], "mixed_wave_sequence")
    if punctuation_kind == "dash":
        return "dash_sequence"
    if punctuation_kind in {"open_punctuation", "close_punctuation"}:
        return punctuation_kind
    if punctuation_kind == "emphasis_punctuation":
        return "emphasis_punctuation_sequence"
    return "punctuation"


def _punctuation_presentation_base(token: PunctuationToken) -> str:
    if token.punctuation_kind == "soft_hyphen":
        return token.original_text
    if token.punctuation_kind != "ellipsis":
        return token.original_text
    if token.source_class == "ascii_dot_ellipsis_sequence":
        full_units, remainder = divmod(int(token.dot_count), 3)
        return ("…" * full_units) + ("." * remainder)
    return token.original_text


def _resolve_vertical_punctuation_form(
    source: str,
    *,
    font_manager,
    face,
) -> tuple[str, tuple[str, ...], tuple[str, ...], bool]:
    try:
        support = font_manager.vertical_punctuation_support(face, source)
        candidates = tuple(str(item) for item in support.candidate_forms)
        supported_forms = tuple(str(item) for item in support.supported_forms)
        selected = str(support.selected_form or source)
        return selected, candidates or (source,), supported_forms, bool(support.supported)
    except Exception:
        return source, (source,), (), False


def _resolve_each_punctuation_grapheme(
    source: str,
    *,
    font_manager,
    face,
) -> tuple[str, tuple[str, ...], tuple[str, ...], bool]:
    selected: list[str] = []
    supported = True
    for cluster in grapheme_clusters(source):
        form, _candidates, supported_forms, cluster_supported = (
            _resolve_vertical_punctuation_form(
                cluster,
                font_manager=font_manager,
                face=face,
            )
        )
        selected.append(form or cluster)
        supported = supported and bool(cluster_supported)
        if not supported_forms:
            supported = False
    presentation = "".join(selected)
    return (
        presentation,
        (presentation,),
        (presentation,) if supported else (),
        supported,
    )


def _punctuation_token_occurrence(token: PunctuationToken) -> dict[str, Any]:
    record = {
        "occurrence_id": token.token_id,
        "token_id": token.token_id,
        "source": token.original_text,
        "normalized": token.presentation_text,
        "source_codepoints": list(token.original_codepoints),
        "normalized_codepoints": list(token.presentation_codepoints),
        "source_start": int(token.translated_start),
        "source_end": int(token.translated_end),
        "normalized_grapheme_start": int(token.presentation_grapheme_start),
        "normalized_grapheme_end": int(token.presentation_grapheme_end),
        "writing_mode": token.writing_mode,
        "kind": token.punctuation_kind,
        "source_class": token.source_class,
        "orientation_policy": token.orientation_policy,
        "render_policy": token.render_policy,
        "exact_multiplicity": int(token.exact_multiplicity),
        "unit_count": int(token.unit_count),
        "supported": bool(token.supported),
        "candidate_forms": list(token.candidate_forms),
        "supported_forms": list(token.supported_forms),
        "sequence_group_count": int(token.sequence_group_count),
    }
    if token.dot_count:
        record["dot_count"] = int(token.dot_count)
    return record


def classify_grapheme(text: str) -> str:
    cluster = str(text or "")
    if not cluster:
        return "empty"
    if cluster.isspace():
        return "space"
    if all(is_default_ignorable_codepoint(char) for char in cluster):
        return "format_control"
    if _is_ellipsis_sequence(cluster):
        return "ellipsis"
    if _is_dash_sequence(cluster):
        return "dash"
    if _is_wave_sequence(cluster):
        return "wave"
    if cluster in OPEN_PUNCTUATION:
        return "open_punctuation"
    if cluster in CLOSE_PUNCTUATION:
        return "close_punctuation"
    if cluster in SYMBOL_CHARS or _is_emoji(cluster):
        return "symbol"
    if _is_latin_grapheme(cluster):
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
    cat = unicodedata.category(cluster[0])
    if cat.startswith("P"):
        return "punctuation"
    if cat.startswith("M"):
        return "mark"
    return "other"


def segment_inline_runs(
    text: str | Sequence[TextToken | PunctuationToken],
    *,
    writing_mode: str,
    language_hint: str = "",
    punctuation_occurrences: Sequence[dict[str, Any]] | None = None,
    symbol_occurrences: Sequence[dict[str, Any]] | None = None,
    latin_lexical_policy: bool = False,
) -> list[InlineTextRun]:
    if isinstance(text, str):
        tokens: list[TextToken | PunctuationToken] = build_lossless_text_tokens(text)
    else:
        tokens = list(text or [])
        if not all(isinstance(item, (TextToken, PunctuationToken)) for item in tokens):
            raise TypeError("segment_inline_runs_requires_text_or_lossless_tokens")
    runs: list[InlineTextRun] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        kind = _token_break_class(token)
        start = index
        group: list[TextToken | PunctuationToken] = [token]
        text_value = token.presentation_text
        if isinstance(token, PunctuationToken) and token.punctuation_kind in {"wave", "dash"}:
            while index + 1 < len(tokens):
                next_token = tokens[index + 1]
                if not (
                    isinstance(next_token, PunctuationToken)
                    and next_token.punctuation_kind == token.punctuation_kind
                ):
                    break
                group.append(next_token)
                index += 1
        elif not isinstance(token, PunctuationToken) and kind in {"latin", "number"}:
            while index + 1 < len(tokens):
                next_token = tokens[index + 1]
                if isinstance(next_token, PunctuationToken):
                    if (
                        latin_lexical_policy
                        and _latin_internal_joiner(
                            tokens,
                            index + 1,
                            base_kind=kind,
                        )
                    ):
                        group.append(next_token)
                        index += 1
                        continue
                    break
                next_kind = _token_break_class(next_token)
                if next_kind not in {kind, "format_control"}:
                    break
                group.append(next_token)
                index += 1
        elif not isinstance(token, PunctuationToken) and kind in {"rtl", "complex"}:
            script = _script_for_run(text_value, kind)
            while index + 1 < len(tokens):
                next_token = tokens[index + 1]
                if isinstance(next_token, PunctuationToken):
                    break
                next_kind = _token_break_class(next_token)
                if next_kind == "format_control":
                    group.append(next_token)
                    text_value = "".join(item.presentation_text for item in group)
                    index += 1
                    continue
                next_script = _script_for_run(next_token.presentation_text, next_kind)
                if next_kind != kind or next_script != script:
                    break
                group.append(next_token)
                text_value = "".join(item.presentation_text for item in group)
                index += 1
        text_value = "".join(item.presentation_text for item in group)
        original_text = "".join(item.original_text for item in group)
        script = _script_for_run(text_value, kind)
        direction = _direction_for_script(script)
        role = _role_for_token_group(kind, group, text_value)
        presentation_start = int(group[0].presentation_grapheme_start)
        presentation_end = int(group[-1].presentation_grapheme_end)
        metadata: dict[str, Any] = {
            "token_ids": [item.token_id for item in group],
            "lossless_tokens": [item.to_audit_dict() for item in group],
            "original_text": original_text,
            "translated_start": int(group[0].translated_start),
            "translated_end": int(group[-1].translated_end),
            "atomic_break": bool(group and all(item.atomic_break for item in group)),
        }
        if role == "latin_word":
            metadata["letter_stacking_allowed"] = False
        if direction == "rtl" and _bidi_get_display is not None:
            metadata["bidi_visual_text"] = _bidi_get_display(text_value)
        punctuation_evidence = [
            _punctuation_token_occurrence(item)
            for item in group
            if isinstance(item, PunctuationToken)
        ]
        symbol_evidence = [
            {
                "token_id": item.token_id,
                "symbol": item.original_text,
                "source": item.original_text,
                "normalized": item.presentation_text,
                "source_codepoints": list(item.original_codepoints),
                "normalized_codepoints": list(item.presentation_codepoints),
                "source_start": int(item.translated_start),
                "source_end": int(item.translated_end),
                "normalized_grapheme_start": int(item.presentation_grapheme_start),
                "normalized_grapheme_end": int(item.presentation_grapheme_end),
                "kind": "symbol",
            }
            for item in group
            if not isinstance(item, PunctuationToken) and item.kind == "symbol"
        ]
        if not punctuation_evidence and isinstance(text, str):
            punctuation_evidence = _occurrences_for_span(
                punctuation_occurrences,
                presentation_start,
                presentation_end,
            )
        if not symbol_evidence and isinstance(text, str):
            symbol_evidence = _occurrences_for_span(
                symbol_occurrences,
                presentation_start,
                presentation_end,
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
                grapheme_start=presentation_start,
                grapheme_end=presentation_end,
                script=script,
                direction=direction,
                language=_language_for_script(script, language_hint),
                role=role,
                break_class=kind,
                original_text=original_text,
                translated_start=int(group[0].translated_start),
                translated_end=int(group[-1].translated_end),
                token_start=start,
                token_end=index + 1,
                token_ids=tuple(item.token_id for item in group),
                metadata=metadata,
            )
        )
        index += 1
    return _coalesce_format_control_runs(runs)


_LATIN_APOSTROPHE_JOINERS = {"'", "’", "ʼ", "＇"}
_LATIN_HYPHEN_JOINERS = {"-", "‐", "‑"}
_LATIN_PERIOD_JOINERS = {"."}


def _latin_internal_joiner(
    tokens: Sequence[TextToken | PunctuationToken],
    punctuation_index: int,
    *,
    base_kind: str,
) -> bool:
    """Return whether one punctuation token belongs inside a Latin word.

    The rule is enabled only by the explicit target-presentation policy. It
    preserves the existing lossless punctuation token and merely prevents an
    ordinary line break inside contractions, possessives, compounds,
    abbreviations, and decimal numbers.
    """

    values = list(tokens or [])
    if punctuation_index <= 0 or punctuation_index + 1 >= len(values):
        return False
    token = values[punctuation_index]
    if not isinstance(token, PunctuationToken):
        return False
    before = values[punctuation_index - 1]
    after = values[punctuation_index + 1]
    if isinstance(before, PunctuationToken) or isinstance(after, PunctuationToken):
        return False
    before_kind = _token_break_class(before)
    after_kind = _token_break_class(after)
    text = str(token.presentation_text or token.original_text or "")
    if text in _LATIN_APOSTROPHE_JOINERS:
        return bool(
            base_kind == "latin"
            and before_kind == "latin"
            and after_kind == "latin"
        )
    if text in _LATIN_HYPHEN_JOINERS:
        return bool(
            base_kind in {"latin", "number"}
            and before_kind in {"latin", "number"}
            and after_kind in {"latin", "number"}
        )
    if text in _LATIN_PERIOD_JOINERS:
        return bool(
            (base_kind == "latin" and before_kind == after_kind == "latin")
            or (base_kind == "number" and before_kind == after_kind == "number")
        )
    return False


def _token_break_class(token: TextToken | PunctuationToken) -> str:
    if isinstance(token, PunctuationToken):
        if token.punctuation_kind == "soft_hyphen":
            return "soft_hyphen"
        if token.punctuation_kind in {"ellipsis", "dash", "wave"}:
            return token.punctuation_kind
        if (
            token.punctuation_kind == "emphasis_punctuation"
            and len(grapheme_clusters(token.original_text)) == 1
            and token.original_text in CLOSE_PUNCTUATION
        ):
            return "close_punctuation"
        if token.punctuation_kind in {"open_punctuation", "close_punctuation"}:
            return token.punctuation_kind
        return "punctuation"
    if token.kind == "space" or (
        not token.presentation_text and token.original_text.isspace()
    ):
        return "space"
    return token.kind or classify_grapheme(token.presentation_text)


def _role_for_token_group(
    kind: str,
    group: Sequence[TextToken | PunctuationToken],
    presentation_text: str,
) -> str:
    if group and all(isinstance(item, PunctuationToken) for item in group):
        token = group[0]
        if token.punctuation_kind == "ellipsis":
            return "ellipsis_sequence"
        if token.punctuation_kind == "dash":
            return "dash_sequence"
        if token.punctuation_kind == "wave":
            return "wave_sequence"
        if token.punctuation_kind == "emphasis_punctuation":
            return (
                "punctuation_sequence"
                if len(grapheme_clusters(token.original_text)) > 1
                else "close_punctuation"
            )
        return token.punctuation_kind
    return _role_for_kind(kind, presentation_text)


def _coalesce_format_control_runs(runs: Sequence[InlineTextRun]) -> list[InlineTextRun]:
    """Attach non-rendering controls to adjacent shaped runs without a cell."""

    output: list[InlineTextRun] = []
    pending: list[InlineTextRun] = []
    for run in runs:
        if run.role == "format_control":
            if output:
                previous = output[-1]
                output[-1] = replace(
                    previous,
                    text=previous.text + run.text,
                    normalized_text=previous.normalized_text + run.normalized_text,
                    original_text=previous.original_text + run.original_text,
                    translated_end=run.translated_end,
                    grapheme_end=run.grapheme_end,
                    token_end=run.token_end,
                    token_ids=previous.token_ids + run.token_ids,
                    metadata={
                        **dict(previous.metadata),
                        "token_ids": [*previous.token_ids, *run.token_ids],
                        "lossless_tokens": [
                            *list(previous.metadata.get("lossless_tokens") or []),
                            *list(run.metadata.get("lossless_tokens") or []),
                        ],
                        "original_text": previous.original_text + run.original_text,
                        "translated_end": run.translated_end,
                        "attached_default_ignorable_text": (
                            str(previous.metadata.get("attached_default_ignorable_text") or "")
                            + run.text
                        ),
                    },
                )
            else:
                pending.append(run)
            continue
        if pending:
            prefix = "".join(item.text for item in pending)
            run = replace(
                run,
                text=prefix + run.text,
                normalized_text=prefix + run.normalized_text,
                original_text="".join(item.original_text for item in pending) + run.original_text,
                translated_start=pending[0].translated_start,
                grapheme_start=pending[0].grapheme_start,
                token_start=pending[0].token_start,
                token_ids=tuple(
                    token_id
                    for item in [*pending, run]
                    for token_id in item.token_ids
                ),
                metadata={
                    **dict(run.metadata),
                    "token_ids": [
                        token_id
                        for item in [*pending, run]
                        for token_id in item.token_ids
                    ],
                    "lossless_tokens": [
                        token
                        for item in [*pending, run]
                        for token in list(item.metadata.get("lossless_tokens") or [])
                    ],
                    "original_text": "".join(item.original_text for item in pending) + run.original_text,
                    "translated_start": pending[0].translated_start,
                    "attached_default_ignorable_text": prefix,
                },
            )
            pending.clear()
        output.append(run)
    output.extend(pending)
    return output


def compute_break_opportunities(
    runs: Sequence[InlineTextRun],
    *,
    writing_mode: str,
    target_lexical_spans: Sequence["TargetLexicalSpan"] | None = None,
    target_lexical_boundaries: Sequence["TargetLexicalBoundary"] | None = None,
    language_hint: str = "zh",
) -> list[BreakOpportunity]:
    items = list(runs or [])
    lexical_spans = list(target_lexical_spans or [])
    lexical_boundaries = {
        int(item.token_boundary): item
        for item in list(target_lexical_boundaries or [])
    }
    lexical_evidence_supplied = target_lexical_boundaries is not None
    opportunities: list[BreakOpportunity] = []
    for index in range(1, len(items)):
        before = items[index - 1]
        after = items[index]
        token_boundary = int(after.token_start)
        covering_lexical_spans = [
            span
            for span in lexical_spans
            if int(span.token_start) < token_boundary < int(span.token_end)
            and int(span.grapheme_end) - int(span.grapheme_start) > 1
        ]
        lexical_boundary = lexical_boundaries.get(token_boundary)
        lexical_state = str(
            getattr(lexical_boundary, "state", "unknown") or "unknown"
        )
        confirmed_rank = max(
            0,
            int(getattr(lexical_boundary, "confirmed_keep_rank", 0) or 0),
        )
        weak_rank = max(
            0,
            int(getattr(lexical_boundary, "weak_keep_rank", 0) or 0),
        )
        lexical_contributors = [
            item.to_audit_dict()
            for item in tuple(
                getattr(lexical_boundary, "contributors", ()) or ()
            )
        ]
        reason = "generic_run_boundary"
        strength = "normal"
        allowed = True
        if after.role == "soft_hyphen":
            reason = "manual_soft_hyphen_before_forbidden"
            strength = "forbidden"
            allowed = False
        elif before.role == "soft_hyphen":
            reason = "manual_soft_hyphen"
            strength = "preferred"
        elif after.text in CLOSE_PUNCTUATION or after.role in {"ellipsis_sequence", "dash_sequence", "punctuation_sequence"} and after.text[:1] in CLOSE_PUNCTUATION:
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
            if lexical_state == "confirmed_keep":
                reason = "target_lexical_confirmed_keep_boundary"
                strength = "weak"
            elif lexical_state == "weak_keep":
                reason = "target_lexical_weak_keep_boundary"
                strength = "weak"
            elif lexical_evidence_supplied:
                reason = f"target_lexical_{lexical_state}_boundary"
                strength = "normal"
            else:
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
                    "token_boundary": token_boundary,
                    "target_lexical_boundary": (
                        lexical_state
                        if lexical_evidence_supplied
                        else "unavailable"
                    ),
                    "lexical_integrity_penalty": (
                        float(confirmed_rank)
                        if confirmed_rank
                        else float(weak_rank) / 10.0
                    ),
                    "lexical_boundary_state": lexical_state,
                    "confirmed_lexical_break_rank": int(confirmed_rank),
                    "weak_lexical_break_rank": int(weak_rank),
                    "lexical_evidence_conflict": bool(
                        getattr(lexical_boundary, "conflict", False)
                    ),
                    "lexical_boundary_id": str(
                        getattr(lexical_boundary, "boundary_id", "") or ""
                    ),
                    "lexical_evidence_contributors": lexical_contributors,
                    "lexical_span_ids": [
                        span.span_id for span in covering_lexical_spans
                    ],
                    "lexical_span_texts": [
                        span.text for span in covering_lexical_spans
                    ],
                },
            )
        )
    if writing_mode == "vertical":
        index = 0
        while index < len(items):
            if items[index].break_class != "space":
                index += 1
                continue
            space_start = index
            while index < len(items) and items[index].break_class == "space":
                index += 1
            if space_start <= 0 or index >= len(items):
                continue
            before = items[space_start - 1]
            after = items[index]
            allowed = True
            strength = "preferred"
            reason = "vertical_no_ink_space_boundary"
            if (
                after.text in CLOSE_PUNCTUATION
                or (
                    after.role
                    in {
                        "ellipsis_sequence",
                        "dash_sequence",
                        "punctuation_sequence",
                    }
                    and after.text[:1] in CLOSE_PUNCTUATION
                )
            ):
                allowed = False
                strength = "forbidden"
                reason = "kinsoku_rejected_before_closing_punctuation"
            elif before.text in OPEN_PUNCTUATION:
                allowed = False
                strength = "forbidden"
                reason = "kinsoku_rejected_after_opening_punctuation"
            skipped_spaces = items[space_start:index]
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
                        "token_boundary": int(after.token_start),
                        "target_lexical_boundary": (
                            "unknown" if lexical_evidence_supplied else "unavailable"
                        ),
                        "lexical_integrity_penalty": 0.0,
                        "lexical_boundary_state": "unknown",
                        "confirmed_lexical_break_rank": 0,
                        "weak_lexical_break_rank": 0,
                        "lexical_evidence_conflict": False,
                        "lexical_evidence_contributors": [],
                        "lexical_span_ids": [],
                        "lexical_span_texts": [],
                        "no_ink_space_transport": True,
                        "skipped_space_run_ids": [
                            item.run_id for item in skipped_spaces
                        ],
                        "skipped_space_text": "".join(
                            item.text for item in skipped_spaces
                        ),
                        "skipped_space_token_start": int(
                            skipped_spaces[0].token_start
                        ),
                        "skipped_space_token_end": int(
                            skipped_spaces[-1].token_end
                        ),
                    },
                )
            )
    return _apply_pyicu_strict_line_break_legality(
        items,
        opportunities,
        language_hint=language_hint,
    )


_PYICU_STRICT_PROVIDER_ID = "icu_zh_strict"
_PYICU_STRICT_ITERATORS = threading.local()


def _require_pyicu() -> Any:
    """Load the required renderer legality provider without a silent fallback."""

    try:
        import icu
    except Exception as exc:  # pragma: no cover - exercised by packaged smoke
        raise RuntimeError("pyicu_strict_line_break_provider_unavailable") from exc
    if not hasattr(icu, "BreakIterator") or not hasattr(icu, "Locale"):
        raise RuntimeError("pyicu_strict_line_break_provider_invalid")
    return icu


def _pyicu_strict_locale_id(language_hint: str) -> str:
    """Resolve the target-language locale while forcing ICU strict line rules."""

    raw = str(language_hint or "zh").strip()
    raw = raw.split("@", 1)[0].split("-u-", 1)[0].strip()
    if not raw or raw.lower() in {"auto", "und", "unknown"}:
        raw = "zh"
    language_tag = raw.replace("_", "-")
    if not all(part.isalnum() for part in language_tag.split("-") if part):
        raise RuntimeError(f"pyicu_strict_locale_invalid:{language_hint}")
    icu = _require_pyicu()
    locale = icu.Locale.forLanguageTag(language_tag)
    locale_name = str(locale.getName() or "").strip()
    if not locale_name:
        raise RuntimeError(f"pyicu_strict_locale_invalid:{language_hint}")
    return f"{locale_name}@lb=strict"


def _pyicu_utf16_boundary_map(text: str) -> dict[int, int]:
    mapping = {0: 0}
    offset = 0
    for index, character in enumerate(text):
        offset += 2 if ord(character) > 0xFFFF else 1
        mapping[offset] = index + 1
    return mapping


def pyicu_strict_line_break_boundaries(
    text: str,
    *,
    language_hint: str = "zh",
) -> tuple[int, ...]:
    """Return strict ICU line boundaries as Python code-point offsets.

    ICU exposes UTF-16 offsets.  The renderer's immutable translated spans use
    Python code-point offsets, so every returned boundary is converted and
    checked before it can participate in legality filtering.
    """

    value = str(text or "")
    if not value:
        return (0,)
    icu = _require_pyicu()
    locale_id = _pyicu_strict_locale_id(language_hint)
    iterators = getattr(_PYICU_STRICT_ITERATORS, "by_locale", None)
    if iterators is None:
        iterators = {}
        _PYICU_STRICT_ITERATORS.by_locale = iterators
    iterator = iterators.get(locale_id)
    if iterator is None:
        iterator = icu.BreakIterator.createLineInstance(icu.Locale(locale_id))
        iterators[locale_id] = iterator

    utf16_to_codepoint = _pyicu_utf16_boundary_map(value)
    iterator.setText(value)
    positions = [0]
    for raw_offset in iterator:
        utf16_offset = int(raw_offset)
        if utf16_offset not in utf16_to_codepoint:
            raise RuntimeError(
                "pyicu_line_boundary_not_codepoint_aligned:"
                f"{locale_id}:{utf16_offset}"
            )
        positions.append(utf16_to_codepoint[utf16_offset])
    boundaries = tuple(sorted(set(positions)))
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != len(value):
        raise RuntimeError(f"pyicu_line_boundary_sequence_not_conserved:{locale_id}")
    return boundaries


def _translated_text_from_runs(runs: Sequence[InlineTextRun]) -> str:
    cursor = 0
    chunks: list[str] = []
    for run in runs:
        start = int(run.translated_start)
        end = int(run.translated_end)
        original = str(run.original_text or "")
        if start != cursor or end != start + len(original):
            raise RuntimeError(
                "pyicu_line_break_translated_run_provenance_not_contiguous"
            )
        chunks.append(original)
        cursor = end
    return "".join(chunks)


def _apply_pyicu_strict_line_break_legality(
    runs: Sequence[InlineTextRun],
    opportunities: Sequence[BreakOpportunity],
    *,
    language_hint: str,
) -> list[BreakOpportunity]:
    items = list(runs or [])
    candidates = list(opportunities or [])
    if not items or not candidates:
        return candidates

    translated_text = _translated_text_from_runs(items)
    locale_id = _pyicu_strict_locale_id(language_hint)
    provider_boundaries = set(
        pyicu_strict_line_break_boundaries(
            translated_text,
            language_hint=language_hint,
        )
    )
    runs_by_id = {run.run_id: run for run in items}
    icu = _require_pyicu()
    filtered: list[BreakOpportunity] = []
    for opportunity in candidates:
        after = runs_by_id.get(opportunity.after_run_id)
        if after is None:
            raise RuntimeError("pyicu_line_break_after_run_missing")
        translated_position = int(after.translated_start)
        if translated_position < 0 or translated_position > len(translated_text):
            raise RuntimeError("pyicu_line_break_translated_position_invalid")
        provider_allowed = translated_position in provider_boundaries
        original_allowed = bool(opportunity.allowed)
        effective_allowed = original_allowed and provider_allowed
        reason = opportunity.reason
        strength = opportunity.strength
        if original_allowed and not provider_allowed:
            reason = "uax14_provider_rejected_boundary"
            strength = "forbidden"
        filtered.append(
            replace(
                opportunity,
                allowed=effective_allowed,
                reason=reason,
                strength=strength,
                metadata={
                    **dict(opportunity.metadata),
                    "uax14_provider_id": _PYICU_STRICT_PROVIDER_ID,
                    "uax14_provider_role": "legal_opportunity_filter",
                    "uax14_provider_locale": locale_id,
                    "uax14_translated_position": translated_position,
                    "uax14_provider_allowed": provider_allowed,
                    "uax14_original_allowed": original_allowed,
                    "uax14_only_narrows": True,
                    "jieba_lexical_evidence_retained": True,
                    "pyicu_version": str(icu.VERSION),
                    "icu_version": str(icu.ICU_VERSION),
                    "icu_unicode_version": str(icu.UNICODE_VERSION),
                    "icu_offset_contract": "utf16_to_python_codepoint",
                },
            )
        )
    return filtered


def _symbol_supported(font_manager, face, symbol: str) -> bool:
    try:
        return bool(font_manager.coverage_for_text(face, symbol).supports_text)
    except Exception:
        return False


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
    if (0x41 <= codepoint <= 0x5A) or (0x61 <= codepoint <= 0x7A):
        return True
    if _LATIN_SCRIPT_RE is not None:
        return bool(_LATIN_SCRIPT_RE.fullmatch(char))
    return "LATIN" in unicodedata.name(char, "")


def _is_latin_grapheme(cluster: str) -> bool:
    has_latin_base = False
    for char in str(cluster or ""):
        if _is_latin_char(char):
            has_latin_base = True
            continue
        if unicodedata.category(char).startswith("M"):
            continue
        if is_default_ignorable_codepoint(char):
            continue
        return False
    return has_latin_base


def _is_emoji(cluster: str) -> bool:
    value = str(cluster or "")
    return bool(
        is_emoji_grapheme_cluster(value)
        or (
            _EXTENDED_PICTOGRAPHIC_RE is not None
            and _EXTENDED_PICTOGRAPHIC_RE.search(value)
        )
        or any(0x2600 <= ord(char) <= 0x27BF for char in value)
    )


def _is_ellipsis_sequence(text: str) -> bool:
    return bool(text) and all(char in ELLIPSIS_CHARS for char in text)


def _is_dash_sequence(text: str) -> bool:
    return bool(text) and all(char in _TYPESETTING_DASH_CHARS for char in text)


def _is_wave_sequence(text: str) -> bool:
    return bool(text) and all(char in WAVE_DASH_CHARS for char in text)


def _is_compact_vertical_punctuation_char(text: str) -> bool:
    return bool(text) and text in COMPACT_VERTICAL_PUNCTUATION_CHARS


def _is_compact_vertical_punctuation_sequence(text: str) -> bool:
    clusters = grapheme_clusters(text)
    return len(clusters) > 1 and all(_is_compact_vertical_punctuation_char(cluster) for cluster in clusters)


def _expand_emphasis_punctuation(source: str) -> str:
    return "".join(
        EMPHASIS_PUNCTUATION_EXPANSIONS.get(cluster, cluster)
        for cluster in grapheme_clusters(source)
    )


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
