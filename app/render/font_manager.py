# -*- coding: utf-8 -*-
"""OpenType font registry, coverage, fallback, and metrics services."""
from __future__ import annotations

import hashlib
import os
import statistics
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping, Sequence

from app.models import resolution as model_resolution
from app.render.typesetting_text import (
    is_emoji_grapheme_cluster,
    strict_grapheme_clusters,
)

try:
    from PIL import ImageFont
except Exception:  # pragma: no cover - optional runtime dependency
    ImageFont = None

try:
    from fontTools.ttLib import TTFont
except Exception:  # pragma: no cover - optional runtime dependency
    TTFont = None


FONT_MANAGER_VERSION = "font_manager_v1"
TARGET_OPTICAL_PROFILE_VERSION = "target_optical_profile_v2"
TARGET_OPTICAL_PROFILE_REFERENCE_EM_PX = 64
TARGET_OPTICAL_PROFILE_POLICY_VERSION = "fixed_disjoint_cjk_probe_bank_v1"
TARGET_OPTICAL_PROFILE_AGGREGATION = "median_of_probe_medians_v1"
TARGET_OPTICAL_PROFILE_PROBE_BANK = (
    ("cjk_cosmos_single", "天地玄黄宇宙洪荒"),
    ("cjk_seasons_single", "春夏秋冬東西南北"),
    ("cjk_nature_single", "海山川空月星光雨"),
    ("hiragana_single", "あいうえおかきくけこ"),
    ("katakana_single", "アイウエオカキクケコ"),
    ("cjk_life_two_column", "永語愛無人生活仕事"),
    ("cjk_story_two_column", "漢字仮名読書物語心"),
    ("cjk_motion_two_column", "時風道夢声旅力世界"),
)
TARGET_OPTICAL_PROFILE_CJK_PROBE = tuple(
    sorted(
        {
            cluster
            for _, text in TARGET_OPTICAL_PROFILE_PROBE_BANK
            for cluster in strict_grapheme_clusters(text)
        }
    )
)
_target_optical_policy_digest = hashlib.sha256()
for _target_optical_policy_part in (
    TARGET_OPTICAL_PROFILE_VERSION,
    TARGET_OPTICAL_PROFILE_POLICY_VERSION,
    TARGET_OPTICAL_PROFILE_AGGREGATION,
    str(TARGET_OPTICAL_PROFILE_REFERENCE_EM_PX),
    *(
        f"{probe_id}\x00{text}"
        for probe_id, text in TARGET_OPTICAL_PROFILE_PROBE_BANK
    ),
):
    _target_optical_policy_digest.update(
        str(_target_optical_policy_part).encode("utf-8")
    )
    _target_optical_policy_digest.update(b"\x00")
TARGET_OPTICAL_PROFILE_POLICY_ID = (
    f"{TARGET_OPTICAL_PROFILE_POLICY_VERSION}:"
    f"{_target_optical_policy_digest.hexdigest()}"
)
LATIN_TARGET_OPTICAL_PROFILE_POLICY_VERSION = (
    "fixed_disjoint_latin_probe_bank_v1"
)
LATIN_TARGET_OPTICAL_PROFILE_PROBE_BANK = (
    ("latin_uppercase_single", "ABCDEFGHJKLMNPRSTUVWXYZ"),
    ("latin_lowercase_single", "abcdefghijklmnopqrstuvwxyz"),
    ("latin_digits_single", "0123456789"),
)
_latin_target_optical_policy_digest = hashlib.sha256()
for _latin_target_optical_policy_part in (
    TARGET_OPTICAL_PROFILE_VERSION,
    LATIN_TARGET_OPTICAL_PROFILE_POLICY_VERSION,
    TARGET_OPTICAL_PROFILE_AGGREGATION,
    str(TARGET_OPTICAL_PROFILE_REFERENCE_EM_PX),
    *(
        f"{probe_id}\x00{text}"
        for probe_id, text in LATIN_TARGET_OPTICAL_PROFILE_PROBE_BANK
    ),
):
    _latin_target_optical_policy_digest.update(
        str(_latin_target_optical_policy_part).encode("utf-8")
    )
    _latin_target_optical_policy_digest.update(b"\x00")
LATIN_TARGET_OPTICAL_PROFILE_POLICY_ID = (
    f"{LATIN_TARGET_OPTICAL_PROFILE_POLICY_VERSION}:"
    f"{_latin_target_optical_policy_digest.hexdigest()}"
)
DEFAULT_FALLBACK_CHAIN = "cjk-sc"

SYMBOL_FALLBACK_CHARS = ("☆", "★", "♡", "❤", "♪")
VERTICAL_PUNCTUATION_FORMS = {
    "!": ("︕", "！"),
    "！": ("︕", "！"),
    "!!": ("‼", "︕︕", "！！"),
    "！！": ("‼", "︕︕", "！！"),
    "?": ("︖", "？"),
    "？": ("︖", "？"),
    "??": ("⁇", "︖︖", "？？"),
    "？？": ("⁇", "︖︖", "？？"),
    "!?": ("⁉", "︕︖", "！？"),
    "！？": ("⁉", "︕︖", "！？"),
    "?!": ("⁈", "︖︕", "？！"),
    "？！": ("⁈", "︖︕", "？！"),
    ",": ("︐", "，"),
    "，": ("︐", "，"),
    "、": ("︑", "、"),
    "。": ("︒", "。"),
    ":": ("︓", "："),
    "：": ("︓", "："),
    ";": ("︔", "；"),
    "；": ("︔", "；"),
    ".": ("︒", "．", "."),
    "...": ("…", "︙"),
    "…": ("…", "︙"),
    "……": ("……", "︙︙"),
    "-": ("︱",),
    "—": ("︱",),
    "―": ("︱",),
    "－": ("︱",),
    "--": ("︱︱",),
    "——": ("︱︱",),
    "~": ("~", "︴"),
    "～": ("～", "︴"),
    "〜": ("〜", "︴"),
    "〰": ("〰", "︴"),
    "︴": ("︴",),
}

REQUIRED_FONT_ROLES = (
    ("sans_regular", "NotoSansCJKsc-Regular.otf", "noto_sans_cjk_sc_regular", ""),
    ("sans_medium", "NotoSansCJKsc-Medium.otf", "noto_sans_cjk_sc_medium", "noto_sans_cjk_sc_regular"),
    ("sans_bold", "NotoSansCJKsc-Bold.otf", "noto_sans_cjk_sc_bold", "noto_sans_cjk_sc_medium"),
    ("sans_black", "NotoSansCJKsc-Black.otf", "noto_sans_cjk_sc_black", "noto_sans_cjk_sc_bold"),
    ("serif_regular", "NotoSerifCJKsc-Regular.otf", "noto_serif_cjk_sc_regular", ""),
    ("serif_semibold", "NotoSerifCJKsc-SemiBold.otf", "noto_serif_cjk_sc_semibold", "noto_serif_cjk_sc_bold"),
    ("serif_bold", "NotoSerifCJKsc-Bold.otf", "noto_serif_cjk_sc_bold", "noto_serif_cjk_sc_semibold"),
    ("mono_regular", "NotoSansMonoCJKsc-Regular.otf", "noto_sans_mono_cjk_sc_regular", "noto_sans_cjk_sc_regular"),
    ("latin_sans_regular", "NotoSans[wdth,wght].ttf", "noto_sans_latin_condensed_regular", "noto_sans_cjk_sc_regular"),
    ("latin_sans_medium", "NotoSans[wdth,wght].ttf", "noto_sans_latin_condensed_medium", "noto_sans_cjk_sc_medium"),
    ("latin_sans_bold", "NotoSans[wdth,wght].ttf", "noto_sans_latin_condensed_bold", "noto_sans_cjk_sc_bold"),
    ("latin_sans_black", "NotoSans[wdth,wght].ttf", "noto_sans_latin_condensed_black", "noto_sans_cjk_sc_black"),
)

LATIN_TARGET_ROLE_MAP = {
    "sans_regular": "latin_sans_regular",
    "sans_medium": "latin_sans_medium",
    "sans_bold": "latin_sans_bold",
    "sans_black": "latin_sans_black",
}


class FontManagerError(RuntimeError):
    """Raised when font metrics are requested without a usable font."""


@dataclass(frozen=True)
class FontFace:
    face_id: str
    family: str
    style_class: str
    weight: str
    path: str
    source: str = "noto_cjk_sc_core"
    serif: bool = False
    monospace: bool = False
    priority: int = 100
    variations: tuple[tuple[str, float], ...] = ()

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "font_manager_version": FONT_MANAGER_VERSION,
            "face_id": self.face_id,
            "family": self.family,
            "style_class": self.style_class,
            "weight": self.weight,
            "path": self.path,
            "source": self.source,
            "serif": bool(self.serif),
            "monospace": bool(self.monospace),
            "priority": int(self.priority),
            "variations": {
                str(tag): float(value) for tag, value in self.variations
            },
        }


@dataclass
class GlyphCoverage:
    face_id: str
    font_path: str
    text: str
    supported_chars: list[str] = field(default_factory=list)
    missing_chars: list[str] = field(default_factory=list)
    ignored_chars: list[str] = field(default_factory=list)

    @property
    def supports_text(self) -> bool:
        return not self.missing_chars

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "font_manager_version": FONT_MANAGER_VERSION,
            "face_id": self.face_id,
            "font_path": self.font_path,
            "text": self.text,
            "supports_text": self.supports_text,
            "supported_chars": list(self.supported_chars),
            "missing_chars": list(self.missing_chars),
            "ignored_chars": list(self.ignored_chars),
        }


@dataclass
class FontResolution:
    requested_family: str
    requested_weight: str
    style_class: str
    fallback_chain_key: str
    writing_mode: str
    primary_face: FontFace | None
    fallback_faces: list[FontFace] = field(default_factory=list)
    missing_glyphs: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    registered_role: FontRoleStatus | None = None
    logical_role_id: str = ""
    physical_role_id: str = ""

    @property
    def usable(self) -> bool:
        return self.primary_face is not None

    def to_audit_dict(self) -> dict[str, Any]:
        audit = {
            "font_manager_version": FONT_MANAGER_VERSION,
            "requested_family": self.requested_family,
            "requested_weight": self.requested_weight,
            "style_class": self.style_class,
            "fallback_chain_key": self.fallback_chain_key,
            "writing_mode": self.writing_mode,
            "usable": self.usable,
            "primary_face": self.primary_face.to_audit_dict() if self.primary_face else None,
            "fallback_faces": [face.to_audit_dict() for face in self.fallback_faces],
            "missing_glyphs": list(self.missing_glyphs),
            "issues": list(self.issues),
            "registered_role": (
                self.registered_role.to_audit_dict()
                if self.registered_role is not None
                else None
            ),
            "logical_role_id": self.logical_role_id,
            "physical_role_id": self.physical_role_id,
        }
        return audit


@dataclass
class RunFontResolution:
    run_id: str
    text: str
    selected_face: FontFace | None
    coverage: GlyphCoverage
    fallback_used: bool = False
    fallback_index: int = 0
    selection_reason: str = ""
    missing_glyphs: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "font_manager_version": FONT_MANAGER_VERSION,
            "run_id": self.run_id,
            "text": self.text,
            "selected_face": self.selected_face.to_audit_dict() if self.selected_face else None,
            "coverage": self.coverage.to_audit_dict(),
            "fallback_used": bool(self.fallback_used),
            "fallback_index": int(self.fallback_index),
            "selection_reason": self.selection_reason,
            "missing_glyphs": list(self.missing_glyphs),
            "issues": list(self.issues),
        }


@dataclass
class FontSpanResolution:
    """One face-owned, extended-grapheme-aligned span of a logical run."""

    logical_run_id: str
    span_id: str
    text: str
    source_grapheme_start: int
    source_grapheme_end: int
    source_codepoint_start: int
    source_codepoint_end: int
    selected_face: FontFace | None
    coverage: GlyphCoverage
    fallback_used: bool = False
    fallback_index: int = 0
    selection_reason: str = ""
    missing_clusters: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    primary_face_id: str = ""
    face_authority: str = ""

    @property
    def usable(self) -> bool:
        return self.selected_face is not None and self.coverage.supports_text and not self.issues

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "font_manager_version": FONT_MANAGER_VERSION,
            "logical_run_id": self.logical_run_id,
            "span_id": self.span_id,
            "text": self.text,
            "source_grapheme_start": int(self.source_grapheme_start),
            "source_grapheme_end": int(self.source_grapheme_end),
            "source_codepoint_start": int(self.source_codepoint_start),
            "source_codepoint_end": int(self.source_codepoint_end),
            "selected_face": self.selected_face.to_audit_dict() if self.selected_face else None,
            "coverage": self.coverage.to_audit_dict(),
            "fallback_used": bool(self.fallback_used),
            "fallback_index": int(self.fallback_index),
            "selection_reason": self.selection_reason,
            "missing_clusters": list(self.missing_clusters),
            "issues": list(self.issues),
            "primary_face_id": self.primary_face_id,
            "face_authority": self.face_authority,
            "usable": self.usable,
        }


@dataclass(frozen=True)
class OpenTypeMetrics:
    face_id: str
    font_path: str
    font_size: int
    units_per_em: int
    ascender: int
    descender: int
    line_gap: int
    typo_ascender: int
    typo_descender: int
    typo_line_gap: int
    cap_height: int
    x_height: int
    has_vertical_metrics: bool
    vertical_ascender: int
    vertical_descender: int
    vertical_line_gap: int
    scaled_ascender: float
    scaled_descender: float
    scaled_line_gap: float
    scaled_line_height: float
    source: str = "opentype_tables"

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "font_manager_version": FONT_MANAGER_VERSION,
            "face_id": self.face_id,
            "font_path": self.font_path,
            "font_size": int(self.font_size),
            "units_per_em": int(self.units_per_em),
            "ascender": int(self.ascender),
            "descender": int(self.descender),
            "line_gap": int(self.line_gap),
            "typo_ascender": int(self.typo_ascender),
            "typo_descender": int(self.typo_descender),
            "typo_line_gap": int(self.typo_line_gap),
            "cap_height": int(self.cap_height),
            "x_height": int(self.x_height),
            "has_vertical_metrics": bool(self.has_vertical_metrics),
            "vertical_ascender": int(self.vertical_ascender),
            "vertical_descender": int(self.vertical_descender),
            "vertical_line_gap": int(self.vertical_line_gap),
            "scaled_ascender": float(self.scaled_ascender),
            "scaled_descender": float(self.scaled_descender),
            "scaled_line_gap": float(self.scaled_line_gap),
            "scaled_line_height": float(self.scaled_line_height),
            "source": self.source,
        }


@dataclass(frozen=True)
class TargetOpticalGlyphSelection:
    """Per-request glyph evidence kept outside the cached face profile."""

    glyph_set: tuple[str, ...]
    glyph_set_sha256: str
    evidence_source: str
    fallback_probe_used: bool

    @property
    def glyph_count(self) -> int:
        return len(self.glyph_set)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "glyph_set": list(self.glyph_set),
            "glyph_set_sha256": self.glyph_set_sha256,
            "glyph_count": self.glyph_count,
            "evidence_source": self.evidence_source,
            "fallback_probe_used": bool(self.fallback_probe_used),
        }


@dataclass(frozen=True)
class TargetFontOpticalProfile:
    """Fixed registered-face metrics independent of translated glyph content."""

    profile_id: str
    face_id: str
    font_path: str
    font_sha256: str
    profile_policy_id: str
    reference_em_px: int
    writing_mode: str
    probe_ids: tuple[str, ...]
    glyph_set: tuple[str, ...]
    glyph_set_sha256: str
    visible_ink_height_px: float
    visible_ink_height_ratio: float
    advance_px: float
    advance_to_cell_ratio: float
    stem_width_px: float
    stem_to_ink_ratio: float
    ink_coverage_ratio: float
    measurement_source: str = "pillow_raster_opencv_distance_transform"

    @property
    def glyph_count(self) -> int:
        return len(self.glyph_set)

    @property
    def cache_key(self) -> tuple[str, str, str]:
        return (
            self.font_sha256,
            self.profile_policy_id,
            self.writing_mode,
        )

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "font_manager_version": FONT_MANAGER_VERSION,
            "target_optical_profile_version": TARGET_OPTICAL_PROFILE_VERSION,
            "profile_id": self.profile_id,
            "face_id": self.face_id,
            "font_path": self.font_path,
            "font_sha256": self.font_sha256,
            "profile_policy_id": self.profile_policy_id,
            "reference_em_px": int(self.reference_em_px),
            "writing_mode": self.writing_mode,
            "probe_ids": list(self.probe_ids),
            "probe_count": len(self.probe_ids),
            "glyph_set": list(self.glyph_set),
            "glyph_set_sha256": self.glyph_set_sha256,
            "glyph_count": self.glyph_count,
            "visible_ink_height_px": float(self.visible_ink_height_px),
            "visible_ink_height_ratio": float(self.visible_ink_height_ratio),
            "advance_px": float(self.advance_px),
            "advance_to_cell_ratio": float(self.advance_to_cell_ratio),
            "stem_width_px": float(self.stem_width_px),
            "stem_to_ink_ratio": float(self.stem_to_ink_ratio),
            "ink_coverage_ratio": float(self.ink_coverage_ratio),
            "measurement_source": self.measurement_source,
            "cache_key": list(self.cache_key),
        }


@dataclass(frozen=True)
class TargetFontOpticalProfileResolution:
    """One request's evidence provenance plus its shared cached metrics."""

    selection: TargetOpticalGlyphSelection
    profile: TargetFontOpticalProfile

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "target_optical_profile_version": TARGET_OPTICAL_PROFILE_VERSION,
            "selection": self.selection.to_audit_dict(),
            "profile": self.profile.to_audit_dict(),
        }


@dataclass(frozen=True)
class FontRoleStatus:
    role_id: str
    preferred_filename: str
    native_face_id: str
    selected_face_id: str
    native_asset_available: bool
    substitute_face_id: str = ""
    substitution_reason: str = ""

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "font_manager_version": FONT_MANAGER_VERSION,
            "role_id": self.role_id,
            "preferred_filename": self.preferred_filename,
            "native_face_id": self.native_face_id,
            "selected_face_id": self.selected_face_id,
            "native_asset_available": bool(self.native_asset_available),
            "substitute_face_id": self.substitute_face_id,
            "substitution_reason": self.substitution_reason,
        }


@dataclass
class GlyphMetrics:
    face_id: str
    font_path: str
    font_size: int
    glyph: str
    bbox: list[int]
    width: int
    height: int
    advance: float

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "font_manager_version": FONT_MANAGER_VERSION,
            "face_id": self.face_id,
            "font_path": self.font_path,
            "font_size": int(self.font_size),
            "glyph": self.glyph,
            "bbox": list(self.bbox),
            "width": int(self.width),
            "height": int(self.height),
            "advance": float(self.advance),
        }


@dataclass
class TextMetrics:
    face_id: str
    font_path: str
    font_size: int
    writing_mode: str
    text: str
    bbox: list[int]
    width: int
    height: int
    advance: float
    ascent: int
    descent: int
    glyphs: list[GlyphMetrics] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "font_manager_version": FONT_MANAGER_VERSION,
            "face_id": self.face_id,
            "font_path": self.font_path,
            "font_size": int(self.font_size),
            "writing_mode": self.writing_mode,
            "text": self.text,
            "bbox": list(self.bbox),
            "width": int(self.width),
            "height": int(self.height),
            "advance": float(self.advance),
            "ascent": int(self.ascent),
            "descent": int(self.descent),
            "glyphs": [glyph.to_audit_dict() for glyph in self.glyphs],
        }


@dataclass
class VerticalPunctuationSupport:
    source: str
    candidate_forms: list[str]
    supported_forms: list[str]
    selected_form: str = ""

    @property
    def supported(self) -> bool:
        return bool(self.selected_form)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "font_manager_version": FONT_MANAGER_VERSION,
            "source": self.source,
            "candidate_forms": list(self.candidate_forms),
            "supported_forms": list(self.supported_forms),
            "selected_form": self.selected_form,
            "supported": self.supported,
        }


class FontManager:
    """Registry and metrics owner for the new typesetting path."""

    def __init__(
        self,
        *,
        base_dir: str | None = None,
    ) -> None:
        self.base_dir = base_dir
        self._faces: dict[str, FontFace] = {}
        self._cmap_cache: dict[str, set[int]] = {}
        self._font_cache: dict[tuple[str, str, int], Any] = {}
        self._glyph_metrics_cache: dict[tuple[str, str, int, str], GlyphMetrics] = {}
        self._text_metrics_cache: dict[tuple[str, str, int, str, str], TextMetrics] = {}
        self._open_type_metrics_cache: dict[tuple[str, str, int], OpenTypeMetrics] = {}
        self._font_sha256_cache: dict[str, str] = {}
        self._target_optical_profile_cache: dict[
            tuple[str, str, str], TargetFontOpticalProfile
        ] = {}
        self._cache_hits = {
            "font": 0,
            "glyph_metrics": 0,
            "text_metrics": 0,
            "open_type_metrics": 0,
            "target_optical_profile": 0,
            "cmap": 0,
        }
        self._cache_misses = {
            "font": 0,
            "glyph_metrics": 0,
            "text_metrics": 0,
            "open_type_metrics": 0,
            "target_optical_profile": 0,
            "cmap": 0,
        }
        self._register_noto_cjk_sc_core()
        self._register_noto_latin_core()
        self._register_windows_fallbacks()

    @property
    def has_font_pack(self) -> bool:
        return model_resolution.has_noto_cjk_sc_font_pack(self.base_dir)

    @property
    def has_latin_font_pack(self) -> bool:
        return model_resolution.has_noto_latin_font_pack(self.base_dir)

    def available_faces(self) -> list[FontFace]:
        return sorted(self._faces.values(), key=lambda face: (face.priority, face.face_id))

    def face(self, face_id: str) -> FontFace | None:
        return self._faces.get(str(face_id or ""))

    def registered_face(self, face: Any) -> FontFace | None:
        """Return the canonical registry face for an exact face identity."""

        if face is None:
            return None
        face_id = str(getattr(face, "face_id", "") or "").strip()
        path = str(getattr(face, "path", "") or "").strip()
        registered = self.face(face_id)
        if registered is None or not path:
            return None
        if _canonical_font_path(path) != _canonical_font_path(registered.path):
            return None
        return registered

    def _registered_resolution_chain(
        self,
        resolution: FontResolution,
    ) -> list[FontFace]:
        primary = self.registered_face(resolution.primary_face)
        if primary is None:
            return []
        chain = [primary]
        for candidate in resolution.fallback_faces:
            registered = self.registered_face(candidate)
            if registered is None:
                continue
            if all(registered.face_id != item.face_id for item in chain):
                chain.append(registered)
        return chain

    def resolve_font(
        self,
        resolved_style: Mapping[str, Any] | None = None,
        *,
        fallback_chain_key: str = DEFAULT_FALLBACK_CHAIN,
        writing_mode: str = "vertical",
        text: str = "",
    ) -> FontResolution:
        style = resolved_style if isinstance(resolved_style, Mapping) else {}
        primary_font_role = str(style.get("primary_font_role") or "").strip()
        physical_font_role = _target_physical_font_role(
            primary_font_role,
            target_script=str(style.get("target_script") or ""),
            writing_mode=writing_mode,
        )
        role_inventory = {
            item.role_id: item for item in self.required_role_inventory()
        }
        role_status = role_inventory.get(physical_font_role)
        if (
            physical_font_role in set(LATIN_TARGET_ROLE_MAP.values())
            and (
                role_status is None
                or not bool(role_status.native_asset_available)
            )
        ):
            return FontResolution(
                requested_family=primary_font_role,
                requested_weight="",
                style_class="",
                fallback_chain_key=fallback_chain_key,
                writing_mode=writing_mode,
                primary_face=None,
                fallback_faces=[],
                missing_glyphs=list(_unique_chars(text)),
                issues=["missing_latin_font_pack"],
                registered_role=role_status,
                logical_role_id=primary_font_role,
                physical_role_id=physical_font_role,
            )
        selected = (
            self.face(role_status.selected_face_id)
            if role_status is not None and role_status.selected_face_id
            else None
        )
        requested_family = selected.family if selected is not None else primary_font_role
        requested_weight = selected.weight if selected is not None else ""
        style_class = selected.style_class if selected is not None else ""
        if selected is None:
            return FontResolution(
                requested_family=requested_family,
                requested_weight=requested_weight,
                style_class=style_class,
                fallback_chain_key=fallback_chain_key,
                writing_mode=writing_mode,
                primary_face=None,
                fallback_faces=[],
                missing_glyphs=list(_unique_chars(text)),
                issues=["registered_primary_font_role_unavailable"],
                registered_role=role_status,
                logical_role_id=primary_font_role,
                physical_role_id=physical_font_role,
            )
        chain = self._fallback_chain(
            requested_family=requested_family,
            requested_weight=requested_weight,
            style_class=style_class,
            fallback_chain_key=fallback_chain_key,
        )
        chain = [selected, *[face for face in chain if face.face_id != selected.face_id]]
        issues: list[str] = []
        style_faces = [face for face in chain if face.source != "windows_system_font"]
        if not style_faces:
            return FontResolution(
                requested_family=requested_family,
                requested_weight=requested_weight,
                style_class=style_class,
                fallback_chain_key=fallback_chain_key,
                writing_mode=writing_mode,
                primary_face=None,
                fallback_faces=[],
                missing_glyphs=list(_unique_chars(text)),
                issues=["missing_font_pack"],
                registered_role=role_status,
                logical_role_id=primary_font_role,
                physical_role_id=physical_font_role,
            )
        missing_glyphs: list[str] = []
        if text:
            for char in _unique_chars(text):
                if _ignore_coverage_char(char):
                    continue
                if not any(self.coverage_for_text(face, char).supports_text for face in chain):
                    missing_glyphs.append(char)
            if missing_glyphs:
                issues.append("missing_glyphs")
        core_resolution = FontResolution(
            requested_family=requested_family,
            requested_weight=requested_weight,
            style_class=style_class,
            fallback_chain_key=fallback_chain_key,
            writing_mode=writing_mode,
            primary_face=selected,
            fallback_faces=[face for face in chain if face.face_id != selected.face_id],
            missing_glyphs=missing_glyphs,
            issues=issues,
            registered_role=role_status,
            logical_role_id=primary_font_role,
            physical_role_id=physical_font_role,
        )
        return core_resolution

    def resolve_run_font(
        self,
        resolution: FontResolution,
        text: str,
        *,
        run_id: str = "",
    ) -> RunFontResolution:
        value = str(text or "")
        chain = self._registered_resolution_chain(resolution)
        if not chain:
            coverage = GlyphCoverage(
                face_id="",
                font_path="",
                text=value,
                missing_chars=[char for char in _unique_chars(value) if not _ignore_coverage_char(char)],
            )
            return RunFontResolution(
                run_id=str(run_id or ""),
                text=value,
                selected_face=None,
                coverage=coverage,
                fallback_used=False,
                fallback_index=-1,
                selection_reason="registered_primary_face_unavailable",
                missing_glyphs=list(coverage.missing_chars),
                issues=["registered_primary_face_unavailable"],
            )

        for index, face in enumerate(chain):
            coverage = self.coverage_for_text(face, value)
            if coverage.supports_text:
                return RunFontResolution(
                    run_id=str(run_id or ""),
                    text=value,
                    selected_face=face,
                    coverage=coverage,
                    fallback_used=index > 0,
                    fallback_index=index,
                    selection_reason=(
                        "registered_primary_role_full_run_coverage"
                        if index == 0
                        else "registered_coverage_fallback_full_run"
                    ),
                )

        primary = chain[0]
        coverage = self.coverage_for_text(primary, value)
        unresolved = [
            char
            for char in _unique_chars(value)
            if not _ignore_coverage_char(char)
            and not any(self.coverage_for_text(face, char).supports_text for face in chain)
        ]
        issues = ["missing_glyphs"] if unresolved else ["run_requires_face_segmentation"]
        return RunFontResolution(
            run_id=str(run_id or ""),
            text=value,
            selected_face=primary,
            coverage=coverage,
            fallback_used=False,
            fallback_index=0,
            selection_reason="no_single_face_covers_complete_run",
            missing_glyphs=unresolved or list(coverage.missing_chars),
            issues=issues,
        )

    def resolve_run_font_spans(
        self,
        resolution: FontResolution,
        text: str,
        *,
        run_id: str = "",
        script: str = "",
        direction: str = "",
        role: str = "",
        writing_mode: str = "horizontal",
    ) -> list[FontSpanResolution]:
        """Realize one registered primary face plus coverage-only fallbacks.

        Primary-covered graphemes always retain the arbitrated registered role.
        A fallback may own only an extended grapheme that the primary cannot
        cover. Contextual and semantic atomic runs use one complete registered
        face or report an explicit unresolved span.
        """

        value = str(text or "")
        logical_run_id = str(run_id or "")
        clusters = strict_grapheme_clusters(value)
        codepoint_offsets = _cluster_codepoint_offsets(clusters)
        chain = self._registered_resolution_chain(resolution)
        primary = chain[0] if chain else None
        primary_face_id = primary.face_id if primary is not None else ""

        if primary is None:
            return [
                _unresolved_font_span(
                    logical_run_id=logical_run_id,
                    span_id=logical_run_id,
                    text=value,
                    grapheme_start=0,
                    grapheme_end=len(clusters),
                    codepoint_start=0,
                    codepoint_end=len(value),
                    missing_clusters=list(clusters),
                    issue="registered_primary_face_unavailable",
                    primary_face_id=primary_face_id,
                )
            ]

        primary_coverage = self.coverage_for_text(primary, value)
        if primary_coverage.supports_text:
            return [
                FontSpanResolution(
                    logical_run_id=logical_run_id,
                    span_id=logical_run_id,
                    text=value,
                    source_grapheme_start=0,
                    source_grapheme_end=len(clusters),
                    source_codepoint_start=0,
                    source_codepoint_end=len(value),
                    selected_face=primary,
                    coverage=primary_coverage,
                    fallback_used=False,
                    fallback_index=0,
                    selection_reason="registered_primary_role_full_run_coverage",
                    missing_clusters=[],
                    issues=[],
                    primary_face_id=primary_face_id,
                    face_authority="registered_primary_role",
                )
            ]

        must_remain_atomic = _font_run_must_remain_atomic(
            script=script,
            direction=direction,
            role=role,
            writing_mode=writing_mode,
        )
        if must_remain_atomic or len(clusters) <= 1:
            fallback_chain = _preferred_sequence_chain(chain[1:], value)
            selected = _first_covering_face(self, fallback_chain, value)
            if selected is not None:
                _, face, coverage = selected
                fallback_index = next(
                    index
                    for index, item in enumerate(chain)
                    if item.face_id == face.face_id
                )
                return [
                    FontSpanResolution(
                        logical_run_id=logical_run_id,
                        span_id=logical_run_id,
                        text=value,
                        source_grapheme_start=0,
                        source_grapheme_end=len(clusters),
                        source_codepoint_start=0,
                        source_codepoint_end=len(value),
                        selected_face=face,
                        coverage=coverage,
                        fallback_used=True,
                        fallback_index=fallback_index,
                        selection_reason=(
                            "registered_coverage_fallback_full_atomic_run"
                        ),
                        missing_clusters=[],
                        issues=[],
                        primary_face_id=primary_face_id,
                        face_authority="registered_glyph_coverage_fallback",
                    )
                ]
            return [
                _unresolved_font_span(
                    logical_run_id=logical_run_id,
                    span_id=logical_run_id,
                    text=value,
                    grapheme_start=0,
                    grapheme_end=len(clusters),
                    codepoint_start=0,
                    codepoint_end=len(value),
                    missing_clusters=list(clusters),
                    issue=(
                        "atomic_run_requires_single_covering_face"
                        if must_remain_atomic
                        else "missing_glyphs"
                    ),
                    primary_face_id=primary_face_id,
                )
            ]

        assignments: list[tuple[str, FontFace | None, int]] = []
        for cluster in clusters:
            if self.coverage_for_text(primary, cluster).supports_text:
                assignments.append((cluster, primary, 0))
                continue
            fallback_chain = _preferred_sequence_chain(chain[1:], cluster)
            selected = _first_covering_face(self, fallback_chain, cluster)
            if selected is None:
                assignments.append((cluster, None, -1))
                continue
            _, face, _ = selected
            fallback_index = next(
                index
                for index, item in enumerate(chain)
                if item.face_id == face.face_id
            )
            assignments.append((cluster, face, fallback_index))

        groups: list[tuple[int, int, FontFace | None, int]] = []
        group_start = 0
        group_face = assignments[0][1] if assignments else None
        group_index = assignments[0][2] if assignments else -1
        for index, (_, face, fallback_index) in enumerate(assignments[1:], start=1):
            face_id = face.face_id if face is not None else ""
            group_face_id = group_face.face_id if group_face is not None else ""
            if face_id != group_face_id or fallback_index != group_index:
                groups.append((group_start, index, group_face, group_index))
                group_start = index
                group_face = face
                group_index = fallback_index
        if assignments:
            groups.append((group_start, len(assignments), group_face, group_index))

        spans: list[FontSpanResolution] = []
        for span_index, (start, end, face, fallback_index) in enumerate(groups):
            span_text = "".join(clusters[start:end])
            codepoint_start = codepoint_offsets[start]
            codepoint_end = codepoint_offsets[end]
            span_id = (
                logical_run_id
                if len(groups) == 1
                else f"{logical_run_id}:fs{span_index:03d}"
            )
            if face is None:
                spans.append(
                    _unresolved_font_span(
                        logical_run_id=logical_run_id,
                        span_id=span_id,
                        text=span_text,
                        grapheme_start=start,
                        grapheme_end=end,
                        codepoint_start=codepoint_start,
                        codepoint_end=codepoint_end,
                        missing_clusters=list(clusters[start:end]),
                        issue="missing_glyphs",
                        primary_face_id=primary_face_id,
                    )
                )
                continue
            coverage = self.coverage_for_text(face, span_text)
            spans.append(
                FontSpanResolution(
                    logical_run_id=logical_run_id,
                    span_id=span_id,
                    text=span_text,
                    source_grapheme_start=start,
                    source_grapheme_end=end,
                    source_codepoint_start=codepoint_start,
                    source_codepoint_end=codepoint_end,
                    selected_face=face,
                    coverage=coverage,
                    fallback_used=fallback_index > 0,
                    fallback_index=fallback_index,
                    selection_reason=(
                        "registered_coverage_fallback_grapheme_span"
                        if fallback_index > 0
                        else "registered_primary_role_grapheme_span"
                    ),
                    primary_face_id=primary_face_id,
                    face_authority=(
                        "registered_glyph_coverage_fallback"
                        if fallback_index > 0
                        else "registered_primary_role"
                    ),
                )
            )
        return spans

    def coverage_for_text(self, face: FontFace | None, text: str) -> GlyphCoverage:
        if face is None:
            raise FontManagerError("font face is unavailable")
        cmap = self._font_cmap(face.path)
        supported: list[str] = []
        missing: list[str] = []
        ignored: list[str] = []
        for char in _unique_chars(text):
            if _ignore_coverage_char(char):
                ignored.append(char)
                continue
            if ord(char) in cmap:
                supported.append(char)
            else:
                missing.append(char)
        return GlyphCoverage(
            face_id=face.face_id,
            font_path=face.path,
            text=str(text or ""),
            supported_chars=supported,
            missing_chars=missing,
            ignored_chars=ignored,
        )

    def symbol_coverage(
        self,
        face: FontFace | None,
        symbols: Sequence[str] = SYMBOL_FALLBACK_CHARS,
    ) -> dict[str, bool]:
        coverage = self.coverage_for_text(face, "".join(symbols))
        missing = set(coverage.missing_chars)
        return {symbol: symbol not in missing for symbol in symbols}

    def vertical_punctuation_support(
        self,
        face: FontFace | None,
        source: str,
    ) -> VerticalPunctuationSupport:
        candidates = list(VERTICAL_PUNCTUATION_FORMS.get(str(source or ""), (str(source or ""),)))
        supported: list[str] = []
        for candidate in candidates:
            if self.coverage_for_text(face, candidate).supports_text:
                supported.append(candidate)
        return VerticalPunctuationSupport(
            source=str(source or ""),
            candidate_forms=candidates,
            supported_forms=supported,
            selected_form=supported[0] if supported else "",
        )

    def load_font(self, face: FontFace, size: int):
        if face is None:
            raise FontManagerError("font face is unavailable")
        if ImageFont is None:
            raise FontManagerError("Pillow ImageFont is unavailable")
        font_size = max(1, int(size))
        key = (face.face_id, face.path, font_size)
        if key in self._font_cache:
            self._cache_hits["font"] += 1
            return self._font_cache[key]
        self._cache_misses["font"] += 1
        font = ImageFont.truetype(face.path, font_size)
        _apply_pillow_font_variations(font, face.variations)
        self._font_cache[key] = font
        return font

    def glyph_metrics(self, face: FontFace, glyph: str, size: int) -> GlyphMetrics:
        if face is None:
            raise FontManagerError("font face is unavailable")
        font_size = max(1, int(size))
        key = (face.face_id, face.path, font_size, str(glyph or ""))
        if key in self._glyph_metrics_cache:
            self._cache_hits["glyph_metrics"] += 1
            return self._glyph_metrics_cache[key]
        self._cache_misses["glyph_metrics"] += 1
        font = self.load_font(face, font_size)
        text = str(glyph or "")
        bbox_tuple = font.getbbox(text) if text else (0, 0, 0, 0)
        bbox = [int(value) for value in bbox_tuple]
        width = max(0, bbox[2] - bbox[0])
        height = max(0, bbox[3] - bbox[1])
        try:
            advance = float(font.getlength(text))
        except Exception:
            advance = float(width)
        metrics = GlyphMetrics(
            face_id=face.face_id,
            font_path=face.path,
            font_size=font_size,
            glyph=text,
            bbox=bbox,
            width=width,
            height=height,
            advance=advance,
        )
        self._glyph_metrics_cache[key] = metrics
        return metrics

    def measure_text(
        self,
        face: FontFace,
        text: str,
        *,
        size: int,
        writing_mode: str = "horizontal",
    ) -> TextMetrics:
        if face is None:
            raise FontManagerError("font face is unavailable")
        font_size = max(1, int(size))
        mode = str(writing_mode or "horizontal").strip().lower() or "horizontal"
        key = (face.face_id, face.path, font_size, mode, str(text or ""))
        if key in self._text_metrics_cache:
            self._cache_hits["text_metrics"] += 1
            return self._text_metrics_cache[key]
        self._cache_misses["text_metrics"] += 1
        font = self.load_font(face, font_size)
        try:
            ascent, descent = font.getmetrics()
        except Exception:
            ascent, descent = 0, 0
        text_value = str(text or "")
        glyphs = [self.glyph_metrics(face, char, font_size) for char in text_value]
        if mode.startswith("vert"):
            width = max((glyph.width for glyph in glyphs), default=0)
            height = sum(max(1, glyph.height) for glyph in glyphs)
            advance = sum(glyph.advance for glyph in glyphs)
            bbox = [0, 0, width, height]
        else:
            bbox_tuple = font.getbbox(text_value) if text_value else (0, 0, 0, 0)
            bbox = [int(value) for value in bbox_tuple]
            width = max(0, bbox[2] - bbox[0])
            height = max(0, bbox[3] - bbox[1])
            try:
                advance = float(font.getlength(text_value))
            except Exception:
                advance = float(width)
        metrics = TextMetrics(
            face_id=face.face_id,
            font_path=face.path,
            font_size=font_size,
            writing_mode=mode,
            text=text_value,
            bbox=bbox,
            width=width,
            height=height,
            advance=advance,
            ascent=int(ascent),
            descent=int(descent),
            glyphs=glyphs,
        )
        self._text_metrics_cache[key] = metrics
        return metrics

    def open_type_metrics(self, face: FontFace, *, size: int) -> OpenTypeMetrics:
        if face is None:
            raise FontManagerError("font face is unavailable")
        if TTFont is None:
            raise FontManagerError("fontTools TTFont is unavailable")
        font_size = max(1, int(size))
        key = (face.face_id, face.path, font_size)
        if key in self._open_type_metrics_cache:
            self._cache_hits["open_type_metrics"] += 1
            return self._open_type_metrics_cache[key]
        self._cache_misses["open_type_metrics"] += 1
        font = None
        try:
            font = TTFont(face.path, fontNumber=0, lazy=True)
            head = font.get("head")
            hhea = font.get("hhea")
            os2 = font.get("OS/2")
            vhea = font.get("vhea")
            units_per_em = max(1, int(getattr(head, "unitsPerEm", 1000) or 1000))
            ascender = int(getattr(hhea, "ascent", 0) or 0)
            descender = int(getattr(hhea, "descent", 0) or 0)
            line_gap = int(getattr(hhea, "lineGap", 0) or 0)
            typo_ascender = int(getattr(os2, "sTypoAscender", ascender) or ascender)
            typo_descender = int(getattr(os2, "sTypoDescender", descender) or descender)
            typo_line_gap = int(getattr(os2, "sTypoLineGap", line_gap) or line_gap)
            cap_height = int(getattr(os2, "sCapHeight", 0) or 0)
            x_height = int(getattr(os2, "sxHeight", 0) or 0)
            vertical_ascender = int(getattr(vhea, "ascent", 0) or 0)
            vertical_descender = int(getattr(vhea, "descent", 0) or 0)
            vertical_line_gap = int(getattr(vhea, "lineGap", 0) or 0)
            scale = float(font_size) / float(units_per_em)
            scaled_ascender = float(ascender) * scale
            scaled_descender = float(descender) * scale
            scaled_line_gap = float(line_gap) * scale
            scaled_line_height = float(ascender - descender + line_gap) * scale
            metrics = OpenTypeMetrics(
                face_id=face.face_id,
                font_path=face.path,
                font_size=font_size,
                units_per_em=units_per_em,
                ascender=ascender,
                descender=descender,
                line_gap=line_gap,
                typo_ascender=typo_ascender,
                typo_descender=typo_descender,
                typo_line_gap=typo_line_gap,
                cap_height=cap_height,
                x_height=x_height,
                has_vertical_metrics=vhea is not None,
                vertical_ascender=vertical_ascender,
                vertical_descender=vertical_descender,
                vertical_line_gap=vertical_line_gap,
                scaled_ascender=round(scaled_ascender, 6),
                scaled_descender=round(scaled_descender, 6),
                scaled_line_gap=round(scaled_line_gap, 6),
                scaled_line_height=round(scaled_line_height, 6),
            )
        finally:
            if font is not None:
                font.close()
        self._open_type_metrics_cache[key] = metrics
        return metrics

    def target_optical_profile(
        self,
        face: FontFace,
        writing_mode: str,
        *,
        profile_key: str = "cjk",
    ) -> TargetFontOpticalProfileResolution:
        """Measure one fixed registered-face profile at the reference em.

        The returned value contains continuous realized metrics only. Source
        style tiers, target em selection, readability, fitting, and layout are
        deliberately outside this owner. Translation content is not an input.
        """

        registered = self.face(face.face_id) if face is not None else None
        if (
            registered is None
            or registered.path != face.path
            or registered.source != "noto_cjk_sc_core"
        ):
            raise FontManagerError("target optical profile requires a registered Noto CJK SC face")
        mode = _normalize_optical_writing_mode(writing_mode)
        normalized_profile_key = str(profile_key or "").strip().lower()
        if normalized_profile_key == "cjk":
            profile_policy_id = TARGET_OPTICAL_PROFILE_POLICY_ID
            evidence_source = "fixed_disjoint_cjk_probe_bank"
            measurement_source = (
                "fixed_disjoint_cjk_probe_bank_median_of_probe_medians_"
                "pillow_raster_opencv_distance_transform"
            )
            padded_measurement = False
        elif normalized_profile_key == "latin":
            profile_policy_id = LATIN_TARGET_OPTICAL_PROFILE_POLICY_ID
            evidence_source = "fixed_disjoint_latin_probe_bank"
            measurement_source = (
                "fixed_disjoint_latin_probe_bank_median_of_probe_medians_"
                "pillow_raster_padded_opencv_distance_transform"
            )
            padded_measurement = True
        else:
            raise FontManagerError(
                f"unsupported target optical profile key: {profile_key!r}"
            )
        font_sha256 = self._font_file_sha256(registered.path)
        key = (
            font_sha256,
            profile_policy_id,
            mode,
        )
        cached = self._target_optical_profile_cache.get(key)
        if cached is not None:
            self._cache_hits["target_optical_profile"] += 1
            return TargetFontOpticalProfileResolution(
                selection=TargetOpticalGlyphSelection(
                    glyph_set=cached.glyph_set,
                    glyph_set_sha256=cached.glyph_set_sha256,
                    evidence_source=evidence_source,
                    fallback_probe_used=False,
                ),
                profile=cached,
            )
        self._cache_misses["target_optical_profile"] += 1

        probe_glyph_sets = (
            _fixed_optical_probe_glyph_sets(self, registered)
            if normalized_profile_key == "cjk"
            else _fixed_latin_optical_probe_glyph_sets(self, registered)
        )
        glyph_set = tuple(
            sorted(
                {
                    glyph
                    for _, probe_glyphs in probe_glyph_sets
                    for glyph in probe_glyphs
                }
            )
        )
        if not glyph_set:
            raise FontManagerError(
                "registered target face has no usable fixed optical probe glyph"
            )
        glyph_set_sha256 = _optical_glyph_set_sha256(glyph_set)
        selection = TargetOpticalGlyphSelection(
            glyph_set=glyph_set,
            glyph_set_sha256=glyph_set_sha256,
            evidence_source=evidence_source,
            fallback_probe_used=False,
        )
        font = self.load_font(registered, TARGET_OPTICAL_PROFILE_REFERENCE_EM_PX)
        probe_measurements = tuple(
            tuple(
                _measure_optical_glyph(
                    font,
                    glyph,
                    writing_mode=mode,
                    pad_boundary=padded_measurement,
                )
                for glyph in probe_glyphs
            )
            for _, probe_glyphs in probe_glyph_sets
        )
        per_probe = tuple(
            {
                "visible_ink_height_px": statistics.median(
                    item["visible_ink_height_px"]
                    for item in measurements
                ),
                "advance_px": statistics.median(
                    item["advance_px"]
                    for item in measurements
                ),
                "stem_width_px": statistics.median(
                    item["stem_width_px"]
                    for item in measurements
                ),
                "stem_to_ink_ratio": statistics.median(
                    item["stem_width_px"]
                    for item in measurements
                )
                / statistics.median(
                    item["visible_ink_height_px"]
                    for item in measurements
                ),
                "ink_coverage_ratio": statistics.median(
                    item["ink_coverage_ratio"]
                    for item in measurements
                ),
            }
            for measurements in probe_measurements
        )
        visible_ink_height_px = statistics.median(
            item["visible_ink_height_px"] for item in per_probe
        )
        advance_px = statistics.median(
            item["advance_px"] for item in per_probe
        )
        stem_width_px = statistics.median(
            item["stem_width_px"] for item in per_probe
        )
        stem_to_ink_ratio = statistics.median(
            item["stem_to_ink_ratio"] for item in per_probe
        )
        ink_coverage_ratio = statistics.median(
            item["ink_coverage_ratio"] for item in per_probe
        )
        reference_em = float(TARGET_OPTICAL_PROFILE_REFERENCE_EM_PX)
        profile_digest = hashlib.sha256(
            "\x00".join(
                (
                    TARGET_OPTICAL_PROFILE_VERSION,
                    registered.face_id,
                    font_sha256,
                    profile_policy_id,
                    mode,
                )
            ).encode("utf-8")
        ).hexdigest()
        profile = TargetFontOpticalProfile(
            profile_id=f"{TARGET_OPTICAL_PROFILE_VERSION}:{profile_digest}",
            face_id=registered.face_id,
            font_path=registered.path,
            font_sha256=font_sha256,
            profile_policy_id=profile_policy_id,
            reference_em_px=TARGET_OPTICAL_PROFILE_REFERENCE_EM_PX,
            writing_mode=mode,
            probe_ids=tuple(
                probe_id for probe_id, _ in probe_glyph_sets
            ),
            glyph_set=glyph_set,
            glyph_set_sha256=glyph_set_sha256,
            visible_ink_height_px=round(float(visible_ink_height_px), 6),
            visible_ink_height_ratio=round(
                float(visible_ink_height_px) / reference_em, 8
            ),
            advance_px=round(float(advance_px), 6),
            advance_to_cell_ratio=round(float(advance_px) / reference_em, 8),
            stem_width_px=round(float(stem_width_px), 6),
            stem_to_ink_ratio=round(
                float(stem_to_ink_ratio), 8
            ),
            ink_coverage_ratio=round(float(ink_coverage_ratio), 8),
            measurement_source=measurement_source,
        )
        self._target_optical_profile_cache[key] = profile
        return TargetFontOpticalProfileResolution(
            selection=selection,
            profile=profile,
        )

    def _font_file_sha256(self, path: str) -> str:
        normalized = os.path.abspath(str(path or ""))
        cached = self._font_sha256_cache.get(normalized)
        if cached:
            return cached
        digest = hashlib.sha256()
        try:
            with open(normalized, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise FontManagerError(
                f"registered target font hash unavailable: {normalized}"
            ) from exc
        value = digest.hexdigest()
        self._font_sha256_cache[normalized] = value
        return value

    def required_role_inventory(self) -> list[FontRoleStatus]:
        statuses: list[FontRoleStatus] = []
        for role_id, filename, native_face_id, substitute_face_id in REQUIRED_FONT_ROLES:
            native = self._faces.get(native_face_id)
            substitute = self._faces.get(substitute_face_id) if substitute_face_id else None
            selected = native or substitute
            statuses.append(
                FontRoleStatus(
                    role_id=role_id,
                    preferred_filename=filename,
                    native_face_id=native_face_id,
                    selected_face_id=selected.face_id if selected else "",
                    native_asset_available=native is not None,
                    substitute_face_id=substitute.face_id if native is None and substitute is not None else "",
                    substitution_reason=(
                        "native_role_asset_missing_explicit_substitute"
                        if native is None and substitute is not None
                        else ""
                    ),
                )
            )
        return statuses

    def cache_stats(self) -> dict[str, dict[str, int]]:
        return {
            "hits": dict(self._cache_hits),
            "misses": dict(self._cache_misses),
        }

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "font_manager_version": FONT_MANAGER_VERSION,
            "base_dir": self.base_dir or "",
            "font_pack_dir": model_resolution.noto_cjk_sc_font_dir(self.base_dir),
            "has_font_pack": self.has_font_pack,
            "faces": [face.to_audit_dict() for face in self.available_faces()],
            "required_role_inventory": [item.to_audit_dict() for item in self.required_role_inventory()],
            "cache_stats": self.cache_stats(),
        }

    def _register_noto_cjk_sc_core(self) -> None:
        font_dir = model_resolution.noto_cjk_sc_font_dir(self.base_dir)
        candidates = [
            (
                "noto_sans_cjk_sc_regular",
                "Noto Sans CJK SC",
                "dialogue",
                "regular",
                "NotoSansCJKsc-Regular.otf",
                False,
                False,
                10,
            ),
            (
                "noto_sans_cjk_sc_medium",
                "Noto Sans CJK SC",
                "medium",
                "medium",
                "NotoSansCJKsc-Medium.otf",
                False,
                False,
                15,
            ),
            (
                "noto_sans_cjk_sc_bold",
                "Noto Sans CJK SC",
                "bold",
                "bold",
                "NotoSansCJKsc-Bold.otf",
                False,
                False,
                20,
            ),
            (
                "noto_sans_cjk_sc_black",
                "Noto Sans CJK SC",
                "heavy",
                "black",
                "NotoSansCJKsc-Black.otf",
                False,
                False,
                30,
            ),
            (
                "noto_serif_cjk_sc_regular",
                "Noto Serif CJK SC",
                "serif",
                "regular",
                "NotoSerifCJKsc-Regular.otf",
                True,
                False,
                40,
            ),
            (
                "noto_serif_cjk_sc_semibold",
                "Noto Serif CJK SC",
                "serif_semibold",
                "semibold",
                "NotoSerifCJKsc-SemiBold.otf",
                True,
                False,
                45,
            ),
            (
                "noto_serif_cjk_sc_bold",
                "Noto Serif CJK SC",
                "serif_bold",
                "bold",
                "NotoSerifCJKsc-Bold.otf",
                True,
                False,
                50,
            ),
            (
                "noto_sans_mono_cjk_sc_regular",
                "Noto Sans Mono CJK SC",
                "monospace",
                "regular",
                "NotoSansMonoCJKsc-Regular.otf",
                False,
                True,
                60,
            ),
        ]
        for face_id, family, style_class, weight, filename, serif, monospace, priority in candidates:
            path = os.path.join(font_dir, filename)
            if not os.path.isfile(path):
                continue
            self._faces[face_id] = FontFace(
                face_id=face_id,
                family=family,
                style_class=style_class,
                weight=weight,
                path=path,
                serif=serif,
                monospace=monospace,
                priority=priority,
            )

    def _register_noto_latin_core(self) -> None:
        path = model_resolution.resolve_noto_latin_variable_font_file(
            self.base_dir
        )
        if not path:
            return
        candidates = (
            (
                "noto_sans_latin_condensed_regular",
                "dialogue",
                "regular",
                400.0,
                70,
            ),
            (
                "noto_sans_latin_condensed_medium",
                "medium",
                "medium",
                500.0,
                75,
            ),
            (
                "noto_sans_latin_condensed_bold",
                "bold",
                "bold",
                700.0,
                80,
            ),
            (
                "noto_sans_latin_condensed_black",
                "heavy",
                "black",
                900.0,
                85,
            ),
        )
        for face_id, style_class, weight, weight_axis, priority in candidates:
            self._faces[face_id] = FontFace(
                face_id=face_id,
                family="Noto Sans",
                style_class=style_class,
                weight=weight,
                path=path,
                source="noto_latin_variable_core",
                serif=False,
                monospace=False,
                priority=priority,
                variations=(("wdth", 75.0), ("wght", weight_axis)),
            )

    def _register_windows_fallbacks(self) -> None:
        if os.name != "nt":
            return
        windows_dir = os.environ.get("WINDIR") or r"C:\Windows"
        font_dir = os.path.join(windows_dir, "Fonts")
        candidates = [
            ("windows_arial_regular", "Arial", "script_fallback", "regular", "arial.ttf", False, False, 200),
            ("windows_nirmala_ui_regular", "Nirmala UI", "script_fallback", "regular", "Nirmala.ttc", False, False, 210),
            ("windows_segoe_ui_symbol", "Segoe UI Symbol", "symbol_fallback", "regular", "seguisym.ttf", False, False, 220),
            ("windows_segoe_ui_emoji", "Segoe UI Emoji", "emoji_fallback", "regular", "seguiemj.ttf", False, False, 230),
        ]
        for face_id, family, style_class, weight, filename, serif, monospace, priority in candidates:
            path = os.path.join(font_dir, filename)
            if not os.path.isfile(path):
                continue
            self._faces[face_id] = FontFace(
                face_id=face_id,
                family=family,
                style_class=style_class,
                weight=weight,
                path=path,
                source="windows_system_font",
                serif=serif,
                monospace=monospace,
                priority=priority,
            )

    def _fallback_chain(
        self,
        *,
        requested_family: str,
        requested_weight: str,
        style_class: str,
        fallback_chain_key: str,
    ) -> list[FontFace]:
        ids: list[str] = []
        family_key = requested_family.strip().lower()
        requested_basename = os.path.basename(requested_family.strip()).lower()
        style_key = style_class.strip().lower()
        weight_key = requested_weight.strip().lower()
        if requested_basename:
            ids.extend(
                face.face_id
                for face in self.available_faces()
                if os.path.basename(face.path).lower() == requested_basename
            )
        if "mono" in family_key or "mono" in style_key:
            ids.extend(["noto_sans_mono_cjk_sc_regular", "noto_sans_cjk_sc_regular"])
        elif "serif" in family_key or "serif" in style_key or style_key == "narration":
            if weight_key in {"bold", "black", "heavy"}:
                ids.extend(["noto_serif_cjk_sc_bold", "noto_serif_cjk_sc_semibold", "noto_serif_cjk_sc_regular"])
            elif weight_key == "semibold":
                ids.extend(["noto_serif_cjk_sc_semibold", "noto_serif_cjk_sc_bold", "noto_serif_cjk_sc_regular"])
            else:
                ids.extend(["noto_serif_cjk_sc_regular", "noto_serif_cjk_sc_semibold", "noto_serif_cjk_sc_bold"])
        elif weight_key in {"black", "heavy"} or style_key in {"heavy", "impact"}:
            ids.extend(["noto_sans_cjk_sc_black", "noto_sans_cjk_sc_bold", "noto_sans_cjk_sc_medium"])
        elif weight_key == "bold" or style_key in {"bold", "emphasis"}:
            ids.extend(["noto_sans_cjk_sc_bold", "noto_sans_cjk_sc_black", "noto_sans_cjk_sc_medium"])
        elif weight_key == "semibold":
            ids.extend(["noto_sans_cjk_sc_bold", "noto_sans_cjk_sc_medium", "noto_sans_cjk_sc_regular"])
        elif weight_key == "medium":
            ids.extend(["noto_sans_cjk_sc_medium", "noto_sans_cjk_sc_regular", "noto_sans_cjk_sc_bold"])
        else:
            ids.extend(["noto_sans_cjk_sc_regular", "noto_sans_cjk_sc_medium", "noto_sans_cjk_sc_bold", "noto_sans_cjk_sc_black"])
        if fallback_chain_key == "serif-first":
            ids = ["noto_serif_cjk_sc_regular", "noto_serif_cjk_sc_semibold", "noto_serif_cjk_sc_bold", *ids]
        ids.extend([
            "noto_sans_cjk_sc_regular",
            "noto_sans_cjk_sc_medium",
            "noto_sans_cjk_sc_bold",
            "noto_sans_cjk_sc_black",
            "noto_serif_cjk_sc_regular",
            "noto_serif_cjk_sc_semibold",
            "noto_serif_cjk_sc_bold",
            "noto_sans_mono_cjk_sc_regular",
            "windows_arial_regular",
            "windows_nirmala_ui_regular",
            "windows_segoe_ui_symbol",
            "windows_segoe_ui_emoji",
        ])
        chain: list[FontFace] = []
        seen: set[str] = set()
        for face_id in ids:
            if face_id in seen:
                continue
            seen.add(face_id)
            face = self._faces.get(face_id)
            if face:
                chain.append(face)
        return chain

    def _font_cmap(self, path: str) -> set[int]:
        if path in self._cmap_cache:
            self._cache_hits["cmap"] += 1
            return self._cmap_cache[path]
        self._cache_misses["cmap"] += 1
        cmap: set[int] = set()
        if TTFont is None:
            self._cmap_cache[path] = cmap
            return cmap
        font = None
        try:
            font = TTFont(path, fontNumber=0, lazy=True)
            cmap_table = font.get("cmap")
            if cmap_table is not None:
                for table in cmap_table.tables:
                    cmap.update(int(codepoint) for codepoint in table.cmap.keys())
        finally:
            if font is not None:
                font.close()
        self._cmap_cache[path] = cmap
        return cmap


def _unique_chars(text: str) -> list[str]:
    chars: list[str] = []
    seen: set[str] = set()
    for char in str(text or ""):
        if char in seen:
            continue
        seen.add(char)
        chars.append(char)
    return chars


@lru_cache(maxsize=512)
def _canonical_font_path(path: str) -> str:
    return os.path.normcase(
        os.path.realpath(os.path.abspath(str(path or "")))
    )


def _normalize_optical_writing_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode.startswith("vert"):
        return "vertical"
    if mode.startswith("horiz"):
        return "horizontal"
    raise FontManagerError(f"unsupported optical-profile writing mode: {value!r}")


def _usable_optical_cjk_graphemes(
    manager: FontManager,
    face: FontFace,
    text: str,
) -> tuple[str, ...]:
    candidates: set[str] = set()
    for cluster in strict_grapheme_clusters(str(text or "")):
        if not cluster or not any(_is_cjk_codepoint(char) for char in cluster):
            continue
        visible_categories = {
            unicodedata.category(char)
            for char in cluster
            if unicodedata.category(char) not in {"Cf", "Mn", "Me"}
        }
        if not visible_categories or all(category.startswith("P") for category in visible_categories):
            continue
        if manager.coverage_for_text(face, cluster).supports_text:
            candidates.add(cluster)
    return tuple(sorted(candidates))


def _fixed_optical_probe_glyph_sets(
    manager: FontManager,
    face: FontFace,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    probes: list[tuple[str, tuple[str, ...]]] = []
    for probe_id, text in TARGET_OPTICAL_PROFILE_PROBE_BANK:
        glyph_set = _usable_optical_cjk_graphemes(
            manager,
            face,
            text,
        )
        if not glyph_set:
            raise FontManagerError(
                "registered target face lacks fixed optical probe support: "
                f"{probe_id}"
            )
        probes.append((probe_id, glyph_set))
    return tuple(probes)


def _fixed_latin_optical_probe_glyph_sets(
    manager: FontManager,
    face: FontFace,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    probes: list[tuple[str, tuple[str, ...]]] = []
    for probe_id, text in LATIN_TARGET_OPTICAL_PROFILE_PROBE_BANK:
        glyph_set = tuple(
            cluster
            for cluster in strict_grapheme_clusters(text)
            if cluster
            and manager.coverage_for_text(face, cluster).supports_text
        )
        if not glyph_set:
            raise FontManagerError(
                "registered target face lacks fixed Latin optical probe "
                f"support: {probe_id}"
            )
        probes.append((probe_id, glyph_set))
    return tuple(probes)


def _is_cjk_codepoint(char: str) -> bool:
    codepoint = ord(char)
    return any(
        start <= codepoint <= end
        for start, end in (
            (0x2E80, 0x2FFF),
            (0x3040, 0x30FF),
            (0x3100, 0x312F),
            (0x3130, 0x318F),
            (0x31A0, 0x31BF),
            (0x31F0, 0x31FF),
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xA960, 0xA97F),
            (0xAC00, 0xD7AF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2FA1F),
        )
    )


def _optical_glyph_set_sha256(glyph_set: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for glyph in glyph_set:
        encoded = str(glyph).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _measure_optical_glyph(
    font: Any,
    glyph: str,
    *,
    writing_mode: str,
    pad_boundary: bool = False,
) -> dict[str, float]:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - required local dependencies
        raise FontManagerError("target optical metrics require numpy and OpenCV") from exc

    # The fixed probe bank measures one upright CJK/kana glyph at a time.  Its
    # frozen v2 policy has identical ink/advance metrics in horizontal and
    # vertical modes; writing mode remains bound by the profile cache/id.
    # Passing ``direction="ttb"`` asks Pillow to shape text through libraqm,
    # which is unnecessary for these single-glyph measurements and is not
    # available in the standard macOS wheel.
    direction = None
    try:
        mask = font.getmask(glyph, mode="L", direction=direction)
        width, height = mask.size
        raster = np.asarray(mask, dtype=np.uint8).reshape((height, width))
        advance = float(font.getlength(glyph, direction=direction))
    except Exception as exc:
        raise FontManagerError(f"target optical glyph raster failed: {glyph!r}") from exc
    binary = raster >= 128
    ys, xs = np.nonzero(binary)
    if xs.size <= 0 or ys.size <= 0:
        raise FontManagerError(f"target optical glyph has no visible ink: {glyph!r}")
    ink = np.ascontiguousarray(
        binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1],
        dtype=np.uint8,
    )
    visible_ink_height_px = float(ink.shape[0])
    distance_input = (
        np.pad(ink, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        if pad_boundary
        else ink
    )
    distance = cv2.distanceTransform(distance_input, cv2.DIST_L2, 5)
    positive = distance[distance > 0.0]
    if positive.size <= 0:
        raise FontManagerError(f"target optical glyph has no measurable stem: {glyph!r}")
    stem_width_px = float(np.percentile(positive * 2.0, 75))
    return {
        "visible_ink_height_px": visible_ink_height_px,
        "advance_px": advance,
        "stem_width_px": stem_width_px,
        "ink_coverage_ratio": float(ink.mean()),
    }


def _first_covering_face(
    manager: FontManager,
    chain: Sequence[FontFace],
    text: str,
) -> tuple[int, FontFace, GlyphCoverage] | None:
    for index, face in enumerate(chain):
        coverage = manager.coverage_for_text(face, text)
        if coverage.supports_text:
            return index, face, coverage
    return None


def _preferred_sequence_chain(chain: Sequence[FontFace], text: str) -> list[FontFace]:
    faces = list(chain)
    if not _requires_emoji_sequence_face(text):
        return faces
    return sorted(
        faces,
        key=lambda face: (
            0 if face.style_class == "emoji_fallback" else 1,
            faces.index(face),
        ),
    )


def _requires_emoji_sequence_face(text: str) -> bool:
    value = str(text or "")
    if not value:
        return False
    return any(
        is_emoji_grapheme_cluster(cluster)
        for cluster in strict_grapheme_clusters(value)
    )


def _font_run_must_remain_atomic(
    *,
    script: str,
    direction: str,
    role: str,
    writing_mode: str,
) -> bool:
    script_key = str(script or "").strip()
    direction_key = str(direction or "").strip().lower()
    role_key = str(role or "").strip().lower()
    mode_key = str(writing_mode or "").strip().lower()
    complex_scripts = {
        "Arab",
        "Hebr",
        "Deva",
        "Beng",
        "Guru",
        "Gujr",
        "Orya",
        "Taml",
        "Telu",
        "Knda",
        "Mlym",
        "Sinh",
        "Thai",
    }
    semantic_atomic_roles = {
        "complex_script",
        "symbol",
        "ellipsis_sequence",
        "dash_sequence",
        "wave_sequence",
        "punctuation_sequence",
    }
    if script_key in complex_scripts or direction_key == "rtl":
        return True
    if role_key in semantic_atomic_roles:
        return True
    return mode_key == "vertical" and role_key in {"latin_word", "numeric_token"}


def _cluster_codepoint_offsets(clusters: Sequence[str]) -> list[int]:
    offsets = [0]
    for cluster in clusters:
        offsets.append(offsets[-1] + len(cluster))
    return offsets


def _unresolved_font_span(
    *,
    logical_run_id: str,
    span_id: str,
    text: str,
    grapheme_start: int,
    grapheme_end: int,
    codepoint_start: int,
    codepoint_end: int,
    missing_clusters: Sequence[str],
    issue: str,
    primary_face_id: str = "",
) -> FontSpanResolution:
    missing_chars = [
        char
        for char in _unique_chars(text)
        if not _ignore_coverage_char(char)
    ]
    coverage = GlyphCoverage(
        face_id="",
        font_path="",
        text=str(text or ""),
        missing_chars=missing_chars,
        ignored_chars=[
            char for char in _unique_chars(text) if _ignore_coverage_char(char)
        ],
    )
    return FontSpanResolution(
        logical_run_id=logical_run_id,
        span_id=span_id,
        text=str(text or ""),
        source_grapheme_start=int(grapheme_start),
        source_grapheme_end=int(grapheme_end),
        source_codepoint_start=int(codepoint_start),
        source_codepoint_end=int(codepoint_end),
        selected_face=None,
        coverage=coverage,
        fallback_used=False,
        fallback_index=-1,
        selection_reason=str(issue or "unresolved_font_span"),
        missing_clusters=list(missing_clusters),
        issues=[str(issue or "unresolved_font_span")],
        primary_face_id=str(primary_face_id or ""),
        face_authority="unresolved_missing_glyph_coverage",
    )


def _ignore_coverage_char(char: str) -> bool:
    from app.render.typesetting_text import source_char_requires_visible_glyph

    return not source_char_requires_visible_glyph(char)


def _target_physical_font_role(
    logical_role_id: str,
    *,
    target_script: str,
    writing_mode: str,
) -> str:
    role = str(logical_role_id or "").strip()
    if (
        str(target_script or "").strip() == "Latn"
        and str(writing_mode or "").strip().lower() == "horizontal"
    ):
        return LATIN_TARGET_ROLE_MAP.get(role, role)
    return role


def _apply_pillow_font_variations(
    font: Any,
    variations: Sequence[tuple[str, float]],
) -> None:
    requested = {str(tag): float(value) for tag, value in variations}
    if not requested:
        return
    axes = list(font.get_variation_axes())
    coordinates: list[float] = []
    for axis in axes:
        name = axis.get("name", b"")
        if isinstance(name, bytes):
            name = name.decode("ascii", errors="ignore")
        normalized = str(name or "").strip().lower()
        tag = "wght" if normalized == "weight" else "wdth" if normalized == "width" else ""
        value = requested.get(tag, float(axis.get("default") or 0.0))
        minimum = float(axis.get("minimum") or value)
        maximum = float(axis.get("maximum") or value)
        coordinates.append(max(minimum, min(maximum, float(value))))
    font.set_variation_by_axes(coordinates)


def default_font_manager(*, base_dir: str | None = None) -> FontManager:
    return FontManager(base_dir=base_dir)
