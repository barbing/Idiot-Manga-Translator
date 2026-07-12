# -*- coding: utf-8 -*-
"""OpenType font registry, coverage, fallback, and metrics services."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.models import resolution as model_resolution

try:
    from PIL import ImageFont
except Exception:  # pragma: no cover - optional runtime dependency
    ImageFont = None

try:
    from fontTools.ttLib import TTFont
except Exception:  # pragma: no cover - optional runtime dependency
    TTFont = None


FONT_MANAGER_VERSION = "font_manager_v1"
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
)


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

    @property
    def usable(self) -> bool:
        return self.primary_face is not None

    def to_audit_dict(self) -> dict[str, Any]:
        return {
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
        }


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

    def __init__(self, *, base_dir: str | None = None) -> None:
        self.base_dir = base_dir
        self._faces: dict[str, FontFace] = {}
        self._cmap_cache: dict[str, set[int]] = {}
        self._font_cache: dict[tuple[str, int], Any] = {}
        self._glyph_metrics_cache: dict[tuple[str, int, str], GlyphMetrics] = {}
        self._text_metrics_cache: dict[tuple[str, int, str, str], TextMetrics] = {}
        self._open_type_metrics_cache: dict[tuple[str, int], OpenTypeMetrics] = {}
        self._cache_hits = {
            "font": 0,
            "glyph_metrics": 0,
            "text_metrics": 0,
            "open_type_metrics": 0,
            "cmap": 0,
        }
        self._cache_misses = {
            "font": 0,
            "glyph_metrics": 0,
            "text_metrics": 0,
            "open_type_metrics": 0,
            "cmap": 0,
        }
        self._register_noto_cjk_sc_core()
        self._register_windows_fallbacks()

    @property
    def has_font_pack(self) -> bool:
        return model_resolution.has_noto_cjk_sc_font_pack(self.base_dir)

    def available_faces(self) -> list[FontFace]:
        return sorted(self._faces.values(), key=lambda face: (face.priority, face.face_id))

    def face(self, face_id: str) -> FontFace | None:
        return self._faces.get(str(face_id or ""))

    def resolve_font(
        self,
        resolved_style: Mapping[str, Any] | None = None,
        *,
        fallback_chain_key: str = DEFAULT_FALLBACK_CHAIN,
        writing_mode: str = "vertical",
        text: str = "",
    ) -> FontResolution:
        style = resolved_style if isinstance(resolved_style, Mapping) else {}
        requested_family = str(
            style.get("font_family")
            or style.get("font")
            or style.get("family")
            or ""
        )
        requested_weight = _normalize_weight(style.get("font_weight") or style.get("weight") or "")
        style_class = str(style.get("style_class") or style.get("font_style") or "").strip().lower()
        chain = self._fallback_chain(
            requested_family=requested_family,
            requested_weight=requested_weight,
            style_class=style_class,
            fallback_chain_key=fallback_chain_key,
        )
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
            )
        selected = style_faces[0]
        missing_glyphs: list[str] = []
        if text:
            for char in _unique_chars(text):
                if _ignore_coverage_char(char):
                    continue
                if not any(self.coverage_for_text(face, char).supports_text for face in chain):
                    missing_glyphs.append(char)
            if missing_glyphs:
                issues.append("missing_glyphs")
        return FontResolution(
            requested_family=requested_family,
            requested_weight=requested_weight,
            style_class=style_class,
            fallback_chain_key=fallback_chain_key,
            writing_mode=writing_mode,
            primary_face=selected,
            fallback_faces=[face for face in chain if face.face_id != selected.face_id],
            missing_glyphs=missing_glyphs,
            issues=issues,
        )

    def resolve_run_font(
        self,
        resolution: FontResolution,
        text: str,
        *,
        run_id: str = "",
    ) -> RunFontResolution:
        value = str(text or "")
        chain: list[FontFace] = []
        if resolution.primary_face is not None:
            chain.append(resolution.primary_face)
        chain.extend(
            face
            for face in resolution.fallback_faces
            if face is not None and all(face.face_id != item.face_id for item in chain)
        )
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
                selection_reason="missing_font_pack",
                missing_glyphs=list(coverage.missing_chars),
                issues=["missing_font_pack"],
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
                    selection_reason="primary_face_full_run_coverage" if index == 0 else "fallback_face_full_run_coverage",
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
        key = (face.path, font_size)
        if key in self._font_cache:
            self._cache_hits["font"] += 1
            return self._font_cache[key]
        self._cache_misses["font"] += 1
        font = ImageFont.truetype(face.path, font_size)
        self._font_cache[key] = font
        return font

    def glyph_metrics(self, face: FontFace, glyph: str, size: int) -> GlyphMetrics:
        if face is None:
            raise FontManagerError("font face is unavailable")
        font_size = max(1, int(size))
        key = (face.path, font_size, str(glyph or ""))
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
        key = (face.path, font_size, mode, str(text or ""))
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
        key = (face.path, font_size)
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


def _normalize_weight(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"700", "800", "900"}:
        return "bold" if text == "700" else "black"
    if text in {"600", "semibold", "semi-bold"}:
        return "semibold"
    if text in {"500", "medium"}:
        return "medium"
    if text in {"black", "heavy"}:
        return "black"
    if text == "bold":
        return "bold"
    return text or "regular"


def _unique_chars(text: str) -> list[str]:
    chars: list[str] = []
    seen: set[str] = set()
    for char in str(text or ""):
        if char in seen:
            continue
        seen.add(char)
        chars.append(char)
    return chars


def _ignore_coverage_char(char: str) -> bool:
    if not char:
        return True
    if char.isspace():
        return True
    codepoint = ord(char)
    return 0xFE00 <= codepoint <= 0xFE0F


def default_font_manager(*, base_dir: str | None = None) -> FontManager:
    return FontManager(base_dir=base_dir)
