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
    "...": ("︙",),
    "…": ("︙",),
    "……": ("︙︙",),
    "--": ("︱︱",),
    "——": ("︱︱",),
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
        self._cache_hits = {"font": 0, "glyph_metrics": 0, "text_metrics": 0, "cmap": 0}
        self._cache_misses = {"font": 0, "glyph_metrics": 0, "text_metrics": 0, "cmap": 0}
        self._register_noto_cjk_sc_core()

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
        if not chain:
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
        selected = chain[0]
        missing_glyphs: list[str] = []
        if text:
            for face in chain:
                coverage = self.coverage_for_text(face, text)
                if coverage.supports_text:
                    selected = face
                    missing_glyphs = []
                    break
                if face is chain[0]:
                    missing_glyphs = list(coverage.missing_chars)
            else:
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
                "noto_serif_cjk_sc_bold",
                "Noto Serif CJK SC",
                "serif_bold",
                "bold",
                "NotoSerifCJKsc-Bold.otf",
                True,
                False,
                50,
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
        style_key = style_class.strip().lower()
        weight_key = requested_weight.strip().lower()
        if "serif" in family_key or style_key in {"serif", "narration", "caption_serif"}:
            ids.extend(["noto_serif_cjk_sc_regular", "noto_serif_cjk_sc_bold"])
        elif "mono" in family_key or style_key in {"mono", "monospace"}:
            ids.extend(["noto_sans_cjk_sc_regular"])
        elif weight_key in {"black", "heavy"} or style_key in {"heavy", "impact"}:
            ids.extend(["noto_sans_cjk_sc_black", "noto_sans_cjk_sc_bold"])
        elif weight_key == "bold" or style_key in {"bold", "emphasis"}:
            ids.extend(["noto_sans_cjk_sc_bold", "noto_sans_cjk_sc_black"])
        else:
            ids.extend(["noto_sans_cjk_sc_regular", "noto_sans_cjk_sc_bold", "noto_sans_cjk_sc_black"])
        if fallback_chain_key == "serif-first":
            ids = ["noto_serif_cjk_sc_regular", "noto_serif_cjk_sc_bold", *ids]
        ids.extend([
            "noto_sans_cjk_sc_regular",
            "noto_sans_cjk_sc_bold",
            "noto_sans_cjk_sc_black",
            "noto_serif_cjk_sc_regular",
            "noto_serif_cjk_sc_bold",
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
    if text in {"black", "heavy"}:
        return "black"
    if text in {"bold", "semibold", "semi-bold", "medium"}:
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
