# -*- coding: utf-8 -*-
"""HarfBuzz shaping boundary for the Stage 4 typesetting engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.render.typesetting_text import (
    is_default_ignorable_codepoint,
    source_text_requires_visible_glyph,
)

try:
    import uharfbuzz as hb
except Exception:  # pragma: no cover - tested through runtime behavior
    hb = None

try:
    from fontTools.ttLib import TTFont
except Exception:  # pragma: no cover - optional dependency already present
    TTFont = None


COMPLEX_SCRIPTS = {"Arab", "Hebr", "Deva", "Beng", "Guru", "Gujr", "Orya", "Taml", "Telu", "Knda", "Mlym", "Sinh", "Thai"}


@dataclass(frozen=True)
class ShapedGlyph:
    glyph_id: int
    glyph_name: str
    cluster: int
    text: str
    x_advance: float
    y_advance: float
    x_offset: float
    y_offset: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "glyph_id": int(self.glyph_id),
            "glyph_name": self.glyph_name,
            "cluster": int(self.cluster),
            "text": self.text,
            "x_advance": float(self.x_advance),
            "y_advance": float(self.y_advance),
            "x_offset": float(self.x_offset),
            "y_offset": float(self.y_offset),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ShapedRun:
    text: str
    normalized_text: str
    font_face_id: str
    font_path: str
    font_size: int
    direction: str
    script: str
    language: str
    features: dict[str, bool]
    glyphs: list[ShapedGlyph]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "font_face_id": self.font_face_id,
            "font_path": self.font_path,
            "font_size": int(self.font_size),
            "direction": self.direction,
            "script": self.script,
            "language": self.language,
            "features": dict(self.features),
            "glyphs": [glyph.to_audit_dict() for glyph in self.glyphs],
            "metadata": dict(self.metadata),
        }


class HarfBuzzShaper:
    def __init__(self, font_manager) -> None:
        self.font_manager = font_manager
        self._font_data_cache: dict[str, bytes] = {}
        self._face_cache: dict[str, Any] = {}
        self._ttfont_cache: dict[str, Any] = {}

    def shape_text(
        self,
        text: str,
        *,
        face,
        font_size: int,
        writing_mode: str,
        language: str = "",
        script: str = "",
        direction: str = "",
        features: dict[str, bool] | None = None,
    ) -> ShapedRun:
        if hb is None:
            raise RuntimeError("harfbuzz_unavailable")
        if face is None:
            raise RuntimeError("font_face_unavailable")
        text_value = str(text or "")
        size = max(1, int(font_size))
        mode = str(writing_mode or "").lower()
        resolved_direction = _normalize_direction(direction, mode)
        resolved_script = str(script or "").strip() or ""
        resolved_language = str(language or "").strip()
        resolved_features = dict(features or {})
        resolved_features.setdefault("kern", True)
        if resolved_direction == "ttb":
            resolved_features.setdefault("vert", True)
            resolved_features.setdefault("vrt2", True)
        else:
            resolved_features.setdefault("liga", True)

        hb_face = self._hb_face(face.path)
        hb_font = hb.Font(hb_face)
        hb_font.scale = (size * 64, size * 64)
        buffer = hb.Buffer()
        buffer.add_str(text_value)
        remove_default_ignorables = getattr(
            getattr(hb, "BufferFlags", None),
            "REMOVE_DEFAULT_IGNORABLES",
            None,
        )
        if remove_default_ignorables is not None:
            buffer.flags |= remove_default_ignorables
        if resolved_direction:
            buffer.direction = resolved_direction
        if resolved_script:
            buffer.script = resolved_script
        if resolved_language:
            buffer.language = resolved_language
        buffer.guess_segment_properties()
        if resolved_direction:
            buffer.direction = resolved_direction
        if resolved_script:
            buffer.script = resolved_script
        if resolved_language:
            buffer.language = resolved_language
        hb.shape(hb_font, buffer, resolved_features)

        infos = list(buffer.glyph_infos)
        positions = list(buffer.glyph_positions)
        glyphs: list[ShapedGlyph] = []
        removed_nonvisual_notdef: list[dict[str, Any]] = []
        cluster_values = [int(info.cluster) for info in infos]
        for index, (info, pos) in enumerate(zip(infos, positions)):
            cluster = int(info.cluster)
            glyph_id = int(info.codepoint)
            cluster_text = _cluster_text(text_value, cluster, cluster_values)
            if glyph_id == 0 and not source_text_requires_visible_glyph(cluster_text):
                removed_nonvisual_notdef.append(
                    {
                        "glyph_id": glyph_id,
                        "cluster": cluster,
                        "text": cluster_text,
                        "reason": "unicode_default_ignorable_no_visible_ink",
                    }
                )
                continue
            glyphs.append(
                ShapedGlyph(
                    glyph_id=glyph_id,
                    glyph_name=self._glyph_name(face.path, glyph_id),
                    cluster=cluster,
                    text=cluster_text,
                    x_advance=float(pos.x_advance) / 64.0,
                    y_advance=float(pos.y_advance) / 64.0,
                    x_offset=float(pos.x_offset) / 64.0,
                    y_offset=float(pos.y_offset) / 64.0,
                    metadata={"glyph_index": index},
                )
            )

        final_direction = str(buffer.direction or resolved_direction or "").lower()
        final_script = str(buffer.script or resolved_script or "")
        final_language = str(buffer.language or resolved_language or "")
        default_ignorable_codepoints = [
            f"U+{ord(char):04X}"
            for char in text_value
            if is_default_ignorable_codepoint(char)
        ]
        default_ignorable_metadata = (
            {
                "default_ignorable_policy": "unicode_property_no_visible_ink",
                "default_ignorable_codepoints": default_ignorable_codepoints,
                "removed_nonvisual_notdef_count": len(removed_nonvisual_notdef),
                "removed_nonvisual_notdef_glyphs": removed_nonvisual_notdef,
            }
            if default_ignorable_codepoints or removed_nonvisual_notdef
            else {}
        )
        return ShapedRun(
            text=text_value,
            normalized_text=text_value,
            font_face_id=str(getattr(face, "face_id", "")),
            font_path=str(getattr(face, "path", "")),
            font_size=size,
            direction=final_direction,
            script=final_script,
            language=final_language,
            features={str(key): bool(value) for key, value in resolved_features.items()},
            glyphs=glyphs,
            metadata={
                "shaping_engine": "harfbuzz",
                "position_scale": "26.6_to_px",
                "complex_script": final_script in COMPLEX_SCRIPTS or final_direction == "rtl",
                "writing_mode": writing_mode,
                **default_ignorable_metadata,
            },
        )

    def _hb_face(self, path: str):
        if path in self._face_cache:
            return self._face_cache[path]
        data = self._font_data_cache.get(path)
        if data is None:
            with open(path, "rb") as handle:
                data = handle.read()
            self._font_data_cache[path] = data
        face = hb.Face(data)
        self._face_cache[path] = face
        return face

    def _glyph_name(self, path: str, glyph_id: int) -> str:
        if TTFont is None:
            return f"gid{glyph_id}"
        font = self._ttfont_cache.get(path)
        if font is None:
            font = TTFont(path, fontNumber=0, lazy=True)
            self._ttfont_cache[path] = font
        try:
            return str(font.getGlyphName(int(glyph_id)))
        except Exception:
            return f"gid{glyph_id}"


def _normalize_direction(direction: str, writing_mode: str) -> str:
    value = str(direction or "").strip().lower()
    if value in {"ltr", "rtl", "ttb", "btt"}:
        return value
    mode = str(writing_mode or "").strip().lower()
    if mode.startswith("vert"):
        return "ttb"
    return "ltr"


def _cluster_text(text: str, cluster: int, clusters: list[int] | None = None) -> str:
    if not text:
        return ""
    if cluster < 0 or cluster >= len(text):
        return ""
    bounds = sorted({int(value) for value in (clusters or []) if 0 <= int(value) <= len(text)})
    bounds.append(len(text))
    end = len(text)
    for value in bounds:
        if value > cluster:
            end = value
            break
    return text[cluster:end] or text[cluster]
