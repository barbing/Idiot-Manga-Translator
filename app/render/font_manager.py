# -*- coding: utf-8 -*-
"""OpenType font registry, coverage, fallback, and metrics services."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
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
DEFAULT_FALLBACK_CHAIN = "cjk-sc"
TARGET_FONT_REQUEST_VERSION = "target_font_request_v1"
TARGET_FONT_REQUEST_PROVENANCE = "parent_style_arbitrator_source_label_taxonomy_v1"
OPTIONAL_TARGET_FONT_STYLE_CLASSES = {
    "calligraphic",
    "handwritten",
    "display",
    "rounded",
}

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
    optional_target_face_resolution: dict[str, Any] = field(default_factory=dict)

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
        }
        if self.optional_target_face_resolution:
            audit["optional_target_face_resolution"] = dict(
                self.optional_target_face_resolution
            )
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
        optional_font_catalog: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.base_dir = base_dir
        self._faces: dict[str, FontFace] = {}
        self._cmap_cache: dict[str, set[int]] = {}
        self._font_cache: dict[tuple[str, int], Any] = {}
        self._glyph_metrics_cache: dict[tuple[str, int, str], GlyphMetrics] = {}
        self._text_metrics_cache: dict[tuple[str, int, str, str], TextMetrics] = {}
        self._open_type_metrics_cache: dict[tuple[str, int], OpenTypeMetrics] = {}
        catalog = (
            list(optional_font_catalog)
            if optional_font_catalog is not None
            else _default_optional_target_font_catalog()
        )
        self._optional_font_catalog: dict[str, dict[str, Any]] = {}
        self._duplicate_optional_font_face_ids: set[str] = set()
        for raw_record in catalog:
            if not isinstance(raw_record, Mapping):
                continue
            record = dict(raw_record)
            face_id = str(record.get("catalog_face_id") or "").strip()
            if not face_id:
                continue
            if face_id in self._optional_font_catalog:
                self._duplicate_optional_font_face_ids.add(face_id)
                continue
            self._optional_font_catalog[face_id] = record
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
        )
        optional_face, optional_audit = self._resolve_optional_target_face(
            style=style,
            text=str(text or ""),
            requested_weight=requested_weight,
        )
        if optional_face is None:
            core_resolution.optional_target_face_resolution = optional_audit
            return core_resolution
        return FontResolution(
            requested_family=requested_family,
            requested_weight=requested_weight,
            style_class=style_class,
            fallback_chain_key=fallback_chain_key,
            writing_mode=writing_mode,
            primary_face=optional_face,
            fallback_faces=[
                face for face in chain if face.face_id != optional_face.face_id
            ],
            missing_glyphs=[],
            issues=[],
            optional_target_face_resolution=optional_audit,
        )

    def _resolve_optional_target_face(
        self,
        *,
        style: Mapping[str, Any],
        text: str,
        requested_weight: str,
    ) -> tuple[FontFace | None, dict[str, Any]]:
        raw_request = style.get("target_font_request")
        if raw_request in (None, "", {}):
            return None, {}
        audit: dict[str, Any] = {
            "contract_version": TARGET_FONT_REQUEST_VERSION,
            "status": "fallback_core",
            "reason_codes": [],
            "coverage_complete": False,
        }

        def reject(*reasons: str) -> tuple[None, dict[str, Any]]:
            current = list(audit.get("reason_codes") or [])
            for reason in reasons:
                value = str(reason or "").strip()
                if value and value not in current:
                    current.append(value)
            audit["reason_codes"] = current
            return None, audit

        if not isinstance(raw_request, Mapping):
            return reject("optional_font_request_malformed")
        request = dict(raw_request)
        audit["request"] = dict(request)
        required_request_keys = {
            "contract_version",
            "catalog_face_id",
            "style_class",
            "weight",
            "source_label",
            "provenance",
        }
        if set(request) != required_request_keys:
            return reject("optional_font_request_malformed")
        if request.get("contract_version") != TARGET_FONT_REQUEST_VERSION:
            return reject("optional_font_request_contract_invalid")
        if request.get("provenance") != TARGET_FONT_REQUEST_PROVENANCE:
            return reject("optional_font_request_provenance_invalid")

        face_id = str(request.get("catalog_face_id") or "").strip()
        style_class = str(request.get("style_class") or "").strip().lower()
        request_weight = _normalize_weight(request.get("weight"))
        audit.update(
            {
                "catalog_face_id": face_id,
                "requested_style_class": style_class,
                "requested_weight": request_weight,
                "source_label": str(request.get("source_label") or ""),
            }
        )
        if style_class not in OPTIONAL_TARGET_FONT_STYLE_CLASSES:
            return reject("optional_font_style_class_unsupported")
        if face_id in self._duplicate_optional_font_face_ids:
            return reject("optional_font_catalog_face_duplicate")
        raw_catalog = self._optional_font_catalog.get(face_id)
        if not isinstance(raw_catalog, Mapping):
            return reject("optional_font_catalog_face_missing")
        catalog = dict(raw_catalog)
        audit["catalog"] = dict(catalog)
        required_catalog_keys = {
            "catalog_face_id",
            "path",
            "sha256",
            "face_index",
            "family",
            "subfamily",
            "weight",
            "weight_class",
            "style_class",
            "style_class_provenance",
            "source",
        }
        if not required_catalog_keys.issubset(catalog):
            return reject("optional_font_catalog_record_malformed")
        catalog_style = str(catalog.get("style_class") or "").strip().lower()
        catalog_weight = _normalize_weight(catalog.get("weight"))
        if catalog_style != style_class:
            return reject("optional_font_style_class_mismatch")
        if request_weight != catalog_weight or request_weight != requested_weight:
            return reject("optional_font_weight_mismatch")

        path = os.path.abspath(str(catalog.get("path") or ""))
        audit["verified_path"] = path
        if not path or not os.path.isabs(path) or not os.path.isfile(path):
            return reject("optional_font_file_missing")
        try:
            expected_hash = str(catalog.get("sha256") or "").strip().lower()
            with open(path, "rb") as handle:
                verified_hash = hashlib.sha256(handle.read()).hexdigest()
            audit["verified_sha256"] = verified_hash
        except Exception:
            return reject("optional_font_sha256_unreadable")
        if len(expected_hash) != 64 or verified_hash != expected_hash:
            return reject("optional_font_sha256_mismatch")
        if TTFont is None:
            return reject("optional_font_opentype_audit_unavailable")

        font = None
        try:
            face_index = int(catalog.get("face_index") or 0)
            font = TTFont(path, fontNumber=face_index, lazy=True)
            family_names = _font_name_values(font, 1)
            subfamily_names = _font_name_values(font, 2)
            expected_family = str(catalog.get("family") or "").strip()
            expected_subfamily = str(catalog.get("subfamily") or "").strip()
            verified_family = _matching_font_name(expected_family, family_names)
            verified_subfamily = _matching_font_name(
                expected_subfamily, subfamily_names
            )
            os2 = font.get("OS/2")
            verified_weight_class = int(getattr(os2, "usWeightClass", 0) or 0)
            expected_weight_class = int(catalog.get("weight_class") or 0)
            audit.update(
                {
                    "verified_face_index": face_index,
                    "verified_family": verified_family,
                    "verified_subfamily": verified_subfamily,
                    "verified_weight_class": verified_weight_class,
                }
            )
        except Exception:
            return reject("optional_font_opentype_metadata_unreadable")
        finally:
            if font is not None:
                font.close()
        if not verified_family:
            return reject("optional_font_family_mismatch")
        if not verified_subfamily:
            return reject("optional_font_subfamily_mismatch")
        if verified_weight_class != expected_weight_class:
            return reject("optional_font_weight_class_mismatch")
        if not _weight_class_is_exact(catalog_weight, verified_weight_class):
            return reject("optional_font_weight_class_mismatch")

        face = FontFace(
            face_id=face_id,
            family=verified_family,
            style_class=catalog_style,
            weight=catalog_weight,
            path=path,
            source="optional_target_font",
            serif=False,
            monospace=False,
            priority=5,
        )
        coverage = self.coverage_for_text(face, text)
        coverage_complete = bool(coverage.supports_text)
        audit["coverage_complete"] = coverage_complete
        audit["coverage"] = coverage.to_audit_dict()
        rejection_reasons: list[str] = []
        if not coverage_complete:
            rejection_reasons.append("optional_font_incomplete_text_coverage")
        if _contains_complex_script(text):
            rejection_reasons.append("optional_font_complex_script_conflict")
        if rejection_reasons:
            return reject(*rejection_reasons)

        audit["status"] = "resolved"
        audit["reason_codes"] = ["optional_font_exact_audit_resolved"]
        return face, audit

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
        """Resolve a logical run without splitting an extended grapheme.

        The exact existing whole-run resolution is the fast path. Only a run
        that genuinely needs more than one face receives synthetic span IDs.
        Contextual shaping domains and semantic atomic runs fail closed when no
        single face covers them.
        """

        value = str(text or "")
        logical_run_id = str(run_id or "")
        clusters = strict_grapheme_clusters(value)
        codepoint_offsets = _cluster_codepoint_offsets(clusters)
        chain = _font_resolution_chain(resolution)

        whole = self.resolve_run_font(resolution, value, run_id=logical_run_id)
        preferred_chain = _preferred_sequence_chain(chain, value)
        if preferred_chain != chain:
            preferred_whole = _first_covering_face(self, preferred_chain, value)
            if preferred_whole is not None:
                preferred_index, preferred_face, preferred_coverage = preferred_whole
                original_index = next(
                    index for index, face in enumerate(chain) if face.face_id == preferred_face.face_id
                )
                whole = RunFontResolution(
                    run_id=logical_run_id,
                    text=value,
                    selected_face=preferred_face,
                    coverage=preferred_coverage,
                    fallback_used=original_index > 0,
                    fallback_index=original_index,
                    selection_reason=(
                        "emoji_sequence_face_full_run_coverage"
                        if preferred_face.style_class == "emoji_fallback"
                        else "sequence_face_full_run_coverage"
                    ),
                )
        if whole.selected_face is not None and whole.coverage.supports_text:
            return [
                FontSpanResolution(
                    logical_run_id=logical_run_id,
                    span_id=logical_run_id,
                    text=value,
                    source_grapheme_start=0,
                    source_grapheme_end=len(clusters),
                    source_codepoint_start=0,
                    source_codepoint_end=len(value),
                    selected_face=whole.selected_face,
                    coverage=whole.coverage,
                    fallback_used=whole.fallback_used,
                    fallback_index=whole.fallback_index,
                    selection_reason=whole.selection_reason,
                    missing_clusters=[],
                    issues=list(whole.issues),
                )
            ]

        if not chain:
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
                    issue="missing_font_pack",
                )
            ]

        assignments: list[tuple[str, FontFace | None, int]] = []
        for cluster in clusters:
            cluster_chain = _preferred_sequence_chain(chain, cluster)
            selected = _first_covering_face(self, cluster_chain, cluster)
            if selected is None:
                assignments.append((cluster, None, -1))
                continue
            _, face, _ = selected
            original_index = next(
                index for index, item in enumerate(chain) if item.face_id == face.face_id
            )
            assignments.append((cluster, face, original_index))

        must_remain_atomic = _font_run_must_remain_atomic(
            script=script,
            direction=direction,
            role=role,
            writing_mode=writing_mode,
        )
        if must_remain_atomic or len(clusters) <= 1:
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
                )
            ]

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
            span_id = f"{logical_run_id}:fs{span_index:03d}"
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
                        "cluster_safe_fallback_span"
                        if fallback_index > 0
                        else "cluster_safe_primary_span"
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


def _default_optional_target_font_catalog() -> list[dict[str, Any]]:
    windows_dir = os.environ.get("WINDIR") or r"C:\Windows"
    return [
        {
            "catalog_face_id": "stxingkai_regular",
            "path": os.path.abspath(
                os.path.join(windows_dir, "Fonts", "STXINGKA.TTF")
            ),
            "sha256": (
                "7f901dfb0526d542740264fb5ba8dfe48"
                "3293ad060270d273593e2f04a69d080"
            ),
            "face_index": 0,
            "family": "STXingkai",
            "subfamily": "Regular",
            "weight": "regular",
            "weight_class": 400,
            "style_class": "calligraphic",
            "style_class_provenance": "curated_windows_target_face_v1",
            "source": "installed_windows_font_exact_identity",
        }
    ]


def _font_name_values(font: Any, name_id: int) -> list[str]:
    table = font.get("name") if font is not None else None
    values: list[str] = []
    if table is None:
        return values
    for record in table.names:
        if int(getattr(record, "nameID", -1)) != int(name_id):
            continue
        try:
            value = str(record.toUnicode() or "").strip()
        except Exception:
            continue
        if value and value not in values:
            values.append(value)
    return values


def _matching_font_name(expected: str, values: Sequence[str]) -> str:
    expected_key = str(expected or "").strip().casefold()
    for value in values:
        if str(value or "").strip().casefold() == expected_key:
            return str(value)
    return ""


def _weight_class_is_exact(weight: str, weight_class: int) -> bool:
    expected = {
        "regular": 400,
        "medium": 500,
        "semibold": 600,
        "bold": 700,
        "black": 900,
    }
    return int(weight_class) == int(expected.get(str(weight or ""), -1))


def _contains_complex_script(text: str) -> bool:
    ranges = (
        (0x0590, 0x08FF),  # Hebrew, Arabic, Syriac, and related scripts.
        (0x0900, 0x0DFF),  # Indic scripts.
        (0x0E00, 0x0E7F),  # Thai.
        (0x1000, 0x109F),  # Myanmar.
        (0x1780, 0x17FF),  # Khmer.
    )
    return any(
        start <= ord(char) <= end
        for char in str(text or "")
        for start, end in ranges
    )


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


def _font_resolution_chain(resolution: FontResolution) -> list[FontFace]:
    chain: list[FontFace] = []
    if resolution.primary_face is not None:
        chain.append(resolution.primary_face)
    chain.extend(
        face
        for face in resolution.fallback_faces
        if face is not None and all(face.face_id != item.face_id for item in chain)
    )
    return chain


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
    )


def _ignore_coverage_char(char: str) -> bool:
    from app.render.typesetting_text import source_char_requires_visible_glyph

    return not source_char_requires_visible_glyph(char)


def default_font_manager(*, base_dir: str | None = None) -> FontManager:
    return FontManager(base_dir=base_dir)
