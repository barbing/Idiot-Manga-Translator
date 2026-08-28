# -*- coding: utf-8 -*-
"""Pure loaded-project projection for the native GUI shell.

This adapter is the only GUI-5 boundary that reads the persisted project
shape.  It projects immutable base records through ``project_effective_page``
and returns typed, immutable view inputs.  It never mutates the loaded mapping,
invokes a pipeline owner, or substitutes source pixels for cleaned/final
artifacts.

The carriers in this module deliberately do not import ``app.ui.editor`` or
PySide6.  The shell adapts ``CanvasArtifactReferences`` and
``OverlayShapeData`` to its Qt canvas types at the final view boundary.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
from typing import Any

from app.io.project import validate_project_schema_v2
from app.pipeline.hierarchy_revision_contracts import (
    EffectiveParentLineage,
    EffectiveUserRootSnapshot,
    HierarchyRevisionDescriptor,
    ParentIdentityNamespace,
    ParentOrigin,
    ParentStageRequirement,
    RevisionRequiredAction,
    RevisionStage,
    RevisionStageState,
    RootIdentityNamespace,
)
from app.pipeline.ocr_revision_contracts import (
    OcrRevisionError,
    OcrSourceRevisionArtifact,
    OriginalPageAssetBinding,
)
from app.pipeline.translation_revision_contracts import (
    TranslationRevisionArtifact,
)
from app.project_edits.contracts import (
    CANONICAL_WRITING_MODES,
    EditDomain,
    EditTargetKind,
    TargetTextRevisionBaseV1,
    canonical_render_box,
    canonical_render_fill_color,
    canonical_render_outline_color,
    canonical_render_outline_width,
    canonical_render_preferred_size,
    canonical_render_shadow_blur,
    canonical_render_shadow_color,
    canonical_render_shadow_offset,
    canonical_render_font_role,
    canonical_render_font_weight_tier,
    canonical_render_line_height,
    canonical_render_rotation,
    thaw_json,
)
from app.project_edits.commands import (
    ParentGeometryCommandError,
    ParentGeometryCommandErrorCode,
    page_canvas_size_for_project_page,
)
from app.project_edits.fingerprints import canonical_sha256
from app.project_edits.glossary_commands import (
    GlossaryEntryV1,
    project_glossary_snapshot,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.manual_cleanup import (
    UserParentCleanupCoverageTargetV1,
    user_parent_cleanup_coverage_target_from_snapshot,
)
from app.project_edits.projection import (
    EffectivePageSnapshot,
    EffectiveParentSnapshot,
    ProjectionIssueKind,
    TargetFreshness,
    automatic_ordered_parent_ids_for_page,
    automatic_render_box,
    automatic_render_hard_bounds,
    automatic_render_fill_color,
    automatic_render_outline_color,
    automatic_render_outline_width,
    automatic_render_preferred_size,
    automatic_render_shadow_blur,
    automatic_render_shadow_color,
    automatic_render_shadow_offset,
    automatic_render_shadow_enabled,
    automatic_render_font_role,
    automatic_render_font_weight_tier,
    automatic_render_line_height,
    automatic_render_rotation,
    automatic_render_writing_mode,
    cleaned_base_automatic_lineage,
    project_effective_page,
    target_text_revision_base_for_parent,
)
from app.project_edits.ocr_revision_service import (
    resolve_original_page_asset_binding,
)
from app.ui.ui_contract import (
    OVERLAY_IDS,
    ArtifactState,
    Authority,
    CleanupState,
    PageState,
    Presentation,
    resolve_editor_status_presentation,
    resolve_state_presentation,
)
from app.ui.project_hub.new_project_dialog import named_project_display_name
from app.ui.viewmodels.presentation_model import (
    build_page_presentation,
    page_presentation_input_from_effective_snapshot,
)
from app.ui.viewmodels.project_model import PageRow, ParentRow, ProjectRow


_LANGUAGE_LABELS = {
    "ja": "Japanese",
    "ja-jp": "Japanese",
    "zh": "Chinese",
    "zh-cn": "Simplified Chinese",
    "zh-hans": "Simplified Chinese",
    "en": "English",
    "en-us": "English",
}
_CLEANUP_PROVENANCE = "user_manual_cleanup"
_CATALOG_KINDS = {
    "cleaned_page_bases": "Cleaned",
    "rendered_pages": "Final",
    "parent_layers": "Parent layer",
}


def _required_text(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _validated_count(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative")
    return value


@dataclass(frozen=True, slots=True)
class ProjectMetadata:
    project_id: str
    name: str
    project_path: str
    schema_version: str
    source_language: str
    target_language: str
    page_count: int
    completed_count: int
    recoverable: bool

    def __post_init__(self) -> None:
        for field_name in (
            "project_id",
            "name",
            "project_path",
            "schema_version",
            "source_language",
            "target_language",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        page_count = _validated_count(self.page_count, "page_count")
        completed_count = _validated_count(self.completed_count, "completed_count")
        if completed_count > page_count:
            raise ValueError("completed_count cannot exceed page_count")
        if not isinstance(self.recoverable, bool):
            raise TypeError("recoverable must be a boolean")


@dataclass(frozen=True, slots=True)
class ArtifactRevisionReference:
    kind: str
    revision_id: str
    asset_path: str | None
    state: ArtifactState
    provenance: str
    current: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _required_text(self.kind, "kind"))
        object.__setattr__(
            self,
            "revision_id",
            _required_text(self.revision_id, "revision_id"),
        )
        if self.asset_path is not None:
            object.__setattr__(
                self,
                "asset_path",
                _required_text(self.asset_path, "asset_path"),
            )
        object.__setattr__(self, "state", ArtifactState(self.state))
        object.__setattr__(
            self,
            "provenance",
            str(self.provenance or "unknown").strip() or "unknown",
        )
        if not isinstance(self.current, bool):
            raise TypeError("current must be a boolean")


@dataclass(frozen=True, slots=True)
class EditHistoryReference:
    """Immutable user-facing history carrier without exposing edit payloads."""

    record_id: str
    domain: str
    operation: str
    target_kind: str
    target_id: str
    field_name: str
    created_at: str
    active: bool
    effective: bool
    is_control: bool
    issue_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "record_id",
            "domain",
            "operation",
            "target_kind",
            "target_id",
            "created_at",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "field_name", str(self.field_name or "").strip())
        for field_name in ("active", "effective", "is_control"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")
        if self.is_control and (self.active or self.effective):
            raise ValueError("ledger-control history cannot be active or effective")
        if self.effective and not self.active:
            raise ValueError("an effective history edit must be active")
        if not isinstance(self.issue_codes, tuple) or any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in self.issue_codes
        ):
            raise ValueError("issue_codes must contain non-empty strings")
        object.__setattr__(
            self,
            "issue_codes",
            tuple(sorted(set(self.issue_codes))),
        )


@dataclass(frozen=True, slots=True)
class CanvasArtifactReferences:
    page_id: str
    original_path: str | None
    cleaned_path: str | None
    final_path: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "page_id", _required_text(self.page_id, "page_id"))
        for field_name in ("original_path", "cleaned_path", "final_path"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _required_text(value, field_name),
                )


@dataclass(frozen=True, slots=True)
class OverlayShapeData:
    overlay_id: str
    shape_id: str
    kind: str
    points: tuple[float, ...]
    label: str
    parent_id: str = ""
    selected: bool = False

    def __post_init__(self) -> None:
        if self.overlay_id not in OVERLAY_IDS:
            raise ValueError(f"unsupported overlay: {self.overlay_id!r}")
        object.__setattr__(
            self,
            "shape_id",
            _required_text(self.shape_id, "shape_id"),
        )
        if self.kind not in {"rect", "line", "polygon"}:
            raise ValueError(f"unsupported overlay kind: {self.kind!r}")
        if isinstance(self.points, (str, bytes, bytearray)) or not isinstance(
            self.points, Sequence
        ):
            raise TypeError("points must be a numeric sequence")
        points: list[float] = []
        for value in self.points:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("overlay points must be numeric")
            points.append(float(value))
        if self.kind in {"rect", "line"} and len(points) != 4:
            raise ValueError(f"{self.kind} overlays require four values")
        if self.kind == "polygon" and (len(points) < 6 or len(points) % 2):
            raise ValueError("polygon overlays require at least three points")
        if self.kind == "rect" and (points[2] <= 0.0 or points[3] <= 0.0):
            raise ValueError("rect overlay width and height must be positive")
        object.__setattr__(self, "points", tuple(points))
        object.__setattr__(self, "label", str(self.label or ""))
        object.__setattr__(self, "parent_id", str(self.parent_id or "").strip())
        if not isinstance(self.selected, bool):
            raise TypeError("selected must be a boolean")


@dataclass(frozen=True, slots=True)
class RasterOverlayData:
    overlay_id: str
    asset_path: str
    asset_sha256: str
    canvas_size: tuple[int, int]
    label: str

    def __post_init__(self) -> None:
        if self.overlay_id not in {"cleanupMask", "protectedRegions"}:
            raise ValueError(f"unsupported raster overlay: {self.overlay_id!r}")
        object.__setattr__(
            self,
            "asset_path",
            _required_text(self.asset_path, "asset_path"),
        )
        digest = str(self.asset_sha256 or "").strip().lower()
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError("asset_sha256 must be a SHA-256 digest")
        object.__setattr__(self, "asset_sha256", digest)
        if (
            not isinstance(self.canvas_size, tuple)
            or len(self.canvas_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.canvas_size
            )
        ):
            raise ValueError("canvas_size must contain two positive integers")
        object.__setattr__(self, "label", _required_text(self.label, "label"))


@dataclass(frozen=True, slots=True)
class OverlayAvailabilityData:
    overlay_id: str
    available: bool
    tooltip: str

    def __post_init__(self) -> None:
        if self.overlay_id not in OVERLAY_IDS:
            raise ValueError(f"unsupported overlay: {self.overlay_id!r}")
        if not isinstance(self.available, bool):
            raise TypeError("available must be a boolean")
        object.__setattr__(self, "tooltip", _required_text(self.tooltip, "tooltip"))


def _box_contains(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    return bool(
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[0] + inner[2] <= outer[0] + outer[2]
        and inner[1] + inner[3] <= outer[1] + outer[3]
    )


@dataclass(frozen=True, slots=True)
class ProjectedParent:
    parent_row: ParentRow
    effective: EffectiveParentSnapshot
    automatic_source_text: str | None
    selected_model_source_revision: OcrSourceRevisionArtifact | None
    selected_model_translation_revision: TranslationRevisionArtifact | None
    target_text_revision_base: TargetTextRevisionBaseV1 | None
    automatic_target_text: str | None
    mapped_pipeline_source_text: str | None
    mapped_pipeline_target_text: str | None
    user_source_text: str | None
    user_target_text: str | None
    effective_render_style: tuple[tuple[str, Any], ...]
    effective_render_layout: tuple[tuple[str, Any], ...]
    automatic_writing_mode: str | None
    user_writing_mode: str | None
    effective_writing_mode: str | None
    writing_mode_authority: str
    automatic_line_height: float | None
    user_line_height: float | None
    effective_line_height: float | None
    line_height_authority: str
    automatic_rotation: float | None
    user_rotation: float | None
    effective_rotation: float | None
    rotation_authority: str
    automatic_render_box: tuple[int, int, int, int] | None
    automatic_render_hard_bounds: tuple[int, int, int, int] | None
    user_render_box: tuple[int, int, int, int] | None
    effective_render_box: tuple[int, int, int, int] | None
    render_box_authority: str
    automatic_font_role: str | None
    user_font_role: str | None
    effective_font_role: str | None
    font_role_authority: str
    automatic_font_weight_tier: str | None
    user_font_weight_tier: str | None
    effective_font_weight_tier: str | None
    font_weight_tier_authority: str
    automatic_fill_color: str | None
    user_fill_color: str | None
    unresolved_user_fill_color: str | None
    effective_fill_color: str | None
    fill_color_authority: str
    automatic_outline_color: str | None
    user_outline_color: str | None
    unresolved_user_outline_color: str | None
    effective_outline_color: str | None
    outline_color_authority: str
    automatic_outline_width: float | None
    user_outline_width: float | None
    effective_outline_width: float | None
    outline_width_authority: str
    automatic_preferred_size: float | None
    user_preferred_size: float | None
    effective_preferred_size: float | None
    preferred_size_authority: str
    automatic_shadow_blur: float | None
    user_shadow_blur: float | None
    effective_shadow_blur: float | None
    shadow_blur_authority: str
    automatic_shadow_color: str | None
    user_shadow_color: str | None
    effective_shadow_color: str | None
    shadow_color_authority: str
    automatic_shadow_offset: tuple[float, float] | None
    user_shadow_offset: tuple[float, float] | None
    effective_shadow_offset: tuple[float, float] | None
    shadow_offset_authority: str
    automatic_shadow_enabled: bool | None
    user_shadow_enabled: bool | None
    effective_shadow_enabled: bool | None
    shadow_enabled_authority: str
    render_required: bool
    writing_mode_unavailable_reason: str
    line_height_unavailable_reason: str
    rotation_unavailable_reason: str
    render_box_unavailable_reason: str
    font_role_unavailable_reason: str
    font_weight_tier_unavailable_reason: str
    fill_color_unavailable_reason: str
    outline_color_unavailable_reason: str
    outline_width_unavailable_reason: str
    preferred_size_unavailable_reason: str
    shadow_blur_unavailable_reason: str
    shadow_color_unavailable_reason: str
    shadow_offset_unavailable_reason: str
    shadow_visibility_unavailable_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.parent_row, ParentRow):
            raise TypeError("parent_row must be ParentRow")
        if not isinstance(self.effective, EffectiveParentSnapshot):
            raise TypeError("effective must be EffectiveParentSnapshot")
        if self.parent_row.parent_id != self.effective.parent_id:
            raise ValueError("parent row and effective snapshot identities differ")
        for field_name in (
            "automatic_source_text",
            "automatic_target_text",
            "mapped_pipeline_source_text",
            "mapped_pipeline_target_text",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
        if (
            self.selected_model_source_revision is not None
            and not isinstance(
                self.selected_model_source_revision,
                OcrSourceRevisionArtifact,
            )
        ):
            raise TypeError(
                "selected_model_source_revision must be OcrSourceRevisionArtifact or None"
            )
        if (
            self.selected_model_translation_revision is not None
            and not isinstance(
                self.selected_model_translation_revision,
                TranslationRevisionArtifact,
            )
        ):
            raise TypeError(
                "selected_model_translation_revision must be "
                "TranslationRevisionArtifact or None"
            )
        expected_target_base = target_text_revision_base_for_parent(self.effective)
        if self.target_text_revision_base != expected_target_base:
            raise ValueError(
                "target_text_revision_base must match the effective target revision"
            )
        for field_name in ("user_source_text", "user_target_text"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
        for field_name in ("effective_render_style", "effective_render_layout"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple) or any(
                not isinstance(item, tuple) or len(item) != 2 for item in value
            ):
                raise TypeError(f"{field_name} must be a tuple of field pairs")
        for field_name in (
            "automatic_writing_mode",
            "user_writing_mode",
            "effective_writing_mode",
        ):
            value = getattr(self, field_name)
            if value is not None and value not in CANONICAL_WRITING_MODES:
                raise ValueError(f"{field_name} must be canonical or None")
        if self.writing_mode_authority not in {
            "automatic",
            "user",
            "unavailable",
        }:
            raise ValueError(
                "writing_mode_authority must be automatic, user, or unavailable"
            )
        for field_name in (
            "automatic_line_height",
            "user_line_height",
            "effective_line_height",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    canonical_render_line_height(value, field_name=field_name),
                )
        if self.line_height_authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "line_height_authority must be automatic, user, or unavailable"
            )
        if self.line_height_authority == "automatic":
            if self.user_line_height is not None:
                raise ValueError(
                    "automatic line-height authority cannot carry a user value"
                )
            if self.effective_line_height != self.automatic_line_height:
                raise ValueError(
                    "automatic line-height authority must expose the automatic value"
                )
        elif self.line_height_authority == "user":
            if self.user_line_height is None:
                raise ValueError("user line-height authority requires a user value")
            if self.effective_line_height != self.user_line_height:
                raise ValueError(
                    "user line-height authority must expose the user value"
                )
        elif any(
            value is not None
            for value in (
                self.automatic_line_height,
                self.user_line_height,
                self.effective_line_height,
            )
        ):
            raise ValueError("unavailable line-height authority cannot carry values")
        for field_name in (
            "automatic_rotation",
            "user_rotation",
            "effective_rotation",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    canonical_render_rotation(value, field_name=field_name),
                )
        if self.rotation_authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "rotation_authority must be automatic, user, or unavailable"
            )
        if self.rotation_authority == "automatic":
            if self.user_rotation is not None:
                raise ValueError(
                    "automatic rotation authority cannot carry a user value"
                )
            if self.effective_rotation != self.automatic_rotation:
                raise ValueError(
                    "automatic rotation authority must expose the automatic value"
                )
        elif self.rotation_authority == "user":
            if self.user_rotation is None:
                raise ValueError("user rotation authority requires a user value")
            if self.effective_rotation != self.user_rotation:
                raise ValueError(
                    "user rotation authority must expose the user value"
                )
        elif any(
            value is not None
            for value in (
                self.automatic_rotation,
                self.user_rotation,
                self.effective_rotation,
            )
        ):
            raise ValueError("unavailable rotation authority cannot carry values")
        for field_name in (
            "automatic_render_box",
            "automatic_render_hard_bounds",
            "user_render_box",
            "effective_render_box",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    canonical_render_box(value, field_name=field_name),
                )
        if self.render_box_authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "render_box_authority must be automatic, user, or unavailable"
            )
        if self.render_box_authority == "automatic":
            if self.user_render_box is not None:
                raise ValueError(
                    "automatic render-box authority cannot carry a user value"
                )
            if self.effective_render_box != self.automatic_render_box:
                raise ValueError(
                    "automatic render-box authority must expose the automatic value"
                )
        elif self.render_box_authority == "user":
            if self.user_render_box is None:
                raise ValueError("user render-box authority requires a user value")
            if self.effective_render_box != self.user_render_box:
                raise ValueError(
                    "user render-box authority must expose the user value"
                )
        elif any(
            value is not None
            for value in (
                self.automatic_render_box,
                self.automatic_render_hard_bounds,
                self.user_render_box,
                self.effective_render_box,
            )
        ):
            raise ValueError("unavailable render-box authority cannot carry values")
        if self.render_box_authority != "unavailable":
            if (
                self.automatic_render_box is None
                or self.automatic_render_hard_bounds is None
                or self.effective_render_box is None
            ):
                raise ValueError(
                    "available render box requires automatic box and hard bounds"
                )
            if (
                not _box_contains(
                    self.automatic_render_hard_bounds,
                    self.automatic_render_box,
                )
                or not _box_contains(
                    self.automatic_render_hard_bounds,
                    self.effective_render_box,
                )
            ):
                raise ValueError("render box must stay inside automatic hard bounds")
        for field_name in (
            "automatic_font_role",
            "user_font_role",
            "effective_font_role",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    canonical_render_font_role(value, field_name=field_name),
                )
        if self.font_role_authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "font_role_authority must be automatic, user, or unavailable"
            )
        if self.font_role_authority == "automatic":
            if self.user_font_role is not None:
                raise ValueError(
                    "automatic font-role authority cannot carry a user value"
                )
            if self.effective_font_role != self.automatic_font_role:
                raise ValueError(
                    "automatic font-role authority must expose the automatic value"
                )
        elif self.font_role_authority == "user":
            if self.user_font_role is None:
                raise ValueError("user font-role authority requires a user value")
            if self.effective_font_role != self.user_font_role:
                raise ValueError(
                    "user font-role authority must expose the user value"
                )
        elif any(
            value is not None
            for value in (
                self.automatic_font_role,
                self.user_font_role,
                self.effective_font_role,
            )
        ):
            raise ValueError("unavailable font-role authority cannot carry values")
        for field_name in (
            "automatic_font_weight_tier",
            "user_font_weight_tier",
            "effective_font_weight_tier",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    canonical_render_font_weight_tier(
                        value,
                        field_name=field_name,
                    ),
                )
        if self.font_weight_tier_authority not in {
            "automatic",
            "user",
            "unavailable",
        }:
            raise ValueError(
                "font_weight_tier_authority must be automatic, user, or unavailable"
            )
        if self.font_weight_tier_authority == "automatic":
            if self.user_font_weight_tier is not None:
                raise ValueError(
                    "automatic font-weight authority cannot carry a user value"
                )
            if self.effective_font_weight_tier != self.automatic_font_weight_tier:
                raise ValueError(
                    "automatic font-weight authority must expose the automatic value"
                )
        elif self.font_weight_tier_authority == "user":
            if self.user_font_weight_tier is None:
                raise ValueError("user font-weight authority requires a user value")
            if self.effective_font_weight_tier != self.user_font_weight_tier:
                raise ValueError(
                    "user font-weight authority must expose the user value"
                )
        elif any(
            value is not None
            for value in (
                self.automatic_font_weight_tier,
                self.user_font_weight_tier,
                self.effective_font_weight_tier,
            )
        ):
            raise ValueError("unavailable font-weight authority cannot carry values")
        for field_name in (
            "automatic_fill_color",
            "user_fill_color",
            "effective_fill_color",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    canonical_render_fill_color(value, field_name=field_name),
                )
        if (
            self.unresolved_user_fill_color is not None
            and not isinstance(self.unresolved_user_fill_color, str)
        ):
            raise TypeError("unresolved_user_fill_color must be a string or None")
        if self.fill_color_authority not in {
            "automatic",
            "user",
            "unresolved",
            "unavailable",
        }:
            raise ValueError(
                "fill_color_authority must be automatic, user, unresolved, or unavailable"
            )
        if self.fill_color_authority == "automatic":
            if self.user_fill_color is not None:
                raise ValueError(
                    "automatic fill-color authority cannot carry a user value"
                )
            if self.unresolved_user_fill_color is not None:
                raise ValueError(
                    "automatic fill-color authority cannot carry an unresolved value"
                )
            if self.effective_fill_color != self.automatic_fill_color:
                raise ValueError(
                    "automatic fill-color authority must expose the automatic value"
                )
        elif self.fill_color_authority == "user":
            if self.user_fill_color is None:
                raise ValueError("user fill-color authority requires a user value")
            if self.unresolved_user_fill_color is not None:
                raise ValueError(
                    "user fill-color authority cannot carry an unresolved value"
                )
            if self.effective_fill_color != self.user_fill_color:
                raise ValueError(
                    "user fill-color authority must expose the user value"
                )
        elif self.fill_color_authority == "unresolved":
            if self.user_fill_color is not None:
                raise ValueError(
                    "unresolved fill-color authority cannot carry a canonical user value"
                )
            if self.effective_fill_color is not None:
                raise ValueError(
                    "unresolved fill-color authority cannot expose an effective value"
                )
        elif any(
            value is not None
            for value in (
                self.automatic_fill_color,
                self.user_fill_color,
                self.unresolved_user_fill_color,
                self.effective_fill_color,
            )
        ):
            raise ValueError("unavailable fill-color authority cannot carry values")
        for field_name in (
            "automatic_outline_color",
            "user_outline_color",
            "effective_outline_color",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    canonical_render_outline_color(value, field_name=field_name),
                )
        if (
            self.unresolved_user_outline_color is not None
            and not isinstance(self.unresolved_user_outline_color, str)
        ):
            raise TypeError("unresolved_user_outline_color must be a string or None")
        if self.outline_color_authority not in {
            "automatic",
            "user",
            "unresolved",
            "unavailable",
        }:
            raise ValueError(
                "outline_color_authority must be automatic, user, unresolved, or unavailable"
            )
        if self.outline_color_authority == "automatic":
            if self.user_outline_color is not None or self.unresolved_user_outline_color is not None:
                raise ValueError("automatic outline-color authority cannot carry a user value")
            if self.effective_outline_color != self.automatic_outline_color:
                raise ValueError("automatic outline-color authority must expose the automatic value")
        elif self.outline_color_authority == "user":
            if self.user_outline_color is None or self.unresolved_user_outline_color is not None:
                raise ValueError("user outline-color authority requires one canonical user value")
            if self.effective_outline_color != self.user_outline_color:
                raise ValueError("user outline-color authority must expose the user value")
        elif self.outline_color_authority == "unresolved":
            if self.user_outline_color is not None or self.effective_outline_color is not None:
                raise ValueError("unresolved outline-color authority cannot expose an effective value")
        elif any(
            value is not None
            for value in (
                self.automatic_outline_color,
                self.user_outline_color,
                self.unresolved_user_outline_color,
                self.effective_outline_color,
            )
        ):
            raise ValueError("unavailable outline-color authority cannot carry values")
        for field_name in (
            "automatic_outline_width",
            "user_outline_width",
            "effective_outline_width",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    canonical_render_outline_width(value, field_name=field_name),
                )
        if self.outline_width_authority not in {
            "automatic",
            "user",
            "unavailable",
        }:
            raise ValueError(
                "outline_width_authority must be automatic, user, or unavailable"
            )
        if self.outline_width_authority == "automatic":
            if self.user_outline_width is not None:
                raise ValueError(
                    "automatic outline-width authority cannot carry a user value"
                )
            if self.effective_outline_width != self.automatic_outline_width:
                raise ValueError(
                    "automatic outline-width authority must expose the automatic value"
                )
        elif self.outline_width_authority == "user":
            if self.user_outline_width is None:
                raise ValueError(
                    "user outline-width authority requires a user value"
                )
            if self.effective_outline_width != self.user_outline_width:
                raise ValueError(
                    "user outline-width authority must expose the user value"
                )
        elif any(
            value is not None
            for value in (
                self.automatic_outline_width,
                self.user_outline_width,
                self.effective_outline_width,
            )
        ):
            raise ValueError("unavailable outline-width authority cannot carry values")
        for field_name in (
            "automatic_preferred_size",
            "user_preferred_size",
            "effective_preferred_size",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    canonical_render_preferred_size(value, field_name=field_name),
                )
        if self.preferred_size_authority not in {
            "automatic",
            "user",
            "unavailable",
        }:
            raise ValueError(
                "preferred_size_authority must be automatic, user, or unavailable"
            )
        if self.preferred_size_authority == "automatic":
            if self.user_preferred_size is not None:
                raise ValueError(
                    "automatic preferred-size authority cannot carry a user value"
                )
            if self.effective_preferred_size != self.automatic_preferred_size:
                raise ValueError(
                    "automatic preferred-size authority must expose the automatic value"
                )
        elif self.preferred_size_authority == "user":
            if self.user_preferred_size is None:
                raise ValueError(
                    "user preferred-size authority requires a user value"
                )
            if self.effective_preferred_size != self.user_preferred_size:
                raise ValueError(
                    "user preferred-size authority must expose the user value"
                )
        elif any(
            value is not None
            for value in (
                self.automatic_preferred_size,
                self.user_preferred_size,
                self.effective_preferred_size,
            )
        ):
            raise ValueError("unavailable preferred-size authority cannot carry values")
        for field_name in (
            "automatic_shadow_blur",
            "user_shadow_blur",
            "effective_shadow_blur",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    canonical_render_shadow_blur(value, field_name=field_name),
                )
        if self.shadow_blur_authority not in {
            "automatic",
            "user",
            "unavailable",
        }:
            raise ValueError(
                "shadow_blur_authority must be automatic, user, or unavailable"
            )
        if self.shadow_blur_authority == "automatic":
            if self.user_shadow_blur is not None:
                raise ValueError(
                    "automatic shadow-blur authority cannot carry a user value"
                )
            if self.effective_shadow_blur != self.automatic_shadow_blur:
                raise ValueError(
                    "automatic shadow-blur authority must expose the automatic value"
                )
        elif self.shadow_blur_authority == "user":
            if self.user_shadow_blur is None:
                raise ValueError("user shadow-blur authority requires a user value")
            if self.effective_shadow_blur != self.user_shadow_blur:
                raise ValueError(
                    "user shadow-blur authority must expose the user value"
                )
        elif any(
            value is not None
            for value in (
                self.automatic_shadow_blur,
                self.user_shadow_blur,
                self.effective_shadow_blur,
            )
        ):
            raise ValueError("unavailable shadow-blur authority cannot carry values")
        for field_name in (
            "automatic_shadow_color",
            "user_shadow_color",
            "effective_shadow_color",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    canonical_render_shadow_color(value, field_name=field_name),
                )
        if self.shadow_color_authority not in {
            "automatic",
            "user",
            "unavailable",
        }:
            raise ValueError(
                "shadow_color_authority must be automatic, user, or unavailable"
            )
        if self.shadow_color_authority == "automatic":
            if self.user_shadow_color is not None:
                raise ValueError(
                    "automatic shadow-color authority cannot carry a user value"
                )
            if self.effective_shadow_color != self.automatic_shadow_color:
                raise ValueError(
                    "automatic shadow-color authority must expose the automatic value"
                )
        elif self.shadow_color_authority == "user":
            if self.user_shadow_color is None:
                raise ValueError("user shadow-color authority requires a user value")
            if self.effective_shadow_color != self.user_shadow_color:
                raise ValueError(
                    "user shadow-color authority must expose the user value"
                )
        elif any(
            value is not None
            for value in (
                self.automatic_shadow_color,
                self.user_shadow_color,
                self.effective_shadow_color,
            )
        ):
            raise ValueError("unavailable shadow-color authority cannot carry values")
        for field_name in (
            "automatic_shadow_offset",
            "user_shadow_offset",
            "effective_shadow_offset",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    canonical_render_shadow_offset(value, field_name=field_name),
                )
        if self.shadow_offset_authority not in {
            "automatic",
            "user",
            "unavailable",
        }:
            raise ValueError(
                "shadow_offset_authority must be automatic, user, or unavailable"
            )
        if self.shadow_offset_authority == "automatic":
            if self.user_shadow_offset is not None:
                raise ValueError(
                    "automatic shadow-offset authority cannot carry a user value"
                )
            if self.effective_shadow_offset != self.automatic_shadow_offset:
                raise ValueError(
                    "automatic shadow-offset authority must expose the automatic value"
                )
        elif self.shadow_offset_authority == "user":
            if self.user_shadow_offset is None:
                raise ValueError("user shadow-offset authority requires a user value")
            if self.effective_shadow_offset != self.user_shadow_offset:
                raise ValueError(
                    "user shadow-offset authority must expose the user value"
                )
        elif any(
            value is not None
            for value in (
                self.automatic_shadow_offset,
                self.user_shadow_offset,
                self.effective_shadow_offset,
            )
        ):
            raise ValueError("unavailable shadow-offset authority cannot carry values")
        for field_name in (
            "automatic_shadow_enabled",
            "user_shadow_enabled",
            "effective_shadow_enabled",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{field_name} must be a boolean or None")
        if self.shadow_enabled_authority not in {
            "automatic",
            "user",
            "unavailable",
        }:
            raise ValueError(
                "shadow_enabled_authority must be automatic, user, or unavailable"
            )
        if self.shadow_enabled_authority == "automatic":
            if self.user_shadow_enabled is not None:
                raise ValueError(
                    "automatic shadow-visibility authority cannot carry a user value"
                )
            if self.effective_shadow_enabled != self.automatic_shadow_enabled:
                raise ValueError(
                    "automatic shadow-visibility authority must expose the automatic value"
                )
        elif self.shadow_enabled_authority == "user":
            if self.user_shadow_enabled is not False:
                raise ValueError(
                    "user shadow-visibility authority requires exactly false"
                )
            if self.effective_shadow_enabled is not False:
                raise ValueError(
                    "user shadow-visibility authority must expose false"
                )
        elif any(
            value is not None
            for value in (
                self.automatic_shadow_enabled,
                self.user_shadow_enabled,
                self.effective_shadow_enabled,
            )
        ):
            raise ValueError(
                "unavailable shadow-visibility authority cannot carry values"
            )
        if not isinstance(self.render_required, bool):
            raise TypeError("render_required must be a boolean")
        if not isinstance(self.writing_mode_unavailable_reason, str):
            raise TypeError("writing_mode_unavailable_reason must be a string")
        if not isinstance(self.line_height_unavailable_reason, str):
            raise TypeError("line_height_unavailable_reason must be a string")
        if not isinstance(self.rotation_unavailable_reason, str):
            raise TypeError("rotation_unavailable_reason must be a string")
        if not isinstance(self.render_box_unavailable_reason, str):
            raise TypeError("render_box_unavailable_reason must be a string")
        if not isinstance(self.font_role_unavailable_reason, str):
            raise TypeError("font_role_unavailable_reason must be a string")
        if not isinstance(self.font_weight_tier_unavailable_reason, str):
            raise TypeError("font_weight_tier_unavailable_reason must be a string")
        if not isinstance(self.fill_color_unavailable_reason, str):
            raise TypeError("fill_color_unavailable_reason must be a string")
        if not isinstance(self.outline_color_unavailable_reason, str):
            raise TypeError("outline_color_unavailable_reason must be a string")
        if not isinstance(self.outline_width_unavailable_reason, str):
            raise TypeError("outline_width_unavailable_reason must be a string")
        if not isinstance(self.preferred_size_unavailable_reason, str):
            raise TypeError("preferred_size_unavailable_reason must be a string")
        if not isinstance(self.shadow_blur_unavailable_reason, str):
            raise TypeError("shadow_blur_unavailable_reason must be a string")
        if not isinstance(self.shadow_color_unavailable_reason, str):
            raise TypeError("shadow_color_unavailable_reason must be a string")
        if not isinstance(self.shadow_offset_unavailable_reason, str):
            raise TypeError("shadow_offset_unavailable_reason must be a string")
        if not isinstance(self.shadow_visibility_unavailable_reason, str):
            raise TypeError(
                "shadow_visibility_unavailable_reason must be a string"
            )
        if self.effective.origin is ParentOrigin.AUTOMATIC:
            if self.selected_model_source_revision is not None:
                raise ValueError(
                    "automatic parent cannot expose a selected model source revision"
                )
            if self.selected_model_translation_revision is not None:
                raise ValueError(
                    "automatic parent cannot expose a selected model translation revision"
                )
            if self.target_text_revision_base is not None:
                raise ValueError(
                    "automatic parent cannot expose a target revision base"
                )
            if (
                self.mapped_pipeline_source_text is not None
                or self.mapped_pipeline_target_text is not None
            ):
                raise ValueError(
                    "automatic parent cannot expose mapped pipeline evidence"
                )
            if self.automatic_source_text is None or self.automatic_target_text is None:
                raise ValueError("automatic parent requires automatic text evidence")
            if self.parent_row.origin is not ParentOrigin.AUTOMATIC:
                raise ValueError("automatic projected parent row has another origin")
        else:
            if self.parent_row.origin is not ParentOrigin.USER:
                raise ValueError("user projected parent row has another origin")
            if self.automatic_source_text is not None or self.automatic_target_text is not None:
                raise ValueError("user parent cannot expose automatic text evidence")
            mapping = self.effective.source_evidence_mapping
            selected_revision = self.selected_model_source_revision
            selected_target = self.selected_model_translation_revision
            if mapping is not None:
                if (
                    selected_revision is not None
                    or selected_target is not None
                    or self.target_text_revision_base is not None
                    or self.effective.source_revision_id is not None
                    or self.effective.target_revision_id is not None
                    or self.mapped_pipeline_source_text != mapping.source_text
                    or self.mapped_pipeline_target_text != mapping.target_text
                    or self.effective.source_authority != "user"
                    or self.effective.source_text != mapping.source_text
                    or self.user_source_text is not None
                ):
                    raise ValueError(
                        "mapped user parent source provenance is inconsistent"
                    )
                if self.effective.target_authority == "unavailable":
                    if (
                        mapping.target_text is not None
                        or self.effective.target_text is not None
                        or self.user_target_text is not None
                    ):
                        raise ValueError(
                            "mapped target-unavailable provenance is inconsistent"
                        )
                elif self.effective.target_authority == "mapped_automatic":
                    if (
                        mapping.target_text is None
                        or self.effective.target_text != mapping.target_text
                        or self.user_target_text is not None
                    ):
                        raise ValueError(
                            "mapped pipeline target provenance is inconsistent"
                        )
                elif self.effective.target_authority == "user":
                    if (
                        mapping.target_text is None
                        or self.user_target_text is None
                        or self.user_target_text != self.effective.target_text
                    ):
                        raise ValueError(
                            "mapped user target override provenance is inconsistent"
                        )
                else:
                    raise ValueError(
                        "mapped user parent target authority is unsupported"
                    )
            else:
                if (
                    self.mapped_pipeline_source_text is not None
                    or self.mapped_pipeline_target_text is not None
                ):
                    raise ValueError(
                        "unmapped user parent cannot expose mapped pipeline evidence"
                    )
            if mapping is not None:
                pass
            elif self.effective.source_authority == "unavailable":
                if (
                    selected_revision is not None
                    or self.effective.source_text is not None
                    or self.user_source_text is not None
                ):
                    raise ValueError(
                        "source-unavailable user parent cannot expose source evidence"
                    )
            elif self.effective.source_authority == "ocr_revision":
                if (
                    selected_revision is None
                    or self.user_source_text is not None
                    or self.effective.source_text != selected_revision.source_text
                    or self.effective.source_revision_id != selected_revision.revision_id
                ):
                    raise ValueError(
                        "user parent selected model revision provenance is inconsistent"
                    )
            elif self.effective.source_authority == "user":
                if (
                    self.user_source_text is None
                    or self.user_source_text != self.effective.source_text
                ):
                    raise ValueError(
                        "user source authority requires one exact source edit"
                    )
            else:
                raise ValueError("user parent source authority is unsupported")
            if mapping is not None:
                pass
            elif self.effective.target_authority == "unavailable":
                if (
                    selected_target is not None
                    or self.target_text_revision_base is not None
                    or self.user_target_text is not None
                    or self.effective.target_text is not None
                    or self.effective.target_revision_id is not None
                ):
                    raise ValueError(
                        "target-unavailable user parent cannot expose target evidence"
                    )
            elif self.effective.target_authority == "translation_revision":
                if (
                    selected_target is None
                    or self.target_text_revision_base is None
                    or self.user_target_text is not None
                    or self.effective.target_text != selected_target.target_text
                    or self.effective.target_revision_id != selected_target.revision_id
                ):
                    raise ValueError(
                        "user parent selected model translation provenance is inconsistent"
                    )
            elif self.effective.target_authority == "user":
                if (
                    selected_target is None
                    or self.target_text_revision_base is None
                    or self.user_target_text is None
                    or self.user_target_text != self.effective.target_text
                    or self.effective.target_revision_id
                    != selected_target.revision_id
                ):
                    raise ValueError(
                        "user target authority requires one exact revision-backed edit"
                    )
            else:
                raise ValueError("user parent target authority is unsupported")
            if self.effective_render_style or self.effective_render_layout:
                raise ValueError("pending user parent cannot expose style or layout facts")
            if self.writing_mode_authority != "unavailable":
                raise ValueError("user parent writing mode must remain unavailable")
            if self.line_height_authority != "unavailable":
                raise ValueError("user parent line height must remain unavailable")
            if self.rotation_authority != "unavailable":
                raise ValueError("user parent rotation must remain unavailable")
            if self.render_box_authority != "unavailable":
                raise ValueError("user parent render box must remain unavailable")
            if self.font_role_authority != "unavailable":
                raise ValueError("user parent font role must remain unavailable")
            if self.font_weight_tier_authority != "unavailable":
                raise ValueError("user parent font weight must remain unavailable")
            if self.fill_color_authority != "unavailable":
                raise ValueError("user parent fill color must remain unavailable")
            if self.outline_color_authority != "unavailable":
                raise ValueError("user parent outline color must remain unavailable")
            if self.render_required:
                raise ValueError("pending user parent cannot claim render eligibility")
        line_height_eligible = bool(
            not self.effective.excluded
            and self.render_required
            and self.automatic_line_height is not None
            and self.effective_line_height is not None
        )
        if line_height_eligible and self.line_height_unavailable_reason:
            raise ValueError(
                "eligible line-height projection cannot carry an unavailable reason"
            )
        if not line_height_eligible and not self.line_height_unavailable_reason:
            raise ValueError(
                "ineligible line-height projection requires an unavailable reason"
            )
        rotation_eligible = bool(
            not self.effective.excluded
            and self.render_required
            and self.automatic_rotation is not None
            and self.effective_rotation is not None
        )
        if rotation_eligible and self.rotation_unavailable_reason:
            raise ValueError(
                "eligible rotation projection cannot carry an unavailable reason"
            )
        if not rotation_eligible and not self.rotation_unavailable_reason:
            raise ValueError(
                "ineligible rotation projection requires an unavailable reason"
            )
        render_box_eligible = bool(
            not self.effective.excluded
            and self.render_required
            and self.automatic_render_box is not None
            and self.automatic_render_hard_bounds is not None
            and self.effective_render_box is not None
        )
        if render_box_eligible and self.render_box_unavailable_reason:
            raise ValueError(
                "eligible render-box projection cannot carry an unavailable reason"
            )
        if not render_box_eligible and not self.render_box_unavailable_reason:
            raise ValueError(
                "ineligible render-box projection requires an unavailable reason"
            )
        font_role_eligible = bool(
            not self.effective.excluded
            and self.render_required
            and self.automatic_font_role is not None
            and self.effective_font_role is not None
        )
        if font_role_eligible and self.font_role_unavailable_reason:
            raise ValueError(
                "eligible font-role projection cannot carry an unavailable reason"
            )
        if not font_role_eligible and not self.font_role_unavailable_reason:
            raise ValueError(
                "ineligible font-role projection requires an unavailable reason"
            )
        font_weight_tier_eligible = bool(
            not self.effective.excluded
            and self.render_required
            and self.automatic_font_weight_tier is not None
            and self.effective_font_weight_tier is not None
        )
        if font_weight_tier_eligible and self.font_weight_tier_unavailable_reason:
            raise ValueError(
                "eligible font-weight projection cannot carry an unavailable reason"
            )
        if (
            not font_weight_tier_eligible
            and not self.font_weight_tier_unavailable_reason
        ):
            raise ValueError(
                "ineligible font-weight projection requires an unavailable reason"
            )
        fill_color_eligible = bool(
            not self.effective.excluded
            and self.render_required
            and self.automatic_fill_color is not None
            and self.effective_fill_color is not None
            and self.fill_color_authority != "unresolved"
        )
        if fill_color_eligible and self.fill_color_unavailable_reason:
            raise ValueError(
                "eligible fill-color projection cannot carry an unavailable reason"
            )
        if not fill_color_eligible and not self.fill_color_unavailable_reason:
            raise ValueError(
                "ineligible fill-color projection requires an unavailable reason"
            )
        outline_color_eligible = bool(
            not self.effective.excluded
            and self.render_required
            and self.automatic_outline_color is not None
            and self.effective_outline_color is not None
            and self.outline_color_authority != "unresolved"
        )
        if outline_color_eligible and self.outline_color_unavailable_reason:
            raise ValueError(
                "eligible outline-color projection cannot carry an unavailable reason"
            )
        if not outline_color_eligible and not self.outline_color_unavailable_reason:
            raise ValueError(
                "ineligible outline-color projection requires an unavailable reason"
            )

    @property
    def origin(self) -> ParentOrigin:
        return self.effective.origin

    @property
    def identity_namespace(self) -> ParentIdentityNamespace:
        return self.effective.identity_namespace

    @property
    def root_identity_namespace(self) -> RootIdentityNamespace:
        return self.effective.root_identity_namespace

    @property
    def workflow_area_bbox(self) -> tuple[int, int, int, int] | None:
        value = thaw_json(self.effective.workflow_area_bbox)
        if value is None:
            return None
        if (
            not isinstance(value, list)
            or len(value) != 4
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise ValueError("projected workflow area must contain four exact integers")
        return tuple(value)

    @property
    def lineage(self) -> EffectiveParentLineage | None:
        return self.effective.lineage

    @property
    def stage_requirements(self) -> tuple[ParentStageRequirement, ...]:
        return self.effective.stage_requirements

    @property
    def automatic_evidence_available(self) -> bool:
        return self.origin is ParentOrigin.AUTOMATIC

    @property
    def execution_ready(self) -> bool:
        return all(
            item.state is RevisionStageState.CURRENT
            for item in self.stage_requirements
        )


@dataclass(frozen=True, slots=True)
class ProjectedPage:
    page_row: PageRow
    effective: EffectivePageSnapshot
    parents: tuple[ProjectedParent, ...]
    automatic_ordered_parent_ids: tuple[str, ...]
    canvas_artifacts: CanvasArtifactReferences
    canvas_size: tuple[int, int] | None
    original_page_binding: OriginalPageAssetBinding | None
    original_page_binding_problem: str
    overlays: tuple[OverlayShapeData, ...]
    raster_overlays: tuple[RasterOverlayData, ...]
    overlay_availability: tuple[OverlayAvailabilityData, ...]
    original_artifact_state: ArtifactState
    cleaned_artifact_state: ArtifactState
    final_artifact_state: ArtifactState
    cleanup_state: CleanupState
    edit_history: tuple[EditHistoryReference, ...]
    artifact_history: tuple[ArtifactRevisionReference, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.page_row, PageRow):
            raise TypeError("page_row must be PageRow")
        if not isinstance(self.effective, EffectivePageSnapshot):
            raise TypeError("effective must be EffectivePageSnapshot")
        if self.page_row.page_id != self.effective.page_id:
            raise ValueError("page row and effective snapshot identities differ")
        if self.canvas_artifacts.page_id != self.effective.page_id:
            raise ValueError("canvas and effective page identities differ")
        if self.canvas_size is not None and (
            not isinstance(self.canvas_size, tuple)
            or len(self.canvas_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.canvas_size
            )
            or self.canvas_size[0] * self.canvas_size[1] > 50_000_000
        ):
            raise ValueError(
                "canvas_size must contain two positive integers within the safety limit"
            )
        if (
            self.original_page_binding is not None
            and not isinstance(self.original_page_binding, OriginalPageAssetBinding)
        ):
            raise TypeError(
                "original_page_binding must be OriginalPageAssetBinding or None"
            )
        if not isinstance(self.original_page_binding_problem, str):
            raise TypeError("original_page_binding_problem must be a string")
        binding_problem = self.original_page_binding_problem.strip()
        if self.original_page_binding is None and not binding_problem:
            raise ValueError("missing original-page binding requires a reason")
        if self.original_page_binding is not None and binding_problem:
            raise ValueError("available original-page binding cannot have a reason")
        if (
            self.original_page_binding is not None
            and self.canvas_size is not None
            and self.original_page_binding.canvas_size != self.canvas_size
        ):
            raise ValueError("original-page binding and projected canvas differ")
        object.__setattr__(
            self,
            "original_page_binding_problem",
            binding_problem,
        )
        if any(not isinstance(parent, ProjectedParent) for parent in self.parents):
            raise TypeError("parents must contain ProjectedParent values")
        parent_ids = tuple(parent.effective.parent_id for parent in self.parents)
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError("projected parent identities must be unique")
        if parent_ids != self.effective.hierarchy.ordered_parent_ids:
            raise ValueError("projected parents do not match effective hierarchy order")
        if (
            not isinstance(self.automatic_ordered_parent_ids, tuple)
            or any(
                not isinstance(parent_id, str) or not parent_id
                for parent_id in self.automatic_ordered_parent_ids
            )
            or len(self.automatic_ordered_parent_ids)
            != len(set(self.automatic_ordered_parent_ids))
        ):
            raise ValueError("automatic reading order must contain unique parent IDs")
        automatic_parent_ids = tuple(
            parent.effective.parent_id
            for parent in self.parents
            if parent.origin is ParentOrigin.AUTOMATIC
        )
        merge_consumed_automatic_ids = tuple(
            source_parent_id
            for parent in self.parents
            if parent.origin is ParentOrigin.USER
            and parent.effective.lineage is not None
            and parent.effective.lineage.order_policy == "replace_sources"
            for source_parent_id in parent.effective.lineage.source_parent_ids
        )
        if frozenset(self.automatic_ordered_parent_ids) != frozenset(
            (*automatic_parent_ids, *merge_consumed_automatic_ids)
        ) or frozenset(automatic_parent_ids).intersection(
            merge_consumed_automatic_ids
        ):
            raise ValueError(
                "automatic reading order must contain exactly active or "
                "merge-retained pipeline parents"
            )
        if any(not isinstance(item, OverlayShapeData) for item in self.overlays):
            raise TypeError("overlays must contain OverlayShapeData values")
        known_parent_ids = frozenset(parent_ids)
        if any(
            item.parent_id and item.parent_id not in known_parent_ids
            for item in self.overlays
        ):
            raise ValueError("overlay references an unknown projected parent")
        if any(
            not isinstance(item, RasterOverlayData) for item in self.raster_overlays
        ):
            raise TypeError("raster_overlays must contain RasterOverlayData values")
        raster_ids = tuple(item.overlay_id for item in self.raster_overlays)
        if len(raster_ids) != len(set(raster_ids)):
            raise ValueError("raster overlay identities must be unique")
        if any(
            not isinstance(item, OverlayAvailabilityData)
            for item in self.overlay_availability
        ):
            raise TypeError(
                "overlay_availability must contain OverlayAvailabilityData values"
            )
        if tuple(item.overlay_id for item in self.overlay_availability) != OVERLAY_IDS:
            raise ValueError("overlay availability must exactly match the UI contract")
        if any(
            not isinstance(item, EditHistoryReference)
            for item in self.edit_history
        ):
            raise TypeError(
                "edit_history must contain EditHistoryReference values"
            )
        if any(
            not isinstance(item, ArtifactRevisionReference)
            for item in self.artifact_history
        ):
            raise TypeError(
                "artifact_history must contain ArtifactRevisionReference values"
            )
        for field_name in (
            "original_artifact_state",
            "cleaned_artifact_state",
            "final_artifact_state",
        ):
            object.__setattr__(
                self,
                field_name,
                ArtifactState(getattr(self, field_name)),
            )
        object.__setattr__(self, "cleanup_state", CleanupState(self.cleanup_state))

    @property
    def parent_rows(self) -> tuple[ParentRow, ...]:
        return tuple(parent.parent_row for parent in self.parents)

    @property
    def hierarchy_revision(self) -> HierarchyRevisionDescriptor:
        descriptor = self.effective.hierarchy.descriptor
        if not isinstance(descriptor, HierarchyRevisionDescriptor):
            raise ValueError("effective page is missing its hierarchy descriptor")
        return descriptor

    @property
    def user_roots(self) -> tuple[EffectiveUserRootSnapshot, ...]:
        return self.effective.hierarchy.user_roots

    @property
    def stage_requirements(self) -> tuple[ParentStageRequirement, ...]:
        return self.effective.stage_requirements

    @property
    def execution_ready(self) -> bool:
        return self.effective.execution_ready

    def parent(self, parent_id: str) -> ProjectedParent:
        identity = _required_text(parent_id, "parent_id")
        for parent in self.parents:
            if parent.effective.parent_id == identity:
                return parent
        raise KeyError(f"projected parent is missing: {identity}")

    def user_parent_cleanup_coverage_target(
        self,
        parent_id: str,
    ) -> UserParentCleanupCoverageTargetV1:
        """Build the exact public coverage target without duplicating lineage."""

        identity = _required_text(parent_id, "parent_id")
        self.parent(identity)
        if self.original_page_binding is None:
            raise ValueError(self.original_page_binding_problem)
        return user_parent_cleanup_coverage_target_from_snapshot(
            self.effective,
            identity,
            original_page=self.original_page_binding,
        )

    def overlays_for_parent(
        self,
        parent_id: str | None,
    ) -> tuple[OverlayShapeData, ...]:
        """Return immutable overlay values with one parent's shapes selected."""

        identity = str(parent_id or "").strip()
        if identity:
            self.parent(identity)
        return tuple(
            replace(
                shape,
                selected=bool(identity) and shape.parent_id == identity,
            )
            for shape in self.overlays
        )


@dataclass(frozen=True, slots=True)
class ProjectUiProjection:
    metadata: ProjectMetadata
    source_project_fingerprint: str
    project_row: ProjectRow
    page_rows: tuple[PageRow, ...]
    pages: tuple[ProjectedPage, ...]
    glossary_entries: tuple[GlossaryEntryV1, ...]
    glossary_fingerprint: str
    glossary_history: tuple[EditHistoryReference, ...]
    glossary_stale_page_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, ProjectMetadata):
            raise TypeError("metadata must be ProjectMetadata")
        source_fingerprint = str(self.source_project_fingerprint or "").strip().lower()
        if len(source_fingerprint) != 64 or any(
            value not in "0123456789abcdef" for value in source_fingerprint
        ):
            raise ValueError("source_project_fingerprint must be a SHA-256 digest")
        object.__setattr__(
            self,
            "source_project_fingerprint",
            source_fingerprint,
        )
        if not isinstance(self.project_row, ProjectRow):
            raise TypeError("project_row must be ProjectRow")
        if self.project_row.project_id != self.metadata.project_id:
            raise ValueError("project row and metadata identities differ")
        if any(not isinstance(page, ProjectedPage) for page in self.pages):
            raise TypeError("pages must contain ProjectedPage values")
        expected_rows = tuple(page.page_row for page in self.pages)
        if self.page_rows != expected_rows:
            raise ValueError("page_rows must exactly match projected pages")
        page_ids = tuple(page.page_row.page_id for page in self.pages)
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("projected page identities must be unique")
        if len(self.pages) != self.metadata.page_count:
            raise ValueError("metadata page_count does not match projected pages")
        if any(
            not isinstance(entry, GlossaryEntryV1)
            for entry in self.glossary_entries
        ):
            raise TypeError("glossary_entries must contain GlossaryEntryV1 values")
        if tuple(entry.entry_id for entry in self.glossary_entries) != tuple(
            sorted(entry.entry_id for entry in self.glossary_entries)
        ):
            raise ValueError("project glossary entries must be sorted")
        fingerprint = str(self.glossary_fingerprint or "").strip().lower()
        if len(fingerprint) != 64 or any(
            value not in "0123456789abcdef" for value in fingerprint
        ):
            raise ValueError("glossary_fingerprint must be a SHA-256 digest")
        object.__setattr__(self, "glossary_fingerprint", fingerprint)
        if any(
            not isinstance(item, EditHistoryReference)
            for item in self.glossary_history
        ):
            raise TypeError(
                "glossary_history must contain EditHistoryReference values"
            )
        if len(self.glossary_stale_page_ids) != len(
            set(self.glossary_stale_page_ids)
        ) or any(
            page_id not in set(page_ids)
            for page_id in self.glossary_stale_page_ids
        ):
            raise ValueError("glossary stale-page identities are invalid")

    def page(self, page_id: str) -> ProjectedPage:
        identity = _required_text(page_id, "page_id")
        for page in self.pages:
            if page.effective.page_id == identity:
                return page
        raise KeyError(f"projected page is missing: {identity}")


def _normalized_project_path(project_path: str) -> str:
    value = _required_text(project_path, "project_path")
    return str(Path(os.path.expandvars(value)).expanduser().resolve(strict=False))


def _available_asset_path(value: object, *, project_path: str) -> str | None:
    raw = _optional_text(value)
    if raw is None:
        return None
    candidate = Path(os.path.expandvars(raw)).expanduser()
    if not candidate.is_absolute():
        candidate = Path(project_path).parent / candidate
    candidate = candidate.resolve(strict=False)
    try:
        return str(candidate) if candidate.is_file() else None
    except OSError:
        return None


def _validated_asset_path(
    value: object,
    expected_sha256: object,
    *,
    project_path: str,
) -> tuple[str, str] | None:
    digest = str(expected_sha256 or "").strip().lower()
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        return None
    path = _available_asset_path(value, project_path=project_path)
    if path is None:
        return None
    actual = hashlib.sha256()
    try:
        with open(path, "rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                actual.update(chunk)
    except OSError:
        return None
    return (path, digest) if actual.hexdigest() == digest else None


def _canvas_size(value: object) -> tuple[int, int] | None:
    if (
        isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, Sequence)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in value
        )
    ):
        return None
    return int(value[0]), int(value[1])


def _record_asset(record: Mapping[str, Any]) -> object:
    direct = record.get("asset") or record.get("image_path") or record.get("path")
    if direct:
        return direct
    nested = record.get("artifact")
    if isinstance(nested, Mapping):
        return nested.get("asset") or nested.get("image_path") or nested.get("path")
    nested = record.get("cleaned_page_base")
    if isinstance(nested, Mapping):
        return nested.get("image_path") or nested.get("cache_path")
    return None


def _record_state(
    record: Mapping[str, Any],
    *,
    project_path: str,
    stale: bool = False,
) -> tuple[str | None, ArtifactState]:
    asset_path = _available_asset_path(_record_asset(record), project_path=project_path)
    if record.get("valid") is False:
        return asset_path, ArtifactState.INVALID
    if asset_path is None:
        return None, ArtifactState.MISSING
    if stale:
        return asset_path, ArtifactState.STALE
    return asset_path, ArtifactState.VALID


def _catalogs(project: Mapping[str, Any]) -> Mapping[str, Any]:
    catalogs = project.get("artifact_revisions")
    if not isinstance(catalogs, Mapping):
        raise ValueError("artifact revision catalog is missing")
    return catalogs


def _catalog_page_records(
    project: Mapping[str, Any],
    *,
    catalog_id: str,
    page_id: str,
    known_page_ids: frozenset[str],
) -> tuple[Mapping[str, Any], ...]:
    values = _catalogs(project).get(catalog_id)
    if not isinstance(values, list):
        raise ValueError(f"artifact catalog {catalog_id} must be a list")
    result: list[Mapping[str, Any]] = []
    for record in values:
        if not isinstance(record, Mapping):
            raise ValueError(f"artifact catalog {catalog_id} records must be mappings")
        record_page_id = _required_text(record.get("page_id"), "artifact.page_id")
        if record_page_id not in known_page_ids:
            raise ValueError(
                f"artifact revision references an unknown page: {record_page_id}"
            )
        _required_text(record.get("revision_id"), "artifact.revision_id")
        if record_page_id == page_id:
            result.append(record)
    return tuple(result)


def _base_parent_map(page: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    values = page.get("parent_execution_bundles") or []
    if not isinstance(values, (list, tuple)):
        raise ValueError("parent_execution_bundles must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    page_id = _required_text(page.get("page_id"), "page.page_id")
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("parent execution bundles must be mappings")
        parent_id = _required_text(value.get("parent_id"), "parent.parent_id")
        if parent_id in result:
            raise ValueError(f"parent identity is duplicated: {parent_id}")
        if _required_text(value.get("page_id"), "parent.page_id") != page_id:
            raise ValueError("parent execution bundle page identity differs from page")
        result[parent_id] = value
    return result


def _automatic_source_text(parent: Mapping[str, Any]) -> str:
    value = parent.get("source_text")
    if value is None:
        value = parent.get("ocr_text")
    return str(value or "")


def _automatic_target_text(parent: Mapping[str, Any]) -> str:
    value = parent.get("translated_text")
    if value is None:
        value = parent.get("translation")
    return str(value or "")


def _authority_values(
    parent: EffectiveParentSnapshot,
    *,
    automatic_target_text: str,
) -> tuple[Authority, Authority]:
    source = (
        Authority.AUTOMATIC
        if parent.source_authority == "automatic"
        else Authority.USER_EDIT
    )
    # A source correction makes the automatic target historical, not
    # user-authored.  Only an explicit target override may promote authority to
    # Your edit; freshness is presented independently as Stale translation.
    if parent.target_authority == "automatic":
        target = Authority.AUTOMATIC
    elif parent.target_text == automatic_target_text:
        # Keep Existing Target publishes the exact historical automatic value
        # as a typed target override.  It is user-owned without becoming a new
        # translation, so retain the established USER_RETAINED presentation.
        target = Authority.USER_RETAINED
    else:
        target = Authority.USER_EDIT
    return source, target


def _parent_presentation(
    parent: EffectiveParentSnapshot,
    *,
    target_authority: Authority,
) -> Presentation:
    if parent.excluded:
        return resolve_editor_status_presentation(excluded=True)
    issue_kinds = frozenset(issue.kind for issue in parent.issues)
    if issue_kinds & {
        ProjectionIssueKind.CONFLICT,
        ProjectionIssueKind.ORPHANED,
        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
    }:
        state = (
            PageState.CONFLICT
            if issue_kinds
            & {ProjectionIssueKind.CONFLICT, ProjectionIssueKind.ORPHANED}
            else PageState.ERROR
        )
        return resolve_state_presentation("page", state)
    if ProjectionIssueKind.MISSING_DEPENDENCY in issue_kinds:
        return resolve_state_presentation("page", PageState.MISSING)
    if (
        parent.target_freshness is TargetFreshness.STALE
        or issue_kinds
        & {
            ProjectionIssueKind.STALE_DEPENDENCY,
            ProjectionIssueKind.STALE_EDIT_BASE,
        }
    ):
        return resolve_state_presentation("page", PageState.STALE)
    return resolve_state_presentation("authority", target_authority)


def _selected_model_source_revision(
    effective: EffectiveParentSnapshot,
) -> OcrSourceRevisionArtifact | None:
    if not effective.source_revision_metadata:
        if effective.source_revision_id is not None:
            raise ValueError("source revision identity has no typed metadata")
        return None
    value = thaw_json(dict(effective.source_revision_metadata))
    artifact = OcrSourceRevisionArtifact.from_record(value)
    lineage = effective.lineage
    if (
        lineage is None
        or artifact.page_id == ""
        or artifact.parent_id != effective.parent_id
        or artifact.root_id != effective.root_id
        or artifact.parent_authored_edit_id != lineage.authored_edit_id
        or artifact.revision_id != effective.source_revision_id
        or artifact.hierarchy_revision_id == ""
        or artifact.hierarchy_fingerprint == ""
    ):
        raise ValueError("selected model OCR revision lineage is inconsistent")
    return artifact


def _selected_model_translation_revision(
    effective: EffectiveParentSnapshot,
) -> TranslationRevisionArtifact | None:
    if not effective.target_revision_metadata:
        if effective.target_revision_id is not None:
            raise ValueError("target revision identity has no typed metadata")
        return None
    value = thaw_json(dict(effective.target_revision_metadata))
    artifact = TranslationRevisionArtifact.from_record(value)
    try:
        source_artifact = OcrSourceRevisionArtifact.from_record(
            thaw_json(dict(effective.source_revision_metadata))
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "selected model translation source lineage is unavailable"
        ) from exc
    lineage = effective.lineage
    if (
        lineage is None
        or artifact.page_id == ""
        or artifact.parent_id != effective.parent_id
        or artifact.root_id != effective.root_id
        or artifact.parent_authored_edit_id != lineage.authored_edit_id
        or artifact.parent_role != effective.role
        or artifact.revision_id != effective.target_revision_id
        or artifact.source_text != effective.source_text
        or artifact.source_authority != effective.source_authority
        or artifact.source_revision_id != effective.source_revision_id
        or artifact.source_revision_id != source_artifact.revision_id
        or artifact.source_selection_edit_id != source_artifact.selection_edit_id
    ):
        raise ValueError(
            "selected model translation revision lineage is inconsistent"
        )
    return artifact


def _projected_user_parent(effective: EffectiveParentSnapshot) -> ProjectedParent:
    """Project user topology plus only its published typed owner revisions."""

    if effective.origin is not ParentOrigin.USER:
        raise ValueError("user-parent projection requires user origin")
    if effective.identity_namespace is not ParentIdentityNamespace.USER_PARENT_V1:
        raise ValueError("user parent has another identity namespace")
    if effective.root_identity_namespace is not RootIdentityNamespace.USER_ROOT_V1:
        raise ValueError("user parent has another root identity namespace")
    if effective.lineage is None or not effective.stage_requirements:
        raise ValueError("user parent is missing lineage or stage requirements")
    lineage = effective.lineage
    source_evidence_mapping = effective.source_evidence_mapping
    evidence_backed = source_evidence_mapping is not None
    absent_automatic_values = (
        effective.bundle_id,
        effective.automatic_fingerprint,
        thaw_json(effective.automatic_geometry),
        thaw_json(effective.render_allowed_area),
        thaw_json(effective.root_bbox),
    )
    if any(value is not None for value in absent_automatic_values):
        raise ValueError("pending user parent contains fabricated automatic evidence")
    effective_geometry = thaw_json(effective.geometry)
    if evidence_backed:
        if tuple(effective_geometry or ()) != tuple(
            thaw_json(effective.workflow_area_bbox) or ()
        ):
            raise ValueError(
                "pipeline-merged user parent must expose its exact merged bbox"
            )
    elif effective_geometry is not None:
        raise ValueError("pending user parent contains fabricated geometry evidence")
    if (
        effective.source_authority not in {"unavailable", "ocr_revision", "user"}
        or effective.target_authority
        not in {
            "unavailable",
            "translation_revision",
            "mapped_automatic",
            "user",
        }
        or effective.automatic_render_style
        or effective.render_style_overrides
        or effective.automatic_render_layout
        or effective.render_layout_overrides
    ):
        raise ValueError("pending user parent contains unavailable stage facts")
    selected_revision = _selected_model_source_revision(effective)
    selected_translation = _selected_model_translation_revision(effective)
    target_revision_base = target_text_revision_base_for_parent(effective)
    if source_evidence_mapping is not None:
        if (
            selected_revision is not None
            or effective.source_authority != "user"
            or effective.source_text != source_evidence_mapping.source_text
            or effective.source_revision_id is not None
        ):
            raise ValueError(
                "mapped user parent does not match its immutable OCR sources"
            )
        user_source_text = None
    elif effective.source_authority == "unavailable":
        if effective.source_text is not None or selected_revision is not None:
            raise ValueError("source-unavailable user parent exposes source evidence")
        user_source_text = None
    elif effective.source_authority == "ocr_revision":
        if (
            selected_revision is None
            or effective.source_text != selected_revision.source_text
        ):
            raise ValueError("selected model OCR revision does not match effective source")
        user_source_text = None
    else:
        if effective.source_text is None:
            raise ValueError("user source authority requires exact effective text")
        user_source_text = effective.source_text

    if source_evidence_mapping is not None:
        if selected_translation is not None or target_revision_base is not None:
            raise ValueError(
                "mapped user parent cannot expose a selected revision base"
            )
        mapped_target_text = source_evidence_mapping.target_text
        if effective.target_authority == "unavailable":
            if (
                mapped_target_text is not None
                or effective.target_text is not None
                or effective.target_freshness is not TargetFreshness.UNAVAILABLE
            ):
                raise ValueError(
                    "mapped target-unavailable state is inconsistent"
                )
            user_target_text = None
        elif effective.target_authority == "mapped_automatic":
            if (
                mapped_target_text is None
                or effective.target_text != mapped_target_text
                or effective.target_freshness is not TargetFreshness.CURRENT
            ):
                raise ValueError(
                    "mapped pipeline translation does not match the effective target"
                )
            user_target_text = None
        elif effective.target_authority == "user":
            if (
                mapped_target_text is None
                or effective.target_text is None
                or effective.target_freshness is not TargetFreshness.CURRENT
            ):
                raise ValueError(
                    "mapped target override requires exact effective user text"
                )
            user_target_text = effective.target_text
        else:
            raise ValueError("mapped user target authority is unsupported")
    elif effective.target_authority == "unavailable":
        if (
            effective.target_text is not None
            or effective.target_freshness
            not in {TargetFreshness.UNAVAILABLE, TargetFreshness.STALE}
            or selected_translation is not None
            or target_revision_base is not None
        ):
            raise ValueError("target-unavailable user parent exposes target evidence")
        user_target_text = None
    elif effective.target_authority == "translation_revision":
        if (
            selected_translation is None
            or target_revision_base is None
            or effective.target_text != selected_translation.target_text
            or effective.target_freshness is not TargetFreshness.CURRENT
        ):
            raise ValueError(
                "selected model translation does not match the effective target"
            )
        user_target_text = None
    else:
        if (
            effective.target_text is None
            or effective.target_freshness is not TargetFreshness.CURRENT
            or selected_translation is None
            or target_revision_base is None
        ):
            raise ValueError("user target authority requires exact effective text")
        user_target_text = effective.target_text

    source_requirements = tuple(
        value
        for value in effective.stage_requirements
        if value.stage is RevisionStage.SOURCE
    )
    translation_requirements = tuple(
        value
        for value in effective.stage_requirements
        if value.stage is RevisionStage.TRANSLATION
    )
    if len(source_requirements) != 1 or len(translation_requirements) != 1:
        raise ValueError("user parent requires exact SOURCE and TRANSLATION states")
    source_current = bool(
        source_requirements[0].state is RevisionStageState.CURRENT
        and source_requirements[0].required_action is RevisionRequiredAction.NONE
    )
    translation_current = bool(
        translation_requirements[0].state is RevisionStageState.CURRENT
        and translation_requirements[0].required_action
        is RevisionRequiredAction.NONE
    )
    translation_missing = bool(
        translation_requirements[0].state is RevisionStageState.MISSING
        and translation_requirements[0].required_action
        is RevisionRequiredAction.EXPLICIT_RUN
    )
    if source_evidence_mapping is not None:
        if not source_current:
            raise ValueError("mapped pipeline OCR SOURCE transition is invalid")
        if source_evidence_mapping.target_text is None:
            if (
                effective.target_authority != "unavailable"
                or not translation_missing
            ):
                raise ValueError(
                    "mapped pipeline target-missing transition is invalid"
                )
        elif (
            not translation_current
            or effective.target_authority not in {"mapped_automatic", "user"}
        ):
            raise ValueError(
                "mapped pipeline TRANSLATION transition is invalid"
            )
    elif selected_revision is None:
        if lineage.order_policy == "replace_sources":
            if (
                not source_current
                or effective.source_authority != "user"
                or effective.target_authority != "unavailable"
                or not translation_missing
            ):
                raise ValueError(
                    "pipeline-merged user parent stage transition is invalid"
                )
        else:
            if not (
                source_requirements[0].state is RevisionStageState.MISSING
                and source_requirements[0].required_action
                is RevisionRequiredAction.EXPLICIT_RUN
            ):
                raise ValueError("pending user parent SOURCE requirement is invalid")
            if effective.target_authority != "unavailable":
                raise ValueError("source-missing user parent cannot expose a target")
    elif not source_current:
        raise ValueError("selected OCR revision SOURCE transition is invalid")
    elif effective.target_authority == "unavailable" and not translation_missing:
        raise ValueError("selected OCR revision TRANSLATION transition is invalid")
    elif effective.target_authority != "unavailable" and not translation_current:
        raise ValueError("selected target revision TRANSLATION transition is invalid")

    if not source_current:
        presentation_label = "Source required"
    elif not translation_current:
        presentation_label = "Translation required"
    else:
        presentation_label = "Later revisions required"

    presentation = Presentation(
        label=presentation_label,
        tone="warning",
        icon="warning",
    )
    row = ParentRow(
        parent_id=effective.parent_id,
        reading_order=effective.reading_order,
        parent_role=effective.role,
        source_text=effective.source_text,
        target_text=effective.target_text,
        excluded=effective.excluded,
        source_authority=(
            Authority.USER_EDIT
            if effective.source_authority == "user"
            and source_evidence_mapping is None
            else None
        ),
        target_authority=(
            Authority.USER_EDIT
            if effective.target_authority == "user"
            else None
        ),
        presentation=presentation,
        origin=effective.origin,
        identity_namespace=effective.identity_namespace,
        root_identity_namespace=effective.root_identity_namespace,
        stage_requirements=effective.stage_requirements,
    )
    unavailable = (
        "Unavailable until the required explicit source, style, and render-eligibility "
        "revisions are published for this user parent."
    )
    return ProjectedParent(
        parent_row=row,
        effective=effective,
        automatic_source_text=None,
        selected_model_source_revision=selected_revision,
        selected_model_translation_revision=selected_translation,
        target_text_revision_base=target_revision_base,
        automatic_target_text=None,
        mapped_pipeline_source_text=(
            effective.source_evidence_mapping.source_text
            if effective.source_evidence_mapping is not None
            else None
        ),
        mapped_pipeline_target_text=(
            effective.source_evidence_mapping.target_text
            if effective.source_evidence_mapping is not None
            else None
        ),
        user_source_text=user_source_text,
        user_target_text=user_target_text,
        effective_render_style=(),
        effective_render_layout=(),
        automatic_writing_mode=None,
        user_writing_mode=None,
        effective_writing_mode=None,
        writing_mode_authority="unavailable",
        automatic_line_height=None,
        user_line_height=None,
        effective_line_height=None,
        line_height_authority="unavailable",
        automatic_rotation=None,
        user_rotation=None,
        effective_rotation=None,
        rotation_authority="unavailable",
        automatic_render_box=None,
        automatic_render_hard_bounds=None,
        user_render_box=None,
        effective_render_box=None,
        render_box_authority="unavailable",
        automatic_font_role=None,
        user_font_role=None,
        effective_font_role=None,
        font_role_authority="unavailable",
        automatic_font_weight_tier=None,
        user_font_weight_tier=None,
        effective_font_weight_tier=None,
        font_weight_tier_authority="unavailable",
        automatic_fill_color=None,
        user_fill_color=None,
        unresolved_user_fill_color=None,
        effective_fill_color=None,
        fill_color_authority="unavailable",
        automatic_outline_color=None,
        user_outline_color=None,
        unresolved_user_outline_color=None,
        effective_outline_color=None,
        outline_color_authority="unavailable",
        automatic_outline_width=None,
        user_outline_width=None,
        effective_outline_width=None,
        outline_width_authority="unavailable",
        automatic_preferred_size=None,
        user_preferred_size=None,
        effective_preferred_size=None,
        preferred_size_authority="unavailable",
        automatic_shadow_blur=None,
        user_shadow_blur=None,
        effective_shadow_blur=None,
        shadow_blur_authority="unavailable",
        automatic_shadow_color=None,
        user_shadow_color=None,
        effective_shadow_color=None,
        shadow_color_authority="unavailable",
        automatic_shadow_offset=None,
        user_shadow_offset=None,
        effective_shadow_offset=None,
        shadow_offset_authority="unavailable",
        automatic_shadow_enabled=None,
        user_shadow_enabled=None,
        effective_shadow_enabled=None,
        shadow_enabled_authority="unavailable",
        render_required=False,
        writing_mode_unavailable_reason=unavailable,
        line_height_unavailable_reason=unavailable,
        rotation_unavailable_reason=unavailable,
        render_box_unavailable_reason=unavailable,
        font_role_unavailable_reason=unavailable,
        font_weight_tier_unavailable_reason=unavailable,
        fill_color_unavailable_reason=unavailable,
        outline_color_unavailable_reason=unavailable,
        outline_width_unavailable_reason=unavailable,
        preferred_size_unavailable_reason=unavailable,
        shadow_blur_unavailable_reason=unavailable,
        shadow_color_unavailable_reason=unavailable,
        shadow_offset_unavailable_reason=unavailable,
        shadow_visibility_unavailable_reason=unavailable,
    )


def _merged_fields(
    automatic: tuple[tuple[str, Any], ...],
    overrides: tuple[tuple[str, Any], ...],
) -> tuple[tuple[str, Any], ...]:
    fields = dict(automatic)
    fields.update(dict(overrides))
    return tuple(sorted(fields.items()))


def _projected_parent(
    effective: EffectiveParentSnapshot,
    automatic: Mapping[str, Any] | None,
    *,
    unresolved_fill_color: tuple[bool, str | None],
    unresolved_outline_color: tuple[bool, str | None],
) -> ProjectedParent:
    if effective.origin is ParentOrigin.USER:
        if automatic is not None:
            raise ValueError("user parent must not resolve through an automatic bundle")
        if unresolved_fill_color != (False, None):
            raise ValueError("pending user parent cannot carry a fill-color edit")
        if unresolved_outline_color != (False, None):
            raise ValueError("pending user parent cannot carry an outline-color edit")
        return _projected_user_parent(effective)
    if effective.origin is not ParentOrigin.AUTOMATIC:
        raise ValueError("projected parent has an unsupported origin")
    if automatic is None:
        raise ValueError("automatic parent is missing its exact bundle")
    automatic_target_text = _automatic_target_text(automatic)
    automatic_writing_mode = automatic_render_writing_mode(automatic)
    automatic_line_height = automatic_render_line_height(automatic)
    automatic_rotation = automatic_render_rotation(automatic)
    automatic_box = automatic_render_box(automatic)
    automatic_hard_bounds = automatic_render_hard_bounds(automatic)
    automatic_font_role = automatic_render_font_role(automatic)
    automatic_font_weight_tier = automatic_render_font_weight_tier(automatic)
    automatic_fill_color = automatic_render_fill_color(automatic)
    automatic_outline_color = automatic_render_outline_color(automatic)
    automatic_outline_width = automatic_render_outline_width(automatic)
    automatic_preferred_size = automatic_render_preferred_size(automatic)
    automatic_shadow_blur = automatic_render_shadow_blur(automatic)
    automatic_shadow_color = automatic_render_shadow_color(automatic)
    automatic_shadow_offset = automatic_render_shadow_offset(automatic)
    automatic_shadow_enabled = automatic_render_shadow_enabled(automatic)
    render_style_overrides = dict(effective.render_style_overrides)
    render_layout_overrides = dict(effective.render_layout_overrides)
    raw_user_writing_mode = render_layout_overrides.get("writing_mode")
    user_writing_mode = (
        raw_user_writing_mode
        if isinstance(raw_user_writing_mode, str)
        and raw_user_writing_mode in CANONICAL_WRITING_MODES
        else None
    )
    has_user_writing_mode = "writing_mode" in render_layout_overrides
    effective_writing_mode = (
        user_writing_mode if has_user_writing_mode else automatic_writing_mode
    )
    writing_mode_authority = "user" if has_user_writing_mode else "automatic"
    has_user_line_height = "line_height" in render_layout_overrides
    user_line_height = (
        canonical_render_line_height(
            render_layout_overrides["line_height"],
            field_name="render_layout.line_height",
        )
        if has_user_line_height
        else None
    )
    effective_line_height = (
        user_line_height if has_user_line_height else automatic_line_height
    )
    line_height_authority = "user" if has_user_line_height else "automatic"
    has_user_rotation = "rotation" in render_layout_overrides
    user_rotation = (
        canonical_render_rotation(
            render_layout_overrides["rotation"],
            field_name="render_layout.rotation",
        )
        if has_user_rotation
        else None
    )
    effective_rotation = (
        user_rotation if has_user_rotation else automatic_rotation
    )
    rotation_authority = "user" if has_user_rotation else "automatic"
    has_user_render_box = "render_box" in render_layout_overrides
    user_render_box = (
        canonical_render_box(
            render_layout_overrides["render_box"],
            field_name="render_layout.render_box",
        )
        if has_user_render_box
        else None
    )
    effective_render_box = (
        user_render_box if has_user_render_box else automatic_box
    )
    render_box_authority = "user" if has_user_render_box else "automatic"
    has_user_font_role = "font_role" in render_style_overrides
    user_font_role = (
        canonical_render_font_role(
            render_style_overrides["font_role"],
            field_name="render_style.font_role",
        )
        if has_user_font_role
        else None
    )
    effective_font_role = (
        user_font_role if has_user_font_role else automatic_font_role
    )
    font_role_authority = "user" if has_user_font_role else "automatic"
    has_user_font_weight_tier = "font_weight_tier" in render_style_overrides
    user_font_weight_tier = (
        canonical_render_font_weight_tier(
            render_style_overrides["font_weight_tier"],
            field_name="render_style.font_weight_tier",
        )
        if has_user_font_weight_tier
        else None
    )
    effective_font_weight_tier = (
        user_font_weight_tier
        if has_user_font_weight_tier
        else automatic_font_weight_tier
    )
    font_weight_tier_authority = (
        "user" if has_user_font_weight_tier else "automatic"
    )
    has_unresolved_fill_color, unresolved_user_fill_color = (
        unresolved_fill_color
    )
    has_user_fill_color = "fill_color" in render_style_overrides
    raw_user_fill_color = render_style_overrides.get("fill_color")
    user_fill_color: str | None = None
    if has_user_fill_color:
        try:
            user_fill_color = canonical_render_fill_color(
                raw_user_fill_color,
                field_name="render_style.fill_color",
            )
        except (TypeError, ValueError):
            if isinstance(raw_user_fill_color, str):
                unresolved_user_fill_color = raw_user_fill_color
    if has_unresolved_fill_color:
        effective_fill_color = None
        fill_color_authority = "unresolved"
    elif not has_user_fill_color:
        effective_fill_color = automatic_fill_color
        fill_color_authority = "automatic"
    elif user_fill_color is not None:
        effective_fill_color = user_fill_color
        fill_color_authority = "user"
    else:  # pragma: no cover - invalid active edits are projected as issues
        effective_fill_color = None
        fill_color_authority = "unresolved"
    has_unresolved_outline_color, unresolved_user_outline_color = (
        unresolved_outline_color
    )
    has_user_outline_color = "outline_color" in render_style_overrides
    raw_user_outline_color = render_style_overrides.get("outline_color")
    user_outline_color: str | None = None
    if has_user_outline_color:
        try:
            user_outline_color = canonical_render_outline_color(
                raw_user_outline_color,
                field_name="render_style.outline_color",
            )
        except (TypeError, ValueError):
            if isinstance(raw_user_outline_color, str):
                unresolved_user_outline_color = raw_user_outline_color
    if has_unresolved_outline_color:
        effective_outline_color = None
        outline_color_authority = "unresolved"
    elif not has_user_outline_color:
        effective_outline_color = automatic_outline_color
        outline_color_authority = "automatic"
    elif user_outline_color is not None:
        effective_outline_color = user_outline_color
        outline_color_authority = "user"
    else:  # pragma: no cover - invalid active edits are projected as issues
        effective_outline_color = None
        outline_color_authority = "unresolved"
    has_user_outline_width = "outline_width" in render_style_overrides
    user_outline_width = (
        canonical_render_outline_width(
            render_style_overrides["outline_width"],
            field_name="render_style.outline_width",
        )
        if has_user_outline_width
        else None
    )
    effective_outline_width = (
        user_outline_width if has_user_outline_width else automatic_outline_width
    )
    outline_width_authority = (
        "user" if has_user_outline_width else "automatic"
    )
    has_user_preferred_size = "preferred_size" in render_style_overrides
    user_preferred_size = (
        canonical_render_preferred_size(
            render_style_overrides["preferred_size"],
            field_name="render_style.preferred_size",
        )
        if has_user_preferred_size
        else None
    )
    effective_preferred_size = (
        user_preferred_size if has_user_preferred_size else automatic_preferred_size
    )
    preferred_size_authority = (
        "user" if has_user_preferred_size else "automatic"
    )
    has_user_shadow_blur = "shadow_blur" in render_style_overrides
    user_shadow_blur = (
        canonical_render_shadow_blur(
            render_style_overrides["shadow_blur"],
            field_name="render_style.shadow_blur",
        )
        if has_user_shadow_blur
        else None
    )
    effective_shadow_blur = (
        user_shadow_blur if has_user_shadow_blur else automatic_shadow_blur
    )
    shadow_blur_authority = "user" if has_user_shadow_blur else "automatic"
    has_user_shadow_color = "shadow_color" in render_style_overrides
    user_shadow_color = (
        canonical_render_shadow_color(
            render_style_overrides["shadow_color"],
            field_name="render_style.shadow_color",
        )
        if has_user_shadow_color
        else None
    )
    effective_shadow_color = (
        user_shadow_color if has_user_shadow_color else automatic_shadow_color
    )
    shadow_color_authority = "user" if has_user_shadow_color else "automatic"
    has_user_shadow_offset = "shadow_offset" in render_style_overrides
    user_shadow_offset = (
        canonical_render_shadow_offset(
            render_style_overrides["shadow_offset"],
            field_name="render_style.shadow_offset",
        )
        if has_user_shadow_offset
        else None
    )
    effective_shadow_offset = (
        user_shadow_offset if has_user_shadow_offset else automatic_shadow_offset
    )
    shadow_offset_authority = (
        "user" if has_user_shadow_offset else "automatic"
    )
    has_user_shadow_enabled = "shadow_enabled" in render_style_overrides
    raw_user_shadow_enabled = render_style_overrides.get("shadow_enabled")
    user_shadow_enabled = (
        False if has_user_shadow_enabled and raw_user_shadow_enabled is False else None
    )
    effective_shadow_enabled = (
        user_shadow_enabled if has_user_shadow_enabled else automatic_shadow_enabled
    )
    shadow_enabled_authority = (
        "user" if has_user_shadow_enabled else "automatic"
    )
    render_required = automatic.get("render_required") is True
    if effective.excluded:
        writing_mode_unavailable_reason = (
            "Restore this excluded parent before editing its writing mode."
        )
    elif not render_required:
        writing_mode_unavailable_reason = (
            "Writing mode is unavailable because this parent does not require rendering."
        )
    elif automatic_writing_mode is None:
        writing_mode_unavailable_reason = (
            "The automatic writing mode is unavailable or noncanonical."
        )
    elif effective_writing_mode is None:
        writing_mode_unavailable_reason = (
            "The effective writing mode is unavailable or noncanonical."
        )
    else:
        writing_mode_unavailable_reason = ""
    if effective.excluded:
        line_height_unavailable_reason = (
            "Restore this excluded parent before editing its line height."
        )
    elif not render_required:
        line_height_unavailable_reason = (
            "Line height is unavailable because this parent does not require rendering."
        )
    elif automatic_line_height is None:
        line_height_unavailable_reason = (
            "The automatic line height is unavailable or outside 0.5 through 10.0."
        )
    elif effective_line_height is None:
        line_height_unavailable_reason = (
            "The effective line height is unavailable or outside 0.5 through 10.0."
        )
    else:
        line_height_unavailable_reason = ""
    if effective.excluded:
        rotation_unavailable_reason = (
            "Restore this excluded parent before editing its rotation."
        )
    elif not render_required:
        rotation_unavailable_reason = (
            "Rotation is unavailable because this parent does not require rendering."
        )
    elif automatic_rotation is None:
        rotation_unavailable_reason = (
            "The automatic rotation effect contract is invalid or unavailable."
        )
    elif effective_rotation is None:
        rotation_unavailable_reason = (
            "The effective rotation is unavailable or outside -45 through 45 degrees."
        )
    else:
        rotation_unavailable_reason = ""
    if effective.excluded:
        render_box_unavailable_reason = (
            "Restore this excluded parent before editing its render box."
        )
    elif not render_required:
        render_box_unavailable_reason = (
            "Render box is unavailable because this parent does not require rendering."
        )
    elif automatic_box is None or automatic_hard_bounds is None:
        render_box_unavailable_reason = (
            "The automatic target box or hard bounds are unavailable or invalid."
        )
    elif effective_render_box is None:
        render_box_unavailable_reason = (
            "The effective render box is unavailable or invalid."
        )
    elif not _box_contains(automatic_hard_bounds, effective_render_box):
        render_box_unavailable_reason = (
            "The effective render box exceeds automatic hard bounds."
        )
    else:
        render_box_unavailable_reason = ""
    if effective.excluded:
        font_role_unavailable_reason = (
            "Restore this excluded parent before editing its font role."
        )
    elif not render_required:
        font_role_unavailable_reason = (
            "Font role is unavailable because this parent does not require rendering."
        )
    elif automatic_font_role is None:
        font_role_unavailable_reason = (
            "The automatic registered font role is unavailable or invalid."
        )
    elif effective_font_role is None:
        font_role_unavailable_reason = (
            "The effective registered font role is unavailable or invalid."
        )
    else:
        font_role_unavailable_reason = ""
    if effective.excluded:
        font_weight_tier_unavailable_reason = (
            "Restore this excluded parent before editing its font weight."
        )
    elif not render_required:
        font_weight_tier_unavailable_reason = (
            "Font weight is unavailable because this parent does not require rendering."
        )
    elif automatic_font_weight_tier is None:
        font_weight_tier_unavailable_reason = (
            "The automatic registered font-weight tier is unavailable or invalid."
        )
    elif effective_font_weight_tier is None:
        font_weight_tier_unavailable_reason = (
            "The effective registered font-weight tier is unavailable or invalid."
        )
    else:
        font_weight_tier_unavailable_reason = ""
    if effective.excluded:
        fill_color_unavailable_reason = (
            "Restore this excluded parent before editing its fill color."
        )
    elif not render_required:
        fill_color_unavailable_reason = (
            "Fill color is unavailable because this parent does not require rendering."
        )
    elif automatic_fill_color is None:
        fill_color_unavailable_reason = (
            "The automatic fill color is missing or is not exact opaque #RRGGBB."
        )
    elif fill_color_authority == "unresolved":
        if unresolved_user_fill_color is not None:
            fill_color_unavailable_reason = (
                "The saved fill-color edit "
                f"{unresolved_user_fill_color!r} is unsupported. Only exact opaque "
                "#RRGGBB is editable; alpha is never coerced."
            )
        else:
            fill_color_unavailable_reason = (
                "The saved fill-color edit has an unsupported non-text value. "
                "Only exact opaque #RRGGBB is editable."
            )
    elif effective_fill_color is None:
        fill_color_unavailable_reason = (
            "The effective fill color is unavailable or is not exact opaque #RRGGBB."
        )
    else:
        fill_color_unavailable_reason = ""
    if effective.excluded:
        outline_color_unavailable_reason = (
            "Restore this excluded parent before editing its outline color."
        )
    elif not render_required:
        outline_color_unavailable_reason = (
            "Outline color is unavailable because this parent does not require rendering."
        )
    elif automatic_outline_color is None:
        outline_color_unavailable_reason = (
            "The automatic outline color is missing or is not exact opaque #RRGGBB."
        )
    elif outline_color_authority == "unresolved":
        if unresolved_user_outline_color is not None:
            outline_color_unavailable_reason = (
                "The saved outline-color edit "
                f"{unresolved_user_outline_color!r} is unsupported. Only exact opaque "
                "#RRGGBB is editable; alpha is never coerced."
            )
        else:
            outline_color_unavailable_reason = (
                "The saved outline-color edit has an unsupported non-text value. "
                "Only exact opaque #RRGGBB is editable."
            )
    elif effective_outline_color is None:
        outline_color_unavailable_reason = (
            "The effective outline color is unavailable or is not exact opaque #RRGGBB."
        )
    else:
        outline_color_unavailable_reason = ""
    if effective.excluded:
        outline_width_unavailable_reason = (
            "Restore this excluded parent before editing its outline width."
        )
    elif not render_required:
        outline_width_unavailable_reason = (
            "Outline width is unavailable because this parent does not require rendering."
        )
    elif automatic_outline_width is None:
        outline_width_unavailable_reason = (
            "The automatic outline width is missing or outside 0 through 128 pixels."
        )
    elif effective_outline_width is None:
        outline_width_unavailable_reason = (
            "The effective outline width is unavailable or outside 0 through 128 pixels."
        )
    else:
        outline_width_unavailable_reason = ""
    if effective.excluded:
        preferred_size_unavailable_reason = (
            "Restore this excluded parent before editing its preferred size."
        )
    elif not render_required:
        preferred_size_unavailable_reason = (
            "Preferred size is unavailable because this parent does not require rendering."
        )
    elif automatic_preferred_size is None:
        preferred_size_unavailable_reason = (
            "The automatic preferred size is missing or outside 0.1 through 2048 pixels."
        )
    elif effective_preferred_size is None:
        preferred_size_unavailable_reason = (
            "The effective preferred size is unavailable or outside 0.1 through 2048 pixels."
        )
    else:
        preferred_size_unavailable_reason = ""
    if effective.excluded:
        shadow_blur_unavailable_reason = (
            "Restore this excluded parent before editing its shadow blur."
        )
    elif not render_required:
        shadow_blur_unavailable_reason = (
            "Shadow blur is unavailable because this parent does not require rendering."
        )
    elif automatic_shadow_blur is None:
        shadow_blur_unavailable_reason = (
            "A strictly valid visible automatic shadow blur from 0 through 64 pixels is unavailable."
        )
    elif effective_shadow_blur is None:
        shadow_blur_unavailable_reason = (
            "The effective shadow blur is unavailable or outside 0 through 64 pixels."
        )
    else:
        shadow_blur_unavailable_reason = ""
    if effective.excluded:
        shadow_color_unavailable_reason = (
            "Restore this excluded parent before editing its shadow color."
        )
    elif not render_required:
        shadow_color_unavailable_reason = (
            "Shadow color is unavailable because this parent does not require rendering."
        )
    elif automatic_shadow_color is None:
        shadow_color_unavailable_reason = (
            "A strictly valid visible automatic RGB/RGBA shadow color is unavailable."
        )
    elif effective_shadow_color is None:
        shadow_color_unavailable_reason = (
            "The effective shadow color is unavailable or malformed."
        )
    else:
        shadow_color_unavailable_reason = ""
    if effective.excluded:
        shadow_offset_unavailable_reason = (
            "Restore this excluded parent before editing its shadow offset."
        )
    elif not render_required:
        shadow_offset_unavailable_reason = (
            "Shadow offset is unavailable because this parent does not require rendering."
        )
    elif automatic_shadow_offset is None:
        shadow_offset_unavailable_reason = (
            "A strictly valid visible automatic shadow offset from -256 through 256 pixels is unavailable."
        )
    elif effective_shadow_offset is None:
        shadow_offset_unavailable_reason = (
            "The effective shadow offset is unavailable or outside -256 through 256 pixels."
        )
    else:
        shadow_offset_unavailable_reason = ""
    if effective.excluded:
        shadow_visibility_unavailable_reason = (
            "Restore this excluded parent before editing shadow visibility."
        )
    elif not render_required:
        shadow_visibility_unavailable_reason = (
            "Shadow visibility is unavailable because this parent does not require rendering."
        )
    elif automatic_shadow_enabled is not True:
        shadow_visibility_unavailable_reason = (
            "A strictly valid visible automatic shadow is unavailable."
        )
    elif has_user_shadow_enabled and user_shadow_enabled is not False:
        shadow_visibility_unavailable_reason = (
            "The saved shadow-visibility edit is unsupported; only Hidden is editable."
        )
    elif effective_shadow_enabled not in {True, False}:
        shadow_visibility_unavailable_reason = (
            "The effective shadow visibility is unavailable or invalid."
        )
    else:
        shadow_visibility_unavailable_reason = ""
    source_authority, target_authority = _authority_values(
        effective,
        automatic_target_text=automatic_target_text,
    )
    presentation = _parent_presentation(
        effective,
        target_authority=target_authority,
    )
    row = ParentRow(
        parent_id=effective.parent_id,
        reading_order=effective.reading_order,
        parent_role=effective.role or "unclassified",
        source_text=effective.source_text,
        target_text=effective.target_text,
        excluded=effective.excluded,
        source_authority=source_authority,
        target_authority=target_authority,
        presentation=presentation,
        origin=effective.origin,
        identity_namespace=effective.identity_namespace,
        root_identity_namespace=effective.root_identity_namespace,
        stage_requirements=effective.stage_requirements,
    )
    return ProjectedParent(
        parent_row=row,
        effective=effective,
        automatic_source_text=_automatic_source_text(automatic),
        selected_model_source_revision=None,
        selected_model_translation_revision=None,
        target_text_revision_base=None,
        automatic_target_text=automatic_target_text,
        mapped_pipeline_source_text=None,
        mapped_pipeline_target_text=None,
        user_source_text=(
            effective.source_text if effective.source_authority != "automatic" else None
        ),
        user_target_text=(
            effective.target_text if effective.target_authority != "automatic" else None
        ),
        effective_render_style=_merged_fields(
            effective.automatic_render_style,
            effective.render_style_overrides,
        ),
        effective_render_layout=_merged_fields(
            effective.automatic_render_layout,
            effective.render_layout_overrides,
        ),
        automatic_writing_mode=automatic_writing_mode,
        user_writing_mode=user_writing_mode,
        effective_writing_mode=effective_writing_mode,
        writing_mode_authority=writing_mode_authority,
        automatic_line_height=automatic_line_height,
        user_line_height=user_line_height,
        effective_line_height=effective_line_height,
        line_height_authority=line_height_authority,
        automatic_rotation=automatic_rotation,
        user_rotation=user_rotation,
        effective_rotation=effective_rotation,
        rotation_authority=rotation_authority,
        automatic_render_box=automatic_box,
        automatic_render_hard_bounds=automatic_hard_bounds,
        user_render_box=user_render_box,
        effective_render_box=effective_render_box,
        render_box_authority=render_box_authority,
        automatic_font_role=automatic_font_role,
        user_font_role=user_font_role,
        effective_font_role=effective_font_role,
        font_role_authority=font_role_authority,
        automatic_font_weight_tier=automatic_font_weight_tier,
        user_font_weight_tier=user_font_weight_tier,
        effective_font_weight_tier=effective_font_weight_tier,
        font_weight_tier_authority=font_weight_tier_authority,
        automatic_fill_color=automatic_fill_color,
        user_fill_color=user_fill_color,
        unresolved_user_fill_color=unresolved_user_fill_color,
        effective_fill_color=effective_fill_color,
        fill_color_authority=fill_color_authority,
        automatic_outline_color=automatic_outline_color,
        user_outline_color=user_outline_color,
        unresolved_user_outline_color=unresolved_user_outline_color,
        effective_outline_color=effective_outline_color,
        outline_color_authority=outline_color_authority,
        automatic_outline_width=automatic_outline_width,
        user_outline_width=user_outline_width,
        effective_outline_width=effective_outline_width,
        outline_width_authority=outline_width_authority,
        automatic_preferred_size=automatic_preferred_size,
        user_preferred_size=user_preferred_size,
        effective_preferred_size=effective_preferred_size,
        preferred_size_authority=preferred_size_authority,
        automatic_shadow_blur=automatic_shadow_blur,
        user_shadow_blur=user_shadow_blur,
        effective_shadow_blur=effective_shadow_blur,
        shadow_blur_authority=shadow_blur_authority,
        automatic_shadow_color=automatic_shadow_color,
        user_shadow_color=user_shadow_color,
        effective_shadow_color=effective_shadow_color,
        shadow_color_authority=shadow_color_authority,
        automatic_shadow_offset=automatic_shadow_offset,
        user_shadow_offset=user_shadow_offset,
        effective_shadow_offset=effective_shadow_offset,
        shadow_offset_authority=shadow_offset_authority,
        automatic_shadow_enabled=automatic_shadow_enabled,
        user_shadow_enabled=user_shadow_enabled,
        effective_shadow_enabled=effective_shadow_enabled,
        shadow_enabled_authority=shadow_enabled_authority,
        render_required=render_required,
        writing_mode_unavailable_reason=writing_mode_unavailable_reason,
        line_height_unavailable_reason=line_height_unavailable_reason,
        rotation_unavailable_reason=rotation_unavailable_reason,
        render_box_unavailable_reason=render_box_unavailable_reason,
        font_role_unavailable_reason=font_role_unavailable_reason,
        font_weight_tier_unavailable_reason=font_weight_tier_unavailable_reason,
        fill_color_unavailable_reason=fill_color_unavailable_reason,
        outline_color_unavailable_reason=outline_color_unavailable_reason,
        outline_width_unavailable_reason=outline_width_unavailable_reason,
        preferred_size_unavailable_reason=preferred_size_unavailable_reason,
        shadow_blur_unavailable_reason=shadow_blur_unavailable_reason,
        shadow_color_unavailable_reason=shadow_color_unavailable_reason,
        shadow_offset_unavailable_reason=shadow_offset_unavailable_reason,
        shadow_visibility_unavailable_reason=shadow_visibility_unavailable_reason,
    )


def _unresolved_fill_color_for_parent(
    effective: EffectiveParentSnapshot,
    ledger: ProjectEditLedger,
) -> tuple[bool, str | None]:
    """Return one active rejected fill slot without coercing its raw value."""

    edit_ids = {
        edit_id
        for issue in effective.issues
        if issue.kind is ProjectionIssueKind.INVALID_EFFECTIVE_VALUE
        and issue.domain == EditDomain.RENDER_STYLE.value
        and issue.target_kind == EditTargetKind.PARENT.value
        and issue.target_id == effective.parent_id
        and issue.reason == "render_style.fill_color_requires_opaque_rgb"
        for edit_id in issue.edit_ids
    }
    candidates: list[object] = []
    for edit_id in sorted(edit_ids):
        record = ledger.get(edit_id)
        if record is None:
            continue
        payload = thaw_json(record.payload)
        fields = payload.get("fields") if isinstance(payload, Mapping) else None
        if (
            record.domain is EditDomain.RENDER_STYLE
            and record.operation == "set_fields"
            and record.target.kind is EditTargetKind.PARENT
            and record.target.parent_id == effective.parent_id
            and isinstance(fields, Mapping)
            and tuple(fields) == ("fill_color",)
        ):
            candidates.append(fields["fill_color"])
    if not candidates:
        return False, None
    if len(candidates) != 1:
        return True, None
    value = candidates[0]
    return True, value if isinstance(value, str) else None


def _unresolved_outline_color_for_parent(
    effective: EffectiveParentSnapshot,
    ledger: ProjectEditLedger,
) -> tuple[bool, str | None]:
    """Return one active rejected outline slot without coercing its raw value."""

    edit_ids = {
        edit_id
        for issue in effective.issues
        if issue.kind is ProjectionIssueKind.INVALID_EFFECTIVE_VALUE
        and issue.domain == EditDomain.RENDER_STYLE.value
        and issue.target_kind == EditTargetKind.PARENT.value
        and issue.target_id == effective.parent_id
        and issue.reason == "render_style.outline_color_requires_opaque_rgb"
        for edit_id in issue.edit_ids
    }
    candidates: list[object] = []
    for edit_id in sorted(edit_ids):
        record = ledger.get(edit_id)
        if record is None:
            continue
        payload = thaw_json(record.payload)
        fields = payload.get("fields") if isinstance(payload, Mapping) else None
        if (
            record.domain is EditDomain.RENDER_STYLE
            and record.operation == "set_fields"
            and record.target.kind is EditTargetKind.PARENT
            and record.target.parent_id == effective.parent_id
            and isinstance(fields, Mapping)
            and tuple(fields) == ("outline_color",)
        ):
            candidates.append(fields["outline_color"])
    if not candidates:
        return False, None
    if len(candidates) != 1:
        return True, None
    value = candidates[0]
    return True, value if isinstance(value, str) else None


def _rect_points(value: object) -> tuple[float, float, float, float] | None:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return None
    if len(value) != 4:
        return None
    points: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        points.append(float(item))
    if points[2] <= 0.0 or points[3] <= 0.0:
        return None
    return points[0], points[1], points[2], points[3]


def _line_points(value: object) -> tuple[float, float, float, float] | None:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return None
    if len(value) != 4:
        return None
    points: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        points.append(float(item))
    return points[0], points[1], points[2], points[3]


def _xyxy_points(value: object) -> tuple[float, float, float, float] | None:
    points = _line_points(value)
    if points is None:
        return None
    x1, y1, x2, y2 = points
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


_PROTECTED_AUTHORIZATIONS = frozenset(
    {
        "protect_sfx_decorative",
        "protect_art_or_non_text",
        "review_unknown_not_cleanup",
        "outside_cleanup_scope",
    }
)


def _region_has_protected_authority(region: Mapping[str, Any]) -> bool:
    if not (
        region.get("must_not_mutate") is True
        or region.get("text_area_must_not_mutate") is True
    ):
        return False
    if region.get("explicit_protected_authority") is True:
        return True
    states = {
        str(region.get(field) or "").strip()
        for field in (
            "authorization_state",
            "cleanup_authorization",
            "semantic_authorization_state",
            "text_area_cleanup_authorization",
            "text_area_semantic_authorization_state",
        )
    }
    for field in ("protected_authority_states", "semantic_unit_states"):
        values = region.get(field)
        if isinstance(values, (list, tuple)):
            states.update(str(value).strip() for value in values)
    return bool(states & _PROTECTED_AUTHORIZATIONS)


def _region_bounds(region: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    if "xyxy" in region:
        return _xyxy_points(region.get("xyxy"))
    if "bounds" in region:
        return _xyxy_points(region.get("bounds"))
    if "bbox" in region:
        return _rect_points(region.get("bbox"))
    return None


def _protected_region_overlays(
    page: Mapping[str, Any],
    automatic_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[OverlayShapeData, ...]:
    region_values: list[Mapping[str, Any]] = []
    seen_region_ids: set[str] = set()
    for field in ("regions", "source_regions"):
        values = page.get(field) or ()
        if not isinstance(values, (list, tuple)):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            region_id = _optional_text(value.get("region_id") or value.get("id"))
            if region_id is None or region_id in seen_region_ids:
                continue
            seen_region_ids.add(region_id)
            region_values.append(value)
    parent_ids = frozenset(automatic_by_id)
    parent_by_bundle = {
        str(value.get("bundle_id") or "").strip(): parent_id
        for parent_id, value in automatic_by_id.items()
        if str(value.get("bundle_id") or "").strip()
    }
    result: list[OverlayShapeData] = []
    for region in region_values:
        if not _region_has_protected_authority(region):
            continue
        region_id = _optional_text(region.get("region_id") or region.get("id"))
        bounds = _region_bounds(region)
        if region_id is None or bounds is None:
            continue
        parent_id = ""
        explicit_parent = _optional_text(
            region.get("parent_id")
            or region.get("text_block_parent_id")
            or region.get("parent_execution_parent_id")
        )
        if explicit_parent in parent_ids:
            parent_id = explicit_parent or ""
        else:
            bundle_id = _optional_text(region.get("parent_execution_bundle_id"))
            if bundle_id in parent_by_bundle:
                parent_id = parent_by_bundle[bundle_id]
        result.append(
            OverlayShapeData(
                overlay_id="protectedRegions",
                shape_id=f"protected-region:{region_id}",
                kind="rect",
                points=bounds,
                label="Automatic protected region",
                parent_id=parent_id,
            )
        )
    return tuple(result)


def _proof_overlays(
    snapshot: EffectivePageSnapshot,
    automatic_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[OverlayShapeData, ...]:
    cleaned = thaw_json(snapshot.cleaned_page_base)
    lineage = cleaned_base_automatic_lineage(cleaned)
    if lineage is None:
        return ()
    committed = {
        str(value).strip()
        for value in lineage.get("cleanup_committed_region_ids") or ()
        if str(value).strip()
    }
    blocked = {
        str(value).strip()
        for value in lineage.get("cleanup_blocked_region_ids") or ()
        if str(value).strip()
    }
    result: list[OverlayShapeData] = []
    for parent in snapshot.parents:
        automatic = automatic_by_id.get(parent.parent_id)
        if automatic is None:
            if parent.origin is ParentOrigin.USER:
                continue
            raise ValueError("automatic cleanup evidence is missing its parent bundle")
        identities = {
            value
            for value in (
                parent.parent_id,
                str(automatic.get("bundle_id") or "").strip(),
            )
            if value
        }
        states: list[str] = []
        if identities & committed:
            states.append("committed record")
        if identities & blocked:
            states.append("blocked record")
        if not states:
            continue
        bounds = _rect_points(automatic.get("cleanup_target_bbox"))
        if bounds is None:
            bounds = _rect_points(automatic.get("parent_bbox"))
        if bounds is None:
            continue
        result.append(
            OverlayShapeData(
                overlay_id="proof",
                shape_id=f"{parent.parent_id}:cleanup-evidence",
                kind="rect",
                points=bounds,
                label=f"Automatic cleanup evidence - {' + '.join(states)}",
                parent_id=parent.parent_id,
            )
        )
    return tuple(result)


def _manual_raster_overlays(
    snapshot: EffectivePageSnapshot,
    *,
    project_path: str,
) -> tuple[RasterOverlayData, ...]:
    cleaned = thaw_json(snapshot.cleaned_page_base)
    if (
        not isinstance(cleaned, Mapping)
        or cleaned.get("valid") is not True
        or str(cleaned.get("provenance") or "") != _CLEANUP_PROVENANCE
    ):
        return ()
    receipt = cleaned.get("manual_cleanup_receipt")
    canvas_size = _canvas_size(cleaned.get("canvas_size"))
    if (
        not isinstance(receipt, Mapping)
        or str(receipt.get("status") or "") != "committed"
        or str(receipt.get("page_id") or "") != snapshot.page_id
        or canvas_size is None
        or _canvas_size(receipt.get("canvas_size")) != canvas_size
    ):
        return ()
    result: list[RasterOverlayData] = []
    for overlay_id, prefix, label in (
        ("cleanupMask", "effective", "Manual cleanup effective mask"),
        ("protectedRegions", "protect", "Manual cleanup protected mask"),
    ):
        hash_field = f"{prefix}_mask_sha256"
        expected = str(cleaned.get(hash_field) or "").strip().lower()
        if str(receipt.get(hash_field) or "").strip().lower() != expected:
            continue
        validated = _validated_asset_path(
            cleaned.get(f"{prefix}_mask_asset"),
            expected,
            project_path=project_path,
        )
        if validated is None:
            continue
        asset_path, asset_sha256 = validated
        result.append(
            RasterOverlayData(
                overlay_id=overlay_id,
                asset_path=asset_path,
                asset_sha256=asset_sha256,
                canvas_size=canvas_size,
                label=label,
            )
        )
    return tuple(result)


def _overlay_availability(
    overlays: tuple[OverlayShapeData, ...],
    raster_overlays: tuple[RasterOverlayData, ...],
) -> tuple[OverlayAvailabilityData, ...]:
    available_ids = {
        *(item.overlay_id for item in overlays),
        *(item.overlay_id for item in raster_overlays),
    }
    missing = {
        "parentBounds": "No exact parent bounds are available for this page.",
        "renderBox": "No exact effective render boxes are available for this page.",
        "sourceFootprint": "No exact source footprints are available for this page.",
        "baseline": "No explicit baseline geometry is available for this page.",
        "cleanupMask": "No validated manual cleanup effective mask is available.",
        "protectedRegions": (
            "No validated manual protect mask or explicitly protected region bounds are available."
        ),
        "proof": "No cleanup evidence is bound to an exact parent or bundle with bounds.",
    }
    return tuple(
        OverlayAvailabilityData(
            overlay_id=overlay_id,
            available=overlay_id in available_ids,
            tooltip=(
                "Available for the current page."
                if overlay_id in available_ids
                else missing[overlay_id]
            ),
        )
        for overlay_id in OVERLAY_IDS
    )


def _overlays(
    snapshot: EffectivePageSnapshot,
    automatic_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[OverlayShapeData, ...]:
    values: list[OverlayShapeData] = []
    for parent in snapshot.parents:
        label = f"{parent.reading_order + 1}. {parent.parent_id}"
        parent_bounds = _rect_points(parent.geometry)
        if parent_bounds is not None:
            values.append(
                OverlayShapeData(
                    overlay_id="parentBounds",
                    shape_id=f"{parent.parent_id}:parent-bounds",
                    kind="rect",
                    points=parent_bounds,
                    label=label,
                    parent_id=parent.parent_id,
                )
            )
        render_box = _rect_points(parent.render_allowed_area)
        if render_box is not None:
            values.append(
                OverlayShapeData(
                    overlay_id="renderBox",
                    shape_id=f"{parent.parent_id}:render-box",
                    kind="rect",
                    points=render_box,
                    label=label,
                    parent_id=parent.parent_id,
                )
            )
        automatic = automatic_by_id.get(parent.parent_id)
        if automatic is None:
            if parent.origin is ParentOrigin.USER:
                continue
            raise ValueError("automatic overlay evidence is missing its parent bundle")
        source_footprint = _rect_points(
            automatic.get("cleanup_target_bbox")
            or automatic.get("source_text_bbox")
            or automatic.get("text_bbox")
        )
        if source_footprint is not None:
            values.append(
                OverlayShapeData(
                    overlay_id="sourceFootprint",
                    shape_id=f"{parent.parent_id}:source-footprint",
                    kind="rect",
                    points=source_footprint,
                    label=label,
                    parent_id=parent.parent_id,
                )
            )
        layout = dict(parent.automatic_render_layout)
        layout.update(dict(parent.render_layout_overrides))
        baseline = _line_points(layout.get("baseline"))
        if baseline is not None:
            values.append(
                OverlayShapeData(
                    overlay_id="baseline",
                    shape_id=f"{parent.parent_id}:baseline",
                    kind="line",
                    points=baseline,
                    label=label,
                    parent_id=parent.parent_id,
                )
            )
    return tuple(values)


def _original_reference(
    page: Mapping[str, Any],
    snapshot: EffectivePageSnapshot,
    *,
    project_path: str,
) -> tuple[str | None, ArtifactState]:
    cleaned = thaw_json(snapshot.cleaned_page_base)
    nested_source = None
    if isinstance(cleaned, Mapping):
        nested = cleaned.get("cleaned_page_base")
        if isinstance(nested, Mapping):
            nested_source = nested.get("source_image_path")
        nested_source = cleaned.get("source_image_path") or nested_source
    raw = page.get("image_path") or page.get("source_image_path") or nested_source
    path = _available_asset_path(raw, project_path=project_path)
    return path, ArtifactState.VALID if path is not None else ArtifactState.MISSING


def _cleaned_reference(
    snapshot: EffectivePageSnapshot,
    *,
    project_path: str,
) -> tuple[str | None, ArtifactState]:
    cleaned = thaw_json(snapshot.cleaned_page_base)
    if not snapshot.cleaned_base_revision_id or not isinstance(cleaned, Mapping):
        return None, ArtifactState.MISSING
    path, state = _record_state(cleaned, project_path=project_path)
    cleanup_issue_kinds = frozenset(
        issue.kind for issue in snapshot.issues if issue.domain == "cleanup"
    )
    if state is ArtifactState.VALID:
        if ProjectionIssueKind.MISSING_DEPENDENCY in cleanup_issue_kinds:
            state = ArtifactState.MISSING
        elif ProjectionIssueKind.INVALID_EFFECTIVE_VALUE in cleanup_issue_kinds:
            state = ArtifactState.INVALID
        elif cleanup_issue_kinds & {
            ProjectionIssueKind.STALE_DEPENDENCY,
            ProjectionIssueKind.STALE_EDIT_BASE,
        }:
            state = ArtifactState.STALE
    return path if state is not ArtifactState.MISSING else None, state


def _selected_final_record(
    records: tuple[Mapping[str, Any], ...],
    snapshot: EffectivePageSnapshot,
    page: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    exact = tuple(
        record
        for record in records
        if str(record.get("effective_page_fingerprint") or "")
        == snapshot.effective_fingerprint
    )
    if exact:
        return exact[-1]
    current = tuple(record for record in records if record.get("current") is True)
    if current:
        return current[-1]
    if records:
        return records[-1]
    output_path = _optional_text(page.get("output_path"))
    if output_path is None:
        return None
    return {
        "revision_id": f"page-output:{snapshot.page_id}",
        "page_id": snapshot.page_id,
        "provenance": "automatic",
        "current": True,
        "asset": output_path,
    }


def _final_reference(
    selected: Mapping[str, Any] | None,
    snapshot: EffectivePageSnapshot,
    *,
    project_path: str,
) -> tuple[str | None, ArtifactState]:
    if selected is None:
        return None, ArtifactState.MISSING
    fingerprint = str(selected.get("effective_page_fingerprint") or "")
    stale = bool(
        (fingerprint and fingerprint != snapshot.effective_fingerprint)
        or (snapshot.applied_edit_ids and not fingerprint)
    )
    path, state = _record_state(
        selected,
        project_path=project_path,
        stale=stale,
    )
    return path if state is not ArtifactState.MISSING else None, state


def _history(
    project: Mapping[str, Any],
    snapshot: EffectivePageSnapshot,
    *,
    page_id: str,
    project_path: str,
    known_page_ids: frozenset[str],
) -> tuple[ArtifactRevisionReference, ...]:
    history: list[ArtifactRevisionReference] = []
    selected_cleaned_id = snapshot.cleaned_base_revision_id
    for catalog_id in ("cleaned_page_bases", "rendered_pages", "parent_layers"):
        records = _catalog_page_records(
            project,
            catalog_id=catalog_id,
            page_id=page_id,
            known_page_ids=known_page_ids,
        )
        for record in records:
            revision_id = _required_text(record.get("revision_id"), "revision_id")
            if catalog_id == "rendered_pages":
                fingerprint = str(record.get("effective_page_fingerprint") or "")
                current = bool(
                    fingerprint == snapshot.effective_fingerprint
                    if fingerprint
                    else record.get("current") is True and not snapshot.applied_edit_ids
                )
                stale = bool(
                    (fingerprint and fingerprint != snapshot.effective_fingerprint)
                    or (snapshot.applied_edit_ids and not fingerprint)
                )
            elif catalog_id == "cleaned_page_bases":
                current = revision_id == selected_cleaned_id
                stale = False
            else:
                current = bool(record.get("current"))
                stale = False
            asset_path, state = _record_state(
                record,
                project_path=project_path,
                stale=stale,
            )
            history.append(
                ArtifactRevisionReference(
                    kind=_CATALOG_KINDS[catalog_id],
                    revision_id=revision_id,
                    asset_path=asset_path,
                    state=state,
                    provenance=str(record.get("provenance") or "unknown"),
                    current=current,
                )
            )
    return tuple(history)


def _edit_history(
    ledger: ProjectEditLedger,
    snapshot: EffectivePageSnapshot,
    *,
    page_id: str,
) -> tuple[EditHistoryReference, ...]:
    """Project append-only edit history into safe, presentation-only facts."""

    active_ids = frozenset(ledger.state().active_edit_ids)
    effective_ids = frozenset(snapshot.applied_edit_ids)
    history: list[EditHistoryReference] = []
    for record in ledger.records_for_page(page_id):
        target_record = (
            ledger.get(record.target.edit_id)
            if record.is_control
            else record
        )
        if (
            target_record is not None
            and target_record.domain is EditDomain.GLOSSARY
        ):
            continue
        is_control = bool(record.is_control)
        target_kind = record.target.kind.value
        target_id = {
            "project": record.project_id,
            "page": record.page_id,
            "parent": record.target.parent_id,
            "artifact": record.target.artifact_id,
            "edit": record.target.edit_id,
        }[target_kind]
        issue_codes = tuple(
            sorted(
                {
                    f"{issue.kind.value}:{issue.reason}"
                    for issue in snapshot.issues
                    if record.edit_id in issue.edit_ids
                }
            )
        )
        field_name = ""
        if record.domain.value == "render_layout":
            payload = thaw_json(record.payload)
            fields = payload.get("fields") if isinstance(payload, Mapping) else None
            if record.operation == "set_fields" and isinstance(fields, Mapping):
                if tuple(fields) == ("writing_mode",):
                    field_name = "writing_mode"
                elif tuple(fields) == ("line_height",):
                    field_name = "line_height"
                elif tuple(fields) == ("rotation",):
                    field_name = "rotation"
                elif tuple(fields) == ("render_box",):
                    field_name = "render_box"
            elif record.operation == "restore_automatic" and isinstance(
                fields,
                (list, tuple),
            ):
                if tuple(fields) == ("writing_mode",):
                    field_name = "writing_mode"
                elif tuple(fields) == ("line_height",):
                    field_name = "line_height"
                elif tuple(fields) == ("rotation",):
                    field_name = "rotation"
                elif tuple(fields) == ("render_box",):
                    field_name = "render_box"
        elif record.domain.value == "render_style":
            payload = thaw_json(record.payload)
            fields = payload.get("fields") if isinstance(payload, Mapping) else None
            if record.operation == "set_fields" and isinstance(fields, Mapping):
                if tuple(fields) == ("fill_color",):
                    field_name = "fill_color"
                elif tuple(fields) == ("font_role",):
                    field_name = "font_role"
                elif tuple(fields) == ("font_weight_tier",):
                    field_name = "font_weight_tier"
                elif tuple(fields) == ("outline_color",):
                    field_name = "outline_color"
                elif tuple(fields) == ("outline_width",):
                    field_name = "outline_width"
                elif tuple(fields) == ("preferred_size",):
                    field_name = "preferred_size"
                elif tuple(fields) == ("shadow_color",):
                    field_name = "shadow_color"
                elif tuple(fields) == ("shadow_enabled",):
                    field_name = "shadow_enabled"
                elif tuple(fields) == ("shadow_blur",):
                    field_name = "shadow_blur"
                elif tuple(fields) == ("shadow_offset",):
                    field_name = "shadow_offset"
            elif record.operation == "restore_automatic" and isinstance(
                fields,
                (list, tuple),
            ):
                if tuple(fields) == ("fill_color",):
                    field_name = "fill_color"
                elif tuple(fields) == ("font_role",):
                    field_name = "font_role"
                elif tuple(fields) == ("font_weight_tier",):
                    field_name = "font_weight_tier"
                elif tuple(fields) == ("outline_color",):
                    field_name = "outline_color"
                elif tuple(fields) == ("outline_width",):
                    field_name = "outline_width"
                elif tuple(fields) == ("preferred_size",):
                    field_name = "preferred_size"
                elif tuple(fields) == ("shadow_color",):
                    field_name = "shadow_color"
                elif tuple(fields) == ("shadow_enabled",):
                    field_name = "shadow_enabled"
                elif tuple(fields) == ("shadow_blur",):
                    field_name = "shadow_blur"
                elif tuple(fields) == ("shadow_offset",):
                    field_name = "shadow_offset"
        history.append(
            EditHistoryReference(
                record_id=record.edit_id,
                domain=record.domain.value,
                operation=record.operation,
                target_kind=target_kind,
                target_id=target_id,
                field_name=field_name,
                created_at=record.created_at,
                active=bool(not is_control and record.edit_id in active_ids),
                effective=bool(
                    not is_control and record.edit_id in effective_ids
                ),
                is_control=is_control,
                issue_codes=issue_codes,
            )
        )
    return tuple(history)


def _project_glossary_history(
    ledger: ProjectEditLedger,
    snapshots: tuple[EffectivePageSnapshot, ...],
) -> tuple[EditHistoryReference, ...]:
    """Project project-scoped glossary history without exposing payload data."""

    if not snapshots:
        return ()
    active_ids = frozenset(ledger.state().active_edit_ids)
    effective_ids = frozenset(snapshots[0].applied_edit_ids)
    issue_by_edit: dict[str, set[str]] = {}
    for snapshot in snapshots:
        for issue in snapshot.issues:
            for edit_id in issue.edit_ids:
                issue_by_edit.setdefault(edit_id, set()).add(
                    f"{issue.kind.value}:{issue.reason}"
                )
    glossary_edit_ids = {
        record.edit_id
        for record in ledger.edits
        if not record.is_control and record.domain is EditDomain.GLOSSARY
    }
    history: list[EditHistoryReference] = []
    for record in ledger.edits:
        target_record = (
            ledger.get(record.target.edit_id)
            if record.is_control
            else record
        )
        if (
            target_record is None
            or target_record.edit_id not in glossary_edit_ids
        ):
            continue
        is_control = bool(record.is_control)
        if target_record.operation == "set_entry":
            entry = target_record.payload.get("entry")
            field_name = (
                str(entry.get("entry_id") or "")
                if isinstance(entry, Mapping)
                else ""
            )
        else:
            field_name = str(target_record.payload.get("entry_id") or "")
        history.append(
            EditHistoryReference(
                record_id=record.edit_id,
                domain=record.domain.value,
                operation=record.operation,
                target_kind=record.target.kind.value,
                target_id=(
                    record.target.edit_id
                    if is_control
                    else record.project_id
                ),
                field_name=field_name,
                created_at=record.created_at,
                active=bool(not is_control and record.edit_id in active_ids),
                effective=bool(not is_control and record.edit_id in effective_ids),
                is_control=is_control,
                issue_codes=tuple(sorted(issue_by_edit.get(record.edit_id, ()))),
            )
        )
    return tuple(history)


def _page_filename(page: Mapping[str, Any], original_path: str | None) -> str:
    explicit = _optional_text(page.get("file_name") or page.get("image_name"))
    if explicit is not None:
        return explicit
    if original_path is not None:
        return Path(original_path).name
    raw = _optional_text(page.get("image_path") or page.get("source_image_path"))
    if raw is not None:
        return Path(raw).name
    return _required_text(page.get("page_id"), "page_id")


def _page_progress(
    original: ArtifactState,
    cleaned: ArtifactState,
    final: ArtifactState,
) -> int:
    if final is ArtifactState.VALID:
        return 100
    if cleaned is ArtifactState.VALID:
        return 80
    if original is ArtifactState.VALID:
        return 20
    return 0


def _original_page_binding_for_ocr(
    project: Mapping[str, Any],
    *,
    page_id: str,
    project_path: str,
    parents: tuple[ProjectedParent, ...],
) -> tuple[OriginalPageAssetBinding | None, str]:
    selected = tuple(
        parent.selected_model_source_revision.original_page
        for parent in parents
        if parent.selected_model_source_revision is not None
    )
    distinct = tuple(dict.fromkeys(selected))
    if len(distinct) > 1:
        raise ValueError("selected OCR revisions disagree on original-page identity")
    source_runnable = any(
        parent.origin is ParentOrigin.USER
        and any(
            requirement.stage is RevisionStage.SOURCE
            and requirement.state is RevisionStageState.MISSING
            and requirement.required_action is RevisionRequiredAction.EXPLICIT_RUN
            for requirement in parent.stage_requirements
        )
        for parent in parents
    )
    if source_runnable:
        try:
            binding = resolve_original_page_asset_binding(
                project,
                page_id=page_id,
                project_path=project_path,
            )
        except OcrRevisionError as exc:
            return None, str(exc) or "The original page is unavailable for OCR."
        if distinct and distinct[0] != binding:
            raise ValueError("selected OCR revision and committed original page differ")
        return binding, ""
    if distinct:
        return distinct[0], ""
    return None, "No explicit OCR revision is available for this page."


def _projected_page(
    project: Mapping[str, Any],
    page: Mapping[str, Any],
    ledger: ProjectEditLedger,
    *,
    ordinal: int,
    project_path: str,
    known_page_ids: frozenset[str],
) -> ProjectedPage:
    page_id = _required_text(page.get("page_id"), "page.page_id")
    snapshot = project_effective_page(project, ledger, page_id=page_id)
    if snapshot.page_id != page_id:
        raise ValueError("effective projection returned the wrong page identity")
    automatic_by_id = _base_parent_map(page)
    effective_parent_ids = tuple(parent.parent_id for parent in snapshot.parents)
    automatic_effective_ids = {
        parent.parent_id
        for parent in snapshot.parents
        if parent.origin is ParentOrigin.AUTOMATIC
    }
    merge_consumed_automatic_ids = {
        source_parent_id
        for parent in snapshot.parents
        if parent.origin is ParentOrigin.USER
        and parent.lineage is not None
        and parent.lineage.order_policy == "replace_sources"
        for source_parent_id in parent.lineage.source_parent_ids
    }
    if set(automatic_by_id) != (
        automatic_effective_ids | merge_consumed_automatic_ids
    ) or automatic_effective_ids.intersection(merge_consumed_automatic_ids):
        raise ValueError(
            "automatic bundles do not exactly match active or merge-retained pipeline evidence"
        )
    if any(
        parent.origin is ParentOrigin.USER
        and parent.parent_id in automatic_by_id
        for parent in snapshot.parents
    ):
        raise ValueError("user parent identity collides with automatic evidence")
    parents = tuple(
        _projected_parent(
            parent,
            automatic_by_id.get(parent.parent_id),
            unresolved_fill_color=_unresolved_fill_color_for_parent(
                parent,
                ledger,
            ),
            unresolved_outline_color=_unresolved_outline_color_for_parent(
                parent,
                ledger,
            ),
        )
        for parent in snapshot.parents
    )
    original_path, original_state = _original_reference(
        page,
        snapshot,
        project_path=project_path,
    )
    cleaned_path, cleaned_state = _cleaned_reference(
        snapshot,
        project_path=project_path,
    )
    rendered_records = _catalog_page_records(
        project,
        catalog_id="rendered_pages",
        page_id=page_id,
        known_page_ids=known_page_ids,
    )
    selected_final = _selected_final_record(rendered_records, snapshot, page)
    final_path, final_state = _final_reference(
        selected_final,
        snapshot,
        project_path=project_path,
    )
    cleanup_state = (
        CleanupState.COMMITTED
        if snapshot.cleaned_base_provenance == _CLEANUP_PROVENANCE
        else CleanupState.IDLE
    )
    presentation_source = page_presentation_input_from_effective_snapshot(
        snapshot,
        required_artifact_state=cleaned_state,
        displayed_final_artifact_state=final_state,
        cleanup_state=cleanup_state,
    )
    row = PageRow(
        page_id=page_id,
        file_name=_page_filename(page, original_path),
        ordinal=ordinal,
        parent_count=len(parents),
        progress_percent=_page_progress(original_state, cleaned_state, final_state),
        presentation=build_page_presentation(presentation_source),
        thumbnail_path=original_path or "",
    )
    overlays = (
        _overlays(snapshot, automatic_by_id)
        + _protected_region_overlays(page, automatic_by_id)
        + _proof_overlays(snapshot, automatic_by_id)
    )
    raster_overlays = _manual_raster_overlays(
        snapshot,
        project_path=project_path,
    )
    try:
        canvas_size = page_canvas_size_for_project_page(
            page,
            project_path=project_path,
        )
    except ParentGeometryCommandError as exc:
        if exc.code is not ParentGeometryCommandErrorCode.CANVAS_UNAVAILABLE:
            raise
        canvas_size = None
    original_page_binding, original_page_binding_problem = (
        _original_page_binding_for_ocr(
            project,
            page_id=page_id,
            project_path=project_path,
            parents=parents,
        )
    )
    return ProjectedPage(
        page_row=row,
        effective=snapshot,
        parents=parents,
        automatic_ordered_parent_ids=automatic_ordered_parent_ids_for_page(page),
        canvas_artifacts=CanvasArtifactReferences(
            page_id=page_id,
            original_path=original_path,
            cleaned_path=cleaned_path,
            final_path=final_path,
        ),
        canvas_size=canvas_size,
        original_page_binding=original_page_binding,
        original_page_binding_problem=original_page_binding_problem,
        overlays=overlays,
        raster_overlays=raster_overlays,
        overlay_availability=_overlay_availability(overlays, raster_overlays),
        original_artifact_state=original_state,
        cleaned_artifact_state=cleaned_state,
        final_artifact_state=final_state,
        cleanup_state=cleanup_state,
        edit_history=_edit_history(
            ledger,
            snapshot,
            page_id=page_id,
        ),
        artifact_history=_history(
            project,
            snapshot,
            page_id=page_id,
            project_path=project_path,
            known_page_ids=known_page_ids,
        ),
    )


def _language_values(project: Mapping[str, Any]) -> tuple[str, str]:
    metadata = project.get("project")
    if not isinstance(metadata, Mapping):
        raise ValueError("project metadata is missing")
    settings = project.get("settings")
    config: Mapping[str, Any] = {}
    if isinstance(settings, Mapping) and isinstance(
        settings.get("project_config"), Mapping
    ):
        container = settings["project_config"]
        project_config = container.get("project")
        if isinstance(project_config, Mapping):
            config = project_config
    language = metadata.get("language")
    language_mapping = language if isinstance(language, Mapping) else {}
    source = str(
        config.get("source_language") or language_mapping.get("source") or "Unknown"
    ).strip()
    target = str(
        config.get("target_language") or language_mapping.get("target") or "Unknown"
    ).strip()
    source_label = _LANGUAGE_LABELS.get(source.lower(), source or "Unknown")
    target_label = _LANGUAGE_LABELS.get(target.lower(), target or "Unknown")
    return source_label, target_label


def _recoverable(project: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    values = (
        metadata.get("recoverable"),
        project.get("recovery_available"),
    )
    for value in values:
        if value is None:
            continue
        if not isinstance(value, bool):
            raise TypeError("recoverable project state must be a boolean")
        return value
    return False


def _project_presentation(
    pages: tuple[ProjectedPage, ...],
    *,
    recoverable: bool,
) -> Presentation:
    if recoverable:
        return resolve_state_presentation("page", PageState.RECOVERY)
    page_states = tuple(page.page_row.presentation.source.page_state for page in pages)
    for state in (
        PageState.CONFLICT,
        PageState.ERROR,
        PageState.MISSING,
        PageState.STALE,
    ):
        if state in page_states:
            return resolve_state_presentation("page", state)
    if any(
        page.cleaned_artifact_state in {ArtifactState.MISSING, ArtifactState.INVALID}
        or page.final_artifact_state in {ArtifactState.MISSING, ArtifactState.INVALID}
        for page in pages
    ):
        return resolve_state_presentation("page", PageState.MISSING)
    if any(page.final_artifact_state is ArtifactState.STALE for page in pages):
        return resolve_state_presentation("page", PageState.STALE)
    return resolve_state_presentation("page", PageState.NORMAL)


def project_ui_projection(
    project: Mapping[str, Any],
    ledger: ProjectEditLedger,
    *,
    project_path: str,
) -> ProjectUiProjection:
    """Project one loaded schema-2 project into immutable native-shell inputs.

    The passed ledger must be the exact embedded/materialized ledger carried by
    the loaded project.  Candidate edits belong to GUI-6 command services, not
    this read-only GUI-5 adapter.
    """

    if not isinstance(project, Mapping):
        raise TypeError("project must be a mapping")
    if not isinstance(ledger, ProjectEditLedger):
        raise TypeError("ledger must be ProjectEditLedger")
    validate_project_schema_v2(project)
    metadata_mapping = project.get("project")
    if not isinstance(metadata_mapping, Mapping):
        raise ValueError("project metadata is missing")
    project_id = _required_text(metadata_mapping.get("project_id"), "project_id")
    if ledger.project_id != project_id:
        raise ValueError("project and ledger identities differ")
    embedded_mapping = project.get("edit_ledger")
    if not isinstance(embedded_mapping, Mapping):
        raise ValueError("embedded edit ledger is missing")
    embedded = ProjectEditLedger.from_dict(embedded_mapping)
    if embedded.project_id != project_id or embedded.fingerprint() != ledger.fingerprint():
        raise ValueError("passed ledger differs from the loaded project ledger")

    normalized_path = _normalized_project_path(project_path)
    raw_pages = project.get("pages")
    if not isinstance(raw_pages, list):
        raise ValueError("project pages must be a list")
    page_ids: list[str] = []
    for page in raw_pages:
        if not isinstance(page, Mapping):
            raise ValueError("project pages must contain mappings")
        page_ids.append(_required_text(page.get("page_id"), "page.page_id"))
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("project page identities must be unique")
    known_page_ids = frozenset(page_ids)
    pages = tuple(
        _projected_page(
            project,
            page,
            ledger,
            ordinal=index + 1,
            project_path=normalized_path,
            known_page_ids=known_page_ids,
        )
        for index, page in enumerate(raw_pages)
    )
    try:
        glossary = project_glossary_snapshot(project, ledger)
        automatic_glossary = project_glossary_snapshot(
            project,
            ProjectEditLedger(project_id=project_id),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("project glossary projection is invalid") from exc
    if glossary.page_ids != tuple(page_ids):
        raise ValueError("project glossary page order differs from the project")
    glossary_changed = glossary.fingerprint != automatic_glossary.fingerprint
    glossary_stale_page_ids = tuple(
        page.effective.page_id
        for page in pages
        if glossary_changed
        and any(
            parent.effective.target_text.strip()
            for parent in page.parents
            if not parent.effective.excluded
        )
    )
    source_language, target_language = _language_values(project)
    filename = Path(normalized_path).name
    named_file_value = (
        named_project_display_name(normalized_path)
        if filename.casefold().endswith(".yomiframe.json")
        else ""
    )
    name = (
        named_file_value
        or str(metadata_mapping.get("name") or "").strip()
        or Path(normalized_path).stem
    )
    recoverable = _recoverable(project, metadata_mapping)
    completed_count = sum(
        page.final_artifact_state is ArtifactState.VALID for page in pages
    )
    metadata = ProjectMetadata(
        project_id=project_id,
        name=name,
        project_path=normalized_path,
        schema_version=_required_text(project.get("schema_version"), "schema_version"),
        source_language=source_language,
        target_language=target_language,
        page_count=len(pages),
        completed_count=completed_count,
        recoverable=recoverable,
    )
    project_row = ProjectRow(
        project_id=project_id,
        name=name,
        path=normalized_path,
        language_pair=f"{source_language} -> {target_language}",
        page_count=len(pages),
        completed_count=completed_count,
        recoverable=recoverable,
        presentation=_project_presentation(pages, recoverable=recoverable),
        thumbnail_path=(
            (pages[0].canvas_artifacts.original_path or "") if pages else ""
        ),
    )
    return ProjectUiProjection(
        metadata=metadata,
        source_project_fingerprint=canonical_sha256(project),
        project_row=project_row,
        page_rows=tuple(page.page_row for page in pages),
        pages=pages,
        glossary_entries=glossary.entries,
        glossary_fingerprint=glossary.fingerprint,
        glossary_history=_project_glossary_history(
            ledger,
            glossary.effective_pages,
        ),
        glossary_stale_page_ids=glossary_stale_page_ids,
    )


__all__ = [
    "ArtifactRevisionReference",
    "CanvasArtifactReferences",
    "EditHistoryReference",
    "OverlayAvailabilityData",
    "OverlayShapeData",
    "ProjectMetadata",
    "ProjectUiProjection",
    "ProjectedPage",
    "ProjectedParent",
    "RasterOverlayData",
    "project_ui_projection",
]
