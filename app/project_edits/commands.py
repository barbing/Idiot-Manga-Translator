# -*- coding: utf-8 -*-
"""Typed GUI commands for durable project edits.

GUI-6 command services coordinate the existing immutable ledger, central
projector, invalidation owner, and page/global compare-and-swap store.  They
never invoke OCR, translation, cleanup, rendering, or any other pipeline
owner.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
import re
from typing import Any, Mapping
import uuid

from app.pipeline.hierarchy_revision_contracts import ParentOrigin
from app.pipeline.ocr_revision_contracts import OcrSourceRevisionArtifact
from app.pipeline.translation_revision_contracts import TranslationRevisionArtifact
from app.io.project_edit_store import (
    ProjectEditCommitReceipt,
    ProjectEditReadSnapshot,
    ProjectEditStore,
    StalePageEditHeadError,
    StaleProjectEditHeadError,
)
from .contracts import (
    CANONICAL_WRITING_MODES,
    USER_PARENT_ID_PREFIX,
    USER_PARENT_IDENTITY_NAMESPACE,
    USER_ROOT_ID_PREFIX,
    USER_ROOT_IDENTITY_NAMESPACE,
    EditDomain,
    EditTarget,
    EditTargetKind,
    ParentSourceEvidenceMappingV1,
    ParentStageRequirement,
    ProjectEdit,
    SourceTextRevisionBaseV1,
    TargetTextRevisionBaseV1,
    canonical_render_fill_color,
    canonical_render_outline_color,
    canonical_render_outline_width,
    canonical_render_preferred_size,
    canonical_render_shadow_blur,
    canonical_render_font_role,
    canonical_render_line_height,
    canonical_render_rotation,
    create_project_edit,
    thaw_json,
    validate_user_parent_identity_pair,
)
from .fingerprints import canonical_sha256, project_id_for
from .invalidation import (
    Dependency,
    InvalidationAction,
    InvalidationResult,
    InvalidationScope,
    invalidation_for_control,
    invalidation_for_edit,
)
from .ledger import ProjectEditLedger
from .projection import (
    EffectivePageSnapshot,
    EffectiveParentSnapshot,
    ProjectionIssue,
    TargetFreshness,
    automatic_ordered_parent_ids_for_page,
    automatic_render_fill_color,
    automatic_render_outline_color,
    automatic_render_outline_width,
    automatic_render_preferred_size,
    automatic_render_shadow_blur,
    automatic_render_shadow_enabled,
    automatic_render_font_role,
    automatic_render_line_height,
    automatic_render_rotation,
    automatic_render_writing_mode,
    effective_source_fingerprint,
    field_base_fingerprint,
    project_effective_page,
    source_text_revision_base_for_parent,
    target_text_revision_base_for_parent,
)


_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class SourceTextOperation(str, Enum):
    REPLACE = "replace"
    RESTORE_AUTOMATIC = "restore_automatic"
    RESTORE_SELECTED_REVISION = "restore_selected_revision"


class TargetTextOperation(str, Enum):
    REPLACE = "replace"
    RESTORE_AUTOMATIC = "restore_automatic"
    RESTORE_SELECTED_REVISION = "restore_selected_revision"
    RESTORE_MAPPED_PIPELINE = "restore_mapped_pipeline"


class ParentMembershipOperation(str, Enum):
    EXCLUDE = "exclude"
    RESTORE = "restore"


class ParentGeometryOperation(str, Enum):
    SET_GEOMETRY = "set_geometry"


class AddUserParentOperation(str, Enum):
    ADD = "add_user_parent"


class SplitUserParentOperation(str, Enum):
    SPLIT = "split_user_parent"


class MergePipelineParentsOperation(str, Enum):
    MERGE = "merge_pipeline_parents"


class SplitUserParentOrientation(str, Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


class ReadingOrderOperation(str, Enum):
    SET = "set_reading_order"


class RenderLayoutWritingModeOperation(str, Enum):
    SET = "set"
    RESTORE_AUTOMATIC = "restore_automatic"


class RenderLayoutLineHeightOperation(str, Enum):
    SET = "set"
    RESTORE_AUTOMATIC = "restore_automatic"


class RenderLayoutRotationOperation(str, Enum):
    SET = "set"
    RESTORE_AUTOMATIC = "restore_automatic"


class RenderStyleFillColorOperation(str, Enum):
    SET = "set"
    RESTORE_AUTOMATIC = "restore_automatic"


class RenderStyleOutlineColorOperation(str, Enum):
    SET = "set"
    RESTORE_AUTOMATIC = "restore_automatic"


class RenderStyleOutlineWidthOperation(str, Enum):
    SET = "set"
    RESTORE_AUTOMATIC = "restore_automatic"


class RenderStylePreferredSizeOperation(str, Enum):
    SET = "set"
    RESTORE_AUTOMATIC = "restore_automatic"


class RenderStyleShadowVisibilityOperation(str, Enum):
    HIDE = "hide"
    RESTORE_AUTOMATIC = "restore_automatic"


class RenderStyleShadowBlurOperation(str, Enum):
    SET = "set"
    RESTORE_AUTOMATIC = "restore_automatic"


class RenderStyleFontRoleOperation(str, Enum):
    SET = "set"
    RESTORE_AUTOMATIC = "restore_automatic"


class EditHistoryOperation(str, Enum):
    REVOKE = "revoke"
    REAPPLY = "reapply"


class SourceTextCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    SOURCE_SLOT_CONFLICT = "source_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    INVALID_OPERATION = "invalid_operation"
    REVISION_BASE_REQUIRED = "revision_base_required"
    REVISION_BASE_NOT_ALLOWED = "revision_base_not_allowed"
    REVISION_ID_MISMATCH = "revision_id_mismatch"
    REVISION_SELECTION_MISMATCH = "revision_selection_mismatch"
    REVISION_ARTIFACT_MISMATCH = "revision_artifact_mismatch"
    PARENT_LINEAGE_MISMATCH = "parent_lineage_mismatch"
    NO_OP = "no_op"
    PROJECTION_REJECTED = "projection_rejected"


class TargetTextCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    TARGET_SLOT_CONFLICT = "target_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    INVALID_OPERATION = "invalid_operation"
    REVISION_BASE_REQUIRED = "revision_base_required"
    REVISION_BASE_NOT_ALLOWED = "revision_base_not_allowed"
    REVISION_ID_MISMATCH = "revision_id_mismatch"
    REVISION_SELECTION_MISMATCH = "revision_selection_mismatch"
    REVISION_ARTIFACT_MISMATCH = "revision_artifact_mismatch"
    REVISION_SOURCE_MISMATCH = "revision_source_mismatch"
    REVISION_HIERARCHY_MISMATCH = "revision_hierarchy_mismatch"
    MAPPED_BASE_REQUIRED = "mapped_base_required"
    MAPPED_BASE_NOT_ALLOWED = "mapped_base_not_allowed"
    MAPPED_BASE_MISMATCH = "mapped_base_mismatch"
    PARENT_LINEAGE_MISMATCH = "parent_lineage_mismatch"
    NO_OP = "no_op"
    PROJECTION_REJECTED = "projection_rejected"


class ParentMembershipCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    MEMBERSHIP_SLOT_CONFLICT = "membership_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class ParentGeometryCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    CANVAS_UNAVAILABLE = "canvas_unavailable"
    INVALID_GEOMETRY = "invalid_geometry"
    GEOMETRY_OUT_OF_BOUNDS = "geometry_out_of_bounds"
    NO_OP = "no_op"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    GEOMETRY_SLOT_CONFLICT = "geometry_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class AddUserParentCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    CANVAS_UNAVAILABLE = "canvas_unavailable"
    INVALID_WORKFLOW_AREA = "invalid_workflow_area"
    WORKFLOW_AREA_OUT_OF_BOUNDS = "workflow_area_out_of_bounds"
    IDENTITY_COLLISION = "identity_collision"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class SplitUserParentCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    SOURCE_PARENT_NOT_FOUND = "source_parent_not_found"
    SOURCE_PARENT_NOT_STANDALONE = "source_parent_not_standalone"
    SOURCE_PARENT_EXCLUDED = "source_parent_excluded"
    CANVAS_UNAVAILABLE = "canvas_unavailable"
    INVALID_SPLIT_OFFSET = "invalid_split_offset"
    SOURCE_EVIDENCE_PARTITION_INVALID = "source_evidence_partition_invalid"
    IDENTITY_COLLISION = "identity_collision"
    SPLIT_SLOT_CONFLICT = "split_slot_conflict"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class MergePipelineParentsCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    SOURCE_PARENT_NOT_FOUND = "source_parent_not_found"
    SOURCE_PARENT_NOT_AUTOMATIC = "source_parent_not_automatic"
    SOURCE_PARENT_EXCLUDED = "source_parent_excluded"
    SOURCE_PARENT_EDITED = "source_parent_edited"
    SOURCE_EVIDENCE_UNAVAILABLE = "source_evidence_unavailable"
    ROLE_MISMATCH = "role_mismatch"
    SOURCES_NOT_CONSECUTIVE = "sources_not_consecutive"
    CANVAS_UNAVAILABLE = "canvas_unavailable"
    IDENTITY_COLLISION = "identity_collision"
    MERGE_SLOT_CONFLICT = "merge_slot_conflict"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class ReadingOrderCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    SELECTED_PARENT_EXCLUDED = "selected_parent_excluded"
    AUTOMATIC_ORDER_UNAVAILABLE = "automatic_order_unavailable"
    INVALID_PERMUTATION = "invalid_permutation"
    EXCLUDED_PARENT_MOVED = "excluded_parent_moved"
    MULTIPLE_PARENTS_MOVED = "multiple_parents_moved"
    NO_OP = "no_op"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    READING_ORDER_SLOT_CONFLICT = "reading_order_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class RenderLayoutWritingModeCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_EXCLUDED = "parent_excluded"
    PARENT_NOT_RENDER_REQUIRED = "parent_not_render_required"
    AUTOMATIC_WRITING_MODE_UNAVAILABLE = "automatic_writing_mode_unavailable"
    NO_OP = "no_op"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    WRITING_MODE_SLOT_CONFLICT = "writing_mode_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class RenderLayoutLineHeightCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_EXCLUDED = "parent_excluded"
    PARENT_NOT_RENDER_REQUIRED = "parent_not_render_required"
    AUTOMATIC_LINE_HEIGHT_UNAVAILABLE = "automatic_line_height_unavailable"
    NO_OP = "no_op"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    LINE_HEIGHT_SLOT_CONFLICT = "line_height_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class RenderLayoutRotationCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_EXCLUDED = "parent_excluded"
    PARENT_NOT_RENDER_REQUIRED = "parent_not_render_required"
    AUTOMATIC_ROTATION_UNAVAILABLE = "automatic_rotation_unavailable"
    NO_OP = "no_op"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    ROTATION_SLOT_CONFLICT = "rotation_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class RenderStyleFillColorCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_EXCLUDED = "parent_excluded"
    PARENT_NOT_RENDER_REQUIRED = "parent_not_render_required"
    AUTOMATIC_FILL_COLOR_UNAVAILABLE = "automatic_fill_color_unavailable"
    NO_OP = "no_op"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    FILL_COLOR_SLOT_CONFLICT = "fill_color_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class RenderStyleOutlineColorCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_EXCLUDED = "parent_excluded"
    PARENT_NOT_RENDER_REQUIRED = "parent_not_render_required"
    AUTOMATIC_OUTLINE_COLOR_UNAVAILABLE = "automatic_outline_color_unavailable"
    NO_OP = "no_op"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    OUTLINE_COLOR_SLOT_CONFLICT = "outline_color_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class RenderStyleOutlineWidthCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_EXCLUDED = "parent_excluded"
    PARENT_NOT_RENDER_REQUIRED = "parent_not_render_required"
    AUTOMATIC_OUTLINE_WIDTH_UNAVAILABLE = "automatic_outline_width_unavailable"
    NO_OP = "no_op"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    OUTLINE_WIDTH_SLOT_CONFLICT = "outline_width_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class RenderStylePreferredSizeCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_EXCLUDED = "parent_excluded"
    PARENT_NOT_RENDER_REQUIRED = "parent_not_render_required"
    AUTOMATIC_PREFERRED_SIZE_UNAVAILABLE = "automatic_preferred_size_unavailable"
    NO_OP = "no_op"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    PREFERRED_SIZE_SLOT_CONFLICT = "preferred_size_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class RenderStyleShadowVisibilityCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_EXCLUDED = "parent_excluded"
    PARENT_NOT_RENDER_REQUIRED = "parent_not_render_required"
    AUTOMATIC_SHADOW_UNAVAILABLE = "automatic_shadow_unavailable"
    NO_OP = "no_op"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    SHADOW_VISIBILITY_SLOT_CONFLICT = "shadow_visibility_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class RenderStyleShadowBlurCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_EXCLUDED = "parent_excluded"
    PARENT_NOT_RENDER_REQUIRED = "parent_not_render_required"
    AUTOMATIC_SHADOW_UNAVAILABLE = "automatic_shadow_unavailable"
    NO_OP = "no_op"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    SHADOW_BLUR_SLOT_CONFLICT = "shadow_blur_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class RenderStyleFontRoleCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_EXCLUDED = "parent_excluded"
    PARENT_NOT_RENDER_REQUIRED = "parent_not_render_required"
    AUTOMATIC_FONT_ROLE_UNAVAILABLE = "automatic_font_role_unavailable"
    NO_OP = "no_op"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    FONT_ROLE_SLOT_CONFLICT = "font_role_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class EditHistoryCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    TARGET_EDIT_NOT_FOUND = "target_edit_not_found"
    TARGET_EDIT_PAGE_MISMATCH = "target_edit_page_mismatch"
    CONTROL_TARGET_FORBIDDEN = "control_target_forbidden"
    ARTIFACT_TARGET_FORBIDDEN = "artifact_target_forbidden"
    ALREADY_ACTIVE = "already_active"
    ALREADY_REVOKED = "already_revoked"
    ACTIVE_DEPENDENT_EDIT = "active_dependent_edit"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class SourceTextCommandError(RuntimeError):
    """A fail-closed source-text command failure with a stable UI code."""

    def __init__(
        self,
        code: SourceTextCommandErrorCode,
        message: str,
    ) -> None:
        self.code = SourceTextCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class TargetTextCommandError(RuntimeError):
    """A fail-closed target-text command failure with a stable UI code."""

    def __init__(
        self,
        code: TargetTextCommandErrorCode,
        message: str,
    ) -> None:
        self.code = TargetTextCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class ParentMembershipCommandError(RuntimeError):
    """A fail-closed parent-membership command failure with a stable UI code."""

    def __init__(
        self,
        code: ParentMembershipCommandErrorCode,
        message: str,
    ) -> None:
        self.code = ParentMembershipCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class ParentGeometryCommandError(RuntimeError):
    """A fail-closed parent-geometry command failure with a stable UI code."""

    def __init__(
        self,
        code: ParentGeometryCommandErrorCode,
        message: str,
    ) -> None:
        self.code = ParentGeometryCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class AddUserParentCommandError(RuntimeError):
    """A fail-closed user-parent topology command failure."""

    def __init__(
        self,
        code: AddUserParentCommandErrorCode,
        message: str,
    ) -> None:
        self.code = AddUserParentCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class SplitUserParentCommandError(RuntimeError):
    """A fail-closed standalone user-parent split failure."""

    def __init__(
        self,
        code: SplitUserParentCommandErrorCode,
        message: str,
    ) -> None:
        self.code = SplitUserParentCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class MergePipelineParentsCommandError(RuntimeError):
    """A fail-closed pipeline-parent merge failure."""

    def __init__(
        self,
        code: MergePipelineParentsCommandErrorCode,
        message: str,
    ) -> None:
        self.code = MergePipelineParentsCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class ReadingOrderCommandError(RuntimeError):
    """A fail-closed page-wide reading-order command failure."""

    def __init__(
        self,
        code: ReadingOrderCommandErrorCode,
        message: str,
    ) -> None:
        self.code = ReadingOrderCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class RenderLayoutWritingModeCommandError(RuntimeError):
    """A fail-closed writing-mode command failure with a stable UI code."""

    def __init__(
        self,
        code: RenderLayoutWritingModeCommandErrorCode,
        message: str,
    ) -> None:
        self.code = RenderLayoutWritingModeCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class RenderLayoutLineHeightCommandError(RuntimeError):
    """A fail-closed line-height command failure with a stable UI code."""

    def __init__(
        self,
        code: RenderLayoutLineHeightCommandErrorCode,
        message: str,
    ) -> None:
        self.code = RenderLayoutLineHeightCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class RenderLayoutRotationCommandError(RuntimeError):
    """A fail-closed rotation command failure with a stable UI code."""

    def __init__(
        self,
        code: RenderLayoutRotationCommandErrorCode,
        message: str,
    ) -> None:
        self.code = RenderLayoutRotationCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class RenderStyleFillColorCommandError(RuntimeError):
    """A fail-closed opaque fill-color command failure with a stable UI code."""

    def __init__(
        self,
        code: RenderStyleFillColorCommandErrorCode,
        message: str,
    ) -> None:
        self.code = RenderStyleFillColorCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class RenderStyleOutlineColorCommandError(RuntimeError):
    """A fail-closed opaque outline-color command failure."""

    def __init__(
        self,
        code: RenderStyleOutlineColorCommandErrorCode,
        message: str,
    ) -> None:
        self.code = RenderStyleOutlineColorCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class RenderStyleOutlineWidthCommandError(RuntimeError):
    """A fail-closed outline-width command failure."""

    def __init__(
        self,
        code: RenderStyleOutlineWidthCommandErrorCode,
        message: str,
    ) -> None:
        self.code = RenderStyleOutlineWidthCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class RenderStylePreferredSizeCommandError(RuntimeError):
    """A fail-closed preferred-size command failure."""

    def __init__(
        self,
        code: RenderStylePreferredSizeCommandErrorCode,
        message: str,
    ) -> None:
        self.code = RenderStylePreferredSizeCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class RenderStyleShadowVisibilityCommandError(RuntimeError):
    """A fail-closed shadow-visibility command failure."""

    def __init__(
        self,
        code: RenderStyleShadowVisibilityCommandErrorCode,
        message: str,
    ) -> None:
        self.code = RenderStyleShadowVisibilityCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class RenderStyleShadowBlurCommandError(RuntimeError):
    """A fail-closed shadow-blur command failure."""

    def __init__(
        self,
        code: RenderStyleShadowBlurCommandErrorCode,
        message: str,
    ) -> None:
        self.code = RenderStyleShadowBlurCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class RenderStyleFontRoleCommandError(RuntimeError):
    """A fail-closed registered font-role command failure."""

    def __init__(
        self,
        code: RenderStyleFontRoleCommandErrorCode,
        message: str,
    ) -> None:
        self.code = RenderStyleFontRoleCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class EditHistoryCommandError(RuntimeError):
    """A fail-closed durable-history command failure with a stable UI code."""

    def __init__(
        self,
        code: EditHistoryCommandErrorCode,
        message: str,
    ) -> None:
        self.code = EditHistoryCommandErrorCode(code)
        super().__init__(str(message))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


def _require_identity(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty exact identifier")
    return value


def _require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a SHA-256 hex digest")
    candidate = value.lower()
    if len(candidate) != 64 or any(
        character not in "0123456789abcdef" for character in candidate
    ):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    return candidate


def create_user_parent_identity(
    identity_hex: str | None = None,
) -> tuple[str, str]:
    """Create one stable parent/root pair for a command and all of its retries."""

    suffix = uuid.uuid4().hex if identity_hex is None else str(identity_hex)
    if (
        len(suffix) != 32
        or suffix != suffix.lower()
        or any(character not in "0123456789abcdef" for character in suffix)
    ):
        raise ValueError("identity_hex must contain 32 lowercase hex characters")
    return USER_PARENT_ID_PREFIX + suffix, USER_ROOT_ID_PREFIX + suffix


@dataclass(frozen=True)
class SourceTextCommand:
    """One exact, revision-bound selected-parent source-text command."""

    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: SourceTextOperation
    text: str
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str
    revision_base: SourceTextRevisionBaseV1 | None = None

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        object.__setattr__(
            self,
            "project_id",
            _require_identity(self.project_id, "project_id"),
        )
        object.__setattr__(
            self,
            "page_id",
            _require_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _require_identity(self.parent_id, "parent_id"),
        )
        operation = SourceTextOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if operation in {
            SourceTextOperation.RESTORE_AUTOMATIC,
            SourceTextOperation.RESTORE_SELECTED_REVISION,
        } and self.text != "":
            raise ValueError(f"{operation.value} must not carry replacement text")
        revision_base = self.revision_base
        if revision_base is not None and not isinstance(
            revision_base,
            SourceTextRevisionBaseV1,
        ):
            raise TypeError("revision_base must be a SourceTextRevisionBaseV1")
        if (
            operation is SourceTextOperation.RESTORE_SELECTED_REVISION
            and revision_base is None
        ):
            raise ValueError("restore_selected_revision requires revision_base")
        if (
            operation is SourceTextOperation.RESTORE_AUTOMATIC
            and revision_base is not None
        ):
            raise ValueError("restore_automatic must not carry revision_base")
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _require_sha256(
                self.expected_effective_page_fingerprint,
                "expected_effective_page_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "expected_page_head_sha256",
            _require_sha256(
                self.expected_page_head_sha256,
                "expected_page_head_sha256",
            ),
        )
        object.__setattr__(
            self,
            "expected_global_head_sha256",
            _require_sha256(
                self.expected_global_head_sha256,
                "expected_global_head_sha256",
            ),
        )


@dataclass(frozen=True)
class TargetTextCommand:
    """One exact, revision-bound selected-parent target-text command."""

    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: TargetTextOperation
    text: str
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str
    revision_base: TargetTextRevisionBaseV1 | None = None
    source_evidence_base: ParentSourceEvidenceMappingV1 | None = None

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        object.__setattr__(
            self,
            "project_id",
            _require_identity(self.project_id, "project_id"),
        )
        object.__setattr__(
            self,
            "page_id",
            _require_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _require_identity(self.parent_id, "parent_id"),
        )
        operation = TargetTextOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if operation in {
            TargetTextOperation.RESTORE_AUTOMATIC,
            TargetTextOperation.RESTORE_SELECTED_REVISION,
            TargetTextOperation.RESTORE_MAPPED_PIPELINE,
        } and self.text != "":
            raise ValueError(f"{operation.value} must not carry replacement text")
        revision_base = self.revision_base
        if revision_base is not None and not isinstance(
            revision_base,
            TargetTextRevisionBaseV1,
        ):
            raise TypeError("revision_base must be a TargetTextRevisionBaseV1")
        if (
            operation is TargetTextOperation.RESTORE_SELECTED_REVISION
            and revision_base is None
        ):
            raise ValueError("restore_selected_revision requires revision_base")
        if (
            operation is TargetTextOperation.RESTORE_AUTOMATIC
            and revision_base is not None
        ):
            raise ValueError("restore_automatic must not carry revision_base")
        source_evidence_base = self.source_evidence_base
        if source_evidence_base is not None and not isinstance(
            source_evidence_base,
            ParentSourceEvidenceMappingV1,
        ):
            raise TypeError(
                "source_evidence_base must be a ParentSourceEvidenceMappingV1"
            )
        if revision_base is not None and source_evidence_base is not None:
            raise ValueError("target commands may carry only one immutable base")
        if (
            operation is TargetTextOperation.RESTORE_MAPPED_PIPELINE
            and source_evidence_base is None
        ):
            raise ValueError("restore_mapped_pipeline requires source_evidence_base")
        if (
            operation
            in {
                TargetTextOperation.RESTORE_AUTOMATIC,
                TargetTextOperation.RESTORE_SELECTED_REVISION,
            }
            and source_evidence_base is not None
        ):
            raise ValueError(
                f"{operation.value} must not carry source_evidence_base"
            )
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _require_sha256(
                self.expected_effective_page_fingerprint,
                "expected_effective_page_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "expected_page_head_sha256",
            _require_sha256(
                self.expected_page_head_sha256,
                "expected_page_head_sha256",
            ),
        )
        object.__setattr__(
            self,
            "expected_global_head_sha256",
            _require_sha256(
                self.expected_global_head_sha256,
                "expected_global_head_sha256",
            ),
        )


@dataclass(frozen=True)
class ParentMembershipCommand:
    """One revision-bound selected-parent exclude or restore command."""

    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: ParentMembershipOperation
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        object.__setattr__(
            self,
            "project_id",
            _require_identity(self.project_id, "project_id"),
        )
        object.__setattr__(
            self,
            "page_id",
            _require_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _require_identity(self.parent_id, "parent_id"),
        )
        object.__setattr__(
            self,
            "operation",
            ParentMembershipOperation(self.operation),
        )
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _require_sha256(
                self.expected_effective_page_fingerprint,
                "expected_effective_page_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "expected_page_head_sha256",
            _require_sha256(
                self.expected_page_head_sha256,
                "expected_page_head_sha256",
            ),
        )
        object.__setattr__(
            self,
            "expected_global_head_sha256",
            _require_sha256(
                self.expected_global_head_sha256,
                "expected_global_head_sha256",
            ),
        )


def _exact_bbox(value: Any, field_name: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{field_name} must contain four exact integers")
    bbox = tuple(int(item) for item in value)
    if bbox[0] < 0 or bbox[1] < 0:
        raise ValueError(f"{field_name} origin must not be negative")
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError(f"{field_name} width and height must be positive")
    return bbox


@dataclass(frozen=True)
class ParentGeometryCommand:
    """One revision-bound selected-parent structural geometry command."""

    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: ParentGeometryOperation
    bbox: tuple[int, int, int, int]
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        object.__setattr__(
            self,
            "project_id",
            _require_identity(self.project_id, "project_id"),
        )
        object.__setattr__(
            self,
            "page_id",
            _require_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _require_identity(self.parent_id, "parent_id"),
        )
        object.__setattr__(
            self,
            "operation",
            ParentGeometryOperation(self.operation),
        )
        object.__setattr__(self, "bbox", _exact_bbox(self.bbox, "bbox"))
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class AddUserParentCommand:
    """One revision-bound standalone user-parent topology request."""

    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    root_id: str
    role: str
    workflow_area_bbox: tuple[int, int, int, int]
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str
    operation: AddUserParentOperation = AddUserParentOperation.ADD

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        for field_name in ("project_id", "page_id", "parent_id", "root_id"):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        validate_user_parent_identity_pair(self.parent_id, self.root_id)
        if self.role not in {"speech", "caption"}:
            raise ValueError("role must be speech or caption")
        object.__setattr__(
            self,
            "workflow_area_bbox",
            _exact_bbox(self.workflow_area_bbox, "workflow_area_bbox"),
        )
        object.__setattr__(
            self,
            "operation",
            AddUserParentOperation(self.operation),
        )
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class SplitUserParentCommand:
    """One revision-bound split of an Add-created standalone user parent."""

    command_id: str
    project_id: str
    page_id: str
    source_parent_id: str
    first_parent_id: str
    first_root_id: str
    second_parent_id: str
    second_root_id: str
    orientation: SplitUserParentOrientation
    split_offset: int
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str
    operation: SplitUserParentOperation = SplitUserParentOperation.SPLIT

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        for field_name in (
            "project_id",
            "page_id",
            "source_parent_id",
            "first_parent_id",
            "first_root_id",
            "second_parent_id",
            "second_root_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        validate_user_parent_identity_pair(
            self.first_parent_id,
            self.first_root_id,
        )
        validate_user_parent_identity_pair(
            self.second_parent_id,
            self.second_root_id,
        )
        if len(
            {
                self.source_parent_id,
                self.first_parent_id,
                self.second_parent_id,
            }
        ) != 3:
            raise ValueError("source and child parent identities must be unique")
        if self.first_root_id == self.second_root_id:
            raise ValueError("child root identities must be unique")
        object.__setattr__(
            self,
            "orientation",
            SplitUserParentOrientation(self.orientation),
        )
        if (
            isinstance(self.split_offset, bool)
            or not isinstance(self.split_offset, int)
            or self.split_offset <= 0
        ):
            raise ValueError("split_offset must be a positive exact integer")
        object.__setattr__(
            self,
            "operation",
            SplitUserParentOperation(self.operation),
        )
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class MergePipelineParentsCommand:
    """Merge two consecutive immutable pipeline parents into one user parent."""

    command_id: str
    project_id: str
    page_id: str
    source_parent_ids: tuple[str, str]
    merged_parent_id: str
    merged_root_id: str
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str
    operation: MergePipelineParentsOperation = MergePipelineParentsOperation.MERGE

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        for field_name in (
            "project_id",
            "page_id",
            "merged_parent_id",
            "merged_root_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        if (
            not isinstance(self.source_parent_ids, tuple)
            or len(self.source_parent_ids) != 2
        ):
            raise ValueError("source_parent_ids must contain exactly two identities")
        source_parent_ids = tuple(
            _require_identity(value, f"source_parent_ids[{index}]")
            for index, value in enumerate(self.source_parent_ids)
        )
        if len(set(source_parent_ids)) != 2:
            raise ValueError("source_parent_ids must contain two unique identities")
        object.__setattr__(self, "source_parent_ids", source_parent_ids)
        validate_user_parent_identity_pair(
            self.merged_parent_id,
            self.merged_root_id,
        )
        if self.merged_parent_id in set(source_parent_ids):
            raise ValueError("merged parent identity must differ from source parents")
        object.__setattr__(
            self,
            "operation",
            MergePipelineParentsOperation(self.operation),
        )
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class ReadingOrderCommand:
    """One revision-bound, page-wide selected-parent order permutation."""

    command_id: str
    project_id: str
    page_id: str
    selected_parent_id: str
    operation: ReadingOrderOperation
    ordered_parent_ids: tuple[str, ...]
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        for field_name in ("project_id", "page_id", "selected_parent_id"):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "operation",
            ReadingOrderOperation(self.operation),
        )
        if not isinstance(self.ordered_parent_ids, tuple):
            raise TypeError("ordered_parent_ids must be an exact tuple")
        if not self.ordered_parent_ids:
            raise ValueError("ordered_parent_ids must not be empty")
        normalized = tuple(
            _require_identity(parent_id, f"ordered_parent_ids[{index}]")
            for index, parent_id in enumerate(self.ordered_parent_ids)
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("ordered_parent_ids must not contain duplicates")
        if self.selected_parent_id not in normalized:
            raise ValueError(
                "selected_parent_id must occur in ordered_parent_ids"
            )
        object.__setattr__(self, "ordered_parent_ids", normalized)
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class RenderLayoutWritingModeCommand:
    """One revision-bound canonical selected-parent writing-mode command."""

    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: RenderLayoutWritingModeOperation
    writing_mode: str
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        object.__setattr__(
            self,
            "project_id",
            _require_identity(self.project_id, "project_id"),
        )
        object.__setattr__(
            self,
            "page_id",
            _require_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _require_identity(self.parent_id, "parent_id"),
        )
        operation = RenderLayoutWritingModeOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if not isinstance(self.writing_mode, str):
            raise TypeError("writing_mode must be a string")
        if operation is RenderLayoutWritingModeOperation.SET:
            if self.writing_mode not in CANONICAL_WRITING_MODES:
                raise ValueError(
                    "writing_mode must be exactly 'horizontal' or 'vertical'"
                )
        elif self.writing_mode != "":
            raise ValueError(
                "restore_automatic must not carry a writing_mode value"
            )
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class RenderLayoutLineHeightCommand:
    """One revision-bound selected-parent line-height command."""

    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: RenderLayoutLineHeightOperation
    line_height: float | None
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        for field_name in ("project_id", "page_id", "parent_id"):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        operation = RenderLayoutLineHeightOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderLayoutLineHeightOperation.SET:
            object.__setattr__(
                self,
                "line_height",
                canonical_render_line_height(self.line_height),
            )
        elif self.line_height is not None:
            raise ValueError(
                "restore_automatic must not carry a line_height value"
            )
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class RenderLayoutRotationCommand:
    """One revision-bound selected-parent clockwise rotation command."""

    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: RenderLayoutRotationOperation
    rotation: float | None
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        for field_name in ("project_id", "page_id", "parent_id"):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        operation = RenderLayoutRotationOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderLayoutRotationOperation.SET:
            object.__setattr__(
                self,
                "rotation",
                canonical_render_rotation(self.rotation),
            )
        elif self.rotation is not None:
            raise ValueError(
                "restore_automatic must not carry a rotation value"
            )
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class RenderStyleFillColorCommand:
    """One revision-bound selected-parent opaque fill-color command."""

    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: RenderStyleFillColorOperation
    fill_color: str | None
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        for field_name in ("project_id", "page_id", "parent_id"):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        operation = RenderStyleFillColorOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleFillColorOperation.SET:
            object.__setattr__(
                self,
                "fill_color",
                canonical_render_fill_color(self.fill_color),
            )
        elif self.fill_color is not None:
            raise ValueError(
                "restore_automatic must not carry a fill_color value"
            )
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class RenderStyleOutlineColorCommand:
    """One revision-bound selected-parent opaque outline-color command."""

    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: RenderStyleOutlineColorOperation
    outline_color: str | None
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        for field_name in ("project_id", "page_id", "parent_id"):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        operation = RenderStyleOutlineColorOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleOutlineColorOperation.SET:
            object.__setattr__(
                self,
                "outline_color",
                canonical_render_outline_color(self.outline_color),
            )
        elif self.outline_color is not None:
            raise ValueError(
                "restore_automatic must not carry an outline_color value"
            )
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class RenderStyleOutlineWidthCommand:
    """One revision-bound selected-parent outline-width command."""

    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: RenderStyleOutlineWidthOperation
    outline_width: float | None
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        for field_name in ("project_id", "page_id", "parent_id"):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        operation = RenderStyleOutlineWidthOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleOutlineWidthOperation.SET:
            object.__setattr__(
                self,
                "outline_width",
                canonical_render_outline_width(self.outline_width),
            )
        elif self.outline_width is not None:
            raise ValueError(
                "restore_automatic must not carry an outline_width value"
            )
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class RenderStylePreferredSizeCommand:
    """One revision-bound selected-parent preferred-size command."""

    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: RenderStylePreferredSizeOperation
    preferred_size: float | None
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        for field_name in ("project_id", "page_id", "parent_id"):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        operation = RenderStylePreferredSizeOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStylePreferredSizeOperation.SET:
            object.__setattr__(
                self,
                "preferred_size",
                canonical_render_preferred_size(self.preferred_size),
            )
        elif self.preferred_size is not None:
            raise ValueError(
                "restore_automatic must not carry a preferred_size value"
            )
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class RenderStyleShadowVisibilityCommand:
    """One revision-bound selected-parent shadow-visibility command."""

    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: RenderStyleShadowVisibilityOperation
    shadow_enabled: bool | None
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        for field_name in ("project_id", "page_id", "parent_id"):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        operation = RenderStyleShadowVisibilityOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleShadowVisibilityOperation.HIDE:
            if self.shadow_enabled is not False:
                if not isinstance(self.shadow_enabled, bool):
                    raise TypeError("shadow_enabled must be the boolean false")
                raise ValueError("Hide must carry only shadow_enabled=false")
        elif self.shadow_enabled is not None:
            raise ValueError(
                "restore_automatic must not carry a shadow_enabled value"
            )
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class RenderStyleShadowBlurCommand:
    """One revision-bound selected-parent shadow-blur command."""

    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: RenderStyleShadowBlurOperation
    shadow_blur: float | None
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        for field_name in ("project_id", "page_id", "parent_id"):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        operation = RenderStyleShadowBlurOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleShadowBlurOperation.SET:
            object.__setattr__(
                self,
                "shadow_blur",
                canonical_render_shadow_blur(self.shadow_blur),
            )
        elif self.shadow_blur is not None:
            raise ValueError("restore_automatic must not carry a shadow_blur value")
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class RenderStyleFontRoleCommand:
    """One revision-bound selected-parent registered font-role command."""

    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: RenderStyleFontRoleOperation
    font_role: str | None
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        for field_name in ("project_id", "page_id", "parent_id"):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        operation = RenderStyleFontRoleOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleFontRoleOperation.SET:
            object.__setattr__(
                self,
                "font_role",
                canonical_render_font_role(self.font_role),
            )
        elif self.font_role is not None:
            raise ValueError("restore_automatic must not carry a font_role value")
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class EditHistoryCommand:
    """One exact, revision-bound durable Revoke or Reapply request."""

    command_id: str
    project_id: str
    page_id: str
    target_edit_id: str
    operation: EditHistoryOperation
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        command_id = _require_identity(self.command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)
        for field_name in ("project_id", "page_id", "target_edit_id"):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "operation",
            EditHistoryOperation(self.operation),
        )
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class SourceTextCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    before_source_text: str
    after_source_text: str
    before_source_authority: str
    after_source_authority: str
    before_target_text: str
    after_target_text: str
    before_target_authority: str
    after_target_authority: str
    before_target_freshness: TargetFreshness
    after_target_freshness: TargetFreshness
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class TargetTextCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    before_target_text: str
    after_target_text: str
    before_target_authority: str
    after_target_authority: str
    source_fingerprint: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class ParentMembershipCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    before_excluded: bool
    after_excluded: bool
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class ParentGeometryCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    before_bbox: tuple[int, int, int, int]
    after_bbox: tuple[int, int, int, int]
    canvas_size: tuple[int, int]
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class AddUserParentCommandReceipt:
    command_id: str
    edit: ProjectEdit
    parent_id: str
    root_id: str
    role: str
    workflow_area_bbox: tuple[int, int, int, int]
    canvas_size: tuple[int, int]
    before_hierarchy_revision_id: str
    after_hierarchy_revision_id: str
    before_hierarchy_fingerprint: str
    after_hierarchy_fingerprint: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    stage_requirements: tuple[ParentStageRequirement, ...]
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class SplitUserParentCommandReceipt:
    command_id: str
    edit: ProjectEdit
    source_parent_id: str
    source_root_id: str
    source_role: str
    source_workflow_area_bbox: tuple[int, int, int, int]
    orientation: SplitUserParentOrientation
    split_offset: int
    child_parent_ids: tuple[str, str]
    child_root_ids: tuple[str, str]
    child_workflow_area_bboxes: tuple[
        tuple[int, int, int, int],
        tuple[int, int, int, int],
    ]
    canvas_size: tuple[int, int]
    before_hierarchy_revision_id: str
    after_hierarchy_revision_id: str
    before_hierarchy_fingerprint: str
    after_hierarchy_fingerprint: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    child_stage_requirements: tuple[
        tuple[ParentStageRequirement, ...],
        tuple[ParentStageRequirement, ...],
    ]
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class MergePipelineParentsCommandReceipt:
    command_id: str
    edit: ProjectEdit
    source_parent_ids: tuple[str, str]
    source_root_ids: tuple[str, str]
    source_automatic_fingerprints: tuple[str, str]
    source_bboxes: tuple[
        tuple[int, int, int, int],
        tuple[int, int, int, int],
    ]
    source_texts: tuple[str, str]
    source_text_fingerprints: tuple[str, str]
    source_role: str
    merged_parent_id: str
    merged_root_id: str
    merged_workflow_area_bbox: tuple[int, int, int, int]
    merged_source_text: str
    canvas_size: tuple[int, int]
    before_hierarchy_revision_id: str
    after_hierarchy_revision_id: str
    before_hierarchy_fingerprint: str
    after_hierarchy_fingerprint: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    stage_requirements: tuple[ParentStageRequirement, ...]
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class ReadingOrderCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    selected_parent_id: str
    automatic_ordered_parent_ids: tuple[str, ...]
    before_ordered_parent_ids: tuple[str, ...]
    after_ordered_parent_ids: tuple[str, ...]
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class RenderLayoutWritingModeCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    automatic_writing_mode: str
    before_writing_mode: str
    after_writing_mode: str
    before_writing_mode_authority: str
    after_writing_mode_authority: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class RenderLayoutLineHeightCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    automatic_line_height: float
    before_line_height: float
    after_line_height: float
    before_line_height_authority: str
    after_line_height_authority: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class RenderLayoutRotationCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    automatic_rotation: float
    before_rotation: float
    after_rotation: float
    before_rotation_authority: str
    after_rotation_authority: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class RenderStyleFillColorCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    automatic_fill_color: str
    before_fill_color: str
    after_fill_color: str
    before_fill_color_authority: str
    after_fill_color_authority: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class RenderStyleOutlineColorCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    automatic_outline_color: str
    before_outline_color: str
    after_outline_color: str
    before_outline_color_authority: str
    after_outline_color_authority: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class RenderStyleOutlineWidthCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    automatic_outline_width: float
    before_outline_width: float
    after_outline_width: float
    before_outline_width_authority: str
    after_outline_width_authority: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class RenderStylePreferredSizeCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    automatic_preferred_size: float
    before_preferred_size: float
    after_preferred_size: float
    before_preferred_size_authority: str
    after_preferred_size_authority: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class RenderStyleShadowVisibilityCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    automatic_shadow_enabled: bool
    before_shadow_enabled: bool
    after_shadow_enabled: bool
    before_shadow_enabled_authority: str
    after_shadow_enabled_authority: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class RenderStyleShadowBlurCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    automatic_shadow_blur: float
    before_shadow_blur: float
    after_shadow_blur: float
    before_shadow_blur_authority: str
    after_shadow_blur_authority: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class RenderStyleFontRoleCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    automatic_font_role: str
    before_font_role: str
    after_font_role: str
    before_font_role_authority: str
    after_font_role_authority: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


@dataclass(frozen=True)
class EditHistoryCommandReceipt:
    command_id: str
    target_edit: ProjectEdit
    control_edit: ProjectEdit
    before_active: bool
    after_active: bool
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    before_issues: tuple[ProjectionIssue, ...]
    after_issues: tuple[ProjectionIssue, ...]
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


def _source_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise SourceTextCommandError(
            SourceTextCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise SourceTextCommandError(
            SourceTextCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is {reason}: {page_id}",
        )
    return matches[0]


def _source_effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = [parent for parent in snapshot.parents if parent.parent_id == parent_id]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise SourceTextCommandError(
            SourceTextCommandErrorCode.PARENT_NOT_FOUND,
            f"Effective parent identity is {reason}: {parent_id}",
        )
    return matches[0]


def _active_source_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    parent_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.SOURCE_TEXT
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(
        edit for edit in candidates if edit.edit_id not in superseded_ids
    )
    if len(heads) != 1:
        raise SourceTextCommandError(
            SourceTextCommandErrorCode.SOURCE_SLOT_CONFLICT,
            "Source text has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _source_slot_has_ancestor(
    ledger: ProjectEditLedger,
    *,
    descendant: ProjectEdit,
    ancestor_edit_id: str,
) -> bool:
    cursor: ProjectEdit | None = descendant
    visited: set[str] = set()
    while cursor is not None and cursor.edit_id not in visited:
        visited.add(cursor.edit_id)
        if cursor.edit_id == ancestor_edit_id:
            return True
        predecessor_id = cursor.supersedes_edit_id
        if predecessor_id is None:
            return False
        predecessor = ledger.get(predecessor_id)
        if (
            predecessor is None
            or predecessor.is_control
            or predecessor.page_id != descendant.page_id
            or predecessor.domain is not EditDomain.SOURCE_TEXT
            or predecessor.target != descendant.target
        ):
            return False
        cursor = predecessor
    return False


def _source_revision_base_artifact(
    parent: EffectiveParentSnapshot,
) -> OcrSourceRevisionArtifact:
    try:
        artifact = OcrSourceRevisionArtifact.from_record(
            dict(parent.source_revision_metadata)
        )
    except (TypeError, ValueError) as exc:
        raise SourceTextCommandError(
            SourceTextCommandErrorCode.REVISION_ARTIFACT_MISMATCH,
            "The selected model OCR artifact is unavailable.",
        ) from exc
    return artifact


def _require_revision_backed_source_state(
    *,
    snapshot: ProjectEditReadSnapshot,
    parent: EffectiveParentSnapshot,
    slot_head: ProjectEdit | None,
    command: SourceTextCommand,
) -> tuple[SourceTextRevisionBaseV1, OcrSourceRevisionArtifact]:
    revision_base = command.revision_base
    if revision_base is None:
        raise SourceTextCommandError(
            SourceTextCommandErrorCode.REVISION_BASE_REQUIRED,
            "The selected user parent requires its model OCR revision base.",
        )
    try:
        observed_base = source_text_revision_base_for_parent(parent)
    except (TypeError, ValueError) as exc:
        raise SourceTextCommandError(
            SourceTextCommandErrorCode.REVISION_ARTIFACT_MISMATCH,
            "The selected model OCR revision is inconsistent.",
        ) from exc
    if observed_base != revision_base:
        raise SourceTextCommandError(
            SourceTextCommandErrorCode.REVISION_ID_MISMATCH,
            "The selected model OCR revision changed.",
        )
    artifact = _source_revision_base_artifact(parent)
    if (
        artifact.revision_id != revision_base.source_revision_id
        or artifact.page_id != command.page_id
        or artifact.parent_id != command.parent_id
        or canonical_sha256(artifact.to_record())
        != revision_base.artifact_sha256
        or effective_source_fingerprint(
            artifact.parent_id,
            artifact.source_text,
        )
        != revision_base.source_fingerprint
    ):
        raise SourceTextCommandError(
            SourceTextCommandErrorCode.REVISION_ARTIFACT_MISMATCH,
            "The selected model OCR artifact binding changed.",
        )
    selection_edit = snapshot.ledger.get(revision_base.selection_edit_id)
    if (
        slot_head is None
        or selection_edit is None
        or selection_edit.domain is not EditDomain.SOURCE_TEXT
        or selection_edit.operation != "select_revision"
        or selection_edit.page_id != command.page_id
        or selection_edit.target.kind is not EditTargetKind.PARENT
        or selection_edit.target.parent_id != command.parent_id
        or str(selection_edit.payload.get("revision_id") or "")
        != revision_base.source_revision_id
        or not _source_slot_has_ancestor(
            snapshot.ledger,
            descendant=slot_head,
            ancestor_edit_id=revision_base.selection_edit_id,
        )
    ):
        raise SourceTextCommandError(
            SourceTextCommandErrorCode.REVISION_SELECTION_MISMATCH,
            "The selected model OCR edit lineage changed.",
        )
    lineage = parent.lineage
    if (
        parent.origin is not ParentOrigin.USER
        or lineage is None
        or artifact.root_id != parent.root_id
        or artifact.parent_authored_edit_id != lineage.authored_edit_id
    ):
        raise SourceTextCommandError(
            SourceTextCommandErrorCode.PARENT_LINEAGE_MISMATCH,
            "The selected user-parent lineage changed.",
        )
    return revision_base, artifact


def _project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is {reason}: {page_id}",
        )
    return matches[0]


def _effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = [parent for parent in snapshot.parents if parent.parent_id == parent_id]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.PARENT_NOT_FOUND,
            f"Effective parent identity is {reason}: {parent_id}",
        )
    return matches[0]


def _active_target_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    parent_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.TARGET_TEXT
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(
        edit for edit in candidates if edit.edit_id not in superseded_ids
    )
    if len(heads) != 1:
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.TARGET_SLOT_CONFLICT,
            "Target text has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _target_slot_has_ancestor(
    ledger: ProjectEditLedger,
    *,
    descendant: ProjectEdit,
    ancestor_edit_id: str,
) -> bool:
    cursor: ProjectEdit | None = descendant
    visited: set[str] = set()
    while cursor is not None and cursor.edit_id not in visited:
        visited.add(cursor.edit_id)
        if cursor.edit_id == ancestor_edit_id:
            return True
        predecessor_id = cursor.supersedes_edit_id
        if predecessor_id is None:
            return False
        predecessor = ledger.get(predecessor_id)
        if (
            predecessor is None
            or predecessor.is_control
            or predecessor.page_id != descendant.page_id
            or predecessor.domain is not EditDomain.TARGET_TEXT
            or predecessor.target != descendant.target
        ):
            return False
        cursor = predecessor
    return False


def _revision_base_artifact(
    parent: EffectiveParentSnapshot,
) -> TranslationRevisionArtifact:
    try:
        artifact = TranslationRevisionArtifact.from_record(
            dict(parent.target_revision_metadata)
        )
    except (TypeError, ValueError) as exc:
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.REVISION_ARTIFACT_MISMATCH,
            "The selected model translation artifact is unavailable.",
        ) from exc
    return artifact


def _require_revision_backed_target_state(
    *,
    snapshot: ProjectEditReadSnapshot,
    effective_page: EffectivePageSnapshot,
    parent: EffectiveParentSnapshot,
    slot_head: ProjectEdit | None,
    command: TargetTextCommand,
) -> tuple[TargetTextRevisionBaseV1, TranslationRevisionArtifact]:
    revision_base = command.revision_base
    if revision_base is None:
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.REVISION_BASE_REQUIRED,
            "The selected user parent requires its model translation revision base.",
        )
    if parent.target_revision_id != revision_base.translation_revision_id:
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.REVISION_ID_MISMATCH,
            "The selected model translation revision changed.",
        )
    artifact = _revision_base_artifact(parent)
    if (
        artifact.revision_id != revision_base.translation_revision_id
        or artifact.page_id != command.page_id
        or artifact.parent_id != command.parent_id
    ):
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.REVISION_ID_MISMATCH,
            "The selected model translation revision identity changed.",
        )
    if canonical_sha256(artifact.to_record()) != revision_base.artifact_sha256:
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.REVISION_ARTIFACT_MISMATCH,
            "The selected model translation artifact hash changed.",
        )
    source_fingerprint = effective_source_fingerprint(
        parent.parent_id,
        parent.source_text,
    )
    try:
        source_slot_head = _active_source_slot_head(
            snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
    except SourceTextCommandError as exc:
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.REVISION_SOURCE_MISMATCH,
            "The selected model translation source slot is conflicted.",
        ) from exc
    if (
        source_fingerprint != revision_base.source_fingerprint
        or artifact.source_fingerprint != revision_base.source_fingerprint
        or artifact.source_text != parent.source_text
        or artifact.source_authority != parent.source_authority
        or artifact.source_revision_id != revision_base.source_revision_id
        or parent.source_revision_id != revision_base.source_revision_id
        or artifact.source_selection_edit_id
        != revision_base.source_selection_edit_id
        or source_slot_head is None
        or not _source_slot_has_ancestor(
            snapshot.ledger,
            descendant=source_slot_head,
            ancestor_edit_id=revision_base.source_selection_edit_id,
        )
    ):
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.REVISION_SOURCE_MISMATCH,
            "The selected model translation source binding changed.",
        )
    if (
        artifact.hierarchy_revision_id
        != revision_base.hierarchy_revision_id
        or artifact.hierarchy_fingerprint
        != revision_base.hierarchy_fingerprint
    ):
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.REVISION_HIERARCHY_MISMATCH,
            "The selected model translation hierarchy binding changed.",
        )
    lineage = parent.lineage
    authored_edit = (
        snapshot.ledger.get(lineage.authored_edit_id)
        if lineage is not None
        else None
    )
    if (
        parent.origin is not ParentOrigin.USER
        or lineage is None
        or artifact.root_id != parent.root_id
        or artifact.parent_authored_edit_id != lineage.authored_edit_id
        or authored_edit is None
        or authored_edit.domain is not EditDomain.STRUCTURAL
        or authored_edit.operation != AddUserParentOperation.ADD.value
        or authored_edit.target.kind is not EditTargetKind.PARENT
        or authored_edit.target.parent_id != command.parent_id
        or str(authored_edit.payload.get("root_id") or "") != parent.root_id
    ):
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.PARENT_LINEAGE_MISMATCH,
            "The selected user-parent Add lineage changed.",
        )
    if (
        artifact.selection_edit_id != revision_base.selection_edit_id
        or slot_head is None
    ):
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.REVISION_SELECTION_MISMATCH,
            "The selected model translation edit lineage changed.",
        )
    selection_edit = snapshot.ledger.get(revision_base.selection_edit_id)
    if (
        selection_edit is None
        or selection_edit.domain is not EditDomain.TARGET_TEXT
        or selection_edit.operation != "select_revision"
        or selection_edit.page_id != command.page_id
        or selection_edit.target.kind is not EditTargetKind.PARENT
        or selection_edit.target.parent_id != command.parent_id
        or str(selection_edit.payload.get("revision_id") or "")
        != revision_base.translation_revision_id
        or str(selection_edit.payload.get("source_fingerprint") or "")
        != revision_base.source_fingerprint
        or not _target_slot_has_ancestor(
            snapshot.ledger,
            descendant=slot_head,
            ancestor_edit_id=revision_base.selection_edit_id,
        )
    ):
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.REVISION_SELECTION_MISMATCH,
            "The original model translation selection is not the current slot ancestor.",
        )
    try:
        observed_base = target_text_revision_base_for_parent(parent)
    except (TypeError, ValueError) as exc:
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.REVISION_ARTIFACT_MISMATCH,
            "The selected model translation base is invalid.",
        ) from exc
    if observed_base != revision_base:
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.REVISION_ARTIFACT_MISMATCH,
            "The selected model translation base changed.",
        )
    return revision_base, artifact


def _require_mapped_target_state(
    *,
    snapshot: ProjectEditReadSnapshot,
    parent: EffectiveParentSnapshot,
    slot_head: ProjectEdit | None,
    command: TargetTextCommand,
) -> ParentSourceEvidenceMappingV1:
    source_evidence_base = command.source_evidence_base
    if source_evidence_base is None:
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.MAPPED_BASE_REQUIRED,
            "The selected user parent requires its mapped pipeline evidence base.",
        )
    if parent.source_evidence_mapping != source_evidence_base:
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.MAPPED_BASE_MISMATCH,
            "The selected parent detection/OCR evidence mapping changed.",
        )
    if (
        source_evidence_base.page_id != command.page_id
        or source_evidence_base.target_text is None
        or parent.source_text != source_evidence_base.source_text
        or effective_source_fingerprint(parent.parent_id, parent.source_text)
        != effective_source_fingerprint(
            command.parent_id,
            source_evidence_base.source_text,
        )
    ):
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.MAPPED_BASE_MISMATCH,
            "The selected parent mapped source or target evidence changed.",
        )
    lineage = parent.lineage
    authored_edit = (
        snapshot.ledger.get(lineage.authored_edit_id)
        if lineage is not None
        else None
    )
    merge_lineage_matches = bool(
        authored_edit is not None
        and authored_edit.operation
        == MergePipelineParentsOperation.MERGE.value
        and authored_edit.target.kind is EditTargetKind.PARENT
        and authored_edit.target.parent_id == command.parent_id
        and str(authored_edit.payload.get("merged_root_id") or "")
        == parent.root_id
        and tuple(
            str(value)
            for value in authored_edit.payload.get("source_parent_ids") or ()
        )
        == source_evidence_base.source_parent_ids
        and tuple(
            str(value)
            for value in authored_edit.payload.get("source_root_ids") or ()
        )
        == source_evidence_base.source_root_ids
        and str(authored_edit.payload.get("merged_source_text") or "")
        == source_evidence_base.source_text
    )
    split_lineage_matches = False
    if (
        authored_edit is not None
        and authored_edit.operation == SplitUserParentOperation.SPLIT.value
        and authored_edit.target.kind is EditTargetKind.PARENT
    ):
        child_parent_ids = tuple(
            str(value)
            for value in authored_edit.payload.get("child_parent_ids") or ()
        )
        child_root_ids = tuple(
            str(value)
            for value in authored_edit.payload.get("child_root_ids") or ()
        )
        child_mapping_values = tuple(
            authored_edit.payload.get("child_source_evidence_mappings") or ()
        )
        if (
            child_parent_ids.count(command.parent_id) == 1
            and len(child_parent_ids) == len(child_root_ids)
            and len(child_parent_ids) == len(child_mapping_values)
        ):
            child_index = child_parent_ids.index(command.parent_id)
            try:
                split_mapping = ParentSourceEvidenceMappingV1.from_dict(
                    child_mapping_values[child_index]
                )
            except (TypeError, ValueError):
                split_mapping = None
            split_lineage_matches = bool(
                child_root_ids[child_index] == parent.root_id
                and split_mapping == source_evidence_base
            )
    if (
        parent.origin is not ParentOrigin.USER
        or lineage is None
        or authored_edit is None
        or authored_edit.is_control
        or authored_edit.domain is not EditDomain.STRUCTURAL
        or authored_edit.operation
        not in {
            MergePipelineParentsOperation.MERGE.value,
            SplitUserParentOperation.SPLIT.value,
        }
        or authored_edit.page_id != command.page_id
        or lineage.parent_id != command.parent_id
        or lineage.root_id != parent.root_id
        or not (merge_lineage_matches or split_lineage_matches)
    ):
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.PARENT_LINEAGE_MISMATCH,
            "The selected mapped parent topology lineage changed.",
        )
    if slot_head is not None:
        try:
            slot_base_value = slot_head.payload.get("source_evidence_base")
            slot_base = ParentSourceEvidenceMappingV1.from_dict(slot_base_value)
        except (TypeError, ValueError) as exc:
            raise TargetTextCommandError(
                TargetTextCommandErrorCode.MAPPED_BASE_MISMATCH,
                "The active target edit does not retain mapped pipeline evidence.",
            ) from exc
        if slot_base != source_evidence_base:
            raise TargetTextCommandError(
                TargetTextCommandErrorCode.MAPPED_BASE_MISMATCH,
                "The active target edit belongs to a different mapped pipeline base.",
            )
    return source_evidence_base


def _membership_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise ParentMembershipCommandError(
            ParentMembershipCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise ParentMembershipCommandError(
            ParentMembershipCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is {reason}: {page_id}",
        )
    return matches[0]


def _membership_effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = [parent for parent in snapshot.parents if parent.parent_id == parent_id]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise ParentMembershipCommandError(
            ParentMembershipCommandErrorCode.PARENT_NOT_FOUND,
            f"Effective parent identity is {reason}: {parent_id}",
        )
    return matches[0]


def _active_membership_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    parent_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.STRUCTURAL
        and edit.operation in {
            ParentMembershipOperation.EXCLUDE.value,
            ParentMembershipOperation.RESTORE.value,
        }
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(
        edit for edit in candidates if edit.edit_id not in superseded_ids
    )
    if len(heads) != 1:
        raise ParentMembershipCommandError(
            ParentMembershipCommandErrorCode.MEMBERSHIP_SLOT_CONFLICT,
            "Parent membership has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _geometry_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise ParentGeometryCommandError(
            ParentGeometryCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise ParentGeometryCommandError(
            ParentGeometryCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is {reason}: {page_id}",
        )
    return matches[0]


def _geometry_effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = [parent for parent in snapshot.parents if parent.parent_id == parent_id]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise ParentGeometryCommandError(
            ParentGeometryCommandErrorCode.PARENT_NOT_FOUND,
            f"Effective parent identity is {reason}: {parent_id}",
        )
    return matches[0]


def _active_geometry_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    parent_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.STRUCTURAL
        and edit.operation == ParentGeometryOperation.SET_GEOMETRY.value
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(
        edit for edit in candidates if edit.edit_id not in superseded_ids
    )
    if len(heads) != 1:
        raise ParentGeometryCommandError(
            ParentGeometryCommandErrorCode.GEOMETRY_SLOT_CONFLICT,
            "Parent geometry has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _reading_order_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = tuple(
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    )
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is {reason}: {page_id}",
        )
    return matches[0]


def _active_reading_order_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.STRUCTURAL
        and edit.operation == ReadingOrderOperation.SET.value
        and edit.target.kind is EditTargetKind.PAGE
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(
        edit for edit in candidates if edit.edit_id not in superseded_ids
    )
    if len(heads) != 1:
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.READING_ORDER_SLOT_CONFLICT,
            "Reading order has competing active page edits; resolve the conflict first.",
        )
    return heads[0]


def _require_exact_reading_order_invalidation(
    invalidation: InvalidationResult,
    *,
    page_id: str,
) -> None:
    actual = tuple(
        (
            effect.dependency,
            effect.action,
            effect.scope,
            effect.target_ids,
            effect.reason,
        )
        for effect in invalidation.effects
    )
    expected = tuple(
        sorted(
            (
                (
                    Dependency.HIERARCHY,
                    InvalidationAction.RECOMPUTE,
                    InvalidationScope.PAGE,
                    (page_id,),
                    "effective_reading_order_changed",
                ),
                (
                    Dependency.LAYOUT_RENDER,
                    InvalidationAction.RECOMPUTE,
                    InvalidationScope.PAGE,
                    (page_id,),
                    "effective_reading_order_changed",
                ),
                (
                    Dependency.PAGE_OUTPUT,
                    InvalidationAction.RECOMPUTE,
                    InvalidationScope.PAGE,
                    (page_id,),
                    "effective_reading_order_changed",
                ),
            ),
            key=lambda item: (item[0].value, item[2].value, item[3], item[1].value, item[4]),
        )
    )
    if invalidation.unresolved_facts or actual != expected:
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Reading-order invalidation must affect only hierarchy, page layout order, and page output.",
        )


def _writing_mode_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise RenderLayoutWritingModeCommandError(
            RenderLayoutWritingModeCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderLayoutWritingModeCommandError(
            RenderLayoutWritingModeCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is {reason}: {page_id}",
        )
    return matches[0]


def _writing_mode_automatic_parent(
    page: Mapping[str, Any],
    *,
    page_id: str,
    parent_id: str,
) -> Mapping[str, Any]:
    parents = page.get("parent_execution_bundles") or ()
    if not isinstance(parents, (list, tuple)):
        raise RenderLayoutWritingModeCommandError(
            RenderLayoutWritingModeCommandErrorCode.PARENT_NOT_FOUND,
            "Automatic parent records are unavailable.",
        )
    matches = [
        parent
        for parent in parents
        if isinstance(parent, Mapping)
        and str(parent.get("parent_id") or "").strip() == parent_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderLayoutWritingModeCommandError(
            RenderLayoutWritingModeCommandErrorCode.PARENT_NOT_FOUND,
            f"Automatic parent identity is {reason}: {parent_id}",
        )
    parent = matches[0]
    if str(parent.get("page_id") or "").strip() != page_id:
        raise RenderLayoutWritingModeCommandError(
            RenderLayoutWritingModeCommandErrorCode.PARENT_NOT_FOUND,
            "Automatic parent page identity does not match the command.",
        )
    return parent


def _writing_mode_effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = [parent for parent in snapshot.parents if parent.parent_id == parent_id]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderLayoutWritingModeCommandError(
            RenderLayoutWritingModeCommandErrorCode.PARENT_NOT_FOUND,
            f"Effective parent identity is {reason}: {parent_id}",
        )
    return matches[0]


def _writing_mode_edit_field(edit: ProjectEdit) -> str | None:
    if edit.domain is not EditDomain.RENDER_LAYOUT:
        return None
    payload = thaw_json(edit.payload)
    if not isinstance(payload, Mapping):
        return None
    fields = payload.get("fields")
    if isinstance(fields, Mapping) and len(fields) == 1:
        return str(next(iter(fields)))
    if (
        isinstance(fields, (list, tuple))
        and len(fields) == 1
        and isinstance(fields[0], str)
    ):
        return fields[0]
    return None


def _active_writing_mode_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    parent_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.RENDER_LAYOUT
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
        and _writing_mode_edit_field(edit) == "writing_mode"
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(
        edit for edit in candidates if edit.edit_id not in superseded_ids
    )
    if len(heads) != 1:
        raise RenderLayoutWritingModeCommandError(
            RenderLayoutWritingModeCommandErrorCode.WRITING_MODE_SLOT_CONFLICT,
            "Writing mode has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _effective_writing_mode_state(
    parent: EffectiveParentSnapshot,
    *,
    automatic_writing_mode: str,
) -> tuple[str, str]:
    overrides = dict(parent.render_layout_overrides)
    if "writing_mode" not in overrides:
        return automatic_writing_mode, "automatic"
    value = overrides["writing_mode"]
    if not isinstance(value, str) or value not in CANONICAL_WRITING_MODES:
        raise RenderLayoutWritingModeCommandError(
            RenderLayoutWritingModeCommandErrorCode.PROJECTION_REJECTED,
            "The effective projector produced a noncanonical writing mode.",
        )
    return value, "user"


def _require_exact_writing_mode_invalidation(
    invalidation: InvalidationResult,
    *,
    parent_id: str,
) -> None:
    actual = tuple(
        (
            effect.dependency,
            effect.action,
            effect.scope,
            effect.target_ids,
            effect.reason,
        )
        for effect in invalidation.effects
    )
    expected = (
        (
            Dependency.LAYOUT_RENDER,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PARENT,
            (parent_id,),
            "render_layout_override",
        ),
        (
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            (parent_id,),
            "render_layout_override",
        ),
    )
    if invalidation.unresolved_facts or actual != expected:
        raise RenderLayoutWritingModeCommandError(
            RenderLayoutWritingModeCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Writing-mode invalidation must affect only parent layout and page output.",
        )


def _line_height_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise RenderLayoutLineHeightCommandError(
            RenderLayoutLineHeightCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderLayoutLineHeightCommandError(
            RenderLayoutLineHeightCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is {reason}: {page_id}",
        )
    return matches[0]


def _line_height_automatic_parent(
    page: Mapping[str, Any],
    *,
    page_id: str,
    parent_id: str,
) -> Mapping[str, Any]:
    parents = page.get("parent_execution_bundles") or ()
    if not isinstance(parents, (list, tuple)):
        raise RenderLayoutLineHeightCommandError(
            RenderLayoutLineHeightCommandErrorCode.PARENT_NOT_FOUND,
            "Automatic parent records are unavailable.",
        )
    matches = [
        parent
        for parent in parents
        if isinstance(parent, Mapping)
        and str(parent.get("parent_id") or "").strip() == parent_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderLayoutLineHeightCommandError(
            RenderLayoutLineHeightCommandErrorCode.PARENT_NOT_FOUND,
            f"Automatic parent identity is {reason}: {parent_id}",
        )
    parent = matches[0]
    if str(parent.get("page_id") or "").strip() != page_id:
        raise RenderLayoutLineHeightCommandError(
            RenderLayoutLineHeightCommandErrorCode.PARENT_NOT_FOUND,
            "Automatic parent page identity does not match the command.",
        )
    return parent


def _line_height_effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = [parent for parent in snapshot.parents if parent.parent_id == parent_id]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderLayoutLineHeightCommandError(
            RenderLayoutLineHeightCommandErrorCode.PARENT_NOT_FOUND,
            f"Effective parent identity is {reason}: {parent_id}",
        )
    return matches[0]


def _active_line_height_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    parent_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.RENDER_LAYOUT
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
        and _writing_mode_edit_field(edit) == "line_height"
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(
        edit for edit in candidates if edit.edit_id not in superseded_ids
    )
    if len(heads) != 1:
        raise RenderLayoutLineHeightCommandError(
            RenderLayoutLineHeightCommandErrorCode.LINE_HEIGHT_SLOT_CONFLICT,
            "Line height has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _effective_line_height_state(
    parent: EffectiveParentSnapshot,
    *,
    automatic_line_height: float,
) -> tuple[float, str]:
    overrides = dict(parent.render_layout_overrides)
    if "line_height" not in overrides:
        return automatic_line_height, "automatic"
    try:
        value = canonical_render_line_height(overrides["line_height"])
    except (TypeError, ValueError) as exc:
        raise RenderLayoutLineHeightCommandError(
            RenderLayoutLineHeightCommandErrorCode.PROJECTION_REJECTED,
            "The effective projector produced an invalid line height.",
        ) from exc
    return value, "user"


def _require_exact_line_height_invalidation(
    invalidation: InvalidationResult,
    *,
    parent_id: str,
) -> None:
    actual = tuple(
        (
            effect.dependency,
            effect.action,
            effect.scope,
            effect.target_ids,
            effect.reason,
        )
        for effect in invalidation.effects
    )
    expected = (
        (
            Dependency.LAYOUT_RENDER,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PARENT,
            (parent_id,),
            "render_layout_override",
        ),
        (
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            (parent_id,),
            "render_layout_override",
        ),
    )
    if invalidation.unresolved_facts or actual != expected:
        raise RenderLayoutLineHeightCommandError(
            RenderLayoutLineHeightCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Line-height invalidation must affect only parent layout and page output.",
        )


def _rotation_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise RenderLayoutRotationCommandError(
            RenderLayoutRotationCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderLayoutRotationCommandError(
            RenderLayoutRotationCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is {reason}: {page_id}",
        )
    return matches[0]


def _rotation_automatic_parent(
    page: Mapping[str, Any],
    *,
    page_id: str,
    parent_id: str,
) -> Mapping[str, Any]:
    parents = page.get("parent_execution_bundles") or ()
    if not isinstance(parents, (list, tuple)):
        raise RenderLayoutRotationCommandError(
            RenderLayoutRotationCommandErrorCode.PARENT_NOT_FOUND,
            "Automatic parent records are unavailable.",
        )
    matches = [
        parent
        for parent in parents
        if isinstance(parent, Mapping)
        and str(parent.get("parent_id") or "").strip() == parent_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderLayoutRotationCommandError(
            RenderLayoutRotationCommandErrorCode.PARENT_NOT_FOUND,
            f"Automatic parent identity is {reason}: {parent_id}",
        )
    parent = matches[0]
    if str(parent.get("page_id") or "").strip() != page_id:
        raise RenderLayoutRotationCommandError(
            RenderLayoutRotationCommandErrorCode.PARENT_NOT_FOUND,
            "Automatic parent page identity does not match the command.",
        )
    return parent


def _rotation_effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = [parent for parent in snapshot.parents if parent.parent_id == parent_id]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderLayoutRotationCommandError(
            RenderLayoutRotationCommandErrorCode.PARENT_NOT_FOUND,
            f"Effective parent identity is {reason}: {parent_id}",
        )
    return matches[0]


def _active_rotation_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    parent_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.RENDER_LAYOUT
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
        and _writing_mode_edit_field(edit) == "rotation"
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(
        edit for edit in candidates if edit.edit_id not in superseded_ids
    )
    if len(heads) != 1:
        raise RenderLayoutRotationCommandError(
            RenderLayoutRotationCommandErrorCode.ROTATION_SLOT_CONFLICT,
            "Rotation has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _effective_rotation_state(
    parent: EffectiveParentSnapshot,
    *,
    automatic_rotation: float,
) -> tuple[float, str]:
    overrides = dict(parent.render_layout_overrides)
    if "rotation" not in overrides:
        return automatic_rotation, "automatic"
    try:
        value = canonical_render_rotation(overrides["rotation"])
    except (TypeError, ValueError) as exc:
        raise RenderLayoutRotationCommandError(
            RenderLayoutRotationCommandErrorCode.PROJECTION_REJECTED,
            "The effective projector produced an invalid rotation.",
        ) from exc
    return value, "user"


def _require_exact_rotation_invalidation(
    invalidation: InvalidationResult,
    *,
    parent_id: str,
) -> None:
    actual = tuple(
        (
            effect.dependency,
            effect.action,
            effect.scope,
            effect.target_ids,
            effect.reason,
        )
        for effect in invalidation.effects
    )
    expected = (
        (
            Dependency.LAYOUT_RENDER,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PARENT,
            (parent_id,),
            "render_layout_override",
        ),
        (
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            (parent_id,),
            "render_layout_override",
        ),
    )
    if invalidation.unresolved_facts or actual != expected:
        raise RenderLayoutRotationCommandError(
            RenderLayoutRotationCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Rotation invalidation must affect only parent layout and page output.",
        )


def _fill_color_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise RenderStyleFillColorCommandError(
            RenderStyleFillColorCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderStyleFillColorCommandError(
            RenderStyleFillColorCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is {reason}: {page_id}",
        )
    return matches[0]


def _fill_color_automatic_parent(
    page: Mapping[str, Any],
    *,
    page_id: str,
    parent_id: str,
) -> Mapping[str, Any]:
    parents = page.get("parent_execution_bundles") or ()
    if not isinstance(parents, (list, tuple)):
        raise RenderStyleFillColorCommandError(
            RenderStyleFillColorCommandErrorCode.PARENT_NOT_FOUND,
            "Automatic parent records are unavailable.",
        )
    matches = [
        parent
        for parent in parents
        if isinstance(parent, Mapping)
        and str(parent.get("parent_id") or "").strip() == parent_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderStyleFillColorCommandError(
            RenderStyleFillColorCommandErrorCode.PARENT_NOT_FOUND,
            f"Automatic parent identity is {reason}: {parent_id}",
        )
    parent = matches[0]
    if str(parent.get("page_id") or "").strip() != page_id:
        raise RenderStyleFillColorCommandError(
            RenderStyleFillColorCommandErrorCode.PARENT_NOT_FOUND,
            "Automatic parent page identity does not match the command.",
        )
    return parent


def _fill_color_effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = [parent for parent in snapshot.parents if parent.parent_id == parent_id]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderStyleFillColorCommandError(
            RenderStyleFillColorCommandErrorCode.PARENT_NOT_FOUND,
            f"Effective parent identity is {reason}: {parent_id}",
        )
    return matches[0]


def _render_style_edit_field(edit: ProjectEdit) -> str | None:
    if edit.domain is not EditDomain.RENDER_STYLE:
        return None
    payload = thaw_json(edit.payload)
    if not isinstance(payload, Mapping):
        return None
    fields = payload.get("fields")
    if isinstance(fields, Mapping) and len(fields) == 1:
        return str(next(iter(fields)))
    if (
        isinstance(fields, (list, tuple))
        and len(fields) == 1
        and isinstance(fields[0], str)
    ):
        return fields[0]
    return None


def _active_fill_color_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    parent_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.RENDER_STYLE
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
        and _render_style_edit_field(edit) == "fill_color"
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(
        edit for edit in candidates if edit.edit_id not in superseded_ids
    )
    if len(heads) != 1:
        raise RenderStyleFillColorCommandError(
            RenderStyleFillColorCommandErrorCode.FILL_COLOR_SLOT_CONFLICT,
            "Fill color has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _effective_fill_color_state(
    parent: EffectiveParentSnapshot,
    *,
    automatic_fill_color: str,
) -> tuple[str, str]:
    overrides = dict(parent.render_style_overrides)
    if "fill_color" not in overrides:
        return automatic_fill_color, "automatic"
    try:
        value = canonical_render_fill_color(overrides["fill_color"])
    except (TypeError, ValueError) as exc:
        raise RenderStyleFillColorCommandError(
            RenderStyleFillColorCommandErrorCode.PROJECTION_REJECTED,
            "The effective projector produced an invalid fill color.",
        ) from exc
    return value, "user"


def _require_exact_fill_color_invalidation(
    invalidation: InvalidationResult,
    *,
    parent_id: str,
) -> None:
    actual = tuple(
        (
            effect.dependency,
            effect.action,
            effect.scope,
            effect.target_ids,
            effect.reason,
        )
        for effect in invalidation.effects
    )
    expected = (
        (
            Dependency.LAYOUT_RENDER,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PARENT,
            (parent_id,),
            "render_style_override",
        ),
        (
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            (parent_id,),
            "render_style_override",
        ),
    )
    if invalidation.unresolved_facts or actual != expected:
        raise RenderStyleFillColorCommandError(
            RenderStyleFillColorCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Fill-color invalidation must affect only parent layout and page output.",
        )


def _outline_color_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise RenderStyleOutlineColorCommandError(
            RenderStyleOutlineColorCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderStyleOutlineColorCommandError(
            RenderStyleOutlineColorCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is {reason}: {page_id}",
        )
    return matches[0]


def _outline_color_automatic_parent(
    page: Mapping[str, Any],
    *,
    page_id: str,
    parent_id: str,
) -> Mapping[str, Any]:
    parents = page.get("parent_execution_bundles") or ()
    if not isinstance(parents, (list, tuple)):
        raise RenderStyleOutlineColorCommandError(
            RenderStyleOutlineColorCommandErrorCode.PARENT_NOT_FOUND,
            "Automatic parent records are unavailable.",
        )
    matches = [
        parent
        for parent in parents
        if isinstance(parent, Mapping)
        and str(parent.get("parent_id") or "").strip() == parent_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderStyleOutlineColorCommandError(
            RenderStyleOutlineColorCommandErrorCode.PARENT_NOT_FOUND,
            f"Automatic parent identity is {reason}: {parent_id}",
        )
    parent = matches[0]
    if str(parent.get("page_id") or "").strip() != page_id:
        raise RenderStyleOutlineColorCommandError(
            RenderStyleOutlineColorCommandErrorCode.PARENT_NOT_FOUND,
            "Automatic parent page identity does not match the command.",
        )
    return parent


def _outline_color_effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = [parent for parent in snapshot.parents if parent.parent_id == parent_id]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderStyleOutlineColorCommandError(
            RenderStyleOutlineColorCommandErrorCode.PARENT_NOT_FOUND,
            f"Effective parent identity is {reason}: {parent_id}",
        )
    return matches[0]


def _render_style_edit_field(edit: ProjectEdit) -> str | None:
    if edit.domain is not EditDomain.RENDER_STYLE:
        return None
    payload = thaw_json(edit.payload)
    if not isinstance(payload, Mapping):
        return None
    fields = payload.get("fields")
    if isinstance(fields, Mapping) and len(fields) == 1:
        return str(next(iter(fields)))
    if (
        isinstance(fields, (list, tuple))
        and len(fields) == 1
        and isinstance(fields[0], str)
    ):
        return fields[0]
    return None


def _active_outline_color_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    parent_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.RENDER_STYLE
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
        and _render_style_edit_field(edit) == "outline_color"
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(
        edit for edit in candidates if edit.edit_id not in superseded_ids
    )
    if len(heads) != 1:
        raise RenderStyleOutlineColorCommandError(
            RenderStyleOutlineColorCommandErrorCode.OUTLINE_COLOR_SLOT_CONFLICT,
            "Outline color has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _effective_outline_color_state(
    parent: EffectiveParentSnapshot,
    *,
    automatic_outline_color: str,
) -> tuple[str, str]:
    overrides = dict(parent.render_style_overrides)
    if "outline_color" not in overrides:
        return automatic_outline_color, "automatic"
    try:
        value = canonical_render_outline_color(overrides["outline_color"])
    except (TypeError, ValueError) as exc:
        raise RenderStyleOutlineColorCommandError(
            RenderStyleOutlineColorCommandErrorCode.PROJECTION_REJECTED,
            "The effective projector produced an invalid outline color.",
        ) from exc
    return value, "user"


def _require_exact_outline_color_invalidation(
    invalidation: InvalidationResult,
    *,
    parent_id: str,
) -> None:
    actual = tuple(
        (
            effect.dependency,
            effect.action,
            effect.scope,
            effect.target_ids,
            effect.reason,
        )
        for effect in invalidation.effects
    )
    expected = (
        (
            Dependency.LAYOUT_RENDER,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PARENT,
            (parent_id,),
            "render_style_override",
        ),
        (
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            (parent_id,),
            "render_style_override",
        ),
    )
    if invalidation.unresolved_facts or actual != expected:
        raise RenderStyleOutlineColorCommandError(
            RenderStyleOutlineColorCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Outline-color invalidation must affect only parent layout and page output.",
        )


def _outline_width_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    try:
        return _outline_color_project_page(project, page_id)
    except RenderStyleOutlineColorCommandError as exc:
        raise RenderStyleOutlineWidthCommandError(
            RenderStyleOutlineWidthCommandErrorCode(exc.code.value),
            str(exc),
        ) from exc


def _outline_width_automatic_parent(
    page: Mapping[str, Any],
    *,
    page_id: str,
    parent_id: str,
) -> Mapping[str, Any]:
    try:
        return _outline_color_automatic_parent(
            page,
            page_id=page_id,
            parent_id=parent_id,
        )
    except RenderStyleOutlineColorCommandError as exc:
        raise RenderStyleOutlineWidthCommandError(
            RenderStyleOutlineWidthCommandErrorCode(exc.code.value),
            str(exc),
        ) from exc


def _outline_width_effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    try:
        return _outline_color_effective_parent(snapshot, parent_id)
    except RenderStyleOutlineColorCommandError as exc:
        raise RenderStyleOutlineWidthCommandError(
            RenderStyleOutlineWidthCommandErrorCode(exc.code.value),
            str(exc),
        ) from exc


def _active_outline_width_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    parent_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.RENDER_STYLE
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
        and _render_style_edit_field(edit) == "outline_width"
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(
        edit for edit in candidates if edit.edit_id not in superseded_ids
    )
    if len(heads) != 1:
        raise RenderStyleOutlineWidthCommandError(
            RenderStyleOutlineWidthCommandErrorCode.OUTLINE_WIDTH_SLOT_CONFLICT,
            "Outline width has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _effective_outline_width_state(
    parent: EffectiveParentSnapshot,
    *,
    automatic_outline_width: float,
) -> tuple[float, str]:
    overrides = dict(parent.render_style_overrides)
    if "outline_width" not in overrides:
        return automatic_outline_width, "automatic"
    try:
        value = canonical_render_outline_width(overrides["outline_width"])
    except (TypeError, ValueError) as exc:
        raise RenderStyleOutlineWidthCommandError(
            RenderStyleOutlineWidthCommandErrorCode.PROJECTION_REJECTED,
            "The effective projector produced an invalid outline width.",
        ) from exc
    return value, "user"


def _require_exact_outline_width_invalidation(
    invalidation: InvalidationResult,
    *,
    parent_id: str,
) -> None:
    actual = tuple(
        (
            effect.dependency,
            effect.action,
            effect.scope,
            effect.target_ids,
            effect.reason,
        )
        for effect in invalidation.effects
    )
    expected = (
        (
            Dependency.LAYOUT_RENDER,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PARENT,
            (parent_id,),
            "render_style_override",
        ),
        (
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            (parent_id,),
            "render_style_override",
        ),
    )
    if invalidation.unresolved_facts or actual != expected:
        raise RenderStyleOutlineWidthCommandError(
            RenderStyleOutlineWidthCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Outline-width invalidation must affect only parent layout and page output.",
        )


def _preferred_size_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    try:
        return _outline_color_project_page(project, page_id)
    except RenderStyleOutlineColorCommandError as exc:
        raise RenderStylePreferredSizeCommandError(
            RenderStylePreferredSizeCommandErrorCode(exc.code.value),
            str(exc),
        ) from exc


def _preferred_size_automatic_parent(
    page: Mapping[str, Any],
    *,
    page_id: str,
    parent_id: str,
) -> Mapping[str, Any]:
    try:
        return _outline_color_automatic_parent(
            page,
            page_id=page_id,
            parent_id=parent_id,
        )
    except RenderStyleOutlineColorCommandError as exc:
        raise RenderStylePreferredSizeCommandError(
            RenderStylePreferredSizeCommandErrorCode(exc.code.value),
            str(exc),
        ) from exc


def _preferred_size_effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    try:
        return _outline_color_effective_parent(snapshot, parent_id)
    except RenderStyleOutlineColorCommandError as exc:
        raise RenderStylePreferredSizeCommandError(
            RenderStylePreferredSizeCommandErrorCode(exc.code.value),
            str(exc),
        ) from exc


def _active_preferred_size_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    parent_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.RENDER_STYLE
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
        and _render_style_edit_field(edit) == "preferred_size"
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(
        edit for edit in candidates if edit.edit_id not in superseded_ids
    )
    if len(heads) != 1:
        raise RenderStylePreferredSizeCommandError(
            RenderStylePreferredSizeCommandErrorCode.PREFERRED_SIZE_SLOT_CONFLICT,
            "Preferred size has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _effective_preferred_size_state(
    parent: EffectiveParentSnapshot,
    *,
    automatic_preferred_size: float,
) -> tuple[float, str]:
    overrides = dict(parent.render_style_overrides)
    if "preferred_size" not in overrides:
        return automatic_preferred_size, "automatic"
    try:
        value = canonical_render_preferred_size(overrides["preferred_size"])
    except (TypeError, ValueError) as exc:
        raise RenderStylePreferredSizeCommandError(
            RenderStylePreferredSizeCommandErrorCode.PROJECTION_REJECTED,
            "The effective projector produced an invalid preferred size.",
        ) from exc
    return value, "user"


def _require_exact_preferred_size_invalidation(
    invalidation: InvalidationResult,
    *,
    parent_id: str,
) -> None:
    actual = tuple(
        (
            effect.dependency,
            effect.action,
            effect.scope,
            effect.target_ids,
            effect.reason,
        )
        for effect in invalidation.effects
    )
    expected = (
        (
            Dependency.LAYOUT_RENDER,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PARENT,
            (parent_id,),
            "render_style_override",
        ),
        (
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            (parent_id,),
            "render_style_override",
        ),
    )
    if invalidation.unresolved_facts or actual != expected:
        raise RenderStylePreferredSizeCommandError(
            RenderStylePreferredSizeCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Preferred-size invalidation must affect only parent layout and page output.",
        )


def _shadow_visibility_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    try:
        return _preferred_size_project_page(project, page_id)
    except RenderStylePreferredSizeCommandError as exc:
        raise RenderStyleShadowVisibilityCommandError(
            RenderStyleShadowVisibilityCommandErrorCode(exc.code.value),
            str(exc),
        ) from exc


def _shadow_visibility_automatic_parent(
    page: Mapping[str, Any],
    *,
    page_id: str,
    parent_id: str,
) -> Mapping[str, Any]:
    try:
        return _preferred_size_automatic_parent(
            page,
            page_id=page_id,
            parent_id=parent_id,
        )
    except RenderStylePreferredSizeCommandError as exc:
        raise RenderStyleShadowVisibilityCommandError(
            RenderStyleShadowVisibilityCommandErrorCode(exc.code.value),
            str(exc),
        ) from exc


def _shadow_visibility_effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    try:
        return _preferred_size_effective_parent(snapshot, parent_id)
    except RenderStylePreferredSizeCommandError as exc:
        raise RenderStyleShadowVisibilityCommandError(
            RenderStyleShadowVisibilityCommandErrorCode(exc.code.value),
            str(exc),
        ) from exc


def _active_shadow_visibility_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    parent_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.RENDER_STYLE
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
        and _render_style_edit_field(edit) == "shadow_enabled"
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(
        edit for edit in candidates if edit.edit_id not in superseded_ids
    )
    if len(heads) != 1:
        raise RenderStyleShadowVisibilityCommandError(
            RenderStyleShadowVisibilityCommandErrorCode.SHADOW_VISIBILITY_SLOT_CONFLICT,
            "Shadow visibility has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _effective_shadow_visibility_state(
    parent: EffectiveParentSnapshot,
) -> tuple[bool, str]:
    overrides = dict(parent.render_style_overrides)
    if "shadow_enabled" not in overrides:
        return True, "automatic"
    value = overrides["shadow_enabled"]
    if value is not False:
        raise RenderStyleShadowVisibilityCommandError(
            RenderStyleShadowVisibilityCommandErrorCode.PROJECTION_REJECTED,
            "The effective projector produced an invalid shadow-visibility value.",
        )
    return False, "user"


def _require_exact_shadow_visibility_invalidation(
    invalidation: InvalidationResult,
    *,
    parent_id: str,
) -> None:
    actual = tuple(
        (
            effect.dependency,
            effect.action,
            effect.scope,
            effect.target_ids,
            effect.reason,
        )
        for effect in invalidation.effects
    )
    expected = (
        (
            Dependency.LAYOUT_RENDER,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PARENT,
            (parent_id,),
            "render_style_override",
        ),
        (
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            (parent_id,),
            "render_style_override",
        ),
    )
    if invalidation.unresolved_facts or actual != expected:
        raise RenderStyleShadowVisibilityCommandError(
            RenderStyleShadowVisibilityCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Shadow-visibility invalidation must affect only parent layout and page output.",
        )


def _shadow_blur_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    try:
        return _preferred_size_project_page(project, page_id)
    except RenderStylePreferredSizeCommandError as exc:
        raise RenderStyleShadowBlurCommandError(
            RenderStyleShadowBlurCommandErrorCode(exc.code.value),
            str(exc),
        ) from exc


def _shadow_blur_automatic_parent(
    page: Mapping[str, Any],
    *,
    page_id: str,
    parent_id: str,
) -> Mapping[str, Any]:
    try:
        return _preferred_size_automatic_parent(
            page,
            page_id=page_id,
            parent_id=parent_id,
        )
    except RenderStylePreferredSizeCommandError as exc:
        raise RenderStyleShadowBlurCommandError(
            RenderStyleShadowBlurCommandErrorCode(exc.code.value),
            str(exc),
        ) from exc


def _shadow_blur_effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    try:
        return _preferred_size_effective_parent(snapshot, parent_id)
    except RenderStylePreferredSizeCommandError as exc:
        raise RenderStyleShadowBlurCommandError(
            RenderStyleShadowBlurCommandErrorCode(exc.code.value),
            str(exc),
        ) from exc


def _active_shadow_blur_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    parent_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.RENDER_STYLE
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
        and _render_style_edit_field(edit) == "shadow_blur"
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(
        edit for edit in candidates if edit.edit_id not in superseded_ids
    )
    if len(heads) != 1:
        raise RenderStyleShadowBlurCommandError(
            RenderStyleShadowBlurCommandErrorCode.SHADOW_BLUR_SLOT_CONFLICT,
            "Shadow blur has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _effective_shadow_blur_state(
    parent: EffectiveParentSnapshot,
    *,
    automatic_shadow_blur: float,
) -> tuple[float, str]:
    overrides = dict(parent.render_style_overrides)
    if "shadow_blur" not in overrides:
        return automatic_shadow_blur, "automatic"
    try:
        value = canonical_render_shadow_blur(overrides["shadow_blur"])
    except (TypeError, ValueError) as exc:
        raise RenderStyleShadowBlurCommandError(
            RenderStyleShadowBlurCommandErrorCode.PROJECTION_REJECTED,
            "The effective projector produced an invalid shadow blur.",
        ) from exc
    return value, "user"


def _require_exact_shadow_blur_invalidation(
    invalidation: InvalidationResult,
    *,
    parent_id: str,
) -> None:
    actual = tuple(
        (
            effect.dependency,
            effect.action,
            effect.scope,
            effect.target_ids,
            effect.reason,
        )
        for effect in invalidation.effects
    )
    expected = (
        (
            Dependency.LAYOUT_RENDER,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PARENT,
            (parent_id,),
            "render_style_override",
        ),
        (
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            (parent_id,),
            "render_style_override",
        ),
    )
    if invalidation.unresolved_facts or actual != expected:
        raise RenderStyleShadowBlurCommandError(
            RenderStyleShadowBlurCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Shadow-blur invalidation must affect only parent layout and page output.",
        )


def _font_role_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    try:
        return _fill_color_project_page(project, page_id)
    except RenderStyleFillColorCommandError as exc:
        raise RenderStyleFontRoleCommandError(
            RenderStyleFontRoleCommandErrorCode.PAGE_NOT_FOUND,
            str(exc),
        ) from exc


def _font_role_automatic_parent(
    page: Mapping[str, Any],
    *,
    page_id: str,
    parent_id: str,
) -> Mapping[str, Any]:
    try:
        return _fill_color_automatic_parent(
            page,
            page_id=page_id,
            parent_id=parent_id,
        )
    except RenderStyleFillColorCommandError as exc:
        raise RenderStyleFontRoleCommandError(
            RenderStyleFontRoleCommandErrorCode.PARENT_NOT_FOUND,
            str(exc),
        ) from exc


def _font_role_effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = [parent for parent in snapshot.parents if parent.parent_id == parent_id]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderStyleFontRoleCommandError(
            RenderStyleFontRoleCommandErrorCode.PARENT_NOT_FOUND,
            f"Effective parent identity is {reason}: {parent_id}",
        )
    return matches[0]


def _active_font_role_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    parent_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.RENDER_STYLE
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
        and _render_style_edit_field(edit) == "font_role"
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(edit for edit in candidates if edit.edit_id not in superseded_ids)
    if len(heads) != 1:
        raise RenderStyleFontRoleCommandError(
            RenderStyleFontRoleCommandErrorCode.FONT_ROLE_SLOT_CONFLICT,
            "Font role has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _effective_font_role_state(
    parent: EffectiveParentSnapshot,
    *,
    automatic_font_role: str,
) -> tuple[str, str]:
    overrides = dict(parent.render_style_overrides)
    if "font_role" not in overrides:
        return automatic_font_role, "automatic"
    try:
        value = canonical_render_font_role(overrides["font_role"])
    except (TypeError, ValueError) as exc:
        raise RenderStyleFontRoleCommandError(
            RenderStyleFontRoleCommandErrorCode.PROJECTION_REJECTED,
            "The effective projector produced an invalid font role.",
        ) from exc
    return value, "user"


def _require_exact_font_role_invalidation(
    invalidation: InvalidationResult,
    *,
    parent_id: str,
) -> None:
    actual = tuple(
        (
            effect.dependency,
            effect.action,
            effect.scope,
            effect.target_ids,
            effect.reason,
        )
        for effect in invalidation.effects
    )
    expected = (
        (
            Dependency.LAYOUT_RENDER,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PARENT,
            (parent_id,),
            "render_style_override",
        ),
        (
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            (parent_id,),
            "render_style_override",
        ),
    )
    if invalidation.unresolved_facts or actual != expected:
        raise RenderStyleFontRoleCommandError(
            RenderStyleFontRoleCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Font-role invalidation must affect only parent layout and page output.",
        )


def page_canvas_size_for_project_page(
    page: Mapping[str, Any],
    *,
    project_path: str,
) -> tuple[int, int]:
    """Resolve one page's authoritative pixel canvas without decoding pixels."""

    cleaned = page.get("cleaned_page_base")
    if not isinstance(cleaned, Mapping):
        cleaned = {}
    explicit = page.get("canvas_size") or cleaned.get("canvas_size")
    if (
        isinstance(explicit, (list, tuple))
        and len(explicit) == 2
        and all(
            not isinstance(value, bool) and isinstance(value, int) and value > 0
            for value in explicit
        )
    ):
        canvas = (int(explicit[0]), int(explicit[1]))
        if canvas[0] * canvas[1] <= 50_000_000:
            return canvas

    raw_path = (
        page.get("image_path")
        or page.get("source_image_path")
        or cleaned.get("source_image_path")
        or cleaned.get("image_path")
        or cleaned.get("cache_path")
    )
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ParentGeometryCommandError(
            ParentGeometryCommandErrorCode.CANVAS_UNAVAILABLE,
            "Page canvas dimensions are unavailable.",
        )
    candidate = os.path.expandvars(os.path.expanduser(raw_path.strip()))
    if not os.path.isabs(candidate):
        candidate = os.path.join(os.path.dirname(project_path), candidate)
    candidate = os.path.abspath(candidate)
    if not os.path.isfile(candidate):
        raise ParentGeometryCommandError(
            ParentGeometryCommandErrorCode.CANVAS_UNAVAILABLE,
            "The page image required to validate geometry is unavailable.",
        )
    try:
        from PIL import Image

        with Image.open(candidate) as opened:
            width, height = int(opened.width), int(opened.height)
    except Exception as exc:
        raise ParentGeometryCommandError(
            ParentGeometryCommandErrorCode.CANVAS_UNAVAILABLE,
            "The page image dimensions could not be validated.",
        ) from exc
    if width <= 0 or height <= 0 or width * height > 50_000_000:
        raise ParentGeometryCommandError(
            ParentGeometryCommandErrorCode.CANVAS_UNAVAILABLE,
            "The page canvas dimensions are invalid or exceed the safety limit.",
        )
    return width, height


def _validate_bbox_within_canvas(
    bbox: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
) -> None:
    try:
        x, y, width, height = _exact_bbox(bbox, "bbox")
    except (TypeError, ValueError) as exc:
        raise ParentGeometryCommandError(
            ParentGeometryCommandErrorCode.INVALID_GEOMETRY,
            str(exc),
        ) from exc
    page_width, page_height = canvas_size
    if x + width > page_width or y + height > page_height:
        raise ParentGeometryCommandError(
            ParentGeometryCommandErrorCode.GEOMETRY_OUT_OF_BOUNDS,
            "Parent geometry must remain fully contained by the page canvas.",
        )


class SourceTextCommandService:
    """Persist one source-text command through the existing GUI-1 owners.

    The caller owns the ``ProjectEditStore`` lifetime and therefore also owns
    the thread on which its SQLite connection is created and used.
    """

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: SourceTextCommand,
    ) -> SourceTextCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, SourceTextCommand):
            raise TypeError("command must be a SourceTextCommand")
        if self._edit_store.project_id != command.project_id:
            raise SourceTextCommandError(
                SourceTextCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise SourceTextCommandError(
                SourceTextCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )

        read_snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        materialized = read_snapshot.project
        if project_id_for(materialized) != command.project_id:
            raise SourceTextCommandError(
                SourceTextCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if read_snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise SourceTextCommandError(
                SourceTextCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the command was prepared.",
            )
        if read_snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise SourceTextCommandError(
                SourceTextCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the command was prepared.",
            )
        if read_snapshot.ledger.get(command.command_id) is not None:
            raise SourceTextCommandError(
                SourceTextCommandErrorCode.DUPLICATE_COMMAND,
                "The source-text command was already recorded.",
            )

        page = _source_project_page(materialized, command.page_id)
        before_page = project_effective_page(
            materialized,
            read_snapshot.ledger,
            page_id=command.page_id,
        )
        if (
            before_page.effective_fingerprint
            != command.expected_effective_page_fingerprint
        ):
            raise SourceTextCommandError(
                SourceTextCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _source_effective_parent(before_page, command.parent_id)
        slot_head = _active_source_slot_head(
            read_snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        target = EditTarget(
            EditTargetKind.PARENT,
            parent_id=command.parent_id,
        )
        revision_base: SourceTextRevisionBaseV1 | None = None
        revision_artifact: OcrSourceRevisionArtifact | None = None
        payload: Mapping[str, Any]
        if before_parent.origin is ParentOrigin.AUTOMATIC:
            if command.revision_base is not None:
                raise SourceTextCommandError(
                    SourceTextCommandErrorCode.REVISION_BASE_NOT_ALLOWED,
                    "Automatic parents do not accept a model-revision base.",
                )
            if command.operation not in {
                SourceTextOperation.REPLACE,
                SourceTextOperation.RESTORE_AUTOMATIC,
            }:
                raise SourceTextCommandError(
                    SourceTextCommandErrorCode.INVALID_OPERATION,
                    "Automatic parents support Replace or Restore Automatic only.",
                )
            if (
                command.operation is SourceTextOperation.REPLACE
                and before_parent.source_authority == "user"
                and before_parent.source_text == command.text
            ) or (
                command.operation is SourceTextOperation.RESTORE_AUTOMATIC
                and before_parent.source_authority == "automatic"
            ):
                raise SourceTextCommandError(
                    SourceTextCommandErrorCode.NO_OP,
                    "The requested source authority and text are already effective.",
                )
            payload = (
                {"text": command.text}
                if command.operation is SourceTextOperation.REPLACE
                else {}
            )
            base_fingerprint = field_base_fingerprint(
                project=materialized,
                page=page,
                target=target,
                domain=EditDomain.SOURCE_TEXT,
                operation=command.operation.value,
                payload=payload,
            )
            if base_fingerprint is None:
                raise SourceTextCommandError(
                    SourceTextCommandErrorCode.PARENT_NOT_FOUND,
                    "Automatic source-text evidence is unavailable for this parent.",
                )
            base_revision_id = before_parent.base_revision_id
        elif before_parent.origin is ParentOrigin.USER:
            if before_parent.source_evidence_mapping is not None:
                raise SourceTextCommandError(
                    SourceTextCommandErrorCode.INVALID_OPERATION,
                    "Mapped user parents retain their pipeline OCR mapping; edit an explicit OCR revision instead.",
                )
            if command.operation not in {
                SourceTextOperation.REPLACE,
                SourceTextOperation.RESTORE_SELECTED_REVISION,
            }:
                raise SourceTextCommandError(
                    SourceTextCommandErrorCode.INVALID_OPERATION,
                    "Revision-backed user parents support Replace or Restore Selected Model OCR only.",
                )
            revision_base, revision_artifact = (
                _require_revision_backed_source_state(
                    snapshot=read_snapshot,
                    parent=before_parent,
                    slot_head=slot_head,
                    command=command,
                )
            )
            if (
                command.operation is SourceTextOperation.REPLACE
                and before_parent.source_authority == "user"
                and before_parent.source_text == command.text
            ) or (
                command.operation
                is SourceTextOperation.RESTORE_SELECTED_REVISION
                and before_parent.source_authority == "ocr_revision"
                and before_parent.source_revision_id
                == revision_base.source_revision_id
            ):
                raise SourceTextCommandError(
                    SourceTextCommandErrorCode.NO_OP,
                    "The requested source authority, revision, and text are already effective.",
                )
            payload = (
                {
                    "text": command.text,
                    "revision_base": revision_base.to_dict(),
                }
                if command.operation is SourceTextOperation.REPLACE
                else {"revision_base": revision_base.to_dict()}
            )
            base_revision_id = revision_base.source_revision_id
            base_fingerprint = revision_base.artifact_sha256
        else:
            raise SourceTextCommandError(
                SourceTextCommandErrorCode.PARENT_LINEAGE_MISMATCH,
                "The selected parent origin is unsupported.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.SOURCE_TEXT,
            operation=command.operation.value,
            payload=payload,
            base_revision_id=base_revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )

        candidate_ledger = read_snapshot.ledger.append(edit)
        after_page = project_effective_page(
            materialized,
            candidate_ledger,
            page_id=command.page_id,
        )
        after_parent = _source_effective_parent(after_page, command.parent_id)
        if edit.edit_id not in after_page.applied_edit_ids:
            raise SourceTextCommandError(
                SourceTextCommandErrorCode.PROJECTION_REJECTED,
                "The source-text edit was not accepted by the effective projector.",
            )
        if command.operation is SourceTextOperation.REPLACE:
            accepted = (
                after_parent.source_text == command.text
                and after_parent.source_authority == "user"
            )
        elif command.operation is SourceTextOperation.RESTORE_AUTOMATIC:
            accepted = after_parent.source_authority == "automatic"
        else:
            assert revision_base is not None
            assert revision_artifact is not None
            accepted = (
                after_parent.source_authority == "ocr_revision"
                and after_parent.source_revision_id
                == revision_base.source_revision_id
                and after_parent.source_text
                == revision_artifact.source_text
            )
        if not accepted:
            raise SourceTextCommandError(
                SourceTextCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested source state.",
            )

        invalidation = invalidation_for_edit(edit)
        cleanup_effects = tuple(
            effect
            for effect in invalidation.effects
            if effect.dependency is Dependency.CLEANUP_BASE
        )
        if invalidation.unresolved_facts or cleanup_effects:
            raise SourceTextCommandError(
                SourceTextCommandErrorCode.PROJECTION_REJECTED,
                "Source-text invalidation must be resolved and keep cleanup base.",
            )

        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=read_snapshot.page_head_sha256,
                expected_global_head_sha256=read_snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise SourceTextCommandError(
                SourceTextCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before the source text was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise SourceTextCommandError(
                SourceTextCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before the source text was committed.",
            ) from exc

        return SourceTextCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            before_source_text=before_parent.source_text,
            after_source_text=after_parent.source_text,
            before_source_authority=before_parent.source_authority,
            after_source_authority=after_parent.source_authority,
            before_target_text=before_parent.target_text,
            after_target_text=after_parent.target_text,
            before_target_authority=before_parent.target_authority,
            after_target_authority=after_parent.target_authority,
            before_target_freshness=before_parent.target_freshness,
            after_target_freshness=after_parent.target_freshness,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


class TargetTextCommandService:
    """Persist one target-text command through the existing GUI-1 owners.

    The caller owns the ``ProjectEditStore`` lifetime and therefore also owns
    the thread on which its SQLite connection is created and used.
    """

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: TargetTextCommand,
    ) -> TargetTextCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, TargetTextCommand):
            raise TypeError("command must be a TargetTextCommand")
        if self._edit_store.project_id != command.project_id:
            raise TargetTextCommandError(
                TargetTextCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise TargetTextCommandError(
                TargetTextCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )

        read_snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        materialized = read_snapshot.project
        if project_id_for(materialized) != command.project_id:
            raise TargetTextCommandError(
                TargetTextCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if read_snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise TargetTextCommandError(
                TargetTextCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the command was prepared.",
            )
        if read_snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise TargetTextCommandError(
                TargetTextCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the command was prepared.",
            )
        if read_snapshot.ledger.get(command.command_id) is not None:
            raise TargetTextCommandError(
                TargetTextCommandErrorCode.DUPLICATE_COMMAND,
                "The target-text command was already recorded.",
            )

        page = _project_page(materialized, command.page_id)
        before_page = project_effective_page(
            materialized,
            read_snapshot.ledger,
            page_id=command.page_id,
        )
        if (
            before_page.effective_fingerprint
            != command.expected_effective_page_fingerprint
        ):
            raise TargetTextCommandError(
                TargetTextCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _effective_parent(before_page, command.parent_id)
        slot_head = _active_target_slot_head(
            read_snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        target = EditTarget(
            EditTargetKind.PARENT,
            parent_id=command.parent_id,
        )
        source_fingerprint = effective_source_fingerprint(
            before_parent.parent_id,
            before_parent.source_text,
        )
        revision_base: TargetTextRevisionBaseV1 | None = None
        revision_artifact: TranslationRevisionArtifact | None = None
        source_evidence_base: ParentSourceEvidenceMappingV1 | None = None
        payload: Mapping[str, Any]
        if before_parent.origin is ParentOrigin.AUTOMATIC:
            if command.revision_base is not None:
                raise TargetTextCommandError(
                    TargetTextCommandErrorCode.REVISION_BASE_NOT_ALLOWED,
                    "Automatic parents do not accept a model-revision base.",
                )
            if command.source_evidence_base is not None:
                raise TargetTextCommandError(
                    TargetTextCommandErrorCode.MAPPED_BASE_NOT_ALLOWED,
                    "Automatic parents do not accept a mapped pipeline base.",
                )
            if command.operation not in {
                TargetTextOperation.REPLACE,
                TargetTextOperation.RESTORE_AUTOMATIC,
            }:
                raise TargetTextCommandError(
                    TargetTextCommandErrorCode.INVALID_OPERATION,
                    "Automatic parents support Replace or Restore Automatic only.",
                )
            if (
                command.operation is TargetTextOperation.REPLACE
                and before_parent.target_authority == "user"
                and before_parent.target_text == command.text
            ) or (
                command.operation is TargetTextOperation.RESTORE_AUTOMATIC
                and before_parent.target_authority == "automatic"
            ):
                raise TargetTextCommandError(
                    TargetTextCommandErrorCode.NO_OP,
                    "The requested target authority and text are already effective.",
                )
            if command.operation is TargetTextOperation.REPLACE:
                payload = {
                    "text": command.text,
                    "source_fingerprint": source_fingerprint,
                }
            else:
                payload = {}
            base_fingerprint = field_base_fingerprint(
                project=materialized,
                page=page,
                target=target,
                domain=EditDomain.TARGET_TEXT,
                operation=command.operation.value,
                payload=payload,
            )
            if base_fingerprint is None:
                raise TargetTextCommandError(
                    TargetTextCommandErrorCode.PARENT_NOT_FOUND,
                    "Automatic target-text evidence is unavailable for this parent.",
                )
            base_revision_id = before_parent.base_revision_id
        elif before_parent.origin is ParentOrigin.USER:
            if before_parent.source_evidence_mapping is not None:
                if command.revision_base is not None:
                    raise TargetTextCommandError(
                        TargetTextCommandErrorCode.REVISION_BASE_NOT_ALLOWED,
                        "Mapped user parents do not accept a model-revision base.",
                    )
                if command.operation not in {
                    TargetTextOperation.REPLACE,
                    TargetTextOperation.RESTORE_MAPPED_PIPELINE,
                }:
                    raise TargetTextCommandError(
                        TargetTextCommandErrorCode.INVALID_OPERATION,
                        "Mapped user parents support Replace or Restore Mapped Pipeline Translation only.",
                    )
                source_evidence_base = _require_mapped_target_state(
                    snapshot=read_snapshot,
                    parent=before_parent,
                    slot_head=slot_head,
                    command=command,
                )
                if (
                    command.operation is TargetTextOperation.REPLACE
                    and before_parent.target_authority == "user"
                    and before_parent.target_text == command.text
                ) or (
                    command.operation
                    is TargetTextOperation.RESTORE_MAPPED_PIPELINE
                    and before_parent.target_authority == "mapped_automatic"
                    and before_parent.target_text
                    == source_evidence_base.target_text
                ):
                    raise TargetTextCommandError(
                        TargetTextCommandErrorCode.NO_OP,
                        "The requested mapped target authority and text are already effective.",
                    )
                if command.operation is TargetTextOperation.REPLACE:
                    payload = {
                        "text": command.text,
                        "source_fingerprint": source_fingerprint,
                        "source_evidence_base": source_evidence_base.to_dict(),
                    }
                else:
                    payload = {
                        "source_evidence_base": source_evidence_base.to_dict()
                    }
                base_revision_id = before_parent.lineage.authored_edit_id
                base_fingerprint = source_evidence_base.fingerprint
            else:
                if command.source_evidence_base is not None:
                    raise TargetTextCommandError(
                        TargetTextCommandErrorCode.MAPPED_BASE_NOT_ALLOWED,
                        "Revision-backed user parents do not accept a mapped pipeline base.",
                    )
                if command.operation not in {
                    TargetTextOperation.REPLACE,
                    TargetTextOperation.RESTORE_SELECTED_REVISION,
                }:
                    raise TargetTextCommandError(
                        TargetTextCommandErrorCode.INVALID_OPERATION,
                        "Revision-backed user parents support Replace or Restore Selected Model Translation only.",
                    )
                revision_base, revision_artifact = (
                    _require_revision_backed_target_state(
                        snapshot=read_snapshot,
                        effective_page=before_page,
                        parent=before_parent,
                        slot_head=slot_head,
                        command=command,
                    )
                )
                if (
                    command.operation is TargetTextOperation.REPLACE
                    and before_parent.target_authority == "user"
                    and before_parent.target_text == command.text
                ) or (
                    command.operation
                    is TargetTextOperation.RESTORE_SELECTED_REVISION
                    and before_parent.target_authority == "translation_revision"
                    and before_parent.target_revision_id
                    == revision_base.translation_revision_id
                ):
                    raise TargetTextCommandError(
                        TargetTextCommandErrorCode.NO_OP,
                        "The requested target authority, revision, and text are already effective.",
                    )
                if command.operation is TargetTextOperation.REPLACE:
                    payload = {
                        "text": command.text,
                        "source_fingerprint": source_fingerprint,
                        "revision_base": revision_base.to_dict(),
                    }
                else:
                    payload = {"revision_base": revision_base.to_dict()}
                base_revision_id = revision_base.translation_revision_id
                base_fingerprint = revision_base.artifact_sha256
        else:
            raise TargetTextCommandError(
                TargetTextCommandErrorCode.PARENT_LINEAGE_MISMATCH,
                "The selected parent origin is unsupported.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.TARGET_TEXT,
            operation=command.operation.value,
            payload=payload,
            base_revision_id=base_revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )

        candidate_ledger = read_snapshot.ledger.append(edit)
        after_page = project_effective_page(
            materialized,
            candidate_ledger,
            page_id=command.page_id,
        )
        after_parent = _effective_parent(after_page, command.parent_id)
        if edit.edit_id not in after_page.applied_edit_ids:
            raise TargetTextCommandError(
                TargetTextCommandErrorCode.PROJECTION_REJECTED,
                "The target-text edit was not accepted by the effective projector.",
            )
        if command.operation is TargetTextOperation.REPLACE:
            accepted = (
                after_parent.target_text == command.text
                and after_parent.target_authority == "user"
            )
            if revision_base is not None:
                try:
                    accepted = accepted and (
                        target_text_revision_base_for_parent(after_parent)
                        == revision_base
                    )
                except (TypeError, ValueError):
                    accepted = False
            elif source_evidence_base is not None:
                accepted = accepted and (
                    after_parent.source_evidence_mapping
                    == source_evidence_base
                    and after_parent.target_revision_id is None
                )
        elif command.operation is TargetTextOperation.RESTORE_SELECTED_REVISION:
            assert revision_base is not None
            assert revision_artifact is not None
            try:
                accepted = (
                    after_parent.target_text == revision_artifact.target_text
                    and after_parent.target_authority == "translation_revision"
                    and after_parent.target_revision_id
                    == revision_base.translation_revision_id
                    and target_text_revision_base_for_parent(after_parent)
                    == revision_base
                )
            except (TypeError, ValueError):
                accepted = False
        elif command.operation is TargetTextOperation.RESTORE_MAPPED_PIPELINE:
            assert source_evidence_base is not None
            accepted = (
                after_parent.target_text == source_evidence_base.target_text
                and after_parent.target_authority == "mapped_automatic"
                and after_parent.target_revision_id is None
                and after_parent.source_evidence_mapping
                == source_evidence_base
            )
        else:
            accepted = after_parent.target_authority == "automatic"
        if not accepted:
            raise TargetTextCommandError(
                TargetTextCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested target state.",
            )

        invalidation = invalidation_for_edit(edit)
        expected_translation_action = (
            InvalidationAction.NEW_REVISION
            if command.operation
            in {
                TargetTextOperation.RESTORE_SELECTED_REVISION,
                TargetTextOperation.RESTORE_MAPPED_PIPELINE,
            }
            else InvalidationAction.USER_CURRENT
        )
        expected_effects = {
            (
                Dependency.TRANSLATION,
                expected_translation_action,
                InvalidationScope.PARENT,
                (command.parent_id,),
            ),
            (
                Dependency.LAYOUT_RENDER,
                InvalidationAction.RECOMPUTE,
                InvalidationScope.PARENT,
                (command.parent_id,),
            ),
            (
                Dependency.PAGE_OUTPUT,
                InvalidationAction.RECOMPUTE,
                InvalidationScope.PAGE,
                (command.parent_id,),
            ),
        }
        observed_effects = {
            (
                effect.dependency,
                effect.action,
                effect.scope,
                effect.target_ids,
            )
            for effect in invalidation.effects
        }
        if invalidation.unresolved_facts or observed_effects != expected_effects:
            raise TargetTextCommandError(
                TargetTextCommandErrorCode.PROJECTION_REJECTED,
                "Target-text invalidation differs from the exact edit contract.",
            )

        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=read_snapshot.page_head_sha256,
                expected_global_head_sha256=read_snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise TargetTextCommandError(
                TargetTextCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before the target text was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise TargetTextCommandError(
                TargetTextCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before the target text was committed.",
            ) from exc

        return TargetTextCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            before_target_text=before_parent.target_text,
            after_target_text=after_parent.target_text,
            before_target_authority=before_parent.target_authority,
            after_target_authority=after_parent.target_authority,
            source_fingerprint=source_fingerprint,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


class ParentMembershipCommandService:
    """Persist one parent-membership command through existing GUI-1 owners.

    The caller owns the ``ProjectEditStore`` lifetime and therefore also owns
    the thread on which its SQLite connection is created and used.
    """

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: ParentMembershipCommand,
    ) -> ParentMembershipCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, ParentMembershipCommand):
            raise TypeError("command must be a ParentMembershipCommand")
        if self._edit_store.project_id != command.project_id:
            raise ParentMembershipCommandError(
                ParentMembershipCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise ParentMembershipCommandError(
                ParentMembershipCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )

        read_snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        materialized = read_snapshot.project
        if project_id_for(materialized) != command.project_id:
            raise ParentMembershipCommandError(
                ParentMembershipCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if read_snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise ParentMembershipCommandError(
                ParentMembershipCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the command was prepared.",
            )
        if read_snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise ParentMembershipCommandError(
                ParentMembershipCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the command was prepared.",
            )
        if read_snapshot.ledger.get(command.command_id) is not None:
            raise ParentMembershipCommandError(
                ParentMembershipCommandErrorCode.DUPLICATE_COMMAND,
                "The parent-membership command was already recorded.",
            )

        page = _membership_project_page(materialized, command.page_id)
        before_page = project_effective_page(
            materialized,
            read_snapshot.ledger,
            page_id=command.page_id,
        )
        if (
            before_page.effective_fingerprint
            != command.expected_effective_page_fingerprint
        ):
            raise ParentMembershipCommandError(
                ParentMembershipCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _membership_effective_parent(
            before_page,
            command.parent_id,
        )
        slot_head = _active_membership_slot_head(
            read_snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        target = EditTarget(
            EditTargetKind.PARENT,
            parent_id=command.parent_id,
        )
        payload: Mapping[str, Any] = {}
        base_fingerprint = field_base_fingerprint(
            project=materialized,
            page=page,
            target=target,
            domain=EditDomain.STRUCTURAL,
            operation=command.operation.value,
            payload=payload,
        )
        if base_fingerprint is None:
            raise ParentMembershipCommandError(
                ParentMembershipCommandErrorCode.PARENT_NOT_FOUND,
                "Automatic parent membership evidence is unavailable.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.STRUCTURAL,
            operation=command.operation.value,
            payload=payload,
            base_revision_id=before_parent.base_revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )

        candidate_ledger = read_snapshot.ledger.append(edit)
        after_page = project_effective_page(
            materialized,
            candidate_ledger,
            page_id=command.page_id,
        )
        after_parent = _membership_effective_parent(
            after_page,
            command.parent_id,
        )
        expected_excluded = command.operation is ParentMembershipOperation.EXCLUDE
        if (
            edit.edit_id not in after_page.applied_edit_ids
            or after_parent.excluded is not expected_excluded
        ):
            raise ParentMembershipCommandError(
                ParentMembershipCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested parent membership.",
            )

        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=read_snapshot.page_head_sha256,
                expected_global_head_sha256=read_snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise ParentMembershipCommandError(
                ParentMembershipCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before parent membership was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise ParentMembershipCommandError(
                ParentMembershipCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before parent membership was committed.",
            ) from exc

        return ParentMembershipCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            before_excluded=before_parent.excluded,
            after_excluded=after_parent.excluded,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation_for_edit(edit),
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


class AddUserParentCommandService:
    """Persist one standalone, pending user parent without invoking an owner."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: AddUserParentCommand,
    ) -> AddUserParentCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, AddUserParentCommand):
            raise TypeError("command must be an AddUserParentCommand")
        if self._edit_store.project_id != command.project_id:
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )
        snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        return self.execute_materialized(snapshot=snapshot, command=command)

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditReadSnapshot,
        command: AddUserParentCommand,
    ) -> AddUserParentCommandReceipt:
        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, AddUserParentCommand):
            raise TypeError("command must be an AddUserParentCommand")
        if self._edit_store.project_id != command.project_id:
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        materialized = snapshot.project
        if (
            project_id_for(materialized) != command.project_id
            or snapshot.ledger.project_id != command.project_id
        ):
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the add-parent command was prepared.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the add-parent command was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.DUPLICATE_COMMAND,
                "The add-parent command was already recorded.",
            )

        page = _add_user_parent_project_page(materialized, command.page_id)
        try:
            canvas_size = page_canvas_size_for_project_page(
                page,
                project_path=self._edit_store.project_path,
            )
        except ParentGeometryCommandError as exc:
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.CANVAS_UNAVAILABLE,
                "Page canvas dimensions are unavailable for Add Parent.",
            ) from exc
        x, y, width, height = command.workflow_area_bbox
        if x + width > canvas_size[0] or y + height > canvas_size[1]:
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.WORKFLOW_AREA_OUT_OF_BOUNDS,
                "The workflow area must remain fully contained by the page canvas.",
            )
        _require_user_parent_identities_available(
            materialized,
            snapshot.ledger,
            parent_id=command.parent_id,
            root_id=command.root_id,
        )
        try:
            before_page = project_effective_page(
                materialized,
                snapshot.ledger,
                page_id=command.page_id,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.PROJECTION_REJECTED,
                "The effective page could not be projected before Add Parent.",
            ) from exc
        if (
            before_page.effective_fingerprint
            != command.expected_effective_page_fingerprint
        ):
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after Add Parent was prepared.",
            )

        payload: Mapping[str, Any] = {
            "identity_namespace": USER_PARENT_IDENTITY_NAMESPACE,
            "root_id": command.root_id,
            "root_identity_namespace": USER_ROOT_IDENTITY_NAMESPACE,
            "role": command.role,
            "workflow_area_bbox": list(command.workflow_area_bbox),
            "canvas_size": list(canvas_size),
            "order_policy": "append",
        }
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=EditTarget(
                EditTargetKind.PARENT,
                parent_id=command.parent_id,
            ),
            domain=EditDomain.STRUCTURAL,
            operation=command.operation.value,
            payload=payload,
            base_revision_id=before_page.hierarchy.revision_id,
            base_fingerprint=before_page.hierarchy.fingerprint,
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        _require_exact_add_user_parent_invalidation(
            invalidation,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )

        try:
            candidate_ledger = snapshot.ledger.append(edit)
            after_page = project_effective_page(
                materialized,
                candidate_ledger,
                page_id=command.page_id,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector rejected Add Parent.",
            ) from exc
        parent_matches = tuple(
            parent
            for parent in after_page.parents
            if parent.parent_id == command.parent_id
        )
        root_matches = tuple(
            root
            for root in after_page.hierarchy.user_roots
            if root.root_id == command.root_id
        )
        if len(parent_matches) != 1 or len(root_matches) != 1:
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.PROJECTION_REJECTED,
                "The projector did not materialize one exact user parent/root.",
            )
        projected_parent = parent_matches[0]
        if (
            edit.edit_id not in after_page.applied_edit_ids
            or edit.edit_id not in after_page.hierarchy.descriptor.active_structural_edit_ids
            or after_page.hierarchy.revision_id == before_page.hierarchy.revision_id
            or projected_parent.origin.value != "user"
            or projected_parent.bundle_id is not None
            or projected_parent.automatic_fingerprint is not None
            or thaw_json(projected_parent.automatic_geometry) is not None
            or thaw_json(projected_parent.geometry) is not None
            or thaw_json(projected_parent.render_allowed_area) is not None
            or thaw_json(projected_parent.root_bbox) is not None
            or tuple(thaw_json(projected_parent.workflow_area_bbox))
            != command.workflow_area_bbox
            or projected_parent.source_text is not None
            or projected_parent.target_text is not None
            or projected_parent.target_freshness is not TargetFreshness.UNAVAILABLE
            or len(projected_parent.stage_requirements) != 8
            or any(
                edit.edit_id in issue.edit_ids for issue in after_page.issues
            )
        ):
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.PROJECTION_REJECTED,
                "The projected user parent contains invalid or fabricated state.",
            )

        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before Add Parent was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before Add Parent was committed.",
            ) from exc
        if commit_receipt.artifact_revision_ids:
            raise AddUserParentCommandError(
                AddUserParentCommandErrorCode.PROJECTION_REJECTED,
                "Add Parent must not publish artifact revisions.",
            )

        return AddUserParentCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            parent_id=command.parent_id,
            root_id=command.root_id,
            role=command.role,
            workflow_area_bbox=command.workflow_area_bbox,
            canvas_size=canvas_size,
            before_hierarchy_revision_id=before_page.hierarchy.revision_id,
            after_hierarchy_revision_id=after_page.hierarchy.revision_id,
            before_hierarchy_fingerprint=before_page.hierarchy.fingerprint,
            after_hierarchy_fingerprint=after_page.hierarchy.fingerprint,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            stage_requirements=projected_parent.stage_requirements,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


def split_user_parent_bboxes(
    source_bbox: tuple[int, int, int, int],
    *,
    orientation: SplitUserParentOrientation,
    split_offset: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Return the canonical left/right or top/bottom integer partition."""

    x, y, width, height = _exact_bbox(source_bbox, "source_bbox")
    orientation = SplitUserParentOrientation(orientation)
    if isinstance(split_offset, bool) or not isinstance(split_offset, int):
        raise ValueError("split_offset must be an exact integer")
    if orientation is SplitUserParentOrientation.VERTICAL:
        if split_offset <= 0 or split_offset >= width:
            raise ValueError("vertical split_offset must lie strictly inside width")
        return (
            (x, y, split_offset, height),
            (x + split_offset, y, width - split_offset, height),
        )
    if split_offset <= 0 or split_offset >= height:
        raise ValueError("horizontal split_offset must lie strictly inside height")
    return (
        (x, y, width, split_offset),
        (x, y + split_offset, width, height - split_offset),
    )


class SplitUserParentCommandService:
    """Persist one topology-only split without invoking any pipeline owner."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: SplitUserParentCommand,
    ) -> SplitUserParentCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, SplitUserParentCommand):
            raise TypeError("command must be a SplitUserParentCommand")
        if self._edit_store.project_id != command.project_id:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )
        snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        return self.execute_materialized(snapshot=snapshot, command=command)

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditReadSnapshot,
        command: SplitUserParentCommand,
    ) -> SplitUserParentCommandReceipt:
        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, SplitUserParentCommand):
            raise TypeError("command must be a SplitUserParentCommand")
        if self._edit_store.project_id != command.project_id:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        materialized = snapshot.project
        if (
            project_id_for(materialized) != command.project_id
            or snapshot.ledger.project_id != command.project_id
        ):
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after Split Parent was prepared.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after Split Parent was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.DUPLICATE_COMMAND,
                "The Split Parent command was already recorded.",
            )

        try:
            before_page = project_effective_page(
                materialized,
                snapshot.ledger,
                page_id=command.page_id,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.PROJECTION_REJECTED,
                "The effective page could not be projected before Split Parent.",
            ) from exc
        if (
            before_page.effective_fingerprint
            != command.expected_effective_page_fingerprint
        ):
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after Split Parent was prepared.",
            )
        if any(
            edit.domain is EditDomain.STRUCTURAL
            and edit.operation == SplitUserParentOperation.SPLIT.value
            and edit.target.kind is EditTargetKind.PARENT
            and edit.target.parent_id == command.source_parent_id
            for edit in snapshot.ledger.active_edits(page_id=command.page_id)
        ):
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.SPLIT_SLOT_CONFLICT,
                "The selected parent already has an active Split Parent edit.",
            )
        source_matches = tuple(
            parent
            for parent in before_page.parents
            if parent.parent_id == command.source_parent_id
        )
        if len(source_matches) != 1:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.SOURCE_PARENT_NOT_FOUND,
                "The selected source parent is unavailable.",
            )
        source_parent = source_matches[0]
        if source_parent.excluded:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.SOURCE_PARENT_EXCLUDED,
                "Restore the selected parent before splitting it.",
            )
        lineage = source_parent.lineage
        source_authored_edit = (
            snapshot.ledger.get(lineage.authored_edit_id)
            if lineage is not None
            else None
        )
        active_edit_ids = set(snapshot.ledger.state().active_edit_ids)
        legacy_standalone_source = bool(
            source_parent.origin is ParentOrigin.USER
            and lineage is not None
            and lineage.order_policy == "append"
            and source_authored_edit is not None
            and source_authored_edit.edit_id in active_edit_ids
            and source_authored_edit.domain is EditDomain.STRUCTURAL
            and source_authored_edit.operation == AddUserParentOperation.ADD.value
            and source_authored_edit.target.kind is EditTargetKind.PARENT
            and source_authored_edit.target.parent_id == command.source_parent_id
            and str(source_authored_edit.payload.get("root_id") or "")
            == source_parent.root_id
            and source_parent.source_evidence_mapping is None
        )
        source_mapping = source_parent.source_evidence_mapping
        evidence_backed_source = bool(
            source_parent.origin is ParentOrigin.USER
            and lineage is not None
            and lineage.order_policy == "replace_sources"
            and source_authored_edit is not None
            and source_authored_edit.edit_id in active_edit_ids
            and source_authored_edit.domain is EditDomain.STRUCTURAL
            and source_authored_edit.operation
            == MergePipelineParentsOperation.MERGE.value
            and source_authored_edit.target.kind is EditTargetKind.PARENT
            and source_authored_edit.target.parent_id == command.source_parent_id
            and str(source_authored_edit.payload.get("merged_root_id") or "")
            == source_parent.root_id
            and source_mapping is not None
            and source_mapping.page_id == command.page_id
            and source_mapping.source_parent_ids == lineage.source_parent_ids
            and source_mapping.source_root_ids == lineage.source_root_ids
            and source_mapping.source_automatic_fingerprints
            == lineage.source_automatic_fingerprints
            and source_mapping.source_text == source_parent.source_text
            and source_mapping.workflow_bbox
            == tuple(lineage.workflow_area_bbox)
        )
        if not legacy_standalone_source and not evidence_backed_source:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.SOURCE_PARENT_NOT_STANDALONE,
                "Split Parent requires an active evidence-backed merged parent.",
            )
        try:
            page = _add_user_parent_project_page(materialized, command.page_id)
        except AddUserParentCommandError as exc:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.PAGE_NOT_FOUND,
                "Project page identity is unavailable for Split Parent.",
            ) from exc
        try:
            canvas_size = page_canvas_size_for_project_page(
                page,
                project_path=self._edit_store.project_path,
            )
        except ParentGeometryCommandError as exc:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.CANVAS_UNAVAILABLE,
                "Page canvas dimensions are unavailable for Split Parent.",
            ) from exc
        if tuple(lineage.canvas_size) != canvas_size:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "The selected parent's page canvas changed.",
            )
        source_bbox = tuple(lineage.workflow_area_bbox)
        try:
            child_bboxes = split_user_parent_bboxes(
                source_bbox,
                orientation=command.orientation,
                split_offset=command.split_offset,
            )
        except (TypeError, ValueError) as exc:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.INVALID_SPLIT_OFFSET,
                str(exc),
            ) from exc
        child_source_mappings: tuple[
            ParentSourceEvidenceMappingV1,
            ParentSourceEvidenceMappingV1,
        ] | None = None
        if evidence_backed_source:
            assert source_mapping is not None
            try:
                child_source_mappings = source_mapping.partition(child_bboxes)
            except (TypeError, ValueError) as exc:
                raise SplitUserParentCommandError(
                    SplitUserParentCommandErrorCode.SOURCE_EVIDENCE_PARTITION_INVALID,
                    "The split must keep every mapped detection/OCR source wholly inside exactly one child.",
                ) from exc
        for child_parent_id, child_root_id in (
            (command.first_parent_id, command.first_root_id),
            (command.second_parent_id, command.second_root_id),
        ):
            try:
                _require_user_parent_identities_available(
                    materialized,
                    snapshot.ledger,
                    parent_id=child_parent_id,
                    root_id=child_root_id,
                )
            except AddUserParentCommandError as exc:
                raise SplitUserParentCommandError(
                    SplitUserParentCommandErrorCode.IDENTITY_COLLISION,
                    "A Split Parent child identity is already reserved.",
                ) from exc

        child_parent_ids = (
            command.first_parent_id,
            command.second_parent_id,
        )
        child_root_ids = (command.first_root_id, command.second_root_id)
        payload_values: dict[str, Any] = {
            "identity_namespace": USER_PARENT_IDENTITY_NAMESPACE,
            "root_identity_namespace": USER_ROOT_IDENTITY_NAMESPACE,
            "source_root_id": source_parent.root_id,
            "source_authored_edit_id": lineage.authored_edit_id,
            "source_role": source_parent.role,
            "source_workflow_area_bbox": list(source_bbox),
            "canvas_size": list(canvas_size),
            "orientation": command.orientation.value,
            "split_offset": command.split_offset,
            "child_parent_ids": list(child_parent_ids),
            "child_root_ids": list(child_root_ids),
            "child_workflow_area_bboxes": [list(bbox) for bbox in child_bboxes],
            "order_policy": "replace_source",
        }
        if child_source_mappings is not None:
            payload_values["child_source_evidence_mappings"] = [
                mapping.to_dict() for mapping in child_source_mappings
            ]
        payload: Mapping[str, Any] = payload_values
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=EditTarget(
                EditTargetKind.PARENT,
                parent_id=command.source_parent_id,
            ),
            domain=EditDomain.STRUCTURAL,
            operation=command.operation.value,
            payload=payload,
            base_revision_id=before_page.hierarchy.revision_id,
            base_fingerprint=before_page.hierarchy.fingerprint,
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        _require_exact_split_user_parent_invalidation(
            invalidation,
            page_id=command.page_id,
            child_parent_ids=child_parent_ids,
        )
        try:
            candidate_ledger = snapshot.ledger.append(edit)
            after_page = project_effective_page(
                materialized,
                candidate_ledger,
                page_id=command.page_id,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector rejected Split Parent.",
            ) from exc
        projected_children = tuple(
            next(
                (
                    parent
                    for parent in after_page.parents
                    if parent.parent_id == child_parent_id
                ),
                None,
            )
            for child_parent_id in child_parent_ids
        )
        before_order = before_page.hierarchy.ordered_parent_ids
        expected_order: list[str] = []
        for parent_id in before_order:
            if parent_id == command.source_parent_id:
                expected_order.extend(child_parent_ids)
            else:
                expected_order.append(parent_id)
        child_requirements: list[tuple[ParentStageRequirement, ...]] = []
        children_are_exact = True
        for index, projected_child in enumerate(projected_children):
            if projected_child is None or projected_child.lineage is None:
                children_are_exact = False
                break
            child_lineage = projected_child.lineage
            child_requirements.append(projected_child.stage_requirements)
            expected_mapping = (
                child_source_mappings[index]
                if child_source_mappings is not None
                else None
            )
            evidence_projection_is_exact = bool(
                expected_mapping is not None
                and projected_child.source_evidence_mapping == expected_mapping
                and thaw_json(projected_child.geometry) == list(child_bboxes[index])
                and projected_child.source_text == expected_mapping.source_text
                and projected_child.source_authority == "user"
                and projected_child.target_text == expected_mapping.target_text
                and projected_child.target_authority
                == (
                    "mapped_automatic"
                    if expected_mapping.target_text is not None
                    else "unavailable"
                )
                and projected_child.target_freshness
                is (
                    TargetFreshness.CURRENT
                    if expected_mapping.target_text is not None
                    else TargetFreshness.UNAVAILABLE
                )
            )
            legacy_projection_is_exact = bool(
                expected_mapping is None
                and projected_child.source_evidence_mapping is None
                and thaw_json(projected_child.geometry) is None
                and projected_child.source_text is None
                and projected_child.target_text is None
                and projected_child.target_freshness
                is TargetFreshness.UNAVAILABLE
            )
            if (
                projected_child.origin is not ParentOrigin.USER
                or projected_child.root_id != child_root_ids[index]
                or projected_child.role != source_parent.role
                or projected_child.bundle_id is not None
                or projected_child.automatic_fingerprint is not None
                or thaw_json(projected_child.automatic_geometry) is not None
                or thaw_json(projected_child.render_allowed_area) is not None
                or thaw_json(projected_child.root_bbox) is not None
                or tuple(thaw_json(projected_child.workflow_area_bbox))
                != child_bboxes[index]
                or not (
                    evidence_projection_is_exact or legacy_projection_is_exact
                )
                or child_lineage.order_policy != "replace_source"
                or child_lineage.source_parent_id != command.source_parent_id
                or child_lineage.source_root_id != source_parent.root_id
                or child_lineage.source_authored_edit_id != lineage.authored_edit_id
                or child_lineage.split_orientation != command.orientation.value
                or child_lineage.split_ordinal != index
                or len(projected_child.stage_requirements) != 8
            ):
                children_are_exact = False
                break
        if (
            edit.edit_id not in after_page.applied_edit_ids
            or edit.edit_id
            not in after_page.hierarchy.descriptor.active_structural_edit_ids
            or command.source_parent_id
            in after_page.hierarchy.ordered_parent_ids
            or after_page.hierarchy.ordered_parent_ids != tuple(expected_order)
            or len(projected_children) != 2
            or not children_are_exact
            or any(edit.edit_id in issue.edit_ids for issue in after_page.issues)
        ):
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.PROJECTION_REJECTED,
                "The projected split topology contains invalid or fabricated state.",
            )

        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before Split Parent was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before Split Parent was committed.",
            ) from exc
        if (
            commit_receipt.edit_ids != (edit.edit_id,)
            or commit_receipt.artifact_revision_ids
        ):
            raise SplitUserParentCommandError(
                SplitUserParentCommandErrorCode.PROJECTION_REJECTED,
                "Split Parent must publish one edit and zero artifacts.",
            )
        return SplitUserParentCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            source_parent_id=command.source_parent_id,
            source_root_id=source_parent.root_id,
            source_role=source_parent.role,
            source_workflow_area_bbox=source_bbox,
            orientation=command.orientation,
            split_offset=command.split_offset,
            child_parent_ids=child_parent_ids,
            child_root_ids=child_root_ids,
            child_workflow_area_bboxes=child_bboxes,
            canvas_size=canvas_size,
            before_hierarchy_revision_id=before_page.hierarchy.revision_id,
            after_hierarchy_revision_id=after_page.hierarchy.revision_id,
            before_hierarchy_fingerprint=before_page.hierarchy.fingerprint,
            after_hierarchy_fingerprint=after_page.hierarchy.fingerprint,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            child_stage_requirements=(
                child_requirements[0],
                child_requirements[1],
            ),
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


def merged_pipeline_parent_bbox(
    source_bboxes: tuple[
        tuple[int, int, int, int],
        tuple[int, int, int, int],
    ],
) -> tuple[int, int, int, int]:
    """Return the canonical enclosing integer bbox for two pipeline parents."""

    if not isinstance(source_bboxes, tuple) or len(source_bboxes) != 2:
        raise ValueError("source_bboxes must contain exactly two bboxes")
    first = _exact_bbox(source_bboxes[0], "source_bboxes[0]")
    second = _exact_bbox(source_bboxes[1], "source_bboxes[1]")
    left = min(first[0], second[0])
    top = min(first[1], second[1])
    right = max(first[0] + first[2], second[0] + second[2])
    bottom = max(first[1] + first[3], second[1] + second[3])
    return (left, top, right - left, bottom - top)


class MergePipelineParentsCommandService:
    """Persist one pipeline-backed merge without invoking any pipeline owner."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: MergePipelineParentsCommand,
    ) -> MergePipelineParentsCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, MergePipelineParentsCommand):
            raise TypeError("command must be a MergePipelineParentsCommand")
        if self._edit_store.project_id != command.project_id:
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )
        snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        return self.execute_materialized(snapshot=snapshot, command=command)

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditReadSnapshot,
        command: MergePipelineParentsCommand,
    ) -> MergePipelineParentsCommandReceipt:
        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, MergePipelineParentsCommand):
            raise TypeError("command must be a MergePipelineParentsCommand")
        if self._edit_store.project_id != command.project_id:
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        materialized = snapshot.project
        if (
            project_id_for(materialized) != command.project_id
            or snapshot.ledger.project_id != command.project_id
        ):
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after Merge Parent was prepared.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after Merge Parent was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.DUPLICATE_COMMAND,
                "The Merge Parent command was already recorded.",
            )
        if any(
            edit.domain is EditDomain.STRUCTURAL
            and edit.operation == MergePipelineParentsOperation.MERGE.value
            and set(
                str(value)
                for value in edit.payload.get("source_parent_ids") or ()
            ).intersection(command.source_parent_ids)
            for edit in snapshot.ledger.active_edits(page_id=command.page_id)
        ):
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.MERGE_SLOT_CONFLICT,
                "A selected pipeline parent already belongs to an active merge.",
            )
        try:
            before_page = project_effective_page(
                materialized,
                snapshot.ledger,
                page_id=command.page_id,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.PROJECTION_REJECTED,
                "The effective page could not be projected before Merge Parent.",
            ) from exc
        if (
            before_page.effective_fingerprint
            != command.expected_effective_page_fingerprint
        ):
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after Merge Parent was prepared.",
            )

        source_parents: list[EffectiveParentSnapshot] = []
        for source_parent_id in command.source_parent_ids:
            matches = tuple(
                parent
                for parent in before_page.parents
                if parent.parent_id == source_parent_id
            )
            if len(matches) != 1:
                raise MergePipelineParentsCommandError(
                    MergePipelineParentsCommandErrorCode.SOURCE_PARENT_NOT_FOUND,
                    "A selected pipeline parent is unavailable.",
                )
            source_parent = matches[0]
            if source_parent.excluded:
                raise MergePipelineParentsCommandError(
                    MergePipelineParentsCommandErrorCode.SOURCE_PARENT_EXCLUDED,
                    "Restore both selected pipeline parents before merging them.",
                )
            if (
                source_parent.origin is not ParentOrigin.AUTOMATIC
                or source_parent.automatic_fingerprint is None
                or source_parent.lineage is not None
            ):
                raise MergePipelineParentsCommandError(
                    MergePipelineParentsCommandErrorCode.SOURCE_PARENT_NOT_AUTOMATIC,
                    "Merge Parent requires two immutable pipeline parents.",
                )
            parent_local_edit_ids = tuple(
                edit_id
                for edit_id in source_parent.applied_edit_ids
                if (
                    (record := snapshot.ledger.get(edit_id)) is not None
                    and record.target.kind is EditTargetKind.PARENT
                    and record.target.parent_id == source_parent.parent_id
                )
            )
            if parent_local_edit_ids or source_parent.render_override_edit_ids:
                raise MergePipelineParentsCommandError(
                    MergePipelineParentsCommandErrorCode.SOURCE_PARENT_EDITED,
                    "Revoke parent-local edits in History before merging this pipeline parent.",
                )
            source_parents.append(source_parent)
        first_parent, second_parent = source_parents
        if (
            first_parent.role not in {"speech", "caption"}
            or second_parent.role != first_parent.role
        ):
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.ROLE_MISMATCH,
                "Merge Parent requires two pipeline parents with the same role.",
            )
        predecessor_order = before_page.hierarchy.ordered_parent_ids
        try:
            first_index = predecessor_order.index(command.source_parent_ids[0])
        except ValueError as exc:
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.SOURCE_PARENT_NOT_FOUND,
                "A selected pipeline parent is absent from effective order.",
            ) from exc
        if (
            first_index + 1 >= len(predecessor_order)
            or predecessor_order[first_index + 1] != command.source_parent_ids[1]
        ):
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.SOURCES_NOT_CONSECUTIVE,
                "Merge Parent requires consecutive parents in effective reading order.",
            )
        try:
            source_bboxes = tuple(
                _exact_bbox(
                    tuple(thaw_json(parent.geometry)),
                    f"source_bboxes[{index}]",
                )
                for index, parent in enumerate(source_parents)
            )
        except (TypeError, ValueError) as exc:
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.SOURCE_EVIDENCE_UNAVAILABLE,
                "A selected pipeline parent does not expose an exact integer bbox.",
            ) from exc
        source_texts = tuple(parent.source_text for parent in source_parents)
        if any(
            not isinstance(text, str) or not text or not text.strip()
            for text in source_texts
        ):
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.SOURCE_EVIDENCE_UNAVAILABLE,
                "A selected pipeline parent does not expose non-empty OCR text.",
            )
        source_root_ids = tuple(parent.root_id for parent in source_parents)
        if any(not root_id for root_id in source_root_ids):
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.SOURCE_EVIDENCE_UNAVAILABLE,
                "A selected pipeline parent does not expose root identity.",
            )
        source_automatic_fingerprints = tuple(
            str(parent.automatic_fingerprint) for parent in source_parents
        )
        source_text_fingerprints = tuple(
            effective_source_fingerprint(parent.parent_id, str(parent.source_text))
            for parent in source_parents
        )
        source_target_texts = tuple(
            str(parent.target_text or "") for parent in source_parents
        )
        source_target_text_fingerprints = tuple(
            canonical_sha256(
                {
                    "parent_id": parent.parent_id,
                    "target_text": target_text,
                }
            )
            for parent, target_text in zip(source_parents, source_target_texts)
        )
        try:
            page = _add_user_parent_project_page(materialized, command.page_id)
            canvas_size = page_canvas_size_for_project_page(
                page,
                project_path=self._edit_store.project_path,
            )
        except AddUserParentCommandError as exc:
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.PAGE_NOT_FOUND,
                "Project page identity is unavailable for Merge Parent.",
            ) from exc
        except ParentGeometryCommandError as exc:
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.CANVAS_UNAVAILABLE,
                "Page canvas dimensions are unavailable for Merge Parent.",
            ) from exc
        merged_bbox = merged_pipeline_parent_bbox(source_bboxes)  # type: ignore[arg-type]
        if (
            merged_bbox[0] + merged_bbox[2] > canvas_size[0]
            or merged_bbox[1] + merged_bbox[3] > canvas_size[1]
        ):
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.SOURCE_EVIDENCE_UNAVAILABLE,
                "Merged pipeline bbox falls outside the page canvas.",
            )
        try:
            _require_user_parent_identities_available(
                materialized,
                snapshot.ledger,
                parent_id=command.merged_parent_id,
                root_id=command.merged_root_id,
            )
        except AddUserParentCommandError as exc:
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.IDENTITY_COLLISION,
                "The merged user parent identity is already reserved.",
            ) from exc
        merged_source_text = "".join(str(text) for text in source_texts)
        expected_source_mapping = ParentSourceEvidenceMappingV1(
            page_id=command.page_id,
            source_parent_ids=command.source_parent_ids,
            source_root_ids=source_root_ids,
            source_bundle_ids=tuple(str(parent.bundle_id) for parent in source_parents),
            source_automatic_fingerprints=source_automatic_fingerprints,
            source_bboxes=source_bboxes,  # type: ignore[arg-type]
            source_texts=tuple(str(text) for text in source_texts),
            source_text_fingerprints=source_text_fingerprints,
            source_target_texts=source_target_texts,
            source_target_text_fingerprints=source_target_text_fingerprints,
            source_reading_orders=tuple(
                parent.reading_order for parent in source_parents
            ),
            source_roles=tuple(parent.role for parent in source_parents),
            primary_source_parent_id=command.source_parent_ids[0],
        )
        merged_target_text = expected_source_mapping.target_text
        payload: Mapping[str, Any] = {
            "identity_namespace": USER_PARENT_IDENTITY_NAMESPACE,
            "root_identity_namespace": USER_ROOT_IDENTITY_NAMESPACE,
            "merged_root_id": command.merged_root_id,
            "source_parent_ids": list(command.source_parent_ids),
            "source_root_ids": list(source_root_ids),
            "source_automatic_fingerprints": list(
                source_automatic_fingerprints
            ),
            "source_bboxes": [list(bbox) for bbox in source_bboxes],
            "source_texts": list(source_texts),
            "source_text_fingerprints": list(source_text_fingerprints),
            "source_role": first_parent.role,
            "merged_workflow_area_bbox": list(merged_bbox),
            "merged_source_text": merged_source_text,
            "canvas_size": list(canvas_size),
            "predecessor_ordered_parent_ids": list(predecessor_order),
            "order_policy": "replace_sources",
        }
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=EditTarget(
                EditTargetKind.PARENT,
                parent_id=command.merged_parent_id,
            ),
            domain=EditDomain.STRUCTURAL,
            operation=command.operation.value,
            payload=payload,
            base_revision_id=before_page.hierarchy.revision_id,
            base_fingerprint=before_page.hierarchy.fingerprint,
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        _require_exact_merge_pipeline_parents_invalidation(
            invalidation,
            page_id=command.page_id,
            merged_parent_id=command.merged_parent_id,
        )
        try:
            candidate_ledger = snapshot.ledger.append(edit)
            after_page = project_effective_page(
                materialized,
                candidate_ledger,
                page_id=command.page_id,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector rejected Merge Parent.",
            ) from exc
        projected_merged = next(
            (
                parent
                for parent in after_page.parents
                if parent.parent_id == command.merged_parent_id
            ),
            None,
        )
        expected_order = (
            *predecessor_order[:first_index],
            command.merged_parent_id,
            *predecessor_order[first_index + 2 :],
        )
        unrelated_before = {
            parent.parent_id: replace(parent, reading_order=0)
            for parent in before_page.parents
            if parent.parent_id not in set(command.source_parent_ids)
        }
        unrelated_after = {
            parent.parent_id: replace(parent, reading_order=0)
            for parent in after_page.parents
            if parent.parent_id != command.merged_parent_id
        }
        lineage = projected_merged.lineage if projected_merged is not None else None
        if (
            projected_merged is None
            or lineage is None
            or projected_merged.origin is not ParentOrigin.USER
            or projected_merged.root_id != command.merged_root_id
            or projected_merged.role != first_parent.role
            or projected_merged.bundle_id is not None
            or projected_merged.automatic_fingerprint is not None
            or thaw_json(projected_merged.automatic_geometry) is not None
            or tuple(thaw_json(projected_merged.geometry)) != merged_bbox
            or tuple(thaw_json(projected_merged.workflow_area_bbox)) != merged_bbox
            or thaw_json(projected_merged.render_allowed_area) is not None
            or thaw_json(projected_merged.root_bbox) is not None
            or projected_merged.source_text != merged_source_text
            or projected_merged.source_authority != "user"
            or projected_merged.source_evidence_mapping != expected_source_mapping
            or projected_merged.target_text != merged_target_text
            or projected_merged.target_authority
            != ("mapped_automatic" if merged_target_text is not None else "unavailable")
            or projected_merged.target_freshness
            is not (
                TargetFreshness.CURRENT
                if merged_target_text is not None
                else TargetFreshness.UNAVAILABLE
            )
            or projected_merged.automatic_render_style
            or projected_merged.render_style_overrides
            or projected_merged.automatic_render_layout
            or projected_merged.render_layout_overrides
            or lineage.order_policy != "replace_sources"
            or lineage.source_parent_ids != command.source_parent_ids
            or lineage.source_root_ids != source_root_ids
            or lineage.source_automatic_fingerprints
            != source_automatic_fingerprints
            or after_page.hierarchy.ordered_parent_ids != expected_order
            or unrelated_after != unrelated_before
            or edit.edit_id not in after_page.applied_edit_ids
            or edit.edit_id
            not in after_page.hierarchy.descriptor.active_structural_edit_ids
            or any(edit.edit_id in issue.edit_ids for issue in after_page.issues)
        ):
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.PROJECTION_REJECTED,
                "The projected merged topology contains invalid or fabricated state.",
            )
        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before Merge Parent was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before Merge Parent was committed.",
            ) from exc
        if (
            commit_receipt.edit_ids != (edit.edit_id,)
            or commit_receipt.artifact_revision_ids
        ):
            raise MergePipelineParentsCommandError(
                MergePipelineParentsCommandErrorCode.PROJECTION_REJECTED,
                "Merge Parent must publish one edit and zero artifacts.",
            )
        return MergePipelineParentsCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            source_parent_ids=command.source_parent_ids,
            source_root_ids=source_root_ids,  # type: ignore[arg-type]
            source_automatic_fingerprints=source_automatic_fingerprints,  # type: ignore[arg-type]
            source_bboxes=source_bboxes,  # type: ignore[arg-type]
            source_texts=source_texts,  # type: ignore[arg-type]
            source_text_fingerprints=source_text_fingerprints,  # type: ignore[arg-type]
            source_role=first_parent.role,
            merged_parent_id=command.merged_parent_id,
            merged_root_id=command.merged_root_id,
            merged_workflow_area_bbox=merged_bbox,
            merged_source_text=merged_source_text,
            canvas_size=canvas_size,
            before_hierarchy_revision_id=before_page.hierarchy.revision_id,
            after_hierarchy_revision_id=after_page.hierarchy.revision_id,
            before_hierarchy_fingerprint=before_page.hierarchy.fingerprint,
            after_hierarchy_fingerprint=after_page.hierarchy.fingerprint,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            stage_requirements=projected_merged.stage_requirements,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


def _add_user_parent_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise AddUserParentCommandError(
            AddUserParentCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = tuple(
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    )
    if len(matches) != 1:
        raise AddUserParentCommandError(
            AddUserParentCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is not exact: {page_id}",
        )
    return matches[0]


def _require_user_parent_identities_available(
    project: Mapping[str, Any],
    ledger: ProjectEditLedger,
    *,
    parent_id: str,
    root_id: str,
) -> None:
    reserved_parent_ids: set[str] = set()
    reserved_root_ids: set[str] = set()
    for page in project.get("pages") or ():
        if not isinstance(page, Mapping):
            continue
        bundles = page.get("parent_execution_bundles") or ()
        if not isinstance(bundles, (list, tuple)):
            continue
        for bundle in bundles:
            if not isinstance(bundle, Mapping):
                continue
            automatic_parent_id = str(bundle.get("parent_id") or "").strip()
            automatic_root_id = str(
                bundle.get("root_id") or bundle.get("text_block_root_id") or ""
            ).strip()
            if automatic_parent_id:
                reserved_parent_ids.add(automatic_parent_id)
            if automatic_root_id:
                reserved_root_ids.add(automatic_root_id)
    for record in ledger.edits:
        if (
            record.is_control
            or record.domain is not EditDomain.STRUCTURAL
            or record.target.kind is not EditTargetKind.PARENT
        ):
            continue
        if record.operation == AddUserParentOperation.ADD.value:
            reserved_parent_ids.add(record.target.parent_id)
            reserved_root_ids.add(str(record.payload.get("root_id") or ""))
        elif record.operation == SplitUserParentOperation.SPLIT.value:
            reserved_parent_ids.update(
                str(value)
                for value in record.payload.get("child_parent_ids") or ()
            )
            reserved_root_ids.update(
                str(value)
                for value in record.payload.get("child_root_ids") or ()
            )
        elif record.operation == MergePipelineParentsOperation.MERGE.value:
            reserved_parent_ids.add(record.target.parent_id)
            reserved_root_ids.add(
                str(record.payload.get("merged_root_id") or "")
            )
    if parent_id in reserved_parent_ids or root_id in reserved_root_ids:
        raise AddUserParentCommandError(
            AddUserParentCommandErrorCode.IDENTITY_COLLISION,
            "User parent/root identity is already reserved in this project.",
        )


def _require_exact_add_user_parent_invalidation(
    invalidation: InvalidationResult,
    *,
    page_id: str,
    parent_id: str,
) -> None:
    expected = {
        (Dependency.HIERARCHY, InvalidationAction.NEW_REVISION, InvalidationScope.PAGE, (page_id,)),
        (Dependency.SOURCE, InvalidationAction.RERUN, InvalidationScope.PARENT, (parent_id,)),
        (Dependency.TRANSLATION, InvalidationAction.RERUN, InvalidationScope.PARENT, (parent_id,)),
        (Dependency.CLEANUP_BASE, InvalidationAction.REBUILD, InvalidationScope.PAGE, (page_id,)),
        (Dependency.STYLE_CACHE, InvalidationAction.RERUN, InvalidationScope.STYLE_CACHE_PREFIX, (page_id,)),
        (Dependency.RENDER_ELIGIBILITY, InvalidationAction.RERUN, InvalidationScope.PARENT, (parent_id,)),
        (Dependency.LAYOUT_RENDER, InvalidationAction.RECOMPUTE, InvalidationScope.PARENT, (parent_id,)),
        (Dependency.PAGE_OUTPUT, InvalidationAction.RECOMPUTE, InvalidationScope.PAGE, (page_id,)),
    }
    actual = {
        (effect.dependency, effect.action, effect.scope, effect.target_ids)
        for effect in invalidation.effects
    }
    if invalidation.unresolved_facts or actual != expected:
        raise AddUserParentCommandError(
            AddUserParentCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Add Parent invalidation must contain exactly eight resolved effects.",
        )


def _require_exact_split_user_parent_invalidation(
    invalidation: InvalidationResult,
    *,
    page_id: str,
    child_parent_ids: tuple[str, str],
) -> None:
    expected = {
        (
            Dependency.HIERARCHY,
            InvalidationAction.NEW_REVISION,
            InvalidationScope.PAGE,
            (page_id,),
        ),
        (
            Dependency.CLEANUP_BASE,
            InvalidationAction.REBUILD,
            InvalidationScope.PAGE,
            (page_id,),
        ),
        (
            Dependency.STYLE_CACHE,
            InvalidationAction.RERUN,
            InvalidationScope.STYLE_CACHE_PREFIX,
            (page_id,),
        ),
        (
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            (page_id,),
        ),
    }
    for child_parent_id in child_parent_ids:
        expected.update(
            {
                (
                    Dependency.SOURCE,
                    InvalidationAction.RERUN,
                    InvalidationScope.PARENT,
                    (child_parent_id,),
                ),
                (
                    Dependency.TRANSLATION,
                    InvalidationAction.RERUN,
                    InvalidationScope.PARENT,
                    (child_parent_id,),
                ),
                (
                    Dependency.RENDER_ELIGIBILITY,
                    InvalidationAction.RERUN,
                    InvalidationScope.PARENT,
                    (child_parent_id,),
                ),
                (
                    Dependency.LAYOUT_RENDER,
                    InvalidationAction.RECOMPUTE,
                    InvalidationScope.PARENT,
                    (child_parent_id,),
                ),
            }
        )
    actual = {
        (effect.dependency, effect.action, effect.scope, effect.target_ids)
        for effect in invalidation.effects
    }
    if invalidation.unresolved_facts or actual != expected:
        raise SplitUserParentCommandError(
            SplitUserParentCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Split Parent invalidation must contain exactly twelve resolved effects.",
        )


def _require_exact_merge_pipeline_parents_invalidation(
    invalidation: InvalidationResult,
    *,
    page_id: str,
    merged_parent_id: str,
) -> None:
    expected = {
        (
            Dependency.HIERARCHY,
            InvalidationAction.NEW_REVISION,
            InvalidationScope.PAGE,
            (page_id,),
        ),
        (
            Dependency.SOURCE,
            InvalidationAction.KEEP,
            InvalidationScope.PARENT,
            (merged_parent_id,),
        ),
        (
            Dependency.TRANSLATION,
            InvalidationAction.RERUN,
            InvalidationScope.PARENT,
            (merged_parent_id,),
        ),
        (
            Dependency.CLEANUP_BASE,
            InvalidationAction.REBUILD,
            InvalidationScope.PAGE,
            (page_id,),
        ),
        (
            Dependency.STYLE_CACHE,
            InvalidationAction.RERUN,
            InvalidationScope.STYLE_CACHE_PREFIX,
            (page_id,),
        ),
        (
            Dependency.RENDER_ELIGIBILITY,
            InvalidationAction.RERUN,
            InvalidationScope.PARENT,
            (merged_parent_id,),
        ),
        (
            Dependency.LAYOUT_RENDER,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PARENT,
            (merged_parent_id,),
        ),
        (
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            (page_id,),
        ),
    }
    actual = {
        (effect.dependency, effect.action, effect.scope, effect.target_ids)
        for effect in invalidation.effects
    }
    if invalidation.unresolved_facts or actual != expected:
        raise MergePipelineParentsCommandError(
            MergePipelineParentsCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Merge Parent invalidation must contain exactly eight resolved effects.",
        )


class ParentGeometryCommandService:
    """Persist one selected-parent geometry edit through GUI-owned owners only."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: ParentGeometryCommand,
    ) -> ParentGeometryCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, ParentGeometryCommand):
            raise TypeError("command must be a ParentGeometryCommand")
        if self._edit_store.project_id != command.project_id:
            raise ParentGeometryCommandError(
                ParentGeometryCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise ParentGeometryCommandError(
                ParentGeometryCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )

        read_snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        materialized = read_snapshot.project
        if project_id_for(materialized) != command.project_id:
            raise ParentGeometryCommandError(
                ParentGeometryCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if read_snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise ParentGeometryCommandError(
                ParentGeometryCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the command was prepared.",
            )
        if read_snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise ParentGeometryCommandError(
                ParentGeometryCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the command was prepared.",
            )
        if read_snapshot.ledger.get(command.command_id) is not None:
            raise ParentGeometryCommandError(
                ParentGeometryCommandErrorCode.DUPLICATE_COMMAND,
                "The parent-geometry command was already recorded.",
            )

        page = _geometry_project_page(materialized, command.page_id)
        canvas_size = page_canvas_size_for_project_page(
            page,
            project_path=self._edit_store.project_path,
        )
        _validate_bbox_within_canvas(command.bbox, canvas_size)
        before_page = project_effective_page(
            materialized,
            read_snapshot.ledger,
            page_id=command.page_id,
        )
        if (
            before_page.effective_fingerprint
            != command.expected_effective_page_fingerprint
        ):
            raise ParentGeometryCommandError(
                ParentGeometryCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _geometry_effective_parent(
            before_page,
            command.parent_id,
        )
        try:
            before_bbox = _exact_bbox(
                thaw_json(before_parent.geometry),
                "effective parent geometry",
            )
        except (TypeError, ValueError) as exc:
            raise ParentGeometryCommandError(
                ParentGeometryCommandErrorCode.INVALID_GEOMETRY,
                "Existing effective parent geometry is not an exact pixel bbox.",
            ) from exc
        if before_bbox == command.bbox:
            raise ParentGeometryCommandError(
                ParentGeometryCommandErrorCode.NO_OP,
                "The requested parent geometry is already effective.",
            )
        slot_head = _active_geometry_slot_head(
            read_snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        target = EditTarget(
            EditTargetKind.PARENT,
            parent_id=command.parent_id,
        )
        payload: Mapping[str, Any] = {
            "bbox": list(command.bbox),
            "canvas_size": list(canvas_size),
        }
        base_fingerprint = field_base_fingerprint(
            project=materialized,
            page=page,
            target=target,
            domain=EditDomain.STRUCTURAL,
            operation=command.operation.value,
            payload=payload,
        )
        if base_fingerprint is None:
            raise ParentGeometryCommandError(
                ParentGeometryCommandErrorCode.PARENT_NOT_FOUND,
                "Automatic parent geometry evidence is unavailable.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.STRUCTURAL,
            operation=command.operation.value,
            payload=payload,
            base_revision_id=before_parent.base_revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        if invalidation.unresolved_facts:
            raise ParentGeometryCommandError(
                ParentGeometryCommandErrorCode.INVALIDATION_UNRESOLVED,
                "Parent geometry invalidation requires unresolved pipeline facts.",
            )

        candidate_ledger = read_snapshot.ledger.append(edit)
        after_page = project_effective_page(
            materialized,
            candidate_ledger,
            page_id=command.page_id,
        )
        after_parent = _geometry_effective_parent(
            after_page,
            command.parent_id,
        )
        try:
            after_bbox = _exact_bbox(
                thaw_json(after_parent.geometry),
                "projected parent geometry",
            )
        except (TypeError, ValueError) as exc:
            raise ParentGeometryCommandError(
                ParentGeometryCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector produced invalid parent geometry.",
            ) from exc
        if (
            edit.edit_id not in after_page.applied_edit_ids
            or after_bbox != command.bbox
        ):
            raise ParentGeometryCommandError(
                ParentGeometryCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested parent geometry.",
            )

        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=read_snapshot.page_head_sha256,
                expected_global_head_sha256=read_snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise ParentGeometryCommandError(
                ParentGeometryCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before parent geometry was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise ParentGeometryCommandError(
                ParentGeometryCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before parent geometry was committed.",
            ) from exc

        return ParentGeometryCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            before_bbox=before_bbox,
            after_bbox=after_bbox,
            canvas_size=canvas_size,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


class ReadingOrderCommandService:
    """Persist one complete page reading-order permutation."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: ReadingOrderCommand,
    ) -> ReadingOrderCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, ReadingOrderCommand):
            raise TypeError("command must be a ReadingOrderCommand")
        if self._edit_store.project_id != command.project_id:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )
        snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        return self.execute_materialized(snapshot=snapshot, command=command)

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditReadSnapshot,
        command: ReadingOrderCommand,
    ) -> ReadingOrderCommandReceipt:
        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, ReadingOrderCommand):
            raise TypeError("command must be a ReadingOrderCommand")
        if self._edit_store.project_id != command.project_id:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        materialized = snapshot.project
        if project_id_for(materialized) != command.project_id:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if snapshot.ledger.project_id != command.project_id:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized ledger identity does not match the command.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the reading-order command was prepared.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the reading-order command was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.DUPLICATE_COMMAND,
                "The reading-order command was already recorded.",
            )

        page = _reading_order_project_page(materialized, command.page_id)
        try:
            automatic_order = automatic_ordered_parent_ids_for_page(page)
        except (TypeError, ValueError) as exc:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.AUTOMATIC_ORDER_UNAVAILABLE,
                "Automatic parent reading order is not exact and unique.",
            ) from exc
        try:
            before_page = project_effective_page(
                materialized,
                snapshot.ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.PROJECTION_REJECTED,
                "The effective page could not be projected before reordering.",
            ) from exc
        if (
            before_page.effective_fingerprint
            != command.expected_effective_page_fingerprint
        ):
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )

        before_order = before_page.hierarchy.ordered_parent_ids
        proposed_order = command.ordered_parent_ids
        if (
            len(proposed_order) != len(before_order)
            or len(set(proposed_order)) != len(proposed_order)
            or frozenset(proposed_order) != frozenset(before_order)
        ):
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.INVALID_PERMUTATION,
                "Reading order must contain every current page parent exactly once.",
            )
        parent_by_id = {parent.parent_id: parent for parent in before_page.parents}
        selected_parent = parent_by_id.get(command.selected_parent_id)
        if selected_parent is None:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.PARENT_NOT_FOUND,
                "The selected parent is unavailable on this page.",
            )
        if selected_parent.excluded:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.SELECTED_PARENT_EXCLUDED,
                "Excluded parents cannot be moved in reading order.",
            )
        if proposed_order == before_order:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.NO_OP,
                "The requested reading order is already effective.",
            )
        for index, parent_id in enumerate(before_order):
            parent = parent_by_id[parent_id]
            if parent.excluded and proposed_order[index] != parent_id:
                raise ReadingOrderCommandError(
                    ReadingOrderCommandErrorCode.EXCLUDED_PARENT_MOVED,
                    "Excluded parents must remain in their existing absolute slots.",
                )
        before_other_active = tuple(
            parent_id
            for parent_id in before_order
            if parent_id != command.selected_parent_id
            and not parent_by_id[parent_id].excluded
        )
        proposed_other_active = tuple(
            parent_id
            for parent_id in proposed_order
            if parent_id != command.selected_parent_id
            and not parent_by_id[parent_id].excluded
        )
        if before_other_active != proposed_other_active:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.MULTIPLE_PARENTS_MOVED,
                "A reading-order command may move only its selected active parent.",
            )

        slot_head = _active_reading_order_slot_head(
            snapshot.ledger,
            page_id=command.page_id,
        )
        target = EditTarget(EditTargetKind.PAGE)
        payload: Mapping[str, Any] = {
            "selected_parent_id": command.selected_parent_id,
            "ordered_parent_ids": list(proposed_order),
        }
        base_fingerprint = before_page.hierarchy.fingerprint
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.STRUCTURAL,
            operation=ReadingOrderOperation.SET.value,
            payload=payload,
            base_revision_id=before_page.hierarchy.revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        _require_exact_reading_order_invalidation(
            invalidation,
            page_id=command.page_id,
        )

        try:
            candidate_ledger = snapshot.ledger.append(edit)
            after_page = project_effective_page(
                materialized,
                candidate_ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector rejected the page permutation.",
            ) from exc
        after_order = after_page.hierarchy.ordered_parent_ids
        if (
            edit.edit_id not in after_page.applied_edit_ids
            or after_order != proposed_order
        ):
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested page order.",
            )

        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before reading order was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before reading order was committed.",
            ) from exc

        return ReadingOrderCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            selected_parent_id=command.selected_parent_id,
            automatic_ordered_parent_ids=automatic_order,
            before_ordered_parent_ids=before_order,
            after_ordered_parent_ids=after_order,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


class RenderLayoutWritingModeCommandService:
    """Persist one canonical writing-mode edit through GUI-owned owners only."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: RenderLayoutWritingModeCommand,
    ) -> RenderLayoutWritingModeCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, RenderLayoutWritingModeCommand):
            raise TypeError(
                "command must be a RenderLayoutWritingModeCommand"
            )
        if self._edit_store.project_id != command.project_id:
            raise RenderLayoutWritingModeCommandError(
                RenderLayoutWritingModeCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise RenderLayoutWritingModeCommandError(
                RenderLayoutWritingModeCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )

        read_snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        materialized = read_snapshot.project
        if project_id_for(materialized) != command.project_id:
            raise RenderLayoutWritingModeCommandError(
                RenderLayoutWritingModeCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if read_snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise RenderLayoutWritingModeCommandError(
                RenderLayoutWritingModeCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the command was prepared.",
            )
        if read_snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise RenderLayoutWritingModeCommandError(
                RenderLayoutWritingModeCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the command was prepared.",
            )
        if read_snapshot.ledger.get(command.command_id) is not None:
            raise RenderLayoutWritingModeCommandError(
                RenderLayoutWritingModeCommandErrorCode.DUPLICATE_COMMAND,
                "The writing-mode command was already recorded.",
            )

        page = _writing_mode_project_page(materialized, command.page_id)
        before_page = project_effective_page(
            materialized,
            read_snapshot.ledger,
            page_id=command.page_id,
        )
        if (
            before_page.effective_fingerprint
            != command.expected_effective_page_fingerprint
        ):
            raise RenderLayoutWritingModeCommandError(
                RenderLayoutWritingModeCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _writing_mode_effective_parent(
            before_page,
            command.parent_id,
        )
        automatic_parent = _writing_mode_automatic_parent(
            page,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        if before_parent.excluded:
            raise RenderLayoutWritingModeCommandError(
                RenderLayoutWritingModeCommandErrorCode.PARENT_EXCLUDED,
                "Writing mode is unavailable while the parent is excluded.",
            )
        if automatic_parent.get("render_required") is not True:
            raise RenderLayoutWritingModeCommandError(
                RenderLayoutWritingModeCommandErrorCode.PARENT_NOT_RENDER_REQUIRED,
                "Writing mode is available only for render-required parents.",
            )
        automatic_writing_mode = automatic_render_writing_mode(automatic_parent)
        if automatic_writing_mode is None:
            raise RenderLayoutWritingModeCommandError(
                RenderLayoutWritingModeCommandErrorCode.AUTOMATIC_WRITING_MODE_UNAVAILABLE,
                "The automatic render writing mode is unavailable or noncanonical.",
            )
        slot_head = _active_writing_mode_slot_head(
            read_snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        before_writing_mode, before_authority = _effective_writing_mode_state(
            before_parent,
            automatic_writing_mode=automatic_writing_mode,
        )
        if command.operation is RenderLayoutWritingModeOperation.SET:
            if command.writing_mode == before_writing_mode:
                raise RenderLayoutWritingModeCommandError(
                    RenderLayoutWritingModeCommandErrorCode.NO_OP,
                    "The requested writing mode is already effective.",
                )
            operation = "set_fields"
            payload: Mapping[str, Any] = {
                "fields": {"writing_mode": command.writing_mode}
            }
        else:
            if slot_head is None or slot_head.operation == "restore_automatic":
                raise RenderLayoutWritingModeCommandError(
                    RenderLayoutWritingModeCommandErrorCode.NO_OP,
                    "Writing mode already uses the automatic value.",
                )
            operation = "restore_automatic"
            payload = {"fields": ("writing_mode",)}

        target = EditTarget(
            EditTargetKind.PARENT,
            parent_id=command.parent_id,
        )
        base_fingerprint = field_base_fingerprint(
            project=materialized,
            page=page,
            target=target,
            domain=EditDomain.RENDER_LAYOUT,
            operation=operation,
            payload=payload,
        )
        if base_fingerprint is None:
            raise RenderLayoutWritingModeCommandError(
                RenderLayoutWritingModeCommandErrorCode.AUTOMATIC_WRITING_MODE_UNAVAILABLE,
                "The automatic render writing mode is unavailable for fingerprinting.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.RENDER_LAYOUT,
            operation=operation,
            payload=payload,
            base_revision_id=before_parent.base_revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        _require_exact_writing_mode_invalidation(
            invalidation,
            parent_id=command.parent_id,
        )

        candidate_ledger = read_snapshot.ledger.append(edit)
        after_page = project_effective_page(
            materialized,
            candidate_ledger,
            page_id=command.page_id,
        )
        after_parent = _writing_mode_effective_parent(
            after_page,
            command.parent_id,
        )
        after_writing_mode, after_authority = _effective_writing_mode_state(
            after_parent,
            automatic_writing_mode=automatic_writing_mode,
        )
        if command.operation is RenderLayoutWritingModeOperation.SET:
            accepted = (
                after_writing_mode == command.writing_mode
                and after_authority == "user"
            )
        else:
            accepted = (
                after_writing_mode == automatic_writing_mode
                and after_authority == "automatic"
            )
        if edit.edit_id not in after_page.applied_edit_ids or not accepted:
            raise RenderLayoutWritingModeCommandError(
                RenderLayoutWritingModeCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested writing mode.",
            )

        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=read_snapshot.page_head_sha256,
                expected_global_head_sha256=read_snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise RenderLayoutWritingModeCommandError(
                RenderLayoutWritingModeCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before writing mode was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise RenderLayoutWritingModeCommandError(
                RenderLayoutWritingModeCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before writing mode was committed.",
            ) from exc

        return RenderLayoutWritingModeCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            automatic_writing_mode=automatic_writing_mode,
            before_writing_mode=before_writing_mode,
            after_writing_mode=after_writing_mode,
            before_writing_mode_authority=before_authority,
            after_writing_mode_authority=after_authority,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


class RenderLayoutLineHeightCommandService:
    """Persist one canonical line-height edit through GUI-owned owners only."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: RenderLayoutLineHeightCommand,
    ) -> RenderLayoutLineHeightCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, RenderLayoutLineHeightCommand):
            raise TypeError("command must be a RenderLayoutLineHeightCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderLayoutLineHeightCommandError(
                RenderLayoutLineHeightCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise RenderLayoutLineHeightCommandError(
                RenderLayoutLineHeightCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )

        read_snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        materialized = read_snapshot.project
        if project_id_for(materialized) != command.project_id:
            raise RenderLayoutLineHeightCommandError(
                RenderLayoutLineHeightCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if read_snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise RenderLayoutLineHeightCommandError(
                RenderLayoutLineHeightCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the command was prepared.",
            )
        if read_snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise RenderLayoutLineHeightCommandError(
                RenderLayoutLineHeightCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the command was prepared.",
            )
        if read_snapshot.ledger.get(command.command_id) is not None:
            raise RenderLayoutLineHeightCommandError(
                RenderLayoutLineHeightCommandErrorCode.DUPLICATE_COMMAND,
                "The line-height command was already recorded.",
            )

        page = _line_height_project_page(materialized, command.page_id)
        before_page = project_effective_page(
            materialized,
            read_snapshot.ledger,
            page_id=command.page_id,
        )
        if (
            before_page.effective_fingerprint
            != command.expected_effective_page_fingerprint
        ):
            raise RenderLayoutLineHeightCommandError(
                RenderLayoutLineHeightCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _line_height_effective_parent(
            before_page,
            command.parent_id,
        )
        automatic_parent = _line_height_automatic_parent(
            page,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        if before_parent.excluded:
            raise RenderLayoutLineHeightCommandError(
                RenderLayoutLineHeightCommandErrorCode.PARENT_EXCLUDED,
                "Line height is unavailable while the parent is excluded.",
            )
        if automatic_parent.get("render_required") is not True:
            raise RenderLayoutLineHeightCommandError(
                RenderLayoutLineHeightCommandErrorCode.PARENT_NOT_RENDER_REQUIRED,
                "Line height is available only for render-required parents.",
            )
        automatic_line_height = automatic_render_line_height(automatic_parent)
        if automatic_line_height is None:
            raise RenderLayoutLineHeightCommandError(
                RenderLayoutLineHeightCommandErrorCode.AUTOMATIC_LINE_HEIGHT_UNAVAILABLE,
                "The automatic render line height is unavailable or invalid.",
            )
        slot_head = _active_line_height_slot_head(
            read_snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        before_line_height, before_authority = _effective_line_height_state(
            before_parent,
            automatic_line_height=automatic_line_height,
        )
        if command.operation is RenderLayoutLineHeightOperation.SET:
            if command.line_height == before_line_height:
                raise RenderLayoutLineHeightCommandError(
                    RenderLayoutLineHeightCommandErrorCode.NO_OP,
                    "The requested line height is already effective.",
                )
            operation = "set_fields"
            payload: Mapping[str, Any] = {
                "fields": {"line_height": command.line_height}
            }
        else:
            if (
                before_authority == "automatic"
                or slot_head is None
                or slot_head.operation == "restore_automatic"
            ):
                raise RenderLayoutLineHeightCommandError(
                    RenderLayoutLineHeightCommandErrorCode.NO_OP,
                    "Line height already uses the automatic value.",
                )
            operation = "restore_automatic"
            payload = {"fields": ("line_height",)}

        target = EditTarget(
            EditTargetKind.PARENT,
            parent_id=command.parent_id,
        )
        base_fingerprint = field_base_fingerprint(
            project=materialized,
            page=page,
            target=target,
            domain=EditDomain.RENDER_LAYOUT,
            operation=operation,
            payload=payload,
        )
        if base_fingerprint is None:
            raise RenderLayoutLineHeightCommandError(
                RenderLayoutLineHeightCommandErrorCode.AUTOMATIC_LINE_HEIGHT_UNAVAILABLE,
                "The automatic render line height is unavailable for fingerprinting.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.RENDER_LAYOUT,
            operation=operation,
            payload=payload,
            base_revision_id=before_parent.base_revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        _require_exact_line_height_invalidation(
            invalidation,
            parent_id=command.parent_id,
        )

        candidate_ledger = read_snapshot.ledger.append(edit)
        after_page = project_effective_page(
            materialized,
            candidate_ledger,
            page_id=command.page_id,
        )
        after_parent = _line_height_effective_parent(
            after_page,
            command.parent_id,
        )
        after_line_height, after_authority = _effective_line_height_state(
            after_parent,
            automatic_line_height=automatic_line_height,
        )
        if command.operation is RenderLayoutLineHeightOperation.SET:
            accepted = (
                after_line_height == command.line_height
                and after_authority == "user"
            )
        else:
            accepted = (
                after_line_height == automatic_line_height
                and after_authority == "automatic"
            )
        if edit.edit_id not in after_page.applied_edit_ids or not accepted:
            raise RenderLayoutLineHeightCommandError(
                RenderLayoutLineHeightCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested line height.",
            )

        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=read_snapshot.page_head_sha256,
                expected_global_head_sha256=read_snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise RenderLayoutLineHeightCommandError(
                RenderLayoutLineHeightCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before line height was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise RenderLayoutLineHeightCommandError(
                RenderLayoutLineHeightCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before line height was committed.",
            ) from exc

        return RenderLayoutLineHeightCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            automatic_line_height=automatic_line_height,
            before_line_height=before_line_height,
            after_line_height=after_line_height,
            before_line_height_authority=before_authority,
            after_line_height_authority=after_authority,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


class RenderLayoutRotationCommandService:
    """Persist one exact clockwise rotation through GUI-owned owners only."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: RenderLayoutRotationCommand,
    ) -> RenderLayoutRotationCommandReceipt:
        """Materialize once, then execute against that exact read snapshot."""

        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, RenderLayoutRotationCommand):
            raise TypeError("command must be a RenderLayoutRotationCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )
        snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        return self.execute_materialized(snapshot=snapshot, command=command)

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditReadSnapshot,
        command: RenderLayoutRotationCommand,
    ) -> RenderLayoutRotationCommandReceipt:
        """Execute from one worker-owned atomic project/ledger/head snapshot."""

        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, RenderLayoutRotationCommand):
            raise TypeError("command must be a RenderLayoutRotationCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        materialized = snapshot.project
        if project_id_for(materialized) != command.project_id:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if snapshot.ledger.project_id != command.project_id:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized ledger identity does not match the command.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the rotation command was prepared.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the rotation command was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.DUPLICATE_COMMAND,
                "The rotation command was already recorded.",
            )

        page = _rotation_project_page(materialized, command.page_id)
        try:
            before_page = project_effective_page(
                materialized,
                snapshot.ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.PROJECTION_REJECTED,
                "The effective page could not be projected before rotation editing.",
            ) from exc
        if (
            before_page.effective_fingerprint
            != command.expected_effective_page_fingerprint
        ):
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _rotation_effective_parent(
            before_page,
            command.parent_id,
        )
        automatic_parent = _rotation_automatic_parent(
            page,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        if before_parent.excluded:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.PARENT_EXCLUDED,
                "Rotation is unavailable while the parent is excluded.",
            )
        if automatic_parent.get("render_required") is not True:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.PARENT_NOT_RENDER_REQUIRED,
                "Rotation is available only for render-required parents.",
            )
        automatic_rotation = automatic_render_rotation(automatic_parent)
        if automatic_rotation is None:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.AUTOMATIC_ROTATION_UNAVAILABLE,
                "The automatic parent rotation is unavailable or invalid.",
            )
        slot_head = _active_rotation_slot_head(
            snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        before_rotation, before_authority = _effective_rotation_state(
            before_parent,
            automatic_rotation=automatic_rotation,
        )
        if command.operation is RenderLayoutRotationOperation.SET:
            if command.rotation == before_rotation:
                raise RenderLayoutRotationCommandError(
                    RenderLayoutRotationCommandErrorCode.NO_OP,
                    "The requested rotation is already effective.",
                )
            operation = "set_fields"
            payload: Mapping[str, Any] = {
                "fields": {"rotation": command.rotation}
            }
        else:
            if (
                before_authority == "automatic"
                or slot_head is None
                or slot_head.operation == "restore_automatic"
            ):
                raise RenderLayoutRotationCommandError(
                    RenderLayoutRotationCommandErrorCode.NO_OP,
                    "Rotation already uses the automatic value.",
                )
            operation = "restore_automatic"
            payload = {"fields": ("rotation",)}

        target = EditTarget(
            EditTargetKind.PARENT,
            parent_id=command.parent_id,
        )
        try:
            base_fingerprint = field_base_fingerprint(
                project=materialized,
                page=page,
                target=target,
                domain=EditDomain.RENDER_LAYOUT,
                operation=operation,
                payload=payload,
            )
        except (TypeError, ValueError) as exc:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.AUTOMATIC_ROTATION_UNAVAILABLE,
                "The automatic parent rotation is invalid for fingerprinting.",
            ) from exc
        if base_fingerprint is None:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.AUTOMATIC_ROTATION_UNAVAILABLE,
                "The automatic parent rotation is unavailable for fingerprinting.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.RENDER_LAYOUT,
            operation=operation,
            payload=payload,
            base_revision_id=before_parent.base_revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        _require_exact_rotation_invalidation(
            invalidation,
            parent_id=command.parent_id,
        )

        try:
            candidate_ledger = snapshot.ledger.append(edit)
            after_page = project_effective_page(
                materialized,
                candidate_ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector rejected the rotation edit.",
            ) from exc
        after_parent = _rotation_effective_parent(
            after_page,
            command.parent_id,
        )
        after_rotation, after_authority = _effective_rotation_state(
            after_parent,
            automatic_rotation=automatic_rotation,
        )
        if command.operation is RenderLayoutRotationOperation.SET:
            accepted = (
                after_rotation == command.rotation
                and after_authority == "user"
            )
        else:
            accepted = (
                after_rotation == automatic_rotation
                and after_authority == "automatic"
            )
        if edit.edit_id not in after_page.applied_edit_ids or not accepted:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested rotation.",
            )

        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before rotation was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise RenderLayoutRotationCommandError(
                RenderLayoutRotationCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before rotation was committed.",
            ) from exc

        return RenderLayoutRotationCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            automatic_rotation=automatic_rotation,
            before_rotation=before_rotation,
            after_rotation=after_rotation,
            before_rotation_authority=before_authority,
            after_rotation_authority=after_authority,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


class RenderStyleFillColorCommandService:
    """Persist one exact opaque fill color through GUI-owned owners only."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: RenderStyleFillColorCommand,
    ) -> RenderStyleFillColorCommandReceipt:
        """Materialize once, then execute against that exact read snapshot."""

        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, RenderStyleFillColorCommand):
            raise TypeError("command must be a RenderStyleFillColorCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )
        snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        return self.execute_materialized(snapshot=snapshot, command=command)

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditReadSnapshot,
        command: RenderStyleFillColorCommand,
    ) -> RenderStyleFillColorCommandReceipt:
        """Execute from one worker-owned atomic project/ledger/head snapshot."""

        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, RenderStyleFillColorCommand):
            raise TypeError("command must be a RenderStyleFillColorCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        materialized = snapshot.project
        if project_id_for(materialized) != command.project_id:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if snapshot.ledger.project_id != command.project_id:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized ledger identity does not match the command.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the fill-color command was prepared.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the fill-color command was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.DUPLICATE_COMMAND,
                "The fill-color command was already recorded.",
            )

        page = _fill_color_project_page(materialized, command.page_id)
        try:
            before_page = project_effective_page(
                materialized,
                snapshot.ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.PROJECTION_REJECTED,
                "The effective page could not be projected before fill-color editing.",
            ) from exc
        if (
            before_page.effective_fingerprint
            != command.expected_effective_page_fingerprint
        ):
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _fill_color_effective_parent(
            before_page,
            command.parent_id,
        )
        automatic_parent = _fill_color_automatic_parent(
            page,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        if before_parent.excluded:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.PARENT_EXCLUDED,
                "Fill color is unavailable while the parent is excluded.",
            )
        if automatic_parent.get("render_required") is not True:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.PARENT_NOT_RENDER_REQUIRED,
                "Fill color is available only for render-required parents.",
            )
        automatic_fill_color = automatic_render_fill_color(automatic_parent)
        if automatic_fill_color is None:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.AUTOMATIC_FILL_COLOR_UNAVAILABLE,
                "The automatic parent fill color is unavailable or invalid.",
            )
        slot_head = _active_fill_color_slot_head(
            snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        before_fill_color, before_authority = _effective_fill_color_state(
            before_parent,
            automatic_fill_color=automatic_fill_color,
        )
        if command.operation is RenderStyleFillColorOperation.SET:
            if command.fill_color == before_fill_color:
                raise RenderStyleFillColorCommandError(
                    RenderStyleFillColorCommandErrorCode.NO_OP,
                    "The requested fill color is already effective.",
                )
            operation = "set_fields"
            payload: Mapping[str, Any] = {
                "fields": {"fill_color": command.fill_color}
            }
        else:
            if (
                before_authority == "automatic"
                and slot_head is None
            ) or (
                slot_head is not None
                and slot_head.operation == "restore_automatic"
            ):
                raise RenderStyleFillColorCommandError(
                    RenderStyleFillColorCommandErrorCode.NO_OP,
                    "Fill color already uses the automatic value.",
                )
            operation = "restore_automatic"
            payload = {"fields": ("fill_color",)}

        target = EditTarget(
            EditTargetKind.PARENT,
            parent_id=command.parent_id,
        )
        try:
            base_fingerprint = field_base_fingerprint(
                project=materialized,
                page=page,
                target=target,
                domain=EditDomain.RENDER_STYLE,
                operation=operation,
                payload=payload,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.AUTOMATIC_FILL_COLOR_UNAVAILABLE,
                "The automatic parent fill color is invalid for fingerprinting.",
            ) from exc
        if base_fingerprint is None:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.AUTOMATIC_FILL_COLOR_UNAVAILABLE,
                "The automatic parent fill color is unavailable for fingerprinting.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.RENDER_STYLE,
            operation=operation,
            payload=payload,
            base_revision_id=before_parent.base_revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        _require_exact_fill_color_invalidation(
            invalidation,
            parent_id=command.parent_id,
        )

        try:
            candidate_ledger = snapshot.ledger.append(edit)
            after_page = project_effective_page(
                materialized,
                candidate_ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector rejected the fill-color edit.",
            ) from exc
        after_parent = _fill_color_effective_parent(
            after_page,
            command.parent_id,
        )
        after_fill_color, after_authority = _effective_fill_color_state(
            after_parent,
            automatic_fill_color=automatic_fill_color,
        )
        if command.operation is RenderStyleFillColorOperation.SET:
            accepted = (
                after_fill_color == command.fill_color
                and after_authority == "user"
            )
        else:
            accepted = (
                after_fill_color == automatic_fill_color
                and after_authority == "automatic"
            )
        if edit.edit_id not in after_page.applied_edit_ids or not accepted:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested fill color.",
            )

        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before fill color was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise RenderStyleFillColorCommandError(
                RenderStyleFillColorCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before fill color was committed.",
            ) from exc

        return RenderStyleFillColorCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            automatic_fill_color=automatic_fill_color,
            before_fill_color=before_fill_color,
            after_fill_color=after_fill_color,
            before_fill_color_authority=before_authority,
            after_fill_color_authority=after_authority,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


class RenderStyleOutlineColorCommandService:
    """Persist one exact opaque outline color through GUI-owned owners only."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: RenderStyleOutlineColorCommand,
    ) -> RenderStyleOutlineColorCommandReceipt:
        """Materialize once, then execute against that exact read snapshot."""

        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, RenderStyleOutlineColorCommand):
            raise TypeError("command must be a RenderStyleOutlineColorCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )
        snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        return self.execute_materialized(snapshot=snapshot, command=command)

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditReadSnapshot,
        command: RenderStyleOutlineColorCommand,
    ) -> RenderStyleOutlineColorCommandReceipt:
        """Execute from one worker-owned atomic project/ledger/head snapshot."""

        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, RenderStyleOutlineColorCommand):
            raise TypeError("command must be a RenderStyleOutlineColorCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        materialized = snapshot.project
        if project_id_for(materialized) != command.project_id:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if snapshot.ledger.project_id != command.project_id:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized ledger identity does not match the command.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the outline-color command was prepared.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the outline-color command was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.DUPLICATE_COMMAND,
                "The outline-color command was already recorded.",
            )

        page = _outline_color_project_page(materialized, command.page_id)
        try:
            before_page = project_effective_page(
                materialized,
                snapshot.ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.PROJECTION_REJECTED,
                "The effective page could not be projected before outline-color editing.",
            ) from exc
        if (
            before_page.effective_fingerprint
            != command.expected_effective_page_fingerprint
        ):
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _outline_color_effective_parent(
            before_page,
            command.parent_id,
        )
        automatic_parent = _outline_color_automatic_parent(
            page,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        if before_parent.excluded:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.PARENT_EXCLUDED,
                "Outline color is unavailable while the parent is excluded.",
            )
        if automatic_parent.get("render_required") is not True:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.PARENT_NOT_RENDER_REQUIRED,
                "Outline color is available only for render-required parents.",
            )
        automatic_outline_color = automatic_render_outline_color(automatic_parent)
        if automatic_outline_color is None:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.AUTOMATIC_OUTLINE_COLOR_UNAVAILABLE,
                "The automatic parent outline color is unavailable or invalid.",
            )
        slot_head = _active_outline_color_slot_head(
            snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        before_outline_color, before_authority = _effective_outline_color_state(
            before_parent,
            automatic_outline_color=automatic_outline_color,
        )
        if command.operation is RenderStyleOutlineColorOperation.SET:
            if command.outline_color == before_outline_color:
                raise RenderStyleOutlineColorCommandError(
                    RenderStyleOutlineColorCommandErrorCode.NO_OP,
                    "The requested outline color is already effective.",
                )
            operation = "set_fields"
            payload: Mapping[str, Any] = {
                "fields": {"outline_color": command.outline_color}
            }
        else:
            if (
                before_authority == "automatic"
                and slot_head is None
            ) or (
                slot_head is not None
                and slot_head.operation == "restore_automatic"
            ):
                raise RenderStyleOutlineColorCommandError(
                    RenderStyleOutlineColorCommandErrorCode.NO_OP,
                    "Outline color already uses the automatic value.",
                )
            operation = "restore_automatic"
            payload = {"fields": ("outline_color",)}

        target = EditTarget(
            EditTargetKind.PARENT,
            parent_id=command.parent_id,
        )
        try:
            base_fingerprint = field_base_fingerprint(
                project=materialized,
                page=page,
                target=target,
                domain=EditDomain.RENDER_STYLE,
                operation=operation,
                payload=payload,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.AUTOMATIC_OUTLINE_COLOR_UNAVAILABLE,
                "The automatic parent outline color is invalid for fingerprinting.",
            ) from exc
        if base_fingerprint is None:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.AUTOMATIC_OUTLINE_COLOR_UNAVAILABLE,
                "The automatic parent outline color is unavailable for fingerprinting.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.RENDER_STYLE,
            operation=operation,
            payload=payload,
            base_revision_id=before_parent.base_revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        _require_exact_outline_color_invalidation(
            invalidation,
            parent_id=command.parent_id,
        )

        try:
            candidate_ledger = snapshot.ledger.append(edit)
            after_page = project_effective_page(
                materialized,
                candidate_ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector rejected the outline-color edit.",
            ) from exc
        after_parent = _outline_color_effective_parent(
            after_page,
            command.parent_id,
        )
        after_outline_color, after_authority = _effective_outline_color_state(
            after_parent,
            automatic_outline_color=automatic_outline_color,
        )
        if command.operation is RenderStyleOutlineColorOperation.SET:
            accepted = (
                after_outline_color == command.outline_color
                and after_authority == "user"
            )
        else:
            accepted = (
                after_outline_color == automatic_outline_color
                and after_authority == "automatic"
            )
        if edit.edit_id not in after_page.applied_edit_ids or not accepted:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested outline color.",
            )

        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before outline color was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise RenderStyleOutlineColorCommandError(
                RenderStyleOutlineColorCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before outline color was committed.",
            ) from exc

        return RenderStyleOutlineColorCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            automatic_outline_color=automatic_outline_color,
            before_outline_color=before_outline_color,
            after_outline_color=after_outline_color,
            before_outline_color_authority=before_authority,
            after_outline_color_authority=after_authority,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


class RenderStyleOutlineWidthCommandService:
    """Persist one exact outline width through GUI-owned owners only."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: RenderStyleOutlineWidthCommand,
    ) -> RenderStyleOutlineWidthCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, RenderStyleOutlineWidthCommand):
            raise TypeError("command must be a RenderStyleOutlineWidthCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )
        snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        return self.execute_materialized(snapshot=snapshot, command=command)

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditReadSnapshot,
        command: RenderStyleOutlineWidthCommand,
    ) -> RenderStyleOutlineWidthCommandReceipt:
        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, RenderStyleOutlineWidthCommand):
            raise TypeError("command must be a RenderStyleOutlineWidthCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        materialized = snapshot.project
        if (
            project_id_for(materialized) != command.project_id
            or snapshot.ledger.project_id != command.project_id
        ):
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the outline-width command was prepared.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the outline-width command was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.DUPLICATE_COMMAND,
                "The outline-width command was already recorded.",
            )

        page = _outline_width_project_page(materialized, command.page_id)
        try:
            before_page = project_effective_page(
                materialized,
                snapshot.ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.PROJECTION_REJECTED,
                "The effective page could not be projected before outline-width editing.",
            ) from exc
        if before_page.effective_fingerprint != command.expected_effective_page_fingerprint:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _outline_width_effective_parent(before_page, command.parent_id)
        automatic_parent = _outline_width_automatic_parent(
            page,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        if before_parent.excluded:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.PARENT_EXCLUDED,
                "Outline width is unavailable while the parent is excluded.",
            )
        if automatic_parent.get("render_required") is not True:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.PARENT_NOT_RENDER_REQUIRED,
                "Outline width is available only for render-required parents.",
            )
        automatic_width = automatic_render_outline_width(automatic_parent)
        if automatic_width is None:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.AUTOMATIC_OUTLINE_WIDTH_UNAVAILABLE,
                "The automatic parent outline width is unavailable or invalid.",
            )
        slot_head = _active_outline_width_slot_head(
            snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        before_width, before_authority = _effective_outline_width_state(
            before_parent,
            automatic_outline_width=automatic_width,
        )
        if command.operation is RenderStyleOutlineWidthOperation.SET:
            if command.outline_width == before_width:
                raise RenderStyleOutlineWidthCommandError(
                    RenderStyleOutlineWidthCommandErrorCode.NO_OP,
                    "The requested outline width is already effective.",
                )
            operation = "set_fields"
            payload: Mapping[str, Any] = {
                "fields": {"outline_width": command.outline_width}
            }
        else:
            if (
                before_authority == "automatic" and slot_head is None
            ) or (
                slot_head is not None and slot_head.operation == "restore_automatic"
            ):
                raise RenderStyleOutlineWidthCommandError(
                    RenderStyleOutlineWidthCommandErrorCode.NO_OP,
                    "Outline width already uses the automatic value.",
                )
            operation = "restore_automatic"
            payload = {"fields": ("outline_width",)}

        target = EditTarget(EditTargetKind.PARENT, parent_id=command.parent_id)
        try:
            base_fingerprint = field_base_fingerprint(
                project=materialized,
                page=page,
                target=target,
                domain=EditDomain.RENDER_STYLE,
                operation=operation,
                payload=payload,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.AUTOMATIC_OUTLINE_WIDTH_UNAVAILABLE,
                "The automatic parent outline width is invalid for fingerprinting.",
            ) from exc
        if base_fingerprint is None:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.AUTOMATIC_OUTLINE_WIDTH_UNAVAILABLE,
                "The automatic parent outline width is unavailable for fingerprinting.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.RENDER_STYLE,
            operation=operation,
            payload=payload,
            base_revision_id=before_parent.base_revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        _require_exact_outline_width_invalidation(
            invalidation,
            parent_id=command.parent_id,
        )
        try:
            candidate_ledger = snapshot.ledger.append(edit)
            after_page = project_effective_page(
                materialized,
                candidate_ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector rejected the outline-width edit.",
            ) from exc
        after_parent = _outline_width_effective_parent(after_page, command.parent_id)
        after_width, after_authority = _effective_outline_width_state(
            after_parent,
            automatic_outline_width=automatic_width,
        )
        accepted = (
            after_width == command.outline_width and after_authority == "user"
            if command.operation is RenderStyleOutlineWidthOperation.SET
            else after_width == automatic_width and after_authority == "automatic"
        )
        if edit.edit_id not in after_page.applied_edit_ids or not accepted:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested outline width.",
            )
        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before outline width was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise RenderStyleOutlineWidthCommandError(
                RenderStyleOutlineWidthCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before outline width was committed.",
            ) from exc
        return RenderStyleOutlineWidthCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            automatic_outline_width=automatic_width,
            before_outline_width=before_width,
            after_outline_width=after_width,
            before_outline_width_authority=before_authority,
            after_outline_width_authority=after_authority,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


class RenderStylePreferredSizeCommandService:
    """Persist one exact preferred size through GUI-owned owners only."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: RenderStylePreferredSizeCommand,
    ) -> RenderStylePreferredSizeCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, RenderStylePreferredSizeCommand):
            raise TypeError("command must be a RenderStylePreferredSizeCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )
        snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        return self.execute_materialized(snapshot=snapshot, command=command)

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditReadSnapshot,
        command: RenderStylePreferredSizeCommand,
    ) -> RenderStylePreferredSizeCommandReceipt:
        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, RenderStylePreferredSizeCommand):
            raise TypeError("command must be a RenderStylePreferredSizeCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        materialized = snapshot.project
        if (
            project_id_for(materialized) != command.project_id
            or snapshot.ledger.project_id != command.project_id
        ):
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the preferred-size command was prepared.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the preferred-size command was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.DUPLICATE_COMMAND,
                "The preferred-size command was already recorded.",
            )

        page = _preferred_size_project_page(materialized, command.page_id)
        try:
            before_page = project_effective_page(
                materialized,
                snapshot.ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.PROJECTION_REJECTED,
                "The effective page could not be projected before preferred-size editing.",
            ) from exc
        if before_page.effective_fingerprint != command.expected_effective_page_fingerprint:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _preferred_size_effective_parent(before_page, command.parent_id)
        automatic_parent = _preferred_size_automatic_parent(
            page,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        if before_parent.excluded:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.PARENT_EXCLUDED,
                "Preferred size is unavailable while the parent is excluded.",
            )
        if automatic_parent.get("render_required") is not True:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.PARENT_NOT_RENDER_REQUIRED,
                "Preferred size is available only for render-required parents.",
            )
        automatic_size = automatic_render_preferred_size(automatic_parent)
        if automatic_size is None:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.AUTOMATIC_PREFERRED_SIZE_UNAVAILABLE,
                "The automatic parent preferred size is unavailable or invalid.",
            )
        slot_head = _active_preferred_size_slot_head(
            snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        before_size, before_authority = _effective_preferred_size_state(
            before_parent,
            automatic_preferred_size=automatic_size,
        )
        if command.operation is RenderStylePreferredSizeOperation.SET:
            if command.preferred_size == before_size:
                raise RenderStylePreferredSizeCommandError(
                    RenderStylePreferredSizeCommandErrorCode.NO_OP,
                    "The requested preferred size is already effective.",
                )
            operation = "set_fields"
            payload: Mapping[str, Any] = {
                "fields": {"preferred_size": command.preferred_size}
            }
        else:
            if (
                before_authority == "automatic" and slot_head is None
            ) or (
                slot_head is not None and slot_head.operation == "restore_automatic"
            ):
                raise RenderStylePreferredSizeCommandError(
                    RenderStylePreferredSizeCommandErrorCode.NO_OP,
                    "Preferred size already uses the automatic value.",
                )
            operation = "restore_automatic"
            payload = {"fields": ("preferred_size",)}

        target = EditTarget(EditTargetKind.PARENT, parent_id=command.parent_id)
        try:
            base_fingerprint = field_base_fingerprint(
                project=materialized,
                page=page,
                target=target,
                domain=EditDomain.RENDER_STYLE,
                operation=operation,
                payload=payload,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.AUTOMATIC_PREFERRED_SIZE_UNAVAILABLE,
                "The automatic parent preferred size is invalid for fingerprinting.",
            ) from exc
        if base_fingerprint is None:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.AUTOMATIC_PREFERRED_SIZE_UNAVAILABLE,
                "The automatic parent preferred size is unavailable for fingerprinting.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.RENDER_STYLE,
            operation=operation,
            payload=payload,
            base_revision_id=before_parent.base_revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        _require_exact_preferred_size_invalidation(
            invalidation,
            parent_id=command.parent_id,
        )
        try:
            candidate_ledger = snapshot.ledger.append(edit)
            after_page = project_effective_page(
                materialized,
                candidate_ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector rejected the preferred-size edit.",
            ) from exc
        after_parent = _preferred_size_effective_parent(after_page, command.parent_id)
        after_size, after_authority = _effective_preferred_size_state(
            after_parent,
            automatic_preferred_size=automatic_size,
        )
        accepted = (
            after_size == command.preferred_size and after_authority == "user"
            if command.operation is RenderStylePreferredSizeOperation.SET
            else after_size == automatic_size and after_authority == "automatic"
        )
        if edit.edit_id not in after_page.applied_edit_ids or not accepted:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested preferred size.",
            )
        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before preferred size was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise RenderStylePreferredSizeCommandError(
                RenderStylePreferredSizeCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before preferred size was committed.",
            ) from exc
        return RenderStylePreferredSizeCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            automatic_preferred_size=automatic_size,
            before_preferred_size=before_size,
            after_preferred_size=after_size,
            before_preferred_size_authority=before_authority,
            after_preferred_size_authority=after_authority,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )



class RenderStyleShadowVisibilityCommandService:
    """Hide one exact automatic shadow through GUI-owned owners only."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: RenderStyleShadowVisibilityCommand,
    ) -> RenderStyleShadowVisibilityCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, RenderStyleShadowVisibilityCommand):
            raise TypeError("command must be a RenderStyleShadowVisibilityCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )
        snapshot = self._edit_store.materialize_project_snapshot(
            project, page_id=command.page_id
        )
        return self.execute_materialized(snapshot=snapshot, command=command)

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditReadSnapshot,
        command: RenderStyleShadowVisibilityCommand,
    ) -> RenderStyleShadowVisibilityCommandReceipt:
        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, RenderStyleShadowVisibilityCommand):
            raise TypeError("command must be a RenderStyleShadowVisibilityCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        materialized = snapshot.project
        if (
            project_id_for(materialized) != command.project_id
            or snapshot.ledger.project_id != command.project_id
        ):
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the shadow command was prepared.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the shadow command was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.DUPLICATE_COMMAND,
                "The shadow-visibility command was already recorded.",
            )

        page = _shadow_visibility_project_page(materialized, command.page_id)
        try:
            before_page = project_effective_page(
                materialized, snapshot.ledger, page_id=command.page_id
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.PROJECTION_REJECTED,
                "The effective page could not be projected before shadow editing.",
            ) from exc
        if before_page.effective_fingerprint != command.expected_effective_page_fingerprint:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _shadow_visibility_effective_parent(
            before_page, command.parent_id
        )
        automatic_parent = _shadow_visibility_automatic_parent(
            page, page_id=command.page_id, parent_id=command.parent_id
        )
        if before_parent.excluded:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.PARENT_EXCLUDED,
                "Shadow visibility is unavailable while the parent is excluded.",
            )
        if automatic_parent.get("render_required") is not True:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.PARENT_NOT_RENDER_REQUIRED,
                "Shadow visibility is available only for render-required parents.",
            )
        if automatic_render_shadow_enabled(automatic_parent) is not True:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.AUTOMATIC_SHADOW_UNAVAILABLE,
                "The automatic parent does not have one valid visible shadow.",
            )
        slot_head = _active_shadow_visibility_slot_head(
            snapshot.ledger, page_id=command.page_id, parent_id=command.parent_id
        )
        before_enabled, before_authority = _effective_shadow_visibility_state(
            before_parent
        )
        if command.operation is RenderStyleShadowVisibilityOperation.HIDE:
            if before_enabled is False:
                raise RenderStyleShadowVisibilityCommandError(
                    RenderStyleShadowVisibilityCommandErrorCode.NO_OP,
                    "The selected shadow is already hidden.",
                )
            operation = "set_fields"
            payload: Mapping[str, Any] = {"fields": {"shadow_enabled": False}}
        else:
            if (
                before_authority == "automatic" and slot_head is None
            ) or (
                slot_head is not None and slot_head.operation == "restore_automatic"
            ):
                raise RenderStyleShadowVisibilityCommandError(
                    RenderStyleShadowVisibilityCommandErrorCode.NO_OP,
                    "Shadow visibility already uses the automatic effect.",
                )
            operation = "restore_automatic"
            payload = {"fields": ("shadow_enabled",)}

        target = EditTarget(EditTargetKind.PARENT, parent_id=command.parent_id)
        try:
            base_fingerprint = field_base_fingerprint(
                project=materialized,
                page=page,
                target=target,
                domain=EditDomain.RENDER_STYLE,
                operation=operation,
                payload=payload,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.AUTOMATIC_SHADOW_UNAVAILABLE,
                "The automatic shadow is invalid for fingerprinting.",
            ) from exc
        if base_fingerprint is None:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.AUTOMATIC_SHADOW_UNAVAILABLE,
                "The automatic shadow is unavailable for fingerprinting.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.RENDER_STYLE,
            operation=operation,
            payload=payload,
            base_revision_id=before_parent.base_revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        _require_exact_shadow_visibility_invalidation(
            invalidation, parent_id=command.parent_id
        )
        try:
            candidate_ledger = snapshot.ledger.append(edit)
            after_page = project_effective_page(
                materialized, candidate_ledger, page_id=command.page_id
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector rejected the shadow-visibility edit.",
            ) from exc
        after_parent = _shadow_visibility_effective_parent(
            after_page, command.parent_id
        )
        after_enabled, after_authority = _effective_shadow_visibility_state(
            after_parent
        )
        accepted = (
            after_enabled is False and after_authority == "user"
            if command.operation is RenderStyleShadowVisibilityOperation.HIDE
            else after_enabled is True and after_authority == "automatic"
        )
        if edit.edit_id not in after_page.applied_edit_ids or not accepted:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested shadow state.",
            )
        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before shadow visibility was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise RenderStyleShadowVisibilityCommandError(
                RenderStyleShadowVisibilityCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before shadow visibility was committed.",
            ) from exc
        return RenderStyleShadowVisibilityCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            automatic_shadow_enabled=True,
            before_shadow_enabled=before_enabled,
            after_shadow_enabled=after_enabled,
            before_shadow_enabled_authority=before_authority,
            after_shadow_enabled_authority=after_authority,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


class RenderStyleShadowBlurCommandService:
    """Persist one exact shadow blur radius through GUI-owned owners only."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: RenderStyleShadowBlurCommand,
    ) -> RenderStyleShadowBlurCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, RenderStyleShadowBlurCommand):
            raise TypeError("command must be a RenderStyleShadowBlurCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )
        snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        return self.execute_materialized(snapshot=snapshot, command=command)

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditReadSnapshot,
        command: RenderStyleShadowBlurCommand,
    ) -> RenderStyleShadowBlurCommandReceipt:
        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, RenderStyleShadowBlurCommand):
            raise TypeError("command must be a RenderStyleShadowBlurCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        materialized = snapshot.project
        if (
            project_id_for(materialized) != command.project_id
            or snapshot.ledger.project_id != command.project_id
        ):
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the shadow-blur command was prepared.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the shadow-blur command was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.DUPLICATE_COMMAND,
                "The shadow-blur command was already recorded.",
            )

        page = _shadow_blur_project_page(materialized, command.page_id)
        try:
            before_page = project_effective_page(
                materialized,
                snapshot.ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.PROJECTION_REJECTED,
                "The effective page could not be projected before shadow-blur editing.",
            ) from exc
        if before_page.effective_fingerprint != command.expected_effective_page_fingerprint:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _shadow_blur_effective_parent(before_page, command.parent_id)
        automatic_parent = _shadow_blur_automatic_parent(
            page,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        if before_parent.excluded:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.PARENT_EXCLUDED,
                "Shadow blur is unavailable while the parent is excluded.",
            )
        if automatic_parent.get("render_required") is not True:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.PARENT_NOT_RENDER_REQUIRED,
                "Shadow blur is available only for render-required parents.",
            )
        automatic_blur = automatic_render_shadow_blur(automatic_parent)
        if automatic_blur is None:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.AUTOMATIC_SHADOW_UNAVAILABLE,
                "The automatic parent does not have one valid visible shadow.",
            )
        slot_head = _active_shadow_blur_slot_head(
            snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        before_blur, before_authority = _effective_shadow_blur_state(
            before_parent,
            automatic_shadow_blur=automatic_blur,
        )
        if command.operation is RenderStyleShadowBlurOperation.SET:
            if command.shadow_blur == before_blur:
                raise RenderStyleShadowBlurCommandError(
                    RenderStyleShadowBlurCommandErrorCode.NO_OP,
                    "The requested shadow blur is already effective.",
                )
            operation = "set_fields"
            payload: Mapping[str, Any] = {
                "fields": {"shadow_blur": command.shadow_blur}
            }
        else:
            if (
                before_authority == "automatic" and slot_head is None
            ) or (
                slot_head is not None and slot_head.operation == "restore_automatic"
            ):
                raise RenderStyleShadowBlurCommandError(
                    RenderStyleShadowBlurCommandErrorCode.NO_OP,
                    "Shadow blur already uses the automatic value.",
                )
            operation = "restore_automatic"
            payload = {"fields": ("shadow_blur",)}

        target = EditTarget(EditTargetKind.PARENT, parent_id=command.parent_id)
        try:
            base_fingerprint = field_base_fingerprint(
                project=materialized,
                page=page,
                target=target,
                domain=EditDomain.RENDER_STYLE,
                operation=operation,
                payload=payload,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.AUTOMATIC_SHADOW_UNAVAILABLE,
                "The automatic shadow is invalid for fingerprinting.",
            ) from exc
        if base_fingerprint is None:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.AUTOMATIC_SHADOW_UNAVAILABLE,
                "The automatic shadow is unavailable for fingerprinting.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.RENDER_STYLE,
            operation=operation,
            payload=payload,
            base_revision_id=before_parent.base_revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        _require_exact_shadow_blur_invalidation(
            invalidation,
            parent_id=command.parent_id,
        )
        try:
            candidate_ledger = snapshot.ledger.append(edit)
            after_page = project_effective_page(
                materialized,
                candidate_ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector rejected the shadow-blur edit.",
            ) from exc
        after_parent = _shadow_blur_effective_parent(after_page, command.parent_id)
        after_blur, after_authority = _effective_shadow_blur_state(
            after_parent,
            automatic_shadow_blur=automatic_blur,
        )
        accepted = (
            after_blur == command.shadow_blur and after_authority == "user"
            if command.operation is RenderStyleShadowBlurOperation.SET
            else after_blur == automatic_blur and after_authority == "automatic"
        )
        if edit.edit_id not in after_page.applied_edit_ids or not accepted:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested shadow blur.",
            )
        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before shadow blur was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise RenderStyleShadowBlurCommandError(
                RenderStyleShadowBlurCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before shadow blur was committed.",
            ) from exc
        return RenderStyleShadowBlurCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            automatic_shadow_blur=automatic_blur,
            before_shadow_blur=before_blur,
            after_shadow_blur=after_blur,
            before_shadow_blur_authority=before_authority,
            after_shadow_blur_authority=after_authority,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


class RenderStyleFontRoleCommandService:
    """Persist one exact registered font role through GUI-owned owners only."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: RenderStyleFontRoleCommand,
    ) -> RenderStyleFontRoleCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, RenderStyleFontRoleCommand):
            raise TypeError("command must be a RenderStyleFontRoleCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )
        snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        return self.execute_materialized(snapshot=snapshot, command=command)

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditReadSnapshot,
        command: RenderStyleFontRoleCommand,
    ) -> RenderStyleFontRoleCommandReceipt:
        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, RenderStyleFontRoleCommand):
            raise TypeError("command must be a RenderStyleFontRoleCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        materialized = snapshot.project
        if (
            project_id_for(materialized) != command.project_id
            or snapshot.ledger.project_id != command.project_id
        ):
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the font-role command was prepared.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the font-role command was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.DUPLICATE_COMMAND,
                "The font-role command was already recorded.",
            )

        page = _font_role_project_page(materialized, command.page_id)
        try:
            before_page = project_effective_page(
                materialized,
                snapshot.ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.PROJECTION_REJECTED,
                "The effective page could not be projected before font-role editing.",
            ) from exc
        if before_page.effective_fingerprint != command.expected_effective_page_fingerprint:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _font_role_effective_parent(before_page, command.parent_id)
        automatic_parent = _font_role_automatic_parent(
            page,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        if before_parent.excluded:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.PARENT_EXCLUDED,
                "Font role is unavailable while the parent is excluded.",
            )
        if automatic_parent.get("render_required") is not True:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.PARENT_NOT_RENDER_REQUIRED,
                "Font role is available only for render-required parents.",
            )
        automatic_font_role = automatic_render_font_role(automatic_parent)
        if automatic_font_role is None:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.AUTOMATIC_FONT_ROLE_UNAVAILABLE,
                "The automatic registered font role is unavailable or invalid.",
            )
        slot_head = _active_font_role_slot_head(
            snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        before_font_role, before_authority = _effective_font_role_state(
            before_parent,
            automatic_font_role=automatic_font_role,
        )
        if command.operation is RenderStyleFontRoleOperation.SET:
            if command.font_role == before_font_role:
                raise RenderStyleFontRoleCommandError(
                    RenderStyleFontRoleCommandErrorCode.NO_OP,
                    "The requested font role is already effective.",
                )
            operation = "set_fields"
            payload: Mapping[str, Any] = {"fields": {"font_role": command.font_role}}
        else:
            if (before_authority == "automatic" and slot_head is None) or (
                slot_head is not None and slot_head.operation == "restore_automatic"
            ):
                raise RenderStyleFontRoleCommandError(
                    RenderStyleFontRoleCommandErrorCode.NO_OP,
                    "Font role already uses the automatic value.",
                )
            operation = "restore_automatic"
            payload = {"fields": ("font_role",)}

        target = EditTarget(EditTargetKind.PARENT, parent_id=command.parent_id)
        try:
            base_fingerprint = field_base_fingerprint(
                project=materialized,
                page=page,
                target=target,
                domain=EditDomain.RENDER_STYLE,
                operation=operation,
                payload=payload,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.AUTOMATIC_FONT_ROLE_UNAVAILABLE,
                "The automatic font role is invalid for fingerprinting.",
            ) from exc
        if base_fingerprint is None:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.AUTOMATIC_FONT_ROLE_UNAVAILABLE,
                "The automatic font role is unavailable for fingerprinting.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.RENDER_STYLE,
            operation=operation,
            payload=payload,
            base_revision_id=before_parent.base_revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        _require_exact_font_role_invalidation(
            invalidation,
            parent_id=command.parent_id,
        )
        try:
            candidate_ledger = snapshot.ledger.append(edit)
            after_page = project_effective_page(
                materialized,
                candidate_ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector rejected the font-role edit.",
            ) from exc
        after_parent = _font_role_effective_parent(after_page, command.parent_id)
        after_font_role, after_authority = _effective_font_role_state(
            after_parent,
            automatic_font_role=automatic_font_role,
        )
        accepted = (
            (command.operation is RenderStyleFontRoleOperation.SET
             and after_font_role == command.font_role and after_authority == "user")
            or (command.operation is RenderStyleFontRoleOperation.RESTORE_AUTOMATIC
                and after_font_role == automatic_font_role
                and after_authority == "automatic")
        )
        if edit.edit_id not in after_page.applied_edit_ids or not accepted:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested font role.",
            )
        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before font role was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise RenderStyleFontRoleCommandError(
                RenderStyleFontRoleCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before font role was committed.",
            ) from exc
        return RenderStyleFontRoleCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            automatic_font_role=automatic_font_role,
            before_font_role=before_font_role,
            after_font_role=after_font_role,
            before_font_role_authority=before_authority,
            after_font_role_authority=after_authority,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


def _history_json_references_any(value: Any, identities: frozenset[str]) -> bool:
    if isinstance(value, str):
        return value in identities
    if isinstance(value, Mapping):
        return any(
            _history_json_references_any(item, identities)
            for item in value.values()
        )
    if isinstance(value, (tuple, list)):
        return any(
            _history_json_references_any(item, identities) for item in value
        )
    return False


def _history_artifact_dependent_active_edit_ids(
    snapshot: ProjectEditReadSnapshot,
    target_edit: ProjectEdit,
) -> tuple[str, ...]:
    """Return active selection edits whose immutable artifacts reference target."""

    try:
        target_index = next(
            index
            for index, record in enumerate(snapshot.ledger.edits)
            if record.edit_id == target_edit.edit_id
        )
    except StopIteration:
        return ()
    identities = {
        target_edit.edit_id,
        str(target_edit.payload.get("revision_id") or ""),
    }
    if target_edit.target.kind is EditTargetKind.PARENT:
        identities.add(target_edit.target.parent_id)
    if target_edit.domain is EditDomain.STRUCTURAL:
        identities.update(
            str(value)
            for value in (
                target_edit.payload.get("root_id"),
                target_edit.payload.get("merged_root_id"),
                *(target_edit.payload.get("child_parent_ids") or ()),
                *(target_edit.payload.get("child_root_ids") or ()),
            )
            if str(value or "")
        )
    stable_identities = frozenset(value for value in identities if value)
    if not stable_identities:
        return ()

    artifact_index: dict[str, Mapping[str, Any]] = {}
    catalogs = snapshot.project.get("artifact_revisions")
    if isinstance(catalogs, Mapping):
        for records in catalogs.values():
            if not isinstance(records, (tuple, list)):
                continue
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                revision_id = str(record.get("revision_id") or "")
                if revision_id:
                    artifact_index[revision_id] = record

    active_ids = set(snapshot.ledger.state().active_edit_ids)
    result: list[str] = []
    for record in snapshot.ledger.edits[target_index + 1 :]:
        if record.is_control or record.edit_id not in active_ids:
            continue
        revision_id = str(record.payload.get("revision_id") or "")
        artifact = artifact_index.get(revision_id)
        if artifact is not None and _history_json_references_any(
            artifact,
            stable_identities,
        ):
            result.append(record.edit_id)
    return tuple(result)


class EditHistoryCommandService:
    """Persist one durable Revoke/Reapply control without invoking a pipeline owner."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: EditHistoryCommand,
    ) -> EditHistoryCommandReceipt:
        """Materialize once, then execute against that exact read snapshot."""

        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(command, EditHistoryCommand):
            raise TypeError("command must be an EditHistoryCommand")
        if self._edit_store.project_id != command.project_id:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if project_id_for(project) != command.project_id:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the command.",
            )
        snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        return self.execute_materialized(snapshot=snapshot, command=command)

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditReadSnapshot,
        command: EditHistoryCommand,
    ) -> EditHistoryCommandReceipt:
        """Execute from the worker-owned atomic project/ledger/head snapshot."""

        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, EditHistoryCommand):
            raise TypeError("command must be an EditHistoryCommand")
        if self._edit_store.project_id != command.project_id:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        materialized = snapshot.project
        if project_id_for(materialized) != command.project_id:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if snapshot.ledger.project_id != command.project_id:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized ledger identity does not match the command.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the history command was prepared.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the history command was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.DUPLICATE_COMMAND,
                "The history command was already recorded.",
            )

        _history_project_page(materialized, command.page_id)
        target_edit = snapshot.ledger.get(command.target_edit_id)
        if target_edit is None:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.TARGET_EDIT_NOT_FOUND,
                "The selected persisted edit no longer exists.",
            )
        if target_edit.is_control:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.CONTROL_TARGET_FORBIDDEN,
                "History control records are read-only evidence.",
            )
        if target_edit.project_id != command.project_id:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "The selected edit belongs to another project.",
            )
        if target_edit.page_id != command.page_id:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.TARGET_EDIT_PAGE_MISMATCH,
                "The selected edit belongs to another page.",
            )
        if target_edit.target.kind is EditTargetKind.ARTIFACT:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.ARTIFACT_TARGET_FORBIDDEN,
                "Artifact revision history is read-only evidence.",
            )
        _require_supported_history_target(target_edit)

        state = snapshot.ledger.state()
        before_active = target_edit.edit_id in set(state.active_edit_ids)
        if command.operation is EditHistoryOperation.REVOKE and not before_active:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.ALREADY_REVOKED,
                "The selected edit is already revoked.",
            )
        if command.operation is EditHistoryOperation.REAPPLY and before_active:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.ALREADY_ACTIVE,
                "The selected edit is already active.",
            )
        if command.operation is EditHistoryOperation.REVOKE:
            dependent_edit_ids = tuple(
                dict.fromkeys(
                    (
                        *snapshot.ledger.dependent_active_edit_ids(
                            target_edit.edit_id
                        ),
                        *_history_artifact_dependent_active_edit_ids(
                            snapshot,
                            target_edit,
                        ),
                    )
                )
            )
            if dependent_edit_ids:
                raise EditHistoryCommandError(
                    EditHistoryCommandErrorCode.ACTIVE_DEPENDENT_EDIT,
                    "Revoke would strand later active edits that reference the "
                    "user parent/root: " + ", ".join(dependent_edit_ids),
                )

        try:
            before_page = project_effective_page(
                materialized,
                snapshot.ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.PROJECTION_REJECTED,
                "The effective page could not be projected before the history command.",
            ) from exc
        if (
            before_page.effective_fingerprint
            != command.expected_effective_page_fingerprint
        ):
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the history command was prepared.",
            )

        try:
            candidate_ledger = (
                snapshot.ledger.revoke(
                    target_edit.edit_id,
                    event_id=command.command_id,
                )
                if command.operation is EditHistoryOperation.REVOKE
                else snapshot.ledger.reapply(
                    target_edit.edit_id,
                    event_id=command.command_id,
                )
            )
            control_edit = candidate_ledger.edits[-1]
            after_page = project_effective_page(
                materialized,
                candidate_ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.PROJECTION_REJECTED,
                "The effective page rejected the history transition.",
            ) from exc

        after_state = candidate_ledger.state()
        after_active = target_edit.edit_id in set(after_state.active_edit_ids)
        expected_after_active = command.operation is EditHistoryOperation.REAPPLY
        if after_active is not expected_after_active:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.PROJECTION_REJECTED,
                "The ledger did not produce the requested active state.",
            )
        try:
            invalidation = invalidation_for_control(
                control_edit,
                target_edit=target_edit,
                before_effective_page=before_page,
                after_effective_page=after_page,
            )
        except (TypeError, ValueError) as exc:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.INVALIDATION_UNRESOLVED,
                "The history transition has no exact GUI-owned invalidation.",
            ) from exc
        if invalidation.unresolved_facts:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.INVALIDATION_UNRESOLVED,
                "The history transition requires unavailable dependency facts.",
            )

        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (control_edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before the history command was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise EditHistoryCommandError(
                EditHistoryCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before the history command was committed.",
            ) from exc

        return EditHistoryCommandReceipt(
            command_id=command.command_id,
            target_edit=target_edit,
            control_edit=control_edit,
            before_active=before_active,
            after_active=after_active,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            before_issues=before_page.issues,
            after_issues=after_page.issues,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


def _history_project_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise EditHistoryCommandError(
            EditHistoryCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = tuple(
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    )
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise EditHistoryCommandError(
            EditHistoryCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is {reason}: {page_id}",
        )
    return matches[0]


def _require_supported_history_target(edit: ProjectEdit) -> None:
    supported = False
    if edit.domain in {
        EditDomain.SOURCE_TEXT,
        EditDomain.TARGET_TEXT,
        EditDomain.REVIEW_METADATA,
    }:
        supported = edit.target.kind is EditTargetKind.PARENT
    elif edit.domain is EditDomain.STRUCTURAL:
        supported = bool(
            (
                edit.target.kind is EditTargetKind.PARENT
                and edit.operation
                in {
                    ParentGeometryOperation.SET_GEOMETRY.value,
                    AddUserParentOperation.ADD.value,
                    SplitUserParentOperation.SPLIT.value,
                    MergePipelineParentsOperation.MERGE.value,
                }
            )
            or (
                edit.target.kind is EditTargetKind.PAGE
                and edit.operation == ReadingOrderOperation.SET.value
            )
        )
    elif edit.domain is EditDomain.RENDER_LAYOUT:
        supported = (
            edit.target.kind is EditTargetKind.PARENT
            and _writing_mode_edit_field(edit)
            in {"writing_mode", "line_height", "rotation", "render_box"}
        )
    elif edit.domain is EditDomain.RENDER_STYLE:
        supported = (
            edit.target.kind is EditTargetKind.PARENT
            and _render_style_edit_field(edit)
            in {
                "fill_color",
                "outline_color",
                "outline_width",
                    "preferred_size",
                "shadow_enabled",
                "shadow_blur",
                "shadow_color",
                "shadow_offset",
                "font_role",
                "font_weight_tier",
            }
        )
    elif edit.domain is EditDomain.CLEANUP:
        supported = edit.target.kind is EditTargetKind.PAGE
    elif edit.domain is EditDomain.GLOSSARY:
        supported = bool(
            edit.target.kind is EditTargetKind.PROJECT
            and edit.operation in {"set_entry", "remove_entry"}
        )
    if not supported:
        raise EditHistoryCommandError(
            EditHistoryCommandErrorCode.INVALIDATION_UNRESOLVED,
            "The selected edit does not yet have an exact durable-history invalidation.",
        )


from .shadow_offset_commands import (
    RenderStyleShadowOffsetCommand,
    RenderStyleShadowOffsetCommandError,
    RenderStyleShadowOffsetCommandErrorCode,
    RenderStyleShadowOffsetCommandReceipt,
    RenderStyleShadowOffsetCommandService,
    RenderStyleShadowOffsetOperation,
)

from .shadow_color_commands import (
    RenderStyleShadowColorCommand,
    RenderStyleShadowColorCommandError,
    RenderStyleShadowColorCommandErrorCode,
    RenderStyleShadowColorCommandReceipt,
    RenderStyleShadowColorCommandService,
    RenderStyleShadowColorOperation,
)

from .font_weight_tier_commands import (
    RenderStyleFontWeightTierCommand,
    RenderStyleFontWeightTierCommandError,
    RenderStyleFontWeightTierCommandErrorCode,
    RenderStyleFontWeightTierCommandReceipt,
    RenderStyleFontWeightTierCommandService,
    RenderStyleFontWeightTierOperation,
)

from .render_box_commands import (
    RenderLayoutRenderBoxCommand,
    RenderLayoutRenderBoxCommandError,
    RenderLayoutRenderBoxCommandErrorCode,
    RenderLayoutRenderBoxCommandReceipt,
    RenderLayoutRenderBoxCommandService,
    RenderLayoutRenderBoxOperation,
)


__all__ = [
    "AddUserParentCommand",
    "AddUserParentCommandError",
    "AddUserParentCommandErrorCode",
    "AddUserParentCommandReceipt",
    "AddUserParentCommandService",
    "AddUserParentOperation",
    "create_user_parent_identity",
    "SplitUserParentCommand",
    "SplitUserParentCommandError",
    "SplitUserParentCommandErrorCode",
    "SplitUserParentCommandReceipt",
    "SplitUserParentCommandService",
    "SplitUserParentOperation",
    "SplitUserParentOrientation",
    "split_user_parent_bboxes",
    "MergePipelineParentsCommand",
    "MergePipelineParentsCommandError",
    "MergePipelineParentsCommandErrorCode",
    "MergePipelineParentsCommandReceipt",
    "MergePipelineParentsCommandService",
    "MergePipelineParentsOperation",
    "merged_pipeline_parent_bbox",
    "EditHistoryCommand",
    "EditHistoryCommandError",
    "EditHistoryCommandErrorCode",
    "EditHistoryCommandReceipt",
    "EditHistoryCommandService",
    "EditHistoryOperation",
    "ParentGeometryCommand",
    "ParentGeometryCommandError",
    "ParentGeometryCommandErrorCode",
    "ParentGeometryCommandReceipt",
    "ParentGeometryCommandService",
    "ParentGeometryOperation",
    "page_canvas_size_for_project_page",
    "ReadingOrderCommand",
    "ReadingOrderCommandError",
    "ReadingOrderCommandErrorCode",
    "ReadingOrderCommandReceipt",
    "ReadingOrderCommandService",
    "ReadingOrderOperation",
    "ParentMembershipCommand",
    "ParentMembershipCommandError",
    "ParentMembershipCommandErrorCode",
    "ParentMembershipCommandReceipt",
    "ParentMembershipCommandService",
    "ParentMembershipOperation",
    "RenderLayoutLineHeightCommand",
    "RenderLayoutLineHeightCommandError",
    "RenderLayoutLineHeightCommandErrorCode",
    "RenderLayoutLineHeightCommandReceipt",
    "RenderLayoutLineHeightCommandService",
    "RenderLayoutLineHeightOperation",
    "RenderLayoutRotationCommand",
    "RenderLayoutRotationCommandError",
    "RenderLayoutRotationCommandErrorCode",
    "RenderLayoutRotationCommandReceipt",
    "RenderLayoutRotationCommandService",
    "RenderLayoutRotationOperation",
    "RenderLayoutRenderBoxCommand",
    "RenderLayoutRenderBoxCommandError",
    "RenderLayoutRenderBoxCommandErrorCode",
    "RenderLayoutRenderBoxCommandReceipt",
    "RenderLayoutRenderBoxCommandService",
    "RenderLayoutRenderBoxOperation",
    "RenderLayoutWritingModeCommand",
    "RenderLayoutWritingModeCommandError",
    "RenderLayoutWritingModeCommandErrorCode",
    "RenderLayoutWritingModeCommandReceipt",
    "RenderLayoutWritingModeCommandService",
    "RenderLayoutWritingModeOperation",
    "RenderStyleFillColorCommand",
    "RenderStyleFillColorCommandError",
    "RenderStyleFillColorCommandErrorCode",
    "RenderStyleFillColorCommandReceipt",
    "RenderStyleFillColorCommandService",
    "RenderStyleFillColorOperation",
    "RenderStyleOutlineColorCommand",
    "RenderStyleOutlineColorCommandError",
    "RenderStyleOutlineColorCommandErrorCode",
    "RenderStyleOutlineColorCommandReceipt",
    "RenderStyleOutlineColorCommandService",
    "RenderStyleOutlineColorOperation",
    "RenderStyleOutlineWidthCommand",
    "RenderStyleOutlineWidthCommandError",
    "RenderStyleOutlineWidthCommandErrorCode",
    "RenderStyleOutlineWidthCommandReceipt",
    "RenderStyleOutlineWidthCommandService",
    "RenderStyleOutlineWidthOperation",
    "RenderStylePreferredSizeCommand",
    "RenderStylePreferredSizeCommandError",
    "RenderStylePreferredSizeCommandErrorCode",
    "RenderStylePreferredSizeCommandReceipt",
    "RenderStylePreferredSizeCommandService",
    "RenderStylePreferredSizeOperation",
    "RenderStyleShadowVisibilityCommand",
    "RenderStyleShadowVisibilityCommandError",
    "RenderStyleShadowVisibilityCommandErrorCode",
    "RenderStyleShadowVisibilityCommandReceipt",
    "RenderStyleShadowVisibilityCommandService",
    "RenderStyleShadowVisibilityOperation",
    "RenderStyleShadowBlurCommand",
    "RenderStyleShadowBlurCommandError",
    "RenderStyleShadowBlurCommandErrorCode",
    "RenderStyleShadowBlurCommandReceipt",
    "RenderStyleShadowBlurCommandService",
    "RenderStyleShadowBlurOperation",
    "RenderStyleShadowColorCommand",
    "RenderStyleShadowColorCommandError",
    "RenderStyleShadowColorCommandErrorCode",
    "RenderStyleShadowColorCommandReceipt",
    "RenderStyleShadowColorCommandService",
    "RenderStyleShadowColorOperation",
    "RenderStyleShadowOffsetCommand",
    "RenderStyleShadowOffsetCommandError",
    "RenderStyleShadowOffsetCommandErrorCode",
    "RenderStyleShadowOffsetCommandReceipt",
    "RenderStyleShadowOffsetCommandService",
    "RenderStyleShadowOffsetOperation",
    "RenderStyleFontRoleCommand",
    "RenderStyleFontRoleCommandError",
    "RenderStyleFontRoleCommandErrorCode",
    "RenderStyleFontRoleCommandReceipt",
    "RenderStyleFontRoleCommandService",
    "RenderStyleFontRoleOperation",
    "RenderStyleFontWeightTierCommand",
    "RenderStyleFontWeightTierCommandError",
    "RenderStyleFontWeightTierCommandErrorCode",
    "RenderStyleFontWeightTierCommandReceipt",
    "RenderStyleFontWeightTierCommandService",
    "RenderStyleFontWeightTierOperation",
    "SourceTextCommand",
    "SourceTextCommandError",
    "SourceTextCommandErrorCode",
    "SourceTextCommandReceipt",
    "SourceTextCommandService",
    "SourceTextOperation",
    "SourceTextRevisionBaseV1",
    "TargetTextCommand",
    "TargetTextCommandError",
    "TargetTextCommandErrorCode",
    "TargetTextCommandReceipt",
    "TargetTextCommandService",
    "TargetTextOperation",
    "TargetTextRevisionBaseV1",
]
