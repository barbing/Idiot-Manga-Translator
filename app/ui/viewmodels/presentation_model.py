# -*- coding: utf-8 -*-
"""Framework-neutral GUI-5 presentation projection.

The executable Hybrid Pro contract owns labels, tones, icons, and precedence.
This module only supplies typed inputs and immutable aggregate snapshots for Qt
models.  Parent selection is deliberately absent from every page-health input.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.project_edits.projection import (
    EffectivePageSnapshot,
    ProjectionIssueKind,
    TargetFreshness,
)
from app.ui.ui_contract import (
    ArtifactState,
    CleanupState,
    PageState,
    Presentation,
    PreviewState,
    WorkspacePagePresentation,
    resolve_editor_status_presentation,
    resolve_workspace_page_presentation,
)


# Public semantic alias used by Workspace view-model consumers.  The contract
# remains owned by ui_contract.py rather than copied into this module.
WorkspacePageStatus = WorkspacePagePresentation


def _enum_value(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class PagePresentationInput:
    """All aggregate state needed to present one page.

    ``selected_parent_id`` is intentionally not a field.  Selecting or
    inspecting a parent can therefore never alter aggregate page health.
    """

    page_state: PageState = PageState.NORMAL
    required_artifact_state: ArtifactState = ArtifactState.VALID
    displayed_final_artifact_state: ArtifactState = ArtifactState.VALID
    preview_state: PreviewState = PreviewState.IDLE
    cleanup_state: CleanupState = CleanupState.IDLE
    excluded: bool = False
    page_dirty: bool = False
    stale: bool = False
    has_warnings: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "page_state", PageState(self.page_state))
        object.__setattr__(
            self,
            "required_artifact_state",
            ArtifactState(self.required_artifact_state),
        )
        object.__setattr__(
            self,
            "displayed_final_artifact_state",
            ArtifactState(self.displayed_final_artifact_state),
        )
        object.__setattr__(self, "preview_state", PreviewState(self.preview_state))
        object.__setattr__(self, "cleanup_state", CleanupState(self.cleanup_state))
        for field_name in ("excluded", "page_dirty", "stale", "has_warnings"):
            object.__setattr__(
                self,
                field_name,
                _require_bool(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True, slots=True)
class PagePresentationSnapshot:
    """Immutable Workspace and Editor projections of the same page input."""

    source: PagePresentationInput
    workspace: WorkspacePagePresentation
    editor: Presentation
    needs_review: bool
    accessibility_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, PagePresentationInput):
            raise TypeError("source must be PagePresentationInput")
        if not isinstance(self.workspace, WorkspacePagePresentation):
            raise TypeError("workspace must be WorkspacePagePresentation")
        if not isinstance(self.editor, Presentation):
            raise TypeError("editor must be Presentation")
        _require_bool(self.needs_review, "needs_review")
        if not isinstance(self.accessibility_text, str) or not self.accessibility_text.strip():
            raise ValueError("accessibility_text must not be empty")


def build_page_presentation(
    source: PagePresentationInput,
) -> PagePresentationSnapshot:
    """Resolve Workspace and Editor state through the sole UI contract."""

    if not isinstance(source, PagePresentationInput):
        raise TypeError("source must be PagePresentationInput")
    workspace = resolve_workspace_page_presentation(source.page_state)
    editor = resolve_editor_status_presentation(
        page_state=source.page_state,
        required_artifact_state=source.required_artifact_state,
        displayed_final_artifact_state=source.displayed_final_artifact_state,
        preview_state=source.preview_state,
        cleanup_state=source.cleanup_state,
        excluded=source.excluded,
        page_dirty=source.page_dirty,
        stale=source.stale,
        has_warnings=source.has_warnings,
    )
    tone = _enum_value(editor.tone)
    needs_review = tone in {"warning", "error"}
    accessibility_text = (
        f"{editor.label}. Workflow owner: {workspace.owner}. "
        f"Workspace status: {workspace.label}."
    )
    return PagePresentationSnapshot(
        source=source,
        workspace=workspace,
        editor=editor,
        needs_review=needs_review,
        accessibility_text=accessibility_text,
    )


_CONFLICT_ISSUES = frozenset(
    {
        ProjectionIssueKind.CONFLICT,
        ProjectionIssueKind.ORPHANED,
    }
)
_ERROR_ISSUES = frozenset({ProjectionIssueKind.INVALID_EFFECTIVE_VALUE})
_MISSING_ISSUES = frozenset({ProjectionIssueKind.MISSING_DEPENDENCY})
_STALE_ISSUES = frozenset(
    {
        ProjectionIssueKind.STALE_EDIT_BASE,
        ProjectionIssueKind.STALE_DEPENDENCY,
    }
)


def aggregate_page_state(snapshot: EffectivePageSnapshot) -> PageState:
    """Derive selection-invariant page state from the full effective page."""

    if not isinstance(snapshot, EffectivePageSnapshot):
        raise TypeError("snapshot must be EffectivePageSnapshot")
    issue_kinds = frozenset(issue.kind for issue in snapshot.issues)
    issue_kinds |= frozenset(
        issue.kind for parent in snapshot.parents for issue in parent.issues
    )
    if issue_kinds & _CONFLICT_ISSUES:
        return PageState.CONFLICT
    if issue_kinds & _ERROR_ISSUES:
        return PageState.ERROR
    if issue_kinds & _MISSING_ISSUES:
        return PageState.MISSING
    if issue_kinds & _STALE_ISSUES or any(
        parent.target_freshness is TargetFreshness.STALE
        for parent in snapshot.parents
    ):
        return PageState.STALE
    return PageState.NORMAL


def page_presentation_input_from_effective_snapshot(
    snapshot: EffectivePageSnapshot,
    *,
    page_state: PageState | str | None = None,
    required_artifact_state: ArtifactState = ArtifactState.VALID,
    displayed_final_artifact_state: ArtifactState = ArtifactState.VALID,
    preview_state: PreviewState = PreviewState.IDLE,
    cleanup_state: CleanupState = CleanupState.IDLE,
) -> PagePresentationInput:
    """Build aggregate presentation input without reading a selected parent."""

    if not isinstance(snapshot, EffectivePageSnapshot):
        raise TypeError("snapshot must be EffectivePageSnapshot")
    derived_state = aggregate_page_state(snapshot) if page_state is None else PageState(page_state)
    parents = tuple(snapshot.parents)
    excluded = bool(parents) and all(parent.excluded for parent in parents)
    stale = derived_state is PageState.STALE or any(
        parent.target_freshness is TargetFreshness.STALE for parent in parents
    )
    has_warnings = bool(snapshot.issues) or any(parent.issues for parent in parents)
    return PagePresentationInput(
        page_state=derived_state,
        required_artifact_state=required_artifact_state,
        displayed_final_artifact_state=displayed_final_artifact_state,
        preview_state=preview_state,
        cleanup_state=cleanup_state,
        excluded=excluded,
        page_dirty=bool(snapshot.applied_edit_ids),
        stale=stale,
        has_warnings=has_warnings,
    )


__all__ = [
    "PagePresentationInput",
    "PagePresentationSnapshot",
    "WorkspacePageStatus",
    "aggregate_page_state",
    "build_page_presentation",
    "page_presentation_input_from_effective_snapshot",
]
