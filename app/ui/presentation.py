# -*- coding: utf-8 -*-
"""Pure, shared UX presentation rules for the production Qt surfaces.

This module deliberately owns only user-facing labels and priorities.  It does
not interpret pipeline artifacts, mutate settings, or decide command
eligibility; callers pass already-typed application facts into these helpers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NextActionPresentation:
    """One plain-language next action rendered by a GUI surface."""

    label: str
    detail: str
    tone: str = "info"


def provider_lifecycle_summary(
    *,
    configured: bool,
    tested: bool,
    active: bool,
) -> str:
    """Describe the provider lifecycle without exposing transport internals."""

    setup = "Configured" if configured else "Needs setup"
    validation = "Validated" if tested else "Not validated"
    activation = "Active for translation" if active else "Not active"
    return f"{setup} · {validation} · {activation}"


def workspace_next_action(
    *,
    run_detail: str,
    busy: bool,
    can_start: bool,
    start_reason: str,
    has_project: bool,
    has_pages: bool,
    recovery_required: bool,
) -> NextActionPresentation:
    """Return the single highest-priority action for the Workspace summary."""

    detail = str(run_detail or "").strip()
    reason = str(start_reason or "").strip()
    if recovery_required:
        return NextActionPresentation(
            "Recover the completed project",
            detail or "Use the recovery action before starting another run.",
            "error",
        )
    if busy:
        return NextActionPresentation(
            "Run in progress",
            detail or "The current page will settle before the next page begins.",
            "info",
        )
    if can_start:
        return NextActionPresentation(
            "Start translation",
            reason or "Memory is checked before any model loads.",
            "ready",
        )
    if not has_project:
        return NextActionPresentation(
            "Open or create a project",
            "Choose source pages from Project Hub before starting translation.",
            "muted",
        )
    if not has_pages:
        return NextActionPresentation(
            "Add source pages",
            "This project has no source pages to translate.",
            "warning",
        )
    return NextActionPresentation(
        "Resolve the next requirement",
        reason or detail or "Open the selected page to review what is required next.",
        "warning",
    )


def editor_preview_action(tab: str) -> tuple[str, str]:
    """Return the primary preview label and its accessible explanation."""

    normalized = str(tab or "text").strip().casefold()
    if normalized == "style":
        return (
            "Preview style on page",
            "Preview the saved effective style on the current page.",
        )
    if normalized == "layout":
        return (
            "Preview layout on page",
            "Preview the saved effective layout on the current page.",
        )
    return (
        "Preview final page",
        "Preview the current saved text, style, and layout as final page pixels.",
    )


__all__ = [
    "NextActionPresentation",
    "editor_preview_action",
    "provider_lifecycle_summary",
    "workspace_next_action",
]
