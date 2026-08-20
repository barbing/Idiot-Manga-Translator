# -*- coding: utf-8 -*-
"""Framework-neutral presentation and responsive-layout contracts.

This module is the native counterpart of the accepted Hybrid Pro
``uiContract.js``.  It contains identifiers and pure policy only: importing it
must never import Qt, project services, Torch, or pipeline implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class StateDomain(_StringEnum):
    PAGE = "page"
    ARTIFACT = "artifact"
    ASYNC = "async"
    PREVIEW = "preview"
    CLEANUP = "cleanup"
    AUTHORITY = "authority"


class PageState(_StringEnum):
    QUEUED = "queued"
    NORMAL = "normal"
    STALE = "stale"
    CONFLICT = "conflict"
    ERROR = "error"
    MISSING = "missing"
    RECOVERY = "recovery"
    LOADING = "loading"
    RECOVERING = "recovering"


class ArtifactState(_StringEnum):
    VALID = "valid"
    MISSING = "missing"
    INVALID = "invalid"
    STALE = "stale"
    PROTOTYPE_STANDIN = "prototype-standin"


class AsyncState(_StringEnum):
    IDLE = "idle"
    LOADING = "loading"
    SUCCESS = "success"


class PreviewState(_StringEnum):
    IDLE = "idle"
    PREVIEWING = "previewing"
    READY = "ready"


class CleanupState(_StringEnum):
    IDLE = "idle"
    PREVIEWING = "previewing"
    PREVIEW_READY = "preview-ready"
    COMMITTING = "committing"
    COMMITTED = "committed"


class Authority(_StringEnum):
    AUTOMATIC = "automatic"
    USER_EDIT = "user-edit"
    USER_RETAINED = "user-retained"


class PresentationTone(_StringEnum):
    READY = "ready"
    EDITING = "editing"
    WARNING = "warning"
    ERROR = "error"
    MUTED = "muted"
    QUEUED = "queued"
    INFO = "info"


class WidthTier(_StringEnum):
    NARROW = "narrow"
    COMPACT = "compact"
    STANDARD = "standard"
    WIDE = "wide"


class HeightTier(_StringEnum):
    SHORT = "short"
    STANDARD = "standard"


class FontScaleTier(_StringEnum):
    NOMINAL = "nominal"
    LARGE = "large"
    MAX = "max"


class Density(_StringEnum):
    COMFORTABLE = "comfortable"
    COMPACT = "compact"


NAVIGATION_IDS = ("hub", "workspace", "editor", "settings")
INSPECTOR_TAB_IDS = ("text", "style", "layout", "cleanup", "history")
CANVAS_VIEW_IDS = ("original", "cleaned", "final", "compare")
ACTIVITY_FACET_IDS = ("project", "run", "page", "runtime")
OVERLAY_IDS = (
    "parentBounds",
    "renderBox",
    "sourceFootprint",
    "baseline",
    "cleanupMask",
    "protectedRegions",
    "proof",
)

PAGE_STATE_IDS = tuple(item.value for item in PageState)
ARTIFACT_STATE_IDS = tuple(item.value for item in ArtifactState)
ASYNC_STATE_IDS = tuple(item.value for item in AsyncState)
PREVIEW_STATE_IDS = tuple(item.value for item in PreviewState)
CLEANUP_STATE_IDS = tuple(item.value for item in CleanupState)
AUTHORITY_IDS = tuple(item.value for item in Authority)
PRESENTATION_TONE_IDS = tuple(item.value for item in PresentationTone)
WIDTH_TIER_IDS = tuple(item.value for item in WidthTier)
HEIGHT_TIER_IDS = tuple(item.value for item in HeightTier)
FONT_SCALE_TIER_IDS = tuple(item.value for item in FontScaleTier)
DENSITY_IDS = tuple(item.value for item in Density)
SUPPORTED_FONT_SCALES = (100, 125, 150, 175, 200)


@dataclass(frozen=True, slots=True)
class Presentation:
    label: str
    tone: str
    icon: str


@dataclass(frozen=True, slots=True)
class WorkspacePagePresentation:
    label: str
    tone: str
    icon: str
    owner: str


@dataclass(frozen=True, slots=True)
class CleanupActivityContext:
    visible: bool
    active: bool
    attention: bool
    label: str
    tone: str


@dataclass(frozen=True, slots=True)
class LayoutMode:
    width: float
    height: float
    font_scale: float
    density: str
    width_tier: str
    height_tier: str
    font_scale_tier: str
    composition_tier: str
    wide_composition: bool
    short_viewport: bool
    scrollable_toolbar: bool
    accessible_reflow: bool
    maximum_reflow: bool


@dataclass(frozen=True, slots=True)
class ActivityDockBounds:
    min: int
    preferred: int
    max: int
    resizable: bool


def _presentation(label: str, tone: str, icon: str) -> Presentation:
    return Presentation(label=label, tone=tone, icon=icon)


_DOMAIN_STATE_IDS_MUTABLE: dict[str, tuple[str, ...]] = {
    "page": PAGE_STATE_IDS,
    "artifact": ARTIFACT_STATE_IDS,
    "async": ASYNC_STATE_IDS,
    "preview": PREVIEW_STATE_IDS,
    "cleanup": CLEANUP_STATE_IDS,
    "authority": AUTHORITY_IDS,
}
DOMAIN_STATE_IDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    _DOMAIN_STATE_IDS_MUTABLE
)

_STATE_PRESENTATIONS_MUTABLE: dict[str, Mapping[str, Presentation]] = {
    "page": MappingProxyType(
        {
            "queued": _presentation("Queued", "queued", "clock"),
            "normal": _presentation("Ready", "ready", "check-circle"),
            "stale": _presentation("Source changed", "warning", "warning"),
            "conflict": _presentation("Conflict", "error", "x-circle"),
            "error": _presentation("Stage error", "error", "x-circle"),
            "missing": _presentation("Missing artifact", "warning", "warning"),
            "recovery": _presentation(
                "Recovery available", "warning", "lifebuoy"
            ),
            "loading": _presentation("Running", "editing", "spinner"),
            "recovering": _presentation(
                "Recovering base", "editing", "spinner"
            ),
        }
    ),
    "artifact": MappingProxyType(
        {
            "valid": _presentation("Valid", "ready", "check-circle"),
            "missing": _presentation("Missing", "warning", "warning"),
            "invalid": _presentation("Invalid", "error", "x-circle"),
            "stale": _presentation("Stale", "warning", "warning"),
            "prototype-standin": _presentation(
                "Prototype stand-in", "warning", "lifebuoy"
            ),
        }
    ),
    "async": MappingProxyType(
        {
            "idle": _presentation("Idle", "muted", "circle"),
            "loading": _presentation("Running", "editing", "spinner"),
            "success": _presentation("Complete", "ready", "check-circle"),
        }
    ),
    "preview": MappingProxyType(
        {
            "idle": _presentation("No preview pending", "muted", "circle"),
            "previewing": _presentation(
                "Rendering preview", "editing", "spinner"
            ),
            "ready": _presentation("Preview ready", "ready", "check-circle"),
        }
    ),
    "cleanup": MappingProxyType(
        {
            "idle": _presentation("No cleanup draft", "muted", "circle"),
            "previewing": _presentation(
                "Creating temporary preview", "editing", "spinner"
            ),
            "preview-ready": _presentation(
                "Preview ready for review", "editing", "eye"
            ),
            "committing": _presentation(
                "Committing immutable revision", "editing", "spinner"
            ),
            "committed": _presentation(
                "Cleanup revision committed", "ready", "check-circle"
            ),
        }
    ),
    "authority": MappingProxyType(
        {
            "automatic": _presentation("Automatic", "muted", "robot"),
            "user-edit": _presentation("Your edit", "editing", "pencil"),
            "user-retained": _presentation(
                "Retained target", "ready", "bookmark"
            ),
        }
    ),
}
STATE_PRESENTATIONS: Mapping[str, Mapping[str, Presentation]] = MappingProxyType(
    _STATE_PRESENTATIONS_MUTABLE
)

_WORKSPACE_PAGE_OWNERS: Mapping[str, str] = MappingProxyType(
    {
        "queued": "Pipeline",
        "normal": "Complete",
        "stale": "User review",
        "conflict": "User review",
        "error": "Translation",
        "missing": "Cleanup",
        "recovery": "Translation",
        "loading": "Translation",
        "recovering": "Cleanup",
    }
)


def _string_id(value: Any) -> str:
    return value.value if isinstance(value, _StringEnum) else value


def resolve_state_presentation(
    domain: str | StateDomain,
    state: str | _StringEnum,
) -> Presentation:
    """Return the canonical label, tone, and icon for one domain state."""

    domain_id = _string_id(domain)
    state_id = _string_id(state)
    states = DOMAIN_STATE_IDS.get(domain_id)
    if states is None:
        raise ValueError(f"Unknown state domain: {domain_id}")
    if state_id not in states:
        raise ValueError(f"Unknown {domain_id} state: {state_id}")
    return STATE_PRESENTATIONS[domain_id][state_id]


def resolve_workspace_page_presentation(
    page_state: str | PageState,
) -> WorkspacePagePresentation:
    """Resolve Workspace status by owning stage, not Editor artifact health."""

    page_state_id = _string_id(page_state)
    state = resolve_state_presentation(StateDomain.PAGE, page_state_id)
    return WorkspacePagePresentation(
        label=state.label,
        tone=state.tone,
        icon=state.icon,
        owner=_WORKSPACE_PAGE_OWNERS[page_state_id],
    )


def resolve_editor_status_presentation(
    *,
    page_state: str | PageState = "normal",
    required_artifact_state: str | ArtifactState = "valid",
    displayed_final_artifact_state: str | ArtifactState = "valid",
    preview_state: str | PreviewState = "idle",
    cleanup_state: str | CleanupState = "idle",
    excluded: bool = False,
    page_dirty: bool = False,
    stale: bool = False,
    has_warnings: bool = False,
) -> Presentation:
    """Resolve the Editor composite status with one cross-platform precedence."""

    page_state_id = _string_id(page_state)
    required_artifact_id = _string_id(required_artifact_state)
    displayed_artifact_id = _string_id(displayed_final_artifact_state)
    preview_state_id = _string_id(preview_state)
    cleanup_state_id = _string_id(cleanup_state)
    resolve_state_presentation("page", page_state_id)
    resolve_state_presentation("artifact", required_artifact_id)
    resolve_state_presentation("artifact", displayed_artifact_id)
    resolve_state_presentation("preview", preview_state_id)
    resolve_state_presentation("cleanup", cleanup_state_id)

    if excluded:
        return _presentation("Excluded", "muted", "prohibit")
    if page_state_id in {"conflict", "error"}:
        return resolve_state_presentation("page", page_state_id)
    if required_artifact_id in {"missing", "invalid"}:
        return resolve_state_presentation("artifact", required_artifact_id)
    if page_state_id in {"queued", "missing", "recovering", "loading"}:
        return resolve_state_presentation("page", page_state_id)
    if preview_state_id == "previewing":
        return resolve_state_presentation("preview", preview_state_id)
    if cleanup_state_id in {"previewing", "committing"}:
        return resolve_state_presentation("cleanup", cleanup_state_id)
    if page_state_id == "stale":
        return resolve_state_presentation("page", page_state_id)
    if page_dirty or stale or has_warnings:
        return _presentation("Review needed", "warning", "warning")
    if required_artifact_id in {"stale", "prototype-standin"}:
        return resolve_state_presentation("artifact", required_artifact_id)
    if displayed_artifact_id in {"stale", "prototype-standin"}:
        return resolve_state_presentation("artifact", displayed_artifact_id)
    if page_state_id == "recovery":
        return resolve_state_presentation("page", page_state_id)
    if preview_state_id == "ready":
        return resolve_state_presentation("preview", preview_state_id)
    return resolve_state_presentation("page", "normal")


def activity_cleanup_context(
    *,
    cleanup_state: str | CleanupState = "idle",
    cleanup_dirty: bool = False,
    required_artifact_state: str | ArtifactState = "valid",
    inspector_tab: str = "text",
) -> CleanupActivityContext:
    """Resolve contextual Cleanup prominence without adding a fifth facet."""

    cleanup_state_id = _string_id(cleanup_state)
    artifact_state_id = _string_id(required_artifact_state)
    cleanup_presentation = resolve_state_presentation(
        "cleanup", cleanup_state_id
    )
    artifact_presentation = resolve_state_presentation(
        "artifact", artifact_state_id
    )
    if inspector_tab not in INSPECTOR_TAB_IDS:
        raise ValueError(f"Unknown inspector tab: {inspector_tab}")
    if not isinstance(cleanup_dirty, bool):
        raise TypeError("cleanup_dirty must be a boolean")

    active = inspector_tab == "cleanup"
    artifact_needs_attention = artifact_state_id != "valid"
    cleanup_has_state = cleanup_state_id != "idle"
    visible = (
        active or cleanup_dirty or cleanup_has_state or artifact_needs_attention
    )
    attention = cleanup_dirty or artifact_needs_attention

    if artifact_needs_attention:
        resolved = artifact_presentation
    elif cleanup_has_state:
        resolved = cleanup_presentation
    elif cleanup_dirty:
        resolved = _presentation(
            "Cleanup draft changed", "warning", "warning"
        )
    elif active:
        resolved = _presentation(
            "Cleanup workspace", "editing", "paint-brush"
        )
    else:
        resolved = _presentation(
            "Cleanup available", "muted", "paint-brush"
        )

    return CleanupActivityContext(
        visible=visible,
        active=active,
        attention=attention,
        label=resolved.label,
        tone=resolved.tone,
    )


def _finite_number(value: Any, name: str, minimum: float) -> float:
    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be a finite number greater than or equal to {minimum}"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a finite number greater than or equal to {minimum}"
        ) from exc
    if not math.isfinite(number) or number < minimum:
        raise ValueError(
            f"{name} must be a finite number greater than or equal to {minimum}"
        )
    return number


def resolve_layout_mode(
    *,
    width: float,
    height: float,
    font_scale: float = 100,
    density: str | Density = "comfortable",
) -> LayoutMode:
    """Resolve orthogonal width, height, font-scale, and density modes."""

    viewport_width = _finite_number(width, "width", 320)
    viewport_height = _finite_number(height, "height", 320)
    application_font_scale = _finite_number(font_scale, "font_scale", 100)
    if application_font_scale > 200:
        raise ValueError("font_scale must be less than or equal to 200")
    density_id = _string_id(density)
    if density_id not in DENSITY_IDS:
        raise ValueError(f"Unknown density: {density_id}")

    if viewport_width >= 1850:
        width_tier = "wide"
    elif viewport_width <= 1150:
        width_tier = "narrow"
    elif viewport_width <= 1450:
        width_tier = "compact"
    else:
        width_tier = "standard"

    height_tier = "short" if viewport_height <= 820 else "standard"
    if application_font_scale >= 180:
        font_scale_tier = "max"
    elif application_font_scale >= 150:
        font_scale_tier = "large"
    else:
        font_scale_tier = "nominal"

    wide_composition = viewport_width >= 1850 and viewport_height >= 900
    if wide_composition:
        composition_tier = "wide"
    elif viewport_width <= 1150:
        composition_tier = "narrow"
    elif viewport_width <= 1450:
        composition_tier = "compact"
    else:
        composition_tier = "standard"

    return LayoutMode(
        width=viewport_width,
        height=viewport_height,
        font_scale=application_font_scale,
        density=density_id,
        width_tier=width_tier,
        height_tier=height_tier,
        font_scale_tier=font_scale_tier,
        composition_tier=composition_tier,
        wide_composition=wide_composition,
        short_viewport=height_tier == "short",
        scrollable_toolbar=application_font_scale >= 125,
        accessible_reflow=application_font_scale >= 150,
        maximum_reflow=application_font_scale >= 180,
    )


def _checked_dock_bounds(
    minimum: int,
    preferred: int,
    maximum: int,
) -> ActivityDockBounds:
    if not minimum <= preferred <= maximum:
        raise ValueError(
            "Invalid Activity dock bounds: "
            f"{minimum} <= {preferred} <= {maximum}"
        )
    return ActivityDockBounds(
        min=minimum,
        preferred=preferred,
        max=maximum,
        resizable=maximum - minimum >= 32,
    )


def _layout_mode(value: LayoutMode | Mapping[str, Any]) -> LayoutMode:
    if isinstance(value, LayoutMode):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("viewport_or_mode must be a LayoutMode or mapping")
    return resolve_layout_mode(
        width=value.get("width"),
        height=value.get("height"),
        font_scale=value.get("font_scale", value.get("fontScale", 100)),
        density=value.get("density", "comfortable"),
    )


def resolve_activity_dock_bounds(
    viewport_or_mode: LayoutMode | Mapping[str, Any],
) -> ActivityDockBounds:
    """Return the sole canonical Activity height contract in logical pixels."""

    mode = _layout_mode(viewport_or_mode)
    if mode.height_tier == "short":
        return _checked_dock_bounds(260, 260, 260)
    if mode.wide_composition:
        return _checked_dock_bounds(320, 320, 360)
    if mode.width <= 1280:
        return _checked_dock_bounds(296, 296, 296)
    return _checked_dock_bounds(324, 324, 360)


def clamp_activity_dock_height(
    value: Any,
    viewport_or_mode: LayoutMode | Mapping[str, Any],
) -> int:
    """Clamp persisted Activity geometry through the rendering policy."""

    bounds = resolve_activity_dock_bounds(viewport_or_mode)
    try:
        requested = float(value)
    except (TypeError, ValueError):
        requested = math.nan
    fallback = requested if math.isfinite(requested) else bounds.preferred
    rounded = math.floor(fallback + 0.5) if fallback >= 0 else math.ceil(fallback - 0.5)
    return min(bounds.max, max(bounds.min, rounded))


__all__ = [
    "ACTIVITY_FACET_IDS",
    "ARTIFACT_STATE_IDS",
    "ASYNC_STATE_IDS",
    "AUTHORITY_IDS",
    "CANVAS_VIEW_IDS",
    "CLEANUP_STATE_IDS",
    "DENSITY_IDS",
    "DOMAIN_STATE_IDS",
    "FONT_SCALE_TIER_IDS",
    "HEIGHT_TIER_IDS",
    "INSPECTOR_TAB_IDS",
    "NAVIGATION_IDS",
    "OVERLAY_IDS",
    "PAGE_STATE_IDS",
    "PREVIEW_STATE_IDS",
    "PRESENTATION_TONE_IDS",
    "STATE_PRESENTATIONS",
    "SUPPORTED_FONT_SCALES",
    "WIDTH_TIER_IDS",
    "ActivityDockBounds",
    "ArtifactState",
    "AsyncState",
    "Authority",
    "CleanupActivityContext",
    "CleanupState",
    "Density",
    "FontScaleTier",
    "HeightTier",
    "LayoutMode",
    "PageState",
    "Presentation",
    "PresentationTone",
    "PreviewState",
    "StateDomain",
    "WidthTier",
    "WorkspacePagePresentation",
    "activity_cleanup_context",
    "clamp_activity_dock_height",
    "resolve_activity_dock_bounds",
    "resolve_editor_status_presentation",
    "resolve_layout_mode",
    "resolve_state_presentation",
    "resolve_workspace_page_presentation",
]
