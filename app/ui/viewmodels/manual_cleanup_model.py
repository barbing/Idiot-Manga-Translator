# -*- coding: utf-8 -*-
"""Framework-neutral mask document and state for manual cleanup editing."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import io
import math
import os
from typing import Sequence
import uuid

from PIL import Image, ImageDraw

from app.project_edits.manual_cleanup import (
    ManualCleanupAvailabilityCode,
    ManualCleanupContext,
    ManualCleanupFailureCode,
    ManualCleanupParameters,
    ManualCleanupPreflight,
    ManualCleanupPreviewLease,
    ManualCleanupProgress,
    ManualCleanupRebaseReview,
    ManualCleanupReceipt,
    ManualCleanupStage,
    ManualCleanupStatus,
    UserParentCleanupCoverageTargetV1,
)


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SHA256_ZERO = "0" * 64


def _canonical_coverage_target(
    value: UserParentCleanupCoverageTargetV1 | None,
) -> UserParentCleanupCoverageTargetV1 | None:
    """Round-trip the public target so UI commands retain canonical bytes."""

    if value is None:
        return None
    if not isinstance(value, UserParentCleanupCoverageTargetV1):
        raise TypeError(
            "coverage_target must be a UserParentCleanupCoverageTargetV1 or None"
        )
    return UserParentCleanupCoverageTargetV1.from_dict(value.to_dict())


class ManualCleanupTool(str, Enum):
    RECTANGLE = "rectangle"
    LASSO = "lasso"
    BRUSH = "brush"
    ERASER = "eraser"
    PROTECT = "protect"


class ManualCleanupMaskLayer(str, Enum):
    ERASE = "erase"
    PROTECT = "protect"


class ManualCleanupMaskAction(str, Enum):
    ADD = "add"
    SUBTRACT = "subtract"


class ManualCleanupWorkerMode(str, Enum):
    PREVIEW = "preview"
    COMMIT = "commit"


class ManualCleanupViewPhase(str, Enum):
    IDLE = "idle"
    DIRTY = "dirty"
    PREVIEWING = "previewing"
    PREVIEW_READY = "preview_ready"
    COMMITTING = "committing"
    COMMITTED = "committed"
    STALE = "stale"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ManualCleanupWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    FINDING_PAGE = "finding_page"
    PROJECTING = "projecting"
    READING_EDIT_HEADS = "reading_edit_heads"
    PREFLIGHT = "preflight"
    PREVIEWING = "previewing"
    OPENING_COMMIT_STORE = "opening_commit_store"
    COMMITTING = "committing"
    CLOSING_EDIT_STORE = "closing_edit_store"


class ManualCleanupWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    WORKER_REUSED = "worker_reused"
    INVALID_REQUEST = ManualCleanupFailureCode.INVALID_REQUEST.value
    COVERAGE_TARGET_INVALID = ManualCleanupFailureCode.COVERAGE_TARGET_INVALID.value
    COVERAGE_TARGET_STALE = ManualCleanupFailureCode.COVERAGE_TARGET_STALE.value
    COVERAGE_CONFLICT = ManualCleanupFailureCode.COVERAGE_CONFLICT.value
    ORIGINAL_ASSET_MISMATCH = ManualCleanupFailureCode.ORIGINAL_ASSET_MISMATCH.value
    MISSING_BASE = ManualCleanupFailureCode.MISSING_BASE.value
    STALE_BASE = ManualCleanupFailureCode.STALE_BASE.value
    INVALID_MASK = ManualCleanupFailureCode.INVALID_MASK.value
    BACKEND_UNAVAILABLE = ManualCleanupFailureCode.BACKEND_UNAVAILABLE.value
    BACKEND_FAILED = ManualCleanupFailureCode.BACKEND_FAILED.value
    PREVIEW_STALE = ManualCleanupFailureCode.PREVIEW_STALE.value
    STORE_UNAVAILABLE = ManualCleanupFailureCode.STORE_UNAVAILABLE.value
    COMMIT_STALE = ManualCleanupFailureCode.COMMIT_STALE.value
    ARTIFACT_INVALID = ManualCleanupFailureCode.ARTIFACT_INVALID.value


@dataclass(frozen=True, slots=True)
class PagePoint:
    """One finite page-space point; bounds are enforced by the document."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if isinstance(self.x, bool) or isinstance(self.y, bool):
            raise TypeError("page coordinates must be numbers")
        x = float(self.x)
        y = float(self.y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("page coordinates must be finite")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)


@dataclass(frozen=True, slots=True)
class ManualCleanupMaskCommand:
    """Immutable drawing intent expressed only in original page coordinates."""

    command_id: str
    tool: ManualCleanupTool
    layer: ManualCleanupMaskLayer
    action: ManualCleanupMaskAction
    points: tuple[PagePoint, ...]
    radius_px: float = 0.0

    def __post_init__(self) -> None:
        command_id = str(self.command_id or "").strip()
        if not command_id:
            raise ValueError("command_id is required")
        if not isinstance(self.tool, ManualCleanupTool):
            raise TypeError("tool must be ManualCleanupTool")
        if not isinstance(self.layer, ManualCleanupMaskLayer):
            raise TypeError("layer must be ManualCleanupMaskLayer")
        if not isinstance(self.action, ManualCleanupMaskAction):
            raise TypeError("action must be ManualCleanupMaskAction")
        points = tuple(self.points)
        if any(not isinstance(point, PagePoint) for point in points):
            raise TypeError("points must contain PagePoint values")
        radius = float(self.radius_px)
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("radius_px must be finite and non-negative")

        if self.tool is ManualCleanupTool.RECTANGLE:
            if len(points) != 2:
                raise ValueError("rectangle requires exactly two points")
            if self.layer is not ManualCleanupMaskLayer.ERASE:
                raise ValueError("rectangle draws on the erase mask")
            if self.action is not ManualCleanupMaskAction.ADD:
                raise ValueError("rectangle must add to the erase mask")
            if points[0].x == points[1].x or points[0].y == points[1].y:
                raise ValueError("rectangle must have non-zero area")
            if radius:
                raise ValueError("rectangle does not use radius_px")
        elif self.tool is ManualCleanupTool.LASSO:
            if len(points) < 3:
                raise ValueError("lasso requires at least three points")
            if self.layer is not ManualCleanupMaskLayer.ERASE:
                raise ValueError("lasso draws on the erase mask")
            if self.action is not ManualCleanupMaskAction.ADD:
                raise ValueError("lasso must add to the erase mask")
            if (
                max(point.x for point in points) == min(point.x for point in points)
                or max(point.y for point in points) == min(point.y for point in points)
            ):
                raise ValueError("lasso must enclose non-zero bounds")
            if radius:
                raise ValueError("lasso does not use radius_px")
        else:
            if not points:
                raise ValueError("stroke tools require at least one point")
            if radius <= 0.0:
                raise ValueError("stroke tools require a positive radius_px")
            if self.tool is ManualCleanupTool.BRUSH:
                expected = (
                    ManualCleanupMaskLayer.ERASE,
                    ManualCleanupMaskAction.ADD,
                )
            elif self.tool is ManualCleanupTool.PROTECT:
                expected = (
                    ManualCleanupMaskLayer.PROTECT,
                    ManualCleanupMaskAction.ADD,
                )
            elif self.tool is ManualCleanupTool.ERASER:
                expected = (self.layer, ManualCleanupMaskAction.SUBTRACT)
            else:  # pragma: no cover - exhaustive enum defense
                raise ValueError(f"unsupported cleanup tool: {self.tool}")
            if (self.layer, self.action) != expected:
                raise ValueError("stroke tool layer/action contract is invalid")

        object.__setattr__(self, "command_id", command_id)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "radius_px", radius)

    @classmethod
    def create(
        cls,
        tool: ManualCleanupTool,
        points: Sequence[PagePoint | tuple[float, float]],
        *,
        eraser_layer: ManualCleanupMaskLayer = ManualCleanupMaskLayer.ERASE,
        radius_px: float = 0.0,
        command_id: str = "",
    ) -> "ManualCleanupMaskCommand":
        tool = ManualCleanupTool(tool)
        normalized = tuple(
            point if isinstance(point, PagePoint) else PagePoint(*point)
            for point in points
        )
        if tool is ManualCleanupTool.PROTECT:
            layer = ManualCleanupMaskLayer.PROTECT
            action = ManualCleanupMaskAction.ADD
        elif tool is ManualCleanupTool.ERASER:
            layer = ManualCleanupMaskLayer(eraser_layer)
            action = ManualCleanupMaskAction.SUBTRACT
        else:
            layer = ManualCleanupMaskLayer.ERASE
            action = ManualCleanupMaskAction.ADD
        return cls(
            command_id=command_id or uuid.uuid4().hex,
            tool=tool,
            layer=layer,
            action=action,
            points=normalized,
            radius_px=radius_px,
        )


@dataclass(frozen=True, slots=True)
class ManualCleanupMaskDocumentSnapshot:
    page_id: str
    canvas_size: tuple[int, int]
    commands: tuple[ManualCleanupMaskCommand, ...]
    revision: int
    undo_depth: int
    redo_depth: int

    @property
    def has_erase_marks(self) -> bool:
        return any(
            command.layer is ManualCleanupMaskLayer.ERASE
            and command.action is ManualCleanupMaskAction.ADD
            for command in self.commands
        )


class ManualCleanupMaskDocument:
    """Page-bound local command document with linear-memory undo/redo."""

    _CLEAR = object()

    def __init__(self, page_id: str, canvas_size: tuple[int, int]) -> None:
        page_id = str(page_id or "").strip()
        if not page_id:
            raise ValueError("page_id is required")
        if (
            not isinstance(canvas_size, tuple)
            or len(canvas_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in canvas_size)
            or canvas_size[0] <= 0
            or canvas_size[1] <= 0
        ):
            raise ValueError("canvas_size must contain two positive integers")
        self._page_id = page_id
        self._canvas_size = canvas_size
        self._history: list[ManualCleanupMaskCommand | object] = []
        self._cursor = 0
        self._revision = 0

    @property
    def snapshot(self) -> ManualCleanupMaskDocumentSnapshot:
        active: list[ManualCleanupMaskCommand] = []
        for item in self._history[: self._cursor]:
            if item is self._CLEAR:
                active.clear()
            elif isinstance(item, ManualCleanupMaskCommand):
                active.append(item)
            else:  # pragma: no cover - private history invariant
                raise RuntimeError("manual cleanup history is invalid")
        return ManualCleanupMaskDocumentSnapshot(
            page_id=self._page_id,
            canvas_size=self._canvas_size,
            commands=tuple(active),
            revision=self._revision,
            undo_depth=self._cursor,
            redo_depth=len(self._history) - self._cursor,
        )

    def append(self, command: ManualCleanupMaskCommand) -> ManualCleanupMaskDocumentSnapshot:
        if not isinstance(command, ManualCleanupMaskCommand):
            raise TypeError("command must be ManualCleanupMaskCommand")
        width, height = self._canvas_size
        if any(
            point.x < 0.0
            or point.y < 0.0
            or point.x > width
            or point.y > height
            for point in command.points
        ):
            raise ValueError("mask command extends outside page coordinates")
        if any(
            isinstance(existing, ManualCleanupMaskCommand)
            and existing.command_id == command.command_id
            for existing in self._history[: self._cursor]
        ):
            raise ValueError(f"duplicate mask command ID: {command.command_id}")
        self._append_history(command)
        return self.snapshot

    def clear(self) -> ManualCleanupMaskDocumentSnapshot:
        if self.snapshot.commands:
            self._append_history(self._CLEAR)
        return self.snapshot

    def undo(self) -> ManualCleanupMaskDocumentSnapshot:
        if self._cursor <= 0:
            return self.snapshot
        self._cursor -= 1
        self._revision += 1
        return self.snapshot

    def redo(self) -> ManualCleanupMaskDocumentSnapshot:
        if self._cursor >= len(self._history):
            return self.snapshot
        self._cursor += 1
        self._revision += 1
        return self.snapshot

    def reset_after_commit(self) -> ManualCleanupMaskDocumentSnapshot:
        """Clear transient mask history after its exact pixels are committed."""

        self._history.clear()
        self._cursor = 0
        self._revision += 1
        return self.snapshot

    def _append_history(self, item: ManualCleanupMaskCommand | object) -> None:
        if self._cursor < len(self._history):
            del self._history[self._cursor :]
        self._history.append(item)
        self._cursor = len(self._history)
        self._revision += 1


@dataclass(frozen=True, slots=True)
class ManualCleanupMaskPayload:
    """Page-sized one-channel PNG bytes exported from one document revision."""

    page_id: str
    canvas_size: tuple[int, int]
    document_revision: int
    erase_mask_png: bytes
    protect_mask_png: bytes | None


def rasterize_mask_document(
    snapshot: ManualCleanupMaskDocumentSnapshot,
) -> ManualCleanupMaskPayload:
    """Rasterize ordered vector commands without leaking image objects to Qt."""

    if not isinstance(snapshot, ManualCleanupMaskDocumentSnapshot):
        raise TypeError("snapshot must be ManualCleanupMaskDocumentSnapshot")
    erase = Image.new("L", snapshot.canvas_size, 0)
    protect = Image.new("L", snapshot.canvas_size, 0)
    protect_used = False
    for command in snapshot.commands:
        target = erase if command.layer is ManualCleanupMaskLayer.ERASE else protect
        if command.layer is ManualCleanupMaskLayer.PROTECT:
            protect_used = True
        draw = ImageDraw.Draw(target)
        value = 255 if command.action is ManualCleanupMaskAction.ADD else 0
        points = [(point.x, point.y) for point in command.points]
        if command.tool is ManualCleanupTool.RECTANGLE:
            draw.rectangle((points[0], points[1]), fill=value)
        elif command.tool is ManualCleanupTool.LASSO:
            draw.polygon(points, fill=value)
        else:
            radius = command.radius_px
            width = max(1, int(round(radius * 2.0)))
            if len(points) > 1:
                draw.line(points, fill=value, width=width, joint="curve")
            for x, y in (points if len(points) == 1 else (points[0], points[-1])):
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=value,
                )
    return ManualCleanupMaskPayload(
        page_id=snapshot.page_id,
        canvas_size=snapshot.canvas_size,
        document_revision=snapshot.revision,
        erase_mask_png=_image_png_bytes(erase),
        protect_mask_png=(
            _image_png_bytes(protect) if protect_used else None
        ),
    )


def _image_png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=6)
    return stream.getvalue()


@dataclass(frozen=True, slots=True)
class ManualCleanupContextCommand:
    """Mask-free command that resolves the exact page/base comparison context."""

    project_path: str
    page_id: str
    coverage_target: UserParentCleanupCoverageTargetV1 | None = None

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        page_id = str(self.page_id or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        if not page_id:
            raise ValueError("page_id is required")
        coverage_target = _canonical_coverage_target(self.coverage_target)
        if coverage_target is not None and coverage_target.page_id != page_id:
            raise ValueError("coverage target belongs to another page")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(self, "page_id", page_id)
        object.__setattr__(self, "coverage_target", coverage_target)


@dataclass(frozen=True, slots=True)
class ManualCleanupWorkerCommand:
    """Exact one-page request carried from the UI thread to one worker."""

    project_path: str
    page_id: str
    erase_mask_png: bytes
    protect_mask_png: bytes | None = None
    parameters: ManualCleanupParameters = field(default_factory=ManualCleanupParameters)
    mode: ManualCleanupWorkerMode = ManualCleanupWorkerMode.PREVIEW
    preview_lease: ManualCleanupPreviewLease | None = None
    rebase_review: ManualCleanupRebaseReview | None = None
    operation_id: str = ""
    transaction_id: str = ""
    coverage_target: UserParentCleanupCoverageTargetV1 | None = None

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        page_id = str(self.page_id or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        if not page_id:
            raise ValueError("page_id is required")
        erase = self._png(self.erase_mask_png, "erase_mask_png")
        protect = (
            None
            if self.protect_mask_png is None
            else self._png(self.protect_mask_png, "protect_mask_png")
        )
        if not isinstance(self.parameters, ManualCleanupParameters):
            raise TypeError("parameters must be ManualCleanupParameters")
        if not isinstance(self.mode, ManualCleanupWorkerMode):
            raise TypeError("mode must be ManualCleanupWorkerMode")
        lease = self.preview_lease
        rebase_review = self.rebase_review
        coverage_target = _canonical_coverage_target(self.coverage_target)
        if coverage_target is not None and coverage_target.page_id != page_id:
            raise ValueError("coverage target belongs to another page")
        operation_id = str(self.operation_id or "").strip()
        transaction_id = str(self.transaction_id or "").strip()
        if self.mode is ManualCleanupWorkerMode.PREVIEW:
            if lease is not None:
                raise ValueError("preview commands cannot carry a preview lease")
            if coverage_target is not None and rebase_review is not None:
                raise ValueError(
                    "user-parent coverage requires a newly authored mask, not rebase review"
                )
            if rebase_review is not None:
                if not isinstance(rebase_review, ManualCleanupRebaseReview):
                    raise TypeError("rebase_review must be ManualCleanupRebaseReview")
                if rebase_review.page_id != page_id:
                    raise ValueError("rebase review belongs to another page")
                if rebase_review.parameters != self.parameters:
                    raise ValueError("rebase parameters differ from the saved review")
                if (
                    erase != rebase_review.erase_mask_png
                    or protect != rebase_review.protect_mask_png
                ):
                    raise ValueError("rebase preview must use the exact saved masks")
        else:
            if not isinstance(lease, ManualCleanupPreviewLease):
                raise ValueError("commit commands require a preview lease")
            if lease.page_id != page_id:
                raise ValueError("preview lease belongs to another page")
            if lease.parameters != self.parameters:
                raise ValueError("commit parameters differ from the preview")
            if lease.coverage_target != coverage_target:
                raise ValueError("commit coverage target differs from the preview")
            if operation_id and operation_id != lease.operation_id:
                raise ValueError("commit operation_id differs from the preview")
            operation_id = lease.operation_id
            if rebase_review is not None:
                raise ValueError("commit commands use the preview lease, not rebase review")
        for value, name in (
            (operation_id, "operation_id"),
            (transaction_id, "transaction_id"),
        ):
            if value and not all(character.isalnum() or character in "-_" for character in value):
                raise ValueError(f"{name} must be path-safe")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(self, "page_id", page_id)
        object.__setattr__(self, "erase_mask_png", erase)
        object.__setattr__(self, "protect_mask_png", protect)
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "transaction_id", transaction_id)
        object.__setattr__(self, "coverage_target", coverage_target)

    @staticmethod
    def _png(payload: bytes, field_name: str) -> bytes:
        if not isinstance(payload, bytes):
            raise TypeError(f"{field_name} must be bytes")
        value = bytes(payload)
        if not value.startswith(_PNG_SIGNATURE):
            raise ValueError(f"{field_name} must be PNG bytes")
        return value

    @property
    def payload_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.erase_mask_png)
        digest.update(self.protect_mask_png or b"")
        digest.update(repr(self.parameters.to_dict()).encode("utf-8"))
        if self.rebase_review is not None:
            digest.update(self.rebase_review.binding_sha256.encode("ascii"))
        if self.coverage_target is not None:
            digest.update(
                self.coverage_target.coverage_dependency_fingerprint.encode("ascii")
            )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ManualCleanupWorkerFailure:
    code: ManualCleanupWorkerFailureCode
    stage: ManualCleanupWorkerStage
    project_path: str
    page_id: str
    message: str
    exception_type: str = ""
    service_code: ManualCleanupFailureCode | None = None
    service_stage: ManualCleanupStage | None = None
    preflight: ManualCleanupPreflight | None = None


@dataclass(frozen=True, slots=True)
class ManualCleanupCancellationState:
    page_id: str
    enabled: bool
    message: str = ""


@dataclass(frozen=True, slots=True)
class ManualCleanupViewState:
    phase: ManualCleanupViewPhase
    selected_tool: ManualCleanupTool
    eraser_layer: ManualCleanupMaskLayer
    brush_radius_px: float
    parameters: ManualCleanupParameters
    document: ManualCleanupMaskDocumentSnapshot
    command: ManualCleanupWorkerCommand | None
    message: str
    preflight: ManualCleanupPreflight | None
    progress: ManualCleanupProgress | None
    receipt: ManualCleanupReceipt | None
    preview_lease: ManualCleanupPreviewLease | None
    rebase_review: ManualCleanupRebaseReview | None
    coverage_target: UserParentCleanupCoverageTargetV1 | None
    failure: ManualCleanupWorkerFailure | None
    cancellation_locked: bool

    @property
    def busy(self) -> bool:
        return self.phase in {
            ManualCleanupViewPhase.PREVIEWING,
            ManualCleanupViewPhase.COMMITTING,
        }

    @property
    def editing_enabled(self) -> bool:
        return not self.busy and self.rebase_review is None

    @property
    def undo_enabled(self) -> bool:
        return self.editing_enabled and self.document.undo_depth > 0

    @property
    def redo_enabled(self) -> bool:
        return self.editing_enabled and self.document.redo_depth > 0

    @property
    def reset_enabled(self) -> bool:
        return self.editing_enabled and bool(self.document.commands)

    @property
    def preview_enabled(self) -> bool:
        return (
            self.editing_enabled
            and self.document.has_erase_marks
            and self.phase
            in {
                ManualCleanupViewPhase.DIRTY,
                ManualCleanupViewPhase.STALE,
                ManualCleanupViewPhase.CANCELLED,
                ManualCleanupViewPhase.FAILED,
            }
        ) or (
            not self.busy
            and self.rebase_review is not None
            and self.phase is ManualCleanupViewPhase.STALE
        )

    @property
    def commit_enabled(self) -> bool:
        return (
            self.phase is ManualCleanupViewPhase.PREVIEW_READY
            and self.preview_lease is not None
        )

    @property
    def cancel_enabled(self) -> bool:
        return self.busy and not self.cancellation_locked


class ManualCleanupEditorModel:
    """UI-thread state reducer; it owns no Qt, images, files, or SQLite."""

    def __init__(
        self,
        page_id: str,
        canvas_size: tuple[int, int],
        *,
        parameters: ManualCleanupParameters | None = None,
        coverage_target: UserParentCleanupCoverageTargetV1 | None = None,
    ) -> None:
        self.document = ManualCleanupMaskDocument(page_id, canvas_size)
        self._coverage_target = _canonical_coverage_target(coverage_target)
        if (
            self._coverage_target is not None
            and self._coverage_target.canvas_size
            != self.document.snapshot.canvas_size
        ):
            raise ValueError("coverage target canvas differs from the editor")
        self._phase = ManualCleanupViewPhase.IDLE
        self._selected_tool = ManualCleanupTool.BRUSH
        self._eraser_layer = ManualCleanupMaskLayer.ERASE
        self._brush_radius_px = 12.0
        self._parameters = parameters or ManualCleanupParameters()
        self._command: ManualCleanupWorkerCommand | None = None
        self._message = self._empty_mask_message()
        self._preflight: ManualCleanupPreflight | None = None
        self._progress: ManualCleanupProgress | None = None
        self._receipt: ManualCleanupReceipt | None = None
        self._preview_lease: ManualCleanupPreviewLease | None = None
        self._rebase_review: ManualCleanupRebaseReview | None = None
        self._failure: ManualCleanupWorkerFailure | None = None
        self._cancellation_locked = False

    @classmethod
    def from_context(
        cls,
        context: ManualCleanupContext,
        *,
        parameters: ManualCleanupParameters | None = None,
        coverage_target: UserParentCleanupCoverageTargetV1 | None = None,
    ) -> "ManualCleanupEditorModel":
        if not isinstance(context, ManualCleanupContext):
            raise TypeError("context must be ManualCleanupContext")
        if not context.ready or len(context.canvas_size) != 2:
            raise ValueError("manual cleanup context is not ready")
        canonical_target = _canonical_coverage_target(coverage_target)
        if canonical_target is not None:
            if context.rebase_review is not None:
                raise ValueError("coverage context cannot reuse a stale cleanup mask")
            if (
                canonical_target.page_id != context.page_id
                or canonical_target.canvas_size != context.canvas_size
                or canonical_target.input_cleaned_base_revision_id
                != context.input_base_revision_id
                or canonical_target.input_cleaned_base_content_sha256
                != context.input_base_sha256
            ):
                raise ValueError(
                    "coverage target differs from the resolved cleanup context"
                )
        model = cls(
            context.page_id,
            context.canvas_size,
            parameters=(
                context.rebase_review.parameters
                if context.rebase_review is not None
                else parameters
            ),
            coverage_target=canonical_target,
        )
        if context.rebase_review is not None:
            model._rebase_review = context.rebase_review
            model._phase = ManualCleanupViewPhase.STALE
            model._message = (
                "Review the saved mask on Current Cleaned. Preview reruns "
                "inpainting on this base; the old result is never reused."
            )
        return model

    @property
    def state(self) -> ManualCleanupViewState:
        return ManualCleanupViewState(
            phase=self._phase,
            selected_tool=self._selected_tool,
            eraser_layer=self._eraser_layer,
            brush_radius_px=self._brush_radius_px,
            parameters=self._parameters,
            document=self.document.snapshot,
            command=self._command,
            message=self._message,
            preflight=self._preflight,
            progress=self._progress,
            receipt=self._receipt,
            preview_lease=self._preview_lease,
            rebase_review=self._rebase_review,
            coverage_target=self._coverage_target,
            failure=self._failure,
            cancellation_locked=self._cancellation_locked,
        )

    def select_tool(
        self,
        tool: ManualCleanupTool,
        *,
        eraser_layer: ManualCleanupMaskLayer | None = None,
    ) -> ManualCleanupViewState:
        self._require_editable()
        self._selected_tool = ManualCleanupTool(tool)
        if eraser_layer is not None:
            self._eraser_layer = ManualCleanupMaskLayer(eraser_layer)
        return self.state

    def set_brush_radius(self, radius_px: float) -> ManualCleanupViewState:
        self._require_editable()
        radius = float(radius_px)
        if not math.isfinite(radius) or not 0.5 <= radius <= 256.0:
            raise ValueError("brush radius must be between 0.5 and 256 pixels")
        self._brush_radius_px = radius
        return self.state

    def set_parameters(
        self,
        *,
        grow_px: int | None = None,
        feather_px: int | None = None,
    ) -> ManualCleanupViewState:
        self._require_editable()
        candidate = ManualCleanupParameters(
            grow_px=self._parameters.grow_px if grow_px is None else grow_px,
            feather_px=(
                self._parameters.feather_px if feather_px is None else feather_px
            ),
            backend_id=self._parameters.backend_id,
            use_gpu=self._parameters.use_gpu,
        )
        if candidate != self._parameters:
            self._parameters = candidate
            self._mark_document_changed("Cleanup parameters changed.")
        return self.state

    def add_command(
        self,
        command: ManualCleanupMaskCommand,
    ) -> ManualCleanupViewState:
        self._require_editable()
        self.document.append(command)
        self._mark_document_changed("Mask updated.")
        return self.state

    def add_points(
        self,
        points: Sequence[PagePoint | tuple[float, float]],
    ) -> ManualCleanupViewState:
        radius = (
            0.0
            if self._selected_tool
            in {ManualCleanupTool.RECTANGLE, ManualCleanupTool.LASSO}
            else self._brush_radius_px
        )
        return self.add_command(
            ManualCleanupMaskCommand.create(
                self._selected_tool,
                points,
                eraser_layer=self._eraser_layer,
                radius_px=radius,
            )
        )

    def export_mask_payload(self) -> ManualCleanupMaskPayload:
        """Export immutable PNG bytes for a worker command."""

        if self.state.busy:
            raise RuntimeError("mask export is disabled during cleanup work")
        if self._rebase_review is not None:
            return ManualCleanupMaskPayload(
                page_id=self._rebase_review.page_id,
                canvas_size=self._rebase_review.canvas_size,
                document_revision=self.document.snapshot.revision,
                erase_mask_png=self._rebase_review.erase_mask_png,
                protect_mask_png=self._rebase_review.protect_mask_png,
            )
        return rasterize_mask_document(self.document.snapshot)

    def undo(self) -> ManualCleanupViewState:
        self._require_editable()
        before = self.document.snapshot.revision
        self.document.undo()
        if self.document.snapshot.revision != before:
            self._mark_document_changed("Mask edit undone.")
        return self.state

    def redo(self) -> ManualCleanupViewState:
        self._require_editable()
        before = self.document.snapshot.revision
        self.document.redo()
        if self.document.snapshot.revision != before:
            self._mark_document_changed("Mask edit restored.")
        return self.state

    def clear(self) -> ManualCleanupViewState:
        self._require_editable()
        before = self.document.snapshot.revision
        self.document.clear()
        if self.document.snapshot.revision != before:
            self._mark_document_changed("Mask cleared.")
        return self.state

    def begin(self, command: ManualCleanupWorkerCommand) -> ManualCleanupViewState:
        if not isinstance(command, ManualCleanupWorkerCommand):
            raise TypeError("command must be ManualCleanupWorkerCommand")
        if self.state.busy:
            raise RuntimeError("manual cleanup work is already active")
        if command.page_id != self.document.snapshot.page_id:
            raise ValueError("worker command belongs to another page")
        if command.parameters != self._parameters:
            raise ValueError("worker command parameters differ from the editor")
        if command.coverage_target != self._coverage_target:
            raise ValueError("worker coverage target differs from the editor")
        if (
            command.mode is ManualCleanupWorkerMode.PREVIEW
            and command.rebase_review != self._rebase_review
        ):
            raise ValueError("worker command rebase review differs from the editor")
        if command.mode is ManualCleanupWorkerMode.PREVIEW:
            if not self.state.preview_enabled:
                raise RuntimeError("manual cleanup preview is not currently available")
            phase = ManualCleanupViewPhase.PREVIEWING
            message = "Preparing manual cleanup preview..."
        else:
            if not self.state.commit_enabled:
                raise RuntimeError("only the current completed preview can be committed")
            if command.preview_lease != self._preview_lease:
                raise ValueError("commit command does not own the current preview")
            phase = ManualCleanupViewPhase.COMMITTING
            message = "Validating preview for commit..."
        self._phase = phase
        self._command = command
        self._message = message
        self._progress = None
        self._failure = None
        self._cancellation_locked = False
        return self.state

    def accept_preflight(self, value: ManualCleanupPreflight) -> ManualCleanupViewState:
        self._require_worker_page(value.page_id)
        self._preflight = value
        self._message = value.message
        if not value.ready:
            self._phase = (
                ManualCleanupViewPhase.STALE
                if value.code
                in {
                    ManualCleanupAvailabilityCode.MISSING_BASE,
                    ManualCleanupAvailabilityCode.STALE_BASE,
                }
                else ManualCleanupViewPhase.FAILED
            )
            self._cancellation_locked = False
        return self.state

    def accept_progress(self, value: ManualCleanupProgress) -> ManualCleanupViewState:
        self._require_worker_page(value.page_id)
        self._progress = value
        self._message = value.message
        if value.stage is ManualCleanupStage.PERSISTING:
            self._cancellation_locked = True
        return self.state

    def accept_cancellation_state(
        self,
        value: ManualCleanupCancellationState,
    ) -> ManualCleanupViewState:
        self._require_worker_page(value.page_id)
        self._cancellation_locked = not value.enabled
        if value.message:
            self._message = value.message
        return self.state

    def accept_preview(self, value: ManualCleanupReceipt) -> ManualCleanupViewState:
        self._require_worker_page(value.page_id)
        if value.status is not ManualCleanupStatus.PREVIEW_READY:
            raise ValueError("preview signal requires a PREVIEW_READY receipt")
        if value.preview_lease is None:
            raise ValueError("preview receipt has no lease")
        if (
            value.coverage_target != self._coverage_target
            or value.preview_lease.coverage_target != self._coverage_target
        ):
            raise ValueError("preview receipt coverage target differs from the editor")
        self._phase = ManualCleanupViewPhase.PREVIEW_READY
        self._receipt = value
        self._preview_lease = value.preview_lease
        self._message = (
            "Preview ready. Review it, then explicitly confirm this clean base."
            if self._coverage_target is not None
            else "Manual cleanup preview ready."
        )
        self._cancellation_locked = False
        return self.state

    def accept_commit(self, value: ManualCleanupReceipt) -> ManualCleanupViewState:
        self._require_worker_page(value.page_id)
        if value.status is not ManualCleanupStatus.COMMITTED:
            raise ValueError("commit signal requires a COMMITTED receipt")
        if value.coverage_target != self._coverage_target:
            raise ValueError("commit receipt coverage target differs from the editor")
        self.document.reset_after_commit()
        self._phase = ManualCleanupViewPhase.COMMITTED
        self._receipt = value
        self._preview_lease = None
        self._rebase_review = None
        self._message = (
            "Selected parent cleanup is current. Later owner stages remain explicit."
            if self._coverage_target is not None
            else "Manual cleanup revision committed."
        )
        self._cancellation_locked = False
        return self.state

    def accept_cancelled(self, value: ManualCleanupReceipt) -> ManualCleanupViewState:
        self._require_worker_page(value.page_id)
        if value.status is not ManualCleanupStatus.CANCELLED:
            raise ValueError("cancel signal requires a CANCELLED receipt")
        if value.coverage_target != self._coverage_target:
            raise ValueError("cancel receipt coverage target differs from the editor")
        self._phase = ManualCleanupViewPhase.CANCELLED
        self._receipt = value
        self._preview_lease = None
        self._message = "Manual cleanup cancelled; no current revision changed."
        self._cancellation_locked = False
        return self.state

    def accept_failure(
        self,
        value: ManualCleanupWorkerFailure,
    ) -> ManualCleanupViewState:
        self._require_worker_page(value.page_id)
        self._phase = (
            ManualCleanupViewPhase.STALE
            if value.code
            in {
                ManualCleanupWorkerFailureCode.MISSING_BASE,
                ManualCleanupWorkerFailureCode.STALE_BASE,
                ManualCleanupWorkerFailureCode.PREVIEW_STALE,
                ManualCleanupWorkerFailureCode.COMMIT_STALE,
            }
            else ManualCleanupViewPhase.FAILED
        )
        self._failure = value
        self._message = value.message
        self._cancellation_locked = False
        return self.state

    def mark_stale(self, message: str) -> ManualCleanupViewState:
        if self.state.busy:
            raise RuntimeError("cannot mark an active cleanup operation stale")
        self._phase = ManualCleanupViewPhase.STALE
        self._preview_lease = None
        self._message = str(message or "The selected clean base changed.")
        return self.state

    def _mark_document_changed(self, message: str) -> None:
        self._phase = (
            ManualCleanupViewPhase.DIRTY
            if self.document.snapshot.commands
            else ManualCleanupViewPhase.IDLE
        )
        self._command = None
        self._preflight = None
        self._progress = None
        self._receipt = None
        self._preview_lease = None
        self._failure = None
        self._cancellation_locked = False
        self._message = (
            message
            if self.document.snapshot.commands
            else self._empty_mask_message()
        )

    def _empty_mask_message(self) -> str:
        if self._coverage_target is not None:
            return (
                "Draw a non-empty erase mask. The dashed workflow area is a guide only."
            )
        return "Draw an erase mask to begin."

    def _require_editable(self) -> None:
        if self.state.busy:
            raise RuntimeError("mask editing is disabled during cleanup work")

    def _require_worker_page(self, page_id: str) -> None:
        if self._command is None:
            raise RuntimeError("no manual cleanup worker command is active")
        if str(page_id or "") != self._command.page_id:
            raise ValueError("worker event belongs to another page")


def worker_failure_from_preflight(
    command: ManualCleanupWorkerCommand,
    preflight: ManualCleanupPreflight,
) -> ManualCleanupWorkerFailure:
    code = {
        ManualCleanupAvailabilityCode.MISSING_BASE: ManualCleanupWorkerFailureCode.MISSING_BASE,
        ManualCleanupAvailabilityCode.STALE_BASE: ManualCleanupWorkerFailureCode.STALE_BASE,
        ManualCleanupAvailabilityCode.INVALID_MASK: ManualCleanupWorkerFailureCode.INVALID_MASK,
        ManualCleanupAvailabilityCode.BACKEND_UNAVAILABLE: ManualCleanupWorkerFailureCode.BACKEND_UNAVAILABLE,
        ManualCleanupAvailabilityCode.INVALID_COVERAGE_TARGET: ManualCleanupWorkerFailureCode.COVERAGE_TARGET_INVALID,
        ManualCleanupAvailabilityCode.STALE_COVERAGE_TARGET: ManualCleanupWorkerFailureCode.COVERAGE_TARGET_STALE,
        ManualCleanupAvailabilityCode.COVERAGE_CONFLICT: ManualCleanupWorkerFailureCode.COVERAGE_CONFLICT,
        ManualCleanupAvailabilityCode.ORIGINAL_ASSET_MISMATCH: ManualCleanupWorkerFailureCode.ORIGINAL_ASSET_MISMATCH,
    }.get(preflight.code, ManualCleanupWorkerFailureCode.INVALID_REQUEST)
    return ManualCleanupWorkerFailure(
        code=code,
        stage=ManualCleanupWorkerStage.PREFLIGHT,
        project_path=command.project_path,
        page_id=command.page_id,
        message=preflight.message,
        service_code=(
            ManualCleanupFailureCode(code.value)
            if code.value in {value.value for value in ManualCleanupFailureCode}
            else None
        ),
        service_stage=ManualCleanupStage.VALIDATING,
        preflight=preflight,
    )


def genesis_edit_heads() -> tuple[str, str]:
    """Return the page/global heads of a project with no GUI sidecar."""

    return _SHA256_ZERO, _SHA256_ZERO
