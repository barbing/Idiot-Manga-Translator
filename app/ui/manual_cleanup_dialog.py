# -*- coding: utf-8 -*-
"""Reusable, page-bounded manual-cleanup editor for the Qt GUI.

The canvas owns only user-authored vector mask commands.  Cleanup execution,
artifact publication, and project mutation remain behind the typed worker
boundary in :mod:`app.ui.manual_cleanup_worker`.
"""
from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets
from app.project_edits.manual_cleanup import (
    ManualCleanupContext,
    ManualCleanupParameters,
    ManualCleanupReceipt,
    UserParentCleanupCoverageTargetV1,
)
from app.ui.design_system.components import WheelSafeSpinBox
from app.ui.design_system.dialogs import HybridDialog
from app.ui.manual_cleanup_worker import (
    ManualCleanupContextWorker,
    ManualCleanupWorker,
    discard_manual_cleanup_preview,
)
from app.ui.viewmodels.manual_cleanup_model import (
    ManualCleanupCancellationState,
    ManualCleanupContextCommand,
    ManualCleanupEditorModel,
    ManualCleanupMaskAction,
    ManualCleanupMaskCommand,
    ManualCleanupMaskLayer,
    ManualCleanupTool,
    ManualCleanupViewPhase,
    ManualCleanupViewState,
    ManualCleanupWorkerCommand,
    ManualCleanupWorkerFailure,
    ManualCleanupWorkerMode,
)


class ManualCleanupCanvas(QtWidgets.QGraphicsView):
    """Image-first page canvas with independent erase/protect overlays."""

    operation_added = QtCore.Signal(object)
    cursor_moved = QtCore.Signal(int, int)

    _ERASE_COLOR = QtGui.QColor(255, 72, 92, 104)
    _PROTECT_COLOR = QtGui.QColor(39, 214, 222, 126)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._background = self._scene.addPixmap(QtGui.QPixmap())
        self._background.setZValue(0)
        self._workflow_guide = QtWidgets.QGraphicsRectItem()
        guide_pen = QtGui.QPen(QtGui.QColor(255, 193, 72, 230), 2.0)
        guide_pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        guide_pen.setCosmetic(True)
        self._workflow_guide.setPen(guide_pen)
        self._workflow_guide.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        self._workflow_guide.setZValue(6)
        self._workflow_guide.setVisible(False)
        self._scene.addItem(self._workflow_guide)
        self._rebase_erase_overlay = self._scene.addPixmap(QtGui.QPixmap())
        self._rebase_erase_overlay.setZValue(8)
        self._rebase_protect_overlay = self._scene.addPixmap(QtGui.QPixmap())
        self._rebase_protect_overlay.setZValue(9)
        self._erase_overlay = QtWidgets.QGraphicsPathItem()
        self._erase_overlay.setBrush(self._ERASE_COLOR)
        self._erase_overlay.setPen(QtCore.Qt.NoPen)
        self._erase_overlay.setZValue(10)
        self._scene.addItem(self._erase_overlay)
        self._protect_overlay = QtWidgets.QGraphicsPathItem()
        self._protect_overlay.setBrush(self._PROTECT_COLOR)
        self._protect_overlay.setPen(QtCore.Qt.NoPen)
        self._protect_overlay.setZValue(11)
        self._scene.addItem(self._protect_overlay)
        self._gesture_item = QtWidgets.QGraphicsPathItem()
        self._gesture_item.setZValue(20)
        self._scene.addItem(self._gesture_item)
        self._canvas_size = QtCore.QSize()
        self._tool = ManualCleanupTool.RECTANGLE
        self._eraser_layer = ManualCleanupMaskLayer.ERASE
        self._brush_radius = 14.0
        self._commands: tuple[ManualCleanupMaskCommand, ...] = ()
        self._rebase_binding = ""
        self._editing_enabled = False
        self._rebase_overlay_requested = False
        self._rebase_overlay_visible = True
        self._gesture_points: list[QtCore.QPointF] = []
        self._gesture_active = False
        self._fit_pending = False
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setBackgroundBrush(QtGui.QColor(17, 25, 36))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setRenderHints(
            QtGui.QPainter.Antialiasing
            | QtGui.QPainter.SmoothPixmapTransform
        )
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setMouseTracking(True)

    @property
    def canvas_size(self) -> tuple[int, int]:
        return (self._canvas_size.width(), self._canvas_size.height())

    def set_canvas_size(self, width: int, height: int) -> None:
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError("canvas dimensions must be positive")
        self._canvas_size = QtCore.QSize(width, height)
        self._scene.setSceneRect(0.0, 0.0, float(width), float(height))
        self._fit_pending = True
        self._redraw_overlay()

    def set_workflow_area_guide(
        self,
        value: tuple[int, int, int, int] | None,
    ) -> None:
        """Show non-authoritative lineage geometry without touching masks."""

        if value is None:
            self._workflow_guide.setRect(QtCore.QRectF())
            self._workflow_guide.setVisible(False)
            return
        if (
            not isinstance(value, tuple)
            or len(value) != 4
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise ValueError("workflow area guide must contain four exact integers")
        x, y, width, height = value
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("workflow area guide must have positive page bounds")
        if (
            not self._canvas_size.isEmpty()
            and (
                x + width > self._canvas_size.width()
                or y + height > self._canvas_size.height()
            )
        ):
            raise ValueError("workflow area guide must remain inside the page")
        self._workflow_guide.setRect(
            QtCore.QRectF(float(x), float(y), float(width), float(height))
        )
        self._workflow_guide.setVisible(True)

    def set_image_path(self, path: str) -> bool:
        pixmap = QtGui.QPixmap(str(path or ""))
        if pixmap.isNull():
            self._background.setPixmap(QtGui.QPixmap())
            return False
        if self._canvas_size.isEmpty():
            self.set_canvas_size(pixmap.width(), pixmap.height())
        if pixmap.size() != self._canvas_size:
            pixmap = pixmap.scaled(
                self._canvas_size,
                QtCore.Qt.IgnoreAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        self._background.setPixmap(pixmap)
        if self._fit_pending:
            self.fit_page()
        return True

    def apply_state(self, state: ManualCleanupViewState) -> None:
        if not isinstance(state, ManualCleanupViewState):
            raise TypeError("state must be a ManualCleanupViewState")
        self._tool = state.selected_tool
        self._eraser_layer = state.eraser_layer
        self._brush_radius = state.brush_radius_px
        commands = state.document.commands
        if commands != self._commands:
            self._commands = commands
            self._redraw_overlay()
        binding = (
            state.rebase_review.binding_sha256
            if state.rebase_review is not None
            else ""
        )
        if binding != self._rebase_binding:
            self._rebase_binding = binding
            self._rebase_overlay_requested = bool(binding)
            if state.rebase_review is None:
                self._rebase_erase_overlay.setPixmap(QtGui.QPixmap())
                self._rebase_protect_overlay.setPixmap(QtGui.QPixmap())
            else:
                self._rebase_erase_overlay.setPixmap(
                    self._mask_overlay_pixmap(
                        state.rebase_review.erase_mask_png,
                        self._ERASE_COLOR,
                    )
                )
                self._rebase_protect_overlay.setPixmap(
                    self._mask_overlay_pixmap(
                        state.rebase_review.protect_mask_png,
                        self._PROTECT_COLOR,
                    )
                )
        self._update_rebase_overlay_visibility()
        self.set_editing_enabled(state.editing_enabled)
        self._update_cursor()

    def set_editing_enabled(self, enabled: bool) -> None:
        """Lock mask gestures without disabling page inspection and zoom."""

        self._editing_enabled = bool(enabled)
        if not self._editing_enabled and self._gesture_active:
            self._gesture_active = False
            self._gesture_points = []
            self._gesture_item.setPath(QtGui.QPainterPath())
        self._update_cursor()

    def set_rebase_overlay_visible(self, visible: bool) -> None:
        self._rebase_overlay_visible = bool(visible)
        self._update_rebase_overlay_visibility()

    def _update_rebase_overlay_visibility(self) -> None:
        visible = self._rebase_overlay_requested and self._rebase_overlay_visible
        self._rebase_erase_overlay.setVisible(visible)
        self._rebase_protect_overlay.setVisible(visible)

    @staticmethod
    def _mask_overlay_pixmap(payload: bytes, color: QtGui.QColor) -> QtGui.QPixmap:
        image = QtGui.QImage.fromData(payload, "PNG")
        if image.isNull():
            return QtGui.QPixmap()
        indexed = image.convertToFormat(QtGui.QImage.Format_Indexed8)
        alpha = color.alpha()
        indexed.setColorTable(
            [
                QtGui.qRgba(
                    color.red(),
                    color.green(),
                    color.blue(),
                    int(round(alpha * value / 255.0)),
                )
                for value in range(256)
            ]
        )
        return QtGui.QPixmap.fromImage(indexed)

    def set_tool(
        self,
        tool: ManualCleanupTool,
        *,
        eraser_layer: ManualCleanupMaskLayer = ManualCleanupMaskLayer.ERASE,
    ) -> None:
        self._tool = ManualCleanupTool(tool)
        self._eraser_layer = ManualCleanupMaskLayer(eraser_layer)
        self._update_cursor()

    def _update_cursor(self) -> None:
        if not self._editing_enabled:
            self.setCursor(QtCore.Qt.ArrowCursor)
            return
        self.setCursor(
            QtCore.Qt.CrossCursor
            if self._tool
            in {ManualCleanupTool.RECTANGLE, ManualCleanupTool.LASSO}
            else QtCore.Qt.BlankCursor
        )

    def set_brush_radius(self, radius_px: float) -> None:
        radius = float(radius_px)
        if not 0.5 <= radius <= 256.0:
            raise ValueError("brush radius must be between 0.5 and 256 pixels")
        self._brush_radius = radius

    def fit_page(self) -> None:
        if self._canvas_size.isEmpty():
            return
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self._fit_pending = False

    def zoom_by(self, factor: float) -> None:
        if factor <= 0:
            return
        current = self.transform().m11()
        target = current * factor
        if 0.05 <= target <= 16.0:
            self.scale(factor, factor)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if event.modifiers() & QtCore.Qt.ControlModifier:
            self.zoom_by(1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15)
            event.accept()
            return
        super().wheelEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._fit_pending:
            self.fit_page()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            not self._editing_enabled
            or event.button() != QtCore.Qt.LeftButton
            or self._canvas_size.isEmpty()
        ):
            super().mousePressEvent(event)
            return
        point = self._bounded_scene_point(event.position().toPoint())
        if point is None:
            super().mousePressEvent(event)
            return
        self._gesture_points = [point]
        self._gesture_active = True
        self._update_gesture_item()
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        point = self._bounded_scene_point(event.position().toPoint())
        if point is not None:
            self.cursor_moved.emit(int(point.x()), int(point.y()))
        if not self._gesture_active or point is None:
            super().mouseMoveEvent(event)
            return
        if self._tool is ManualCleanupTool.RECTANGLE:
            if len(self._gesture_points) == 1:
                self._gesture_points.append(point)
            else:
                self._gesture_points[-1] = point
        else:
            previous = self._gesture_points[-1]
            if QtCore.QLineF(previous, point).length() >= 1.0:
                self._gesture_points.append(point)
        self._update_gesture_item()
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton or not self._gesture_active:
            super().mouseReleaseEvent(event)
            return
        point = self._bounded_scene_point(event.position().toPoint())
        if point is not None:
            if self._tool is ManualCleanupTool.RECTANGLE:
                if len(self._gesture_points) == 1:
                    self._gesture_points.append(point)
                else:
                    self._gesture_points[-1] = point
            elif point != self._gesture_points[-1]:
                self._gesture_points.append(point)
        points = tuple((item.x(), item.y()) for item in self._gesture_points)
        self._gesture_active = False
        self._gesture_points = []
        self._gesture_item.setPath(QtGui.QPainterPath())
        minimum = 3 if self._tool is ManualCleanupTool.LASSO else 1
        if len(points) >= minimum:
            try:
                command = ManualCleanupMaskCommand.create(
                    self._tool,
                    points,
                    eraser_layer=self._eraser_layer,
                    radius_px=(
                        0.0
                        if self._tool
                        in {ManualCleanupTool.RECTANGLE, ManualCleanupTool.LASSO}
                        else self._brush_radius
                    ),
                )
            except (TypeError, ValueError):
                command = None
            if command is not None:
                self.operation_added.emit(command)
        event.accept()

    def _bounded_scene_point(self, viewport_point: QtCore.QPoint) -> QtCore.QPointF | None:
        point = self.mapToScene(viewport_point)
        rect = self._scene.sceneRect()
        if not rect.contains(point):
            return None
        return QtCore.QPointF(
            min(max(point.x(), 0.0), float(self._canvas_size.width() - 1)),
            min(max(point.y(), 0.0), float(self._canvas_size.height() - 1)),
        )

    def _update_gesture_item(self) -> None:
        if not self._gesture_points:
            self._gesture_item.setPath(QtGui.QPainterPath())
            return
        points = tuple(self._gesture_points)
        path = QtGui.QPainterPath()
        if self._tool is ManualCleanupTool.RECTANGLE:
            path.addRect(QtCore.QRectF(points[0], points[-1]).normalized())
        elif self._tool is ManualCleanupTool.LASSO:
            path.moveTo(points[0])
            for point in points[1:]:
                path.lineTo(point)
            if len(points) >= 3:
                path.closeSubpath()
        elif len(points) == 1:
            path.addEllipse(points[0], self._brush_radius, self._brush_radius)
        else:
            path.moveTo(points[0])
            for point in points[1:]:
                path.lineTo(point)
        layer = (
            ManualCleanupMaskLayer.PROTECT
            if self._tool is ManualCleanupTool.PROTECT
            else self._eraser_layer
            if self._tool is ManualCleanupTool.ERASER
            else ManualCleanupMaskLayer.ERASE
        )
        color = (
            self._PROTECT_COLOR
            if layer is ManualCleanupMaskLayer.PROTECT
            else self._ERASE_COLOR
        )
        self._gesture_item.setPath(path)
        if self._tool in {ManualCleanupTool.RECTANGLE, ManualCleanupTool.LASSO}:
            self._gesture_item.setBrush(color)
            self._gesture_item.setPen(QtGui.QPen(color.lighter(150), 1.5))
        else:
            if len(points) == 1:
                self._gesture_item.setBrush(color)
                self._gesture_item.setPen(QtCore.Qt.NoPen)
            else:
                pen = QtGui.QPen(color, float(self._brush_radius * 2.0))
                pen.setCapStyle(QtCore.Qt.RoundCap)
                pen.setJoinStyle(QtCore.Qt.RoundJoin)
                self._gesture_item.setBrush(QtCore.Qt.NoBrush)
                self._gesture_item.setPen(pen)

    def _redraw_overlay(self) -> None:
        if self._canvas_size.isEmpty():
            self._erase_overlay.setPath(QtGui.QPainterPath())
            self._protect_overlay.setPath(QtGui.QPainterPath())
            return
        self._erase_overlay.setPath(
            self._resolved_layer_path(ManualCleanupMaskLayer.ERASE)
        )
        self._protect_overlay.setPath(
            self._resolved_layer_path(ManualCleanupMaskLayer.PROTECT)
        )

    @staticmethod
    def _path_for_command(command: ManualCleanupMaskCommand) -> QtGui.QPainterPath:
        points = [QtCore.QPointF(point.x, point.y) for point in command.points]
        path = QtGui.QPainterPath()
        if command.tool is ManualCleanupTool.RECTANGLE:
            first = points[0]
            last = points[-1]
            path.addRect(QtCore.QRectF(first, last).normalized())
            return path
        if len(points) == 1:
            radius = command.radius_px
            path.addEllipse(points[0], radius, radius)
            return path
        path.moveTo(points[0])
        for point in points[1:]:
            path.lineTo(point)
        if command.tool is ManualCleanupTool.LASSO:
            path.closeSubpath()
        return path

    @classmethod
    def _command_area_path(
        cls,
        command: ManualCleanupMaskCommand,
    ) -> QtGui.QPainterPath:
        path = cls._path_for_command(command)
        if command.tool in {ManualCleanupTool.RECTANGLE, ManualCleanupTool.LASSO}:
            return path
        if len(command.points) == 1:
            return path
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(float(command.radius_px * 2.0))
        stroker.setCapStyle(QtCore.Qt.RoundCap)
        stroker.setJoinStyle(QtCore.Qt.RoundJoin)
        return stroker.createStroke(path)

    def _resolved_layer_path(
        self,
        layer: ManualCleanupMaskLayer,
    ) -> QtGui.QPainterPath:
        resolved = QtGui.QPainterPath()
        for command in self._commands:
            if command.layer is not layer:
                continue
            area = self._command_area_path(command)
            resolved = (
                resolved.subtracted(area)
                if command.action is ManualCleanupMaskAction.SUBTRACT
                else resolved.united(area)
            )
        return resolved


class ManualCleanupDialog(HybridDialog):
    """Modal page-local cleanup editor backed by one-shot typed workers."""

    cleanup_committed = QtCore.Signal(object)

    _TOOL_LABELS = (
        (ManualCleanupTool.RECTANGLE, "Rectangle", "R"),
        (ManualCleanupTool.LASSO, "Lasso", "L"),
        (ManualCleanupTool.BRUSH, "Brush", "B"),
        (ManualCleanupTool.ERASER, "Eraser", "E"),
        (ManualCleanupTool.PROTECT, "Protect", "P"),
    )
    _CANCEL_MESSAGE = (
        "Cancel requested; current inpainting call will finish, result will not "
        "be applied."
    )

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        project_path: str,
        page_id: str,
        use_gpu: bool,
        pipeline_idle: bool,
        pipeline_block_reason: str = "",
        coverage_target: UserParentCleanupCoverageTargetV1 | None = None,
        initial_tool: str = ManualCleanupTool.BRUSH.value,
        initial_brush_radius: int = 12,
        initial_grow_px: int = 0,
        initial_feather_px: int = 0,
    ) -> None:
        super().__init__(parent)
        project_path = str(project_path or "").strip()
        page_id = str(page_id or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        if not page_id:
            raise ValueError("page_id is required")
        if type(use_gpu) is not bool or type(pipeline_idle) is not bool:
            raise TypeError("use_gpu and pipeline_idle must be bool values")
        try:
            stable_initial_tool = ManualCleanupTool(str(initial_tool))
        except ValueError as exc:
            raise ValueError("initial_tool is not a supported cleanup tool") from exc
        for field_name, value, minimum, maximum in (
            ("initial_brush_radius", initial_brush_radius, 1, 256),
            ("initial_grow_px", initial_grow_px, 0, 64),
            ("initial_feather_px", initial_feather_px, 0, 64),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"{field_name} must be between {minimum} and {maximum}"
                )
        if coverage_target is not None:
            if not isinstance(coverage_target, UserParentCleanupCoverageTargetV1):
                raise TypeError(
                    "coverage_target must be a UserParentCleanupCoverageTargetV1 or None"
                )
            coverage_target = UserParentCleanupCoverageTargetV1.from_dict(
                coverage_target.to_dict()
            )
            if coverage_target.page_id != page_id:
                raise ValueError("coverage target belongs to another page")

        self._project_path = project_path
        self._page_id = page_id
        self._coverage_target = coverage_target
        self._use_gpu = use_gpu
        self._pipeline_idle = pipeline_idle
        self._pipeline_block_reason = str(pipeline_block_reason or "").strip()
        self._initial_tool = stable_initial_tool
        self._initial_brush_radius = int(initial_brush_radius)
        self._initial_grow_px = int(initial_grow_px)
        self._initial_feather_px = int(initial_feather_px)
        self._context: ManualCleanupContext | None = None
        self._model: ManualCleanupEditorModel | None = None
        self._context_thread: QtCore.QThread | None = None
        self._context_worker: ManualCleanupContextWorker | None = None
        self._worker_thread: QtCore.QThread | None = None
        self._worker: ManualCleanupWorker | None = None
        self._local_thread_settlements: dict[
            int,
            tuple[
                QtCore.QThread,
                QtCore.QObject,
                str,
                str,
                Callable[[], None],
            ],
        ] = {}
        self._pending_context_terminal: tuple[str, object] | None = None
        self._pending_worker_terminal: tuple[str, object] | None = None
        self._close_after_worker = False
        self._accept_after_worker = False
        self._committed_receipt: ManualCleanupReceipt | None = None
        self._preview_path = ""
        self._image_paths = ("", "", "")

        self.setWindowTitle(
            (
                f"Confirm Cleanup Coverage - {page_id}"
                if coverage_target is not None
                else f"Manual Cleanup - {page_id}"
            )
        )
        self.setWindowModality(QtCore.Qt.WindowModal)
        self.resize(1320, 860)
        self.setMinimumSize(1040, 700)
        self.setObjectName("manualCleanupDialog")
        self.setAccessibleName(
            "Confirm selected parent cleanup coverage"
            if coverage_target is not None
            else "Manual cleanup editor"
        )
        self.setAccessibleDescription(
            (
                "Author an erase mask, optionally protect pixels, preview the result, "
                "then explicitly confirm the selected parent clean base."
            )
            if coverage_target is not None
            else "Author page-bounded manual cleanup masks and preview before commit."
        )
        self._setup_ui()
        self._set_initial_state()

    @property
    def committed_receipt(self) -> ManualCleanupReceipt | None:
        return self._committed_receipt

    @property
    def coverage_target(self) -> UserParentCleanupCoverageTargetV1 | None:
        return self._coverage_target

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if (
            self._pipeline_idle
            and self._context is None
            and self._context_thread is None
        ):
            QtCore.QTimer.singleShot(0, self._start_context_worker)

    def _wire_local_one_shot_thread(
        self,
        *,
        thread: QtCore.QThread,
        worker: QtCore.QObject,
        thread_attribute: str,
        worker_attribute: str,
        on_settled: Callable[[], None],
    ) -> None:
        """Own either dialog worker under the shared settlement contract."""

        if getattr(self, thread_attribute) is not None:
            raise RuntimeError(f"{thread_attribute} already owns a running thread")
        if getattr(self, worker_attribute) is not None:
            raise RuntimeError(f"{worker_attribute} already owns a running worker")
        key = id(thread)
        if key in self._local_thread_settlements:
            raise RuntimeError("manual cleanup thread is already wired")
        setattr(self, thread_attribute, thread)
        setattr(self, worker_attribute, worker)
        self._local_thread_settlements[key] = (
            thread,
            worker,
            thread_attribute,
            worker_attribute,
            on_settled,
        )
        worker.finished.connect(
            thread.quit,
            QtCore.Qt.ConnectionType.DirectConnection,
        )
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(
            self._settle_local_one_shot_thread,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )

    @QtCore.Slot()
    def _settle_local_one_shot_thread(self) -> None:
        sender = self.sender()
        if not isinstance(sender, QtCore.QThread):
            self._show_context_error(
                "A cleanup worker lost its thread identity; controls remain locked."
            )
            return
        settlement = self._local_thread_settlements.get(id(sender))
        if settlement is None or settlement[0] is not sender:
            self._show_context_error(
                "A cleanup worker lost its settlement record; controls remain locked."
            )
            return
        thread, worker, thread_attribute, worker_attribute, on_settled = settlement
        if (
            getattr(self, thread_attribute) is not thread
            or getattr(self, worker_attribute) is not worker
        ):
            self._show_context_error(
                "Cleanup worker ownership changed before settlement; controls remain locked."
            )
            return
        try:
            joined = thread.wait()
        except RuntimeError:
            joined = False
        if not joined:
            self._show_context_error(
                "A cleanup worker did not stop cleanly; controls remain locked."
            )
            return
        setattr(self, worker_attribute, None)
        setattr(self, thread_attribute, None)
        self._local_thread_settlements.pop(id(thread), None)
        thread.deleteLater()
        on_settled()

    def _queue_context_terminal(self, kind: str, value: object) -> None:
        if self._pending_context_terminal is not None:
            self._show_context_error(
                "Manual cleanup initialization returned more than one result."
            )
            return
        self._pending_context_terminal = (str(kind), value)

    def _queue_worker_terminal(self, kind: str, value: object) -> None:
        if self._pending_worker_terminal is not None:
            self.status_label.setText(
                "Manual cleanup returned more than one terminal result."
            )
            return
        self._pending_worker_terminal = (str(kind), value)

    def _setup_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        header = self.create_dialog_header(
            title=(
                "Confirm cleanup coverage"
                if self._coverage_target is not None
                else "Manual cleanup"
            ),
            subtitle=(
                (
                    "Paint a non-empty erase mask for this selected user parent. "
                    "The dashed workflow area is a guide only; preview, inspect, "
                    "then explicitly confirm the clean base."
                )
                if self._coverage_target is not None
                else (
                    "Paint only the remaining source marks. Automatic cleanup proof "
                    "stays unchanged; commit creates a new clean-base revision."
                )
            ),
            icon_name="cleanup",
            close_accessible_name="Cancel manual cleanup",
        )
        page_badge = QtWidgets.QLabel(self._page_id)
        page_badge.setObjectName("manualCleanupPageBadge")
        page_badge.setAlignment(QtCore.Qt.AlignCenter)
        page_badge.setMinimumWidth(110)
        header.add_trailing_widget(page_badge)
        root.addWidget(header)

        self.coverage_provenance = QtWidgets.QLabel()
        self.coverage_provenance.setObjectName("manualCleanupCoverageProvenance")
        self.coverage_provenance.setProperty("role", "secondary")
        self.coverage_provenance.setWordWrap(True)
        self.coverage_provenance.setAccessibleName(
            "Selected parent cleanup provenance"
        )
        if self._coverage_target is not None:
            role = {
                "speech": "Dialogue",
                "caption": "Caption",
            }.get(
                self._coverage_target.parent_role,
                self._coverage_target.parent_role.replace("_", " ").title(),
            )
            self.coverage_provenance.setText(
                f"Selected user parent · {role} · Source current · Translation "
                "current · Cleanup coverage required · Later stages remain explicit."
            )
            self.coverage_provenance.setAccessibleDescription(
                "This confirmation is bound to one exact selected user parent, its "
                "current source and translation revisions, original page, and input "
                "clean base. Automatic cleanup proof remains unchanged."
            )
            self.coverage_provenance.setVisible(True)
        else:
            self.coverage_provenance.setVisible(False)
        root.addWidget(self.coverage_provenance)

        self.rebase_notice = QtWidgets.QLabel()
        self.rebase_notice.setObjectName("manualCleanupRebaseNotice")
        self.rebase_notice.setWordWrap(True)
        self.rebase_notice.setVisible(False)
        root.addWidget(self.rebase_notice)

        body = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        body.setChildrenCollapsible(False)
        root.addWidget(body, 1)

        canvas_panel = QtWidgets.QWidget(body)
        canvas_layout = QtWidgets.QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(8)

        comparison_row = QtWidgets.QHBoxLayout()
        self.comparison_tabs = QtWidgets.QTabBar()
        self.comparison_tabs.setObjectName("cleanupComparisonTabs")
        self.comparison_tabs.setExpanding(False)
        self.comparison_tabs.addTab("Original")
        self.comparison_tabs.addTab("Current Cleaned")
        self.comparison_tabs.addTab("Preview")
        self.comparison_tabs.setTabEnabled(2, False)
        self.comparison_tabs.setCurrentIndex(1)
        self.comparison_tabs.setAccessibleName("Cleanup comparison image")
        self.comparison_tabs.setAccessibleDescription(
            "Compare the original page, current clean base, and explicit preview."
        )
        comparison_row.addWidget(self.comparison_tabs)
        comparison_row.addStretch(1)
        self.zoom_out_btn = QtWidgets.QToolButton()
        self.zoom_out_btn.setText("-")
        self.zoom_out_btn.setToolTip("Zoom out")
        self.zoom_out_btn.setAccessibleName("Zoom out")
        self.fit_btn = QtWidgets.QToolButton()
        self.fit_btn.setText("Fit")
        self.fit_btn.setToolTip("Fit the full page (F)")
        self.fit_btn.setAccessibleName("Fit full page")
        self.zoom_in_btn = QtWidgets.QToolButton()
        self.zoom_in_btn.setText("+")
        self.zoom_in_btn.setToolTip("Zoom in")
        self.zoom_in_btn.setAccessibleName("Zoom in")
        comparison_row.addWidget(self.zoom_out_btn)
        comparison_row.addWidget(self.fit_btn)
        comparison_row.addWidget(self.zoom_in_btn)
        canvas_layout.addLayout(comparison_row)

        self.canvas = ManualCleanupCanvas()
        self.canvas.setObjectName("manualCleanupCanvas")
        self.canvas.setAccessibleName("Manual cleanup page canvas")
        self.canvas.setAccessibleDescription(
            (
                "Author page-bounded erase and protect masks. The dashed selected-parent "
                "workflow rectangle is a guide only and never becomes mask authority."
            )
            if self._coverage_target is not None
            else "Author page-bounded erase and protect masks."
        )
        canvas_layout.addWidget(self.canvas, 1)

        canvas_footer = QtWidgets.QHBoxLayout()
        self.coordinate_label = QtWidgets.QLabel("x --  y --")
        self.coordinate_label.setObjectName("cleanupCoordinates")
        canvas_footer.addWidget(self.coordinate_label)
        canvas_footer.addStretch(1)
        erase_legend = QtWidgets.QLabel("Erase mask")
        erase_legend.setObjectName("eraseMaskLegend")
        protect_legend = QtWidgets.QLabel("Protect mask")
        protect_legend.setObjectName("protectMaskLegend")
        erase_palette = erase_legend.palette()
        erase_palette.setColor(
            QtGui.QPalette.WindowText,
            QtGui.QColor(self.canvas._ERASE_COLOR).lighter(135),
        )
        erase_legend.setPalette(erase_palette)
        protect_palette = protect_legend.palette()
        protect_palette.setColor(
            QtGui.QPalette.WindowText,
            QtGui.QColor(self.canvas._PROTECT_COLOR).lighter(125),
        )
        protect_legend.setPalette(protect_palette)
        canvas_footer.addWidget(erase_legend)
        canvas_footer.addSpacing(12)
        canvas_footer.addWidget(protect_legend)
        self.workflow_guide_legend = QtWidgets.QLabel(
            "Dashed outline · workflow guide only"
        )
        self.workflow_guide_legend.setObjectName("workflowGuideLegend")
        self.workflow_guide_legend.setProperty("role", "secondary")
        self.workflow_guide_legend.setVisible(self._coverage_target is not None)
        canvas_footer.addSpacing(12)
        canvas_footer.addWidget(self.workflow_guide_legend)
        canvas_layout.addLayout(canvas_footer)

        controls = QtWidgets.QWidget(body)
        controls.setObjectName("manualCleanupControls")
        controls.setMinimumWidth(300 if self._coverage_target is not None else 275)
        controls.setMaximumWidth(440 if self._coverage_target is not None else 340)
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(14, 0, 0, 0)
        controls_layout.setSpacing(10)

        history_group = QtWidgets.QGroupBox("History")
        history_layout = QtWidgets.QHBoxLayout(history_group)
        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.undo_btn.setObjectName("cleanupUndoButton")
        self.undo_btn.setShortcut(QtGui.QKeySequence.Undo)
        self.redo_btn = QtWidgets.QPushButton("Redo")
        self.redo_btn.setObjectName("cleanupRedoButton")
        self.redo_btn.setShortcut(QtGui.QKeySequence.Redo)
        history_layout.addWidget(self.undo_btn)
        history_layout.addWidget(self.redo_btn)
        controls_layout.addWidget(history_group)

        tools_group = QtWidgets.QGroupBox("Mask tools")
        tools_layout = QtWidgets.QGridLayout(tools_group)
        self.tool_group = QtWidgets.QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_buttons: dict[ManualCleanupTool, QtWidgets.QToolButton] = {}
        for index, (tool, label, shortcut) in enumerate(self._TOOL_LABELS):
            button = QtWidgets.QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            button.setMinimumHeight(34)
            button.setObjectName(f"cleanupTool{tool.value.title()}Button")
            button.setToolTip(f"{label} mask tool ({shortcut})")
            self.tool_group.addButton(button)
            self.tool_buttons[tool] = button
            tools_layout.addWidget(button, index // 2, index % 2)
            QtGui.QShortcut(QtGui.QKeySequence(shortcut), self, lambda value=tool: self._select_tool(value))
        self.tool_buttons[ManualCleanupTool.BRUSH].setChecked(True)
        controls_layout.addWidget(tools_group)

        size_group = QtWidgets.QGroupBox("Mask refinement")
        size_layout = QtWidgets.QFormLayout(size_group)
        self.brush_radius = WheelSafeSpinBox()
        self.brush_radius.setObjectName("cleanupBrushRadius")
        self.brush_radius.setRange(1, 256)
        self.brush_radius.setValue(12)
        self.brush_radius.setSuffix(" px")
        self.grow_spin = WheelSafeSpinBox()
        self.grow_spin.setObjectName("cleanupGrowPixels")
        self.grow_spin.setRange(0, 64)
        self.grow_spin.setSuffix(" px")
        self.feather_spin = WheelSafeSpinBox()
        self.feather_spin.setObjectName("cleanupFeatherPixels")
        self.feather_spin.setRange(0, 64)
        self.feather_spin.setSuffix(" px")
        self.eraser_layer = QtWidgets.QComboBox()
        self.eraser_layer.setObjectName("cleanupEraserLayer")
        self.eraser_layer.addItem(
            "Erase red mask",
            ManualCleanupMaskLayer.ERASE.value,
        )
        self.eraser_layer.addItem(
            "Erase cyan protect mask",
            ManualCleanupMaskLayer.PROTECT.value,
        )
        size_layout.addRow("Brush radius", self.brush_radius)
        size_layout.addRow("Grow", self.grow_spin)
        size_layout.addRow("Feather", self.feather_spin)
        size_layout.addRow("Eraser affects", self.eraser_layer)
        controls_layout.addWidget(size_group)

        self.clear_btn = QtWidgets.QPushButton("Clear masks")
        self.clear_btn.setObjectName("cleanupClearButton")
        controls_layout.addWidget(self.clear_btn)

        safety = QtWidgets.QLabel(
            "Red pixels are inpainted. Cyan pixels are protected. The current "
            "clean base is never overwritten."
        )
        safety.setWordWrap(True)
        safety.setObjectName("manualCleanupSafetyNote")
        controls_layout.addWidget(safety)
        controls_layout.addStretch(1)

        body.addWidget(canvas_panel)
        body.addWidget(controls)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 0)

        status_row = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Preparing manual cleanup editor...")
        self.status_label.setObjectName("manualCleanupStatus")
        self.status_label.setWordWrap(True)
        self.status_label.setAccessibleName("Manual cleanup operation status")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setObjectName("manualCleanupProgress")
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumWidth(230)
        self.progress.setAccessibleName("Manual cleanup progress")
        self.cancel_operation_btn = QtWidgets.QPushButton("Cancel operation")
        self.cancel_operation_btn.setObjectName("cleanupCancelOperationButton")
        self.cancel_operation_btn.setEnabled(False)
        self.cancel_operation_btn.setAccessibleName("Cancel cleanup operation")
        self.cancel_operation_btn.setAccessibleDescription(
            "Request cancellation before persistence begins."
        )
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.progress)
        status_row.addWidget(self.cancel_operation_btn)
        root.addLayout(status_row)

        actions = QtWidgets.QHBoxLayout()
        self.preview_btn = QtWidgets.QPushButton(
            "Preview clean-base coverage"
            if self._coverage_target is not None
            else "Preview Cleanup"
        )
        self.preview_btn.setObjectName("cleanupPreviewButton")
        self.preview_btn.setDefault(True)
        self.preview_btn.setAccessibleName(self.preview_btn.text())
        self.preview_btn.setAccessibleDescription(
            "Run a non-publishing cleanup preview for the exact authored masks."
        )
        self.preview_btn.setToolTip(self.preview_btn.accessibleDescription())
        self.commit_btn = QtWidgets.QPushButton(
            "Confirm clean base"
            if self._coverage_target is not None
            else "Commit new clean base"
        )
        self.commit_btn.setObjectName("cleanupCommitButton")
        self.commit_btn.setAccessibleName(self.commit_btn.text())
        self.commit_btn.setAccessibleDescription(
            "Publish only the inspected preview as the selected clean-base revision."
        )
        self.commit_btn.setToolTip(self.commit_btn.accessibleDescription())
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setObjectName("cleanupCancelButton")
        self.cancel_btn.setAccessibleName("Cancel manual cleanup")
        self.cancel_btn.setAccessibleDescription(
            "Close without publishing the current mask or preview."
        )
        self.cancel_btn.setToolTip(self.cancel_btn.accessibleDescription())
        actions.addStretch(1)
        actions.addWidget(self.preview_btn)
        actions.addWidget(self.commit_btn)
        actions.addWidget(self.cancel_btn)
        root.addLayout(actions)

        self.comparison_tabs.currentChanged.connect(self._show_comparison_image)
        self.zoom_out_btn.clicked.connect(lambda: self.canvas.zoom_by(1.0 / 1.15))
        self.zoom_in_btn.clicked.connect(lambda: self.canvas.zoom_by(1.15))
        self.fit_btn.clicked.connect(self.canvas.fit_page)
        QtGui.QShortcut(QtGui.QKeySequence("F"), self, self.canvas.fit_page)
        self.canvas.cursor_moved.connect(
            lambda x, y: self.coordinate_label.setText(f"x {x}  y {y}")
        )
        self.canvas.operation_added.connect(self._add_mask_command)
        for tool, button in self.tool_buttons.items():
            button.clicked.connect(lambda _checked=False, value=tool: self._select_tool(value))
        self.brush_radius.valueChanged.connect(self._set_brush_radius)
        self.grow_spin.valueChanged.connect(self._set_parameters)
        self.feather_spin.valueChanged.connect(self._set_parameters)
        self.eraser_layer.currentIndexChanged.connect(self._set_eraser_layer)
        self.undo_btn.clicked.connect(self._undo)
        self.redo_btn.clicked.connect(self._redo)
        self.clear_btn.clicked.connect(self._clear_masks)
        self.preview_btn.clicked.connect(self._preview)
        self.commit_btn.clicked.connect(self._commit)
        self.cancel_operation_btn.clicked.connect(self._cancel_operation)
        self.cancel_btn.clicked.connect(self._request_close)
        self.setTabOrder(self.preview_btn, self.commit_btn)
        self.setTabOrder(self.commit_btn, self.cancel_btn)

    def _set_initial_state(self) -> None:
        self._set_editor_controls_enabled(False)
        self.preview_btn.setEnabled(False)
        self.commit_btn.setEnabled(False)
        if not self._pipeline_idle:
            message = self._pipeline_block_reason or (
                "Manual cleanup is unavailable while the forward pipeline is running."
            )
            self.status_label.setText(message)
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
        else:
            self.status_label.setText("Loading the selected clean-base revision...")
            self.progress.setRange(0, 0)

    def _start_context_worker(self) -> None:
        if (
            not self._pipeline_idle
            or self._context_thread is not None
            or self._worker_thread is not None
        ):
            return
        try:
            command = ManualCleanupContextCommand(
                project_path=self._project_path,
                page_id=self._page_id,
                coverage_target=self._coverage_target,
            )
        except (TypeError, ValueError) as exc:
            self._show_context_error(str(exc))
            return
        thread = QtCore.QThread(self)
        worker = ManualCleanupContextWorker(command)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.context_ready.connect(self._on_context_ready)
        worker.failure.connect(self._on_context_failure)
        self._wire_local_one_shot_thread(
            thread=thread,
            worker=worker,
            thread_attribute="_context_thread",
            worker_attribute="_context_worker",
            on_settled=self._on_context_thread_finished,
        )
        thread.start()

    @QtCore.Slot(object)
    def _on_context_ready(self, context: object) -> None:
        if self._context_thread is not None:
            self._queue_context_terminal("ready", context)
            return
        self._present_context_ready(context)

    def _present_context_ready(self, context: object) -> None:
        if not isinstance(context, ManualCleanupContext):
            self._show_context_error("Manual cleanup returned an invalid editor context.")
            return
        if context.page_id != self._page_id:
            self._show_context_error("Manual cleanup context belongs to another page.")
            return
        self._context = context
        if not context.ready:
            self._show_context_error(context.message)
            return
        try:
            parameters = ManualCleanupParameters(
                grow_px=self._initial_grow_px,
                feather_px=self._initial_feather_px,
                use_gpu=self._use_gpu,
            )
            self._model = ManualCleanupEditorModel.from_context(
                context,
                parameters=parameters,
                coverage_target=self._coverage_target,
            )
            self._model.set_brush_radius(float(self._initial_brush_radius))
            self._model.select_tool(self._initial_tool)
            self.canvas.set_canvas_size(*context.canvas_size)
            self.canvas.set_workflow_area_guide(
                self._coverage_target.workflow_area_bbox
                if self._coverage_target is not None
                else None
            )
            parameters = self._model.state.parameters
            grow_blocker = QtCore.QSignalBlocker(self.grow_spin)
            feather_blocker = QtCore.QSignalBlocker(self.feather_spin)
            brush_blocker = QtCore.QSignalBlocker(self.brush_radius)
            tool_blockers = tuple(
                QtCore.QSignalBlocker(button)
                for button in self.tool_buttons.values()
            )
            self.grow_spin.setValue(parameters.grow_px)
            self.feather_spin.setValue(parameters.feather_px)
            self.brush_radius.setValue(self._initial_brush_radius)
            self.tool_buttons[self._initial_tool].setChecked(True)
            del grow_blocker, feather_blocker, brush_blocker, tool_blockers
        except (RuntimeError, TypeError, ValueError) as exc:
            self._show_context_error(str(exc))
            return
        self._image_paths = (
            context.source_image_path,
            context.selected_base_path,
            "",
        )
        if context.rebase_review is not None:
            self.rebase_notice.setText(
                "Rebase review - this saved mask is shown over Current Cleaned. "
                "Preview runs the cleanup backend again on the current base; "
                "the previous result pixels are never reused."
            )
            self.rebase_notice.setVisible(True)
            self.preview_btn.setText("Preview on current base")
        else:
            self.rebase_notice.clear()
            self.rebase_notice.setVisible(False)
            self.preview_btn.setText(
                "Preview clean-base coverage"
                if self._coverage_target is not None
                else "Preview Cleanup"
            )
            self.preview_btn.setAccessibleName(self.preview_btn.text())
        self._show_comparison_image(self.comparison_tabs.currentIndex())
        self.status_label.setText(context.message)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self._apply_state(self._model.state)

    @QtCore.Slot(object)
    def _on_context_failure(self, failure: object) -> None:
        if self._context_thread is not None:
            self._queue_context_terminal("failure", failure)
            return
        self._present_context_failure(failure)

    def _present_context_failure(self, failure: object) -> None:
        message = (
            failure.message
            if isinstance(failure, ManualCleanupWorkerFailure)
            else "Manual cleanup context could not be loaded."
        )
        self._show_context_error(message)

    @QtCore.Slot()
    def _on_context_thread_finished(self) -> None:
        if self._close_after_worker:
            self._close_after_worker = False
            self._pending_context_terminal = None
            QtWidgets.QDialog.reject(self)
            return
        pending = self._pending_context_terminal
        self._pending_context_terminal = None
        if pending is None:
            self._show_context_error(
                "Manual cleanup initialization ended without a typed result."
            )
            return
        kind, value = pending
        if kind == "ready":
            self._present_context_ready(value)
        else:
            self._present_context_failure(value)

    def _show_context_error(self, message: str) -> None:
        self.status_label.setText(str(message or "Manual cleanup is unavailable."))
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self._set_editor_controls_enabled(False)
        self.preview_btn.setEnabled(False)
        self.commit_btn.setEnabled(False)

    def _show_comparison_image(self, index: int) -> None:
        if not 0 <= int(index) < len(self._image_paths):
            return
        self.canvas.set_rebase_overlay_visible(int(index) == 1)
        path = self._image_paths[int(index)]
        if path and self.canvas.set_image_path(path):
            return
        if int(index) == 2:
            self.status_label.setText("Create a cleanup preview to inspect the result.")
        elif self._context is not None:
            self.status_label.setText("The selected comparison image is unavailable.")

    def _select_tool(self, tool: ManualCleanupTool) -> None:
        model = self._model
        if model is None or model.state.busy:
            return
        try:
            eraser_layer = ManualCleanupMaskLayer(self.eraser_layer.currentData())
        except (TypeError, ValueError):
            eraser_layer = ManualCleanupMaskLayer.ERASE
        try:
            state = model.select_tool(
                ManualCleanupTool(tool),
                eraser_layer=eraser_layer,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self.status_label.setText(str(exc))
            return
        self.tool_buttons[ManualCleanupTool(tool)].setChecked(True)
        self._apply_state(state)

    def _set_eraser_layer(self) -> None:
        if self._model is not None and self._model.state.selected_tool is ManualCleanupTool.ERASER:
            self._select_tool(ManualCleanupTool.ERASER)

    def _set_brush_radius(self, value: int) -> None:
        model = self._model
        if model is None or model.state.busy:
            return
        try:
            self._apply_state(model.set_brush_radius(float(value)))
        except (RuntimeError, TypeError, ValueError) as exc:
            self.status_label.setText(str(exc))

    def _set_parameters(self) -> None:
        model = self._model
        if model is None or model.state.busy:
            return
        had_preview = model.state.preview_lease is not None
        try:
            state = model.set_parameters(
                grow_px=self.grow_spin.value(),
                feather_px=self.feather_spin.value(),
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self.status_label.setText(str(exc))
            return
        if had_preview and state.preview_lease is None:
            self._discard_preview()
        self._apply_state(state)

    @QtCore.Slot(object)
    def _add_mask_command(self, command: object) -> None:
        model = self._model
        if model is None or not isinstance(command, ManualCleanupMaskCommand):
            return
        had_preview = model.state.preview_lease is not None
        try:
            state = model.add_command(command)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.status_label.setText(str(exc))
            return
        if had_preview:
            self._discard_preview()
        self._apply_state(state)

    def _undo(self) -> None:
        self._mutate_history("undo")

    def _redo(self) -> None:
        self._mutate_history("redo")

    def _clear_masks(self) -> None:
        self._mutate_history("clear")

    def _mutate_history(self, method_name: str) -> None:
        model = self._model
        if model is None or model.state.busy:
            return
        had_preview = model.state.preview_lease is not None
        try:
            state = getattr(model, method_name)()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self.status_label.setText(str(exc))
            return
        if had_preview and state.preview_lease is None:
            self._discard_preview()
        self._apply_state(state)

    def _preview(self) -> None:
        model = self._model
        if model is None or self._worker_thread is not None:
            return
        try:
            payload = model.export_mask_payload()
            command = ManualCleanupWorkerCommand(
                project_path=self._project_path,
                page_id=self._page_id,
                erase_mask_png=payload.erase_mask_png,
                protect_mask_png=payload.protect_mask_png,
                parameters=model.state.parameters,
                mode=ManualCleanupWorkerMode.PREVIEW,
                rebase_review=model.state.rebase_review,
                coverage_target=self._coverage_target,
            )
            state = model.begin(command)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.status_label.setText(str(exc))
            return
        self._start_worker(command, state)

    def _commit(self) -> None:
        model = self._model
        if model is None or self._worker_thread is not None:
            return
        lease = model.state.preview_lease
        if lease is None:
            return
        try:
            payload = model.export_mask_payload()
            command = ManualCleanupWorkerCommand(
                project_path=self._project_path,
                page_id=self._page_id,
                erase_mask_png=payload.erase_mask_png,
                protect_mask_png=payload.protect_mask_png,
                parameters=model.state.parameters,
                mode=ManualCleanupWorkerMode.COMMIT,
                preview_lease=lease,
                coverage_target=self._coverage_target,
            )
            state = model.begin(command)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.status_label.setText(str(exc))
            return
        self._start_worker(command, state)

    def _start_worker(
        self,
        command: ManualCleanupWorkerCommand,
        state: ManualCleanupViewState,
    ) -> None:
        if self._worker_thread is not None or self._context_thread is not None:
            return
        thread = QtCore.QThread(self)
        worker = ManualCleanupWorker(command)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.preflight.connect(self._on_preflight)
        worker.progress.connect(self._on_progress)
        worker.cancellation.connect(self._on_cancellation_state)
        worker.preview_ready.connect(self._on_preview_ready)
        worker.committed.connect(self._on_committed)
        worker.cancelled.connect(self._on_cancelled)
        worker.failure.connect(self._on_failure)
        self._wire_local_one_shot_thread(
            thread=thread,
            worker=worker,
            thread_attribute="_worker_thread",
            worker_attribute="_worker",
            on_settled=self._on_worker_thread_finished,
        )
        self._apply_state(state)
        thread.start()

    @QtCore.Slot(object)
    def _on_preflight(self, value: object) -> None:
        model = self._model
        if model is None:
            return
        try:
            self._apply_state(model.accept_preflight(value))
        except (RuntimeError, TypeError, ValueError):
            return

    @QtCore.Slot(object)
    def _on_progress(self, value: object) -> None:
        model = self._model
        if model is None:
            return
        try:
            self._apply_state(model.accept_progress(value))
        except (RuntimeError, TypeError, ValueError):
            return

    @QtCore.Slot(object)
    def _on_cancellation_state(self, value: object) -> None:
        model = self._model
        if model is None or not isinstance(value, ManualCleanupCancellationState):
            return
        if not model.state.busy:
            return
        try:
            self._apply_state(model.accept_cancellation_state(value))
        except (RuntimeError, TypeError, ValueError):
            return

    @QtCore.Slot(object)
    def _on_preview_ready(self, receipt: object) -> None:
        if self._worker_thread is not None:
            self._queue_worker_terminal("preview", receipt)
            return
        self._present_preview_ready(receipt)

    def _present_preview_ready(self, receipt: object) -> None:
        model = self._model
        if model is None or not isinstance(receipt, ManualCleanupReceipt):
            return
        try:
            state = model.accept_preview(receipt)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.status_label.setText(str(exc))
            return
        lease = receipt.preview_lease
        self._preview_path = lease.result_path if lease is not None else ""
        self._image_paths = (
            self._image_paths[0],
            self._image_paths[1],
            self._preview_path,
        )
        self.comparison_tabs.setTabEnabled(2, bool(self._preview_path))
        if self._preview_path:
            self.comparison_tabs.setCurrentIndex(2)
        self._apply_state(state)

    @QtCore.Slot(object)
    def _on_committed(self, receipt: object) -> None:
        if self._worker_thread is not None:
            self._queue_worker_terminal("committed", receipt)
            return
        self._present_committed(receipt)

    def _present_committed(self, receipt: object) -> None:
        model = self._model
        if model is None or not isinstance(receipt, ManualCleanupReceipt):
            return
        try:
            state = model.accept_commit(receipt)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.status_label.setText(str(exc))
            return
        self._committed_receipt = receipt
        self._accept_after_worker = True
        self._apply_state(state)

    @QtCore.Slot(object)
    def _on_cancelled(self, receipt: object) -> None:
        if self._worker_thread is not None:
            self._queue_worker_terminal("cancelled", receipt)
            return
        self._present_cancelled(receipt)

    def _present_cancelled(self, receipt: object) -> None:
        model = self._model
        if model is None or not isinstance(receipt, ManualCleanupReceipt):
            return
        try:
            self._apply_state(model.accept_cancelled(receipt))
        except (RuntimeError, TypeError, ValueError):
            return

    @QtCore.Slot(object)
    def _on_failure(self, failure: object) -> None:
        if self._worker_thread is not None:
            self._queue_worker_terminal("failure", failure)
            return
        self._present_failure(failure)

    def _present_failure(self, failure: object) -> None:
        model = self._model
        if model is None or not isinstance(failure, ManualCleanupWorkerFailure):
            return
        try:
            self._apply_state(model.accept_failure(failure))
        except (RuntimeError, TypeError, ValueError):
            self.status_label.setText(failure.message)

    @QtCore.Slot()
    def _on_worker_thread_finished(self) -> None:
        pending = self._pending_worker_terminal
        self._pending_worker_terminal = None
        if self._close_after_worker and (
            pending is None or pending[0] != "committed"
        ):
            self._close_after_worker = False
            self._discard_preview()
            QtWidgets.QDialog.reject(self)
            return
        if pending is None:
            self.status_label.setText(
                "Manual cleanup ended without a typed terminal result."
            )
            if self._model is not None:
                self._apply_state(self._model.state)
            return
        kind, value = pending
        if kind == "preview":
            self._present_preview_ready(value)
        elif kind == "committed":
            self._present_committed(value)
        elif kind == "cancelled":
            self._present_cancelled(value)
        else:
            self._present_failure(value)
        if self._accept_after_worker:
            self._accept_after_worker = False
            self._close_after_worker = False
            if self._committed_receipt is not None:
                self.cleanup_committed.emit(self._committed_receipt)
            QtWidgets.QDialog.accept(self)

    def _apply_state(self, state: ManualCleanupViewState) -> None:
        self.canvas.apply_state(state)
        busy = (
            self._context_thread is not None
            or self._worker_thread is not None
            or state.busy
        )
        self._set_editor_controls_enabled(state.editing_enabled and not busy)
        self.undo_btn.setEnabled(state.undo_enabled and not busy)
        self.redo_btn.setEnabled(state.redo_enabled and not busy)
        self.clear_btn.setEnabled(state.reset_enabled and not busy)
        self.preview_btn.setEnabled(state.preview_enabled and not busy)
        self.commit_btn.setEnabled(state.commit_enabled and not busy)
        self.cancel_operation_btn.setEnabled(state.cancel_enabled and busy)
        self.status_label.setText(state.message)
        progress = state.progress
        if busy and progress is None:
            self.progress.setRange(0, 0)
        else:
            total = max(1, int(getattr(progress, "total_steps", 1) or 1))
            completed = min(
                total,
                max(0, int(getattr(progress, "completed_steps", 0) or 0)),
            )
            self.progress.setRange(0, total)
            self.progress.setValue(completed)
        self.tool_buttons[state.selected_tool].setChecked(True)
        self.eraser_layer.setEnabled(
            state.editing_enabled
            and not busy
            and state.selected_tool is ManualCleanupTool.ERASER
        )
        preview_is_default = bool(
            self._coverage_target is None
            or state.phase is not ManualCleanupViewPhase.PREVIEW_READY
        )
        self.preview_btn.setDefault(preview_is_default)
        self.commit_btn.setDefault(not preview_is_default)
        if (
            self._coverage_target is not None
            and state.phase is ManualCleanupViewPhase.PREVIEW_READY
            and self.commit_btn.isEnabled()
            and self.isVisible()
        ):
            self.commit_btn.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _set_editor_controls_enabled(self, enabled: bool) -> None:
        context_ready = self._context is not None and self._context.ready
        self.canvas.setEnabled(context_ready)
        self.canvas.set_editing_enabled(enabled)
        for button in self.tool_buttons.values():
            button.setEnabled(enabled)
        self.brush_radius.setEnabled(enabled)
        self.grow_spin.setEnabled(enabled)
        self.feather_spin.setEnabled(enabled)
        self.eraser_layer.setEnabled(
            enabled
            and self._model is not None
            and self._model.state.selected_tool is ManualCleanupTool.ERASER
        )

    def _cancel_operation(self) -> None:
        worker = self._worker
        if worker is None:
            return
        accepted = worker.request_cancel()
        self.cancel_operation_btn.setEnabled(False)
        if accepted:
            self.status_label.setText(self._CANCEL_MESSAGE)
        else:
            self.status_label.setText(
                "Commit is being written and can no longer be cancelled."
            )

    def _request_close(self) -> None:
        if self._context_thread is not None:
            self._close_after_worker = True
            self.status_label.setText(
                "Finishing manual cleanup initialization before closing..."
            )
            return
        if self._worker_thread is not None:
            self._close_after_worker = True
            self._cancel_operation()
            return
        self._discard_preview()
        self.reject()

    def _discard_preview(self) -> None:
        if not self._preview_path and (
            self._model is None or self._model.state.preview_lease is None
        ):
            return
        try:
            discard_manual_cleanup_preview(self._project_path, self._page_id)
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
        self._preview_path = ""
        self._image_paths = (self._image_paths[0], self._image_paths[1], "")
        self.comparison_tabs.setTabEnabled(2, False)
        if self.comparison_tabs.currentIndex() == 2:
            self.comparison_tabs.setCurrentIndex(1)

    def reject(self) -> None:
        if self._context_thread is not None or self._worker_thread is not None:
            self._request_close()
            return
        self._discard_preview()
        super().reject()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._context_thread is not None or self._worker_thread is not None:
            self._request_close()
            event.ignore()
            return
        self._discard_preview()
        super().closeEvent(event)


__all__ = [
    "ManualCleanupDialog",
    "ManualCleanupCanvas",
]
