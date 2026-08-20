# -*- coding: utf-8 -*-
"""Zoomable page canvas with explicit artifact modes and independent overlays."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.design_system.tokens import overlay_token, theme_token
from app.ui.ui_contract import CANVAS_VIEW_IDS, OVERLAY_IDS


@dataclass(frozen=True, slots=True)
class CanvasArtifactSet:
    """Explicit page artifacts; missing entries are never substituted."""

    page_id: str
    original_path: str | None = None
    cleaned_path: str | None = None
    final_path: str | None = None

    def path_for(self, mode: str) -> str | None:
        if mode == "original":
            return self.original_path
        if mode == "cleaned":
            return self.cleaned_path
        if mode == "final":
            return self.final_path
        raise ValueError(f"unsupported single canvas mode: {mode!r}")


@dataclass(frozen=True, slots=True)
class OverlayShape:
    """One page-coordinate overlay primitive."""

    overlay_id: str
    shape_id: str
    kind: str
    points: tuple[float, ...]
    label: str = ""
    selected: bool = False

    def __post_init__(self) -> None:
        if self.overlay_id not in OVERLAY_IDS:
            raise ValueError(f"unsupported overlay: {self.overlay_id!r}")
        if self.kind not in {"rect", "line", "polygon"}:
            raise ValueError(f"unsupported overlay shape: {self.kind!r}")
        expected = {"rect": 4, "line": 4}
        if self.kind in expected and len(self.points) != expected[self.kind]:
            raise ValueError(f"{self.kind} overlays require {expected[self.kind]} values")
        if self.kind == "polygon" and (len(self.points) < 6 or len(self.points) % 2):
            raise ValueError("polygon overlays require at least three points")


@dataclass(frozen=True, slots=True)
class RasterOverlaySource:
    overlay_id: str
    asset_path: str
    asset_sha256: str
    canvas_size: tuple[int, int]
    label: str = ""

    def __post_init__(self) -> None:
        if self.overlay_id not in {"cleanupMask", "protectedRegions"}:
            raise ValueError(f"unsupported raster overlay: {self.overlay_id!r}")
        if not self.asset_path.strip():
            raise ValueError("asset_path must not be empty")
        digest = self.asset_sha256.strip().lower()
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError("asset_sha256 must be a SHA-256 digest")
        object.__setattr__(self, "asset_sha256", digest)
        if (
            len(self.canvas_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.canvas_size
            )
        ):
            raise ValueError("canvas_size must contain two positive integers")


@dataclass(frozen=True, slots=True)
class OverlayAvailability:
    overlay_id: str
    available: bool
    tooltip: str

    def __post_init__(self) -> None:
        if self.overlay_id not in OVERLAY_IDS:
            raise ValueError(f"unsupported overlay: {self.overlay_id!r}")
        if not isinstance(self.available, bool):
            raise TypeError("available must be a boolean")
        if not self.tooltip.strip():
            raise ValueError("tooltip must not be empty")


class _SplitCompareItem(QtWidgets.QGraphicsItem):
    """Paint Original and Final into one page rect without creating extra buffers."""

    def __init__(
        self,
        original: QtGui.QPixmap,
        final: QtGui.QPixmap,
        *,
        ratio: float = 0.5,
        theme: str = "dark",
    ) -> None:
        super().__init__()
        self._original = original
        self._final = final
        self._ratio = max(0.05, min(0.95, float(ratio)))
        self._theme = theme
        width = max(original.width(), final.width())
        height = max(original.height(), final.height())
        self._rect = QtCore.QRectF(0.0, 0.0, float(width), float(height))

    def boundingRect(self) -> QtCore.QRectF:  # noqa: N802 - Qt API
        return self._rect

    def set_ratio(self, ratio: float) -> None:
        normalized = max(0.05, min(0.95, float(ratio)))
        if normalized == self._ratio:
            return
        self._ratio = normalized
        self.update()

    def paint(
        self,
        painter: QtGui.QPainter,
        _option: QtWidgets.QStyleOptionGraphicsItem,
        _widget: QtWidgets.QWidget | None = None,
    ) -> None:
        split = self._rect.width() * self._ratio
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setClipRect(QtCore.QRectF(0.0, 0.0, split, self._rect.height()))
        painter.drawPixmap(self._rect, self._original, QtCore.QRectF(self._original.rect()))
        painter.restore()
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setClipRect(
            QtCore.QRectF(split, 0.0, self._rect.width() - split, self._rect.height())
        )
        painter.drawPixmap(self._rect, self._final, QtCore.QRectF(self._final.rect()))
        painter.restore()
        divider = QtGui.QPen(QtGui.QColor(overlay_token(self._theme, "selection")))
        divider.setCosmetic(True)
        divider.setWidth(2)
        painter.setPen(divider)
        painter.drawLine(QtCore.QPointF(split, 0.0), QtCore.QPointF(split, self._rect.height()))


class PageCanvasView(QtWidgets.QGraphicsView):
    """Native canvas; artifact absence is explicit and overlays are independent."""

    zoom_changed = QtCore.Signal(int)
    mode_changed = QtCore.Signal(str)
    overlay_changed = QtCore.Signal(str, bool)
    selection_changed = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        scene = QtWidgets.QGraphicsScene()
        super().__init__(scene, parent)
        # QGraphicsView does not own its scene. Keep one explicit Qt/Python
        # owner so an early persisted canvas-mode restore cannot observe a
        # collected scene before the first page is bound.
        scene.setParent(self)
        self._scene = scene
        self.setObjectName("pageCanvas")
        self.setAccessibleName("Page canvas")
        self.setAccessibleDescription(
            "Zoomable Original, Cleaned, Final, or Compare page view with textual overlay state"
        )
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
            | QtGui.QPainter.RenderHint.TextAntialiasing
        )
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self._artifacts = CanvasArtifactSet(page_id="")
        self._mode = "original"
        self._held_mode: str | None = None
        # Hybrid Pro defines 100% as a consistent editor sheet width rather
        # than one source-image pixel per device pixel.  This keeps mixed-DPI
        # manga sources readable and vertically scrollable at the default zoom.
        self._fit_page = False
        self._logical_zoom_percent = 100
        self._reference_sheet_width = 760.0
        self._theme = "dark"
        self._compare_ratio = 0.5
        self._base_item: QtWidgets.QGraphicsItem | None = None
        self._compare_item: _SplitCompareItem | None = None
        self._image_rect = QtCore.QRectF()
        self._overlay_shapes: tuple[OverlayShape, ...] = ()
        self._raster_overlays: dict[str, RasterOverlaySource] = {}
        self._raster_pixmaps: dict[str, QtGui.QPixmap] = {}
        self._overlay_availability = {
            overlay_id: OverlayAvailability(
                overlay_id,
                False,
                "Open a page with projected overlay evidence.",
            )
            for overlay_id in OVERLAY_IDS
        }
        self._overlay_enabled = {overlay_id: False for overlay_id in OVERLAY_IDS}
        self._overlay_items: dict[str, list[QtWidgets.QGraphicsItem]] = {
            overlay_id: [] for overlay_id in OVERLAY_IDS
        }
        self._draft_geometry: tuple[int, int, int, int] | None = None
        self._draft_geometry_parent_id = ""
        self._draft_geometry_item: QtWidgets.QGraphicsRectItem | None = None
        self._workflow_area_draft: tuple[int, int, int, int] | None = None
        self._workflow_area_role = ""
        self._workflow_area_item: QtWidgets.QGraphicsRectItem | None = None
        self._workflow_area_label_item: QtWidgets.QGraphicsTextItem | None = None
        self._split_parent_draft: tuple[
            tuple[int, int, int, int],
            tuple[int, int, int, int],
        ] | None = None
        self._split_parent_source_id = ""
        self._split_parent_items: list[QtWidgets.QGraphicsItem] = []
        self._merge_parent_draft: tuple[
            tuple[int, int, int, int],
            tuple[int, int, int, int],
            tuple[int, int, int, int],
        ] | None = None
        self._merge_parent_source_ids: tuple[str, str] = ("", "")
        self._merge_parent_items: list[QtWidgets.QGraphicsItem] = []
        self._missing_item: QtWidgets.QGraphicsTextItem | None = None
        self.set_theme("dark")

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def artifacts(self) -> CanvasArtifactSet:
        return self._artifacts

    @property
    def enabled_overlays(self) -> tuple[str, ...]:
        return tuple(key for key in OVERLAY_IDS if self._overlay_enabled[key])

    @property
    def draft_geometry(self) -> tuple[int, int, int, int] | None:
        return self._draft_geometry

    @property
    def workflow_area_draft(self) -> tuple[int, int, int, int] | None:
        return self._workflow_area_draft

    @property
    def split_parent_draft(
        self,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
        return self._split_parent_draft

    @property
    def merge_parent_draft(
        self,
    ) -> tuple[
        tuple[int, int, int, int],
        tuple[int, int, int, int],
        tuple[int, int, int, int],
    ] | None:
        return self._merge_parent_draft

    def set_theme(self, theme: str) -> None:
        if theme not in {"dark", "light"}:
            raise ValueError(f"unsupported theme: {theme!r}")
        if self._theme != theme:
            self._raster_pixmaps.clear()
        self._theme = theme
        self.setBackgroundBrush(
            QtGui.QBrush(QtGui.QColor(theme_token(theme, "surface-canvas")))
        )
        if self._artifacts.page_id or self._base_item is not None:
            self._render_mode()
        else:
            self._rebuild_overlays()

    def set_artifacts(self, artifacts: CanvasArtifactSet) -> None:
        if not isinstance(artifacts, CanvasArtifactSet):
            raise TypeError("artifacts must be CanvasArtifactSet")
        if artifacts.page_id != self._artifacts.page_id:
            self._overlay_shapes = ()
            self._raster_overlays.clear()
            self._raster_pixmaps.clear()
            self._draft_geometry = None
            self._draft_geometry_parent_id = ""
            self._draft_geometry_item = None
            self._workflow_area_draft = None
            self._workflow_area_role = ""
            self._workflow_area_item = None
            self._workflow_area_label_item = None
            self._split_parent_draft = None
            self._split_parent_source_id = ""
            self._split_parent_items = []
            self._merge_parent_draft = None
            self._merge_parent_source_ids = ("", "")
            self._merge_parent_items = []
        self._artifacts = artifacts
        self._render_mode()

    def clear_page(self) -> None:
        """Release page-owned pixels and overlays at a project/page boundary."""

        self._artifacts = CanvasArtifactSet(page_id="")
        self._overlay_shapes = ()
        self._raster_overlays.clear()
        self._raster_pixmaps.clear()
        self._draft_geometry = None
        self._draft_geometry_parent_id = ""
        self._draft_geometry_item = None
        self._workflow_area_draft = None
        self._workflow_area_role = ""
        self._workflow_area_item = None
        self._workflow_area_label_item = None
        self._split_parent_draft = None
        self._split_parent_source_id = ""
        self._split_parent_items = []
        self._merge_parent_draft = None
        self._merge_parent_source_ids = ("", "")
        self._merge_parent_items = []
        self._overlay_enabled = {overlay_id: False for overlay_id in OVERLAY_IDS}
        scene = self.scene()
        self._base_item = None
        self._compare_item = None
        self._overlay_items = {overlay_id: [] for overlay_id in OVERLAY_IDS}
        self._missing_item = None
        # QGraphicsScene owns its items, while PySide also keeps wrappers for
        # every item referenced above.  Release those wrappers before the
        # synchronous C++ clear so cumulative editor refreshes cannot block on
        # deleting scene-owned objects that Python still retains.
        scene.clear()
        self._show_missing("No page selected")
        scene.setSceneRect(self._image_rect.adjusted(-32.0, -32.0, 32.0, 32.0))
        self._update_accessible_description()

    def set_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in CANVAS_VIEW_IDS:
            raise ValueError(f"unsupported canvas mode: {mode!r}")
        if self._mode == normalized:
            return
        self._mode = normalized
        self._held_mode = None
        self._render_mode()
        self.mode_changed.emit(normalized)

    def hold_original(self, active: bool) -> None:
        if active:
            if self._held_mode is None:
                self._held_mode = self._mode
                self._mode = "original"
                self._render_mode()
        elif self._held_mode is not None:
            self._mode = self._held_mode
            self._held_mode = None
            self._render_mode()

    def set_compare_ratio(self, ratio: float) -> None:
        self._compare_ratio = max(0.05, min(0.95, float(ratio)))
        if self._compare_item is not None:
            self._compare_item.set_ratio(self._compare_ratio)

    def set_overlay_shapes(self, shapes: Iterable[OverlayShape]) -> None:
        stable = tuple(shapes)
        if any(not isinstance(item, OverlayShape) for item in stable):
            raise TypeError("overlay shapes must be OverlayShape values")
        self._overlay_shapes = stable
        self._rebuild_overlays()
        if any(item.selected for item in stable):
            QtCore.QTimer.singleShot(0, self.focus_selected_overlay)

    def focus_selected_overlay(self) -> None:
        """Bring the selected parent into view without changing canvas zoom."""

        selected = next(
            (
                shape
                for overlay_id in ("renderBox", "parentBounds")
                for shape in self._overlay_shapes
                if shape.selected
                and shape.overlay_id == overlay_id
                and shape.kind == "rect"
            ),
            None,
        )
        if selected is None:
            return
        x, y, width, height = selected.points
        rect = QtCore.QRectF(x, y, width, height).adjusted(
            -24.0,
            -24.0,
            24.0,
            24.0,
        )
        self.ensureVisible(rect, 24, 24)

    def set_raster_overlays(self, sources: Iterable[RasterOverlaySource]) -> None:
        stable = tuple(sources)
        if any(not isinstance(item, RasterOverlaySource) for item in stable):
            raise TypeError("raster overlays must be RasterOverlaySource values")
        if len({item.overlay_id for item in stable}) != len(stable):
            raise ValueError("raster overlay identities must be unique")
        new_sources = {item.overlay_id: item for item in stable}
        self._raster_pixmaps = {
            overlay_id: pixmap
            for overlay_id, pixmap in self._raster_pixmaps.items()
            if new_sources.get(overlay_id) == self._raster_overlays.get(overlay_id)
            and self._overlay_enabled[overlay_id]
        }
        self._raster_overlays = new_sources
        self._rebuild_overlays()

    def set_overlay_availability(
        self,
        values: Iterable[OverlayAvailability],
    ) -> None:
        stable = tuple(values)
        if any(not isinstance(item, OverlayAvailability) for item in stable):
            raise TypeError("overlay availability must contain typed values")
        if tuple(item.overlay_id for item in stable) != OVERLAY_IDS:
            raise ValueError("overlay availability must exactly match the UI contract")
        self._overlay_availability = {item.overlay_id: item for item in stable}
        for item in stable:
            if not item.available:
                self._overlay_enabled[item.overlay_id] = False
                self._raster_pixmaps.pop(item.overlay_id, None)
        self._rebuild_overlays()

    def set_overlay_enabled(self, overlay_id: str, enabled: bool) -> None:
        if overlay_id not in OVERLAY_IDS:
            raise ValueError(f"unsupported overlay: {overlay_id!r}")
        value = bool(enabled) and self._overlay_availability[overlay_id].available
        if self._overlay_enabled[overlay_id] == value:
            return
        self._overlay_enabled[overlay_id] = value
        if not value:
            self._raster_pixmaps.pop(overlay_id, None)
        self._rebuild_overlays()
        self._update_accessible_description()
        self.overlay_changed.emit(overlay_id, value)

    def set_draft_geometry(
        self,
        bbox: tuple[int, int, int, int] | None,
        *,
        parent_id: str = "",
    ) -> None:
        """Show one passive page-coordinate draft outline without mutating state."""

        if bbox is None:
            stable = None
            stable_parent_id = ""
        else:
            if (
                not isinstance(bbox, tuple)
                or len(bbox) != 4
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in bbox
                )
                or bbox[0] < 0
                or bbox[1] < 0
                or bbox[2] <= 0
                or bbox[3] <= 0
            ):
                raise ValueError(
                    "draft geometry must be a non-negative integer x/y/w/h bbox"
                )
            stable = bbox
            stable_parent_id = str(parent_id or "").strip()
        if (
            self._draft_geometry == stable
            and self._draft_geometry_parent_id == stable_parent_id
        ):
            return
        self._draft_geometry = stable
        self._draft_geometry_parent_id = stable_parent_id
        if stable is not None:
            self._workflow_area_draft = None
            self._workflow_area_role = ""
            self._split_parent_draft = None
            self._split_parent_source_id = ""
            self._merge_parent_draft = None
            self._merge_parent_source_ids = ("", "")
        self._rebuild_overlays()
        self._update_accessible_description()

    def set_workflow_area_draft(
        self,
        bbox: tuple[int, int, int, int] | None,
        *,
        role: str = "",
    ) -> None:
        """Show one passive Add Parent workflow-area draft on the page."""

        stable_role = str(role or "").strip()
        if stable_role not in {"", "speech", "caption"}:
            raise ValueError("workflow-area role must be speech, caption, or empty")
        if bbox is None:
            stable = None
            stable_role = ""
        else:
            if (
                not isinstance(bbox, tuple)
                or len(bbox) != 4
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in bbox
                )
                or bbox[0] < 0
                or bbox[1] < 0
                or bbox[2] <= 0
                or bbox[3] <= 0
            ):
                raise ValueError(
                    "workflow area must be a non-negative integer x/y/w/h bbox"
                )
            stable = bbox
        if (
            self._workflow_area_draft == stable
            and self._workflow_area_role == stable_role
        ):
            return
        self._workflow_area_draft = stable
        self._workflow_area_role = stable_role
        if stable is not None:
            self._draft_geometry = None
            self._draft_geometry_parent_id = ""
            self._split_parent_draft = None
            self._split_parent_source_id = ""
            self._merge_parent_draft = None
            self._merge_parent_source_ids = ("", "")
        self._rebuild_overlays()
        self._update_accessible_description()

    def set_split_parent_draft(
        self,
        child_bboxes: tuple[
            tuple[int, int, int, int],
            tuple[int, int, int, int],
        ] | None,
        *,
        source_parent_id: str = "",
    ) -> None:
        """Show a passive two-child partition without changing project state."""

        stable_source = str(source_parent_id or "").strip()
        if child_bboxes is None:
            stable = None
            stable_source = ""
        else:
            if not isinstance(child_bboxes, tuple) or len(child_bboxes) != 2:
                raise TypeError("split child_bboxes must contain exactly two bboxes")
            normalized: list[tuple[int, int, int, int]] = []
            for bbox in child_bboxes:
                if (
                    not isinstance(bbox, tuple)
                    or len(bbox) != 4
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in bbox
                    )
                    or bbox[0] < 0
                    or bbox[1] < 0
                    or bbox[2] <= 0
                    or bbox[3] <= 0
                ):
                    raise ValueError("split child bbox must be valid page coordinates")
                normalized.append(bbox)
            stable = (normalized[0], normalized[1])
            if not stable_source:
                raise ValueError("source_parent_id is required for a split draft")
        if (
            self._split_parent_draft == stable
            and self._split_parent_source_id == stable_source
        ):
            return
        self._split_parent_draft = stable
        self._split_parent_source_id = stable_source
        if stable is not None:
            self._draft_geometry = None
            self._draft_geometry_parent_id = ""
            self._workflow_area_draft = None
            self._workflow_area_role = ""
            self._merge_parent_draft = None
            self._merge_parent_source_ids = ("", "")
        self._rebuild_overlays()
        self._update_accessible_description()

    def set_merge_parent_draft(
        self,
        source_bboxes: tuple[
            tuple[int, int, int, int],
            tuple[int, int, int, int],
        ] | None,
        *,
        merged_bbox: tuple[int, int, int, int] | None = None,
        source_parent_ids: tuple[str, str] = ("", ""),
    ) -> None:
        """Show a passive two-source merge draft without changing project state."""

        if source_bboxes is None:
            stable = None
            stable_ids = ("", "")
        else:
            if (
                not isinstance(source_bboxes, tuple)
                or len(source_bboxes) != 2
                or merged_bbox is None
            ):
                raise TypeError("merge draft requires two source bboxes and one merged bbox")
            boxes = (*source_bboxes, merged_bbox)
            if any(
                not isinstance(bbox, tuple)
                or len(bbox) != 4
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in bbox
                )
                or bbox[0] < 0
                or bbox[1] < 0
                or bbox[2] <= 0
                or bbox[3] <= 0
                for bbox in boxes
            ):
                raise ValueError("merge draft bboxes must be valid page coordinates")
            stable_ids = tuple(str(value or "").strip() for value in source_parent_ids)
            if len(stable_ids) != 2 or not all(stable_ids) or stable_ids[0] == stable_ids[1]:
                raise ValueError("merge draft requires two distinct source parent identities")
            stable = (source_bboxes[0], source_bboxes[1], merged_bbox)
        if self._merge_parent_draft == stable and self._merge_parent_source_ids == stable_ids:
            return
        self._merge_parent_draft = stable
        self._merge_parent_source_ids = stable_ids
        if stable is not None:
            self._draft_geometry = None
            self._draft_geometry_parent_id = ""
            self._workflow_area_draft = None
            self._workflow_area_role = ""
            self._split_parent_draft = None
            self._split_parent_source_id = ""
        self._rebuild_overlays()
        self._update_accessible_description()

    def set_fit_page(self, fit: bool = True) -> None:
        self._fit_page = bool(fit)
        if self._fit_page:
            self.fit_page()

    def fit_page(self) -> None:
        if self._image_rect.isEmpty():
            return
        self._fit_page = True
        self.resetTransform()
        self.fitInView(self._image_rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self._emit_zoom()

    def set_zoom_percent(self, percent: int) -> None:
        value = max(10, min(800, int(percent)))
        self._fit_page = False
        self._logical_zoom_percent = value
        self.resetTransform()
        factor = self._logical_zoom_factor(value)
        self.scale(factor, factor)
        self._emit_zoom()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            self._fit_page = False
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            current = self.transform().m11()
            base = self._logical_zoom_factor(100)
            target = max(base * 0.1, min(base * 8.0, current * factor))
            self.scale(target / current, target / current)
            self._logical_zoom_percent = int(round(target / base * 100))
            self._emit_zoom()
            event.accept()
            return
        super().wheelEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._fit_page:
            self.fit_page()

    def _load_pixmap(self, path: str | None) -> QtGui.QPixmap | None:
        if not path or not os.path.isfile(path):
            return None
        pixmap = QtGui.QPixmap(path)
        return pixmap if not pixmap.isNull() else None

    def _render_mode(self) -> None:
        scene = self.scene()
        self._draft_geometry_item = None
        self._workflow_area_item = None
        self._workflow_area_label_item = None
        self._split_parent_items = []
        self._merge_parent_items = []
        self._base_item = None
        self._compare_item = None
        self._missing_item = None
        self._overlay_items = {overlay_id: [] for overlay_id in OVERLAY_IDS}
        # Drop every Python-held graphics-item wrapper before QGraphicsScene
        # destroys the corresponding C++ objects.  This ordering is important
        # after long runs that repeatedly rebuild Page Editor shells.
        scene.clear()

        if self._mode == "compare":
            original = self._load_pixmap(self._artifacts.original_path)
            final = self._load_pixmap(self._artifacts.final_path)
            if original is not None and final is not None:
                item = _SplitCompareItem(
                    original,
                    final,
                    ratio=self._compare_ratio,
                    theme=self._theme,
                )
                scene.addItem(item)
                self._base_item = item
                self._compare_item = item
                self._image_rect = item.boundingRect()
                self._add_corner_label("Original", left=True)
                self._add_corner_label("Final", left=False)
            else:
                missing = "Original" if original is None else "Final"
                self._show_missing(f"{missing} image unavailable — Compare is disabled")
        else:
            path = self._artifacts.path_for(self._mode)
            pixmap = self._load_pixmap(path)
            if pixmap is None:
                labels = {
                    "original": "Original image unavailable",
                    "cleaned": "Cleaned image unavailable — Missing cleaned base",
                    "final": "Final image unavailable — Render this page first",
                }
                self._show_missing(labels[self._mode])
            else:
                item = scene.addPixmap(pixmap)
                item.setTransformationMode(
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
                self._base_item = item
                self._image_rect = item.boundingRect()
                self._add_corner_label(self._mode.title(), left=True)

        scene.setSceneRect(self._image_rect.adjusted(-32.0, -32.0, 32.0, 32.0))
        self._rebuild_overlays()
        self._update_accessible_description()
        if self._fit_page:
            QtCore.QTimer.singleShot(0, self.fit_page)
        else:
            QtCore.QTimer.singleShot(
                0,
                lambda: self.set_zoom_percent(self._logical_zoom_percent),
            )

    def _show_missing(self, text: str) -> None:
        self._image_rect = QtCore.QRectF(0.0, 0.0, 720.0, 480.0)
        item = self.scene().addText(text)
        item.setDefaultTextColor(
            QtGui.QColor(theme_token(self._theme, "content-muted"))
        )
        font = item.font()
        font.setPointSize(13)
        font.setBold(True)
        item.setFont(font)
        bounds = item.boundingRect()
        item.setPos(
            (self._image_rect.width() - bounds.width()) / 2,
            (self._image_rect.height() - bounds.height()) / 2,
        )
        self._missing_item = item

    def _add_corner_label(self, text: str, *, left: bool) -> None:
        label = self.scene().addText(text)
        label.setDefaultTextColor(
            QtGui.QColor(theme_token(self._theme, "content-primary"))
        )
        label.setZValue(20.0)
        background = QtWidgets.QGraphicsRectItem(label.boundingRect().adjusted(-6, -3, 6, 3))
        background.setBrush(
            QtGui.QBrush(QtGui.QColor(theme_token(self._theme, "surface-panel-raised")))
        )
        background.setPen(
            QtGui.QPen(QtGui.QColor(theme_token(self._theme, "border-default")))
        )
        background.setZValue(19.0)
        y = 12.0
        x = 12.0 if left else self._image_rect.width() - label.boundingRect().width() - 12.0
        label.setPos(x, y)
        background.setPos(x, y)
        self.scene().addItem(background)

    def _rebuild_overlays(self) -> None:
        scene = self.scene()
        draft_item = self._draft_geometry_item
        if draft_item is not None and draft_item.scene() is scene:
            scene.removeItem(draft_item)
        self._draft_geometry_item = None
        workflow_item = self._workflow_area_item
        if workflow_item is not None and workflow_item.scene() is scene:
            scene.removeItem(workflow_item)
        workflow_label = self._workflow_area_label_item
        if workflow_label is not None and workflow_label.scene() is scene:
            scene.removeItem(workflow_label)
        self._workflow_area_item = None
        self._workflow_area_label_item = None
        for item in self._split_parent_items:
            if item.scene() is scene:
                scene.removeItem(item)
        self._split_parent_items = []
        for item in self._merge_parent_items:
            if item.scene() is scene:
                scene.removeItem(item)
        self._merge_parent_items = []
        for items in self._overlay_items.values():
            for item in items:
                if item.scene() is scene:
                    scene.removeItem(item)
        self._overlay_items = {overlay_id: [] for overlay_id in OVERLAY_IDS}
        if self._image_rect.isEmpty():
            return
        enabled_overlay_ids = {
            overlay_id
            for overlay_id, enabled in self._overlay_enabled.items()
            if enabled
        }
        occupied_label_rects: list[QtCore.QRectF] = []
        for shape in self._overlay_shapes:
            item = self._graphics_item(shape)
            item.setZValue(30.0 if shape.selected else 25.0)
            item.setVisible(self._overlay_enabled[shape.overlay_id])
            item.setData(0, shape.shape_id)
            item.setToolTip(shape.label)
            scene.addItem(item)
            self._overlay_items[shape.overlay_id].append(item)
            display_label = (
                "Effective render box"
                if shape.selected and shape.overlay_id == "renderBox"
                else shape.label
            )
            show_label = bool(display_label) and (
                len(enabled_overlay_ids) <= 1
                or shape.overlay_id in {"renderBox", "proof"}
            )
            if show_label:
                label = scene.addText(display_label)
                label.setDefaultTextColor(
                    QtGui.QColor(
                        overlay_token(self._theme, self._overlay_token_role(shape.overlay_id))
                    )
                )
                label.setZValue(item.zValue() + 1.0)
                bounds = item.boundingRect()
                available = self._image_rect.adjusted(2.0, 2.0, -2.0, -2.0)
                natural = label.boundingRect()
                max_width = max(1.0, min(260.0, available.width()))
                if natural.width() > max_width:
                    label.setTextWidth(max_width)
                label_bounds = label.boundingRect()
                anchor = (
                    bounds.bottomLeft()
                    if shape.overlay_id == "proof"
                    else bounds.topLeft()
                )
                preferred_top = (
                    anchor.y() + 4.0
                    if shape.overlay_id == "proof"
                    else anchor.y() - label_bounds.height() - 4.0
                )
                min_top = available.top() - label_bounds.top()
                max_top = available.bottom() - label_bounds.bottom()
                base_top = min(max(preferred_top, min_top), max_top)
                step = label_bounds.height() + 4.0
                candidate_tops = [base_top]
                for lane in range(1, len(occupied_label_rects) + 3):
                    candidate_tops.extend(
                        (
                            min(max(base_top + lane * step, min_top), max_top),
                            min(max(base_top - lane * step, min_top), max_top),
                        )
                    )
                min_left = available.left() - label_bounds.left()
                max_left = available.right() - label_bounds.right()
                left = min(max(anchor.x(), min_left), max_left)
                chosen_rect: QtCore.QRectF | None = None
                for top in candidate_tops:
                    candidate = label_bounds.translated(left, top)
                    padded = candidate.adjusted(-2.0, -2.0, 2.0, 2.0)
                    if not any(
                        padded.intersects(existing)
                        for existing in occupied_label_rects
                    ):
                        label.setPos(left, top)
                        chosen_rect = candidate
                        break
                if chosen_rect is None:
                    label.setPos(left, base_top)
                    chosen_rect = label_bounds.translated(left, base_top)
                label.setVisible(self._overlay_enabled[shape.overlay_id])
                if label.isVisible():
                    occupied_label_rects.append(chosen_rect)
                self._overlay_items[shape.overlay_id].append(label)
            if (
                shape.selected
                and shape.overlay_id == "renderBox"
                and isinstance(item, QtWidgets.QGraphicsRectItem)
            ):
                selection = QtGui.QColor(
                    overlay_token(self._theme, "selection")
                )
                handle_pen = QtGui.QPen(selection)
                handle_pen.setCosmetic(True)
                handle_pen.setWidth(1)
                handle_brush = QtGui.QBrush(selection)
                bounds = item.rect()
                for index, point in enumerate(
                    (
                        bounds.topLeft(),
                        bounds.topRight(),
                        bounds.bottomLeft(),
                        bounds.bottomRight(),
                    ),
                    1,
                ):
                    handle = QtWidgets.QGraphicsRectItem(
                        QtCore.QRectF(
                            point.x() - 4.0,
                            point.y() - 4.0,
                            8.0,
                            8.0,
                        )
                    )
                    handle.setPen(handle_pen)
                    handle.setBrush(handle_brush)
                    handle.setZValue(item.zValue() + 2.0)
                    handle.setVisible(item.isVisible())
                    handle.setData(0, f"selected-render-box-handle-{index}")
                    scene.addItem(handle)
                    self._overlay_items[shape.overlay_id].append(handle)
        for overlay_id, source in self._raster_overlays.items():
            if not self._overlay_enabled[overlay_id]:
                continue
            pixmap = self._raster_pixmap(source)
            if pixmap is None:
                continue
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            item.setZValue(24.0)
            item.setOpacity(0.42)
            item.setData(0, f"raster:{overlay_id}:{source.asset_sha256}")
            item.setToolTip(source.label)
            scene.addItem(item)
            self._overlay_items[overlay_id].append(item)
        if self._draft_geometry is not None:
            x, y, width, height = self._draft_geometry
            draft = QtWidgets.QGraphicsRectItem(
                QtCore.QRectF(float(x), float(y), float(width), float(height))
            )
            color = QtGui.QColor(overlay_token(self._theme, "selection"))
            pen = QtGui.QPen(color)
            pen.setCosmetic(True)
            pen.setWidth(3)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            draft.setPen(pen)
            fill = QtGui.QColor(color)
            fill.setAlpha(28)
            draft.setBrush(QtGui.QBrush(fill))
            draft.setZValue(40.0)
            draft.setData(0, "geometry-draft")
            draft.setToolTip("Unapplied selected-parent geometry draft")
            scene.addItem(draft)
            self._draft_geometry_item = draft
        if self._workflow_area_draft is not None:
            x, y, width, height = self._workflow_area_draft
            workflow = QtWidgets.QGraphicsRectItem(
                QtCore.QRectF(float(x), float(y), float(width), float(height))
            )
            color = QtGui.QColor(overlay_token(self._theme, "selection"))
            pen = QtGui.QPen(color)
            pen.setCosmetic(True)
            pen.setWidth(3)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            workflow.setPen(pen)
            fill = QtGui.QColor(color)
            fill.setAlpha(28)
            workflow.setBrush(QtGui.QBrush(fill))
            workflow.setZValue(41.0)
            workflow.setData(0, "workflow-area-draft")
            role_label = {
                "speech": "Dialogue",
                "caption": "Caption",
            }.get(self._workflow_area_role, "Add Parent")
            tooltip = f"Unapplied {role_label} workflow area"
            workflow.setToolTip(tooltip)
            scene.addItem(workflow)
            label = scene.addText(f"{role_label} workflow area draft")
            label.setDefaultTextColor(color)
            label.setZValue(42.0)
            label.setData(0, "workflow-area-draft-label")
            label.setToolTip(tooltip)
            label_height = label.boundingRect().height()
            label_y = max(0.0, float(y) - label_height)
            label.setPos(float(x), label_y)
            self._workflow_area_item = workflow
            self._workflow_area_label_item = label
        if self._split_parent_draft is not None:
            color = QtGui.QColor(overlay_token(self._theme, "selection"))
            child_names = ("Child 1", "Child 2")
            for index, bbox in enumerate(self._split_parent_draft):
                x, y, width, height = bbox
                rect = QtWidgets.QGraphicsRectItem(
                    QtCore.QRectF(float(x), float(y), float(width), float(height))
                )
                pen = QtGui.QPen(color)
                pen.setCosmetic(True)
                pen.setWidth(3)
                pen.setStyle(QtCore.Qt.PenStyle.DashLine)
                rect.setPen(pen)
                fill = QtGui.QColor(color)
                fill.setAlpha(22 if index == 0 else 38)
                rect.setBrush(QtGui.QBrush(fill))
                rect.setZValue(43.0)
                rect.setData(0, f"split-parent-draft-{index + 1}")
                rect.setToolTip(
                    f"Unapplied {child_names[index]} Split Parent partition"
                )
                scene.addItem(rect)
                label = scene.addText(child_names[index])
                label.setDefaultTextColor(color)
                label.setZValue(44.0)
                label.setData(0, f"split-parent-draft-label-{index + 1}")
                label_y = float(y) + 4.0
                first_bbox, second_bbox = self._split_parent_draft
                vertical_partition = (
                    first_bbox[1] == second_bbox[1]
                    and first_bbox[3] == second_bbox[3]
                    and first_bbox[0] + first_bbox[2] == second_bbox[0]
                )
                if vertical_partition and index == 1:
                    label_y += label.boundingRect().height() + 4.0
                label.setPos(float(x) + 4.0, label_y)
                scene.addItem(label)
                self._split_parent_items.extend((rect, label))
        if self._merge_parent_draft is not None:
            source_color = QtGui.QColor(overlay_token(self._theme, "selection"))
            merged_color = QtGui.QColor(overlay_token(self._theme, "source"))
            for index, bbox in enumerate(self._merge_parent_draft[:2]):
                x, y, width, height = bbox
                rect = QtWidgets.QGraphicsRectItem(
                    QtCore.QRectF(float(x), float(y), float(width), float(height))
                )
                pen = QtGui.QPen(source_color)
                pen.setCosmetic(True)
                pen.setWidth(2)
                pen.setStyle(QtCore.Qt.PenStyle.DotLine)
                rect.setPen(pen)
                rect.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                rect.setZValue(43.0)
                rect.setData(0, f"merge-parent-source-{index + 1}")
                rect.setToolTip(f"Pipeline source {index + 1} for unapplied Merge Parent")
                scene.addItem(rect)
                self._merge_parent_items.append(rect)
            x, y, width, height = self._merge_parent_draft[2]
            merged = QtWidgets.QGraphicsRectItem(
                QtCore.QRectF(float(x), float(y), float(width), float(height))
            )
            pen = QtGui.QPen(merged_color)
            pen.setCosmetic(True)
            pen.setWidth(3)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            merged.setPen(pen)
            fill = QtGui.QColor(merged_color)
            fill.setAlpha(30)
            merged.setBrush(QtGui.QBrush(fill))
            merged.setZValue(44.0)
            merged.setData(0, "merge-parent-draft")
            merged.setToolTip("Unapplied enclosing Merge Parent range")
            scene.addItem(merged)
            label = scene.addText("Merged range draft")
            application = QtWidgets.QApplication.instance()
            base_point_size = (
                application.property("yomiframeBasePointSize")
                if application is not None
                else None
            )
            if (
                isinstance(base_point_size, (int, float))
                and not isinstance(base_point_size, bool)
                and float(base_point_size) > 0.0
            ):
                label_font = label.font()
                label_font.setPointSizeF(float(base_point_size))
                label.setFont(label_font)
            label.setDefaultTextColor(merged_color)
            label.setZValue(45.0)
            label.setData(0, "merge-parent-draft-label")
            label.setToolTip("Unapplied enclosing Merge Parent range")
            label.setPos(
                float(x) + 4.0,
                max(
                    float(y) + 4.0,
                    float(y + height) - label.boundingRect().height() - 4.0,
                ),
            )
            scene.addItem(label)
            self._merge_parent_items.extend((merged, label))

    def _raster_pixmap(self, source: RasterOverlaySource) -> QtGui.QPixmap | None:
        cached = self._raster_pixmaps.get(source.overlay_id)
        if cached is not None:
            return cached
        reader = QtGui.QImageReader(source.asset_path)
        reader.setAutoTransform(False)
        image = reader.read()
        if image.isNull() or (image.width(), image.height()) != source.canvas_size:
            return None
        alpha = image.convertToFormat(QtGui.QImage.Format.Format_Grayscale8)
        tinted = QtGui.QImage(
            image.width(),
            image.height(),
            QtGui.QImage.Format.Format_ARGB32_Premultiplied,
        )
        tinted.fill(
            QtGui.QColor(
                overlay_token(
                    self._theme,
                    self._overlay_token_role(source.overlay_id),
                )
            )
        )
        tinted.setAlphaChannel(alpha)
        pixmap = QtGui.QPixmap.fromImage(tinted)
        if pixmap.isNull():
            return None
        self._raster_pixmaps[source.overlay_id] = pixmap
        return pixmap

    def _graphics_item(self, shape: OverlayShape) -> QtWidgets.QGraphicsItem:
        token_role = (
            "selection"
            if shape.selected and shape.overlay_id == "renderBox"
            else self._overlay_token_role(shape.overlay_id)
        )
        color = QtGui.QColor(overlay_token(self._theme, token_role))
        pen = QtGui.QPen(color)
        pen.setCosmetic(True)
        pen.setWidth(3 if shape.selected else 2)
        if shape.kind == "rect":
            x, y, width, height = shape.points
            item: QtWidgets.QGraphicsItem = QtWidgets.QGraphicsRectItem(
                QtCore.QRectF(x, y, width, height)
            )
        elif shape.kind == "line":
            x1, y1, x2, y2 = shape.points
            item = QtWidgets.QGraphicsLineItem(x1, y1, x2, y2)
        else:
            polygon = QtGui.QPolygonF(
                [
                    QtCore.QPointF(shape.points[index], shape.points[index + 1])
                    for index in range(0, len(shape.points), 2)
                ]
            )
            item = QtWidgets.QGraphicsPolygonItem(polygon)
        if isinstance(
            item,
            (
                QtWidgets.QGraphicsRectItem,
                QtWidgets.QGraphicsLineItem,
                QtWidgets.QGraphicsPolygonItem,
            ),
        ):
            item.setPen(pen)
        if (
            shape.selected
            and shape.overlay_id == "renderBox"
            and isinstance(item, QtWidgets.QGraphicsRectItem)
        ):
            fill = QtGui.QColor(color)
            fill.setAlpha(20)
            item.setBrush(QtGui.QBrush(fill))
        return item

    @staticmethod
    def _overlay_token_role(overlay_id: str) -> str:
        return {
            "parentBounds": "parent",
            "renderBox": "render",
            "sourceFootprint": "source",
            "baseline": "baseline",
            "cleanupMask": "cleanup",
            "protectedRegions": "protected",
            "proof": "proof",
        }[overlay_id]

    def _emit_zoom(self) -> None:
        base = self._logical_zoom_factor(100)
        value = int(round(self.transform().m11() / base * 100))
        self._logical_zoom_percent = max(10, min(800, value))
        self.zoom_changed.emit(self._logical_zoom_percent)

    def _logical_zoom_factor(self, percent: int) -> float:
        width = self._image_rect.width()
        base = self._reference_sheet_width / width if width > 0 else 1.0
        return base * max(10, min(800, int(percent))) / 100.0

    def _update_accessible_description(self) -> None:
        overlays = ", ".join(self.enabled_overlays) or "no overlays"
        availability = "artifact available" if self._missing_item is None else "artifact missing"
        if self._draft_geometry is None:
            draft = ""
        else:
            x, y, width, height = self._draft_geometry
            parent = (
                f" for parent {self._draft_geometry_parent_id}"
                if self._draft_geometry_parent_id
                else ""
            )
            draft = (
                f"; geometry draft{parent}: x {x}, y {y}, "
                f"width {width}, height {height}"
            )
        if self._workflow_area_draft is None:
            workflow = ""
        else:
            x, y, width, height = self._workflow_area_draft
            role = {
                "speech": "Dialogue",
                "caption": "Caption",
            }.get(self._workflow_area_role, "Add Parent")
            workflow = (
                f"; {role} workflow area draft: x {x}, y {y}, "
                f"width {width}, height {height}"
            )
        if self._split_parent_draft is None:
            split = ""
        else:
            first, second = self._split_parent_draft
            split = (
                "; unapplied Split Parent draft for the selected user parent: "
                f"child 1 {first}; child 2 {second}"
            )
        if self._merge_parent_draft is None:
            merge = ""
        else:
            first, second, merged = self._merge_parent_draft
            merge = (
                "; unapplied Merge Parent draft: "
                f"pipeline source 1 {first}; pipeline source 2 {second}; "
                f"enclosing merged range {merged}"
            )
        self.setAccessibleDescription(
            f"{self._mode.title()} mode; {availability}; showing "
            f"{overlays}{draft}{workflow}{split}{merge}."
        )
