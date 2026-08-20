# -*- coding: utf-8 -*-
"""Dense model delegates shared by the native Hub, Workspace, and Editor."""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.design_system.icons import hybrid_icon
from app.ui.design_system.tokens import theme_token
from app.ui.viewmodels.project_model import PageRole, ProjectRole


def _theme() -> str:
    application = QtWidgets.QApplication.instance()
    value = str(application.property("yomiframeTheme") or "dark") if application else "dark"
    return value if value in {"dark", "light"} else "dark"


def _tone_color(tone: str) -> QtGui.QColor:
    role = {
        "ready": "status-success",
        "editing": "accent-primary",
        "warning": "status-warning",
        "error": "status-danger",
        "queued": "content-muted",
        "muted": "content-muted",
        "info": "accent-primary",
    }.get(str(tone), "content-muted")
    return QtGui.QColor(theme_token(_theme(), role))


def _text(
    painter: QtGui.QPainter,
    rect: QtCore.QRectF,
    value: object,
    *,
    color: QtGui.QColor,
    weight: QtGui.QFont.Weight = QtGui.QFont.Weight.Normal,
    size_delta: float = 0.0,
    alignment: QtCore.Qt.AlignmentFlag = QtCore.Qt.AlignmentFlag.AlignLeft,
) -> None:
    painter.save()
    font = painter.font()
    font.setWeight(weight)
    if size_delta:
        font.setPointSizeF(max(1.0, font.pointSizeF() + size_delta))
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(
        rect,
        int(alignment | QtCore.Qt.AlignmentFlag.AlignVCenter),
        str(value),
    )
    painter.restore()


class ProjectCardDelegate(QtWidgets.QStyledItemDelegate):
    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtCore.QSize:
        del option, index
        # Follow the owning QListView's responsive grid.  Leave one pixel of
        # layout slack so two 1440 px cards never wrap because of rounding.
        view = self.parent()
        grid_width = (
            view.gridSize().width()
            if isinstance(view, QtWidgets.QListView)
            else 670
        )
        return QtCore.QSize(max(1, grid_width - 1), 154)

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        painter.save()
        second_column = option.rect.left() > 0
        left_inset = 7 if second_column else 1
        right_inset = 0 if second_column else 6
        rect = QtCore.QRectF(
            option.rect.adjusted(left_inset, 0, -right_inset, 0)
        )
        selected = bool(option.state & QtWidgets.QStyle.StateFlag.State_Selected)
        featured = index.row() == 0
        background = theme_token(
            _theme(),
            "surface-panel-raised" if featured or selected else "surface-panel",
        )
        painter.setBrush(QtGui.QColor(background))
        painter.setPen(
            QtGui.QColor(
                theme_token(
                    _theme(),
                    "border-strong" if featured or selected else "border-default",
                )
            )
        )
        painter.drawRoundedRect(rect, 8.0, 8.0)
        name = index.data(int(ProjectRole.NAME)) or index.data()
        language_pair = index.data(int(ProjectRole.LANGUAGE_PAIR)) or ""
        page_count = int(index.data(int(ProjectRole.PAGE_COUNT)) or 0)
        completed = int(index.data(int(ProjectRole.COMPLETED_COUNT)) or 0)
        status = index.data(int(ProjectRole.STATUS_LABEL)) or ""
        tone = index.data(int(ProjectRole.STATUS_TONE)) or "muted"
        thumbnail = str(index.data(int(ProjectRole.THUMBNAIL_PATH)) or "")
        updated = str(index.data(int(ProjectRole.UPDATED_LABEL)) or "Local project")
        primary = QtGui.QColor(theme_token(_theme(), "content-primary"))
        secondary = QtGui.QColor(theme_token(_theme(), "content-secondary"))
        muted = QtGui.QColor(theme_token(_theme(), "content-muted"))
        line_height = max(18, painter.fontMetrics().lineSpacing())
        text_left = rect.left() + 16
        if thumbnail and os.path.isfile(thumbnail):
            pixmap = QtGui.QPixmap(thumbnail)
            if not pixmap.isNull():
                thumb_rect = QtCore.QRectF(
                    rect.left(), rect.top(), 112, rect.height()
                )
                scaled = pixmap.scaled(
                    int(thumb_rect.width()),
                    int(thumb_rect.height()),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                source_x = max(0, (scaled.width() - int(thumb_rect.width())) // 2)
                source_y = max(0, (scaled.height() - int(thumb_rect.height())) // 2)
                painter.setClipRect(thumb_rect)
                painter.drawPixmap(
                    thumb_rect,
                    scaled,
                    QtCore.QRectF(
                        source_x,
                        source_y,
                        thumb_rect.width(),
                        thumb_rect.height(),
                    ),
                )
                painter.setClipping(False)
                text_left = thumb_rect.right() + 16
        first_top = rect.top() + 20
        second_top = first_top + line_height + 5
        progress_top = second_top + line_height + 13
        third_top = progress_top + 20
        _text(
            painter,
            QtCore.QRectF(text_left, first_top, rect.right() - text_left - 120, line_height),
            name,
            color=primary,
            weight=QtGui.QFont.Weight.DemiBold,
            size_delta=1.0,
        )
        _text(
            painter,
            QtCore.QRectF(text_left, second_top, rect.right() - text_left - 20, line_height),
            language_pair,
            color=secondary,
        )
        progress_rect = QtCore.QRectF(
            text_left,
            progress_top,
            max(80.0, rect.right() - text_left - 58),
            5,
        )
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(theme_token(_theme(), "surface-control")))
        painter.drawRoundedRect(progress_rect, 2.5, 2.5)
        percent = int(round((completed / page_count) * 100)) if page_count else 0
        filled = QtCore.QRectF(
            progress_rect.left(),
            progress_rect.top(),
            progress_rect.width() * percent / 100.0,
            progress_rect.height(),
        )
        painter.setBrush(QtGui.QColor(theme_token(_theme(), "accent-primary")))
        painter.drawRoundedRect(filled, 2.5, 2.5)
        _text(
            painter,
            QtCore.QRectF(progress_rect.right() + 7, progress_top - 7, 44, line_height),
            f"{percent}%",
            color=secondary,
            alignment=QtCore.Qt.AlignmentFlag.AlignRight,
            size_delta=-0.5,
        )
        count_text = f"{page_count} pages · {updated}" if page_count else "Open to inspect"
        _text(
            painter,
            QtCore.QRectF(text_left, third_top, rect.right() - text_left - 90, line_height),
            count_text,
            color=muted,
            size_delta=-0.5,
        )
        status_text = str(status)
        status_font = painter.font()
        status_font.setPointSizeF(max(1.0, status_font.pointSizeF() - 0.5))
        status_width = QtGui.QFontMetrics(status_font).horizontalAdvance(status_text) + 32
        status_rect = QtCore.QRectF(
            rect.right() - status_width - 10,
            first_top,
            status_width,
            line_height + 4,
        )
        tone_key = str(tone)
        surface_role = {
            "ready": "status-success-surface",
            "editing": "accent-primary-surface",
            "warning": "status-warning-surface",
            "error": "status-danger-surface",
        }.get(tone_key, "surface-control")
        border_role = {
            "ready": "status-success-border",
            "editing": "accent-primary-border",
            "warning": "status-warning-border",
            "error": "status-danger-border",
        }.get(tone_key, "border-default")
        painter.setPen(QtGui.QPen(QtGui.QColor(theme_token(_theme(), border_role))))
        painter.setBrush(QtGui.QColor(theme_token(_theme(), surface_role)))
        painter.drawRoundedRect(status_rect, 7.0, 7.0)
        status_icon_name = {
            "ready": "success",
            "editing": "editor",
            "warning": "warning",
            "error": "warning",
        }.get(tone_key, "status-muted")
        hybrid_icon(status_icon_name, _theme()).paint(
            painter,
            QtCore.QRect(
                int(status_rect.left() + 7),
                int(status_rect.top() + 5),
                12,
                12,
            ),
        )
        _text(
            painter,
            QtCore.QRectF(
                status_rect.left() + 22,
                status_rect.top(),
                status_rect.width() - 28,
                status_rect.height(),
            ),
            status_text,
            color=_tone_color(str(tone)),
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft,
            size_delta=-0.5,
        )
        action_text = "Open" if index.row() == 0 else "Preview only"
        action_color = QtGui.QColor(
            theme_token(
                _theme(),
                "accent-text" if index.row() == 0 else "content-disabled",
            )
        )
        action_rect = QtCore.QRectF(
            rect.right() - 86,
            third_top,
            72,
            line_height,
        )
        _text(
            painter,
            action_rect,
            action_text,
            color=action_color,
            alignment=QtCore.Qt.AlignmentFlag.AlignRight,
            size_delta=-0.5,
        )
        if index.row() == 0:
            hybrid_icon("arrow-right", _theme()).paint(
                painter,
                QtCore.QRect(
                    int(rect.right() - 12),
                    int(third_top + 3),
                    12,
                    12,
                ),
            )
        painter.restore()


class PageRailDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(
        self,
        *,
        compact: bool,
        collapsed: bool = False,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._compact = bool(compact)
        self._collapsed = bool(collapsed)

    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtCore.QSize:
        metrics = QtGui.QFontMetrics(option.font)
        if self._collapsed:
            return QtCore.QSize(70, max(58, metrics.lineSpacing() + 18))
        # Hybrid Pro's default thumbnail rail is a dense 54px row; its compact
        # list is 38px.  Keep the width bounded to the rail viewport so long
        # status labels elide instead of manufacturing a horizontal scrollbar.
        base_height = 54 if self._compact else 38
        return QtCore.QSize(
            166,
            max(base_height, metrics.lineSpacing() * 2 + 12),
        )

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        painter.save()
        rect = QtCore.QRectF(option.rect.adjusted(1, 1, -1, -1))
        selected = bool(option.state & QtWidgets.QStyle.StateFlag.State_Selected)
        if selected:
            painter.setBrush(QtGui.QColor(theme_token(_theme(), "surface-selected")))
            painter.setPen(QtGui.QColor(theme_token(_theme(), "accent-primary-border")))
            painter.drawRoundedRect(rect, 4.0, 4.0)
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor(theme_token(_theme(), "accent-primary")),
                    2.0,
                )
            )
            painter.drawLine(
                QtCore.QPointF(rect.left() + 1, rect.top() + 4),
                QtCore.QPointF(rect.left() + 1, rect.bottom() - 4),
            )
        file_name = index.data(int(PageRole.FILE_NAME)) or index.data()
        ordinal = int(index.data(int(PageRole.ORDINAL)) or 0)
        thumbnail = str(index.data(int(PageRole.THUMBNAIL_PATH)) or "")
        status_role = PageRole.STATUS_LABEL if self._compact else PageRole.WORKSPACE_STATUS_LABEL
        tone_role = PageRole.STATUS_TONE if self._compact else PageRole.WORKSPACE_STATUS_TONE
        status = index.data(int(status_role)) or ""
        tone = str(index.data(int(tone_role)) or "muted")
        primary = QtGui.QColor(theme_token(_theme(), "content-primary"))
        secondary = QtGui.QColor(theme_token(_theme(), "content-secondary"))
        muted = QtGui.QColor(theme_token(_theme(), "content-muted"))
        line_height = max(16, painter.fontMetrics().lineSpacing())
        thumb_width = 39 if self._compact else 29
        thumb_height = 49 if self._compact else 33
        x = rect.left() + 3
        if not self._collapsed:
            _text(
                painter,
                QtCore.QRectF(x, rect.top(), 18, rect.height()),
                f"{ordinal:02d}",
                color=secondary,
                size_delta=-0.5,
                alignment=QtCore.Qt.AlignmentFlag.AlignHCenter,
            )
            x += 23
        if thumbnail and os.path.isfile(thumbnail):
            pixmap = QtGui.QPixmap(thumbnail)
            if not pixmap.isNull():
                thumb_rect = QtCore.QRectF(
                    x,
                    rect.center().y() - thumb_height / 2,
                    thumb_width,
                    thumb_height,
                )
                scaled = pixmap.scaled(
                    int(thumb_rect.width()),
                    int(thumb_rect.height()),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                target = QtCore.QRectF(
                    thumb_rect.center().x() - scaled.width() / 2,
                    thumb_rect.center().y() - scaled.height() / 2,
                    scaled.width(),
                    scaled.height(),
                )
                painter.drawPixmap(target, scaled, QtCore.QRectF(scaled.rect()))
        if self._collapsed:
            painter.restore()
            return
        x += thumb_width + 6
        text_width = max(1, int(rect.right() - x - 6))
        metrics = painter.fontMetrics()
        display_name = metrics.elidedText(
            str(file_name),
            QtCore.Qt.TextElideMode.ElideRight,
            text_width,
        )
        display_status = metrics.elidedText(
            str(status),
            QtCore.Qt.TextElideMode.ElideRight,
            max(1, text_width - 12),
        )
        primary_top = rect.top() + max(2, (rect.height() - line_height * 2) / 2)
        _text(
            painter,
            QtCore.QRectF(x, primary_top, text_width, line_height),
            display_name,
            color=primary,
            weight=QtGui.QFont.Weight.DemiBold,
        )
        painter.setBrush(_tone_color(tone))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        status_top = primary_top + line_height
        painter.drawEllipse(QtCore.QPointF(x + 4, status_top + line_height / 2), 3.5, 3.5)
        _text(
            painter,
            QtCore.QRectF(x + 12, status_top, text_width - 12, line_height),
            display_status,
            color=_tone_color(tone) if tone != "muted" else muted,
            size_delta=-0.5,
        )
        painter.restore()


class WorkspacePageQueueDelegate(QtWidgets.QStyledItemDelegate):
    """Dense queue row matching the selected Hybrid Pro workspace table."""

    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtCore.QSize:
        del index
        metrics = QtGui.QFontMetrics(option.font)
        return QtCore.QSize(520, max(53, metrics.lineSpacing() * 2 + 15))

    @staticmethod
    def _draw_elided(
        painter: QtGui.QPainter,
        rect: QtCore.QRectF,
        value: object,
        *,
        color: QtGui.QColor,
        weight: QtGui.QFont.Weight = QtGui.QFont.Weight.Normal,
        alignment: QtCore.Qt.AlignmentFlag = QtCore.Qt.AlignmentFlag.AlignLeft,
    ) -> None:
        painter.save()
        font = painter.font()
        font.setWeight(weight)
        painter.setFont(font)
        painter.setPen(color)
        text = painter.fontMetrics().elidedText(
            str(value),
            QtCore.Qt.TextElideMode.ElideRight,
            max(1, int(rect.width())),
        )
        painter.drawText(
            rect,
            int(alignment | QtCore.Qt.AlignmentFlag.AlignVCenter),
            text,
        )
        painter.restore()

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        painter.save()
        rect = QtCore.QRectF(option.rect)
        selected = bool(option.state & QtWidgets.QStyle.StateFlag.State_Selected)
        if selected:
            painter.fillRect(rect, QtGui.QColor(theme_token(_theme(), "surface-selected")))
        painter.setPen(QtGui.QColor(theme_token(_theme(), "border-subtle")))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())

        primary = QtGui.QColor(theme_token(_theme(), "content-primary"))
        secondary = QtGui.QColor(theme_token(_theme(), "content-secondary"))
        muted = QtGui.QColor(theme_token(_theme(), "content-muted"))
        file_name = str(index.data(int(PageRole.FILE_NAME)) or index.data() or "")
        thumbnail = str(index.data(int(PageRole.THUMBNAIL_PATH)) or "")
        status = str(index.data(int(PageRole.WORKSPACE_STATUS_LABEL)) or "")
        tone = str(index.data(int(PageRole.WORKSPACE_STATUS_TONE)) or "muted")
        owner = str(index.data(int(PageRole.OWNER)) or "—")
        elapsed = str(index.data(int(PageRole.ELAPSED_LABEL)) or "—")

        left = rect.left() + 11
        thumb_rect = QtCore.QRectF(
            left,
            rect.center().y() - 19.5,
            30,
            39,
        )
        if thumbnail and os.path.isfile(thumbnail):
            pixmap = QtGui.QPixmap(thumbnail)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    int(thumb_rect.width()),
                    int(thumb_rect.height()),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                target = QtCore.QRectF(
                    thumb_rect.center().x() - scaled.width() / 2,
                    thumb_rect.center().y() - scaled.height() / 2,
                    scaled.width(),
                    scaled.height(),
                )
                painter.drawPixmap(target, scaled, QtCore.QRectF(scaled.rect()))

        page_left = thumb_rect.right() + 9
        inner_left = rect.left() + 11
        action_width = 36.0
        track_width = max(1.0, rect.width() - 22.0 - action_width)
        unit = track_width / 3.85
        status_left = inner_left + unit * 1.3
        owner_left = status_left + unit
        progress_left = owner_left + unit * 0.9
        action_left = progress_left + unit * 0.65
        page_rect = QtCore.QRectF(
            page_left,
            rect.top(),
            max(40.0, status_left - page_left - 12),
            rect.height(),
        )
        self._draw_elided(
            painter,
            page_rect,
            file_name,
            color=primary,
            weight=QtGui.QFont.Weight.DemiBold,
        )

        status_color = _tone_color(tone)
        status_cell_width = max(70.0, owner_left - status_left - 12)
        status_font = painter.font()
        status_font.setWeight(QtGui.QFont.Weight.DemiBold)
        status_width = min(
            status_cell_width - 8,
            max(
                58.0,
                QtGui.QFontMetrics(status_font).horizontalAdvance(status) + 36.0,
            ),
        )
        status_rect = QtCore.QRectF(
            status_left,
            rect.top() + rect.height() / 2 - 14,
            status_width,
            28,
        )
        pill = status_rect.adjusted(0, 2, 0, -2)
        fill = QtGui.QColor(status_color)
        fill.setAlpha(28 if _theme() == "light" else 52)
        painter.setBrush(fill)
        painter.setPen(status_color)
        painter.drawRoundedRect(pill, 12, 12)
        status_icon_name = {
            "ready": "success",
            "editing": "editor",
            "warning": "warning",
            "error": "warning",
        }.get(tone, "status-muted")
        hybrid_icon(status_icon_name, _theme()).paint(
            painter,
            QtCore.QRect(
                int(pill.left() + 8),
                int(pill.center().y() - 6),
                12,
                12,
            ),
        )
        self._draw_elided(
            painter,
            pill.adjusted(25, 0, -8, 0),
            status,
            color=status_color,
            weight=QtGui.QFont.Weight.DemiBold,
        )
        self._draw_elided(
            painter,
            QtCore.QRectF(
                owner_left,
                rect.top(),
                max(40.0, progress_left - owner_left - 12),
                rect.height(),
            ),
            owner,
            color=secondary,
        )
        self._draw_elided(
            painter,
            QtCore.QRectF(
                progress_left,
                rect.top(),
                max(36.0, action_left - progress_left - 12),
                rect.height(),
            ),
            elapsed,
            color=muted,
            alignment=QtCore.Qt.AlignmentFlag.AlignRight,
        )
        action_rect = QtCore.QRectF(
            action_left,
            rect.top(),
            rect.right() - action_left - 12,
            rect.height(),
        )
        action_icon = hybrid_icon("caret-right", _theme()).pixmap(
            QtCore.QSize(13, 13)
        )
        painter.drawPixmap(
            QtCore.QPointF(
                action_rect.right() - action_icon.width(),
                action_rect.center().y() - action_icon.height() / 2,
            ),
            action_icon,
        )
        painter.restore()


__all__ = ["PageRailDelegate", "ProjectCardDelegate", "WorkspacePageQueueDelegate"]
