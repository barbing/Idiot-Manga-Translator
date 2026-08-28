# -*- coding: utf-8 -*-
"""Canvas-first Page Editor shell for the staged native GUI."""
from __future__ import annotations

from collections.abc import Mapping
import math

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.design_system import (
    CommandButton,
    SectionHeader,
    WheelSafeComboBox,
    WheelSafeDoubleSpinBox,
    WheelSafeSpinBox,
    apply_semantic_properties,
    metric_pixels,
)
from app.ui.editor.activity_dock import ActivityDock
from app.ui.editor.canvas import (
    CanvasArtifactSet,
    OverlayAvailability,
    OverlayShape,
    PageCanvasView,
    RasterOverlaySource,
)
from app.ui.design_system.delegates import PageRailDelegate
from app.ui.design_system.icons import hybrid_icon
from app.ui.presentation import editor_preview_action
from app.ui.ui_contract import (
    CANVAS_VIEW_IDS,
    INSPECTOR_TAB_IDS,
    OVERLAY_IDS,
    ActivityDockBounds,
    LayoutMode,
)


def _qt_normalized_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _exact_offset(value: str, normalized_offset: int) -> int:
    """Map one QTextDocument offset back into the exact external string."""

    exact_offset = 0
    document_offset = 0
    while document_offset < normalized_offset and exact_offset < len(value):
        if value[exact_offset] == "\r":
            exact_offset += 1
            if exact_offset < len(value) and value[exact_offset] == "\n":
                exact_offset += 1
        else:
            exact_offset += 1
        document_offset += 1
    if document_offset != normalized_offset:
        raise ValueError("QTextDocument offset is outside the exact text")
    return exact_offset


def _python_offset_from_qt(value: str, qt_offset: int) -> int:
    """Translate a UTF-16 QTextCursor position into a Python string offset."""

    consumed = 0
    if qt_offset < 0:
        raise ValueError("QTextCursor offset must be non-negative")
    for index, character in enumerate(value):
        consumed += 2 if ord(character) > 0xFFFF else 1
        if consumed == qt_offset:
            return index + 1
        if consumed > qt_offset:
            raise ValueError("QTextCursor offset splits a Unicode code point")
    if consumed != qt_offset:
        raise ValueError("QTextCursor offset is outside the exact text")
    return len(value)


class _ExactTextEdit(QtWidgets.QPlainTextEdit):
    """QPlainTextEdit adapter that preserves exact external CR/LF sequences.

    Qt normalizes every paragraph separator to ``\n`` internally.  The editor
    keeps an exact companion string so changing ordinary characters does not
    silently rewrite existing CRLF or CR separators at the command boundary.
    Pasted text keeps its own separators; keyboard-created lines use the
    selected text's first existing separator, falling back to ``\n``.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._exact_text = ""
        self._normalized_text = ""
        self._pending_insert_exact: str | None = None
        self._programmatic_update = False
        self._history_replay = False
        self._history_identity = ""
        self._exact_history: list[tuple[str, int, int]] = [("", 0, 0)]
        self._history_index = 0
        self.setUndoRedoEnabled(False)
        self.textChanged.connect(self._synchronize_exact_text)

    def set_exact_text(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("exact text must be a string")
        normalized = _qt_normalized_text(value)
        self._exact_text = value
        self._normalized_text = normalized
        if super().toPlainText() == normalized:
            self._reset_exact_history()
            return
        self._programmatic_update = True
        try:
            super().setPlainText(normalized)
        finally:
            self._programmatic_update = False
        self._reset_exact_history()

    def exact_text(self) -> str:
        return self._exact_text

    def bind_history_identity(self, identity: str) -> None:
        if not isinstance(identity, str):
            raise TypeError("text history identity must be a string")
        if identity == self._history_identity:
            return
        self._history_identity = identity
        self._reset_exact_history()

    def insertFromMimeData(self, source: QtCore.QMimeData) -> None:  # noqa: N802
        exact = source.text() if source.hasText() else None
        self._pending_insert_exact = exact
        try:
            super().insertFromMimeData(source)
        finally:
            self._pending_insert_exact = None

    def createMimeDataFromSelection(self) -> QtCore.QMimeData:  # noqa: N802
        mime = QtCore.QMimeData()
        mime.setText(self._exact_selection_text())
        return mime

    def copy(self) -> None:
        if self.textCursor().hasSelection():
            QtWidgets.QApplication.clipboard().setMimeData(
                self.createMimeDataFromSelection()
            )

    def cut(self) -> None:
        cursor = self.textCursor()
        if self.isReadOnly() or not cursor.hasSelection():
            return
        self.copy()
        cursor.removeSelectedText()

    def undo(self) -> None:
        if self._history_index <= 0:
            return
        self._history_index -= 1
        self._restore_exact_history()

    def redo(self) -> None:
        if self._history_index >= len(self._exact_history) - 1:
            return
        self._history_index += 1
        self._restore_exact_history()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if event.matches(QtGui.QKeySequence.StandardKey.Undo):
            self.undo()
            event.accept()
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Redo):
            self.redo()
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:  # noqa: N802
        menu = QtWidgets.QMenu(self)
        undo_action = menu.addAction("Undo", self.undo)
        undo_action.setEnabled(self._history_index > 0)
        redo_action = menu.addAction("Redo", self.redo)
        redo_action.setEnabled(self._history_index < len(self._exact_history) - 1)
        menu.addSeparator()
        cursor = self.textCursor()
        cut_action = menu.addAction("Cut", self.cut)
        cut_action.setEnabled(not self.isReadOnly() and cursor.hasSelection())
        copy_action = menu.addAction("Copy", self.copy)
        copy_action.setEnabled(cursor.hasSelection())
        paste_action = menu.addAction("Paste", self.paste)
        paste_action.setEnabled(not self.isReadOnly() and self.canPaste())
        delete_action = menu.addAction("Delete")
        delete_action.setEnabled(not self.isReadOnly() and cursor.hasSelection())
        delete_action.triggered.connect(self._delete_selection)
        menu.addSeparator()
        select_action = menu.addAction("Select All", self.selectAll)
        select_action.setEnabled(bool(self._normalized_text))
        menu.exec(event.globalPos())
        menu.deleteLater()

    @QtCore.Slot()
    def _synchronize_exact_text(self) -> None:
        current = super().toPlainText()
        if self._programmatic_update:
            self._normalized_text = current
            return
        previous = self._normalized_text
        prefix = 0
        shared_limit = min(len(previous), len(current))
        while prefix < shared_limit and previous[prefix] == current[prefix]:
            prefix += 1
        suffix = 0
        previous_tail = len(previous) - prefix
        current_tail = len(current) - prefix
        while (
            suffix < previous_tail
            and suffix < current_tail
            and previous[len(previous) - 1 - suffix]
            == current[len(current) - 1 - suffix]
        ):
            suffix += 1

        previous_end = len(previous) - suffix
        current_end = len(current) - suffix
        exact_start = _exact_offset(self._exact_text, prefix)
        exact_end = _exact_offset(self._exact_text, previous_end)
        inserted = current[prefix:current_end]
        pending = self._pending_insert_exact
        exact_inserted = (
            pending
            if pending is not None and _qt_normalized_text(pending) == inserted
            else inserted.replace("\n", self._preferred_separator())
        )
        self._exact_text = (
            self._exact_text[:exact_start]
            + exact_inserted
            + self._exact_text[exact_end:]
        )
        self._normalized_text = current
        if not self._history_replay:
            self._record_exact_history()

    def _preferred_separator(self) -> str:
        value = self._exact_text
        for index, character in enumerate(value):
            if character == "\r":
                return "\r\n" if value[index : index + 2] == "\r\n" else "\r"
            if character == "\n":
                return "\n"
        return "\n"

    def _exact_selection_text(self) -> str:
        cursor = self.textCursor()
        if not cursor.hasSelection():
            return ""
        start = _python_offset_from_qt(
            self._normalized_text, cursor.selectionStart()
        )
        end = _python_offset_from_qt(
            self._normalized_text, cursor.selectionEnd()
        )
        return self._exact_text[
            _exact_offset(self._exact_text, start) : _exact_offset(
                self._exact_text, end
            )
        ]

    @QtCore.Slot()
    def _delete_selection(self) -> None:
        cursor = self.textCursor()
        if cursor.hasSelection() and not self.isReadOnly():
            cursor.removeSelectedText()

    def _reset_exact_history(self) -> None:
        cursor = self.textCursor()
        self._exact_history = [
            (self._exact_text, cursor.position(), cursor.anchor())
        ]
        self._history_index = 0

    def _record_exact_history(self) -> None:
        cursor = self.textCursor()
        entry = (self._exact_text, cursor.position(), cursor.anchor())
        self._exact_history = self._exact_history[: self._history_index + 1]
        if self._exact_history and self._exact_history[-1][0] == self._exact_text:
            self._exact_history[-1] = entry
            return
        self._exact_history.append(entry)
        self._history_index = len(self._exact_history) - 1

    def _restore_exact_history(self) -> None:
        value, position, anchor = self._exact_history[self._history_index]
        self._exact_text = value
        self._normalized_text = _qt_normalized_text(value)
        self._history_replay = True
        self._programmatic_update = True
        try:
            super().setPlainText(self._normalized_text)
            cursor = self.textCursor()
            cursor.setPosition(max(0, min(anchor, self.document().characterCount() - 1)))
            cursor.setPosition(
                max(0, min(position, self.document().characterCount() - 1)),
                QtGui.QTextCursor.MoveMode.KeepAnchor,
            )
            self.setTextCursor(cursor)
        finally:
            self._programmatic_update = False
            self._history_replay = False


class _BadgeTextEditFrame(QtWidgets.QFrame):
    """Overlay one compact provenance badge without changing editor geometry."""

    def __init__(
        self,
        editor: _ExactTextEdit,
        badge: QtWidgets.QLabel,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("badgeTextEditFrame")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(editor)
        badge.setParent(self)
        badge.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        badge.raise_()
        self._badge = badge

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._badge.adjustSize()
        self._badge.move(
            max(6, self.width() - self._badge.width() - 7),
            7,
        )


class PageEditorView(QtWidgets.QWidget):
    """Canvas-first editor using typed models and explicit artifact commands."""

    page_selected = QtCore.Signal(str)
    parent_selected = QtCore.Signal(str)
    rerender_requested = QtCore.Signal(str)
    manual_cleanup_requested = QtCore.Signal(str)
    hub_requested = QtCore.Signal()
    workspace_requested = QtCore.Signal()
    settings_requested = QtCore.Signal()
    layout_changed = QtCore.Signal()
    target_text_draft_changed = QtCore.Signal(str)
    target_text_apply_requested = QtCore.Signal()
    target_text_cancel_requested = QtCore.Signal()
    target_text_restore_requested = QtCore.Signal()
    target_text_keep_existing_requested = QtCore.Signal()
    source_text_draft_changed = QtCore.Signal(str)
    source_text_apply_requested = QtCore.Signal()
    source_text_cancel_requested = QtCore.Signal()
    source_text_restore_requested = QtCore.Signal()
    ocr_revision_requested = QtCore.Signal()
    ocr_revision_cancel_requested = QtCore.Signal()
    translation_revision_requested = QtCore.Signal()
    translation_revision_cancel_requested = QtCore.Signal()
    parent_exclusion_requested = QtCore.Signal(bool)
    parent_geometry_draft_changed = QtCore.Signal(object)
    parent_geometry_apply_requested = QtCore.Signal()
    parent_geometry_cancel_requested = QtCore.Signal()
    reading_order_move_earlier_requested = QtCore.Signal()
    reading_order_move_later_requested = QtCore.Signal()
    reading_order_apply_requested = QtCore.Signal()
    reading_order_cancel_requested = QtCore.Signal()
    merge_parent_partner_changed = QtCore.Signal(object)
    merge_parent_requested = QtCore.Signal()
    merge_parent_cancel_requested = QtCore.Signal()
    split_parent_orientation_changed = QtCore.Signal(object)
    split_parent_offset_changed = QtCore.Signal(int)
    split_parent_requested = QtCore.Signal()
    split_parent_cancel_requested = QtCore.Signal()
    writing_mode_draft_changed = QtCore.Signal(str)
    writing_mode_apply_requested = QtCore.Signal()
    writing_mode_cancel_requested = QtCore.Signal()
    writing_mode_restore_requested = QtCore.Signal()
    line_height_draft_changed = QtCore.Signal(float)
    line_height_apply_requested = QtCore.Signal()
    line_height_cancel_requested = QtCore.Signal()
    line_height_restore_requested = QtCore.Signal()
    rotation_draft_changed = QtCore.Signal(float)
    rotation_apply_requested = QtCore.Signal()
    rotation_cancel_requested = QtCore.Signal()
    rotation_restore_requested = QtCore.Signal()
    render_box_draft_changed = QtCore.Signal(object)
    render_box_apply_requested = QtCore.Signal()
    render_box_cancel_requested = QtCore.Signal()
    render_box_restore_requested = QtCore.Signal()
    font_role_draft_changed = QtCore.Signal(str)
    font_role_apply_requested = QtCore.Signal()
    font_role_cancel_requested = QtCore.Signal()
    font_role_restore_requested = QtCore.Signal()
    font_weight_tier_draft_changed = QtCore.Signal(str)
    font_weight_tier_apply_requested = QtCore.Signal()
    font_weight_tier_cancel_requested = QtCore.Signal()
    font_weight_tier_restore_requested = QtCore.Signal()
    fill_color_draft_changed = QtCore.Signal(str)
    fill_color_apply_requested = QtCore.Signal()
    fill_color_cancel_requested = QtCore.Signal()
    fill_color_restore_requested = QtCore.Signal()
    outline_color_draft_changed = QtCore.Signal(str)
    outline_color_apply_requested = QtCore.Signal()
    outline_color_cancel_requested = QtCore.Signal()
    outline_color_restore_requested = QtCore.Signal()
    outline_width_draft_changed = QtCore.Signal(float)
    outline_width_apply_requested = QtCore.Signal()
    outline_width_cancel_requested = QtCore.Signal()
    outline_width_restore_requested = QtCore.Signal()
    preferred_size_draft_changed = QtCore.Signal(float)
    preferred_size_apply_requested = QtCore.Signal()
    preferred_size_cancel_requested = QtCore.Signal()
    preferred_size_restore_requested = QtCore.Signal()
    shadow_color_draft_changed = QtCore.Signal(str)
    shadow_color_apply_requested = QtCore.Signal()
    shadow_color_cancel_requested = QtCore.Signal()
    shadow_color_restore_requested = QtCore.Signal()
    shadow_blur_draft_changed = QtCore.Signal(float)
    shadow_blur_apply_requested = QtCore.Signal()
    shadow_blur_cancel_requested = QtCore.Signal()
    shadow_blur_restore_requested = QtCore.Signal()
    shadow_offset_draft_changed = QtCore.Signal(object)
    shadow_offset_apply_requested = QtCore.Signal()
    shadow_offset_cancel_requested = QtCore.Signal()
    shadow_offset_restore_requested = QtCore.Signal()
    shadow_visibility_draft_changed = QtCore.Signal(bool)
    shadow_visibility_apply_requested = QtCore.Signal()
    shadow_visibility_cancel_requested = QtCore.Signal()
    shadow_visibility_restore_requested = QtCore.Signal()
    add_user_parent_role_changed = QtCore.Signal(object)
    add_user_parent_workflow_area_changed = QtCore.Signal(object)
    add_user_parent_requested = QtCore.Signal()
    add_user_parent_cancel_requested = QtCore.Signal()
    history_selection_changed = QtCore.Signal(str)
    history_revoke_requested = QtCore.Signal()
    history_reapply_requested = QtCore.Signal()
    render_override_reset_scope_changed = QtCore.Signal(str)
    render_override_reset_field_group_changed = QtCore.Signal(str)
    render_override_reset_requested = QtCore.Signal()
    render_override_reset_cancel_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("pageEditor")
        self.setAccessibleName("Page Editor")
        self._page_id_role: int | None = None
        self._parent_id_role: int | None = None
        self._current_page_id = ""
        self._current_parent_id = ""
        self._parent_excluded = False
        self._geometry_programmatic_update = False
        self._geometry_canvas_size = (1, 1)
        self._merge_parent_programmatic_update = False
        self._split_parent_programmatic_update = False
        self._writing_mode_programmatic_update = False
        self._line_height_programmatic_update = False
        self._rotation_programmatic_update = False
        self._render_box_programmatic_update = False
        self._font_role_programmatic_update = False
        self._font_weight_tier_programmatic_update = False
        self._fill_color_programmatic_update = False
        self._outline_color_programmatic_update = False
        self._outline_width_programmatic_update = False
        self._preferred_size_programmatic_update = False
        self._shadow_color_programmatic_update = False
        self._shadow_blur_programmatic_update = False
        self._shadow_offset_programmatic_update = False
        self._shadow_visibility_programmatic_update = False
        self._add_user_parent_programmatic_update = False
        self._add_user_parent_canvas_size = (1, 1)
        self._add_user_parent_show_canvas_draft = False
        self._add_user_parent_panel_requested = False
        self._add_user_parent_draft_present = False
        self._add_user_parent_busy = False
        self._add_user_parent_editing_enabled = False
        self._text_provenance_details_requested = False
        self._source_provenance_available = False
        self._target_provenance_available = False
        self._layout_mode: LayoutMode | None = None
        self._activity_bounds = ActivityDockBounds(
            min=320,
            preferred=320,
            max=360,
            resizable=True,
        )
        self._activity_requested_height = 320
        self._canvas_focus_active = False
        self._inspector_hidden_by_toolbar = False
        self._page_rail_collapsed = False
        self._icon_theme = "dark"
        self._history_payload: tuple[
            tuple[str, ...],
            tuple[str, ...],
            str,
            tuple[str, ...],
        ] | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.vertical_splitter.setObjectName("editorVerticalSplitter")
        self.vertical_splitter.setChildrenCollapsible(False)
        self.vertical_splitter.splitterMoved.connect(lambda *_: self.layout_changed.emit())

        self.editor_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.editor_splitter.setObjectName("editorWorkspaceSplitter")
        self.editor_splitter.setChildrenCollapsible(False)
        self.editor_splitter.splitterMoved.connect(lambda *_: self.layout_changed.emit())
        self.page_rail = self._build_page_rail()
        self.canvas_panel = self._build_canvas_panel()
        self.inspector = self._build_inspector()
        self.editor_splitter.addWidget(self.page_rail)
        self.editor_splitter.addWidget(self.canvas_panel)
        self.editor_splitter.addWidget(self.inspector)
        self.editor_splitter.setStretchFactor(0, 0)
        self.editor_splitter.setStretchFactor(1, 1)
        self.editor_splitter.setStretchFactor(2, 0)
        self.editor_splitter.setSizes((184, 820, 368))
        self.vertical_splitter.addWidget(self.editor_splitter)

        self.activity_dock = ActivityDock()
        self.activity_dock.hub_requested.connect(self.hub_requested)
        self.activity_dock.workspace_requested.connect(self.workspace_requested)
        self.activity_dock.settings_requested.connect(self.settings_requested)
        self.activity_dock.inspector_requested.connect(self.show_inspector_tab)
        self.activity_dock.expanded_changed.connect(
            self._activity_expansion_changed
        )
        self.activity_dock.tab_changed.connect(lambda *_: self.layout_changed.emit())
        self.vertical_splitter.addWidget(self.activity_dock)
        self.vertical_splitter.setStretchFactor(0, 1)
        self.vertical_splitter.setStretchFactor(1, 0)
        self.vertical_splitter.setSizes((580, 320))
        root.addWidget(self.vertical_splitter)

    def refresh_icons(self, theme: str) -> None:
        self._icon_theme = str(theme)
        self.page_rail_toggle.setIcon(
            hybrid_icon(
                "caret-right" if self._page_rail_collapsed else "caret-down",
                self._icon_theme,
            )
        )
        self.page_search_action.setIcon(hybrid_icon("search", self._icon_theme))
        for button, icon_name in (
            (self.page_previous_button, "caret-left"),
            (self.page_next_button, "caret-right"),
            (self.page_grid_button, "grid"),
            (self.page_list_button, "list"),
            (self.select_tool_button, "select"),
            (self.pan_tool_button, "pan"),
            (self.zoom_out_button, "zoom-out"),
            (self.zoom_in_button, "zoom-in"),
            (self.overlay_button, "overlays"),
            (self.hold_original_button, "eye"),
            (self.canvas_focus_button, "fullscreen"),
            (self.inspector_toggle_button, "sidebar"),
            (self.inspector_more_button, "more"),
            (self.previous_parent_button, "caret-left"),
            (self.next_parent_button, "caret-right"),
            (self.history_revoke_button, "undo"),
            (self.history_reapply_button, "redo"),
            (self.render_override_reset_toggle, "more"),
        ):
            button.setIcon(hybrid_icon(icon_name, self._icon_theme))
        self.mode_buttons["compare"].setIcon(
            hybrid_icon("brand", self._icon_theme)
        )
        self.rerender_button.setIcon(
            hybrid_icon("play", self._icon_theme, active=True)
        )
        for tool_id, button in getattr(self, "cleanup_tool_buttons", {}).items():
            button.setIcon(
                hybrid_icon(
                    {
                        "brush": "cleanup",
                        "lasso": "lasso",
                        "rectangle": "rectangle",
                        "eraser": "eraser",
                        "protect": "shield",
                    }[tool_id],
                    self._icon_theme,
                    active=button.isChecked(),
                )
            )
        self.activity_dock.refresh_icons(self._icon_theme)

    def _build_page_rail(self) -> QtWidgets.QWidget:
        rail = QtWidgets.QFrame()
        rail.setObjectName("pageRail")
        rail.setProperty("role", "panel")
        rail.setMinimumWidth(84)
        rail.setMaximumWidth(280)
        layout = QtWidgets.QVBoxLayout(rail)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        heading = QtWidgets.QFrame()
        heading.setObjectName("pageRailHeading")
        heading.setFixedHeight(40)
        project_row = QtWidgets.QHBoxLayout(heading)
        project_row.setContentsMargins(13, 0, 8, 0)
        project_row.setSpacing(14)
        self.project_label = QtWidgets.QLabel("Project")
        self.project_label.setObjectName("pageRailProjectLabel")
        self.project_name = QtWidgets.QLabel("No project")
        self.project_name.setObjectName("pageRailProjectName")
        self.project_name.setWordWrap(False)
        project_row.addWidget(self.project_label)
        project_row.addWidget(self.project_name, 1)
        project_row.addStretch(1)
        self.page_rail_toggle = QtWidgets.QToolButton()
        self.page_rail_toggle.setObjectName("pageRailToggle")
        self.page_rail_toggle.setIcon(hybrid_icon("caret-down"))
        self.page_rail_toggle.setAccessibleName("Collapse page navigator")
        self.page_rail_toggle.setToolTip("Collapse page navigator")
        self.page_rail_toggle.clicked.connect(self._toggle_page_rail_collapsed)
        project_row.addWidget(self.page_rail_toggle)
        layout.addWidget(heading)

        search_band = QtWidgets.QFrame()
        search_band.setObjectName("pageRailSearchBand")
        search_band.setFixedHeight(36)
        search_layout = QtWidgets.QVBoxLayout(search_band)
        search_layout.setContentsMargins(11, 4, 11, 3)
        search_layout.setSpacing(0)
        self.page_search = QtWidgets.QLineEdit()
        self.page_search.setPlaceholderText("Search pages")
        self.page_search.setClearButtonEnabled(True)
        self.page_search.setAccessibleName("Search pages")
        self.page_search.setFixedHeight(29)
        self.page_search_action = self.page_search.addAction(
            hybrid_icon("search", self._icon_theme),
            QtWidgets.QLineEdit.ActionPosition.LeadingPosition,
        )
        search_layout.addWidget(self.page_search)
        layout.addWidget(search_band)
        self.page_search_band = search_band

        self.page_list = QtWidgets.QListView()
        self.page_list.setObjectName("editorPageList")
        self.page_list.setAccessibleName("Page navigator")
        self.page_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.page_list.setWordWrap(True)
        self.page_list.setSpacing(2)
        self.page_list.setUniformItemSizes(True)
        self.page_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.page_list.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.page_list.setItemDelegate(
            PageRailDelegate(compact=True, parent=self.page_list)
        )
        self.page_list.activated.connect(self._activate_page)
        self.page_list.clicked.connect(self._activate_page)
        layout.addWidget(self.page_list, 1)
        footer = QtWidgets.QFrame()
        footer.setObjectName("pageRailFooter")
        footer_layout = QtWidgets.QHBoxLayout(footer)
        footer_layout.setContentsMargins(12, 0, 8, 0)
        footer_layout.setSpacing(2)
        self.page_count = QtWidgets.QLabel("Page\n—")
        self.page_count.setProperty("role", "secondary")
        self.page_count.setWordWrap(True)
        self.page_count.setMinimumWidth(38)
        self.page_count.setMaximumWidth(42)
        self.page_count.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        footer_layout.addWidget(self.page_count)
        footer_layout.addStretch(1)
        self.page_previous_button = QtWidgets.QToolButton()
        self.page_previous_button.setObjectName("pageRailPrevious")
        self.page_previous_button.setIcon(hybrid_icon("caret-left"))
        self.page_previous_button.setAccessibleName("Previous page")
        self.page_previous_button.setToolTip("Previous page")
        self.page_previous_button.setEnabled(False)
        self.page_previous_button.clicked.connect(
            lambda: self._activate_page_offset(-1)
        )
        footer_layout.addWidget(self.page_previous_button)
        self.page_next_button = QtWidgets.QToolButton()
        self.page_next_button.setObjectName("pageRailNext")
        self.page_next_button.setIcon(hybrid_icon("caret-right"))
        self.page_next_button.setAccessibleName("Next page")
        self.page_next_button.setToolTip("Next page")
        self.page_next_button.setEnabled(False)
        self.page_next_button.clicked.connect(
            lambda: self._activate_page_offset(1)
        )
        footer_layout.addWidget(self.page_next_button)
        self.page_grid_button = QtWidgets.QToolButton()
        self.page_grid_button.setObjectName("pageRailGrid")
        self.page_grid_button.setCheckable(True)
        self.page_grid_button.setChecked(True)
        self.page_grid_button.setIcon(hybrid_icon("grid"))
        self.page_grid_button.setAccessibleName("Show page filmstrip")
        self.page_grid_button.setToolTip("Show page thumbnails and status")
        self.page_grid_button.clicked.connect(
            lambda: self._set_page_rail_mode(compact=True)
        )
        footer_layout.addWidget(self.page_grid_button)
        self.page_list_button = QtWidgets.QToolButton()
        self.page_list_button.setObjectName("pageRailList")
        self.page_list_button.setCheckable(True)
        self.page_list_button.setIcon(hybrid_icon("list"))
        self.page_list_button.setAccessibleName("Show compact page list")
        self.page_list_button.setToolTip("Show a compact page list")
        self.page_list_button.clicked.connect(
            lambda: self._set_page_rail_mode(compact=False)
        )
        footer_layout.addWidget(self.page_list_button)
        layout.addWidget(footer)
        return rail

    def _build_canvas_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setObjectName("canvasWorkspace")
        panel.setProperty("role", "canvas")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QtWidgets.QFrame()
        self.canvas_toolbar = toolbar
        toolbar.setObjectName("canvasToolbar")
        toolbar.setProperty("role", "header")
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 0, 8, 0)
        toolbar_layout.setSpacing(3)
        mode_strip = QtWidgets.QFrame()
        mode_strip.setObjectName("canvasModeStrip")
        mode_layout = QtWidgets.QHBoxLayout(mode_strip)
        mode_layout.setContentsMargins(2, 2, 2, 2)
        mode_layout.setSpacing(2)
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons: dict[str, QtWidgets.QToolButton] = {}
        for index, mode in enumerate(CANVAS_VIEW_IDS):
            button = QtWidgets.QToolButton()
            button.setProperty("role", "command")
            button.setProperty("variant", "quiet")
            button.setText(mode.title())
            button.setCheckable(True)
            button.setProperty("canvasMode", mode)
            button.setAccessibleName(f"Show {mode} page")
            if mode == "original":
                button.setChecked(True)
            if mode == "compare":
                button.setIcon(hybrid_icon("brand"))
                button.setIconSize(QtCore.QSize(14, 14))
                button.setToolButtonStyle(
                    QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
                )
                button.setFixedWidth(84)
            else:
                button.setFixedWidth(68)
            button.clicked.connect(
                lambda _checked=False, value=mode: self.canvas.set_mode(value)
            )
            self.mode_group.addButton(button, index)
            self.mode_buttons[mode] = button
            mode_layout.addWidget(button)
        toolbar_layout.addWidget(mode_strip)
        divider = QtWidgets.QFrame()
        divider.setObjectName("canvasToolbarDivider")
        divider.setFixedSize(1, 25)
        toolbar_layout.addWidget(divider)

        self.canvas_tool_group = QtWidgets.QButtonGroup(self)
        self.canvas_tool_group.setExclusive(True)
        self.select_tool_button = QtWidgets.QToolButton()
        self.select_tool_button.setObjectName("canvasSelectTool")
        self.select_tool_button.setCheckable(True)
        self.select_tool_button.setChecked(True)
        self.select_tool_button.setIcon(hybrid_icon("select"))
        self.select_tool_button.setAccessibleName("Select canvas evidence")
        self.select_tool_button.clicked.connect(
            lambda: self._set_canvas_tool("select")
        )
        self.canvas_tool_group.addButton(self.select_tool_button)
        toolbar_layout.addWidget(self.select_tool_button)
        self.pan_tool_button = QtWidgets.QToolButton()
        self.pan_tool_button.setObjectName("canvasPanTool")
        self.pan_tool_button.setCheckable(True)
        self.pan_tool_button.setIcon(hybrid_icon("pan"))
        self.pan_tool_button.setAccessibleName("Pan canvas")
        self.pan_tool_button.clicked.connect(lambda: self._set_canvas_tool("pan"))
        self.canvas_tool_group.addButton(self.pan_tool_button)
        toolbar_layout.addWidget(self.pan_tool_button)
        self.zoom_out_button = QtWidgets.QToolButton()
        self.zoom_out_button.setIcon(hybrid_icon("zoom-out"))
        self.zoom_out_button.setAccessibleName("Zoom out")
        self.zoom_out_button.clicked.connect(lambda: self._adjust_canvas_zoom(-10))
        toolbar_layout.addWidget(self.zoom_out_button)
        self.zoom_in_button = QtWidgets.QToolButton()
        self.zoom_in_button.setIcon(hybrid_icon("zoom-in"))
        self.zoom_in_button.setAccessibleName("Zoom in")
        self.zoom_in_button.clicked.connect(lambda: self._adjust_canvas_zoom(10))
        toolbar_layout.addWidget(self.zoom_in_button)

        self.fit_button = QtWidgets.QToolButton()
        self.fit_button.setObjectName("canvasFitButton")
        self.fit_button.setProperty("role", "command")
        self.fit_button.setProperty("variant", "quiet")
        self.fit_button.setText("Fit page")
        self.fit_button.setFixedWidth(54)
        self.fit_button.clicked.connect(self.canvas_fit_page)
        self.fit_button.setAccessibleName("Fit page in canvas")
        toolbar_layout.addWidget(self.fit_button)
        self.hold_original_button = QtWidgets.QToolButton()
        self.hold_original_button.setObjectName("canvasHoldButton")
        self.hold_original_button.setProperty("role", "command")
        self.hold_original_button.setProperty("variant", "quiet")
        self.hold_original_button.setText("Hold original")
        self.hold_original_button.setFixedWidth(94)
        self.hold_original_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.hold_original_button.setAccessibleName("Hold to compare with Original")
        self.hold_original_button.pressed.connect(lambda: self.canvas.hold_original(True))
        self.hold_original_button.released.connect(lambda: self.canvas.hold_original(False))

        self.overlay_button = QtWidgets.QToolButton()
        self.overlay_button.setObjectName("canvasOverlayButton")
        self.overlay_button.setProperty("role", "command")
        self.overlay_button.setProperty("variant", "quiet")
        self.overlay_button.setIcon(hybrid_icon("overlays"))
        self.overlay_button.setToolTip("Choose editor overlays")
        self.overlay_button.setAccessibleName("Page evidence overlays")
        self.overlay_button.setAccessibleDescription(
            "Choose the available projected evidence overlays for this page"
        )
        self.overlay_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        overlay_menu = QtWidgets.QMenu(self.overlay_button)
        overlay_menu.setToolTipsVisible(True)
        self.overlay_actions: dict[str, QtGui.QAction] = {}
        labels = {
            "parentBounds": "Parent bounds",
            "renderBox": "Effective render box",
            "sourceFootprint": "Source footprint",
            "baseline": "Baseline and columns",
            "cleanupMask": "Cleanup mask",
            "protectedRegions": "Protected regions",
            "proof": "Cleanup evidence",
        }
        for overlay_id in OVERLAY_IDS:
            action = overlay_menu.addAction(labels[overlay_id])
            action.setCheckable(True)
            action.setEnabled(False)
            action.setToolTip("Open a page with projected overlay evidence.")
            action.toggled.connect(
                lambda enabled, value=overlay_id: self.canvas.set_overlay_enabled(value, enabled)
            )
            self.overlay_actions[overlay_id] = action
        self.overlay_button.setMenu(overlay_menu)
        toolbar_layout.addWidget(self.overlay_button)
        self.hold_original_button.setIcon(hybrid_icon("eye"))
        toolbar_layout.addWidget(self.hold_original_button)
        toolbar_layout.addStretch(1)
        self.zoom_label = QtWidgets.QLabel("100%")
        self.zoom_label.setMinimumWidth(32)
        self.zoom_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        toolbar_layout.addWidget(self.zoom_label)
        self.zoom_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(100)
        self.zoom_slider.setAccessibleName("Canvas zoom")
        self.zoom_slider.valueChanged.connect(self._set_canvas_zoom)
        toolbar_layout.addWidget(self.zoom_slider)
        self.canvas_focus_button = QtWidgets.QToolButton()
        self.canvas_focus_button.setCheckable(True)
        self.canvas_focus_button.setIcon(hybrid_icon("fullscreen"))
        self.canvas_focus_button.setAccessibleName("Enter canvas focus")
        self.canvas_focus_button.clicked.connect(self._toggle_canvas_focus)
        toolbar_layout.addWidget(self.canvas_focus_button)
        self.inspector_toggle_button = QtWidgets.QToolButton()
        self.inspector_toggle_button.setCheckable(True)
        self.inspector_toggle_button.setIcon(hybrid_icon("sidebar"))
        self.inspector_toggle_button.setAccessibleName("Hide inspector")
        self.inspector_toggle_button.clicked.connect(
            self._toggle_inspector_visibility
        )
        toolbar_layout.addWidget(self.inspector_toggle_button)
        self.rerender_button = QtWidgets.QPushButton("Preview final page")
        self.rerender_button.setObjectName("primaryCommand")
        self.rerender_button.setProperty("role", "command")
        self.rerender_button.setProperty("variant", "primary")
        self.rerender_button.setAccessibleName("Preview final page")
        self.rerender_button.setAccessibleDescription(
            "Preview the current saved text, style, and layout as final page pixels."
        )
        self.rerender_button.clicked.connect(
            lambda: self.rerender_requested.emit(self._current_page_id)
            if self._current_page_id
            else None
        )
        toolbar_layout.addWidget(self.rerender_button)
        toolbar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        toolbar.setFixedHeight(48)
        toolbar.setMinimumWidth(toolbar_layout.sizeHint().width() + 16)
        self.toolbar_scroll = QtWidgets.QScrollArea()
        self.toolbar_scroll.setObjectName("canvasToolbarScroll")
        self.toolbar_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.toolbar_scroll.setWidgetResizable(True)
        self.toolbar_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.toolbar_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.toolbar_scroll.setWidget(toolbar)
        self.toolbar_scroll.setFixedHeight(48)
        layout.addWidget(self.toolbar_scroll)

        self.canvas = PageCanvasView()
        self.canvas.zoom_changed.connect(self._sync_zoom)
        self.canvas.mode_changed.connect(self._sync_mode)
        layout.addWidget(self.canvas, 1)
        self._set_canvas_tool("select")
        return panel

    def _build_inspector(self) -> QtWidgets.QWidget:
        inspector = QtWidgets.QFrame()
        inspector.setObjectName("pageInspector")
        inspector.setProperty("role", "panel")
        inspector.setMinimumWidth(300)
        inspector.setMaximumWidth(520)
        layout = QtWidgets.QVBoxLayout(inspector)
        layout.setContentsMargins(10, 8, 10, 4)
        layout.setSpacing(metric_pixels("space-1"))
        identity = QtWidgets.QHBoxLayout()
        self.inspector_identity_layout = identity
        self.inspector_page = QtWidgets.QLabel("No page")
        self.inspector_page.setProperty("role", "eyebrow")
        self.inspector_page.setWordWrap(True)
        self.inspector_parent = QtWidgets.QLabel("No parent selected")
        self.inspector_parent.setProperty("role", "metric")
        self.inspector_parent.setWordWrap(True)
        identity.addWidget(self.inspector_page)
        identity.addStretch(1)
        identity.addWidget(self.inspector_parent)
        layout.addLayout(identity)
        self.parent_list = WheelSafeComboBox()
        self.parent_list.setAccessibleName("Selected parent")
        self.parent_list.currentIndexChanged.connect(self._activate_parent)
        self.parent_list.setVisible(False)
        layout.addWidget(self.parent_list)
        membership = QtWidgets.QFrame()
        membership.setProperty("role", "panel-raised")
        membership_layout = QtWidgets.QVBoxLayout(membership)
        membership_layout.setContentsMargins(6, 4, 4, 4)
        membership_layout.setSpacing(6)
        self.parent_membership_status = QtWidgets.QLabel(
            "Select a parent to manage membership"
        )
        self.parent_membership_status.setProperty("role", "secondary")
        self.parent_membership_status.setProperty("tone", "muted")
        self.parent_membership_status.setWordWrap(True)
        self.parent_membership_status.setAccessibleName("Parent membership status")
        membership_layout.addWidget(self.parent_membership_status)
        self.parent_membership_button = QtWidgets.QPushButton("Exclude Parent")
        self.parent_membership_button.setProperty("role", "command")
        self.parent_membership_button.setProperty("variant", "secondary")
        self.parent_membership_button.setAccessibleName(
            "Exclude selected parent from the effective page"
        )
        self.parent_membership_button.setToolTip(
            "Append a reversible membership edit. Automatic detection evidence "
            "is retained and rendering remains explicit."
        )
        self.parent_membership_button.setEnabled(False)
        self.parent_membership_button.clicked.connect(
            lambda: self.parent_exclusion_requested.emit(
                not self._parent_excluded
            )
        )
        membership_layout.addWidget(
            self.parent_membership_button,
            0,
            QtCore.Qt.AlignmentFlag.AlignRight,
        )
        self.parent_membership_frame = membership
        self.inspector_tabs = QtWidgets.QTabWidget()
        self.inspector_tabs.setObjectName("inspectorTabs")
        self.inspector_tabs.setDocumentMode(True)
        # Every inspector page owns its vertical scrolling.  The tab container
        # must therefore yield to the enclosing inspector height instead of
        # propagating a page size hint and pushing the persistent footer beyond
        # the panel at large fonts or after a state reveals a second action row.
        self.inspector_tabs.setMinimumHeight(0)
        self.inspector_tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        self._inspector_index: dict[str, int] = {}
        for tab in INSPECTOR_TAB_IDS:
            builder = getattr(self, f"_build_{tab}_tab")
            index = self.inspector_tabs.addTab(builder(), tab.title())
            self._inspector_index[tab] = index
        layout.addWidget(self.inspector_tabs, 1)
        self.inspector_footer = QtWidgets.QFrame()
        self.inspector_footer.setObjectName("inspectorFooter")
        footer_layout = QtWidgets.QHBoxLayout(self.inspector_footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(metric_pixels("space-2"))
        self.rerender_button.setIcon(hybrid_icon("play"))
        footer_layout.addWidget(self.rerender_button, 1)
        self.canvas_toolbar.setMinimumWidth(
            self.canvas_toolbar.layout().sizeHint().width() + 16
        )
        self.target_restore_button.setText("Restore target")
        footer_layout.addWidget(self.target_restore_button)
        self.inspector_more_button = QtWidgets.QToolButton()
        self.inspector_more_button.setProperty("role", "command")
        self.inspector_more_button.setProperty("variant", "secondary")
        self.inspector_more_button.setIcon(hybrid_icon("more"))
        self.inspector_more_button.setAccessibleName("More selected parent actions")
        self.inspector_more_button.setToolTip(
            "Additional selected-parent actions remain explicit in the inspector tabs."
        )
        self.inspector_more_menu = QtWidgets.QMenu(self.inspector_more_button)
        self.inspector_add_parent_action = self.inspector_more_menu.addAction(
            hybrid_icon("new"),
            "Add Parent…",
        )
        self.inspector_add_parent_action.setCheckable(True)
        self.inspector_add_parent_action.setEnabled(False)
        self.inspector_add_parent_action.triggered.connect(
            self._toggle_add_user_parent_panel
        )
        self.inspector_more_menu.addSeparator()
        self.inspector_toggle_details_action = self.inspector_more_menu.addAction(
            "Show explicit override controls"
        )
        self.inspector_toggle_details_action.triggered.connect(
            self._toggle_current_inspector_details
        )
        self.inspector_reset_overrides_action = self.inspector_more_menu.addAction(
            "Reset render overrides…"
        )
        self.inspector_reset_overrides_action.triggered.connect(
            self._open_render_override_reset
        )
        self.inspector_more_button.setMenu(self.inspector_more_menu)
        self.inspector_more_button.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        footer_layout.addWidget(self.inspector_more_button)
        layout.addWidget(self.inspector_footer)
        self.inspector_tabs.currentChanged.connect(
            self._sync_inspector_footer_for_tab
        )
        self._sync_inspector_footer_for_tab(self.inspector_tabs.currentIndex())
        return inspector

    def _sync_inspector_footer_for_tab(self, index: int) -> None:
        tab = next(
            (
                tab_id
                for tab_id, tab_index in self._inspector_index.items()
                if tab_index == int(index)
            ),
            "text",
        )
        self.inspector_footer.setVisible(tab not in {"cleanup", "history"})
        label, description = editor_preview_action(tab)
        self.rerender_button.setText(label)
        self.rerender_button.setAccessibleName(label)
        self.rerender_button.setAccessibleDescription(description)
        self.rerender_button.setToolTip(description)
        self.target_restore_button.setVisible(tab == "text")
        self.inspector_more_button.setAccessibleName(
            f"More {tab} actions"
        )
        self.inspector_toggle_details_action.setEnabled(
            tab in {"style", "layout"}
        )
        details_visible = False
        if tab == "style":
            details_visible = any(
                not card.isHidden() for card in self._style_detail_cards
            )
        elif tab == "layout":
            details_visible = any(
                not card.isHidden() for card in self._layout_detail_cards
            )
        self.inspector_toggle_details_action.setText(
            "Hide explicit override controls"
            if details_visible
            else "Show explicit override controls"
        )
        self._sync_cleanup_overlay_visibility()

    def _toggle_current_inspector_details(self) -> None:
        index = self.inspector_tabs.currentIndex()
        if index == self._inspector_index.get("style"):
            self.style_more_button.toggle()
        elif index == self._inspector_index.get("layout"):
            self.layout_more_button.toggle()

    def _toggle_add_user_parent_panel(self, requested: bool) -> None:
        """Reveal the hierarchy command without displacing selected-parent facts."""

        if not requested and (self._add_user_parent_draft_present or self._add_user_parent_busy):
            requested = True
        self._add_user_parent_panel_requested = bool(requested)
        self._sync_add_user_parent_panel_visibility()
        if not self.add_user_parent_card.isVisible():
            return
        self.show_inspector_tab("text")
        self.text_inspector_scroll.ensureWidgetVisible(
            self.add_user_parent_card,
            12,
            12,
        )
        if self.add_user_parent_role.isEnabled():
            self.add_user_parent_role.setFocus(
                QtCore.Qt.FocusReason.ShortcutFocusReason
            )

    def _sync_add_user_parent_panel_visibility(self) -> None:
        visible = bool(
            self._add_user_parent_panel_requested
            or self._add_user_parent_draft_present
            or self._add_user_parent_busy
        )
        self.add_user_parent_card.setVisible(visible)
        self.inspector_add_parent_action.setChecked(visible)
        self.inspector_add_parent_action.setEnabled(
            bool(self._add_user_parent_editing_enabled and not self._add_user_parent_busy)
        )

    def _dismiss_add_user_parent_panel(self) -> None:
        self._add_user_parent_panel_requested = False
        if not self._add_user_parent_draft_present and not self._add_user_parent_busy:
            self._sync_add_user_parent_panel_visibility()

    def _toggle_text_provenance_details(self) -> None:
        self._text_provenance_details_requested = (
            not self._text_provenance_details_requested
        )
        self._sync_text_provenance_detail_visibility()

    def _sync_text_provenance_detail_visibility(self) -> None:
        requested = bool(self._text_provenance_details_requested)
        self.source_authority_summary.setVisible(
            bool(requested and self._source_provenance_available)
        )
        self.target_authority_summary.setVisible(
            bool(requested and self._target_provenance_available)
        )
        self.provenance_details_button.setText(
            "Hide details" if requested else "Why three?"
        )

    def _open_render_override_reset(self) -> None:
        self.show_inspector_tab("history")
        self.render_override_reset_toggle.setChecked(True)

    def _scroll_tab(self) -> tuple[QtWidgets.QScrollArea, QtWidgets.QVBoxLayout]:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        content = QtWidgets.QWidget()
        content.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(2, 6, 2, 6)
        layout.setSpacing(10)
        scroll.setWidget(content)
        return scroll, layout

    def _build_text_tab(self) -> QtWidgets.QWidget:
        scroll, layout = self._scroll_tab()
        self.text_inspector_scroll = scroll
        # The Text inspector is the densest decision surface in the prototype:
        # source provenance, target provenance, draft actions, and freshness
        # must remain readable together.  Use the semantic compact section gap
        # so the 200% layout does not clip the final target action row.
        layout.setSpacing(metric_pixels("space-3"))
        segment_nav = QtWidgets.QFrame()
        segment_nav.setObjectName("textSegmentNavigation")
        segment_layout = QtWidgets.QHBoxLayout(segment_nav)
        segment_layout.setContentsMargins(0, 0, 0, 0)
        segment_layout.setSpacing(metric_pixels("space-2"))
        self.parent_segment_label = QtWidgets.QLabel("Text segment — / —")
        self.parent_segment_label.setProperty("role", "secondary")
        segment_layout.addWidget(self.parent_segment_label)
        segment_layout.addStretch(1)
        self.previous_parent_button = QtWidgets.QToolButton()
        self.previous_parent_button.setProperty("role", "command")
        self.previous_parent_button.setProperty("variant", "quiet")
        self.previous_parent_button.setIcon(hybrid_icon("caret-left"))
        self.previous_parent_button.setAccessibleName("Previous text segment")
        self.previous_parent_button.clicked.connect(
            lambda: self._activate_parent_offset(-1)
        )
        segment_layout.addWidget(self.previous_parent_button)
        self.next_parent_button = QtWidgets.QToolButton()
        self.next_parent_button.setProperty("role", "command")
        self.next_parent_button.setProperty("variant", "quiet")
        self.next_parent_button.setIcon(hybrid_icon("caret-right"))
        self.next_parent_button.setAccessibleName("Next text segment")
        self.next_parent_button.clicked.connect(
            lambda: self._activate_parent_offset(1)
        )
        segment_layout.addWidget(self.next_parent_button)
        layout.addWidget(segment_nav)
        self.parent_list.currentIndexChanged.connect(
            lambda _index: self._sync_parent_segment()
        )
        self._sync_parent_segment()

        add_parent_card = QtWidgets.QFrame()
        self.add_user_parent_card = add_parent_card
        add_parent_card.setObjectName("addUserParentCard")
        apply_semantic_properties(
            add_parent_card,
            role="panel-raised",
            accessible_name="Add Parent",
            accessible_description=(
                "Create one standalone Dialogue or Caption parent from a page "
                "workflow area. This command changes hierarchy only."
            ),
        )
        add_parent_layout = QtWidgets.QVBoxLayout(add_parent_card)
        add_parent_layout.setContentsMargins(
            metric_pixels("space-3"),
            metric_pixels("space-3"),
            metric_pixels("space-3"),
            metric_pixels("space-3"),
        )
        add_parent_layout.setSpacing(metric_pixels("space-2"))
        add_parent_layout.addWidget(
            SectionHeader(
                "Add Parent",
                subtitle=(
                    "Define an image-backed workflow area for a later OCR revision; "
                    "typing text never creates or renders a parent. Translation, style, "
                    "cleanup, layout, and rendering remain explicit follow-up revisions."
                ),
                parent=add_parent_card,
            )
        )
        add_parent_form = QtWidgets.QFormLayout()
        self.add_user_parent_form = add_parent_form
        add_parent_form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        add_parent_form.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        add_parent_form.setHorizontalSpacing(metric_pixels("space-3"))
        add_parent_form.setVerticalSpacing(metric_pixels("space-2"))

        self.add_user_parent_role = WheelSafeComboBox()
        self.add_user_parent_role.setObjectName("addUserParentRole")
        self.add_user_parent_role.addItem("Choose role", None)
        self.add_user_parent_role.addItem("Dialogue", "speech")
        self.add_user_parent_role.addItem("Caption", "caption")
        self.add_user_parent_role.setAccessibleName("New parent role")
        self.add_user_parent_role.setAccessibleDescription(
            "Choose Dialogue or Caption for the standalone user parent."
        )
        self.add_user_parent_role.setMinimumHeight(
            metric_pixels("target-default")
        )
        self.add_user_parent_role.currentIndexChanged.connect(
            self._add_user_parent_role_value_changed
        )
        role_label = QtWidgets.QLabel("Role")
        role_label.setBuddy(self.add_user_parent_role)
        add_parent_form.addRow(role_label, self.add_user_parent_role)

        self.add_user_parent_workflow_spins: dict[str, QtWidgets.QSpinBox] = {}
        for field_name, label_text, minimum in (
            ("x", "X", -1),
            ("y", "Y", -1),
            ("width", "Width", 0),
            ("height", "Height", 0),
        ):
            spin = WheelSafeSpinBox()
            spin.setObjectName(
                "addUserParent" + field_name.title() + "SpinBox"
            )
            spin.setRange(minimum, 1)
            spin.setSpecialValueText("—")
            spin.setKeyboardTracking(False)
            spin.setMinimumHeight(metric_pixels("target-default"))
            spin.setAccessibleName(f"Workflow area {label_text.lower()}")
            spin.setAccessibleDescription(
                "Page-pixel component of the unapplied workflow area."
            )
            spin.valueChanged.connect(
                self._add_user_parent_workflow_area_value_changed
            )
            label = QtWidgets.QLabel(label_text)
            label.setBuddy(spin)
            add_parent_form.addRow(label, spin)
            self.add_user_parent_workflow_spins[field_name] = spin
        add_parent_layout.addLayout(add_parent_form)

        add_parent_actions = QtWidgets.QHBoxLayout()
        add_parent_actions.setSpacing(metric_pixels("space-2"))
        self.add_user_parent_add_button = CommandButton(
            "Add Parent",
            command_id="add-user-parent",
            variant="primary",
            parent=add_parent_card,
        )
        self.add_user_parent_add_button.setToolTip(
            "Persist only this standalone hierarchy addition. No pipeline stage runs."
        )
        self.add_user_parent_add_button.clicked.connect(
            self.add_user_parent_requested
        )
        self.add_user_parent_add_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        add_parent_actions.addWidget(self.add_user_parent_add_button)
        self.add_user_parent_cancel_button = CommandButton(
            "Cancel",
            command_id="cancel-add-user-parent-draft",
            variant="secondary",
            parent=add_parent_card,
        )
        self.add_user_parent_cancel_button.setToolTip(
            "Discard only the unapplied Add Parent draft."
        )
        self.add_user_parent_cancel_button.clicked.connect(
            self.add_user_parent_cancel_requested
        )
        self.add_user_parent_cancel_button.clicked.connect(
            self._dismiss_add_user_parent_panel
        )
        self.add_user_parent_cancel_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        add_parent_actions.addWidget(self.add_user_parent_cancel_button)
        add_parent_actions.addStretch(1)
        add_parent_layout.addLayout(add_parent_actions)
        self.add_user_parent_status = QtWidgets.QLabel(
            "Open a saved page to add a standalone parent."
        )
        self.add_user_parent_status.setWordWrap(True)
        apply_semantic_properties(
            self.add_user_parent_status,
            role="secondary",
            tone="muted",
            accessible_name="Add Parent status",
            accessible_description=(
                "Hierarchy-only status. No source, translation, style, cleanup, "
                "layout, or rendering command is implied."
            ),
        )
        add_parent_layout.addWidget(self.add_user_parent_status)
        layout.addWidget(add_parent_card)
        add_parent_card.setVisible(False)
        QtWidgets.QWidget.setTabOrder(
            self.add_user_parent_role,
            self.add_user_parent_workflow_spins["x"],
        )
        QtWidgets.QWidget.setTabOrder(
            self.add_user_parent_workflow_spins["x"],
            self.add_user_parent_workflow_spins["y"],
        )
        QtWidgets.QWidget.setTabOrder(
            self.add_user_parent_workflow_spins["y"],
            self.add_user_parent_workflow_spins["width"],
        )
        QtWidgets.QWidget.setTabOrder(
            self.add_user_parent_workflow_spins["width"],
            self.add_user_parent_workflow_spins["height"],
        )
        QtWidgets.QWidget.setTabOrder(
            self.add_user_parent_workflow_spins["height"],
            self.add_user_parent_add_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.add_user_parent_add_button,
            self.add_user_parent_cancel_button,
        )

        authority_heading = QtWidgets.QHBoxLayout()
        authority_title = QtWidgets.QLabel("Authority comparison")
        authority_title.setProperty("role", "section")
        authority_heading.addWidget(authority_title)
        authority_heading.addStretch(1)
        self.provenance_details_button = QtWidgets.QPushButton("Why three?")
        self.provenance_details_button.setObjectName("provenanceDetailsButton")
        self.provenance_details_button.setProperty("role", "command")
        self.provenance_details_button.setProperty("variant", "quiet")
        self.provenance_details_button.setAccessibleName(
            "Explain source and target authority"
        )
        self.provenance_details_button.setAccessibleDescription(
            "Show the immutable Automatic, selected model, Your edit, and Effective provenance details."
        )
        self.provenance_details_button.clicked.connect(
            self._toggle_text_provenance_details
        )
        authority_heading.addWidget(self.provenance_details_button)
        layout.addLayout(authority_heading)
        comparison = QtWidgets.QHBoxLayout()
        self.target_authority_comparison_layout = comparison
        self.automatic_text = self._authority_card(comparison, "AUTOMATIC", "automatic")
        self.user_text = self._authority_card(comparison, "YOUR EDIT", "user")
        self.effective_text = self._authority_card(comparison, "EFFECTIVE RESULT", "effective")
        layout.addLayout(comparison)
        layout.addSpacing(metric_pixels("space-2"))
        self.source_heading_layout = QtWidgets.QHBoxLayout()
        self.source_heading_layout.setSpacing(metric_pixels("space-2"))
        self.source_field_label = QtWidgets.QLabel("Source (Japanese)")
        self.source_field_label.setProperty("role", "eyebrow")
        self.source_heading_layout.addWidget(self.source_field_label)
        self.source_heading_layout.addStretch(1)
        layout.addLayout(self.source_heading_layout)
        self.source_authority_summary = QtWidgets.QLabel(
            "Automatic OCR: —\nYour edit: No edit"
        )
        self.source_authority_summary.setWordWrap(True)
        self.source_authority_summary.setProperty("role", "secondary")
        self.source_authority_summary.setAccessibleName(
            "Source text provenance comparison"
        )
        self.source_authority_summary.setAccessibleDescription(
            "Compares immutable Automatic source, the selected model OCR revision, "
            "Your edit, and the Effective source without merging their provenance."
        )
        self.source_authority_summary.setVisible(False)
        layout.addWidget(self.source_authority_summary)
        ocr_actions = QtWidgets.QHBoxLayout()
        ocr_actions.setSpacing(6)
        self.ocr_rerun_button = QtWidgets.QPushButton("Rerun OCR")
        self.ocr_rerun_button.setObjectName("sourceRerunLink")
        self.ocr_rerun_button.setProperty("role", "command")
        self.ocr_rerun_button.setProperty("variant", "quiet")
        self.ocr_rerun_button.setAccessibleName(
            "Rerun OCR for the selected user parent"
        )
        self.ocr_rerun_button.setAccessibleDescription(
            "Creates a selected model OCR revision from the exact workflow area. "
            "It does not create Automatic source or a user source edit, and it "
            "does not run translation or later stages."
        )
        self.ocr_rerun_button.setToolTip(
            "Run only the selected OCR owner and publish a new model source revision."
        )
        self.ocr_rerun_button.setEnabled(False)
        self.ocr_rerun_button.clicked.connect(self.ocr_revision_requested)
        self.source_heading_layout.addWidget(self.ocr_rerun_button)
        self.ocr_cancel_button = QtWidgets.QPushButton("Cancel OCR")
        self.ocr_cancel_button.setProperty("role", "command")
        self.ocr_cancel_button.setProperty("variant", "secondary")
        self.ocr_cancel_button.setAccessibleName(
            "Cancel the selected model OCR revision"
        )
        self.ocr_cancel_button.setAccessibleDescription(
            "Requests cooperative cancellation. Active OCR inference may finish "
            "before its result is discarded; no revision is then published."
        )
        self.ocr_cancel_button.setToolTip(
            "Request cancellation. Non-preemptive inference may finish before discard."
        )
        self.ocr_cancel_button.setEnabled(False)
        self.ocr_cancel_button.setVisible(False)
        self.ocr_cancel_button.clicked.connect(
            self.ocr_revision_cancel_requested
        )
        ocr_actions.addWidget(self.ocr_cancel_button, 1)
        layout.addLayout(ocr_actions)
        self.ocr_revision_status = QtWidgets.QLabel(
            "Select a pending user parent to run OCR"
        )
        self.ocr_revision_status.setProperty("role", "secondary")
        self.ocr_revision_status.setProperty("tone", "muted")
        self.ocr_revision_status.setWordWrap(True)
        self.ocr_revision_status.setAccessibleName("OCR revision status")
        self.ocr_revision_status.setAccessibleDescription(
            "Reports ready, active, deferred cancellation, failure, stale, and "
            "recovery states for the selected model OCR revision."
        )
        self.ocr_revision_status.setVisible(False)
        layout.addWidget(self.ocr_revision_status)
        self.source_text = _ExactTextEdit()
        self.source_text.setAccessibleName("Source text editor")
        self.source_text.setAccessibleDescription(
            "Edit the exact selected-parent source text. Apply publishes a "
            "typed source edit; OCR, translation, cleanup, and Preview remain "
            "separate explicit commands."
        )
        self.source_text.setPlaceholderText("Enter the exact source text")
        self.source_text.setMaximumHeight(64)
        self.source_text.setTabChangesFocus(True)
        self.source_text.setEnabled(False)
        self.source_text.textChanged.connect(
            lambda: self.source_text_draft_changed.emit(
                self.source_text.exact_text()
            )
        )
        self.source_authority_badge = QtWidgets.QLabel("Automatic OCR")
        self.source_authority_badge.setObjectName("textAuthorityBadge")
        self.source_authority_badge.setProperty("authority", "automatic")
        self.source_authority_badge.setAccessibleName("Source text authority")
        self.source_text_frame = _BadgeTextEditFrame(
            self.source_text,
            self.source_authority_badge,
        )
        layout.addWidget(self.source_text_frame)
        source_actions = QtWidgets.QGridLayout()
        source_actions.setHorizontalSpacing(6)
        source_actions.setVerticalSpacing(6)
        self.source_apply_button = QtWidgets.QPushButton("Apply Source")
        self.source_apply_button.setProperty("role", "command")
        self.source_apply_button.setProperty("variant", "primary")
        self.source_apply_button.setAccessibleName("Apply source text edit")
        self.source_apply_button.setToolTip(
            "Save the exact source text as a user edit. No module reruns automatically."
        )
        self.source_apply_button.setEnabled(False)
        self.source_apply_button.setVisible(False)
        self.source_apply_button.clicked.connect(self.source_text_apply_requested)
        source_actions.addWidget(self.source_apply_button, 0, 0)
        self.source_cancel_button = QtWidgets.QPushButton("Cancel")
        self.source_cancel_button.setProperty("role", "command")
        self.source_cancel_button.setProperty("variant", "secondary")
        self.source_cancel_button.setAccessibleName("Cancel source text draft")
        self.source_cancel_button.setToolTip(
            "Discard only the unapplied source-text draft."
        )
        self.source_cancel_button.setEnabled(False)
        self.source_cancel_button.setVisible(False)
        self.source_cancel_button.clicked.connect(self.source_text_cancel_requested)
        source_actions.addWidget(self.source_cancel_button, 0, 1)
        self.source_restore_button = QtWidgets.QPushButton("Restore Automatic")
        self.source_restore_button.setProperty("role", "command")
        self.source_restore_button.setProperty("variant", "secondary")
        self.source_restore_button.setAccessibleName("Restore automatic source text")
        self.source_restore_button.setToolTip(
            "Publish an explicit restore edit; automatic OCR evidence is not changed."
        )
        self.source_restore_button.setEnabled(False)
        self.source_restore_button.setVisible(False)
        self.source_restore_button.clicked.connect(
            self.source_text_restore_requested
        )
        source_actions.addWidget(self.source_restore_button, 1, 0, 1, 2)
        layout.addLayout(source_actions)
        self.source_edit_status = QtWidgets.QLabel(
            "Select a parent to edit source text"
        )
        self.source_edit_status.setProperty("role", "secondary")
        self.source_edit_status.setProperty("tone", "muted")
        self.source_edit_status.setWordWrap(True)
        self.source_edit_status.setAccessibleName("Source text edit status")
        self.source_edit_status.setVisible(False)
        layout.addWidget(self.source_edit_status)

        self.target_heading_layout = QtWidgets.QHBoxLayout()
        self.target_heading_layout.setSpacing(metric_pixels("space-2"))
        self.target_field_label = QtWidgets.QLabel("Target (Simplified Chinese)")
        self.target_field_label.setProperty("role", "eyebrow")
        self.target_heading_layout.addWidget(self.target_field_label)
        self.target_heading_layout.addStretch(1)
        layout.addLayout(self.target_heading_layout)
        self.target_authority_summary = QtWidgets.QLabel(
            "Automatic target: —\n"
            "Selected model translation revision: Not selected\n"
            "Your edit: No edit\n"
            "Effective target: Unavailable"
        )
        self.target_authority_summary.setWordWrap(True)
        self.target_authority_summary.setProperty("role", "secondary")
        self.target_authority_summary.setAccessibleName(
            "Target text provenance comparison"
        )
        self.target_authority_summary.setAccessibleDescription(
            "Compares immutable Automatic target, the selected model translation "
            "revision, Your edit, and the Effective target without merging provenance."
        )
        self.target_authority_summary.setVisible(False)
        layout.addWidget(self.target_authority_summary)
        self.target_text = _ExactTextEdit()
        self.target_text.setAccessibleName("Target text editor")
        self.target_text.setAccessibleDescription(
            "Edit the exact selected-parent target text. Replace publishes a "
            "typed edit; Preview this page remains a separate command."
        )
        self.target_text.setPlaceholderText("Enter the exact target text")
        self.target_text.setMaximumHeight(64)
        self.target_text.textChanged.connect(
            lambda: self.target_text_draft_changed.emit(
                self.target_text.exact_text()
            )
        )
        layout.addWidget(self.target_text)
        target_actions = QtWidgets.QGridLayout()
        self._target_actions_layout = target_actions
        target_actions.setHorizontalSpacing(6)
        target_actions.setVerticalSpacing(6)
        self.target_apply_button = QtWidgets.QPushButton("Replace")
        self.target_apply_button.setProperty("role", "command")
        self.target_apply_button.setProperty("variant", "primary")
        self.target_apply_button.setAccessibleName("Replace target text")
        self.target_apply_button.setToolTip(
            "Replace the effective target with this exact user edit. Rendering remains explicit."
        )
        self.target_apply_button.setVisible(False)
        self.target_apply_button.clicked.connect(self.target_text_apply_requested)
        target_actions.addWidget(self.target_apply_button, 0, 0)
        self.target_cancel_button = QtWidgets.QPushButton("Cancel")
        self.target_cancel_button.setProperty("role", "command")
        self.target_cancel_button.setProperty("variant", "secondary")
        self.target_cancel_button.setAccessibleName("Cancel target text draft")
        self.target_cancel_button.setToolTip(
            "Discard only the unapplied target-text draft."
        )
        self.target_cancel_button.setVisible(False)
        self.target_cancel_button.clicked.connect(self.target_text_cancel_requested)
        target_actions.addWidget(self.target_cancel_button, 0, 1)
        self.target_restore_button = QtWidgets.QPushButton("Restore Automatic")
        self.target_restore_button.setProperty("role", "command")
        self.target_restore_button.setProperty("variant", "secondary")
        self.target_restore_button.setAccessibleName(
            "Restore automatic target text"
        )
        self.target_restore_button.setToolTip(
            "Publish an explicit restore edit; automatic evidence is not changed."
        )
        self.target_restore_button.clicked.connect(
            self.target_text_restore_requested
        )
        target_actions.addWidget(self.target_restore_button, 1, 0)
        self._target_restore_spans_columns = False
        self.target_keep_existing_button = QtWidgets.QPushButton(
            "Keep Existing Target"
        )
        self.target_keep_existing_button.setProperty("role", "command")
        self.target_keep_existing_button.setProperty("variant", "secondary")
        self.target_keep_existing_button.setAccessibleName(
            "Keep existing stale target as an explicit user edit"
        )
        self.target_keep_existing_button.setToolTip(
            "Acknowledge the current source by publishing the exact historical "
            "target as a typed user edit. Translation does not run."
        )
        self.target_keep_existing_button.clicked.connect(
            self.target_text_keep_existing_requested
        )
        self.target_keep_existing_button.setVisible(False)
        self.target_keep_existing_button.setEnabled(False)
        target_actions.addWidget(self.target_keep_existing_button, 1, 1)
        self.target_text.setTabChangesFocus(True)
        layout.addLayout(target_actions)
        self.target_edit_status = QtWidgets.QLabel("Select a parent to edit target text")
        self.target_edit_status.setProperty("role", "secondary")
        self.target_edit_status.setProperty("tone", "muted")
        self.target_edit_status.setWordWrap(True)
        self.target_edit_status.setAccessibleName("Target text edit status")
        self.target_edit_status.setVisible(False)
        layout.addWidget(self.target_edit_status)
        translation_actions = QtWidgets.QHBoxLayout()
        translation_actions.setSpacing(6)
        self.translation_rerun_button = QtWidgets.QPushButton(
            "Retranslate Parent"
        )
        self.translation_rerun_button.setObjectName("targetRerunLink")
        self.translation_rerun_button.setProperty("role", "command")
        self.translation_rerun_button.setProperty("variant", "quiet")
        self.translation_rerun_button.setAccessibleName(
            "Retranslate the selected user parent"
        )
        self.translation_rerun_button.setAccessibleDescription(
            "Runs only the selected translation owner with the frozen source, "
            "provider, settings, glossary, context, and hierarchy. It publishes "
            "a selected model target revision and does not run later stages."
        )
        self.translation_rerun_button.setToolTip(
            "Run only translation for this exact user parent; later revisions remain explicit."
        )
        self.translation_rerun_button.setEnabled(False)
        self.translation_rerun_button.clicked.connect(
            self.translation_revision_requested
        )
        self.target_heading_layout.addWidget(self.translation_rerun_button)
        self.translation_cancel_button = QtWidgets.QPushButton(
            "Cancel Translation"
        )
        self.translation_cancel_button.setProperty("role", "command")
        self.translation_cancel_button.setProperty("variant", "secondary")
        self.translation_cancel_button.setAccessibleName(
            "Cancel the selected model translation revision"
        )
        self.translation_cancel_button.setAccessibleDescription(
            "Requests deferred cancellation. Active provider inference may finish "
            "before its result is discarded; no revision is then published."
        )
        self.translation_cancel_button.setToolTip(
            "Request cancellation. Non-preemptive inference may finish before discard."
        )
        self.translation_cancel_button.setEnabled(False)
        self.translation_cancel_button.setVisible(False)
        self.translation_cancel_button.clicked.connect(
            self.translation_revision_cancel_requested
        )
        translation_actions.addWidget(self.translation_cancel_button, 1)
        layout.addLayout(translation_actions)
        self.translation_revision_status = QtWidgets.QLabel(
            "Select a source-current user parent to run translation"
        )
        self.translation_revision_status.setProperty("role", "secondary")
        self.translation_revision_status.setProperty("tone", "muted")
        self.translation_revision_status.setWordWrap(True)
        self.translation_revision_status.setAccessibleName(
            "Translation revision status"
        )
        self.translation_revision_status.setAccessibleDescription(
            "Reports ready, active, deferred cancellation, failure, stale, and "
            "recovery states for the selected model translation revision."
        )
        self.translation_revision_status.setVisible(False)
        layout.addWidget(self.translation_revision_status)
        QtWidgets.QWidget.setTabOrder(self.source_text, self.source_apply_button)
        QtWidgets.QWidget.setTabOrder(
            self.source_apply_button,
            self.source_cancel_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.source_cancel_button,
            self.source_restore_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.source_restore_button,
            self.ocr_rerun_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.ocr_rerun_button,
            self.ocr_cancel_button,
        )
        QtWidgets.QWidget.setTabOrder(self.ocr_cancel_button, self.target_text)
        QtWidgets.QWidget.setTabOrder(self.target_text, self.target_apply_button)
        QtWidgets.QWidget.setTabOrder(
            self.target_apply_button,
            self.target_cancel_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.target_cancel_button,
            self.target_restore_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.target_restore_button,
            self.target_keep_existing_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.target_keep_existing_button,
            self.translation_rerun_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.translation_rerun_button,
            self.translation_cancel_button,
        )
        self.text_freshness = QtWidgets.QLabel("Freshness: Current")
        self.text_freshness.setProperty("role", "secondary")
        self.text_freshness.setWordWrap(True)
        self.text_freshness.setVisible(False)
        layout.addWidget(self.text_freshness)
        layout.addWidget(self.parent_membership_frame)
        layout.addStretch(1)
        return scroll

    @QtCore.Slot(int)
    def _add_user_parent_role_value_changed(self, _index: int) -> None:
        if self._add_user_parent_programmatic_update:
            return
        self.add_user_parent_role_changed.emit(
            self.add_user_parent_role.currentData()
        )
        self._sync_add_user_parent_canvas_draft()

    @QtCore.Slot(int)
    def _add_user_parent_workflow_area_value_changed(self, _value: int) -> None:
        if self._add_user_parent_programmatic_update:
            return
        self.add_user_parent_workflow_area_changed.emit(
            self._add_user_parent_partial_workflow_area()
        )
        self._sync_add_user_parent_canvas_draft()

    def _add_user_parent_partial_workflow_area(
        self,
    ) -> tuple[int | None, int | None, int | None, int | None] | None:
        values: list[int | None] = []
        for field_name in ("x", "y", "width", "height"):
            spin = self.add_user_parent_workflow_spins[field_name]
            values.append(None if spin.value() == spin.minimum() else spin.value())
        if all(value is None for value in values):
            return None
        return tuple(values)  # type: ignore[return-value]

    def _sync_add_user_parent_canvas_draft(self) -> None:
        partial = self._add_user_parent_partial_workflow_area()
        role = self.add_user_parent_role.currentData()
        bbox: tuple[int, int, int, int] | None = None
        if (
            self._add_user_parent_show_canvas_draft
            and partial is not None
            and all(value is not None for value in partial)
        ):
            x, y, width, height = partial
            assert x is not None and y is not None
            assert width is not None and height is not None
            page_width, page_height = self._add_user_parent_canvas_size
            if (
                x >= 0
                and y >= 0
                and width > 0
                and height > 0
                and x + width <= page_width
                and y + height <= page_height
            ):
                bbox = (x, y, width, height)
        self.canvas.set_workflow_area_draft(
            bbox,
            role=role if role in {"speech", "caption"} else "",
        )

    def set_add_user_parent_editor_state(
        self,
        *,
        draft_role: str | None,
        draft_workflow_area_bbox: (
            tuple[int | None, int | None, int | None, int | None] | None
        ),
        canvas_size: tuple[int, int] | None,
        editing_enabled: bool,
        add_enabled: bool,
        cancel_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one page-bound Add Parent draft without dispatching work."""

        if draft_role not in {None, "speech", "caption"}:
            raise ValueError("draft_role must be speech, caption, or None")
        if canvas_size is None:
            stable_canvas_size = (1, 1)
        elif (
            not isinstance(canvas_size, tuple)
            or len(canvas_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in canvas_size
            )
        ):
            raise ValueError("canvas_size must be a positive integer pair or None")
        else:
            stable_canvas_size = canvas_size
        if draft_workflow_area_bbox is not None and (
            not isinstance(draft_workflow_area_bbox, tuple)
            or len(draft_workflow_area_bbox) != 4
            or any(
                value is not None
                and (isinstance(value, bool) or not isinstance(value, int))
                for value in draft_workflow_area_bbox
            )
        ):
            raise ValueError(
                "draft_workflow_area_bbox must be four integer-or-None values"
            )

        self._add_user_parent_canvas_size = stable_canvas_size
        self._add_user_parent_programmatic_update = True
        try:
            role_index = self.add_user_parent_role.findData(draft_role)
            self.add_user_parent_role.setCurrentIndex(max(0, role_index))
            partial = draft_workflow_area_bbox or (None, None, None, None)
            page_width, page_height = stable_canvas_size
            ranges = {
                "x": (-1, max(0, page_width - 1)),
                "y": (-1, max(0, page_height - 1)),
                "width": (0, page_width),
                "height": (0, page_height),
            }
            for field_name, value in zip(
                ("x", "y", "width", "height"),
                partial,
            ):
                spin = self.add_user_parent_workflow_spins[field_name]
                minimum, maximum = ranges[field_name]
                spin.setRange(minimum, maximum)
                spin.setValue(minimum if value is None else value)
        finally:
            self._add_user_parent_programmatic_update = False

        enabled = bool(editing_enabled and not busy)
        self._add_user_parent_editing_enabled = bool(editing_enabled)
        self._add_user_parent_busy = bool(busy)
        self._add_user_parent_draft_present = bool(
            draft_role is not None
            or (
                draft_workflow_area_bbox is not None
                and any(value is not None for value in draft_workflow_area_bbox)
            )
        )
        self._add_user_parent_show_canvas_draft = bool(editing_enabled or busy)
        self.add_user_parent_role.setEnabled(enabled)
        for spin in self.add_user_parent_workflow_spins.values():
            spin.setEnabled(enabled)
        self.add_user_parent_add_button.setEnabled(bool(add_enabled and not busy))
        self.add_user_parent_cancel_button.setEnabled(bool(cancel_enabled))
        self.add_user_parent_status.setText(str(status_text))
        apply_semantic_properties(
            self.add_user_parent_status,
            role="secondary",
            tone=str(status_tone),
            accessible_name="Add Parent status",
            accessible_description=(
                str(status_text)
                + " This is a hierarchy-only command; no source, translation, "
                "style, cleanup, layout, or rendering stage runs automatically."
            ),
        )
        self._sync_add_user_parent_canvas_draft()
        self._sync_add_user_parent_panel_visibility()

    @staticmethod
    def _authority_card(
        layout: QtWidgets.QHBoxLayout,
        title: str,
        authority: str,
    ) -> QtWidgets.QLabel:
        frame = QtWidgets.QFrame()
        frame.setObjectName("authorityCard")
        frame.setProperty("role", "panel-raised")
        frame.setProperty("authority", authority)
        frame_layout = QtWidgets.QVBoxLayout(frame)
        frame_layout.setContentsMargins(8, 8, 8, 8)
        label = QtWidgets.QLabel(title)
        label.setProperty("role", "eyebrow")
        value = QtWidgets.QLabel("—")
        value.setWordWrap(True)
        value.setMinimumHeight(62)
        frame_layout.addWidget(label)
        frame_layout.addWidget(value, 1)
        layout.addWidget(frame, 1)
        return value

    @staticmethod
    def _facade_readout(
        accessible_name: str,
        *,
        suffix: str = "",
    ) -> QtWidgets.QLineEdit:
        field = QtWidgets.QLineEdit()
        field.setObjectName("resolvedEvidenceField")
        field.setReadOnly(True)
        # These values are evidence summaries, not editing controls.  Keeping
        # them out of the keyboard chain prevents the compact facade from
        # intercepting Tab before the real style/layout commands below it.
        field.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        field.setAccessibleName(accessible_name)
        field.setProperty("suffix", suffix)
        return field

    def _build_style_tab(self) -> QtWidgets.QWidget:
        scroll, layout = self._scroll_tab()
        font_role_card = QtWidgets.QFrame()
        font_role_card.setObjectName("fontRoleCard")
        font_role_card.setProperty("role", "panel-raised")
        font_role_layout = QtWidgets.QVBoxLayout(font_role_card)
        font_role_layout.setContentsMargins(8, 8, 8, 8)
        font_role_layout.setSpacing(6)
        font_role_title = QtWidgets.QLabel("Registered Font Role")
        font_role_title.setProperty("role", "section")
        font_role_layout.addWidget(font_role_title)
        font_role_help = QtWidgets.QLabel(
            "Choose one existing CJK font role for the selected render-required "
            "parent. Set and Restore Automatic publish only this style field; "
            "font discovery and Preview remain separate."
        )
        font_role_help.setWordWrap(True)
        font_role_help.setProperty("role", "secondary")
        font_role_layout.addWidget(font_role_help)
        font_role_summary = QtWidgets.QFormLayout()
        font_role_summary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        font_role_summary.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.font_role_automatic = QtWidgets.QLabel("—")
        self.font_role_automatic.setWordWrap(True)
        self.font_role_automatic.setAccessibleName("Automatic registered font role")
        font_role_summary.addRow("Automatic", self.font_role_automatic)
        self.font_role_user = QtWidgets.QLabel("No edit")
        self.font_role_user.setWordWrap(True)
        self.font_role_user.setAccessibleName("User registered font-role edit")
        font_role_summary.addRow("Your edit", self.font_role_user)
        self.font_role_effective = QtWidgets.QLabel("—")
        self.font_role_effective.setWordWrap(True)
        self.font_role_effective.setAccessibleName("Effective registered font role")
        font_role_summary.addRow("Effective", self.font_role_effective)
        self.font_role_authority = QtWidgets.QLabel("—")
        self.font_role_authority.setWordWrap(True)
        self.font_role_authority.setAccessibleName("Font-role authority")
        font_role_summary.addRow("Authority", self.font_role_authority)
        font_role_layout.addLayout(font_role_summary)
        self.font_role_choice = WheelSafeComboBox()
        self.font_role_choice.setObjectName("fontRoleChoice")
        self.font_role_choice.setAccessibleName("Selected parent registered font role")
        self.font_role_choice.setAccessibleDescription(
            "Choose one existing sans or serif CJK font role. No font is loaded "
            "or discovered by this control."
        )
        for role_id, label in (
            ("sans_regular", "Sans Regular"),
            ("sans_medium", "Sans Medium"),
            ("sans_bold", "Sans Bold"),
            ("sans_black", "Sans Black"),
            ("serif_regular", "Serif Regular"),
            ("serif_semibold", "Serif Semibold"),
            ("serif_bold", "Serif Bold"),
        ):
            self.font_role_choice.addItem(label, role_id)
        self.font_role_choice.currentIndexChanged.connect(
            self._font_role_value_changed
        )
        font_role_layout.addWidget(self.font_role_choice)
        font_role_actions = QtWidgets.QHBoxLayout()
        self.font_role_set_button = QtWidgets.QPushButton("Set")
        self.font_role_set_button.setObjectName("fontRoleSetButton")
        self.font_role_set_button.setProperty("role", "command")
        self.font_role_set_button.setProperty("variant", "primary")
        self.font_role_set_button.setAccessibleName(
            "Set selected parent registered font role"
        )
        self.font_role_set_button.setToolTip(
            "Publish only this registered font-role edit. Rendering does not start."
        )
        self.font_role_set_button.clicked.connect(self.font_role_apply_requested)
        font_role_actions.addWidget(self.font_role_set_button)
        self.font_role_cancel_button = QtWidgets.QPushButton("Cancel")
        self.font_role_cancel_button.setObjectName("fontRoleCancelButton")
        self.font_role_cancel_button.setProperty("role", "command")
        self.font_role_cancel_button.setProperty("variant", "secondary")
        self.font_role_cancel_button.setAccessibleName("Cancel font-role draft")
        self.font_role_cancel_button.setToolTip(
            "Discard only the unapplied registered font role."
        )
        self.font_role_cancel_button.clicked.connect(
            self.font_role_cancel_requested
        )
        font_role_actions.addWidget(self.font_role_cancel_button)
        self.font_role_restore_button = QtWidgets.QPushButton("Restore Automatic")
        self.font_role_restore_button.setObjectName("fontRoleRestoreButton")
        self.font_role_restore_button.setProperty("role", "command")
        self.font_role_restore_button.setProperty("variant", "secondary")
        self.font_role_restore_button.setAccessibleName(
            "Restore automatic registered font role"
        )
        self.font_role_restore_button.setToolTip(
            "Remove only the selected parent's font-role override. Automatic "
            "style evidence remains immutable."
        )
        self.font_role_restore_button.clicked.connect(
            self.font_role_restore_requested
        )
        font_role_actions.addWidget(self.font_role_restore_button)
        font_role_layout.addLayout(font_role_actions)
        self.font_role_status = QtWidgets.QLabel(
            "Select a render-required parent to edit font role"
        )
        self.font_role_status.setProperty("role", "secondary")
        self.font_role_status.setProperty("tone", "muted")
        self.font_role_status.setWordWrap(True)
        self.font_role_status.setAccessibleName("Font-role edit status")
        font_role_layout.addWidget(self.font_role_status)
        QtWidgets.QWidget.setTabOrder(
            self.font_role_choice,
            self.font_role_set_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.font_role_set_button,
            self.font_role_cancel_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.font_role_cancel_button,
            self.font_role_restore_button,
        )
        layout.addWidget(font_role_card)
        font_weight_tier_card = QtWidgets.QFrame()
        font_weight_tier_card.setObjectName("fontWeightTierCard")
        font_weight_tier_card.setProperty("role", "panel-raised")
        font_weight_tier_layout = QtWidgets.QVBoxLayout(font_weight_tier_card)
        font_weight_tier_layout.setContentsMargins(8, 8, 8, 8)
        font_weight_tier_layout.setSpacing(6)
        font_weight_tier_title = QtWidgets.QLabel("Registered Font Weight")
        font_weight_tier_title.setProperty("role", "section")
        font_weight_tier_layout.addWidget(font_weight_tier_title)
        font_weight_tier_help = QtWidgets.QLabel(
            "Choose one existing registered weight tier within the automatic "
            "font family. Set and Restore Automatic publish only this style "
            "field; font discovery and Preview remain separate."
        )
        font_weight_tier_help.setWordWrap(True)
        font_weight_tier_help.setProperty("role", "secondary")
        font_weight_tier_layout.addWidget(font_weight_tier_help)
        font_weight_tier_summary = QtWidgets.QFormLayout()
        font_weight_tier_summary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        font_weight_tier_summary.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.font_weight_tier_automatic = QtWidgets.QLabel("—")
        self.font_weight_tier_automatic.setWordWrap(True)
        self.font_weight_tier_automatic.setAccessibleName(
            "Automatic registered font weight"
        )
        font_weight_tier_summary.addRow(
            "Automatic", self.font_weight_tier_automatic
        )
        self.font_weight_tier_user = QtWidgets.QLabel("No edit")
        self.font_weight_tier_user.setWordWrap(True)
        self.font_weight_tier_user.setAccessibleName(
            "User registered font-weight edit"
        )
        font_weight_tier_summary.addRow("Your edit", self.font_weight_tier_user)
        self.font_weight_tier_effective = QtWidgets.QLabel("—")
        self.font_weight_tier_effective.setWordWrap(True)
        self.font_weight_tier_effective.setAccessibleName(
            "Effective registered font weight"
        )
        font_weight_tier_summary.addRow(
            "Effective", self.font_weight_tier_effective
        )
        self.font_weight_tier_authority = QtWidgets.QLabel("—")
        self.font_weight_tier_authority.setWordWrap(True)
        self.font_weight_tier_authority.setAccessibleName("Font-weight authority")
        font_weight_tier_summary.addRow(
            "Authority", self.font_weight_tier_authority
        )
        font_weight_tier_layout.addLayout(font_weight_tier_summary)
        self.font_weight_tier_choice = WheelSafeComboBox()
        self.font_weight_tier_choice.setObjectName("fontWeightTierChoice")
        self.font_weight_tier_choice.setAccessibleName(
            "Selected parent registered font weight"
        )
        self.font_weight_tier_choice.setAccessibleDescription(
            "Choose Slender, Base, Emphasis, or Heavy within the automatic "
            "registered family. No font is loaded or discovered by this control."
        )
        for tier_id, label in (
            ("slender", "Slender"),
            ("base", "Base"),
            ("emphasis", "Emphasis"),
            ("heavy", "Heavy"),
        ):
            self.font_weight_tier_choice.addItem(label, tier_id)
        self.font_weight_tier_choice.currentIndexChanged.connect(
            self._font_weight_tier_value_changed
        )
        font_weight_tier_layout.addWidget(self.font_weight_tier_choice)
        font_weight_tier_actions = QtWidgets.QHBoxLayout()
        self.font_weight_tier_set_button = QtWidgets.QPushButton("Set")
        self.font_weight_tier_set_button.setObjectName("fontWeightTierSetButton")
        self.font_weight_tier_set_button.setProperty("role", "command")
        self.font_weight_tier_set_button.setProperty("variant", "primary")
        self.font_weight_tier_set_button.setAccessibleName(
            "Set selected parent registered font weight"
        )
        self.font_weight_tier_set_button.setToolTip(
            "Publish only this registered font-weight edit. Rendering does not start."
        )
        self.font_weight_tier_set_button.clicked.connect(
            self.font_weight_tier_apply_requested
        )
        font_weight_tier_actions.addWidget(self.font_weight_tier_set_button)
        self.font_weight_tier_cancel_button = QtWidgets.QPushButton("Cancel")
        self.font_weight_tier_cancel_button.setObjectName(
            "fontWeightTierCancelButton"
        )
        self.font_weight_tier_cancel_button.setProperty("role", "command")
        self.font_weight_tier_cancel_button.setProperty("variant", "secondary")
        self.font_weight_tier_cancel_button.setAccessibleName(
            "Cancel font-weight draft"
        )
        self.font_weight_tier_cancel_button.setToolTip(
            "Discard only the unapplied registered font weight."
        )
        self.font_weight_tier_cancel_button.clicked.connect(
            self.font_weight_tier_cancel_requested
        )
        font_weight_tier_actions.addWidget(self.font_weight_tier_cancel_button)
        self.font_weight_tier_restore_button = QtWidgets.QPushButton(
            "Restore Automatic"
        )
        self.font_weight_tier_restore_button.setObjectName(
            "fontWeightTierRestoreButton"
        )
        self.font_weight_tier_restore_button.setProperty("role", "command")
        self.font_weight_tier_restore_button.setProperty("variant", "secondary")
        self.font_weight_tier_restore_button.setAccessibleName(
            "Restore automatic registered font weight"
        )
        self.font_weight_tier_restore_button.setToolTip(
            "Remove only the selected parent's font-weight override. Automatic "
            "style evidence remains immutable."
        )
        self.font_weight_tier_restore_button.clicked.connect(
            self.font_weight_tier_restore_requested
        )
        font_weight_tier_actions.addWidget(self.font_weight_tier_restore_button)
        font_weight_tier_layout.addLayout(font_weight_tier_actions)
        self.font_weight_tier_status = QtWidgets.QLabel(
            "Select a render-required parent to edit font weight"
        )
        self.font_weight_tier_status.setProperty("role", "secondary")
        self.font_weight_tier_status.setProperty("tone", "muted")
        self.font_weight_tier_status.setWordWrap(True)
        self.font_weight_tier_status.setAccessibleName("Font-weight edit status")
        font_weight_tier_layout.addWidget(self.font_weight_tier_status)
        QtWidgets.QWidget.setTabOrder(
            self.font_weight_tier_choice,
            self.font_weight_tier_set_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.font_weight_tier_set_button,
            self.font_weight_tier_cancel_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.font_weight_tier_cancel_button,
            self.font_weight_tier_restore_button,
        )
        layout.addWidget(font_weight_tier_card)
        QtWidgets.QWidget.setTabOrder(
            self.font_role_restore_button,
            self.font_weight_tier_choice,
        )
        fill_color_card = QtWidgets.QFrame()
        fill_color_card.setObjectName("fillColorCard")
        fill_color_card.setProperty("role", "panel-raised")
        fill_color_layout = QtWidgets.QVBoxLayout(fill_color_card)
        fill_color_layout.setContentsMargins(8, 8, 8, 8)
        fill_color_layout.setSpacing(6)
        fill_color_title = QtWidgets.QLabel("Opaque Fill Color")
        fill_color_title.setProperty("role", "section")
        fill_color_layout.addWidget(fill_color_title)
        fill_color_help = QtWidgets.QLabel(
            "Set one exact opaque #RRGGBB text color for the selected parent. "
            "Alpha values are shown as unsupported and are never coerced. Set "
            "and Restore Automatic publish only this style field; Preview remains "
            "explicit."
        )
        fill_color_help.setWordWrap(True)
        fill_color_help.setProperty("role", "secondary")
        fill_color_layout.addWidget(fill_color_help)
        fill_color_summary = QtWidgets.QFormLayout()
        fill_color_summary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        fill_color_summary.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.fill_color_automatic = QtWidgets.QLabel("—")
        self.fill_color_automatic.setWordWrap(True)
        self.fill_color_automatic.setAccessibleName("Automatic opaque fill color")
        fill_color_summary.addRow("Automatic", self.fill_color_automatic)
        self.fill_color_user = QtWidgets.QLabel("No edit")
        self.fill_color_user.setWordWrap(True)
        self.fill_color_user.setAccessibleName("User opaque fill-color edit")
        fill_color_summary.addRow("Your edit", self.fill_color_user)
        self.fill_color_effective = QtWidgets.QLabel("—")
        self.fill_color_effective.setWordWrap(True)
        self.fill_color_effective.setAccessibleName("Effective opaque fill color")
        fill_color_summary.addRow("Effective", self.fill_color_effective)
        self.fill_color_authority = QtWidgets.QLabel("—")
        self.fill_color_authority.setWordWrap(True)
        self.fill_color_authority.setAccessibleName("Fill-color authority")
        fill_color_summary.addRow("Authority", self.fill_color_authority)
        fill_color_layout.addLayout(fill_color_summary)
        self.fill_color_swatch = QtWidgets.QFrame()
        self.fill_color_swatch.setObjectName("fillColorSwatch")
        self.fill_color_swatch.setProperty("role", "color-swatch")
        self.fill_color_swatch.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.fill_color_swatch.setMinimumSize(48, 48)
        self.fill_color_swatch.setMaximumHeight(64)
        self.fill_color_swatch.setAccessibleName(
            "Opaque fill-color preview swatch"
        )
        fill_color_layout.addWidget(self.fill_color_swatch)
        self.fill_color_edit = QtWidgets.QLineEdit()
        self.fill_color_edit.setObjectName("fillColorHexEdit")
        self.fill_color_edit.setMaxLength(64)
        self.fill_color_edit.setPlaceholderText("#RRGGBB")
        self.fill_color_edit.setAccessibleName(
            "Selected parent opaque fill color"
        )
        self.fill_color_edit.setAccessibleDescription(
            "Enter exactly a hash followed by six hexadecimal digits. Alpha "
            "channels are unsupported."
        )
        self.fill_color_edit.textChanged.connect(
            self._fill_color_value_changed
        )
        fill_color_layout.addWidget(self.fill_color_edit)
        self.fill_color_set_button = QtWidgets.QPushButton("Set")
        self.fill_color_set_button.setObjectName("fillColorSetButton")
        self.fill_color_set_button.setProperty("role", "command")
        self.fill_color_set_button.setProperty("variant", "primary")
        self.fill_color_set_button.setAccessibleName(
            "Set selected parent opaque fill color"
        )
        self.fill_color_set_button.setToolTip(
            "Publish only this opaque fill-color edit. Rendering does not start."
        )
        self.fill_color_set_button.clicked.connect(
            self.fill_color_apply_requested
        )
        fill_color_layout.addWidget(self.fill_color_set_button)
        self.fill_color_cancel_button = QtWidgets.QPushButton("Cancel")
        self.fill_color_cancel_button.setObjectName("fillColorCancelButton")
        self.fill_color_cancel_button.setProperty("role", "command")
        self.fill_color_cancel_button.setProperty("variant", "secondary")
        self.fill_color_cancel_button.setAccessibleName(
            "Cancel opaque fill-color draft"
        )
        self.fill_color_cancel_button.setToolTip(
            "Discard only the unapplied opaque fill color."
        )
        self.fill_color_cancel_button.clicked.connect(
            self.fill_color_cancel_requested
        )
        fill_color_layout.addWidget(self.fill_color_cancel_button)
        self.fill_color_restore_button = QtWidgets.QPushButton(
            "Restore Automatic"
        )
        self.fill_color_restore_button.setObjectName("fillColorRestoreButton")
        self.fill_color_restore_button.setProperty("role", "command")
        self.fill_color_restore_button.setProperty("variant", "secondary")
        self.fill_color_restore_button.setAccessibleName(
            "Restore automatic opaque fill color"
        )
        self.fill_color_restore_button.setToolTip(
            "Remove only the selected parent's fill-color override. Automatic "
            "style evidence remains immutable."
        )
        self.fill_color_restore_button.clicked.connect(
            self.fill_color_restore_requested
        )
        fill_color_layout.addWidget(self.fill_color_restore_button)
        self.fill_color_status = QtWidgets.QLabel(
            "Select a render-required parent to edit fill color"
        )
        self.fill_color_status.setProperty("role", "secondary")
        self.fill_color_status.setProperty("tone", "muted")
        self.fill_color_status.setWordWrap(True)
        self.fill_color_status.setAccessibleName("Fill-color edit status")
        fill_color_layout.addWidget(self.fill_color_status)
        layout.addWidget(fill_color_card)
        outline_color_card = QtWidgets.QFrame()
        outline_color_card.setObjectName("outlineColorCard")
        outline_color_card.setProperty("role", "panel-raised")
        outline_color_layout = QtWidgets.QVBoxLayout(outline_color_card)
        outline_color_layout.setContentsMargins(8, 8, 8, 8)
        outline_color_layout.setSpacing(6)
        outline_color_title = QtWidgets.QLabel("Opaque Outline Color")
        outline_color_title.setProperty("role", "section")
        outline_color_layout.addWidget(outline_color_title)
        outline_color_help = QtWidgets.QLabel(
            "Set one exact opaque #RRGGBB outline color for the selected parent. "
            "Alpha values are shown as unsupported and are never coerced. Set "
            "and Restore Automatic publish only this style field; Preview remains "
            "explicit."
        )
        outline_color_help.setWordWrap(True)
        outline_color_help.setProperty("role", "secondary")
        outline_color_layout.addWidget(outline_color_help)
        outline_color_summary = QtWidgets.QFormLayout()
        outline_color_summary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        outline_color_summary.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.outline_color_automatic = QtWidgets.QLabel("—")
        self.outline_color_automatic.setWordWrap(True)
        self.outline_color_automatic.setAccessibleName("Automatic opaque outline color")
        outline_color_summary.addRow("Automatic", self.outline_color_automatic)
        self.outline_color_user = QtWidgets.QLabel("No edit")
        self.outline_color_user.setWordWrap(True)
        self.outline_color_user.setAccessibleName("User opaque outline-color edit")
        outline_color_summary.addRow("Your edit", self.outline_color_user)
        self.outline_color_effective = QtWidgets.QLabel("—")
        self.outline_color_effective.setWordWrap(True)
        self.outline_color_effective.setAccessibleName("Effective opaque outline color")
        outline_color_summary.addRow("Effective", self.outline_color_effective)
        self.outline_color_authority = QtWidgets.QLabel("—")
        self.outline_color_authority.setWordWrap(True)
        self.outline_color_authority.setAccessibleName("Outline-color authority")
        outline_color_summary.addRow("Authority", self.outline_color_authority)
        outline_color_layout.addLayout(outline_color_summary)
        self.outline_color_swatch = QtWidgets.QFrame()
        self.outline_color_swatch.setObjectName("outlineColorSwatch")
        self.outline_color_swatch.setProperty("role", "color-swatch")
        self.outline_color_swatch.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.outline_color_swatch.setMinimumSize(48, 48)
        self.outline_color_swatch.setMaximumHeight(64)
        self.outline_color_swatch.setAccessibleName(
            "Opaque outline-color preview swatch"
        )
        outline_color_layout.addWidget(self.outline_color_swatch)
        self.outline_color_edit = QtWidgets.QLineEdit()
        self.outline_color_edit.setObjectName("outlineColorHexEdit")
        self.outline_color_edit.setMaxLength(64)
        self.outline_color_edit.setPlaceholderText("#RRGGBB")
        self.outline_color_edit.setAccessibleName(
            "Selected parent opaque outline color"
        )
        self.outline_color_edit.setAccessibleDescription(
            "Enter exactly a hash followed by six hexadecimal digits. Alpha "
            "channels are unsupported."
        )
        self.outline_color_edit.textChanged.connect(
            self._outline_color_value_changed
        )
        outline_color_layout.addWidget(self.outline_color_edit)
        self.outline_color_set_button = QtWidgets.QPushButton("Set")
        self.outline_color_set_button.setObjectName("outlineColorSetButton")
        self.outline_color_set_button.setProperty("role", "command")
        self.outline_color_set_button.setProperty("variant", "primary")
        self.outline_color_set_button.setAccessibleName(
            "Set selected parent opaque outline color"
        )
        self.outline_color_set_button.setToolTip(
            "Publish only this opaque outline-color edit. Rendering does not start."
        )
        self.outline_color_set_button.clicked.connect(
            self.outline_color_apply_requested
        )
        outline_color_layout.addWidget(self.outline_color_set_button)
        self.outline_color_cancel_button = QtWidgets.QPushButton("Cancel")
        self.outline_color_cancel_button.setObjectName("outlineColorCancelButton")
        self.outline_color_cancel_button.setProperty("role", "command")
        self.outline_color_cancel_button.setProperty("variant", "secondary")
        self.outline_color_cancel_button.setAccessibleName(
            "Cancel opaque outline-color draft"
        )
        self.outline_color_cancel_button.setToolTip(
            "Discard only the unapplied opaque outline color."
        )
        self.outline_color_cancel_button.clicked.connect(
            self.outline_color_cancel_requested
        )
        outline_color_layout.addWidget(self.outline_color_cancel_button)
        self.outline_color_restore_button = QtWidgets.QPushButton(
            "Restore Automatic"
        )
        self.outline_color_restore_button.setObjectName("outlineColorRestoreButton")
        self.outline_color_restore_button.setProperty("role", "command")
        self.outline_color_restore_button.setProperty("variant", "secondary")
        self.outline_color_restore_button.setAccessibleName(
            "Restore automatic opaque outline color"
        )
        self.outline_color_restore_button.setToolTip(
            "Remove only the selected parent's outline-color override. Automatic "
            "style evidence remains immutable."
        )
        self.outline_color_restore_button.clicked.connect(
            self.outline_color_restore_requested
        )
        outline_color_layout.addWidget(self.outline_color_restore_button)
        self.outline_color_status = QtWidgets.QLabel(
            "Select a render-required parent to edit outline color"
        )
        self.outline_color_status.setProperty("role", "secondary")
        self.outline_color_status.setProperty("tone", "muted")
        self.outline_color_status.setWordWrap(True)
        self.outline_color_status.setAccessibleName("Outline-color edit status")
        outline_color_layout.addWidget(self.outline_color_status)
        layout.addWidget(outline_color_card)
        outline_width_card = QtWidgets.QFrame()
        outline_width_card.setObjectName("outlineWidthCard")
        outline_width_card.setProperty("role", "panel-raised")
        outline_width_layout = QtWidgets.QVBoxLayout(outline_width_card)
        outline_width_layout.setContentsMargins(8, 8, 8, 8)
        outline_width_layout.setSpacing(6)
        outline_width_title = QtWidgets.QLabel("Outline Width")
        outline_width_title.setProperty("role", "section")
        outline_width_layout.addWidget(outline_width_title)
        outline_width_help = QtWidgets.QLabel(
            "Set the selected parent's exact outline stroke width from 0 through "
            "128 pixels. Zero disables the rendered outline. Set and Restore "
            "Automatic publish only this style field; Preview remains explicit."
        )
        outline_width_help.setWordWrap(True)
        outline_width_help.setProperty("role", "secondary")
        outline_width_layout.addWidget(outline_width_help)
        outline_width_summary = QtWidgets.QFormLayout()
        outline_width_summary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        outline_width_summary.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.outline_width_automatic = QtWidgets.QLabel("—")
        self.outline_width_automatic.setWordWrap(True)
        self.outline_width_automatic.setAccessibleName(
            "Automatic outline width in pixels"
        )
        outline_width_summary.addRow("Automatic", self.outline_width_automatic)
        self.outline_width_user = QtWidgets.QLabel("No edit")
        self.outline_width_user.setWordWrap(True)
        self.outline_width_user.setAccessibleName("User outline-width edit in pixels")
        outline_width_summary.addRow("Your edit", self.outline_width_user)
        self.outline_width_effective = QtWidgets.QLabel("—")
        self.outline_width_effective.setWordWrap(True)
        self.outline_width_effective.setAccessibleName(
            "Effective outline width in pixels"
        )
        outline_width_summary.addRow("Effective", self.outline_width_effective)
        self.outline_width_authority = QtWidgets.QLabel("—")
        self.outline_width_authority.setWordWrap(True)
        self.outline_width_authority.setAccessibleName("Outline-width authority")
        outline_width_summary.addRow("Authority", self.outline_width_authority)
        outline_width_layout.addLayout(outline_width_summary)
        self.outline_width_edit = WheelSafeDoubleSpinBox()
        self.outline_width_edit.setObjectName("outlineWidthSpinBox")
        self.outline_width_edit.setRange(0.0, 128.0)
        self.outline_width_edit.setDecimals(3)
        self.outline_width_edit.setSingleStep(0.25)
        self.outline_width_edit.setSuffix(" px")
        self.outline_width_edit.setKeyboardTracking(False)
        self.outline_width_edit.setAccessibleName(
            "Selected parent outline width in pixels"
        )
        self.outline_width_edit.setAccessibleDescription(
            "Enter a finite outline stroke width from 0 through 128 pixels. "
            "Zero disables the rendered outline."
        )
        self.outline_width_edit.valueChanged.connect(
            self._outline_width_value_changed
        )
        outline_width_layout.addWidget(self.outline_width_edit)
        self.outline_width_set_button = QtWidgets.QPushButton("Set")
        self.outline_width_set_button.setObjectName("outlineWidthSetButton")
        self.outline_width_set_button.setProperty("role", "command")
        self.outline_width_set_button.setProperty("variant", "primary")
        self.outline_width_set_button.setAccessibleName(
            "Set selected parent outline width"
        )
        self.outline_width_set_button.setToolTip(
            "Publish only this outline-width edit. Rendering does not start."
        )
        self.outline_width_set_button.clicked.connect(
            self.outline_width_apply_requested
        )
        outline_width_layout.addWidget(self.outline_width_set_button)
        self.outline_width_cancel_button = QtWidgets.QPushButton("Cancel")
        self.outline_width_cancel_button.setObjectName("outlineWidthCancelButton")
        self.outline_width_cancel_button.setProperty("role", "command")
        self.outline_width_cancel_button.setProperty("variant", "secondary")
        self.outline_width_cancel_button.setAccessibleName(
            "Cancel outline-width draft"
        )
        self.outline_width_cancel_button.setToolTip(
            "Discard only the unapplied outline width."
        )
        self.outline_width_cancel_button.clicked.connect(
            self.outline_width_cancel_requested
        )
        outline_width_layout.addWidget(self.outline_width_cancel_button)
        self.outline_width_restore_button = QtWidgets.QPushButton(
            "Restore Automatic"
        )
        self.outline_width_restore_button.setObjectName("outlineWidthRestoreButton")
        self.outline_width_restore_button.setProperty("role", "command")
        self.outline_width_restore_button.setProperty("variant", "secondary")
        self.outline_width_restore_button.setAccessibleName(
            "Restore automatic outline width"
        )
        self.outline_width_restore_button.setToolTip(
            "Remove only the selected parent's outline-width override. Automatic "
            "style evidence remains immutable."
        )
        self.outline_width_restore_button.clicked.connect(
            self.outline_width_restore_requested
        )
        outline_width_layout.addWidget(self.outline_width_restore_button)
        self.outline_width_status = QtWidgets.QLabel(
            "Select a render-required parent to edit outline width"
        )
        self.outline_width_status.setProperty("role", "secondary")
        self.outline_width_status.setProperty("tone", "muted")
        self.outline_width_status.setWordWrap(True)
        self.outline_width_status.setAccessibleName("Outline-width edit status")
        outline_width_layout.addWidget(self.outline_width_status)
        layout.addWidget(outline_width_card)
        preferred_size_card = QtWidgets.QFrame()
        preferred_size_card.setObjectName("preferredSizeCard")
        preferred_size_card.setProperty("role", "panel-raised")
        preferred_size_layout = QtWidgets.QVBoxLayout(preferred_size_card)
        preferred_size_layout.setContentsMargins(8, 8, 8, 8)
        preferred_size_layout.setSpacing(6)
        preferred_size_title = QtWidgets.QLabel("Preferred Size")
        preferred_size_title.setProperty("role", "section")
        preferred_size_layout.addWidget(preferred_size_title)
        preferred_size_help = QtWidgets.QLabel(
            "Set the selected parent's exact preferred rendered text size from "
            "0.1 through 2048 pixels. This is a fit-quality target, never a "
            "minimum or admission rule. Set and Restore Automatic publish only "
            "this style field; Preview remains explicit."
        )
        preferred_size_help.setWordWrap(True)
        preferred_size_help.setProperty("role", "secondary")
        preferred_size_layout.addWidget(preferred_size_help)
        preferred_size_summary = QtWidgets.QFormLayout()
        preferred_size_summary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        preferred_size_summary.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.preferred_size_automatic = QtWidgets.QLabel("—")
        self.preferred_size_automatic.setWordWrap(True)
        self.preferred_size_automatic.setAccessibleName(
            "Automatic preferred size in pixels"
        )
        preferred_size_summary.addRow("Automatic", self.preferred_size_automatic)
        self.preferred_size_user = QtWidgets.QLabel("No edit")
        self.preferred_size_user.setWordWrap(True)
        self.preferred_size_user.setAccessibleName(
            "User preferred-size edit in pixels"
        )
        preferred_size_summary.addRow("Your edit", self.preferred_size_user)
        self.preferred_size_effective = QtWidgets.QLabel("—")
        self.preferred_size_effective.setWordWrap(True)
        self.preferred_size_effective.setAccessibleName(
            "Effective preferred size in pixels"
        )
        preferred_size_summary.addRow("Effective", self.preferred_size_effective)
        self.preferred_size_authority = QtWidgets.QLabel("—")
        self.preferred_size_authority.setWordWrap(True)
        self.preferred_size_authority.setAccessibleName("Preferred-size authority")
        preferred_size_summary.addRow("Authority", self.preferred_size_authority)
        preferred_size_layout.addLayout(preferred_size_summary)
        self.preferred_size_edit = WheelSafeDoubleSpinBox()
        self.preferred_size_edit.setObjectName("preferredSizeSpinBox")
        self.preferred_size_edit.setRange(0.1, 2048.0)
        self.preferred_size_edit.setDecimals(3)
        self.preferred_size_edit.setSingleStep(1.0)
        self.preferred_size_edit.setSuffix(" px")
        self.preferred_size_edit.setKeyboardTracking(False)
        self.preferred_size_edit.setAccessibleName(
            "Selected parent preferred size in pixels"
        )
        self.preferred_size_edit.setAccessibleDescription(
            "Enter a finite preferred rendered text size from 0.1 through 2048 "
            "pixels. This quality target does not guarantee a minimum size."
        )
        self.preferred_size_edit.valueChanged.connect(
            self._preferred_size_value_changed
        )
        preferred_size_layout.addWidget(self.preferred_size_edit)
        self.preferred_size_set_button = QtWidgets.QPushButton("Set")
        self.preferred_size_set_button.setObjectName("preferredSizeSetButton")
        self.preferred_size_set_button.setProperty("role", "command")
        self.preferred_size_set_button.setProperty("variant", "primary")
        self.preferred_size_set_button.setAccessibleName(
            "Set selected parent preferred size"
        )
        self.preferred_size_set_button.setToolTip(
            "Publish only this preferred-size edit. Rendering does not start."
        )
        self.preferred_size_set_button.clicked.connect(
            self.preferred_size_apply_requested
        )
        preferred_size_layout.addWidget(self.preferred_size_set_button)
        self.preferred_size_cancel_button = QtWidgets.QPushButton("Cancel")
        self.preferred_size_cancel_button.setObjectName("preferredSizeCancelButton")
        self.preferred_size_cancel_button.setProperty("role", "command")
        self.preferred_size_cancel_button.setProperty("variant", "secondary")
        self.preferred_size_cancel_button.setAccessibleName(
            "Cancel preferred-size draft"
        )
        self.preferred_size_cancel_button.setToolTip(
            "Discard only the unapplied preferred size."
        )
        self.preferred_size_cancel_button.clicked.connect(
            self.preferred_size_cancel_requested
        )
        preferred_size_layout.addWidget(self.preferred_size_cancel_button)
        self.preferred_size_restore_button = QtWidgets.QPushButton(
            "Restore Automatic"
        )
        self.preferred_size_restore_button.setObjectName(
            "preferredSizeRestoreButton"
        )
        self.preferred_size_restore_button.setProperty("role", "command")
        self.preferred_size_restore_button.setProperty("variant", "secondary")
        self.preferred_size_restore_button.setAccessibleName(
            "Restore automatic preferred size"
        )
        self.preferred_size_restore_button.setToolTip(
            "Remove only the selected parent's preferred-size override. Automatic "
            "style evidence remains immutable."
        )
        self.preferred_size_restore_button.clicked.connect(
            self.preferred_size_restore_requested
        )
        preferred_size_layout.addWidget(self.preferred_size_restore_button)
        self.preferred_size_status = QtWidgets.QLabel(
            "Select a render-required parent to edit preferred size"
        )
        self.preferred_size_status.setProperty("role", "secondary")
        self.preferred_size_status.setProperty("tone", "muted")
        self.preferred_size_status.setWordWrap(True)
        self.preferred_size_status.setAccessibleName("Preferred-size edit status")
        preferred_size_layout.addWidget(self.preferred_size_status)
        layout.addWidget(preferred_size_card)
        shadow_color_card = QtWidgets.QFrame()
        shadow_color_card.setObjectName("shadowColorCard")
        shadow_color_card.setProperty("role", "panel-raised")
        shadow_color_layout = QtWidgets.QVBoxLayout(shadow_color_card)
        shadow_color_layout.setContentsMargins(8, 8, 8, 8)
        shadow_color_layout.setSpacing(6)
        shadow_color_title = QtWidgets.QLabel("Shadow Color")
        shadow_color_title.setProperty("role", "section")
        shadow_color_layout.addWidget(shadow_color_title)
        shadow_color_help = QtWidgets.QLabel(
            "Set one exact #RRGGBB or #RRGGBBAA shadow color for the selected "
            "parent. Six digits mean opaque alpha; transparent RGBA remains an "
            "explicit edit. Set and Restore Automatic publish only this style "
            "field; Preview remains explicit."
        )
        shadow_color_help.setWordWrap(True)
        shadow_color_help.setProperty("role", "secondary")
        shadow_color_layout.addWidget(shadow_color_help)
        shadow_color_summary = QtWidgets.QFormLayout()
        shadow_color_summary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        shadow_color_summary.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.shadow_color_automatic = QtWidgets.QLabel("—")
        self.shadow_color_automatic.setWordWrap(True)
        self.shadow_color_automatic.setAccessibleName("Automatic shadow color")
        shadow_color_summary.addRow("Automatic", self.shadow_color_automatic)
        self.shadow_color_user = QtWidgets.QLabel("No edit")
        self.shadow_color_user.setWordWrap(True)
        self.shadow_color_user.setAccessibleName("User shadow-color edit")
        shadow_color_summary.addRow("Your edit", self.shadow_color_user)
        self.shadow_color_effective = QtWidgets.QLabel("—")
        self.shadow_color_effective.setWordWrap(True)
        self.shadow_color_effective.setAccessibleName("Effective shadow color")
        shadow_color_summary.addRow("Effective", self.shadow_color_effective)
        self.shadow_color_authority = QtWidgets.QLabel("—")
        self.shadow_color_authority.setWordWrap(True)
        self.shadow_color_authority.setAccessibleName("Shadow-color authority")
        shadow_color_summary.addRow("Authority", self.shadow_color_authority)
        shadow_color_layout.addLayout(shadow_color_summary)
        self.shadow_color_swatch = QtWidgets.QFrame()
        self.shadow_color_swatch.setObjectName("shadowColorSwatch")
        self.shadow_color_swatch.setProperty("role", "color-swatch")
        self.shadow_color_swatch.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.shadow_color_swatch.setMinimumSize(48, 48)
        self.shadow_color_swatch.setMaximumHeight(64)
        self.shadow_color_swatch.setAccessibleName("Shadow-color RGBA preview swatch")
        shadow_color_layout.addWidget(self.shadow_color_swatch)
        self.shadow_color_edit = QtWidgets.QLineEdit()
        self.shadow_color_edit.setObjectName("shadowColorHexEdit")
        self.shadow_color_edit.setMaxLength(64)
        self.shadow_color_edit.setPlaceholderText("#RRGGBB or #RRGGBBAA")
        self.shadow_color_edit.setAccessibleName("Selected parent shadow color")
        self.shadow_color_edit.setAccessibleDescription(
            "Enter exactly a hash followed by six RGB or eight RGBA hexadecimal "
            "digits. Six digits use opaque alpha."
        )
        self.shadow_color_edit.textChanged.connect(
            self._shadow_color_value_changed
        )
        shadow_color_layout.addWidget(self.shadow_color_edit)
        self.shadow_color_set_button = QtWidgets.QPushButton("Set")
        self.shadow_color_set_button.setObjectName("shadowColorSetButton")
        self.shadow_color_set_button.setProperty("role", "command")
        self.shadow_color_set_button.setProperty("variant", "primary")
        self.shadow_color_set_button.setAccessibleName(
            "Set selected parent shadow color"
        )
        self.shadow_color_set_button.setToolTip(
            "Publish only this shadow-color edit. Rendering does not start."
        )
        self.shadow_color_set_button.clicked.connect(
            self.shadow_color_apply_requested
        )
        shadow_color_layout.addWidget(self.shadow_color_set_button)
        self.shadow_color_cancel_button = QtWidgets.QPushButton("Cancel")
        self.shadow_color_cancel_button.setObjectName("shadowColorCancelButton")
        self.shadow_color_cancel_button.setProperty("role", "command")
        self.shadow_color_cancel_button.setProperty("variant", "secondary")
        self.shadow_color_cancel_button.setAccessibleName(
            "Cancel shadow-color draft"
        )
        self.shadow_color_cancel_button.setToolTip(
            "Discard only the unapplied shadow color."
        )
        self.shadow_color_cancel_button.clicked.connect(
            self.shadow_color_cancel_requested
        )
        shadow_color_layout.addWidget(self.shadow_color_cancel_button)
        self.shadow_color_restore_button = QtWidgets.QPushButton(
            "Restore Automatic"
        )
        self.shadow_color_restore_button.setObjectName("shadowColorRestoreButton")
        self.shadow_color_restore_button.setProperty("role", "command")
        self.shadow_color_restore_button.setProperty("variant", "secondary")
        self.shadow_color_restore_button.setAccessibleName(
            "Restore automatic shadow color"
        )
        self.shadow_color_restore_button.setToolTip(
            "Remove only the selected parent's shadow-color override. Automatic "
            "effect evidence remains immutable."
        )
        self.shadow_color_restore_button.clicked.connect(
            self.shadow_color_restore_requested
        )
        shadow_color_layout.addWidget(self.shadow_color_restore_button)
        self.shadow_color_status = QtWidgets.QLabel(
            "Select a render-required parent to edit shadow color"
        )
        self.shadow_color_status.setProperty("role", "secondary")
        self.shadow_color_status.setProperty("tone", "muted")
        self.shadow_color_status.setWordWrap(True)
        self.shadow_color_status.setAccessibleName("Shadow-color edit status")
        shadow_color_layout.addWidget(self.shadow_color_status)
        layout.addWidget(shadow_color_card)
        shadow_blur_card = QtWidgets.QFrame()
        shadow_blur_card.setObjectName("shadowBlurCard")
        shadow_blur_card.setProperty("role", "panel-raised")
        shadow_blur_layout = QtWidgets.QVBoxLayout(shadow_blur_card)
        shadow_blur_layout.setContentsMargins(8, 8, 8, 8)
        shadow_blur_layout.setSpacing(6)
        shadow_blur_title = QtWidgets.QLabel("Shadow Blur")
        shadow_blur_title.setProperty("role", "section")
        shadow_blur_layout.addWidget(shadow_blur_title)
        shadow_blur_help = QtWidgets.QLabel(
            "Set the selected parent's exact shadow blur radius from 0 through "
            "64 pixels. Zero is an explicit sharp-edge value, not Restore "
            "Automatic. Color, opacity, offset, rotation, and visibility remain "
            "unchanged; Preview remains explicit."
        )
        shadow_blur_help.setWordWrap(True)
        shadow_blur_help.setProperty("role", "secondary")
        shadow_blur_layout.addWidget(shadow_blur_help)
        shadow_blur_summary = QtWidgets.QFormLayout()
        shadow_blur_summary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        shadow_blur_summary.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.shadow_blur_automatic = QtWidgets.QLabel("—")
        self.shadow_blur_automatic.setWordWrap(True)
        self.shadow_blur_automatic.setAccessibleName(
            "Automatic shadow blur in pixels"
        )
        shadow_blur_summary.addRow("Automatic", self.shadow_blur_automatic)
        self.shadow_blur_user = QtWidgets.QLabel("No edit")
        self.shadow_blur_user.setWordWrap(True)
        self.shadow_blur_user.setAccessibleName(
            "User shadow-blur edit in pixels"
        )
        shadow_blur_summary.addRow("Your edit", self.shadow_blur_user)
        self.shadow_blur_effective = QtWidgets.QLabel("—")
        self.shadow_blur_effective.setWordWrap(True)
        self.shadow_blur_effective.setAccessibleName(
            "Effective shadow blur in pixels"
        )
        shadow_blur_summary.addRow("Effective", self.shadow_blur_effective)
        self.shadow_blur_authority = QtWidgets.QLabel("—")
        self.shadow_blur_authority.setWordWrap(True)
        self.shadow_blur_authority.setAccessibleName("Shadow-blur authority")
        shadow_blur_summary.addRow("Authority", self.shadow_blur_authority)
        shadow_blur_layout.addLayout(shadow_blur_summary)
        self.shadow_blur_edit = WheelSafeDoubleSpinBox()
        self.shadow_blur_edit.setObjectName("shadowBlurSpinBox")
        self.shadow_blur_edit.setRange(0.0, 64.0)
        self.shadow_blur_edit.setDecimals(3)
        self.shadow_blur_edit.setSingleStep(1.0)
        self.shadow_blur_edit.setSuffix(" px")
        self.shadow_blur_edit.setKeyboardTracking(False)
        self.shadow_blur_edit.setAccessibleName(
            "Selected parent shadow blur in pixels"
        )
        self.shadow_blur_edit.setAccessibleDescription(
            "Enter a finite shadow blur radius from 0 through 64 pixels. Zero "
            "keeps the shadow visible with no blur."
        )
        self.shadow_blur_edit.valueChanged.connect(
            self._shadow_blur_value_changed
        )
        shadow_blur_layout.addWidget(self.shadow_blur_edit)
        self.shadow_blur_set_button = QtWidgets.QPushButton("Set")
        self.shadow_blur_set_button.setObjectName("shadowBlurSetButton")
        self.shadow_blur_set_button.setProperty("role", "command")
        self.shadow_blur_set_button.setProperty("variant", "primary")
        self.shadow_blur_set_button.setAccessibleName(
            "Set selected parent shadow blur"
        )
        self.shadow_blur_set_button.setToolTip(
            "Publish only this shadow-blur edit. Rendering does not start."
        )
        self.shadow_blur_set_button.clicked.connect(
            self.shadow_blur_apply_requested
        )
        shadow_blur_layout.addWidget(self.shadow_blur_set_button)
        self.shadow_blur_cancel_button = QtWidgets.QPushButton("Cancel")
        self.shadow_blur_cancel_button.setObjectName("shadowBlurCancelButton")
        self.shadow_blur_cancel_button.setProperty("role", "command")
        self.shadow_blur_cancel_button.setProperty("variant", "secondary")
        self.shadow_blur_cancel_button.setAccessibleName(
            "Cancel shadow-blur draft"
        )
        self.shadow_blur_cancel_button.setToolTip(
            "Discard only the unapplied shadow blur."
        )
        self.shadow_blur_cancel_button.clicked.connect(
            self.shadow_blur_cancel_requested
        )
        shadow_blur_layout.addWidget(self.shadow_blur_cancel_button)
        self.shadow_blur_restore_button = QtWidgets.QPushButton(
            "Restore Automatic"
        )
        self.shadow_blur_restore_button.setObjectName(
            "shadowBlurRestoreButton"
        )
        self.shadow_blur_restore_button.setProperty("role", "command")
        self.shadow_blur_restore_button.setProperty("variant", "secondary")
        self.shadow_blur_restore_button.setAccessibleName(
            "Restore automatic shadow blur"
        )
        self.shadow_blur_restore_button.setToolTip(
            "Remove only the selected parent's shadow-blur override. Automatic "
            "style evidence remains immutable."
        )
        self.shadow_blur_restore_button.clicked.connect(
            self.shadow_blur_restore_requested
        )
        shadow_blur_layout.addWidget(self.shadow_blur_restore_button)
        self.shadow_blur_status = QtWidgets.QLabel(
            "Select a render-required parent to edit shadow blur"
        )
        self.shadow_blur_status.setProperty("role", "secondary")
        self.shadow_blur_status.setProperty("tone", "muted")
        self.shadow_blur_status.setWordWrap(True)
        self.shadow_blur_status.setAccessibleName("Shadow-blur edit status")
        shadow_blur_layout.addWidget(self.shadow_blur_status)
        layout.addWidget(shadow_blur_card)
        shadow_offset_card = QtWidgets.QFrame()
        shadow_offset_card.setObjectName("shadowOffsetCard")
        shadow_offset_card.setProperty("role", "panel-raised")
        shadow_offset_layout = QtWidgets.QVBoxLayout(shadow_offset_card)
        shadow_offset_layout.setContentsMargins(8, 8, 8, 8)
        shadow_offset_layout.setSpacing(6)
        shadow_offset_title = QtWidgets.QLabel("Shadow Offset")
        shadow_offset_title.setProperty("role", "section")
        shadow_offset_layout.addWidget(shadow_offset_title)
        shadow_offset_help = QtWidgets.QLabel(
            "Set the selected parent's exact shadow translation from -256 through "
            "256 pixels on each axis. Zero is an explicit component. Color, "
            "opacity, blur, rotation, and visibility remain unchanged; Preview "
            "remains explicit."
        )
        shadow_offset_help.setWordWrap(True)
        shadow_offset_help.setProperty("role", "secondary")
        shadow_offset_layout.addWidget(shadow_offset_help)
        shadow_offset_summary = QtWidgets.QFormLayout()
        shadow_offset_summary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        shadow_offset_summary.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.shadow_offset_automatic = QtWidgets.QLabel("—")
        self.shadow_offset_automatic.setWordWrap(True)
        self.shadow_offset_automatic.setAccessibleName(
            "Automatic shadow offset in X and Y pixels"
        )
        shadow_offset_summary.addRow("Automatic", self.shadow_offset_automatic)
        self.shadow_offset_user = QtWidgets.QLabel("No edit")
        self.shadow_offset_user.setWordWrap(True)
        self.shadow_offset_user.setAccessibleName(
            "User shadow-offset edit in X and Y pixels"
        )
        shadow_offset_summary.addRow("Your edit", self.shadow_offset_user)
        self.shadow_offset_effective = QtWidgets.QLabel("—")
        self.shadow_offset_effective.setWordWrap(True)
        self.shadow_offset_effective.setAccessibleName(
            "Effective shadow offset in X and Y pixels"
        )
        shadow_offset_summary.addRow("Effective", self.shadow_offset_effective)
        self.shadow_offset_authority = QtWidgets.QLabel("—")
        self.shadow_offset_authority.setWordWrap(True)
        self.shadow_offset_authority.setAccessibleName("Shadow-offset authority")
        shadow_offset_summary.addRow("Authority", self.shadow_offset_authority)
        shadow_offset_layout.addLayout(shadow_offset_summary)
        shadow_offset_draft = QtWidgets.QFormLayout()
        self.shadow_offset_x_edit = WheelSafeDoubleSpinBox()
        self.shadow_offset_x_edit.setObjectName("shadowOffsetXSpinBox")
        self.shadow_offset_y_edit = WheelSafeDoubleSpinBox()
        self.shadow_offset_y_edit.setObjectName("shadowOffsetYSpinBox")
        for axis, control in (("X", self.shadow_offset_x_edit), ("Y", self.shadow_offset_y_edit)):
            control.setRange(-256.0, 256.0)
            control.setDecimals(3)
            control.setSingleStep(1.0)
            control.setSuffix(" px")
            control.setKeyboardTracking(False)
            control.setAccessibleName(f"Selected parent shadow offset {axis} in pixels")
            control.setAccessibleDescription(
                f"Enter a finite shadow offset {axis} component from -256 through 256 pixels."
            )
            control.valueChanged.connect(self._shadow_offset_value_changed)
            shadow_offset_draft.addRow(axis, control)
        shadow_offset_layout.addLayout(shadow_offset_draft)
        self.shadow_offset_set_button = QtWidgets.QPushButton("Set")
        self.shadow_offset_set_button.setObjectName("shadowOffsetSetButton")
        self.shadow_offset_set_button.setProperty("role", "command")
        self.shadow_offset_set_button.setProperty("variant", "primary")
        self.shadow_offset_set_button.setAccessibleName(
            "Set selected parent shadow offset"
        )
        self.shadow_offset_set_button.setToolTip(
            "Publish only this shadow-offset edit. Rendering does not start."
        )
        self.shadow_offset_set_button.clicked.connect(
            self.shadow_offset_apply_requested
        )
        shadow_offset_layout.addWidget(self.shadow_offset_set_button)
        self.shadow_offset_cancel_button = QtWidgets.QPushButton("Cancel")
        self.shadow_offset_cancel_button.setObjectName("shadowOffsetCancelButton")
        self.shadow_offset_cancel_button.setProperty("role", "command")
        self.shadow_offset_cancel_button.setProperty("variant", "secondary")
        self.shadow_offset_cancel_button.setAccessibleName(
            "Cancel shadow-offset draft"
        )
        self.shadow_offset_cancel_button.setToolTip(
            "Discard only the unapplied shadow offset."
        )
        self.shadow_offset_cancel_button.clicked.connect(
            self.shadow_offset_cancel_requested
        )
        shadow_offset_layout.addWidget(self.shadow_offset_cancel_button)
        self.shadow_offset_restore_button = QtWidgets.QPushButton(
            "Restore Automatic"
        )
        self.shadow_offset_restore_button.setObjectName(
            "shadowOffsetRestoreButton"
        )
        self.shadow_offset_restore_button.setProperty("role", "command")
        self.shadow_offset_restore_button.setProperty("variant", "secondary")
        self.shadow_offset_restore_button.setAccessibleName(
            "Restore automatic shadow offset"
        )
        self.shadow_offset_restore_button.setToolTip(
            "Remove only the selected parent's shadow-offset override. Automatic "
            "style evidence remains immutable."
        )
        self.shadow_offset_restore_button.clicked.connect(
            self.shadow_offset_restore_requested
        )
        shadow_offset_layout.addWidget(self.shadow_offset_restore_button)
        self.shadow_offset_status = QtWidgets.QLabel(
            "Select a render-required parent to edit shadow offset"
        )
        self.shadow_offset_status.setProperty("role", "secondary")
        self.shadow_offset_status.setProperty("tone", "muted")
        self.shadow_offset_status.setWordWrap(True)
        self.shadow_offset_status.setAccessibleName("Shadow-offset edit status")
        shadow_offset_layout.addWidget(self.shadow_offset_status)
        layout.addWidget(shadow_offset_card)
        shadow_visibility_card = QtWidgets.QFrame()
        shadow_visibility_card.setObjectName("shadowVisibilityCard")
        shadow_visibility_card.setProperty("role", "panel-raised")
        shadow_visibility_layout = QtWidgets.QVBoxLayout(shadow_visibility_card)
        shadow_visibility_layout.setContentsMargins(8, 8, 8, 8)
        shadow_visibility_layout.setSpacing(6)
        shadow_visibility_title = QtWidgets.QLabel("Shadow Visibility")
        shadow_visibility_title.setProperty("role", "section")
        shadow_visibility_layout.addWidget(shadow_visibility_title)
        shadow_visibility_help = QtWidgets.QLabel(
            "Hide a strictly valid visible automatic shadow without changing its "
            "immutable color, opacity, offset, blur, or rotation facts. This edit "
            "cannot create or enable a shadow. Set and Restore Automatic publish "
            "only visibility; Preview remains explicit."
        )
        shadow_visibility_help.setWordWrap(True)
        shadow_visibility_help.setProperty("role", "secondary")
        shadow_visibility_layout.addWidget(shadow_visibility_help)
        shadow_visibility_summary = QtWidgets.QFormLayout()
        shadow_visibility_summary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        shadow_visibility_summary.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.shadow_visibility_automatic = QtWidgets.QLabel("—")
        self.shadow_visibility_automatic.setWordWrap(True)
        self.shadow_visibility_automatic.setAccessibleName(
            "Automatic shadow visibility"
        )
        shadow_visibility_summary.addRow(
            "Automatic", self.shadow_visibility_automatic
        )
        self.shadow_visibility_user = QtWidgets.QLabel("No edit")
        self.shadow_visibility_user.setWordWrap(True)
        self.shadow_visibility_user.setAccessibleName("User shadow-visibility edit")
        shadow_visibility_summary.addRow("Your edit", self.shadow_visibility_user)
        self.shadow_visibility_effective = QtWidgets.QLabel("—")
        self.shadow_visibility_effective.setWordWrap(True)
        self.shadow_visibility_effective.setAccessibleName(
            "Effective shadow visibility"
        )
        shadow_visibility_summary.addRow(
            "Effective", self.shadow_visibility_effective
        )
        self.shadow_visibility_authority = QtWidgets.QLabel("—")
        self.shadow_visibility_authority.setWordWrap(True)
        self.shadow_visibility_authority.setAccessibleName(
            "Shadow-visibility authority"
        )
        shadow_visibility_summary.addRow(
            "Authority", self.shadow_visibility_authority
        )
        shadow_visibility_layout.addLayout(shadow_visibility_summary)
        self.shadow_visibility_edit = QtWidgets.QCheckBox("Show shadow")
        self.shadow_visibility_edit.setObjectName("shadowVisibilityCheckBox")
        self.shadow_visibility_edit.setAccessibleName(
            "Show selected parent automatic shadow"
        )
        self.shadow_visibility_edit.setAccessibleDescription(
            "Uncheck to draft Hidden. This control cannot create or enable a shadow; "
            "use Restore Automatic to remove a saved Hidden edit."
        )
        self.shadow_visibility_edit.toggled.connect(
            self._shadow_visibility_value_changed
        )
        shadow_visibility_layout.addWidget(self.shadow_visibility_edit)
        self.shadow_visibility_set_button = QtWidgets.QPushButton("Set")
        self.shadow_visibility_set_button.setObjectName(
            "shadowVisibilitySetButton"
        )
        self.shadow_visibility_set_button.setProperty("role", "command")
        self.shadow_visibility_set_button.setProperty("variant", "primary")
        self.shadow_visibility_set_button.setAccessibleName(
            "Set selected parent shadow visibility to Hidden"
        )
        self.shadow_visibility_set_button.setToolTip(
            "Publish only shadow_enabled=false. Rendering does not start."
        )
        self.shadow_visibility_set_button.clicked.connect(
            self.shadow_visibility_apply_requested
        )
        shadow_visibility_layout.addWidget(self.shadow_visibility_set_button)
        self.shadow_visibility_cancel_button = QtWidgets.QPushButton("Cancel")
        self.shadow_visibility_cancel_button.setObjectName(
            "shadowVisibilityCancelButton"
        )
        self.shadow_visibility_cancel_button.setProperty("role", "command")
        self.shadow_visibility_cancel_button.setProperty("variant", "secondary")
        self.shadow_visibility_cancel_button.setAccessibleName(
            "Cancel shadow-visibility draft"
        )
        self.shadow_visibility_cancel_button.setToolTip(
            "Discard only the unapplied shadow-visibility change."
        )
        self.shadow_visibility_cancel_button.clicked.connect(
            self.shadow_visibility_cancel_requested
        )
        shadow_visibility_layout.addWidget(self.shadow_visibility_cancel_button)
        self.shadow_visibility_restore_button = QtWidgets.QPushButton(
            "Restore Automatic"
        )
        self.shadow_visibility_restore_button.setObjectName(
            "shadowVisibilityRestoreButton"
        )
        self.shadow_visibility_restore_button.setProperty("role", "command")
        self.shadow_visibility_restore_button.setProperty("variant", "secondary")
        self.shadow_visibility_restore_button.setAccessibleName(
            "Restore automatic shadow visibility"
        )
        self.shadow_visibility_restore_button.setToolTip(
            "Remove only shadow_enabled=false. Automatic effect facts remain immutable."
        )
        self.shadow_visibility_restore_button.clicked.connect(
            self.shadow_visibility_restore_requested
        )
        shadow_visibility_layout.addWidget(self.shadow_visibility_restore_button)
        self.shadow_visibility_status = QtWidgets.QLabel(
            "Select a render-required parent to edit shadow visibility"
        )
        self.shadow_visibility_status.setProperty("role", "secondary")
        self.shadow_visibility_status.setProperty("tone", "muted")
        self.shadow_visibility_status.setWordWrap(True)
        self.shadow_visibility_status.setAccessibleName(
            "Shadow-visibility edit status"
        )
        shadow_visibility_layout.addWidget(self.shadow_visibility_status)
        layout.addWidget(shadow_visibility_card)
        QtWidgets.QWidget.setTabOrder(
            self.fill_color_restore_button,
            self.outline_color_edit,
        )
        QtWidgets.QWidget.setTabOrder(
            self.outline_color_edit,
            self.outline_color_set_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.outline_color_set_button,
            self.outline_color_cancel_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.outline_color_cancel_button,
            self.outline_color_restore_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.outline_color_restore_button,
            self.outline_width_edit,
        )
        QtWidgets.QWidget.setTabOrder(
            self.outline_width_edit,
            self.outline_width_set_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.outline_width_set_button,
            self.outline_width_cancel_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.outline_width_cancel_button,
            self.outline_width_restore_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.outline_width_restore_button,
            self.preferred_size_edit,
        )
        QtWidgets.QWidget.setTabOrder(
            self.preferred_size_edit,
            self.preferred_size_set_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.preferred_size_set_button,
            self.preferred_size_cancel_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.preferred_size_cancel_button,
            self.preferred_size_restore_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.preferred_size_restore_button,
            self.shadow_color_edit,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_color_edit,
            self.shadow_color_set_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_color_set_button,
            self.shadow_color_cancel_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_color_cancel_button,
            self.shadow_color_restore_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_color_restore_button,
            self.shadow_blur_edit,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_blur_edit,
            self.shadow_blur_set_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_blur_set_button,
            self.shadow_blur_cancel_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_blur_cancel_button,
            self.shadow_blur_restore_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_blur_restore_button,
            self.shadow_offset_x_edit,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_offset_x_edit,
            self.shadow_offset_y_edit,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_offset_y_edit,
            self.shadow_offset_set_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_offset_set_button,
            self.shadow_offset_cancel_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_offset_cancel_button,
            self.shadow_offset_restore_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_offset_restore_button,
            self.shadow_visibility_edit,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_visibility_edit,
            self.shadow_visibility_set_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_visibility_set_button,
            self.shadow_visibility_cancel_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.shadow_visibility_cancel_button,
            self.shadow_visibility_restore_button,
        )
        self._style_detail_cards = (
            font_role_card,
            font_weight_tier_card,
            fill_color_card,
            outline_color_card,
            outline_width_card,
            preferred_size_card,
            shadow_color_card,
            shadow_blur_card,
            shadow_offset_card,
            shadow_visibility_card,
        )
        self.style_resolved_card = QtWidgets.QFrame()
        self.style_resolved_card.setObjectName("styleResolvedCard")
        style_resolved_layout = QtWidgets.QVBoxLayout(self.style_resolved_card)
        style_resolved_layout.setContentsMargins(8, 8, 8, 8)
        style_resolved_layout.setSpacing(8)
        style_note = QtWidgets.QLabel(
            "Automatic style is preserved. Only fields explicitly opened as an "
            "override change the effective render plan."
        )
        style_note.setObjectName("styleResolvedNotice")
        style_note.setProperty("role", "status-banner")
        style_note.setProperty("tone", "info")
        style_note.setWordWrap(True)
        style_resolved_layout.addWidget(style_note)
        style_summary = QtWidgets.QHBoxLayout()
        self.style_automatic_check = QtWidgets.QCheckBox("Automatic resolved style")
        self.style_automatic_check.setChecked(True)
        self.style_automatic_check.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.style_automatic_check.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        self.style_automatic_check.setAccessibleName("Automatic resolved style")
        style_summary.addWidget(self.style_automatic_check)
        style_summary.addStretch(1)
        self.style_resolved_summary = QtWidgets.QLabel("—")
        self.style_resolved_summary.setObjectName("styleResolvedSummary")
        self.style_resolved_summary.setProperty("role", "section")
        self.style_resolved_summary.setAccessibleName("Resolved style summary")
        style_summary.addWidget(self.style_resolved_summary)
        style_resolved_layout.addLayout(style_summary)
        style_grid = QtWidgets.QGridLayout()
        style_grid.setHorizontalSpacing(8)
        style_grid.setVerticalSpacing(6)
        self.style_facade_fields: dict[str, QtWidgets.QLineEdit] = {}
        for index, (field_id, label, accessible_name) in enumerate(
            (
                ("font_family", "Font family", "Resolved font family"),
                ("resolved_face", "Resolved face", "Resolved font face"),
                ("weight", "Weight", "Resolved font weight"),
                ("preferred_size", "Preferred size", "Resolved preferred font size"),
                ("minimum_size", "Minimum size", "Resolved minimum font size"),
                ("maximum_size", "Maximum size", "Resolved maximum font size"),
                ("fill", "Fill", "Resolved fill color"),
                ("outline", "Outline", "Resolved outline color"),
            )
        ):
            row, column = divmod(index, 2)
            field_column = column * 2
            caption = QtWidgets.QLabel(label)
            caption.setProperty("role", "secondary")
            field = self._facade_readout(accessible_name)
            self.style_facade_fields[field_id] = field
            style_grid.addWidget(caption, row * 2, field_column)
            style_grid.addWidget(field, row * 2 + 1, field_column)
        style_resolved_layout.addLayout(style_grid)
        style_actions = QtWidgets.QHBoxLayout()
        self.style_preview_button = QtWidgets.QPushButton("Preview style")
        self.style_preview_button.setObjectName("previewStyleButton")
        self.style_preview_button.setProperty("role", "command")
        self.style_preview_button.setProperty("variant", "primary")
        self.style_preview_button.setIcon(hybrid_icon("play"))
        self.style_preview_button.setAccessibleName("Preview style on this page")
        self.style_preview_button.clicked.connect(
            lambda: self.rerender_requested.emit(self._current_page_id)
            if self._current_page_id
            else None
        )
        style_actions.addWidget(self.style_preview_button, 1)
        self.style_more_button = QtWidgets.QToolButton()
        self.style_more_button.setObjectName("styleAdvancedOverridesButton")
        self.style_more_button.setCheckable(True)
        self.style_more_button.setIcon(hybrid_icon("more"))
        self.style_more_button.setText("Edit style")
        self.style_more_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.style_more_button.setToolTip("Show explicit style override controls")
        self.style_more_button.setAccessibleName("Show explicit style override controls")
        self.style_more_button.toggled.connect(self._set_style_details_visible)
        style_actions.addWidget(self.style_more_button)
        style_resolved_layout.addLayout(style_actions)
        self.style_preview_button.setVisible(False)
        self.style_more_button.setVisible(True)
        layout.insertWidget(0, self.style_resolved_card)

        self.style_legacy_values = QtWidgets.QWidget()
        style_legacy_layout = QtWidgets.QVBoxLayout(self.style_legacy_values)
        style_legacy_layout.setContentsMargins(0, 0, 0, 0)
        self.style_form = QtWidgets.QFormLayout()
        self.style_form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        style_legacy_layout.addLayout(self.style_form)
        self.style_authority = QtWidgets.QLabel(
            "Automatic style is immutable. Explicit overrides appear separately."
        )
        self.style_authority.setWordWrap(True)
        self.style_authority.setProperty("role", "secondary")
        style_legacy_layout.addWidget(self.style_authority)
        layout.addWidget(self.style_legacy_values)
        # Lead with resolved evidence and one Preview action.  The More menu
        # keeps every explicit override keyboard reachable without presenting
        # all low-frequency controls at equal visual weight.
        self._set_style_details_visible(False)
        layout.addStretch(1)
        return scroll

    def _set_style_details_visible(self, visible: bool) -> None:
        show = bool(visible)
        for card in getattr(self, "_style_detail_cards", ()):
            card.setVisible(show)
        if hasattr(self, "style_legacy_values"):
            self.style_legacy_values.setVisible(show)
        if hasattr(self, "style_more_button"):
            blocker = QtCore.QSignalBlocker(self.style_more_button)
            self.style_more_button.setChecked(show)
            del blocker
            self.style_more_button.setToolTip(
                "Hide explicit style override controls"
                if show
                else "Show explicit style override controls"
            )
            self.style_more_button.setText(
                "Hide controls" if show else "Edit style"
            )
            self.style_more_button.setAccessibleName(
                "Hide explicit style override controls"
                if show
                else "Show explicit style override controls"
            )
        if (
            hasattr(self, "inspector_toggle_details_action")
            and self.inspector_tabs.currentIndex()
            == self._inspector_index.get("style")
        ):
            self.inspector_toggle_details_action.setText(
                "Hide explicit override controls"
                if show
                else "Show explicit override controls"
            )

    def _update_style_facade(
        self,
        *,
        style_values: Mapping[str, object],
        effective_font_role: str | None,
        effective_font_weight_tier: str | None,
        effective_preferred_size: float | None,
        effective_fill_color: str | None,
        effective_outline_color: str | None,
    ) -> None:
        if not hasattr(self, "style_facade_fields"):
            return
        family = {
            "sans_regular": "Noto Sans CJK SC",
            "sans_medium": "Noto Sans CJK SC",
            "sans_bold": "Noto Sans CJK SC",
            "sans_black": "Noto Sans CJK SC",
            "serif_regular": "Noto Serif CJK SC",
            "serif_semibold": "Noto Serif CJK SC",
            "serif_bold": "Noto Serif CJK SC",
        }.get(str(effective_font_role or ""), "—")
        face = (
            "CJK Core Serif"
            if str(effective_font_role or "").startswith("serif_")
            else "CJK Core Sans"
            if effective_font_role
            else "—"
        )
        weight = {
            "slender": "Regular",
            "base": "Medium",
            "emphasis": "Semibold",
            "heavy": "Black",
        }.get(str(effective_font_weight_tier or ""), "—")
        minimum_size = style_values.get("minimum_size", style_values.get("min_size"))
        maximum_size = style_values.get("maximum_size", style_values.get("max_size"))
        interval = style_values.get("target_preferred_em_interval_px")
        if (
            (minimum_size is None or maximum_size is None)
            and isinstance(interval, (tuple, list))
            and len(interval) == 2
        ):
            if minimum_size is None:
                minimum_size = interval[0]
            if maximum_size is None:
                maximum_size = interval[1]

        def _number(value: object, suffix: str = "") -> str:
            if value is None:
                return "—"
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return str(value)
            text = str(int(numeric)) if numeric.is_integer() else f"{numeric:g}"
            return f"{text}{suffix}"

        values = {
            "font_family": family,
            "resolved_face": face,
            "weight": weight,
            "preferred_size": _number(effective_preferred_size, " px"),
            "minimum_size": _number(minimum_size, " px"),
            "maximum_size": _number(maximum_size, " px"),
            "fill": str(effective_fill_color or "—"),
            "outline": str(effective_outline_color or "—"),
        }
        for field_id, field in self.style_facade_fields.items():
            field.setText(values[field_id])
        self.style_resolved_summary.setText(
            f"{face} · {weight} · {_number(effective_preferred_size, 'px')}"
        )
        self.style_preview_button.setEnabled(bool(self._current_page_id))

    def _build_layout_tab(self) -> QtWidgets.QWidget:
        scroll, layout = self._scroll_tab()
        reading_order_card = QtWidgets.QFrame()
        reading_order_card.setObjectName("readingOrderCard")
        reading_order_card.setProperty("role", "panel-raised")
        reading_order_layout = QtWidgets.QVBoxLayout(reading_order_card)
        reading_order_layout.setContentsMargins(8, 8, 8, 8)
        reading_order_layout.setSpacing(6)
        reading_order_title = QtWidgets.QLabel("Reading Order")
        reading_order_title.setProperty("role", "section")
        reading_order_layout.addWidget(reading_order_title)
        reading_order_help = QtWidgets.QLabel(
            "Move the selected included parent earlier or later among other "
            "included parents. Excluded parents keep their absolute page slots. "
            "Apply publishes one reversible page-wide order; Preview remains explicit."
        )
        reading_order_help.setWordWrap(True)
        reading_order_help.setProperty("role", "secondary")
        reading_order_layout.addWidget(reading_order_help)
        reading_order_summary = QtWidgets.QFormLayout()
        reading_order_summary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        reading_order_summary.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.reading_order_automatic = QtWidgets.QLabel("—")
        self.reading_order_automatic.setWordWrap(True)
        self.reading_order_automatic.setAccessibleName("Automatic page reading order")
        reading_order_summary.addRow("Automatic", self.reading_order_automatic)
        self.reading_order_effective = QtWidgets.QLabel("—")
        self.reading_order_effective.setWordWrap(True)
        self.reading_order_effective.setAccessibleName("Effective page reading order")
        reading_order_summary.addRow("Effective", self.reading_order_effective)
        self.reading_order_proposed = QtWidgets.QLabel("—")
        self.reading_order_proposed.setWordWrap(True)
        self.reading_order_proposed.setAccessibleName("Proposed page reading order")
        reading_order_summary.addRow("Proposed", self.reading_order_proposed)
        reading_order_layout.addLayout(reading_order_summary)
        reading_order_move_actions = QtWidgets.QHBoxLayout()
        self.reading_order_earlier_button = QtWidgets.QPushButton("Earlier")
        self.reading_order_earlier_button.setObjectName("readingOrderEarlierButton")
        self.reading_order_earlier_button.setProperty("role", "command")
        self.reading_order_earlier_button.setProperty("variant", "secondary")
        self.reading_order_earlier_button.setAccessibleName(
            "Move selected parent earlier in reading order"
        )
        self.reading_order_earlier_button.setToolTip(
            "Move the selected included parent across the nearest included parent."
        )
        self.reading_order_earlier_button.clicked.connect(
            self.reading_order_move_earlier_requested
        )
        reading_order_move_actions.addWidget(self.reading_order_earlier_button)
        self.reading_order_later_button = QtWidgets.QPushButton("Later")
        self.reading_order_later_button.setObjectName("readingOrderLaterButton")
        self.reading_order_later_button.setProperty("role", "command")
        self.reading_order_later_button.setProperty("variant", "secondary")
        self.reading_order_later_button.setAccessibleName(
            "Move selected parent later in reading order"
        )
        self.reading_order_later_button.setToolTip(
            "Move the selected included parent across the nearest included parent."
        )
        self.reading_order_later_button.clicked.connect(
            self.reading_order_move_later_requested
        )
        reading_order_move_actions.addWidget(self.reading_order_later_button)
        reading_order_move_actions.addStretch(1)
        reading_order_layout.addLayout(reading_order_move_actions)
        reading_order_commit_actions = QtWidgets.QHBoxLayout()
        reading_order_commit_actions.addStretch(1)
        self.reading_order_cancel_button = QtWidgets.QPushButton("Cancel")
        self.reading_order_cancel_button.setObjectName("readingOrderCancelButton")
        self.reading_order_cancel_button.setProperty("role", "command")
        self.reading_order_cancel_button.setProperty("variant", "secondary")
        self.reading_order_cancel_button.setAccessibleName(
            "Cancel proposed page reading order"
        )
        self.reading_order_cancel_button.setToolTip(
            "Discard only the unapplied page reading-order proposal."
        )
        self.reading_order_cancel_button.clicked.connect(
            self.reading_order_cancel_requested
        )
        reading_order_commit_actions.addWidget(self.reading_order_cancel_button)
        self.reading_order_apply_button = QtWidgets.QPushButton("Apply Order")
        self.reading_order_apply_button.setObjectName("readingOrderApplyButton")
        self.reading_order_apply_button.setProperty("role", "command")
        self.reading_order_apply_button.setProperty("variant", "primary")
        self.reading_order_apply_button.setAccessibleName(
            "Apply proposed page reading order"
        )
        self.reading_order_apply_button.setToolTip(
            "Publish one reversible page-wide order. No pipeline or Preview work starts."
        )
        self.reading_order_apply_button.clicked.connect(
            self.reading_order_apply_requested
        )
        reading_order_commit_actions.addWidget(self.reading_order_apply_button)
        reading_order_layout.addLayout(reading_order_commit_actions)
        self.reading_order_status = QtWidgets.QLabel(
            "Select an included parent to edit page reading order"
        )
        self.reading_order_status.setProperty("role", "secondary")
        self.reading_order_status.setProperty("tone", "muted")
        self.reading_order_status.setWordWrap(True)
        self.reading_order_status.setAccessibleName("Reading order edit status")
        reading_order_layout.addWidget(self.reading_order_status)
        layout.addWidget(reading_order_card)
        merge_parent_card = QtWidgets.QFrame()
        merge_parent_card.setObjectName("mergeParentCard")
        merge_parent_card.setProperty("role", "panel-raised")
        merge_parent_layout = QtWidgets.QVBoxLayout(merge_parent_card)
        merge_parent_layout.setContentsMargins(8, 8, 8, 8)
        merge_parent_layout.setSpacing(6)
        merge_parent_title = QtWidgets.QLabel("Merge Parent")
        merge_parent_title.setProperty("role", "section")
        merge_parent_layout.addWidget(merge_parent_title)
        merge_parent_help = QtWidgets.QLabel(
            "Combine two adjacent compatible pipeline text blocks. The draft "
            "shows their enclosing page range and exact ordered OCR text; Merge "
            "publishes one reversible structural edit and starts no later owner."
        )
        merge_parent_help.setWordWrap(True)
        merge_parent_help.setProperty("role", "secondary")
        merge_parent_layout.addWidget(merge_parent_help)
        merge_parent_form = QtWidgets.QFormLayout()
        merge_parent_form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        merge_parent_form.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.merge_parent_partner = WheelSafeComboBox()
        self.merge_parent_partner.setObjectName("mergeParentPartner")
        self.merge_parent_partner.setAccessibleName("Merge Parent partner")
        self.merge_parent_partner.setAccessibleDescription(
            "Choose one immediately adjacent pipeline text block with the same role."
        )
        self.merge_parent_partner.currentIndexChanged.connect(
            self._merge_parent_partner_value_changed
        )
        merge_parent_form.addRow("Adjacent block", self.merge_parent_partner)
        merge_parent_layout.addLayout(merge_parent_form)
        self.merge_parent_summary = QtWidgets.QLabel("No merge draft")
        self.merge_parent_summary.setWordWrap(True)
        self.merge_parent_summary.setProperty("role", "secondary")
        self.merge_parent_summary.setAccessibleName("Merge Parent draft summary")
        merge_parent_layout.addWidget(self.merge_parent_summary)
        merge_parent_actions = QtWidgets.QHBoxLayout()
        merge_parent_actions.addStretch(1)
        self.merge_parent_cancel_button = QtWidgets.QPushButton("Cancel")
        self.merge_parent_cancel_button.setObjectName("mergeParentCancelButton")
        self.merge_parent_cancel_button.setProperty("role", "command")
        self.merge_parent_cancel_button.setProperty("variant", "secondary")
        self.merge_parent_cancel_button.setAccessibleName("Cancel Merge Parent draft")
        self.merge_parent_cancel_button.setToolTip(
            "Discard only the unapplied Merge Parent draft."
        )
        self.merge_parent_cancel_button.clicked.connect(
            self.merge_parent_cancel_requested
        )
        merge_parent_actions.addWidget(self.merge_parent_cancel_button)
        self.merge_parent_apply_button = QtWidgets.QPushButton("Merge Parent")
        self.merge_parent_apply_button.setObjectName("mergeParentApplyButton")
        self.merge_parent_apply_button.setProperty("role", "command")
        self.merge_parent_apply_button.setProperty("variant", "primary")
        self.merge_parent_apply_button.setAccessibleName(
            "Merge selected pipeline text blocks"
        )
        self.merge_parent_apply_button.setToolTip(
            "Publish one structural merge with enclosing geometry and ordered OCR. "
            "Translation and every later owner remain explicit."
        )
        self.merge_parent_apply_button.clicked.connect(self.merge_parent_requested)
        merge_parent_actions.addWidget(self.merge_parent_apply_button)
        merge_parent_layout.addLayout(merge_parent_actions)
        self.merge_parent_status = QtWidgets.QLabel(
            "Select an eligible pipeline parent to prepare a merge"
        )
        self.merge_parent_status.setProperty("role", "secondary")
        self.merge_parent_status.setProperty("tone", "muted")
        self.merge_parent_status.setWordWrap(True)
        self.merge_parent_status.setAccessibleName("Merge Parent status")
        merge_parent_layout.addWidget(self.merge_parent_status)
        QtWidgets.QWidget.setTabOrder(
            self.merge_parent_partner,
            self.merge_parent_apply_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.merge_parent_apply_button,
            self.merge_parent_cancel_button,
        )
        layout.addWidget(merge_parent_card)
        split_parent_card = QtWidgets.QFrame()
        split_parent_card.setObjectName("splitParentCard")
        split_parent_card.setProperty("role", "panel-raised")
        split_parent_layout = QtWidgets.QVBoxLayout(split_parent_card)
        split_parent_layout.setContentsMargins(8, 8, 8, 8)
        split_parent_layout.setSpacing(6)
        split_parent_title = QtWidgets.QLabel("Split Parent")
        split_parent_title.setProperty("role", "section")
        split_parent_layout.addWidget(split_parent_title)
        split_parent_help = QtWidgets.QLabel(
            "Partition one standalone Add-created user parent into two exact "
            "page rectangles. The draft outline is only a guide; Split publishes "
            "one reversible structural edit and runs no downstream owner."
        )
        split_parent_help.setWordWrap(True)
        split_parent_help.setProperty("role", "secondary")
        split_parent_layout.addWidget(split_parent_help)
        split_parent_form = QtWidgets.QFormLayout()
        split_parent_form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        split_parent_form.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.split_parent_orientation = WheelSafeComboBox()
        self.split_parent_orientation.setObjectName("splitParentOrientation")
        self.split_parent_orientation.addItem("Choose direction", None)
        self.split_parent_orientation.addItem("Vertical (left/right)", "vertical")
        self.split_parent_orientation.addItem("Horizontal (top/bottom)", "horizontal")
        self.split_parent_orientation.setAccessibleName("Split Parent direction")
        self.split_parent_orientation.setAccessibleDescription(
            "Choose left then right or top then bottom child order."
        )
        self.split_parent_orientation.currentIndexChanged.connect(
            self._split_parent_orientation_value_changed
        )
        split_parent_form.addRow("Direction", self.split_parent_orientation)
        self.split_parent_offset = WheelSafeSpinBox()
        self.split_parent_offset.setObjectName("splitParentOffset")
        self.split_parent_offset.setSuffix(" px")
        self.split_parent_offset.setRange(1, 1)
        self.split_parent_offset.setAccessibleName("Split Parent divider position")
        self.split_parent_offset.setAccessibleDescription(
            "Exact page-pixel divider inside the selected workflow area."
        )
        self.split_parent_offset.valueChanged.connect(
            self._split_parent_offset_value_changed
        )
        split_parent_form.addRow("Divider", self.split_parent_offset)
        split_parent_layout.addLayout(split_parent_form)
        self.split_parent_summary = QtWidgets.QLabel("No split draft")
        self.split_parent_summary.setWordWrap(True)
        self.split_parent_summary.setProperty("role", "secondary")
        self.split_parent_summary.setAccessibleName("Split Parent partition summary")
        split_parent_layout.addWidget(self.split_parent_summary)
        split_parent_actions = QtWidgets.QHBoxLayout()
        split_parent_actions.addStretch(1)
        self.split_parent_cancel_button = QtWidgets.QPushButton("Cancel")
        self.split_parent_cancel_button.setObjectName("splitParentCancelButton")
        self.split_parent_cancel_button.setProperty("role", "command")
        self.split_parent_cancel_button.setProperty("variant", "secondary")
        self.split_parent_cancel_button.setAccessibleName("Cancel Split Parent draft")
        self.split_parent_cancel_button.setToolTip(
            "Discard only the unapplied Split Parent partition."
        )
        self.split_parent_cancel_button.clicked.connect(
            self.split_parent_cancel_requested
        )
        split_parent_actions.addWidget(self.split_parent_cancel_button)
        self.split_parent_apply_button = QtWidgets.QPushButton("Split Parent")
        self.split_parent_apply_button.setObjectName("splitParentApplyButton")
        self.split_parent_apply_button.setProperty("role", "command")
        self.split_parent_apply_button.setProperty("variant", "primary")
        self.split_parent_apply_button.setAccessibleName(
            "Split selected standalone user parent"
        )
        self.split_parent_apply_button.setToolTip(
            "Publish one topology-only split. OCR, translation, cleanup, style, "
            "layout, and rendering remain explicit."
        )
        self.split_parent_apply_button.clicked.connect(self.split_parent_requested)
        split_parent_actions.addWidget(self.split_parent_apply_button)
        split_parent_layout.addLayout(split_parent_actions)
        self.split_parent_status = QtWidgets.QLabel(
            "Select a standalone Add-created user parent to split"
        )
        self.split_parent_status.setProperty("role", "secondary")
        self.split_parent_status.setProperty("tone", "muted")
        self.split_parent_status.setWordWrap(True)
        self.split_parent_status.setAccessibleName("Split Parent status")
        split_parent_layout.addWidget(self.split_parent_status)
        QtWidgets.QWidget.setTabOrder(
            self.split_parent_orientation,
            self.split_parent_offset,
        )
        QtWidgets.QWidget.setTabOrder(
            self.split_parent_offset,
            self.split_parent_apply_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.split_parent_apply_button,
            self.split_parent_cancel_button,
        )
        layout.addWidget(split_parent_card)
        geometry_title = QtWidgets.QLabel("Structural Geometry")
        geometry_title.setProperty("role", "section")
        layout.addWidget(geometry_title)
        geometry_help = QtWidgets.QLabel(
            "Page-coordinate parent bounds. This changes upstream crop authority, "
            "not renderer typesetting layout. Revalidation remains explicit."
        )
        geometry_help.setWordWrap(True)
        geometry_help.setProperty("role", "secondary")
        layout.addWidget(geometry_help)
        geometry_card = QtWidgets.QFrame()
        geometry_card.setObjectName("structuralGeometryCard")
        geometry_card.setProperty("role", "panel-raised")
        geometry_layout = QtWidgets.QVBoxLayout(geometry_card)
        geometry_layout.setContentsMargins(8, 8, 8, 8)
        geometry_layout.setSpacing(6)
        self.parent_geometry_summary = QtWidgets.QLabel(
            "Automatic: —\nEffective: —"
        )
        self.parent_geometry_summary.setWordWrap(True)
        self.parent_geometry_summary.setProperty("role", "secondary")
        self.parent_geometry_summary.setAccessibleName(
            "Automatic and effective parent geometry"
        )
        geometry_layout.addWidget(self.parent_geometry_summary)
        geometry_grid = QtWidgets.QGridLayout()
        geometry_grid.setHorizontalSpacing(6)
        geometry_grid.setVerticalSpacing(4)
        self.parent_geometry_spins: dict[str, QtWidgets.QSpinBox] = {}
        for index, (field_name, label_text) in enumerate(
            (("x", "X"), ("y", "Y"), ("width", "W"), ("height", "H"))
        ):
            label = QtWidgets.QLabel(label_text)
            spin = WheelSafeSpinBox()
            spin.setObjectName(f"parentGeometry{field_name.title()}Spin")
            spin.setRange(1 if field_name in {"width", "height"} else 0, 1)
            spin.setAccessibleName(f"Parent geometry {field_name}")
            spin.setToolTip(
                "Exact page pixel coordinate. Apply publishes a reversible "
                "structural edit; the canvas outline is only a draft."
            )
            spin.valueChanged.connect(self._parent_geometry_value_changed)
            row, column = divmod(index, 2)
            geometry_grid.addWidget(label, row, column * 2)
            geometry_grid.addWidget(spin, row, column * 2 + 1)
            self.parent_geometry_spins[field_name] = spin
        geometry_layout.addLayout(geometry_grid)
        self.parent_geometry_canvas = QtWidgets.QLabel("Page: —")
        self.parent_geometry_canvas.setProperty("role", "secondary")
        geometry_layout.addWidget(self.parent_geometry_canvas)
        geometry_actions = QtWidgets.QHBoxLayout()
        geometry_actions.addStretch(1)
        self.parent_geometry_cancel_button = QtWidgets.QPushButton("Cancel")
        self.parent_geometry_cancel_button.setProperty("role", "command")
        self.parent_geometry_cancel_button.setProperty("variant", "secondary")
        self.parent_geometry_cancel_button.setAccessibleName(
            "Cancel parent geometry draft"
        )
        self.parent_geometry_cancel_button.clicked.connect(
            self.parent_geometry_cancel_requested
        )
        geometry_actions.addWidget(self.parent_geometry_cancel_button)
        self.parent_geometry_apply_button = QtWidgets.QPushButton("Apply Geometry")
        self.parent_geometry_apply_button.setProperty("role", "command")
        self.parent_geometry_apply_button.setProperty("variant", "primary")
        self.parent_geometry_apply_button.setAccessibleName(
            "Apply selected parent structural geometry"
        )
        self.parent_geometry_apply_button.setToolTip(
            "Save this parent bbox. OCR, translation, cleanup, style, and "
            "rendering are not run automatically."
        )
        self.parent_geometry_apply_button.clicked.connect(
            self.parent_geometry_apply_requested
        )
        geometry_actions.addWidget(self.parent_geometry_apply_button)
        geometry_layout.addLayout(geometry_actions)
        self.parent_geometry_status = QtWidgets.QLabel(
            "Select a parent to edit structural geometry"
        )
        self.parent_geometry_status.setProperty("role", "secondary")
        self.parent_geometry_status.setProperty("tone", "muted")
        self.parent_geometry_status.setWordWrap(True)
        self.parent_geometry_status.setAccessibleName("Parent geometry edit status")
        geometry_layout.addWidget(self.parent_geometry_status)
        layout.addWidget(geometry_card)
        render_layout_title = QtWidgets.QLabel("Render Layout")
        render_layout_title.setProperty("role", "section")
        layout.addWidget(render_layout_title)
        writing_mode_card = QtWidgets.QFrame()
        writing_mode_card.setObjectName("writingModeCard")
        writing_mode_card.setProperty("role", "panel-raised")
        writing_mode_layout = QtWidgets.QVBoxLayout(writing_mode_card)
        writing_mode_layout.setContentsMargins(8, 8, 8, 8)
        writing_mode_layout.setSpacing(6)
        writing_mode_title = QtWidgets.QLabel("Writing Mode")
        writing_mode_title.setProperty("role", "section")
        writing_mode_layout.addWidget(writing_mode_title)
        writing_mode_help = QtWidgets.QLabel(
            "Choose the selected parent's renderer-backed text direction. "
            "Set and Restore Automatic only publish this layout field; Preview "
            "remains explicit."
        )
        writing_mode_help.setWordWrap(True)
        writing_mode_help.setProperty("role", "secondary")
        writing_mode_layout.addWidget(writing_mode_help)
        writing_mode_summary = QtWidgets.QFormLayout()
        writing_mode_summary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        writing_mode_summary.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.writing_mode_automatic = QtWidgets.QLabel("—")
        self.writing_mode_automatic.setWordWrap(True)
        self.writing_mode_automatic.setAccessibleName("Automatic writing mode")
        writing_mode_summary.addRow("Automatic", self.writing_mode_automatic)
        self.writing_mode_user = QtWidgets.QLabel("No edit")
        self.writing_mode_user.setWordWrap(True)
        self.writing_mode_user.setAccessibleName("User writing mode edit")
        writing_mode_summary.addRow("Your edit", self.writing_mode_user)
        self.writing_mode_effective = QtWidgets.QLabel("—")
        self.writing_mode_effective.setWordWrap(True)
        self.writing_mode_effective.setAccessibleName("Effective writing mode")
        writing_mode_summary.addRow("Effective", self.writing_mode_effective)
        self.writing_mode_authority = QtWidgets.QLabel("—")
        self.writing_mode_authority.setWordWrap(True)
        self.writing_mode_authority.setAccessibleName("Writing mode authority")
        writing_mode_summary.addRow("Authority", self.writing_mode_authority)
        writing_mode_layout.addLayout(writing_mode_summary)
        self.writing_mode_combo = WheelSafeComboBox()
        self.writing_mode_combo.setObjectName("writingModeCombo")
        self.writing_mode_combo.setAccessibleName("Selected parent writing mode")
        self.writing_mode_combo.setAccessibleDescription(
            "Horizontal stores horizontal. Vertical stores vertical. Set publishes "
            "only the selected parent's writing-mode field."
        )
        self.writing_mode_combo.addItem("Horizontal", "horizontal")
        self.writing_mode_combo.addItem("Vertical", "vertical")
        self.writing_mode_combo.currentIndexChanged.connect(
            self._writing_mode_value_changed
        )
        writing_mode_layout.addWidget(self.writing_mode_combo)
        self.writing_mode_set_button = QtWidgets.QPushButton("Set")
        self.writing_mode_set_button.setObjectName("writingModeSetButton")
        self.writing_mode_set_button.setProperty("role", "command")
        self.writing_mode_set_button.setProperty("variant", "primary")
        self.writing_mode_set_button.setAccessibleName(
            "Set selected parent writing mode"
        )
        self.writing_mode_set_button.setToolTip(
            "Publish only this writing-mode edit. Rendering does not start automatically."
        )
        self.writing_mode_set_button.clicked.connect(
            self.writing_mode_apply_requested
        )
        writing_mode_layout.addWidget(self.writing_mode_set_button)
        self.writing_mode_cancel_button = QtWidgets.QPushButton("Cancel")
        self.writing_mode_cancel_button.setObjectName("writingModeCancelButton")
        self.writing_mode_cancel_button.setProperty("role", "command")
        self.writing_mode_cancel_button.setProperty("variant", "secondary")
        self.writing_mode_cancel_button.setAccessibleName(
            "Cancel writing mode draft"
        )
        self.writing_mode_cancel_button.setToolTip(
            "Discard only the unapplied writing-mode selection."
        )
        self.writing_mode_cancel_button.clicked.connect(
            self.writing_mode_cancel_requested
        )
        writing_mode_layout.addWidget(self.writing_mode_cancel_button)
        self.writing_mode_restore_button = QtWidgets.QPushButton(
            "Restore Automatic"
        )
        self.writing_mode_restore_button.setObjectName(
            "writingModeRestoreButton"
        )
        self.writing_mode_restore_button.setProperty("role", "command")
        self.writing_mode_restore_button.setProperty("variant", "secondary")
        self.writing_mode_restore_button.setAccessibleName(
            "Restore automatic writing mode"
        )
        self.writing_mode_restore_button.setToolTip(
            "Remove only the selected parent's writing-mode override. Automatic "
            "render-style evidence remains unchanged."
        )
        self.writing_mode_restore_button.clicked.connect(
            self.writing_mode_restore_requested
        )
        writing_mode_layout.addWidget(self.writing_mode_restore_button)
        self.writing_mode_status = QtWidgets.QLabel(
            "Select a render-required parent to edit writing mode"
        )
        self.writing_mode_status.setProperty("role", "secondary")
        self.writing_mode_status.setProperty("tone", "muted")
        self.writing_mode_status.setWordWrap(True)
        self.writing_mode_status.setAccessibleName("Writing mode edit status")
        writing_mode_layout.addWidget(self.writing_mode_status)
        layout.addWidget(writing_mode_card)
        line_height_card = QtWidgets.QFrame()
        line_height_card.setObjectName("lineHeightCard")
        line_height_card.setProperty("role", "panel-raised")
        line_height_layout = QtWidgets.QVBoxLayout(line_height_card)
        line_height_layout.setContentsMargins(8, 8, 8, 8)
        line_height_layout.setSpacing(6)
        line_height_title = QtWidgets.QLabel("Line Height")
        line_height_title.setProperty("role", "section")
        line_height_layout.addWidget(line_height_title)
        line_height_help = QtWidgets.QLabel(
            "Set an exact renderer-backed line-height ratio from 0.5 through "
            "10.0 for the selected parent. Set and Restore Automatic publish "
            "only this layout field; Preview remains explicit."
        )
        line_height_help.setWordWrap(True)
        line_height_help.setProperty("role", "secondary")
        line_height_layout.addWidget(line_height_help)
        line_height_summary = QtWidgets.QFormLayout()
        line_height_summary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        line_height_summary.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.line_height_automatic = QtWidgets.QLabel("—")
        self.line_height_automatic.setWordWrap(True)
        self.line_height_automatic.setAccessibleName("Automatic line height")
        line_height_summary.addRow("Automatic", self.line_height_automatic)
        self.line_height_user = QtWidgets.QLabel("No edit")
        self.line_height_user.setWordWrap(True)
        self.line_height_user.setAccessibleName("User line height edit")
        line_height_summary.addRow("Your edit", self.line_height_user)
        self.line_height_effective = QtWidgets.QLabel("—")
        self.line_height_effective.setWordWrap(True)
        self.line_height_effective.setAccessibleName("Effective line height")
        line_height_summary.addRow("Effective", self.line_height_effective)
        self.line_height_authority = QtWidgets.QLabel("—")
        self.line_height_authority.setWordWrap(True)
        self.line_height_authority.setAccessibleName("Line height authority")
        line_height_summary.addRow("Authority", self.line_height_authority)
        line_height_layout.addLayout(line_height_summary)
        self.line_height_spin = WheelSafeDoubleSpinBox()
        self.line_height_spin.setObjectName("lineHeightSpinBox")
        self.line_height_spin.setRange(0.5, 10.0)
        self.line_height_spin.setDecimals(6)
        self.line_height_spin.setSingleStep(0.05)
        self.line_height_spin.setKeyboardTracking(False)
        self.line_height_spin.setAccessibleName("Selected parent line height")
        self.line_height_spin.setAccessibleDescription(
            "Exact ratio from 0.5 through 10.0. Set publishes only the selected "
            "parent's line-height field."
        )
        self.line_height_spin.valueChanged.connect(
            self._line_height_value_changed
        )
        line_height_layout.addWidget(self.line_height_spin)
        self.line_height_set_button = QtWidgets.QPushButton("Set")
        self.line_height_set_button.setObjectName("lineHeightSetButton")
        self.line_height_set_button.setProperty("role", "command")
        self.line_height_set_button.setProperty("variant", "primary")
        self.line_height_set_button.setAccessibleName(
            "Set selected parent line height"
        )
        self.line_height_set_button.setToolTip(
            "Publish only this line-height edit. Rendering does not start automatically."
        )
        self.line_height_set_button.clicked.connect(
            self.line_height_apply_requested
        )
        line_height_layout.addWidget(self.line_height_set_button)
        self.line_height_cancel_button = QtWidgets.QPushButton("Cancel")
        self.line_height_cancel_button.setObjectName("lineHeightCancelButton")
        self.line_height_cancel_button.setProperty("role", "command")
        self.line_height_cancel_button.setProperty("variant", "secondary")
        self.line_height_cancel_button.setAccessibleName(
            "Cancel line height draft"
        )
        self.line_height_cancel_button.setToolTip(
            "Discard only the unapplied line-height ratio."
        )
        self.line_height_cancel_button.clicked.connect(
            self.line_height_cancel_requested
        )
        line_height_layout.addWidget(self.line_height_cancel_button)
        self.line_height_restore_button = QtWidgets.QPushButton(
            "Restore Automatic"
        )
        self.line_height_restore_button.setObjectName(
            "lineHeightRestoreButton"
        )
        self.line_height_restore_button.setProperty("role", "command")
        self.line_height_restore_button.setProperty("variant", "secondary")
        self.line_height_restore_button.setAccessibleName(
            "Restore automatic line height"
        )
        self.line_height_restore_button.setToolTip(
            "Remove only the selected parent's line-height override. Automatic "
            "render-style evidence remains unchanged."
        )
        self.line_height_restore_button.clicked.connect(
            self.line_height_restore_requested
        )
        line_height_layout.addWidget(self.line_height_restore_button)
        self.line_height_status = QtWidgets.QLabel(
            "Select a render-required parent to edit line height"
        )
        self.line_height_status.setProperty("role", "secondary")
        self.line_height_status.setProperty("tone", "muted")
        self.line_height_status.setWordWrap(True)
        self.line_height_status.setAccessibleName("Line height edit status")
        line_height_layout.addWidget(self.line_height_status)
        layout.addWidget(line_height_card)
        rotation_card = QtWidgets.QFrame()
        rotation_card.setObjectName("rotationCard")
        rotation_card.setProperty("role", "panel-raised")
        rotation_layout = QtWidgets.QVBoxLayout(rotation_card)
        rotation_layout.setContentsMargins(8, 8, 8, 8)
        rotation_layout.setSpacing(6)
        rotation_title = QtWidgets.QLabel("Rotation")
        rotation_title.setProperty("role", "section")
        rotation_layout.addWidget(rotation_title)
        rotation_help = QtWidgets.QLabel(
            "Set exact clockwise rotation from -45 through 45 degrees for the "
            "selected parent. The renderer keeps the visual-center pivot. Set "
            "and Restore Automatic publish only this layout field; Preview "
            "remains explicit."
        )
        rotation_help.setWordWrap(True)
        rotation_help.setProperty("role", "secondary")
        rotation_layout.addWidget(rotation_help)
        rotation_summary = QtWidgets.QFormLayout()
        rotation_summary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        rotation_summary.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.rotation_automatic = QtWidgets.QLabel("—")
        self.rotation_automatic.setWordWrap(True)
        self.rotation_automatic.setAccessibleName("Automatic rotation")
        rotation_summary.addRow("Automatic", self.rotation_automatic)
        self.rotation_user = QtWidgets.QLabel("No edit")
        self.rotation_user.setWordWrap(True)
        self.rotation_user.setAccessibleName("User rotation edit")
        rotation_summary.addRow("Your edit", self.rotation_user)
        self.rotation_effective = QtWidgets.QLabel("—")
        self.rotation_effective.setWordWrap(True)
        self.rotation_effective.setAccessibleName("Effective rotation")
        rotation_summary.addRow("Effective", self.rotation_effective)
        self.rotation_authority = QtWidgets.QLabel("—")
        self.rotation_authority.setWordWrap(True)
        self.rotation_authority.setAccessibleName("Rotation authority")
        rotation_summary.addRow("Authority", self.rotation_authority)
        rotation_layout.addLayout(rotation_summary)
        self.rotation_spin = WheelSafeDoubleSpinBox()
        self.rotation_spin.setObjectName("rotationSpinBox")
        self.rotation_spin.setRange(-45.0, 45.0)
        self.rotation_spin.setDecimals(6)
        self.rotation_spin.setSingleStep(1.0)
        self.rotation_spin.setSuffix("°")
        self.rotation_spin.setKeyboardTracking(False)
        self.rotation_spin.setAccessibleName(
            "Selected parent clockwise rotation"
        )
        self.rotation_spin.setAccessibleDescription(
            "Exact clockwise degrees from -45 through 45. Set publishes only "
            "the selected parent's rotation field around its visual center."
        )
        self.rotation_spin.valueChanged.connect(self._rotation_value_changed)
        rotation_layout.addWidget(self.rotation_spin)
        self.rotation_set_button = QtWidgets.QPushButton("Set")
        self.rotation_set_button.setObjectName("rotationSetButton")
        self.rotation_set_button.setProperty("role", "command")
        self.rotation_set_button.setProperty("variant", "primary")
        self.rotation_set_button.setAccessibleName(
            "Set selected parent rotation"
        )
        self.rotation_set_button.setToolTip(
            "Publish only this clockwise rotation edit. Rendering does not "
            "start automatically."
        )
        self.rotation_set_button.clicked.connect(self.rotation_apply_requested)
        rotation_layout.addWidget(self.rotation_set_button)
        self.rotation_cancel_button = QtWidgets.QPushButton("Cancel")
        self.rotation_cancel_button.setObjectName("rotationCancelButton")
        self.rotation_cancel_button.setProperty("role", "command")
        self.rotation_cancel_button.setProperty("variant", "secondary")
        self.rotation_cancel_button.setAccessibleName("Cancel rotation draft")
        self.rotation_cancel_button.setToolTip(
            "Discard only the unapplied clockwise rotation."
        )
        self.rotation_cancel_button.clicked.connect(
            self.rotation_cancel_requested
        )
        rotation_layout.addWidget(self.rotation_cancel_button)
        self.rotation_restore_button = QtWidgets.QPushButton(
            "Restore Automatic"
        )
        self.rotation_restore_button.setObjectName("rotationRestoreButton")
        self.rotation_restore_button.setProperty("role", "command")
        self.rotation_restore_button.setProperty("variant", "secondary")
        self.rotation_restore_button.setAccessibleName(
            "Restore automatic rotation"
        )
        self.rotation_restore_button.setToolTip(
            "Remove only the selected parent's rotation override. Automatic "
            "parent-layer-effect evidence remains unchanged."
        )
        self.rotation_restore_button.clicked.connect(
            self.rotation_restore_requested
        )
        rotation_layout.addWidget(self.rotation_restore_button)
        self.rotation_status = QtWidgets.QLabel(
            "Select a render-required parent to edit rotation"
        )
        self.rotation_status.setProperty("role", "secondary")
        self.rotation_status.setProperty("tone", "muted")
        self.rotation_status.setWordWrap(True)
        self.rotation_status.setAccessibleName("Rotation edit status")
        rotation_layout.addWidget(self.rotation_status)
        layout.addWidget(rotation_card)
        render_box_card = QtWidgets.QFrame()
        render_box_card.setObjectName("renderBoxCard")
        render_box_card.setProperty("role", "panel-raised")
        render_box_layout = QtWidgets.QVBoxLayout(render_box_card)
        render_box_layout.setContentsMargins(8, 8, 8, 8)
        render_box_layout.setSpacing(6)
        render_box_title = QtWidgets.QLabel("Render Box")
        render_box_title.setProperty("role", "section")
        render_box_layout.addWidget(render_box_title)
        render_box_help = QtWidgets.QLabel(
            "Set the exact renderer target rectangle inside immutable automatic "
            "hard bounds. Structural parent geometry and Preview remain separate."
        )
        render_box_help.setWordWrap(True)
        render_box_help.setProperty("role", "secondary")
        render_box_layout.addWidget(render_box_help)
        render_box_summary = QtWidgets.QFormLayout()
        render_box_summary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        render_box_summary.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.render_box_automatic = QtWidgets.QLabel("—")
        self.render_box_automatic.setWordWrap(True)
        self.render_box_automatic.setAccessibleName("Automatic render box")
        render_box_summary.addRow("Automatic", self.render_box_automatic)
        self.render_box_hard_bounds = QtWidgets.QLabel("—")
        self.render_box_hard_bounds.setWordWrap(True)
        self.render_box_hard_bounds.setAccessibleName("Automatic render hard bounds")
        render_box_summary.addRow("Hard bounds", self.render_box_hard_bounds)
        self.render_box_user = QtWidgets.QLabel("No edit")
        self.render_box_user.setWordWrap(True)
        self.render_box_user.setAccessibleName("User render-box edit")
        render_box_summary.addRow("Your edit", self.render_box_user)
        self.render_box_effective = QtWidgets.QLabel("—")
        self.render_box_effective.setWordWrap(True)
        self.render_box_effective.setAccessibleName("Effective render box")
        render_box_summary.addRow("Effective", self.render_box_effective)
        self.render_box_authority = QtWidgets.QLabel("—")
        self.render_box_authority.setWordWrap(True)
        self.render_box_authority.setAccessibleName("Render-box authority")
        render_box_summary.addRow("Authority", self.render_box_authority)
        render_box_layout.addLayout(render_box_summary)
        render_box_grid = QtWidgets.QGridLayout()
        self.render_box_spins: dict[str, QtWidgets.QSpinBox] = {}
        for index, (field_name, label) in enumerate(
            (("x", "X"), ("y", "Y"), ("width", "Width"), ("height", "Height"))
        ):
            caption = QtWidgets.QLabel(label)
            spin = WheelSafeSpinBox()
            spin.setObjectName(f"renderBox{field_name.title()}SpinBox")
            spin.setRange(-100000 if field_name in {"x", "y"} else 1, 100000)
            spin.setKeyboardTracking(False)
            spin.setAccessibleName(f"Selected parent render box {label.lower()}")
            spin.setAccessibleDescription(
                "Exact integer renderer target-box component. The completed box "
                "must stay inside automatic hard bounds."
            )
            spin.valueChanged.connect(self._render_box_value_changed)
            self.render_box_spins[field_name] = spin
            row, column = divmod(index, 2)
            render_box_grid.addWidget(caption, row * 2, column)
            render_box_grid.addWidget(spin, row * 2 + 1, column)
        render_box_layout.addLayout(render_box_grid)
        render_box_actions = QtWidgets.QHBoxLayout()
        self.render_box_set_button = QtWidgets.QPushButton("Set")
        self.render_box_set_button.setObjectName("renderBoxSetButton")
        self.render_box_set_button.setProperty("role", "command")
        self.render_box_set_button.setProperty("variant", "primary")
        self.render_box_set_button.setAccessibleName("Set selected parent render box")
        self.render_box_set_button.setToolTip(
            "Publish only this render-box edit. Rendering does not start automatically."
        )
        self.render_box_set_button.clicked.connect(self.render_box_apply_requested)
        render_box_actions.addWidget(self.render_box_set_button)
        self.render_box_cancel_button = QtWidgets.QPushButton("Cancel")
        self.render_box_cancel_button.setObjectName("renderBoxCancelButton")
        self.render_box_cancel_button.setProperty("role", "command")
        self.render_box_cancel_button.setProperty("variant", "secondary")
        self.render_box_cancel_button.setAccessibleName("Cancel render-box draft")
        self.render_box_cancel_button.setToolTip("Discard only the unapplied render box.")
        self.render_box_cancel_button.clicked.connect(self.render_box_cancel_requested)
        render_box_actions.addWidget(self.render_box_cancel_button)
        self.render_box_restore_button = QtWidgets.QPushButton("Restore Automatic")
        self.render_box_restore_button.setObjectName("renderBoxRestoreButton")
        self.render_box_restore_button.setProperty("role", "command")
        self.render_box_restore_button.setProperty("variant", "secondary")
        self.render_box_restore_button.setAccessibleName("Restore automatic render box")
        self.render_box_restore_button.setToolTip(
            "Remove only the selected parent's target-box override. Automatic "
            "hard bounds and structural geometry remain immutable."
        )
        self.render_box_restore_button.clicked.connect(
            self.render_box_restore_requested
        )
        render_box_actions.addWidget(self.render_box_restore_button)
        render_box_layout.addLayout(render_box_actions)
        self.render_box_status = QtWidgets.QLabel(
            "Select a render-required parent to edit its render box"
        )
        self.render_box_status.setProperty("role", "secondary")
        self.render_box_status.setProperty("tone", "muted")
        self.render_box_status.setWordWrap(True)
        self.render_box_status.setAccessibleName("Render-box edit status")
        render_box_layout.addWidget(self.render_box_status)
        QtWidgets.QWidget.setTabOrder(
            self.render_box_spins["x"], self.render_box_spins["y"]
        )
        QtWidgets.QWidget.setTabOrder(
            self.render_box_spins["y"], self.render_box_spins["width"]
        )
        QtWidgets.QWidget.setTabOrder(
            self.render_box_spins["width"], self.render_box_spins["height"]
        )
        QtWidgets.QWidget.setTabOrder(
            self.render_box_spins["height"], self.render_box_set_button
        )
        QtWidgets.QWidget.setTabOrder(
            self.render_box_set_button, self.render_box_cancel_button
        )
        QtWidgets.QWidget.setTabOrder(
            self.render_box_cancel_button, self.render_box_restore_button
        )
        layout.addWidget(render_box_card)
        QtWidgets.QWidget.setTabOrder(
            self.rotation_restore_button,
            self.render_box_spins["x"],
        )
        self._layout_detail_cards = (
            reading_order_card,
            merge_parent_card,
            split_parent_card,
            geometry_card,
            writing_mode_card,
            line_height_card,
            rotation_card,
            render_box_card,
        )
        self.layout_resolved_card = QtWidgets.QFrame()
        self.layout_resolved_card.setObjectName("layoutResolvedCard")
        layout_resolved_layout = QtWidgets.QVBoxLayout(self.layout_resolved_card)
        layout_resolved_layout.setContentsMargins(8, 8, 8, 8)
        layout_resolved_layout.setSpacing(8)
        layout_note = QtWidgets.QLabel(
            "Adjust the selected parent inside page bounds. Automatic structural "
            "evidence remains unchanged beneath explicit overrides."
        )
        layout_note.setObjectName("layoutResolvedNotice")
        layout_note.setProperty("role", "status-banner")
        layout_note.setProperty("tone", "info")
        layout_note.setWordWrap(True)
        layout_resolved_layout.addWidget(layout_note)
        layout_grid = QtWidgets.QGridLayout()
        layout_grid.setHorizontalSpacing(8)
        layout_grid.setVerticalSpacing(6)
        self.layout_facade_fields: dict[str, QtWidgets.QLineEdit] = {}
        for index, (field_id, label, accessible_name) in enumerate(
            (
                ("x", "X", "Effective render box x"),
                ("y", "Y", "Effective render box y"),
                ("width", "Width", "Effective render box width"),
                ("height", "Height", "Effective render box height"),
                ("rotation", "Rotation", "Effective rotation"),
                ("line_height", "Line height", "Effective line height"),
                ("letter_spacing", "Letter spacing", "Effective letter spacing"),
                ("column_spacing", "Column spacing", "Effective column spacing"),
                ("writing_mode", "Writing mode", "Effective writing mode"),
                ("alignment", "Alignment", "Effective text alignment"),
            )
        ):
            row, column = divmod(index, 2)
            field_column = column * 2
            caption = QtWidgets.QLabel(label)
            caption.setProperty("role", "secondary")
            field = self._facade_readout(accessible_name)
            self.layout_facade_fields[field_id] = field
            layout_grid.addWidget(caption, row * 2, field_column)
            layout_grid.addWidget(field, row * 2 + 1, field_column)
        layout_resolved_layout.addLayout(layout_grid)
        layout_actions = QtWidgets.QHBoxLayout()
        self.layout_preview_button = QtWidgets.QPushButton("Preview layout")
        self.layout_preview_button.setObjectName("previewLayoutButton")
        self.layout_preview_button.setProperty("role", "command")
        self.layout_preview_button.setProperty("variant", "primary")
        self.layout_preview_button.setIcon(hybrid_icon("play"))
        self.layout_preview_button.setAccessibleName("Preview layout on this page")
        self.layout_preview_button.clicked.connect(
            lambda: self.rerender_requested.emit(self._current_page_id)
            if self._current_page_id
            else None
        )
        layout_actions.addWidget(self.layout_preview_button, 1)
        self.layout_more_button = QtWidgets.QToolButton()
        self.layout_more_button.setObjectName("layoutAdvancedOverridesButton")
        self.layout_more_button.setCheckable(True)
        self.layout_more_button.setIcon(hybrid_icon("more"))
        self.layout_more_button.setText("Edit layout")
        self.layout_more_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.layout_more_button.setToolTip("Show explicit layout and topology controls")
        self.layout_more_button.setAccessibleName(
            "Show explicit layout and topology controls"
        )
        self.layout_more_button.toggled.connect(self._set_layout_details_visible)
        layout_actions.addWidget(self.layout_more_button)
        layout_resolved_layout.addLayout(layout_actions)
        self.layout_preview_button.setVisible(False)
        self.layout_more_button.setVisible(True)
        layout.insertWidget(0, self.layout_resolved_card)

        self.layout_legacy_values = QtWidgets.QWidget()
        layout_legacy_layout = QtWidgets.QVBoxLayout(self.layout_legacy_values)
        layout_legacy_layout.setContentsMargins(0, 0, 0, 0)
        self.layout_form = QtWidgets.QFormLayout()
        self.layout_form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        layout_legacy_layout.addLayout(self.layout_form)
        self.layout_diagnostic = QtWidgets.QLabel("No layout diagnostics")
        self.layout_diagnostic.setWordWrap(True)
        self.layout_diagnostic.setProperty("role", "secondary")
        layout_legacy_layout.addWidget(self.layout_diagnostic)
        layout.addWidget(self.layout_legacy_values)
        # Keep the resolved layout summary primary.  Structural and field-level
        # edits stay explicitly available through the More menu.
        self._set_layout_details_visible(False)
        layout.addStretch(1)
        return scroll

    def _set_layout_details_visible(self, visible: bool) -> None:
        show = bool(visible)
        for card in getattr(self, "_layout_detail_cards", ()):
            card.setVisible(show)
        if hasattr(self, "layout_legacy_values"):
            self.layout_legacy_values.setVisible(show)
        if hasattr(self, "layout_more_button"):
            blocker = QtCore.QSignalBlocker(self.layout_more_button)
            self.layout_more_button.setChecked(show)
            del blocker
            self.layout_more_button.setToolTip(
                "Hide explicit layout and topology controls"
                if show
                else "Show explicit layout and topology controls"
            )
            self.layout_more_button.setText(
                "Hide controls" if show else "Edit layout"
            )
            self.layout_more_button.setAccessibleName(
                "Hide explicit layout and topology controls"
                if show
                else "Show explicit layout and topology controls"
            )
        if (
            hasattr(self, "inspector_toggle_details_action")
            and self.inspector_tabs.currentIndex()
            == self._inspector_index.get("layout")
        ):
            self.inspector_toggle_details_action.setText(
                "Hide explicit override controls"
                if show
                else "Show explicit override controls"
            )

    def _update_layout_facade(
        self,
        *,
        layout_values: Mapping[str, object],
        effective_render_box: tuple[int, int, int, int] | None,
        effective_rotation: float | None,
        effective_line_height: float | None,
        effective_writing_mode: str | None,
    ) -> None:
        if not hasattr(self, "layout_facade_fields"):
            return

        def _number(value: object, suffix: str = "") -> str:
            if value is None:
                return "—"
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return str(value)
            text = str(int(numeric)) if numeric.is_integer() else f"{numeric:g}"
            return f"{text}{suffix}"

        x, y, width, height = effective_render_box or (None, None, None, None)
        values = {
            "x": _number(x, " px"),
            "y": _number(y, " px"),
            "width": _number(width, " px"),
            "height": _number(height, " px"),
            "rotation": _number(effective_rotation, "°"),
            "line_height": _number(effective_line_height),
            "letter_spacing": _number(layout_values.get("letter_spacing"), " px"),
            "column_spacing": _number(layout_values.get("column_spacing"), " px"),
            "writing_mode": {
                "vertical": "Vertical right-to-left",
                "vertical_rl": "Vertical right-to-left",
                "horizontal": "Horizontal top-to-bottom",
                "horizontal_tb": "Horizontal top-to-bottom",
            }.get(str(effective_writing_mode or ""), "—"),
            "alignment": str(layout_values.get("alignment") or "—").replace("_", " ").title(),
        }
        for field_id, field in self.layout_facade_fields.items():
            field.setText(values[field_id])
        self.layout_preview_button.setEnabled(bool(self._current_page_id))

    def _build_cleanup_tab(self) -> QtWidgets.QWidget:
        scroll, layout = self._scroll_tab()
        intro = QtWidgets.QLabel(
            "Refine the page-owned clean base. Preview the affected region "
            "before committing a new cleanup revision."
        )
        intro.setObjectName("cleanupInspectorNotice")
        intro.setProperty("role", "status-banner")
        intro.setProperty("tone", "info")
        intro.setWordWrap(True)
        intro.setAccessibleName("Cleanup editor guidance")
        layout.addWidget(intro)

        self._cleanup_facade_tool = "brush"
        self._cleanup_facade_enabled = False
        self.cleanup_tool_group = QtWidgets.QButtonGroup(self)
        self.cleanup_tool_group.setExclusive(True)
        self.cleanup_tool_buttons: dict[str, QtWidgets.QToolButton] = {}
        tools = QtWidgets.QHBoxLayout()
        tools.setSpacing(5)
        for tool_id, label, icon_name in (
            ("brush", "Brush", "cleanup"),
            ("lasso", "Lasso", "lasso"),
            ("rectangle", "Rectangle", "rectangle"),
            ("eraser", "Eraser", "eraser"),
            ("protect", "Protect", "shield"),
        ):
            button = QtWidgets.QToolButton()
            button.setObjectName(f"cleanupFacade{tool_id.title()}Button")
            button.setText(label)
            button.setIcon(hybrid_icon(icon_name, self._icon_theme))
            button.setToolButtonStyle(
                QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon
            )
            button.setCheckable(True)
            button.setAccessibleName(f"Select {label.lower()} cleanup tool")
            button.setToolTip(
                f"Prepare the {label.lower()} tool for the manual cleanup workspace."
            )
            button.clicked.connect(
                lambda _checked=False, value=tool_id: self._set_cleanup_facade_tool(
                    value
                )
            )
            self.cleanup_tool_group.addButton(button)
            self.cleanup_tool_buttons[tool_id] = button
            tools.addWidget(button, 1)
        self.cleanup_tool_buttons["brush"].setChecked(True)
        layout.addLayout(tools)

        brush_heading = QtWidgets.QHBoxLayout()
        brush_label = QtWidgets.QLabel("Brush size")
        brush_label.setProperty("role", "secondary")
        self.cleanup_brush_value = QtWidgets.QLabel("24 px")
        self.cleanup_brush_value.setProperty("role", "section")
        brush_heading.addWidget(brush_label)
        brush_heading.addStretch(1)
        brush_heading.addWidget(self.cleanup_brush_value)
        layout.addLayout(brush_heading)
        self.cleanup_brush_size = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.cleanup_brush_size.setObjectName("cleanupFacadeBrushSize")
        self.cleanup_brush_size.setRange(4, 64)
        self.cleanup_brush_size.setValue(24)
        self.cleanup_brush_size.setAccessibleName("Cleanup brush size")
        self.cleanup_brush_size.valueChanged.connect(
            lambda value: self.cleanup_brush_value.setText(f"{value} px")
        )
        self.cleanup_brush_size.valueChanged.connect(
            self._update_cleanup_facade_dirty
        )
        layout.addWidget(self.cleanup_brush_size)

        parameter_grid = QtWidgets.QGridLayout()
        self.cleanup_grow = WheelSafeSpinBox()
        self.cleanup_grow.setObjectName("cleanupFacadeGrow")
        self.cleanup_grow.setRange(0, 24)
        self.cleanup_grow.setValue(2)
        self.cleanup_grow.setSuffix(" px")
        self.cleanup_grow.setAccessibleName("Cleanup mask grow")
        self.cleanup_feather = WheelSafeSpinBox()
        self.cleanup_feather.setObjectName("cleanupFacadeFeather")
        self.cleanup_feather.setRange(0, 16)
        self.cleanup_feather.setValue(1)
        self.cleanup_feather.setSuffix(" px")
        self.cleanup_feather.setAccessibleName("Cleanup mask feather")
        self.cleanup_grow.valueChanged.connect(self._update_cleanup_facade_dirty)
        self.cleanup_feather.valueChanged.connect(
            self._update_cleanup_facade_dirty
        )
        parameter_grid.addWidget(QtWidgets.QLabel("Grow"), 0, 0)
        parameter_grid.addWidget(QtWidgets.QLabel("Feather"), 0, 1)
        parameter_grid.addWidget(self.cleanup_grow, 1, 0)
        parameter_grid.addWidget(self.cleanup_feather, 1, 1)
        self.cleanup_show_mask = QtWidgets.QCheckBox("Show mask")
        self.cleanup_show_mask.setChecked(True)
        self.cleanup_show_mask.setAccessibleName("Show cleanup erase mask")
        self.cleanup_show_protected = QtWidgets.QCheckBox("Show protected")
        self.cleanup_show_protected.setChecked(True)
        self.cleanup_show_protected.setAccessibleName(
            "Show cleanup protected pixels"
        )
        self.cleanup_show_mask.toggled.connect(
            self._sync_cleanup_overlay_visibility
        )
        self.cleanup_show_protected.toggled.connect(
            self._sync_cleanup_overlay_visibility
        )
        parameter_grid.addWidget(self.cleanup_show_mask, 2, 0)
        parameter_grid.addWidget(self.cleanup_show_protected, 2, 1)
        layout.addLayout(parameter_grid)

        proof = QtWidgets.QFrame()
        proof.setObjectName("cleanupProofRow")
        proof.setProperty("role", "panel-raised")
        proof_layout = QtWidgets.QHBoxLayout(proof)
        proof_layout.setContentsMargins(8, 6, 8, 6)
        proof_icon = QtWidgets.QLabel()
        proof_icon.setPixmap(hybrid_icon("shield", self._icon_theme).pixmap(16, 16))
        proof_layout.addWidget(proof_icon)
        proof_copy = QtWidgets.QVBoxLayout()
        proof_label = QtWidgets.QLabel("Automatic proof")
        proof_label.setProperty("role", "secondary")
        self.cleanup_proof_value = QtWidgets.QLabel("Verified · immutable")
        self.cleanup_proof_value.setProperty("role", "section")
        proof_copy.addWidget(proof_label)
        proof_copy.addWidget(self.cleanup_proof_value)
        proof_layout.addLayout(proof_copy, 1)
        clean_base_copy = QtWidgets.QVBoxLayout()
        clean_base_label = QtWidgets.QLabel("Clean base")
        clean_base_label.setProperty("role", "secondary")
        self.cleanup_revision_readout = self._facade_readout(
            "Current cleaned base revision"
        )
        clean_base_copy.addWidget(clean_base_label)
        clean_base_copy.addWidget(self.cleanup_revision_readout)
        proof_layout.addLayout(clean_base_copy, 1)
        layout.addWidget(proof)

        preview_container = QtWidgets.QWidget()
        previews = QtWidgets.QHBoxLayout(preview_container)
        previews.setContentsMargins(0, 0, 0, 0)
        self.cleanup_artifact_previews: dict[str, QtWidgets.QLabel] = {}
        for preview_id, label in (
            ("original", "Original"),
            ("cleaned", "Current"),
            ("preview", "Preview"),
        ):
            frame = QtWidgets.QFrame()
            frame.setObjectName("cleanupArtifactPreview")
            frame.setProperty("role", "panel-raised")
            frame_layout = QtWidgets.QVBoxLayout(frame)
            frame_layout.setContentsMargins(4, 4, 4, 4)
            image = QtWidgets.QLabel("No artifact")
            image.setObjectName(f"cleanup{preview_id.title()}Preview")
            image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            image.setFixedHeight(62)
            image.setScaledContents(False)
            image.setAccessibleName(f"{label} cleanup artifact preview")
            frame_layout.addWidget(image)
            caption = QtWidgets.QLabel(label)
            caption.setProperty("role", "secondary")
            caption.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            frame_layout.addWidget(caption)
            previews.addWidget(frame, 1)
            self.cleanup_artifact_previews[preview_id] = image
        self.cleanup_artifact_preview_container = preview_container
        self.cleanup_artifact_preview_container.setVisible(False)
        layout.addWidget(self.cleanup_artifact_preview_container)

        self.cleanup_state = QtWidgets.QLabel("No cleaned base selected")
        self.cleanup_state.setObjectName("cleanupStateLabel")
        self.cleanup_state.setProperty("role", "section")
        self.cleanup_state.setAccessibleName("Cleanup stage status")
        self.cleanup_state.setVisible(True)
        layout.addWidget(self.cleanup_state)
        self.cleanup_detail = QtWidgets.QLabel(
            "Automatic proof and manual revision provenance appear here."
        )
        self.cleanup_detail.setWordWrap(True)
        self.cleanup_detail.setProperty("role", "secondary")
        self.cleanup_detail.setAccessibleName("Cleanup provenance and next action")
        self.cleanup_detail.setVisible(True)
        layout.addWidget(self.cleanup_detail)
        actions = QtWidgets.QHBoxLayout()
        self.cleanup_cancel_button = QtWidgets.QPushButton("Cancel draft")
        self.cleanup_cancel_button.setObjectName("cleanupFacadeCancelButton")
        self.cleanup_cancel_button.setProperty("role", "command")
        self.cleanup_cancel_button.setProperty("variant", "quiet")
        self.cleanup_cancel_button.setEnabled(False)
        self.cleanup_cancel_button.setAccessibleName("Cancel cleanup draft")
        self.cleanup_cancel_button.clicked.connect(self._reset_cleanup_facade)
        actions.addWidget(self.cleanup_cancel_button)
        self.cleanup_button = QtWidgets.QPushButton("Preview cleanup")
        self.cleanup_button.setObjectName("openManualCleanupButton")
        self.cleanup_button.setProperty("role", "command")
        self.cleanup_button.setProperty("variant", "secondary")
        self.cleanup_button.setIcon(hybrid_icon("eye", self._icon_theme))
        self.cleanup_button.setAccessibleName("Preview cleanup")
        self.cleanup_button.setAccessibleDescription(
            "Open the page-bounded mask workspace and create a non-publishing preview."
        )
        self.cleanup_button.setToolTip(
            "Open the page-bounded mask workspace and create a non-publishing preview."
        )
        self.cleanup_button.clicked.connect(
            lambda: self.manual_cleanup_requested.emit(self._current_page_id)
            if self._current_page_id
            else None
        )
        actions.addWidget(self.cleanup_button)
        self.cleanup_commit_button = QtWidgets.QPushButton("Commit revision")
        self.cleanup_commit_button.setObjectName("cleanupFacadeCommitButton")
        self.cleanup_commit_button.setProperty("role", "command")
        self.cleanup_commit_button.setProperty("variant", "primary")
        self.cleanup_commit_button.setAccessibleName("Commit cleanup revision")
        self.cleanup_commit_button.setAccessibleDescription(
            "Open the cleanup workspace; a reviewed preview is required before commit."
        )
        self.cleanup_commit_button.clicked.connect(
            lambda: self.manual_cleanup_requested.emit(self._current_page_id)
            if self._current_page_id
            else None
        )
        actions.addWidget(self.cleanup_commit_button)
        layout.addLayout(actions)
        for button in (*self.cleanup_tool_buttons.values(), self.cleanup_button, self.cleanup_commit_button):
            button.setEnabled(False)
        for control in (
            self.cleanup_brush_size,
            self.cleanup_grow,
            self.cleanup_feather,
            self.cleanup_show_mask,
            self.cleanup_show_protected,
        ):
            control.setEnabled(False)
        layout.addStretch(1)
        return scroll

    def _set_cleanup_facade_tool(self, tool_id: str) -> None:
        if tool_id not in self.cleanup_tool_buttons:
            raise ValueError(f"unsupported cleanup facade tool: {tool_id!r}")
        self._cleanup_facade_tool = tool_id
        for candidate, button in self.cleanup_tool_buttons.items():
            blocker = QtCore.QSignalBlocker(button)
            button.setChecked(candidate == tool_id)
            del blocker
        self.refresh_icons(self._icon_theme)
        self._update_cleanup_facade_dirty()

    def _update_cleanup_facade_dirty(self, _value: object = None) -> None:
        dirty = bool(
            self._cleanup_facade_tool != "brush"
            or self.cleanup_brush_size.value() != 24
            or self.cleanup_grow.value() != 2
            or self.cleanup_feather.value() != 1
        )
        self.cleanup_cancel_button.setEnabled(
            bool(dirty and self.cleanup_button.isEnabled())
        )

    def _reset_cleanup_facade(self) -> None:
        self._set_cleanup_facade_tool("brush")
        self.cleanup_brush_size.setValue(24)
        self.cleanup_grow.setValue(2)
        self.cleanup_feather.setValue(1)
        self.cleanup_cancel_button.setEnabled(False)

    def cleanup_facade_defaults(self) -> dict[str, object]:
        """Return typed visual defaults for the real manual-cleanup workspace."""

        return {
            "tool": self._cleanup_facade_tool,
            "brush_radius": self.cleanup_brush_size.value(),
            "grow_px": self.cleanup_grow.value(),
            "feather_px": self.cleanup_feather.value(),
        }

    def _sync_cleanup_overlay_visibility(self, _checked: object = None) -> None:
        """Make the compact cleanup visibility controls operate on the canvas."""

        cleanup_active = bool(
            hasattr(self, "inspector_tabs")
            and self.inspector_tabs.currentIndex()
            == self._inspector_index.get("cleanup")
        )
        controls = (
            ("cleanupMask", getattr(self, "cleanup_show_mask", None)),
            (
                "protectedRegions",
                getattr(self, "cleanup_show_protected", None),
            ),
        )
        for overlay_id, control in controls:
            action = self.overlay_actions.get(overlay_id)
            if action is None or control is None:
                continue
            requested = bool(
                cleanup_active and control.isChecked() and action.isEnabled()
            )
            if action.isChecked() != requested:
                action.setChecked(requested)

    def _refresh_cleanup_visibility_controls(self) -> None:
        """Keep facade visibility controls truthful to projected evidence."""

        controls = (
            (
                "cleanupMask",
                self.cleanup_show_mask,
                "Show the projected cleanup erase mask on the page.",
            ),
            (
                "protectedRegions",
                self.cleanup_show_protected,
                "Show the projected cleanup protected pixels on the page.",
            ),
        )
        for overlay_id, control, available_copy in controls:
            action = self.overlay_actions[overlay_id]
            available = bool(self._cleanup_facade_enabled and action.isEnabled())
            control.setEnabled(available)
            detail = available_copy if available else action.toolTip()
            control.setToolTip(detail)
            control.setStatusTip(detail)
            control.setAccessibleDescription(detail)
        self._sync_cleanup_overlay_visibility()

    def _update_cleanup_artifact_previews(self, artifacts: CanvasArtifactSet) -> None:
        paths = {
            "original": artifacts.original_path,
            "cleaned": artifacts.cleaned_path,
            "preview": None,
        }
        for preview_id, label in self.cleanup_artifact_previews.items():
            path = paths[preview_id]
            pixmap = QtGui.QPixmap(str(path)) if path else QtGui.QPixmap()
            if pixmap.isNull():
                label.setPixmap(QtGui.QPixmap())
                label.setText(
                    "Not created" if preview_id == "preview" else "Unavailable"
                )
                continue
            label.setText("")
            label.setPixmap(
                pixmap.scaled(
                    QtCore.QSize(92, 62),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )

    def _build_history_tab(self) -> QtWidgets.QWidget:
        scroll, layout = self._scroll_tab()
        reset_card = QtWidgets.QFrame()
        reset_card.setObjectName("renderOverrideResetCard")
        reset_card.setProperty("role", "panel-raised")
        reset_layout = QtWidgets.QVBoxLayout(reset_card)
        reset_title = QtWidgets.QLabel("Reset Render Overrides")
        reset_title.setProperty("role", "section")
        reset_title.setWordWrap(True)
        reset_layout.addWidget(reset_title)
        reset_help = QtWidgets.QLabel(
            "Append reversible Restore Automatic records for existing GUI-supported "
            "style or layout overrides. Preview and later owners remain explicit."
        )
        reset_help.setWordWrap(True)
        reset_help.setProperty("role", "secondary")
        reset_layout.addWidget(reset_help)
        reset_form = QtWidgets.QFormLayout()
        self.render_override_reset_scope = WheelSafeComboBox()
        self.render_override_reset_scope.setObjectName("renderOverrideResetScope")
        self.render_override_reset_scope.setAccessibleName("Render override reset scope")
        self.render_override_reset_scope.setAccessibleDescription(
            "Choose selected parent, current page, or entire project."
        )
        for label, value in (
            ("Selected parent", "selected_parent"),
            ("Current page", "current_page"),
            ("Entire project", "entire_project"),
        ):
            self.render_override_reset_scope.addItem(label, value)
        self.render_override_reset_scope.currentIndexChanged.connect(
            lambda index: self.render_override_reset_scope_changed.emit(
                str(self.render_override_reset_scope.itemData(index) or "")
            )
        )
        reset_form.addRow("Scope", self.render_override_reset_scope)
        self.render_override_reset_fields = WheelSafeComboBox()
        self.render_override_reset_fields.setObjectName("renderOverrideResetFields")
        self.render_override_reset_fields.setAccessibleName("Render override field group")
        self.render_override_reset_fields.setAccessibleDescription(
            "Choose Style, Layout, or both supported field groups."
        )
        for label, value in (
            ("Style and Layout", "style_and_layout"),
            ("Style", "style"),
            ("Layout", "layout"),
        ):
            self.render_override_reset_fields.addItem(label, value)
        self.render_override_reset_fields.currentIndexChanged.connect(
            lambda index: self.render_override_reset_field_group_changed.emit(
                str(self.render_override_reset_fields.itemData(index) or "")
            )
        )
        reset_form.addRow("Fields", self.render_override_reset_fields)
        reset_layout.addLayout(reset_form)
        self.render_override_reset_summary = QtWidgets.QLabel(
            "Select a page and parent to inspect resettable overrides."
        )
        self.render_override_reset_summary.setWordWrap(True)
        self.render_override_reset_summary.setProperty("role", "secondary")
        self.render_override_reset_summary.setAccessibleName(
            "Render override reset inventory"
        )
        reset_layout.addWidget(self.render_override_reset_summary)
        self.render_override_reset_status = QtWidgets.QLabel(
            "No render-override reset is active."
        )
        self.render_override_reset_status.setWordWrap(True)
        self.render_override_reset_status.setProperty("role", "secondary")
        self.render_override_reset_status.setProperty("tone", "muted")
        self.render_override_reset_status.setAccessibleName(
            "Render override reset status"
        )
        reset_layout.addWidget(self.render_override_reset_status)
        reset_actions = QtWidgets.QHBoxLayout()
        self.render_override_reset_button = QtWidgets.QPushButton("Reset Overrides")
        self.render_override_reset_button.setObjectName("renderOverrideResetButton")
        self.render_override_reset_button.setProperty("role", "command")
        self.render_override_reset_button.setProperty("variant", "danger")
        self.render_override_reset_button.setAccessibleName("Reset render overrides")
        self.render_override_reset_button.setAccessibleDescription(
            "Append reversible Restore Automatic records for the selected inventory."
        )
        self.render_override_reset_button.clicked.connect(
            self.render_override_reset_requested.emit
        )
        reset_actions.addWidget(self.render_override_reset_button)
        self.render_override_reset_cancel_button = QtWidgets.QPushButton("Cancel")
        self.render_override_reset_cancel_button.setObjectName(
            "renderOverrideResetCancelButton"
        )
        self.render_override_reset_cancel_button.setProperty("role", "command")
        self.render_override_reset_cancel_button.setProperty("variant", "secondary")
        self.render_override_reset_cancel_button.setAccessibleName(
            "Cancel render override reset"
        )
        self.render_override_reset_cancel_button.setAccessibleDescription(
            "Cancel before persistence begins."
        )
        self.render_override_reset_cancel_button.clicked.connect(
            self.render_override_reset_cancel_requested.emit
        )
        reset_actions.addWidget(self.render_override_reset_cancel_button)
        reset_actions.addStretch(1)
        reset_layout.addLayout(reset_actions)
        self.render_override_reset_card = reset_card
        self.render_override_reset_card.setVisible(False)
        history_header = QtWidgets.QHBoxLayout()
        history_title = QtWidgets.QLabel("Page + parent ledgers")
        history_title.setProperty("role", "section")
        history_scope = QtWidgets.QLabel("Undo stack is page-wide")
        history_scope.setProperty("role", "secondary")
        history_scope.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        history_header.addWidget(history_title)
        history_header.addStretch(1)
        history_header.addWidget(history_scope)
        layout.addLayout(history_header)
        guidance = QtWidgets.QLabel(
            "Select a user edit to append a durable Revoke or Reapply record. "
            "Automatic evidence and prior history are never rewritten."
        )
        guidance.setWordWrap(True)
        guidance.setProperty("role", "secondary")
        guidance.setAccessibleName("Edit history guidance")
        self.history_list = QtWidgets.QListWidget()
        self.history_list.setObjectName("editorHistoryList")
        self.history_list.setAccessibleName("Project edit and artifact history")
        self.history_list.setMinimumHeight(190)
        self.history_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.history_list.currentItemChanged.connect(
            self._history_current_item_changed
        )
        layout.addWidget(self.history_list, 1)
        self.history_status = QtWidgets.QLabel("Select a user edit to manage history")
        self.history_status.setWordWrap(True)
        self.history_status.setProperty("role", "secondary")
        self.history_status.setProperty("tone", "muted")
        self.history_status.setAccessibleName("Edit history action status")
        layout.addWidget(self.history_status)
        actions = QtWidgets.QHBoxLayout()
        self.history_revoke_button = QtWidgets.QPushButton(
            "Undo latest page command"
        )
        self.history_revoke_button.setProperty("role", "command")
        self.history_revoke_button.setProperty("variant", "secondary")
        self.history_revoke_button.setIcon(hybrid_icon("undo", self._icon_theme))
        self.history_revoke_button.setAccessibleName("Revoke selected edit")
        self.history_revoke_button.setToolTip(
            "Append a durable revoke record for the selected user edit"
        )
        self.history_revoke_button.clicked.connect(
            self.history_revoke_requested.emit
        )
        actions.addWidget(self.history_revoke_button)
        self.history_reapply_button = QtWidgets.QPushButton("Redo")
        self.history_reapply_button.setProperty("role", "command")
        self.history_reapply_button.setProperty("variant", "secondary")
        self.history_reapply_button.setIcon(hybrid_icon("redo", self._icon_theme))
        self.history_reapply_button.setAccessibleName("Reapply selected edit")
        self.history_reapply_button.setToolTip(
            "Append a durable reapply record for the selected user edit"
        )
        self.history_reapply_button.clicked.connect(
            self.history_reapply_requested.emit
        )
        actions.addWidget(self.history_reapply_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        guidance.setObjectName("historyScopeNotice")
        guidance.setProperty("role", "status-banner")
        guidance.setProperty("tone", "info")
        guidance.setText(
            "Undo follows one chronological page-wide command stack. The durable "
            "ledgers remain scoped, append-only, and visible here."
        )
        layout.addWidget(guidance)
        self.render_override_reset_toggle = QtWidgets.QToolButton()
        self.render_override_reset_toggle.setObjectName(
            "renderOverrideResetToggle"
        )
        self.render_override_reset_toggle.setText("Reset render overrides")
        self.render_override_reset_toggle.setIcon(
            hybrid_icon("more", self._icon_theme)
        )
        self.render_override_reset_toggle.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.render_override_reset_toggle.setCheckable(True)
        self.render_override_reset_toggle.setAccessibleName(
            "Show render override reset controls"
        )
        self.render_override_reset_toggle.toggled.connect(
            self.render_override_reset_card.setVisible
        )
        self.render_override_reset_toggle.setVisible(True)
        layout.addWidget(self.render_override_reset_toggle)
        layout.addWidget(self.render_override_reset_card)
        self.set_history_editor_state(
            selected_record_id="",
            action="",
            action_enabled=False,
            busy=False,
            status_text="Select a user edit to manage history",
        )
        self.set_render_override_reset_state(
            scope="selected_parent",
            field_group="style_and_layout",
            summary_text="Select a page and parent to inspect resettable overrides.",
            reset_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="No render-override reset is active.",
        )
        QtWidgets.QWidget.setTabOrder(
            self.render_override_reset_scope,
            self.render_override_reset_fields,
        )
        QtWidgets.QWidget.setTabOrder(
            self.render_override_reset_fields,
            self.render_override_reset_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.render_override_reset_button,
            self.render_override_reset_cancel_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.render_override_reset_cancel_button,
            self.history_list,
        )
        return scroll

    def set_models(
        self,
        *,
        pages: QtCore.QAbstractItemModel,
        parents: QtCore.QAbstractItemModel,
        page_id_role: int,
        parent_id_role: int,
    ) -> None:
        if not isinstance(pages, QtCore.QAbstractItemModel):
            raise TypeError("pages must be a QAbstractItemModel")
        if not isinstance(parents, QtCore.QAbstractItemModel):
            raise TypeError("parents must be a QAbstractItemModel")
        self.page_list.setModel(pages)
        self.parent_list.setModel(parents)
        self._page_id_role = int(page_id_role)
        self._parent_id_role = int(parent_id_role)
        self._update_page_navigation_buttons()
        self._sync_parent_segment()

    def set_project_name(self, name: str) -> None:
        self.project_name.setText(name)

    def set_current_page(
        self,
        *,
        page_id: str,
        display_name: str,
        ordinal: int,
        total: int,
        artifacts: CanvasArtifactSet,
        overlays: tuple[OverlayShape, ...] = (),
        raster_overlays: tuple[RasterOverlaySource, ...] = (),
        overlay_availability: tuple[OverlayAvailability, ...] = (),
    ) -> None:
        self._current_page_id = page_id
        self.inspector_page.setText(display_name)
        self.page_count.setText(f"Page\n{ordinal} / {total}")
        self.canvas.set_artifacts(artifacts)
        self._update_cleanup_artifact_previews(artifacts)
        if overlay_availability:
            self.set_overlay_availability(overlay_availability)
        self.canvas.set_overlay_shapes(overlays)
        self.canvas.set_raster_overlays(raster_overlays)
        self.rerender_button.setEnabled(bool(page_id))
        self.cleanup_button.setEnabled(bool(page_id))
        if self._page_id_role is not None and self.page_list.model() is not None:
            model = self.page_list.model()
            for row in range(model.rowCount()):
                index = model.index(row, 0)
                if str(index.data(self._page_id_role) or "") == page_id:
                    self.page_list.setCurrentIndex(index)
                    self.page_list.scrollTo(
                        index,
                        QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter,
                    )
                    break
        self._update_page_navigation_buttons()

    def set_preview_state(self, *, enabled: bool, tooltip: str) -> None:
        """Apply the shell-owned page-preview gate without emitting work."""

        preview_enabled = bool(enabled and self._current_page_id)
        for button in (
            self.rerender_button,
            self.style_preview_button,
            self.layout_preview_button,
        ):
            button.setEnabled(preview_enabled)
            button.setToolTip(str(tooltip))
            button.setStatusTip(str(tooltip))
            button.setAccessibleDescription(str(tooltip))

    def clear_current_page(self) -> None:
        """Clear every page/parent-owned view when no page is available."""

        self._current_page_id = ""
        self.inspector_page.setText("No page selected")
        self.page_count.setText("Page — / —")
        self.canvas.clear_page()
        self._update_cleanup_artifact_previews(CanvasArtifactSet(page_id=""))
        self.set_add_user_parent_editor_state(
            draft_role=None,
            draft_workflow_area_bbox=None,
            canvas_size=None,
            editing_enabled=False,
            add_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Open a saved page to add a standalone parent.",
            status_tone="muted",
        )
        unavailable = tuple(
            OverlayAvailability(
                overlay_id,
                False,
                "Open a page with projected overlay evidence.",
            )
            for overlay_id in OVERLAY_IDS
        )
        self.set_overlay_availability(unavailable)
        self.rerender_button.setEnabled(False)
        self.cleanup_button.setEnabled(False)
        self.set_cleanup_summary(
            label="No page selected",
            detail="Open a page to inspect its cleaned base.",
            enabled=False,
        )
        self.set_history(("No page selected",))
        self.set_render_override_reset_state(
            scope="selected_parent",
            field_group="style_and_layout",
            summary_text="Select a page and parent to inspect resettable overrides.",
            reset_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="No render-override reset is active.",
        )
        self.clear_current_parent()
        if self.page_list.model() is not None:
            blocker = QtCore.QSignalBlocker(self.page_list)
            self.page_list.setCurrentIndex(QtCore.QModelIndex())
            del blocker

    def set_overlay_availability(
        self,
        values: tuple[OverlayAvailability, ...],
    ) -> None:
        self.canvas.set_overlay_availability(values)
        by_id = {item.overlay_id: item for item in values}
        for overlay_id in OVERLAY_IDS:
            state = by_id[overlay_id]
            action = self.overlay_actions[overlay_id]
            if not state.available and action.isChecked():
                action.setChecked(False)
            action.setEnabled(state.available)
            action.setToolTip(state.tooltip)
            action.setStatusTip(state.tooltip)
        self._refresh_cleanup_visibility_controls()

    def set_current_parent(
        self,
        *,
        parent_id: str,
        parent_label: str | None = None,
        automatic_text: str,
        user_text: str | None,
        effective_text: str,
        source_text: str,
        target_text: str,
        freshness: str,
        excluded: bool = False,
        style_values: Mapping[str, object] | None = None,
        layout_values: Mapping[str, object] | None = None,
        layout_diagnostic: str = "No layout diagnostics",
        automatic_source_text: str | None = None,
        user_source_text: str | None = None,
        automatic_writing_mode: str | None = None,
        user_writing_mode: str | None = None,
        effective_writing_mode: str | None = None,
        writing_mode_authority: str = "automatic",
        automatic_line_height: float | None = None,
        user_line_height: float | None = None,
        effective_line_height: float | None = None,
        line_height_authority: str = "automatic",
        automatic_rotation: float | None = None,
        user_rotation: float | None = None,
        effective_rotation: float | None = None,
        rotation_authority: str = "automatic",
        automatic_render_box: tuple[int, int, int, int] | None = None,
        automatic_render_hard_bounds: tuple[int, int, int, int] | None = None,
        user_render_box: tuple[int, int, int, int] | None = None,
        effective_render_box: tuple[int, int, int, int] | None = None,
        render_box_authority: str = "automatic",
        automatic_font_role: str | None = None,
        user_font_role: str | None = None,
        effective_font_role: str | None = None,
        font_role_authority: str = "automatic",
        automatic_font_weight_tier: str | None = None,
        user_font_weight_tier: str | None = None,
        effective_font_weight_tier: str | None = None,
        font_weight_tier_authority: str = "automatic",
        automatic_fill_color: str | None = None,
        user_fill_color: str | None = None,
        unresolved_user_fill_color: str | None = None,
        effective_fill_color: str | None = None,
        fill_color_authority: str = "automatic",
        automatic_outline_color: str | None = None,
        user_outline_color: str | None = None,
        unresolved_user_outline_color: str | None = None,
        effective_outline_color: str | None = None,
        outline_color_authority: str = "automatic",
        automatic_outline_width: float | None = None,
        user_outline_width: float | None = None,
        effective_outline_width: float | None = None,
        outline_width_authority: str = "automatic",
        automatic_preferred_size: float | None = None,
        user_preferred_size: float | None = None,
        effective_preferred_size: float | None = None,
        preferred_size_authority: str = "automatic",
        automatic_shadow_color: str | None = None,
        user_shadow_color: str | None = None,
        effective_shadow_color: str | None = None,
        shadow_color_authority: str = "automatic",
        automatic_shadow_blur: float | None = None,
        user_shadow_blur: float | None = None,
        effective_shadow_blur: float | None = None,
        shadow_blur_authority: str = "automatic",
        automatic_shadow_offset: tuple[float, float] | None = None,
        user_shadow_offset: tuple[float, float] | None = None,
        effective_shadow_offset: tuple[float, float] | None = None,
        shadow_offset_authority: str = "automatic",
        automatic_shadow_enabled: bool | None = None,
        user_shadow_enabled: bool | None = None,
        effective_shadow_enabled: bool | None = None,
        shadow_enabled_authority: str = "automatic",
        render_required: bool = False,
        writing_mode_unavailable_reason: str | None = None,
        line_height_unavailable_reason: str | None = None,
        rotation_unavailable_reason: str | None = None,
        render_box_unavailable_reason: str | None = None,
        font_role_unavailable_reason: str | None = None,
        font_weight_tier_unavailable_reason: str | None = None,
        fill_color_unavailable_reason: str | None = None,
        outline_color_unavailable_reason: str | None = None,
        outline_width_unavailable_reason: str | None = None,
        preferred_size_unavailable_reason: str | None = None,
        shadow_color_unavailable_reason: str | None = None,
        shadow_blur_unavailable_reason: str | None = None,
        shadow_offset_unavailable_reason: str | None = None,
        shadow_visibility_unavailable_reason: str | None = None,
    ) -> None:
        parent_changed = self._current_parent_id != parent_id
        self._current_parent_id = parent_id
        if parent_changed:
            self._add_user_parent_panel_requested = False
        visible_parent = str(parent_label or "").strip()
        self.inspector_parent.setText(
            visible_parent or parent_id or "No parent selected"
        )
        self.automatic_text.setText(self._target_authority_text(automatic_text))
        self.user_text.setText(
            "No edit" if user_text is None else self._target_authority_text(user_text)
        )
        self.effective_text.setText(self._target_authority_text(effective_text))
        automatic_source = (
            source_text if automatic_source_text is None else automatic_source_text
        )
        self.set_ocr_revision_state(
            automatic_source_text=automatic_source,
            model_revision_id=None,
            model_revision_text=None,
            model_revision_engine=None,
            user_source_text=user_source_text,
            effective_source_text=source_text,
            effective_source_authority=(
                "user" if user_source_text is not None else "automatic"
            ),
            rerun_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text=(
                "Explicit Rerun OCR is currently available only for a pending "
                "user parent."
            ),
            status_tone="muted",
        )
        self.set_translation_revision_state(
            automatic_target_text=automatic_text,
            model_revision_id=None,
            model_revision_text=None,
            model_provider=None,
            model_id=None,
            user_target_text=user_text,
            effective_target_text=effective_text,
            effective_target_authority=(
                "user" if user_text is not None else "automatic"
            ),
            rerun_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text=(
                "Explicit translation revision is available only for a "
                "source-current user parent."
            ),
            status_tone="muted",
        )
        self.set_source_text_editor_state(
            draft_text=source_text,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            status_text="Preparing the selected-parent source edit state…",
            status_tone="muted",
        )
        self.set_target_text_editor_state(
            draft_text=target_text,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            status_text="Preparing the selected-parent edit state…",
            status_tone="muted",
        )
        self.set_parent_membership_state(
            excluded=bool(excluded),
            enabled=False,
            busy=False,
            status_text="Preparing the selected-parent membership…",
            status_tone="muted",
        )
        self.set_parent_geometry_state(
            automatic_bbox=None,
            effective_bbox=None,
            draft_bbox=None,
            canvas_size=None,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Preparing the selected-parent geometry…",
            status_tone="muted",
        )
        self.set_reading_order_editor_state(
            automatic_order=(),
            effective_order=(),
            proposed_order=(),
            selected_parent_id="",
            excluded_parent_ids=(),
            move_earlier_enabled=False,
            move_later_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Preparing the page reading-order edit state…",
            status_tone="muted",
        )
        self.set_merge_parent_editor_state(
            candidates=(),
            selected_partner_id="",
            source_parent_ids=None,
            source_bboxes=None,
            merged_bbox=None,
            merged_source_text="",
            editing_enabled=False,
            merge_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Preparing the Merge Parent pipeline selection…",
            status_tone="muted",
        )
        self.set_split_parent_editor_state(
            source_bbox=None,
            child_bboxes=None,
            orientation=None,
            split_offset=None,
            editing_enabled=False,
            split_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Preparing the Split Parent topology state…",
            status_tone="muted",
        )
        self.set_writing_mode_editor_state(
            automatic_writing_mode=automatic_writing_mode,
            user_writing_mode=user_writing_mode,
            effective_writing_mode=effective_writing_mode,
            draft_writing_mode=effective_writing_mode,
            writing_mode_authority=writing_mode_authority,
            render_required=render_required,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=(
                writing_mode_unavailable_reason
                or "Preparing the selected-parent writing-mode edit state…"
            ),
            status_tone=(
                "warning" if writing_mode_unavailable_reason else "muted"
            ),
        )
        self.set_line_height_editor_state(
            automatic_line_height=automatic_line_height,
            user_line_height=user_line_height,
            effective_line_height=effective_line_height,
            draft_line_height=effective_line_height,
            line_height_authority=line_height_authority,
            render_required=render_required,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=(
                line_height_unavailable_reason
                or "Preparing the selected-parent line-height edit state…"
            ),
            status_tone=(
                "warning" if line_height_unavailable_reason else "muted"
            ),
        )
        self.set_rotation_editor_state(
            automatic_rotation=automatic_rotation,
            user_rotation=user_rotation,
            effective_rotation=effective_rotation,
            draft_rotation=effective_rotation,
            rotation_authority=rotation_authority,
            render_required=render_required,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=(
                rotation_unavailable_reason
                or "Preparing the selected-parent rotation edit state…"
            ),
            status_tone=(
                "warning" if rotation_unavailable_reason else "muted"
            ),
        )
        self.set_render_box_editor_state(
            automatic_render_box=automatic_render_box,
            automatic_hard_bounds=automatic_render_hard_bounds,
            user_render_box=user_render_box,
            effective_render_box=effective_render_box,
            draft_render_box=effective_render_box,
            render_box_authority=render_box_authority,
            render_required=render_required,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=(
                render_box_unavailable_reason
                or "Preparing the selected-parent render-box edit state…"
            ),
            status_tone=(
                "warning" if render_box_unavailable_reason else "muted"
            ),
        )
        self.set_font_role_editor_state(
            automatic_font_role=automatic_font_role,
            user_font_role=user_font_role,
            effective_font_role=effective_font_role,
            draft_font_role=effective_font_role,
            font_role_authority=font_role_authority,
            render_required=render_required,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=(
                font_role_unavailable_reason
                or "Preparing the selected-parent font-role edit state…"
            ),
            status_tone=(
                "warning" if font_role_unavailable_reason else "muted"
            ),
        )
        self.set_font_weight_tier_editor_state(
            automatic_font_weight_tier=automatic_font_weight_tier,
            user_font_weight_tier=user_font_weight_tier,
            effective_font_weight_tier=effective_font_weight_tier,
            draft_font_weight_tier=effective_font_weight_tier,
            font_weight_tier_authority=font_weight_tier_authority,
            render_required=render_required,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=(
                font_weight_tier_unavailable_reason
                or "Preparing the selected-parent font-weight edit state…"
            ),
            status_tone=(
                "warning" if font_weight_tier_unavailable_reason else "muted"
            ),
        )
        self.set_fill_color_editor_state(
            automatic_fill_color=automatic_fill_color,
            user_fill_color=user_fill_color,
            unresolved_user_fill_color=unresolved_user_fill_color,
            effective_fill_color=effective_fill_color,
            draft_fill_color=effective_fill_color,
            fill_color_authority=fill_color_authority,
            render_required=render_required,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=(
                fill_color_unavailable_reason
                or "Preparing the selected-parent fill-color edit state…"
            ),
            status_tone=(
                "warning" if fill_color_unavailable_reason else "muted"
            ),
        )
        self.set_outline_color_editor_state(
            automatic_outline_color=automatic_outline_color,
            user_outline_color=user_outline_color,
            unresolved_user_outline_color=unresolved_user_outline_color,
            effective_outline_color=effective_outline_color,
            draft_outline_color=effective_outline_color,
            outline_color_authority=outline_color_authority,
            render_required=render_required,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=(
                outline_color_unavailable_reason
                or "Preparing the selected-parent outline-color edit state…"
            ),
            status_tone=(
                "warning" if outline_color_unavailable_reason else "muted"
            ),
        )
        self.set_outline_width_editor_state(
            automatic_outline_width=automatic_outline_width,
            user_outline_width=user_outline_width,
            effective_outline_width=effective_outline_width,
            draft_outline_width=effective_outline_width,
            outline_width_authority=outline_width_authority,
            render_required=render_required,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=(
                outline_width_unavailable_reason
                or "Preparing the selected-parent outline-width edit state…"
            ),
            status_tone=(
                "warning" if outline_width_unavailable_reason else "muted"
            ),
        )
        self.set_preferred_size_editor_state(
            automatic_preferred_size=automatic_preferred_size,
            user_preferred_size=user_preferred_size,
            effective_preferred_size=effective_preferred_size,
            draft_preferred_size=effective_preferred_size,
            preferred_size_authority=preferred_size_authority,
            render_required=render_required,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=(
                preferred_size_unavailable_reason
                or "Preparing the selected-parent preferred-size edit state…"
            ),
            status_tone=(
                "warning" if preferred_size_unavailable_reason else "muted"
            ),
        )
        self.set_shadow_color_editor_state(
            automatic_shadow_color=automatic_shadow_color,
            user_shadow_color=user_shadow_color,
            effective_shadow_color=effective_shadow_color,
            draft_shadow_color=effective_shadow_color,
            shadow_color_authority=shadow_color_authority,
            render_required=render_required,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=(
                shadow_color_unavailable_reason
                or "Preparing the selected-parent shadow-color edit state…"
            ),
            status_tone=(
                "warning" if shadow_color_unavailable_reason else "muted"
            ),
        )
        self.set_shadow_blur_editor_state(
            automatic_shadow_blur=automatic_shadow_blur,
            user_shadow_blur=user_shadow_blur,
            effective_shadow_blur=effective_shadow_blur,
            draft_shadow_blur=effective_shadow_blur,
            shadow_blur_authority=shadow_blur_authority,
            render_required=render_required,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=(
                shadow_blur_unavailable_reason
                or "Preparing the selected-parent shadow-blur edit state…"
            ),
            status_tone=(
                "warning" if shadow_blur_unavailable_reason else "muted"
            ),
        )
        self.set_shadow_offset_editor_state(
            automatic_shadow_offset=automatic_shadow_offset,
            user_shadow_offset=user_shadow_offset,
            effective_shadow_offset=effective_shadow_offset,
            draft_shadow_offset=effective_shadow_offset,
            shadow_offset_authority=shadow_offset_authority,
            render_required=render_required,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=(
                shadow_offset_unavailable_reason
                or "Preparing the selected-parent shadow-offset edit state…"
            ),
            status_tone=(
                "warning" if shadow_offset_unavailable_reason else "muted"
            ),
        )
        self.set_shadow_visibility_editor_state(
            automatic_shadow_enabled=automatic_shadow_enabled,
            user_shadow_enabled=user_shadow_enabled,
            effective_shadow_enabled=effective_shadow_enabled,
            draft_shadow_enabled=effective_shadow_enabled,
            shadow_enabled_authority=shadow_enabled_authority,
            render_required=render_required,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=(
                shadow_visibility_unavailable_reason
                or "Preparing the selected-parent shadow-visibility edit state…"
            ),
            status_tone=(
                "warning" if shadow_visibility_unavailable_reason else "muted"
            ),
        )
        self.text_freshness.setText(f"Freshness: {freshness}")
        stable_style_values = style_values or {}
        stable_layout_values = layout_values or {}
        self._replace_form(self.style_form, stable_style_values)
        self._replace_form(self.layout_form, stable_layout_values)
        self.layout_diagnostic.setText(layout_diagnostic)
        self._update_style_facade(
            style_values=stable_style_values,
            effective_font_role=effective_font_role,
            effective_font_weight_tier=effective_font_weight_tier,
            effective_preferred_size=effective_preferred_size,
            effective_fill_color=effective_fill_color,
            effective_outline_color=effective_outline_color,
        )
        self._update_layout_facade(
            layout_values=stable_layout_values,
            effective_render_box=effective_render_box,
            effective_rotation=effective_rotation,
            effective_line_height=effective_line_height,
            effective_writing_mode=effective_writing_mode,
        )
        if self._parent_id_role is not None and self.parent_list.model() is not None:
            model = self.parent_list.model()
            for row in range(model.rowCount()):
                index = model.index(row, 0)
                if str(index.data(self._parent_id_role) or "") == parent_id:
                    blocker = QtCore.QSignalBlocker(self.parent_list)
                    self.parent_list.setCurrentIndex(row)
                    del blocker
                    break
        self._sync_parent_segment()

    def set_current_user_parent(
        self,
        *,
        parent_id: str,
        role: str,
        workflow_area_bbox: tuple[int, int, int, int],
        stage_summary: str,
    ) -> None:
        """Show one pending standalone parent without fabricating automatic facts."""

        stable_parent_id = str(parent_id or "").strip()
        if not stable_parent_id:
            raise ValueError("parent_id is required")
        if role not in {"speech", "caption"}:
            raise ValueError("role must be speech or caption")
        if (
            not isinstance(workflow_area_bbox, tuple)
            or len(workflow_area_bbox) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in workflow_area_bbox
            )
            or workflow_area_bbox[0] < 0
            or workflow_area_bbox[1] < 0
            or workflow_area_bbox[2] <= 0
            or workflow_area_bbox[3] <= 0
        ):
            raise ValueError("workflow_area_bbox must be a valid integer x/y/w/h bbox")
        stable_stage_summary = str(stage_summary or "").strip()
        if not stable_stage_summary:
            raise ValueError("stage_summary is required")

        role_label = "Dialogue" if role == "speech" else "Caption"
        parent_changed = self._current_parent_id != stable_parent_id
        self._current_parent_id = stable_parent_id
        if parent_changed:
            self._add_user_parent_panel_requested = False
        self.inspector_parent.setText(f"{role_label} · User parent")
        self.automatic_text.setText("Unavailable — no automatic parent evidence")
        self.user_text.setText("No target revision")
        self.effective_text.setText("Pending required revisions")
        self.set_ocr_revision_state(
            automatic_source_text=None,
            model_revision_id=None,
            model_revision_text=None,
            model_revision_engine=None,
            user_source_text=None,
            effective_source_text=None,
            effective_source_authority="unavailable",
            rerun_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Preparing the selected-parent OCR revision state...",
            status_tone="muted",
        )
        self.set_translation_revision_state(
            automatic_target_text=None,
            model_revision_id=None,
            model_revision_text=None,
            model_provider=None,
            model_id=None,
            user_target_text=None,
            effective_target_text=None,
            effective_target_authority="unavailable",
            rerun_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Preparing the selected-parent translation revision state...",
            status_tone="muted",
        )
        self.set_source_text_editor_state(
            draft_text="",
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            status_text=(
                "Source is required as an explicit revision for this user parent."
            ),
            status_tone="warning",
            history_identity=f"{stable_parent_id}:user-source-unavailable",
        )
        self.set_target_text_editor_state(
            draft_text="",
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            status_text=(
                "Target is unavailable until the required source and translation "
                "revisions are complete."
            ),
            status_tone="warning",
            history_identity=f"{stable_parent_id}:user-target-unavailable",
            restore_selected_model_translation=True,
        )
        self.set_parent_membership_state(
            excluded=False,
            enabled=False,
            busy=False,
            status_text=(
                "This standalone user parent can be removed through History."
            ),
            status_tone="muted",
        )
        self.set_parent_geometry_state(
            automatic_bbox=None,
            effective_bbox=None,
            draft_bbox=None,
            canvas_size=None,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text=(
                "The saved workflow area is page intent, not automatic or render geometry."
            ),
            status_tone="muted",
        )
        self.set_reading_order_editor_state(
            automatic_order=(),
            effective_order=(),
            proposed_order=(),
            selected_parent_id="",
            excluded_parent_ids=(),
            move_earlier_enabled=False,
            move_later_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Preparing the page reading-order edit state…",
            status_tone="muted",
        )
        self.set_merge_parent_editor_state(
            candidates=(),
            selected_partner_id="",
            source_parent_ids=None,
            source_bboxes=None,
            merged_bbox=None,
            merged_source_text="",
            editing_enabled=False,
            merge_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Merge Parent requires adjacent pipeline parents.",
            status_tone="muted",
        )
        self.set_split_parent_editor_state(
            source_bbox=workflow_area_bbox,
            child_bboxes=None,
            orientation=None,
            split_offset=None,
            editing_enabled=False,
            split_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Preparing the Split Parent topology state…",
            status_tone="muted",
        )
        unavailable_copy = (
            "Unavailable for a user parent until its explicit forward revision is complete."
        )
        self.set_writing_mode_editor_state(
            automatic_writing_mode=None,
            user_writing_mode=None,
            effective_writing_mode=None,
            draft_writing_mode=None,
            writing_mode_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=unavailable_copy,
            status_tone="warning",
        )
        self.set_line_height_editor_state(
            automatic_line_height=None,
            user_line_height=None,
            effective_line_height=None,
            draft_line_height=None,
            line_height_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=unavailable_copy,
            status_tone="warning",
        )
        self.set_rotation_editor_state(
            automatic_rotation=None,
            user_rotation=None,
            effective_rotation=None,
            draft_rotation=None,
            rotation_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=unavailable_copy,
            status_tone="warning",
        )
        self.set_render_box_editor_state(
            automatic_render_box=None,
            automatic_hard_bounds=None,
            user_render_box=None,
            effective_render_box=None,
            draft_render_box=None,
            render_box_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=unavailable_copy,
            status_tone="warning",
        )
        self.set_font_role_editor_state(
            automatic_font_role=None,
            user_font_role=None,
            effective_font_role=None,
            draft_font_role=None,
            font_role_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=unavailable_copy,
            status_tone="warning",
        )
        self.set_font_weight_tier_editor_state(
            automatic_font_weight_tier=None,
            user_font_weight_tier=None,
            effective_font_weight_tier=None,
            draft_font_weight_tier=None,
            font_weight_tier_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=unavailable_copy,
            status_tone="warning",
        )
        self.set_fill_color_editor_state(
            automatic_fill_color=None,
            user_fill_color=None,
            unresolved_user_fill_color=None,
            effective_fill_color=None,
            draft_fill_color=None,
            fill_color_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=unavailable_copy,
            status_tone="warning",
        )
        self.set_outline_color_editor_state(
            automatic_outline_color=None,
            user_outline_color=None,
            unresolved_user_outline_color=None,
            effective_outline_color=None,
            draft_outline_color=None,
            outline_color_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=unavailable_copy,
            status_tone="warning",
        )
        self.set_outline_width_editor_state(
            automatic_outline_width=None,
            user_outline_width=None,
            effective_outline_width=None,
            draft_outline_width=None,
            outline_width_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=unavailable_copy,
            status_tone="warning",
        )
        self.set_preferred_size_editor_state(
            automatic_preferred_size=None,
            user_preferred_size=None,
            effective_preferred_size=None,
            draft_preferred_size=None,
            preferred_size_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=unavailable_copy,
            status_tone="warning",
        )
        self.set_shadow_color_editor_state(
            automatic_shadow_color=None,
            user_shadow_color=None,
            effective_shadow_color=None,
            draft_shadow_color=None,
            shadow_color_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=unavailable_copy,
            status_tone="warning",
        )
        self.set_shadow_blur_editor_state(
            automatic_shadow_blur=None,
            user_shadow_blur=None,
            effective_shadow_blur=None,
            draft_shadow_blur=None,
            shadow_blur_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=unavailable_copy,
            status_tone="warning",
        )
        self.set_shadow_offset_editor_state(
            automatic_shadow_offset=None,
            user_shadow_offset=None,
            effective_shadow_offset=None,
            draft_shadow_offset=None,
            shadow_offset_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=unavailable_copy,
            status_tone="warning",
        )
        self.set_shadow_visibility_editor_state(
            automatic_shadow_enabled=None,
            user_shadow_enabled=None,
            effective_shadow_enabled=None,
            draft_shadow_enabled=None,
            shadow_enabled_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text=unavailable_copy,
            status_tone="warning",
        )
        x, y, width, height = workflow_area_bbox
        self.text_freshness.setText("Freshness: Unavailable — source required")
        self._replace_form(self.style_form, {})
        self._replace_form(self.layout_form, {})
        self.layout_diagnostic.setText(
            f"Workflow area: x {x}, y {y}, width {width}, height {height}. "
            + stable_stage_summary
        )
        self._update_style_facade(
            style_values={},
            effective_font_role=None,
            effective_font_weight_tier=None,
            effective_preferred_size=None,
            effective_fill_color=None,
            effective_outline_color=None,
        )
        self._update_layout_facade(
            layout_values={},
            effective_render_box=None,
            effective_rotation=None,
            effective_line_height=None,
            effective_writing_mode=None,
        )
        if self._parent_id_role is not None and self.parent_list.model() is not None:
            model = self.parent_list.model()
            for row in range(model.rowCount()):
                index = model.index(row, 0)
                if str(index.data(self._parent_id_role) or "") == stable_parent_id:
                    blocker = QtCore.QSignalBlocker(self.parent_list)
                    self.parent_list.setCurrentIndex(row)
                    del blocker
                    break
        self._sync_parent_segment()

    @staticmethod
    def _target_authority_text(value: str) -> str:
        """Render an exact empty target distinctly from an absent edit."""

        if not isinstance(value, str):
            raise TypeError("target authority text must be a string")
        return value if value else "(empty text)"

    @staticmethod
    def _public_provider_label(value: str) -> str:
        """Reduce provider provenance to a safe public product label."""

        if not isinstance(value, str):
            raise TypeError("provider label must be a string")
        label = value.strip().split("(", 1)[0].strip()
        if not label or "://" in label or "/" in label or "\\" in label:
            return "Configured provider"
        return {
            "deepseek": "DeepSeek",
            "gguf": "GGUF",
            "mangaocr": "MangaOCR",
            "ollama": "Ollama",
        }.get(label.casefold(), label)

    @staticmethod
    def _public_model_label(value: str) -> str:
        """Show a public model name without a local path or endpoint."""

        if not isinstance(value, str):
            raise TypeError("model label must be a string")
        model = value.strip()
        if not model or "://" in model:
            return "Configured model"
        leaf = model.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        return leaf or "Configured model"

    def clear_current_parent(self) -> None:
        """Clear every parent-owned field when the current page has no parent."""

        self._current_parent_id = ""
        self.inspector_parent.setText("No parent selected")
        self.automatic_text.setText("—")
        self.user_text.setText("No edit")
        self.effective_text.setText("—")
        self.set_ocr_revision_state(
            automatic_source_text=None,
            model_revision_id=None,
            model_revision_text=None,
            model_revision_engine=None,
            user_source_text=None,
            effective_source_text=None,
            effective_source_authority="unavailable",
            rerun_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Select a pending user parent to run OCR",
            status_tone="muted",
        )
        self.set_translation_revision_state(
            automatic_target_text=None,
            model_revision_id=None,
            model_revision_text=None,
            model_provider=None,
            model_id=None,
            user_target_text=None,
            effective_target_text=None,
            effective_target_authority="unavailable",
            rerun_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Select a source-current user parent to run translation",
            status_tone="muted",
        )
        self.set_source_text_editor_state(
            draft_text="",
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            status_text="Select a parent to edit source text",
            status_tone="muted",
            history_identity="",
        )
        self.set_target_text_editor_state(
            draft_text="",
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            status_text="Select a parent to edit target text",
            status_tone="muted",
            history_identity="",
        )
        self.set_parent_membership_state(
            excluded=False,
            enabled=False,
            busy=False,
            status_text="Select a parent to manage membership",
            status_tone="muted",
        )
        self.set_parent_geometry_state(
            automatic_bbox=None,
            effective_bbox=None,
            draft_bbox=None,
            canvas_size=None,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Select a parent to edit structural geometry",
            status_tone="muted",
        )
        self.set_reading_order_editor_state(
            automatic_order=(),
            effective_order=(),
            proposed_order=(),
            selected_parent_id="",
            excluded_parent_ids=(),
            move_earlier_enabled=False,
            move_later_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Select an included parent to edit page reading order",
            status_tone="muted",
        )
        self.set_merge_parent_editor_state(
            candidates=(),
            selected_partner_id="",
            source_parent_ids=None,
            source_bboxes=None,
            merged_bbox=None,
            merged_source_text="",
            editing_enabled=False,
            merge_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Select an eligible pipeline parent to prepare a merge",
            status_tone="muted",
        )
        self.set_split_parent_editor_state(
            source_bbox=None,
            child_bboxes=None,
            orientation=None,
            split_offset=None,
            editing_enabled=False,
            split_enabled=False,
            cancel_enabled=False,
            busy=False,
            status_text="Select a standalone Add-created user parent to split",
            status_tone="muted",
        )
        self.set_writing_mode_editor_state(
            automatic_writing_mode=None,
            user_writing_mode=None,
            effective_writing_mode=None,
            draft_writing_mode=None,
            writing_mode_authority="automatic",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text="Select a render-required parent to edit writing mode",
            status_tone="muted",
        )
        self.set_line_height_editor_state(
            automatic_line_height=None,
            user_line_height=None,
            effective_line_height=None,
            draft_line_height=None,
            line_height_authority="automatic",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text="Select a render-required parent to edit line height",
            status_tone="muted",
        )
        self.set_rotation_editor_state(
            automatic_rotation=None,
            user_rotation=None,
            effective_rotation=None,
            draft_rotation=None,
            rotation_authority="automatic",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text="Select a render-required parent to edit rotation",
            status_tone="muted",
        )
        self.set_render_box_editor_state(
            automatic_render_box=None,
            automatic_hard_bounds=None,
            user_render_box=None,
            effective_render_box=None,
            draft_render_box=None,
            render_box_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text="Select a render-required parent to edit the render box",
            status_tone="muted",
        )
        self.set_font_role_editor_state(
            automatic_font_role=None,
            user_font_role=None,
            effective_font_role=None,
            draft_font_role=None,
            font_role_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text="Select a render-required parent to edit font role",
            status_tone="muted",
        )
        self.set_font_weight_tier_editor_state(
            automatic_font_weight_tier=None,
            user_font_weight_tier=None,
            effective_font_weight_tier=None,
            draft_font_weight_tier=None,
            font_weight_tier_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text="Select a render-required parent to edit font weight",
            status_tone="muted",
        )
        self.set_fill_color_editor_state(
            automatic_fill_color=None,
            user_fill_color=None,
            unresolved_user_fill_color=None,
            effective_fill_color=None,
            draft_fill_color=None,
            fill_color_authority="automatic",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text="Select a render-required parent to edit fill color",
            status_tone="muted",
        )
        self.set_outline_color_editor_state(
            automatic_outline_color=None,
            user_outline_color=None,
            unresolved_user_outline_color=None,
            effective_outline_color=None,
            draft_outline_color=None,
            outline_color_authority="automatic",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text="Select a render-required parent to edit outline color",
            status_tone="muted",
        )
        self.set_outline_width_editor_state(
            automatic_outline_width=None,
            user_outline_width=None,
            effective_outline_width=None,
            draft_outline_width=None,
            outline_width_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text="Select a render-required parent to edit outline width",
            status_tone="muted",
        )
        self.set_preferred_size_editor_state(
            automatic_preferred_size=None,
            user_preferred_size=None,
            effective_preferred_size=None,
            draft_preferred_size=None,
            preferred_size_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text="Select a render-required parent to edit preferred size",
            status_tone="muted",
        )
        self.set_shadow_color_editor_state(
            automatic_shadow_color=None,
            user_shadow_color=None,
            effective_shadow_color=None,
            draft_shadow_color=None,
            shadow_color_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text="Select a render-required parent to edit shadow color",
            status_tone="muted",
        )
        self.set_shadow_blur_editor_state(
            automatic_shadow_blur=None,
            user_shadow_blur=None,
            effective_shadow_blur=None,
            draft_shadow_blur=None,
            shadow_blur_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text="Select a render-required parent to edit shadow blur",
            status_tone="muted",
        )
        self.set_shadow_offset_editor_state(
            automatic_shadow_offset=None,
            user_shadow_offset=None,
            effective_shadow_offset=None,
            draft_shadow_offset=None,
            shadow_offset_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text="Select a render-required parent to edit shadow offset",
            status_tone="muted",
        )
        self.set_shadow_visibility_editor_state(
            automatic_shadow_enabled=None,
            user_shadow_enabled=None,
            effective_shadow_enabled=None,
            draft_shadow_enabled=None,
            shadow_enabled_authority="unavailable",
            render_required=False,
            editing_enabled=False,
            apply_enabled=False,
            cancel_enabled=False,
            restore_enabled=False,
            busy=False,
            status_text="Select a render-required parent to edit shadow visibility",
            status_tone="muted",
        )
        self.text_freshness.setText("Freshness: —")
        self._replace_form(self.style_form, {})
        self._replace_form(self.layout_form, {})
        self.layout_diagnostic.setText("No layout diagnostics")
        self._update_style_facade(
            style_values={},
            effective_font_role=None,
            effective_font_weight_tier=None,
            effective_preferred_size=None,
            effective_fill_color=None,
            effective_outline_color=None,
        )
        self._update_layout_facade(
            layout_values={},
            effective_render_box=None,
            effective_rotation=None,
            effective_line_height=None,
            effective_writing_mode=None,
        )
        blocker = QtCore.QSignalBlocker(self.parent_list)
        self.parent_list.setCurrentIndex(-1)
        del blocker
        self._sync_parent_segment()

    def set_source_text_editor_state(
        self,
        *,
        draft_text: str,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        status_text: str,
        status_tone: str = "muted",
        history_identity: str | None = None,
        restore_label: str = "Restore Automatic",
    ) -> None:
        """Apply one immutable source-edit view-model state to the inspector."""

        if not isinstance(draft_text, str):
            raise TypeError("draft_text must be a string")
        if history_identity is not None:
            self.source_text.bind_history_identity(history_identity)
        if self.source_text.exact_text() != draft_text:
            blocker = QtCore.QSignalBlocker(self.source_text)
            self.source_text.set_exact_text(draft_text)
            del blocker
        self.source_text.setEnabled(bool(editing_enabled))
        self.source_apply_button.setEnabled(bool(apply_enabled))
        self.source_apply_button.setVisible(bool(apply_enabled))
        self.source_cancel_button.setEnabled(bool(cancel_enabled))
        self.source_cancel_button.setVisible(bool(cancel_enabled))
        stable_restore_label = str(restore_label or "").strip()
        if not stable_restore_label:
            raise ValueError("restore_label is required")
        self.source_restore_button.setText(stable_restore_label)
        self.source_restore_button.setAccessibleName(stable_restore_label)
        self.source_restore_button.setToolTip(
            (
                "Restore the immutable selected model OCR revision."
                if stable_restore_label == "Restore Selected Model OCR"
                else "Remove the user source override and restore automatic OCR."
            )
        )
        self.source_restore_button.setEnabled(bool(restore_enabled))
        self.source_restore_button.setVisible(bool(restore_enabled))
        self.source_edit_status.setText(str(status_text))
        self.source_edit_status.setProperty("tone", str(status_tone))
        self.source_edit_status.setVisible(
            str(status_tone) in {"warning", "error", "info"}
        )
        self.source_edit_status.style().unpolish(self.source_edit_status)
        self.source_edit_status.style().polish(self.source_edit_status)

    def set_ocr_revision_state(
        self,
        *,
        automatic_source_text: str | None,
        model_revision_id: str | None,
        model_revision_text: str | None,
        model_revision_engine: str | None,
        user_source_text: str | None,
        effective_source_text: str | None,
        effective_source_authority: str,
        rerun_enabled: bool,
        cancel_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Render four source provenance lanes without collapsing authority."""

        for field_name, value in (
            ("automatic_source_text", automatic_source_text),
            ("model_revision_id", model_revision_id),
            ("model_revision_text", model_revision_text),
            ("model_revision_engine", model_revision_engine),
            ("user_source_text", user_source_text),
            ("effective_source_text", effective_source_text),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
        if (model_revision_id is None) != (model_revision_text is None):
            raise ValueError(
                "selected model revision identity and text must travel together"
            )
        if model_revision_id is None and model_revision_engine is not None:
            raise ValueError("model revision engine requires a selected revision")
        if model_revision_id is not None and model_revision_engine is None:
            raise ValueError("selected model revision requires engine provenance")
        if effective_source_authority not in {
            "automatic",
            "model",
            "user",
            "unavailable",
        }:
            raise ValueError("unsupported effective source authority")

        automatic_copy = (
            "Unavailable - user-owned topology has no Automatic source"
            if automatic_source_text is None
            else self._target_authority_text(automatic_source_text)
        )
        if model_revision_text is None:
            model_copy = "Not selected"
        else:
            model_copy = (
                f"{self._public_provider_label(model_revision_engine)}: "
                + self._target_authority_text(model_revision_text)
            )
        user_copy = (
            "No edit"
            if user_source_text is None
            else self._target_authority_text(user_source_text)
        )
        effective_copy = (
            "Unavailable"
            if effective_source_text is None
            else self._target_authority_text(effective_source_text)
        )
        authority_label = {
            "automatic": "Automatic source",
            "model": "Selected model OCR revision",
            "user": "Your edit",
            "unavailable": "Unavailable",
        }[effective_source_authority]
        summary = (
            f"Automatic source: {automatic_copy}\n"
            f"Selected model OCR revision: {model_copy}\n"
            f"Your edit: {user_copy}\n"
            f"Effective source ({authority_label}): {effective_copy}"
        )
        self.source_authority_summary.setText(summary)
        self.source_authority_summary.setAccessibleDescription(
            "Source provenance comparison. " + summary.replace("\n", ". ")
        )
        self._source_provenance_available = any(
            value is not None
            for value in (
                automatic_source_text,
                model_revision_text,
                user_source_text,
                effective_source_text,
            )
        )
        self._sync_text_provenance_detail_visibility()
        badge_text = {
            "automatic": "Automatic OCR",
            "model": "Selected model OCR",
            "user": "Your source edit",
            "unavailable": "OCR unavailable",
        }[effective_source_authority]
        self.source_authority_badge.setText(badge_text)
        self.source_authority_badge.setProperty(
            "authority",
            effective_source_authority,
        )
        self.source_authority_badge.setAccessibleDescription(
            f"Effective source authority: {authority_label}."
        )
        self.source_authority_badge.style().unpolish(self.source_authority_badge)
        self.source_authority_badge.style().polish(self.source_authority_badge)
        self.source_text_frame.updateGeometry()
        self.ocr_rerun_button.setEnabled(bool(rerun_enabled))
        self.ocr_cancel_button.setEnabled(bool(cancel_enabled))
        self.ocr_cancel_button.setVisible(bool(cancel_enabled or busy))
        self.ocr_revision_status.setText(str(status_text))
        self.ocr_revision_status.setProperty("tone", str(status_tone))
        self.ocr_revision_status.setVisible(
            bool(busy or str(status_tone) in {"warning", "error"})
        )
        self.ocr_revision_status.setProperty(
            "state",
            "busy" if busy else "idle",
        )
        self.ocr_revision_status.setAccessibleDescription(
            "OCR revision status. " + str(status_text)
        )
        self.ocr_revision_status.style().unpolish(self.ocr_revision_status)
        self.ocr_revision_status.style().polish(self.ocr_revision_status)

    def set_translation_revision_state(
        self,
        *,
        automatic_target_text: str | None,
        model_revision_id: str | None,
        model_revision_text: str | None,
        model_provider: str | None,
        model_id: str | None,
        user_target_text: str | None,
        effective_target_text: str | None,
        effective_target_authority: str,
        rerun_enabled: bool,
        cancel_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
        mapped_pipeline_target_text: str | None = None,
    ) -> None:
        """Render four target provenance lanes without collapsing authority."""

        for field_name, value in (
            ("automatic_target_text", automatic_target_text),
            ("model_revision_id", model_revision_id),
            ("model_revision_text", model_revision_text),
            ("model_provider", model_provider),
            ("model_id", model_id),
            ("user_target_text", user_target_text),
            ("effective_target_text", effective_target_text),
            ("mapped_pipeline_target_text", mapped_pipeline_target_text),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
        model_values = (
            model_revision_id,
            model_revision_text,
            model_provider,
            model_id,
        )
        if any(value is None for value in model_values) and any(
            value is not None for value in model_values
        ):
            raise ValueError(
                "selected model revision identity, text, provider, and model "
                "must travel together"
            )
        if effective_target_authority not in {
            "automatic",
            "model",
            "mapped",
            "user",
            "unavailable",
        }:
            raise ValueError("unsupported effective target authority")

        automatic_copy = (
            "Unavailable - user-owned topology has no Automatic target"
            if automatic_target_text is None
            else self._target_authority_text(automatic_target_text)
        )
        if model_revision_text is None:
            model_copy = "Not selected"
        else:
            model_copy = (
                f"{self._public_provider_label(model_provider)} / "
                f"{self._public_model_label(model_id)}: "
                + self._target_authority_text(model_revision_text)
            )
        user_copy = (
            "No edit"
            if user_target_text is None
            else self._target_authority_text(user_target_text)
        )
        mapped_copy = (
            None
            if mapped_pipeline_target_text is None
            else self._target_authority_text(mapped_pipeline_target_text)
        )
        effective_copy = (
            "Unavailable"
            if effective_target_text is None
            else self._target_authority_text(effective_target_text)
        )
        authority_label = {
            "automatic": "Automatic target",
            "model": "Selected model translation revision",
            "mapped": "Mapped pipeline translation",
            "user": "Your edit",
            "unavailable": "Unavailable",
        }[effective_target_authority]
        summary_lines = [
            f"Automatic target: {automatic_copy}",
            f"Selected model translation revision: {model_copy}",
        ]
        if mapped_copy is not None:
            summary_lines.append(f"Mapped pipeline translation: {mapped_copy}")
        summary_lines.extend(
            (
                f"Your edit: {user_copy}",
                f"Effective target ({authority_label}): {effective_copy}",
            )
        )
        summary = "\n".join(summary_lines)
        self.target_authority_summary.setText(summary)
        self.target_authority_summary.setAccessibleDescription(
            "Target provenance comparison. " + summary.replace("\n", ". ")
        )
        self._target_provenance_available = any(
            value is not None
            for value in (
                automatic_target_text,
                model_revision_text,
                mapped_pipeline_target_text,
                user_target_text,
                effective_target_text,
            )
        )
        self._sync_text_provenance_detail_visibility()
        self.translation_rerun_button.setEnabled(bool(rerun_enabled))
        self.translation_cancel_button.setEnabled(bool(cancel_enabled))
        self.translation_cancel_button.setVisible(bool(cancel_enabled or busy))
        self.translation_revision_status.setText(str(status_text))
        self.translation_revision_status.setProperty("tone", str(status_tone))
        self.translation_revision_status.setVisible(
            bool(busy or str(status_tone) in {"warning", "error"})
        )
        self.translation_revision_status.setProperty(
            "state",
            "busy" if busy else "idle",
        )
        self.translation_revision_status.setAccessibleDescription(
            "Translation revision status. " + str(status_text)
        )
        self.translation_revision_status.style().unpolish(
            self.translation_revision_status
        )
        self.translation_revision_status.style().polish(
            self.translation_revision_status
        )

    def set_target_text_editor_state(
        self,
        *,
        draft_text: str,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        status_text: str,
        status_tone: str = "muted",
        history_identity: str | None = None,
        keep_existing_enabled: bool = False,
        freshness: str | None = None,
        restore_selected_model_translation: bool = False,
        restore_mapped_pipeline_translation: bool = False,
    ) -> None:
        """Apply one immutable target-edit view-model state to the inspector."""

        if not isinstance(draft_text, str):
            raise TypeError("draft_text must be a string")
        if history_identity is not None:
            self.target_text.bind_history_identity(history_identity)
        if self.target_text.exact_text() != draft_text:
            blocker = QtCore.QSignalBlocker(self.target_text)
            self.target_text.set_exact_text(draft_text)
            del blocker
        self.target_text.setEnabled(bool(editing_enabled))
        self.target_apply_button.setEnabled(bool(apply_enabled))
        self.target_apply_button.setVisible(bool(apply_enabled))
        self.target_cancel_button.setEnabled(bool(cancel_enabled))
        self.target_cancel_button.setVisible(bool(cancel_enabled))
        if restore_selected_model_translation and restore_mapped_pipeline_translation:
            raise ValueError("target restore presentation has competing bases")
        spans_columns = bool(
            restore_selected_model_translation
            or restore_mapped_pipeline_translation
        )
        if spans_columns != self._target_restore_spans_columns:
            if spans_columns:
                footer_layout = self.inspector_footer.layout()
                if not isinstance(footer_layout, QtWidgets.QHBoxLayout):
                    raise RuntimeError("inspector footer layout is unavailable")
                footer_layout.removeWidget(self.target_restore_button)
                self._target_actions_layout.removeWidget(self.target_restore_button)
                self._target_actions_layout.addWidget(
                    self.target_restore_button,
                    1,
                    0,
                    1,
                    2 if spans_columns else 1,
                )
            elif self.target_restore_button.parentWidget() is not self.inspector_footer:
                self._target_actions_layout.removeWidget(self.target_restore_button)
                footer_layout = self.inspector_footer.layout()
                if not isinstance(footer_layout, QtWidgets.QHBoxLayout):
                    raise RuntimeError("inspector footer layout is unavailable")
                footer_layout.insertWidget(1, self.target_restore_button)
            self._target_restore_spans_columns = spans_columns
        if restore_selected_model_translation:
            self.target_restore_button.setText(
                "Restore Selected Model Translation"
            )
            self.target_restore_button.setAccessibleName(
                "Restore Selected Model Translation"
            )
            self.target_restore_button.setAccessibleDescription(
                "Publish one target edit that re-exposes the exact immutable "
                "selected model translation without running a model."
            )
            self.target_restore_button.setToolTip(
                "Re-expose the exact selected model translation. No model or later owner runs."
            )
        elif restore_mapped_pipeline_translation:
            self.target_restore_button.setText(
                "Restore Mapped Pipeline Translation"
            )
            self.target_restore_button.setAccessibleName(
                "Restore Mapped Pipeline Translation"
            )
            self.target_restore_button.setAccessibleDescription(
                "Publish one target edit that re-exposes the exact translation "
                "mapped from immutable pipeline detection and OCR evidence."
            )
            self.target_restore_button.setToolTip(
                "Re-expose the exact mapped pipeline translation. No owner or pipeline stage runs."
            )
        else:
            self.target_restore_button.setText("Restore target")
            self.target_restore_button.setAccessibleName(
                "Restore automatic target text"
            )
            self.target_restore_button.setAccessibleDescription(
                "Publish one target edit that restores immutable Automatic target evidence."
            )
            self.target_restore_button.setToolTip(
                "Publish an explicit restore edit; automatic evidence is not changed."
            )
        QtWidgets.QWidget.setTabOrder(self.target_text, self.target_apply_button)
        QtWidgets.QWidget.setTabOrder(
            self.target_apply_button,
            self.target_cancel_button,
        )
        QtWidgets.QWidget.setTabOrder(
            self.target_cancel_button,
            self.target_restore_button,
        )
        self.target_restore_button.setEnabled(bool(restore_enabled))
        self.target_keep_existing_button.setVisible(bool(keep_existing_enabled))
        self.target_keep_existing_button.setEnabled(bool(keep_existing_enabled))
        if freshness is not None:
            self.text_freshness.setText(f"Freshness: {freshness}")
            self.text_freshness.setVisible(
                str(freshness).strip().lower() not in {"", "current"}
            )
        self.target_edit_status.setText(str(status_text))
        self.target_edit_status.setProperty("tone", str(status_tone))
        self.target_edit_status.setVisible(
            str(status_tone) in {"warning", "error", "info"}
        )
        self.target_edit_status.style().unpolish(self.target_edit_status)
        self.target_edit_status.style().polish(self.target_edit_status)

    def set_parent_membership_state(
        self,
        *,
        excluded: bool,
        enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply the selected parent's immutable membership command state."""

        self._parent_excluded = bool(excluded)
        self.parent_membership_button.setText(
            "Restore Parent" if self._parent_excluded else "Exclude Parent"
        )
        self.parent_membership_button.setAccessibleName(
            (
                "Restore selected parent to the effective page"
                if self._parent_excluded
                else "Exclude selected parent from the effective page"
            )
        )
        self.parent_membership_button.setEnabled(bool(enabled and not busy))
        self.parent_membership_status.setText(str(status_text))
        self.parent_membership_status.setProperty("tone", str(status_tone))
        self.parent_membership_status.style().unpolish(
            self.parent_membership_status
        )
        self.parent_membership_status.style().polish(
            self.parent_membership_status
        )

    def set_parent_geometry_state(
        self,
        *,
        automatic_bbox: tuple[int, int, int, int] | None,
        effective_bbox: tuple[int, int, int, int] | None,
        draft_bbox: tuple[int, int, int, int] | None,
        canvas_size: tuple[int, int] | None,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one immutable structural-geometry editor state."""

        for field_name, bbox in (
            ("automatic_bbox", automatic_bbox),
            ("effective_bbox", effective_bbox),
            ("draft_bbox", draft_bbox),
        ):
            if bbox is not None and (
                not isinstance(bbox, tuple)
                or len(bbox) != 4
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in bbox
                )
            ):
                raise TypeError(f"{field_name} must be an integer bbox or None")
        if canvas_size is not None and (
            not isinstance(canvas_size, tuple)
            or len(canvas_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in canvas_size
            )
        ):
            raise TypeError("canvas_size must contain two positive integers")
        stable_canvas = canvas_size or (1, 1)
        self._geometry_canvas_size = stable_canvas
        self._geometry_programmatic_update = True
        try:
            values = draft_bbox or effective_bbox or automatic_bbox or (0, 0, 1, 1)
            self.parent_geometry_spins["x"].setRange(
                0,
                max(stable_canvas[0] - 1, values[0]),
            )
            self.parent_geometry_spins["y"].setRange(
                0,
                max(stable_canvas[1] - 1, values[1]),
            )
            self.parent_geometry_spins["x"].setValue(values[0])
            self.parent_geometry_spins["y"].setValue(values[1])
            self._update_parent_geometry_ranges(preserve_bbox=values)
            self.parent_geometry_spins["width"].setValue(values[2])
            self.parent_geometry_spins["height"].setValue(values[3])
        finally:
            self._geometry_programmatic_update = False
        for spin in self.parent_geometry_spins.values():
            spin.setEnabled(bool(editing_enabled and not busy))
        self.parent_geometry_apply_button.setEnabled(
            bool(apply_enabled and not busy)
        )
        self.parent_geometry_cancel_button.setEnabled(
            bool(cancel_enabled and not busy)
        )
        self.parent_geometry_summary.setText(
            "Automatic: "
            + self._geometry_label(automatic_bbox)
            + "\nEffective: "
            + self._geometry_label(effective_bbox)
        )
        self.parent_geometry_canvas.setText(
            f"Page: {stable_canvas[0]} × {stable_canvas[1]} px"
            if canvas_size is not None
            else "Page: —"
        )
        self.parent_geometry_status.setText(str(status_text))
        self.parent_geometry_status.setProperty("tone", str(status_tone))
        self.parent_geometry_status.style().unpolish(self.parent_geometry_status)
        self.parent_geometry_status.style().polish(self.parent_geometry_status)
        draft_visible = bool(
            draft_bbox is not None
            and effective_bbox is not None
            and draft_bbox != effective_bbox
        )
        self.canvas.set_draft_geometry(
            draft_bbox if draft_visible else None,
            parent_id=self._current_parent_id if draft_visible else "",
        )

    def set_merge_parent_editor_state(
        self,
        *,
        candidates: tuple[tuple[str, str], ...],
        selected_partner_id: str,
        source_parent_ids: tuple[str, str] | None,
        source_bboxes: tuple[
            tuple[int, int, int, int],
            tuple[int, int, int, int],
        ] | None,
        merged_bbox: tuple[int, int, int, int] | None,
        merged_source_text: str,
        editing_enabled: bool,
        merge_enabled: bool,
        cancel_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one passive pipeline-backed Merge Parent editor state."""

        stable_candidates = tuple(candidates)
        if any(
            not isinstance(value, tuple)
            or len(value) != 2
            or not all(isinstance(item, str) and item for item in value)
            for value in stable_candidates
        ):
            raise TypeError("merge candidates must contain non-empty identity/label pairs")
        if len({value[0] for value in stable_candidates}) != len(stable_candidates):
            raise ValueError("merge candidates must have unique identities")
        stable_partner = str(selected_partner_id or "")
        if stable_partner and stable_partner not in {value[0] for value in stable_candidates}:
            raise ValueError("selected merge partner is not a listed candidate")
        if source_bboxes is None:
            if source_parent_ids is not None or merged_bbox is not None:
                raise ValueError("merge draft geometry must be supplied together")
        else:
            if (
                source_parent_ids is None
                or len(source_parent_ids) != 2
                or len(source_bboxes) != 2
                or merged_bbox is None
            ):
                raise ValueError("merge draft requires two sources and one merged bbox")
        self._merge_parent_programmatic_update = True
        try:
            self.merge_parent_partner.clear()
            self.merge_parent_partner.addItem("Choose adjacent pipeline block", None)
            for parent_id, label in stable_candidates:
                self.merge_parent_partner.addItem(label, parent_id)
            index = self.merge_parent_partner.findData(stable_partner or None)
            self.merge_parent_partner.setCurrentIndex(max(0, index))
        finally:
            self._merge_parent_programmatic_update = False
        controls_enabled = bool(editing_enabled and not busy)
        self.merge_parent_partner.setEnabled(controls_enabled)
        self.merge_parent_apply_button.setEnabled(bool(merge_enabled and not busy))
        self.merge_parent_cancel_button.setEnabled(bool(cancel_enabled and not busy))
        if source_bboxes is None or merged_bbox is None:
            self.merge_parent_summary.setText("No merge draft")
        else:
            self.merge_parent_summary.setText(
                "Source 1: "
                + self._geometry_label(source_bboxes[0])
                + "\nSource 2: "
                + self._geometry_label(source_bboxes[1])
                + "\nMerged range: "
                + self._geometry_label(merged_bbox)
                + "\nOrdered OCR: "
                + merged_source_text
            )
        self.merge_parent_status.setText(str(status_text))
        self.merge_parent_status.setProperty("tone", str(status_tone))
        self.merge_parent_status.setAccessibleDescription(
            f"{status_text} No translation, cleanup, style, layout, or rendering owner starts automatically."
        )
        self.merge_parent_status.style().unpolish(self.merge_parent_status)
        self.merge_parent_status.style().polish(self.merge_parent_status)
        self.canvas.set_merge_parent_draft(
            source_bboxes,
            merged_bbox=merged_bbox,
            source_parent_ids=source_parent_ids or ("", ""),
        )

    def set_split_parent_editor_state(
        self,
        *,
        source_bbox: tuple[int, int, int, int] | None,
        child_bboxes: tuple[
            tuple[int, int, int, int],
            tuple[int, int, int, int],
        ] | None,
        orientation: str | None,
        split_offset: int | None,
        editing_enabled: bool,
        split_enabled: bool,
        cancel_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one immutable topology-only Split Parent editor state."""

        if source_bbox is not None and (
            not isinstance(source_bbox, tuple)
            or len(source_bbox) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in source_bbox
            )
        ):
            raise TypeError("source_bbox must be an exact integer bbox or None")
        if orientation not in {None, "vertical", "horizontal"}:
            raise ValueError("orientation must be vertical, horizontal, or None")
        if split_offset is not None and (
            isinstance(split_offset, bool) or not isinstance(split_offset, int)
        ):
            raise TypeError("split_offset must be an exact integer or None")
        if child_bboxes is not None and (
            not isinstance(child_bboxes, tuple)
            or len(child_bboxes) != 2
            or any(
                not isinstance(bbox, tuple)
                or len(bbox) != 4
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in bbox
                )
                for bbox in child_bboxes
            )
        ):
            raise TypeError("child_bboxes must contain exactly two integer bboxes")
        self._split_parent_programmatic_update = True
        try:
            index = self.split_parent_orientation.findData(orientation)
            self.split_parent_orientation.setCurrentIndex(max(0, index))
            limit = 1
            if source_bbox is not None and orientation is not None:
                limit = (
                    source_bbox[2]
                    if orientation == "vertical"
                    else source_bbox[3]
                )
            self.split_parent_offset.setRange(1, max(1, limit - 1))
            self.split_parent_offset.setValue(
                max(1, min(int(split_offset or 1), max(1, limit - 1)))
            )
        finally:
            self._split_parent_programmatic_update = False
        controls_enabled = bool(editing_enabled and not busy)
        self.split_parent_orientation.setEnabled(controls_enabled)
        self.split_parent_offset.setEnabled(
            bool(controls_enabled and orientation is not None)
        )
        self.split_parent_apply_button.setEnabled(bool(split_enabled and not busy))
        self.split_parent_cancel_button.setEnabled(bool(cancel_enabled and not busy))
        if child_bboxes is None:
            self.split_parent_summary.setText("No split draft")
        else:
            first_label, second_label = (
                ("Left", "Right")
                if orientation == "vertical"
                else ("Top", "Bottom")
            )
            self.split_parent_summary.setText(
                f"{first_label}: {self._geometry_label(child_bboxes[0])}\n"
                f"{second_label}: {self._geometry_label(child_bboxes[1])}"
            )
        self.split_parent_status.setText(str(status_text))
        self.split_parent_status.setProperty("tone", str(status_tone))
        self.split_parent_status.setAccessibleDescription(
            f"{status_text} No downstream owner starts automatically."
        )
        self.split_parent_status.style().unpolish(self.split_parent_status)
        self.split_parent_status.style().polish(self.split_parent_status)
        self.canvas.set_split_parent_draft(
            child_bboxes,
            source_parent_id=self._current_parent_id if child_bboxes else "",
        )

    def set_reading_order_editor_state(
        self,
        *,
        automatic_order: tuple[str, ...],
        effective_order: tuple[str, ...],
        proposed_order: tuple[str, ...],
        selected_parent_id: str,
        excluded_parent_ids: tuple[str, ...],
        move_earlier_enabled: bool,
        move_later_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
        parent_labels: Mapping[str, str] | None = None,
        merge_retained_automatic_parent_ids: tuple[str, ...] = (),
    ) -> None:
        """Apply one immutable page-wide reading-order editor state."""

        orders = {
            "automatic_order": automatic_order,
            "effective_order": effective_order,
            "proposed_order": proposed_order,
        }
        for field_name, order in orders.items():
            if not isinstance(order, tuple) or any(
                not isinstance(parent_id, str) or not parent_id.strip()
                for parent_id in order
            ):
                raise TypeError(f"{field_name} must contain non-empty parent IDs")
            if len(order) != len(set(order)):
                raise ValueError(f"{field_name} must not contain duplicate parent IDs")
        automatic_parent_ids = frozenset(automatic_order)
        effective_parent_ids = frozenset(effective_order)
        proposed_parent_ids = frozenset(proposed_order)
        if not isinstance(merge_retained_automatic_parent_ids, tuple) or any(
            not isinstance(parent_id, str) or not parent_id.strip()
            for parent_id in merge_retained_automatic_parent_ids
        ):
            raise TypeError(
                "merge_retained_automatic_parent_ids must contain non-empty parent IDs"
            )
        merge_retained = frozenset(merge_retained_automatic_parent_ids)
        if len(merge_retained) != len(merge_retained_automatic_parent_ids):
            raise ValueError("merge-retained automatic parent IDs must be unique")
        if effective_parent_ids != proposed_parent_ids:
            raise ValueError(
                "effective and proposed reading orders must reference the same parent IDs"
            )
        if (
            not merge_retained.issubset(automatic_parent_ids)
            or merge_retained.intersection(effective_parent_ids)
            or not automatic_parent_ids.issubset(
                effective_parent_ids | merge_retained
            )
        ):
            raise ValueError(
                "automatic reading order must contain only effective or merge-retained parent IDs"
            )
        parent_ids = effective_parent_ids
        selected_parent_id = str(selected_parent_id or "").strip()
        if selected_parent_id and selected_parent_id not in parent_ids:
            raise ValueError("selected parent is absent from the reading order")
        if not isinstance(excluded_parent_ids, tuple) or any(
            not isinstance(parent_id, str) or not parent_id.strip()
            for parent_id in excluded_parent_ids
        ):
            raise TypeError("excluded_parent_ids must contain non-empty parent IDs")
        excluded = frozenset(excluded_parent_ids)
        if not excluded.issubset(parent_ids):
            raise ValueError("excluded parent is absent from the reading order")
        labels = {
            str(parent_id).strip(): str(label).strip()
            for parent_id, label in (parent_labels or {}).items()
        }
        if any(not parent_id or not label for parent_id, label in labels.items()):
            raise ValueError("parent_labels must contain non-empty identity/label pairs")
        if not frozenset(labels).issubset(parent_ids | merge_retained):
            raise ValueError("parent_labels references a parent outside the reading order")
        self.reading_order_automatic.setText(
            self._reading_order_label(
                automatic_order,
                selected_parent_id=selected_parent_id,
                excluded_parent_ids=excluded,
                parent_labels=labels,
            )
        )
        self.reading_order_effective.setText(
            self._reading_order_label(
                effective_order,
                selected_parent_id=selected_parent_id,
                excluded_parent_ids=excluded,
                parent_labels=labels,
            )
        )
        self.reading_order_proposed.setText(
            self._reading_order_label(
                proposed_order,
                selected_parent_id=selected_parent_id,
                excluded_parent_ids=excluded,
                parent_labels=labels,
            )
        )
        self.reading_order_earlier_button.setEnabled(
            bool(move_earlier_enabled and not busy)
        )
        self.reading_order_later_button.setEnabled(
            bool(move_later_enabled and not busy)
        )
        self.reading_order_apply_button.setEnabled(bool(apply_enabled and not busy))
        self.reading_order_cancel_button.setEnabled(bool(cancel_enabled and not busy))
        self.reading_order_status.setText(str(status_text))
        self.reading_order_status.setProperty("tone", str(status_tone))
        selected_label = labels.get(selected_parent_id, selected_parent_id or "none")
        self.reading_order_status.setAccessibleDescription(
            f"Selected parent: {selected_label}. {status_text}"
        )
        self.reading_order_status.style().unpolish(self.reading_order_status)
        self.reading_order_status.style().polish(self.reading_order_status)

    @staticmethod
    def _reading_order_label(
        order: tuple[str, ...],
        *,
        selected_parent_id: str,
        excluded_parent_ids: frozenset[str],
        parent_labels: Mapping[str, str],
    ) -> str:
        if not order:
            return "—"
        lines: list[str] = []
        for index, parent_id in enumerate(order, start=1):
            qualifiers: list[str] = []
            if parent_id == selected_parent_id:
                qualifiers.append("selected")
            if parent_id in excluded_parent_ids:
                qualifiers.append("excluded · fixed slot")
            suffix = f" ({'; '.join(qualifiers)})" if qualifiers else ""
            lines.append(f"{index}. {parent_labels.get(parent_id, parent_id)}{suffix}")
        return "\n".join(lines)

    def set_writing_mode_editor_state(
        self,
        *,
        automatic_writing_mode: str | None,
        user_writing_mode: str | None,
        effective_writing_mode: str | None,
        draft_writing_mode: str | None,
        writing_mode_authority: str,
        render_required: bool,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one immutable writing-mode editor state to the Layout card."""

        canonical = {"horizontal", "vertical"}
        for field_name, value in (
            ("automatic_writing_mode", automatic_writing_mode),
            ("user_writing_mode", user_writing_mode),
            ("effective_writing_mode", effective_writing_mode),
            ("draft_writing_mode", draft_writing_mode),
        ):
            if value is not None and value not in canonical:
                raise ValueError(f"{field_name} must be horizontal, vertical, or None")
        if writing_mode_authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "writing_mode_authority must be automatic, user, or unavailable"
            )
        self.writing_mode_automatic.setText(
            self._writing_mode_label(automatic_writing_mode)
        )
        self.writing_mode_user.setText(
            "No edit"
            if user_writing_mode is None
            else self._writing_mode_label(user_writing_mode)
        )
        self.writing_mode_effective.setText(
            self._writing_mode_label(effective_writing_mode)
        )
        self.writing_mode_authority.setText(
            "Your edit"
            if writing_mode_authority == "user"
            else (
                "Unavailable"
                if writing_mode_authority == "unavailable"
                else "Automatic"
            )
        )
        self._writing_mode_programmatic_update = True
        try:
            index = self.writing_mode_combo.findData(draft_writing_mode)
            self.writing_mode_combo.setCurrentIndex(index)
        finally:
            self._writing_mode_programmatic_update = False
        self.writing_mode_combo.setEnabled(bool(editing_enabled and not busy))
        self.writing_mode_set_button.setEnabled(bool(apply_enabled and not busy))
        self.writing_mode_cancel_button.setEnabled(bool(cancel_enabled and not busy))
        self.writing_mode_restore_button.setEnabled(
            bool(restore_enabled and not busy)
        )
        self.writing_mode_status.setText(str(status_text))
        self.writing_mode_status.setProperty("tone", str(status_tone))
        self.writing_mode_status.setAccessibleDescription(
            "Render required: "
            + ("yes" if render_required else "no")
            + ". "
            + str(status_text)
        )
        self.writing_mode_status.style().unpolish(self.writing_mode_status)
        self.writing_mode_status.style().polish(self.writing_mode_status)

    @staticmethod
    def _writing_mode_label(value: str | None) -> str:
        if value == "horizontal":
            return "Horizontal"
        if value == "vertical":
            return "Vertical"
        return "—"

    @QtCore.Slot()
    def _writing_mode_value_changed(self) -> None:
        if self._writing_mode_programmatic_update:
            return
        value = self.writing_mode_combo.currentData()
        if isinstance(value, str) and value in {"horizontal", "vertical"}:
            self.writing_mode_draft_changed.emit(value)

    def set_line_height_editor_state(
        self,
        *,
        automatic_line_height: float | None,
        user_line_height: float | None,
        effective_line_height: float | None,
        draft_line_height: float | None,
        line_height_authority: str,
        render_required: bool,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one immutable line-height editor state to the Layout card."""

        values = {
            field_name: self._canonical_line_height(value, field_name)
            for field_name, value in (
                ("automatic_line_height", automatic_line_height),
                ("user_line_height", user_line_height),
                ("effective_line_height", effective_line_height),
                ("draft_line_height", draft_line_height),
            )
        }
        if line_height_authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "line_height_authority must be automatic, user, or unavailable"
            )
        self.line_height_automatic.setText(
            self._line_height_label(values["automatic_line_height"])
        )
        self.line_height_user.setText(
            "No edit"
            if values["user_line_height"] is None
            else self._line_height_label(values["user_line_height"])
        )
        self.line_height_effective.setText(
            self._line_height_label(values["effective_line_height"])
        )
        self.line_height_authority.setText(
            "Your edit"
            if line_height_authority == "user"
            else (
                "Unavailable"
                if line_height_authority == "unavailable"
                else "Automatic"
            )
        )
        self._line_height_programmatic_update = True
        try:
            if values["draft_line_height"] is not None:
                self.line_height_spin.setValue(values["draft_line_height"])
            else:
                self.line_height_spin.setValue(self.line_height_spin.minimum())
        finally:
            self._line_height_programmatic_update = False
        self.line_height_spin.setEnabled(bool(editing_enabled and not busy))
        self.line_height_set_button.setEnabled(bool(apply_enabled and not busy))
        self.line_height_cancel_button.setEnabled(bool(cancel_enabled and not busy))
        self.line_height_restore_button.setEnabled(
            bool(restore_enabled and not busy)
        )
        self.line_height_status.setText(str(status_text))
        self.line_height_status.setProperty("tone", str(status_tone))
        self.line_height_status.setAccessibleDescription(
            "Render required: "
            + ("yes" if render_required else "no")
            + ". Valid range: 0.5 through 10.0. "
            + str(status_text)
        )
        self.line_height_status.style().unpolish(self.line_height_status)
        self.line_height_status.style().polish(self.line_height_status)

    @staticmethod
    def _canonical_line_height(
        value: float | None,
        field_name: str,
    ) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must be a finite number")
        result = float(value)
        if not math.isfinite(result) or not 0.5 <= result <= 10.0:
            raise ValueError(f"{field_name} must be between 0.5 and 10.0")
        return result

    @staticmethod
    def _line_height_label(value: float | None) -> str:
        if value is None:
            return "—"
        label = format(value, ".15g")
        return label if "." in label else f"{label}.0"

    @QtCore.Slot(float)
    def _line_height_value_changed(self, value: float) -> None:
        if self._line_height_programmatic_update:
            return
        line_height = self._canonical_line_height(value, "line_height")
        if line_height is not None:
            self.line_height_draft_changed.emit(line_height)

    def set_rotation_editor_state(
        self,
        *,
        automatic_rotation: float | None,
        user_rotation: float | None,
        effective_rotation: float | None,
        draft_rotation: float | None,
        rotation_authority: str,
        render_required: bool,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one immutable rotation editor state to the Layout card."""

        values = {
            field_name: self._canonical_rotation(value, field_name)
            for field_name, value in (
                ("automatic_rotation", automatic_rotation),
                ("user_rotation", user_rotation),
                ("effective_rotation", effective_rotation),
                ("draft_rotation", draft_rotation),
            )
        }
        if rotation_authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "rotation_authority must be automatic, user, or unavailable"
            )
        self.rotation_automatic.setText(
            self._rotation_label(values["automatic_rotation"])
        )
        self.rotation_user.setText(
            "No edit"
            if values["user_rotation"] is None
            else self._rotation_label(values["user_rotation"])
        )
        self.rotation_effective.setText(
            self._rotation_label(values["effective_rotation"])
        )
        self.rotation_authority.setText(
            "Your edit"
            if rotation_authority == "user"
            else (
                "Unavailable"
                if rotation_authority == "unavailable"
                else "Automatic"
            )
        )
        self._rotation_programmatic_update = True
        try:
            if values["draft_rotation"] is not None:
                self.rotation_spin.setValue(values["draft_rotation"])
            else:
                self.rotation_spin.setValue(0.0)
        finally:
            self._rotation_programmatic_update = False
        self.rotation_spin.setEnabled(bool(editing_enabled and not busy))
        self.rotation_set_button.setEnabled(bool(apply_enabled and not busy))
        self.rotation_cancel_button.setEnabled(bool(cancel_enabled and not busy))
        self.rotation_restore_button.setEnabled(
            bool(restore_enabled and not busy)
        )
        self.rotation_status.setText(str(status_text))
        self.rotation_status.setProperty("tone", str(status_tone))
        self.rotation_status.setAccessibleDescription(
            "Render required: "
            + ("yes" if render_required else "no")
            + ". Valid range: -45 through 45 clockwise degrees; pivot: visual center. "
            + str(status_text)
        )
        self.rotation_status.style().unpolish(self.rotation_status)
        self.rotation_status.style().polish(self.rotation_status)

    @staticmethod
    def _canonical_rotation(
        value: float | None,
        field_name: str,
    ) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must be a finite number")
        result = float(value)
        if not math.isfinite(result) or not -45.0 <= result <= 45.0:
            raise ValueError(f"{field_name} must be between -45 and 45")
        return result

    @staticmethod
    def _rotation_label(value: float | None) -> str:
        if value is None:
            return "—"
        label = format(value, ".15g")
        if "." not in label:
            label = f"{label}.0"
        return f"{label}° clockwise"

    @QtCore.Slot(float)
    def _rotation_value_changed(self, value: float) -> None:
        if self._rotation_programmatic_update:
            return
        rotation = self._canonical_rotation(value, "rotation")
        if rotation is not None:
            self.rotation_draft_changed.emit(rotation)

    def set_render_box_editor_state(
        self,
        *,
        automatic_render_box: tuple[int, int, int, int] | None,
        automatic_hard_bounds: tuple[int, int, int, int] | None,
        user_render_box: tuple[int, int, int, int] | None,
        effective_render_box: tuple[int, int, int, int] | None,
        draft_render_box: tuple[int, int, int, int] | None,
        render_box_authority: str,
        render_required: bool,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one exact selected-parent target-box state to Layout."""

        values = {
            field_name: self._canonical_render_box(value, field_name)
            for field_name, value in (
                ("automatic_render_box", automatic_render_box),
                ("automatic_hard_bounds", automatic_hard_bounds),
                ("user_render_box", user_render_box),
                ("effective_render_box", effective_render_box),
                ("draft_render_box", draft_render_box),
            )
        }
        if render_box_authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "render_box_authority must be automatic, user, or unavailable"
            )
        self.render_box_automatic.setText(
            self._render_box_label(values["automatic_render_box"])
        )
        self.render_box_hard_bounds.setText(
            self._render_box_label(values["automatic_hard_bounds"])
        )
        self.render_box_user.setText(
            "No edit"
            if values["user_render_box"] is None
            else self._render_box_label(values["user_render_box"])
        )
        self.render_box_effective.setText(
            self._render_box_label(values["effective_render_box"])
        )
        self.render_box_authority.setText(
            "Your edit"
            if render_box_authority == "user"
            else "Unavailable"
            if render_box_authority == "unavailable"
            else "Automatic"
        )
        self._render_box_programmatic_update = True
        try:
            draft = values["draft_render_box"] or (0, 0, 1, 1)
            for field_name, item in zip(
                ("x", "y", "width", "height"),
                draft,
            ):
                self.render_box_spins[field_name].setValue(item)
        finally:
            self._render_box_programmatic_update = False
        enabled = bool(editing_enabled and not busy)
        for spin in self.render_box_spins.values():
            spin.setEnabled(enabled)
        self.render_box_set_button.setEnabled(bool(apply_enabled and not busy))
        self.render_box_cancel_button.setEnabled(bool(cancel_enabled and not busy))
        self.render_box_restore_button.setEnabled(bool(restore_enabled and not busy))
        self.render_box_status.setText(str(status_text))
        self.render_box_status.setProperty("tone", str(status_tone))
        self.render_box_status.setAccessibleDescription(
            "Render required: "
            + ("yes" if render_required else "no")
            + ". Values are exact integer X, Y, width, and height inside immutable "
            "automatic hard bounds. Structural geometry and Preview remain separate. "
            + str(status_text)
        )
        self.render_box_status.style().unpolish(self.render_box_status)
        self.render_box_status.style().polish(self.render_box_status)

    @staticmethod
    def _canonical_render_box(
        value: tuple[int, int, int, int] | None,
        field_name: str,
    ) -> tuple[int, int, int, int] | None:
        if value is None:
            return None
        if (
            not isinstance(value, tuple)
            or len(value) != 4
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise TypeError(f"{field_name} must contain four exact integers")
        if value[2] <= 0 or value[3] <= 0:
            raise ValueError(f"{field_name} width and height must be positive")
        return value

    @staticmethod
    def _render_box_label(value: tuple[int, int, int, int] | None) -> str:
        if value is None:
            return "—"
        return f"X {value[0]} · Y {value[1]} · W {value[2]} · H {value[3]}"

    @QtCore.Slot(int)
    def _render_box_value_changed(self, _value: int) -> None:
        if self._render_box_programmatic_update:
            return
        self.render_box_draft_changed.emit(
            tuple(
                self.render_box_spins[field_name].value()
                for field_name in ("x", "y", "width", "height")
            )
        )

    def set_font_role_editor_state(
        self,
        *,
        automatic_font_role: str | None,
        user_font_role: str | None,
        effective_font_role: str | None,
        draft_font_role: str | None,
        font_role_authority: str,
        render_required: bool,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one immutable registered font-role editor state to Style."""

        role_ids = {
            str(self.font_role_choice.itemData(index))
            for index in range(self.font_role_choice.count())
        }
        values = {
            field_name: self._canonical_font_role(value, role_ids, field_name)
            for field_name, value in (
                ("automatic_font_role", automatic_font_role),
                ("user_font_role", user_font_role),
                ("effective_font_role", effective_font_role),
                ("draft_font_role", draft_font_role),
            )
        }
        if font_role_authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "font_role_authority must be automatic, user, or unavailable"
            )
        self.font_role_automatic.setText(
            self._font_role_label(values["automatic_font_role"])
        )
        self.font_role_user.setText(
            self._font_role_label(values["user_font_role"], none_label="No edit")
        )
        self.font_role_effective.setText(
            self._font_role_label(values["effective_font_role"])
        )
        self.font_role_authority.setText(
            "Your edit"
            if font_role_authority == "user"
            else "Unavailable"
            if font_role_authority == "unavailable"
            else "Automatic"
        )
        self._font_role_programmatic_update = True
        try:
            role = values["draft_font_role"]
            index = self.font_role_choice.findData(role) if role is not None else -1
            self.font_role_choice.setCurrentIndex(index)
        finally:
            self._font_role_programmatic_update = False
        self.font_role_choice.setEnabled(bool(editing_enabled and not busy))
        self.font_role_set_button.setEnabled(bool(apply_enabled and not busy))
        self.font_role_cancel_button.setEnabled(bool(cancel_enabled and not busy))
        self.font_role_restore_button.setEnabled(bool(restore_enabled and not busy))
        self.font_role_status.setText(str(status_text))
        self.font_role_status.setProperty("tone", str(status_tone))
        self.font_role_status.setAccessibleDescription(
            "Render required: "
            + ("yes" if render_required else "no")
            + ". Values are existing registered CJK roles only. "
            + str(status_text)
        )
        self.font_role_status.style().unpolish(self.font_role_status)
        self.font_role_status.style().polish(self.font_role_status)

    @staticmethod
    def _canonical_font_role(
        value: str | None,
        role_ids: set[str],
        field_name: str,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or value not in role_ids:
            raise ValueError(f"{field_name} must be a registered font role")
        return value

    def _font_role_label(
        self,
        value: str | None,
        *,
        none_label: str = "—",
    ) -> str:
        if value is None:
            return none_label
        index = self.font_role_choice.findData(value)
        if index < 0:
            raise ValueError("font role has no visible registered choice")
        return str(self.font_role_choice.itemText(index))

    @QtCore.Slot(int)
    def _font_role_value_changed(self, index: int) -> None:
        if self._font_role_programmatic_update or index < 0:
            return
        value = self.font_role_choice.itemData(index)
        if isinstance(value, str):
            self.font_role_draft_changed.emit(value)

    def set_font_weight_tier_editor_state(
        self,
        *,
        automatic_font_weight_tier: str | None,
        user_font_weight_tier: str | None,
        effective_font_weight_tier: str | None,
        draft_font_weight_tier: str | None,
        font_weight_tier_authority: str,
        render_required: bool,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one immutable registered font-weight-tier editor state."""

        tier_ids = {
            str(self.font_weight_tier_choice.itemData(index))
            for index in range(self.font_weight_tier_choice.count())
        }
        values = {
            field_name: self._canonical_font_weight_tier(
                value, tier_ids, field_name
            )
            for field_name, value in (
                ("automatic_font_weight_tier", automatic_font_weight_tier),
                ("user_font_weight_tier", user_font_weight_tier),
                ("effective_font_weight_tier", effective_font_weight_tier),
                ("draft_font_weight_tier", draft_font_weight_tier),
            )
        }
        if font_weight_tier_authority not in {
            "automatic",
            "user",
            "unavailable",
        }:
            raise ValueError(
                "font_weight_tier_authority must be automatic, user, or unavailable"
            )
        self.font_weight_tier_automatic.setText(
            self._font_weight_tier_label(values["automatic_font_weight_tier"])
        )
        self.font_weight_tier_user.setText(
            self._font_weight_tier_label(
                values["user_font_weight_tier"], none_label="No edit"
            )
        )
        self.font_weight_tier_effective.setText(
            self._font_weight_tier_label(values["effective_font_weight_tier"])
        )
        self.font_weight_tier_authority.setText(
            "Your edit"
            if font_weight_tier_authority == "user"
            else "Unavailable"
            if font_weight_tier_authority == "unavailable"
            else "Automatic"
        )
        self._font_weight_tier_programmatic_update = True
        try:
            tier = values["draft_font_weight_tier"]
            index = (
                self.font_weight_tier_choice.findData(tier)
                if tier is not None
                else -1
            )
            self.font_weight_tier_choice.setCurrentIndex(index)
        finally:
            self._font_weight_tier_programmatic_update = False
        self.font_weight_tier_choice.setEnabled(bool(editing_enabled and not busy))
        self.font_weight_tier_set_button.setEnabled(bool(apply_enabled and not busy))
        self.font_weight_tier_cancel_button.setEnabled(
            bool(cancel_enabled and not busy)
        )
        self.font_weight_tier_restore_button.setEnabled(
            bool(restore_enabled and not busy)
        )
        self.font_weight_tier_status.setText(str(status_text))
        self.font_weight_tier_status.setProperty("tone", str(status_tone))
        self.font_weight_tier_status.setAccessibleDescription(
            "Render required: "
            + ("yes" if render_required else "no")
            + ". Values are existing registered weight tiers within the "
            "automatic font family only. "
            + str(status_text)
        )
        self.font_weight_tier_status.style().unpolish(
            self.font_weight_tier_status
        )
        self.font_weight_tier_status.style().polish(self.font_weight_tier_status)

    @staticmethod
    def _canonical_font_weight_tier(
        value: str | None,
        tier_ids: set[str],
        field_name: str,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or value not in tier_ids:
            raise ValueError(f"{field_name} must be a registered font weight tier")
        return value

    def _font_weight_tier_label(
        self,
        value: str | None,
        *,
        none_label: str = "—",
    ) -> str:
        if value is None:
            return none_label
        index = self.font_weight_tier_choice.findData(value)
        if index < 0:
            raise ValueError("font weight tier has no visible registered choice")
        return str(self.font_weight_tier_choice.itemText(index))

    @QtCore.Slot(int)
    def _font_weight_tier_value_changed(self, index: int) -> None:
        if self._font_weight_tier_programmatic_update or index < 0:
            return
        value = self.font_weight_tier_choice.itemData(index)
        if isinstance(value, str):
            self.font_weight_tier_draft_changed.emit(value)

    def set_fill_color_editor_state(
        self,
        *,
        automatic_fill_color: str | None,
        user_fill_color: str | None,
        unresolved_user_fill_color: str | None,
        effective_fill_color: str | None,
        draft_fill_color: str | None,
        fill_color_authority: str,
        render_required: bool,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one immutable opaque fill-color editor state to Style."""

        values = {
            field_name: self._canonical_fill_color(value, field_name)
            for field_name, value in (
                ("automatic_fill_color", automatic_fill_color),
                ("user_fill_color", user_fill_color),
                ("effective_fill_color", effective_fill_color),
            )
        }
        if draft_fill_color is not None and not isinstance(draft_fill_color, str):
            raise TypeError("draft_fill_color must be a string or None")
        if (
            unresolved_user_fill_color is not None
            and not isinstance(unresolved_user_fill_color, str)
        ):
            raise TypeError("unresolved_user_fill_color must be a string or None")
        if fill_color_authority not in {
            "automatic",
            "user",
            "unresolved",
            "unavailable",
        }:
            raise ValueError(
                "fill_color_authority must be automatic, user, unresolved, or unavailable"
            )
        self.fill_color_automatic.setText(
            values["automatic_fill_color"] or "—"
        )
        if fill_color_authority == "unresolved":
            self.fill_color_user.setText(
                f"{unresolved_user_fill_color!r} (unsupported)"
                if unresolved_user_fill_color is not None
                else "Unsupported saved value"
            )
            self.fill_color_effective.setText("Unavailable")
            self.fill_color_authority.setText("Unresolved saved edit")
        else:
            self.fill_color_user.setText(
                values["user_fill_color"] or "No edit"
            )
            self.fill_color_effective.setText(
                values["effective_fill_color"] or "—"
            )
            self.fill_color_authority.setText(
                "Your edit"
                if fill_color_authority == "user"
                else (
                    "Unavailable"
                    if fill_color_authority == "unavailable"
                    else "Automatic"
                )
            )
        self._fill_color_programmatic_update = True
        try:
            self.fill_color_edit.setText(draft_fill_color or "")
        finally:
            self._fill_color_programmatic_update = False
        draft_swatch: str | None = None
        if draft_fill_color is not None:
            try:
                draft_swatch = self._canonical_fill_color(
                    draft_fill_color,
                    "draft_fill_color",
                )
            except (TypeError, ValueError):
                draft_swatch = None
        self._set_fill_color_swatch(
            draft_swatch or values["effective_fill_color"]
        )
        self.fill_color_edit.setEnabled(bool(editing_enabled and not busy))
        self.fill_color_set_button.setEnabled(bool(apply_enabled and not busy))
        self.fill_color_cancel_button.setEnabled(bool(cancel_enabled and not busy))
        self.fill_color_restore_button.setEnabled(
            bool(restore_enabled and not busy)
        )
        self.fill_color_status.setText(str(status_text))
        self.fill_color_status.setProperty("tone", str(status_tone))
        self.fill_color_status.setAccessibleDescription(
            "Render required: "
            + ("yes" if render_required else "no")
            + ". Valid form: exact opaque #RRGGBB; alpha is unsupported. "
            + str(status_text)
        )
        self.fill_color_status.style().unpolish(self.fill_color_status)
        self.fill_color_status.style().polish(self.fill_color_status)

    @staticmethod
    def _canonical_fill_color(
        value: str | None,
        field_name: str,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        if (
            len(value) != 7
            or not value.startswith("#")
            or any(character not in "0123456789abcdefABCDEF" for character in value[1:])
        ):
            raise ValueError(f"{field_name} must be exact opaque #RRGGBB")
        return value.upper()

    def _set_fill_color_swatch(self, value: str | None) -> None:
        palette = self.fill_color_swatch.palette()
        if value is None:
            self.fill_color_swatch.setAutoFillBackground(False)
            self.fill_color_swatch.setAccessibleDescription(
                "No effective opaque fill color is available"
            )
            return
        color = QtGui.QColor(value)
        if not color.isValid() or color.alpha() != 255:
            raise ValueError("swatch color must be an opaque #RRGGBB value")
        palette.setColor(QtGui.QPalette.ColorRole.Window, color)
        self.fill_color_swatch.setPalette(palette)
        self.fill_color_swatch.setAutoFillBackground(True)
        self.fill_color_swatch.setAccessibleDescription(
            f"Opaque fill-color swatch {value}"
        )

    @QtCore.Slot(str)
    def _fill_color_value_changed(self, value: str) -> None:
        if self._fill_color_programmatic_update:
            return
        self.fill_color_draft_changed.emit(str(value))

    def set_outline_color_editor_state(
        self,
        *,
        automatic_outline_color: str | None,
        user_outline_color: str | None,
        unresolved_user_outline_color: str | None,
        effective_outline_color: str | None,
        draft_outline_color: str | None,
        outline_color_authority: str,
        render_required: bool,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one immutable opaque outline-color editor state to Style."""

        values = {
            field_name: self._canonical_outline_color(value, field_name)
            for field_name, value in (
                ("automatic_outline_color", automatic_outline_color),
                ("user_outline_color", user_outline_color),
                ("effective_outline_color", effective_outline_color),
            )
        }
        if draft_outline_color is not None and not isinstance(draft_outline_color, str):
            raise TypeError("draft_outline_color must be a string or None")
        if (
            unresolved_user_outline_color is not None
            and not isinstance(unresolved_user_outline_color, str)
        ):
            raise TypeError("unresolved_user_outline_color must be a string or None")
        if outline_color_authority not in {
            "automatic",
            "user",
            "unresolved",
            "unavailable",
        }:
            raise ValueError(
                "outline_color_authority must be automatic, user, unresolved, or unavailable"
            )
        self.outline_color_automatic.setText(
            values["automatic_outline_color"] or "—"
        )
        if outline_color_authority == "unresolved":
            self.outline_color_user.setText(
                f"{unresolved_user_outline_color!r} (unsupported)"
                if unresolved_user_outline_color is not None
                else "Unsupported saved value"
            )
            self.outline_color_effective.setText("Unavailable")
            self.outline_color_authority.setText("Unresolved saved edit")
        else:
            self.outline_color_user.setText(
                values["user_outline_color"] or "No edit"
            )
            self.outline_color_effective.setText(
                values["effective_outline_color"] or "—"
            )
            self.outline_color_authority.setText(
                "Your edit"
                if outline_color_authority == "user"
                else (
                    "Unavailable"
                    if outline_color_authority == "unavailable"
                    else "Automatic"
                )
            )
        self._outline_color_programmatic_update = True
        try:
            self.outline_color_edit.setText(draft_outline_color or "")
        finally:
            self._outline_color_programmatic_update = False
        draft_swatch: str | None = None
        if draft_outline_color is not None:
            try:
                draft_swatch = self._canonical_outline_color(
                    draft_outline_color,
                    "draft_outline_color",
                )
            except (TypeError, ValueError):
                draft_swatch = None
        self._set_outline_color_swatch(
            draft_swatch or values["effective_outline_color"]
        )
        self.outline_color_edit.setEnabled(bool(editing_enabled and not busy))
        self.outline_color_set_button.setEnabled(bool(apply_enabled and not busy))
        self.outline_color_cancel_button.setEnabled(bool(cancel_enabled and not busy))
        self.outline_color_restore_button.setEnabled(
            bool(restore_enabled and not busy)
        )
        self.outline_color_status.setText(str(status_text))
        self.outline_color_status.setProperty("tone", str(status_tone))
        self.outline_color_status.setAccessibleDescription(
            "Render required: "
            + ("yes" if render_required else "no")
            + ". Valid form: exact opaque #RRGGBB; alpha is unsupported. "
            + str(status_text)
        )
        self.outline_color_status.style().unpolish(self.outline_color_status)
        self.outline_color_status.style().polish(self.outline_color_status)

    @staticmethod
    def _canonical_outline_color(
        value: str | None,
        field_name: str,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        if (
            len(value) != 7
            or not value.startswith("#")
            or any(character not in "0123456789abcdefABCDEF" for character in value[1:])
        ):
            raise ValueError(f"{field_name} must be exact opaque #RRGGBB")
        return value.upper()

    def _set_outline_color_swatch(self, value: str | None) -> None:
        palette = self.outline_color_swatch.palette()
        if value is None:
            self.outline_color_swatch.setAutoFillBackground(False)
            self.outline_color_swatch.setAccessibleDescription(
                "No effective opaque outline color is available"
            )
            return
        color = QtGui.QColor(value)
        if not color.isValid() or color.alpha() != 255:
            raise ValueError("swatch color must be an opaque #RRGGBB value")
        palette.setColor(QtGui.QPalette.ColorRole.Window, color)
        self.outline_color_swatch.setPalette(palette)
        self.outline_color_swatch.setAutoFillBackground(True)
        self.outline_color_swatch.setAccessibleDescription(
            f"Opaque outline-color swatch {value}"
        )

    @QtCore.Slot(str)
    def _outline_color_value_changed(self, value: str) -> None:
        if self._outline_color_programmatic_update:
            return
        self.outline_color_draft_changed.emit(str(value))

    def set_outline_width_editor_state(
        self,
        *,
        automatic_outline_width: float | None,
        user_outline_width: float | None,
        effective_outline_width: float | None,
        draft_outline_width: float | None,
        outline_width_authority: str,
        render_required: bool,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one immutable outline-width editor state to Style."""

        def canonical(value: float | None, field_name: str) -> float | None:
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric or None")
            result = float(value)
            if not math.isfinite(result) or not 0.0 <= result <= 128.0:
                raise ValueError(f"{field_name} must be finite from 0 through 128")
            return result

        automatic = canonical(automatic_outline_width, "automatic_outline_width")
        user = canonical(user_outline_width, "user_outline_width")
        effective = canonical(effective_outline_width, "effective_outline_width")
        draft = canonical(draft_outline_width, "draft_outline_width")
        if outline_width_authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "outline_width_authority must be automatic, user, or unavailable"
            )
        self.outline_width_automatic.setText(
            "—" if automatic is None else f"{automatic:g} px"
        )
        self.outline_width_user.setText(
            "No edit" if user is None else f"{user:g} px"
        )
        self.outline_width_effective.setText(
            "—" if effective is None else f"{effective:g} px"
        )
        self.outline_width_authority.setText(
            "Your edit"
            if outline_width_authority == "user"
            else (
                "Automatic"
                if outline_width_authority == "automatic"
                else "Unavailable"
            )
        )
        self._outline_width_programmatic_update = True
        try:
            self.outline_width_edit.setValue(draft if draft is not None else 0.0)
        finally:
            self._outline_width_programmatic_update = False
        self.outline_width_edit.setEnabled(bool(editing_enabled and not busy))
        self.outline_width_set_button.setEnabled(bool(apply_enabled and not busy))
        self.outline_width_cancel_button.setEnabled(bool(cancel_enabled and not busy))
        self.outline_width_restore_button.setEnabled(
            bool(restore_enabled and not busy)
        )
        self.outline_width_status.setText(str(status_text))
        self.outline_width_status.setProperty("tone", str(status_tone))
        self.outline_width_status.setAccessibleDescription(
            "Render required: "
            + ("yes" if render_required else "no")
            + ". Valid range: 0 through 128 pixels; zero disables the outline. "
            + str(status_text)
        )
        self.outline_width_status.style().unpolish(self.outline_width_status)
        self.outline_width_status.style().polish(self.outline_width_status)

    @QtCore.Slot(float)
    def _outline_width_value_changed(self, value: float) -> None:
        if self._outline_width_programmatic_update:
            return
        self.outline_width_draft_changed.emit(float(value))

    def set_preferred_size_editor_state(
        self,
        *,
        automatic_preferred_size: float | None,
        user_preferred_size: float | None,
        effective_preferred_size: float | None,
        draft_preferred_size: float | None,
        preferred_size_authority: str,
        render_required: bool,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one immutable preferred-size editor state to Style."""

        def canonical(value: float | None, field_name: str) -> float | None:
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric or None")
            result = float(value)
            if not math.isfinite(result) or not 0.1 <= result <= 2048.0:
                raise ValueError(
                    f"{field_name} must be finite from 0.1 through 2048"
                )
            return result

        automatic = canonical(
            automatic_preferred_size, "automatic_preferred_size"
        )
        user = canonical(user_preferred_size, "user_preferred_size")
        effective = canonical(effective_preferred_size, "effective_preferred_size")
        draft = canonical(draft_preferred_size, "draft_preferred_size")
        if preferred_size_authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "preferred_size_authority must be automatic, user, or unavailable"
            )
        self.preferred_size_automatic.setText(
            "—" if automatic is None else f"{automatic:g} px"
        )
        self.preferred_size_user.setText(
            "No edit" if user is None else f"{user:g} px"
        )
        self.preferred_size_effective.setText(
            "—" if effective is None else f"{effective:g} px"
        )
        self.preferred_size_authority.setText(
            "Your edit"
            if preferred_size_authority == "user"
            else (
                "Automatic"
                if preferred_size_authority == "automatic"
                else "Unavailable"
            )
        )
        self._preferred_size_programmatic_update = True
        try:
            self.preferred_size_edit.setValue(
                draft if draft is not None else 0.1
            )
        finally:
            self._preferred_size_programmatic_update = False
        self.preferred_size_edit.setEnabled(bool(editing_enabled and not busy))
        self.preferred_size_set_button.setEnabled(bool(apply_enabled and not busy))
        self.preferred_size_cancel_button.setEnabled(
            bool(cancel_enabled and not busy)
        )
        self.preferred_size_restore_button.setEnabled(
            bool(restore_enabled and not busy)
        )
        self.preferred_size_status.setText(str(status_text))
        self.preferred_size_status.setProperty("tone", str(status_tone))
        self.preferred_size_status.setAccessibleDescription(
            "Render required: "
            + ("yes" if render_required else "no")
            + ". Valid range: 0.1 through 2048 pixels. This is a fit-quality "
            "target, not a minimum or admission rule. "
            + str(status_text)
        )
        self.preferred_size_status.style().unpolish(self.preferred_size_status)
        self.preferred_size_status.style().polish(self.preferred_size_status)

    @QtCore.Slot(float)
    def _preferred_size_value_changed(self, value: float) -> None:
        if self._preferred_size_programmatic_update:
            return
        self.preferred_size_draft_changed.emit(float(value))

    def set_shadow_color_editor_state(
        self,
        *,
        automatic_shadow_color: str | None,
        user_shadow_color: str | None,
        effective_shadow_color: str | None,
        draft_shadow_color: str | None,
        shadow_color_authority: str,
        render_required: bool,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one immutable RGBA shadow-color editor state to Style."""

        values = {
            field_name: self._canonical_shadow_color(value, field_name)
            for field_name, value in (
                ("automatic_shadow_color", automatic_shadow_color),
                ("user_shadow_color", user_shadow_color),
                ("effective_shadow_color", effective_shadow_color),
            )
        }
        if draft_shadow_color is not None and not isinstance(draft_shadow_color, str):
            raise TypeError("draft_shadow_color must be a string or None")
        if shadow_color_authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "shadow_color_authority must be automatic, user, or unavailable"
            )
        self.shadow_color_automatic.setText(
            values["automatic_shadow_color"] or "—"
        )
        self.shadow_color_user.setText(values["user_shadow_color"] or "No edit")
        self.shadow_color_effective.setText(
            values["effective_shadow_color"] or "—"
        )
        self.shadow_color_authority.setText(
            "Your edit"
            if shadow_color_authority == "user"
            else (
                "Automatic"
                if shadow_color_authority == "automatic"
                else "Unavailable"
            )
        )
        self._shadow_color_programmatic_update = True
        try:
            self.shadow_color_edit.setText(draft_shadow_color or "")
        finally:
            self._shadow_color_programmatic_update = False
        draft_swatch: str | None = None
        if draft_shadow_color is not None:
            try:
                draft_swatch = self._canonical_shadow_color(
                    draft_shadow_color,
                    "draft_shadow_color",
                )
            except (TypeError, ValueError):
                draft_swatch = None
        self._set_shadow_color_swatch(
            draft_swatch or values["effective_shadow_color"]
        )
        self.shadow_color_edit.setEnabled(bool(editing_enabled and not busy))
        self.shadow_color_set_button.setEnabled(bool(apply_enabled and not busy))
        self.shadow_color_cancel_button.setEnabled(bool(cancel_enabled and not busy))
        self.shadow_color_restore_button.setEnabled(
            bool(restore_enabled and not busy)
        )
        self.shadow_color_status.setText(str(status_text))
        self.shadow_color_status.setProperty("tone", str(status_tone))
        self.shadow_color_status.setAccessibleDescription(
            "Render required: "
            + ("yes" if render_required else "no")
            + ". Valid form: exact #RRGGBB or #RRGGBBAA; six digits use opaque "
            "alpha and transparent RGBA remains an explicit Set. "
            + str(status_text)
        )
        self.shadow_color_status.style().unpolish(self.shadow_color_status)
        self.shadow_color_status.style().polish(self.shadow_color_status)

    @staticmethod
    def _canonical_shadow_color(
        value: str | None,
        field_name: str,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        if (
            len(value) not in {7, 9}
            or not value.startswith("#")
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in value[1:]
            )
        ):
            raise ValueError(f"{field_name} must be exact #RRGGBB or #RRGGBBAA")
        canonical = value.upper()
        return canonical + "FF" if len(canonical) == 7 else canonical

    def _set_shadow_color_swatch(self, value: str | None) -> None:
        palette = self.shadow_color_swatch.palette()
        if value is None:
            self.shadow_color_swatch.setAutoFillBackground(False)
            self.shadow_color_swatch.setAccessibleDescription(
                "No effective shadow color is available"
            )
            return
        canonical = self._canonical_shadow_color(value, "swatch color")
        assert canonical is not None
        color = QtGui.QColor(
            int(canonical[1:3], 16),
            int(canonical[3:5], 16),
            int(canonical[5:7], 16),
            int(canonical[7:9], 16),
        )
        palette.setColor(QtGui.QPalette.ColorRole.Window, color)
        self.shadow_color_swatch.setPalette(palette)
        self.shadow_color_swatch.setAutoFillBackground(True)
        self.shadow_color_swatch.setAccessibleDescription(
            f"Shadow-color RGBA swatch {canonical}"
        )

    @QtCore.Slot(str)
    def _shadow_color_value_changed(self, value: str) -> None:
        if self._shadow_color_programmatic_update:
            return
        self.shadow_color_draft_changed.emit(str(value))

    def set_shadow_blur_editor_state(
        self,
        *,
        automatic_shadow_blur: float | None,
        user_shadow_blur: float | None,
        effective_shadow_blur: float | None,
        draft_shadow_blur: float | None,
        shadow_blur_authority: str,
        render_required: bool,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one immutable shadow-blur editor state to Style."""

        def canonical(value: float | None, field_name: str) -> float | None:
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric or None")
            result = float(value)
            if not math.isfinite(result) or not 0.0 <= result <= 64.0:
                raise ValueError(f"{field_name} must be finite from 0 through 64")
            return result

        automatic = canonical(
            automatic_shadow_blur, "automatic_shadow_blur"
        )
        user = canonical(user_shadow_blur, "user_shadow_blur")
        effective = canonical(effective_shadow_blur, "effective_shadow_blur")
        draft = canonical(draft_shadow_blur, "draft_shadow_blur")
        if shadow_blur_authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "shadow_blur_authority must be automatic, user, or unavailable"
            )
        self.shadow_blur_automatic.setText(
            "—" if automatic is None else f"{automatic:g} px"
        )
        self.shadow_blur_user.setText(
            "No edit" if user is None else f"{user:g} px"
        )
        self.shadow_blur_effective.setText(
            "—" if effective is None else f"{effective:g} px"
        )
        self.shadow_blur_authority.setText(
            "Your edit"
            if shadow_blur_authority == "user"
            else (
                "Automatic"
                if shadow_blur_authority == "automatic"
                else "Unavailable"
            )
        )
        self._shadow_blur_programmatic_update = True
        try:
            self.shadow_blur_edit.setValue(draft if draft is not None else 0.0)
        finally:
            self._shadow_blur_programmatic_update = False
        self.shadow_blur_edit.setEnabled(bool(editing_enabled and not busy))
        self.shadow_blur_set_button.setEnabled(bool(apply_enabled and not busy))
        self.shadow_blur_cancel_button.setEnabled(
            bool(cancel_enabled and not busy)
        )
        self.shadow_blur_restore_button.setEnabled(
            bool(restore_enabled and not busy)
        )
        self.shadow_blur_status.setText(str(status_text))
        self.shadow_blur_status.setProperty("tone", str(status_tone))
        self.shadow_blur_status.setAccessibleDescription(
            "Render required: "
            + ("yes" if render_required else "no")
            + ". Valid range: 0 through 64 pixels; zero is an explicit blur "
            "value, not Restore Automatic. "
            + str(status_text)
        )
        self.shadow_blur_status.style().unpolish(self.shadow_blur_status)
        self.shadow_blur_status.style().polish(self.shadow_blur_status)

    @QtCore.Slot(float)
    def _shadow_blur_value_changed(self, value: float) -> None:
        if self._shadow_blur_programmatic_update:
            return
        self.shadow_blur_draft_changed.emit(float(value))

    def set_shadow_offset_editor_state(
        self,
        *,
        automatic_shadow_offset: tuple[float, float] | None,
        user_shadow_offset: tuple[float, float] | None,
        effective_shadow_offset: tuple[float, float] | None,
        draft_shadow_offset: tuple[float, float] | None,
        shadow_offset_authority: str,
        render_required: bool,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one immutable X/Y shadow-offset editor state to Style."""

        def canonical(
            value: tuple[float, float] | None,
            field_name: str,
        ) -> tuple[float, float] | None:
            if value is None:
                return None
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
            ):
                raise TypeError(f"{field_name} must contain X and Y or be None")
            result = (float(value[0]), float(value[1]))
            if any(
                not math.isfinite(item) or not -256.0 <= item <= 256.0
                for item in result
            ):
                raise ValueError(
                    f"{field_name} components must be finite from -256 through 256"
                )
            return result

        automatic = canonical(automatic_shadow_offset, "automatic_shadow_offset")
        user = canonical(user_shadow_offset, "user_shadow_offset")
        effective = canonical(effective_shadow_offset, "effective_shadow_offset")
        draft = canonical(draft_shadow_offset, "draft_shadow_offset")
        if shadow_offset_authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "shadow_offset_authority must be automatic, user, or unavailable"
            )

        def label(value: tuple[float, float] | None) -> str:
            return "—" if value is None else f"X {value[0]:g} px · Y {value[1]:g} px"

        self.shadow_offset_automatic.setText(label(automatic))
        self.shadow_offset_user.setText("No edit" if user is None else label(user))
        self.shadow_offset_effective.setText(label(effective))
        self.shadow_offset_authority.setText(
            "Your edit"
            if shadow_offset_authority == "user"
            else (
                "Automatic"
                if shadow_offset_authority == "automatic"
                else "Unavailable"
            )
        )
        self._shadow_offset_programmatic_update = True
        try:
            self.shadow_offset_x_edit.setValue(draft[0] if draft is not None else 0.0)
            self.shadow_offset_y_edit.setValue(draft[1] if draft is not None else 0.0)
        finally:
            self._shadow_offset_programmatic_update = False
        for control in (self.shadow_offset_x_edit, self.shadow_offset_y_edit):
            control.setEnabled(bool(editing_enabled and not busy))
        self.shadow_offset_set_button.setEnabled(bool(apply_enabled and not busy))
        self.shadow_offset_cancel_button.setEnabled(
            bool(cancel_enabled and not busy)
        )
        self.shadow_offset_restore_button.setEnabled(
            bool(restore_enabled and not busy)
        )
        self.shadow_offset_status.setText(str(status_text))
        self.shadow_offset_status.setProperty("tone", str(status_tone))
        self.shadow_offset_status.setAccessibleDescription(
            "Render required: "
            + ("yes" if render_required else "no")
            + ". Valid range: -256 through 256 pixels on each axis; zero is "
            "an explicit component, not Restore Automatic. "
            + str(status_text)
        )
        self.shadow_offset_status.style().unpolish(self.shadow_offset_status)
        self.shadow_offset_status.style().polish(self.shadow_offset_status)

    @QtCore.Slot(float)
    def _shadow_offset_value_changed(self, _value: float) -> None:
        if self._shadow_offset_programmatic_update:
            return
        self.shadow_offset_draft_changed.emit(
            (
                float(self.shadow_offset_x_edit.value()),
                float(self.shadow_offset_y_edit.value()),
            )
        )

    def set_shadow_visibility_editor_state(
        self,
        *,
        automatic_shadow_enabled: bool | None,
        user_shadow_enabled: bool | None,
        effective_shadow_enabled: bool | None,
        draft_shadow_enabled: bool | None,
        shadow_enabled_authority: str,
        render_required: bool,
        editing_enabled: bool,
        apply_enabled: bool,
        cancel_enabled: bool,
        restore_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        """Apply one immutable hide-or-restore shadow-visibility state."""

        for field_name, value in (
            ("automatic_shadow_enabled", automatic_shadow_enabled),
            ("user_shadow_enabled", user_shadow_enabled),
            ("effective_shadow_enabled", effective_shadow_enabled),
            ("draft_shadow_enabled", draft_shadow_enabled),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{field_name} must be a boolean or None")
        if user_shadow_enabled not in {None, False}:
            raise ValueError("user_shadow_enabled supports only false or None")
        if draft_shadow_enabled not in {None, False, True}:
            raise ValueError("draft_shadow_enabled must be a boolean or None")
        if shadow_enabled_authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "shadow_enabled_authority must be automatic, user, or unavailable"
            )

        def label(value: bool | None, *, user: bool = False) -> str:
            if value is None:
                return "No edit" if user else "—"
            return "Visible" if value else "Hidden"

        self.shadow_visibility_automatic.setText(label(automatic_shadow_enabled))
        self.shadow_visibility_user.setText(
            label(user_shadow_enabled, user=True)
        )
        self.shadow_visibility_effective.setText(label(effective_shadow_enabled))
        self.shadow_visibility_authority.setText(
            "Your edit"
            if shadow_enabled_authority == "user"
            else (
                "Automatic"
                if shadow_enabled_authority == "automatic"
                else "Unavailable"
            )
        )
        self._shadow_visibility_programmatic_update = True
        try:
            self.shadow_visibility_edit.setChecked(
                bool(draft_shadow_enabled)
                if draft_shadow_enabled is not None
                else False
            )
        finally:
            self._shadow_visibility_programmatic_update = False
        self.shadow_visibility_edit.setEnabled(bool(editing_enabled and not busy))
        self.shadow_visibility_set_button.setEnabled(bool(apply_enabled and not busy))
        self.shadow_visibility_cancel_button.setEnabled(
            bool(cancel_enabled and not busy)
        )
        self.shadow_visibility_restore_button.setEnabled(
            bool(restore_enabled and not busy)
        )
        self.shadow_visibility_status.setText(str(status_text))
        self.shadow_visibility_status.setProperty("tone", str(status_tone))
        self.shadow_visibility_status.setAccessibleDescription(
            "Render required: "
            + ("yes" if render_required else "no")
            + ". Only a visible automatic shadow may be hidden. This control "
            "cannot create or enable a shadow. "
            + str(status_text)
        )
        self.shadow_visibility_status.style().unpolish(
            self.shadow_visibility_status
        )
        self.shadow_visibility_status.style().polish(
            self.shadow_visibility_status
        )

    @QtCore.Slot(bool)
    def _shadow_visibility_value_changed(self, value: bool) -> None:
        if self._shadow_visibility_programmatic_update:
            return
        self.shadow_visibility_draft_changed.emit(bool(value))

    @staticmethod
    def _geometry_label(value: tuple[int, int, int, int] | None) -> str:
        if value is None:
            return "—"
        return f"x {value[0]}, y {value[1]}, w {value[2]}, h {value[3]}"

    @QtCore.Slot()
    def _parent_geometry_value_changed(self) -> None:
        if self._geometry_programmatic_update:
            return
        self._geometry_programmatic_update = True
        try:
            self._update_parent_geometry_ranges()
            bbox = self._parent_geometry_values()
        finally:
            self._geometry_programmatic_update = False
        self.parent_geometry_draft_changed.emit(bbox)

    @QtCore.Slot(int)
    def _merge_parent_partner_value_changed(self, _index: int) -> None:
        if self._merge_parent_programmatic_update:
            return
        self.merge_parent_partner_changed.emit(
            self.merge_parent_partner.currentData()
        )

    @QtCore.Slot(int)
    def _split_parent_orientation_value_changed(self, _index: int) -> None:
        if self._split_parent_programmatic_update:
            return
        self.split_parent_orientation_changed.emit(
            self.split_parent_orientation.currentData()
        )

    @QtCore.Slot(int)
    def _split_parent_offset_value_changed(self, value: int) -> None:
        if self._split_parent_programmatic_update:
            return
        self.split_parent_offset_changed.emit(int(value))

    def _update_parent_geometry_ranges(
        self,
        *,
        preserve_bbox: tuple[int, int, int, int] | None = None,
    ) -> None:
        page_width, page_height = self._geometry_canvas_size
        x = self.parent_geometry_spins["x"].value()
        y = self.parent_geometry_spins["y"].value()
        preserved_width = preserve_bbox[2] if preserve_bbox is not None else 1
        preserved_height = preserve_bbox[3] if preserve_bbox is not None else 1
        self.parent_geometry_spins["width"].setRange(
            1,
            max(1, page_width - x, preserved_width),
        )
        self.parent_geometry_spins["height"].setRange(
            1,
            max(1, page_height - y, preserved_height),
        )

    def _parent_geometry_values(self) -> tuple[int, int, int, int]:
        return (
            self.parent_geometry_spins["x"].value(),
            self.parent_geometry_spins["y"].value(),
            self.parent_geometry_spins["width"].value(),
            self.parent_geometry_spins["height"].value(),
        )

    @staticmethod
    def _replace_form(
        form: QtWidgets.QFormLayout,
        values: Mapping[str, object],
    ) -> None:
        while form.rowCount():
            form.removeRow(0)
        if not values:
            value = QtWidgets.QLabel("No effective values")
            value.setProperty("role", "secondary")
            form.addRow(value)
            return
        for key, raw in values.items():
            value = QtWidgets.QLabel(str(raw))
            value.setWordWrap(True)
            value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(key.replace("_", " ").title(), value)

    def set_cleanup_summary(
        self,
        *,
        label: str,
        detail: str,
        enabled: bool,
        action_text: str = "Open Manual Cleanup",
        accessible_description: str = "",
        revision_label: str = "",
    ) -> None:
        label = str(label or "Cleanup unavailable").strip()
        detail = str(detail or "Cleanup provenance is unavailable.").strip()
        action_text = str(action_text or "Open Manual Cleanup").strip()
        description = str(accessible_description or detail).strip()
        self.cleanup_state.setText(label)
        self.cleanup_detail.setText(detail)
        revision_label = str(revision_label or "").strip()
        self.cleanup_revision_readout.setText(revision_label or "—")
        visual_action = "Preview cleanup" if enabled else action_text
        self.cleanup_button.setText(visual_action)
        self.cleanup_button.setAccessibleName(visual_action)
        self.cleanup_button.setAccessibleDescription(description)
        self.cleanup_button.setToolTip(description)
        self.cleanup_button.setStatusTip(description)
        cleanup_enabled = bool(enabled and self._current_page_id)
        self._cleanup_facade_enabled = cleanup_enabled
        self.cleanup_button.setEnabled(cleanup_enabled)
        self.cleanup_commit_button.setEnabled(cleanup_enabled)
        for button in self.cleanup_tool_buttons.values():
            button.setEnabled(cleanup_enabled)
        for control in (
            self.cleanup_brush_size,
            self.cleanup_grow,
            self.cleanup_feather,
        ):
            control.setEnabled(cleanup_enabled)
        self._refresh_cleanup_visibility_controls()
        self._update_cleanup_facade_dirty()

    def set_history(
        self,
        entries: tuple[str, ...],
        *,
        record_ids: tuple[str, ...] | None = None,
        selected_record_id: str = "",
        scopes: tuple[str, ...] | None = None,
    ) -> None:
        if not isinstance(entries, tuple) or any(
            not isinstance(value, str) or not value for value in entries
        ):
            raise ValueError("history entries must contain non-empty strings")
        identities = record_ids if record_ids is not None else tuple("" for _ in entries)
        if (
            not isinstance(identities, tuple)
            or len(identities) != len(entries)
            or any(not isinstance(value, str) for value in identities)
        ):
            raise ValueError("history record IDs must align with history entries")
        selected_record_id = str(selected_record_id or "").strip()
        normalized_scopes = scopes if scopes is not None else tuple(
            "page" for _ in entries
        )
        if (
            not isinstance(normalized_scopes, tuple)
            or len(normalized_scopes) != len(entries)
            or any(value not in {"page", "parent"} for value in normalized_scopes)
        ):
            raise ValueError("history scopes must align as page or parent values")
        payload = (entries, identities, selected_record_id, normalized_scopes)
        if payload == self._history_payload:
            return
        self.activity_dock.set_history(entries)
        blocker = QtCore.QSignalBlocker(self.history_list)
        self.history_list.clear()
        selected_row = -1
        grouped = tuple(zip(entries, identities, normalized_scopes))
        for scope, heading in (("page", "Page commands"), ("parent", "Selected parent")):
            scoped = tuple(item for item in grouped if item[2] == scope)
            if not scoped:
                continue
            header = QtWidgets.QListWidgetItem(f"{heading}                                      {len(scoped)}")
            header.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            header_font = header.font()
            header_font.setBold(True)
            header.setFont(header_font)
            header.setSizeHint(QtCore.QSize(0, 28))
            self.history_list.addItem(header)
            for offset, (label, record_id, _scope) in enumerate(scoped):
                identity = str(record_id or "").strip()
                detail = "Your edit" if identity else "Immutable workflow record"
                age = (
                    f"Latest {scope} ledger entry"
                    if offset == 0
                    else f"{offset + 1} entries ago"
                )
                item = QtWidgets.QListWidgetItem(
                    f"{label}\n{detail}\n{age}"
                )
                item.setIcon(
                    hybrid_icon(
                        "status-editing" if identity else "status-ready",
                        self._icon_theme,
                    )
                )
                item.setData(QtCore.Qt.ItemDataRole.UserRole, identity)
                item.setToolTip(label)
                item.setSizeHint(QtCore.QSize(0, 62))
                self.history_list.addItem(item)
                if identity and identity == selected_record_id:
                    selected_row = self.history_list.count() - 1
        self.history_list.setCurrentRow(selected_row)
        del blocker
        self._history_payload = payload

    def set_history_editor_state(
        self,
        *,
        selected_record_id: str,
        action: str,
        action_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        identity = str(selected_record_id or "").strip()
        normalized_action = str(action or "").strip()
        if normalized_action not in {"", "revoke", "reapply"}:
            raise ValueError("unsupported edit-history action")
        enabled = bool(identity and action_enabled and not busy)
        self.history_revoke_button.setEnabled(
            bool(enabled and normalized_action == "revoke")
        )
        self.history_reapply_button.setEnabled(
            bool(enabled and normalized_action == "reapply")
        )
        self.history_list.setEnabled(not busy)
        self.history_status.setText(str(status_text))
        self.history_status.setProperty("tone", str(status_tone))
        self.history_status.setAccessibleDescription(str(status_text))
        self.history_status.setVisible(True)
        self.history_status.style().unpolish(self.history_status)
        self.history_status.style().polish(self.history_status)

    def set_render_override_reset_state(
        self,
        *,
        scope: str,
        field_group: str,
        summary_text: str,
        reset_enabled: bool,
        cancel_enabled: bool,
        busy: bool,
        status_text: str,
        status_tone: str = "muted",
    ) -> None:
        scope = str(scope or "").strip()
        field_group = str(field_group or "").strip()
        scope_index = self.render_override_reset_scope.findData(scope)
        field_index = self.render_override_reset_fields.findData(field_group)
        if scope_index < 0 or field_index < 0:
            raise ValueError("unsupported render-override reset selection")
        scope_blocker = QtCore.QSignalBlocker(self.render_override_reset_scope)
        fields_blocker = QtCore.QSignalBlocker(self.render_override_reset_fields)
        self.render_override_reset_scope.setCurrentIndex(scope_index)
        self.render_override_reset_fields.setCurrentIndex(field_index)
        del scope_blocker, fields_blocker
        self.render_override_reset_scope.setEnabled(not busy)
        self.render_override_reset_fields.setEnabled(not busy)
        self.render_override_reset_button.setEnabled(bool(reset_enabled and not busy))
        self.render_override_reset_cancel_button.setEnabled(
            bool(cancel_enabled and busy)
        )
        self.render_override_reset_summary.setText(str(summary_text))
        self.render_override_reset_summary.setAccessibleDescription(str(summary_text))
        self.render_override_reset_status.setText(str(status_text))
        self.render_override_reset_status.setAccessibleDescription(str(status_text))
        self.render_override_reset_status.setProperty("tone", str(status_tone))
        self.render_override_reset_status.style().unpolish(
            self.render_override_reset_status
        )
        self.render_override_reset_status.style().polish(
            self.render_override_reset_status
        )

    @QtCore.Slot(object, object)
    def _history_current_item_changed(
        self,
        current: QtWidgets.QListWidgetItem | None,
        _previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
        record_id = (
            str(current.data(QtCore.Qt.ItemDataRole.UserRole) or "").strip()
            if current is not None
            else ""
        )
        self.history_selection_changed.emit(record_id)

    def show_inspector_tab(self, tab: str) -> None:
        if tab not in self._inspector_index:
            raise ValueError(f"unsupported inspector tab: {tab!r}")
        self.inspector_tabs.setCurrentIndex(self._inspector_index[tab])

    def set_layout_mode(
        self,
        mode: LayoutMode,
        bounds: ActivityDockBounds,
        activity_height: int | None = None,
    ) -> None:
        if not isinstance(mode, LayoutMode):
            raise TypeError("mode must be LayoutMode")
        if not isinstance(bounds, ActivityDockBounds):
            raise TypeError("bounds must be ActivityDockBounds")
        self._layout_mode = mode
        self._activity_bounds = bounds
        high_scale = mode.font_scale_tier in {"large", "max"}
        responsive_form_wrap = (
            QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows
            if high_scale
            else QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.style_form.setRowWrapPolicy(responsive_form_wrap)
        self.layout_form.setRowWrapPolicy(responsive_form_wrap)
        self.add_user_parent_form.setRowWrapPolicy(responsive_form_wrap)
        # The accepted 1280/1440 editor retains a readable thumbnail rail.
        # Collapse it only for a genuinely narrow viewport or accessibility
        # reflow, not merely because the width tier is compact.
        compact_width = mode.width_tier == "narrow"
        expanded_accessible_panes = high_scale and not compact_width
        self.inspector_identity_layout.setDirection(
            QtWidgets.QBoxLayout.Direction.TopToBottom
            if high_scale
            else QtWidgets.QBoxLayout.Direction.LeftToRight
        )
        self.inspector_identity_layout.setAlignment(
            self.inspector_parent,
            (
                QtCore.Qt.AlignmentFlag.AlignLeft
                if high_scale
                else QtCore.Qt.AlignmentFlag.AlignRight
            ),
        )
        authority_reflow = compact_width or mode.font_scale >= 125
        self.target_authority_comparison_layout.setDirection(
            QtWidgets.QBoxLayout.Direction.TopToBottom
            if authority_reflow
            else QtWidgets.QBoxLayout.Direction.LeftToRight
        )
        if self._page_rail_collapsed:
            self.page_rail.setMinimumWidth(84)
            self.page_rail.setMaximumWidth(84)
            self.page_search_band.setVisible(False)
        elif expanded_accessible_panes:
            # A non-narrow high-scale workspace has enough width to preserve
            # complete page-stage and parent-stage copy.  Give both side panes
            # room before the center canvas takes its stretch allocation.
            self.page_rail.setMinimumWidth(320)
            self.page_rail.setMaximumWidth(360)
            self.inspector.setMinimumWidth(720)
            self.inspector.setMaximumWidth(880)
            self.page_search_band.setVisible(False)
        elif high_scale or compact_width:
            # Keep page identity readable at enlarged scale.  The list itself
            # remains scrollable, but the rail must not collapse into a strip
            # that exposes only thumbnails and clipped status text.
            self.page_rail.setMinimumWidth(84)
            self.page_rail.setMaximumWidth(184)
            self.inspector.setMinimumWidth(300)
            self.inspector.setMaximumWidth(520)
            self.page_search_band.setVisible(False)
        else:
            self.page_rail.setMinimumWidth(84)
            self.page_rail.setMaximumWidth(280)
            self.inspector.setMinimumWidth(300)
            self.inspector.setMaximumWidth(520)
            self.page_search_band.setVisible(True)
        self.activity_dock.set_layout_mode(mode)
        self.activity_dock.set_height_bounds(bounds)
        requested = bounds.preferred if activity_height is None else int(activity_height)
        self._activity_requested_height = min(
            bounds.max,
            max(bounds.min, requested),
        )
        sizes = self.vertical_splitter.sizes()
        if len(sizes) == 2:
            if self.activity_dock.expanded:
                preferred = (
                    self._activity_requested_height
                    + self.activity_dock.collapsed_height
                )
            else:
                preferred = self.activity_dock.collapsed_height
            self._set_activity_splitter_height(preferred)
            QtCore.QTimer.singleShot(
                0,
                lambda value=preferred: self._set_activity_splitter_height(value),
            )

    @QtCore.Slot(bool)
    def _activity_expansion_changed(self, expanded: bool) -> None:
        preferred = (
            self._activity_requested_height
            + self.activity_dock.collapsed_height
            if expanded
            else self.activity_dock.collapsed_height
        )
        self._set_activity_splitter_height(preferred)
        self.layout_changed.emit()

    def _set_activity_splitter_height(self, preferred: int) -> None:
        sizes = self.vertical_splitter.sizes()
        if len(sizes) != 2:
            return
        total = sum(sizes)
        self.vertical_splitter.setSizes((max(1, total - int(preferred)), int(preferred)))

    def canvas_fit_page(self) -> None:
        self.canvas.fit_page()

    def _set_page_rail_mode(self, *, compact: bool) -> None:
        self.page_grid_button.setChecked(bool(compact))
        self.page_list_button.setChecked(not bool(compact))
        self.page_list.setItemDelegate(
            PageRailDelegate(
                compact=bool(compact),
                collapsed=self._page_rail_collapsed,
                parent=self.page_list,
            )
        )
        self.page_list.doItemsLayout()

    def _toggle_page_rail_collapsed(self) -> None:
        self._page_rail_collapsed = not self._page_rail_collapsed
        collapsed = self._page_rail_collapsed
        self.project_label.setVisible(not collapsed)
        self.project_name.setVisible(not collapsed)
        self.page_count.setVisible(not collapsed)
        self.page_previous_button.setVisible(not collapsed)
        self.page_next_button.setVisible(not collapsed)
        self.page_list_button.setVisible(not collapsed)
        self.page_rail_toggle.setIcon(
            hybrid_icon(
                "caret-right" if collapsed else "caret-down",
                self._icon_theme,
            )
        )
        action = "Expand" if collapsed else "Collapse"
        self.page_rail_toggle.setAccessibleName(f"{action} page navigator")
        self.page_rail_toggle.setToolTip(f"{action} page navigator")
        self._set_page_rail_mode(compact=self.page_grid_button.isChecked())
        if self._layout_mode is not None:
            self.set_layout_mode(
                self._layout_mode,
                self._activity_bounds,
                self._activity_requested_height,
            )
        else:
            self.page_rail.setMinimumWidth(84)
            self.page_rail.setMaximumWidth(84 if collapsed else 280)
            self.page_search_band.setVisible(not collapsed)
        self.layout_changed.emit()

    def _activate_page_offset(self, offset: int) -> None:
        model = self.page_list.model()
        if model is None or model.rowCount() <= 0:
            return
        current = self.page_list.currentIndex()
        row = current.row() if current.isValid() else 0
        target_row = max(0, min(model.rowCount() - 1, row + int(offset)))
        target = model.index(target_row, 0)
        if target.isValid() and target_row != row:
            self.page_list.setCurrentIndex(target)
            self._activate_page(target)
        self._update_page_navigation_buttons()

    def _update_page_navigation_buttons(self) -> None:
        model = self.page_list.model()
        rows = model.rowCount() if model is not None else 0
        current = self.page_list.currentIndex()
        row = current.row() if current.isValid() else -1
        self.page_previous_button.setEnabled(rows > 0 and row > 0)
        self.page_next_button.setEnabled(rows > 0 and 0 <= row < rows - 1)

    def _set_canvas_tool(self, tool: str) -> None:
        if tool not in {"select", "pan"}:
            raise ValueError("canvas tool must be select or pan")
        select = tool == "select"
        self.select_tool_button.setChecked(select)
        self.pan_tool_button.setChecked(not select)
        self.canvas.setDragMode(
            QtWidgets.QGraphicsView.DragMode.NoDrag
            if select
            else QtWidgets.QGraphicsView.DragMode.ScrollHandDrag
        )

    def _adjust_canvas_zoom(self, delta: int) -> None:
        self.zoom_slider.setValue(
            max(
                self.zoom_slider.minimum(),
                min(self.zoom_slider.maximum(), self.zoom_slider.value() + int(delta)),
            )
        )

    def _toggle_canvas_focus(self) -> None:
        self._canvas_focus_active = bool(self.canvas_focus_button.isChecked())
        self.page_rail.setVisible(not self._canvas_focus_active)
        self.inspector.setVisible(not self._canvas_focus_active)
        self.activity_dock.setVisible(not self._canvas_focus_active)
        self.canvas_focus_button.setAccessibleName(
            "Exit canvas focus" if self._canvas_focus_active else "Enter canvas focus"
        )

    def exit_canvas_focus(self) -> None:
        """Leave canvas-only focus mode without altering the current tool."""

        if not self._canvas_focus_active:
            return
        self.canvas_focus_button.setChecked(False)
        self._toggle_canvas_focus()
        self.canvas.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)

    def _toggle_inspector_visibility(self) -> None:
        self._inspector_hidden_by_toolbar = bool(
            self.inspector_toggle_button.isChecked()
        )
        self.inspector.setVisible(not self._inspector_hidden_by_toolbar)
        self.inspector_toggle_button.setAccessibleName(
            "Show inspector" if self._inspector_hidden_by_toolbar else "Hide inspector"
        )

    def _set_canvas_zoom(self, value: int) -> None:
        if self.zoom_slider.signalsBlocked():
            return
        self.canvas.set_zoom_percent(value)

    def _sync_zoom(self, value: int) -> None:
        self.zoom_label.setText(f"{value}%")
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(max(self.zoom_slider.minimum(), min(self.zoom_slider.maximum(), value)))
        self.zoom_slider.blockSignals(False)

    def _sync_mode(self, mode: str) -> None:
        button = self.mode_buttons.get(mode)
        if button is not None:
            button.setChecked(True)

    def _activate_page(self, index: QtCore.QModelIndex) -> None:
        if self._page_id_role is None:
            return
        page_id = str(index.data(self._page_id_role) or "").strip()
        if page_id and page_id != self._current_page_id:
            self.page_selected.emit(page_id)

    def _activate_parent(self, index: int) -> None:
        if index < 0 or self._parent_id_role is None:
            return
        parent_id = str(self.parent_list.itemData(index, self._parent_id_role) or "").strip()
        if parent_id:
            self.parent_selected.emit(parent_id)

    def _activate_parent_offset(self, offset: int) -> None:
        target = self.parent_list.currentIndex() + int(offset)
        if 0 <= target < self.parent_list.count():
            self.parent_list.setCurrentIndex(target)

    def _sync_parent_segment(self) -> None:
        if not hasattr(self, "parent_segment_label"):
            return
        count = self.parent_list.count()
        current = self.parent_list.currentIndex()
        if count <= 0 or current < 0:
            self.parent_segment_label.setText("Text segment — / —")
            self.previous_parent_button.setEnabled(False)
            self.next_parent_button.setEnabled(False)
            return
        self.parent_segment_label.setText(f"Text segment {current + 1} / {count}")
        self.previous_parent_button.setEnabled(current > 0)
        self.next_parent_button.setEnabled(current + 1 < count)
