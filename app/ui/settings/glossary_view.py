"""Project glossary presentation for the Settings route."""
from __future__ import annotations

from collections.abc import Mapping

from PySide6 import QtCore, QtWidgets

from app.ui.design_system.icons import hybrid_icon
from app.ui.ui_contract import LayoutMode
from app.ui.viewmodels.glossary_model import (
    GlossaryEditorModel,
    GlossaryEditorPhase,
    GlossaryWorkerBusyState,
)


class _GlossaryRowDelegate(QtWidgets.QStyledItemDelegate):
    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtCore.QSize:
        size = super().sizeHint(option, index)
        size.setHeight(max(size.height(), option.fontMetrics.height() * 3))
        return size


class GlossarySettingsPage(QtWidgets.QWidget):
    """Accessible project-scoped glossary editor with explicit commands."""

    command_requested = QtCore.Signal(object)
    cancel_worker_requested = QtCore.Signal()
    open_stale_page_requested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("glossarySettingsPage")
        self.setAccessibleName("Project glossary")
        self._model: GlossaryEditorModel | None = None
        self._page_labels: dict[str, str] = {}
        self._refreshing = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)
        heading = QtWidgets.QLabel("Glossary")
        heading.setObjectName("glossaryHeading")
        heading.setProperty("role", "title")
        self.detail = QtWidgets.QLabel(
            "Validated terms are project data with provenance; completed pages "
            "are never silently retranslated."
        )
        self.detail.setWordWrap(True)
        self.detail.setProperty("role", "secondary")
        root.addWidget(heading)
        root.addWidget(self.detail)

        self.status = QtWidgets.QLabel("Open a project to manage its glossary.")
        self.status.setObjectName("glossaryStatus")
        self.status.setProperty("role", "status-banner")
        self.status.setProperty("tone", "muted")
        self.status.setWordWrap(True)
        self.status.setAccessibleName("Glossary status")
        root.addWidget(self.status)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setObjectName("glossarySplitter")
        self.splitter.setChildrenCollapsible(False)
        root.addWidget(self.splitter, 1)

        browser = QtWidgets.QFrame()
        browser.setProperty("role", "panel")
        browser_layout = QtWidgets.QVBoxLayout(browser)
        browser_layout.setContentsMargins(12, 12, 12, 12)
        browser_layout.setSpacing(8)
        browser_title = QtWidgets.QLabel("Terms")
        browser_title.setProperty("role", "section")
        browser_layout.addWidget(browser_title)
        self.search = QtWidgets.QLineEdit()
        self.search.setObjectName("glossarySearch")
        self.search.setPlaceholderText("Search source, target, or notes")
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName("Search project glossary")
        self.search.textChanged.connect(self._refresh_entries)
        browser_layout.addWidget(self.search)
        self.entries = QtWidgets.QTreeWidget()
        self.entries.setObjectName("glossaryEntries")
        self.entries.setAccessibleName("Project glossary entries")
        self.entries.setHeaderLabels(("Source", "Target", "Notes", "Priority", ""))
        self.entries.setRootIsDecorated(False)
        self.entries.setAlternatingRowColors(True)
        self.entries.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.entries.currentItemChanged.connect(self._select_entry)
        browser_layout.addWidget(self.entries, 1)
        browser_actions = QtWidgets.QHBoxLayout()
        self.new_button = self._button("New", "New glossary entry")
        self.import_button = self._button("Import", "Import glossary JSON or CSV")
        self.export_button = self._button("Export", "Export project glossary")
        self.new_button.setText("Add term")
        self.new_button.setIcon(hybrid_icon("new"))
        self.import_button.setIcon(hybrid_icon("open"))
        self.export_button.setIcon(hybrid_icon("arrow-right"))
        self.new_button.clicked.connect(self._begin_new)
        self.import_button.clicked.connect(self._begin_import)
        self.export_button.clicked.connect(self._begin_export)
        for button in (self.new_button, self.import_button, self.export_button):
            browser_actions.addWidget(button)
        browser_layout.addLayout(browser_actions)
        self.splitter.addWidget(browser)

        editor_scroll = QtWidgets.QScrollArea()
        editor_scroll.setObjectName("glossaryEditorScroll")
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        editor = QtWidgets.QWidget()
        editor_layout = QtWidgets.QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 8, 0)
        editor_layout.setSpacing(10)

        self.entry_card = QtWidgets.QFrame()
        self.entry_card.setProperty("role", "panel")
        entry_layout = QtWidgets.QVBoxLayout(self.entry_card)
        entry_layout.setContentsMargins(12, 12, 12, 12)
        entry_title = QtWidgets.QLabel("Entry")
        entry_title.setProperty("role", "section")
        entry_layout.addWidget(entry_title)
        form = QtWidgets.QFormLayout()
        self.source = QtWidgets.QLineEdit()
        self.source.setObjectName("glossarySource")
        self.source.setAccessibleName("Glossary source")
        self.target = QtWidgets.QLineEdit()
        self.target.setObjectName("glossaryTarget")
        self.target.setAccessibleName("Glossary target")
        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setObjectName("glossaryNotes")
        self.notes.setAccessibleName("Glossary notes")
        self.notes.setPlaceholderText("Optional translator note")
        self.notes.setMaximumHeight(88)
        self.aliases = QtWidgets.QPlainTextEdit()
        self.aliases.setObjectName("glossaryAliases")
        self.aliases.setAccessibleName("Glossary aliases")
        self.aliases.setAccessibleDescription("Enter one source alias per line.")
        self.aliases.setPlaceholderText("One source alias per line")
        self.aliases.setMaximumHeight(100)
        self.priority = QtWidgets.QComboBox()
        self.priority.setObjectName("glossaryPriority")
        self.priority.setAccessibleName("Glossary priority")
        self.priority.addItem("Normal", "soft")
        self.priority.addItem("High", "hard")
        form.addRow("Source", self.source)
        form.addRow("Target", self.target)
        form.addRow("Notes", self.notes)
        form.addRow("Aliases", self.aliases)
        form.addRow("Priority", self.priority)
        entry_layout.addLayout(form)
        self.validation = QtWidgets.QLabel("Select or create a glossary entry.")
        self.validation.setObjectName("glossaryValidation")
        self.validation.setProperty("role", "secondary")
        self.validation.setWordWrap(True)
        self.validation.setAccessibleName("Glossary entry validation")
        entry_layout.addWidget(self.validation)
        entry_actions = QtWidgets.QHBoxLayout()
        self.save_button = self._button("Save Entry", "Save glossary entry", primary=True)
        self.cancel_button = self._button("Cancel Draft", "Cancel glossary draft")
        self.remove_button = self._button("Remove", "Remove glossary entry")
        self.worker_cancel_button = self._button(
            "Cancel Action", "Cancel glossary action before persistence"
        )
        self.save_button.clicked.connect(self._save)
        self.cancel_button.clicked.connect(self._cancel_draft)
        self.remove_button.clicked.connect(self._remove)
        self.worker_cancel_button.clicked.connect(self.cancel_worker_requested)
        for button in (
            self.save_button,
            self.cancel_button,
            self.remove_button,
            self.worker_cancel_button,
        ):
            entry_actions.addWidget(button)
        entry_layout.addLayout(entry_actions)
        editor_layout.addWidget(self.entry_card)

        self.stale_card = QtWidgets.QFrame()
        self.stale_card.setProperty("role", "panel")
        stale_layout = QtWidgets.QVBoxLayout(self.stale_card)
        stale_layout.setContentsMargins(12, 12, 12, 12)
        stale_title = QtWidgets.QLabel("Pages to review")
        stale_title.setProperty("role", "section")
        stale_layout.addWidget(stale_title)
        self.stale_copy = QtWidgets.QLabel(
            "Glossary changes can make completed translations stale. Select a page "
            "and open it; Retranslate remains an explicit Page Editor command."
        )
        self.stale_copy.setWordWrap(True)
        self.stale_copy.setProperty("role", "secondary")
        stale_layout.addWidget(self.stale_copy)
        self.stale_pages = QtWidgets.QTreeWidget()
        self.stale_pages.setObjectName("glossaryStalePages")
        self.stale_pages.setAccessibleName("Pages with potentially stale translations")
        self.stale_pages.setHeaderLabels(("Review", "Page"))
        self.stale_pages.setRootIsDecorated(False)
        self.stale_pages.setMaximumHeight(150)
        self.stale_pages.itemChanged.connect(self._stale_page_changed)
        stale_layout.addWidget(self.stale_pages)
        self.open_page_button = self._button(
            "Open Selected Page", "Open selected stale page in Page Editor"
        )
        self.open_page_button.clicked.connect(self._open_stale_page)
        stale_layout.addWidget(self.open_page_button)
        editor_layout.addWidget(self.stale_card)

        self.history_card = QtWidgets.QFrame()
        self.history_card.setProperty("role", "panel")
        history_layout = QtWidgets.QVBoxLayout(self.history_card)
        history_layout.setContentsMargins(12, 12, 12, 12)
        history_title = QtWidgets.QLabel("Project glossary History")
        history_title.setProperty("role", "section")
        history_layout.addWidget(history_title)
        self.history = QtWidgets.QTreeWidget()
        self.history.setObjectName("glossaryHistory")
        self.history.setAccessibleName("Project glossary History")
        self.history.setHeaderLabels(("Action", "State"))
        self.history.setRootIsDecorated(False)
        self.history.setMaximumHeight(180)
        self.history.currentItemChanged.connect(self._refresh_actions)
        history_layout.addWidget(self.history)
        self.history_button = self._button(
            "Select a History entry", "Revoke or reapply selected glossary edit"
        )
        self.history_button.clicked.connect(self._run_history)
        history_layout.addWidget(self.history_button)
        editor_layout.addWidget(self.history_card)
        editor_layout.addStretch(1)
        self.editor_scroll = editor_scroll
        editor_scroll.setWidget(editor)
        self.splitter.addWidget(editor_scroll)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 5)

        # Hybrid Pro presents the glossary as a compact table first.  The
        # existing typed entry, stale-page, and History editors remain the
        # command owners, but open only after an explicit user action.
        root.removeWidget(heading)
        root.removeWidget(self.detail)
        root.removeWidget(self.status)
        root.removeWidget(self.splitter)
        browser_layout.removeWidget(self.search)
        browser_layout.removeWidget(self.entries)
        for button in (self.new_button, self.import_button, self.export_button):
            browser_actions.removeWidget(button)
        browser.hide()

        self.glossary_header = QtWidgets.QFrame()
        self.glossary_header.setObjectName("glossaryHeader")
        self.glossary_header.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        header_layout = QtWidgets.QHBoxLayout(self.glossary_header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)
        header_copy = QtWidgets.QVBoxLayout()
        header_copy.setContentsMargins(0, 0, 0, 0)
        header_copy.setSpacing(2)
        eyebrow = QtWidgets.QLabel("PROJECT LANGUAGE")
        eyebrow.setProperty("role", "eyebrow")
        header_copy.addWidget(eyebrow)
        header_copy.addWidget(heading)
        header_copy.addWidget(self.detail)
        header_layout.addLayout(header_copy, 1)
        self.export_button.setProperty("variant", "quiet")
        self.import_button.setProperty("variant", "quiet")
        self.new_button.setProperty("variant", "secondary")
        header_actions = QtWidgets.QVBoxLayout()
        header_actions.setContentsMargins(0, 0, 0, 0)
        header_actions.setSpacing(4)
        for button in (self.export_button, self.import_button, self.new_button):
            button.setMinimumWidth(102)
            header_actions.addWidget(button)
        header_layout.addLayout(header_actions)
        root.addWidget(self.glossary_header)

        self.glossary_body = QtWidgets.QWidget()
        self.glossary_body.setObjectName("glossaryBody")
        body_layout = QtWidgets.QVBoxLayout(self.glossary_body)
        body_layout.setContentsMargins(22, 20, 22, 22)
        body_layout.setSpacing(12)
        root.addWidget(self.glossary_body, 1)

        self.glossary_toolbar = QtWidgets.QFrame()
        self.glossary_toolbar.setObjectName("glossaryToolbar")
        self.glossary_toolbar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        toolbar_layout = QtWidgets.QHBoxLayout(self.glossary_toolbar)
        toolbar_layout.setContentsMargins(0, 8, 0, 4)
        toolbar_layout.setSpacing(8)
        self.search.setMinimumWidth(320)
        self.search.setMaximumWidth(430)
        toolbar_layout.addWidget(self.search)
        toolbar_layout.addStretch(1)
        self.stale_summary = QtWidgets.QLabel("Translations are current")
        self.stale_summary.setObjectName("glossaryStaleSummary")
        self.stale_summary.setProperty("role", "status-pill")
        self.stale_summary.setProperty("tone", "ready")
        self.stale_summary.setAccessibleName("Glossary translation freshness")
        toolbar_layout.addWidget(self.stale_summary)
        self.review_pages_button = self._button(
            "Select for retranslation",
            "Select completed pages for explicit retranslation review",
        )
        self.review_pages_button.setProperty("variant", "quiet")
        self.review_pages_button.clicked.connect(self._show_stale_review)
        toolbar_layout.addWidget(self.review_pages_button)
        self.manage_history_button = self._button(
            "History",
            "Open project glossary History",
        )
        self.manage_history_button.setProperty("variant", "quiet")
        self.manage_history_button.clicked.connect(self._show_history)
        toolbar_layout.addWidget(self.manage_history_button)
        body_layout.addWidget(self.glossary_toolbar)

        self.entries.setObjectName("glossaryTable")
        self.entries.setItemDelegate(_GlossaryRowDelegate(self.entries))
        self.entries.setMinimumHeight(160)
        header = self.entries.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        body_layout.addWidget(self.entries, 1)

        self.alias_warning = QtWidgets.QFrame()
        self.alias_warning.setObjectName("glossaryAliasWarning")
        self.alias_warning.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        self.alias_warning.setProperty("role", "status-banner")
        self.alias_warning.setProperty("tone", "warning")
        alias_layout = QtWidgets.QHBoxLayout(self.alias_warning)
        alias_layout.setContentsMargins(10, 8, 10, 8)
        alias_layout.setSpacing(8)
        alias_mark = QtWidgets.QLabel()
        alias_mark.setPixmap(hybrid_icon("warning").pixmap(17, 17))
        alias_layout.addWidget(alias_mark, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        self.alias_warning_copy = QtWidgets.QLabel("Alias review")
        self.alias_warning_copy.setWordWrap(True)
        self.alias_warning_copy.setAccessibleName("Glossary alias review")
        alias_layout.addWidget(self.alias_warning_copy, 1)
        body_layout.addWidget(self.alias_warning)

        body_layout.addWidget(self.status)
        body_layout.addWidget(self.splitter, 2)
        body_layout.addStretch(1)
        self._compact_layout = body_layout
        self._compact_stretch_index = body_layout.count() - 1
        self._detail_panel = ""
        self._advanced = False
        self.splitter.hide()
        self.status.hide()
        self.alias_warning.hide()
        self.export_button.hide()
        self.manage_history_button.hide()

        self._connect_draft_controls()
        self.clear_model()

    @staticmethod
    def _button(text: str, accessible_name: str, *, primary: bool = False) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setProperty("role", "command")
        button.setProperty("variant", "primary" if primary else "secondary")
        button.setAccessibleName(accessible_name)
        return button

    def _connect_draft_controls(self) -> None:
        self.source.textChanged.connect(self._draft_changed)
        self.target.textChanged.connect(self._draft_changed)
        self.notes.textChanged.connect(self._draft_changed)
        self.aliases.textChanged.connect(self._draft_changed)
        self.priority.currentIndexChanged.connect(self._draft_changed)
        QtWidgets.QWidget.setTabOrder(self.search, self.entries)
        QtWidgets.QWidget.setTabOrder(self.entries, self.new_button)
        QtWidgets.QWidget.setTabOrder(self.new_button, self.import_button)
        QtWidgets.QWidget.setTabOrder(self.import_button, self.export_button)
        QtWidgets.QWidget.setTabOrder(self.export_button, self.source)
        QtWidgets.QWidget.setTabOrder(self.source, self.target)
        QtWidgets.QWidget.setTabOrder(self.target, self.notes)
        QtWidgets.QWidget.setTabOrder(self.notes, self.aliases)
        QtWidgets.QWidget.setTabOrder(self.aliases, self.priority)
        QtWidgets.QWidget.setTabOrder(self.priority, self.save_button)
        QtWidgets.QWidget.setTabOrder(self.save_button, self.cancel_button)
        QtWidgets.QWidget.setTabOrder(self.cancel_button, self.remove_button)
        QtWidgets.QWidget.setTabOrder(self.remove_button, self.stale_pages)
        QtWidgets.QWidget.setTabOrder(self.stale_pages, self.open_page_button)
        QtWidgets.QWidget.setTabOrder(self.open_page_button, self.history)
        QtWidgets.QWidget.setTabOrder(self.history, self.history_button)

    @property
    def model(self) -> GlossaryEditorModel | None:
        return self._model

    def bind_model(
        self,
        model: GlossaryEditorModel,
        *,
        page_labels: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(model, GlossaryEditorModel):
            raise TypeError("model must be GlossaryEditorModel")
        self._model = model
        self._page_labels = {
            str(page_id): str(label)
            for page_id, label in dict(page_labels or {}).items()
        }
        self._detail_panel = ""
        self.refresh()

    def clear_model(self, reason: str = "Open a project to manage its glossary.") -> None:
        self._model = None
        self._page_labels.clear()
        self._refreshing = True
        try:
            self.entries.clear()
            self.history.clear()
            self.stale_pages.clear()
            self.source.clear()
            self.target.clear()
            self.notes.clear()
            self.aliases.clear()
            self.priority.setCurrentIndex(0)
        finally:
            self._refreshing = False
        self.status.setText(reason)
        self.status.setProperty("tone", "muted")
        self.status.hide()
        self.validation.setText("Select or create a glossary entry.")
        self._detail_panel = ""
        self._show_detail_panel("")
        self._refresh_entries()
        self._refresh_actions()

    def set_advanced(self, advanced: bool) -> None:
        self._advanced = bool(advanced)
        self.export_button.setVisible(self._advanced)
        self.manage_history_button.setVisible(self._advanced)

    def _show_detail_panel(self, panel: str) -> None:
        if panel not in {"", "entry", "stale", "history"}:
            raise ValueError(f"unsupported glossary detail panel: {panel}")
        self._detail_panel = panel
        self.entry_card.setVisible(panel == "entry")
        self.stale_card.setVisible(panel == "stale")
        self.history_card.setVisible(panel == "history")
        self.splitter.setVisible(bool(panel))
        root = self._compact_layout
        root.setStretch(root.indexOf(self.entries), 0)
        root.setStretch(root.indexOf(self.splitter), 2 if panel else 0)
        root.setStretch(self._compact_stretch_index, 0 if panel else 1)
        root.invalidate()
        if panel:
            self.splitter.setSizes((0, 1000))

    def _show_stale_review(self) -> None:
        if self._model is None:
            return
        self._show_detail_panel("stale")
        self.stale_pages.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _show_history(self) -> None:
        if self._model is None:
            return
        self._show_detail_panel("history")
        self.history.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def set_layout_mode(self, mode: LayoutMode) -> None:
        if not isinstance(mode, LayoutMode):
            raise TypeError("mode must be LayoutMode")
        reflow = bool(mode.accessible_reflow or mode.width_tier == "narrow")
        self.splitter.setOrientation(
            QtCore.Qt.Orientation.Vertical if reflow else QtCore.Qt.Orientation.Horizontal
        )
        if reflow:
            self.splitter.setSizes((360, 720))

    def accept_busy(self, value: GlossaryWorkerBusyState) -> None:
        if not isinstance(value, GlossaryWorkerBusyState):
            raise TypeError("busy state must be GlossaryWorkerBusyState")
        self.status.setText(value.message)
        self.status.setProperty("tone", "editing" if value.busy else "muted")
        self.worker_cancel_button.setEnabled(value.cancellation_enabled)
        # Busy-stage notifications do not change the glossary selection or
        # draft. Rebuilding every tree and text editor from this cross-thread
        # signal is both unnecessary and can stall Qt text layout under a
        # long-lived, heavily themed application session.
        self._refresh_actions()

    def refresh(self, *, preserve_status: bool = False) -> None:
        model = self._model
        if model is None:
            self.clear_model()
            return
        state = model.state
        selected_entry_id = state.selected_entry_id
        selected_history_id = self._selected_data(self.history)
        selected_stale_id = self._selected_data(self.stale_pages)
        self._refreshing = True
        try:
            self._refresh_entries()
            self._populate_draft()
            self._populate_stale_pages(selected_stale_id)
            self._populate_history(selected_history_id)
        finally:
            self._refreshing = False
        if not preserve_status:
            self.status.setText(state.message)
            self.status.setProperty(
                "tone",
                "error"
                if state.phase in {GlossaryEditorPhase.FAILED, GlossaryEditorPhase.STALE}
                else "warning"
                if state.dirty
                else "ready",
            )
            self.status.setVisible(
                bool(state.busy or state.dirty)
                or state.phase in {GlossaryEditorPhase.FAILED, GlossaryEditorPhase.STALE}
            )
        if selected_entry_id:
            match = next(
                (
                    self.entries.topLevelItem(index)
                    for index in range(self.entries.topLevelItemCount())
                    if str(
                        self.entries.topLevelItem(index).data(
                            0,
                            QtCore.Qt.ItemDataRole.UserRole,
                        )
                        or ""
                    )
                    == selected_entry_id
                ),
                None,
            )
            if match is not None:
                self.entries.setCurrentItem(match)
        self._refresh_actions()

    def _refresh_entries(self, *_args: object) -> None:
        model = self._model
        selected_id = self._selected_data(self.entries)
        self.entries.blockSignals(True)
        self.entries.clear()
        if model is not None:
            for entry in model.filtered_entries(self.search.text()):
                item = QtWidgets.QTreeWidgetItem(
                    (
                        entry.source,
                        entry.target,
                        entry.notes,
                        "High" if entry.priority == "hard" else "Normal",
                        "Edit",
                    )
                )
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, entry.entry_id)
                item.setToolTip(0, entry.source)
                item.setToolTip(1, entry.target)
                self.entries.addTopLevelItem(item)
                if entry.entry_id == selected_id:
                    self.entries.setCurrentItem(item)
        self.entries.resizeColumnToContents(3)
        self.entries.resizeColumnToContents(4)
        self.entries.doItemsLayout()
        item_count = self.entries.topLevelItemCount()
        fallback_row_height = max(42, self.entries.fontMetrics().height() * 3)
        measured_rows = [
            max(fallback_row_height, self.entries.sizeHintForRow(index))
            for index in range(min(6, item_count))
        ]
        actual_rows_height = sum(measured_rows)
        visible_rows = max(3, min(6, item_count))
        measured_rows.extend(
            fallback_row_height for _ in range(visible_rows - len(measured_rows))
        )
        header_height = max(30, self.entries.header().sizeHint().height())
        chrome_height = self.entries.frameWidth() * 2 + 16
        measured_height = header_height + sum(measured_rows) + chrome_height
        target_height = max(160, min(200, measured_height))
        self.entries.setMinimumHeight(target_height)
        self.entries.setMaximumHeight(target_height)
        self.entries.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            if (
                item_count <= visible_rows
                and header_height + actual_rows_height + chrome_height
                <= target_height
            )
            else QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.entries.blockSignals(False)

    def _select_entry(
        self,
        current: QtWidgets.QTreeWidgetItem | None,
        _previous: QtWidgets.QTreeWidgetItem | None,
    ) -> None:
        if self._refreshing or self._model is None or current is None:
            return
        try:
            self._model.select_entry(str(current.data(0, QtCore.Qt.ItemDataRole.UserRole) or ""))
        except (KeyError, RuntimeError) as exc:
            self.status.setText(str(exc))
            self.status.setProperty("tone", "warning")
        self.refresh(preserve_status=True)
        self._show_detail_panel("entry")

    def _populate_draft(self) -> None:
        state = self._model.state if self._model is not None else None
        draft = state.draft if state is not None else None
        self.source.setText(draft.source if draft is not None else "")
        self.target.setText(draft.target if draft is not None else "")
        self.notes.setPlainText(draft.notes if draft is not None else "")
        self.aliases.setPlainText("\n".join(draft.aliases) if draft is not None else "")
        self.priority.setCurrentIndex(
            max(0, self.priority.findData(draft.priority if draft is not None else "soft"))
        )
        self.validation.setText(
            self._model.draft_problem() if self._model is not None else "Select or create a glossary entry."
        )

    def _draft_changed(self, *_args: object) -> None:
        if self._refreshing or self._model is None or self._model.state.draft is None:
            return
        aliases = tuple(
            line.strip()
            for line in self.aliases.toPlainText().splitlines()
            if line.strip()
        )
        try:
            self._model.update_draft(
                source=self.source.text(),
                target=self.target.text(),
                notes=self.notes.toPlainText(),
                aliases=aliases,
                priority=str(self.priority.currentData() or "soft"),
            )
        except RuntimeError:
            return
        self.validation.setText(self._model.draft_problem())
        self._refresh_actions()

    def _begin_new(self) -> None:
        if self._model is None:
            return
        try:
            self._model.begin_new()
        except RuntimeError as exc:
            self.status.setText(str(exc))
            return
        self._show_detail_panel("entry")
        self.refresh()
        self.source.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _save(self) -> None:
        if self._model is None:
            return
        try:
            command = self._model.begin_save()
        except (RuntimeError, ValueError) as exc:
            self.validation.setText(str(exc))
            self.validation.setProperty("tone", "error")
            self._refresh_actions()
            return
        self.refresh()
        self.command_requested.emit(command)

    def _cancel_draft(self) -> None:
        if self._model is None:
            return
        try:
            self._model.cancel_draft()
        except RuntimeError as exc:
            self.status.setText(str(exc))
            return
        self._show_detail_panel("")
        self.refresh()

    def _remove(self) -> None:
        if self._model is None:
            return
        try:
            command = self._model.begin_remove()
        except RuntimeError as exc:
            self.status.setText(str(exc))
            return
        self.refresh()
        self.command_requested.emit(command)

    def _begin_import(self) -> None:
        if self._model is None:
            return
        path, _kind = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import project glossary",
            "",
            "Glossary files (*.json *.csv)",
        )
        if path:
            try:
                command = self._model.begin_import(path)
            except RuntimeError as exc:
                self.status.setText(str(exc))
                return
            self.refresh()
            self.command_requested.emit(command)

    def _begin_export(self) -> None:
        if self._model is None:
            return
        path, _kind = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export project glossary",
            "project-glossary.json",
            "JSON (*.json);;CSV (*.csv)",
        )
        if path:
            try:
                command = self._model.begin_export(path)
            except RuntimeError as exc:
                self.status.setText(str(exc))
                return
            self.refresh()
            self.command_requested.emit(command)

    def _populate_stale_pages(self, selected_id: str) -> None:
        self.stale_pages.blockSignals(True)
        self.stale_pages.clear()
        if self._model is not None:
            selected = set(self._model.state.selected_stale_page_ids)
            for page_id in self._model.state.selection.stale_page_ids:
                label = self._page_labels.get(page_id, page_id)
                item = QtWidgets.QTreeWidgetItem(("", label))
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, page_id)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    0,
                    QtCore.Qt.CheckState.Checked
                    if page_id in selected
                    else QtCore.Qt.CheckState.Unchecked,
                )
                self.stale_pages.addTopLevelItem(item)
                if page_id == selected_id:
                    self.stale_pages.setCurrentItem(item)
        self.stale_pages.resizeColumnToContents(0)
        self.stale_pages.blockSignals(False)

    def _stale_page_changed(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        if self._refreshing or self._model is None:
            return
        page_id = str(item.data(0, QtCore.Qt.ItemDataRole.UserRole) or "")
        try:
            self._model.set_stale_page_selected(
                page_id,
                item.checkState(0) is QtCore.Qt.CheckState.Checked,
            )
        except KeyError:
            return
        self._refresh_actions()

    def _open_stale_page(self) -> None:
        if self._model is None:
            return
        page_id = self._selected_data(self.stale_pages)
        selected = self._model.state.selected_stale_page_ids
        if not page_id and selected:
            page_id = selected[0]
        if page_id and page_id in selected:
            self.open_stale_page_requested.emit(page_id)

    def _populate_history(self, selected_id: str) -> None:
        self.history.blockSignals(True)
        self.history.clear()
        if self._model is not None:
            for reference in self._model.state.selection.history:
                if reference.is_control:
                    continue
                label = {
                    "set_entry": "Set glossary entry",
                    "remove_entry": "Remove glossary entry",
                }.get(reference.operation, reference.operation.replace("_", " ").title())
                item = QtWidgets.QTreeWidgetItem(
                    (label, "Active" if reference.active else "Revoked")
                )
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, reference.record_id)
                self.history.addTopLevelItem(item)
                if reference.record_id == selected_id:
                    self.history.setCurrentItem(item)
        self.history.resizeColumnToContents(0)
        self.history.blockSignals(False)

    def _run_history(self) -> None:
        if self._model is None:
            return
        edit_id = self._selected_data(self.history)
        if not edit_id:
            return
        try:
            command = self._model.begin_history(edit_id)
        except RuntimeError as exc:
            self.status.setText(str(exc))
            return
        self.refresh()
        self.command_requested.emit(command)

    @staticmethod
    def _selected_data(tree: QtWidgets.QTreeWidget) -> str:
        item = tree.currentItem()
        return str(item.data(0, QtCore.Qt.ItemDataRole.UserRole) or "") if item else ""

    def _refresh_actions(self, *_args: object) -> None:
        state = self._model.state if self._model is not None else None
        bound = state is not None
        busy = bool(state and state.busy)
        dirty = bool(state and state.dirty)
        problem = self._model.draft_problem() if self._model is not None else ""
        persisted = bool(
            state
            and any(
                entry.entry_id == state.selected_entry_id
                for entry in state.selection.entries
            )
        )
        self.search.setEnabled(bound and not busy and not dirty)
        self.entries.setEnabled(bound and not busy and not dirty)
        self.new_button.setEnabled(bound and not busy and not dirty)
        self.import_button.setEnabled(bound and not busy and not dirty)
        self.export_button.setEnabled(bound and not busy)
        for control in (self.source, self.target, self.notes, self.aliases, self.priority):
            control.setEnabled(bound and not busy and state.draft is not None)
        self.save_button.setEnabled(bound and dirty and not busy and not problem)
        self.cancel_button.setEnabled(bound and dirty and not busy)
        self.remove_button.setEnabled(bound and persisted and not busy and not dirty)
        self.worker_cancel_button.setVisible(busy)
        self.worker_cancel_button.setEnabled(False if not busy else self.worker_cancel_button.isEnabled())
        self.stale_pages.setEnabled(bound and not busy and not dirty)
        selected_stale = self._selected_data(self.stale_pages)
        self.open_page_button.setEnabled(
            bound
            and not busy
            and not dirty
            and bool(
                (selected_stale and selected_stale in state.selected_stale_page_ids)
                or state.selected_stale_page_ids
            )
        )
        self.history.setEnabled(bound and not busy and not dirty)
        stale_count = len(state.selection.stale_page_ids) if state is not None else 0
        self.stale_summary.setText(
            f"{stale_count} completed page{'s' if stale_count != 1 else ''} may be stale"
            if stale_count
            else "Translations are current"
        )
        stale_tone = "warning" if stale_count else "ready"
        if self.stale_summary.property("tone") != stale_tone:
            self.stale_summary.setProperty("tone", stale_tone)
            self.stale_summary.style().unpolish(self.stale_summary)
            self.stale_summary.style().polish(self.stale_summary)
        self.review_pages_button.setVisible(bool(stale_count))
        self.review_pages_button.setEnabled(bound and not busy and not dirty and bool(stale_count))
        self.export_button.setVisible(self._advanced)
        self.manage_history_button.setVisible(self._advanced)
        self.manage_history_button.setEnabled(bound and not busy and not dirty)
        if self._model is not None and self._model.state.draft is not None:
            alias_problem = self._model.draft_problem()
        else:
            alias_problem = ""
        alias_conflict = bool(alias_problem and ("alias" in alias_problem.casefold() or "source" in alias_problem.casefold()))
        self.alias_warning.setVisible(alias_conflict)
        if alias_conflict:
            self.alias_warning_copy.setText(f"Alias review\n{alias_problem}")
            self.status.hide()
        elif state is not None:
            self.status.setVisible(
                bool(state.busy or state.dirty)
                or state.phase
                in {GlossaryEditorPhase.FAILED, GlossaryEditorPhase.STALE}
            )
        history_id = self._selected_data(self.history)
        selected_history = next(
            (
                item
                for item in state.selection.history
                if not item.is_control and item.record_id == history_id
            ),
            None,
        ) if state is not None else None
        if selected_history is None:
            self.history_button.setText("Select a History entry")
            self.history_button.setEnabled(False)
        else:
            self.history_button.setText("Revoke" if selected_history.active else "Reapply")
            self.history_button.setAccessibleName(
                "Revoke glossary History entry"
                if selected_history.active
                else "Reapply glossary History entry"
            )
            self.history_button.setEnabled(not busy and not dirty)


__all__ = ["GlossarySettingsPage"]
