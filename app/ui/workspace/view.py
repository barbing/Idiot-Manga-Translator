# -*- coding: utf-8 -*-
"""Translation Workspace view backed exclusively by typed models."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.design_system.components import HybridComboBox
from app.ui.design_system.delegates import WorkspacePageQueueDelegate
from app.ui.design_system.icons import hybrid_icon
from app.ui.presentation import NextActionPresentation
from app.ui.ui_contract import LayoutMode
from app.ui.viewmodels.project_model import PageRole


class _PageFilterProxy(QtCore.QSortFilterProxyModel):
    """Typed workspace filtering without creating a second page authority."""

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._query = ""
        self._state = "all"
        self.setDynamicSortFilter(True)

    def _invalidate_rows(self) -> None:
        direction = getattr(QtCore.QSortFilterProxyModel, "Direction", None)
        begin = getattr(self, "beginFilterChange", None)
        end = getattr(self, "endFilterChange", None)
        if direction is not None and callable(begin) and callable(end):
            begin()
            end(direction.Rows)
            return
        self.invalidateRowsFilter()

    def set_query(self, value: str) -> None:
        query = str(value or "").strip().casefold()
        if query == self._query:
            return
        self._query = query
        self._invalidate_rows()

    def set_state(self, value: str) -> None:
        state = str(value or "all").strip().casefold()
        if state == self._state:
            return
        self._state = state
        self._invalidate_rows()

    def filterAcceptsRow(
        self,
        source_row: int,
        source_parent: QtCore.QModelIndex,
    ) -> bool:
        model = self.sourceModel()
        if model is None:
            return False
        index = model.index(source_row, 0, source_parent)
        file_name = str(index.data(int(PageRole.FILE_NAME)) or "").casefold()
        label = str(index.data(int(PageRole.WORKSPACE_STATUS_LABEL)) or "").casefold()
        tone = str(index.data(int(PageRole.WORKSPACE_STATUS_TONE)) or "").casefold()
        needs_review = bool(index.data(int(PageRole.NEEDS_REVIEW)))
        if self._query and self._query not in file_name and self._query not in label:
            return False
        if self._state == "all":
            return True
        if self._state == "pending":
            return tone in {"queued", "muted"} or "queued" in label or "pending" in label
        if self._state == "running":
            return tone in {"editing", "info"} or "running" in label
        if self._state == "completed":
            return tone == "ready" and not needs_review
        if self._state == "review":
            return needs_review or "review" in label or "conflict" in label
        if self._state == "stale":
            return "stale" in label
        if self._state == "error":
            return tone == "error" or "error" in label or "failed" in label
        return True


class _StageStep(QtWidgets.QFrame):
    """Prototype-faithful stage row with real icon-library state marks."""

    def __init__(
        self,
        *,
        number: int,
        title: str,
        detail: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("stageStep")
        self.setProperty("state", "pending")
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(10, 5, 10, 5)
        row.setSpacing(9)
        self.number = QtWidgets.QLabel(str(number))
        self.number.setObjectName("stageStepNumber")
        self.number.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.number.setFixedSize(24, 24)
        row.addWidget(self.number)
        copy = QtWidgets.QVBoxLayout()
        copy.setSpacing(1)
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("stageStepTitle")
        self.detail = QtWidgets.QLabel(detail)
        self.detail.setObjectName("stageStepDetail")
        self.detail.setWordWrap(True)
        self.detail.setMinimumWidth(0)
        self.detail.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        copy.addWidget(self.title)
        copy.addWidget(self.detail)
        row.addLayout(copy, 1)
        self.state_icon = QtWidgets.QLabel()
        self.state_icon.setFixedSize(12, 12)
        row.addWidget(self.state_icon)
        self._icon_signature: tuple[str, str] | None = None
        self.setAccessibleName(f"{title} stage: pending. {detail}")

    def set_stage_state(self, state: str, *, theme: str) -> None:
        normalized = str(state)
        self.setProperty("state", normalized)
        self.setAccessibleName(
            f"{self.title.text()} stage: {normalized}. {self.detail.text()}"
        )
        signature = (normalized, str(theme))
        if signature == self._icon_signature:
            return
        icon_name = {
            "complete": "status-ready",
            "active": "status-editing",
        }.get(normalized, "status-queued")
        self.state_icon.setPixmap(
            hybrid_icon(icon_name, theme).pixmap(QtCore.QSize(9, 9))
        )
        self._icon_signature = signature
        self.style().unpolish(self)
        self.style().polish(self)

    def set_detail(self, detail: str) -> None:
        normalized = str(detail or "").strip()
        if not normalized:
            raise ValueError("stage detail must not be empty")
        if self.detail.text() != normalized:
            self.detail.setText(normalized)
        self.setAccessibleName(
            f"{self.title.text()} stage: "
            f"{str(self.property('state') or 'pending')}. {normalized}"
        )


class TranslationWorkspaceView(QtWidgets.QWidget):
    """Page-local run monitor and command surface."""

    start_requested = QtCore.Signal()
    stop_after_page_requested = QtCore.Signal()
    cancel_page_requested = QtCore.Signal()
    retry_requested = QtCore.Signal(str)
    page_editor_requested = QtCore.Signal(str)
    settings_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("translationWorkspace")
        self.setAccessibleName("Translation Workspace")
        self._page_id_role: int | None = None
        self._layout_mode: LayoutMode | None = None
        self._metric_pairs: list[tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = []
        self._page_proxy = _PageFilterProxy(self)
        self._project_page_count = 0
        self._icon_theme = "dark"
        self._runtime_tone = "muted"
        self._memory_check_required = False
        self._memory_checking = False

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.workspace_scroll = QtWidgets.QScrollArea()
        self.workspace_scroll.setObjectName("workspaceScroll")
        self.workspace_scroll.setWidgetResizable(True)
        self.workspace_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.workspace_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.workspace_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        workspace_content = QtWidgets.QWidget()
        workspace_content.setObjectName("workspaceScrollContent")
        root = QtWidgets.QVBoxLayout(workspace_content)
        root.setContentsMargins(42, 26, 42, 0)
        root.setSpacing(0)
        self.workspace_scroll.setWidget(workspace_content)
        outer.addWidget(self.workspace_scroll)

        header = QtWidgets.QHBoxLayout()
        heading = QtWidgets.QVBoxLayout()
        heading.setSpacing(0)
        eyebrow = QtWidgets.QLabel("TRANSLATION WORKSPACE")
        eyebrow.setProperty("role", "eyebrow")
        self.project_title = QtWidgets.QLabel("No project selected")
        self.project_title.setObjectName("surfaceTitle")
        self.project_detail = QtWidgets.QLabel(
            "Page-local execution · every completed page is committed before the next begins."
        )
        self.project_detail.setProperty("role", "secondary")
        heading.addWidget(eyebrow)
        heading.addWidget(self.project_title)
        heading.addWidget(self.project_detail)
        header.addLayout(heading, 1)

        self.command_bar = QtWidgets.QFrame()
        self.command_bar.setObjectName("workspaceHeadingActions")
        self.command_bar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        commands = QtWidgets.QHBoxLayout(self.command_bar)
        commands.setContentsMargins(0, 0, 0, 0)
        commands.setSpacing(8)
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setObjectName("primaryCommand")
        self.start_button.setProperty("role", "command")
        self.start_button.setProperty("variant", "primary")
        self.start_button.setAccessibleName("Start translation")
        self.start_button.setAccessibleDescription(
            "Start the exact effective run shown in Settings"
        )
        self.start_button.clicked.connect(self.start_requested)
        self.stop_button = QtWidgets.QPushButton("Stop after safe point")
        self.stop_button.setProperty("role", "command")
        self.stop_button.setProperty("variant", "secondary")
        self.stop_button.setProperty("tone", "danger")
        self.stop_button.setAccessibleName("Stop after current page")
        self.stop_button.setAccessibleDescription(
            "Request a stop at the next safe page boundary"
        )
        self.stop_button.clicked.connect(self.stop_after_page_requested)
        self.cancel_button = QtWidgets.QPushButton("Cancel current page")
        self.cancel_button.setProperty("role", "command")
        self.cancel_button.setProperty("variant", "secondary")
        self.cancel_button.setProperty("tone", "danger")
        self.cancel_button.setAccessibleName("Cancel current page")
        self.cancel_button.setAccessibleDescription(
            "Unavailable because the current controller supports only a safe-page stop"
        )
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_page_requested)
        commands.addWidget(self.cancel_button)
        commands.addWidget(self.stop_button)
        commands.addWidget(self.start_button)
        self.open_editor_button = QtWidgets.QPushButton("Open in Editor")
        self.open_editor_button.setProperty("role", "command")
        self.open_editor_button.setProperty("variant", "primary")
        self.open_editor_button.setAccessibleName("Open current page in Editor")
        self.open_editor_button.clicked.connect(self._open_current_page)
        commands.addWidget(self.open_editor_button)
        self.effective_run_button = QtWidgets.QPushButton("Review effective run")
        self.effective_run_button.setProperty("role", "command")
        self.effective_run_button.setProperty("variant", "quiet")
        self.effective_run_button.setAccessibleName("Review effective run")
        self.effective_run_button.setAccessibleDescription(
            "Open Settings to review the exact values captured for the next run"
        )
        self.effective_run_button.clicked.connect(self.settings_requested)
        self.effective_run_button.hide()
        self.command_scroll = self._horizontal_scroll(QtWidgets.QWidget())
        self.command_scroll.setAccessibleName("Workspace commands")
        self.command_scroll.hide()
        header.addWidget(self.command_bar)
        root.addLayout(header)
        root.addSpacing(25)

        self.recovery_banner = QtWidgets.QFrame()
        self.recovery_banner.setObjectName("workspaceRecovery")
        recovery_layout = QtWidgets.QHBoxLayout(self.recovery_banner)
        recovery_layout.setContentsMargins(12, 8, 12, 8)
        recovery_layout.setSpacing(10)
        self.recovery_icon = QtWidgets.QLabel()
        self.recovery_icon.setFixedSize(20, 20)
        recovery_layout.addWidget(self.recovery_icon)
        recovery_copy = QtWidgets.QVBoxLayout()
        recovery_copy.setSpacing(3)
        self.recovery_title = QtWidgets.QLabel("A page stopped safely")
        self.recovery_title.setProperty("role", "section")
        recovery_copy.addWidget(self.recovery_title)
        self.recovery_detail = QtWidgets.QLabel(
            "The previous page revision remains intact. Retry from the owning stage when ready."
        )
        self.recovery_detail.setProperty("role", "secondary")
        self.recovery_detail.setWordWrap(True)
        recovery_copy.addWidget(self.recovery_detail)
        recovery_layout.addLayout(recovery_copy, 1)
        self.recovery_retry_button = QtWidgets.QPushButton("Retry selected action")
        self.recovery_retry_button.setProperty("role", "command")
        self.recovery_retry_button.setProperty("variant", "secondary")
        self.recovery_retry_button.clicked.connect(self._retry_selected)
        recovery_layout.addWidget(self.recovery_retry_button)
        self._application_recovery: tuple[str, str] | None = None
        self.recovery_banner.hide()
        root.addWidget(self.recovery_banner)

        self.content_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.content_splitter.setObjectName("workspaceContent")
        self.content_splitter.setChildrenCollapsible(False)

        queue_panel = QtWidgets.QFrame()
        queue_panel.setObjectName("surfacePanel")
        queue_panel.setProperty("role", "panel")
        queue_layout = QtWidgets.QVBoxLayout(queue_panel)
        queue_layout.setContentsMargins(12, 12, 12, 12)
        queue_layout.setSpacing(8)
        queue_header = QtWidgets.QHBoxLayout()
        queue_copy = QtWidgets.QVBoxLayout()
        queue_copy.setSpacing(2)
        queue_title = QtWidgets.QLabel("Page queue")
        queue_title.setProperty("role", "section")
        self.queue_detail = QtWidgets.QLabel(
            "Typed page status and owning stage"
        )
        self.queue_detail.setProperty("role", "secondary")
        queue_copy.addWidget(queue_title)
        queue_copy.addWidget(self.queue_detail)
        self.page_search = QtWidgets.QLineEdit()
        self.page_search.setObjectName("workspacePageSearch")
        self.page_search.setPlaceholderText("Search")
        self.page_search.setClearButtonEnabled(True)
        self.page_search.setAccessibleName("Search project pages")
        self.page_search_action = self.page_search.addAction(
            hybrid_icon("search"),
            QtWidgets.QLineEdit.ActionPosition.LeadingPosition,
        )
        self.page_search.textChanged.connect(self._page_proxy.set_query)
        self.page_filter = HybridComboBox()
        self.page_filter.setObjectName("workspacePageFilter")
        self.page_filter.setAccessibleName("Filter pages")
        self.page_filter.addItem("All states", "all")
        self.page_filter.addItem("Pending", "pending")
        self.page_filter.addItem("Running", "running")
        self.page_filter.addItem("Completed", "completed")
        self.page_filter.addItem("Needs review", "review")
        self.page_filter.addItem("Stale", "stale")
        self.page_filter.addItem("Error", "error")
        self.page_filter.currentIndexChanged.connect(
            lambda _index: self._page_proxy.set_state(
                str(self.page_filter.currentData() or "all")
            )
        )
        queue_header.addLayout(queue_copy)
        queue_header.addStretch(1)
        queue_header.addWidget(self.page_search)
        queue_header.addWidget(self.page_filter)
        queue_layout.addLayout(queue_header)
        queue_columns = QtWidgets.QWidget()
        queue_columns.setObjectName("workspaceQueueColumns")
        queue_columns_layout = QtWidgets.QHBoxLayout(queue_columns)
        queue_columns_layout.setContentsMargins(0, 0, 0, 0)
        queue_columns_layout.setSpacing(0)
        for text, stretch in (
            ("PAGE", 130),
            ("STATUS", 100),
            ("OWNER", 90),
            ("TIME", 65),
        ):
            label = QtWidgets.QLabel(text)
            label.setProperty("role", "eyebrow")
            queue_columns_layout.addWidget(label, stretch)
        action_label = QtWidgets.QLabel("ACTION")
        action_label.setProperty("role", "eyebrow")
        action_label.setFixedWidth(36)
        queue_columns_layout.addWidget(action_label)
        queue_layout.addWidget(queue_columns)
        self.page_list = QtWidgets.QListView()
        self.page_list.setObjectName("workspacePageList")
        self.page_list.setAccessibleName("Project pages")
        self.page_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.page_list.setWordWrap(True)
        self.page_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.page_list.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.page_list.setItemDelegate(
            WorkspacePageQueueDelegate(parent=self.page_list)
        )
        self.page_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.page_list.activated.connect(self._open_selected_page)
        self.page_list.clicked.connect(self._open_selected_page)
        queue_layout.addWidget(self.page_list, 1)
        self.open_page_button = QtWidgets.QPushButton("Open completed page")
        self.open_page_button.setProperty("role", "command")
        self.open_page_button.setProperty("variant", "secondary")
        self.open_page_button.setAccessibleName("Open completed page")
        self.open_page_button.setAccessibleDescription(
            "Open the selected completed page in the Page Editor"
        )
        self.open_page_button.clicked.connect(self._open_current_page)
        queue_layout.addWidget(self.open_page_button)
        self.open_page_button.hide()

        monitor_panel = QtWidgets.QFrame()
        monitor_panel.setObjectName("workspaceRunCard")
        monitor_panel.setProperty("role", "panel")
        monitor_panel.setProperty("importance", "primary")
        monitor = QtWidgets.QVBoxLayout(monitor_panel)
        monitor.setContentsMargins(16, 14, 16, 14)
        monitor.setSpacing(12)
        monitor_header = QtWidgets.QHBoxLayout()
        monitor_identity = QtWidgets.QVBoxLayout()
        monitor_identity.setSpacing(7)
        self.run_state_pill = QtWidgets.QLabel("Ready")
        self.run_state_pill.setObjectName("statusPill")
        self.run_state_pill.setProperty("role", "status-pill")
        self.run_state_pill.setProperty("tone", "muted")
        self.run_state_pill.setAccessibleName("Run state: Ready")
        monitor_identity.addWidget(
            self.run_state_pill,
            0,
            QtCore.Qt.AlignmentFlag.AlignLeft,
        )
        self.monitor_title = QtWidgets.QLabel("No active run")
        self.monitor_title.setObjectName("workspaceRunTitle")
        monitor_identity.addWidget(self.monitor_title)
        monitor_header.addLayout(monitor_identity)
        monitor_header.addStretch(1)
        self.monitor_eta = QtWidgets.QLabel("ETA —")
        self.monitor_eta.setProperty("role", "secondary")
        monitor_header.addWidget(self.monitor_eta)
        monitor.addLayout(monitor_header)
        self.metrics_layout = QtWidgets.QGridLayout()
        self.metrics_layout.setHorizontalSpacing(20)
        self.metrics_layout.setVerticalSpacing(6)
        self.stage_value = self._metric(self.metrics_layout, 0, "CURRENT STAGE", "Ready")
        self.page_value = self._metric(self.metrics_layout, 1, "CURRENT PAGE", "—")
        self.parent_value = self._metric(self.metrics_layout, 2, "CURRENT PARENT", "—")
        self.elapsed_value = self._metric(self.metrics_layout, 3, "ELAPSED", "00:00")
        self.eta_value = self._metric(self.metrics_layout, 4, "ETA", "—")
        self.metrics_holder = QtWidgets.QWidget()
        self.metrics_holder.setLayout(self.metrics_layout)
        self.metrics_holder.hide()
        self.overall_progress = QtWidgets.QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setValue(0)
        self.overall_progress.setFormat("No active run")
        self.overall_progress.setAccessibleName("Overall translation progress")
        monitor.addWidget(self.overall_progress)
        self.run_stats = QtWidgets.QHBoxLayout()
        self.run_stats.setSpacing(34)
        self.run_stat_values: dict[str, QtWidgets.QLabel] = {}
        for key, label in (
            ("done", "Done"),
            ("active", "In progress"),
            ("queued", "Queued"),
            ("errors", "Errors"),
        ):
            group = QtWidgets.QVBoxLayout()
            group.setSpacing(2)
            value = QtWidgets.QLabel("0")
            value.setProperty("role", "metric")
            caption = QtWidgets.QLabel(label)
            caption.setProperty("role", "secondary")
            self.run_stat_values[key] = value
            group.addWidget(value)
            group.addWidget(caption)
            self.run_stats.addLayout(group)
        self.run_stats.addStretch(1)
        monitor.addLayout(self.run_stats)
        self.stage_strip = QtWidgets.QWidget()
        self.stage_strip.setObjectName("workspaceStageStrip")
        stage_layout = QtWidgets.QVBoxLayout(self.stage_strip)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(6)
        self.stage_labels: dict[str, _StageStep] = {}
        for stage_key, stage, detail in (
            ("detection", "Bubble detection", "Captured when Start begins"),
            ("ocr", "OCR", "Captured when Start begins"),
            ("translation", "Translation", "Select a provider in Settings"),
            ("cleanup", "Cleanup", "Waits for Translation"),
            ("rendering", "Rendering", "Waits for clean base"),
        ):
            step = _StageStep(
                number=len(self.stage_labels) + 1,
                title=stage,
                detail=detail,
            )
            stage_layout.addWidget(step)
            self.stage_labels[stage_key] = step
        self.stage_scroll = self._horizontal_scroll(self.stage_strip)
        self.stage_scroll.setObjectName("workspaceStageScroll")
        self.stage_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: 0; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
            "QWidget#workspaceStageStrip { background: transparent; }"
        )
        self.stage_scroll.setAccessibleName("Translation stages")
        self.stage_scroll.setMinimumHeight(250)
        self.stage_scroll.setMaximumHeight(276)
        self.run_detail = QtWidgets.QLabel(
            "Run progress and failures appear here as structured state."
        )
        self.run_detail.setWordWrap(True)
        self.run_detail.setProperty("role", "secondary")

        error_panel = QtWidgets.QFrame()
        error_panel.setObjectName("surfacePanel")
        error_panel.setProperty("role", "panel")
        error_layout = QtWidgets.QVBoxLayout(error_panel)
        error_layout.setContentsMargins(12, 12, 12, 12)
        error_layout.setSpacing(8)
        error_header = QtWidgets.QHBoxLayout()
        error_title = QtWidgets.QLabel("Errors and recovery")
        error_title.setProperty("role", "section")
        self.error_count = QtWidgets.QLabel("0")
        self.error_count.setObjectName("statusPill")
        self.error_count.setProperty("role", "status-pill")
        self.error_count.setProperty("tone", "muted")
        error_header.addWidget(error_title)
        error_header.addStretch(1)
        error_header.addWidget(self.error_count)
        error_layout.addLayout(error_header)
        self.error_list = QtWidgets.QListView()
        self.error_list.setObjectName("structuredErrors")
        self.error_list.setAccessibleName("Structured errors and recovery actions")
        self.error_list.setWordWrap(True)
        error_layout.addWidget(self.error_list, 1)
        self.retry_button = QtWidgets.QPushButton("Retry selected action")
        self.retry_button.setProperty("role", "command")
        self.retry_button.setProperty("variant", "secondary")
        self.retry_button.setAccessibleName("Retry selected recovery action")
        self.retry_button.setAccessibleDescription(
            "Run the typed recovery action for the selected structured error"
        )
        self.retry_button.clicked.connect(self._retry_selected)
        error_layout.addWidget(self.retry_button)

        current_panel = QtWidgets.QFrame()
        current_panel.setObjectName("workspaceCurrentCard")
        current_panel.setProperty("role", "panel")
        current_panel.setProperty("importance", "secondary")
        current_layout = QtWidgets.QVBoxLayout(current_panel)
        current_layout.setContentsMargins(16, 14, 16, 14)
        current_layout.setSpacing(9)
        current_header = QtWidgets.QHBoxLayout()
        current_identity = QtWidgets.QVBoxLayout()
        current_identity.setSpacing(7)
        current_heading = QtWidgets.QLabel("CURRENT PAGE")
        current_heading.setProperty("role", "eyebrow")
        current_identity.addWidget(current_heading)
        self.current_page_summary = QtWidgets.QLabel("No page active")
        self.current_page_summary.setProperty("role", "title")
        current_identity.addWidget(self.current_page_summary)
        current_header.addLayout(current_identity, 1)
        self.current_stage_summary = QtWidgets.QLabel("Waiting to start")
        self.current_stage_summary.setProperty("role", "status-pill")
        self.current_stage_summary.setProperty("tone", "muted")
        current_header.addWidget(
            self.current_stage_summary,
            0,
            QtCore.Qt.AlignmentFlag.AlignTop,
        )
        current_layout.addLayout(current_header)
        current_stage_line = QtWidgets.QHBoxLayout()
        self.current_parent_summary = QtWidgets.QLabel("Parent —")
        self.current_parent_summary.setProperty("role", "secondary")
        current_stage_line.addWidget(self.current_parent_summary)
        current_stage_line.addStretch(1)
        self.current_elapsed_summary = QtWidgets.QLabel("— elapsed")
        self.current_elapsed_summary.setProperty("role", "secondary")
        current_stage_line.addWidget(self.current_elapsed_summary)
        current_layout.addLayout(current_stage_line)
        self.current_page_progress = QtWidgets.QProgressBar()
        self.current_page_progress.setRange(0, 100)
        self.current_page_progress.setValue(0)
        self.current_page_progress.setTextVisible(False)
        self.current_page_progress.setAccessibleName("Current page progress")
        current_layout.addWidget(self.current_page_progress)
        self.current_detail = QtWidgets.QLabel(
            "Waiting for the current stage owner."
        )
        self.current_detail.setWordWrap(True)
        self.current_detail.setProperty("role", "secondary")
        current_layout.addWidget(self.current_detail)
        current_layout.addStretch(1)

        runtime_panel = QtWidgets.QFrame()
        runtime_panel.setObjectName("workspaceRuntimeCard")
        runtime_panel.setProperty("role", "panel")
        runtime_panel.setProperty("importance", "secondary")
        runtime_layout = QtWidgets.QVBoxLayout(runtime_panel)
        runtime_layout.setContentsMargins(16, 14, 16, 14)
        runtime_layout.setSpacing(8)
        runtime_heading = QtWidgets.QLabel("RUNTIME")
        runtime_heading.setProperty("role", "eyebrow")
        runtime_layout.addWidget(runtime_heading)
        runtime_status_row = QtWidgets.QHBoxLayout()
        runtime_status_row.setSpacing(8)
        self.runtime_status = QtWidgets.QLabel("Needs configuration")
        self.runtime_status.setProperty("role", "title")
        runtime_status_row.addWidget(self.runtime_status, 1)
        self.runtime_status_icon = QtWidgets.QLabel()
        self.runtime_status_icon.setFixedSize(24, 24)
        self.runtime_status_icon.setAccessibleName("Runtime status")
        runtime_status_row.addWidget(self.runtime_status_icon)
        runtime_layout.addLayout(runtime_status_row)
        self.runtime_facts_widget = QtWidgets.QWidget()
        self.runtime_facts_layout = QtWidgets.QVBoxLayout(
            self.runtime_facts_widget
        )
        self.runtime_facts_layout.setContentsMargins(0, 0, 0, 0)
        self.runtime_facts_layout.setSpacing(6)
        self.runtime_fact_icons: list[QtWidgets.QLabel] = []
        self.runtime_fact_labels: list[QtWidgets.QLabel] = []
        for _index in range(3):
            fact = QtWidgets.QWidget()
            fact_layout = QtWidgets.QHBoxLayout(fact)
            fact_layout.setContentsMargins(0, 0, 0, 0)
            fact_layout.setSpacing(7)
            icon = QtWidgets.QLabel()
            icon.setFixedSize(16, 16)
            label = QtWidgets.QLabel()
            label.setProperty("role", "secondary")
            fact_layout.addWidget(icon)
            fact_layout.addWidget(label, 1)
            fact.hide()
            self.runtime_facts_layout.addWidget(fact)
            self.runtime_fact_icons.append(icon)
            self.runtime_fact_labels.append(label)
        runtime_layout.addWidget(self.runtime_facts_widget)
        self.runtime_detail = QtWidgets.QLabel(
            "Select a translation provider and review the effective run."
        )
        self.runtime_detail.setWordWrap(True)
        self.runtime_detail.setProperty("role", "secondary")
        runtime_layout.addWidget(self.runtime_detail)
        runtime_layout.addStretch(1)
        self.runtime_settings = QtWidgets.QPushButton("Open Settings")
        self.runtime_settings.setProperty("role", "command")
        self.runtime_settings.setProperty("variant", "quiet")
        self.runtime_settings.setAccessibleName("Open provider settings")
        self.runtime_settings.setAccessibleDescription(
            "Open Settings to configure or validate the selected translation provider"
        )
        self.runtime_settings.clicked.connect(self.settings_requested)
        runtime_layout.addWidget(
            self.runtime_settings,
            0,
            QtCore.Qt.AlignmentFlag.AlignRight,
        )

        self.summary_row_widget = QtWidgets.QWidget()
        self.summary_row = QtWidgets.QBoxLayout(
            QtWidgets.QBoxLayout.Direction.LeftToRight,
            self.summary_row_widget,
        )
        self.summary_row.setContentsMargins(0, 0, 0, 0)
        self.summary_row.setSpacing(12)
        self.summary_row.addWidget(monitor_panel, 6)
        self.summary_row.addWidget(current_panel, 5)
        self.summary_row.addWidget(runtime_panel, 4)
        self.summary_row_widget.setMinimumHeight(156)
        self.summary_row_widget.setMaximumHeight(156)
        root.addWidget(self.summary_row_widget)
        root.addSpacing(12)

        self.next_action_frame = QtWidgets.QFrame()
        self.next_action_frame.setObjectName("workspaceNextAction")
        self.next_action_frame.setProperty("role", "state-callout")
        self.next_action_frame.setProperty("tone", "muted")
        next_action_layout = QtWidgets.QHBoxLayout(self.next_action_frame)
        next_action_layout.setContentsMargins(12, 8, 12, 8)
        next_action_layout.setSpacing(12)
        next_action_eyebrow = QtWidgets.QLabel("NEXT ACTION")
        next_action_eyebrow.setProperty("role", "eyebrow")
        next_action_layout.addWidget(next_action_eyebrow)
        self.next_action_title = QtWidgets.QLabel("Open or create a project")
        self.next_action_title.setProperty("role", "section")
        next_action_layout.addWidget(self.next_action_title)
        self.next_action_detail = QtWidgets.QLabel(
            "Choose source pages from Project Hub before starting translation."
        )
        self.next_action_detail.setProperty("role", "secondary")
        self.next_action_detail.setWordWrap(True)
        next_action_layout.addWidget(self.next_action_detail, 1)
        self.next_action_frame.setAccessibleName("Next action")
        self.next_action_frame.setAccessibleDescription(
            self.next_action_detail.text()
        )
        root.addWidget(self.next_action_frame)
        root.addSpacing(12)

        primary_column = QtWidgets.QWidget()
        primary_layout = QtWidgets.QVBoxLayout(primary_column)
        primary_layout.setContentsMargins(0, 0, 0, 0)
        primary_layout.setSpacing(12)
        primary_layout.addWidget(queue_panel, 1)

        activity_panel = QtWidgets.QFrame()
        activity_panel.setObjectName("workspaceStageActivity")
        activity_panel.setProperty("role", "panel")
        activity_layout = QtWidgets.QVBoxLayout(activity_panel)
        activity_layout.setContentsMargins(12, 12, 12, 12)
        activity_layout.setSpacing(10)
        activity_header_widget = QtWidgets.QWidget()
        activity_header_widget.setMinimumHeight(37)
        activity_header = QtWidgets.QHBoxLayout(activity_header_widget)
        activity_header.setContentsMargins(0, 0, 0, 0)
        activity_title = QtWidgets.QLabel("Stage activity")
        activity_title.setProperty("role", "section")
        activity_header.addWidget(activity_title)
        activity_header.addStretch(1)
        self.activity_page = QtWidgets.QLabel("Page —")
        self.activity_page.setProperty("role", "secondary")
        activity_header.addWidget(self.activity_page)
        activity_layout.addWidget(activity_header_widget)
        activity_layout.addWidget(self.stage_scroll)
        activity_layout.addWidget(self.run_detail)
        self.stage_note = QtWidgets.QFrame()
        self.stage_note.setObjectName("workspaceStageNote")
        stage_note_layout = QtWidgets.QVBoxLayout(self.stage_note)
        stage_note_layout.setContentsMargins(10, 9, 10, 9)
        stage_note_layout.setSpacing(3)
        stage_note_title = QtWidgets.QLabel("No folder barrier")
        stage_note_title.setProperty("role", "section")
        stage_note_layout.addWidget(stage_note_title)
        stage_note_detail = QtWidgets.QLabel(
            "Each page is persisted before the next page begins."
        )
        stage_note_detail.setWordWrap(True)
        stage_note_detail.setProperty("role", "secondary")
        stage_note_layout.addWidget(stage_note_detail)
        activity_layout.addWidget(self.stage_note)
        activity_layout.addStretch(1)

        side_column = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side_column)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(12)
        side_layout.addWidget(activity_panel, 1)
        side_layout.addWidget(error_panel)
        self.error_panel = error_panel
        error_panel.hide()

        self.content_splitter.addWidget(primary_column)
        self.content_splitter.addWidget(side_column)
        self.content_splitter.setStretchFactor(0, 4)
        self.content_splitter.setStretchFactor(1, 1)
        self.content_splitter.setSizes((1050, 310))
        root.addWidget(self.content_splitter, 1)
        self.set_command_state(
            can_start=False,
            can_stop=False,
            can_cancel=False,
            can_open_page=False,
        )
        self.refresh_icons("dark")

    def refresh_icons(self, theme: str) -> None:
        self._icon_theme = str(theme)
        self.page_search_action.setIcon(hybrid_icon("search", theme))
        self.start_button.setIcon(
            hybrid_icon(
                "runtime"
                if self._memory_check_required or self._memory_checking
                else "workspace",
                theme,
                active=True,
            )
        )
        self.stop_button.setIcon(hybrid_icon("stop", theme))
        self.cancel_button.setIcon(hybrid_icon("close", theme))
        self.open_editor_button.setIcon(
            hybrid_icon("arrow-right", theme, active=True)
        )
        self.open_editor_button.setLayoutDirection(
            QtCore.Qt.LayoutDirection.RightToLeft
        )
        self.recovery_icon.setPixmap(
            hybrid_icon("warning", theme).pixmap(QtCore.QSize(19, 19))
        )
        runtime_icon = (
            "success"
            if self._runtime_tone == "ready"
            else "warning"
            if self._runtime_tone in {"warning", "error"}
            else "runtime"
        )
        self.runtime_status_icon.setPixmap(
            hybrid_icon(runtime_icon, theme).pixmap(QtCore.QSize(22, 22))
        )
        for icon in self.runtime_fact_icons:
            icon.setPixmap(
                hybrid_icon("success", theme).pixmap(QtCore.QSize(15, 15))
            )
        for step in self.stage_labels.values():
            step.set_stage_state(
                str(step.property("state") or "pending"),
                theme=theme,
            )

    @staticmethod
    def _horizontal_scroll(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setWidget(widget)
        return scroll

    def _metric(
        self,
        layout: QtWidgets.QGridLayout,
        column: int,
        label: str,
        value: str,
    ) -> QtWidgets.QLabel:
        name = QtWidgets.QLabel(label)
        name.setProperty("role", "eyebrow")
        result = QtWidgets.QLabel(value)
        result.setProperty("role", "metric")
        result.setAccessibleName(f"{label}: {value}")
        layout.addWidget(name, 0, column)
        layout.addWidget(result, 1, column)
        self._metric_pairs.append((name, result))
        return result

    def set_layout_mode(self, mode: LayoutMode) -> None:
        """Reflow the existing workspace without introducing a second UI."""

        if not isinstance(mode, LayoutMode):
            raise TypeError("mode must be LayoutMode")
        self._layout_mode = mode
        accessible = bool(mode.accessible_reflow or mode.width_tier == "narrow")
        orientation = (
            QtCore.Qt.Orientation.Vertical
            if accessible
            else QtCore.Qt.Orientation.Horizontal
        )
        if self.content_splitter.orientation() != orientation:
            self.content_splitter.setOrientation(orientation)
        self.summary_row.setDirection(
            QtWidgets.QBoxLayout.Direction.TopToBottom
            if accessible
            else QtWidgets.QBoxLayout.Direction.LeftToRight
        )
        if accessible:
            self.summary_row_widget.setMinimumHeight(0)
            self.summary_row_widget.setMaximumHeight(16_777_215)
        else:
            self.summary_row_widget.setMinimumHeight(156)
            self.summary_row_widget.setMaximumHeight(156)

        for name, value in self._metric_pairs:
            self.metrics_layout.removeWidget(name)
            self.metrics_layout.removeWidget(value)
        if accessible:
            # Two semantic metric columns remain readable at 150–200% without
            # turning the run monitor into a horizontally clipped five-column row.
            for index, (name, value) in enumerate(self._metric_pairs):
                column = index % 2
                row = (index // 2) * 2
                self.metrics_layout.addWidget(name, row, column)
                self.metrics_layout.addWidget(value, row + 1, column)
        else:
            for column, (name, value) in enumerate(self._metric_pairs):
                self.metrics_layout.addWidget(name, 0, column)
                self.metrics_layout.addWidget(value, 1, column)

        self.command_bar.setMinimumWidth(self.command_bar.sizeHint().width())
        # Stage details are responsive copy, not a horizontally scrolling
        # command strip.  Let the resizable scroll-area viewport own the width
        # and wrap longer runtime/provider labels inside their stage row.
        self.stage_strip.setMinimumWidth(0)
        self.stage_scroll.setMinimumHeight(250)
        self.stage_scroll.setMaximumHeight(276)

    def set_current_page(self, page_id: str | None) -> None:
        """Synchronize and reveal the current page without emitting navigation."""

        model = self.page_list.model()
        if model is None or self._page_id_role is None:
            return
        target = str(page_id or "").strip()
        blocker = QtCore.QSignalBlocker(self.page_list)
        try:
            if not target:
                self.page_list.clearSelection()
                self.page_list.setCurrentIndex(QtCore.QModelIndex())
                return
            for row in range(model.rowCount()):
                index = model.index(row, 0)
                if str(index.data(self._page_id_role) or "").strip() == target:
                    self.page_list.setCurrentIndex(index)
                    self.page_list.scrollTo(
                        index,
                        QtWidgets.QAbstractItemView.ScrollHint.EnsureVisible,
                    )
                    file_name = str(
                        index.data(int(PageRole.FILE_NAME)) or target
                    ).strip()
                    status = str(
                        index.data(int(PageRole.WORKSPACE_STATUS_LABEL)) or "Selected"
                    ).strip()
                    tone = str(
                        index.data(int(PageRole.WORKSPACE_STATUS_TONE)) or "muted"
                    ).strip()
                    parent_count = int(
                        index.data(int(PageRole.PARENT_COUNT)) or 0
                    )
                    self.current_page_summary.setText(file_name)
                    self.current_stage_summary.setText(status)
                    self.current_stage_summary.setProperty("tone", tone)
                    self.current_stage_summary.style().unpolish(
                        self.current_stage_summary
                    )
                    self.current_stage_summary.style().polish(
                        self.current_stage_summary
                    )
                    self.current_parent_summary.setText(
                        f"{parent_count} detected parents"
                        if parent_count
                        else "Detection has not started"
                    )
                    selected_elapsed = str(
                        index.data(int(PageRole.ELAPSED_LABEL)) or "—"
                    ).strip()
                    self.current_elapsed_summary.setText(
                        f"{selected_elapsed} elapsed"
                        if selected_elapsed and selected_elapsed != "—"
                        else "— elapsed"
                    )
                    return
        finally:
            del blocker

    def set_models(
        self,
        *,
        pages: QtCore.QAbstractItemModel,
        errors: QtCore.QAbstractItemModel,
        page_id_role: int,
    ) -> None:
        if not isinstance(pages, QtCore.QAbstractItemModel):
            raise TypeError("pages must be a QAbstractItemModel")
        if not isinstance(errors, QtCore.QAbstractItemModel):
            raise TypeError("errors must be a QAbstractItemModel")
        self._page_proxy.setSourceModel(pages)
        self.page_list.setModel(self._page_proxy)
        self.error_list.setModel(errors)
        self._page_id_role = int(page_id_role)
        for signal in (
            pages.rowsInserted,
            pages.rowsRemoved,
            pages.modelReset,
            pages.dataChanged,
            pages.layoutChanged,
        ):
            signal.connect(self._refresh_queue_stats)
        errors.rowsInserted.connect(self._refresh_error_count)
        errors.rowsRemoved.connect(self._refresh_error_count)
        errors.modelReset.connect(self._refresh_error_count)
        self._refresh_error_count()
        self._refresh_queue_stats()

    def set_project_summary(
        self,
        *,
        title: str,
        language_pair: str,
        page_count: int,
    ) -> None:
        self.project_title.setText(title)
        self.project_title.setAccessibleName(
            f"{title}. {language_pair}. {page_count} pages."
        )
        self._project_page_count = max(0, int(page_count))
        self.project_detail.setText(
            "Page-local execution · every completed page is committed before the next begins."
        )
        self.queue_detail.setText(
            f"{self._project_page_count} pages · typed page status and owning stage"
        )

    def set_prepared_run_mode(self, prepared: bool) -> None:
        """Name the page action truthfully before and after pipeline output exists."""

        if prepared:
            text = "Open source page"
            description = (
                "Open the selected imported source image in Page Editor. "
                "Pipeline results will appear after Start."
            )
        else:
            text = "Open completed page"
            description = "Open the selected persisted page in Page Editor."
        self.open_page_button.setText(text)
        self.open_page_button.setAccessibleName(text)
        self.open_page_button.setAccessibleDescription(description)
        self.open_page_button.setToolTip(description)

    def set_run_state(
        self,
        *,
        label: str,
        tone: str,
        stage: str,
        page: str,
        parent: str,
        elapsed: str,
        eta: str,
        percent: int,
        detail: str,
    ) -> None:
        percent = max(0, min(100, int(percent)))
        self.run_state_pill.setText(label)
        self.run_state_pill.setProperty("tone", tone)
        self.run_state_pill.setAccessibleName(f"Run state: {label}")
        self.run_state_pill.style().unpolish(self.run_state_pill)
        self.run_state_pill.style().polish(self.run_state_pill)
        values = (
            (self.stage_value, stage),
            (self.page_value, page),
            (self.parent_value, parent),
            (self.elapsed_value, elapsed),
            (self.eta_value, eta),
        )
        for widget, value in values:
            widget.setText(value)
        self.monitor_title.setText(
            f"{page} of {self._project_page_count} pages"
            if page and page != "—" and self._project_page_count
            else "No active run"
        )
        self.monitor_eta.setText(f"ETA {eta}" if eta and eta != "—" else "ETA —")
        self.activity_page.setText(
            f"Page {Path(page).stem}"
            if page and page != "—"
            else "Page —"
        )
        has_run_page = bool(page and page != "—")
        has_selected_page = self.page_list.currentIndex().isValid()
        if has_run_page:
            self.current_page_summary.setText(page)
            parent_label = str(parent or "").strip()
            self.current_parent_summary.setText(
                parent_label
                if parent_label.casefold().startswith("parent ")
                else f"Parent {parent_label}"
                if parent_label and parent_label != "—"
                else "Parent —"
            )
            self.current_elapsed_summary.setText(
                f"{elapsed} elapsed" if elapsed and elapsed != "—" else "— elapsed"
            )
            self.current_stage_summary.setText(stage or "Waiting to start")
            self.current_stage_summary.setProperty("tone", tone)
            self.current_stage_summary.style().unpolish(self.current_stage_summary)
            self.current_stage_summary.style().polish(self.current_stage_summary)
        elif not has_selected_page:
            self.current_page_summary.setText("No page active")
            self.current_parent_summary.setText("Parent —")
            self.current_elapsed_summary.setText("— elapsed")
            self.current_stage_summary.setText("Waiting to start")
            self.current_stage_summary.setProperty("tone", "muted")
            self.current_stage_summary.style().unpolish(self.current_stage_summary)
            self.current_stage_summary.style().polish(self.current_stage_summary)
        self.overall_progress.setValue(percent)
        self.overall_progress.setFormat(f"{percent}%")
        self.current_page_progress.setValue(percent)
        self.current_detail.setText(
            detail or "The current owner remains explicit before cleanup."
        )
        self.run_detail.setText(detail)
        normalized_stage = stage.casefold()
        active_index = next(
            (
                index
                for index, key in enumerate(self.stage_labels)
                if key in normalized_stage
            ),
            -1,
        )
        for index, (key, widget) in enumerate(self.stage_labels.items()):
            state = "complete" if 0 <= index < active_index else (
                "active" if index == active_index else "pending"
            )
            widget.set_stage_state(state, theme=self._icon_theme)

    def set_elapsed(self, elapsed: str) -> None:
        """Update live elapsed copy without rebuilding the run presentation."""

        value = str(elapsed or "").strip() or "—"
        self.elapsed_value.setText(value)
        self.current_elapsed_summary.setText(
            f"{value} elapsed" if value != "—" else "— elapsed"
        )

    def set_application_recovery(self, *, title: str, detail: str) -> None:
        """Show a GUI-owned failure without inventing a pipeline receipt."""

        normalized_title = str(title or "").strip()
        normalized_detail = str(detail or "").strip()
        if not normalized_title or not normalized_detail:
            raise ValueError("application recovery title and detail are required")
        self._application_recovery = (normalized_title, normalized_detail)
        self._refresh_error_count()

    def clear_application_recovery(self) -> None:
        self._application_recovery = None
        self._refresh_error_count()

    def set_command_state(
        self,
        *,
        can_start: bool,
        can_stop: bool,
        can_cancel: bool,
        can_open_page: bool,
        start_reason: str = "",
    ) -> None:
        self.start_button.setEnabled(bool(can_start))
        reason = str(start_reason or "").strip()
        self.start_button.setToolTip(reason)
        self.start_button.setStatusTip(reason)
        self.start_button.setAccessibleDescription(
            reason or "Start the exact effective run shown in Settings"
        )
        self.stop_button.setEnabled(bool(can_stop))
        self.cancel_button.setEnabled(bool(can_cancel))
        self.open_page_button.setEnabled(bool(can_open_page))
        self.open_editor_button.setEnabled(bool(can_open_page))
        running = bool(can_stop or can_cancel)
        self.start_button.setVisible(not running)
        self.stop_button.setVisible(running)
        self.cancel_button.setVisible(running)

    def set_start_memory_check_state(
        self,
        *,
        check_required: bool,
        checking: bool,
    ) -> None:
        """Present an explicit recheck action without claiming Start is allowed."""

        self._memory_check_required = bool(check_required)
        self._memory_checking = bool(checking)
        if checking:
            text = "Checking memory"
            accessible_name = "Checking memory budget"
        elif check_required:
            text = "Check memory"
            accessible_name = "Check memory budget"
        else:
            text = "Start"
            accessible_name = "Start translation"
        if self.start_button.text() != text:
            self.start_button.setText(text)
        self.start_button.setAccessibleName(accessible_name)
        self.start_button.setIcon(
            hybrid_icon(
                "runtime" if check_required or checking else "workspace",
                self._icon_theme,
                active=True,
            )
        )

    def set_runtime_summary(
        self,
        *,
        label: str,
        tone: str,
        detail: str,
        facts: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """Present captured runtime truth independently of command eligibility."""

        status = str(label or "Runtime not captured").strip()
        normalized_tone = str(tone or "muted").strip().casefold()
        if normalized_tone not in {"ready", "warning", "error", "info", "muted"}:
            normalized_tone = "muted"
        normalized_facts = tuple(
            (str(text).strip(), str(fact_tone or "muted").strip().casefold())
            for text, fact_tone in facts
            if str(text).strip()
        )[:3]
        previous_runtime_tone = self._runtime_tone
        self._runtime_tone = normalized_tone
        if self.runtime_status.text() != status:
            self.runtime_status.setText(status)
        self.runtime_status.setAccessibleName(f"Runtime status: {status}")
        if str(self.runtime_status.property("tone") or "") != normalized_tone:
            self.runtime_status.setProperty("tone", normalized_tone)
            self.runtime_status.style().unpolish(self.runtime_status)
            self.runtime_status.style().polish(self.runtime_status)
        normalized_detail = str(detail or "").strip()
        if self.runtime_detail.text() != normalized_detail:
            self.runtime_detail.setText(normalized_detail)
        self.runtime_detail.setVisible(not normalized_facts)
        for index, (icon, fact_label) in enumerate(
            zip(self.runtime_fact_icons, self.runtime_fact_labels)
        ):
            fact_widget = fact_label.parentWidget()
            if index < len(normalized_facts):
                text, fact_tone = normalized_facts[index]
                if fact_label.text() != text:
                    fact_label.setText(text)
                fact_label.setAccessibleName(text)
                if str(fact_label.property("tone") or "") != fact_tone:
                    fact_label.setProperty("tone", fact_tone)
                    fact_label.style().unpolish(fact_label)
                    fact_label.style().polish(fact_label)
                fact_widget.show()
            else:
                fact_label.clear()
                fact_widget.hide()
        self.runtime_facts_widget.setVisible(bool(normalized_facts))
        self.runtime_settings.setVisible(
            normalized_tone in {"warning", "error", "muted"}
        )
        if previous_runtime_tone != normalized_tone:
            self.refresh_icons(self._icon_theme)

    def set_stage_details(self, details: Mapping[str, str]) -> None:
        """Bind stage copy to the current prepared or captured runtime truth."""

        unknown = tuple(key for key in details if key not in self.stage_labels)
        if unknown:
            raise ValueError(f"unknown workspace stage detail keys: {unknown!r}")
        for key, detail in details.items():
            self.stage_labels[key].set_detail(detail)

    def set_next_action(self, value: NextActionPresentation) -> None:
        """Present one safe next action without changing command eligibility."""

        if not isinstance(value, NextActionPresentation):
            raise TypeError("value must be NextActionPresentation")
        self.next_action_title.setText(value.label)
        self.next_action_detail.setText(value.detail)
        self.next_action_frame.setProperty("tone", value.tone)
        self.next_action_frame.setAccessibleDescription(
            f"{value.label}. {value.detail}"
        )
        self.next_action_frame.style().unpolish(self.next_action_frame)
        self.next_action_frame.style().polish(self.next_action_frame)

    def _open_selected_page(self, index: QtCore.QModelIndex) -> None:
        if self._page_id_role is None:
            return
        page_id = str(index.data(self._page_id_role) or "").strip()
        if page_id:
            self.page_editor_requested.emit(page_id)

    def _open_current_page(self) -> None:
        index = self.page_list.currentIndex()
        if index.isValid():
            self._open_selected_page(index)

    def _retry_selected(self) -> None:
        index = self.error_list.currentIndex()
        if not index.isValid():
            return
        model = self.error_list.model()
        retry_role = None
        if model is not None:
            for role, name in model.roleNames().items():
                if bytes(name).decode("utf-8", "ignore") == "retryAction":
                    retry_role = role
                    break
        action = str(index.data(retry_role) if retry_role is not None else "").strip()
        if action:
            self.retry_requested.emit(action)

    def _refresh_error_count(self, *_args: object) -> None:
        model = self.error_list.model()
        count = model.rowCount() if model is not None else 0
        self.error_count.setText(str(count))
        self.error_count.setProperty("tone", "error" if count else "muted")
        self.error_count.style().unpolish(self.error_count)
        self.error_count.style().polish(self.error_count)
        self.recovery_banner.setVisible(
            bool(count or self._application_recovery is not None)
        )
        if count and model is not None:
            self.recovery_retry_button.show()
            index = model.index(0, 0)
            self.error_list.setCurrentIndex(index)
            roles = {
                bytes(name).decode("utf-8", "ignore"): role
                for role, name in model.roleNames().items()
            }

            def role_text(name: str) -> str:
                role = roles.get(name)
                return str(index.data(role) if role is not None else "").strip()

            page_id = role_text("pageId")
            stage = role_text("ownerStage").replace("_", " ").strip().title()
            message = role_text("message")
            detail = role_text("detail")
            retry_action = role_text("retryAction").casefold()
            safe = bool(
                index.data(roles["priorStateSafe"])
                if "priorStateSafe" in roles
                else False
            )
            page_label = Path(page_id).stem if page_id else "A page"
            self.recovery_title.setText(
                f"Page {page_label} stopped safely"
                if safe and page_id
                else message or "A page needs attention"
            )
            self.recovery_detail.setText(
                " ".join(value for value in (message, detail) if value)
                or "The previous page revision remains intact. Retry from the owning stage when ready."
            )
            if retry_action == "retry_page" and stage:
                retry_label = f"Retry from {stage}"
            elif retry_action == "relink":
                retry_label = "Relink provider"
            elif retry_action:
                retry_label = "Retry selected action"
            else:
                retry_label = "Review issue"
            self.recovery_retry_button.setText(retry_label)
            self.recovery_retry_button.setAccessibleName(retry_label)
            self.recovery_retry_button.setAccessibleDescription(
                detail or message or "Review the selected structured error"
            )
        elif self._application_recovery is not None:
            title, detail = self._application_recovery
            self.recovery_title.setText(title)
            self.recovery_detail.setText(detail)
            self.recovery_retry_button.hide()

    def _refresh_queue_stats(self, *_args: object) -> None:
        model = self._page_proxy.sourceModel()
        counts = {"done": 0, "active": 0, "queued": 0, "errors": 0}
        if model is not None:
            for row in range(model.rowCount()):
                index = model.index(row, 0)
                tone = str(index.data(int(PageRole.WORKSPACE_STATUS_TONE)) or "").casefold()
                label = str(index.data(int(PageRole.WORKSPACE_STATUS_LABEL)) or "").casefold()
                if tone == "error" or "error" in label or "failed" in label:
                    counts["errors"] += 1
                elif tone in {"editing", "info"} or "running" in label:
                    counts["active"] += 1
                elif tone == "ready" and not bool(index.data(int(PageRole.NEEDS_REVIEW))):
                    counts["done"] += 1
                else:
                    counts["queued"] += 1
        for key, value in counts.items():
            self.run_stat_values[key].setText(str(value))
        self._resize_page_queue()

    def _resize_page_queue(self) -> None:
        rows = self._page_proxy.rowCount()
        height = max(1, rows) * 53 + 2
        self.page_list.setMinimumHeight(height)
        self.page_list.setMaximumHeight(height)
