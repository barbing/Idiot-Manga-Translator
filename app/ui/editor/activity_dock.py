# -*- coding: utf-8 -*-
"""Four-facet Activity Dock for the native Page Editor."""
from __future__ import annotations

from collections.abc import Iterable

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.design_system.icons import hybrid_icon
from app.ui.ui_contract import ACTIVITY_FACET_IDS, ActivityDockBounds, LayoutMode


class _Facet(QtWidgets.QFrame):
    def __init__(
        self,
        facet_id: str,
        title: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if facet_id not in ACTIVITY_FACET_IDS:
            raise ValueError(f"unsupported Activity facet: {facet_id!r}")
        super().__init__(parent)
        self.facet_id = facet_id
        self.setObjectName(f"activityFacet_{facet_id}")
        self.setProperty("role", "facet")
        self.setProperty("activityFacet", facet_id)
        self.setAccessibleName(title)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        self.root = QtWidgets.QVBoxLayout(self)
        self.root.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetNoConstraint)
        self.root.setContentsMargins(12, 10, 12, 10)
        self.root.setSpacing(6)
        header = QtWidgets.QHBoxLayout()
        self.header_layout = header
        title_label = QtWidgets.QLabel(title.upper())
        title_label.setProperty("role", "eyebrow")
        self.status = QtWidgets.QLabel("Ready")
        self.status.setObjectName("statusPill")
        self.status.setProperty("role", "status-pill")
        self.status.setProperty("tone", "muted")
        header.addWidget(title_label)
        header.addStretch(1)
        header.addWidget(self.status)
        self.root.addLayout(header)

    def set_status(self, label: str, tone: str) -> None:
        label_changed = self.status.text() != label
        tone_changed = self.status.property("tone") != tone
        accessible_name = f"{self.accessibleName()} status: {label}"
        if label_changed:
            self.status.setText(label)
        if self.status.accessibleName() != accessible_name:
            self.status.setAccessibleName(accessible_name)
        if tone_changed:
            self.status.setProperty("tone", tone)
            self.status.style().unpolish(self.status)
            self.status.style().polish(self.status)


class _ProjectFacet(_Facet):
    open_hub_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("project", "Project Monitor", parent)
        self.project_name = QtWidgets.QLabel("No project")
        self.project_name.setProperty("role", "section")
        self.root.addWidget(self.project_name)
        metadata = QtWidgets.QHBoxLayout()
        language_group = QtWidgets.QWidget()
        language_layout = QtWidgets.QVBoxLayout(language_group)
        language_layout.setContentsMargins(0, 0, 0, 0)
        language_layout.setSpacing(2)
        language_title = QtWidgets.QLabel("LANGUAGE PAIR")
        language_title.setProperty("role", "eyebrow")
        self.language_pair = QtWidgets.QLabel("—")
        self.language_pair.setProperty("role", "secondary")
        language_layout.addWidget(language_title)
        language_layout.addWidget(self.language_pair)
        metadata.addWidget(language_group, 1)
        pages_group = QtWidgets.QWidget()
        pages_layout = QtWidgets.QVBoxLayout(pages_group)
        pages_layout.setContentsMargins(0, 0, 0, 0)
        pages_layout.setSpacing(2)
        pages_title = QtWidgets.QLabel("PAGES")
        pages_title.setProperty("role", "eyebrow")
        self.total_pages = QtWidgets.QLabel("0")
        self.total_pages.setProperty("role", "metric")
        pages_layout.addWidget(pages_title)
        pages_layout.addWidget(self.total_pages)
        metadata.addWidget(pages_group)
        self.root.addLayout(metadata)
        self.counts = QtWidgets.QGridLayout()
        self.counts.setContentsMargins(0, 4, 0, 4)
        self._count_values: dict[str, QtWidgets.QLabel] = {}
        for index, (key, label) in enumerate(
            (("done", "Done"), ("active", "Active"), ("queued", "Queued"), ("error", "Error"))
        ):
            value = QtWidgets.QLabel(f"0  {label}")
            value.setProperty("state", key)
            value.setAccessibleName(f"0 pages {label.lower()}")
            self.counts.addWidget(value, index // 2, index % 2)
            self._count_values[key] = value
        self.root.addLayout(self.counts)
        self.root.addStretch(1)
        footer = QtWidgets.QHBoxLayout()
        self.checkpoint = QtWidgets.QLabel("Checkpoint not available")
        self.checkpoint.setProperty("role", "secondary")
        footer.addWidget(self.checkpoint)
        footer.addStretch(1)
        button = QtWidgets.QPushButton("Open Hub")
        button.setProperty("role", "command")
        button.setProperty("variant", "quiet")
        button.setAccessibleName("Open Project Hub")
        button.clicked.connect(self.open_hub_requested)
        footer.addWidget(button)
        self.root.addLayout(footer)

    def update_summary(
        self,
        *,
        project_name: str,
        language_pair: str,
        done: int,
        active: int,
        queued: int,
        error: int,
        checkpoint_label: str,
        status_label: str,
        tone: str,
    ) -> None:
        self.project_name.setText(project_name)
        self.language_pair.setText(language_pair)
        for key, count in (
            ("done", done), ("active", active), ("queued", queued), ("error", error)
        ):
            label = key.title()
            self._count_values[key].setText(f"{int(count)}  {label}")
            self._count_values[key].setAccessibleName(f"{int(count)} pages {key}")
        self.total_pages.setText(str(max(0, int(done) + int(active) + int(queued))))
        self.checkpoint.setText(checkpoint_label)
        self.set_status(status_label, tone)


class _RunFacet(_Facet):
    open_workspace_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("run", "Active Run", parent)
        self.page_name = QtWidgets.QLabel("No active run")
        self.page_name.setProperty("role", "section")
        self.root.addWidget(self.page_name)
        metrics = QtWidgets.QHBoxLayout()
        self.stage = self._metric(metrics, "CURRENT STAGE", "Ready")
        self.parent_value = self._metric(metrics, "CURRENT PARENT", "—")
        self.eta = self._metric(metrics, "ETA", "—")
        self.root.addLayout(metrics)
        progress_row = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setAccessibleName("Active run progress")
        self.percent = QtWidgets.QLabel("0%")
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.percent)
        self.root.addLayout(progress_row)
        self.commit_detail = QtWidgets.QLabel("No page transaction in progress")
        self.commit_detail.setProperty("role", "secondary")
        self.root.addWidget(self.commit_detail)
        button = QtWidgets.QPushButton("Open Workspace")
        button.setProperty("role", "command")
        button.setProperty("variant", "quiet")
        button.clicked.connect(self.open_workspace_requested)
        self.header_layout.addWidget(button)

    @staticmethod
    def _metric(
        layout: QtWidgets.QHBoxLayout,
        title: str,
        value: str,
    ) -> QtWidgets.QLabel:
        group = QtWidgets.QWidget()
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(0, 0, 0, 0)
        label = QtWidgets.QLabel(title)
        label.setProperty("role", "eyebrow")
        result = QtWidgets.QLabel(value)
        result.setProperty("role", "metric")
        group_layout.addWidget(label)
        group_layout.addWidget(result)
        layout.addWidget(group, 1)
        return result

    def update_summary(
        self,
        *,
        page_name: str,
        stage: str,
        parent: str,
        eta: str,
        percent: int,
        detail: str,
        status_label: str,
        tone: str,
    ) -> None:
        value = max(0, min(100, int(percent)))
        self.page_name.setText(page_name)
        self.stage.setText(stage)
        self.parent_value.setText(parent)
        self.eta.setText(eta)
        self.progress.setValue(value)
        self.percent.setText(f"{value}%")
        self.commit_detail.setText(detail)
        self.set_status(status_label, tone)


class _RuntimeFacet(_Facet):
    settings_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("runtime", "Provider & Runtime", parent)
        self._last_presentation: tuple[object, ...] | None = None
        self.root.setContentsMargins(10, 6, 10, 6)
        self.root.setSpacing(3)
        self.observation = QtWidgets.QLabel("Observed from the current runtime")
        self.observation.setProperty("role", "secondary")
        self.root.addWidget(self.observation)
        self.rows = QtWidgets.QTableWidget(0, 3)
        self.rows.setObjectName("runtimeObservation")
        self.rows.setProperty("role", "runtime-grid")
        self.rows.setAccessibleName("Read-only provider and runtime information")
        self.rows.horizontalHeader().hide()
        self.rows.verticalHeader().hide()
        self.rows.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.rows.setShowGrid(False)
        self.rows.setWordWrap(True)
        self.rows.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.rows.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.rows.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.rows.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.rows.setMinimumHeight(0)
        self.rows.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        self._row_resize_timer = QtCore.QTimer(self)
        self._row_resize_timer.setSingleShot(True)
        self._row_resize_timer.timeout.connect(self._resize_rows_to_contents)
        self.root.addWidget(self.rows, 1)
        self.observation.setVisible(False)
        settings = QtWidgets.QPushButton("Open Settings")
        settings.setProperty("role", "command")
        settings.setProperty("variant", "quiet")
        settings.setAccessibleDescription(
            "Provider configuration is available only in Settings"
        )
        settings.clicked.connect(self.settings_requested)
        self.header_layout.addWidget(settings)

    def update_rows(
        self,
        rows: Iterable[tuple[str, str, str, str, str]],
        *,
        status_label: str,
        tone: str,
        detail: str = "Observed from the current runtime",
    ) -> None:
        stable_rows = tuple(rows)
        presentation = (stable_rows, status_label, tone, detail)
        if presentation == self._last_presentation:
            return
        self.observation.setText(detail)
        self.rows.clear()
        self.rows.setRowCount((len(stable_rows) + 2) // 3)
        for index, (module, backend, device, status, row_tone) in enumerate(stable_rows):
            row, column = divmod(index, 3)
            item = QtWidgets.QTableWidgetItem(
                f"{module} · {device}\n{backend} · {status}"
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, row_tone)
            item.setFlags(
                item.flags()
                & ~QtCore.Qt.ItemFlag.ItemIsEditable
                & ~QtCore.Qt.ItemFlag.ItemIsSelectable
            )
            self.rows.setItem(row, column, item)
        self._resize_rows_to_contents()
        self.set_status(status_label, tone)
        self._last_presentation = presentation

    def _resize_rows_to_contents(self) -> None:
        if not self.rows.rowCount():
            return
        self.rows.resizeRowsToContents()
        useful_height = max(26, self.rows.fontMetrics().lineSpacing() * 2 + 4)
        for row in range(self.rows.rowCount()):
            self.rows.setRowHeight(row, max(useful_height, self.rows.rowHeight(row)))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        # Section widths settle during the parent layout pass.  Re-measure on
        # the next event turn so wrapped backend/status text cannot be clipped.
        # The child timer is destroyed with this facet, preventing a queued
        # callback from touching the table after the editor has closed.
        self._row_resize_timer.start(0)


class _PageFacet(_Facet):
    inspector_requested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("page", "Current Page & Edit", parent)
        identity = QtWidgets.QHBoxLayout()
        self.thumbnail = QtWidgets.QLabel()
        self.thumbnail.setFixedSize(48, 60)
        self.thumbnail.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.thumbnail.setAccessibleName("Current page thumbnail")
        self.thumbnail.setVisible(False)
        identity.addWidget(self.thumbnail)
        identity_text = QtWidgets.QVBoxLayout()
        self.page_name = QtWidgets.QLabel("No page selected")
        self.page_name.setProperty("role", "section")
        identity_text.addWidget(self.page_name)
        self.parent_value = QtWidgets.QLabel("Selected parent: —")
        self.parent_value.setProperty("role", "secondary")
        self.parent_value.setWordWrap(True)
        identity_text.addWidget(self.parent_value)
        identity.addLayout(identity_text, 1)
        self.root.addLayout(identity)
        facts = QtWidgets.QGridLayout()
        self.authority = self._fact(facts, 0, 0, "AUTHORITY", "Automatic")
        self.artifacts = self._fact(facts, 0, 1, "ARTIFACTS", "Original")
        self.root.addLayout(facts)
        self.authority_strip = QtWidgets.QFrame()
        self.authority_strip.setObjectName("activityAuthorityStrip")
        authority_layout = QtWidgets.QHBoxLayout(self.authority_strip)
        authority_layout.setContentsMargins(0, 0, 0, 0)
        authority_layout.setSpacing(0)
        for index, text in enumerate(("Automatic", "Your edit", "Effective result")):
            value = QtWidgets.QLabel(text)
            value.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            value.setProperty(
                "authority",
                "effective" if index == 2 else "user" if index == 1 else "automatic",
            )
            authority_layout.addWidget(value, 1)
        self.root.addWidget(self.authority_strip)
        states = QtWidgets.QGridLayout()
        states.setHorizontalSpacing(4)
        self.preview = self._fact(states, 0, 0, "PREVIEW", "Idle")
        self.freshness = self._fact(states, 0, 1, "FRESHNESS", "Current")
        self.review = self._fact(states, 0, 2, "REVIEW", "No warnings")
        self.workflow = self._fact(states, 0, 3, "WORKFLOW", "Included")
        self.root.addLayout(states)
        self.provenance = QtWidgets.QLabel("Automatic evidence remains immutable")
        self.provenance.setProperty("role", "secondary")
        self.root.addWidget(self.provenance)

    @staticmethod
    def _fact(
        layout: QtWidgets.QGridLayout,
        row: int,
        column: int,
        title: str,
        value: str,
    ) -> QtWidgets.QLabel:
        box = QtWidgets.QWidget()
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.setContentsMargins(0, 2, 0, 2)
        label = QtWidgets.QLabel(title)
        label.setProperty("role", "eyebrow")
        result = QtWidgets.QLabel(value)
        result.setProperty("role", "metric")
        result.setWordWrap(True)
        box_layout.addWidget(label)
        box_layout.addWidget(result)
        layout.addWidget(box, row, column)
        return result

    def update_summary(
        self,
        *,
        page_name: str,
        parent: str,
        authority: str,
        artifacts: str,
        preview: str,
        freshness: str,
        review: str,
        workflow: str,
        provenance: str,
        status_label: str,
        tone: str,
        thumbnail_path: str | None = None,
    ) -> None:
        self.page_name.setText(page_name)
        self.parent_value.setText(f"Selected parent: {parent}")
        thumbnail_pixmap = QtGui.QPixmap(str(thumbnail_path or ""))
        if thumbnail_pixmap.isNull():
            self.thumbnail.clear()
            self.thumbnail.setVisible(False)
        else:
            self.thumbnail.setPixmap(
                thumbnail_pixmap.scaled(
                    self.thumbnail.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )
            self.thumbnail.setVisible(True)
        for widget, value in (
            (self.authority, authority),
            (self.artifacts, artifacts),
            (self.preview, preview),
            (self.freshness, freshness),
            (self.review, review),
            (self.workflow, workflow),
            (self.provenance, provenance),
        ):
            widget.setText(value)
        self.set_status(status_label, tone)


class ActivityDock(QtWidgets.QWidget):
    """Responsive Activity deck with exactly four canonical facets."""

    expanded_changed = QtCore.Signal(bool)
    tab_changed = QtCore.Signal(str)
    hub_requested = QtCore.Signal()
    workspace_requested = QtCore.Signal()
    settings_requested = QtCore.Signal()
    inspector_requested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("activityDock")
        self.setProperty("role", "dock")
        self.setAccessibleName("Activity Dock")
        # The dock must remain owned by the editor viewport.  Its contents may
        # scroll at large application font scales, but they must never force a
        # top-level window wider than the requested screen geometry.
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._layout_mode: LayoutMode | None = None
        self._icon_theme = "dark"
        self._height_bounds = ActivityDockBounds(
            min=320,
            preferred=320,
            max=360,
            resizable=True,
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetNoConstraint)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        bar = QtWidgets.QFrame()
        self.bar = bar
        bar.setObjectName("activityBar")
        bar.setProperty("role", "dock-bar")
        bar_layout = QtWidgets.QHBoxLayout(bar)
        bar_layout.setContentsMargins(16, 5, 8, 5)
        bar_layout.setSpacing(6)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setProperty("role", "command")
        self.toggle.setProperty("variant", "secondary")
        self.toggle.setText("Activity")
        self.toggle.setIcon(hybrid_icon("caret-down"))
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.clicked.connect(self._toggle_expanded)
        bar_layout.addWidget(self.toggle)

        self.stage_chip = QtWidgets.QLabel("Ready")
        self.stage_chip.setObjectName("activityStageChip")
        self.stage_chip.setProperty("role", "status-pill")
        self.stage_chip.setProperty("tone", "info")
        self.stage_chip.setAccessibleName("Current stage")
        self.stage_chip.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.stage_chip.setMinimumWidth(93)
        self.stage_chip.setMaximumWidth(96)
        bar_layout.addWidget(self.stage_chip)

        self.run_monitor = QtWidgets.QFrame()
        self.run_monitor.setObjectName("activityRunMonitor")
        monitor_layout = QtWidgets.QHBoxLayout(self.run_monitor)
        monitor_layout.setContentsMargins(10, 3, 10, 3)
        monitor_layout.setSpacing(8)
        self.run_project = QtWidgets.QLabel("No project\n0 / 0")
        self.run_project.setObjectName("activityRunProject")
        self.run_project.setProperty("role", "secondary")
        self.run_project.setAccessibleName("Project progress")
        monitor_layout.addWidget(self.run_project)
        self.run_progress = QtWidgets.QProgressBar()
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(0)
        self.run_progress.setTextVisible(False)
        self.run_progress.setAccessibleName("Current project progress")
        monitor_layout.addWidget(self.run_progress, 1)
        self.run_percent = QtWidgets.QLabel("0%")
        self.run_percent.setProperty("role", "metric")
        monitor_layout.addWidget(self.run_percent)
        self.run_eta = QtWidgets.QLabel("—")
        self.run_eta.setObjectName("activityRunEta")
        self.run_eta.setProperty("role", "secondary")
        monitor_layout.addWidget(self.run_eta)
        bar_layout.addWidget(self.run_monitor, 1)

        self.page_identity = QtWidgets.QLabel("PAGE  —")
        self.page_identity.setObjectName("activityPageIdentity")
        self.page_identity.setProperty("role", "secondary")
        self.page_identity.setAccessibleName("Current page and selected parent")
        bar_layout.addWidget(self.page_identity)
        self.page_status = QtWidgets.QLabel("Ready")
        self.page_status.setObjectName("activityPageStatus")
        self.page_status.setProperty("role", "status-pill")
        self.page_status.setProperty("tone", "muted")
        self.page_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.page_status.setMaximumWidth(116)
        bar_layout.addWidget(self.page_status)
        bar_layout.addStretch(1)

        self.tabs = QtWidgets.QButtonGroup(self)
        self.tabs.setExclusive(True)
        self.tab_frame = QtWidgets.QFrame()
        self.tab_frame.setObjectName("activityTabs")
        self.tab_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        tab_layout = QtWidgets.QHBoxLayout(self.tab_frame)
        tab_layout.setContentsMargins(2, 2, 2, 2)
        tab_layout.setSpacing(1)
        self._tab_icon_names = {
            "overview": "grid",
            "history": "history",
            "warnings": "warning",
            "cleanup": "cleanup",
        }
        self.tab_buttons: dict[str, QtWidgets.QToolButton] = {}
        for index, tab in enumerate(("overview", "history", "warnings", "cleanup")):
            button = QtWidgets.QToolButton()
            button.setProperty("role", "command")
            button.setProperty("variant", "quiet")
            button.setText(tab.title())
            button.setIcon(hybrid_icon(self._tab_icon_names[tab]))
            button.setToolButtonStyle(
                QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            )
            button.setCheckable(True)
            button.setProperty("activityTab", tab)
            if index == 0:
                button.setChecked(True)
            button.clicked.connect(
                lambda _checked=False, value=tab: self._select_tab(value)
            )
            self.tabs.addButton(button, index)
            self.tab_buttons[tab] = button
            tab_layout.addWidget(button)
        bar_layout.addWidget(self.tab_frame)
        self.summary = QtWidgets.QLabel("Project ready · No active run", self)
        self.summary.setVisible(False)
        root.addWidget(bar)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.setObjectName("activityStack")
        self.stack.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.overview_scroll = QtWidgets.QScrollArea()
        self.overview_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.overview_scroll.setWidgetResizable(True)
        self.overview_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.overview_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.overview_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.overview = QtWidgets.QWidget()
        self.overview.setObjectName("activityOverview")
        self.overview.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        self.overview_layout = QtWidgets.QGridLayout(self.overview)
        self.overview_layout.setSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetNoConstraint
        )
        self.overview_layout.setContentsMargins(12, 9, 12, 11)
        self.overview_layout.setHorizontalSpacing(10)
        self.overview_layout.setVerticalSpacing(10)
        self.project_facet = _ProjectFacet()
        self.run_facet = _RunFacet()
        self.runtime_facet = _RuntimeFacet()
        self.page_facet = _PageFacet()
        self.project_facet.open_hub_requested.connect(self.hub_requested)
        self.run_facet.open_workspace_requested.connect(self.workspace_requested)
        self.runtime_facet.settings_requested.connect(self.settings_requested)
        self.page_facet.inspector_requested.connect(self.inspector_requested)
        self._install_overview_grid(composite=True)
        self.overview_scroll.setWidget(self.overview)
        self.stack.addWidget(self.overview_scroll)
        self.history_view = self._list_page("Project edit and artifact history")
        self.warnings_view = self._list_page("Warnings and recovery actions")
        self.cleanup_view = self._list_page(
            "Manual cleanup is contextual. Open the Cleanup inspector to review it."
        )
        self.stack.addWidget(self.history_view)
        self.stack.addWidget(self.warnings_view)
        self.stack.addWidget(self.cleanup_view)
        root.addWidget(self.stack, 1)

    @staticmethod
    def _list_page(empty_text: str) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        label = QtWidgets.QLabel(empty_text)
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setProperty("role", "secondary")
        layout.addWidget(label, 1)
        return page

    def set_layout_mode(self, mode: LayoutMode) -> None:
        if not isinstance(mode, LayoutMode):
            raise TypeError("mode must be a LayoutMode")
        previous_composite = (
            self._layout_mode is None
            or (
                self._layout_mode.composition_tier != "narrow"
                and not self._layout_mode.accessible_reflow
            )
        )
        next_composite = (
            mode.composition_tier != "narrow"
            and not mode.accessible_reflow
        )
        self._layout_mode = mode
        dense_bar = bool(mode.accessible_reflow or mode.width_tier == "narrow")
        self.run_monitor.setVisible(not dense_bar)
        self.page_identity.setVisible(not dense_bar)
        for button in self.tab_buttons.values():
            button.setToolButtonStyle(
                QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
                if dense_bar
                else QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            )
        self._set_accessible_vertical_overflow(
            mode.accessible_reflow or mode.short_viewport
        )
        if previous_composite != next_composite:
            self._install_overview_grid(composite=next_composite)
        else:
            self._refresh_overview_minimum_width(composite=next_composite)

    def _set_accessible_vertical_overflow(self, enabled: bool) -> None:
        policy = (
            QtWidgets.QSizePolicy.Policy.Preferred
            if enabled
            else QtWidgets.QSizePolicy.Policy.Ignored
        )
        constraint = (
            QtWidgets.QLayout.SizeConstraint.SetMinimumSize
            if enabled
            else QtWidgets.QLayout.SizeConstraint.SetNoConstraint
        )
        self.overview.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            policy,
        )
        self.overview_layout.setSizeConstraint(constraint)
        for facet in (
            self.project_facet,
            self.run_facet,
            self.runtime_facet,
            self.page_facet,
        ):
            facet.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Preferred,
                policy,
            )

    def _install_overview_grid(self, *, composite: bool) -> None:
        for widget in (
            self.project_facet,
            self.run_facet,
            self.runtime_facet,
            self.page_facet,
        ):
            self.overview_layout.removeWidget(widget)
        for column in range(4):
            self.overview_layout.setColumnStretch(column, 0)
        for row in range(2):
            self.overview_layout.setRowStretch(row, 0)
            self.overview_layout.setRowMinimumHeight(row, 0)
        if composite:
            # This is the accepted Activity composition: project and page
            # context remain stable at the sides while run/runtime share the
            # central column.  It avoids the four equally narrow cards that
            # caused the prototype crowding defect.
            self.overview_layout.addWidget(self.project_facet, 0, 0, 2, 1)
            self.overview_layout.addWidget(self.run_facet, 0, 1)
            self.overview_layout.addWidget(self.runtime_facet, 1, 1)
            self.overview_layout.addWidget(self.page_facet, 0, 2, 2, 1)
            self.overview_layout.setColumnStretch(0, 3)
            self.overview_layout.setColumnStretch(1, 5)
            self.overview_layout.setColumnStretch(2, 4)
            self.overview_layout.setRowMinimumHeight(0, 166)
            self.overview_layout.setRowMinimumHeight(1, 124)
            self.overview_layout.setRowStretch(0, 4)
            self.overview_layout.setRowStretch(1, 3)
        else:
            # Narrow windows use a 2x2 deck.  Horizontal and vertical scroll
            # remain available when accessibility scaling makes even this
            # composition larger than its viewport.
            self.overview_layout.addWidget(self.project_facet, 0, 0)
            self.overview_layout.addWidget(self.run_facet, 0, 1)
            self.overview_layout.addWidget(self.runtime_facet, 1, 0)
            self.overview_layout.addWidget(self.page_facet, 1, 1)
            for column in range(2):
                self.overview_layout.setColumnStretch(column, 1)
            for row in range(2):
                self.overview_layout.setRowStretch(row, 1)
        self._refresh_overview_minimum_width(composite=composite)

    def _refresh_overview_minimum_width(self, *, composite: bool) -> None:
        self.overview_layout.activate()
        useful_width = self.overview_layout.minimumSize().width()
        design_floor = 980 if composite else 700
        self.overview.setMinimumWidth(max(design_floor, useful_width))

    def set_summary(self, text: str) -> None:
        self.summary.setText(text)
        self.stage_chip.setText(self.run_facet.stage.text())
        self.stage_chip.setAccessibleDescription(
            f"Current stage {self.run_facet.stage.text()}"
        )
        self.stage_chip.setProperty("tone", self.run_facet.status.property("tone"))
        self.stage_chip.style().unpolish(self.stage_chip)
        self.stage_chip.style().polish(self.stage_chip)

        counts: dict[str, int] = {}
        for key, label in self.project_facet._count_values.items():
            try:
                counts[key] = int(label.text().split(maxsplit=1)[0])
            except (TypeError, ValueError, IndexError):
                counts[key] = 0
        # Error is a review state that can overlap the queued/active totals.
        # The prototype's page count is the workflow total, not a sum of every
        # status badge.
        total = (
            counts.get("done", 0)
            + counts.get("active", 0)
            + counts.get("queued", 0)
        )
        self.run_project.setText(
            f"{self.project_facet.project_name.text()}\n"
            f"{counts.get('done', 0)} / {total}"
        )
        percent = self.run_facet.progress.value()
        self.run_progress.setValue(percent)
        self.run_percent.setText(f"{percent}%")
        self.run_eta.setText(self.run_facet.eta.text())

        page_name = self.page_facet.page_name.text()
        selected_parent = self.page_facet.parent_value.text().removeprefix(
            "Selected parent: "
        )
        self.page_identity.setText(f"PAGE  {page_name}")
        self.page_identity.setAccessibleDescription(
            f"Page {page_name}; selected parent {selected_parent}"
        )
        self.page_status.setText(self.page_facet.status.text())
        self.page_status.setProperty("tone", self.page_facet.status.property("tone"))
        self.page_status.style().unpolish(self.page_status)
        self.page_status.style().polish(self.page_status)
        review = self.page_facet.review.text().strip().casefold()
        warning_count = 0
        if "warning" in review:
            try:
                warning_count = int(review.split(maxsplit=1)[0])
            except (TypeError, ValueError, IndexError):
                warning_count = 1
        self.tab_buttons["warnings"].setText(
            f"Warnings {warning_count}" if warning_count else "Warnings"
        )

    @property
    def expanded(self) -> bool:
        return self.toggle.isChecked()

    @property
    def collapsed_height(self) -> int:
        return max(1, self.bar.sizeHint().height())

    def set_height_bounds(self, bounds: ActivityDockBounds) -> None:
        if not isinstance(bounds, ActivityDockBounds):
            raise TypeError("bounds must be ActivityDockBounds")
        self._height_bounds = bounds
        self._apply_expanded_height_policy()

    def _apply_expanded_height_policy(self) -> None:
        if self.expanded:
            # The Hybrid Pro contract sizes the scrollable Activity panel;
            # the persistent tab/summary bar sits above that panel.
            bar_height = self.collapsed_height
            self.setMinimumHeight(self._height_bounds.min + bar_height)
            self.setMaximumHeight(self._height_bounds.max + bar_height)
        else:
            value = self.collapsed_height
            self.setMinimumHeight(value)
            self.setMaximumHeight(value)

    def set_expanded(self, expanded: bool) -> None:
        value = bool(expanded)
        self.toggle.setChecked(value)
        self.stack.setVisible(value)
        self.toggle.setIcon(
            hybrid_icon(
                "caret-down" if value else "caret-up",
                self._icon_theme,
            )
        )
        self._apply_expanded_height_policy()
        self.expanded_changed.emit(value)

    def _toggle_expanded(self, expanded: bool) -> None:
        self.stack.setVisible(bool(expanded))
        self.toggle.setIcon(
            hybrid_icon(
                "caret-down" if expanded else "caret-up",
                self._icon_theme,
            )
        )
        self._apply_expanded_height_policy()
        self.expanded_changed.emit(bool(expanded))

    def refresh_icons(self, theme: str) -> None:
        self._icon_theme = str(theme)
        self.toggle.setIcon(
            hybrid_icon(
                "caret-down" if self.expanded else "caret-up",
                self._icon_theme,
            )
        )
        for tab, button in self.tab_buttons.items():
            button.setIcon(
                hybrid_icon(self._tab_icon_names[tab], self._icon_theme)
            )

    @property
    def selected_tab(self) -> str:
        button = self.tabs.checkedButton()
        return str(button.property("activityTab") or "overview") if button else "overview"

    def select_tab(self, tab: str) -> None:
        self._select_tab(tab)
        for button in self.tabs.buttons():
            button.setChecked(button.property("activityTab") == tab)

    def _select_tab(self, tab: str) -> None:
        tabs = ("overview", "history", "warnings", "cleanup")
        if tab not in tabs:
            raise ValueError(f"unsupported Activity tab: {tab!r}")
        self.stack.setCurrentIndex(tabs.index(tab))
        self.tab_changed.emit(tab)
