# -*- coding: utf-8 -*-
"""Prototype-faithful Project Hub backed by typed project rows."""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.design_system.delegates import ProjectCardDelegate
from app.ui.design_system.icons import hybrid_icon
from app.ui.design_system.tokens import theme_token
from app.ui.viewmodels.project_model import ProjectRole


def _theme() -> str:
    application = QtWidgets.QApplication.instance()
    value = str(application.property("yomiframeTheme") or "dark") if application else "dark"
    return value if value in {"dark", "light"} else "dark"


class _CoverLabel(QtWidgets.QWidget):
    """Image-backed cover cropped like the Hybrid Pro object-fit cover slot."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap = QtGui.QPixmap()
        self.setObjectName("hubProjectCover")
        self.setAccessibleName("Current project page preview")
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumSize(250, 250)
        self.setMaximumWidth(250)

    def set_source(self, path: str) -> None:
        value = str(path or "").strip()
        self._pixmap = QtGui.QPixmap(value) if value and os.path.isfile(value) else QtGui.QPixmap()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(theme_token(_theme(), "surface-control")))
        if self._pixmap.isNull():
            hybrid_icon("open", _theme()).paint(painter, self.rect().adjusted(96, 96, -96, -96))
            return
        target = self.rect()
        scaled = self._pixmap.scaled(
            target.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        source = QtCore.QRect(
            max(0, (scaled.width() - target.width()) // 2),
            max(0, int((scaled.height() - target.height()) * 0.13)),
            target.width(),
            target.height(),
        )
        painter.drawPixmap(target, scaled, source)


class _ProjectFilterProxy(QtCore.QSortFilterProxyModel):
    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._query = ""
        self._attention_only = False
        self._excluded_path = ""
        self.setDynamicSortFilter(True)

    def set_query(self, value: str) -> None:
        query = str(value or "").strip().casefold()
        if query == self._query:
            return
        self._query = query
        self.invalidateRowsFilter()

    def set_attention_only(self, value: bool) -> None:
        active = bool(value)
        if active == self._attention_only:
            return
        self._attention_only = active
        self.invalidateRowsFilter()

    def set_excluded_path(self, value: str) -> None:
        """Hide the featured project from the secondary recent-project grid."""

        path = str(value or "").strip().casefold()
        if path == self._excluded_path:
            return
        self._excluded_path = path
        self.invalidateRowsFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QtCore.QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None:
            return False
        index = model.index(source_row, 0, source_parent)
        name = str(index.data(int(ProjectRole.NAME)) or "").casefold()
        pair = str(index.data(int(ProjectRole.LANGUAGE_PAIR)) or "").casefold()
        status = str(index.data(int(ProjectRole.STATUS_LABEL)) or "").strip().casefold()
        path = str(index.data(int(ProjectRole.PATH)) or "").strip().casefold()
        if self._excluded_path and path == self._excluded_path:
            return False
        if self._query and self._query not in name and self._query not in pair:
            return False
        if self._attention_only and status in {"ready", "complete", "completed"}:
            return False
        return True


class ProjectHubView(QtWidgets.QWidget):
    """Create/open/recover entry surface matching the Hybrid Pro prototype."""

    new_project_requested = QtCore.Signal()
    open_project_requested = QtCore.Signal(str)
    recent_project_requested = QtCore.Signal(str)
    recover_project_requested = QtCore.Signal(str)
    relink_project_requested = QtCore.Signal(str)
    workspace_requested = QtCore.Signal()
    settings_requested = QtCore.Signal()

    PROJECT_PATH_ROLE = int(ProjectRole.PATH)
    PROJECT_RECOVERABLE_ROLE = int(ProjectRole.RECOVERABLE)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("projectHub")
        self.setAccessibleName("Project Hub")
        self._hero_project_path = ""
        self._hero_recoverable = False
        self._attention_only = False
        self._recent_resize_timer = QtCore.QTimer(self)
        self._recent_resize_timer.setSingleShot(True)
        self._recent_resize_timer.timeout.connect(self._resize_recent_projects)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.hub_scroll = QtWidgets.QScrollArea()
        self.hub_scroll.setObjectName("projectHubScroll")
        self.hub_scroll.setWidgetResizable(True)
        self.hub_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.hub_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.hub_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        hub_content = QtWidgets.QWidget()
        hub_content.setObjectName("projectHubScrollContent")
        root = QtWidgets.QVBoxLayout(hub_content)
        root.setContentsMargins(42, 26, 42, 42)
        self.hub_scroll.setWidget(hub_content)
        outer.addWidget(self.hub_scroll)
        root.setSpacing(0)

        heading_row = QtWidgets.QHBoxLayout()
        heading_row.setSpacing(26)
        heading = QtWidgets.QVBoxLayout()
        heading.setSpacing(1)
        eyebrow = QtWidgets.QLabel("PROJECT HUB")
        eyebrow.setProperty("role", "eyebrow")
        title = QtWidgets.QLabel("Continue translating")
        title.setObjectName("surfaceTitle")
        subtitle = QtWidgets.QLabel("Open, recover, and manage projects without losing the last valid revision.")
        subtitle.setWordWrap(True)
        subtitle.setProperty("role", "secondary")
        heading.addWidget(eyebrow)
        heading.addWidget(title)
        heading.addWidget(subtitle)
        heading_row.addLayout(heading, 1)

        heading_actions = QtWidgets.QHBoxLayout()
        heading_actions.setSpacing(8)
        self.open_button = self._button("Open project", variant="secondary")
        self.open_button.clicked.connect(self._choose_project)
        heading_actions.addWidget(self.open_button)
        self.new_button = self._button("New project", variant="primary")
        self.new_button.clicked.connect(self.new_project_requested)
        heading_actions.addWidget(self.new_button)
        heading_row.addLayout(heading_actions)
        heading_row.setAlignment(
            heading_actions,
            QtCore.Qt.AlignmentFlag.AlignBottom,
        )
        root.addLayout(heading_row)
        root.addSpacing(22)

        self.hero = QtWidgets.QFrame()
        self.hero.setObjectName("hubCallout")
        self.hero.setProperty("role", "panel")
        self.hero.setMinimumHeight(356)
        hero_layout = QtWidgets.QHBoxLayout(self.hero)
        hero_layout.setContentsMargins(0, 0, 0, 0)
        hero_layout.setSpacing(0)
        self.hero_cover = _CoverLabel()
        hero_layout.addWidget(self.hero_cover)

        hero_copy = QtWidgets.QWidget()
        hero_copy.setObjectName("hubCalloutCopy")
        copy_layout = QtWidgets.QVBoxLayout(hero_copy)
        copy_layout.setContentsMargins(24, 22, 24, 22)
        copy_layout.setSpacing(10)
        copy_layout.addStretch(1)
        self.recovery_pill = QtWidgets.QToolButton()
        self.recovery_pill.setText("Recovery available")
        self.recovery_pill.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.recovery_pill.setFixedHeight(24)
        self.recovery_pill.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.recovery_pill.setProperty("role", "status-pill")
        self.recovery_pill.setProperty("tone", "warning")
        copy_layout.addWidget(self.recovery_pill, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.hero_title = QtWidgets.QLabel("No project selected")
        self.hero_title.setObjectName("hubProjectTitle")
        copy_layout.addWidget(self.hero_title)
        self.hero_description = QtWidgets.QLabel("Open a project to resume from its last valid page-local checkpoint.")
        self.hero_description.setWordWrap(True)
        self.hero_description.setMaximumWidth(620)
        self.hero_description.setProperty("role", "secondary")
        copy_layout.addWidget(self.hero_description)
        hero_meta = QtWidgets.QHBoxLayout()
        hero_meta.setSpacing(18)
        self.hero_meta_icons: list[tuple[QtWidgets.QLabel, str]] = []
        self.hero_pages = QtWidgets.QLabel("— pages")
        self.hero_pair = QtWidgets.QLabel("Languages unavailable")
        self.hero_updated = QtWidgets.QLabel("Local project")
        for label, icon_name in (
            (self.hero_pages, "file-text"),
            (self.hero_pair, "translate"),
            (self.hero_updated, "clock"),
        ):
            label.setProperty("role", "secondary")
            item = QtWidgets.QWidget()
            item_layout = QtWidgets.QHBoxLayout(item)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(6)
            icon = QtWidgets.QLabel()
            icon.setFixedSize(16, 16)
            icon.setAccessibleName(f"{label.text()} metadata")
            self.hero_meta_icons.append((icon, icon_name))
            item_layout.addWidget(icon)
            item_layout.addWidget(label)
            hero_meta.addWidget(item)
        hero_meta.addStretch(1)
        copy_layout.addLayout(hero_meta)
        hero_actions = QtWidgets.QHBoxLayout()
        hero_actions.setSpacing(8)
        self.resume_button = self._button("Resume in Editor", variant="primary")
        self.resume_button.clicked.connect(self._resume_hero)
        hero_actions.addWidget(self.resume_button)
        self.workspace_button = self._button("Open Workspace", variant="secondary")
        self.workspace_button.clicked.connect(self.workspace_requested)
        hero_actions.addWidget(self.workspace_button)
        hero_actions.addStretch(1)
        copy_layout.addLayout(hero_actions)
        copy_layout.addStretch(1)
        hero_layout.addWidget(hero_copy, 1)

        self.health = QtWidgets.QFrame()
        self.health.setObjectName("hubHealth")
        self.health.setProperty("role", "panel-raised")
        self.health.setMinimumWidth(280)
        self.health.setMaximumWidth(280)
        health_layout = QtWidgets.QVBoxLayout(self.health)
        health_layout.setContentsMargins(24, 24, 24, 24)
        health_layout.setSpacing(9)
        health_layout.addStretch(1)
        health_heading = QtWidgets.QLabel("Project health")
        health_heading.setProperty("role", "secondary")
        health_layout.addWidget(health_heading)
        self.health_status = QtWidgets.QLabel("Open a project")
        self.health_status.setProperty("role", "section")
        health_layout.addWidget(self.health_status)
        self.health_progress = QtWidgets.QProgressBar()
        self.health_progress.setRange(0, 100)
        self.health_progress.setTextVisible(False)
        self.health_progress.setAccessibleName("Project progress")
        health_layout.addWidget(self.health_progress)
        self.health_page_count = QtWidgets.QLabel("No page state loaded")
        self.health_page_count.setProperty("role", "secondary")
        health_layout.addWidget(self.health_page_count)
        self.health_checks: list[tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = []
        for text in ("Checkpoint verified", "Provider connected", "Runtime assets ready"):
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(7)
            icon = QtWidgets.QLabel()
            icon.setFixedSize(15, 15)
            icon.setAccessibleName(f"{text} status")
            label = QtWidgets.QLabel(text)
            label.setObjectName("hubHealthCheck")
            label.setProperty("role", "secondary")
            label.setProperty("readyText", text)
            row_layout.addWidget(icon)
            row_layout.addWidget(label, 1)
            self.health_checks.append((icon, label))
            health_layout.addWidget(row)
        health_layout.addStretch(1)
        hero_layout.addWidget(self.health)
        root.addWidget(self.hero)
        root.addSpacing(26)

        recent_header = QtWidgets.QHBoxLayout()
        recent_copy = QtWidgets.QVBoxLayout()
        recent_copy.setSpacing(3)
        recent_title = QtWidgets.QLabel("Recent projects")
        recent_title.setProperty("role", "section")
        recent_copy.addWidget(recent_title)
        self.recent_subtitle = QtWidgets.QLabel("Local projects and recoverable sessions")
        self.recent_subtitle.setProperty("role", "secondary")
        recent_copy.addWidget(self.recent_subtitle)
        recent_header.addLayout(recent_copy, 1)
        self.attention_button = self._button("Needs attention", variant="quiet")
        self.attention_button.setCheckable(True)
        self.attention_button.clicked.connect(self._toggle_attention_filter)
        recent_header.addWidget(self.attention_button)
        root.addLayout(recent_header)
        root.addSpacing(19)

        self.recent_projects = QtWidgets.QListView()
        self.recent_projects.setObjectName("recentProjects")
        self.recent_projects.setAccessibleName("Recent projects")
        self.recent_projects.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.recent_projects.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.recent_projects.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        self.recent_projects.setWrapping(True)
        self.recent_projects.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.recent_projects.setMovement(QtWidgets.QListView.Movement.Static)
        self.recent_projects.setItemAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        self.recent_projects.setSpacing(0)
        self.recent_projects.setGridSize(QtCore.QSize(670, 167))
        self.recent_projects.setUniformItemSizes(True)
        self.recent_projects.setWordWrap(True)
        self.recent_projects.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.recent_projects.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.recent_projects.setItemDelegate(ProjectCardDelegate(self.recent_projects))
        # The list viewport can resize after the outer page scrollbar appears
        # without resizing ProjectHubView itself.  Track that final viewport
        # width so the responsive card grid is recalculated at the real size.
        self.recent_projects.installEventFilter(self)
        self.recent_projects.viewport().installEventFilter(self)
        self.recent_projects.activated.connect(self._activate_recent)
        self.recent_projects.clicked.connect(self._activate_recent)
        self._recent_proxy = _ProjectFilterProxy(self)
        self._hero_uninspected = False
        root.addWidget(self.recent_projects, 1)

        self.empty_label = QtWidgets.QLabel("No recent projects yet. Start a translation or open an existing project.")
        self.empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setProperty("role", "secondary")
        self.empty_label.setVisible(False)
        root.addWidget(self.empty_label, 1)

        # Compatibility-only controls retained for existing typed updates.
        self.runtime_pill = QtWidgets.QLabel("Runtime checking", self)
        self.runtime_pill.hide()
        self.readiness_summary = QtWidgets.QLabel("", self)
        self.readiness_summary.hide()
        self.runtime_rows = QtWidgets.QTreeWidget(self)
        self.runtime_rows.hide()
        self.open_settings_button = QtWidgets.QPushButton("Open Settings", self)
        self.open_settings_button.clicked.connect(self.settings_requested)
        self.open_settings_button.hide()
        self.recent_filter = QtWidgets.QLineEdit(self)
        self.recent_filter.textChanged.connect(self._recent_proxy.set_query)
        self.recent_filter.hide()
        self.recover_button = QtWidgets.QPushButton("Recover checkpoint", self)
        self.recover_button.clicked.connect(self._choose_recovery)
        self.recover_button.hide()

        self.refresh_icons(_theme())
        self._sync_current_project()

    @staticmethod
    def _button(text: str, *, variant: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setProperty("role", "command")
        button.setProperty("variant", variant)
        button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        button.setAccessibleName(text)
        return button

    def refresh_icons(self, theme: str) -> None:
        self.open_button.setIcon(hybrid_icon("open", theme, secondary=True))
        self.new_button.setIcon(hybrid_icon("new", theme, active=True))
        self.resume_button.setIcon(hybrid_icon("arrow-right", theme, active=True))
        self.resume_button.setLayoutDirection(QtCore.Qt.LayoutDirection.RightToLeft)
        self.workspace_button.setIcon(QtGui.QIcon())
        self.attention_button.setIcon(hybrid_icon("filter", theme))
        self.recovery_pill.setIcon(hybrid_icon("warning", theme))
        for icon, icon_name in self.hero_meta_icons:
            icon.setPixmap(hybrid_icon(icon_name, theme).pixmap(QtCore.QSize(16, 16)))
        ready = hybrid_icon("success", theme).pixmap(QtCore.QSize(14, 14))
        for icon, _label in self.health_checks:
            icon.setPixmap(ready)

    def set_recent_projects_model(self, model: QtCore.QAbstractItemModel) -> None:
        if not isinstance(model, QtCore.QAbstractItemModel):
            raise TypeError("model must be a QAbstractItemModel")
        self._recent_proxy.setSourceModel(model)
        self.recent_projects.setModel(self._recent_proxy)
        self._update_empty_state()
        for signal in (model.rowsInserted, model.rowsRemoved, model.modelReset, model.dataChanged, model.layoutChanged):
            signal.connect(self._update_empty_state)

    def set_runtime_status(self, *, label: str, tone: str, detail: str, rows: tuple[tuple[str, str, str], ...] = ()) -> None:
        self.runtime_pill.setText(label)
        self.runtime_pill.setProperty("tone", tone)
        self.runtime_pill.setAccessibleName(f"Runtime status: {label}")
        self.readiness_summary.setText(detail)
        self.runtime_rows.clear()
        for module, status, row_tone in rows:
            item = QtWidgets.QTreeWidgetItem((module, status))
            item.setData(1, QtCore.Qt.ItemDataRole.UserRole, row_tone)
            self.runtime_rows.addTopLevelItem(item)
        ready = str(tone).strip().casefold() in {"ready", "success"}
        checks = (True, ready, ready)
        if rows:
            checks = (
                True,
                any("provider" in module.casefold() and row_tone in {"ready", "success"} for module, _status, row_tone in rows),
                any("asset" in module.casefold() and row_tone in {"ready", "success"} for module, _status, row_tone in rows),
            )
        for (icon, check), available in zip(self.health_checks, checks):
            icon.setEnabled(bool(available))
            check.setEnabled(bool(available))
            check.setProperty("tone", "ready" if available else "muted")
        if self._hero_uninspected:
            self._set_uninspected_health_checks()

    def _set_uninspected_health_checks(self) -> None:
        labels = (
            "Checkpoint not inspected",
            "Provider not inspected",
            "Runtime not inspected",
        )
        for (icon, label), text in zip(self.health_checks, labels):
            icon.setEnabled(False)
            label.setEnabled(False)
            label.setText(text)
            label.setProperty("tone", "muted")
            label.setAccessibleName(text)

    def _update_empty_state(self, *_args: object) -> None:
        self._sync_current_project()
        source = self._recent_proxy.sourceModel()
        source_empty = source is None or source.rowCount() == 0
        recent_empty = self._recent_proxy.rowCount() == 0
        self.hero.setVisible(not source_empty)
        self.recent_projects.setVisible(not recent_empty)
        self.empty_label.setVisible(source_empty or recent_empty)
        self.empty_label.setText(
            "No recent projects yet. Start a translation or open an existing project."
            if source_empty
            else "No projects need attention."
            if self._attention_only
            else "No other recent projects."
        )
        self._schedule_recent_projects_resize()

    def _schedule_recent_projects_resize(self) -> None:
        """Coalesce card-grid resizes under a timer owned by this view."""

        self._recent_resize_timer.start(0)

    def _resize_recent_projects(self) -> None:
        model = self.recent_projects.model()
        rows = model.rowCount() if model is not None else 0
        if rows <= 0:
            return
        available = max(1, self.recent_projects.viewport().width())
        # Hybrid Pro uses three columns above 1500 px, two columns down to
        # 1050 px, and one below that breakpoint.  The viewport is narrower
        # than the window by the page padding, so use the corresponding
        # content-width thresholds and size every grid cell from the live
        # viewport instead of the old fixed 670 px card.
        columns = 3 if available >= 1400 else 2 if available >= 1020 else 1
        # QListView's icon layout needs one pixel of trailing slack even when
        # the viewport is an exact multiple of the grid width; without it the
        # second card wraps to the next row at the prototype's 1440 px size.
        grid_width = max(1, available // columns - (1 if columns > 1 else 0))
        grid = QtCore.QSize(grid_width, 167)
        if self.recent_projects.gridSize() != grid:
            self.recent_projects.setGridSize(grid)
        visual_rows = (rows + columns - 1) // columns
        height = max(167, visual_rows * 167)
        if (
            self.recent_projects.minimumHeight() != height
            or self.recent_projects.maximumHeight() != height
        ):
            self.recent_projects.setMinimumHeight(height)
            self.recent_projects.setMaximumHeight(height)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._schedule_recent_projects_resize()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if (
            watched in {self.recent_projects, self.recent_projects.viewport()}
            and event.type() == QtCore.QEvent.Type.Resize
        ):
            self._schedule_recent_projects_resize()
        return super().eventFilter(watched, event)

    def _sync_current_project(self) -> None:
        model = self._recent_proxy.sourceModel()
        index = (
            model.index(0, 0)
            if model is not None and model.rowCount()
            else QtCore.QModelIndex()
        )
        if not index.isValid():
            self._recent_proxy.set_excluded_path("")
            self._hero_project_path = ""
            self._hero_recoverable = False
            self.resume_button.setEnabled(False)
            self.workspace_button.setEnabled(False)
            return
        name = str(index.data(int(ProjectRole.NAME)) or index.data() or "Project")
        path = str(index.data(int(ProjectRole.PATH)) or "").strip()
        self._recent_proxy.set_excluded_path(path)
        pair = str(index.data(int(ProjectRole.LANGUAGE_PAIR)) or "Languages unavailable")
        total = int(index.data(int(ProjectRole.PAGE_COUNT)) or 0)
        complete = int(index.data(int(ProjectRole.COMPLETED_COUNT)) or 0)
        recoverable = bool(index.data(int(ProjectRole.RECOVERABLE)))
        status = str(index.data(int(ProjectRole.STATUS_LABEL)) or "Ready")
        thumbnail = str(index.data(int(ProjectRole.THUMBNAIL_PATH)) or "")
        updated = str(index.data(int(ProjectRole.UPDATED_LABEL)) or "Local project")
        percent = int(round(complete * 100 / total)) if total else 0
        uninspected = bool(
            path
            and total == 0
            and pair.casefold().startswith("open to inspect")
        )
        self._hero_uninspected = uninspected
        self._hero_project_path = path
        self._hero_recoverable = recoverable
        self.hero_title.setText(name)
        self.hero_title.setAccessibleName(f"Current project: {name}")
        self.hero_cover.set_source(thumbnail)
        self.recovery_pill.setText(
            "Not inspected"
            if uninspected
            else "Recovery available"
            if recoverable
            else status
        )
        self.recovery_pill.setProperty(
            "tone",
            "muted"
            if uninspected
            else "warning"
            if recoverable
            else "ready",
        )
        self.recovery_pill.style().unpolish(self.recovery_pill)
        self.recovery_pill.style().polish(self.recovery_pill)
        self.hero_description.setText(
            "Open this project to load its pages, languages, checkpoint, provider, and runtime status."
            if uninspected
            else "An editing session was preserved after the last page-local checkpoint. Automated evidence and your edit ledger are intact."
            if recoverable
            else "The latest durable project revision is ready to continue in the editor."
        )
        self.hero_pages.setText("Pages not loaded" if uninspected else f"{total} pages")
        self.hero_pair.setText(pair)
        self.hero_updated.setText(updated)
        self.health_status.setText(
            "Open to inspect"
            if uninspected
            else "Ready to resume"
            if path
            else "Open a project"
        )
        self.health_progress.setValue(percent)
        self.health_progress.setVisible(not uninspected)
        self.health_page_count.setText(
            "Page count not loaded"
            if uninspected
            else f"{complete} of {total} pages processed"
        )
        self.resume_button.setText(
            "Open project" if uninspected else "Resume in Editor"
        )
        self.resume_button.setAccessibleName(self.resume_button.text())
        self.resume_button.setEnabled(bool(path))
        self.workspace_button.setEnabled(bool(path and not uninspected))
        self.workspace_button.setToolTip(
            "Open the project before entering Workspace" if uninspected else ""
        )
        if uninspected:
            self._set_uninspected_health_checks()
        else:
            for _icon, label in self.health_checks:
                ready_text = str(label.property("readyText") or "")
                if ready_text:
                    label.setText(ready_text)
                    label.setAccessibleName(f"{ready_text} status")
        self.recent_projects.setCurrentIndex(
            self._recent_proxy.index(0, 0)
            if self._recent_proxy.rowCount()
            else QtCore.QModelIndex()
        )

    def _toggle_attention_filter(self, checked: bool) -> None:
        self._attention_only = bool(checked)
        self._recent_proxy.set_attention_only(self._attention_only)
        self.attention_button.setText("Show all" if checked else "Needs attention")
        self.recent_subtitle.setText("Projects requiring attention" if checked else "Local projects and recoverable sessions")
        self._update_empty_state()

    def _resume_hero(self) -> None:
        if not self._hero_project_path:
            return
        if self._hero_recoverable:
            self.recover_project_requested.emit(self._hero_project_path)
        else:
            self.recent_project_requested.emit(self._hero_project_path)

    def _activate_recent(self, index: QtCore.QModelIndex) -> None:
        path = str(index.data(self.PROJECT_PATH_ROLE) or "").strip()
        recoverable = bool(index.data(self.PROJECT_RECOVERABLE_ROLE))
        if not path:
            return
        (self.recover_project_requested if recoverable else self.recent_project_requested).emit(path)

    def _choose_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open YomiFrame project", "", "YomiFrame project (*.json);;All files (*)")
        if path:
            self.open_project_requested.emit(path)

    def _choose_recovery(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Recover YomiFrame checkpoint", "", "YomiFrame project (*.json);;All files (*)")
        if path:
            self.recover_project_requested.emit(path)
