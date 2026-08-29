# -*- coding: utf-8 -*-
"""Registry-backed Settings surface with one SettingsViewModel authority."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping
import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from app.config.module_registry import (
    DEFAULT_MODULE_REGISTRY,
    ModuleSchemaRegistry,
    SettingDefinition,
    SettingValueType,
)
from app.config.provider_profiles import (
    GenerationSettings,
    GGUFProviderOptions,
    OllamaProviderOptions,
    ProviderCapability,
    ProviderKind,
    ProviderProfile,
    ProviderTestStatus,
)
from app.config.settings_contracts import (
    ApplicationPreferences,
    CredentialReference,
    CredentialReferenceKind,
    DEFAULT_SHORTCUT_BINDINGS,
    ModuleConfig,
    RuntimeStatus,
)
from app.pipeline.status_contracts import (
    PipelineLifecycleEvent,
    PipelineRunState,
    PipelineStage,
    PipelineStageEvent,
)
from app.platform_services.runtime_assets import runtime_asset_catalog
from app.ui.design_system.components import (
    HybridComboBox,
    WheelSafeDoubleSpinBox,
    WheelSafeSpinBox,
)
from app.ui.design_system.dialogs import HybridConfirmDialog, HybridDialog
from app.ui.design_system.geometry import MODULE_POLICY_GEOMETRY
from app.ui.design_system.icons import hybrid_icon
from app.ui.presentation import provider_lifecycle_summary
from app.ui.theme import platform_presentation
from app.ui.viewmodels.settings_model import (
    EffectiveRunSummary,
    SettingsViewModel,
    provider_kind_label,
    provider_test_status_label,
)
from app.ui.settings.glossary_view import GlossarySettingsPage
from app.ui.ui_contract import LayoutMode
from app.ui.viewmodels.glossary_model import GlossaryEditorModel


_PROVIDER_ISSUE_LABELS = {
    "local_model_path_required": "Select a local GGUF model",
    "endpoint_required": "Enter a provider endpoint",
    "model_id_required": "Enter a model ID",
    "credential_reference_required": "Enter and test an API credential",
}


class SettingsView(QtWidgets.QWidget):
    """Sole provider-configuration and typed settings presentation."""

    appearance_changed = QtCore.Signal(str, str, int, bool)
    provider_test_requested = QtCore.Signal(str)
    credential_link_requested = QtCore.Signal(str)
    credential_delete_requested = QtCore.Signal(object)
    apply_requested = QtCore.Signal()
    applied = QtCore.Signal()
    cancelled = QtCore.Signal()
    reset_layout_requested = QtCore.Signal()
    glossary_command_requested = QtCore.Signal(object)
    glossary_cancel_requested = QtCore.Signal()
    glossary_open_page_requested = QtCore.Signal(str)
    runtime_verify_requested = QtCore.Signal()
    runtime_asset_action_requested = QtCore.Signal(str)
    runtime_open_folder_requested = QtCore.Signal()
    shortcut_change_requested = QtCore.Signal(str)
    shortcuts_changed = QtCore.Signal(object)

    CATEGORIES = (
        ("general", "General"),
        ("appearance", "Appearance"),
        ("providers", "Providers"),
        ("modules", "Modules"),
        ("runtime", "Runtime assets"),
        ("glossary", "Glossary"),
        ("shortcuts", "Shortcuts"),
    )

    def __init__(
        self,
        view_model: SettingsViewModel | None = None,
        *,
        registry: ModuleSchemaRegistry = DEFAULT_MODULE_REGISTRY,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("settingsSurface")
        self.setAccessibleName("Settings")
        self._view_model = view_model
        self._registry = registry
        self._platform_copy = platform_presentation()
        self._advanced = False
        self._module_controls: dict[str, QtWidgets.QWidget] = {}
        self._profile_refreshing = False
        self._runtime_checking = False
        self._runtime_downloading_asset_id = ""
        self._module_policy_reflow = False
        self._module_policy_compact_identity = False
        self._effective_run_summary: EffectiveRunSummary | None = None
        self._pipeline_lifecycle: PipelineLifecycleEvent | None = None
        self._pipeline_stage: PipelineStageEvent | None = None
        self._project_scope_name = "No project open"
        self._effective_theme: str | None = None
        self._provider_test_messages: dict[str, tuple[str, bool]] = {}
        self._pending_credential_deletions: list[CredentialReference] = []
        self._category_rows: dict[
            str,
            tuple[QtWidgets.QWidget, QtWidgets.QLabel, QtWidgets.QLabel],
        ] = {}

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = QtWidgets.QFrame()
        self.sidebar.setObjectName("settingsSidebar")
        self.sidebar.setProperty("role", "panel")
        self.sidebar.setFixedWidth(265)
        side_layout = QtWidgets.QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(14, 22, 14, 24)
        side_layout.setSpacing(10)
        eyebrow = QtWidgets.QLabel("SETTINGS")
        eyebrow.setProperty("role", "eyebrow")
        eyebrow.setContentsMargins(9, 0, 9, 0)
        title = QtWidgets.QLabel("Configuration")
        title.setProperty("role", "title")
        title.setContentsMargins(9, 0, 9, 0)
        self.sidebar_detail = QtWidgets.QLabel(
            "Application, project, module, and provider scopes remain distinct."
        )
        self.sidebar_detail.setProperty("role", "secondary")
        self.sidebar_detail.setWordWrap(True)
        self.sidebar_detail.setContentsMargins(9, 0, 9, 0)
        side_layout.addWidget(eyebrow)
        side_layout.addWidget(title)
        side_layout.addWidget(self.sidebar_detail)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search settings")
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName("Search settings")
        self.search_action = self.search.addAction(
            hybrid_icon("search"),
            QtWidgets.QLineEdit.ActionPosition.LeadingPosition,
        )
        self.search.textChanged.connect(self._filter_visible_controls)
        search_host = QtWidgets.QWidget()
        search_layout = QtWidgets.QHBoxLayout(search_host)
        search_layout.setContentsMargins(6, 10, 6, 2)
        search_layout.addWidget(self.search)
        side_layout.addWidget(search_host)
        self.category_list = QtWidgets.QListWidget()
        self.category_list.setObjectName("settingsCategories")
        self.category_list.setAccessibleName("Settings categories")
        category_icons = {
            "general": "general",
            "appearance": "appearance",
            "providers": "providers",
            "modules": "modules",
            "runtime": "runtime",
            "glossary": "glossary",
            "shortcuts": "shortcuts",
        }
        for category_id, label in self.CATEGORIES:
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, category_id)
            item.setData(QtCore.Qt.ItemDataRole.AccessibleTextRole, label)
            item.setText("")
            item.setSizeHint(QtCore.QSize(0, 40))
            self.category_list.addItem(item)
            row = QtWidgets.QWidget()
            row.setProperty("settingsCategory", True)
            row.setAccessibleName(label)
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(9, 0, 9, 0)
            row_layout.setSpacing(8)
            icon = QtWidgets.QLabel()
            icon.setObjectName("settingsCategoryIcon")
            icon.setFixedSize(18, 18)
            icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            icon.setProperty("iconName", category_icons[category_id])
            title_label = QtWidgets.QLabel(label)
            title_label.setObjectName("settingsCategoryLabel")
            caret = QtWidgets.QLabel()
            caret.setObjectName("settingsCategoryCaret")
            caret.setFixedSize(14, 18)
            caret.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            row_layout.addWidget(icon)
            row_layout.addWidget(title_label, 1)
            row_layout.addWidget(caret)
            self.category_list.setItemWidget(item, row)
            self._category_rows[category_id] = (row, icon, caret)
        self.category_list.currentRowChanged.connect(self._select_category)
        self.category_list.setFixedHeight(len(self.CATEGORIES) * 42)
        side_layout.addWidget(self.category_list)
        side_layout.addStretch(1)
        level = QtWidgets.QFrame()
        self.settings_level = level
        level.setProperty("role", "panel-raised")
        level_layout = QtWidgets.QHBoxLayout(level)
        level_layout.setContentsMargins(3, 3, 3, 3)
        self.basic_button = QtWidgets.QPushButton("Basic")
        self.advanced_button = QtWidgets.QPushButton("Advanced")
        for button in (self.basic_button, self.advanced_button):
            button.setCheckable(True)
            button.setProperty("role", "command")
            button.setProperty("variant", "quiet")
            level_layout.addWidget(button, 1)
        self.basic_button.setChecked(True)
        self.basic_button.clicked.connect(lambda: self._set_advanced(False))
        self.advanced_button.clicked.connect(lambda: self._set_advanced(True))
        level.setFixedWidth(213)
        side_layout.addWidget(
            level,
            0,
            QtCore.Qt.AlignmentFlag.AlignHCenter,
        )
        side_layout.addStretch(1)
        self.project_scope = QtWidgets.QFrame()
        self.project_scope.setObjectName("settingsProjectScope")
        scope_layout = QtWidgets.QHBoxLayout(self.project_scope)
        scope_layout.setContentsMargins(11, 9, 11, 9)
        scope_layout.setSpacing(9)
        self.project_scope_icon = QtWidgets.QLabel()
        self.project_scope_icon.setFixedSize(18, 18)
        self.project_scope_icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        scope_copy = QtWidgets.QVBoxLayout()
        scope_copy.setSpacing(2)
        scope_title = QtWidgets.QLabel("Project scope")
        scope_title.setProperty("role", "section")
        self.project_scope_value = QtWidgets.QLabel(self._project_scope_name)
        self.project_scope_value.setProperty("role", "secondary")
        scope_copy.addWidget(scope_title)
        scope_copy.addWidget(self.project_scope_value)
        scope_layout.addWidget(self.project_scope_icon)
        scope_layout.addLayout(scope_copy, 1)
        side_layout.addWidget(self.project_scope)
        root.addWidget(self.sidebar)

        content = QtWidgets.QWidget()
        content.setProperty("role", "shell")
        self.content_layout = QtWidgets.QVBoxLayout(content)
        self.content_layout.setContentsMargins(42, 24, 44, 24)
        self.content_layout.setSpacing(12)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.setObjectName("settingsStack")
        self._pages: dict[str, QtWidgets.QWidget] = {}
        self._pages["general"] = self._build_general_page()
        self._pages["appearance"] = self._build_appearance_page()
        self._pages["providers"] = self._build_providers_page()
        self._pages["modules"] = self._build_modules_page()
        self._pages["runtime"] = self._build_runtime_page()
        self.glossary = GlossarySettingsPage()
        self.glossary.command_requested.connect(self.glossary_command_requested)
        self.glossary.cancel_worker_requested.connect(self.glossary_cancel_requested)
        self.glossary.open_stale_page_requested.connect(
            self.glossary_open_page_requested
        )
        self._pages["glossary"] = self.glossary
        self._pages["shortcuts"] = self._build_shortcuts_page()
        for category_id, _label in self.CATEGORIES:
            self.stack.addWidget(self._pages[category_id])
        self.content_layout.addWidget(self.stack, 1)

        footer = QtWidgets.QFrame()
        self.global_footer = footer
        footer.setProperty("role", "dock-bar")
        footer_layout = QtWidgets.QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 8, 10, 8)
        self.pending_label = QtWidgets.QLabel("No pending changes")
        self.pending_label.setProperty("role", "secondary")
        footer_layout.addWidget(self.pending_label)
        footer_layout.addStretch(1)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setProperty("role", "command")
        self.cancel_button.setProperty("variant", "secondary")
        self.cancel_button.clicked.connect(self.cancel_changes)
        footer_layout.addWidget(self.cancel_button)
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.setProperty("role", "command")
        self.apply_button.setProperty("variant", "primary")
        self.apply_button.clicked.connect(self.apply_changes)
        footer_layout.addWidget(self.apply_button)
        self.content_layout.addWidget(footer)
        root.addWidget(content, 1)

        self.category_list.setCurrentRow(1)
        if self._view_model is not None:
            self.refresh_from_model()
        else:
            self._set_bound(False)

    @property
    def view_model(self) -> SettingsViewModel | None:
        return self._view_model

    def bind_view_model(self, view_model: SettingsViewModel) -> None:
        if not isinstance(view_model, SettingsViewModel):
            raise TypeError("view_model must be SettingsViewModel")
        self._view_model = view_model
        self._set_bound(True)
        self.refresh_from_model()

    def _build_general_page(self) -> QtWidgets.QWidget:
        self.general_apply_button = QtWidgets.QPushButton("Saved")
        self.general_apply_button.setObjectName("generalApplyButton")
        self.general_apply_button.setProperty("role", "command")
        self.general_apply_button.setProperty("variant", "secondary")
        self.general_apply_button.setIcon(hybrid_icon("check"))
        self.general_apply_button.setAccessibleName("Apply general settings")
        self.general_apply_button.clicked.connect(self.apply_changes)
        page, layout = self._page(
            "APPLICATION AND PROJECT",
            "General settings",
            "Defaults are scoped explicitly; the effective run snapshot remains immutable once execution begins.",
            action=self.general_apply_button,
        )
        cards = QtWidgets.QGridLayout()
        cards.setContentsMargins(0, 0, 0, 0)
        cards.setHorizontalSpacing(12)
        cards.setVerticalSpacing(12)

        application_card = QtWidgets.QFrame()
        application_card.setObjectName("generalApplicationDefaultsCard")
        application_card.setProperty("role", "panel-raised")
        application_layout = QtWidgets.QVBoxLayout(application_card)
        application_layout.setContentsMargins(16, 14, 16, 16)
        application_layout.setSpacing(12)
        application_header = QtWidgets.QHBoxLayout()
        application_icon = QtWidgets.QLabel()
        application_icon.setPixmap(hybrid_icon("general").pixmap(18, 18))
        application_copy = QtWidgets.QVBoxLayout()
        application_title = QtWidgets.QLabel("Application defaults")
        application_title.setProperty("role", "section")
        application_detail = QtWidgets.QLabel(
            "Used only when a project has no explicit value."
        )
        application_detail.setProperty("role", "secondary")
        application_copy.addWidget(application_title)
        application_copy.addWidget(application_detail)
        self.application_defaults_status = QtWidgets.QLabel("Valid")
        self.application_defaults_status.setProperty("role", "status-pill")
        self.application_defaults_status.setProperty("tone", "ready")
        application_header.addWidget(application_icon)
        application_header.addLayout(application_copy, 1)
        application_header.addWidget(self.application_defaults_status)
        application_layout.addLayout(application_header)
        application_form = QtWidgets.QGridLayout()
        application_form.setContentsMargins(0, 0, 0, 0)
        application_form.setHorizontalSpacing(12)
        application_form.setVerticalSpacing(6)
        application_form.setColumnStretch(0, 1)
        application_form.setColumnStretch(1, 1)
        self.ui_language = HybridComboBox()
        self.ui_language.addItem("System", "system")
        self.ui_language.addItem("English", "English")
        self.ui_language.currentIndexChanged.connect(self._update_application)
        application_form.addWidget(
            self._settings_field("UI language", self.ui_language), 0, 0
        )
        self.new_project_location = QtWidgets.QLineEdit()
        self.new_project_location.setAccessibleName("New-project location")
        self.new_project_location.editingFinished.connect(self._update_application)
        application_form.addWidget(
            self._settings_field("New-project location", self.new_project_location),
            0,
            1,
        )
        self.autosave_interval = WheelSafeSpinBox()
        self.autosave_interval.setRange(5, 3600)
        self.autosave_interval.setSuffix(" sec")
        self.autosave_interval.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.autosave_interval.setAccessibleName("Autosave interval")
        self.autosave_interval.valueChanged.connect(self._update_application)
        application_form.addWidget(
            self._settings_field("Autosave interval", self.autosave_interval), 1, 0
        )
        self.open_last_project = HybridComboBox()
        self.open_last_project.setAccessibleName("Open last project")
        for label, value in (("Ask", "ask"), ("Always", "always"), ("Never", "never")):
            self.open_last_project.addItem(label, value)
        self.open_last_project.currentIndexChanged.connect(self._update_application)
        application_form.addWidget(
            self._settings_field("Open last project", self.open_last_project), 1, 1
        )
        application_layout.addLayout(application_form)
        cards.addWidget(application_card, 0, 0)

        project_card = QtWidgets.QFrame()
        project_card.setObjectName("generalProjectDefaultsCard")
        project_card.setProperty("role", "panel-raised")
        project_layout = QtWidgets.QVBoxLayout(project_card)
        project_layout.setContentsMargins(16, 14, 16, 16)
        project_layout.setSpacing(12)
        project_header = QtWidgets.QHBoxLayout()
        project_icon = QtWidgets.QLabel()
        project_icon.setPixmap(hybrid_icon("project-scope").pixmap(18, 18))
        project_copy = QtWidgets.QVBoxLayout()
        project_title = QtWidgets.QLabel("Project defaults")
        project_title.setProperty("role", "section")
        project_detail = QtWidgets.QLabel("Stored in project.json without credentials.")
        project_detail.setProperty("role", "secondary")
        project_copy.addWidget(project_title)
        project_copy.addWidget(project_detail)
        self.project_defaults_status = QtWidgets.QLabel("Project")
        self.project_defaults_status.setProperty("role", "status-pill")
        self.project_defaults_status.setProperty("tone", "ready")
        project_header.addWidget(project_icon)
        project_header.addLayout(project_copy, 1)
        project_header.addWidget(self.project_defaults_status)
        project_layout.addLayout(project_header)
        project_form = QtWidgets.QGridLayout()
        project_form.setContentsMargins(0, 0, 0, 0)
        project_form.setHorizontalSpacing(12)
        project_form.setVerticalSpacing(6)
        project_form.setColumnStretch(0, 1)
        project_form.setColumnStretch(1, 1)
        self.source_language = HybridComboBox()
        self.source_language.addItem("Japanese", "Japanese")
        self.source_language.currentIndexChanged.connect(self._update_project)
        project_form.addWidget(
            self._settings_field("Source language", self.source_language), 0, 0
        )
        self.target_language = HybridComboBox()
        self.target_language.addItem("Simplified Chinese", "Simplified Chinese")
        self.target_language.addItem("English", "English")
        self.target_language.currentIndexChanged.connect(self._update_project)
        project_form.addWidget(
            self._settings_field("Target language", self.target_language), 0, 1
        )
        self.output_convention = HybridComboBox()
        self.output_convention.addItem("Sibling output folder", "sibling_output_folder")
        self.output_convention.addItem("Project exports", "project_exports")
        self.output_convention.currentIndexChanged.connect(self._update_project)
        project_form.addWidget(
            self._settings_field("Output convention", self.output_convention), 1, 0
        )
        self.completed_page_policy = HybridComboBox()
        self.completed_page_policy.addItem("Open for review", "open_for_review")
        self.completed_page_policy.addItem(
            "Continue automatically", "continue_automatically"
        )
        self.completed_page_policy.currentIndexChanged.connect(self._update_project)
        project_form.addWidget(
            self._settings_field(
                "Completed-page policy", self.completed_page_policy
            ),
            1,
            1,
        )
        project_layout.addLayout(project_form)
        cards.addWidget(project_card, 0, 1)

        self.effective_run_card = QtWidgets.QFrame()
        self.effective_run_card.setObjectName("generalEffectiveRunCard")
        self.effective_run_card.setProperty("role", "panel-raised")
        summary_layout = QtWidgets.QVBoxLayout(self.effective_run_card)
        summary_layout.setContentsMargins(16, 14, 16, 16)
        summary_header = QtWidgets.QHBoxLayout()
        summary_copy = QtWidgets.QVBoxLayout()
        summary_title = QtWidgets.QLabel("Effective run summary")
        summary_title.setProperty("role", "section")
        summary_detail = QtWidgets.QLabel(
            "What the next page-local transaction will actually use."
        )
        summary_detail.setProperty("role", "secondary")
        summary_copy.addWidget(summary_title)
        summary_copy.addWidget(summary_detail)
        self.reset_project_defaults_button = QtWidgets.QPushButton(
            "Reset project overrides"
        )
        self.reset_project_defaults_button.setProperty("role", "command")
        self.reset_project_defaults_button.setProperty("variant", "quiet")
        self.reset_project_defaults_button.clicked.connect(
            self._reset_project_defaults
        )
        summary_header.addLayout(summary_copy, 1)
        summary_header.addWidget(self.reset_project_defaults_button)
        summary_layout.addLayout(summary_header)
        self.effective_run_status = QtWidgets.QLabel("No run candidate")
        self.effective_run_status.setProperty("role", "status-pill")
        self.effective_run_status.setProperty("tone", "muted")
        summary_layout.addWidget(
            self.effective_run_status,
            0,
            QtCore.Qt.AlignmentFlag.AlignLeft,
        )
        self.effective_run_values = QtWidgets.QHBoxLayout()
        self.effective_run_values.setContentsMargins(0, 0, 0, 0)
        self.effective_run_values.setSpacing(8)
        self.effective_run_cards: list[QtWidgets.QFrame] = []
        self.effective_run_value_labels: dict[str, QtWidgets.QLabel] = {}
        self.effective_run_note_labels: dict[str, QtWidgets.QLabel] = {}
        for key, title in (
            ("translation", "Translation"),
            ("cleanup", "Cleanup"),
            ("renderer", "Renderer"),
        ):
            value_card = QtWidgets.QFrame()
            value_card.setProperty("role", "effective-run-value")
            value_card.setAccessibleName(f"Effective {title.lower()}")
            value_layout = QtWidgets.QVBoxLayout(value_card)
            value_layout.setContentsMargins(10, 9, 10, 9)
            value_layout.setSpacing(3)
            caption = QtWidgets.QLabel(title)
            caption.setProperty("role", "caption")
            value = QtWidgets.QLabel()
            value.setProperty("role", "section")
            value.setWordWrap(True)
            note = QtWidgets.QLabel()
            note.setProperty("role", "secondary")
            note.setWordWrap(True)
            value_layout.addWidget(caption)
            value_layout.addWidget(value)
            value_layout.addWidget(note)
            value_card.hide()
            self.effective_run_values.addWidget(value_card, 1)
            self.effective_run_cards.append(value_card)
            self.effective_run_value_labels[key] = value
            self.effective_run_note_labels[key] = note
        summary_layout.addLayout(self.effective_run_values)
        cards.addWidget(self.effective_run_card, 1, 0, 1, 2)

        self.reset_layout_button = QtWidgets.QPushButton("Reset workspace layout")
        self.reset_layout_button.setProperty("role", "command")
        self.reset_layout_button.setProperty("variant", "secondary")
        self.reset_layout_button.clicked.connect(self.reset_layout_requested)
        self.reset_layout_button.hide()
        application_layout.addWidget(self.reset_layout_button)
        layout.addLayout(cards)
        layout.addStretch(1)
        return page

    def _build_appearance_page(self) -> QtWidgets.QWidget:
        page, layout = self._page(
            "INTERFACE",
            "Appearance",
            "Theme, density, and font scale change the desktop UI only; manga render typography is separate.",
        )
        self.appearance_layout = QtWidgets.QBoxLayout(
            QtWidgets.QBoxLayout.Direction.LeftToRight
        )
        theme_card = QtWidgets.QFrame()
        theme_card.setObjectName("appearanceThemeCard")
        theme_card.setProperty("role", "panel")
        theme_card.setMinimumHeight(248)
        theme_layout = QtWidgets.QVBoxLayout(theme_card)
        theme_layout.setContentsMargins(16, 14, 16, 16)
        theme_layout.setSpacing(10)
        theme_title = QtWidgets.QLabel("Theme")
        theme_title.setProperty("role", "section")
        theme_layout.addWidget(theme_title)
        theme_detail = QtWidgets.QLabel(
            "Both themes share the same semantic state tokens."
        )
        theme_detail.setWordWrap(True)
        theme_detail.setProperty("role", "secondary")
        theme_layout.addWidget(theme_detail)
        self.theme_group = QtWidgets.QButtonGroup(self)
        self.theme_group.setExclusive(True)
        theme_buttons = QtWidgets.QHBoxLayout()
        theme_buttons.setSpacing(10)
        self.dark_theme = QtWidgets.QToolButton()
        self.light_theme = QtWidgets.QToolButton()
        for index, (button, title, detail, dark) in enumerate(
            (
                (self.dark_theme, "Graphite dark", "Canvas-focused", True),
                (self.light_theme, "Paper light", "Editorial review", False),
            )
        ):
            button.setText("")
            button.setCheckable(True)
            button.setProperty("role", "theme-choice")
            button.setProperty("themeTitle", title)
            button.setProperty("themeDetail", detail)
            button.setAccessibleName(title)
            button.setAccessibleDescription(detail)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            choice_layout = QtWidgets.QVBoxLayout(button)
            choice_layout.setContentsMargins(10, 10, 10, 10)
            choice_layout.setSpacing(4)
            swatch = QtWidgets.QFrame()
            swatch.setObjectName("appearanceThemeSwatch")
            swatch.setProperty("theme", "dark" if dark else "light")
            swatch.setMinimumHeight(48)
            swatch.setMaximumHeight(48)
            swatch.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            choice_title = QtWidgets.QLabel(title)
            choice_title.setObjectName("appearanceThemeChoiceTitle")
            choice_title.setProperty("role", "section")
            choice_title.setAttribute(
                QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
            )
            choice_detail = QtWidgets.QLabel(detail)
            choice_detail.setObjectName("appearanceThemeChoiceDetail")
            choice_detail.setProperty("role", "secondary")
            choice_detail.setAttribute(
                QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
            )
            choice_layout.addWidget(swatch)
            choice_layout.addWidget(choice_title)
            choice_layout.addWidget(choice_detail)
            self.theme_group.addButton(button, index)
            theme_buttons.addWidget(button, 1)
        self.dark_theme.clicked.connect(self._update_application)
        self.light_theme.clicked.connect(self._update_application)
        theme_layout.addLayout(theme_buttons)
        self.appearance_layout.addWidget(theme_card, 1)

        readability = QtWidgets.QFrame()
        readability.setObjectName("appearanceReadabilityCard")
        readability.setProperty("role", "panel")
        readability.setMinimumHeight(248)
        readability_layout = QtWidgets.QVBoxLayout(readability)
        readability_layout.setContentsMargins(16, 14, 16, 16)
        readability_layout.setSpacing(10)
        readability_title = QtWidgets.QLabel("Readability")
        readability_title.setProperty("role", "section")
        readability_layout.addWidget(readability_title)
        readability_detail = QtWidgets.QLabel(
            "Preview common Windows display and UI-scale combinations."
        )
        readability_detail.setWordWrap(True)
        readability_detail.setProperty("role", "secondary")
        readability_layout.addWidget(readability_detail)
        font_scale_control = QtWidgets.QWidget()
        font_scale_control_layout = QtWidgets.QVBoxLayout(font_scale_control)
        font_scale_control_layout.setContentsMargins(0, 0, 0, 0)
        font_scale_control_layout.setSpacing(6)
        font_scale_caption = QtWidgets.QWidget()
        font_scale_caption_layout = QtWidgets.QHBoxLayout(font_scale_caption)
        font_scale_caption_layout.setContentsMargins(0, 0, 0, 0)
        font_scale_caption_layout.setSpacing(8)
        font_scale_label = QtWidgets.QLabel("UI font scale")
        font_scale_label.setProperty("role", "secondary")
        self.font_scale = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.font_scale.setRange(100, 200)
        self.font_scale.setSingleStep(5)
        self.font_scale.setPageStep(25)
        self.font_scale.setTickInterval(25)
        self.font_scale.setAccessibleName("UI font scale")
        self.font_scale_value = QtWidgets.QLabel("100%")
        self.font_scale_value.setMinimumWidth(48)
        self.font_scale.valueChanged.connect(self._font_scale_changed)
        font_scale_caption_layout.addWidget(font_scale_label)
        font_scale_caption_layout.addStretch(1)
        font_scale_caption_layout.addWidget(self.font_scale_value)
        font_scale_control_layout.addWidget(font_scale_caption)
        font_scale_control_layout.addWidget(self.font_scale)
        readability_layout.addWidget(font_scale_control)
        self.density = HybridComboBox()
        self.density.addItem("Comfortable", "comfortable")
        self.density.addItem("Compact", "compact")
        self.density.currentIndexChanged.connect(self._update_application)
        readability_layout.addWidget(
            self._settings_field("Information density", self.density)
        )
        self.reduced_motion = QtWidgets.QCheckBox("Reduce non-essential motion")
        self.reduced_motion.toggled.connect(self._update_application)
        readability_layout.addWidget(self.reduced_motion)
        self.appearance_layout.addWidget(readability, 1)
        layout.addLayout(self.appearance_layout)

        live_preview = QtWidgets.QFrame()
        live_preview.setObjectName("appearanceLivePreview")
        live_preview.setProperty("role", "panel")
        live_layout = QtWidgets.QVBoxLayout(live_preview)
        live_layout.setContentsMargins(16, 14, 16, 16)
        live_layout.setSpacing(12)
        live_header = QtWidgets.QHBoxLayout()
        live_copy = QtWidgets.QVBoxLayout()
        live_title = QtWidgets.QLabel("Live preview")
        live_title.setProperty("role", "section")
        live_detail = QtWidgets.QLabel(
            "The desktop shell updates immediately; Apply remains scoped to persisted preferences."
        )
        live_detail.setWordWrap(True)
        live_detail.setProperty("role", "secondary")
        live_copy.addWidget(live_title)
        live_copy.addWidget(live_detail)
        live_status = QtWidgets.QLabel("Contrast checked")
        live_status.setObjectName("appearanceContrastStatus")
        live_status.setProperty("role", "status-pill")
        live_status.setProperty("tone", "ready")
        live_status.setAccessibleName("Appearance contrast checked")
        live_header.addLayout(live_copy, 1)
        live_header.addWidget(live_status, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        live_layout.addLayout(live_header)
        preview = QtWidgets.QFrame()
        preview.setObjectName("appearancePreviewSample")
        preview_layout = QtWidgets.QHBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)
        preview_nav = QtWidgets.QLabel("YomiFrame")
        preview_nav.setObjectName("appearancePreviewNav")
        preview_canvas = QtWidgets.QLabel("Manga canvas")
        preview_canvas.setObjectName("appearancePreviewCanvas")
        preview_inspector = QtWidgets.QWidget()
        preview_inspector.setObjectName("appearancePreviewInspector")
        preview_inspector_layout = QtWidgets.QVBoxLayout(preview_inspector)
        preview_inspector_layout.setContentsMargins(14, 10, 14, 10)
        preview_inspector_layout.setSpacing(2)
        preview_authority = QtWidgets.QLabel("Automatic")
        preview_authority.setProperty("role", "section")
        preview_result = QtWidgets.QLabel("Your edit · Effective result")
        preview_result.setProperty("role", "secondary")
        preview_inspector_layout.addWidget(preview_authority)
        preview_inspector_layout.addWidget(preview_result)
        for sample in (preview_nav, preview_canvas):
            sample.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(preview_nav, 2)
        preview_layout.addWidget(preview_canvas, 5)
        preview_layout.addWidget(preview_inspector, 3)
        live_layout.addWidget(preview)
        layout.addWidget(live_preview)

        layout.addStretch(1)
        return page

    def _build_providers_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        heading = QtWidgets.QWidget()
        heading.setObjectName("providerHeading")
        heading_layout = QtWidgets.QHBoxLayout(heading)
        heading_layout.setContentsMargins(0, 0, 0, 0)
        heading_copy = QtWidgets.QVBoxLayout()
        heading_copy.setSpacing(5)
        eyebrow = QtWidgets.QLabel("TRANSLATION")
        eyebrow.setProperty("role", "eyebrow")
        title = QtWidgets.QLabel("Provider profiles")
        title.setObjectName("surfaceTitle")
        title.setProperty("role", "title")
        detail = QtWidgets.QLabel(
            "Public transport settings are stored separately from secure credential references."
        )
        detail.setProperty("role", "secondary")
        detail.setWordWrap(True)
        heading_copy.addWidget(eyebrow)
        heading_copy.addWidget(title)
        heading_copy.addWidget(detail)
        heading_layout.addLayout(heading_copy, 1)
        self._provider_heading_actions = QtWidgets.QHBoxLayout()
        self._provider_heading_actions.setContentsMargins(0, 0, 0, 0)
        heading_layout.addLayout(self._provider_heading_actions)
        layout.addWidget(heading)
        self.providers_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Horizontal
        )
        self.providers_splitter.setObjectName("providersWorkspace")
        self.providers_splitter.setChildrenCollapsible(False)
        self.providers_splitter.setHandleWidth(1)
        profiles_panel = QtWidgets.QFrame()
        profiles_panel.setObjectName("providerProfilesPanel")
        profiles_panel.setProperty("role", "panel")
        profiles_panel.setMinimumWidth(280)
        profiles_panel.setMaximumWidth(280)
        profile_layout = QtWidgets.QVBoxLayout(profiles_panel)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        self.add_provider = QtWidgets.QToolButton()
        self.add_provider.setObjectName("settingsNewProviderButton")
        self.add_provider.setText("New profile")
        self.add_provider.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.add_provider.setProperty("role", "command")
        self.add_provider.setProperty("variant", "secondary")
        self.add_provider.setAccessibleName("Create provider profile")
        self.add_provider.setIcon(hybrid_icon("new"))
        self.add_provider.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        menu = QtWidgets.QMenu(self.add_provider)
        for kind, label in (
            (ProviderKind.GGUF, "GGUF local"),
            (ProviderKind.OLLAMA, "Ollama"),
            (ProviderKind.DEEPSEEK, "DeepSeek API"),
            (ProviderKind.OPENAI_COMPATIBLE, "OpenAI-compatible"),
        ):
            action = menu.addAction(label)
            action.triggered.connect(lambda _checked=False, value=kind: self._add_profile(value))
        self.add_provider.setMenu(menu)
        self._provider_heading_actions.addWidget(
            self.add_provider,
            0,
            QtCore.Qt.AlignmentFlag.AlignTop,
        )
        self.profile_list = QtWidgets.QListWidget()
        self.profile_list.setObjectName("providerProfileList")
        self.profile_list.setAccessibleName("Provider profiles")
        self.profile_list.setIconSize(QtCore.QSize(24, 24))
        self.profile_list.currentRowChanged.connect(self._show_selected_profile)
        profile_layout.addWidget(self.profile_list, 1)
        add_row = QtWidgets.QHBoxLayout()
        self.delete_provider = QtWidgets.QPushButton("Delete profile")
        self.delete_provider.setProperty("role", "command")
        self.delete_provider.setProperty("variant", "quiet")
        self.delete_provider.setProperty("tone", "danger")
        self.delete_provider.setAccessibleName("Delete provider profile")
        self.delete_provider.setAccessibleDescription(
            "Remove the selected provider profile after confirmation."
        )
        self.delete_provider.clicked.connect(self._remove_profile)
        self.delete_provider.hide()
        add_row.addStretch(1)
        add_row.addWidget(self.delete_provider)
        profile_layout.addLayout(add_row)
        self.providers_splitter.addWidget(profiles_panel)

        editor = QtWidgets.QFrame()
        editor.setObjectName("providerEditorCard")
        editor.setProperty("role", "panel")
        editor_layout = QtWidgets.QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)
        provider_header_widget = QtWidgets.QWidget()
        provider_header_widget.setObjectName("providerEditorHeader")
        provider_header_widget.setMinimumHeight(82)
        provider_header = QtWidgets.QHBoxLayout()
        provider_header.setContentsMargins(22, 0, 22, 0)
        provider_header.setSpacing(11)
        provider_header_widget.setLayout(provider_header)
        self.profile_summary_icon = QtWidgets.QLabel()
        self.profile_summary_icon.setObjectName("providerSummaryIcon")
        self.profile_summary_icon.setFixedSize(33, 33)
        self.profile_summary_icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.profile_summary_icon.setAccessibleName("Selected provider type")
        provider_header.addWidget(self.profile_summary_icon)
        provider_identity = QtWidgets.QVBoxLayout()
        provider_identity.setSpacing(3)
        self.profile_summary_name = QtWidgets.QLabel("No provider selected")
        self.profile_summary_name.setProperty("role", "title")
        self.profile_kind = QtWidgets.QLabel("Select or add a provider profile")
        self.profile_kind.setProperty("role", "secondary")
        provider_identity.addWidget(self.profile_summary_name)
        provider_identity.addWidget(self.profile_kind)
        provider_header.addLayout(provider_identity, 1)
        self.active_provider_status = QtWidgets.QToolButton()
        self.active_provider_status.setText("No active translation provider")
        self.active_provider_status.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.active_provider_status.setIconSize(QtCore.QSize(12, 12))
        self.active_provider_status.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.active_provider_status.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        self.active_provider_status.setProperty("role", "status-pill")
        self.active_provider_status.setProperty("tone", "warning")
        provider_header.addWidget(
            self.active_provider_status,
            0,
            QtCore.Qt.AlignmentFlag.AlignVCenter,
        )
        editor_layout.addWidget(provider_header_widget)
        self.provider_editor_body = QtWidgets.QScrollArea()
        self.provider_editor_body.setObjectName("providerEditorBody")
        self.provider_editor_body.setWidgetResizable(True)
        self.provider_editor_body.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.provider_editor_body.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        provider_body = QtWidgets.QWidget()
        provider_body.setObjectName("providerEditorBodyContent")
        provider_body_layout = QtWidgets.QVBoxLayout(provider_body)
        provider_body_layout.setContentsMargins(0, 0, 0, 0)
        provider_body_layout.setSpacing(0)
        self.provider_empty_state = QtWidgets.QFrame()
        self.provider_empty_state.setObjectName("providerEmptyState")
        self.provider_empty_state.setProperty("role", "empty-state")
        self.provider_empty_state.setAccessibleName("No provider profile selected")
        self.provider_empty_state.setAccessibleDescription(
            "Create a GGUF, Ollama, DeepSeek, or OpenAI-compatible profile "
            "with the New profile menu."
        )
        empty_layout = QtWidgets.QVBoxLayout(self.provider_empty_state)
        empty_layout.setContentsMargins(32, 32, 32, 32)
        empty_layout.setSpacing(10)
        empty_layout.addStretch(1)
        empty_icon = QtWidgets.QLabel()
        empty_icon.setObjectName("providerEmptyStateIcon")
        empty_icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_icon.setPixmap(hybrid_icon("provider").pixmap(30, 30))
        empty_title = QtWidgets.QLabel("No provider profile yet")
        empty_title.setObjectName("providerEmptyStateTitle")
        empty_title.setProperty("role", "title")
        empty_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_detail = QtWidgets.QLabel(
            "Create a GGUF, Ollama, DeepSeek, or OpenAI-compatible profile."
        )
        empty_detail.setObjectName("providerEmptyStateDetail")
        empty_detail.setProperty("role", "secondary")
        empty_detail.setWordWrap(True)
        empty_detail.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_detail.setMinimumWidth(440)
        empty_detail.setMaximumWidth(520)
        empty_layout.addWidget(empty_icon, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        empty_layout.addWidget(empty_title)
        empty_layout.addWidget(empty_detail, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        empty_layout.addStretch(1)
        provider_body_layout.addWidget(self.provider_empty_state, 1)
        self.provider_editor_body.setWidget(provider_body)
        editor_layout.addWidget(self.provider_editor_body, 1)
        base_form_widget = QtWidgets.QWidget()
        base_form_widget.setObjectName("providerForm")
        self.provider_base_form = base_form_widget
        base_form = QtWidgets.QGridLayout(base_form_widget)
        base_form.setContentsMargins(22, 22, 22, 14)
        base_form.setHorizontalSpacing(14)
        base_form.setVerticalSpacing(14)

        def field_widget(
            label_text: str,
            control: QtWidgets.QWidget,
        ) -> tuple[QtWidgets.QWidget, QtWidgets.QLabel]:
            field = QtWidgets.QWidget()
            field_layout = QtWidgets.QVBoxLayout(field)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(5)
            label = QtWidgets.QLabel(label_text)
            label.setProperty("role", "field-label")
            field_layout.addWidget(label)
            field_layout.addWidget(control)
            return field, label

        self.profile_name = QtWidgets.QLineEdit()
        self.profile_type = HybridComboBox()
        for kind in (
            ProviderKind.GGUF,
            ProviderKind.OLLAMA,
            ProviderKind.DEEPSEEK,
            ProviderKind.OPENAI_COMPATIBLE,
        ):
            display_type = {
                ProviderKind.DEEPSEEK: "DeepSeek API",
                ProviderKind.OPENAI_COMPATIBLE: "Custom OpenAI-compatible",
            }.get(kind, provider_kind_label(kind))
            self.profile_type.addItem(display_type, kind)
        self.profile_type.setAccessibleName("Provider type")
        self.profile_type.currentIndexChanged.connect(self._change_provider_type)
        self.profile_endpoint = QtWidgets.QLineEdit()
        self.profile_model = QtWidgets.QLineEdit()
        self.profile_path = QtWidgets.QLineEdit()
        self.profile_path.setPlaceholderText("Local GGUF model path")
        for control in (
            self.profile_name,
            self.profile_endpoint,
            self.profile_model,
            self.profile_path,
        ):
            control.editingFinished.connect(self._update_profile)
        self.profile_name_field, self.profile_name_label = field_widget(
            "Profile name",
            self.profile_name,
        )
        self.profile_type_field, self.profile_type_label = field_widget(
            "Provider type",
            self.profile_type,
        )
        self.profile_endpoint_field, self.profile_endpoint_label = field_widget(
            "Base URL",
            self.profile_endpoint,
        )
        self.profile_model_field, self.profile_model_label = field_widget(
            "Model",
            self.profile_model,
        )
        local_model_row = QtWidgets.QWidget()
        local_model_layout = QtWidgets.QHBoxLayout(local_model_row)
        local_model_layout.setContentsMargins(0, 0, 0, 0)
        local_model_layout.setSpacing(8)
        local_model_layout.addWidget(self.profile_path, 1)
        self.browse_provider_model = QtWidgets.QPushButton("Browse…")
        self.browse_provider_model.setProperty("role", "command")
        self.browse_provider_model.setProperty("variant", "secondary")
        self.browse_provider_model.setAccessibleName("Browse for local GGUF model")
        self.browse_provider_model.clicked.connect(self._browse_provider_model)
        local_model_layout.addWidget(self.browse_provider_model)
        self.profile_path_row = local_model_row
        self.profile_path_field, self.profile_path_label = field_widget(
            "Local model",
            local_model_row,
        )

        self.credential_status = QtWidgets.QLineEdit(
            "Not linked — Test connection will ask for an API key"
        )
        self.credential_status.setObjectName("secureCredentialReference")
        self.credential_status.setReadOnly(True)
        self.credential_status.setAccessibleName("Credential reference status")
        self.link_credential = QtWidgets.QPushButton("Enter API key")
        self.link_credential.setProperty("role", "command")
        self.link_credential.setProperty("variant", "secondary")
        self.link_credential.clicked.connect(self._link_credential)
        credential_row = QtWidgets.QWidget()
        credential_layout = QtWidgets.QHBoxLayout(credential_row)
        credential_layout.setContentsMargins(0, 0, 0, 0)
        credential_layout.setSpacing(4)
        credential_layout.addWidget(self.credential_status, 1)
        credential_layout.addWidget(self.link_credential)
        self.credential_field, self.credential_label = field_widget(
            "Credential reference",
            credential_row,
        )

        base_form.addWidget(self.profile_name_field, 0, 0)
        base_form.addWidget(self.profile_type_field, 0, 1)
        base_form.addWidget(self.profile_endpoint_field, 1, 0, 1, 2)
        base_form.addWidget(self.profile_path_field, 1, 0, 1, 2)
        base_form.addWidget(self.profile_model_field, 2, 0)
        base_form.addWidget(self.credential_field, 2, 1)
        base_form.setColumnStretch(0, 1)
        base_form.setColumnStretch(1, 1)
        provider_body_layout.addWidget(base_form_widget)

        advanced_form_widget = QtWidgets.QWidget()
        advanced_form_widget.setObjectName("providerAdvancedForm")
        self.provider_advanced_form = advanced_form_widget
        form = QtWidgets.QFormLayout(advanced_form_widget)
        form.setContentsMargins(22, 0, 22, 14)

        self.provider_runtime_label = QtWidgets.QLabel("Provider runtime")
        self.provider_runtime_label.setProperty("role", "section")
        form.addRow(self.provider_runtime_label)
        self.provider_prompt_style_label = QtWidgets.QLabel("Prompt format")
        self.provider_prompt_style = HybridComboBox()
        for label, value in (
            ("Sakura", "sakura"),
            ("Qwen", "qwen"),
            ("Plain", "plain"),
        ):
            self.provider_prompt_style.addItem(label, value)
        self.provider_context_tokens_label = QtWidgets.QLabel("Context tokens")
        self.provider_context_tokens = WheelSafeSpinBox()
        self.provider_context_tokens.setRange(512, 32768)
        self.provider_gpu_layers_label = QtWidgets.QLabel("GPU layers")
        self.provider_gpu_layers = WheelSafeSpinBox()
        self.provider_gpu_layers.setRange(-1, 200)
        self.provider_gpu_layers.setSpecialValueText("Automatic")
        self.provider_gpu_layers.setAccessibleName("GGUF GPU layers")
        self.provider_gpu_layers.setAccessibleDescription(
            "Automatic fits the highest safe layer count when Start checks the "
            "current accelerator memory budget. Explicit values remain exact."
        )
        self.provider_gpu_layers.setToolTip(
            "Automatic fits this run to current VRAM without changing the model, "
            "context, or saved provider profile."
        )
        self.provider_threads_label = QtWidgets.QLabel("CPU threads")
        self.provider_threads = WheelSafeSpinBox()
        self.provider_threads.setRange(1, 128)
        self.provider_batch_label = QtWidgets.QLabel("Prompt batch")
        self.provider_batch = WheelSafeSpinBox()
        self.provider_batch.setRange(64, 4096)
        self.provider_generation_label = QtWidgets.QLabel("Generation defaults")
        self.provider_generation_label.setProperty("role", "section")
        self.provider_temperature_label = QtWidgets.QLabel("Temperature")
        self.provider_temperature = WheelSafeDoubleSpinBox()
        self.provider_temperature.setRange(0.0, 2.0)
        self.provider_temperature.setDecimals(2)
        self.provider_temperature.setSingleStep(0.05)
        self.provider_top_p_label = QtWidgets.QLabel("Top P")
        self.provider_top_p = WheelSafeDoubleSpinBox()
        self.provider_top_p.setRange(0.01, 1.0)
        self.provider_top_p.setDecimals(2)
        self.provider_top_p.setSingleStep(0.05)
        self.provider_max_tokens_label = QtWidgets.QLabel("Maximum output tokens")
        self.provider_max_tokens = WheelSafeSpinBox()
        self.provider_max_tokens.setRange(0, 1_000_000)
        self.provider_max_tokens.setSpecialValueText("Provider default")
        for numeric_control in (
            self.provider_context_tokens,
            self.provider_gpu_layers,
            self.provider_threads,
            self.provider_batch,
            self.provider_temperature,
            self.provider_top_p,
            self.provider_max_tokens,
        ):
            numeric_control.setButtonSymbols(
                QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
            )
        form.addRow(self.provider_prompt_style_label, self.provider_prompt_style)
        form.addRow(self.provider_context_tokens_label, self.provider_context_tokens)
        form.addRow(self.provider_gpu_layers_label, self.provider_gpu_layers)
        form.addRow(self.provider_threads_label, self.provider_threads)
        form.addRow(self.provider_batch_label, self.provider_batch)
        form.addRow(self.provider_generation_label)
        form.addRow(self.provider_temperature_label, self.provider_temperature)
        form.addRow(self.provider_top_p_label, self.provider_top_p)
        form.addRow(self.provider_max_tokens_label, self.provider_max_tokens)
        self.provider_prompt_style.currentIndexChanged.connect(self._update_profile)
        for control in (
            self.provider_context_tokens,
            self.provider_gpu_layers,
            self.provider_threads,
            self.provider_batch,
            self.provider_temperature,
            self.provider_top_p,
            self.provider_max_tokens,
        ):
            control.editingFinished.connect(self._update_profile)
        provider_body_layout.addWidget(advanced_form_widget)
        self.provider_safety_callout = QtWidgets.QFrame()
        self.provider_safety_callout.setObjectName("providerSafetyCallout")
        safety_layout = QtWidgets.QHBoxLayout(self.provider_safety_callout)
        safety_layout.setContentsMargins(12, 10, 12, 10)
        safety_layout.setSpacing(10)
        self.provider_safety_icon = QtWidgets.QLabel()
        self.provider_safety_icon.setObjectName("providerSafetyIcon")
        self.provider_safety_icon.setFixedSize(20, 20)
        self.provider_safety_icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        safety_layout.addWidget(
            self.provider_safety_icon,
            0,
            QtCore.Qt.AlignmentFlag.AlignTop,
        )
        safety_copy = QtWidgets.QVBoxLayout()
        safety_copy.setSpacing(2)
        self.provider_safety_title = QtWidgets.QLabel(
            "Credentials stay outside project files"
        )
        self.provider_safety_title.setProperty("role", "section")
        self.provider_safety_detail = QtWidgets.QLabel(
            f"Only an opaque {self._platform_copy.credential_store_label} reference is stored with this profile."
        )
        self.provider_safety_detail.setWordWrap(True)
        self.provider_safety_detail.setProperty("role", "secondary")
        safety_copy.addWidget(self.provider_safety_title)
        safety_copy.addWidget(self.provider_safety_detail)
        safety_layout.addLayout(safety_copy, 1)
        self.test_provider = QtWidgets.QPushButton("Test provider")
        self.test_provider.setProperty("role", "command")
        self.test_provider.setProperty("variant", "secondary")
        self.test_provider.setIcon(hybrid_icon("play"))
        self.test_provider.clicked.connect(self._test_provider)
        self.use_translation_provider = QtWidgets.QPushButton("Use for translation")
        self.use_translation_provider.setProperty("role", "command")
        self.use_translation_provider.setProperty("variant", "secondary")
        self.use_translation_provider.clicked.connect(self._use_translation_provider)
        self.profile_validation = QtWidgets.QLabel("Select a profile")
        self.profile_validation.setWordWrap(True)
        self.profile_validation.setProperty("role", "secondary")
        self.profile_validation.setContentsMargins(22, 0, 22, 8)
        provider_body_layout.addWidget(self.profile_validation)
        provider_body_layout.addStretch(1)
        safety_host = QtWidgets.QWidget()
        self.provider_safety_host = safety_host
        safety_host_layout = QtWidgets.QVBoxLayout(safety_host)
        safety_host_layout.setContentsMargins(22, 0, 22, 18)
        safety_host_layout.addWidget(self.provider_safety_callout)
        provider_body_layout.addWidget(safety_host)
        provider_commit = QtWidgets.QFrame()
        self.provider_commit = provider_commit
        provider_commit.setObjectName("providerCommitBar")
        provider_commit.setProperty("role", "dock-bar")
        provider_commit_layout = QtWidgets.QHBoxLayout(provider_commit)
        provider_commit_layout.setContentsMargins(22, 9, 22, 9)
        provider_commit_layout.addWidget(self.test_provider)
        provider_commit_layout.addWidget(self.use_translation_provider)
        provider_commit_layout.addStretch(1)
        self.provider_cancel_button = QtWidgets.QPushButton("Cancel")
        self.provider_cancel_button.setProperty("role", "command")
        self.provider_cancel_button.clicked.connect(self.cancel_changes)
        provider_commit_layout.addWidget(self.provider_cancel_button)
        self.provider_apply_button = QtWidgets.QPushButton("Apply changes")
        self.provider_apply_button.setProperty("role", "command")
        self.provider_apply_button.setProperty("variant", "primary")
        self.provider_apply_button.setIcon(hybrid_icon("check"))
        self.provider_apply_button.clicked.connect(self.apply_changes)
        provider_commit_layout.addWidget(self.provider_apply_button)
        editor_layout.addWidget(provider_commit)
        self.providers_splitter.addWidget(editor)
        self.providers_splitter.setStretchFactor(0, 2)
        self.providers_splitter.setStretchFactor(1, 5)
        self.providers_splitter.setSizes((280, 820))
        self.providers_splitter.setMinimumHeight(562)
        self.providers_splitter.setMaximumHeight(562)
        layout.addWidget(self.providers_splitter)
        layout.addStretch(1)
        return page

    def set_layout_mode(self, mode: LayoutMode) -> None:
        if not isinstance(mode, LayoutMode):
            raise TypeError("mode must be LayoutMode")
        reflow = bool(mode.accessible_reflow or mode.width_tier == "narrow")
        high_scale = mode.font_scale_tier in {"large", "max"}
        if high_scale:
            # Preserve readable navigation and Basic/Advanced controls at the
            # supported 150-200% application font scales.  Wide/standard
            # windows have room for the full labels; narrow windows retain the
            # nominal rail width while the content surface reflows vertically.
            self.sidebar.setFixedWidth(265 if mode.width_tier == "narrow" else 360)
        else:
            self.sidebar.setFixedWidth(196 if reflow else 265)
        level_inset = 24 if high_scale else 52
        self.settings_level.setFixedWidth(
            max(144, self.sidebar.width() - level_inset)
        )
        self.sidebar_detail.setVisible(not reflow)
        self.appearance_layout.setDirection(
            QtWidgets.QBoxLayout.Direction.TopToBottom
            if reflow
            else QtWidgets.QBoxLayout.Direction.LeftToRight
        )
        self.providers_splitter.setOrientation(
            QtCore.Qt.Orientation.Vertical
            if reflow
            else QtCore.Qt.Orientation.Horizontal
        )
        if reflow or high_scale:
            self.providers_splitter.setMinimumHeight(0)
            self.providers_splitter.setMaximumHeight(16_777_215)
        else:
            self.providers_splitter.setMinimumHeight(562)
            self.providers_splitter.setMaximumHeight(562)
        self.providers_splitter.setSizes(
            (260, 620) if reflow else (300, 820)
        )
        self.glossary.set_layout_mode(mode)
        if reflow:
            horizontal_margin = 22
        elif mode.width_tier == "wide" or high_scale:
            horizontal_margin = 48
        else:
            horizontal_margin = 42
        self.content_layout.setContentsMargins(
            horizontal_margin,
            24,
            horizontal_margin,
            24,
        )
        self._module_policy_reflow = reflow
        self._module_policy_compact_identity = mode.width_tier == "narrow"
        self._sync_module_policy_geometry()

    def bind_glossary_model(
        self,
        model: GlossaryEditorModel,
        *,
        page_labels: dict[str, str] | None = None,
    ) -> None:
        """Bind the independently persisted project-glossary editor."""

        self.glossary.bind_model(model, page_labels=page_labels)

    def clear_glossary_model(
        self,
        reason: str = "Open a project to manage its glossary.",
    ) -> None:
        self.glossary.clear_model(reason)

    def _build_modules_page(self) -> QtWidgets.QWidget:
        self.module_validate_button = QtWidgets.QPushButton("Validate")
        self.module_validate_button.setProperty("role", "command")
        self.module_validate_button.setProperty("variant", "secondary")
        self.module_validate_button.setIcon(hybrid_icon("check"))
        self.module_validate_button.setAccessibleName("Validate module policies")
        self.module_validate_button.clicked.connect(self._validate_module_policies)
        page, layout = self._page(
            "WORKFLOW MODULES",
            "Module policies",
            "Choose supported implementations without changing ownership or page-by-page execution order.",
            action=self.module_validate_button,
        )
        self.modules_scroll = QtWidgets.QScrollArea()
        self.modules_scroll.setWidgetResizable(True)
        self.modules_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.modules_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.modules_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.modules_content = QtWidgets.QWidget()
        self.modules_layout = QtWidgets.QVBoxLayout(self.modules_content)
        self.modules_layout.setContentsMargins(0, 0, 8, 0)
        self.modules_layout.setSpacing(10)
        self._module_policy_rows: list[QtWidgets.QFrame] = []
        self._module_policy_layouts: dict[str, QtWidgets.QGridLayout] = {}
        self._module_policy_indices: dict[str, QtWidgets.QLabel] = {}
        self._module_policy_forms: dict[str, QtWidgets.QFormLayout] = {}
        self._module_policy_identity_hosts: dict[str, QtWidgets.QWidget] = {}
        self._module_policy_descriptions: dict[str, QtWidgets.QLabel] = {}
        self._module_policy_states: dict[str, QtWidgets.QLabel] = {}
        policy_rows = (
            (
                "detection",
                "Bubble detection",
                "Source text-area detection",
            ),
            ("ocr", "OCR", "Japanese source recognition"),
            ("translation", "Translation", "Selected provider profile"),
            ("cleanup", "Cleanup", "Waits for translation"),
            (
                "rendering",
                "Rendering",
                "Waits for clean base",
            ),
        )
        for index, (stage_id, title, detail) in enumerate(policy_rows, 1):
            row = QtWidgets.QFrame()
            row.setObjectName("modulePolicyRow")
            row.setProperty("moduleStage", stage_id)
            row.setProperty("role", "panel-raised")
            row.setProperty("searchText", f"{stage_id} {title} {detail}".casefold())
            row_layout = QtWidgets.QGridLayout(row)
            row_layout.setContentsMargins(12, 10, 12, 10)
            row_layout.setHorizontalSpacing(MODULE_POLICY_GEOMETRY.row_gap)
            row_layout.setVerticalSpacing(MODULE_POLICY_GEOMETRY.row_gap)
            number = QtWidgets.QLabel(str(index))
            number.setObjectName("modulePolicyIndex")
            number.setFixedSize(
                MODULE_POLICY_GEOMETRY.index_width,
                MODULE_POLICY_GEOMETRY.index_width,
            )
            number.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            identity = QtWidgets.QWidget()
            identity.setObjectName("modulePolicyIdentity")
            identity.setFixedWidth(MODULE_POLICY_GEOMETRY.expanded_identity_width)
            identity.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Preferred,
                QtWidgets.QSizePolicy.Policy.Maximum,
            )
            copy = QtWidgets.QVBoxLayout(identity)
            copy.setContentsMargins(0, 0, 0, 0)
            copy.setSpacing(3)
            name = QtWidgets.QLabel(title)
            name.setProperty("role", "section")
            name.setWordWrap(True)
            description = QtWidgets.QLabel(detail)
            description.setProperty("role", "secondary")
            description.setWordWrap(True)
            copy.addWidget(name)
            copy.addWidget(description)
            form = QtWidgets.QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.setHorizontalSpacing(
                MODULE_POLICY_GEOMETRY.form_horizontal_spacing
            )
            form.setVerticalSpacing(MODULE_POLICY_GEOMETRY.form_vertical_spacing)
            form.setLabelAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            form.setRowWrapPolicy(
                QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows
            )
            form.setFieldGrowthPolicy(
                QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
            state = QtWidgets.QLabel("Configured")
            state.setObjectName("modulePolicyState")
            state.setProperty("role", "status-pill")
            state.setProperty("tone", "ready")
            state.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            row_layout.addWidget(
                number,
                0,
                0,
                QtCore.Qt.AlignmentFlag.AlignTop,
            )
            row_layout.addWidget(
                identity,
                0,
                1,
                QtCore.Qt.AlignmentFlag.AlignTop,
            )
            row_layout.addLayout(form, 0, 2)
            row_layout.addWidget(
                state,
                0,
                3,
                QtCore.Qt.AlignmentFlag.AlignTop,
            )
            row_layout.setColumnStretch(2, 1)
            self.modules_layout.addWidget(row)
            self._module_policy_rows.append(row)
            self._module_policy_layouts[stage_id] = row_layout
            self._module_policy_indices[stage_id] = number
            self._module_policy_forms[stage_id] = form
            self._module_policy_identity_hosts[stage_id] = identity
            self._module_policy_descriptions[stage_id] = description
            self._module_policy_states[stage_id] = state
        self.modules_layout.addStretch(1)
        self.modules_scroll.setWidget(self.modules_content)
        layout.addWidget(self.modules_scroll, 1)
        return page

    def _build_runtime_page(self) -> QtWidgets.QWidget:
        self.runtime_verify_all_button = QtWidgets.QPushButton("Verify all")
        self.runtime_verify_all_button.setObjectName("runtimeVerifyAllButton")
        self.runtime_verify_all_button.setProperty("role", "command")
        self.runtime_verify_all_button.setProperty("variant", "secondary")
        self.runtime_verify_all_button.setIcon(hybrid_icon("redo"))
        self.runtime_verify_all_button.setAccessibleName("Verify all runtime assets")
        self.runtime_verify_all_button.clicked.connect(self.runtime_verify_requested)
        page, layout = self._page(
            "LOCAL DEPLOYMENT",
            "Runtime assets",
            "Inspect, repair, or download individual assets without a blocking startup modal.",
            action=self.runtime_verify_all_button,
        )
        self.runtime_asset_rows: dict[
            str, tuple[QtWidgets.QLabel, QtWidgets.QLabel, QtWidgets.QPushButton]
        ] = {}
        self._runtime_asset_specs = {
            item.asset_id: item for item in runtime_asset_catalog()
        }
        for asset_id, spec in self._runtime_asset_specs.items():
            name = spec.name
            detail = spec.detail
            row = QtWidgets.QFrame()
            row.setObjectName("runtimeAssetRow")
            row.setProperty("role", "panel-raised")
            row.setMinimumHeight(66)
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(12, 10, 12, 10)
            mark = QtWidgets.QLabel()
            mark.setProperty("role", "runtime-icon")
            mark.setFixedSize(34, 34)
            mark.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            mark.setPixmap(hybrid_icon("runtime").pixmap(19, 19))
            copy = QtWidgets.QVBoxLayout()
            title = QtWidgets.QLabel(name)
            title.setProperty("role", "section")
            description = QtWidgets.QLabel(detail)
            description.setProperty("role", "secondary")
            copy.addWidget(title)
            copy.addWidget(description)
            state = QtWidgets.QLabel("Not checked")
            state.setProperty("role", "status-pill")
            state.setProperty("tone", "muted")
            action = QtWidgets.QPushButton("Details")
            action.setProperty("role", "command")
            action.setProperty("variant", "secondary")
            action.setFixedWidth(92)
            action.setAccessibleName(f"Open {name} runtime details")
            action.clicked.connect(
                lambda _checked=False, value=asset_id: (
                    self.runtime_asset_action_requested.emit(value)
                )
            )
            row_layout.addWidget(mark)
            row_layout.addLayout(copy, 1)
            row_layout.addWidget(state)
            row_layout.addWidget(action)
            layout.addWidget(row)
            self.runtime_asset_rows[asset_id] = (description, state, action)
        location = QtWidgets.QFrame()
        location.setObjectName("runtimeLocationCard")
        location.setProperty("role", "panel-raised")
        location.setMinimumHeight(62)
        location_layout = QtWidgets.QHBoxLayout(location)
        location_copy = QtWidgets.QVBoxLayout()
        location_title = QtWidgets.QLabel("Managed runtime root")
        location_title.setProperty("role", "section")
        self.runtime_root_value = QtWidgets.QLabel(
            self._platform_copy.runtime_root_label
        )
        self.runtime_root_value.setProperty("role", "secondary")
        location_copy.addWidget(location_title)
        location_copy.addWidget(self.runtime_root_value)
        self.runtime_open_folder_button = QtWidgets.QPushButton("Open folder")
        self.runtime_open_folder_button.setProperty("role", "command")
        self.runtime_open_folder_button.setProperty("variant", "secondary")
        self.runtime_open_folder_button.clicked.connect(
            self.runtime_open_folder_requested
        )
        location_layout.addLayout(location_copy, 1)
        location_layout.addWidget(self.runtime_open_folder_button)
        layout.addWidget(location)
        layout.addStretch(1)
        return page

    def _build_shortcuts_page(self) -> QtWidgets.QWidget:
        page, layout = self._page(
            "KEYBOARD",
            "Shortcuts",
            "Every primary editor command remains reachable without a pointer.",
        )
        self._shortcut_rows: list[QtWidgets.QFrame] = []
        self._shortcut_binding_edits: dict[str, QtWidgets.QLineEdit] = {}
        self._shortcut_commands = dict(
            (
                ("select", "Select tool"),
                ("pan", "Pan canvas"),
                ("undo", "Undo edit"),
                ("redo", "Redo edit"),
                ("preview", "Preview page"),
                ("exit_focus", "Exit focus"),
            )
        )
        shortcut_table = QtWidgets.QWidget()
        shortcut_table.setObjectName("shortcutTable")
        shortcut_table_layout = QtWidgets.QVBoxLayout(shortcut_table)
        shortcut_table_layout.setContentsMargins(0, 0, 0, 0)
        shortcut_table_layout.setSpacing(0)
        for shortcut_id, command in self._shortcut_commands.items():
            shortcut = str(DEFAULT_SHORTCUT_BINDINGS[shortcut_id])
            row = QtWidgets.QFrame()
            row.setObjectName("shortcutBindingRow")
            row.setProperty("shortcutId", shortcut_id)
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(14, 9, 14, 9)
            title = QtWidgets.QLabel(command)
            title.setProperty("role", "section")
            binding = QtWidgets.QLineEdit(shortcut)
            binding.setReadOnly(True)
            binding.setAccessibleName(f"{command} shortcut")
            binding.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            binding.setFixedWidth(150)
            self._shortcut_binding_edits[shortcut_id] = binding
            change = QtWidgets.QPushButton("Change")
            change.setProperty("role", "command")
            change.setProperty("variant", "quiet")
            change.setFixedWidth(70)
            change.setAccessibleName(f"Change {command} shortcut")
            change.clicked.connect(
                lambda _checked=False, value=shortcut_id: (
                    self._open_shortcut_dialog(value)
                )
            )
            row_layout.addWidget(title, 1)
            row_layout.addWidget(binding)
            row_layout.addWidget(change)
            shortcut_table_layout.addWidget(row)
            self._shortcut_rows.append(row)
        layout.addWidget(shortcut_table)
        note = QtWidgets.QFrame()
        note.setObjectName("shortcutKeyboardNote")
        note.setProperty("role", "state-callout")
        note.setProperty("tone", "ready")
        note_layout = QtWidgets.QHBoxLayout(note)
        note_icon = QtWidgets.QLabel()
        note_icon.setPixmap(hybrid_icon("shortcuts").pixmap(18, 18))
        note_copy = QtWidgets.QVBoxLayout()
        note_title = QtWidgets.QLabel("Keyboard navigation enabled")
        note_title.setProperty("role", "section")
        note_detail = QtWidgets.QLabel(
            "Tab order follows page rail, canvas tools, inspector, and Activity Dock."
        )
        note_detail.setProperty("role", "secondary")
        note_detail.setWordWrap(True)
        note_copy.addWidget(note_title)
        note_copy.addWidget(note_detail)
        note_layout.addWidget(note_icon)
        note_layout.addLayout(note_copy, 1)
        layout.addWidget(note)
        layout.addStretch(1)
        return page

    @staticmethod
    def _placeholder_page(title: str, detail: str) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        heading = QtWidgets.QLabel(title)
        heading.setProperty("role", "title")
        copy = QtWidgets.QLabel(detail)
        copy.setWordWrap(True)
        copy.setProperty("role", "secondary")
        layout.addWidget(heading)
        layout.addWidget(copy)
        layout.addStretch(1)
        return page

    @staticmethod
    def _settings_field(
        label: str,
        control: QtWidgets.QWidget,
    ) -> QtWidgets.QWidget:
        field = QtWidgets.QWidget()
        field.setProperty("role", "settings-field")
        field_layout = QtWidgets.QVBoxLayout(field)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(6)
        caption = QtWidgets.QLabel(label)
        caption.setProperty("role", "secondary")
        field_layout.addWidget(caption)
        field_layout.addWidget(control)
        return field

    @staticmethod
    def _page(
        eyebrow: str,
        title: str,
        subtitle: str,
        *,
        action: QtWidgets.QWidget | None = None,
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
        page = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(20)
        header = QtWidgets.QWidget()
        header.setObjectName("settingsPageHeading")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(16)
        heading_copy = QtWidgets.QVBoxLayout()
        heading_copy.setContentsMargins(0, 0, 0, 0)
        heading_copy.setSpacing(4)
        context = QtWidgets.QLabel(str(eyebrow).upper())
        context.setProperty("role", "eyebrow")
        heading = QtWidgets.QLabel(title)
        heading.setObjectName("settingsPageTitle")
        heading.setProperty("role", "title")
        detail = QtWidgets.QLabel(subtitle)
        detail.setWordWrap(True)
        detail.setProperty("role", "secondary")
        heading_copy.addWidget(context)
        heading_copy.addWidget(heading)
        heading_copy.addWidget(detail)
        header_layout.addLayout(heading_copy, 1)
        if action is not None:
            header_layout.addWidget(
                action,
                0,
                QtCore.Qt.AlignmentFlag.AlignTop,
            )
        root.addWidget(header)
        body = QtWidgets.QWidget()
        body.setObjectName("settingsPageBody")
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(22, 0, 22, 22)
        body_layout.setSpacing(12)
        root.addWidget(body, 1)
        return page, body_layout

    def refresh_from_model(self) -> None:
        model = self._view_model
        if model is None:
            return
        draft = model.draft
        self._profile_refreshing = True
        try:
            app = draft.application
            self._set_theme_selection(self._effective_theme or app.theme)
            self.font_scale.setValue(app.font_scale)
            self.font_scale_value.setText(f"{app.font_scale}%")
            self._select_data(self.density, app.density)
            self.reduced_motion.setChecked(app.reduced_motion)
            self._select_data(self.ui_language, app.ui_language)
            self.new_project_location.setText(app.new_project_location)
            self.autosave_interval.setValue(app.autosave_interval_seconds)
            self._select_data(self.open_last_project, app.open_last_project)
            self._select_data(self.source_language, draft.project.source_language)
            self._select_data(self.target_language, draft.project.target_language)
            self._select_data(
                self.output_convention, draft.project.output_convention
            )
            self._select_data(
                self.completed_page_policy, draft.project.completed_page_policy
            )
            self._refresh_sidebar_chrome()
            self._refresh_profiles()
            self._render_module_controls()
            self._refresh_runtime_assets()
            self._refresh_shortcut_bindings()
            self._refresh_pending()
        finally:
            self._profile_refreshing = False

    def set_effective_theme(self, theme: str) -> None:
        """Keep the selected Appearance card aligned with the live shell."""

        value = str(theme or "").strip().casefold()
        if value not in {"dark", "light"}:
            raise ValueError(f"unsupported effective theme: {theme!r}")
        self._effective_theme = value
        self._set_theme_selection(value)

    def _set_theme_selection(self, theme: str) -> None:
        value = str(theme or "").strip().casefold()
        self.dark_theme.setChecked(value == "dark")
        self.light_theme.setChecked(value == "light")

    def set_project_scope(self, project_name: str | None) -> None:
        """Present the currently loaded project in the prototype scope card."""

        normalized = str(project_name or "").strip()
        self._project_scope_name = normalized or "No project open"
        self.project_scope_value.setText(self._project_scope_name)

    def refresh_icons(self, theme: str) -> None:
        value = str(theme)
        self.search_action.setIcon(hybrid_icon("search", value))
        self.add_provider.setIcon(
            hybrid_icon("new", value, secondary=True)
        )
        self.test_provider.setIcon(
            hybrid_icon("play", value, secondary=True)
        )
        self.provider_apply_button.setIcon(
            hybrid_icon("check", value, active=True)
        )
        self._refresh_sidebar_chrome(value)

    def _refresh_sidebar_chrome(self, theme: str | None = None) -> None:
        value = theme or (
            str(self._view_model.draft.application.theme or "dark")
            if self._view_model is not None
            else "dark"
        )
        for category_id, (row, icon, caret) in self._category_rows.items():
            item = next(
                (
                    self.category_list.item(index)
                    for index in range(self.category_list.count())
                    if self.category_list.item(index).data(
                        QtCore.Qt.ItemDataRole.UserRole
                    )
                    == category_id
                ),
                None,
            )
            selected = bool(item is not None and item.isSelected())
            icon_name = str(icon.property("iconName") or category_id)
            icon.setPixmap(
                hybrid_icon(icon_name, value, accent=selected).pixmap(17, 17)
            )
            caret.setPixmap(hybrid_icon("caret-right", value).pixmap(11, 11))
            row.setProperty("active", selected)
            row.style().unpolish(row)
            row.style().polish(row)
        self.project_scope_icon.setPixmap(
            hybrid_icon("project-scope", value).pixmap(17, 17)
        )

    def set_effective_run_summary(self, summary: EffectiveRunSummary | None) -> None:
        self._effective_run_summary = summary
        if summary is None:
            self.effective_run_status.setText("No run candidate")
            self.effective_run_status.setProperty("tone", "muted")
            self.effective_run_status.show()
            for card in self.effective_run_cards:
                card.hide()
            self.effective_run_status.style().unpolish(self.effective_run_status)
            self.effective_run_status.style().polish(self.effective_run_status)
            self._refresh_module_stage_presentations()
            return
        self.effective_run_status.setText("Ready" if summary.ready else "Needs attention")
        self.effective_run_status.setProperty("tone", "ready" if summary.ready else "warning")
        provider = summary.provider.split(" (", 1)[0].strip() or summary.provider
        cleanup_token = summary.cleanup_and_style.split("/", 1)[0].strip()
        cleanup = (
            "Lama · local"
            if cleanup_token.casefold() in {"ai", "lama", "lama · automatic style observation"}
            or "lama" in cleanup_token.casefold()
            else cleanup_token
        )
        values = {
            "translation": (provider, "Project override"),
            "cleanup": (cleanup, "Application default"),
            "renderer": ("Hybrid typesetting", "Project policy"),
        }
        full_description = (
            f"{summary.language_pair}; provider {summary.provider}; model {summary.model}; "
            f"detection and OCR {summary.detection_and_ocr}; cleanup and style "
            f"{summary.cleanup_and_style}; runtime {summary.runtime}; snapshot "
            f"{summary.snapshot_id}."
        )
        for key, (value, note) in values.items():
            self.effective_run_value_labels[key].setText(value)
            self.effective_run_note_labels[key].setText(note)
        for card in self.effective_run_cards:
            card.setAccessibleDescription(full_description)
            card.setToolTip(full_description)
            card.show()
        self.effective_run_status.hide()
        self.effective_run_status.style().unpolish(self.effective_run_status)
        self.effective_run_status.style().polish(self.effective_run_status)
        self._refresh_module_stage_presentations()

    def set_pipeline_presentation(
        self,
        lifecycle: PipelineLifecycleEvent | None,
        stage: PipelineStageEvent | None,
    ) -> None:
        """Present the typed pipeline lifecycle without owning execution state."""

        if lifecycle is not None and not isinstance(lifecycle, PipelineLifecycleEvent):
            raise TypeError("lifecycle must be PipelineLifecycleEvent or None")
        if stage is not None and not isinstance(stage, PipelineStageEvent):
            raise TypeError("stage must be PipelineStageEvent or None")
        if lifecycle is not None and stage is not None:
            if lifecycle.run_id != stage.run_id:
                raise ValueError("lifecycle and stage must belong to the same run")
        self._pipeline_lifecycle = lifecycle
        self._pipeline_stage = stage
        self._refresh_module_stage_presentations()

    @staticmethod
    def _module_stage_index(stage: PipelineStage | None) -> int | None:
        if stage in {
            PipelineStage.VALIDATION,
            PipelineStage.INITIALIZATION,
            PipelineStage.PRESCAN,
        }:
            return 0
        if stage is PipelineStage.DETECTION:
            return 0
        if stage in {PipelineStage.OCR, PipelineStage.HIERARCHY}:
            return 1
        if stage in {PipelineStage.TRANSLATION, PipelineStage.SOURCE_GLYPH}:
            return 2
        if stage is PipelineStage.CLEANUP:
            return 3
        if stage in {
            PipelineStage.STYLE,
            PipelineStage.RENDERING,
        }:
            return 4
        if stage in {PipelineStage.PERSISTENCE, PipelineStage.FINALIZING}:
            return 5
        return None

    def _module_details(self) -> dict[str, str]:
        summary = self._effective_run_summary
        if summary is None:
            selected_profile = (
                self._view_model.draft.profile_for_role("translation")
                if self._view_model is not None
                else None
            )
            if selected_profile is not None:
                translation_detail = selected_profile.display_name
            elif (
                self._view_model is not None
                and self._view_model.draft.provider_profiles
            ):
                translation_detail = "Select a provider profile"
            else:
                translation_detail = "Create a provider profile in Providers"
            return {
                "detection": "Source text-area detection",
                "ocr": "Japanese source recognition",
                "translation": translation_detail,
                "cleanup": "Waits for translation",
                "rendering": "Waits for clean base",
            }
        detector_ocr = [
            value.strip()
            for value in summary.detection_and_ocr.split("/", 1)
        ]
        ocr = detector_ocr[1] if len(detector_ocr) > 1 else "OCR"
        source_language = summary.language_pair.split("→", 1)[0].strip()
        provider = summary.provider.split(" (", 1)[0].strip() or summary.provider
        device = "CUDA" if "gpu" in summary.runtime.casefold() else "CPU"
        return {
            "detection": f"Model loaded · {device}",
            "ocr": f"{ocr} · {source_language}",
            "translation": provider,
            "cleanup": "Waits for Translation",
            "rendering": "Waits for clean base",
        }

    def _refresh_module_stage_presentations(self) -> None:
        if not hasattr(self, "_module_policy_states"):
            return
        details = self._module_details()
        lifecycle = self._pipeline_lifecycle
        stage_index = self._module_stage_index(
            self._pipeline_stage.stage if self._pipeline_stage is not None else None
        )
        translation_needs_setup = (
            self._view_model is not None
            and self._view_model.draft.profile_for_role("translation") is None
        )
        pipeline_owns_stage_state = lifecycle is not None and lifecycle.state in {
            PipelineRunState.VALIDATING,
            PipelineRunState.RUNNING,
            PipelineRunState.STOP_REQUESTED,
            PipelineRunState.CANCELLING,
        }
        stage_ids = ("detection", "ocr", "translation", "cleanup", "rendering")
        for index, stage_id in enumerate(stage_ids):
            state = "Configured"
            tone = "ready"
            if lifecycle is not None:
                if lifecycle.state is PipelineRunState.COMPLETED:
                    state = "Ready"
                elif lifecycle.state in {
                    PipelineRunState.RUNNING,
                    PipelineRunState.STOP_REQUESTED,
                    PipelineRunState.CANCELLING,
                    PipelineRunState.FAILED,
                    PipelineRunState.STOPPED,
                } and stage_index is not None:
                    if index < stage_index:
                        state = "Ready"
                    elif index == stage_index:
                        if lifecycle.state is PipelineRunState.RUNNING:
                            state, tone = "Running", "editing"
                        elif lifecycle.state in {
                            PipelineRunState.STOP_REQUESTED,
                            PipelineRunState.CANCELLING,
                        }:
                            state, tone = "Stopping", "warning"
                        elif lifecycle.state is PipelineRunState.FAILED:
                            state, tone = "Failed", "error"
                        else:
                            state, tone = "Stopped", "warning"
                    elif index == stage_index + 1 and lifecycle.state is PipelineRunState.RUNNING:
                        state, tone = "Queued", "queued"
                    else:
                        state, tone = "Waiting", "queued"
                elif lifecycle.state is PipelineRunState.VALIDATING:
                    state, tone = ("Queued", "queued") if index == 0 else ("Waiting", "queued")
            if (
                stage_id == "translation"
                and translation_needs_setup
                and not pipeline_owns_stage_state
            ):
                state, tone = "Needs setup", "warning"
            description = self._module_policy_descriptions[stage_id]
            description.setText(details[stage_id])
            status = self._module_policy_states[stage_id]
            status.setText(state)
            status.setProperty("tone", tone)
            status.setAccessibleName(f"{stage_id} module {state.casefold()}")
            status.style().unpolish(status)
            status.style().polish(status)
        self._sync_module_policy_geometry()

    def apply_changes(self) -> None:
        if self._view_model is None:
            return
        if not self._view_model.dirty:
            return
        self.apply_requested.emit()

    def confirm_applied(self) -> None:
        """Advance the baseline only after the shell publishes every scope."""

        if self._view_model is None:
            return
        self._view_model.apply()
        pending_deletions = tuple(self._pending_credential_deletions)
        self._pending_credential_deletions.clear()
        self._refresh_pending()
        self.applied.emit()
        for reference in pending_deletions:
            self.credential_delete_requested.emit(reference)

    def reject_apply(self) -> None:
        """Keep the editable draft intact after a persistence refusal."""

        self._refresh_pending()

    def cancel_changes(self) -> None:
        if self._view_model is None:
            return
        self._view_model.cancel()
        self._pending_credential_deletions.clear()
        self.refresh_from_model()
        self.cancelled.emit()

    def _select_category(self, row: int) -> None:
        if 0 <= row < self.stack.count():
            self.stack.setCurrentIndex(row)
            if self.CATEGORIES[row][0] == "modules":
                self._render_module_controls()
            self._refresh_pending()
            self._refresh_sidebar_chrome()

    def _set_advanced(self, advanced: bool) -> None:
        self._advanced = bool(advanced)
        self.basic_button.setChecked(not self._advanced)
        self.advanced_button.setChecked(self._advanced)
        self.reset_layout_button.setVisible(self._advanced)
        self._render_module_controls()
        self.glossary.set_advanced(self._advanced)
        if hasattr(self, "profile_list"):
            self._show_selected_profile(self.profile_list.currentRow())

    def _update_application(self, *_args: object) -> None:
        if self._profile_refreshing or self._view_model is None:
            return
        current = self._view_model.draft.application
        updated = replace(
            current,
            theme="light" if self.light_theme.isChecked() else "dark",
            density=str(self.density.currentData()),
            font_scale=int(self.font_scale.value()),
            reduced_motion=self.reduced_motion.isChecked(),
            ui_language=str(self.ui_language.currentData()),
            new_project_location=self.new_project_location.text().strip(),
            autosave_interval_seconds=int(self.autosave_interval.value()),
            open_last_project=str(self.open_last_project.currentData()),
        )
        self._view_model.replace_application(updated)
        self._refresh_pending()
        self.appearance_changed.emit(
            updated.theme,
            updated.density,
            updated.font_scale,
            updated.reduced_motion,
        )

    def _update_project(self, *_args: object) -> None:
        if self._profile_refreshing or self._view_model is None:
            return
        current = self._view_model.draft.project
        updated = replace(
            current,
            source_language=str(self.source_language.currentData()),
            target_language=str(self.target_language.currentData()),
            output_convention=str(self.output_convention.currentData()),
            completed_page_policy=str(self.completed_page_policy.currentData()),
        )
        self._view_model.replace_project(updated)
        self._refresh_pending()

    def _reset_project_defaults(self) -> None:
        if self._view_model is None:
            return
        current = self._view_model.draft.project
        self._view_model.replace_project(
            replace(
                current,
                source_language="Japanese",
                target_language="Simplified Chinese",
                output_convention="sibling_output_folder",
                completed_page_policy="open_for_review",
            )
        )
        self.refresh_from_model()

    def _font_scale_changed(self, value: int) -> None:
        snapped = max(100, min(200, int(round(value / 5.0) * 5)))
        if snapped != value:
            self.font_scale.blockSignals(True)
            self.font_scale.setValue(snapped)
            self.font_scale.blockSignals(False)
        self.font_scale_value.setText(f"{snapped}%")
        self._update_application()

    def _refresh_profiles(self) -> None:
        if self._view_model is None:
            return
        selected = self._selected_profile_id()
        active_profile_id = self._view_model.draft.translation_profile_id
        self.profile_list.blockSignals(True)
        self.profile_list.clear()
        icon_for_kind = {
            ProviderKind.GGUF: "provider-gguf",
            ProviderKind.OLLAMA: "provider-ollama",
            ProviderKind.DEEPSEEK: "provider",
            ProviderKind.OPENAI_COMPATIBLE: "provider",
        }
        theme = str(self._view_model.draft.application.theme or "dark")
        for profile in self._view_model.draft.provider_profiles:
            active = " · Active" if profile.profile_id == active_profile_id else ""
            detail = {
                ProviderKind.GGUF: "llama.cpp runtime",
                ProviderKind.OLLAMA: "Local provider",
                ProviderKind.DEEPSEEK: "DeepSeek API",
                ProviderKind.OPENAI_COMPATIBLE: "OpenAI-compatible API",
            }[profile.kind]
            item = QtWidgets.QListWidgetItem(
                f"{profile.display_name}\n{detail} · "
                f"{provider_test_status_label(profile.last_test_status)}{active}"
            )
            accessible_text = item.text()
            item.setData(QtCore.Qt.ItemDataRole.UserRole, profile.profile_id)
            item.setData(
                QtCore.Qt.ItemDataRole.AccessibleTextRole,
                accessible_text,
            )
            item.setText("")
            item.setSizeHint(QtCore.QSize(0, 62))
            self.profile_list.addItem(item)
            row = QtWidgets.QWidget()
            row.setProperty("providerProfileRow", True)
            row.setAccessibleName(accessible_text)
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(9)
            mark = QtWidgets.QLabel()
            mark.setObjectName("providerListMark")
            mark.setFixedSize(33, 33)
            mark.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            mark.setPixmap(
                hybrid_icon(
                    icon_for_kind[profile.kind],
                    theme,
                    active=profile.profile_id == active_profile_id,
                ).pixmap(17, 17)
            )
            copy = QtWidgets.QVBoxLayout()
            copy.setSpacing(3)
            name = QtWidgets.QLabel(profile.display_name)
            name.setObjectName("providerListName")
            subtitle = QtWidgets.QLabel(detail)
            subtitle.setObjectName("providerListDetail")
            copy.addWidget(name)
            copy.addWidget(subtitle)
            state = QtWidgets.QLabel()
            state.setFixedSize(18, 18)
            state.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            state.setPixmap(
                hybrid_icon(
                    "success" if profile.runtime_ready else "status-muted",
                    theme,
                ).pixmap(16, 16)
            )
            row_layout.addWidget(mark)
            row_layout.addLayout(copy, 1)
            row_layout.addWidget(state)
            self.profile_list.setItemWidget(item, row)
        self.profile_list.blockSignals(False)
        if self.profile_list.count():
            row = next(
                (
                    index
                    for index in range(self.profile_list.count())
                    if self.profile_list.item(index).data(QtCore.Qt.ItemDataRole.UserRole)
                    == selected
                ),
                0,
            )
            self.profile_list.setCurrentRow(row)
        else:
            self._show_selected_profile(-1)

    def _selected_profile_id(self) -> str:
        item = self.profile_list.currentItem()
        return str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "") if item else ""

    def _selected_profile(self) -> ProviderProfile | None:
        if self._view_model is None:
            return None
        profile_id = self._selected_profile_id()
        return next(
            (profile for profile in self._view_model.draft.provider_profiles if profile.profile_id == profile_id),
            None,
        )

    def _show_selected_profile(self, _row: int) -> None:
        profile = self._selected_profile()
        theme = (
            str(self._view_model.draft.application.theme or "dark")
            if self._view_model is not None
            else "dark"
        )
        self._profile_refreshing = True
        try:
            enabled = profile is not None
            for control in (
                self.profile_name,
                self.profile_type,
                self.profile_endpoint,
                self.profile_model,
                self.profile_path,
                self.browse_provider_model,
                self.link_credential,
                self.test_provider,
                self.delete_provider,
                self.use_translation_provider,
                self.provider_prompt_style,
                self.provider_context_tokens,
                self.provider_gpu_layers,
                self.provider_threads,
                self.provider_batch,
                self.provider_temperature,
                self.provider_top_p,
                self.provider_max_tokens,
            ):
                control.setEnabled(enabled)
            self.provider_empty_state.setVisible(profile is None)
            self.provider_base_form.setVisible(profile is not None)
            self.provider_safety_host.setVisible(profile is not None)
            self.provider_commit.setVisible(profile is not None)
            self.delete_provider.setVisible(profile is not None)
            if profile is None:
                self.profile_summary_icon.clear()
                self.profile_summary_name.setText("No provider selected")
                self.profile_kind.setText("Select or add a provider profile")
                for control in (
                    self.profile_name,
                    self.profile_endpoint,
                    self.profile_model,
                    self.profile_path,
                ):
                    control.clear()
                self.profile_type.setCurrentIndex(-1)
                self.credential_status.setText(
                    "Not linked — Test connection will ask for an API key"
                )
                self.profile_validation.setText("Add a provider profile to configure it.")
                self.profile_validation.setVisible(False)
                self.profile_endpoint_field.setVisible(False)
                self.profile_model_field.setVisible(False)
                self.profile_path_field.setVisible(False)
                self.provider_advanced_form.setVisible(False)
                self.provider_safety_title.setText("Connection details")
                self.provider_safety_detail.setText(
                    "Choose GGUF, Ollama, or DeepSeek to see the exact connection steps."
                )
                self.active_provider_status.setText("No active translation provider")
                self.active_provider_status.setProperty("tone", "warning")
                self.active_provider_status.setIcon(
                    hybrid_icon("status-muted", theme)
                )
                self.use_translation_provider.setVisible(False)
                return
            self.profile_summary_name.setText(profile.display_name)
            icon_name = {
                ProviderKind.GGUF: "provider-gguf",
                ProviderKind.OLLAMA: "provider-ollama",
                ProviderKind.DEEPSEEK: "provider",
                ProviderKind.OPENAI_COMPATIBLE: "provider",
            }[profile.kind]
            self.profile_summary_icon.setPixmap(
                hybrid_icon(icon_name, theme, active=True).pixmap(20, 20)
            )
            self.profile_name.setText(profile.display_name)
            self._select_data(self.profile_type, profile.kind)
            self.profile_endpoint.setText(profile.endpoint or "")
            self.profile_model.setText(profile.model_id or "")
            self.profile_path.setText(profile.local_model_path or "")
            for control in (
                self.profile_name,
                self.profile_endpoint,
                self.profile_model,
                self.profile_path,
            ):
                control.setCursorPosition(0)
            self.profile_path.setToolTip(profile.local_model_path or "")
            is_gguf = profile.kind is ProviderKind.GGUF
            is_ollama = profile.kind is ProviderKind.OLLAMA
            is_api = profile.kind in {
                ProviderKind.DEEPSEEK,
                ProviderKind.OPENAI_COMPATIBLE,
            }
            safety_tone = "ready"
            if is_gguf:
                self.profile_path.setPlaceholderText("Select a local .gguf model file")
                self.provider_safety_title.setText("Local model connection")
                if profile.gguf_options and profile.gguf_options.n_gpu_layers == -1:
                    safety_tone = "ready"
                    safety_detail = (
                        "Automatic fits the highest safe GPU-layer count immediately "
                        "before Start without changing model quality or the saved profile."
                    )
                else:
                    safety_tone = "warning"
                    safety_detail = (
                        "Manual GPU layers disable Automatic fitting. This exact value "
                        "will be blocked if the current model and memory reserve do not fit."
                    )
                self.provider_safety_detail.setText(safety_detail)
            elif is_ollama:
                self.profile_endpoint.setPlaceholderText("http://127.0.0.1:11434")
                self.profile_model.setPlaceholderText("Installed Ollama model name")
                explicit_model = str(profile.model_id or "").strip()
                if not explicit_model or explicit_model == "auto-detect":
                    safety_tone = "warning"
                    self.provider_safety_title.setText("Ollama model required")
                    self.provider_safety_detail.setText(
                        "Select one explicit Ollama model so size, context, residency, "
                        "and processor allocation can be checked before Start."
                    )
                else:
                    safety_tone = "ready"
                    self.provider_safety_title.setText("Ollama resource inspection")
                    self.provider_safety_detail.setText(
                        "The configured endpoint, exact model, context, and current "
                        "residency are inspected before Start. Remote capacity remains "
                        "server-owned and is disclosed separately."
                    )
            elif profile.kind is ProviderKind.DEEPSEEK:
                self.profile_endpoint.setPlaceholderText("https://api.deepseek.com")
                self.profile_model.setPlaceholderText("deepseek-v4-flash")
                if (
                    profile.credential_ref is not None
                    and profile.credential_ref.kind
                    is CredentialReferenceKind.ENVIRONMENT_VARIABLE
                ):
                    self.provider_safety_title.setText("Environment credential")
                    self.provider_safety_detail.setText(
                        f"{profile.credential_ref.reference} is resolved only while "
                        "testing or starting; no secret is saved in the profile or project."
                    )
                else:
                    self.provider_safety_title.setText(
                        "Secret stays outside project.json"
                    )
                    self.provider_safety_detail.setText(
                        f"Only an opaque {self._platform_copy.credential_store_label} reference is saved with this profile."
                    )
            elif is_api:
                self.profile_endpoint.setPlaceholderText("OpenAI-compatible API endpoint")
                self.profile_model.setPlaceholderText("Provider model ID")
                if (
                    profile.credential_ref is not None
                    and profile.credential_ref.kind
                    is CredentialReferenceKind.ENVIRONMENT_VARIABLE
                ):
                    self.provider_safety_title.setText("Environment credential")
                    self.provider_safety_detail.setText(
                        f"{profile.credential_ref.reference} is resolved only while "
                        "testing or starting; no secret is saved in the profile or project."
                    )
                else:
                    self.provider_safety_title.setText(
                        "Secret stays outside project.json"
                    )
                    self.provider_safety_detail.setText(
                        f"Only an opaque {self._platform_copy.credential_store_label} reference is saved with this profile."
                    )
            self.profile_endpoint_field.setVisible(not is_gguf)
            self.profile_model_field.setVisible(not is_gguf)
            self.profile_path_field.setVisible(is_gguf)
            self.provider_advanced_form.setVisible(self._advanced)
            runtime_visible = bool(self._advanced and (is_gguf or is_ollama))
            self.provider_runtime_label.setVisible(runtime_visible)
            self.provider_prompt_style_label.setVisible(
                bool(self._advanced and is_gguf)
            )
            self.provider_prompt_style.setVisible(bool(self._advanced and is_gguf))
            self.provider_context_tokens_label.setVisible(runtime_visible)
            self.provider_context_tokens.setVisible(runtime_visible)
            for control in (
                self.provider_gpu_layers_label,
                self.provider_gpu_layers,
                self.provider_threads_label,
                self.provider_threads,
                self.provider_batch_label,
                self.provider_batch,
            ):
                control.setVisible(bool(self._advanced and is_gguf))
            generation_visible = self._advanced
            for control in (
                self.provider_generation_label,
                self.provider_temperature_label,
                self.provider_temperature,
                self.provider_top_p_label,
                self.provider_top_p,
                self.provider_max_tokens_label,
                self.provider_max_tokens,
            ):
                control.setVisible(generation_visible)
            if is_gguf:
                options = profile.gguf_options or GGUFProviderOptions()
                self._select_data(self.provider_prompt_style, options.prompt_style)
                self.provider_context_tokens.setValue(options.n_ctx)
                self.provider_gpu_layers.setValue(options.n_gpu_layers)
                self.provider_threads.setValue(options.n_threads)
                self.provider_batch.setValue(options.n_batch)
            elif is_ollama:
                options = profile.ollama_options or OllamaProviderOptions()
                self.provider_context_tokens.setValue(options.context_tokens)
            generation = profile.generation_defaults
            self.provider_temperature.setValue(generation.temperature)
            self.provider_top_p.setValue(generation.top_p)
            self.provider_max_tokens.setValue(generation.max_output_tokens or 0)
            supports_credentials = (
                ProviderCapability.CREDENTIAL_REFERENCE in profile.capabilities
            )
            credential = profile.credential_ref
            if supports_credentials:
                if credential is None:
                    self.credential_status.setText("Not linked")
                    self.credential_status.setToolTip(
                        "Test connection will ask for an API key"
                    )
                elif credential.kind is CredentialReferenceKind.ENVIRONMENT_VARIABLE:
                    self.credential_status.setText(
                        f"Environment variable: {credential.reference}"
                    )
                    self.credential_status.setToolTip(
                        "The environment variable is resolved when testing or starting; "
                        "its value is never stored in the profile"
                    )
                else:
                    self.credential_status.setText("Credential reference linked")
                    self.credential_status.setToolTip(
                        "Test connection will verify whether the referenced credential is available"
                    )
            else:
                self.credential_status.setText("Not required")
                self.credential_status.setToolTip(
                    "This provider does not use a stored credential"
                )
            self.credential_status.setCursorPosition(0)
            issues = profile.configuration_issues
            test_blocking_issues = tuple(
                issue
                for issue in issues
                if issue != "credential_reference_required"
            )
            issue_text = " · ".join(
                _PROVIDER_ISSUE_LABELS.get(
                    issue,
                    issue.replace("_", " ").capitalize(),
                )
                for issue in issues
            )
            self.profile_validation.setText(
                (
                    "Ready to validate the local model file"
                    if is_gguf and not issues
                    else "Ready for connection testing"
                    if not issues
                    else "Ready to enter an API key and test this connection"
                    if issues == ("credential_reference_required",)
                    else issue_text
                )
            )
            self.link_credential.setVisible(supports_credentials)
            self.link_credential.setEnabled(supports_credentials)
            self.link_credential.setText(
                (
                    "Replace" if credential is not None else "Enter API key"
                )
                if supports_credentials
                else "No credential required"
            )
            self.link_credential.setAccessibleName(self.link_credential.text())
            self.link_credential.setToolTip(
                (
                    "Enter an API key for one test, then optionally save it in "
                    f"{self._platform_copy.credential_store_label}"
                )
                if supports_credentials
                else "This provider does not use a stored credential"
            )
            supports_test = bool(
                ProviderCapability.CONNECTION_TEST in profile.capabilities
                or is_gguf
            )
            self.test_provider.setText(
                "Validate model file" if is_gguf else "Test connection"
            )
            self.test_provider.setAccessibleName(self.test_provider.text())
            self.test_provider.setEnabled(not test_blocking_issues and supports_test)
            if issues == ("credential_reference_required",) and supports_test:
                self.test_provider.setToolTip(
                    "Enter an API key, test the connection, and optionally save it "
                    f"in {self._platform_copy.credential_store_label}"
                )
            elif test_blocking_issues:
                self.test_provider.setToolTip(
                    " · ".join(
                        _PROVIDER_ISSUE_LABELS.get(
                            issue,
                            issue.replace("_", " ").capitalize(),
                        )
                        for issue in test_blocking_issues
                    )
                )
            elif not supports_test:
                self.test_provider.setToolTip(
                    "This provider does not expose a connection test"
                )
            else:
                self.test_provider.setToolTip("")
            active = bool(
                self._view_model is not None
                and self._view_model.draft.translation_profile_id
                == profile.profile_id
            )
            profile_detail = {
                ProviderKind.GGUF: "llama.cpp runtime",
                ProviderKind.OLLAMA: "Local provider",
                ProviderKind.DEEPSEEK: "DeepSeek API",
                ProviderKind.OPENAI_COMPATIBLE: "OpenAI-compatible API",
            }[profile.kind]
            configured = not bool(issues)
            tested = profile.last_test_status is not ProviderTestStatus.NOT_TESTED
            self.profile_kind.setText(
                provider_lifecycle_summary(
                    configured=configured,
                    tested=tested,
                    active=active,
                )
            )
            self.profile_kind.setAccessibleDescription(
                (
                    f"{profile_detail}. Used by {self._project_scope_name}."
                    if active and self._project_scope_name != "No project open"
                    else f"{profile_detail}. Public profile data."
                )
            )
            if profile.runtime_ready:
                connection_text = "Connected"
                connection_tone = "ready"
            elif profile.last_test_status in {
                ProviderTestStatus.ERROR,
                ProviderTestStatus.UNAVAILABLE,
            }:
                connection_text = "Connection failed"
                connection_tone = "error"
            else:
                connection_text = "Not tested"
                connection_tone = "muted"
            self.active_provider_status.setText(connection_text)
            self.active_provider_status.setIcon(
                hybrid_icon(
                    "success"
                    if profile.runtime_ready
                    else "warning"
                    if connection_tone == "error"
                    else "status-muted",
                    theme,
                )
            )
            self.active_provider_status.setProperty(
                "tone", connection_tone
            )
            self.active_provider_status.style().unpolish(self.active_provider_status)
            self.active_provider_status.style().polish(self.active_provider_status)
            self.use_translation_provider.setEnabled(
                bool(
                    not active
                    and profile.transport_available
                    and profile.runtime_ready
                )
            )
            self.use_translation_provider.setVisible(not active)
            self.profile_validation.setVisible(
                bool(
                    issues
                    or profile.last_test_status
                    in {
                        ProviderTestStatus.ERROR,
                        ProviderTestStatus.UNAVAILABLE,
                    }
                )
            )
            test_message = self._provider_test_messages.get(profile.profile_id)
            if test_message is not None:
                message, warning = test_message
                self.profile_validation.setText(message)
                self.profile_validation.setProperty(
                    "tone", "error" if warning else "ready"
                )
                self.profile_validation.setVisible(True)
                self.profile_validation.style().unpolish(self.profile_validation)
                self.profile_validation.style().polish(self.profile_validation)
            self.provider_safety_icon.setPixmap(
                hybrid_icon("shield", theme).pixmap(18, 18)
            )
            self.provider_safety_callout.setProperty("tone", safety_tone)
            self.provider_safety_callout.style().unpolish(
                self.provider_safety_callout
            )
            self.provider_safety_callout.style().polish(
                self.provider_safety_callout
            )
            self.use_translation_provider.setToolTip(
                (
                    ""
                    if profile.runtime_ready
                    else "Test this exact provider configuration before selecting it"
                )
                if profile.transport_available
                else "Translation transport is not available for this provider type"
            )
        finally:
            self._profile_refreshing = False

    def _update_profile(self) -> None:
        if self._profile_refreshing or self._view_model is None:
            return
        profile = self._selected_profile()
        if profile is None:
            return
        try:
            gguf_options = profile.gguf_options
            ollama_options = profile.ollama_options
            if profile.kind is ProviderKind.GGUF:
                gguf_options = GGUFProviderOptions(
                    prompt_style=str(self.provider_prompt_style.currentData()),
                    n_ctx=int(self.provider_context_tokens.value()),
                    n_gpu_layers=int(self.provider_gpu_layers.value()),
                    n_threads=int(self.provider_threads.value()),
                    n_batch=int(self.provider_batch.value()),
                )
            elif profile.kind is ProviderKind.OLLAMA:
                ollama_options = OllamaProviderOptions(
                    context_tokens=int(self.provider_context_tokens.value())
                )
            updated = replace(
                profile,
                display_name=self.profile_name.text().strip(),
                endpoint=self.profile_endpoint.text().strip() or None,
                model_id=self.profile_model.text().strip() or None,
                local_model_path=self.profile_path.text().strip() or None,
                gguf_options=gguf_options,
                ollama_options=ollama_options,
                generation_defaults=GenerationSettings(
                    temperature=float(self.provider_temperature.value()),
                    top_p=float(self.provider_top_p.value()),
                    max_output_tokens=(
                        int(self.provider_max_tokens.value()) or None
                    ),
                    stop_sequences=profile.generation_defaults.stop_sequences,
                ),
            )
            if (
                updated.public_configuration_fingerprint
                != profile.public_configuration_fingerprint
            ):
                self._provider_test_messages[profile.profile_id] = (
                    "Configuration changed. Re-test this provider; Start will run "
                    "a new memory-safety check.",
                    True,
                )
            profiles = tuple(
                updated if item.profile_id == profile.profile_id else item
                for item in self._view_model.draft.provider_profiles
            )
            self._view_model.replace_profiles(profiles)
        except (TypeError, ValueError) as exc:
            self.profile_validation.setText(str(exc))
            self.profile_validation.setProperty("tone", "error")
            return
        self._refresh_profiles()
        self._refresh_pending()

    def _change_provider_type(self) -> None:
        """Replace one draft profile through the prototype's explicit type control."""

        if self._profile_refreshing or self._view_model is None:
            return
        profile = self._selected_profile()
        raw_kind = self.profile_type.currentData()
        if profile is None or raw_kind is None:
            return
        try:
            kind = ProviderKind(raw_kind)
        except (TypeError, ValueError):
            return
        if kind is profile.kind:
            return
        defaults: dict[str, Any] = {
            "endpoint": None,
            "model_id": None,
            "local_model_path": None,
        }
        if kind is ProviderKind.OLLAMA:
            defaults.update(
                endpoint="http://127.0.0.1:11434",
                model_id=None,
            )
        elif kind is ProviderKind.DEEPSEEK:
            defaults.update(
                endpoint="https://api.deepseek.com",
                model_id="deepseek-v4-flash",
            )
        elif kind is ProviderKind.OPENAI_COMPATIBLE:
            defaults.update(endpoint="https://api.openai.com/v1")
        updated = ProviderProfile(
            profile_id=profile.profile_id,
            display_name=profile.display_name,
            kind=kind,
            request_policy=profile.request_policy,
            generation_defaults=profile.generation_defaults,
            last_test_status=ProviderTestStatus.NOT_TESTED,
            **defaults,
        )
        was_active = (
            self._view_model.draft.translation_profile_id == profile.profile_id
        )
        self._view_model.replace_profiles(
            updated if item.profile_id == profile.profile_id else item
            for item in self._view_model.draft.provider_profiles
        )
        if was_active and not updated.transport_available:
            self._view_model.select_translation_provider(None)
        self._refresh_profiles()
        self._refresh_pending()

    def _add_profile(self, kind: ProviderKind) -> None:
        if self._view_model is None:
            return
        profile_id = f"profile-{uuid.uuid4().hex[:12]}"
        kwargs: dict[str, Any] = {
            "profile_id": profile_id,
            "display_name": f"New {provider_kind_label(kind)} profile",
            "kind": kind,
        }
        if kind is ProviderKind.OLLAMA:
            kwargs.update(endpoint="http://127.0.0.1:11434", model_id=None)
        elif kind is ProviderKind.DEEPSEEK:
            kwargs.update(
                endpoint="https://api.deepseek.com",
                model_id="deepseek-v4-flash",
            )
        elif kind is ProviderKind.OPENAI_COMPATIBLE:
            kwargs.update(endpoint="https://api.openai.com/v1", model_id="")
        profile = ProviderProfile(**kwargs)
        self._view_model.replace_profiles((*self._view_model.draft.provider_profiles, profile))
        if self._view_model.draft.translation_profile_id is None and profile.transport_available:
            self._view_model.select_translation_provider(profile.profile_id)
        self._refresh_profiles()
        self.profile_list.setCurrentRow(self.profile_list.count() - 1)
        self._refresh_pending()

    def _remove_profile(self) -> None:
        if self._view_model is None:
            return
        profile = self._selected_profile()
        if profile is None:
            return
        if not HybridConfirmDialog.ask(
            self,
            title="Delete provider profile",
            message=(
                f"Delete {profile.display_name} from provider profiles? "
                "The profile is removed only after Apply changes."
            ),
            confirm_text="Delete profile",
            destructive=True,
        ):
            return
        credential = profile.credential_ref
        if (
            credential is not None
            and credential.kind
            in {
                CredentialReferenceKind.WINDOWS_CREDENTIAL,
                CredentialReferenceKind.SYSTEM_KEYRING,
            }
        ):
            delete_credential = HybridConfirmDialog.ask(
                self,
                title="Delete stored credential too?",
                message=(
                    f"Also delete this profile's saved API credential from "
                    f"{self._platform_copy.credential_store_label}? Keep it to remove only the profile."
                ),
                confirm_text="Delete credential",
                cancel_text="Keep credential",
                destructive=True,
            )
            if (
                delete_credential
                and credential not in self._pending_credential_deletions
            ):
                self._pending_credential_deletions.append(credential)
        self._view_model.replace_profiles(
            profile_item
            for profile_item in self._view_model.draft.provider_profiles
            if profile_item.profile_id != profile.profile_id
        )
        for role in ("translation", "discovery"):
            if self._view_model.draft.project.provider_profile_references.get(role) == profile.profile_id:
                self._view_model.select_provider(role, None)
        self._provider_test_messages.pop(profile.profile_id, None)
        self._refresh_profiles()
        self._refresh_pending()

    def _test_provider(self) -> None:
        profile = self._selected_profile()
        if profile is not None:
            if (
                profile.credential_ref is None
                and ProviderCapability.CREDENTIAL_REFERENCE in profile.capabilities
            ):
                self.credential_link_requested.emit(profile.profile_id)
            else:
                self.provider_test_requested.emit(profile.profile_id)

    def set_provider_test_result_message(
        self,
        profile_id: str,
        message: str,
        *,
        warning: bool,
    ) -> None:
        identity = str(profile_id or "").strip()
        detail = str(message or "").strip()
        if not identity or not detail:
            return
        self._provider_test_messages[identity] = (detail, bool(warning))
        profile = self._selected_profile()
        if profile is not None and profile.profile_id == identity:
            self.profile_validation.setText(detail)
            self.profile_validation.setProperty(
                "tone", "error" if warning else "ready"
            )
            self.profile_validation.setVisible(True)
            self.profile_validation.style().unpolish(self.profile_validation)
            self.profile_validation.style().polish(self.profile_validation)

    def _use_translation_provider(self) -> None:
        if self._view_model is None:
            return
        profile = self._selected_profile()
        if profile is None or not profile.transport_available:
            return
        self._view_model.select_translation_provider(profile.profile_id)
        self._refresh_profiles()
        self._refresh_pending()

    def _browse_provider_model(self) -> None:
        profile = self._selected_profile()
        if profile is None or profile.kind is not ProviderKind.GGUF:
            return
        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select local GGUF model",
            self.profile_path.text().strip(),
            "GGUF models (*.gguf);;All files (*)",
        )
        if not selected:
            return
        self.profile_path.setText(selected)
        self._update_profile()

    def _link_credential(self) -> None:
        profile = self._selected_profile()
        if profile is not None:
            self.credential_link_requested.emit(profile.profile_id)

    def _sync_module_policy_geometry(self) -> None:
        """Apply one shared native column contract to every module card."""

        if not hasattr(self, "_module_policy_forms"):
            return
        geometry = MODULE_POLICY_GEOMETRY
        identity_width = geometry.identity_width(
            compact=self._module_policy_compact_identity
        )
        for identity in self._module_policy_identity_hosts.values():
            identity.setFixedWidth(identity_width)

        for stage_id, layout in self._module_policy_layouts.items():
            number = self._module_policy_indices[stage_id]
            identity = self._module_policy_identity_hosts[stage_id]
            form = self._module_policy_forms[stage_id]
            state = self._module_policy_states[stage_id]
            layout.removeWidget(number)
            layout.removeWidget(identity)
            layout.removeItem(form)
            layout.removeWidget(state)
            for column in range(4):
                layout.setColumnStretch(column, 0)
            if self._module_policy_reflow:
                # Keep stage identity and state in one readable header, then
                # give the controls the full remaining card width. This is a
                # real reflow, not a clipped horizontal desktop row.
                layout.addWidget(
                    number,
                    0,
                    0,
                    2,
                    1,
                    QtCore.Qt.AlignmentFlag.AlignTop,
                )
                layout.addWidget(
                    identity,
                    0,
                    1,
                    QtCore.Qt.AlignmentFlag.AlignTop,
                )
                layout.addWidget(
                    state,
                    0,
                    2,
                    QtCore.Qt.AlignmentFlag.AlignTop
                    | QtCore.Qt.AlignmentFlag.AlignRight,
                )
                layout.addLayout(form, 1, 1, 1, 2)
                layout.setColumnStretch(1, 1)
            else:
                layout.addWidget(
                    number,
                    0,
                    0,
                    QtCore.Qt.AlignmentFlag.AlignTop,
                )
                layout.addWidget(
                    identity,
                    0,
                    1,
                    QtCore.Qt.AlignmentFlag.AlignTop,
                )
                layout.addLayout(form, 0, 2)
                layout.addWidget(
                    state,
                    0,
                    3,
                    QtCore.Qt.AlignmentFlag.AlignTop,
                )
                layout.setColumnStretch(2, 1)

        labels: list[QtWidgets.QWidget] = []
        for form in self._module_policy_forms.values():
            form.setHorizontalSpacing(geometry.form_horizontal_spacing)
            form.setVerticalSpacing(geometry.form_vertical_spacing)
            form.setRowWrapPolicy(
                QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows
                if self._module_policy_reflow
                else QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows
            )
            for row in range(form.rowCount()):
                item = form.itemAt(
                    row,
                    QtWidgets.QFormLayout.ItemRole.LabelRole,
                )
                if item is None or item.widget() is None:
                    continue
                label = item.widget()
                label.setMinimumWidth(0)
                label.setMaximumWidth(16_777_215)
                labels.append(label)

        if labels and not self._module_policy_reflow:
            label_width = max(
                geometry.minimum_label_width,
                *(label.sizeHint().width() for label in labels),
            )
            for label in labels:
                label.setFixedWidth(label_width)

        states = tuple(self._module_policy_states.values())
        for state in states:
            state.setMinimumWidth(0)
            state.setMaximumWidth(16_777_215)
        if states:
            status_width = max(
                geometry.minimum_status_width,
                *(state.sizeHint().width() for state in states),
            )
            for state in states:
                state.setFixedWidth(status_width)
        self.modules_layout.invalidate()
        self.modules_content.updateGeometry()

    def _render_module_controls(self) -> None:
        for form in self._module_policy_forms.values():
            while form.rowCount():
                form.removeRow(0)
        self._module_controls.clear()
        if self._view_model is None:
            return
        configs = {
            config.module_id: config
            for config in self._view_model.draft.module_configs
        }

        def configured_value(qualified_id: str) -> tuple[SettingDefinition, Any]:
            definition = self._registry.get_setting(qualified_id)
            config = configs.get(definition.module_id)
            value = (
                config.values.get(definition.setting_id, definition.default)
                if config is not None
                else definition.default
            )
            return definition, value

        def fixed_backend(
            qualified_id: str,
            text: str,
            description: str,
        ) -> QtWidgets.QLineEdit:
            control = QtWidgets.QLineEdit(text)
            control.setObjectName("moduleFixedBackend")
            control.setReadOnly(True)
            control.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            control.setAccessibleName(qualified_id)
            control.setAccessibleDescription(description)
            control.setToolTip(description)
            return control

        if not self._advanced:
            detection_definition, detection_value = configured_value(
                "detection.engine"
            )
            detection = fixed_backend(
                detection_definition.qualified_id,
                str(detection_value),
                "The detection backend is fixed by the supported module registry.",
            )
            detection.setProperty("moduleStage", "detection")
            self._module_policy_forms["detection"].addRow("Backend", detection)
            self._module_controls[detection_definition.qualified_id] = detection

            ocr_definition, ocr_value = configured_value("ocr.engine")
            ocr = self._setting_control(ocr_definition, ocr_value)
            ocr.setObjectName("moduleBackendSelector")
            ocr.setProperty("moduleStage", "ocr")
            ocr.setAccessibleName("OCR backend")
            ocr.setAccessibleDescription(ocr_definition.description)
            self._module_policy_forms["ocr"].addRow("Engine", ocr)
            self._module_controls[ocr_definition.qualified_id] = ocr

            profiles = tuple(self._view_model.draft.provider_profiles)
            if profiles:
                translation: QtWidgets.QWidget = HybridComboBox()
                translation.setObjectName("moduleBackendSelector")
                translation.setProperty("moduleStage", "translation")
                selected_profile_id = self._view_model.draft.translation_profile_id
                if not any(
                    profile.profile_id == selected_profile_id for profile in profiles
                ):
                    translation.addItem("Select a provider profile", None)
                    translation.setItemData(
                        0,
                        "Choose the provider profile used by future translation runs.",
                        QtCore.Qt.ItemDataRole.ToolTipRole,
                    )
                for profile in profiles:
                    translation.addItem(profile.display_name, profile.profile_id)
                    translation.setItemData(
                        translation.count() - 1,
                        (
                            f"{provider_kind_label(profile.kind)} · "
                            f"{provider_test_status_label(profile.last_test_status)}"
                        ),
                        QtCore.Qt.ItemDataRole.ToolTipRole,
                    )
                self._select_data(
                    translation,
                    self._view_model.draft.translation_profile_id,
                )
                translation.currentIndexChanged.connect(
                    lambda _index, widget=translation: (
                        self._set_translation_module_profile(widget.currentData())
                    )
                )
                translation.setAccessibleName("Translation provider profile")
                translation.setAccessibleDescription(
                    "Select one configured provider profile for future translation runs."
                )
            else:
                translation = fixed_backend(
                    "translation.provider_profile",
                    "No provider profiles configured",
                    "Create and test a provider profile in Providers before translation.",
                )
                translation.setProperty("moduleStage", "translation")
            self._module_policy_forms["translation"].addRow(
                "Provider", translation
            )
            self._module_controls["translation.provider_profile"] = translation

            cleanup_definition, cleanup_value = configured_value(
                "cleanup.inpaint_mode"
            )
            cleanup_label = (
                "AI inpainting" if cleanup_value == "ai" else str(cleanup_value)
            )
            cleanup = fixed_backend(
                cleanup_definition.qualified_id,
                cleanup_label,
                "AI inpainting is the only supported cleanup policy; legacy values remain history only.",
            )
            cleanup.setProperty("moduleStage", "cleanup")
            self._module_policy_forms["cleanup"].addRow("Policy", cleanup)
            self._module_controls[cleanup_definition.qualified_id] = cleanup

            rendering = fixed_backend(
                "rendering.pipeline",
                "Hybrid typesetting",
                "Rendering combines automatic style evidence with explicit user overrides.",
            )
            rendering.setProperty("moduleStage", "rendering")
            self._module_policy_forms["rendering"].addRow("Renderer", rendering)
            self._module_controls["rendering.pipeline"] = rendering
            self._filter_visible_controls(self.search.text())
            self._refresh_module_stage_presentations()
            self._sync_module_policy_geometry()
            return
        visible = self._registry.visible_settings(advanced=self._advanced)
        stage_for_module = {
            "detection": "detection",
            "ocr": "ocr",
            "translation": "translation",
            "cleanup": "cleanup",
            "source_style": "rendering",
            "renderer": "rendering",
            "runtime": "rendering",
        }
        for module in self._registry.modules:
            definitions = [item for item in visible if item.module_id == module.module_id]
            if not definitions:
                continue
            stage_id = stage_for_module[module.module_id]
            form = self._module_policy_forms[stage_id]
            row = next(
                item
                for item in self._module_policy_rows
                if item.property("moduleStage") == stage_id
            )
            existing_search = str(row.property("searchText") or "")
            row.setProperty(
                "searchText",
                (
                    existing_search
                    + " "
                    + module.module_id
                    + " "
                    + " ".join(item.setting_id for item in definitions)
                ).casefold(),
            )
            config = configs.get(module.module_id)
            for definition in definitions:
                value = (
                    config.values.get(definition.setting_id, definition.default)
                    if config is not None
                    else definition.default
                )
                control = self._setting_control(definition, value)
                control.setAccessibleName(definition.qualified_id)
                control.setAccessibleDescription(definition.description)
                label = {
                    "renderer.font_name": "Default Output Font",
                    "translation.gguf_cross_page_context": (
                        "GGUF Cross-Page Context"
                    ),
                    "cleanup.inpaint_model_id": "Inpaint Model ID",
                    "runtime.use_gpu": (
                        "Use GPU"
                        if self._platform_copy.accelerator_label == "CUDA"
                        else "Allow acceleration"
                    ),
                }.get(
                    definition.qualified_id,
                    definition.setting_id.replace("_", " ").title(),
                )
                form.addRow(label, control)
                if definition.qualified_id == "renderer.font_name":
                    detail = QtWidgets.QLabel(definition.description)
                    detail.setWordWrap(True)
                    detail.setProperty("role", "secondary")
                    detail.setAccessibleName("Output font behavior")
                    form.addRow("", detail)
                self._module_controls[definition.qualified_id] = control
        self._filter_visible_controls(self.search.text())
        self._refresh_module_stage_presentations()
        self._sync_module_policy_geometry()

    def _validate_module_policies(self) -> None:
        if self._view_model is None:
            return
        for row in self._module_policy_rows:
            state = row.findChild(QtWidgets.QLabel, "modulePolicyState")
            if state is None:
                continue
            state.setText("Configured")
            state.setProperty("tone", "ready")
            state.setAccessibleName(
                f"{row.property('moduleStage')} module policy configured"
            )
            state.style().unpolish(state)
            state.style().polish(state)
        self._refresh_module_stage_presentations()

    def set_runtime_status(self, status: RuntimeStatus | None) -> None:
        if self._view_model is None:
            return
        self._view_model.set_runtime_status(status)
        self._refresh_runtime_assets()

    def set_runtime_checking(self, checking: bool) -> None:
        self._runtime_checking = bool(checking)
        self.runtime_verify_all_button.setEnabled(not self._runtime_checking)
        self.runtime_verify_all_button.setText(
            "Verifying…" if self._runtime_checking else "Verify all"
        )
        self.runtime_verify_all_button.setAccessibleDescription(
            "Local runtime verification is active."
            if self._runtime_checking
            else "Check installed local runtime assets without loading a model."
        )
        self._refresh_runtime_assets()

    def set_runtime_downloading(self, asset_id: str | None) -> None:
        normalized = str(asset_id or "").strip()
        if normalized and normalized not in self.runtime_asset_rows:
            raise ValueError("unsupported runtime asset")
        self._runtime_downloading_asset_id = normalized
        self._refresh_runtime_assets()

    def accept_shortcut_binding(self, shortcut_id: str, sequence: str) -> None:
        """Publish one validated application-scoped shortcut draft."""

        if self._view_model is None:
            raise RuntimeError("settings model is unavailable")
        if shortcut_id not in self._shortcut_commands:
            raise ValueError("unsupported shortcut command")
        normalized = QtGui.QKeySequence(str(sequence)).toString(
            QtGui.QKeySequence.SequenceFormat.PortableText
        )
        if not normalized:
            raise ValueError("shortcut sequence is required")
        current = self._view_model.draft.application
        bindings = dict(current.shortcut_bindings)
        duplicate = next(
            (
                command_id
                for command_id, value in bindings.items()
                if command_id != shortcut_id
                and QtGui.QKeySequence(value).matches(QtGui.QKeySequence(normalized))
                is QtGui.QKeySequence.SequenceMatch.ExactMatch
            ),
            "",
        )
        if duplicate:
            raise ValueError(
                f"Shortcut is already assigned to {self._shortcut_commands[duplicate]}."
            )
        bindings[shortcut_id] = normalized
        updated = replace(current, shortcut_bindings=bindings)
        self._view_model.replace_application(updated)
        self._refresh_shortcut_bindings()
        self._refresh_pending()
        self.shortcut_change_requested.emit(shortcut_id)
        self.shortcuts_changed.emit(dict(updated.shortcut_bindings))

    def _open_shortcut_dialog(self, shortcut_id: str) -> None:
        if self._view_model is None:
            return
        command = self._shortcut_commands[shortcut_id]
        dialog = HybridDialog(self)
        dialog.setObjectName("shortcutCaptureDialog")
        dialog.setWindowTitle(f"Change {command} shortcut")
        dialog.setModal(True)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(16)
        layout.addWidget(
            dialog.create_dialog_header(
                title=f"Change {command} shortcut",
                subtitle=(
                    f"Press the key combination to use for {command.lower()}."
                ),
                icon_name="shortcuts",
                close_accessible_name="Cancel shortcut change",
            )
        )
        editor = QtWidgets.QKeySequenceEdit(
            QtGui.QKeySequence(
                self._view_model.draft.application.shortcut_bindings[shortcut_id]
            )
        )
        editor.setObjectName("shortcutSequenceEdit")
        editor.setMaximumSequenceLength(1)
        editor.setAccessibleName(f"New {command} shortcut")
        layout.addWidget(editor)
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.setProperty("role", "command")
        cancel_button.setProperty("variant", "quiet")
        cancel_button.setAccessibleName("Cancel shortcut change")
        cancel_button.clicked.connect(dialog.reject)
        apply_button = QtWidgets.QPushButton("Apply shortcut")
        apply_button.setProperty("role", "command")
        apply_button.setProperty("variant", "primary")
        apply_button.setAccessibleName("Apply shortcut")
        apply_button.clicked.connect(dialog.accept)
        apply_button.setDefault(True)
        footer.addWidget(cancel_button)
        footer.addWidget(apply_button)
        layout.addLayout(footer)
        QtWidgets.QWidget.setTabOrder(editor, cancel_button)
        QtWidgets.QWidget.setTabOrder(cancel_button, apply_button)
        if dialog.dialog_header is not None:
            QtWidgets.QWidget.setTabOrder(
                apply_button,
                dialog.dialog_header.close_button,
            )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        try:
            self.accept_shortcut_binding(
                shortcut_id,
                editor.keySequence().toString(
                    QtGui.QKeySequence.SequenceFormat.PortableText
                ),
            )
        except ValueError as exc:
            self.pending_status.setText(str(exc))
            self.pending_status.setProperty("tone", "error")
            self.pending_status.setAccessibleDescription(str(exc))
            self.pending_status.style().unpolish(self.pending_status)
            self.pending_status.style().polish(self.pending_status)

    def _refresh_shortcut_bindings(self) -> None:
        if not hasattr(self, "_shortcut_binding_edits"):
            return
        bindings = (
            self._view_model.draft.application.shortcut_bindings
            if self._view_model is not None
            else DEFAULT_SHORTCUT_BINDINGS
        )
        for shortcut_id, editor in self._shortcut_binding_edits.items():
            editor.setText(str(bindings[shortcut_id]))

    def _refresh_runtime_assets(self) -> None:
        if not hasattr(self, "runtime_asset_rows"):
            return
        status = self._view_model.runtime_status if self._view_model is not None else None
        installed = status.installed_assets if status is not None else {}
        for asset_id, (description, state, action) in self.runtime_asset_rows.items():
            if self._runtime_downloading_asset_id:
                active = asset_id == self._runtime_downloading_asset_id
                if active:
                    state.setText("Downloading")
                    state.setProperty("tone", "editing")
                    action.setText("Downloading…")
                action.setEnabled(False)
                state.style().unpolish(state)
                state.style().polish(state)
                continue
            if self._runtime_checking:
                state.setText("Checking")
                state.setProperty("tone", "editing")
                action.setEnabled(False)
                state.style().unpolish(state)
                state.style().polish(state)
                continue
            raw = installed.get(asset_id)
            ready = False
            managed_download = bool(
                self._runtime_asset_specs[asset_id].preparer is not None
            )
            detail = description.text()
            if isinstance(raw, bool):
                ready = raw
            elif isinstance(raw, Mapping):
                ready = bool(raw.get("ready") or raw.get("installed"))
                detail = str(raw.get("detail") or raw.get("version") or detail)
                managed_download = bool(
                    raw.get("managed_download", managed_download)
                )
            elif raw is not None:
                ready = str(raw).casefold() in {"ready", "installed", "true"}
                detail = str(raw)
            label = "Ready" if ready else "Not installed" if raw is not None else "Not checked"
            tone = "ready" if ready else "warning" if raw is not None else "muted"
            description.setText(detail)
            state.setText(label)
            state.setProperty("tone", tone)
            state.setAccessibleName(f"{asset_id} runtime asset: {label}")
            action.setText(
                "Details"
                if ready or raw is None
                else "Download" if managed_download else "Instructions"
            )
            action.setEnabled(True)
            state.style().unpolish(state)
            state.style().polish(state)
        self.runtime_verify_all_button.setEnabled(
            self._view_model is not None
            and not self._runtime_checking
            and not self._runtime_downloading_asset_id
        )

    def _setting_control(
        self,
        definition: SettingDefinition,
        value: Any,
    ) -> QtWidgets.QWidget:
        if definition.value_type is SettingValueType.BOOLEAN:
            control = QtWidgets.QCheckBox()
            control.setChecked(bool(value))
            control.toggled.connect(
                lambda checked, item=definition: self._set_module_value(item, checked)
            )
            return control
        if definition.value_type is SettingValueType.ENUM:
            control = HybridComboBox()
            for option in self._registry.supported_values(definition.qualified_id):
                control.addItem(str(option), option)
            self._select_data(control, value)
            control.currentIndexChanged.connect(
                lambda _index, item=definition, widget=control: self._set_module_value(
                    item, widget.currentData()
                )
            )
            return control
        if definition.value_type is SettingValueType.INTEGER:
            control = WheelSafeSpinBox()
            control.setButtonSymbols(
                QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
            )
            control.setRange(
                int(definition.minimum if definition.minimum is not None else -2147483647),
                int(definition.maximum if definition.maximum is not None else 2147483647),
            )
            control.setValue(int(value))
            control.valueChanged.connect(
                lambda number, item=definition: self._set_module_value(item, number)
            )
            return control
        if definition.value_type is SettingValueType.NUMBER:
            control = WheelSafeDoubleSpinBox()
            control.setButtonSymbols(
                QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
            )
            control.setDecimals(4)
            control.setRange(
                float(definition.minimum if definition.minimum is not None else -1e9),
                float(definition.maximum if definition.maximum is not None else 1e9),
            )
            control.setValue(float(value))
            control.valueChanged.connect(
                lambda number, item=definition: self._set_module_value(item, number)
            )
            return control
        control = QtWidgets.QLineEdit(str(value))
        control.editingFinished.connect(
            lambda item=definition, widget=control: self._set_module_value(item, widget.text())
        )
        return control

    def _set_translation_module_profile(self, profile_id: object) -> None:
        if self._view_model is None or self._profile_refreshing:
            return
        normalized = str(profile_id or "").strip()
        self._view_model.select_translation_provider(normalized or None)
        self._refresh_profiles()
        self._refresh_module_stage_presentations()
        self._refresh_pending()

    def _set_module_value(self, definition: SettingDefinition, value: Any) -> None:
        if self._view_model is None or self._profile_refreshing:
            return
        configs = {config.module_id: config for config in self._view_model.draft.module_configs}
        module = self._registry.get_module(definition.module_id)
        current = configs.get(
            definition.module_id,
            ModuleConfig(
                module_id=definition.module_id,
                module_schema_version=module.schema_version,
            ),
        )
        values = dict(current.values)
        values[definition.setting_id] = value
        try:
            self._view_model.replace_module(replace(current, values=values))
        except (TypeError, ValueError):
            self.refresh_from_model()
            return
        self._refresh_pending()

    def _filter_visible_controls(self, text: str) -> None:
        query = text.strip().casefold()
        keywords = {
            "general": "general language autosave project output",
            "appearance": "appearance theme font scale density motion contrast",
            "providers": "providers api credential endpoint model retry timeout gguf ollama deepseek",
            "modules": "modules detection ocr translation cleanup source style output defaults",
            "runtime": "runtime assets cuda pyicu models download",
            "glossary": "glossary terms aliases language",
            "shortcuts": "shortcuts keyboard keys bindings",
        }
        for index in range(self.category_list.count()):
            item = self.category_list.item(index)
            category_id = str(
                item.data(QtCore.Qt.ItemDataRole.UserRole) or ""
            )
            item.setHidden(bool(query and query not in keywords.get(category_id, "")))
        for index in range(self.modules_layout.count()):
            widget = self.modules_layout.itemAt(index).widget()
            if widget is None:
                continue
            haystack = str(widget.property("searchText") or "")
            widget.setVisible(not query or query in haystack)

    def _refresh_pending(self) -> None:
        dirty = bool(self._view_model and self._view_model.dirty)
        current_category = (
            self.CATEGORIES[self.stack.currentIndex()][0]
            if 0 <= self.stack.currentIndex() < len(self.CATEGORIES)
            else ""
        )
        self.global_footer.setVisible(bool(dirty and current_category != "providers"))
        self.pending_label.setText("Pending changes" if dirty else "No pending changes")
        self.pending_label.setProperty("tone", "warning" if dirty else "muted")
        self.apply_button.setEnabled(dirty)
        self.cancel_button.setEnabled(dirty)
        self.provider_apply_button.setEnabled(dirty)
        self.provider_cancel_button.setEnabled(dirty)
        self.general_apply_button.setEnabled(dirty)
        self.general_apply_button.setText("Apply" if dirty else "Saved")
        self.application_defaults_status.setText("Draft" if dirty else "Valid")
        self.application_defaults_status.setProperty(
            "tone", "editing" if dirty else "ready"
        )
        self.project_defaults_status.setText("Draft" if dirty else "Project")
        self.project_defaults_status.setProperty(
            "tone", "editing" if dirty else "ready"
        )
        for label in (
            self.application_defaults_status,
            self.project_defaults_status,
        ):
            label.style().unpolish(label)
            label.style().polish(label)

    def _set_bound(self, bound: bool) -> None:
        self.apply_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.provider_apply_button.setEnabled(False)
        self.provider_cancel_button.setEnabled(False)
        self.general_apply_button.setEnabled(False)
        self.general_apply_button.setText("Saved")
        self.global_footer.setVisible(False)
        self.pending_label.setText(
            "No settings authority available" if not bound else "No pending changes"
        )

    @staticmethod
    def _select_data(combo: QtWidgets.QComboBox, value: Any) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
