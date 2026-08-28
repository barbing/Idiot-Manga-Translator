# -*- coding: utf-8 -*-
"""Main window UI."""
import importlib.util
import os
from dataclasses import replace
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets
from app.config.defaults import get_defaults
from app.config.credential_store import (
    CompositeCredentialResolver,
    EnvironmentCredentialResolver,
    WindowsCredentialStore,
)
from app.config.module_registry import DEFAULT_MODULE_REGISTRY
from app.config.provider_profiles import (
    GGUFProviderOptions,
    GenerationSettings,
    ModelGenerationOverride,
    OllamaProviderOptions,
    ProviderKind,
    ProviderProfileStore,
)
from app.config.run_settings_compiler import (
    CompilationResult,
    RunInvocation,
    materialize_pipeline_settings,
)
from app.config.settings_contracts import (
    ApplicationPreferences,
    ModuleConfig,
    ProjectConfig,
    RunSettingsSnapshot,
    SettingsScope,
    canonical_fingerprint,
)
from app.config.settings_migration import (
    LegacyRunInvocationDefaults,
    LegacySettingsMigrationResult,
    legacy_project_settings_seed_required,
    migrate_legacy_qsettings_once,
    publish_legacy_migration_marker_last,
)
from app.config.settings_store import (
    ApplicationSettingsDocument,
    ApplicationSettingsStore,
    InactiveLegacyMigrationEvidence,
    LegacyMigrationIssueEvidence,
)
from app.pipeline.controller import (
    PipelineController,
    PipelineRuntimeBinding,
    PipelineSettings,
    _normalize_ocr_engine_name,
)
from app.models.ollama import list_models
from app.ui.theme import apply_dark_palette, apply_light_palette
from app.io.project import (
    load_project,
    load_project_for_editing,
    migrate_project_schema_v2,
    project_storage_is_checkpoint_descriptor,
    read_project_settings,
    save_project_schema_v2_atomic,
    with_project_settings,
)
from app.ui.style_guide_editor import StyleGuideEditor
from app.ui.region_review import RegionReviewDialog
from app.models.downloader import ModelDownloader
from app.platform_services.paths import qt_platform_paths
from app.ui.dialogs.download_dialog import DownloadDialog
from app.ui.viewmodels.settings_model import (
    LegacyShellSettingsProjection,
    SettingsDraft,
    SettingsViewModel,
    rebind_run_snapshot_project,
)

import glob
import logging
import re

logger = logging.getLogger(__name__)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YomiFrame v1.2.0")
        self.resize(1200, 720)
        logger.info("MainWindow initialized")
        
        self._defaults = get_defaults()
        self._model_overrides: dict[str, dict] = {}
        self._pipeline = PipelineController(self)
        self._apply_theme(self._defaults.theme)
        self._style_editor: StyleGuideEditor | None = None
        self._review_dialog: RegionReviewDialog | None = None
        self._page_review: QtWidgets.QDialog | None = None
        self._running = False
        self._pyicu_runtime_ready = False
        self._page_cache: dict[int, dict] = {}
        self._thumb_cache: dict[str, QtGui.QPixmap] = {}
        self._processing_phase = 0
        self._processing_timer = QtCore.QTimer(self)
        self._processing_timer.timeout.connect(self._pulse_processing)
        self._processing_timer.start(500)
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._refresh_import_preview)
        self._last_preview_dir = ""
        self._download_thread: QtCore.QThread | None = None
        self._download_dialog: DownloadDialog | None = None
        self._download_worker: ModelDownloader | None = None
        self._settings_model: SettingsViewModel | None = None
        self._application_settings_store: ApplicationSettingsStore | None = None
        self._provider_profile_store: ProviderProfileStore | None = None
        self._application_settings_document = ApplicationSettingsDocument()
        self._last_run_snapshot: RunSettingsSnapshot | None = None
        self._active_project_id: str | None = None
        self._active_run_json_path: str | None = None
        self._pending_run_snapshot: RunSettingsSnapshot | None = None
        self._pending_project_id: str | None = None
        self._pending_run_json_path: str | None = None
        self._settings_projection_guard = False
        self._setup_ui()
        self._connect_signals()
        self.start_btn.setEnabled(False)
        self._pipeline.status.consistency_issue.connect(self._on_consistency_issue)
        
        
        # Check for model updates once UI is ready
        self._refresh_gguf_list()
        QtCore.QTimer.singleShot(500, self._check_required_models)

    def _refresh_gguf_list(self):
        """Scan models directory for GGUF files."""
        models_dir = os.path.join(os.getcwd(), "models")
        # Find all .gguf files recursively
        files = glob.glob(os.path.join(models_dir, "**", "*.gguf"), recursive=True)
        
        # Normalize paths
        paths = [os.path.abspath(p) for p in files]
        
        # Update ComboBoxes
        for combo in (self.gguf_model_path, self.settings_gguf_model_path):
            current_text = combo.currentText()
            combo.clear()
            
            # Add found items with Basename as text, Full Path as data
            for path in paths:
                combo.addItem(os.path.basename(path), path)
            
            # Policy: Only keep current selection if it still exists
            if current_text:
                # current_text might be a basename or a full path or a relative path
                # We need to find if this text matches any of our new items (by data or text)
                found = False
                for idx in range(combo.count()):
                    data = combo.itemData(idx)
                    text = combo.itemText(idx)
                    # Check against data (full path) or text (basename) or raw current_text (path)
                    if (data and os.path.normpath(data) == os.path.normpath(current_text)) or \
                       (text == current_text):
                        combo.setCurrentIndex(idx)
                        found = True
                        break
                
                # If not found but physically exists (e.g. user typed a path manually not in Models dir)
                # re-add it? User said "doesn't require memory", so maybe strictly scan.
                # But if user put a file outside models dir, we should probably respect it if they just typed it?
                # Actually user said "this list doesn't require our memory function".
                # So let's stick to scanned items. If not in scanned, ignore.
                pass
            
            if combo.currentIndex() == -1 and combo.count() > 0:
                 combo.setCurrentIndex(0)

    def _check_required_models(self):
        """Check for critical models and download if missing."""
        downloader = ModelDownloader(self)
        models_dir = os.path.join(os.getcwd(), "models")

        missing = []
        runtime_ready = downloader.check_pyicu_runtime()
        self._set_pyicu_runtime_ready(runtime_ready)
        if not runtime_ready:
            if downloader.can_install_pyicu_runtime():
                missing.append("pyicu_runtime")
            else:
                runtime_error = downloader.pyicu_runtime_error or (
                    "The required PyICU runtime is unavailable."
                )
                self._set_pyicu_runtime_ready(False, runtime_error)
                QtWidgets.QMessageBox.critical(
                    self,
                    "Required Runtime Unavailable",
                    runtime_error,
                )
        if not downloader.check_comic_text_detector(models_dir):
            missing.append("comic_text_detector")
        if not downloader.check_bubble_detection(models_dir):
            missing.append("bubble_detection")
        if not downloader.check_paddle_ocr_vl(models_dir):
            missing.append("paddle_ocr_vl")
        if not downloader.check_manga_ocr(models_dir):
            missing.append("manga_ocr")
        if self._ai_inpaint_runtime_available() and not downloader.check_big_lama(models_dir):
            missing.append("cleanup_inpaint")
        if not downloader.check_ner(models_dir):
            missing.append("ner")
        if not downloader.check_font_detection(models_dir):
            missing.append("font_detection")
            
        if missing:
            reply = QtWidgets.QMessageBox.warning(
                self,
                "Missing Runtime or Models",
                f"The following required assets are missing: {', '.join(missing)}.\n"
                "Download them now?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self._run_model_download(missing) # Pass list
            else:
                if "pyicu_runtime" in missing:
                    self._set_pyicu_runtime_ready(
                        False,
                        "Required PyICU runtime was not installed; translation rendering is disabled.",
                    )
                else:
                    self.status_bar.showMessage("Warning: Models missing.", 5000)

    def _set_pyicu_runtime_ready(self, ready: bool, message: str = "") -> None:
        self._pyicu_runtime_ready = bool(ready)
        if not self._running:
            self.start_btn.setEnabled(self._pyicu_runtime_ready)
        if message:
            self.status_bar.showMessage(message, 10000)
            if self.log_view:
                self.log_view.appendPlainText(message)

    def _ai_inpaint_runtime_available(self) -> bool:
        try:
            return all(
                importlib.util.find_spec(module_name) is not None
                for module_name in ("torch", "numpy", "PIL", "cv2")
            )
        except Exception:
            return False

    def _run_model_download(self, model_keys):
        """Run a managed download dialog."""
        # Normalize input to list
        if isinstance(model_keys, str):
            model_keys = [model_keys]
        if self._download_thread and self._download_thread.isRunning():
            self.status_bar.showMessage("A download is already running.", 3000)
            return
             
        downloader = ModelDownloader()
        dialog = DownloadDialog(self, title="Downloading Model Assets")
        dialog.set_downloader(downloader)
        self._download_worker = downloader
        self._download_dialog = dialog
        
        self._download_thread = QtCore.QThread(self)
        
        # Queue tasks BEFORE moving to thread
        models_dir = os.path.join(os.getcwd(), "models")

        if "pyicu_runtime" in model_keys:
             downloader.prepare_pyicu_runtime()
        if "comic_text_detector" in model_keys:
             downloader.prepare_comic_text_detector(models_dir)
        if "bubble_detection" in model_keys:
             downloader.prepare_bubble_detection(models_dir)
        if "manga_ocr" in model_keys:
             downloader.prepare_manga_ocr(models_dir)
        if "paddle_ocr_vl" in model_keys:
             downloader.prepare_paddle_ocr_vl(models_dir)
        if "cleanup_inpaint" in model_keys or "big_lama" in model_keys:
             downloader.prepare_big_lama(models_dir)
        if "sakura" in model_keys:
             downloader.prepare_sakura(models_dir)
        if "qwen" in model_keys:
             downloader.prepare_qwen(models_dir)
        if "ner" in model_keys:
             downloader.prepare_ner(models_dir)
        if "font_detection" in model_keys:
             downloader.prepare_font_detection(models_dir)

        downloader.moveToThread(self._download_thread)
        
        # Connect signal to SLOT (running in new thread context)
        self._download_thread.started.connect(downloader.process_queue)

        # Cleanup
        downloader.finished.connect(self._download_thread.quit)
        downloader.finished.connect(downloader.deleteLater)
        model_key_for_callback = model_keys[0] if len(model_keys) == 1 else "batch"
        # Handle completion for single items or batch
        self._download_thread.finished.connect(lambda: self._on_download_complete(model_key_for_callback))
        self._download_thread.finished.connect(self._on_download_thread_finished)
        
        self._download_thread.start()
        dialog.exec()
        # Ensure worker thread has fully stopped before local dialog refs are released.
        if self._download_thread and self._download_thread.isRunning():
            self._download_thread.quit()
            self._download_thread.wait(5000)

    def _on_download_thread_finished(self) -> None:
        thread = self._download_thread
        if thread:
            thread.deleteLater()
        self._download_thread = None
        self._download_worker = None
        self._download_dialog = None

    def _on_download_complete(self, model_key: str):
        """Post-download actions."""
        self._refresh_gguf_models() # Refreshes all lists
        runtime_was_required = not self._pyicu_runtime_ready
        runtime_checker = ModelDownloader(self)
        runtime_ready = runtime_checker.check_pyicu_runtime()
        runtime_message = ""
        if not runtime_ready and runtime_was_required:
            runtime_message = runtime_checker.pyicu_runtime_error or (
                "Required PyICU runtime setup did not complete; rendering remains disabled."
            )
        self._set_pyicu_runtime_ready(runtime_ready, runtime_message)
        
        if model_key == "sakura":
            # Auto-set the path
            sakura_path = os.path.join(os.getcwd(), "models", "sakura", "sakura-14b-qwen3-v1.5-q6k.gguf")
            if os.path.exists(sakura_path):
                self._add_gguf_model(sakura_path)
                self.settings_gguf_model_path.setCurrentText(sakura_path)
                self.gguf_model_path.setCurrentText(sakura_path)
                self.status_bar.showMessage("Sakura model selected.", 3000)
                
        elif model_key == "qwen":
             qwen_path = os.path.join(os.getcwd(), "models", "qwen", "Qwen3-14B-Q6_K.gguf")
             if os.path.exists(qwen_path):
                 self._add_gguf_model(qwen_path)
                 self.settings_discovery_gguf_path.setCurrentText(qwen_path)
                 self._set_gguf_combo(self.settings_discovery_gguf_path, qwen_path)
                 self.status_bar.showMessage("Qwen model selected for Deep Scan.", 3000)

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Hidden advanced groups (created for settings/pipeline defaults).
        self._render_group = self._group_render()
        self._perf_group = self._group_performance()
        self._theme_group = self._group_theme()
        self._models_group = self._group_models_main()
        self._paths_group = self._group_paths_settings()

        nav_panel = self._build_nav_panel()
        center_panel = self._build_center_panel()
        right_panel = self._build_right_panel()

        layout.addWidget(nav_panel)
        layout.addWidget(center_panel, 1)
        layout.addWidget(right_panel)

        self.status_bar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self._active_page_index = 0

    def _build_nav_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        panel.setFixedWidth(220)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        icons = self._load_icons()

        brand = QtWidgets.QHBoxLayout()
        brand_icon = QtWidgets.QLabel("")
        brand_icon.setAlignment(QtCore.Qt.AlignCenter)
        brand_icon.setFixedSize(34, 34)
        brand_icon.setStyleSheet("QLabel { background-color: #1b2230; border-radius: 10px; font-size: 18px; }")
        brand_title = QtWidgets.QLabel("YomiFrame")
        brand_title.setStyleSheet("QLabel { font-size: 18px; font-weight: 600; }")
        if icons.get("brand"):
            brand_icon.setPixmap(icons["brand"].pixmap(18, 18))
        brand.addWidget(brand_icon)
        brand.addWidget(brand_title)
        brand.addStretch(1)
        layout.addLayout(brand)

        self.nav_home = QtWidgets.QPushButton()
        self.nav_queue = QtWidgets.QPushButton()
        self.nav_library = QtWidgets.QPushButton()
        self.nav_settings = QtWidgets.QPushButton()
        for key, btn in (
            ("home", self.nav_home),
            ("queue", self.nav_queue),
            ("library", self.nav_library),
            ("settings", self.nav_settings),
        ):
            btn.setText(key.capitalize() if key != "home" else "Home")
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setProperty("nav", True)
            # btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            # btn.setAutoRaise(True)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setIconSize(QtCore.QSize(18, 18))
            btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            btn.setFixedHeight(40)
            
            # Use the platform UI family while preserving the legacy control size.
            f = QtGui.QFontDatabase.systemFont(
                QtGui.QFontDatabase.SystemFont.GeneralFont
            )
            f.setPointSize(14)
            f.setStyleStrategy(QtGui.QFont.PreferAntialias)
            btn.setFont(f)
            
            if icons.get(key):
                btn.setIcon(icons[key])
            layout.addWidget(btn)
        self.nav_home.setChecked(True)

        layout.addStretch(1)

        lang_block = QtWidgets.QVBoxLayout()
        source_label = QtWidgets.QLabel("Source")
        self.source_lang = QtWidgets.QComboBox()
        self.source_lang.addItems(["Japanese"])
        target_label = QtWidgets.QLabel("Target")
        self.target_lang = QtWidgets.QComboBox()
        self.target_lang.addItems(["Simplified Chinese", "English"])
        lang_block.addWidget(source_label)
        lang_block.addWidget(self.source_lang)
        lang_block.addSpacing(6)
        lang_block.addWidget(target_label)
        lang_block.addWidget(self.target_lang)
        layout.addLayout(lang_block)

        self.start_btn = QtWidgets.QPushButton("Start Translation")
        self.start_btn.setFixedHeight(38)
        self.start_btn.setObjectName("startBtn")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setFixedHeight(34)
        self.stop_btn.setObjectName("stopBtn")
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        self.stop_btn.setEnabled(False)

        return panel

    def _build_center_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)

        self.center_stack = QtWidgets.QStackedWidget()
        self.home_page = self._build_home_center()
        self.queue_page = self._build_queue_center()
        self.library_page = self._build_library_center()
        self.settings_page = self._build_settings_center()
        self.center_stack.addWidget(self.home_page)
        self.center_stack.addWidget(self.queue_page)
        self.center_stack.addWidget(self.library_page)
        self.center_stack.addWidget(self.settings_page)
        vbox.addWidget(self.center_stack, 1)

        return panel

    def _build_right_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        panel.setFixedWidth(320)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)

        self.right_stack = QtWidgets.QStackedWidget()
        self.home_right = self._build_home_right()
        self.queue_right = self._build_queue_right()
        self.library_right = self._build_library_right()
        self.settings_right = self._build_settings_right()
        self.right_stack.addWidget(self.home_right)
        self.right_stack.addWidget(self.queue_right)
        self.right_stack.addWidget(self.library_right)
        self.right_stack.addWidget(self.settings_right)
        vbox.addWidget(self.right_stack, 1)

        return panel

    def _build_home_center(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)

        summary_box = QtWidgets.QGroupBox("Total Progress")
        summary_layout = QtWidgets.QVBoxLayout(summary_box)
        
        # Two-stage progress for Pre-Scan mode
        self.prescan_label = QtWidgets.QLabel("Pre-Scan: Scanning for names...")
        self.prescan_bar = QtWidgets.QProgressBar()
        self.prescan_bar.setValue(0)
        self.prescan_bar.setVisible(False)  # Hidden until prescan starts
        self.prescan_label.setVisible(False)
        summary_layout.addWidget(self.prescan_label)
        summary_layout.addWidget(self.prescan_bar)
        
        self.progress_title = QtWidgets.QLabel("Total Progress: 0%")
        self.progress_title.setWordWrap(True)
        self.overall_bar = QtWidgets.QProgressBar()
        self.overall_bar.setValue(0)
        self.processing_label = QtWidgets.QLabel("Processing Page 0 of 0...")
        self.processing_label.setWordWrap(True)
        status_row = QtWidgets.QHBoxLayout()
        self.eta_label = QtWidgets.QLabel("ETA: --")
        status_row.addStretch(1)
        status_row.addWidget(self.eta_label)
        summary_layout.addWidget(self.progress_title)
        summary_layout.addWidget(self.overall_bar)
        summary_layout.addWidget(self.processing_label)
        summary_layout.addLayout(status_row)
        self.total_time_label = QtWidgets.QLabel("Total: --")
        self.page_time_label = QtWidgets.QLabel("Page: --")
        summary_layout.addWidget(self.total_time_label)
        summary_layout.addWidget(self.page_time_label)
        vbox.addWidget(summary_box)

        queue_box = QtWidgets.QGroupBox("")
        queue_layout = QtWidgets.QVBoxLayout(queue_box)
        self.queue_list = QtWidgets.QListWidget()
        self.queue_list.setViewMode(QtWidgets.QListView.IconMode)
        self.queue_list.setResizeMode(QtWidgets.QListView.Adjust)
        self.queue_list.setMovement(QtWidgets.QListView.Static)
        self.queue_list.setUniformItemSizes(True)
        self.queue_list.setIconSize(QtCore.QSize(120, 170))
        self.queue_list.setSpacing(8)
        self.queue_list.setGridSize(QtCore.QSize(140, 210))
        self.queue_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.queue_list.setWordWrap(True)
        self.queue_placeholder = self._build_empty_state(
            "No Images Loaded",
            "Drag & Drop Folder Here",
            "fa5s.folder-open",
        )
        self.queue_stack = QtWidgets.QStackedLayout()
        self.queue_stack.addWidget(self.queue_placeholder)
        self.queue_stack.addWidget(self.queue_list)
        queue_layout.addLayout(self.queue_stack)
        vbox.addWidget(queue_box, 1)

        bottom_tabs = QtWidgets.QTabWidget()
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.review_stub = QtWidgets.QWidget()
        review_layout = QtWidgets.QVBoxLayout(self.review_stub)
        review_layout.addWidget(QtWidgets.QLabel("Double-click any page to review in split view."))
        self.review_btn = QtWidgets.QPushButton("Review Regions")
        review_layout.addWidget(self.review_btn)
        review_layout.addStretch(1)
        bottom_tabs.addTab(self.log_view, "Live Log")
        bottom_tabs.addTab(self.review_stub, "Region Review")
        vbox.addWidget(bottom_tabs)

        return panel

    def _build_queue_center(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)
        header = QtWidgets.QLabel("Queue")
        header.setStyleSheet("QLabel { font-size: 16px; font-weight: 600; }")
        vbox.addWidget(header)
        self.queue_table = QtWidgets.QTableWidget(0, 3)
        self.queue_table.setHorizontalHeaderLabels(["Page", "Status", "Action"])
        header = self.queue_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        header.setMinimumSectionSize(120)
        self.queue_table.setColumnWidth(2, 180)
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.verticalHeader().setDefaultSectionSize(52)
        self.queue_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.queue_table_placeholder = self._build_empty_state(
            "Queue is empty",
            "Start a translation to see items here.",
            "fa5s.inbox",
        )
        self.queue_table_stack = QtWidgets.QStackedLayout()
        self.queue_table_stack.addWidget(self.queue_table_placeholder)
        self.queue_table_stack.addWidget(self.queue_table)
        queue_container = QtWidgets.QWidget(self)
        queue_container.setLayout(self.queue_table_stack)
        vbox.addWidget(queue_container, 1)
        return panel

    def _build_library_center(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)
        header = QtWidgets.QLabel("Library")
        header.setStyleSheet("QLabel { font-size: 16px; font-weight: 600; }")
        vbox.addWidget(header)
        self.library_list = QtWidgets.QListWidget()
        self.library_list.setViewMode(QtWidgets.QListView.IconMode)
        self.library_list.setResizeMode(QtWidgets.QListView.Adjust)
        self.library_list.setMovement(QtWidgets.QListView.Static)
        self.library_list.setUniformItemSizes(True)
        self.library_list.setIconSize(QtCore.QSize(140, 200))
        self.library_list.setSpacing(12)
        self.library_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.library_list.setWordWrap(True)
        self.library_placeholder = self._build_empty_state(
            "No translations yet",
            "Completed pages will appear here.",
            "fa5s.images",
        )
        self.library_stack = QtWidgets.QStackedLayout()
        self.library_stack.addWidget(self.library_placeholder)
        self.library_stack.addWidget(self.library_list)
        library_container = QtWidgets.QWidget(self)
        library_container.setLayout(self.library_stack)
        vbox.addWidget(library_container, 1)
        return panel

    def _build_settings_center(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)
        header = QtWidgets.QLabel("Settings")
        header.setStyleSheet("QLabel { font-size: 16px; font-weight: 600; }")
        vbox.addWidget(header)
        self.settings_tabs = QtWidgets.QTabWidget()
        self.settings_general = QtWidgets.QWidget()
        general_layout = QtWidgets.QVBoxLayout(self.settings_general)
        general_layout.addWidget(self._perf_group)
        general_layout.addWidget(self._theme_group)
        general_layout.addStretch(1)

        self.settings_models = self._build_models_settings()
        self.settings_render = QtWidgets.QWidget()
        render_layout = QtWidgets.QVBoxLayout(self.settings_render)
        render_layout.addWidget(self._render_group)
        render_layout.addStretch(1)

        self.settings_tabs.addTab(self.settings_general, "General")
        self.settings_tabs.addTab(self.settings_models, "Models")
        self.settings_tabs.addTab(self.settings_render, "Rendering")
        vbox.addWidget(self.settings_tabs, 1)
        return panel

    def _build_home_right(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)
        self.home_right_stack = QtWidgets.QStackedLayout()

        default_panel = QtWidgets.QWidget(self)
        default_layout = QtWidgets.QVBoxLayout(default_panel)
        default_layout.setContentsMargins(0, 0, 0, 0)
        default_layout.setSpacing(10)
        default_layout.addWidget(self._group_project_files_main())
        default_layout.addWidget(self._group_style_guide())
        default_layout.addStretch(1)
        self.home_right_stack.addWidget(default_panel)

        self.home_inspector_panel = QtWidgets.QWidget(self)
        inspector_layout = QtWidgets.QVBoxLayout(self.home_inspector_panel)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(10)
        self.inspector_group = QtWidgets.QGroupBox("Live Inspection")
        inspector_form = QtWidgets.QVBoxLayout(self.inspector_group)
        self.inspector_title = QtWidgets.QLabel("No page selected")
        self.inspector_title.setStyleSheet("QLabel { font-weight: 600; }")
        self.inspector_table = QtWidgets.QTableWidget(0, 2)
        self.inspector_table.setHorizontalHeaderLabels(["Detected", "Translation"])
        self.inspector_table.horizontalHeader().setStretchLastSection(True)
        self.inspector_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.inspector_table.verticalHeader().setVisible(False)
        self.inspector_table.setWordWrap(True)
        self.inspector_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        inspector_form.addWidget(self.inspector_title)
        inspector_form.addWidget(self.inspector_table, 1)
        self.inspector_back = QtWidgets.QPushButton("Back to Project Files")
        inspector_form.addWidget(self.inspector_back)
        inspector_layout.addWidget(self.inspector_group)
        inspector_layout.addStretch(1)
        self.home_right_stack.addWidget(self.home_inspector_panel)

        vbox.addLayout(self.home_right_stack, 1)
        return panel

    def _build_queue_right(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)
        group = QtWidgets.QGroupBox("Job Details")
        layout = QtWidgets.QFormLayout(group)
        self.job_file = QtWidgets.QLabel("--")
        self.job_status = QtWidgets.QLabel("--")
        self.job_page = QtWidgets.QLabel("--")
        self.job_stage = QtWidgets.QLabel("--")
        layout.addRow("File", self.job_file)
        layout.addRow("Status", self.job_status)
        layout.addRow("Page", self.job_page)
        layout.addRow("Stage", self.job_stage)
        self.job_open = QtWidgets.QPushButton("Open Folder")
        layout.addRow("", self.job_open)
        vbox.addWidget(group)
        vbox.addStretch(1)
        return panel

    def _build_library_right(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)
        group = QtWidgets.QGroupBox("File Info")
        layout = QtWidgets.QFormLayout(group)
        self.library_file = QtWidgets.QLabel("--")
        self.library_pages = QtWidgets.QLabel("--")
        self.library_size = QtWidgets.QLabel("--")
        layout.addRow("File", self.library_file)
        layout.addRow("Pages", self.library_pages)
        layout.addRow("Size", self.library_size)
        self.library_open = QtWidgets.QPushButton("Open Folder")
        layout.addRow("", self.library_open)
        vbox.addWidget(group)
        vbox.addStretch(1)
        return panel

    def _build_settings_right(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)

        summary_group = QtWidgets.QGroupBox("Effective Run")
        summary_layout = QtWidgets.QFormLayout(summary_group)
        self.settings_effective_status = QtWidgets.QLabel("Not evaluated")
        self.settings_effective_status.setWordWrap(True)
        self.settings_effective_language = QtWidgets.QLabel("--")
        self.settings_effective_language.setWordWrap(True)
        self.settings_effective_provider = QtWidgets.QLabel("--")
        self.settings_effective_provider.setWordWrap(True)
        self.settings_effective_model = QtWidgets.QLabel("--")
        self.settings_effective_model.setWordWrap(True)
        self.settings_effective_detection = QtWidgets.QLabel("--")
        self.settings_effective_detection.setWordWrap(True)
        self.settings_effective_cleanup = QtWidgets.QLabel("--")
        self.settings_effective_cleanup.setWordWrap(True)
        self.settings_effective_runtime = QtWidgets.QLabel("--")
        self.settings_effective_runtime.setWordWrap(True)
        self.settings_effective_snapshot = QtWidgets.QLabel("--")
        self.settings_effective_snapshot.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self.settings_effective_validation = QtWidgets.QLabel(
            "Validation has not run."
        )
        self.settings_effective_validation.setWordWrap(True)
        summary_layout.addRow("Status", self.settings_effective_status)
        summary_layout.addRow("Language", self.settings_effective_language)
        summary_layout.addRow("Provider", self.settings_effective_provider)
        summary_layout.addRow("Model", self.settings_effective_model)
        summary_layout.addRow("Detection / OCR", self.settings_effective_detection)
        summary_layout.addRow("Cleanup / Style", self.settings_effective_cleanup)
        summary_layout.addRow("Runtime", self.settings_effective_runtime)
        summary_layout.addRow("Snapshot", self.settings_effective_snapshot)
        summary_layout.addRow("Validation", self.settings_effective_validation)
        vbox.addWidget(summary_group)

        group = QtWidgets.QGroupBox("Save Options")
        layout = QtWidgets.QFormLayout(group)
        self.settings_save_folder = QtWidgets.QLineEdit()
        self.settings_save_folder.setReadOnly(True)
        layout.addRow("Save Folder", self.settings_save_folder)
        self.settings_auto_import = QtWidgets.QCheckBox("Auto import assets")
        self.settings_auto_import.setChecked(True)
        layout.addRow("", self.settings_auto_import)
        self.settings_open_folder = QtWidgets.QPushButton("Open Folder")
        layout.addRow("", self.settings_open_folder)
        vbox.addWidget(group)
        vbox.addStretch(1)
        return panel

    def _build_models_settings(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)
        self.settings_models_group = self._group_models_settings()
        scroll = QtWidgets.QScrollArea(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_body = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_body)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)
        scroll_layout.addWidget(self.settings_models_group)
        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_body)
        vbox.addWidget(scroll)
        return panel

    def _group_project_files_main(self) -> QtWidgets.QGroupBox:
        if hasattr(self, "_project_files_group"):
            return self._project_files_group
        group = QtWidgets.QGroupBox("Project Files")
        layout = QtWidgets.QFormLayout(group)
        self.import_dir = QtWidgets.QLineEdit(self._defaults.import_dir)
        self.import_browse = QtWidgets.QPushButton("Browse")
        self.export_dir = QtWidgets.QLineEdit(self._defaults.export_dir)
        self.export_browse = QtWidgets.QPushButton("Browse")
        layout.addRow("Import Folder", self._hbox(self.import_dir, self.import_browse))
        layout.addRow("Export Folder", self._hbox(self.export_dir, self.export_browse))
        self.output_suffix = QtWidgets.QLineEdit(self._defaults.output_suffix)
        self.json_path = QtWidgets.QLineEdit(self._defaults.json_path)
        self.json_browse = QtWidgets.QPushButton("Browse")
        layout.addRow("Project JSON", self._hbox(self.json_path, self.json_browse))
        layout.addRow("Filename Suffix", self.output_suffix)
        self._project_files_group = group
        return group

    def _group_paths_settings(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Paths")
        layout = QtWidgets.QFormLayout(group)
        self.settings_import_dir = QtWidgets.QLineEdit(self._defaults.import_dir)
        self.settings_export_dir = QtWidgets.QLineEdit(self._defaults.export_dir)
        self.settings_json_path = QtWidgets.QLineEdit(self._defaults.json_path)
        self.settings_output_suffix = QtWidgets.QLineEdit(self._defaults.output_suffix)
        for field in (
            self.settings_import_dir,
            self.settings_export_dir,
            self.settings_json_path,
            self.settings_output_suffix,
        ):
            field.setReadOnly(True)
        layout.addRow("Import Folder", self.settings_import_dir)
        layout.addRow("Export Folder", self.settings_export_dir)
        layout.addRow("Project JSON", self.settings_json_path)
        layout.addRow("Filename Suffix", self.settings_output_suffix)
        return group

    def _group_language(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Language")
        layout = QtWidgets.QFormLayout(group)
        self.source_lang = QtWidgets.QComboBox()
        self.source_lang.addItems(["Japanese"])
        self.target_lang = QtWidgets.QComboBox()
        self.target_lang.addItems(["Simplified Chinese", "English"])
        layout.addRow("Source", self.source_lang)
        layout.addRow("Target", self.target_lang)
        return group

    def _group_style_guide(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Style & Glossary")
        layout = QtWidgets.QFormLayout(group)
        self.style_path = QtWidgets.QLineEdit("")
        self.style_browse = QtWidgets.QPushButton("Browse")
        self.style_edit = QtWidgets.QPushButton("Open Editor")
        layout.addRow("Guide JSON", self._hbox(self.style_path, self.style_browse))
        self.auto_glossary = QtWidgets.QCheckBox("Auto-Glossary\n(Name Memory)")
        self.auto_glossary.setChecked(self._defaults.auto_glossary)
        self.auto_glossary.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.auto_glossary.setToolTip(
            "Build and apply a name/term memory during translation.\n"
            "This is the main switch for consistent proper nouns and alias handling."
        )
        self.use_ollama_discovery = QtWidgets.QCheckBox("Experimental Deep Discovery\n(Optional, slower)")
        self.use_ollama_discovery.setToolTip(
            "Experimental accuracy boost for difficult discovery cases.\n"
            "Uses an additional LLM path for background name/entity discovery.\n"
            "Leave this off for the normal fast local workflow."
        )
        self.use_ollama_discovery.setChecked(False) # Default off for safety/performance
        
        self.mismatch_warning_label = QtWidgets.QLabel("⚠️ Performance Warning: Using different models causes slow reloading.")
        self.mismatch_warning_label.setStyleSheet("color: #ef4444; font-size: 11px; font-weight: 600; margin-top: 4px;")
        self.mismatch_warning_label.setVisible(False)
        self.mismatch_warning_label.setWordWrap(True)

        self.discovery_model_combo = QtWidgets.QComboBox()
        self.discovery_model_combo.addItems(["auto-detect"])
        
        # Pre-Scan Mode: Build glossary before translation starts
        self.prescan_enabled = QtWidgets.QCheckBox("Build Glossary Before Translation")
        self.prescan_enabled.setToolTip(
            "Scan the chapter/volume before translation starts and build the glossary upfront.\n"
            "Recommended for volumes and best proper-noun consistency."
        )
        self.prescan_enabled.setChecked(False)
        
        layout.addRow("", self.style_edit)
        layout.addRow("", self.auto_glossary)
        layout.addRow("", self.use_ollama_discovery)
        layout.addRow("", self.prescan_enabled)
        return group

    def _group_models_main(self) -> QtWidgets.QGroupBox:
        if hasattr(self, "_models_group"):
            return self._models_group
        group = QtWidgets.QGroupBox("Advanced Model Settings")
        layout = QtWidgets.QGridLayout(group)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        self.detector_engine = QtWidgets.QComboBox()
        self.detector_engine.addItems(
            self._supported_setting_texts("detection.engine")
        )
        self.detector_engine.setCurrentText(self._defaults.detector_engine)
        
        self.detector_input_size = QtWidgets.QComboBox()
        self.detector_input_size.addItems(
            self._supported_setting_texts("detection.input_size")
        )
        self.detector_input_size.setCurrentText("640")
        
        det_layout = self._hbox(self.detector_engine, self.detector_input_size)
    
        layout.addWidget(QtWidgets.QLabel("Text Detector"), 0, 0)
        layout.addWidget(det_layout, 1, 0)
        self.ocr_engine = QtWidgets.QComboBox()
        self.ocr_engine.addItems(self._supported_setting_texts("ocr.engine"))
        self.ocr_engine.setCurrentText(self._defaults.ocr_engine)
        layout.addWidget(QtWidgets.QLabel("OCR Engine"), 0, 1)
        layout.addWidget(self.ocr_engine, 1, 1)
        self.translator_backend = QtWidgets.QComboBox()
        self.translator_backend.addItems(["Ollama", "GGUF", "DeepSeek"])
        self.translator_backend.setCurrentText(self._defaults.translator_backend)
        layout.addWidget(QtWidgets.QLabel("Translator"), 2, 0)
        layout.addWidget(self.translator_backend, 3, 0)
        self.ollama_model = QtWidgets.QComboBox()
        self.ollama_model.addItems(["auto-detect"])
        self.model_refresh = QtWidgets.QPushButton("Refresh")
        model_row = self._hbox(self.ollama_model, self.model_refresh)
        layout.addWidget(QtWidgets.QLabel("Ollama Model"), 2, 1)
        layout.addWidget(model_row, 3, 1)
        self.gguf_model_path = QtWidgets.QComboBox()
        self.gguf_model_path.setEditable(True)
        self.gguf_model_path.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.gguf_browse = QtWidgets.QPushButton("Browse")
        layout.addWidget(QtWidgets.QLabel("GGUF Model"), 4, 0, 1, 2)
        layout.addWidget(self._hbox(self.gguf_model_path, self.gguf_browse), 5, 0, 1, 2)
        self.gguf_prompt_style = QtWidgets.QComboBox()
        self.gguf_prompt_style.addItems(["sakura", "qwen", "plain"])
        self.gguf_prompt_style.setCurrentText(self._defaults.gguf_prompt_style)
        layout.addWidget(QtWidgets.QLabel("GGUF Prompt"), 6, 0, 1, 2)
        layout.addWidget(self.gguf_prompt_style, 7, 0, 1, 2)
        self.gguf_n_ctx = QtWidgets.QSpinBox()
        self.gguf_n_ctx.setRange(512, 32768)
        self.gguf_n_ctx.setValue(self._defaults.gguf_n_ctx)
        layout.addWidget(QtWidgets.QLabel("GGUF Context"), 8, 0)
        layout.addWidget(self.gguf_n_ctx, 9, 0)
        self.gguf_n_gpu_layers = QtWidgets.QSpinBox()
        self.gguf_n_gpu_layers.setRange(-1, 200)
        self.gguf_n_gpu_layers.setValue(self._defaults.gguf_n_gpu_layers)
        layout.addWidget(QtWidgets.QLabel("GGUF GPU Layers"), 8, 1)
        layout.addWidget(self.gguf_n_gpu_layers, 9, 1)
        self.gguf_n_threads = QtWidgets.QSpinBox()
        self.gguf_n_threads.setRange(1, 128)
        self.gguf_n_threads.setValue(self._defaults.gguf_n_threads)
        layout.addWidget(QtWidgets.QLabel("GGUF Threads"), 10, 0)
        layout.addWidget(self.gguf_n_threads, 11, 0)
        self.gguf_n_batch = QtWidgets.QSpinBox()
        self.gguf_n_batch.setRange(64, 4096)
        self.gguf_n_batch.setValue(self._defaults.gguf_n_batch)
        layout.addWidget(QtWidgets.QLabel("GGUF Batch"), 10, 1)
        layout.addWidget(self.gguf_n_batch, 11, 1)
        self._models_group = group
        return group

    def _group_models_settings(self) -> QtWidgets.QGroupBox:
        # Main Container Group
        main_group = QtWidgets.QGroupBox("Model Configuration")
        main_layout = QtWidgets.QVBoxLayout(main_group)
        main_layout.setSpacing(10)

        # 1. Core Components (Detector, OCR)
        # =================================================
        grp_core = QtWidgets.QGroupBox("Core Components")
        l_core = QtWidgets.QGridLayout(grp_core)
        
        self.settings_detector_engine = QtWidgets.QComboBox()
        self.settings_detector_engine.addItems(
            self._supported_setting_texts("detection.engine")
        )
        self.settings_detector_engine.setCurrentText(self._defaults.detector_engine)
        
        self.settings_detector_input_size = QtWidgets.QComboBox()
        self.settings_detector_input_size.addItems(
            self._supported_setting_texts("detection.input_size")
        )
        self.settings_detector_input_size.setCurrentText("640")
        
        self.settings_ocr_engine = QtWidgets.QComboBox()
        self.settings_ocr_engine.addItems(
            self._supported_setting_texts("ocr.engine")
        )
        self.settings_ocr_engine.setCurrentText(self._defaults.ocr_engine)
        
        l_core.addWidget(QtWidgets.QLabel("Text Detector"), 0, 0)
        l_core.addWidget(self._hbox(self.settings_detector_engine, self.settings_detector_input_size), 1, 0)
        l_core.addWidget(QtWidgets.QLabel("OCR Engine"), 0, 1)
        l_core.addWidget(self.settings_ocr_engine, 1, 1)
        
        main_layout.addWidget(grp_core)

        # 2. Translation Settings
        # =================================================
        grp_trans = QtWidgets.QGroupBox("Translation")
        l_trans = QtWidgets.QGridLayout(grp_trans)
        l_trans.setColumnStretch(1, 1)
        
        self.settings_translator_backend = QtWidgets.QComboBox()
        self.settings_translator_backend.addItems(["Ollama", "GGUF", "DeepSeek"])
        self.settings_translator_backend.setCurrentText(self._defaults.translator_backend)
        
        self.settings_ollama_model = QtWidgets.QComboBox()
        self.settings_ollama_model.addItems(["auto-detect"])
        self.settings_model_refresh = QtWidgets.QPushButton("Refresh")
        
        self.settings_gguf_model_path = QtWidgets.QComboBox()
        self.settings_gguf_model_path.setEditable(True)
        self.settings_gguf_model_path.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.settings_gguf_browse = QtWidgets.QPushButton("Browse")
        self.settings_dl_sakura = QtWidgets.QPushButton("Download Sakura 14B")
        self.settings_dl_sakura.clicked.connect(lambda: self._run_model_download("sakura"))
        
        self.settings_gguf_prompt_style = QtWidgets.QComboBox()
        self.settings_gguf_prompt_style.addItems(["sakura", "qwen", "plain"])
        self.settings_gguf_prompt_style.setCurrentText(self._defaults.gguf_prompt_style)

        self.settings_deepseek_model = QtWidgets.QLineEdit(self._defaults.deepseek_model)
        self.settings_deepseek_base_url = QtWidgets.QLineEdit(self._defaults.deepseek_base_url)
        
        l_trans.addWidget(QtWidgets.QLabel("Backend"), 0, 0)
        l_trans.addWidget(self.settings_translator_backend, 0, 1)
        
        l_trans.addWidget(QtWidgets.QLabel("Ollama Model"), 1, 0)
        l_trans.addWidget(self._hbox(self.settings_ollama_model, self.settings_model_refresh), 1, 1)
        
        l_trans.addWidget(QtWidgets.QLabel("GGUF Model"), 2, 0)
        l_trans.addWidget(self._hbox(self.settings_gguf_model_path, self.settings_gguf_browse, self.settings_dl_sakura), 2, 1)
        
        l_trans.addWidget(QtWidgets.QLabel("GGUF Prompt"), 3, 0)
        l_trans.addWidget(self.settings_gguf_prompt_style, 3, 1)

        l_trans.addWidget(QtWidgets.QLabel("DeepSeek Model"), 4, 0)
        l_trans.addWidget(self.settings_deepseek_model, 4, 1)

        l_trans.addWidget(QtWidgets.QLabel("DeepSeek Base URL"), 5, 0)
        l_trans.addWidget(self.settings_deepseek_base_url, 5, 1)

        self.settings_trans_warning = QtWidgets.QLabel(
            "⚠️ GGUF model not found. Browse a valid .gguf file or place it under models/."
        )
        self.settings_trans_warning.setStyleSheet("color: #ef4444; font-size: 10px; font-weight: 600;")
        self.settings_trans_warning.setVisible(False)
        self.settings_trans_warning.setWordWrap(True)
        l_trans.addWidget(self.settings_trans_warning, 6, 1)

        self.settings_trans_ollama_warning = QtWidgets.QLabel(
            "⚠️ Ollama server not available. Install Ollama and run 'ollama serve'."
        )
        self.settings_trans_ollama_warning.setStyleSheet("color: #ef4444; font-size: 10px; font-weight: 600;")
        self.settings_trans_ollama_warning.setVisible(False)
        self.settings_trans_ollama_warning.setWordWrap(True)
        l_trans.addWidget(self.settings_trans_ollama_warning, 7, 1)

        self.settings_trans_deepseek_warning = QtWidgets.QLabel(
            "⚠️ DeepSeek backend selected, but its credential reference is not linked."
        )
        self.settings_trans_deepseek_warning.setStyleSheet("color: #ef4444; font-size: 10px; font-weight: 600;")
        self.settings_trans_deepseek_warning.setVisible(False)
        self.settings_trans_deepseek_warning.setWordWrap(True)
        l_trans.addWidget(self.settings_trans_deepseek_warning, 8, 1)
        
        main_layout.addWidget(grp_trans)

        # 3. Experimental Deep Scan (Glossary) Settings
        # =================================================
        grp_scan = QtWidgets.QGroupBox("Experimental Deep Scan")
        l_scan = QtWidgets.QGridLayout(grp_scan)
        l_scan.setColumnStretch(1, 1)

        self.settings_scan_note = QtWidgets.QLabel(
            "Experimental feature. Keep this off for the normal Auto-Glossary workflow."
        )
        self.settings_scan_note.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self.settings_scan_note.setWordWrap(True)
        l_scan.addWidget(self.settings_scan_note, 0, 0, 1, 2)
        
        self.settings_discovery_backend = QtWidgets.QComboBox()
        self.settings_discovery_backend.addItems(
            self._supported_setting_texts("translation.discovery_backend")
        )
        
        self.settings_discovery_ollama_model = QtWidgets.QComboBox()
        self.settings_discovery_ollama_model.addItems(["auto-detect"])
        
        self.settings_discovery_gguf_path = QtWidgets.QComboBox()
        self.settings_discovery_gguf_path.setEditable(True)
        self.settings_discovery_gguf_path.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.settings_discovery_gguf_browse = QtWidgets.QPushButton("Browse")
        self.settings_discovery_gguf_browse.clicked.connect(self._browse_discovery_gguf)

        self.settings_dl_qwen = QtWidgets.QPushButton("Download Qwen 14B")
        self.settings_dl_qwen.setToolTip("Download Qwen3-14B-GGUF for Deep Scan")
        self.settings_dl_qwen.clicked.connect(lambda: self._run_model_download("qwen"))

        self.settings_mismatch_warning = QtWidgets.QLabel("⚠️ Mismatch: Using different models causes slow reloading.")
        self.settings_mismatch_warning.setStyleSheet("color: #ef4444; font-size: 10px; font-weight: 600;")
        self.settings_mismatch_warning.setVisible(False)
        self.settings_mismatch_warning.setWordWrap(True)

        l_scan.addWidget(QtWidgets.QLabel("Backend"), 1, 0)
        l_scan.addWidget(self.settings_discovery_backend, 1, 1)
        
        l_scan.addWidget(QtWidgets.QLabel("Ollama Model"), 2, 0)
        l_scan.addWidget(self.settings_discovery_ollama_model, 2, 1)
        
        l_scan.addWidget(QtWidgets.QLabel("GGUF Model"), 3, 0)
        l_scan.addWidget(self._hbox(self.settings_discovery_gguf_path, self.settings_discovery_gguf_browse, self.settings_dl_qwen), 3, 1)
        
        l_scan.addWidget(self.settings_mismatch_warning, 4, 1)

        self.settings_scan_gguf_warning = QtWidgets.QLabel(
            "⚠️ GGUF model not found. Browse a valid .gguf file or place it under models/."
        )
        self.settings_scan_gguf_warning.setStyleSheet("color: #ef4444; font-size: 10px; font-weight: 600;")
        self.settings_scan_gguf_warning.setVisible(False)
        self.settings_scan_gguf_warning.setWordWrap(True)
        l_scan.addWidget(self.settings_scan_gguf_warning, 5, 1)

        self.settings_scan_ollama_warning = QtWidgets.QLabel(
            "⚠️ Ollama server not available. Install Ollama and run 'ollama serve'."
        )
        self.settings_scan_ollama_warning.setStyleSheet("color: #ef4444; font-size: 10px; font-weight: 600;")
        self.settings_scan_ollama_warning.setVisible(False)
        self.settings_scan_ollama_warning.setWordWrap(True)
        l_scan.addWidget(self.settings_scan_ollama_warning, 6, 1)
        
        main_layout.addWidget(grp_scan)

        # 4. Generation & Hardware
        # =================================================
        grp_gen = QtWidgets.QGroupBox("Generation & Hardware")
        l_gen = QtWidgets.QGridLayout(grp_gen)
        
        self.settings_gen_preset = QtWidgets.QComboBox()
        self.settings_gen_preset.addItems(["Precise (Default)", "Balanced", "Creative", "Custom"])
        
        self.settings_ollama_temp = QtWidgets.QDoubleSpinBox()
        self.settings_ollama_temp.setRange(0.0, 1.0)
        self.settings_ollama_temp.setSingleStep(0.1)
        self.settings_ollama_temp.setValue(0.2)
        self.settings_ollama_temp.setPrefix("Ollama Temp: ")
        
        self.settings_ollama_top_p = QtWidgets.QDoubleSpinBox()
        self.settings_ollama_top_p.setRange(0.0, 1.0)
        self.settings_ollama_top_p.setValue(0.9)
        self.settings_ollama_top_p.setPrefix("Top P: ")
        
        self.settings_ollama_ctx = QtWidgets.QSpinBox()
        self.settings_ollama_ctx.setRange(2048, 32768)
        self.settings_ollama_ctx.setSingleStep(2048)
        self.settings_ollama_ctx.setValue(4096)
        self.settings_ollama_ctx.setPrefix("Ctx: ")
        
        self.settings_gguf_temp = QtWidgets.QDoubleSpinBox()
        self.settings_gguf_temp.setRange(0.0, 1.0)
        self.settings_gguf_temp.setValue(0.2)
        self.settings_gguf_temp.setPrefix("GGUF Temp: ")
        
        self.settings_gguf_top_p = QtWidgets.QDoubleSpinBox()
        self.settings_gguf_top_p.setRange(0.0, 1.0)
        self.settings_gguf_top_p.setValue(0.95)
        self.settings_gguf_top_p.setPrefix("Top P: ")
        
        self.settings_gguf_n_ctx = QtWidgets.QSpinBox()
        self.settings_gguf_n_ctx.setRange(512, 32768)
        self.settings_gguf_n_ctx.setValue(self._defaults.gguf_n_ctx)
        self.settings_gguf_n_ctx.setPrefix("Ctx: ")

        self.settings_gguf_n_gpu_layers = QtWidgets.QSpinBox()
        self.settings_gguf_n_gpu_layers.setRange(-1, 200)
        self.settings_gguf_n_gpu_layers.setValue(self._defaults.gguf_n_gpu_layers)
        self.settings_gguf_n_gpu_layers.setPrefix("GPU Layers: ")

        self.settings_gguf_n_threads = QtWidgets.QSpinBox()
        self.settings_gguf_n_threads.setRange(1, 128)
        self.settings_gguf_n_threads.setValue(self._defaults.gguf_n_threads)
        self.settings_gguf_n_threads.setPrefix("Threads: ")

        self.settings_gguf_n_batch = QtWidgets.QSpinBox()
        self.settings_gguf_n_batch.setRange(64, 4096)
        self.settings_gguf_n_batch.setValue(self._defaults.gguf_n_batch)
        self.settings_gguf_n_batch.setPrefix("Batch: ")

        l_gen.addWidget(QtWidgets.QLabel("Preset"), 0, 0)
        l_gen.addWidget(self.settings_gen_preset, 0, 1)
        
        l_gen.addWidget(self.settings_ollama_temp, 1, 0)
        l_gen.addWidget(self.settings_ollama_top_p, 1, 1)
        l_gen.addWidget(self.settings_ollama_ctx, 1, 2)
        
        l_gen.addWidget(self.settings_gguf_temp, 2, 0)
        l_gen.addWidget(self.settings_gguf_top_p, 2, 1)
        l_gen.addWidget(self.settings_gguf_n_ctx, 2, 2)
        
        l_gen.addWidget(self.settings_gguf_n_gpu_layers, 3, 0)
        l_gen.addWidget(self.settings_gguf_n_threads, 3, 1)
        l_gen.addWidget(self.settings_gguf_n_batch, 3, 2)
        
        main_layout.addWidget(grp_gen)

        # 5. Per-Model Overrides (Advanced)
        # =================================================
        self.override_group = QtWidgets.QGroupBox("Per-Model Overrides (Advanced)")
        self.override_group.setCheckable(True)
        self.override_group.setChecked(False)
        overrides_layout = QtWidgets.QVBoxLayout(self.override_group)

        self.override_body = QtWidgets.QWidget()
        self.override_body.setVisible(False)
        overrides_layout.addWidget(self.override_body)
        self.override_group.toggled.connect(self.override_body.setVisible)

        l_override = QtWidgets.QGridLayout(self.override_body)
        l_override.setColumnStretch(1, 1)

        self.override_backend = QtWidgets.QComboBox()
        self.override_backend.addItems(["Ollama", "GGUF"])

        self.override_model = QtWidgets.QComboBox()
        self.override_model.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)

        self.override_enabled = QtWidgets.QCheckBox("Enable overrides for this model")

        l_override.addWidget(QtWidgets.QLabel("Backend"), 0, 0)
        l_override.addWidget(self.override_backend, 0, 1)
        l_override.addWidget(QtWidgets.QLabel("Model"), 1, 0)
        l_override.addWidget(self.override_model, 1, 1)
        l_override.addWidget(self.override_enabled, 2, 1)

        self.override_stack = QtWidgets.QStackedWidget()
        l_override.addWidget(self.override_stack, 3, 0, 1, 2)

        # Ollama override page
        override_ollama = QtWidgets.QWidget()
        l_ov_ollama = QtWidgets.QGridLayout(override_ollama)
        self.override_ollama_temp = QtWidgets.QDoubleSpinBox()
        self.override_ollama_temp.setRange(0.0, 1.0)
        self.override_ollama_temp.setSingleStep(0.1)
        self.override_ollama_temp.setPrefix("Ollama Temp: ")
        self.override_ollama_top_p = QtWidgets.QDoubleSpinBox()
        self.override_ollama_top_p.setRange(0.0, 1.0)
        self.override_ollama_top_p.setPrefix("Top P: ")
        self.override_ollama_ctx = QtWidgets.QSpinBox()
        self.override_ollama_ctx.setRange(2048, 32768)
        self.override_ollama_ctx.setSingleStep(2048)
        self.override_ollama_ctx.setPrefix("Ctx: ")
        l_ov_ollama.addWidget(self.override_ollama_temp, 0, 0)
        l_ov_ollama.addWidget(self.override_ollama_top_p, 0, 1)
        l_ov_ollama.addWidget(self.override_ollama_ctx, 0, 2)
        self.override_stack.addWidget(override_ollama)

        # GGUF override page
        override_gguf = QtWidgets.QWidget()
        l_ov_gguf = QtWidgets.QGridLayout(override_gguf)
        self.override_gguf_temp = QtWidgets.QDoubleSpinBox()
        self.override_gguf_temp.setRange(0.0, 1.0)
        self.override_gguf_temp.setPrefix("GGUF Temp: ")
        self.override_gguf_top_p = QtWidgets.QDoubleSpinBox()
        self.override_gguf_top_p.setRange(0.0, 1.0)
        self.override_gguf_top_p.setPrefix("Top P: ")
        self.override_gguf_n_ctx = QtWidgets.QSpinBox()
        self.override_gguf_n_ctx.setRange(512, 32768)
        self.override_gguf_n_ctx.setPrefix("Ctx: ")
        self.override_gguf_n_gpu_layers = QtWidgets.QSpinBox()
        self.override_gguf_n_gpu_layers.setRange(-1, 200)
        self.override_gguf_n_gpu_layers.setPrefix("GPU Layers: ")
        self.override_gguf_n_threads = QtWidgets.QSpinBox()
        self.override_gguf_n_threads.setRange(1, 128)
        self.override_gguf_n_threads.setPrefix("Threads: ")
        self.override_gguf_n_batch = QtWidgets.QSpinBox()
        self.override_gguf_n_batch.setRange(64, 4096)
        self.override_gguf_n_batch.setPrefix("Batch: ")
        l_ov_gguf.addWidget(self.override_gguf_temp, 0, 0)
        l_ov_gguf.addWidget(self.override_gguf_top_p, 0, 1)
        l_ov_gguf.addWidget(self.override_gguf_n_ctx, 0, 2)
        l_ov_gguf.addWidget(self.override_gguf_n_gpu_layers, 1, 0)
        l_ov_gguf.addWidget(self.override_gguf_n_threads, 1, 1)
        l_ov_gguf.addWidget(self.override_gguf_n_batch, 1, 2)
        self.override_stack.addWidget(override_gguf)

        actions_row = QtWidgets.QWidget()
        actions_layout = QtWidgets.QHBoxLayout(actions_row)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.addStretch(1)
        self.override_copy_btn = QtWidgets.QPushButton("Copy from Global Preset")
        self.override_reset_btn = QtWidgets.QPushButton("Reset to Default")
        actions_layout.addWidget(self.override_copy_btn)
        actions_layout.addWidget(self.override_reset_btn)
        l_override.addWidget(actions_row, 4, 0, 1, 2)

        main_layout.addWidget(self.override_group)
        
        # Add stretcher to push everything up
        main_layout.addStretch(1)
        
        return main_group

    def _group_render(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Rendering")
        layout = QtWidgets.QFormLayout(group)
        self.font_name = QtWidgets.QComboBox()
        self.font_name.setEditable(True)
        self.font_name.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        fonts = QtGui.QFontDatabase.families()
        fonts = sorted(set(fonts), key=str.lower)
        if self._defaults.font_name and self._defaults.font_name not in fonts:
            fonts.insert(0, self._defaults.font_name)
        self.font_name.addItems(fonts)
        self.font_name.setCurrentText(self._defaults.font_name)
        layout.addRow("Font", self.font_name)
        self.font_detection = QtWidgets.QComboBox()
        self.font_detection.addItems(
            self._supported_setting_texts("source_style.font_detection")
        )
        self.font_detection.setCurrentText(self._defaults.font_detection)
        layout.addRow("Font Detection", self.font_detection)
        self.inpaint_mode = QtWidgets.QComboBox()
        self.inpaint_mode.addItems(
            self._supported_setting_texts("cleanup.inpaint_mode")
        )
        self.inpaint_mode.setCurrentText(self._defaults.inpaint_mode)
        layout.addRow("Inpainting", self.inpaint_mode)
        
        # Cleanup inpainting is fixed; this field is provenance/status only.
        self.inpaint_model_id = QtWidgets.QLineEdit(self._defaults.inpaint_model)
        self.inpaint_model_id.setReadOnly(True)
        layout.addRow("AI Model (fixed)", self.inpaint_model_id)
        
        return group


    def _group_performance(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Performance")
        layout = QtWidgets.QFormLayout(group)
        self.use_gpu = QtWidgets.QCheckBox("Allow acceleration when available")
        self.use_gpu.setChecked(True)
        layout.addRow("", self.use_gpu)
        return group

    def _group_theme(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Theme")
        layout = QtWidgets.QFormLayout(group)
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.setCurrentText(self._defaults.theme)
        self.theme_combo.setFixedWidth(150)
        layout.addRow("Mode", self.theme_combo)
        return group

    def _build_empty_state(self, title: str, subtitle: str, icon_name: str) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        icon_label = QtWidgets.QLabel("")
        icon_label.setAlignment(QtCore.Qt.AlignCenter)
        icon_label.setFixedSize(64, 64)
        icon_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        icon_label.setStyleSheet("QLabel { color: #44526a; }")
        try:
            import qtawesome as qta
            icon = qta.icon(icon_name, color="#42506a")
            icon_label.setPixmap(icon.pixmap(48, 48))
        except Exception:
            icon_label.setText("")
        title_label = QtWidgets.QLabel(title)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setWordWrap(True)
        title_label.setStyleSheet("QLabel { color: #c6d4ea; font-size: 15px; font-weight: 600; }")
        title_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        subtitle_label = QtWidgets.QLabel(subtitle)
        subtitle_label.setAlignment(QtCore.Qt.AlignCenter)
        subtitle_label.setWordWrap(True)
        subtitle_label.setStyleSheet("QLabel { color: #8fa1bf; font-size: 13px; }")
        subtitle_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        layout.addStretch(1)
        layout.addWidget(icon_label, 0, QtCore.Qt.AlignHCenter)
        layout.addWidget(title_label, 0, QtCore.Qt.AlignHCenter)
        layout.addWidget(subtitle_label, 0, QtCore.Qt.AlignHCenter)
        layout.addStretch(1)
        return widget

    def _hbox(self, *widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget)
        return box

    def _load_icons(self) -> dict[str, QtGui.QIcon]:
        icons: dict[str, QtGui.QIcon] = {}
        try:
            import qtawesome as qta
        except Exception:
            return icons
        accent = "#6ee7ff"
        icons["brand"] = qta.icon("fa5s.magic", color=accent)
        icons["home"] = qta.icon("fa5s.home", color="#cfe3ff")
        icons["queue"] = qta.icon("fa5s.tasks", color="#cfe3ff")
        icons["library"] = qta.icon("fa5s.th-large", color="#cfe3ff")
        icons["settings"] = qta.icon("fa5s.cog", color="#cfe3ff")
        return icons

    def _switch_page(self, index: int) -> None:
        self._active_page_index = index
        self.center_stack.setCurrentIndex(index)
        self.right_stack.setCurrentIndex(index)
        if index == 3:
            self._refresh_effective_run_summary()

    @staticmethod
    def _supported_setting_texts(qualified_id: str) -> list[str]:
        """Project the editable combo vocabulary from the module registry."""

        return [
            str(value)
            for value in DEFAULT_MODULE_REGISTRY.supported_values(qualified_id)
        ]

    def _sync_models_to_settings(self) -> None:
        self.settings_detector_engine.setCurrentText(self.detector_engine.currentText())
        self.settings_detector_input_size.setCurrentText(self.detector_input_size.currentText())
        self.settings_ocr_engine.setCurrentText(self.ocr_engine.currentText())
        self.settings_translator_backend.setCurrentText(self.translator_backend.currentText())
        self._set_combo_text(self.settings_ollama_model, self.ollama_model.currentText())
        self._set_gguf_combo(self.settings_gguf_model_path, self._selected_gguf_model_path())
        self.settings_gguf_prompt_style.setCurrentText(self.gguf_prompt_style.currentText())
        self.settings_gguf_n_ctx.setValue(self.gguf_n_ctx.value())
        self.settings_gguf_n_gpu_layers.setValue(self.gguf_n_gpu_layers.value())
        self.settings_gguf_n_threads.setValue(self.gguf_n_threads.value())
        self.settings_gguf_n_batch.setValue(self.gguf_n_batch.value())

        self.settings_gguf_n_threads.setValue(self.gguf_n_threads.value())
        self.settings_gguf_n_batch.setValue(self.gguf_n_batch.value())
        self.settings_gen_preset.setCurrentText(self.gen_preset.currentText()) if hasattr(self, "gen_preset") else None

    def _sync_settings_to_models(self) -> None:
        self.detector_engine.setCurrentText(self.settings_detector_engine.currentText())
        self.detector_input_size.setCurrentText(self.settings_detector_input_size.currentText())
        self.ocr_engine.setCurrentText(self.settings_ocr_engine.currentText())
        self.translator_backend.setCurrentText(self.settings_translator_backend.currentText())
        self._set_combo_text(self.ollama_model, self.settings_ollama_model.currentText())
        self._set_gguf_combo(self.gguf_model_path, self._settings_selected_gguf())
        self.gguf_prompt_style.setCurrentText(self.settings_gguf_prompt_style.currentText())
        self.gguf_n_ctx.setValue(self.settings_gguf_n_ctx.value())
        self.gguf_n_gpu_layers.setValue(self.settings_gguf_n_gpu_layers.value())
        self.gguf_n_threads.setValue(self.settings_gguf_n_threads.value())
        self.gguf_n_batch.setValue(self.settings_gguf_n_batch.value())

    def _on_preset_changed(self, text: str) -> None:
        # Block signals to prevent _set_custom_preset from firing
        self.settings_ollama_temp.blockSignals(True)
        self.settings_ollama_top_p.blockSignals(True)
        self.settings_gguf_temp.blockSignals(True)
        self.settings_gguf_top_p.blockSignals(True)

        if text.startswith("Precise"):
            self.settings_ollama_temp.setValue(0.1)
            self.settings_ollama_top_p.setValue(0.9)
            self.settings_gguf_temp.setValue(0.1)
            self.settings_gguf_top_p.setValue(0.9)
        elif text == "Balanced":
            self.settings_ollama_temp.setValue(0.3)
            self.settings_ollama_top_p.setValue(0.9)
            self.settings_gguf_temp.setValue(0.3)
            self.settings_gguf_top_p.setValue(0.9)
        elif text == "Creative":
            self.settings_ollama_temp.setValue(0.7)
            self.settings_ollama_top_p.setValue(0.95)
            self.settings_gguf_temp.setValue(0.7)
            self.settings_gguf_top_p.setValue(0.95)
            
        self.settings_ollama_temp.blockSignals(False)
        self.settings_ollama_top_p.blockSignals(False)
        self.settings_gguf_temp.blockSignals(False)
        self.settings_gguf_top_p.blockSignals(False)

    def _set_custom_preset(self) -> None:
        self.settings_gen_preset.blockSignals(True)
        self.settings_gen_preset.setCurrentText("Custom")
        self.settings_gen_preset.blockSignals(False)

    def _sync_paths_to_settings(self) -> None:
        self.settings_import_dir.setText(self.import_dir.text().strip())
        self.settings_export_dir.setText(self.export_dir.text().strip())
        self.settings_json_path.setText(self.json_path.text().strip())
        self.settings_output_suffix.setText(self.output_suffix.text().strip())
        self.settings_save_folder.setText(self.export_dir.text().strip())

    def _sync_paths_from_settings(self) -> None:
        if self.settings_import_dir.text().strip():
            self.import_dir.setText(self.settings_import_dir.text().strip())
        if self.settings_export_dir.text().strip():
            self.export_dir.setText(self.settings_export_dir.text().strip())
        if self.settings_json_path.text().strip():
            self.json_path.setText(self.settings_json_path.text().strip())
        if self.settings_output_suffix.text().strip():
            self.output_suffix.setText(self.settings_output_suffix.text().strip())
        self.settings_save_folder.setText(self.export_dir.text().strip())
        self._schedule_import_preview()

    def _set_combo_text(self, combo: QtWidgets.QComboBox, value: str) -> None:
        if value is None:
            return
        text = str(value).strip()
        if not text:
            return
        index = combo.findText(text)
        if index < 0:
            combo.addItem(text)
            index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _set_gguf_combo(self, combo: QtWidgets.QComboBox, path: str) -> None:
        if not path:
            return
        text = str(path).strip()
        if not text:
            return
        normalized = os.path.abspath(text)
        for idx in range(combo.count()):
            data = combo.itemData(idx)
            if isinstance(data, str) and os.path.abspath(data) == normalized:
                combo.setCurrentIndex(idx)
                return
        norm_text = os.path.normpath(text).lower()
        for idx in range(combo.count()):
            item_text = combo.itemText(idx)
            if item_text and os.path.normpath(item_text).lower() == norm_text:
                combo.setCurrentIndex(idx)
                return
        base = os.path.basename(text)
        if base:
            match_idx = None
            for idx in range(combo.count()):
                item_data = combo.itemData(idx)
                item_text = combo.itemText(idx)
                candidates = []
                if isinstance(item_data, str) and item_data:
                    candidates.append(os.path.basename(item_data))
                if item_text:
                    candidates.append(os.path.basename(item_text))
                if any(c.lower() == base.lower() for c in candidates):
                    if match_idx is not None:
                        match_idx = None
                        break
                    match_idx = idx
            if match_idx is not None:
                combo.setCurrentIndex(match_idx)
                return
        combo.addItem(os.path.basename(normalized), normalized)
        combo.setCurrentIndex(combo.count() - 1)

    def _override_key(self, backend: str, model_id: str) -> str:
        return f"{backend.lower()}::{model_id}"

    def _current_override_target(self) -> tuple[str, str] | None:
        backend = self.override_backend.currentText()
        model_data = self.override_model.currentData()
        model_text = self.override_model.currentText().strip()
        model_id = model_data if isinstance(model_data, str) and model_data else model_text
        if not model_id:
            return None
        return backend, model_id

    def _global_override_values(self, backend: str) -> dict:
        if backend == "Ollama":
            return {
                "ollama_temp": float(self.settings_ollama_temp.value()),
                "ollama_top_p": float(self.settings_ollama_top_p.value()),
                "ollama_ctx": int(self.settings_ollama_ctx.value()),
            }
        return {
            "gguf_temp": float(self.settings_gguf_temp.value()),
            "gguf_top_p": float(self.settings_gguf_top_p.value()),
            "gguf_n_ctx": int(self.settings_gguf_n_ctx.value()),
            "gguf_n_gpu_layers": int(self.settings_gguf_n_gpu_layers.value()),
            "gguf_n_threads": int(self.settings_gguf_n_threads.value()),
            "gguf_n_batch": int(self.settings_gguf_n_batch.value()),
        }

    def _refresh_override_model_list(self) -> None:
        target = self._current_override_target()
        backend = self.override_backend.currentText()
        current_id = target[1] if target and target[0] == backend else ""
        self.override_model.blockSignals(True)
        self.override_model.clear()
        if backend == "Ollama":
            for idx in range(self.settings_ollama_model.count()):
                text = self.settings_ollama_model.itemText(idx)
                if not text:
                    continue
                self.override_model.addItem(text, text)
        else:
            for _, full_path in self._iter_gguf_models():
                self.override_model.addItem(os.path.basename(full_path), full_path)
        if current_id:
            for idx in range(self.override_model.count()):
                data = self.override_model.itemData(idx)
                text = self.override_model.itemText(idx)
                if (isinstance(data, str) and data == current_id) or text == current_id:
                    self.override_model.setCurrentIndex(idx)
                    break
        if self.override_model.count() > 0 and self.override_model.currentIndex() < 0:
            self.override_model.setCurrentIndex(0)
        self.override_model.blockSignals(False)
        self._load_override_for_selection()

    def _set_override_fields_enabled(self, enabled: bool) -> None:
        self.override_stack.setEnabled(enabled)

    def _load_override_for_selection(self) -> None:
        target = self._current_override_target()
        if not target:
            return
        backend, model_id = target
        key = self._override_key(backend, model_id)
        record = self._model_overrides.get(key, {})
        enabled = bool(record.get("enabled"))
        self.override_enabled.blockSignals(True)
        self.override_enabled.setChecked(enabled)
        self.override_enabled.blockSignals(False)
        self._set_override_fields_enabled(enabled)

        values = self._global_override_values(backend)
        if record:
            values.update(record.get("values", {}))

        self._apply_override_values(backend, values)
        self.override_stack.setCurrentIndex(0 if backend == "Ollama" else 1)

    def _apply_override_values(self, backend: str, values: dict) -> None:
        if backend == "Ollama":
            self.override_ollama_temp.blockSignals(True)
            self.override_ollama_top_p.blockSignals(True)
            self.override_ollama_ctx.blockSignals(True)
            self.override_ollama_temp.setValue(float(values.get("ollama_temp", self.settings_ollama_temp.value())))
            self.override_ollama_top_p.setValue(float(values.get("ollama_top_p", self.settings_ollama_top_p.value())))
            self.override_ollama_ctx.setValue(int(values.get("ollama_ctx", self.settings_ollama_ctx.value())))
            self.override_ollama_temp.blockSignals(False)
            self.override_ollama_top_p.blockSignals(False)
            self.override_ollama_ctx.blockSignals(False)
        else:
            self.override_gguf_temp.blockSignals(True)
            self.override_gguf_top_p.blockSignals(True)
            self.override_gguf_n_ctx.blockSignals(True)
            self.override_gguf_n_gpu_layers.blockSignals(True)
            self.override_gguf_n_threads.blockSignals(True)
            self.override_gguf_n_batch.blockSignals(True)
            self.override_gguf_temp.setValue(float(values.get("gguf_temp", self.settings_gguf_temp.value())))
            self.override_gguf_top_p.setValue(float(values.get("gguf_top_p", self.settings_gguf_top_p.value())))
            self.override_gguf_n_ctx.setValue(int(values.get("gguf_n_ctx", self.settings_gguf_n_ctx.value())))
            self.override_gguf_n_gpu_layers.setValue(int(values.get("gguf_n_gpu_layers", self.settings_gguf_n_gpu_layers.value())))
            self.override_gguf_n_threads.setValue(int(values.get("gguf_n_threads", self.settings_gguf_n_threads.value())))
            self.override_gguf_n_batch.setValue(int(values.get("gguf_n_batch", self.settings_gguf_n_batch.value())))
            self.override_gguf_temp.blockSignals(False)
            self.override_gguf_top_p.blockSignals(False)
            self.override_gguf_n_ctx.blockSignals(False)
            self.override_gguf_n_gpu_layers.blockSignals(False)
            self.override_gguf_n_threads.blockSignals(False)
            self.override_gguf_n_batch.blockSignals(False)

    def _read_override_values(self, backend: str) -> dict:
        if backend == "Ollama":
            return {
                "ollama_temp": float(self.override_ollama_temp.value()),
                "ollama_top_p": float(self.override_ollama_top_p.value()),
                "ollama_ctx": int(self.override_ollama_ctx.value()),
            }
        return {
            "gguf_temp": float(self.override_gguf_temp.value()),
            "gguf_top_p": float(self.override_gguf_top_p.value()),
            "gguf_n_ctx": int(self.override_gguf_n_ctx.value()),
            "gguf_n_gpu_layers": int(self.override_gguf_n_gpu_layers.value()),
            "gguf_n_threads": int(self.override_gguf_n_threads.value()),
            "gguf_n_batch": int(self.override_gguf_n_batch.value()),
        }

    def _update_override_record(self) -> None:
        target = self._current_override_target()
        if not target:
            return
        backend, model_id = target
        key = self._override_key(backend, model_id)
        record = self._model_overrides.get(key, {})
        record["enabled"] = bool(self.override_enabled.isChecked())
        if record["enabled"]:
            record["values"] = self._read_override_values(backend)
        self._model_overrides[key] = record

    def _get_override_values(self, backend: str, model_id: str) -> dict | None:
        key = self._override_key(backend, model_id)
        record = self._model_overrides.get(key)
        if not isinstance(record, dict):
            return None
        if not record.get("enabled"):
            return None
        values = record.get("values")
        if isinstance(values, dict):
            return values
        return None

    def _on_override_enabled_changed(self, enabled: bool) -> None:
        self._set_override_fields_enabled(enabled)
        self._update_override_record()

    def _on_override_value_changed(self) -> None:
        if not self.override_enabled.isChecked():
            return
        self._update_override_record()

    def _on_override_copy_global(self) -> None:
        target = self._current_override_target()
        if not target:
            return
        backend, _ = target
        values = self._global_override_values(backend)
        self._apply_override_values(backend, values)
        if self.override_enabled.isChecked():
            self._update_override_record()

    def _on_override_reset(self) -> None:
        target = self._current_override_target()
        if not target:
            return
        backend, model_id = target
        key = self._override_key(backend, model_id)
        if key in self._model_overrides:
            del self._model_overrides[key]
        self.override_enabled.setChecked(False)
        self._apply_override_values(backend, self._global_override_values(backend))

    def _browse_discovery_gguf(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Deep Scan GGUF Model", "", "GGUF Models (*.gguf);;All Files (*)"
        )
        if not path:
            return
        self._add_gguf_model(path)
        # Ensure it is selected in the discovery combo
        self._set_gguf_combo(self.settings_discovery_gguf_path, path)

    def _update_discovery_ui(self) -> None:
        self._check_model_mismatch()
        self._update_scan_warnings()

    def _update_translation_warning(self) -> None:
        backend = self.settings_translator_backend.currentText()
        if backend != "GGUF":
            self.settings_trans_warning.setVisible(False)
        else:
            path = self.settings_gguf_model_path.currentData()
            if not isinstance(path, str) or not path:
                path = self.settings_gguf_model_path.currentText().strip()
            if not path or not os.path.isfile(path):
                self.settings_trans_warning.setText(
                    "⚠️ GGUF model not found. Browse a valid .gguf file or place it under models/."
                )
                self.settings_trans_warning.setVisible(True)
            elif not self._gguf_runtime_available():
                self.settings_trans_warning.setText(
                    "⚠️ llama-cpp-python is not available in this environment. "
                    "Install it or switch Translator backend to Ollama."
                )
                self.settings_trans_warning.setVisible(True)
            else:
                self.settings_trans_warning.setVisible(False)
        self.settings_trans_ollama_warning.setVisible(False)
        if backend == "Ollama":
            self.settings_trans_ollama_warning.setVisible(not self._ollama_available())
        self.settings_trans_deepseek_warning.setVisible(False)
        if backend == "DeepSeek":
            if not self.settings_deepseek_model.text().strip():
                self.settings_trans_deepseek_warning.setText("⚠️ DeepSeek backend selected, but the model name is empty.")
                self.settings_trans_deepseek_warning.setVisible(True)
            elif not self.settings_deepseek_base_url.text().strip():
                self.settings_trans_deepseek_warning.setText("⚠️ DeepSeek backend selected, but the base URL is empty.")
                self.settings_trans_deepseek_warning.setVisible(True)
            elif not self._deepseek_key_configured():
                self.settings_trans_deepseek_warning.setText(
                    "⚠️ DeepSeek backend selected, but its credential reference is not linked."
                )
                self.settings_trans_deepseek_warning.setVisible(True)

    def _update_scan_warnings(self) -> None:
        backend = self.settings_discovery_backend.currentText()
        if backend == "GGUF":
            path = self.settings_discovery_gguf_path.currentData()
            if not isinstance(path, str) or not path:
                path = self.settings_discovery_gguf_path.currentText().strip()
            if not path or not os.path.isfile(path):
                self.settings_scan_gguf_warning.setText(
                    "⚠️ GGUF model not found. Browse a valid .gguf file or place it under models/."
                )
                self.settings_scan_gguf_warning.setVisible(True)
            elif not self._gguf_runtime_available():
                self.settings_scan_gguf_warning.setText(
                    "⚠️ llama-cpp-python is not available in this environment. "
                    "Install it or switch Deep Scan backend to Ollama."
                )
                self.settings_scan_gguf_warning.setVisible(True)
            else:
                self.settings_scan_gguf_warning.setVisible(False)
        else:
            self.settings_scan_gguf_warning.setVisible(False)
        self.settings_scan_ollama_warning.setVisible(False)
        if backend == "Ollama":
            self.settings_scan_ollama_warning.setVisible(not self._ollama_available())

    def _ollama_available(self) -> bool:
        try:
            from app.translate.ollama_client import OllamaClient
            return OllamaClient().is_available(timeout=1)
        except Exception:
            return False

    def _deepseek_key_configured(self) -> bool:
        if self._settings_model is None:
            return False
        profile = self._settings_model.draft.translation_profile
        return bool(
            profile is not None
            and profile.kind is ProviderKind.DEEPSEEK
            and profile.credential_ref is not None
        )

    def _gguf_runtime_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
            return True
        except Exception:
            return False

    def _settings_selected_gguf(self) -> str:
        data = self.settings_gguf_model_path.currentData()
        if isinstance(data, str) and data:
            return data
        return self.settings_gguf_model_path.currentText().strip()

    def _connect_signals(self) -> None:
        self.start_btn.clicked.connect(self._toggle_start)
        self.stop_btn.clicked.connect(self._pipeline.stop)
        self.theme_combo.currentTextChanged.connect(self._apply_theme)
        self.model_refresh.clicked.connect(self._refresh_models)
        self.settings_model_refresh.clicked.connect(self._refresh_models)
        self.review_btn.clicked.connect(self._open_region_review)

        self.import_browse.clicked.connect(self._choose_import_dir)
        self.export_browse.clicked.connect(self._choose_export_dir)
        self.json_browse.clicked.connect(self._choose_json_path)
        self.json_path.editingFinished.connect(self._on_project_path_edited)
        self.style_browse.clicked.connect(self._choose_style_path)
        self.style_edit.clicked.connect(self._open_style_editor)
        self.gguf_browse.clicked.connect(self._choose_gguf_model)
        self.settings_gguf_browse.clicked.connect(self._choose_gguf_model)
        
        # Deep Scan signals
        self.settings_discovery_backend.currentTextChanged.connect(self._update_discovery_ui)
        self.settings_discovery_ollama_model.currentTextChanged.connect(self._check_model_mismatch)
        
        # Model verification signals
        self.settings_ollama_model.currentTextChanged.connect(self._check_model_mismatch)
        self.settings_translator_backend.currentTextChanged.connect(self._check_model_mismatch)
        self.use_ollama_discovery.toggled.connect(self._check_model_mismatch)
        self.auto_glossary.toggled.connect(self._update_glossary_controls)
        self.settings_translator_backend.currentTextChanged.connect(self._update_translation_warning)
        self.settings_gguf_model_path.currentTextChanged.connect(self._update_translation_warning)
        self.settings_deepseek_model.textChanged.connect(self._update_translation_warning)
        self.settings_deepseek_base_url.textChanged.connect(self._update_translation_warning)
        self.settings_discovery_backend.currentTextChanged.connect(self._update_scan_warnings)
        self.settings_discovery_gguf_path.currentTextChanged.connect(self._update_scan_warnings)
        
        # Generation Presets
        self.settings_gen_preset.currentTextChanged.connect(self._on_preset_changed)
        # Verify if manual changes trigger Custom
        self.settings_ollama_temp.valueChanged.connect(self._set_custom_preset)
        self.settings_ollama_top_p.valueChanged.connect(self._set_custom_preset)
        self.settings_gguf_temp.valueChanged.connect(self._set_custom_preset)
        self.settings_gguf_top_p.valueChanged.connect(self._set_custom_preset)
        self.settings_gguf_n_ctx.valueChanged.connect(self._set_custom_preset)

        # Per-model overrides
        self.override_backend.currentTextChanged.connect(self._refresh_override_model_list)
        self.override_model.currentIndexChanged.connect(self._load_override_for_selection)
        self.override_enabled.toggled.connect(self._on_override_enabled_changed)
        self.override_copy_btn.clicked.connect(self._on_override_copy_global)
        self.override_reset_btn.clicked.connect(self._on_override_reset)
        self.override_ollama_temp.valueChanged.connect(self._on_override_value_changed)
        self.override_ollama_top_p.valueChanged.connect(self._on_override_value_changed)
        self.override_ollama_ctx.valueChanged.connect(self._on_override_value_changed)
        self.override_gguf_temp.valueChanged.connect(self._on_override_value_changed)
        self.override_gguf_top_p.valueChanged.connect(self._on_override_value_changed)
        self.override_gguf_n_ctx.valueChanged.connect(self._on_override_value_changed)
        self.override_gguf_n_gpu_layers.valueChanged.connect(self._on_override_value_changed)
        self.override_gguf_n_threads.valueChanged.connect(self._on_override_value_changed)
        self.override_gguf_n_batch.valueChanged.connect(self._on_override_value_changed)

        self.nav_home.clicked.connect(lambda: self._switch_page(0))
        self.nav_queue.clicked.connect(lambda: self._switch_page(1))
        self.nav_library.clicked.connect(lambda: self._switch_page(2))
        self.nav_settings.clicked.connect(lambda: self._switch_page(3))
        self.queue_list.currentItemChanged.connect(self._on_queue_selected)
        self.queue_table.itemSelectionChanged.connect(self._on_queue_table_selected)
        self.library_list.itemSelectionChanged.connect(self._on_library_selected)
        self.job_open.clicked.connect(self._open_job_folder)
        self.library_open.clicked.connect(self._open_library_folder)
        self.settings_open_folder.clicked.connect(self._open_export_folder)
        self.inspector_back.clicked.connect(self._show_home_default_panel)

        self._pipeline.status.progress_changed.connect(self._set_progress)
        self._pipeline.status.eta_changed.connect(self._set_eta)
        self._pipeline.status.page_changed.connect(self._set_page)
        self._pipeline.status.total_time_changed.connect(self._set_total_time)
        self._pipeline.status.page_time_changed.connect(self._set_page_time)
        self._pipeline.status.message.connect(self._handle_message)
        self._pipeline.status.queue_reset.connect(self._set_queue)
        self._pipeline.status.queue_item.connect(self._update_queue_item)
        self._pipeline.status.page_ready.connect(self._on_page_ready)
        # Two-Pass Pipeline: prescan progress signals
        self._pipeline.status.prescan_started.connect(self._on_prescan_started)
        self._pipeline.status.prescan_progress.connect(self._on_prescan_progress)
        self._pipeline.status.prescan_finished.connect(self._on_prescan_finished)
        self.queue_list.itemDoubleClicked.connect(self._open_page_review)
        self.queue_list.verticalScrollBar().valueChanged.connect(self._update_visible_thumbnails)
        self.import_dir.textChanged.connect(self._schedule_import_preview)

        self._refresh_models()
        self._refresh_gguf_models()
        self._refresh_import_preview()
        self._sync_models_to_settings()
        self._sync_paths_to_settings()
        self._initialize_settings_authority()
        self._connect_live_settings_authority()

    def _connect_live_settings_authority(self) -> None:
        """Route every editable workflow control into one typed draft.

        The visible Settings controls and the shared Home controls are inputs
        only.  Hidden legacy model controls are updated solely as projections
        so neither Start nor page navigation has to choose a mirror direction.
        """

        home_combo_inputs = (
            self.detector_engine,
            self.detector_input_size,
            self.ocr_engine,
            self.translator_backend,
            self.ollama_model,
            self.gguf_model_path,
            self.gguf_prompt_style,
        )
        for widget in home_combo_inputs:
            widget.currentTextChanged.connect(self._dispatch_home_settings_edit)
        for widget in (
            self.gguf_n_ctx,
            self.gguf_n_gpu_layers,
            self.gguf_n_threads,
            self.gguf_n_batch,
        ):
            widget.valueChanged.connect(self._dispatch_home_settings_edit)

        combo_inputs = (
            self.source_lang,
            self.target_lang,
            self.inpaint_mode,
            self.font_detection,
            self.font_name,
            self.theme_combo,
            self.settings_detector_engine,
            self.settings_detector_input_size,
            self.settings_ocr_engine,
            self.settings_translator_backend,
            self.settings_ollama_model,
            self.settings_gguf_model_path,
            self.settings_gguf_prompt_style,
            self.settings_discovery_backend,
            self.settings_discovery_ollama_model,
            self.settings_discovery_gguf_path,
            self.settings_gen_preset,
        )
        for widget in combo_inputs:
            widget.currentTextChanged.connect(self._dispatch_settings_edit)

        numeric_inputs = (
            self.settings_ollama_temp,
            self.settings_ollama_top_p,
            self.settings_ollama_ctx,
            self.settings_gguf_temp,
            self.settings_gguf_top_p,
            self.settings_gguf_n_ctx,
            self.settings_gguf_n_gpu_layers,
            self.settings_gguf_n_threads,
            self.settings_gguf_n_batch,
            self.override_ollama_temp,
            self.override_ollama_top_p,
            self.override_ollama_ctx,
            self.override_gguf_temp,
            self.override_gguf_top_p,
            self.override_gguf_n_ctx,
            self.override_gguf_n_gpu_layers,
            self.override_gguf_n_threads,
            self.override_gguf_n_batch,
        )
        for widget in numeric_inputs:
            widget.valueChanged.connect(self._dispatch_settings_edit)

        boolean_inputs = (
            self.use_gpu,
            self.auto_glossary,
            self.prescan_enabled,
            self.use_ollama_discovery,
            self.override_enabled,
        )
        for widget in boolean_inputs:
            widget.toggled.connect(self._dispatch_settings_edit)
        self.override_copy_btn.clicked.connect(self._dispatch_settings_edit)
        self.override_reset_btn.clicked.connect(self._dispatch_settings_edit)

        text_inputs = (
            self.output_suffix,
            self.style_path,
            self.inpaint_model_id,
            self.settings_deepseek_model,
            self.settings_deepseek_base_url,
        )
        for widget in text_inputs:
            widget.editingFinished.connect(self._dispatch_settings_edit)

        for widget in (self.import_dir, self.export_dir, self.json_path):
            widget.textChanged.connect(self._dispatch_project_path_edit)
        self.output_suffix.textChanged.connect(self._dispatch_project_path_edit)

    def _dispatch_home_settings_edit(self, *_args) -> None:
        if self._settings_projection_guard or self._settings_model is None:
            return
        self._settings_projection_guard = True
        try:
            self._sync_models_to_settings()
        finally:
            self._settings_projection_guard = False
        self._dispatch_settings_edit()

    def _dispatch_project_path_edit(self, *_args) -> None:
        if self._settings_projection_guard:
            return
        self._settings_projection_guard = True
        try:
            self._sync_paths_to_settings()
        finally:
            self._settings_projection_guard = False
        self._refresh_effective_run_summary()

    def _dispatch_settings_edit(self, *_args) -> None:
        if self._settings_projection_guard or self._settings_model is None:
            return
        previous = self._settings_model.draft
        try:
            self._settings_model.replace_application(
                replace(
                    previous.application,
                    theme=self.theme_combo.currentText(),
                )
            )
            draft = self._settings_model.replace_from_legacy_shell(
                self._settings_projection()
            )
        except (TypeError, ValueError) as exc:
            self._settings_model.replace_draft(previous)
            self._apply_settings_draft_to_shell(previous)
            self.status_bar.showMessage(f"Settings edit was not applied: {exc}", 5000)
            return
        self._apply_settings_draft_to_shell(draft)
        self._refresh_effective_run_summary()

    def _effective_run_preview_project_id(self) -> str:
        json_path = self.json_path.text().strip()
        normalized = os.path.normcase(os.path.abspath(json_path or "project.json"))
        if (
            self._active_project_id
            and self._active_run_json_path == normalized
        ):
            return self._active_project_id
        return f"project:gui-preview:{canonical_fingerprint({'json_path': normalized})[:32]}"

    def _refresh_effective_run_summary(self) -> None:
        """Refresh the read-only candidate shown before a run can start."""

        if self._settings_model is None or not hasattr(
            self, "settings_effective_status"
        ):
            return
        try:
            summary = self._settings_model.preview_effective_run_summary(
                project_id=self._effective_run_preview_project_id(),
                invocation=RunInvocation(
                    import_dir=self.import_dir.text().strip(),
                    export_dir=self.export_dir.text().strip(),
                    json_path=self.json_path.text().strip(),
                ),
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self.settings_effective_status.setText("Unavailable")
            self.settings_effective_validation.setText(str(exc))
            return

        status = (
            "Configuration ready"
            if summary.ready
            else "Configuration needs attention"
        )
        if summary.pending_changes:
            status += " · Pending changes"
        self.settings_effective_status.setText(status)
        self.settings_effective_language.setText(summary.language_pair)
        self.settings_effective_provider.setText(summary.provider)
        self.settings_effective_model.setText(summary.model)
        self.settings_effective_detection.setText(summary.detection_and_ocr)
        self.settings_effective_cleanup.setText(summary.cleanup_and_style)
        self.settings_effective_runtime.setText(summary.runtime)
        snapshot_hash = summary.snapshot_id.rsplit(":", 1)[-1]
        self.settings_effective_snapshot.setText(snapshot_hash[:12])
        if summary.issues:
            visible = [issue.message for issue in summary.issues[:2]]
            remaining = len(summary.issues) - len(visible)
            if remaining:
                visible.append(f"+{remaining} more")
            self.settings_effective_validation.setText("; ".join(visible))
        else:
            self.settings_effective_validation.setText("No blocking issues.")

    def _apply_theme(self, theme: str) -> None:
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        if theme == "light":
            apply_light_palette(app)
        else:
            apply_dark_palette(app)

    def _choose_import_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Manga Folder")
        if path:
            self.import_dir.setText(path)
            self.settings_import_dir.setText(path)
            self._schedule_import_preview()

    def _choose_export_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if path:
            self.export_dir.setText(path)
            self.settings_export_dir.setText(path)
            if not self.json_path.text().strip():
                self.json_path.setText(f"{path}\\project.json")
                self.settings_json_path.setText(self.json_path.text().strip())

    def _choose_json_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select Project JSON", filter="JSON Files (*.json)")
        if path:
            self.json_path.setText(path)
            self.settings_json_path.setText(path)
            self._hydrate_typed_project_path(path)

    def _on_project_path_edited(self) -> None:
        self._hydrate_typed_project_path(self.json_path.text().strip())

    def _hydrate_typed_project_path(self, path: str) -> None:
        if self._running or self._settings_model is None or not os.path.isfile(path):
            return
        normalized_path = os.path.normcase(os.path.abspath(path))
        if normalized_path == self._active_run_json_path:
            return
        try:
            stored_project = load_project_for_editing(path)
            stored_settings = read_project_settings(stored_project)
            application_modules = self._module_configs_for_scope(
                self._settings_model.draft.module_configs,
                SettingsScope.APPLICATION,
            )
            modules = {
                config.module_id: config for config in application_modules
            }
            modules.update(
                {
                    config.module_id: config
                    for config in stored_settings.module_configs
                }
            )
            self._settings_model.replace_project(stored_settings.project_config)
            self._settings_model.replace_modules(
                tuple(modules[key] for key in sorted(modules))
            )
            self._settings_model.apply()
            self._last_run_snapshot = stored_settings.last_run_snapshot
            self._active_project_id = str(
                (stored_project.get("project") or {}).get("project_id") or ""
            ) or None
            self._active_run_json_path = normalized_path
            self._apply_settings_draft_to_shell(self._settings_model.draft)
            self._refresh_effective_run_summary()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.status_bar.showMessage(
                f"Project settings could not be loaded: {exc}",
                8000,
            )

    def _choose_style_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Style Guide", filter="JSON Files (*.json)")
        if path:
            self.style_path.setText(path)
            self._dispatch_settings_edit()

    def _choose_gguf_model(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select GGUF Model", filter="GGUF Files (*.gguf)")
        if path:
            self._add_gguf_model(path)

    def _schedule_import_preview(self) -> None:
        if self._running:
            return
        self._preview_timer.start(250)

    def _refresh_import_preview(self) -> None:
        if self._running:
            return
        folder = self.import_dir.text().strip()
        if not folder or not os.path.isdir(folder):
            if self.queue_list.count() > 0:
                self._set_queue([])
            self._last_preview_dir = ""
            return
        if folder == self._last_preview_dir and self.queue_list.count() > 0:
            return
        self._last_preview_dir = folder
        items = self._list_images(folder)
        self._set_queue(items)

    def _list_images(self, folder: str) -> list[str]:
        if not folder or not os.path.isdir(folder):
            return []
        allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        names = []
        for entry in os.listdir(folder):
            _, ext = os.path.splitext(entry)
            if ext.lower() in allowed:
                names.append(entry)
        names.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)])
        return names

    def _models_dir(self) -> str:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(base, "models")

    def _iter_gguf_models(self) -> list[tuple[str, str]]:
        root = self._models_dir()
        results: list[tuple[str, str]] = []
        if not os.path.isdir(root):
            return results
        for dirpath, _, files in os.walk(root):
            for name in files:
                if not name.lower().endswith(".gguf"):
                    continue
                full_path = os.path.join(dirpath, name)
                rel_path = os.path.relpath(full_path, root)
                results.append((rel_path, full_path))
        results.sort(key=lambda item: item[0].lower())
        return results

    def _resolve_gguf_selection(self, candidate: str, models: list[tuple[str, str]]) -> str | None:
        if not candidate:
            return None
        text = str(candidate).strip()
        if not text:
            return None
        norm_text = os.path.normpath(text).lower()
        for display, full_path in models:
            if os.path.normpath(full_path).lower() == norm_text:
                return full_path
        matches = [full_path for display, full_path in models if os.path.normpath(display).lower() == norm_text]
        if len(matches) == 1:
            return matches[0]
        base = os.path.basename(text).lower()
        if base:
            matches = [full_path for _, full_path in models if os.path.basename(full_path).lower() == base]
            if len(matches) == 1:
                return matches[0]
        return None

    def _add_gguf_model(self, path: str) -> None:
        if not path:
            return
        path = os.path.abspath(path)
        for combo in (self.gguf_model_path, self.settings_gguf_model_path, self.settings_discovery_gguf_path):
            existing = False
            for idx in range(combo.count()):
                if combo.itemData(idx) == path:
                    existing = True
                    combo.setCurrentIndex(idx)
                    break
            if not existing:
                display = os.path.basename(path)
                combo.addItem(display, path)
                combo.setCurrentIndex(combo.count() - 1)

    def _refresh_gguf_models(self) -> None:
        current = self._selected_gguf_model_path()
        settings_current = self._settings_selected_gguf()
        # Capture current selection for Deep Scan GGUF
        ds_gguf_current = self.settings_discovery_gguf_path.currentData()
        if not ds_gguf_current:
             ds_gguf_current = self.settings_discovery_gguf_path.currentText().strip()

        combos = (self.gguf_model_path, self.settings_gguf_model_path, self.settings_discovery_gguf_path)
        for combo in combos:
            combo.blockSignals(True)
            combo.clear()
            
        models = self._iter_gguf_models()
        if models:
            for _, full_path in models:
                display = os.path.basename(full_path)
                for combo in combos:
                     combo.addItem(display, full_path)
        
        resolved_default = self._resolve_gguf_selection(self._defaults.gguf_model_path, models)
        resolved_current = self._resolve_gguf_selection(current, models)
        resolved_settings = self._resolve_gguf_selection(settings_current, models)
        if resolved_default:
            self._add_gguf_model(resolved_default)
        elif resolved_current:
            self._add_gguf_model(resolved_current)
        elif resolved_settings:
            self._add_gguf_model(resolved_settings)
        elif models:
            self.gguf_model_path.setCurrentIndex(0)
            
        # Restore Deep Scan selection explicitly (it's independent)
        resolved_ds = self._resolve_gguf_selection(ds_gguf_current, models)
        if resolved_ds:
             # Try to find existing item first to avoid duplicates
             found = False
             combo = self.settings_discovery_gguf_path
             for idx in range(combo.count()):
                 # Check against Data (Absolute Path) or Text (Display Name)
                 if combo.itemData(idx) == resolved_ds or combo.itemText(idx) == resolved_ds:
                     combo.setCurrentIndex(idx)
                     found = True
                     break
             if not found:
                 self._set_gguf_combo(combo, resolved_ds)

        for combo in combos:
            combo.blockSignals(False)
        self._update_translation_warning()
        self._update_scan_warnings()
        self._refresh_override_model_list()

    def _selected_gguf_model_path(self) -> str:
        data = self.gguf_model_path.currentData()
        if isinstance(data, str) and data:
            return data
        text = self.gguf_model_path.currentText().strip()
        if text:
            return text
        return self._settings_selected_gguf()

    def _open_style_editor(self) -> None:
        if not self._style_editor:
            self._style_editor = StyleGuideEditor(self)
        path = self.style_path.text().strip()
        self._style_editor.set_path(path)
        if path:
            try:
                self._style_editor.load_from_path(path)
            except Exception:
                pass
        if self._style_editor.exec() == QtWidgets.QDialog.Accepted:
            if self._style_editor._path:
                self.style_path.setText(self._style_editor._path)

    def _refresh_models(self) -> None:
        models = list_models()
        current = self.ollama_model.currentText()
        settings_current = self.settings_ollama_model.currentText()
        
        # Also need to refresh settings_discovery_ollama_model
        discovery_current = self.settings_discovery_ollama_model.currentText()
        
        for combo in (self.ollama_model, self.settings_ollama_model, self.settings_discovery_ollama_model):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(["auto-detect"])
            if models:
                combo.addItems(models)
            combo.blockSignals(False)
            
        if current:
            self._set_combo_text(self.ollama_model, current)
        if settings_current:
            self._set_combo_text(self.settings_ollama_model, settings_current)
        if discovery_current:
            self._set_combo_text(self.settings_discovery_ollama_model, discovery_current)
        self._update_translation_warning()
        self._update_scan_warnings()
        self._refresh_override_model_list()

    def _set_progress(self, value: int) -> None:
        self.overall_bar.setValue(value)
        self.progress_title.setText(f"Total Progress: {value}%")

    def _set_eta(self, eta_text: str) -> None:
        self.eta_label.setText(f"ETA: {eta_text}")

    def _set_page(self, current: int, total: int) -> None:
        self.processing_label.setText(f"Processing Page {current} of {total}...")

    def _set_total_time(self, text: str) -> None:
        self.total_time_label.setText(text)

    def _set_page_time(self, text: str) -> None:
        self.page_time_label.setText(text)

    def _on_prescan_started(self) -> None:
        """Show prescan progress bar when pre-scan begins."""
        self.prescan_label.setVisible(True)
        self.prescan_bar.setVisible(True)
        self.prescan_bar.setValue(0)
        self.prescan_label.setText("Pre-Scan: Scanning for names...")
        self.progress_title.setText("Translation: Waiting for pre-scan...")
        self.overall_bar.setValue(0)

    def _on_prescan_progress(self, value: int) -> None:
        """Update prescan progress bar."""
        self.prescan_bar.setValue(value)
        self.prescan_label.setText(f"Pre-Scan: {value}%")

    def _on_prescan_finished(self) -> None:
        """Hide prescan progress bar and enable translation progress."""
        self.prescan_bar.setValue(100)
        self.prescan_label.setText("Pre-Scan: Complete ✓")
        self.progress_title.setText("Translation: Starting...")

    def _set_queue(self, items: list) -> None:
        self.queue_list.clear()
        self.queue_table.setRowCount(0)
        self.library_list.clear()
        self.total_time_label.setText("Total: 00:00")
        self.page_time_label.setText("Page: --")
        self.progress_title.setText("Total Progress: 0%")
        self.overall_bar.setValue(0)
        self._page_cache = {}
        self._thumb_cache = {}
        for row_index, name in enumerate(items):
            item = QtWidgets.QListWidgetItem("") # Hide text overlay
            item.setData(QtCore.Qt.UserRole, {"path": os.path.join(self.import_dir.text().strip(), name), "status": "pending"})
            item.setSizeHint(QtCore.QSize(140, 210))
            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
            self.queue_list.addItem(item)
            self._refresh_item_text(item)
            table_row = self.queue_table.rowCount()
            self.queue_table.insertRow(table_row)
            page_item = QtWidgets.QTableWidgetItem(name)
            status_item = QtWidgets.QTableWidgetItem("pending")
            action_btn = QtWidgets.QPushButton("Remove")
            action_btn.setProperty("tableAction", True)
            action_btn.setMinimumWidth(120)
            action_btn.setMinimumHeight(28)
            action_btn.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
            action_btn.clicked.connect(self._on_queue_remove_clicked)
            action_cell = QtWidgets.QWidget()
            action_layout = QtWidgets.QHBoxLayout(action_cell)
            action_layout.setContentsMargins(6, 2, 6, 2)
            action_layout.addStretch(1)
            action_layout.addWidget(action_btn)
            action_layout.addStretch(1)
            self.queue_table.setItem(table_row, 0, page_item)
            self.queue_table.setItem(table_row, 1, status_item)
            self.queue_table.setCellWidget(table_row, 2, action_cell)
            self.queue_table.setRowHeight(table_row, 52)
        self._update_visible_thumbnails()
        self._update_queue_placeholder()
        self.job_file.setText("--")
        self.job_status.setText("--")
        self.job_page.setText("--")
        self.job_stage.setText("--")
        self.library_file.setText("--")
        self.library_pages.setText("--")
        self.library_size.setText("--")

    def _update_queue_item(self, index: int, status: str) -> None:
        item = self.queue_list.item(index)
        if not item:
            return
        data = item.data(QtCore.Qt.UserRole) or {}
        data["status"] = status
        item.setData(QtCore.Qt.UserRole, data)
        self._refresh_item_text(item)
        self._update_thumbnail_item(item)
        if 0 <= index < self.queue_table.rowCount():
            status_item = self.queue_table.item(index, 1)
            if status_item:
                status_item.setText(status)
        self._update_queue_placeholder()

    def _toggle_start(self) -> None:
        if self._running:
            self._pipeline.stop()
            return
        if self._start_pipeline():
            self._set_running(True)

    def _set_running(self, running: bool) -> None:
        self._running = running
        if running:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        else:
            self.start_btn.setEnabled(self._pyicu_runtime_ready)
            self.stop_btn.setEnabled(False)

    def _refresh_item_text(self, item: QtWidgets.QListWidgetItem) -> None:
        # Status is now indicated by border/icon, and name is hidden for cleaner UI.
        item.setText("")
        item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)

    def _load_thumbnail(self, path: str, size: QtCore.QSize | None = None) -> QtGui.QPixmap | None:
        if not path:
            return None
        if size is None:
            size = self.queue_list.iconSize()
        cache_key = f"{path}|{size.width()}x{size.height()}"
        cached = self._thumb_cache.get(cache_key)
        if cached:
            return cached
        image = QtGui.QImage(path)
        if image.isNull():
            return None
        pixmap = QtGui.QPixmap.fromImage(image)
        pixmap = pixmap.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self._thumb_cache[cache_key] = pixmap
        return pixmap

    def _update_thumbnail_item(self, item: QtWidgets.QListWidgetItem) -> None:
        data = item.data(QtCore.Qt.UserRole) or {}
        path = data.get("path", "")
        status = data.get("status", "pending")
        confidence = data.get("confidence")
        needs_review = bool(data.get("needs_review"))
        base = self._load_thumbnail(path, self.queue_list.iconSize())
        if not base:
            return
        icon = QtGui.QIcon(self._decorate_thumbnail(base, status, confidence, needs_review))
        item.setIcon(icon)
        if status.startswith("error"):
            item.setBackground(QtGui.QColor(120, 30, 30, 90))
        elif status.startswith("processing"):
            item.setBackground(QtGui.QColor(40, 80, 140, 80))
        elif status.startswith("done"):
            item.setBackground(QtGui.QColor(40, 120, 80, 60))
        else:
            item.setBackground(QtGui.QColor(0, 0, 0, 0))

    def _decorate_thumbnail(self, pixmap: QtGui.QPixmap, status: str, confidence, needs_review: bool) -> QtGui.QPixmap:
        decorated = QtGui.QPixmap(pixmap)
        painter = QtGui.QPainter(decorated)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        if status.startswith("processing"):
            phase = self._processing_phase % 3
            color = QtGui.QColor(60, 150, 255)
        elif status.startswith("error"):
            color = QtGui.QColor(220, 80, 80)
        elif needs_review or (confidence is not None and confidence < 0.7):
            color = QtGui.QColor(255, 200, 0)
        elif status.startswith("done"):
            color = QtGui.QColor(60, 200, 120)
        else:
            color = QtGui.QColor(80, 90, 110)
        pen = QtGui.QPen(color, 4)
        painter.setPen(pen)
        painter.drawRoundedRect(2, 2, decorated.width() - 4, decorated.height() - 4, 10, 10)
        if status.startswith("processing"):
            spinner_color = QtGui.QColor(120, 190, 255)
            pen = QtGui.QPen(spinner_color, 3)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            radius = 10
            cx = decorated.width() - 22
            cy = decorated.height() - 20
            rect = QtCore.QRect(cx - radius, cy - radius, radius * 2, radius * 2)
            start_angle = (self._processing_phase * 60) % 360
            painter.drawArc(rect, start_angle * 16, 240 * 16)
        if status.startswith("done"):
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(60, 200, 120))
            painter.drawEllipse(QtCore.QPoint(18, decorated.height() - 18), 10, 10)
            painter.setPen(QtGui.QPen(QtGui.QColor(12, 22, 18), 2))
            painter.drawLine(12, decorated.height() - 18, 16, decorated.height() - 14)
            painter.drawLine(16, decorated.height() - 14, 24, decorated.height() - 22)
        painter.end()
        return decorated

    def _pulse_processing(self) -> None:
        self._processing_phase = (self._processing_phase + 1) % 3
        self._update_visible_thumbnails()

    def _update_queue_placeholder(self) -> None:
        if not hasattr(self, "queue_stack"):
            return
        if self.queue_list.count() == 0:
            self.queue_stack.setCurrentIndex(0)
        else:
            self.queue_stack.setCurrentIndex(1)
        self._update_queue_table_placeholder()
        self._update_library_placeholder()

    def _update_queue_table_placeholder(self) -> None:
        if not hasattr(self, "queue_table_stack"):
            return
        if self.queue_table.rowCount() == 0:
            self.queue_table_stack.setCurrentIndex(0)
        else:
            self.queue_table_stack.setCurrentIndex(1)

    def _update_library_placeholder(self) -> None:
        if not hasattr(self, "library_stack"):
            return
        if self.library_list.count() == 0:
            self.library_stack.setCurrentIndex(0)
        else:
            self.library_stack.setCurrentIndex(1)

    def _update_visible_thumbnails(self) -> None:
        viewport = self.queue_list.viewport()
        if viewport is None:
            return
        rect = viewport.rect()
        for index in range(self.queue_list.count()):
            item = self.queue_list.item(index)
            if not item:
                continue
            item_rect = self.queue_list.visualItemRect(item)
            if not rect.intersects(item_rect):
                continue
            self._update_thumbnail_item(item)

    def _check_model_mismatch(self) -> None:
        """
        Check if the user has selected different models for Translation and Hybrid Discovery.
        """
        # Only relevant if using Ollama for translation
        if self.settings_translator_backend.currentText() != "Ollama":
            self.settings_mismatch_warning.setVisible(False)
            return

        # Only relevant if Hybrid Discovery is enabled AND using Ollama backend
        if not self.use_ollama_discovery.isChecked():
            self.settings_mismatch_warning.setVisible(False)
            return
            
        if self.settings_discovery_backend.currentText() != "Ollama":
             self.settings_mismatch_warning.setVisible(False)
             return
            
        trans_model = self.settings_ollama_model.currentText()
        disc_model = self.settings_discovery_ollama_model.currentText()
        
        # If discovery is "Auto-detect", our backend logic handles it safely (matches trans_model)
        if not disc_model or disc_model.lower() == "auto-detect":
            self.settings_mismatch_warning.setVisible(False)
            return
            
        # If models match (exact string match), no problem
        if trans_model == disc_model:
            self.settings_mismatch_warning.setVisible(False)
            return
            
        # Mismatch detected!
        self.settings_mismatch_warning.setVisible(True)

    def _update_glossary_controls(self) -> None:
        enabled = self.auto_glossary.isChecked()
        if not enabled:
            self.prescan_enabled.blockSignals(True)
            self.use_ollama_discovery.blockSignals(True)
            self.prescan_enabled.setChecked(False)
            self.use_ollama_discovery.setChecked(False)
            self.prescan_enabled.blockSignals(False)
            self.use_ollama_discovery.blockSignals(False)

        self.prescan_enabled.setEnabled(enabled)
        self.use_ollama_discovery.setEnabled(enabled)

        if enabled:
            self.prescan_enabled.setToolTip(
                "Scan the chapter/volume before translation starts and build the glossary upfront.\n"
                "Recommended for volumes and best proper-noun consistency."
            )
            self.use_ollama_discovery.setToolTip(
                "Optional accuracy boost for difficult discovery cases.\n"
                "Uses an additional LLM path for background name/entity discovery.\n"
                "Leave this off for the normal fast local workflow."
            )
        else:
            disabled_note = "\nDisabled until Auto-Glossary / Name Memory is enabled."
            self.prescan_enabled.setToolTip(
                "Scan the chapter/volume before translation starts and build the glossary upfront.\n"
                "Recommended for volumes and best proper-noun consistency."
                + disabled_note
            )
            self.use_ollama_discovery.setToolTip(
                "Optional accuracy boost for difficult discovery cases.\n"
                "Uses an additional LLM path for background name/entity discovery.\n"
                "Leave this off for the normal fast local workflow."
                + disabled_note
            )
        self._check_model_mismatch()

    def _on_page_ready(self, index: int, page_record: dict) -> None:
        self._page_cache[index] = page_record
        item = self.queue_list.item(index)
        

        if not item:
            return
        confidence, needs_review = self._compute_page_confidence(page_record)
        data = item.data(QtCore.Qt.UserRole) or {}
        data["confidence"] = confidence
        data["needs_review"] = needs_review
        data["output_path"] = page_record.get("output_path", "")
        data["page"] = page_record
        item.setData(QtCore.Qt.UserRole, data)
        self._refresh_item_text(item)
        self._update_thumbnail_item(item)
        if self.queue_list.currentItem() is item:
            self._on_queue_selected(item, None)
        output_path = data.get("output_path", "")
        if output_path:
            lib_item = QtWidgets.QListWidgetItem(os.path.basename(output_path))
            lib_item.setData(QtCore.Qt.UserRole, {"path": output_path, "page": page_record})
            pixmap = self._load_thumbnail(output_path, self.library_list.iconSize())
            if pixmap:
                scaled = pixmap.scaled(
                    self.library_list.iconSize(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                lib_item.setIcon(QtGui.QIcon(scaled))
            lib_item.setSizeHint(QtCore.QSize(170, 230))
            self.library_list.addItem(lib_item)
        self._update_library_placeholder()

    def _compute_page_confidence(self, page_record: dict) -> tuple[float, bool]:
        regions = page_record.get("regions", [])
        scores = []
        needs_review = False
        for region in regions:
            flags = region.get("flags", {})
            if flags.get("needs_review"):
                needs_review = True
            conf = region.get("confidence", {})
            det = float(conf.get("det", 1.0))
            ocr = float(conf.get("ocr", 1.0))
            trans = float(conf.get("trans", 1.0))
            score = min(det, ocr, trans)
            scores.append(score)
        if not scores:
            return 1.0, needs_review
        avg = sum(scores) / max(1, len(scores))
        if avg < 0.7:
            needs_review = True
        return avg, needs_review

    def _on_queue_selected(self, current, _previous) -> None:
        item = current if isinstance(current, QtWidgets.QListWidgetItem) else self.queue_list.currentItem()
        if not item:
            if hasattr(self, "home_right_stack"):
                self.home_right_stack.setCurrentIndex(0)
            return
        data = item.data(QtCore.Qt.UserRole) or {}
        page = data.get("page") or {}
        filename = os.path.basename(data.get("path", "")) or item.text().split("\n")[0]
        regions = page.get("regions") or []
        self.inspector_title.setText(filename or "Selected page")
        self.inspector_table.setRowCount(0)
        rows = []
        for region in regions:
            ocr_text = str(region.get("ocr_text") or "").strip()
            translation = str(region.get("translation") or "").strip()
            if not ocr_text and not translation:
                continue
            rows.append((ocr_text, translation))
        if not rows:
            self.inspector_title.setText(f"{filename} (no OCR yet)")
        else:
            self.inspector_table.setRowCount(len(rows))
            for row_index, (ocr_text, translation) in enumerate(rows):
                det_item = QtWidgets.QTableWidgetItem(ocr_text)
                trans_item = QtWidgets.QTableWidgetItem(translation)
                det_item.setFlags(det_item.flags() & ~QtCore.Qt.ItemIsEditable)
                trans_item.setFlags(trans_item.flags() & ~QtCore.Qt.ItemIsEditable)
                det_item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
                trans_item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
                self.inspector_table.setItem(row_index, 0, det_item)
                self.inspector_table.setItem(row_index, 1, trans_item)
        if hasattr(self, "home_right_stack"):
            self.home_right_stack.setCurrentIndex(1)

    def _show_home_default_panel(self) -> None:
        if hasattr(self, "home_right_stack"):
            self.home_right_stack.setCurrentIndex(0)
        if hasattr(self, "queue_list"):
            self.queue_list.clearSelection()

    def _on_queue_table_selected(self) -> None:
        row = self.queue_table.currentRow()
        if row < 0:
            return
        item = self.queue_list.item(row)
        data = item.data(QtCore.Qt.UserRole) if item else {}
        path = str((data or {}).get("path", ""))
        status = str((data or {}).get("status", "--"))
        self.job_file.setText(os.path.basename(path) if path else "--")
        self.job_status.setText(status or "--")
        self.job_page.setText(str(row + 1))
        self.job_stage.setText(status.split()[0] if status else "--")
        self._job_selected_path = path

    def _on_library_selected(self) -> None:
        item = self.library_list.currentItem()
        data = item.data(QtCore.Qt.UserRole) if item else {}
        path = str((data or {}).get("path", ""))
        page = (data or {}).get("page") or {}
        self.library_file.setText(os.path.basename(path) if path else "--")
        self.library_pages.setText(str(page.get("index", "--")))
        size = os.path.getsize(path) if path and os.path.isfile(path) else 0
        self.library_size.setText(f"{size / 1024:.1f} KB" if size else "--")
        self._library_selected_path = path

    def _on_queue_remove_clicked(self) -> None:
        btn = self.sender()
        if not btn:
            return
        for row in range(self.queue_table.rowCount()):
            if self.queue_table.cellWidget(row, 2) is btn:
                QtWidgets.QMessageBox.information(
                    self,
                    "Remove",
                    "Removing items from the running queue isn't supported yet.",
                )
                return

    def _open_job_folder(self) -> None:
        self._open_containing_folder(getattr(self, "_job_selected_path", ""))

    def _open_library_folder(self) -> None:
        self._open_containing_folder(getattr(self, "_library_selected_path", ""))

    def _open_export_folder(self) -> None:
        path = self.export_dir.text().strip()
        if not path:
            path = self.settings_save_folder.text().strip()
        if path:
            self._open_containing_folder(path)

    def _open_containing_folder(self, path: str) -> None:
        if not path:
            return
        if os.path.isfile(path):
            path = os.path.dirname(path)
        if not os.path.isdir(path):
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    def _open_page_review(self, item: QtWidgets.QListWidgetItem) -> None:
        data = item.data(QtCore.Qt.UserRole) or {}
        page = data.get("page")
        if not page:
            return
        from app.ui.page_review import PageReviewDialog
        pipeline_idle = not self._running
        pipeline_block_reason = (
            ""
            if pipeline_idle
            else "Manual cleanup is unavailable while the forward pipeline is running."
        )
        dialog = PageReviewDialog(
            self,
            page_record=page,
            json_path=self.json_path.text().strip(),
            use_gpu=self.use_gpu.isChecked(),
            pipeline_idle=pipeline_idle,
            pipeline_block_reason=pipeline_block_reason,
        )
        dialog.exec()

    def _typed_model_overrides(self, backend: str) -> tuple[ModelGenerationOverride, ...]:
        prefix = f"{backend.lower()}::"
        overrides: list[ModelGenerationOverride] = []
        for key in sorted(self._model_overrides):
            if not key.lower().startswith(prefix):
                continue
            record = self._model_overrides.get(key) or {}
            if not isinstance(record, dict) or not bool(record.get("enabled")):
                continue
            values = record.get("values") or {}
            if not isinstance(values, dict):
                continue
            model_id = key[len(prefix):].strip()
            if not model_id:
                continue
            if backend == "Ollama":
                fields = {
                    "temperature": values.get("ollama_temp"),
                    "top_p": values.get("ollama_top_p"),
                    "ollama_context_tokens": values.get("ollama_ctx"),
                }
            else:
                fields = {
                    "temperature": values.get("gguf_temp"),
                    "top_p": values.get("gguf_top_p"),
                    "gguf_n_ctx": values.get("gguf_n_ctx"),
                    "gguf_n_gpu_layers": values.get("gguf_n_gpu_layers"),
                    "gguf_n_threads": values.get("gguf_n_threads"),
                    "gguf_n_batch": values.get("gguf_n_batch"),
                }
            if any(value is not None for value in fields.values()):
                overrides.append(
                    ModelGenerationOverride(model_id=model_id, **fields)
                )
                continue
            profile = (
                self._settings_model.draft.translation_profile
                if self._settings_model is not None
                else None
            )
            if profile is not None:
                previous = next(
                    (
                        item
                        for item in profile.model_overrides
                        if item.model_id == model_id
                    ),
                    None,
                )
                if previous is not None:
                    overrides.append(previous)
        return tuple(overrides)

    def _settings_projection(self) -> LegacyShellSettingsProjection:
        backend = self.settings_translator_backend.currentText()
        return LegacyShellSettingsProjection(
            source_language=self.source_lang.currentText(),
            target_language=self.target_lang.currentText(),
            output_suffix=self.output_suffix.text().strip() or self._defaults.output_suffix,
            glossary_reference=self.style_path.text().strip() or None,
            detector_engine=self.settings_detector_engine.currentText(),
            detector_input_size=int(
                self.settings_detector_input_size.currentText()
            ),
            ocr_engine=_normalize_ocr_engine_name(
                self.settings_ocr_engine.currentText()
            ),
            inpaint_mode=self.inpaint_mode.currentText(),
            inpaint_model_id=self.inpaint_model_id.text().strip(),
            font_detection=self.font_detection.currentText(),
            font_name=self.font_name.currentText().strip(),
            use_gpu=self.use_gpu.isChecked(),
            auto_glossary=self.auto_glossary.isChecked(),
            prescan_enabled=self.prescan_enabled.isChecked(),
            use_ollama_discovery=self.use_ollama_discovery.isChecked(),
            discovery_backend=self.settings_discovery_backend.currentText(),
            translation_backend=backend,
            ollama_model=self.settings_ollama_model.currentText().strip(),
            ollama_endpoint="http://127.0.0.1:11434",
            ollama_generation=GenerationSettings(
                temperature=float(self.settings_ollama_temp.value()),
                top_p=float(self.settings_ollama_top_p.value()),
            ),
            ollama_options=OllamaProviderOptions(
                context_tokens=int(self.settings_ollama_ctx.value()),
            ),
            gguf_model_path=self._settings_selected_gguf(),
            gguf_generation=GenerationSettings(
                temperature=float(self.settings_gguf_temp.value()),
                top_p=float(self.settings_gguf_top_p.value()),
            ),
            gguf_options=GGUFProviderOptions(
                prompt_style=self.settings_gguf_prompt_style.currentText(),
                n_ctx=int(self.settings_gguf_n_ctx.value()),
                n_gpu_layers=int(self.settings_gguf_n_gpu_layers.value()),
                n_threads=int(self.settings_gguf_n_threads.value()),
                n_batch=int(self.settings_gguf_n_batch.value()),
            ),
            deepseek_model=(
                self.settings_deepseek_model.text().strip()
                or self._defaults.deepseek_model
            ),
            deepseek_endpoint=(
                self.settings_deepseek_base_url.text().strip()
                or self._defaults.deepseek_base_url
            ),
            translation_overrides=(
                self._typed_model_overrides(backend)
                if backend in {"Ollama", "GGUF"}
                else ()
            ),
            discovery_ollama_model=self.settings_discovery_ollama_model.currentText().strip(),
            discovery_gguf_model_path=(
                self.settings_discovery_gguf_path.currentData()
                or self.settings_discovery_gguf_path.currentText().strip()
            ),
        )

    def _capture_settings_draft(self) -> SettingsDraft:
        if self._settings_model is None:
            raise RuntimeError("typed settings authority is unavailable")
        draft = self._settings_model.apply()
        self._persist_public_settings()
        return draft

    def _project_id_for_run(self, json_path: str) -> str:
        if json_path and os.path.isfile(json_path):
            try:
                project = load_project_for_editing(json_path)
                project_id = str((project.get("project") or {}).get("project_id") or "")
                if project_id:
                    return project_id
            except Exception as exc:
                logger.warning("Could not read typed project identity: %s", exc)
        normalized = os.path.normcase(os.path.abspath(json_path or "project.json"))
        return f"project:gui:{canonical_fingerprint({'json_path': normalized})[:32]}"

    def _compile_pipeline_settings(
        self,
        whitelist: list[str] | None = None,
    ) -> tuple[PipelineSettings, CompilationResult]:
        if not self.json_path.text().strip() and self.export_dir.text().strip():
            self.json_path.setText(
                os.path.join(self.export_dir.text().strip(), "project.json")
            )
        self._capture_settings_draft()
        if self._settings_model is None:
            raise RuntimeError("typed settings authority is unavailable")
        json_path = self.json_path.text().strip()
        project_id = self._project_id_for_run(json_path)
        result = self._settings_model.preview_run(
            project_id=project_id,
            invocation=RunInvocation(
                import_dir=self.import_dir.text().strip(),
                export_dir=self.export_dir.text().strip(),
                json_path=json_path,
                files_whitelist=tuple(whitelist or ()),
            ),
        )
        settings = materialize_pipeline_settings(result)
        return settings, result

    def _get_pipeline_settings(
        self,
        whitelist: list[str] | None = None,
    ) -> PipelineSettings:
        """Compatibility accessor backed exclusively by the typed compiler."""

        settings, _ = self._compile_pipeline_settings(whitelist)
        return settings

    def _start_pipeline(self, whitelist: list[str] = None) -> bool:
        logger.info(f"Attempting to start pipeline. Whitelist: {whitelist}")
        runtime_checker = ModelDownloader(self)
        if not runtime_checker.check_pyicu_runtime():
            runtime_error = runtime_checker.pyicu_runtime_error or (
                "PyICU 2.16.2 with ICU 78.3 is required before rendering can start."
            )
            self._set_pyicu_runtime_ready(False, runtime_error)
            QtWidgets.QMessageBox.warning(
                self,
                "Required Runtime Unavailable",
                runtime_error,
            )
            return False
        self._set_pyicu_runtime_ready(True)
        if not self.import_dir.text().strip():
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Manga Folder")
            if not path:
                return False
            self.import_dir.setText(path)
        
        try:
            settings, compilation = self._compile_pipeline_settings(whitelist)
            runtime_binding = self._resolve_pipeline_runtime_binding(compilation)
        except (RuntimeError, TypeError, ValueError) as exc:
            message = str(exc)
            self.status_bar.showMessage(message, 8000)
            if self.log_view:
                self.log_view.appendPlainText(message)
            QtWidgets.QMessageBox.warning(
                self,
                "Run Settings Error",
                message,
            )
            return False
        started = self._pipeline.start(
            settings,
            runtime_binding=runtime_binding,
        )
        if not started:
            self._clear_pending_run_settings()
            return False
        self._pending_run_snapshot = compilation.snapshot
        self._pending_project_id = compilation.snapshot.project_id
        self._pending_run_json_path = os.path.normcase(
            os.path.abspath(settings.json_path)
        )
        return True

    def _resolve_pipeline_runtime_binding(
        self,
        compilation: CompilationResult,
    ) -> PipelineRuntimeBinding:
        """Resolve an opaque credential reference only for this Start call."""

        binding = compilation.runtime_binding
        reference = binding.credential_reference
        resolved_credential: str | None = None
        if reference is not None:
            resolver = CompositeCredentialResolver(
                environment=EnvironmentCredentialResolver(),
                windows=WindowsCredentialStore(),
            )
            resolved_credential = resolver.resolve(reference)
            if not resolved_credential:
                raise RuntimeError(
                    "The selected provider credential is unavailable. Relink it in "
                    "Settings; DeepSeek migration uses the DEEPSEEK_API_KEY "
                    "environment reference and does not read legacy key files."
                )
        if binding.provider_kind is ProviderKind.DEEPSEEK and not resolved_credential:
            raise RuntimeError(
                "DeepSeek requires a resolved credential reference before Start."
            )
        return PipelineRuntimeBinding(
            provider_kind=binding.provider_kind,
            resolved_credential=resolved_credential,
        )

    def _on_consistency_issue(self, pages: list[int]) -> None:
        """Handle end-of-run consistency notification."""
        from app.ui.dialogs.consistency_dialog import ConsistencyDialog
        
        # Convert indices to filenames for display if possible, or just pass indices
        # The dialog expects filenames (str) but we are passing indices (int)?
        # Let's check ConsistencyDialog.__init__
        # It takes 'filenames: list[str]'.
        
        # We need to get filenames.
        filenames = []
        if self.queue_list.count() > 0:
            for idx in pages:
                if idx < self.queue_list.count():
                    item = self.queue_list.item(idx)
                    data = item.data(QtCore.Qt.UserRole) or {}
                    path = data.get("path", "")
                    if path:
                        filenames.append(os.path.basename(path))
        else:
            import_dir = self.import_dir.text().strip()
            if import_dir and os.path.isdir(import_dir):
                allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                names = []
                for entry in os.listdir(import_dir):
                    _, ext = os.path.splitext(entry)
                    if ext.lower() in allowed:
                        names.append(entry)
                names.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)])
                for idx in pages:
                    if idx < len(names):
                        filenames.append(names[idx])
            if not filenames:
                filenames = [f"Page {i+1}" for i in pages]

        while True:
            dialog = ConsistencyDialog(filenames, self)
            result = dialog.exec()
            
            if result == QtWidgets.QDialog.Accepted:
                # Re-translate selected pages
                self._start_pipeline(whitelist=filenames)
                break
            elif result == 100:
                # Deep Scan Requested
                self.status_bar.showMessage("Starting Deep Scan (LLM)...")
                settings = self._get_pipeline_settings()
                self._pipeline.start_deep_scan(settings)
                
                # Show blocking progress dialog
                progress = QtWidgets.QProgressDialog("Scanning text for entities...", "Cancel", 0, 0, self)
                progress.setWindowModality(QtCore.Qt.WindowModal)
                progress.setMinimumDuration(0)
                
                worker = self._pipeline.deep_scan_worker
                worker.finished.connect(progress.accept)
                
                progress.exec()
                
                if progress.wasCanceled():
                    break
                    
                QtWidgets.QMessageBox.information(self, "Deep Scan Complete", "Glossary has been updated.\nYou can now choose to re-translate the pages.")
                continue # Loop back to show dialog again
            else:
                break



    def _open_region_review(self) -> None:
        if not self._review_dialog:
            self._review_dialog = RegionReviewDialog(self)
        path = self.json_path.text().strip()
        if path:
            self._review_dialog.set_path(path)
        self._review_dialog.exec()

    def _handle_message(self, message: str) -> None:
        self.status_bar.showMessage(message)
        if self.log_view:
            self.log_view.appendPlainText(message)
        if message in {"Completed", "Stopped"}:
            self._set_running(False)
            if message == "Completed":
                self._last_run_snapshot = self._pending_run_snapshot
                self._active_project_id = self._pending_project_id
                self._active_run_json_path = self._pending_run_json_path
                try:
                    self._persist_project_settings_if_idle(completed_run=True)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    logger.warning("Typed project settings were not published: %s", exc)
            self._clear_pending_run_settings()
        if message.startswith("Failed") or "required" in message:
            self._set_running(False)
        if (
            "PaddleOCR-VL failed" in message
            or "PaddleOCR-VL runtime failed" in message
            or "Failed to initialize models" in message
            or "NumPy ABI mismatch" in message
            or "PyTorch DLL load failed" in message
        ):
            QtWidgets.QMessageBox.critical(self, "Dependency Error", message)

    def _clear_pending_run_settings(self) -> None:
        self._pending_run_snapshot = None
        self._pending_project_id = None
        self._pending_run_json_path = None

    def closeEvent(self, event) -> None:
        self._save_settings()
        super().closeEvent(event)

    def _save_settings(self) -> None:
        # Window geometry remains application-shell state.  Workflow settings
        # are persisted only through the typed stores below.
        settings = QtCore.QSettings("MangaTranslator", "Pro")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        if self._settings_model is not None:
            try:
                self._capture_settings_draft()
                self._persist_project_settings_if_idle()
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning("Typed settings were not saved: %s", exc)

    @staticmethod
    def _module_configs_for_scope(
        configs: tuple[ModuleConfig, ...],
        scope: SettingsScope,
    ) -> tuple[ModuleConfig, ...]:
        selected: list[ModuleConfig] = []
        for config in configs:
            module = DEFAULT_MODULE_REGISTRY.get_module(config.module_id)
            values = {
                key: value
                for key, value in config.values.items()
                if module.definitions[key].scope is scope
            }
            legacy_values = {
                key: value
                for key, value in config.legacy_values.items()
                if key in module.definitions
                and module.definitions[key].scope is scope
            }
            if values or legacy_values:
                selected.append(
                    ModuleConfig(
                        module_id=config.module_id,
                        module_schema_version=config.module_schema_version,
                        values=values,
                        legacy_values=legacy_values,
                    )
                )
        return tuple(sorted(selected, key=lambda item: item.module_id))

    def _settings_storage_directory(self) -> Path:
        return qt_platform_paths().config_root

    def _initialize_settings_authority(self) -> None:
        root = self._settings_storage_directory()
        self._application_settings_store = ApplicationSettingsStore(
            root / "settings.json"
        )
        self._provider_profile_store = ProviderProfileStore(
            root / "provider_profiles.json"
        )
        application_document = self._application_settings_store.load()
        profiles = self._provider_profile_store.load()

        legacy_settings = QtCore.QSettings("MangaTranslator", "Pro")
        migration = migrate_legacy_qsettings_once(
            legacy_settings,
            application_document.migration_markers,
        )
        invocation_defaults = LegacyRunInvocationDefaults()
        project = ProjectConfig()
        project_modules: tuple[ModuleConfig, ...] = ()
        last_snapshot: RunSettingsSnapshot | None = None
        marker_document: ApplicationSettingsDocument | None = None
        migration_has_project_payload = False

        if migration is not None:
            invocation_defaults = migration.run_invocation_defaults
            recent_projects = application_document.application_preferences.recent_projects
            if invocation_defaults.json_path:
                recent_projects = tuple(
                    dict.fromkeys(
                        (*recent_projects, invocation_defaults.json_path)
                    )
                )
            workspace_layout = dict(
                migration.application_preferences.workspace_layout
            )
            workspace_layout.update(
                application_document.application_preferences.workspace_layout
            )
            migrated_application_modules = {
                config.module_id: config
                for config in self._module_configs_for_scope(
                    migration.module_configs,
                    SettingsScope.APPLICATION,
                )
            }
            migrated_application_modules.update(
                {
                    config.module_id: config
                    for config in application_document.application_module_configs
                }
            )
            evidence_by_source = {
                item.source_fingerprint: item
                for item in application_document.legacy_migration_evidence
            }
            evidence = self._inactive_migration_evidence(migration)
            evidence_by_source[evidence.source_fingerprint] = evidence
            application_document = ApplicationSettingsDocument(
                application_preferences=replace(
                    application_document.application_preferences,
                    recent_projects=recent_projects,
                    workspace_layout=workspace_layout,
                ),
                application_module_configs=tuple(
                    migrated_application_modules[key]
                    for key in sorted(migrated_application_modules)
                ),
                migration_markers=application_document.migration_markers,
                legacy_migration_evidence=tuple(
                    evidence_by_source[key] for key in sorted(evidence_by_source)
                ),
            )
            marker_document = replace(
                application_document,
                migration_markers=(
                    *application_document.migration_markers,
                    migration.migration_marker,
                ),
            )
            project = migration.project_config
            project_modules = self._module_configs_for_scope(
                migration.module_configs,
                SettingsScope.PROJECT,
            )
            by_id = {profile.profile_id: profile for profile in profiles}
            for profile in migration.provider_profiles:
                by_id.setdefault(profile.profile_id, profile)
            profiles = tuple(by_id[key] for key in sorted(by_id))
            migration_has_project_payload = bool(
                project != ProjectConfig()
                or project_modules
                or migration.provider_profiles
            )

        candidate_path = invocation_defaults.json_path
        if not candidate_path:
            for recent in reversed(
                application_document.application_preferences.recent_projects
            ):
                if os.path.isfile(recent):
                    candidate_path = recent
                    break
        candidate_loaded = False
        project_publication = None
        project_publication_deferred = False
        if candidate_path and os.path.isfile(candidate_path):
            try:
                checkpoint_descriptor_active = project_storage_is_checkpoint_descriptor(
                    candidate_path
                )
                raw_project = load_project(candidate_path)
                stored_project = load_project_for_editing(candidate_path)
                stored_settings = read_project_settings(stored_project)
                migration_seed_required = bool(
                    migration is not None
                    and legacy_project_settings_seed_required(raw_project)
                )
                if migration_seed_required and checkpoint_descriptor_active:
                    # The checkpoint descriptor remains controller-owned.  Keep
                    # the migrated project settings in the current draft, but
                    # publish neither project/profile state nor the one-time
                    # marker until a later startup sees a finalized project.
                    project_publication_deferred = True
                    logger.info(
                        "Deferred legacy project settings migration while a "
                        "forward checkpoint descriptor is active."
                    )
                elif migration_seed_required:
                    migrated_project = with_project_settings(
                        stored_project,
                        project_config=migration.project_config,
                        module_configs=self._module_configs_for_scope(
                            migration.module_configs,
                            SettingsScope.PROJECT,
                        ),
                        last_run_snapshot=stored_settings.last_run_snapshot,
                    )
                    project_publication = lambda: save_project_schema_v2_atomic(
                        candidate_path,
                        migrated_project,
                        defer_if_checkpoint=True,
                    )
                else:
                    project = stored_settings.project_config
                    project_modules = stored_settings.module_configs
                last_snapshot = stored_settings.last_run_snapshot
                self._active_project_id = str(
                    (stored_project.get("project") or {}).get("project_id") or ""
                ) or None
                self._active_run_json_path = os.path.normcase(
                    os.path.abspath(candidate_path)
                )
                self.json_path.setText(candidate_path)
                self.export_dir.setText(os.path.dirname(candidate_path))
                pages = stored_project.get("pages") or []
                if isinstance(pages, list) and pages:
                    image_path = str((pages[0] or {}).get("image_path") or "")
                    if image_path:
                        self.import_dir.setText(os.path.dirname(image_path))
                candidate_loaded = True
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning("Could not hydrate typed project settings: %s", exc)
        elif not invocation_defaults.is_empty:
            self.import_dir.setText(invocation_defaults.import_dir)
            self.export_dir.setText(invocation_defaults.export_dir)
            self.json_path.setText(invocation_defaults.json_path)

        if migration is not None and marker_document is not None:
            migration_can_commit = (
                not migration_has_project_payload
                or (candidate_loaded and not project_publication_deferred)
            )
            if migration_can_commit:
                migration_published = publish_legacy_migration_marker_last(
                    publish_project=project_publication,
                    publish_provider_profiles=lambda: self._provider_profile_store.save(
                        profiles
                    ),
                    publish_application_marker=lambda: self._application_settings_store.save(
                        marker_document
                    ),
                )
                if migration_published:
                    application_document = marker_document
                else:
                    project_publication_deferred = True
                    logger.info(
                        "Deferred legacy project settings migration because a "
                        "forward checkpoint became active before publication."
                    )

        module_by_id = {
            config.module_id: config
            for config in application_document.application_module_configs
        }
        module_by_id.update({config.module_id: config for config in project_modules})
        initial = SettingsDraft(
            application=application_document.application_preferences,
            project=project,
            module_configs=tuple(
                module_by_id[key] for key in sorted(module_by_id)
            ),
            provider_profiles=profiles,
        )
        self._application_settings_document = application_document
        self._last_run_snapshot = last_snapshot
        self._settings_model = SettingsViewModel(initial)
        self._apply_settings_draft_to_shell(initial)
        self._refresh_effective_run_summary()

        geometry = legacy_settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        window_state = legacy_settings.value("windowState")
        if window_state is not None:
            self.restoreState(window_state)

    @staticmethod
    def _inactive_migration_evidence(
        migration: LegacySettingsMigrationResult,
    ) -> InactiveLegacyMigrationEvidence:
        return InactiveLegacyMigrationEvidence(
            migration_version=migration.migration_version,
            source_fingerprint=migration.source_fingerprint,
            legacy_values=migration.legacy_values,
            issues=tuple(
                LegacyMigrationIssueEvidence(
                    key=issue.key,
                    reason=issue.reason,
                )
                for issue in migration.issues
            ),
            unresolved_provider_profile_references=(
                migration.unresolved_provider_profile_references
            ),
        )

    def _apply_settings_draft_to_shell(self, draft: SettingsDraft) -> None:
        previous_guard = self._settings_projection_guard
        self._settings_projection_guard = True
        try:
            self._apply_settings_draft_to_widgets(draft)
        finally:
            self._settings_projection_guard = previous_guard

    def _apply_settings_draft_to_widgets(self, draft: SettingsDraft) -> None:
        self.theme_combo.setCurrentText(draft.application.theme)
        self.source_lang.setCurrentText(draft.project.source_language)
        self.target_lang.setCurrentText(draft.project.target_language)
        self.output_suffix.setText(draft.project.output_suffix)
        self.style_path.setText(draft.project.glossary_reference or "")
        module_values = {
            config.module_id: dict(config.values)
            for config in draft.module_configs
        }
        detection = module_values.get("detection", {})
        self.detector_engine.setCurrentText(
            str(detection.get("engine", self._defaults.detector_engine))
        )
        self.detector_input_size.setCurrentText(
            str(detection.get("input_size", 640))
        )
        self.ocr_engine.setCurrentText(
            str(module_values.get("ocr", {}).get("engine", self._defaults.ocr_engine))
        )
        cleanup = module_values.get("cleanup", {})
        self.inpaint_mode.setCurrentText(
            str(cleanup.get("inpaint_mode", self._defaults.inpaint_mode))
        )
        self.font_detection.setCurrentText(
            str(
                module_values.get("source_style", {}).get(
                    "font_detection", self._defaults.font_detection
                )
            )
        )
        self._set_combo_text(
            self.font_name,
            str(
                module_values.get("renderer", {}).get(
                    "font_name", self._defaults.font_name
                )
            ),
        )
        self.use_gpu.setChecked(
            bool(module_values.get("runtime", {}).get("use_gpu", True))
        )
        translation = module_values.get("translation", {})
        self.auto_glossary.setChecked(
            bool(translation.get("auto_glossary", self._defaults.auto_glossary))
        )
        self.prescan_enabled.setChecked(
            bool(translation.get("prescan_enabled", False))
        )
        self.use_ollama_discovery.setChecked(
            bool(translation.get("use_ollama_discovery", False))
        )
        self.settings_discovery_backend.setCurrentText(
            str(translation.get("discovery_backend", "Ollama"))
        )
        profile = draft.translation_profile
        if profile is not None:
            backend = {
                ProviderKind.OLLAMA: "Ollama",
                ProviderKind.GGUF: "GGUF",
                ProviderKind.DEEPSEEK: "DeepSeek",
                ProviderKind.OPENAI_COMPATIBLE: "OpenAI-compatible",
            }[profile.kind]
            if backend != "OpenAI-compatible":
                self.translator_backend.setCurrentText(backend)
            generation = profile.generation_defaults
            if profile.kind is ProviderKind.OLLAMA:
                self._set_combo_text(self.ollama_model, profile.model_id or "auto-detect")
                self.settings_ollama_temp.setValue(generation.temperature)
                self.settings_ollama_top_p.setValue(generation.top_p)
                if profile.ollama_options is not None:
                    self.settings_ollama_ctx.setValue(
                        profile.ollama_options.context_tokens
                    )
            elif profile.kind is ProviderKind.GGUF:
                if profile.local_model_path:
                    self._add_gguf_model(profile.local_model_path)
                self.settings_gguf_temp.setValue(generation.temperature)
                self.settings_gguf_top_p.setValue(generation.top_p)
                if profile.gguf_options is not None:
                    self.gguf_prompt_style.setCurrentText(
                        profile.gguf_options.prompt_style
                    )
                    self.gguf_n_ctx.setValue(profile.gguf_options.n_ctx)
                    self.gguf_n_gpu_layers.setValue(
                        profile.gguf_options.n_gpu_layers
                    )
                    self.gguf_n_threads.setValue(profile.gguf_options.n_threads)
                    self.gguf_n_batch.setValue(profile.gguf_options.n_batch)
            elif profile.kind is ProviderKind.DEEPSEEK:
                self.settings_deepseek_model.setText(profile.model_id or "")
                self.settings_deepseek_base_url.setText(profile.endpoint or "")
            self._model_overrides = {}
            for override in profile.model_overrides:
                values: dict[str, object] = {}
                if profile.kind is ProviderKind.OLLAMA:
                    values = {
                        key: value
                        for key, value in {
                            "ollama_temp": override.temperature,
                            "ollama_top_p": override.top_p,
                            "ollama_ctx": override.ollama_context_tokens,
                        }.items()
                        if value is not None
                    }
                    key = f"ollama::{override.model_id}"
                elif profile.kind is ProviderKind.GGUF:
                    values = {
                        key: value
                        for key, value in {
                            "gguf_temp": override.temperature,
                            "gguf_top_p": override.top_p,
                            "gguf_n_ctx": override.gguf_n_ctx,
                            "gguf_n_gpu_layers": override.gguf_n_gpu_layers,
                            "gguf_n_threads": override.gguf_n_threads,
                            "gguf_n_batch": override.gguf_n_batch,
                        }.items()
                        if value is not None
                    }
                    key = f"gguf::{override.model_id}"
                else:
                    continue
                self._model_overrides[key] = {"enabled": True, "values": values}
        discovery_profile = draft.discovery_profile
        if discovery_profile is not None:
            if discovery_profile.kind is ProviderKind.OLLAMA:
                self._set_combo_text(
                    self.settings_discovery_ollama_model,
                    discovery_profile.model_id or "auto-detect",
                )
            elif (
                discovery_profile.kind is ProviderKind.GGUF
                and discovery_profile.local_model_path
            ):
                self._set_gguf_combo(
                    self.settings_discovery_gguf_path,
                    discovery_profile.local_model_path,
                )
        self._sync_models_to_settings()
        self._sync_paths_to_settings()
        self._update_glossary_controls()
        self._update_discovery_ui()
        self._refresh_override_model_list()
        self._update_translation_warning()
        self._update_scan_warnings()

    def _persist_public_settings(self) -> None:
        if (
            self._settings_model is None
            or self._application_settings_store is None
            or self._provider_profile_store is None
        ):
            return
        draft = self._settings_model.draft
        recent_projects = list(draft.application.recent_projects)
        project_path = self.json_path.text().strip()
        if project_path:
            recent_projects = [
                item for item in recent_projects if item != project_path
            ] + [project_path]
        application = replace(
            draft.application,
            recent_projects=tuple(recent_projects[-20:]),
        )
        self._settings_model.replace_application(application)
        self._settings_model.apply()
        document = ApplicationSettingsDocument(
            application_preferences=application,
            application_module_configs=self._module_configs_for_scope(
                self._settings_model.draft.module_configs,
                SettingsScope.APPLICATION,
            ),
            migration_markers=self._application_settings_document.migration_markers,
            legacy_migration_evidence=(
                self._application_settings_document.legacy_migration_evidence
            ),
        )
        self._application_settings_store.save(document)
        self._provider_profile_store.save(
            self._settings_model.draft.provider_profiles
        )
        self._application_settings_document = document

    def _persist_project_settings_if_idle(
        self,
        *,
        completed_run: bool = False,
    ) -> None:
        if self._running or self._settings_model is None:
            return
        path = self.json_path.text().strip()
        if not path or not os.path.isfile(path):
            return
        normalized_path = os.path.normcase(os.path.abspath(path))
        raw = load_project(path)
        if str(raw.get("schema_version") or "1") == "1":
            project = migrate_project_schema_v2(
                raw,
                project_id=self._active_project_id,
            )
        else:
            project = load_project_for_editing(path)
        project_id = str((project.get("project") or {}).get("project_id") or "")
        snapshot = self._last_run_snapshot
        if snapshot is not None:
            if normalized_path != self._active_run_json_path:
                snapshot = None
            elif snapshot.project_id != project_id:
                if completed_run:
                    snapshot = rebind_run_snapshot_project(snapshot, project_id)
                    self._last_run_snapshot = snapshot
                    self._active_project_id = project_id
                else:
                    snapshot = None
        updated = with_project_settings(
            project,
            project_config=self._settings_model.draft.project,
            module_configs=self._module_configs_for_scope(
                self._settings_model.draft.module_configs,
                SettingsScope.PROJECT,
            ),
            last_run_snapshot=snapshot,
        )
        save_project_schema_v2_atomic(path, updated)
