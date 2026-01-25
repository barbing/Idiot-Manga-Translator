# -*- coding: utf-8 -*-
"""Main window UI."""
import os
from PySide6 import QtCore, QtGui, QtWidgets
from app.config.defaults import get_defaults
from app.pipeline.controller import PipelineController, PipelineSettings
from app.models.ollama import list_models
from app.ui.theme import apply_dark_palette, apply_light_palette
from app.io.project import load_project
from app.render.renderer import render_translations
from app.ui.style_guide_editor import StyleGuideEditor
from app.ui.region_review import RegionReviewDialog


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Manga Translator Pro")
        self.resize(1200, 720)

        self._defaults = get_defaults()
        self._pipeline = PipelineController(self)
        self._style_editor: StyleGuideEditor | None = None
        self._review_dialog: RegionReviewDialog | None = None
        self._page_review: QtWidgets.QDialog | None = None
        self._running = False
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
        self._setup_ui()
        self._connect_signals()
        self._apply_theme(self._defaults.theme)

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
        brand_title = QtWidgets.QLabel("MT Pro")
        brand_title.setStyleSheet("QLabel { font-size: 18px; font-weight: 600; }")
        if icons.get("brand"):
            brand_icon.setPixmap(icons["brand"].pixmap(18, 18))
        brand.addWidget(brand_icon)
        brand.addWidget(brand_title)
        brand.addStretch(1)
        layout.addLayout(brand)

        self.nav_home = QtWidgets.QToolButton()
        self.nav_queue = QtWidgets.QToolButton()
        self.nav_library = QtWidgets.QToolButton()
        self.nav_settings = QtWidgets.QToolButton()
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
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            btn.setAutoRaise(True)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setIconSize(QtCore.QSize(18, 18))
            btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            btn.setFixedHeight(40)
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
        self._start_style = "QPushButton { background-color: #3b82f6; border-color: #2563eb; color: #f8fbff; }"
        self._stop_style = "QPushButton { background-color: #1f2937; border-color: #374151; }"
        self.start_btn.setStyleSheet(self._start_style)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setFixedHeight(34)
        self.stop_btn.setStyleSheet(self._stop_style)
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
        vbox.addWidget(self.settings_models_group)
        vbox.addStretch(1)
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
        self.auto_glossary = QtWidgets.QCheckBox("Auto glossary\n(consistent terms)")
        self.auto_glossary.setChecked(self._defaults.auto_glossary)
        self.auto_glossary.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        layout.addRow("", self.style_edit)
        layout.addRow("", self.auto_glossary)
        return group

    def _group_models_main(self) -> QtWidgets.QGroupBox:
        if hasattr(self, "_models_group"):
            return self._models_group
        group = QtWidgets.QGroupBox("Advanced Model Settings")
        layout = QtWidgets.QGridLayout(group)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        self.detector_engine = QtWidgets.QComboBox()
        self.detector_engine.addItems(["PaddleOCR", "ComicTextDetector"])
        self.detector_engine.setCurrentText(self._defaults.detector_engine)
        
        self.detector_input_size = QtWidgets.QComboBox()
        self.detector_input_size.addItems(["640", "1024", "1280"])
        self.detector_input_size.setCurrentText("640")
        
        det_layout = self._hbox(self.detector_engine, self.detector_input_size)
    
        layout.addWidget(QtWidgets.QLabel("Text Detector"), 0, 0)
        layout.addWidget(det_layout, 1, 0)
        self.ocr_engine = QtWidgets.QComboBox()
        self.ocr_engine.addItems(["PaddleOCR", "MangaOCR"])
        self.ocr_engine.setCurrentText(self._defaults.ocr_engine)
        layout.addWidget(QtWidgets.QLabel("OCR Engine"), 0, 1)
        layout.addWidget(self.ocr_engine, 1, 1)
        self.translator_backend = QtWidgets.QComboBox()
        self.translator_backend.addItems(["Ollama", "GGUF"])
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
        group = QtWidgets.QGroupBox("Models")
        layout = QtWidgets.QGridLayout(group)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        self.settings_detector_engine = QtWidgets.QComboBox()
        self.settings_detector_engine.addItems(["PaddleOCR", "ComicTextDetector"])
        self.settings_ocr_engine = QtWidgets.QComboBox()
        self.settings_ocr_engine.addItems(["PaddleOCR", "MangaOCR"])
        self.settings_translator_backend = QtWidgets.QComboBox()
        self.settings_translator_backend.addItems(["Ollama", "GGUF"])
        self.settings_ollama_model = QtWidgets.QComboBox()
        self.settings_ollama_model.addItems(["auto-detect"])
        self.settings_model_refresh = QtWidgets.QPushButton("Refresh")
        settings_model_row = self._hbox(self.settings_ollama_model, self.settings_model_refresh)
        self.settings_gguf_model_path = QtWidgets.QComboBox()
        self.settings_gguf_model_path.setEditable(True)
        self.settings_gguf_model_path.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.settings_gguf_browse = QtWidgets.QPushButton("Browse")
        self.settings_gguf_prompt_style = QtWidgets.QComboBox()
        self.settings_gguf_prompt_style.addItems(["sakura", "qwen", "plain"])
        self.settings_gguf_n_ctx = QtWidgets.QSpinBox()
        self.settings_gguf_n_ctx.setRange(512, 32768)
        self.settings_gguf_n_gpu_layers = QtWidgets.QSpinBox()
        self.settings_gguf_n_gpu_layers.setRange(-1, 200)
        self.settings_gguf_n_threads = QtWidgets.QSpinBox()
        self.settings_gguf_n_threads.setRange(1, 128)
        self.settings_gguf_n_batch = QtWidgets.QSpinBox()
        self.settings_gguf_n_batch.setRange(64, 4096)
        self.settings_detector_engine.setCurrentText(self._defaults.detector_engine)
        self.settings_ocr_engine.setCurrentText(self._defaults.ocr_engine)
        self.settings_translator_backend.setCurrentText(self._defaults.translator_backend)
        self.settings_gguf_prompt_style.setCurrentText(self._defaults.gguf_prompt_style)
        self.settings_gguf_n_ctx.setValue(self._defaults.gguf_n_ctx)
        self.settings_gguf_n_gpu_layers.setValue(self._defaults.gguf_n_gpu_layers)
        self.settings_gguf_n_threads.setValue(self._defaults.gguf_n_threads)
        self.settings_gguf_n_batch.setValue(self._defaults.gguf_n_batch)

        layout.addWidget(QtWidgets.QLabel("Text Detector"), 0, 0)
        
        self.settings_detector_input_size = QtWidgets.QComboBox()
        self.settings_detector_input_size.addItems(["640", "1024", "1280"])
        self.settings_detector_input_size.setCurrentText("640")
        
        det_row = self._hbox(self.settings_detector_engine, self.settings_detector_input_size)
        layout.addWidget(det_row, 1, 0)
        layout.addWidget(QtWidgets.QLabel("OCR Engine"), 0, 1)
        layout.addWidget(self.settings_ocr_engine, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Translator"), 2, 0)
        layout.addWidget(self.settings_translator_backend, 3, 0)
        layout.addWidget(QtWidgets.QLabel("Ollama Model"), 2, 1)
        layout.addWidget(settings_model_row, 3, 1)
        layout.addWidget(QtWidgets.QLabel("GGUF Model"), 4, 0, 1, 2)
        layout.addWidget(self._hbox(self.settings_gguf_model_path, self.settings_gguf_browse), 5, 0, 1, 2)
        layout.addWidget(QtWidgets.QLabel("GGUF Prompt"), 6, 0, 1, 2)
        layout.addWidget(self.settings_gguf_prompt_style, 7, 0, 1, 2)
        layout.addWidget(QtWidgets.QLabel("GGUF Context"), 8, 0)
        layout.addWidget(self.settings_gguf_n_ctx, 9, 0)
        layout.addWidget(QtWidgets.QLabel("GGUF GPU Layers"), 8, 1)
        layout.addWidget(self.settings_gguf_n_gpu_layers, 9, 1)
        layout.addWidget(QtWidgets.QLabel("GGUF Threads"), 10, 0)
        layout.addWidget(self.settings_gguf_n_threads, 11, 0)
        layout.addWidget(QtWidgets.QLabel("GGUF Batch"), 10, 1)
        layout.addWidget(self.settings_gguf_n_batch, 11, 1)
        return group

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
        self.font_detection.addItems(["off", "heuristic"])
        self.font_detection.setCurrentText(self._defaults.font_detection)
        layout.addRow("Font Detection", self.font_detection)
        self.filter_background = QtWidgets.QCheckBox("Ignore background/onoma text")
        self.filter_background.setChecked(True)
        self.filter_strength = QtWidgets.QComboBox()
        self.filter_strength.addItems(["normal", "aggressive"])
        self.filter_strength.setCurrentText(self._defaults.filter_strength)
        layout.addRow("", self.filter_background)
        layout.addRow("Filter Strength", self.filter_strength)
        self.inpaint_mode = QtWidgets.QComboBox()
        self.inpaint_mode.addItems(["fast", "ai", "off"])
        self.inpaint_mode.setCurrentText(self._defaults.inpaint_mode)
        layout.addRow("Inpainting", self.inpaint_mode)
        
        # New: AI Model ID - read from defaults
        self.inpaint_model_id = QtWidgets.QLineEdit(self._defaults.inpaint_model)
        layout.addRow("AI Model ID", self.inpaint_model_id)
        
        return group


    def _group_performance(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Performance")
        layout = QtWidgets.QFormLayout(group)
        self.use_gpu = QtWidgets.QCheckBox("Enable GPU when available")
        self.use_gpu.setChecked(True)
        self.fast_mode = QtWidgets.QCheckBox("Fast Mode (detector/inpaint only)")
        self.fast_mode.setChecked(self._defaults.fast_mode)
        layout.addRow("", self.use_gpu)
        layout.addRow("", self.fast_mode)
        return group

    def _group_theme(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Theme")
        layout = QtWidgets.QFormLayout(group)
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.setCurrentText(self._defaults.theme)
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
        prev = self._active_page_index
        if prev == 3:
            self._sync_settings_to_models()
            self._sync_paths_from_settings()
        if index == 3:
            self._sync_models_to_settings()
            self._sync_paths_to_settings()
        self._active_page_index = index
        self.center_stack.setCurrentIndex(index)
        self.right_stack.setCurrentIndex(index)

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
        normalized = os.path.abspath(path)
        for idx in range(combo.count()):
            data = combo.itemData(idx)
            if isinstance(data, str) and os.path.abspath(data) == normalized:
                combo.setCurrentIndex(idx)
                return
        combo.addItem(os.path.basename(normalized), normalized)
        combo.setCurrentIndex(combo.count() - 1)

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
        self.style_browse.clicked.connect(self._choose_style_path)
        self.style_edit.clicked.connect(self._open_style_editor)
        self.gguf_browse.clicked.connect(self._choose_gguf_model)
        self.settings_gguf_browse.clicked.connect(self._choose_gguf_model)

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
        self.queue_list.itemDoubleClicked.connect(self._open_page_review)
        self.queue_list.verticalScrollBar().valueChanged.connect(self._update_visible_thumbnails)
        self.import_dir.textChanged.connect(self._schedule_import_preview)

        self._refresh_models()
        self._refresh_gguf_models()
        self._refresh_import_preview()
        self._sync_models_to_settings()
        self._sync_paths_to_settings()
        self._load_saved_settings()

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

    def _choose_style_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Style Guide", filter="JSON Files (*.json)")
        if path:
            self.style_path.setText(path)

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
        names.sort()
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

    def _add_gguf_model(self, path: str) -> None:
        if not path:
            return
        path = os.path.abspath(path)
        for combo in (self.gguf_model_path, self.settings_gguf_model_path):
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
        for combo in (self.gguf_model_path, self.settings_gguf_model_path):
            combo.blockSignals(True)
            combo.clear()
        models = self._iter_gguf_models()
        if models:
            for display, full_path in models:
                self.gguf_model_path.addItem(display, full_path)
                self.settings_gguf_model_path.addItem(display, full_path)
        if self._defaults.gguf_model_path:
            self._add_gguf_model(self._defaults.gguf_model_path)
        elif current:
            self._add_gguf_model(current)
        elif settings_current:
            self._add_gguf_model(settings_current)
        elif models:
            self.gguf_model_path.setCurrentIndex(0)
        for combo in (self.gguf_model_path, self.settings_gguf_model_path):
            combo.blockSignals(False)

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
        for combo in (self.ollama_model, self.settings_ollama_model):
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
            self.start_btn.setEnabled(True)
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
        dialog = PageReviewDialog(
            self,
            page_record=page,
            json_path=self.json_path.text().strip(),
            output_suffix=self.output_suffix.text().strip(),
            font_name=self.font_name.currentText().strip(),
            inpaint_mode=self.inpaint_mode.currentText(),
            use_gpu=self.use_gpu.isChecked(),
        )
        dialog.exec()

    def _start_pipeline(self) -> bool:
        self._sync_settings_to_models()
        self._sync_paths_from_settings()
        if not self.import_dir.text().strip():
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Manga Folder")
            if not path:
                return False
            self.import_dir.setText(path)
        if not self.json_path.text().strip() and self.export_dir.text().strip():
            self.json_path.setText(f"{self.export_dir.text().strip()}\\project.json")
        settings = PipelineSettings(
            import_dir=self.import_dir.text().strip(),
            export_dir=self.export_dir.text().strip(),
            json_path=self.json_path.text().strip(),
            output_suffix=self.output_suffix.text().strip(),
            source_lang=self.source_lang.currentText(),
            target_lang=self.target_lang.currentText(),
            ollama_model=self.ollama_model.currentText(),
            style_guide_path=self.style_path.text().strip(),
            font_name=self.font_name.currentText().strip(),
            use_gpu=self.use_gpu.isChecked(),
            filter_background=self.filter_background.isChecked(),
            filter_strength=self.filter_strength.currentText(),
            detector_engine=self.detector_engine.currentText(),
            ocr_engine=self.ocr_engine.currentText(),
            inpaint_mode=self.inpaint_mode.currentText(),
            font_detection=self.font_detection.currentText(),
            translator_backend=self.translator_backend.currentText(),
            gguf_model_path=self._selected_gguf_model_path(),
            gguf_prompt_style=self.gguf_prompt_style.currentText(),
            gguf_n_ctx=self.gguf_n_ctx.value(),
            gguf_n_gpu_layers=self.gguf_n_gpu_layers.value(),
            gguf_n_threads=self.gguf_n_threads.value(),
            gguf_n_batch=self.gguf_n_batch.value(),
            fast_mode=self.fast_mode.isChecked(),
            auto_glossary=self.auto_glossary.isChecked(),
            detector_input_size=int(self.detector_input_size.currentText()),
            inpaint_model_id=self.inpaint_model_id.text().strip(),
        )
        self._pipeline.start(settings)
        return True

    def _reapply_from_json(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Project JSON", filter="JSON Files (*.json)")
        if not path:
            return
        try:
            data = load_project(path)
        except Exception as exc:
            self.status_bar.showMessage(f"Failed to load JSON: {exc}")
            return
        pages = data.get("pages", [])
        if not pages:
            self.status_bar.showMessage("No pages in project JSON.")
            return
        export_dir = self.export_dir.text().strip()
        if not export_dir:
            export_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Export Folder")
            if not export_dir:
                return
            self.export_dir.setText(export_dir)
        suffix = self.output_suffix.text().strip() or "_translated"
        total = len(pages)
        self.queue_list.clear()
        self.queue_table.setRowCount(0)
        for page in pages:
            name = os.path.basename(page.get("image_path", ""))
            self.queue_list.addItem(f"{name} - pending")
            table_row = self.queue_table.rowCount()
            self.queue_table.insertRow(table_row)
            self.queue_table.setItem(table_row, 0, QtWidgets.QTableWidgetItem(name))
            self.queue_table.setItem(table_row, 1, QtWidgets.QTableWidgetItem("pending"))
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
            self.queue_table.setCellWidget(table_row, 2, action_cell)
        self._update_queue_placeholder()
        self._update_queue_table_placeholder()
        for idx, page in enumerate(pages, start=1):
            image_path = page.get("image_path", "")
            if not image_path:
                continue
            filename = os.path.basename(image_path)
            output_path = os.path.join(export_dir, f"{os.path.splitext(filename)[0]}{suffix}{os.path.splitext(filename)[1]}")
            regions = page.get("regions", [])
            try:
                render_translations(
                    image_path,
                    output_path,
                    regions,
                    self.font_name.currentText().strip(),
                    inpaint_mode=self.inpaint_mode.currentText(),
                    use_gpu=self.use_gpu.isChecked(),
                    model_id=self.inpaint_model_id.text().strip(),
                )
                self._update_queue_item(idx - 1, "done")
            except Exception as exc:
                self._update_queue_item(idx - 1, "error")
                self.status_bar.showMessage(f"Re-apply failed: {exc}")
            self.overall_bar.setValue(int(idx / total * 100))
            self.processing_label.setText(f"Processing Page {idx} of {total}...")

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
        if message.startswith("Failed") or "required" in message:
            self._set_running(False)
        if (
            "PaddleOCR is not installed" in message
            or "Failed to initialize models" in message
            or "NumPy ABI mismatch" in message
            or "PyTorch DLL load failed" in message
        ):
            QtWidgets.QMessageBox.critical(self, "Dependency Error", message)

    def closeEvent(self, event) -> None:
        self._save_settings()
        super().closeEvent(event)

    def _save_settings(self) -> None:
        settings = QtCore.QSettings("MangaTranslator", "Pro")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        
        # Paths
        settings.setValue("import_dir", self.import_dir.text())
        settings.setValue("export_dir", self.export_dir.text())
        settings.setValue("json_path", self.json_path.text())
        
        # Models & Params
        settings.setValue("source_lang", self.source_lang.currentText())
        settings.setValue("target_lang", self.target_lang.currentText())
        settings.setValue("detector_engine", self.detector_engine.currentText())
        settings.setValue("detector_input_size", self.detector_input_size.currentText())
        settings.setValue("ocr_engine", self.ocr_engine.currentText())
        settings.setValue("translator_backend", self.translator_backend.currentText())
        settings.setValue("inpaint_mode", self.inpaint_mode.currentText())
        settings.setValue("filter_strength", self.filter_strength.currentText())
        settings.setValue("use_gpu", self.use_gpu.isChecked())
        settings.setValue("fast_mode", self.fast_mode.isChecked())
        settings.setValue("auto_glossary", self.auto_glossary.isChecked())
        settings.setValue("font_name", self.font_name.currentText())
        
        # GGUF
        settings.setValue("gguf_model_path", self._selected_gguf_model_path())
        settings.setValue("gguf_n_gpu_layers", self.gguf_n_gpu_layers.value())

    def _load_saved_settings(self) -> None:
        settings = QtCore.QSettings("MangaTranslator", "Pro")
        geo = settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        state = settings.value("windowState")
        if state:
            self.restoreState(state)

        def _restore(widget, key, type_func=str):
            val = settings.value(key)
            if val is not None:
                val = type_func(val)
                if isinstance(widget, QtWidgets.QComboBox):
                    widget.setCurrentText(val)
                elif isinstance(widget, QtWidgets.QLineEdit):
                    widget.setText(val)
                elif isinstance(widget, QtWidgets.QCheckBox):
                    widget.setChecked(val == "true" if isinstance(val, str) else bool(val))
                elif isinstance(widget, QtWidgets.QSpinBox):
                    widget.setValue(int(val))

        _restore(self.import_dir, "import_dir")
        _restore(self.export_dir, "export_dir")
        _restore(self.json_path, "json_path")
        _restore(self.source_lang, "source_lang")
        _restore(self.target_lang, "target_lang")
        
        _restore(self.detector_engine, "detector_engine")
        _restore(self.detector_input_size, "detector_input_size")
        _restore(self.ocr_engine, "ocr_engine")
        _restore(self.translator_backend, "translator_backend")
        _restore(self.inpaint_mode, "inpaint_mode")
        _restore(self.filter_strength, "filter_strength")
        _restore(self.use_gpu, "use_gpu", type_func=lambda x: x == "true" if isinstance(x, str) else bool(x))
        _restore(self.fast_mode, "fast_mode", type_func=lambda x: x == "true" if isinstance(x, str) else bool(x))
        _restore(self.auto_glossary, "auto_glossary", type_func=lambda x: x == "true" if isinstance(x, str) else bool(x))
        _restore(self.font_name, "font_name")
        _restore(self.gguf_n_gpu_layers, "gguf_n_gpu_layers", type_func=int)
        
        gguf_path = settings.value("gguf_model_path")
        if gguf_path:
            self._add_gguf_model(str(gguf_path))
            
        self._sync_models_to_settings()
        self._sync_paths_to_settings()
