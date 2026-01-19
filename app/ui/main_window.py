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
        self.setWindowTitle("Manga Translator")
        self.resize(1200, 720)

        self._defaults = get_defaults()
        self._pipeline = PipelineController(self)
        self._style_editor: StyleGuideEditor | None = None
        self._review_dialog: RegionReviewDialog | None = None
        self._setup_ui()
        self._connect_signals()
        self._apply_theme(self._defaults.theme)

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, central)
        splitter.setChildrenCollapsible(False)

        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(splitter)

        self.status_bar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.status_bar)

    def _build_left_panel(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(10)

        scroll = QtWidgets.QScrollArea(container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        vbox.addWidget(scroll)

        content = QtWidgets.QWidget(scroll)
        scroll.setWidget(content)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(6, 6, 6, 6)
        content_layout.setSpacing(12)

        content_layout.addWidget(self._group_import())
        content_layout.addWidget(self._group_output())
        content_layout.addWidget(self._group_language())
        content_layout.addWidget(self._group_style_guide())
        content_layout.addWidget(self._group_models())
        content_layout.addWidget(self._group_render())
        content_layout.addWidget(self._group_performance())
        content_layout.addWidget(self._group_theme())
        content_layout.addStretch(1)

        return container

    def _build_right_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(12, 12, 12, 12)
        vbox.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.reapply_btn = QtWidgets.QPushButton("Import JSON (Re-apply)")
        self.review_btn = QtWidgets.QPushButton("Review Regions")
        header.addWidget(self.start_btn)
        header.addWidget(self.stop_btn)
        header.addWidget(self.reapply_btn)
        header.addWidget(self.review_btn)
        header.addStretch(1)
        vbox.addLayout(header)

        summary_box = QtWidgets.QGroupBox("Progress")
        summary_layout = QtWidgets.QVBoxLayout(summary_box)
        self.overall_bar = QtWidgets.QProgressBar()
        self.overall_bar.setValue(0)
        self.eta_label = QtWidgets.QLabel("ETA: --")
        self.page_label = QtWidgets.QLabel("Page: 0 / 0")
        self.total_time_label = QtWidgets.QLabel("Total: --")
        self.page_time_label = QtWidgets.QLabel("Page: --")
        summary_layout.addWidget(self.overall_bar)
        summary_layout.addWidget(self.eta_label)
        summary_layout.addWidget(self.page_label)
        summary_layout.addWidget(self.total_time_label)
        summary_layout.addWidget(self.page_time_label)
        vbox.addWidget(summary_box)

        queue_box = QtWidgets.QGroupBox("Queue")
        queue_layout = QtWidgets.QVBoxLayout(queue_box)
        self.queue_list = QtWidgets.QListWidget()
        queue_layout.addWidget(self.queue_list)
        vbox.addWidget(queue_box)

        return panel

    def _group_import(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Import")
        layout = QtWidgets.QFormLayout(group)
        self.import_dir = QtWidgets.QLineEdit(self._defaults.import_dir)
        self.import_browse = QtWidgets.QPushButton("Browse")
        layout.addRow("Manga Folder", self._hbox(self.import_dir, self.import_browse))
        return group

    def _group_output(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Output")
        layout = QtWidgets.QFormLayout(group)
        self.export_dir = QtWidgets.QLineEdit(self._defaults.export_dir)
        self.export_browse = QtWidgets.QPushButton("Browse")
        layout.addRow("Export Folder", self._hbox(self.export_dir, self.export_browse))
        self.output_suffix = QtWidgets.QLineEdit(self._defaults.output_suffix)
        layout.addRow("Filename Suffix", self.output_suffix)
        self.json_path = QtWidgets.QLineEdit(self._defaults.json_path)
        self.json_browse = QtWidgets.QPushButton("Browse")
        layout.addRow("Project JSON", self._hbox(self.json_path, self.json_browse))
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
        group = QtWidgets.QGroupBox("Style Guide")
        layout = QtWidgets.QFormLayout(group)
        self.style_path = QtWidgets.QLineEdit("")
        self.style_browse = QtWidgets.QPushButton("Browse")
        self.style_edit = QtWidgets.QPushButton("Open Editor")
        layout.addRow("Guide JSON", self._hbox(self.style_path, self.style_browse))
        layout.addRow("", self.style_edit)
        return group

    def _group_models(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Models")
        layout = QtWidgets.QFormLayout(group)
        self.detector_engine = QtWidgets.QComboBox()
        self.detector_engine.addItems(["PaddleOCR", "ComicTextDetector"])
        self.detector_engine.setCurrentText(self._defaults.detector_engine)
        layout.addRow("Text Detector", self.detector_engine)
        self.ocr_engine = QtWidgets.QComboBox()
        self.ocr_engine.addItems(["PaddleOCR", "MangaOCR"])
        self.ocr_engine.setCurrentText(self._defaults.ocr_engine)
        layout.addRow("OCR Engine", self.ocr_engine)
        self.translator_backend = QtWidgets.QComboBox()
        self.translator_backend.addItems(["Ollama", "GGUF"])
        self.translator_backend.setCurrentText(self._defaults.translator_backend)
        layout.addRow("Translator", self.translator_backend)
        self.ollama_model = QtWidgets.QComboBox()
        self.ollama_model.addItems(["auto-detect"])
        self.model_refresh = QtWidgets.QPushButton("Refresh")
        model_row = self._hbox(self.ollama_model, self.model_refresh)
        layout.addRow("Ollama Model", model_row)
        self.gguf_model_path = QtWidgets.QComboBox()
        self.gguf_model_path.setEditable(True)
        self.gguf_model_path.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.gguf_browse = QtWidgets.QPushButton("Browse")
        layout.addRow("GGUF Model", self._hbox(self.gguf_model_path, self.gguf_browse))
        self.gguf_prompt_style = QtWidgets.QComboBox()
        self.gguf_prompt_style.addItems(["sakura", "qwen", "plain"])
        self.gguf_prompt_style.setCurrentText(self._defaults.gguf_prompt_style)
        layout.addRow("GGUF Prompt", self.gguf_prompt_style)
        self.gguf_n_ctx = QtWidgets.QSpinBox()
        self.gguf_n_ctx.setRange(512, 32768)
        self.gguf_n_ctx.setValue(self._defaults.gguf_n_ctx)
        layout.addRow("GGUF Context", self.gguf_n_ctx)
        self.gguf_n_gpu_layers = QtWidgets.QSpinBox()
        self.gguf_n_gpu_layers.setRange(-1, 200)
        self.gguf_n_gpu_layers.setValue(self._defaults.gguf_n_gpu_layers)
        layout.addRow("GGUF GPU Layers", self.gguf_n_gpu_layers)
        self.gguf_n_threads = QtWidgets.QSpinBox()
        self.gguf_n_threads.setRange(1, 128)
        self.gguf_n_threads.setValue(self._defaults.gguf_n_threads)
        layout.addRow("GGUF Threads", self.gguf_n_threads)
        self.gguf_n_batch = QtWidgets.QSpinBox()
        self.gguf_n_batch.setRange(64, 4096)
        self.gguf_n_batch.setValue(self._defaults.gguf_n_batch)
        layout.addRow("GGUF Batch", self.gguf_n_batch)
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
        layout.addRow("", self.filter_background)
        layout.addRow("Filter Strength", self.filter_strength)
        self.inpaint_mode = QtWidgets.QComboBox()
        self.inpaint_mode.addItems(["fast", "ai", "off"])
        self.inpaint_mode.setCurrentText(self._defaults.inpaint_mode)
        layout.addRow("Inpainting", self.inpaint_mode)
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

    def _hbox(self, *widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget)
        return box

    def _connect_signals(self) -> None:
        self.start_btn.clicked.connect(self._start_pipeline)
        self.stop_btn.clicked.connect(self._pipeline.stop)
        self.theme_combo.currentTextChanged.connect(self._apply_theme)
        self.model_refresh.clicked.connect(self._refresh_models)
        self.reapply_btn.clicked.connect(self._reapply_from_json)
        self.review_btn.clicked.connect(self._open_region_review)

        self.import_browse.clicked.connect(self._choose_import_dir)
        self.export_browse.clicked.connect(self._choose_export_dir)
        self.json_browse.clicked.connect(self._choose_json_path)
        self.style_browse.clicked.connect(self._choose_style_path)
        self.style_edit.clicked.connect(self._open_style_editor)
        self.gguf_browse.clicked.connect(self._choose_gguf_model)

        self._pipeline.status.progress_changed.connect(self.overall_bar.setValue)
        self._pipeline.status.eta_changed.connect(self._set_eta)
        self._pipeline.status.page_changed.connect(self._set_page)
        self._pipeline.status.total_time_changed.connect(self._set_total_time)
        self._pipeline.status.page_time_changed.connect(self._set_page_time)
        self._pipeline.status.message.connect(self._handle_message)
        self._pipeline.status.queue_reset.connect(self._set_queue)
        self._pipeline.status.queue_item.connect(self._update_queue_item)

        self._refresh_models()
        self._refresh_gguf_models()

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

    def _choose_export_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if path:
            self.export_dir.setText(path)
            if not self.json_path.text().strip():
                self.json_path.setText(f"{path}\\project.json")

    def _choose_json_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select Project JSON", filter="JSON Files (*.json)")
        if path:
            self.json_path.setText(path)

    def _choose_style_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Style Guide", filter="JSON Files (*.json)")
        if path:
            self.style_path.setText(path)

    def _choose_gguf_model(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select GGUF Model", filter="GGUF Files (*.gguf)")
        if path:
            self._add_gguf_model(path)

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
        existing = False
        for idx in range(self.gguf_model_path.count()):
            if self.gguf_model_path.itemData(idx) == path:
                existing = True
                self.gguf_model_path.setCurrentIndex(idx)
                break
        if not existing:
            display = os.path.basename(path)
            self.gguf_model_path.addItem(display, path)
            self.gguf_model_path.setCurrentIndex(self.gguf_model_path.count() - 1)

    def _refresh_gguf_models(self) -> None:
        current = self._selected_gguf_model_path()
        self.gguf_model_path.blockSignals(True)
        self.gguf_model_path.clear()
        models = self._iter_gguf_models()
        if models:
            for display, full_path in models:
                self.gguf_model_path.addItem(display, full_path)
        if self._defaults.gguf_model_path:
            self._add_gguf_model(self._defaults.gguf_model_path)
        elif current:
            self._add_gguf_model(current)
        elif models:
            self.gguf_model_path.setCurrentIndex(0)
        self.gguf_model_path.blockSignals(False)

    def _selected_gguf_model_path(self) -> str:
        data = self.gguf_model_path.currentData()
        if isinstance(data, str) and data:
            return data
        return self.gguf_model_path.currentText().strip()

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
        self.ollama_model.blockSignals(True)
        self.ollama_model.clear()
        self.ollama_model.addItems(["auto-detect"])
        if models:
            self.ollama_model.addItems(models)
        if current:
            index = self.ollama_model.findText(current)
            if index >= 0:
                self.ollama_model.setCurrentIndex(index)
        self.ollama_model.blockSignals(False)

    def _set_eta(self, eta_text: str) -> None:
        self.eta_label.setText(f"ETA: {eta_text}")

    def _set_page(self, current: int, total: int) -> None:
        self.page_label.setText(f"Page: {current} / {total}")

    def _set_total_time(self, text: str) -> None:
        self.total_time_label.setText(text)

    def _set_page_time(self, text: str) -> None:
        self.page_time_label.setText(text)

    def _set_queue(self, items: list) -> None:
        self.queue_list.clear()
        self.total_time_label.setText("Total: 00:00")
        self.page_time_label.setText("Page: --")
        for name in items:
            self.queue_list.addItem(f"{name} - pending")

    def _update_queue_item(self, index: int, status: str) -> None:
        item = self.queue_list.item(index)
        if not item:
            return
        text = item.text().split(" - ")[0]
        item.setText(f"{text} - {status}")

    def _start_pipeline(self) -> None:
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
        )
        self._pipeline.start(settings)

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
        for page in pages:
            self.queue_list.addItem(f"{page.get('image_path', '')} - pending")
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
                )
                self._update_queue_item(idx - 1, "done")
            except Exception as exc:
                self._update_queue_item(idx - 1, "error")
                self.status_bar.showMessage(f"Re-apply failed: {exc}")
            self.overall_bar.setValue(int(idx / total * 100))
            self.page_label.setText(f"Page: {idx} / {total}")

    def _open_region_review(self) -> None:
        if not self._review_dialog:
            self._review_dialog = RegionReviewDialog(self)
        path = self.json_path.text().strip()
        if path:
            self._review_dialog.set_path(path)
        self._review_dialog.exec()

    def _handle_message(self, message: str) -> None:
        self.status_bar.showMessage(message)
        if (
            "PaddleOCR is not installed" in message
            or "Failed to initialize models" in message
            or "NumPy ABI mismatch" in message
            or "PyTorch DLL load failed" in message
        ):
            QtWidgets.QMessageBox.critical(self, "Dependency Error", message)
