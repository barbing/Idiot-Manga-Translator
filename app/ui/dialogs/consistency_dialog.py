from PySide6 import QtWidgets, QtCore

class ConsistencyDialog(QtWidgets.QDialog):
    def __init__(self, filenames: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Consistency Check")
        self.resize(500, 400)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        label = QtWidgets.QLabel(
            f"The Auto-Glossary system detected that the following {len(filenames)} pages "
            "may have been translated before key glossary terms were discovered.\n\n"
            "Would you like to re-translate them now using the complete glossary?"
        )
        label.setWordWrap(True)
        label.setStyleSheet("font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(label)
        
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.addItems(filenames)
        layout.addWidget(self.list_widget)
        
        buttons = QtWidgets.QDialogButtonBox()
        self.retranslate_btn = buttons.addButton("Re-translate Pages", QtWidgets.QDialogButtonBox.AcceptRole)
        self.ignore_btn = buttons.addButton("Ignore", QtWidgets.QDialogButtonBox.RejectRole)
        self.deep_scan_btn = buttons.addButton("Deep Scan (LLM)", QtWidgets.QDialogButtonBox.ActionRole)
        self.deep_scan_btn.setToolTip("Use the configured LLM to deep-scan all text for missed names/terms.\nSlow but very accurate.")
        
        self.retranslate_btn.clicked.connect(self.accept)
        self.ignore_btn.clicked.connect(self.reject)
        self.deep_scan_btn.clicked.connect(self._on_deep_scan)
        
        layout.addWidget(buttons)
        
    def _on_deep_scan(self):
        """Trigger deep scan."""
        # We'll use a special return code for Deep Scan
        self.done(100)  # 100 = Custom code for Deep Scan

    def selected_pages(self) -> list[int]:
        """Return list of selected pages (currently all)."""
        # In future we could allow selection in the list widget
        return list(range(self.list_widget.count()))
