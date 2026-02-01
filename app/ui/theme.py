# -*- coding: utf-8 -*-
"""Simple light/dark palette helpers."""
from PySide6 import QtGui


def apply_dark_palette(app) -> None:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(22, 22, 24))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(236, 236, 236))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(18, 18, 20))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(30, 30, 32))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(0, 0, 0))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(236, 236, 236))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(38, 38, 42))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(236, 236, 236))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(86, 180, 240))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(150, 150, 150))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(150, 150, 150))
    app.setPalette(palette)
    app.setStyleSheet(_dark_stylesheet())


def apply_light_palette(app) -> None:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(245, 245, 245))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(235, 235, 235))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(0, 0, 0))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(245, 245, 245))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(76, 163, 224))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
    app.setPalette(palette)
    app.setStyleSheet(_light_stylesheet())


def _dark_stylesheet() -> str:
    return """
* { 
  font-size: 13px; 
  font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
}
QMainWindow {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
    stop:0 #0f172a, stop:0.5 #1e293b, stop:1 #0f172a);
}
QWidget {
  color: #e2e8f0;
}
QGroupBox {
  border: 1px solid #334155;
  border-radius: 8px;
  margin-top: 12px;
  padding: 12px;
  background: rgba(30, 41, 59, 0.5);
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 10px;
  padding: 0 4px;
  color: #38bdf8;
  font-weight: 600;
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 1px;
}
QLineEdit, QComboBox, QPlainTextEdit, QSpinBox {
  background-color: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 6px 10px;
  color: #f8fafc;
  selection-background-color: #0ea5e9;
  selection-color: #ffffff;
}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QSpinBox:focus {
  border: 1px solid #38bdf8;
  background-color: #1e293b;
}
QListWidget, QTableWidget {
  background-color: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  color: #f1f5f9;
  gridline-color: #1e293b;
}
QHeaderView::section {
  background-color: #1e293b;
  color: #94a3b8;
  border: none;
  border-bottom: 2px solid #334155;
  border-right: 1px solid #334155;
  padding: 6px 8px;
  font-weight: 600;
}
QTableWidget::item {
  padding: 6px 8px;
  border-bottom: 1px solid #1e293b;
}
QTableWidget::item:selected, QListWidget::item:selected {
  background-color: rgba(14, 165, 233, 0.2);
  border: 1px solid #0ea5e9;
  color: #ffffff;
}
QTableWidget::item:hover {
  background-color: rgba(56, 189, 248, 0.1);
}
QComboBox QAbstractItemView {
  background-color: #0f172a;
  border: 1px solid #334155;
  selection-background-color: #0ea5e9;
  color: #f1f5f9;
}
QTabWidget::pane {
  border: 1px solid #334155;
  border-radius: 6px;
  background: rgba(30, 41, 59, 0.5);
}
QTabBar::tab {
  background: #0f172a;
  color: #94a3b8;
  padding: 8px 16px;
  border: 1px solid #1e293b;
  margin-right: 2px;
  border-top-left-radius: 6px;
  border-top-right-radius: 6px;
}
QTabBar::tab:hover {
  background: #1e293b;
  color: #e2e8f0;
}
QTabBar::tab:selected {
  background: #1e293b;
  color: #38bdf8;
  border-top: 2px solid #38bdf8;
  border-bottom: 1px solid transparent;
}
QPushButton {
  background-color: #1e293b;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 6px 14px;
  color: #e2e8f0;
  font-weight: 600;
}
QPushButton:hover {
  background-color: #334155;
  border-color: #94a3b8;
}
QPushButton:pressed {
  background-color: #0f172a;
}
QPushButton:checked {
  background-color: #0284c7;
  border-color: #0ea5e9;
  color: #ffffff;
}
/* Primary Start Button with Neon Glow */
QPushButton#startBtn {
  background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0ea5e9, stop:1 #0284c7);
  border: 1px solid #0ea5e9;
  color: #ffffff;
  font-size: 14px;
  font-weight: 700;
}
QPushButton#startBtn:hover {
  background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #38bdf8, stop:1 #0ea5e9);
  border-color: #7dd3fc;
}
/* Stop Button (Semantically Red but Polished) */
QPushButton#stopBtn {
  background-color: rgba(220, 38, 38, 0.1);
  border: 1px solid #dc2626;
  color: #ef4444;
}
QPushButton#stopBtn:hover {
  background-color: #dc2626;
  color: #ffffff;
}
QPushButton[nav="true"], QToolButton[nav="true"] {
  background: transparent;
  border: none;
  border-left: 3px solid transparent;
  padding: 10px 16px;
  text-align: left;
  color: #94a3b8;
  font-family: "Segoe UI";
  font-size: 14px;
}
QPushButton[nav="true"]:hover, QToolButton[nav="true"]:hover {
  color: #e2e8f0;
  background: rgba(51, 65, 85, 0.3);
  font-size: 14px;
  font-family: "Microsoft YaHei", "Segoe UI";
}
QPushButton[nav="true"]:checked, QToolButton[nav="true"]:checked {
  color: #38bdf8;
  border-left: 3px solid #38bdf8;
  background: linear-gradient(90deg, rgba(14, 165, 233, 0.1) 0%, transparent 100%);
  background-color: rgba(15, 23, 42, 0.5); /* Fallback */
  font-weight: 600;
  font-size: 14px;
  font-family: "Microsoft YaHei", "Segoe UI";
}
QProgressBar {
  border: 1px solid #334155;
  border-radius: 6px;
  text-align: center;
  color: #e2e8f0;
  background-color: #0f172a;
}
QProgressBar::chunk {
  background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #a855f7, stop:1 #22d3ee);
  border-radius: 4px;
}
QScrollBar:vertical {
  background: #0f172a;
  width: 10px;
}
QScrollBar::handle:vertical {
  background: #334155;
  min-height: 20px;
  border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
  background: #475569;
}
QStatusBar {
  background-color: #0f172a;
  color: #64748b;
  border-top: 1px solid #1e293b;
}
QCheckBox {
  color: #e2e8f0;
  spacing: 8px;
}
QCheckBox::indicator {
  width: 16px;
  height: 16px;
  background: #0f172a;
  border: 1px solid #475569;
  border-radius: 4px;
}
QCheckBox::indicator:checked {
  background: #0ea5e9;
  border-color: #0ea5e9;
  image: url(resources/check.svg);
}
QCheckBox::indicator:hover {
  border-color: #38bdf8;
}
QToolTip {
  background-color: #1e293b;
  color: #e2e8f0;
  border: 1px solid #334155;
  font-size: 12px;
  font-family: "Microsoft YaHei", "Segoe UI";
}
"""


def _light_stylesheet() -> str:
    return """
* { 
  font-size: 13px; 
  font-family: "Segoe UI", sans-serif;
}
QMainWindow {
  background-color: #f8fafc;
}
QWidget {
  color: #0f172a;
}
QGroupBox {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  margin-top: 10px;
  padding: 12px;
  background-color: #ffffff;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 10px;
  padding: 0 4px;
  color: #64748b;
  font-weight: 600;
}
QLineEdit, QComboBox, QPlainTextEdit, QSpinBox {
  background-color: #ffffff;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 6px 10px;
  color: #0f172a;
}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QSpinBox:focus {
  border: 1px solid #0ea5e9;
  outline: none;
}
QListWidget, QTableWidget {
  background-color: #ffffff;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  color: #0f172a;
  gridline-color: #e2e8f0;
}
QHeaderView::section {
  background-color: #f1f5f9;
  color: #475569;
  border: none;
  border-bottom: 2px solid #cbd5e1;
  border-right: 1px solid #e2e8f0;
  padding: 6px 8px;
  font-weight: 600;
}
QTableWidget::item {
  padding: 6px;
  border-bottom: 1px solid #f1f5f9;
}
QListWidget::item:selected, QTableWidget::item:selected {
  background-color: #0ea5e9;
  color: #ffffff;
}
QListWidget::item:hover, QTableWidget::item:hover {
  background-color: #f1f5f9;
}
QTabWidget::pane {
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  background-color: #f1f5f9;
}
QTabBar::tab {
  background: #ffffff;
  color: #64748b;
  padding: 8px 16px;
  border: 1px solid transparent;
}
QTabBar::tab:selected {
  background: #f1f5f9;
  color: #0f172a;
  border: 1px solid #e2e8f0;
  border-bottom: 1px solid #f1f5f9;
  font-weight: 500;
}
QPushButton {
  background-color: #ffffff;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 6px 14px;
  color: #0f172a;
  font-weight: 500;
}
QPushButton:hover {
  background-color: #f8fafc;
  border-color: #94a3b8;
}
QPushButton:pressed {
  background-color: #e2e8f0;
}
QPushButton:checked {
  background-color: #0ea5e9;
  border-color: #0ea5e9;
  color: #ffffff;
}
/* Start Button (Primary) */
QPushButton#startBtn {
  background-color: #0ea5e9;
  border: 1px solid #0ea5e9;
  color: #ffffff;
  font-weight: 600;
}
QPushButton#startBtn:hover {
  background-color: #0284c7;
}
/* Stop Button (Danger) */
QPushButton#stopBtn {
  background-color: #ffffff;
  border: 1px solid #ef4444;
  color: #ef4444;
  font-weight: 600;
}
QPushButton#stopBtn:hover {
  background-color: #ef4444;
  color: #ffffff;
}
QPushButton#stopBtn:disabled {
  border-color: #e2e8f0;
  color: #cbd5e1;
  background-color: #f8fafc;
}
QPushButton[nav="true"], QToolButton[nav="true"] {
  background: transparent;
  border: none;
  border-radius: 6px;
  padding: 10px 12px;
  text-align: left;
  color: #0f172a;
}
QPushButton[nav="true"]:hover, QToolButton[nav="true"]:hover {
  background-color: #e2e8f0;
  font-size: 14px;
  font-family: "Segoe UI";
}
QPushButton[nav="true"]:checked, QToolButton[nav="true"]:checked {
  background-color: #e2e8f0;
  color: #0ea5e9;
  font-weight: 600;
  font-size: 14px;
  font-family: "Segoe UI";
}
QProgressBar {
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  text-align: center;
  color: #475569;
}
QProgressBar::chunk {
  background-color: #22c55e;
  border-radius: 5px;
}
QScrollBar:vertical {
  background: #ffffff;
  width: 10px;
}
QScrollBar::handle:vertical {
  background: #cbd5e1;
  border-radius: 5px;
}
QStatusBar {
  background-color: #f8fafc;
  color: #64748b;
  border-top: 1px solid #cbd5e1;
}
"""
