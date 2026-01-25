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
QMainWindow {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
    stop:0 #0b1018, stop:0.5 #0d1422, stop:1 #0a0f16);
}
QWidget {
  color: #e6edf7;
}
QGroupBox {
  border: 1px solid #1f2a3a;
  border-radius: 14px;
  margin-top: 12px;
  padding: 10px;
  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
    stop:0 #121a28, stop:1 #0f1726);
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 14px;
  padding: 0 6px 0 6px;
  color: #d6e1f2;
  font-weight: 600;
}
QLineEdit, QComboBox, QPlainTextEdit, QListWidget, QTableWidget, QSpinBox {
  background-color: #0f1622;
  border: 1px solid #263244;
  border-radius: 10px;
  padding: 7px 10px;
  color: #edf3ff;
}
QHeaderView::section {
  background-color: #121a28;
  color: #c9d5ea;
  border: none;
  padding: 6px 8px;
}
QTableWidget {
  gridline-color: #182233;
}
QComboBox QAbstractItemView {
  background-color: #0f1622;
  border: 1px solid #263244;
  selection-background-color: #1a2a3a;
  color: #edf3ff;
}
QComboBox::drop-down {
  border-left: 1px solid #263244;
  width: 24px;
}
QComboBox::down-arrow {
  image: none;
  width: 0;
  height: 0;
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: 6px solid #7aa2d6;
}
QListWidget::item, QTableWidget::item {
  padding: 6px;
  border-bottom: 1px solid #1a2331;
}
QListWidget::item:selected, QTableWidget::item:selected {
  background-color: #152538;
  color: #f2f7ff;
}
QListWidget::item:hover, QTableWidget::item:hover {
  background-color: #122030;
}
QTabWidget::pane {
  border: 1px solid #1f2a3a;
  border-radius: 12px;
  padding: 4px;
}
QTabBar::tab {
  background: #0f1622;
  color: #9db1cc;
  padding: 7px 14px;
  border-top-left-radius: 10px;
  border-top-right-radius: 10px;
}
QTabBar::tab:selected {
  background: #132236;
  color: #e9f1ff;
}
QPushButton {
  background-color: #141c29;
  border: 1px solid #2a394d;
  border-radius: 10px;
  padding: 8px 12px;
  color: #e7eef9;
}
QPushButton[tableAction="true"] {
  padding: 4px 12px;
  border-radius: 8px;
}
QPushButton[nav="true"] {
  background: transparent;
  border: 0px solid transparent;
  border-left: 3px solid transparent;
  border-radius: 12px;
  padding: 10px 12px;
  text-align: left;
}
QPushButton[nav="true"]:hover {
  background-color: rgba(22, 34, 52, 160);
}
QPushButton[nav="true"]:checked {
  background-color: rgba(27, 46, 74, 210);
  border-color: #2c3d55;
  border-left: 3px solid #5b8cff;
  border-top-right-radius: 2px;
  border-bottom-right-radius: 2px;
  color: #ecf4ff;
}
QToolButton[nav="true"] {
  background: transparent;
  border: 0px solid transparent;
  border-left: 3px solid transparent;
  border-radius: 12px;
  padding: 10px 12px;
  text-align: left;
}
QToolButton[nav="true"]:hover {
  background-color: rgba(22, 34, 52, 160);
}
QToolButton[nav="true"]:checked {
  background-color: rgba(27, 46, 74, 210);
  border-color: #2c3d55;
  border-left: 3px solid #5b8cff;
  border-top-right-radius: 2px;
  border-bottom-right-radius: 2px;
  color: #ecf4ff;
}
QPushButton:hover {
  background-color: #1a2536;
}
QPushButton:pressed {
  background-color: #101826;
}
QPushButton:checked {
  background-color: #132a45;
  border-color: #3c8cff;
  color: #e9f2ff;
}
QProgressBar {
  border: 1px solid #243246;
  border-radius: 10px;
  text-align: right;
  padding: 3px;
  color: #c7d3e6;
  background-color: #0e1420;
}
QProgressBar::chunk {
  background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
    stop:0 #a855f7, stop:0.45 #7c3aed, stop:0.7 #60a5fa, stop:1 #38bdf8);
  border-radius: 8px;
  margin: 1px;
}
QScrollBar:vertical {
  background: #0f141d;
  width: 12px;
  margin: 0;
}
QScrollBar::handle:vertical {
  background: #243248;
  min-height: 20px;
  border-radius: 6px;
}
QScrollBar:horizontal {
  background: #0f141d;
  height: 12px;
  margin: 0;
}
QScrollBar::handle:horizontal {
  background: #243248;
  min-width: 20px;
  border-radius: 6px;
}
QStatusBar {
  color: #b9c6dc;
}
QToolTip {
  background-color: #1b2331;
  color: #e6edf7;
  border: 1px solid #2a394d;
}
"""


def _light_stylesheet() -> str:
    return """
QMainWindow {
  background-color: #f2f5f9;
}
QGroupBox {
  border: 1px solid #d5dbe5;
  border-radius: 10px;
  margin-top: 12px;
  padding: 10px;
  background-color: #ffffff;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 12px;
  padding: 0 4px 0 4px;
  color: #1f2937;
}
QLineEdit, QComboBox, QPlainTextEdit, QListWidget, QTableWidget {
  background-color: #ffffff;
  border: 1px solid #d5dbe5;
  border-radius: 8px;
  padding: 6px 8px;
  color: #1f2937;
}
QComboBox QAbstractItemView {
  background-color: #ffffff;
  border: 1px solid #d5dbe5;
  selection-background-color: #e5eef9;
  color: #1f2937;
}
QComboBox::drop-down {
  border-left: 1px solid #d5dbe5;
  width: 22px;
}
QComboBox::down-arrow {
  image: none;
  width: 0;
  height: 0;
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: 6px solid #6b7280;
}
QListWidget::item, QTableWidget::item {
  padding: 6px;
  border-bottom: 1px solid #eceff4;
}
QListWidget::item:selected, QTableWidget::item:selected {
  background-color: #e5eef9;
  color: #1f2937;
}
QListWidget::item:hover, QTableWidget::item:hover {
  background-color: #f0f5fb;
}
QTabWidget::pane {
  border: 1px solid #d5dbe5;
  border-radius: 8px;
  padding: 2px;
}
QTabBar::tab {
  background: #ffffff;
  color: #4b5563;
  padding: 6px 12px;
  border-top-left-radius: 8px;
  border-top-right-radius: 8px;
}
QTabBar::tab:selected {
  background: #e5eef9;
  color: #1f2937;
}
QPushButton {
  background-color: #f2f5fb;
  border: 1px solid #d5dbe5;
  border-radius: 8px;
  padding: 7px 12px;
  color: #1f2937;
}
QPushButton:hover {
  background-color: #e7edf6;
}
QPushButton:pressed {
  background-color: #d9e2ee;
}
QPushButton:checked {
  background-color: #dbe9ff;
  border-color: #3b82f6;
  color: #1f2937;
}
QProgressBar {
  border: 1px solid #d5dbe5;
  border-radius: 8px;
  text-align: right;
  padding: 3px;
  color: #4b5563;
}
QProgressBar::chunk {
  background-color: #3b82f6;
  border-radius: 6px;
}
QScrollBar:vertical {
  background: #f1f3f6;
  width: 12px;
  margin: 0;
}
QScrollBar::handle:vertical {
  background: #b8c2d1;
  min-height: 20px;
  border-radius: 6px;
}
QScrollBar:horizontal {
  background: #f1f3f6;
  height: 12px;
  margin: 0;
}
QScrollBar::handle:horizontal {
  background: #b8c2d1;
  min-width: 20px;
  border-radius: 6px;
}
QStatusBar {
  color: #1f2937;
}
"""
