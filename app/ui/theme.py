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
QGroupBox {
  border: 1px solid #2f3238;
  border-radius: 8px;
  margin-top: 10px;
  padding: 8px;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 10px;
  padding: 0 4px 0 4px;
  color: #e6e6e6;
}
QLineEdit, QComboBox, QPlainTextEdit, QListWidget, QTableWidget {
  background-color: #15171b;
  border: 1px solid #2c2f36;
  border-radius: 6px;
  padding: 4px 6px;
  color: #f0f0f0;
}
QComboBox QAbstractItemView {
  background-color: #15171b;
  border: 1px solid #2c2f36;
  selection-background-color: #2b3848;
  color: #f0f0f0;
}
QListWidget::item, QTableWidget::item {
  padding: 6px;
  border-bottom: 1px solid #1f2228;
}
QListWidget::item:selected, QTableWidget::item:selected {
  background-color: #2b3848;
  color: #f8f8f8;
}
QListWidget::item:hover, QTableWidget::item:hover {
  background-color: #1f2730;
}
QPushButton {
  background-color: #2c313a;
  border: 1px solid #3a3f49;
  border-radius: 6px;
  padding: 6px 10px;
}
QPushButton:hover {
  background-color: #353c47;
}
QPushButton:pressed {
  background-color: #21262d;
}
QProgressBar {
  border: 1px solid #2c2f36;
  border-radius: 6px;
  text-align: right;
  padding: 2px;
}
QProgressBar::chunk {
  background-color: #56b4f0;
  border-radius: 4px;
}
QScrollBar:vertical {
  background: #1d1f24;
  width: 10px;
  margin: 0;
}
QScrollBar::handle:vertical {
  background: #3a3f49;
  min-height: 20px;
  border-radius: 4px;
}
QScrollBar:horizontal {
  background: #1d1f24;
  height: 10px;
  margin: 0;
}
QScrollBar::handle:horizontal {
  background: #3a3f49;
  min-width: 20px;
  border-radius: 4px;
}
QStatusBar {
  color: #e0e0e0;
}
"""


def _light_stylesheet() -> str:
    return """
QGroupBox {
  border: 1px solid #d4d6db;
  border-radius: 8px;
  margin-top: 10px;
  padding: 8px;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 10px;
  padding: 0 4px 0 4px;
  color: #1e1f22;
}
QLineEdit, QComboBox, QPlainTextEdit, QListWidget, QTableWidget {
  background-color: #ffffff;
  border: 1px solid #d4d6db;
  border-radius: 6px;
  padding: 4px 6px;
  color: #1e1f22;
}
QComboBox QAbstractItemView {
  background-color: #ffffff;
  border: 1px solid #d4d6db;
  selection-background-color: #dfeaf7;
  color: #1e1f22;
}
QListWidget::item, QTableWidget::item {
  padding: 6px;
  border-bottom: 1px solid #eceff4;
}
QListWidget::item:selected, QTableWidget::item:selected {
  background-color: #dfeaf7;
  color: #1e1f22;
}
QListWidget::item:hover, QTableWidget::item:hover {
  background-color: #eef3fa;
}
QPushButton {
  background-color: #f1f3f6;
  border: 1px solid #d4d6db;
  border-radius: 6px;
  padding: 6px 10px;
}
QPushButton:hover {
  background-color: #e7eaf0;
}
QPushButton:pressed {
  background-color: #d9dde6;
}
QProgressBar {
  border: 1px solid #d4d6db;
  border-radius: 6px;
  text-align: right;
  padding: 2px;
}
QProgressBar::chunk {
  background-color: #4ca3e0;
  border-radius: 4px;
}
QScrollBar:vertical {
  background: #f1f3f6;
  width: 10px;
  margin: 0;
}
QScrollBar::handle:vertical {
  background: #c9ced8;
  min-height: 20px;
  border-radius: 4px;
}
QScrollBar:horizontal {
  background: #f1f3f6;
  height: 10px;
  margin: 0;
}
QScrollBar::handle:horizontal {
  background: #c9ced8;
  min-width: 20px;
  border-radius: 4px;
}
QStatusBar {
  color: #1e1f22;
}
"""
