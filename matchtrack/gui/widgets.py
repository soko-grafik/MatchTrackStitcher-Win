"""
Reusable input widgets and logging viewer components for MatchTrack-Stitcher GUI.
"""

import os
from PySide6.QtCore import Qt, Signal, Slot, QObject, QEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QDoubleSpinBox,
    QSpinBox, QComboBox, QToolButton, QPushButton, QCheckBox, QPlainTextEdit,
    QApplication, QScrollArea, QAbstractSpinBox
)

from ..logger import get_gui_handler, get_log_file_path, GuiLogRecord


class NoWheelSlider(QSlider):
    """
    QSlider that ignores mouse-wheel events so that scrolling over the slider
    scrolls the parent QScrollArea instead of accidentally modifying values.
    """
    def wheelEvent(self, event):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """
    QDoubleSpinBox that ignores mouse-wheel events to prevent accidental value changes.
    """
    def wheelEvent(self, event):
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """
    QSpinBox that ignores mouse-wheel events to prevent accidental value changes.
    """
    def wheelEvent(self, event):
        event.ignore()


class NoWheelComboBox(QComboBox):
    """
    QComboBox that ignores mouse-wheel events when popup is closed to prevent accidental selection changes.
    """
    def wheelEvent(self, event):
        event.ignore()


class ScrollAreaWheelFilter(QObject):
    """
    Event filter for interactive input controls (Sliders, SpinBoxes, ComboBoxes)
    inside scroll areas: ensures mouse-wheel events scroll the enclosing QScrollArea
    instead of altering input values.
    """
    def eventFilter(self, obj, event):
        try:
            if event is not None and event.type() == QEvent.Type.Wheel:
                if isinstance(obj, (QSlider, QAbstractSpinBox, QComboBox)):
                    parent = obj.parentWidget()
                    while parent is not None:
                        if isinstance(parent, QScrollArea):
                            QApplication.sendEvent(parent.viewport(), event)
                            return True
                        parent = parent.parentWidget()
                    event.ignore()
                    return True
            return super().eventFilter(obj, event) if hasattr(super(), "eventFilter") else False
        except Exception:
            return False


class LabeledSliderSpinBox(QWidget):
    """
    Ergonomic combined widget with:
    - Title Label + precise formatted QDoubleSpinBox
    - Smooth horizontal QSlider underneath with real-time feedback
    - ↺ Quick-Reset Button to return to default value.
    """
    valueChanged = Signal(float)

    def __init__(self, label_text: str, min_val: float, max_val: float, default_val: float, 
                 step: float = 0.1, suffix: str = "°", decimals: int = 1, parent=None):
        super().__init__(parent)
        self.default_val = default_val
        self.step = step
        self.multiplier = int(round(1.0 / step)) if step < 1.0 else 1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 6)
        layout.setSpacing(3)

        # Header row: Title Label + SpinBox
        header = QHBoxLayout()
        self.lbl_title = QLabel(label_text)
        self.lbl_title.setStyleSheet("font-weight: 600; color: #d4d4d8; font-size: 12px;")
        
        self.spin = NoWheelDoubleSpinBox()
        self.spin.setRange(min_val, max_val)
        self.spin.setValue(default_val)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        if suffix:
            self.spin.setSuffix(f" {suffix}")
        self.spin.setStyleSheet("background-color: #111117; border: 1px solid #282839; border-radius: 6px; padding: 2px 8px; font-weight: 600; color: #60a5fa;")

        header.addWidget(self.lbl_title)
        header.addStretch()
        header.addWidget(self.spin)
        layout.addLayout(header)

        # Slider row: Smooth Slider + ↺ Reset Button
        slider_row = QHBoxLayout()
        slider_row.setSpacing(6)

        self.slider = NoWheelSlider(Qt.Horizontal)
        self.slider.setTracking(True)
        self.slider.setRange(int(round(min_val * self.multiplier)), int(round(max_val * self.multiplier)))
        self.slider.setValue(int(round(default_val * self.multiplier)))

        btn_reset = QToolButton()
        btn_reset.setText("↺")
        btn_reset.setToolTip(f"Auf Standardwert ({default_val}{suffix}) zurücksetzen")
        btn_reset.setStyleSheet("background-color: #1e1e28; border: 1px solid #2e2e3e; border-radius: 6px; color: #9ca3af; font-weight: bold; padding: 2px 7px;")
        btn_reset.clicked.connect(self.reset_default)

        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(btn_reset)
        layout.addLayout(slider_row)

        # Connect signals
        self.spin.valueChanged.connect(self._on_spin_changed)
        self.slider.valueChanged.connect(self._on_slider_changed)

    def _on_spin_changed(self, val: float):
        slider_val = int(round(val * self.multiplier))
        if self.slider.value() != slider_val:
            self.slider.blockSignals(True)
            self.slider.setValue(slider_val)
            self.slider.blockSignals(False)
        self.valueChanged.emit(val)

    def _on_slider_changed(self, s_val: int):
        val = s_val / float(self.multiplier)
        if abs(self.spin.value() - val) > 1e-4:
            self.spin.blockSignals(True)
            self.spin.setValue(val)
            self.spin.blockSignals(False)
        self.valueChanged.emit(val)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, val: float):
        self.spin.setValue(val)

    def reset_default(self):
        self.setValue(self.default_val)


class LogEmitter(QObject):
    log_signal = Signal(object)


class LogViewerWidget(QWidget):
    """
    Real-Time System Log Viewer with HTML color formatting,
    level filtering, clipboard export, and direct log file opening.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.emitter = LogEmitter()
        self.emitter.log_signal.connect(self._on_log_signal)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header Toolbar
        tb_layout = QHBoxLayout()
        tb_layout.setSpacing(6)

        self.combo_filter = QComboBox()
        self.combo_filter.addItem("Alle Logs (Debug, Info, Warn, Fehler)", "ALL")
        self.combo_filter.addItem("Nur Info, Warnungen & Fehler", "INFO")
        self.combo_filter.addItem("Nur Warnungen & Fehler", "WARNING")
        self.combo_filter.addItem("Nur Fehler (Errors)", "ERROR")
        self.combo_filter.currentIndexChanged.connect(self.refresh_display)

        self.chk_autoscroll = QCheckBox("Auto-Scroll")
        self.chk_autoscroll.setChecked(True)
        self.chk_autoscroll.setStyleSheet("color: #ccc;")

        btn_copy = QPushButton("📋 Kopieren")
        btn_copy.setToolTip("Gesamtes Log in die Zwischenablage kopieren")
        btn_copy.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_copy.clicked.connect(self.copy_to_clipboard)

        btn_open_file = QPushButton("📂 Datei")
        btn_open_file.setToolTip("matchtrack.log im Texteditor öffnen")
        btn_open_file.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_open_file.clicked.connect(self.open_log_file)

        btn_clear = QPushButton("🗑️ Leeren")
        btn_clear.setToolTip("Loganzeige leeren")
        btn_clear.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_clear.clicked.connect(self.clear_logs)

        tb_layout.addWidget(self.combo_filter, 1)
        tb_layout.addWidget(self.chk_autoscroll)
        tb_layout.addWidget(btn_copy)
        tb_layout.addWidget(btn_open_file)
        tb_layout.addWidget(btn_clear)
        layout.addLayout(tb_layout)

        # Log Output Box
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(1500)
        self.txt_log.setStyleSheet("""
            QPlainTextEdit {
                background-color: #0b0f14;
                color: #e0e0e0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                line-height: 1.3;
                border: 1px solid #233140;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        layout.addWidget(self.txt_log, 1)

        # Subscribe to GuiLogHandler
        self.gui_handler = get_gui_handler()
        self.gui_handler.subscribe(lambda rec: self.emitter.log_signal.emit(rec))

        # Initial populate
        self.refresh_display()

    @Slot(object)
    def _on_log_signal(self, rec: GuiLogRecord):
        if self._matches_filter(rec.level):
            formatted_line = self._format_record_html(rec)
            self.txt_log.appendHtml(formatted_line)
            if self.chk_autoscroll.isChecked():
                self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def _matches_filter(self, level: str) -> bool:
        mode = self.combo_filter.currentData()
        if mode == "ALL":
            return True
        if mode == "INFO":
            return level in ("INFO", "WARNING", "ERROR", "CRITICAL")
        if mode == "WARNING":
            return level in ("WARNING", "ERROR", "CRITICAL")
        if mode == "ERROR":
            return level in ("ERROR", "CRITICAL")
        return True

    def _format_record_html(self, rec: GuiLogRecord) -> str:
        color = "#a0a0a0"
        tag_bg = "#222"
        if rec.level == "ERROR":
            color = "#ff6b6b"
            tag_bg = "#5c1d1d"
        elif rec.level == "WARNING":
            color = "#feca57"
            tag_bg = "#574218"
        elif rec.level == "INFO":
            color = "#48dbfb"
            tag_bg = "#193c4d"
        elif rec.level == "DEBUG":
            color = "#747d8c"
            tag_bg = "#2f3542"

        msg_safe = rec.message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (
            f'<span style="color: #636e72;">[{rec.timestamp}]</span> '
            f'<span style="background-color: {tag_bg}; color: {color}; font-weight: bold; padding: 1px 4px; border-radius: 2px;">{rec.level:<7}</span> '
            f'<span style="color: #dfe6e9;">{msg_safe}</span>'
        )

    def refresh_display(self):
        self.txt_log.clear()
        records = self.gui_handler.get_all_records()
        for rec in records:
            if self._matches_filter(rec.level):
                self.txt_log.appendHtml(self._format_record_html(rec))
        if self.chk_autoscroll.isChecked():
            self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def copy_to_clipboard(self):
        text = self.txt_log.toPlainText()
        QApplication.clipboard().setText(text)

    def open_log_file(self):
        log_path = get_log_file_path()
        if os.path.exists(log_path):
            try:
                os.startfile(log_path)
            except Exception:
                import subprocess
                subprocess.Popen(["notepad.exe", log_path])

    def clear_logs(self):
        self.gui_handler.clear()
        self.txt_log.clear()

