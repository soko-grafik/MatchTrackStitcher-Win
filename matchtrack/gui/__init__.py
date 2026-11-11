"""
GUI module for MatchTrack-Stitcher.
Provides the main application window and reusable modular UI components.
"""

from .main_window import MainWindow, launch_gui
from .viewport import InteractivePanoramaViewport
from .obs_output_settings import OBSOutputSettingsWidget
from .live_controller import LiveStreamMixin
from .cloud_controller import CloudIntegrationMixin
from .wide_angle_controller import WideAngleControllerMixin
from .workers import RenderWorker, CloudUploadWorker
from .dialogs import DjiQRCodeDialog, CloudLoginDialog, OBSAdvancedOutputDialog
from .widgets import (
    NoWheelSlider, NoWheelDoubleSpinBox, NoWheelSpinBox, NoWheelComboBox,
    ScrollAreaWheelFilter, LabeledSliderSpinBox, LogEmitter, LogViewerWidget
)

__all__ = [
    "MainWindow",
    "launch_gui",
    "InteractivePanoramaViewport",
    "OBSOutputSettingsWidget",
    "LiveStreamMixin",
    "CloudIntegrationMixin",
    "WideAngleControllerMixin",
    "RenderWorker",
    "CloudUploadWorker",
    "DjiQRCodeDialog",
    "CloudLoginDialog",
    "OBSAdvancedOutputDialog",
    "NoWheelSlider",
    "NoWheelDoubleSpinBox",
    "NoWheelSpinBox",
    "NoWheelComboBox",
    "ScrollAreaWheelFilter",
    "LabeledSliderSpinBox",
    "LogEmitter",
    "LogViewerWidget",
]
