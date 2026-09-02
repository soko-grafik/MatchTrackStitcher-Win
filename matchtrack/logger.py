"""
Central Logging Infrastructure for MatchTrack-Stitcher.
Provides unified console, file, and in-memory GUI log stream with timestamps and log levels.
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import List, Optional, Callable
from .paths import get_log_file_path

LOG_FORMAT = "[%(asctime)s.%(msecs)03d] [%(levelname)-7s] [%(threadName)-12s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class GuiLogRecord:
    def __init__(self, timestamp: str, level: str, thread_name: str, message: str):
        self.timestamp = timestamp
        self.level = level
        self.thread_name = thread_name
        self.message = message

    def formatted(self) -> str:
        return f"[{self.timestamp}] [{self.level:<7}] {self.message}"


class GuiLogHandler(logging.Handler):
    """Logging handler that retains recent logs and invokes GUI subscriber callbacks."""
    def __init__(self, max_records: int = 2000):
        super().__init__()
        self.max_records = max_records
        self.records: List[GuiLogRecord] = []
        self._subscribers: List[Callable[[GuiLogRecord], None]] = []

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            dt_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
            log_item = GuiLogRecord(
                timestamp=dt_str,
                level=record.levelname,
                thread_name=record.threadName,
                message=record.getMessage()
            )
            self.records.append(log_item)
            if len(self.records) > self.max_records:
                self.records.pop(0)

            for sub in list(self._subscribers):
                try:
                    sub(log_item)
                except Exception:
                    pass
        except Exception:
            self.handleError(record)

    def subscribe(self, callback: Callable[[GuiLogRecord], None]):
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[GuiLogRecord], None]):
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def get_all_records(self) -> List[GuiLogRecord]:
        return list(self.records)

    def clear(self):
        self.records.clear()


# Global logger instance and GUI handler
_gui_handler = GuiLogHandler()
_logger_initialized = False

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Initializes root application logger with file, console, and GUI handlers."""
    global _logger_initialized
    root_logger = logging.getLogger("matchtrack")
    root_logger.setLevel(level)

    if _logger_initialized:
        return root_logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. File Handler (Rotating 10MB x 3 backups)
    log_file = get_log_file_path()
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=3, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not initialize log file at {log_file}: {e}")

    # 3. GUI Handler
    _gui_handler.setLevel(level)
    _gui_handler.setFormatter(formatter)
    root_logger.addHandler(_gui_handler)

    _logger_initialized = True
    root_logger.info(f"MatchTrack-Stitcher Logger gestartet. Logdatei: {log_file}")
    return root_logger

def get_logger(name: str = "matchtrack") -> logging.Logger:
    """Returns child or root logger for matchtrack namespace."""
    if not _logger_initialized:
        setup_logging()
    if name == "matchtrack":
        return logging.getLogger("matchtrack")
    return logging.getLogger(f"matchtrack.{name}")

def get_gui_handler() -> GuiLogHandler:
    return _gui_handler

