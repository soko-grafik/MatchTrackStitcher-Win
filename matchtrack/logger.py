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
import faulthandler
import threading
import traceback
import platform

from .paths import get_log_file_path, get_crash_log_file_path

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
_crash_handlers_installed = False


class FlushingRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that immediately flushes every log message to disk."""
    def emit(self, record: logging.LogRecord):
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


def get_process_memory_mb() -> float:
    """Returns current process RAM usage (Working Set) in MB without external dependencies."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ('cb', wintypes.DWORD),
                    ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t)
                ]

            psapi = ctypes.windll.psapi
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            h = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(h, ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / (1024 * 1024)
        except Exception:
            pass
    return 0.0


def install_crash_handlers(logger_instance: Optional[logging.Logger] = None):
    """
    Installs:
    1. Low-level OS fault handler (faulthandler) for C/C++ segmentation faults, access violations (0xC0000005)
    2. sys.excepthook for unhandled exceptions in the main thread
    3. threading.excepthook for unhandled exceptions in any background thread
    """
    global _crash_handlers_installed
    if _crash_handlers_installed:
        return

    log = logger_instance or get_logger("crash")

    # 1. Low-level OS fault handler (catches hard crashes in OpenCV, FFmpeg, Qt, etc.)
    try:
        crash_log_path = get_crash_log_file_path()
        os.makedirs(os.path.dirname(os.path.abspath(crash_log_path)), exist_ok=True)
        crash_file = open(crash_log_path, "a", encoding="utf-8")
        crash_file.write(
            f"\n{'='*75}\n"
            f"[CRASH-WATCHDOG AKTIV] Initialisiert am {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"PID: {os.getpid()} | Python: {platform.python_version()} | Platform: {platform.platform()}\n"
            f"{'='*75}\n"
        )
        crash_file.flush()
        faulthandler.enable(file=crash_file, all_threads=True)
        log.info(f"Nativer Crash-Handler (faulthandler) aktiv -> {crash_log_path}")
    except Exception as e:
        log.warning(f"Konnte faulthandler nicht initialisieren: {e}")

    # 2. Main thread uncaught exception hook
    def sys_excepthook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_str = "".join(tb_lines)
        log.critical(f"FATAL: Unbehandelte Ausnahme im MainThread:\n{tb_str}")
        for h in log.handlers:
            try:
                h.flush()
            except Exception:
                pass
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = sys_excepthook

    # 3. Threading uncaught exception hook (Python 3.8+)
    if hasattr(threading, "excepthook"):
        def thread_excepthook(args):
            if issubclass(args.exc_type, KeyboardInterrupt):
                return
            tb_lines = traceback.format_exception(args.exc_type, args.exc_value, args.exc_tb)
            tb_str = "".join(tb_lines)
            th_name = args.thread.name if args.thread else "Unbekannt"
            log.critical(f"FATAL: Unbehandelte Ausnahme in Thread '{th_name}':\n{tb_str}")
            for h in log.handlers:
                try:
                    h.flush()
                except Exception:
                    pass

        threading.excepthook = thread_excepthook

    _crash_handlers_installed = True
    log.info("Globale Ausnahme- und Thread-Crash-Handler erfolgreich installiert.")


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

    # 2. File Handler (Rotating 10MB x 3 backups, always auto-flushing)
    log_file = get_log_file_path()
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = FlushingRotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=3, encoding="utf-8")
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

    # Auto-install crash handlers
    install_crash_handlers(root_logger)

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

