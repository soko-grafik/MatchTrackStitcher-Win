"""
MatchTrack-Stitcher Live Capture Engine.
Supports:
- Any standard USB webcam (Logitech, Microsoft, Razer, OBS Virtual Camera, etc.)
- DJI Osmo Action cameras in USB-C UVC Webcam mode
- Dual-Webcam Stitching or Single-Webcam Direct Broadcast
- Direct network stream ingest (RTSP / RTMP / HTTP streams)
- Real-time device discovery via Windows DirectShow / MediaFoundation & PnP
- Dedicated low-latency capture threads with ring buffers and automatic re-connection
"""
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any

import cv2
import numpy as np

from .logger import get_logger

logger = get_logger("live_capture")


@dataclass
class VideoDeviceInfo:
    index: int
    name: str
    device_id: str = ""
    is_uvc: bool = True
    preferred_backend: int = cv2.CAP_ANY


def enumerate_video_devices(max_devices: int = 10) -> List[VideoDeviceInfo]:
    """
    Discovers all available camera and webcam devices on Windows using:
    1. DirectShow System Device Enumerator via pygrabber (friendly device names)
    2. PowerShell PnP entity querying as fallback
    3. Multi-backend OpenCV probing (MediaFoundation MSMF, DirectShow, CAP_ANY)
    """
    discovered_names: List[str] = []

    # Method 1: DirectShow FilterGraph device enumeration (most accurate for Windows USB webcams)
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        discovered_names = graph.get_input_devices()
        logger.debug(f"DirectShow Webcams gefunden ({len(discovered_names)}): {discovered_names}")
    except Exception as e:
        logger.debug(f"FilterGraph DirectShow Enumeration fehlgeschlagen oder nicht verfügbar: {e}")

    # Method 2: Windows PnP fallback if FilterGraph is empty
    if not discovered_names and sys.platform == "win32":
        try:
            ps_cmd = (
                "Get-CimInstance Win32_PnPEntity | "
                "Where-Object { $_.PNPClass -eq 'Camera' -or $_.Service -eq 'usbvideo' -or $_.PNPClass -eq 'Image' } | "
                "Select-Object -ExpandProperty Name"
            )
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=4
            )
            if res.returncode == 0:
                discovered_names = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        except Exception as e:
            logger.warning(f"Fehler bei Windows Kamera-PnP-Abfrage: {e}")

    devices: List[VideoDeviceInfo] = []
    limit = len(discovered_names) if discovered_names else max_devices

    for idx in range(limit):
        # Probe backends for this index to ensure OpenCV can capture from it
        working_backend: Optional[int] = None
        for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    working_backend = backend
                    cap.release()
                    break
                cap.release()
            except Exception:
                pass

        # If a backend opened it, or if it was explicitly listed in system DirectShow devices
        if working_backend is not None or (idx < len(discovered_names) and discovered_names[idx]):
            name = discovered_names[idx] if idx < len(discovered_names) else f"USB-Webcam {idx}"
            pref_b = working_backend if working_backend is not None else cv2.CAP_ANY
            devices.append(VideoDeviceInfo(
                index=idx,
                name=name,
                is_uvc=True,
                preferred_backend=pref_b
            ))

    return devices


class CameraCaptureThread(threading.Thread):
    """
    Dedicated worker thread capturing frames continuously from a single video source
    (standard USB webcam, DJI Action UVC mode, or RTSP/RTMP stream) with zero accumulated latency.
    """
    def __init__(
        self,
        source: Any,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        name: str = "Camera",
        preferred_backend: Optional[int] = None
    ):
        super().__init__(daemon=True, name=f"LiveCapture_{name}")
        self.source = source  # int (e.g. 0, 1) or str (RTSP/RTMP URL)
        self.req_width = width
        self.req_height = height
        self.req_fps = fps
        self.camera_name = name
        self.preferred_backend = preferred_backend

        self.cap: Optional[cv2.VideoCapture] = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_timestamp: float = 0.0
        self.frame_count: int = 0
        self.is_connected: bool = False
        self.actual_fps: float = 0.0

    def run(self):
        logger.info(f"Starte Live-Capture Thread für '{self.camera_name}' (Quelle: {self.source})...")
        fps_calc_time = time.time()
        fps_calc_frames = 0

        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                self._open_capture()
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(1.0)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning(f"Kamerabildverlust auf '{self.camera_name}'. Versuche Reconnect...")
                with self.lock:
                    self.is_connected = False
                self._close_capture()
                time.sleep(1.0)
                continue

            now = time.time()
            with self.lock:
                self.latest_frame = frame
                self.latest_timestamp = now
                self.frame_count += 1
                self.is_connected = True

            fps_calc_frames += 1
            if now - fps_calc_time >= 1.0:
                with self.lock:
                    self.actual_fps = fps_calc_frames / (now - fps_calc_time)
                fps_calc_time = now
                fps_calc_frames = 0

        self._close_capture()
        logger.info(f"Live-Capture Thread für '{self.camera_name}' beendet.")

    def _open_capture(self):
        self._close_capture()
        try:
            if isinstance(self.source, int):
                # Try preferred backend first, followed by MediaFoundation, DirectShow, and default CAP_ANY
                backends_to_try = []
                if self.preferred_backend is not None:
                    backends_to_try.append(self.preferred_backend)
                for b in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
                    if b not in backends_to_try:
                        backends_to_try.append(b)

                cap = None
                for b in backends_to_try:
                    try:
                        c = cv2.VideoCapture(self.source, b)
                        if c.isOpened():
                            cap = c
                            self.preferred_backend = b
                            break
                        c.release()
                    except Exception:
                        pass

                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
                    cap.set(cv2.CAP_PROP_FPS, self.req_fps)
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for real-time responsiveness
                    except Exception:
                        pass
                    try:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    except Exception:
                        pass
            else:
                # Network Stream (RTSP / RTMP / HTTP)
                cap = cv2.VideoCapture(str(self.source), cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap and cap.isOpened():
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                logger.info(
                    f"Kamera '{self.camera_name}' erfolgreich geöffnet: "
                    f"{actual_w}x{actual_h} @ {actual_fps:.0f} FPS (Backend: {self.preferred_backend})"
                )
                self.cap = cap
                with self.lock:
                    self.is_connected = True
            else:
                if cap:
                    cap.release()
                self.cap = None
        except Exception as e:
            logger.warning(f"Konnte Kamera '{self.camera_name}' nicht öffnen: {e}")
            self.cap = None

    def _close_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float]:
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.latest_timestamp
            return None, 0.0

    def stop(self):
        self.stop_event.set()


class DualLiveCaptureManager:
    """
    Coordinates real-time live ingestion from one or two cameras (USB webcams, UVC, DJI Action)
    supporting both Single-Webcam streaming and Dual-Webcam panorama stitching.
    """
    def __init__(self):
        self.cam_left: Optional[CameraCaptureThread] = None
        self.cam_right: Optional[CameraCaptureThread] = None
        self.single_mode: bool = False
        self.is_running: bool = False

    def start(
        self,
        source_left: Any,
        source_right: Optional[Any] = None,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        backend_left: Optional[int] = None,
        backend_right: Optional[int] = None
    ):
        self.stop()
        self.single_mode = False

        if source_left is not None and source_left != -1 and str(source_left) != "-1":
            self.cam_left = CameraCaptureThread(
                source_left, width, height, fps, name="Links / Einzelkamera", preferred_backend=backend_left
            )
            self.cam_left.start()

        if source_right is not None and source_right != -1 and str(source_right) != "-1":
            self.cam_right = CameraCaptureThread(
                source_right, width, height, fps, name="Rechts", preferred_backend=backend_right
            )
            self.cam_right.start()
        else:
            self.cam_right = None
            self.single_mode = True

        self.is_running = (self.cam_left is not None and self.cam_left.is_alive()) or \
                          (self.cam_right is not None and self.cam_right.is_alive())

        mode_name = "Einzelkamera-Modus" if self.single_mode else "Dual-Stitching Modus"
        logger.info(f"Live-Capture gestartet ({mode_name} - Links: {source_left}, Rechts: {source_right})")

    def stop(self):
        if self.cam_left:
            self.cam_left.stop()
            self.cam_left = None
        if self.cam_right:
            self.cam_right.stop()
            self.cam_right = None
        self.is_running = False
        self.single_mode = False

    def get_synced_frames(self, max_diff_sec: float = 0.08) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Retrieves the latest available frame pair from both cameras (or single frame in single mode).
        Returns: (frame_left, frame_right, is_synchronized)
        In single mode: (frame_left, None, True)
        """
        if self.single_mode:
            if not self.cam_left:
                return None, None, False
            fl, _ = self.cam_left.get_latest_frame()
            return fl, None, True

        if not self.cam_left or not self.cam_right:
            return None, None, False

        fl, tl = self.cam_left.get_latest_frame()
        fr, tr = self.cam_right.get_latest_frame()

        if fl is None or fr is None:
            return fl, fr, False

        diff = abs(tl - tr)
        is_synced = diff <= max_diff_sec
        return fl, fr, is_synced

    def get_status(self) -> Dict[str, Any]:
        return {
            "single_mode": self.single_mode,
            "left_connected": self.cam_left.is_connected if self.cam_left else False,
            "right_connected": self.cam_right.is_connected if self.cam_right else False,
            "left_fps": self.cam_left.actual_fps if self.cam_left else 0.0,
            "right_fps": self.cam_right.actual_fps if self.cam_right else 0.0,
            "is_running": self.is_running
        }
