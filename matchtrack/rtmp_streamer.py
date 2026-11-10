"""
MatchTrack-Stitcher OBS-Style RTMP Live-Streaming Engine.
Features:
- Hardware-accelerated H.264/HEVC encoding via NVIDIA NVENC, Intel QuickSync (QSV), or x264
- Low-latency RTMP/RTMPS/SRT output matching OBS Studio pipeline specifications
- Multi-destination streaming: Simultaneous Broadcast (16:9), Panorama (8:3 / 32:9), and Raw Feeds
- Non-blocking stdin pipe queue preventing UI thread stalls during network jitter
- Real-time stream telemetry: bitrate (kbps), actual FPS, dropped frames, and connection health
"""
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List

import numpy as np

from .paths import get_ffmpeg_path
from .logger import get_logger

logger = get_logger("rtmp_streamer")


@dataclass
class RTMPChannelConfig:
    channel_id: str                      # e.g. "broadcast_16x9", "panorama_8x3", "cam1", "cam2"
    name: str                            # Human-readable title
    stream_url: str                      # e.g. "rtmp://matchtrack.de:1935/live/stream_broadcast"
    stream_key: str = ""                 # Optional stream key appended to URL
    enabled: bool = True
    width: int = 1920
    height: int = 1080
    fps: int = 30
    bitrate_kbps: int = 3000
    codec: str = "h264_nvenc"           # "h264_nvenc", "h264_qsv", "libx264", "hevc_nvenc"
    keyframe_interval_sec: float = 2.0   # Standard GOP size for RTMP/HLS (2 seconds)
    audio_enabled: bool = True
    audio_device_name: str = ""          # DirectShow audio device name, e.g. "Mikrofon (DJI Action 4)"
    source_feed: str = "broadcast_16x9"  # "broadcast_16x9", "panorama_8x3", "cam1", "cam2"

    # OBS Studio Advanced Output & Encoder Settings
    output_mode: str = "advanced"        # "simple" or "advanced"
    audio_track: int = 1                 # 1..6
    audio_encoder: str = "aac"           # "aac" (FFmpeg AAC)
    rescale_output: bool = False
    rescale_width: int = 1920
    rescale_height: int = 1080
    rate_control: str = "vbr"            # "vbr", "cbr", "cqp", "lossless"
    max_bitrate_kbps: int = 5000         # Maximum Bitrate for VBR
    cqp_value: int = 23                  # For CQP rate control
    preset: str = "p4"                   # "p1".."p7" for nvenc, "ultrafast".."slow" for x264
    tuning: str = "hq"                   # "hq", "ll", "ull", "lossless"
    multipass: str = "qres"              # "disabled", "qres", "fullres"
    profile: str = "main"                # "main", "high", "baseline"
    lookahead: bool = True               # Lookahead
    adaptive_quant: bool = True          # Spatial AQ (Adaptive Quantisierung)
    b_frames: int = 2                    # B-Frames
    custom_options: str = ""             # Benutzerdefinierte Kodiereroptionen


@dataclass
class StreamStats:
    channel_id: str
    is_live: bool = False
    uptime_sec: float = 0.0
    current_bitrate_kbps: float = 0.0
    current_fps: float = 0.0
    frames_sent: int = 0
    frames_dropped: int = 0
    status_text: str = "Bereit"


class RTMPChannelStreamer:
    """
    Manages an individual FFmpeg RTMP streaming subprocess with non-blocking pipe writes
    and real-time telemetry extraction from stderr.
    """
    def __init__(self, config: RTMPChannelConfig):
        self.config = config
        self.proc: Optional[subprocess.Popen] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=45)  # ~1.5 seconds frame buffer
        self.stop_event = threading.Event()
        self.writer_thread: Optional[threading.Thread] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.start_time: float = 0.0
        self.frames_sent: int = 0
        self.frames_dropped: int = 0
        self.current_fps: float = 0.0
        self.current_bitrate_kbps: float = 0.0
        self.status_text: str = "Gestoppt"
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        if self.is_running:
            return

        self.stop_event.clear()
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        full_dest = self.config.stream_url.strip()
        if self.config.stream_key and not full_dest.endswith(self.config.stream_key):
            full_dest = f"{full_dest.rstrip('/')}/{self.config.stream_key.lstrip('/')}"

        ffmpeg_bin = get_ffmpeg_path()
        gop_size = max(15, int(self.config.fps * self.config.keyframe_interval_sec))

        # Base raw video input via stdin
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "info",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.config.width}x{self.config.height}",
            "-r", str(self.config.fps),
            "-i", "-",
        ]

        # Audio input (DirectShow audio capture or silent dummy generator)
        if self.config.audio_enabled and self.config.audio_device_name:
            cmd.extend([
                "-f", "dshow",
                "-i", f"audio={self.config.audio_device_name}",
                "-c:a", "aac",
                "-b:a", "160k",
                "-ar", "48000"
            ])
        else:
            # Generate silent audio stream to keep RTMP players synchronized
            cmd.extend([
                "-f", "lavfi",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-c:a", "aac",
                "-b:a", "128k"
            ])

        # Hardware/Software Video Encoder Flags (OBS Studio Advanced Mode)
        bitrate_k = max(200, self.config.bitrate_kbps)
        maxrate_k = max(bitrate_k, getattr(self.config, 'max_bitrate_kbps', int(bitrate_k * 1.5)))
        bufsize_k = max(bitrate_k * 2, maxrate_k)
        rc_mode = getattr(self.config, 'rate_control', 'vbr').lower()
        preset_val = getattr(self.config, 'preset', 'p4')
        tune_val = getattr(self.config, 'tuning', 'hq')
        multipass_val = getattr(self.config, 'multipass', 'qres')
        profile_val = getattr(self.config, 'profile', 'main')
        b_frames = getattr(self.config, 'b_frames', 2)
        lookahead = getattr(self.config, 'lookahead', True)
        spatial_aq = getattr(self.config, 'adaptive_quant', True)

        codec_str = self.config.codec.lower()
        if "nvenc" in codec_str:
            v_codec = "h264_nvenc" if "h264" in codec_str else "hevc_nvenc"
            cmd.extend([
                "-c:v", v_codec,
                "-preset", preset_val,
                "-tune", tune_val,
                "-profile:v", profile_val,
                "-bf", str(b_frames),
                "-g", str(gop_size),
                "-keyint_min", str(gop_size),
                "-sc_threshold", "0",
                "-pix_fmt", "yuv420p"
            ])

            # Multipass Mode
            if multipass_val in ("qres", "fullres"):
                cmd.extend(["-multipass", multipass_val])
            else:
                cmd.extend(["-multipass", "disabled"])

            # Rate Control Method (VBR / CBR / CQP / Lossless)
            if rc_mode == "cbr":
                cmd.extend([
                    "-rc", "cbr",
                    "-b:v", f"{bitrate_k}k",
                    "-maxrate", f"{bitrate_k}k",
                    "-bufsize", f"{bitrate_k}k"
                ])
            elif rc_mode == "cqp":
                cqp = getattr(self.config, 'cqp_value', 23)
                cmd.extend(["-rc", "constqp", "-qp", str(cqp)])
            elif rc_mode == "lossless":
                cmd.extend(["-tune", "lossless", "-qp", "0"])
            else:  # vbr (Standard OBS)
                cmd.extend([
                    "-rc", "vbr",
                    "-b:v", f"{bitrate_k}k",
                    "-maxrate", f"{maxrate_k}k",
                    "-bufsize", f"{bufsize_k}k"
                ])

            # Lookahead
            if lookahead:
                cmd.extend(["-rc-lookahead", "32"])

            # Adaptive Quantisierung (Psycho Visual Tuning)
            if spatial_aq:
                cmd.extend(["-spatial-aq", "1", "-temporal-aq", "1"])

        elif "qsv" in codec_str:
            cmd.extend([
                "-c:v", "h264_qsv",
                "-preset", "veryfast",
                "-profile:v", profile_val,
                "-b:v", f"{bitrate_k}k",
                "-maxrate", f"{maxrate_k}k",
                "-bufsize", f"{bufsize_k}k",
                "-g", str(gop_size),
                "-bf", str(b_frames),
                "-pix_fmt", "nv12"
            ])
        else:
            # Fallback x264 software CPU encoder
            x264_preset = "medium" if preset_val in ("p4", "p5") else ("fast" if preset_val in ("p2", "p3") else "veryfast")
            cmd.extend([
                "-c:v", "libx264",
                "-preset", x264_preset,
                "-profile:v", profile_val,
                "-bf", str(b_frames),
                "-g", str(gop_size),
                "-keyint_min", str(gop_size),
                "-sc_threshold", "0",
                "-pix_fmt", "yuv420p"
            ])
            if rc_mode == "cbr":
                cmd.extend(["-b:v", f"{bitrate_k}k", "-minrate", f"{bitrate_k}k", "-maxrate", f"{bitrate_k}k", "-bufsize", f"{bitrate_k}k"])
            elif rc_mode == "cqp":
                cqp = getattr(self.config, 'cqp_value', 23)
                cmd.extend(["-crf", str(cqp)])
            else:
                cmd.extend(["-b:v", f"{bitrate_k}k", "-maxrate", f"{maxrate_k}k", "-bufsize", f"{bufsize_k}k"])

        # Rescaling (Ausgabe umskalieren)
        if getattr(self.config, 'rescale_output', False) and getattr(self.config, 'rescale_width', 0) > 0 and getattr(self.config, 'rescale_height', 0) > 0:
            cmd.extend(["-vf", f"scale={self.config.rescale_width}:{self.config.rescale_height}"])

        # Benutzerdefinierte Kodiereroptionen
        custom_opts = getattr(self.config, 'custom_options', '').strip()
        if custom_opts:
            import shlex
            try:
                cmd.extend(shlex.split(custom_opts))
            except Exception as e:
                logger.warning(f"Fehler beim Parsen benutzerdefinierter Optionen '{custom_opts}': {e}")

        # FLV / RTMP Output muxer settings
        cmd.extend([
            "-f", "flv",
            "-flvflags", "no_duration_filesize",
            full_dest
        ])

        logger.info(f"Starte RTMP-Streamer '{self.config.name}': {full_dest} (Encoder: {self.config.codec}, {bitrate_k} kbps)")

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
        except Exception as e:
            logger.error(f"Fehler beim Starten von FFmpeg für RTMP '{self.config.name}': {e}")
            self.status_text = f"Fehler: {e}"
            return

        self.start_time = time.time()
        self.frames_sent = 0
        self.frames_dropped = 0
        self.status_text = "Verbinde mit Server..."

        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name=f"RTMPWriter_{self.config.channel_id}"
        )
        self.writer_thread.start()

        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name=f"RTMPMonitor_{self.config.channel_id}"
        )
        self.monitor_thread.start()

    def _writer_loop(self):
        """Streams BGR frames from internal buffer into FFmpeg stdin."""
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if frame is None or self.proc is None or self.proc.stdin is None:
                break

            try:
                if frame.shape[0] != self.config.height or frame.shape[1] != self.config.width:
                    import cv2
                    frame = cv2.resize(frame, (self.config.width, self.config.height), interpolation=cv2.INTER_LINEAR)

                self.proc.stdin.write(frame.tobytes())
                self.proc.stdin.flush()
                with self._lock:
                    self.frames_sent += 1
            except (BrokenPipeError, OSError) as e:
                logger.warning(f"RTMP-Pipe '{self.config.name}' unterbrochen: {e}")
                with self._lock:
                    self.status_text = "Verbindung getrennt"
                break
            except Exception as e:
                logger.error(f"Schreibfehler bei RTMP '{self.config.name}': {e}")
                break

    def _monitor_loop(self):
        """Parses FFmpeg stderr logs in real time to capture live bitrate, fps, and connection status."""
        regex_fps = re.compile(r"fps=\s*([\d\.]+)")
        regex_bitrate = re.compile(r"bitrate=\s*([\d\.]+)kbits/s")

        if not self.proc or not self.proc.stderr:
            return

        for raw_line in iter(self.proc.stderr.readline, b''):
            if self.stop_event.is_set():
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            m_fps = regex_fps.search(line)
            m_br = regex_bitrate.search(line)

            with self._lock:
                if m_fps:
                    try:
                        self.current_fps = float(m_fps.group(1))
                    except ValueError:
                        pass
                if m_br:
                    try:
                        self.current_bitrate_kbps = float(m_br.group(1))
                    except ValueError:
                        pass
                if "error" in line.lower() or "failed" in line.lower():
                    self.status_text = f"Warnung: {line[:60]}"
                elif self.current_bitrate_kbps > 10.0:
                    self.status_text = "🔴 LIVE"

        with self._lock:
            if not self.stop_event.is_set():
                self.status_text = "Getrennt (Server beendet)"

    def push_frame(self, frame: np.ndarray):
        """Pushes a raw BGR frame to the streaming queue without blocking the caller."""
        if not self.is_running:
            return

        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            # Drop the oldest frame to prevent latency build-up
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            with self._lock:
                self.frames_dropped += 1
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def stop(self):
        self.stop_event.set()
        try:
            self.frame_queue.put_nowait(None)
        except Exception:
            pass

        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

        with self._lock:
            self.status_text = "Gestoppt"
            self.current_fps = 0.0
            self.current_bitrate_kbps = 0.0

    def get_stats(self) -> StreamStats:
        with self._lock:
            is_live = self.is_running and (self.current_bitrate_kbps > 5.0 or self.frames_sent > 10)
            uptime = max(0.0, time.time() - self.start_time) if self.is_running else 0.0
            return StreamStats(
                channel_id=self.config.channel_id,
                is_live=is_live,
                uptime_sec=uptime,
                current_bitrate_kbps=self.current_bitrate_kbps,
                current_fps=self.current_fps,
                frames_sent=self.frames_sent,
                frames_dropped=self.frames_dropped,
                status_text=self.status_text
            )


class MultiStreamManager:
    """
    Central OBS-Style Live-Streaming Manager for MatchTrack-Stitcher.
    Coordinates concurrent streaming of 16:9 Broadcast, Panorama, and camera raw feeds.
    """
    def __init__(self):
        self.streamers: Dict[str, RTMPChannelStreamer] = {}
        self.telemetry_callback: Optional[Callable[[Dict[str, StreamStats]], None]] = None
        self._telemetry_timer: Optional[threading.Thread] = None
        self._stop_telemetry = threading.Event()

    def configure_channel(self, config: RTMPChannelConfig):
        """Adds or updates a streaming channel."""
        if config.channel_id in self.streamers:
            if self.streamers[config.channel_id].is_running:
                self.streamers[config.channel_id].stop()
        self.streamers[config.channel_id] = RTMPChannelStreamer(config)

    def start_all_enabled(self):
        """Starts all enabled RTMP streaming channels."""
        self._stop_telemetry.clear()
        for s in self.streamers.values():
            if s.config.enabled:
                s.start()

        if self._telemetry_timer is None or not self._telemetry_timer.is_alive():
            self._telemetry_timer = threading.Thread(target=self._telemetry_loop, daemon=True, name="StreamTelemetry")
            self._telemetry_timer.start()

    def stop_all(self):
        """Stops all active RTMP streams."""
        self._stop_telemetry.set()
        for s in self.streamers.values():
            s.stop()

    def push_frame(self, channel_id: str, frame: np.ndarray):
        """Routes a frame to the designated channel streamer."""
        streamer = self.streamers.get(channel_id)
        if streamer and streamer.is_running:
            streamer.push_frame(frame)

    def push_feeds(self, feeds: Dict[str, Optional[np.ndarray]]):
        """
        Routes incoming video feeds (e.g. 'broadcast_16x9', 'panorama_8x3', 'cam1', 'cam2')
        to all active channel streamers according to their configured source_feed.
        """
        for streamer in self.streamers.values():
            if not streamer.is_running:
                continue
            feed_name = streamer.config.source_feed
            frame = feeds.get(feed_name)
            # Graceful fallbacks if a specific feed is missing (e.g. cam2 in single-cam mode)
            if frame is None:
                if feed_name == "cam2":
                    frame = feeds.get("cam1")
                elif feed_name == "panorama_8x3":
                    frame = feeds.get("broadcast_16x9")
                elif feed_name == "broadcast_16x9":
                    frame = feeds.get("cam1")
            if frame is not None:
                streamer.push_frame(frame)

    def is_any_live(self) -> bool:
        return any(s.is_running for s in self.streamers.values())

    def get_all_stats(self) -> Dict[str, StreamStats]:
        return {cid: s.get_stats() for cid, s in self.streamers.items()}

    def _telemetry_loop(self):
        while not self._stop_telemetry.is_set():
            stats = self.get_all_stats()
            if self.telemetry_callback:
                try:
                    self.telemetry_callback(stats)
                except Exception:
                    pass
            time.sleep(0.5)

