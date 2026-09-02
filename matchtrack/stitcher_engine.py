"""
Core High-Performance Stitching & Video Render Engine.
Supports frame-accurate seeking, preview generation, and hardware-accelerated (NVENC) batch export.
"""
import os
import time
import subprocess
import threading
import queue
from typing import Optional, Callable, Dict, Any, Tuple
import cv2
import numpy as np
from .rig_geometry import RigConfiguration
from .lut_generator import generate_remap_luts, RemapLUTs, apply_stitch
from .tactical_warp import generate_tactical_16x9_luts, apply_tactical_16x9_warp
from .color_matcher import ColorExposureMatcher
from .ai_tracker import AIBroadcastTracker, BroadcastConfig
from .audio_sync import has_audio_stream
from .paths import get_ffmpeg_path
from .logger import get_logger

logger = get_logger("engine")


class VideoSource:
    """Manages video capture, seeking, and metadata for a single video file."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.cap = cv2.VideoCapture(filepath, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {filepath}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / max(self.fps, 1.0)
        self._lock = threading.Lock()
        logger.info(f"Video geöffnet: '{os.path.basename(filepath)}' | {self.width}x{self.height} @ {self.fps:.2f} FPS | {self.total_frames} Frames ({self.duration_sec:.1f}s)")

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Reads a specific frame safely with locking."""
        with self._lock:
            if frame_number < 0 or frame_number >= self.total_frames:
                return None
            
            curr_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if curr_pos != frame_number:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = self.cap.read()
            if ret:
                return frame
            return None

    def release(self):
        with self._lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()


def get_encoder_flags(codec: str, bitrate_mbps: int) -> list:
    """Returns FFmpeg video encoding parameters based on chosen hardware encoder."""
    ffmpeg_bin = get_ffmpeg_path()
    try:
        res = subprocess.run([ffmpeg_bin, "-hide_banner", "-encoders"], capture_output=True, text=True)
        encoder_list = res.stdout
    except Exception as e:
        logger.warning(f"Konnte FFmpeg Encoder-Liste nicht abfragen: {e}")
        encoder_list = ""

    if "hevc_nvenc" in codec or codec == "hevc_nvenc":
        real_codec = "hevc_nvenc" if "hevc_nvenc" in encoder_list else "libx265"
        if real_codec == "hevc_nvenc":
            if "hq" in codec:
                return ["-c:v", "hevc_nvenc", "-preset", "p6", "-tune", "hq", "-spatial-aq", "1", "-rc", "vbr", "-cq", "18", "-b:v", f"{bitrate_mbps}M"]
            else:
                return ["-c:v", "hevc_nvenc", "-preset", "p4", "-tune", "hq", "-spatial-aq", "0", "-rc", "vbr", "-cq", "22", "-b:v", f"{bitrate_mbps}M", "-maxrate", f"{int(bitrate_mbps * 1.5)}M", "-bufsize", f"{int(bitrate_mbps * 2)}M"]
        else:
            logger.warning("hevc_nvenc nicht verfügbar, wechsle auf libx265 CPU Encoder.")
            return ["-c:v", "libx265", "-preset", "fast", "-crf", "20"]
    elif "h264_nvenc" in codec or codec == "h264_nvenc":
        real_codec = "h264_nvenc" if "h264_nvenc" in encoder_list else "libx264"
        if real_codec == "h264_nvenc":
            if "hq" in codec:
                return ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-spatial-aq", "1", "-rc", "vbr", "-cq", "18", "-b:v", f"{bitrate_mbps}M"]
            else:
                return ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-spatial-aq", "0", "-rc", "vbr", "-cq", "20", "-b:v", f"{bitrate_mbps}M", "-maxrate", f"{int(bitrate_mbps * 1.5)}M", "-bufsize", f"{int(bitrate_mbps * 2)}M"]
        else:
            logger.warning("h264_nvenc nicht verfügbar, wechsle auf libx264 CPU Encoder.")
            return ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]
    elif "qsv" in codec:
        return ["-c:v", "hevc_qsv", "-preset", "medium", "-b:v", f"{bitrate_mbps}M"]
    else:
        return ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]


class StitcherEngine:
    """Main rendering engine for live preview and full-length match rendering."""
    def __init__(self, rig: Optional[RigConfiguration] = None):
        self.rig = rig or RigConfiguration()
        self.video_left: Optional[VideoSource] = None
        self.video_right: Optional[VideoSource] = None
        self.video_panorama: Optional[VideoSource] = None
        self.frame_offset_right: int = 0  # Number of frames Right is shifted relative to Left
        
        # Precomputed LUTs cache
        self.cached_luts: Optional[RemapLUTs] = None
        self.cached_resolution: Tuple[int, int] = (0, 0)
        self.color_matcher = ColorExposureMatcher()
        self.ai_broadcast = AIBroadcastTracker()
        
        # Rendering state
        self._stop_render = threading.Event()

    def is_panorama_mode(self) -> bool:
        """True if operating directly on a single pre-stitched 32:9 panorama video file."""
        return self.video_panorama is not None

    def close(self):
        """Releases all open video capture resources."""
        if self.video_left:
            self.video_left.release()
            self.video_left = None
        if self.video_right:
            self.video_right.release()
            self.video_right = None
        if self.video_panorama:
            self.video_panorama.release()
            self.video_panorama = None

    def load_panorama_video(self, path_pano: str):
        """Loads a single pre-stitched 32:9 panorama video file for direct 16:9 Follow-Cam processing."""
        if self.video_left:
            self.video_left.release()
            self.video_left = None
        if self.video_right:
            self.video_right.release()
            self.video_right = None
        if self.video_panorama:
            self.video_panorama.release()

        self.video_panorama = VideoSource(path_pano)
        self.ai_broadcast.reset()
        logger.info(f"32:9 Panorama-Mastervideo geladen: '{os.path.basename(path_pano)}' ({self.video_panorama.width}x{self.video_panorama.height}, {self.video_panorama.total_frames} Frames)")

    def load_videos(self, path_left: str, path_right: str):
        """Opens left and right camera video files and adapts intrinsics to video resolution."""
        if self.video_panorama:
            self.video_panorama.release()
            self.video_panorama = None
        if self.video_left:
            self.video_left.release()
        if self.video_right:
            self.video_right.release()

        self.video_left = VideoSource(path_left)
        self.video_right = VideoSource(path_right)

        # Automatically update camera intrinsics to match actual video stream dimensions
        if self.video_left.width > 0 and self.video_left.height > 0:
            self.rig.left_camera.set_resolution(self.video_left.width, self.video_left.height)
        if self.video_right.width > 0 and self.video_right.height > 0:
            self.rig.right_camera.set_resolution(self.video_right.width, self.video_right.height)

        self.ai_broadcast.reset()
        self.invalidate_luts()

    def get_max_duration_frames(self) -> int:
        """Returns total playable frames referencing the active video source."""
        if self.video_panorama:
            return self.video_panorama.total_frames
        if self.video_left:
            return self.video_left.total_frames
        return 0

    def get_fps(self) -> float:
        """Returns the frame rate of the loaded video source."""
        if self.video_panorama:
            return self.video_panorama.fps
        if self.video_left:
            return self.video_left.fps
        return 30.0

    def update_luts(self, out_width: int, out_height: int) -> RemapLUTs:
        """Generates or retrieves cached LookUp Tables for the given resolution."""
        if (self.cached_luts is None or 
            self.cached_resolution != (out_width, out_height)):
            self.cached_luts = generate_remap_luts(self.rig, out_width, out_height)
            self.cached_resolution = (out_width, out_height)
        return self.cached_luts

    def invalidate_luts(self):
        """Forces recalculation of LUTs when rig parameters change."""
        self.cached_luts = None

    def render_preview_frame(self, frame_index: int, preview_width: int = 2560, preview_height: int = 720, view_mode: str = "32:9", tactical_mode: str = "points") -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int], Optional[Tuple[int, int]], float]:
        """
        Renders or extracts a panorama frame and returns:
        (stitched_frame, (crop_x, crop_y, crop_w, crop_h), (ball_x, ball_y), current_zoom).
        Supports view_mode='16:9_tactical' with tactical_mode='points' (interactive setup) or 'full' (fullscreen output).
        """
        # Mode 1: Single 32:9 Panorama Video
        if self.video_panorama is not None:
            raw_frame = self.video_panorama.get_frame(frame_index)
            if raw_frame is None:
                return None, (0, 0, 0, 0), None, 1.0

            if (raw_frame.shape[1], raw_frame.shape[0]) != (preview_width, preview_height):
                pano_frame = cv2.resize(raw_frame, (preview_width, preview_height), interpolation=cv2.INTER_LINEAR)
            else:
                pano_frame = raw_frame

            tx, ty, tz, ball_found, ball_px = self.ai_broadcast.detect_action(pano_frame)
            self.ai_broadcast.update_camera(tx, ty, tz)
            crop_box = self.ai_broadcast.get_crop_rect(preview_width, preview_height)

            return pano_frame, crop_box, ball_px, self.ai_broadcast.cam_zoom

        # Mode 2: 2-Camera Live Stitching Rig
        if not self.video_left or not self.video_right:
            return None, (0, 0, 0, 0), None, 1.0

        idx_l = frame_index
        idx_r = frame_index - self.frame_offset_right

        frame_l = self.video_left.get_frame(idx_l)
        frame_r = self.video_right.get_frame(idx_r)

        if frame_l is None and frame_r is None:
            return None, (0, 0, 0, 0), None, 1.0

        if frame_l is None:
            frame_l = np.zeros((self.video_right.height, self.video_right.width, 3), dtype=np.uint8)
        if frame_r is None:
            frame_r = np.zeros((self.video_left.height, self.video_left.width, 3), dtype=np.uint8)

        luts = self.update_luts(preview_width, preview_height)
        stitched = apply_stitch(frame_l, frame_r, luts)

        # AI Ball & Player Detection + Dynamic Auto-Zoom
        tx, ty, tz, ball_found, ball_px = self.ai_broadcast.detect_action(stitched)
        self.ai_broadcast.update_camera(tx, ty, tz)
        crop_box = self.ai_broadcast.get_crop_rect(preview_width, preview_height)

        return stitched, crop_box, ball_px, self.ai_broadcast.cam_zoom

    def render_video_to_file(self, 
                             output_filepath: str, 
                             out_width: int = 3840, 
                             out_height: int = 1080,
                             mode: str = "32:9",
                             codec: str = "hevc_nvenc",
                             bitrate_mbps: int = 50,
                             start_frame: int = 0,
                             end_frame: Optional[int] = None,
                             audio_source: str = "left",
                             progress_callback: Optional[Callable[[int, int, float, float], None]] = None) -> bool:
        """
        Hardware-accelerated multi-threaded batch render pipeline.
        Uses non-blocking parallel reader, multi-threaded worker pool, and live FFmpeg error logging.
        Strictly renders only from start_frame up to end_frame with synchronized audio track.
        """
        # If in Standalone Panorama mode, delegate to direct panorama processor
        if self.is_panorama_mode():
            def cb_wrapper(p, t, f, eta, stage_text=""):
                if progress_callback:
                    progress_callback(p, t, f, eta)

            if mode == "21:10":
                return self.convert_panorama_to_21x10(
                    output_filepath=output_filepath,
                    out_width=out_width,
                    out_height=out_height,
                    codec=codec,
                    bitrate_mbps=bitrate_mbps,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    progress_callback=cb_wrapper
                )
            elif mode == "16:9_autocam":
                return self.render_broadcast_from_panorama(
                    output_filepath=output_filepath,
                    out_width=out_width,
                    out_height=out_height,
                    codec=codec,
                    bitrate_mbps=bitrate_mbps,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    use_lookahead=True,
                    progress_callback=cb_wrapper
                )
            elif mode == "16:9_tactical":
                return self.convert_panorama_to_16x9_tactical(
                    output_filepath=output_filepath,
                    out_width=out_width,
                    out_height=out_height,
                    codec=codec,
                    bitrate_mbps=bitrate_mbps,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    progress_callback=cb_wrapper
                )
            else:
                return self.convert_panorama_to_21x10(
                    output_filepath=output_filepath,
                    out_width=out_width,
                    out_height=out_height,
                    codec=codec,
                    bitrate_mbps=bitrate_mbps,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    progress_callback=cb_wrapper
                )

        if not self.video_left or not self.video_right:
            logger.error("Render-Start abgebrochen: Quellvideos nicht geladen.")
            return False

        self._stop_render.clear()
        fps = self.video_left.fps
        total_avail = self.get_max_duration_frames()
        
        start_frame = max(0, start_frame)
        if end_frame is None or end_frame > total_avail:
            end_frame = total_avail
        
        frames_to_process = max(0, end_frame - start_frame)
        if frames_to_process == 0:
            logger.warning(f"Keine Frames zu rendern: start_frame={start_frame}, end_frame={end_frame}")
            return False

        logger.info(f"=== Render-Export gestartet ===")
        logger.info(f"Ziel-Datei:        {output_filepath}")
        logger.info(f"Modus / Format:    {mode} ({out_width}x{out_height} @ {fps:.2f} FPS)")
        logger.info(f"Frame-Bereich:     {start_frame} bis {end_frame} (Gesamt: {frames_to_process} Frames)")
        logger.info(f"Encoder / Bitrate: {codec} @ {bitrate_mbps} Mbps")

        tactical_map_x, tactical_map_y = None, None
        # Intermediate stitch resolution for 16:9 modes
        if mode == "16:9_autocam":
            inter_h = max(out_height, 1440)
            inter_w = int(round(inter_h * (32.0 / 9.0)))
            luts = self.update_luts(inter_w, inter_h)
            self.ai_broadcast.reset()
            logger.info(f"16:9 AutoCam Zwischenauflösung: {inter_w}x{inter_h}")
        elif mode == "16:9_tactical":
            inter_h = max(out_height, 1080)
            inter_w = int(round(inter_h * (32.0 / 9.0)))
            luts = self.update_luts(inter_w, inter_h)
            tactical_map_x, tactical_map_y = generate_tactical_16x9_luts(
                inter_w, inter_h,
                out_width, out_height,
                self.ai_broadcast.config.pitch_corners,
                margin_percent=getattr(self.ai_broadcast.config, 'tactical_margin', 0.0)
            )
            logger.info(f"16:9 Taktik-Warp Zwischenauflösung: {inter_w}x{inter_h} -> Ziel: {out_width}x{out_height}")
        else:
            luts = self.update_luts(out_width, out_height)

        # Get hardware profile encoder flags
        codec_flags = get_encoder_flags(codec, bitrate_mbps)

        # Open dedicated capture streams with FFmpeg backend
        cap_l = cv2.VideoCapture(self.video_left.filepath, cv2.CAP_FFMPEG)
        if not cap_l.isOpened():
            cap_l = cv2.VideoCapture(self.video_left.filepath)

        cap_r = cv2.VideoCapture(self.video_right.filepath, cv2.CAP_FFMPEG)
        if not cap_r.isOpened():
            cap_r = cv2.VideoCapture(self.video_right.filepath)

        # Seek once at start_frame
        cap_l.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        right_start_idx = start_frame - self.frame_offset_right
        if 0 < right_start_idx < self.video_right.total_frames:
            cap_r.set(cv2.CAP_PROP_POS_FRAMES, right_start_idx)

        input_queue = queue.Queue(maxsize=16)
        output_dict = {}
        output_lock = threading.Lock()
        output_cond = threading.Condition(output_lock)

        def reader_thread():
            logger.debug("Reader-Thread gestartet.")
            try:
                for f_idx in range(start_frame, end_frame):
                    if self._stop_render.is_set():
                        break
                    
                    ret_l, frame_l = cap_l.read()
                    if not ret_l:
                        frame_l = None
                        if f_idx < end_frame - 1:
                            logger.warning(f"Kamera Links: Kein Frame bei Index {f_idx} (EOF oder Lesefehler)")
                    
                    r_idx = f_idx - self.frame_offset_right
                    if 0 <= r_idx < self.video_right.total_frames:
                        ret_r, frame_r = cap_r.read()
                        if not ret_r:
                            frame_r = None
                            logger.warning(f"Kamera Rechts: Kein Frame bei Index {r_idx}")
                    else:
                        frame_r = None
                    
                    # Push to worker queue (blocks if queue full)
                    while not self._stop_render.is_set():
                        try:
                            input_queue.put((f_idx, frame_l, frame_r), timeout=0.2)
                            break
                        except queue.Full:
                            continue
            except Exception as e:
                logger.error(f"Unerwarteter Fehler im Reader-Thread: {e}", exc_info=True)
            finally:
                # Sentinel to signal worker termination to all workers
                for _ in range(num_workers):
                    while not self._stop_render.is_set():
                        try:
                            input_queue.put(None, timeout=0.2)
                            break
                        except queue.Full:
                            continue
                logger.debug("Reader-Thread beendet.")

        def worker_thread(worker_id: int):
            logger.debug(f"Worker-Thread {worker_id} gestartet.")
            while not self._stop_render.is_set():
                try:
                    item = input_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if item is None:
                    break

                f_idx, fl, fr = item
                try:
                    if fl is None and fr is None:
                        stitched = np.zeros((luts.out_height, luts.out_width, 3), dtype=np.uint8)
                    elif fl is None:
                        fl = np.zeros((self.video_right.height, self.video_right.width, 3), dtype=np.uint8)
                        stitched = apply_stitch(fl, fr, luts)
                    elif fr is None:
                        fr = np.zeros((self.video_left.height, self.video_left.width, 3), dtype=np.uint8)
                        stitched = apply_stitch(fl, fr, luts)
                    else:
                        stitched = apply_stitch(fl, fr, luts)
                except Exception as e:
                    logger.error(f"Fehler beim Stitchen von Frame {f_idx} (Worker {worker_id}): {e}", exc_info=True)
                    stitched = np.zeros((luts.out_height, luts.out_width, 3), dtype=np.uint8)

                # Unconditionally insert into output_dict and notify consumer
                # (Backpressure is maintained upstream via input_queue maxsize)
                with output_cond:
                    output_dict[f_idx] = stitched
                    output_cond.notify_all()

            logger.debug(f"Worker-Thread {worker_id} beendet.")

        num_workers = min(8, max(2, (os.cpu_count() or 4) - 2))
        logger.info(f"Starte {num_workers} parallele Worker-Threads für Bildberechnung...")
        t_reader = threading.Thread(target=reader_thread, daemon=True, name="FrameReader")
        t_workers = [threading.Thread(target=worker_thread, args=(i,), daemon=True, name=f"StitchWorker-{i}") for i in range(num_workers)]

        t_reader.start()
        for tw in t_workers:
            tw.start()

        # Configure audio source and timing
        duration_sec = frames_to_process / max(fps, 1.0)
        t_start_l = start_frame / max(fps, 1.0)
        t_start_r = (start_frame - self.frame_offset_right) / max(fps, 1.0)

        has_audio_l = has_audio_stream(self.video_left.filepath) if self.video_left else False
        has_audio_r = has_audio_stream(self.video_right.filepath) if self.video_right else False

        effective_audio_source = (audio_source or "none").lower()
        if effective_audio_source == "mix":
            if not has_audio_l and not has_audio_r:
                logger.warning("Weder linke noch rechte Kamera besitzt eine Audiospur. Exportiere ohne Audio.")
                effective_audio_source = "none"
            elif has_audio_l and not has_audio_r:
                logger.warning("Rechte Kamera hat keine Audiospur. Verwende Audiospur der linken Kamera.")
                effective_audio_source = "left"
            elif not has_audio_l and has_audio_r:
                logger.warning("Linke Kamera hat keine Audiospur. Verwende Audiospur der rechten Kamera.")
                effective_audio_source = "right"
        elif effective_audio_source == "left" and not has_audio_l:
            logger.warning("Linke Kamera hat keine Audiospur. Exportiere ohne Audio.")
            effective_audio_source = "none"
        elif effective_audio_source == "right" and not has_audio_r:
            logger.warning("Rechte Kamera hat keine Audiospur. Exportiere ohne Audio.")
            effective_audio_source = "none"

        audio_inputs = []
        audio_mapping = []
        audio_codec_flags = []

        if effective_audio_source == "left":
            audio_inputs = [
                "-ss", f"{t_start_l:.4f}",
                "-t", f"{duration_sec:.4f}",
                "-i", self.video_left.filepath
            ]
            audio_mapping = ["-map", "0:v:0", "-map", "1:a:0"]
            audio_codec_flags = ["-c:a", "aac", "-b:a", "192k"]
            logger.info(f"Audio-Quelle: Linke Kamera / Video A (Start: {t_start_l:.2f}s, Dauer: {duration_sec:.2f}s)")
        elif effective_audio_source == "right":
            if t_start_r >= 0:
                audio_inputs = [
                    "-ss", f"{t_start_r:.4f}",
                    "-t", f"{duration_sec:.4f}",
                    "-i", self.video_right.filepath
                ]
                audio_mapping = ["-map", "0:v:0", "-map", "1:a:0"]
                audio_codec_flags = ["-c:a", "aac", "-b:a", "192k"]
            else:
                # Right camera started later than export start: delay audio by delta_t
                delay_ms = int(round((-t_start_r) * 1000))
                audio_inputs = ["-i", self.video_right.filepath]
                filter_complex = f"[1:a]adelay=delays={delay_ms}|{delay_ms}:all=1,atrim=0:{duration_sec:.4f},asetpts=PTS-STARTPTS[aout]"
                audio_mapping = ["-filter_complex", filter_complex, "-map", "0:v:0", "-map", "[aout]"]
                audio_codec_flags = ["-c:a", "aac", "-b:a", "192k"]
            logger.info(f"Audio-Quelle: Rechte Kamera / Video B (Offset: {self.frame_offset_right} Frames, Start: {t_start_r:.2f}s, Dauer: {duration_sec:.2f}s)")
        elif effective_audio_source == "mix":
            audio_inputs = [
                "-i", self.video_left.filepath,
                "-i", self.video_right.filepath
            ]
            left_filter = f"[1:a]atrim=start={t_start_l:.4f}:duration={duration_sec:.4f},asetpts=PTS-STARTPTS[al]"
            if t_start_r >= 0:
                right_filter = f"[2:a]atrim=start={t_start_r:.4f}:duration={duration_sec:.4f},asetpts=PTS-STARTPTS[ar]"
            else:
                delay_ms = int(round((-t_start_r) * 1000))
                right_filter = f"[2:a]adelay=delays={delay_ms}|{delay_ms}:all=1,atrim=0:{duration_sec:.4f},asetpts=PTS-STARTPTS[ar]"
            filter_complex = f"{left_filter};{right_filter};[al][ar]amix=inputs=2:duration=first:dropout_transition=0[aout]"
            audio_mapping = ["-filter_complex", filter_complex, "-map", "0:v:0", "-map", "[aout]"]
            audio_codec_flags = ["-c:a", "aac", "-b:a", "192k"]
            logger.info(f"Audio-Quelle: Mix (Links + Rechts synchronisiert)")
        else:
            audio_inputs = []
            audio_mapping = ["-map", "0:v:0", "-an"]
            audio_codec_flags = []
            logger.info("Audio-Quelle: Deaktiviert (Stumm / Kein Audio)")

        # Build FFmpeg command
        ffmpeg_bin = get_ffmpeg_path()
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{out_width}x{out_height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
        ]
        ffmpeg_cmd.extend(audio_inputs)
        ffmpeg_cmd.extend(audio_mapping)
        ffmpeg_cmd.extend(codec_flags)
        ffmpeg_cmd.extend(audio_codec_flags)
        ffmpeg_cmd.extend(["-pix_fmt", "yuv420p", "-shortest", output_filepath])

        logger.info(f"FFmpeg Befehl: {' '.join(ffmpeg_cmd)}")
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        # Background thread to capture and log FFmpeg stderr in real time
        ffmpeg_stderr_lines = []
        def monitor_ffmpeg_stderr():
            try:
                for raw_line in iter(proc.stderr.readline, b''):
                    decoded = raw_line.decode('utf-8', errors='replace').strip()
                    if decoded:
                        ffmpeg_stderr_lines.append(decoded)
                        if len(ffmpeg_stderr_lines) > 100:
                            ffmpeg_stderr_lines.pop(0)
                        if any(w in decoded.lower() for w in ["error", "fatal", "failed", "invalid", "overflow", "nvenc", "cannot"]):
                            logger.warning(f"[FFmpeg] {decoded}")
                        else:
                            logger.debug(f"[FFmpeg] {decoded}")
            except Exception:
                pass

        t_ffmpeg_monitor = threading.Thread(target=monitor_ffmpeg_stderr, daemon=True, name="FFmpegMonitor")
        t_ffmpeg_monitor.start()

        start_time = time.time()
        processed = 0
        render_success = False
        last_progress_time = time.time()
        stalled_warning_logged = False

        try:
            for f_idx in range(start_frame, end_frame):
                if self._stop_render.is_set():
                    logger.info("Render-Vorgang wurde abgebrochen.")
                    break

                # Retrieve stitched frame strictly in order
                with output_cond:
                    while f_idx not in output_dict and not self._stop_render.is_set():
                        # 1. Check if FFmpeg has crashed
                        if proc.poll() is not None:
                            err_msg = "\n".join(ffmpeg_stderr_lines[-8:]) if ffmpeg_stderr_lines else "(keine Meldung)"
                            logger.error(f"FFmpeg wurde unerwartet beendet (Exit-Code {proc.poll()}) bei Frame {f_idx}!\nFFmpeg-Fehlerausgabe:\n{err_msg}")
                            self._stop_render.set()
                            break

                        # 2. Check if all worker threads and reader died
                        if not t_reader.is_alive() and all(not tw.is_alive() for tw in t_workers) and f_idx not in output_dict:
                            logger.error(f"Pipeline-Stillstand: Reader und alle Worker beendet, Frame {f_idx} fehlt!")
                            self._stop_render.set()
                            break

                        # 3. Watchdog check
                        wait_elapsed = time.time() - last_progress_time
                        if wait_elapsed > 10.0 and not stalled_warning_logged:
                            logger.warning(f"Warte auf Frame {f_idx} seit {wait_elapsed:.1f}s... (Puffer: {len(output_dict)} Frames, Input-Queue: {input_queue.qsize()}, Reader aktiv: {t_reader.is_alive()}, Worker aktiv: {sum(tw.is_alive() for tw in t_workers)}/{len(t_workers)})")
                            stalled_warning_logged = True

                        if wait_elapsed > 60.0:
                            logger.error(f"Render-Timeout: 60s keine neuen Frames bei Frame {f_idx}. Breche ab.")
                            self._stop_render.set()
                            break

                        output_cond.wait(timeout=0.2)

                    if self._stop_render.is_set():
                        break
                    stitched = output_dict.pop(f_idx)
                    output_cond.notify_all()

                last_progress_time = time.time()
                stalled_warning_logged = False

                if mode == "16:9_autocam":
                    try:
                        final_frame = self.ai_broadcast.extract_16x9_frame(stitched, out_width, out_height)
                    except Exception as e:
                        logger.error(f"Fehler in AI Broadcast Extraktion bei Frame {f_idx}: {e}", exc_info=True)
                        final_frame = cv2.resize(stitched, (out_width, out_height))
                elif mode == "16:9_tactical":
                    try:
                        final_frame = apply_tactical_16x9_warp(stitched, tactical_map_x, tactical_map_y)
                    except Exception as e:
                        logger.error(f"Fehler in 16:9 Taktik-Warp bei Frame {f_idx}: {e}", exc_info=True)
                        final_frame = cv2.resize(stitched, (out_width, out_height))
                else:
                    final_frame = stitched

                try:
                    proc.stdin.write(final_frame.tobytes())
                except (BrokenPipeError, OSError) as e:
                    err_msg = "\n".join(ffmpeg_stderr_lines[-8:]) if ffmpeg_stderr_lines else "(keine Meldung)"
                    logger.error(f"Schreibfehler auf FFmpeg-Pipe bei Frame {f_idx}: {e}\nLetzte FFmpeg-Meldungen:\n{err_msg}")
                    self._stop_render.set()
                    break
                finally:
                    del stitched
                    if 'final_frame' in locals():
                        del final_frame

                processed += 1

                if progress_callback and (processed % 15 == 0 or processed == frames_to_process):
                    elapsed = time.time() - start_time
                    current_fps = processed / max(elapsed, 0.001)
                    eta_sec = (frames_to_process - processed) / max(current_fps, 0.001)
                    progress_callback(processed, frames_to_process, current_fps, eta_sec)

                if processed % 150 == 0 or processed == frames_to_process:
                    elapsed = time.time() - start_time
                    current_fps = processed / max(elapsed, 0.001)
                    eta_sec = (frames_to_process - processed) / max(current_fps, 0.001)
                    m_eta, s_eta = divmod(int(eta_sec), 60)
                    pct = int((processed / max(frames_to_process, 1)) * 100)
                    logger.info(f"Render-Status: {processed}/{frames_to_process} ({pct}%) | {current_fps:.1f} FPS | Restzeit: {m_eta:02d}:{s_eta:02d}")

            if not self._stop_render.is_set() and processed == frames_to_process:
                render_success = True
                logger.info(f"🎉 Rendering erfolgreich abgeschlossen! ({processed} Frames in {time.time()-start_time:.1f}s)")

        except Exception as e:
            logger.error(f"Schwerer Ausnahmefehler während des Renderns: {e}", exc_info=True)
            render_success = False
        finally:
            if not render_success:
                self._stop_render.set()
            with output_cond:
                output_dict.clear()
                output_cond.notify_all()

            # Drain input queue
            while not input_queue.empty():
                try:
                    input_queue.get_nowait()
                except Exception:
                    break

            if proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            proc.wait()
            cap_l.release()
            cap_r.release()
            t_reader.join(timeout=1.0)
            for tw in t_workers:
                tw.join(timeout=0.5)

            if render_success:
                self._stop_render.clear()

        return render_success

    def render_broadcast_from_panorama(self,
                                       output_filepath: str,
                                       source_filepath: Optional[str] = None,
                                       out_width: int = 1920,
                                       out_height: int = 1080,
                                       codec: str = "hevc_nvenc",
                                       bitrate_mbps: int = 50,
                                       start_frame: int = 0,
                                       end_frame: Optional[int] = None,
                                       use_lookahead: bool = True,
                                       progress_callback: Optional[Callable[[int, int, float, float, str], None]] = None) -> bool:
        """
        Ultra-fast direct 16:9 Follow-Cam render from an existing 32:9 panorama video file.
        Stage 1 (Lookahead): Offline trajectory scan & bidirectional Gaussian smoothing.
        Stage 2: High-speed frame extraction, cropping, and hardware-accelerated NVENC encoding.
        """
        src_path = source_filepath or (self.video_panorama.filepath if self.video_panorama else None)
        if not src_path or not os.path.exists(src_path):
            logger.error("Render-Start abgebrochen: Kein 32:9 Panorama-Quellvideo angegeben.")
            return False

        self._stop_render.clear()
        
        cap_info = cv2.VideoCapture(src_path, cv2.CAP_FFMPEG)
        if not cap_info.isOpened():
            cap_info = cv2.VideoCapture(src_path)
        if not cap_info.isOpened():
            logger.error(f"Konnte Quellvideo nicht öffnen: {src_path}")
            return False

        fps = float(cap_info.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        pano_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        pano_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_info.release()

        start_frame = max(0, start_frame)
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames

        frames_to_process = max(0, end_frame - start_frame)
        if frames_to_process == 0:
            logger.warning(f"Keine Frames zu rendern: start_frame={start_frame}, end_frame={end_frame}")
            return False

        logger.info(f"=== Direkter 16:9 Follow-Cam Export gestartet ===")
        logger.info(f"Quelle (32:9):     {src_path} ({pano_w}x{pano_h} @ {fps:.2f} FPS)")
        logger.info(f"Ziel-Datei (16:9): {output_filepath} ({out_width}x{out_height})")
        logger.info(f"Frame-Bereich:     {start_frame} bis {end_frame} (Gesamt: {frames_to_process} Frames)")
        logger.info(f"Encoder / Bitrate: {codec} @ {bitrate_mbps} Mbps")
        logger.info(f"Lookahead-Modus:   {'Aktiviert (2-Pass Filmisch)' if use_lookahead else 'Deaktiviert (1-Pass)'}")

        # Stage 1: Precompute Trajectory if Lookahead is enabled
        if use_lookahead:
            def traj_cb(curr, total, speed):
                if progress_callback:
                    eta = (total - curr) / max(speed, 0.001)
                    progress_callback(curr, total, speed, eta, "Stufe 1/2: KI-Trajektorie (Lookahead)...")

            step_val = getattr(self.ai_broadcast.config, "scan_step", 5)
            self.ai_broadcast.generate_offline_trajectory(
                video_source=src_path,
                start_frame=start_frame,
                end_frame=end_frame,
                step=step_val,
                lookahead_window_sec=1.8,
                stop_event=self._stop_render,
                progress_callback=traj_cb
            )

            if self._stop_render.is_set():
                logger.info("Export während der Trajektorienberechnung abgebrochen.")
                return False

        # Stage 2: Hardware-Accelerated Video Crop & NVENC Encode
        ffmpeg_bin = get_ffmpeg_path()
        codec_flags = get_encoder_flags(codec, bitrate_mbps)

        # Audio configuration: pass through / sync from 32:9 source
        duration_sec = frames_to_process / max(fps, 1.0)
        t_start = start_frame / max(fps, 1.0)
        has_audio = has_audio_stream(src_path)

        if has_audio:
            audio_inputs = ["-ss", f"{t_start:.4f}", "-t", f"{duration_sec:.4f}", "-i", src_path]
            audio_mapping = ["-map", "0:v:0", "-map", "1:a:0"]
            audio_codec_flags = ["-c:a", "aac", "-b:a", "192k"]
        else:
            audio_inputs = []
            audio_mapping = ["-map", "0:v:0", "-an"]
            audio_codec_flags = []

        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{out_width}x{out_height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
        ]
        ffmpeg_cmd.extend(audio_inputs)
        ffmpeg_cmd.extend(audio_mapping)
        ffmpeg_cmd.extend(codec_flags)
        ffmpeg_cmd.extend(audio_codec_flags)
        ffmpeg_cmd.extend(["-pix_fmt", "yuv420p", "-shortest", output_filepath])

        logger.info(f"FFmpeg Befehl: {' '.join(ffmpeg_cmd)}")
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        ffmpeg_stderr_lines = []
        def monitor_ffmpeg():
            try:
                for raw_line in iter(proc.stderr.readline, b''):
                    decoded = raw_line.decode('utf-8', errors='replace').strip()
                    if decoded:
                        ffmpeg_stderr_lines.append(decoded)
                        if len(ffmpeg_stderr_lines) > 100:
                            ffmpeg_stderr_lines.pop(0)
                        if any(w in decoded.lower() for w in ["error", "fatal", "failed", "cannot"]):
                            logger.warning(f"[FFmpeg] {decoded}")
            except Exception:
                pass

        t_mon = threading.Thread(target=monitor_ffmpeg, daemon=True, name="FFmpegMonitor")
        t_mon.start()

        cap = cv2.VideoCapture(src_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        start_time = time.time()
        processed = 0
        render_success = False

        try:
            for f_idx in range(start_frame, end_frame):
                if self._stop_render.is_set():
                    break

                ret, pano_frame = cap.read()
                if not ret or pano_frame is None:
                    logger.warning(f"Kein Frame bei Index {f_idx} aus 32:9 Video.")
                    break

                if use_lookahead:
                    final_frame = self.ai_broadcast.extract_16x9_frame_with_trajectory(pano_frame, f_idx, out_width, out_height)
                else:
                    final_frame = self.ai_broadcast.extract_16x9_frame(pano_frame, out_width, out_height, frame_index=f_idx)

                try:
                    proc.stdin.write(final_frame.tobytes())
                except (BrokenPipeError, OSError) as e:
                    logger.error(f"Schreibfehler auf FFmpeg-Pipe bei Frame {f_idx}: {e}")
                    break

                processed += 1

                if progress_callback and (processed % 15 == 0 or processed == frames_to_process):
                    elapsed = time.time() - start_time
                    current_fps = processed / max(elapsed, 0.001)
                    eta_sec = (frames_to_process - processed) / max(current_fps, 0.001)
                    progress_callback(processed, frames_to_process, current_fps, eta_sec, "Stufe 2/2: 16:9 Follow-Cam Rendering...")

            if not self._stop_render.is_set() and processed == frames_to_process:
                render_success = True
                logger.info(f"🎉 16:9 Follow-Cam erfolgreich gerendert! ({processed} Frames in {time.time()-start_time:.1f}s)")
        except Exception as e:
            logger.error(f"Fehler beim 16:9 Broadcast Rendering: {e}", exc_info=True)
            render_success = False
        finally:
            self._stop_render.set()
            if proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            proc.wait()
            cap.release()

        return render_success

    def convert_panorama_to_21x10(self,
                                  output_filepath: str,
                                  source_filepath: Optional[str] = None,
                                  out_width: int = 2520,
                                  out_height: int = 1200,
                                  codec: str = "hevc_nvenc",
                                  bitrate_mbps: int = 50,
                                  start_frame: int = 0,
                                  end_frame: Optional[int] = None,
                                  progress_callback: Optional[Callable[[int, int, float, float, str], None]] = None) -> bool:
        """
        Ultra-fast direct 32:9 to 21:10 Panorama conversion with even horizontal squeezing.
        Reads 32:9 panorama master, evenly resizes/squeezes each frame to (out_width, out_height),
        and encodes with hardware-accelerated NVENC/QuickSync/CPU FFmpeg pipe with synchronized audio.
        """
        src_path = source_filepath or (self.video_panorama.filepath if self.video_panorama else None)
        if not src_path or not os.path.exists(src_path):
            logger.error("Konvertierungs-Start abgebrochen: Kein Panorama-Quellvideo angegeben.")
            return False

        self._stop_render.clear()

        cap_info = cv2.VideoCapture(src_path, cv2.CAP_FFMPEG)
        if not cap_info.isOpened():
            cap_info = cv2.VideoCapture(src_path)
        if not cap_info.isOpened():
            logger.error(f"Konnte Quellvideo nicht öffnen: {src_path}")
            return False

        fps = float(cap_info.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        pano_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        pano_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_info.release()

        start_frame = max(0, start_frame)
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames

        frames_to_process = max(0, end_frame - start_frame)
        if frames_to_process == 0:
            logger.warning(f"Keine Frames zu rendern: start_frame={start_frame}, end_frame={end_frame}")
            return False

        logger.info(f"=== 32:9 zu 21:10 Panorama-Wandlung gestartet ===")
        logger.info(f"Quelle (32:9):     {src_path} ({pano_w}x{pano_h} @ {fps:.2f} FPS)")
        logger.info(f"Ziel-Datei (21:10): {output_filepath} ({out_width}x{out_height})")
        logger.info(f"Frame-Bereich:     {start_frame} bis {end_frame} (Gesamt: {frames_to_process} Frames)")
        logger.info(f"Encoder / Bitrate: {codec} @ {bitrate_mbps} Mbps")

        ffmpeg_bin = get_ffmpeg_path()
        codec_flags = get_encoder_flags(codec, bitrate_mbps)

        duration_sec = frames_to_process / max(fps, 1.0)
        t_start = start_frame / max(fps, 1.0)
        has_audio = has_audio_stream(src_path)

        if has_audio:
            audio_inputs = ["-ss", f"{t_start:.4f}", "-t", f"{duration_sec:.4f}", "-i", src_path]
            audio_mapping = ["-map", "0:v:0", "-map", "1:a:0"]
            audio_codec_flags = ["-c:a", "aac", "-b:a", "192k"]
        else:
            audio_inputs = []
            audio_mapping = ["-map", "0:v:0", "-an"]
            audio_codec_flags = []

        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{out_width}x{out_height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
        ]
        ffmpeg_cmd.extend(audio_inputs)
        ffmpeg_cmd.extend(audio_mapping)
        ffmpeg_cmd.extend(codec_flags)
        ffmpeg_cmd.extend(audio_codec_flags)
        ffmpeg_cmd.extend(["-pix_fmt", "yuv420p", "-shortest", output_filepath])

        logger.info(f"FFmpeg Befehl: {' '.join(ffmpeg_cmd)}")
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        ffmpeg_stderr_lines = []
        def monitor_ffmpeg():
            try:
                for raw_line in iter(proc.stderr.readline, b''):
                    decoded = raw_line.decode('utf-8', errors='replace').strip()
                    if decoded:
                        ffmpeg_stderr_lines.append(decoded)
                        if len(ffmpeg_stderr_lines) > 100:
                            ffmpeg_stderr_lines.pop(0)
                        if any(w in decoded.lower() for w in ["error", "fatal", "failed", "cannot"]):
                            logger.warning(f"[FFmpeg] {decoded}")
            except Exception:
                pass

        t_mon = threading.Thread(target=monitor_ffmpeg, daemon=True, name="FFmpegMonitor")
        t_mon.start()

        cap = cv2.VideoCapture(src_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        start_time = time.time()
        processed = 0
        render_success = False

        try:
            for f_idx in range(start_frame, end_frame):
                if self._stop_render.is_set():
                    break

                ret, pano_frame = cap.read()
                if not ret or pano_frame is None:
                    logger.warning(f"Kein Frame bei Index {f_idx} aus 32:9 Video.")
                    break

                if (pano_frame.shape[1], pano_frame.shape[0]) != (out_width, out_height):
                    final_frame = cv2.resize(pano_frame, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
                else:
                    final_frame = pano_frame

                try:
                    proc.stdin.write(final_frame.tobytes())
                except (BrokenPipeError, OSError) as e:
                    logger.error(f"Schreibfehler auf FFmpeg-Pipe bei Frame {f_idx}: {e}")
                    break

                processed += 1

                if progress_callback and (processed % 15 == 0 or processed == frames_to_process):
                    elapsed = time.time() - start_time
                    current_fps = processed / max(elapsed, 0.001)
                    eta_sec = (frames_to_process - processed) / max(current_fps, 0.001)
                    progress_callback(processed, frames_to_process, current_fps, eta_sec, "21:10 Konvertierung...")

            if not self._stop_render.is_set() and processed == frames_to_process:
                render_success = True
                logger.info(f"🎉 21:10 Panorama erfolgreich konvertiert! ({processed} Frames in {time.time()-start_time:.1f}s)")
        except Exception as e:
            logger.error(f"Fehler bei der 21:10 Panorama-Konvertierung: {e}", exc_info=True)
            render_success = False
        finally:
            self._stop_render.set()
            if proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            proc.wait()
            cap.release()

        return render_success

    def convert_panorama_to_16x9_tactical(self,
                                          output_filepath: str,
                                          source_filepath: Optional[str] = None,
                                          out_width: int = 1920,
                                          out_height: int = 1080,
                                          codec: str = "hevc_nvenc",
                                          bitrate_mbps: int = 50,
                                          start_frame: int = 0,
                                          end_frame: Optional[int] = None,
                                          progress_callback: Optional[Callable[[int, int, float, float, str], None]] = None) -> bool:
        """
        Transforms a wide 32:9 panorama video into a planar 16:9 full-pitch tactical overview
        using 6-point dual-quad mesh homography without any virtual camera panning.
        """
        src_path = source_filepath or (self.video_panorama.filepath if self.video_panorama else None)
        if not src_path or not os.path.exists(src_path):
            logger.error("Konvertierungs-Start abgebrochen: Kein Panorama-Quellvideo angegeben.")
            return False

        self._stop_render.clear()

        cap_info = cv2.VideoCapture(src_path, cv2.CAP_FFMPEG)
        if not cap_info.isOpened():
            cap_info = cv2.VideoCapture(src_path)
        if not cap_info.isOpened():
            logger.error(f"Konnte Quellvideo nicht öffnen: {src_path}")
            return False

        fps = float(cap_info.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        pano_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        pano_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_info.release()

        start_frame = max(0, start_frame)
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames

        frames_to_process = max(0, end_frame - start_frame)
        if frames_to_process == 0:
            logger.warning(f"Keine Frames zu rendern: start_frame={start_frame}, end_frame={end_frame}")
            return False

        logger.info(f"=== 16:9 Taktik-Warp Konvertierung gestartet ===")
        logger.info(f"Quelle (Panorama):   {src_path} ({pano_w}x{pano_h} @ {fps:.2f} FPS)")
        logger.info(f"Ziel-Datei (16:9):   {output_filepath} ({out_width}x{out_height})")
        logger.info(f"Frame-Bereich:       {start_frame} bis {end_frame} (Gesamt: {frames_to_process} Frames)")
        logger.info(f"Encoder / Bitrate:   {codec} @ {bitrate_mbps} Mbps")

        # Generate 16:9 Tactical Remap LUTs once
        map_x, map_y = generate_tactical_16x9_luts(
            pano_w, pano_h,
            out_width, out_height,
            self.ai_broadcast.config.pitch_corners,
            margin_percent=getattr(self.ai_broadcast.config, 'tactical_margin', 0.0)
        )

        ffmpeg_bin = get_ffmpeg_path()
        codec_flags = get_encoder_flags(codec, bitrate_mbps)

        duration_sec = frames_to_process / max(fps, 1.0)
        t_start = start_frame / max(fps, 1.0)
        has_audio = has_audio_stream(src_path)

        if has_audio:
            audio_inputs = ["-ss", f"{t_start:.4f}", "-t", f"{duration_sec:.4f}", "-i", src_path]
            audio_mapping = ["-map", "0:v:0", "-map", "1:a:0"]
            audio_codec_flags = ["-c:a", "aac", "-b:a", "192k"]
        else:
            audio_inputs = []
            audio_mapping = ["-map", "0:v:0", "-an"]
            audio_codec_flags = []

        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{out_width}x{out_height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
        ]
        ffmpeg_cmd.extend(audio_inputs)
        ffmpeg_cmd.extend(audio_mapping)
        ffmpeg_cmd.extend(codec_flags)
        ffmpeg_cmd.extend(audio_codec_flags)
        ffmpeg_cmd.extend(["-pix_fmt", "yuv420p", "-shortest", output_filepath])

        logger.info(f"FFmpeg Befehl: {' '.join(ffmpeg_cmd)}")
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        ffmpeg_stderr_lines = []
        def monitor_ffmpeg():
            try:
                for raw_line in iter(proc.stderr.readline, b''):
                    decoded = raw_line.decode('utf-8', errors='replace').strip()
                    if decoded:
                        ffmpeg_stderr_lines.append(decoded)
                        if len(ffmpeg_stderr_lines) > 100:
                            ffmpeg_stderr_lines.pop(0)
                        if any(w in decoded.lower() for w in ["error", "fatal", "failed", "cannot"]):
                            logger.warning(f"[FFmpeg] {decoded}")
            except Exception:
                pass

        t_mon = threading.Thread(target=monitor_ffmpeg, daemon=True, name="FFmpegMonitorTactical")
        t_mon.start()

        cap = cv2.VideoCapture(src_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src_path)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        start_time = time.time()
        processed = 0
        render_success = False

        try:
            for f_idx in range(start_frame, end_frame):
                if self._stop_render.is_set():
                    break

                ret, pano_frame = cap.read()
                if not ret or pano_frame is None:
                    logger.warning(f"Kein Frame bei Index {f_idx} aus Panorama-Video.")
                    break

                final_frame = apply_tactical_16x9_warp(pano_frame, map_x, map_y)

                try:
                    proc.stdin.write(final_frame.tobytes())
                except (BrokenPipeError, OSError) as e:
                    logger.error(f"Schreibfehler auf FFmpeg-Pipe bei Frame {f_idx}: {e}")
                    break

                processed += 1

                if progress_callback and (processed % 15 == 0 or processed == frames_to_process):
                    elapsed = time.time() - start_time
                    current_fps = processed / max(elapsed, 0.001)
                    eta_sec = (frames_to_process - processed) / max(current_fps, 0.001)
                    progress_callback(processed, frames_to_process, current_fps, eta_sec, "16:9 Taktik-Warp...")

            if not self._stop_render.is_set() and processed == frames_to_process:
                render_success = True
                logger.info(f"🎉 16:9 Taktik-Warp erfolgreich exportiert! ({processed} Frames in {time.time()-start_time:.1f}s)")
        except Exception as e:
            logger.error(f"Fehler beim 16:9 Taktik-Warp: {e}", exc_info=True)
            render_success = False
        finally:
            self._stop_render.set()
            if proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            proc.wait()
            cap.release()

        return render_success

    def render_two_stage_broadcast(self,
                                   output_16x9_filepath: str,
                                   output_32x9_filepath: Optional[str] = None,
                                   out_16x9_width: int = 1920,
                                   out_16x9_height: int = 1080,
                                   out_32x9_width: int = 3840,
                                   out_32x9_height: int = 1080,
                                   codec: str = "hevc_nvenc",
                                   bitrate_mbps: int = 50,
                                   start_frame: int = 0,
                                   end_frame: Optional[int] = None,
                                   audio_source: str = "left",
                                   use_lookahead: bool = True,
                                   keep_32x9: bool = True,
                                   progress_callback: Optional[Callable[[int, int, float, float, str], None]] = None) -> bool:
        """
        Full 2-stage automated broadcast pipeline:
        Stage 1: Stitches 2 raw cameras into a master 32:9 panorama video.
        Stage 2: Generates the 16:9 broadcast follow-cam video from the 32:9 master with lookahead smoothing.
        """
        if not output_32x9_filepath:
            base, ext = os.path.splitext(output_16x9_filepath)
            output_32x9_filepath = f"{base}_master_32x9{ext}"

        logger.info("=== Starte Zweistufige Broadcast-Generierung ===")
        logger.info(f"Stufe 1 Ziel: {output_32x9_filepath}")
        logger.info(f"Stufe 2 Ziel: {output_16x9_filepath}")

        # Stage 1: Render 32:9 Panorama Master
        def stage1_cb(proc, total, fps, eta):
            if progress_callback:
                progress_callback(proc, total, fps, eta, "Stufe 1/2: 32:9 Panorama wird gestitcht...")

        ok1 = self.render_video_to_file(
            output_filepath=output_32x9_filepath,
            out_width=out_32x9_width,
            out_height=out_32x9_height,
            mode="32:9",
            codec=codec,
            bitrate_mbps=bitrate_mbps,
            start_frame=start_frame,
            end_frame=end_frame,
            audio_source=audio_source,
            progress_callback=stage1_cb
        )

        if not ok1 or self._stop_render.is_set():
            logger.error("Stufe 1 (32:9 Panorama Master) fehlgeschlagen oder abgebrochen.")
            return False

        self._stop_render.clear()

        # Stage 2: Render 16:9 Follow-Cam from the newly created 32:9 Master
        frames_in_master = max(1, (end_frame or self.get_max_duration_frames()) - start_frame)

        ok2 = self.render_broadcast_from_panorama(
            output_filepath=output_16x9_filepath,
            source_filepath=output_32x9_filepath,
            out_width=out_16x9_width,
            out_height=out_16x9_height,
            codec=codec,
            bitrate_mbps=bitrate_mbps,
            start_frame=0,
            end_frame=frames_in_master,
            use_lookahead=use_lookahead,
            progress_callback=progress_callback
        )

        if not keep_32x9 and os.path.exists(output_32x9_filepath):
            try:
                os.remove(output_32x9_filepath)
                logger.info("Temporäre 32:9 Datei nach erfolgreichem 16:9 Export bereinigt.")
            except Exception as e:
                logger.warning(f"Konnte temporäre 32:9 Datei nicht löschen: {e}")

        return ok2

    def cancel_render(self):
        """Cancels an ongoing render process."""
        logger.info("Abbruch des Render-Vorgangs angefordert.")
        self._stop_render.set()
