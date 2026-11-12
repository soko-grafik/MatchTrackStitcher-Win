"""
Background QThread workers for MatchTrack-Stitcher video export and cloud upload.
"""

import os
import time
import threading
import cv2
import numpy as np
from typing import Optional, Any, Dict, Tuple
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from ..stitcher_engine import StitcherEngine
from ..cloud_uploader import MatchTrackOnlineClient
from ..logger import get_logger

logger = get_logger("gui.workers")


class RenderWorker(QThread):
    """Background worker for non-blocking video export (32:9 master, 16:9 follow-cam, or both)."""
    progress_signal = Signal(int, int, float, float, str)
    preview_signal = Signal(object)  # Emits QImage for live stepping follow-cam preview
    finished_signal = Signal(bool, str)

    def __init__(self, 
                 engine: StitcherEngine, 
                 out_path: str, 
                 width: int, 
                 height: int, 
                 mode: str, 
                 codec: str, 
                 bitrate: int, 
                 start_frame: int = 0, 
                 end_frame: Optional[int] = None, 
                 audio_source: str = "left",
                 use_lookahead: bool = True,
                 wide_angle_rectifier: Optional[Any] = None,
                 crop_rect: Optional[Tuple[float, float, float, float]] = None,
                 stretch_x: float = 1.0,
                 stretch_y: float = 1.0,
                 keep_32x9: bool = False,
                 upload_32x9_cloud: bool = False):
        super().__init__()
        self.engine = engine
        self.out_path = out_path
        self.width = width
        self.height = height
        self.mode = mode
        self.codec = codec
        self.bitrate = bitrate
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.audio_source = audio_source
        self.use_lookahead = use_lookahead
        self.wide_angle_rectifier = wide_angle_rectifier
        self.crop_rect = crop_rect
        self.stretch_x = stretch_x
        self.stretch_y = stretch_y
        self.keep_32x9 = keep_32x9
        self.upload_32x9_cloud = upload_32x9_cloud
        self.rendered_16x9_path: Optional[str] = None
        self.rendered_32x9_path: Optional[str] = None

    def run(self):
        try:
            def callback(processed, total, fps, eta, stage_text=""):
                self.progress_signal.emit(processed, total, fps, eta, stage_text)

            last_preview_time = 0.0
            def preview_callback(frame_bgr: np.ndarray):
                nonlocal last_preview_time
                now = time.time()
                # Update stepping preview ~4 times/sec (250ms interval) to ensure zero impact on NVENC speed
                if (now - last_preview_time) >= 0.25:
                    last_preview_time = now
                    try:
                        h, w = frame_bgr.shape[:2]
                        target_w = 480
                        target_h = int(target_w * (h / max(1, w)))
                        if target_h % 2 != 0:
                            target_h += 1
                        small = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                        qimg = QImage(rgb.data, target_w, target_h, target_w * 3, QImage.Format_RGB888).copy()
                        self.preview_signal.emit(qimg)
                    except Exception:
                        pass

            if self.mode == "wide_angle":
                self.rendered_16x9_path = self.out_path
                success = self.engine.render_wide_angle_video(
                    output_filepath=self.out_path,
                    out_width=self.width,
                    out_height=self.height,
                    codec=self.codec,
                    bitrate_mbps=self.bitrate,
                    start_frame=self.start_frame,
                    end_frame=self.end_frame,
                    wide_angle_rectifier=self.wide_angle_rectifier,
                    progress_callback=callback,
                    preview_callback=preview_callback
                )
            elif self.engine.is_panorama_mode():
                if self.mode == "custom_crop":
                    self.rendered_32x9_path = self.out_path
                    success = self.engine.convert_panorama_to_21x10(
                        output_filepath=self.out_path,
                        out_width=self.width,
                        out_height=self.height,
                        codec=self.codec,
                        bitrate_mbps=self.bitrate,
                        start_frame=self.start_frame,
                        end_frame=self.end_frame,
                        progress_callback=callback,
                        crop_rect=self.crop_rect,
                        stretch_x=self.stretch_x,
                        stretch_y=self.stretch_y,
                        preview_callback=preview_callback
                    )
                elif self.mode == "21:10":
                    self.rendered_32x9_path = self.out_path
                    success = self.engine.convert_panorama_to_21x10(
                        output_filepath=self.out_path,
                        out_width=self.width,
                        out_height=self.height,
                        codec=self.codec,
                        bitrate_mbps=self.bitrate,
                        start_frame=self.start_frame,
                        end_frame=self.end_frame,
                        progress_callback=callback,
                        preview_callback=preview_callback
                    )
                elif self.mode == "16:9_tactical":
                    self.rendered_16x9_path = self.out_path
                    success = self.engine.convert_panorama_to_16x9_tactical(
                        output_filepath=self.out_path,
                        out_width=self.width,
                        out_height=self.height,
                        codec=self.codec,
                        bitrate_mbps=self.bitrate,
                        start_frame=self.start_frame,
                        end_frame=self.end_frame,
                        progress_callback=callback,
                        preview_callback=preview_callback
                    )
                elif self.mode == "16:9_autocam":
                    self.rendered_16x9_path = self.out_path
                    success = self.engine.render_broadcast_from_panorama(
                        output_filepath=self.out_path,
                        out_width=self.width,
                        out_height=self.height,
                        codec=self.codec,
                        bitrate_mbps=self.bitrate,
                        start_frame=self.start_frame,
                        end_frame=self.end_frame,
                        use_lookahead=self.use_lookahead,
                        progress_callback=callback,
                        preview_callback=preview_callback
                    )
                else:
                    self.rendered_32x9_path = self.out_path
                    success = self.engine.convert_panorama_to_21x10(
                        output_filepath=self.out_path,
                        out_width=self.width,
                        out_height=self.height,
                        codec=self.codec,
                        bitrate_mbps=self.bitrate,
                        start_frame=self.start_frame,
                        end_frame=self.end_frame,
                        progress_callback=callback,
                        preview_callback=preview_callback
                    )
            elif self.mode == "both":
                base, ext = os.path.splitext(self.out_path)
                tag = "8x3" if (hasattr(self.engine, 'is_4_3_mode') and self.engine.is_4_3_mode()) else "32x9"
                pano_master_path = f"{base}_master_{tag}{ext}"
                self.rendered_16x9_path = self.out_path
                self.rendered_32x9_path = pano_master_path
                success = self.engine.render_two_stage_broadcast(
                    output_16x9_filepath=self.out_path,
                    output_32x9_filepath=pano_master_path,
                    out_16x9_width=self.width,
                    out_16x9_height=self.height,
                    codec=self.codec,
                    bitrate_mbps=self.bitrate,
                    start_frame=self.start_frame,
                    end_frame=self.end_frame,
                    audio_source=self.audio_source,
                    use_lookahead=self.use_lookahead,
                    keep_32x9=True,
                    progress_callback=callback,
                    preview_callback=preview_callback
                )
            elif self.mode == "16:9_autocam":
                base, ext = os.path.splitext(self.out_path)
                tag = "8x3" if (hasattr(self.engine, 'is_4_3_mode') and self.engine.is_4_3_mode()) else "32x9"
                pano_master_path = f"{base}_master_{tag}{ext}"
                self.rendered_16x9_path = self.out_path
                should_keep_32x9 = bool(self.keep_32x9 or self.upload_32x9_cloud)
                self.rendered_32x9_path = pano_master_path if should_keep_32x9 else None
                success = self.engine.render_two_stage_broadcast(
                    output_16x9_filepath=self.out_path,
                    output_32x9_filepath=pano_master_path,
                    out_16x9_width=self.width,
                    out_16x9_height=self.height,
                    codec=self.codec,
                    bitrate_mbps=self.bitrate,
                    start_frame=self.start_frame,
                    end_frame=self.end_frame,
                    audio_source=self.audio_source,
                    use_lookahead=self.use_lookahead,
                    keep_32x9=should_keep_32x9,
                    progress_callback=callback,
                    preview_callback=preview_callback
                )
            elif self.mode == "16:9_tactical":
                self.rendered_16x9_path = self.out_path
                def cb_tact(proc, total, fps, eta):
                    self.progress_signal.emit(proc, total, fps, eta, "16:9 Taktik-Warp Rendering...")

                success = self.engine.render_video_to_file(
                    output_filepath=self.out_path,
                    out_width=self.width,
                    out_height=self.height,
                    mode="16:9_tactical",
                    codec=self.codec,
                    bitrate_mbps=self.bitrate,
                    start_frame=self.start_frame,
                    end_frame=self.end_frame,
                    audio_source=self.audio_source,
                    progress_callback=cb_tact,
                    preview_callback=preview_callback
                )
            elif self.mode == "21:10":
                self.rendered_32x9_path = self.out_path
                def cb_2110(proc, total, fps, eta):
                    self.progress_signal.emit(proc, total, fps, eta, "21:10 Panorama Rendering...")

                success = self.engine.render_video_to_file(
                    output_filepath=self.out_path,
                    out_width=self.width,
                    out_height=self.height,
                    mode="21:10",
                    codec=self.codec,
                    bitrate_mbps=self.bitrate,
                    start_frame=self.start_frame,
                    end_frame=self.end_frame,
                    audio_source=self.audio_source,
                    progress_callback=cb_2110,
                    preview_callback=preview_callback
                )
            elif self.mode == "custom_crop":
                self.rendered_32x9_path = self.out_path
                def cb_crop(proc, total, fps, eta):
                    self.progress_signal.emit(proc, total, fps, eta, "Photoshop Zuschnitt & Transform Rendering...")

                success = self.engine.render_video_to_file(
                    output_filepath=self.out_path,
                    out_width=self.width,
                    out_height=self.height,
                    mode="custom_crop",
                    codec=self.codec,
                    bitrate_mbps=self.bitrate,
                    start_frame=self.start_frame,
                    end_frame=self.end_frame,
                    audio_source=self.audio_source,
                    progress_callback=cb_crop,
                    crop_rect=self.crop_rect,
                    stretch_x=self.stretch_x,
                    stretch_y=self.stretch_y,
                    preview_callback=preview_callback
                )
            else: # Panorama (8:3 or 32:9)
                self.rendered_32x9_path = self.out_path
                is_8x3 = hasattr(self.engine, 'is_4_3_mode') and self.engine.is_4_3_mode()
                pano_title = "8:3 Panorama" if is_8x3 else "32:9 Panorama"
                def cb_pano(proc, total, fps, eta):
                    self.progress_signal.emit(proc, total, fps, eta, f"{pano_title} Rendering...")

                success = self.engine.render_video_to_file(
                    output_filepath=self.out_path,
                    out_width=self.width,
                    out_height=self.height,
                    mode="8:3" if is_8x3 else "32:9",
                    codec=self.codec,
                    bitrate_mbps=self.bitrate,
                    start_frame=self.start_frame,
                    end_frame=self.end_frame,
                    audio_source=self.audio_source,
                    progress_callback=cb_pano
                )

            if success:
                self.finished_signal.emit(True, "Export erfolgreich abgeschlossen!")
            else:
                self.finished_signal.emit(False, "Export wurde abgebrochen oder es trat ein Fehler auf.")
        except Exception as e:
            logger.critical(f"Kritischer Ausnahmefehler im RenderWorker: {e}", exc_info=True)
            self.finished_signal.emit(False, f"Schwerwiegender interner Fehler: {e}")


class CloudUploadWorker(QThread):
    """Background worker for resumable chunked HTTPS upload to MatchTrack Online."""
    progress_signal = Signal(object, object, float, float, str)
    finished_signal = Signal(bool, str, dict)

    def __init__(self, client: MatchTrackOnlineClient, upload_kwargs: Dict[str, Any]):
        super().__init__()
        self.client = client
        self.upload_kwargs = upload_kwargs
        self.stop_event = threading.Event()

    def cancel(self):
        self.stop_event.set()

    def run(self):
        try:
            def cb(sent_b, total_b, speed_mb_s, eta_s, msg):
                self.progress_signal.emit(sent_b, total_b, speed_mb_s, eta_s, msg)

            res = self.client.upload_match(
                stop_event=self.stop_event,
                progress_callback=cb,
                **self.upload_kwargs
            )
            self.finished_signal.emit(True, res.get("message", "Upload erfolgreich abgeschlossen!"), res)
        except InterruptedError as ie:
            self.finished_signal.emit(False, str(ie), {})
        except Exception as e:
            logger.error(f"CloudUploadWorker Fehler: {e}", exc_info=True)
            self.finished_signal.emit(False, str(e), {})
