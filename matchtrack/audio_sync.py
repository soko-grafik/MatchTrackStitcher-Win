"""
Automated Audio Cross-Correlation & Time Synchronization.
Extracts audio tracks from left & right DJI video files using FFmpeg, calculates FFT cross-correlation,
and computes frame-accurate synchronization offsets.
"""
import os
import subprocess
import tempfile
import numpy as np
from scipy import signal
from typing import Tuple, Optional, Dict, Any
from .paths import get_ffmpeg_path, get_ffprobe_path


def has_audio_stream(video_path: str) -> bool:
    """
    Checks whether a video file contains at least one readable audio stream using ffprobe.
    """
    if not video_path or not os.path.exists(video_path):
        return False
    ffprobe = get_ffprobe_path()
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        video_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return len(res.stdout.strip()) > 0
    except Exception:
        return False


def extract_audio_pcm(video_path: str, duration_sec: float = 120.0, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """
    Extracts PCM audio from a video file using FFmpeg.
    Returns 1D float32 numpy array.
    """
    if not os.path.exists(video_path):
        return None

    cmd = [
        get_ffmpeg_path(),
        "-y",

        "-t", str(duration_sec),
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-"
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_audio, _ = proc.communicate(timeout=30)
        if proc.returncode != 0 or len(raw_audio) == 0:
            return None
        
        audio = np.frombuffer(raw_audio, dtype=np.float32)
        return audio
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return None


def calculate_audio_sync_offset(video_left: str, 
                                 video_right: str, 
                                 fps: float = 60.0, 
                                 duration_sec: float = 120.0,
                                 sample_rate: int = 16000) -> Tuple[int, float, float]:
    """
    Finds the exact temporal offset between Left and Right video using audio cross-correlation.
    
    Returns:
        (frame_offset_right, time_offset_sec, confidence_score)
        A positive frame_offset_right means the RIGHT video started LATER than the LEFT video
        (i.e. to align them, drop 'frame_offset_right' frames from Left or start Right at 0).
    """
    audio_l = extract_audio_pcm(video_left, duration_sec=duration_sec, sample_rate=sample_rate)
    audio_r = extract_audio_pcm(video_right, duration_sec=duration_sec, sample_rate=sample_rate)

    if audio_l is None or audio_r is None or len(audio_l) < sample_rate or len(audio_r) < sample_rate:
        return 0, 0.0, 0.0

    # Normalize waveforms
    audio_l = audio_l - np.mean(audio_l)
    audio_r = audio_r - np.mean(audio_r)
    std_l = np.std(audio_l) + 1e-7
    std_r = np.std(audio_r) + 1e-7
    audio_l = audio_l / std_l
    audio_r = audio_r / std_r

    # Fast Fourier Transform based Cross-Correlation
    # Correlate signal: audio_l with audio_r
    corr = signal.correlate(audio_l, audio_r, mode='full', method='fft')
    lags = signal.correlation_lags(len(audio_l), len(audio_r), mode='full')

    # Peak detection
    peak_idx = np.argmax(np.abs(corr))
    peak_val = corr[peak_idx]
    best_lag = lags[peak_idx]

    # Calculate time delay and frame delay
    delay_sec = best_lag / float(sample_rate)
    frame_offset = int(np.round(delay_sec * fps))

    # Confidence calculation: Ratio between peak and median background noise
    abs_corr = np.abs(corr)
    median_val = np.median(abs_corr) + 1e-7
    confidence = float(np.clip((np.abs(peak_val) / median_val) / 20.0, 0.0, 1.0))

    return frame_offset, delay_sec, confidence
