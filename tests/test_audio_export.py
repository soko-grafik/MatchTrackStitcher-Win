import os
import sys
import subprocess
import numpy as np

# Ensure import paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from matchtrack.paths import get_ffmpeg_path, get_ffprobe_path
from matchtrack.audio_sync import has_audio_stream
from matchtrack.stitcher_engine import StitcherEngine
from matchtrack.rig_geometry import RigConfiguration


def test_audio_pipeline():
    ffmpeg = get_ffmpeg_path()
    ffprobe = get_ffprobe_path()
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    os.makedirs(out_dir, exist_ok=True)
    
    test_audio_l = os.path.join(out_dir, "test_audio_left.mp4")
    test_audio_r = os.path.join(out_dir, "test_audio_right.mp4")
    
    # 1. Create dummy left media (sine 440 Hz) and right media (sine 880 Hz) with audio + video
    cmd_l = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-f", "lavfi", "-i", "testsrc=duration=6:size=640x360:rate=30",
        "-c:v", "libx264", "-c:a", "aac", "-b:a", "128k",
        test_audio_l
    ]
    subprocess.run(cmd_l, check=True, capture_output=True)
    
    cmd_r = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=6",
        "-f", "lavfi", "-i", "testsrc=duration=6:size=640x360:rate=30",
        "-c:v", "libx264", "-c:a", "aac", "-b:a", "128k",
        test_audio_r
    ]
    subprocess.run(cmd_r, check=True, capture_output=True)
    print("Created test media files with audio tracks.")

    # 2. Test has_audio_stream
    assert has_audio_stream(test_audio_l) is True, "Left video should have audio"
    assert has_audio_stream(test_audio_r) is True, "Right video should have audio"
    print("[OK] has_audio_stream verified.")

    # 3. Test StitcherEngine export with audio_source='left'
    engine = StitcherEngine(RigConfiguration())
    engine.load_videos(test_audio_l, test_audio_r)
    engine.frame_offset_right = 15 # 0.5s offset

    out_left = os.path.join(out_dir, "engine_out_audio_left.mp4")
    success = engine.render_video_to_file(
        output_filepath=out_left,
        out_width=1280,
        out_height=360,
        codec="libx264",
        start_frame=30, # 1.0s
        end_frame=90,   # 3.0s -> 60 frames = 2.0s
        audio_source="left"
    )
    assert success is True, "Render with audio_source='left' failed"
    probe_res = subprocess.run([ffprobe, "-show_streams", out_left], capture_output=True, text=True)
    assert "codec_type=video" in probe_res.stdout and "codec_type=audio" in probe_res.stdout, "Left audio render missing audio stream"
    print("[OK] audio_source='left' exported successfully with audio.")

    # 4. Test StitcherEngine export with audio_source='right' (positive start time)
    out_right = os.path.join(out_dir, "engine_out_audio_right.mp4")
    success = engine.render_video_to_file(
        output_filepath=out_right,
        out_width=1280,
        out_height=360,
        codec="libx264",
        start_frame=30, # 1.0s -> t_r = (30-15)/30 = 0.5s
        end_frame=90,
        audio_source="right"
    )
    assert success is True, "Render with audio_source='right' failed"
    probe_res = subprocess.run([ffprobe, "-show_streams", out_right], capture_output=True, text=True)
    assert "codec_type=video" in probe_res.stdout and "codec_type=audio" in probe_res.stdout, "Right audio render missing audio stream"
    print("[OK] audio_source='right' exported successfully with audio.")

    # 5. Test StitcherEngine export with audio_source='right' (negative start time requiring adelay)
    engine.frame_offset_right = 45 # t_r = (15 - 45)/30 = -1.0s
    out_right_delay = os.path.join(out_dir, "engine_out_audio_right_delayed.mp4")
    success = engine.render_video_to_file(
        output_filepath=out_right_delay,
        out_width=1280,
        out_height=360,
        codec="libx264",
        start_frame=15, # 0.5s -> right started 1.5s after left -> delay 1.0s
        end_frame=75,
        audio_source="right"
    )
    assert success is True, "Render with audio_source='right' (delayed) failed"
    probe_res = subprocess.run([ffprobe, "-show_streams", out_right_delay], capture_output=True, text=True)
    assert "codec_type=video" in probe_res.stdout and "codec_type=audio" in probe_res.stdout, "Delayed right audio render missing audio stream"
    print("[OK] audio_source='right' with adelay exported successfully with audio.")

    # 6. Test StitcherEngine export with audio_source='mix'
    out_mix = os.path.join(out_dir, "engine_out_audio_mix.mp4")
    success = engine.render_video_to_file(
        output_filepath=out_mix,
        out_width=1280,
        out_height=360,
        codec="libx264",
        start_frame=15,
        end_frame=75,
        audio_source="mix"
    )
    assert success is True, "Render with audio_source='mix' failed"
    probe_res = subprocess.run([ffprobe, "-show_streams", out_mix], capture_output=True, text=True)
    assert "codec_type=video" in probe_res.stdout and "codec_type=audio" in probe_res.stdout, "Mix audio render missing audio stream"
    print("[OK] audio_source='mix' exported successfully with audio.")

    # 7. Test StitcherEngine export with audio_source='none' (muted)
    out_none = os.path.join(out_dir, "engine_out_audio_none.mp4")
    success = engine.render_video_to_file(
        output_filepath=out_none,
        out_width=1280,
        out_height=360,
        codec="libx264",
        start_frame=15,
        end_frame=75,
        audio_source="none"
    )
    assert success is True, "Render with audio_source='none' failed"
    probe_res = subprocess.run([ffprobe, "-show_streams", out_none], capture_output=True, text=True)
    assert "codec_type=video" in probe_res.stdout, "Muted render should have video"
    assert "codec_type=audio" not in probe_res.stdout, "Muted render should NOT have audio stream"
    print("[OK] audio_source='none' exported successfully without audio.")

    print("\n🎉 ALL AUDIO EXPORT PIPELINE TESTS PASSED!")


if __name__ == "__main__":
    test_audio_pipeline()

