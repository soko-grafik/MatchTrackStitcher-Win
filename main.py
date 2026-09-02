"""
MatchTrack-Stitcher Main Entry Point.
Launches the GUI or CLI batch renderer.
"""
import sys
import argparse
from matchtrack.gui import launch_gui
from matchtrack.rig_geometry import RigConfiguration
from matchtrack.stitcher_engine import StitcherEngine
from matchtrack.logger import setup_logging, get_logger


def main():
    setup_logging()
    logger = get_logger("main")
    parser = argparse.ArgumentParser(description="MatchTrack-Stitcher: 32:9 Soccer Panorama Video Stitcher & 16:9 Broadcast Follow-Cam")
    parser.add_argument("--left", type=str, help="Path to left camera video file")
    parser.add_argument("--right", type=str, help="Path to right camera video file")
    parser.add_argument("--panorama", type=str, help="Path to existing 32:9 panorama video file (1-video mode)")
    parser.add_argument("--out", type=str, help="Path to output stitched video file")
    parser.add_argument("--mode", type=str, default="32:9", choices=["32:9", "21:10", "16:9", "both"], help="Export mode: '32:9' panorama, '21:10' squeezed panorama, '16:9' follow-cam broadcast, or 'both'")
    parser.add_argument("--config", type=str, help="Path to rig config json file")
    parser.add_argument("--resolution", type=str, default=None, help="Output resolution WxH (e.g. 1920x1080, 2520x1200, or 3840x1080)")
    parser.add_argument("--codec", type=str, default="hevc_nvenc", help="FFmpeg video codec (e.g. hevc_nvenc, h264_nvenc, libx264)")
    parser.add_argument("--bitrate", type=int, default=50, help="Bitrate in Mbps (default: 50)")
    parser.add_argument("--audio-source", type=str, default="left", choices=["left", "right", "mix", "none"], help="Audio source for exported video (left, right, mix, none)")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index for trimming")
    parser.add_argument("--end-frame", type=int, default=None, help="End frame index for trimming")
    parser.add_argument("--no-lookahead", action="store_true", help="Disable 2-pass lookahead trajectory smoothing")
    parser.add_argument("--cli", action="store_true", help="Run in CLI headless mode without GUI")
    
    args = parser.parse_args()

    is_cli = args.cli or (args.out and (args.panorama or (args.left and args.right)))

    if is_cli:
        if not args.out:
            print("Error: CLI mode requires --out argument.")
            sys.exit(1)

        print(f"=== MatchTrack-Stitcher CLI ===")
        use_lookahead = not args.no_lookahead

        def print_progress(processed, total, fps, eta, stage_text=""):
            pct = int((processed / max(total, 1)) * 100)
            m, s = divmod(int(eta), 60)
            prefix = f"[{stage_text}] " if stage_text else ""
            print(f"\r{prefix}Progress: {processed}/{total} ({pct}%) | {fps:.1f} FPS | ETA: {m:02d}:{s:02d}", end="", flush=True)

        if args.panorama:
            # Standalone Panorama Mode -> 16:9 Follow-Cam or 21:10 Squeezed Conversion
            print(f"Input Panorama: {args.panorama}")
            print(f"Output:         {args.out}")
            print(f"Mode:           {args.mode}")
            
            engine = StitcherEngine()
            engine.load_panorama_video(args.panorama)
            
            if args.mode == "21:10":
                res = args.resolution or "2520x1200"
                w, h = map(int, res.split("x"))
                print(f"Converting Panorama 32:9 to 21:10 ({w}x{h}) with codec {args.codec}...")
                success = engine.convert_panorama_to_21x10(
                    output_filepath=args.out,
                    out_width=w,
                    out_height=h,
                    codec=args.codec,
                    bitrate_mbps=args.bitrate,
                    start_frame=args.start_frame,
                    end_frame=args.end_frame,
                    progress_callback=print_progress
                )
            else:
                print(f"Lookahead:      {use_lookahead}")
                res = args.resolution or "1920x1080"
                w, h = map(int, res.split("x"))
                print(f"Rendering Broadcast 16:9 to {w}x{h} with codec {args.codec}...")
                success = engine.render_broadcast_from_panorama(
                    output_filepath=args.out,
                    out_width=w,
                    out_height=h,
                    codec=args.codec,
                    bitrate_mbps=args.bitrate,
                    start_frame=args.start_frame,
                    end_frame=args.end_frame,
                    use_lookahead=use_lookahead,
                    progress_callback=print_progress
                )
        else:
            # 2-Camera Mode (32:9 / 21:10 Rig)
            if not args.left or not args.right:
                print("Error: 2-camera mode requires both --left and --right video paths.")
                sys.exit(1)

            print(f"Left Video:   {args.left}")
            print(f"Right Video:  {args.right}")
            print(f"Audio Source: {args.audio_source}")
            print(f"Output:       {args.out}")
            print(f"Mode:         {args.mode}")

            rig = RigConfiguration.load_from_json(args.config) if args.config else RigConfiguration()
            engine = StitcherEngine(rig)
            engine.load_videos(args.left, args.right)

            if args.mode == "16:9":
                res = args.resolution or "1920x1080"
                w, h = map(int, res.split("x"))
                print(f"Rendering 2-Stage Broadcast 16:9 to {w}x{h} with codec {args.codec}...")
                success = engine.render_two_stage_broadcast(
                    output_16x9_filepath=args.out,
                    out_16x9_width=w,
                    out_16x9_height=h,
                    codec=args.codec,
                    bitrate_mbps=args.bitrate,
                    start_frame=args.start_frame,
                    end_frame=args.end_frame,
                    audio_source=args.audio_source,
                    use_lookahead=use_lookahead,
                    keep_32x9=False,
                    progress_callback=print_progress
                )
            elif args.mode == "both":
                res = args.resolution or "1920x1080"
                w, h = map(int, res.split("x"))
                print(f"Rendering 2-Stage Dual Export (32:9 Master + 16:9 Broadcast)...")
                success = engine.render_two_stage_broadcast(
                    output_16x9_filepath=args.out,
                    out_16x9_width=w,
                    out_16x9_height=h,
                    codec=args.codec,
                    bitrate_mbps=args.bitrate,
                    start_frame=args.start_frame,
                    end_frame=args.end_frame,
                    audio_source=args.audio_source,
                    use_lookahead=use_lookahead,
                    keep_32x9=True,
                    progress_callback=print_progress
                )
            elif args.mode == "21:10":
                res = args.resolution or "2520x1200"
                w, h = map(int, res.split("x"))
                print(f"Rendering 21:10 Panorama (Evenly Squeezed) to {w}x{h} with codec {args.codec}...")
                def cb_2110(proc, total, fps, eta):
                    print_progress(proc, total, fps, eta, "21:10 Panorama")

                success = engine.render_video_to_file(
                    output_filepath=args.out,
                    out_width=w,
                    out_height=h,
                    mode="21:10",
                    codec=args.codec,
                    bitrate_mbps=args.bitrate,
                    start_frame=args.start_frame,
                    end_frame=args.end_frame,
                    audio_source=args.audio_source,
                    progress_callback=cb_2110
                )
            else: # 32:9
                res = args.resolution or "3840x1080"
                w, h = map(int, res.split("x"))
                print(f"Rendering 32:9 Panorama to {w}x{h} with codec {args.codec}...")
                def cb_329(proc, total, fps, eta):
                    print_progress(proc, total, fps, eta, "32:9 Panorama")

                success = engine.render_video_to_file(
                    output_filepath=args.out,
                    out_width=w,
                    out_height=h,
                    mode="32:9",
                    codec=args.codec,
                    bitrate_mbps=args.bitrate,
                    start_frame=args.start_frame,
                    end_frame=args.end_frame,
                    audio_source=args.audio_source,
                    progress_callback=cb_329
                )

        print("\nRender Complete!" if success else "\nRender Failed.")
    else:
        # Launch modern PySide6 Qt GUI
        launch_gui()


if __name__ == "__main__":
    main()
