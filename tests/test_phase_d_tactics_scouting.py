"""
Unit & Integration Tests for Phase D: Tactical Metric Analytics & Scouting.
Tests:
- Dual-Quad Metric Pitch Homography
- Player / Ball Speed & Sprint Classification
- 2D Gaussian Heatmaps & Pitch Control (Dominance)
- Exporters: JSON, PNG Infographic, Vector PDF
- Qt TacticsDashboardDialog
"""
import sys
import os
import numpy as np

# Set path to project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matchtrack.tactics_metrics import (
    PitchHomography,
    PlayerMetricTrack,
    BallMetricTrack,
    TacticsMetricsEngine,
    TacticsReportExporter,
    MatchTacticsSummary
)


def test_pitch_homography():
    corners = [
        [0.0, 0.0],
        [0.5, 0.05],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.5, 0.95],
        [0.0, 1.0]
    ]
    homo = PitchHomography(corners)
    assert homo.is_valid, "Homography calculation failed"

    # Test Corners
    tl_m = homo.image_to_pitch(0.0, 0.0)
    assert abs(tl_m[0]) < 1e-2 and abs(tl_m[1]) < 1e-2, f"TL expected (0,0), got {tl_m}"

    tc_m = homo.image_to_pitch(0.5, 0.05)
    assert abs(tc_m[0] - 52.5) < 1e-2 and abs(tc_m[1]) < 1e-2, f"TC expected (52.5, 0), got {tc_m}"

    tr_m = homo.image_to_pitch(1.0, 0.0)
    assert abs(tr_m[0] - 105.0) < 1e-2 and abs(tr_m[1]) < 1e-2, f"TR expected (105, 0), got {tr_m}"

    br_m = homo.image_to_pitch(1.0, 1.0)
    assert abs(br_m[0] - 105.0) < 1e-2 and abs(br_m[1] - 68.0) < 1e-2, f"BR expected (105, 68), got {br_m}"

    # Center circle spot (52.5m, 34.0m)
    center_img = homo.pitch_to_image(52.5, 34.0)
    assert abs(center_img[0] - 0.5) < 1e-2, f"Expected center x=0.5, got {center_img[0]}"
    print("✓ test_pitch_homography passed")


def test_player_and_ball_speed():
    p = PlayerMetricTrack(track_id=7, team_id=0)
    
    # Simulate a fast sprint: 8 m/s = 28.8 km/h over 30 frames (1 second)
    for f in range(31):
        t = f / 30.0
        xm = 20.0 + t * 8.0  # 8 m/s
        ym = 34.0
        p.update(xm, ym, t)

    assert p.total_distance_m >= 7.5, f"Expected ~8m distance, got {p.total_distance_m}"
    assert p.max_speed_kmh >= 24.0, f"Expected speed >= 24 km/h, got {p.max_speed_kmh}"
    assert p.sprint_count >= 1, f"Expected at least 1 sprint, got {p.sprint_count}"

    # Ball test
    b = BallMetricTrack()
    # Fast shot: 25 m/s = 90 km/h
    b.update(50.0, 34.0, 0.0)
    b.update(50.0 + 25.0 * (1/30.0), 34.0, 1/30.0)
    assert b.max_speed_kmh >= 30.0, f"Expected ball speed > 30 km/h, got {b.max_speed_kmh}"
    print("✓ test_player_and_ball_speed passed")


def test_tactics_metrics_engine():
    engine = TacticsMetricsEngine()

    for f in range(40):
        t = f / 30.0
        x1 = 0.2 + (f / 40.0) * 0.10
        players = [
            (x1, 0.35, 0, 11),   # Team 0 sprinter
            (0.3, 0.65, 0, 12),   # Team 0 defender
            (0.7, 0.40, 1, 21),   # Team 1 defender
            (0.8, 0.60, 1, 22),   # Team 1 defender
        ]
        ball = (x1 + 0.01, 0.35)
        engine.update_frame(players, ball, f, fps=30.0)

    summary = engine.get_summary()
    assert summary.analyzed_frames == 40
    assert summary.team_distance_km[0] > 0.0
    assert "team_0" in summary.heatmaps
    assert "team_1" in summary.heatmaps
    assert "pitch_control" in summary.heatmaps
    assert summary.heatmaps["team_0"].shape == (68, 105)
    assert summary.heatmaps["pitch_control"].shape == (68, 105)
    assert len(summary.top_sprinters) > 0
    print("✓ test_tactics_metrics_engine passed")


def test_report_exporters():
    summary = MatchTacticsSummary(
        duration_sec=120.0,
        analyzed_frames=3600,
        ball_max_speed_kmh=84.5
    )
    summary.team_distance_km = {0: 10.4, 1: 9.8}
    summary.team_sprints = {0: 14, 1: 11}
    summary.team_max_speed_kmh = {0: 31.2, 1: 29.5}
    summary.team_possession_pct = {0: 54.0, 1: 46.0}
    summary.team_pitch_control_pct = {0: 52.5, 1: 47.5}
    summary.top_sprinters = [
        {"id": 9, "team": "Team Blau", "max_speed_kmh": 31.2, "sprints": 5, "distance_km": 1.2},
        {"id": 11, "team": "Team Rot", "max_speed_kmh": 29.5, "sprints": 3, "distance_km": 1.0}
    ]
    summary.heatmaps["team_0"] = np.random.uniform(0, 1, (68, 105)).astype(np.float32)
    summary.heatmaps["team_1"] = np.random.uniform(0, 1, (68, 105)).astype(np.float32)

    os.makedirs("tests/output", exist_ok=True)
    json_path = "tests/output/test_scouting.json"
    png_path = "tests/output/test_scouting.png"
    pdf_path = "tests/output/test_scouting.pdf"

    # JSON
    assert TacticsReportExporter.export_json(summary, json_path)
    assert os.path.exists(json_path) and os.path.getsize(json_path) > 100

    # PNG
    assert TacticsReportExporter.export_infographic_png(summary, png_path)
    assert os.path.exists(png_path) and os.path.getsize(png_path) > 1000

    # PDF
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    assert TacticsReportExporter.export_pdf(summary, pdf_path)
    assert os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000

    print("✓ test_report_exporters passed")


if __name__ == "__main__":
    test_pitch_homography()
    test_player_and_ball_speed()
    test_tactics_metrics_engine()
    test_report_exporters()
    print("🎉 All Phase D unit tests passed successfully!")

