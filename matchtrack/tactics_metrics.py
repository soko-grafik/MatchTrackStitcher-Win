"""
Tactical Metric Analytics & Scouting Engine (SkillCorner & TacticAI Principles).
Computes:
1. Metric Pitch Homography: Maps 6-point panorama polygon to FIFA 105m x 68m coordinates.
2. Speeds (km/h), Sprints (>25.2 km/h), High-Intensity Runs, Distances (km).
3. 2D Gaussian Kernel Heatmaps for Teams (Blau/Rot) and Ball.
4. Territorial Dominance & Pitch Control (Voronoi-based Spatial Influence).
5. Pass & Shot Speed Analysis.
6. Vector-crisp PDF, PNG Infographic, and JSON Scouting Reports.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import math
import json
import os
import numpy as np
import cv2

from .logger import get_logger

logger = get_logger("tactics")


class PitchHomography:
    """
    Transforms coordinates between image/panorama normalized space [0..1]
    and real FIFA pitch metric coordinates [0..105m, 0..68m] using dual-quad homography.
    """
    PITCH_LENGTH_M = 105.0
    PITCH_WIDTH_M = 68.0
    HALF_LENGTH_M = 52.5

    def __init__(self, pitch_corners: Optional[List[List[float]]] = None):
        self.H_left: Optional[np.ndarray] = None
        self.H_right: Optional[np.ndarray] = None
        self.H_left_inv: Optional[np.ndarray] = None
        self.H_right_inv: Optional[np.ndarray] = None
        self.is_valid = False
        self.update_corners(pitch_corners)

    def update_corners(self, pitch_corners: Optional[List[List[float]]]):
        """
        Updates homography matrices from 6 pitch polygon points [TL, TC, TR, BR, BC, BL].
        """
        if not pitch_corners:
            corners = [
                [0.0, 0.05], [0.5, 0.05], [1.0, 0.05],
                [1.0, 0.95], [0.5, 0.95], [0.0, 0.95]
            ]
        elif len(pitch_corners) == 4:
            tl, tr, br, bl = pitch_corners
            tc = [(tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5]
            bc = [(bl[0] + br[0]) * 0.5, (bl[1] + br[1]) * 0.5]
            corners = [tl, tc, tr, br, bc, bl]
        elif len(pitch_corners) == 6:
            corners = pitch_corners
        else:
            corners = [
                [0.0, 0.05], [0.5, 0.05], [1.0, 0.05],
                [1.0, 0.95], [0.5, 0.95], [0.0, 0.95]
            ]

        try:
            c = np.array(corners, dtype=np.float32)

            # Left quad: TL, TC, BC, BL -> [0,0], [52.5,0], [52.5,68], [0,68]
            src_l = c[[0, 1, 4, 5]]
            dst_l = np.array([
                [0.0, 0.0],
                [self.HALF_LENGTH_M, 0.0],
                [self.HALF_LENGTH_M, self.PITCH_WIDTH_M],
                [0.0, self.PITCH_WIDTH_M]
            ], dtype=np.float32)

            # Right quad: TC, TR, BR, BC -> [52.5,0], [105,0], [105,68], [52.5,68]
            src_r = c[[1, 2, 3, 4]]
            dst_r = np.array([
                [self.HALF_LENGTH_M, 0.0],
                [self.PITCH_LENGTH_M, 0.0],
                [self.PITCH_LENGTH_M, self.PITCH_WIDTH_M],
                [self.HALF_LENGTH_M, self.PITCH_WIDTH_M]
            ], dtype=np.float32)

            self.H_left = cv2.getPerspectiveTransform(src_l, dst_l)
            self.H_right = cv2.getPerspectiveTransform(src_r, dst_r)
            self.H_left_inv = np.linalg.inv(self.H_left)
            self.H_right_inv = np.linalg.inv(self.H_right)
            self.is_valid = True
        except Exception as e:
            logger.warning(f"Konnte Pitch-Homographie nicht berechnen: {e}")
            self.is_valid = False

    def image_to_pitch(self, norm_x: float, norm_y: float) -> Tuple[float, float]:
        """Maps normalized image coordinate (0..1, 0..1) to pitch meters (0..105, 0..68)."""
        if not self.is_valid:
            return (norm_x * self.PITCH_LENGTH_M, norm_y * self.PITCH_WIDTH_M)

        pt = np.array([[[norm_x, norm_y]]], dtype=np.float32)
        if norm_x <= 0.5:
            res = cv2.perspectiveTransform(pt, self.H_left)[0][0]
        else:
            res = cv2.perspectiveTransform(pt, self.H_right)[0][0]

        # Clamp safely to extended pitch boundaries (with 5m out-of-bounds safety margin)
        mx = float(np.clip(res[0], -5.0, self.PITCH_LENGTH_M + 5.0))
        my = float(np.clip(res[1], -5.0, self.PITCH_WIDTH_M + 5.0))
        return (mx, my)

    def pitch_to_image(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """Maps pitch meters to normalized image coordinates (0..1, 0..1)."""
        if not self.is_valid:
            return (x_m / self.PITCH_LENGTH_M, y_m / self.PITCH_WIDTH_M)

        pt = np.array([[[x_m, y_m]]], dtype=np.float32)
        if x_m <= self.HALF_LENGTH_M:
            res = cv2.perspectiveTransform(pt, self.H_left_inv)[0][0]
        else:
            res = cv2.perspectiveTransform(pt, self.H_right_inv)[0][0]

        ix = float(np.clip(res[0], 0.0, 1.0))
        iy = float(np.clip(res[1], 0.0, 1.0))
        return (ix, iy)


@dataclass
class PlayerMetricTrack:
    """Tracks physical and tactical performance metrics for a single player ID."""
    track_id: int
    team_id: int  # 0 = Blau, 1 = Rot, 2 = Schiri
    positions_m: List[Tuple[float, float, float]] = field(default_factory=list)  # (x_m, y_m, time_sec)
    current_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    total_distance_m: float = 0.0
    sprint_count: int = 0
    high_intensity_count: int = 0
    is_sprinting: bool = False
    zone_counts: Dict[str, int] = field(default_factory=lambda: {
        "defensive_third": 0,
        "middle_third": 0,
        "attacking_third": 0
    })

    def update(self, x_m: float, y_m: float, time_sec: float):
        """Updates position, distance and smoothed velocity in km/h."""
        if not self.positions_m:
            self.positions_m.append((x_m, y_m, time_sec))
            self._update_zone(x_m)
            return

        last_x, last_y, last_t = self.positions_m[-1]
        dt = max(1e-3, time_sec - last_t)
        
        # Calculate distance
        dx = x_m - last_x
        dy = y_m - last_y
        dist = math.hypot(dx, dy)

        # Discard physical impossibilities (teleportation > 40 km/h over sub-frame)
        raw_speed_kmh = (dist / dt) * 3.6
        if raw_speed_kmh > 42.0:
            # Outlier or track switch, ignore speed spike
            raw_speed_kmh = self.current_speed_kmh

        # Exponential moving average filter for speed (alpha = 0.35)
        self.current_speed_kmh = 0.35 * raw_speed_kmh + 0.65 * self.current_speed_kmh
        if self.current_speed_kmh > self.max_speed_kmh:
            self.max_speed_kmh = self.current_speed_kmh

        # Sprint classification (FIFA threshold: 25.2 km/h = 7.0 m/s)
        if self.current_speed_kmh >= 25.2:
            if not self.is_sprinting:
                self.sprint_count += 1
                self.is_sprinting = True
        elif self.current_speed_kmh < 22.0:
            self.is_sprinting = False

        if self.current_speed_kmh >= 19.8:
            self.high_intensity_count += 1

        self.total_distance_m += dist
        self.positions_m.append((x_m, y_m, time_sec))
        self._update_zone(x_m)

        # Keep buffer manageable (max 1800 points = 1 min history)
        if len(self.positions_m) > 1800:
            self.positions_m.pop(0)

    def _update_zone(self, x_m: float):
        if self.team_id == 0:  # Team Blau attacking towards right (x=105)
            if x_m < 35.0:
                self.zone_counts["defensive_third"] += 1
            elif x_m < 70.0:
                self.zone_counts["middle_third"] += 1
            else:
                self.zone_counts["attacking_third"] += 1
        else:  # Team Rot attacking towards left (x=0)
            if x_m > 70.0:
                self.zone_counts["defensive_third"] += 1
            elif x_m > 35.0:
                self.zone_counts["middle_third"] += 1
            else:
                self.zone_counts["attacking_third"] += 1


@dataclass
class BallMetricTrack:
    """Tracks soccer ball physical metrics (speed, trajectory, max shot velocity)."""
    positions_m: List[Tuple[float, float, float]] = field(default_factory=list)
    current_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0

    def update(self, x_m: float, y_m: float, time_sec: float):
        if not self.positions_m:
            self.positions_m.append((x_m, y_m, time_sec))
            return

        last_x, last_y, last_t = self.positions_m[-1]
        dt = max(1e-3, time_sec - last_t)
        dist = math.hypot(x_m - last_x, y_m - last_y)

        raw_speed_kmh = (dist / dt) * 3.6
        if raw_speed_kmh < 150.0:  # Maximum realistic shot speed: 150 km/h
            self.current_speed_kmh = 0.40 * raw_speed_kmh + 0.60 * self.current_speed_kmh
            if self.current_speed_kmh > self.max_speed_kmh:
                self.max_speed_kmh = self.current_speed_kmh

        self.positions_m.append((x_m, y_m, time_sec))
        if len(self.positions_m) > 600:
            self.positions_m.pop(0)


@dataclass
class MatchTacticsSummary:
    """Aggregated match statistics container."""
    duration_sec: float = 0.0
    analyzed_frames: int = 0
    team_distance_km: Dict[int, float] = field(default_factory=lambda: {0: 0.0, 1: 0.0})
    team_sprints: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    team_max_speed_kmh: Dict[int, float] = field(default_factory=lambda: {0: 0.0, 1: 0.0})
    team_possession_pct: Dict[int, float] = field(default_factory=lambda: {0: 50.0, 1: 50.0})
    team_pitch_control_pct: Dict[int, float] = field(default_factory=lambda: {0: 50.0, 1: 50.0})
    team_zone_distribution: Dict[int, Dict[str, float]] = field(default_factory=lambda: {
        0: {"defensive_third": 33.3, "middle_third": 33.4, "attacking_third": 33.3},
        1: {"defensive_third": 33.3, "middle_third": 33.4, "attacking_third": 33.3}
    })
    top_sprinters: List[Dict[str, Any]] = field(default_factory=list)
    ball_max_speed_kmh: float = 0.0
    heatmaps: Dict[str, np.ndarray] = field(default_factory=dict)  # "team_0", "team_1", "ball", "pitch_control"


class TacticsMetricsEngine:
    """
    Central real-time tactical analysis engine.
    Accumulates player and ball positions, builds 2D Gaussian heatmaps, and computes
    FIFA-grade performance and pitch dominance metrics.
    """
    GRID_W = 105
    GRID_H = 68

    def __init__(self, pitch_corners: Optional[List[List[float]]] = None):
        self.homography = PitchHomography(pitch_corners)
        self.player_tracks: Dict[int, PlayerMetricTrack] = {}
        self.ball_track = BallMetricTrack()
        
        # 2D Grid Accumulators (105m x 68m, 1m cells)
        self.heatmap_team_0 = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)
        self.heatmap_team_1 = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)
        self.heatmap_ball = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)
        self.pitch_control_map = np.full((self.GRID_H, self.GRID_W), 0.5, dtype=np.float32)

        self.frames_analyzed = 0
        self.total_duration_sec = 0.0

        # Precompute 2D Gaussian kernel (radius 4m, sigma = 2.0m)
        self.kernel_rad = 4
        k_size = self.kernel_rad * 2 + 1
        ax = np.arange(-self.kernel_rad, self.kernel_rad + 1)
        xx, yy = np.meshgrid(ax, ax)
        self.gaussian_kernel = np.exp(-(xx**2 + yy**2) / (2.0 * (2.0**2))).astype(np.float32)

    def set_pitch_corners(self, pitch_corners: List[List[float]]):
        """Updates homography when pitch calibration changes."""
        self.homography.update_corners(pitch_corners)

    def update_frame(
        self,
        players: List[Tuple[float, float, int, int]],  # (norm_x, norm_y, team_id, track_id)
        ball_norm: Optional[Tuple[float, float]],
        frame_idx: int,
        fps: float = 30.0
    ):
        """
        Ingests a video frame's detection data and updates spatial metrics.
        """
        time_sec = frame_idx / max(1.0, fps)
        self.total_duration_sec = max(self.total_duration_sec, time_sec)
        self.frames_analyzed += 1

        active_team_0_pts = []
        active_team_1_pts = []

        # 1. Update Players
        for (nx, ny, team_id, track_id) in players:
            if team_id not in (0, 1):  # Skip referees for team metrics
                continue

            mx, my = self.homography.image_to_pitch(nx, ny)

            if track_id not in self.player_tracks:
                self.player_tracks[track_id] = PlayerMetricTrack(track_id=track_id, team_id=team_id)

            track = self.player_tracks[track_id]
            track.update(mx, my, time_sec)

            # Accumulate on team heatmap
            gx = int(round(np.clip(mx, 0, self.GRID_W - 1)))
            gy = int(round(np.clip(my, 0, self.GRID_H - 1)))
            self._splat_gaussian(self.heatmap_team_0 if team_id == 0 else self.heatmap_team_1, gx, gy)

            if team_id == 0:
                active_team_0_pts.append((mx, my))
            else:
                active_team_1_pts.append((mx, my))

        # 2. Update Ball
        if ball_norm is not None:
            bx, by = self.homography.image_to_pitch(ball_norm[0], ball_norm[1])
            self.ball_track.update(bx, by, time_sec)
            bgx = int(round(np.clip(bx, 0, self.GRID_W - 1)))
            bgy = int(round(np.clip(by, 0, self.GRID_H - 1)))
            self._splat_gaussian(self.heatmap_ball, bgx, bgy)

        # 3. Update Pitch Control (Territorial Dominance via Spatial Influence)
        if active_team_0_pts or active_team_1_pts:
            self._compute_pitch_control(active_team_0_pts, active_team_1_pts)

    def _splat_gaussian(self, grid: np.ndarray, cx: int, cy: int):
        """Adds a 2D Gaussian splat centered at (cx, cy)."""
        r = self.kernel_rad
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(self.GRID_W, cx + r + 1)
        y2 = min(self.GRID_H, cy + r + 1)

        kx1 = x1 - (cx - r)
        ky1 = y1 - (cy - r)
        kx2 = kx1 + (x2 - x1)
        ky2 = ky1 + (y2 - y1)

        grid[y1:y2, x1:x2] += self.gaussian_kernel[ky1:ky2, kx1:kx2]

    def _compute_pitch_control(self, team_0: List[Tuple[float, float]], team_1: List[Tuple[float, float]]):
        """Computes continuous pitch control map [0.0 = full Team 0, 1.0 = full Team 1]."""
        if not team_0 and not team_1:
            return

        # Subsample grid (step 2) for ultra-fast calculation
        ys, xs = np.mgrid[0:self.GRID_H:2, 0:self.GRID_W:2]
        pts = np.vstack([xs.ravel(), ys.ravel()]).T

        # Calculate distances to nearest players of both teams
        if team_0:
            arr_0 = np.array(team_0, dtype=np.float32)
            dists_0 = np.min(np.linalg.norm(pts[:, None, :] - arr_0[None, :, :], axis=2), axis=1)
        else:
            dists_0 = np.full(len(pts), 999.0, dtype=np.float32)

        if team_1:
            arr_1 = np.array(team_1, dtype=np.float32)
            dists_1 = np.min(np.linalg.norm(pts[:, None, :] - arr_1[None, :, :], axis=2), axis=1)
        else:
            dists_1 = np.full(len(pts), 999.0, dtype=np.float32)

        # Logistic sigmoid probability of team 1 controlling the cell
        # diff > 0 -> Team 0 closer -> control near 0.0
        # diff < 0 -> Team 1 closer -> control near 1.0
        diff = dists_0 - dists_1
        control_sub = 1.0 / (1.0 + np.exp(-0.25 * diff))
        control_grid = control_sub.reshape(ys.shape[0], xs.shape[1])

        # Resize back to full 105x68 resolution
        full_control = cv2.resize(control_grid.astype(np.float32), (self.GRID_W, self.GRID_H), interpolation=cv2.INTER_LINEAR)
        # Running average with existing map
        self.pitch_control_map = 0.85 * self.pitch_control_map + 0.15 * full_control

    def get_player_speed(self, track_id: int) -> float:
        """Returns current speed in km/h for a given player ID."""
        track = self.player_tracks.get(track_id)
        return track.current_speed_kmh if track else 0.0

    def get_ball_speed(self) -> float:
        """Returns current ball speed in km/h."""
        return self.ball_track.current_speed_kmh

    def get_summary(self) -> MatchTacticsSummary:
        """Computes comprehensive tactical summary metrics."""
        summary = MatchTacticsSummary(
            duration_sec=self.total_duration_sec,
            analyzed_frames=self.frames_analyzed,
            ball_max_speed_kmh=self.ball_track.max_speed_kmh
        )

        dist_0 = sum(t.total_distance_m for t in self.player_tracks.values() if t.team_id == 0) / 1000.0
        dist_1 = sum(t.total_distance_m for t in self.player_tracks.values() if t.team_id == 1) / 1000.0
        summary.team_distance_km[0] = round(dist_0, 2)
        summary.team_distance_km[1] = round(dist_1, 2)

        sprints_0 = sum(t.sprint_count for t in self.player_tracks.values() if t.team_id == 0)
        sprints_1 = sum(t.sprint_count for t in self.player_tracks.values() if t.team_id == 1)
        summary.team_sprints[0] = sprints_0
        summary.team_sprints[1] = sprints_1

        max_spd_0 = max([t.max_speed_kmh for t in self.player_tracks.values() if t.team_id == 0] or [0.0])
        max_spd_1 = max([t.max_speed_kmh for t in self.player_tracks.values() if t.team_id == 1] or [0.0])
        summary.team_max_speed_kmh[0] = round(max_spd_0, 1)
        summary.team_max_speed_kmh[1] = round(max_spd_1, 1)

        # Pitch control territorial percentage
        mean_control_team_1 = float(np.mean(self.pitch_control_map)) * 100.0
        mean_control_team_0 = 100.0 - mean_control_team_1
        summary.team_pitch_control_pct[0] = round(mean_control_team_0, 1)
        summary.team_pitch_control_pct[1] = round(mean_control_team_1, 1)

        # Possession approximation (based on pitch territory and heatmap presence)
        tot_heat_0 = float(np.sum(self.heatmap_team_0))
        tot_heat_1 = float(np.sum(self.heatmap_team_1))
        tot_heat = max(1e-4, tot_heat_0 + tot_heat_1)
        poss_0 = (tot_heat_0 / tot_heat) * 100.0
        poss_1 = (tot_heat_1 / tot_heat) * 100.0
        summary.team_possession_pct[0] = round(poss_0, 1)
        summary.team_possession_pct[1] = round(poss_1, 1)

        # Zone Distribution
        for team_id in (0, 1):
            team_tracks = [t for t in self.player_tracks.values() if t.team_id == team_id]
            d_cnt = sum(t.zone_counts["defensive_third"] for t in team_tracks)
            m_cnt = sum(t.zone_counts["middle_third"] for t in team_tracks)
            a_cnt = sum(t.zone_counts["attacking_third"] for t in team_tracks)
            tot_z = max(1, d_cnt + m_cnt + a_cnt)
            summary.team_zone_distribution[team_id] = {
                "defensive_third": round((d_cnt / tot_z) * 100.0, 1),
                "middle_third": round((m_cnt / tot_z) * 100.0, 1),
                "attacking_third": round((a_cnt / tot_z) * 100.0, 1)
            }

        # Top 5 Sprinters
        sorted_players = sorted(self.player_tracks.values(), key=lambda t: t.max_speed_kmh, reverse=True)
        for p in sorted_players[:5]:
            team_str = "Team Blau" if p.team_id == 0 else "Team Rot"
            summary.top_sprinters.append({
                "id": p.track_id,
                "team": team_str,
                "max_speed_kmh": round(p.max_speed_kmh, 1),
                "sprints": p.sprint_count,
                "distance_km": round(p.total_distance_m / 1000.0, 2)
            })

        # Normalized heatmaps [0.0 ... 1.0]
        max_h0 = max(1e-4, float(np.max(self.heatmap_team_0)))
        max_h1 = max(1e-4, float(np.max(self.heatmap_team_1)))
        max_hb = max(1e-4, float(np.max(self.heatmap_ball)))
        summary.heatmaps["team_0"] = (self.heatmap_team_0 / max_h0).astype(np.float32)
        summary.heatmaps["team_1"] = (self.heatmap_team_1 / max_h1).astype(np.float32)
        summary.heatmaps["ball"] = (self.heatmap_ball / max_hb).astype(np.float32)
        summary.heatmaps["pitch_control"] = self.pitch_control_map.copy()

        return summary

    def reset(self):
        """Resets all metrics and accumulators."""
        self.player_tracks.clear()
        self.ball_track = BallMetricTrack()
        self.heatmap_team_0.fill(0.0)
        self.heatmap_team_1.fill(0.0)
        self.heatmap_ball.fill(0.0)
        self.pitch_control_map.fill(0.5)
        self.frames_analyzed = 0
        self.total_duration_sec = 0.0


class TacticsReportExporter:
    """
    Exports match analysis data into:
    1. Printable Vector PDF (A4 via PySide6 QPdfWriter)
    2. High-Resolution 1920x1080 Tactical Infographic PNG
    3. Structured JSON Scouting Data
    """
    @staticmethod
    def export_json(summary: MatchTacticsSummary, out_path: str) -> bool:
        """Exports metrics as JSON."""
        try:
            data = {
                "match_metrics": {
                    "duration_seconds": round(summary.duration_sec, 2),
                    "analyzed_frames": summary.analyzed_frames,
                    "ball_max_speed_kmh": summary.ball_max_speed_kmh,
                    "team_blue": {
                        "distance_km": summary.team_distance_km[0],
                        "sprints_total": summary.team_sprints[0],
                        "max_speed_kmh": summary.team_max_speed_kmh[0],
                        "possession_pct": summary.team_possession_pct[0],
                        "pitch_control_pct": summary.team_pitch_control_pct[0],
                        "zone_distribution": summary.team_zone_distribution[0]
                    },
                    "team_red": {
                        "distance_km": summary.team_distance_km[1],
                        "sprints_total": summary.team_sprints[1],
                        "max_speed_kmh": summary.team_max_speed_kmh[1],
                        "possession_pct": summary.team_possession_pct[1],
                        "pitch_control_pct": summary.team_pitch_control_pct[1],
                        "zone_distribution": summary.team_zone_distribution[1]
                    },
                    "top_sprinters": summary.top_sprinters
                }
            }
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"Taktik-JSON erfolgreich exportiert nach: {out_path}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Exportieren des Taktik-JSON: {e}")
            return False

    @staticmethod
    def render_tactical_infographic_bgr(summary: MatchTacticsSummary, w: int = 1920, h: int = 1080) -> np.ndarray:
        """
        Renders a stunning dark-theme Samsung One UI infographic dashboard (1920x1080 BGR).
        """
        canvas = np.full((h, w, 3), 18, dtype=np.uint8)  # Deep slate #121212

        # 1. Header Banner
        cv2.rectangle(canvas, (0, 0), (w, 80), (28, 28, 36), -1)
        cv2.putText(canvas, "MatchTrack AI - Taktik- & Scouting-Report", (40, 52), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        dur_min = int(summary.duration_sec // 60)
        dur_sec = int(summary.duration_sec % 60)
        sub_title = f"Analysierte Spielzeit: {dur_min:02d}:{dur_sec:02d} ({summary.analyzed_frames} Frames) | FIFA Spielfeld (105 x 68 m)"
        cv2.putText(canvas, sub_title, (40, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 150, 165), 1, cv2.LINE_AA)

        # 2. Key Metric Cards
        cards_y = 110
        card_w = (w - 100) // 4
        card_h = 100

        metrics_info = [
            ("Ballbesitz", f"{summary.team_possession_pct[0]}% : {summary.team_possession_pct[1]}%", "Team Blau vs. Team Rot"),
            ("Gesamtdistanz", f"{summary.team_distance_km[0]} km : {summary.team_distance_km[1]} km", "Gelaufene Distanz"),
            ("Sprints (>25 km/h)", f"{summary.team_sprints[0]} : {summary.team_sprints[1]}", "Sprint-Intensitaet"),
            ("Max. Schuss-Speed", f"{summary.ball_max_speed_kmh:.1f} km/h", "Gemessene Ball-Geschwindigkeit")
        ]

        for i, (title, val, subtitle) in enumerate(metrics_info):
            cx = 40 + i * (card_w + 7)
            cv2.rectangle(canvas, (cx, cards_y), (cx + card_w, cards_y + card_h), (32, 34, 46), -1)
            cv2.rectangle(canvas, (cx, cards_y), (cx + card_w, cards_y + card_h), (50, 55, 75), 1)
            cv2.putText(canvas, title, (cx + 16, cards_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 175, 195), 1, cv2.LINE_AA)
            cv2.putText(canvas, val, (cx + 16, cards_y + 68), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, subtitle, (cx + 16, cards_y + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 130, 145), 1, cv2.LINE_AA)

        # 3. Two Pitches: Left = Team Blau Heatmap, Right = Team Rot Heatmap
        pitch_y = 240
        pw = (w - 100) // 2
        ph = int(pw * (68.0 / 105.0))

        # Render Left Pitch (Team Blau)
        TacticsReportExporter._draw_pitch_with_heatmap(
            canvas, 40, pitch_y, pw, ph,
            summary.heatmaps.get("team_0"),
            team_color_bgr=(235, 130, 50),
            title="Team Blau - Raumabdeckung & Heatmap"
        )

        # Render Right Pitch (Team Rot)
        TacticsReportExporter._draw_pitch_with_heatmap(
            canvas, 60 + pw, pitch_y, pw, ph,
            summary.heatmaps.get("team_1"),
            team_color_bgr=(50, 70, 235),
            title="Team Rot - Raumabdeckung & Heatmap"
        )

        # 4. Bottom Table: Top Sprinters & Zone Distribution
        bot_y = pitch_y + ph + 30
        bot_h = h - bot_y - 30

        # Zone Distribution Box
        zw = (w - 100) // 2
        cv2.rectangle(canvas, (40, bot_y), (40 + zw, bot_y + bot_h), (32, 34, 46), -1)
        cv2.rectangle(canvas, (40, bot_y), (40 + zw, bot_y + bot_h), (50, 55, 75), 1)
        cv2.putText(canvas, "Zonen-Verteilung (Drittel)", (60, bot_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        z0 = summary.team_zone_distribution[0]
        z1 = summary.team_zone_distribution[1]
        line_b = f"Team Blau:  Abwehr {z0['defensive_third']}% | Mitte {z0['middle_third']}% | Angriff {z0['attacking_third']}%"
        line_r = f"Team Rot:   Abwehr {z1['defensive_third']}% | Mitte {z1['middle_third']}% | Angriff {z1['attacking_third']}%"
        cv2.putText(canvas, line_b, (60, bot_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 140, 60), 1, cv2.LINE_AA)
        cv2.putText(canvas, line_r, (60, bot_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 90, 240), 1, cv2.LINE_AA)

        # Top Sprinter Box
        cv2.rectangle(canvas, (60 + zw, bot_y), (60 + 2 * zw, bot_y + bot_h), (32, 34, 46), -1)
        cv2.rectangle(canvas, (60 + zw, bot_y), (60 + 2 * zw, bot_y + bot_h), (50, 55, 75), 1)
        cv2.putText(canvas, "Top-Sprinter & High-Speed Leaderboard", (80 + zw, bot_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        row_y = bot_y + 70
        for sp in summary.top_sprinters[:3]:
            txt = f"ID #{sp['id']} ({sp['team']}):  {sp['max_speed_kmh']} km/h  |  {sp['sprints']} Sprints  |  {sp['distance_km']} km"
            col = (235, 140, 60) if sp['team'] == "Team Blau" else (60, 90, 240)
            cv2.putText(canvas, txt, (80 + zw, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
            row_y += 30

        return canvas

    @staticmethod
    def _draw_pitch_with_heatmap(canvas: np.ndarray, rx: int, ry: int, pw: int, ph: int, heatmap: Optional[np.ndarray], team_color_bgr: Tuple[int, int, int], title: str):
        """Draws standard FIFA pitch outline and blends heatmap."""
        # Turf base
        cv2.rectangle(canvas, (rx, ry), (rx + pw, ry + ph), (18, 32, 22), -1)

        # Blend heatmap if available
        if heatmap is not None and heatmap.size > 0:
            heat_resized = cv2.resize(heatmap, (pw, ph), interpolation=cv2.INTER_LINEAR)
            # Create color mask
            heat_norm = np.clip(heat_resized * 255.0, 0, 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
            
            # Mask out background where heat is minimal
            mask = heat_norm > 15
            turf_roi = canvas[ry:ry+ph, rx:rx+pw]
            blended = cv2.addWeighted(heat_color, 0.65, turf_roi, 0.35, 0)
            turf_roi[mask] = blended[mask]

        # Pitch lines (Antialiased White)
        col_line = (200, 215, 205)
        cv2.rectangle(canvas, (rx, ry), (rx + pw, ry + ph), col_line, 1, cv2.LINE_AA)
        
        # Halfway line & center circle
        mid_x = rx + pw // 2
        mid_y = ry + ph // 2
        cv2.line(canvas, (mid_x, ry), (mid_x, ry + ph), col_line, 1, cv2.LINE_AA)
        radius = int(pw * (9.15 / 105.0))
        cv2.circle(canvas, (mid_x, mid_y), radius, col_line, 1, cv2.LINE_AA)
        cv2.circle(canvas, (mid_x, mid_y), 2, col_line, -1)

        # Penalty boxes
        pb_w = int(pw * (16.5 / 105.0))
        pb_h = int(ph * (40.32 / 68.0))
        pby = ry + (ph - pb_h) // 2
        cv2.rectangle(canvas, (rx, pby), (rx + pb_w, pby + pb_h), col_line, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (rx + pw - pb_w, pby), (rx + pw, pby + pb_h), col_line, 1, cv2.LINE_AA)

        # Pitch title badge
        cv2.rectangle(canvas, (rx + 10, ry + 10), (rx + 360, ry + 36), (0, 0, 0), -1)
        cv2.putText(canvas, title, (rx + 18, ry + 28), cv2.FONT_HERSHEY_DUPLEX, 0.55, team_color_bgr, 1, cv2.LINE_AA)

    @staticmethod
    def export_infographic_png(summary: MatchTacticsSummary, out_path: str) -> bool:
        """Saves tactical infographic dashboard to PNG."""
        try:
            img = TacticsReportExporter.render_tactical_infographic_bgr(summary, 1920, 1080)
            cv2.imwrite(out_path, img)
            logger.info(f"Taktik-Infografik erfolgreich exportiert nach: {out_path}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Exportieren der Taktik-Infografik: {e}")
            return False

    @staticmethod
    def export_pdf(summary: MatchTacticsSummary, out_path: str, match_title: str = "MatchTrack Spielanalyse") -> bool:
        """
        Generates a vector-sharp, professional A4 PDF coaching report using PySide6 QPdfWriter.
        """
        try:
            from PySide6.QtGui import QPdfWriter, QPainter, QPageSize, QPageLayout, QColor, QFont, QPen, QImage, QPainterPath
            from PySide6.QtCore import QRectF, QPointF, Qt

            writer = QPdfWriter(out_path)
            writer.setPageSize(QPageSize(QPageSize.A4))
            writer.setPageOrientation(QPageLayout.Portrait)
            writer.setResolution(300)

            painter = QPainter(writer)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

            # A4 page size in 300 DPI: approx. 2480 x 3508 pixels
            page_w = 2480
            page_h = 3508

            # Deep dark background
            painter.fillRect(0, 0, page_w, page_h, QColor("#0d1117"))

            # Header Banner
            painter.fillRect(0, 0, page_w, 240, QColor("#161b22"))
            painter.setPen(QColor("#38bdf8"))
            painter.setFont(QFont("Segoe UI", 24, QFont.Bold))
            painter.drawText(80, 110, "⚽ MatchTrack AI — Taktik- & Leistungsbericht")

            painter.setPen(QColor("#8b949e"))
            painter.setFont(QFont("Segoe UI", 13))
            dur_m = int(summary.duration_sec // 60)
            dur_s = int(summary.duration_sec % 60)
            painter.drawText(80, 175, f"{match_title} | Analysierte Spieldauer: {dur_m:02d}:{dur_s:02d} ({summary.analyzed_frames} Frames) | Spielfeld: 105 × 68 m")

            # 4 KPI Summary Cards
            card_y = 280
            card_w = (page_w - 240) // 4
            card_h = 240

            kpis = [
                ("Ballbesitz", f"{summary.team_possession_pct[0]}% : {summary.team_possession_pct[1]}%", "Team Blau vs. Team Rot"),
                ("Gesamtdistanz", f"{summary.team_distance_km[0]} : {summary.team_distance_km[1]} km", "Gelaufene Distanz"),
                ("Sprints (>25 km/h)", f"{summary.team_sprints[0]} : {summary.team_sprints[1]}", "Intensitaets-Laeufe"),
                ("Max. Ball-Speed", f"{summary.ball_max_speed_kmh:.1f} km/h", "Spitzengeschwindigkeit")
            ]

            for i, (ktitle, kval, ksub) in enumerate(kpis):
                cx = 80 + i * (card_w + 26)
                rect_c = QRectF(cx, card_y, card_w, card_h)
                painter.fillRect(rect_c, QColor("#21262d"))
                painter.setPen(QPen(QColor("#30363d"), 2))
                painter.drawRect(rect_c)

                painter.setPen(QColor("#8b949e"))
                painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
                painter.drawText(cx + 25, card_y + 60, ktitle)

                painter.setPen(QColor("#ffffff"))
                painter.setFont(QFont("Segoe UI", 18, QFont.Bold))
                painter.drawText(cx + 25, card_y + 140, kval)

                painter.setPen(QColor("#6e7681"))
                painter.setFont(QFont("Segoe UI", 9))
                painter.drawText(cx + 25, card_y + 195, ksub)

            # Two Pitches with Heatmaps (Team Blau and Team Rot)
            pw = page_w - 160
            ph = int(pw * 0.45 * (68.0 / 105.0))
            single_pw = (pw - 60) // 2

            # Section Title
            painter.setPen(QColor("#f0f6fc"))
            painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
            painter.drawText(80, 580, "🗺️ Positions-Heatmaps & Raumabdeckung")

            # Draw Team Blau Pitch
            TacticsReportExporter._draw_qt_pitch_heatmap(painter, 80, 620, single_pw, ph, summary.heatmaps.get("team_0"), QColor("#38bdf8"), "Team Blau (Position Heatmap)")
            # Draw Team Rot Pitch
            TacticsReportExporter._draw_qt_pitch_heatmap(painter, 140 + single_pw, 620, single_pw, ph, summary.heatmaps.get("team_1"), QColor("#ef4444"), "Team Rot (Position Heatmap)")

            # Zonen-Verteilung (Drittel) Section
            zy = 620 + ph + 80
            painter.setPen(QColor("#f0f6fc"))
            painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
            painter.drawText(80, zy, "📊 Taktische Zonenverteilung & Pitch Control")

            z_box_w = (page_w - 220) // 2
            z_box_h = 420

            # Team Blau Zone Box
            painter.fillRect(QRectF(80, zy + 30, z_box_w, z_box_h), QColor("#21262d"))
            painter.setPen(QPen(QColor("#30363d"), 2))
            painter.drawRect(QRectF(80, zy + 30, z_box_w, z_box_h))

            painter.setPen(QColor("#38bdf8"))
            painter.setFont(QFont("Segoe UI", 13, QFont.Bold))
            painter.drawText(110, zy + 85, "Team Blau — Zonenpräsenz")

            z0 = summary.team_zone_distribution[0]
            painter.setPen(QColor("#c9d1d9"))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(110, zy + 155, f"• Eigenes Drittel (Abwehr):    {z0['defensive_third']}%")
            painter.drawText(110, zy + 225, f"• Mittelfeld-Zone:             {z0['middle_third']}%")
            painter.drawText(110, zy + 295, f"• Angriffsdrittel:             {z0['attacking_third']}%")
            painter.drawText(110, zy + 365, f"• Territoriale Dominanz:       {summary.team_pitch_control_pct[0]}% Control")

            # Team Rot Zone Box
            painter.fillRect(QRectF(140 + z_box_w, zy + 30, z_box_w, z_box_h), QColor("#21262d"))
            painter.setPen(QPen(QColor("#30363d"), 2))
            painter.drawRect(QRectF(140 + z_box_w, zy + 30, z_box_w, z_box_h))

            painter.setPen(QColor("#ef4444"))
            painter.setFont(QFont("Segoe UI", 13, QFont.Bold))
            painter.drawText(170 + z_box_w, zy + 85, "Team Rot — Zonenpräsenz")

            z1 = summary.team_zone_distribution[1]
            painter.setPen(QColor("#c9d1d9"))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(170 + z_box_w, zy + 155, f"• Eigenes Drittel (Abwehr):    {z1['defensive_third']}%")
            painter.drawText(170 + z_box_w, zy + 225, f"• Mittelfeld-Zone:             {z1['middle_third']}%")
            painter.drawText(170 + z_box_w, zy + 295, f"• Angriffsdrittel:             {z1['attacking_third']}%")
            painter.drawText(170 + z_box_w, zy + 365, f"• Territoriale Dominanz:       {summary.team_pitch_control_pct[1]}% Control")

            # Top Sprinters Table Section
            ty = zy + z_box_h + 90
            painter.setPen(QColor("#f0f6fc"))
            painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
            painter.drawText(80, ty, "⚡ Physische Spitzenwerte (Sprint-Leaderboard)")

            tbl_rect = QRectF(80, ty + 30, page_w - 160, 520)
            painter.fillRect(tbl_rect, QColor("#21262d"))
            painter.setPen(QPen(QColor("#30363d"), 2))
            painter.drawRect(tbl_rect)

            # Table Header
            painter.setPen(QColor("#8b949e"))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(120, ty + 80, "SPIELER-ID")
            painter.drawText(480, ty + 80, "MANNSCHAFT")
            painter.drawText(950, ty + 80, "MAX SPEED")
            painter.drawText(1400, ty + 80, "SPRINTS (>25 KM/H)")
            painter.drawText(1950, ty + 80, "DISTANZ (KM)")

            row_y = ty + 145
            for idx, p in enumerate(summary.top_sprinters[:5]):
                painter.setPen(QColor("#f0f6fc"))
                painter.setFont(QFont("Segoe UI", 11, QFont.Bold if idx == 0 else QFont.Normal))
                painter.drawText(120, row_y, f"#{p['id']}")
                
                col_t = QColor("#38bdf8") if p['team'] == "Team Blau" else QColor("#ef4444")
                painter.setPen(col_t)
                painter.drawText(480, row_y, p['team'])

                painter.setPen(QColor("#f59e0b") if p['max_speed_kmh'] >= 28.0 else QColor("#f0f6fc"))
                painter.drawText(950, row_y, f"{p['max_speed_kmh']} km/h")
                painter.drawText(1400, row_y, str(p['sprints']))
                painter.drawText(1950, row_y, f"{p['distance_km']} km")
                
                # Subtle separator line
                painter.setPen(QPen(QColor("#30363d"), 1))
                painter.drawLine(100, row_y + 25, page_w - 100, row_y + 25)
                row_y += 75

            # Footer
            painter.setPen(QColor("#6e7681"))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(80, page_h - 60, "Erstellt von MatchTrack-Stitcher AI Sports Vision Engine • Gültig für Trainer, Scouts und Taktik-Analysten.")

            painter.end()
            logger.info(f"Taktik-PDF erfolgreich exportiert nach: {out_path}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Exportieren des Taktik-PDF: {e}", exc_info=True)
            return False

    @staticmethod
    def _draw_qt_pitch_heatmap(painter: Any, rx: int, ry: int, pw: int, ph: int, heatmap: Optional[np.ndarray], team_col: Any, title: str):
        """Helper to render vector pitch outline with heatmap into QPainter."""
        from PySide6.QtGui import QColor, QPen, QFont, QImage
        from PySide6.QtCore import QRectF, QPointF, Qt

        # Turf
        painter.fillRect(QRectF(rx, ry, pw, ph), QColor("#0d1b11"))

        # Blend heatmap image if available
        if heatmap is not None and heatmap.size > 0:
            heat_resized = cv2.resize(heatmap, (pw, ph), interpolation=cv2.INTER_LINEAR)
            heat_norm = np.clip(heat_resized * 255.0, 0, 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
            heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
            qimg = QImage(heat_rgb.data, pw, ph, heat_rgb.strides[0], QImage.Format_RGB888)
            painter.setOpacity(0.60)
            painter.drawImage(rx, ry, qimg)
            painter.setOpacity(1.0)

        # Lines
        pen_line = QPen(QColor(220, 235, 225, 200), 2)
        painter.setPen(pen_line)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(rx, ry, pw, ph))

        mid_x = rx + pw * 0.5
        mid_y = ry + ph * 0.5
        painter.drawLine(QPointF(mid_x, ry), QPointF(mid_x, ry + ph))
        
        radius = (9.15 / 105.0) * pw
        painter.drawEllipse(QPointF(mid_x, mid_y), radius, radius)

        # Penalty boxes
        pb_w = (16.5 / 105.0) * pw
        pb_h = (40.32 / 68.0) * ph
        pby = ry + (ph - pb_h) * 0.5
        painter.drawRect(QRectF(rx, pby, pb_w, pb_h))
        painter.drawRect(QRectF(rx + pw - pb_w, pby, pb_w, pb_h))

        # Title Tag
        tag_rect = QRectF(rx + 15, ry + 12, 380, 42)
        painter.fillRect(tag_rect, QColor(0, 0, 0, 210))
        painter.setPen(team_col)
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(tag_rect, Qt.AlignCenter, title)

