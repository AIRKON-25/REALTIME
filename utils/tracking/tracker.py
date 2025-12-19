import glob
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

COLOR_LABELS = ("red", "pink", "green", "white", "yellow", "purple", "black")
VALID_COLORS = {color: color for color in COLOR_LABELS}


def normalize_color_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    color = str(value).strip().lower()
    if not color or color == "none":
        return None
    return VALID_COLORS.get(color)

# 이럼 이제 맨 처음 정지되어있을때는 못잡을거임,, 랜덤임 음음음
# 최소 이동량(미터 단위 추정). 이보다 작으면 정지로 간주해 yaw 보정 생략
FORWARD_HEADING_MIN_DIST = 0.001
# SORT 보정: 감지 yaw가 추정 yaw과 너무 반대면 180° 뒤집어서 사용
YAW_SORT_CORRECTION_THRESHOLD = 150.0  # deg
# 검출 yaw를 180° 주기로만 신뢰(앞/뒤 동일)하고, 이동 방향으로 부호를 고정하기 위한 파라미터
HEADING_LOCK_ANGLE_THR = 45.0
HEADING_UNLOCK_ANGLE_THR = 140.0
HEADING_LOCK_FRAMES = 2
HEADING_ALIGN_MIN_DIST = 0.001

# Kalman noise/variance tuning for smoother yet responsive tracks
POS_INIT_COV_SCALE = 250.0
POS_PROCESS_NOISE_SCALE = 1.0
POS_MEAS_NOISE_SCALE = 2.0
YAW_PROCESS_NOISE_SCALE = 0.05
YAW_MEAS_NOISE_SCALE = 0.5
SIZE_PROCESS_NOISE_SCALE = 0.01
SIZE_MEAS_NOISE_SCALE = 0.5
# 측정 업데이트 시 프레임 간격이 길어질수록 관측값 영향 줄이는 계수
MEAS_GAP_SMOOTH_FACTOR = 0.15

# History-based smoothing parameters
STATE_HISTORY_SIZE = 8
DEFAULT_SMOOTH_WINDOW = 5 # 최근 이만큼의 프레임을 반영해서 부드럽게 ㅇㅇ

def wrap_deg(angle):
    """[-180, 180)로 정규화"""
    a = (angle + 180.0) % 360.0
    if a < 0:
        a += 360.0
    return a - 180.0


def nearest_equivalent_deg(meas: float, ref: float, period: float = 360.0) -> float:
    """
    Convert measurement into the equivalent angle closest to the reference.
    period=360 for general angle, 180 for fore/aft symmetric models.
    """
    d = meas - ref
    d = (d + period / 2.0) % period - period / 2.0
    return ref + d


def carla_to_aabb(detection: np.ndarray) -> np.ndarray:
    # detection: [class, x_c, y_c, l, w, yaw_deg]
    x_c, y_c, l, w, yaw_deg = detection[1:6]
    yaw = math.radians(yaw_deg)

    dx, dy = l / 2.0, w / 2.0
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    rotated_corners = corners @ R.T + np.array([x_c, y_c])

    x_min = np.min(rotated_corners[:, 0])
    x_max = np.max(rotated_corners[:, 0])
    y_min = np.min(rotated_corners[:, 1])
    y_max = np.max(rotated_corners[:, 1])

    aabb_width = x_max - x_min
    aabb_height = y_max - y_min

    return np.array([x_min, y_min, aabb_width, aabb_height])


def iou_bbox(boxA: np.ndarray, boxB: np.ndarray) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2]) * max(0.0, boxA[3])
    areaB = max(0.0, boxB[2]) * max(0.0, boxB[3])

    denom = areaA + areaB - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def iou_batch(detections_carla: np.ndarray, tracks: List["Track"]) -> np.ndarray:
    cost_matrix = np.zeros((len(detections_carla), len(tracks)), dtype=np.float32)
    for i, det_carla in enumerate(detections_carla):
        det_aabb = carla_to_aabb(det_carla)
        for j, track in enumerate(tracks):
            pred_xc, pred_yc = track.kf_pos.x[:2].flatten()
            temp_obb = np.array([0, pred_xc, pred_yc, track.car_length, track.car_width, track.car_yaw])
            pred_aabb = carla_to_aabb(temp_obb)
            cost_matrix[i, j] = 1.0 - iou_bbox(det_aabb, pred_aabb)
    return cost_matrix


class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3
    DELETED = 4


class Track:
    track_id_counter = 0

    def __init__(
        self,
        bbox_init: np.ndarray,
        confirm_hits: int = 3,
        color: Optional[str] = None,
        color_lock_streak: int = 5,
        pos_process_noise_scale: float = POS_PROCESS_NOISE_SCALE,
        pos_meas_noise_scale: float = POS_MEAS_NOISE_SCALE,
        yaw_process_noise_scale: float = YAW_PROCESS_NOISE_SCALE,
        yaw_meas_noise_scale: float = YAW_MEAS_NOISE_SCALE,
        size_process_noise_scale: float = SIZE_PROCESS_NOISE_SCALE,
        size_meas_noise_scale: float = SIZE_MEAS_NOISE_SCALE,
    ):
        # bbox_init: [class, x_c, y_c, l, w, yaw_deg]
        self.id = Track.track_id_counter
        Track.track_id_counter += 1

        self.cls = bbox_init[0]
        self.car_length = bbox_init[3]
        self.car_width = bbox_init[4]
        self.car_yaw = bbox_init[5]

        self.kf_pos = KalmanFilter(dim_x=4, dim_z=2)
        self.kf_pos.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf_pos.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf_pos.x[:2] = bbox_init[1:3].reshape((2, 1))
        self.last_pos = np.array(bbox_init[1:3], dtype=float)
        
        # 💡 파라미터 조정
        self.kf_pos.P *= POS_INIT_COV_SCALE
        self.kf_pos.Q *= pos_process_noise_scale
        self.kf_pos.R *= pos_meas_noise_scale
        self._pos_base_R = self.kf_pos.R.copy()
        self._gap_smooth_factor = MEAS_GAP_SMOOTH_FACTOR

        self.kf_yaw = self._init_2d_kf(
            initial_value=self.car_yaw,
            Q_scale=yaw_process_noise_scale,
            R_scale=yaw_meas_noise_scale,
        )
        self.kf_length = self._init_2d_kf(
            initial_value=self.car_length,
            Q_scale=size_process_noise_scale,
            R_scale=size_meas_noise_scale,
        )
        self.kf_width = self._init_2d_kf(
            initial_value=self.car_width,
            Q_scale=size_process_noise_scale,
            R_scale=size_meas_noise_scale,
        )

        self.time_since_update = 0
        self.hits = 1
        self.age = 1
        self.state = TrackState.TENTATIVE
        self.history: List[np.ndarray] = []
        self.confirm_hits = confirm_hits

        self.color_counts: Counter = Counter()
        self.current_color: Optional[str] = None
        self.total_color_votes = 0
        self.color_streak_color: Optional[str] = None
        self.color_streak: int = 0
        self.color_lock: Optional[str] = None
        self.color_lock_streak = max(1, int(color_lock_streak))
        self._update_color(color)

        self._append_history_entry()

        # 이동 방향 기반 yaw 부호 고정용 상태
        self.heading_locked: bool = False
        self.heading_lock_score: int = 0
        self.locked_heading: Optional[float] = None
    def _init_2d_kf(self, initial_value: float, Q_scale: float = 0.1, R_scale: float = 1.0) -> KalmanFilter:
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1, 1], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.x[0] = initial_value
        kf.P *= 10.0
        kf.Q *= Q_scale
        kf.R *= R_scale
        return kf

    def predict(self) -> None:
        self.kf_pos.predict()
        self.kf_yaw.predict()
        self.kf_length.predict()
        self.kf_width.predict()

        self.car_yaw = wrap_deg(self.kf_yaw.x[0, 0])
        self.kf_yaw.x[0, 0] = self.car_yaw
        self.car_length = max(0.0, self.kf_length.x[0, 0])
        self.car_width = max(0.0, self.kf_width.x[0, 0])

        self.age += 1
        if self.state != TrackState.DELETED:
            self.time_since_update += 1

    def update(self, bbox: np.ndarray, color: Optional[str] = None) -> None:
        measurement = np.asarray(bbox, dtype=float)
        self.cls = measurement[0]

        missed_frames = max(0, int(self.time_since_update) - 1)
        meas_scale = 1.0 + missed_frames * self._gap_smooth_factor
        if meas_scale != 1.0:
            self.kf_pos.R = self._pos_base_R * meas_scale
        else:
            self.kf_pos.R = self._pos_base_R
        self.kf_pos.update(measurement[1:3].reshape((2, 1)))
        self.kf_pos.R = self._pos_base_R

        yaw_det = float(measurement[5])
        yaw_det = self._align_measurement_yaw(yaw_det, measurement[1:3])
        yaw_meas = nearest_equivalent_deg(yaw_det, self.kf_yaw.x[0, 0], period=180.0)
        self.kf_yaw.update(np.array([[yaw_meas]]))
        self.car_yaw = wrap_deg(self.kf_yaw.x[0, 0])
        self.kf_yaw.x[0, 0] = self.car_yaw

        self.kf_length.update(np.array([[measurement[3]]]))
        self.kf_width.update(np.array([[measurement[4]]]))
        self.car_length = max(0.0, self.kf_length.x[0, 0])
        self.car_width = max(0.0, self.kf_width.x[0, 0])
        current_xy = self.kf_pos.x[:2].flatten()
        self._enforce_forward_heading(current_xy)

        self.time_since_update = 0
        self.hits += 1
        if self.state == TrackState.TENTATIVE and self.hits >= self.confirm_hits:
            self.state = TrackState.CONFIRMED
        elif self.state in (TrackState.CONFIRMED, TrackState.LOST):
            self.state = TrackState.CONFIRMED

        self._update_color(color)
        self._append_history_entry()

    def _update_color(self, color: Optional[str]) -> None:
        normalized = normalize_color_label(color)
        if not normalized:
            return
        # 잠금된 상태에서 다른 색이 들어오면 무시 (수동 해제/변경만 허용)
        if self.color_lock and normalized != self.color_lock:
            return

        if normalized == self.color_streak_color:
            self.color_streak += 1
        else:
            self.color_streak_color = normalized
            self.color_streak = 1

        self.color_counts[normalized] += 1
        self.total_color_votes += 1
        self.current_color = self.color_counts.most_common(1)[0][0]

        if not self.color_lock and self.color_streak >= self.color_lock_streak:
            self.color_lock = normalized

    def force_set_color(self, color: Optional[str]) -> None:
        """
        강제로 색상 라벨을 지정하거나 제거한다.
        """
        normalized = normalize_color_label(color)
        self.color_counts.clear()
        self.current_color = None
        self.total_color_votes = 0
        self.color_streak_color = None
        self.color_streak = 0
        self.color_lock = None
        if normalized:
            self.color_counts[normalized] = 1
            self.total_color_votes = 1
            self.current_color = normalized
            self.color_streak_color = normalized
            self.color_streak = self.color_lock_streak
            self.color_lock = normalized

    def _append_history_entry(self, state: Optional[np.ndarray] = None) -> None:
        if state is None:
            state = self._assemble_state()
        self.history.append(state)
        if len(self.history) > STATE_HISTORY_SIZE:
            self.history.pop(0)

    def _compute_heading_from_motion(self, meas_xy: np.ndarray) -> Optional[float]:
        """
        이동 벡터로부터 heading(rad) → deg 반환. 이동량이 충분치 않으면 None.
        """
        if self.last_pos is None:
            return None
        dx = float(meas_xy[0] - self.last_pos[0])
        dy = float(meas_xy[1] - self.last_pos[1])
        dist = math.hypot(dx, dy)
        if dist < HEADING_ALIGN_MIN_DIST:
            return None
        return wrap_deg(math.degrees(math.atan2(dy, dx)))

    def _align_measurement_yaw(self, yaw_det: float, meas_xy: np.ndarray) -> float:
        """
        검출 yaw는 앞/뒤가 뒤바뀌기 쉬우므로 180° 주기로만 신뢰하고,
        이동 방향(heading)과 가장 가까운 부호로 고정한다.
        """
        yaw_det = wrap_deg(yaw_det)
        # 1) 기존 상태와 180° 주기 기준으로 가깝게 정규화(앞/뒤 동일하게 취급)
        yaw_det = nearest_equivalent_deg(yaw_det, self.car_yaw, period=180.0)

        heading = self._compute_heading_from_motion(meas_xy)
        if heading is None:
            # 이동이 거의 없으면 잠정적으로 상태 근처로만 클램프
            if self.heading_locked and self.locked_heading is not None:
                return nearest_equivalent_deg(yaw_det, self.locked_heading, period=180.0)
            return yaw_det

        # 2) 이동 방향과 가장 가까운 부호 선택(180° 주기)
        yaw_heading = nearest_equivalent_deg(yaw_det, heading, period=180.0)
        diff = abs(wrap_deg(yaw_heading - heading))

        # 3) heading 일관성 점수로 잠금/해제 판단
        if diff <= HEADING_LOCK_ANGLE_THR:
            self.heading_lock_score = min(self.heading_lock_score + 1, HEADING_LOCK_FRAMES)
        else:
            self.heading_lock_score = max(self.heading_lock_score - 1, 0)

        if not self.heading_locked and self.heading_lock_score >= HEADING_LOCK_FRAMES:
            self.heading_locked = True
            self.locked_heading = heading
        elif self.heading_locked:
            if diff > HEADING_UNLOCK_ANGLE_THR:
                self.heading_locked = False
                self.heading_lock_score = 0
                self.locked_heading = None
            else:
                # 잠금 상태에서는 heading을 따라가되 180° 주기로만 조정
                self.locked_heading = heading

        if self.heading_locked and self.locked_heading is not None:
            return nearest_equivalent_deg(yaw_heading, self.locked_heading, period=180.0)
        return yaw_heading

    def get_color(self) -> Optional[str]:
        return self.color_lock if self.color_lock else self.current_color

    def get_color_confidence(self) -> float:
        color = self.get_color()
        if not color or self.total_color_votes == 0:
            return 0.0
        return self.color_counts.get(color, 0) / float(self.total_color_votes)

    def get_velocity(self) -> Tuple[float, float]:
        vx, vy = self.kf_pos.x[2:4].flatten()
        return float(vx), float(vy)

    def get_speed(self) -> float:
        vx, vy = self.get_velocity()
        return float(math.hypot(vx, vy))

    def _assemble_state(self) -> np.ndarray:
        vx, vy = self.get_velocity()
        return np.array([
            self.cls,
            self.kf_pos.x[0, 0],
            self.kf_pos.x[1, 0],
            self.car_length,
            self.car_width,
            self.car_yaw,
            vx,
            vy,
        ], dtype=float)

    def get_state(self, smooth_window: int = 1) -> np.ndarray:
        base_state = self._assemble_state()
        if smooth_window <= 1:
            return base_state

        samples: List[np.ndarray] = list(self.history[-smooth_window:])
        if self.time_since_update > 0 or not samples:
            samples.append(base_state)
        samples = samples[-smooth_window:]
        if not samples:
            return base_state

        stacked = np.vstack(samples)
        pos_size = np.mean(stacked[:, 1:5], axis=0)
        yaw_vals = stacked[:, 5]
        yaw_rad = np.deg2rad(yaw_vals)
        yaw_mean = wrap_deg(math.degrees(math.atan2(np.mean(np.sin(yaw_rad)), np.mean(np.cos(yaw_rad)))))
        vel_vals = np.mean(stacked[:, 6:8], axis=0) if stacked.shape[1] >= 8 else np.zeros(2, dtype=float)

        smoothed_state = np.array([
            self.cls,
            pos_size[0],
            pos_size[1],
            max(0.0, pos_size[2]),
            max(0.0, pos_size[3]),
            yaw_mean,
            vel_vals[0],
            vel_vals[1],
        ], dtype=float)
        return smoothed_state

    def _enforce_forward_heading(self, current_xy): # 이동방향과 yaw 맞추기
        if self.heading_locked:
            # 방향이 잠겨 있으면 좌표만 기록하고 별도의 뒤집기 생략
            self.last_pos = np.array(current_xy, dtype=float)
            return
        if self.last_pos is None:
            self.last_pos = np.array(current_xy, dtype=float)
            return
        dx = float(current_xy[0] - self.last_pos[0])
        dy = float(current_xy[1] - self.last_pos[1])
        dist = math.hypot(dx, dy)
        if dist >= FORWARD_HEADING_MIN_DIST:
            heading = wrap_deg(math.degrees(math.atan2(dy, dx)))
            diff = abs(wrap_deg(self.car_yaw - heading))
            if diff > 90.0:
                self.car_yaw = wrap_deg(self.car_yaw - 180.0)
                self.kf_yaw.x[0, 0] = self.car_yaw
        self.last_pos = np.array(current_xy, dtype=float)

    def force_flip_yaw(self, offset_deg: float = 180.0) -> None:
        """
        외부 명령으로 yaw를 강제 뒤집을 때 사용. heading 잠금은 해제한다.
        """
        self.car_yaw = wrap_deg(self.car_yaw + offset_deg)
        self.kf_yaw.x[0, 0] = self.car_yaw
        self.heading_locked = False
        self.heading_lock_score = 0
        self.locked_heading = None

    def force_set_yaw(self, yaw_deg: float) -> None:
        """
        명령으로 yaw 값을 직접 지정한다. heading 락은 해제한다.
        """
        self.car_yaw = wrap_deg(float(yaw_deg))
        self.kf_yaw.x[0, 0] = self.car_yaw
        self.heading_locked = False
        self.heading_lock_score = 0
        self.locked_heading = None


class SortTracker:
    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        color_penalty: float = 0.3,
        smooth_window: int = DEFAULT_SMOOTH_WINDOW,
        color_lock_streak: int = 5,
        pos_process_noise_scale: float = POS_PROCESS_NOISE_SCALE,
        pos_meas_noise_scale: float = POS_MEAS_NOISE_SCALE,
        yaw_process_noise_scale: float = YAW_PROCESS_NOISE_SCALE,
        yaw_meas_noise_scale: float = YAW_MEAS_NOISE_SCALE,
        size_process_noise_scale: float = SIZE_PROCESS_NOISE_SCALE,
        size_meas_noise_scale: float = SIZE_MEAS_NOISE_SCALE,
    ):
        self.tracks: List[Track] = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.color_penalty = color_penalty
        self.smooth_window = max(1, smooth_window)
        self.color_lock_streak = max(1, int(color_lock_streak))
        self.last_matches: List[Tuple[int, int]] = []
        self.pos_process_noise_scale = float(pos_process_noise_scale)
        self.pos_meas_noise_scale = float(pos_meas_noise_scale)
        self.yaw_process_noise_scale = float(yaw_process_noise_scale)
        self.yaw_meas_noise_scale = float(yaw_meas_noise_scale)
        self.size_process_noise_scale = float(size_process_noise_scale)
        self.size_meas_noise_scale = float(size_meas_noise_scale)

    def update(
        self,
        detections_carla: np.ndarray,
        detection_colors: Optional[List[Optional[str]]] = None,
    ) -> np.ndarray:
        """
        Update tracker with detections.
        detections_carla: Nx6 array [class, x_center, y_center, length, width, angle]
        detection_colors: optional list aligned with detections (None entries allowed)
        """
        if detections_carla is None:
            detections_carla = np.zeros((0, 6), dtype=float)
        detections_carla = np.asarray(detections_carla, dtype=float)
        self.last_matches = []

        for track in self.tracks:
            track.predict()

        active_tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

        matched_indices: List[Tuple[int, int]] = []
        unmatched_detections = list(range(len(detections_carla)))
        unmatched_tracks = list(range(len(active_tracks)))
        det_colors = self._prepare_detection_colors(detection_colors, len(detections_carla)) # 색상정규화, 길이맞추기

        if len(detections_carla) > 0 and len(active_tracks) > 0:
            cost_matrix = iou_batch(detections_carla, active_tracks)
            for i, det_color in enumerate(det_colors):
                if not det_color:
                    continue
                for j, track in enumerate(active_tracks):
                    cost_matrix[i, j] += self._color_cost(det_color, track.get_color())

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if 1.0 - cost_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c))
                    if r in unmatched_detections:
                        unmatched_detections.remove(r)
                    if c in unmatched_tracks:
                        unmatched_tracks.remove(c)

        for det_idx, track_idx in matched_indices:
            track = active_tracks[track_idx]
            color = det_colors[det_idx] if det_colors else None
            track.update(detections_carla[det_idx], color=color)
            self.last_matches.append((track.id, det_idx))

        for track_idx in unmatched_tracks:
            track = active_tracks[track_idx]
            if track.state == TrackState.CONFIRMED:
                track.state = TrackState.LOST
            elif track.state == TrackState.LOST:
                if track.time_since_update > self.max_age:
                    track.state = TrackState.DELETED
            elif track.state == TrackState.TENTATIVE:
                track.state = TrackState.DELETED

        for det_idx in unmatched_detections:
            color = det_colors[det_idx] if det_colors else None
            new_track = Track(
                detections_carla[det_idx],
                confirm_hits=self.min_hits,
                color=color,
                color_lock_streak=self.color_lock_streak,
                pos_process_noise_scale=self.pos_process_noise_scale,
                pos_meas_noise_scale=self.pos_meas_noise_scale,
                yaw_process_noise_scale=self.yaw_process_noise_scale,
                yaw_meas_noise_scale=self.yaw_meas_noise_scale,
                size_process_noise_scale=self.size_process_noise_scale,
                size_meas_noise_scale=self.size_meas_noise_scale,
            )
            self.tracks.append(new_track)

        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

        output_results = []
        for track in self.tracks:
            if track.state in (TrackState.CONFIRMED, TrackState.LOST):
                state = track.get_state(smooth_window=self.smooth_window)
                output_results.append(np.array([track.id, *state], dtype=float))

        return np.array(output_results) if output_results else np.array([])

    def _prepare_detection_colors(
        self,
        detection_colors: Optional[List[Optional[str]]],
        count: int,
    ) -> List[Optional[str]]:
        if not detection_colors:
            return [None] * count
        colors: List[Optional[str]] = []
        for idx in range(count):
            val = detection_colors[idx] if idx < len(detection_colors) else None
            colors.append(normalize_color_label(val))
        return colors

    def _color_cost(self, detection_color: Optional[str], track_color: Optional[str]) -> float:
        if not detection_color or not track_color:
            return 0.0
        if detection_color == track_color:
            return 0.0
        return self.color_penalty

    def get_latest_matches(self) -> List[Tuple[int, int]]:
        return list(self.last_matches)

    def get_track_attributes(self) -> Dict[int, dict]:
        attrs: Dict[int, dict] = {}
        for track in self.tracks:
            if track.state in (TrackState.CONFIRMED, TrackState.LOST):
                attrs[track.id] = {
                    "color": track.get_color(),
                    "color_confidence": track.get_color_confidence(),
                    "color_locked": bool(track.color_lock),
                    "velocity": track.get_velocity(),
                    "speed": track.get_speed(),
                }
        return attrs

    @staticmethod
    def _state_name(state_val: int) -> str:
        if state_val == TrackState.TENTATIVE:
            return "tentative"
        if state_val == TrackState.CONFIRMED:
            return "confirmed"
        if state_val == TrackState.LOST:
            return "lost"
        if state_val == TrackState.DELETED:
            return "deleted"
        return "unknown"

    def list_tracks(self) -> List[dict]:
        """
        현재 유지 중인 트랙의 기본 정보를 반환한다. (삭제된 트랙 제외)
        """
        items: List[dict] = []
        for track in self.tracks:
            if track.state == TrackState.DELETED:
                continue
            state_vec = track.get_state(smooth_window=self.smooth_window)
            items.append({
                "id": track.id,
                "state": self._state_name(track.state),
                "age": track.age,
                "time_since_update": track.time_since_update,
                "color": track.get_color(),
                "color_confidence": track.get_color_confidence(),
                "class": float(state_vec[0]),
                "cx": float(state_vec[1]),
                "cy": float(state_vec[2]),
                "length": float(state_vec[3]),
                "width": float(state_vec[4]),
                "yaw": float(state_vec[5]),
                "vx": float(state_vec[6]),
                "vy": float(state_vec[7]),
                "speed": float(math.hypot(state_vec[6], state_vec[7])),
            })
        return items

    def force_flip_yaw(self, track_id: int, offset_deg: float = 180.0) -> bool:
        """
        지정한 track id의 yaw를 강제로 뒤집는다.
        """
        for track in self.tracks:
            if track.id == track_id and track.state != TrackState.DELETED:
                track.force_flip_yaw(offset_deg)
                return True
        return False

    def force_set_yaw(self, track_id: int, yaw_deg: float) -> bool:
        """Force-set yaw (degrees) for a track id."""
        for track in self.tracks:
            if track.id == track_id and track.state != TrackState.DELETED:
                track.force_set_yaw(yaw_deg)
                return True
        return False

    def force_set_color(self, track_id: int, color: Optional[str]) -> bool:
        """
        지정한 track id의 색상 라벨을 강제로 설정하거나 제거한다.
        """
        for track in self.tracks:
            if track.id == track_id and track.state != TrackState.DELETED:
                track.force_set_color(color)
                return True
        return False


def load_detections_from_file(filepath: str) -> np.ndarray:
    try:
        if os.path.getsize(filepath) == 0:
            return np.array([])

        data = pd.read_csv(
            filepath,
            header=None,
            sep=r'[,\s]+',
            engine='python',
            dtype=float,
        ).values

        if data.shape[1] != 6:
            raise ValueError(f"파일 {filepath}의 열 개수가 예상 (6개)와 다릅니다: {data.shape[1]}")
        return data

    except Exception as e:
        print(f"파일 로드 오류: {filepath}. 오류: {e}")
        return np.array([])


def main_tracking():
    input_folder = "/merge_dist_wbf_drop"
    file_pattern = os.path.join(input_folder, "merged_frame_*.txt")
    frame_files = sorted(glob.glob(file_pattern))

    if not frame_files:
        print(f"오류: '{input_folder}' 폴더에서 파일을 찾을 수 없습니다. 경로와 파일명을 확인해주세요.")
        return

    tracker = SortTracker(max_age=10, min_hits=3, iou_threshold=0.15)
    all_tracking_results = []

    print(f"총 {len(frame_files)}개의 프레임 파일 로드됨. 추적 시작...")

    for frame_idx, filepath in enumerate(frame_files):
        detections = load_detections_from_file(filepath)
        tracked_objects = tracker.update(detections, None)

        if len(tracked_objects) > 0:
            frame_id_column = np.full((tracked_objects.shape[0], 1), frame_idx)
            frame_results = np.hstack((frame_id_column, tracked_objects))
            all_tracking_results.append(frame_results)

        if (frame_idx + 1) % 50 == 0 or frame_idx == len(frame_files) - 1:
            print(f"--- 프레임 {frame_idx + 1} / {len(frame_files)} 처리 완료. 현재 활성 트랙 수: {len(tracker.tracks)}")

    if all_tracking_results:
        try:
            final_results = np.vstack(all_tracking_results)

            print("\n✅ 추적 완료. 최종 결과를 'tracking_output.txt'에 저장합니다.")

            header = "frame_id, track_id, class, x_center, y_center, length, width, angle"
            np.savetxt(
                "tracking_output.txt",
                final_results,
                fmt=['%d', '%d', '%d', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f'],
                delimiter=',',
                header=header,
                comments='',
            )
        except Exception as e:
            print(f"\n⚠️ 최종 결과 통합/저장 중 오류 발생: {e}")
    else:
        print("\n⚠️ 추적된 객체가 없습니다. (모든 프레임에서 Confirmed/Lost 상태의 트랙이 없었음)")

# main_tracking() 
