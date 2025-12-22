import math
import os
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from utils.tracking.geometry import (
    carla_to_aabb,
    iou_bbox,
    nearest_equivalent_deg,
    wrap_deg,
)
from utils.colors import normalize_color_label

from utils.tracking._constants import (
    ASSOC_CENTER_NORM,
    ASSOC_CENTER_WEIGHT,
    CAR_MAX_AGE,
    CAR_MIN_HITS,
    CAR_POS_MEAS_NOISE,
    CAR_POS_PROCESS_NOISE,
    CAR_SIZE_MEAS_NOISE,
    CAR_SIZE_PROCESS_NOISE,
    CAR_SMOOTH_WINDOW,
    CAR_OUTPUT_EMA_ALPHA,
    CAR_YAW_MEAS_NOISE,
    CAR_YAW_PROCESS_NOISE,
    CLS_CAR_TO_OBS_ASPECT_RATIO_MAX,
    CLS_CAR_TO_OBS_CENTER_DIST_MAX_M,
    CLS_CAR_TO_OBS_IOU_MIN,
    CLS_CAR_TO_OBS_REQUIRE_CENTER_DIST,
    CLS_CAR_TO_OBS_REQUIRE_IOU,
    CLS_CAR_TO_OBS_REQUIRE_LOW_SPEED,
    CLS_CAR_TO_OBS_REQUIRE_SIZE_MISMATCH,
    CLS_CAR_TO_OBS_SIZE_RATIO_MAX,
    CLS_CAR_TO_OBS_SIZE_RATIO_MIN,
    CLS_CAR_TO_OBS_SPEED_MAX,
    CLS_CAR_TO_OBS_STOP_FRAMES,
    CLS_HISTORY_LEN,
    CLS_VOTE_K_CAR_TO_OBS,
    CLS_VOTE_K_OBS_TO_CAR,
    DEFAULT_COLOR_LOCK_STREAK,
    DEFAULT_COLOR_PENALTY,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_AGE,
    DEFAULT_MIN_HITS,
    DEFAULT_SMOOTH_WINDOW,
    LAST_CHANCE_ASPECT_RATIO_MAX,
    LAST_CHANCE_CENTER_DIST_GATE_M,
    LAST_CHANCE_COLOR_GATE,
    LAST_CHANCE_DIR_DIFF_MAX_DEG,
    LAST_CHANCE_DIST_WEIGHT,
    LAST_CHANCE_IOU_WEIGHT,
    LAST_CHANCE_LOOKBACK,
    LAST_CHANCE_SIZE_RATIO_MAX,
    LAST_CHANCE_SIZE_RATIO_MIN,
    FORWARD_HEADING_MIN_DIST,
    HEADING_ALIGN_MIN_DIST,
    HEADING_FLIP_THRESHOLD,
    HEADING_LOCK_ANGLE_THR,
    HEADING_LOCK_FRAMES,
    HEADING_UNLOCK_ANGLE_THR,
    KF_STATE_INIT_COV_SCALE,
    MEAS_GAP_SMOOTH_FACTOR,
    OBSTACLE_JUMP_GATE_NORM_DIST,
    OBSTACLE_MAX_AGE,
    OBSTACLE_MIN_HITS,
    OBSTACLE_OUTPUT_EMA_ALPHA,
    OBSTACLE_POS_MEAS_NOISE,
    OBSTACLE_POS_PROCESS_NOISE,
    OBSTACLE_SIZE_MEAS_NOISE,
    OBSTACLE_SIZE_PROCESS_NOISE,
    OBSTACLE_VEL_DAMPING,
    OBSTACLE_YAW_MEAS_NOISE,
    OBSTACLE_YAW_PROCESS_NOISE,
    POS_INIT_COV_SCALE,
    POS_MEAS_NOISE_SCALE,
    POS_PROCESS_NOISE_SCALE,
    REACTIVATE_ASPECT_RATIO_MAX,
    REACTIVATE_CENTER_DIST_GATE_M,
    REACTIVATE_DIR_DIFF_MAX_DEG,
    REACTIVATE_GATE_COST_INF,
    REACTIVATE_IOU_THRESHOLD,
    REACTIVATE_SIZE_RATIO_MAX,
    REACTIVATE_SIZE_RATIO_MIN,
    REACTIVATE_SIZE_ROLLBACK_ENABLE,
    REACTIVATE_SIZE_ROLLBACK_FRAMES,
    REACTIVATE_SIZE_ROLLBACK_RATIO_MAX,
    REACTIVATE_SIZE_ROLLBACK_RATIO_MIN,
    REID_COLOR_LAMBDA,
    REID_COLOR_MISMATCH_PENALTY,
    REID_DIR_LAMBDA,
    REID_DIR_SPEED_MIN,
    REID_SIZE_EPS,
    REID_SIZE_LAMBDA,
    SIZE_MEAS_NOISE_SCALE,
    SIZE_PROCESS_NOISE_SCALE,
    SPLIT_GUARD_CLOSE_DIST_M,
    SPLIT_GUARD_DIR_DIFF_MAX_DEG,
    SPLIT_GUARD_ENABLE,
    SPLIT_GUARD_FRAMES,
    STATE_HISTORY_SIZE,
    YAW_MEAS_NOISE_SCALE,
    YAW_PERIOD_DEG,
    YAW_PROCESS_NOISE_SCALE,
    OBSTACLE_OUTPUT_SNAP_EPS,
    OBSTACLE_STOP_ALPHA_SCALE,
    OBSTACLE_STOP_FRAMES,
    OBSTACLE_STOP_Q_SCALE,
    OBSTACLE_STOP_SPEED_THR,
    CAR_OUTPUT_SNAP_EPS,
    CAR_STOP_SPEED_THR,
    CAR_STOP_ALPHA_SCALE,
    CAR_STOP_FRAMES,
    CAR_STOP_Q_SCALE,
)


@dataclass
class TrackerConfigBase:
    max_age: int
    min_hits: int
    pos_process_noise_scale: float
    pos_meas_noise_scale: float
    yaw_process_noise_scale: float
    yaw_meas_noise_scale: float
    size_process_noise_scale: float
    size_meas_noise_scale: float
    smooth_window: int
    vel_damping: float = 0.0
    output_ema_alpha: Optional[float] = None
    jump_distance_gate: Optional[float] = None
    output_snap_eps: float = 0.0
    stop_speed_thresh: float = 0.0
    stop_frames: int = 0
    stop_alpha_scale: float = 1.0
    stop_q_scale: float = 1.0


class TrackerConfigCar(TrackerConfigBase):
    def __init__(
        self,
        max_age: int = CAR_MAX_AGE,
        min_hits: int = CAR_MIN_HITS,
        pos_process_noise_scale: float = CAR_POS_PROCESS_NOISE,
        pos_meas_noise_scale: float = CAR_POS_MEAS_NOISE,
        yaw_process_noise_scale: float = CAR_YAW_PROCESS_NOISE,
        yaw_meas_noise_scale: float = CAR_YAW_MEAS_NOISE,
        size_process_noise_scale: float = CAR_SIZE_PROCESS_NOISE,
        size_meas_noise_scale: float = CAR_SIZE_MEAS_NOISE,
        smooth_window: int = CAR_SMOOTH_WINDOW,
        output_ema_alpha: Optional[float] = CAR_OUTPUT_EMA_ALPHA,
    ):
        super().__init__(
            max_age=max_age,
            min_hits=min_hits,
            pos_process_noise_scale=pos_process_noise_scale,
            pos_meas_noise_scale=pos_meas_noise_scale,
            yaw_process_noise_scale=yaw_process_noise_scale,
            yaw_meas_noise_scale=yaw_meas_noise_scale,
            size_process_noise_scale=size_process_noise_scale,
            size_meas_noise_scale=size_meas_noise_scale,
            smooth_window=smooth_window,
            vel_damping=0.0,
            output_ema_alpha=output_ema_alpha,
            jump_distance_gate=None,
            output_snap_eps=CAR_OUTPUT_SNAP_EPS,
            stop_speed_thresh=CAR_STOP_SPEED_THR,
            stop_frames=CAR_STOP_FRAMES,
            stop_alpha_scale=CAR_STOP_ALPHA_SCALE,
            stop_q_scale=CAR_STOP_Q_SCALE,
        )


class TrackerConfigObstacle(TrackerConfigBase):
    def __init__(
        self,
        max_age: int = OBSTACLE_MAX_AGE,
        min_hits: int = OBSTACLE_MIN_HITS,
        pos_process_noise_scale: float = OBSTACLE_POS_PROCESS_NOISE,
        pos_meas_noise_scale: float = OBSTACLE_POS_MEAS_NOISE,
        yaw_process_noise_scale: float = OBSTACLE_YAW_PROCESS_NOISE,
        yaw_meas_noise_scale: float = OBSTACLE_YAW_MEAS_NOISE,
        size_process_noise_scale: float = OBSTACLE_SIZE_PROCESS_NOISE,
        size_meas_noise_scale: float = OBSTACLE_SIZE_MEAS_NOISE,
        smooth_window: int = DEFAULT_SMOOTH_WINDOW,
        vel_damping: float = OBSTACLE_VEL_DAMPING,
        output_ema_alpha: Optional[float] = OBSTACLE_OUTPUT_EMA_ALPHA,
        jump_distance_gate: Optional[float] = OBSTACLE_JUMP_GATE_NORM_DIST,
        output_snap_eps: float = OBSTACLE_OUTPUT_SNAP_EPS,
        stop_speed_thresh: float = OBSTACLE_STOP_SPEED_THR,
        stop_frames: int = OBSTACLE_STOP_FRAMES,
        stop_alpha_scale: float = OBSTACLE_STOP_ALPHA_SCALE,
        stop_q_scale: float = OBSTACLE_STOP_Q_SCALE,
    ):
        super().__init__(
            max_age=max_age,
            min_hits=min_hits,
            pos_process_noise_scale=pos_process_noise_scale,
            pos_meas_noise_scale=pos_meas_noise_scale,
            yaw_process_noise_scale=yaw_process_noise_scale,
            yaw_meas_noise_scale=yaw_meas_noise_scale,
            size_process_noise_scale=size_process_noise_scale,
            size_meas_noise_scale=size_meas_noise_scale,
            smooth_window=smooth_window,
            vel_damping=vel_damping,
            output_ema_alpha=output_ema_alpha,
            jump_distance_gate=jump_distance_gate,
            output_snap_eps=output_snap_eps,
            stop_speed_thresh=stop_speed_thresh,
            stop_frames=stop_frames,
            stop_alpha_scale=stop_alpha_scale,
            stop_q_scale=stop_q_scale,
        )


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
        min_hit: int = DEFAULT_MIN_HITS,
        color: Optional[str] = None,
        color_lock_streak: int = DEFAULT_COLOR_LOCK_STREAK,
        pos_process_noise_scale: float = POS_PROCESS_NOISE_SCALE,
        pos_meas_noise_scale: float = POS_MEAS_NOISE_SCALE,
        yaw_process_noise_scale: float = YAW_PROCESS_NOISE_SCALE,
        yaw_meas_noise_scale: float = YAW_MEAS_NOISE_SCALE,
        size_process_noise_scale: float = SIZE_PROCESS_NOISE_SCALE,
        size_meas_noise_scale: float = SIZE_MEAS_NOISE_SCALE,
        class_config: Optional[TrackerConfigBase] = None,
    ):
        # bbox_init: [class, x_c, y_c, l, w, yaw_deg]
        self.id = Track.track_id_counter
        Track.track_id_counter += 1

        self.cls = bbox_init[0]
        try:
            cls_init = int(bbox_init[0])
        except (TypeError, ValueError):
            cls_init = 0
        self.cls_state = cls_init
        self.cls_hist = deque([cls_init], maxlen=max(1, CLS_HISTORY_LEN))
        self.last_cls_change_frame = 0
        self.low_speed_streak = 0
        self.car_length = bbox_init[3]
        self.car_width = bbox_init[4]
        self.car_yaw = bbox_init[5]

        self.kf_pos = KalmanFilter(dim_x=4, dim_z=2)
        self.kf_pos.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf_pos.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf_pos.x[:2] = bbox_init[1:3].reshape((2, 1))
        self.last_pos = np.array(bbox_init[1:3], dtype=float)
        
        self.kf_pos.P = np.eye(4) * POS_INIT_COV_SCALE
        self.kf_pos.Q = np.eye(4) * POS_PROCESS_NOISE_SCALE
        self.kf_pos.R = np.eye(2) * POS_MEAS_NOISE_SCALE
        self._gap_smooth_factor = MEAS_GAP_SMOOTH_FACTOR

        self.kf_yaw = self._init_2d_kf(
            initial_value=self.car_yaw,
            Q_scale=YAW_PROCESS_NOISE_SCALE,
            R_scale=YAW_MEAS_NOISE_SCALE,
        )
        self.kf_length = self._init_2d_kf(
            initial_value=self.car_length,
            Q_scale=SIZE_PROCESS_NOISE_SCALE,
            R_scale=SIZE_MEAS_NOISE_SCALE,
        )
        self.kf_width = self._init_2d_kf(
            initial_value=self.car_width,
            Q_scale=SIZE_PROCESS_NOISE_SCALE,
            R_scale=SIZE_MEAS_NOISE_SCALE,
        )
        self.current_config: Optional[TrackerConfigBase] = None
        self._pos_base_R = self.kf_pos.R.copy()
        self._base_Q_pos = self.kf_pos.Q.copy()
        self._base_Q_yaw = self.kf_yaw.Q.copy()
        self._base_Q_size = self.kf_length.Q.copy()
        self._ema_state: Optional[np.ndarray] = None
        if class_config:
            self.apply_config(class_config)

        self.time_since_update = 0
        self.hits = 1
        self.age = 1
        self.state = TrackState.TENTATIVE
        self.history: List[np.ndarray] = []
        self.min_hit = min_hit

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
        self.reactivation_guard_frames: int = 0
        self.reactivation_size_ref: Optional[Tuple[float, float]] = None
        self.split_guard_frames: int = 0
    def _init_2d_kf(self, initial_value: float, Q_scale: float = 0.1, R_scale: float = 1.0) -> KalmanFilter:
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1, 1], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.x[0] = initial_value
        kf.P *= KF_STATE_INIT_COV_SCALE
        kf.Q *= Q_scale
        kf.R *= R_scale
        return kf

    def apply_config(self, cfg: TrackerConfigBase) -> None:
        self.current_config = cfg
        self.min_hit = cfg.min_hits
        self.kf_pos.Q = np.eye(4) * POS_PROCESS_NOISE_SCALE * cfg.pos_process_noise_scale
        self.kf_pos.R = np.eye(2) * POS_MEAS_NOISE_SCALE * cfg.pos_meas_noise_scale
        self._pos_base_R = self.kf_pos.R.copy()
        self._base_Q_pos = self.kf_pos.Q.copy()
        self.kf_yaw.Q = np.eye(2) * cfg.yaw_process_noise_scale
        self.kf_yaw.R = np.eye(1) * cfg.yaw_meas_noise_scale
        self._base_Q_yaw = self.kf_yaw.Q.copy()
        self.kf_length.Q = np.eye(2) * cfg.size_process_noise_scale
        self.kf_length.R = np.eye(1) * cfg.size_meas_noise_scale
        self.kf_width.Q = np.eye(2) * cfg.size_process_noise_scale
        self.kf_width.R = np.eye(1) * cfg.size_meas_noise_scale
        self._base_Q_size = self.kf_length.Q.copy()

    def predict(self, cfg: Optional[TrackerConfigBase] = None) -> None:
        if cfg:
            self.apply_config(cfg)
        self.kf_pos.predict()
        self.kf_yaw.predict()
        self.kf_length.predict()
        self.kf_width.predict()

        self.car_yaw = wrap_deg(self.kf_yaw.x[0, 0])
        self.kf_yaw.x[0, 0] = self.car_yaw
        self.car_length = max(0.0, self.kf_length.x[0, 0])
        self.car_width = max(0.0, self.kf_width.x[0, 0])

        if cfg and cfg.vel_damping > 0.0:
            damp = max(0.0, min(1.0, cfg.vel_damping))
            self.kf_pos.x[2, 0] *= (1.0 - damp)
            self.kf_pos.x[3, 0] *= (1.0 - damp)
        if cfg and cfg.stop_q_scale < 1.0 and self._is_stopped(cfg):
            scale = max(cfg.stop_q_scale, 0.0)
            self.kf_pos.Q = self._base_Q_pos * scale
            self.kf_yaw.Q = self._base_Q_yaw * scale
            self.kf_length.Q = self._base_Q_size * scale
            self.kf_width.Q = self._base_Q_size * scale
        else:
            self.kf_pos.Q = self._base_Q_pos
            self.kf_yaw.Q = self._base_Q_yaw
            self.kf_length.Q = self._base_Q_size
            self.kf_width.Q = self._base_Q_size

        self.age += 1
        if self.state != TrackState.DELETED:
            self.time_since_update += 1

    def update(self, bbox: np.ndarray, color: Optional[str] = None, cfg: Optional[TrackerConfigBase] = None) -> None:
        measurement = np.asarray(bbox, dtype=float)
        self.cls = measurement[0]
        if cfg:
            self.apply_config(cfg)

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
        yaw_meas = nearest_equivalent_deg(yaw_det, self.kf_yaw.x[0, 0], period=YAW_PERIOD_DEG)
        self.kf_yaw.update(np.array([[yaw_meas]]))
        # self.kf_yaw.update(np.array([[yaw_det]]))
        self.car_yaw = wrap_deg(self.kf_yaw.x[0, 0])
        self.kf_yaw.x[0, 0] = self.car_yaw

        self.kf_length.update(np.array([[measurement[3]]]))
        self.kf_width.update(np.array([[measurement[4]]]))
        self.car_length = max(0.0, self.kf_length.x[0, 0])
        self.car_width = max(0.0, self.kf_width.x[0, 0])
        current_xy = self.kf_pos.x[:2].flatten()
        # self._enforce_forward_heading(current_xy)

        self.time_since_update = 0
        self.hits += 1
        if self.state == TrackState.TENTATIVE and self.hits >= self.min_hit:
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
        yaw_det = nearest_equivalent_deg(yaw_det, self.car_yaw, period=YAW_PERIOD_DEG)

        heading = self._compute_heading_from_motion(meas_xy)
        if heading is None:
            # 이동이 거의 없으면 잠정적으로 상태 근처로만 클램프
            if self.heading_locked and self.locked_heading is not None:
                return nearest_equivalent_deg(yaw_det, self.locked_heading, period=YAW_PERIOD_DEG)
            return yaw_det

        # 2) 이동 방향과 가장 가까운 부호 선택(180° 주기)
        yaw_heading = nearest_equivalent_deg(yaw_det, heading, period=YAW_PERIOD_DEG)
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
            return nearest_equivalent_deg(yaw_heading, self.locked_heading, period=YAW_PERIOD_DEG)
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
        cls_out = self.cls_state if self.cls_state is not None else self.cls
        return np.array([
            cls_out,
            self.kf_pos.x[0, 0],
            self.kf_pos.x[1, 0],
            self.car_length,
            self.car_width,
            self.car_yaw,
            vx,
            vy,
        ], dtype=float)

    def get_state(self, smooth_window: int = 1, ema_alpha: Optional[float] = None, snap_eps: float = 0.0, stop_alpha_scale: float = 1.0, is_stopped: bool = False) -> np.ndarray:
        base_state = self._assemble_state()
        if smooth_window <= 1:
            out = base_state
        else:
            samples: List[np.ndarray] = list(self.history[-smooth_window:])
            if self.time_since_update > 0 or not samples:
                samples.append(base_state)
            samples = samples[-smooth_window:]
            if not samples:
                out = base_state
            else:
                stacked = np.vstack(samples)
                pos_size = np.mean(stacked[:, 1:5], axis=0)
                yaw_vals = stacked[:, 5]
                yaw_rad = np.deg2rad(yaw_vals)
                yaw_mean = wrap_deg(math.degrees(math.atan2(np.mean(np.sin(yaw_rad)), np.mean(np.cos(yaw_rad)))))
                vel_vals = np.mean(stacked[:, 6:8], axis=0) if stacked.shape[1] >= 8 else np.zeros(2, dtype=float)

                cls_out = self.cls_state if self.cls_state is not None else self.cls
                out = np.array([
                    cls_out,
                    pos_size[0],
                    pos_size[1],
                    max(0.0, pos_size[2]),
                    max(0.0, pos_size[3]),
                    yaw_mean,
                    vel_vals[0],
                    vel_vals[1],
                ], dtype=float)

        last_output = self._ema_state
        if snap_eps > 0.0 and last_output is not None:
            disp = math.hypot(out[1] - last_output[1], out[2] - last_output[2])
            if disp <= snap_eps:
                out = last_output

        if ema_alpha is None or ema_alpha <= 0.0 or ema_alpha >= 1.0:
            return out
        alpha = ema_alpha
        if is_stopped and stop_alpha_scale > 0.0:
            alpha = max(min(alpha * stop_alpha_scale, 1.0), 0.0)
        if self._ema_state is None:
            self._ema_state = out.copy()
        else:
            beta = 1.0 - alpha
            yaw_prev = self._ema_state[5]
            yaw_new = out[5]
            yaw_diff = wrap_deg(yaw_new - yaw_prev)
            blended_yaw = wrap_deg(yaw_prev + alpha * yaw_diff)
            blended = self._ema_state * beta + out * alpha
            blended[5] = blended_yaw
            self._ema_state = blended
        return self._ema_state

    def _is_stopped(self, cfg: TrackerConfigBase) -> bool:
        if cfg.stop_frames <= 0 or cfg.stop_speed_thresh <= 0.0:
            return False
        recent = list(self.history[-cfg.stop_frames:]) if self.history else []
        if len(recent) < cfg.stop_frames:
            return False
        speeds = []
        for st in recent:
            if len(st) >= 8:
                vx, vy = st[6], st[7]
            else:
                vx, vy = 0.0, 0.0
            speeds.append(math.hypot(vx, vy))
        if not speeds:
            return False
        return float(np.mean(speeds)) <= cfg.stop_speed_thresh

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
            if diff > HEADING_FLIP_THRESHOLD:
                self.car_yaw = wrap_deg(self.car_yaw - YAW_PERIOD_DEG)
                self.kf_yaw.x[0, 0] = self.car_yaw
        self.last_pos = np.array(current_xy, dtype=float)

    def force_flip_yaw(self, offset_deg: float = YAW_PERIOD_DEG) -> None:
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
        max_age: int = DEFAULT_MAX_AGE,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        color_penalty: float = DEFAULT_COLOR_PENALTY,
        smooth_window: int = DEFAULT_SMOOTH_WINDOW,
        pos_process_noise_scale: float = POS_PROCESS_NOISE_SCALE,
        pos_meas_noise_scale: float = POS_MEAS_NOISE_SCALE,
        yaw_process_noise_scale: float = YAW_PROCESS_NOISE_SCALE,
        yaw_meas_noise_scale: float = YAW_MEAS_NOISE_SCALE,
        size_process_noise_scale: float = SIZE_PROCESS_NOISE_SCALE,
        size_meas_noise_scale: float = SIZE_MEAS_NOISE_SCALE,
        car_config: Optional[TrackerConfigBase] = None,
        obstacle_config: Optional[TrackerConfigBase] = None,
        assoc_center_weight: float = ASSOC_CENTER_WEIGHT,
        assoc_center_norm: float = ASSOC_CENTER_NORM,
        debug_logging: bool = False,
    ):
        self.tracks: List[Track] = []
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.color_penalty = color_penalty
        self.smooth_window = max(1, smooth_window)
        self.last_matches: List[Tuple[int, int]] = []
        self.pos_process_noise_scale = float(pos_process_noise_scale)
        self.pos_meas_noise_scale = float(pos_meas_noise_scale)
        self.yaw_process_noise_scale = float(yaw_process_noise_scale)
        self.yaw_meas_noise_scale = float(yaw_meas_noise_scale)
        self.size_process_noise_scale = float(size_process_noise_scale)
        self.size_meas_noise_scale = float(size_meas_noise_scale)
        self.config_car = car_config or TrackerConfigCar()
        self.config_obstacle = obstacle_config or TrackerConfigObstacle()
        self.default_config = TrackerConfigCar(
            max_age=max_age,
            min_hits=DEFAULT_MIN_HITS,
            pos_process_noise_scale=pos_process_noise_scale,
            pos_meas_noise_scale=pos_meas_noise_scale,
            yaw_process_noise_scale=yaw_process_noise_scale,
            yaw_meas_noise_scale=yaw_meas_noise_scale,
            size_process_noise_scale=size_process_noise_scale,
            size_meas_noise_scale=size_meas_noise_scale,
            smooth_window=smooth_window,
        )
        self.assoc_center_weight = max(0.0, assoc_center_weight)
        self.assoc_center_norm = max(assoc_center_norm, 1e-3)
        self.reactivate_iou_threshold = REACTIVATE_IOU_THRESHOLD
        self.reactivate_center_dist_gate_m = REACTIVATE_CENTER_DIST_GATE_M
        self.reactivate_size_ratio_min = REACTIVATE_SIZE_RATIO_MIN
        self.reactivate_size_ratio_max = REACTIVATE_SIZE_RATIO_MAX
        self.reactivate_aspect_ratio_max = REACTIVATE_ASPECT_RATIO_MAX
        self.reactivate_dir_diff_max_deg = REACTIVATE_DIR_DIFF_MAX_DEG
        self.reactivate_gate_cost_inf = float(REACTIVATE_GATE_COST_INF)
        self.reid_color_lambda = REID_COLOR_LAMBDA
        self.reid_size_lambda = REID_SIZE_LAMBDA
        self.reid_dir_lambda = REID_DIR_LAMBDA
        self.reid_color_mismatch_penalty = REID_COLOR_MISMATCH_PENALTY
        self.reid_dir_speed_min = REID_DIR_SPEED_MIN
        self.reid_size_eps = REID_SIZE_EPS
        self.last_chance_lookback = LAST_CHANCE_LOOKBACK
        self.last_chance_center_dist_gate_m = LAST_CHANCE_CENTER_DIST_GATE_M
        self.last_chance_size_ratio_min = LAST_CHANCE_SIZE_RATIO_MIN
        self.last_chance_size_ratio_max = LAST_CHANCE_SIZE_RATIO_MAX
        self.last_chance_aspect_ratio_max = LAST_CHANCE_ASPECT_RATIO_MAX
        self.last_chance_dir_diff_max_deg = LAST_CHANCE_DIR_DIFF_MAX_DEG
        self.last_chance_color_gate = LAST_CHANCE_COLOR_GATE
        self.last_chance_dist_weight = LAST_CHANCE_DIST_WEIGHT
        self.last_chance_iou_weight = LAST_CHANCE_IOU_WEIGHT
        self.reactivate_size_rollback_enable = REACTIVATE_SIZE_ROLLBACK_ENABLE
        self.reactivate_size_rollback_frames = REACTIVATE_SIZE_ROLLBACK_FRAMES
        self.reactivate_size_rollback_ratio_min = REACTIVATE_SIZE_ROLLBACK_RATIO_MIN
        self.reactivate_size_rollback_ratio_max = REACTIVATE_SIZE_ROLLBACK_RATIO_MAX
        self.split_guard_enable = SPLIT_GUARD_ENABLE
        self.split_guard_close_dist_m = SPLIT_GUARD_CLOSE_DIST_M
        self.split_guard_frames = SPLIT_GUARD_FRAMES
        self.split_guard_dir_diff_max_deg = SPLIT_GUARD_DIR_DIFF_MAX_DEG
        self.frame_idx = 0
        self.cls_history_len = max(1, int(CLS_HISTORY_LEN))
        self.cls_vote_k_car_to_obs = max(1, min(int(CLS_VOTE_K_CAR_TO_OBS), self.cls_history_len))
        self.cls_vote_k_obs_to_car = max(1, min(int(CLS_VOTE_K_OBS_TO_CAR), self.cls_history_len))
        self.cls_car_to_obs_require_iou = bool(CLS_CAR_TO_OBS_REQUIRE_IOU)
        self.cls_car_to_obs_iou_min = float(CLS_CAR_TO_OBS_IOU_MIN)
        self.cls_car_to_obs_require_center_dist = bool(CLS_CAR_TO_OBS_REQUIRE_CENTER_DIST)
        self.cls_car_to_obs_center_dist_max_m = float(CLS_CAR_TO_OBS_CENTER_DIST_MAX_M)
        self.cls_car_to_obs_require_size_mismatch = bool(CLS_CAR_TO_OBS_REQUIRE_SIZE_MISMATCH)
        self.cls_car_to_obs_size_ratio_min = float(CLS_CAR_TO_OBS_SIZE_RATIO_MIN)
        self.cls_car_to_obs_size_ratio_max = float(CLS_CAR_TO_OBS_SIZE_RATIO_MAX)
        self.cls_car_to_obs_aspect_ratio_max = float(CLS_CAR_TO_OBS_ASPECT_RATIO_MAX)
        self.cls_car_to_obs_require_low_speed = bool(CLS_CAR_TO_OBS_REQUIRE_LOW_SPEED)
        self.cls_car_to_obs_speed_max = float(CLS_CAR_TO_OBS_SPEED_MAX)
        self.cls_car_to_obs_stop_frames = max(1, int(CLS_CAR_TO_OBS_STOP_FRAMES))
        self.prev_meas_count: Optional[int] = None
        self.last_metrics: Dict[str, int] = {}
        self.debug_logging = debug_logging
        self.last_debug_info: Dict[str, object] = {}

    def update(
        self,
        detections_carla: np.ndarray,
        detection_colors: Optional[List[Optional[str]]] = None,
    ) -> np.ndarray:
        """Update tracks with current detections and return active results."""
        if detections_carla is None:
            detections_carla = np.zeros((0, 6), dtype=float)
        detections_carla = np.asarray(detections_carla, dtype=float)
        self.last_matches = []
        self.frame_idx += 1

        predicted_states: List[dict] = []
        for track in self.tracks:
            cfg = self._config_for(track.cls)
            track.predict(cfg)
            if self.debug_logging and track.state != TrackState.DELETED:
                predicted_states.append({
                    "id": track.id,
                    "cls": float(track.cls),
                    "cx": float(track.kf_pos.x[0, 0]),
                    "cy": float(track.kf_pos.x[1, 0]),
                    "length": float(track.car_length),
                    "width": float(track.car_width),
                    "yaw": float(track.car_yaw),
                })

        active_tracks = [t for t in self.tracks if t.state != TrackState.DELETED]
        if self.split_guard_enable:
            self._update_split_guard(active_tracks)

        det_colors = self._prepare_detection_colors(detection_colors, len(detections_carla))
        unmatched_detections = set(range(len(detections_carla)))

        matched_stage1: List[Tuple[int, int]] = []
        matched_stage2: List[Tuple[int, int]] = []
        matched_last_chance: List[Tuple[int, int]] = []
        matched_track_ids: set = set()
        debug_reasons: Counter = Counter()
        centers_cost = None
        cost_stats_stage1 = None
        cost_stats_stage2 = None
        cost_stats_last = None
        cost_shape_stage1 = (0, 0)

        tracked_tracks = [t for t in active_tracks if t.state in (TrackState.CONFIRMED, TrackState.TENTATIVE)]
        if len(detections_carla) > 0 and tracked_tracks:
            iou_matrix = iou_batch(detections_carla, tracked_tracks, return_iou=True)
            cost_matrix = 1.0 - iou_matrix
            if self.assoc_center_weight > 0.0:
                centers_cost = self._center_distance_cost(detections_carla, tracked_tracks)
                cost_matrix = (1.0 - self.assoc_center_weight) * cost_matrix + self.assoc_center_weight * centers_cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                iou_val = iou_matrix[r, c]
                cfg = self._config_for(tracked_tracks[c].cls)
                if iou_val < self.iou_threshold:
                    debug_reasons["tracked_below_iou"] += 1
                    continue
                if cfg.jump_distance_gate:
                    if centers_cost is None:
                        centers_cost = self._center_distance_cost(detections_carla, tracked_tracks)
                    if centers_cost[r, c] > cfg.jump_distance_gate / self.assoc_center_norm:
                        debug_reasons["tracked_jump_gate"] += 1
                        continue
                matched_stage1.append((r, c))
                unmatched_detections.discard(r)
            flat = cost_matrix.flatten()
            if flat.size > 0:
                cost_stats_stage1 = {
                    "min": float(np.min(flat)),
                    "mean": float(np.mean(flat)),
                    "max": float(np.max(flat)),
                }
            cost_shape_stage1 = cost_matrix.shape
        elif self.debug_logging:
            cost_stats_stage1 = {"min": None, "mean": None, "max": None}

        for det_idx, track_idx in matched_stage1:
            track = tracked_tracks[track_idx]
            det = detections_carla[det_idx]
            det_iou = self._det_track_iou(det, track)
            det_dist = self._center_distance_m(det, track)
            color = det_colors[det_idx] if det_colors else None
            track.update(det, color=color, cfg=self._config_for(track.cls))
            self._maybe_update_class(
                track,
                det[0],
                self.frame_idx,
                det_iou=det_iou,
                det_center_dist_m=det_dist,
                det=det,
            )
            self.last_matches.append((track.id, det_idx))
            matched_track_ids.add(track.id)

        lost_tracks = [t for t in active_tracks if t.state == TrackState.LOST]
        if unmatched_detections and lost_tracks:
            det_indices = list(unmatched_detections)
            det_subset = detections_carla[det_indices]
            iou_matrix = iou_batch(det_subset, lost_tracks, return_iou=True)
            cost_matrix = (1.0 - iou_matrix).astype(float, copy=True)
            for i, det_idx in enumerate(det_indices):
                det = detections_carla[det_idx]
                det_color = det_colors[det_idx] if det_colors else None
                det_dir = self._det_direction_deg(det)
                for j, track in enumerate(lost_tracks):
                    dir_gate = self._dir_gate_for_track(self.reactivate_dir_diff_max_deg, track)
                    if not self._passes_match_gates(
                        det,
                        track,
                        det_color,
                        det_dir,
                        dist_gate_m=self.reactivate_center_dist_gate_m,
                        size_ratio_min=self.reactivate_size_ratio_min,
                        size_ratio_max=self.reactivate_size_ratio_max,
                        aspect_ratio_max=self.reactivate_aspect_ratio_max,
                        dir_diff_max_deg=dir_gate,
                        color_gate=False,
                        debug_reasons=debug_reasons,
                        reason_prefix="reactivate",
                    ):
                        cost_matrix[i, j] = self.reactivate_gate_cost_inf
                        continue
                    penalty = self.reid_color_lambda * self._reid_color_penalty(det_color, track.get_color())
                    penalty += self.reid_size_lambda * self._reid_size_penalty(det, track)
                    penalty += self.reid_dir_lambda * self._reid_dir_penalty(det_dir, track)
                    cost_matrix[i, j] += penalty

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] >= self.reactivate_gate_cost_inf:
                    continue
                if iou_matrix[r, c] < self.reactivate_iou_threshold:
                    debug_reasons["reactivate_below_iou"] += 1
                    continue
                det_idx = det_indices[r]
                matched_stage2.append((det_idx, c))
                unmatched_detections.discard(det_idx)
            flat = cost_matrix.flatten()
            if flat.size > 0:
                cost_stats_stage2 = {
                    "min": float(np.min(flat)),
                    "mean": float(np.mean(flat)),
                    "max": float(np.max(flat)),
                }
        elif self.debug_logging:
            cost_stats_stage2 = {"min": None, "mean": None, "max": None}

        for det_idx, track_idx in matched_stage2:
            track = lost_tracks[track_idx]
            det = detections_carla[det_idx]
            det_iou = self._det_track_iou(det, track)
            det_dist = self._center_distance_m(det, track)
            color = det_colors[det_idx] if det_colors else None
            self._mark_reactivated(track)
            track.update(det, color=color, cfg=self._config_for(track.cls))
            self._maybe_update_class(
                track,
                det[0],
                self.frame_idx,
                det_iou=det_iou,
                det_center_dist_m=det_dist,
                det=det,
            )
            self.last_matches.append((track.id, det_idx))
            matched_track_ids.add(track.id)

        measurement_drop = (
            self.prev_meas_count is not None
            and len(detections_carla) < self.prev_meas_count
            and len(detections_carla) < len(active_tracks)
        )

        for track in active_tracks:
            if track.id in matched_track_ids:
                continue
            cfg = self._config_for(track.cls)
            if measurement_drop and track.state == TrackState.CONFIRMED:
                pass
            elif track.state == TrackState.CONFIRMED:
                track.state = TrackState.LOST
            elif track.state == TrackState.LOST:
                if track.time_since_update > cfg.max_age:
                    track.state = TrackState.DELETED
            elif track.state == TrackState.TENTATIVE:
                track.state = TrackState.DELETED
            track.low_speed_streak = 0

        last_chance_tracks: List[Track] = []
        if unmatched_detections and self.last_chance_lookback > 0:
            last_chance_tracks = [
                t for t in self.tracks
                if t.state == TrackState.LOST and t.time_since_update <= self.last_chance_lookback
            ]
            if last_chance_tracks:
                det_indices = list(unmatched_detections)
                det_subset = detections_carla[det_indices]
                iou_matrix = iou_batch(det_subset, last_chance_tracks, return_iou=True)
                cost_matrix = np.full(
                    (len(det_indices), len(last_chance_tracks)),
                    self.reactivate_gate_cost_inf,
                    dtype=float,
                )
                for i, det_idx in enumerate(det_indices):
                    det = detections_carla[det_idx]
                    det_color = det_colors[det_idx] if det_colors else None
                    det_dir = self._det_direction_deg(det)
                    for j, track in enumerate(last_chance_tracks):
                        dir_gate = self._dir_gate_for_track(self.last_chance_dir_diff_max_deg, track)
                        if not self._passes_match_gates(
                            det,
                            track,
                            det_color,
                            det_dir,
                            dist_gate_m=self.last_chance_center_dist_gate_m,
                            size_ratio_min=self.last_chance_size_ratio_min,
                            size_ratio_max=self.last_chance_size_ratio_max,
                            aspect_ratio_max=self.last_chance_aspect_ratio_max,
                            dir_diff_max_deg=dir_gate,
                            color_gate=self.last_chance_color_gate,
                            debug_reasons=debug_reasons,
                            reason_prefix="last_chance",
                        ):
                            continue
                        dist = self._center_distance_m(det, track)
                        dist_norm = dist / max(self.last_chance_center_dist_gate_m, self.reid_size_eps)
                        iou_cost = 1.0 - iou_matrix[i, j]
                        cost_matrix[i, j] = (
                            self.last_chance_dist_weight * dist_norm
                            + self.last_chance_iou_weight * iou_cost
                        )

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] >= self.reactivate_gate_cost_inf:
                        continue
                    det_idx = det_indices[r]
                    matched_last_chance.append((det_idx, c))
                    unmatched_detections.discard(det_idx)
                flat = cost_matrix.flatten()
                if flat.size > 0:
                    cost_stats_last = {
                        "min": float(np.min(flat)),
                        "mean": float(np.mean(flat)),
                        "max": float(np.max(flat)),
                    }
        elif self.debug_logging:
            cost_stats_last = {"min": None, "mean": None, "max": None}

        for det_idx, track_idx in matched_last_chance:
            track = last_chance_tracks[track_idx]
            det = detections_carla[det_idx]
            det_iou = self._det_track_iou(det, track)
            det_dist = self._center_distance_m(det, track)
            color = det_colors[det_idx] if det_colors else None
            self._mark_reactivated(track)
            track.update(det, color=color, cfg=self._config_for(track.cls))
            self._maybe_update_class(
                track,
                det[0],
                self.frame_idx,
                det_iou=det_iou,
                det_center_dist_m=det_dist,
                det=det,
            )
            self.last_matches.append((track.id, det_idx))
            matched_track_ids.add(track.id)

        self._apply_reactivation_size_guard(debug_reasons)

        for det_idx in list(unmatched_detections):
            color = det_colors[det_idx] if det_colors else None
            cfg = self._config_for(detections_carla[det_idx][0] if len(detections_carla[det_idx]) > 0 else None)
            new_track = Track(
                bbox_init=detections_carla[det_idx],
                color=color,
                class_config=cfg,
            )
            new_track.last_cls_change_frame = int(self.frame_idx)
            self.tracks.append(new_track)

        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

        output_results = []
        for track in self.tracks:
            if track.state in (TrackState.CONFIRMED, TrackState.LOST):
                cfg = self._config_for(track.cls)
                stopped = track._is_stopped(cfg)
                state = track.get_state(
                    smooth_window=cfg.smooth_window,
                    ema_alpha=cfg.output_ema_alpha,
                    snap_eps=cfg.output_snap_eps,
                    stop_alpha_scale=cfg.stop_alpha_scale,
                    is_stopped=stopped,
                )
                output_results.append(np.array([track.id, *state], dtype=float))

        unmatched_track_ids = [
            int(track.id) for track in self.tracks
            if track.state != TrackState.DELETED and track.id not in matched_track_ids
        ]
        self.last_metrics = {
            "num_tracks": len(active_tracks),
            "num_measurements": len(detections_carla),
            "num_matched": len(matched_stage1) + len(matched_stage2) + len(matched_last_chance),
            "num_matched_stage1": len(matched_stage1),
            "num_matched_stage2": len(matched_stage2),
            "num_matched_last_chance": len(matched_last_chance),
            "num_unmatched_tracks": len(unmatched_track_ids),
            "num_unmatched_measurements": len(unmatched_detections),
        }
        self.prev_meas_count = len(detections_carla)
        matched_meas_classes = None
        if self.debug_logging:
            matched_meas_classes = [
                (int(track_id), int(detections_carla[det_idx][0]))
                for track_id, det_idx in self.last_matches
            ]
            self.last_debug_info = {
                "predicted_tracks": predicted_states,
                "matched": [(int(tid), int(det_idx)) for tid, det_idx in self.last_matches],
                "matched_stage1": [(int(tracked_tracks[c].id), int(r)) for r, c in matched_stage1],
                "matched_stage2": [(int(lost_tracks[c].id), int(det_idx)) for det_idx, c in matched_stage2],
                "matched_last_chance": [(int(last_chance_tracks[c].id), int(det_idx)) for det_idx, c in matched_last_chance],
                "matched_measured_classes": matched_meas_classes,
                "unmatched_tracks": unmatched_track_ids,
                "unmatched_detections": sorted(unmatched_detections),
                "unmatched_reasons": dict(debug_reasons),
                "cost_stats": cost_stats_stage1,
                "cost_stats_stage2": cost_stats_stage2,
                "cost_stats_last_chance": cost_stats_last,
                "cost_matrix_shape": cost_shape_stage1,
            }
        else:
            self.last_debug_info = {}
        return np.array(output_results) if output_results else np.array([])

    def _prepare_detection_colors(
        self,
        detection_colors: Optional[List[Optional[str]]],
        count: int,
    ) -> List[Optional[str]]:
        """
        검출 색상 라벨을 정규화하고, 검출 개수에 맞게 길이를 조정
        """
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

    def _center_distance_cost(self, detections_carla: np.ndarray, tracks: List["Track"]) -> np.ndarray:
        cost = np.zeros((len(detections_carla), len(tracks)), dtype=np.float32)
        for i, det in enumerate(detections_carla):
            dx = det[3] if len(det) > 3 else 1.0
            dy = det[4] if len(det) > 4 else dx
            norm = max(self.assoc_center_norm * 0.5 * (abs(dx) + abs(dy)), 1e-3)
            for j, track in enumerate(tracks):
                pred_xc, pred_yc = track.kf_pos.x[:2].flatten()
                dist = math.hypot(det[1] - pred_xc, det[2] - pred_yc) / norm
                cost[i, j] = float(dist)
        return cost

    def _center_distance_m(self, det: np.ndarray, track: "Track") -> float:
        pred_xc, pred_yc = track.kf_pos.x[:2].flatten()
        return float(math.hypot(det[1] - pred_xc, det[2] - pred_yc))

    def _det_track_iou(self, det: np.ndarray, track: "Track") -> float:
        det_aabb = carla_to_aabb(det)
        pred_xc, pred_yc = track.kf_pos.x[:2].flatten()
        temp_obb = np.array([0, pred_xc, pred_yc, track.car_length, track.car_width, track.car_yaw])
        pred_aabb = carla_to_aabb(temp_obb)
        return float(iou_bbox(det_aabb, pred_aabb))

    @staticmethod
    def _is_car_class(cls_val: int) -> bool:
        return int(cls_val) == 0

    def _size_mismatch(self, det: Optional[np.ndarray], track: "Track") -> bool:
        if det is None:
            return False
        ratio_min = self.cls_car_to_obs_size_ratio_min
        ratio_max = self.cls_car_to_obs_size_ratio_max
        if ratio_min > 0.0 and ratio_max > 0.0:
            if not self._size_ratio_ok(det, track, ratio_min, ratio_max):
                return True
        if self.cls_car_to_obs_aspect_ratio_max > 0.0:
            if not self._aspect_ratio_ok(det, track, self.cls_car_to_obs_aspect_ratio_max):
                return True
        return False

    def _maybe_update_class(
        self,
        track: "Track",
        det_cls_raw: Optional[float],
        frame_idx: int,
        det_iou: Optional[float] = None,
        det_center_dist_m: Optional[float] = None,
        det: Optional[np.ndarray] = None,
    ) -> None:
        if det_cls_raw is None:
            return
        try:
            det_cls = int(det_cls_raw)
        except (TypeError, ValueError):
            return

        if track.cls_state is None:
            track.cls_state = det_cls

        if self.cls_history_len > 0:
            track.cls_hist.append(det_cls)

        if self.cls_car_to_obs_require_low_speed:
            speed = track.get_speed()
            if speed <= self.cls_car_to_obs_speed_max:
                track.low_speed_streak += 1
            else:
                track.low_speed_streak = 0

        current_cls = int(track.cls_state)
        if det_cls == current_cls:
            return

        current_is_car = self._is_car_class(current_cls)
        target_is_car = self._is_car_class(det_cls)
        if current_is_car == target_is_car:
            return

        if len(track.cls_hist) < self.cls_history_len:
            return

        votes = sum(1 for cls in track.cls_hist if self._is_car_class(cls) == target_is_car)
        if current_is_car and not target_is_car:
            if votes < self.cls_vote_k_car_to_obs:
                return
            if self.cls_car_to_obs_require_iou:
                if det_iou is None or det_iou < self.cls_car_to_obs_iou_min:
                    return
            if self.cls_car_to_obs_require_center_dist:
                if det_center_dist_m is None or det_center_dist_m > self.cls_car_to_obs_center_dist_max_m:
                    return
            if self.cls_car_to_obs_require_size_mismatch:
                if not self._size_mismatch(det, track):
                    return
            if self.cls_car_to_obs_require_low_speed:
                if track.low_speed_streak < self.cls_car_to_obs_stop_frames:
                    return
        else:
            if votes < self.cls_vote_k_obs_to_car:
                return

        track.cls_state = det_cls
        track.last_cls_change_frame = int(frame_idx)
        track.cls_hist.clear()
        track.cls_hist.append(det_cls)

    def _track_direction_deg(self, track: "Track") -> Optional[float]:
        vx, vy = track.get_velocity()
        speed = math.hypot(vx, vy)
        if speed >= self.reid_dir_speed_min:
            return wrap_deg(math.degrees(math.atan2(vy, vx)))
        yaw = float(track.car_yaw)
        if not math.isfinite(yaw):
            return None
        return wrap_deg(yaw)

    def _det_direction_deg(self, det: np.ndarray) -> Optional[float]:
        if det is None or len(det) < 6:
            return None
        yaw = float(det[5])
        if not math.isfinite(yaw):
            return None
        return wrap_deg(yaw)

    def _dir_gate_for_track(self, base_gate: float, track: "Track") -> float:
        gate = base_gate
        if self.split_guard_enable and track.split_guard_frames > 0 and self.split_guard_dir_diff_max_deg > 0.0:
            if gate > 0.0:
                gate = min(gate, self.split_guard_dir_diff_max_deg)
            else:
                gate = self.split_guard_dir_diff_max_deg
        return gate

    def _size_ratio_ok(self, det: np.ndarray, track: "Track", ratio_min: float, ratio_max: float) -> bool:
        if ratio_min <= 0.0 or ratio_max <= 0.0:
            return True
        eps = self.reid_size_eps
        det_l = max(float(det[3]), eps)
        det_w = max(float(det[4]), eps)
        tr_l = max(float(track.car_length), eps)
        tr_w = max(float(track.car_width), eps)
        ratio_l = det_l / tr_l
        ratio_w = det_w / tr_w
        return ratio_min <= ratio_l <= ratio_max and ratio_min <= ratio_w <= ratio_max

    def _aspect_ratio_ok(self, det: np.ndarray, track: "Track", max_ratio: float) -> bool:
        if max_ratio <= 0.0:
            return True
        eps = self.reid_size_eps
        det_l = max(float(det[3]), eps)
        det_w = max(float(det[4]), eps)
        tr_l = max(float(track.car_length), eps)
        tr_w = max(float(track.car_width), eps)
        det_ar = det_l / max(det_w, eps)
        tr_ar = tr_l / max(tr_w, eps)
        ratio = det_ar / max(tr_ar, eps)
        ratio = max(ratio, 1.0 / max(ratio, eps))
        return ratio <= max_ratio

    def _passes_match_gates(
        self,
        det: np.ndarray,
        track: "Track",
        det_color: Optional[str],
        det_dir: Optional[float],
        dist_gate_m: float,
        size_ratio_min: float,
        size_ratio_max: float,
        aspect_ratio_max: float,
        dir_diff_max_deg: float,
        color_gate: bool,
        debug_reasons: Optional[Counter] = None,
        reason_prefix: str = "",
    ) -> bool:
        if dist_gate_m > 0.0:
            dist = self._center_distance_m(det, track)
            if dist > dist_gate_m:
                if debug_reasons is not None:
                    debug_reasons[f"{reason_prefix}_dist_gate"] += 1
                return False
        if not self._size_ratio_ok(det, track, size_ratio_min, size_ratio_max):
            if debug_reasons is not None:
                debug_reasons[f"{reason_prefix}_size_gate"] += 1
            return False
        if not self._aspect_ratio_ok(det, track, aspect_ratio_max):
            if debug_reasons is not None:
                debug_reasons[f"{reason_prefix}_aspect_gate"] += 1
            return False
        if dir_diff_max_deg > 0.0:
            track_dir = self._track_direction_deg(track)
            if det_dir is not None and track_dir is not None:
                diff = abs(wrap_deg(det_dir - track_dir))
                if diff > dir_diff_max_deg:
                    if debug_reasons is not None:
                        debug_reasons[f"{reason_prefix}_dir_gate"] += 1
                    return False
        if color_gate:
            track_color = track.get_color()
            if det_color and track_color and det_color != track_color:
                if debug_reasons is not None:
                    debug_reasons[f"{reason_prefix}_color_gate"] += 1
                return False
        return True

    def _reid_color_penalty(self, det_color: Optional[str], track_color: Optional[str]) -> float:
        if not det_color or not track_color:
            return 0.0
        if det_color == track_color:
            return 0.0
        return float(self.reid_color_mismatch_penalty)

    def _reid_size_penalty(self, det: np.ndarray, track: "Track") -> float:
        eps = self.reid_size_eps
        det_l = max(float(det[3]), eps)
        det_w = max(float(det[4]), eps)
        tr_l = max(float(track.car_length), eps)
        tr_w = max(float(track.car_width), eps)
        ratio_l = det_l / tr_l
        ratio_w = det_w / tr_w
        return 0.5 * (abs(math.log(ratio_l)) + abs(math.log(ratio_w)))

    def _reid_dir_penalty(self, det_dir: Optional[float], track: "Track") -> float:
        if det_dir is None:
            return 0.0
        track_dir = self._track_direction_deg(track)
        if track_dir is None:
            return 0.0
        diff = abs(wrap_deg(det_dir - track_dir))
        return diff / float(YAW_PERIOD_DEG)

    def _mark_reactivated(self, track: "Track") -> None:
        if not self.reactivate_size_rollback_enable or self.reactivate_size_rollback_frames <= 0:
            return
        track.reactivation_guard_frames = self.reactivate_size_rollback_frames
        track.reactivation_size_ref = (float(track.car_length), float(track.car_width))

    def _apply_reactivation_size_guard(self, debug_reasons: Optional[Counter] = None) -> None:
        if not self.reactivate_size_rollback_enable or self.reactivate_size_rollback_frames <= 0:
            return
        for track in self.tracks:
            if track.reactivation_guard_frames <= 0:
                continue
            ref = track.reactivation_size_ref
            if not ref:
                track.reactivation_guard_frames = max(track.reactivation_guard_frames - 1, 0)
                continue
            ref_l, ref_w = ref
            eps = self.reid_size_eps
            ratio_l = max(float(track.car_length), eps) / max(float(ref_l), eps)
            ratio_w = max(float(track.car_width), eps) / max(float(ref_w), eps)
            if (
                ratio_l < self.reactivate_size_rollback_ratio_min
                or ratio_l > self.reactivate_size_rollback_ratio_max
                or ratio_w < self.reactivate_size_rollback_ratio_min
                or ratio_w > self.reactivate_size_rollback_ratio_max
            ):
                track.state = TrackState.LOST
                track.time_since_update = max(track.time_since_update, 1)
                track.reactivation_guard_frames = 0
                track.reactivation_size_ref = None
                if debug_reasons is not None:
                    debug_reasons["reactivate_size_rollback"] += 1
                continue
            track.reactivation_guard_frames -= 1
            if track.reactivation_guard_frames <= 0:
                track.reactivation_size_ref = None

    def _update_split_guard(self, tracks: List["Track"]) -> None:
        if not self.split_guard_enable or self.split_guard_frames <= 0:
            return
        for track in tracks:
            if track.split_guard_frames > 0:
                track.split_guard_frames -= 1
        candidates = [t for t in tracks if t.state in (TrackState.CONFIRMED, TrackState.TENTATIVE)]
        for i in range(len(candidates)):
            ti = candidates[i]
            xi, yi = ti.kf_pos.x[:2].flatten()
            for j in range(i + 1, len(candidates)):
                tj = candidates[j]
                xj, yj = tj.kf_pos.x[:2].flatten()
                if math.hypot(xi - xj, yi - yj) <= self.split_guard_close_dist_m:
                    ti.split_guard_frames = max(ti.split_guard_frames, self.split_guard_frames)
                    tj.split_guard_frames = max(tj.split_guard_frames, self.split_guard_frames)

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

    def get_last_metrics(self) -> Dict[str, int]:
        return dict(self.last_metrics)

    def get_last_debug_info(self) -> Dict[str, object]:
        return dict(self.last_debug_info)

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
            cfg = self._config_for(track.cls)
            state_vec = track.get_state(
                smooth_window=cfg.smooth_window,
                ema_alpha=cfg.output_ema_alpha,
                snap_eps=cfg.output_snap_eps,
                stop_alpha_scale=cfg.stop_alpha_scale,
                is_stopped=track._is_stopped(cfg),
            )
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

    def force_flip_yaw(self, track_id: int, offset_deg: float = YAW_PERIOD_DEG) -> bool:
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

    def _config_for(self, cls_id: Optional[float]) -> TrackerConfigBase:
        if cls_id is None:
            return self.default_config
        if int(cls_id) == 0:
            return self.config_car
        return self.config_obstacle


def iou_batch(detections_carla: np.ndarray, tracks: List["Track"], return_iou: bool = False) -> np.ndarray:
    """
    Calculate IoU or cost matrix between detections and predicted tracks.
    If return_iou is True, returns IoU values; otherwise returns 1 - IoU cost.
    """
    cost_matrix = np.zeros((len(detections_carla), len(tracks)), dtype=np.float32)
    for i, det_carla in enumerate(detections_carla):
        det_aabb = carla_to_aabb(det_carla)
        for j, track in enumerate(tracks):
            pred_xc, pred_yc = track.kf_pos.x[:2].flatten()
            temp_obb = np.array([0, pred_xc, pred_yc, track.car_length, track.car_width, track.car_yaw])
            pred_aabb = carla_to_aabb(temp_obb)
            iou_val = iou_bbox(det_aabb, pred_aabb)
            cost_matrix[i, j] = iou_val if return_iou else 1.0 - iou_val
    return cost_matrix
