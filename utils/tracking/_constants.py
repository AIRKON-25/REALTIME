# tracking 관련 상수들 정의

# 이럼 이제 맨 처음 정지되어있을때는 못잡을거임,, 랜덤임 음음음
# 최소 이동량(미터 단위 추정). 이보다 작으면 정지로 간주해 yaw 보정 생략
FORWARD_HEADING_MIN_DIST = 0.001
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

# Tracking defaults and shared thresholds
DEFAULT_CONFIRM_HITS = 3
DEFAULT_COLOR_LOCK_STREAK = 5
DEFAULT_MAX_AGE = 3
DEFAULT_MIN_HITS = 3
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_COLOR_PENALTY = 0.3
YAW_PERIOD_DEG = 180.0
HEADING_FLIP_THRESHOLD = 90.0
KF_STATE_INIT_COV_SCALE = 10.0

