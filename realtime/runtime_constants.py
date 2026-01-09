# 하이퍼파라미터를 여기에 정의
# 주석으로 빡세게 분리, 이름 명확히 하기

# color
COLOR_BIAS_STRENGTH = 0.3 # 색상 보정 강도 (0~1)
COLOR_BIAS_MIN_VOTES = 2 # 색상 보정에 필요한 최소 누적 관측 수

# size
vehicle_fixed_length = 2.5  # 차량 고정 길이 기본값
vehicle_fixed_width = 1.4   # 차량 고정 너비 기본값
rubberCone_fixed_length = 1.6  # 라바콘 고정 길이
rubberCone_fixed_width = 1.6   # 라바콘 고정 너비
barricade_fixed_length = 1.0  # 바리케이드 고정 길이
barricade_fixed_width = 0.5   # 바리케이드 고정 너비

# path progress
PATH_PROGRESS_DIST_M = 1.5  # 경로 진행으로 간주할 최대 거리 게이트(m)
PATH_PROGRESS_HEAD_WINDOW = 10  # 경로 진행 판단 시 path_future 앞쪽에서만 찾을 최대 점 개수
PATH_REID_DIST_M = 5   # 경로 기반 재ID 거리 게이트(m)
PATH_REID_YAW_DEG = 45.0    # 경로 기반 재ID yaw 차이 게이트(도)
ROUTE_CHANGE_DISTANCE_M = 15.0  # routeChanged 판정을 위한 거리 게이트(m)
ROUTE_CHANGE_DRAW_POINTS_PER_SEC = 5.0  # 경로를 웹에서 초당 그리는 점 개수 추정치
ROUTE_CHANGE_EXTRA_SECONDS = 2.0  # 경로 표시 버퍼 시간(초)
ROUTE_CHANGE_MIN_SECONDS = 1.0  # 최소 표시 시간(초)

# 조기 ID 해제/재매칭 (LOST 상태 핸드오프) 파라미터
EARLY_RELEASE_LOST_FRAMES = 10  # LOST 경과 프레임이 이 이상이면 외부 ID 조기 해제 시도
EARLY_REID_DIST_M = 3.0         # 조기 해제된 ID를 붙일 때 거리 게이트(보수적)
EARLY_REID_YAW_DEG = 30.0       # 조기 해제된 ID를 붙일 때 yaw 게이트(보수적)

# velocity smoothing (외부 전송용)
VELOCITY_EWMA_ALPHA = 0.2  # 최근 속도 가중치(0~1, 클수록 민감)
VELOCITY_DT_MIN = 0.05      # 속도 계산에 사용할 최소 시간 간격(초)
VELOCITY_DT_MAX = 1.0       # 이보다 오래되면 리셋
VELOCITY_SPEED_WINDOW = 5  # 속도 크기 중앙값 필터 윈도우 길이
VELOCITY_MAX_MPS = 25.0       # 속도 크기 상한(스파이크 클램프)
VELOCITY_ZERO_THRESH = 0.1  # 이 미만이면 0으로 취급

#EXT_COLOR_ID = {1: "red", 2: "yellow", 3: "green", 4: "purple", 5: "white"}

# 건들면 혼남
YELLOW = 'yellow'
PURPLE = 'purple'
WHITE = 'white'
GREEN = 'green'
RED = 'red'
# 건들면 혼남

# 여기서 바꾸삼
EXT_COLOR_ID = {
    1: YELLOW, 
    2: GREEN, 
    3: PURPLE, 
    4: WHITE, 
    5: RED
}
