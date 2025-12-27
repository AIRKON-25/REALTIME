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
PATH_REID_DIST_M = 5   # 경로 기반 재ID 거리 게이트(m)
PATH_REID_YAW_DEG = 45.0    # 경로 기반 재ID yaw 차이 게이트(도)
