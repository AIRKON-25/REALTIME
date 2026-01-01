import argparse
import time

from realtime.server_core import RealtimeServer
from realtime.status_core import StatusServer
from utils.tracking.tracker import TrackerConfigCar, TrackerConfigObstacle
from utils.tracking._constants import (
    ASSOC_CENTER_NORM,
    ASSOC_CENTER_WEIGHT,
    IOU_CLUSTER_THR,
    CAR_MAX_AGE,
    CAR_MIN_HITS,
    OBSTACLE_MAX_AGE,
    OBSTACLE_MIN_HITS,
)

def main():
    ap = argparse.ArgumentParser() # 현장에서 바로 바뀌는 파라미터들만 인자로 빼자
    # ap.add_argument("--cam-positions-json", default="camera_position.json")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--iou-cluster-thr", type=float, default=IOU_CLUSTER_THR)
    ap.add_argument("--tracker-fixed-length", type=float, default=None)
    ap.add_argument("--tracker-fixed-width", type=float, default=None)
    ap.add_argument("--size-mode", choices=["bbox", "fixed", "mesh"], default="mesh")
    # ap.add_argument("--fixed-length", type=float, default=4.4) # 인지에서 사용하는 기본값
    # ap.add_argument("--fixed-width", type=float, default=2.7)
    ap.add_argument("--udp-port", type=int, default=50050, help="단일 UDP 포트로 모든 카메라 데이터 수신")
    ap.add_argument("--tx-host", default="192.168.0.100", help="트래킹 결과 전송 호스트(제어컴)")
    ap.add_argument("--tx-port", type=int, default=60050)
    ap.add_argument("--tx-protocol", choices=["udp", "tcp"], default="udp")
    ap.add_argument("--carla-host", default=None, help="CARLA 서버 전송 호스트(우리 이제 안쓰지않나)")
    ap.add_argument("--carla-port", type=int, default=61000)
    ap.add_argument("--web-host", default="0.0.0.0", help="웹소켓 서버 바인드 호스트")
    ap.add_argument("--web-port", type=int, default=18000)
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--debug-http-host", default="0.0.0.0", help="브로드캐스트 스냅샷용 디버그 HTTP 호스트")
    ap.add_argument("--debug-http-port", type=int, default=18110, help="브로드캐스트 스냅샷용 디버그 HTTP 포트")
    ap.add_argument("--no-debug-http", action="store_true", help="디버그 HTTP 엔드포인트 비활성화")
    ap.add_argument("--car-count", type=int, default=5, help="number of cars to lock into IDs 1..N (1-5)")
    ap.add_argument("--cmd-host", default="0.0.0.0", help="yaw/색상 명령 서버 바인드 호스트(미입력시 비활성)")
    ap.add_argument("--cmd-port", type=int, default=18100, help="yaw/색상 명령 서버 포트")
    ap.add_argument("--car-max-age", type=int, default=CAR_MAX_AGE)
    ap.add_argument("--car-min-hits", type=int, default=CAR_MIN_HITS)
    ap.add_argument("--obs-max-age", type=int, default=OBSTACLE_MAX_AGE)
    ap.add_argument("--obs-min-hits", type=int, default=OBSTACLE_MIN_HITS)
    ap.add_argument("--assoc-center-weight", type=float, default=ASSOC_CENTER_WEIGHT)
    ap.add_argument("--assoc-center-norm", type=float, default=ASSOC_CENTER_NORM)
    ap.add_argument("--debug-assoc", action="store_true", help="헝가리안 매칭/비용 행렬 디버그 로그")
    ap.add_argument("--log-pipeline", dest="log_pipeline", action="store_true", help="퓨전/트래킹 로그 출력")
    ap.add_argument("--no-log-pipeline", dest="log_pipeline", action="store_false", help="퓨전/트래킹 로그 비활성화")
    ap.add_argument("--log-udp-packets", dest="log_udp_packets", action="store_true", help="UDP 수신 패킷 로그 출력")
    ap.add_argument("--no-log-udp-packets", dest="log_udp_packets", action="store_false", help="UDP 수신 패킷 로그 비활성화")
    ap.add_argument("--lane-map-path", default="lanes.json", help="GT yaw lane map json path (lanes.json)")
    ap.add_argument("--status-udp-host", default="0.0.0.0", help="status receiver bind host")
    ap.add_argument("--status-udp-port", type=int, default=60070, help="status receiver UDP port")
    ap.add_argument("--status-http-host", default="0.0.0.0", help="status HTTP host (omit to disable HTTP)")
    ap.add_argument("--status-http-port", type=int, default=18080, help="status HTTP port (omit to disable HTTP)")
    ap.add_argument("--status-log-udp", action="store_true", help="log status UDP payloads")
    ap.add_argument("--dashboard", dest="dashboard", action="store_true", default=True, help="enable rich dashboard output")
    ap.add_argument("--no-dashboard", dest="dashboard", action="store_false", help="disable rich dashboard output")
    ap.add_argument("--dashboard-refresh-hz", type=float, default=4.0, help="dashboard refresh rate")
    ap.set_defaults(log_pipeline=True, log_udp_packets=False)
    args = ap.parse_args()
    if args.car_count < 1 or args.car_count > 5:
        ap.error("--car-count must be between 1 and 5")

    debug_http_host = None if args.no_debug_http else args.debug_http_host
    debug_http_port = None if args.no_debug_http else args.debug_http_port

    cam_ports = {
        "cam1": 101,
        "cam2": 102,
        "cam3": 103,
        "cam4": 104,
    }

    car_cfg = TrackerConfigCar(max_age=args.car_max_age, min_hits=args.car_min_hits)
    obs_cfg = TrackerConfigObstacle(max_age=args.obs_max_age, min_hits=args.obs_min_hits)

    status_server = StatusServer(
        udp_host=args.status_udp_host,
        udp_port=args.status_udp_port,
        log_packets=args.status_log_udp,
        http_host=args.status_http_host,
        http_port=args.status_http_port,
    )
    server = RealtimeServer(
        cam_ports=cam_ports,
        # cam_positions_path=args.cam_positions_json,
        fps=args.fps,
        iou_cluster_thr=args.iou_cluster_thr,
        single_port=args.udp_port,
        tx_host=args.tx_host,
        tx_port=args.tx_port,
        tx_protocol=args.tx_protocol,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        car_id_count=args.car_count,
        # tracker_fixed_length=args.tracker_fixed_length,
        # tracker_fixed_width=args.tracker_fixed_width,
        command_host=args.cmd_host,
        command_port=args.cmd_port,
        ws_host=None if args.no_web else args.web_host,
        ws_port=args.web_port,
        tracker_config_car=car_cfg,
        tracker_config_obstacle=obs_cfg,
        assoc_center_weight=args.assoc_center_weight,
        assoc_center_norm=args.assoc_center_norm,
        debug_assoc_logging=args.debug_assoc,
        log_pipeline=args.log_pipeline,
        log_udp_packets=args.log_udp_packets,
        lane_map_path=args.lane_map_path,
        status_state=status_server.state,
        debug_http_host=debug_http_host,
        debug_http_port=debug_http_port,
        dashboard=args.dashboard,
        dashboard_refresh_hz=args.dashboard_refresh_hz,
    )
    status_server.start()
    server.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        status_server.stop()
        server.receiver.stop()
        if server.track_tx:
            server.track_tx.close()
        if server.carla_tx:
            server.carla_tx.close()


if __name__ == "__main__":
    main()
