import argparse
import time

from realtime.server_core import RealtimeServer
from utils.tracking._constants import IOU_CLUSTER_THR

def main():
    ap = argparse.ArgumentParser() # 현장에서 바로 바뀌는 파라미터들만 인자로 빼자
    # ap.add_argument("--cam-positions-json", default="camera_position.json")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--iou-cluster-thr", type=float, default=IOU_CLUSTER_THR)
    ap.add_argument("--tracker-fixed-length", type=float, default=None)
    ap.add_argument("--tracker-fixed-width", type=float, default=None)
    ap.add_argument("--size-mode", choices=["bbox", "fixed", "mesh"], default="mesh")
    ap.add_argument("--fixed-length", type=float, default=4.5)
    ap.add_argument("--fixed-width", type=float, default=1.8)
    ap.add_argument("--udp-port", type=int, default=50050, help="단일 UDP 포트로 모든 카메라 데이터 수신")
    ap.add_argument("--tx-host", default=None, help="트래킹 결과 전송 호스트(제어컴)")
    ap.add_argument("--tx-port", type=int, default=60050)
    ap.add_argument("--tx-protocol", choices=["udp", "tcp"], default="udp")
    ap.add_argument("--carla-host", default=None, help="CARLA 서버 전송 호스트(우리 이제 안쓰지않나)")
    ap.add_argument("--carla-port", type=int, default=61000)
    ap.add_argument("--web-host", default="0.0.0.0", help="웹소켓 서버 바인드 호스트")
    ap.add_argument("--web-port", type=int, default=18000)
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--cmd-host", default="0.0.0.0", help="yaw/색상 명령 서버 바인드 호스트(미입력시 비활성)")
    ap.add_argument("--cmd-port", type=int, default=18100, help="yaw/색상 명령 서버 포트")
    args = ap.parse_args()

    cam_ports = {
        "cam1": 101,
        "cam2": 102,
        "cam3": 103,
        "cam4": 104,
    }

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
        tracker_fixed_length=args.tracker_fixed_length,
        tracker_fixed_width=args.tracker_fixed_width,
        command_host=args.cmd_host,
        command_port=args.cmd_port,
        ws_host=None if args.no_web else args.web_host,
        ws_port=args.web_port,
    )
    server.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.receiver.stop()
        if server.track_tx:
            server.track_tx.close()
        if server.carla_tx:
            server.carla_tx.close()


if __name__ == "__main__":
    main()
