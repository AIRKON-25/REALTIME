import argparse
import time

from realtime.server_core import RealtimeFusionServer, parse_cam_ports

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam-ports", default="cam1:50050,cam2:50051")
    # ap.add_argument("--cam-positions-json", default="camera_position.json")
    ap.add_argument("--local-ply-dir", default="outputs",
                    help="카메라별 로컬 PLY를 찾을 루트(패턴: cam_<id>_*.ply)")
    ap.add_argument("--local-lut-dir", default="outputs",
                    help="카메라별 LUT(npz)를 찾을 루트(패턴: cam_<id>_*.npz)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--iou-thr", type=float, default=0.01)
    ap.add_argument("--udp-port", type=int, default=50050)
    ap.add_argument("--tx-host", default=None)
    ap.add_argument("--tx-port", type=int, default=60050)
    ap.add_argument("--tx-protocol", choices=["udp", "tcp"], default="udp")
    ap.add_argument("--carla-host", default=None)
    ap.add_argument("--carla-port", type=int, default=61000)
    ap.add_argument("--global-ply", default="pointcloud/real_coshow_map_1127.ply")
    ap.add_argument("--vehicle-glb", default="pointcloud/car.glb")
    ap.add_argument("--web-host", default="0.0.0.0")
    ap.add_argument("--web-port", type=int, default=18000)
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--overlay-base-url", type=str, default=None)
    ap.add_argument("--tracker-fixed-length", type=float, default=None)
    ap.add_argument("--tracker-fixed-width", type=float, default=None)
    ap.add_argument("--size-mode", choices=["bbox", "fixed", "mesh"], default="mesh")
    ap.add_argument("--fixed-length", type=float, default=4.5)
    ap.add_argument("--fixed-width", type=float, default=1.8)
    ap.add_argument("--height-scale", type=float, default=0.5,
                    help="bbox/fixed 모드에서 차량 높이 = width * height_scale")
    ap.add_argument("--mesh-scale", type=float, default=1.0,
                    help="size-mode=mesh 일 때 GLB에 곱할 크기 배율")
    ap.add_argument("--mesh-height", type=float, default=0.0,
                    help="size-mode=mesh 일 때 지면에서 높이 보정값(0이면 mesh-scale 적용)")
    ap.add_argument("--z-offset", type=float, default=0.0,
                    help="모든 박스에 추가할 z 오프셋")
    ap.add_argument("--invert-bev-y", dest="invert_bev_y", action="store_true")
    ap.add_argument("--no-invert-bev-y", dest="invert_bev_y", action="store_false")
    ap.set_defaults(invert_bev_y=True)
    ap.add_argument("--normalize-vehicle", dest="normalize_vehicle", action="store_true")
    ap.add_argument("--no-normalize-vehicle", dest="normalize_vehicle", action="store_false")
    ap.set_defaults(normalize_vehicle=True)
    ap.add_argument("--vehicle-y-up", dest="vehicle_y_up", action="store_true")
    ap.add_argument("--vehicle-z-up", dest="vehicle_y_up", action="store_false")
    ap.set_defaults(vehicle_y_up=True)
    ap.add_argument("--flip-ply-y", dest="flip_ply_y", action="store_true",
                    help="global ply의 Y축을 반전하여 로드")
    ap.add_argument("--no-flip-ply-y", dest="flip_ply_y", action="store_false",
                    help="global ply Y축 반전하지 않음")
    ap.set_defaults(flip_ply_y=True)
    ap.add_argument("--flip-marker-x", dest="flip_marker_x", action="store_true")
    ap.add_argument("--no-flip-marker-x", dest="flip_marker_x", action="store_false")
    ap.set_defaults(flip_marker_x=False)
    ap.add_argument("--flip-marker-y", dest="flip_marker_y", action="store_true")
    ap.add_argument("--no-flip-marker-y", dest="flip_marker_y", action="store_false")
    ap.set_defaults(flip_marker_y=True)
    ap.add_argument("--cmd-host", default="0.0.0.0",
                    help="yaw/색상 명령 서버 바인드 호스트(미입력시 비활성)")
    ap.add_argument("--cmd-port", type=int, default=18100,
                    help="yaw/색상 명령 서버 포트")

    args = ap.parse_args()

    cam_ports = parse_cam_ports(args.cam_ports)

    server = RealtimeFusionServer(
        cam_ports=cam_ports,
        # cam_positions_path=args.cam_positions_json,
        local_ply_dir=args.local_ply_dir,
        local_lut_dir=args.local_lut_dir,
        fps=args.fps,
        iou_cluster_thr=args.iou_thr,
        single_port=args.udp_port,
        tx_host=args.tx_host,
        tx_port=args.tx_port,
        tx_protocol=args.tx_protocol,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        global_ply=args.global_ply,
        vehicle_glb=args.vehicle_glb,
        tracker_fixed_length=args.tracker_fixed_length,
        tracker_fixed_width=args.tracker_fixed_width,
        command_host=args.cmd_host,
        command_port=args.cmd_port,
        flip_ply_y=args.flip_ply_y,
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
