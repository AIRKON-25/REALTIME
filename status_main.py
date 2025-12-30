#!/usr/bin/env python3
import argparse
import time

from realtime.status_core import StatusServer


def main():
    ap = argparse.ArgumentParser(description="Status receiver for udp_status_broadcaster.py")
    ap.add_argument("--udp-host", default="0.0.0.0", help="UDP bind host (default: 0.0.0.0)")
    ap.add_argument("--udp-port", type=int, default=60070, help="UDP listen port (default: 60070)")
    ap.add_argument("--log-udp", action="store_true", help="Log raw UDP payloads")
    ap.add_argument("--http-host", default=None, help="HTTP bind host (omit to disable)")
    ap.add_argument("--http-port", type=int, default=None, help="HTTP port (omit to disable)")
    args = ap.parse_args()

    server = StatusServer(
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        log_packets=args.log_udp,
        http_host=args.http_host,
        http_port=args.http_port,
    )
    server.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
