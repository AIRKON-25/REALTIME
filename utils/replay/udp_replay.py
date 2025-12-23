#!/usr/bin/env python3
"""
Replay captured UDP packets (from udp_capture.py) to mimic live edge_http.py streams.
"""

import argparse
import base64
import json
import socket
import time
from typing import List


def parse_args():
    ap = argparse.ArgumentParser("Replay NDJSON-captured UDP packets")
    ap.add_argument("--inp", required=True, help="NDJSON log path (from udp_capture.py)")
    ap.add_argument("--dst-host", default="127.0.0.1", help="Replay destination host")
    ap.add_argument("--dst-port", type=int, default=50050, help="Replay destination port")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed (1.0 = real-time)")
    ap.add_argument("--start-offset", type=float, default=0.0, help="Skip initial seconds of log")
    ap.add_argument("--limit", type=int, default=None, help="Max packets to send")
    ap.add_argument("--no-delay", action="store_true", help="Send as fast as possible (ignore timing)")
    return ap.parse_args()


def load_entries(path: str, start_offset: float, limit: int = None) -> List[dict]:
    entries: List[dict] = []
    first_ts = None
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "payload_b64" not in obj:
                continue
            ts = float(obj.get("recv_ts", 0.0))
            if first_ts is None:
                first_ts = ts
            if ts - first_ts < start_offset:
                continue
            entries.append(obj)
            if limit is not None and len(entries) >= limit:
                break
    return entries


def main():
    args = parse_args()
    entries = load_entries(args.inp, args.start_offset, args.limit)
    if not entries:
        print("[udp_replay] no entries to replay")
        return

    base_recv_ts = float(entries[0].get("recv_ts", time.time()))
    start_time = time.time()
    speed = max(1e-6, args.speed) if not args.no_delay else 1.0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(
        f"[udp_replay] replaying {len(entries)} packets to "
        f"{args.dst_host}:{args.dst_port} (speed={speed}x, no_delay={args.no_delay})"
    )

    try:
        for idx, obj in enumerate(entries):
            payload_b64 = obj.get("payload_b64")
            if not payload_b64:
                continue
            try:
                payload = base64.b64decode(payload_b64)
            except Exception:
                continue

            if not args.no_delay:
                pkt_ts = float(obj.get("recv_ts", base_recv_ts))
                target = start_time + (pkt_ts - base_recv_ts) / speed
                delay = target - time.time()
                if delay > 0:
                    time.sleep(delay)

            sock.sendto(payload, (args.dst_host, args.dst_port))

            if (idx + 1) % 100 == 0 or (idx + 1) == len(entries):
                print(f"[udp_replay] sent {idx + 1}/{len(entries)}")
    except KeyboardInterrupt:
        print("\n[udp_replay] interrupted")
    finally:
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()


"""

python udp_replay.py --inp utils/replay/session1.ndjson --dst-host 127.0.0.1 --dst-port 50050 --speed 1.0

"""