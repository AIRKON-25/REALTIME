#!/usr/bin/env python3
"""
UDP packet capture utility for edge_http.py streams.

Listens on a UDP port, stores each packet with the server-side receive
timestamp so logs can be replayed later in a time-aligned way.
Output format: NDJSON, one object per line:
{
  "recv_ts": <float>,            # server receive time (seconds)
  "remote": ["ip", port],        # sender address
  "payload_b64": "<base64>",     # raw UDP payload
  "payload_len": <int>,          # raw length in bytes
  "camera_id": <int|null>,       # optional, parsed from payload
  "timestamp": <float|null>,     # optional, parsed (edge send time)
  "capture_ts": <float|null>     # optional, parsed (camera capture time)
}
"""

import argparse
import base64
import json
import socket
import time


def parse_args():
    ap = argparse.ArgumentParser("Capture UDP packets from edge_http.py")
    ap.add_argument("--host", default="0.0.0.0", help="Bind host")
    ap.add_argument("--port", type=int, default=50050, help="Bind port")
    ap.add_argument("--out", required=True, help="NDJSON output path")
    ap.add_argument("--max-bytes", type=int, default=65507, help="Recv buffer size")
    ap.add_argument("--no-parse", action="store_true", help="Do not parse JSON payload for meta")
    return ap.parse_args()


def maybe_parse_payload(data: bytes, disable_parse: bool):
    if disable_parse:
        return {}
    try:
        msg = json.loads(data.decode("utf-8"))
    except Exception:
        return {}
    if not isinstance(msg, dict):
        return {}
    return {
        "camera_id": msg.get("camera_id"),
        "timestamp": msg.get("timestamp"),
        "capture_ts": msg.get("capture_ts"),
    }


def main():
    args = parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    print(f"[udp_capture] listening on {args.host}:{args.port}, writing to {args.out}")

    with open(args.out, "a", encoding="utf-8") as fp:
        try:
            while True:
                data, addr = sock.recvfrom(args.max_bytes)
                recv_ts = time.time()
                entry = {
                    "recv_ts": recv_ts,
                    "remote": [addr[0], addr[1]],
                    "payload_b64": base64.b64encode(data).decode("ascii"),
                    "payload_len": len(data),
                }
                entry.update(maybe_parse_payload(data, args.no_parse))
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fp.flush()
        except KeyboardInterrupt:
            print("\n[udp_capture] stopped")
        finally:
            try:
                sock.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
