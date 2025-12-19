#!/usr/bin/env python3

import argparse
import json
import os
import socket
import sys
import time
from typing import List, Optional


def load_records(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[Replay] WARN: line {line_no} JSON decode error: {exc}", file=sys.stderr)
                continue
            payload = obj.get("payload")
            if payload is None:
                print(f"[Replay] WARN: line {line_no} missing 'payload', skipping", file=sys.stderr)
                continue
            records.append({
                "payload": payload,
                "delay_ms": obj.get("delay_ms"),
                "sent_ts": obj.get("sent_ts"),
                "capture_ts": obj.get("capture_ts"),
                "cam_id": obj.get("cam_id"),
            })
    return records


def compute_sleep_seconds(prev_sent_ts: Optional[float],
                          rec: dict,
                          fps: Optional[float]) -> float:
    if fps is not None and fps > 0:
        return 1.0 / fps
    delay_ms = rec.get("delay_ms")
    if delay_ms is not None:
        return max(0.0, float(delay_ms) / 1000.0)
    sent_ts = rec.get("sent_ts")
    if prev_sent_ts is not None and sent_ts is not None:
        return max(0.0, float(sent_ts) - float(prev_sent_ts))
    return 0.0


def run(args):
    if not os.path.isfile(args.file):
        raise FileNotFoundError(f"NDJSON file not found: {args.file}")

    records = load_records(args.file)
    if not records:
        print("[Replay] No records to send.", file=sys.stderr)
        return

    addr = (args.udp_host, args.udp_port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[Replay] Loaded {len(records)} records. Sending to {addr} (Ctrl+C to stop).")
    if args.fps is not None:
        print(f"[Replay] Overriding timing with FPS={args.fps:.3f}")

    try:
        while True:
            prev_sent_ts = None
            for idx, rec in enumerate(records):
                sleep_s = compute_sleep_seconds(prev_sent_ts, rec, args.fps)
                if sleep_s > 0:
                    time.sleep(sleep_s)

                payload_bytes = str(rec["payload"]).encode("utf-8")
                sock.sendto(payload_bytes, addr)
                prev_sent_ts = rec.get("sent_ts", prev_sent_ts)

                if args.verbose:
                    preview = payload_bytes.decode("utf-8", errors="replace")
                    print(f"[Replay] #{idx:05d} sent ({len(payload_bytes)} bytes): {preview}")

            if not args.loop:
                break
    except KeyboardInterrupt:
        print("[Replay] Interrupted, stopping.")
    finally:
        try:
            sock.close()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser("Replay UDP payloads recorded with --record-udp")
    parser.add_argument("--file", required=True, help="NDJSON file produced by edge_http.py --record-udp")
    parser.add_argument("--udp-host", default="127.0.0.1", help="Target host to send UDP payloads to")
    parser.add_argument("--udp-port", type=int, default=50050, help="Target port to send UDP payloads to")
    parser.add_argument("--fps", type=float, default=None,
                        help="Force a constant FPS; if unset uses recorded delay_ms/sent_ts intervals")
    parser.add_argument("--loop", action="store_true", help="Loop playback until interrupted")
    parser.add_argument("--verbose", action="store_true", help="Print each payload as it is sent")
    return parser.parse_args()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
