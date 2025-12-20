import json
import queue
import socket
import threading
from typing import Optional


class CommandServer:
    """
    간단한 TCP 명령 서버. 각 연결은 JSON 한 줄을 보내고 응답을 받고 종료
    yaw/색상 명령 처리를 위해 RealtimeServer에서 사용됨
    """

    def __init__(self, host: str, port: int, command_queue: "queue.Queue[dict]", response_timeout: float = 2.0):
        self.host = host
        self.port = int(port)
        self.q = command_queue
        self.response_timeout = response_timeout
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._serve_forever, daemon=True)
        self._thread.start()
        print(f"[CommandServer] listening on {self.host}:{self.port}")

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass

    def _serve_forever(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((self.host, self.port))
                srv.listen()
                self._sock = srv
                while self._running:
                    try:
                        conn, addr = srv.accept()
                    except OSError:
                        break
                    threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
        except Exception as exc:
            print(f"[CommandServer] server error: {exc}")

    def _handle_client(self, conn: socket.socket, addr):
        with conn:
            try:
                data = self._recv_all(conn)
                if not data:
                    return
                response = self._process_payload(data)
            except Exception as exc:
                response = {"status": "error", "message": str(exc)}
            try:
                payload = (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
                conn.sendall(payload)
            except Exception:
                pass

    def _recv_all(self, conn: socket.socket) -> str:
        buf = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b"\n" in chunk:
                break
        return buf.decode("utf-8").strip()

    def _process_payload(self, data: str) -> dict:
        if not data:
            return {"status": "error", "message": "empty payload"}
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return {"status": "error", "message": "invalid json"}
        if not isinstance(payload, dict):
            return {"status": "error", "message": "payload must be object"}
        cmd = payload.get("cmd")
        if not cmd:
            return {"status": "error", "message": "cmd required"}
        response_q: queue.Queue = queue.Queue(maxsize=1)
        item = {"cmd": cmd, "payload": payload, "response": response_q}
        try:
            self.q.put_nowait(item)
        except queue.Full:
            return {"status": "error", "message": "server busy"}
        try:
            return response_q.get(timeout=self.response_timeout)
        except queue.Empty:
            return {"status": "error", "message": "command timeout"}
