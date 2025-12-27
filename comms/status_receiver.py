import json
import socket
import threading
from typing import Callable, Optional, Tuple


class StatusReceiver:
    """
    단일 포트에서 JSON 패킷을 수신해 콜백에 전달하는 간단한 UDP 리시버.

    콜백 시그니처: on_message(payload: dict, addr: Tuple[str, int])
    """

    def __init__(
        self,
        port: int,
        host: str = "0.0.0.0",
        max_bytes: int = 16384,
        log_packets: bool = False,
        on_message: Optional[Callable[[dict, Tuple[str, int]], None]] = None,
    ):
        self.host = host
        self.port = int(port)
        self.max_bytes = max_bytes
        self.log_packets = log_packets
        self.on_message = on_message
        self.sock: Optional[socket.socket] = None
        self.th: Optional[threading.Thread] = None
        self.running = False

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.th = threading.Thread(target=self._rx_loop, daemon=True)
        self.th.start()
        print(f"[StatusReceiver] listening on {self.host}:{self.port}")

    def stop(self) -> None:
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        if self.th:
            self.th.join(timeout=0.5)

    def _rx_loop(self) -> None:
        while self.running:
            try:
                data, addr = self.sock.recvfrom(self.max_bytes)
            except OSError:
                # 소켓이 닫힐 때 발생; 종료
                break
            except Exception as exc:
                print(f"[StatusReceiver] recv error: {exc}")
                continue
            try:
                payload = json.loads(data.decode("utf-8"))
            except Exception as exc:
                if self.log_packets:
                    print(f"[StatusReceiver] decode/json error: {exc} raw={data!r}")
                continue
            if self.log_packets:
                try:
                    print(f"[StatusReceiver] recv from={addr} type={payload.get('type')} payload={payload}")
                except Exception:
                    pass
            if self.on_message:
                try:
                    self.on_message(payload, addr)
                except Exception as exc:
                    print(f"[StatusReceiver] on_message error: {exc}")
