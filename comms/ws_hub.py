import asyncio
import json
import threading
from typing import Awaitable, Callable, Optional

import websockets  # pip install websockets


class WebSocketHub:
    """
    프론트엔드(React)와 실시간으로 JSON을 주고받는 간단한 WebSocket 서버.
    ws://<host>:<port>/monitor 로 접속하면 된다.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9001,
        path: str = "/monitor",
        on_message: Optional[
            Callable[[str, "websockets.WebSocketServerProtocol"], Awaitable[Optional[dict]]]
        ] = None,
    ):
        self.host = host
        self.port = int(port)
        self.path = path
        self._on_message = on_message
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients: "set[websockets.WebSocketServerProtocol]" = set()
        self._thread: Optional[threading.Thread] = None
        self._initial_messages: list[dict] = []
        self._initial_lock = threading.Lock()

    async def _handler(self, websocket):
        # websockets 15.x에서는 websocket.path 가 None 또는 '' 인 경우가 많음
        # 따라서 경로 체크는 제거하는 것이 안전하다.
        self._clients.add(websocket)
        print(f"[WebSocketHub] client connected ({len(self._clients)} total)")
        with self._initial_lock:
            initial_messages = list(self._initial_messages)
        if initial_messages:
            for message in initial_messages:
                try:
                    await websocket.send(json.dumps(message, ensure_ascii=False))
                except Exception as exc:
                    print(f"[WebSocketHub] initial send failed: {exc}")
                    break

        try:
            async for raw in websocket:
                if not self._on_message:
                    continue
                try:
                    response = await self._on_message(raw, websocket)
                except Exception as exc:
                    print(f"[WebSocketHub] on_message error: {exc}")
                    continue
                if response is None:
                    continue
                try:
                    await websocket.send(json.dumps(response, ensure_ascii=False))
                except Exception as exc:
                    print(f"[WebSocketHub] response send failed: {exc}")
                    break
        except Exception as exc:
            print(f"[WebSocketHub] client error: {exc}")
        finally:
            self._clients.discard(websocket)
            print(f"[WebSocketHub] client disconnected ({len(self._clients)} remaining)")

    async def _run_server(self):
        async with websockets.serve(self._handler, self.host, self.port):
            print(f"[WebSocketHub] listening on ws://{self.host}:{self.port}{self.path}")
            await asyncio.Future()  # run forever

    def start(self):
        if self._thread:
            return

        def _runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_server())

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    def set_initial_messages(self, messages: list[dict]):
        with self._initial_lock:
            self._initial_messages = list(messages)

    def broadcast(self, message: dict):
        """
        모든 연결된 클라이언트에 JSON 메시지를 전송한다.
        message는 dict로 주면 내부에서 json.dumps 한다.
        """
        if not self._loop or not self._clients:
            return
        data = json.dumps(message, ensure_ascii=False)

        async def _send_all():
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

        asyncio.run_coroutine_threadsafe(_send_all(), self._loop)
