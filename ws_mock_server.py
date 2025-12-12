import asyncio
import json
import random
import time

import websockets  # pip install websockets

clients = set()


async def handler(ws):
    clients.add(ws)
    print("[MockWS] client connected, total:", len(clients))
    try:
        async for _ in ws:
            # 클라이언트에서 오는 메시지는 무시 (단방향 브로드캐스트)
            pass
    except Exception as exc:
        print("[MockWS] client error:", exc)
    finally:
        clients.discard(ws)
        print("[MockWS] client disconnected, total:", len(clients))


def make_snapshot() -> dict:
    """
    프론트에서 기대하는 ServerMessage 형태:
      { type: "snapshot", payload: { ...ServerSnapshot } }
    만약 타입이 안 맞으면 App.tsx에서 JSON 파싱은 되지만
    state 업데이트가 안 될 수 있으니, types.ts 구조와 맞춰줌.
    """

    cars_on_map = []
    cars_status = []

    # 차량 3대 랜덤 생성
    for i in range(3):
        tid = i + 1
        car_id = f"car-{tid}"
        map_id = f"mcar-{tid}"

        # 0~1 사이 랜덤 위치
        x = random.random()
        y = random.random()
        yaw = random.uniform(0, 360)

        color_choices = ["#f52629", "#48ad0d", "#3b82f6", "#facc15"]
        color = random.choice(color_choices)

        cars_on_map.append(
            {
                "id": map_id,
                "carId": car_id,
                "x": x,
                "y": y,
                "yaw": yaw,
                "color": color,
                "status": "normal",
            }
        )

        cars_status.append(
            {
                "id": car_id,
                "color": color,
                "speed": random.uniform(0, 10),
                "battery": random.randint(50, 100),
                "fromLabel": "-",
                "toLabel": "-",
                "cameraId": None,
                "routeChanged": False,
            }
        )

    snapshot = {
        "type": "snapshot",
        "payload": {
            "carsOnMap": cars_on_map,
            "carsStatus": cars_status,
            "camerasOnMap": [],
            "camerasStatus": [],
            "incident": None,
            "routeChanges": [],
        },
    }
    return snapshot


async def producer():
    """
    0.5초마다 연결된 모든 클라이언트에 snapshot 전송
    """
    while True:
        if clients:
            msg = make_snapshot()
            data = json.dumps(msg, ensure_ascii=False)
            dead = []
            for ws in list(clients):
                try:
                    await ws.send(data)
                except Exception as exc:
                    print("[MockWS] send error:", exc)
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)
        await asyncio.sleep(0.5)


async def main():
    async with websockets.serve(handler, "0.0.0.0", 18000):
        print("[MockWS] listening on ws://0.0.0.0:18000/ (no path check)")
        await producer()


if __name__ == "__main__":
    asyncio.run(main())
