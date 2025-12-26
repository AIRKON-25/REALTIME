import asyncio
import json
import math
import time

import websockets

HOST = "0.0.0.0"
PORT = 18000

CAMERAS_ON_MAP = [
    {"id": "camMarker-1", "cameraId": "cam-1", "x": -0.01, "y": 0.5},
    {"id": "camMarker-3", "cameraId": "cam-2", "x": 1.01, "y": 0.5},
]

CAMERAS_STATUS = [
    {"id": "cam-1", "name": "Camera 1", "streamUrl": "/assets/map-track.png"},
    {"id": "cam-2", "name": "Camera 2", "streamUrl": "/assets/map-track.png"},
]

CAR_DEFS = [
    {"car_id": "car-1", "color": "red", "phase": 0.0},
    {"car_id": "car-2", "color": "blue", "phase": 2.1},
    {"car_id": "car-3", "color": "green", "phase": 4.2},
]

OBSTACLE_ID = "ob-1"


def build_car_message(ts: float) -> dict:
    cars_on_map = []
    cars_status = []
    for idx, car in enumerate(CAR_DEFS):
        angle = ts * 0.4 + car["phase"]
        x = 0.5 + 0.28 * math.cos(angle)
        y = 0.5 + 0.28 * math.sin(angle)
        yaw = (math.degrees(angle) + 360.0) % 360.0
        cars_on_map.append(
            {
                "id": f"m{car['car_id']}",
                "carId": car["car_id"],
                "x": x,
                "y": y,
                "yaw": yaw,
                "color": car["color"],
                "status": "normal",
            }
        )
        cars_status.append(
            {
                "id": car["car_id"],
                "color": car["color"],
                "class": 0,
                "speed": round(2.5 + idx * 0.4, 2),
                "battery": 95 - idx * 5,
                "fromLabel": "Origin",
                "toLabel": "Destination",
                "cameraId": "cam-1" if idx == 0 else None,
                "routeChanged": False,
            }
        )
    return {
        "type": "carStatus",
        "ts": ts,
        "data": {
            "mode": "snapshot",
            "carsOnMap": cars_on_map,
            "carsStatus": cars_status,
        },
    }


def build_obstacle_upsert(ts: float) -> dict:
    kind = "rubberCone" if int(ts) % 2 == 0 else "barricade"
    cls = 1 if kind == "rubberCone" else 2
    obstacle = {
        "id": OBSTACLE_ID,
        "obstacleId": OBSTACLE_ID,
        "x": 0.62,
        "y": 0.25,
        "kind": kind,
    }
    status = {
        "id": OBSTACLE_ID,
        "class": cls,
        "cameraId": "cam-2",
    }
    return {
        "type": "obstacleStatus",
        "ts": ts,
        "data": {
            "mode": "delta",
            "upserts": [obstacle],
            "statusUpserts": [status],
        },
    }


def build_obstacle_clear(ts: float) -> dict:
    return {
        "type": "obstacleStatus",
        "ts": ts,
        "data": {
            "mode": "delta",
            "deletes": [OBSTACLE_ID],
            "statusDeletes": [OBSTACLE_ID],
        },
    }


def build_route_change(ts: float) -> dict:
    return {
        "type": "carRouteChange",
        "ts": ts,
        "data": {
            "changes": [
                {
                    "carId": "car-1",
                    "newRoute": [
                        {"x": 0.58, "y": 0.62},
                        {"x": 0.62, "y": 0.5},
                        {"x": 0.68, "y": 0.38},
                    ],
                },
                {
                    "carId": "car-2",
                    "newRoute": [
                        {"x": 0.35, "y": 0.52},
                        {"x": 0.3, "y": 0.44},
                        {"x": 0.26, "y": 0.38},
                    ],
                },
            ],
        },
    }


async def handler(websocket):
    print("[ws_simulator] client connected")
    cam_status = {
        "type": "camStatus",
        "ts": time.time(),
        "data": {"camerasOnMap": CAMERAS_ON_MAP, "camerasStatus": CAMERAS_STATUS},
    }
    await websocket.send(json.dumps(cam_status))

    last_obstacle_active = None
    route_sent = False

    try:
        while True:
            ts = time.time()
            await websocket.send(json.dumps(build_car_message(ts)))

            obstacle_active = (int(ts) // 8) % 2 == 0
            if obstacle_active != last_obstacle_active:
                if obstacle_active:
                    await websocket.send(json.dumps(build_obstacle_upsert(ts)))
                    route_sent = False
                else:
                    await websocket.send(json.dumps(build_obstacle_clear(ts)))
                last_obstacle_active = obstacle_active

            if obstacle_active and not route_sent:
                await websocket.send(json.dumps(build_route_change(ts)))
                route_sent = True

            await asyncio.sleep(0.1)
    except Exception as exc:
        print(f"[ws_simulator] client closed: {exc}")


async def main():
    async with websockets.serve(handler, HOST, PORT):
        print(f"[ws_simulator] ws://{HOST}:{PORT}/monitor")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
