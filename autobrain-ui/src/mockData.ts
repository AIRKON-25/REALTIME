// src/mockData.ts
import type {
  CameraOnMap,
  CameraStatus,
  CarOnMap,
  CarStatus,
  Incident,
  ObstacleOnMap,
  RouteChangeStep,
  ServerSnapshot,
} from "./types";

const mockCamerasOnMap: CameraOnMap[] = [
  { id: "c-top", x: 0.5, y: 0.03, cameraId: "cam-1" },
  { id: "c-right", x: 0.97, y: 0.5, cameraId: "cam-2" },
  { id: "c-bottom", x: 0.5, y: 0.97, cameraId: "cam-3" },
  { id: "c-left", x: 0.03, y: 0.5, cameraId: "cam-4" },
];

const mockCamerasStatus: CameraStatus[] = [
  { id: "cam-1", name: "camera 1", streamUrl: "/assets/camera1.png" },
  { id: "cam-2", name: "camera 2", streamUrl: "/assets/camera2.png" },
  { id: "cam-3", name: "camera 3", streamUrl: "/assets/camera3.png" },
  { id: "cam-4", name: "camera 4", streamUrl: "/assets/camera4.png" },
];

const mockCarsOnMap: CarOnMap[] = [
  {
    id: "mcar-1",
    carId: "car-1",
    x: 0.1,
    y: 0.8,
    yaw: 0,
    color: "red",
    status: "normal",
  },
  {
    id: "mcar-2",
    carId: "car-2",
    x: 0.2,
    y: 0.3,
    yaw: 90,
    color: "blue",
    status: "routeChanged",
  },
  {
    id: "mcar-3",
    carId: "car-3",
    x: 0.75,
    y: 0.25,
    yaw: 180,
    color: "green",
    status: "routeChanged",
  },
  {
    id: "mcar-4",
    carId: "car-4",
    x: 0.8,
    y: 0.8,
    yaw: 270,
    color: "yellow",
    status: "normal",
  },
  {
    id: "mcar-5",
    carId: "car-5",
    x: 0.4,
    y: 0.15,
    yaw: 90,
    color: "blue",
    status: "normal",
  },
];

const mockCarsStatus: CarStatus[] = [
  {
    id: "car-1",
    color: "red",
    speed: 3,
    battery: 90,
    fromLabel: "Origin",
    toLabel: "Destination",
    cameraId: "cam-3",
  },
  {
    id: "car-2",
    color: "purple",
    speed: 3,
    battery: 90,
    fromLabel: "Origin",
    toLabel: "Destination",
    cameraId: "cam-1",
    routeChanged: true,
  },
  {
    id: "car-3",
    color: "green",
    speed: 3,
    battery: 90,
    fromLabel: "Origin",
    toLabel: "Destination",
    cameraId: "cam-2",
    routeChanged: true,
  },
  {
    id: "car-4",
    color: "yellow",
    speed: 3,
    battery: 90,
    fromLabel: "Origin",
    toLabel: "Destination",
  },
  {
    id: "car-5",
    color: "blue",
    speed: 3,
    battery: 90,
    fromLabel: "Origin",
    toLabel: "Destination",
  },
];

const mockObstacle: ObstacleOnMap = {
  id: "ob-1",
  obstacleId: "ob-1",
  x: 0.5,
  y: 0.1,
  kind: "cone",
};

const mockIncident: Incident = {
  id: "inc-1",
  title: "[Obstacle]",
  description: "Traffic slowdown in Section 1 due to a traffic cone",
  obstacle: mockObstacle,
  cameraId: "cam-3",
  relatedCarIds: ["car-2", "car-3"],
};

const mockRouteSteps: RouteChangeStep[] = [
  {
    carId: "car-2",
    from: { x: 0.55, y: 0.1 },
    to: { x: 0.4, y: 0.1 },
  },
  {
    carId: "car-3",
    from: { x: 0.55, y: 0.1 },
    to: { x: 0.7, y: 0.1 },
  },
];

// ✅ 실제로 App에서 쓸 하나짜리 스냅샷
export const mockSnapshot: ServerSnapshot = {
  carsOnMap: mockCarsOnMap,
  carsStatus: mockCarsStatus,
  camerasOnMap: mockCamerasOnMap,
  camerasStatus: mockCamerasStatus,
  incident: mockIncident,
  routeChanges: mockRouteSteps,
};
