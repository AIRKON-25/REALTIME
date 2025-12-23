// types.ts
export type CameraId = string;
export type CarId = string;
export type IncidentId = string;

export type CarColor = "red" | "green" | "blue" | "yellow" | "purple";

export type ViewMode =
  | "default"          // 기본: 맵 + CarStatus + Incident(옵션)
  | "carFocused"       // 차량 눌렀을 때
  | "cameraFocused"    // 카메라 눌렀을 때
  | "incidentFocused"; // Incident 눌렀을 때

export interface MapObjectBase {
  id: string;
  x: number;   // 0 ~ 1
  y: number;   // 0 ~ 1
}

export interface CarOnMap extends MapObjectBase {
  carId: CarId;
  yaw: number; // degree
  color: CarColor;
  // 예: "default" | "routeChanged" | "alert" 등 확장 가능
  status: "normal" | "routeChanged";
}

export interface CameraOnMap extends MapObjectBase {
  cameraId: CameraId;
}

export type ObstacleClass = "cone" | "block" | "pedestrian";

export interface ObstacleOnMap extends MapObjectBase {
  obstacleId: string;
  kind: ObstacleClass;
}

export interface CarStatus {
  id: CarId;
  color: CarColor;
  class?: number; // 1 => obstacle (rubber cone)
  speed: number; // m/s
  battery: number; // 0 ~ 100
  fromLabel: string; // 출발지
  toLabel: string;   // 목적지
  cameraId?: CameraId; // 현재 이 차를 보고 있는 카메라
  routeChanged?: boolean;
}

export interface CameraStatus {
  id: CameraId;
  name: string; // e.g. "camera 3"
  // 실제 스트림 URL (팀에서 나중에 교체)
  streamUrl: string;
}

export interface Incident {
  id: IncidentId;
  title: string;       // e.g. "[Obstacle]"
  description: string; // e.g. "Traffic slowdown in Section 1..."
  obstacle?: ObstacleOnMap;
  cameraId?: CameraId; // 이 Incident를 비추는 카메라
  relatedCarIds?: CarId[];
}

export interface RouteChangeStep {
  carId: CarId;
  from: { x: number; y: number };
  to: { x: number; y: number };
}

export interface ServerSnapshot { // wkddoanfdpeogksdjWJrn
  carsOnMap: CarOnMap[];
  carsStatus: CarStatus[];
  camerasOnMap: CameraOnMap[];
  camerasStatus: CameraStatus[];
  incident: Incident | null;
  routeChanges: RouteChangeStep[];
}

// 서버에서 보내주는 메시지 타입
export type ServerMessage =
  | { type: "snapshot"; payload: ServerSnapshot }
  | { type: "partial"; payload: Partial<ServerSnapshot> };
  // partial은 선택사항: 일부만 업데이트하고 싶을 때 사용
