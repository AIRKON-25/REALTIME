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

export type ObstacleClass = "rubberCone" | "barricade";

export interface ObstacleOnMap extends MapObjectBase {
  obstacleId: string;
  kind: ObstacleClass;
}

export interface ObstacleStatus {
  id: string;
  class: number;
  cameraId?: CameraId;
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

export interface RoutePoint {
  x: number;
  y: number;
}

export interface RouteChangeStep {
  carId: CarId;
  from: RoutePoint;
  to: RoutePoint;
}

export interface CarRouteChange {
  carId: CarId;
  newRoute: RoutePoint[];
}

export interface CamStatusPacket {
  camerasOnMap: CameraOnMap[];
  camerasStatus: CameraStatus[];
}

export interface CarStatusSnapshot {
  mode?: "snapshot";
  carsOnMap: CarOnMap[];
  carsStatus: CarStatus[];
}

export interface CarStatusDelta {
  mode: "delta";
  carsOnMapUpserts?: CarOnMap[];
  carsOnMapDeletes?: CarId[];
  carsStatusUpserts?: CarStatus[];
  carsStatusDeletes?: CarId[];
}

export type CarStatusPacket = CarStatusSnapshot | CarStatusDelta;

export interface ObstacleStatusSnapshot {
  mode?: "snapshot";
  obstaclesOnMap: ObstacleOnMap[];
  obstaclesStatus?: ObstacleStatus[];
  incident?: Incident | null;
}

export interface ObstacleStatusDelta {
  mode: "delta";
  upserts?: ObstacleOnMap[];
  deletes?: string[];
  statusUpserts?: ObstacleStatus[];
  statusDeletes?: string[];
  incident?: Incident | null;
}

export type ObstacleStatusPacket = ObstacleStatusSnapshot | ObstacleStatusDelta;

export interface RouteChangePacket {
  incidentId?: IncidentId;
  obstacleId?: string;
  changes?: CarRouteChange[];
  steps?: RouteChangeStep[];
}

export interface MonitorState {
  carsOnMap: CarOnMap[];
  carsStatus: CarStatus[];
  camerasOnMap: CameraOnMap[];
  camerasStatus: CameraStatus[];
  obstaclesOnMap: ObstacleOnMap[];
  obstaclesStatus: ObstacleStatus[];
  incident: Incident | null;
  routeChanges: CarRouteChange[];
}

// 서버에서 보내주는 메시지 타입
export type RealtimeMessage =
  | { type: "camStatus"; ts: number; data: CamStatusPacket }
  | { type: "carStatus"; ts: number; data: CarStatusPacket }
  | { type: "obstacleStatus"; ts: number; data: ObstacleStatusPacket }
  | { type: "carRouteChange"; ts: number; data: RouteChangePacket };
