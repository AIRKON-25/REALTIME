// types.ts
export type CameraId = string;
export type CarId = string;

export type CarColor = "red" | "green" | "blue" | "yellow" | "purple" | "white";

export type ViewMode =
  | "default"          // 기본: 맵 + CarStatus
  | "carFocused"       // 차량을 선택했을 때
  | "cameraFocused"    // 카메라를 선택했을 때
  | "incidentFocused"; // 장애물 알림을 볼 때

export interface MapObjectBase {
  id: string;
  x: number;   // 0 ~ 1
  y: number;   // 0 ~ 1
}

export interface CarOnMap extends MapObjectBase {
  carId: CarId;
  yaw: number; // degree
  color: CarColor;
  // e.g. "default" | "routeChanged" | "alert" 로 확장 여지
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
  cameraId?: CameraId; // 이 차량을 보고 있는 카메라
  routeChanged?: boolean;
}

export interface CameraStatus {
  id: CameraId;
  name: string; // e.g. "camera 3"
  streamUrl: string;
  streamBEVUrl: string;
}

export interface RoutePoint {
  x: number;
  y: number;
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
}

export interface ObstacleStatusDelta {
  mode: "delta";
  upserts?: ObstacleOnMap[];
  deletes?: string[];
  statusUpserts?: ObstacleStatus[];
  statusDeletes?: string[];
}

export type ObstacleStatusPacket = ObstacleStatusSnapshot | ObstacleStatusDelta;

export interface RouteChangePacket {
  changes?: CarRouteChange[];
}

export interface MonitorState {
  carsOnMap: CarOnMap[];
  carsStatus: CarStatus[];
  camerasOnMap: CameraOnMap[];
  camerasStatus: CameraStatus[];
  obstaclesOnMap: ObstacleOnMap[];
  obstaclesStatus: ObstacleStatus[];
  routeChanges: CarRouteChange[];
}

export interface AdminResponseMessage {
  type: "adminResponse";
  requestId?: number | string;
  cmd?: string;
  response?: Record<string, unknown>;
  status?: "ok" | "error";
  message?: string;
  ts?: number;
}

// 서버에서 오는 실시간 메시지 형태
export type RealtimeMessage =
  | { type: "camStatus"; ts: number; data: CamStatusPacket }
  | { type: "carStatus"; ts: number; data: CarStatusPacket }
  | { type: "obstacleStatus"; ts: number; data: ObstacleStatusPacket }
  | { type: "carRouteChange"; ts: number; data: RouteChangePacket }
  | AdminResponseMessage;
