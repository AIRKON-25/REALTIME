// types.ts
export type CameraId = string;
export type CarId = string;
export type TrafficLightId = number;

export type CarColor = "red" | "green" | "yellow" | "purple" | "white";
export type TrafficLightSignal = "red" | "yellow" | "green";

export type ViewMode =
  | "default"          // base: map + car status list
  | "carFocused"       // a car is selected
  | "cameraFocused"    // camera is selected
  | "incidentFocused"; // incident (obstacle) is selected

export interface MapObjectBase {
  id: string;
  x: number;   // 0 ~ 1
  y: number;   // 0 ~ 1
}

export interface CarOnMap extends MapObjectBase {
  carId: CarId;
  yaw: number; // degree
  color: CarColor;
  status: "normal" | "routeChanged";
}

export interface CameraOnMap extends MapObjectBase {
  cameraId: CameraId;
}

export interface TrafficLightOnMap extends MapObjectBase {
  trafficLightId: TrafficLightId;
  yaw: number; // degree
}

export type ObstacleClass = "rubberCone" | "barricade";

export interface ObstacleOnMap extends MapObjectBase {
  obstacleId: string;
  kind: ObstacleClass;
}

export interface ObstacleStatus {
  id: string;
  class: number;
  cameraIds?: CameraId[];
}

export interface TrafficLightStatus {
  trafficLightId: TrafficLightId;
  light: TrafficLightSignal;
  left_green?: boolean;
}

export interface CarStatus {
  car_id: CarId;
  color: CarColor;
  class?: number; // 1 => obstacle (rubber cone)
  speed: number; // m/s
  battery: number; // 0 ~ 100
  path_future: RoutePoint[]; // future route
  category: string; // normal | battery 굳이
  resolution: string; // path의 해상도 각 점의 간격
  cameraIds?: CameraId[]; // cameras currently seeing this car
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

export interface TrafficLightStatusSnapshot {
  mode?: "snapshot";
  trafficLightsOnMap: TrafficLightOnMap[];
  trafficLightsStatus: TrafficLightStatus[];
}

export interface TrafficLightStatusDelta {
  mode: "delta";
  trafficLightsOnMapUpserts?: TrafficLightOnMap[];
  trafficLightsOnMapDeletes?: TrafficLightId[];
  trafficLightsStatusUpserts?: TrafficLightStatus[];
  trafficLightsStatusDeletes?: TrafficLightId[];
}

export type TrafficLightStatusPacket = TrafficLightStatusSnapshot | TrafficLightStatusDelta;

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

export type ClusterStage = "preCluster" | "postCluster";

export interface ClusterStageSnapshot {
  stage: ClusterStage;
  mode?: "snapshot";
  carsOnMap: CarOnMap[];
  carsStatus: CarStatus[];
  obstaclesOnMap: ObstacleOnMap[];
  obstaclesStatus: ObstacleStatus[];
}

export type ClusterStagePacket = ClusterStageSnapshot;

export interface MonitorState {
  carsOnMap: CarOnMap[];
  carsStatus: CarStatus[];
  camerasOnMap: CameraOnMap[];
  camerasStatus: CameraStatus[];
  obstaclesOnMap: ObstacleOnMap[];
  obstaclesStatus: ObstacleStatus[];
  routeChanges: CarRouteChange[];
  trafficLightsOnMap: TrafficLightOnMap[];
  trafficLightsStatus: TrafficLightStatus[];
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

export type RealtimeMessage =
  | { type: "camStatus"; ts: number; data: CamStatusPacket }
  | { type: "carStatus"; ts: number; data: CarStatusPacket }
  | { type: "trafficLightStatus"; ts: number; data: TrafficLightStatusPacket }
  | { type: "obstacleStatus"; ts: number; data: ObstacleStatusPacket }
  | { type: "carRouteChange"; ts: number; data: RouteChangePacket }
  | { type: "clusterStage"; ts: number; data: ClusterStagePacket }
  | AdminResponseMessage;
