// App.tsx
import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import type {
  CameraId,
  CarId,
  CarColor,
  IncidentId,
  ViewMode,
  MonitorState,
  RealtimeMessage,
  CameraStatus,
  CarStatusPacket,
  ObstacleStatusPacket,
  RouteChangePacket,
  CarRouteChange,
} from "./types";

import { Header } from "./components/Header";
import { Layout } from "./components/Layout";
import { MapView } from "./components/MapView";
import { CarStatusPanel } from "./components/CarStatusPanel";
import { IncidentPanel } from "./components/IncidentPanel";
import { MonitoringPanel } from "./components/MonitoringPanel";

// 접속한 호스트/포트를 따라가도록 설정
const WS_HOST = import.meta.env.VITE_WS_HOST || window.location.hostname;
const WS_PORT = import.meta.env.VITE_WS_PORT || "18000";
const WS_URL = `ws://${WS_HOST}:${WS_PORT}/monitor`;
const ROUTE_FLASH_MS = 600;
const ROUTE_FLASH_GAP_MS = 250;

const emptyState: MonitorState = {
  carsOnMap: [],
  carsStatus: [],
  camerasOnMap: [],
  camerasStatus: [],
  obstaclesOnMap: [],
  obstaclesStatus: [],
  incident: null,
  routeChanges: [],
};

const mergeByKey = <T,>(
  prev: T[],
  upserts: T[] | undefined,
  deletes: string[] | undefined,
  keyFn: (item: T) => string
): T[] => {
  if ((!upserts || upserts.length === 0) && (!deletes || deletes.length === 0)) {
    return prev;
  }
  const byKey = new Map(prev.map((item) => [keyFn(item), item]));
  if (deletes) {
    deletes.forEach((id) => byKey.delete(id));
  }
  if (upserts) {
    upserts.forEach((item) => byKey.set(keyFn(item), item));
  }
  return Array.from(byKey.values());
};

const applyCarStatus = (
  prev: MonitorState,
  data: CarStatusPacket
): MonitorState => {
  if (data.mode === "delta") {
    const carsOnMap = mergeByKey(
      prev.carsOnMap,
      data.carsOnMapUpserts,
      data.carsOnMapDeletes,
      (item) => item.carId
    );
    const carsStatus = mergeByKey(
      prev.carsStatus,
      data.carsStatusUpserts,
      data.carsStatusDeletes,
      (item) => item.id
    );
    return { ...prev, carsOnMap, carsStatus };
  }
  return { ...prev, carsOnMap: data.carsOnMap, carsStatus: data.carsStatus };
};

const applyObstacleStatus = (
  prev: MonitorState,
  data: ObstacleStatusPacket
): MonitorState => {
  let next = prev;
  if (data.mode === "delta") {
    const obstaclesOnMap = mergeByKey(
      prev.obstaclesOnMap,
      data.upserts,
      data.deletes,
      (item) => item.obstacleId
    );
    const obstaclesStatus = mergeByKey(
      prev.obstaclesStatus,
      data.statusUpserts,
      data.statusDeletes,
      (item) => item.id
    );
    next = { ...prev, obstaclesOnMap, obstaclesStatus };
  } else {
    next = {
      ...prev,
      obstaclesOnMap: data.obstaclesOnMap,
      obstaclesStatus: data.obstaclesStatus ?? prev.obstaclesStatus,
    };
  }
  if (Object.prototype.hasOwnProperty.call(data, "incident")) {
    const incident = data.incident ?? null;
    return { ...next, incident };
  }
  return next;
};

const deriveRouteChanges = (
  steps: RouteChangePacket["steps"]
): CarRouteChange[] => {
  if (!steps || steps.length === 0) {
    return [];
  }
  return steps.map((step) => ({
    carId: step.carId,
    oldRoute: [],
    newRoute: [step.from, step.to],
  }));
};

const applyRouteChange = (
  prev: MonitorState,
  data: RouteChangePacket
): MonitorState => {
  if (data.changes && data.changes.length > 0) {
    return { ...prev, routeChanges: data.changes };
  }
  return { ...prev, routeChanges: deriveRouteChanges(data.steps) };
};

function App() {
  // 🔹 서버에서 오는 전체 상태
  const [serverState, setServerState] = useState<MonitorState>(emptyState);
  const [hasCamStatus, setHasCamStatus] = useState<boolean>(false);

  // 🔹 뷰 모드 관련 상태 (사용자 인터랙션용)
  const [viewMode, setViewMode] = useState<ViewMode>("default");
  const [selectedCarId, setSelectedCarId] = useState<CarId | null>(null);
  const [selectedCameraId, setSelectedCameraId] = useState<CameraId | null>(
    null
  );
  const [activeIncidentId, setActiveIncidentId] = useState<IncidentId | null>(
    null
  );
  const [routeFlashPhase, setRouteFlashPhase] = useState<
    "none" | "old" | "new"
  >("none");
  const routeFlashTimersRef = useRef<number[]>([]);

  // ===========================
  //  1) WebSocket 연결 목업쓸떈 제외
  // ===========================
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: number | null = null;

    const connect = () => {
      ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log("WebSocket connected");
      };

      ws.onmessage = (event) => {
        try {
          const msg: RealtimeMessage = JSON.parse(event.data);
          if (msg.type === "camStatus") {
            setHasCamStatus(true);
          }
          setServerState((prev) => {
            switch (msg.type) {
              case "camStatus":
                return {
                  ...prev,
                  camerasOnMap: msg.data.camerasOnMap,
                  camerasStatus: msg.data.camerasStatus,
                };
              case "carStatus":
                return applyCarStatus(prev, msg.data);
              case "obstacleStatus":
                return applyObstacleStatus(prev, msg.data);
              case "carRouteChange":
                return applyRouteChange(prev, msg.data);
              default:
                return prev;
            }
          });
        } catch (e) {
          console.error("Invalid message from server:", e);
        }
      };

      ws.onclose = () => {
        console.warn("WebSocket closed, retry in 2s");
        if (!reconnectTimer) {
          reconnectTimer = window.setTimeout(() => {
            reconnectTimer = null;
            connect();
          }, 2000);
        }
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        ws?.close();
      };
    };

    connect();

    return () => {
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, []);

  // 서버에서 아직 아무것도 안 보냈을 때의 기본 값들
  const carsOnMap = serverState.carsOnMap;
  const carsStatus = serverState.carsStatus;
  const camerasOnMap = serverState.camerasOnMap;
  const camerasStatus = serverState.camerasStatus;
  const obstaclesOnMap = serverState.obstaclesOnMap;
  const incident = serverState.incident;
  const routeChanges = serverState.routeChanges;

  const carColorById = useMemo(() => {
    const allowed: readonly CarColor[] = ["red", "green", "blue", "yellow", "purple"];
    const normalize = (color: string | undefined): CarColor | undefined => {
      const normalized = (color ?? "").toString().trim().toLowerCase();
      return allowed.includes(normalized as CarColor)
        ? (normalized as CarColor)
        : undefined;
    };
    const map: Record<CarId, CarColor> = {};
    carsOnMap.forEach((car) => {
      const n = normalize(car.color);
      if (n) map[car.carId] = n;
    });
    carsStatus.forEach((status) => {
      const n = normalize(status.color);
      if (n) map[status.id] = n;
    });
    return map;
  }, [carsOnMap, carsStatus]);

  const isIncidentActive =
    !!incident && activeIncidentId === incident.id;

  useEffect(() => {
    routeFlashTimersRef.current.forEach((timer) => window.clearTimeout(timer));
    routeFlashTimersRef.current = [];
    if (routeChanges.length === 0) {
      setRouteFlashPhase("none");
      return;
    }
    setRouteFlashPhase("old");
    const hideOld = window.setTimeout(
      () => setRouteFlashPhase("none"),
      ROUTE_FLASH_MS
    );
    const showNew = window.setTimeout(
      () => setRouteFlashPhase("new"),
      ROUTE_FLASH_MS + ROUTE_FLASH_GAP_MS
    );
    const hideNew = window.setTimeout(
      () => setRouteFlashPhase("none"),
      ROUTE_FLASH_MS + ROUTE_FLASH_GAP_MS + ROUTE_FLASH_MS
    );
    routeFlashTimersRef.current = [hideOld, showNew, hideNew];
    return () => {
      routeFlashTimersRef.current.forEach((timer) => window.clearTimeout(timer));
      routeFlashTimersRef.current = [];
    };
  }, [routeChanges]);

  // ===========================
  //  2) 뷰 모드 계산
  // ===========================
  useEffect(() => {
    if (isIncidentActive && selectedCarId) {
      setViewMode("incidentFocused");
    } else if (isIncidentActive) {
      setViewMode("incidentFocused");
    } else if (selectedCarId) {
      setViewMode("carFocused");
    } else if (selectedCameraId) {
      setViewMode("cameraFocused");
    } else {
      setViewMode("default");
    }
  }, [isIncidentActive, selectedCarId, selectedCameraId]);

  // ===========================
  //  3) 클릭 핸들러들
  // ===========================
  const handleCarClick = (carId: CarId) => {
    setSelectedCarId(carId);
    const carStatus = carsStatus.find((c) => c.id === carId);
    if (carStatus?.cameraId) {
      setSelectedCameraId(carStatus.cameraId);
    }
  };

  const handleCameraClick = (cameraId: CameraId) => {
    setSelectedCameraId(cameraId);
    setSelectedCarId(null);
  };

  const handleIncidentClick = () => {
    if (!incident) return;
    if (isIncidentActive) {
      setActiveIncidentId(null);
    } else {
      setActiveIncidentId(incident.id);
      if (incident.cameraId) setSelectedCameraId(incident.cameraId);
    }
  };

  const handleBackToDefault = () => {
    setSelectedCarId(null);
    setSelectedCameraId(null);
    setActiveIncidentId(null);
    setViewMode("default");
  };

  // Incident가 비추는 영역에 있는 차량들
  const carsStatusForPanel = useMemo(() => carsStatus, [carsStatus]);

  const vehiclesInIncidentView = useMemo(() => {
    if (!incident?.relatedCarIds) return [];
    return carsStatusForPanel.filter((car) =>
      incident.relatedCarIds!.includes(car.id)
    );
  }, [incident, carsStatusForPanel]);

  const visibleRouteChanges = useMemo(() => {
    if (routeFlashPhase === "old") {
      return routeChanges.map((change) => ({ ...change, newRoute: [] }));
    }
    if (routeFlashPhase === "new") {
      return routeChanges.map((change) => ({ ...change, oldRoute: [] }));
    }
    return [];
  }, [routeChanges, routeFlashPhase]);

// Monitoring에 실제로 띄울 카메라 선택 로직
  const monitoringCameraId: CameraId | null = useMemo(() => {
    if (selectedCameraId) return selectedCameraId;
    if (incident?.cameraId && isIncidentActive) return incident.cameraId;
    const car = carsStatus.find((c) => c.id === selectedCarId);
    return car?.cameraId ?? null;
  }, [selectedCameraId, incident, isIncidentActive, selectedCarId, carsStatus]);

  // const monitoringCamera =
  //   monitoringCameraId &&
  //   camerasStatus.find((cam) => cam.id === monitoringCameraId);
  const monitoringCamera: CameraStatus | null = useMemo(() => {
    if (!monitoringCameraId) return null;
    const found = camerasStatus.find((cam) => cam.id === monitoringCameraId);
    return found ?? null;
  }, [monitoringCameraId, camerasStatus]);


  const isLoading = !hasCamStatus;
  const showBackButton =
    viewMode !== "default" || !!selectedCarId || !!selectedCameraId || !!incident;

  return (
    <div className="app-root">
      <Header />
      <Layout
        viewMode={viewMode}
        hasIncident={!!incident}
        showBackButton={showBackButton}
        onBackClick={handleBackToDefault}
      >
        {/* LEFT: MAP */}
        <div
          className={`layout__map-inner ${
            showBackButton ? "layout__map-inner--has-back" : ""
          }`}
        >
          {isLoading && (
            <div className="map__loading">Waiting for server data...</div>
          )}

          <MapView
            mapImage="/assets/map-track.png"
            carsOnMap={carsOnMap}
            camerasOnMap={camerasOnMap}
            obstacles={obstaclesOnMap}
            activeCameraId={monitoringCameraId}
            activeCarId={selectedCarId}
            routeChanges={visibleRouteChanges}
            onCarClick={handleCarClick}
            onCameraClick={handleCameraClick}
          />
        </div>

        {/* RIGHT: PANELS */}
        <div className="right-panels">
          {/* Car Status 영역 */}
          {viewMode === "default" && (
            <CarStatusPanel
              cars={carsStatusForPanel}
              carColorById={carColorById}
              selectedCarId={selectedCarId}
              onCarClick={handleCarClick}
              scrollable
            />
          )}

          {viewMode === "carFocused" && selectedCarId && (
            <CarStatusPanel
              cars={carsStatusForPanel}
              carColorById={carColorById}
              selectedCarId={selectedCarId}
              onCarClick={handleCarClick}
              detailOnly
            />
          )}

          {viewMode === "incidentFocused" && selectedCarId && (
            <CarStatusPanel
              cars={vehiclesInIncidentView}
              selectedCarId={selectedCarId}
              onCarClick={handleCarClick}
              detailOnly
            />
          )}

          {viewMode === "incidentFocused" &&
            !selectedCarId &&
            vehiclesInIncidentView.length > 0 && (
              <CarStatusPanel
                cars={vehiclesInIncidentView}
                carColorById={carColorById}
                selectedCarId={null}
                onCarClick={handleCarClick}
                scrollable
              />
            )}

          {/* Incident */}
          {(viewMode === "default" || viewMode === "incidentFocused") && (
            <IncidentPanel
              incident={incident}
              isActive={isIncidentActive}
              onClick={handleIncidentClick}
            />
          )}

          {/* Monitoring View */}
          {(viewMode === "cameraFocused" ||
            viewMode === "carFocused" ||
            viewMode === "incidentFocused") && (
            <MonitoringPanel camera={monitoringCamera ?? null} />
          )}
        </div>
      </Layout>
    </div>
  );
}

export default App;
