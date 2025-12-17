// App.tsx
import { useEffect, useMemo, useState } from "react";
import "./App.css";
import type {
  CameraId,
  CarId,
  CarColor,
  IncidentId,
  RouteChangeStep,
  ViewMode,
  ServerSnapshot,
  ServerMessage,
  CameraStatus, 
} from "./types";

import { Header } from "./components/Header";
import { Layout } from "./components/Layout";
import { MapView } from "./components/MapView";
import { CarStatusPanel } from "./components/CarStatusPanel";
import { IncidentPanel } from "./components/IncidentPanel";
import { MonitoringPanel } from "./components/MonitoringPanel";
import { mockSnapshot } from "./mockData"; // 더미데이터

const WS_URL = "ws://localhost:18000/monitor"; // 서버에서 여기에 열어주면 됨
const USE_MOCK = false; // true면 mockSnapshot만 사용

function App() {
  // 🔹 서버에서 오는 전체 상태
  const [serverState, setServerState] = useState<ServerSnapshot | null>((
  USE_MOCK ? mockSnapshot : null));

  // 🔹 뷰 모드 관련 상태 (사용자 인터랙션용)
  const [viewMode, setViewMode] = useState<ViewMode>("default");
  const [selectedCarId, setSelectedCarId] = useState<CarId | null>(null);
  const [selectedCameraId, setSelectedCameraId] = useState<CameraId | null>(
    null
  );
  const [activeIncidentId, setActiveIncidentId] = useState<IncidentId | null>(
    null
  );

  // 🔹 경로 변경 애니메이션용
  const [activeRouteIdx, setActiveRouteIdx] = useState<number | null>(null);

  // ===========================
  //  1) WebSocket 연결 목업쓸떈 제외
  // ===========================
  useEffect(() => {
    if (USE_MOCK) { //더미 모드에서는 WebSocket 연결 안 함
    return;
  }
    let ws: WebSocket | null = null;
    let reconnectTimer: number | null = null;

    const connect = () => {
      ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log("WebSocket connected");
      };

      ws.onmessage = (event) => {
        try {
          const msg: ServerMessage = JSON.parse(event.data);

          setServerState((prev) => {
            if (msg.type === "snapshot") {
              return msg.payload;
            }
            if (msg.type === "partial") {
              return prev ? { ...prev, ...msg.payload } : (msg.payload as ServerSnapshot);
            }
            return prev;
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
  const carsOnMap = serverState?.carsOnMap ?? [];
  const carsStatus = serverState?.carsStatus ?? [];
  const camerasOnMap = serverState?.camerasOnMap ?? [];
  const camerasStatus = serverState?.camerasStatus ?? [];
  const incident = serverState?.incident ?? null;
  const routeSequence: RouteChangeStep[] = serverState?.routeChanges ?? [];

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
  //  3) 경로 변경 애니메이션
  // ===========================
  useEffect(() => {
    if (!isIncidentActive || routeSequence.length === 0) {
      setActiveRouteIdx(null);
      return;
    }

    let idx = 0;
    setActiveRouteIdx(idx);

    const interval = window.setInterval(() => {
      idx = (idx + 1) % routeSequence.length;
      setActiveRouteIdx(idx);
    }, 1000);

    return () => window.clearInterval(interval);
  }, [isIncidentActive, routeSequence]);

  const activeRouteStep: RouteChangeStep | null = useMemo(() => {
    if (activeRouteIdx == null) return null;
    return routeSequence[activeRouteIdx];
  }, [activeRouteIdx, routeSequence]);

  // ===========================
  //  4) 클릭 핸들러들
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
  const carsStatusForPanel = useMemo(
    () =>
      carsStatus.filter((car) => {
        const cls = car.class ?? (car as { cls?: number | string } | undefined)?.cls;
        return Number(cls) !== 1;
      }),
    [carsStatus]
  );

  const vehiclesInIncidentView = useMemo(() => {
    if (!incident?.relatedCarIds) return [];
    return carsStatusForPanel.filter((car) =>
      incident.relatedCarIds!.includes(car.id)
    );
  }, [incident, carsStatusForPanel]);

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


  const isLoading = !serverState;
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
            carsStatus={carsStatus}
            camerasOnMap={camerasOnMap}
            obstacles={
              isIncidentActive && incident?.obstacle ? [incident.obstacle] : []
            }
            activeCameraId={monitoringCameraId}
            activeCarId={selectedCarId}
            activeRouteStep={activeRouteStep}
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
