// App.tsx
import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import type {
  CameraId,
  CarId,
  CarColor,
  ViewMode,
  MonitorState,
  RealtimeMessage,
  CameraStatus,
  CarStatusPacket,
  ObstacleStatusPacket,
  RouteChangePacket,
  TrafficLightStatusPacket,
  CarRouteChange,
  RoutePoint,
  ClusterStage,
} from "./types";

import { Header } from "./components/Header";
import { AdminControlsPanel } from "./components/AdminControlsPanel";
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
type AppMode = "monitor" | "admin";
type PageRoute = "monitor" | "admin" | "clusterBefore" | "clusterAfter";

const resolveRoute = (pathname: string): PageRoute => {
  const normalized = pathname.toLowerCase();
  if (normalized.startsWith("/admin")) return "admin";
  if (normalized.startsWith("/cluster/before") || normalized.startsWith("/cluster/pre")) return "clusterBefore";
  if (normalized.startsWith("/cluster/after") || normalized.startsWith("/cluster/post")) return "clusterAfter";
  return "monitor";
};

const CLUSTER_BEFORE_STAGE: ClusterStage = "preCluster";
const CLUSTER_AFTER_STAGE: ClusterStage = "postCluster";

const emptyState: MonitorState = {
  carsOnMap: [],
  carsStatus: [],
  camerasOnMap: [],
  camerasStatus: [],
  obstaclesOnMap: [],
  obstaclesStatus: [],
  routeChanges: [],
  trafficLightsOnMap: [],
  trafficLightsStatus: [],
};

const mergeByKey = <T,>(
  prev: T[],
  upserts: T[] | undefined,
  deletes: (string | number)[] | undefined,
  keyFn: (item: T) => string
): T[] => {
  if ((!upserts || upserts.length === 0) && (!deletes || deletes.length === 0)) {
    return prev;
  }
  const byKey = new Map(prev.map((item) => [keyFn(item), item]));
  if (deletes) {
    deletes.forEach((id) => byKey.delete(id.toString()));
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
      (item) => item.car_id
    );
    return { ...prev, carsOnMap, carsStatus };
  }
  return { ...prev, carsOnMap: data.carsOnMap, carsStatus: data.carsStatus };
};

const applyObstacleStatus = (
  prev: MonitorState,
  data: ObstacleStatusPacket
): MonitorState => {
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
    return { ...prev, obstaclesOnMap, obstaclesStatus };
  }
  const obstaclesOnMap = data.obstaclesOnMap ?? [];
  const obstaclesStatus = data.obstaclesStatus ?? [];
  return { ...prev, obstaclesOnMap, obstaclesStatus };
};

const applyRouteChange = (
  prev: MonitorState,
  data: RouteChangePacket
): MonitorState => {
  const changes = data.changes ?? [];
  return { ...prev, routeChanges: changes };
};

const applyTrafficLightStatus = (
  prev: MonitorState,
  data: TrafficLightStatusPacket
): MonitorState => {
  if (data.mode === "delta") {
    const trafficLightsOnMap = mergeByKey(
      prev.trafficLightsOnMap,
      data.trafficLightsOnMapUpserts,
      data.trafficLightsOnMapDeletes,
      (item) => item.trafficLightId.toString()
    );
    const trafficLightsStatus = mergeByKey(
      prev.trafficLightsStatus,
      data.trafficLightsStatusUpserts,
      data.trafficLightsStatusDeletes,
      (item) => item.trafficLightId.toString()
    );
    return { ...prev, trafficLightsOnMap, trafficLightsStatus };
  }
  return {
    ...prev,
    trafficLightsOnMap: data.trafficLightsOnMap,
    trafficLightsStatus: data.trafficLightsStatus,
  };
};

function App() {
  // 🔹 서버에서 오는 전체 상태
  const initialRoute = resolveRoute(window.location.pathname);
  const [pageRoute] = useState<PageRoute>(initialRoute);
  const isAdminPage = pageRoute === "admin";
  const isClusterBefore = pageRoute === "clusterBefore";
  const isClusterAfter = pageRoute === "clusterAfter";
  const [serverState, setServerState] = useState<MonitorState>(emptyState);
  const [clusterBeforeState, setClusterBeforeState] = useState<MonitorState>(emptyState);
  const [clusterAfterState, setClusterAfterState] = useState<MonitorState>(emptyState);
  const [hasCamStatus, setHasCamStatus] = useState<boolean>(false);
  const [appMode, setAppMode] = useState<AppMode>(() =>
    isAdminPage ? "admin" : "monitor"
  );

  const handleModeChange = (mode: AppMode) => {
    if (!isAdminPage) return;
    setAppMode(mode);
  };

  // 🔹 뷰 모드 관련 상태 (사용자 인터랙션용)
  const [viewMode, setViewMode] = useState<ViewMode>("default");
  const [selectedCarId, setSelectedCarId] = useState<CarId | null>(null);
  const [selectedCameraIds, setSelectedCameraIds] = useState<CameraId[]>([]);
  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(null);
  const [routeFlashPhase, setRouteFlashPhase] = useState<"none" | "new">("none");
  const [carPathFlashKey, setCarPathFlashKey] = useState<number>(0);
  const routeFlashTimersRef = useRef<number[]>([]);

  // ===========================
  //  1) WebSocket 연결 목업쓸떈 제외
  // ===========================
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: number | null = null;

    const resetMonitorState = () => {
      setHasCamStatus(false);
      setServerState(emptyState);
      setClusterBeforeState(emptyState);
      setClusterAfterState(emptyState);
    };

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
            const applyCam = (prev: MonitorState) => ({
              ...prev,
              camerasOnMap: msg.data.camerasOnMap,
              camerasStatus: msg.data.camerasStatus,
            });
            setServerState(applyCam);
            setClusterBeforeState(applyCam);
            setClusterAfterState(applyCam);
            return;
          }
          if (msg.type === "trafficLightStatus") {
            setServerState((prev) => applyTrafficLightStatus(prev, msg.data));
            setClusterBeforeState((prev) => applyTrafficLightStatus(prev, msg.data));
            setClusterAfterState((prev) => applyTrafficLightStatus(prev, msg.data));
            return;
          }
          if (msg.type === "clusterStage") {
            const applyStage = (prev: MonitorState) => ({
              ...prev,
              carsOnMap: msg.data.carsOnMap,
              carsStatus: msg.data.carsStatus,
              obstaclesOnMap: msg.data.obstaclesOnMap ?? [],
              obstaclesStatus: msg.data.obstaclesStatus ?? [],
            });
            if (msg.data.stage === CLUSTER_BEFORE_STAGE) {
              setClusterBeforeState(applyStage);
            } else if (msg.data.stage === CLUSTER_AFTER_STAGE) {
              setClusterAfterState(applyStage);
            }
            return;
          }
          setServerState((prev) => {
            switch (msg.type) {
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
          if (msg.type === "carRouteChange") {
            setClusterBeforeState((prev) => applyRouteChange(prev, msg.data));
            setClusterAfterState((prev) => applyRouteChange(prev, msg.data));
          }
        } catch (e) {
          console.error("Invalid message from server:", e);
        }
      };

      ws.onclose = () => {
        console.warn("WebSocket closed, retry in 2s");
        resetMonitorState();
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
  const activeState = useMemo(
    () => {
      if (isClusterBefore) return clusterBeforeState;
      if (isClusterAfter) return clusterAfterState;
      return serverState;
    },
    [isClusterBefore, isClusterAfter, clusterBeforeState, clusterAfterState, serverState]
  );
  const carsOnMap = activeState.carsOnMap;
  const carsStatus = activeState.carsStatus;
  const camerasOnMap = activeState.camerasOnMap;
  const camerasStatus = activeState.camerasStatus;
  const obstaclesOnMap = activeState.obstaclesOnMap;
  const obstaclesStatus = activeState.obstaclesStatus;
  const routeChanges = activeState.routeChanges;
  const trafficLightsOnMap = activeState.trafficLightsOnMap;
  const trafficLightsStatus = activeState.trafficLightsStatus;
  const selectedIncident = useMemo(
    () => obstaclesStatus.find((ob) => ob.id === selectedIncidentId) || null,
    [obstaclesStatus, selectedIncidentId]
  );

  useEffect(() => {
    if (selectedIncidentId && !selectedIncident) {
      setSelectedIncidentId(null);
    }
  }, [selectedIncidentId, selectedIncident]);

  const carColorById = useMemo(() => {
    const allowed: readonly CarColor[] = ["red", "green", "yellow", "purple", "white"];
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
      if (n) map[status.car_id] = n;
    });
    return map;
  }, [carsOnMap, carsStatus]);

  const cameraNameById = useMemo(() => {
    const map: Record<CameraId, string> = {};
    camerasStatus.forEach((cam) => {
      map[cam.id] = cam.name;
    });
    return map;
  }, [camerasStatus]);

  const cameraPosById = useMemo(() => {
    const map: Record<CameraId, { x: number; y: number }> = {};
    camerasOnMap.forEach((cam) => {
      map[cam.cameraId] = { x: cam.x, y: cam.y };
    });
    return map;
  }, [camerasOnMap]);

  const carPaths = useMemo(() => {
    const map: Record<CarId, RoutePoint[]> = {};

    // 1) 우선 routeChange 메시지에서 온 경로 사용
    routeChanges.forEach((change) => {
      const pts = (change.newRoute ?? [])
        .map((pt) => {
          const x = Number((pt as any).x);
          const y = Number((pt as any).y);
          if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
          return { x, y };
        })
        .filter((p): p is RoutePoint => Boolean(p));
      if (pts.length > 1) {
        map[change.carId] = pts;
      }
    });

    // 2) 경로 업데이트가 없을 때는 carStatus의 path_future를 보조로 사용
    carsStatus.forEach((status) => {
      if (map[status.car_id]) return;
      const pts = (status.path_future ?? [])
        .map((pt) => {
          const x = Number((pt as any).x);
          const y = Number((pt as any).y);
          if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
          return { x, y };
        })
        .filter((p): p is RoutePoint => Boolean(p));
      if (pts.length > 1) {
        map[status.car_id] = pts;
      }
    });

    return map;
  }, [routeChanges, carsStatus]);

  const routeProgressByCar = useMemo(() => {
    const map: Record<CarId, { idx: number; total: number; ratio: number }> = {};
    carsStatus.forEach((status) => {
      const idx = Number(status.route_progress_idx ?? 0);
      const total = Number(status.route_progress_total ?? 0);
      const ratio = Number(status.route_progress_ratio ?? 0);
      map[status.car_id] = {
        idx: Number.isFinite(idx) ? idx : 0,
        total: Number.isFinite(total) ? total : 0,
        ratio: Number.isFinite(ratio) ? ratio : 0,
      };
    });
    return map;
  }, [carsStatus]);

  const routeChangesFromStatus = useMemo(() => {
    const changes: CarRouteChange[] = [];
    Object.entries(carPaths).forEach(([carId, pts]) => {
      if (!pts || pts.length < 2) return;
      const status = carsStatus.find((c) => c.car_id === carId);
      const category = (status?.category ?? "").toString().trim().toLowerCase();
      const isBatteryCategory = category.startsWith("battery");
      if (!status?.routeChanged && !isBatteryCategory) return;
      changes.push({ carId, newRoute: pts, visible: true });
    });
    return changes;
  }, [carPaths, carsStatus]);

  const mergedRouteChanges = useMemo(() => {
    const byCar = new Map<CarId, CarRouteChange>();
    routeChanges.forEach((change) => {
      if (!change?.carId || !change.newRoute || change.newRoute.length < 2) return;
      byCar.set(change.carId, change);
    });
    routeChangesFromStatus.forEach((change) => {
      if (byCar.has(change.carId)) return;
      byCar.set(change.carId, change);
    });
    return Array.from(byCar.values());
  }, [routeChanges, routeChangesFromStatus]);

  useEffect(() => {
    routeFlashTimersRef.current.forEach((timer) => window.clearTimeout(timer));
    routeFlashTimersRef.current = [];
    if (mergedRouteChanges.length === 0) {
      setRouteFlashPhase("none");
      return;
    }
    setRouteFlashPhase("new");
    const hideNew = window.setTimeout(
      () => setRouteFlashPhase("none"),
      ROUTE_FLASH_MS
    );
    routeFlashTimersRef.current = [hideNew];
    return () => {
      routeFlashTimersRef.current.forEach((timer) => window.clearTimeout(timer));
      routeFlashTimersRef.current = [];
    };
  }, [mergedRouteChanges]);

  // ===========================
  //  2) 뷰 모드 계산
  // ===========================
  useEffect(() => {
    if (selectedCarId) {
      setViewMode("carFocused");
    } else if (selectedCameraIds.length > 0) {
      setViewMode("cameraFocused");
    } else if (selectedIncidentId) {
      setViewMode("incidentFocused");
    } else {
      setViewMode("default");
    }
  }, [selectedCarId, selectedIncidentId, selectedCameraIds]);

  // ===========================
  //  3) 클릭 핸들러들
  // ===========================
  const handleCarClick = (carId: CarId) => {
    const nextSelected = selectedCarId === carId ? null : carId;
    setSelectedCarId(nextSelected);
    setSelectedIncidentId(null);
    setSelectedCameraIds([]);
    if (nextSelected) {
      setCarPathFlashKey(Date.now());
    }
  };

  const handleCameraClick = (cameraId: CameraId) => {
    setSelectedCarId(null);
    setSelectedIncidentId(null);
    setSelectedCameraIds((prev) => {
      if (prev.includes(cameraId)) {
        return prev.filter((id) => id !== cameraId);
      }
      if (prev.length < 2) {
        return [...prev, cameraId];
      }
      return [prev[1], cameraId];
    });
  };

  const handleIncidentClick = (incidentId: string | null) => {
    if (!incidentId || selectedIncidentId === incidentId) {
      setSelectedIncidentId(null);
    } else {
      setSelectedIncidentId(incidentId);
    }
    setSelectedCarId(null);
    setSelectedCameraIds([]);
  };

  // 장애물 알림 시 보여줄 차량 목록 (현재는 전체 차량)
  const carsStatusForPanel = useMemo(() => carsStatus, [carsStatus]);

  const vehiclesInAlertView = useMemo(() => carsStatusForPanel, [carsStatusForPanel]);

  const visibleRouteChanges = useMemo(() => {
    if (routeFlashPhase === "new") {
      return mergedRouteChanges;
    }
    return [];
  }, [mergedRouteChanges, routeFlashPhase]);

// Monitoring에 실제로 띄울 카메라 선택 로직 가장 가까운 애로
  const rawMonitoringCameraIds: CameraId[] = useMemo(() => {
    const uniq = (ids?: CameraId[]) =>
      Array.from(new Set((ids ?? []).filter((id): id is CameraId => Boolean(id))));

    const nearestCamera = (
      candidates: CameraId[] | undefined,
      target?: { x: number; y: number },
      allowFallbackToAll = false
    ) => {
      let ids = uniq(candidates);
      if (allowFallbackToAll && ids.length === 0) {
        ids = uniq(camerasOnMap.map((c) => c.cameraId));
      }
      if (!ids.length) return undefined;
      if (!target) return ids[0];
      let best: CameraId | undefined;
      let bestDist = Number.POSITIVE_INFINITY;
      ids.forEach((id) => {
        const pos = cameraPosById[id];
        if (!pos) return;
        const dx = pos.x - target.x;
        const dy = pos.y - target.y;
        const dist = dx * dx + dy * dy;
        if (dist < bestDist) {
          bestDist = dist;
          best = id;
        }
      });
      return best ?? ids[0];
    };

    if (viewMode === "cameraFocused" && selectedCameraIds.length > 0) {
      return uniq(selectedCameraIds).slice(0, 2);
    }

    if (viewMode === "carFocused" && selectedCarId) {
      const carStatus = carsStatus.find((c) => c.car_id === selectedCarId);
      const carPos = carsOnMap.find((c) => c.carId === selectedCarId);
      const cam = nearestCamera(
        carStatus?.cameraIds,
        carPos ? { x: carPos.x, y: carPos.y } : undefined,
        true
      );
      return cam ? [cam] : [];
    }

    if (viewMode === "incidentFocused") {
      const targetIncident = selectedIncident ?? obstaclesStatus.find((ob) => ob.cameraIds && ob.cameraIds.length);
      const targetPos = targetIncident
        ? obstaclesOnMap.find((ob) => ob.obstacleId === targetIncident.id || ob.id === targetIncident.id)
        : undefined;
      const cam = nearestCamera(
        targetIncident?.cameraIds,
        targetPos ? { x: targetPos.x, y: targetPos.y } : undefined,
        true
      );
      return cam ? [cam] : [];
    }

    return [];
  }, [
    viewMode,
    selectedCameraIds,
    selectedCarId,
    selectedIncident,
    obstaclesStatus,
    carsStatus,
    camerasStatus,
    carsOnMap,
    obstaclesOnMap,
    cameraPosById,
    camerasOnMap,
  ]);

  const [monitoringCameraIds, setMonitoringCameraIds] = useState<CameraId[]>(rawMonitoringCameraIds);
  const lastMonitoringUpdateRef = useRef<number>(0);
  const monitoringUpdateTimerRef = useRef<number | null>(null);

  useEffect(() => {
    const now = Date.now();
    const elapsed = now - lastMonitoringUpdateRef.current;

    const applyUpdate = () => {
      lastMonitoringUpdateRef.current = Date.now();
      setMonitoringCameraIds(rawMonitoringCameraIds);
      monitoringUpdateTimerRef.current = null;
    };

    if (elapsed >= 1000) {
      applyUpdate();
    } else {
      if (monitoringUpdateTimerRef.current !== null) {
        window.clearTimeout(monitoringUpdateTimerRef.current);
      }
      monitoringUpdateTimerRef.current = window.setTimeout(applyUpdate, 1000 - elapsed);
    }

    return () => {
      if (monitoringUpdateTimerRef.current !== null) {
        window.clearTimeout(monitoringUpdateTimerRef.current);
        monitoringUpdateTimerRef.current = null;
      }
    };
  }, [rawMonitoringCameraIds]);

  const monitoringCameras: CameraStatus[] = useMemo(() => {
    if (monitoringCameraIds.length === 0) return [];
    const byId = new Map(camerasStatus.map((cam) => [cam.id, cam]));
    return monitoringCameraIds
      .map((id) => byId.get(id))
      .filter((cam): cam is CameraStatus => !!cam);
  }, [monitoringCameraIds, camerasStatus]);

  const monitoringFrames = useMemo(() => {
    if (monitoringCameraIds.length === 0) return [];

    if (viewMode === "incidentFocused") {
      const cam = monitoringCameras[0];
      if (cam) {
        return [{ label: cam.name || cam.id, url: cam.streamUrl }];
      }
      return [];
    }

    if (viewMode === "cameraFocused") {
      if (monitoringCameras.length === 1) {
        const cam = monitoringCameras[0];
        const labelBase = cam.name || cam.id;
        return [
          { label: `${labelBase} (Main)`, url: cam.streamUrl },
          { label: `${labelBase} (BEV)`, url: cam.streamBEVUrl || cam.streamUrl },
        ];
      }
      return monitoringCameras.slice(0, 2).map((cam, idx) => ({
        label: `${cam.name || cam.id}`,
        url: cam.streamUrl,
      }));
    }

    return monitoringCameras.slice(0, 2).map((cam) => ({
      label: cam.name || cam.id,
      url: cam.streamUrl,
    }));
  }, [monitoringCameraIds, monitoringCameras, viewMode]);


  const isLoading = !hasCamStatus;

  return (
    <div className="app-root">
      <Header mode={appMode} onModeChange={handleModeChange} adminEnabled={isAdminPage} />
      <Layout
        viewMode={viewMode}
      >
        {/* LEFT: MAP */}
        <div className="layout__map-inner">
          {isLoading && (
            <div className="map__loading">Waiting for server data...</div>
          )}

          <MapView
            mapImage="/assets/map-track.png"
            carsOnMap={carsOnMap}
            camerasOnMap={camerasOnMap}
            obstacles={obstaclesOnMap}
            trafficLightsOnMap={trafficLightsOnMap}
            trafficLightsStatus={trafficLightsStatus}
            activeCameraIds={monitoringCameraIds}
            activeCarId={selectedCarId}
            routeChanges={visibleRouteChanges}
            routeProgressByCar={routeProgressByCar}
            carPaths={carPaths}
            carPathFlashKey={carPathFlashKey}
            onCameraClick={handleCameraClick}
          />
        </div>

        {/* RIGHT: PANELS */}
        <div className="right-panels">
          {appMode === "admin" ? (
            <>
              <CarStatusPanel
                cars={carsStatusForPanel}
                carColorById={carColorById}
                selectedCarId={selectedCarId}
                onCarClick={handleCarClick}
                scrollable
              />
              <AdminControlsPanel
                wsUrl={WS_URL}
                selectedCarId={selectedCarId}
              />
            </>
          ) : (
            <>
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
                  cars={carsStatusForPanel.filter(
                    (c) => c.car_id === selectedCarId
                  )}
                  carColorById={carColorById}
                  selectedCarId={selectedCarId}
                  onCarClick={handleCarClick}
                  detailOnly
                />
              )}

              {/* Incident */}
              {obstaclesStatus.length > 0 &&
                viewMode !== "carFocused" &&
                viewMode !== "cameraFocused" && (
                <IncidentPanel
                  alerts={
                    viewMode === "incidentFocused" && selectedIncidentId
                      ? obstaclesStatus.filter((ob) => ob.id === selectedIncidentId)
                      : obstaclesStatus
                  }
                  cameraNames={cameraNameById}
                  isActive={obstaclesStatus.length > 0}
                  onSelect={handleIncidentClick}
                />
              )}

              {/* Monitoring View */}
              {viewMode === "cameraFocused" && monitoringFrames.length > 0 && (
                <MonitoringPanel frames={monitoringFrames} />
              )}
              {viewMode === "carFocused" && monitoringFrames.length > 0 && (
                <MonitoringPanel frames={monitoringFrames} />
              )}
              {viewMode === "incidentFocused" &&
                monitoringFrames.length > 0 && (
                  <MonitoringPanel frames={monitoringFrames} />
                )}
            </>
          )}
        </div>
      </Layout>
    </div>
  );
}

export default App;
