// components/MapView.tsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import type {
  CameraId,
  CameraOnMap,
  CarOnMap,
  CarId,
  ObstacleOnMap,
  TrafficLightOnMap,
  TrafficLightStatus,
  CarRouteChange,
  RoutePoint,
} from "../types";

import CameraIcon from "../assets/camera-icon.svg";
import CameraIconActive from "../assets/camera-icon-active.svg";
import CameraIconVertical from "../assets/camera-icon_vertical.svg";
import CameraIconVerticalActive from "../assets/camera-icon-vertical-active.svg";
import ArrowRect from "../assets/arrow-rectangle.png";
import ArrowHead from "../assets/arrow.png";

const normalizeCarColor = (color: string | undefined) => {
  const normalized = (color ?? "").toString().trim().toLowerCase();
  const allowed = ["red", "green", "yellow", "purple", "white"] as const;
  return (allowed as readonly string[]).includes(normalized) ? normalized : "red";
};

const isSideCamera = (cam: CameraOnMap) => {
  const distX = Math.min(cam.x, 1 - cam.x);
  const distY = Math.min(cam.y, 1 - cam.y);
  return distX < distY;
};

const getCameraIconSrc = (cam: CameraOnMap, isActive: boolean) => {
  if (isSideCamera(cam)) {
    return isActive ? CameraIconVerticalActive : CameraIconVertical;
  }
  return isActive ? CameraIconActive : CameraIcon;
};

interface RouteSprite {
  id: string;
  x: number;
  y: number;
  angleDeg: number;
  kind: "rect" | "arrow";
  carId: CarId;
}

const ROUTE_STEP = 0.035; // normalized distance per sprite (~3.5% of map width/height)
const RECT_BASE_ANGLE = -90; // adjust if your PNG default orientation differs
const ARROW_BASE_ANGLE = 180; // arrow.png points left by default
const CLICK_PATH_DURATION_MS = 2000;

const buildRouteSprites = (
  carId: CarId,
  points: RoutePoint[],
  options?: { carPos?: { x: number; y: number }; idPrefix?: string }
): RouteSprite[] => {
  if (!points || points.length < 2) return [];
  const prefix = options?.idPrefix ?? "route";
  const samples: RouteSprite[] = [];
  let counter = 0;

  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = Math.hypot(dx, dy);
    if (dist === 0) continue;
    const steps = Math.max(1, Math.floor(dist / ROUTE_STEP));
    const stepSize = dist / steps;
    const angleDeg = (Math.atan2(dy, dx) * 180) / Math.PI;
    for (let s = 0; s < steps; s++) {
      const t = ((s + 0.5) * stepSize) / dist;
      samples.push({
        id: `${prefix}-${carId}-seg-${i}-${s}-${counter++}`,
        x: a.x + dx * t,
        y: a.y + dy * t,
        angleDeg: angleDeg + RECT_BASE_ANGLE,
        kind: "rect",
        carId,
      });
    }
  }

  if (options?.carPos && samples.length > 0) {
    const { carPos } = options;
    let nearestIdx = 0;
    let nearestDist = Number.POSITIVE_INFINITY;
    samples.forEach((sample, idx) => {
      const d = Math.hypot(sample.x - carPos.x, sample.y - carPos.y);
      if (d < nearestDist) {
        nearestDist = d;
        nearestIdx = idx;
      }
    });
    samples.splice(0, nearestIdx);
  }

  if (points.length >= 2) {
    const tailPoint = points[points.length - 1];
    const prevPoint = points[points.length - 2];
    const dx = tailPoint.x - prevPoint.x;
    const dy = tailPoint.y - prevPoint.y;
    const forwardDeg = (Math.atan2(dy, dx) * 180) / Math.PI;
    samples.push({
      id: `${prefix}-${carId}-arrow-${counter}`,
      x: tailPoint.x,
      y: tailPoint.y,
      angleDeg: forwardDeg + ARROW_BASE_ANGLE,
      kind: "arrow",
      carId,
    });
  }

  return samples;
};

interface MapViewProps {
  mapImage: string;
  carsOnMap: CarOnMap[];
  camerasOnMap: CameraOnMap[];
  obstacles: ObstacleOnMap[];
  trafficLightsOnMap: TrafficLightOnMap[];
  trafficLightsStatus: TrafficLightStatus[];
  activeCameraIds: CameraId[];
  activeCarId: CarId | null;
  routeChanges: CarRouteChange[];
  routeProgressByCar?: Record<CarId, { idx: number; total: number; ratio: number }>;
  carPaths?: Record<CarId, RoutePoint[]>;
  carPathFlashKey?: number;
  onCameraClick?: (cameraId: CameraId) => void;
  sizeScale?: number; // optional: tweak overlay element sizing together
}

export const MapView = ({
  mapImage,
  carsOnMap,
  camerasOnMap,
  obstacles,
  trafficLightsOnMap,
  trafficLightsStatus,
  activeCameraIds,
  activeCarId,
  routeChanges,
  routeProgressByCar,
  carPaths,
  carPathFlashKey,
  onCameraClick,
  sizeScale = 1,
}: MapViewProps) => {
  const mapContentRef = useRef<HTMLDivElement | null>(null);
  const mapImageRef = useRef<HTMLImageElement | null>(null);
  const [routeSprites, setRouteSprites] = useState<RouteSprite[]>([]);
  const [visibleRouteCounts, setVisibleRouteCounts] = useState<Record<CarId, number>>({});
  const [clickedRouteSprites, setClickedRouteSprites] = useState<RouteSprite[]>([]);
  const [visibleClickedRouteCounts, setVisibleClickedRouteCounts] = useState<Record<CarId, number>>({});
  const [mapLayout, setMapLayout] = useState({
    contentWidth: 0,
    contentHeight: 0,
    mapWidth: 0,
    mapHeight: 0,
    offsetX: 0,
    offsetY: 0,
  });
  const clickPathTimerRef = useRef<number | null>(null);
  const clickPathStepTimersRef = useRef<number[]>([]);
  const routeHideTimerRef = useRef<number | null>(null);
  const routeStepTimersRef = useRef<number[]>([]);
  const lastRouteChangeSigRef = useRef<string | null>(null);
  const carsOnMapRef = useRef<CarOnMap[]>([]);
  const carPathsRef = useRef<Record<CarId, RoutePoint[]>>({});

  const recomputeLayout = useCallback(() => {
    const contentEl = mapContentRef.current;
    if (!contentEl) return;
    const contentRect = contentEl.getBoundingClientRect();
    const contentWidth = contentRect.width;
    const contentHeight = contentRect.height;
    if (!contentWidth || !contentHeight) return;

    const naturalWidth = mapImageRef.current?.naturalWidth || contentWidth;
    const naturalHeight = mapImageRef.current?.naturalHeight || contentHeight;
    const imageRatio = naturalWidth / Math.max(naturalHeight, 1);
    const containerRatio = contentWidth / contentHeight;

    let mapWidth = contentWidth;
    let mapHeight = contentHeight;
    let offsetX = 0;
    let offsetY = 0;

    if (containerRatio > imageRatio) {
      mapHeight = contentHeight;
      mapWidth = contentHeight * imageRatio;
      offsetX = (contentWidth - mapWidth) / 2;
    } else {
      mapWidth = contentWidth;
      mapHeight = contentWidth / imageRatio;
      offsetY = (contentHeight - mapHeight) / 2;
    }

    setMapLayout((prev) => {
      if (
        prev.contentWidth === contentWidth &&
        prev.contentHeight === contentHeight &&
        prev.mapWidth === mapWidth &&
        prev.mapHeight === mapHeight &&
        prev.offsetX === offsetX &&
        prev.offsetY === offsetY
      ) {
        return prev;
      }
      return { contentWidth, contentHeight, mapWidth, mapHeight, offsetX, offsetY };
    });
  }, []);

  useEffect(() => {
    recomputeLayout();
    const observer = new ResizeObserver(() => recomputeLayout());
    const target = mapContentRef.current;
    const imageEl = mapImageRef.current;
    if (target) observer.observe(target);
    if (imageEl) observer.observe(imageEl);
    window.addEventListener("resize", recomputeLayout);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", recomputeLayout);
    };
  }, [recomputeLayout]);

  useEffect(() => {
    carsOnMapRef.current = carsOnMap;
  }, [carsOnMap]);

  useEffect(() => {
    carPathsRef.current = carPaths ?? {};
  }, [carPaths]);

  const mapScale = useMemo(() => {
    const basisWidth = mapLayout.mapWidth || mapLayout.contentWidth;
    if (!basisWidth) return 1;
    const normalizedSizeScale = sizeScale > 0 ? sizeScale : 1;
    const scale = (basisWidth / 1000) * normalizedSizeScale;
    return Math.min(1.4, Math.max(0.4, scale));
  }, [mapLayout, sizeScale]);

  useEffect(() => {
    if (!routeChanges.length) return;
    const sig = JSON.stringify(routeChanges);
    if (sig === lastRouteChangeSigRef.current) return;
    lastRouteChangeSigRef.current = sig;

    const sprites: RouteSprite[] = [];
    routeChanges.forEach((change, changeIdx) => {
      if (change.visible === false) return;
      if (!change.newRoute || change.newRoute.length < 2) return;
      const progress = routeProgressByCar?.[change.carId];
      const startIdx = Math.max(0, Math.floor(progress?.idx ?? 0));
      const trimmed = change.newRoute.slice(startIdx);
      if (trimmed.length < 2) return;
      const carPos = carsOnMapRef.current.find((c) => c.carId === change.carId);
      sprites.push(
        ...buildRouteSprites(change.carId, trimmed, {
          carPos: carPos ? { x: carPos.x, y: carPos.y } : undefined,
          idPrefix: `change-${changeIdx}-${Date.now()}`,
        })
      );
    });
    setRouteSprites(sprites);
  }, [routeChanges]);

  useEffect(() => {
    if (routeHideTimerRef.current) {
      window.clearTimeout(routeHideTimerRef.current);
      routeHideTimerRef.current = null;
    }
    routeStepTimersRef.current.forEach((t) => window.clearInterval(t));
    routeStepTimersRef.current = [];

    if (!routeSprites.length) {
      setVisibleRouteCounts({});
      return;
    }

    const timers: number[] = [];
    const doneCars = new Set<CarId>();
    const byCar: Record<CarId, RouteSprite[]> = {};
    routeSprites.forEach((sprite) => {
      if (!byCar[sprite.carId]) byCar[sprite.carId] = [];
      byCar[sprite.carId].push(sprite);
    });

    setVisibleRouteCounts(() => {
      const next: Record<CarId, number> = {};
      Object.entries(byCar).forEach(([carId, sprites]) => {
        if (sprites.length === 0) return;
        next[carId] = 0; // always restart animation so the final arrow is shown
      });
      return next;
    });

    Object.entries(byCar).forEach(([carId, sprites]) => {
      const total = sprites.length;
      if (total === 0) return;
      const timer = window.setInterval(() => {
        setVisibleRouteCounts((prev) => {
          const now = prev[carId] ?? 0;
          if (now >= total) return prev;
          const next = now + 1;
          if (next >= total) {
            doneCars.add(carId);
            if (doneCars.size === Object.keys(byCar).length && !routeHideTimerRef.current) {
              routeHideTimerRef.current = window.setTimeout(() => {
                setRouteSprites([]);
                setVisibleRouteCounts({});
                lastRouteChangeSigRef.current = null;
                routeHideTimerRef.current = null;
              }, CLICK_PATH_DURATION_MS);
            }
          }
          return { ...prev, [carId]: next };
        });
      }, 90);
      timers.push(timer);
    });
    routeStepTimersRef.current = timers;

    return () => {
      timers.forEach((t) => window.clearInterval(t));
      if (routeHideTimerRef.current) {
        window.clearTimeout(routeHideTimerRef.current);
        routeHideTimerRef.current = null;
      }
      routeStepTimersRef.current = [];
    };
  }, [routeSprites]);

  useEffect(() => {
    if (clickPathTimerRef.current) {
      window.clearTimeout(clickPathTimerRef.current);
      clickPathTimerRef.current = null;
    }
    clickPathStepTimersRef.current.forEach((t) => window.clearInterval(t));
    clickPathStepTimersRef.current = [];

    if (!activeCarId) {
      setClickedRouteSprites([]);
      setVisibleClickedRouteCounts({});
      return;
    }

    const progress = routeProgressByCar?.[activeCarId];
    const startIdx = Math.max(0, Math.floor(progress?.idx ?? 0));
    const points = (carPathsRef.current?.[activeCarId] || []).slice(startIdx);
    if (!points || points.length < 2) {
      setClickedRouteSprites([]);
      setVisibleClickedRouteCounts({});
      return;
    }

    const carPos = carsOnMapRef.current.find((c) => c.carId === activeCarId);
    const sprites = buildRouteSprites(activeCarId, points, {
      carPos: carPos ? { x: carPos.x, y: carPos.y } : undefined,
      idPrefix: `click-${activeCarId}-${carPathFlashKey ?? "0"}`,
    });
    setClickedRouteSprites(sprites);
    setVisibleClickedRouteCounts({ [activeCarId]: 0 });

    if (sprites.length) {
      const total = sprites.length;
      const timer = window.setInterval(() => {
        setVisibleClickedRouteCounts((prev) => {
          const current = prev[activeCarId] ?? 0;
          if (current >= total) return prev;
          const next = current + 1;
          if (next >= total) {
            window.clearInterval(timer);
            clickPathStepTimersRef.current = [];
            if (!clickPathTimerRef.current) {
              clickPathTimerRef.current = window.setTimeout(() => {
                setClickedRouteSprites([]);
                setVisibleClickedRouteCounts({});
                clickPathTimerRef.current = null;
              }, CLICK_PATH_DURATION_MS);
            }
          }
          return { ...prev, [activeCarId]: next };
        });
      }, 90);
      clickPathStepTimersRef.current.push(timer);
    }

    return () => {
      if (clickPathTimerRef.current) {
        window.clearTimeout(clickPathTimerRef.current);
        clickPathTimerRef.current = null;
      }
      clickPathStepTimersRef.current.forEach((t) => window.clearInterval(t));
      clickPathStepTimersRef.current = [];
    };
  }, [activeCarId, carPathFlashKey]);

  const points = [
    { x: 0, y: 0 },
    { x: 1, y: 1 },
    ...camerasOnMap.map((cam) => ({ x: cam.x, y: cam.y })),
  ];

  const minX = Math.min(...points.map((p) => p.x));
  const maxX = Math.max(...points.map((p) => p.x));
  const minY = Math.min(...points.map((p) => p.y));
  const maxY = Math.max(...points.map((p) => p.y));

  // Camera icon size (px) to ensure padding accounts for its radius.
  const cameraSizePx = 60 * mapScale;
  const cameraRadiusPx = cameraSizePx / 2;
  const trafficLightSizePx = 58 * mapScale;

  const paddingStyle = {
    paddingLeft: `calc(${Math.max(0, -minX) * 100}% + ${cameraRadiusPx}px)`,
    paddingRight: `calc(${Math.max(0, maxX - 1) * 100}% + ${cameraRadiusPx}px)`,
    paddingTop: `calc(${Math.max(0, -minY) * 100}% + ${cameraRadiusPx}px)`,
    paddingBottom: `calc(${Math.max(0, maxY - 1) * 100}% + ${cameraRadiusPx}px)`,
  };

  const mapStyle = {
    ...paddingStyle,
  } as CSSProperties;

  const overlayStyle: CSSProperties = {
    position: "absolute",
    left: mapLayout.offsetX || 0,
    top: mapLayout.offsetY || 0,
    width: mapLayout.mapWidth || "100%",
    height: mapLayout.mapHeight || "100%",
  };

  const carsWithVisibleRoute = useMemo(() => {
    const ids = new Set<CarId>();
    routeSprites.forEach((s) => ids.add(s.carId));
    clickedRouteSprites.forEach((s) => ids.add(s.carId));
    return ids;
  }, [routeSprites, clickedRouteSprites]);

  return (
    <div className="map" style={mapStyle}>
      <div className="map__content" ref={mapContentRef}>
        <div className="map__image-wrapper">
          <img
            ref={mapImageRef}
            src={mapImage}
            alt="track map"
            className="map__image"
            onLoad={recomputeLayout}
          />
        </div>

        <div className="map__overlay" style={overlayStyle}>
          {/* Cameras */}
          {camerasOnMap.map((cam) => {
            const isActive = activeCameraIds.includes(cam.cameraId);
            const iconSrc = getCameraIconSrc(cam, isActive);
            return (
              <img
                key={cam.id}
                src={iconSrc}
                alt={`${cam.cameraId} camera`}
                className={
                  isActive
                    ? "map__camera-icon map__camera-icon--active"
                    : "map__camera-icon"
                }
                style={{
                  position: "absolute",
                  left: `${cam.x * 100}%`,
                  top: `${cam.y * 100}%`,
                  transform: "translate(-50%, -50%)",
                }}
                onClick={() => onCameraClick?.(cam.cameraId)}
              />
            );
          })}

          {/* Cars */}
          {carsOnMap.map((car) => {
            const isSelected = car.carId === activeCarId;
            const isRouteChanged = car.status === "routeChanged";
            const isOnTop = carsWithVisibleRoute.has(car.carId);
            const safeColor = normalizeCarColor(car.color);
            const carImage = `/assets/car-${safeColor}.svg`;
            const transform = `translate(-50%, -50%) rotate(${car.yaw}deg)`;
            
            return (
              <div
                key={car.id}
                className={`map__car ${isSelected ? "map__car--active" : ""} ${
                  isRouteChanged ? "map__car--warning" : ""
                } ${isOnTop ? "map__car--on-top" : ""}`}
                style={{
                  left: `${car.x * 100}%`,
                  top: `${car.y * 100}%`,
                  transform,
                  backgroundImage: `url(${carImage})`,
                  backgroundSize: "cover",
                  backgroundPosition: "center",
                  backgroundRepeat: "no-repeat",
                }}
                aria-label={`${car.carId} icon`}
              />
            );
          })}

          {/* Traffic Lights */}
          {trafficLightsOnMap.map((tl) => {
            const status = trafficLightsStatus.find(
              (s) => s.trafficLightId === tl.trafficLightId
            );
            const color = (status?.light ?? "green").toLowerCase();
            const left = status?.left_green;
            const isFourWay = [2, 8, 5].includes(tl.trafficLightId);
            const variant = isFourWay ? 4 : 3;
            const suffix = left ? "-left" : "";
            const imgSrc = `/assets/trafficLight${variant}-${color}${suffix}.png`;
            const transform = `translate(-50%, -50%) rotate(${tl.yaw}deg)`;
            return (
              <img
                key={tl.id}
                src={imgSrc}
                alt={`traffic light ${tl.trafficLightId}`}
                className="map__traffic-light"
                style={{
                  position: "absolute",
                  left: `${tl.x * 100}%`,
                  top: `${tl.y * 100}%`,
                  width: `${trafficLightSizePx}px`,
                  transform,
                }}
              />
            );
          })}

          {/* Obstacles */}
          {obstacles.map((ob) => {
            const obstacleSrc =
              ob.kind === "barricade" ? "/assets/barricade.svg" : "/assets/rubberCone.svg";
            const obstacleAlt =
              ob.kind === "barricade" ? "barricade obstacle" : "rubber cone obstacle";
            return (
              <div
                key={ob.id}
                className={`map__obstacle ${
                  ob.kind === "barricade"
                    ? "map__obstacle--barricade"
                    : "map__obstacle--cone"
                }`}
                style={{
                  left: `${ob.x * 100}%`,
                  top: `${ob.y * 100}%`,
                  transform: "translate(-50%, -100%)",
                }}
              >
                <img src={obstacleSrc} alt={obstacleAlt} className="map__obstacle-icon" />
              </div>
            );
          })}

          {/* Route change sprites (PNG) */}
          {routeSprites.length > 0 && (
            <div className="map__route-layer">
              {(() => {
                const progress: Record<CarId, number> = {};
                return routeSprites.map((sprite) => {
                  const visibleCount = visibleRouteCounts[sprite.carId] ?? 0;
                  const shownSoFar = progress[sprite.carId] ?? 0;
                  if (shownSoFar >= visibleCount) return null;
                  progress[sprite.carId] = shownSoFar + 1;
                  const isArrow = sprite.kind === "arrow";
                  const size = isArrow ? 34 * mapScale : 26 * mapScale;
                  const imgSrc = isArrow ? ArrowHead : ArrowRect;
                  return (
                    <img
                      key={sprite.id}
                      src={imgSrc}
                      alt={isArrow ? "route arrow" : "route segment"}
                      className={isArrow ? "map__route-arrow" : "map__route-rect"}
                      style={{
                        left: `${sprite.x * 100}%`,
                        top: `${sprite.y * 100}%`,
                        width: `${size}px`,
                        height: isArrow ? `${size}px` : `${size * 0.45}px`,
                        transform: `translate(-50%, -50%) rotate(${sprite.angleDeg}deg)`,
                      }}
                    />
                  );
                });
              })()}
            </div>
          )}

          {/* Clicked car path sprites (PNG) */}
          {clickedRouteSprites.length > 0 && (
            <div className="map__route-layer">
              {(() => {
                const progress: Record<CarId, number> = {};
                return clickedRouteSprites.map((sprite) => {
                  const visibleCount = visibleClickedRouteCounts[sprite.carId] ?? 0;
                  const shownSoFar = progress[sprite.carId] ?? 0;
                  if (shownSoFar >= visibleCount) return null;
                  progress[sprite.carId] = shownSoFar + 1;
                  const isArrow = sprite.kind === "arrow";
                  const size = isArrow ? 34 * mapScale : 26 * mapScale;
                  const imgSrc = isArrow ? ArrowHead : ArrowRect;
                  return (
                    <img
                      key={sprite.id}
                      src={imgSrc}
                      alt={isArrow ? "car path arrow" : "car path segment"}
                      className={isArrow ? "map__route-arrow" : "map__route-rect"}
                      style={{
                        left: `${sprite.x * 100}%`,
                        top: `${sprite.y * 100}%`,
                        width: `${size}px`,
                        height: isArrow ? `${size}px` : `${size * 0.45}px`,
                        transform: `translate(-50%, -50%) rotate(${sprite.angleDeg}deg)`,
                      }}
                    />
                  );
                });
              })()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
