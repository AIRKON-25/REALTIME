// components/MapView.tsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import type {
  CameraId,
  CameraOnMap,
  CarOnMap,
  CarId,
  ObstacleOnMap,
  CarRouteChange,
} from "../types";

import CameraIcon from "../assets/camera-icon.svg";
import CameraIconActive from "../assets/camera-icon-active.svg";
import CameraIconVertical from "../assets/camera-icon_vertical.svg";
import CameraIconVerticalActive from "../assets/camera-icon-vertical-active.svg";

const normalizeCarColor = (color: string | undefined) => {
  const normalized = (color ?? "").toString().trim().toLowerCase();
  const allowed = ["red", "green", "blue", "yellow", "purple", "white"] as const;
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

interface MapViewProps {
  mapImage: string;
  carsOnMap: CarOnMap[];
  camerasOnMap: CameraOnMap[];
  obstacles: ObstacleOnMap[];
  activeCameraIds: CameraId[];
  activeCarId: CarId | null;
  routeChanges: CarRouteChange[];
  onCarClick?: (carId: CarId) => void;
  onCameraClick?: (cameraId: CameraId) => void;
  sizeScale?: number; // optional: tweak overlay element sizing together
}

export const MapView = ({
  mapImage,
  carsOnMap,
  camerasOnMap,
  obstacles,
  activeCameraIds,
  activeCarId,
  routeChanges,
  onCarClick,
  onCameraClick,
  sizeScale = 1,
}: MapViewProps) => {
  const mapContentRef = useRef<HTMLDivElement | null>(null);
  const mapImageRef = useRef<HTMLImageElement | null>(null);
  const [mapLayout, setMapLayout] = useState({
    contentWidth: 0,
    contentHeight: 0,
    mapWidth: 0,
    mapHeight: 0,
    offsetX: 0,
    offsetY: 0,
  });

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

  const mapScale = useMemo(() => {
    const basisWidth = mapLayout.mapWidth || mapLayout.contentWidth;
    if (!basisWidth) return 1;
    const normalizedSizeScale = sizeScale > 0 ? sizeScale : 1;
    const scale = (basisWidth / 1000) * normalizedSizeScale;
    return Math.min(1.4, Math.max(0.4, scale));
  }, [mapLayout, sizeScale]);

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

  const paddingStyle = {
    paddingLeft: `calc(${Math.max(0, -minX) * 100}% + ${cameraRadiusPx}px)`,
    paddingRight: `calc(${Math.max(0, maxX - 1) * 100}% + ${cameraRadiusPx}px)`,
    paddingTop: `calc(${Math.max(0, -minY) * 100}% + ${cameraRadiusPx}px)`,
    paddingBottom: `calc(${Math.max(0, maxY - 1) * 100}% + ${cameraRadiusPx}px)`,
  };

  const mapStyle = {
    ...paddingStyle,
    ["--map-scale" as string]: mapScale,
  } as CSSProperties;

  const overlayStyle: CSSProperties = {
    position: "absolute",
    left: mapLayout.offsetX || 0,
    top: mapLayout.offsetY || 0,
    width: mapLayout.mapWidth || "100%",
    height: mapLayout.mapHeight || "100%",
  };

  const buildRoutePoints = (route: { x: number; y: number }[]) =>
    route.map((point) => `${point.x * 100},${point.y * 100}`).join(" ");

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
            const safeColor = normalizeCarColor(car.color);
            const carImage = `/assets/car-${safeColor}.png`;
            const transform = `translate(-50%, -50%) rotate(${car.yaw}deg)`;
            
            return (
              <button
                key={car.id}
                className={`map__car ${isSelected ? "map__car--active" : ""} ${
                  isRouteChanged ? "map__car--warning" : ""
                }`}
                style={{
                  left: `${car.x * 100}%`,
                  top: `${car.y * 100}%`,
                  transform,
                }}
                onClick={() => onCarClick?.(car.carId)}
              >
                <img
                  src={carImage}
                  alt={`${car.carId} icon`}
                  className="map__car-image"
                  onError={(e) => {
                    if (e.currentTarget.src.endsWith("/assets/car-red.png")) return;
                    e.currentTarget.src = "/assets/car-red.png";
                  }}
                />
              </button>
            );
          })}

          {/* Obstacles */}
          {obstacles.map((ob) => {
            const obstacleSrc =
              ob.kind === "barricade" ? "/assets/barricade.png" : "/assets/rubberCone.png";
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

          {/* Route change arrows */}
          {routeChanges.length > 0 && (
            <svg
              className="map__route-svg"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              style={{ width: "100%", height: "100%" }}
            >
              <defs>
                <marker
                  id="arrow-new"
                  markerWidth="8"
                  markerHeight="8"
                  refX="5"
                  refY="3"
                  orient="auto"
                >
                  <path d="M0,0 L0,6 L6,3 z" fill="var(--danger)" />
                </marker>
              </defs>
              {routeChanges.map((change, idx) => {
                const newPoints = buildRoutePoints(change.newRoute);
                return (
                  <g key={`${change.carId}-${idx}`}>
                    {change.newRoute.length > 1 && (
                      <polyline
                        points={newPoints}
                        className="map__route-line map__route-line--new"
                        markerEnd="url(#arrow-new)"
                      />
                    )}
                  </g>
                );
              })}
            </svg>
          )}
        </div>
      </div>
    </div>
  );
};
