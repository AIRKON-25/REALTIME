// components/MapView.tsx
import type {
  CameraId,
  CameraOnMap,
  CarOnMap,
  CarId,
  CarStatus,
  ObstacleOnMap,
  RouteChangeStep,
} from "../types";

import CameraIcon from "../assets/camera-icon.svg";

const normalizeCarColor = (color: string | undefined) => {
  const normalized = (color ?? "").toString().trim().toLowerCase();
  const allowed = ["red", "green", "blue", "yellow", "purple"] as const;
  return (allowed as readonly string[]).includes(normalized) ? normalized : "red";
};

interface MapViewProps {
  mapImage: string;
  carsOnMap: CarOnMap[];
  carsStatus: CarStatus[];
  camerasOnMap: CameraOnMap[];
  obstacles: ObstacleOnMap[];
  activeCameraId: CameraId | null;
  activeCarId: CarId | null;
  activeRouteStep: RouteChangeStep | null;
  onCarClick?: (carId: CarId) => void;
  onCameraClick?: (cameraId: CameraId) => void;
}

export const MapView = ({
  mapImage,
  carsOnMap,
  carsStatus,
  camerasOnMap,
  obstacles,
  activeCameraId,
  activeCarId,
  activeRouteStep,
  onCarClick,
  onCameraClick,
}: MapViewProps) => {
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
  const cameraRadiusPx = 30;

  const paddingStyle = {
    paddingLeft: `calc(${Math.max(0, -minX) * 100}% + ${cameraRadiusPx}px)`,
    paddingRight: `calc(${Math.max(0, maxX - 1) * 100}% + ${cameraRadiusPx}px)`,
    paddingTop: `calc(${Math.max(0, -minY) * 100}% + ${cameraRadiusPx}px)`,
    paddingBottom: `calc(${Math.max(0, maxY - 1) * 100}% + ${cameraRadiusPx}px)`,
  };

  const carStatusById = carsStatus.reduce<Record<CarId, CarStatus>>((acc, status) => {
    acc[status.id] = status;
    return acc;
  }, {});

  return (
    <div className="map" style={paddingStyle}>
      <div className="map__content">
        <div className="map__image-wrapper">
          <img src={mapImage} alt="track map" className="map__image" />
        </div>

        {/* Cameras */}
        {camerasOnMap.map((cam) => (
          <img
            key={cam.id}
            src={CameraIcon}
            className={
              cam.cameraId === activeCameraId
                ? "map__camera-icon map__camera-icon--active"
                : "map__camera-icon"
            }
            style={{
              position: "absolute",
              left: `${cam.x * 100}%`,
              top: `${cam.y * 100}%`,
              width: 60,
              height: 60,
              transform: "translate(-50%, -50%)",
            }}
            onClick={() => onCameraClick?.(cam.cameraId)}
          />
        ))}

        {/* Cars */}
        {carsOnMap.map((car) => {
          const isSelected = car.carId === activeCarId;
          const isRouteChanged = car.status === "routeChanged";
          const carStatus = carStatusById[car.carId];
          const classValue =
            carStatus?.class ?? (carStatus as { cls?: number | string } | undefined)?.cls;
          const isCone = Number(classValue) === 1;
          const safeColor = normalizeCarColor(car.color);
          const carImage = isCone
            ? "/assets/rubberCone.png"
            : `/assets/car-${safeColor}.png`;
          const imageClass = isCone
            ? "map__car-image map__car-image--cone"
            : "map__car-image";
          const transform = isCone
            ? "translate(-50%, -50%)"
            : `translate(-50%, -50%) rotate(${car.yaw}deg)`;
          
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
                className={imageClass}
                onError={(e) => {
                  if (isCone) return;
                  if (e.currentTarget.src.endsWith("/assets/car-red.png")) return;
                  e.currentTarget.src = "/assets/car-red.png";
                }}
              />
            </button>
          );
        })}

        {/* Obstacles */}
        {obstacles.map((ob) => (
          <div
            key={ob.id}
            className="map__obstacle map__obstacle--cone"
            style={{
              left: `${ob.x * 100}%`,
              top: `${ob.y * 100}%`,
              transform: "translate(-50%, -100%)",
            }}
          >
            <span className="map__obstacle-icon" />
          </div>
        ))}

        {/* Route change arrow */}
        {activeRouteStep && (
          <svg className="map__route-svg">
            <defs>
              <marker
                id="arrow"
                markerWidth="8"
                markerHeight="8"
                refX="5"
                refY="3"
                orient="auto"
              >
                <path d="M0,0 L0,6 L6,3 z" />
              </marker>
            </defs>
            <line
              x1={`${activeRouteStep.from.x * 100}%`}
              y1={`${activeRouteStep.from.y * 100}%`}
              x2={`${activeRouteStep.to.x * 100}%`}
              y2={`${activeRouteStep.to.y * 100}%`}
              className="map__route-line"
              markerEnd="url(#arrow)"
            />
          </svg>
        )}
      </div>
    </div>
  );
};
