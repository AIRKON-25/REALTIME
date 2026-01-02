// components/CarStatusPanel.tsx
import { useEffect, useMemo, useRef, useState } from "react";
import type { CarColor, CarId, CarStatus } from "../types";

const normalizeCarColor = (
  color: string | undefined,
  fallback?: CarColor
): CarColor | undefined => {
  const normalized = (color ?? "").toString().trim().toLowerCase();
  const allowed: readonly CarColor[] = ["red", "green", "yellow", "purple", "white"];
  if (allowed.includes(normalized as CarColor)) return normalized as CarColor;
  return fallback;
};

const getIdOrder = (id: string | undefined | null): number | string => {
  if (!id) return "";
  const match = id.toString().match(/(\d+(?:\.\d+)?)/);
  if (match) {
    const num = Number(match[1]);
    if (Number.isFinite(num)) return num;
  }
  return id;
};

interface CarStatusPanelProps {
  cars: CarStatus[];
  carColorById?: Record<CarId, CarColor>;
  selectedCarId: CarId | null;
  onCarClick?: (id: CarId) => void;
  scrollable?: boolean;
  detailOnly?: boolean; // true면 선택된 차 1개만 큰 카드로 표시
}

export const CarStatusPanel = ({
  cars,
  carColorById,
  selectedCarId,
  onCarClick,
  scrollable,
  detailOnly,
}: CarStatusPanelProps) => {
  const [speedById, setSpeedById] = useState<Record<CarId, number>>({});
  const sortedCars = useMemo(() => {
    return [...cars].sort((a, b) => {
      const aOrder = getIdOrder(a.car_id);
      const bOrder = getIdOrder(b.car_id);
      if (typeof aOrder === "number" && typeof bOrder === "number") {
        return aOrder - bOrder;
      }
      return aOrder.toString().localeCompare(bOrder.toString());
    });
  }, [cars]);

  useEffect(() => {
    const next: Record<CarId, number> = {};
    sortedCars.forEach((car) => {
      next[car.car_id] = car.speed;
    });
    setSpeedById(next);
  }, [sortedCars]);

  if (detailOnly) {
    const car =
      sortedCars.find((c) => c.car_id === selectedCarId) ?? sortedCars[0];
    if (!car) return null;
    return (
      <section className="panel panel--card">
        <h2 className="panel__title">Car Status</h2>
        <CarStatusCard
          car={car}
          carColorById={carColorById}
          speedById={speedById}
          selected
          detailed
          onClick={onCarClick}
        />
      </section>
    );
  }

  return (
    <section className="panel panel--card">
      <h2 className="panel__title">Car Status</h2>
      <div
        className={`car-list ${scrollable ? "car-list--scrollable" : ""}`}
      >
        {sortedCars.map((car) => (
          <CarStatusCard
            key={car.car_id}
            car={car}
            carColorById={carColorById}
            speedById={speedById}
            selected={car.car_id === selectedCarId}
            onClick={onCarClick}
          />
        ))}
        {sortedCars.length === 0 && (
          <div className="panel__empty">No vehicles detected</div>
        )}
      </div>
    </section>
  );
};

interface CarStatusCardProps {
  car: CarStatus;
  carColorById?: Record<CarId, CarColor>;
  speedById?: Record<CarId, number>;
  selected?: boolean;
  detailed?: boolean;
  onClick?: (id: CarId) => void;
}

const CarStatusCard = ({
  car,
  carColorById,
  speedById,
  selected,
  detailed,
  onClick,
}: CarStatusCardProps) => {
  const statusColor = normalizeCarColor(car.color);
  const carKey = car.car_id;
  const mappedColor = carColorById?.[carKey];
  const safeColor = normalizeCarColor(statusColor ?? mappedColor, "red")!;
  const isBatteryCategory = (car.category ?? "").toString().trim().toLowerCase().startsWith("battery");
  const speedValue = speedById?.[carKey] ?? car.speed;
  const speedText = speedValue.toFixed(2);
  const isRouteChanged = !!car.routeChanged && !isBatteryCategory;
  const labelId = carKey || "car";
  const primarySrc = isBatteryCategory
    ? `/assets/carB-${safeColor}.svg`
    : isRouteChanged
      ? `/assets/carR-${safeColor}.svg`
      : `/assets/carS-${safeColor}.svg`;
  const fallbackSrc = isBatteryCategory
    ? "/assets/carB-red.svg"
    : isRouteChanged
      ? `/assets/carS-${safeColor}.svg`
      : "/assets/carS-red.svg";
  const batteryIconSrc = (() => {
    if (isBatteryCategory) return "/assets/batteryOut.png";
    const level = Number(car.battery);
    if (!Number.isFinite(level)) return "/assets/battery30.png";
    if (level >= 81) return "/assets/battery100.png";
    if (level >= 51) return "/assets/battery80.png";
    if (level >= 31) return "/assets/battery50.png";
    return "/assets/battery30.png";
  })();

  return (
    <button
      className={`car-card ${selected ? "car-card--active" : ""} ${
        isRouteChanged ? "car-card--alert" : ""
      }`}
      onClick={() => onClick?.(carKey)}
    >
      <div className="car-card__body">
        <div className="car-card__info">
          <img
            src={primarySrc}
            alt={`${labelId} icon`}
            className="car-card__icon"
            onError={(e) => {
              if (e.currentTarget.src === fallbackSrc) return;
              e.currentTarget.src = fallbackSrc;
            }}
          />
          <span className="car-card__id">
            ID : {(car.car_id ?? "car-").toString().replace("car-", "")}
          </span>
        </div>
        {isRouteChanged ? (
          <div className="car-card__route-changed">Route Changed!</div>
        ) : (
          <div className="car-card__metrics">
            <div className="car-card__metric">
              <img src="/assets/speed.png" alt="speed" className="car-card__metric-icon" />
              <span className="car-card__metric-text">{speedText} m/s</span>
            </div>
            <div className="car-card__metric">
              <img
                src={batteryIconSrc}
                alt="battery"
                className={`car-card__metric-icon ${isBatteryCategory ? "car-card__metric-icon--battery-out" : ""}`}
              />
              <span
                className={`car-card__metric-text ${
                  isBatteryCategory ? "car-card__metric-text--to-charger" : ""
                }`}
              >
                {isBatteryCategory ? "To Charger" : `${car.battery}%`}
              </span>
            </div>
          </div>
        )}
        {detailed && (
          <div className="car-card__detail-extra">
            {/* 필요하면 더 많은 정보 추가 가능 */}
          </div>
        )}
      </div>
    </button>
  );
};
