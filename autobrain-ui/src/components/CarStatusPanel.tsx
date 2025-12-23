// components/CarStatusPanel.tsx
import { useEffect, useRef, useState } from "react";
import type { CarColor, CarId, CarStatus } from "../types";

const normalizeCarColor = (
  color: string | undefined,
  fallback?: CarColor
): CarColor | undefined => {
  const normalized = (color ?? "").toString().trim().toLowerCase();
  const allowed: readonly CarColor[] = ["red", "green", "blue", "yellow", "purple"];
  if (allowed.includes(normalized as CarColor)) return normalized as CarColor;
  return fallback;
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
  const latestCarsRef = useRef<CarStatus[]>(cars);
  const [speedById, setSpeedById] = useState<Record<CarId, number>>({});

  useEffect(() => {
    latestCarsRef.current = cars;
  }, [cars]);

  useEffect(() => {
    const updateSpeeds = () => {
      const next: Record<CarId, number> = {};
      latestCarsRef.current.forEach((car) => {
        next[car.id] = car.speed;
      });
      setSpeedById(next);
    };
    updateSpeeds();
    const timer = window.setInterval(updateSpeeds, 1000);
    return () => window.clearInterval(timer);
  }, []);

  if (detailOnly) {
    const car = cars.find((c) => c.id === selectedCarId) ?? cars[0];
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
        {cars.map((car) => (
          <CarStatusCard
            key={car.id}
            car={car}
            carColorById={carColorById}
            speedById={speedById}
            selected={car.id === selectedCarId}
            onClick={onCarClick}
          />
        ))}
        {cars.length === 0 && (
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
  const mappedColor = carColorById?.[car.id];
  const safeColor = normalizeCarColor(statusColor ?? mappedColor, "red")!;
  const speedValue = speedById?.[car.id] ?? car.speed;
  const speedText = speedValue.toFixed(2);

  return (
    <button
      className={`car-card ${selected ? "car-card--active" : ""} ${
        car.routeChanged ? "car-card--alert" : ""
      }`}
      onClick={() => onClick?.(car.id)}
    >
      <img
        src={`/assets/carS-${safeColor}.png`}
        alt={`${car.id} icon`}
        className="car-card__icon"
        onError={(e) => {
          if (e.currentTarget.src.endsWith("/assets/carS-red.png")) return;
          e.currentTarget.src = "/assets/carS-red.png";
        }}
      />
      <div className="car-card__body">
        <div className="car-card__row">
          <span className="car-card__id">ID : {car.id.replace("car-", "")}</span>
          <span className="car-card__speed">{speedText} m/s</span>
          <span className="car-card__battery">{car.battery}%</span>
        </div>
        <div className="car-card__row car-card__row--labels">
          <span>출발지</span>
          <span>목적지</span>
        </div>
        {car.routeChanged && (
          <div className="car-card__route-changed">Route Changed!</div>
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
