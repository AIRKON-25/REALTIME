// components/IncidentPanel.tsx
import type { ObstacleStatus } from "../types";

interface IncidentPanelProps {
  alert: ObstacleStatus | null;
  cameraLabel?: string | null;
  isActive: boolean;
}

export const IncidentPanel = ({ alert, cameraLabel, isActive }: IncidentPanelProps) => {
  const cameraText = cameraLabel ?? alert?.cameraId ?? "";

  return (
    <section
      className={`panel panel--card incident ${
        isActive ? "incident--active" : ""
      }`}
    >
      <h2 className="panel__title">Incident Alert</h2>
      {alert ? (
        <div className="incident__content">
          <div className="incident__icon">
            <img
              src="/assets/warning-signs.png"
              alt="Obstacle warning"
              className="incident__icon-image"
            />
          </div>
          <div>
            <div className="incident__title">Obstacle detected</div>
            <div className="incident__description">
              ID: {alert.id} · Class: {alert.class}
              {cameraText ? ` · Camera: ${cameraText}` : ""}
            </div>
          </div>
        </div>
      ) : (
        <div className="panel__empty">No obstacle alerts</div>
      )}
    </section>
  );
};
