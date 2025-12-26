// components/IncidentPanel.tsx
import type { CameraId, ObstacleStatus } from "../types";

interface IncidentPanelProps {
  alerts: ObstacleStatus[];
  cameraNames?: Record<CameraId, string>;
  isActive: boolean;
}

export const IncidentPanel = ({ alerts, cameraNames, isActive }: IncidentPanelProps) => {
  return (
    <section
      className={`panel panel--card incident ${
        isActive ? "incident--active" : ""
      }`}
    >
      <h2 className="panel__title">Incident Alert</h2>
      {alerts.length > 0 ? (
        <div className="incident__list">
          {alerts.map((alert) => {
            const cameraText =
              (alert.cameraId && cameraNames?.[alert.cameraId]) || alert.cameraId || "";
            return (
              <div key={alert.id} className="incident__content">
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
            );
          })}
        </div>
      ) : (
        <div className="panel__empty">No obstacle alerts</div>
      )}
    </section>
  );
};
