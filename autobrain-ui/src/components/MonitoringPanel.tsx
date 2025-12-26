// components/MonitoringPanel.tsx
import type { CameraStatus } from "../types";

interface MonitoringPanelProps {
  cameras: CameraStatus[];
}

export const MonitoringPanel = ({ cameras }: MonitoringPanelProps) => {
  const visibleCameras = cameras.slice(0, 2);

  return (
    <section className="panel panel--card monitoring">
      <h2 className="panel__title">Monitoring View</h2>
      {visibleCameras.length > 0 ? (
        <div
          className={`monitoring__grid ${
            visibleCameras.length > 1 ? "monitoring__grid--dual" : ""
          }`}
        >
          {visibleCameras.map((camera) => (
            <div key={camera.id} className="monitoring__camera">
              <div className="monitoring__camera-label">{camera.name}</div>
              <div className="monitoring__frame">
                <img
                  src={camera.streamUrl}
                  alt={camera.name}
                  className="monitoring__image"
                />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="panel__empty">Select a camera or vehicle</div>
      )}
    </section>
  );
};
