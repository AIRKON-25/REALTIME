// components/MonitoringPanel.tsx
import type { CameraStatus } from "../types";

interface MonitoringPanelProps {
  camera: CameraStatus | null;
}

export const MonitoringPanel = ({ camera }: MonitoringPanelProps) => {
  return (
    <section className="panel panel--card monitoring">
      <h2 className="panel__title">Monitoring View</h2>
      {camera ? (
        <>
          <div className="monitoring__camera-label">{camera.name}</div>
          <div className="monitoring__frame">
            {/* 실제로는 video 혹은 <img src={camera.streamUrl} /> */}
            <img
              src={camera.streamUrl}
              alt={camera.name}
              className="monitoring__image"
            />
          </div>
        </>
      ) : (
        <div className="panel__empty">Select a camera or vehicle</div>
      )}
    </section>
  );
};
