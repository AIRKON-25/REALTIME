// components/MonitoringPanel.tsx
interface MonitoringFrame {
  label: string;
  url: string;
}

interface MonitoringPanelProps {
  frames: MonitoringFrame[];
}

export const MonitoringPanel = ({ frames }: MonitoringPanelProps) => {
  if (frames.length === 0) {
    return null;
  }

  return (
    <section className="panel panel--card monitoring">
      <h2 className="panel__title">Monitoring View</h2>
      <div
        className={`monitoring__grid ${
          frames.length > 1 ? "monitoring__grid--dual" : ""
        }`}
      >
        {frames.map((frame, idx) => (
          <div key={`${frame.label}-${idx}`} className="monitoring__camera">
            <div className="monitoring__camera-label">{frame.label}</div>
            <div className="monitoring__frame">
              <img
                src={frame.url}
                alt={frame.label}
                className="monitoring__image"
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};
