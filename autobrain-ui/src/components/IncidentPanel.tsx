// components/IncidentPanel.tsx
import type { Incident } from "../types";

interface IncidentPanelProps {
  incident: Incident | null;
  isActive: boolean;
  onClick?: () => void;
}

export const IncidentPanel = ({ incident, isActive, onClick }: IncidentPanelProps) => {
  return (
    <section
      className={`panel panel--card incident ${
        isActive ? "incident--active" : ""
      }`}
      onClick={onClick}
    >
      <h2 className="panel__title">Incident Alert</h2>
      {incident ? (
        <div className="incident__content">
          <div className="incident__icon" />
          <div>
            <div className="incident__title">{incident.title}</div>
            <div className="incident__description">{incident.description}</div>
          </div>
        </div>
      ) : (
        <div className="panel__empty">No incidents</div>
      )}
    </section>
  );
};
