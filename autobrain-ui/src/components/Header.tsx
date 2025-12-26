// components/Header.tsx
import { useEffect, useMemo, useState } from "react";

type AppMode = "monitor" | "admin";

interface HeaderProps {
  mode: AppMode;
  onModeChange: (mode: AppMode) => void;
  adminEnabled?: boolean;
}

export const Header = ({ mode, onModeChange, adminEnabled = false }: HeaderProps) => {
  const [now, setNow] = useState(() => new Date());
  const formatter = useMemo(
    () =>
      new Intl.DateTimeFormat(undefined, {
        year: "numeric",
        month: "short",
        day: "2-digit",
        hour: "numeric",
        minute: "2-digit",
      }),
    []
  );

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const formattedTime = formatter.format(now);

  return (
    <header className="header">
      <div className="header__left">
        <img src="/assets/autobrain-logo.svg" alt="AutoBrain logo" className="header__logo-img" />
        <div className="header__logo-text">
          <span className="header__logo-subtitle">Vehicle Monitoring System</span>
        </div>
      </div>
      <div className="header__center">
        <span className="header__status-dot header__status-dot--on" />
        <span className="header__status-label">CONNECTED</span>
        <span className="header__status-dot header__status-dot--on" />
        <span className="header__status-label">SERVER</span>
        <span className="header__status-dot header__status-dot--on" />
        <span className="header__status-label">NETWORK</span>
      </div>
      <div className="header__right">
        {adminEnabled && (
          <div className="header__mode">
            <button
              type="button"
              className={`header__mode-button${mode === "monitor" ? " header__mode-button--active" : ""}`}
              onClick={() => onModeChange("monitor")}
            >
              Monitor
            </button>
            <button
              type="button"
              className={`header__mode-button${mode === "admin" ? " header__mode-button--active" : ""}`}
              onClick={() => onModeChange("admin")}
            >
              Admin
            </button>
          </div>
        )}
        <span>{formattedTime}</span>
      </div>
    </header>
  );
};
