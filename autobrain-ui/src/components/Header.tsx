// components/Header.tsx
import { useEffect, useMemo, useState } from "react";

export const Header = () => {
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
        <div className="header__logo-mark" />
        <div className="header__logo-text">
          <span className="header__logo-title">AutoBrain</span>
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
        <span>{formattedTime}</span>
      </div>
    </header>
  );
};
