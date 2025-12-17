// components/Header.tsx
export const Header = () => {
  return (
    <header className="header">
      <div className="header__left">
        {/* 로고는 나중에 SVG로 교체 */}
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
        {/* 실제 시간은 나중에 Date로 바꾸면 됨 */}
        <span>Nov 17, 2025&nbsp;&nbsp;10:12 AM</span>
      </div>
    </header>
  );
};
