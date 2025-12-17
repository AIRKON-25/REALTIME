// components/Layout.tsx
import type { ReactNode } from "react";
import type { ViewMode } from "../types";

interface LayoutProps {
  viewMode: ViewMode;
  hasIncident: boolean;
  showBackButton: boolean;
  onBackClick?: () => void;
  children: ReactNode;
}

export const Layout = ({
  viewMode,
  hasIncident,
  showBackButton,
  onBackClick,
  children,
}: LayoutProps) => {
  const [mapArea, rightPanels] = Array.isArray(children) ? children : [children];

  const layoutClass = `layout layout--${viewMode}`;

  return (
    <main className={layoutClass}>
      <section className="layout__map">
        {mapArea}
        {showBackButton && (
          <button className="layout__back-button" onClick={onBackClick}>
            Back to default
          </button>
        )}
      </section>
      <section className="layout__side">{rightPanels}</section>
    </main>
  );
};
