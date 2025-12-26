// components/Layout.tsx
import type { ReactNode } from "react";
import type { ViewMode } from "../types";

interface LayoutProps {
  viewMode: ViewMode;
  hasIncident: boolean;
  children: ReactNode;
}

export const Layout = ({
  viewMode,
  hasIncident,
  children,
}: LayoutProps) => {
  const [mapArea, rightPanels] = Array.isArray(children) ? children : [children];

  const layoutClass = `layout layout--${viewMode}`;

  return (
    <main className={layoutClass}>
      <section className="layout__map">
        {mapArea}
      </section>
      <section className="layout__side">{rightPanels}</section>
    </main>
  );
};
