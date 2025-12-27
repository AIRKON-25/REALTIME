// components/Layout.tsx
import type { ReactNode } from "react";
import type { ViewMode } from "../types";

interface LayoutProps {
  viewMode: ViewMode;
  children: ReactNode;
}

export const Layout = ({
  viewMode,
  children,
}: LayoutProps) => {
  const [mapArea, rightPanels] = Array.isArray(children) ? children : [children];

  const layoutClass = `layout layout--${viewMode}`;
  const sideWidth = viewMode === "default" ? 277 : 630;

  return (
    <main className={layoutClass} style={{ ["--layout-side-width" as const]: `${sideWidth}px` }}>
      <section className="layout__map">
        {mapArea}
      </section>
      <section className="layout__side">{rightPanels}</section>
    </main>
  );
};
