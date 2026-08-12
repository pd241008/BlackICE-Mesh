import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  children: ReactNode;
  tone?: "standard" | "ctos" | "alert";
  floating?: boolean;
}

const toneClass: Record<NonNullable<PanelProps["tone"]>, string> = {
  standard: "brutal",
  ctos: "brutal-ctos",
  alert: "brutal-alert",
};

export function Panel({
  title,
  children,
  tone = "standard",
  floating = false,
}: PanelProps) {
  const frame = floating ? "glass-panel-floating" : "glass-panel";
  return (
    <section className={`${frame} ${toneClass[tone]} relative z-10 p-4`}>
      <header className="mb-3 flex items-center justify-between border-b-2 border-dashed border-ash/60 pb-2">
        <h2 className="text-sm font-bold tracking-[0.2em] text-ctos uppercase">
          {"//"} {title}
        </h2>
        <span className="flex gap-1.5">
          <span className="h-2.5 w-2.5 bg-alert" />
          <span className="h-2.5 w-2.5 bg-amber" />
          <span className="h-2.5 w-2.5 bg-ctos" />
        </span>
      </header>
      {children}
    </section>
  );
}
