interface BarProps {
  label: string;
  value: number;
  accent?: "ctos" | "alert" | "amber";
}

const barAccent: Record<NonNullable<BarProps["accent"]>, string> = {
  ctos: "bg-ctos",
  alert: "bg-alert",
  amber: "bg-amber",
};

export function Bar({ label, value, accent = "ctos" }: BarProps) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-xs">
        <span className="tracking-[0.2em] text-ash uppercase">{label}</span>
        <span className="font-bold text-ghost">{clamped.toFixed(2)}%</span>
      </div>
      <div className="brutal h-5 w-full overflow-hidden bg-void">
        <div
          className={`h-full ${barAccent[accent]}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}
