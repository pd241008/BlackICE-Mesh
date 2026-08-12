interface StatCardProps {
  label: string;
  value: string;
  delta?: string;
  accent?: "ctos" | "alert" | "amber";
}

const accentText: Record<NonNullable<StatCardProps["accent"]>, string> = {
  ctos: "text-ctos",
  alert: "text-alert",
  amber: "text-amber",
};

export function StatCard({ label, value, delta, accent = "ctos" }: StatCardProps) {
  return (
    <div className="brutal glass-panel flex flex-col gap-1 p-3">
      <span className="text-[10px] tracking-[0.25em] text-ash uppercase">
        {label}
      </span>
      <span className={`text-3xl font-bold ${accentText[accent]}`}>{value}</span>
      {delta ? (
        <span className="text-xs text-ghost">{delta}</span>
      ) : (
        <span className="text-xs text-ash">---</span>
      )}
    </div>
  );
}
