import { Panel } from "@/components/ui/Panel";

interface MetricsPanelProps {
  clean: number;
  robust: number;
  asr: number;
  samples?: number;
}

export function MetricsPanel({
  clean,
  robust,
  asr,
  samples,
}: MetricsPanelProps) {
  return (
    <Panel title="DEFENCE DIAGNOSTICS" tone="ctos" floating>
      <div className="grid grid-cols-1 gap-3">
        <div>
          <div className="mb-1 flex justify-between text-xs">
            <span className="tracking-[0.2em] text-ash uppercase">
              Clean Accuracy
            </span>
            <span className="font-bold text-ctos">{clean.toFixed(2)}%</span>
          </div>
          <div className="brutal h-5 w-full bg-void">
            <div className="h-full bg-ctos" style={{ width: `${clean}%` }} />
          </div>
        </div>

        <div>
          <div className="mb-1 flex justify-between text-xs">
            <span className="tracking-[0.2em] text-ash uppercase">
              Robust Accuracy
            </span>
            <span className="font-bold text-amber">{robust.toFixed(2)}%</span>
          </div>
          <div className="brutal h-5 w-full bg-void">
            <div className="h-full bg-amber" style={{ width: `${robust}%` }} />
          </div>
        </div>

        <div>
          <div className="mb-1 flex justify-between text-xs">
            <span className="tracking-[0.2em] text-ash uppercase">
              Attack Success Rate
            </span>
            <span className="font-bold text-alert">{asr.toFixed(2)}%</span>
          </div>
          <div className="brutal h-5 w-full bg-void">
            <div className="h-full bg-alert" style={{ width: `${asr}%` }} />
          </div>
        </div>

        {samples !== undefined ? (
          <p className="mt-2 border-t-2 border-dashed border-ash/60 pt-2 text-xs text-ash">
            SAMPLES EVALUATED: <span className="text-ghost">{samples}</span>
          </p>
        ) : null}
      </div>
    </Panel>
  );
}
