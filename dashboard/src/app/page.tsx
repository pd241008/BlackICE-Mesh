"use client";

import { useCallback, useState } from "react";
import { AttackConsole } from "@/components/AttackConsole";
import { EventLog } from "@/components/EventLog";
import { MetricsPanel } from "@/components/MetricsPanel";
import { ServiceStatus } from "@/components/ServiceStatus";
import { TerminalHeader } from "@/components/TerminalHeader";
import { StatCard } from "@/components/ui/StatCard";
import { useTelemetry } from "@/hooks/useTelemetry";

export default function Home() {
  const { events, connected } = useTelemetry();
  const [logLines, setLogLines] = useState<string[]>([]);

  const latest = events.find(
    (ev) =>
      ev.clean_accuracy !== undefined &&
      ev.robust_accuracy !== undefined &&
      ev.attack_success_rate !== undefined,
  );

  const onDispatched = useCallback((line: string) => {
    setLogLines((prev) => [line, ...prev].slice(0, 8));
  }, []);

  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <div className="grid-backdrop" />
      <TerminalHeader wsConnected={connected} />

      <main className="relative z-10 mx-auto grid max-w-7xl grid-cols-1 gap-4 p-4 lg:grid-cols-12">
        {/* Left rail: diagnostics */}
        <div className="flex flex-col gap-4 lg:col-span-7">
          <section className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <StatCard
              label="Clean Accuracy"
              value={latest ? `${latest.clean_accuracy?.toFixed(2)}%` : "--"}
              delta={latest ? `eps=${latest.epsilon}` : "no data"}
              accent="ctos"
            />
            <StatCard
              label="Robust Accuracy"
              value={latest ? `${latest.robust_accuracy?.toFixed(2)}%` : "--"}
              delta={latest ? `samples=${latest.samples}` : "no data"}
              accent="amber"
            />
            <StatCard
              label="Attack Success Rate"
              value={latest ? `${latest.attack_success_rate?.toFixed(2)}%` : "--"}
              delta={latest ? `drop=${latest.relative_drop}%` : "no data"}
              accent="alert"
            />
          </section>

          <MetricsPanel
            clean={latest?.clean_accuracy ?? 0}
            robust={latest?.robust_accuracy ?? 0}
            asr={latest?.attack_success_rate ?? 0}
            samples={latest?.samples}
          />

          <ServiceStatus
            gateway={events.length > 0 ? "online" : "idle"}
            ws={connected ? "online" : "degraded"}
            mlActive={events.some((ev) => ev.clean_accuracy !== undefined)}
            brokerActive={events.length > 0}
          />
        </div>

        {/* Right rail: execution */}
        <div className="flex flex-col gap-4 lg:col-span-5">
          <AttackConsole onDispatched={onDispatched} />
          <EventLog events={events} />

          <div className="brutal glass-panel p-3 text-xs leading-relaxed">
            <p className="mb-1 tracking-[0.25em] text-ash uppercase">
              {"// kernel journal"}
            </p>
            <ul className="flex flex-col gap-1">
              {logLines.length === 0 ? (
                <li className="text-ash">&gt; no commands yet</li>
              ) : (
                logLines.map((line, i) => (
                  <li key={i} className="text-ghost">
                    &gt; {line}
                  </li>
                ))
              )}
            </ul>
          </div>
        </div>
      </main>

      <footer className="relative z-10 border-t-2 border-grid px-4 py-3 text-center text-[10px] tracking-[0.35em] text-ash uppercase">
        BlackICE-Mesh // gateway-service · dacm-engine · ml-optimizer · dashboard
      </footer>
    </div>
  );
}
