"use client";

import { useEffect, useState } from "react";
import { fetchHealth } from "@/lib/api";
import type { ServiceStatus } from "@/lib/types";

interface TerminalHeaderProps {
  wsConnected: boolean;
}

export function TerminalHeader({ wsConnected }: TerminalHeaderProps) {
  const [gateway, setGateway] = useState<ServiceStatus>("offline");
  const [upTime, setUpTime] = useState(0);

  useEffect(() => {
    const tick = () => setUpTime((t) => t + 1);
    const timer = setInterval(tick, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        await fetchHealth();
        if (!cancelled) setGateway("online");
      } catch {
        if (!cancelled) setGateway("offline");
      }
    };
    poll();
    const timer = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const mm = Math.floor(upTime / 60);
  const ss = upTime % 60;

  return (
    <header className="brutal glass-panel sticky top-0 z-20 flex items-center justify-between px-4 py-2">
      <div className="flex items-center gap-3">
        <span className="text-lg font-bold text-ctos">▚▞ BLACKICE-MESH</span>
        <span className="hidden text-xs tracking-[0.3em] text-ash uppercase md:inline">
          adversarial telemetry mesh
        </span>
      </div>

      <div className="flex items-center gap-4 text-xs">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-ctos" />
          UPTIME {mm.toString().padStart(2, "0")}:{ss.toString().padStart(2, "0")}
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className={`h-2 w-2 rounded-full ${
              gateway === "online" ? "bg-ctos" : "bg-alert"
            }`}
          />
          GATEWAY {gateway.toUpperCase()}
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className={`h-2 w-2 rounded-full ${
              wsConnected ? "bg-ctos" : "bg-amber"
            }`}
          />
          WS {wsConnected ? "LIVE" : "RETRY"}
        </span>
      </div>
    </header>
  );
}
