"use client";

import { useState } from "react";
import { Panel } from "@/components/ui/Panel";
import { dispatchJob } from "@/lib/api";
import type { JobType } from "@/lib/types";

interface AttackConsoleProps {
  onDispatched: (label: string) => void;
}

const ATTACKS: Array<{ key: JobType; label: string }> = [
  { key: "attack.fgsm", label: "FGSM" },
  { key: "attack.pgd", label: "PGD" },
  { key: "attack.jsma", label: "JSMA" },
];

export function AttackConsole({ onDispatched }: AttackConsoleProps) {
  const [epsilon, setEpsilon] = useState(0.15);
  const [alpha, setAlpha] = useState(0.01);
  const [steps, setSteps] = useState(40);
  const [busy, setBusy] = useState<JobType | null>(null);
  const [last, setLast] = useState<string | null>(null);

  async function fire(job: { key: JobType; label: string }) {
    setBusy(job.key);
    try {
      const payload: Record<string, number> = { epsilon };
      if (job.key === "attack.pgd") {
        payload.alpha = alpha;
        payload.steps = steps;
      }
      await dispatchJob({ type: job.key, payload });
      setLast(`${job.label} dispatched @ eps=${epsilon}`);
      onDispatched(`${job.label} JOB QUEUED → eps=${epsilon}`);
    } catch (err) {
      setLast(
        `dispatch failed: ${err instanceof Error ? err.message : String(err)}`,
      );
      onDispatched(`DISPATCH FAILED → ${job.label}`);
    } finally {
      setBusy(null);
    }
  }

  return (
    <Panel title="ATTACK CONSOLE" tone="alert" floating>
      <div className="flex flex-col gap-3">
        <div className="grid grid-cols-3 gap-2">
          <label className="flex flex-col gap-1 text-[10px] tracking-widest text-ash uppercase">
            Epsilon
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={epsilon}
              onChange={(e) => setEpsilon(Number(e.target.value))}
              className="brutal bg-void px-2 py-1 text-sm text-ctos"
            />
          </label>
          <label className="flex flex-col gap-1 text-[10px] tracking-widest text-ash uppercase">
            Alpha
            <input
              type="number"
              step="0.001"
              min="0"
              value={alpha}
              onChange={(e) => setAlpha(Number(e.target.value))}
              className="brutal bg-void px-2 py-1 text-sm text-ctos"
            />
          </label>
          <label className="flex flex-col gap-1 text-[10px] tracking-widest text-ash uppercase">
            Steps
            <input
              type="number"
              step="1"
              min="1"
              value={steps}
              onChange={(e) => setSteps(Number(e.target.value))}
              className="brutal bg-void px-2 py-1 text-sm text-ctos"
            />
          </label>
        </div>

        <div className="grid grid-cols-3 gap-2">
          {ATTACKS.map((job) => (
            <button
              key={job.key}
              type="button"
              onClick={() => fire(job)}
              disabled={busy !== null}
              className="brutal-alert bg-panel px-3 py-2 text-sm font-bold tracking-[0.2em] text-ghost uppercase transition-colors hover:bg-alert hover:text-void disabled:cursor-not-allowed disabled:opacity-40"
            >
              {busy === job.key ? "···" : job.label}
            </button>
          ))}
        </div>

        <p className="min-h-4 border-t-2 border-dashed border-ash/60 pt-2 text-xs text-ghost">
          {last ?? "> await payload injection ..."}
        </p>
      </div>
    </Panel>
  );
}
