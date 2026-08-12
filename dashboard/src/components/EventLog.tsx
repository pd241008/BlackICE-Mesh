import { Panel } from "@/components/ui/Panel";
import type { TelemetryResult } from "@/lib/types";

interface EventLogProps {
  events: TelemetryResult[];
}

function formatEvent(ev: TelemetryResult): string {
  const bits = [`[${ev.type}]`, ev.job_id.slice(0, 8)];
  if (ev.clean_accuracy !== undefined) bits.push(`CLEAN=${ev.clean_accuracy}%`);
  if (ev.robust_accuracy !== undefined)
    bits.push(`ROBUST=${ev.robust_accuracy}%`);
  if (ev.attack_success_rate !== undefined)
    bits.push(`ASR=${ev.attack_success_rate}%`);
  if (ev.error) bits.push(`ERR=${ev.error}`);
  return bits.join(" ");
}

export function EventLog({ events }: EventLogProps) {
  return (
    <Panel title="TELEMETRY FEED" floating>
      <ul className="flex max-h-72 flex-col gap-1 overflow-y-auto pr-2 text-xs leading-relaxed">
        {events.length === 0 ? (
          <li className="text-ash">
            &gt; awaiting telemetry over websocket
            <span className="cursor-blink" />
          </li>
        ) : (
          events.map((ev) => (
            <li
              key={ev.job_id}
              className={ev.error ? "text-alert" : "text-ctos"}
            >
              <span className="text-ash">&gt;</span> {formatEvent(ev)}
            </li>
          ))
        )}
      </ul>
    </Panel>
  );
}
