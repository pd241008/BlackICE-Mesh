import type { ServiceStatus } from "@/lib/types";

interface ServiceStatusProps {
  gateway: ServiceStatus;
  ws: ServiceStatus;
  mlActive: boolean;
  brokerActive: boolean;
}

const tone: Record<ServiceStatus, string> = {
  online: "bg-ctos text-void",
  degraded: "bg-amber text-void",
  offline: "bg-alert text-void",
  idle: "bg-ash text-void",
};

export function ServiceStatus({
  gateway,
  ws,
  mlActive,
  brokerActive,
}: ServiceStatusProps) {
  const services: Array<{ name: string; status: ServiceStatus }> = [
    { name: "GATEWAY", status: gateway },
    { name: "MESH-WS", status: ws },
    { name: "ML-OPTIMIZER", status: mlActive ? "online" : "idle" },
    { name: "DACM-ENGINE", status: "idle" },
    { name: "RABBITMQ", status: brokerActive ? "online" : "idle" },
    { name: "POSTGRES", status: brokerActive ? "online" : "idle" },
  ];

  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {services.map((svc) => (
        <div
          key={svc.name}
          className="brutal glass-panel flex items-center gap-2 px-3 py-2 text-xs"
        >
          <span className={`h-2.5 w-2.5 ${tone[svc.status]}`} />
          <span className="tracking-[0.15em] text-ghost">{svc.name}</span>
        </div>
      ))}
    </div>
  );
}
