import type {
  ApiEnvelope,
  HealthStatus,
  JobRequest,
  QueuedJob,
  TelemetryResult,
} from "./types";

export const GATEWAY_URL =
  process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${GATEWAY_URL}${path}`, init);
  const body = (await res.json()) as ApiEnvelope<T>;
  if (!body.success || body.error) {
    throw new Error(body.error ?? `request failed with status ${res.status}`);
  }
  return body.data as T;
}

export async function dispatchJob(req: JobRequest): Promise<QueuedJob> {
  return request<QueuedJob>("/api/v1/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export async function fetchHealth(): Promise<HealthStatus> {
  return request<HealthStatus>("/api/v1/health");
}

export interface ResultsResponse {
  results: TelemetryResult[];
}

export async function fetchRecentResults(): Promise<TelemetryResult[]> {
  const data = await request<ResultsResponse>("/api/v1/results");
  return data.results;
}
