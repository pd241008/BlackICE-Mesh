export interface ApiEnvelope<T> {
  success: boolean;
  data: T | null;
  error: string | null;
}

export type JobType =
  | "attack.fgsm"
  | "attack.pgd"
  | "attack.jsma"
  | "defence.adversarial"
  | "defence.ensemble"
  | "evaluate.baseline"
  | "train.ensemble";

export type JobPayload = Record<string, number | string | boolean>;

export interface JobRequest {
  type: JobType;
  payload: JobPayload;
}

export interface QueuedJob {
  job_id: string;
  type: JobType;
  status: "queued";
}

export interface TelemetryResult {
  job_id: string;
  type: JobType;
  clean_accuracy?: number;
  robust_accuracy?: number;
  attack_success_rate?: number;
  relative_drop?: number;
  confidence_drop?: number;
  perturbed_features?: number;
  epsilon?: number;
  alpha?: number;
  steps?: number;
  samples?: number;
  defence_method?: string;
  num_models?: number;
  status?: string;
  error?: string;
}

export interface HealthStatus {
  status: "ok";
  service: string;
}

export type ServiceStatus =
  | "online"
  | "degraded"
  | "offline"
  | "idle";
