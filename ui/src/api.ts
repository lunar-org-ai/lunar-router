/**
 * Backend client. Single place that knows wire format / endpoints.
 *
 * In dev, Vite proxies /v1/* → http://127.0.0.1:8002 (see vite.config.ts).
 * In prod, the UI is served from the same origin as the backend, so the
 * relative paths just work.
 */

export interface HistoryMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface AgentRunResponse {
  response: string | null;
  trace_id: string;
  duration_ms: number;
  success: boolean;
  error: string | null;
}

export class ApiError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function runAgent(
  request: string,
  history?: HistoryMessage[],
): Promise<AgentRunResponse> {
  const res = await fetch('/v1/webhook', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ request, history }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as AgentRunResponse;
}

export interface IntrospectToolCall {
  tool: string;
  input: Record<string, unknown>;
  output_preview: string;
}

export interface IntrospectResponse {
  response: string;
  tool_calls: IntrospectToolCall[];
  success: boolean;
  error: string | null;
  model: string | null;
  iterations: number;
}

export async function introspect(
  request: string,
  history?: HistoryMessage[],
): Promise<IntrospectResponse> {
  const res = await fetch('/v1/introspect', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ request, history }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as IntrospectResponse;
}

// ---------- versions ----------

export interface LessonSummary {
  id: string;
  version: string | null;
  kind: string;
  status: string;
  title: string;
  summary: string;
  voice: string | null;
  delta: { overall_score?: number; pass_rate?: number; per_rubric?: Record<string, number> };
  mutations: string[];
  parent_version: string | null;
  candidate_id: string | null;
  promoted_at: string | null;
  ledger_entry_id: string | null;
  proposal_source: string | null;
  n_traces: number | null;
}

export interface VersionInfo {
  id: string;
  is_live: boolean;
  status: 'live' | 'rolled_back' | 'archived';
  snapshot_path: string;
  promoted_at: string | null;
  rolled_back_at: string | null;
  lesson: LessonSummary | null;
}

export interface RollbackResult {
  version: string;
  previous_version: string;
  rolled_back: boolean;
}

export async function listVersions(): Promise<VersionInfo[]> {
  const res = await fetch('/v1/versions');
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as VersionInfo[];
}

// ---------- lessons ----------
//
// AHE three-pillar fields ride along on LessonSummary:
//   component  → mutations
//   experience → delta (rubric movement) + (future) linked traces
//   decision   → voice + proposal_source + (future) prediction.rationale

export async function listLessons(): Promise<LessonSummary[]> {
  const res = await fetch('/v1/lessons');
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as LessonSummary[];
}

// ---------- metrics overview ----------

export interface MetricsOverview {
  today_count: number;
  active_5min: number;
  pending_review: number;
  trust_score: number;
  trust_score_delta_30d: number;
  trust_history_30d: number[];
  resolution_rate: number | null;
  avg_latency_ms: number | null;
  avg_cost_usd: number | null;
  csat: number | null;
  computed_at: string;
}

export async function getMetricsOverview(): Promise<MetricsOverview> {
  const res = await fetch('/v1/metrics/overview');
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as MetricsOverview;
}

// ---------- traces (Technical / Traces) ----------

export interface TraceStageView {
  stage: string | null;
  technique: string;
  variant: string;
  duration_ms: number;
  docs_in: number;
  docs_out: number;
  response_set: boolean | null;
  routing_model: string | null;
  error: string | null;
}

export interface TraceSummary {
  trace_id: string;
  timestamp: string;
  request: string;
  response: string | null;
  duration_ms: number;
  success: boolean;
  error: string | null;
  agent_version: string | null;
  n_stages: number;
}

export interface TracesPage {
  date: string;
  available_dates: string[];
  total_filtered: number;
  items: TraceSummary[];
  has_more: boolean;
}

export interface TraceDetail extends TraceSummary {
  stages: TraceStageView[];
  metadata: Record<string, unknown>;
}

export async function listTraces(opts: {
  date?: string;
  limit?: number;
  offset?: number;
  success?: boolean;
  agent_version?: string;
  q?: string;
} = {}): Promise<TracesPage> {
  const params = new URLSearchParams();
  if (opts.date) params.set('date', opts.date);
  if (typeof opts.limit === 'number') params.set('limit', String(opts.limit));
  if (typeof opts.offset === 'number') params.set('offset', String(opts.offset));
  if (typeof opts.success === 'boolean') params.set('success', String(opts.success));
  if (opts.agent_version) params.set('agent_version', opts.agent_version);
  if (opts.q) params.set('q', opts.q);
  const qs = params.toString();
  const url = qs ? `/v1/traces?${qs}` : '/v1/traces';
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as TracesPage;
}

export async function getTrace(id: string): Promise<TraceDetail> {
  const res = await fetch(`/v1/traces/${encodeURIComponent(id)}`);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as TraceDetail;
}

// ---------- agent config ----------

export interface AgentPromptView {
  path: string;
  content: string;
}

export interface AgentModelsView {
  small: string | null;
  big: string | null;
  confidence_threshold: number | null;
}

export interface IntegrationStatus {
  name: string;
  available: boolean;
  detail: string | null;
}

export interface AgentKeyStatus {
  name: string;
  env_var: string;
  set: boolean;
  mask: string | null;
}

export interface AgentConfigView {
  version: string;
  description: string | null;
  system_prompt: AgentPromptView;
  models: AgentModelsView;
  integrations: IntegrationStatus[];
  keys: AgentKeyStatus[];
}

export interface ManualEditResult {
  new_version: string;
  lesson_id: string;
  parent_version: string;
}

export type PromptUpdateResponse = AgentPromptView & ManualEditResult;
export type RouteUpdateResponse = AgentModelsView & ManualEditResult;

export async function getAgentConfig(): Promise<AgentConfigView> {
  const res = await fetch('/v1/agent/config');
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as AgentConfigView;
}

export async function updatePrompt(content: string): Promise<PromptUpdateResponse> {
  const res = await fetch('/v1/agent/prompt', {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ content }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as PromptUpdateResponse;
}

export async function updateRoute(p: {
  small?: string;
  big?: string;
  confidence_threshold?: number;
}): Promise<RouteUpdateResponse> {
  const res = await fetch('/v1/agent/route', {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(p),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as RouteUpdateResponse;
}

// ---------- policy ----------

export type PolicyMode = 'auto' | 'review' | 'off';

export interface AutoRollbackView {
  csat_drop: number;
  resolution_drop: number;
  window_hours: number;
  notify_channels: string[];
}

export interface PolicyView {
  mode: PolicyMode | string;
  auto_min_lift: number;
  overrides: Record<string, PolicyMode | string>;
  auto_rollback: AutoRollbackView;
}

export async function getPolicy(): Promise<PolicyView> {
  const res = await fetch('/v1/policy');
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as PolicyView;
}

export async function updatePolicy(p: {
  mode: string;
  auto_min_lift: number;
  overrides?: Record<string, string>;
  auto_rollback?: AutoRollbackView;
}): Promise<PolicyView> {
  const res = await fetch('/v1/policy', {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(p),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as PolicyView;
}

export async function getLesson(id: string): Promise<LessonSummary> {
  const res = await fetch(`/v1/lessons/${encodeURIComponent(id)}`);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as LessonSummary;
}

export async function approveLesson(
  id: string,
  reviewer?: string,
): Promise<LessonSummary> {
  const res = await fetch(`/v1/lessons/${encodeURIComponent(id)}/approve`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ reviewer }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as LessonSummary;
}

export async function rejectLesson(
  id: string,
  reason?: string,
): Promise<LessonSummary> {
  const res = await fetch(`/v1/lessons/${encodeURIComponent(id)}/reject`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ reason }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as LessonSummary;
}

export interface RubricResult {
  rubric: string;
  type: string;
  score: number;
  passed: boolean;
  detail: string | null;
}

export interface LessonTraceCase {
  golden_id: string;
  request: string;
  response: string | null;
  duration_ms: number | null;
  success: boolean;
  error: string | null;
  trace_id: string | null;
  rubric_results: RubricResult[];
}

export interface LessonTracesResponse {
  lesson_id: string;
  candidate_id: string | null;
  suite: string | null;
  agent_version: string | null;
  has_report: boolean;
  note: string | null;
  cases: LessonTraceCase[];
}

export async function getLessonTraces(id: string): Promise<LessonTracesResponse> {
  const res = await fetch(`/v1/lessons/${encodeURIComponent(id)}/traces`);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as LessonTracesResponse;
}

export async function requeueLesson(id: string): Promise<LessonSummary> {
  const res = await fetch(`/v1/lessons/${encodeURIComponent(id)}/requeue`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: '{}',
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as LessonSummary;
}

export async function rollbackVersion(version: string, reason?: string): Promise<RollbackResult> {
  const res = await fetch(`/v1/versions/${encodeURIComponent(version)}/rollback`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ reason }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as RollbackResult;
}
