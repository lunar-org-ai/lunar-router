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
  routing_decision: Record<string, unknown> | null;
  error: string | null;
}

export interface HistoryTurn {
  role: string;
  content: string;
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
  routing_model: string | null;
  session_id: string | null;
  n_turns: number;
  // P16.2 — cost telemetry. Optional because pre-P16.2 traces lack these.
  tokens_in?: number | null;
  tokens_out?: number | null;
  cost_usd?: number | null;
  // P16.3 — flag state (server-stamped from traces/flagged/<date>.jsonl).
  flagged?: boolean;
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
  history: HistoryTurn[];
}

export interface SessionTurn {
  trace_id: string;
  timestamp: string;
  request: string;
  response: string | null;
  success: boolean;
  error: string | null;
  agent_version: string | null;
  duration_ms: number;
}

export interface SessionDetail {
  session_id: string;
  n_turns: number;
  started_at: string;
  last_at: string;
  turns: SessionTurn[];
}

export async function getSession(id: string): Promise<SessionDetail> {
  const res = await fetch(`/v1/sessions/${encodeURIComponent(id)}`);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as SessionDetail;
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

// P16.2 — CSAT signal.
export interface TraceFeedbackEntry {
  trace_id: string;
  score: number;
  comment: string | null;
  at: string;
}

export interface TraceFeedbackResponse extends TraceFeedbackEntry {
  n_total: number;
}

export async function submitTraceFeedback(
  trace_id: string,
  score: number,
  comment?: string,
): Promise<TraceFeedbackResponse> {
  const res = await fetch(`/v1/traces/${encodeURIComponent(trace_id)}/feedback`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ score, comment: comment || null }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as TraceFeedbackResponse;
}

export async function listTraceFeedback(trace_id: string): Promise<TraceFeedbackEntry[]> {
  const res = await fetch(`/v1/traces/${encodeURIComponent(trace_id)}/feedback`);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as TraceFeedbackEntry[];
}

// P16.3 — flag persistence + auto-flag.
export interface TraceFlagEntry {
  trace_id: string;
  reason: string | null;
  source: 'manual' | 'csat_low' | 'latency_outlier' | 'error' | 'unflag';
  at: string;
}

export interface TraceFlagResponse {
  trace_id: string;
  flagged: boolean;
  last_row: TraceFlagEntry;
}

export async function flagTrace(
  trace_id: string,
  reason?: string,
): Promise<TraceFlagResponse> {
  const res = await fetch(`/v1/traces/${encodeURIComponent(trace_id)}/flag`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ reason: reason || null }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as TraceFlagResponse;
}

export async function unflagTrace(trace_id: string): Promise<TraceFlagResponse> {
  const res = await fetch(`/v1/traces/${encodeURIComponent(trace_id)}/flag`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as TraceFlagResponse;
}

export async function listTraceFlag(trace_id: string): Promise<TraceFlagEntry[]> {
  const res = await fetch(`/v1/traces/${encodeURIComponent(trace_id)}/flag`);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as TraceFlagEntry[];
}

export interface LiveTraceEvent {
  trace_id: string;
  timestamp: string;
  session_id: string | null;
  agent_version: string | null;
  duration_ms: number;
  success: boolean;
  error: string | null;
  n_stages: number;
  n_turns: number;
  request_preview: string;
}

export interface TraceStreamHandlers {
  onTrace: (event: LiveTraceEvent) => void;
  onOpen?: () => void;
  onError?: (e: Event) => void;
}

/**
 * Subscribe to /v1/traces/stream (SSE). Returns a close fn.
 * The browser auto-reconnects on transient drops.
 */
export function subscribeTraces(handlers: TraceStreamHandlers): () => void {
  const es = new EventSource('/v1/traces/stream');
  if (handlers.onOpen) es.addEventListener('open', handlers.onOpen);
  if (handlers.onError) es.addEventListener('error', handlers.onError);
  es.addEventListener('trace', (msg) => {
    try {
      const event = JSON.parse((msg as MessageEvent<string>).data) as LiveTraceEvent;
      handlers.onTrace(event);
    } catch {
      // malformed event; skip
    }
  });
  return () => es.close();
}

// ---------- evals (Technical / Eval suites) ----------

export interface RubricSpec {
  name: string;
  type: string;
  params: Record<string, unknown>;
}

export interface GoldenView {
  id: string;
  request: string;
  expected: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface SuiteSummary {
  name: string;
  description: string | null;
  n_goldens: number;
  n_rubrics: number;
  aggregation: string;
  last_run_at: string | null;
  last_overall_score: number | null;
  last_pass_rate: number | null;
  last_agent_version: string | null;
  n_runs: number;
  author: 'human' | 'agent';
  baseline_pass_rate: number | null;
}

export interface SuiteDetail extends SuiteSummary {
  goldens: GoldenView[];
  rubrics: RubricSpec[];
}

export interface ReportSummary {
  report_id: string;
  suite: string;
  agent_version: string;
  started_at: string;
  finished_at: string;
  overall_score: number;
  pass_rate: number;
  n_passed: number;
  n_total: number;
  is_candidate: boolean;
  candidate_id: string | null;
}

export interface ReportCase {
  golden_id: string;
  request: string;
  response: string | null;
  duration_ms: number | null;
  success: boolean;
  error: string | null;
  trace_id: string | null;
  rubric_results: RubricResult[];
}

export interface ReportDetail extends ReportSummary {
  cases: ReportCase[];
  per_rubric: Record<string, unknown>;
}

export async function listSuites(): Promise<SuiteSummary[]> {
  const res = await fetch('/v1/evals/suites');
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as SuiteSummary[];
}

export async function getSuite(name: string): Promise<SuiteDetail> {
  const res = await fetch(`/v1/evals/suites/${encodeURIComponent(name)}`);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as SuiteDetail;
}

export async function listReports(opts: {
  suite?: string;
  limit?: number;
  candidate_only?: boolean;
} = {}): Promise<ReportSummary[]> {
  const params = new URLSearchParams();
  if (opts.suite) params.set('suite', opts.suite);
  if (typeof opts.limit === 'number') params.set('limit', String(opts.limit));
  if (opts.candidate_only) params.set('candidate_only', 'true');
  const qs = params.toString();
  const url = qs ? `/v1/evals/reports?${qs}` : '/v1/evals/reports';
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as ReportSummary[];
}

export async function getReport(reportId: string): Promise<ReportDetail> {
  const res = await fetch(`/v1/evals/reports/${encodeURIComponent(reportId)}`);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as ReportDetail;
}

export async function runSuite(name: string): Promise<ReportSummary> {
  const res = await fetch(`/v1/evals/suites/${encodeURIComponent(name)}/run`, {
    method: 'POST',
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as ReportSummary;
}

export interface RunAllResult {
  reports: ReportSummary[];
  errors: { suite: string; error: string }[];
}

export async function runAllSuites(): Promise<RunAllResult> {
  const res = await fetch('/v1/evals/run_all', { method: 'POST' });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as RunAllResult;
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


// --- P15.3 — Router config feeds ---

export interface RouterConfigView {
  version: number | null;
  k: number;
  model_count: number;
  cost_weight: number;
  embedder_model: string;
  embedding_dim: number;
  last_fit_at: string | null;
  fitted_from: Record<string, unknown> | null;
  cold_start: boolean;
}

export interface RouterRuleHistoryEntry {
  when: string;
  what: string;
  lesson_id?: string | null;
}

export interface RouterRuleApi {
  id: string;
  name: string;
  when: string;
  then: string;
  share: number;
  cost: number;
  auth: 'agent' | 'human';
  enabled: boolean;
  isDefault?: boolean;
  rationale: string;
  history: RouterRuleHistoryEntry[];
  samples: string[];
}

export interface RouterCandidateView {
  lesson_id: string;
  version: number;
  title: string;
  summary: string;
  delta: Record<string, unknown>;
  created_at: string | null;
  review_link: string;
}

export interface RouterHealthView {
  cold_start: boolean;
  version: number | null;
  k: number | null;
  model_count: number | null;
  cost_weight: number | null;
  last_fit_at: string | null;
  last_fit_age_hours: number | null;
  trace_count_since_last_fit: number;
  drift_score: number | null;
  drift_baseline: number | null;
  needs_reclustering: boolean | null;
  current_avg_error: number | null;
  current_win_rate: number | null;
  cluster_distribution: Record<string, number> | null;
  fitted_from: Record<string, unknown> | null;
  sample_size: number;
}

async function _getJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as T;
}

export const getRouterConfig = () => _getJson<RouterConfigView>('/v1/router/config');
export const getRouterRules = () => _getJson<RouterRuleApi[]>('/v1/router/rules');
export const getRouterCandidates = () =>
  _getJson<RouterCandidateView[]>('/v1/router/candidates');
export const getRouterHealth = () => _getJson<RouterHealthView>('/v1/router/health');

// ===========================================================================
// P15.4.2 — Datasets
// ===========================================================================

export interface DatasetView {
  id: string;
  name: string;
  desc: string;
  size: number;
  source: string;
  sourceType: 'auto' | 'manual';
  fresh: string;
  use: string[];
  owner: 'agent' | 'human';
  growing: boolean;
}

export interface DatasetSampleView {
  id: string;
  preview: string;
  tag: string | null;
}

export interface DatasetHistoryEntry {
  when: string;
  what: string;
}

export interface DatasetDetail extends DatasetView {
  samples: DatasetSampleView[];
  history: DatasetHistoryEntry[];
}

export interface DatasetHealth {
  name: string;
  size: number;
  cluster_distribution: Record<string, number>;
  coverage_gap_score: number | null;
  last_curation_at: string | null;
}

export interface DatasetCreateRequest {
  name: string;
  desc?: string;
  source?: string;
  sourceType?: 'auto' | 'manual';
  use?: string[];
  owner?: 'agent' | 'human';
  growing?: boolean;
}

export interface DatasetUpdateRequest {
  desc?: string;
  use?: string[];
  growing?: boolean;
}

export interface DatasetListFilters {
  use?: string;
  owner?: 'agent' | 'human';
  sourceType?: 'auto' | 'manual';
}

export async function getDatasets(filters: DatasetListFilters = {}): Promise<DatasetView[]> {
  const params = new URLSearchParams();
  if (filters.use) params.set('use', filters.use);
  if (filters.owner) params.set('owner', filters.owner);
  if (filters.sourceType) params.set('sourceType', filters.sourceType);
  const qs = params.toString();
  return _getJson<DatasetView[]>(qs ? `/v1/datasets?${qs}` : '/v1/datasets');
}

export const getDataset = (name: string) =>
  _getJson<DatasetDetail>(`/v1/datasets/${encodeURIComponent(name)}`);

export const getDatasetHealth = (name: string) =>
  _getJson<DatasetHealth>(`/v1/datasets/${encodeURIComponent(name)}/health`);

export async function createDataset(body: DatasetCreateRequest): Promise<DatasetView> {
  const res = await fetch('/v1/datasets', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${text.slice(0, 200)}`);
  }
  return (await res.json()) as DatasetView;
}

export async function updateDataset(
  name: string,
  body: DatasetUpdateRequest,
): Promise<DatasetView> {
  const res = await fetch(`/v1/datasets/${encodeURIComponent(name)}`, {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${text.slice(0, 200)}`);
  }
  return (await res.json()) as DatasetView;
}

export const exportDatasetUrl = (name: string) =>
  `/v1/datasets/${encodeURIComponent(name)}/export`;

export async function deleteDataset(name: string): Promise<void> {
  const res = await fetch(`/v1/datasets/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  });
  if (!res.ok && res.status !== 204) {
    const text = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${text.slice(0, 200)}`);
  }
}

// ─── Onboarding (P1.11) ─────────────────────────────────────────

export interface OnboardingState {
  template: string | null;
  name: string;
  company: string;
  prompt: string;
  model: string;
  tools: string[];
  channels: string[];
  completed: boolean;
  completed_at: string | null;
  skipped: boolean;
}

export interface OnboardingCompleteRequest {
  template: string | null;
  name: string;
  company: string;
  prompt: string;
  model: string;
  tools: string[];
  channels: string[];
}

export const getOnboardingState = () =>
  _getJson<OnboardingState>('/v1/onboarding/state');

export async function completeOnboarding(
  body: OnboardingCompleteRequest,
): Promise<OnboardingState> {
  const res = await fetch('/v1/onboarding/complete', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

export async function skipOnboarding(): Promise<OnboardingState> {
  const res = await fetch('/v1/onboarding/skip', { method: 'POST' });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

// Conversational onboarding turn (P1.12)

export interface OnboardingChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface OnboardingTurnConfig {
  name: string;
  model: string;
  prompt: string;
  tools: string[];
  channels: string[];
}

export interface OnboardingJustAdded {
  tool?: string | null;
  model?: string | null;
  channel?: string | null;
}

export interface OnboardingTurnResponse {
  reply: string;
  config: OnboardingTurnConfig;
  justAdded: OnboardingJustAdded | null;
  ready: boolean;
}

export async function onboardingTurn(
  messages: OnboardingChatMessage[],
): Promise<OnboardingTurnResponse> {
  const res = await fetch('/v1/onboarding/turn', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

export interface OnboardingTransportInfo {
  transport: 'claude_code_cli' | 'anthropic_api' | 'none';
  cwd: string;
  claude_version: string | null;
}

export const getOnboardingTransport = () =>
  _getJson<OnboardingTransportInfo>('/v1/onboarding/transport');

// ─── Multi-agent (P2.0) ─────────────────────────────────────────

export interface AgentSummary {
  id: string;
  name: string;
  template: string | null;
  model: string;
  description: string;
  created_at: string;
  updated_at: string;
  onboarding_completed_at: string | null;
  is_active: boolean;
}

export interface AgentListResponse {
  agents: AgentSummary[];
  active: string | null;
}

export interface AgentCreateRequest {
  name: string;
  prompt: string;
  model?: string;
  template?: string | null;
  company?: string;
  tools?: string[];
  channels?: string[];
  activate?: boolean;
}

export const listAgents = () => _getJson<AgentListResponse>('/v1/agents');

export const getAgentById = (id: string) =>
  _getJson<AgentSummary>(`/v1/agents/${encodeURIComponent(id)}`);

export async function createAgent(body: AgentCreateRequest): Promise<AgentSummary> {
  const res = await fetch('/v1/agents', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

export interface AgentUpdateRequest {
  name?: string;
  description?: string;
  model?: string;
}

export async function updateAgent(
  id: string,
  body: AgentUpdateRequest,
): Promise<AgentSummary> {
  const res = await fetch(`/v1/agents/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

export async function activateAgent(id: string): Promise<{ active: string; agent_version: string }> {
  const res = await fetch(`/v1/agents/${encodeURIComponent(id)}/activate`, {
    method: 'POST',
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

export async function deleteAgentById(id: string): Promise<void> {
  const res = await fetch(`/v1/agents/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
  if (!res.ok && res.status !== 204) {
    const text = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${text.slice(0, 200)}`);
  }
}

// Per-agent BYOK secrets (P3.1)

export interface ProviderSecretStatus {
  set: boolean;
  source: 'per-agent' | 'global' | 'unset';
  mask: string | null;
  var: string;
}

export interface AgentSecretsResponse {
  agent_id: string;
  providers: Record<string, ProviderSecretStatus>;
}

export const getAgentSecrets = (id: string) =>
  _getJson<AgentSecretsResponse>(`/v1/agents/${encodeURIComponent(id)}/secrets`);

export async function putAgentSecrets(
  id: string,
  body: { anthropic?: string; openai?: string },
): Promise<AgentSecretsResponse> {
  const res = await fetch(`/v1/agents/${encodeURIComponent(id)}/secrets`, {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

// Per-agent improvement brain config (P3.2)

export interface ImprovementConfig {
  enabled: boolean;
  transport: 'auto' | 'claude_code_cli' | 'anthropic_api' | 'disabled';
  model: string;
  cadence_minutes: number;
  notes: string;
}

export const getAgentImprovement = (id: string) =>
  _getJson<ImprovementConfig>(`/v1/agents/${encodeURIComponent(id)}/improvement`);

export async function putAgentImprovement(
  id: string,
  body: Partial<ImprovementConfig>,
): Promise<ImprovementConfig> {
  const res = await fetch(`/v1/agents/${encodeURIComponent(id)}/improvement`, {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new ApiError(res.status, `backend ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

// Per-agent channel status (P3.3)

export interface ChannelStatus {
  connected: boolean;
  meta: Record<string, unknown>;
}

export interface AgentChannelsResponse {
  agent_id: string;
  channels: Record<string, ChannelStatus>;
}

export const getAgentChannels = (id: string) =>
  _getJson<AgentChannelsResponse>(`/v1/agents/${encodeURIComponent(id)}/channels`);
