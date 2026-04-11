export type TabId = 'efficiency' | 'model-performance' | 'training';

export interface KpiValue {
  value: number | string;
  delta_pct?: number | null;
}

// Cost Breakdown
export interface CostBreakdown {
  provider_baseline: number;
  routing_actual: number;
  routing_savings: number;
  training_investment: number;
  net_savings: number;
  roi_pct: number;
  monthly_projection: number;
}

// Distillation Job Summary
export interface DistillationJobSummary {
  job_id: string;
  name: string;
  status: string;
  teacher_model: string;
  student_model: string;
  cost_accrued: number;
  created_at: string;
  completed_at: string | null;
}

// Efficiency
export interface EfficiencyData {
  kpis: Record<string, KpiValue>;
  model_distribution: Record<string, unknown>[];
  cost_savings_trend: { date: string; saved: number; baseline: number; actual: number }[];
  model_breakdown: {
    model: string;
    requests: number;
    accuracy: number;
    avg_cost: number;
    traffic_pct: number;
  }[];
  cost_breakdown: CostBreakdown | null;
  distillation_jobs: DistillationJobSummary[];
}

// Distillation Summary (inside training)
export interface DistillationSummary {
  total_jobs: number;
  completed_jobs: number;
  running_jobs: number;
  failed_jobs: number;
  total_training_cost: number;
  latest_completed_job: {
    job_id: string;
    name: string;
    teacher_model: string;
    student_model: string;
    cost: number;
    completed_at: string;
  } | null;
}

// Model Performance
export interface ModelPerformanceData {
  kpis: Record<string, unknown>;
  cluster_accuracy: { model: string; clusters: Record<string, number> }[];
  leaderboard: {
    model: string;
    accuracy: number;
    cost: number;
    strongest_clusters: number[];
    weakest_clusters: number[];
  }[];
  teacher_student: {
    teacher: string;
    student: string;
    teacher_accuracy: number;
    teacher_cost: number;
  } | null;
}

// Training Activity
export interface AdvisorSignal {
  name: string;
  value: number;
  threshold?: number;
  triggered: boolean;
}

export interface AdvisorDecision {
  id: string;
  timestamp: string;
  recommendation: string;
  confidence: number;
  reason: string;
  source: string;
  signals: AdvisorSignal[];
}

export interface TrainingCycle {
  id: string;
  timestamp: string;
  promoted: boolean;
  reason: string;
  baseline: Record<string, number>;
  new_metrics: Record<string, number>;
}

export interface TrainingActivityData {
  kpis: Record<string, unknown>;
  training_history: { date: string; promoted: boolean; reason: string }[];
  signal_trends: { date: string; signal: string; value: number; triggered: boolean }[];
  advisor_decisions: AdvisorDecision[];
  training_cycles: TrainingCycle[];
  distillation_summary: DistillationSummary | null;
  training_runs_detail: {
    run_id: string;
    name: string;
    date: string;
    outcome: string;
    confidence: number;
    cost: number;
    duration: string;
    reason: string;
  }[];
}

// ── Routing Intelligence (real data from llm_traces) ────────────────────

export interface RoutingDecisionAPI {
  request_id: string;
  model_chosen: string;
  provider: string;
  reason: string;
  cost: number;
  latency: number;
  tokens_in: number;
  tokens_out: number;
  outcome: 'success' | 'error';
  timestamp: string;
}

export interface WinRatePoint {
  date: string;
  router: number;
  baseline: number;
}

export interface ConfidenceBucket {
  bucket: string;
  count: number;
}

export interface EfficiencyTrendPoint {
  date: string;
  score: number;
}

export interface ModelUsageItem {
  model: string;
  provider: string;
  count: number;
  percentage: number;
  avg_cost: number;
  avg_latency: number;
  error_rate: number;
}

export interface DailyVolumePoint {
  date: string;
  count: number;
  avg_latency: number;
  p95_latency: number;
  error_count: number;
  total_cost: number;
}

export interface LatencyPercentilesItem {
  model: string;
  p50: number;
  p75: number;
  p95: number;
  p99: number;
}

export interface ErrorBreakdownItem {
  category: string;
  count: number;
}

export interface RoutingIntelligenceData {
  decisions: RoutingDecisionAPI[];
  win_rate: WinRatePoint[];
  confidence_distribution: ConfidenceBucket[];
  efficiency_trend: EfficiencyTrendPoint[];
  model_usage: ModelUsageItem[];
  daily_volume: DailyVolumePoint[];
  latency_percentiles: LatencyPercentilesItem[];
  error_breakdown: ErrorBreakdownItem[];
  p95_latency: number;
  cache_hit_rate: number;
  total_tokens: number;
  avg_tokens_per_s: number;
}

// ── Advisor Config (real data) ──────────────────────────────────────────

export interface AdvisorConfigData {
  threshold: number;
  strategy: string;
  model_targets: string[];
  next_trigger_estimate: string | null;
  data_accumulation_rate: number;
  traces_since_last_training: number;
}

// Combined state
export interface RouterIntelligenceState {
  activeTab: TabId;
  selectedDays: number;
  loading: boolean;
  error: string | null;
  efficiency: EfficiencyData | null;
  models: ModelPerformanceData | null;
  training: TrainingActivityData | null;
}
