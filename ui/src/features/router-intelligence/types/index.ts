export type TabId = 'efficiency' | 'model-performance' | 'training';

export interface KpiValue {
  value: number | string;
  delta_pct?: number | null;
}

// Efficiency
export interface EfficiencyData {
  kpis: Record<string, KpiValue>;
  model_distribution: Record<string, unknown>[];
  cost_savings_trend: { date: string; saved: number; baseline: number; actual: number }[];
  model_breakdown: { model: string; requests: number; accuracy: number; avg_cost: number; traffic_pct: number }[];
}

// Model Performance
export interface ModelPerformanceData {
  kpis: Record<string, unknown>;
  cluster_accuracy: { model: string; clusters: Record<string, number> }[];
  leaderboard: { model: string; accuracy: number; cost: number; strongest_clusters: number[]; weakest_clusters: number[] }[];
  teacher_student: { teacher: string; student: string; teacher_accuracy: number; teacher_cost: number } | null;
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
