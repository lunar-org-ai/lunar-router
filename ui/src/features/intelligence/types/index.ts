export type IntelligenceTabId = 'overview' | 'costs' | 'distillation' | 'models' | 'routing';

export type Period = '7d' | '14d' | '30d';

export const PERIOD_TO_DAYS: Record<Period, number> = {
  '7d': 7,
  '14d': 14,
  '30d': 30,
};

export interface UnifiedModelRow {
  model: string;
  provider: string;
  requests: number;
  trafficPct: number;
  accuracy: number | null;
  avgCost: number;
  totalCost: number;
}

export interface RoutingDecision {
  requestId: string;
  cluster: number;
  modelChosen: string;
  reason: string;
  cost: number;
  latency: number;
  outcome: 'success' | 'error';
  timestamp: string;
}

export interface TrainingRunDetail {
  runId: string;
  name: string;
  date: string;
  outcome: 'promoted' | 'rejected';
  confidence: number;
  cost: number;
  duration: string;
  reason: string;
  teacherModel: string;
  studentModel: string;
  qualityScore: number;
  status: string;
}

export interface ModelCapability {
  model: string;
  provider: string;
  contextWindow: number;
  supportsVision: boolean;
  supportsFunctionCalling: boolean;
  costTier: 'low' | 'medium' | 'high' | 'premium';
}
