export type TabId = 'overview' | 'cost' | 'perf' | 'deployments';

export interface KPI {
  label: string;
  value: string;
  change: string;
  isPositive: boolean;
  icon: 'dollar' | 'activity' | 'trending' | 'alert';
  subtitle?: string;
}

export interface CostByProvider {
  provider: string;
  cost: number;
  icon?: string;
  name?: string;
  value?: number;
  isOpentracy?: boolean;
}

export interface UsageByModel {
  model: string;
  name?: string;
  requests: number;
  icon?: string;
  isOpentracy?: boolean;
  cost?: number;
  latency?: number;
}

export interface Alert {
  type: 'info' | 'warning' | 'error';
  message: string;
  timestamp?: string;
}

export interface TimeSeriesData {
  date: string;
  cost: number;
  opentracyCost?: number;
  externalCost?: number;
}

export interface CostByTask {
  task: string;
  name?: string;
  cost: number;
  icon?: string;
  isOpentracy?: boolean;
}

export interface ExpensiveRequest {
  id: string;
  cost: number;
  model: string;
  icon?: string;
  promptSize: number;
  date: string;
  isOpentracy?: boolean;
}

export interface LatencyData {
  key: string;
  name?: string;
  value: number;
  icon?: string;
  isOpentracy?: boolean;
}

export interface LatencyHistogram {
  bucket: string;
  count: number;
}

export interface ErrorData {
  date: string;
  errors: number;
}

export interface ErrorTableItem {
  date: string;
  model: string;
  reason: string;
  requestId: string;
}

export interface ProviderSummary {
  totalCost: number;
  totalRequests: number;
  providers: CostByProvider[];
  models: UsageByModel[];
}

export interface OverviewData {
  kpis: KPI[];
  providers: CostByProvider[];
  models: UsageByModel[];
  alerts: Alert[];
  opentracy: ProviderSummary;
  external: ProviderSummary;
}

export interface CostAnalysisData {
  timeSeries: TimeSeriesData[];
  costByTask: CostByTask[];
  expensiveRequests: ExpensiveRequest[];
  opentracyCosts: CostByTask[];
  externalCosts: CostByTask[];
}

export interface PerformanceData {
  latencyBy: LatencyData[];
  latencyHistogram: LatencyHistogram[];
  errors: ErrorData[];
  errorsTable: ErrorTableItem[];
}

export interface ObservabilityMetrics {
  loading: boolean;
  error: string | null;
  overviewData: OverviewData | null;
  costData: CostAnalysisData | null;
  performanceData: PerformanceData | null;
  selectedDays: number;
  updateDateRange: (days: number) => void;
  refreshData: () => void;
}

/* ── Re-export shared deployment types used within this feature ── */

import type { DeploymentMetricsData } from '@/types/deploymentTypes';

export type {
  DeploymentMetricsData,
  TimeSeriesPoint,
  EKSTimeSeriesPoint,
} from '@/types/deploymentTypes';

// Deployment Metrics
export interface DeploymentKPI {
  activeDeployments: number;
  totalDeployments: number;
  avgLatency: number;
  avgSuccessRate: number;
  totalInvocations: number;
  totalCost: number;
}

export interface DeploymentWithMetrics {
  id: string;
  name: string;
  model: string;
  instance: string;
  status: string;
  metricsData: DeploymentMetricsData | null;
}
