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

// Separate overview data for OpenTracy vs External providers
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
  // Separated data
  opentracy: ProviderSummary;
  external: ProviderSummary;
}

export interface CostAnalysisData {
  timeSeries: TimeSeriesData[];
  costByTask: CostByTask[];
  expensiveRequests: ExpensiveRequest[];
  // Separated data
  opentracyCosts: CostByTask[];
  externalCosts: CostByTask[];
}

export interface PerformanceData {
  latencyBy: LatencyData[];
  latencyHistogram: LatencyHistogram[];
  errors: ErrorData[];
  errorsTable: ErrorTableItem[];
}
