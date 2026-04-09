export type IntelligenceTabId = 'overview' | 'costs' | 'performance' | 'models';
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
