import { useState, useEffect, useCallback } from 'react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import {
  Activity,
  CheckCircle,
  Clock,
  Hash,
  RefreshCw,
  TrendingUp,
  XCircle,
  Zap,
} from 'lucide-react';

import { API_BASE_URL } from '../../config/api';

interface DeploymentMetricsChartsProps {
  deploymentId: string;
  isVisible: boolean;
}

interface TimeSeriesPoint {
  timestamp: string;
  value: number;
}

interface MetricsData {
  deployment_id: string;
  latest: {
    cpu_utilization?: number;
    memory_utilization?: number;
    gpu_utilization?: number;
    gpu_memory_utilization?: number;
    model_latency_ms: number;
    invocations: number;
    timestamp: string;
  };
  inference_stats: {
    total_inferences: number;
    successful: number;
    failed: number;
    success_rate: number;
    avg_latency_ms: number;
    total_tokens: number;
    total_cost_usd: number;
  };
  time_series: {
    invocations?: TimeSeriesPoint[];
    model_latency?: TimeSeriesPoint[];
    p95_latency?: TimeSeriesPoint[];
    tokens?: TimeSeriesPoint[];
    errors?: TimeSeriesPoint[];
    throughput?: TimeSeriesPoint[];
  };
}

type ChartKey = 'latency' | 'invocations' | 'tokens' | 'throughput';

export function DeploymentMetricsCharts({ deploymentId, isVisible }: DeploymentMetricsChartsProps) {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeChart, setActiveChart] = useState<ChartKey>('latency');

  const fetchMetrics = useCallback(async () => {
    if (!deploymentId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${API_BASE_URL}/v1/deployments/${deploymentId}/metrics?minutes=60&period=60`
      );
      if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.status}`);
      const data = (await res.json()) as MetricsData;
      setMetrics(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load metrics');
    } finally {
      setLoading(false);
    }
  }, [deploymentId]);

  useEffect(() => {
    if (isVisible && deploymentId) {
      fetchMetrics();
      const interval = setInterval(fetchMetrics, 10_000);
      return () => clearInterval(interval);
    }
  }, [isVisible, deploymentId, fetchMetrics]);

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getChartConfig = () => {
    const ts = metrics?.time_series ?? {};
    switch (activeChart) {
      case 'latency':
        return {
          data: ts.model_latency ?? [],
          color: '#909090',
          label: 'Latency (avg)',
          unit: 'ms',
          gradient: 'url(#latencyGradient)',
        };
      case 'invocations':
        return {
          data: ts.invocations ?? [],
          color: '#c0c0c0',
          label: 'Requests',
          unit: '',
          gradient: 'url(#invocationsGradient)',
        };
      case 'tokens':
        return {
          data: ts.tokens ?? [],
          color: '#a0a0a0',
          label: 'Tokens',
          unit: '',
          gradient: 'url(#tokensGradient)',
        };
      case 'throughput':
        return {
          data: ts.throughput ?? [],
          color: '#b0b0b0',
          label: 'Throughput',
          unit: ' tok/s',
          gradient: 'url(#throughputGradient)',
        };
    }
  };

  if (!isVisible) return null;

  if (loading && !metrics) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="flex items-center gap-3 text-muted-foreground">
          <RefreshCw className="w-5 h-5 animate-spin" />
          <span>Loading metrics...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4 text-center">
        <p className="text-destructive text-sm">{error}</p>
        <button
          onClick={fetchMetrics}
          className="mt-2 text-sm text-destructive hover:underline"
        >
          Try again
        </button>
      </div>
    );
  }

  const stats = metrics?.inference_stats;
  const total = stats?.total_inferences ?? 0;
  const succRate = stats?.success_rate ?? 100;

  const chartConfig = getChartConfig();
  const chartData = (chartConfig.data ?? []).map((p) => ({
    time: formatTimestamp(p.timestamp),
    value: p.value,
  }));

  return (
    <div className="space-y-4">
      {/* KPI cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard
          icon={<Hash className="w-4 h-4" />}
          label="Total requests"
          value={total.toLocaleString()}
          onClick={() => setActiveChart('invocations')}
          active={activeChart === 'invocations'}
        />
        <StatCard
          icon={
            succRate >= 95 ? (
              <CheckCircle className="w-4 h-4" />
            ) : (
              <XCircle className="w-4 h-4" />
            )
          }
          label="Success rate"
          value={`${succRate.toFixed(1)}%`}
          accent={succRate >= 95 ? 'good' : 'bad'}
          onClick={undefined}
          active={false}
        />
        <StatCard
          icon={<Clock className="w-4 h-4" />}
          label="Avg latency"
          value={`${(stats?.avg_latency_ms ?? 0).toFixed(0)}ms`}
          onClick={() => setActiveChart('latency')}
          active={activeChart === 'latency'}
        />
        <StatCard
          icon={<Zap className="w-4 h-4" />}
          label="Total tokens"
          value={(stats?.total_tokens ?? 0).toLocaleString()}
          onClick={() => setActiveChart('tokens')}
          active={activeChart === 'tokens'}
        />
      </div>

      {/* Failed indicator */}
      {(stats?.failed ?? 0) > 0 && (
        <div className="flex items-center gap-2 rounded-md border border-destructive/20 bg-destructive/5 px-3 py-2 text-xs text-destructive">
          <XCircle className="w-3.5 h-3.5" />
          {stats?.failed} failed request{(stats?.failed ?? 0) === 1 ? '' : 's'} in the last hour
        </div>
      )}

      {/* Time-series chart */}
      <div className="bg-muted/40 rounded-lg border border-border p-4">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-sm font-semibold flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-muted-foreground" />
            {chartConfig.label} over time
          </h4>
          <div className="flex gap-1">
            <ChartTab label="Latency" active={activeChart === 'latency'} onClick={() => setActiveChart('latency')} />
            <ChartTab label="Requests" active={activeChart === 'invocations'} onClick={() => setActiveChart('invocations')} />
            <ChartTab label="Tokens" active={activeChart === 'tokens'} onClick={() => setActiveChart('tokens')} />
            <ChartTab label="Tok/s" active={activeChart === 'throughput'} onClick={() => setActiveChart('throughput')} />
          </div>
        </div>

        {chartData.length > 0 ? (
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <defs>
                  <linearGradient id="latencyGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#909090" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#909090" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="invocationsGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#c0c0c0" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#c0c0c0" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="tokensGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#a0a0a0" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#a0a0a0" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="throughputGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#b0b0b0" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#b0b0b0" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#808080' }} tickLine={false} axisLine={{ stroke: '#333' }} />
                <YAxis tick={{ fontSize: 10, fill: '#808080' }} tickLine={false} axisLine={{ stroke: '#333' }} tickFormatter={(v) => `${v}${chartConfig.unit}`} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                    fontSize: '12px',
                    color: '#ededed',
                  }}
                  formatter={(value) => [`${Number(value ?? 0).toFixed(2)}${chartConfig.unit}`, chartConfig.label]}
                />
                <Area type="monotone" dataKey="value" stroke={chartConfig.color} strokeWidth={2} fill={chartConfig.gradient} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-48 flex flex-col items-center justify-center text-muted-foreground text-sm gap-1">
            <Activity className="w-6 h-6 opacity-40" />
            <span>No data yet — send a request to populate metrics</span>
          </div>
        )}
      </div>

      <div className="flex justify-between items-center text-xs text-muted-foreground">
        <span>Auto-refreshing every 10s</span>
        <button
          onClick={fetchMetrics}
          disabled={loading}
          className="flex items-center gap-1.5 hover:text-foreground transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>
    </div>
  );
}

interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  onClick?: () => void;
  active: boolean;
  accent?: 'good' | 'bad';
}

function StatCard({ icon, label, value, onClick, active, accent }: StatCardProps) {
  const accentClass =
    accent === 'good'
      ? 'text-green-500'
      : accent === 'bad'
        ? 'text-destructive'
        : 'text-foreground';
  return (
    <button
      onClick={onClick}
      disabled={!onClick}
      className={[
        'rounded-lg p-3 border text-left w-full transition-all duration-200',
        active
          ? 'bg-muted border-border ring-2 ring-accent/40 shadow-sm'
          : 'bg-muted/40 border-border hover:bg-muted',
        onClick ? 'cursor-pointer' : 'cursor-default',
      ].join(' ')}
    >
      <div className="mb-1 text-muted-foreground">{icon}</div>
      <div className={`text-lg font-bold ${accentClass}`}>{value}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
    </button>
  );
}

interface ChartTabProps {
  label: string;
  active: boolean;
  onClick: () => void;
}

function ChartTab({ label, active, onClick }: ChartTabProps) {
  return (
    <button
      onClick={onClick}
      className={[
        'px-2 py-1 text-xs rounded-md transition-colors',
        active
          ? 'bg-accent/15 text-accent font-medium border border-accent/30'
          : 'text-muted-foreground hover:bg-muted hover:text-foreground',
      ].join(' ')}
    >
      {label}
    </button>
  );
}
