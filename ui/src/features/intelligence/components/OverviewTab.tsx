import { useMemo } from 'react';
import { Hash, DollarSign, Target, Clock, AlertCircle, Activity } from 'lucide-react';
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, LabelList } from 'recharts';
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from '@/components/ui/chart';
import { Badge } from '@/components/ui/badge';
import { formatCost } from '@/utils/formatUtils';
import { formatNumber, formatPercent } from '../utils/intelligenceHelpers';
import type { IntelligenceData } from '../hooks/useIntelligenceData';
import { MetricCard } from './shared/MetricCard';
import { BaselineComparison } from './shared/BaselineComparison';
import { OverviewSkeleton, EmptyState, ErrorState } from './shared';

const CHART_COLORS = ['var(--chart-1)', 'var(--chart-2)', 'var(--chart-3)'];

interface OverviewTabProps {
  data: IntelligenceData;
}

export function OverviewTab({ data }: OverviewTabProps) {
  const { loading, error, efficiency, overviewData, refreshData } = data;

  if (loading) return <OverviewSkeleton />;
  if (error) return <ErrorState error={error} onRetry={refreshData} />;
  if (!efficiency && !overviewData)
    return (
      <EmptyState
        message="No data available yet. Start routing requests to see metrics."
        onRefresh={refreshData}
      />
    );

  return <OverviewContent data={data} />;
}

function OverviewContent({ data }: { data: IntelligenceData }) {
  const {
    efficiency,
    overviewData,
    performanceData,
    routingIntelligence,
    unifiedModelRows,
    selectedDays,
  } = data;

  const kpis = efficiency?.kpis;
  const costSaved = (kpis?.cost_saved as { value: number } | undefined)?.value ?? 0;
  const quality = (kpis?.quality_score as { value: number } | undefined)?.value ?? 0;
  const totalReqs = (kpis?.requests_routed as { value: number } | undefined)?.value ?? 0;

  const externalCost = overviewData?.kpis?.find((k) => k.icon === 'dollar')?.value ?? '$0.00';
  const latencyP95 = overviewData?.kpis?.find((k) => k.icon === 'trending')?.value ?? '0ms';
  const errorRate = overviewData?.kpis?.find((k) => k.icon === 'alert')?.value ?? '0.0%';

  const cb = efficiency?.cost_breakdown;

  const topModels = useMemo(() => unifiedModelRows.slice(0, 5), [unifiedModelRows]);

  const modelNames = useMemo(() => {
    if (!efficiency?.model_distribution?.length) return [];
    return Array.from(
      new Set(
        efficiency.model_distribution.flatMap((d) => Object.keys(d).filter((k) => k !== 'date'))
      )
    );
  }, [efficiency]);

  const areaConfig = useMemo<ChartConfig>(() => {
    const cfg: ChartConfig = {};
    modelNames.forEach((name, i) => {
      cfg[name] = { label: name, color: CHART_COLORS[i % CHART_COLORS.length] };
    });
    return cfg;
  }, [modelNames]);

  const barData = useMemo(() => {
    if (!overviewData?.models) return [];
    return [...overviewData.models]
      .filter((m) => !m.isOpentracy)
      .sort((a, b) => b.requests - a.requests)
      .slice(0, 8)
      .map((m, i) => ({
        model: `model-${i}`,
        requests: m.requests,
        fill: 'var(--chart-1)',
      }));
  }, [overviewData]);

  const barConfig = useMemo<ChartConfig>(() => {
    if (!overviewData?.models) return { requests: { label: 'Requests' } };
    const cfg: ChartConfig = { requests: { label: 'Requests' } };
    [...overviewData.models]
      .filter((m) => !m.isOpentracy)
      .sort((a, b) => b.requests - a.requests)
      .slice(0, 8)
      .forEach((m, i) => {
        cfg[`model-${i}`] = {
          label: m.model || m.name || 'Unknown',
          color: 'var(--chart-1)',
        };
      });
    return cfg;
  }, [overviewData]);

  const latencyHistogram = performanceData?.latencyHistogram ?? [];
  const latencyConfig = useMemo<ChartConfig>(
    () => ({ count: { label: 'Requests', color: 'var(--chart-2)' } }),
    []
  );

  // Real efficiency trend from /v1/intelligence/routing
  const efficiencyTrend = routingIntelligence?.efficiency_trend ?? [];
  const efficiencyTrendConfig = useMemo<ChartConfig>(
    () => ({ score: { label: 'Efficiency', color: 'var(--chart-1)' } }),
    []
  );

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-6">
        <MetricCard
          label="Requests Routed"
          value={formatNumber(Number(totalReqs))}
          icon={Hash}
          tooltip="Total number of requests processed by the router in this period"
        />
        <MetricCard
          label="Cost Saved"
          value={`$${Number(costSaved).toFixed(4)}`}
          icon={DollarSign}
          tooltip="Cost savings compared to always routing to the most expensive model"
        />
        <MetricCard
          label="Quality Score"
          value={formatPercent(Number(quality))}
          icon={Target}
          tooltip="Percentage of requests where the router chose the optimal model"
        />
        <MetricCard
          label={`API Cost (${selectedDays}d)`}
          value={externalCost}
          icon={DollarSign}
          tooltip="Total cost of external API calls to LLM providers"
        />
        <MetricCard
          label="Latency P95"
          value={latencyP95}
          icon={Clock}
          tooltip="95th percentile response time across all requests"
        />
        <MetricCard
          label="Error Rate"
          value={errorRate}
          icon={AlertCircle}
          tooltip="Percentage of requests that returned an error"
        />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle className="text-base">Model Distribution Over Time</CardTitle>
            <CardDescription>Stacked model usage &middot; {selectedDays}d</CardDescription>
          </CardHeader>
          <CardContent>
            {efficiency?.model_distribution && efficiency.model_distribution.length > 0 ? (
              <ChartContainer config={areaConfig} className="h-72 w-full">
                <AreaChart data={efficiency.model_distribution}>
                  <defs>
                    {modelNames.map((name, i) => (
                      <linearGradient key={name} id={`fill-ov-${name}`} x1="0" y1="0" x2="0" y2="1">
                        <stop
                          offset="0%"
                          stopColor={CHART_COLORS[i % CHART_COLORS.length]}
                          stopOpacity={0.4}
                        />
                        <stop
                          offset="100%"
                          stopColor={CHART_COLORS[i % CHART_COLORS.length]}
                          stopOpacity={0.05}
                        />
                      </linearGradient>
                    ))}
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-border" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={40} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <ChartLegend content={<ChartLegendContent />} />
                  {modelNames.map((name, i) => (
                    <Area
                      key={name}
                      type="monotone"
                      dataKey={name}
                      stackId="1"
                      fill={`url(#fill-ov-${name})`}
                      stroke={CHART_COLORS[i % CHART_COLORS.length]}
                      strokeWidth={1.5}
                    />
                  ))}
                </AreaChart>
              </ChartContainer>
            ) : (
              <ChartEmpty icon={Activity} message="No distribution data yet" />
            )}
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-base">Usage by Model</CardTitle>
            <CardDescription>Total requests &middot; {selectedDays}d</CardDescription>
            <CardAction>
              <Badge variant="secondary" className="tabular-nums text-xs">
                {barData.reduce((s, m) => s + m.requests, 0).toLocaleString()} total
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent>
            {barData.length > 0 ? (
              <ChartContainer config={barConfig}>
                <BarChart accessibilityLayer data={barData} layout="vertical" margin={{ left: 0 }}>
                  <YAxis
                    dataKey="model"
                    type="category"
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(value) =>
                      barConfig[value as keyof typeof barConfig]?.label as string
                    }
                    width={110}
                    tick={{ fontSize: 11 }}
                  />
                  <XAxis dataKey="requests" type="number" hide />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                  <Bar dataKey="requests" radius={5}>
                    <LabelList
                      dataKey="requests"
                      position="right"
                      formatter={(value: string | number) => formatNumber(Number(value))}
                      className="fill-foreground text-xs"
                    />
                  </Bar>
                </BarChart>
              </ChartContainer>
            ) : (
              <ChartEmpty icon={Activity} message="No model usage data" />
            )}
          </CardContent>
        </Card>
      </div>

      {cb && (cb.provider_baseline > 0 || cb.routing_actual > 0) && (
        <BaselineComparison cb={cb} variant="full" />
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Latency Distribution</CardTitle>
            <CardDescription>Request count by response time bucket</CardDescription>
          </CardHeader>
          <CardContent>
            {latencyHistogram.length > 0 ? (
              <ChartContainer config={latencyConfig} className="h-64 w-full">
                <BarChart data={latencyHistogram}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-border" />
                  <XAxis
                    dataKey="bucket"
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={40} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar dataKey="count" fill="var(--chart-2)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ChartContainer>
            ) : (
              <ChartEmpty icon={Clock} message="No latency data yet" />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Router Efficiency Trend</CardTitle>
            <CardDescription>
              Routing optimization score over time &middot; {selectedDays}d
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer config={efficiencyTrendConfig} className="h-64 w-full">
              <AreaChart data={efficiencyTrend}>
                <defs>
                  <linearGradient id="fillEffTrend" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="var(--chart-1)" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-border" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => {
                    const d = new Date(v);
                    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                  }}
                />
                <YAxis
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  width={40}
                  tickFormatter={(v) => formatPercent(Number(v))}
                  domain={[0, 1]}
                />
                <ChartTooltip
                  content={
                    <ChartTooltipContent formatter={(v: number) => formatPercent(Number(v))} />
                  }
                />
                <Area
                  type="monotone"
                  dataKey="score"
                  fill="url(#fillEffTrend)"
                  stroke="var(--chart-1)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ChartContainer>
          </CardContent>
        </Card>
      </div>

      {topModels.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Top 5 Most-Used Models</CardTitle>
            <CardDescription>By request volume &middot; {selectedDays}d</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
              {topModels.map((m) => (
                <div key={m.model} className="space-y-1 rounded-lg border p-4">
                  <p className="truncate text-sm font-medium">{m.model}</p>
                  <p className="text-xs text-muted-foreground">{m.provider}</p>
                  <div className="flex items-baseline gap-2 pt-1">
                    <span className="text-lg font-semibold tabular-nums">
                      {formatNumber(m.requests)}
                    </span>
                    <span className="text-xs text-muted-foreground">requests</span>
                  </div>
                  <p className="text-xs tabular-nums text-muted-foreground">
                    {m.trafficPct.toFixed(1)}% traffic &middot; {formatCost(m.totalCost)}
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function ChartEmpty({ icon: Icon, message }: { icon: React.ElementType; message: string }) {
  return (
    <div className="flex h-56 flex-col items-center justify-center gap-3">
      <div className="flex size-12 items-center justify-center rounded-full bg-muted">
        <Icon className="size-5 text-muted-foreground" />
      </div>
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  );
}
