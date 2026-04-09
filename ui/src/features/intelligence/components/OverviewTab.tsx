import { useMemo } from 'react';
import { DollarSign, Target, Hash, Activity, Clock, AlertCircle } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid } from 'recharts';
import { Bar, BarChart, LabelList } from 'recharts';
import { KpiCard } from '@/components/shared/KpiCard';
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
import type { IntelligenceData } from '../hooks/useIntelligenceData';
import { ModelBreakdownTable } from './shared/ModelBreakdownTable';
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
    return <EmptyState message="No data available" onRefresh={refreshData} />;

  return <OverviewContent data={data} />;
}

function OverviewContent({ data }: { data: IntelligenceData }) {
  const { efficiency, overviewData, unifiedModelRows, selectedDays } = data;

  const kpis = efficiency?.kpis;
  const costSaved = kpis?.cost_saved?.value ?? 0;
  const quality = kpis?.quality_score?.value ?? 0;
  const totalReqs = kpis?.requests_routed?.value ?? 0;

  const externalCost = overviewData?.kpis?.find((k) => k.icon === 'dollar')?.value ?? '$0.00';
  const latencyP95 = overviewData?.kpis?.find((k) => k.icon === 'trending')?.value ?? '0ms';
  const errorRate = overviewData?.kpis?.find((k) => k.icon === 'alert')?.value ?? '0.0%';

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
      .filter((m) => !m.isLunar)
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
      .filter((m) => !m.isLunar)
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

  return (
    <div className="space-y-6">
      {/* KPIs */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-6">
        <KpiCard
          label="Cost Saved"
          value={`$${Number(costSaved).toFixed(4)}`}
          icon={DollarSign}
          subtitle="vs baseline"
        />
        <KpiCard
          label="Quality Score"
          value={`${(Number(quality) * 100).toFixed(1)}%`}
          icon={Target}
          subtitle="routing win rate"
        />
        <KpiCard label="Requests Routed" value={String(totalReqs)} icon={Hash} />
        <KpiCard label={`API Cost (${selectedDays}d)`} value={externalCost} icon={DollarSign} />
        <KpiCard label="Latency P95" value={latencyP95} icon={Clock} />
        <KpiCard label="Error Rate" value={errorRate} icon={AlertCircle} />
      </div>

      {/* Charts */}
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
                      <linearGradient key={name} id={`fill-${name}`} x1="0" y1="0" x2="0" y2="1">
                        <stop
                          offset="0%"
                          stopColor={CHART_COLORS[i % CHART_COLORS.length]}
                          stopOpacity={0.5}
                        />
                        <stop
                          offset="100%"
                          stopColor={CHART_COLORS[i % CHART_COLORS.length]}
                          stopOpacity={0.05}
                        />
                      </linearGradient>
                    ))}
                  </defs>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    vertical={false}
                    className="stroke-border/50"
                  />
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
                      fill={`url(#fill-${name})`}
                      stroke={CHART_COLORS[i % CHART_COLORS.length]}
                      strokeWidth={1.5}
                    />
                  ))}
                </AreaChart>
              </ChartContainer>
            ) : (
              <div className="flex h-56 flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border/60">
                <Activity className="size-8 text-muted-foreground/30" />
                <p className="text-sm text-muted-foreground">No distribution data yet</p>
              </div>
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
                      formatter={(value: string | number) => Number(value).toLocaleString()}
                      className="fill-foreground text-xs"
                    />
                  </Bar>
                </BarChart>
              </ChartContainer>
            ) : (
              <div className="flex h-56 flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border/60">
                <Activity className="size-8 text-muted-foreground/30" />
                <p className="text-sm text-muted-foreground">No model usage data</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Table */}
      <ModelBreakdownTable rows={unifiedModelRows} />
    </div>
  );
}
