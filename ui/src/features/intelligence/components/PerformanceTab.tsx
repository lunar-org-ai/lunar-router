import { useMemo } from 'react';
import { Activity, Clock, Brain, Users, Zap, BarChart3, TrendingUp } from 'lucide-react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { KpiCard } from '@/components/shared/KpiCard';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from '@/components/ui/chart';
import { formatCost } from '@/utils/formatUtils';
import type { IntelligenceData } from '../hooks/useIntelligenceData';
import { PerformanceSkeleton, EmptyState, ErrorState } from './shared';
import { ModelPerformanceContent } from './ModelPerformanceSection';

interface PerformanceTabProps {
  data: IntelligenceData;
}

export function PerformanceTab({ data }: PerformanceTabProps) {
  const { loading, error, training, models, refreshData } = data;

  if (loading) return <PerformanceSkeleton />;
  if (error) return <ErrorState error={error} onRetry={refreshData} />;
  if (!training && !models)
    return <EmptyState message="No performance data available" onRefresh={refreshData} />;

  return <PerformanceContent data={data} />;
}

function PerformanceContent({ data }: { data: IntelligenceData }) {
  const { training, models, costData, selectedDays } = data;

  const kpis = training?.kpis as Record<string, unknown> | undefined;
  const advisorStatus = kpis?.advisor_status as
    | { recommendation: string; confidence: number }
    | undefined;

  const avgDailyCost = useMemo(() => {
    const externalCosts = costData?.externalCosts || [];
    const totalCost = externalCosts.reduce((sum, item) => sum + item.cost, 0);
    return totalCost / selectedDays;
  }, [costData, selectedDays]);

  const signalChartData = useMemo(() => {
    if (!training?.signal_trends?.length) return [];
    const signalMap = new Map<string, Record<string, number | string>>();
    for (const s of training.signal_trends) {
      const dateKey = s.date.split('T')[0];
      if (!signalMap.has(dateKey)) signalMap.set(dateKey, { date: dateKey });
      const entry = signalMap.get(dateKey)!;
      entry[s.signal] = s.value;
    }
    return Array.from(signalMap.values());
  }, [training]);

  const signalConfig = useMemo<ChartConfig>(
    () => ({
      error_rate_increase: { label: 'Error Rate', color: 'var(--chart-1)' },
      drift_ratio: { label: 'Drift Ratio', color: 'var(--chart-3)' },
      high_severity_issues: { label: 'Issues', color: 'var(--chart-4)' },
    }),
    []
  );

  return (
    <div className="space-y-6">
      {/* Training KPIs */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <KpiCard label="Training Runs" value={String(kpis?.training_runs ?? 0)} icon={Activity} />
        <KpiCard
          label="Last Training"
          value={
            kpis?.last_training
              ? new Date(kpis.last_training as string).toLocaleDateString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  year: 'numeric',
                })
              : 'Never'
          }
          icon={Clock}
        />
        <KpiCard
          label="Advisor Status"
          value={advisorStatus?.recommendation ?? 'none'}
          icon={Brain}
          subtitle={
            advisorStatus ? `${(advisorStatus.confidence * 100).toFixed(0)}% confidence` : undefined
          }
        />
        <KpiCard label="Models Updated" value={String(kpis?.models_updated ?? 0)} icon={Users} />
        <KpiCard label="Avg Daily Cost" value={formatCost(avgDailyCost)} icon={Zap} />
      </div>

      {/* Training Charts */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Training History</CardTitle>
            <CardDescription>Run outcomes over time</CardDescription>
          </CardHeader>
          <CardContent>
            {training?.training_history && training.training_history.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart
                  data={training.training_history.map((h, i) => ({
                    ...h,
                    idx: i + 1,
                    value: 1,
                  }))}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    vertical={false}
                    className="stroke-border/50"
                  />
                  <XAxis
                    dataKey="idx"
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    label={{
                      value: 'Run #',
                      position: 'insideBottom',
                      offset: -5,
                      style: { fontSize: 11, fill: 'var(--color-muted-foreground)' },
                    }}
                  />
                  <YAxis hide />
                  <ChartTooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.[0]) return null;
                      const d = payload[0].payload as {
                        promoted: boolean;
                        reason: string;
                        date: string;
                      };
                      return (
                        <div className="rounded-md border bg-popover px-3 py-2 text-sm shadow-md">
                          <p className="font-medium">{d.promoted ? 'Promoted' : 'Rejected'}</p>
                          <p className="text-muted-foreground">{d.reason}</p>
                          <p className="mt-1 text-xs text-muted-foreground">{d.date}</p>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {training.training_history.map((h, i) => (
                      <Cell key={i} fill={h.promoted ? 'var(--chart-2)' : 'var(--chart-1)'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-56 flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border/60">
                <div className="flex size-12 items-center justify-center rounded-full bg-muted/50">
                  <BarChart3 className="size-5 text-muted-foreground/50" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium text-muted-foreground">No training runs yet</p>
                  <p className="mt-0.5 text-xs text-muted-foreground/70">
                    The advisor will trigger training when needed.
                  </p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Signal Trends</CardTitle>
            <CardDescription>Drift and issue signals over time</CardDescription>
          </CardHeader>
          <CardContent>
            {signalChartData.length > 0 ? (
              <ChartContainer config={signalConfig} className="h-62.5 w-full">
                <LineChart data={signalChartData}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    vertical={false}
                    className="stroke-border/50"
                  />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={35} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <ChartLegend content={<ChartLegendContent />} />
                  <Line
                    type="monotone"
                    dataKey="error_rate_increase"
                    stroke="var(--chart-1)"
                    strokeWidth={1.5}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="drift_ratio"
                    stroke="var(--chart-3)"
                    strokeWidth={1.5}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="high_severity_issues"
                    stroke="var(--chart-4)"
                    strokeWidth={1.5}
                    dot={false}
                  />
                </LineChart>
              </ChartContainer>
            ) : (
              <div className="flex h-56 flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border/60">
                <div className="flex size-12 items-center justify-center rounded-full bg-muted/50">
                  <TrendingUp className="size-5 text-muted-foreground/50" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium text-muted-foreground">No signal data yet</p>
                  <p className="mt-0.5 text-xs text-muted-foreground/70">
                    Signals will appear after routing begins.
                  </p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Model Performance */}
      {models && <ModelPerformanceContent data={models} />}
    </div>
  );
}
