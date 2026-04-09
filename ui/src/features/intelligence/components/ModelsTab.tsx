import { useMemo, useState } from 'react';
import { DollarSign, Filter, BarChart3, Zap, TrendingUp, Activity } from 'lucide-react';
import { Bar, BarChart, LabelList, Pie, PieChart, XAxis, YAxis } from 'recharts';
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
import { Button } from '@/components/ui/button';
import { KpiCard } from '@/components/shared/KpiCard';
import { formatCost } from '@/utils/formatUtils';
import type { IntelligenceData } from '../hooks/useIntelligenceData';
import { ModelBreakdownTable } from './shared/ModelBreakdownTable';
import { ModelsSkeleton, EmptyState, ErrorState } from './shared';

const CHART_COLORS = ['var(--chart-1)', 'var(--chart-2)', 'var(--chart-3)'];

const PROVIDERS = ['OpenAI', 'Groq', 'Anthropic', 'Meta'] as const;

interface ModelsTabProps {
  data: IntelligenceData;
}

export function ModelsTab({ data }: ModelsTabProps) {
  const { loading, error, unifiedModelRows, refreshData } = data;

  if (loading) return <ModelsSkeleton />;
  if (error) return <ErrorState error={error} onRetry={refreshData} />;
  if (unifiedModelRows.length === 0)
    return <EmptyState message="No model data available" onRefresh={refreshData} />;

  return <ModelsContent data={data} />;
}

function ModelsContent({ data }: { data: IntelligenceData }) {
  const { costData, unifiedModelRows, selectedDays } = data;
  const [activeProviders, setActiveProviders] = useState<Set<string>>(new Set());

  const toggleProvider = (p: string) => {
    setActiveProviders((prev) => {
      const next = new Set(prev);
      if (next.has(p)) next.delete(p);
      else next.add(p);
      return next;
    });
  };

  const filteredRows = useMemo(() => {
    if (activeProviders.size === 0) return unifiedModelRows;
    return unifiedModelRows.filter((r) => activeProviders.has(r.provider));
  }, [unifiedModelRows, activeProviders]);

  // Cost by model bar data
  const costByModelData = useMemo(
    () =>
      (costData?.externalCosts || []).slice(0, 6).map((item, i) => ({
        task: `task-${i}`,
        cost: item.cost,
        fill: 'var(--chart-2)',
      })),
    [costData]
  );

  const maxBarCost = useMemo(
    () => Math.max(...costByModelData.map((d) => d.cost), 0),
    [costByModelData]
  );
  const barDomainMax = useMemo(() => (maxBarCost > 0 ? maxBarCost * 1.15 : 1), [maxBarCost]);

  const barConfig = useMemo<ChartConfig>(() => {
    const cfg: ChartConfig = { cost: { label: 'Cost' } };
    (costData?.externalCosts || []).slice(0, 6).forEach((item, i) => {
      cfg[`task-${i}`] = { label: item.task, color: 'var(--chart-2)' };
    });
    return cfg;
  }, [costData]);

  // Aggregated stats
  const totalCost = useMemo(
    () => (costData?.externalCosts || []).reduce((s, c) => s + c.cost, 0),
    [costData]
  );
  const totalRequests = useMemo(
    () => unifiedModelRows.reduce((s, r) => s + r.requests, 0),
    [unifiedModelRows]
  );
  const uniqueProviders = useMemo(
    () => new Set(unifiedModelRows.map((r) => r.provider)).size,
    [unifiedModelRows]
  );
  const avgCostPerReq = useMemo(
    () => (totalRequests > 0 ? totalCost / totalRequests : 0),
    [totalCost, totalRequests]
  );
  const topModel = useMemo(
    () => (unifiedModelRows.length > 0 ? unifiedModelRows[0] : null),
    [unifiedModelRows]
  );

  // Provider breakdown for pie chart
  const providerBreakdown = useMemo(() => {
    const map = new Map<string, { requests: number; cost: number }>();
    for (const row of unifiedModelRows) {
      const existing = map.get(row.provider) ?? { requests: 0, cost: 0 };
      existing.requests += row.requests;
      existing.cost += row.totalCost;
      map.set(row.provider, existing);
    }
    return Array.from(map.entries())
      .sort((a, b) => b[1].requests - a[1].requests)
      .map(([provider, stats], i) => ({
        provider: `prov-${i}`,
        providerName: provider,
        requests: stats.requests,
        cost: stats.cost,
        fill: `var(--color-prov-${i})`,
      }));
  }, [unifiedModelRows]);

  const pieConfig = useMemo<ChartConfig>(() => {
    const cfg: ChartConfig = { requests: { label: 'Requests' } };
    providerBreakdown.forEach((p, i) => {
      cfg[`prov-${i}`] = { label: p.providerName, color: CHART_COLORS[i % CHART_COLORS.length] };
    });
    return cfg;
  }, [providerBreakdown]);

  return (
    <div className="space-y-6">
      {/* Summary KPIs */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-5">
        <KpiCard label="Total Models" value={String(unifiedModelRows.length)} icon={BarChart3} />
        <KpiCard label="Providers" value={String(uniqueProviders)} icon={Filter} />
        <KpiCard
          label={`Total Requests (${selectedDays}d)`}
          value={totalRequests.toLocaleString()}
          icon={Activity}
        />
        <KpiCard
          label={`Total Cost (${selectedDays}d)`}
          value={formatCost(totalCost)}
          icon={DollarSign}
        />
        <KpiCard label="Avg Cost / Request" value={formatCost(avgCostPerReq)} icon={Zap} />
      </div>

      {/* Top Model highlight */}
      {topModel && (
        <Card className="border-chart-2/20 bg-chart-2/5">
          <CardContent className="flex flex-wrap items-center gap-6 py-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="size-4 text-chart-2" />
              <span className="text-sm font-medium text-muted-foreground">Most Used Model</span>
            </div>
            <span className="text-sm font-semibold">{topModel.model}</span>
            <Badge variant="outline" className="text-xs">
              {topModel.provider}
            </Badge>
            <span className="text-sm tabular-nums text-muted-foreground">
              {topModel.requests.toLocaleString()} requests
            </span>
            <span className="text-sm tabular-nums text-muted-foreground">
              {topModel.trafficPct.toFixed(1)}% traffic
            </span>
            <span className="text-sm tabular-nums text-muted-foreground">
              {formatCost(topModel.totalCost)} cost
            </span>
          </CardContent>
        </Card>
      )}

      {/* Charts row: Cost by Model + Provider Distribution */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle className="text-base">Cost by Model</CardTitle>
            <CardDescription>Top external API costs &middot; {selectedDays}d</CardDescription>
            <CardAction>
              <Badge variant="secondary" className="tabular-nums text-xs">
                {formatCost(totalCost)} total
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent>
            {costByModelData.length > 0 ? (
              <ChartContainer config={barConfig} className="h-52 w-full">
                <BarChart
                  accessibilityLayer
                  data={costByModelData}
                  layout="vertical"
                  margin={{ left: 0 }}
                >
                  <YAxis
                    dataKey="task"
                    type="category"
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(value) =>
                      barConfig[value as keyof typeof barConfig]?.label as string
                    }
                    width={110}
                    tick={{ fontSize: 11 }}
                  />
                  <XAxis dataKey="cost" type="number" hide domain={[0, barDomainMax]} />
                  <ChartTooltip
                    cursor={false}
                    content={
                      <ChartTooltipContent
                        formatter={(value: number) => [formatCost(Number(value)), 'Cost']}
                      />
                    }
                  />
                  <Bar dataKey="cost" radius={5} fill="var(--chart-2)">
                    <LabelList
                      dataKey="cost"
                      position="right"
                      offset={8}
                      className="fill-foreground"
                      fontSize={11}
                      formatter={(value: string | number) => formatCost(Number(value))}
                    />
                  </Bar>
                </BarChart>
              </ChartContainer>
            ) : (
              <div className="flex h-40 flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border/60">
                <DollarSign className="size-8 text-muted-foreground/30" />
                <p className="text-sm text-muted-foreground">No cost data</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Provider distribution donut */}
        <Card className="flex flex-col lg:col-span-2">
          <CardHeader className="items-center pb-0">
            <CardTitle className="text-base">Request Distribution</CardTitle>
            <CardDescription>By provider &middot; {selectedDays}d</CardDescription>
            <CardAction>
              <Badge variant="secondary" className="tabular-nums text-xs">
                {totalRequests.toLocaleString()} total
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent className="flex-1 p-0">
            {providerBreakdown.length > 0 ? (
              <ChartContainer config={pieConfig} className="mx-auto aspect-square max-h-56">
                <PieChart>
                  <ChartTooltip
                    cursor={false}
                    content={
                      <ChartTooltipContent
                        formatter={(value: number) => [
                          `${Number(value).toLocaleString()} requests`,
                          '',
                        ]}
                      />
                    }
                  />
                  <Pie
                    data={providerBreakdown}
                    dataKey="requests"
                    nameKey="provider"
                    innerRadius={45}
                    strokeWidth={2}
                  />
                  <ChartLegend
                    content={<ChartLegendContent nameKey="provider" />}
                    className="-translate-y-2 flex-wrap gap-2 *:basis-1/3 *:justify-center"
                  />
                </PieChart>
              </ChartContainer>
            ) : (
              <div className="flex h-40 flex-col items-center justify-center gap-2">
                <Activity className="size-8 text-muted-foreground/30" />
                <p className="text-sm text-muted-foreground">No provider data</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Provider filter + table */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted-foreground">Filter:</span>
        {PROVIDERS.map((p) => (
          <Button
            key={p}
            variant={activeProviders.has(p) ? 'default' : 'outline'}
            size="sm"
            className={`h-7 rounded-full px-3 text-xs transition-all ${
              activeProviders.has(p) ? 'shadow-sm' : ''
            }`}
            onClick={() => toggleProvider(p)}
          >
            {p}
          </Button>
        ))}
        {activeProviders.size > 0 && (
          <Button
            variant="ghost"
            size="sm"
            className="h-7 text-xs text-muted-foreground"
            onClick={() => setActiveProviders(new Set())}
          >
            Clear
          </Button>
        )}
      </div>

      <ModelBreakdownTable rows={filteredRows} />
    </div>
  );
}
