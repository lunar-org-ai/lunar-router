import { useMemo, useState } from 'react';
import { DollarSign, Filter, BarChart3 } from 'lucide-react';
import { Bar, BarChart, LabelList, XAxis, YAxis } from 'recharts';
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
} from '@/components/ui/chart';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { KpiCard } from '@/components/shared/KpiCard';
import { formatCost } from '@/utils/formatUtils';
import type { IntelligenceData } from '../hooks/useIntelligenceData';
import { ModelBreakdownTable } from './shared/ModelBreakdownTable';
import { ModelsSkeleton, EmptyState, ErrorState } from './shared';

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
      if (next.has(p)) {
        next.delete(p);
      } else {
        next.add(p);
      }
      return next;
    });
  };

  const filteredRows = useMemo(() => {
    if (activeProviders.size === 0) return unifiedModelRows;
    return unifiedModelRows.filter((r) => activeProviders.has(r.provider));
  }, [unifiedModelRows, activeProviders]);

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

  const totalCost = useMemo(
    () => (costData?.externalCosts || []).reduce((s, c) => s + c.cost, 0),
    [costData]
  );

  const uniqueProviders = useMemo(
    () => new Set(unifiedModelRows.map((r) => r.provider)).size,
    [unifiedModelRows]
  );

  const avgAccuracy = useMemo(() => {
    const rows = unifiedModelRows.filter((r) => (r.accuracy ?? 0) > 0);
    if (rows.length === 0) return 0;
    return rows.reduce((s, r) => s + (r.accuracy ?? 0), 0) / rows.length;
  }, [unifiedModelRows]);

  return (
    <div className="space-y-6">
      {/* Summary KPIs */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <KpiCard label="Total Models" value={String(unifiedModelRows.length)} icon={BarChart3} />
        <KpiCard label="Providers" value={String(uniqueProviders)} icon={Filter} />
        <KpiCard
          label={`Total Cost (${selectedDays}d)`}
          value={formatCost(totalCost)}
          icon={DollarSign}
        />
        <KpiCard
          label="Avg Accuracy"
          value={avgAccuracy > 0 ? `${(avgAccuracy * 100).toFixed(1)}%` : '-'}
          icon={BarChart3}
        />
      </div>

      {/* Provider Filters + Chart side by side */}
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

        {/* Provider Filters */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-base">Filter by Provider</CardTitle>
            <CardDescription>
              {activeProviders.size > 0
                ? `${activeProviders.size} selected`
                : 'Showing all providers'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {PROVIDERS.map((p) => (
                <Button
                  key={p}
                  variant={activeProviders.has(p) ? 'default' : 'outline'}
                  size="sm"
                  className={`h-8 rounded-full px-4 text-xs transition-all ${
                    activeProviders.has(p) ? 'shadow-sm' : ''
                  }`}
                  onClick={() => toggleProvider(p)}
                >
                  {p}
                </Button>
              ))}
            </div>
            {activeProviders.size > 0 && (
              <Button
                variant="ghost"
                size="sm"
                className="mt-3 h-7 text-xs text-muted-foreground"
                onClick={() => setActiveProviders(new Set())}
              >
                Clear filters
              </Button>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Model Breakdown Table */}
      <ModelBreakdownTable rows={filteredRows} />
    </div>
  );
}
