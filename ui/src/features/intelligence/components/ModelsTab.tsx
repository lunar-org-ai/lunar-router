import { useMemo, useState } from 'react';
import { DollarSign, Filter, BarChart3, Zap, Activity, Eye, Wrench, Download } from 'lucide-react';
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
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { formatCost } from '@/utils/formatUtils';
import { formatNumber, costTierLabel, exportTableToCsv } from '../utils/intelligenceHelpers';
import type { IntelligenceData } from '../hooks/useIntelligenceData';
import { MetricCard } from './shared/MetricCard';
import { ModelBreakdownTable } from './shared/ModelBreakdownTable';
import { ModelPerformanceContent } from './ModelPerformanceSection';
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
    return <EmptyState message="No model data available yet." onRefresh={refreshData} />;

  return <ModelsContent data={data} />;
}

function ModelsContent({ data }: { data: IntelligenceData }) {
  const { costData, unifiedModelRows, modelCapabilities, selectedDays } = data;
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

  const providerBreakdown = useMemo(() => {
    const map = new Map<string, { requests: number; cost: number; errors: number }>();
    for (const row of unifiedModelRows) {
      const existing = map.get(row.provider) ?? { requests: 0, cost: 0, errors: 0 };
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

  const costEfficiency = useMemo(() => {
    return unifiedModelRows
      .filter((r) => r.totalCost > 0)
      .map((r) => ({
        model: r.model,
        requestsPerDollar: r.totalCost > 0 ? r.requests / r.totalCost : 0,
      }))
      .sort((a, b) => b.requestsPerDollar - a.requestsPerDollar)
      .slice(0, 8);
  }, [unifiedModelRows]);

  const handleExportCapabilities = () => {
    const headers = [
      'Model',
      'Provider',
      'Context Window',
      'Vision',
      'Function Calling',
      'Cost Tier',
    ];
    const rows = modelCapabilities.map((c) => [
      c.model,
      c.provider,
      formatNumber(c.contextWindow),
      c.supportsVision ? 'Yes' : 'No',
      c.supportsFunctionCalling ? 'Yes' : 'No',
      costTierLabel(c.costTier),
    ]);
    exportTableToCsv(headers, rows, 'model-capabilities');
  };

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-5">
        <MetricCard
          label="Total Models"
          value={String(unifiedModelRows.length)}
          icon={BarChart3}
          tooltip="Number of unique models used in routing"
        />
        <MetricCard
          label="Providers"
          value={String(uniqueProviders)}
          icon={Filter}
          tooltip="Number of distinct LLM providers"
        />
        <MetricCard
          label={`Total Requests (${selectedDays}d)`}
          value={formatNumber(totalRequests)}
          icon={Activity}
          tooltip="Total requests across all models in the period"
        />
        <MetricCard
          label={`Total Cost (${selectedDays}d)`}
          value={formatCost(totalCost)}
          icon={DollarSign}
          tooltip="Combined cost of all model API calls"
        />
        <MetricCard
          label="Avg Cost / Request"
          value={formatCost(avgCostPerReq)}
          icon={Zap}
          tooltip="Average cost per individual request across all models"
        />
      </div>

      {modelCapabilities.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Model Capability Comparison</CardTitle>
            <CardDescription>Feature support and cost tier across models</CardDescription>
            <CardAction>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 gap-1.5"
                    onClick={handleExportCapabilities}
                  >
                    <Download className="size-3" />
                    CSV
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Export capabilities as CSV</TooltipContent>
              </Tooltip>
            </CardAction>
          </CardHeader>
          <CardContent className="px-6">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead>Provider</TableHead>
                  <TableHead className="text-right">Context Window</TableHead>
                  <TableHead className="text-center">Vision</TableHead>
                  <TableHead className="text-center">Functions</TableHead>
                  <TableHead>Cost Tier</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {modelCapabilities.map((cap) => (
                  <TableRow key={cap.model}>
                    <TableCell className="font-medium">{cap.model}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className="text-xs font-normal">
                        {cap.provider}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {formatNumber(cap.contextWindow)}
                    </TableCell>
                    <TableCell className="text-center">
                      {cap.supportsVision ? (
                        <Eye className="mx-auto size-4 text-foreground" />
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </TableCell>
                    <TableCell className="text-center">
                      {cap.supportsFunctionCalling ? (
                        <Wrench className="mx-auto size-4 text-foreground" />
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary" className="text-xs">
                        {costTierLabel(cap.costTier)}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

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
              <ChartEmpty icon={DollarSign} message="No cost data" />
            )}
          </CardContent>
        </Card>

        <Card className="flex flex-col lg:col-span-2">
          <CardHeader className="items-center pb-0">
            <CardTitle className="text-base">Request Distribution</CardTitle>
            <CardDescription>By provider &middot; {selectedDays}d</CardDescription>
            <CardAction>
              <Badge variant="secondary" className="tabular-nums text-xs">
                {formatNumber(totalRequests)} total
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
                          `${formatNumber(Number(value))} requests`,
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
              <ChartEmpty icon={Activity} message="No provider data" />
            )}
          </CardContent>
        </Card>
      </div>

      {costEfficiency.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Cost Efficiency Score</CardTitle>
            <CardDescription>Requests per dollar — higher is more efficient</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {costEfficiency.slice(0, 4).map((item) => (
                <div key={item.model} className="space-y-1 rounded-lg border p-4">
                  <p className="truncate text-sm font-medium">{item.model}</p>
                  <p className="text-lg font-semibold tabular-nums">
                    {formatNumber(Math.round(item.requestsPerDollar))}
                  </p>
                  <p className="text-xs text-muted-foreground">requests / $1</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted-foreground">Filter:</span>
        {PROVIDERS.map((p) => (
          <Button
            key={p}
            variant={activeProviders.has(p) ? 'default' : 'outline'}
            size="sm"
            className="h-7 rounded-full px-3 text-xs"
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

      {/* Model Performance: Cluster Accuracy + Leaderboard */}
      {data.models && <ModelPerformanceContent data={data.models} />}
    </div>
  );
}

function ChartEmpty({ icon: Icon, message }: { icon: React.ElementType; message: string }) {
  return (
    <div className="flex h-40 flex-col items-center justify-center gap-2">
      <Icon className="size-8 text-muted-foreground" />
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  );
}
