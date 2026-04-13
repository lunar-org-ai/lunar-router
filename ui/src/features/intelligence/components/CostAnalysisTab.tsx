import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { DollarSign, TrendingUp, Zap, ArrowUpRight, PiggyBank, Download } from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Pie,
  PieChart,
  XAxis,
  YAxis,
} from 'recharts';
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
import { formatNumber, exportTableToCsv } from '../utils/intelligenceHelpers';
import type { IntelligenceData } from '../hooks/useIntelligenceData';
import { MetricCard } from './shared/MetricCard';
import { BaselineComparison } from './shared/BaselineComparison';
import { CostsSkeleton, EmptyState, ErrorState } from './shared';

const CHART_COLORS = ['var(--chart-1)', 'var(--chart-2)', 'var(--chart-3)'];

interface CostAnalysisTabProps {
  data: IntelligenceData;
}

export function CostAnalysisTab({ data }: CostAnalysisTabProps) {
  const { loading, error, costData, efficiency, refreshData } = data;

  if (loading) return <CostsSkeleton />;
  if (error) return <ErrorState error={error} onRetry={refreshData} />;
  if (!costData && !efficiency)
    return (
      <EmptyState
        message="No cost data available yet. Costs will appear after routing begins."
        onRefresh={refreshData}
      />
    );

  return <CostAnalysisContent data={data} />;
}

function CostAnalysisContent({ data }: { data: IntelligenceData }) {
  const navigate = useNavigate();
  const { costData, efficiency, overviewData, selectedDays } = data;

  const totals = useMemo(() => {
    const externalCosts = costData?.externalCosts || [];
    const totalCost = externalCosts.reduce((sum, item) => sum + item.cost, 0);
    const projected = totalCost * (30 / selectedDays);
    return { totalCost, projected };
  }, [costData, selectedDays]);

  const kpis = efficiency?.kpis;
  const avgCost = (kpis?.avg_cost_per_request as { value: number } | undefined)?.value ?? 0;
  const costSaved = (kpis?.cost_saved as { value: number } | undefined)?.value ?? 0;

  const cb = efficiency?.cost_breakdown;

  const externalProviders = overviewData?.providers?.filter((p) => !p.isLunar) ?? [];

  const pieData = useMemo(
    () =>
      externalProviders.map((p, i) => ({
        provider: `provider-${i}`,
        cost: p.cost,
        fill: `var(--color-provider-${i})`,
      })),
    [externalProviders]
  );

  const pieConfig = useMemo<ChartConfig>(() => {
    const cfg: ChartConfig = { cost: { label: 'Cost' } };
    externalProviders.forEach((p, i) => {
      cfg[`provider-${i}`] = {
        label: p.provider,
        color: CHART_COLORS[i % CHART_COLORS.length],
      };
    });
    return cfg;
  }, [externalProviders]);

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

  const savingsConfig = useMemo<ChartConfig>(
    () => ({
      baseline: { label: 'Baseline', color: 'var(--chart-1)' },
      actual: { label: 'Actual', color: 'var(--chart-2)' },
    }),
    []
  );

  const costOverTimeConfig = useMemo<ChartConfig>(
    () => ({ cost: { label: 'Cost', color: 'var(--chart-2)' } }),
    []
  );

  const externalRequests = costData?.expensiveRequests?.filter((r) => !r.isLunar) ?? [];

  const handleExportExpensive = () => {
    const headers = ['Cost', 'Model', 'Tokens', 'Date'];
    const rows = externalRequests
      .slice(0, 8)
      .map((req) => [formatCost(req.cost), req.model, req.promptSize, req.date]);
    exportTableToCsv(headers, rows, 'expensive-requests');
  };

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label={`Total Cost (${selectedDays}d)`}
          value={formatCost(totals.totalCost)}
          icon={DollarSign}
          tooltip="Sum of all external API costs for the selected period"
        />
        <MetricCard
          label="Avg Cost / Request"
          value={`$${Number(avgCost).toFixed(6)}`}
          icon={Zap}
          tooltip="Average cost per routed request across all models"
        />
        <MetricCard
          label="Monthly Projection"
          value={formatCost(totals.projected)}
          icon={TrendingUp}
          tooltip="Projected monthly cost based on current spending rate"
        />
        <MetricCard
          label="Cost Saved vs Baseline"
          value={`$${Number(costSaved).toFixed(4)}`}
          icon={PiggyBank}
          tooltip="Total savings from routing vs always using the most expensive model"
        />
      </div>

      {cb && <BaselineComparison cb={cb} variant="full" />}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Cost Savings Trend</CardTitle>
            <CardDescription>Baseline vs actual cost &middot; {selectedDays}d</CardDescription>
          </CardHeader>
          <CardContent>
            {efficiency?.cost_savings_trend && efficiency.cost_savings_trend.length > 0 ? (
              <ChartContainer config={savingsConfig} className="h-72 w-full">
                <AreaChart data={efficiency.cost_savings_trend}>
                  <defs>
                    <linearGradient id="fillBaseline" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.2} />
                      <stop offset="100%" stopColor="var(--chart-1)" stopOpacity={0.02} />
                    </linearGradient>
                    <linearGradient id="fillActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--chart-2)" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="var(--chart-2)" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-border" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => `$${v}`}
                    width={50}
                  />
                  <ChartTooltip
                    content={
                      <ChartTooltipContent formatter={(v: number) => `$${Number(v).toFixed(4)}`} />
                    }
                  />
                  <ChartLegend content={<ChartLegendContent />} />
                  <Area
                    type="monotone"
                    dataKey="baseline"
                    fill="url(#fillBaseline)"
                    stroke="var(--chart-1)"
                    strokeWidth={1.5}
                  />
                  <Area
                    type="monotone"
                    dataKey="actual"
                    fill="url(#fillActual)"
                    stroke="var(--chart-2)"
                    strokeWidth={1.5}
                  />
                </AreaChart>
              </ChartContainer>
            ) : (
              <ChartEmpty icon={TrendingUp} message="No cost savings data yet" />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Cost Over Time</CardTitle>
            <CardDescription>Daily spending &middot; {selectedDays}d</CardDescription>
            <CardAction>
              <Badge variant="secondary" className="tabular-nums text-xs">
                {formatCost(totals.totalCost / selectedDays)}/day avg
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent>
            {costData?.timeSeries && costData.timeSeries.length > 0 ? (
              <ChartContainer config={costOverTimeConfig} className="h-72 w-full">
                <AreaChart data={costData.timeSeries}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" className="stroke-border" />
                  <XAxis
                    dataKey="date"
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => {
                      const d = new Date(v);
                      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                    }}
                    tick={{ fontSize: 11 }}
                  />
                  <ChartTooltip
                    content={
                      <ChartTooltipContent
                        formatter={(value: number) => [formatCost(Number(value)), 'Cost']}
                      />
                    }
                  />
                  <defs>
                    <linearGradient id="fillCostIntel" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--chart-2)" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="var(--chart-2)" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <Area
                    dataKey="cost"
                    type="monotone"
                    fill="url(#fillCostIntel)"
                    stroke="var(--chart-2)"
                    strokeWidth={2}
                  />
                </AreaChart>
              </ChartContainer>
            ) : (
              <ChartEmpty icon={TrendingUp} message="No time series data" />
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle className="text-base">Cost by Model</CardTitle>
            <CardDescription>Top external API costs &middot; {selectedDays}d</CardDescription>
            <CardAction>
              <Badge variant="secondary" className="tabular-nums text-xs">
                {formatCost(totals.totalCost)} total
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
            <CardTitle className="text-base">Cost by Provider</CardTitle>
            <CardDescription>By provider &middot; {selectedDays}d</CardDescription>
            <CardAction>
              <Badge variant="secondary" className="tabular-nums text-xs">
                {formatCost(totals.totalCost)} total
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent className="flex-1 p-0">
            {pieData.length > 0 ? (
              <ChartContainer config={pieConfig} className="mx-auto aspect-square max-h-56">
                <PieChart>
                  <ChartTooltip
                    cursor={false}
                    content={
                      <ChartTooltipContent
                        formatter={(value: number) => [`$${Number(value).toFixed(2)}`, '']}
                      />
                    }
                  />
                  <Pie
                    data={pieData}
                    dataKey="cost"
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
              <ChartEmpty icon={DollarSign} message="No external provider costs" />
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Most Expensive Requests</CardTitle>
          <CardDescription>Click a row to view the trace</CardDescription>
          <CardAction>
            {externalRequests.length > 0 && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 gap-1.5"
                    onClick={handleExportExpensive}
                  >
                    <Download className="size-3" />
                    CSV
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Export as CSV</TooltipContent>
              </Tooltip>
            )}
          </CardAction>
        </CardHeader>
        <CardContent className="px-6">
          {externalRequests.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Cost</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Tokens</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead className="w-8" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {externalRequests.slice(0, 8).map((req, i) => (
                  <TableRow
                    key={req.id || i}
                    className="cursor-pointer"
                    onClick={() => navigate(`/traces?trace=${req.id}`)}
                  >
                    <TableCell>
                      <Badge variant="secondary" className="tabular-nums">
                        {formatCost(req.cost)}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-medium">{req.model}</TableCell>
                    <TableCell className="tabular-nums text-muted-foreground">
                      {formatNumber(req.promptSize)}
                    </TableCell>
                    <TableCell className="text-muted-foreground">{req.date}</TableCell>
                    <TableCell className="text-right">
                      <Button variant="ghost" size="icon" className="size-7">
                        <ArrowUpRight className="size-3.5" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="flex h-32 flex-col items-center justify-center gap-2">
              <p className="text-sm text-muted-foreground">No expensive requests recorded</p>
            </div>
          )}
        </CardContent>
      </Card>
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
