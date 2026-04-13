import { useMemo } from 'react';
import {
  Route,
  Target,
  Download,
  DollarSign,
  Clock,
  Zap,
  Activity,
  AlertTriangle,
} from 'lucide-react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid } from 'recharts';
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '@/components/ui/table';
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
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { formatCost } from '@/utils/formatUtils';
import {
  formatDateWithYear,
  formatNumber,
  formatPercent,
  exportTableToCsv,
} from '../utils/intelligenceHelpers';
import type { IntelligenceData } from '../hooks/useIntelligenceData';
import { MetricCard } from './shared/MetricCard';
import { BaselineComparison } from './shared/BaselineComparison';
import { RoutingSkeleton, EmptyState, ErrorState } from './shared';

const CHART_COLORS = [
  'var(--chart-1)',
  'var(--chart-2)',
  'var(--chart-3)',
  'var(--chart-4)',
  'var(--chart-5)',
];

interface RoutingIntelligenceTabProps {
  data: IntelligenceData;
}

export function RoutingIntelligenceTab({ data }: RoutingIntelligenceTabProps) {
  const { loading, error, routingDecisions, refreshData } = data;

  if (loading) return <RoutingSkeleton />;
  if (error) return <ErrorState error={error} onRetry={refreshData} />;
  if (routingDecisions.length === 0)
    return (
      <EmptyState
        message="No routing decisions recorded yet. Decisions will appear after routing begins."
        onRefresh={refreshData}
      />
    );

  return <RoutingContent data={data} />;
}

function RoutingContent({ data }: { data: IntelligenceData }) {
  const { routingDecisions, efficiency, routingIntelligence, selectedDays } = data;

  const cb = efficiency?.cost_breakdown;
  const ri = routingIntelligence;

  const successRate = useMemo(() => {
    if (routingDecisions.length === 0) return 0;
    const successes = routingDecisions.filter((d) => d.outcome === 'success').length;
    return successes / routingDecisions.length;
  }, [routingDecisions]);

  const avgLatency = useMemo(() => {
    if (routingDecisions.length === 0) return 0;
    return routingDecisions.reduce((s, d) => s + d.latency, 0) / routingDecisions.length;
  }, [routingDecisions]);

  const avgCost = useMemo(() => {
    if (routingDecisions.length === 0) return 0;
    return routingDecisions.reduce((s, d) => s + d.cost, 0) / routingDecisions.length;
  }, [routingDecisions]);

  const winRateData = ri?.win_rate ?? [];
  const winRateConfig = useMemo<ChartConfig>(
    () => ({
      router: { label: 'Router', color: 'var(--chart-1)' },
      baseline: { label: 'Baseline', color: 'var(--chart-3)' },
    }),
    []
  );

  const confidenceData = ri?.confidence_distribution ?? [];
  const confidenceConfig = useMemo<ChartConfig>(
    () => ({ count: { label: 'Decisions', color: 'var(--chart-2)' } }),
    []
  );

  const modelUsage = ri?.model_usage ?? [];

  const dailyVolume = ri?.daily_volume ?? [];
  const volumeConfig = useMemo<ChartConfig>(
    () => ({
      count: { label: 'Requests', color: 'var(--chart-1)' },
      avg_latency: { label: 'Avg Latency', color: 'var(--chart-3)' },
    }),
    []
  );

  const latencyPercentiles = ri?.latency_percentiles ?? [];
  const latencyConfig = useMemo<ChartConfig>(
    () => ({
      p50: { label: 'P50', color: 'var(--chart-1)' },
      p75: { label: 'P75', color: 'var(--chart-2)' },
      p95: { label: 'P95', color: 'var(--chart-3)' },
      p99: { label: 'P99', color: 'var(--chart-4)' },
    }),
    []
  );

  const costByModel = useMemo(
    () =>
      modelUsage
        .filter((m) => m.avg_cost > 0)
        .map((m) => ({
          model: m.model,
          avg_cost: m.avg_cost,
          total_cost: m.avg_cost * m.count,
        }))
        .sort((a, b) => b.total_cost - a.total_cost),
    [modelUsage]
  );
  const costConfig = useMemo<ChartConfig>(
    () => ({
      avg_cost: { label: 'Avg Cost', color: 'var(--chart-2)' },
      total_cost: { label: 'Total Cost', color: 'var(--chart-1)' },
    }),
    []
  );

  const handleExportDecisions = () => {
    const headers = [
      'Request ID',
      'Model',
      'Provider',
      'Reason',
      'Cost',
      'Latency',
      'Tokens In',
      'Tokens Out',
      'Outcome',
      'Date',
    ];
    const rows = routingDecisions.map((d) => [
      d.requestId,
      d.modelChosen,
      d.provider,
      d.reason,
      formatCost(d.cost),
      `${d.latency.toFixed(0)}ms`,
      String(d.tokensIn),
      String(d.tokensOut),
      d.outcome,
      formatDateWithYear(d.timestamp),
    ]);
    exportTableToCsv(headers, rows, 'routing-decisions');
  };

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        <MetricCard
          label="Routing Decisions"
          value={formatNumber(routingDecisions.length)}
          icon={Route}
          tooltip="Total routing decisions in this period"
        />
        <MetricCard
          label="Success Rate"
          value={formatPercent(successRate)}
          icon={Target}
          tooltip="Percentage of decisions that resulted in a successful response"
        />
        <MetricCard
          label="Avg Latency"
          value={`${avgLatency.toFixed(0)}ms`}
          icon={Clock}
          tooltip="Average end-to-end latency for routed requests"
        />
        <MetricCard
          label="P95 Latency"
          value={`${(ri?.p95_latency ?? 0).toFixed(0)}ms`}
          icon={Activity}
          tooltip="95th percentile latency — 95% of requests are faster than this"
        />
        <MetricCard
          label="Avg Cost / Decision"
          value={formatCost(avgCost)}
          icon={DollarSign}
          tooltip="Average cost per routing decision"
        />
        <MetricCard
          label="Throughput"
          value={ri?.avg_tokens_per_s ? `${ri.avg_tokens_per_s.toFixed(0)} tok/s` : '—'}
          icon={Zap}
          tooltip="Average tokens generated per second across all models"
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Request Volume & Latency</CardTitle>
          <CardDescription>
            Daily request count with average latency overlay &middot; {selectedDays}d
          </CardDescription>
        </CardHeader>
        <CardContent>
          {dailyVolume.length > 0 ? (
            <ChartContainer config={volumeConfig} className="h-64 w-full">
              <BarChart data={dailyVolume}>
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
                  yAxisId="left"
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  width={40}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  width={50}
                  tickFormatter={(v) => `${Number(v).toFixed(0)}ms`}
                />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
                <Bar
                  yAxisId="left"
                  dataKey="count"
                  fill="var(--chart-1)"
                  radius={[4, 4, 0, 0]}
                  opacity={0.8}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="avg_latency"
                  stroke="var(--chart-3)"
                  strokeWidth={2}
                  dot={false}
                />
              </BarChart>
            </ChartContainer>
          ) : (
            <ChartEmpty label="No volume data available" />
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Routing Win Rate Over Time</CardTitle>
            <CardDescription>Router accuracy vs baseline &middot; {selectedDays}d</CardDescription>
          </CardHeader>
          <CardContent>
            {winRateData.length > 0 ? (
              <ChartContainer config={winRateConfig} className="h-64 w-full">
                <LineChart data={winRateData}>
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
                  <ChartLegend content={<ChartLegendContent />} />
                  <Line
                    type="monotone"
                    dataKey="router"
                    stroke="var(--chart-1)"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="baseline"
                    stroke="var(--chart-3)"
                    strokeWidth={1.5}
                    strokeDasharray="4 4"
                    dot={false}
                  />
                </LineChart>
              </ChartContainer>
            ) : (
              <ChartEmpty label="No win rate data available" />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Routing Confidence Distribution</CardTitle>
            <CardDescription>Distribution of confidence scores across decisions</CardDescription>
          </CardHeader>
          <CardContent>
            {confidenceData.length > 0 ? (
              <ChartContainer config={confidenceConfig} className="h-64 w-full">
                <BarChart data={confidenceData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-border" />
                  <XAxis
                    dataKey="bucket"
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={35} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar dataKey="count" fill="var(--chart-2)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ChartContainer>
            ) : (
              <ChartEmpty label="No confidence data available" />
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {latencyPercentiles.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Latency Percentiles by Model</CardTitle>
              <CardDescription>P50, P75, P95 and P99 latency per model (ms)</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={latencyConfig} className="h-64 w-full">
                <BarChart data={latencyPercentiles} layout="vertical">
                  <CartesianGrid
                    strokeDasharray="3 3"
                    horizontal={false}
                    className="stroke-border"
                  />
                  <XAxis
                    type="number"
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => `${Number(v).toFixed(0)}ms`}
                  />
                  <YAxis
                    dataKey="model"
                    type="category"
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    width={110}
                  />
                  <ChartTooltip
                    content={
                      <ChartTooltipContent formatter={(v: number) => `${Number(v).toFixed(0)}ms`} />
                    }
                  />
                  <ChartLegend content={<ChartLegendContent />} />
                  <Bar dataKey="p50" fill="var(--chart-1)" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="p95" fill="var(--chart-3)" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="p99" fill="var(--chart-4)" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        )}

        {costByModel.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Cost by Model</CardTitle>
              <CardDescription>Total and average cost per request by model</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={costConfig} className="h-64 w-full">
                <BarChart data={costByModel} layout="vertical">
                  <CartesianGrid
                    strokeDasharray="3 3"
                    horizontal={false}
                    className="stroke-border"
                  />
                  <XAxis
                    type="number"
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => formatCost(Number(v))}
                  />
                  <YAxis
                    dataKey="model"
                    type="category"
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    width={110}
                  />
                  <ChartTooltip
                    content={
                      <ChartTooltipContent formatter={(v: number) => formatCost(Number(v))} />
                    }
                  />
                  <ChartLegend content={<ChartLegendContent />} />
                  <Bar dataKey="total_cost" fill="var(--chart-1)" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="avg_cost" fill="var(--chart-2)" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        )}
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Router Decision Log</CardTitle>
          <CardDescription>
            Recent routing decisions with outcomes &middot; {selectedDays}d
          </CardDescription>
          <CardAction>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1.5"
                  onClick={handleExportDecisions}
                >
                  <Download className="size-3" />
                  CSV
                </Button>
              </TooltipTrigger>
              <TooltipContent>Export decisions as CSV</TooltipContent>
            </Tooltip>
          </CardAction>
        </CardHeader>
        <CardContent className="px-6">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Request ID</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Provider</TableHead>
                <TableHead>Reason</TableHead>
                <TableHead className="text-right">Cost</TableHead>
                <TableHead className="text-right">Latency</TableHead>
                <TableHead className="text-right">Tokens</TableHead>
                <TableHead>Outcome</TableHead>
                <TableHead>Date</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {routingDecisions.slice(0, 20).map((d) => (
                <TableRow key={d.requestId}>
                  <TableCell className="font-mono text-xs">{d.requestId}</TableCell>
                  <TableCell className="font-medium">{d.modelChosen}</TableCell>
                  <TableCell className="text-sm text-muted-foreground">{d.provider}</TableCell>
                  <TableCell className="max-w-40 truncate text-sm text-muted-foreground">
                    {d.reason}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{formatCost(d.cost)}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {d.latency.toFixed(0)}ms
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-xs text-muted-foreground">
                    {d.tokensIn > 0 || d.tokensOut > 0
                      ? `${d.tokensIn.toLocaleString()} / ${d.tokensOut.toLocaleString()}`
                      : '—'}
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={d.outcome === 'success' ? 'default' : 'destructive'}
                      className="text-xs"
                    >
                      {d.outcome === 'success' ? 'Success' : 'Error'}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {formatDateWithYear(d.timestamp)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {cb && <BaselineComparison cb={cb} variant="compact" />}
    </div>
  );
}

function ChartEmpty({ label }: { label: string }) {
  return (
    <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
      <div className="flex flex-col items-center gap-2">
        <AlertTriangle className="size-5 opacity-50" />
        <span>{label}</span>
      </div>
    </div>
  );
}
