import { useMemo } from 'react';
import { Layers, Trophy, Coins, Sparkles, BarChart3, Download } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';
import {
  Card,
  CardAction,
  CardHeader,
  CardTitle,
  CardContent,
  CardDescription,
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
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
  type ChartConfig,
} from '@/components/ui/chart';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { formatPercent, exportTableToCsv } from '../utils/intelligenceHelpers';
import type { ModelPerformanceData } from '@/features/router-intelligence/types';
import { MetricCard } from './shared/MetricCard';

const CHART_COLORS = ['var(--chart-1)', 'var(--chart-2)', 'var(--chart-3)'];

interface Props {
  data: ModelPerformanceData;
}

export function ModelPerformanceContent({ data }: Props) {
  const kpis = data.kpis as Record<string, unknown>;
  const bestModel = kpis.best_model as { model: string; accuracy: number } | undefined;
  const cheapestModel = kpis.cheapest_model as { model: string; cost: number } | undefined;
  const bestValue = kpis.best_value as { model: string; ratio: number } | undefined;

  const clusterIds =
    data.cluster_accuracy.length > 0
      ? Object.keys(data.cluster_accuracy[0].clusters).sort((a, b) => Number(a) - Number(b))
      : [];

  const clusterChartData = clusterIds.map((cid) => {
    const row: Record<string, unknown> = { cluster: `C${cid}` };
    for (const m of data.cluster_accuracy) {
      row[m.model] = m.clusters[cid] ?? 0;
    }
    return row;
  });

  const modelNames = data.cluster_accuracy.map((m) => m.model);

  const clusterConfig = useMemo<ChartConfig>(() => {
    const cfg: ChartConfig = {};
    modelNames.forEach((name, i) => {
      cfg[name] = { label: name, color: CHART_COLORS[i % CHART_COLORS.length] };
    });
    return cfg;
  }, [modelNames]);

  const handleExportLeaderboard = () => {
    const headers = [
      'Rank',
      'Model',
      'Accuracy',
      'Cost/1k',
      'Strongest Clusters',
      'Weakest Clusters',
    ];
    const rows = data.leaderboard.map((row, i) => [
      i + 1,
      row.model,
      formatPercent(row.accuracy),
      `$${row.cost.toFixed(6)}`,
      row.strongest_clusters.map((c) => `C${c}`).join(', '),
      row.weakest_clusters.map((c) => `C${c}`).join(', '),
    ]);
    exportTableToCsv(headers, rows, 'model-leaderboard');
  };

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricCard
          label="Models Profiled"
          value={String(kpis.models_profiled ?? 0)}
          icon={Layers}
          tooltip="Number of models that have been profiled for accuracy"
        />
        <MetricCard
          label="Best Model"
          value={bestModel?.model ?? '—'}
          icon={Trophy}
          subtitle={bestModel ? `${formatPercent(bestModel.accuracy)} accuracy` : undefined}
          tooltip="Model with the highest overall accuracy across clusters"
        />
        <MetricCard
          label="Cheapest Model"
          value={cheapestModel?.model ?? '—'}
          icon={Coins}
          subtitle={cheapestModel ? `$${cheapestModel.cost}/1k` : undefined}
          tooltip="Model with the lowest cost per 1,000 tokens"
        />
        <MetricCard
          label="Best Value"
          value={bestValue?.model ?? '—'}
          icon={Sparkles}
          subtitle={bestValue ? `${bestValue.ratio.toFixed(0)} acc/cost ratio` : undefined}
          tooltip="Model with the best accuracy-to-cost ratio"
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Accuracy by Cluster</CardTitle>
          <CardDescription>Per-model accuracy across request clusters</CardDescription>
        </CardHeader>
        <CardContent>
          {clusterChartData.length > 0 ? (
            <ChartContainer config={clusterConfig} className="h-75 w-full">
              <BarChart data={clusterChartData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-border" />
                <XAxis
                  dataKey="cluster"
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => formatPercent(Number(v))}
                  domain={[0, 1]}
                  width={40}
                />
                <ChartTooltip
                  content={
                    <ChartTooltipContent formatter={(v: number) => formatPercent(Number(v))} />
                  }
                />
                <ChartLegend content={<ChartLegendContent />} />
                {modelNames.map((name, i) => (
                  <Bar
                    key={name}
                    dataKey={name}
                    fill={CHART_COLORS[i % CHART_COLORS.length]}
                    radius={[4, 4, 0, 0]}
                  />
                ))}
              </BarChart>
            </ChartContainer>
          ) : (
            <ChartEmpty message="No cluster data available" />
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Model Leaderboard</CardTitle>
          <CardDescription>Ranked by accuracy across all clusters</CardDescription>
          <CardAction>
            {data.leaderboard.length > 0 && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 gap-1.5"
                    onClick={handleExportLeaderboard}
                  >
                    <Download className="size-3" />
                    CSV
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Export leaderboard as CSV</TooltipContent>
              </Tooltip>
            )}
          </CardAction>
        </CardHeader>
        <CardContent>
          {data.leaderboard.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-10">#</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead className="text-right">Accuracy</TableHead>
                  <TableHead className="text-right">Cost/1k</TableHead>
                  <TableHead>Strongest Clusters</TableHead>
                  <TableHead>Weakest Clusters</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.leaderboard.map((row, i) => (
                  <TableRow key={row.model}>
                    <TableCell className="text-muted-foreground">{i + 1}</TableCell>
                    <TableCell className="font-medium">{row.model}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {formatPercent(row.accuracy)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      ${row.cost.toFixed(6)}
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {row.strongest_clusters.map((c) => (
                          <Badge key={c} variant="outline" className="text-xs">
                            C{c}
                          </Badge>
                        ))}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {row.weakest_clusters.map((c) => (
                          <Badge key={c} variant="secondary" className="text-xs">
                            C{c}
                          </Badge>
                        ))}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No leaderboard data available
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function ChartEmpty({ message }: { message: string }) {
  return (
    <div className="flex h-56 flex-col items-center justify-center gap-3">
      <div className="flex size-12 items-center justify-center rounded-full bg-muted">
        <BarChart3 className="size-5 text-muted-foreground" />
      </div>
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  );
}
