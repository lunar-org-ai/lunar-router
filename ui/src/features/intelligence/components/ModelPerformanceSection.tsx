import { Layers, Trophy, Coins, Sparkles, BarChart3 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';
import { KpiCard } from '@/components/shared/KpiCard';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
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
import type { ModelPerformanceData } from '@/features/router-intelligence/types';
import { useMemo } from 'react';

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

  return (
    <div className="space-y-6">
      {/* KPIs */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <KpiCard label="Models Profiled" value={String(kpis.models_profiled ?? 0)} icon={Layers} />
        <KpiCard
          label="Best Model"
          value={bestModel?.model ?? '-'}
          icon={Trophy}
          subtitle={bestModel ? `${(bestModel.accuracy * 100).toFixed(1)}% accuracy` : undefined}
        />
        <KpiCard
          label="Cheapest Model"
          value={cheapestModel?.model ?? '-'}
          icon={Coins}
          subtitle={cheapestModel ? `$${cheapestModel.cost}/1k` : undefined}
        />
        <KpiCard
          label="Best Value"
          value={bestValue?.model ?? '-'}
          icon={Sparkles}
          subtitle={bestValue ? `${bestValue.ratio.toFixed(0)} acc/cost ratio` : undefined}
        />
      </div>

      {/* Accuracy by Cluster */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Accuracy by Cluster</CardTitle>
          <CardDescription>Per-model accuracy across request clusters</CardDescription>
        </CardHeader>
        <CardContent>
          {clusterChartData.length > 0 ? (
            <ChartContainer config={clusterConfig} className="h-75 w-full">
              <BarChart data={clusterChartData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  vertical={false}
                  className="stroke-border/50"
                />
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
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                  domain={[0, 1]}
                  width={40}
                />
                <ChartTooltip
                  content={
                    <ChartTooltipContent
                      formatter={(v: number) => `${(Number(v) * 100).toFixed(1)}%`}
                    />
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
            <div className="flex h-56 flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border/60">
              <div className="flex size-12 items-center justify-center rounded-full bg-muted/50">
                <BarChart3 className="size-5 text-muted-foreground/50" />
              </div>
              <div className="text-center">
                <p className="text-sm font-medium text-muted-foreground">No cluster data</p>
                <p className="mt-0.5 text-xs text-muted-foreground/70">
                  Load model profiles to see per-cluster accuracy.
                </p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Model Leaderboard */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Model Leaderboard</CardTitle>
          <CardDescription>Ranked by accuracy across all clusters</CardDescription>
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
                  <TableRow key={row.model} className="transition-colors hover:bg-muted/30">
                    <TableCell className="text-muted-foreground">{i + 1}</TableCell>
                    <TableCell className="font-medium">{row.model}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {(row.accuracy * 100).toFixed(1)}%
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
              No models profiled yet.
            </p>
          )}
        </CardContent>
      </Card>

      {/* Distilled Model Comparison */}
      {data.teacher_student && (
        <Card className="border-primary/20 bg-primary/2">
          <CardHeader>
            <CardTitle className="text-base">Distilled Model Comparison</CardTitle>
            <CardDescription>Teacher vs student model performance</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-8">
              <div className="space-y-1.5">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Teacher
                </p>
                <p className="text-lg font-semibold">{data.teacher_student.teacher}</p>
                <div className="flex gap-3 text-sm text-muted-foreground">
                  <span>{(data.teacher_student.teacher_accuracy * 100).toFixed(1)}% accuracy</span>
                  <span>${data.teacher_student.teacher_cost}/1k tokens</span>
                </div>
              </div>
              <div className="space-y-1.5">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Student
                </p>
                <p className="text-lg font-semibold">{data.teacher_student.student}</p>
                <Badge variant="outline" className="text-xs">
                  distilled
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
