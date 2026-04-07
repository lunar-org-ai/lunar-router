import { Layers, Trophy, Coins, Sparkles } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { KpiCard } from '@/components/shared/KpiCard';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import type { ModelPerformanceData } from '../../types';

const CHART_COLORS = ['var(--chart-1)', 'var(--chart-2)', 'var(--chart-3)', 'var(--chart-4)', 'var(--chart-5)'];

interface Props {
  data: ModelPerformanceData;
}

export function ModelPerformanceContent({ data }: Props) {
  const kpis = data.kpis as Record<string, unknown>;
  const bestModel = kpis.best_model as { model: string; accuracy: number } | undefined;
  const cheapestModel = kpis.cheapest_model as { model: string; cost: number } | undefined;
  const bestValue = kpis.best_value as { model: string; ratio: number } | undefined;

  // Build chart data: one bar group per cluster, one bar per model
  const clusterIds = data.cluster_accuracy.length > 0
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

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard label="Models Profiled" value={String(kpis.models_profiled ?? 0)} icon={Layers} />
        <KpiCard label="Best Model" value={bestModel?.model ?? '-'} icon={Trophy} subtitle={bestModel ? `${(bestModel.accuracy * 100).toFixed(1)}% accuracy` : undefined} />
        <KpiCard label="Cheapest Model" value={cheapestModel?.model ?? '-'} icon={Coins} subtitle={cheapestModel ? `$${cheapestModel.cost}/1k` : undefined} />
        <KpiCard label="Best Value" value={bestValue?.model ?? '-'} icon={Sparkles} subtitle={bestValue ? `${bestValue.ratio.toFixed(0)} acc/cost ratio` : undefined} />
      </div>

      {/* Cluster Accuracy Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Accuracy by Cluster</CardTitle>
        </CardHeader>
        <CardContent>
          {clusterChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={clusterChartData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="cluster" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} domain={[0, 1]} />
                <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
                <Legend />
                {modelNames.map((name, i) => (
                  <Bar key={name} dataKey={name} fill={CHART_COLORS[i % CHART_COLORS.length]} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="py-12 text-center text-muted-foreground">No cluster data. Load model profiles to see per-cluster accuracy.</p>
          )}
        </CardContent>
      </Card>

      {/* Model Leaderboard */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Model Leaderboard</CardTitle>
        </CardHeader>
        <CardContent>
          {data.leaderboard.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>#</TableHead>
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
                    <TableCell>{i + 1}</TableCell>
                    <TableCell className="font-medium">{row.model}</TableCell>
                    <TableCell className="text-right">{(row.accuracy * 100).toFixed(1)}%</TableCell>
                    <TableCell className="text-right">${row.cost.toFixed(6)}</TableCell>
                    <TableCell>
                      {row.strongest_clusters.map((c) => (
                        <Badge key={c} variant="outline" className="mr-1">C{c}</Badge>
                      ))}
                    </TableCell>
                    <TableCell>
                      {row.weakest_clusters.map((c) => (
                        <Badge key={c} variant="secondary" className="mr-1">C{c}</Badge>
                      ))}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="py-8 text-center text-muted-foreground">No models profiled yet.</p>
          )}
        </CardContent>
      </Card>

      {/* Teacher vs Student (conditional) */}
      {data.teacher_student && (
        <Card className="border-2 border-dashed border-primary/30">
          <CardHeader>
            <CardTitle className="text-base">Distilled Model Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <p className="text-sm text-muted-foreground">Teacher</p>
                <p className="text-lg font-semibold">{data.teacher_student.teacher}</p>
                <p className="text-sm">{(data.teacher_student.teacher_accuracy * 100).toFixed(1)}% accuracy</p>
                <p className="text-sm">${data.teacher_student.teacher_cost}/1k tokens</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Student</p>
                <p className="text-lg font-semibold">{data.teacher_student.student}</p>
                <Badge variant="outline" className="mt-1">distilled</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
