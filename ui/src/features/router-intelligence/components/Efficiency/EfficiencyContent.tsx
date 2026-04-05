import { DollarSign, Target, Zap, Hash } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { KpiCard } from '@/components/shared/KpiCard';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import type { EfficiencyData } from '../../types';

const CHART_COLORS = ['var(--chart-1)', 'var(--chart-2)', 'var(--chart-3)', 'var(--chart-4)', 'var(--chart-5)'];

interface Props {
  data: EfficiencyData;
  selectedDays: number;
  onTimeRangeChange: (days: number) => void;
}

export function EfficiencyContent({ data, selectedDays, onTimeRangeChange }: Props) {
  const kpis = data.kpis;
  const costSaved = kpis.cost_saved?.value ?? 0;
  const quality = kpis.quality_score?.value ?? 0;
  const avgCost = kpis.avg_cost_per_request?.value ?? 0;
  const totalReqs = kpis.requests_routed?.value ?? 0;

  // Get all model names from distribution data
  const modelNames = Array.from(
    new Set(data.model_distribution.flatMap((d) => Object.keys(d).filter((k) => k !== 'date')))
  );

  return (
    <div className="space-y-6">
      {/* Time range */}
      <div className="flex gap-2">
        {[7, 14, 30].map((d) => (
          <button
            key={d}
            onClick={() => onTimeRangeChange(d)}
            className={`rounded-md px-3 py-1 text-sm ${
              selectedDays === d ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground hover:bg-muted/80'
            }`}
          >
            {d}d
          </button>
        ))}
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard label="Cost Saved" value={`$${Number(costSaved).toFixed(4)}`} icon={DollarSign} subtitle="vs always-best-model baseline" />
        <KpiCard label="Quality Score" value={`${(Number(quality) * 100).toFixed(1)}%`} icon={Target} subtitle="routing accuracy (win rate)" />
        <KpiCard label="Avg Cost / Request" value={`$${Number(avgCost).toFixed(6)}`} icon={Zap} />
        <KpiCard label="Requests Routed" value={String(totalReqs)} icon={Hash} />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Model Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Model Distribution Over Time</CardTitle>
          </CardHeader>
          <CardContent>
            {data.model_distribution.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={data.model_distribution}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Legend />
                  {modelNames.map((name, i) => (
                    <Area
                      key={name}
                      type="monotone"
                      dataKey={name}
                      stackId="1"
                      fill={CHART_COLORS[i % CHART_COLORS.length]}
                      stroke={CHART_COLORS[i % CHART_COLORS.length]}
                      fillOpacity={0.7}
                    />
                  ))}
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <p className="py-12 text-center text-muted-foreground">No distribution data yet. Route some prompts to see model usage.</p>
            )}
          </CardContent>
        </Card>

        {/* Cost Savings Trend */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Cost Savings Trend</CardTitle>
          </CardHeader>
          <CardContent>
            {data.cost_savings_trend.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={data.cost_savings_trend}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `$${v}`} />
                  <Tooltip formatter={(v: number) => `$${v.toFixed(4)}`} />
                  <Legend />
                  <Area type="monotone" dataKey="baseline" fill="var(--chart-1)" stroke="var(--chart-1)" fillOpacity={0.15} name="Baseline" />
                  <Area type="monotone" dataKey="actual" fill="var(--chart-2)" stroke="var(--chart-2)" fillOpacity={0.4} name="Actual" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <p className="py-12 text-center text-muted-foreground">No cost data yet. Enable ClickHouse to track savings.</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Model Breakdown Table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Per-Model Breakdown</CardTitle>
        </CardHeader>
        <CardContent>
          {data.model_breakdown.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead className="text-right">Requests</TableHead>
                  <TableHead className="text-right">Accuracy</TableHead>
                  <TableHead className="text-right">Avg Cost</TableHead>
                  <TableHead className="text-right">Traffic %</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.model_breakdown.map((row) => (
                  <TableRow key={row.model}>
                    <TableCell className="font-medium">{row.model}</TableCell>
                    <TableCell className="text-right">{row.requests.toLocaleString()}</TableCell>
                    <TableCell className="text-right">{(row.accuracy * 100).toFixed(1)}%</TableCell>
                    <TableCell className="text-right">${row.avg_cost.toFixed(6)}</TableCell>
                    <TableCell className="text-right">
                      <Badge variant="outline">{row.traffic_pct.toFixed(1)}%</Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="py-8 text-center text-muted-foreground">No routing data yet.</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
