import { Activity, Clock, Brain, Users } from 'lucide-react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { KpiCard } from '@/components/shared/KpiCard';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import type { TrainingActivityData } from '../../types';

const REC_COLORS: Record<string, string> = {
  train_now: 'bg-red-100 text-red-800',
  wait: 'bg-green-100 text-green-800',
  investigate: 'bg-yellow-100 text-yellow-800',
  none: 'bg-gray-100 text-gray-800',
};

interface Props {
  data: TrainingActivityData;
  selectedDays: number;
  onTimeRangeChange: (days: number) => void;
}

export function TrainingActivityContent({ data, selectedDays, onTimeRangeChange }: Props) {
  const kpis = data.kpis as Record<string, unknown>;
  const advisorStatus = kpis.advisor_status as { recommendation: string; confidence: number } | undefined;

  // Prepare signal trends: pivot into {date, error_rate, drift_ratio, issue_count}
  const signalMap = new Map<string, Record<string, number>>();
  for (const s of data.signal_trends) {
    const dateKey = s.date.split('T')[0];
    if (!signalMap.has(dateKey)) signalMap.set(dateKey, { date: 0 } as unknown as Record<string, number>);
    const entry = signalMap.get(dateKey)!;
    (entry as Record<string, unknown>)['date'] = dateKey;
    entry[s.signal] = s.value;
  }
  const signalChartData = Array.from(signalMap.values());

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
        <KpiCard label="Training Runs" value={String(kpis.training_runs ?? 0)} icon={Activity} />
        <KpiCard
          label="Last Training"
          value={kpis.last_training ? new Date(kpis.last_training as string).toLocaleDateString() : 'Never'}
          icon={Clock}
        />
        <KpiCard
          label="Advisor Status"
          value={advisorStatus?.recommendation ?? 'none'}
          icon={Brain}
          subtitle={advisorStatus ? `${(advisorStatus.confidence * 100).toFixed(0)}% confidence` : undefined}
        />
        <KpiCard label="Models Updated" value={String(kpis.models_updated ?? 0)} icon={Users} />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Training History */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Training History</CardTitle>
          </CardHeader>
          <CardContent>
            {data.training_history.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={data.training_history.map((h, i) => ({ ...h, idx: i + 1, value: 1 }))}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="idx" tick={{ fontSize: 12 }} label={{ value: 'Run #', position: 'insideBottom', offset: -5 }} />
                  <YAxis hide />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.[0]) return null;
                      const d = payload[0].payload;
                      return (
                        <div className="rounded-md border bg-background p-2 text-sm shadow-sm">
                          <p className="font-medium">{d.promoted ? 'Promoted' : 'Rejected'}</p>
                          <p className="text-muted-foreground">{d.reason}</p>
                          <p className="text-xs text-muted-foreground">{d.date}</p>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {data.training_history.map((h, i) => (
                      <Cell key={i} fill={h.promoted ? 'var(--chart-2)' : 'var(--chart-1)'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="py-12 text-center text-muted-foreground">No training runs yet. The advisor will trigger training when needed.</p>
            )}
          </CardContent>
        </Card>

        {/* Signal Trends */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Signal Trends</CardTitle>
          </CardHeader>
          <CardContent>
            {signalChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={signalChartData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="error_rate_increase" stroke="var(--chart-1)" name="Error Rate" dot={false} />
                  <Line type="monotone" dataKey="drift_ratio" stroke="var(--chart-3)" name="Drift Ratio" dot={false} />
                  <Line type="monotone" dataKey="high_severity_issues" stroke="var(--chart-4)" name="Issues" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className="py-12 text-center text-muted-foreground">No signal data yet.</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Advisor Decisions */}
      {data.advisor_decisions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Recent Advisor Decisions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {data.advisor_decisions.slice(0, 10).map((d) => (
              <div key={d.id} className="rounded-lg border p-3">
                <div className="flex items-center gap-2">
                  <Badge className={REC_COLORS[d.recommendation] || REC_COLORS.none}>
                    {d.recommendation}
                  </Badge>
                  <span className="text-sm text-muted-foreground">{d.confidence ? `${(d.confidence * 100).toFixed(0)}%` : ''}</span>
                  <span className="ml-auto text-xs text-muted-foreground">{d.source} &middot; {new Date(d.timestamp).toLocaleString()}</span>
                </div>
                <p className="mt-1 text-sm">{d.reason}</p>
                {d.signals.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {d.signals.map((s, i) => (
                      <Badge key={i} variant={s.triggered ? 'default' : 'outline'} className="text-xs">
                        {s.name}: {typeof s.value === 'number' ? s.value.toFixed(3) : s.value}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Training Cycles Table */}
      {data.training_cycles.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Training Cycle Results (Before / After)</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Date</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">AUROC</TableHead>
                  <TableHead className="text-right">Win Rate</TableHead>
                  <TableHead className="text-right">APGR</TableHead>
                  <TableHead>Reason</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.training_cycles.map((c) => (
                  <TableRow key={c.id}>
                    <TableCell className="text-sm">{new Date(c.timestamp).toLocaleDateString()}</TableCell>
                    <TableCell>
                      <Badge variant={c.promoted ? 'default' : 'secondary'}>
                        {c.promoted ? 'Promoted' : 'Rejected'}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right text-sm">
                      {c.baseline.auroc?.toFixed(4) ?? '-'} &rarr; {c.new_metrics.auroc?.toFixed(4) ?? '-'}
                    </TableCell>
                    <TableCell className="text-right text-sm">
                      {c.baseline.win_rate ? `${(c.baseline.win_rate * 100).toFixed(1)}%` : '-'} &rarr;{' '}
                      {c.new_metrics.win_rate ? `${(c.new_metrics.win_rate * 100).toFixed(1)}%` : '-'}
                    </TableCell>
                    <TableCell className="text-right text-sm">
                      {c.baseline.apgr ? `${(c.baseline.apgr * 100).toFixed(1)}%` : '-'} &rarr;{' '}
                      {c.new_metrics.apgr ? `${(c.new_metrics.apgr * 100).toFixed(1)}%` : '-'}
                    </TableCell>
                    <TableCell className="max-w-[200px] truncate text-sm text-muted-foreground">{c.reason}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
