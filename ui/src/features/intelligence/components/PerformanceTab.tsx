import { useMemo, useState } from 'react';
import {
  Brain,
  Zap,
  BarChart3,
  TrendingUp,
  GraduationCap,
  CheckCircle2,
  XCircle,
  Loader2,
  Ban,
  Download,
  ChevronDown,
  ChevronRight,
  ArrowRight,
  Activity,
  Clock,
  Users,
  AlertTriangle,
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  Label,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';
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
  formatStatus,
  formatPercent,
  exportTableToCsv,
} from '../utils/intelligenceHelpers';
import type { IntelligenceData } from '../hooks/useIntelligenceData';
import type {
  DistillationSummary,
  TrainingActivityData,
  ModelPerformanceData,
} from '@/features/router-intelligence/types';
import type { TrainingRunDetail } from '../types';
import { MetricCard } from './shared/MetricCard';
import { PerformanceSkeleton, EmptyState, ErrorState } from './shared';

const CHART_COLORS = [
  'var(--chart-1)',
  'var(--chart-2)',
  'var(--chart-3)',
  'var(--chart-4)',
  'var(--chart-5)',
];

const STATUS_META: Record<
  string,
  {
    label: string;
    variant: 'default' | 'secondary' | 'destructive' | 'outline';
    icon: React.ElementType;
  }
> = {
  completed: { label: 'Completed', variant: 'default', icon: CheckCircle2 },
  failed: { label: 'Failed', variant: 'destructive', icon: XCircle },
  running: { label: 'Running', variant: 'outline', icon: Loader2 },
  cancelled: { label: 'Cancelled', variant: 'secondary', icon: Ban },
};

const REC_COLORS: Record<string, 'default' | 'secondary' | 'destructive' | 'outline'> = {
  train_now: 'destructive',
  wait: 'default',
  investigate: 'outline',
  none: 'secondary',
};

const SIGNAL_CHART_CONFIG: ChartConfig = {
  error_rate_increase: { label: 'Error Rate', color: 'var(--chart-1)' },
  drift_ratio: { label: 'Drift Ratio', color: 'var(--chart-3)' },
  high_severity_issues: { label: 'Issues', color: 'var(--chart-4)' },
};

interface PerformanceTabProps {
  data: IntelligenceData;
}

export function PerformanceTab({ data }: PerformanceTabProps) {
  const { loading, error, training, refreshData } = data;

  if (loading) return <PerformanceSkeleton />;
  if (error) return <ErrorState error={error} onRetry={refreshData} />;
  if (!training)
    return <EmptyState message="No distillation data available yet." onRefresh={refreshData} />;

  return <PerformanceContent data={data} />;
}

function PerformanceContent({ data }: { data: IntelligenceData }) {
  const { training, trainingRuns, selectedDays } = data;
  const distSummary = training?.distillation_summary ?? null;

  const stats = useMemo(() => buildStats(trainingRuns, distSummary), [trainingRuns, distSummary]);
  const teacherData = useMemo(
    () => buildModelPieData(trainingRuns, 'teacherModel'),
    [trainingRuns]
  );
  const studentData = useMemo(
    () => buildModelPieData(trainingRuns, 'studentModel'),
    [trainingRuns]
  );

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-6">
        <MetricCard
          label="Total Jobs"
          value={String(stats.total)}
          icon={GraduationCap}
          tooltip="Total distillation jobs executed"
        />
        <MetricCard
          label="Completed"
          value={String(stats.completed)}
          icon={CheckCircle2}
          tooltip="Successfully completed distillation jobs"
        />
        <MetricCard
          label="Failed"
          value={String(stats.failed)}
          icon={XCircle}
          tooltip="Failed distillation jobs"
        />
        <MetricCard
          label="Success Rate"
          value={formatPercent(stats.successRate, true)}
          icon={TrendingUp}
          tooltip="Percentage of jobs completed successfully"
        />
        <MetricCard
          label="Avg Quality"
          value={stats.avgQuality > 0 ? formatPercent(stats.avgQuality, true) : '—'}
          icon={Brain}
          tooltip="Average quality score across completed runs"
        />
        <MetricCard
          label="Training Cost"
          value={formatCost(stats.totalCost)}
          icon={Zap}
          tooltip="Total cost of all distillation runs"
        />
      </div>

      {distSummary && <PipelineCard summary={distSummary} />}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ModelDonutCard
          title="Teacher Models"
          description="Distribution across teacher models"
          data={teacherData.items}
          config={teacherData.config}
          total={teacherData.total}
        />
        <ModelDonutCard
          title="Student Models"
          description="Distribution across student models"
          data={studentData.items}
          config={studentData.config}
          total={studentData.total}
        />
      </div>

      {data.models?.teacher_student && <TeacherVsStudentCard ts={data.models.teacher_student} />}

      {training && <TrainingActivitySection training={training} selectedDays={selectedDays} />}
      {trainingRuns.length > 0 && <RunsTable runs={trainingRuns} days={selectedDays} />}
    </div>
  );
}

function PipelineCard({ summary }: { summary: DistillationSummary }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <GraduationCap className="size-4 text-muted-foreground" />
          <CardTitle className="text-base">Distillation Pipeline</CardTitle>
        </div>
        <CardDescription>Model training pipeline status</CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Status counters — evenly spaced across card */}
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <StatusBox
            icon={CheckCircle2}
            iconClass="text-emerald-500"
            bgClass="bg-emerald-500/10"
            count={summary.completed_jobs}
            label="Completed"
          />
          <StatusBox
            icon={Loader2}
            iconClass="text-blue-500 animate-spin"
            bgClass="bg-blue-500/10"
            count={summary.running_jobs}
            label="Running"
          />
          <StatusBox
            icon={XCircle}
            iconClass="text-red-500"
            bgClass="bg-red-500/10"
            count={summary.failed_jobs}
            label="Failed"
          />
          <StatusBox
            icon={GraduationCap}
            iconClass="text-muted-foreground"
            bgClass="bg-muted"
            count={summary.total_jobs}
            label="Total"
          />
        </div>

        {summary.latest_completed_job && (
          <div className="flex flex-col gap-3 rounded-lg border bg-muted/30 p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground">Latest completed job</p>
              <p className="text-sm font-medium">{summary.latest_completed_job.name}</p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-muted-foreground">Teacher</span>
                <Badge variant="outline" className="text-xs font-mono">
                  {summary.latest_completed_job.teacher_model || '—'}
                </Badge>
              </div>
              <ArrowRight className="hidden size-3 text-muted-foreground sm:block" />
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-muted-foreground">Student</span>
                <Badge variant="secondary" className="text-xs font-mono">
                  {summary.latest_completed_job.student_model || '—'}
                </Badge>
              </div>
              {summary.latest_completed_job.cost > 0 && (
                <Badge variant="outline" className="text-xs">
                  {formatCost(summary.latest_completed_job.cost)}
                </Badge>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function StatusBox({
  icon: Icon,
  iconClass,
  bgClass,
  count,
  label,
}: {
  icon: React.ElementType;
  iconClass: string;
  bgClass: string;
  count: number;
  label: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-lg border bg-card p-3">
      <div className={`flex size-9 shrink-0 items-center justify-center rounded-full ${bgClass}`}>
        <Icon className={`size-4 ${iconClass}`} />
      </div>
      <div>
        <p className="text-xl font-semibold tabular-nums leading-none">{count}</p>
        <p className="mt-0.5 text-xs text-muted-foreground">{label}</p>
      </div>
    </div>
  );
}

interface PieItem {
  name: string;
  count: number;
  fill: string;
}

function ModelDonutCard({
  title,
  description,
  data,
  config,
  total,
}: {
  title: string;
  description: string;
  data: PieItem[];
  config: ChartConfig;
  total: number;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        {data.length > 0 ? (
          <ChartContainer config={config} className="mx-auto aspect-square max-h-52">
            <PieChart>
              <ChartTooltip content={<ChartTooltipContent nameKey="name" hideLabel />} />
              <Pie
                data={data}
                dataKey="count"
                nameKey="name"
                innerRadius={55}
                outerRadius={80}
                strokeWidth={2}
                stroke="hsl(var(--card))"
              >
                {data.map((entry) => (
                  <Cell key={entry.name} fill={entry.fill} />
                ))}
                <Label
                  content={({ viewBox }) => {
                    if (viewBox && 'cx' in viewBox && 'cy' in viewBox) {
                      return (
                        <text
                          x={viewBox.cx}
                          y={viewBox.cy}
                          textAnchor="middle"
                          dominantBaseline="middle"
                        >
                          <tspan
                            x={viewBox.cx}
                            y={(viewBox.cy ?? 0) - 6}
                            className="fill-foreground text-2xl font-semibold"
                          >
                            {total}
                          </tspan>
                          <tspan
                            x={viewBox.cx}
                            y={(viewBox.cy ?? 0) + 14}
                            className="fill-muted-foreground text-xs"
                          >
                            runs
                          </tspan>
                        </text>
                      );
                    }
                    return null;
                  }}
                />
              </Pie>
            </PieChart>
          </ChartContainer>
        ) : (
          <ChartEmpty icon={BarChart3} message={`No ${title.toLowerCase()} data`} />
        )}
        {data.length > 0 && (
          <div className="mt-2 space-y-1.5">
            {data.map((d) => (
              <div key={d.name} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <span
                    className="inline-block size-2.5 rounded-full"
                    style={{ backgroundColor: d.fill }}
                  />
                  <code className="font-mono text-xs">{d.name}</code>
                </div>
                <span className="tabular-nums text-muted-foreground">
                  {d.count}{' '}
                  <span className="text-xs">
                    ({total > 0 ? ((d.count / total) * 100).toFixed(0) : 0}%)
                  </span>
                </span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

type Filter = 'all' | 'completed' | 'failed' | 'running' | 'cancelled';

function RunsTable({ runs, days }: { runs: TrainingRunDetail[]; days: number }) {
  const [filter, setFilter] = useState<Filter>('all');
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const filtered = useMemo(
    () => (filter === 'all' ? runs : runs.filter((r) => r.status === filter)),
    [runs, filter]
  );

  const counts = useMemo(() => {
    const c: Record<string, number> = { all: runs.length };
    for (const r of runs) c[r.status] = (c[r.status] ?? 0) + 1;
    return c;
  }, [runs]);

  const exportCsv = () => {
    const headers = ['Name', 'Date', 'Status', 'Teacher', 'Student', 'Quality', 'Duration'];
    const rows = filtered.map((r) => [
      r.name,
      formatDateWithYear(r.date),
      r.status || r.outcome,
      r.teacherModel,
      r.studentModel,
      r.qualityScore > 0 ? formatPercent(r.qualityScore, true) : '',
      humanDuration(r.duration),
    ]);
    exportTableToCsv(headers, rows, 'distillation-runs');
  };

  const filterOptions: { key: Filter; label: string }[] = [
    { key: 'all', label: 'All' },
    { key: 'completed', label: 'Completed' },
    { key: 'failed', label: 'Failed' },
    { key: 'running', label: 'Running' },
    { key: 'cancelled', label: 'Cancelled' },
  ];

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Distillation Runs</CardTitle>
        <CardDescription>Detailed history of all runs &middot; {days}d</CardDescription>
        <CardAction>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="outline" size="sm" className="h-7 gap-1.5" onClick={exportCsv}>
                <Download className="size-3" />
                CSV
              </Button>
            </TooltipTrigger>
            <TooltipContent>Export runs as CSV</TooltipContent>
          </Tooltip>
        </CardAction>
      </CardHeader>

      <div className="flex items-center gap-3 px-5 pb-2 text-sm">
        {filterOptions.map((f) => {
          const n = counts[f.key] ?? 0;
          if (f.key !== 'all' && n === 0) return null;
          const active = filter === f.key;
          return (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className={`tabular-nums ${active ? 'font-medium text-foreground' : 'text-muted-foreground hover:text-foreground'}`}
            >
              {f.label} <span className="text-xs opacity-60">{n}</span>
            </button>
          );
        })}
      </div>

      <CardContent className="px-5 pt-0">
        <div className="max-h-[520px] overflow-auto">
          <Table>
            <TableHeader className="sticky top-0 z-10 bg-card">
              <TableRow>
                <TableHead className="w-6" />
                <TableHead>Name</TableHead>
                <TableHead>Date</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Teacher</TableHead>
                <TableHead>Student</TableHead>
                <TableHead className="text-right">Quality</TableHead>
                <TableHead>Duration</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.slice(0, 30).map((run) => {
                const open = expandedId === run.runId;
                return (
                  <RunRow
                    key={run.runId}
                    run={run}
                    open={open}
                    toggle={() => setExpandedId(open ? null : run.runId)}
                  />
                );
              })}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}

function RunRow({
  run,
  open,
  toggle,
}: {
  run: TrainingRunDetail;
  open: boolean;
  toggle: () => void;
}) {
  const meta = STATUS_META[run.status] ?? STATUS_META.failed;
  const Icon = meta.icon;
  const dur = humanDuration(run.duration);

  return (
    <>
      <TableRow className="cursor-pointer" onClick={toggle}>
        <TableCell className="w-6 px-1">
          {open ? (
            <ChevronDown className="size-3 text-muted-foreground" />
          ) : (
            <ChevronRight className="size-3 text-muted-foreground" />
          )}
        </TableCell>
        <TableCell className="max-w-44 truncate font-medium">{run.name}</TableCell>
        <TableCell className="text-sm text-muted-foreground">
          {formatDateWithYear(run.date)}
        </TableCell>
        <TableCell>
          <Badge variant={meta.variant} className="gap-1 text-xs">
            <Icon className={`size-3 ${run.status === 'running' ? 'animate-spin' : ''}`} />
            {meta.label}
          </Badge>
        </TableCell>
        <TableCell>
          {run.teacherModel ? (
            <code className="font-mono text-xs">{run.teacherModel}</code>
          ) : (
            <span className="text-muted-foreground">—</span>
          )}
        </TableCell>
        <TableCell>
          {run.studentModel ? (
            <code className="font-mono text-xs">{run.studentModel}</code>
          ) : (
            <span className="text-muted-foreground">—</span>
          )}
        </TableCell>
        <TableCell className="text-right tabular-nums text-sm">
          {run.qualityScore > 0 ? formatPercent(run.qualityScore, true) : '—'}
        </TableCell>
        <TableCell className="text-sm text-muted-foreground">{dur}</TableCell>
      </TableRow>

      {open && (
        <TableRow className="bg-muted/20 hover:bg-muted/20">
          <TableCell colSpan={8} className="px-8 py-2.5">
            <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs sm:grid-cols-4">
              <DetailField label="Run ID" value={run.runId.slice(0, 8)} mono />
              <DetailField label="Outcome" value={formatStatus(run.outcome)} />
              <DetailField
                label="Confidence"
                value={run.confidence > 0 ? formatPercent(run.confidence, true) : '—'}
              />
              <DetailField label="Details" value={run.reason || '—'} />
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  );
}

function DetailField({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <p className="text-muted-foreground">{label}</p>
      <p className={`${mono ? 'font-mono' : ''} truncate`}>{value}</p>
    </div>
  );
}

function TeacherVsStudentCard({
  ts,
}: {
  ts: NonNullable<ModelPerformanceData['teacher_student']>;
}) {
  return (
    <Card className="border-2 border-dashed border-primary/30">
      <CardHeader>
        <div className="flex items-center gap-2">
          <GraduationCap className="size-4 text-muted-foreground" />
          <CardTitle className="text-base">Distilled Model Comparison</CardTitle>
        </div>
        <CardDescription>Teacher vs student model accuracy &amp; cost</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
          <div className="space-y-1.5 rounded-lg border p-4">
            <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Teacher
            </p>
            <p className="text-lg font-semibold">{ts.teacher}</p>
            <p className="text-sm tabular-nums">
              {ts.teacher_accuracy > 0
                ? `${formatPercent(ts.teacher_accuracy, true)} accuracy`
                : 'Accuracy not profiled yet'}
            </p>
            <p className="text-sm tabular-nums text-muted-foreground">
              {ts.teacher_cost > 0 ? `$${ts.teacher_cost}/1k tokens` : '—'}
            </p>
          </div>
          <div className="space-y-1.5 rounded-lg border p-4">
            <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Student
            </p>
            <p className="text-lg font-semibold">{ts.student}</p>
            <Badge variant="secondary" className="mt-1 text-xs">
              distilled
            </Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function TrainingActivitySection({
  training,
  selectedDays,
}: {
  training: TrainingActivityData;
  selectedDays: number;
}) {
  const kpis = training.kpis as Record<string, unknown>;
  const advisorStatus = kpis.advisor_status as
    | { recommendation: string; confidence: number }
    | undefined;

  // Pivot signal trends: { date, error_rate_increase, drift_ratio, high_severity_issues }
  const signalChartData = useMemo(() => {
    const map = new Map<string, Record<string, number | string>>();
    for (const s of training.signal_trends) {
      const dateKey = s.date.split('T')[0];
      if (!map.has(dateKey)) map.set(dateKey, { date: dateKey });
      const entry = map.get(dateKey)!;
      entry[s.signal] = s.value;
    }
    return Array.from(map.values());
  }, [training.signal_trends]);

  const hasActivity =
    training.training_history.length > 0 ||
    signalChartData.length > 0 ||
    training.advisor_decisions.length > 0 ||
    training.training_cycles.length > 0;

  if (!hasActivity) return null;

  return (
    <>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricCard
          label="Training Runs"
          value={String(kpis.training_runs ?? 0)}
          icon={Activity}
          tooltip="Total auto-training runs triggered by the advisor"
        />
        <MetricCard
          label="Last Training"
          value={
            kpis.last_training
              ? new Date(kpis.last_training as string).toLocaleDateString()
              : 'Never'
          }
          icon={Clock}
          tooltip="Date of the most recent training run"
        />
        <MetricCard
          label="Advisor Status"
          value={
            advisorStatus?.recommendation && advisorStatus.recommendation !== 'none'
              ? advisorStatus.recommendation
              : 'Idle'
          }
          icon={Brain}
          subtitle={
            advisorStatus && advisorStatus.confidence > 0
              ? `${(advisorStatus.confidence * 100).toFixed(0)}% confidence`
              : 'No active recommendation'
          }
          tooltip="Current recommendation from the training advisor"
        />
        <MetricCard
          label="Models Updated"
          value={Number(kpis.models_updated ?? 0) > 0 ? String(kpis.models_updated) : '—'}
          icon={Users}
          tooltip="Number of models updated via auto-training"
        />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {training.training_history.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Training History</CardTitle>
              <CardDescription>Promoted vs rejected runs &middot; {selectedDays}d</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer
                config={{
                  promoted: { label: 'Promoted', color: 'var(--chart-2)' },
                  rejected: { label: 'Rejected', color: 'var(--chart-1)' },
                }}
                className="h-56 w-full"
              >
                <BarChart
                  data={training.training_history.map((h, i) => ({
                    ...h,
                    idx: i + 1,
                    value: 1,
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-border" />
                  <XAxis dataKey="idx" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis hide />
                  <ChartTooltip
                    content={
                      <ChartTooltipContent
                        formatter={(
                          _v: number,
                          _n: string,
                          item: { payload: { promoted: boolean; reason: string; date: string } }
                        ) => [
                          `${item.payload.promoted ? 'Promoted' : 'Rejected'} — ${item.payload.reason}`,
                          item.payload.date,
                        ]}
                      />
                    }
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {training.training_history.map((h, i) => (
                      <Cell key={i} fill={h.promoted ? 'var(--chart-2)' : 'var(--chart-1)'} />
                    ))}
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        )}

        {signalChartData.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Signal Trends</CardTitle>
              <CardDescription>Advisor monitoring signals &middot; {selectedDays}d</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={SIGNAL_CHART_CONFIG} className="h-56 w-full">
                <LineChart data={signalChartData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-border" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={36} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <ChartLegend content={<ChartLegendContent />} />
                  <Line
                    type="monotone"
                    dataKey="error_rate_increase"
                    stroke="var(--chart-1)"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="drift_ratio"
                    stroke="var(--chart-3)"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="high_severity_issues"
                    stroke="var(--chart-4)"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ChartContainer>
            </CardContent>
          </Card>
        )}
      </div>

      {training.advisor_decisions.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <AlertTriangle className="size-4 text-muted-foreground" />
              <CardTitle className="text-base">Recent Advisor Decisions</CardTitle>
            </div>
            <CardDescription>
              Auto-training recommendations based on monitoring signals
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {training.advisor_decisions.slice(0, 10).map((d) => (
              <div key={d.id} className="rounded-lg border p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant={REC_COLORS[d.recommendation] ?? 'secondary'}>
                    {d.recommendation}
                  </Badge>
                  {d.confidence > 0 && (
                    <span className="text-sm tabular-nums text-muted-foreground">
                      {(d.confidence * 100).toFixed(0)}%
                    </span>
                  )}
                  <span className="ml-auto text-xs text-muted-foreground">
                    {d.source} &middot; {new Date(d.timestamp).toLocaleString()}
                  </span>
                </div>
                <p className="mt-1 text-sm">{d.reason}</p>
                {d.signals.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {d.signals.map((s, i) => (
                      <Badge
                        key={i}
                        variant={s.triggered ? 'default' : 'outline'}
                        className="text-xs"
                      >
                        {s.name}:{' '}
                        {typeof s.value === 'number' ? s.value.toFixed(3) : String(s.value)}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {training.training_cycles.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Training Cycle Results</CardTitle>
            <CardDescription>
              Before / After metric comparison for each training cycle
            </CardDescription>
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
                {training.training_cycles.map((c) => (
                  <TableRow key={c.id}>
                    <TableCell className="text-sm">
                      {new Date(c.timestamp).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      <Badge variant={c.promoted ? 'default' : 'secondary'}>
                        {c.promoted ? 'Promoted' : 'Rejected'}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right tabular-nums text-sm">
                      {c.baseline.auroc?.toFixed(4) ?? '—'} &rarr;{' '}
                      {c.new_metrics.auroc?.toFixed(4) ?? '—'}
                    </TableCell>
                    <TableCell className="text-right tabular-nums text-sm">
                      {c.baseline.win_rate ? `${(c.baseline.win_rate * 100).toFixed(1)}%` : '—'}{' '}
                      &rarr;{' '}
                      {c.new_metrics.win_rate
                        ? `${(c.new_metrics.win_rate * 100).toFixed(1)}%`
                        : '—'}
                    </TableCell>
                    <TableCell className="text-right tabular-nums text-sm">
                      {c.baseline.apgr ? `${(c.baseline.apgr * 100).toFixed(1)}%` : '—'} &rarr;{' '}
                      {c.new_metrics.apgr ? `${(c.new_metrics.apgr * 100).toFixed(1)}%` : '—'}
                    </TableCell>
                    <TableCell className="max-w-[200px] truncate text-sm text-muted-foreground">
                      {c.reason}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </>
  );
}

function ChartEmpty({ icon: Icon, message }: { icon: React.ElementType; message: string }) {
  return (
    <div className="flex h-48 flex-col items-center justify-center gap-3">
      <div className="flex size-10 items-center justify-center rounded-full bg-muted">
        <Icon className="size-4 text-muted-foreground" />
      </div>
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  );
}

interface Stats {
  total: number;
  completed: number;
  failed: number;
  running: number;
  successRate: number;
  avgQuality: number;
  totalCost: number;
}

function buildStats(runs: TrainingRunDetail[], summary: DistillationSummary | null): Stats {
  const completed = summary?.completed_jobs ?? runs.filter((r) => r.status === 'completed').length;
  const failed = summary?.failed_jobs ?? runs.filter((r) => r.status === 'failed').length;
  const running = summary?.running_jobs ?? runs.filter((r) => r.status === 'running').length;
  const total = summary?.total_jobs ?? runs.length;
  const successRate = total > 0 ? completed / total : 0;

  const scored = runs.filter((r) => r.qualityScore > 0);
  const avgQuality =
    scored.length > 0 ? scored.reduce((s, r) => s + r.qualityScore, 0) / scored.length : 0;

  return {
    total,
    completed,
    failed,
    running,
    successRate,
    avgQuality,
    totalCost: summary?.total_training_cost ?? 0,
  };
}

function buildModelPieData(
  runs: TrainingRunDetail[],
  field: 'teacherModel' | 'studentModel'
): { items: PieItem[]; config: ChartConfig; total: number } {
  const map = new Map<string, number>();
  for (const r of runs) {
    const name = r[field];
    if (!name) continue;
    map.set(name, (map.get(name) ?? 0) + 1);
  }
  const sorted = Array.from(map.entries()).sort((a, b) => b[1] - a[1]);
  const items: PieItem[] = sorted.map(([name, count], i) => ({
    name,
    count,
    fill: CHART_COLORS[i % CHART_COLORS.length],
  }));
  const config: ChartConfig = {};
  for (const item of items) {
    config[item.name] = { label: item.name, color: item.fill };
  }
  const total = items.reduce((s, d) => s + d.count, 0);
  return { items, config, total };
}

function humanDuration(raw: string): string {
  if (!raw || raw === '0s' || raw === '—') return '—';
  return raw;
}
