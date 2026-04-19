import { useMemo } from 'react';
import {
  DollarSign,
  Activity,
  TrendingUp,
  AlertCircle,
  Info,
  AlertTriangle,
  type LucideIcon,
} from 'lucide-react';
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
import { Badge } from '@/components/ui/badge';

import type { OverviewData, KPI } from '../../types';
import { KpiCard } from '@/components/shared/KpiCard';
import { TimeRangeSelector } from '../shared/TimeRangeSelector';
import type { TimeRange } from '../../constants';

interface OverviewContentProps {
  data: OverviewData;
  selectedDays: number;
  onTimeRangeChange: (days: TimeRange) => void;
}

const ICON_MAP: Record<KPI['icon'], LucideIcon> = {
  dollar: DollarSign,
  activity: Activity,
  trending: TrendingUp,
  alert: AlertCircle,
};

const CHART_THEME_COLORS = [
  'var(--chart-1)',
  'var(--chart-2)',
  'var(--chart-3)',
  'var(--chart-4)',
  'var(--chart-5)',
];

const ALERT_STYLES = {
  info: {
    icon: Info,
    bg: 'bg-blue-500/10 dark:bg-blue-500/5',
    border: 'border-blue-500/20',
    iconColor: 'text-blue-500',
  },
  warning: {
    icon: AlertTriangle,
    bg: 'bg-amber-500/10 dark:bg-amber-500/5',
    border: 'border-amber-500/20',
    iconColor: 'text-amber-500',
  },
  error: {
    icon: AlertCircle,
    bg: 'bg-destructive/10 dark:bg-destructive/5',
    border: 'border-destructive/20',
    iconColor: 'text-destructive',
  },
};

export function OverviewContent({ data, selectedDays, onTimeRangeChange }: OverviewContentProps) {
  const externalProviders = data.providers.filter((p) => !p.isOpentracy);

  const pieData = useMemo(
    () =>
      externalProviders.map((p, i) => ({
        provider: `provider-${i}`,
        cost: p.cost,
        fill: `var(--color-provider-${i})`,
      })),
    [externalProviders]
  );

  const totalCost = useMemo(() => pieData.reduce((sum, p) => sum + p.cost, 0), [pieData]);

  const pieConfig = useMemo<ChartConfig>(() => {
    const cfg: ChartConfig = {
      cost: { label: 'Cost' },
    };
    externalProviders.forEach((p, i) => {
      cfg[`provider-${i}`] = {
        label: p.provider,
        color: CHART_THEME_COLORS[i % CHART_THEME_COLORS.length],
      };
    });
    return cfg;
  }, [externalProviders]);

  const barData = useMemo(
    () =>
      [...data.models]
        .sort((a, b) => b.requests - a.requests)
        .slice(0, 8)
        .map((m, i) => ({
          model: `model-${i}`,
          requests: m.requests,
          fill: `var(--color-model-${i})`,
        })),
    [data.models]
  );

  const barConfig = useMemo<ChartConfig>(() => {
    const cfg: ChartConfig = {
      requests: { label: 'Requests' },
    };
    [...data.models]
      .sort((a, b) => b.requests - a.requests)
      .slice(0, 8)
      .forEach((m, i) => {
        cfg[`model-${i}`] = {
          label: m.model || m.name || 'Unknown',
          color: CHART_THEME_COLORS[i % CHART_THEME_COLORS.length],
        };
      });
    return cfg;
  }, [data.models]);

  return (
    <div className="space-y-6">
      <TimeRangeSelector value={selectedDays} onChange={onTimeRangeChange} />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {data.kpis.map((kpi, i) => (
          <KpiCard
            key={i}
            label={kpi.label}
            value={kpi.value}
            icon={ICON_MAP[kpi.icon]}
            change={kpi.change}
            isPositive={kpi.isPositive}
            subtitle={kpi.subtitle}
          />
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card className="flex flex-col">
          <CardHeader className="items-center pb-0">
            <CardTitle>Cost by Provider</CardTitle>
            <CardDescription>External API costs &middot; {selectedDays}d</CardDescription>
            <CardAction>
              <Badge variant="secondary" className="tabular-nums">
                ${totalCost.toFixed(2)} total
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent className="flex-1 p-0">
            {pieData.length > 0 ? (
              <ChartContainer config={pieConfig} className="mx-auto aspect-square max-h-80">
                <PieChart>
                  <ChartTooltip
                    cursor={false}
                    content={
                      <ChartTooltipContent
                        formatter={(value: number) => [`$${Number(value).toFixed(2)}`, '']}
                      />
                    }
                  />
                  <Pie data={pieData} dataKey="cost" nameKey="provider" innerRadius={70} />
                  <ChartLegend
                    content={<ChartLegendContent nameKey="provider" />}
                    className="-translate-y-2 flex-wrap gap-2 *:basis-1/4 *:justify-center"
                  />
                </PieChart>
              </ChartContainer>
            ) : (
              <div className="flex h-56 flex-col items-center justify-center gap-2 rounded-lg border border-dashed">
                <DollarSign className="size-8 text-muted-foreground/40" />
                <p className="text-sm text-muted-foreground">No external provider costs</p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="overflow-hidden">
          <CardHeader>
            <CardTitle>Usage by Model</CardTitle>
            <CardDescription>Total requests &middot; {selectedDays}d</CardDescription>
            <CardAction>
              <Badge variant="secondary" className="tabular-nums">
                {barData.reduce((s, m) => s + m.requests, 0).toLocaleString()} total
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent>
            {barData.length > 0 ? (
              <ChartContainer config={barConfig}>
                <BarChart accessibilityLayer data={barData} layout="vertical" margin={{ left: 0 }}>
                  <YAxis
                    dataKey="model"
                    type="category"
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(value) =>
                      barConfig[value as keyof typeof barConfig]?.label as string
                    }
                    width={115}
                  />
                  <XAxis dataKey="requests" type="number" hide />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                  <Bar dataKey="requests" radius={5}>
                    <LabelList
                      dataKey="requests"
                      position="right"
                      formatter={(value: string | number) => Number(value).toLocaleString()}
                      className="fill-foreground text-xs"
                    />
                  </Bar>
                </BarChart>
              </ChartContainer>
            ) : (
              <div className="flex h-56 flex-col items-center justify-center gap-2 rounded-lg border border-dashed">
                <Activity className="size-8 text-muted-foreground/40" />
                <p className="text-sm text-muted-foreground">No model usage data</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {data.alerts.length > 0 && (
        <Card className="overflow-hidden">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <div className="flex size-6 items-center justify-center rounded-md bg-chart-2/10">
                <Info className="size-3.5 text-chart-2" />
              </div>
              <CardTitle className="text-sm font-medium">Insights & Alerts</CardTitle>
              <Badge variant="outline" className="ml-auto text-[10px]">
                {data.alerts.length}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            {data.alerts.map((alert, i) => {
              const style = ALERT_STYLES[alert.type];
              const AlertIcon = style.icon;
              return (
                <div
                  key={i}
                  className={`flex items-start gap-3 rounded-lg border p-3 ${style.bg} ${style.border}`}
                >
                  <div className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-background">
                    <AlertIcon className={`size-3 ${style.iconColor}`} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm leading-relaxed">{alert.message}</p>
                    {alert.timestamp && (
                      <p className="mt-1 text-xs text-muted-foreground">{alert.timestamp}</p>
                    )}
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
