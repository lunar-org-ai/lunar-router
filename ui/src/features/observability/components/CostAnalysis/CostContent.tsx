import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { DollarSign, TrendingUp, Zap, ArrowUpRight, Sparkles, ChevronDown } from 'lucide-react';
import { Area, AreaChart, Bar, BarChart, CartesianGrid, LabelList, XAxis, YAxis } from 'recharts';

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
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@/components/ui/collapsible';
import { Item, ItemContent, ItemDescription, ItemTitle } from '@/components/ui/item';

import { formatCost } from '@/utils/formatUtils';
import type { CostAnalysisData } from '../../types';
import { KpiCard } from '@/components/shared/KpiCard';
import { TimeRangeSelector } from '../shared/TimeRangeSelector';
import type { TimeRange } from '../../constants';
import { Button } from '@/components/ui/button';

interface CostContentProps {
  data: CostAnalysisData;
  selectedDays: number;
  onTimeRangeChange: (days: TimeRange) => void;
}

const costChartConfig: ChartConfig = {
  cost: { label: 'Cost', color: 'var(--color-chart-2)' },
};

export function CostContent({ data, selectedDays, onTimeRangeChange }: CostContentProps) {
  const navigate = useNavigate();
  const [insightsOpen, setInsightsOpen] = useState(true);

  const totals = useMemo(() => {
    const externalCosts = data.externalCosts || [];
    const totalCost = externalCosts.reduce((sum, item) => sum + item.cost, 0);
    const projected = totalCost * (30 / selectedDays);
    const avgDaily = totalCost / selectedDays;
    return { totalCost, projected, avgDaily };
  }, [data.externalCosts, selectedDays]);

  const barData = useMemo(
    () =>
      (data.externalCosts || []).slice(0, 6).map((item, i) => ({
        task: `task-${i}`,
        cost: item.cost,
      })),
    [data.externalCosts]
  );

  const maxBarCost = useMemo(() => Math.max(...barData.map((item) => item.cost), 0), [barData]);
  const barDomainMax = useMemo(() => (maxBarCost > 0 ? maxBarCost * 1.15 : 1), [maxBarCost]);

  const barConfig = useMemo<ChartConfig>(() => {
    const cfg: ChartConfig = {
      cost: { label: 'Cost' },
    };

    (data.externalCosts || []).slice(0, 6).forEach((item, i) => {
      cfg[`task-${i}`] = {
        label: item.task,
        color: 'var(--chart-2)',
      };
    });

    return cfg;
  }, [data.externalCosts]);

  const externalRequests = data.expensiveRequests.filter((r) => !r.isOpentracy);

  return (
    <div className="space-y-6">
      <TimeRangeSelector value={selectedDays} onChange={onTimeRangeChange} />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <KpiCard
          label={`Total Cost (${selectedDays}d)`}
          value={formatCost(totals.totalCost)}
          icon={DollarSign}
        />
        <KpiCard
          label="Monthly Projection"
          value={formatCost(totals.projected)}
          icon={TrendingUp}
        />
        <KpiCard label="Avg Daily Cost" value={formatCost(totals.avgDaily)} icon={Zap} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-7">
        <Card className="overflow-hidden lg:col-span-4">
          <CardHeader>
            <CardTitle>Cost Over Time</CardTitle>
            <CardDescription>Daily spending trend</CardDescription>
            <CardAction>
              <Badge variant="secondary" className="tabular-nums">
                {formatCost(totals.avgDaily)}/day avg
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent>
            {data.timeSeries.length > 0 ? (
              <ChartContainer config={costChartConfig} className="h-72 w-full">
                <AreaChart data={data.timeSeries}>
                  <CartesianGrid
                    vertical={false}
                    strokeDasharray="3 3"
                    className="stroke-muted/50"
                  />
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
                    <linearGradient id="fillCostImproved" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--color-chart-2)" stopOpacity={0.3} />
                      <stop offset="50%" stopColor="var(--color-chart-2)" stopOpacity={0.1} />
                      <stop offset="100%" stopColor="var(--color-chart-2)" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <Area
                    dataKey="cost"
                    type="monotone"
                    fill="url(#fillCostImproved)"
                    stroke="var(--color-chart-2)"
                    strokeWidth={2}
                  />
                </AreaChart>
              </ChartContainer>
            ) : (
              <div className="flex h-56 flex-col items-center justify-center gap-2 rounded-lg border border-dashed">
                <TrendingUp className="size-8 text-muted-foreground/40" />
                <p className="text-sm text-muted-foreground">No time series data</p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="overflow-hidden lg:col-span-3">
          <CardHeader>
            <CardTitle>Cost by Model</CardTitle>
            <CardDescription>External API costs only</CardDescription>
            <CardAction>
              <Badge variant="secondary" className="tabular-nums">
                {formatCost(totals.totalCost)} total
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent>
            {barData.length > 0 ? (
              <ChartContainer config={barConfig}>
                <BarChart accessibilityLayer data={barData} layout="vertical" margin={{ left: 0 }}>
                  <YAxis
                    dataKey="task"
                    type="category"
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(value) =>
                      barConfig[value as keyof typeof barConfig]?.label as string
                    }
                    width={82}
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
                  <Bar dataKey="cost" radius={5} fill="var(--chart-1)">
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
              <div className="flex h-52 flex-col items-center justify-center gap-2 rounded-lg border border-dashed">
                <DollarSign className="size-8 text-muted-foreground/40" />
                <p className="text-sm text-muted-foreground">No cost data</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="overflow-hidden">
        <CardHeader>
          <CardTitle>Most Expensive Requests</CardTitle>
          <CardDescription>Click a row to view the trace</CardDescription>
        </CardHeader>
        <CardContent className="pl-10 pr-10">
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
                      <Badge variant="secondary">{formatCost(req.cost)}</Badge>
                    </TableCell>
                    <TableCell className="font-medium">{req.model}</TableCell>
                    <TableCell className="tabular-nums text-muted-foreground">
                      {req.promptSize.toLocaleString()}
                    </TableCell>
                    <TableCell className="text-muted-foreground">{req.date}</TableCell>
                    <TableCell className="text-right">
                      <Button variant="ghost" size="icon" className="h-8 px-2">
                        <ArrowUpRight />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="flex h-32 flex-col items-center justify-center gap-2 rounded-lg border border-dashed">
              <Sparkles className="size-6 text-muted-foreground/40" />
              <p className="text-sm text-muted-foreground">No expensive requests recorded</p>
            </div>
          )}
        </CardContent>
      </Card>

      <Collapsible open={insightsOpen} onOpenChange={setInsightsOpen}>
        <Card className="overflow-hidden">
          <CollapsibleTrigger asChild>
            <CardHeader className="cursor-pointer transition-colors">
              <CardTitle>Cost Optimization Insights</CardTitle>
              <CardDescription>Recommendations to reduce external API spend</CardDescription>
              <CardAction>
                <Button variant="ghost" size="icon">
                  <ChevronDown />
                </Button>
              </CardAction>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent className="grid grid-cols-1 gap-3 md:grid-cols-2">
              <Item variant="outline">
                <ItemContent>
                  <ItemTitle>Optimize Prompts</ItemTitle>
                  <ItemDescription className="mt-1 leading-relaxed">
                    Save approximately{' '}
                    <span className="font-semibold text-foreground">
                      {formatCost(totals.projected * 0.15)}/month
                    </span>{' '}
                    by optimizing prompts and reducing output tokens.
                  </ItemDescription>
                </ItemContent>
              </Item>
              <Item variant="outline">
                <ItemContent>
                  <ItemTitle>Billing Note</ItemTitle>
                  <ItemDescription className="mt-1 leading-relaxed">
                    External API costs shown. OpenTracy deployments are billed hourly and not included
                    here.
                  </ItemDescription>
                </ItemContent>
              </Item>
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>
    </div>
  );
}
