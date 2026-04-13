import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { formatCost } from '@/utils/formatUtils';
import type { CostBreakdown } from '@/features/router-intelligence/types';

interface BaselineComparisonProps {
  cb: CostBreakdown;
  variant?: 'full' | 'compact';
  title?: string;
  description?: string;
}

export function BaselineComparison({
  cb,
  variant = 'full',
  title,
  description,
}: BaselineComparisonProps) {
  if (variant === 'compact') {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{title ?? 'Baseline Model Comparison'}</CardTitle>
          <CardDescription>
            {description ?? 'Router actual cost vs always-cheapest vs always-best'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-3">
            <StatCell
              label="Router (Actual)"
              value={formatCost(cb.routing_actual)}
              description="Smart routing with trained model"
            />
            <StatCell
              label="Always Best Model"
              value={formatCost(cb.provider_baseline)}
              description="Most accurate, most expensive"
            />
            <StatCell
              label="Savings"
              value={formatCost(cb.routing_savings)}
              description={`${
                cb.provider_baseline > 0
                  ? ((cb.routing_savings / cb.provider_baseline) * 100).toFixed(1) + '%'
                  : '0.0%'
              } reduction`}
              highlight={cb.routing_savings > 0}
            />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title ?? 'Baseline Model Comparison'}</CardTitle>
        <CardDescription>
          {description ?? 'Cost comparison: smart routing vs always using the most expensive model'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <StatCell
            label="Provider Baseline"
            value={formatCost(cb.provider_baseline)}
            description="All reqs at most expensive model"
            accentColor="var(--chart-1)"
          />
          <StatCell
            label="Actual Router Cost"
            value={formatCost(cb.routing_actual)}
            description="Cost with smart routing"
            accentColor="var(--chart-2)"
          />
          <StatCell
            label="Net Savings"
            value={formatCost(cb.net_savings)}
            description={cb.roi_pct > 0 ? `${cb.roi_pct.toFixed(0)}% ROI` : 'vs baseline'}
            accentColor="var(--chart-3)"
            highlight={cb.net_savings > 0}
          />
          <StatCell
            label="Monthly Projection"
            value={`${formatCost(cb.monthly_projection)}/mo`}
            description="Based on current trend"
          />
        </div>
      </CardContent>
    </Card>
  );
}

function StatCell({
  label,
  value,
  description,
  accentColor,
  highlight,
}: {
  label: string;
  value: string;
  description: string;
  accentColor?: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={`relative overflow-hidden rounded-lg border p-4 ${
        highlight ? 'border-emerald-500/30 bg-emerald-500/5' : ''
      }`}
    >
      {accentColor && (
        <div
          className="absolute left-0 top-0 h-full w-1 rounded-l-lg"
          style={{ backgroundColor: accentColor }}
        />
      )}
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 text-xl font-semibold tabular-nums">{value}</p>
      <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>
    </div>
  );
}
