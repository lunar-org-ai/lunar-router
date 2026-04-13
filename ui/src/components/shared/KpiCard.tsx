import { TrendingDown, TrendingUp, type LucideIcon } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import {
  Card,
  CardAction,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';

interface KpiCardProps {
  label: string;
  value: string;
  icon?: LucideIcon;
  subtitle?: string;
  change?: string;
  isPositive?: boolean;
  className?: string;
}

export function KpiCard({
  label,
  value,
  icon: Icon,
  subtitle,
  change,
  isPositive = true,
}: KpiCardProps) {
  const hasValidChange = Boolean(change && !change.includes('N/A'));
  const TrendIcon = isPositive ? TrendingUp : TrendingDown;

  return (
    <Card className="@container/card h-full">
      <CardHeader>
        <div className="flex items-center gap-1">
          {Icon && <Icon className="size-4 text-muted-foreground" />}
          <CardDescription>{label}</CardDescription>
        </div>
        <CardTitle className="text-xl font-semibold tabular-nums @[250px]/card:text-2xl">
          {value}
        </CardTitle>
        {hasValidChange && (
          <CardAction>
            <Badge variant="outline">
              <TrendIcon />
              {isPositive ? '+' : ''}
              {change}
            </Badge>
          </CardAction>
        )}
      </CardHeader>
      {subtitle && (
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          <div className="line-clamp-1 flex gap-2 font-medium">{subtitle}</div>
        </CardFooter>
      )}
    </Card>
  );
}
