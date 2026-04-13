import type { LucideIcon } from 'lucide-react';
import { KpiCard } from '@/components/shared/KpiCard';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

interface MetricCardProps {
  label: string;
  value: string;
  icon?: LucideIcon;
  subtitle?: string;
  tooltip?: string;
}

export function MetricCard({ tooltip, ...props }: MetricCardProps) {
  const card = <KpiCard {...props} />;
  if (!tooltip) return <div className="h-full">{card}</div>;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="h-full">{card}</div>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-64">
        <p className="text-xs">{tooltip}</p>
      </TooltipContent>
    </Tooltip>
  );
}
