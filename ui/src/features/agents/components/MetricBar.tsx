import { motion } from 'framer-motion';

import { cn } from '@/lib/utils';
import type { EvalMetric } from '@/features/agents/types';

type MetricBarProps = {
  metric: EvalMetric;
  index?: number;
};

type Resolved = {
  width: number;
  label: string;
  textColor: string;
  barColor: string;
};

function resolveMetric(value: EvalMetric['value']): Resolved {
  if (value === 'pass') {
    return {
      width: 100,
      label: 'Pass ✓',
      textColor: 'text-emerald-500',
      barColor: 'bg-emerald-500',
    };
  }
  if (value === 'fail') {
    return {
      width: 100,
      label: 'Fail ✕',
      textColor: 'text-rose-500',
      barColor: 'bg-rose-500',
    };
  }
  if (value >= 80) {
    return {
      width: value,
      label: `${value}%`,
      textColor: 'text-emerald-500',
      barColor: 'bg-emerald-500',
    };
  }
  if (value >= 50) {
    return {
      width: value,
      label: `${value}%`,
      textColor: 'text-amber-500',
      barColor: 'bg-amber-500',
    };
  }
  return {
    width: value,
    label: `${value}%`,
    textColor: 'text-rose-500',
    barColor: 'bg-rose-500',
  };
}

export function MetricBar({ metric, index = 0 }: MetricBarProps) {
  const resolved = resolveMetric(metric.value);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <span className="text-sm text-foreground">{metric.name}</span>
        <span className={cn('font-mono text-xs', resolved.textColor)}>{resolved.label}</span>
      </div>
      <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-muted/40">
        <motion.div
          className={cn('absolute inset-y-0 left-0 rounded-full', resolved.barColor)}
          initial={{ width: 0 }}
          animate={{ width: `${resolved.width}%` }}
          transition={{ delay: index * 0.06, duration: 0.5, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
}
