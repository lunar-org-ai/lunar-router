import { cn } from '@/lib/utils';
import type { AgentStatus } from '@/features/agents/types';

type Tone = {
  dot: string;
  text: string;
  label: string;
};

const TONE: Record<AgentStatus, Tone> = {
  healthy: { dot: 'bg-emerald-500', text: 'text-emerald-500', label: 'Healthy' },
  degraded: { dot: 'bg-amber-500', text: 'text-amber-500', label: 'Degraded' },
  silent: { dot: 'bg-muted-foreground/60', text: 'text-muted-foreground', label: 'Silent' },
};

type StatusPillProps = {
  status: AgentStatus;
  className?: string;
};

export function StatusPill({ status, className }: StatusPillProps) {
  const tone = TONE[status];
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider',
        tone.text,
        className
      )}
    >
      <span className={cn('size-1.5 rounded-full', tone.dot)} />
      {tone.label}
    </span>
  );
}
