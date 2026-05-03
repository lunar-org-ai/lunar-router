import { motion } from 'framer-motion';

type MetricTileProps = {
  label: string;
  value: string;
  hint?: string;
  delay?: number;
};

export function MetricTile({ label, value, hint, delay = 0 }: MetricTileProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay, ease: [0.16, 1, 0.3, 1] }}
      className="flex flex-col gap-1.5 rounded-xl border border-border/40 bg-card/30 px-4 py-3"
    >
      <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
        {label}
      </span>
      <span className="text-xl font-medium tracking-tight tabular-nums">{value}</span>
      {hint ? (
        <span className="font-mono text-[11px] text-muted-foreground/60">{hint}</span>
      ) : null}
    </motion.div>
  );
}
