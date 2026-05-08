export const Sparkline = ({
  data,
  w = 240,
  h = 56,
  accent = 'var(--primary)',
  fill = true,
}: {
  data: number[];
  w?: number;
  h?: number;
  accent?: string;
  fill?: boolean;
}) => {
  if (!data || !data.length) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const step = w / (data.length - 1);
  const pts = data.map((v, i) => [i * step, h - ((v - min) / range) * (h - 8) - 4] as const);
  const path = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ');
  const area = `${path} L${w} ${h} L0 ${h} Z`;
  const last = pts[pts.length - 1];
  return (
    <svg className="sparkline" width={w} height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      {fill && <path d={area} fill={accent} fillOpacity="0.08" />}
      <path d={path} stroke={accent} strokeWidth="1.5" fill="none" strokeLinejoin="round" />
      <circle cx={last[0]} cy={last[1]} r="3" fill={accent} />
    </svg>
  );
};
