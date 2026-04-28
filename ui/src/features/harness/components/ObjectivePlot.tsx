/**
 * ObjectivePlot — time-series line of an objective's value with action
 * markers (signals, decisions, actions) rendered as clickable dots.
 *
 * The plot is the primary surface for answering "why did X move on
 * Tuesday?" per the Step 4 acceptance test: the line shows the
 * movement; the markers show what the harness did about it; clicking a
 * marker opens the chain drawer with the full causal descent.
 */

import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  useHarnessService,
  type LedgerEntry,
  type Objective,
  type ObjectiveMeasurementPoint,
  type ObjectiveTimeSeries,
} from '@/services/harnessService';
import { Skeleton } from '@/components/ui/skeleton';
import { markerColorFor } from './ledgerFormat';

interface ObjectivePlotProps {
  objective: Objective;
  hours?: number;
  onMarkerClick?: (entryId: string) => void;
}

interface MarkerPoint {
  t: number;
  y: number;
  marker: LedgerEntry;
  color: string;
}

function toEpoch(ts: string): number {
  const n = new Date(ts).getTime();
  return Number.isFinite(n) ? n : 0;
}

function formatTick(t: number): string {
  // Short form: "Mon 14:00"
  const d = new Date(t);
  return d.toLocaleDateString(undefined, { weekday: 'short' }) +
    ' ' +
    d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

function precisionFor(unit: string): number {
  if (unit === 'USD') return 4;
  if (unit === 'ratio') return 3;
  if (unit === 'ms') return 0;
  return 2;
}

export function ObjectivePlot({
  objective,
  hours = 168,
  onMarkerClick,
}: ObjectivePlotProps) {
  const service = useHarnessService();
  const [series, setSeries] = useState<ObjectiveTimeSeries | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    service.getObjectiveTimeSeries(objective.id, hours).then((data) => {
      setSeries(data);
      setLoading(false);
    });
  }, [service, objective.id, hours]);

  const { lineData, markerPoints, yDomain } = useMemo(() => {
    if (!series) {
      return {
        lineData: [] as { t: number; value: number }[],
        markerPoints: [] as MarkerPoint[],
        yDomain: undefined as [number, number] | undefined,
      };
    }

    const line = series.measurements
      .map((m: ObjectiveMeasurementPoint) => ({ t: toEpoch(m.ts), value: m.value }))
      .filter((p) => p.t > 0)
      .sort((a, b) => a.t - b.t);

    // Y-domain the chart explicitly so markers can sit at `yMin` (below
    // the line) without clipping. Fall back to target/baseline when no
    // measurements exist so the plot still frames something useful.
    const values = line.map((p) => p.value);
    if (objective.target !== null) values.push(objective.target);
    if (objective.baseline !== null) values.push(objective.baseline);

    const min = values.length ? Math.min(...values) : 0;
    const max = values.length ? Math.max(...values) : 1;
    const pad = (max - min) * 0.1 || Math.abs(max) * 0.1 || 1;
    const yMin = min - pad;
    const yMax = max + pad;

    // Markers sit just above yMin so they live "on the x-axis" per the
    // plan, without overlapping the line.
    const markerY = yMin + pad * 0.3;
    const markers: MarkerPoint[] = series.markers
      .map((m) => ({
        t: toEpoch(m.ts),
        y: markerY,
        marker: m,
        color: markerColorFor(m),
      }))
      .filter((p) => p.t > 0)
      .sort((a, b) => a.t - b.t);

    return {
      lineData: line,
      markerPoints: markers,
      yDomain: [yMin, yMax] as [number, number],
    };
  }, [series, objective.baseline, objective.target]);

  if (loading) {
    return <Skeleton className="h-48 w-full" />;
  }

  if (lineData.length === 0 && markerPoints.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-xs text-muted-foreground">
        No data yet. The trigger engine populates this once it starts ticking.
      </div>
    );
  }

  const precision = precisionFor(objective.unit);

  return (
    <div className="h-48 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="2 3" vertical={false} />
          <XAxis
            type="number"
            dataKey="t"
            domain={['dataMin', 'dataMax']}
            tickFormatter={formatTick}
            tick={{ fontSize: 10 }}
            stroke="currentColor"
            allowDuplicatedCategory={false}
          />
          <YAxis
            type="number"
            domain={yDomain}
            tick={{ fontSize: 10 }}
            tickFormatter={(v: number) => v.toFixed(precision)}
            stroke="currentColor"
            width={50}
          />

          {objective.target !== null && (
            <ReferenceLine
              y={objective.target}
              stroke="#10b981"
              strokeDasharray="3 3"
              label={{ value: 'target', fontSize: 10, position: 'right', fill: '#10b981' }}
            />
          )}
          {objective.baseline !== null && (
            <ReferenceLine
              y={objective.baseline}
              stroke="#64748b"
              strokeDasharray="3 3"
              label={{ value: 'baseline', fontSize: 10, position: 'right', fill: '#64748b' }}
            />
          )}

          <Tooltip
            contentStyle={{
              fontSize: 11,
              padding: '4px 8px',
              background: 'hsl(var(--background))',
              border: '1px solid hsl(var(--border))',
              borderRadius: 6,
            }}
            labelFormatter={(l) => formatTick(Number(l))}
            formatter={(v: number | string) => {
              const n = typeof v === 'number' ? v : Number(v);
              return [Number.isFinite(n) ? n.toFixed(precision) : v, objective.unit];
            }}
          />

          <Line
            data={lineData}
            type="monotone"
            dataKey="value"
            dot={false}
            stroke="hsl(var(--primary))"
            strokeWidth={2}
            isAnimationActive={false}
            name={objective.id}
          />

          <Scatter
            data={markerPoints}
            dataKey="y"
            fill="#f59e0b"
            shape={(props: { cx?: number; cy?: number; payload?: MarkerPoint }) => {
              const { cx, cy, payload } = props;
              if (cx === undefined || cy === undefined || !payload) return <g />;
              return (
                <g
                  style={{ cursor: 'pointer' }}
                  onClick={() => onMarkerClick?.(payload.marker.id)}
                >
                  <circle cx={cx} cy={cy} r={5} fill={payload.color} stroke="white" strokeWidth={1.5} />
                </g>
              );
            }}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
