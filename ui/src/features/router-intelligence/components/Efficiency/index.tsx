import type { EfficiencyData } from '../../types';
import { EfficiencyContent } from './EfficiencyContent';

interface Props {
  data: EfficiencyData | null;
  loading: boolean;
  error: string | null;
  selectedDays: number;
  onTimeRangeChange: (days: number) => void;
}

export function EfficiencyTab({ data, loading, error, selectedDays, onTimeRangeChange }: Props) {
  if (loading) return <div className="py-12 text-center text-muted-foreground">Loading efficiency data...</div>;
  if (error) return <div className="py-12 text-center text-destructive">Error: {error}</div>;
  if (!data) return <div className="py-12 text-center text-muted-foreground">No data available.</div>;
  return <EfficiencyContent data={data} selectedDays={selectedDays} onTimeRangeChange={onTimeRangeChange} />;
}
