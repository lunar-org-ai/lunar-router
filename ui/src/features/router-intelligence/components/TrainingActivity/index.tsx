import type { TrainingActivityData } from '../../types';
import { TrainingActivityContent } from './TrainingActivityContent';

interface Props {
  data: TrainingActivityData | null;
  loading: boolean;
  error: string | null;
  selectedDays: number;
  onTimeRangeChange: (days: number) => void;
}

export function TrainingActivityTab({ data, loading, error, selectedDays, onTimeRangeChange }: Props) {
  if (loading) return <div className="py-12 text-center text-muted-foreground">Loading training data...</div>;
  if (error) return <div className="py-12 text-center text-destructive">Error: {error}</div>;
  if (!data) return <div className="py-12 text-center text-muted-foreground">No data available.</div>;
  return <TrainingActivityContent data={data} selectedDays={selectedDays} onTimeRangeChange={onTimeRangeChange} />;
}
