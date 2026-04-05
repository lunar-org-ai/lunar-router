import type { ModelPerformanceData } from '../../types';
import { ModelPerformanceContent } from './ModelPerformanceContent';

interface Props {
  data: ModelPerformanceData | null;
  loading: boolean;
  error: string | null;
}

export function ModelPerformanceTab({ data, loading, error }: Props) {
  if (loading) return <div className="py-12 text-center text-muted-foreground">Loading model data...</div>;
  if (error) return <div className="py-12 text-center text-destructive">Error: {error}</div>;
  if (!data) return <div className="py-12 text-center text-muted-foreground">No data available.</div>;
  return <ModelPerformanceContent data={data} />;
}
