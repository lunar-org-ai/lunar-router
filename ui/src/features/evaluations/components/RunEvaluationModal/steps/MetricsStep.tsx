import { useState } from 'react';
import { Check, Sparkles, Loader2 } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { FieldLegend, FieldSet } from '@/components/ui/field';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import { useEvaluationsService } from '../../../api/evaluationsService';
import type { EvaluationMetric } from '../../../types';

interface MetricsStepProps {
  builtin: EvaluationMetric[];
  custom: EvaluationMetric[];
  selectedMetrics: string[];
  datasetId?: string;
  onToggle: (metricId: string) => void;
  onSuggestedMetricsCreated?: () => void;
}

export function MetricsStep({ builtin, custom, selectedMetrics, datasetId, onToggle, onSuggestedMetricsCreated }: MetricsStepProps) {
  const service = useEvaluationsService();
  const [suggesting, setSuggesting] = useState(false);
  const [suggested, setSuggested] = useState(false);

  const handleSuggest = async () => {
    if (!datasetId) return;
    setSuggesting(true);
    try {
      const result = await service.suggestMetrics('no-auth', datasetId);
      const suggestions = result.suggestions || [];
      // Auto-select the suggested metrics
      for (const s of suggestions) {
        if (s.metric_id && !selectedMetrics.includes(s.metric_id)) {
          onToggle(s.metric_id);
        }
      }
      setSuggested(true);
      onSuggestedMetricsCreated?.();
    } catch (err) {
      console.error('[MetricsStep] Suggest failed:', err);
    } finally {
      setSuggesting(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold mb-1">Select Metrics</h3>
          <p className="text-sm text-muted-foreground">
            Choose which metrics to use for scoring model responses.{' '}
            {selectedMetrics.length > 0 && (
              <Badge variant="secondary" className="ml-1">
                {selectedMetrics.length} selected
              </Badge>
            )}
          </p>
        </div>
        {datasetId && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={handleSuggest}
            disabled={suggesting || suggested}
          >
            {suggesting ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Sparkles className="size-3.5" />
            )}
            {suggesting ? 'Analyzing...' : suggested ? 'Suggested' : 'Suggest with AI'}
          </Button>
        )}
      </div>

      {builtin.length > 0 && (
        <FieldSet>
          <FieldLegend variant="label">Built-in</FieldLegend>
          <div className="flex flex-wrap gap-2">
            {builtin.map((metric) => (
              <MetricToggle
                key={metric.metric_id}
                metric={metric}
                isSelected={selectedMetrics.includes(metric.metric_id)}
                onToggle={onToggle}
              />
            ))}
          </div>
        </FieldSet>
      )}

      {builtin.length > 0 && custom.length > 0 && <Separator />}

      {custom.length > 0 && (
        <FieldSet>
          <FieldLegend variant="label">
            <span className="flex items-center gap-1.5">
              AI Suggested
              <Sparkles className="size-3 text-amber-500" />
            </span>
          </FieldLegend>
          <div className="flex flex-wrap gap-2">
            {custom.map((metric) => (
              <MetricToggle
                key={metric.metric_id}
                metric={metric}
                isSelected={selectedMetrics.includes(metric.metric_id)}
                onToggle={onToggle}
              />
            ))}
          </div>
        </FieldSet>
      )}
    </div>
  );
}

interface MetricToggleProps {
  metric: EvaluationMetric;
  isSelected: boolean;
  onToggle: (metricId: string) => void;
}

function MetricToggle({ metric, isSelected, onToggle }: MetricToggleProps) {
  return (
    <Button
      type="button"
      variant={isSelected ? 'default' : 'outline'}
      size="sm"
      onClick={() => onToggle(metric.metric_id)}
      className={cn('transition-all', isSelected && 'shadow-sm')}
    >
      {isSelected && <Check className="size-3.5" />}
      {metric.name}
    </Button>
  );
}
