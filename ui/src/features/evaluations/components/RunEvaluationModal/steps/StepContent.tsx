import type { Dataset, EvaluationMetric, AvailableModel } from '../../../types';
import type { RunEvaluationFormData } from '../../../hooks/useRunEvaluationWizard';

import { GeneralStep } from './GeneralStep';
import { ModelsStep } from './ModelsStep';
import { MetricsStep } from './MetricsStep';

type StepId = 'general' | 'models' | 'metrics';

interface StepContentProps {
  stepId: StepId;
  formData: RunEvaluationFormData;
  datasets: Dataset[];
  models: AvailableModel[];
  metrics: { builtin: EvaluationMetric[]; custom: EvaluationMetric[] };
  onFormChange: (partial: Partial<RunEvaluationFormData>) => void;
  onRefreshMetrics?: () => void;
}

export function StepContent({
  stepId,
  formData,
  datasets,
  models,
  metrics,
  onFormChange,
  onRefreshMetrics,
}: StepContentProps) {
  switch (stepId) {
    case 'general':
      return (
        <GeneralStep
          name={formData.name}
          datasetId={formData.datasetId}
          datasets={datasets}
          onNameChange={(name) => onFormChange({ name })}
          onDatasetChange={(datasetId) => onFormChange({ datasetId })}
        />
      );

    case 'models':
      return (
        <ModelsStep
          models={models}
          selectedModels={formData.models}
          onToggle={(modelId) => {
            const next = formData.models.includes(modelId)
              ? formData.models.filter((id) => id !== modelId)
              : [...formData.models, modelId];
            onFormChange({ models: next });
          }}
        />
      );

    case 'metrics':
      return (
        <MetricsStep
          builtin={metrics.builtin}
          custom={metrics.custom}
          selectedMetrics={formData.metrics}
          datasetId={formData.datasetId}
          onToggle={(metricId) => {
            const next = formData.metrics.includes(metricId)
              ? formData.metrics.filter((id) => id !== metricId)
              : [...formData.metrics, metricId];
            onFormChange({ metrics: next });
          }}
          onSuggestedMetricsCreated={onRefreshMetrics}
        />
      );
  }
}
