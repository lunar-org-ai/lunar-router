import { useEffect, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useDatasets } from '@/hooks/useDatasets';
import { useDatasetSamples } from './useDatasetSamples';
import type { Dataset, ViewTab } from '../types';
import type { BondConfig } from '../components/BondEnhancementModal';

export function useDatasetDetailPage() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const navigate = useNavigate();
  const { getDataset, deleteDataset } = useDatasets();

  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [pageLoading, setPageLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<ViewTab>('general');
  const [showBondModal, setShowBondModal] = useState(false);

  const samples = useDatasetSamples({ datasetId });

  useEffect(() => {
    if (!datasetId) return;
    setPageLoading(true);
    // Metadata-only — the samples are loaded separately by useDatasetSamples
    // so there's no reason to fetch them twice. Opt out explicitly now that
    // the server defaults to hydrating samples.
    getDataset(datasetId, { include_samples: false })
      .then((result) => {
        if (result) {
          setDataset({
            id: datasetId,
            name: (result as any).dataset?.name ?? (result as any).name ?? 'Dataset',
            description: (result as any).dataset?.description ?? (result as any).description,
            source: (result as any).dataset?.source ?? (result as any).source,
            samples_count: (result as any).dataset?.samples_count ?? result.samples?.length ?? 0,
            created_at: (result as any).dataset?.created_at ?? (result as any).created_at,
            updated_at: (result as any).dataset?.updated_at ?? (result as any).updated_at,
          });
        }
      })
      .finally(() => setPageLoading(false));
  }, [datasetId, getDataset]);

  const handleExportJSON = useCallback(() => {
    if (!dataset) return;
    const data = samples.filteredSamples.map((s) => ({
      input: s.input,
      output: s.expected_output || s.output,
    }));
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${dataset.name.replace(/\s+/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [dataset, samples.filteredSamples]);

  const handleBondEnhance = useCallback(async (config: BondConfig) => {
    console.log('BOND Enhancement config:', config);
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }, []);

  const handleDelete = useCallback(async () => {
    if (!datasetId) return;
    const success = await deleteDataset(datasetId);
    if (success) navigate('/distill-datasets');
  }, [datasetId, deleteDataset, navigate]);

  return {
    dataset,
    pageLoading,
    activeTab,
    setActiveTab,
    showBondModal,
    setShowBondModal,
    samples,
    handleExportJSON,
    handleBondEnhance,
    handleDelete,
    navigate,
  };
}
