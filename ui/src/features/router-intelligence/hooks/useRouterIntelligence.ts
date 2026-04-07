import { useState, useCallback, useEffect } from 'react';
import type { TabId, EfficiencyData, ModelPerformanceData, TrainingActivityData } from '../types';
import { fetchEfficiencyData, fetchModelPerformanceData, fetchTrainingActivityData } from '../api/routerIntelligenceService';

export function useRouterIntelligence() {
  const [activeTab, setActiveTab] = useState<TabId>('efficiency');
  const [selectedDays, setSelectedDays] = useState(30);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [efficiency, setEfficiency] = useState<EfficiencyData | null>(null);
  const [models, setModels] = useState<ModelPerformanceData | null>(null);
  const [training, setTraining] = useState<TrainingActivityData | null>(null);

  const fetchData = useCallback(async (tab: TabId, days: number) => {
    setLoading(true);
    setError(null);
    try {
      switch (tab) {
        case 'efficiency': {
          const data = await fetchEfficiencyData(days);
          setEfficiency(data);
          break;
        }
        case 'model-performance': {
          const data = await fetchModelPerformanceData();
          setModels(data);
          break;
        }
        case 'training': {
          const data = await fetchTrainingActivityData(days);
          setTraining(data);
          break;
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData(activeTab, selectedDays);
  }, [activeTab, selectedDays, fetchData]);

  const refreshData = useCallback(() => {
    fetchData(activeTab, selectedDays);
  }, [activeTab, selectedDays, fetchData]);

  const handleTimeRangeChange = useCallback((days: number) => {
    setSelectedDays(days);
  }, []);

  return {
    activeTab,
    setActiveTab,
    selectedDays,
    loading,
    error,
    efficiency,
    models,
    training,
    refreshData,
    handleTimeRangeChange,
  };
}
