import { RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { PageHeader } from '@/components/shared/PageHeader';
import { PageTabs } from '@/components/shared/PageTabs';
import { useRouterIntelligence } from '../hooks/useRouterIntelligence';
import { TABS } from '../constants';
import { EfficiencyTab } from './Efficiency';
import { ModelPerformanceTab } from './ModelPerformance';
import { TrainingActivityTab } from './TrainingActivity';

export default function RouterIntelligencePage() {
  const {
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
  } = useRouterIntelligence();

  return (
    <div className="min-h-screen bg-background">
      <PageHeader
        title="Router Intelligence"
        actions={[
          <Button key="refresh" variant="outline" size="sm" onClick={refreshData} disabled={loading}>
            <RefreshCw className={`mr-2 size-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>,
        ]}
      />

      <PageTabs tabs={TABS} value={activeTab} onValueChange={setActiveTab} />

      <main className="mx-auto max-w-7xl px-6 py-8">
        {activeTab === 'efficiency' && (
          <EfficiencyTab
            data={efficiency}
            loading={loading}
            error={error}
            selectedDays={selectedDays}
            onTimeRangeChange={handleTimeRangeChange}
          />
        )}
        {activeTab === 'model-performance' && (
          <ModelPerformanceTab data={models} loading={loading} error={error} />
        )}
        {activeTab === 'training' && (
          <TrainingActivityTab
            data={training}
            loading={loading}
            error={error}
            selectedDays={selectedDays}
            onTimeRangeChange={handleTimeRangeChange}
          />
        )}
      </main>
    </div>
  );
}
