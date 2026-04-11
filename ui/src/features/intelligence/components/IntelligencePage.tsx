import { useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { createElement } from 'react';
import {
  RefreshCw,
  BarChart3,
  DollarSign,
  GraduationCap,
  Layers,
  Route,
  Activity,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { PageHeader } from '@/components/shared/PageHeader';
import { PageTabs } from '@/components/shared/PageTabs';
import type { PageTab } from '@/components/shared/PageTabs';
import { useIntelligenceData } from '../hooks/useIntelligenceData';
import type { IntelligenceTabId, Period } from '../types';
import { OverviewTab } from './OverviewTab';
import { CostAnalysisTab } from './CostAnalysisTab';
import { PerformanceTab } from './PerformanceTab';
import { ModelsTab } from './ModelsTab';
import { RoutingIntelligenceTab } from './RoutingIntelligenceTab';

const TABS: PageTab<IntelligenceTabId>[] = [
  { id: 'overview', label: 'Overview', icon: createElement(BarChart3, { className: 'size-4' }) },
  { id: 'costs', label: 'Cost Analysis', icon: createElement(DollarSign, { className: 'size-4' }) },
  {
    id: 'distillation',
    label: 'Distillation',
    icon: createElement(GraduationCap, { className: 'size-4' }),
  },
  { id: 'models', label: 'Models', icon: createElement(Layers, { className: 'size-4' }) },
  {
    id: 'routing',
    label: 'Routing Intelligence',
    icon: createElement(Route, { className: 'size-4' }),
  },
];

const VALID_TABS = new Set<string>([
  'overview',
  'costs',
  'distillation',
  'training',
  'models',
  'routing',
]);
const VALID_PERIODS = new Set<string>(['7d', '14d', '30d']);

function parseTab(value: string | null): IntelligenceTabId {
  if (value && VALID_TABS.has(value)) return value as IntelligenceTabId;
  return 'overview';
}

function parsePeriod(value: string | null): Period {
  if (value && VALID_PERIODS.has(value)) return value as Period;
  return '30d';
}

export default function IntelligencePage() {
  const [searchParams, setSearchParams] = useSearchParams();

  const activeTab = parseTab(searchParams.get('tab'));
  const period = parsePeriod(searchParams.get('period'));

  const setActiveTab = useCallback(
    (tab: IntelligenceTabId) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set('tab', tab);
        return next;
      });
    },
    [setSearchParams]
  );

  const setPeriod = useCallback(
    (p: Period) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set('period', p);
        return next;
      });
    },
    [setSearchParams]
  );

  const data = useIntelligenceData(period);

  return (
    <div className="min-h-screen bg-background">
      <PageHeader
        title="Intelligence & Observability"
        actions={[
          <PeriodSelector key="period" value={period} onChange={setPeriod} />,
          <Tooltip key="refresh">
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                onClick={data.refreshData}
                disabled={data.loading}
                className="gap-1.5"
              >
                <RefreshCw className={`size-3.5 ${data.loading ? 'animate-spin' : ''}`} />
                <span className="hidden sm:inline">Refresh</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent>Refresh all metrics</TooltipContent>
          </Tooltip>,
        ]}
      />

      <PageTabs tabs={TABS} value={activeTab} onValueChange={setActiveTab} />

      <main className="mx-auto max-w-350 px-6 py-6">
        {activeTab === 'overview' && <OverviewTab data={data} />}
        {activeTab === 'costs' && <CostAnalysisTab data={data} />}
        {activeTab === 'distillation' && <PerformanceTab data={data} />}
        {activeTab === 'training' && <TrainingTab data={data} />}
        {activeTab === 'models' && <ModelsTab data={data} />}
        {activeTab === 'routing' && <RoutingIntelligenceTab data={data} />}
      </main>
    </div>
  );
}

function PeriodSelector({ value, onChange }: { value: Period; onChange: (p: Period) => void }) {
  const options: Period[] = ['7d', '14d', '30d'];

  return (
    <div className="inline-flex items-center gap-0.5 rounded-lg border p-0.5">
      {options.map((opt) => (
        <Button
          key={opt}
          size="sm"
          variant={value === opt ? 'default' : 'ghost'}
          className={`h-7 px-3 text-xs font-medium ${
            value === opt ? '' : 'text-muted-foreground hover:text-foreground'
          }`}
          onClick={() => onChange(opt)}
        >
          {opt}
        </Button>
      ))}
    </div>
  );
}
