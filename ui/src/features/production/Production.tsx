import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Beaker, Server } from 'lucide-react';

import { PageHeader } from '@/components/shared/PageHeader';
import { PageTabs, type PageTab } from '@/components/shared/PageTabs';
import type { DistillationJob } from '@/types/distillationTypes';
import { useDistillation } from '@/hooks/useDistillation';

import { useProductionModels } from './hooks/useProductionModels';
import { useDeploymentActions } from './hooks/useDeploymentActions';

import { StatsBar } from './components/StatsBar';
import { DeploymentTab } from './components/ModelsTab';
import { DistilledTab } from './components/DistilledTab';
import { DistilledDeployDialog } from './components/DistilledTab/DistilledDeployDialog';
import {
  DeployProgressDialog,
  type DeployProgressState,
} from './components/ModelsTab/DeploymentProgress/DeployProgressDialog';

type TabId = 'deployments' | 'distilled';

const TABS: PageTab<TabId>[] = [
  { id: 'deployments', label: 'Active Models', icon: <Server /> },
  { id: 'distilled', label: 'Distilled Models', icon: <Beaker /> },
];

export default function Production() {
  const [activeTab, setActiveTab] = useState<TabId>('distilled');
  const [searchTerm, setSearchTerm] = useState('');

  // Deploy progress dialog state
  const [progressState, setProgressState] = useState<DeployProgressState>(null);
  const [isProgressOpen, setIsProgressOpen] = useState(false);
  const [progressAlreadyDeployed, setProgressAlreadyDeployed] = useState(false);

  const navigate = useNavigate();

  const models = useProductionModels();
  const actions = useDeploymentActions({ allModels: models.allModels });
  const { listDeployments } = actions;

  const distillation = useDistillation();
  const [deployingJobIds, setDeployingJobIds] = useState<Set<string>>(new Set());
  const [deployModalJobId, setDeployModalJobId] = useState<string | null>(null);
  const [enrichedJobs, setEnrichedJobs] = useState<Map<string, DistillationJob>>(new Map());
  const enrichingRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    listDeployments();
  }, [listDeployments]);

  const filteredDeployments = useMemo(() => {
    if (!searchTerm) return actions.deployments;
    const query = searchTerm.toLowerCase();
    return actions.deployments.filter((deployment) => {
      const modelName =
        models.allModels.find((m) => m.id === deployment.selectedModel)?.name ??
        deployment.selectedModel;
      return modelName.toLowerCase().includes(query);
    });
  }, [actions.deployments, searchTerm, models.allModels]);

  // Distilled models: completed jobs
  const completedJobs = useMemo(
    () => distillation.jobs.filter((j) => j.status === 'completed'),
    [distillation.jobs]
  );

  // Enrich completed jobs with full config data when distilled tab is active
  useEffect(() => {
    if (activeTab !== 'distilled') return;

    const jobsToEnrich = completedJobs.filter(
      (j) =>
        !enrichedJobs.has(j.id) &&
        !enrichingRef.current.has(j.id) &&
        (!j.config.student_model || !j.config.teacher_model)
    );

    if (jobsToEnrich.length === 0) return;

    jobsToEnrich.forEach((j) => enrichingRef.current.add(j.id));

    Promise.all(
      jobsToEnrich.map(async (j) => {
        const full = await distillation.getJob(j.id);
        if (full) {
          setEnrichedJobs((prev) => new Map(prev).set(j.id, full));
        }
        enrichingRef.current.delete(j.id);
      })
    );
  }, [activeTab, completedJobs, enrichedJobs, distillation]);

  const resolvedCompletedJobs = useMemo(
    () => completedJobs.map((j) => enrichedJobs.get(j.id) ?? j),
    [completedJobs, enrichedJobs]
  );

  const filteredCompletedJobs = useMemo(() => {
    if (!searchTerm) return resolvedCompletedJobs;
    const query = searchTerm.toLowerCase();
    return resolvedCompletedJobs.filter((job) => job.name.toLowerCase().includes(query));
  }, [resolvedCompletedJobs, searchTerm]);

  const isSearchDisabled =
    activeTab === 'deployments'
      ? actions.deployments.length === 0
      : completedJobs.length === 0;

  const closeProgressDialog = useCallback(() => {
    setIsProgressOpen(false);
    setProgressState(null);
    setProgressAlreadyDeployed(false);
  }, []);

  const handleOpenDeployDialog = useCallback((jobId: string) => {
    setDeployModalJobId(jobId);
  }, []);

  const handleConfirmDeploy = useCallback(
    async (instanceType: string) => {
      const jobId = deployModalJobId;
      if (!jobId) return;
      setDeployModalJobId(null);

      setDeployingJobIds((prev) => new Set(prev).add(jobId));
      setProgressState('creating');
      setProgressAlreadyDeployed(false);
      setIsProgressOpen(true);

      try {
        const result = await distillation.deployJob(jobId, instanceType);
        if (result) {
          setProgressAlreadyDeployed(!!result.already_deployed);
          setProgressState('success');
          setActiveTab('deployments');
          listDeployments();
        } else {
          setProgressState('error');
        }
      } catch {
        setProgressState('error');
      } finally {
        setDeployingJobIds((prev) => {
          const next = new Set(prev);
          next.delete(jobId);
          return next;
        });
      }
    },
    [deployModalJobId, distillation, listDeployments]
  );

  const handleViewResults = useCallback(
    (jobId: string) => {
      navigate(`/distill-job/${jobId}/results`);
    },
    [navigate]
  );

  return (
    <div>
      <PageHeader
        title="Deployments"
        description="Run distilled models locally with llama.cpp"
      />

      <PageTabs tabs={TABS} value={activeTab} onValueChange={setActiveTab} />

      <div className="max-w-6xl mx-auto px-6 py-6">
        <StatsBar
          activeTab={activeTab}
          deployments={actions.deployments}
          totalModelsCount={models.allModels.length}
          downloadingModels={models.inProgressModels}
          readyModels={models.readyModels}
          completedJobsCount={completedJobs.length}
          searchTerm={searchTerm}
          onSearchChange={setSearchTerm}
          isSearchDisabled={isSearchDisabled}
        />

        {activeTab === 'deployments' && (
          <DeploymentTab
            deployments={actions.deployments}
            filteredDeployments={filteredDeployments}
            allModels={models.allModels}
            isLoading={actions.loading}
            deletingDeploymentIds={actions.deletingDeployments}
            pausingDeploymentIds={actions.pausingDeployments}
            resumingDeploymentIds={actions.resumingDeployments}
            searchTerm={searchTerm}
            onBrowseLibrary={() => setActiveTab('distilled')}
            onPause={(d) => actions.handlePause(d.id)}
            onResume={(d) => actions.handleResume(d.id)}
            onDelete={(d) => actions.handleDelete(d.id)}
          />
        )}

        {activeTab === 'distilled' && (
          <DistilledTab
            completedJobs={resolvedCompletedJobs}
            filteredJobs={filteredCompletedJobs}
            isLoading={distillation.loading}
            deployingJobIds={deployingJobIds}
            searchTerm={searchTerm}
            onGoToDistillLab={() => navigate('/distill-jobs')}
            onDeploy={handleOpenDeployDialog}
            onViewResults={handleViewResults}
          />
        )}
      </div>

      <DeployProgressDialog
        isOpen={isProgressOpen}
        state={progressState}
        alreadyDeployed={progressAlreadyDeployed}
        onClose={closeProgressDialog}
      />

      <DistilledDeployDialog
        open={deployModalJobId !== null}
        onOpenChange={(open) => {
          if (!open) setDeployModalJobId(null);
        }}
        jobName={resolvedCompletedJobs.find((j) => j.id === deployModalJobId)?.name ?? ''}
        onDeploy={handleConfirmDeploy}
        isDeploying={deployModalJobId !== null && deployingJobIds.has(deployModalJobId)}
      />
    </div>
  );
}
