import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';

import { AppLayout } from '@/app/AppLayout';
import { FullScreenSpinner } from '@/components/shared/FullScreenSpinner';

const Intelligence = lazy(() => import('@/features/intelligence'));
const Traces = lazy(() => import('@/views/Traces'));
const DataSources = lazy(() => import('@/views/DataSources'));
const Production = lazy(() => import('@/features/production/Production'));
const DistillDatasets = lazy(() => import('@/features/distill-dataset'));
const DatasetDetail = lazy(
  () => import('@/features/distill-dataset/components/DatasetDetail/DatasetDetailPage')
);
const Evaluations = lazy(() => import('@/features/evaluations'));
const DistillJobs = lazy(() => import('@/views/DistillJobs'));
const Harness = lazy(() => import('@/features/harness/HarnessPage'));
const NewDistillationJob = lazy(() => import('@/views/NewDistillationJob'));
const DistillationJobView = lazy(() => import('@/views/DistillationJobView'));
const DistillationResults = lazy(() => import('@/views/DistillationResults'));

export function AppRoutes() {
  return (
    <Suspense fallback={<FullScreenSpinner />}>
      <Routes>
        <Route index element={<Navigate to="/traces" replace />} />

        <Route element={<AppLayout />}>
          <Route path="traces" element={<Traces />} />
          <Route path="intelligence" element={<Intelligence />} />
          <Route path="observability" element={<Navigate to="/intelligence" replace />} />
          <Route path="router-intelligence" element={<Navigate to="/intelligence" replace />} />
          <Route path="data-sources" element={<DataSources />} />
          <Route path="distill-datasets" element={<DistillDatasets />} />
          <Route path="distill-datasets/:datasetId" element={<DatasetDetail />} />
          <Route path="distill-datasets/:datasetId/:tab" element={<DatasetDetail />} />
          <Route path="distill-jobs" element={<DistillJobs />} />
          <Route path="distill-new" element={<NewDistillationJob />} />
          <Route path="distill-job/:jobId" element={<DistillationJobView />} />
          <Route path="distill-job/:jobId/results" element={<DistillationResults />} />
          <Route path="distill-metrics" element={<Evaluations />} />
          <Route path="production" element={<Production />} />
          <Route path="deployments" element={<Navigate to="/production" replace />} />
          <Route path="harness" element={<Harness />} />
        </Route>

        <Route path="*" element={<Navigate to="/traces" replace />} />
      </Routes>
    </Suspense>
  );
}
