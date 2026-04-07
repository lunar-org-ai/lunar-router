import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';

import { AppLayout } from '@/app/AppLayout';
import { FullScreenSpinner } from '@/components/shared/FullScreenSpinner';

const Observability = lazy(() => import('@/features/observability'));
const Traces = lazy(() => import('@/views/Traces'));
const DataSources = lazy(() => import('@/views/DataSources'));
const DistillDatasets = lazy(() => import('@/features/distill-dataset'));
const DatasetDetail = lazy(
  () => import('@/features/distill-dataset/components/DatasetDetail/DatasetDetailPage')
);
const Evaluations = lazy(() => import('@/features/evaluations'));
const DistillJobs = lazy(() => import('@/views/DistillJobs'));
const NewDistillationJob = lazy(() => import('@/views/NewDistillationJob'));
const DistillationJobView = lazy(() => import('@/views/DistillationJobView'));
const DistillationResults = lazy(() => import('@/views/DistillationResults'));
const RouterIntelligence = lazy(() => import('@/features/router-intelligence'));

export function AppRoutes() {
  return (
    <Suspense fallback={<FullScreenSpinner />}>
      <Routes>
        <Route index element={<Navigate to="traces" replace />} />

        <Route element={<AppLayout />}>
          <Route path="traces" element={<Traces />} />
          <Route path="observability" element={<Observability />} />
          <Route path="data-sources" element={<DataSources />} />
          <Route path="distill-datasets" element={<DistillDatasets />} />
          <Route path="distill-datasets/:datasetId" element={<DatasetDetail />} />
          <Route path="distill-datasets/:datasetId/:tab" element={<DatasetDetail />} />
          <Route path="distill-jobs" element={<DistillJobs />} />
          <Route path="distill-new" element={<NewDistillationJob />} />
          <Route path="distill-job/:jobId" element={<DistillationJobView />} />
          <Route path="distill-job/:jobId/results" element={<DistillationResults />} />
          <Route path="distill-metrics" element={<Evaluations />} />
          <Route path="router-intelligence" element={<RouterIntelligence />} />
        </Route>

        <Route path="*" element={<Navigate to="traces" replace />} />
      </Routes>
    </Suspense>
  );
}
