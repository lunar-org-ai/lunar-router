import { Plus, Database } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { PageHeader } from '@/components/shared/PageHeader';
import { SearchBar } from '@/components/shared/SearchBar';
import { ListEmpty } from '@/features/evaluations/components/shared/ListEmpty';
import { ListSkeleton } from '@/features/evaluations/components/shared/ListSkeleton';
import { ItemGroup } from '@/components/ui/item';
import { SOURCE_OPTIONS } from '../../constants';
import { useDatasetsPage } from '../../hooks/UseDatasetsPage';
import { DatasetRow } from './DatasetRow';
import { CreateDatasetModal } from '../CreateDatasetModal/CreateDatasetModal';
import { useTutorialStep } from '@/components/Tutorial';

export default function DatasetsPage() {
  const page = useDatasetsPage();

  useTutorialStep(3, !page.isEmpty);

  return (
    <div>
      <PageHeader
        title="Distill Datasets"
        action={
          <Button size="sm" onClick={() => page.setShowCreateModal(true)}>
            <Plus className="size-4" />
            Create Dataset
          </Button>
        }
      />

      <div className="mx-auto max-w-6xl px-6 py-6 space-y-6">
        <div className="space-y-3">
          <SearchBar
            value={page.searchTerm}
            onChange={page.setSearchTerm}
            placeholder="Search datasets…"
            resultCount={page.filteredDatasets.length}
            resultCountPosition="outside"
            filters={
              <Select value={page.sourceFilter} onValueChange={page.setSourceFilter}>
                <SelectTrigger size="sm" className="w-fit min-w-32">
                  <SelectValue placeholder="Source" />
                </SelectTrigger>
                <SelectContent>
                  {SOURCE_OPTIONS.map(({ value, label }) => (
                    <SelectItem key={value} value={value}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            }
          />

          {page.datasetsLoading ? (
            <ListSkeleton showCreateButton />
          ) : page.filteredDatasets.length === 0 ? (
            <ListEmpty
              isEmpty={page.isEmpty}
              icon={<Database />}
              emptyTitle="No datasets yet"
              emptyDescription="Create a dataset manually, from traces, or describe a topic and let AI find matching data."
              noResultsTitle="No matching datasets"
              noResultsDescription="Try adjusting your search or filters."
              createLabel="Create Dataset"
              onCreateClick={page.isEmpty ? () => page.setShowCreateModal(true) : undefined}
            />
          ) : (
            <ScrollArea className="max-h-[calc(100vh-280px)]">
              <ItemGroup className="space-y-1.5 pr-2">
                {page.filteredDatasets.map((dataset) => (
                  <DatasetRow
                    key={dataset.id}
                    dataset={dataset}
                    onClick={() => page.navigateToDataset(dataset.id)}
                    onDelete={page.handleDelete}
                    isDeleting={page.deletingDatasetId === dataset.id}
                  />
                ))}
              </ItemGroup>
            </ScrollArea>
          )}
        </div>
      </div>

      <CreateDatasetModal
        open={page.showCreateModal}
        onClose={page.handleCloseCreateModal}
        onCreate={page.handleCreate}
        onCreateFromTopic={page.handleCreateFromTopic}
        onImport={page.handleImport}
        onAnalyzeTraces={page.handleAnalyzeTraces}
        onImportTraces={page.handleImportTraces}
        traces={page.traces}
        tracesLoading={page.tracesLoading}
        onCreateFromTraces={page.handleCreateFromTraces}
        clusters={page.clusters}
        clustersLoading={page.clustersLoading}
        onCreateFromCluster={page.handleCreateFromCluster}
        onTriggerClustering={page.handleTriggerClustering}
      />
    </div>
  );
}
