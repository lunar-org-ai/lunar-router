/**
 * Cluster Datasets page — matches pureai-interface DatasetsPage UX.
 * List layout with ItemRow, ScrollArea, ListEmpty/ListSkeleton.
 */

import { useState, type MouseEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Database,
  Download,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  Sparkles,
  RefreshCw,
  ChevronRight,
  BarChart3,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { PageHeader } from '@/components/shared/PageHeader';
import { SearchBar } from '@/components/shared/SearchBar';
import { ListEmpty } from '@/features/evaluations/components/shared/ListEmpty';
import { ListSkeleton } from '@/features/evaluations/components/shared/ListSkeleton';
import { ItemGroup } from '@/components/ui/item';
import { ItemRow } from '@/components/shared/ItemRow';
import { useClusterDatasets } from '../hooks/useClusterDatasets';
import type { ClusterDataset } from '@/services/clusteringService';

// --- Status badge (same style as SourceBadge) ---

const STATUS_CONFIG = {
  qualified: { icon: CheckCircle2, label: 'Qualified', variant: 'default' as const },
  candidate: { icon: Clock, label: 'Review', variant: 'secondary' as const },
  rejected: { icon: XCircle, label: 'Rejected', variant: 'outline' as const },
};

function StatusBadge({ status }: { status: string }) {
  const config = STATUS_CONFIG[status as keyof typeof STATUS_CONFIG] || STATUS_CONFIG.candidate;
  const Icon = config.icon;
  return (
    <Badge variant={config.variant} className="gap-1 font-medium">
      <Icon className="size-3" />
      {config.label}
    </Badge>
  );
}

// --- Dataset row (same pattern as DatasetRow) ---

function ClusterDatasetRow({
  dataset,
  onExport,
  onQualify,
  onClick,
}: {
  dataset: ClusterDataset;
  onExport: (id: number) => void;
  onQualify: (id: number, status: 'qualified' | 'rejected') => void;
  onClick: () => void;
}) {
  const descriptionParts: React.ReactNode[] = [
    <span key="traces" className="inline-flex items-center gap-1 tabular-nums">
      <Database className="size-3 shrink-0" />
      {dataset.trace_count.toLocaleString()} traces
    </span>,
    <span key="coherence" className="inline-flex items-center gap-1 tabular-nums">
      <BarChart3 className="size-3 shrink-0" />
      {(dataset.coherence_score * 100).toFixed(0)}% coherent
    </span>,
    dataset.top_models.length > 0 && (
      <span key="models" className="tabular-nums">
        {dataset.top_models.slice(0, 2).join(', ')}
      </span>
    ),
    dataset.avg_cost_usd > 0 && (
      <span key="cost" className="tabular-nums">
        $
        {dataset.avg_cost_usd < 0.01
          ? dataset.avg_cost_usd.toFixed(5)
          : dataset.avg_cost_usd.toFixed(3)}
        /req
      </span>
    ),
  ].filter(Boolean) as React.ReactNode[];

  return (
    <ItemRow
      name={dataset.domain_label || `Cluster ${dataset.cluster_id}`}
      badge={<StatusBadge status={dataset.status} />}
      descriptionParts={descriptionParts}
      onClick={onClick}
      actions={
        <>
          {dataset.status === 'candidate' && (
            <Button
              variant="ghost"
              size="icon"
              className="size-8 opacity-0 group-hover/item:opacity-100"
              onClick={(e: MouseEvent) => {
                e.stopPropagation();
                onQualify(dataset.cluster_id, 'qualified');
              }}
              aria-label="Qualify"
            >
              <CheckCircle2 className="size-4 text-green-500" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="size-8 opacity-0 group-hover/item:opacity-100"
            onClick={(e: MouseEvent) => {
              e.stopPropagation();
              onExport(dataset.cluster_id);
            }}
            aria-label="Export JSONL"
          >
            <Download className="size-4" />
          </Button>
          <Button variant="ghost" size="icon" className="size-8" aria-label="Open">
            <ChevronRight className="size-4" />
          </Button>
        </>
      }
    />
  );
}

// --- Filter options ---

const STATUS_OPTIONS = [
  { value: 'active', label: 'Active' },
  { value: 'all', label: 'All' },
  { value: 'qualified', label: 'Qualified' },
  { value: 'candidate', label: 'Review' },
  { value: 'rejected', label: 'Rejected' },
];

// --- Page ---

export default function ClusterDatasetsPage() {
  const page = useClusterDatasets();
  const navigate = useNavigate();
  const [clusterDays] = useState(30);

  // "active" = hide rejected (default view)
  const visibleDatasets =
    page.statusFilter === 'active'
      ? page.filteredDatasets.filter((d) => d.status !== 'rejected')
      : page.filteredDatasets;

  const isEmpty = page.datasets.length === 0 && !page.loading;

  return (
    <div>
      <PageHeader
        title="Domain Datasets"
        action={
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={page.refreshDatasets}
              disabled={page.loading}
            >
              <RefreshCw className="size-4" />
            </Button>
            <Button
              size="sm"
              onClick={() => page.triggerClustering(clusterDays)}
              disabled={page.running}
            >
              {page.running ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Sparkles className="size-4" />
              )}
              {page.running ? 'Clustering...' : 'Cluster from Traces'}
            </Button>
          </div>
        }
      />

      <div className="mx-auto max-w-6xl px-6 py-6 space-y-6">
        <div className="space-y-3">
          <SearchBar
            value={page.searchTerm}
            onChange={page.setSearchTerm}
            placeholder="Search datasets..."
            resultCount={visibleDatasets.length}
            resultCountPosition="outside"
            filters={
              <select
                value={page.statusFilter}
                onChange={(e) => page.setStatusFilter(e.target.value)}
                className="h-8 rounded-md border bg-background px-2 text-sm"
              >
                {STATUS_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            }
          />

          {page.loading ? (
            <ListSkeleton showCreateButton />
          ) : visibleDatasets.length === 0 ? (
            <ListEmpty
              isEmpty={isEmpty}
              icon={<Database />}
              emptyTitle="No datasets yet"
              emptyDescription="Click 'Cluster from Traces' to automatically group your production traffic into domain-specific datasets."
              noResultsTitle="No matching datasets"
              noResultsDescription="Try adjusting your search or filter."
              createLabel="Cluster from Traces"
              onCreateClick={isEmpty ? () => page.triggerClustering(clusterDays) : undefined}
            />
          ) : (
            <ScrollArea className="max-h-[calc(100vh-280px)]">
              <ItemGroup className="space-y-1.5 pr-2">
                {visibleDatasets.map((dataset) => (
                  <ClusterDatasetRow
                    key={`${dataset.run_id}-${dataset.cluster_id}`}
                    dataset={dataset}
                    onExport={page.exportDataset}
                    onQualify={page.qualifyDataset}
                    onClick={() =>
                      navigate(`/distill-datasets/${dataset.run_id}/${dataset.cluster_id}`)
                    }
                  />
                ))}
              </ItemGroup>
            </ScrollArea>
          )}
        </div>
      </div>
    </div>
  );
}
