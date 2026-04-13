import { useState, useCallback, useEffect } from 'react';
import {
  Search,
  X,
  AlertCircle,
  Loader2,
  Database,
  CheckCircle2,
  BarChart3,
  Sparkles,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { ScrollArea } from '@/components/ui/scroll-area';
import type { ClusterDataset } from '@/services/clusteringService';

interface ClusterModeProps {
  clusters: ClusterDataset[];
  clustersLoading: boolean;
  disabled: boolean;
  onCreateFromCluster: (name: string, runId: string, clusterId: number) => Promise<void>;
  onSuccess: (name: string) => void;
  onTriggerClustering?: (days?: number) => Promise<void>;
}

export function ClusterMode({
  clusters,
  clustersLoading,
  disabled,
  onCreateFromCluster,
  onSuccess,
  onTriggerClustering,
}: ClusterModeProps) {
  const [name, setName] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCluster, setSelectedCluster] = useState<ClusterDataset | null>(null);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isDisabled = disabled || creating;

  const filteredClusters = clusters.filter((c) => {
    if (!searchTerm) return true;
    const q = searchTerm.toLowerCase();
    return (
      c.domain_label.toLowerCase().includes(q) ||
      c.short_description.toLowerCase().includes(q) ||
      c.top_models.some((m) => m.toLowerCase().includes(q))
    );
  });

  // Auto-fill name when selecting a cluster
  useEffect(() => {
    if (selectedCluster && !name) {
      setName(selectedCluster.domain_label);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCluster]);

  const handleCreate = useCallback(async () => {
    setError(null);
    if (!name.trim()) {
      setError('Please enter a dataset name');
      return;
    }
    if (!selectedCluster) {
      setError('Select a cluster');
      return;
    }
    setCreating(true);
    try {
      await onCreateFromCluster(name.trim(), selectedCluster.run_id, selectedCluster.cluster_id);
      onSuccess(name.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create dataset');
      setCreating(false);
    }
  }, [name, selectedCluster, onCreateFromCluster, onSuccess]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Dataset name */}
      <div className="px-6 pt-5">
        <div className="space-y-1.5">
          <Label htmlFor="cluster-dataset-name">Dataset Name</Label>
          <Input
            id="cluster-dataset-name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Customer Support Queries"
            disabled={isDisabled}
          />
        </div>
      </div>

      {/* Search */}
      <div className="flex items-center gap-3 px-6 pt-4">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search clusters..."
            className="h-8 pl-9 pr-9 text-xs"
            disabled={isDisabled}
          />
          {searchTerm && (
            <button
              type="button"
              onClick={() => setSearchTerm('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground transition-colors hover:text-foreground"
            >
              <X className="size-3.5" />
            </button>
          )}
        </div>
        <span className="text-xs text-muted-foreground whitespace-nowrap">
          {filteredClusters.length} cluster{filteredClusters.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Cluster list */}
      <div className="flex-1 min-h-0 px-6 pt-3 pb-2">
        {clustersLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="size-6 animate-spin text-muted-foreground" />
          </div>
        ) : filteredClusters.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 gap-3">
            <Database className="size-8 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              {clusters.length === 0
                ? 'No clusters available yet.'
                : 'No clusters match your search.'}
            </p>
            {clusters.length === 0 && onTriggerClustering && (
              <Button
                type="button"
                size="sm"
                onClick={() => onTriggerClustering()}
                disabled={isDisabled}
              >
                <Sparkles className="size-4" />
                Run Clustering Pipeline
              </Button>
            )}
          </div>
        ) : (
          <ScrollArea className="h-75">
            <div className="space-y-2 pr-2">
              {filteredClusters.map((cluster) => {
                const isSelected =
                  selectedCluster?.run_id === cluster.run_id &&
                  selectedCluster?.cluster_id === cluster.cluster_id;
                return (
                  <button
                    key={`${cluster.run_id}-${cluster.cluster_id}`}
                    type="button"
                    disabled={isDisabled}
                    onClick={() => setSelectedCluster(isSelected ? null : cluster)}
                    className={`w-full text-left rounded-lg border p-3 transition-colors ${
                      isSelected
                        ? 'border-primary bg-primary/5 ring-1 ring-primary'
                        : 'border-border hover:border-muted-foreground/30 hover:bg-muted/50'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="text-sm font-medium truncate">{cluster.domain_label}</span>
                        <Badge
                          variant={cluster.status === 'qualified' ? 'default' : 'secondary'}
                          className="gap-1 shrink-0 text-[10px]"
                        >
                          {cluster.status === 'qualified' && <CheckCircle2 className="size-2.5" />}
                          {cluster.status}
                        </Badge>
                      </div>
                      {isSelected && <CheckCircle2 className="size-4 text-primary shrink-0" />}
                    </div>
                    {cluster.short_description && (
                      <p className="text-xs text-muted-foreground mt-1 line-clamp-1">
                        {cluster.short_description}
                      </p>
                    )}
                    <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <Database className="size-3" />
                        {cluster.trace_count} traces
                      </span>
                      <span className="flex items-center gap-1">
                        <BarChart3 className="size-3" />
                        {(cluster.coherence_score * 100).toFixed(0)}% coherence
                      </span>
                      {cluster.top_models.slice(0, 2).map((m) => (
                        <Badge key={m} variant="outline" className="text-[10px] px-1 py-0">
                          {m}
                        </Badge>
                      ))}
                    </div>
                  </button>
                );
              })}
            </div>
          </ScrollArea>
        )}
      </div>

      {/* Error + Action */}
      <div className="px-6 pb-4 pt-2 border-t space-y-3">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="size-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">
            {selectedCluster
              ? `Selected: ${selectedCluster.domain_label}`
              : 'Select a cluster to create a dataset from'}
          </span>
          <Button
            type="button"
            size="sm"
            onClick={handleCreate}
            disabled={isDisabled || !selectedCluster || !name.trim()}
          >
            {creating ? <Loader2 className="size-4 animate-spin mr-1" /> : null}
            Create from Cluster
          </Button>
        </div>
      </div>
    </div>
  );
}
