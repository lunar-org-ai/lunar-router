import { useState, useCallback } from 'react';
import { Database } from 'lucide-react';
import { TracesTableHeader } from './TracesTableHeader';
import { TracesTableRow } from './TracesTableRow';
import type { TraceItem } from '@/types/analyticsType';
import { Table, TableBody } from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { useClusteringService, type ClusterDataset } from '@/services/clusteringService';
import { toast } from 'sonner';

interface TracesTableProps {
  traces: TraceItem[];
  expandedRows: Record<string, boolean>;
  onToggleExpand: (id: string, e: React.MouseEvent) => void;
  onSelectTrace: (trace: TraceItem) => void;
}

export function TracesTable({
  traces,
  expandedRows,
  onToggleExpand,
  onSelectTrace,
}: TracesTableProps) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [datasets, setDatasets] = useState<ClusterDataset[]>([]);
  const [showPicker, setShowPicker] = useState(false);
  const [assigning, setAssigning] = useState(false);
  const service = useClusteringService();

  const toggleSelect = useCallback((id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  const toggleAll = useCallback(() => {
    if (selectedIds.size === traces.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(traces.map(t => t.id)));
    }
  }, [selectedIds.size, traces]);

  const handleOpenPicker = useCallback(async () => {
    try {
      const data = await service.listDatasets();
      setDatasets(data.datasets || []);
      setShowPicker(true);
    } catch {
      toast.error('Failed to load datasets');
    }
  }, [service]);

  const handleAssign = useCallback(async (ds: ClusterDataset) => {
    setAssigning(true);
    try {
      const ids = Array.from(selectedIds);
      const result = await service.assignTracesToDataset(ds.run_id, ds.cluster_id, ids);
      toast.success(`Added ${result.assigned} traces to "${ds.domain_label}"`);
      setSelectedIds(new Set());
      setShowPicker(false);
    } catch {
      toast.error('Failed to assign traces');
    }
    setAssigning(false);
  }, [selectedIds, service]);

  return (
    <div className="space-y-0">
      {/* Selection bar */}
      {selectedIds.size > 0 && (
        <div className="flex items-center gap-3 px-4 py-2 bg-primary/5 border border-primary/20 rounded-t-lg">
          <span className="text-sm font-medium">{selectedIds.size} selected</span>
          <Button size="sm" variant="outline" onClick={handleOpenPicker}>
            <Database className="size-3.5" />
            Add to Dataset
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setSelectedIds(new Set())}>
            Clear
          </Button>
        </div>
      )}

      {/* Dataset picker dropdown */}
      {showPicker && (
        <div className="border border-border rounded-lg bg-background shadow-lg p-3 max-h-64 overflow-y-auto">
          <div className="text-xs font-medium text-muted-foreground mb-2">Select a dataset:</div>
          {datasets.length === 0 ? (
            <p className="text-xs text-muted-foreground py-2">No datasets available. Run clustering first.</p>
          ) : (
            <div className="space-y-1">
              {datasets.map(ds => (
                <button
                  key={`${ds.run_id}-${ds.cluster_id}`}
                  className="w-full text-left px-3 py-2 rounded-md hover:bg-muted text-sm flex items-center justify-between"
                  onClick={() => handleAssign(ds)}
                  disabled={assigning}
                >
                  <div>
                    <span className="font-medium">{ds.domain_label}</span>
                    <span className="text-muted-foreground ml-2">{ds.trace_count} traces</span>
                  </div>
                  <Badge variant={ds.status === 'qualified' ? 'default' : 'secondary'} className="text-xs">
                    {ds.status}
                  </Badge>
                </button>
              ))}
            </div>
          )}
          <Button size="sm" variant="ghost" className="w-full mt-2" onClick={() => setShowPicker(false)}>
            Cancel
          </Button>
        </div>
      )}

      <div className="overflow-hidden rounded-lg border">
        <Table>
          <TracesTableHeader onToggleAll={toggleAll} allSelected={selectedIds.size === traces.length && traces.length > 0} />
          <TableBody>
            {traces.map((trace, index) => (
              <TracesTableRow
                key={trace.id}
                trace={trace}
                index={index}
                isExpanded={expandedRows[trace.id]}
                isSelected={selectedIds.has(trace.id)}
                onToggleSelect={toggleSelect}
                onToggleExpand={onToggleExpand}
                onSelect={onSelectTrace}
              />
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
