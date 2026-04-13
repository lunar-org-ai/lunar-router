/**
 * Cluster Dataset detail page — same UX as DatasetDetailPage.
 * Shows traces as samples with SamplesExplorer (table/card views).
 */

import { useState, useRef } from 'react';
import {
  Loader2,
  Database,
  ArrowLeft,
  Download,
  Plus,
  CheckCircle2,
  BarChart3,
  Upload,
  X,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { PageHeader } from '@/components/shared/PageHeader';
import { SamplesExplorer } from './DatasetDetail/SamplesExplorer';
import { useClusterDatasetDetail } from '../hooks/useClusterDatasetDetail';

export default function ClusterDatasetDetailPage() {
  const {
    dataset,
    loading,
    samples,
    handleExportJSON,
    handleAddTraces,
    navigate,
    runId,
    clusterId,
  } = useClusterDatasetDetail();

  const [showAddForm, setShowAddForm] = useState(false);
  const [addInput, setAddInput] = useState('');
  const [addOutput, setAddOutput] = useState('');
  const [addModel, setAddModel] = useState('');
  const [isAdding, setIsAdding] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleAddSingle = async () => {
    if (!addInput.trim()) return;
    setIsAdding(true);
    await handleAddTraces([
      { input: addInput.trim(), output: addOutput.trim(), model: addModel.trim() || undefined },
    ]);
    setAddInput('');
    setAddOutput('');
    setIsAdding(false);
  };

  const handleFileImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsAdding(true);
    try {
      const text = await file.text();
      let traces: any[] = [];
      // Try JSON array
      try {
        const parsed = JSON.parse(text);
        traces = Array.isArray(parsed) ? parsed : parsed.traces || [];
      } catch {
        // JSONL
        traces = text
          .trim()
          .split('\n')
          .map((line) => {
            try {
              return JSON.parse(line);
            } catch {
              return null;
            }
          })
          .filter(Boolean);
      }
      if (traces.length > 0) {
        await handleAddTraces(traces);
      }
    } catch {
      // toast handled in hook
    }
    setIsAdding(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="size-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!dataset) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <div className="size-16 rounded-2xl bg-muted flex items-center justify-center">
          <Database className="size-8 text-muted-foreground" />
        </div>
        <div className="text-center space-y-1">
          <p className="text-sm font-medium text-foreground">Dataset not found</p>
          <p className="text-xs text-muted-foreground">
            The dataset may have been deleted or is unavailable
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => navigate('/distill-datasets')}>
          <ArrowLeft className="size-4" />
          Back to Datasets
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="shrink-0 border-b border-border">
        <div className="px-6 py-4 max-w-6xl mx-auto">
          <div className="flex items-center gap-3 mb-1">
            <Button
              variant="ghost"
              size="icon"
              className="size-8"
              onClick={() => navigate('/distill-datasets')}
            >
              <ArrowLeft className="size-4" />
            </Button>
            <h1 className="text-lg font-semibold">{dataset.domain_label}</h1>
            <Badge
              variant={dataset.status === 'qualified' ? 'default' : 'secondary'}
              className="gap-1"
            >
              {dataset.status === 'qualified' && <CheckCircle2 className="size-3" />}
              {dataset.status}
            </Badge>
          </div>
          {dataset.short_description && (
            <p className="text-sm text-muted-foreground ml-11">{dataset.short_description}</p>
          )}

          {/* Stats bar */}
          <div className="flex items-center gap-6 mt-3 ml-11 text-sm">
            <div className="flex items-center gap-1.5">
              <Database className="size-3.5 text-muted-foreground" />
              <span className="font-medium">{dataset.trace_count}</span>
              <span className="text-muted-foreground">traces</span>
            </div>
            <div className="flex items-center gap-1.5">
              <BarChart3 className="size-3.5 text-muted-foreground" />
              <span className="font-medium">{(dataset.coherence_score * 100).toFixed(0)}%</span>
              <span className="text-muted-foreground">coherence</span>
            </div>
            {dataset.top_models.length > 0 && (
              <div className="flex items-center gap-1.5">
                {dataset.top_models.slice(0, 3).map((m) => (
                  <Badge key={m} variant="outline" className="text-xs">
                    {m}
                  </Badge>
                ))}
              </div>
            )}
            <div className="flex-1" />
            <Button variant="outline" size="sm" onClick={() => setShowAddForm(!showAddForm)}>
              <Plus className="size-4" />
              Add Traces
            </Button>
            <Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()}>
              <Upload className="size-4" />
              Import File
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,.jsonl"
              className="hidden"
              onChange={handleFileImport}
            />
            <Button variant="outline" size="sm" onClick={handleExportJSON}>
              <Download className="size-4" />
              Export JSONL
            </Button>
          </div>

          {/* Rules */}
          {(dataset.inclusion_rule || dataset.exclusion_rule) && (
            <div className="mt-3 ml-11 flex gap-4 text-xs">
              {dataset.inclusion_rule && (
                <span>
                  <span className="text-green-600 font-medium">Include: </span>
                  <span className="text-muted-foreground">{dataset.inclusion_rule}</span>
                </span>
              )}
              {dataset.exclusion_rule && (
                <span>
                  <span className="text-red-600 font-medium">Exclude: </span>
                  <span className="text-muted-foreground">{dataset.exclusion_rule}</span>
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Add Traces Form */}
      {showAddForm && (
        <div className="shrink-0 border-b border-border bg-muted/30">
          <div className="px-6 py-4 max-w-6xl mx-auto">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium">Add a trace manually</span>
              <Button
                variant="ghost"
                size="icon"
                className="size-6"
                onClick={() => setShowAddForm(false)}
              >
                <X className="size-3.5" />
              </Button>
            </div>
            <div className="grid grid-cols-[1fr_1fr_auto_auto] gap-3">
              <textarea
                className="rounded-md border border-input bg-background px-3 py-2 text-sm min-h-[60px] resize-none"
                placeholder="Input / Prompt"
                value={addInput}
                onChange={(e) => setAddInput(e.target.value)}
              />
              <textarea
                className="rounded-md border border-input bg-background px-3 py-2 text-sm min-h-[60px] resize-none"
                placeholder="Output / Response"
                value={addOutput}
                onChange={(e) => setAddOutput(e.target.value)}
              />
              <input
                className="rounded-md border border-input bg-background px-3 py-2 text-sm h-[60px] w-40"
                placeholder="Model (optional)"
                value={addModel}
                onChange={(e) => setAddModel(e.target.value)}
              />
              <Button
                size="sm"
                className="h-[60px]"
                onClick={handleAddSingle}
                disabled={!addInput.trim() || isAdding}
              >
                {isAdding ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Plus className="size-4" />
                )}
                Add
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {samples.loading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="size-8 animate-spin text-muted-foreground" />
          </div>
        ) : samples.filteredSamples.length > 0 || samples.hasActiveFilters ? (
          <div
            className="border-t border-border"
            style={{ height: 'calc(100vh - 260px)', minHeight: '400px' }}
          >
            <SamplesExplorer
              filteredSamples={samples.filteredSamples}
              paginatedSamples={samples.paginatedSamples}
              totalPages={samples.totalPages}
              currentPage={samples.currentPage}
              startIndex={samples.startIndex}
              samplesPerPage={samples.samplesPerPage}
              searchTerm={samples.searchTerm}
              hasActiveFilters={samples.hasActiveFilters}
              expandedRows={samples.expandedRows}
              inputLengthFilter={samples.inputLengthFilter}
              outputLengthFilter={samples.outputLengthFilter}
              onSearchChange={samples.setSearchTerm}
              onPageChange={samples.setCurrentPage}
              onSamplesPerPageChange={samples.setSamplesPerPage}
              onToggleExpand={samples.toggleRowExpand}
              onClearFilters={samples.clearFilters}
              onClearInputFilter={() => samples.setInputLengthFilter(null)}
              onClearOutputFilter={() => samples.setOutputLengthFilter(null)}
            />
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-20 gap-3">
            <div className="size-16 rounded-2xl bg-muted flex items-center justify-center">
              <Database className="size-8 text-muted-foreground" />
            </div>
            <div className="text-center space-y-1">
              <p className="text-sm font-medium text-foreground">No samples yet</p>
              <p className="text-xs text-muted-foreground">
                This cluster has no traces with content
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
