import { useState, useEffect, useCallback } from 'react';
import { Loader2, Save, Trash2, Play, Clock, Database } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Skeleton } from '@/components/ui/skeleton';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '@/components/ui/table';
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from '@/components/ui/select';
import { useEvaluationsService } from '@/features/evaluations/api/evaluationsService';
import { INTERVAL_OPTIONS } from '../constants';
import type { Dataset, CollectRun, Trace } from '../types';

interface AutoCollectTabProps {
  dataset: Dataset;
}

const ALL_MODELS_VALUE = '__all__';

export function AutoCollectTab({ dataset }: AutoCollectTabProps) {
  const accessToken = 'no-auth';
  const {
    getAutoCollectConfig,
    getAutoCollectHistory,
    putAutoCollectConfig,
    deleteAutoCollectConfig,
    triggerAutoCollect,
    listTraces,
  } = useEvaluationsService();

  const [enabled, setEnabled] = useState(false);
  const [sourceModel, setSourceModel] = useState('');
  const [maxSamples, setMaxSamples] = useState(5000);
  const [traceModels, setTraceModels] = useState<string[]>([]);
  const [intervalMinutes, setIntervalMinutes] = useState(60);
  const [qualityThreshold, setQualityThreshold] = useState(0.5);
  const [selectionRate, setSelectionRate] = useState(0.3);

  const [lastCollectedAt, setLastCollectedAt] = useState<string | null>(null);
  const [totalCollected, setTotalCollected] = useState(0);
  const [configExists, setConfigExists] = useState(false);

  const [history, setHistory] = useState<CollectRun[]>([]);

  const [loadingConfig, setLoadingConfig] = useState(true);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [triggering, setTriggering] = useState(false);

  const loadData = useCallback(async () => {
    if (!accessToken) return;
    setLoadingConfig(true);

    try {
      const [cfg, runs, tracesResult] = await Promise.all([
        getAutoCollectConfig(accessToken, dataset.id),
        getAutoCollectHistory(accessToken, dataset.id, 20).catch(() => []),
        listTraces(accessToken, { limit: 500 }).catch(() => ({ traces: [] as Trace[], total: 0 })),
      ]);

      // Extract unique model IDs from traces
      const models = [...new Set(tracesResult.traces.map((t) => t.model_id))]
        .filter((m) => m && m !== 'unknown')
        .sort();
      setTraceModels(models);

      if (cfg) {
        setConfigExists(true);
        setEnabled(cfg.enabled);
        setSourceModel(cfg.source_model || '');
        setMaxSamples(cfg.max_samples);
        setIntervalMinutes(cfg.collection_interval_minutes);
        setQualityThreshold(cfg.curation_config.quality_threshold);
        setSelectionRate(cfg.curation_config.selection_rate);
        setLastCollectedAt(cfg.last_collected_at || null);
        setTotalCollected(cfg.total_collected);
      } else {
        setConfigExists(false);
      }

      setHistory(runs);
    } catch (err) {
      console.error('[AutoCollectTab] Failed to load data:', err);
    } finally {
      setLoadingConfig(false);
    }
  }, [accessToken, dataset.id, getAutoCollectConfig, getAutoCollectHistory, listTraces]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleSave = async () => {
    if (!accessToken) return;
    setSaving(true);
    try {
      const saved = await putAutoCollectConfig(accessToken, dataset.id, {
        enabled,
        source_model: sourceModel || undefined,
        max_samples: maxSamples,
        collection_interval_minutes: intervalMinutes,
        curation_config: {
          quality_threshold: qualityThreshold,
          selection_rate: selectionRate,
          agent_weights: { quality: 0.4, diversity: 0.3, difficulty: 0.3 },
        },
      });
      setConfigExists(true);
      setLastCollectedAt(saved.last_collected_at || null);
      setTotalCollected(saved.total_collected);
      toast.success('Auto-collect configuration saved');
    } catch {
      toast.error('Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!accessToken) return;
    setDeleting(true);
    try {
      await deleteAutoCollectConfig(accessToken, dataset.id);
      setConfigExists(false);
      setEnabled(false);
      setSourceModel('');
      setMaxSamples(5000);
      setIntervalMinutes(60);
      setQualityThreshold(0.5);
      setSelectionRate(0.3);
      setLastCollectedAt(null);
      setTotalCollected(0);
      toast.success('Auto-collect configuration removed');
    } catch {
      toast.error('Failed to remove configuration');
    } finally {
      setDeleting(false);
    }
  };

  const handleTrigger = async () => {
    if (!accessToken) return;
    setTriggering(true);
    try {
      await triggerAutoCollect(accessToken, dataset.id);
      toast.success('Collection started — results will appear shortly');
    } catch {
      toast.error('Failed to trigger collection');
    } finally {
      setTriggering(false);
    }
  };

  if (loadingConfig) {
    return (
      <ScrollArea className="h-full">
        <div className="p-6 space-y-6">
          <div className="space-y-2">
            <Skeleton className="h-5 w-48" />
            <Skeleton className="h-3 w-80" />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <Card key={i}>
                <CardContent className="p-4 space-y-3">
                  <Skeleton className="h-4 w-40" />
                  <Skeleton className="h-3 w-full" />
                  <Skeleton className="h-8 w-full" />
                </CardContent>
              </Card>
            ))}
          </div>
          <Card>
            <CardContent className="p-4 space-y-3">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-3 w-full" />
              <Skeleton className="h-32 w-full" />
            </CardContent>
          </Card>
        </div>
      </ScrollArea>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-6">
        {/* Section A — Config Form */}
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="text-sm font-medium">Collection Configuration</CardTitle>
            <CardDescription className="text-xs">
              Configure how and when traces are automatically collected into this dataset.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            {/* Enable toggle */}
            <div className="flex items-center justify-between p-4 rounded-xl border border-border bg-background-secondary">
              <div className="space-y-0.5">
                <Label className="text-sm font-medium cursor-pointer" htmlFor="enable-autocollect">
                  Enable Auto-Collect
                </Label>
                <p className="text-xs text-foreground-secondary">
                  Automatically collect and curate traces into this dataset
                </p>
              </div>
              <div className="flex items-center gap-2">
                {enabled && (
                  <Badge variant="secondary" className="text-xs bg-secondary text-primary border-0">
                    Active
                  </Badge>
                )}
                <Switch id="enable-autocollect" checked={enabled} onCheckedChange={setEnabled} />
              </div>
            </div>

            <Separator />

            {/* Fields */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-xs text-foreground-secondary">Source Model</Label>
                <Select
                  value={sourceModel || ALL_MODELS_VALUE}
                  onValueChange={(v) => setSourceModel(v === ALL_MODELS_VALUE ? '' : v)}
                >
                  <SelectTrigger className="text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={ALL_MODELS_VALUE}>All models</SelectItem>
                    {traceModels.map((model) => (
                      <SelectItem key={model} value={model}>
                        {model}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label className="text-xs text-foreground-secondary">Max Samples</Label>
                <Input
                  type="number"
                  value={maxSamples}
                  onChange={(e) => setMaxSamples(Number(e.target.value))}
                  min={1}
                  className="text-sm"
                />
              </div>

              <div className="space-y-2">
                <Label className="text-xs text-foreground-secondary">Collection Interval</Label>
                <Select
                  value={String(intervalMinutes)}
                  onValueChange={(v) => setIntervalMinutes(Number(v))}
                >
                  <SelectTrigger className="text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {INTERVAL_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={String(opt.value)}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Curation Settings */}
            <div className="rounded-xl border border-border bg-background-secondary p-4 space-y-5">
              <p className="text-xs font-semibold text-foreground-secondary uppercase tracking-wider">
                Curation Settings
              </p>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-foreground-secondary">Quality Threshold</Label>
                  <Badge variant="outline" className="text-xs font-mono tabular-nums">
                    {qualityThreshold.toFixed(2)}
                  </Badge>
                </div>
                <Slider
                  min={0}
                  max={1}
                  step={0.05}
                  value={[qualityThreshold]}
                  onValueChange={([v]) => setQualityThreshold(v)}
                />
                <p className="text-xs text-foreground-muted">
                  Minimum quality score for traces to be included
                </p>
              </div>

              <Separator />

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-foreground-secondary">Selection Rate</Label>
                  <Badge variant="outline" className="text-xs font-mono tabular-nums">
                    {selectionRate.toFixed(2)}
                  </Badge>
                </div>
                <Slider
                  min={0}
                  max={1}
                  step={0.05}
                  value={[selectionRate]}
                  onValueChange={([v]) => setSelectionRate(v)}
                />
                <p className="text-xs text-foreground-muted">
                  Fraction of scored traces to keep after curation
                </p>
              </div>
            </div>

            {/* Action buttons */}
            <div className="flex items-center gap-2 pt-1">
              <Button size="sm" onClick={handleSave} disabled={saving}>
                {saving ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Save className="w-4 h-4" />
                )}
                Save
              </Button>
              {configExists && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleDelete}
                  disabled={deleting}
                  className="text-error hover:text-error"
                >
                  {deleting ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Trash2 className="w-4 h-4" />
                  )}
                  Remove Config
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Section B — Actions */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <Button
                size="sm"
                variant="outline"
                onClick={handleTrigger}
                disabled={triggering || !configExists}
              >
                {triggering ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
                Collect Now
              </Button>

              <Separator orientation="vertical" className="h-5" />

              <div className="flex items-center gap-4 text-sm text-foreground-secondary">
                {lastCollectedAt && (
                  <span className="flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5" />
                    Last: {new Date(lastCollectedAt).toLocaleString()}
                  </span>
                )}
                <span className="flex items-center gap-1.5">
                  <Database className="w-3.5 h-3.5" />
                  <span className="font-medium text-foreground">
                    {totalCollected.toLocaleString()}
                  </span>{' '}
                  collected
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Section C — History Table */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Collection History</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {history.length === 0 ? (
              <div className="text-center py-10 px-4">
                <div className="w-10 h-10 rounded-full bg-background-secondary flex items-center justify-center mx-auto mb-3">
                  <Clock className="w-5 h-5 text-foreground-muted" />
                </div>
                <p className="text-sm text-foreground-secondary">No collection runs yet</p>
                <p className="text-xs text-foreground-muted mt-1">
                  Trigger a run or enable auto-collect to get started
                </p>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow className="hover:bg-transparent">
                    <TableHead className="text-xs font-medium text-foreground-secondary pl-6">
                      Time
                    </TableHead>
                    <TableHead className="text-xs font-medium text-foreground-secondary text-right">
                      Found
                    </TableHead>
                    <TableHead className="text-xs font-medium text-foreground-secondary text-right">
                      Dedup
                    </TableHead>
                    <TableHead className="text-xs font-medium text-foreground-secondary text-right">
                      Scored
                    </TableHead>
                    <TableHead className="text-xs font-medium text-foreground-secondary text-right">
                      Selected
                    </TableHead>
                    <TableHead className="text-xs font-medium text-foreground-secondary text-right pr-6">
                      Added
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {history.map((run) => (
                    <TableRow key={run.run_id}>
                      <TableCell className="text-xs text-foreground-secondary pl-6">
                        {new Date(run.created_at).toLocaleString()}
                      </TableCell>
                      <TableCell className="text-xs text-foreground text-right tabular-nums">
                        {run.traces_found.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-xs text-foreground text-right tabular-nums">
                        {run.traces_after_dedup.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-xs text-foreground text-right tabular-nums">
                        {run.traces_scored.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-xs text-foreground text-right tabular-nums">
                        {run.traces_selected.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-right pr-6">
                        <Badge
                          variant="secondary"
                          className="text-xs tabular-nums bg-secondary text-primary border-0"
                        >
                          +{run.samples_added.toLocaleString()}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>
    </ScrollArea>
  );
}
