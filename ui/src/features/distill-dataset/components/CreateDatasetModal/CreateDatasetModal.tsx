import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { AlertCircle, Sparkles } from 'lucide-react';
import { useCreateDatasetModal } from '../../hooks/UseCreateDatasetModal';
import { ModeSelector } from './ModeSelector';
import { DatasetNameFields } from './DatasetNameFields';
import { ManualMode } from './ManualMode';
import { ImportMode } from './ImportMode';
import { SmartImportMode } from './SmartImportMode';
import { TopicMode } from './TopicMode';
import { TracesMode } from './TracesMode';
import { ClusterMode } from './ClusterMode';
import { SuccessOverlay } from './SuccessOverlay';
import type {
  Dataset,
  Trace,
  CreateMode,
  CreateDatasetRequest,
  CreateFromInstructionRequest,
  CreateFromInstructionResponse,
  AnalyzeTracesResponse,
  ImportTracesResponse,
} from '../../types';
import type { ClusterDataset } from '@/services/clusteringService';

interface FooterState {
  secondary: string;
  primary: string;
  onSecondary: () => void;
  onPrimary?: () => void;
  isSubmit: boolean;
}

const SUBMIT_LABEL: Partial<Record<CreateMode, string>> = {
  import: 'Import Dataset',
  'smart-import': 'Import Traces',
  topic: 'Find & Create',
  manual: 'Create Dataset',
};

const MODE_DESCRIPTIONS: Record<CreateMode, string> = {
  topic: "Describe a topic and we'll find matching traces automatically.",
  generate: 'Describe what you need and AI will generate samples from scratch.',
  traces: 'Select traces from your API and Playground usage.',
  manual: 'Add input/output samples by hand, one at a time.',
  import: 'Upload a CSV or JSON file to populate the dataset.',
  'smart-import': 'Upload any JSON traces — AI auto-detects the schema.',
  cluster: 'Select a cluster of traces to create a dataset from.',
};

interface CreateDatasetModalProps {
  open: boolean;
  loading?: boolean;
  onClose: () => void;
  onCreate: (request: CreateDatasetRequest) => Promise<Dataset | null | void>;
  onImport: (
    file: File,
    name: string,
    autoCollectInstruction?: string
  ) => Promise<Dataset | null | void>;
  onCreateFromTopic?: (
    request: CreateFromInstructionRequest
  ) => Promise<CreateFromInstructionResponse | null | undefined>;
  traces?: Trace[];
  tracesLoading?: boolean;
  onCreateFromTraces?: (name: string, traceIds: string[]) => Promise<void>;
  onAnalyzeTraces?: (data: any[]) => Promise<AnalyzeTracesResponse>;
  onImportTraces?: (
    name: string,
    data: any[],
    mapping: any,
    description?: string
  ) => Promise<ImportTracesResponse>;
  clusters?: ClusterDataset[];
  clustersLoading?: boolean;
  onCreateFromCluster?: (name: string, runId: string, clusterId: number) => Promise<void>;
  onTriggerClustering?: (days?: number) => Promise<void>;
}

export function CreateDatasetModal({
  open,
  loading,
  onClose,
  onCreate,
  onImport,
  onCreateFromTopic,
  traces,
  tracesLoading,
  onCreateFromTraces,
  onAnalyzeTraces,
  onImportTraces,
  clusters,
  clustersLoading,
  onCreateFromCluster,
  onTriggerClustering,
}: CreateDatasetModalProps) {
  const modal = useCreateDatasetModal({
    open,
    onClose,
    onCreate,
    onImport,
    onCreateFromTopic,
    onAnalyzeTraces,
    onImportTraces,
  });

  const isDisabled = modal.isDisabled || !!loading;

  const showAutoCollect =
    modal.mode !== 'traces' &&
    modal.mode !== 'cluster' &&
    !(modal.mode === 'topic' && modal.topicPhase !== 'idle');

  const showAutoCollectTextarea = modal.mode === 'manual' || modal.mode === 'import';

  const footerState = ((): FooterState => {
    const isTopicFinished =
      modal.mode === 'topic' && ['done', 'no-match', 'error'].includes(modal.topicPhase);
    const isTopicRetry = modal.mode === 'topic' && ['no-match', 'error'].includes(modal.topicPhase);

    if (isTopicRetry)
      return {
        secondary: 'Try Again',
        primary: 'Close',
        onSecondary: modal.handleTopicRetry,
        onPrimary: modal.handleClose,
        isSubmit: false,
      };
    if (isTopicFinished)
      return {
        secondary: 'Close',
        primary: 'Done',
        onSecondary: modal.handleClose,
        onPrimary: modal.handleClose,
        isSubmit: false,
      };

    const loadingLabel = modal.mode === 'topic' ? 'Scanning traces…' : 'Creating…';

    return {
      secondary: 'Cancel',
      primary: isDisabled ? loadingLabel : (SUBMIT_LABEL[modal.mode] ?? 'Create'),
      onSecondary: modal.handleClose,
      isSubmit: true,
    };
  })();

  const renderModeContent = () => {
    switch (modal.mode) {
      case 'topic':
        return (
          <TopicMode
            name={modal.name}
            description={modal.description}
            topic={modal.topic}
            topicPhase={modal.topicPhase}
            isProcessing={modal.isTopicProcessing}
            agentLog={modal.agentLog}
            topicResult={modal.topicResult}
            error={modal.error}
            disabled={isDisabled}
            onNameChange={modal.setName}
            onDescriptionChange={modal.setDescription}
            onTopicChange={modal.setTopic}
            inputRef={modal.inputRef}
          />
        );

      case 'traces':
        return (
          <TracesMode
            traces={traces ?? []}
            tracesLoading={!!tracesLoading}
            disabled={isDisabled}
            onCreateFromTraces={onCreateFromTraces!}
            onSuccess={modal.showSuccess}
          />
        );

      case 'cluster':
        return (
          <ClusterMode
            clusters={clusters ?? []}
            clustersLoading={!!clustersLoading}
            disabled={isDisabled}
            onCreateFromCluster={onCreateFromCluster!}
            onSuccess={modal.showSuccess}
            onTriggerClustering={onTriggerClustering}
          />
        );

      case 'smart-import':
        return (
          <SmartImportMode
            name={modal.name}
            description={modal.description}
            disabled={isDisabled}
            file={modal.smartImportFile}
            phase={modal.smartImportPhase}
            analysis={modal.smartImportAnalysis}
            recordCount={modal.smartImportRecordCount}
            onNameChange={modal.setName}
            onDescriptionChange={modal.setDescription}
            onFileSelect={modal.handleSmartImportFileSelect}
            inputRef={modal.inputRef}
          />
        );

      default:
        return (
          <div className="space-y-4">
            <DatasetNameFields
              name={modal.name}
              description={modal.description}
              disabled={isDisabled}
              onNameChange={modal.setName}
              onDescriptionChange={modal.setDescription}
              inputRef={modal.inputRef}
              nameRequired
            />
            {modal.mode === 'import' ? (
              <ImportMode
                file={modal.file}
                disabled={isDisabled}
                onFileChange={modal.handleFileChange}
              />
            ) : (
              <ManualMode
                samples={modal.samples}
                disabled={isDisabled}
                onAddSample={modal.addSample}
                onRemoveSample={modal.removeSample}
                onUpdateSample={modal.updateSample}
              />
            )}
          </div>
        );
    }
  };

  const dialogWidth =
    modal.mode === 'traces' || modal.mode === 'cluster' ? 'max-w-5xl' : 'max-w-2xl';

  return (
    <Dialog open={open} onOpenChange={(v) => !v && modal.handleClose()}>
      <DialogContent
        className={`flex ${dialogWidth} flex-col gap-0 p-0 overflow-hidden transition-[max-width] duration-200`}
        showCloseButton={!modal.isTopicProcessing}
      >
        <form onSubmit={modal.submit} className="relative flex max-h-[85vh] flex-col">
          <DialogHeader className="px-6 pt-6 pb-4">
            <DialogTitle>Create Dataset</DialogTitle>
            <DialogDescription>{MODE_DESCRIPTIONS[modal.mode]}</DialogDescription>
          </DialogHeader>

          <Separator />

          <div className="px-6 pt-4">
            <ModeSelector
              mode={modal.mode}
              onModeChange={modal.setMode}
              disabled={isDisabled}
              showTopic={!!onCreateFromTopic}
              showTraces={!!onCreateFromTraces && !!traces}
              showCluster={!!onCreateFromCluster && !!clusters}
            />
          </div>

          {modal.mode === 'traces' || modal.mode === 'cluster' ? (
            <div className="flex min-h-0 flex-1 flex-col">{renderModeContent()}</div>
          ) : (
            <>
              <ScrollArea className="max-h-[50vh]">
                <div className="px-6 py-5 space-y-5">{renderModeContent()}</div>
              </ScrollArea>

              <div className="px-6 pb-4 space-y-4">
                {showAutoCollect && (
                  <div className="space-y-3">
                    <label className="flex cursor-pointer items-center gap-2">
                      <Checkbox
                        checked={modal.keepBuilding}
                        onCheckedChange={(v) => modal.setKeepBuilding(!!v)}
                        disabled={isDisabled}
                      />
                      <span className="text-sm text-muted-foreground">
                        Keep collecting matching traces over time
                      </span>
                      <Sparkles className="size-3.5 text-muted-foreground" />
                    </label>

                    {modal.keepBuilding && showAutoCollectTextarea && (
                      <div className="space-y-1.5">
                        <Label>What traces to collect</Label>
                        <Textarea
                          rows={2}
                          value={modal.autoCollectInstruction}
                          onChange={(e) => modal.setAutoCollectInstruction(e.target.value)}
                          placeholder="e.g. customer support conversations, Python coding questions…"
                          disabled={isDisabled}
                        />
                        <p className="text-xs text-muted-foreground">
                          Describe the type of traces to automatically collect from your API
                          traffic.
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {modal.error && modal.mode !== 'topic' && (
                  <Alert variant="destructive">
                    <AlertCircle className="size-4" />
                    <AlertDescription>{modal.error}</AlertDescription>
                  </Alert>
                )}
              </div>
            </>
          )}

          {modal.mode !== 'traces' && modal.mode !== 'cluster' && (
            <DialogFooter className="px-6 py-4 border-t">
              <Button
                type="button"
                variant="outline"
                onClick={footerState.onSecondary}
                disabled={footerState.isSubmit && isDisabled}
              >
                {footerState.secondary}
              </Button>
              <Button
                type={footerState.isSubmit ? 'submit' : 'button'}
                onClick={footerState.onPrimary}
                disabled={footerState.isSubmit && isDisabled}
                loading={footerState.isSubmit && isDisabled}
              >
                {footerState.primary}
              </Button>
            </DialogFooter>
          )}

          {modal.successName && modal.mode !== 'topic' && (
            <SuccessOverlay name={modal.successName} />
          )}
        </form>
      </DialogContent>
    </Dialog>
  );
}
