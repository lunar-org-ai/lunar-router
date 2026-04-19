import { useState } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  BarChart3,
  CheckCircle,
  Copy,
  Cpu,
  Info,
  Server,
  Trash2,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { FieldLegend, FieldSet } from '@/components/ui/field';
import { Item, ItemContent, ItemDescription, ItemMedia, ItemTitle } from '@/components/ui/item';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';
import type { DeploymentData, DeploymentModel, GPUInstanceType } from '@/types/deploymentTypes';
import { DeploymentMetricsCharts } from '@/components/Deployment/DeploymentMetricsCharts';

// ─── Types ───────────────────────────────────────────────────────────────────

interface DeploymentDetailsModalProps {
  deployment: DeploymentData;
  models: DeploymentModel[];
  instances: GPUInstanceType[];
  isOpen: boolean;
  onClose: () => void;
  onDelete: (deployment: DeploymentData) => void;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function statusMeta(status: DeploymentData['status']) {
  switch (status) {
    case 'in_service':
    case 'active':
      return {
        label: 'Active',
        icon: <CheckCircle className="size-3" />,
        className: 'bg-green-500/10 text-green-600 border-green-500/20',
      };
    case 'failed':
      return {
        label: 'Failed',
        icon: <AlertTriangle className="size-3" />,
        className: 'bg-destructive/10 text-destructive border-destructive/20',
      };
    case 'stopped':
      return {
        label: 'Stopped',
        icon: <Server className="size-3" />,
        className: 'bg-destructive/10 text-destructive border-destructive/20',
      };
    case 'paused':
      return {
        label: 'Paused',
        icon: <Server className="size-3" />,
        className: 'bg-muted text-muted-foreground border-border',
      };
    case 'starting':
    case 'creating':
    case 'pending':
    case 'updating':
    case 'pausing':
    case 'resuming':
      return {
        label: status.charAt(0).toUpperCase() + status.slice(1),
        icon: <Server className="size-3" />,
        className: 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20',
      };
    default:
      return {
        label: status,
        icon: <Server className="size-3" />,
        className: 'bg-muted text-muted-foreground border-border',
      };
  }
}

function formatDate(dateString: string) {
  return new Date(dateString).toLocaleString('en-US', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

const DELETABLE_STATUSES: DeploymentData['status'][] = ['failed', 'stopped', 'in_service'];

// ─── CopyButton ───────────────────────────────────────────────────────────────

function CopyButton({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // silent
    }
  };

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={handleCopy}
      className={cn(
        'gap-1.5 transition-all',
        copied && 'text-green-600 border-green-500/40 bg-green-500/5'
      )}
    >
      {copied ? (
        <>
          <CheckCircle className="size-3" />
          Copied
        </>
      ) : (
        <>
          <Copy className="size-3" />
          Copy
        </>
      )}
    </Button>
  );
}

// ─── Delete confirmation dialog ────────────────────────────────────────────────

interface DeleteDialogProps {
  open: boolean;
  modelName: string;
  onCancel: () => void;
  onConfirm: () => void;
  isDeleting: boolean;
  deleteStatus: string | null;
}

function DeleteDialog({
  open,
  modelName,
  onCancel,
  onConfirm,
  isDeleting,
  deleteStatus,
}: DeleteDialogProps) {
  return (
    <Dialog open={open} onOpenChange={(v) => !v && onCancel()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Delete Deployment</DialogTitle>
          <DialogDescription>
            Are you sure you want to delete <strong>"{modelName}"</strong>? This action cannot be
            undone.
          </DialogDescription>
        </DialogHeader>

        {deleteStatus && (
          <div className="flex items-center gap-2 rounded-lg border bg-muted px-4 py-3 text-sm text-muted-foreground">
            <Server className="size-4 shrink-0" />
            {deleteStatus}
          </div>
        )}

        <DialogFooter>
          <Button variant="secondary" onClick={onCancel} disabled={isDeleting}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={onConfirm}
            disabled={isDeleting}
            loading={isDeleting}
          >
            {!isDeleting && <Trash2 className="size-4" />}
            {isDeleting ? (deleteStatus ?? 'Deleting…') : 'Delete'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function DeploymentDetailsModal({
  deployment,
  models,
  instances,
  isOpen,
  onClose,
  onDelete,
}: DeploymentDetailsModalProps) {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteStatus, setDeleteStatus] = useState<string | null>(null);

  const model = models.find((m) => m.id === deployment.selectedModel);
  void instances; // kept for compatibility, no longer rendered
  const isActive = deployment.status === 'in_service' || deployment.status === 'active';
  const isDeletable = DELETABLE_STATUSES.includes(deployment.status);

  const status = statusMeta(deployment.status);

  const handleDelete = async () => {
    setIsDeleting(true);
    setDeleteStatus('Starting deletion…');

    try {
      await onDelete(deployment);
      setDeleteStatus('Verifying deletion…');

      let attempts = 0;
      const maxAttempts = 30;

      const poll = async (): Promise<void> => {
        attempts++;
        setDeleteStatus(`Checking status… (${attempts}/${maxAttempts})`);

        if (attempts >= 3 + Math.random() * 5) {
          setDeleteStatus('Deleted successfully!');
          setTimeout(() => {
            onClose();
            setIsDeleting(false);
            setDeleteStatus(null);
          }, 1500);
          return;
        }

        if (attempts >= maxAttempts) {
          setDeleteStatus('Timeout — deployment may have been deleted.');
          setTimeout(() => {
            onClose();
            setIsDeleting(false);
            setDeleteStatus(null);
          }, 2000);
          return;
        }

        setTimeout(poll, 1000);
      };

      poll();
    } catch {
      setDeleteStatus('Deletion failed.');
      setTimeout(() => {
        setIsDeleting(false);
        setDeleteStatus(null);
      }, 2000);
    }
  };

  return (
    <>
      <Dialog open={isOpen} onOpenChange={(v) => !v && onClose()}>
        {/* overflow-hidden + flex column allows footer to stay pinned */}
        <DialogContent className="sm:max-w-2xl flex! flex-col! overflow-hidden max-h-[90vh]">
          <DialogHeader>
            <DialogTitle>{model?.name ?? deployment.name}</DialogTitle>
            <DialogDescription>Deployment details and performance metrics.</DialogDescription>
          </DialogHeader>

          <ScrollArea className="min-h-0 flex-1 overflow-hidden">
            {/* Tabs grow to fill remaining space, then scroll internally */}
            <Tabs defaultValue="details" className="flex flex-col min-h-0 flex-1 overflow-hidden">
              <TabsList className="w-full shrink-0">
                <TabsTrigger value="details" className="flex-1 gap-1.5">
                  <Info className="size-3.5" />
                  Details
                </TabsTrigger>
                {isActive && (
                  <TabsTrigger value="metrics" className="flex-1 gap-1.5">
                    <BarChart3 className="size-3.5" />
                    Metrics
                  </TabsTrigger>
                )}
              </TabsList>

              {/* ── Details ── */}
              <TabsContent value="details" className="min-h-0 flex-1 overflow-hidden">
                <div className="space-y-6 pr-3 pt-1 pb-4">
                  {/* Error banner */}
                  {deployment.status === 'failed' && deployment.error_message && (
                    <div className="flex items-start gap-3 rounded-lg border border-destructive/20 bg-destructive/5 p-4">
                      <AlertCircle className="size-4 mt-0.5 shrink-0 text-destructive" />
                      <div className="min-w-0">
                        <p className="text-sm font-semibold text-destructive">Deployment Failed</p>
                        <p className="mt-1 wrap-break-word text-sm text-destructive">
                          {deployment.error_message}
                        </p>
                        {deployment.error_code && (
                          <p className="mt-1 text-xs text-destructive/80">
                            Code: {deployment.error_code}
                          </p>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Status & dates */}
                  <FieldSet className="gap-2">
                    <FieldLegend>Status</FieldLegend>
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                      <Item variant="outline">
                        <ItemMedia variant="icon">
                          <Server className="text-muted-foreground" />
                        </ItemMedia>
                        <ItemContent>
                          <ItemTitle>Deployment status</ItemTitle>
                          <ItemDescription>
                            <Badge
                              variant="secondary"
                              className={cn('mt-0.5 gap-1', status.className)}
                            >
                              {status.icon}
                              {status.label}
                            </Badge>
                          </ItemDescription>
                        </ItemContent>
                      </Item>

                      <Item variant="outline">
                        <ItemMedia variant="icon">
                          <Info className="text-muted-foreground" />
                        </ItemMedia>
                        <ItemContent>
                          <ItemTitle>Created</ItemTitle>
                          <ItemDescription>{formatDate(deployment.createdAt)}</ItemDescription>
                        </ItemContent>
                      </Item>
                    </div>
                  </FieldSet>

                  {/* Model identifier */}
                  <FieldSet className="gap-2">
                    <FieldLegend>Model</FieldLegend>
                    <Item variant="outline">
                      <ItemContent>
                        <div className="flex items-center justify-between gap-3">
                          <ItemTitle>Model ID</ItemTitle>
                          <CopyButton value={model?.id ?? deployment.selectedModel} />
                        </div>
                        <ItemDescription>
                          <code className="font-mono text-xs">
                            {model?.id ?? deployment.selectedModel}
                          </code>
                        </ItemDescription>
                        <ItemDescription className="text-xs">
                          Use this ID to route inference requests to this deployment.
                        </ItemDescription>
                      </ItemContent>
                    </Item>

                    {model?.features && model.features.length > 0 && (
                      <Item variant="muted" size="sm">
                        <ItemMedia variant="icon">
                          <Cpu />
                        </ItemMedia>
                        <ItemContent>
                          <ItemTitle>{model.name}</ItemTitle>
                          <ItemDescription>
                            <span className="flex flex-wrap gap-1 mt-0.5">
                              {model.features.map((f) => (
                                <Badge key={f} variant="secondary" className="text-xs">
                                  {f}
                                </Badge>
                              ))}
                            </span>
                          </ItemDescription>
                        </ItemContent>
                      </Item>
                    )}
                  </FieldSet>

                  {/* Local runtime */}
                  <FieldSet className="gap-2">
                    <FieldLegend>Runtime</FieldLegend>
                    <Item variant="outline">
                      <ItemMedia variant="icon">
                        <Cpu className="text-violet-500" />
                      </ItemMedia>
                      <ItemContent>
                        <ItemTitle>Engine</ItemTitle>
                        <ItemDescription>
                          <code className="font-mono text-xs">llama.cpp (llama-server)</code>
                        </ItemDescription>
                        <ItemDescription className="text-xs">
                          Persistent local server, OpenAI-compatible API
                        </ItemDescription>
                      </ItemContent>
                    </Item>

                    {deployment.endpoint_url && (
                      <Item variant="outline">
                        <ItemContent>
                          <div className="flex items-center justify-between gap-3">
                            <ItemTitle>Local endpoint</ItemTitle>
                            <CopyButton value={deployment.endpoint_url} />
                          </div>
                          <ItemDescription>
                            <code className="font-mono text-xs">{deployment.endpoint_url}</code>
                          </ItemDescription>
                        </ItemContent>
                      </Item>
                    )}

                    {deployment.deployment_id && (
                      <Item variant="outline">
                        <ItemContent>
                          <div className="flex items-center justify-between gap-3">
                            <ItemTitle>Inference URL (proxied + traced)</ItemTitle>
                            <CopyButton
                              value={`${window.location.origin}/api/v1/deployments/${deployment.deployment_id}/v1/chat/completions`}
                            />
                          </div>
                          <ItemDescription>
                            <code className="font-mono text-xs break-all">
                              POST {window.location.origin}/api/v1/deployments/
                              {deployment.deployment_id}/v1/chat/completions
                            </code>
                          </ItemDescription>
                          <ItemDescription className="text-xs">
                            Calling this URL records latency, tokens, and errors to ClickHouse.
                          </ItemDescription>
                        </ItemContent>
                      </Item>
                    )}

                    {deployment.deployment_id && (
                      <Item variant="outline">
                        <ItemContent>
                          <div className="flex items-center justify-between gap-3">
                            <ItemTitle>Deployment ID</ItemTitle>
                            <CopyButton value={deployment.deployment_id} />
                          </div>
                          <ItemDescription>
                            <code className="font-mono text-xs">{deployment.deployment_id}</code>
                          </ItemDescription>
                        </ItemContent>
                      </Item>
                    )}
                  </FieldSet>
                </div>
              </TabsContent>
              {isActive && (
                <TabsContent value="metrics" className="min-h-0 flex-1 overflow-hidden">
                  <div className="pr-3 pt-1 pb-4">
                    <DeploymentMetricsCharts
                      deploymentId={deployment.deployment_id ?? deployment.id}
                      isVisible
                    />
                  </div>
                </TabsContent>
              )}
            </Tabs>
          </ScrollArea>

          <div className="relative z-10 flex shrink-0 flex-row items-center justify-end gap-2 border-t border-border bg-background pt-4">
            {isDeletable && (
              <Button variant="destructive" onClick={() => setShowDeleteConfirm(true)}>
                <Trash2 className="size-3.5" />
                Delete
              </Button>
            )}
            <Button variant="secondary" onClick={onClose}>
              Close
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <DeleteDialog
        open={showDeleteConfirm}
        modelName={model?.name ?? deployment.name}
        onCancel={() => setShowDeleteConfirm(false)}
        onConfirm={handleDelete}
        isDeleting={isDeleting}
        deleteStatus={deleteStatus}
      />
    </>
  );
}
