import { useState, useRef, useEffect } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  CardFooter,
  CardTitle,
  CardDescription,
  CardAction,
} from '@/components/ui/card';

import type { DeploymentData, DeploymentModel, GPUInstanceType } from '@/types/deploymentTypes';
import { DeploymentDetailsModal } from '../DeploymentDetailsModal';
import { StatusBadge } from './StatusBadge';
import { Actions } from './Actions';
import { InfoRow } from './InfoRow';
import { Button } from '@/components/ui/button';
import { Copy } from 'lucide-react';
import { toast } from 'sonner';
import { SpecsModal } from '../SpecsModal';

type Props = {
  deployment: DeploymentData;
  models: DeploymentModel[];
  instances: GPUInstanceType[];
  onDelete?: (deployment: DeploymentData) => void;
  onPause?: (deployment: DeploymentData) => void;
  onResume?: (deployment: DeploymentData) => void;
  isDeleting?: boolean;
  isPausing?: boolean;
  isResuming?: boolean;
  isHighlighted?: boolean;
};

export const DeploymentCard = ({
  deployment,
  models,
  instances,
  isHighlighted,
  ...actions
}: Props) => {
  const [showDetails, setShowDetails] = useState(false);
  const [showAPI, setShowAPI] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  // Scroll into view when highlighted
  useEffect(() => {
    if (isHighlighted && cardRef.current) {
      cardRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [isHighlighted]);

  const model = models.find((m) => m.id === deployment.selectedModel);
  const instance = instances.find((i) => i.id === deployment.selectedInstance);

  return (
    <Card ref={cardRef} className={isHighlighted ? 'animate-highlight-border' : ''}>
      <CardHeader>
        <CardTitle>{model?.name ?? deployment.name}</CardTitle>
        <CardDescription>
          Created {new Date(deployment.createdAt).toLocaleDateString()}
        </CardDescription>
        <CardAction>
          <StatusBadge status={deployment.status} />
        </CardAction>
      </CardHeader>

      <CardContent className="space-y-2">
        <InfoRow label="Model" value={model?.name ?? deployment.selectedModel} />
        <InfoRow label="Engine" value="llama.cpp" />
        {deployment.endpoint_url && (
          <InfoRow
            label="Endpoint"
            value={
              <code className="text-xs font-mono">
                {(() => {
                  try {
                    const u = new URL(deployment.endpoint_url);
                    return `${u.hostname}:${u.port || '80'}`;
                  } catch {
                    return deployment.endpoint_url;
                  }
                })()}
              </code>
            }
          />
        )}

        {deployment.deployment_id && (
          <div className="flex items-center justify-between gap-2">
            <span className="text-muted-foreground text-sm">Deployment ID</span>
            <div className="flex items-center gap-2">
              <code className="text-xs font-mono px-2 py-1 rounded bg-muted">
                {deployment.deployment_id.slice(0, 8)}...
                <Button
                  variant="ghost"
                  size="icon-xs"
                  onClick={() => {
                    navigator.clipboard.writeText(deployment.deployment_id!);
                    toast.success('ID copied');
                  }}
                >
                  <Copy className="h-3 w-3" />
                </Button>
              </code>
            </div>
          </div>
        )}
      </CardContent>

      <CardFooter>
        <Actions
          {...actions}
          deployment={deployment}
          onShowDetails={() => setShowDetails(true)}
          onShowAPI={() => setShowAPI(true)}
        />
      </CardFooter>

      <DeploymentDetailsModal
        deployment={deployment}
        models={models}
        instances={instances}
        isOpen={showDetails}
        onClose={() => setShowDetails(false)}
        onDelete={(dep) => {
          actions.onDelete?.(dep);
          setShowDetails(false);
        }}
      />

      <SpecsModal
        deployment={deployment}
        modelId={model?.id || deployment.selectedModel}
        isOpen={showAPI}
        onClose={() => setShowAPI(false)}
      />
    </Card>
  );
};
