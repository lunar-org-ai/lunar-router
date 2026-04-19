import { useEffect, useRef } from 'react';

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import type { DeploymentError } from '../../../utils/deploymentErrorHandling';
import { CreatingScreen } from './CreatingScreen';
import { SuccessScreen } from './SuccessScreen';
import { ErrorScreen } from './ErrorScreen';

export type DeployProgressState = 'creating' | 'success' | 'error' | null;

const AUTO_CLOSE_DELAY = 10_000;

interface DeployProgressDialogProps {
  isOpen: boolean;
  state: DeployProgressState;
  error?: DeploymentError;
  selectedInstanceId?: string;
  availableInstanceIds?: string[];
  alreadyDeployed?: boolean;
  onRetry?: () => void;
  onSelectAlternative?: (instanceId: string) => void;
  onChangeInstance?: () => void;
  onClose: () => void;
}

export function DeployProgressDialog({
  isOpen,
  state,
  error,
  selectedInstanceId,
  availableInstanceIds = [],
  alreadyDeployed,
  onRetry,
  onSelectAlternative,
  onChangeInstance,
  onClose,
}: DeployProgressDialogProps) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Auto-close after success
  useEffect(() => {
    if (state === 'success') {
      timerRef.current = setTimeout(() => onClose(), AUTO_CLOSE_DELAY);
    }
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [state, onClose]);

  // Prevent closing while creating
  const handleOpenChange = (open: boolean) => {
    if (!open && state !== 'creating') {
      onClose();
    }
  };

  const renderContent = () => {
    switch (state) {
      case 'creating':
        return <CreatingScreen />;
      case 'success':
        return (
          <SuccessScreen
            onClose={onClose}
            autoCloseMs={AUTO_CLOSE_DELAY}
            alreadyDeployed={alreadyDeployed}
          />
        );
      case 'error':
        if (!error) {
          return (
            <div className="text-center py-12">
              <p className="text-sm text-muted-foreground">An unexpected error occurred</p>
            </div>
          );
        }
        return (
          <ErrorScreen
            error={error}
            selectedInstanceId={selectedInstanceId}
            availableInstanceIds={availableInstanceIds}
            onRetry={onRetry}
            onSelectAlternative={onSelectAlternative}
            onChangeInstance={onChangeInstance}
            onClose={onClose}
          />
        );
      default:
        return null;
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <DialogContent
        className="max-w-2xl gap-0 p-0 overflow-hidden border-0"
        showCloseButton={state !== 'creating'}
      >
        <DialogHeader className="sr-only">
          <DialogTitle>Deployment Progress</DialogTitle>
          <DialogDescription>Tracking your deployment creation status</DialogDescription>
        </DialogHeader>
        <div className="px-6 py-6">{renderContent()}</div>
      </DialogContent>
    </Dialog>
  );
}
