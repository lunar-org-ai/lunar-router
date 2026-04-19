import { useState } from 'react';
import { Cpu, Zap } from 'lucide-react';

import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';

type RuntimeMode = 'cpu-small' | 'cpu-large';

interface DistilledDeployDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  jobName: string;
  onDeploy: (instanceType: RuntimeMode) => void;
  isDeploying: boolean;
}

const TIERS: { id: RuntimeMode; label: string; icon: typeof Cpu; specs: string; desc: string }[] = [
  {
    id: 'cpu-small',
    label: 'GPU offload (recommended)',
    icon: Zap,
    specs: 'llama.cpp · all layers on GPU when available',
    desc: 'Uses the local GPU for inference; falls back to CPU if no GPU is detected.',
  },
  {
    id: 'cpu-large',
    label: 'CPU only',
    icon: Cpu,
    specs: 'llama.cpp · CPU threads only',
    desc: 'No GPU acceleration. Slower but works on any machine.',
  },
];

export function DistilledDeployDialog({
  open,
  onOpenChange,
  jobName,
  onDeploy,
  isDeploying,
}: DistilledDeployDialogProps) {
  const [selectedType, setSelectedType] = useState<RuntimeMode>('cpu-small');

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Deploy "{jobName}" locally</DialogTitle>
          <DialogDescription>
            llama-server will load the smallest GGUF artifact (q4_k_m preferred) on this machine.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3 py-2">
          {TIERS.map((tier) => {
            const Icon = tier.icon;
            return (
              <button
                key={tier.id}
                type="button"
                onClick={() => setSelectedType(tier.id)}
                className={cn(
                  'flex items-start gap-3 rounded-lg border p-4 text-left transition-colors',
                  selectedType === tier.id
                    ? 'border-primary ring-2 ring-primary/20 bg-primary/5'
                    : 'border-border hover:border-muted-foreground/30'
                )}
              >
                <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-muted">
                  <Icon className="h-4 w-4 text-muted-foreground" />
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium leading-none">{tier.label}</p>
                  <p className="text-xs text-muted-foreground">{tier.specs}</p>
                  <p className="text-xs text-muted-foreground/70">{tier.desc}</p>
                </div>
              </button>
            );
          })}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={isDeploying}>
            Cancel
          </Button>
          <Button onClick={() => onDeploy(selectedType)} disabled={isDeploying}>
            {isDeploying ? 'Deploying...' : 'Deploy'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
