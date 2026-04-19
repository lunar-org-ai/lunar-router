import { useEffect, useState } from 'react';
import { CheckCircle2, Cpu, FileBox, HeartPulse, Server } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';

const DEPLOYMENT_STEPS = [
  { label: 'Locating GGUF artifact', icon: FileBox },
  { label: 'Launching llama-server', icon: Server },
  { label: 'Loading model into memory', icon: Cpu },
  { label: 'Waiting for health check', icon: HeartPulse },
] as const;

const TICK_INTERVAL = 100;

interface SuccessScreenProps {
  onClose: () => void;
  autoCloseMs: number;
  alreadyDeployed?: boolean;
}

export function SuccessScreen({ onClose, autoCloseMs, alreadyDeployed }: SuccessScreenProps) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setElapsed((prev) => prev + TICK_INTERVAL), TICK_INTERVAL);
    return () => clearInterval(id);
  }, []);

  const progressPct = Math.min((elapsed / autoCloseMs) * 100, 100);
  const secondsLeft = Math.max(Math.ceil((autoCloseMs - elapsed) / 1000), 0);

  return (
    <Card className="border-0 shadow-none">
      <CardContent className="pt-6 pb-6">
        <div className="flex flex-col items-center text-center mb-6">
          <div className="relative mb-4">
            <div className="w-12 h-12 bg-green-500/10 rounded-full flex items-center justify-center relative">
              <CheckCircle2 className="w-6 h-6 text-green-500" />
            </div>
          </div>

          <h3 className="text-base font-semibold mb-0.5">
            {alreadyDeployed ? 'Already Deployed' : 'Deployment Created'}
          </h3>
          <p className="text-sm text-muted-foreground max-w-sm">
            {alreadyDeployed
              ? 'This model is already running. Switching you to Active Models.'
              : 'llama-server is starting up. The model will be ready shortly.'}
          </p>
        </div>

        <div className="space-y-2 max-w-xs mx-auto mb-6">
          {DEPLOYMENT_STEPS.map(({ label, icon: Icon }) => (
            <div
              key={label}
              className="flex items-center gap-2.5 text-sm rounded-md px-3 py-1.5 bg-muted/40"
            >
              <Icon className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
              <span className="text-foreground/80 flex-1 text-xs">{label}</span>
              <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
            </div>
          ))}
        </div>

        <div className="max-w-xs mx-auto space-y-2.5">
          <Progress
            value={progressPct}
            className="h-1 [&>div]:bg-primary [&>div]:transition-all [&>div]:duration-100"
          />
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground">Closing in {secondsLeft}s</p>
            <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={onClose}>
              Close now
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
