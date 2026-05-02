import { useState } from 'react';
import { ArrowLeft, ArrowRight, Check, Copy, Play, Terminal } from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { cn } from '@/lib/utils';
import {
  type AgentFramework,
  FRAMEWORK_OPTIONS,
  type FrameworkOption,
} from '@/features/agents/state';

type ImportAgentModalProps = {
  open: boolean;
  step: 1 | 2;
  framework: AgentFramework | null;
  onOpenChange: (open: boolean) => void;
  onSelectFramework: (framework: AgentFramework) => void;
  onAdvance: () => void;
  onRunSimulation: () => void;
};

export function ImportAgentModal({
  open,
  step,
  framework,
  onOpenChange,
  onSelectFramework,
  onAdvance,
  onRunSimulation,
}: ImportAgentModalProps) {
  const selected = FRAMEWORK_OPTIONS.find((option) => option.id === framework) ?? null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl gap-6">
        <DialogHeader>
          <DialogTitle>Import Agent</DialogTitle>
          <DialogDescription>
            {step === 1
              ? 'Pick the framework your agent is built with. OpenTracy will auto-instrument it.'
              : 'Add this snippet to your agent code, then run a simulation to see the topology.'}
          </DialogDescription>
        </DialogHeader>

        {step === 1 ? (
          <FrameworkPicker selectedId={framework} onSelect={onSelectFramework} />
        ) : selected ? (
          <CodeStep option={selected} />
        ) : null}

        <DialogFooter className="sm:justify-between">
          {step === 2 ? (
            <Button variant="ghost" onClick={() => onOpenChange(false)} className="gap-1.5">
              <ArrowLeft className="size-3.5" />
              Cancel
            </Button>
          ) : (
            <Button variant="ghost" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
          )}

          {step === 1 ? (
            <Button onClick={onAdvance} disabled={!framework} className="gap-1.5">
              Continue
              <ArrowRight className="size-3.5" />
            </Button>
          ) : (
            <Button onClick={onRunSimulation} className="gap-1.5">
              <Play className="size-3.5 fill-current" />
              Run Simulation
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

type FrameworkPickerProps = {
  selectedId: AgentFramework | null;
  onSelect: (id: AgentFramework) => void;
};

function FrameworkPicker({ selectedId, onSelect }: FrameworkPickerProps) {
  return (
    <div className="grid grid-cols-2 gap-3">
      {FRAMEWORK_OPTIONS.map((option) => {
        const active = option.id === selectedId;
        return (
          <button
            key={option.id}
            type="button"
            onClick={() => onSelect(option.id)}
            className={cn(
              'flex flex-col gap-1.5 rounded-xl border bg-card/30 px-4 py-3 text-left transition-colors',
              active
                ? 'border-foreground/40 bg-card/60'
                : 'border-border/40 hover:border-border/70 hover:bg-card/50'
            )}
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">{option.name}</span>
              {active ? (
                <Check className="size-3.5 text-emerald-500" />
              ) : (
                <span className="size-3.5" />
              )}
            </div>
            <span className="font-mono text-[11px] text-muted-foreground">{option.pkg}</span>
          </button>
        );
      })}
    </div>
  );
}

type CodeStepProps = {
  option: FrameworkOption;
};

function CodeStep({ option }: CodeStepProps) {
  const [copied, setCopied] = useState(false);

  const installCommand = `pip install ${option.pkg}`;

  const handleCopy = (text: string) => {
    if (typeof navigator === 'undefined' || !navigator.clipboard) return;
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    });
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2 rounded-lg border border-border/40 bg-card/30 px-3 py-2">
        <Terminal className="size-3.5 shrink-0 text-muted-foreground" />
        <code className="flex-1 font-mono text-[12px] text-foreground">{installCommand}</code>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => handleCopy(installCommand)}
          className="h-6 gap-1 px-2 text-[11px]"
        >
          {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
          {copied ? 'Copied' : 'Copy'}
        </Button>
      </div>

      <div className="rounded-lg border border-border/40 bg-card/30">
        <div className="flex items-center justify-between border-b border-border/40 px-3 py-2">
          <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
            agent.py
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleCopy(option.snippet)}
            className="h-6 gap-1 px-2 text-[11px]"
          >
            {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
            {copied ? 'Copied' : 'Copy'}
          </Button>
        </div>
        <pre className="overflow-x-auto px-4 py-3 font-mono text-[12px] leading-relaxed text-muted-foreground">
          <code>{option.snippet}</code>
        </pre>
      </div>
    </div>
  );
}
