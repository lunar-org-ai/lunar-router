import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { ArrowLeft, ArrowRight, Check, Copy, Play, Terminal } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import {
  type AgentFramework,
  FRAMEWORK_OPTIONS,
  type FrameworkOption,
} from '@/features/agents/state';

type ImportFlowProps = {
  step: 1 | 2;
  framework: AgentFramework | null;
  onSelectFramework: (framework: AgentFramework) => void;
  onAdvance: () => void;
  onRunSimulation: () => void;
  onCancel: () => void;
};

export function ImportFlow({
  step,
  framework,
  onSelectFramework,
  onAdvance,
  onRunSimulation,
  onCancel,
}: ImportFlowProps) {
  const selected = FRAMEWORK_OPTIONS.find((option) => option.id === framework) ?? null;

  return (
    <div className="flex flex-1 items-center justify-center px-8 py-12">
      <div className="flex w-full max-w-xl flex-col gap-6">
        <Stepper current={step} />

        <AnimatePresence mode="wait">
          {step === 1 ? (
            <motion.div
              key="step-1"
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 8 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
              className="flex flex-col gap-5"
            >
              <Heading
                title="Choose a framework"
                description="Pick the framework your agent is built with — OpenTracy will auto-instrument it."
              />
              <FrameworkPicker selectedId={framework} onSelect={onSelectFramework} />
              <Footer>
                <Button variant="ghost" onClick={onCancel} className="text-muted-foreground">
                  Cancel
                </Button>
                <Button onClick={onAdvance} disabled={!framework} className="gap-1.5">
                  Continue
                  <ArrowRight className="size-3.5" />
                </Button>
              </Footer>
            </motion.div>
          ) : selected ? (
            <motion.div
              key="step-2"
              initial={{ opacity: 0, x: 8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -8 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
              className="flex flex-col gap-5"
            >
              <Heading
                title="Add to your code"
                description={`Drop this snippet into your ${selected.name} app, then run a simulation to preview the topology.`}
              />
              <CodeStep option={selected} />
              <Footer>
                <Button variant="ghost" onClick={onCancel} className="gap-1.5 text-muted-foreground">
                  <ArrowLeft className="size-3.5" />
                  Cancel
                </Button>
                <Button onClick={onRunSimulation} className="gap-1.5">
                  <Play className="size-3.5 fill-current" />
                  Run Simulation
                </Button>
              </Footer>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </div>
  );
}

type StepperProps = {
  current: 1 | 2;
};

function Stepper({ current }: StepperProps) {
  return (
    <div className="flex items-center gap-3 text-[11px] uppercase tracking-wider text-muted-foreground/70">
      <StepDot index={1} current={current} label="Framework" />
      <span className="h-px w-6 bg-border/50" />
      <StepDot index={2} current={current} label="Connect" />
    </div>
  );
}

type StepDotProps = {
  index: 1 | 2;
  current: 1 | 2;
  label: string;
};

function StepDot({ index, current, label }: StepDotProps) {
  const isActive = index === current;
  const isDone = index < current;
  return (
    <div className="flex items-center gap-2">
      <span
        className={cn(
          'flex size-5 items-center justify-center rounded-full border text-[10px] transition-colors',
          isActive && 'border-foreground/50 bg-foreground text-background',
          isDone && 'border-emerald-500/60 bg-emerald-500/10 text-emerald-500',
          !isActive && !isDone && 'border-border/50 text-muted-foreground'
        )}
      >
        {isDone ? <Check className="size-3" /> : index}
      </span>
      <span className={cn(isActive ? 'text-foreground' : '')}>{label}</span>
    </div>
  );
}

type HeadingProps = {
  title: string;
  description: string;
};

function Heading({ title, description }: HeadingProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <h2 className="text-lg font-medium tracking-tight">{title}</h2>
      <p className="font-mono text-xs leading-relaxed text-muted-foreground">{description}</p>
    </div>
  );
}

type FooterProps = {
  children: React.ReactNode;
};

function Footer({ children }: FooterProps) {
  return <div className="flex items-center justify-between pt-2">{children}</div>;
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
  const [copied, setCopied] = useState<'install' | 'snippet' | null>(null);

  const installCommand = `pip install ${option.pkg}`;

  const handleCopy = (text: string, target: 'install' | 'snippet') => {
    if (typeof navigator === 'undefined' || !navigator.clipboard) return;
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(target);
      setTimeout(() => setCopied(null), 1400);
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
          onClick={() => handleCopy(installCommand, 'install')}
          className="h-6 gap-1 px-2 text-[11px]"
        >
          {copied === 'install' ? <Check className="size-3" /> : <Copy className="size-3" />}
          {copied === 'install' ? 'Copied' : 'Copy'}
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
            onClick={() => handleCopy(option.snippet, 'snippet')}
            className="h-6 gap-1 px-2 text-[11px]"
          >
            {copied === 'snippet' ? <Check className="size-3" /> : <Copy className="size-3" />}
            {copied === 'snippet' ? 'Copied' : 'Copy'}
          </Button>
        </div>
        <pre className="overflow-x-auto px-4 py-3 font-mono text-[12px] leading-relaxed text-muted-foreground">
          <code>{option.snippet}</code>
        </pre>
      </div>
    </div>
  );
}
