import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { ArrowLeft, ArrowRight, Check, Copy, Plug, Terminal } from 'lucide-react';
import { CrewAI, LangChain, LangGraph, OpenAI } from '@lobehub/icons';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import {
  FRAMEWORK_OPTIONS,
  type FrameworkOption,
} from '@/features/agents/state';
import type { AgentFramework } from '@/features/agents/types';
import type { ImportPhase } from '@/features/agents/hooks/useAgentImport';

const FRAMEWORK_ICONS: Record<AgentFramework, React.ComponentType<{ size?: number }>> = {
  langchain: LangChain.Avatar,
  langgraph: LangGraph.Avatar,
  crewai: CrewAI.Avatar,
  'openai-agents': OpenAI.Avatar,
};

type OnboardingFlowProps = {
  phase: Exclude<ImportPhase, 'closed'>;
  step: 1 | 2;
  framework: AgentFramework | null;
  name: string;
  onSelectFramework: (framework: AgentFramework) => void;
  onSetName: (name: string) => void;
  onAdvance: () => void;
  onBack: () => void;
  onSubmit: () => void;
  onCancel: () => void;
};

export function OnboardingFlow({
  phase,
  step,
  framework,
  name,
  onSelectFramework,
  onSetName,
  onAdvance,
  onBack,
  onSubmit,
  onCancel,
}: OnboardingFlowProps) {
  if (phase === 'connecting') {
    return <ConnectingPane />;
  }

  const selectedFramework =
    FRAMEWORK_OPTIONS.find((option) => option.id === framework) ?? null;
  const advanceDisabled = step === 1 && !framework;

  return (
    <div className="flex flex-1 items-center justify-center px-8 py-12">
      <div className="flex w-full max-w-xl flex-col gap-6">
        <Stepper current={step} />

        <AnimatePresence mode="wait">
          {step === 1 ? (
            <Pane key="step-1" direction="forward">
              <Heading
                title="Choose a framework"
                description="Pick the framework your agent is built with — OpenTracy will auto-instrument it."
              />
              <FrameworkPicker selectedId={framework} onSelect={onSelectFramework} />
            </Pane>
          ) : null}

          {step === 2 && selectedFramework ? (
            <Pane key="step-2" direction="backward">
              <Heading
                title="Add to your code"
                description={`Pick a project name, drop this snippet into your ${selectedFramework.name} app, and OpenTracy will use that name as the agent.`}
              />
              <CodeStep
                option={selectedFramework}
                projectName={name}
                onProjectNameChange={onSetName}
              />
            </Pane>
          ) : null}
        </AnimatePresence>

        <Footer>
          {step === 1 ? (
            <Button variant="ghost" onClick={onCancel} className="text-muted-foreground">
              Cancel
            </Button>
          ) : (
            <Button variant="ghost" onClick={onBack} className="gap-1.5 text-muted-foreground">
              <ArrowLeft className="size-3.5" />
              Back
            </Button>
          )}

          {step === 1 ? (
            <Button onClick={onAdvance} disabled={advanceDisabled} className="gap-1.5">
              Continue
              <ArrowRight className="size-3.5" />
            </Button>
          ) : (
            <Button onClick={onSubmit} className="gap-1.5">
              <Plug className="size-3.5" />
              Connect
            </Button>
          )}
        </Footer>
      </div>
    </div>
  );
}

function ConnectingPane() {
  return (
    <div className="flex flex-1 items-center justify-center px-8 py-12">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
        className="flex flex-col items-center gap-5"
      >
        <div className="relative flex size-16 items-center justify-center">
          <motion.div
            aria-hidden
            className="absolute inset-0 rounded-full border-2 border-border/30 border-t-foreground/70"
            animate={{ rotate: 360 }}
            transition={{ duration: 1.4, ease: 'linear', repeat: Infinity }}
          />
          <Plug className="size-5 text-foreground/80" />
        </div>
        <div className="flex flex-col items-center gap-1">
          <span className="text-sm font-medium tracking-tight">Instrumenting your agent</span>
          <span className="font-mono text-[11px] text-muted-foreground/70">
            Connecting to OpenTracy…
          </span>
        </div>
      </motion.div>
    </div>
  );
}

type PaneProps = {
  direction: 'forward' | 'backward';
  children: React.ReactNode;
};

function Pane({ direction, children }: PaneProps) {
  const offset = direction === 'forward' ? -10 : 10;
  return (
    <motion.div
      initial={{ opacity: 0, x: -offset }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: offset }}
      transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
      className="flex flex-col gap-5"
    >
      {children}
    </motion.div>
  );
}

type StepperProps = {
  current: 1 | 2;
};

function Stepper({ current }: StepperProps) {
  return (
    <div className="flex items-center gap-3 text-[11px] uppercase tracking-wider text-muted-foreground/70">
      <span className="flex items-center gap-1.5 text-foreground">
        <Plug className="size-3" />
        Import
      </span>
      <span className="h-px w-4 bg-border/50" />
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
        const Icon = FRAMEWORK_ICONS[option.id];
        return (
          <button
            key={option.id}
            type="button"
            onClick={() => onSelect(option.id)}
            className={cn(
              'flex items-center gap-3 rounded-xl border bg-card/30 px-3.5 py-3 text-left transition-colors',
              active
                ? 'border-foreground/40 bg-card/60'
                : 'border-border/40 hover:border-border/70 hover:bg-card/50'
            )}
          >
            <span className="flex size-9 shrink-0 items-center justify-center overflow-hidden rounded-lg">
              <Icon size={36} />
            </span>
            <div className="flex min-w-0 flex-1 flex-col gap-0.5">
              <span className="text-sm font-medium leading-tight">{option.name}</span>
              <span className="truncate font-mono text-[11px] text-muted-foreground">
                {option.pkg}
              </span>
            </div>
            {active ? <Check className="size-3.5 shrink-0 text-emerald-500" /> : null}
          </button>
        );
      })}
    </div>
  );
}

type CodeStepProps = {
  option: FrameworkOption;
  projectName: string;
  onProjectNameChange: (value: string) => void;
};

function CodeStep({ option, projectName, onProjectNameChange }: CodeStepProps) {
  const [copied, setCopied] = useState<'install' | 'snippet' | null>(null);

  const installCommand = `pip install ${option.pkg}`;
  const snippet = option.buildSnippet(projectName);

  const handleCopy = (text: string, target: 'install' | 'snippet') => {
    if (typeof navigator === 'undefined' || !navigator.clipboard) return;
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(target);
      setTimeout(() => setCopied(null), 1400);
    });
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <label htmlFor="project-name" className="text-xs text-muted-foreground">
          Project name
        </label>
        <Input
          id="project-name"
          value={projectName}
          onChange={(event) => onProjectNameChange(event.target.value)}
          placeholder="my-agent"
          className="font-mono text-sm"
        />
        <span className="font-mono text-[11px] text-muted-foreground/70">
          This name flows through to <code>register(project=...)</code> and becomes the agent
          name in OpenTracy.
        </span>
      </div>

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
            onClick={() => handleCopy(snippet, 'snippet')}
            className="h-6 gap-1 px-2 text-[11px]"
          >
            {copied === 'snippet' ? <Check className="size-3" /> : <Copy className="size-3" />}
            {copied === 'snippet' ? 'Copied' : 'Copy'}
          </Button>
        </div>
        <pre className="overflow-x-auto px-4 py-3 font-mono text-[12px] leading-relaxed text-muted-foreground">
          <code>{snippet}</code>
        </pre>
      </div>
    </div>
  );
}
