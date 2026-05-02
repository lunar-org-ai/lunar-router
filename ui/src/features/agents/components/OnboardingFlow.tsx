import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Copy,
  Headphones,
  Library,
  type LucideIcon,
  Play,
  Plug,
  Sparkles,
  Telescope,
  Terminal,
} from 'lucide-react';
import { CrewAI, LangChain, LangGraph, OpenAI } from '@lobehub/icons';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { cn } from '@/lib/utils';
import {
  type AgentFramework,
  FRAMEWORK_OPTIONS,
  type FrameworkOption,
  type OnboardingMode,
} from '@/features/agents/state';
import { AGENT_TEMPLATES, type AgentTemplate } from '@/features/agents/templates';

const FRAMEWORK_ICONS: Record<AgentFramework, React.ComponentType<{ size?: number }>> = {
  langchain: LangChain.Avatar,
  langgraph: LangGraph.Avatar,
  crewai: CrewAI.Avatar,
  'openai-agents': OpenAI.Avatar,
};

type TemplateIconMeta = {
  Icon: LucideIcon;
  tone: string;
};

const TEMPLATE_ICONS: Record<string, TemplateIconMeta> = {
  'support-bot': { Icon: Headphones, tone: 'bg-amber-500/10 text-amber-500' },
  'rag-agent': { Icon: Library, tone: 'bg-sky-500/10 text-sky-500' },
  'research-agent': { Icon: Telescope, tone: 'bg-violet-500/10 text-violet-500' },
};

type OnboardingFlowProps = {
  mode: OnboardingMode;
  step: 1 | 2;
  framework: AgentFramework | null;
  templateId: string | null;
  name: string;
  onSelectFramework: (framework: AgentFramework) => void;
  onSelectTemplate: (templateId: string) => void;
  onSetName: (name: string) => void;
  onAdvance: () => void;
  onRunSimulation: () => void;
  onCancel: () => void;
};

export function OnboardingFlow({
  mode,
  step,
  framework,
  templateId,
  name,
  onSelectFramework,
  onSelectTemplate,
  onSetName,
  onAdvance,
  onRunSimulation,
  onCancel,
}: OnboardingFlowProps) {
  const isImport = mode === 'import';
  const selectedFramework = isImport
    ? FRAMEWORK_OPTIONS.find((option) => option.id === framework) ?? null
    : null;
  const selectedTemplate = !isImport
    ? AGENT_TEMPLATES.find((template) => template.id === templateId) ?? null
    : null;

  const advanceDisabled = step === 1 ? (isImport ? !framework : !templateId) : false;

  return (
    <div className="flex flex-1 items-center justify-center px-8 py-12">
      <div className="flex w-full max-w-xl flex-col gap-6">
        <Stepper mode={mode} current={step} />

        <AnimatePresence mode="wait">
          {step === 1 && isImport ? (
            <Pane key="import-1" direction="forward">
              <Heading
                title="Choose a framework"
                description="Pick the framework your agent is built with — OpenTracy will auto-instrument it."
              />
              <FrameworkPicker selectedId={framework} onSelect={onSelectFramework} />
            </Pane>
          ) : null}

          {step === 1 && !isImport ? (
            <Pane key="create-1" direction="forward">
              <Heading
                title="Pick a template"
                description="Start from a pre-built topology and tweak it later. Each template ships with sample evaluations."
              />
              <TemplatePicker selectedId={templateId} onSelect={onSelectTemplate} />
            </Pane>
          ) : null}

          {step === 2 && isImport && selectedFramework ? (
            <Pane key="import-2" direction="backward">
              <Heading
                title="Add to your code"
                description={`Drop this snippet into your ${selectedFramework.name} app, then run a simulation to preview the topology.`}
              />
              <CodeStep option={selectedFramework} />
            </Pane>
          ) : null}

          {step === 2 && !isImport && selectedTemplate ? (
            <Pane key="create-2" direction="backward">
              <Heading
                title="Name this agent"
                description={`You're creating a "${selectedTemplate.name}". Give it a name and an optional description before generating it.`}
              />
              <NameStep template={selectedTemplate} name={name} onChange={onSetName} />
            </Pane>
          ) : null}
        </AnimatePresence>

        <Footer>
          {step === 1 ? (
            <Button variant="ghost" onClick={onCancel} className="text-muted-foreground">
              Cancel
            </Button>
          ) : (
            <Button variant="ghost" onClick={onCancel} className="gap-1.5 text-muted-foreground">
              <ArrowLeft className="size-3.5" />
              Cancel
            </Button>
          )}

          {step === 1 ? (
            <Button onClick={onAdvance} disabled={advanceDisabled} className="gap-1.5">
              Continue
              <ArrowRight className="size-3.5" />
            </Button>
          ) : (
            <Button onClick={onRunSimulation} className="gap-1.5">
              {isImport ? (
                <>
                  <Play className="size-3.5 fill-current" />
                  Run Simulation
                </>
              ) : (
                <>
                  <Sparkles className="size-3.5" />
                  Generate Agent
                </>
              )}
            </Button>
          )}
        </Footer>
      </div>
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
  mode: OnboardingMode;
  current: 1 | 2;
};

function Stepper({ mode, current }: StepperProps) {
  const labels = mode === 'import' ? ['Framework', 'Connect'] : ['Template', 'Name'];
  const Icon = mode === 'import' ? Plug : Sparkles;
  return (
    <div className="flex items-center gap-3 text-[11px] uppercase tracking-wider text-muted-foreground/70">
      <span className="flex items-center gap-1.5 text-foreground">
        <Icon className="size-3" />
        {mode === 'import' ? 'Import' : 'Create'}
      </span>
      <span className="h-px w-4 bg-border/50" />
      <StepDot index={1} current={current} label={labels[0]} />
      <span className="h-px w-6 bg-border/50" />
      <StepDot index={2} current={current} label={labels[1]} />
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

type TemplatePickerProps = {
  selectedId: string | null;
  onSelect: (id: string) => void;
};

function TemplatePicker({ selectedId, onSelect }: TemplatePickerProps) {
  return (
    <div className="flex flex-col gap-2">
      {AGENT_TEMPLATES.map((template) => {
        const active = template.id === selectedId;
        const meta = TEMPLATE_ICONS[template.id];
        const Icon = meta?.Icon;
        return (
          <button
            key={template.id}
            type="button"
            onClick={() => onSelect(template.id)}
            className={cn(
              'flex items-start gap-3 rounded-xl border bg-card/30 px-3.5 py-3 text-left transition-colors',
              active
                ? 'border-foreground/40 bg-card/60'
                : 'border-border/40 hover:border-border/70 hover:bg-card/50'
            )}
          >
            {Icon ? (
              <span
                className={cn(
                  'flex size-9 shrink-0 items-center justify-center rounded-lg',
                  meta.tone
                )}
              >
                <Icon className="size-4" />
              </span>
            ) : null}
            <div className="flex min-w-0 flex-1 flex-col gap-0.5">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm font-medium leading-tight">{template.name}</span>
                <span className="font-mono text-[11px] text-muted-foreground">
                  {template.nodes.length} nodes
                </span>
              </div>
              <span className="font-mono text-[11px] leading-relaxed text-muted-foreground">
                {template.description}
              </span>
            </div>
            {active ? (
              <Check className="mt-1 size-3.5 shrink-0 text-emerald-500" />
            ) : null}
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

type NameStepProps = {
  template: AgentTemplate;
  name: string;
  onChange: (name: string) => void;
};

function NameStep({ template, name, onChange }: NameStepProps) {
  const [description, setDescription] = useState(template.description);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-1.5">
        <label htmlFor="agent-name" className="text-xs text-muted-foreground">
          Name
        </label>
        <Input
          id="agent-name"
          value={name}
          onChange={(event) => onChange(event.target.value)}
          placeholder={template.name}
          className="font-mono text-sm"
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <label htmlFor="agent-description" className="text-xs text-muted-foreground">
          Description <span className="text-muted-foreground/60">(optional)</span>
        </label>
        <Textarea
          id="agent-description"
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          rows={3}
          className="font-mono text-xs leading-relaxed"
        />
      </div>
    </div>
  );
}
