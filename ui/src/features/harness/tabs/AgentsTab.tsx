/**
 * Agents tab — browse, inspect, and manually run harness agents.
 *
 * Left column lists agents grouped by their role folder
 * (inspectors / proposers / critics / narrators) with a search box and
 * compact rows so the page stays usable as the agent count grows.
 * Right column shows the selected agent's prompt + output schema and a
 * runner.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Bot,
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  Eye,
  FileText,
  Lightbulb,
  Loader2,
  MousePointer2,
  Play,
  Scale,
  Search,
  Wrench,
  X,
} from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from '@/components/ui/empty';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import {
  useHarnessService,
  type AgentConfig,
  type AgentRunResult,
} from '@/services/harnessService';

// ---------------------------------------------------------------------------
// role normalization + visuals
// ---------------------------------------------------------------------------

type RoleKey = 'inspector' | 'proposer' | 'critic' | 'narrator' | 'agent';

const ROLE_ORDER: RoleKey[] = ['inspector', 'proposer', 'critic', 'narrator', 'agent'];

const ROLE_META: Record<
  RoleKey,
  {
    label: string;
    description: string;
    Icon: typeof Eye;
    badgeClass: string;
  }
> = {
  inspector: {
    label: 'Inspectors',
    description: 'Read-only — surface findings, no writes',
    Icon: Eye,
    badgeClass: 'bg-violet-500/10 text-violet-700 dark:text-violet-400 border-violet-500/30',
  },
  proposer: {
    label: 'Proposers',
    description: 'Suggest writes — gated through the critic',
    Icon: Lightbulb,
    badgeClass: 'bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/30',
  },
  critic: {
    label: 'Critics',
    description: 'Judge proposals before they execute',
    Icon: Scale,
    badgeClass: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/30',
  },
  narrator: {
    label: 'Narrators',
    description: 'Summarize ledger activity for humans',
    Icon: BookOpen,
    badgeClass: 'bg-sky-500/10 text-sky-700 dark:text-sky-400 border-sky-500/30',
  },
  agent: {
    label: 'Other',
    description: 'Uncategorized agents',
    Icon: Bot,
    badgeClass: 'bg-muted text-muted-foreground border-border',
  },
};

function normalizeRole(role: string | undefined): RoleKey {
  if (!role) return 'agent';
  // Server returns plural directory names; normalize to singular keys.
  const stripped = role.toLowerCase().replace(/s$/, '');
  if (
    stripped === 'inspector' ||
    stripped === 'proposer' ||
    stripped === 'critic' ||
    stripped === 'narrator'
  ) {
    return stripped;
  }
  return 'agent';
}

// ---------------------------------------------------------------------------
// agent row
// ---------------------------------------------------------------------------

function AgentRow({
  agent,
  isSelected,
  onSelect,
}: {
  agent: AgentConfig;
  isSelected: boolean;
  onSelect: () => void;
}) {
  const role = normalizeRole(agent.role);
  const { Icon } = ROLE_META[role];
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`group flex w-full items-center gap-2.5 rounded-md border px-2.5 py-2 text-left transition-colors ${
        isSelected
          ? 'border-primary/60 bg-primary/5'
          : 'border-transparent hover:bg-accent/40 hover:border-border'
      }`}
    >
      <Icon
        className={`size-3.5 shrink-0 ${isSelected ? 'text-primary' : 'text-muted-foreground'}`}
      />
      <div className="min-w-0 flex-1">
        <div className="truncate text-xs font-medium font-mono">{agent.name}</div>
        {agent.description && (
          <div className="truncate text-[11px] text-muted-foreground">{agent.description}</div>
        )}
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------
// detail viewers
// ---------------------------------------------------------------------------

function PromptViewer({ prompt }: { prompt: string }) {
  const [expanded, setExpanded] = useState(false);
  const lines = prompt.split('\n');

  return (
    <div className="space-y-1">
      <button
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
        <FileText className="size-3" />
        System Prompt ({lines.length} lines)
      </button>
      {expanded && (
        <pre className="text-xs bg-muted p-3 rounded-md overflow-auto max-h-64 whitespace-pre-wrap">
          {prompt}
        </pre>
      )}
    </div>
  );
}

function OutputSchemaViewer({
  fields,
}: {
  fields: Record<string, { type: string; description?: string }>;
}) {
  if (Object.keys(fields).length === 0) return null;

  return (
    <div className="space-y-1">
      <span className="text-xs text-muted-foreground flex items-center gap-1">
        <Wrench className="size-3" />
        Output Schema
      </span>
      <div className="grid gap-1.5">
        {Object.entries(fields).map(([name, spec]) => (
          <div
            key={name}
            className="grid grid-cols-[auto_auto_1fr] items-baseline gap-x-2 text-xs min-w-0"
          >
            <code className="bg-muted px-1.5 py-0.5 rounded font-mono shrink-0">{name}</code>
            <span className="text-muted-foreground shrink-0">{spec.type}</span>
            {spec.description && (
              <span className="text-muted-foreground/60 wrap-break-word min-w-0">
                — {spec.description}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function ResultViewer({ result }: { result: AgentRunResult | null }) {
  const [copied, setCopied] = useState(false);

  if (!result) return null;

  const json = JSON.stringify(result.result, null, 2);

  const handleCopy = () => {
    navigator.clipboard.writeText(json);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Result</span>
        <Button variant="ghost" size="icon" className="size-7" onClick={handleCopy}>
          {copied ? <Check className="size-3 text-green-500" /> : <Copy className="size-3" />}
        </Button>
      </div>
      <pre className="text-xs bg-muted p-3 rounded-md overflow-auto max-h-80 whitespace-pre-wrap font-mono">
        {json}
      </pre>
    </div>
  );
}

// ---------------------------------------------------------------------------
// page
// ---------------------------------------------------------------------------

export function AgentsTab() {
  const service = useHarnessService();

  const [agents, setAgents] = useState<AgentConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<AgentConfig | null>(null);
  const [input, setInput] = useState('');
  const [running, setRunning] = useState(false);
  const [useTools, setUseTools] = useState(false);
  const [result, setResult] = useState<AgentRunResult | null>(null);
  const [search, setSearch] = useState('');

  useEffect(() => {
    service.listAgents().then((list) => {
      setAgents(list);
      if (list.length > 0) setSelected(list[0]);
      setLoading(false);
    });
  }, [service]);

  const handleRun = useCallback(async () => {
    if (!selected || !input.trim()) return;

    setRunning(true);
    setResult(null);
    try {
      const res = await service.runAgent(selected.name, input, useTools);
      setResult(res);
      toast.success(`Agent ${selected.name} completed`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Agent run failed');
    } finally {
      setRunning(false);
    }
  }, [selected, input, useTools, service]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return agents;
    return agents.filter(
      (a) => a.name.toLowerCase().includes(q) || (a.description ?? '').toLowerCase().includes(q)
    );
  }, [agents, search]);

  const grouped = useMemo(() => {
    const buckets = new Map<RoleKey, AgentConfig[]>();
    for (const role of ROLE_ORDER) buckets.set(role, []);
    for (const agent of filtered) {
      const key = normalizeRole(agent.role);
      buckets.get(key)!.push(agent);
    }
    for (const list of buckets.values()) {
      list.sort((a, b) => a.name.localeCompare(b.name));
    }
    return ROLE_ORDER.map((role) => ({ role, agents: buckets.get(role)! })).filter(
      (g) => g.agents.length > 0
    );
  }, [filtered]);

  const selectedRole = selected ? normalizeRole(selected.role) : 'agent';
  const SelectedIcon = ROLE_META[selectedRole].Icon;

  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-12 md:col-span-4 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-muted-foreground">
            Agents
            {!loading && agents.length > 0 && (
              <span className="ml-2 font-mono text-xs text-muted-foreground/70 tabular-nums">
                {filtered.length}/{agents.length}
              </span>
            )}
          </h3>
        </div>

        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground pointer-events-none" />
          <Input
            placeholder="Search agents…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-9 pl-8 pr-8"
          />
          {search && (
            <button
              type="button"
              onClick={() => setSearch('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              aria-label="Clear search"
            >
              <X className="size-3.5" />
            </button>
          )}
        </div>

        {loading ? (
          <div className="space-y-2">
            {[1, 2, 3, 4, 5].map((i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <Empty className="border-0 py-8">
            <EmptyHeader>
              <EmptyTitle className="text-sm">No matches</EmptyTitle>
              <EmptyDescription className="text-xs">
                Nothing for "{search}". Try a shorter query.
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        ) : (
          <ScrollArea className="h-[calc(100vh-22rem)] min-h-64 pr-2">
            <div className="space-y-4">
              {grouped.map(({ role, agents: list }) => {
                const meta = ROLE_META[role];
                const RoleIcon = meta.Icon;
                return (
                  <div key={role} className="space-y-1.5">
                    <div className="flex items-center gap-2 px-1">
                      <RoleIcon className="size-3.5 text-muted-foreground" />
                      <span className="text-[11px] uppercase tracking-wider font-medium text-muted-foreground">
                        {meta.label}
                      </span>
                      <span className="font-mono text-[10px] text-muted-foreground/70 tabular-nums">
                        {list.length}
                      </span>
                    </div>
                    <div className="space-y-0.5">
                      {list.map((agent) => (
                        <AgentRow
                          key={agent.name}
                          agent={agent}
                          isSelected={selected?.name === agent.name}
                          onSelect={() => {
                            setSelected(agent);
                            setResult(null);
                          }}
                        />
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </ScrollArea>
        )}
      </div>

      <div className="col-span-12 md:col-span-8 space-y-4">
        {selected ? (
          <>
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-start gap-3 flex-wrap">
                  <div className="flex items-center gap-2 min-w-0 flex-1">
                    <SelectedIcon className="size-5 text-primary shrink-0" />
                    <CardTitle className="font-mono break-all leading-tight">
                      {selected.name}
                    </CardTitle>
                  </div>
                  <div className="flex items-center gap-1.5 flex-wrap shrink-0">
                    <Badge
                      variant="outline"
                      className={`font-mono text-[10px] ${ROLE_META[selectedRole].badgeClass}`}
                    >
                      {ROLE_META[selectedRole].label.replace(/s$/, '').toLowerCase()}
                    </Badge>
                    <Badge variant="secondary" className="font-mono text-[10px] max-w-45 truncate">
                      {selected.model}
                    </Badge>
                    <Badge variant="outline" className="font-mono text-[10px]">
                      temp {selected.temperature}
                    </Badge>
                    <Badge variant="outline" className="font-mono text-[10px]">
                      {selected.output_schema.type}
                    </Badge>
                  </div>
                </div>
                {selected.description && (
                  <p className="text-sm text-muted-foreground pt-2">{selected.description}</p>
                )}
              </CardHeader>
              <CardContent className="space-y-3">
                <PromptViewer prompt={selected.system_prompt} />
                <OutputSchemaViewer fields={selected.output_schema.fields} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Run Agent</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Textarea
                  className="min-h-32 font-mono text-sm resize-y"
                  placeholder={`Enter input for ${selected.name}…`}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                />
                <div className="flex items-center justify-between gap-3">
                  <Label
                    htmlFor="use-tools"
                    className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer"
                  >
                    <Switch id="use-tools" checked={useTools} onCheckedChange={setUseTools} />
                    Enable tools (multi-turn)
                  </Label>
                  <Button size="sm" onClick={handleRun} disabled={running || !input.trim()}>
                    {running ? (
                      <Loader2 className="size-4 animate-spin" />
                    ) : (
                      <Play className="size-4" />
                    )}
                    {running ? 'Running…' : 'Run'}
                  </Button>
                </div>

                <ResultViewer result={result} />
              </CardContent>
            </Card>
          </>
        ) : (
          <Empty>
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <MousePointer2 />
              </EmptyMedia>
              <EmptyTitle>Select an agent</EmptyTitle>
              <EmptyDescription>
                Pick an agent on the left to view its prompt, output schema, and run it manually
                with custom input.
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        )}
      </div>
    </div>
  );
}
