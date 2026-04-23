/**
 * Agents tab — browse, inspect, and manually run harness agents.
 * Extracted verbatim from the original HarnessPage so existing
 * workflow (list agents → select → run) stays intact when the page
 * moved to a tabbed layout.
 */

import { useCallback, useEffect, useState } from 'react';
import {
  Bot,
  Play,
  Loader2,
  Copy,
  Check,
  ChevronDown,
  ChevronRight,
  FileText,
  Wrench,
} from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import {
  useHarnessService,
  type AgentConfig,
  type AgentRunResult,
} from '@/services/harnessService';

function AgentCard({
  agent,
  isSelected,
  onSelect,
}: {
  agent: AgentConfig;
  isSelected: boolean;
  onSelect: () => void;
}) {
  return (
    <Card
      className={`cursor-pointer transition-colors hover:border-primary/40 ${
        isSelected ? 'border-primary bg-accent/30' : ''
      }`}
      onClick={onSelect}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bot className="size-4 text-primary" />
            <CardTitle className="text-sm font-medium">{agent.name}</CardTitle>
          </div>
          <Badge variant="secondary" className="text-xs">
            {agent.model}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <p className="text-xs text-muted-foreground">{agent.description}</p>
        <div className="flex gap-2 mt-2">
          <Badge variant="outline" className="text-xs">
            temp: {agent.temperature}
          </Badge>
          <Badge variant="outline" className="text-xs">
            {agent.output_schema.type}
          </Badge>
          {Object.keys(agent.output_schema.fields).length > 0 && (
            <Badge variant="outline" className="text-xs">
              {Object.keys(agent.output_schema.fields).length} fields
            </Badge>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

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
      <div className="grid gap-1">
        {Object.entries(fields).map(([name, spec]) => (
          <div key={name} className="flex items-center gap-2 text-xs">
            <code className="bg-muted px-1.5 py-0.5 rounded font-mono">{name}</code>
            <span className="text-muted-foreground">{spec.type}</span>
            {spec.description && (
              <span className="text-muted-foreground/60">— {spec.description}</span>
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

export function AgentsTab() {
  const service = useHarnessService();

  const [agents, setAgents] = useState<AgentConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<AgentConfig | null>(null);
  const [input, setInput] = useState('');
  const [running, setRunning] = useState(false);
  const [useTools, setUseTools] = useState(false);
  const [result, setResult] = useState<AgentRunResult | null>(null);

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

  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-4 space-y-3">
        <h3 className="text-sm font-medium text-muted-foreground">Agents</h3>
        {loading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <Card key={i}>
                <CardHeader>
                  <Skeleton className="h-4 w-24" />
                </CardHeader>
                <CardContent>
                  <Skeleton className="h-3 w-full" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          agents.map((agent) => (
            <AgentCard
              key={agent.name}
              agent={agent}
              isSelected={selected?.name === agent.name}
              onSelect={() => {
                setSelected(agent);
                setResult(null);
              }}
            />
          ))
        )}
      </div>

      <div className="col-span-8 space-y-4">
        {selected ? (
          <>
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <Bot className="size-5 text-primary" />
                  <CardTitle>{selected.name}</CardTitle>
                </div>
                <p className="text-sm text-muted-foreground">{selected.description}</p>
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
                <textarea
                  className="w-full min-h-32 p-3 text-sm bg-muted rounded-md border-0 resize-y font-mono focus:outline-none focus:ring-1 focus:ring-primary"
                  placeholder={`Enter input for ${selected.name}...`}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                />
                <div className="flex items-center justify-between">
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={useTools}
                      onChange={(e) => setUseTools(e.target.checked)}
                      className="rounded"
                    />
                    Enable tools (multi-turn)
                  </label>
                  <Button size="sm" onClick={handleRun} disabled={running || !input.trim()}>
                    {running ? (
                      <Loader2 className="size-4 animate-spin" />
                    ) : (
                      <Play className="size-4" />
                    )}
                    {running ? 'Running...' : 'Run'}
                  </Button>
                </div>

                <ResultViewer result={result} />
              </CardContent>
            </Card>
          </>
        ) : (
          <div className="flex items-center justify-center h-64 text-muted-foreground">
            Select an agent to get started
          </div>
        )}
      </div>
    </div>
  );
}
