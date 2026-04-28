/**
 * SetupGuide — on-page walkthrough that turns the harness from a
 * configured-but-idle service into a running loop. The harness accepts
 * two orchestrators that share the same provider key + critic gate:
 *
 *   1. Provider key — foundation; powers the critic and the autonomous
 *      loop. Saved in ~/.opentracy/secrets.json.
 *   2. Driver mode — pick one or both:
 *        a. Autonomous loop (Anthropic-style) — trigger engine runs
 *           recipes using the saved key. Live as soon as the key is set.
 *        b. Claude Code via MCP — operator-in-the-loop. Copy a single
 *           `claude mcp add` command on the box where Claude Code lives.
 *   3. Try it — sample prompts that work in either mode.
 *
 * The component owns its own polling. It calls the parent's
 * onConfigured callback when the critic flips ready (step 1 done) so
 * the Proposals tab can drop the empty-state and load the live list.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, Copy, Loader2, RefreshCw } from 'lucide-react';
import { toast } from 'sonner';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  useHarnessService,
  type HarnessSetupStatus,
} from '@/services/harnessService';

interface SetupGuideProps {
  /** Called when both `mcp` and `critic.ready` are satisfied so the
   * parent can swap to its post-setup view. */
  onConfigured?: () => void;
  /** When false, render a compact one-line status row instead of the
   * full three-step card. Used by the slim header strip. */
  compact?: boolean;
}

const PROVIDER_OPTIONS = [
  { value: 'anthropic', label: 'Anthropic (Claude)' },
];

function deriveMcpUrl(path: string): string {
  // Use the host the user's browser is on. Works through VS Code port
  // forwarding, public IP, or anything in between — whatever they
  // typed into the address bar is what their Claude Code can also reach.
  if (typeof window === 'undefined') return path;
  const origin = window.location.origin;
  return `${origin}${path}`;
}

export function SetupGuide({ onConfigured, compact = false }: SetupGuideProps) {
  const service = useHarnessService();
  const [status, setStatus] = useState<HarnessSetupStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [savingKey, setSavingKey] = useState(false);
  const [provider, setProvider] = useState<string>('anthropic');
  const [apiKey, setApiKey] = useState('');
  const [copied, setCopied] = useState(false);

  const refresh = useCallback(() => {
    setLoading(true);
    service.getSetupStatus().then((s) => {
      setStatus(s);
      setLoading(false);
    });
  }, [service]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Prefer the critic's provider as the default selection so the user
  // doesn't have to think about which key matters.
  useEffect(() => {
    if (status?.critic.provider) setProvider(status.critic.provider);
  }, [status?.critic.provider]);

  // Notify parent when the loop is fully ready.
  useEffect(() => {
    if (status?.critic.ready && onConfigured) onConfigured();
  }, [status?.critic.ready, onConfigured]);

  const mcpCommand = useMemo(() => {
    const url = status?.mcp ? deriveMcpUrl(status.mcp.path) : deriveMcpUrl('/mcp/');
    const name = status?.mcp.name ?? 'opentracy-harness';
    return `claude mcp add --transport http -s user ${name} ${url}`;
  }, [status]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(mcpCommand);
      setCopied(true);
      toast.success('Copied to clipboard');
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error('Copy failed — select the text manually');
    }
  };

  const handleSaveKey = async () => {
    if (!apiKey) {
      toast.error('Paste an API key first');
      return;
    }
    setSavingKey(true);
    try {
      await service.saveProviderKey(provider, apiKey.trim());
      toast.success(`${provider} key saved`);
      setApiKey('');
      refresh();
    } catch (err) {
      toast.error('Save failed', {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setSavingKey(false);
    }
  };

  const criticReady = status?.critic.ready ?? false;
  const criticProvider = status?.critic.provider ?? 'anthropic';
  const missing = status?.missing_providers ?? [];

  if (compact) {
    return (
      <div className="flex items-center gap-2 text-xs">
        <StepDot done={criticReady} />
        <span className={criticReady ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'}>
          {criticReady
            ? `Critic ready — ${status?.critic.model}`
            : `Critic offline — missing ${criticProvider} key`}
        </span>
      </div>
    );
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-base">Set up the harness</CardTitle>
        <Button
          variant="ghost"
          size="sm"
          onClick={refresh}
          disabled={loading}
          className="h-7"
        >
          <RefreshCw className={`size-3.5 mr-1 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Step 1: Provider key — foundation for both driver modes */}
        <Step
          n={1}
          done={criticReady}
          title="Add a provider key"
          subtitle={
            criticReady
              ? `${status?.critic.provider} key configured — the critic gate is live and the autonomous loop can run.`
              : `Powers the budget critic (pinned to ${status?.critic.model ?? '—'}) and the autonomous loop. Without it, every write returns rejected_by_critic.`
          }
        >
          {!criticReady && (
            <div className="space-y-2">
              <div className="flex items-end gap-2">
                <div className="w-44">
                  <Label className="text-xs text-muted-foreground">Provider</Label>
                  <Select value={provider} onValueChange={setProvider}>
                    <SelectTrigger className="h-9">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {PROVIDER_OPTIONS.map((p) => (
                        <SelectItem key={p.value} value={p.value}>
                          {p.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex-1">
                  <Label className="text-xs text-muted-foreground">API key</Label>
                  <Input
                    type="password"
                    placeholder="paste key…"
                    autoComplete="off"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    className="h-9 font-mono"
                  />
                </div>
                <Button
                  onClick={handleSaveKey}
                  disabled={savingKey || apiKey.length === 0}
                  className="h-9"
                >
                  {savingKey && <Loader2 className="size-3.5 mr-1 animate-spin" />}
                  Save key
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Stored in <code className="font-mono">~/.opentracy/secrets.json</code> on the
                server. Never sent anywhere except the provider you select.
              </p>
            </div>
          )}
          {criticReady && (
            <p className="text-xs text-emerald-600 dark:text-emerald-400">
              ✓ {status?.critic.provider} key configured.
            </p>
          )}
          {missing.length > 0 && missing.some((p) => p !== criticProvider) && (
            <p className="mt-2 text-xs text-muted-foreground">
              Other agents reference: {missing.join(', ')}. The critic alone is enough to start; add others later as needed.
            </p>
          )}
        </Step>

        {/* Step 2: Pick a driver mode (or both) */}
        <Step
          n={2}
          done={false}
          title="Pick how the harness runs"
          subtitle="Either — or both — can drive. They share the same critic gate and ledger."
        >
          <div className="grid gap-3 md:grid-cols-2">
            {/* A. Autonomous loop */}
            <div className="rounded-md border bg-muted/20 p-3 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="text-sm font-medium">Autonomous loop</div>
                <Badge
                  variant="outline"
                  className={
                    criticReady
                      ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/30 font-mono text-[10px]'
                      : 'bg-muted text-muted-foreground font-mono text-[10px]'
                  }
                >
                  {criticReady ? 'live' : 'needs key'}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                Trigger engine fires on its tick, sensors emit signals,
                policies dispatch recipes — all using the saved{' '}
                {status?.critic.provider ?? 'provider'} key. No extra setup
                once step 1 is ✓.
              </p>
            </div>

            {/* B. Claude Code via MCP */}
            <div className="rounded-md border bg-muted/20 p-3 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="text-sm font-medium">Claude Code (MCP)</div>
                <Badge
                  variant="outline"
                  className="bg-muted text-muted-foreground font-mono text-[10px]"
                >
                  optional
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                Operator-in-the-loop. Run once on the machine where Claude
                Code lives — uses HTTP transport, user scope.
              </p>
              <div className="flex items-center gap-2">
                <code className="flex-1 truncate rounded-md bg-muted px-2 py-1.5 font-mono text-[11px]">
                  {mcpCommand}
                </code>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleCopy}
                  className="h-8 shrink-0"
                >
                  {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
                </Button>
              </div>
              <p className="text-[11px] text-muted-foreground">
                Verify with <code className="font-mono">claude mcp list</code> —
                you should see{' '}
                <code className="font-mono">
                  {status?.mcp.name ?? 'opentracy-harness'}
                </code>{' '}
                ✓ Connected.
              </p>
            </div>
          </div>
        </Step>

        {/* Step 3: Try it — works for either mode */}
        <Step
          n={3}
          done={criticReady}
          title="Try it"
          subtitle="In Claude Code, ask one of these. The autonomous loop will surface the same kinds of proposals on its own cadence."
        >
          <div className="space-y-1.5 text-sm">
            <SamplePrompt text="list pending harness proposals" />
            <SamplePrompt text="list_objectives — show their current trends" />
            <SamplePrompt text="propose a run_eval action against cost_per_successful_completion" />
          </div>
        </Step>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

function Step({
  n,
  done,
  title,
  subtitle,
  children,
}: {
  n: number;
  done: boolean;
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex gap-3">
      <StepDot n={n} done={done} />
      <div className="flex-1 space-y-2">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-medium text-sm">{title}</h3>
            {done && (
              <Badge
                variant="outline"
                className="bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/30 font-mono text-[10px]"
              >
                done
              </Badge>
            )}
          </div>
          <p className="text-xs text-muted-foreground">{subtitle}</p>
        </div>
        {children}
      </div>
    </div>
  );
}

function StepDot({ n, done }: { n?: number; done: boolean }) {
  if (done)
    return (
      <div className="flex size-6 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-700 dark:text-emerald-400">
        <Check className="size-3.5" />
      </div>
    );
  if (n === undefined)
    return (
      <div className="size-2 rounded-full bg-amber-500" />
    );
  return (
    <div className="flex size-6 items-center justify-center rounded-full border bg-muted text-xs font-mono text-muted-foreground">
      {n}
    </div>
  );
}

function SamplePrompt({ text }: { text: string }) {
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success('Copied');
    } catch {
      // ignore
    }
  };
  return (
    <button
      onClick={handleCopy}
      className="block w-full rounded-md border bg-muted/30 px-3 py-1.5 text-left font-mono text-xs hover:bg-muted/60 transition-colors"
      title="Click to copy"
    >
      {text}
    </button>
  );
}
