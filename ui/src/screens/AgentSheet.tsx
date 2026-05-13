/**
 * AgentSheet — slide-over for Brain / Hands / Channels / Keys.
 *
 * Ported from the OpenTracy Evolution design bundle (screens/AgentSheet.jsx).
 * Layout proportions match the source: 560px wide right-anchored panel,
 * 24px section padding, 28px section gaps, 18px head padding.
 */

import { useEffect, useState } from 'react';
import { Icon, type IconName } from '../components/Icon';
import { Tag } from '../components/Tag';
import {
  ApiError,
  addMCPServer,
  connectApiChannel,
  connectWebChannel,
  connectWhatsAppChannel,
  disconnectApiChannel,
  disconnectSlackChannel,
  disconnectWebChannel,
  disconnectWhatsAppChannel,
  discoverMCPTools,
  getAgentChannels,
  getAgentImprovement,
  getAgentSecrets,
  getApiChannel,
  getSlackChannel,
  getWebChannel,
  getWhatsAppChannel,
  listAgents,
  listMCPServers,
  putAgentImprovement,
  putAgentSecrets,
  removeMCPServer,
  rotateApiChannel,
  rotateWebChannelSecret,
  updateAgent,
  updateMCPServer,
  updateWebChannel,
  type AgentChannelsResponse,
  type AgentSecretsResponse,
  type AgentSummary,
  type ApiChannelConnectResponse,
  type ApiChannelStatus,
  type ImprovementConfig,
  type MCPServer,
  type MCPTool,
  type SlackChannelStatus,
  type WebChannelConnectResponse,
  type WebChannelStatus,
  type WebWidgetSettings,
  type WhatsAppChannelStatus,
} from '../api';

type Tab = 'brain' | 'hands' | 'mouths' | 'keys';

interface Tool {
  id: string;
  name: string;
  desc: string;
  on: boolean;
  src: 'code' | 'mcp' | 'builtin';
}

interface Channel {
  id: string;
  name: string;
  desc: string;
  on: boolean;
  vol: string | null;
}

interface Key {
  id: string;
  name: string;
  mask: string;
  valid: boolean;
}

const TABS: Array<{ id: Tab; label: string }> = [
  { id: 'brain', label: 'Brain' },
  { id: 'hands', label: 'Hands' },
  { id: 'mouths', label: 'Channels' },
  { id: 'keys', label: 'Keys' },
];

const MODELS: Array<{ id: string; name: string; meta: string }> = [
  { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5', meta: 'Anthropic · fast + cheap' },
  { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6', meta: 'Anthropic · default' },
  { id: 'claude-opus-4-7', name: 'Claude Opus 4.7', meta: 'Anthropic · strongest reasoning' },
  { id: 'gpt-4o-mini', name: 'GPT-4o mini', meta: 'OpenAI · cheap + capable' },
  { id: 'gpt-4o', name: 'GPT-4o', meta: 'OpenAI · multimodal default' },
  { id: 'gpt-5', name: 'GPT-5', meta: 'OpenAI · frontier' },
];

const INITIAL_PROMPT = `You are a customer support agent for an online store.
You are concise, friendly, and action-oriented.
When the customer is frustrated, acknowledge once and pivot to action.`;

const INITIAL_TOOLS: Tool[] = [
  { id: 't1', name: 'lookup_order', desc: 'tools/lookup_order.py', on: true, src: 'code' },
  { id: 't2', name: 'create_refund', desc: 'tools/create_refund.py', on: true, src: 'code' },
  { id: 't3', name: 'shopify-mcp', desc: '12 tools · MCP server', on: true, src: 'mcp' },
  { id: 't4', name: 'web_search', desc: 'Built-in', on: false, src: 'builtin' },
];

const INITIAL_CHANNELS: Channel[] = [
  { id: 'whatsapp', name: 'WhatsApp', desc: '+55 11 9 4002-8922', on: true, vol: '~140 / day' },
  { id: 'api', name: 'REST API', desc: 'api.opentracy.dev/v1/chat', on: true, vol: '~80 / day' },
  { id: 'widget', name: 'Web widget', desc: 'embed.opentracy.dev/w/8af2', on: true, vol: '~60 / day' },
  { id: 'slack', name: 'Slack', desc: 'Not connected', on: false, vol: null },
];

const INITIAL_KEYS: Key[] = [
  { id: 'k1', name: 'Anthropic', mask: 'sk-ant-…f72a', valid: true },
  { id: 'k2', name: 'OpenAI', mask: 'sk-…1c0d', valid: true },
  { id: 'k3', name: 'Stripe', mask: 'sk_live_…b3', valid: true },
];

const toolIcon = (src: Tool['src']): IconName =>
  src === 'mcp' ? 'route' : src === 'builtin' ? 'sparkles' : 'code';

const channelIcon = (id: Channel['id']): IconName => (id === 'api' ? 'code' : 'chat');

type Drill = null | 'widget' | 'slack' | 'whatsapp' | 'api';

export const AgentSheet = ({ onClose }: { onClose: () => void }) => {
  const [tab, setTab] = useState<Tab>('brain');
  const [model, setModel] = useState('claude-sonnet-4-6');
  const [prompt, setPrompt] = useState(INITIAL_PROMPT);
  const [tools, setTools] = useState<Tool[]>(INITIAL_TOOLS);
  const [channels, setChannels] = useState<Channel[]>(INITIAL_CHANNELS);
  const [keys] = useState<Key[]>(INITIAL_KEYS);
  const [drill, setDrill] = useState<Drill>(null);
  const [activeAgent, setActiveAgent] = useState<AgentSummary | null>(null);
  const [savingModel, setSavingModel] = useState(false);
  const [secrets, setSecrets] = useState<AgentSecretsResponse | null>(null);
  const [anthropicDraft, setAnthropicDraft] = useState('');
  const [openaiDraft, setOpenaiDraft] = useState('');
  const [savingKey, setSavingKey] = useState<'anthropic' | 'openai' | null>(null);
  const [keyError, setKeyError] = useState<string | null>(null);

  const loadSecrets = async (id: string) => {
    try {
      const next = await getAgentSecrets(id);
      setSecrets(next);
    } catch (e) {
      if (!(e instanceof ApiError)) console.warn('getAgentSecrets failed', e);
    }
  };

  // Load the active agent on mount so the Brain tab reflects the
  // operator's real configuration, not a hardcoded default.
  useEffect(() => {
    let cancelled = false;
    void listAgents()
      .then(async (res) => {
        if (cancelled) return;
        const active = res.agents.find((a) => a.id === res.active) ?? null;
        if (active) {
          setActiveAgent(active);
          setModel(active.model);
          await loadSecrets(active.id);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const pickModel = async (next: string) => {
    if (!activeAgent || next === model || savingModel) return;
    const prev = model;
    setModel(next);  // optimistic
    setSavingModel(true);
    try {
      await updateAgent(activeAgent.id, { model: next });
    } catch (e) {
      // Revert on failure so the UI doesn't lie about saved state.
      setModel(prev);
      if (!(e instanceof ApiError)) console.warn('updateAgent failed', e);
    } finally {
      setSavingModel(false);
    }
  };

  const toggleTool = (id: string) =>
    setTools((ts) => ts.map((x) => (x.id === id ? { ...x, on: !x.on } : x)));
  const toggleChannel = (id: string) =>
    setChannels((cs) => cs.map((x) => (x.id === id ? { ...x, on: !x.on } : x)));

  const wide = drill !== null;

  return (
    <>
      <div className="sheet-backdrop" onClick={onClose} />
      <div className={`sheet ${wide ? 'wide' : ''}`} role="dialog" aria-label="Agent settings">
        <div className="sheet-head">
          <div className="sidebar-mark" style={{ width: 26, height: 26, borderRadius: 8 }} />
          <div style={{ flex: 1 }}>
            <h2>support-agent</h2>
            <div className="dim" style={{ fontSize: 12, marginTop: 2 }}>
              <span className="mono">v0.40</span> · live ·{' '}
              <span style={{ color: 'var(--accent-fg)' }}>● healthy</span>
            </div>
          </div>
          <button className="btn ghost sm" onClick={onClose} aria-label="Close">
            <Icon name="x" size={14} />
          </button>
        </div>

        {/* Tabs hide while drilled — each channel panel owns its own back nav. */}
        {!drill && (
          <div className="sheet-tabs">
            {TABS.map((t) => (
              <button
                key={t.id}
                className={`tab ${tab === t.id ? 'active' : ''}`}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>
        )}

        {drill && activeAgent ? (
          drill === 'widget' ? (
            <WebWidgetPanel agentId={activeAgent.id} onBack={() => setDrill(null)} />
          ) : drill === 'slack' ? (
            <SlackChannelPanel agentId={activeAgent.id} onBack={() => setDrill(null)} />
          ) : drill === 'whatsapp' ? (
            <WhatsAppChannelPanel agentId={activeAgent.id} onBack={() => setDrill(null)} />
          ) : (
            <ApiChannelPanel agentId={activeAgent.id} onBack={() => setDrill(null)} />
          )
        ) : (
        <div className="sheet-body">
          {tab === 'brain' && (
            <>
              <div className="sheet-section">
                <h3>System prompt</h3>
                <p className="desc">
                  What the agent always knows. Claude Code edits this file when it learns something
                  new — your changes here become its starting point.
                </p>
                <textarea
                  className="prompt"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                />
                <div style={{ display: 'flex', gap: 8, marginTop: 10, alignItems: 'center' }}>
                  <span className="dim mono" style={{ fontSize: 11.5 }}>
                    system_prompt.md · {prompt.split('\n').length} lines
                  </span>
                  <button className="btn sm" style={{ marginLeft: 'auto' }}>
                    <Icon name="code" size={12} /> Open in Claude Code
                  </button>
                </div>
              </div>

              <div className="sheet-section">
                <h3>Default model</h3>
                <p className="desc">
                  The router can override this per request — see Routing in Policies.
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  {MODELS.map((m) => (
                    <button
                      key={m.id}
                      className="row-item"
                      style={{
                        cursor: savingModel ? 'progress' : 'pointer',
                        borderColor: model === m.id ? 'var(--foreground)' : undefined,
                        margin: 0,
                        textAlign: 'left',
                        opacity: savingModel ? 0.7 : 1,
                      }}
                      onClick={() => void pickModel(m.id)}
                      disabled={savingModel}
                    >
                      <div className="rmain">
                        <div className="rname">{m.name}</div>
                        <div className="rmeta">{m.meta}</div>
                      </div>
                      {model === m.id && <Icon name="check" size={14} />}
                    </button>
                  ))}
                </div>
                {activeAgent && (
                  <div className="dim" style={{ fontSize: 11.5, marginTop: 8 }}>
                    Changes write to <span className="mono">agents/{activeAgent.id}/pipeline/route.yaml</span> immediately.
                    Live <span className="mono">/run</span> picks them up on the next activate.
                  </div>
                )}
              </div>

              {activeAgent && (
                <BYOKSection
                  agent={activeAgent}
                  secrets={secrets}
                  anthropicDraft={anthropicDraft}
                  openaiDraft={openaiDraft}
                  setAnthropicDraft={setAnthropicDraft}
                  setOpenaiDraft={setOpenaiDraft}
                  saving={savingKey}
                  error={keyError}
                  onSave={async (provider) => {
                    setSavingKey(provider);
                    setKeyError(null);
                    try {
                      const body = provider === 'anthropic'
                        ? { anthropic: anthropicDraft }
                        : { openai: openaiDraft };
                      const next = await putAgentSecrets(activeAgent.id, body);
                      setSecrets(next);
                      if (provider === 'anthropic') setAnthropicDraft('');
                      else setOpenaiDraft('');
                    } catch (e) {
                      setKeyError(e instanceof Error ? e.message : String(e));
                    } finally {
                      setSavingKey(null);
                    }
                  }}
                  onRemove={async (provider) => {
                    setSavingKey(provider);
                    setKeyError(null);
                    try {
                      const body = provider === 'anthropic'
                        ? { anthropic: '' }
                        : { openai: '' };
                      const next = await putAgentSecrets(activeAgent.id, body);
                      setSecrets(next);
                    } catch (e) {
                      setKeyError(e instanceof Error ? e.message : String(e));
                    } finally {
                      setSavingKey(null);
                    }
                  }}
                />
              )}

              <div className="sheet-section">
                <h3>Self-improvement engineer</h3>
                <p className="desc">
                  The brain that reads traces, drafts changes, runs evals, and opens proposals.
                  Pick which transport powers it + which model it uses.
                </p>
                {activeAgent && (
                  <ImprovementSection agentId={activeAgent.id} />
                )}
              </div>
            </>
          )}

          {tab === 'hands' && activeAgent && <HandsTab agentId={activeAgent.id} />}

          {tab === 'mouths' && activeAgent && (
            <ChannelsTab
              agentId={activeAgent.id}
              onConfigureWidget={() => setDrill('widget')}
              onConnectSlack={() => setDrill('slack')}
              onConfigureWhatsApp={() => setDrill('whatsapp')}
              onConfigureApi={() => setDrill('api')}
            />
          )}

          {tab === 'keys' && (
            <div className="sheet-section">
              <h3>API keys</h3>
              <p className="desc">Stored encrypted. Used by the model router and tools.</p>
              {keys.map((k) => (
                <div className="row-item on" key={k.id}>
                  <div className="ricon">
                    <Icon name="shield" size={14} />
                  </div>
                  <div className="rmain">
                    <div className="rname">{k.name}</div>
                    <div className="rmeta mono">{k.mask}</div>
                  </div>
                  {k.valid ? (
                    <Tag kind="success">
                      <span className="dot" />
                      Valid
                    </Tag>
                  ) : (
                    <Tag kind="bad">
                      <span className="dot" />
                      Invalid
                    </Tag>
                  )}
                </div>
              ))}
              <button className="add-btn" style={{ marginTop: 8 }}>
                + Add API key
              </button>
            </div>
          )}
        </div>
        )}
      </div>
    </>
  );
};

// ─── BYOK section (P3.1) ────────────────────────────────────────
// Per-agent Anthropic + OpenAI keys. Resolution status comes from the
// server (per-agent | global | unset) so the operator sees which keys
// are wired and where they live before they overwrite.
interface BYOKSectionProps {
  agent: AgentSummary;
  secrets: AgentSecretsResponse | null;
  anthropicDraft: string;
  openaiDraft: string;
  setAnthropicDraft: (s: string) => void;
  setOpenaiDraft: (s: string) => void;
  saving: 'anthropic' | 'openai' | null;
  error: string | null;
  onSave: (provider: 'anthropic' | 'openai') => Promise<void>;
  onRemove: (provider: 'anthropic' | 'openai') => Promise<void>;
}

const BYOKSection = ({
  agent,
  secrets,
  anthropicDraft,
  openaiDraft,
  setAnthropicDraft,
  setOpenaiDraft,
  saving,
  error,
  onSave,
  onRemove,
}: BYOKSectionProps) => {
  const anthropic = secrets?.providers.anthropic;
  const openai = secrets?.providers.openai;
  return (
    <div className="sheet-section">
      <h3>Provider keys (BYOK)</h3>
      <p className="desc">
        Per-agent API keys for <span className="mono">{agent.id}</span>. Files live in{' '}
        <span className="mono">agents/{agent.id}/secrets.env</span> (gitignored, mode 0600).
        When unset, the server falls back to the global <span className="mono">.env</span>.
      </p>

      <KeyRow
        label="Anthropic"
        status={anthropic}
        draft={anthropicDraft}
        setDraft={setAnthropicDraft}
        saving={saving === 'anthropic'}
        placeholder="sk-ant-api03-…"
        onSave={() => void onSave('anthropic')}
        onRemove={() => void onRemove('anthropic')}
      />

      <KeyRow
        label="OpenAI"
        status={openai}
        draft={openaiDraft}
        setDraft={setOpenaiDraft}
        saving={saving === 'openai'}
        placeholder="sk-proj-…"
        onSave={() => void onSave('openai')}
        onRemove={() => void onRemove('openai')}
      />

      {error && (
        <div className="dim" style={{ fontSize: 11.5, color: 'var(--bad-fg)', marginTop: 8 }}>
          {error}
        </div>
      )}
    </div>
  );
};

const KeyRow = ({
  label,
  status,
  draft,
  setDraft,
  saving,
  placeholder,
  onSave,
  onRemove,
}: {
  label: string;
  status: { set: boolean; source: string; mask: string | null; var: string } | undefined;
  draft: string;
  setDraft: (s: string) => void;
  saving: boolean;
  placeholder: string;
  onSave: () => void;
  onRemove: () => void;
}) => {
  const isPerAgent = status?.source === 'per-agent';
  const isGlobal = status?.source === 'global';
  return (
    <div className="byok-row">
      <div className="byok-row-head">
        <span className="byok-label">{label}</span>
        {status?.set ? (
          <Tag kind={isPerAgent ? 'success' : 'warn'}>
            <span className="dot" />
            {isPerAgent ? 'per-agent' : 'global .env'}
          </Tag>
        ) : (
          <Tag kind="bad">
            <span className="dot" />
            unset
          </Tag>
        )}
        {status?.mask && (
          <span className="mono dim" style={{ fontSize: 11.5, marginLeft: 'auto' }}>
            {status.mask}
          </span>
        )}
      </div>
      <div className="byok-row-input">
        <input
          type="password"
          autoComplete="off"
          placeholder={isPerAgent ? `Replace ${label} key` : placeholder}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          disabled={saving}
        />
        <button
          className="btn sm primary"
          onClick={onSave}
          disabled={saving || !draft.trim()}
        >
          {saving ? 'Saving…' : isPerAgent ? 'Rotate' : 'Save'}
        </button>
        {isPerAgent && (
          <button
            className="btn sm ghost"
            onClick={onRemove}
            disabled={saving}
            title={`Remove the per-agent key; ${label} falls back to global .env`}
          >
            Remove
          </button>
        )}
      </div>
      {isGlobal && (
        <div className="dim" style={{ fontSize: 11.5, marginTop: 6 }}>
          Inherited from the global <span className="mono">.env</span>. Save a key here to
          override per-agent.
        </div>
      )}
    </div>
  );
};

// ─── Self-improvement engineer section (P3.2) ──────────────────
const TRANSPORTS: Array<{ id: ImprovementConfig['transport']; name: string; meta: string }> = [
  { id: 'auto', name: 'Auto', meta: 'Picks claude CLI if installed, else API.' },
  { id: 'claude_code_cli', name: 'Claude Code CLI', meta: 'Runs `claude --print` locally. Has filesystem + MCP access.' },
  { id: 'anthropic_api', name: 'Anthropic API', meta: 'Direct SDK call. Sandboxed, no fs.' },
  { id: 'disabled', name: 'Disabled', meta: 'No autonomous improvement. The agent stays static.' },
];

const IMPROVEMENT_MODELS: Array<{ id: string; name: string }> = [
  { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5' },
  { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6' },
  { id: 'claude-opus-4-7', name: 'Claude Opus 4.7' },
];

const ImprovementSection = ({ agentId }: { agentId: string }) => {
  const [cfg, setCfg] = useState<ImprovementConfig | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void getAgentImprovement(agentId)
      .then((next) => {
        if (!cancelled) setCfg(next);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [agentId]);

  const patch = async (delta: Partial<ImprovementConfig>) => {
    if (!cfg || saving) return;
    const prev = cfg;
    const optimistic = { ...cfg, ...delta };
    setCfg(optimistic);
    setSaving(true);
    setError(null);
    try {
      const next = await putAgentImprovement(agentId, delta);
      setCfg(next);
    } catch (e) {
      setCfg(prev);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  if (!cfg) {
    return <div className="dim" style={{ fontSize: 12 }}>Loading…</div>;
  }

  return (
    <div className="improvement-stack">
      <div className="improvement-toggle">
        <label className="improvement-toggle-label">
          <input
            type="checkbox"
            checked={cfg.enabled && cfg.transport !== 'disabled'}
            onChange={(e) => void patch({ enabled: e.target.checked })}
            disabled={saving}
          />
          <span>
            <strong>Autonomous improvement</strong>
            <span className="dim" style={{ fontSize: 11.5, marginLeft: 8 }}>
              {cfg.enabled && cfg.transport !== 'disabled' ? 'ON' : 'OFF'}
            </span>
          </span>
        </label>
        <div className="dim" style={{ fontSize: 12, marginTop: 4 }}>
          When ON, the wakeup loop runs the proposer/critic on a cadence.
        </div>
      </div>

      <div className="improvement-field">
        <div className="improvement-field-label">Transport</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          {TRANSPORTS.map((t) => (
            <button
              key={t.id}
              className="row-item"
              style={{
                cursor: saving ? 'progress' : 'pointer',
                borderColor: cfg.transport === t.id ? 'var(--foreground)' : undefined,
                margin: 0,
                textAlign: 'left',
                opacity: saving ? 0.7 : 1,
              }}
              onClick={() => void patch({ transport: t.id })}
              disabled={saving}
            >
              <div className="rmain">
                <div className="rname">{t.name}</div>
                <div className="rmeta">{t.meta}</div>
              </div>
              {cfg.transport === t.id && <Icon name="check" size={14} />}
            </button>
          ))}
        </div>
      </div>

      <div className="improvement-field">
        <div className="improvement-field-label">Model</div>
        <select
          className="improvement-select"
          value={cfg.model}
          onChange={(e) => void patch({ model: e.target.value })}
          disabled={saving || cfg.transport === 'claude_code_cli'}
        >
          {IMPROVEMENT_MODELS.map((m) => (
            <option key={m.id} value={m.id}>{m.name}</option>
          ))}
        </select>
        {cfg.transport === 'claude_code_cli' && (
          <div className="dim" style={{ fontSize: 11.5, marginTop: 6 }}>
            Claude Code CLI uses whichever model your local <span className="mono">claude</span> is
            logged in with — this dropdown only matters for the Anthropic API transport.
          </div>
        )}
      </div>

      <div className="improvement-field">
        <div className="improvement-field-label">Cadence (minutes)</div>
        <input
          type="number"
          className="improvement-cadence"
          min={0}
          max={1440}
          step={5}
          value={cfg.cadence_minutes}
          onChange={(e) => void patch({ cadence_minutes: Number(e.target.value) })}
          disabled={saving}
        />
        <div className="dim" style={{ fontSize: 11.5, marginTop: 4 }}>
          How often the wakeup loop fires (when enabled). Zero disables the timer; the brain
          still runs on every Nth /run via the trace counter.
        </div>
      </div>

      {error && (
        <div className="dim" style={{ fontSize: 11.5, color: 'var(--bad-fg)' }}>
          {error}
        </div>
      )}
    </div>
  );
};

// ─── Channels tab (P3.5 — cards list + drill-in) ────────────────
// Top-level view is a card per channel; tapping "Configure" opens a
// full-panel drill-in (handled by AgentSheet's drill state). The status
// snapshot here drives the connected/not-configured pill and the
// inline meta row (number, endpoint, etc.).
interface ChannelsTabProps {
  agentId: string;
  onConfigureWidget: () => void;
  onConnectSlack: () => void;
  onConfigureWhatsApp: () => void;
  onConfigureApi: () => void;
}

const ChannelsTab = ({
  agentId,
  onConfigureWidget,
  onConnectSlack,
  onConfigureWhatsApp,
  onConfigureApi,
}: ChannelsTabProps) => {
  const [status, setStatus] = useState<AgentChannelsResponse | null>(null);

  const refresh = async () => {
    try {
      const next = await getAgentChannels(agentId);
      setStatus(next);
    } catch (e) {
      if (!(e instanceof ApiError)) console.warn('getAgentChannels failed', e);
    }
  };

  useEffect(() => {
    void refresh();
    // Refresh whenever the drill closes (operator may have just connected).
    const onFocus = () => void refresh();
    window.addEventListener('focus', onFocus);
    return () => window.removeEventListener('focus', onFocus);
  }, [agentId]);

  const wa = status?.channels.whatsapp;
  const web = status?.channels.web;
  const slack = status?.channels.slack;
  const api = status?.channels.api;

  const cards: ChannelCardSpec[] = [
    {
      id: 'whatsapp',
      name: 'WhatsApp',
      tileClass: 't-whatsapp',
      icon: <WhatsAppGlyph />,
      desc: 'Reply to customers from their WhatsApp Business inbox.',
      connected: !!wa?.connected,
      meta: wa?.connected
        ? {
            primary: ((wa.meta as { from_number?: string } | undefined)?.from_number ?? '').replace(
              /^whatsapp:/,
              '',
            ),
            vol: null,
          }
        : null,
      primaryAction: {
        label: wa?.connected ? 'Configure' : 'Connect WhatsApp',
        icon: wa?.connected ? 'settings' : 'link',
        primary: !wa?.connected,
        onClick: onConfigureWhatsApp,
      },
    },
    {
      id: 'widget',
      name: 'Web Widget',
      tileClass: 't-widget',
      icon: <WidgetGlyph />,
      desc: 'Embed this agent on your website with a floating chat widget.',
      connected: !!web?.connected,
      meta: web?.connected
        ? {
            primary: (web.meta as { widget_id?: string } | undefined)?.widget_id ?? '',
            vol: null,
          }
        : null,
      primaryAction: {
        label: web?.connected ? 'Configure Widget' : 'Add Widget',
        icon: web?.connected ? 'sliders' : 'link',
        primary: !web?.connected,
        onClick: onConfigureWidget,
      },
    },
    {
      id: 'slack',
      name: 'Slack',
      tileClass: 't-slack',
      icon: <SlackGlyph />,
      desc: 'Add the agent to your workspace as a bot — answer threads or @mentions.',
      connected: !!slack?.connected,
      meta: slack?.connected
        ? {
            primary: (slack.meta as { team_name?: string } | undefined)?.team_name ?? '',
            vol: null,
          }
        : null,
      primaryAction: {
        label: slack?.connected ? 'Configure' : 'Connect Slack',
        icon: slack?.connected ? 'settings' : 'link',
        primary: !slack?.connected,
        onClick: onConnectSlack,
      },
    },
    {
      id: 'api',
      name: 'REST API',
      tileClass: 't-api',
      icon: <ApiGlyph />,
      desc: 'Call the agent directly from your backend — same brain, programmable.',
      connected: !!api?.connected,
      meta: api?.connected
        ? {
            primary: `api/${agentId}/chat`,
            vol: null,
          }
        : null,
      primaryAction: {
        label: api?.connected ? 'Configure' : 'Connect',
        icon: api?.connected ? 'settings' : 'link',
        primary: !api?.connected,
        onClick: onConfigureApi,
      },
    },
  ];

  return (
    <div className="sheet-section">
      <h3>Where the agent talks</h3>
      <p className="desc">
        Each channel becomes a source of traces, feedback, and signal. The agent's brain, hands,
        and policies are shared across all of them.
      </p>
      <div className="channel-list">
        {cards.map((c) => (
          <div className={`channel-card ${c.connected ? 'connected' : ''}`} key={c.id}>
            <div className="channel-head">
              <div className={`channel-tile ${c.tileClass}`}>{c.icon}</div>
              <div className="channel-body">
                <div className="channel-title-row">
                  <span className="channel-name">{c.name}</span>
                  {c.connected ? (
                    <span className="channel-status connected">
                      <span className="dot" />
                      Connected
                    </span>
                  ) : (
                    <span className="channel-status notconfigured">Not configured</span>
                  )}
                </div>
                <div className="channel-desc">{c.desc}</div>
              </div>
            </div>

            {c.connected && c.meta && c.meta.primary && (
              <div className="channel-meta">
                <span className="mono">{c.meta.primary}</span>
              </div>
            )}

            <div className="channel-actions">
              <button
                className={`btn sm ${c.primaryAction.primary ? 'primary' : ''}`}
                onClick={c.primaryAction.onClick}
              >
                <Icon name={c.primaryAction.icon as IconName} size={12} />
                {c.primaryAction.label}
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

interface ChannelCardSpec {
  id: string;
  name: string;
  tileClass: string;
  icon: React.ReactNode;
  desc: string;
  connected: boolean;
  meta: { primary: string; vol: string | null } | null;
  primaryAction: { label: string; icon: string; primary?: boolean; onClick: () => void };
}
// ─── Drill-in shell (back nav + breadcrumb + status pill) ──────
interface DrillFrameProps {
  channelName: string;
  glyph: React.ReactNode;
  connected: boolean;
  onBack: () => void;
  topActions?: React.ReactNode;
  children: React.ReactNode;
}

const ChannelDrillFrame = ({
  channelName,
  glyph,
  connected,
  onBack,
  topActions,
  children,
}: DrillFrameProps) => (
  <div className="wcfg">
    <div className="wcfg-topbar">
      <button className="wcfg-back" onClick={onBack}>
        <Icon name="chevronLeft" size={14} /> Channels
      </button>
      <div className="wcfg-crumb">
        <span className="sep">/</span>
        <span className="now">
          {glyph}
          {channelName}
        </span>
        {connected ? (
          <span className="channel-status connected">
            <span className="dot" />
            Connected
          </span>
        ) : (
          <span className="channel-status notconfigured">Not configured</span>
        )}
      </div>
      {topActions && <div className="wcfg-top-actions">{topActions}</div>}
    </div>
    {children}
  </div>
);

// ─── Shared copy-button hook ────────────────────────────────────
function useCopy() {
  const [copied, setCopied] = useState<Record<string, boolean>>({});
  const copy = (key: string, text: string) => {
    void navigator.clipboard?.writeText(text);
    setCopied((c) => ({ ...c, [key]: true }));
    setTimeout(() => setCopied((c) => ({ ...c, [key]: false })), 1500);
  };
  return { copied, copy };
}

// ─── Endpoint card (POST <url> + signing secret reveal) ────────
interface EndpointCardProps {
  method?: string;
  url: string;
  secrets?: Array<{ label: string; value: string }>;
  reveal: boolean;
  onReveal: () => void;
  onRotate?: () => void;
  extraRows?: React.ReactNode;
}

const EndpointCard = ({
  method = 'POST',
  url,
  secrets = [],
  reveal,
  onReveal,
  onRotate,
  extraRows,
}: EndpointCardProps) => {
  const { copied, copy } = useCopy();
  return (
    <div className="endpoint-card">
      <div className="endpoint-row">
        <span className="endpoint-method">{method}</span>
        <span className="endpoint-url">{url}</span>
        <button
          className={`endpoint-copy ${copied.url ? 'ok' : ''}`}
          onClick={() => copy('url', url)}
        >
          <Icon name={copied.url ? 'check' : 'copy'} size={11} />
          {copied.url ? 'Copied' : 'Copy'}
        </button>
      </div>
      {secrets.map((s, i) => (
        <div className="endpoint-secret" key={s.label} style={i === secrets.length - 1 && !extraRows ? { borderBottom: 0 } : undefined}>
          <div className="k">{s.label}</div>
          <div className="v">
            {reveal && s.value
              ? s.value
              : '••••••••••••••••••••••••••••••••'}
          </div>
          <div className="endpoint-secret-actions">
            <button className="btn sm ghost" onClick={onReveal} title={reveal ? 'Hide' : 'Reveal'}>
              <Icon name="eye" size={12} />
            </button>
            <button className="btn sm ghost" onClick={() => copy(`sec-${i}`, s.value)}>
              <Icon name={copied[`sec-${i}`] ? 'check' : 'copy'} size={12} />
            </button>
            {onRotate && i === 0 && (
              <button className="btn sm ghost" onClick={onRotate} title="Rotate">
                <Icon name="refresh" size={12} />
              </button>
            )}
          </div>
        </div>
      ))}
      {extraRows}
    </div>
  );
};

// ─── Web Widget panel — preview + appearance + behavior + endpoint + domains ──
const WebWidgetPanel = ({ agentId, onBack }: { agentId: string; onBack: () => void }) => {
  const [status, setStatus] = useState<WebChannelStatus | null>(null);
  const [freshSecret, setFreshSecret] = useState<WebChannelConnectResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [showSecret, setShowSecret] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(true);
  const [newDomain, setNewDomain] = useState('');
  const [error, setError] = useState<string | null>(null);
  const { copied, copy } = useCopy();

  const refresh = async () => {
    try {
      setStatus(await getWebChannel(agentId));
    } catch (e) {
      if (!(e instanceof ApiError)) console.warn('getWebChannel failed', e);
    }
  };

  useEffect(() => {
    void refresh();
  }, [agentId]);

  const connect = async () => {
    setBusy(true);
    setError(null);
    try {
      const next = await connectWebChannel(agentId);
      setFreshSecret(next);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const rotate = async () => {
    if (!confirm('Rotate signing secret? Any backend verifying the old secret will start failing.')) return;
    setBusy(true);
    setError(null);
    try {
      const next = await rotateWebChannelSecret(agentId);
      setFreshSecret(next);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const disconnect = async () => {
    if (!confirm('Disconnect the Web Widget? The embed script on your site will stop responding.')) return;
    setBusy(true);
    setError(null);
    try {
      await disconnectWebChannel(agentId);
      setStatus({ connected: false });
      setFreshSecret(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const patch = async (
    patchBody: { allowed_domains?: string[]; settings?: Partial<WebWidgetSettings> },
  ) => {
    setBusy(true);
    setError(null);
    try {
      setStatus(await updateWebChannel(agentId, patchBody));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const addDomain = () => {
    const d = newDomain.trim();
    if (!d) return;
    const existing = status?.allowed_domains ?? [];
    if (existing.includes(d)) {
      setNewDomain('');
      return;
    }
    void patch({ allowed_domains: [...existing, d] });
    setNewDomain('');
  };
  const removeDomain = (d: string) => {
    const existing = status?.allowed_domains ?? [];
    void patch({ allowed_domains: existing.filter((x) => x !== d) });
  };

  const settings: WebWidgetSettings = status?.settings ?? {
    position: 'br',
    shape: 'circle',
    accent: 'green',
    greeting: '',
    welcome: '',
    fallback: '',
    show_greeting: true,
    require_email: false,
    pill_label: 'Chat',
  };
  const setS = (delta: Partial<WebWidgetSettings>) =>
    void patch({ settings: { ...settings, ...delta } });

  const isConnected = !!status?.connected;
  const widgetId = freshSecret?.widget_id ?? status?.widget_id ?? '';
  const messageUrl =
    freshSecret?.message_url ??
    status?.message_url ??
    (typeof window !== 'undefined' && widgetId
      ? `${window.location.origin}/widget/${widgetId}/message`
      : '');
  const embedUrl =
    freshSecret?.embed_url ??
    status?.embed_url ??
    (typeof window !== 'undefined' && widgetId
      ? `${window.location.origin}/widget/${widgetId}/v1.js`
      : '');
  const accentCss = ACCENT_CSS[settings.accent] ?? ACCENT_CSS.green;
  const embedSnippet = widgetId
    ? `<!-- OpenTracy Web Widget -->
<script>
  (function(w,d){
    w.OpenTracy = w.OpenTracy || { agent: "${widgetId}" };
    var s = d.createElement("script");
    s.src = "${embedUrl}";
    s.async = true;
    d.body.appendChild(s);
  })(window, document);
</script>`
    : '';

  if (!isConnected) {
    return (
      <ChannelDrillFrame
        channelName="Web Widget"
        glyph={<WidgetGlyph />}
        connected={false}
        onBack={onBack}
      >
        <div className="wcfg-config-pane" style={{ padding: '32px 24px' }}>
          <div className="connect-hero">
            <div className="connect-hero-tile"><WidgetGlyph /></div>
            <h2>Embed support-agent on your website</h2>
            <p>
              A floating chat widget that drops into your site with a single script tag. The
              agent's brain, tools, and policies are shared with all other channels.
            </p>
            <div className="perms">
              <div className="perms-label">You'll get</div>
              <ul>
                <li><Icon name="check" size={14} /><span>A copy-pasteable <span className="mono">&lt;script&gt;</span> snippet for your site</span></li>
                <li><Icon name="check" size={14} /><span>An HMAC signing secret you can verify in your backend</span></li>
                <li><Icon name="check" size={14} /><span>Origin-pinned <span className="mono">/widget/&lt;id&gt;/message</span> webhook</span></li>
                <li><Icon name="check" size={14} /><span>Live previews + appearance customization</span></li>
              </ul>
            </div>
            <button
              className="btn primary"
              style={{ height: 38, padding: '0 18px', fontSize: 13.5 }}
              onClick={() => void connect()}
              disabled={busy}
            >
              <Icon name="link" size={14} />
              {busy ? 'Adding…' : 'Add Web Widget'}
            </button>
            {error && (
              <div className="dim" style={{ fontSize: 12, color: 'var(--bad-fg)', marginTop: 10 }}>
                {error}
              </div>
            )}
          </div>
        </div>
      </ChannelDrillFrame>
    );
  }

  return (
    <ChannelDrillFrame
      channelName="Web Widget"
      glyph={<WidgetGlyph />}
      connected
      onBack={onBack}
    >
      <div className="wcfg-grid">
        {/* Preview pane */}
        <div className="wcfg-preview-pane" style={{ ['--widget-accent' as string]: accentCss } as React.CSSProperties}>
          <div className="wcfg-preview-head">
            <span className="wcfg-preview-label">Live preview</span>
            <div className="wcfg-preview-segment">
              <button className={previewOpen ? 'on' : ''} onClick={() => setPreviewOpen(true)}>Open</button>
              <button className={!previewOpen ? 'on' : ''} onClick={() => setPreviewOpen(false)}>Closed</button>
            </div>
          </div>

          <FakeBrowser>
            <WidgetPreview
              open={previewOpen}
              settings={settings}
              accentCss={accentCss}
              onToggle={() => setPreviewOpen((v) => !v)}
            />
          </FakeBrowser>

          <div style={{ fontSize: 11.5, color: 'var(--fg-subtle)', lineHeight: 1.5 }}>
            Preview reflects current settings. Visitors see the launcher first; tapping it opens the chat panel.
          </div>
        </div>

        {/* Config pane */}
        <div className="wcfg-config-pane">
          {freshSecret && (
            <div className="wcfg-section">
              <div className="wcfg-section-head">
                <h3>Signing secret</h3>
                <span className="desc">
                  Save this now — you won't see it again. Use it to verify outbound webhooks.
                </span>
              </div>
              <div className="api-channel-token-banner">
                <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 4 }}>
                  <Icon name="info" size={12} /> One-time view
                </div>
                <code className="api-channel-token">{freshSecret.signing_secret}</code>
              </div>
            </div>
          )}

          {/* Install */}
          <div className="wcfg-section">
            <div className="wcfg-section-head">
              <h3>Install</h3>
              <span className="desc">
                Paste this snippet before <span className="mono">&lt;/body&gt;</span> on every page you want the widget on.
              </span>
            </div>
            <div className="code-block">
              <button
                className={`copy-btn ${copied.embed ? 'ok' : ''}`}
                onClick={() => copy('embed', embedSnippet)}
              >
                <Icon name={copied.embed ? 'check' : 'copy'} size={11} />
                {copied.embed ? 'Copied' : 'Copy'}
              </button>
              {embedSnippet.split('\n').map((line, i) => <div key={i}>{line || ' '}</div>)}
            </div>
          </div>

          {/* Appearance */}
          <div className="wcfg-section">
            <div className="wcfg-section-head">
              <h3>Appearance</h3>
              <span className="desc">How the widget looks on the page.</span>
            </div>

            <div className="wcfg-row">
              <div className="k">Position</div>
              <div className="v">
                <div className="wcfg-segment">
                  <button className={settings.position === 'br' ? 'on' : ''} onClick={() => setS({ position: 'br' })}>↘ Bottom right</button>
                  <button className={settings.position === 'bl' ? 'on' : ''} onClick={() => setS({ position: 'bl' })}>↙ Bottom left</button>
                </div>
              </div>
            </div>

            <div className="wcfg-row">
              <div className="k">Accent color</div>
              <div className="v">
                <div className="wcfg-swatches">
                  {(Object.entries(ACCENT_CSS) as Array<[WebWidgetSettings['accent'], string]>).map(([k, v]) => (
                    <button
                      key={k}
                      className={`wcfg-swatch ${settings.accent === k ? 'on' : ''}`}
                      onClick={() => setS({ accent: k })}
                      title={k}
                    >
                      <span className="fill" style={{ background: v }} />
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="wcfg-row">
              <div className="k">Launcher shape</div>
              <div className="v">
                <div className="wcfg-shape">
                  <button className={settings.shape === 'circle' ? 'on' : ''} onClick={() => setS({ shape: 'circle' })}>
                    <span className="preview circle"><Icon name="chat" size={12} /></span>
                    Circle
                  </button>
                  <button className={settings.shape === 'rounded' ? 'on' : ''} onClick={() => setS({ shape: 'rounded' })}>
                    <span className="preview rounded"><Icon name="chat" size={12} /></span>
                    Rounded
                  </button>
                  <button className={settings.shape === 'pill' ? 'on' : ''} onClick={() => setS({ shape: 'pill' })}>
                    <span className="preview rounded" style={{ width: 36, borderRadius: 999 }}><Icon name="chat" size={10} /></span>
                    Pill + label
                  </button>
                </div>
              </div>
            </div>

            {settings.shape === 'pill' && (
              <div className="wcfg-row">
                <div className="k">Pill label</div>
                <div className="v">
                  <input
                    className="wcfg-input"
                    value={settings.pill_label}
                    onChange={(e) => setS({ pill_label: e.target.value })}
                    placeholder="Chat"
                  />
                </div>
              </div>
            )}

            <div className="wcfg-row tall">
              <div className="k">
                Greeting bubble
                <span className="k-sub">Peek over the launcher when a visitor arrives.</span>
              </div>
              <div className="v" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <input
                  className="wcfg-input"
                  value={settings.greeting}
                  onChange={(e) => setS({ greeting: e.target.value })}
                  disabled={!settings.show_greeting}
                  placeholder="Have a question? Ask away."
                />
                <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--fg-muted)' }}>
                  <button className={`switch ${settings.show_greeting ? 'on' : ''}`} onClick={() => setS({ show_greeting: !settings.show_greeting })} />
                  Show greeting bubble after 4s
                </label>
              </div>
            </div>
          </div>

          {/* Behavior */}
          <div className="wcfg-section">
            <div className="wcfg-section-head">
              <h3>Behavior</h3>
              <span className="desc">What the agent says and when it asks for help.</span>
            </div>
            <div className="wcfg-row tall">
              <div className="k">
                Welcome message
                <span className="k-sub">First message in the chat panel.</span>
              </div>
              <div className="v">
                <textarea
                  className="wcfg-textarea"
                  value={settings.welcome}
                  onChange={(e) => setS({ welcome: e.target.value })}
                  rows={2}
                />
              </div>
            </div>
            <div className="wcfg-row tall">
              <div className="k">
                Fallback
                <span className="k-sub">Used when the agent escalates to a human.</span>
              </div>
              <div className="v">
                <textarea
                  className="wcfg-textarea"
                  value={settings.fallback}
                  onChange={(e) => setS({ fallback: e.target.value })}
                  rows={2}
                />
              </div>
            </div>
            <div className="wcfg-row">
              <div className="k">Ask for email</div>
              <div className="v" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <button className={`switch ${settings.require_email ? 'on' : ''}`} onClick={() => setS({ require_email: !settings.require_email })} />
                <span style={{ fontSize: 12.5, color: 'var(--fg-muted)' }}>Before the visitor sends their first message</span>
              </div>
            </div>
          </div>

          {/* Public endpoint */}
          <div className="wcfg-section">
            <div className="wcfg-section-head">
              <h3>Public endpoint</h3>
              <span className="desc">
                Where messages from the embedded widget arrive. Origin-gated; signed with HMAC-SHA256 if you proxy through your backend.
              </span>
            </div>
            <EndpointCard
              url={messageUrl}
              secrets={[{ label: 'Signing secret', value: freshSecret?.signing_secret ?? '' }]}
              reveal={showSecret && !!freshSecret}
              onReveal={() => setShowSecret((s) => !s)}
              onRotate={() => void rotate()}
            />
          </div>

          {/* Allowed domains */}
          <div className="wcfg-section">
            <div className="wcfg-section-head">
              <h3>Allowed domains</h3>
              <span className="desc">The widget only accepts messages from origins listed here. Leave empty to allow localhost for testing.</span>
            </div>
            <div className="domain-list">
              {(status?.allowed_domains ?? []).map((d) => (
                <span key={d} className="domain-chip">
                  {d}
                  <button className="remove" onClick={() => removeDomain(d)} disabled={busy} title="Remove">
                    <Icon name="x" size={10} />
                  </button>
                </span>
              ))}
              <input
                className="domain-input"
                placeholder="add a domain…"
                value={newDomain}
                onChange={(e) => setNewDomain(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') addDomain(); }}
                disabled={busy}
              />
            </div>
          </div>

          {/* Danger zone */}
          <div className="wcfg-section">
            <div className="wcfg-section-head"><h3>Danger zone</h3></div>
            <div className="danger-zone">
              <div>
                <div className="dz-title">Disconnect Web Widget</div>
                <div className="dz-meta">The embed script will stop responding. Existing traces stay in the log.</div>
              </div>
              <button className="btn sm danger" onClick={() => void disconnect()} disabled={busy}>
                Disconnect
              </button>
            </div>
          </div>

          {error && (
            <div className="dim" style={{ fontSize: 11.5, color: 'var(--bad-fg)' }}>
              {error}
            </div>
          )}
        </div>
      </div>
    </ChannelDrillFrame>
  );
};

const ACCENT_CSS: Record<WebWidgetSettings['accent'], string> = {
  green: 'oklch(0.55 0.14 150)',
  blue: 'oklch(0.55 0.14 250)',
  plum: 'oklch(0.55 0.14 320)',
  slate: 'oklch(0.30 0.005 80)',
  brand: 'oklch(0.55 0.14 35)',
};

// ── Faux browser frame for widget preview ────────────────────
const FakeBrowser = ({ children }: { children: React.ReactNode }) => (
  <div className="fakebrowser">
    <div className="fb-chrome">
      <div className="fb-dots"><i /><i /><i /></div>
      <div className="fb-url">
        <Icon name="shield" size={10} />
        acmestore.com/order/3318
      </div>
    </div>
    <div className="fb-body">
      <div className="fb-nav">
        <div className="fb-logo">
          <span className="fb-logo-mark" />
          Acme Store
        </div>
        <div className="fb-nav-links">
          <span>Shop</span>
          <span>Orders</span>
          <span>Help</span>
        </div>
        <span className="fb-nav-buy">Cart · 2</span>
      </div>
      <div className="fb-hero">
        <h1>Your order is on its way.</h1>
        <p>Track delivery, request changes, or chat with us — we usually reply in under a minute.</p>
        <div className="fb-pillrow">
          <span className="tag info">In transit · ETA Wed</span>
          <span className="tag">ORD-3318-A</span>
        </div>
      </div>
      <div className="fb-skeleton">
        <div className="fb-card" />
        <div className="fb-card" />
        <div className="fb-card" />
      </div>
      {children}
    </div>
  </div>
);

// ── Floating widget rendered inside the preview ──────────────
const WidgetPreview = ({
  open,
  settings,
  accentCss,
  onToggle,
}: {
  open: boolean;
  settings: WebWidgetSettings;
  accentCss: string;
  onToggle: () => void;
}) => (
  <div className={`widgetprev pos-${settings.position}`}>
    {open ? (
      <div className="widgetprev-panel">
        <div className="widgetprev-panel-head">
          <div className="av"><span className="mark" /></div>
          <div style={{ minWidth: 0 }}>
            <div className="title">support-agent</div>
            <div className="sub"><span className="dot" />Replies instantly</div>
          </div>
          <button className="close" onClick={onToggle} aria-label="Close"><Icon name="x" size={14} /></button>
        </div>
        <div className="widgetprev-thread">
          {settings.welcome && (
            <div className="wp-msg agent"><div className="bubble">{settings.welcome}</div></div>
          )}
          <div className="wp-msg user"><div className="bubble">Where's my order ORD-3318-A?</div></div>
          <div className="wp-msg agent"><div className="bubble">Looking that up… it shipped Monday and is out for delivery today. Want me to text you when it arrives?</div></div>
        </div>
        <div className="widgetprev-input">
          <div className="field">Type a message…</div>
          <button className="send" aria-label="Send" style={{ background: accentCss }}><Icon name="send" size={12} /></button>
        </div>
        <div className="widgetprev-foot">Powered by OpenTracy</div>
      </div>
    ) : (
      <div className="widgetprev-launcher-stack">
        {settings.show_greeting && settings.greeting && (
          <div className="widgetprev-greeting">
            <div className="agent-line"><span className="dot" />support-agent</div>
            {settings.greeting}
          </div>
        )}
        <button
          className={`widgetprev-launcher shape-${settings.shape}`}
          onClick={onToggle}
          aria-label="Open chat"
          style={{ background: accentCss }}
        >
          <Icon name="chat" size={settings.shape === 'pill' ? 16 : 22} />
          {settings.shape === 'pill' && <span>{settings.pill_label || 'Chat'}</span>}
        </button>
      </div>
    )}
  </div>
);

// ─── REST API panel — code playground + endpoint card + routes ──
const ApiChannelPanel = ({ agentId, onBack }: { agentId: string; onBack: () => void }) => {
  const [details, setDetails] = useState<ApiChannelStatus | null>(null);
  const [freshToken, setFreshToken] = useState<ApiChannelConnectResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lang, setLang] = useState<'curl' | 'js' | 'python'>('curl');
  const { copied, copy } = useCopy();

  useEffect(() => {
    let cancelled = false;
    void getApiChannel(agentId)
      .then((next) => {
        if (!cancelled) setDetails(next);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [agentId]);

  const connect = async () => {
    setBusy(true);
    setError(null);
    try {
      const next = await connectApiChannel(agentId);
      setFreshToken(next);
      setDetails({ connected: true, token_mask: next.token_mask, created_at: next.created_at });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const rotate = async () => {
    if (!confirm('Rotate the API token? The current token stops working immediately.')) return;
    setBusy(true);
    setError(null);
    try {
      const next = await rotateApiChannel(agentId);
      setFreshToken(next);
      setDetails({ connected: true, token_mask: next.token_mask, created_at: next.created_at });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const disconnect = async () => {
    if (!confirm('Disconnect the API channel? The current token will stop working.')) return;
    setBusy(true);
    setError(null);
    try {
      await disconnectApiChannel(agentId);
      setDetails({ connected: false });
      setFreshToken(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const isConnected = !!details?.connected;
  const tokenForSample = freshToken?.token ?? '$OPENTRACY_KEY';
  const endpoint =
    freshToken?.public_url ??
    (typeof window !== 'undefined' ? `${window.location.origin}/api/${agentId}/chat` : '');

  const samples: Record<typeof lang, string> = {
    curl: `curl -X POST ${endpoint} \\
  -H "Authorization: Bearer ${tokenForSample}" \\
  -H "Content-Type: application/json" \\
  -d '{
    "request": "Where is order ORD-3318-A?"
  }'`,
    js: `// npm i opentracy
const res = await fetch("${endpoint}", {
  method: "POST",
  headers: {
    "Authorization": "Bearer ${tokenForSample}",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ request: "Where is order ORD-3318-A?" }),
});
const data = await res.json();
console.log(data.response);`,
    python: `# pip install requests
import os, requests
res = requests.post(
    "${endpoint}",
    headers={"Authorization": f"Bearer {os.environ['OPENTRACY_KEY']}"},
    json={"request": "Where is order ORD-3318-A?"},
)
print(res.json()["response"])`,
  };

  const sampleResponse = `{
  "trace_id": "trc_8af2…",
  "response": "Order ORD-3318-A shipped Monday, out for delivery today, ETA 6pm.",
  "success": true,
  "duration_ms": 1380
}`;

  if (!isConnected) {
    return (
      <ChannelDrillFrame
        channelName="REST API"
        glyph={<ApiGlyph />}
        connected={false}
        onBack={onBack}
      >
        <div className="wcfg-config-pane" style={{ padding: '32px 24px' }}>
          <div className="connect-hero">
            <div className="connect-hero-tile"><ApiGlyph /></div>
            <h2>Call support-agent from your backend</h2>
            <p>
              A single POST endpoint with bearer-token auth. Every call routes through the same
              brain, tools, and policies as Slack, WhatsApp, and the Web Widget.
            </p>
            <div className="perms">
              <div className="perms-label">You'll get</div>
              <ul>
                <li><Icon name="check" size={14} /><span>One bearer token to call <span className="mono">POST /api/{agentId}/chat</span></span></li>
                <li><Icon name="check" size={14} /><span>cURL / Node / Python quickstart snippets</span></li>
                <li><Icon name="check" size={14} /><span>Every request becomes a trace in Evolution</span></li>
              </ul>
            </div>
            <button
              className="btn primary"
              style={{ height: 38, padding: '0 18px', fontSize: 13.5 }}
              onClick={() => void connect()}
              disabled={busy}
            >
              {busy ? 'Connecting…' : 'Connect REST API'}
            </button>
            {error && (
              <div className="dim" style={{ fontSize: 12, color: 'var(--bad-fg)', marginTop: 10 }}>{error}</div>
            )}
          </div>
        </div>
      </ChannelDrillFrame>
    );
  }

  return (
    <ChannelDrillFrame
      channelName="REST API"
      glyph={<ApiGlyph />}
      connected
      onBack={onBack}
    >
      <div className="wcfg-grid">
        {/* Preview pane — code playground */}
        <div className="wcfg-preview-pane">
          <div className="wcfg-preview-head">
            <span className="wcfg-preview-label">Quickstart</span>
            <div className="wcfg-preview-segment" style={{ marginLeft: 'auto' }}>
              <button className={lang === 'curl' ? 'on' : ''} onClick={() => setLang('curl')}>cURL</button>
              <button className={lang === 'js' ? 'on' : ''} onClick={() => setLang('js')}>Node</button>
              <button className={lang === 'python' ? 'on' : ''} onClick={() => setLang('python')}>Python</button>
            </div>
          </div>
          <div className="code-block" style={{ minHeight: 200, fontSize: 12, lineHeight: 1.7 }}>
            <button
              className={`copy-btn ${copied.sample ? 'ok' : ''}`}
              onClick={() => copy('sample', samples[lang])}
            >
              <Icon name={copied.sample ? 'check' : 'copy'} size={11} />
              {copied.sample ? 'Copied' : 'Copy'}
            </button>
            {samples[lang].split('\n').map((line, i) => <div key={i}>{line || ' '}</div>)}
          </div>
          <div style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: 0.08, color: 'var(--fg-subtle)', fontWeight: 600, marginTop: 6 }}>Response</div>
          <div className="code-block" style={{ fontSize: 11.5 }}>
            {sampleResponse.split('\n').map((line, i) => <div key={i}>{line || ' '}</div>)}
          </div>
          <div style={{ fontSize: 11.5, color: 'var(--fg-subtle)', lineHeight: 1.5 }}>
            Same agent. Every call routes through the same brain, tools, and policies as Slack, WhatsApp, and the Web Widget.
          </div>
        </div>

        {/* Config pane */}
        <div className="wcfg-config-pane">
          {freshToken && (
            <div className="wcfg-section">
              <div className="wcfg-section-head">
                <h3>Token</h3>
                <span className="desc">Save this now — you won't see it again.</span>
              </div>
              <div className="api-channel-token-banner">
                <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 4 }}>
                  <Icon name="info" size={12} /> One-time view
                </div>
                <code className="api-channel-token">{freshToken.token}</code>
              </div>
            </div>
          )}

          <div className="wcfg-section">
            <div className="wcfg-section-head">
              <h3>Endpoint</h3>
              <span className="desc">All calls go through this one URL.</span>
            </div>
            <div className="endpoint-card">
              <div className="endpoint-row">
                <span className="endpoint-method">POST</span>
                <span className="endpoint-url">{endpoint}</span>
                <button
                  className={`endpoint-copy ${copied.url ? 'ok' : ''}`}
                  onClick={() => copy('url', endpoint)}
                >
                  <Icon name={copied.url ? 'check' : 'copy'} size={11} />
                  {copied.url ? 'Copied' : 'Copy'}
                </button>
              </div>
              <div className="endpoint-secret" style={{ borderBottom: 0 }}>
                <div className="k">Active token</div>
                <div className="v mono" style={{ fontSize: 12 }}>
                  {details?.token_mask ?? '—'}
                  {details?.last_used_at && (
                    <span style={{ color: 'var(--fg-muted)', marginLeft: 8 }}>
                      · last used {new Date(details.last_used_at).toLocaleString()}
                    </span>
                  )}
                </div>
                <div className="endpoint-secret-actions">
                  <button className="btn sm ghost" onClick={() => void rotate()} disabled={busy} title="Rotate">
                    <Icon name="refresh" size={12} />
                  </button>
                </div>
              </div>
            </div>
          </div>

          <div className="wcfg-section">
            <div className="wcfg-section-head"><h3>Danger zone</h3></div>
            <div className="danger-zone">
              <div>
                <div className="dz-title">Disconnect REST API</div>
                <div className="dz-meta">The current token stops working immediately. Reconnect any time to mint a new one.</div>
              </div>
              <button className="btn sm danger" onClick={() => void disconnect()} disabled={busy}>
                Disconnect
              </button>
            </div>
          </div>

          {error && (
            <div className="dim" style={{ fontSize: 11.5, color: 'var(--bad-fg)' }}>{error}</div>
          )}
        </div>
      </div>
    </ChannelDrillFrame>
  );
};

// ─── Slack panel — hero / connecting / configured with preview ──
const SlackChannelPanel = ({ agentId, onBack }: { agentId: string; onBack: () => void }) => {
  const [status, setStatus] = useState<SlackChannelStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSecret, setShowSecret] = useState(false);
  const { copied, copy } = useCopy();

  const refresh = async () => {
    try {
      setStatus(await getSlackChannel(agentId));
    } catch (e) {
      if (!(e instanceof ApiError)) console.warn('getSlackChannel failed', e);
    }
  };

  useEffect(() => {
    void refresh();
  }, [agentId]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get('slack_connected') === agentId) {
      void refresh();
      params.delete('slack_connected');
      const next = params.toString();
      window.history.replaceState(
        {},
        '',
        next ? `${window.location.pathname}?${next}` : window.location.pathname,
      );
    }
  }, [agentId]);

  const connect = () => {
    if (!status?.install_url) return;
    window.location.href = status.install_url;
  };

  const disconnect = async () => {
    if (!confirm('Disconnect Slack? The agent will stop receiving messages from this workspace.')) return;
    setBusy(true);
    setError(null);
    try {
      await disconnectSlackChannel(agentId);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!status) {
    return (
      <ChannelDrillFrame channelName="Slack" glyph={<SlackGlyphColored />} connected={false} onBack={onBack}>
        <div className="wcfg-config-pane">
          <div className="dim" style={{ fontSize: 12 }}>Loading…</div>
        </div>
      </ChannelDrillFrame>
    );
  }

  // Pre-connect hero (covers both: backend not configured + ready to connect)
  if (!status.connected) {
    return (
      <ChannelDrillFrame channelName="Slack" glyph={<SlackGlyphColored />} connected={false} onBack={onBack}>
        <div className="wcfg-config-pane" style={{ padding: '32px 24px' }}>
          <div className="connect-hero">
            <div className="connect-hero-tile"><SlackGlyphColored size={32} /></div>
            <h2>Add support-agent to your Slack workspace</h2>
            <p>
              The agent will respond to messages in channels you choose and DMs. Same brain, same
              policies — Slack is just another conversation surface.
            </p>
            <div className="perms">
              <div className="perms-label">It will be allowed to</div>
              <ul>
                <li><Icon name="check" size={14} /><span>Post messages and reply in threads as the agent's bot user</span></li>
                <li><Icon name="check" size={14} /><span>Read messages in channels it's invited to (not other channels)</span></li>
                <li><Icon name="check" size={14} /><span>Respond to direct messages and @-mentions</span></li>
              </ul>
            </div>
            {status.configured ? (
              <button
                className="btn primary"
                style={{ height: 38, padding: '0 18px', fontSize: 13.5 }}
                onClick={connect}
                disabled={busy}
              >
                <SlackGlyphColored size={14} />
                Add to Slack
              </button>
            ) : (
              <>
                <div className="dim" style={{ fontSize: 12, marginTop: 4, lineHeight: 1.5 }}>
                  Operator must register a Slack app and set <span className="mono">SLACK_CLIENT_ID</span>,{' '}
                  <span className="mono">SLACK_CLIENT_SECRET</span>, <span className="mono">SLACK_SIGNING_SECRET</span>,{' '}
                  and <span className="mono">PUBLIC_BASE_URL</span> on the backend before this button works.
                </div>
                <button className="btn primary" style={{ height: 38, padding: '0 18px', fontSize: 13.5, opacity: 0.5 }} disabled>
                  <SlackGlyphColored size={14} /> Add to Slack
                </button>
              </>
            )}
            {error && <div className="dim" style={{ fontSize: 12, color: 'var(--bad-fg)', marginTop: 10 }}>{error}</div>}
          </div>
        </div>
      </ChannelDrillFrame>
    );
  }

  // Configured view with Slack-style preview
  return (
    <ChannelDrillFrame
      channelName="Slack"
      glyph={<SlackGlyphColored />}
      connected
      onBack={onBack}
    >
      <div className="wcfg-grid">
        {/* Preview pane */}
        <div className="wcfg-preview-pane">
          <div className="wcfg-preview-head">
            <span className="wcfg-preview-label">Live preview</span>
            <span style={{ fontSize: 11, color: 'var(--fg-subtle)', marginLeft: 'auto' }}>
              {status.team_name} · #support
            </span>
          </div>
          <SlackPreview />
          <div style={{ fontSize: 11.5, color: 'var(--fg-subtle)', lineHeight: 1.5 }}>
            How the agent appears in Slack when @-mentioned.
          </div>
        </div>

        {/* Config pane */}
        <div className="wcfg-config-pane">
          <div className="wcfg-section">
            <div className="wcfg-section-head"><h3>Workspace</h3></div>
            <div className="row-item on" style={{ marginBottom: 0 }}>
              <div className="ricon" style={{ background: 'oklch(0.94 0.05 320)', color: 'oklch(0.42 0.13 320)' }}>
                <SlackGlyphColored size={16} />
              </div>
              <div className="rmain">
                <div className="rname">{status.team_name}</div>
                <div className="rmeta">
                  <span className="mono">{status.team_id}</span>
                  {status.installed_at && <> · installed {new Date(status.installed_at).toLocaleDateString()}</>}
                </div>
              </div>
              <Tag kind="success"><span className="dot" />Active</Tag>
            </div>
          </div>

          {status.events_url && (
            <div className="wcfg-section">
              <div className="wcfg-section-head">
                <h3>Events endpoint</h3>
                <span className="desc">Slack posts events here. Already wired during install — you only need this if you re-add the app manually.</span>
              </div>
              <div className="endpoint-card">
                <div className="endpoint-row">
                  <span className="endpoint-method">POST</span>
                  <span className="endpoint-url">{status.events_url}</span>
                  <button
                    className={`endpoint-copy ${copied.url ? 'ok' : ''}`}
                    onClick={() => copy('url', status.events_url ?? '')}
                  >
                    <Icon name={copied.url ? 'check' : 'copy'} size={11} />
                    {copied.url ? 'Copied' : 'Copy'}
                  </button>
                </div>
                <div className="endpoint-secret" style={{ borderBottom: 0 }}>
                  <div className="k">Subscribed to</div>
                  <div className="v" style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
                    <span className="tag">app_mention</span>
                    <span className="tag">message.im</span>
                  </div>
                  <div />
                </div>
              </div>
            </div>
          )}

          <div className="wcfg-section">
            <div className="wcfg-section-head"><h3>Danger zone</h3></div>
            <div className="danger-zone">
              <div>
                <div className="dz-title">Disconnect Slack</div>
                <div className="dz-meta">
                  Removes <span className="mono">support-agent</span> from the workspace and stops listening for events.
                </div>
              </div>
              <button className="btn sm danger" onClick={() => void disconnect()} disabled={busy}>
                {busy ? 'Disconnecting…' : 'Disconnect'}
              </button>
            </div>
          </div>

          {error && <div className="dim" style={{ fontSize: 11.5, color: 'var(--bad-fg)' }}>{error}</div>}

          {/* Reveal toggle is invisible-but-needed for the EndpointCard prop; keep it bound */}
          <button style={{ display: 'none' }} onClick={() => setShowSecret((v) => !v)}>{showSecret ? '_' : '_'}</button>
        </div>
      </div>
    </ChannelDrillFrame>
  );
};

// ─── Slack-style chat preview (static mock) ─────────────────────
const SlackPreview = () => (
  <div className="slackprev">
    <div className="slk-rail">
      <div className="slk-ws"><span className="mark" /> Workspace</div>
      <div className="slk-section-label">Channels</div>
      <div className="slk-channel on"><span className="hash">#</span>support</div>
      <div className="slk-channel"><span className="hash">#</span>customer-questions</div>
      <div className="slk-channel"><span className="hash">#</span>help-desk</div>
      <div className="slk-channel"><span className="hash">#</span>general</div>
      <div className="slk-section-label">Direct messages</div>
      <div className="slk-channel dm"><span className="av bot" />support-agent</div>
    </div>
    <div className="slk-main">
      <div className="slk-head">
        <span className="hash">#</span> support
        <span style={{ marginLeft: 8, fontWeight: 400, fontSize: 12, color: 'var(--fg-muted)' }}>· customer issue triage</span>
      </div>
      <div className="slk-thread">
        <div className="slk-msg">
          <div className="av" />
          <div>
            <div className="who">Maya R. <span className="ts">2:38 PM</span></div>
            <div className="body">
              <span className="mention">@support-agent</span> can you check on ORD-3318-A? Customer says it hasn't arrived.
            </div>
          </div>
        </div>
        <div className="slk-msg bot">
          <div className="av" />
          <div>
            <div className="who">support-agent <span className="botbadge">APP</span><span className="ts">2:38 PM</span></div>
            <div className="body">
              Found it. Order <span className="mono" style={{ fontSize: 11.5 }}>ORD-3318-A</span> shipped Monday, currently out for delivery. ETA today by 6pm.
            </div>
          </div>
        </div>
      </div>
      <div className="slk-compose">Message #support</div>
    </div>
  </div>
);

// ─── WhatsApp panel — preview + business profile + behavior ─────
const WhatsAppChannelPanel = ({ agentId, onBack }: { agentId: string; onBack: () => void }) => {
  const [status, setStatus] = useState<WhatsAppChannelStatus | null>(null);
  const [accountSid, setAccountSid] = useState('');
  const [authToken, setAuthToken] = useState('');
  const [fromNumber, setFromNumber] = useState('');
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { copied, copy } = useCopy();

  const refresh = async () => {
    try {
      setStatus(await getWhatsAppChannel(agentId));
    } catch (e) {
      if (!(e instanceof ApiError)) console.warn('getWhatsAppChannel failed', e);
    }
  };

  useEffect(() => {
    void refresh();
  }, [agentId]);

  const save = async () => {
    if (!accountSid.trim() || !authToken.trim() || !fromNumber.trim()) {
      setError('All three fields are required.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await connectWhatsAppChannel(agentId, {
        account_sid: accountSid.trim(),
        auth_token: authToken.trim(),
        from_number: fromNumber.trim(),
      });
      setAccountSid('');
      setAuthToken('');
      setFromNumber('');
      setEditing(false);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const disconnect = async () => {
    if (!confirm('Disconnect WhatsApp? The agent will stop receiving messages on this number.')) return;
    setBusy(true);
    setError(null);
    try {
      await disconnectWhatsAppChannel(agentId);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!status) {
    return (
      <ChannelDrillFrame channelName="WhatsApp" glyph={<WhatsAppGlyphColored />} connected={false} onBack={onBack}>
        <div className="wcfg-config-pane">
          <div className="dim" style={{ fontSize: 12 }}>Loading…</div>
        </div>
      </ChannelDrillFrame>
    );
  }

  // Pre-connect: BYOK form for Twilio creds (no global env-var setup needed).
  if (!status.connected || editing) {
    const showForm = !status.connected || editing;
    return (
      <ChannelDrillFrame
        channelName="WhatsApp"
        glyph={<WhatsAppGlyphColored />}
        connected={!!status.connected}
        onBack={onBack}
      >
        <div className="wcfg-config-pane" style={{ padding: '32px 24px' }}>
          <div className="connect-hero">
            <div className="connect-hero-tile"><WhatsAppGlyphColored size={32} /></div>
            <h2>{status.connected ? 'Rotate Twilio credentials' : 'Connect a WhatsApp number'}</h2>
            <p>
              Bring your own Twilio number. The Twilio Sandbox is free and works in minutes; live
              numbers run around $1/mo. Paste the creds below — they live in{' '}
              <span className="mono">agents/{agentId}/integrations/whatsapp.json</span> (mode 0600).
            </p>
            {showForm && (
              <div className="whatsapp-form" style={{ width: '100%', maxWidth: 480 }}>
                <label className="whatsapp-field">
                  <span>Account SID</span>
                  <input
                    className="whatsapp-input mono"
                    autoComplete="off"
                    placeholder="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                    value={accountSid}
                    onChange={(e) => setAccountSid(e.target.value)}
                    disabled={busy}
                  />
                </label>
                <label className="whatsapp-field">
                  <span>Auth Token</span>
                  <input
                    className="whatsapp-input mono"
                    type="password"
                    autoComplete="off"
                    placeholder="paste your Twilio Auth Token"
                    value={authToken}
                    onChange={(e) => setAuthToken(e.target.value)}
                    disabled={busy}
                  />
                </label>
                <label className="whatsapp-field">
                  <span>From Number</span>
                  <input
                    className="whatsapp-input mono"
                    autoComplete="off"
                    placeholder="whatsapp:+14155238886"
                    value={fromNumber}
                    onChange={(e) => setFromNumber(e.target.value)}
                    disabled={busy}
                  />
                </label>
                <div style={{ display: 'flex', gap: 8, marginTop: 12, justifyContent: 'center' }}>
                  <button className="btn primary" style={{ height: 36, padding: '0 16px' }} onClick={() => void save()} disabled={busy}>
                    {busy ? 'Saving…' : status.connected ? 'Rotate' : 'Connect WhatsApp'}
                  </button>
                  {editing && (
                    <button
                      className="btn ghost"
                      style={{ height: 36, padding: '0 16px' }}
                      onClick={() => {
                        setEditing(false);
                        setAccountSid('');
                        setAuthToken('');
                        setFromNumber('');
                      }}
                      disabled={busy}
                    >
                      Cancel
                    </button>
                  )}
                </div>
              </div>
            )}
            {!status.configured && (
              <div className="dim" style={{ fontSize: 12, marginTop: 4 }}>
                Set <span className="mono">PUBLIC_BASE_URL</span> on the backend so Twilio can reach
                the inbound webhook before connecting.
              </div>
            )}
            {error && <div className="dim" style={{ fontSize: 12, color: 'var(--bad-fg)', marginTop: 10 }}>{error}</div>}
          </div>
        </div>
      </ChannelDrillFrame>
    );
  }

  // Configured view with WhatsApp-style preview
  return (
    <ChannelDrillFrame
      channelName="WhatsApp"
      glyph={<WhatsAppGlyphColored />}
      connected
      onBack={onBack}
    >
      <div className="wcfg-grid">
        {/* Preview pane */}
        <div className="wcfg-preview-pane">
          <div className="wcfg-preview-head">
            <span className="wcfg-preview-label">Live preview</span>
            <span style={{ fontSize: 11, color: 'var(--fg-subtle)', marginLeft: 'auto' }}>
              {(status.from_number ?? '').replace(/^whatsapp:/, '')}
            </span>
          </div>
          <WhatsAppPreview />
          <div style={{ fontSize: 11.5, color: 'var(--fg-subtle)', lineHeight: 1.5 }}>
            How a customer sees the agent on their WhatsApp app.
          </div>
        </div>

        {/* Config pane */}
        <div className="wcfg-config-pane">
          <div className="wcfg-section">
            <div className="wcfg-section-head"><h3>Business profile</h3></div>
            <div className="row-item on" style={{ marginBottom: 0 }}>
              <div className="ricon" style={{ background: 'oklch(0.94 0.06 150)', color: 'oklch(0.42 0.13 150)' }}>
                <WhatsAppGlyphColored size={18} />
              </div>
              <div className="rmain">
                <div className="rname mono" style={{ fontSize: 13 }}>
                  {(status.from_number ?? '').replace(/^whatsapp:/, '')}
                </div>
                <div className="rmeta">
                  Twilio · SID {status.account_sid_mask}
                  {status.installed_at && <> · installed {new Date(status.installed_at).toLocaleDateString()}</>}
                </div>
              </div>
              <Tag kind="success"><span className="dot" />Active</Tag>
            </div>
          </div>

          {status.inbound_url && (
            <div className="wcfg-section">
              <div className="wcfg-section-head">
                <h3>Inbound webhook</h3>
                <span className="desc">Configure this in Twilio → Messaging → Sender → Webhooks.</span>
              </div>
              <div className="endpoint-card">
                <div className="endpoint-row">
                  <span className="endpoint-method">POST</span>
                  <span className="endpoint-url">{status.inbound_url}</span>
                  <button
                    className={`endpoint-copy ${copied.url ? 'ok' : ''}`}
                    onClick={() => copy('url', status.inbound_url ?? '')}
                  >
                    <Icon name={copied.url ? 'check' : 'copy'} size={11} />
                    {copied.url ? 'Copied' : 'Copy'}
                  </button>
                </div>
              </div>
            </div>
          )}

          <div className="wcfg-section">
            <div className="wcfg-section-head"><h3>Manage</h3></div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn sm" onClick={() => setEditing(true)} disabled={busy}>
                <Icon name="refresh" size={12} /> Rotate credentials
              </button>
            </div>
          </div>

          <div className="wcfg-section">
            <div className="wcfg-section-head"><h3>Danger zone</h3></div>
            <div className="danger-zone">
              <div>
                <div className="dz-title">Disconnect WhatsApp</div>
                <div className="dz-meta">The number stays on Twilio — we just stop forwarding messages to the agent.</div>
              </div>
              <button className="btn sm danger" onClick={() => void disconnect()} disabled={busy}>
                Disconnect
              </button>
            </div>
          </div>

          {error && <div className="dim" style={{ fontSize: 11.5, color: 'var(--bad-fg)' }}>{error}</div>}
        </div>
      </div>
    </ChannelDrillFrame>
  );
};

// ─── WhatsApp-style chat preview (static mock) ──────────────────
const WhatsAppPreview = () => (
  <div className="waprev">
    <div className="wa-head">
      <div className="av">A</div>
      <div className="meta">
        <div className="name">support-agent</div>
        <div className="pres">online</div>
      </div>
      <div className="icons">
        <Icon name="phone" size={16} />
        <Icon name="search" size={16} />
      </div>
    </div>
    <div className="wa-body">
      <span className="wa-day">TODAY</span>
      <div className="wa-bubble out">
        Hey, where's my order ORD-3318-A?
        <span className="time">2:38 PM <span className="tick">✓✓</span></span>
      </div>
      <div className="wa-bubble in">
        Hi! It shipped Monday and is out for delivery today, ETA 6pm. Want me to text you when it arrives?
        <span className="time">2:38 PM</span>
      </div>
    </div>
    <div className="wa-input">
      <div className="field">Type a message…</div>
      <button className="send"><Icon name="send" size={14} /></button>
    </div>
  </div>
);
// ─── Channel glyphs (monochrome SVG, currentColor) ──────────────
const WhatsAppGlyph = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3.5 20.5l1.4-4.6A8 8 0 1 1 8.4 19.4l-4.9 1.1z" />
    <path d="M8.6 9.8c.2 1.2 1 2.6 2.2 3.7 1.2 1.1 2.7 1.8 3.8 1.9.6 0 1.3-.4 1.5-.9.1-.2.1-.4-.1-.6l-.9-.8a.4.4 0 0 0-.5 0l-.5.4a.4.4 0 0 1-.4 0 5.5 5.5 0 0 1-2.3-2.2.4.4 0 0 1 0-.4l.4-.5a.4.4 0 0 0 0-.5l-.8-.9c-.2-.2-.4-.2-.6-.1-.5.2-.9.9-.8 1.4z" />
  </svg>
);

const SlackGlyph = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="10" width="8" height="4" rx="2" />
    <rect x="13" y="10" width="8" height="4" rx="2" />
    <rect x="10" y="3" width="4" height="8" rx="2" />
    <rect x="10" y="13" width="4" height="8" rx="2" />
  </svg>
);

const WidgetGlyph = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="4" width="18" height="13" rx="2.5" />
    <line x1="3" y1="8" x2="21" y2="8" />
    <circle cx="6" cy="6" r="0.5" fill="currentColor" />
    <circle cx="8" cy="6" r="0.5" fill="currentColor" />
    <path d="M14 19c0 1.5 1.5 2.5 3.5 2.5-.5-.6-.5-1.5 0-2.2" />
    <rect x="14.2" y="11" width="5.8" height="5" rx="1.8" />
  </svg>
);

const ApiGlyph = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
    <path d="M7 8l-4 4 4 4" />
    <path d="M17 8l4 4-4 4" />
    <path d="M14 5l-4 14" />
  </svg>
);

// Colored variants — Slack's classic 4-square mark + filled WhatsApp bubble.
// Used in the drill-in breadcrumbs + workspace rows where the design wants
// brand colors instead of monochrome.
const SlackGlyphColored = ({ size = 14 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24">
    <rect x="3" y="10" width="8" height="4" rx="2" fill="oklch(0.72 0.18 30)" />
    <rect x="13" y="10" width="8" height="4" rx="2" fill="oklch(0.65 0.15 150)" />
    <rect x="10" y="3" width="4" height="8" rx="2" fill="oklch(0.72 0.18 70)" />
    <rect x="10" y="13" width="4" height="8" rx="2" fill="oklch(0.55 0.18 290)" />
  </svg>
);

const WhatsAppGlyphColored = ({ size = 14 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="oklch(0.55 0.16 150)">
    <path d="M3.5 20.5l1.4-4.6A8 8 0 1 1 8.4 19.4l-4.9 1.1z" />
    <path d="M8.6 9.8c.2 1.2 1 2.6 2.2 3.7 1.2 1.1 2.7 1.8 3.8 1.9.6 0 1.3-.4 1.5-.9.1-.2.1-.4-.1-.6l-.9-.8a.4.4 0 0 0-.5 0l-.5.4a.4.4 0 0 1-.4 0 5.5 5.5 0 0 1-2.3-2.2.4.4 0 0 1 0-.4l.4-.5a.4.4 0 0 0 0-.5l-.8-.9c-.2-.2-.4-.2-.6-.1-.5.2-.9.9-.8 1.4z" fill="white" />
  </svg>
);

// ─── Hands tab (P3.4 — MCP tool integration) ────────────────────
const HandsTab = ({ agentId }: { agentId: string }) => {
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [tools, setTools] = useState<MCPTool[]>([]);
  const [discovering, setDiscovering] = useState(false);
  const [showAdd, setShowAdd] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const res = await listMCPServers(agentId);
      setServers(res.servers);
    } catch (e) {
      if (!(e instanceof ApiError)) console.warn('listMCPServers failed', e);
    }
  };

  const discover = async () => {
    setDiscovering(true);
    setError(null);
    try {
      const res = await discoverMCPTools(agentId);
      setTools(res.tools);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDiscovering(false);
    }
  };

  useEffect(() => {
    void refresh().then(() => void discover());
  }, [agentId]);

  const onAdd = async (s: { name: string; command: string; args: string[]; env: Record<string, string>; description: string }) => {
    setError(null);
    try {
      await addMCPServer(agentId, { ...s, transport: 'stdio', enabled: true });
      setShowAdd(false);
      await refresh();
      await discover();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const onToggle = async (name: string, enabled: boolean) => {
    try {
      await updateMCPServer(agentId, name, { enabled });
      await refresh();
      await discover();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const onRemove = async (name: string) => {
    if (!confirm(`Remove MCP server "${name}"? This stops exposing its tools to the agent.`)) return;
    try {
      await removeMCPServer(agentId, name);
      await refresh();
      await discover();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  // Group discovered tools under their server.
  const toolsByServer: Record<string, MCPTool[]> = {};
  for (const t of tools) {
    if (!toolsByServer[t.server_name]) toolsByServer[t.server_name] = [];
    toolsByServer[t.server_name].push(t);
  }

  return (
    <div className="sheet-section">
      <h3>Tools</h3>
      <p className="desc">
        MCP servers the agent can call as tools during <span className="mono">/run</span>. Each
        server is a local process (stdio transport) — operator provides the command and args.
        Tool catalog refreshes on every change.
      </p>

      {servers.length === 0 && !showAdd && (
        <div className="dim" style={{ fontSize: 12.5, lineHeight: 1.55, marginTop: 12 }}>
          No MCP servers connected yet. Add one (e.g. <span className="mono">npx -y @modelcontextprotocol/server-filesystem /tmp</span>)
          and the agent will be able to use its tools.
        </div>
      )}

      {servers.map((s) => (
        <div className={`row-item ${s.enabled ? 'on' : ''}`} key={s.name} style={{ display: 'block' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div className="ricon"><Icon name="route" size={14} /></div>
            <div className="rmain" style={{ flex: 1 }}>
              <div className="rname mono">{s.name}</div>
              <div className="rmeta">
                <span className="mono">{s.command} {s.args.join(' ')}</span>
                {toolsByServer[s.name] && <> · {toolsByServer[s.name].length} tools</>}
              </div>
            </div>
            <button
              className={`switch ${s.enabled ? 'on' : ''}`}
              onClick={() => void onToggle(s.name, !s.enabled)}
              aria-label={`Toggle ${s.name}`}
            />
            <button className="btn ghost sm" onClick={() => void onRemove(s.name)}>
              <Icon name="x" size={11} />
            </button>
          </div>
          {toolsByServer[s.name] && toolsByServer[s.name].length > 0 && (
            <div className="mcp-tool-list">
              {toolsByServer[s.name].map((t) => (
                <div className="mcp-tool" key={t.qualified_name}>
                  <span className="mono">{t.tool_name}</span>
                  <span className="dim" style={{ fontSize: 11.5 }}>
                    {t.description || '(no description)'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}

      {showAdd ? (
        <AddMCPForm
          onSubmit={onAdd}
          onCancel={() => setShowAdd(false)}
        />
      ) : (
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <button className="btn primary sm" onClick={() => setShowAdd(true)}>
            <Icon name="route" size={12} /> Connect MCP server
          </button>
          <button className="btn ghost sm" onClick={() => void discover()} disabled={discovering}>
            {discovering ? 'Refreshing…' : 'Refresh tools'}
          </button>
        </div>
      )}

      {error && (
        <div className="dim" style={{ fontSize: 11.5, color: 'var(--bad-fg)', marginTop: 10 }}>
          {error}
        </div>
      )}
    </div>
  );
};

const AddMCPForm = ({
  onSubmit,
  onCancel,
}: {
  onSubmit: (s: { name: string; command: string; args: string[]; env: Record<string, string>; description: string }) => void | Promise<void>;
  onCancel: () => void;
}) => {
  const [name, setName] = useState('');
  const [command, setCommand] = useState('');
  const [argsText, setArgsText] = useState('');
  const [envText, setEnvText] = useState('');
  const [description, setDescription] = useState('');

  const submit = () => {
    const args = argsText.split(/\s+/).filter(Boolean);
    const env: Record<string, string> = {};
    for (const line of envText.split('\n')) {
      const [k, ...rest] = line.split('=');
      if (k && k.trim()) env[k.trim()] = rest.join('=').trim();
    }
    void onSubmit({
      name: name.trim(),
      command: command.trim(),
      args,
      env,
      description: description.trim(),
    });
  };

  return (
    <div className="mcp-form">
      <div className="dim" style={{ fontSize: 11.5, lineHeight: 1.5, marginBottom: 10 }}>
        Example: filesystem server →{' '}
        <span className="mono">command=npx</span>{' '}
        <span className="mono">args=-y @modelcontextprotocol/server-filesystem /tmp</span>
      </div>
      <label className="mcp-field">
        <span>Name</span>
        <input
          className="mcp-input mono"
          placeholder="filesystem"
          value={name}
          onChange={(e) => setName(e.target.value)}
          autoComplete="off"
        />
      </label>
      <label className="mcp-field">
        <span>Command</span>
        <input
          className="mcp-input mono"
          placeholder="npx"
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          autoComplete="off"
        />
      </label>
      <label className="mcp-field">
        <span>Args (space-separated)</span>
        <input
          className="mcp-input mono"
          placeholder="-y @modelcontextprotocol/server-filesystem /tmp"
          value={argsText}
          onChange={(e) => setArgsText(e.target.value)}
          autoComplete="off"
        />
      </label>
      <label className="mcp-field">
        <span>Env (KEY=VAL per line, optional)</span>
        <textarea
          className="mcp-input mono"
          rows={2}
          placeholder="FOO=bar"
          value={envText}
          onChange={(e) => setEnvText(e.target.value)}
        />
      </label>
      <label className="mcp-field">
        <span>Description (optional)</span>
        <input
          className="mcp-input"
          placeholder="What does this server expose?"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
      </label>
      <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
        <button
          className="btn primary sm"
          onClick={submit}
          disabled={!name.trim() || !command.trim()}
        >
          Connect
        </button>
        <button className="btn ghost sm" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  );
};
