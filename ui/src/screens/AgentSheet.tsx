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

// ─── Drill-in shell (back nav + breadcrumb) ─────────────────────
interface DrillFrameProps {
  channelName: string;
  tileClass: string;
  glyph: React.ReactNode;
  connected: boolean;
  onBack: () => void;
  children: React.ReactNode;
}

const ChannelDrillFrame = ({
  channelName,
  tileClass,
  glyph,
  connected,
  onBack,
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
          <span className={`channel-tile sm ${tileClass}`}>{glyph}</span>
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
    </div>
    <div className="wcfg-config-pane">{children}</div>
  </div>
);

// ─── Web Widget panel (P3.5) ─────────────────────────────────────
const WebWidgetPanel = ({ agentId, onBack }: { agentId: string; onBack: () => void }) => {
  const [status, setStatus] = useState<WebChannelStatus | null>(null);
  const [freshSecret, setFreshSecret] = useState<WebChannelConnectResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [showSecret, setShowSecret] = useState(false);
  const [copied, setCopied] = useState<Record<string, boolean>>({});
  const [newDomain, setNewDomain] = useState('');
  const [error, setError] = useState<string | null>(null);

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

  const copy = (key: string, text: string) => {
    void navigator.clipboard?.writeText(text);
    setCopied((c) => ({ ...c, [key]: true }));
    setTimeout(() => setCopied((c) => ({ ...c, [key]: false })), 1600);
  };

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
    if (!confirm('Rotate signing secret? Any backend verifying the old secret will start failing.')) {
      return;
    }
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
    if (!confirm('Disconnect the Web Widget? The embed script on your site will stop responding.')) {
      return;
    }
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

  const addDomain = async () => {
    const d = newDomain.trim();
    if (!d) return;
    const existing = status?.allowed_domains ?? [];
    setBusy(true);
    setError(null);
    try {
      const next = await updateWebChannel(agentId, {
        allowed_domains: [...existing, d],
      });
      setStatus(next);
      setNewDomain('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const removeDomain = async (d: string) => {
    const existing = status?.allowed_domains ?? [];
    setBusy(true);
    setError(null);
    try {
      const next = await updateWebChannel(agentId, {
        allowed_domains: existing.filter((x) => x !== d),
      });
      setStatus(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

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

  return (
    <ChannelDrillFrame
      channelName="Web Widget"
      tileClass="t-widget"
      glyph={<WidgetGlyph />}
      connected={isConnected}
      onBack={onBack}
    >
      {!isConnected ? (
        <div className="wcfg-section">
          <div className="wcfg-section-head">
            <h3>Add the widget</h3>
            <span className="desc">
              Mint a widget ID + signing secret. You'll paste the embed snippet on your site;
              we'll route every visitor message to this agent.
            </span>
          </div>
          <button className="btn primary sm" onClick={() => void connect()} disabled={busy}>
            {busy ? 'Adding…' : 'Add Web Widget'}
          </button>
          {error && (
            <div className="dim" style={{ fontSize: 11.5, color: 'var(--bad-fg)', marginTop: 8 }}>
              {error}
            </div>
          )}
        </div>
      ) : (
        <>
          {freshSecret && (
            <div className="wcfg-section">
              <div className="wcfg-section-head">
                <h3>Signing secret</h3>
                <span className="desc">
                  Save this now — you won't see it again. Use it to verify webhook signatures
                  if you proxy widget messages through your backend.
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

          <div className="wcfg-section">
            <div className="wcfg-section-head">
              <h3>Install</h3>
              <span className="desc">
                Paste this snippet before <span className="mono">&lt;/body&gt;</span> on every
                page you want the widget on.
              </span>
            </div>
            <pre className="api-channel-curl">{embedSnippet}</pre>
            <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
              <button
                className={`btn sm ${copied.embed ? 'primary' : ''}`}
                onClick={() => copy('embed', embedSnippet)}
              >
                <Icon name={copied.embed ? 'check' : 'copy'} size={12} />
                {copied.embed ? 'Copied' : 'Copy snippet'}
              </button>
              <button className="btn sm ghost" onClick={() => copy('url', messageUrl)}>
                <Icon name={copied.url ? 'check' : 'copy'} size={12} />
                {copied.url ? 'Copied' : 'Copy endpoint URL'}
              </button>
            </div>
          </div>

          <div className="wcfg-section">
            <div className="wcfg-section-head">
              <h3>Signing secret</h3>
              <span className="desc">Stored on the backend. Rotate to revoke without disconnecting.</span>
            </div>
            <div className="wcfg-row">
              <div className="k">Current</div>
              <div className="v" style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                <span className="mono" style={{ fontSize: 12 }}>
                  {showSecret && freshSecret ? freshSecret.signing_secret : (status?.signing_secret_mask ?? '••••••••')}
                </span>
                {freshSecret && (
                  <button className="btn sm ghost" onClick={() => setShowSecret((s) => !s)}>
                    <Icon name="eye" size={12} /> {showSecret ? 'Hide' : 'Reveal'}
                  </button>
                )}
                <button className="btn sm" onClick={() => void rotate()} disabled={busy}>
                  <Icon name="refresh" size={12} /> Rotate
                </button>
              </div>
            </div>
          </div>

          <div className="wcfg-section">
            <div className="wcfg-section-head">
              <h3>Allowed domains</h3>
              <span className="desc">
                The widget only accepts messages from origins listed here. Leave empty to allow
                localhost for testing.
              </span>
            </div>
            <div className="domain-list">
              {(status?.allowed_domains ?? []).map((d) => (
                <span key={d} className="domain-chip">
                  {d}
                  <button
                    className="remove"
                    onClick={() => void removeDomain(d)}
                    disabled={busy}
                    title="Remove"
                  >
                    <Icon name="x" size={10} />
                  </button>
                </span>
              ))}
              <input
                className="domain-input"
                placeholder="add a domain…"
                value={newDomain}
                onChange={(e) => setNewDomain(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') void addDomain();
                }}
                disabled={busy}
              />
            </div>
          </div>

          <div className="wcfg-section">
            <div className="wcfg-section-head">
              <h3>Disconnect</h3>
            </div>
            <div className="danger-zone">
              <div>
                <div className="dz-title">Disconnect Web Widget</div>
                <div className="dz-meta">
                  The embed script will stop responding. Existing traces stay in the log.
                </div>
              </div>
              <button className="btn sm danger" onClick={() => void disconnect()} disabled={busy}>
                Disconnect
              </button>
            </div>
          </div>

          {error && (
            <div className="dim" style={{ fontSize: 11.5, color: 'var(--bad-fg)', marginTop: 8 }}>
              {error}
            </div>
          )}
        </>
      )}
    </ChannelDrillFrame>
  );
};

// ─── API channel drill panel (refactored from ApiChannelCard) ───
const ApiChannelPanel = ({ agentId, onBack }: { agentId: string; onBack: () => void }) => {
  const [details, setDetails] = useState<ApiChannelStatus | null>(null);
  const [freshToken, setFreshToken] = useState<ApiChannelConnectResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
  const curlBase =
    freshToken?.public_url ??
    (typeof window !== 'undefined' ? `${window.location.origin}/api/${agentId}/chat` : '');
  const curlToken = freshToken?.token ?? '<your-token>';
  const curl = `curl -X POST ${curlBase} \\\n  -H "Authorization: Bearer ${curlToken}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"request": "Hello, agent."}'`;

  return (
    <ChannelDrillFrame
      channelName="REST API"
      tileClass="t-api"
      glyph={<ApiGlyph />}
      connected={isConnected}
      onBack={onBack}
    >
      {!isConnected ? (
        <div className="wcfg-section">
          <div className="wcfg-section-head">
            <h3>Connect</h3>
            <span className="desc">
              Mint a bearer token. Callers POST to{' '}
              <span className="mono">/api/{agentId}/chat</span>.
            </span>
          </div>
          <button className="btn primary sm" onClick={() => void connect()} disabled={busy}>
            {busy ? 'Connecting…' : 'Connect REST API'}
          </button>
        </div>
      ) : (
        <>
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
              <span className="desc">
                {details?.token_mask && (
                  <>
                    Active token: <span className="mono">{details.token_mask}</span>
                    {details.last_used_at && (
                      <> · last used {new Date(details.last_used_at).toLocaleString()}</>
                    )}
                  </>
                )}
              </span>
            </div>
            <pre className="api-channel-curl">{curl}</pre>
          </div>

          <div className="wcfg-section">
            <div className="wcfg-section-head">
              <h3>Manage</h3>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn sm" onClick={() => void rotate()} disabled={busy}>
                <Icon name="refresh" size={12} /> Rotate token
              </button>
              <button className="btn sm danger" onClick={() => void disconnect()} disabled={busy}>
                Disconnect
              </button>
            </div>
          </div>
        </>
      )}

      {error && (
        <div className="dim" style={{ fontSize: 11.5, color: 'var(--bad-fg)', marginTop: 8 }}>
          {error}
        </div>
      )}
    </ChannelDrillFrame>
  );
};

// ─── Slack channel drill panel ─────────────────────────────────
const SlackChannelPanel = ({ agentId, onBack }: { agentId: string; onBack: () => void }) => {
  const [status, setStatus] = useState<SlackChannelStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    if (!confirm('Disconnect Slack? The agent will stop receiving messages from this workspace.')) {
      return;
    }
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

  return (
    <ChannelDrillFrame
      channelName="Slack"
      tileClass="t-slack"
      glyph={<SlackGlyph />}
      connected={!!status?.connected}
      onBack={onBack}
    >
      {!status ? (
        <div className="dim" style={{ fontSize: 12 }}>Loading…</div>
      ) : !status.configured ? (
        <div className="wcfg-section">
          <div className="wcfg-section-head">
            <h3>Operator setup required</h3>
            <span className="desc">
              Register a Slack app and set{' '}
              <span className="mono">SLACK_CLIENT_ID</span>,{' '}
              <span className="mono">SLACK_CLIENT_SECRET</span>,{' '}
              <span className="mono">SLACK_SIGNING_SECRET</span>, and{' '}
              <span className="mono">PUBLIC_BASE_URL</span> on the backend. {status.detail}
            </span>
          </div>
        </div>
      ) : !status.connected ? (
        <div className="wcfg-section">
          <div className="wcfg-section-head">
            <h3>Connect a workspace</h3>
            <span className="desc">
              Click below to launch Slack's install flow. After granting permissions you'll be
              redirected back here.
            </span>
          </div>
          <button className="btn primary sm" onClick={connect} disabled={busy}>
            Connect Slack
          </button>
        </div>
      ) : (
        <>
          <div className="wcfg-section">
            <div className="wcfg-section-head">
              <h3>Workspace</h3>
              <span className="desc">
                Connected to <strong>{status.team_name}</strong>
                {status.team_id && (
                  <> · <span className="mono">{status.team_id}</span></>
                )}
                {status.installed_at && (
                  <> · installed {new Date(status.installed_at).toLocaleDateString()}</>
                )}
              </span>
            </div>
          </div>
          {status.events_url && (
            <div className="wcfg-section">
              <div className="wcfg-section-head">
                <h3>Events URL</h3>
                <span className="desc">
                  Configure this in Slack app → Event Subscriptions.
                </span>
              </div>
              <pre className="api-channel-curl">{status.events_url}</pre>
            </div>
          )}
          <div className="wcfg-section">
            <div className="wcfg-section-head">
              <h3>Manage</h3>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn sm danger" onClick={() => void disconnect()} disabled={busy}>
                {busy ? 'Disconnecting…' : 'Disconnect Slack'}
              </button>
            </div>
          </div>
        </>
      )}

      {error && (
        <div className="dim" style={{ fontSize: 11.5, color: 'var(--bad-fg)', marginTop: 8 }}>
          {error}
        </div>
      )}
    </ChannelDrillFrame>
  );
};

// ─── WhatsApp channel drill panel ─────────────────────────────
const WhatsAppChannelPanel = ({ agentId, onBack }: { agentId: string; onBack: () => void }) => {
  const [status, setStatus] = useState<WhatsAppChannelStatus | null>(null);
  const [accountSid, setAccountSid] = useState('');
  const [authToken, setAuthToken] = useState('');
  const [fromNumber, setFromNumber] = useState('');
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    if (!confirm('Disconnect WhatsApp? The agent will stop receiving messages on this number.')) {
      return;
    }
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

  const showForm = !status?.connected || editing;

  return (
    <ChannelDrillFrame
      channelName="WhatsApp"
      tileClass="t-whatsapp"
      glyph={<WhatsAppGlyph />}
      connected={!!status?.connected}
      onBack={onBack}
    >
      {!status ? (
        <div className="dim" style={{ fontSize: 12 }}>Loading…</div>
      ) : !status.configured ? (
        <div className="wcfg-section">
          <div className="wcfg-section-head">
            <h3>Operator setup required</h3>
            <span className="desc">
              Set <span className="mono">PUBLIC_BASE_URL</span> on the backend so Twilio can
              reach the inbound webhook. {status.detail}
            </span>
          </div>
        </div>
      ) : (
        <>
          {status.connected && !editing && (
            <div className="wcfg-section">
              <div className="wcfg-section-head">
                <h3>Connected number</h3>
                <span className="desc">
                  <span className="mono">{status.from_number}</span>
                  {status.account_sid_mask && (
                    <> · SID <span className="mono">{status.account_sid_mask}</span></>
                  )}
                  {status.installed_at && (
                    <> · installed {new Date(status.installed_at).toLocaleDateString()}</>
                  )}
                </span>
              </div>
            </div>
          )}

          {showForm && (
            <div className="wcfg-section">
              <div className="wcfg-section-head">
                <h3>{status.connected ? 'Rotate credentials' : 'Connect a Twilio number'}</h3>
                <span className="desc">
                  From Twilio Console → Account → API keys & tokens. Sandbox{' '}
                  <span className="mono">From</span> is{' '}
                  <span className="mono">whatsapp:+14155238886</span>.
                </span>
              </div>
              <div className="whatsapp-form">
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
                <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
                  <button className="btn primary sm" onClick={() => void save()} disabled={busy}>
                    {busy ? 'Saving…' : status.connected ? 'Rotate' : 'Connect'}
                  </button>
                  {editing && (
                    <button
                      className="btn ghost sm"
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
            </div>
          )}

          {status.connected && status.inbound_url && !editing && (
            <div className="wcfg-section">
              <div className="wcfg-section-head">
                <h3>Twilio inbound URL</h3>
                <span className="desc">
                  Configure this in Messaging → Sender → Webhooks.
                </span>
              </div>
              <pre className="api-channel-curl">{status.inbound_url}</pre>
            </div>
          )}

          {status.connected && !editing && (
            <div className="wcfg-section">
              <div className="wcfg-section-head">
                <h3>Manage</h3>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="btn sm" onClick={() => setEditing(true)} disabled={busy}>
                  Rotate credentials
                </button>
                <button className="btn sm danger" onClick={() => void disconnect()} disabled={busy}>
                  Disconnect
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {error && (
        <div className="dim" style={{ fontSize: 11.5, color: 'var(--bad-fg)', marginTop: 8 }}>
          {error}
        </div>
      )}
    </ChannelDrillFrame>
  );
};

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
