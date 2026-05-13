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
  getAgentSecrets,
  listAgents,
  putAgentSecrets,
  updateAgent,
  type AgentSecretsResponse,
  type AgentSummary,
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

export const AgentSheet = ({ onClose }: { onClose: () => void }) => {
  const [tab, setTab] = useState<Tab>('brain');
  const [model, setModel] = useState('claude-sonnet-4-6');
  const [prompt, setPrompt] = useState(INITIAL_PROMPT);
  const [tools, setTools] = useState<Tool[]>(INITIAL_TOOLS);
  const [channels, setChannels] = useState<Channel[]>(INITIAL_CHANNELS);
  const [keys] = useState<Key[]>(INITIAL_KEYS);
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

  return (
    <>
      <div className="sheet-backdrop" onClick={onClose} />
      <div className="sheet" role="dialog" aria-label="Agent settings">
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
                  Claude Code reads traces, drafts changes, runs evals, opens proposals.
                </p>
                <div className="row-item on">
                  <div className="ricon">
                    <Icon name="code" size={16} />
                  </div>
                  <div className="rmain">
                    <div className="rname">Claude Code</div>
                    <div className="rmeta">
                      <span className="mono">github.com/you/support-agent</span> · last activity 2h
                      ago
                    </div>
                  </div>
                  <Tag kind="success">
                    <span className="dot" />
                    Connected
                  </Tag>
                </div>
              </div>
            </>
          )}

          {tab === 'hands' && (
            <div className="sheet-section">
              <h3>Tools</h3>
              <p className="desc">Functions, code, and MCP servers the agent can call.</p>
              {tools.map((t) => (
                <div className={`row-item ${t.on ? 'on' : ''}`} key={t.id}>
                  <div className="ricon">
                    <Icon name={toolIcon(t.src)} size={14} />
                  </div>
                  <div className="rmain">
                    <div
                      className="rname"
                      style={{
                        fontFamily: t.src === 'mcp' ? 'inherit' : 'var(--font-mono)',
                        fontSize: t.src === 'mcp' ? 13.5 : 13,
                      }}
                    >
                      {t.name}
                    </div>
                    <div className="rmeta">{t.desc}</div>
                  </div>
                  <button
                    className={`switch ${t.on ? 'on' : ''}`}
                    onClick={() => toggleTool(t.id)}
                    aria-label={`Toggle ${t.name}`}
                  />
                </div>
              ))}
              <button className="add-btn" style={{ marginTop: 8 }}>
                + Add tool or MCP server
              </button>
            </div>
          )}

          {tab === 'mouths' && (
            <div className="sheet-section">
              <h3>Where the agent talks</h3>
              <p className="desc">Each channel becomes a source of traces, feedback, and signal.</p>
              {channels.map((c) => (
                <div className={`row-item ${c.on ? 'on' : ''}`} key={c.id}>
                  <div className="ricon">
                    <Icon name={channelIcon(c.id)} size={14} />
                  </div>
                  <div className="rmain">
                    <div className="rname">{c.name}</div>
                    <div className="rmeta">
                      {c.desc}
                      {c.vol ? ` · ${c.vol}` : ''}
                    </div>
                  </div>
                  <button
                    className={`switch ${c.on ? 'on' : ''}`}
                    onClick={() => toggleChannel(c.id)}
                    aria-label={`Toggle ${c.name}`}
                  />
                </div>
              ))}
              <button className="add-btn" style={{ marginTop: 8 }}>
                + Add channel
              </button>
            </div>
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
