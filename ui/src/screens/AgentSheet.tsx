/**
 * AgentSheet — slide-over for configuring the agent's editable surface.
 *
 * AutoHarness paper (arxiv 2603.03329) treats the agent as a single
 * editable surface refined through proposals. Manual operator edits go
 * through the same machinery as harness-driven candidates: every save
 * snapshots the live agent, bumps the version, writes a promote ledger
 * entry, and emits a Lesson with proposal_source="human". Result: manual
 * changes show up in Evolution alongside auto changes and roll back the
 * same way.
 *
 * Tab map:
 *   Brain  — system prompt R/W, default-model R/W, Claude Code status
 *   Hands  — built-in MCP server status; user-added tools deferred
 *   Channels — webhook channel real status; other channels deferred
 *   Keys   — env-var-backed key presence; provisioning deferred
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Icon, type IconName } from '../components/Icon';
import { Tag } from '../components/Tag';
import {
  ApiError,
  getAgentConfig,
  updatePrompt,
  updateRoute,
  type AgentConfigView,
  type AgentKeyStatus,
  type IntegrationStatus,
} from '../api';

const KNOWN_MODELS: Array<{ id: string; name: string; meta: string }> = [
  { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6', meta: 'Anthropic · default fallback' },
  { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5', meta: 'Anthropic · fast & cheap' },
  { id: 'claude-opus-4-7', name: 'Claude Opus 4.7', meta: 'Anthropic · max capability' },
  { id: 'gpt-5', name: 'GPT-5', meta: 'OpenAI' },
];

const integrationIcon = (name: string): IconName => {
  if (name.includes('Claude Code')) return 'code';
  if (name.includes('MCP')) return 'route';
  if (name.includes('Webhook') || name.includes('webhook')) return 'send';
  return 'sparkles';
};

const channelIcon = (name: string): IconName => {
  const n = name.toLowerCase();
  if (n.includes('whatsapp') || n.includes('slack')) return 'chat';
  if (n.includes('webhook') || n.includes('api')) return 'code';
  return 'chat';
};

export const AgentSheet = ({ onClose }: { onClose: () => void }) => {
  const [tab, setTab] = useState<'brain' | 'hands' | 'mouths' | 'keys'>('brain');
  const [config, setConfig] = useState<AgentConfigView | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [savedNote, setSavedNote] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // Brain — local drafts so the user can revert before saving
  const [promptDraft, setPromptDraft] = useState('');
  const [smallModelDraft, setSmallModelDraft] = useState<string | null>(null);
  const [bigModelDraft, setBigModelDraft] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const cfg = await getAgentConfig();
      setConfig(cfg);
      setPromptDraft(cfg.system_prompt.content);
      setSmallModelDraft(cfg.models.small);
      setBigModelDraft(cfg.models.big);
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `Backend ${e.status}: ${e.message}`
          : `Network error: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const promptDirty = !!config && promptDraft !== config.system_prompt.content;
  const modelsDirty =
    !!config &&
    (smallModelDraft !== config.models.small || bigModelDraft !== config.models.big);

  const savePrompt = async () => {
    setSaving(true);
    setError(null);
    try {
      const res = await updatePrompt(promptDraft);
      setConfig((c) =>
        c
          ? {
              ...c,
              version: res.new_version || c.version,
              system_prompt: { ...c.system_prompt, content: res.content },
            }
          : c,
      );
      setSavedNote(`Saved as ${res.new_version} · lesson ${res.lesson_id}`);
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `Save failed: ${e.status} — ${e.message}`
          : `Save failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setSaving(false);
    }
  };

  const saveModels = async () => {
    setSaving(true);
    setError(null);
    try {
      const body: { small?: string; big?: string } = {};
      if (smallModelDraft && smallModelDraft !== config?.models.small)
        body.small = smallModelDraft;
      if (bigModelDraft && bigModelDraft !== config?.models.big) body.big = bigModelDraft;
      const res = await updateRoute(body);
      setConfig((c) =>
        c
          ? {
              ...c,
              version: res.new_version || c.version,
              models: {
                small: res.small,
                big: res.big,
                confidence_threshold: res.confidence_threshold,
              },
            }
          : c,
      );
      setSavedNote(`Saved as ${res.new_version} · lesson ${res.lesson_id}`);
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `Save failed: ${e.status} — ${e.message}`
          : `Save failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setSaving(false);
    }
  };

  const claudeCodeIntegration = useMemo<IntegrationStatus | null>(
    () => config?.integrations.find((i) => i.name.includes('Claude Code')) ?? null,
    [config],
  );
  const mcpIntegration = useMemo<IntegrationStatus | null>(
    () => config?.integrations.find((i) => i.name.includes('MCP')) ?? null,
    [config],
  );
  const webhookIntegration = useMemo<IntegrationStatus | null>(
    () => config?.integrations.find((i) => i.name.includes('Webhook')) ?? null,
    [config],
  );

  const tabs = [
    { id: 'brain' as const, label: 'Brain' },
    { id: 'hands' as const, label: 'Hands' },
    { id: 'mouths' as const, label: 'Channels' },
    { id: 'keys' as const, label: 'Keys' },
  ];

  return (
    <>
      <div className="sheet-backdrop" onClick={onClose} />
      <div className="sheet" role="dialog" aria-label="Agent settings">
        <div className="sheet-head">
          <div className="sidebar-mark" style={{ width: 26, height: 26, borderRadius: 8 }} />
          <div style={{ flex: 1 }}>
            <h2>support-agent</h2>
            <div className="dim" style={{ fontSize: 12, marginTop: 2 }}>
              <span className="mono">{config?.version ?? '…'}</span> · live ·{' '}
              <span style={{ color: 'var(--accent-fg)' }}>● healthy</span>
            </div>
          </div>
          <button className="btn ghost sm" onClick={onClose}>
            <Icon name="x" size={14} />
          </button>
        </div>
        <div className="sheet-tabs">
          {tabs.map((t) => (
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
          {loading && !config && (
            <div className="dim" style={{ padding: 32, fontSize: 13 }}>Loading…</div>
          )}

          {error && (
            <div
              className="card card-pad"
              style={{ borderColor: 'var(--bad)', margin: 16 }}
            >
              <p className="dim" style={{ color: 'var(--bad)', margin: 0, fontSize: 12.5 }}>
                {error}
              </p>
            </div>
          )}

          {savedNote && (
            <div
              className="dim"
              style={{
                padding: '8px 16px',
                background: 'var(--accent-soft)',
                color: 'var(--accent-fg)',
                fontSize: 12.5,
                borderBottom: '1px solid var(--border)',
              }}
            >
              {savedNote}
            </div>
          )}

          {config && tab === 'brain' && (
            <>
              <div className="sheet-section">
                <h3>System prompt</h3>
                <p className="desc">
                  The agent's editable surface. Saving here writes{' '}
                  <span className="mono">{config.system_prompt.path}</span>, snapshots the
                  current version, bumps the agent version, and records a Lesson with{' '}
                  <span className="mono">proposal_source: human</span> — same machinery the
                  harness uses for auto edits.
                </p>
                <textarea
                  className="prompt"
                  value={promptDraft}
                  onChange={(e) => setPromptDraft(e.target.value)}
                />
                <div
                  style={{
                    display: 'flex',
                    gap: 8,
                    marginTop: 10,
                    alignItems: 'center',
                  }}
                >
                  <span className="dim mono" style={{ fontSize: 11.5 }}>
                    {config.system_prompt.path} · {promptDraft.split('\n').length} lines
                  </span>
                  <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
                    <button
                      className="btn sm ghost"
                      disabled={!promptDirty || saving}
                      onClick={() => setPromptDraft(config.system_prompt.content)}
                    >
                      Reset
                    </button>
                    <button
                      className="btn sm primary"
                      disabled={!promptDirty || saving}
                      onClick={savePrompt}
                    >
                      <Icon name="check" size={12} />{' '}
                      {saving ? 'Saving…' : 'Save & version'}
                    </button>
                  </div>
                </div>
              </div>

              <div className="sheet-section">
                <h3>Default models</h3>
                <p className="desc">
                  The router uses <span className="mono">small</span> by default and
                  escalates to <span className="mono">big</span> when confidence drops.
                  Saving rewrites <span className="mono">agent/pipeline/route.yaml</span>
                  {' '}and records a Lesson under <span className="mono">kind: router</span>.
                </p>
                <div style={{ marginBottom: 14 }}>
                  <div
                    className="dim"
                    style={{
                      fontSize: 11,
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                      marginBottom: 6,
                    }}
                  >
                    Small / fast
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    {KNOWN_MODELS.map((m) => (
                      <button
                        key={`small-${m.id}`}
                        className="row-item"
                        style={{
                          cursor: 'pointer',
                          borderColor:
                            smallModelDraft === m.id ? 'var(--fg)' : undefined,
                          margin: 0,
                          textAlign: 'left',
                        }}
                        onClick={() => setSmallModelDraft(m.id)}
                      >
                        <div className="rmain">
                          <div className="rname">{m.name}</div>
                          <div className="rmeta">{m.meta}</div>
                        </div>
                        {smallModelDraft === m.id && <Icon name="check" size={14} />}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <div
                    className="dim"
                    style={{
                      fontSize: 11,
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                      marginBottom: 6,
                    }}
                  >
                    Big / escalation
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    {KNOWN_MODELS.map((m) => (
                      <button
                        key={`big-${m.id}`}
                        className="row-item"
                        style={{
                          cursor: 'pointer',
                          borderColor: bigModelDraft === m.id ? 'var(--fg)' : undefined,
                          margin: 0,
                          textAlign: 'left',
                        }}
                        onClick={() => setBigModelDraft(m.id)}
                      >
                        <div className="rmain">
                          <div className="rname">{m.name}</div>
                          <div className="rmeta">{m.meta}</div>
                        </div>
                        {bigModelDraft === m.id && <Icon name="check" size={14} />}
                      </button>
                    ))}
                  </div>
                </div>
                <div
                  style={{
                    display: 'flex',
                    gap: 8,
                    marginTop: 12,
                    justifyContent: 'flex-end',
                  }}
                >
                  <button
                    className="btn sm ghost"
                    disabled={!modelsDirty || saving}
                    onClick={() => {
                      setSmallModelDraft(config.models.small);
                      setBigModelDraft(config.models.big);
                    }}
                  >
                    Reset
                  </button>
                  <button
                    className="btn sm primary"
                    disabled={!modelsDirty || saving}
                    onClick={saveModels}
                  >
                    <Icon name="check" size={12} />{' '}
                    {saving ? 'Saving…' : 'Save & version'}
                  </button>
                </div>
              </div>

              <div className="sheet-section">
                <h3>Self-improvement engineer</h3>
                <p className="desc">
                  Claude Code reads traces, drafts changes, runs evals, opens proposals.
                </p>
                <div className={`row-item ${claudeCodeIntegration?.available ? 'on' : ''}`}>
                  <div className="ricon">
                    <Icon name="code" size={16} />
                  </div>
                  <div className="rmain">
                    <div className="rname">Claude Code</div>
                    <div className="rmeta">
                      {claudeCodeIntegration?.detail || 'Status unknown'}
                    </div>
                  </div>
                  <Tag kind={claudeCodeIntegration?.available ? 'success' : 'bad'}>
                    <span className="dot" />
                    {claudeCodeIntegration?.available ? 'Connected' : 'Not configured'}
                  </Tag>
                </div>
              </div>
            </>
          )}

          {config && tab === 'hands' && (
            <div className="sheet-section">
              <h3>Tools</h3>
              <p className="desc">Functions, code, and MCP servers the agent can call.</p>
              {mcpIntegration && (
                <div className={`row-item ${mcpIntegration.available ? 'on' : ''}`}>
                  <div className="ricon">
                    <Icon name={integrationIcon(mcpIntegration.name)} size={14} />
                  </div>
                  <div className="rmain">
                    <div className="rname">{mcpIntegration.name}</div>
                    <div className="rmeta">{mcpIntegration.detail}</div>
                  </div>
                  <Tag kind={mcpIntegration.available ? 'success' : ''}>
                    <span className="dot" />
                    {mcpIntegration.available ? 'Available' : 'Missing'}
                  </Tag>
                </div>
              )}
              <div className="row-item" style={{ opacity: 0.55 }}>
                <div className="ricon">
                  <Icon name="route" size={14} />
                </div>
                <div className="rmain">
                  <div className="rname">External MCP servers</div>
                  <div className="rmeta">User-added MCP connections</div>
                </div>
                <Tag>Coming soon</Tag>
              </div>
              <div className="row-item" style={{ opacity: 0.55 }}>
                <div className="ricon">
                  <Icon name="code" size={14} />
                </div>
                <div className="rmain">
                  <div className="rname">Code-defined tools</div>
                  <div className="rmeta">Python functions registered as agent tools</div>
                </div>
                <Tag>Coming soon</Tag>
              </div>
              <div
                className="dim"
                style={{ fontSize: 12, marginTop: 12, lineHeight: 1.5 }}
              >
                A general tool registry isn't wired yet — the agent today calls techniques
                via its YAML pipeline, not a tool API. The introspection MCP is the only
                tool surface live.
              </div>
            </div>
          )}

          {config && tab === 'mouths' && (
            <div className="sheet-section">
              <h3>Where the agent talks</h3>
              <p className="desc">
                Each channel becomes a source of traces, feedback, and signal.
              </p>
              {webhookIntegration && (
                <div className={`row-item ${webhookIntegration.available ? 'on' : ''}`}>
                  <div className="ricon">
                    <Icon name={channelIcon(webhookIntegration.name)} size={14} />
                  </div>
                  <div className="rmain">
                    <div className="rname">Webhook (REST)</div>
                    <div className="rmeta mono">{webhookIntegration.detail}</div>
                  </div>
                  <Tag kind="success">
                    <span className="dot" />
                    Active
                  </Tag>
                </div>
              )}
              {[
                { id: 'whatsapp', name: 'WhatsApp', desc: 'Twilio / Meta business adapter' },
                { id: 'slack', name: 'Slack', desc: 'Slack app + bot token' },
                { id: 'widget', name: 'Web widget', desc: 'Embeddable chat widget' },
              ].map((c) => (
                <div className="row-item" style={{ opacity: 0.55 }} key={c.id}>
                  <div className="ricon">
                    <Icon name={channelIcon(c.name)} size={14} />
                  </div>
                  <div className="rmain">
                    <div className="rname">{c.name}</div>
                    <div className="rmeta">{c.desc}</div>
                  </div>
                  <Tag>Coming soon</Tag>
                </div>
              ))}
              <div
                className="dim"
                style={{ fontSize: 12, marginTop: 12, lineHeight: 1.5 }}
              >
                The webhook channel is the only adapter wired. Other channels need their
                own adapters before they can route traffic.
              </div>
            </div>
          )}

          {config && tab === 'keys' && (
            <div className="sheet-section">
              <h3>API keys</h3>
              <p className="desc">
                Provided as environment variables today (no secret store yet). The runtime
                checks each <span className="mono">env_var</span> at startup and surfaces
                the masked value here.
              </p>
              {config.keys.map((k: AgentKeyStatus) => (
                <div className={`row-item ${k.set ? 'on' : ''}`} key={k.env_var}>
                  <div className="ricon">
                    <Icon name="shield" size={14} />
                  </div>
                  <div className="rmain">
                    <div className="rname">{k.name}</div>
                    <div className="rmeta mono">
                      {k.set ? k.mask : `${k.env_var} (unset)`}
                    </div>
                  </div>
                  {k.set ? (
                    <Tag kind="success">
                      <span className="dot" />
                      Set
                    </Tag>
                  ) : (
                    <Tag kind="bad">
                      <span className="dot" />
                      Missing
                    </Tag>
                  )}
                </div>
              ))}
              <div
                className="dim"
                style={{ fontSize: 12, marginTop: 12, lineHeight: 1.5 }}
              >
                Provisioning new keys through the UI requires a secret store; deferred
                until that lands. Today: export env vars before starting the runtime.
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
};
