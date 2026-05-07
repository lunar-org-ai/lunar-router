import { useState } from 'react';
import { Icon, type IconName } from '../components/Icon';
import { Tag } from '../components/Tag';

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

interface ApiKey {
  id: string;
  name: string;
  mask: string;
  valid: boolean;
}

const initialTools: Tool[] = [
  { id: 't1', name: 'lookup_order', desc: 'tools/lookup_order.py', on: true, src: 'code' },
  { id: 't2', name: 'create_refund', desc: 'tools/create_refund.py', on: true, src: 'code' },
  { id: 't3', name: 'shopify-mcp', desc: '12 tools · MCP server', on: true, src: 'mcp' },
  { id: 't4', name: 'web_search', desc: 'Built-in', on: false, src: 'builtin' },
];

const initialChannels: Channel[] = [
  { id: 'whatsapp', name: 'WhatsApp', desc: '+55 11 9 4002-8922', on: true, vol: '~140 / day' },
  { id: 'api', name: 'REST API', desc: 'api.opentracy.dev/v1/chat', on: true, vol: '~80 / day' },
  { id: 'widget', name: 'Web widget', desc: 'embed.opentracy.dev/w/8af2', on: true, vol: '~60 / day' },
  { id: 'slack', name: 'Slack', desc: 'Not connected', on: false, vol: null },
];

const initialKeys: ApiKey[] = [
  { id: 'k1', name: 'Anthropic', mask: 'sk-ant-…f72a', valid: true },
  { id: 'k2', name: 'OpenAI', mask: 'sk-…1c0d', valid: true },
  { id: 'k3', name: 'Stripe', mask: 'sk_live_…b3', valid: true },
];

const models = [
  { id: 'claude-sonnet-4-5', name: 'Claude Sonnet 4.5', meta: 'Anthropic · default' },
  { id: 'gpt-5', name: 'GPT-5', meta: 'OpenAI' },
  { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5', meta: 'Anthropic · fast & cheap' },
  { id: 'llama-4-scout', name: 'Llama 4 Scout', meta: 'Self-hosted' },
];

export const AgentSheet = ({ onClose }: { onClose: () => void }) => {
  const [tab, setTab] = useState<'brain' | 'hands' | 'mouths' | 'keys'>('brain');
  const [model, setModel] = useState('claude-sonnet-4-5');
  const [prompt, setPrompt] = useState(
    `You are a customer support agent for an online store.\nYou are concise, friendly, and action-oriented.\nWhen the customer is frustrated, acknowledge once and pivot to action.`
  );
  const [tools, setTools] = useState<Tool[]>(initialTools);
  const [channels, setChannels] = useState<Channel[]>(initialChannels);
  const [keys] = useState<ApiKey[]>(initialKeys);

  const tabs = [
    { id: 'brain' as const, label: 'Brain' },
    { id: 'hands' as const, label: 'Hands' },
    { id: 'mouths' as const, label: 'Channels' },
    { id: 'keys' as const, label: 'Keys' },
  ];

  const toolIcon = (src: Tool['src']): IconName =>
    src === 'mcp' ? 'route' : src === 'builtin' ? 'sparkles' : 'code';
  const channelIcon = (id: string): IconName => (id === 'api' ? 'code' : 'chat');

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
          <button className="btn ghost sm" onClick={onClose}>
            <Icon name="x" size={14} />
          </button>
        </div>
        <div className="sheet-tabs">
          {tabs.map((t) => (
            <button key={t.id} className={`tab ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
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
                  What the agent always knows. Claude Code edits this file when it learns something new — your changes
                  here become its starting point.
                </p>
                <textarea className="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
                <div style={{ display: 'flex', gap: 8, marginTop: 10, alignItems: 'center' }}>
                  <span className="dim mono" style={{ fontSize: 11.5 }}>
                    system_prompt.md · 3 lines
                  </span>
                  <button className="btn sm" style={{ marginLeft: 'auto' }}>
                    <Icon name="code" size={12} /> Open in Claude Code
                  </button>
                </div>
              </div>

              <div className="sheet-section">
                <h3>Default model</h3>
                <p className="desc">The router can override this per request — see Routing in Policies.</p>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  {models.map((m) => (
                    <button
                      key={m.id}
                      className="row-item"
                      style={{
                        cursor: 'pointer',
                        borderColor: model === m.id ? 'var(--fg)' : undefined,
                        margin: 0,
                        textAlign: 'left',
                      }}
                      onClick={() => setModel(m.id)}
                    >
                      <div className="rmain">
                        <div className="rname">{m.name}</div>
                        <div className="rmeta">{m.meta}</div>
                      </div>
                      {model === m.id && <Icon name="check" size={14} />}
                    </button>
                  ))}
                </div>
              </div>

              <div className="sheet-section">
                <h3>Self-improvement engineer</h3>
                <p className="desc">Claude Code reads traces, drafts changes, runs evals, opens proposals.</p>
                <div className="row-item on">
                  <div className="ricon">
                    <Icon name="code" size={16} />
                  </div>
                  <div className="rmain">
                    <div className="rname">Claude Code</div>
                    <div className="rmeta">
                      <span className="mono">github.com/you/support-agent</span> · last activity 2h ago
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
                      className="rname mono"
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
                    onClick={() => setTools((ts) => ts.map((x) => (x.id === t.id ? { ...x, on: !x.on } : x)))}
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
                    onClick={() => setChannels((cs) => cs.map((x) => (x.id === c.id ? { ...x, on: !x.on } : x)))}
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
