/**
 * Onboarding — Guided conversational flow (port of claude-design
 * "Onboarding Guided.html"). ChatGPT/Claude-style 3-pane layout that
 * materializes progressively:
 *
 *   • G1: minimal centered welcome (no sidebar, no right panel)
 *   • G2: user describes intent → right panel slides in with a live
 *         agent-config preview
 *   • G3: tone/topic settled → sidebar fades in as a draft agent
 *   • G4: channel chosen → Slack connect card inline; sidebar partial
 *   • G5: Slack connected → model picker; right panel highlights channel
 *   • G6: model picked → "Ready to launch" summary; everything filled
 *
 * Server contract is unchanged (V2 session: /session, /say, /decide,
 * /rewind). Phase + decisions drive which panes are visible and what
 * the right panel shows — the brain owns the prose, the session module
 * owns the state machine, the UI owns the visual progression.
 */
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type FormEvent,
  type KeyboardEvent,
  type ReactNode,
} from 'react';

import {
  completeOnboarding,
  decideOnboardingV2,
  getOnboardingV2Session,
  getSlackConnectStatus,
  rewindOnboardingV2,
  sayOnboardingV2,
  skipOnboarding,
  type ChannelPickerCard,
  type ConnectApiCard,
  type ConnectSlackCard,
  type ConnectWebCard,
  type ConnectWhatsappCard,
  type ModelPickerCard,
  type OnboardingCard,
  type OnboardingPhase,
  type OnboardingState,
  type OnboardingV2Session,
  type OnboardingV2Turn,
  type TracePreviewCard,
} from '../api';
import { Icon } from '../components/Icon';

interface OnboardingProps {
  onDone: (next: OnboardingState | null) => void;
}

// ─── Atoms ───────────────────────────────────────────────────────

const SlackGlyph = ({ size = 16 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" style={{ display: 'block', flexShrink: 0 }}>
    <path d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zM6.313 15.165a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313z" fill="#E01E5A" />
    <path d="M8.834 5.042a2.528 2.528 0 0 1-2.521-2.52A2.528 2.528 0 0 1 8.834 0a2.528 2.528 0 0 1 2.521 2.522v2.52H8.834zM8.834 6.313a2.528 2.528 0 0 1 2.521 2.521 2.528 2.528 0 0 1-2.521 2.521H2.522A2.528 2.528 0 0 1 0 8.834a2.528 2.528 0 0 1 2.522-2.521h6.312z" fill="#36C5F0" />
    <path d="M18.956 8.834a2.528 2.528 0 0 1 2.522-2.521A2.528 2.528 0 0 1 24 8.834a2.528 2.528 0 0 1-2.522 2.521h-2.522V8.834zM17.688 8.834a2.528 2.528 0 0 1-2.523 2.521 2.527 2.527 0 0 1-2.52-2.521V2.522A2.527 2.527 0 0 1 15.165 0a2.528 2.528 0 0 1 2.523 2.522v6.312z" fill="#2EB67D" />
    <path d="M15.165 18.956a2.528 2.528 0 0 1 2.523 2.522A2.528 2.528 0 0 1 15.165 24a2.527 2.527 0 0 1-2.52-2.522v-2.522h2.52zM15.165 17.688a2.527 2.527 0 0 1-2.52-2.523 2.526 2.526 0 0 1 2.52-2.52h6.313A2.527 2.527 0 0 1 24 15.165a2.528 2.528 0 0 1-2.522 2.523h-6.313z" fill="#ECB22E" />
  </svg>
);

const ROT_WORDS = [
  'support agent',
  'sales bot',
  'refund flow',
  'lead qualifier',
  'onboarding guide',
  'research assistant',
];

const RotatingWord = ({ words = ROT_WORDS, interval = 2200 }: { words?: string[]; interval?: number }) => {
  const [idx, setIdx] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setIdx((i) => (i + 1) % words.length), interval);
    return () => clearInterval(t);
  }, [words, interval]);
  return (
    <span className="gd-rot">
      <span className="gd-rot-sizer" aria-hidden="true">{words[idx]}</span>
      {words.map((w, i) => (
        <span
          key={w}
          className={`gd-rot-item ${
            i === idx
              ? 'is-on'
              : i === (idx - 1 + words.length) % words.length
                ? 'is-out'
                : ''
          }`}
          aria-hidden={i !== idx}
        >
          {w}
        </span>
      ))}
      <span className="gd-rot-sr" aria-live="polite">{words[idx]}</span>
    </span>
  );
};

// ─── Sidebar (left) ──────────────────────────────────────────────

type SidebarState = 'empty' | 'draft' | 'partial' | 'ready';

const Sidebar = ({
  state,
  progress,
  agentName,
  channel,
  model,
  toolCount,
}: {
  state: Exclude<SidebarState, 'empty'>;
  progress: number;
  agentName: string;
  channel: ReactNode | null;
  model: string | null;
  toolCount: number;
}) => {
  const dotClass = state === 'ready' ? 'active' : state === 'draft' ? 'draft' : '';
  const subtitle =
    state === 'ready' ? 'Ready to launch' : state === 'partial' ? 'Configuring…' : 'Drafting…';

  const rows: Array<{ k: string; v: ReactNode; pending: boolean }> = [
    { k: 'Status', v: subtitle, pending: false },
    { k: 'Channel', v: channel ?? 'Not chosen', pending: !channel },
    { k: 'Model', v: model ?? 'Not chosen', pending: !model },
  ];
  if (state === 'ready') {
    rows.push({ k: 'Tools', v: `${toolCount} selected`, pending: false });
  }

  return (
    <aside className="gd-side">
      <div className="gd-side-head">
        <div className="gd-side-mark" />
        <div className="gd-side-brand">OpenTracy <span className="dim">Evolution</span></div>
      </div>

      <div className="gd-side-section">
        <div className="gd-side-label">Your agent</div>
        <div className={`gd-side-agent ${dotClass}`}>
          <div className="name">
            <span className="dot" />
            {agentName}
          </div>
          <div className="gd-side-meta">
            {rows.map((r) => (
              <div key={r.k} className="gd-side-meta-row">
                <span className="k">{r.k}</span>
                <span className={`v ${r.pending ? 'pending' : ''}`}>{r.v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="gd-side-progress">
        <div className="gd-side-progress-label">
          <span>Setup progress</span>
          <span>{progress}%</span>
        </div>
        <div className="gd-side-progress-bar">
          <div className="gd-side-progress-fill" style={{ width: `${progress}%` }} />
        </div>
      </div>

      <button className="gd-side-new" disabled>
        <Icon name="plus" size={13} />
        New agent
      </button>

      <div className="gd-side-foot">
        <div className="av">JM</div>
        <span>Jamie Marsh · Acme</span>
      </div>
    </aside>
  );
};

// ─── Right panel ─────────────────────────────────────────────────

type PanelStage = 'initial' | 'channel' | 'connected' | 'full';

const Panel = ({
  stage,
  agentName,
  purpose,
  tone,
  channel,
  model,
  updatedSection,
}: {
  stage: PanelStage;
  agentName: string;
  purpose: string;
  tone: string | null;
  channel: ReactNode | null;
  model: string | null;
  updatedSection: 'behavior' | 'tone' | 'channel' | 'model' | null;
}) => {
  const promptBody = (
    <>
      You are a {agentName} agent.{'\n\n'}
      <strong>Behavior</strong>{'\n'}
      {purpose ? `• ${purpose}` : '• [awaiting your description…]'}{'\n\n'}
      <strong>Tone</strong>{'\n'}
      {tone ? (
        <span className={updatedSection === 'tone' ? 'new' : undefined}>{tone}</span>
      ) : (
        <span className="pending">[awaiting tone…]</span>
      )}
    </>
  );

  const sectionClass = (key: 'behavior' | 'tone' | 'channel' | 'model') =>
    `gd-panel-section ${updatedSection === key ? 'gd-section-updated' : ''}`;

  return (
    <div className="gd-panel">
      <div className="gd-panel-head">
        <span className="gd-panel-h">Agent config</span>
        <span className="gd-panel-live"><span className="dot" /> live</span>
      </div>
      <div className="gd-panel-body">

        <div className="gd-panel-section">
          <div className="gd-panel-name">{agentName}</div>
          <div style={{ fontSize: 12, color: 'var(--fg-muted)', marginTop: 2 }}>v0.1 · draft</div>
        </div>

        <div className={sectionClass(updatedSection === 'tone' ? 'tone' : 'behavior')}>
          <h4 className="gd-panel-section-h">
            Agent instructions
            {(updatedSection === 'behavior' || updatedSection === 'tone') && <span className="live-dot" />}
          </h4>
          <div className="gd-panel-prompt">{promptBody}</div>
        </div>

        <div className={sectionClass('channel')}>
          <h4 className="gd-panel-section-h">Channel</h4>
          {channel ? (
            <div className="gd-panel-chips">
              <span className="gd-panel-chip accent">{channel}</span>
            </div>
          ) : (
            <div className="gd-panel-empty">Not chosen yet…</div>
          )}
        </div>

        <div className={sectionClass('model')}>
          <h4 className="gd-panel-section-h">Model</h4>
          {model ? (
            <div className="gd-panel-chips">
              <span className="gd-panel-chip accent">{model}</span>
              {stage === 'full' && <span className="gd-panel-chip">+ Sonnet fallback</span>}
            </div>
          ) : (
            <div className="gd-panel-empty">Not chosen yet…</div>
          )}
        </div>

      </div>
    </div>
  );
};

const PanelEmpty = () => (
  <div className="gd-panel">
    <div className="gd-panel-head">
      <span className="gd-panel-h">Agent config</span>
      <span className="gd-panel-live" style={{ color: 'var(--fg-subtle)' }}>idle</span>
    </div>
    <div className="gd-panel-empty-hero">
      <div className="icon"><Icon name="file" size={16} /></div>
      <div className="text">Your agent's instructions and config will appear here as we go.</div>
    </div>
  </div>
);

// ─── Cards (re-skinned with gd-card) ─────────────────────────────

const channelIcon = (id: string): ReactNode => {
  if (id === 'slack') return <SlackGlyph size={24} />;
  if (id === 'whatsapp') return <Icon name="phone" size={22} style={{ color: '#25D366' }} />;
  if (id === 'web') return <Icon name="globe" size={22} style={{ color: 'var(--info-fg)' }} />;
  if (id === 'api') return <Icon name="code" size={22} style={{ color: 'var(--fg)' }} />;
  return <Icon name="chat" size={22} />;
};

const ChannelPickerView = ({
  card,
  onPick,
  disabled,
  pendingPick,
}: {
  card: ChannelPickerCard;
  onPick: (id: string) => void;
  disabled: boolean;
  pendingPick: string | null;
}) => {
  // Local "armed" state: the option the user just clicked. Falls back to
  // the server's recommendation when nothing is armed. We highlight the
  // armed option visually until the server's next session arrives and
  // replaces the picker with a settled chip.
  const [armed, setArmed] = useState<string | null>(null);
  const selected = pendingPick ?? armed ?? card.recommended_id;

  const pick = (id: string) => {
    setArmed(id);
    onPick(id);
  };

  return (
    <div className="gd-card">
      <div className="gd-card-head">
        <span className="gd-card-h">Pick a channel</span>
        <span className="gd-card-sub">Start with one — add more later</span>
      </div>
      <div className="gd-card-body">
        <div className="gd-channels">
          {card.options.map((opt) => (
            <button
              key={opt.id}
              type="button"
              className={`gd-channel ${opt.id === selected ? 'on' : ''}`}
              onClick={() => pick(opt.id)}
              disabled={disabled}
            >
              <div className="icon">{channelIcon(opt.id)}</div>
              <div className="name">{opt.name}</div>
              <div className="sub">{opt.sub}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

const ModelPickerView = ({
  card,
  onPick,
  disabled,
  pendingPick,
}: {
  card: ModelPickerCard;
  onPick: (id: string) => void;
  disabled: boolean;
  pendingPick: string | null;
}) => {
  const [armed, setArmed] = useState<string | null>(null);
  const selected = pendingPick ?? armed ?? card.recommended_id;

  const pick = (id: string) => {
    setArmed(id);
    onPick(id);
  };

  return (
    <div className="gd-card">
      <div className="gd-card-head">
        <span className="gd-card-h">Choose a model</span>
        <span className="gd-card-sub">You can switch anytime</span>
      </div>
      <div className="gd-card-body">
        <div className="gd-models">
          {card.options.map((opt) => {
            const isSelected = opt.id === selected;
            const isRec = opt.id === card.recommended_id;
            return (
              <button
                key={opt.id}
                type="button"
                className={`gd-model ${isSelected ? 'on' : ''}`}
                onClick={() => pick(opt.id)}
                disabled={disabled}
              >
                <span className="gd-model-radio" />
                <div className="gd-model-info">
                  <div className="gd-model-name">{opt.name}</div>
                  <div className="gd-model-meta">
                    {opt.cost_per_million_in} / 1M in · ~{opt.p50_latency_s}s p50 · {isRec ? card.rationale : opt.tag}
                  </div>
                </div>
                {isRec && <span className="rec">Recommended</span>}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

const ConnectSlackView = ({
  card,
  plaintextKey,
  liveStatus,
}: {
  card: ConnectSlackCard;
  plaintextKey: string | null;
  liveStatus: 'waiting' | 'connected';
}) => {
  const [revealed, setRevealed] = useState(false);
  const [copied, setCopied] = useState(false);
  const masked = card.agent_key_preview;
  const display = revealed && plaintextKey ? plaintextKey : masked;

  const copy = async () => {
    const v = plaintextKey ?? masked;
    if (!v) return;
    try {
      await navigator.clipboard.writeText(v);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      /* clipboard fail in insecure contexts — ignore */
    }
  };

  const installHref = card.install_url || '#';

  return (
    <div className="gd-card">
      <div className="gd-card-head">
        <SlackGlyph size={14} />
        <span className="gd-card-h">Connect Slack</span>
        <span className="gd-card-sub">{liveStatus === 'connected' ? 'Connected' : '~30 seconds'}</span>
      </div>
      <div className="gd-card-body">
        <div className="gd-slack">
          <div style={{ fontSize: 12.5, color: 'var(--fg-muted)', lineHeight: 1.55 }}>
            Install the OpenTracy app in your workspace, then paste the agent key below in the Slack app config.
          </div>
          <div className="gd-slack-row">
            <span className="label">Your agent key</span>
            <button
              type="button"
              className="button-link"
              onClick={() => setRevealed((v) => !v)}
              disabled={!plaintextKey}
            >
              {revealed ? 'Hide' : 'Reveal'}
            </button>
          </div>
          <div className="gd-input">
            <Icon name="lock" size={13} style={{ color: 'var(--fg-subtle)' }} />
            <input value={display ?? ''} readOnly placeholder="ot_live_…" />
            <button
              type="button"
              className="button-link"
              onClick={copy}
              style={{ marginRight: -4 }}
            >
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
          {liveStatus === 'connected' && (
            <div className="gd-connected">
              <div className="icon"><Icon name="check" size={12} /></div>
              <div>
                <div style={{ fontWeight: 500 }}>Slack connected</div>
                <div className="meta">First message received</div>
              </div>
            </div>
          )}
        </div>
      </div>
      <div className="gd-card-actions">
        <a
          className="gd-btn-primary"
          href={installHref}
          target="_blank"
          rel="noreferrer noopener"
          style={{ textDecoration: 'none' }}
        >
          <SlackGlyph size={13} />
          Open install page
        </a>
        {liveStatus !== 'connected' && (
          <span style={{ fontSize: 11.5, color: 'var(--fg-subtle)' }}>
            Waiting for your first message in Slack… (polling 3s)
          </span>
        )}
      </div>
    </div>
  );
};

const CopyableInput = ({
  value,
  plaintextKey,
  fallbackValue,
}: {
  value: string | null;
  plaintextKey: string | null;
  fallbackValue: string;
}) => {
  const [revealed, setRevealed] = useState(false);
  const [copied, setCopied] = useState(false);
  const display = revealed && plaintextKey ? plaintextKey : (value || fallbackValue);
  const copyValue = plaintextKey ?? value ?? fallbackValue;

  const copy = async () => {
    if (!copyValue) return;
    try {
      await navigator.clipboard.writeText(copyValue);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      /* clipboard fail in insecure contexts */
    }
  };

  return (
    <div className="gd-input">
      <Icon name="lock" size={13} style={{ color: 'var(--fg-subtle)' }} />
      <input value={display ?? ''} readOnly />
      {plaintextKey && (
        <button type="button" className="button-link" onClick={() => setRevealed((v) => !v)}>
          {revealed ? 'Hide' : 'Reveal'}
        </button>
      )}
      <button type="button" className="button-link" onClick={copy} style={{ marginRight: -4 }}>
        {copied ? 'Copied' : 'Copy'}
      </button>
    </div>
  );
};

const CodeBlock = ({ children }: { children: string }) => {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(children);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      /* ignore */
    }
  };
  return (
    <div className="gd-codeblock">
      <pre>{children}</pre>
      <button type="button" className="gd-codeblock-copy" onClick={copy}>
        <Icon name={copied ? 'check' : 'copy'} size={12} /> {copied ? 'Copied' : 'Copy'}
      </button>
    </div>
  );
};

const ConnectWhatsappView = ({
  card,
  liveStatus,
}: {
  card: ConnectWhatsappCard;
  liveStatus: 'waiting' | 'connected';
}) => (
  <div className="gd-card">
    <div className="gd-card-head">
      <Icon name="phone" size={14} style={{ color: '#25D366' }} />
      <span className="gd-card-h">Connect WhatsApp</span>
      <span className="gd-card-sub">{liveStatus === 'connected' ? 'Connected' : 'Meta Business setup'}</span>
    </div>
    <div className="gd-card-body">
      <div className="gd-slack">
        <div style={{ fontSize: 12.5, color: 'var(--fg-muted)', lineHeight: 1.55 }}>
          In Meta Business Manager, point your WhatsApp webhook at the URL below and paste the verify token.
        </div>
        <div className="gd-slack-row">
          <span className="label">Webhook URL</span>
        </div>
        <CopyableInput value={card.webhook_url} plaintextKey={null} fallbackValue="" />
        <div className="gd-slack-row">
          <span className="label">Verify token</span>
        </div>
        <CopyableInput value={card.verify_token_preview} plaintextKey={null} fallbackValue="" />
        {liveStatus === 'connected' && (
          <div className="gd-connected">
            <div className="icon"><Icon name="check" size={12} /></div>
            <div>
              <div style={{ fontWeight: 500 }}>WhatsApp connected</div>
              <div className="meta">First message received</div>
            </div>
          </div>
        )}
      </div>
    </div>
    <div className="gd-card-actions">
      <a className="gd-btn-primary" href="https://business.facebook.com/" target="_blank" rel="noreferrer noopener" style={{ textDecoration: 'none' }}>
        Open Meta Business
      </a>
      {liveStatus !== 'connected' && (
        <span style={{ fontSize: 11.5, color: 'var(--fg-subtle)' }}>
          Waiting for your first WhatsApp message…
        </span>
      )}
    </div>
  </div>
);

const ConnectWebView = ({
  card,
  liveStatus,
}: {
  card: ConnectWebCard;
  liveStatus: 'waiting' | 'connected';
}) => (
  <div className="gd-card">
    <div className="gd-card-head">
      <Icon name="globe" size={14} style={{ color: 'var(--info-fg)' }} />
      <span className="gd-card-h">Embed Web widget</span>
      <span className="gd-card-sub">{liveStatus === 'connected' ? 'Connected' : 'Drop the snippet on your site'}</span>
    </div>
    <div className="gd-card-body">
      <div className="gd-slack">
        <div style={{ fontSize: 12.5, color: 'var(--fg-muted)', lineHeight: 1.55 }}>
          Paste this snippet before <code style={{ fontFamily: 'var(--font-mono)' }}>&lt;/body&gt;</code> on every page where the widget should appear.
        </div>
        <CodeBlock>{card.embed_snippet}</CodeBlock>
        {liveStatus === 'connected' && (
          <div className="gd-connected">
            <div className="icon"><Icon name="check" size={12} /></div>
            <div>
              <div style={{ fontWeight: 500 }}>Widget connected</div>
              <div className="meta">First visitor message received</div>
            </div>
          </div>
        )}
      </div>
    </div>
    <div className="gd-card-actions">
      {liveStatus !== 'connected' && (
        <span style={{ fontSize: 11.5, color: 'var(--fg-subtle)' }}>
          Waiting for your first widget message…
        </span>
      )}
    </div>
  </div>
);

const ConnectApiView = ({
  card,
  plaintextKey,
  liveStatus,
}: {
  card: ConnectApiCard;
  plaintextKey: string | null;
  liveStatus: 'waiting' | 'connected';
}) => (
  <div className="gd-card">
    <div className="gd-card-head">
      <Icon name="code" size={14} />
      <span className="gd-card-h">Use the HTTP API</span>
      <span className="gd-card-sub">{liveStatus === 'connected' ? 'Connected' : 'curl it from anywhere'}</span>
    </div>
    <div className="gd-card-body">
      <div className="gd-slack">
        <div className="gd-slack-row">
          <span className="label">Agent key</span>
        </div>
        <CopyableInput value={card.agent_key_preview} plaintextKey={plaintextKey} fallbackValue="ot_live_…" />
        <div style={{ fontSize: 12.5, color: 'var(--fg-muted)', lineHeight: 1.55, marginTop: 4 }}>
          Then send a request:
        </div>
        <CodeBlock>{card.curl_example}</CodeBlock>
        {liveStatus === 'connected' && (
          <div className="gd-connected">
            <div className="icon"><Icon name="check" size={12} /></div>
            <div>
              <div style={{ fontWeight: 500 }}>API connected</div>
              <div className="meta">First call received</div>
            </div>
          </div>
        )}
      </div>
    </div>
  </div>
);

const TracePreviewView = ({ card }: { card: TracePreviewCard }) => {
  const summary = card.summary ?? {};
  return (
    <div className="gd-card">
      <div className="gd-card-head">
        <Icon name="hash" size={12} />
        <span className="gd-card-h">{String(summary.channel ?? 'support-test')}</span>
        <span className="gd-card-sub">{card.trace_id}</span>
      </div>
      <div className="gd-card-body" style={{ fontSize: 13, color: 'var(--fg-muted)' }}>
        First real message — {String(summary.turn_count ?? 1)} turn · {String(summary.duration_s ?? 0)}s
      </div>
    </div>
  );
};

const RenderCard = ({
  card,
  onPick,
  disabled,
  plaintextKey,
  liveStatus,
  pendingPick,
}: {
  card: OnboardingCard;
  onPick: (key: 'model' | 'channel', value: string) => void;
  disabled: boolean;
  plaintextKey: string | null;
  liveStatus: 'waiting' | 'connected';
  pendingPick: { key: 'model' | 'channel'; value: string } | null;
}) => {
  switch (card.type) {
    case 'model_picker':
      return (
        <ModelPickerView
          card={card}
          onPick={(v) => onPick('model', v)}
          disabled={disabled}
          pendingPick={pendingPick?.key === 'model' ? pendingPick.value : null}
        />
      );
    case 'channel_picker':
      return (
        <ChannelPickerView
          card={card}
          onPick={(v) => onPick('channel', v)}
          disabled={disabled}
          pendingPick={pendingPick?.key === 'channel' ? pendingPick.value : null}
        />
      );
    case 'connect_slack':
      return <ConnectSlackView card={card} plaintextKey={plaintextKey} liveStatus={liveStatus} />;
    case 'connect_whatsapp':
      return <ConnectWhatsappView card={card} liveStatus={liveStatus} />;
    case 'connect_web':
      return <ConnectWebView card={card} liveStatus={liveStatus} />;
    case 'connect_api':
      return <ConnectApiView card={card} plaintextKey={plaintextKey} liveStatus={liveStatus} />;
    case 'trace_preview':
      return <TracePreviewView card={card} />;
    default:
      return null;
  }
};

// ─── Turn renderers ──────────────────────────────────────────────

const AssistantTurn = ({
  turn,
  onPick,
  pending,
  plaintextKey,
  liveStatus,
  pendingPick,
}: {
  turn: OnboardingV2Turn;
  onPick: (key: 'model' | 'channel', value: string) => void;
  pending: boolean;
  plaintextKey: string | null;
  liveStatus: 'waiting' | 'connected';
  pendingPick: { key: 'model' | 'channel'; value: string } | null;
}) => (
  <div className="gd-msg claude">
    <div className="gd-msg-meta">
      <div className="av c">C</div>
      <span className="name">Claude</span>
      <span>· OpenTracy</span>
    </div>
    <div className="gd-msg-body">
      {turn.text && turn.text.split('\n\n').map((p, i) => <p key={i}>{p}</p>)}
      {turn.cards.map((c, i) => (
        <RenderCard
          key={i}
          card={c}
          onPick={onPick}
          disabled={pending}
          plaintextKey={plaintextKey}
          liveStatus={liveStatus}
          pendingPick={pendingPick}
        />
      ))}
    </div>
  </div>
);

const UserTurn = ({
  turn,
  onEdit,
  disabled,
}: {
  turn: OnboardingV2Turn;
  onEdit: (key: 'model' | 'channel') => void;
  disabled: boolean;
}) => {
  const isChip = !turn.text && turn.decision_key;
  return (
    <div className="gd-msg user">
      <div className="gd-msg-meta">
        <div className="av u">JM</div>
        <span className="name">You</span>
      </div>
      <div className="gd-msg-body">
        {isChip ? (
          <span className="gd-settled">
            <Icon name="check" size={11} />
            {turn.decision_label}
            <button
              type="button"
              className="edit"
              onClick={() => onEdit(turn.decision_key as 'model' | 'channel')}
              disabled={disabled}
            >
              edit
            </button>
          </span>
        ) : (
          <p>{turn.text}</p>
        )}
      </div>
    </div>
  );
};

// ─── Composers ───────────────────────────────────────────────────

const WelcomeComposer = ({
  onSubmit,
  pending,
}: {
  onSubmit: (text: string) => void;
  pending: boolean;
}) => {
  const [value, setValue] = useState('');
  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    const v = value.trim();
    if (!v || pending) return;
    onSubmit(v);
    setValue('');
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };
  return (
    <form className="gd-welcome-input" onSubmit={submit}>
      <textarea
        placeholder="What should your agent help with?"
        rows={2}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKey}
        disabled={pending}
      />
      <div className="gd-welcome-input-foot">
        <div className="gd-welcome-input-tools">
          <button type="button" className="gd-welcome-tool" title="Attach" disabled>
            <Icon name="file" size={14} />
          </button>
          <button type="button" className="gd-welcome-tool" title="Link" disabled>
            <Icon name="link" size={14} />
          </button>
        </div>
        <button
          type="submit"
          className="gd-welcome-send"
          disabled={pending || !value.trim()}
          aria-label="Send"
        >
          <Icon name="arrowUp" size={14} />
        </button>
      </div>
    </form>
  );
};

const ChatComposer = ({
  onSubmit,
  pending,
  placeholder,
}: {
  onSubmit: (text: string) => void;
  pending: boolean;
  placeholder: string;
}) => {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    const v = value.trim();
    if (!v || pending) return;
    onSubmit(v);
    setValue('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  }, [value]);

  return (
    <form className="gd-composer" onSubmit={submit}>
      <div className="gd-composer-inner">
        <textarea
          ref={textareaRef}
          placeholder={placeholder}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKey}
          rows={1}
          disabled={pending}
        />
        <button
          type="submit"
          className="gd-composer-send"
          disabled={pending || !value.trim()}
          aria-label="Send"
        >
          <Icon name="arrowUp" size={14} />
        </button>
      </div>
    </form>
  );
};

// ─── Screen ──────────────────────────────────────────────────────

const PLACEHOLDERS: Record<OnboardingPhase, string> = {
  intent: 'Tell Claude what your agent should do…',
  model: 'Reply or pick from the card above.',
  channel: "Pick a card above or just type — e.g. \"Slack\"",
  connect: 'Paste the token above, or reply with anything to add to the prompt…',
  live: 'Ask Claude anything — e.g. "show me yesterday\'s tone issues"',
  done: 'Onboarding complete — head to the dashboard.',
};

const STARTERS: Array<{ label: string; message: string }> = [
  { label: 'Customer support', message: 'I want a customer support agent for my Shopify store — answer order and refund questions.' },
  { label: 'Sales outreach', message: 'I want a sales outreach agent that qualifies inbound leads for our SaaS.' },
  { label: 'Internal helpdesk', message: 'I want an internal helpdesk agent for IT and HR questions inside our company.' },
  { label: 'Research assistant', message: 'I want a research assistant that summarises long PDFs and answers questions about them.' },
  { label: 'Refund flow', message: 'I want a refund-flow agent that handles refund requests and escalates billing issues.' },
];

// Phase → progress + sidebar/panel visibility.
const visibilityFor = (
  phase: OnboardingPhase,
  hasUserMsg: boolean,
  slackConnected: boolean,
): {
  showWelcome: boolean;
  sidebarState: SidebarState;
  panelStage: PanelStage | 'empty';
  progress: number;
} => {
  if (!hasUserMsg && phase === 'intent') {
    return { showWelcome: true, sidebarState: 'empty', panelStage: 'empty', progress: 0 };
  }
  if (phase === 'intent') {
    return { showWelcome: false, sidebarState: 'empty', panelStage: 'initial', progress: 15 };
  }
  if (phase === 'model') {
    return { showWelcome: false, sidebarState: 'draft', panelStage: 'initial', progress: 45 };
  }
  if (phase === 'channel') {
    return { showWelcome: false, sidebarState: 'draft', panelStage: 'channel', progress: 60 };
  }
  if (phase === 'connect') {
    return {
      showWelcome: false,
      sidebarState: 'partial',
      panelStage: slackConnected ? 'connected' : 'channel',
      progress: slackConnected ? 85 : 70,
    };
  }
  // live / done
  return { showWelcome: false, sidebarState: 'ready', panelStage: 'full', progress: 100 };
};

const channelLabel = (id: string | undefined): ReactNode => {
  if (id === 'slack') return (<><SlackGlyph size={11} /> Slack</>);
  if (id === 'whatsapp') return (<><Icon name="phone" size={11} /> WhatsApp</>);
  if (id === 'web') return (<><Icon name="globe" size={11} /> Web</>);
  return id ?? null;
};

const modelLabel = (id: string | undefined): string | null => {
  if (!id) return null;
  if (id.includes('haiku')) return 'Haiku 4';
  if (id.includes('sonnet')) return 'Sonnet 4';
  if (id.includes('opus')) return 'Opus 4';
  return id;
};

export const Onboarding = ({ onDone }: OnboardingProps) => {
  const [session, setSession] = useState<OnboardingV2Session | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [plaintextKey, setPlaintextKey] = useState<string | null>(null);
  const [slackInstalled, setSlackInstalled] = useState(false);
  const threadEndRef = useRef<HTMLDivElement | null>(null);
  const prevDecisionsRef = useRef<string>('');
  const [updatedSection, setUpdatedSection] = useState<'behavior' | 'tone' | 'channel' | 'model' | null>(null);
  // What the user just clicked on a picker card. Cleared the moment the
  // server's reply arrives (the picker is replaced by a chip anyway) —
  // this only exists to give the click instant visual feedback.
  const [pendingPick, setPendingPick] = useState<{ key: 'model' | 'channel'; value: string } | null>(null);

  // Cold load.
  useEffect(() => {
    let cancelled = false;
    void getOnboardingV2Session()
      .then((s) => {
        if (cancelled) return;
        setSession(s);
        if (s.agent_key_plaintext_once) setPlaintextKey(s.agent_key_plaintext_once);
        setSlackInstalled(Boolean(s.slack?.installed));
      })
      .catch((e) => {
        if (!cancelled) setError(String(e?.message ?? e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Autoscroll on new turn.
  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [session?.turns.length]);

  // Poll Slack status while connecting.
  useEffect(() => {
    if (!session || session.phase !== 'connect') return;
    if (slackInstalled) return;
    const t = window.setInterval(async () => {
      try {
        const status = await getSlackConnectStatus();
        if (status.installed) setSlackInstalled(true);
      } catch {
        /* network blip — retry next tick */
      }
    }, 3000);
    return () => window.clearInterval(t);
  }, [session?.phase, slackInstalled, session]);

  // Watch decisions/phase changes → drive the "just updated" pulse on the
  // right panel. Compare the latest decisions snapshot to the previous one;
  // the first key that differs is the section we highlight.
  useEffect(() => {
    if (!session) return;
    const next = JSON.stringify({
      phase: session.phase,
      decisions: session.decisions,
      slack: slackInstalled,
    });
    if (prevDecisionsRef.current && prevDecisionsRef.current !== next) {
      const prev = JSON.parse(prevDecisionsRef.current);
      const d = session.decisions as Record<string, unknown>;
      const pd = prev.decisions as Record<string, unknown>;
      let section: typeof updatedSection = null;
      if (d.purpose !== pd.purpose) section = 'behavior';
      else if (d.tone !== pd.tone) section = 'tone';
      else if (d.channel !== pd.channel || slackInstalled !== prev.slack) section = 'channel';
      else if (d.model !== pd.model) section = 'model';
      else if (session.phase !== prev.phase) {
        section = session.phase === 'channel' ? 'tone' : session.phase === 'connect' ? 'channel' : section;
      }
      if (section) {
        setUpdatedSection(section);
        const t = window.setTimeout(() => setUpdatedSection(null), 2400);
        return () => window.clearTimeout(t);
      }
    }
    prevDecisionsRef.current = next;
  }, [session, slackInstalled]);

  const applySession = (s: OnboardingV2Session) => {
    setSession(s);
    if (s.agent_key_plaintext_once) setPlaintextKey(s.agent_key_plaintext_once);
    setSlackInstalled((prev) => prev || Boolean(s.slack?.installed));
  };

  const handleSay = async (text: string) => {
    setPending(true);
    setError(null);
    try {
      const next = await sayOnboardingV2(text);
      applySession(next);
    } catch (e) {
      setError(String((e as Error)?.message ?? e));
    } finally {
      setPending(false);
    }
  };

  const handlePick = async (key: 'model' | 'channel', value: string) => {
    setPending(true);
    setError(null);
    setPendingPick({ key, value });
    try {
      const next = await decideOnboardingV2(key, value);
      applySession(next);
    } catch (e) {
      setError(String((e as Error)?.message ?? e));
    } finally {
      setPending(false);
      setPendingPick(null);
    }
  };

  const handleSkip = async () => {
    setPending(true);
    setError(null);
    try {
      const state = await skipOnboarding();
      onDone(state);
    } catch (e) {
      setError(String((e as Error)?.message ?? e));
      setPending(false);
    }
  };

  const handleEdit = async (key: 'model' | 'channel') => {
    setPending(true);
    setError(null);
    try {
      const next = await rewindOnboardingV2(key);
      if (key === 'model') setPlaintextKey(null);
      applySession(next);
    } catch (e) {
      setError(String((e as Error)?.message ?? e));
    } finally {
      setPending(false);
    }
  };

  const finish = async () => {
    if (!session) return;
    setPending(true);
    setError(null);
    try {
      const decisions = session.decisions as Record<string, string>;
      const next = await completeOnboarding({
        template: null,
        name: session.agent_id ?? '',
        company: '',
        prompt: decisions.purpose ?? '',
        model: decisions.model ?? 'claude-sonnet-4-6',
        tools: [],
        channels: decisions.channel ? [decisions.channel] : [],
      });
      onDone(next);
    } catch (e) {
      setError(String((e as Error)?.message ?? e));
    } finally {
      setPending(false);
    }
  };

  // Derived state.
  const hasUserMsg = useMemo(
    () => Boolean(session?.turns.some((t) => t.role === 'user')),
    [session?.turns],
  );

  if (error && !session) {
    return (
      <div className="gd-wrap" style={{ '--sidebar-w': '0px', '--panel-w': '0px' } as CSSProperties}>
        <div />
        <main className="gd-main">
          <div className="gd-thread">
            <div className="gd-thread-inner">
              <div className="gd-msg">
                <div className="gd-msg-body">
                  Could not load the onboarding session. <strong>{error}</strong>
                </div>
              </div>
            </div>
          </div>
        </main>
        <div />
      </div>
    );
  }

  if (!session) {
    return (
      <div className="gd-wrap" style={{ '--sidebar-w': '0px', '--panel-w': '0px' } as CSSProperties}>
        <div />
        <main className="gd-main">
          <div className="gd-thread">
            <div className="gd-thread-inner">
              <div className="gd-msg">
                <div className="gd-msg-body">
                  <span className="gd-typing"><span /><span /><span /></span>
                </div>
              </div>
            </div>
          </div>
        </main>
        <div />
      </div>
    );
  }

  const decisions = (session.decisions ?? {}) as Record<string, string>;
  const { showWelcome, sidebarState, panelStage, progress } = visibilityFor(
    session.phase,
    hasUserMsg,
    slackInstalled,
  );

  // ─── Welcome (G1) ─────────────────────────────────────────────
  if (showWelcome) {
    return (
      <div className="gd-wrap" style={{ '--sidebar-w': '0px', '--panel-w': '0px' } as CSSProperties}>
        <div />
        <main className="gd-main">
          <div className="gd-welcome">
            <div className="gd-welcome-brand">
              <span className="mark" />
              <span>OpenTracy <span className="dim">Evolution</span></span>
            </div>
            <div className="gd-welcome-stage">
              <h1 className="gd-welcome-h">
                Let's create your first <RotatingWord />.
              </h1>
              <p className="gd-welcome-sub">
                Describe what it should do — I'll handle the model, prompt and channel. You can tweak everything as we go.
              </p>
              <WelcomeComposer onSubmit={handleSay} pending={pending} />
              <div className="gd-welcome-starters">
                <div className="gd-welcome-starters-label">Or start from a template</div>
                {STARTERS.map((s) => (
                  <button
                    key={s.label}
                    type="button"
                    className="gd-welcome-starter"
                    onClick={() => handleSay(s.message)}
                    disabled={pending}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
              {error && (
                <div className="gd-error" style={{ marginTop: 18, marginLeft: 0 }}>
                  <Icon name="warn" size={12} /> {error}
                </div>
              )}
              <button
                type="button"
                className="gd-skip"
                onClick={handleSkip}
                disabled={pending}
              >
                Skip onboarding — I'll set up later
              </button>
            </div>
          </div>
        </main>
        <div />
      </div>
    );
  }

  // ─── Conversation (G2–G6) ─────────────────────────────────────
  const sidebarW = sidebarState === 'empty' ? '0px' : '240px';
  const panelW = panelStage === 'empty' ? '0px' : '340px';

  const liveStatus: 'waiting' | 'connected' = slackInstalled ? 'connected' : 'waiting';
  const showLiveCta = session.phase === 'live';

  const agentName = session.agent_id || (decisions.purpose ? 'checkout-support' : 'Untitled agent');
  const channelNode = decisions.channel ? channelLabel(decisions.channel) : null;
  const modelStr = modelLabel(decisions.model);
  const purposeShort = decisions.purpose
    ? decisions.purpose.length > 110 ? `${decisions.purpose.slice(0, 107)}…` : decisions.purpose
    : '';
  const toneStr = decisions.tone ?? null;

  return (
    <div
      className="gd-wrap"
      style={{ '--sidebar-w': sidebarW, '--panel-w': panelW } as CSSProperties}
    >
      {sidebarState !== 'empty' ? (
        <Sidebar
          state={sidebarState}
          progress={progress}
          agentName={agentName}
          channel={channelNode}
          model={modelStr}
          toolCount={3}
        />
      ) : (
        <div />
      )}

      <main className="gd-main">
        <div className="gd-main-toolbar">
          <button
            type="button"
            className="gd-toolbar-link"
            onClick={handleSkip}
            disabled={pending}
            title="Skip onboarding and go straight to the dashboard"
          >
            Skip onboarding
          </button>
        </div>
        <div className="gd-thread">
          <div className="gd-thread-inner">
            {session.turns.map((t) =>
              t.role === 'assistant' ? (
                <AssistantTurn
                  key={t.id}
                  turn={t}
                  onPick={handlePick}
                  pending={pending}
                  plaintextKey={plaintextKey}
                  liveStatus={liveStatus}
                  pendingPick={pendingPick}
                />
              ) : (
                <UserTurn key={t.id} turn={t} onEdit={handleEdit} disabled={pending} />
              ),
            )}
            {pending && (
              <div className="gd-msg claude">
                <div className="gd-msg-meta">
                  <div className="av c">C</div>
                  <span className="name">Claude</span>
                </div>
                <div className="gd-msg-body">
                  <span className="gd-typing"><span /><span /><span /></span>
                </div>
              </div>
            )}
            {showLiveCta && (
              <div style={{ display: 'flex', justifyContent: 'center', paddingTop: 8 }}>
                <button
                  type="button"
                  className="gd-btn-primary"
                  onClick={finish}
                  disabled={pending}
                >
                  Go to dashboard
                  <Icon name="arrowUp" size={12} />
                </button>
              </div>
            )}
            {error && (
              <div className="gd-error">
                <Icon name="warn" size={12} /> {error}
              </div>
            )}
            <div ref={threadEndRef} />
          </div>
        </div>
        <ChatComposer
          onSubmit={handleSay}
          pending={pending}
          placeholder={PLACEHOLDERS[session.phase] ?? 'Type a message…'}
        />
      </main>

      {panelStage !== 'empty' ? (
        <Panel
          stage={panelStage}
          agentName={agentName}
          purpose={purposeShort}
          tone={toneStr}
          channel={
            decisions.channel === 'slack'
              ? (<><SlackGlyph size={12} /> Slack · acme-co</>)
              : channelNode
          }
          model={modelStr ? `Claude ${modelStr}` : null}
          updatedSection={updatedSection}
        />
      ) : (
        <PanelEmpty />
      )}
    </div>
  );
};
