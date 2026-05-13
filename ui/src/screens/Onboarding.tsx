/**
 * P1.12 — Conversational onboarding (Claude-led, Direction A).
 *
 * 4 stages: Welcome → Chat → Confirm → Launching.
 *
 * Translated from claude-design/screens/Onboarding.jsx (chat variant) —
 * markup + class names preserved so the design CSS in styles.css
 * renders the same shapes. Wired to the real backend:
 *
 *   - POST /v1/onboarding/turn → drives the chat; returns reply +
 *     materialized config + justAdded badge + ready flag. Server
 *     falls back to a scripted 5-turn flow when no ANTHROPIC_API_KEY,
 *     so the UI never deadlocks.
 *   - POST /v1/onboarding/complete → persists the confirmed config
 *     (writes agent/prompts/system.md, records an agent_created Lesson).
 *   - POST /v1/onboarding/skip → escape hatch from header.
 */

import { useEffect, useRef, useState } from 'react';

import {
  completeOnboarding,
  onboardingTurn,
  skipOnboarding,
  type OnboardingChatMessage,
  type OnboardingJustAdded,
  type OnboardingState,
  type OnboardingTurnConfig,
} from '../api';
import { Icon } from '../components/Icon';

interface Model {
  id: string;
  name: string;
  desc: string;
  recommended?: boolean;
}

const MODELS: Model[] = [
  { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5', desc: 'Fastest and cheapest. Good for high volume.' },
  { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6', desc: 'Smart, fast default for most agents.', recommended: true },
  { id: 'claude-opus-4-7', name: 'Claude Opus 4.7', desc: 'Strongest reasoning. Use when accuracy matters.' },
];

const CHANNEL_LABELS: Record<string, string> = {
  web: 'Web chat widget',
  whatsapp: 'WhatsApp',
  slack: 'Slack',
  email: 'Email',
  api: 'API only',
};

const STARTERS = [
  {
    title: 'Customer support',
    hint: 'Refunds, order lookups, ticket handoff.',
    seed:
      'A customer support agent for our Shopify store. Needs to handle order lookups, refund questions, and hand off to a human on anything billing-related.',
  },
  {
    title: 'SDR / sales',
    hint: 'Qualify inbound leads, book meetings.',
    seed:
      'An SDR that qualifies inbound demo requests with BANT, then books a meeting on my calendar. Should never pitch — just qualify.',
  },
  {
    title: 'Research assistant',
    hint: 'Search, cite sources, distinguish fact from inference.',
    seed:
      'A research assistant that summarizes papers and articles I drop in. Must cite sources and flag uncertainty.',
  },
  {
    title: 'Internal helpdesk',
    hint: 'Slack bot for IT/HR questions.',
    seed:
      "An internal helpdesk bot for our Slack workspace. Should answer IT and HR questions from our notion docs and open a ticket if it can't.",
  },
];

const DEFAULT_CONFIG: OnboardingTurnConfig = {
  name: '',
  model: 'claude-sonnet-4-6',
  prompt: '',
  tools: [],
  channels: [],
};

type Stage = 'welcome' | 'chat' | 'confirm' | 'launching';

interface ChatMsg {
  role: 'user' | 'assistant';
  text: string;
  justAdded?: OnboardingJustAdded | null;
}

interface OnboardingProps {
  onDone: (next: OnboardingState | null) => void;
}

export const Onboarding = ({ onDone }: OnboardingProps) => {
  const [stage, setStage] = useState<Stage>('welcome');
  const [config, setConfig] = useState<OnboardingTurnConfig>(DEFAULT_CONFIG);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [thinking, setThinking] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const callTurn = async (history: ChatMsg[]) => {
    setThinking(true);
    setError(null);
    try {
      const apiMessages: OnboardingChatMessage[] = history.map((m) => ({
        role: m.role,
        content: m.text,
      }));
      const out = await onboardingTurn(apiMessages);
      setConfig((c) => ({ ...c, ...out.config }));
      setMessages((m) => [
        ...m,
        { role: 'assistant', text: out.reply, justAdded: out.justAdded ?? null },
      ]);
      if (out.ready) setReady(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setThinking(false);
    }
  };

  const handleFirstSend = async (text: string) => {
    setStage('chat');
    const next: ChatMsg[] = [{ role: 'user', text }];
    setMessages(next);
    await callTurn(next);
  };

  const handleReply = async (text: string) => {
    const next = [...messages, { role: 'user' as const, text }];
    setMessages(next);
    await callTurn(next);
  };

  const startConfirm = () => setStage('confirm');

  const launch = async () => {
    setStage('launching');
    try {
      const next = await completeOnboarding({
        template: null,
        name: config.name,
        company: '',
        prompt: config.prompt,
        model: config.model,
        tools: config.tools,
        channels: config.channels,
      });
      await new Promise((r) => setTimeout(r, 1800));
      onDone(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStage('confirm');
    }
  };

  const skip = async () => {
    try {
      const next = await skipOnboarding();
      onDone(next);
    } catch {
      onDone(null);
    }
  };

  if (stage === 'launching') return <Launching name={config.name} />;

  return (
    <div className="onbA">
      <Header
        stage={stage}
        onSkip={skip}
        onBack={stage === 'confirm' ? () => setStage('chat') : null}
      />
      {stage === 'welcome' && <Welcome onSend={handleFirstSend} />}
      {stage === 'chat' && (
        <ChatStage
          messages={messages}
          thinking={thinking}
          config={config}
          ready={ready}
          error={error}
          onReply={handleReply}
          onConfirm={startConfirm}
        />
      )}
      {stage === 'confirm' && (
        <Confirm
          config={config}
          onChangeConfig={setConfig}
          onAsk={(text) => {
            setStage('chat');
            setReady(false);
            void handleReply(text);
          }}
          onLaunch={() => void launch()}
          error={error}
        />
      )}
    </div>
  );
};

// ─── Shared chrome ────────────────────────────────────────────────────────
const Header = ({
  stage,
  onSkip,
  onBack,
}: {
  stage: Stage;
  onSkip: () => void;
  onBack: (() => void) | null;
}) => (
  <header className="onbA-head">
    <div className="onbA-brand">
      <div className="sidebar-mark" />
      <span>
        OpenTracy <span className="dim">Evolution</span>
      </span>
    </div>
    <div className="onbA-head-right">
      {stage === 'chat' && (
        <span className="onbA-stagetag">
          <span className="onbA-stagetag-dot" /> Building agent
        </span>
      )}
      {onBack && (
        <button className="btn ghost sm" onClick={onBack}>
          ← Back to chat
        </button>
      )}
      <button className="btn ghost sm" onClick={onSkip}>
        Skip and use the wizard
      </button>
    </div>
  </header>
);

const ClaudeAvatar = ({ size = 28, thinking = false }: { size?: number; thinking?: boolean }) => (
  <div
    className={`onbA-avatar onbA-avatar-c ${thinking ? 'onbA-avatar-thinking' : ''}`}
    style={{ width: size, height: size, fontSize: size * 0.45 }}
  >
    C
  </div>
);

const UserAvatar = ({ size = 28, label = 'You' }: { size?: number; label?: string }) => (
  <div
    className="onbA-avatar onbA-avatar-u"
    style={{ width: size, height: size, fontSize: size * 0.36 }}
  >
    {label.slice(0, 2).toUpperCase()}
  </div>
);

// ─── Welcome ──────────────────────────────────────────────────────────────
const Welcome = ({ onSend }: { onSend: (text: string) => void }) => {
  const [text, setText] = useState('');
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    taRef.current?.focus();
  }, []);

  const submit = () => {
    const t = text.trim();
    if (t.length < 8) return;
    onSend(t);
  };

  return (
    <div className="onbA-welcome">
      <div className="onbA-welcome-inner">
        <div className="onbA-welcome-greet">
          <ClaudeAvatar size={52} />
          <h1 className="onbA-h1">Let's build your first agent.</h1>
          <p className="onbA-sub">
            Tell me what it should do and who it's for. I'll set up the prompt, model, tools,
            and channels — you can review and tweak everything before launch.
          </p>
        </div>

        <div className="onbA-composer">
          <textarea
            ref={taRef}
            className="onbA-composer-input"
            rows={3}
            placeholder="e.g. A support agent for our Shopify checkout. Needs order lookups, refund policy, hand off to a human on billing."
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submit();
            }}
          />
          <div className="onbA-composer-foot">
            <div className="onbA-composer-attach">
              <button className="onbA-pill" disabled>
                <Icon name="file" size={12} /> Paste docs
              </button>
              <button className="onbA-pill" disabled>
                <Icon name="code" size={12} /> Connect repo
              </button>
            </div>
            <button
              className="onbA-send"
              onClick={submit}
              disabled={text.trim().length < 8}
              title="Send (⌘↵)"
            >
              <Icon name="arrowUp" size={14} />
            </button>
          </div>
        </div>

        <div className="onbA-starters">
          <div className="onbA-starters-label">Or start from a recipe</div>
          <div className="onbA-starters-grid">
            {STARTERS.map((s) => (
              <button
                key={s.title}
                className="onbA-starter"
                onClick={() => onSend(s.seed)}
              >
                <div className="onbA-starter-title">{s.title}</div>
                <div className="onbA-starter-hint dim">{s.hint}</div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// ─── Chat stage (split: messages + live config) ─────────────────────────
const ChatStage = ({
  messages,
  thinking,
  config,
  ready,
  error,
  onReply,
  onConfirm,
}: {
  messages: ChatMsg[];
  thinking: boolean;
  config: OnboardingTurnConfig;
  ready: boolean;
  error: string | null;
  onReply: (text: string) => void;
  onConfirm: () => void;
}) => {
  const [draft, setDraft] = useState('');
  const taRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    taRef.current?.focus();
  }, [thinking]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, thinking]);

  const submit = () => {
    const t = draft.trim();
    if (!t || thinking) return;
    onReply(t);
    setDraft('');
  };

  return (
    <div className="onbA-split">
      <section className="onbA-chat-pane">
        <div className="onbA-chat-scroll" ref={scrollRef}>
          {messages.map((m, i) => (
            <Message key={i} role={m.role} text={m.text} justAdded={m.justAdded ?? undefined} />
          ))}
          {thinking && (
            <div className="onbA-msg onbA-msg-c">
              <ClaudeAvatar thinking />
              <div className="onbA-bubble onbA-bubble-thinking">
                <span className="onbA-typing">
                  <span />
                  <span />
                  <span />
                </span>
              </div>
            </div>
          )}
          {error && !thinking && (
            <div className="onbA-msg onbA-msg-c">
              <ClaudeAvatar />
              <div
                className="onbA-bubble"
                style={{ background: 'var(--bad-soft)', color: 'var(--bad-fg)', border: '1px solid var(--bad-fg)' }}
              >
                Sorry — something went wrong sending that. Try again? <br />
                <span className="dim" style={{ fontSize: 11 }}>{error}</span>
              </div>
            </div>
          )}
        </div>

        {ready && !thinking && (
          <div className="onbA-readybar">
            <div>
              <div className="onbA-readybar-title">Ready when you are.</div>
              <div className="onbA-readybar-sub dim">
                Or keep chatting to refine — I'll update the config live.
              </div>
            </div>
            <button className="btn primary" onClick={onConfirm}>
              Review and launch <Icon name="chevron" size={13} />
            </button>
          </div>
        )}

        <div className="onbA-replybar">
          <textarea
            ref={taRef}
            className="onbA-reply-input"
            rows={1}
            placeholder={thinking ? 'Claude is thinking…' : 'Reply…'}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            disabled={thinking}
          />
          <button
            className="onbA-send onbA-send-sm"
            onClick={submit}
            disabled={!draft.trim() || thinking}
          >
            <Icon name="arrowUp" size={13} />
          </button>
        </div>
      </section>

      <aside className="onbA-config-pane">
        <LiveConfig config={config} />
      </aside>
    </div>
  );
};

const Message = ({
  role,
  text,
  justAdded,
}: {
  role: 'user' | 'assistant';
  text: string;
  justAdded?: OnboardingJustAdded;
}) => {
  if (role === 'user') {
    return (
      <div className="onbA-msg onbA-msg-u">
        <div className="onbA-bubble onbA-bubble-u">{text}</div>
        <UserAvatar />
      </div>
    );
  }
  return (
    <div className="onbA-msg onbA-msg-c">
      <ClaudeAvatar />
      <div className="onbA-msg-body">
        <div className="onbA-bubble">{text}</div>
        {justAdded && <JustAdded {...justAdded} />}
      </div>
    </div>
  );
};

const JustAdded = ({
  tool,
  model,
  channel,
}: {
  tool?: string | null;
  model?: string | null;
  channel?: string | null;
}) => (
  <div className="onbA-justadded">
    {tool && (
      <span className="onbA-justadded-row">
        <span className="dim">added tool</span>
        <span className="onbA-mono-chip">{tool}</span>
      </span>
    )}
    {model && (
      <span className="onbA-justadded-row">
        <span className="dim">switched model</span>
        <span className="onbA-chip-soft">{MODELS.find((m) => m.id === model)?.name || model}</span>
      </span>
    )}
    {channel && (
      <span className="onbA-justadded-row">
        <span className="dim">added channel</span>
        <span className="onbA-chip">{CHANNEL_LABELS[channel] || channel}</span>
      </span>
    )}
  </div>
);

// ─── Live config card (right pane) ──────────────────────────────────────
const LiveConfig = ({ config }: { config: OnboardingTurnConfig }) => {
  const model = MODELS.find((m) => m.id === config.model);
  const completeness = computeCompleteness(config);

  return (
    <div className="onbA-lc">
      <div className="onbA-lc-head">
        <div className="onbA-lc-label">BUILDING</div>
        <div className="onbA-lc-pct">{completeness}%</div>
      </div>
      <div className="onbA-lc-progress">
        <span style={{ width: `${completeness}%` }} />
      </div>

      <div className="onbA-lc-row">
        <div className="onbA-lc-k">name</div>
        <div className="onbA-lc-v mono">
          {config.name || <em className="onbA-empty">unnamed-agent</em>}
        </div>
      </div>
      <div className="onbA-lc-row">
        <div className="onbA-lc-k">model</div>
        <div className="onbA-lc-v">
          {model ? (
            <span className="onbA-chip-soft">{model.name}</span>
          ) : (
            <em className="onbA-empty">—</em>
          )}
        </div>
      </div>

      <div className="onbA-lc-row onbA-lc-row-block">
        <div className="onbA-lc-k">prompt</div>
        {config.prompt ? (
          <div className="onbA-lc-prompt">{config.prompt}</div>
        ) : (
          <div className="onbA-lc-empty">
            <em>Empty — Claude will fill this as you chat.</em>
          </div>
        )}
      </div>

      <div className="onbA-lc-row onbA-lc-row-block">
        <div className="onbA-lc-k">tools · {config.tools.length}</div>
        {config.tools.length === 0 ? (
          <div className="onbA-lc-empty">
            <em>None yet.</em>
          </div>
        ) : (
          <div className="onbA-lc-chips">
            {config.tools.map((t) => (
              <span key={t} className="onbA-mono-chip">
                {t}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="onbA-lc-row onbA-lc-row-block">
        <div className="onbA-lc-k">channels · {config.channels.length}</div>
        {config.channels.length === 0 ? (
          <div className="onbA-lc-empty">
            <em>Not decided yet.</em>
          </div>
        ) : (
          <div className="onbA-lc-chips">
            {config.channels.map((c) => (
              <span key={c} className="onbA-chip">
                {CHANNEL_LABELS[c] || c}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="onbA-lc-foot">
        <Icon name="info" size={12} />
        <span>Everything here is editable in the next step — and changeable forever after.</span>
      </div>
    </div>
  );
};

function computeCompleteness(c: OnboardingTurnConfig): number {
  let n = 0;
  if (c.name) n += 20;
  if (c.model) n += 15;
  if (c.prompt && c.prompt.length > 40) n += 30;
  if (c.tools.length > 0) n += 15;
  if (c.channels.length > 0) n += 20;
  return Math.min(100, n);
}

// ─── Confirm ────────────────────────────────────────────────────────────
const Confirm = ({
  config,
  onChangeConfig,
  onAsk,
  onLaunch,
  error,
}: {
  config: OnboardingTurnConfig;
  onChangeConfig: (c: OnboardingTurnConfig) => void;
  onAsk: (text: string) => void;
  onLaunch: () => void;
  error: string | null;
}) => {
  const [askText, setAskText] = useState('');
  const [editingPrompt, setEditingPrompt] = useState(false);
  const [promptDraft, setPromptDraft] = useState(config.prompt);
  const model = MODELS.find((m) => m.id === config.model) || MODELS[1];

  const sendAsk = () => {
    const t = askText.trim();
    if (!t) return;
    onAsk(t);
  };

  const savePrompt = () => {
    onChangeConfig({ ...config, prompt: promptDraft });
    setEditingPrompt(false);
  };

  return (
    <div className="onbA-confirm">
      <div className="onbA-confirm-inner">
        <div className="onbA-confirm-head">
          <ClaudeAvatar size={32} />
          <div>
            <h2 className="onbA-confirm-h">Here's what I built. Look right?</h2>
            <p className="onbA-confirm-sub dim">
              Anything you can click below, you can edit. Or ask me to change something.
            </p>
          </div>
        </div>

        <div className="onbA-spec">
          <div className="onbA-spec-top">
            <div>
              <span className="onbA-spec-name mono">{config.name || 'unnamed-agent'}</span>
              <span className="onbA-spec-ver dim"> v0.1 · draft</span>
            </div>
            <button className="btn ghost sm" disabled>
              Rename
            </button>
          </div>

          <div className="onbA-spec-row">
            <div className="onbA-spec-k">model</div>
            <div className="onbA-spec-v">
              <span className="onbA-chip-soft">{model.name}</span>
              <span className="dim" style={{ marginLeft: 8, fontSize: 12 }}>
                fallback Sonnet · router learns later
              </span>
            </div>
          </div>

          <div className="onbA-spec-row">
            <div className="onbA-spec-k">tools</div>
            <div className="onbA-spec-v onbA-spec-chips">
              {config.tools.length === 0 ? (
                <em className="dim">None</em>
              ) : (
                config.tools.map((t) => (
                  <span key={t} className="onbA-mono-chip">
                    {t}
                  </span>
                ))
              )}
              <button className="onbA-add-chip" disabled>
                + tool
              </button>
            </div>
          </div>

          <div className="onbA-spec-row">
            <div className="onbA-spec-k">channels</div>
            <div className="onbA-spec-v onbA-spec-chips">
              {config.channels.length === 0 ? (
                <em className="dim">None</em>
              ) : (
                config.channels.map((c) => (
                  <span key={c} className="onbA-chip">
                    {CHANNEL_LABELS[c] || c}
                  </span>
                ))
              )}
              <button className="onbA-add-chip" disabled>
                + channel
              </button>
            </div>
          </div>

          <div className="onbA-spec-row onbA-spec-row-block">
            <div className="onbA-spec-k-row">
              <div className="onbA-spec-k">prompt</div>
              {!editingPrompt && (
                <button
                  className="btn ghost sm"
                  onClick={() => {
                    setPromptDraft(config.prompt);
                    setEditingPrompt(true);
                  }}
                >
                  Edit
                </button>
              )}
              {editingPrompt && (
                <div style={{ display: 'flex', gap: 6 }}>
                  <button className="btn ghost sm" onClick={() => setEditingPrompt(false)}>
                    Cancel
                  </button>
                  <button className="btn sm" onClick={savePrompt}>
                    Save
                  </button>
                </div>
              )}
            </div>
            {editingPrompt ? (
              <textarea
                className="onbA-spec-prompt-edit"
                value={promptDraft}
                onChange={(e) => setPromptDraft(e.target.value)}
                rows={8}
              />
            ) : (
              <pre className="onbA-spec-prompt">
                {config.prompt || <em className="dim">No prompt yet.</em>}
              </pre>
            )}
          </div>
        </div>

        {error && (
          <div
            className="onbA-confirm-caveat"
            style={{ color: 'var(--bad-fg)' }}
          >
            <Icon name="info" size={13} />
            <span>Couldn't save: {error}</span>
          </div>
        )}

        <div className="onbA-confirm-actions">
          <div className="onbA-ask">
            <ClaudeAvatar size={20} />
            <input
              className="onbA-ask-input"
              placeholder="Ask Claude to change something…"
              value={askText}
              onChange={(e) => setAskText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') sendAsk();
              }}
            />
            <button
              className="onbA-send onbA-send-xs"
              onClick={sendAsk}
              disabled={!askText.trim()}
            >
              <Icon name="arrowUp" size={11} />
            </button>
          </div>
          <button className="btn primary lg" onClick={onLaunch}>
            Launch agent <Icon name="chevron" size={14} />
          </button>
        </div>

        <div className="onbA-confirm-caveat">
          <Icon name="info" size={13} />
          <span>The first conversation will be traced. You'll see suggestions to improve once a few have run.</span>
        </div>
      </div>
    </div>
  );
};

// ─── Launching animation ────────────────────────────────────────────────
const LAUNCH_STEPS = [
  'Compiling agent…',
  'Wiring tools…',
  'Connecting channels…',
  'Watching for the first conversation.',
];

const Launching = ({ name }: { name: string }) => {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const t = setInterval(
      () => setStep((s) => Math.min(s + 1, LAUNCH_STEPS.length - 1)),
      520,
    );
    return () => clearInterval(t);
  }, []);
  return (
    <div className="onbA onbA-launching">
      <div className="onbA-launch-inner">
        <div className="onbA-launch-mark" />
        <div className="onbA-launch-name mono">{name || 'unnamed-agent'}</div>
        <ul className="onbA-launch-list">
          {LAUNCH_STEPS.map((s, i) => (
            <li key={i} className={i < step ? 'done' : i === step ? 'on' : 'idle'}>
              {i < step ? (
                <Icon name="check" size={12} />
              ) : i === step ? (
                <span className="onbA-launch-dot" />
              ) : (
                <span className="onbA-launch-tick" />
              )}
              <span>{s}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};
