/**
 * P1.11 — Day-0 onboarding (full-viewport wizard).
 *
 * 4 steps: Identity → Brain → Hands → Channels → Launch animation.
 *
 * Translated from claude-design/screens/Onboarding.jsx 1:1 in markup
 * structure + class names so the design CSS in styles.css renders the
 * same shapes. Behavior added on top:
 *   - launch() POSTs /v1/onboarding/complete and waits for the real
 *     state from the backend instead of setTimeout-ing for 2.4s.
 *   - Skip POSTs /v1/onboarding/skip so the rest of the UI knows to
 *     stop showing onboarding without the operator finishing it.
 */

import { useEffect, useState } from 'react';

import {
  completeOnboarding,
  skipOnboarding,
  type OnboardingCompleteRequest,
  type OnboardingState,
} from '../api';
import { Icon, type IconName } from '../components/Icon';

interface Template {
  id: string;
  name: string;
  desc: string;
  icon: IconName;
  prompt: string;
  tools: string[];
  channels: string[];
}

const TEMPLATES: Template[] = [
  {
    id: 'support',
    name: 'Customer support',
    desc: 'Answer questions, look up orders, hand off when needed.',
    icon: 'chat',
    prompt:
      "You are a customer support agent for {{company}}.\nBe warm, direct, and honest. Acknowledge frustration once, then move to action.\nUse tools to look up real data — never guess. If you can't help, offer to hand off to a human.",
    tools: ['lookup_order', 'search_kb', 'create_ticket'],
    channels: ['web', 'whatsapp'],
  },
  {
    id: 'sales',
    name: 'Sales / SDR',
    desc: 'Qualify leads, schedule meetings, follow up with context.',
    icon: 'bolt',
    prompt:
      "You are an SDR for {{company}}.\nYour job is to qualify leads using BANT, not to pitch.\nAsk one question at a time. Book a meeting when the lead is qualified. Be concise.",
    tools: ['lookup_account', 'check_calendar', 'book_meeting'],
    channels: ['email', 'web'],
  },
  {
    id: 'research',
    name: 'Research assistant',
    desc: 'Search, summarize, cite sources, surface what matters.',
    icon: 'flask',
    prompt:
      "You are a research assistant.\nAlways cite sources. Distinguish between what you know and what you've looked up.\nFlag uncertainty explicitly. Prefer recent, primary sources.",
    tools: ['web_search', 'fetch_url', 'save_note'],
    channels: ['web', 'slack'],
  },
  {
    id: 'blank',
    name: 'Start blank',
    desc: 'Describe your own agent. Best when you know what you want.',
    icon: 'sparkles',
    prompt: '',
    tools: [],
    channels: ['web'],
  },
];

interface Model {
  id: string;
  name: string;
  desc: string;
  recommended?: boolean;
}

const MODELS: Model[] = [
  { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6', desc: 'Smart, fast, good default for most agents', recommended: true },
  { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5', desc: 'Fastest and cheapest. Good for high volume.' },
  { id: 'claude-opus-4-7', name: 'Claude Opus 4.7', desc: 'Strongest reasoning. Use when accuracy matters most.' },
];

interface Channel {
  id: string;
  name: string;
  desc: string;
  icon: IconName;
}

const CHANNELS: Channel[] = [
  { id: 'web', name: 'Web chat widget', desc: 'Embed in your site.', icon: 'chat' },
  { id: 'whatsapp', name: 'WhatsApp', desc: 'Twilio or Meta Cloud API.', icon: 'chat' },
  { id: 'slack', name: 'Slack', desc: 'Bot in your workspace.', icon: 'chat' },
  { id: 'email', name: 'Email', desc: 'Inbound + outbound.', icon: 'inbox' },
  { id: 'api', name: 'API only', desc: 'Call it from your own code.', icon: 'code' },
];

interface Step {
  id: string;
  label: string;
  sub: string;
}

const STEPS: Step[] = [
  { id: 'identity', label: 'Identity', sub: 'Who is this agent?' },
  { id: 'brain', label: 'Brain', sub: 'How should it think?' },
  { id: 'hands', label: 'Hands', sub: 'What can it do?' },
  { id: 'channels', label: 'Channels', sub: 'Where does it live?' },
];

interface OnboardConfig {
  template: string | null;
  name: string;
  company: string;
  prompt: string;
  model: string;
  tools: string[];
  channels: string[];
}

interface OnboardingProps {
  onDone: (next: OnboardingState | null) => void;
}

export const Onboarding = ({ onDone }: OnboardingProps) => {
  const [stepIdx, setStepIdx] = useState(0);
  const [config, setConfig] = useState<OnboardConfig>({
    template: null,
    name: '',
    company: '',
    prompt: '',
    model: 'claude-sonnet-4-6',
    tools: [],
    channels: [],
  });
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);

  const set = <K extends keyof OnboardConfig>(k: K, v: OnboardConfig[K]) =>
    setConfig((c) => ({ ...c, [k]: v }));

  const pickTemplate = (t: Template) => {
    setConfig((c) => ({
      ...c,
      template: t.id,
      prompt: t.prompt,
      tools: t.tools,
      channels: t.channels,
      name:
        c.name ||
        (t.id === 'support' ? 'support-agent'
          : t.id === 'sales' ? 'sdr-agent'
          : t.id === 'research' ? 'research-assistant'
          : ''),
    }));
  };

  const canAdvance = (() => {
    if (stepIdx === 0) return !!config.template && config.name.trim().length > 0;
    if (stepIdx === 1) return config.prompt.trim().length > 10 && !!config.model;
    if (stepIdx === 2) return true;
    if (stepIdx === 3) return config.channels.length > 0;
    return false;
  })();

  const back = () => stepIdx > 0 && setStepIdx(stepIdx - 1);

  const launch = async () => {
    setLaunching(true);
    setLaunchError(null);
    try {
      const body: OnboardingCompleteRequest = { ...config };
      const next = await completeOnboarding(body);
      // Hold the launch animation for a beat so it doesn't flash by.
      await new Promise((r) => setTimeout(r, 1800));
      onDone(next);
    } catch (e) {
      setLaunching(false);
      setLaunchError(e instanceof Error ? e.message : String(e));
    }
  };

  const next = () => {
    if (stepIdx < STEPS.length - 1) setStepIdx(stepIdx + 1);
    else void launch();
  };

  const skip = async () => {
    try {
      const next = await skipOnboarding();
      onDone(next);
    } catch {
      onDone(null);
    }
  };

  if (launching) return <OnboardLaunching name={config.name} />;

  return (
    <div className="onb">
      <div className="onb-head">
        <div className="onb-brand">
          <div className="sidebar-mark" />
          <span>
            OpenTracy <span className="dim">Evolution</span>
          </span>
        </div>
        <button className="btn ghost sm" onClick={skip}>Skip setup</button>
      </div>

      <div className="onb-body">
        <aside className="onb-rail">
          <div className="onb-rail-title">Set up your first agent</div>
          <div className="onb-rail-sub dim">
            About 2 minutes. You can change everything later — that's the whole point.
          </div>
          <ol className="onb-steps">
            {STEPS.map((s, i) => {
              const state = i < stepIdx ? 'done' : i === stepIdx ? 'on' : 'idle';
              return (
                <li key={s.id} className={`onb-step ${state}`}>
                  <span className="onb-step-num">
                    {state === 'done' ? <Icon name="check" size={12} /> : <span>{i + 1}</span>}
                  </span>
                  <div className="onb-step-text">
                    <div className="onb-step-label">{s.label}</div>
                    <div className="onb-step-sub dim">{s.sub}</div>
                  </div>
                </li>
              );
            })}
          </ol>
        </aside>

        <main className="onb-main">
          <div className="onb-content">
            {stepIdx === 0 && (
              <StepIdentity config={config} set={set} pickTemplate={pickTemplate} />
            )}
            {stepIdx === 1 && <StepBrain config={config} set={set} />}
            {stepIdx === 2 && <StepHands config={config} set={set} />}
            {stepIdx === 3 && <StepChannels config={config} set={set} />}
            {launchError && (
              <div
                className="onb-launch-note"
                style={{ marginTop: 24, background: 'var(--bad-soft)', color: 'var(--bad-fg)' }}
              >
                <Icon name="info" size={14} />
                <span>Couldn't save onboarding: {launchError}</span>
              </div>
            )}
          </div>

          <div className="onb-foot">
            <button className="btn ghost" onClick={back} disabled={stepIdx === 0}>
              ← Back
            </button>
            <span className="dim" style={{ fontSize: 12.5 }}>
              Step {stepIdx + 1} of {STEPS.length}
            </span>
            <button className="btn primary" onClick={next} disabled={!canAdvance}>
              {stepIdx === STEPS.length - 1 ? 'Launch agent' : 'Continue'}{' '}
              <Icon name="chevron" size={14} />
            </button>
          </div>
        </main>

        <aside className="onb-preview">
          <div className="onb-preview-label dim">PREVIEW</div>
          <AgentPreviewCard config={config} />
        </aside>
      </div>
    </div>
  );
};

// ─── Step 1: Identity + template ────────────────────────────────
interface StepProps {
  config: OnboardConfig;
  set: <K extends keyof OnboardConfig>(k: K, v: OnboardConfig[K]) => void;
}

const StepIdentity = ({
  config,
  set,
  pickTemplate,
}: StepProps & { pickTemplate: (t: Template) => void }) => (
  <>
    <h1 className="onb-title">What kind of agent are you building?</h1>
    <p className="onb-sub">
      Pick the closest match. We'll fill in sensible defaults — you can change anything
      in the next steps.
    </p>

    <div className="onb-templates">
      {TEMPLATES.map((t) => (
        <button
          key={t.id}
          className={`onb-template ${config.template === t.id ? 'on' : ''}`}
          onClick={() => pickTemplate(t)}
        >
          <div className="onb-tpl-icon">
            <Icon name={t.icon} size={18} />
          </div>
          <div className="onb-tpl-name">{t.name}</div>
          <div className="onb-tpl-desc dim">{t.desc}</div>
          {config.template === t.id && (
            <div className="onb-tpl-check">
              <Icon name="check" size={14} />
            </div>
          )}
        </button>
      ))}
    </div>

    {config.template && (
      <div className="onb-fields">
        <label className="onb-field">
          <span className="onb-field-label">Agent name</span>
          <input
            className="onb-input"
            placeholder="e.g. support-agent"
            value={config.name}
            onChange={(e) => set('name', e.target.value)}
          />
          <span className="onb-field-help dim">
            Lowercase, no spaces. This shows up in logs and the top bar.
          </span>
        </label>
        <label className="onb-field">
          <span className="onb-field-label">
            Company or product <span className="dim">(optional)</span>
          </span>
          <input
            className="onb-input"
            placeholder="e.g. Acme"
            value={config.company}
            onChange={(e) => set('company', e.target.value)}
          />
          <span className="onb-field-help dim">
            We'll weave it into the system prompt for you.
          </span>
        </label>
      </div>
    )}
  </>
);

// ─── Step 2: Brain ─────────────────────────────────────────────
const StepBrain = ({ config, set }: StepProps) => {
  const filled = config.prompt.replace(/\{\{company\}\}/g, config.company || 'your company');
  return (
    <>
      <h1 className="onb-title">Give it a brain.</h1>
      <p className="onb-sub">
        The system prompt is the agent's job description. Plus the model that powers it.
      </p>

      <div className="onb-field" style={{ marginBottom: 24 }}>
        <span className="onb-field-label">System prompt</span>
        <textarea
          className="onb-textarea"
          rows={9}
          value={filled}
          onChange={(e) => set('prompt', e.target.value)}
          placeholder="You are a helpful agent that…"
        />
        <span className="onb-field-help dim">
          Write it like you'd brief a new teammate. Keep it under ~300 words. Don't worry
          about perfection — your agent will rewrite this as it learns.
        </span>
      </div>

      <div className="onb-field">
        <span className="onb-field-label">Model</span>
        <div className="onb-models">
          {MODELS.map((m) => (
            <button
              key={m.id}
              className={`onb-model ${config.model === m.id ? 'on' : ''}`}
              onClick={() => set('model', m.id)}
            >
              <div className="onb-model-radio">
                <span />
              </div>
              <div style={{ flex: 1, textAlign: 'left' }}>
                <div className="onb-model-name">
                  {m.name}
                  {m.recommended && <span className="onb-model-rec">Recommended</span>}
                </div>
                <div className="onb-model-desc dim">{m.desc}</div>
              </div>
            </button>
          ))}
        </div>
        <span className="onb-field-help dim" style={{ marginTop: 10 }}>
          You can add cheaper fallbacks per intent later — the router learns which model
          to pick.
        </span>
      </div>
    </>
  );
};

// ─── Step 3: Hands ─────────────────────────────────────────────
const SUGGESTED_TOOLS: { id: string; name: string; desc: string }[] = [
  { id: 'lookup_order', name: 'lookup_order', desc: 'Find an order by ID.' },
  { id: 'search_kb', name: 'search_kb', desc: 'Search your knowledge base.' },
  { id: 'create_ticket', name: 'create_ticket', desc: 'Open a ticket for a human.' },
  { id: 'lookup_account', name: 'lookup_account', desc: 'Look up a customer account.' },
  { id: 'check_calendar', name: 'check_calendar', desc: 'See available meeting slots.' },
  { id: 'book_meeting', name: 'book_meeting', desc: 'Schedule a meeting.' },
  { id: 'web_search', name: 'web_search', desc: 'Search the public web.' },
  { id: 'fetch_url', name: 'fetch_url', desc: 'Read a URL.' },
  { id: 'save_note', name: 'save_note', desc: 'Save a note to your store.' },
];

const StepHands = ({ config, set }: StepProps) => {
  const toggle = (id: string) => {
    set(
      'tools',
      config.tools.includes(id) ? config.tools.filter((t) => t !== id) : [...config.tools, id],
    );
  };
  return (
    <>
      <h1 className="onb-title">Give it hands.</h1>
      <p className="onb-sub">
        Tools the agent can call. Skip this if you just want to chat — you can add them
        anytime.
      </p>

      <div className="onb-tools">
        {SUGGESTED_TOOLS.map((t) => (
          <button
            key={t.id}
            className={`onb-tool ${config.tools.includes(t.id) ? 'on' : ''}`}
            onClick={() => toggle(t.id)}
          >
            <div className="onb-tool-check">
              <Icon name="check" size={12} />
            </div>
            <div style={{ flex: 1, minWidth: 0, textAlign: 'left' }}>
              <div className="onb-tool-name mono">{t.name}</div>
              <div className="onb-tool-desc dim">{t.desc}</div>
            </div>
          </button>
        ))}
      </div>

      <div className="onb-mcp">
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 500, fontSize: 13.5 }}>Or connect an MCP server</div>
          <div className="dim" style={{ fontSize: 12.5, marginTop: 2 }}>
            Already have your tools defined? Paste an MCP endpoint and they'll show up
            automatically.
          </div>
        </div>
        <button className="btn" disabled>
          Connect MCP
        </button>
      </div>
    </>
  );
};

// ─── Step 4: Channels ──────────────────────────────────────────
const StepChannels = ({ config, set }: StepProps) => {
  const toggle = (id: string) => {
    set(
      'channels',
      config.channels.includes(id)
        ? config.channels.filter((c) => c !== id)
        : [...config.channels, id],
    );
  };
  return (
    <>
      <h1 className="onb-title">Where does it live?</h1>
      <p className="onb-sub">
        Pick at least one channel. Connections are configured after launch — for now we
        just stub them.
      </p>

      <div className="onb-channels">
        {CHANNELS.map((c) => (
          <button
            key={c.id}
            className={`onb-channel ${config.channels.includes(c.id) ? 'on' : ''}`}
            onClick={() => toggle(c.id)}
          >
            <div className="onb-channel-icon">
              <Icon name={c.icon} size={18} />
            </div>
            <div style={{ flex: 1, textAlign: 'left' }}>
              <div className="onb-channel-name">{c.name}</div>
              <div className="onb-channel-desc dim">{c.desc}</div>
            </div>
            <div className="onb-channel-check">
              <Icon name="check" size={14} />
            </div>
          </button>
        ))}
      </div>

      <div className="onb-launch-note">
        <Icon name="sparkles" size={14} />
        <span>
          On launch, we'll start collecting traces and look for things to improve. Your
          first lesson usually shows up within 24h.
        </span>
      </div>
    </>
  );
};

// ─── Live preview card ──────────────────────────────────────────
const AgentPreviewCard = ({ config }: { config: OnboardConfig }) => {
  const tpl = TEMPLATES.find((t) => t.id === config.template);
  const model = MODELS.find((m) => m.id === config.model);
  return (
    <div className="onb-preview-card">
      <div className="onb-preview-row">
        <div className="onb-preview-pill">
          <span className="dot pending" />
          <span className="mono">{config.name || 'unnamed-agent'}</span>
          <span className="ver">v0.1 · draft</span>
        </div>
      </div>

      <div className="onb-preview-block">
        <div className="onb-preview-label dim">PURPOSE</div>
        <div className="onb-preview-value">
          {tpl ? tpl.name : <span className="dim">Not chosen yet.</span>}
        </div>
      </div>

      <div className="onb-preview-block">
        <div className="onb-preview-label dim">SYSTEM PROMPT</div>
        <div className="onb-preview-prompt">
          {config.prompt ? (
            <span>
              {config.prompt
                .replace(/\{\{company\}\}/g, config.company || 'your company')
                .slice(0, 200)}
              {config.prompt.length > 200 ? '…' : ''}
            </span>
          ) : (
            <span className="dim">Empty.</span>
          )}
        </div>
      </div>

      <div className="onb-preview-block">
        <div className="onb-preview-label dim">MODEL</div>
        <div className="onb-preview-value">
          {model ? model.name : <span className="dim">—</span>}
        </div>
      </div>

      <div className="onb-preview-block">
        <div className="onb-preview-label dim">TOOLS · {config.tools.length}</div>
        <div className="onb-preview-chips">
          {config.tools.length === 0 ? (
            <span className="dim" style={{ fontSize: 12.5 }}>
              None yet.
            </span>
          ) : (
            config.tools.map((t) => (
              <span key={t} className="onb-chip mono">
                {t}
              </span>
            ))
          )}
        </div>
      </div>

      <div className="onb-preview-block">
        <div className="onb-preview-label dim">CHANNELS · {config.channels.length}</div>
        <div className="onb-preview-chips">
          {config.channels.length === 0 ? (
            <span className="dim" style={{ fontSize: 12.5 }}>
              None yet.
            </span>
          ) : (
            config.channels.map((c) => {
              const ch = CHANNELS.find((x) => x.id === c);
              return (
                <span key={c} className="onb-chip">
                  {ch?.name || c}
                </span>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};

// ─── Launch animation ───────────────────────────────────────────
const LAUNCH_STEPS = [
  'Building agent…',
  'Wiring tools…',
  'Connecting channels…',
  'Watching for the first conversation.',
];

const OnboardLaunching = ({ name }: { name: string }) => {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const t = setInterval(
      () => setStep((s) => Math.min(s + 1, LAUNCH_STEPS.length - 1)),
      500,
    );
    return () => clearInterval(t);
  }, []);
  return (
    <div className="onb onb-launching">
      <div className="onb-launch-inner">
        <div className="onb-launch-mark" />
        <div className="onb-launch-name mono">{name || 'unnamed-agent'}</div>
        <ul className="onb-launch-list">
          {LAUNCH_STEPS.map((s, i) => (
            <li key={i} className={i < step ? 'done' : i === step ? 'on' : 'idle'}>
              {i < step ? (
                <Icon name="check" size={12} />
              ) : i === step ? (
                <span className="onb-launch-dot" />
              ) : (
                <span className="onb-launch-tick" />
              )}
              <span>{s}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};
