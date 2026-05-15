/**
 * NewAgentModal — focused create-agent modal for already-onboarded
 * operators (P16.7).
 *
 * Pick a starter template, give it a slug-friendly name, pick a model.
 * Hits POST /v1/agents and activates the result. "Use guided setup"
 * jumps into the full Onboarding flow for users who want the long form.
 */
import { useEffect, useRef, useState, type FormEvent } from 'react';
import { Icon, type IconName } from './Icon';
import { ApiError, createAgent } from '../api';

type TemplateId = 'support' | 'sdr' | 'research' | 'helpdesk' | 'blank';

interface Template {
  id: TemplateId;
  name: string;
  hint: string;
  icon: IconName;
  channels: string[];
  model: string;
  /** Seed prompt — backend requires a non-empty value at create time. */
  prompt: string;
}

const TEMPLATES: Template[] = [
  { id: 'support',  name: 'Customer support',  hint: 'Refunds, order lookups, ticket handoff.',  icon: 'chat',     channels: ['web'],   model: 'claude-sonnet-4-5',
    prompt: 'You are a customer support agent. Be concise, acknowledge once, then move to action. Hand off to a human when stuck.' },
  { id: 'sdr',      name: 'SDR / sales',       hint: 'Qualify inbound leads, book meetings.',    icon: 'bolt',     channels: ['web'],   model: 'claude-sonnet-4-5',
    prompt: 'You are an inbound SDR. Qualify the lead in three turns, then book a meeting.' },
  { id: 'research', name: 'Research',          hint: 'Search, cite sources, flag uncertainty.',  icon: 'search',   channels: ['api'],   model: 'claude-sonnet-4-5',
    prompt: 'You are a research assistant. Cite every claim, flag uncertainty explicitly.' },
  { id: 'helpdesk', name: 'Internal helpdesk', hint: 'Slack bot for IT/HR questions.',            icon: 'inbox',    channels: ['slack'], model: 'claude-haiku-4-5',
    prompt: 'You are an internal IT/HR helpdesk in Slack. Answer in one paragraph; route tickets to the right team.' },
  { id: 'blank',    name: 'Blank',             hint: 'Start from scratch — empty prompt.',       icon: 'sparkles', channels: [],        model: 'claude-sonnet-4-5',
    prompt: 'You are a helpful assistant.' },
];

const MODELS = [
  { id: 'claude-sonnet-4-5', name: 'Sonnet 4.5', meta: 'Smart default' },
  { id: 'claude-haiku-4-5',  name: 'Haiku 4.5',  meta: 'Fast · cheap' },
  { id: 'claude-opus-4',     name: 'Opus 4',     meta: 'Top reasoning' },
];

const slugify = (s: string): string =>
  s
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 32) || 'new-agent';

interface NewAgentModalProps {
  onClose: () => void;
  /** Called after a successful create. We reload the shell so every
   *  data fetcher rebinds to the new active agent — same pattern
   *  AgentSwitcher uses on activate. */
  onCreated: () => void;
  /** Hop into the full Onboarding flow for the long-form path. */
  onGuided: () => void;
}

export const NewAgentModal = ({ onClose, onCreated, onGuided }: NewAgentModalProps) => {
  const [tpl, setTpl] = useState<TemplateId>('support');
  const [name, setName] = useState('');
  const [model, setModel] = useState('claude-sonnet-4-5');
  const [touched, setTouched] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Mirror the template's recommended model unless the user changed it.
  useEffect(() => {
    const t = TEMPLATES.find((x) => x.id === tpl);
    if (t) setModel(t.model);
  }, [tpl]);

  // Suggest a slug-name from the template until the operator types one.
  useEffect(() => {
    if (touched) return;
    const t = TEMPLATES.find((x) => x.id === tpl);
    setName(t ? slugify(t.name) : '');
  }, [tpl, touched]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // ESC closes.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const slug = slugify(name);
    if (!slug || slug === 'new-agent') return;
    setSubmitting(true);
    setError(null);
    const template = TEMPLATES.find((x) => x.id === tpl);
    try {
      await createAgent({
        name: slug,
        prompt: template?.prompt ?? 'You are a helpful assistant.',
        model,
        template: tpl === 'blank' ? null : tpl,
        channels: template?.channels ?? [],
        activate: true,
      });
      onCreated();
    } catch (err) {
      const detail =
        err instanceof ApiError ? err.message : (err as Error)?.message ?? 'Create failed.';
      setError(detail);
      setSubmitting(false);
    }
  };

  return (
    <>
      <div className="modal-backdrop" onClick={onClose} />
      <div className="modal new-agent-modal" role="dialog" aria-label="Create new agent">
        <form onSubmit={submit}>
          <div className="modal-head">
            <h2>Create a new agent</h2>
            <button type="button" className="btn ghost sm" onClick={onClose} aria-label="Close">
              <Icon name="x" size={14} />
            </button>
          </div>

          <div className="modal-body">
            <div className="field">
              <label>Start from</label>
              <div className="na-template-grid">
                {TEMPLATES.map((t) => (
                  <button
                    type="button"
                    key={t.id}
                    className={`na-template ${tpl === t.id ? 'on' : ''}`}
                    onClick={() => setTpl(t.id)}
                  >
                    <span className="na-template-icon"><Icon name={t.icon} size={14} /></span>
                    <span className="na-template-text">
                      <span className="na-template-name">{t.name}</span>
                      <span className="na-template-hint">{t.hint}</span>
                    </span>
                    {tpl === t.id && (
                      <span className="na-template-check"><Icon name="check" size={12} /></span>
                    )}
                  </button>
                ))}
              </div>
            </div>

            <div className="field">
              <label htmlFor="na-name">Agent name</label>
              <input
                id="na-name"
                ref={inputRef}
                type="text"
                value={name}
                placeholder="e.g. checkout-support"
                onChange={(e) => { setTouched(true); setName(e.target.value); }}
              />
              <div className="field-hint mono">Used as slug · {slugify(name) || '—'}</div>
            </div>

            <div className="field">
              <label>Default model</label>
              <div className="na-model-grid">
                {MODELS.map((m) => (
                  <button
                    type="button"
                    key={m.id}
                    className={`na-model ${model === m.id ? 'on' : ''}`}
                    onClick={() => setModel(m.id)}
                  >
                    <span className="na-model-name">{m.name}</span>
                    <span className="na-model-meta">{m.meta}</span>
                  </button>
                ))}
              </div>
            </div>

            {error && <div className="field-hint" style={{ color: 'var(--bad-fg)' }}>{error}</div>}
          </div>

          <div className="modal-foot na-foot">
            <button type="button" className="btn ghost" onClick={onGuided}>
              <Icon name="sparkles" size={14} />
              Use guided setup
              <Icon name="chevron" size={12} />
            </button>
            <div className="na-foot-right">
              <button type="button" className="btn" onClick={onClose}>Cancel</button>
              <button
                type="submit"
                className="btn primary"
                disabled={!name.trim() || submitting}
              >
                {submitting ? (
                  <><span className="na-spin" /> Creating…</>
                ) : (
                  <>Create agent</>
                )}
              </button>
            </div>
          </div>
        </form>
      </div>
    </>
  );
};
