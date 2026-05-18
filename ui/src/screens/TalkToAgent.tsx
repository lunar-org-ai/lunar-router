import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Icon } from '../components/Icon';
import { ThinkingGhost } from '../components/ThinkingGhost';
import { Button } from '../components/ui/button';
import { Textarea } from '../components/ui/textarea';
import {
  ApiError,
  introspect,
  listAgents,
  type AgentSummary,
  type HistoryMessage,
  type IntrospectToolCall,
} from '../api';

interface Msg {
  who: 'agent' | 'user';
  text: string;
  toolCalls?: IntrospectToolCall[];
  iterations?: number;
  model?: string | null;
  error?: boolean;
}

interface Suggestion {
  short: string;
  full: string;
  hint: string;
}

// Suggestions are framed as "ask the agent about itself" — that's what
// the /introspect endpoint actually answers well. Generic chat (just
// asking the agent a question) goes through /run, not this screen.
const SUGGESTIONS: Suggestion[] = [
  {
    short: 'What changed?',
    full: 'What changed today?',
    hint: 'Recent promotions + their predicted lift.',
  },
  {
    short: 'What are you learning?',
    full: "What are you learning right now and where's the evidence?",
    hint: 'Open lessons + the traces that triggered them.',
  },
  {
    short: 'Show last rollback',
    full: 'Show me your last rollback and why it happened.',
    hint: 'Auto-rollback decisions with before/after metrics.',
  },
  {
    short: 'Predictions hit rate',
    full: 'Which predictions have we gotten right vs wrong?',
    hint: 'AHE verification outcomes by component.',
  },
];

export const TalkToAgent = () => {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [activeAgent, setActiveAgent] = useState<AgentSummary | null>(null);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [msgs]);

  // Fetch the active agent so the welcome screen can address it by name.
  useEffect(() => {
    let cancelled = false;
    void listAgents()
      .then((res) => {
        if (cancelled) return;
        const active = res.agents.find((a) => a.id === res.active) ?? null;
        setActiveAgent(active);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const ask = async (q: string) => {
    if (!q.trim() || loading) return;
    setMsgs((m) => [...m, { who: 'user', text: q }]);
    setInput('');
    setLoading(true);

    // The introspect endpoint expects the full history (excluding the
    // current question, which it appends). Old code dropped the first
    // canned msg via .slice(1); we no longer pre-seed an agent message,
    // so send everything as-is.
    const history: HistoryMessage[] = msgs.map((m) => ({
      role: m.who === 'user' ? 'user' : 'assistant',
      content: m.text,
    }));

    try {
      const result = await introspect(q, history);
      setMsgs((m) => [
        ...m,
        {
          who: 'agent',
          text: result.response,
          toolCalls: result.tool_calls,
          iterations: result.iterations,
          model: result.model,
          error: !result.success,
        },
      ]);
    } catch (e) {
      const message =
        e instanceof ApiError
          ? `Backend ${e.status}: ${e.message}`
          : `Network error: ${e instanceof Error ? e.message : String(e)}`;
      setMsgs((m) => [...m, { who: 'agent', text: message, error: true }]);
    } finally {
      setLoading(false);
    }
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      ask(input);
    }
  };

  const isEmpty = msgs.length === 0 && !loading;
  const reset = () => setMsgs([]);

  return (
    <div className="content flex flex-1 min-h-0 flex-col">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <h1 className="page-title">Talk to your agent</h1>
          <p className="page-sub">
            Ask the agent about its own behavior — what it's learning, why it changed, how it
            decided. It cites traces and evals where relevant.
          </p>
        </div>
        {msgs.length > 0 && (
          <Button variant="ghost" size="sm" onClick={reset} className="mt-1">
            Clear
          </Button>
        )}
      </div>

      <div className="mx-auto flex w-full max-w-[760px] flex-1 min-h-0 flex-col">
        {isEmpty ? (
          <TalkWelcome
            agentName={activeAgent?.name || activeAgent?.id || null}
            suggestions={SUGGESTIONS}
            onPick={ask}
            disabled={loading}
          />
        ) : (
          <div
            ref={scrollRef}
            className="flex flex-1 min-h-0 flex-col gap-[18px] overflow-y-auto px-1 py-2"
          >
            {msgs.map((m, i) => (
              <Message key={i} msg={m} />
            ))}
            {loading && (
              <div className="flex items-start gap-3">
                <Avatar who="agent" />
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-muted-foreground mb-1">Agent</div>
                  <ThinkingGhost />
                </div>
              </div>
            )}
          </div>
        )}

        <div className="mt-3 flex items-end gap-2 rounded-[var(--radius)] border border-border bg-card px-3 py-2.5">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask the agent anything…"
            rows={1}
            className="min-h-6 max-h-[120px] flex-1 resize-none border-none bg-transparent px-0 py-1 text-sm shadow-none focus-visible:ring-0 dark:bg-transparent"
          />
          <Button
            size="sm"
            onClick={() => ask(input)}
            disabled={!input.trim() || loading}
          >
            <Icon name="send" size={12} />
          </Button>
        </div>
      </div>
    </div>
  );
};

const Avatar = ({ who }: { who: 'agent' | 'user' }) => (
  <div
    className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
      who === 'agent'
        ? 'bg-gradient-to-br from-foreground from-50% to-primary to-50% text-transparent'
        : 'bg-[var(--bg-sunken)] text-muted-foreground'
    }`}
  >
    {who === 'user' ? 'You' : ''}
  </div>
);

const Message = ({ msg: m }: { msg: Msg }) => (
  <div className="flex items-start gap-3">
    <Avatar who={m.who} />
    <div className="flex-1 min-w-0">
      <div className="text-xs text-muted-foreground mb-1">{m.who === 'agent' ? 'Agent' : 'You'}</div>
      <div className={`md text-sm leading-relaxed text-foreground ${m.error ? 'dim' : ''}`}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown>
      </div>
      {m.toolCalls && m.toolCalls.length > 0 && (
        <p className="dim mt-1 text-[11px]">
          {m.iterations} iter · {m.model ?? '?'} · tools:{' '}
          {m.toolCalls.map((t) => t.tool).join(', ')}
        </p>
      )}
      {m.toolCalls && m.toolCalls.length === 0 && m.model && (
        <p className="dim mt-1 text-[11px]">{m.model}</p>
      )}
    </div>
  </div>
);

// ─── Welcome panel (P2.2.2) ─────────────────────────────────────
// Replaces the pre-seeded "Hi I'm your agent" canned message with
// a real empty state: big avatar, agent name in the headline, four
// suggestion cards. The first user message kicks off real chat.
const TalkWelcome = ({
  agentName,
  suggestions,
  onPick,
  disabled,
}: {
  agentName: string | null;
  suggestions: Suggestion[];
  onPick: (q: string) => void;
  disabled: boolean;
}) => (
  <div className="talk-welcome">
    <div className="talk-welcome-avatar">
      <Avatar who="agent" />
    </div>
    <h2 className="talk-welcome-title">
      Talk to <span className="talk-welcome-name">{agentName || 'your agent'}</span>
    </h2>
    <p className="talk-welcome-sub dim">
      Ask anything about its behavior — recent changes, decisions, what it's learning.
      It cites traces and evals where relevant.
    </p>
    <div className="talk-welcome-grid">
      {suggestions.map((s) => (
        <button
          key={s.short}
          className="talk-welcome-card"
          onClick={() => onPick(s.full)}
          disabled={disabled}
        >
          <div className="talk-welcome-card-q">{s.short}</div>
          <div className="talk-welcome-card-hint dim">{s.hint}</div>
        </button>
      ))}
    </div>
  </div>
);
