import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Icon } from '../components/Icon';
import { ThinkingGhost } from '../components/ThinkingGhost';
import { Button } from '../components/ui/button';
import { Textarea } from '../components/ui/textarea';
import { ApiError, introspect, type HistoryMessage, type IntrospectToolCall } from '../api';

interface Msg {
  who: 'agent' | 'user';
  text: string;
  toolCalls?: IntrospectToolCall[];
  iterations?: number;
  model?: string | null;
  error?: boolean;
}

const QUICK = [
  'What changed today?',
  'Show me recent promotions and their predictions',
  'Which predictions have we gotten right vs wrong?',
  'Why was v0.0.2 rolled back?',
];

export const TalkToAgent = () => {
  const [msgs, setMsgs] = useState<Msg[]>([
    {
      who: 'agent',
      text: "Hi. I'm your agent. You can ask me about anything I've changed, why I made a decision, or what I'm working on. I'll show evidence where I have it.",
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [msgs]);

  const ask = async (q: string) => {
    if (!q.trim() || loading) return;
    setMsgs((m) => [...m, { who: 'user', text: q }]);
    setInput('');
    setLoading(true);

    const history: HistoryMessage[] = msgs
      .slice(1)
      .map((m) => ({ role: m.who === 'user' ? 'user' : 'assistant', content: m.text }));

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

  return (
    <div className="content flex flex-1 min-h-0 flex-col">
      <h1 className="page-title">Talk to your agent</h1>
      <p className="page-sub">
        Ask the agent about its own behavior — what it's learning, why it changed, how it decided.
        It cites traces and evals where relevant.
      </p>

      <div className="mb-3.5 flex flex-wrap gap-1.5">
        {QUICK.map((q) => (
          <Button
            key={q}
            variant="outline"
            size="sm"
            className="h-7 rounded-full px-3 text-xs"
            onClick={() => ask(q)}
            disabled={loading}
          >
            {q}
          </Button>
        ))}
      </div>

      <div className="mx-auto flex w-full max-w-[760px] flex-1 min-h-0 flex-col">
        <div ref={scrollRef} className="flex flex-1 min-h-0 flex-col gap-[18px] overflow-y-auto px-1 py-2">
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
