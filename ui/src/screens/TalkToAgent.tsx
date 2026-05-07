import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { Icon } from '../components/Icon';
import { ApiError, runAgent, type HistoryMessage } from '../api';

interface Msg {
  who: 'agent' | 'user';
  text: string;
  traceId?: string;
  durationMs?: number;
  error?: boolean;
}

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

    // Build history from prior turns (skip the initial greeting)
    const history: HistoryMessage[] = msgs
      .slice(1)
      .map((m) => ({ role: m.who === 'user' ? 'user' : 'assistant', content: m.text }));

    try {
      const result = await runAgent(q, history);
      setMsgs((m) => [
        ...m,
        {
          who: 'agent',
          text: result.response ?? '(empty response)',
          traceId: result.trace_id,
          durationMs: result.duration_ms,
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

  const quick = [
    'Where is order #555?',
    "What's your refund policy?",
    'I want to cancel my order',
    'How long does shipping take to Brazil?',
  ];

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      ask(input);
    }
  };

  return (
    <div className="content">
      <h1 className="page-title">Talk to your agent</h1>
      <p className="page-sub">
        Ask the agent about its own behavior — what it's learning, why it changed, how it decided. It cites traces and
        evals where relevant.
      </p>

      <div className="chat-quick">
        {quick.map((q) => (
          <button key={q} className="chip" onClick={() => ask(q)}>
            {q}
          </button>
        ))}
      </div>

      <div className="chat-wrap">
        <div className="chat-msgs" ref={scrollRef}>
          {msgs.map((m, i) => (
            <div key={i} className={`msg ${m.who}`}>
              <div className="av">{m.who === 'user' ? 'You' : ''}</div>
              <div className="body">
                <div className="who">{m.who === 'agent' ? 'Agent' : 'You'}</div>
                {m.text.split('\n').map((line, j) => (
                  <p key={j} className={m.error ? 'dim' : ''}>{line}</p>
                ))}
                {m.traceId && (
                  <p className="dim" style={{ fontSize: 11, marginTop: 4 }}>
                    trace {m.traceId.slice(0, 8)}… · {m.durationMs?.toFixed(1)}ms
                  </p>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="msg agent">
              <div className="av"></div>
              <div className="body">
                <div className="who">Agent</div>
                <p className="dim">thinking…</p>
              </div>
            </div>
          )}
        </div>

        <div className="chat-input">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask the agent anything…"
            rows={1}
          />
          <button
            className="btn primary sm"
            onClick={() => ask(input)}
            disabled={!input.trim() || loading}
          >
            <Icon name="send" size={12} />
          </button>
        </div>
      </div>
    </div>
  );
};
