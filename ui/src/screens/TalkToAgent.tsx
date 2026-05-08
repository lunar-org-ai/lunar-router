import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Icon } from '../components/Icon';
import { ThinkingGhost } from '../components/ThinkingGhost';
import { ApiError, introspect, type HistoryMessage, type IntrospectToolCall } from '../api';

interface Msg {
  who: 'agent' | 'user';
  text: string;
  toolCalls?: IntrospectToolCall[];
  iterations?: number;
  model?: string | null;
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

  const quick = [
    'What changed today?',
    'Show me recent promotions and their predictions',
    'Which predictions have we gotten right vs wrong?',
    'Why was v0.0.2 rolled back?',
  ];

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      ask(input);
    }
  };

  return (
    <div className="content content-chat">
      <h1 className="page-title">Talk to your agent</h1>
      <p className="page-sub">
        Ask the agent about its own behavior — what it's learning, why it changed, how it decided.
        It cites traces and evals where relevant.
      </p>

      <div className="chat-quick">
        {quick.map((q) => (
          <button key={q} className="chip" onClick={() => ask(q)} disabled={loading}>
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
                <div className={`md ${m.error ? 'dim' : ''}`}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown>
                </div>
                {m.toolCalls && m.toolCalls.length > 0 && (
                  <p className="dim" style={{ fontSize: 11, marginTop: 4 }}>
                    {m.iterations} iter · {m.model ?? '?'} · tools:{' '}
                    {m.toolCalls.map((t) => t.tool).join(', ')}
                  </p>
                )}
                {m.toolCalls && m.toolCalls.length === 0 && m.model && (
                  <p className="dim" style={{ fontSize: 11, marginTop: 4 }}>
                    {m.model}
                  </p>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="msg agent thinking-msg">
              <div className="av"></div>
              <div className="body">
                <div className="who">Agent</div>
                <ThinkingGhost />
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
