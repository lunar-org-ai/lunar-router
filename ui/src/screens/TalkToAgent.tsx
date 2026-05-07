import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { Icon } from '../components/Icon';

interface Msg {
  who: 'agent' | 'user';
  text: string;
}

const canned: Record<string, string> = {
  'why did you change':
    "I noticed 47 failed traces in the last 7 days where customers sent order IDs with dashes. The lookup tool wants canonical form, so the model kept failing. I wrote a small wrapper that normalizes inputs at the edge — projected lift is +25 points on order-lookup success. It's waiting for your review.",
  'what are you learning':
    "Two things this week:\n\n1. Order ID formats vary more than I assumed — I drafted a normalizer.\n2. My empathy phrasing was too long. I shortened it after seeing CSAT drop on long preambles.\n\nI also tried 'concise mode' but rolled it back when CSAT dropped 0.6 in 24 hours.",
  rollback:
    'Sure — I can roll the live agent back to v0.39 (Small-model routing). That undoes the empathy prompt change. Want me to do it now or schedule it?',
};

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

  const ask = (q: string) => {
    if (!q.trim() || loading) return;
    setMsgs((m) => [...m, { who: 'user', text: q }]);
    setInput('');
    setLoading(true);
    setTimeout(() => {
      const lower = q.toLowerCase();
      const key = Object.keys(canned).find((k) => lower.includes(k));
      const reply =
        (key && canned[key]) ||
        "I'd look at the relevant traces, evals, and recent decisions to answer that. (This is a prototype — in the full product I'd reason over the live data.)";
      setMsgs((m) => [...m, { who: 'agent', text: reply }]);
      setLoading(false);
    }, 700);
  };

  const quick = [
    'Why did you change the empathy prompt?',
    'What are you learning right now?',
    'Show me your last rollback',
    'How did you decide on small-model routing?',
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
                  <p key={j}>{line}</p>
                ))}
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
