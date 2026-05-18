import { useEffect, useState } from 'react';

interface Step {
  label: string;
  meta?: string;
}

const DEFAULT_STEPS: Step[] = [
  { label: 'Reading the ledger', meta: '0.3s' },
  { label: 'Checking distilled epochs', meta: '0.4s' },
  { label: 'Reviewing predictions vs outcomes', meta: '0.2s' },
  { label: 'Drafting answer' },
];

interface Props {
  steps?: Step[];
  /** ms between transitions; the last step is held until unmount. */
  intervalMs?: number;
}

export const ThinkingGhost = ({ steps = DEFAULT_STEPS, intervalMs = 900 }: Props) => {
  const [active, setActive] = useState(0);

  useEffect(() => {
    if (active >= steps.length - 1) return;
    const t = setTimeout(() => setActive((a) => a + 1), intervalMs);
    return () => clearTimeout(t);
  }, [active, steps.length, intervalMs]);

  return (
    <div className="trajectory">
      <div className="trail" aria-hidden="true">
        <span className="trail-line" />
        <span
          className="trail-fill"
          style={{ height: `${(active / (steps.length - 1)) * 100}%` }}
        />
      </div>

      {steps.map((s, i) => {
        const state = i < active ? 'done' : i === active ? 'active' : 'pending';
        return (
          <div key={i} className={`tstep ${state}`}>
            <span className="tnode">
              {state === 'done' && (
                <svg viewBox="0 0 10 10" width="8" height="8" aria-hidden="true">
                  <path
                    d="M2 5.2 L4.2 7.4 L8 3"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
              {state === 'active' && (
                <svg
                  className="tghost"
                  viewBox="0 0 14 16"
                  width="11"
                  height="13"
                  aria-hidden="true"
                >
                  <path
                    d="M7 1.2 C3.4 1.2 1.6 3.6 1.6 7.2 L1.6 14 C1.6 14.7 2.2 15 2.7 14.5 L3.8 13.6 C4.2 13.3 4.7 13.3 5.1 13.6 L5.9 14.3 C6.3 14.6 6.8 14.6 7.2 14.3 L8 13.6 C8.4 13.3 8.9 13.3 9.3 13.6 L10.4 14.5 C10.9 14.9 11.6 14.7 11.6 14 L11.6 7.2 C11.6 3.6 9.6 1.2 7 1.2 Z"
                    fill="currentColor"
                  />
                  <circle cx="5" cy="6.6" r="0.7" fill="var(--card)" />
                  <circle cx="9" cy="6.6" r="0.7" fill="var(--card)" />
                </svg>
              )}
            </span>
            <span className="tlabel">{s.label}</span>
            {s.meta && state === 'done' && <span className="tmeta">{s.meta}</span>}
            {state === 'active' && <span className="tcaret" />}
          </div>
        );
      })}
    </div>
  );
};
