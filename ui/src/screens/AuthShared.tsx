/**
 * Components shared between Login + Register screens (P16.6).
 *
 *   AuthHeader / AuthFooter  — slim chrome around the form card
 *   GoogleG                  — multi-color Google "G" mark
 *   AuthGhost                — animated mascot used inside the right pane
 *   MiniSparkline            — 36px-high trust-score line for the right pane
 *   RotatingWord             — cycles through synonyms with slide+blur
 *   EvolutionPanel           — the entire right column of the auth split
 *   isEmail / passStrength   — small validators reused across both forms
 *
 * The structure mirrors the design bundle's `auth.jsx` 1:1 so a visual
 * diff between the prototype and the app stays tractable.
 */
import { Link } from '@tanstack/react-router';
import { useEffect, useState } from 'react';

export const GoogleG = () => (
  <svg viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <path fill="#4285F4" d="M17.64 9.205c0-.639-.057-1.252-.164-1.841H9v3.481h4.844a4.14 4.14 0 0 1-1.796 2.716v2.259h2.908c1.702-1.567 2.684-3.875 2.684-6.615z"/>
    <path fill="#34A853" d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.859-3.048.859-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z"/>
    <path fill="#FBBC05" d="M3.964 10.71A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.042l3.007-2.332z"/>
    <path fill="#EA4335" d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.958L3.964 7.29C4.672 5.163 6.656 3.58 9 3.58z"/>
  </svg>
);

export const AuthGhost = () => (
  <svg className="auth-ghost-scene" width="92" height="100" viewBox="0 0 92 100" aria-hidden="true" overflow="visible">
    <g stroke="var(--accent-fg)" strokeWidth="1.6" strokeLinecap="round">
      <line className="gx-spark s1" x1="20" y1="22" x2="14" y2="14"/>
      <line className="gx-spark s2" x1="70" y1="18" x2="76" y2="10"/>
      <line className="gx-spark s3" x1="78" y1="34" x2="84" y2="32"/>
    </g>
    <g className="gx-ghost">
      <path
        d="M46 14 C24 14 12 30 12 58 L12 90 C12 92.5 14.5 93.6 16.4 91.8 L20.2 88.2 C21.4 87 23.4 87 24.6 88.2 L28.4 91.8 C29.6 93 31.6 93 32.8 91.8 L36.6 88.2 C37.8 87 39.8 87 41 88.2 L44.8 91.8 C46 93 48 93 49.2 91.8 L53 88.2 C54.2 87 56.2 87 57.4 88.2 L61.2 91.8 C63.1 93.6 65.6 92.5 65.6 90 L65.6 58 C65.6 30 58 14 46 14 Z"
        fill="var(--bg-elev)"
        stroke="var(--fg)"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <ellipse cx="35" cy="50" rx="2.4" ry="3.4" fill="var(--fg)"/>
      <ellipse cx="56" cy="50" rx="2.4" ry="3.4" fill="var(--fg)"/>
      <path d="M35 60 Q45.5 67 56 60" fill="none" stroke="var(--fg)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
    </g>
  </svg>
);

interface MiniSparklineProps {
  data: number[];
  width?: number;
  height?: number;
}

export const MiniSparkline = ({ data, width = 220, height = 36 }: MiniSparklineProps) => {
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = Math.max(1, max - min);
  const stepX = width / (data.length - 1);
  const pts = data.map((v, i) => [i * stepX, height - 4 - ((v - min) / range) * (height - 8)] as const);
  const d = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
  const [lastX, lastY] = pts[pts.length - 1];
  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <path d={`${d} L ${width} ${height} L 0 ${height} Z`} fill="var(--accent-soft)" opacity="0.5"/>
      <path d={d} fill="none" stroke="var(--accent-fg)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
      <circle cx={lastX} cy={lastY} r="2.6" fill="var(--accent-fg)"/>
    </svg>
  );
};

const ROTATE_WORDS = ['learning', 'growing', 'scaling', 'tuning'];

interface RotatingWordProps {
  words?: string[];
  intervalMs?: number;
}

export const RotatingWord = ({ words = ROTATE_WORDS, intervalMs = 2200 }: RotatingWordProps) => {
  const [idx, setIdx] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setIdx((i) => (i + 1) % words.length), intervalMs);
    return () => clearInterval(t);
  }, [words, intervalMs]);

  return (
    <span className="rot-word">
      <span className="rot-word-sizer" aria-hidden="true">{words[idx]}</span>
      {words.map((w, i) => {
        const cls = i === idx
          ? 'is-on'
          : i === (idx - 1 + words.length) % words.length
            ? 'is-out'
            : '';
        return (
          <span key={w} className={`rot-word-item ${cls}`} aria-hidden={i !== idx}>
            {w}
          </span>
        );
      })}
      <span className="rot-word-sr" aria-live="polite">{words[idx]}</span>
    </span>
  );
};

const TRUST_DATA = [72, 74, 73, 75, 78, 77, 79, 82, 81, 83, 85, 84, 86, 87];

export const EvolutionPanel = ({ variant }: { variant: 'login' | 'register' }) => {
  const headline = variant === 'login'
    ? <>The agent kept <RotatingWord/> while you were gone.</>
    : <>Your agent will start <RotatingWord/> from its first conversation.</>;

  return (
    <div className="evo-panel">
      <div className="evo-live">
        <span className="pulse"/>
        <span style={{ fontWeight: 500 }}>Live</span>
        <span className="sep"/>
        <span className="stat"><b>23</b><span>in conversation</span></span>
        <span className="sep"/>
        <span className="stat"><b>284</b><span>today</span></span>
      </div>

      <h2 className="evo-headline">{headline}</h2>

      <div className="evo-ghost-row">
        <AuthGhost/>
        <div style={{ fontSize: 12.5, color: 'var(--fg-muted)', lineHeight: 1.5 }}>
          <div style={{ color: 'var(--fg)', fontWeight: 500, marginBottom: 2 }}>checkout-support</div>
          watched 1,284 traces · evolved 7 times this week
        </div>
      </div>

      <div className="evo-lesson">
        <div className="evo-lesson-head">
          <span className="node"/>
          <span className="date">2 days ago · auto-promoted</span>
          <span className="tag"><span className="dot"/> +6% CSAT</span>
        </div>
        <div className="evo-lesson-quote">
          I noticed I was apologizing too much when customers were frustrated. I learned to acknowledge once and move to action.
        </div>
        <div className="evo-lesson-foot">
          <div className="stats">
            <span className="s"><b>412</b><span>traces affected</span></span>
            <span className="s"><b>−0.6s</b><span>avg. time-to-resolution</span></span>
          </div>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>v0.14 → v0.15</span>
        </div>
      </div>

      <div className="evo-trust">
        <div>
          <div className="evo-trust-label">Trust score</div>
          <div className="evo-trust-num">87<span className="dim"> / 100</span></div>
        </div>
        <div className="evo-trust-spark">
          <MiniSparkline data={TRUST_DATA}/>
        </div>
      </div>

      <div className="evo-tag-row">
        <span className="chip on"><span className="dot"/> Traces every conversation</span>
        <span className="chip on"><span className="dot"/> Proposes its own improvements</span>
        <span className="chip on"><span className="dot"/> Rolls back when worse</span>
      </div>
    </div>
  );
};

interface AuthHeaderProps {
  swapTo: '/login' | '/register';
  swapPrompt: string;
  swapCta: string;
}

export const AuthHeader = ({ swapTo, swapPrompt, swapCta }: AuthHeaderProps) => (
  <header className="auth-header">
    <Link to="/" className="auth-brand">
      <span className="auth-brand-mark" aria-hidden="true"/>
      <span>OpenTracy <span className="dim">Evolution</span></span>
    </Link>
    <Link to={swapTo} className="auth-header-link">
      <span className="hide-xs">{swapPrompt}</span><strong>{swapCta}</strong>
    </Link>
  </header>
);

export const AuthFooter = () => (
  <footer className="auth-page-foot">
    <div>© 2026 OpenTracy</div>
    <div className="right">
      <a href="#">Privacy</a>
      <a href="#">Terms</a>
      <a href="#">Status</a>
    </div>
  </footer>
);

// ── Validators ─────────────────────────────────────────────────────────────
export const isEmail = (v: string) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v.trim());

export const passStrength = (v: string): number => {
  if (!v) return 0;
  let s = 0;
  if (v.length >= 8) s++;
  if (v.length >= 12) s++;
  if (/[A-Z]/.test(v) && /[a-z]/.test(v)) s++;
  if (/\d/.test(v) && /[^A-Za-z0-9]/.test(v)) s++;
  return Math.min(s, 4);
};

export const strengthLabel = (s: number) =>
  ['', 'Weak', 'Fair', 'Good', 'Strong'][s] || '';
