/**
 * GhostMascot — three states (writing / thinking / sleeping) ported from the
 * Claude Design source, adapted to the platform's color tokens.
 *
 * Colors:
 *   - Glow + accents → var(--accent)  (tracks user's accent choice)
 *   - Body           → var(--bg-card) (neutral, not green-tinted)
 *   - Outlines       → var(--fg)
 *   - Surfaces       → var(--bg-muted) / var(--bg-sunken)
 *   - Speed trails   → var(--warn) / var(--info) / var(--accent)
 *
 * Animations live in styles.css under "ghost mascot" so themes can adjust.
 */

import { useId } from 'react';

export type GhostState = 'writing' | 'thinking' | 'sleeping';

interface Props {
  state?: GhostState;
  size?: number;
  className?: string;
}

export const GhostMascot = ({ state = 'thinking', size = 220, className = '' }: Props) => {
  const filterId = useId().replace(/:/g, '');

  return (
    <div
      className={`ghost-mascot ${className}`}
      style={{ width: size, aspectRatio: '4 / 3' }}
    >
      <svg viewBox="0 0 300 200" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <filter id={`glow-${filterId}`} x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur in="SourceAlpha" stdDeviation="10" result="blur" />
            <feFlood floodColor="var(--accent)" floodOpacity="0.35" result="glowColor" />
            <feComposite in="glowColor" in2="blur" operator="in" result="glow" />
            <feMerge>
              <feMergeNode in="glow" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {state === 'writing' && <WritingGhost filterId={filterId} />}
        {state === 'thinking' && <ThinkingGhostSvg filterId={filterId} />}
        {state === 'sleeping' && <SleepingGhost filterId={filterId} />}
      </svg>
    </div>
  );
};

// ---------- writing ----------

const WritingGhost = ({ filterId }: { filterId: string }) => (
  <>
    {/* Action lines around the keyboard */}
    <g className="gm-pulse" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" opacity="0.85">
      <line x1="60" y1="60" x2="70" y2="70" />
      <line x1="75" y1="45" x2="82" y2="60" />
      <line x1="100" y1="40" x2="100" y2="55" />
    </g>

    {/* Floating checkmark */}
    <g className="gm-float gm-d100" transform="translate(180, 50)">
      <circle cx="20" cy="20" r="15" fill="var(--bg)" stroke="var(--accent)" strokeWidth="2.5" />
      <path
        d="M 12 20 L 17 25 L 27 14"
        fill="none"
        stroke="var(--accent)"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </g>

    {/* Coffee cup */}
    <g transform="translate(30, 125)">
      <path
        className="gm-steam-1"
        d="M 12 -5 Q 8 -10 12 -15 Q 16 -20 12 -25"
        fill="none"
        stroke="var(--fg-subtle)"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        className="gm-steam-2"
        d="M 20 -2 Q 25 -8 20 -14 Q 15 -20 20 -26"
        fill="none"
        stroke="var(--fg-subtle)"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M 5 0 L 27 0 L 23 35 L 9 35 Z"
        fill="var(--bg-card)"
        stroke="var(--fg)"
        strokeWidth="3"
        strokeLinejoin="round"
      />
      <rect x="2" y="-4" width="28" height="4" rx="1" fill="var(--bg-card)" stroke="var(--fg)" strokeWidth="3" />
      <polygon
        points="7,12 25,12 24,22 8,22"
        fill="var(--bg-sunken)"
        stroke="var(--fg)"
        strokeWidth="3"
        strokeLinejoin="round"
      />
    </g>

    {/* Ghost body */}
    <g className="gm-bob">
      <path
        d="M 60 160 C 60 80, 75 50, 115 50 C 155 50, 170 80, 170 160
           Q 155 150 142.5 160 Q 128.75 150 115 160 Q 101.25 150 87.5 160 Q 73.75 150 60 160 Z"
        fill="var(--bg-card)"
        stroke="var(--fg)"
        strokeWidth="4"
        strokeLinejoin="round"
        filter={`url(#glow-${filterId})`}
      />
      <g className="gm-blink">
        <ellipse cx="100" cy="85" rx="4" ry="6" fill="var(--fg)" />
        <ellipse cx="125" cy="85" rx="4" ry="6" fill="var(--fg)" />
      </g>
      <path
        d="M 108 100 Q 112.5 105 117 100"
        fill="none"
        stroke="var(--fg)"
        strokeWidth="3"
        strokeLinecap="round"
      />
      <path
        className="gm-type-l"
        d="M 85 135 C 95 145, 105 145, 110 152"
        fill="none"
        stroke="var(--fg)"
        strokeWidth="3"
        strokeLinecap="round"
      />
      <path
        className="gm-type-r"
        d="M 115 130 C 120 145, 130 145, 135 150"
        fill="none"
        stroke="var(--fg)"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </g>

    {/* Desk */}
    <line x1="20" y1="160" x2="280" y2="160" stroke="var(--fg)" strokeWidth="4" strokeLinecap="round" />

    {/* Laptop */}
    <g transform="translate(145, 90)">
      <polygon
        points="15,65 30,-5 100,-5 85,65"
        fill="var(--bg-muted)"
        stroke="var(--fg)"
        strokeWidth="3"
        strokeLinejoin="round"
      />
      <rect
        x="0"
        y="65"
        width="100"
        height="5"
        rx="2"
        fill="var(--bg-sunken)"
        stroke="var(--fg)"
        strokeWidth="3"
      />
      <path
        d="M 58 30 C 58 20, 68 20, 68 30 Q 65 28 63 30 Q 60 28 58 30 Z"
        fill="var(--accent)"
        opacity="0.55"
      />
      <circle cx="61" cy="25" r="1" fill="var(--bg-sunken)" />
      <circle cx="65" cy="25" r="1" fill="var(--bg-sunken)" />
    </g>
  </>
);

// ---------- thinking ----------

const ThinkingGhostSvg = ({ filterId }: { filterId: string }) => (
  <>
    {/* Thought bubbles + AI chip */}
    <g className="gm-float">
      <circle cx="150" cy="80" r="4" fill="var(--bg)" stroke="var(--accent)" strokeWidth="2.5" />
      <circle cx="165" cy="65" r="7" fill="var(--bg)" stroke="var(--accent)" strokeWidth="2.5" />
      <path
        d="M 185 35 C 185 20, 205 10, 220 15 C 235 0, 265 5, 270 25 C 290 25, 290 50, 275 60 C 285 75, 255 85, 240 80 C 225 95, 190 85, 185 70 C 170 65, 175 40, 185 35 Z"
        fill="var(--bg-elev)"
        stroke="var(--accent)"
        strokeWidth="3"
        strokeLinejoin="round"
      />
      <g transform="translate(210, 30)">
        <rect x="10" y="10" width="30" height="25" rx="3" fill="var(--bg)" stroke="var(--accent)" strokeWidth="2.5" />
        <line x1="15" y1="10" x2="15" y2="6" stroke="var(--accent)" strokeWidth="2" />
        <line x1="25" y1="10" x2="25" y2="6" stroke="var(--accent)" strokeWidth="2" />
        <line x1="35" y1="10" x2="35" y2="6" stroke="var(--accent)" strokeWidth="2" />
        <line x1="15" y1="35" x2="15" y2="39" stroke="var(--accent)" strokeWidth="2" />
        <line x1="25" y1="35" x2="25" y2="39" stroke="var(--accent)" strokeWidth="2" />
        <line x1="35" y1="35" x2="35" y2="39" stroke="var(--accent)" strokeWidth="2" />
        <line x1="10" y1="16" x2="6" y2="16" stroke="var(--accent)" strokeWidth="2" />
        <line x1="10" y1="28" x2="6" y2="28" stroke="var(--accent)" strokeWidth="2" />
        <line x1="40" y1="16" x2="44" y2="16" stroke="var(--accent)" strokeWidth="2" />
        <line x1="40" y1="28" x2="44" y2="28" stroke="var(--accent)" strokeWidth="2" />
        <text x="25" y="27" fontSize="12" fontWeight="bold" fill="var(--accent)" textAnchor="middle">
          AI
        </text>
      </g>
    </g>

    {/* Bouncing dots */}
    <g transform="translate(220, 100)">
      <circle cx="0" cy="0" r="3" fill="var(--accent)" className="gm-pulse gm-d100" />
      <circle cx="15" cy="0" r="3" fill="var(--accent)" className="gm-pulse gm-d200" />
      <circle cx="30" cy="0" r="3" fill="var(--fg-muted)" className="gm-pulse gm-d300" />
    </g>

    {/* Ghost body */}
    <g>
      <path
        d="M 60 160 C 60 80, 75 50, 115 50 C 155 50, 170 80, 170 160
           Q 155 150 142.5 160 Q 128.75 150 115 160 Q 101.25 150 87.5 160 Q 73.75 150 60 160 Z"
        fill="var(--bg-card)"
        stroke="var(--fg)"
        strokeWidth="4"
        strokeLinejoin="round"
        filter={`url(#glow-${filterId})`}
      />
      <g className="gm-blink">
        <ellipse cx="108" cy="85" rx="4" ry="6" fill="var(--fg)" />
        <ellipse cx="132" cy="85" rx="4" ry="6" fill="var(--fg)" />
      </g>
      <path
        d="M 115 105 Q 120 102 125 105"
        fill="none"
        stroke="var(--fg)"
        strokeWidth="3"
        strokeLinecap="round"
      />
      <path
        className="gm-bob"
        d="M 95 125 C 100 135, 115 135, 120 120 C 125 110, 120 105, 115 105 C 110 105, 105 110, 105 115"
        fill="var(--bg-card)"
        stroke="var(--fg)"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </g>

    <line x1="20" y1="160" x2="280" y2="160" stroke="var(--fg)" strokeWidth="4" strokeLinecap="round" />

    <g transform="translate(145, 90)">
      <polygon
        points="15,65 30,-5 100,-5 85,65"
        fill="var(--bg-muted)"
        stroke="var(--fg)"
        strokeWidth="3"
        strokeLinejoin="round"
      />
      <rect x="0" y="65" width="100" height="5" rx="2" fill="var(--bg-sunken)" stroke="var(--fg)" strokeWidth="3" />
      <path
        d="M 58 30 C 58 20, 68 20, 68 30 Q 65 28 63 30 Q 60 28 58 30 Z"
        fill="var(--accent)"
        opacity="0.55"
      />
      <circle cx="61" cy="25" r="1" fill="var(--bg-sunken)" />
      <circle cx="65" cy="25" r="1" fill="var(--bg-sunken)" />
    </g>
  </>
);

// ---------- sleeping ----------

const SleepingGhost = ({ filterId }: { filterId: string }) => (
  <>
    <text x="175" y="75" fill="var(--accent)" fontWeight="bold" fontSize="16" className="gm-z gm-z-1">
      z
    </text>
    <text x="195" y="55" fill="var(--accent)" fontWeight="bold" fontSize="22" className="gm-z gm-z-2">
      Z
    </text>
    <text x="215" y="35" fill="var(--accent)" fontWeight="bold" fontSize="28" className="gm-z gm-z-3">
      Z
    </text>

    <g className="gm-float">
      {/* Speed/comet trails — platform palette */}
      <g strokeLinecap="round" fill="none" strokeWidth="3.5">
        <path d="M 75 160 Q 45 165 25 170" stroke="var(--warn)" className="gm-line gm-d100" />
        <path d="M 85 170 Q 55 175 35 180" stroke="var(--info)" className="gm-line gm-d200" />
        <path d="M 95 180 Q 65 185 45 190" stroke="var(--accent)" className="gm-line gm-d300" />
        <path d="M 115 170 Q 90 185 65 195" stroke="var(--warn)" className="gm-line gm-d100" />
      </g>

      <path
        d="M 150 60
           C 185 60, 195 90, 195 130
           C 195 145, 185 160, 175 155
           C 165 150, 160 165, 150 160
           C 140 155, 135 165, 125 160
           C 110 150, 85 175, 65 180
           C 80 170, 95 155, 105 130
           C 110 90, 115 60, 150 60 Z"
        fill="var(--bg-card)"
        stroke="var(--fg)"
        strokeWidth="4"
        strokeLinejoin="round"
        filter={`url(#glow-${filterId})`}
      />

      <path d="M 125 110 Q 133 118 141 110" fill="none" stroke="var(--fg)" strokeWidth="3.5" strokeLinecap="round" />
      <path d="M 155 110 Q 163 118 171 110" fill="none" stroke="var(--fg)" strokeWidth="3.5" strokeLinecap="round" />
      <path d="M 143 125 Q 148 132 153 125" fill="none" stroke="var(--fg)" strokeWidth="3" strokeLinecap="round" />
    </g>
  </>
);
