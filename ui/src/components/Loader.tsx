/**
 * Loader — full-viewport boot loader (P16.7).
 *
 * Used during slow transitions where the chrome-less spinner inside a
 * button isn't enough: initial boot while we resolve auth mode, the
 * second half of a Google sign-in redirect (idToken → backend → tenant
 * Bearer), etc. Smaller in-button spinners stay as-is.
 *
 * Solo variant from the design bundle's Loader.html: ghost float +
 * blink + sparkle ring + cycling caption + progress dots.
 */
import { useEffect, useState } from 'react';

const MESSAGES = [
  'Preparing your agent…',
  'Learning in progress…',
  'Optimizing your experience…',
];

interface LoaderProps {
  /** Override the cycling caption with a fixed message. */
  caption?: string;
  /** Render the "OpenTracy Evolution" wordmark in the top-left. */
  brand?: boolean;
}

export const Loader = ({ caption, brand = true }: LoaderProps) => {
  const [idx, setIdx] = useState(0);
  const [fading, setFading] = useState(false);

  useEffect(() => {
    if (caption) return;
    const cycle = window.setInterval(() => {
      setFading(true);
      window.setTimeout(() => {
        setIdx((i) => (i + 1) % MESSAGES.length);
        setFading(false);
      }, 450);
    }, 2800);
    return () => window.clearInterval(cycle);
  }, [caption]);

  const text = caption ?? MESSAGES[idx];

  return (
    <>
      {brand && (
        <div className="loader-brand" aria-label="OpenTracy">
          <span className="mark" aria-hidden="true" />
          <span>OpenTracy <span className="dim">Evolution</span></span>
        </div>
      )}
      <div className="loader-stage" role="status" aria-live="polite">
        <div className="loader-art">
          <svg
            viewBox="0 0 160 100"
            width={160}
            height={100}
            style={{ position: 'absolute', bottom: '12%' }}
            aria-hidden="true"
          >
            <ellipse className="g-shadow" cx="80" cy="50" rx="34" ry="6" />
          </svg>

          <div className="g-main" style={{ position: 'relative' }}>
            <svg viewBox="0 0 80 92" width={104} height={104 * (92 / 80)} overflow="visible" aria-hidden="true">
              <path
                className="ghost-path"
                d="M40 6 C 18 6 10 24 10 48 L 10 82 C 10 85 13 86 15 84 L 20 79 C 22 77 25 77 27 79 L 32 84 C 34 86 37 86 39 84 L 44 79 C 46 77 49 77 51 79 L 56 84 C 58 86 61 86 63 84 L 68 79 C 70 77 73 78 73 81 L 73 48 C 73 24 62 6 40 6 Z"
              />
              <ellipse className="ghost-eye g-eye" cx="30" cy="46" rx="2.4" ry="3.6" />
              <ellipse className="ghost-eye g-eye" cx="50" cy="46" rx="2.4" ry="3.6" />
              <circle className="ghost-cheek" cx="22" cy="55" r="2" />
              <circle className="ghost-cheek" cx="58" cy="55" r="2" />
              <path
                d="M32 58 Q40 64 48 58"
                fill="none"
                stroke="var(--fg)"
                strokeWidth="2.4"
                strokeLinecap="round"
              />
            </svg>
            <svg
              width={180}
              height={180}
              viewBox="0 0 180 180"
              style={{ position: 'absolute', top: '-32%', left: '-32%', pointerEvents: 'none' }}
              aria-hidden="true"
            >
              <line className="sparkle s1" x1="32" y1="44" x2="22" y2="34" />
              <line className="sparkle s2" x1="148" y1="38" x2="160" y2="26" />
              <line className="sparkle s3" x1="156" y1="98" x2="170" y2="92" />
              <line className="sparkle s4" x1="22" y1="110" x2="10" y2="118" />
            </svg>
          </div>
        </div>

        <div className="loader-caption">
          <div className={`loader-caption-text ${fading ? 'fading' : ''}`}>{text}</div>
          <div className="loader-dots" aria-hidden="true">
            <i /><i /><i />
          </div>
        </div>
      </div>
    </>
  );
};
