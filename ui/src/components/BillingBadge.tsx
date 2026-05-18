/**
 * BillingBadge — topbar pill showing tier + monthly trace usage.
 *
 * Polls /v1/billing once on mount and every 5 minutes. Hidden when the
 * runtime reports the synthetic "oss" tier (single-tenant local mode),
 * since caps don't apply there.
 *
 * Visual states:
 *   - Free, <80% used → neutral pill: "Free · 245 / 1,000"
 *   - Free, ≥80% used → warning pill: "Free · 920 / 1,000 · Upgrade"
 *   - Paid tiers      → tier name pill: "Starter" (no usage shown unless ≥80%)
 */

import { useEffect, useState } from 'react';
import { getBilling, type BillingSnapshot } from '../api';

const POLL_MS = 5 * 60 * 1000;

const formatTier = (tier: string) =>
  tier.charAt(0).toUpperCase() + tier.slice(1);

const fmt = (n: number) => n.toLocaleString();

interface Props {
  /** When provided, clicking the badge opens the Billing screen. */
  onOpenBilling?: () => void;
}

export const BillingBadge = ({ onOpenBilling }: Props) => {
  const [snap, setSnap] = useState<BillingSnapshot | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () => {
      getBilling()
        .then((s) => {
          if (alive) setSnap(s);
        })
        .catch(() => {
          // Network error or unauth'd visitor — silently hide the badge.
          if (alive) setSnap(null);
        });
    };
    load();
    const iv = setInterval(load, POLL_MS);
    return () => {
      alive = false;
      clearInterval(iv);
    };
  }, []);

  if (!snap || snap.tier === 'oss') return null;

  const { tier, usage, limits } = snap;
  const cap = limits.monthly_traces;
  const used = usage.traces;
  const pct = cap > 0 ? Math.min(100, Math.round((used / cap) * 100)) : 0;
  const isFree = tier === 'free';
  const warn = cap > 0 && pct >= 80;

  const bg = warn ? 'rgba(245, 158, 11, 0.12)' : 'rgba(255,255,255,0.04)';
  const fg = warn ? '#f59e0b' : 'var(--text-2)';
  const border = warn ? 'rgba(245, 158, 11, 0.35)' : 'var(--border)';

  return (
    <button
      type="button"
      onClick={onOpenBilling}
      title={
        cap > 0
          ? `${formatTier(tier)} tier — ${fmt(used)} of ${fmt(cap)} traces used this month`
          : `${formatTier(tier)} tier`
      }
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 8,
        height: 28,
        padding: '0 10px',
        fontSize: 12,
        fontWeight: 500,
        color: fg,
        background: bg,
        border: `1px solid ${border}`,
        borderRadius: 999,
        cursor: onOpenBilling ? 'pointer' : 'default',
      }}
    >
      <span>{formatTier(tier)}</span>
      {cap > 0 && (isFree || warn) && (
        <span style={{ opacity: 0.85 }}>
          {fmt(used)} / {fmt(cap)}
        </span>
      )}
      {warn && (
        <span
          style={{
            color: '#f59e0b',
            fontWeight: 600,
            marginLeft: 2,
          }}
        >
          Upgrade
        </span>
      )}
    </button>
  );
};
