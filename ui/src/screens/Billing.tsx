/**
 * Billing — tenant tier + usage + upgrade screen.
 *
 * Shows:
 *   - current tier card with monthly usage bars (traces, evolutions)
 *   - tier comparison table with the operator's tier highlighted
 *   - per-tier upgrade buttons (disabled until Stripe checkout lands)
 *
 * In OSS mode the runtime returns tier="oss"; we render a friendly
 * "single-tenant local deploy — no billing applies" card instead.
 */

import { useEffect, useState } from 'react';
import {
  ApiError,
  createCheckoutSession,
  getBilling,
  type BillingSnapshot,
} from '../api';

type Tier = 'free' | 'starter' | 'team' | 'scale';

interface TierCard {
  id: Tier;
  name: string;
  price: string;
  tagline: string;
  features: string[];
}

const TIER_CARDS: TierCard[] = [
  {
    id: 'free',
    name: 'Free',
    price: '$0',
    tagline: 'Try Opentracy — every traced request, real evals, no card.',
    features: [
      '1,000 traces / month',
      '1 agent',
      '2 local (stdio) MCP integrations',
      '7-day trace retention',
      'Manual eval suites',
    ],
  },
  {
    id: 'starter',
    name: 'Starter',
    price: '$49 /mo',
    tagline: 'Unlock the evolution loop — your agent improves itself.',
    features: [
      '10,000 traces / month',
      '1 agent',
      '10 integrations (incl. hosted SSE/HTTP)',
      '30-day retention',
      'Daily AHE evolution loop',
    ],
  },
  {
    id: 'team',
    name: 'Team',
    price: '$199 /mo',
    tagline: 'Run multiple agents with shared traces and lessons.',
    features: [
      '100,000 traces / month',
      '5 agents',
      'Unlimited integrations',
      '90-day retention',
      'Continuous AHE',
    ],
  },
  {
    id: 'scale',
    name: 'Scale',
    price: 'Contact us',
    tagline: 'Dedicated KMS, audit logs, SLA, self-hosted option.',
    features: [
      'Unlimited traces',
      'Unlimited agents',
      'Dedicated KMS keyring',
      'Unlimited retention',
      'Priority support + SLA',
    ],
  },
];

const fmt = (n: number) => n.toLocaleString();

const capLabel = (n: number) => (n < 0 ? 'Unlimited' : fmt(n));

interface UsageBarProps {
  label: string;
  used: number;
  cap: number;
}

const UsageBar = ({ label, used, cap }: UsageBarProps) => {
  const unlimited = cap < 0;
  const pct = unlimited ? 0 : Math.min(100, Math.round((used / Math.max(1, cap)) * 100));
  const warn = pct >= 80;
  const danger = pct >= 100;
  const fill = danger ? '#ef4444' : warn ? '#f59e0b' : 'var(--accent-primary)';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
        <span style={{ color: 'var(--text-2)' }}>{label}</span>
        <span style={{ fontWeight: 500 }}>
          {fmt(used)} {unlimited ? '' : `/ ${fmt(cap)}`}
          {!unlimited && (
            <span style={{ color: 'var(--text-3)', marginLeft: 6 }}>({pct}%)</span>
          )}
        </span>
      </div>
      <div
        style={{
          height: 6,
          borderRadius: 3,
          background: 'var(--surface-2)',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: unlimited ? '100%' : `${pct}%`,
            height: '100%',
            background: unlimited ? 'var(--accent-primary)' : fill,
            opacity: unlimited ? 0.25 : 1,
            transition: 'width 200ms',
          }}
        />
      </div>
    </div>
  );
};

export const Billing = () => {
  const [snap, setSnap] = useState<BillingSnapshot | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [checkoutPending, setCheckoutPending] = useState<Tier | null>(null);
  const [checkoutErr, setCheckoutErr] = useState<string | null>(null);

  useEffect(() => {
    getBilling()
      .then(setSnap)
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, []);

  const onUpgrade = async (tier: Tier) => {
    if (tier !== 'starter' && tier !== 'team') return;
    setCheckoutPending(tier);
    setCheckoutErr(null);
    try {
      const { url } = await createCheckoutSession(tier);
      window.location.assign(url);
    } catch (e) {
      const msg =
        e instanceof ApiError && e.status === 503
          ? 'Stripe is not configured on this deployment yet.'
          : e instanceof Error
            ? e.message
            : String(e);
      setCheckoutErr(msg);
      setCheckoutPending(null);
    }
  };

  if (err) {
    return (
      <div className="screen">
        <h2>Billing</h2>
        <div className="card" style={{ color: 'var(--danger)' }}>
          Failed to load billing: {err}
        </div>
      </div>
    );
  }

  if (!snap) {
    return (
      <div className="screen">
        <h2>Billing</h2>
        <div className="card">Loading…</div>
      </div>
    );
  }

  if (snap.tier === 'oss') {
    return (
      <div className="screen">
        <h2>Billing</h2>
        <div className="card" style={{ padding: 24 }}>
          <div style={{ fontSize: 16, fontWeight: 500, marginBottom: 8 }}>
            OSS local deploy
          </div>
          <div style={{ color: 'var(--text-2)', lineHeight: 1.5 }}>
            You're running Opentracy in single-tenant local mode. No
            quotas apply, and there's nothing to bill. The cloud-hosted
            multi-tenant build adds tier-based limits + Stripe billing.
          </div>
        </div>
      </div>
    );
  }

  const currentTier = snap.tier as Tier;

  return (
    <div className="screen" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div>
        <h2 style={{ marginBottom: 4 }}>Billing</h2>
        <div style={{ color: 'var(--text-2)', fontSize: 13 }}>
          Period {snap.period} · Tier{' '}
          <strong style={{ textTransform: 'capitalize' }}>{snap.tier}</strong>
        </div>
      </div>

      <div className="card" style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 500 }}>This month's usage</div>
        <UsageBar
          label="Traces"
          used={snap.usage.traces}
          cap={snap.limits.monthly_traces}
        />
        <UsageBar
          label="Evolution iterations"
          used={snap.usage.evolutions}
          cap={-1 /* uncapped per tier; surface for awareness */}
        />
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: 12,
            paddingTop: 8,
            borderTop: '1px solid var(--border)',
            fontSize: 12,
            color: 'var(--text-2)',
          }}
        >
          <div>
            <div style={{ color: 'var(--text-3)' }}>Agents</div>
            <div style={{ color: 'var(--text-1)', fontWeight: 500 }}>
              up to {capLabel(snap.limits.max_agents)}
            </div>
          </div>
          <div>
            <div style={{ color: 'var(--text-3)' }}>Integrations / agent</div>
            <div style={{ color: 'var(--text-1)', fontWeight: 500 }}>
              {capLabel(snap.limits.max_integrations_per_agent)}
            </div>
          </div>
          <div>
            <div style={{ color: 'var(--text-3)' }}>Retention</div>
            <div style={{ color: 'var(--text-1)', fontWeight: 500 }}>
              {snap.limits.retention_days < 0 ? 'Unlimited' : `${snap.limits.retention_days} days`}
            </div>
          </div>
          <div>
            <div style={{ color: 'var(--text-3)' }}>Rate limit</div>
            <div style={{ color: 'var(--text-1)', fontWeight: 500 }}>
              {snap.limits.rate_limit_per_minute < 0
                ? 'Unlimited'
                : `${snap.limits.rate_limit_per_minute} / min`}
            </div>
          </div>
        </div>
      </div>

      <div>
        <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 12 }}>
          Plans
        </div>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: 12,
          }}
        >
          {TIER_CARDS.map((t) => {
            const isCurrent = t.id === currentTier;
            const isLower =
              ['free', 'starter', 'team', 'scale'].indexOf(t.id) <
              ['free', 'starter', 'team', 'scale'].indexOf(currentTier);
            return (
              <div
                key={t.id}
                className="card"
                style={{
                  padding: 16,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 12,
                  border: isCurrent
                    ? '1px solid var(--accent-primary)'
                    : '1px solid var(--border)',
                  background: isCurrent ? 'var(--accent-soft)' : 'var(--surface-1)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                  <div style={{ fontWeight: 600 }}>{t.name}</div>
                  <div style={{ fontSize: 14, color: 'var(--text-2)' }}>{t.price}</div>
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.4 }}>
                  {t.tagline}
                </div>
                <ul
                  style={{
                    listStyle: 'none',
                    padding: 0,
                    margin: 0,
                    fontSize: 12,
                    color: 'var(--text-2)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 4,
                  }}
                >
                  {t.features.map((f) => (
                    <li key={f}>· {f}</li>
                  ))}
                </ul>
                <button
                  type="button"
                  disabled={
                    isCurrent ||
                    t.id === 'free' ||
                    t.id === 'scale' ||
                    isLower ||
                    checkoutPending !== null
                  }
                  onClick={() => onUpgrade(t.id)}
                  title={
                    t.id === 'scale'
                      ? 'Contact sales to upgrade to Scale'
                      : t.id === 'free'
                        ? 'Free is the default tier'
                        : isLower
                          ? 'Downgrades go through support today'
                          : 'Start a Stripe Checkout for this plan'
                  }
                  style={{
                    marginTop: 'auto',
                    padding: '8px 12px',
                    fontSize: 13,
                    fontWeight: 500,
                    border: '1px solid var(--border)',
                    borderRadius: 6,
                    background: isCurrent
                      ? 'transparent'
                      : isLower || t.id === 'scale' || t.id === 'free'
                        ? 'var(--surface-2)'
                        : 'var(--accent-primary)',
                    color: isCurrent
                      ? 'var(--text-2)'
                      : isLower || t.id === 'scale' || t.id === 'free'
                        ? 'var(--text-3)'
                        : 'white',
                    cursor:
                      isCurrent || isLower || t.id === 'scale' || t.id === 'free'
                        ? 'not-allowed'
                        : 'pointer',
                  }}
                >
                  {isCurrent
                    ? 'Current plan'
                    : checkoutPending === t.id
                      ? 'Redirecting…'
                      : t.id === 'scale'
                        ? 'Contact sales'
                        : isLower
                          ? 'Contact support'
                          : 'Upgrade'}
                </button>
              </div>
            );
          })}
        </div>
        {checkoutErr && (
          <div
            style={{
              marginTop: 12,
              padding: '8px 12px',
              fontSize: 12,
              color: 'var(--danger)',
              background: 'rgba(239, 68, 68, 0.08)',
              border: '1px solid rgba(239, 68, 68, 0.25)',
              borderRadius: 6,
            }}
          >
            {checkoutErr}
          </div>
        )}
      </div>
    </div>
  );
};
