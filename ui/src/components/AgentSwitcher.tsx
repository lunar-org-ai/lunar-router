/**
 * AgentSwitcher — topbar dropdown showing the active agent + the list
 * of all agents in the registry (P2.0, restyled P16.7).
 *
 * Click the pill → opens a dropdown:
 *   - List rows are clickable → activates that agent + reloads
 *   - Footer has "New agent" (opens NewAgentModal) and
 *     "Configure current agent" (opens AgentSheet for the active one)
 *
 * The active agent persists across reloads because the server tracks
 * it in agents/registry.json — no localStorage needed.
 */
import { useEffect, useRef, useState } from 'react';

import {
  activateAgent,
  listAgents,
  type AgentListResponse,
  type AgentSummary,
} from '../api';
import { Icon } from './Icon';

interface AgentSwitcherProps {
  /** Open the AgentSheet drawer for the active agent. */
  onOpenSheet: (agentId: string | null) => void;
  /** Click on "+ New agent" — opens NewAgentModal. */
  onNewAgent: () => void;
}

export const AgentSwitcher = ({ onOpenSheet, onNewAgent }: AgentSwitcherProps) => {
  const [state, setState] = useState<AgentListResponse | null>(null);
  const [open, setOpen] = useState(false);
  const [switching, setSwitching] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  const refresh = async () => {
    try {
      const next = await listAgents();
      setState(next);
    } catch {
      // Surface as a "?-version" pill so the UI never deadlocks.
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [open]);

  const active: AgentSummary | null =
    state?.agents.find((a) => a.id === state.active) ?? null;

  const handlePick = async (a: AgentSummary) => {
    if (a.is_active) {
      setOpen(false);
      onOpenSheet(null);
      return;
    }
    setSwitching(a.id);
    try {
      await activateAgent(a.id);
      await refresh();
      setOpen(false);
      // Force a full reload — every screen needs to re-fetch against
      // the newly-active agent's data. Simpler than threading an
      // "active version" context through every fetcher.
      window.location.reload();
    } catch (e) {
      console.warn('activate failed', e);
    } finally {
      setSwitching(null);
    }
  };

  return (
    <div className="agent-switcher" ref={rootRef}>
      <button
        className="agent-pill"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        title={active ? `${active.name} (${active.model})` : 'No agent'}
      >
        <span className="dot" />
        <span>{active?.name || active?.id || 'no-agent'}</span>
        <span className="ver">{shortModel(active?.model)} · live</span>
        <Icon name="chevronDown" size={12} />
      </button>

      {open && state && (
        <>
          <div className="popover-backdrop" onClick={() => setOpen(false)} />
          <div className="agent-menu" role="menu">
            <div className="agent-menu-label">Your agents</div>
            <div className="agent-menu-list">
              {state.agents.length === 0 && (
                <div style={{ padding: '12px 14px', color: 'var(--fg-muted)', fontSize: 12 }}>
                  No agents yet.
                </div>
              )}
              {state.agents.map((a) => {
                const isSwitching = switching === a.id;
                return (
                  <button
                    key={a.id}
                    className={`agent-menu-item ${a.is_active ? 'on' : ''}`}
                    onClick={() => void handlePick(a)}
                    disabled={isSwitching}
                  >
                    <span className="dot" />
                    <span className="agent-menu-text">
                      <span className="agent-menu-name">{a.name || a.id}</span>
                      <span className="agent-menu-meta">{shortModel(a.model)} · {a.is_active ? 'live' : 'idle'}</span>
                    </span>
                    {a.is_active && <Icon name="check" size={13} />}
                  </button>
                );
              })}
            </div>
            <div className="agent-menu-foot">
              <button
                className="agent-menu-item primary"
                onClick={() => { setOpen(false); onNewAgent(); }}
              >
                <span className="agent-menu-plus"><Icon name="plus" size={12} /></span>
                <span className="agent-menu-text">
                  <span className="agent-menu-name">New agent</span>
                  <span className="agent-menu-meta">Start from a template or scratch</span>
                </span>
              </button>
              {active && (
                <button
                  className="agent-menu-item ghost"
                  onClick={() => { setOpen(false); onOpenSheet(null); }}
                >
                  <Icon name="settings" size={13} />
                  <span className="agent-menu-text">
                    <span className="agent-menu-name">Configure current agent</span>
                  </span>
                </button>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

function shortModel(id: string | undefined | null): string {
  if (!id) return '—';
  // "claude-sonnet-4-6" → "sonnet 4.6"
  const m = /^claude-(haiku|sonnet|opus)-(\d+)-(\d+)/i.exec(id);
  if (m) return `${m[1]} ${m[2]}.${m[3]}`;
  return id;
}
