/**
 * AgentSwitcher — topbar dropdown showing the active agent + the list
 * of all agents in the registry (P2.0).
 *
 * Click the pill → opens a dropdown:
 *   - Active agent gets a check mark
 *   - Other agents are clickable rows → POST /agents/{id}/activate
 *   - "+ New agent" row at the bottom → re-runs the Onboarding flow
 *
 * The current agent persists across reloads naturally because the
 * server tracks it in agents/registry.json — no localStorage needed.
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
  /**
   * Open the AgentSheet drawer. Called with `null` when the operator
   * clicks the active row in the dropdown (loads the active agent), or
   * with a specific agent ID when they click the gear icon next to any
   * row — that lets them peek at another agent's config without
   * activating it.
   */
  onOpenSheet: (agentId: string | null) => void;
  /** Called when the operator picks "+ New agent" — UI swaps to onboarding. */
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
      // Force a full reload — all the screens (Evolution, Traces, etc)
      // need to re-fetch against the newly-active agent's data. Simpler
      // than threading an "active version" context through every fetcher.
      window.location.reload();
    } catch (e) {
      console.warn('activate failed', e);
    } finally {
      setSwitching(null);
    }
  };

  const handlePeek = (e: React.MouseEvent, a: AgentSummary) => {
    // Open the sheet for THIS agent without activating. Stops propagation
    // so the row's outer click (which activates) doesn't fire.
    e.stopPropagation();
    setOpen(false);
    onOpenSheet(a.id);
  };

  return (
    <div className="agent-switcher" ref={rootRef}>
      <button
        className="agent-pill"
        onClick={() => setOpen((o) => !o)}
        title={active ? `${active.name} (${active.model})` : 'No agent'}
      >
        <span className="dot" />
        <span>{active?.name || active?.id || 'no-agent'}</span>
        <span className="ver">{shortModel(active?.model)} · live</span>
        <Icon name="chevronDown" size={12} />
      </button>

      {open && state && (
        <div className="agent-switcher-menu">
          <div className="agent-switcher-label">Switch agent</div>
          {state.agents.length === 0 && (
            <div className="agent-switcher-empty dim">No agents yet.</div>
          )}
          {state.agents.map((a) => {
            const isSwitching = switching === a.id;
            return (
              <div
                key={a.id}
                className={`agent-switcher-row ${a.is_active ? 'on' : ''}`}
              >
                <button
                  className="agent-switcher-row-pick"
                  onClick={() => void handlePick(a)}
                  disabled={isSwitching}
                  title={a.is_active ? 'Open this agent' : `Switch to ${a.name}`}
                >
                  <div className="agent-switcher-row-main">
                    <span className="agent-switcher-row-name mono">{a.id}</span>
                    <span className="agent-switcher-row-sub dim">
                      {a.name} · {shortModel(a.model)}
                    </span>
                  </div>
                  {a.is_active ? (
                    <Icon name="check" size={13} />
                  ) : isSwitching ? (
                    <span className="agent-switcher-spinner" />
                  ) : null}
                </button>
                <button
                  className="agent-switcher-row-peek"
                  onClick={(e) => handlePeek(e, a)}
                  disabled={isSwitching}
                  title={a.is_active ? 'Open settings' : 'Peek at this agent without activating'}
                  aria-label={`Open ${a.name} settings`}
                >
                  <Icon name="settings" size={13} />
                </button>
              </div>
            );
          })}
          <button
            className="agent-switcher-new"
            onClick={() => {
              setOpen(false);
              onNewAgent();
            }}
          >
            <Icon name="sparkles" size={13} /> New agent
          </button>
        </div>
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
