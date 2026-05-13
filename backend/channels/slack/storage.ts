/**
 * Per-agent Slack install storage (P3.3.2).
 *
 * Each agent's Slack workspace credentials live at
 * agents/<agent_id>/integrations/slack.json. Files are written with
 * mode 0600 where supported. We use the same path convention as the
 * Python runtime so runtime/agents/channels.py can read/list these
 * files for things like wakeup status displays.
 */

import { promises as fs, readdirSync, statSync } from 'node:fs'
import path from 'node:path'

export interface SlackInstallation {
  agent_id: string
  team_id: string
  team_name: string
  bot_user_id: string
  bot_token: string
  installer_user_id: string
  installed_at: string
}

function agentsRoot(): string {
  // Resolve relative to the runtime's project root. Server is launched
  // from there in dev (`npm run dev` in backend/) — Restate / Cloud Run
  // wrappers set BACKEND_CWD if they need a different anchor.
  const fromEnv = process.env.OPENTRACY_AGENTS_ROOT
  if (fromEnv) return fromEnv
  // Walk up from cwd looking for `agents/`. Falls back to ./agents.
  let dir = process.cwd()
  for (let i = 0; i < 6; i++) {
    if (statSafe(path.join(dir, 'agents'))) return path.join(dir, 'agents')
    const parent = path.dirname(dir)
    if (parent === dir) break
    dir = parent
  }
  return path.join(process.cwd(), 'agents')
}

function statSafe(p: string): boolean {
  try {
    return statSync(p).isDirectory()
  } catch {
    return false
  }
}

function installPath(agentId: string): string {
  return path.join(agentsRoot(), agentId, 'integrations', 'slack.json')
}

export async function readInstallation(agentId: string): Promise<SlackInstallation | null> {
  try {
    const raw = await fs.readFile(installPath(agentId), 'utf8')
    return JSON.parse(raw) as SlackInstallation
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === 'ENOENT') return null
    throw e
  }
}

export async function writeInstallation(inst: SlackInstallation): Promise<void> {
  const file = installPath(inst.agent_id)
  await fs.mkdir(path.dirname(file), { recursive: true })
  await fs.writeFile(file, JSON.stringify(inst, null, 2), { mode: 0o600 })
}

export async function clearInstallation(agentId: string): Promise<void> {
  try {
    await fs.unlink(installPath(agentId))
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e
  }
}

/**
 * Reverse lookup: find the agent_id that owns a given Slack team_id.
 *
 * Used by the events webhook to route an inbound message to the right
 * agent. If multiple agents have the same team_id (collision when the
 * operator connects two agents to the same workspace), the
 * most-recently installed one wins.
 *
 * Synchronous on purpose — events handlers run hot and we want a
 * single fs.read* per webhook hit.
 */
export function findAgentByTeamId(teamId: string): SlackInstallation | null {
  const root = agentsRoot()
  let entries: string[]
  try {
    entries = readdirSync(root)
  } catch {
    return null
  }
  let best: SlackInstallation | null = null
  for (const entry of entries) {
    if (entry.startsWith('_deleted') || entry.startsWith('.')) continue
    const file = path.join(root, entry, 'integrations', 'slack.json')
    try {
      const raw = require('node:fs').readFileSync(file, 'utf8')
      const inst = JSON.parse(raw) as SlackInstallation
      if (inst.team_id !== teamId) continue
      if (!best || inst.installed_at > best.installed_at) best = inst
    } catch {
      // Not an installed agent or unreadable — skip.
    }
  }
  return best
}
