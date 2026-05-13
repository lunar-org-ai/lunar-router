/**
 * Per-agent Slack app credentials (P3.5 BYOK).
 *
 * Each agent brings its own Slack app — client_id / client_secret /
 * signing_secret — pasted into the UI. Stored at
 * agents/<id>/integrations/slack_app.json (mode 0600), gitignored along
 * with the rest of agents/.
 *
 * If an agent has no per-agent app creds, the OAuth + events flow falls
 * back to global env vars (SLACK_CLIENT_ID / SLACK_CLIENT_SECRET /
 * SLACK_SIGNING_SECRET) so existing single-app deployments keep working.
 */

import { promises as fs, readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'

export interface SlackAppCredentials {
  agent_id: string
  client_id: string
  client_secret: string
  signing_secret: string
  saved_at: string
}

function agentsRoot(): string {
  const fromEnv = process.env.OPENTRACY_AGENTS_ROOT
  if (fromEnv) return fromEnv
  let dir = process.cwd()
  for (let i = 0; i < 6; i++) {
    try {
      if (statSync(path.join(dir, 'agents')).isDirectory()) {
        return path.join(dir, 'agents')
      }
    } catch {
      /* not found, keep climbing */
    }
    const parent = path.dirname(dir)
    if (parent === dir) break
    dir = parent
  }
  return path.join(process.cwd(), 'agents')
}

function credsPath(agentId: string): string {
  return path.join(agentsRoot(), agentId, 'integrations', 'slack_app.json')
}

export async function readAppCredentials(
  agentId: string,
): Promise<SlackAppCredentials | null> {
  try {
    const raw = await fs.readFile(credsPath(agentId), 'utf8')
    return JSON.parse(raw) as SlackAppCredentials
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === 'ENOENT') return null
    throw e
  }
}

export function readAppCredentialsSync(agentId: string): SlackAppCredentials | null {
  try {
    const raw = readFileSync(credsPath(agentId), 'utf8')
    return JSON.parse(raw) as SlackAppCredentials
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === 'ENOENT') return null
    throw e
  }
}

export async function writeAppCredentials(
  agentId: string,
  creds: Omit<SlackAppCredentials, 'agent_id' | 'saved_at'>,
): Promise<void> {
  const file = credsPath(agentId)
  await fs.mkdir(path.dirname(file), { recursive: true })
  const body: SlackAppCredentials = {
    agent_id: agentId,
    saved_at: new Date().toISOString(),
    ...creds,
  }
  await fs.writeFile(file, JSON.stringify(body, null, 2), { mode: 0o600 })
}

export async function clearAppCredentials(agentId: string): Promise<void> {
  try {
    await fs.unlink(credsPath(agentId))
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e
  }
}

/**
 * List all agents that have per-agent Slack app creds installed.
 *
 * Used by the events webhook (parse-then-verify): we don't know which
 * agent the event belongs to until we parse team_id from the body, but
 * we want to verify the signature first. The two ways out are:
 *   1. Try each candidate's signing secret until one verifies (this
 *      function returns those candidates).
 *   2. Parse body unsigned, look up team, verify with their secret.
 * We do (2) in events.ts; this helper exists for tooling / future
 * scanners.
 */
export function listAgentsWithAppCreds(): string[] {
  const root = agentsRoot()
  let entries: string[]
  try {
    entries = readdirSync(root)
  } catch {
    return []
  }
  const out: string[] = []
  for (const entry of entries) {
    if (entry.startsWith('_deleted') || entry.startsWith('.')) continue
    try {
      const raw = readFileSync(path.join(root, entry, 'integrations', 'slack_app.json'), 'utf8')
      JSON.parse(raw)
      out.push(entry)
    } catch {
      /* not installed */
    }
  }
  return out
}

export function maskClientSecret(s: string): string {
  if (!s) return ''
  if (s.length <= 8) return s.slice(0, 2) + '…' + s.slice(-2)
  return s.slice(0, 4) + '…' + s.slice(-4)
}
