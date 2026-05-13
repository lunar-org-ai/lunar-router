/**
 * Per-agent WhatsApp/Twilio credentials storage (P3.3.3).
 *
 * Files live at agents/<agent_id>/integrations/whatsapp.json. Same
 * filesystem convention the runtime + other channels use.
 *
 * The operator pastes their Twilio creds from console.twilio.com into
 * the AgentSheet's Channels tab; the file is written with mode 0600
 * where supported.
 */

import { promises as fs, readdirSync, statSync } from 'node:fs'
import path from 'node:path'

export interface WhatsAppConfig {
  agent_id: string
  provider: 'twilio'
  account_sid: string
  auth_token: string
  from_number: string  // E.164 with whatsapp: prefix, e.g. "whatsapp:+14155238886"
  installer_email: string | null
  installed_at: string
}

function agentsRoot(): string {
  const fromEnv = process.env.OPENTRACY_AGENTS_ROOT
  if (fromEnv) return fromEnv
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

function configPath(agentId: string): string {
  return path.join(agentsRoot(), agentId, 'integrations', 'whatsapp.json')
}

export async function readConfig(agentId: string): Promise<WhatsAppConfig | null> {
  try {
    const raw = await fs.readFile(configPath(agentId), 'utf8')
    return JSON.parse(raw) as WhatsAppConfig
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === 'ENOENT') return null
    throw e
  }
}

export async function writeConfig(cfg: WhatsAppConfig): Promise<void> {
  const file = configPath(cfg.agent_id)
  await fs.mkdir(path.dirname(file), { recursive: true })
  await fs.writeFile(file, JSON.stringify(cfg, null, 2), { mode: 0o600 })
}

export async function clearConfig(agentId: string): Promise<void> {
  try {
    await fs.unlink(configPath(agentId))
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e
  }
}

/**
 * Reverse lookup: find the agent whose configured from_number matches
 * the destination Twilio is delivering to. Twilio webhooks include a
 * ``To`` field with the bot's number (the operator's Twilio number),
 * which we match against each agent's stored ``from_number``.
 *
 * Synchronous on purpose — webhook hot path.
 */
export function findAgentByFromNumber(toNumber: string): WhatsAppConfig | null {
  const root = agentsRoot()
  const target = normalizeNumber(toNumber)
  let entries: string[]
  try {
    entries = readdirSync(root)
  } catch {
    return null
  }
  let best: WhatsAppConfig | null = null
  for (const entry of entries) {
    if (entry.startsWith('_deleted') || entry.startsWith('.')) continue
    const file = path.join(root, entry, 'integrations', 'whatsapp.json')
    try {
      const raw = require('node:fs').readFileSync(file, 'utf8')
      const cfg = JSON.parse(raw) as WhatsAppConfig
      if (normalizeNumber(cfg.from_number) !== target) continue
      if (!best || cfg.installed_at > best.installed_at) best = cfg
    } catch {
      // skip unreadable entries
    }
  }
  return best
}

/** Strip ``whatsapp:`` prefix + non-digit/+ chars so comparisons are
 *  resilient to formatting differences between Twilio webhooks + the
 *  operator's input. */
function normalizeNumber(n: string): string {
  if (!n) return ''
  return n.replace(/^whatsapp:/, '').replace(/[^+\d]/g, '')
}
