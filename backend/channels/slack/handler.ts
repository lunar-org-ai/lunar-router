/**
 * Slack channel — top-level router that mounts the OAuth + events
 * sub-routers + exposes per-agent status / disconnect endpoints to the
 * UI (P3.3.2 + P3.5 BYOK).
 */

import { Hono } from 'hono'
import {
  clearAppCredentials,
  maskClientSecret,
  readAppCredentials,
  writeAppCredentials,
  type SlackAppCredentials,
} from './app_credentials'
import { loadSlackOAuthConfig, SlackConfigError } from './config'
import { eventsRouter } from './events'
import { oauthRouter } from './oauth'
import { clearInstallation, readInstallation } from './storage'

export interface AgentSlackStatus {
  configured: boolean
  connected: boolean
  source: 'per-agent' | 'global' | null
  team_id: string | null
  team_name: string | null
  installer_user_id: string | null
  installed_at: string | null
  install_url: string | null
  events_url: string | null
  client_id_mask: string | null
  detail: string | null
}

export async function getAgentSlackStatus(agentId: string): Promise<AgentSlackStatus> {
  let cfg
  try {
    cfg = loadSlackOAuthConfig(agentId)
  } catch (e) {
    const detail = e instanceof SlackConfigError ? e.message : 'Slack not configured'
    // Even when not configured, show PUBLIC_BASE_URL if set so the
    // operator knows what install URL would look like.
    return {
      configured: false,
      connected: false,
      source: null,
      team_id: null,
      team_name: null,
      installer_user_id: null,
      installed_at: null,
      install_url: null,
      events_url: null,
      client_id_mask: null,
      detail,
    }
  }

  const inst = await readInstallation(agentId)
  return {
    configured: true,
    connected: inst !== null,
    source: cfg.source,
    team_id: inst?.team_id ?? null,
    team_name: inst?.team_name ?? null,
    installer_user_id: inst?.installer_user_id ?? null,
    installed_at: inst?.installed_at ?? null,
    install_url: `${cfg.publicBaseUrl}/slack/install?agent_id=${encodeURIComponent(agentId)}`,
    events_url: `${cfg.publicBaseUrl}/slack/events`,
    client_id_mask: maskClientSecret(cfg.clientId),
    detail: null,
  }
}

export async function disconnectAgentSlack(agentId: string): Promise<void> {
  // Allow disconnect even if config can no longer load — the installation
  // record is what we're clearing here.
  await clearInstallation(agentId)
}

// ─── Per-agent app credentials ─────────────────────────────────────

export interface AgentSlackCredentialsView {
  set: boolean
  client_id_mask: string | null
  signing_secret_mask: string | null
  saved_at: string | null
}

export async function getAgentSlackCredentials(
  agentId: string,
): Promise<AgentSlackCredentialsView> {
  const creds = await readAppCredentials(agentId)
  if (!creds) {
    return { set: false, client_id_mask: null, signing_secret_mask: null, saved_at: null }
  }
  return {
    set: true,
    client_id_mask: maskClientSecret(creds.client_id),
    signing_secret_mask: maskClientSecret(creds.signing_secret),
    saved_at: creds.saved_at,
  }
}

export async function putAgentSlackCredentials(
  agentId: string,
  body: { client_id: string; client_secret: string; signing_secret: string },
): Promise<AgentSlackCredentialsView> {
  if (!body.client_id?.trim() || !body.client_secret?.trim() || !body.signing_secret?.trim()) {
    throw new Error('client_id, client_secret, and signing_secret are all required')
  }
  await writeAppCredentials(agentId, {
    client_id: body.client_id.trim(),
    client_secret: body.client_secret.trim(),
    signing_secret: body.signing_secret.trim(),
  })
  return getAgentSlackCredentials(agentId)
}

export async function deleteAgentSlackCredentials(agentId: string): Promise<void> {
  await clearAppCredentials(agentId)
}

// Top-level router mounted at /v1/slack
export const slackRouter = new Hono()
slackRouter.route('/', oauthRouter)
slackRouter.route('/', eventsRouter)
