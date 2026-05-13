/**
 * Slack channel — top-level router that mounts the OAuth + events
 * sub-routers + exposes per-agent status / disconnect endpoints to the
 * UI (P3.3.2).
 */

import { Hono } from 'hono'
import { isSlackConfigured, loadSlackOAuthConfig, SlackConfigError } from './config'
import { eventsRouter } from './events'
import { oauthRouter } from './oauth'
import { clearInstallation, readInstallation } from './storage'

export interface AgentSlackStatus {
  configured: boolean
  connected: boolean
  team_id: string | null
  team_name: string | null
  installer_user_id: string | null
  installed_at: string | null
  install_url: string | null
  events_url: string | null
  detail: string | null
}

export async function getAgentSlackStatus(agentId: string): Promise<AgentSlackStatus> {
  let cfg
  try {
    cfg = loadSlackOAuthConfig()
  } catch (e) {
    const detail = e instanceof SlackConfigError ? e.message : 'Slack not configured'
    return {
      configured: false,
      connected: false,
      team_id: null,
      team_name: null,
      installer_user_id: null,
      installed_at: null,
      install_url: null,
      events_url: null,
      detail,
    }
  }

  const inst = await readInstallation(agentId)
  return {
    configured: true,
    connected: inst !== null,
    team_id: inst?.team_id ?? null,
    team_name: inst?.team_name ?? null,
    installer_user_id: inst?.installer_user_id ?? null,
    installed_at: inst?.installed_at ?? null,
    install_url: `${cfg.publicBaseUrl}/slack/install?agent_id=${encodeURIComponent(agentId)}`,
    events_url: `${cfg.publicBaseUrl}/slack/events`,
    detail: null,
  }
}

export async function disconnectAgentSlack(agentId: string): Promise<void> {
  if (!isSlackConfigured()) {
    throw new SlackConfigError('Slack not configured')
  }
  await clearInstallation(agentId)
}

// Top-level router mounted at /v1/slack
export const slackRouter = new Hono()
slackRouter.route('/', oauthRouter)
slackRouter.route('/', eventsRouter)
