/**
 * Slack OAuth app config (P3.3.2 + P3.5 BYOK).
 *
 * Each agent can bring its own Slack app credentials, pasted into the
 * UI and stored at agents/<id>/integrations/slack_app.json. When
 * resolving config for an agent, we prefer those over global env vars:
 *
 *   - per-agent slack_app.json  → preferred
 *   - global env (SLACK_CLIENT_ID / SLACK_CLIENT_SECRET / SLACK_SIGNING_SECRET) → fallback
 *
 * PUBLIC_BASE_URL stays global (it's where Slack reaches us) and is
 * required regardless of credential source.
 */

import { readAppCredentialsSync } from './app_credentials'

export interface SlackOAuthConfig {
  clientId: string
  clientSecret: string
  signingSecret: string
  publicBaseUrl: string
  uiBaseUrl: string
  scopes: string
  source: 'per-agent' | 'global'
}

export class SlackConfigError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'SlackConfigError'
  }
}

// Scopes we need for the minimal "answer DMs + mentions" surface.
const REQUIRED_SCOPES = [
  'app_mentions:read',
  'chat:write',
  'im:history',
  'im:read',
  'im:write',
  'team:read',
].join(',')

function trimTrail(s: string): string {
  return s.replace(/\/+$/, '')
}

/**
 * Resolve OAuth config for a given agent, preferring per-agent creds
 * over global env vars. Pass `agentId=null` to force the global path
 * (used by callers that don't know which agent yet — like the env-only
 * health check).
 */
export function loadSlackOAuthConfig(agentId: string | null = null): SlackOAuthConfig {
  const publicBaseUrl = trimTrail(process.env.PUBLIC_BASE_URL ?? '')
  const uiBaseUrl = trimTrail(process.env.UI_BASE_URL ?? '') || publicBaseUrl

  let clientId = ''
  let clientSecret = ''
  let signingSecret = ''
  let source: 'per-agent' | 'global' = 'global'

  if (agentId) {
    const perAgent = readAppCredentialsSync(agentId)
    if (perAgent) {
      clientId = perAgent.client_id
      clientSecret = perAgent.client_secret
      signingSecret = perAgent.signing_secret
      source = 'per-agent'
    }
  }

  if (!clientId) clientId = process.env.SLACK_CLIENT_ID ?? ''
  if (!clientSecret) clientSecret = process.env.SLACK_CLIENT_SECRET ?? ''
  if (!signingSecret) signingSecret = process.env.SLACK_SIGNING_SECRET ?? ''

  const missing: string[] = []
  if (!clientId) missing.push('client_id')
  if (!clientSecret) missing.push('client_secret')
  if (!signingSecret) missing.push('signing_secret')
  if (!publicBaseUrl) missing.push('PUBLIC_BASE_URL')

  if (missing.length > 0) {
    throw new SlackConfigError(
      `Slack is not configured. Missing: ${missing.join(', ')}. ` +
        `Paste app credentials in the agent's Slack panel, or set the env vars on the backend.`,
    )
  }

  return {
    clientId,
    clientSecret,
    signingSecret,
    publicBaseUrl,
    uiBaseUrl,
    scopes: REQUIRED_SCOPES,
    source,
  }
}

/**
 * True iff EITHER global env vars are set OR the given agent has its
 * own per-agent app creds. Pass `agentId=null` to check the env-only
 * path (used for the global "is Slack reachable at all" status).
 */
export function isSlackConfigured(agentId: string | null = null): boolean {
  try {
    loadSlackOAuthConfig(agentId)
    return true
  } catch {
    return false
  }
}
