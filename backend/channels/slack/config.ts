/**
 * Slack OAuth app config (P3.3.2).
 *
 * The operator registers ONE Slack app and brings credentials via env:
 *   - SLACK_CLIENT_ID
 *   - SLACK_CLIENT_SECRET
 *   - SLACK_SIGNING_SECRET
 *   - PUBLIC_BASE_URL    (e.g. https://yourdomain.com — used for redirect_uri)
 *   - UI_BASE_URL        (optional; defaults to PUBLIC_BASE_URL)
 *
 * Each agent's install lives at agents/<id>/integrations/slack.json so
 * the same Slack app can be installed in multiple workspaces; each
 * workspace's install attaches to the agent the operator was setting
 * up when they clicked Connect.
 */

export interface SlackOAuthConfig {
  clientId: string
  clientSecret: string
  signingSecret: string
  publicBaseUrl: string
  uiBaseUrl: string
  scopes: string
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

export function loadSlackOAuthConfig(): SlackOAuthConfig {
  const clientId = process.env.SLACK_CLIENT_ID ?? ''
  const clientSecret = process.env.SLACK_CLIENT_SECRET ?? ''
  const signingSecret = process.env.SLACK_SIGNING_SECRET ?? ''
  const publicBaseUrl = trimTrail(process.env.PUBLIC_BASE_URL ?? '')
  const uiBaseUrl = trimTrail(process.env.UI_BASE_URL ?? '') || publicBaseUrl

  const missing: string[] = []
  if (!clientId) missing.push('SLACK_CLIENT_ID')
  if (!clientSecret) missing.push('SLACK_CLIENT_SECRET')
  if (!signingSecret) missing.push('SLACK_SIGNING_SECRET')
  if (!publicBaseUrl) missing.push('PUBLIC_BASE_URL')

  if (missing.length > 0) {
    throw new SlackConfigError(
      `Slack is not configured. Set these env vars on the backend: ${missing.join(', ')}.`,
    )
  }

  return {
    clientId,
    clientSecret,
    signingSecret,
    publicBaseUrl,
    uiBaseUrl,
    scopes: REQUIRED_SCOPES,
  }
}

export function isSlackConfigured(): boolean {
  try {
    loadSlackOAuthConfig()
    return true
  } catch {
    return false
  }
}
