import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import { logger } from 'hono/logger'
import { apiKeyAuth } from '../auth/api_key'
import { isMultiTenantEnabled } from '../auth/feature'
import { authSessionRouter } from '../auth/session'
import { tenantAuth } from '../auth/tenant'
import { adminRouter } from '../channels/admin/handler'
import { agentRouter } from '../channels/agent/handler'
import { agentsRouter } from '../channels/agents/handler'
import { mcpRouter } from '../channels/mcp/handler'
import { apiChannelRouter } from '../channels/api/handler'
import { slackRouter } from '../channels/slack/handler'
import { whatsappRouter } from '../channels/whatsapp/handler'
import { widgetPublicRouter } from '../channels/widget/handler'
import { datasetRouter } from '../channels/dataset/handler'
import { billingRouter } from '../channels/billing/handler'
import { evalsRouter } from '../channels/evals/handler'
import { introspectRouter } from '../channels/introspect/handler'
import { lessonsRouter } from '../channels/lessons/handler'
import { metricsRouter } from '../channels/metrics/handler'
import { onboardingRouter } from '../channels/onboarding/handler'
import { policyRouter } from '../channels/policy/handler'
import { routerRouter } from '../channels/router/handler'
import { sessionsRouter, tracesRouter } from '../channels/traces/handler'
import { versionsRouter } from '../channels/versions/handler'
import { webhookRouter } from '../channels/webhook/handler'

const app = new Hono()

app.use('*', logger())

app.get('/', (c) =>
  c.json({
    name: 'opentracy-backend',
    version: '0.0.1',
    description: 'TS gateway proxying to runtime',
  }),
)

app.get('/health', (c) => c.json({ status: 'ok' }))

// Slack OAuth + events live outside the /v1/* apiKeyAuth chain because
// Slack itself drives these requests (browser redirect for install, the
// signed events webhook for messages). Each endpoint authenticates
// itself: OAuth via state cookie, events via signature.
app.route('/slack', slackRouter)
// Twilio WhatsApp webhook. Same pattern — Twilio drives it, signs each
// request with HMAC-SHA1 over URL+params; verified inside the handler.
app.route('/whatsapp', whatsappRouter)
// Web widget — embed JS + public inbound messages. Origin-gated, no auth.
app.route('/widget', widgetPublicRouter)
// MCP HTTP/SSE — customer Claude Code CLI connects here with its
// per-tenant Bearer. The runtime authenticates directly; no gateway
// auth layer (would interfere with the per-tenant token).
app.route('/mcp', mcpRouter)

// Auth chain.
//
// OSS local mode (default): a single api-key gate for every /v1/*
// route, same as before P16.1. Operators wire a comma-separated
// BACKEND_API_KEYS env and that's all there is to it.
//
// Hosted/infra mode (OPENTRACY_MULTI_TENANT=1):
//   - /v1/admin/*  → adminAuth (same BACKEND_API_KEYS, but ONLY here)
//   - /v1/*        → tenantAuth (resolves otrcy_live_<…> → tenant_id,
//                   forwards x-tenant-id to the runtime)
// Per-channel proxy code uses `proxyHeaders(c)` to thread the tenant
// header through. In OSS mode that helper returns {} and nothing
// extra is added.
// P16.7 — /v1/auth/* is the public exchange endpoint that hands the
// browser a tenant Bearer in return for a verified Firebase ID token.
// It MUST be mounted before the auth middlewares below so it's not
// gated by them; chicken-and-egg otherwise.
app.route('/v1/auth', authSessionRouter)

if (isMultiTenantEnabled()) {
  app.use('/v1/admin/*', apiKeyAuth)
  app.use('/v1/*', async (c, next) => {
    // Skip tenantAuth for /v1/admin/* (gated by apiKeyAuth above),
    // /v1/auth/* (the public session exchange), and /v1/api/* (the
    // per-agent public chat endpoint; it auths against the agent's
    // own ot_* token, not a tenant Bearer).
    if (
      c.req.path.startsWith('/v1/admin/') ||
      c.req.path.startsWith('/v1/auth/') ||
      c.req.path.startsWith('/v1/api/')
    ) {
      return next()
    }
    return tenantAuth(c, next)
  })
} else {
  app.use('/v1/*', async (c, next) => {
    if (c.req.path.startsWith('/v1/auth/')) return next()
    return apiKeyAuth(c, next)
  })
}
// Operator-only tenant admin (gated by apiKeyAuth above when flag on).
// In OSS mode the routes still exist but require the same BACKEND_API_KEYS
// gate as every other /v1/* route — harmless because nothing actually
// activates the multi-tenant code paths without the flag.
app.route('/v1/admin', adminRouter)
app.route('/v1/webhook', webhookRouter)
app.route('/v1/introspect', introspectRouter)
app.route('/v1/versions', versionsRouter)
app.route('/v1/lessons', lessonsRouter)
app.route('/v1/metrics', metricsRouter)
app.route('/v1/policy', policyRouter)
app.route('/v1/agent', agentRouter)
app.route('/v1/agents', agentsRouter)
app.route('/v1/api', apiChannelRouter)
app.route('/v1/traces', tracesRouter)
app.route('/v1/sessions', sessionsRouter)
app.route('/v1/evals', evalsRouter)
app.route('/v1/router', routerRouter)
app.route('/v1/datasets', datasetRouter)
app.route('/v1/onboarding', onboardingRouter)
app.route('/v1/billing', billingRouter)

// Convention: 8001 = python runtime, 8002 = ts backend.
const port = Number(process.env.PORT ?? 8002)

serve({ fetch: app.fetch, port }, (info) => {
  console.log(`backend listening on http://127.0.0.1:${info.port}`)
})
