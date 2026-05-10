import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import { logger } from 'hono/logger'
import { apiKeyAuth } from '../auth/api_key'
import { agentRouter } from '../channels/agent/handler'
import { evalsRouter } from '../channels/evals/handler'
import { introspectRouter } from '../channels/introspect/handler'
import { lessonsRouter } from '../channels/lessons/handler'
import { metricsRouter } from '../channels/metrics/handler'
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

// All /v1/* routes require auth and are real channels.
app.use('/v1/*', apiKeyAuth)
app.route('/v1/webhook', webhookRouter)
app.route('/v1/introspect', introspectRouter)
app.route('/v1/versions', versionsRouter)
app.route('/v1/lessons', lessonsRouter)
app.route('/v1/metrics', metricsRouter)
app.route('/v1/policy', policyRouter)
app.route('/v1/agent', agentRouter)
app.route('/v1/traces', tracesRouter)
app.route('/v1/sessions', sessionsRouter)
app.route('/v1/evals', evalsRouter)
app.route('/v1/router', routerRouter)

// Convention: 8001 = python runtime, 8002 = ts backend.
const port = Number(process.env.PORT ?? 8002)

serve({ fetch: app.fetch, port }, (info) => {
  console.log(`backend listening on http://127.0.0.1:${info.port}`)
})
