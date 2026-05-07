# connectors/

Outbound integrations. What the agent can talk to *out*: databases, internal APIs,
third-party services. Distinct from `backend/channels/` (inbound).

Each connector is a self-contained adapter exposing a tool registered in
`runtime/tools/`. Adding one is a code release, not an auto-improvement.
