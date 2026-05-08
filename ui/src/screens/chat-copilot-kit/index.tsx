/**
 * Lab experiment: CopilotKit chat surface.
 * Streaming chat backed by /v1/chat/copilot (CopilotRuntime v2 endpoint).
 */

import { CopilotChat, CopilotKit } from '@copilotkit/react-core/v2'
import '@copilotkit/react-ui/v2/styles.css'

export const ChatCopilotKit = () => (
  <div className="content">
    <h1 className="page-title">Lab — CopilotKit</h1>
    <p className="page-sub">Streaming chat with tool calls against the OpenTracy ledger.</p>
    <CopilotKit runtimeUrl="/v1/chat/copilot" useSingleEndpoint={false}>
      <CopilotChat />
    </CopilotKit>
  </div>
)
