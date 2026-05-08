/**
 * Streams against POST /v1/chat/stream using the AI SDK Data Stream Protocol.
 * v0.14 of @assistant-ui/react is headless (primitives only) so we assemble
 * the Thread shell here. Styles live in ui/src/styles.css under `.aui-*`.
 */

import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from '@assistant-ui/react';
import { AssistantChatTransport, useChatRuntime } from '@assistant-ui/react-ai-sdk';

const ChatMessage = () => (
  <>
    <MessagePrimitive.If user>
      <MessagePrimitive.Root className="aui-msg aui-msg--user">
        <MessagePrimitive.Parts />
      </MessagePrimitive.Root>
    </MessagePrimitive.If>
    <MessagePrimitive.If assistant>
      <MessagePrimitive.Root className="aui-msg aui-msg--assistant">
        <MessagePrimitive.Parts />
      </MessagePrimitive.Root>
    </MessagePrimitive.If>
  </>
);

export const ChatAssistantUi = () => {
  const runtime = useChatRuntime({
    transport: new AssistantChatTransport({ api: '/v1/chat/stream' }),
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="content">
        <h1 className="page-title">Lab — assistant-ui</h1>
        <p className="page-sub">
          Streaming chat with tool calls against the OpenTracy ledger.
        </p>

        <div className="card" style={{ marginTop: 16, padding: 0, overflow: 'hidden' }}>
          <ThreadPrimitive.Root className="aui-thread">
            <ThreadPrimitive.Viewport className="aui-viewport">
              <ThreadPrimitive.Empty>
                <p className="dim" style={{ margin: 0 }}>
                  Ask about lessons, versions, traces, or metrics. Try:{' '}
                  <em>“What lessons are pending review?”</em>
                </p>
              </ThreadPrimitive.Empty>
              <ThreadPrimitive.Messages components={{ Message: ChatMessage }} />
            </ThreadPrimitive.Viewport>

            <ComposerPrimitive.Root className="aui-composer">
              <ComposerPrimitive.Input
                className="aui-composer-input"
                placeholder="Ask the OpenTracy ledger…"
                rows={1}
              />
              <ComposerPrimitive.Send className="aui-composer-send">Send</ComposerPrimitive.Send>
            </ComposerPrimitive.Root>
          </ThreadPrimitive.Root>
        </div>
      </div>
    </AssistantRuntimeProvider>
  );
};
