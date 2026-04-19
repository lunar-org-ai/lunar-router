import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import {
  Send,
  Copy,
  Check,
  ChevronDown,
  Loader2,
  Search,
  Plus,
  Trash2,
  MessageSquare,
  Clock,
  DollarSign,
  X,
  History,
} from 'lucide-react';
import type { ConversationMessage, StreamChunk } from '../services/conversationsService';
import { useConversationsService } from '../services/conversationsService';
import { useEvaluationsService } from '../services/evaluationsService';
import { useUser } from '../contexts/UserContext';
import { toast } from 'sonner';
import { Dialog, DialogContent } from '@/components/ui/dialog';
import { cn } from '@/lib/utils';
import { getModelIcon, getModelCategory, getProviderIconByBackend } from '../utils/modelUtils';
import { MarkdownRenderer } from '../components/MarkdownRenderer';
import { MODEL_ICONS } from '../constants/models';
import { PageHeader } from '../components/shared/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { useTutorialStep } from '@/components/Tutorial';

interface AvailableModel {
  id: string;
  name: string;
  provider: string;
  type: 'external' | 'deployment' | 'registered' | 'deployed';
  available: boolean;
  status?: string;
  providers?: string[];
}

interface ModelPanel {
  id: string;
  modelId: string;
  conversationId: string | null;
  messages: ConversationMessage[];
  isLoading: boolean;
  streamingContent: string;
  error: string | null;
}

interface PlaygroundSession {
  id: string;
  title: string;
  panels: { modelId: string; conversationId: string }[];
  createdAt: Date;
}

// Check if model is provided by OpenTracy (deployment or opentracy provider)
function isOpentracyModel(model: AvailableModel): boolean {
  const providerLower = model.provider?.toLowerCase() || '';
  return model.type === 'deployment' || providerLower === 'opentracy' || providerLower === 'deployment';
}

// Check if model is a Bedrock model (based on provider or providers array only)
// If model has multiple providers and one is "bedrock", bedrock takes precedence
function isBedrockModelFromAvailable(model: AvailableModel): boolean {
  // Check if bedrock is in the providers array (takes precedence)
  const hasBedrockInProviders =
    model.providers?.some((p) => p.toLowerCase() === 'bedrock') || false;

  if (hasBedrockInProviders) {
    return true;
  }

  // Check the main provider field
  const providerLower = model.provider?.toLowerCase() || '';
  return providerLower === 'bedrock' || providerLower === 'amazon' || providerLower === 'aws';
}

// Get icons for a model (returns both provider and bedrock icon if applicable)
function getModelIconsFromAvailable(model: AvailableModel): {
  providerIcon: string;
  bedrockIcon?: string;
  isBedrock: boolean;
  isOpentracy: boolean;
} {
  const isOpentracy = isOpentracyModel(model);
  const isBedrock = isBedrockModelFromAvailable(model);

  if (isOpentracy) {
    return {
      providerIcon: MODEL_ICONS.opentracyIcon,
      isBedrock: false,
      isOpentracy: true,
    };
  }

  if (isBedrock) {
    // For Bedrock models, show only the Bedrock icon
    return {
      providerIcon: MODEL_ICONS.bedrockIcon,
      isBedrock: true,
      isOpentracy: false,
    };
  }

  return {
    providerIcon: getProviderIconByBackend(model.provider, model.id),
    isBedrock: false,
    isOpentracy: false,
  };
}

// Get display provider name
function getProviderDisplayName(model: AvailableModel): string {
  if (isOpentracyModel(model)) {
    return 'OpenTracy';
  }
  if (isBedrockModelFromAvailable(model)) {
    return 'AWS Bedrock';
  }
  const providerLower = model.provider?.toLowerCase() || '';
  const providerNames: Record<string, string> = {
    openai: 'OpenAI',
    anthropic: 'Anthropic',
    google: 'Google',
    mistral: 'Mistral',
    meta: 'Meta',
    cohere: 'Cohere',
    deepseek: 'DeepSeek',
    groq: 'Groq',
    together: 'Together',
    perplexity: 'Perplexity',
    sambanova: 'SambaNova',
    cerebras: 'Cerebras',
    bedrock: 'AWS Bedrock',
  };
  return providerNames[providerLower] || model.provider;
}

function getFirstAvailableModelsSortedByProvider(
  models: AvailableModel[],
  count: number
): AvailableModel[] {
  const availableOnly = models.filter((m) => m.available);

  const sorted = [...availableOnly].sort((a, b) => {
    const providerA = getProviderDisplayName(a);
    const providerB = getProviderDisplayName(b);

    // OpenTracy always comes first
    if (providerA === 'OpenTracy' && providerB !== 'OpenTracy') return -1;
    if (providerA !== 'OpenTracy' && providerB === 'OpenTracy') return 1;

    // If both are same provider, sort by model name
    if (providerA === providerB) {
      return a.name.localeCompare(b.name);
    }

    // Otherwise sort providers alphabetically
    return providerA.localeCompare(providerB);
  });

  return sorted.slice(0, count);
}

// Get display model name (with opentracy/ or bedrock/ prefix for special models)
function getModelDisplayName(model: AvailableModel): string {
  if (isOpentracyModel(model)) {
    return `opentracy/${model.name}`;
  }
  if (isBedrockModelFromAvailable(model)) {
    return `bedrock/${model.name}`;
  }
  return model.name;
}

function ModelIconDisplay({
  model,
  className = 'w-5 h-5',
}: {
  model: AvailableModel;
  className?: string;
}) {
  const icons = getModelIconsFromAvailable(model);

  return (
    <img
      src={icons.providerIcon}
      alt={getProviderDisplayName(model)}
      className={className}
      onError={(e) => {
        e.currentTarget.style.display = 'none';
      }}
    />
  );
}

// Legacy version for when we only have modelId
function ModelIconDisplayById({
  modelId,
  className = 'w-5 h-5',
}: {
  modelId: string;
  className?: string;
}) {
  const iconSrc = getModelIcon(modelId);
  return (
    <img
      src={iconSrc}
      alt={getModelCategory(modelId)}
      className={className}
      onError={(e) => {
        e.currentTarget.style.display = 'none';
      }}
    />
  );
}

export default function Compare() {
  const [sessions, setSessions] = useState<PlaygroundSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [panels, setPanels] = useState<ModelPanel[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [modelSearch, setModelSearch] = useState('');
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [showHistory, setShowHistory] = useState(false);

  // Modal state for model selection
  const [modelModal, setModelModal] = useState<{
    open: boolean;
    panelId: string | null;
    isNew: boolean;
  }>({ open: false, panelId: null, isNew: false });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const {
    createConversation,
    listMessages,
    sendMessage,
    sendMessageStream,
    deleteConversation: deleteConversationAPI,
  } = useConversationsService();
  const { listAvailableModels } = useEvaluationsService();
  const { accessToken } = useUser();

  const isAnyLoading = useMemo(() => panels.some((p) => p.isLoading), [panels]);
  const hasMessages = useMemo(() => panels.some((p) => p.messages.length > 0), [panels]);

  useTutorialStep(2, hasMessages);

  // Fetch models
  useEffect(() => {
    if (!accessToken) return;
    const fetchModels = async () => {
      setLoadingModels(true);
      try {
        const models = await listAvailableModels(accessToken);
        const available = models.filter((m) => m.available);
        setAvailableModels(available);
      } catch (err) {
        console.error('Failed to fetch models:', err);
        showToastMessage('Failed to load models', 'error');
      } finally {
        setLoadingModels(false);
      }
    };
    fetchModels();
  }, [accessToken, listAvailableModels]);

  useEffect(() => {
    if (panels.length === 0 && !activeSessionId && availableModels.length > 0) {
      const defaultModels = getFirstAvailableModelsSortedByProvider(availableModels, 2).map(
        (m) => m.id
      );
      setPanels(
        defaultModels.map((modelId, index) => ({
          id: `panel-${Date.now()}-${index}`,
          modelId,
          conversationId: null,
          messages: [],
          isLoading: false,
          streamingContent: '',
          error: null,
        }))
      );
    }
  }, [availableModels, panels.length, activeSessionId]);

  // Load sessions from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem('playground-sessions');
      if (saved) {
        const parsed = JSON.parse(saved);
        setSessions(
          parsed.map((s: PlaygroundSession) => ({
            ...s,
            createdAt: new Date(s.createdAt),
          }))
        );
      }
    } catch (e) {
      console.error('Failed to parse sessions:', e);
    }
  }, []);

  // Save sessions to localStorage
  useEffect(() => {
    if (sessions.length > 0) {
      try {
        localStorage.setItem('playground-sessions', JSON.stringify(sessions));
      } catch (e) {
        console.error('Failed to save sessions:', e);
      }
    }
  }, [sessions]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [panels]);

  const showToastMessage = (msg: string, type: 'success' | 'error' | 'warning' | 'info') => {
    toast[type](msg);
  };

  const deduplicatedModels = useMemo(() => {
    return availableModels.filter((m) => {
      // If the main provider is bedrock, always keep it
      if (m.provider?.toLowerCase() === 'bedrock') {
        return true;
      }

      const hasBedrockInProviders =
        m.providers?.some((p) => p.toLowerCase() === 'bedrock') || false;

      if (hasBedrockInProviders) {
        return false;
      }

      // Keep all other models
      return true;
    });
  }, [availableModels]);

  const filteredModels = useMemo(() => {
    if (!modelSearch) return deduplicatedModels;
    const s = modelSearch.toLowerCase();
    return deduplicatedModels.filter(
      (m) =>
        m.name.toLowerCase().includes(s) ||
        m.id.toLowerCase().includes(s) ||
        m.provider.toLowerCase().includes(s) ||
        (isOpentracyModel(m) && 'opentracy'.includes(s)) ||
        (isBedrockModelFromAvailable(m) && 'bedrock'.includes(s))
    );
  }, [deduplicatedModels, modelSearch]);

  const groupedModels = useMemo(() => {
    const g: Record<string, AvailableModel[]> = {};
    filteredModels.forEach((m) => {
      // Use provider display name for grouping
      const category = getProviderDisplayName(m);
      if (!g[category]) g[category] = [];
      g[category].push(m);
    });
    return g;
  }, [filteredModels]);

  const getModelData = useCallback(
    (id: string) => availableModels.find((m) => m.id === id),
    [availableModels]
  );

  const loadSession = useCallback(
    async (session: PlaygroundSession) => {
      if (!accessToken) return;
      setLoadingSessions(true);
      setShowHistory(false);

      try {
        const loaded: ModelPanel[] = [];

        for (const p of session.panels) {
          try {
            const response = await listMessages(accessToken, p.conversationId);
            console.log('Loaded messages for', p.modelId, ':', response.messages.length);
            loaded.push({
              id: `panel-${Date.now()}-${p.modelId}`,
              modelId: p.modelId,
              conversationId: p.conversationId,
              messages: response.messages || [],
              isLoading: false,
              streamingContent: '',
              error: null,
            });
          } catch (e) {
            console.error('Failed to load messages for panel:', p.modelId, e);
            // Still add the panel but with empty messages and an error
            loaded.push({
              id: `panel-${Date.now()}-${p.modelId}`,
              modelId: p.modelId,
              conversationId: p.conversationId,
              messages: [],
              isLoading: false,
              streamingContent: '',
              error: 'Could not load messages',
            });
          }
        }

        if (loaded.length > 0) {
          setActiveSessionId(session.id);
          setPanels(loaded);
        } else {
          showToastMessage('Could not load session', 'error');
        }
      } catch (e) {
        console.error('Failed to load session:', e);
        showToastMessage('Failed to load session', 'error');
      } finally {
        setLoadingSessions(false);
      }
    },
    [accessToken, listMessages]
  );

  const deleteSession = useCallback(
    async (id: string) => {
      const session = sessions.find((s) => s.id === id);
      if (!session) return;

      if (accessToken) {
        for (const p of session.panels) {
          try {
            await deleteConversationAPI(accessToken, p.conversationId);
          } catch (e) {
            console.error(e);
          }
        }
      }

      setSessions((prev) => prev.filter((s) => s.id !== id));

      if (activeSessionId === id) {
        setActiveSessionId(null);
        if (availableModels.length > 0) {
          const defaultModels = getFirstAvailableModelsSortedByProvider(availableModels, 2).map(
            (m) => m.id
          );
          setPanels(
            defaultModels.map((modelId, index) => ({
              id: `panel-${Date.now()}-${index}`,
              modelId,
              conversationId: null,
              messages: [],
              isLoading: false,
              streamingContent: '',
              error: null,
            }))
          );
        } else {
          setPanels([]);
        }
      }
    },
    [accessToken, activeSessionId, deleteConversationAPI, sessions, availableModels]
  );

  const addModelPanel = useCallback((modelId: string) => {
    setPanels((prev) => [
      ...prev,
      {
        id: `panel-${Date.now()}-${modelId}`,
        modelId,
        conversationId: null,
        messages: [],
        isLoading: false,
        streamingContent: '',
        error: null,
      },
    ]);
    setModelModal({ open: false, panelId: null, isNew: false });
    setModelSearch('');
  }, []);

  const removeModelPanel = useCallback(
    async (panelId: string) => {
      if (panels.length <= 1) {
        showToastMessage('At least one model required', 'warning');
        return;
      }
      const panel = panels.find((p) => p.id === panelId);
      if (panel?.conversationId && accessToken) {
        try {
          await deleteConversationAPI(accessToken, panel.conversationId);
        } catch (e) {
          console.error(e);
        }
      }
      setPanels((prev) => prev.filter((p) => p.id !== panelId));
      if (activeSessionId && panel?.conversationId) {
        setSessions((prev) =>
          prev.map((s) =>
            s.id === activeSessionId
              ? {
                  ...s,
                  panels: s.panels.filter((p) => p.conversationId !== panel.conversationId),
                }
              : s
          )
        );
      }
    },
    [accessToken, activeSessionId, deleteConversationAPI, panels]
  );

  const changeModelForPanel = useCallback(
    async (panelId: string, newModelId: string) => {
      const panel = panels.find((p) => p.id === panelId);
      if (!panel) return;

      if (activeSessionId && accessToken && panel.conversationId) {
        try {
          const conv = await createConversation(accessToken, {
            model: newModelId,
            title: `Playground - ${getModelData(newModelId)?.name || newModelId}`,
          });
          try {
            await deleteConversationAPI(accessToken, panel.conversationId);
          } catch (e) {
            console.error(e);
          }
          setPanels((prev) =>
            prev.map((p) =>
              p.id === panelId
                ? {
                    ...p,
                    modelId: newModelId,
                    conversationId: conv.conversation_id,
                    messages: [],
                    error: null,
                  }
                : p
            )
          );
          setSessions((prev) =>
            prev.map((s) =>
              s.id === activeSessionId
                ? {
                    ...s,
                    panels: s.panels.map((sp) =>
                      sp.conversationId === panel.conversationId
                        ? {
                            modelId: newModelId,
                            conversationId: conv.conversation_id,
                          }
                        : sp
                    ),
                  }
                : s
            )
          );
        } catch (e) {
          console.error(e);
          showToastMessage('Failed to change model', 'error');
        }
      } else {
        setPanels((prev) =>
          prev.map((p) =>
            p.id === panelId
              ? {
                  ...p,
                  modelId: newModelId,
                  conversationId: null,
                  messages: [],
                  error: null,
                }
              : p
          )
        );
      }
      setModelModal({ open: false, panelId: null, isNew: false });
      setModelSearch('');
    },
    [accessToken, activeSessionId, createConversation, deleteConversationAPI, getModelData, panels]
  );

  const handleSend = async () => {
    if (!inputMessage.trim() || isAnyLoading || !accessToken || panels.length === 0) return;
    const content = inputMessage.trim();
    setInputMessage('');

    let currentPanels = [...panels];

    // Create conversations if needed
    if (panels.some((p) => !p.conversationId)) {
      setLoadingSessions(true);
      try {
        const updated: ModelPanel[] = [];
        const configs: PlaygroundSession['panels'] = [];

        for (const panel of panels) {
          if (!panel.conversationId) {
            const conv = await createConversation(accessToken, {
              model: panel.modelId,
              title: `Playground - ${getModelData(panel.modelId)?.name || panel.modelId}`,
            });
            updated.push({ ...panel, conversationId: conv.conversation_id });
            configs.push({
              modelId: panel.modelId,
              conversationId: conv.conversation_id,
            });
          } else {
            updated.push(panel);
            configs.push({
              modelId: panel.modelId,
              conversationId: panel.conversationId,
            });
          }
        }

        const session: PlaygroundSession = {
          id: `session-${Date.now()}`,
          title: content.slice(0, 50) + (content.length > 50 ? '...' : ''),
          panels: configs,
          createdAt: new Date(),
        };

        setSessions((prev) => [session, ...prev]);
        setActiveSessionId(session.id);
        setPanels(updated);
        currentPanels = updated;
      } catch (e) {
        console.error(e);
        showToastMessage('Failed to start session', 'error');
        setLoadingSessions(false);
        return;
      } finally {
        setLoadingSessions(false);
      }
    }

    const userMsg: ConversationMessage = {
      message_id: `user-${Date.now()}`,
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    };
    setPanels((prev) =>
      prev.map((p) => ({
        ...p,
        messages: [...p.messages, userMsg],
        isLoading: true,
        streamingContent: '',
        error: null,
      }))
    );

    await Promise.all(
      currentPanels.map(async (panel) => {
        if (!panel.conversationId) return;
        try {
          let full = '';
          let streamError: string | null = null;
          try {
            await sendMessageStream(
              accessToken,
              panel.conversationId,
              { content },
              (chunk: StreamChunk) => {
                if (chunk.type === 'content' && chunk.content) {
                  full += chunk.content;
                  setPanels((prev) =>
                    prev.map((p) => (p.id === panel.id ? { ...p, streamingContent: full } : p))
                  );
                } else if (chunk.type === 'error' && chunk.error) {
                  streamError = chunk.error;
                  setPanels((prev) =>
                    prev.map((p) =>
                      p.id === panel.id
                        ? {
                            ...p,
                            isLoading: false,
                            streamingContent: '',
                            error: chunk.error || 'Unknown error',
                          }
                        : p
                    )
                  );
                } else if (chunk.type === 'done') {
                  if (!streamError) {
                    const msg: ConversationMessage = {
                      message_id: chunk.message_id || `a-${Date.now()}`,
                      role: 'assistant',
                      content: full,
                      created_at: new Date().toISOString(),
                      model: panel.modelId,
                      latency_ms: chunk.latency_ms,
                    };
                    setPanels((prev) =>
                      prev.map((p) =>
                        p.id === panel.id
                          ? {
                              ...p,
                              messages: [...p.messages, msg],
                              isLoading: false,
                              streamingContent: '',
                            }
                          : p
                      )
                    );
                  }
                }
              },
              () => {
                streamError = 'Connection error';
              }
            );
          } catch {
            streamError = 'Stream error';
          }

          if (streamError && !full) {
            // No content received - try fallback to sync API
            try {
              const res = await sendMessage(accessToken, panel.conversationId, {
                content,
              });
              const msg: ConversationMessage = {
                message_id: res.assistant_message.message_id,
                role: 'assistant',
                content: res.assistant_message.content,
                created_at: res.assistant_message.created_at,
                model: res.assistant_message.model,
                latency_ms: res.usage.latency_ms,
                cost_usd: res.usage.total_cost_usd,
              };
              setPanels((prev) =>
                prev.map((p) =>
                  p.id === panel.id
                    ? {
                        ...p,
                        messages: [...p.messages, msg],
                        isLoading: false,
                        streamingContent: '',
                        error: null,
                      }
                    : p
                )
              );
            } catch {
              // Fallback also failed, reset loading state and show error
              setPanels((prev) =>
                prev.map((p) =>
                  p.id === panel.id
                    ? {
                        ...p,
                        isLoading: false,
                        streamingContent: '',
                        error: streamError || 'Failed to send message',
                      }
                    : p
                )
              );
            }
          } else if (streamError && full) {
            // Partial content received but stream failed - save partial and show error
            const msg: ConversationMessage = {
              message_id: `a-${Date.now()}`,
              role: 'assistant',
              content: full,
              created_at: new Date().toISOString(),
              model: panel.modelId,
            };
            setPanels((prev) =>
              prev.map((p) =>
                p.id === panel.id
                  ? {
                      ...p,
                      messages: [...p.messages, msg],
                      isLoading: false,
                      streamingContent: '',
                      error: `Response incomplete: ${streamError}`,
                    }
                  : p
              )
            );
          }
        } catch (e) {
          setPanels((prev) =>
            prev.map((p) =>
              p.id === panel.id
                ? {
                    ...p,
                    isLoading: false,
                    streamingContent: '',
                    error: e instanceof Error ? e.message : 'Error',
                  }
                : p
            )
          );
        }
      })
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const copyToClipboard = async (text: string, id: string) => {
    await navigator.clipboard.writeText(text);
    setCopiedMessageId(id);
    setTimeout(() => setCopiedMessageId(null), 2000);
  };

  const resetToNew = useCallback(() => {
    if (availableModels.length > 0) {
      const defaults = getFirstAvailableModelsSortedByProvider(availableModels, 2).map((m) => m.id);
      setPanels(
        defaults.map((modelId, i) => ({
          id: `panel-${Date.now()}-${i}`,
          modelId,
          conversationId: null,
          messages: [],
          isLoading: false,
          streamingContent: '',
          error: null,
        }))
      );
    }
    setActiveSessionId(null);
    setInputMessage('');
  }, [availableModels]);

  const openModelSelector = (panelId: string | null, isNew: boolean) => {
    setModelModal({ open: true, panelId, isNew });
    setModelSearch('');
  };

  const closeModelSelector = () => {
    setModelModal({ open: false, panelId: null, isNew: false });
    setModelSearch('');
  };

  return (
    <div className="h-screen flex flex-col bg-background">
      <PageHeader
        title="Compare"
        action={
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowHistory(!showHistory)}
              className={`flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-lg transition-colors ${showHistory ? 'bg-brand/10 text-brand' : 'text-foreground-secondary hover:bg-surface-hover'}`}
            >
              <History className="w-4 h-4" />
              History
            </button>
            <button
              onClick={resetToNew}
              className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-white bg-brand hover:bg-brand/80 rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              New
            </button>
          </div>
        }
      />

      <div className="flex-1 flex overflow-hidden relative">
        {/* History Dropdown */}
        {showHistory && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setShowHistory(false)} />
            <div className="absolute top-0 right-4 mt-2 w-72 bg-surface border border-border rounded-xl shadow-xl z-50 overflow-hidden">
              <div className="p-3 border-b border-border">
                <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider">
                  Recent Sessions
                </span>
              </div>
              <div className="max-h-80 overflow-y-auto">
                {sessions.length === 0 ? (
                  <p className="text-sm text-foreground-muted text-center py-8">No history yet</p>
                ) : (
                  <div className="p-2 space-y-1">
                    {sessions.map((s) => (
                      <div key={s.id} className="group">
                        <button
                          onClick={() => loadSession(s)}
                          className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-all ${activeSessionId === s.id ? 'bg-brand/10' : 'hover:bg-surface-hover'}`}
                        >
                          <MessageSquare className="w-4 h-4 text-foreground-muted flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-foreground truncate">
                              {s.title}
                            </p>
                            <p className="text-xs text-foreground-muted">
                              {s.panels.length} models
                            </p>
                          </div>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteSession(s.id);
                            }}
                            className="opacity-0 group-hover:opacity-100 p-1.5 text-foreground-muted hover:text-error hover:bg-error/10 rounded-md transition-all"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* Main Content */}
        <main className="flex-1 flex flex-col min-w-0">
          {loadingModels ? (
            <div className="flex gap-4 flex-1 overflow-hidden p-4">
              {[1, 2].map((i) => (
                <div key={i} className="flex-1 flex flex-col gap-3 min-w-0">
                  <div className="flex items-center gap-2 px-3 py-2 bg-background-secondary rounded-lg">
                    <Skeleton className="w-5 h-5 rounded-full" />
                    <Skeleton className="h-4 w-32 rounded-md" />
                  </div>

                  <div className="flex-1 flex flex-col gap-4 p-3">
                    <div className="flex justify-end">
                      <Skeleton className="h-10 w-2/3 rounded-xl" />
                    </div>
                    <div className="flex flex-col gap-2">
                      <Skeleton className="h-4 w-full rounded-md" />
                      <Skeleton className="h-4 w-5/6 rounded-md" />
                      <Skeleton className="h-4 w-4/6 rounded-md" />
                    </div>
                    <div className="flex justify-end">
                      <Skeleton className="h-10 w-1/2 rounded-xl" />
                    </div>
                    <div className="flex flex-col gap-2">
                      <Skeleton className="h-4 w-full rounded-md" />
                      <Skeleton className="h-4 w-3/4 rounded-md" />
                    </div>
                  </div>

                  <div className="flex items-center gap-2 px-3 pb-2">
                    <Skeleton className="h-3 w-10 rounded" />
                    <Skeleton className="h-3 w-16 rounded" />
                  </div>
                </div>
              ))}
            </div>
          ) : panels.length === 0 ? (
            <div className="flex-1 flex items-center justify-center text-foreground-muted text-sm">
              No models available
            </div>
          ) : (
            <>
              {/* Model Panels */}
              <div className="flex-1 flex overflow-x-auto">
                {panels.map((panel, idx) => {
                  const model = getModelData(panel.modelId);
                  const streaming = panel.isLoading && panel.streamingContent;

                  return (
                    <div
                      key={panel.id}
                      className={`flex-1 min-w-[320px] flex flex-col bg-surface ${idx < panels.length - 1 ? 'border-r border-border' : ''}`}
                    >
                      {/* Panel Header */}
                      <div className="flex-shrink-0 flex items-center justify-between px-4 py-3 border-b border-border">
                        <button
                          onClick={() => openModelSelector(panel.id, false)}
                          className="flex items-center gap-2 px-3 py-2 bg-background-secondary hover:bg-surface-hover rounded-lg transition-colors"
                        >
                          {model ? (
                            <ModelIconDisplay model={model} className="w-5 h-5" />
                          ) : (
                            <ModelIconDisplayById modelId={panel.modelId} className="w-5 h-5" />
                          )}
                          <span className="text-sm font-medium text-foreground max-w-[160px] truncate">
                            {model ? getModelDisplayName(model) : panel.modelId}
                          </span>
                          <ChevronDown className="w-4 h-4 text-foreground-muted" />
                        </button>
                        {panels.length > 1 && (
                          <button
                            onClick={() => removeModelPanel(panel.id)}
                            className="p-2 text-foreground-muted hover:text-error hover:bg-error/10 rounded-lg transition-colors"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        )}
                      </div>

                      {/* Messages */}
                      <div className="flex-1 overflow-y-auto">
                        {panel.messages.length === 0 && !panel.isLoading && !panel.error ? (
                          <div className="h-full flex items-center justify-center">
                            <span className="text-sm text-foreground-muted">Ready to compare</span>
                          </div>
                        ) : (
                          <div className="p-4 space-y-2">
                            {panel.messages.map((msg) => (
                              <div
                                key={msg.message_id}
                                className={msg.role === 'user' ? 'flex justify-end' : ''}
                              >
                                <div
                                  className={`max-w-[90%] ${
                                    msg.role === 'user'
                                      ? 'bg-secondary text-secondary-foreground rounded-2xl rounded-br-sm px-4 py-1.5 border border-border'
                                      : 'bg-background-secondary rounded-2xl rounded-bl-sm px-4 py-1.5'
                                  }`}
                                >
                                  {msg.role === 'assistant' ? (
                                    <div className="text-sm text-foreground">
                                      <MarkdownRenderer content={msg.content} />
                                    </div>
                                  ) : (
                                    <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                                  )}
                                  {msg.role === 'assistant' && (
                                    <div className="flex items-center gap-4 mt-2 pt-1.5 border-t border-border">
                                      {msg.latency_ms && (
                                        <span className="flex items-center gap-1 text-xs text-foreground-muted">
                                          <Clock className="w-3 h-3" />
                                          {(msg.latency_ms / 1000).toFixed(2)}s
                                        </span>
                                      )}
                                      {msg.cost_usd !== undefined && msg.cost_usd > 0 && (
                                        <span className="flex items-center gap-1 text-xs text-success">
                                          <DollarSign className="w-3 h-3" />$
                                          {msg.cost_usd.toFixed(6)}
                                        </span>
                                      )}
                                      <button
                                        onClick={() => copyToClipboard(msg.content, msg.message_id)}
                                        className="ml-auto text-foreground-muted hover:text-foreground-secondary"
                                      >
                                        {copiedMessageId === msg.message_id ? (
                                          <Check className="w-3.5 h-3.5 text-brand" />
                                        ) : (
                                          <Copy className="w-3.5 h-3.5" />
                                        )}
                                      </button>
                                    </div>
                                  )}
                                </div>
                              </div>
                            ))}

                            {/* Streaming */}
                            {streaming && (
                              <div className="max-w-[90%] bg-background-secondary rounded-2xl rounded-bl-sm px-4 py-3">
                                <div className="text-sm text-foreground">
                                  <MarkdownRenderer content={panel.streamingContent} />
                                  <span className="inline-block w-2 h-4 bg-brand animate-pulse ml-1 rounded-sm" />
                                </div>
                              </div>
                            )}

                            {/* Loading */}
                            {panel.isLoading && !streaming && (
                              <div className="flex flex-col gap-2 px-1 py-2">
                                <Skeleton className="h-4 w-full rounded-md" />
                                <Skeleton className="h-4 w-11/12 rounded-md" />
                                <Skeleton className="h-4 w-4/5 rounded-md" />
                                <Skeleton className="h-4 w-2/3 rounded-md" />
                              </div>
                            )}

                            {/* Error */}
                            {panel.error && (
                              <div className="max-w-[90%] bg-error/10 border border-error/20 rounded-xl px-4 py-3">
                                <p className="text-sm text-error">{panel.error}</p>
                              </div>
                            )}
                          </div>
                        )}
                        <div ref={messagesEndRef} />
                      </div>
                    </div>
                  );
                })}

                {/* Add Model Button */}
                <div className="flex-shrink-0 w-16 flex items-center justify-center bg-background-secondary border-l border-border">
                  <button
                    onClick={() => openModelSelector(null, true)}
                    className="p-3 text-foreground-muted hover:text-brand hover:bg-surface rounded-xl transition-colors"
                    title="Add model"
                  >
                    <Plus className="w-5 h-5" />
                  </button>
                </div>
              </div>

              {/* Input */}
              <div className="flex-shrink-0 bg-surface border-t border-border p-4">
                <div className="max-w-3xl mx-auto">
                  <div className="flex items-end gap-3 bg-background-secondary rounded-lg p-3 border border-border focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/20 transition-all">
                    <textarea
                      ref={textareaRef}
                      placeholder="Message..."
                      value={inputMessage}
                      onChange={(e) => {
                        setInputMessage(e.target.value);
                        e.target.style.height = 'auto';
                        e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px';
                      }}
                      onKeyDown={handleKeyDown}
                      rows={1}
                      disabled={isAnyLoading || loadingSessions || panels.length === 0}
                      className="flex-1 bg-transparent text-foreground text-sm resize-none focus:outline-none disabled:opacity-50 placeholder:text-foreground-muted py-1.5"
                      style={{ minHeight: '24px', maxHeight: '160px' }}
                    />
                    <button
                      onClick={handleSend}
                      disabled={
                        !inputMessage.trim() ||
                        isAnyLoading ||
                        loadingSessions ||
                        panels.length === 0
                      }
                      className="flex-shrink-0 w-9 h-9 bg-brand hover:bg-brand/80 disabled:bg-foreground-muted text-white rounded-xl flex items-center justify-center transition-colors disabled:cursor-not-allowed"
                    >
                      {isAnyLoading || loadingSessions ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Send className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </>
          )}
        </main>
      </div>

      {/* Model Selection Modal */}
      <Dialog open={modelModal.open} onOpenChange={(v) => !v && closeModelSelector()}>
        <DialogContent className={cn('p-0', 'max-w-md')} showCloseButton={false}>
          <div className="overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h2 className="text-lg font-semibold text-foreground">
                {modelModal.isNew ? 'Add Model' : 'Change Model'}
              </h2>
              <button
                onClick={closeModelSelector}
                className="p-2 text-foreground-muted hover:text-foreground rounded-lg hover:bg-surface-hover transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-4 border-b border-border">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-foreground-muted" />
                <input
                  type="text"
                  placeholder="Search models..."
                  value={modelSearch}
                  onChange={(e) => setModelSearch(e.target.value)}
                  className="w-full pl-10 pr-4 py-2.5 text-sm bg-surface border border-border rounded-lg text-foreground placeholder-foreground-muted hover:border-border-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:border-transparent"
                  autoFocus
                />
              </div>
            </div>

            <div className="max-h-80 overflow-y-auto">
              {Object.entries(groupedModels)
                .sort(([providerA], [providerB]) => {
                  // OpenTracy always comes first
                  if (providerA === 'OpenTracy' && providerB !== 'OpenTracy') return -1;
                  if (providerA !== 'OpenTracy' && providerB === 'OpenTracy') return 1;
                  // Otherwise sort alphabetically
                  return providerA.localeCompare(providerB);
                })
                .map(([provider, models]) => (
                  <div key={provider}>
                    <div className="px-4 py-2 bg-background-secondary sticky top-0">
                      <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider">
                        {provider}
                      </span>
                    </div>
                    {models.map((model) => {
                      const currentPanelModel = modelModal.panelId
                        ? panels.find((p) => p.id === modelModal.panelId)?.modelId
                        : null;
                      const selected = currentPanelModel === model.id;
                      const used = panels.some(
                        (p) => p.modelId === model.id && p.id !== modelModal.panelId
                      );

                      return (
                        <button
                          key={model.id}
                          onClick={() => {
                            if (modelModal.isNew) {
                              addModelPanel(model.id);
                            } else if (modelModal.panelId) {
                              changeModelForPanel(modelModal.panelId, model.id);
                            }
                          }}
                          disabled={used}
                          className={`w-full flex items-center gap-3 px-4 py-3 transition-all ${selected ? 'bg-accent/10' : 'hover:bg-surface-hover'} ${used ? 'opacity-40 cursor-not-allowed' : ''}`}
                        >
                          <ModelIconDisplay model={model} className="w-6 h-6" />
                          <div className="flex-1 text-left">
                            <span className="text-sm font-medium text-foreground">
                              {getModelDisplayName(model)}
                            </span>
                          </div>
                          {selected && <div className="w-2 h-2 rounded-full bg-accent" />}
                          {used && <span className="text-xs text-foreground-muted">In use</span>}
                        </button>
                      );
                    })}
                  </div>
                ))}
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
