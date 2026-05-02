export type AgentFramework = 'langchain' | 'langgraph' | 'crewai' | 'openai-agents';

export type OnboardingMode = 'import' | 'create';

export type AgentPhase = 'empty' | 'modal' | 'discovering' | 'evaluating' | 'ready';

export const DEFAULT_PROJECT_NAME = 'my-agent';

export type FrameworkOption = {
  id: AgentFramework;
  name: string;
  pkg: string;
  buildSnippet: (projectName: string) => string;
};

const safeProjectName = (raw: string): string => {
  const trimmed = raw.trim();
  return trimmed.length > 0 ? trimmed : DEFAULT_PROJECT_NAME;
};

export const FRAMEWORK_OPTIONS: FrameworkOption[] = [
  {
    id: 'langchain',
    name: 'LangChain',
    pkg: 'opentracy-langchain',
    buildSnippet: (project) => `from opentracy import register
from opentracy.langchain import LangChainInstrumentor

provider = register(project="${safeProjectName(project)}")
LangChainInstrumentor().instrument(tracer_provider=provider)

# Use LangChain as usual — every chain, tool, and LLM call is traced.`,
  },
  {
    id: 'langgraph',
    name: 'LangGraph',
    pkg: 'opentracy-langgraph',
    buildSnippet: (project) => `from opentracy import register
from opentracy.langgraph import LangGraphInstrumentor

provider = register(project="${safeProjectName(project)}")
LangGraphInstrumentor().instrument(tracer_provider=provider)

# Build your graph normally — every node, edge, and state transition is captured.`,
  },
  {
    id: 'crewai',
    name: 'CrewAI',
    pkg: 'opentracy-crewai',
    buildSnippet: (project) => `from opentracy import register
from opentracy.crewai import CrewAIInstrumentor

provider = register(project="${safeProjectName(project)}")
CrewAIInstrumentor().instrument(tracer_provider=provider)

# Run your crew — every agent decision and tool invocation is recorded.`,
  },
  {
    id: 'openai-agents',
    name: 'OpenAI Agents',
    pkg: 'opentracy-openai-agents',
    buildSnippet: (project) => `from opentracy import register
from opentracy.openai_agents import OpenAIAgentsInstrumentor

provider = register(project="${safeProjectName(project)}")
OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)

# Use the OpenAI Agents SDK as usual.`,
  },
];

export const STORAGE_KEY = 'opentracy.agents.support-bot';

export type StoredOnboarding = {
  mode: OnboardingMode;
  framework: AgentFramework | null;
  templateId: string | null;
  name: string;
  importedAt: string;
};

export function loadStoredOnboarding(): StoredOnboarding | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as StoredOnboarding;
    if (parsed && (parsed.mode === 'import' || parsed.mode === 'create')) {
      return parsed;
    }
    return null;
  } catch {
    return null;
  }
}

export function saveStoredOnboarding(value: Omit<StoredOnboarding, 'importedAt'>): void {
  if (typeof window === 'undefined') return;
  const stored: StoredOnboarding = { ...value, importedAt: new Date().toISOString() };
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
}

export function clearStoredOnboarding(): void {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(STORAGE_KEY);
}
