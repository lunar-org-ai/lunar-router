export type AgentFramework = 'langchain' | 'langgraph' | 'crewai' | 'openai-agents';

export type AgentPhase = 'empty' | 'modal' | 'discovering' | 'evaluating' | 'ready';

export type FrameworkOption = {
  id: AgentFramework;
  name: string;
  pkg: string;
  snippet: string;
};

export const FRAMEWORK_OPTIONS: FrameworkOption[] = [
  {
    id: 'langchain',
    name: 'LangChain',
    pkg: 'opentracy-langchain',
    snippet: `from opentracy import register
from opentracy.langchain import LangChainInstrumentor

provider = register(project="support-bot")
LangChainInstrumentor().instrument(tracer_provider=provider)

# Use LangChain as usual — every chain, tool, and LLM call is traced.`,
  },
  {
    id: 'langgraph',
    name: 'LangGraph',
    pkg: 'opentracy-langgraph',
    snippet: `from opentracy import register
from opentracy.langgraph import LangGraphInstrumentor

provider = register(project="support-bot")
LangGraphInstrumentor().instrument(tracer_provider=provider)

# Build your graph normally — every node, edge, and state transition is captured.`,
  },
  {
    id: 'crewai',
    name: 'CrewAI',
    pkg: 'opentracy-crewai',
    snippet: `from opentracy import register
from opentracy.crewai import CrewAIInstrumentor

provider = register(project="support-bot")
CrewAIInstrumentor().instrument(tracer_provider=provider)

# Run your crew — every agent decision and tool invocation is recorded.`,
  },
  {
    id: 'openai-agents',
    name: 'OpenAI Agents',
    pkg: 'opentracy-openai-agents',
    snippet: `from opentracy import register
from opentracy.openai_agents import OpenAIAgentsInstrumentor

provider = register(project="support-bot")
OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)

# Use the OpenAI Agents SDK as usual.`,
  },
];

export const STORAGE_KEY = 'opentracy.agents.support-bot';

export type StoredImport = {
  framework: AgentFramework;
  importedAt: string;
};

export function loadStoredImport(): StoredImport | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as StoredImport;
    if (parsed && typeof parsed.framework === 'string') return parsed;
    return null;
  } catch {
    return null;
  }
}

export function saveStoredImport(framework: AgentFramework): void {
  if (typeof window === 'undefined') return;
  const value: StoredImport = { framework, importedAt: new Date().toISOString() };
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
}

export function clearStoredImport(): void {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(STORAGE_KEY);
}
