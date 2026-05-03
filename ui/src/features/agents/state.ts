import type { AgentFramework } from '@/features/agents/types';

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

export const STORAGE_KEY = 'opentracy.agents.list';

export type StoredAgent = {
  slug: string;
  name: string;
  framework: AgentFramework;
  importedAt: string;
};

export function loadStoredAgents(): StoredAgent[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (entry): entry is StoredAgent =>
        entry &&
        typeof entry.slug === 'string' &&
        typeof entry.name === 'string' &&
        typeof entry.framework === 'string' &&
        typeof entry.importedAt === 'string'
    );
  } catch {
    return [];
  }
}

export function saveStoredAgents(agents: StoredAgent[]): void {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(agents));
}

export function clearStoredAgents(): void {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(STORAGE_KEY);
}

export function slugify(input: string): string {
  return (
    input
      .toLowerCase()
      .trim()
      .replace(/\s+/g, '-')
      .replace(/[^a-z0-9-]/g, '')
      .replace(/-+/g, '-')
      .replace(/^-+|-+$/g, '') || DEFAULT_PROJECT_NAME
  );
}
