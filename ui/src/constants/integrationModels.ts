import {
  OpenAI,
  Anthropic,
  Azure,
  DeepSeek,
  Mistral,
  Perplexity,
  Cerebras,
  Cohere,
  Groq,
  SambaNova,
  Together,
  Gemini,
  Bedrock,
  HuggingFace,
} from '@lobehub/icons';

export type ProviderModel = {
  name: string;
  key: string;
  icon: React.ComponentType<any>;
};

export const INTEGRATION_MODELS: ProviderModel[] = [
  { name: 'Anthropic', key: 'anthropic_api_key', icon: Anthropic },
  { name: 'AWS Bedrock', key: 'bedrock_api_key', icon: Bedrock },
  { name: 'Azure OpenAI', key: 'azure_api_key', icon: Azure },
  { name: 'Cerebras', key: 'cerebras_api_key', icon: Cerebras },
  { name: 'Cohere', key: 'cohere_api_key', icon: Cohere },
  { name: 'DeepSeek', key: 'deepseek_api_key', icon: DeepSeek },
  { name: 'Gemini', key: 'gemini_api_key', icon: Gemini },
  { name: 'Groq', key: 'groq_api_key', icon: Groq },
  { name: 'HuggingFace', key: 'huggingface_api_key', icon: HuggingFace },
  { name: 'Mistral', key: 'mistral_api_key', icon: Mistral },
  { name: 'OpenAI', key: 'openai_api_key', icon: OpenAI },
  { name: 'Perplexity', key: 'perplexity_api_key', icon: Perplexity },
  { name: 'SambaNova', key: 'sambanova_api_key', icon: SambaNova },
  { name: 'Together AI', key: 'togetherai_api_key', icon: Together },
];

export const getProviderFromKey = (key: string): string => {
  return key.replace('_api_key', '');
};
