import { Brain, Database, Scan, Search, Sparkles } from 'lucide-react';
import type { GeneratePhase, TopicPhase } from '../types';

export const DEFAULT_AUTO_COLLECT = {
  source_model: 'openai/gpt-4o',
  max_samples: 200,
  collection_interval_minutes: 60,
  curation_config: {
    quality_threshold: 0.5,
    selection_rate: 0.3,
    agent_weights: { quality: 0.4, diversity: 0.3, difficulty: 0.3 },
  },
} as const;

export const TOKEN_BUCKETS = [
  { label: '0-50', min: 0, max: 50 },
  { label: '51-100', min: 51, max: 100 },
  { label: '101-200', min: 101, max: 200 },
  { label: '201-500', min: 201, max: 500 },
  { label: '500+', min: 501, max: Infinity },
];

export const INTERVAL_OPTIONS = [
  { label: '15 min', value: 15 },
  { label: '30 min', value: 30 },
  { label: '1 hour', value: 60 },
  { label: '2 hours', value: 120 },
  { label: '6 hours', value: 360 },
  { label: '24 hours', value: 1440 },
];

export const SAMPLES_PER_PAGE_OPTIONS = [50, 100, 200, 500];

export const GENERATE_PHASES: { id: GeneratePhase; label: string; icon: typeof Scan }[] = [
  { id: 'preparing', label: 'Preparing', icon: Brain },
  { id: 'generating', label: 'Generating', icon: Sparkles },
  { id: 'reviewing', label: 'Reviewing', icon: Search },
  { id: 'building', label: 'Building', icon: Database },
];

export const TOPIC_PHASES: { id: TopicPhase; label: string; icon: typeof Scan }[] = [
  { id: 'scanning', label: 'Scanning traces', icon: Scan },
  { id: 'analyzing', label: 'Embedding analysis', icon: Brain },
  { id: 'matching', label: 'AI classification', icon: Search },
  { id: 'building', label: 'Building dataset', icon: Database },
];

export const DATE_RANGES = [
  { label: '24h', value: '1d' },
  { label: '7d', value: '7d' },
  { label: '30d', value: '30d' },
  { label: 'All', value: 'all' },
] as const;

export const TRACES_PAGE_SIZE = 15;
export const TRACES_PAGE_SIZE_OPTIONS = [10, 15, 25, 50, 100];
export const GENERATE_COUNT = { min: 10, max: 500, default: 50 };

export const SOURCE_OPTIONS = [
  { value: 'all', label: 'All Sources' },
  { value: 'manual', label: 'Manual' },
  { value: 'instruction', label: 'Auto Collect' },
  { value: 'auto_collected', label: 'Teacher Traces' },
  { value: 'imported', label: 'Imported' },
  { value: 'synthetic', label: 'Generated' },
] as const;
