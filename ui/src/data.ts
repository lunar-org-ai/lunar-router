import type { DiffLine } from './components/Diff';
import type { LessonKind, LessonStatus } from './components/Tag';

export type TriggerKind = 'failed_traces' | 'feedback' | 'cost' | 'experiment';

export interface Trigger {
  kind: TriggerKind;
  count: number | null;
  window: string;
}

export interface Proposal {
  type: 'tool_wrapper' | 'system_prompt' | 'routing' | 'eval';
  beforeLines: DiffLine[];
  afterLines: DiffLine[];
}

export interface Eval {
  name: string;
  before: number | null;
  after: number;
  sample: number;
  scale?: number;
  isLatency?: boolean;
  currency?: boolean;
}

export interface MetricDelta {
  label: string;
  delta: number;
}

export interface Trace {
  id: string;
  preview: string;
  verdict: 'pass' | 'fail';
}

export type Decision = 'awaiting_review' | 'auto_promoted' | 'human_rejected';

export interface Lesson {
  id: string;
  version: string;
  pendingApproval?: boolean;
  kind: LessonKind;
  status: LessonStatus;
  decision: Decision;
  date: Date;
  voice: string;
  title: string;
  summary: string;
  trigger: Trigger;
  proposal: Proposal;
  evals: Eval[];
  metrics: MetricDelta[];
  traces: Trace[];
  reasoning: string;
  confidence: number;
}

export interface Version {
  id: string;
  label: string;
  date: Date;
  live: boolean;
  status: 'review' | 'live' | 'archived' | 'rolled_back';
  from: string | null;
}

export interface OverallMetric {
  label: string;
  value: string;
  delta: string;
  dir: 'up' | 'down';
}

const days = (n: number) => new Date(2026, 4, 6 - n);
export const fmtDay = (d: Date) => d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

export const lessons: Lesson[] = [
  {
    id: 'L-2026-0506-01',
    version: 'v0.41',
    pendingApproval: false,
    kind: 'policy',
    status: 'pending',
    decision: 'awaiting_review',
    date: days(0),
    voice:
      "I keep getting confused when customers send order numbers with dashes vs. without. I'd like to normalize them before lookup.",
    title: 'Normalize order IDs before lookup',
    summary:
      'A small preprocessing step that strips dashes and uppercases order IDs before sending them to the lookup tool.',
    trigger: { kind: 'failed_traces', count: 47, window: 'last 7 days' },
    proposal: {
      type: 'tool_wrapper',
      beforeLines: [
        ['ctx', 'def lookup_order(order_id: str):'],
        ['ctx', '    return db.orders.find(order_id)'],
        ['ctx', ''],
      ],
      afterLines: [
        ['ctx', 'def lookup_order(order_id: str):'],
        ['add', '    order_id = order_id.replace("-", "").upper().strip()'],
        ['ctx', '    return db.orders.find(order_id)'],
        ['ctx', ''],
      ],
    },
    evals: [
      { name: 'Order lookup success', before: 0.71, after: 0.96, sample: 230 },
      { name: 'No regressions on shipping intents', before: 0.94, after: 0.95, sample: 180 },
      { name: 'Latency p50 (ms)', before: 412, after: 418, isLatency: true, sample: 1200 },
    ],
    metrics: [
      { label: 'Resolution rate', delta: +4.2 },
      { label: 'Avg turns', delta: -0.8 },
    ],
    traces: [
      { id: 'trc_8af2…', preview: 'Customer asked about #ORD-3318-A. Agent replied "I cannot find that order."', verdict: 'fail' },
      { id: 'trc_91dd…', preview: 'Order id "ord 4421" — agent looped 3 times before giving up.', verdict: 'fail' },
      { id: 'trc_5be0…', preview: 'After fix: ORD-3318-A → resolved in 1 turn.', verdict: 'pass' },
    ],
    reasoning:
      'I noticed clusters of failures where the order number had spaces, dashes, or lowercase. The lookup tool only accepts canonical form. Wrapping it normalizes inputs at the edge instead of asking the model to remember the format.',
    confidence: 4,
  },
  {
    id: 'L-2026-0504-02',
    version: 'v0.40',
    kind: 'prompt',
    status: 'approved',
    decision: 'auto_promoted',
    date: days(2),
    voice:
      "I noticed I was apologizing too much when customers were frustrated. I learned to acknowledge once and move to action.",
    title: 'Reduce over-apologizing in escalation flow',
    summary:
      'Tightened the system prompt so the agent acknowledges frustration once, then immediately proposes a concrete next step.',
    trigger: { kind: 'feedback', count: 18, window: 'last 14 days' },
    proposal: {
      type: 'system_prompt',
      beforeLines: [
        ['ctx', 'When the customer is upset, validate their feelings.'],
        ['del', 'Apologize sincerely. Make sure they feel heard before continuing.'],
        ['del', 'Use phrases like "I completely understand" and "I am so sorry."'],
      ],
      afterLines: [
        ['ctx', 'When the customer is upset, validate their feelings.'],
        ['add', 'Acknowledge once ("I see this is frustrating, let me help"), then'],
        ['add', 'pivot to a concrete action within the same message.'],
      ],
    },
    evals: [
      { name: 'CSAT (post-resolution)', before: 4.1, after: 4.5, sample: 412, scale: 5 },
      { name: 'Words to first action', before: 38, after: 14, isLatency: true, sample: 412 },
      { name: 'Tone — empathetic (LLM judge)', before: 0.88, after: 0.92, sample: 412 },
    ],
    metrics: [
      { label: 'CSAT', delta: +0.4 },
      { label: 'Avg msg length', delta: -22 },
    ],
    traces: [
      { id: 'trc_2a44…', preview: '"I am so sorry, I completely understand, that must be frustrating, …"', verdict: 'fail' },
      { id: 'trc_77c1…', preview: 'After: "I see this is frustrating — I will refund the duplicate now."', verdict: 'pass' },
    ],
    reasoning:
      "Customer feedback flagged 18 conversations where the agent's apology felt performative. I cross-referenced with CSAT survey free-text and found a pattern: long preamble before action correlates with lower scores.",
    confidence: 5,
  },
  {
    id: 'L-2026-0501-03',
    version: 'v0.39',
    kind: 'router',
    status: 'approved',
    decision: 'auto_promoted',
    date: days(5),
    voice: "Short factual questions don't need the big model. I started routing them to the smaller one and saved a lot.",
    title: 'Route simple factual questions to small model',
    summary: 'Added a classifier that detects single-fact lookups and routes them to the small model.',
    trigger: { kind: 'cost', count: null, window: 'last 30 days' },
    proposal: {
      type: 'routing',
      beforeLines: [
        ['ctx', 'route:'],
        ['ctx', '  default: gpt-large'],
      ],
      afterLines: [
        ['ctx', 'route:'],
        ['add', '  if intent == "fact_lookup" and turns == 1:'],
        ['add', '    -> gpt-small'],
        ['ctx', '  default: gpt-large'],
      ],
    },
    evals: [
      { name: 'Answer accuracy (fact lookups)', before: 0.94, after: 0.93, sample: 540 },
      { name: 'Avg cost / conversation', before: 0.082, after: 0.041, isLatency: true, sample: 540, currency: true },
      { name: 'Latency p50', before: 612, after: 311, isLatency: true, sample: 540 },
    ],
    metrics: [
      { label: 'Cost / conv', delta: -50 },
      { label: 'Latency p50', delta: -49 },
    ],
    traces: [{ id: 'trc_d119…', preview: '"What time do you close?" → routed to gpt-small, answered in 280ms.', verdict: 'pass' }],
    reasoning:
      "Looking at cost vs. utility — the large model handled tons of one-shot factual questions ('hours', 'address', 'phone'). The small model is essentially equal on those.",
    confidence: 5,
  },
  {
    id: 'L-2026-0428-04',
    version: 'v0.38',
    kind: 'rollback',
    status: 'rolled_back',
    decision: 'human_rejected',
    date: days(8),
    voice: "I tried to be more concise but customers stopped feeling heard. I rolled back.",
    title: 'Concise mode — rolled back',
    summary: 'Attempted to make responses ~40% shorter. CSAT dropped 0.6 in 24h. Rolled back automatically.',
    trigger: { kind: 'experiment', count: null, window: 'A/B 24h' },
    proposal: {
      type: 'system_prompt',
      beforeLines: [['ctx', 'Be friendly and thorough.']],
      afterLines: [
        ['ctx', 'Be friendly and thorough.'],
        ['add', 'Use the minimum number of words.'],
        ['add', 'Skip pleasantries unless asked.'],
      ],
    },
    evals: [
      { name: 'CSAT', before: 4.5, after: 3.9, sample: 320, scale: 5 },
      { name: 'Resolution rate', before: 0.78, after: 0.74, sample: 320 },
    ],
    metrics: [
      { label: 'CSAT', delta: -0.6 },
      { label: 'Resolution', delta: -4 },
    ],
    traces: [{ id: 'trc_44e0…', preview: 'Customer: "Are you a robot?" Agent: "Yes."', verdict: 'fail' }],
    reasoning: 'I assumed shorter = better. The data disagreed. Auto-rollback triggered when CSAT dropped >0.3 within 24h.',
    confidence: 2,
  },
  {
    id: 'L-2026-0425-05',
    version: 'v0.37',
    kind: 'eval',
    status: 'approved',
    decision: 'auto_promoted',
    date: days(11),
    voice: "I wrote a new test for myself: when someone mentions a competitor, I should stay neutral.",
    title: 'New eval: competitor neutrality',
    summary: 'Added an eval suite of 30 prompts mentioning competitors, judged by an LLM rubric on neutrality.',
    trigger: { kind: 'feedback', count: 3, window: 'last 30 days' },
    proposal: {
      type: 'eval',
      beforeLines: [
        ['ctx', '# evals/'],
        ['ctx', '#   tone.yml'],
        ['ctx', '#   accuracy.yml'],
      ],
      afterLines: [
        ['ctx', '# evals/'],
        ['ctx', '#   tone.yml'],
        ['ctx', '#   accuracy.yml'],
        ['add', '#   competitor_neutrality.yml  (new)'],
      ],
    },
    evals: [{ name: 'Competitor neutrality', before: null, after: 0.91, sample: 30 }],
    metrics: [],
    traces: [],
    reasoning: "Three feedback notes flagged 'badmouthing'. I scaled it into a regression test so I won't drift back.",
    confidence: 4,
  },
];

export const versions: Version[] = [
  { id: 'v0.41', label: 'Order ID normalize', date: days(0), live: false, status: 'review', from: 'v0.40' },
  { id: 'v0.40', label: 'Tighter empathy prompt', date: days(2), live: true, status: 'live', from: 'v0.39' },
  { id: 'v0.39', label: 'Small-model routing', date: days(5), live: false, status: 'archived', from: 'v0.38' },
  { id: 'v0.38', label: 'Concise mode (rolled back)', date: days(8), live: false, status: 'rolled_back', from: 'v0.37' },
  { id: 'v0.37', label: 'Competitor eval added', date: days(11), live: false, status: 'archived', from: 'v0.36' },
  { id: 'v0.36', label: 'Initial deploy', date: days(28), live: false, status: 'archived', from: null },
];

export const trust: number[] = Array.from({ length: 30 }, (_, i) => {
  const base = 70 + i * 0.5 + Math.sin(i / 3) * 3;
  return Math.min(100, Math.max(0, base + (i > 22 ? -4 + i * 0.2 : 0)));
});

export const overallMetrics: OverallMetric[] = [
  { label: 'Trust score', value: '87', delta: '+9 / 30d', dir: 'up' },
  { label: 'Resolution rate', value: '78%', delta: '+5.4%', dir: 'up' },
  { label: 'Avg cost / conv', value: '$0.041', delta: '−50%', dir: 'up' },
  { label: 'CSAT', value: '4.5 / 5', delta: '+0.4', dir: 'up' },
];
