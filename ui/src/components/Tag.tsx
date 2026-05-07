import type { ReactNode } from 'react';
import { Icon, type IconName } from './Icon';

export type TagKind = '' | 'success' | 'warn' | 'bad' | 'info' | 'solid';

export const Tag = ({ kind, children }: { kind?: TagKind; children: ReactNode }) => (
  <span className={`tag ${kind || ''}`}>{children}</span>
);

export type LessonStatus = 'pending' | 'approved' | 'rolled_back' | 'auto_promoted' | 'awaiting_review' | 'human_rejected';

export const StatusTag = ({ status }: { status: LessonStatus | string }) => {
  const map: Record<string, { kind: TagKind; label: string; dot: boolean }> = {
    pending: { kind: 'warn', label: 'Awaiting review', dot: true },
    awaiting_review: { kind: 'warn', label: 'Awaiting review', dot: true },
    approved: { kind: 'success', label: 'Approved', dot: true },
    rolled_back: { kind: 'bad', label: 'Rolled back', dot: true },
    auto_promoted: { kind: 'success', label: 'Auto-promoted', dot: true },
    human_rejected: { kind: 'bad', label: 'Rejected', dot: true },
  };
  const m = map[status] || { kind: '' as TagKind, label: status, dot: false };
  return (
    <Tag kind={m.kind}>
      {m.dot && <span className="dot" />}
      {m.label}
    </Tag>
  );
};

export type LessonKind = 'prompt' | 'policy' | 'router' | 'eval' | 'rollback';

export const KindIcon = ({ kind }: { kind: LessonKind }) => {
  const map: Record<LessonKind, IconName> = {
    prompt: 'sparkles',
    policy: 'shield',
    router: 'route',
    eval: 'flask',
    rollback: 'rollback',
  };
  return <Icon name={map[kind] || 'sparkles'} size={14} />;
};

export const KindLabel = ({ kind }: { kind: LessonKind }) => {
  const map: Record<LessonKind, string> = {
    prompt: 'Prompt change',
    policy: 'Behavior change',
    router: 'Routing change',
    eval: 'New self-test',
    rollback: 'Rolled back',
  };
  return <span>{map[kind] || kind}</span>;
};
