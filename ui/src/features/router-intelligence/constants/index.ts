import { createElement } from 'react';
import { TrendingUp, BarChart3, GraduationCap } from 'lucide-react';
import type { PageTab } from '@/components/shared/PageTabs';
import type { TabId } from '../types';

export const TABS: PageTab<TabId>[] = [
  { id: 'efficiency', label: 'Router Efficiency', icon: createElement(TrendingUp, { className: 'size-4' }) },
  { id: 'model-performance', label: 'Model Performance', icon: createElement(BarChart3, { className: 'size-4' }) },
  { id: 'training', label: 'Training Activity', icon: createElement(GraduationCap, { className: 'size-4' }) },
];

export const TIME_RANGE_OPTIONS = [7, 14, 30] as const;
export type TimeRange = (typeof TIME_RANGE_OPTIONS)[number];
