import {
  Activity,
  BarChart3,
  Database,
  Layers,
  Route,
  BookOpen,
} from 'lucide-react';
import type { TabKey } from '../../types/tabs';

export interface MenuItem {
  label: string;
  value: TabKey;
  path: string;
  icon: React.ReactNode;
  description?: string;
  showFor?: ('personal' | 'organization')[];
  external?: boolean;
  badge?: string | number;
  disabled?: boolean;
}

export interface MenuSection {
  id: string;
  label?: string;
  items: MenuItem[];
  collapsible?: boolean;
}

export const MENU_SECTIONS: MenuSection[] = [
  {
    id: 'connect',
    label: 'CONNECT',
    items: [
      {
        label: 'Data Sources',
        value: 'data-sources',
        path: '/data-sources',
        icon: <Database size={20} />,
        description: 'Configure LLM provider API keys',
      },
    ],
  },
  {
    id: 'datasets',
    label: 'DISTILL',
    items: [
      {
        label: 'Datasets',
        value: 'distill-datasets',
        path: '/distill-datasets',
        icon: <Layers size={20} />,
        description: 'Manage datasets and traces',
      },
      {
        label: 'Metrics',
        value: 'distill-metrics',
        path: '/distill-metrics',
        icon: <Activity size={20} />,
        description: 'Evaluation metrics & experiments',
      },
    ],
  },
  {
    id: 'monitor',
    label: 'OBSERVE',
    items: [
      {
        label: 'Traces',
        value: 'traces',
        path: '/traces',
        icon: <Route size={20} />,
        description: 'View routing & inference traces',
      },
      {
        label: 'Observability',
        value: 'observability',
        path: '/observability',
        icon: <BarChart3 size={20} />,
        description: 'Metrics, costs & performance',
      },
    ],
  },
  {
    id: 'config',
    label: '',
    items: [
      {
        label: 'Docs',
        value: 'documentation',
        path: 'https://github.com/lunar-org-ai/lunar-router',
        icon: <BookOpen size={20} />,
        description: 'Documentation',
        external: true,
      },
    ],
  },
];

export const MENU_ITEMS: MenuItem[] = MENU_SECTIONS.flatMap((section) => section.items);

export const ROUTES_MAP = Object.fromEntries(MENU_ITEMS.map((item) => [item.value, item.path]));
