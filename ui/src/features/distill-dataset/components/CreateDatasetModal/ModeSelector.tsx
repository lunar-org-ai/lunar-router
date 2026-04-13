import { useMemo } from 'react';
import { FileText, GitBranch, Search, Upload, Wand2, Boxes } from 'lucide-react';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { CreateMode } from '../../types';

interface ModeSelectorProps {
  mode: CreateMode;
  onModeChange: (mode: CreateMode) => void;
  disabled: boolean;
  showTopic: boolean;
  showTraces: boolean;
  showCluster: boolean;
}

const MODE_CONFIG = [
  { value: 'manual', label: 'Manual', icon: FileText },
  { value: 'import', label: 'Import', icon: Upload },
  { value: 'smart-import', label: 'Smart Import', icon: Wand2 },
  { value: 'topic', label: 'From Topic', icon: Search },
  { value: 'traces', label: 'From Traces', icon: GitBranch },
  { value: 'cluster', label: 'From Cluster', icon: Boxes },
] as const satisfies readonly {
  value: CreateMode;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}[];

export function ModeSelector({
  mode,
  onModeChange,
  disabled,
  showTopic,
  showTraces,
  showCluster,
}: ModeSelectorProps) {
  const visibleModes = useMemo(
    () =>
      MODE_CONFIG.filter((m) => {
        if (m.value === 'topic') return showTopic;
        if (m.value === 'traces') return showTraces;
        if (m.value === 'cluster') return showCluster;
        return true;
      }),
    [showTopic, showTraces, showCluster]
  );

  return (
    <Tabs value={mode} onValueChange={(v) => onModeChange(v as CreateMode)}>
      <TabsList className="w-full">
        {visibleModes.map(({ value, label, icon: Icon }) => (
          <TabsTrigger key={value} value={value} disabled={disabled} className="flex-1 gap-1.5">
            <Icon className="size-4" />
            {label}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}
