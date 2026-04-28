/**
 * Objectives tab — one card per user-declared objective. Each card
 * shows YAML metadata (direction, target, guardrails) plus a time-
 * series plot of the objective's measured value with clickable action
 * markers on the x-axis. Clicking a marker opens the chain drawer so
 * the user can answer "why did this move?" in one interaction.
 */

import { useEffect, useState } from 'react';
import { Shield, Target, TrendingDown, TrendingUp, Users } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import {
  useHarnessService,
  type Objective,
  type ObjectiveDirection,
} from '@/services/harnessService';
import { ChainDrawer } from '../components/ChainDrawer';
import { ObjectivePlot } from '../components/ObjectivePlot';

function DirectionIcon({ direction }: { direction: ObjectiveDirection }) {
  return direction === 'higher_is_better' ? (
    <TrendingUp className="size-4 text-emerald-500" />
  ) : (
    <TrendingDown className="size-4 text-sky-500" />
  );
}

function formatTarget(objective: Objective): string {
  if (objective.target === null) return '—';
  const precision = objective.unit === 'USD' ? 4 : 2;
  return `${objective.target.toFixed(precision)} ${objective.unit}`;
}

function ObjectiveCard({
  objective,
  onMarkerClick,
}: {
  objective: Objective;
  onMarkerClick: (entryId: string) => void;
}) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2 min-w-0">
            <Target className="size-4 text-primary shrink-0" />
            <CardTitle className="text-sm font-mono truncate">{objective.id}</CardTitle>
          </div>
          <Badge variant="secondary" className="gap-1 whitespace-nowrap">
            <DirectionIcon direction={objective.direction} />
            {objective.direction === 'higher_is_better' ? 'higher' : 'lower'}
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground leading-relaxed pt-1">
          {objective.description}
        </p>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Plot — the reason this tab exists. */}
        <ObjectivePlot objective={objective} onMarkerClick={onMarkerClick} />

        <div className="grid grid-cols-3 gap-3 text-xs">
          <div>
            <div className="text-muted-foreground uppercase tracking-wide">Target</div>
            <div className="font-mono tabular-nums mt-0.5">{formatTarget(objective)}</div>
          </div>
          <div>
            <div className="text-muted-foreground uppercase tracking-wide">Window</div>
            <div className="font-mono tabular-nums mt-0.5">{objective.window_hours}h</div>
          </div>
          <div>
            <div className="text-muted-foreground uppercase tracking-wide">Cadence</div>
            <div className="font-mono mt-0.5">{objective.update_cadence}</div>
          </div>
        </div>

        {objective.guardrails.length > 0 && (
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Shield className="size-3" />
              <span className="uppercase tracking-wide">Guardrails</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {objective.guardrails.map((g, i) => (
                <Badge key={i} variant="outline" className="text-xs font-mono">
                  {g.type}
                  {g.threshold !== null && g.threshold !== undefined && ` = ${g.threshold}`}
                  {g.min_n !== null && g.min_n !== undefined && ` (n≥${g.min_n})`}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {objective.owner_agents.length > 0 && (
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Users className="size-3" />
              <span className="uppercase tracking-wide">Owner agents</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {objective.owner_agents.map((a) => (
                <Badge key={a} variant="secondary" className="text-xs font-mono">
                  {a}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function ObjectivesTab() {
  const service = useHarnessService();
  const [objectives, setObjectives] = useState<Objective[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  useEffect(() => {
    service.listObjectives().then((list) => {
      setObjectives(list);
      setLoading(false);
    });
  }, [service]);

  const handleMarkerClick = (entryId: string) => {
    setSelectedId(entryId);
    setDrawerOpen(true);
  };

  if (loading) {
    return (
      <div className="grid gap-3 md:grid-cols-2">
        {[1, 2, 3].map((i) => (
          <Card key={i}>
            <CardHeader>
              <Skeleton className="h-4 w-48" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-20 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (objectives.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-sm text-muted-foreground">
        No objectives defined. Add YAML files under{' '}
        <code className="mx-1 font-mono bg-muted px-1.5 py-0.5 rounded">
          opentracy/harness/objectives/definitions/
        </code>
        .
      </div>
    );
  }

  return (
    <>
      <div className="grid gap-3 md:grid-cols-2">
        {objectives.map((obj) => (
          <ObjectiveCard key={obj.id} objective={obj} onMarkerClick={handleMarkerClick} />
        ))}
      </div>

      <ChainDrawer
        open={drawerOpen}
        rootEntryId={selectedId}
        onClose={() => setDrawerOpen(false)}
      />
    </>
  );
}
