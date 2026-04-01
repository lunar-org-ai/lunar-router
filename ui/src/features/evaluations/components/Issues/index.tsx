import { useState, useMemo, useCallback } from 'react';
import { Loader2, Radar, Clock, Timer } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { SearchBar } from '@/components/shared/SearchBar';
import { useTraceIssues } from '../../hooks/useTraceIssues';
import { IssueSummaryCards } from './IssueSummaryCards';
import { IssueTable } from './IssueTable';
import { IssueDetailSheet } from './IssueDetailSheet';
import { ProblemsEmpty } from './ProblemsEmpty';
import { ProblemsSkeleton } from './Skeleton';
import { SEVERITY_FILTERS, TYPE_FILTERS } from './constants';
import type { TraceIssue, IssueSeverity, IssueType } from '../../types/evaluationsTypes';
import type { EvalPrefillConfig } from '../../types';

const INTERVAL_OPTIONS = [
  { value: '300', label: '5 min' },
  { value: '900', label: '15 min' },
  { value: '1800', label: '30 min' },
  { value: '3600', label: '1 hour' },
  { value: '21600', label: '6 hours' },
  { value: '86400', label: '24 hours' },
];

interface ProblemsTabProps {
  onRunEval?: (config: EvalPrefillConfig) => void;
}

export function ProblemsTab({ onRunEval }: ProblemsTabProps) {
  const {
    issues,
    scanning,
    triggerScan,
    resolveIssue,
    dismissIssue,
    loading,
    schedule,
    scheduleRunning,
    updateSchedule,
  } = useTraceIssues();

  const [selectedIssue, setSelectedIssue] = useState<TraceIssue | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [severityFilter, setSeverityFilter] = useState<IssueSeverity | 'all'>('all');
  const [typeFilter, setTypeFilter] = useState<IssueType | 'all'>('all');

  const filteredIssues = useMemo(() => {
    const term = searchTerm.toLowerCase();
    return issues.filter((issue) => {
      if (severityFilter !== 'all' && issue.severity !== severityFilter) return false;
      if (typeFilter !== 'all' && issue.type !== typeFilter) return false;
      if (
        term &&
        !issue.title.toLowerCase().includes(term) &&
        !issue.model_id.toLowerCase().includes(term)
      ) {
        return false;
      }
      return true;
    });
  }, [issues, searchTerm, severityFilter, typeFilter]);

  const handleResolve = useCallback(
    (id: string) => {
      resolveIssue(id);
      setSelectedIssue(null);
    },
    [resolveIssue]
  );

  const handleDismiss = useCallback(
    (id: string) => {
      dismissIssue(id);
      setSelectedIssue(null);
    },
    [dismissIssue]
  );

  if (loading) return <ProblemsSkeleton />;

  const isEmpty = issues.length === 0;
  const hasResults = filteredIssues.length > 0;

  return (
    <div className="space-y-4 animate-in fade-in-50 duration-300">
      <IssueSummaryCards issues={issues} />

      <div className="flex flex-col sm:flex-row sm:items-center gap-3">
        <SearchBar
          value={searchTerm}
          onChange={setSearchTerm}
          placeholder="Search issues…"
          filters={
            <>
              <Select
                value={severityFilter}
                onValueChange={(v) => setSeverityFilter(v as IssueSeverity | 'all')}
              >
                <SelectTrigger size="sm" className="w-fit min-w-28">
                  <SelectValue placeholder="Severity" />
                </SelectTrigger>
                <SelectContent>
                  {SEVERITY_FILTERS.map((f) => (
                    <SelectItem key={f.id} value={f.id}>
                      {f.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={typeFilter}
                onValueChange={(v) => setTypeFilter(v as IssueType | 'all')}
              >
                <SelectTrigger size="sm" className="w-fit min-w-28">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent>
                  {TYPE_FILTERS.map((f) => (
                    <SelectItem key={f.id} value={f.id}>
                      {f.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          }
        />

        <div className="flex items-center gap-2 ml-auto shrink-0">
          {schedule && (
            <TooltipProvider delayDuration={150}>
              <div className="flex items-center gap-2 rounded-md border px-2 py-1">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="flex items-center gap-1.5">
                      <Timer className="size-3.5 text-muted-foreground" />
                      <Switch
                        checked={schedule.enabled}
                        onCheckedChange={(enabled) => updateSchedule({ enabled })}
                        className="scale-75"
                      />
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    {schedule.enabled ? 'Scheduled scanning active' : 'Enable scheduled scans'}
                  </TooltipContent>
                </Tooltip>

                {schedule.enabled && (
                  <Select
                    value={String(schedule.interval_seconds)}
                    onValueChange={(v) =>
                      updateSchedule({ interval_seconds: parseInt(v, 10) })
                    }
                  >
                    <SelectTrigger size="sm" className="h-6 w-fit min-w-16 text-xs border-0 px-1">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {INTERVAL_OPTIONS.map((o) => (
                        <SelectItem key={o.value} value={o.value}>
                          {o.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}

                {schedule.enabled && schedule.total_runs > 0 && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge variant="secondary" className="text-xs px-1.5">
                        <Clock className="size-3 mr-1" />
                        {schedule.total_runs}
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>
                      {schedule.total_runs} scheduled scans completed,{' '}
                      {schedule.total_issues_found} issues found
                      {schedule.last_run_at && (
                        <>
                          <br />
                          Last run: {new Date(schedule.last_run_at).toLocaleString()}
                        </>
                      )}
                    </TooltipContent>
                  </Tooltip>
                )}
              </div>
            </TooltipProvider>
          )}

          <Button size="sm" onClick={triggerScan} disabled={scanning}>
            {scanning ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Radar className="size-4" />
            )}
            {scanning ? 'Scanning…' : 'Scan Now'}
          </Button>
        </div>
      </div>

      {!hasResults ? (
        <ProblemsEmpty isEmpty={isEmpty} />
      ) : (
        <IssueTable
          issues={filteredIssues}
          onViewDetails={setSelectedIssue}
          onResolve={resolveIssue}
          onDismiss={dismissIssue}
        />
      )}

      <IssueDetailSheet
        open={!!selectedIssue}
        onClose={() => setSelectedIssue(null)}
        issue={selectedIssue}
        onResolve={handleResolve}
        onDismiss={handleDismiss}
        onRunEval={onRunEval}
      />
    </div>
  );
}
