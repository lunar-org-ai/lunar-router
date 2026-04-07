import { WandSparkles, ChevronRight, XCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import { formatRelativeTime } from '../../utils';
import type { TraceIssue } from '../../types/evaluationsTypes';
import { ISSUE_TYPE_LABELS, SEVERITY_VARIANT } from './constants';

interface IssueTableProps {
  issues: TraceIssue[];
  onViewDetails: (issue: TraceIssue) => void;
  onResolve: (id: string) => void;
  onDismiss: (id: string) => void;
}

export function IssueTable({ issues, onViewDetails, onResolve, onDismiss }: IssueTableProps) {
  return (
    <TooltipProvider delayDuration={150}>
      <div className="overflow-hidden rounded-lg border">
        <Table>
          <TableHeader className="bg-muted">
            <TableRow>
              <TableHead>Severity</TableHead>
              <TableHead>Issue</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Model</TableHead>
              <TableHead className="text-right">Confidence</TableHead>
              <TableHead className="text-right">Detected</TableHead>
              <TableHead className="w-24" />
            </TableRow>
          </TableHeader>

          <TableBody>
            {issues.map((issue) => {
              const confidencePercent = Math.round(issue.ai_confidence * 100);

              return (
                <TableRow
                  key={issue.id}
                  className={cn(
                    'cursor-pointer group/row transition-colors',
                    issue.resolved && 'text-muted-foreground'
                  )}
                  onClick={() => onViewDetails(issue)}
                >
                  <TableCell>
                    <Badge variant={SEVERITY_VARIANT[issue.severity]} className="px-1.5">
                      {issue.severity.charAt(0).toUpperCase() + issue.severity.slice(1)}
                    </Badge>
                  </TableCell>

                  <TableCell>
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="font-medium truncate">{issue.title}</span>
                      {issue.dismissed && (
                        <Badge variant="outline" className="px-1.5 text-muted-foreground shrink-0">
                          Dismissed
                        </Badge>
                      )}
                      {issue.resolved && !issue.dismissed && (
                        <Badge variant="outline" className="px-1.5 text-muted-foreground shrink-0">
                          Resolved
                        </Badge>
                      )}
                    </div>
                  </TableCell>

                  <TableCell>
                    <Badge variant="outline" className="px-1.5 text-muted-foreground">
                      {ISSUE_TYPE_LABELS[issue.type] || issue.type}
                    </Badge>
                  </TableCell>

                  <TableCell>
                    <span className="font-mono text-xs text-muted-foreground">
                      {issue.model_id}
                    </span>
                  </TableCell>

                  <TableCell className="text-right">
                    <span className="text-xs tabular-nums font-medium">{confidencePercent}%</span>
                  </TableCell>

                  <TableCell className="text-right">
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {formatRelativeTime(issue.detected_at)}
                    </span>
                  </TableCell>

                  <TableCell>
                    <div className="flex items-center justify-end gap-1">
                      {!issue.resolved && (
                        <>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="size-7 text-muted-foreground hover:text-emerald-600 transition-colors"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onResolve(issue.id);
                                }}
                              >
                                <WandSparkles className="size-4" />
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>Resolve</TooltipContent>
                          </Tooltip>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="size-7 text-muted-foreground hover:text-amber-600 transition-colors"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onDismiss(issue.id);
                                }}
                              >
                                <XCircle className="size-4" />
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>Not an Error</TooltipContent>
                          </Tooltip>
                        </>
                      )}
                      <ChevronRight className="size-4 text-border group-hover/row:text-muted-foreground transition-colors" />
                    </div>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </TooltipProvider>
  );
}
