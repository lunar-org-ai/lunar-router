import { WandSparkles, ChevronRight, XCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { ExpandableText } from '../shared/ExpandableText';
import type { TraceIssue } from '../../types/evaluationsTypes';
import type { EvalPrefillConfig } from '../../types';
import { ISSUE_TYPE_LABELS, SEVERITY_VARIANT } from './constants';

interface IssueDetailSheetProps {
  open: boolean;
  onClose: () => void;
  issue: TraceIssue | null;
  onResolve: (id: string) => void;
  onDismiss: (id: string) => void;
  onRunEval?: (config: EvalPrefillConfig) => void;
}

export function IssueDetailSheet({
  open,
  onClose,
  issue,
  onResolve,
  onDismiss,
  onRunEval,
}: IssueDetailSheetProps) {
  if (!issue) return null;

  const confidencePercent = Math.round(issue.ai_confidence * 100);

  return (
    <Sheet open={open} onOpenChange={(v) => !v && onClose()}>
      <SheetContent className="sm:max-w-lg flex flex-col gap-0">
        <SheetHeader>
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={SEVERITY_VARIANT[issue.severity]}>{issue.severity.toUpperCase()}</Badge>
            <Badge variant="secondary">{ISSUE_TYPE_LABELS[issue.type] || issue.type}</Badge>
            {issue.dismissed && <Badge variant="outline">Dismissed</Badge>}
            {issue.resolved && !issue.dismissed && <Badge variant="outline">Resolved</Badge>}
          </div>
          <SheetTitle>{issue.title}</SheetTitle>
          <SheetDescription>
            Detected {new Date(issue.detected_at).toLocaleString()}
          </SheetDescription>
        </SheetHeader>

        <ScrollArea className="flex-1 min-h-0 px-4">
          <div className="space-y-5 py-4">
            <section className="space-y-1.5">
              <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Analysis
              </h4>
              <p className="text-sm text-muted-foreground leading-relaxed">{issue.description}</p>
            </section>

            <section className="space-y-1.5">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="uppercase tracking-wide font-medium">Confidence</span>
                <span className="tabular-nums font-medium text-foreground">
                  {confidencePercent}%
                </span>
              </div>
              <Progress value={confidencePercent} />
            </section>

            <section className="space-y-1.5">
              <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Model
              </h4>
              <Badge variant="outline" className="font-mono text-xs">
                {issue.model_id}
              </Badge>
            </section>

            <Separator />

            {issue.trace_input && <ExpandableText label="Trace Input" text={issue.trace_input} />}
            {issue.trace_output && (
              <ExpandableText label="Trace Output" text={issue.trace_output} />
            )}

            {issue.suggested_action && (
              <section className="space-y-1.5">
                <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Suggested Action
                </h4>
                <p className="text-sm text-muted-foreground">{issue.suggested_action}</p>
              </section>
            )}
          </div>
        </ScrollArea>

        <SheetFooter>
          {!issue.resolved && (
            <>
              <Button size="sm" onClick={() => onResolve(issue.id)} className="flex-1">
                <WandSparkles className="size-3.5" />
                Resolve
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => onDismiss(issue.id)}
                className="flex-1"
              >
                <XCircle className="size-3.5" />
                Not an Error
              </Button>
            </>
          )}
          {issue.suggested_eval_config && onRunEval && !issue.resolved && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => onRunEval(issue.suggested_eval_config!)}
              className="flex-1"
            >
              <ChevronRight className="size-3.5" />
              Run Eval
            </Button>
          )}
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}
