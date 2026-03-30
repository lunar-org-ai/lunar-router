import { ChevronDown, ChevronUp, Wrench } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { TableRow, TableCell } from '@/components/ui/table';
import type { TraceItem } from '@/types/analyticsType';
import { ModelCell } from './ModelCell';
import { TracesExpandedRow } from './TracesExpandedRow';
import { formatLatency } from '@/utils/formatUtils';

interface TracesTableRowProps {
  trace: TraceItem;
  index: number;
  isExpanded: boolean;
  isSelected?: boolean;
  onToggleSelect?: (id: string, e: React.MouseEvent) => void;
  onToggleExpand: (id: string, e: React.MouseEvent) => void;
  onSelect: (trace: TraceItem) => void;
}

function getOutputDisplay(trace: TraceItem): string {
  if (trace.output_text) return trace.output_text;
  if (trace.output_message?.content) return trace.output_message.content;
  if (trace.output_preview) return trace.output_preview;
  return '';
}

export function TracesTableRow({
  trace,
  isExpanded,
  isSelected,
  onToggleSelect,
  onToggleExpand,
  onSelect,
}: TracesTableRowProps) {
  const formattedDate = trace.created_at ? new Date(trace.created_at).toLocaleString() : 'Unknown';

  const outputText = getOutputDisplay(trace);
  const hasToolCalls = !!trace.output_message?.tool_calls?.length;

  return (
    <>
      <TableRow
        onClick={() => onSelect(trace)}
        className={`cursor-pointer group/row transition-colors ${isSelected ? 'bg-primary/5' : ''}`}
      >
        <TableCell className="w-10" onClick={(e) => e.stopPropagation()}>
          <Checkbox
            checked={isSelected}
            onCheckedChange={() => onToggleSelect?.(trace.id, {} as React.MouseEvent)}
          />
        </TableCell>
        <TableCell>
          <ModelCell modelId={trace.model_id} backend={trace.backend} provider={trace.provider} />
        </TableCell>

        <TableCell className="hidden 2xl:table-cell">
          <span className="max-w-xs truncate block text-sm">{trace.input_preview}</span>
        </TableCell>

        <TableCell className="hidden 2xl:table-cell">
          {hasToolCalls ? (
            <Badge variant="secondary">
              <Wrench className="size-3" />
              {trace.output_message!.tool_calls!.map((tc) => tc.function.name).join(', ')}
            </Badge>
          ) : (
            <span className="max-w-xs truncate block text-sm text-muted-foreground">
              {outputText || '—'}
            </span>
          )}
        </TableCell>

        <TableCell className="hidden xl:table-cell">
          <span className="text-sm text-muted-foreground tabular-nums">{formattedDate}</span>
        </TableCell>

        <TableCell>
          <span className="text-sm tabular-nums font-mono">{formatLatency(trace.latency_s)}</span>
        </TableCell>

        <TableCell className="hidden xl:table-cell text-right">
          <span className="text-sm tabular-nums">{trace.total_tokens.toLocaleString()}</span>
        </TableCell>

        <TableCell>
          <Badge variant={trace.status === 'Success' ? 'secondary' : 'destructive'}>
            {trace.status}
          </Badge>
        </TableCell>

        <TableCell className="w-10">
          <Button
            variant="ghost"
            size="icon"
            className="size-7 text-muted-foreground"
            onClick={(e) => onToggleExpand(trace.id, e)}
          >
            {isExpanded ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
          </Button>
        </TableCell>
      </TableRow>

      {isExpanded && (
        <TableRow className="bg-muted hover:bg-muted">
          <TableCell colSpan={9} className="p-4">
            <TracesExpandedRow trace={trace} onViewDetails={() => onSelect(trace)} />
          </TableCell>
        </TableRow>
      )}
    </>
  );
}
