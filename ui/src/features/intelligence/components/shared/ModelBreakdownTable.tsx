import { useMemo, useState } from 'react';
import { ArrowUpDown, Download } from 'lucide-react';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { formatCost } from '@/utils/formatUtils';
import { exportTableToCsv, formatNumber, formatPercent } from '../../utils/intelligenceHelpers';
import type { UnifiedModelRow } from '../../types';

interface ModelBreakdownTableProps {
  rows: UnifiedModelRow[];
  title?: string;
}

type SortKey = 'model' | 'provider' | 'requests' | 'accuracy' | 'avgCost' | 'totalCost';
type SortDir = 'asc' | 'desc';

const COLUMNS: { key: SortKey; label: string; align?: 'right' }[] = [
  { key: 'model', label: 'Model' },
  { key: 'provider', label: 'Provider' },
  { key: 'requests', label: 'Requests', align: 'right' },
  { key: 'accuracy', label: 'Accuracy', align: 'right' },
  { key: 'avgCost', label: 'Avg Cost', align: 'right' },
  { key: 'totalCost', label: 'Total Cost', align: 'right' },
];

export function ModelBreakdownTable({
  rows,
  title = 'Per-Model Breakdown',
}: ModelBreakdownTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('requests');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const sorted = useMemo(() => {
    return [...rows].sort((a, b) => {
      const aVal = a[sortKey] ?? 0;
      const bVal = b[sortKey] ?? 0;
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      const numA = Number(aVal);
      const numB = Number(bVal);
      return sortDir === 'asc' ? numA - numB : numB - numA;
    });
  }, [rows, sortKey, sortDir]);

  const totalCost = rows.reduce((s, r) => s + r.totalCost, 0);
  const totalReqs = rows.reduce((s, r) => s + r.requests, 0);

  const handleExport = () => {
    const headers = COLUMNS.map((c) => c.label);
    const csvRows = sorted.map((row) => [
      row.model,
      row.provider,
      row.requests,
      row.accuracy !== null ? formatPercent(row.accuracy) : '—',
      formatCost(row.avgCost),
      formatCost(row.totalCost),
    ]);
    exportTableToCsv(headers, csvRows, 'model-breakdown');
  };

  if (rows.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="py-8 text-center text-sm text-muted-foreground">No model data available</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        <CardDescription>
          {rows.length} model{rows.length !== 1 ? 's' : ''} &middot; {formatNumber(totalReqs)}{' '}
          requests &middot; {formatCost(totalCost)} total cost
        </CardDescription>
        <CardAction>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="outline" size="sm" className="h-7 gap-1.5" onClick={handleExport}>
                <Download className="size-3" />
                CSV
              </Button>
            </TooltipTrigger>
            <TooltipContent>Export table as CSV</TooltipContent>
          </Tooltip>
        </CardAction>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              {COLUMNS.map((col) => (
                <TableHead key={col.key} className={col.align === 'right' ? 'text-right' : ''}>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="-ml-3 h-8 gap-1 text-xs font-medium"
                    onClick={() => handleSort(col.key)}
                  >
                    {col.label}
                    <ArrowUpDown
                      className={`size-3 ${sortKey === col.key ? 'text-foreground' : 'text-muted-foreground/50'}`}
                    />
                  </Button>
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {sorted.map((row) => (
              <TableRow key={row.model}>
                <TableCell className="font-medium">{row.model}</TableCell>
                <TableCell>
                  <Badge variant="outline" className="text-xs font-normal">
                    {row.provider}
                  </Badge>
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {formatNumber(row.requests)}
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {row.accuracy !== null ? (
                    formatPercent(row.accuracy)
                  ) : (
                    <span className="text-muted-foreground">—</span>
                  )}
                </TableCell>
                <TableCell className="text-right tabular-nums">{formatCost(row.avgCost)}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {formatCost(row.totalCost)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
