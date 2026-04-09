import { useMemo, useState } from 'react';
import { ArrowUpDown } from 'lucide-react';
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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { formatCost } from '@/utils/formatUtils';
import type { UnifiedModelRow } from '../../types';

interface ModelBreakdownTableProps {
  rows: UnifiedModelRow[];
  title?: string;
}

type SortKey =
  | 'model'
  | 'provider'
  | 'requests'
  | 'trafficPct'
  | 'accuracy'
  | 'avgCost'
  | 'totalCost';
type SortDir = 'asc' | 'desc';

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

  const columns: { key: SortKey; label: string; align?: 'right' }[] = [
    { key: 'model', label: 'Model' },
    { key: 'provider', label: 'Provider' },
    { key: 'requests', label: 'Requests', align: 'right' },
    { key: 'trafficPct', label: 'Traffic %', align: 'right' },
    { key: 'accuracy', label: 'Accuracy', align: 'right' },
    { key: 'avgCost', label: 'Avg Cost', align: 'right' },
    { key: 'totalCost', label: 'Total Cost', align: 'right' },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        <CardDescription>
          {rows.length} model{rows.length !== 1 ? 's' : ''} tracked
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              {columns.map((col) => (
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
              <TableRow key={row.model} className="transition-colors hover:bg-muted/30">
                <TableCell className="font-medium">{row.model}</TableCell>
                <TableCell>
                  <Badge variant="outline" className="text-xs font-normal">
                    {row.provider}
                  </Badge>
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {row.requests.toLocaleString()}
                </TableCell>
                <TableCell className="text-right">
                  <span className="tabular-nums">{row.trafficPct.toFixed(1)}%</span>
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {row.accuracy !== null ? `${(row.accuracy * 100).toFixed(1)}%` : '—'}
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
