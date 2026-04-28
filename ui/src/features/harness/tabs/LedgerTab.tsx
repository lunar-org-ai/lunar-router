/**
 * Ledger tab — filtered table of recent entries. Clicking a row opens
 * the shared ChainDrawer with that entry as root so the user can see
 * the full causal descent.
 */

import { useCallback, useEffect, useState } from 'react';
import { Filter } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  useHarnessService,
  type LedgerEntry,
  type LedgerEntryType,
  type LedgerFilters,
} from '@/services/harnessService';
import { ChainDrawer } from '../components/ChainDrawer';
import { OutcomeIcon, TypeBadge, formatTs } from '../components/ledgerFormat';

const TYPE_OPTIONS: { value: LedgerEntryType | ''; label: string }[] = [
  { value: '', label: 'All types' },
  { value: 'signal', label: 'signal' },
  { value: 'run', label: 'run' },
  { value: 'observation', label: 'observation' },
  { value: 'decision', label: 'decision' },
  { value: 'action', label: 'action' },
  { value: 'proposal', label: 'proposal' },
  { value: 'lesson', label: 'lesson' },
];

export function LedgerTab() {
  const service = useHarnessService();
  const [entries, setEntries] = useState<LedgerEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState<LedgerFilters>({ type: '', limit: 100 });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    service.listLedgerEntries(filters).then((list) => {
      setEntries(list);
      setLoading(false);
    });
  }, [service, filters]);

  useEffect(() => {
    load();
  }, [load]);

  const updateFilter = <K extends keyof LedgerFilters>(key: K, value: LedgerFilters[K]) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="p-3">
          <div className="flex items-center gap-2 flex-wrap">
            <Filter className="size-4 text-muted-foreground" />

            <Select
              value={filters.type ?? ''}
              onValueChange={(v) =>
                updateFilter('type', (v === '__all__' ? '' : v) as LedgerEntryType | '')
              }
            >
              <SelectTrigger className="h-8 w-36">
                <SelectValue placeholder="Type" />
              </SelectTrigger>
              <SelectContent>
                {TYPE_OPTIONS.map((o) => (
                  <SelectItem key={o.value || '__all__'} value={o.value || '__all__'}>
                    {o.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Input
              className="h-8 w-56"
              placeholder="objective_id…"
              value={filters.objective_id ?? ''}
              onChange={(e) => updateFilter('objective_id', e.target.value)}
            />

            <Input
              className="h-8 w-40"
              placeholder="agent…"
              value={filters.agent ?? ''}
              onChange={(e) => updateFilter('agent', e.target.value)}
            />

            <Button size="sm" variant="outline" onClick={load} className="h-8">
              Refresh
            </Button>

            <span className="ml-auto text-xs text-muted-foreground tabular-nums">
              {entries.length} entries
            </span>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-24">Type</TableHead>
                <TableHead className="w-40">Agent</TableHead>
                <TableHead>Objective</TableHead>
                <TableHead className="w-24">Outcome</TableHead>
                <TableHead className="w-40">When</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                Array.from({ length: 4 }).map((_, i) => (
                  <TableRow key={i}>
                    <TableCell colSpan={5}>
                      <Skeleton className="h-4 w-full" />
                    </TableCell>
                  </TableRow>
                ))
              ) : entries.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} className="text-center py-8 text-muted-foreground">
                    No ledger entries match these filters. The trigger engine writes here
                    once it starts running.
                  </TableCell>
                </TableRow>
              ) : (
                entries.map((e) => (
                  <TableRow
                    key={e.id}
                    onClick={() => {
                      setSelectedId(e.id);
                      setDrawerOpen(true);
                    }}
                    className="cursor-pointer"
                  >
                    <TableCell>
                      <TypeBadge type={e.type} />
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {e.agent ?? '—'}
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {e.objective_id ?? <span className="text-muted-foreground">—</span>}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1.5">
                        <OutcomeIcon outcome={e.outcome} />
                        <span className="text-xs">{e.outcome ?? '—'}</span>
                      </div>
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground tabular-nums">
                      {formatTs(e.ts)}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <ChainDrawer
        open={drawerOpen}
        rootEntryId={selectedId}
        onClose={() => setDrawerOpen(false)}
      />
    </div>
  );
}
