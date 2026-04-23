/**
 * Shared chain drawer — opens from either the Ledger table or an
 * Objectives-tab action marker. Given a root ledger entry id, fetches
 * the full BFS chain and renders it as a tree, each node colored and
 * tagged so the reader can follow causation at a glance.
 */

import { useEffect, useState } from 'react';
import { Link2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { Skeleton } from '@/components/ui/skeleton';
import {
  useHarnessService,
  type LedgerEntry,
  type LedgerEntryType,
} from '@/services/harnessService';
import { OutcomeIcon, TypeBadge, formatTs, shortId } from './ledgerFormat';

interface ChainNode extends LedgerEntry {
  depth: number;
}

function buildTreeView(entries: LedgerEntry[]): ChainNode[] {
  if (entries.length === 0) return [];
  const byId = new Map(entries.map((e) => [e.id, e]));
  const rootId = entries[0].id;
  const depthFor = new Map<string, number>();

  const compute = (id: string): number => {
    if (depthFor.has(id)) return depthFor.get(id)!;
    if (id === rootId) {
      depthFor.set(id, 0);
      return 0;
    }
    const entry = byId.get(id);
    if (!entry || !entry.parent_id || !byId.has(entry.parent_id)) {
      depthFor.set(id, 0);
      return 0;
    }
    const d = compute(entry.parent_id) + 1;
    depthFor.set(id, d);
    return d;
  };

  return entries.map((e) => ({ ...e, depth: compute(e.id) }));
}

function ChainNodeCard({ node }: { node: ChainNode }) {
  return (
    <div
      className="relative pl-4 border-l border-border"
      style={{ marginLeft: `${node.depth * 16}px` }}
    >
      <div className="absolute -left-[5px] top-4 size-2.5 rounded-full bg-primary" />
      <Card className="mb-2">
        <CardContent className="p-3 space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            <TypeBadge type={node.type as LedgerEntryType} />
            {node.agent && (
              <span className="text-xs font-mono text-muted-foreground">{node.agent}</span>
            )}
            <OutcomeIcon outcome={node.outcome} />
            {node.duration_ms !== null && (
              <span className="text-xs text-muted-foreground tabular-nums">
                {node.duration_ms}ms
              </span>
            )}
            <span className="text-xs text-muted-foreground ml-auto">{formatTs(node.ts)}</span>
          </div>

          {node.objective_id && (
            <div className="text-xs">
              <span className="text-muted-foreground">objective: </span>
              <code className="font-mono">{node.objective_id}</code>
            </div>
          )}

          {node.tags.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {node.tags.map((t) => (
                <Badge key={t} variant="outline" className="text-[10px] font-mono">
                  {t}
                </Badge>
              ))}
            </div>
          )}

          {Object.keys(node.data).length > 0 && (
            <pre className="text-[11px] bg-muted p-2 rounded overflow-x-auto font-mono leading-relaxed">
              {JSON.stringify(node.data, null, 2)}
            </pre>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

interface ChainDrawerProps {
  open: boolean;
  rootEntryId: string | null;
  onClose: () => void;
}

export function ChainDrawer({ open, rootEntryId, onClose }: ChainDrawerProps) {
  const service = useHarnessService();
  const [chain, setChain] = useState<LedgerEntry[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open || !rootEntryId) return;
    setLoading(true);
    service.getChain(rootEntryId).then((c) => {
      setChain(c);
      setLoading(false);
    });
  }, [open, rootEntryId, service]);

  const nodes = buildTreeView(chain);

  return (
    <Sheet open={open} onOpenChange={(v) => !v && onClose()}>
      <SheetContent className="sm:max-w-xl flex flex-col gap-0">
        <SheetHeader>
          <div className="flex items-center gap-2">
            <Link2 className="size-4 text-primary" />
            <SheetTitle>Causal chain</SheetTitle>
          </div>
          <SheetDescription>
            {rootEntryId && (
              <span className="font-mono text-xs">
                Root: {shortId(rootEntryId)} · {chain.length} entries
              </span>
            )}
          </SheetDescription>
        </SheetHeader>

        <ScrollArea className="flex-1 min-h-0 px-4">
          <div className="py-4">
            {loading ? (
              <div className="space-y-2">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-20 w-full" />
                ))}
              </div>
            ) : nodes.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-8">
                No entries in this chain.
              </p>
            ) : (
              nodes.map((n) => <ChainNodeCard key={n.id} node={n} />)
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
