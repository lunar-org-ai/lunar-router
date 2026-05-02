import { Link2, MousePointer2, Plus } from 'lucide-react';

const ICONS = [MousePointer2, Plus, Link2];

export function AgentSidebar() {
  return (
    <nav
      aria-hidden
      className="flex w-12 shrink-0 flex-col items-center gap-3 border-r border-border/40 py-6"
    >
      {ICONS.map((Icon, index) => (
        <button
          key={index}
          type="button"
          tabIndex={-1}
          className="flex size-8 items-center justify-center rounded-md text-muted-foreground/60 transition-colors hover:bg-card/40 hover:text-muted-foreground"
        >
          <Icon className="size-4" />
        </button>
      ))}
    </nav>
  );
}
