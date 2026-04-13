export function formatDateWithYear(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  if (isNaN(d.getTime())) return '—';
  return d.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

export function formatStatus(status: string): string {
  const map: Record<string, string> = {
    completed: 'Completed',
    running: 'In Progress',
    failed: 'Failed',
    pending: 'Pending',
    none: 'Waiting for data',
    train: 'Training recommended',
    wait: 'Gathering data',
    promote: 'Model promoted',
    reject: 'Run rejected',
    promoted: 'Promoted',
    rejected: 'Rejected',
  };
  return map[status] ?? status.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatPercent(value: number, alreadyFraction = true): string {
  const pct = alreadyFraction ? value * 100 : value;
  return `${pct.toFixed(1)}%`;
}

export function formatNumber(value: number): string {
  return value.toLocaleString();
}

export function costTierLabel(tier: string): string {
  const labels: Record<string, string> = {
    low: 'Low',
    medium: 'Medium',
    high: 'High',
    premium: 'Premium',
  };
  return labels[tier] ?? tier;
}

export function exportTableToCsv(
  headers: string[],
  rows: (string | number)[][],
  filename: string
): void {
  const escape = (cell: string | number): string => {
    const str = String(cell);
    if (str.includes(',') || str.includes('"') || str.includes('\n')) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };
  const csv = [headers.map(escape).join(','), ...rows.map((row) => row.map(escape).join(','))].join(
    '\n'
  );
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${filename}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}
