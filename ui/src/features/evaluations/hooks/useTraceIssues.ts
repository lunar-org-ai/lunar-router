import { useState, useCallback, useEffect, useRef } from 'react';
import { useUser } from '@/contexts/UserContext';
import { useEvaluationsService } from '../api/evaluationsService';
import type { TraceIssue, TraceScan } from '../types/evaluationsTypes';

const POLL_INTERVAL_MS = 1_000;

export interface ScheduleConfig {
  enabled: boolean;
  interval_seconds: number;
  days_lookback: number;
  trace_limit: number;
  last_run_at: string | null;
  next_run_at: string | null;
  total_runs: number;
  total_issues_found: number;
}

function normalizeIssue(raw: Record<string, unknown>): TraceIssue {
  return {
    id: (raw.issue_id as string) || (raw.id as string),
    trace_id: raw.trace_id,
    type: raw.type,
    severity: raw.severity,
    title: raw.title,
    description: raw.description,
    ai_confidence: raw.ai_confidence,
    model_id: raw.model_id,
    trace_input: raw.trace_input,
    trace_output: raw.trace_output,
    detected_at: raw.detected_at,
    resolved: raw.resolved,
    dismissed: raw.dismissed,
    suggested_action: raw.suggested_action,
    suggested_eval_config: raw.suggested_eval_config,
  } as TraceIssue;
}

export function useTraceIssues() {
  const { accessToken } = useUser();
  const service = useEvaluationsService();

  // Backend trace-issue endpoints don't require auth — use empty string as fallback
  const token = accessToken ?? '';

  const [issues, setIssues] = useState<TraceIssue[]>([]);
  const [scanning, setScanning] = useState(false);
  const [lastScan, setLastScan] = useState<TraceScan | null>(null);
  const [loading, setLoading] = useState(true);
  const [schedule, setSchedule] = useState<ScheduleConfig | null>(null);
  const [scheduleRunning, setScheduleRunning] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchIssues = useCallback(async () => {
    try {
      const data = await service.listTraceIssues(token);
      setIssues((data.issues || []).map(normalizeIssue));
    } catch (err) {
      console.error('[useTraceIssues] fetch failed:', err);
    } finally {
      setLoading(false);
    }
  }, [token, service]);

  const fetchSchedule = useCallback(async () => {
    try {
      const data = await service.getScheduleConfig(token);
      setSchedule(data.schedule as ScheduleConfig);
      setScheduleRunning(data.running as boolean);
    } catch (err) {
      console.error('[useTraceIssues] schedule fetch failed:', err);
    }
  }, [token, service]);

  useEffect(() => {
    fetchIssues();
    fetchSchedule();
  }, [fetchIssues, fetchSchedule]);

  const triggerScan = useCallback(async () => {
    if (scanning) return;
    setScanning(true);

    try {
      const { scan_id: scanId } = await service.triggerTraceScan(token);

      setLastScan({
        id: scanId,
        status: 'running',
        traces_scanned: 0,
        issues_found: 0,
        started_at: new Date().toISOString(),
      });

      const poll = setInterval(async () => {
        try {
          const s = await service.getTraceScanStatus(token, scanId);
          const status = s.status as TraceScan['status'];

          setLastScan({
            id: s.scan_id || scanId,
            status,
            traces_scanned: s.traces_scanned ?? 0,
            issues_found: s.issues_found ?? 0,
            started_at: s.started_at,
            completed_at: s.completed_at,
          });

          if (status === 'completed' || status === 'failed') {
            clearInterval(poll);
            pollRef.current = null;
            setScanning(false);
            fetchIssues();
          }
        } catch (err) {
          console.error('[useTraceIssues] poll error:', err);
          clearInterval(poll);
          pollRef.current = null;
          setScanning(false);
        }
      }, POLL_INTERVAL_MS);

      pollRef.current = poll;
    } catch (err) {
      console.error('[useTraceIssues] scan trigger failed:', err);
      setScanning(false);
    }
  }, [scanning, token, service, fetchIssues]);

  const resolveIssue = useCallback(
    async (id: string) => {
      try {
        await service.resolveTraceIssue(token, id);
        setIssues((prev) => prev.map((i) => (i.id === id ? { ...i, resolved: true } : i)));
      } catch (err) {
        console.error('[useTraceIssues] resolve failed:', err);
      }
    },
    [token, service]
  );

  const dismissIssue = useCallback(
    async (id: string, reason?: string) => {
      try {
        await service.dismissTraceIssue(token, id, reason);
        setIssues((prev) =>
          prev.map((i) => (i.id === id ? { ...i, resolved: true, dismissed: true } : i))
        );
      } catch (err) {
        console.error('[useTraceIssues] dismiss failed:', err);
      }
    },
    [token, service]
  );

  const updateSchedule = useCallback(
    async (config: Partial<ScheduleConfig>) => {
      try {
        const data = await service.updateScheduleConfig(token, config);
        setSchedule(data.schedule as ScheduleConfig);
        setScheduleRunning(data.running as boolean);
      } catch (err) {
        console.error('[useTraceIssues] schedule update failed:', err);
      }
    },
    [token, service]
  );

  const refresh = useCallback(() => {
    setLoading(true);
    fetchIssues();
  }, [fetchIssues]);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  return {
    issues,
    scanning,
    lastScan,
    triggerScan,
    resolveIssue,
    dismissIssue,
    refresh,
    loading,
    schedule,
    scheduleRunning,
    updateSchedule,
  };
}
