import { useState, useCallback, useEffect, useRef } from 'react';
import { useUser } from '@/contexts/UserContext';
import { useEvaluationsService } from '../api/evaluationsService';
import type { AvailableModel } from '../types';

export function useAvailableModels() {
  const { accessToken } = useUser();
  const service = useEvaluationsService();
  const loaded = useRef(false);

  const [models, setModels] = useState<AvailableModel[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    if (!accessToken) { setLoading(false); return; }
    try {
      const data = await service.listAvailableModels(accessToken);
      setModels(data);
    } catch (err) {
      console.error('[useAvailableModels] Failed to fetch:', err);
    } finally {
      setLoading(false);
    }
  }, [accessToken, service]);

  useEffect(() => {
    if (loaded.current) return;
    if (!accessToken) { setLoading(false); return; }
    loaded.current = true;
    refresh();
  }, [accessToken, refresh]);

  return { models, loading, refresh };
}
