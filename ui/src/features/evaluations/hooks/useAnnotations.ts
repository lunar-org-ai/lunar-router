import { useState, useCallback, useEffect, useRef } from 'react';
import { useEvaluationsService } from '../api/evaluationsService';
import type {
  AnnotationQueue,
  AnnotationItem,
  AnnotationRubric,
  AnnotationAnalytics,
} from '../types/evaluationsTypes';
import { downloadBlob } from '../utils';

export function useAnnotations() {
  const accessToken = '';
  const service = useEvaluationsService();
  const loaded = useRef(false);

  const [queues, setQueues] = useState<AnnotationQueue[]>([]);
  const [items, setItems] = useState<AnnotationItem[]>([]);
  const [currentItem, setCurrentItem] = useState<AnnotationItem | null>(null);
  const [loading, setLoading] = useState(false);

  const refreshQueues = useCallback(async () => {
    try {
      const data = await service.listAnnotationQueues(accessToken);
      setQueues(data);
    } catch (err) {
      console.error('[useAnnotations] refreshQueues failed:', err);
    }
  }, [service]);

  useEffect(() => {
    if (loaded.current) return;
    loaded.current = true;
    setLoading(true);
    refreshQueues().finally(() => setLoading(false));
  }, [refreshQueues]);

  const createQueue = useCallback(
    async (data: {
      name: string;
      description?: string;
      datasetId: string;
      evaluationId?: string;
      rubric: AnnotationRubric;
    }) => {
      const rubric = data.rubric.criteria.map((c) => ({
        name: c.name,
        description: c.description,
        scale_min: c.scale.min,
        scale_max: c.scale.max,
      }));
      const queue = await service.createAnnotationQueue(accessToken, {
        name: data.name,
        dataset_id: data.datasetId,
        rubric,
      });
      setQueues((prev) => [queue, ...prev]);
    },
    [service]
  );

  const deleteQueue = useCallback(
    async (id: string) => {
      await service.deleteAnnotationQueue(accessToken, id);
      setQueues((prev) => prev.filter((q) => q.id !== id));
    },
    [service]
  );

  const loadItems = useCallback(
    async (queueId: string) => {
      setLoading(true);
      try {
        const [allItems, next] = await Promise.all([
          service.listAnnotationItems(accessToken, queueId),
          service.getNextAnnotationItem(accessToken, queueId),
        ]);
        setItems(allItems);
        setCurrentItem(next);
      } catch (err) {
        console.error('[useAnnotations] loadItems failed:', err);
      } finally {
        setLoading(false);
      }
    },
    [service]
  );

  const getNextItem = useCallback(
    async (queueId: string) => {
      try {
        const next = await service.getNextAnnotationItem(accessToken, queueId);
        setCurrentItem(next);
      } catch (err) {
        console.error('[useAnnotations] getNextItem failed:', err);
        setCurrentItem(null);
      }
    },
    [service]
  );

  const submitAnnotation = useCallback(
    async (itemId: string, scores: Record<string, number>, notes?: string) => {
      if (!currentItem) return;
      const result = await service.submitAnnotationItem(
        accessToken,
        currentItem.queue_id,
        itemId,
        scores,
        notes
      );
      setItems((prev) => prev.map((i) => (i.id === itemId ? result : i)));
      setQueues((prev) =>
        prev.map((q) =>
          q.id === currentItem.queue_id ? { ...q, completed_items: q.completed_items + 1 } : q
        )
      );
    },
    [service, currentItem]
  );

  const skipItem = useCallback(
    async (itemId: string) => {
      if (!currentItem) return;
      const result = await service.skipAnnotationItem(accessToken, currentItem.queue_id, itemId);
      setItems((prev) => prev.map((i) => (i.id === itemId ? result : i)));
      setQueues((prev) =>
        prev.map((q) =>
          q.id === currentItem.queue_id ? { ...q, skipped_items: q.skipped_items + 1 } : q
        )
      );
    },
    [service, currentItem]
  );

  const exportAnnotations = useCallback(
    async (queueId: string, format: 'json' | 'csv' = 'json') => {
      const data = await service.exportAnnotations(accessToken, queueId, format);

      if (format === 'csv' && data.csv) {
        downloadBlob(
          data.csv,
          data.filename || `annotations_${queueId}.csv`,
          'text/csv;charset=utf-8;'
        );
      } else {
        downloadBlob(
          JSON.stringify(data, null, 2),
          `annotations_${queueId}.json`,
          'application/json'
        );
      }
    },
    [service]
  );

  const getAnalytics = useCallback(
    async (queueId: string): Promise<AnnotationAnalytics | null> => {
      try {
        return await service.getAnnotationAnalytics(accessToken, queueId);
      } catch (err) {
        console.error('[useAnnotations] getAnalytics failed:', err);
        return null;
      }
    },
    [service]
  );

  return {
    queues,
    items,
    currentItem,
    loading,
    refreshQueues,
    createQueue,
    deleteQueue,
    loadItems,
    getNextItem,
    submitAnnotation,
    skipItem,
    exportAnnotations,
    getAnalytics,
  };
}
