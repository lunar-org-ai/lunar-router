import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useEvaluationsService } from '../../../api/evaluationsService';
import { extractScoresByModel } from './utils';
import type { Evaluation, ModelRanking, EvaluationResultsMap } from './types';

const MAX_RANKINGS = 6;
const MAX_FETCH_BATCH = 20;

export function useModelLeaderboard(evaluations: Evaluation[]) {
  const accessToken = '';
  const service = useEvaluationsService();
  const serviceRef = useRef(service);
  serviceRef.current = service;

  const requestedRef = useRef<Set<string>>(new Set());
  const [fetchedResults, setFetchedResults] = useState<EvaluationResultsMap>({});

  // Stable fetch function that uses ref to avoid stale closures
  const fetchMissingResults = useCallback((ids: string[], token: string) => {
    ids.forEach((id) => requestedRef.current.add(id));

    Promise.allSettled(
      ids.map(async (id) => {
        const data = await serviceRef.current.getEvaluationResults(token, id, {
          include_samples: false,
        });
        return { id, results: data.results };
      })
    ).then((responses) => {
      const entries = responses
        .filter(
          (
            r
          ): r is PromiseFulfilledResult<{
            id: string;
            results: NonNullable<Evaluation['results']>;
          }> => r.status === 'fulfilled'
        )
        .map((r) => r.value);

      if (entries.length > 0) {
        setFetchedResults((prev) => ({
          ...prev,
          ...Object.fromEntries(entries.map(({ id, results }) => [id, results])),
        }));
      }
    });
  }, []);

  // Fetch missing results for completed evaluations without score data
  useEffect(() => {
    const idsToFetch = evaluations
      .filter((e) => e.status === 'completed')
      .filter((e) => {
        const merged: Evaluation = {
          ...e,
          results: e.results ?? fetchedResults[e.id],
        };
        return (
          Object.keys(extractScoresByModel(merged)).length === 0 && !requestedRef.current.has(e.id)
        );
      })
      .map((e) => e.id)
      .slice(0, MAX_FETCH_BATCH);

    if (idsToFetch.length === 0) return;

    fetchMissingResults(idsToFetch, accessToken);
  }, [evaluations, fetchedResults, fetchMissingResults]);

  // Merge evaluations with fetched results
  const mergedEvaluations = useMemo(
    () =>
      evaluations.map((e) => ({
        ...e,
        results: e.results ?? fetchedResults[e.id],
      })),
    [evaluations, fetchedResults]
  );

  // Compute rankings
  const rankings: ModelRanking[] = useMemo(() => {
    const scores: Record<string, { total: number; count: number }> = {};

    for (const evaluation of mergedEvaluations) {
      if (evaluation.status !== 'completed') continue;

      for (const [model, score] of Object.entries(extractScoresByModel(evaluation))) {
        if (!scores[model]) scores[model] = { total: 0, count: 0 };
        scores[model].total += score;
        scores[model].count += 1;
      }
    }

    return Object.entries(scores)
      .map(([model, { total, count }]) => ({
        model,
        avgScore: total / count,
        evalCount: count,
      }))
      .sort((a, b) => b.avgScore - a.avgScore)
      .slice(0, MAX_RANKINGS);
  }, [mergedEvaluations]);

  const completedCount = mergedEvaluations.filter((e) => e.status === 'completed').length;
  const topScore = rankings[0]?.avgScore ?? 1;

  return { rankings, completedCount, topScore };
}
