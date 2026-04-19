import { useState, useEffect, useCallback } from 'react';
import { usePostHog } from 'posthog-js/react';

import {
  createDeploymentApi,
  listDeploymentsApi,
  deleteDeploymentApi,
  pauseDeploymentApi,
  resumeDeploymentApi,
  type DeploymentRequest,
  type DeploymentResponse,
} from '../api/deploymentService';

import type { DeploymentData } from '@/types/deploymentTypes';
import { useUser } from '@/contexts/UserContext';

const POLLING_INTERVAL_MS = 5_000;

const ACTIVE_STATUSES = [
  'in_service',
  'starting',
  'creating',
  'failed',
  'stopped',
  'deleting',
  'paused',
  'pausing',
  'resuming',
] as const;

function parseErrorMessage(errorMessage?: string) {
  if (!errorMessage) return {};

  const userMessageMatch = errorMessage.match(
    /user_message=(.+?)(?:,\s*error_code=|,\s*[a-zA-Z_]+=|}|$)/
  );
  const errorCodeMatch = errorMessage.match(/error_code=([^,}]+)/);

  return {
    userMessage: userMessageMatch?.[1]?.trim() ?? errorMessage,
    errorCode: errorCodeMatch?.[1]?.trim(),
  };
}

function mapResponseToDeploymentData(apiDep: DeploymentResponse): DeploymentData {
  const { userMessage, errorCode } = parseErrorMessage(apiDep.error_message);

  return {
    id: apiDep.deployment_id,
    deployment_id: apiDep.deployment_id,
    name: `Deployment ${apiDep.deployment_id.slice(0, 8)}`,
    selectedModel: apiDep.model_id ?? 'unknown',
    selectedInstance: apiDep.instance_type ?? 'local',
    endpoint_url: apiDep.endpoint_url ?? '',
    createdAt: apiDep.updated_at ?? new Date().toISOString(),
    status: apiDep.status as DeploymentData['status'],
    error_message: userMessage,
    error_code: errorCode ?? apiDep.error_code,
    modelOptions: {
      maxTokens: 4096,
      dtype: 'bfloat16',
      gpuMemoryUtilization: 0.92,
      maxNumSeqs: 256,
      blockSize: 16,
      swapSpace: 4,
      temperature: 0.7,
      topP: 0.9,
      topK: 50,
    },
    autoscalingConfig: {
      enabled: true,
      maxReplicas: 1,
      versionComment: '',
    },
  };
}

interface State {
  deployments: DeploymentData[];
  loading: boolean;
  creating: boolean;
  error: string | null;
}

export function useDeployments(
  onDeploymentReady?: (d: DeploymentData) => void,
  onDeploymentFailed?: (d: DeploymentData) => void
) {
  const { accessToken, idToken } = useUser();
  const posthog = usePostHog();

  const [state, setState] = useState<State>({
    deployments: [],
    loading: true,
    creating: false,
    error: null,
  });

  const optimisticStatusUpdate = useCallback((id: string, status: DeploymentData['status']) => {
    setState((prev) => ({
      ...prev,
      deployments: prev.deployments.map((d) => (d.id === id ? { ...d, status } : d)),
    }));
  }, []);

  const listDeployments = useCallback(async () => {
    try {
      const apiData = await listDeploymentsApi(accessToken ?? '', [...ACTIVE_STATUSES]);
      const mapped = apiData.map(mapResponseToDeploymentData);

      setState((prev) => ({
        ...prev,
        deployments: mapped,
        loading: false,
        error: null,
      }));
      return mapped;
    } catch (err) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'Error fetching deployments',
      }));
      return [];
    }
  }, [accessToken]);

  const createDeployment = useCallback(
    async (
      payload: DeploymentRequest,
      data: Omit<DeploymentData, 'id' | 'status' | 'createdAt'>
    ) => {
      if (!idToken) throw new Error('ID token not available.');

      setState((prev) => ({ ...prev, creating: true, error: null }));

      try {
        const res = await createDeploymentApi(idToken, payload, data);

        const deployment: DeploymentData = {
          ...data,
          id: res.deployment_id,
          deployment_id: res.deployment_id,
          status: 'pending',
          createdAt: new Date().toISOString(),
        };

        setState((prev) => ({
          ...prev,
          creating: false,
          deployments: [...prev.deployments, deployment],
        }));

        posthog.capture('deployment_created', payload);

        return { deploymentData: deployment, deploymentRes: res };
      } catch (err) {
        setState((prev) => ({
          ...prev,
          creating: false,
          error: err instanceof Error ? err.message : 'Error creating deployment',
        }));
        throw err;
      }
    },
    [idToken, posthog]
  );

  const deleteDeployment = useCallback(
    async (id: string) => {
      try {
        optimisticStatusUpdate(id, 'deleting');
        await deleteDeploymentApi(accessToken ?? '', id);

        posthog.capture('deployment_deleted');

        // Re-fetch to get the updated list without the deleted deployment
        await listDeployments();
        return true;
      } catch (err) {
        // Revert optimistic update on error by re-fetching
        await listDeployments();
        setState((prev) => ({
          ...prev,
          error: err instanceof Error ? err.message : 'Error deleting deployment',
        }));
        return false;
      }
    },
    [accessToken, listDeployments, optimisticStatusUpdate, posthog]
  );

  const pauseDeployment = useCallback(
    async (id: string) => {
      try {
        optimisticStatusUpdate(id, 'pausing');
        await pauseDeploymentApi(accessToken ?? '', id);
        posthog.capture('deployment_paused');
        await listDeployments();
        return true;
      } catch (_err) {
        await listDeployments();
        return false;
      }
    },
    [accessToken, listDeployments, optimisticStatusUpdate, posthog]
  );

  const resumeDeployment = useCallback(
    async (id: string) => {
      try {
        optimisticStatusUpdate(id, 'resuming');
        await resumeDeploymentApi(accessToken ?? '', id);
        posthog.capture('deployment_resumed');
        await listDeployments();
        return true;
      } catch {
        await listDeployments();
        return false;
      }
    },
    [accessToken, listDeployments, optimisticStatusUpdate, posthog]
  );

  // Polling for status updates every 5 seconds
  useEffect(() => {
    const interval = setInterval(async () => {
      const fresh = await listDeployments();

      setState((prev) => {
        fresh.forEach((incoming) => {
          const previous = prev.deployments.find((d) => d.id === incoming.id);

          if (
            incoming.status === 'in_service' &&
            previous &&
            ['pending', 'starting', 'creating'].includes(previous.status)
          ) {
            onDeploymentReady?.(incoming);
          }

          if (incoming.status === 'failed' && previous?.status !== 'failed') {
            onDeploymentFailed?.(incoming);
          }
        });

        return prev;
      });
    }, POLLING_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [accessToken, listDeployments, onDeploymentReady, onDeploymentFailed]);

  return {
    ...state,
    listDeployments,
    createDeployment,
    deleteDeployment,
    pauseDeployment,
    resumeDeployment,
    clearError: () => setState((p) => ({ ...p, error: null })),
  };
}
