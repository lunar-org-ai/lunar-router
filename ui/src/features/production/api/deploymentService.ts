import { API_BASE_URL } from '@/config/api';
import { INSTANCE_TYPES } from '@/features/production/constants/instanceTypes';
import type { DeploymentData, DeploymentMetricsData } from '@/types/deploymentTypes';

const API_BASE = API_BASE_URL;

// Types
export interface DeploymentRequest {
  model_id: string;
  instance_type: string;
  scaling?: { min: number; max: number };
}

export interface DeploymentResponse {
  deployment_id: string;
  endpoint_name?: string;
  status: string;
  model_id?: string;
  instance_type?: string;
  updated_at?: string;
  tenant_id?: string;
  scaling?: { min: number; max: number };
  error_message?: string;
  error_code?: string;
  endpoint_url?: string;
}

// Internal helpers
function buildVllmArgs(data: Partial<DeploymentData>): string {
  const args: string[] = [];
  const opts = data.modelOptions;

  if (opts) {
    if (opts.maxTokens !== undefined) args.push(`--max-model-len ${opts.maxTokens}`);
    if (opts.dtype && opts.dtype !== 'auto') args.push(`--dtype ${opts.dtype}`);
    if (opts.gpuMemoryUtilization !== undefined)
      args.push(`--gpu-memory-utilization ${opts.gpuMemoryUtilization}`);
    if (opts.maxNumSeqs !== undefined) args.push(`--max-num-seqs ${opts.maxNumSeqs}`);
    if (opts.blockSize !== undefined) args.push(`--block-size ${opts.blockSize}`);
    if (opts.swapSpace !== undefined) args.push(`--swap-space ${opts.swapSpace}`);
  }

  const instance = INSTANCE_TYPES.find((i) => i.id === data.selectedInstance);
  const tensorParallelSize = instance?.specs.tensorParallelSize ?? 1;

  // Apply defaults when not explicitly set
  if (!opts?.dtype || opts.dtype === 'auto') args.push('--dtype bfloat16');
  if (!opts?.gpuMemoryUtilization) args.push('--gpu-memory-utilization 0.92');

  args.push(`--tensor-parallel-size ${tensorParallelSize}`);

  return args.join(' ');
}

async function parseErrorResponse(res: Response): Promise<string> {
  const text = await res.text().catch(() => '');
  try {
    const json = JSON.parse(text);
    return json.message ?? json.error ?? text;
  } catch {
    return text;
  }
}

function authHeaders(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token}` };
}

function jsonHeaders(token: string): Record<string, string> {
  return { ...authHeaders(token), 'Content-Type': 'application/json' };
}

// API functions
export async function createDeploymentApi(
  idToken: string,
  payload: DeploymentRequest,
  data: Partial<DeploymentData>
): Promise<DeploymentResponse> {
  const body = {
    ...payload,
    config: { vllm_args: buildVllmArgs(data) },
  };

  const res = await fetch(`${API_BASE}/v1/deployments`, {
    method: 'POST',
    headers: jsonHeaders(idToken),
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const message = await parseErrorResponse(res);
    throw new Error(message || `Failed to create deployment: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function _getDeploymentStatusApi(
  token: string,
  id: string
): Promise<DeploymentResponse> {
  const res = await fetch(`${API_BASE}/v1/deployments/${id}/status`, {
    headers: authHeaders(token),
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch deployment status: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function listDeploymentsApi(
  accessToken: string,
  statuses?: string[]
): Promise<DeploymentResponse[]> {
  const qs = statuses?.length ? `?statuses=${encodeURIComponent(statuses.join(','))}` : '';
  const res = await fetch(`${API_BASE}/v1/deployments${qs}`, {
    headers: authHeaders(accessToken),
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch deployments: ${res.status} ${res.statusText}`);
  }

  const result = await res.json();
  return result.deployments ?? [];
}

export async function deleteDeploymentApi(accessToken: string, id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/v1/deployments/${id}`, {
    method: 'DELETE',
    headers: authHeaders(accessToken),
  });

  if (!res.ok) {
    throw new Error(`Failed to delete deployment: ${res.status} ${res.statusText}`);
  }
}

export async function _getDeploymentMetricsApi(
  accessToken: string,
  deploymentId: string
): Promise<DeploymentMetricsData> {
  const res = await fetch(`${API_BASE}/v1/deployments/${deploymentId}/metrics`, {
    headers: jsonHeaders(accessToken),
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch deployment metrics: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function pauseDeploymentApi(
  accessToken: string,
  deploymentId: string
): Promise<DeploymentResponse> {
  const res = await fetch(`${API_BASE}/v1/deployments/${deploymentId}/pause`, {
    method: 'PATCH',
    headers: jsonHeaders(accessToken),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(
      `Failed to pause deployment: ${res.status} ${res.statusText}${text ? ` - ${text}` : ''}`
    );
  }

  return res.json();
}

export async function resumeDeploymentApi(
  accessToken: string,
  deploymentId: string
): Promise<DeploymentResponse> {
  const res = await fetch(`${API_BASE}/v1/deployments/${deploymentId}/resume`, {
    method: 'PATCH',
    headers: jsonHeaders(accessToken),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(
      `Failed to resume deployment: ${res.status} ${res.statusText}${text ? ` - ${text}` : ''}`
    );
  }

  return res.json();
}
