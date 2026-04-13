import { API_BASE_URL } from '@/config/api';
import type {
  EfficiencyData,
  ModelPerformanceData,
  TrainingActivityData,
  RoutingIntelligenceData,
  AdvisorConfigData,
} from '../types';

const API_BASE = API_BASE_URL;

async function apiCall<T>(endpoint: string): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
  });
  if (!response.ok) {
    throw new Error(`API Error: ${response.status}`);
  }
  return response.json();
}

export async function fetchEfficiencyData(days: number): Promise<EfficiencyData> {
  return apiCall(`/v1/intelligence/efficiency?days=${days}`);
}

export async function fetchModelPerformanceData(): Promise<ModelPerformanceData> {
  return apiCall('/v1/intelligence/models');
}

export async function fetchTrainingActivityData(days: number): Promise<TrainingActivityData> {
  return apiCall(`/v1/intelligence/training?days=${days}`);
}

export async function fetchRoutingIntelligenceData(days: number): Promise<RoutingIntelligenceData> {
  return apiCall(`/v1/intelligence/routing?days=${days}`);
}

export async function fetchAdvisorConfig(): Promise<AdvisorConfigData> {
  return apiCall('/v1/intelligence/advisor');
}
