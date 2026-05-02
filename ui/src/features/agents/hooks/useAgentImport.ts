import { useCallback, useEffect, useMemo, useReducer, useRef } from 'react';

import {
  type AgentFramework,
  type AgentPhase,
  type OnboardingMode,
  clearStoredOnboarding,
  loadStoredOnboarding,
  saveStoredOnboarding,
} from '@/features/agents/state';
import { SUPPORT_TEMPLATE_ID } from '@/features/agents/templates';

const NODE_REVEAL_INTERVAL_MS = 320;
const BUILD_DURATION_MS = 620;
const EVAL_DURATION_MS = 1300;

type State = {
  phase: AgentPhase;
  mode: OnboardingMode | null;
  modalStep: 1 | 2;
  framework: AgentFramework | null;
  templateId: string | null;
  name: string;
  revealedCount: number;
  runId: number;
};

type Action =
  | { type: 'open-import' }
  | { type: 'open-create' }
  | { type: 'close-modal' }
  | { type: 'select-framework'; framework: AgentFramework }
  | { type: 'select-template'; templateId: string }
  | { type: 'set-name'; name: string }
  | { type: 'advance-step' }
  | { type: 'start-discovering' }
  | { type: 'reveal-next' }
  | { type: 'start-evaluating' }
  | { type: 'finish-evaluating' }
  | { type: 'rerun-eval' }
  | { type: 'reset' };

const emptyState: State = {
  phase: 'empty',
  mode: null,
  modalStep: 1,
  framework: null,
  templateId: null,
  name: '',
  revealedCount: 0,
  runId: 0,
};

const initialState = (): State => {
  const stored = loadStoredOnboarding();
  if (stored) {
    return {
      phase: 'ready',
      mode: stored.mode,
      modalStep: 1,
      framework: stored.framework,
      templateId: stored.templateId ?? SUPPORT_TEMPLATE_ID,
      name: stored.name,
      revealedCount: Number.MAX_SAFE_INTEGER,
      runId: 1,
    };
  }
  return emptyState;
};

const stepReadyForCreate = (state: State): boolean => {
  if (state.modalStep === 1) return state.templateId !== null;
  return true;
};

const stepReadyForImport = (state: State): boolean => {
  if (state.modalStep === 1) return state.framework !== null;
  return true;
};

const reducer = (state: State, action: Action): State => {
  switch (action.type) {
    case 'open-import':
      return {
        ...state,
        phase: 'modal',
        mode: 'import',
        modalStep: 1,
        framework: state.framework,
        templateId: SUPPORT_TEMPLATE_ID,
      };
    case 'open-create':
      return {
        ...state,
        phase: 'modal',
        mode: 'create',
        modalStep: 1,
        templateId: state.mode === 'create' ? state.templateId : null,
      };
    case 'close-modal':
      return { ...emptyState };
    case 'select-framework':
      return { ...state, framework: action.framework };
    case 'select-template':
      return { ...state, templateId: action.templateId };
    case 'set-name':
      return { ...state, name: action.name };
    case 'advance-step': {
      const ready = state.mode === 'create' ? stepReadyForCreate(state) : stepReadyForImport(state);
      if (!ready) return state;
      return { ...state, modalStep: 2 };
    }
    case 'start-discovering':
      return { ...state, phase: 'discovering', revealedCount: 0 };
    case 'reveal-next':
      return { ...state, revealedCount: state.revealedCount + 1 };
    case 'start-evaluating':
      return { ...state, phase: 'evaluating', runId: state.runId + 1 };
    case 'finish-evaluating':
      return { ...state, phase: 'ready' };
    case 'rerun-eval':
      return { ...state, phase: 'evaluating', runId: state.runId + 1 };
    case 'reset':
      return { ...emptyState };
    default:
      return state;
  }
};

type UseAgentImportResult = {
  phase: AgentPhase;
  mode: OnboardingMode | null;
  modalStep: 1 | 2;
  framework: AgentFramework | null;
  templateId: string | null;
  name: string;
  revealedCount: number;
  runId: number;
  openImport: () => void;
  openCreate: () => void;
  closeModal: () => void;
  selectFramework: (framework: AgentFramework) => void;
  selectTemplate: (templateId: string) => void;
  setName: (name: string) => void;
  advanceStep: () => void;
  runSimulation: () => void;
  runEval: () => void;
  reset: () => void;
};

export function useAgentImport(totalNodes: number): UseAgentImportResult {
  const [state, dispatch] = useReducer(reducer, undefined, initialState);
  const timeoutsRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  const clearTimers = useCallback(() => {
    timeoutsRef.current.forEach(clearTimeout);
    timeoutsRef.current = [];
  }, []);

  useEffect(() => clearTimers, [clearTimers]);

  useEffect(() => {
    if (state.phase !== 'discovering') return;
    if (state.mode === 'create') {
      const id = setTimeout(() => dispatch({ type: 'start-evaluating' }), BUILD_DURATION_MS);
      timeoutsRef.current.push(id);
      return () => clearTimeout(id);
    }
    if (state.revealedCount < totalNodes) {
      const id = setTimeout(() => dispatch({ type: 'reveal-next' }), NODE_REVEAL_INTERVAL_MS);
      timeoutsRef.current.push(id);
      return () => clearTimeout(id);
    }
    const id = setTimeout(() => dispatch({ type: 'start-evaluating' }), 400);
    timeoutsRef.current.push(id);
    return () => clearTimeout(id);
  }, [state.phase, state.mode, state.revealedCount, totalNodes]);

  useEffect(() => {
    if (state.phase !== 'evaluating') return;
    const id = setTimeout(() => dispatch({ type: 'finish-evaluating' }), EVAL_DURATION_MS);
    timeoutsRef.current.push(id);
    return () => clearTimeout(id);
  }, [state.phase, state.runId]);

  useEffect(() => {
    if (state.phase === 'ready' && state.mode) {
      saveStoredOnboarding({
        mode: state.mode,
        framework: state.framework,
        templateId: state.templateId,
        name: state.name,
      });
    }
  }, [state.phase, state.mode, state.framework, state.templateId, state.name]);

  const openImport = useCallback(() => dispatch({ type: 'open-import' }), []);
  const openCreate = useCallback(() => dispatch({ type: 'open-create' }), []);
  const closeModal = useCallback(() => dispatch({ type: 'close-modal' }), []);
  const selectFramework = useCallback(
    (framework: AgentFramework) => dispatch({ type: 'select-framework', framework }),
    []
  );
  const selectTemplate = useCallback(
    (templateId: string) => dispatch({ type: 'select-template', templateId }),
    []
  );
  const setName = useCallback((name: string) => dispatch({ type: 'set-name', name }), []);
  const advanceStep = useCallback(() => dispatch({ type: 'advance-step' }), []);

  const runSimulation = useCallback(() => {
    clearTimers();
    dispatch({ type: 'start-discovering' });
  }, [clearTimers]);

  const runEval = useCallback(() => {
    if (state.phase !== 'ready') return;
    clearTimers();
    dispatch({ type: 'rerun-eval' });
  }, [state.phase, clearTimers]);

  const reset = useCallback(() => {
    clearTimers();
    clearStoredOnboarding();
    dispatch({ type: 'reset' });
  }, [clearTimers]);

  return useMemo(
    () => ({
      phase: state.phase,
      mode: state.mode,
      modalStep: state.modalStep,
      framework: state.framework,
      templateId: state.templateId,
      name: state.name,
      revealedCount: state.revealedCount,
      runId: state.runId,
      openImport,
      openCreate,
      closeModal,
      selectFramework,
      selectTemplate,
      setName,
      advanceStep,
      runSimulation,
      runEval,
      reset,
    }),
    [
      state,
      openImport,
      openCreate,
      closeModal,
      selectFramework,
      selectTemplate,
      setName,
      advanceStep,
      runSimulation,
      runEval,
      reset,
    ]
  );
}
