import { useCallback, useEffect, useMemo, useReducer, useRef } from 'react';

import {
  type AgentFramework,
  type AgentPhase,
  clearStoredOnboarding,
  loadStoredOnboarding,
  saveStoredOnboarding,
} from '@/features/agents/state';

const NODE_REVEAL_INTERVAL_MS = 240;
const EVAL_DURATION_MS = 1000;

type State = {
  phase: AgentPhase;
  modalStep: 1 | 2;
  framework: AgentFramework | null;
  name: string;
  revealedCount: number;
  runId: number;
};

type Action =
  | { type: 'open-import' }
  | { type: 'close-modal' }
  | { type: 'select-framework'; framework: AgentFramework }
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
  modalStep: 1,
  framework: null,
  name: '',
  revealedCount: 0,
  runId: 0,
};

const initialState = (): State => {
  const stored = loadStoredOnboarding();
  if (stored) {
    return {
      phase: 'ready',
      modalStep: 1,
      framework: stored.framework,
      name: stored.name,
      revealedCount: Number.MAX_SAFE_INTEGER,
      runId: 1,
    };
  }
  return emptyState;
};

const reducer = (state: State, action: Action): State => {
  switch (action.type) {
    case 'open-import':
      return { ...state, phase: 'modal', modalStep: 1 };
    case 'close-modal':
      return { ...emptyState };
    case 'select-framework':
      return { ...state, framework: action.framework };
    case 'set-name':
      return { ...state, name: action.name };
    case 'advance-step':
      if (state.modalStep === 1 && !state.framework) return state;
      return { ...state, modalStep: 2 };
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
  modalStep: 1 | 2;
  framework: AgentFramework | null;
  name: string;
  revealedCount: number;
  runId: number;
  openImport: () => void;
  closeModal: () => void;
  selectFramework: (framework: AgentFramework) => void;
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
    if (state.revealedCount < totalNodes) {
      const id = setTimeout(() => dispatch({ type: 'reveal-next' }), NODE_REVEAL_INTERVAL_MS);
      timeoutsRef.current.push(id);
      return () => clearTimeout(id);
    }
    const id = setTimeout(() => dispatch({ type: 'start-evaluating' }), 400);
    timeoutsRef.current.push(id);
    return () => clearTimeout(id);
  }, [state.phase, state.revealedCount, totalNodes]);

  useEffect(() => {
    if (state.phase !== 'evaluating') return;
    const id = setTimeout(() => dispatch({ type: 'finish-evaluating' }), EVAL_DURATION_MS);
    timeoutsRef.current.push(id);
    return () => clearTimeout(id);
  }, [state.phase, state.runId]);

  useEffect(() => {
    if (state.phase === 'ready') {
      saveStoredOnboarding({
        framework: state.framework,
        name: state.name,
      });
    }
  }, [state.phase, state.framework, state.name]);

  const openImport = useCallback(() => dispatch({ type: 'open-import' }), []);
  const closeModal = useCallback(() => dispatch({ type: 'close-modal' }), []);
  const selectFramework = useCallback(
    (framework: AgentFramework) => dispatch({ type: 'select-framework', framework }),
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
      modalStep: state.modalStep,
      framework: state.framework,
      name: state.name,
      revealedCount: state.revealedCount,
      runId: state.runId,
      openImport,
      closeModal,
      selectFramework,
      setName,
      advanceStep,
      runSimulation,
      runEval,
      reset,
    }),
    [
      state,
      openImport,
      closeModal,
      selectFramework,
      setName,
      advanceStep,
      runSimulation,
      runEval,
      reset,
    ]
  );
}
