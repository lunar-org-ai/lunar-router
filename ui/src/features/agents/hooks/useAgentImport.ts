import { useCallback, useEffect, useMemo, useReducer, useRef } from 'react';

import {
  type AgentFramework,
  type AgentPhase,
  clearStoredImport,
  loadStoredImport,
  saveStoredImport,
} from '@/features/agents/state';

const NODE_REVEAL_INTERVAL_MS = 380;
const EVAL_DURATION_MS = 1600;

type State = {
  phase: AgentPhase;
  modalStep: 1 | 2;
  framework: AgentFramework | null;
  revealedCount: number;
  runId: number;
};

type Action =
  | { type: 'open-modal' }
  | { type: 'close-modal' }
  | { type: 'select-framework'; framework: AgentFramework }
  | { type: 'advance-step' }
  | { type: 'start-discovering'; framework: AgentFramework }
  | { type: 'reveal-next' }
  | { type: 'start-evaluating' }
  | { type: 'finish-evaluating' }
  | { type: 'rerun-eval' }
  | { type: 'reset' };

const initialState = (totalNodes: number): State => {
  const stored = loadStoredImport();
  if (stored) {
    return {
      phase: 'ready',
      modalStep: 1,
      framework: stored.framework,
      revealedCount: totalNodes,
      runId: 1,
    };
  }
  return {
    phase: 'empty',
    modalStep: 1,
    framework: null,
    revealedCount: 0,
    runId: 0,
  };
};

const reducer = (state: State, action: Action): State => {
  switch (action.type) {
    case 'open-modal':
      return { ...state, phase: 'modal', modalStep: 1 };
    case 'close-modal':
      return { ...state, phase: 'empty', modalStep: 1 };
    case 'select-framework':
      return { ...state, framework: action.framework };
    case 'advance-step':
      return state.framework ? { ...state, modalStep: 2 } : state;
    case 'start-discovering':
      return {
        ...state,
        phase: 'discovering',
        framework: action.framework,
        revealedCount: 0,
      };
    case 'reveal-next':
      return { ...state, revealedCount: state.revealedCount + 1 };
    case 'start-evaluating':
      return { ...state, phase: 'evaluating', runId: state.runId + 1 };
    case 'finish-evaluating':
      return { ...state, phase: 'ready' };
    case 'rerun-eval':
      return { ...state, phase: 'evaluating', runId: state.runId + 1 };
    case 'reset':
      return {
        phase: 'empty',
        modalStep: 1,
        framework: null,
        revealedCount: 0,
        runId: 0,
      };
    default:
      return state;
  }
};

type UseAgentImportResult = {
  phase: AgentPhase;
  modalStep: 1 | 2;
  framework: AgentFramework | null;
  revealedCount: number;
  runId: number;
  openImport: () => void;
  closeModal: () => void;
  selectFramework: (framework: AgentFramework) => void;
  advanceStep: () => void;
  runSimulation: () => void;
  runEval: () => void;
  reset: () => void;
};

export function useAgentImport(totalNodes: number): UseAgentImportResult {
  const [state, dispatch] = useReducer(reducer, totalNodes, initialState);
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
    if (state.phase === 'ready' && state.framework) {
      saveStoredImport(state.framework);
    }
  }, [state.phase, state.framework]);

  const openImport = useCallback(() => dispatch({ type: 'open-modal' }), []);
  const closeModal = useCallback(() => dispatch({ type: 'close-modal' }), []);
  const selectFramework = useCallback(
    (framework: AgentFramework) => dispatch({ type: 'select-framework', framework }),
    []
  );
  const advanceStep = useCallback(() => dispatch({ type: 'advance-step' }), []);

  const runSimulation = useCallback(() => {
    if (!state.framework) return;
    clearTimers();
    dispatch({ type: 'start-discovering', framework: state.framework });
  }, [state.framework, clearTimers]);

  const runEval = useCallback(() => {
    if (state.phase !== 'ready') return;
    clearTimers();
    dispatch({ type: 'rerun-eval' });
  }, [state.phase, clearTimers]);

  const reset = useCallback(() => {
    clearTimers();
    clearStoredImport();
    dispatch({ type: 'reset' });
  }, [clearTimers]);

  return useMemo(
    () => ({
      phase: state.phase,
      modalStep: state.modalStep,
      framework: state.framework,
      revealedCount: state.revealedCount,
      runId: state.runId,
      openImport,
      closeModal,
      selectFramework,
      advanceStep,
      runSimulation,
      runEval,
      reset,
    }),
    [state, openImport, closeModal, selectFramework, advanceStep, runSimulation, runEval, reset]
  );
}
