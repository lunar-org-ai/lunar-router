import { useCallback, useEffect, useMemo, useReducer, useRef } from 'react';

import type { AgentFramework } from '@/features/agents/types';

const CONNECTING_DURATION_MS = 1100;

export type ImportPhase = 'closed' | 'modal' | 'connecting';

type State = {
  phase: ImportPhase;
  step: 1 | 2;
  framework: AgentFramework | null;
  name: string;
};

type Action =
  | { type: 'open' }
  | { type: 'close' }
  | { type: 'select-framework'; framework: AgentFramework }
  | { type: 'set-name'; name: string }
  | { type: 'advance' }
  | { type: 'back' }
  | { type: 'start-connecting' }
  | { type: 'reset' };

const initialState: State = {
  phase: 'closed',
  step: 1,
  framework: null,
  name: '',
};

const reducer = (state: State, action: Action): State => {
  switch (action.type) {
    case 'open':
      return { ...initialState, phase: 'modal' };
    case 'close':
      return initialState;
    case 'select-framework':
      return { ...state, framework: action.framework };
    case 'set-name':
      return { ...state, name: action.name };
    case 'advance':
      if (state.step === 1 && !state.framework) return state;
      return { ...state, step: 2 };
    case 'back':
      return { ...state, step: 1 };
    case 'start-connecting':
      return { ...state, phase: 'connecting' };
    case 'reset':
      return initialState;
    default:
      return state;
  }
};

type ImportPayload = {
  framework: AgentFramework;
  name: string;
};

type UseAgentImportOptions = {
  onComplete: (payload: ImportPayload) => void;
};

type UseAgentImportResult = {
  phase: ImportPhase;
  step: 1 | 2;
  framework: AgentFramework | null;
  name: string;
  open: () => void;
  close: () => void;
  selectFramework: (framework: AgentFramework) => void;
  setName: (name: string) => void;
  advance: () => void;
  back: () => void;
  submit: () => void;
};

export function useAgentImport({ onComplete }: UseAgentImportOptions): UseAgentImportResult {
  const [state, dispatch] = useReducer(reducer, initialState);
  const completionRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onCompleteRef = useRef(onComplete);

  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  useEffect(() => {
    return () => {
      if (completionRef.current) clearTimeout(completionRef.current);
    };
  }, []);

  useEffect(() => {
    if (state.phase !== 'connecting') return;
    if (!state.framework) return;
    const framework = state.framework;
    const name = state.name;
    completionRef.current = setTimeout(() => {
      onCompleteRef.current({ framework, name });
      dispatch({ type: 'reset' });
    }, CONNECTING_DURATION_MS);
    return () => {
      if (completionRef.current) clearTimeout(completionRef.current);
    };
  }, [state.phase, state.framework, state.name]);

  const open = useCallback(() => dispatch({ type: 'open' }), []);
  const close = useCallback(() => dispatch({ type: 'close' }), []);
  const selectFramework = useCallback(
    (framework: AgentFramework) => dispatch({ type: 'select-framework', framework }),
    []
  );
  const setName = useCallback((name: string) => dispatch({ type: 'set-name', name }), []);
  const advance = useCallback(() => dispatch({ type: 'advance' }), []);
  const back = useCallback(() => dispatch({ type: 'back' }), []);
  const submit = useCallback(() => dispatch({ type: 'start-connecting' }), []);

  return useMemo(
    () => ({
      phase: state.phase,
      step: state.step,
      framework: state.framework,
      name: state.name,
      open,
      close,
      selectFramework,
      setName,
      advance,
      back,
      submit,
    }),
    [state, open, close, selectFramework, setName, advance, back, submit]
  );
}
