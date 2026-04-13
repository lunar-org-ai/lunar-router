import { useState, useRef, useEffect, useCallback } from 'react';
import type {
  CreateMode,
  GeneratePhase,
  TopicPhase,
  DatasetSample,
  Dataset,
  CreateDatasetRequest,
  CreateFromInstructionRequest,
  CreateFromInstructionResponse,
  GenerateDatasetRequest,
  GenerateDatasetResponse,
  AnalyzeTracesResponse,
  ImportTracesResponse,
} from '../types';
import { GENERATE_COUNT } from '../constants';

interface UseCreateDatasetModalOptions {
  open: boolean;
  onClose: () => void;
  onCreate: (request: CreateDatasetRequest) => Promise<Dataset | null | void>;
  onImport: (
    file: File,
    name: string,
    autoCollectInstruction?: string
  ) => Promise<Dataset | null | void>;
  onCreateFromTopic?: (
    request: CreateFromInstructionRequest
  ) => Promise<CreateFromInstructionResponse | null | undefined>;
  onGenerate?: (
    request: GenerateDatasetRequest
  ) => Promise<GenerateDatasetResponse | null | undefined>;
  onPollGenerate?: (datasetId: string) => Promise<number>;
  onGenerateBackground?: (datasetId: string, name: string, requested: number) => void;
  onAnalyzeTraces?: (data: any[]) => Promise<AnalyzeTracesResponse>;
  onImportTraces?: (
    name: string,
    data: any[],
    mapping: any,
    description?: string
  ) => Promise<ImportTracesResponse>;
}

export function useCreateDatasetModal({
  open,
  onClose,
  onCreate,
  onImport,
  onCreateFromTopic,
  onGenerate,
  onPollGenerate,
  onGenerateBackground,
  onAnalyzeTraces,
  onImportTraces,
}: UseCreateDatasetModalOptions) {
  const [mode, setMode] = useState<CreateMode>('manual');
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [successName, setSuccessName] = useState<string | null>(null);
  const [keepBuilding, setKeepBuilding] = useState(false);
  const [autoCollectInstruction, setAutoCollectInstruction] = useState('');

  const [samples, setSamples] = useState<Partial<DatasetSample>[]>([
    { input: '', expected_output: '' },
  ]);

  const [file, setFile] = useState<File | null>(null);

  const [topic, setTopic] = useState('');
  const [topicPhase, setTopicPhase] = useState<TopicPhase>('idle');
  const [agentLog, setAgentLog] = useState<string[]>([]);
  const [topicResult, setTopicResult] = useState<{ matched: number; scanned: number } | null>(null);

  const [generateInstruction, setGenerateInstruction] = useState('');
  const [generateCount, setGenerateCount] = useState(GENERATE_COUNT.default);
  const [generatePhase, setGeneratePhase] = useState<GeneratePhase>('idle');
  const [generateLog, setGenerateLog] = useState<string[]>([]);
  const [generateResult, setGenerateResult] = useState<{
    count: number;
    requested: number;
  } | null>(null);

  // Smart import state
  const [smartImportFile, setSmartImportFile] = useState<File | null>(null);
  const [smartImportRecords, setSmartImportRecords] = useState<any[]>([]);
  const [smartImportPhase, setSmartImportPhase] = useState<
    'upload' | 'analyzing' | 'preview' | 'importing'
  >('upload');
  const [smartImportAnalysis, setSmartImportAnalysis] = useState<AnalyzeTracesResponse | null>(
    null
  );

  const inputRef = useRef<HTMLInputElement | null>(null);
  const phaseTimers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const generatingRef = useRef<{ datasetId: string; name: string; requested: number } | null>(null);

  useEffect(() => {
    if (open) {
      inputRef.current?.focus();
      setMode('manual');
      setName('');
      setDescription('');
      setError(null);
      setCreating(false);
      setSuccessName(null);
      setKeepBuilding(false);
      setAutoCollectInstruction('');
      setSamples([{ input: '', expected_output: '' }]);
      setFile(null);
      setTopic('');
      setTopicPhase('idle');
      setAgentLog([]);
      setTopicResult(null);
      setGenerateInstruction('');
      setGenerateCount(GENERATE_COUNT.default);
      setGeneratePhase('idle');
      setGenerateLog([]);
      setGenerateResult(null);
      generatingRef.current = null;
      setSmartImportFile(null);
      setSmartImportRecords([]);
      setSmartImportPhase('upload');
      setSmartImportAnalysis(null);
    }
    return () => {
      phaseTimers.current.forEach(clearTimeout);
      phaseTimers.current = [];
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [open]);

  const addLog = useCallback((msg: string) => setAgentLog((prev) => [...prev, msg]), []);
  const addGenerateLog = useCallback((msg: string) => setGenerateLog((prev) => [...prev, msg]), []);

  const addSample = useCallback(() => {
    setSamples((prev) => [...prev, { input: '', expected_output: '' }]);
  }, []);

  const removeSample = useCallback((index: number) => {
    setSamples((prev) => (prev.length > 1 ? prev.filter((_, i) => i !== index) : prev));
  }, []);

  const updateSample = useCallback(
    (index: number, field: 'input' | 'expected_output', value: string) => {
      setSamples((prev) => {
        const updated = [...prev];
        updated[index] = { ...updated[index], [field]: value };
        return updated;
      });
    },
    []
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFile = e.target.files?.[0];
      if (!selectedFile) return;
      const ext = selectedFile.name.split('.').pop()?.toLowerCase();
      if (ext !== 'csv' && ext !== 'json') {
        setError('Please upload a CSV or JSON file');
        return;
      }
      setFile(selectedFile);
      setError(null);
      if (!name) setName(selectedFile.name.replace(/\.(csv|json)$/i, ''));
    },
    [name]
  );

  const isTopicProcessing = ['scanning', 'analyzing', 'matching', 'building'].includes(topicPhase);
  const isGenerateProcessing = ['preparing', 'generating', 'reviewing', 'building'].includes(
    generatePhase
  );
  const isDisabled = creating || isTopicProcessing || isGenerateProcessing;

  const handleClose = useCallback(() => {
    if (isTopicProcessing) return;
    if (isGenerateProcessing && generatingRef.current && onGenerateBackground) {
      const { datasetId, name: n, requested } = generatingRef.current;
      onGenerateBackground(datasetId, n, requested);
    }
    onClose();
  }, [isTopicProcessing, isGenerateProcessing, onGenerateBackground, onClose]);

  const showSuccess = useCallback(
    (finalName: string, delay = 800) => {
      setSuccessName(finalName);
      setTimeout(() => {
        setCreating(false);
        setSuccessName(null);
        onClose();
      }, delay);
    },
    [onClose]
  );

  const submitManual = useCallback(async () => {
    const validSamples = samples.filter((s) => s.input?.trim());
    if (validSamples.length === 0) {
      setError('Please add at least one sample with input');
      setCreating(false);
      return;
    }
    const preparedSamples = validSamples.map((s) => {
      const out = s.expected_output?.trim();
      const sampleData = {
        input: s.input?.trim() || '',
        expected_output: out || undefined,
        output: out || undefined,
      };
      return {
        ...sampleData,
        raw: JSON.stringify({
          input: sampleData.input,
          output: sampleData.output,
          expected_output: sampleData.expected_output,
        }),
      };
    }) as Omit<DatasetSample, 'id' | 'created_at'>[];

    const result = await onCreate({
      name: name.trim(),
      description: description.trim() || undefined,
      source: 'manual',
      samples: preparedSamples,
      auto_collect_instruction: keepBuilding ? autoCollectInstruction.trim() : undefined,
    });
    showSuccess((result as Dataset | null)?.name || name.trim());
  }, [samples, name, description, keepBuilding, autoCollectInstruction, onCreate, showSuccess]);

  const submitImport = useCallback(async () => {
    if (!file) {
      setError('Please select a file to import');
      setCreating(false);
      return;
    }
    const result = await onImport(
      file,
      name.trim(),
      keepBuilding ? autoCollectInstruction.trim() : undefined
    );
    showSuccess((result as Dataset | null)?.name || name.trim());
  }, [file, name, keepBuilding, autoCollectInstruction, onImport, showSuccess]);

  const handleSmartImportFileSelect = useCallback(
    async (selectedFile: File, records: any[]) => {
      setSmartImportFile(selectedFile);
      setSmartImportRecords(records);
      setError(null);
      if (!name) setName(selectedFile.name.replace(/\.json$/i, ''));

      if (onAnalyzeTraces) {
        setSmartImportPhase('analyzing');
        try {
          const analysis = await onAnalyzeTraces(records);
          setSmartImportAnalysis(analysis);
          setSmartImportPhase('preview');
        } catch (err) {
          const msg = err instanceof Error ? err.message : 'Schema analysis failed';
          setError(msg);
          setSmartImportPhase('upload');
          setSmartImportFile(null);
          setSmartImportRecords([]);
        }
      } else {
        setSmartImportPhase('preview');
      }
    },
    [name, onAnalyzeTraces]
  );

  const submitSmartImport = useCallback(async () => {
    if (!smartImportRecords.length || !smartImportAnalysis) {
      setError('Please upload a JSON file first');
      setCreating(false);
      return;
    }
    if (!onImportTraces) {
      setError('Smart import is not available');
      setCreating(false);
      return;
    }

    setSmartImportPhase('importing');
    try {
      const result = await onImportTraces(
        name.trim(),
        smartImportRecords,
        smartImportAnalysis.mapping,
        description.trim() || undefined
      );
      showSuccess(result.name || name.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Import failed');
      setSmartImportPhase('preview');
      setCreating(false);
    }
  }, [smartImportRecords, smartImportAnalysis, name, description, onImportTraces, showSuccess]);

  const submitTopic = useCallback(async () => {
    if (!topic.trim()) {
      setError('Please describe the topic');
      setCreating(false);
      return;
    }
    if (!onCreateFromTopic) {
      setCreating(false);
      return;
    }

    setTopicPhase('scanning');
    setAgentLog([]);
    addLog(`> Searching for topic: "${topic}"`);
    addLog('> Scanning your production traces...');

    const t1 = setTimeout(() => {
      setTopicPhase('analyzing');
      addLog('> Running embedding pre-filter (Titan v2)...');
    }, 1500);
    const t2 = setTimeout(() => {
      setTopicPhase('matching');
      addLog('> AI classification in progress (Sonnet 4.5)...');
    }, 3500);
    const t3 = setTimeout(() => {
      setTopicPhase('building');
      addLog('> Filtering and building dataset...');
    }, 5500);
    phaseTimers.current = [t1, t2, t3];

    try {
      const result = await onCreateFromTopic({
        name: name.trim() || topic.trim(),
        instruction: topic.trim(),
        description: description.trim() || undefined,
        max_samples: keepBuilding ? 200 : undefined,
      });

      phaseTimers.current.forEach(clearTimeout);
      phaseTimers.current = [];

      if (result) {
        setTopicResult({ matched: result.samples_count, scanned: result.traces_scanned });
        setTopicPhase('done');
        addLog(
          `> Done. ${result.traces_matched ?? result.samples_count} traces matched out of ${result.traces_scanned} scanned.`
        );
        addLog(`> Dataset "${result.name}" created with ${result.samples_count} samples.`);
        setSuccessName(result.name);
        setTimeout(() => {
          setCreating(false);
          setSuccessName(null);
          setTopicResult(null);
          setTopicPhase('idle');
          onClose();
        }, 2000);
      } else {
        setTopicPhase('no-match');
        addLog('> No matching traces found for this topic.');
        setCreating(false);
      }
    } catch (err) {
      phaseTimers.current.forEach(clearTimeout);
      phaseTimers.current = [];
      const msg = err instanceof Error ? err.message : 'Unknown error';
      if (
        msg.includes('No traces matching') ||
        msg.includes('No traces found') ||
        msg.includes('no match')
      ) {
        setTopicPhase('no-match');
        const scannedMatch = msg.match(/Scanned (\d+)/i);
        addLog(
          scannedMatch
            ? `> Scanned ${scannedMatch[1]} traces, no matches for "${topic}".`
            : `> ${msg}`
        );
      } else {
        setTopicPhase('error');
        addLog(`> Error: ${msg}`);
        setError(msg);
      }
      setCreating(false);
    }
  }, [topic, name, description, keepBuilding, onCreateFromTopic, onClose, addLog]);

  const submitGenerate = useCallback(async () => {
    if (!generateInstruction.trim()) {
      setError('Please describe what data to generate');
      setCreating(false);
      return;
    }
    if (!onGenerate) {
      setCreating(false);
      return;
    }

    setGeneratePhase('preparing');
    setGenerateLog([]);
    addGenerateLog(`> Instruction: "${generateInstruction}"`);
    addGenerateLog('> Creating dataset and starting generation...');

    try {
      const result = await onGenerate({
        name: name.trim() || generateInstruction.trim().slice(0, 50),
        instruction: generateInstruction.trim(),
        description: description.trim() || undefined,
        count: generateCount,
        auto_collect_instruction: keepBuilding ? generateInstruction.trim() : undefined,
      });

      if (!result) {
        setGeneratePhase('error');
        addGenerateLog('> Failed to create dataset.');
        setError('Failed to create dataset. Try a different instruction.');
        setCreating(false);
        return;
      }

      generatingRef.current = {
        datasetId: result.dataset_id,
        name: result.name,
        requested: generateCount,
      };
      setGeneratePhase('generating');
      addGenerateLog(
        `> Dataset "${result.name}" created. Generating ${generateCount} samples in background...`
      );

      if (onPollGenerate) {
        let lastCount = 0;
        let stablePolls = 0;

        const finishGeneration = (count: number) => {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
          phaseTimers.current.forEach(clearTimeout);
          phaseTimers.current = [];
          setGenerateResult({ count, requested: generateCount });
          setGeneratePhase('done');
          addGenerateLog(`> Done. Generated ${count} of ${generateCount} requested samples.`);
          setSuccessName(result.name);
          setTimeout(() => {
            setCreating(false);
            setSuccessName(null);
            setGenerateResult(null);
            setGeneratePhase('idle');
            onClose();
          }, 2000);
        };

        pollIntervalRef.current = setInterval(async () => {
          try {
            const currentCount = await onPollGenerate(result.dataset_id);
            if (currentCount > lastCount) {
              stablePolls = 0;
              const progress = Math.round((currentCount / generateCount) * 100);
              if (lastCount === 0) {
                setGeneratePhase('reviewing');
                addGenerateLog(
                  `> Samples arriving... ${currentCount}/${generateCount} (${progress}%)`
                );
              } else {
                setGeneratePhase('building');
                addGenerateLog(
                  `> Progress: ${currentCount}/${generateCount} samples (${progress}%)`
                );
              }
              lastCount = currentCount;
              if (currentCount >= generateCount) finishGeneration(currentCount);
            } else if (currentCount > 0) {
              stablePolls++;
              if (stablePolls >= 2) finishGeneration(currentCount);
            }
          } catch {
            /* retry next interval */
          }
        }, 3000);

        const pollTimeout = setTimeout(() => {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
            addGenerateLog(
              '> Generation is taking longer than expected. Check your dataset later.'
            );
            setGeneratePhase('done');
            setGenerateResult({ count: lastCount, requested: generateCount });
            setSuccessName(result.name);
            setTimeout(() => {
              setCreating(false);
              setSuccessName(null);
              setGenerateResult(null);
              setGeneratePhase('idle');
              onClose();
            }, 3000);
          }
        }, 300000);
        phaseTimers.current = [pollTimeout as unknown as ReturnType<typeof setTimeout>];
      } else {
        setGenerateResult({ count: result.samples_count, requested: result.samples_requested });
        setGeneratePhase('done');
        addGenerateLog('> Dataset created. Samples are being generated in the background.');
        setSuccessName(result.name);
        setTimeout(() => {
          setCreating(false);
          setSuccessName(null);
          setGenerateResult(null);
          setGeneratePhase('idle');
          onClose();
        }, 2000);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      setGeneratePhase('error');
      addGenerateLog(`> Error: ${msg}`);
      setError(msg);
      setCreating(false);
    }
  }, [
    generateInstruction,
    name,
    description,
    generateCount,
    keepBuilding,
    onGenerate,
    onPollGenerate,
    onClose,
    addGenerateLog,
  ]);

  const submit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setError(null);

      if (!name.trim() && mode !== 'topic' && mode !== 'traces') {
        setError('Please enter a dataset name');
        return;
      }

      if (creating) return;
      setCreating(true);

      try {
        switch (mode) {
          case 'topic':
            await submitTopic();
            break;
          case 'generate':
            await submitGenerate();
            break;
          case 'import':
            await submitImport();
            break;
          case 'smart-import':
            await submitSmartImport();
            break;
          case 'manual':
            await submitManual();
            break;
          case 'traces':
            break;
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
        setCreating(false);
      }
    },
    [
      mode,
      name,
      creating,
      submitTopic,
      submitGenerate,
      submitImport,
      submitSmartImport,
      submitManual,
    ]
  );

  const handleTopicRetry = useCallback(() => {
    setTopicPhase('idle');
    setAgentLog([]);
    setError(null);
    setCreating(false);
  }, []);

  const handleGenerateRetry = useCallback(() => {
    setGeneratePhase('idle');
    setGenerateLog([]);
    setError(null);
    setCreating(false);
  }, []);

  return {
    mode,
    setMode,
    name,
    setName,
    description,
    setDescription,
    error,
    creating,
    successName,
    isDisabled,
    keepBuilding,
    setKeepBuilding,
    autoCollectInstruction,
    setAutoCollectInstruction,
    inputRef,
    submit,
    handleClose,
    showSuccess,

    samples,
    addSample,
    removeSample,
    updateSample,

    file,
    handleFileChange,

    topic,
    setTopic,
    topicPhase,
    isTopicProcessing,
    agentLog,
    topicResult,
    handleTopicRetry,

    generateInstruction,
    setGenerateInstruction,
    generateCount,
    setGenerateCount,
    generatePhase,
    isGenerateProcessing,
    generateLog,
    generateResult,
    handleGenerateRetry,
    generatingRef,

    smartImportFile,
    smartImportPhase,
    smartImportAnalysis,
    smartImportRecordCount: smartImportRecords.length,
    handleSmartImportFileSelect,
  };
}
