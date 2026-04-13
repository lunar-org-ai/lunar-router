import { useRef } from 'react';
import { FileText, Upload, CheckCircle2, Loader2 } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { DatasetNameFields } from './DatasetNameFields';
import type { AnalyzeTracesResponse } from '../../types';

type SmartImportPhase = 'upload' | 'analyzing' | 'preview' | 'importing';

interface SmartImportModeProps {
  name: string;
  description: string;
  disabled: boolean;
  file: File | null;
  phase: SmartImportPhase;
  analysis: AnalyzeTracesResponse | null;
  recordCount: number;
  onNameChange: (v: string) => void;
  onDescriptionChange: (v: string) => void;
  onFileSelect: (file: File, records: any[]) => void;
  inputRef: React.RefObject<HTMLInputElement | null>;
}

export function SmartImportMode({
  name,
  description,
  disabled,
  file,
  phase,
  analysis,
  recordCount,
  onNameChange,
  onDescriptionChange,
  onFileSelect,
  inputRef,
}: SmartImportModeProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const parsed = JSON.parse(ev.target?.result as string);
        if (!Array.isArray(parsed)) {
          throw new Error('File must contain a JSON array');
        }
        onFileSelect(selectedFile, parsed);
      } catch {
        // Silently handle - error will be shown by parent
      }
    };
    reader.readAsText(selectedFile);
  };

  if (phase === 'upload' || (!file && phase !== 'analyzing')) {
    return (
      <div className="space-y-4">
        <DatasetNameFields
          name={name}
          description={description}
          disabled={disabled}
          onNameChange={onNameChange}
          onDescriptionChange={onDescriptionChange}
          inputRef={inputRef}
          nameRequired
        />
        <Card
          className="cursor-pointer border-dashed border-2 hover:border-primary hover:bg-accent/50 transition-colors"
          onClick={() => fileInputRef.current?.click()}
        >
          <CardContent className="py-8 text-center">
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleFileChange}
              className="hidden"
              disabled={disabled}
            />
            <Upload className="size-8 mx-auto text-muted-foreground" />
            <p className="text-sm text-muted-foreground mt-2">Upload any JSON file with traces</p>
            <p className="text-xs text-muted-foreground mt-1">
              We auto-detect the schema and map fields for you
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (phase === 'analyzing') {
    return (
      <div className="space-y-4">
        <DatasetNameFields
          name={name}
          description={description}
          disabled
          onNameChange={onNameChange}
          onDescriptionChange={onDescriptionChange}
          inputRef={inputRef}
          nameRequired
        />
        <Card>
          <CardContent className="py-8 text-center">
            <Loader2 className="size-8 mx-auto text-primary animate-spin" />
            <p className="text-sm font-medium mt-3">Analyzing schema...</p>
            <p className="text-xs text-muted-foreground mt-1">
              AI is detecting the input/output fields from {recordCount} records
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Preview phase
  if (phase === 'preview' && analysis) {
    return (
      <div className="space-y-4">
        <DatasetNameFields
          name={name}
          description={description}
          disabled={disabled}
          onNameChange={onNameChange}
          onDescriptionChange={onDescriptionChange}
          inputRef={inputRef}
          nameRequired
        />

        <div className="flex items-center gap-2 text-sm">
          <FileText className="size-4 text-primary" />
          <span className="font-medium">{file?.name}</span>
          <span className="text-muted-foreground">{recordCount} records</span>
          <span className="text-xs bg-muted px-1.5 py-0.5 rounded">{analysis.source_format}</span>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-1.5">
            <CheckCircle2 className="size-3.5 text-green-500" />
            <span className="text-xs font-medium">Detected mapping</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-muted rounded px-2.5 py-1.5">
              <span className="text-muted-foreground">Input: </span>
              <code>{analysis.mapping.input.path}</code>
              <span className="text-muted-foreground"> ({analysis.mapping.input.transform})</span>
            </div>
            <div className="bg-muted rounded px-2.5 py-1.5">
              <span className="text-muted-foreground">Output: </span>
              <code>{analysis.mapping.output.path}</code>
              <span className="text-muted-foreground"> ({analysis.mapping.output.transform})</span>
            </div>
          </div>
          {Object.keys(analysis.mapping.metadata).length > 0 && (
            <div className="bg-muted rounded px-2.5 py-1.5 text-xs">
              <span className="text-muted-foreground">Metadata: </span>
              {Object.keys(analysis.mapping.metadata).join(', ')}
            </div>
          )}
        </div>

        <div className="space-y-2">
          <span className="text-xs font-medium">Preview ({analysis.preview.length} samples)</span>
          <ScrollArea className="max-h-[200px]">
            <div className="space-y-2">
              {analysis.preview.map((sample, i) => (
                <div key={i} className="border rounded-md p-2.5 text-xs space-y-1">
                  <div>
                    <span className="font-medium text-muted-foreground">Input: </span>
                    <span className="break-all">
                      {sample.input.slice(0, 150)}
                      {sample.input.length > 150 ? '...' : ''}
                    </span>
                  </div>
                  <div>
                    <span className="font-medium text-muted-foreground">Output: </span>
                    <span className="break-all">
                      {sample.expected_output.slice(0, 150)}
                      {sample.expected_output.length > 150 ? '...' : ''}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </div>
      </div>
    );
  }

  // Importing phase
  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="py-8 text-center">
          <Loader2 className="size-8 mx-auto text-primary animate-spin" />
          <p className="text-sm font-medium mt-3">Importing {recordCount} records...</p>
        </CardContent>
      </Card>
    </div>
  );
}
