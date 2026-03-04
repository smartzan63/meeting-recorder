import { useRef, useState, useCallback, useEffect } from 'react'
import { Loader2 } from 'lucide-react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import { cn } from '@/lib/utils'

type AppStatus = 'idle' | 'recording' | 'stopped' | 'transcribing' | 'done' | 'error'

type Model = {
  key: string
  label: string
  default: boolean
}

type RecordPanelProps = {
  status: AppStatus
  statusMessage: string
  timerSeconds: number
  models: Model[]
  selectedModel: string
  onModelChange: (key: string) => void
  savedWavPath: string | null
  defaultRecordingName: string
  showSaveDialog: boolean
  showProcessPrompt: boolean
  onStartRecording: () => Promise<void>
  onStopRecording: () => Promise<void>
  onSaveRecording: (name: string) => Promise<void>
  onProcessNow: () => Promise<void>
  onSkipProcess: () => void
  onFileUpload: (file: File) => Promise<void>
  onDismissError: () => Promise<void>
  onCancelSave: () => void
}

function formatTimer(seconds: number): string {
  const m = Math.floor(seconds / 60).toString().padStart(2, '0')
  const s = (seconds % 60).toString().padStart(2, '0')
  return `${m}:${s}`
}

export function RecordPanel({
  status,
  statusMessage,
  timerSeconds,
  models,
  selectedModel,
  onModelChange,
  savedWavPath,
  defaultRecordingName,
  showSaveDialog,
  showProcessPrompt,
  onStartRecording,
  onStopRecording,
  onSaveRecording,
  onProcessNow,
  onSkipProcess,
  onFileUpload,
  onDismissError,
  onCancelSave,
}: RecordPanelProps) {
  const [saveName, setSaveName] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [pendingStop, setPendingStop] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Sync saveName when save dialog opens
  useEffect(() => {
    if (showSaveDialog) {
      setSaveName(defaultRecordingName)
    }
  }, [showSaveDialog, defaultRecordingName])

  const isBusy = status === 'recording' || status === 'stopped' || status === 'transcribing'
  const isRecording = status === 'recording'
  const isTranscribing = status === 'transcribing'

  const multipleModels = models.length > 1

  const handleRecordClick = useCallback(async () => {
    if (isRecording) {
      setPendingStop(true)
      try {
        await onStopRecording()
      } finally {
        setPendingStop(false)
      }
    } else {
      await onStartRecording()
    }
  }, [isRecording, onStartRecording, onStopRecording])

  const handleSaveSubmit = useCallback(async () => {
    const name = saveName.trim() || defaultRecordingName
    await onSaveRecording(name)
  }, [saveName, defaultRecordingName, onSaveRecording])

  const handleSaveKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        void handleSaveSubmit()
      }
    },
    [handleSaveSubmit],
  )

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) setSelectedFile(file)
    },
    [],
  )

  const handleProcessFile = useCallback(async () => {
    if (!selectedFile) return
    await onFileUpload(selectedFile)
    setSelectedFile(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [selectedFile, onFileUpload])


  return (
    <div className="flex flex-col items-center gap-6 py-6 px-4 w-full">
      {/* Model selector */}
      {multipleModels && (
        <div className="w-full max-w-xs">
          <Select value={selectedModel} onValueChange={onModelChange} disabled={isBusy}>
            <SelectTrigger className="bg-zinc-900 border-zinc-700 text-zinc-200">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent className="bg-zinc-900 border-zinc-700">
              {models.map((m) => (
                <SelectItem key={m.key} value={m.key} className="text-zinc-200 focus:bg-zinc-800 focus:text-zinc-100">
                  {m.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}


      {/* Record button */}
      <div className="relative flex items-center justify-center">
        {pendingStop ? (
          <button
            disabled
            className="w-44 h-44 rounded-full bg-zinc-700 text-zinc-400 cursor-not-allowed opacity-70 font-semibold text-base"
          >
            Stopping…
          </button>
        ) : isRecording ? (
          <button
            onClick={() => void handleRecordClick()}
            className="w-44 h-44 rounded-full bg-red-600 hover:bg-red-500 text-white font-semibold text-base transition-all ring-4 ring-red-600/30 animate-pulse"
          >
            Stop<br />Recording
          </button>
        ) : isBusy ? (
          <button
            disabled
            className="w-44 h-44 rounded-full bg-muted text-muted-foreground cursor-not-allowed opacity-50 font-semibold text-base"
          >
            {isTranscribing ? 'Transcribing…' : 'Start\nRecording'}
          </button>
        ) : (
          <button
            onClick={() => void handleRecordClick()}
            className="w-44 h-44 rounded-full bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-base transition-all"
          >
            Start<br />Recording
          </button>
        )}
      </div>

      {/* Timer */}
      {isRecording && !pendingStop && (
        <span className="font-mono text-2xl text-zinc-200 tabular-nums">
          {formatTimer(timerSeconds)}
        </span>
      )}

      {/* Status line */}
      {statusMessage && (
        <div className="flex items-center gap-2 text-sm text-zinc-400">
          {isTranscribing && <Loader2 className="h-4 w-4 animate-spin shrink-0" />}
          <span className={status === 'error' ? 'text-red-400' : undefined}>
            {statusMessage}
          </span>
          {status === 'error' && (
            <button
              onClick={() => void onDismissError()}
              className="ml-1 text-xs text-zinc-500 underline hover:text-zinc-300"
            >
              Dismiss
            </button>
          )}
        </div>
      )}

      {/* Process prompt */}
      {showProcessPrompt && savedWavPath && (
        <Card className="w-full max-w-xs border-zinc-800 bg-zinc-900">
          <CardContent className="pt-4 pb-4 px-4 flex flex-col gap-3">
            <p className="text-sm text-zinc-300">
              Saved to <span className="font-mono text-zinc-400 text-xs">{savedWavPath}</span>.
              Process now?
            </p>
            <div className="flex gap-2">
              <Button
                size="sm"
                onClick={() => void onProcessNow()}
                className="flex-1 bg-emerald-700 hover:bg-emerald-600 text-white border-0"
              >
                Process
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={onSkipProcess}
                className="flex-1 border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100"
              >
                Skip
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Divider */}
      <div className="flex items-center w-full max-w-xs gap-3">
        <div className="flex-1 h-px bg-zinc-800" />
        <span className="text-xs text-zinc-600">or</span>
        <div className="flex-1 h-px bg-zinc-800" />
      </div>

      {/* File upload */}
      <div className="flex flex-col items-center gap-3 w-full max-w-xs">
        <label className="cursor-pointer">
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            className="sr-only"
            onChange={handleFileChange}
            disabled={isBusy}
          />
          <span
            className={cn(
              'inline-flex h-9 items-center rounded-md border px-3 text-sm font-medium transition-colors',
              isBusy
                ? 'border-zinc-800 text-zinc-600 cursor-not-allowed'
                : 'border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 cursor-pointer',
            )}
          >
            {selectedFile ? selectedFile.name : 'Choose File'}
          </span>
        </label>
        <Button
          variant="outline"
          size="sm"
          onClick={() => void handleProcessFile()}
          disabled={!selectedFile || isBusy}
          className="w-full border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 disabled:text-zinc-600 disabled:border-zinc-800"
        >
          Process File
        </Button>
      </div>

      {/* Save dialog */}
      <Dialog open={showSaveDialog} onOpenChange={(open) => { if (!open) onCancelSave() }}>
        <DialogContent className="bg-zinc-900 border-zinc-700 text-zinc-100">
          <DialogHeader>
            <DialogTitle>Save Recording</DialogTitle>
          </DialogHeader>
          <div className="py-2">
            <Input
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              onKeyDown={handleSaveKeyDown}
              placeholder={defaultRecordingName}
              className="bg-zinc-800 border-zinc-700 text-zinc-100 placeholder:text-zinc-500"
              autoFocus
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={onCancelSave}
              className="border-zinc-700 text-zinc-300 hover:bg-zinc-800"
            >
              Cancel
            </Button>
            <Button
              onClick={() => void handleSaveSubmit()}
              className="bg-emerald-700 hover:bg-emerald-600 text-white border-0"
            >
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
