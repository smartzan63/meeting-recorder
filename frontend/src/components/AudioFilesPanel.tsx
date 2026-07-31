import { useEffect, useState } from 'react'
import { ChevronRight, Trash2, Play } from 'lucide-react'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { cn, formatLocalDateTime } from '@/lib/utils'

export type AudioRecording = {
  id: string
  file: string
  size_mb: number
  modified: string
  processed: boolean
  source_file: string | null
  source_size_mb: number | null
}

export type OrphanedSource = {
  file: string
  size_mb: number
  modified: string
}

type AudioFilesPanelProps = {
  recordings: AudioRecording[]
  orphanedSources: OrphanedSource[]
  totalMb: number
  providers: { key: string; label: string; primary: boolean }[]
  onProcess: (id: string, provider: string, model: string, language: string) => Promise<void>
  onDelete: (filename: string) => Promise<void>
  onLoad?: (id: string) => void
  busy?: boolean
}

export function AudioFilesPanel({
  recordings,
  orphanedSources,
  totalMb,
  providers,
  onProcess,
  onDelete,
  onLoad,
  busy = false,
}: AudioFilesPanelProps) {
  const [open, setOpen] = useState(false)
  const [processId, setProcessId] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<{ file: string; sizeMb: number; extra?: string; keepsTranscript?: boolean } | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)

  const [provider, setProvider] = useState('')
  const [model, setModel] = useState('')
  const [models, setModels] = useState<{ key: string; label: string; default?: boolean }[]>([])
  const [language, setLanguage] = useState('auto')

  useEffect(() => {
    if (!provider) return
    void fetch(`/providers/${provider}/models`)
      .then((response) => (response.ok ? response.json() : []))
      .then((availableModels) => {
        setModels(availableModels)
        const defaultModel = availableModels.find((m: { default?: boolean }) => m.default) ?? availableModels[0]
        setModel(defaultModel?.key ?? '')
      })
  }, [provider])

  const openProcessDialog = (id: string) => {
    setProvider(providers.find((p) => p.primary)?.key ?? providers[0]?.key ?? '')
    setLanguage('auto')
    setProcessId(id)
  }

  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return
    setDeleting(true)
    setDeleteError(null)
    try {
      await onDelete(deleteTarget.file)
      setDeleteTarget(null)
    } catch (err) {
      setDeleteError((err as Error).message)
    } finally {
      setDeleting(false)
    }
  }

  const unprocessedCount = recordings.filter((r) => !r.processed).length

  return (
    <>
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger asChild>
          <button className="flex w-full items-center justify-between gap-1.5 text-xs font-medium uppercase tracking-widest text-zinc-400 hover:text-zinc-200 transition-colors py-1 select-none">
            <span className="flex items-center gap-1.5">
              <ChevronRight className={cn('h-3.5 w-3.5 transition-transform duration-200', open && 'rotate-90')} />
              Audio files ({recordings.length + orphanedSources.length})
              {unprocessedCount > 0 && (
                <span className="normal-case tracking-normal rounded px-1.5 py-0.5 bg-amber-900/50 text-amber-300 border border-amber-700/50 leading-none">
                  {unprocessedCount} unprocessed
                </span>
              )}
            </span>
            <span className="normal-case tracking-normal text-zinc-600">{totalMb.toFixed(0)} MB</span>
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-2 space-y-1">
          {recordings.map((rec) => (
            <div
              key={rec.file}
              className={cn(
                'group flex items-center justify-between rounded-md px-3 py-2.5 bg-zinc-900 border border-zinc-800 gap-3',
                rec.processed && onLoad && 'cursor-pointer hover:bg-zinc-800 hover:border-zinc-700 transition-colors',
              )}
              onClick={() => { if (rec.processed && onLoad) onLoad(rec.id) }}
            >
              <div className="flex flex-col min-w-0 flex-1">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="text-sm text-zinc-200 truncate" title={rec.file}>{rec.id}</span>
                  <span
                    className={cn(
                      'shrink-0 text-xs px-1.5 py-0.5 rounded leading-none border',
                      rec.processed
                        ? 'bg-emerald-900/40 text-emerald-300 border-emerald-700/50'
                        : 'bg-amber-900/50 text-amber-300 border-amber-700/50',
                    )}
                  >
                    {rec.processed ? 'Processed' : 'Unprocessed'}
                  </span>
                </div>
                <span className="text-xs text-zinc-500">
                  {formatLocalDateTime(rec.modified)} · {rec.size_mb.toFixed(1)} MB
                  {rec.source_file ? ` (+ ${rec.source_size_mb?.toFixed(1) ?? '?'} MB source)` : ''}
                </span>
              </div>
              <div className="flex items-center gap-1 shrink-0">
                {!rec.processed && (
                  <Button
                    variant="ghost"
                    size="icon"
                    title="Process with selected backend"
                    disabled={busy || providers.length === 0}
                    className="h-7 w-7 text-zinc-600 hover:text-emerald-400 hover:bg-transparent opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-20"
                    onClick={(e) => { e.stopPropagation(); openProcessDialog(rec.id) }}
                  >
                    <Play className="h-3.5 w-3.5" />
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="icon"
                  title="Delete audio (keeps transcript)"
                  className="h-7 w-7 text-zinc-600 hover:text-red-400 hover:bg-transparent opacity-0 group-hover:opacity-100 transition-opacity"
                  onClick={(e) => {
                    e.stopPropagation()
                    setDeleteTarget({
                      file: rec.file,
                      sizeMb: rec.size_mb + (rec.source_size_mb ?? 0),
                      extra: rec.source_file ? `Also deletes source ${rec.source_file}.` : undefined,
                      keepsTranscript: rec.processed,
                    })
                  }}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>
          ))}

          {orphanedSources.length > 0 && (
            <>
              <p className="pt-2 text-xs font-medium uppercase tracking-widest text-zinc-500">
                Orphaned OBS sources ({orphanedSources.length})
              </p>
              {orphanedSources.map((src) => (
                <div
                  key={src.file}
                  className="group flex items-center justify-between rounded-md px-3 py-2 bg-zinc-900/60 border border-zinc-800/70 gap-3"
                >
                  <div className="flex flex-col min-w-0 flex-1">
                    <span className="text-sm text-zinc-400 truncate" title={src.file}>{src.file}</span>
                    <span className="text-xs text-zinc-600">
                      {formatLocalDateTime(src.modified)} · {src.size_mb.toFixed(1)} MB
                    </span>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    title="Delete source file"
                    className="h-7 w-7 shrink-0 text-zinc-600 hover:text-red-400 hover:bg-transparent opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={() => setDeleteTarget({ file: src.file, sizeMb: src.size_mb })}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              ))}
            </>
          )}

          {recordings.length === 0 && orphanedSources.length === 0 && (
            <p className="text-xs text-zinc-600 py-2">No audio files.</p>
          )}
        </CollapsibleContent>
      </Collapsible>

      {/* Process dialog */}
      <Dialog open={processId !== null} onOpenChange={(v) => { if (!v) setProcessId(null) }}>
        <DialogContent className="bg-zinc-900 border-zinc-700 text-zinc-100">
          <DialogHeader>
            <DialogTitle>Process recording</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-3">
            <p className="text-sm text-zinc-400 break-all">{processId}</p>
            <div className="flex flex-col gap-1.5">
              <span className="text-xs text-zinc-500">Backend</span>
              <Select value={provider} onValueChange={setProvider}>
                <SelectTrigger className="bg-zinc-900 border-zinc-700 text-zinc-200">
                  <SelectValue placeholder="Select backend" />
                </SelectTrigger>
                <SelectContent className="bg-zinc-900 border-zinc-700">
                  {providers.map((p) => (
                    <SelectItem
                      key={p.key}
                      value={p.key}
                      className="text-zinc-200 focus:bg-zinc-800 focus:text-zinc-100"
                    >
                      {p.label}{p.primary ? ' (primary)' : ''}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-1.5">
              <span className="text-xs text-zinc-500">Model</span>
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger className="bg-zinc-900 border-zinc-700 text-zinc-200"><SelectValue /></SelectTrigger>
                <SelectContent className="bg-zinc-900 border-zinc-700">
                  {models.map((m) => (
                    <SelectItem key={m.key} value={m.key} className="text-zinc-200 focus:bg-zinc-800 focus:text-zinc-100">{m.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-1.5">
              <span className="text-xs text-zinc-500">Language</span>
              <Select value={language} onValueChange={setLanguage}>
                <SelectTrigger className="bg-zinc-900 border-zinc-700 text-zinc-200"><SelectValue /></SelectTrigger>
                <SelectContent className="bg-zinc-900 border-zinc-700">
                  <SelectItem value="auto" className="text-zinc-200">Auto-detect</SelectItem>
                  <SelectItem value="en-US" className="text-zinc-200">English</SelectItem>
                  <SelectItem value="ru-RU" className="text-zinc-200">Russian</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-zinc-500">Auto detects one primary language per recording.</p>
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setProcessId(null)}
              className="border-zinc-700 text-zinc-300 hover:bg-zinc-800"
            >
              Cancel
            </Button>
            <Button
              onClick={() => {
                if (!processId || !provider || !model) return
                const id = processId
                setProcessId(null)
                void onProcess(id, provider, model, language)
              }}
              disabled={!provider || !model}
              className="bg-zinc-100 text-zinc-900 hover:bg-zinc-200 border-0"
            >
              Process
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete confirmation */}
      <Dialog open={deleteTarget !== null} onOpenChange={(v) => { if (!v) { setDeleteTarget(null); setDeleteError(null) } }}>
        <DialogContent className="bg-zinc-900 border-zinc-700 text-zinc-100">
          <DialogHeader>
            <DialogTitle>Delete audio file?</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-zinc-400">
            This will permanently delete{' '}
            <span className="text-zinc-200 font-medium break-all">{deleteTarget?.file}</span>{' '}
            ({deleteTarget?.sizeMb.toFixed(1)} MB).{' '}
            {deleteTarget?.extra}{' '}
            {deleteTarget?.keepsTranscript
              ? 'The transcript is kept, but reprocessing this recording will no longer be possible.'
              : null}{' '}
            This action cannot be undone.
          </p>
          {deleteError && <p className="text-sm text-red-400">{deleteError}</p>}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => { setDeleteTarget(null); setDeleteError(null) }}
              disabled={deleting}
              className="border-zinc-700 text-zinc-300 hover:bg-zinc-800"
            >
              Cancel
            </Button>
            <Button
              onClick={() => void handleDeleteConfirm()}
              disabled={deleting}
              className="bg-red-700 hover:bg-red-600 text-white border-0"
            >
              {deleting ? 'Deleting…' : 'Delete'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
