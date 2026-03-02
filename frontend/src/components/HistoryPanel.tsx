import { useState } from 'react'
import { ChevronRight, Trash2 } from 'lucide-react'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import { cn } from '@/lib/utils'

type HistoryItem = {
  id: string
  source: string
  model: string
  created_at: string
}

type HistoryPanelProps = {
  items: HistoryItem[]
  onLoad: (id: string, source: string) => void
  onDelete: (id: string) => Promise<void>
}

function formatDate(isoString: string): string {
  try {
    return new Date(isoString).toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return isoString
  }
}

export function HistoryPanel({ items, onLoad, onDelete }: HistoryPanelProps) {
  const [open, setOpen] = useState(false)
  const [confirmId, setConfirmId] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)

  if (items.length === 0) return null

  const confirmItem = items.find((i) => i.id === confirmId)

  const handleDeleteConfirm = async () => {
    if (!confirmId) return
    setDeleting(true)
    try {
      await onDelete(confirmId)
    } finally {
      setDeleting(false)
      setConfirmId(null)
    }
  }

  return (
    <>
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger asChild>
          <button className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-widest text-zinc-400 hover:text-zinc-200 transition-colors py-1 select-none">
            <ChevronRight
              className={cn('h-3.5 w-3.5 transition-transform duration-200', open && 'rotate-90')}
            />
            History ({items.length})
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-2 space-y-1">
          {items.map((item) => (
            <div
              key={item.id}
              className="group flex items-center justify-between rounded-md px-3 py-2.5 bg-zinc-900 border border-zinc-800 gap-3 cursor-pointer hover:bg-zinc-800 hover:border-zinc-700 transition-colors"
              onClick={() => onLoad(item.id, item.source)}
            >
              <div className="flex flex-col min-w-0 flex-1">
                <span className="text-sm text-zinc-200 truncate">{item.source}</span>
                <span className="text-xs text-zinc-500">
                  {formatDate(item.created_at)}{item.model ? ` · ${item.model}` : ''}
                </span>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7 shrink-0 text-zinc-600 hover:text-red-400 hover:bg-transparent opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={(e) => {
                  e.stopPropagation()
                  setConfirmId(item.id)
                }}
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </div>
          ))}
        </CollapsibleContent>
      </Collapsible>

      <Dialog open={confirmId !== null} onOpenChange={(v) => { if (!v) setConfirmId(null) }}>
        <DialogContent className="bg-zinc-900 border-zinc-700 text-zinc-100">
          <DialogHeader>
            <DialogTitle>Remove recording?</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-zinc-400">
            This will permanently delete{' '}
            <span className="text-zinc-200 font-medium">{confirmItem?.source}</span>.
            This action cannot be undone.
          </p>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setConfirmId(null)}
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
              {deleting ? 'Removing…' : 'Remove'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
