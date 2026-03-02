import { useMemo, useState } from 'react'
import { FileAudio, Loader2, Copy, Check } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { SummaryCard } from './SummaryCard'
import { HistoryPanel } from './HistoryPanel'

type HistoryItem = {
  id: string
  source: string
  model: string
  created_at: string
}

type TranscriptPanelProps = {
  transcript: string
  originalTranscript: string
  transcriptModel: string
  onTranscriptChange: (text: string) => void
  summaryMarkdown: string
  onSummarize: () => Promise<void>
  onSummaryDismiss: () => void
  isSummarizing: boolean
  history: HistoryItem[]
  onLoadHistory: (id: string, source: string) => void
  onDeleteHistory: (id: string) => Promise<void>
  isLoading: boolean
}

const SPEAKER_COLORS = [
  { badge: 'bg-emerald-900/60 text-emerald-300 border-emerald-700', input: 'border-emerald-700/50 focus-visible:ring-emerald-700' },
  { badge: 'bg-blue-900/60 text-blue-300 border-blue-700', input: 'border-blue-700/50 focus-visible:ring-blue-700' },
  { badge: 'bg-amber-900/60 text-amber-300 border-amber-700', input: 'border-amber-700/50 focus-visible:ring-amber-700' },
  { badge: 'bg-violet-900/60 text-violet-300 border-violet-700', input: 'border-violet-700/50 focus-visible:ring-violet-700' },
]

function extractSpeakers(text: string): string[] {
  const matches = text.match(/SPEAKER_\d+/g)
  if (!matches) return []
  return [...new Set(matches)].sort()
}

export function TranscriptPanel({
  transcript,
  originalTranscript,
  transcriptModel,
  onTranscriptChange,
  summaryMarkdown,
  onSummarize,
  onSummaryDismiss,
  isSummarizing,
  history,
  onLoadHistory,
  onDeleteHistory,
  isLoading,
}: TranscriptPanelProps) {
  const [speakerNames, setSpeakerNames] = useState<Record<string, string>>({})
  const [copied, setCopied] = useState(false)

  const speakers = useMemo(() => extractSpeakers(originalTranscript), [originalTranscript])

  function handleSpeakerChange(tag: string, name: string) {
    const updated = { ...speakerNames, [tag]: name }
    setSpeakerNames(updated)

    // Rebuild display transcript from originalTranscript with all substitutions applied
    let rebuilt = originalTranscript
    for (const [speakerTag, speakerName] of Object.entries(updated)) {
      if (speakerName.trim()) {
        rebuilt = rebuilt.replaceAll(speakerTag, speakerName.trim())
      }
    }
    onTranscriptChange(rebuilt)
  }

  function handleCopy() {
    void navigator.clipboard.writeText(transcript)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const hasTranscript = transcript.trim().length > 0

  return (
    <div className="flex flex-col gap-4 h-full">
      {!hasTranscript && !isLoading && (
        <div className="flex flex-col items-center justify-center flex-1 gap-3 py-20 text-zinc-600">
          <FileAudio className="h-12 w-12" />
          <p className="text-sm">Start a recording or upload a file</p>
        </div>
      )}

      {isLoading && !hasTranscript && (
        <div className="flex flex-col items-center justify-center flex-1 gap-3 py-20 text-zinc-500">
          <Loader2 className="h-10 w-10 animate-spin" />
          <p className="text-sm">Processing…</p>
        </div>
      )}

      {hasTranscript && (
        <div className="flex flex-col gap-4">
          {/* Header row */}
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs font-medium uppercase tracking-widest text-zinc-400">
              Transcript
            </span>
            <div className="flex items-center gap-2">
              {transcriptModel && (
                <Badge variant="outline" className="text-xs text-zinc-400 border-zinc-700">
                  {transcriptModel}
                </Badge>
              )}
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-2 text-xs text-zinc-400 hover:text-zinc-100 gap-1"
                onClick={handleCopy}
              >
                {copied ? (
                  <>
                    <Check className="h-3.5 w-3.5" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="h-3.5 w-3.5" />
                    Copy
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* Speaker editor */}
          {speakers.length > 0 && (
            <div className="flex flex-col gap-2">
              <span className="text-xs text-zinc-500">Speaker names</span>
              <div className="flex flex-col gap-1.5">
                {speakers.map((tag, i) => {
                  const colors = SPEAKER_COLORS[i % SPEAKER_COLORS.length]
                  return (
                    <div key={tag} className="flex items-center gap-2">
                      <Badge
                        className={`shrink-0 text-xs font-mono border ${colors.badge}`}
                      >
                        {tag}
                      </Badge>
                      <Input
                        value={speakerNames[tag] ?? ''}
                        onChange={(e) => handleSpeakerChange(tag, e.target.value)}
                        placeholder={`Name for ${tag}`}
                        className={`h-7 text-xs bg-zinc-900 border-zinc-700 text-zinc-200 placeholder:text-zinc-600 ${colors.input}`}
                      />
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Transcript text */}
          <div className="overflow-y-auto max-h-96 rounded-md border border-zinc-800 bg-zinc-900 p-3">
            <pre className="whitespace-pre-wrap font-mono text-sm text-zinc-200 leading-relaxed">
              {transcript}
            </pre>
          </div>

          {/* Summarize button */}
          {!summaryMarkdown && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => void onSummarize()}
              disabled={isSummarizing}
              className="self-start border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 gap-1.5"
            >
              {isSummarizing && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              {isSummarizing ? 'Summarizing…' : 'Summarize'}
            </Button>
          )}
        </div>
      )}

      {/* Summary */}
      {summaryMarkdown && (
        <SummaryCard markdown={summaryMarkdown} onDismiss={onSummaryDismiss} />
      )}

      {/* History */}
      <HistoryPanel items={history} onLoad={onLoadHistory} onDelete={onDeleteHistory} />
    </div>
  )
}
