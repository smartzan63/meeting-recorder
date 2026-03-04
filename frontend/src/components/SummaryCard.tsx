import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { X } from 'lucide-react'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

type SummaryCardProps = {
  markdown: string
  onDismiss: () => void
}

export function SummaryCard({ markdown, onDismiss }: SummaryCardProps) {
  const [showRaw, setShowRaw] = useState(false)

  return (
    <Card className="border-zinc-800 bg-zinc-900">
      <CardHeader className="pb-3 pt-4 px-4">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium uppercase tracking-widest text-zinc-400">
            Summary
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs text-zinc-400 hover:text-zinc-100"
              onClick={() => setShowRaw((v) => !v)}
            >
              {showRaw ? 'Rendered' : 'Raw'}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-zinc-400 hover:text-zinc-100"
              onClick={onDismiss}
            >
              <X className="h-4 w-4" />
              <span className="sr-only">Dismiss</span>
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        {showRaw ? (
          <pre className="whitespace-pre-wrap font-mono text-sm text-zinc-200 leading-relaxed">
            {markdown}
          </pre>
        ) : (
          <div className="prose prose-invert prose-sm max-w-none text-zinc-200 leading-relaxed [&_h1]:text-zinc-100 [&_h2]:text-zinc-100 [&_h3]:text-zinc-100 [&_strong]:text-zinc-100 [&_ul]:text-zinc-200 [&_ol]:text-zinc-200 [&_li]:text-zinc-200 [&_p]:text-zinc-200">
            <ReactMarkdown>{markdown}</ReactMarkdown>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
