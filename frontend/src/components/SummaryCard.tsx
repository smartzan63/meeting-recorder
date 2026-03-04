import { useState, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import { X, Pencil, Check, Copy } from 'lucide-react'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

type SummaryCardProps = {
  markdown: string
  onDismiss: () => void
  onSummaryChange?: (text: string) => void
}

export function SummaryCard({ markdown, onDismiss, onSummaryChange }: SummaryCardProps) {
  const [showRaw, setShowRaw] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editValue, setEditValue] = useState(markdown)
  const [copied, setCopied] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function handleCopy() {
    void navigator.clipboard.writeText(markdown)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  function handleEditClick() {
    setEditValue(markdown)
    setIsEditing(true)
    setTimeout(() => textareaRef.current?.focus(), 0)
  }

  function handleSave() {
    setIsEditing(false)
    onSummaryChange?.(editValue)
  }

  return (
    <Card className="border-zinc-800 bg-zinc-900">
      <CardHeader className="pb-3 pt-4 px-4">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium uppercase tracking-widest text-zinc-400">
            Summary
          </span>
          <div className="flex items-center gap-2">
            {isEditing ? (
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-2 text-xs text-emerald-400 hover:text-emerald-300"
                onClick={handleSave}
              >
                <Check className="h-3.5 w-3.5 mr-1" />
                Save
              </Button>
            ) : (
              <>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-xs text-zinc-400 hover:text-zinc-100 gap-1"
                  onClick={handleCopy}
                >
                  {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                  {copied ? 'Copied' : 'Copy'}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-xs text-zinc-400 hover:text-zinc-100"
                  onClick={() => setShowRaw((v) => !v)}
                >
                  {showRaw ? 'Rendered' : 'Raw'}
                </Button>
                {onSummaryChange && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 px-2 text-xs text-zinc-400 hover:text-zinc-100"
                    onClick={handleEditClick}
                  >
                    <Pencil className="h-3.5 w-3.5" />
                  </Button>
                )}
              </>
            )}
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
        {isEditing ? (
          <textarea
            ref={textareaRef}
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={handleSave}
            className="w-full min-h-48 bg-zinc-800 border border-zinc-700 rounded-md p-3 text-sm text-zinc-200 font-mono leading-relaxed resize-y focus:outline-none focus:ring-1 focus:ring-zinc-600"
          />
        ) : showRaw ? (
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
