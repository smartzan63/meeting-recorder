import { useReducer, useEffect, useCallback, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Badge } from '@/components/ui/badge'
import { useWebSocket } from '@/hooks/useWebSocket'
import { RecordPanel } from '@/components/RecordPanel'
import { TranscriptPanel } from '@/components/TranscriptPanel'

// --- Types ---

type AppStatus = 'idle' | 'recording' | 'stopped' | 'transcribing' | 'done' | 'error'

type Integrations = {
  confluence: boolean
  notion: boolean
  test_file_path?: string | null
}

type AppState = {
  status: AppStatus
  statusMessage: string
  timerSeconds: number
  transcript: string
  originalTranscript: string
  transcriptModel: string
  savedWavPath: string | null
  defaultRecordingName: string
  showSaveDialog: boolean
  showProcessPrompt: boolean
  summaryMarkdown: string
  isSummarizing: boolean
  selectedModel: string
  translateEnabled: boolean
  history: Array<{ id: string; source: string; model: string; created_at: string; has_summary?: boolean }>
  enrichedSpeakers: Record<string, string>
  speakersList: string[]
  isExporting: boolean
  lastExportUrl: string | null
  currentRecordingName: string | null
  integrations: Integrations
}

type Model = {
  key: string
  label: string
  default: boolean
}

// --- Actions ---

type Action =
  | { type: 'SET_STATUS'; status: AppStatus; message?: string }
  | { type: 'SET_STATUS_MESSAGE'; message: string }
  | { type: 'TIMER_TICK' }
  | { type: 'TIMER_RESET' }
  | { type: 'TIMER_INIT'; seconds: number }
  | { type: 'SET_TRANSCRIPT'; text: string; model: string }
  | { type: 'UPDATE_TRANSCRIPT'; text: string }
  | { type: 'OPEN_SAVE_DIALOG'; defaultName: string }
  | { type: 'SET_SAVED_WAV'; path: string }
  | { type: 'HIDE_PROCESS_PROMPT' }
  | { type: 'HIDE_SAVE_DIALOG' }
  | { type: 'SET_SUMMARY'; markdown: string }
  | { type: 'SET_SUMMARIZING'; value: boolean }
  | { type: 'DISMISS_SUMMARY' }
  | { type: 'SET_MODEL'; key: string }
  | { type: 'SET_TRANSLATE'; value: boolean }
  | { type: 'SET_HISTORY'; items: AppState['history'] }
  | { type: 'SET_ENRICHED_SPEAKERS'; speakers: Record<string, string> }
  | { type: 'SET_SPEAKERS_LIST'; list: string[] }
  | { type: 'SET_EXPORTING'; value: boolean }
  | { type: 'SET_EXPORT_RESULT'; url: string | null }
  | { type: 'SET_RECORDING_NAME'; name: string | null }
  | { type: 'SET_INTEGRATIONS'; integrations: Integrations }

// --- Reducer ---

const initialState: AppState = {
  status: 'idle',
  statusMessage: '',
  timerSeconds: 0,
  transcript: '',
  originalTranscript: '',
  transcriptModel: '',
  savedWavPath: null,
  defaultRecordingName: '',
  showSaveDialog: false,
  showProcessPrompt: false,
  summaryMarkdown: '',
  isSummarizing: false,
  selectedModel: '',
  translateEnabled: false,
  history: [],
  enrichedSpeakers: {},
  speakersList: [],
  isExporting: false,
  lastExportUrl: null,
  currentRecordingName: null,
  integrations: { confluence: false, notion: false },
}

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_STATUS': {
      const base = {
        ...state,
        status: action.status,
        statusMessage: action.message ?? state.statusMessage,
      }
      if (action.status === 'idle') {
        return {
          ...base,
          timerSeconds: 0,
          showSaveDialog: false,
          showProcessPrompt: false,
          statusMessage: '',
        }
      }
      if (action.status === 'recording') {
        return { ...base, timerSeconds: 0, statusMessage: 'Recording…' }
      }
      if (action.status === 'stopped') {
        return { ...base, timerSeconds: 0, statusMessage: '' }
      }
      if (action.status === 'done') {
        return { ...base, statusMessage: action.message ?? 'Transcript ready.' }
      }
      if (action.status === 'error') {
        return { ...base, statusMessage: action.message ?? 'An error occurred.' }
      }
      return base
    }
    case 'SET_STATUS_MESSAGE':
      return { ...state, statusMessage: action.message }
    case 'TIMER_TICK':
      return { ...state, timerSeconds: state.timerSeconds + 1 }
    case 'TIMER_RESET':
      return { ...state, timerSeconds: 0 }
    case 'TIMER_INIT':
      return { ...state, timerSeconds: action.seconds }
    case 'SET_TRANSCRIPT':
      return {
        ...state,
        transcript: action.text,
        originalTranscript: action.text,
        transcriptModel: action.model,
        status: 'done',
        statusMessage: 'Transcript ready.',
        showSaveDialog: false,
        showProcessPrompt: false,
        lastExportUrl: null,
        currentRecordingName: null,
        summaryMarkdown: '',
        enrichedSpeakers: {},
        speakersList: [],
      }
    case 'UPDATE_TRANSCRIPT':
      return { ...state, transcript: action.text }
    case 'OPEN_SAVE_DIALOG':
      return {
        ...state,
        defaultRecordingName: action.defaultName,
        showSaveDialog: true,
        showProcessPrompt: false,
      }
    case 'SET_SAVED_WAV':
      return {
        ...state,
        savedWavPath: action.path,
        showSaveDialog: false,
        showProcessPrompt: true,
      }
    case 'HIDE_PROCESS_PROMPT':
      return { ...state, showProcessPrompt: false }
    case 'HIDE_SAVE_DIALOG':
      return { ...state, showSaveDialog: false }
    case 'SET_SUMMARY':
      return { ...state, summaryMarkdown: action.markdown, isSummarizing: false }
    case 'SET_SUMMARIZING':
      return { ...state, isSummarizing: action.value }
    case 'DISMISS_SUMMARY':
      return { ...state, summaryMarkdown: '' }
    case 'SET_MODEL':
      return { ...state, selectedModel: action.key }
    case 'SET_TRANSLATE':
      return { ...state, translateEnabled: action.value }
    case 'SET_HISTORY':
      return { ...state, history: action.items }
    case 'SET_ENRICHED_SPEAKERS':
      return { ...state, enrichedSpeakers: action.speakers }
    case 'SET_SPEAKERS_LIST':
      return { ...state, speakersList: action.list }
    case 'SET_EXPORTING':
      return { ...state, isExporting: action.value }
    case 'SET_EXPORT_RESULT':
      return { ...state, lastExportUrl: action.url }
    case 'SET_RECORDING_NAME':
      return { ...state, currentRecordingName: action.name }
    case 'SET_INTEGRATIONS':
      return { ...state, integrations: action.integrations }
    default:
      return state
  }
}

// --- API helpers ---

async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${path} failed: ${res.status} ${text}`)
  }
  return res.json() as Promise<T>
}

async function fetchHistory() {
  const res = await fetch('/transcripts')
  if (!res.ok) return []
  return res.json() as Promise<Array<{ id: string; source: string; model: string; created_at: string; has_summary?: boolean }>>
}

async function fetchTranscript(id: string): Promise<{ text: string; meta: { source: string; model: string; created_at: string }; summary?: string; speakers?: Record<string, string>; speakers_list?: string[] }> {
  const res = await fetch(`/transcripts/${id}`)
  if (!res.ok) throw new Error('Failed to load transcript')
  return res.json()
}

// --- App ---

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const stoppingRef = useRef(false)
  const { lastMessage } = useWebSocket()

  // Fetch models
  const { data: models = [] } = useQuery<Model[]>({
    queryKey: ['models'],
    queryFn: async () => {
      const res = await fetch('/models')
      if (!res.ok) return []
      return res.json()
    },
  })

  // Set default model once models load
  useEffect(() => {
    if (models.length > 0 && !state.selectedModel) {
      const def = models.find((m) => m.default) ?? models[0]
      dispatch({ type: 'SET_MODEL', key: def.key })
    }
  }, [models, state.selectedModel])

  // Timer
  useEffect(() => {
    if (state.status === 'recording') {
      timerRef.current = setInterval(() => {
        dispatch({ type: 'TIMER_TICK' })
      }, 1000)
    } else {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }, [state.status])

  // WebSocket message handler
  useEffect(() => {
    if (!lastMessage) return

    if (lastMessage.type === 'status') {
      const serverState = lastMessage.state ?? ''
      const msg = lastMessage.message ?? ''

      if (serverState === 'recording' && (state.status === 'idle' || state.status === 'done')) {
        dispatch({ type: 'SET_STATUS', status: 'recording' })
        if (lastMessage.started_at) {
          const elapsed = Math.floor(Date.now() / 1000 - lastMessage.started_at)
          dispatch({ type: 'TIMER_INIT', seconds: Math.max(0, elapsed) })
        }
      } else if (serverState === 'stopped' && state.status !== 'stopped' && state.status !== 'idle') {
        // Edge case: reconnected while user was naming — go back to idle
        dispatch({ type: 'SET_STATUS', status: 'idle' })
      } else if (serverState === 'transcribing') {
        dispatch({ type: 'SET_STATUS', status: 'transcribing', message: msg || 'Transcribing…' })
      } else if (serverState === 'error') {
        dispatch({ type: 'SET_STATUS', status: 'error', message: msg || 'An error occurred.' })
      } else if (serverState === 'done') {
        dispatch({ type: 'SET_STATUS', status: 'done', message: 'Transcript ready.' })
      } else if (serverState === 'idle' && state.status === 'transcribing') {
        dispatch({ type: 'SET_STATUS', status: 'idle' })
      }
    } else if (lastMessage.type === 'transcript') {
      dispatch({
        type: 'SET_TRANSCRIPT',
        text: lastMessage.text ?? '',
        model: lastMessage.model ?? '',
      })
      if (lastMessage.id) {
        dispatch({ type: 'SET_RECORDING_NAME', name: lastMessage.id })
      }
      if (Array.isArray(lastMessage.speakers_list)) {
        dispatch({ type: 'SET_SPEAKERS_LIST', list: lastMessage.speakers_list })
      }
      // Refresh history
      void fetchHistory().then((items) => dispatch({ type: 'SET_HISTORY', items }))
    }
  }, [lastMessage, state.status])

  // Load history and integrations on mount
  useEffect(() => {
    void fetchHistory().then((items) => dispatch({ type: 'SET_HISTORY', items }))
    void fetch('/integrations').then((r) => r.ok ? r.json() : null).then((data) => {
      if (data) dispatch({ type: 'SET_INTEGRATIONS', integrations: data })
    })
  }, [])

  // --- Handlers ---

  const handleStartRecording = useCallback(async () => {
    try {
      await apiPost('/recording/start')
      dispatch({ type: 'SET_STATUS', status: 'recording' })
    } catch (err) {
      dispatch({ type: 'SET_STATUS', status: 'error', message: String(err) })
    }
  }, [])

  const handleStopRecording = useCallback(async () => {
    if (stoppingRef.current) return
    stoppingRef.current = true
    try {
      const res = await apiPost<{ state: string; default_name: string }>('/recording/stop')
      dispatch({ type: 'OPEN_SAVE_DIALOG', defaultName: res.default_name })
    } catch (err) {
      dispatch({ type: 'SET_STATUS', status: 'error', message: `Failed to stop: ${(err as Error).message}` })
    } finally {
      stoppingRef.current = false
    }
  }, [])

  const handleSaveRecording = useCallback(async (name: string) => {
    try {
      const res = await apiPost<{ wav_path: string; name: string }>('/recording/save', { name })
      dispatch({ type: 'SET_SAVED_WAV', path: res.wav_path })
      dispatch({ type: 'SET_RECORDING_NAME', name: res.name })
    } catch (err) {
      dispatch({ type: 'SET_STATUS', status: 'error', message: String(err) })
    }
  }, [])

  const handleProcessNow = useCallback(async () => {
    if (!state.savedWavPath) return
    try {
      dispatch({ type: 'HIDE_PROCESS_PROMPT' })
      dispatch({ type: 'SET_STATUS', status: 'transcribing', message: 'Transcribing…' })
      await apiPost('/test/process', {
        file: state.savedWavPath,
        model: state.selectedModel,
        task: state.translateEnabled ? 'translate' : 'transcribe',
      })
    } catch (err) {
      dispatch({ type: 'SET_STATUS', status: 'error', message: String(err) })
    }
  }, [state.savedWavPath, state.selectedModel, state.translateEnabled])

  const handleSkipProcess = useCallback(() => {
    dispatch({ type: 'SET_STATUS', status: 'idle' })
  }, [])

  const handleDismissError = useCallback(async () => {
    await apiPost('/reset')
    dispatch({ type: 'SET_STATUS', status: 'idle' })
  }, [])

  const handleCancelSave = useCallback(() => {
    dispatch({ type: 'SET_STATUS', status: 'idle' })
    void apiPost('/reset')
  }, [])

  const handleFileUpload = useCallback(async (file: File) => {
    try {
      dispatch({ type: 'SET_STATUS', status: 'transcribing', message: 'Uploading…' })
      const formData = new FormData()
      formData.append('file', file)
      formData.append('model', state.selectedModel)
      formData.append('task', state.translateEnabled ? 'translate' : 'transcribe')
      const res = await fetch('/upload', { method: 'POST', body: formData })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`Upload failed: ${res.status} ${text}`)
      }
      dispatch({ type: 'SET_STATUS_MESSAGE', message: 'Transcribing…' })
    } catch (err) {
      dispatch({ type: 'SET_STATUS', status: 'error', message: String(err) })
    }
  }, [state.selectedModel, state.translateEnabled])

  const handleEnrichAndSummarize = useCallback(async () => {
    if (!state.transcript) return
    dispatch({ type: 'SET_SUMMARIZING', value: true })
    try {
      // Start from the display transcript — user-edited names already applied here.
      // Enrichment can only fill in remaining SPEAKER_XX tags (unnamed speakers).
      let textForSummary = state.transcript

      // Step 1: enrich — never throws, gracefully returns {} on failure
      try {
        const enrichRes = await apiPost<{ speakers: Record<string, string> }>('/enrich', { text: state.originalTranscript || state.transcript, model: state.selectedModel })
        if (enrichRes.speakers && Object.keys(enrichRes.speakers).length > 0) {
          dispatch({ type: 'SET_ENRICHED_SPEAKERS', speakers: enrichRes.speakers })

          // Apply enrichment on top of the user-edited transcript so user edits win.
          // Any SPEAKER_XX already replaced by the user won't be in state.transcript,
          // so enrichment only fills in tags for speakers the user hasn't named yet.
          let enriched = state.transcript
          for (const [tag, name] of Object.entries(enrichRes.speakers)) {
            if (name.trim()) enriched = enriched.replaceAll(tag, name.trim())
          }
          textForSummary = enriched
        }
      } catch {
        // Enrichment failure is non-fatal — continue to summarize
      }

      // Step 2: summarize using enriched text
      const res = await apiPost<{ summary: string }>('/summarize', {
        text: textForSummary,
        model: state.selectedModel,
        ...(state.currentRecordingName ? { name: state.currentRecordingName } : {}),
      })
      dispatch({ type: 'SET_SUMMARY', markdown: res.summary })
    } catch (err) {
      dispatch({ type: 'SET_SUMMARIZING', value: false })
      dispatch({ type: 'SET_STATUS', status: 'error', message: String(err) })
    }
  }, [state.transcript, state.originalTranscript, state.currentRecordingName])

  const handleSpeakersPersist = useCallback(async (speakers: Record<string, string>) => {
    if (!state.currentRecordingName) return
    await fetch(`/transcripts/${state.currentRecordingName}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speakers }),
    })
  }, [state.currentRecordingName])

  const handleSummaryDismiss = useCallback(() => {
    dispatch({ type: 'DISMISS_SUMMARY' })
  }, [])

  const handleDeleteHistory = useCallback(async (id: string) => {
    const res = await fetch(`/transcripts/${id}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`Delete failed: ${res.status}`)
    const items = await fetchHistory()
    dispatch({ type: 'SET_HISTORY', items })
  }, [])

  const handleLoadHistory = useCallback(async (id: string, _source: string) => {
    try {
      const data = await fetchTranscript(id)
      dispatch({ type: 'SET_TRANSCRIPT', text: data.text, model: data.meta.model })
      dispatch({ type: 'SET_RECORDING_NAME', name: id })
      if (data.speakers_list && data.speakers_list.length > 0) {
        dispatch({ type: 'SET_SPEAKERS_LIST', list: data.speakers_list })
      }
      if (data.speakers && Object.keys(data.speakers).length > 0) {
        dispatch({ type: 'SET_ENRICHED_SPEAKERS', speakers: data.speakers })
      }
      if (data.summary) {
        dispatch({ type: 'SET_SUMMARY', markdown: data.summary })
      }
    } catch (err) {
      dispatch({ type: 'SET_STATUS', status: 'error', message: String(err) })
    }
  }, [])

  const handleTranscriptChange = useCallback((text: string) => {
    dispatch({ type: 'UPDATE_TRANSCRIPT', text })
  }, [])

  const handleExport = useCallback(async (destination: 'confluence' | 'notion') => {
    dispatch({ type: 'SET_EXPORTING', value: true })
    try {
      const title = state.currentRecordingName || 'Meeting Transcript'

      // When we have a saved recording ID, let the backend load originalTranscript + speakers
      // and apply substitution server-side so exports always reflect the latest speaker names.
      const exportBody: Record<string, string> = { destination, title, summary: state.summaryMarkdown }
      if (state.currentRecordingName) {
        exportBody.id = state.currentRecordingName
      } else {
        exportBody.transcript = state.transcript
      }

      const res = await apiPost<{ status: string; url: string }>('/export', exportBody)
      dispatch({ type: 'SET_EXPORT_RESULT', url: res.url })
    } catch (err) {
      dispatch({ type: 'SET_STATUS', status: 'error', message: String(err) })
    } finally {
      dispatch({ type: 'SET_EXPORTING', value: false })
    }
  }, [state.transcript, state.summaryMarkdown, state.currentRecordingName])

  const handleSummaryChange = useCallback((markdown: string) => {
    dispatch({ type: 'SET_SUMMARY', markdown })
    if (state.currentRecordingName) {
      void fetch(`/transcripts/${state.currentRecordingName}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ summary: markdown }),
      })
    }
  }, [state.currentRecordingName])

  const handleLoadTestFile = useCallback(async () => {
    const path = state.integrations.test_file_path
    if (!path) return
    try {
      dispatch({ type: 'SET_STATUS', status: 'transcribing', message: 'Processing test file…' })
      await apiPost('/test/process', { file: path, model: state.selectedModel })
    } catch (err) {
      dispatch({ type: 'SET_STATUS', status: 'error', message: String(err) })
    }
  }, [state.integrations.test_file_path, state.selectedModel])

  const handleDownload = useCallback(() => {
    const blob = new Blob([state.transcript], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'transcript.txt'
    a.click()
    URL.revokeObjectURL(url)
  }, [state.transcript])

  // Provider label
  const providerLabel = models.length === 1 && models[0]?.key === 'azure' ? 'Azure' : 'Gemini'

  const isLoading = state.status === 'transcribing'

  return (
    <div className="dark min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-border shrink-0">
        <h1 className="text-lg font-medium tracking-wide text-muted-foreground">
          Meeting Recorder
        </h1>
        <Badge variant="outline">{providerLabel}</Badge>
      </header>

      {/* Two-column layout */}
      <div className="flex flex-col md:flex-row flex-1 min-h-0">
        {/* Record panel */}
        <aside className="w-full md:w-[380px] md:shrink-0 border-b md:border-b-0 md:border-r border-zinc-800 overflow-y-auto">
          <RecordPanel
            status={state.status}
            statusMessage={state.statusMessage}
            timerSeconds={state.timerSeconds}
            models={models}
            selectedModel={state.selectedModel}
            onModelChange={(key) => dispatch({ type: 'SET_MODEL', key })}
            savedWavPath={state.savedWavPath}
            defaultRecordingName={state.defaultRecordingName}
            showSaveDialog={state.showSaveDialog}
            showProcessPrompt={state.showProcessPrompt}
            onStartRecording={handleStartRecording}
            onStopRecording={handleStopRecording}
            onSaveRecording={handleSaveRecording}
            onProcessNow={handleProcessNow}
            onSkipProcess={handleSkipProcess}
            onFileUpload={handleFileUpload}
            onDismissError={handleDismissError}
            onCancelSave={handleCancelSave}
          />
          {state.integrations.test_file_path && (
            <div className="px-6 pb-4">
              <button
                onClick={() => void handleLoadTestFile()}
                disabled={isLoading}
                className="w-full text-xs text-zinc-500 border border-dashed border-zinc-700 rounded px-3 py-2 hover:border-zinc-500 hover:text-zinc-400 disabled:opacity-40"
              >
                [DEV] Load test file
              </button>
            </div>
          )}
        </aside>

        {/* Transcript panel */}
        <main className="flex-1 overflow-y-auto p-6">
          <TranscriptPanel
            transcript={state.transcript}
            originalTranscript={state.originalTranscript}
            transcriptModel={state.transcriptModel}
            onTranscriptChange={handleTranscriptChange}
            summaryMarkdown={state.summaryMarkdown}
            onEnrichAndSummarize={handleEnrichAndSummarize}
            enrichedSpeakers={state.enrichedSpeakers}
            speakersList={state.speakersList}
            currentRecordingName={state.currentRecordingName}
            onSpeakersPersist={handleSpeakersPersist}
            onSummaryDismiss={handleSummaryDismiss}
            onSummaryChange={handleSummaryChange}
            isSummarizing={state.isSummarizing}
            history={state.history}
            onLoadHistory={handleLoadHistory}
            onDeleteHistory={handleDeleteHistory}
            isLoading={isLoading}
            isExporting={state.isExporting}
            lastExportUrl={state.lastExportUrl}
            onExport={handleExport}
            onDownload={handleDownload}
            integrations={state.integrations}
          />
        </main>
      </div>
    </div>
  )
}
