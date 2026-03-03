import { useEffect, useRef, useState } from 'react'

export type WsMessage = {
  type: 'status' | 'transcript'
  state?: string
  message?: string
  started_at?: number
  text?: string
  model?: string
  id?: string
  speakers_list?: string[]
}

export function useWebSocket(): { lastMessage: WsMessage | null } {
  const [lastMessage, setLastMessage] = useState<WsMessage | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true

    function connect() {
      if (!mountedRef.current) return

      const ws = new WebSocket(`ws://${location.host}/ws`)
      wsRef.current = ws

      ws.onmessage = (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data as string) as WsMessage
          if (mountedRef.current) {
            setLastMessage(data)
          }
        } catch {
          // ignore malformed messages
        }
      }

      ws.onclose = () => {
        if (mountedRef.current) {
          reconnectTimerRef.current = setTimeout(() => {
            connect()
          }, 3000)
        }
      }

      ws.onerror = () => {
        ws.close()
      }
    }

    connect()

    return () => {
      mountedRef.current = false
      if (reconnectTimerRef.current !== null) {
        clearTimeout(reconnectTimerRef.current)
      }
      if (wsRef.current) {
        wsRef.current.onclose = null
        wsRef.current.onerror = null
        wsRef.current.close()
      }
    }
  }, [])

  return { lastMessage }
}
