import { randomUUID } from 'node:crypto'
import type { IncomingMessage, Server } from 'node:http'
import { WebSocketServer, WebSocket } from 'ws'
import { env } from './config/env.js'
import { getSession } from './store/sessionStore.js'
import { computePeakLevel, pcm16ToWav, toBase64 } from './services/audio.js'
import { createDeepgramTranscriptionSocket, synthesizeSpeech } from './services/deepgram.js'
import { analyzeUtterance, buildOpeningLine } from './services/heuristics.js'
import { generateAssistantReply } from './services/llm.js'
import { identifySpeaker } from './services/speakerId.js'
import type { LiveSession, SpeakerRole, TranscriptEntry } from './types.js'

const ASSISTANT_BARGE_IN_PEAK = 0.18

const sendJson = (ws: WebSocket, payload: Record<string, unknown>): void => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload))
  }
}

const matchSessionId = (pathname: string): string | null => {
  const match = pathname.match(/^\/api\/sessions\/([^/]+)\/live$/)
  return match?.[1] ?? null
}

const labelSpeaker = (session: LiveSession, role: SpeakerRole): string => {
  if (role === 'mirror') return 'Mirror'
  if (role === 'unknown') return 'Unknown speaker'
  return session.profiles.find((p) => p.role === role)?.displayName ?? 'Partner'
}

const pushTranscript = (session: LiveSession, entry: TranscriptEntry): void => {
  session.transcript.push(entry)

  if (entry.speakerRole !== 'mirror' && entry.speakerRole !== 'system') {
    session.recentUserTurns.push(entry)
    session.recentUserTurns = session.recentUserTurns.slice(-8)
    session.userTurnsSinceReply += 1
    session.chatHistory.push({
      role: 'user',
      content: `[${entry.speakerLabel}]: ${entry.text}`,
    })
  } else if (entry.speakerRole === 'mirror') {
    session.chatHistory.push({
      role: 'assistant',
      content: entry.text,
    })
  }
}

const estimateSpeechDurationMs = (text: string): number =>
  Math.max(1400, Math.min(7000, text.split(/\s+/).length * 420))

const speakAssistantLine = async (
  session: LiveSession,
  ws: WebSocket,
  text: string,
  replyToken: number,
): Promise<void> => {
  const normalized = text.trim()

  if (!normalized || replyToken !== session.currentReplyToken) {
    return
  }

  const entry: TranscriptEntry = {
    id: randomUUID(),
    speakerRole: 'mirror',
    speakerLabel: 'Mirror',
    text: normalized,
    createdAt: new Date().toISOString(),
    source: 'mirror-reply',
  }

  pushTranscript(session, entry)
  sendJson(ws, { type: 'transcript', entry })

  try {
    const audio = await synthesizeSpeech(normalized)

    if (!audio || replyToken !== session.currentReplyToken) {
      sendJson(ws, { type: 'status', status: 'listening' })
      return
    }

    session.isSpeaking = true
    sendJson(ws, { type: 'status', status: 'speaking' })
    sendJson(ws, { type: 'assistant_audio', replyToken })
    ws.send(audio)

    const tokenAtSend = replyToken
    setTimeout(() => {
      if (session.currentReplyToken === tokenAtSend && session.isSpeaking) {
        session.isSpeaking = false
        sendJson(ws, { type: 'status', status: 'listening' })
      }
    }, estimateSpeechDurationMs(normalized))
  } catch (error) {
    console.error('[reflect-ai-poc-be] Unable to synthesize assistant speech.', error)
    session.isSpeaking = false
    sendJson(ws, { type: 'status', status: 'listening' })
  }
}

const finalizeUtterance = async (
  session: LiveSession,
  ws: WebSocket,
  text: string,
  peakLevel: number,
  audioBuffer: Buffer,
): Promise<void> => {
  const wav = pcm16ToWav(audioBuffer, 16000)
  const identification = await identifySpeaker({
    audioBase64: toBase64(wav),
    profiles: session.profiles,
  })

  const speakerRole = identification.role
  const entry: TranscriptEntry = {
    id: randomUUID(),
    speakerRole,
    speakerLabel: labelSpeaker(session, speakerRole),
    text,
    createdAt: new Date().toISOString(),
    confidence: identification.confidence,
    source: 'deepgram-live',
  }

  pushTranscript(session, entry)
  sendJson(ws, { type: 'transcript', entry })

  const intervention = analyzeUtterance(session, entry, peakLevel)
  if (intervention) {
    session.interventions.push(intervention)
    sendJson(ws, { type: 'intervention', intervention })
  }

  const shouldReply = Boolean(intervention) || session.userTurnsSinceReply >= 2
  if (!shouldReply || session.isProcessingReply) {
    return
  }

  session.isProcessingReply = true
  session.userTurnsSinceReply = 0
  const replyToken = ++session.currentReplyToken
  sendJson(ws, { type: 'status', status: 'thinking' })

  try {
    const reply = intervention ? intervention.line : await generateAssistantReply(session, entry)
    if (replyToken === session.currentReplyToken) {
      await speakAssistantLine(session, ws, reply, replyToken)
    }
  } finally {
    session.isProcessingReply = false
  }
}

const handleDeepgramPayload = async (
  session: LiveSession,
  ws: WebSocket,
  payload: unknown,
): Promise<void> => {
  const message = payload as {
    type?: string
    is_final?: boolean
    channel?: {
      alternatives?: Array<{ transcript?: string }>
    }
  }

  if (message.type !== 'Results') {
    return
  }

  const transcript = message.channel?.alternatives?.[0]?.transcript?.trim() ?? ''

  if (transcript && !message.is_final) {
    sendJson(ws, {
      type: 'transcript',
      interimText: transcript,
      interimSpeakerLabel: 'Listening...',
    })
    return
  }

  if (!message.is_final) {
    return
  }

  const audioBuffer = Buffer.concat(session.currentAudioChunks)
  const peakLevel = session.currentPeakLevel
  session.currentAudioChunks = []
  session.currentPeakLevel = 0

  if (!transcript || audioBuffer.length === 0) {
    return
  }

  await finalizeUtterance(session, ws, transcript, peakLevel, audioBuffer)
}

const handleBrowserControlMessage = (
  session: LiveSession,
  ws: WebSocket,
  rawPayload: string,
): void => {
  try {
    const message = JSON.parse(rawPayload) as { type?: string; replyToken?: unknown }

    if (
      message.type === 'assistant_audio_finished' &&
      typeof message.replyToken === 'number' &&
      message.replyToken === session.currentReplyToken
    ) {
      session.isSpeaking = false
      sendJson(ws, { type: 'status', status: 'listening' })
    }
  } catch {
    // Ignore malformed browser control payloads.
  }
}

export const attachSessionGateway = (server: Server): void => {
  const wss = new WebSocketServer({ noServer: true })

  server.on('upgrade', (request: IncomingMessage, socket, head) => {
    const url = new URL(request.url || '', `http://${request.headers.host || 'localhost'}`)
    const sessionId = matchSessionId(url.pathname)

    if (!sessionId) {
      return
    }

    wss.handleUpgrade(request, socket, head, (ws) => {
      wss.emit('connection', ws, request, sessionId)
    })
  })

  wss.on('connection', (ws: WebSocket, _request: IncomingMessage, sessionId: string) => {
    const session = getSession(sessionId)

    if (!session) {
      sendJson(ws, { type: 'error', message: 'Session not found.' })
      ws.close(4004, 'Session not found')
      return
    }

    session.status = 'live'

    if (!env.DEEPGRAM_API_KEY) {
      sendJson(ws, { type: 'error', message: 'DEEPGRAM_API_KEY is not configured on the backend.' })
      ws.close(4000, 'Deepgram missing')
      return
    }

    const deepgram = createDeepgramTranscriptionSocket()
    let closing = false

    deepgram.on('open', () => {
      sendJson(ws, { type: 'connected', sessionId })
      const openingLine = buildOpeningLine(session)
      const openingToken = ++session.currentReplyToken
      void speakAssistantLine(session, ws, openingLine, openingToken)
    })

    deepgram.on('message', (raw) => {
      try {
        const payload = JSON.parse(raw.toString())
        void handleDeepgramPayload(session, ws, payload)
      } catch (error) {
        console.warn('[reflect-ai-poc-be] Failed to parse Deepgram payload.', error)
      }
    })

    deepgram.on('error', (error) => {
      console.error('[reflect-ai-poc-be] Deepgram realtime socket failed.', error)
      sendJson(ws, { type: 'error', message: 'Deepgram transcription failed during the live session.' })
    })

    deepgram.on('close', () => {
      if (!closing && ws.readyState === WebSocket.OPEN) {
        sendJson(ws, { type: 'status', status: 'error' })
      }
    })

    ws.on('message', (data, isBinary) => {
      if (!isBinary) {
        handleBrowserControlMessage(session, ws, data.toString())
        return
      }

      const chunk = Buffer.from(data as Buffer)
      const chunkPeakLevel = computePeakLevel(chunk)

      if (session.isSpeaking) {
        if (chunkPeakLevel < ASSISTANT_BARGE_IN_PEAK) {
          return
        }

        session.isSpeaking = false
        session.currentReplyToken += 1
        session.currentAudioChunks = []
        session.currentPeakLevel = 0
        sendJson(ws, { type: 'clear_audio' })
        sendJson(ws, { type: 'status', status: 'listening' })
      }

      session.currentAudioChunks.push(chunk)
      if (session.currentAudioChunks.length > 200) {
        session.currentAudioChunks = session.currentAudioChunks.slice(-200)
      }
      session.currentPeakLevel = Math.max(session.currentPeakLevel, chunkPeakLevel)

      if (deepgram.readyState === WebSocket.OPEN) {
        deepgram.send(chunk)
      }
    })

    ws.on('close', () => {
      closing = true
      if (deepgram.readyState === WebSocket.OPEN) {
        deepgram.send(JSON.stringify({ type: 'CloseStream' }))
        deepgram.close()
      }
    })

    ws.on('error', (error) => {
      console.error('[reflect-ai-poc-be] Browser websocket failed.', error)
    })
  })
}
