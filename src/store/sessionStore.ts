import { randomUUID } from 'node:crypto'
import type { LiveSession, SessionSummaryPayload, SpeakerProfileInput } from '../types.js'

const sessions = new Map<string, LiveSession>()

const buildSystemPrompt = (session: LiveSession): string => {
  const partnerA = session.profiles.find((p) => p.role === 'partner_a')?.displayName ?? 'Partner A'
  const partnerB = session.profiles.find((p) => p.role === 'partner_b')?.displayName ?? 'Partner B'

  const parts = [
    'You are Mirror, a firm but useful couples conversation coach.',
    'Stay direct, calm, and clinically grounded.',
    'Keep spoken replies short: usually 1 to 3 sentences.',
    'Interrupt circular arguing, denial, contempt, and attacks on the process.',
    'Do not sound like customer support.',
    `The partners are ${partnerA} and ${partnerB}.`,
  ]

  if (session.previousSummary?.summary) {
    parts.push(`Previous summary: ${session.previousSummary.summary}`)
  }

  if (session.previousSummary?.recommendations.length) {
    parts.push(`Previous recommendations: ${session.previousSummary.recommendations.join(' | ')}`)
  }

  return parts.join('\n')
}

export const createSession = (input: {
  profiles: SpeakerProfileInput[]
  previousSummary?: {
    summary: string
    recommendations: string[]
    nextOpeningPrompt: string
  } | null
}): LiveSession => {
  const session: LiveSession = {
    id: randomUUID(),
    startedAt: new Date().toISOString(),
    status: 'created',
    profiles: input.profiles,
    transcript: [],
    interventions: [],
    previousSummary: input.previousSummary ?? null,
    chatHistory: [],
    recentUserTurns: [],
    userTurnsSinceReply: 0,
    currentAudioChunks: [],
    currentPeakLevel: 0,
    currentReplyToken: 0,
    isProcessingReply: false,
    isSpeaking: false,
  }

  session.chatHistory.push({
    role: 'system',
    content: buildSystemPrompt(session),
  })

  sessions.set(session.id, session)
  return session
}

export const getSession = (sessionId: string): LiveSession | undefined => sessions.get(sessionId)

export const completeSessionRecord = (sessionId: string, summary: SessionSummaryPayload): LiveSession => {
  const session = getSession(sessionId)
  if (!session) {
    throw new Error('Session not found.')
  }

  session.endedAt = new Date().toISOString()
  session.status = 'completed'
  session.chatHistory.push({
    role: 'assistant',
    content: `Final summary: ${summary.summary}`,
  })
  return session
}

export const deleteSession = (sessionId: string): void => {
  sessions.delete(sessionId)
}
