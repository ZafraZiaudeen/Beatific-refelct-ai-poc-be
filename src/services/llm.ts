import type { LiveSession, SessionSummaryPayload, TranscriptEntry } from '../types.js'
import { env } from '../config/env.js'
import { buildFallbackRecommendations } from './heuristics.js'

const cleanReply = (value: string): string =>
  value
    .replace(/^["'\s]+|["'\s]+$/g, '')
    .replace(/\s+/g, ' ')
    .trim()

const clip = (text: string, maxLength = 140): string => {
  const normalized = text.replace(/\s+/g, ' ').trim()
  if (normalized.length <= maxLength) {
    return normalized
  }
  return `${normalized.slice(0, Math.max(0, maxLength - 3)).trimEnd()}...`
}

const extractContent = (payload: unknown): string => {
  const choice = (payload as { choices?: Array<{ message?: { content?: unknown } }> }).choices?.[0]
  const content = choice?.message?.content

  if (typeof content === 'string') {
    return content
  }

  if (Array.isArray(content)) {
    return content
      .map((part) =>
        typeof part === 'object' && part && 'text' in part
          ? String((part as { text?: unknown }).text ?? '')
          : '',
      )
      .join('\n')
  }

  return ''
}

const callOpenRouter = async (messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>, maxTokens: number, temperature: number): Promise<string | null> => {
  if (!env.OPENROUTER_API_KEY) {
    return null
  }

  const response = await fetch(`${env.OPENROUTER_BASE_URL.replace(/\/+$/, '')}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.OPENROUTER_API_KEY}`,
      'content-type': 'application/json',
      'HTTP-Referer': env.OPENROUTER_HTTP_REFERER,
      'X-Title': env.OPENROUTER_APP_NAME,
    },
    body: JSON.stringify({
      model: env.OPENROUTER_MODEL,
      messages,
      max_tokens: maxTokens,
      temperature,
    }),
  })

  if (!response.ok) {
    const details = await response.text().catch(() => '')
    throw new Error(`OpenRouter request failed: ${response.status} ${details}`)
  }

  return extractContent(await response.json())
}

const fallbackAssistantReply = (session: LiveSession, latestEntry: TranscriptEntry): string => {
  const partnerNames = session.profiles.map((p) => p.displayName).join(' and ')
  return `${partnerNames}, stay with the concrete behavior you just named. ${latestEntry.speakerLabel}, say what you need next without repeating the whole argument.`
}

const fallbackSummary = (session: LiveSession): SessionSummaryPayload => {
  const firstUserTurn = session.transcript.find((entry) => entry.speakerRole !== 'mirror')
  const lastUserTurn = [...session.transcript].reverse().find((entry) => entry.speakerRole !== 'mirror')

  return {
    coreConflict: firstUserTurn ? clip(firstUserTurn.text, 80) : 'Escalating communication under pressure',
    summary: firstUserTurn
      ? `The conversation kept returning to this issue: ${clip(firstUserTurn.text, 150)}`
      : 'The session stayed short, so Mirror saved a minimal summary.',
    whatChanged: lastUserTurn
      ? `By the end, the conversation landed on: ${clip(lastUserTurn.text, 120)}`
      : 'The session ended before a clearer shift appeared.',
    recommendations: buildFallbackRecommendations(session),
    nextOpeningPrompt: 'Start by saying whether you followed the last recommendation and what still feels unresolved.',
  }
}

export const generateAssistantReply = async (session: LiveSession, latestEntry: TranscriptEntry): Promise<string> => {
  const recentTranscript = session.transcript
    .slice(-8)
    .map((entry) => `${entry.speakerLabel}: ${entry.text}`)
    .join('\n')

  try {
    const content = await callOpenRouter(
      [
        ...session.chatHistory.slice(-10),
        {
          role: 'user',
          content: [
            'Continue the live same-device couples conversation.',
            'Be direct, useful, and short.',
            'Do not sound theatrical or robotic.',
            `Latest speaker: ${latestEntry.speakerLabel}`,
            `Recent transcript:\n${recentTranscript}`,
          ].join('\n\n'),
        },
      ],
      180,
      0.45,
    )

    return cleanReply(content || '') || fallbackAssistantReply(session, latestEntry)
  } catch {
    return fallbackAssistantReply(session, latestEntry)
  }
}

const parseSummaryJson = (raw: string): SessionSummaryPayload | null => {
  const trimmed = raw.trim()
  const jsonText = trimmed.startsWith('{') ? trimmed : trimmed.match(/\{[\s\S]*\}/)?.[0]

  if (!jsonText) {
    return null
  }

  try {
    const parsed = JSON.parse(jsonText) as Partial<SessionSummaryPayload>
    if (
      typeof parsed.coreConflict === 'string' &&
      typeof parsed.summary === 'string' &&
      typeof parsed.whatChanged === 'string' &&
      Array.isArray(parsed.recommendations) &&
      typeof parsed.nextOpeningPrompt === 'string'
    ) {
      return {
        coreConflict: parsed.coreConflict,
        summary: parsed.summary,
        whatChanged: parsed.whatChanged,
        recommendations: parsed.recommendations.map((item) => String(item)),
        nextOpeningPrompt: parsed.nextOpeningPrompt,
      }
    }
  } catch {
    // Fall through to null.
  }

  return null
}

export const generateSessionSummary = async (session: LiveSession): Promise<SessionSummaryPayload> => {
  const transcript = session.transcript
    .slice(-24)
    .map((entry) => `${entry.speakerLabel}: ${entry.text}`)
    .join('\n')

  try {
    const content = await callOpenRouter(
      [
        {
          role: 'system',
          content: [
            'You create concise end-of-session summaries for a same-device couples conversation coach.',
            'Return only JSON with keys: coreConflict, summary, whatChanged, recommendations, nextOpeningPrompt.',
            'recommendations must be an array of 2 or 3 practical strings.',
          ].join('\n'),
        },
        {
          role: 'user',
          content: JSON.stringify(
            {
              previousSummary: session.previousSummary,
              interventions: session.interventions,
              transcript,
            },
            null,
            2,
          ),
        },
      ],
      700,
      0.3,
    )

    const parsed = content ? parseSummaryJson(content) : null
    return parsed ?? fallbackSummary(session)
  } catch {
    return fallbackSummary(session)
  }
}
