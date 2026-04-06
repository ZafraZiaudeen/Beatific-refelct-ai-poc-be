import express from 'express'
import { z } from 'zod'
import { env } from './config/env.js'
import { createEmbedding } from './services/speakerId.js'
import { generateSessionSummary } from './services/llm.js'
import { completeSessionRecord, createSession, getSession } from './store/sessionStore.js'

const profileSchema = z.object({
  id: z.string(),
  role: z.enum(['partner_a', 'partner_b']),
  displayName: z.string(),
  age: z.string(),
  embedding: z.array(z.number()),
  sampleMeta: z.object({
    durationMs: z.number(),
    sampleRate: z.number(),
  }),
})

const enrollSchema = z.object({
  role: z.enum(['partner_a', 'partner_b']),
  displayName: z.string().min(1),
  age: z.string().min(1),
  audioBase64: z.string().min(1),
  durationMs: z.number().positive(),
  sampleRate: z.number().positive(),
})

const createSessionSchema = z.object({
  profiles: z.array(profileSchema).length(2),
  latestSummary: z
    .object({
      summary: z.string(),
      recommendations: z.array(z.string()),
      nextOpeningPrompt: z.string(),
    })
    .nullable()
    .optional(),
})

export const createApp = () => {
  const app = express()

  app.use(express.json({ limit: '20mb' }))
  app.use((_request, response, next) => {
    response.header('Access-Control-Allow-Origin', env.CLIENT_URL)
    response.header('Access-Control-Allow-Headers', 'content-type')
    response.header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')

    if (_request.method === 'OPTIONS') {
      response.sendStatus(204)
      return
    }

    next()
  })

  app.get('/api/health', (_request, response) => {
    response.json({
      ok: true,
      services: {
        deepgram: Boolean(env.DEEPGRAM_API_KEY),
        openrouter: Boolean(env.OPENROUTER_API_KEY),
        speakerId: env.SPEAKER_ID_URL,
      },
    })
  })

  app.post('/api/profiles/enroll', async (request, response) => {
    const payload = enrollSchema.parse(request.body)
    const embedding = await createEmbedding(payload.audioBase64)
    response.json({ embedding })
  })

  app.post('/api/sessions', (request, response) => {
    const payload = createSessionSchema.parse(request.body)
    const session = createSession({
      profiles: payload.profiles,
      previousSummary: payload.latestSummary ?? null,
    })

    response.status(201).json({
      sessionId: session.id,
      startedAt: session.startedAt,
    })
  })

  app.post('/api/sessions/:sessionId/complete', async (request, response) => {
    const session = getSession(request.params.sessionId)

    if (!session) {
      response.status(404).send('Session not found.')
      return
    }

    const summary = await generateSessionSummary(session)
    const completed = completeSessionRecord(session.id, summary)

    response.json({
      id: completed.id,
      startedAt: completed.startedAt,
      endedAt: completed.endedAt,
      transcript: completed.transcript,
      interventions: completed.interventions,
      summary,
    })
  })

  app.use((error: unknown, _request: express.Request, response: express.Response, _next: express.NextFunction) => {
    console.error('[reflect-ai-poc-be] Request failed.', error)

    if (error instanceof z.ZodError) {
      response.status(400).json({
        message: error.issues.map((issue) => issue.message).join('; '),
      })
      return
    }

    if (error instanceof Error) {
      response.status(500).json({ message: error.message })
      return
    }

    response.status(500).json({ message: 'Unexpected server error.' })
  })

  return app
}
