import { env } from '../config/env.js'
import type { SpeakerIdentificationResult, SpeakerProfileInput } from '../types.js'

const postJson = async <T>(path: string, body: unknown): Promise<T> => {
  const response = await fetch(`${env.SPEAKER_ID_URL.replace(/\/+$/, '')}${path}`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
    },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    const details = await response.text().catch(() => '')
    throw new Error(`Speaker service failed: ${response.status} ${details}`)
  }

  return (await response.json()) as T
}

export const createEmbedding = async (audioBase64: string): Promise<number[]> => {
  const response = await postJson<{ embedding: number[] }>('/embed', { audioBase64 })
  return response.embedding
}

export const identifySpeaker = async (input: {
  audioBase64: string
  profiles: SpeakerProfileInput[]
}): Promise<SpeakerIdentificationResult> => {
  try {
    return await postJson<SpeakerIdentificationResult>('/identify', {
      audioBase64: input.audioBase64,
      profiles: input.profiles.map((profile) => ({
        role: profile.role,
        displayName: profile.displayName,
        embedding: profile.embedding,
      })),
    })
  } catch {
    return {
      role: 'unknown',
      confidence: 0,
      scores: {},
    }
  }
}
