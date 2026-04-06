import WebSocket from 'ws'
import { env } from '../config/env.js'

const buildDeepgramSttUrl = () =>
  'wss://api.deepgram.com/v1/listen?' +
  `model=${encodeURIComponent(env.DEEPGRAM_STT_MODEL)}` +
  '&language=en&encoding=linear16&sample_rate=16000&channels=1' +
  '&endpointing=300&interim_results=true&utterance_end_ms=1200&vad_events=true&punctuate=true&smart_format=true'

const buildDeepgramTtsUrl = () =>
  `https://api.deepgram.com/v1/speak?model=${encodeURIComponent(env.DEEPGRAM_TTS_MODEL)}&encoding=mp3`

export const assertDeepgramConfigured = (): void => {
  if (!env.DEEPGRAM_API_KEY) {
    throw new Error('DEEPGRAM_API_KEY is required for the realtime voice session.')
  }
}

export const createDeepgramTranscriptionSocket = (): WebSocket => {
  assertDeepgramConfigured()

  return new WebSocket(buildDeepgramSttUrl(), {
    headers: {
      Authorization: `Token ${env.DEEPGRAM_API_KEY}`,
    },
  })
}

export const synthesizeSpeech = async (text: string): Promise<Buffer | null> => {
  if (!env.DEEPGRAM_API_KEY || !text.trim()) {
    return null
  }

  const response = await fetch(buildDeepgramTtsUrl(), {
    method: 'POST',
    headers: {
      Authorization: `Token ${env.DEEPGRAM_API_KEY}`,
      'content-type': 'application/json',
    },
    body: JSON.stringify({ text }),
  })

  if (!response.ok) {
    const details = await response.text().catch(() => '')
    throw new Error(`Deepgram TTS failed: ${response.status} ${details}`)
  }

  return Buffer.from(await response.arrayBuffer())
}
