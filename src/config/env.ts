import dotenv from 'dotenv'
import { z } from 'zod'

dotenv.config()

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),
  PORT: z.coerce.number().default(3100),
  CLIENT_URL: z.string().default('http://localhost:5173'),
  DEEPGRAM_API_KEY: z.string().optional(),
  DEEPGRAM_STT_MODEL: z.string().default('nova-3'),
  DEEPGRAM_TTS_MODEL: z.string().default('aura-2-thalia-en'),
  OPENROUTER_API_KEY: z.string().optional(),
  OPENROUTER_MODEL: z.string().default('openai/gpt-4o-mini'),
  OPENROUTER_BASE_URL: z.string().default('https://openrouter.ai/api/v1'),
  OPENROUTER_HTTP_REFERER: z.string().default('http://localhost:5173'),
  OPENROUTER_APP_NAME: z.string().default('Reflect AI POC'),
  SPEAKER_ID_URL: z.string().default('http://127.0.0.1:8200'),
})

const parsed = envSchema.parse(process.env)

export const env = parsed
