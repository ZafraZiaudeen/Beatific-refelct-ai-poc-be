export type PartnerRole = 'partner_a' | 'partner_b'
export type SpeakerRole = PartnerRole | 'mirror' | 'system' | 'unknown'
export type SessionStatus = 'created' | 'live' | 'completed'

export interface EnrollmentSampleMeta {
  durationMs: number
  sampleRate: number
}

export interface SpeakerProfileInput {
  id: string
  role: PartnerRole
  displayName: string
  age: string
  embedding: number[]
  sampleMeta: EnrollmentSampleMeta
}

export interface TranscriptEntry {
  id: string
  speakerRole: SpeakerRole
  speakerLabel: string
  text: string
  createdAt: string
  confidence?: number
  source: 'deepgram-live' | 'mirror-reply' | 'system'
}

export interface InterventionEvent {
  id: string
  triggerType: string
  severity: 'watch' | 'firm' | 'urgent'
  line: string
  createdAt: string
  transcriptRefs: string[]
}

export interface SessionSummaryPayload {
  coreConflict: string
  summary: string
  whatChanged: string
  recommendations: string[]
  nextOpeningPrompt: string
}

export interface LiveSession {
  id: string
  startedAt: string
  endedAt?: string
  status: SessionStatus
  profiles: SpeakerProfileInput[]
  transcript: TranscriptEntry[]
  interventions: InterventionEvent[]
  previousSummary?: {
    summary: string
    recommendations: string[]
    nextOpeningPrompt: string
  } | null
  chatHistory: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>
  recentUserTurns: TranscriptEntry[]
  userTurnsSinceReply: number
  currentAudioChunks: Buffer[]
  currentPeakLevel: number
  currentReplyToken: number
  isProcessingReply: boolean
  isSpeaking: boolean
}

export interface SpeakerIdentificationResult {
  role: SpeakerRole
  confidence: number
  scores: Record<string, number>
}
