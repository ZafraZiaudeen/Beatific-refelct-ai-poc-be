import { randomUUID } from 'node:crypto'
import type { InterventionEvent, LiveSession, TranscriptEntry } from '../types.js'

const DENIAL_PATTERNS = [
  /that'?s not true/i,
  /i never said that/i,
  /you'?re wrong/i,
  /i don'?t do that/i,
]

const AI_ATTACK_PATTERNS = [
  /you'?re just (a|an) (bot|ai|machine)/i,
  /you don'?t know us/i,
  /you don'?t understand/i,
  /shut up/i,
]

const CONTEMPT_PATTERNS = [
  /ridiculous/i,
  /pathetic/i,
  /embarrassing/i,
  /disgusting/i,
  /grow up/i,
]

const CRITICISM_PATTERNS = [
  /you always/i,
  /you never/i,
  /what is wrong with you/i,
]

const clip = (text: string, maxLength = 110): string => {
  const normalized = text.replace(/\s+/g, ' ').trim()
  if (normalized.length <= maxLength) {
    return normalized
  }
  return `${normalized.slice(0, Math.max(0, maxLength - 3)).trimEnd()}...`
}

const countWords = (text: string): number => text.trim().split(/\s+/).filter(Boolean).length

const buildEvent = (
  triggerType: string,
  severity: InterventionEvent['severity'],
  line: string,
  transcriptRefs: string[],
): InterventionEvent => ({
  id: randomUUID(),
  triggerType,
  severity,
  line,
  createdAt: new Date().toISOString(),
  transcriptRefs,
})

const overlapScore = (left: string, right: string): number => {
  const leftWords = left.toLowerCase().split(/\s+/).filter((word) => word.length > 3)
  const rightWords = right.toLowerCase().split(/\s+/).filter((word) => word.length > 3)

  if (leftWords.length === 0 || rightWords.length === 0) {
    return 0
  }

  const shared = leftWords.filter((word) => rightWords.includes(word)).length
  return shared / Math.max(leftWords.length, rightWords.length)
}

const getEvidenceQuote = (session: LiveSession, currentEntry: TranscriptEntry): TranscriptEntry | null =>
  [...session.recentUserTurns]
    .reverse()
    .find((entry) => entry.id !== currentEntry.id && entry.speakerRole === currentEntry.speakerRole) ?? null

export const buildOpeningLine = (session: LiveSession): string => {
  const partnerA = session.profiles.find((profile) => profile.role === 'partner_a')?.displayName ?? 'Partner A'
  const partnerB = session.profiles.find((profile) => profile.role === 'partner_b')?.displayName ?? 'Partner B'

  if (session.previousSummary?.nextOpeningPrompt) {
    return `Welcome back, ${partnerA} and ${partnerB}. Last time I asked you to return to this: ${session.previousSummary.nextOpeningPrompt}`
  }

  return `Welcome, ${partnerA} and ${partnerB}. Start with the real issue, not the polished version. What keeps pulling this conversation off course?`
}

export const analyzeUtterance = (
  session: LiveSession,
  entry: TranscriptEntry,
  peakLevel: number,
): InterventionEvent | null => {
  const transcriptRefs = [entry.id]
  const wordCount = countWords(entry.text)

  if (peakLevel >= 0.9 && wordCount >= 5) {
    return buildEvent(
      'raised_intensity',
      'urgent',
      'Pause. The volume just became the message. Lower it and say the point again without force.',
      transcriptRefs,
    )
  }

  if (entry.speakerRole === 'unknown' && peakLevel >= 0.84 && wordCount >= 6) {
    return buildEvent(
      'overlap_or_chaos',
      'firm',
      'One at a time. I am hearing chaos instead of clarity, so slow down and give one clean sentence each.',
      transcriptRefs,
    )
  }

  if (AI_ATTACK_PATTERNS.some((pattern) => pattern.test(entry.text))) {
    return buildEvent(
      'ai_attack',
      'urgent',
      'Turning on me instead of answering the point is still avoidance. Stay with the behavior that was just named.',
      transcriptRefs,
    )
  }

  if (DENIAL_PATTERNS.some((pattern) => pattern.test(entry.text))) {
    const quote = getEvidenceQuote(session, entry)
    return buildEvent(
      'denial',
      'firm',
      quote
        ? `I am not guessing. A moment ago you said, "${clip(quote.text)}" and now you are denying the pattern. Stay with that contradiction.`
        : 'You are denying the pattern instead of addressing it. Slow down and answer directly.',
      quote ? [entry.id, quote.id] : transcriptRefs,
    )
  }

  if (CONTEMPT_PATTERNS.some((pattern) => pattern.test(entry.text))) {
    return buildEvent(
      'contempt',
      'urgent',
      'That was contempt, not honesty. Say the complaint without attacking the other person.',
      transcriptRefs,
    )
  }

  if (CRITICISM_PATTERNS.some((pattern) => pattern.test(entry.text))) {
    return buildEvent(
      'criticism',
      'firm',
      'That is criticism. Bring it back to one specific behavior instead of attacking the whole person.',
      transcriptRefs,
    )
  }

  const repeatedTurns = session.recentUserTurns
    .slice(-4)
    .filter((recent) => overlapScore(recent.text, entry.text) >= 0.55)

  if (repeatedTurns.length >= 2) {
    return buildEvent(
      'circular_arguing',
      'firm',
      'You are both circling the same point again. Stop repeating the case and name what each of you wants next.',
      [entry.id, ...repeatedTurns.map((turn) => turn.id)],
    )
  }

  return null
}

export const buildFallbackRecommendations = (session: LiveSession): string[] => {
  const triggers = Array.from(new Set(session.interventions.map((event) => event.triggerType)))

  const recommendations = [
    'Set a ten-minute check-in this week where each partner speaks for two uninterrupted minutes.',
    'Before the next conversation, each partner writes one concrete need and one concrete accountability statement.',
  ]

  if (triggers.includes('raised_intensity') || triggers.includes('overlap_or_chaos')) {
    recommendations.unshift('Use a reset rule: if volume rises, pause for sixty seconds before continuing.')
  }

  if (triggers.includes('denial') || triggers.includes('circular_arguing')) {
    recommendations.push("When challenged, repeat the other person's point back before defending your own position.")
  }

  return recommendations.slice(0, 3)
}
