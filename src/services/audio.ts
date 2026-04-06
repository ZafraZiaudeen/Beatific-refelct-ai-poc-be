export const computePeakLevel = (buffer: Buffer): number => {
  if (buffer.length < 2) {
    return 0
  }

  const absoluteSamples: number[] = []
  let sumSquares = 0

  for (let offset = 0; offset < buffer.length - 1; offset += 2) {
    const sample = buffer.readInt16LE(offset) / 32768
    const absolute = Math.abs(sample)
    absoluteSamples.push(absolute)
    sumSquares += sample * sample
  }

  if (absoluteSamples.length === 0) {
    return 0
  }

  absoluteSamples.sort((left, right) => left - right)
  const percentileIndex = Math.max(0, Math.floor(absoluteSamples.length * 0.96) - 1)
  const percentilePeak = absoluteSamples[percentileIndex] ?? 0
  const rmsLevel = Math.sqrt(sumSquares / absoluteSamples.length)

  return Math.max(percentilePeak, rmsLevel * 1.8)
}

export const pcm16ToWav = (pcm: Buffer, sampleRate = 16000): Buffer => {
  const header = Buffer.alloc(44)
  const bytesPerSample = 2
  const byteRate = sampleRate * bytesPerSample

  header.write('RIFF', 0)
  header.writeUInt32LE(36 + pcm.length, 4)
  header.write('WAVE', 8)
  header.write('fmt ', 12)
  header.writeUInt32LE(16, 16)
  header.writeUInt16LE(1, 20)
  header.writeUInt16LE(1, 22)
  header.writeUInt32LE(sampleRate, 24)
  header.writeUInt32LE(byteRate, 28)
  header.writeUInt16LE(bytesPerSample, 32)
  header.writeUInt16LE(16, 34)
  header.write('data', 36)
  header.writeUInt32LE(pcm.length, 40)

  return Buffer.concat([header, pcm])
}

export const toBase64 = (buffer: Buffer): string => buffer.toString('base64')
