export interface ChunkMetadata {
  documentId: string;
  title: string;
  docType: string;
  category: string;
  serviceName: string;
  referenceCode: string;
  publishedDate: string;
  validUntil: string | null;
  versionNumber: number;
}

export interface DocumentChunkItem {
  chunkIndex: number;
  content: string;
  tokenCount: number;
  charCount: number;
  metadata: ChunkMetadata;
}

export function estimateTokenCount(text: string): number {
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.round(words * 1.35));
}

export function chunkDocumentText(
  normalizedText: string,
  metadata: ChunkMetadata,
  chunkSize = 850,
  chunkOverlap = 160,
): DocumentChunkItem[] {
  const safeSize = Math.max(250, Math.min(2400, chunkSize));
  const safeOverlap = Math.max(0, Math.min(Math.floor(safeSize * 0.45), chunkOverlap));

  const paragraphs = normalizedText
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter(Boolean);

  const chunks: string[] = [];
  let current = '';

  const pushChunk = (body: string) => {
    const trimmed = body.trim();
    if (!trimmed) return;
    const prefix = `[${metadata.referenceCode} — ${metadata.title} | ${metadata.category}]\n`;
    chunks.push(`${prefix}${trimmed}`);
  };

  for (const para of paragraphs) {
    if (para.length > safeSize) {
      if (current) {
        pushChunk(current);
        current = '';
      }
      // Split long paragraph on sentence boundaries
      const sentences = para.split(/(?<=[.!?])\s+/);
      let seg = '';
      for (const sentence of sentences) {
        if ((seg + ' ' + sentence).trim().length <= safeSize) {
          seg = (seg + ' ' + sentence).trim();
        } else {
          if (seg) pushChunk(seg);
          if (sentence.length > safeSize) {
            let start = 0;
            while (start < sentence.length) {
              const slice = sentence.slice(start, start + safeSize);
              pushChunk(slice);
              start += Math.max(100, safeSize - safeOverlap);
            }
            seg = '';
          } else {
            const tail = seg.slice(Math.max(0, seg.length - safeOverlap));
            seg = (tail + ' ' + sentence).trim();
          }
        }
      }
      if (seg) current = seg;
      continue;
    }

    if ((current + '\n\n' + para).trim().length <= safeSize) {
      current = current ? `${current}\n\n${para}` : para;
    } else {
      pushChunk(current);
      const tail = current.slice(Math.max(0, current.length - safeOverlap));
      const cleanTail = tail.includes(' ') ? tail.slice(tail.indexOf(' ') + 1) : tail;
      current = cleanTail ? `${cleanTail}\n\n${para}` : para;
      if (current.length > safeSize) {
        current = para;
      }
    }
  }

  if (current) {
    pushChunk(current);
  }

  return chunks.map((content, idx) => ({
    chunkIndex: idx,
    content,
    tokenCount: estimateTokenCount(content),
    charCount: content.length,
    metadata,
  }));
}
