import { sanitizeDocumentText } from '../core/security.ts';

export function normalizeText(raw: string): string {
  const sanitized = sanitizeDocumentText(raw);
  return sanitized
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/\u00A0/g, ' ')
    .replace(/[’‘]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .split('\n')
    .map((line) => line.trim())
    .filter((line, idx, arr) => {
      // Remove isolated page numbers like "Page 1/4" or "- 3 -"
      if (/^(page\s+\d+(\s*\/\s*\d+)?|-\s*\d+\s*-)$/i.test(line)) return false;
      // Avoid consecutive duplicate empty lines
      if (line === '' && idx > 0 && arr[idx - 1] === '') return false;
      return true;
    })
    .join('\n')
    .trim();
}
