import { execFile } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import mammoth from 'mammoth';
import { extractText as extractPdfText } from 'unpdf';
import { performBatchImagesOcr, performDocumentOcr } from './ocr.ts';

const execFileAsync = promisify(execFile);

export type SupportedFormat = 'PDF' | 'DOCX' | 'HTML' | 'TXT' | 'IMAGE';
export type OcrMode = 'auto' | 'force' | 'off';

export interface ExtractionResult {
  text: string;
  ocrUsed: boolean;
  ocrReason: 'image_scan' | 'pdf_scanned_auto' | 'forced' | 'none';
  ocrModel?: string;
  nativeCharCount: number;
  pageCount?: number;
}

export function detectFormat(fileName: string, mimeType?: string): SupportedFormat | null {
  const lower = fileName.toLowerCase();
  const mime = (mimeType || '').toLowerCase();

  if (lower.endsWith('.pdf') || mime === 'application/pdf') return 'PDF';
  if (
    lower.endsWith('.png') ||
    lower.endsWith('.jpg') ||
    lower.endsWith('.jpeg') ||
    lower.endsWith('.webp') ||
    mime === 'image/png' ||
    mime === 'image/jpeg' ||
    mime === 'image/jpg' ||
    mime === 'image/webp'
  ) {
    return 'IMAGE';
  }
  if (
    lower.endsWith('.docx') ||
    mime === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
  ) {
    return 'DOCX';
  }
  if (lower.endsWith('.html') || lower.endsWith('.htm') || mime === 'text/html') return 'HTML';
  if (lower.endsWith('.txt') || mime === 'text/plain') return 'TXT';
  return null;
}

export function resolveOcrMimeType(
  fileName: string,
  format: SupportedFormat,
  mimeType?: string,
): string {
  if (format === 'PDF') return 'application/pdf';
  const lower = fileName.toLowerCase();
  const mime = (mimeType || '').toLowerCase();
  if (lower.endsWith('.png') || mime === 'image/png') return 'image/png';
  if (lower.endsWith('.webp') || mime === 'image/webp') return 'image/webp';
  return 'image/jpeg';
}

function countMeaningfulChars(text: string): number {
  return (text.match(/[a-zA-ZÀ-ÿ0-9]/g) || []).length;
}

async function renderPdfPagesToPngs(inputPdfPath: string, outputDir: string): Promise<string[]> {
  await execFileAsync('gs', [
    '-sDEVICE=png16m',
    '-r150',
    '-dNOPAUSE',
    '-dBATCH',
    '-dSAFER',
    '-q',
    `-sOutputFile=${path.join(outputDir, 'page_%04d.png')}`,
    inputPdfPath,
  ]);
  const files = (await fs.readdir(outputDir))
    .filter((f) => f.startsWith('page_') && f.endsWith('.png'))
    .sort();
  return files.map((f) => path.join(outputDir, f));
}

async function ocrSelectedPdfPagesInBatches(
  buffer: Buffer,
  targetPageIndices0Based: number[],
  batchSize = 3,
  onProgress?: (msg: string) => void,
  checkpointPath?: string,
): Promise<{ pageTextByIndex: Map<number, string>; modelUsed: string }> {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dgecc-ocr-'));
  const inputPath = path.join(tmpDir, 'source.pdf');
  await fs.writeFile(inputPath, buffer);

  try {
    const selected = [...targetPageIndices0Based]
      .filter((i) => i >= 0)
      .sort((a, b) => a - b);
    const batches: number[][] = [];
    for (const pageIdx of selected) {
      const currentBatch = batches[batches.length - 1];
      if (
        currentBatch &&
        currentBatch.length < batchSize &&
        currentBatch[currentBatch.length - 1] + 1 === pageIdx
      ) {
        currentBatch.push(pageIdx);
      } else {
        batches.push([pageIdx]);
      }
    }

    const checkpointData: Record<string, string> = {};
    if (checkpointPath) {
      try {
        const raw = await fs.readFile(checkpointPath, 'utf8');
        const parsed = JSON.parse(raw) as Record<string, string>;
        if (parsed && typeof parsed === 'object') {
          Object.assign(checkpointData, parsed);
        }
      } catch {
        // No existing checkpoint
      }
    }

    const tRender = Date.now();
    const allPagePaths = await renderPdfPagesToPngs(inputPath, tmpDir);
    if (allPagePaths.length === 0) {
      throw new Error('Aucune page PNG générée lors du rendu PDF.');
    }
    onProgress?.(
      `PNG render: ${allPagePaths.length} pages in ${Date.now() - tRender}ms (${selected.length} pages à OCR, ${Object.keys(checkpointData).length} lots en cache)`,
    );

    const pageTextByIndex = new Map<number, string>();
    let primaryModel = 'gemini-3.1-flash-lite';

    const saveCheckpoint = async () => {
      if (!checkpointPath) return;
      await fs.mkdir(path.dirname(checkpointPath), { recursive: true });
      await fs.writeFile(checkpointPath, JSON.stringify(checkpointData), 'utf8');
    };

    for (let bIdx = 0; bIdx < batches.length; bIdx++) {
      const pageIndices = batches[bIdx].filter((i) => i < allPagePaths.length);
      if (pageIndices.length === 0) continue;

      const batchKey = `${pageIndices[0]}-${pageIndices[pageIndices.length - 1]}`;
      if (checkpointData[batchKey] && countMeaningfulChars(checkpointData[batchKey]) >= 20) {
        pageTextByIndex.set(pageIndices[0], checkpointData[batchKey]);
        continue;
      }

      const imgBuffers = await Promise.all(
        pageIndices.map((idx) => fs.readFile(allPagePaths[idx])),
      );
      const tBatch = Date.now();

      try {
        const ocr = await performBatchImagesOcr(
          imgBuffers,
          'image/png',
          'gemini-3.1-flash-lite',
        );
        primaryModel = ocr.modelUsed;
        const text = ocr.text.trim();
        if (text && countMeaningfulChars(text) >= 40) {
          pageTextByIndex.set(pageIndices[0], text);
          checkpointData[batchKey] = text;
          await saveCheckpoint();
        } else if (pageIndices.length > 1) {
          const singleParts: string[] = [];
          for (let k = 0; k < pageIndices.length; k++) {
            const single = await performBatchImagesOcr(
              [imgBuffers[k]],
              'image/png',
              'gemini-3.1-flash-lite',
            );
            primaryModel = single.modelUsed;
            if (single.text.trim()) {
              pageTextByIndex.set(pageIndices[k], single.text.trim());
              singleParts.push(single.text.trim());
            }
          }
          if (singleParts.length > 0) {
            checkpointData[batchKey] = singleParts.join('\n\n');
            await saveCheckpoint();
          }
        }
      } catch (batchErr) {
        onProgress?.(
          `Batch ${bIdx + 1}/${batches.length} error (${batchErr instanceof Error ? batchErr.message : String(batchErr)}), retrying per-page`,
        );
        const singleParts: string[] = [];
        for (let k = 0; k < pageIndices.length; k++) {
          const single = await performBatchImagesOcr(
            [imgBuffers[k]],
            'image/png',
            'gemini-3.1-flash-lite',
          );
          primaryModel = single.modelUsed;
          if (single.text.trim()) {
            pageTextByIndex.set(pageIndices[k], single.text.trim());
            singleParts.push(single.text.trim());
          }
        }
        if (singleParts.length > 0) {
          checkpointData[batchKey] = singleParts.join('\n\n');
          await saveCheckpoint();
        }
      }

      onProgress?.(
        `Batch ${bIdx + 1}/${batches.length} done in ${Date.now() - tBatch}ms (model=${primaryModel})`,
      );
      if (bIdx + 1 < batches.length) {
        await new Promise((r) => setTimeout(r, 1500));
      }
    }

    if (checkpointPath) {
      await fs.rm(checkpointPath, { force: true }).catch(() => {});
    }

    return { pageTextByIndex, modelUsed: primaryModel };
  } finally {
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
  }
}

export async function extractDocumentWithOcr(
  buffer: Buffer,
  format: SupportedFormat,
  options: {
    fileName?: string;
    mimeType?: string;
    ocrMode?: OcrMode;
    onProgress?: (msg: string) => void;
    checkpointPath?: string;
  } = {},
): Promise<ExtractionResult> {
  const ocrMode = options.ocrMode || 'auto';

  switch (format) {
    case 'TXT': {
      const text = buffer.toString('utf8');
      return {
        text,
        ocrUsed: false,
        ocrReason: 'none',
        nativeCharCount: countMeaningfulChars(text),
      };
    }
    case 'HTML': {
      const html = buffer.toString('utf8');
      const text = html
        .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, ' ')
        .replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, ' ')
        .replace(/<\/(p|div|li|h[1-6]|tr|section|article|br)>/gi, '\n')
        .replace(/<[^>]+>/g, ' ')
        .replace(/&nbsp;/gi, ' ')
        .replace(/&amp;/gi, '&')
        .replace(/&lt;/gi, '<')
        .replace(/&gt;/gi, '>')
        .replace(/&quot;/gi, '"')
        .replace(/&#39;/gi, "'");
      return {
        text,
        ocrUsed: false,
        ocrReason: 'none',
        nativeCharCount: countMeaningfulChars(text),
      };
    }
    case 'DOCX': {
      const result = await mammoth.extractRawText({ buffer });
      const text = result.value || '';
      return {
        text,
        ocrUsed: false,
        ocrReason: 'none',
        nativeCharCount: countMeaningfulChars(text),
      };
    }
    case 'IMAGE': {
      if (ocrMode === 'off') {
        throw new Error(
          "L'extraction d'une image scannée nécessite l'activation de la reconnaissance OCR.",
        );
      }
      const mimeType = resolveOcrMimeType(options.fileName || 'scan.png', format, options.mimeType);
      const ocr = await performDocumentOcr(buffer, mimeType);
      return {
        text: ocr.text,
        ocrUsed: true,
        ocrReason: ocrMode === 'force' ? 'forced' : 'image_scan',
        ocrModel: ocr.modelUsed,
        nativeCharCount: 0,
        pageCount: 1,
      };
    }
    case 'PDF': {
      let nativeText = '';
      let pageTexts: string[] = [];
      let pageCount = 1;
      try {
        const uint8 = new Uint8Array(buffer);
        const result = (await extractPdfText(uint8, { mergePages: false })) as {
          text?: string | string[];
          totalPages?: number;
        };
        if (Array.isArray(result.text)) {
          pageTexts = result.text;
          nativeText = result.text.join('\n\n');
        } else if (typeof result.text === 'string') {
          pageTexts = [result.text];
          nativeText = result.text;
        }
        if (typeof result.totalPages === 'number' && result.totalPages > 0) {
          pageCount = result.totalPages;
        }
      } catch {
        nativeText = '';
      }

      const nativeChars = countMeaningfulChars(nativeText);
      const minExpectedChars = Math.max(40, pageCount * 35);
      const scannedPageIndices: number[] = [];
      for (let i = 0; i < pageCount; i++) {
        const pText = pageTexts[i] || '';
        if (countMeaningfulChars(pText) < 60) {
          scannedPageIndices.push(i);
        }
      }
      const hasSignificantScannedPages =
        pageCount > 1 && scannedPageIndices.length / Math.max(1, pageCount) > 0.1;
      const needsAutoOcr = nativeChars < minExpectedChars || hasSignificantScannedPages;

      if (ocrMode === 'force' || (ocrMode === 'auto' && needsAutoOcr)) {
        if (pageCount <= 4) {
          const ocr = await performDocumentOcr(buffer, 'application/pdf');
          const ocrChars = countMeaningfulChars(ocr.text);
          if (ocrChars >= nativeChars || ocrMode === 'force') {
            return {
              text: ocr.text,
              ocrUsed: true,
              ocrReason: ocrMode === 'force' ? 'forced' : 'pdf_scanned_auto',
              ocrModel: ocr.modelUsed,
              nativeCharCount: nativeChars,
              pageCount,
            };
          }
        } else {
          const targetIndices =
            ocrMode === 'force'
              ? Array.from({ length: pageCount }, (_, i) => i)
              : scannedPageIndices;

          try {
            const { pageTextByIndex, modelUsed } = await ocrSelectedPdfPagesInBatches(
              buffer,
              targetIndices,
              3,
              options.onProgress,
              options.checkpointPath,
            );

            const targetSet = new Set(targetIndices);
            const mergedPages: string[] = [];
            for (let i = 0; i < pageCount; i++) {
              const ocrBlock = pageTextByIndex.get(i);
              if (ocrBlock) {
                mergedPages.push(ocrBlock);
              } else if (!targetSet.has(i) && pageTexts[i]?.trim()) {
                mergedPages.push(pageTexts[i].trim());
              }
            }
            const mergedText = mergedPages.join('\n\n');
            return {
              text: mergedText,
              ocrUsed: true,
              ocrReason: ocrMode === 'force' ? 'forced' : 'pdf_scanned_auto',
              ocrModel: modelUsed,
              nativeCharCount: nativeChars,
              pageCount,
            };
          } catch (err) {
            const code = (err as NodeJS.ErrnoException | undefined)?.code;
            if (code === 'ENOENT') {
              const ocr = await performDocumentOcr(buffer, 'application/pdf');
              return {
                text: ocr.text,
                ocrUsed: true,
                ocrReason: ocrMode === 'force' ? 'forced' : 'pdf_scanned_auto',
                ocrModel: ocr.modelUsed,
                nativeCharCount: nativeChars,
                pageCount,
              };
            }
            throw err;
          }
        }
      }

      return {
        text: nativeText,
        ocrUsed: false,
        ocrReason: 'none',
        nativeCharCount: nativeChars,
        pageCount,
      };
    }
  }
}

export async function extractDocumentContent(
  buffer: Buffer,
  format: SupportedFormat,
): Promise<string> {
  const res = await extractDocumentWithOcr(buffer, format, { ocrMode: 'auto' });
  return res.text;
}
