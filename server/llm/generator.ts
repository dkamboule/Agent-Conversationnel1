import { ThinkingLevel } from '@google/genai';
import { RAG_CONFIG } from '../core/config.ts';
import {
  getGeminiClient,
  isTransientGeminiError,
  sleep,
  withGeminiRetry,
} from '../core/geminiClient.ts';
import { DGECC_SYSTEM_INSTRUCTION } from '../rag/context_builder.ts';

export interface LlmProvider {
  modelName: string;
  generate(prompt: string, systemInstruction?: string): Promise<string>;
  generateStream(
    prompt: string,
    onChunk: (delta: string) => void,
    systemInstruction?: string,
  ): Promise<string>;
}

function getThinkingLevelForModel(modelName: string): ThinkingLevel {
  return modelName === 'gemini-3.1-flash-lite'
    ? ThinkingLevel.MINIMAL
    : ThinkingLevel.LOW;
}

export class GeminiLlmProvider implements LlmProvider {
  public readonly modelName: string;
  private readonly fallbackModelName: string;

  constructor(modelName = RAG_CONFIG.llmModel, fallbackModelName = RAG_CONFIG.evalModel) {
    this.modelName = modelName;
    this.fallbackModelName = fallbackModelName;
  }

  private async generateWithModel(
    modelName: string,
    prompt: string,
    systemInstruction: string,
  ): Promise<string> {
    const ai = getGeminiClient();
    const response = await ai.models.generateContent({
      model: modelName,
      contents: prompt,
      config: {
        systemInstruction,
        thinkingConfig: { thinkingLevel: getThinkingLevelForModel(modelName) },
      },
    });
    return (response.text || '').trim();
  }

  async generate(prompt: string, systemInstruction = DGECC_SYSTEM_INSTRUCTION): Promise<string> {
    try {
      return await withGeminiRetry(
        () => this.generateWithModel(this.modelName, prompt, systemInstruction),
        { maxRetries: 2, baseDelayMs: 350 },
      );
    } catch (err) {
      if (isTransientGeminiError(err) && this.fallbackModelName && this.fallbackModelName !== this.modelName) {
        return await withGeminiRetry(
          () => this.generateWithModel(this.fallbackModelName, prompt, systemInstruction),
          { maxRetries: 1, baseDelayMs: 400 },
        );
      }
      throw err;
    }
  }

  private async streamWithModel(
    modelName: string,
    prompt: string,
    onChunk: (delta: string) => void,
    systemInstruction: string,
  ): Promise<{ text: string; emitted: boolean }> {
    const ai = getGeminiClient();
    const stream = await ai.models.generateContentStream({
      model: modelName,
      contents: prompt,
      config: {
        systemInstruction,
        thinkingConfig: { thinkingLevel: getThinkingLevelForModel(modelName) },
      },
    });

    let fullText = '';
    let emitted = false;
    try {
      for await (const chunk of stream) {
        const piece = chunk.text || '';
        if (piece) {
          emitted = true;
          fullText += piece;
          onChunk(piece);
        }
      }
      return { text: fullText.trim(), emitted };
    } catch (err) {
      if (emitted) {
        // Attach marker so caller never duplicates partial output already streamed
        (err as { __streamAlreadyEmitted?: boolean }).__streamAlreadyEmitted = true;
      }
      throw err;
    }
  }

  async generateStream(
    prompt: string,
    onChunk: (delta: string) => void,
    systemInstruction = DGECC_SYSTEM_INSTRUCTION,
  ): Promise<string> {
    const attemptModels =
      this.fallbackModelName && this.fallbackModelName !== this.modelName
        ? [
            { model: this.modelName, retries: 2 },
            { model: this.fallbackModelName, retries: 1 },
          ]
        : [{ model: this.modelName, retries: 2 }];

    let lastError: unknown;

    for (const entry of attemptModels) {
      for (let attempt = 0; attempt <= entry.retries; attempt++) {
        try {
          const result = await this.streamWithModel(entry.model, prompt, onChunk, systemInstruction);
          return result.text;
        } catch (err) {
          lastError = err;
          if ((err as { __streamAlreadyEmitted?: boolean })?.__streamAlreadyEmitted) {
            throw err;
          }
          if (!isTransientGeminiError(err)) {
            throw err;
          }
          if (attempt < entry.retries) {
            const delayMs = 350 * Math.pow(2, attempt) + Math.floor(Math.random() * 120);
            await sleep(delayMs);
          }
        }
      }
    }

    throw lastError;
  }
}

export function getLlmProvider(): LlmProvider {
  return new GeminiLlmProvider(RAG_CONFIG.llmModel, RAG_CONFIG.evalModel);
}
