import { pipeline, TextGenerationPipeline } from '@xenova/transformers';

let generator: TextGenerationPipeline | null = null;
let currentModelId: string | null = null;

export interface GenerateOptions {
  temperature?: number;
  max_new_tokens?: number;
  do_sample?: boolean;
}

export const loadModel = async (
  modelId: string,
  onProgress?: (progress: number) => void
): Promise<void> => {
  if (currentModelId === modelId && generator) {
    return;
  }

  try {
    generator = await pipeline('text-generation', modelId, {
      progress_callback: (progress: { status: string; progress?: number; file?: string }) => {
        if (progress.progress !== undefined && onProgress) {
          onProgress(progress.progress);
        }
      }
    });
    currentModelId = modelId;
  } catch (error) {
    console.error('Failed to load model:', error);
    throw error;
  }
};

export const generateText = async (
  prompt: string,
  options: GenerateOptions = {},
  onToken?: (token: string) => void
): Promise<string> => {
  if (!generator) {
    throw new Error('Model not loaded');
  }

  try {
    const result = await generator(prompt, {
      temperature: options.temperature ?? 0.7,
      max_new_tokens: options.max_new_tokens ?? 256,
      do_sample: options.do_sample ?? true,
      return_full_text: false,
      callback_function: onToken ? (output: { token: { text: string } }[]) => {
        if (output && output.length > 0 && output[0].token) {
          onToken(output[0].token.text);
        }
      } : undefined
    });

    if (Array.isArray(result) && result[0]?.generated_text) {
      return result[0].generated_text;
    }

    return '';
  } catch (error) {
    console.error('Generation error:', error);
    throw error;
  }
};

export const isModelLoaded = (): boolean => {
  return generator !== null;
};

export const getCurrentModelId = (): string | null => {
  return currentModelId;
};
