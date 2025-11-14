import { pipeline, TextGenerationPipeline, env } from '@xenova/transformers';

let generator: TextGenerationPipeline | null = null;
let currentModelId: string | null = null;

env.allowLocalModels = true;
env.allowRemoteModels = true;
env.allowUsageAnalytics = false;

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
    let lastProgress = 0;

    generator = await pipeline('text-generation', modelId, {
      progress_callback: (progress: any) => {
        try {
          if (progress.status === 'downloading' && typeof progress.progress === 'number') {
            lastProgress = Math.min(99, Math.max(lastProgress, progress.progress * 100));
            if (onProgress) {
              onProgress(lastProgress);
            }
          } else if (progress.status === 'progress' && typeof progress.progress === 'number') {
            lastProgress = Math.min(99, Math.max(lastProgress, progress.progress * 100));
            if (onProgress) {
              onProgress(lastProgress);
            }
          } else if (progress.status === 'loading' && onProgress) {
            onProgress(90);
          }
        } catch (e) {
          console.debug('Progress callback error:', e);
        }
      }
    });

    if (onProgress) {
      onProgress(100);
    }
    currentModelId = modelId;
  } catch (error) {
    console.error('Failed to load model:', error);
    let errorMessage = 'Failed to download model';

    if (error instanceof Error) {
      errorMessage = error.message;
      if (error.message.includes('JSON')) {
        errorMessage = 'Network error downloading model. Please check your internet connection and try again.';
      } else if (error.message.includes('404')) {
        errorMessage = 'Model not found. Please try a different model.';
      }
    }

    throw new Error(errorMessage);
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
      return_full_text: false
    });

    if (Array.isArray(result) && result.length > 0) {
      const text = result[0]?.generated_text || '';
      return text;
    }

    return '';
  } catch (error) {
    console.error('Generation error:', error);
    throw new Error(`Generation failed: ${error instanceof Error ? error.message : String(error)}`);
  }
};

export const isModelLoaded = (): boolean => {
  return generator !== null;
};

export const getCurrentModelId = (): string | null => {
  return currentModelId;
};
