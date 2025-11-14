import { useState } from 'react';
import { Download } from 'lucide-react';

interface Model {
  id: string;
  name: string;
  size: string;
  description: string;
}

const AVAILABLE_MODELS: Model[] = [
  {
    id: 'Xenova/gpt2',
    name: 'GPT-2',
    size: '~500MB',
    description: 'Fast and lightweight, great for quick responses'
  },
  {
    id: 'Xenova/TinyLlama-1.1B-Chat-v1.0',
    name: 'TinyLlama 1.1B',
    size: '~650MB',
    description: 'Optimized for chat, balanced performance'
  },
  {
    id: 'Xenova/phi-1_5',
    name: 'Phi-1.5',
    size: '~800MB',
    description: 'Higher quality responses, slower generation'
  }
];

interface ModelSelectorProps {
  onModelSelect: (modelId: string) => void;
  isDownloading: boolean;
  downloadProgress: number;
  error?: string;
}

export default function ModelSelector({ onModelSelect, isDownloading, downloadProgress, error }: ModelSelectorProps) {
  const [selectedModel, setSelectedModel] = useState<string>('');

  const handleDownload = () => {
    if (selectedModel) {
      onModelSelect(selectedModel);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full bg-slate-800/50 backdrop-blur-xl rounded-2xl shadow-2xl p-8 border border-slate-700">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">ZiroPilot</h1>
          <p className="text-slate-300">Choose your AI model to get started</p>
          <p className="text-sm text-slate-400 mt-2">Download once, chat offline forever</p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-600/20 border border-red-600/50 rounded-lg">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

        {!isDownloading ? (
          <>
            <div className="space-y-3 mb-6">
              {AVAILABLE_MODELS.map((model) => (
                <button
                  key={model.id}
                  onClick={() => setSelectedModel(model.id)}
                  className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                    selectedModel === model.id
                      ? 'border-blue-500 bg-blue-500/10'
                      : 'border-slate-600 bg-slate-700/30 hover:border-slate-500'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="text-lg font-semibold text-white">{model.name}</h3>
                      <p className="text-sm text-slate-300 mt-1">{model.description}</p>
                    </div>
                    <span className="text-xs text-slate-400 bg-slate-600/50 px-2 py-1 rounded">
                      {model.size}
                    </span>
                  </div>
                </button>
              ))}
            </div>

            <button
              onClick={handleDownload}
              disabled={!selectedModel}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <Download size={20} />
              Download & Install Model
            </button>
          </>
        ) : (
          <div className="space-y-4">
            <div className="text-center">
              <p className="text-white font-medium mb-4">Downloading model...</p>
              <div className="w-full bg-slate-700 rounded-full h-3 overflow-hidden">
                <div
                  className="bg-blue-500 h-full transition-all duration-300 rounded-full"
                  style={{ width: `${downloadProgress}%` }}
                />
              </div>
              <p className="text-slate-400 text-sm mt-2">{Math.round(downloadProgress)}% complete</p>
            </div>
            <p className="text-xs text-slate-400 text-center">
              This may take a few minutes. The model will be cached for offline use.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
