import { X, Trash2, Download } from 'lucide-react';
import { AppSettings } from '../utils/storage';

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
  settings: AppSettings;
  onSettingsChange: (settings: AppSettings) => void;
  onClearModel: () => void;
  onClearChat: () => void;
  currentModel: string | null;
}

export default function SettingsPanel({
  isOpen,
  onClose,
  settings,
  onSettingsChange,
  onClearModel,
  onClearChat,
  currentModel
}: SettingsPanelProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-800 rounded-2xl shadow-2xl w-full max-w-md border border-slate-700 max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 bg-slate-800 border-b border-slate-700 px-6 py-4 flex items-center justify-between">
          <h2 className="text-xl font-bold text-white">Settings</h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            aria-label="Close settings"
          >
            <X size={20} className="text-slate-300" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          <div>
            <h3 className="text-sm font-semibold text-slate-300 mb-3">Model Information</h3>
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
              <div className="flex items-center gap-2 mb-2">
                <Download size={16} className="text-blue-400" />
                <span className="text-white font-medium">Current Model</span>
              </div>
              <p className="text-sm text-slate-400">
                {currentModel || 'No model loaded'}
              </p>
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-slate-300 mb-3">Generation Settings</h3>
            <div className="space-y-4">
              <div>
                <label className="flex items-center justify-between text-sm text-slate-300 mb-2">
                  Temperature
                  <span className="text-blue-400">{settings.temperature.toFixed(1)}</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={settings.temperature}
                  onChange={(e) => onSettingsChange({ ...settings, temperature: parseFloat(e.target.value) })}
                  className="w-full accent-blue-500"
                />
                <p className="text-xs text-slate-500 mt-1">
                  Lower = more focused, Higher = more creative
                </p>
              </div>

              <div>
                <label className="flex items-center justify-between text-sm text-slate-300 mb-2">
                  Max Tokens
                  <span className="text-blue-400">{settings.maxTokens}</span>
                </label>
                <input
                  type="range"
                  min="50"
                  max="512"
                  step="50"
                  value={settings.maxTokens}
                  onChange={(e) => onSettingsChange({ ...settings, maxTokens: parseInt(e.target.value) })}
                  className="w-full accent-blue-500"
                />
                <p className="text-xs text-slate-500 mt-1">
                  Maximum length of generated responses
                </p>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-slate-300 mb-3">Data Management</h3>
            <div className="space-y-2">
              <button
                onClick={onClearChat}
                className="w-full flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-white px-4 py-3 rounded-lg transition-colors"
              >
                <Trash2 size={16} />
                Clear Chat History
              </button>

              <button
                onClick={onClearModel}
                className="w-full flex items-center gap-2 bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-600/50 px-4 py-3 rounded-lg transition-colors"
              >
                <Trash2 size={16} />
                Delete Model & Reset
              </button>
            </div>
          </div>

          <div className="pt-4 border-t border-slate-700">
            <p className="text-xs text-slate-500 text-center">
              ZiroPilot v1.0 — All processing happens on your device
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
