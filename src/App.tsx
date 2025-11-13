import { useState, useEffect } from 'react';
import ModelSelector from './components/ModelSelector';
import ChatInterface, { Message } from './components/ChatInterface';
import SettingsPanel from './components/SettingsPanel';
import {
  getStoredModel,
  saveModel,
  clearModel,
  getMessages,
  saveMessages,
  clearMessages,
  getSettings,
  saveSettings,
  AppSettings
} from './utils/storage';
import { loadModel, generateText } from './utils/ai';

type AppState = 'loading' | 'model-selection' | 'downloading' | 'chat';

function App() {
  const [appState, setAppState] = useState<AppState>('loading');
  const [currentModel, setCurrentModel] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<AppSettings>(getSettings());

  useEffect(() => {
    const initApp = async () => {
      const storedModel = await getStoredModel();
      if (storedModel) {
        setCurrentModel(storedModel.id);
        setAppState('downloading');
        try {
          await loadModel(storedModel.id, (progress) => {
            setDownloadProgress(progress * 100);
          });
          setDownloadProgress(100);
          const storedMessages = getMessages();
          setMessages(storedMessages);
          setAppState('chat');
        } catch (error) {
          console.error('Failed to load model:', error);
          setAppState('model-selection');
        }
      } else {
        setAppState('model-selection');
      }
    };

    initApp();
  }, []);

  const handleModelSelect = async (modelId: string) => {
    setCurrentModel(modelId);
    setAppState('downloading');
    setDownloadProgress(0);

    try {
      await loadModel(modelId, (progress) => {
        setDownloadProgress(progress * 100);
      });
      await saveModel(modelId, modelId);
      setDownloadProgress(100);
      setAppState('chat');
    } catch (error) {
      console.error('Failed to download model:', error);
      alert('Failed to download model. Please try again.');
      setAppState('model-selection');
    }
  };

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: Date.now()
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    saveMessages(newMessages);
    setIsGenerating(true);

    try {
      const prompt = newMessages
        .map((msg) => `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}`)
        .join('\n') + '\nAssistant:';

      const response = await generateText(prompt, {
        temperature: settings.temperature,
        max_new_tokens: settings.maxTokens,
        do_sample: true
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.trim(),
        timestamp: Date.now()
      };

      const updatedMessages = [...newMessages, assistantMessage];
      setMessages(updatedMessages);
      saveMessages(updatedMessages);
    } catch (error) {
      console.error('Generation failed:', error);
      alert('Failed to generate response. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleClearModel = async () => {
    if (confirm('This will delete the model and reset the app. Continue?')) {
      await clearModel();
      clearMessages();
      setMessages([]);
      setCurrentModel(null);
      setAppState('model-selection');
      setSettingsOpen(false);
    }
  };

  const handleClearChat = () => {
    if (confirm('Clear all chat history?')) {
      clearMessages();
      setMessages([]);
      setSettingsOpen(false);
    }
  };

  const handleSettingsChange = (newSettings: AppSettings) => {
    setSettings(newSettings);
    saveSettings(newSettings);
  };

  if (appState === 'loading') {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-slate-400">Loading ZiroPilot...</p>
        </div>
      </div>
    );
  }

  if (appState === 'model-selection' || appState === 'downloading') {
    return (
      <ModelSelector
        onModelSelect={handleModelSelect}
        isDownloading={appState === 'downloading'}
        downloadProgress={downloadProgress}
      />
    );
  }

  return (
    <>
      <ChatInterface
        messages={messages}
        onSendMessage={handleSendMessage}
        onOpenSettings={() => setSettingsOpen(true)}
        isGenerating={isGenerating}
      />
      <SettingsPanel
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onSettingsChange={handleSettingsChange}
        onClearModel={handleClearModel}
        onClearChat={handleClearChat}
        currentModel={currentModel}
      />
    </>
  );
}

export default App;
