import { useState, useRef, useEffect } from 'react';
import { Send, Settings, Loader2 } from 'lucide-react';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

interface ChatInterfaceProps {
  messages: Message[];
  onSendMessage: (message: string) => void;
  onOpenSettings: () => void;
  isGenerating: boolean;
}

export default function ChatInterface({ messages, onSendMessage, onOpenSettings, isGenerating }: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isGenerating) {
      onSendMessage(input.trim());
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
  };

  return (
    <div className="flex flex-col h-screen bg-slate-900">
      <header className="bg-slate-800 border-b border-slate-700 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">ZiroPilot</h1>
          <p className="text-xs text-slate-400">Offline AI Chat</p>
        </div>
        <button
          onClick={onOpenSettings}
          className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          aria-label="Settings"
        >
          <Settings size={20} className="text-slate-300" />
        </button>
      </header>

      <div className="flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md">
              <h2 className="text-2xl font-semibold text-white mb-2">Start chatting!</h2>
              <p className="text-slate-400">
                Your AI runs completely offline. No data leaves your device.
              </p>
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-800 text-slate-100 border border-slate-700'
                  }`}
                >
                  <p className="whitespace-pre-wrap break-words">{message.content}</p>
                </div>
              </div>
            ))}
            {isGenerating && (
              <div className="flex justify-start">
                <div className="bg-slate-800 border border-slate-700 rounded-2xl px-4 py-3 flex items-center gap-2">
                  <Loader2 size={16} className="text-blue-400 animate-spin" />
                  <span className="text-slate-400 text-sm">Generating...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className="border-t border-slate-700 bg-slate-800 px-4 py-4">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="flex gap-2 items-end">
            <div className="flex-1 bg-slate-900 rounded-lg border border-slate-600 focus-within:border-blue-500 transition-colors">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={handleInput}
                onKeyDown={handleKeyDown}
                placeholder="Type your message..."
                className="w-full bg-transparent text-white placeholder-slate-500 px-4 py-3 resize-none outline-none"
                rows={1}
                disabled={isGenerating}
              />
            </div>
            <button
              type="submit"
              disabled={!input.trim() || isGenerating}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white p-3 rounded-lg transition-colors"
              aria-label="Send message"
            >
              <Send size={20} />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
