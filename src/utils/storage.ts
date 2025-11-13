import { Message } from '../components/ChatInterface';

const DB_NAME = 'ziropilot-db';
const DB_VERSION = 1;
const MODEL_STORE = 'models';
const CHAT_STORE = 'chats';

export interface StoredModel {
  id: string;
  name: string;
  timestamp: number;
}

let db: IDBDatabase | null = null;

export const initDB = (): Promise<IDBDatabase> => {
  return new Promise((resolve, reject) => {
    if (db) {
      resolve(db);
      return;
    }

    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      db = request.result;
      resolve(db);
    };

    request.onupgradeneeded = (event) => {
      const database = (event.target as IDBOpenDBRequest).result;

      if (!database.objectStoreNames.contains(MODEL_STORE)) {
        database.createObjectStore(MODEL_STORE, { keyPath: 'id' });
      }

      if (!database.objectStoreNames.contains(CHAT_STORE)) {
        database.createObjectStore(CHAT_STORE, { keyPath: 'id' });
      }
    };
  });
};

export const saveModel = async (modelId: string, modelName: string): Promise<void> => {
  const database = await initDB();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction([MODEL_STORE], 'readwrite');
    const store = transaction.objectStore(MODEL_STORE);
    const model: StoredModel = {
      id: modelId,
      name: modelName,
      timestamp: Date.now()
    };
    const request = store.put(model);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
};

export const getStoredModel = async (): Promise<StoredModel | null> => {
  const database = await initDB();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction([MODEL_STORE], 'readonly');
    const store = transaction.objectStore(MODEL_STORE);
    const request = store.getAll();

    request.onsuccess = () => {
      const models = request.result;
      resolve(models.length > 0 ? models[0] : null);
    };
    request.onerror = () => reject(request.error);
  });
};

export const clearModel = async (): Promise<void> => {
  const database = await initDB();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction([MODEL_STORE], 'readwrite');
    const store = transaction.objectStore(MODEL_STORE);
    const request = store.clear();

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
};

export const saveMessages = (messages: Message[]): void => {
  localStorage.setItem('ziropilot-messages', JSON.stringify(messages));
};

export const getMessages = (): Message[] => {
  const stored = localStorage.getItem('ziropilot-messages');
  return stored ? JSON.parse(stored) : [];
};

export const clearMessages = (): void => {
  localStorage.removeItem('ziropilot-messages');
};

export interface AppSettings {
  temperature: number;
  maxTokens: number;
  darkMode: boolean;
}

const DEFAULT_SETTINGS: AppSettings = {
  temperature: 0.7,
  maxTokens: 256,
  darkMode: true
};

export const saveSettings = (settings: AppSettings): void => {
  localStorage.setItem('ziropilot-settings', JSON.stringify(settings));
};

export const getSettings = (): AppSettings => {
  const stored = localStorage.getItem('ziropilot-settings');
  return stored ? { ...DEFAULT_SETTINGS, ...JSON.parse(stored) } : DEFAULT_SETTINGS;
};
