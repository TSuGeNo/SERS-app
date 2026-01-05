import { create } from 'zustand';
import { v4 as uuidv4 } from 'uuid';

export interface Message {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
    attachments?: Attachment[];
    visualizations?: Visualization[];
    isLoading?: boolean;
}

export interface Attachment {
    id: string;
    name: string;
    type: string;
    size: number;
    url?: string;
    data?: unknown;
}

export interface Visualization {
    id: string;
    type: 'spectrum' | 'pca' | 'confusion_matrix' | 'histogram' | 'heatmap' | 'line' | 'scatter';
    title: string;
    data: unknown;
    config?: unknown;
}

export interface Session {
    id: string;
    title: string;
    messages: Message[];
    attachments: Attachment[];
    createdAt: Date;
    updatedAt: Date;
    metadata?: {
        taskType?: string;
        framework?: string;
        substrate?: string;
    };
}

interface ChatState {
    sessions: Session[];
    currentSessionId: string | null;
    isProcessing: boolean;

    // Actions
    createSession: () => string;
    setCurrentSession: (sessionId: string) => void;
    getCurrentSession: () => Session | null;
    addMessage: (sessionId: string, message: Omit<Message, 'id' | 'timestamp'>) => string;
    updateMessage: (sessionId: string, messageId: string, updates: Partial<Message>) => void;
    addAttachment: (sessionId: string, attachment: Omit<Attachment, 'id'>) => string;
    removeAttachment: (sessionId: string, attachmentId: string) => void;
    updateSessionMetadata: (sessionId: string, metadata: Session['metadata']) => void;
    deleteSession: (sessionId: string) => void;
    setIsProcessing: (isProcessing: boolean) => void;
    clearSessions: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
    sessions: [],
    currentSessionId: null,
    isProcessing: false,

    createSession: () => {
        const newSession: Session = {
            id: uuidv4(),
            title: 'New Analysis',
            messages: [],
            attachments: [],
            createdAt: new Date(),
            updatedAt: new Date(),
        };

        set((state) => ({
            sessions: [newSession, ...state.sessions],
            currentSessionId: newSession.id,
        }));

        return newSession.id;
    },

    setCurrentSession: (sessionId: string) => {
        set({ currentSessionId: sessionId });
    },

    getCurrentSession: () => {
        const state = get();
        return state.sessions.find((s) => s.id === state.currentSessionId) || null;
    },

    addMessage: (sessionId: string, message: Omit<Message, 'id' | 'timestamp'>) => {
        const messageId = uuidv4();
        const newMessage: Message = {
            ...message,
            id: messageId,
            timestamp: new Date(),
        };

        set((state) => ({
            sessions: state.sessions.map((session) =>
                session.id === sessionId
                    ? {
                        ...session,
                        messages: [...session.messages, newMessage],
                        updatedAt: new Date(),
                        title: session.messages.length === 0 && message.role === 'user'
                            ? message.content.slice(0, 50) + (message.content.length > 50 ? '...' : '')
                            : session.title,
                    }
                    : session
            ),
        }));

        return messageId;
    },

    updateMessage: (sessionId: string, messageId: string, updates: Partial<Message>) => {
        set((state) => ({
            sessions: state.sessions.map((session) =>
                session.id === sessionId
                    ? {
                        ...session,
                        messages: session.messages.map((msg) =>
                            msg.id === messageId ? { ...msg, ...updates } : msg
                        ),
                        updatedAt: new Date(),
                    }
                    : session
            ),
        }));
    },

    addAttachment: (sessionId: string, attachment: Omit<Attachment, 'id'>) => {
        const attachmentId = uuidv4();
        const newAttachment: Attachment = {
            ...attachment,
            id: attachmentId,
        };

        set((state) => ({
            sessions: state.sessions.map((session) =>
                session.id === sessionId
                    ? {
                        ...session,
                        attachments: [...session.attachments, newAttachment],
                        updatedAt: new Date(),
                    }
                    : session
            ),
        }));

        return attachmentId;
    },

    removeAttachment: (sessionId: string, attachmentId: string) => {
        set((state) => ({
            sessions: state.sessions.map((session) =>
                session.id === sessionId
                    ? {
                        ...session,
                        attachments: session.attachments.filter((a) => a.id !== attachmentId),
                        updatedAt: new Date(),
                    }
                    : session
            ),
        }));
    },

    updateSessionMetadata: (sessionId: string, metadata: Session['metadata']) => {
        set((state) => ({
            sessions: state.sessions.map((session) =>
                session.id === sessionId
                    ? {
                        ...session,
                        metadata: { ...session.metadata, ...metadata },
                        updatedAt: new Date(),
                    }
                    : session
            ),
        }));
    },

    deleteSession: (sessionId: string) => {
        set((state) => {
            const newSessions = state.sessions.filter((s) => s.id !== sessionId);
            return {
                sessions: newSessions,
                currentSessionId: state.currentSessionId === sessionId
                    ? newSessions[0]?.id || null
                    : state.currentSessionId,
            };
        });
    },

    setIsProcessing: (isProcessing: boolean) => {
        set({ isProcessing });
    },

    clearSessions: () => {
        set({ sessions: [], currentSessionId: null });
    },
}));
