'use client';

import React from 'react';

interface ModelIconProps {
    className?: string;
    size?: number;
}

// Meta/Llama AI Icon - A stylized llama silhouette
export const LlamaIcon: React.FC<ModelIconProps> = ({ className = '', size = 24 }) => (
    <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className={className}
    >
        <defs>
            <linearGradient id="llamaGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#0668E1" />
                <stop offset="100%" stopColor="#1877F2" />
            </linearGradient>
        </defs>
        <circle cx="12" cy="12" r="11" fill="url(#llamaGrad)" />
        <path
            d="M8 17V10C8 8.5 9 7 11 7C11.5 7 12 7.2 12.3 7.5C12.6 7.2 13.1 7 13.5 7C15.5 7 16.5 8.5 16.5 10V17"
            stroke="white"
            strokeWidth="1.5"
            strokeLinecap="round"
            fill="none"
        />
        <circle cx="10" cy="9.5" r="0.8" fill="white" />
        <circle cx="14.5" cy="9.5" r="0.8" fill="white" />
        <path
            d="M9 6.5C9 5.5 9.5 5 10 5L11 6"
            stroke="white"
            strokeWidth="1"
            strokeLinecap="round"
            fill="none"
        />
        <path
            d="M15 6.5C15 5.5 14.5 5 14 5L13 6"
            stroke="white"
            strokeWidth="1"
            strokeLinecap="round"
            fill="none"
        />
    </svg>
);

// Qwen/Alibaba AI Icon - A stylized brain/neural network
export const QwenIcon: React.FC<ModelIconProps> = ({ className = '', size = 24 }) => (
    <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className={className}
    >
        <defs>
            <linearGradient id="qwenGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#6366f1" />
                <stop offset="100%" stopColor="#8b5cf6" />
            </linearGradient>
        </defs>
        <circle cx="12" cy="12" r="11" fill="url(#qwenGrad)" />
        {/* Brain shape */}
        <path
            d="M12 6C9.5 6 7.5 7.5 7.5 10C7.5 11 8 11.8 8.5 12.5C8 13.2 7.5 14 7.5 15C7.5 17 9 18 11 18H13C15 18 16.5 17 16.5 15C16.5 14 16 13.2 15.5 12.5C16 11.8 16.5 11 16.5 10C16.5 7.5 14.5 6 12 6Z"
            stroke="white"
            strokeWidth="1.2"
            fill="none"
        />
        {/* Neural connections */}
        <circle cx="10" cy="10" r="1" fill="white" />
        <circle cx="14" cy="10" r="1" fill="white" />
        <circle cx="10" cy="14.5" r="1" fill="white" />
        <circle cx="14" cy="14.5" r="1" fill="white" />
        <circle cx="12" cy="12.2" r="0.8" fill="white" />
        <line x1="10.5" y1="10.5" x2="11.5" y2="11.8" stroke="white" strokeWidth="0.5" />
        <line x1="13.5" y1="10.5" x2="12.5" y2="11.8" stroke="white" strokeWidth="0.5" />
        <line x1="10.5" y1="14" x2="11.5" y2="12.8" stroke="white" strokeWidth="0.5" />
        <line x1="13.5" y1="14" x2="12.5" y2="12.8" stroke="white" strokeWidth="0.5" />
    </svg>
);

// DeepSeek AI Icon - A stylized deep neural network / sea creature
export const DeepSeekIcon: React.FC<ModelIconProps> = ({ className = '', size = 24 }) => (
    <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className={className}
    >
        <defs>
            <linearGradient id="deepseekGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#4F46E5" />
                <stop offset="100%" stopColor="#7C3AED" />
            </linearGradient>
        </defs>
        <circle cx="12" cy="12" r="11" fill="url(#deepseekGrad)" />
        {/* Deep layers representation */}
        <path
            d="M6 8h12M6 12h12M6 16h12"
            stroke="white"
            strokeWidth="1.2"
            strokeLinecap="round"
            opacity="0.5"
        />
        {/* Neural nodes */}
        <circle cx="8" cy="8" r="1.5" fill="white" />
        <circle cx="12" cy="8" r="1.5" fill="white" />
        <circle cx="16" cy="8" r="1.5" fill="white" />
        <circle cx="10" cy="12" r="1.5" fill="white" />
        <circle cx="14" cy="12" r="1.5" fill="white" />
        <circle cx="12" cy="16" r="1.5" fill="white" />
        {/* Connections */}
        <line x1="8" y1="9.5" x2="10" y2="10.5" stroke="white" strokeWidth="0.8" />
        <line x1="12" y1="9.5" x2="10" y2="10.5" stroke="white" strokeWidth="0.8" />
        <line x1="12" y1="9.5" x2="14" y2="10.5" stroke="white" strokeWidth="0.8" />
        <line x1="16" y1="9.5" x2="14" y2="10.5" stroke="white" strokeWidth="0.8" />
        <line x1="10" y1="13.5" x2="12" y2="14.5" stroke="white" strokeWidth="0.8" />
        <line x1="14" y1="13.5" x2="12" y2="14.5" stroke="white" strokeWidth="0.8" />
    </svg>
);

// Google Gemini Icon - A stylized sparkle/star pattern
export const GeminiIcon: React.FC<ModelIconProps> = ({ className = '', size = 24 }) => (
    <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className={className}
    >
        <defs>
            <linearGradient id="geminiGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#4285f4" />
                <stop offset="25%" stopColor="#9b72cb" />
                <stop offset="50%" stopColor="#d96570" />
                <stop offset="75%" stopColor="#d96570" />
                <stop offset="100%" stopColor="#9b72cb" />
            </linearGradient>
        </defs>
        <circle cx="12" cy="12" r="11" fill="url(#geminiGrad)" />
        {/* Gemini sparkle pattern */}
        <path
            d="M12 5L12.8 9.5L17 10L12.8 10.5L12 15L11.2 10.5L7 10L11.2 9.5L12 5Z"
            fill="white"
        />
        <path
            d="M16 13L16.5 15L18.5 15.3L16.5 15.7L16 18L15.5 15.7L13.5 15.3L15.5 15L16 13Z"
            fill="white"
            opacity="0.8"
        />
        <path
            d="M8 13L8.4 14.5L10 14.8L8.4 15.2L8 17L7.6 15.2L6 14.8L7.6 14.5L8 13Z"
            fill="white"
            opacity="0.8"
        />
    </svg>
);

// Generic AI Icon for fallback
export const AIIcon: React.FC<ModelIconProps> = ({ className = '', size = 24 }) => (
    <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className={className}
    >
        <defs>
            <linearGradient id="aiGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#6366f1" />
                <stop offset="100%" stopColor="#8b5cf6" />
            </linearGradient>
        </defs>
        <circle cx="12" cy="12" r="11" fill="url(#aiGrad)" />
        <path
            d="M8 16L12 7L16 16"
            stroke="white"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
        />
        <path
            d="M9.5 13H14.5"
            stroke="white"
            strokeWidth="1.5"
            strokeLinecap="round"
        />
    </svg>
);

// Map of model keys to their icons
export const ModelIcons = {
    chatgpt: DeepSeekIcon,  // DeepSeek V3
    claude: DeepSeekIcon,   // DeepSeek R1
    gemini: GeminiIcon,     // Google Gemini
} as const;

// Get icon component for a model
export const getModelIcon = (modelKey: string): React.FC<ModelIconProps> => {
    return ModelIcons[modelKey as keyof typeof ModelIcons] || AIIcon;
};
