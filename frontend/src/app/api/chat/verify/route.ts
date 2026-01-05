import { NextResponse } from 'next/server';

export async function GET() {
    // Always show all models as available - they work with hardcoded API key
    return NextResponse.json({
        api_key_valid: true,
        available_models: 3,
        total_models: 3,
        models: [
            {
                id: 'gemini',
                name: 'Gemini 2.0 Flash',
                provider: 'Google',
                status: 'available',
                description: 'Fast multimodal analysis',
            },
            {
                id: 'chatgpt',
                name: 'DeepSeek V3',
                provider: 'DeepSeek',
                status: 'available',
                description: 'Best for coding & reasoning',
            },
            {
                id: 'claude',
                name: 'DeepSeek R1',
                provider: 'DeepSeek',
                status: 'available',
                description: 'Advanced reasoning model',
            },
        ],
        message: 'All models available via OpenRouter',
    });
}
