import { NextResponse } from 'next/server';

export async function GET() {
    return NextResponse.json({
        api_key_valid: true,
        available_models: 2,
        total_models: 2,
        provider: 'OpenRouter',
        models: [
            {
                id: 'gemini',
                name: 'Gemini 3 Flash',
                provider: 'Google',
                status: 'available',
                description: 'Google\'s latest & fastest AI',
            },
            {
                id: 'claude',
                name: 'Claude Opus 4.5',
                provider: 'Anthropic',
                status: 'available',
                description: 'Most capable reasoning model',
            },
        ],
        message: '2/2 models available',
    });
}
