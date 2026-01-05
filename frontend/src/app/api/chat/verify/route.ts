import { NextResponse } from 'next/server';

export async function GET() {
    return NextResponse.json({
        api_key_valid: true,
        available_models: 2,
        total_models: 2,
        provider: 'OpenRouter (Free)',
        models: [
            {
                id: 'gemini',
                name: 'Gemini 2.0 Flash',
                provider: 'Google',
                status: 'available',
                description: 'Fast multimodal AI',
            },
            {
                id: 'claude',
                name: 'Llama 3.3 70B',
                provider: 'Meta',
                status: 'available',
                description: 'Powerful open model',
            },
        ],
        message: '2/2 models available (Free)',
    });
}
