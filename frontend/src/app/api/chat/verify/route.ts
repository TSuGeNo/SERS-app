import { NextResponse } from 'next/server';

export async function GET() {
    const apiKey = process.env.OPENROUTER_API_KEY;
    const hasValidKey = apiKey && apiKey !== 'your_openrouter_api_key_here';

    return NextResponse.json({
        api_key_valid: hasValidKey,
        available_models: 3,
        total_models: 3,
        models: [
            {
                id: 'gemini',
                name: 'Gemini 2.0 Flash',
                provider: 'Google',
                status: 'available',
            },
            {
                id: 'chatgpt',
                name: 'Llama 3.3 70B',
                provider: 'Meta',
                status: 'available',
            },
            {
                id: 'claude',
                name: 'Qwen 2.5 72B',
                provider: 'Alibaba',
                status: 'available',
            },
        ],
        message: hasValidKey
            ? 'All models available via OpenRouter'
            : 'Demo mode active - all models available with demo responses',
    });
}
