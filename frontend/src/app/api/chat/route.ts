import { NextRequest, NextResponse } from 'next/server';

// OpenRouter API endpoint
const OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions';

// API Keys from environment variables (set these in Vercel dashboard)
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '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';
const CLAUDE_API_KEY = process.env.CLAUDE_API_KEY || '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';

// AI Model configurations - 2 powerful models via OpenRouter
const AI_MODELS: Record<string, {
    openrouterId: string;
    name: string;
    apiKey: string;
    description: string
}> = {
    gemini: {
        openrouterId: 'google/gemini-3-flash-preview',
        name: 'Gemini 3 Flash',
        apiKey: GEMINI_API_KEY,
        description: 'Google\'s latest Gemini 3 model - fast and powerful',
    },
    claude: {
        openrouterId: 'anthropic/claude-opus-4.5',
        name: 'Claude Opus 4.5',
        apiKey: CLAUDE_API_KEY,
        description: 'Anthropic\'s most capable model',
    },
};

const SERS_SYSTEM_PROMPT = `You are an expert AI assistant specializing in Surface-Enhanced Raman Spectroscopy (SERS), spectroscopy data analysis, and scientific programming. You excel at:

1. **Scientific Analysis**: Accurate interpretation of Raman spectra, peak identification, molecular fingerprinting
2. **Programming**: Writing Python code for data analysis, visualization, and spectroscopy workflows
3. **Technical Guidance**: Experimental setup, nanoparticle synthesis, substrate optimization
4. **Data Processing**: Baseline correction, peak fitting, quantitative analysis algorithms

When writing code:
- Use Python with numpy, scipy, matplotlib for spectroscopy
- Include clear comments and explanations
- Provide complete, runnable code examples

When analyzing spectral data:
- Identify characteristic peaks with wavenumber positions (cm⁻¹)
- Suggest molecular assignments based on peak positions
- Discuss relevant vibrational modes

Be helpful, accurate, and provide detailed responses with code examples when appropriate.`;

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { message, model, dataContext } = body;

        // Get model config - default to gemini if invalid model
        const modelKey = model === 'claude' ? 'claude' : 'gemini';
        const modelConfig = AI_MODELS[modelKey];

        if (!modelConfig.apiKey) {
            return NextResponse.json({
                content: generateDemoResponse(message, dataContext),
                model: modelKey,
                demo: true,
            });
        }

        // Build the user message with data context if available
        let userContent = message;
        if (dataContext) {
            userContent = `[Data Context: ${dataContext.pointCount} data points, wavenumber range ${dataContext.wavenumberRange?.min?.toFixed(0)}-${dataContext.wavenumberRange?.max?.toFixed(0)} cm⁻¹, max intensity: ${dataContext.maxIntensity?.toFixed(2)}]\n\n${message}`;
        }

        console.log(`Calling OpenRouter API with model: ${modelConfig.openrouterId}`);

        // Call OpenRouter API
        const response = await fetch(OPENROUTER_API_URL, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${modelConfig.apiKey}`,
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://sers-insight.vercel.app',
                'X-Title': 'SERS-Insight Platform',
            },
            body: JSON.stringify({
                model: modelConfig.openrouterId,
                messages: [
                    { role: 'system', content: SERS_SYSTEM_PROMPT },
                    { role: 'user', content: userContent },
                ],
                max_tokens: 4000,
                temperature: 0.7,
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('OpenRouter API error:', response.status, errorText);

            let errorMessage = 'API error occurred';
            try {
                const errorData = JSON.parse(errorText);
                errorMessage = errorData.error?.message || errorMessage;
            } catch {
                errorMessage = errorText.substring(0, 200);
            }

            return NextResponse.json({
                content: `**API Error**: ${errorMessage}\n\n${generateDemoResponse(message, dataContext)}`,
                model: modelKey,
                demo: true,
                error: errorMessage,
            });
        }

        const data = await response.json();
        const content = data.choices?.[0]?.message?.content || 'No response generated';

        console.log(`Successfully got response from ${modelConfig.name}`);

        return NextResponse.json({
            content,
            model: modelKey,
            modelName: modelConfig.name,
            demo: false,
        });

    } catch (error) {
        console.error('Chat API error:', error);
        return NextResponse.json(
            {
                error: 'Failed to process request',
                details: error instanceof Error ? error.message : 'Unknown error'
            },
            { status: 500 }
        );
    }
}

// Demo response fallback
function generateDemoResponse(message: string, dataContext?: any): string {
    const lowerMessage = message.toLowerCase();

    if (lowerMessage.includes('code') || lowerMessage.includes('python') || lowerMessage.includes('program')) {
        return `## Python Code Example

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# Load your spectrum data
wavenumber = np.linspace(200, 2000, 901)
intensity = np.random.random(901) * 0.5

# Baseline correction
baseline = np.polyval(np.polyfit(wavenumber, intensity, 3), wavenumber)
corrected = intensity - baseline

# Smoothing with Savitzky-Golay filter
smoothed = savgol_filter(corrected, window_length=11, polyorder=3)

# Peak detection
peaks, _ = find_peaks(smoothed, prominence=0.1)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(wavenumber, smoothed, 'b-', label='Processed')
plt.plot(wavenumber[peaks], smoothed[peaks], 'ro', label='Peaks')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity')
plt.legend()
plt.show()
\`\`\``;
    }

    if (dataContext) {
        return `## Spectrum Analysis

Based on your data (${dataContext.pointCount} points, ${dataContext.wavenumberRange?.min?.toFixed(0)}-${dataContext.wavenumberRange?.max?.toFixed(0)} cm⁻¹):

- Max intensity: ${dataContext.maxIntensity?.toFixed(2)} a.u.
- Spectral range: Fingerprint region

### Common Peak Assignments
| Wavenumber | Assignment |
|------------|------------|
| ~1000 cm⁻¹ | Phenylalanine |
| ~1450 cm⁻¹ | CH₂ bending |
| ~1650 cm⁻¹ | Amide I |`;
    }

    return `## SERS-Insight AI Assistant

I can help with:
- **Spectral Analysis**: Peak identification, baseline correction
- **Python Programming**: Data processing, visualization
- **Research**: Methodology, troubleshooting

Ask me anything about SERS or spectroscopy!`;
}
