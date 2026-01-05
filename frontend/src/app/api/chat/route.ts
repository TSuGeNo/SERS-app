import { NextRequest, NextResponse } from 'next/server';

// Google AI Gemini API - Your API Key
const GOOGLE_AI_API_KEY = process.env.GOOGLE_AI_API_KEY || 'AIzaSyCFJmpbVRj51NhcGOz0dO-NegYO690j8KQ';

// Google AI API endpoint
const GOOGLE_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models';

// Available models
const MODELS = {
    gemini: {
        id: 'gemini-1.5-flash',
        name: 'Gemini 1.5 Flash',
    },
    claude: {
        id: 'gemini-1.5-pro',
        name: 'Gemini 1.5 Pro',
    },
};

const SYSTEM_PROMPT = `You are an expert AI assistant for the SERS-Insight platform, specializing in Surface-Enhanced Raman Spectroscopy (SERS) and scientific data analysis.

Your expertise includes:
- Raman spectroscopy peak identification and analysis
- Python programming for scientific data processing
- Baseline correction, smoothing, and peak fitting
- Molecular fingerprinting and compound identification

Provide helpful, accurate responses with code examples when relevant.`;

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { message, model, dataContext } = body;

        // Check API key
        if (!GOOGLE_AI_API_KEY || GOOGLE_AI_API_KEY.length < 30) {
            return NextResponse.json({
                content: getDemoResponse(message),
                model: model || 'gemini',
                demo: true,
                setupRequired: true,
            });
        }

        const modelKey = model === 'claude' ? 'claude' : 'gemini';
        const modelConfig = MODELS[modelKey as keyof typeof MODELS];

        // Build prompt with context
        let prompt = SYSTEM_PROMPT + '\n\nUser: ' + message;
        if (dataContext?.pointCount) {
            prompt = SYSTEM_PROMPT + `\n\n[Data: ${dataContext.pointCount} points, ${dataContext.wavenumberRange?.min?.toFixed(0)}-${dataContext.wavenumberRange?.max?.toFixed(0)} cm⁻¹]\n\nUser: ` + message;
        }

        // Call Google AI
        const response = await fetch(
            `${GOOGLE_API_URL}/${modelConfig.id}:generateContent?key=${GOOGLE_AI_API_KEY}`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    contents: [{ parts: [{ text: prompt }] }],
                    generationConfig: {
                        temperature: 0.7,
                        maxOutputTokens: 4096,
                    },
                }),
            }
        );

        if (!response.ok) {
            const error = await response.text();
            console.error('Google AI error:', error);
            return NextResponse.json({
                content: getDemoResponse(message),
                model: modelKey,
                demo: true,
                error: 'API error - showing demo response',
            });
        }

        const data = await response.json();
        const content = data.candidates?.[0]?.content?.parts?.[0]?.text;

        if (!content) {
            return NextResponse.json({
                content: getDemoResponse(message),
                model: modelKey,
                demo: true,
            });
        }

        return NextResponse.json({
            content,
            model: modelKey,
            modelName: modelConfig.name,
            demo: false,
        });

    } catch (error) {
        console.error('Error:', error);
        return NextResponse.json({
            content: getDemoResponse(''),
            model: 'gemini',
            demo: true,
            error: String(error),
        });
    }
}

function getDemoResponse(message: string): string {
    const lower = message.toLowerCase();

    if (lower.includes('machine learning') || lower.includes('ml')) {
        return `## Machine Learning

Machine Learning (ML) is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

### Types of Machine Learning:

1. **Supervised Learning** - Learning from labeled data
   - Classification (spam detection, image recognition)
   - Regression (price prediction, risk assessment)

2. **Unsupervised Learning** - Finding patterns in unlabeled data
   - Clustering (customer segmentation)
   - Dimensionality reduction (PCA)

3. **Reinforcement Learning** - Learning through trial and error
   - Game playing, robotics

### Common Algorithms:
- Linear/Logistic Regression
- Decision Trees & Random Forests
- Neural Networks & Deep Learning
- Support Vector Machines (SVM)
- K-Means Clustering

### Python Example:
\`\`\`python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
\`\`\``;
    }

    if (lower.includes('code') || lower.includes('python') || lower.includes('peak')) {
        return `## Python Code for Spectral Analysis

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d

# Load spectrum data
wavenumber = np.linspace(200, 2000, 901)
intensity = np.random.random(901) + 0.5 * np.exp(-((wavenumber-1000)**2)/5000)

# Baseline correction (polynomial)
baseline = np.polyval(np.polyfit(wavenumber, intensity, 3), wavenumber)
corrected = intensity - baseline

# Smoothing
smoothed = savgol_filter(corrected, window_length=11, polyorder=3)

# Peak detection
peaks, properties = find_peaks(smoothed, prominence=0.05, height=0.1)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(wavenumber, smoothed, 'b-', label='Processed')
ax.plot(wavenumber[peaks], smoothed[peaks], 'ro', markersize=8, label='Peaks')
ax.set_xlabel('Wavenumber (cm⁻¹)')
ax.set_ylabel('Intensity (a.u.)')
ax.legend()
plt.tight_layout()
plt.show()

print(f"Detected {len(peaks)} peaks at: {wavenumber[peaks]}")
\`\`\``;
    }

    if (lower.includes('sers') || lower.includes('raman')) {
        return `## Surface-Enhanced Raman Spectroscopy (SERS)

SERS is a powerful analytical technique that enhances Raman scattering signals by factors of **10⁶ to 10¹¹** using metallic nanostructures.

### Key Concepts:

1. **Enhancement Mechanisms:**
   - Electromagnetic (EM) enhancement - plasmon resonance
   - Chemical enhancement - charge transfer

2. **Common Substrates:**
   - Gold nanoparticles (stable, biocompatible)
   - Silver nanoparticles (highest enhancement)
   - Nanostructured surfaces

3. **Applications:**
   - Trace detection (pollutants, drugs)
   - Biomedical diagnostics
   - Food safety analysis

### Typical Peak Assignments:
| Wavenumber | Assignment |
|------------|------------|
| ~1000 cm⁻¹ | Ring breathing (aromatic) |
| ~1340 cm⁻¹ | CH₃ deformation |
| ~1580 cm⁻¹ | C=C stretching |
| ~1650 cm⁻¹ | Amide I |`;
    }

    return `## SERS-Insight AI Assistant

I can help you with:

### 🔬 Spectral Analysis
- Peak identification and assignment
- Baseline correction techniques
- Quantitative analysis

### 💻 Programming
- Python code for data processing
- Visualization with matplotlib
- Machine learning for spectroscopy

### 📚 Research Support
- SERS methodology
- Experimental optimization
- Literature guidance

**Ask me anything!** For example:
- "Write Python code to detect peaks in my spectrum"
- "Explain machine learning for spectroscopy"
- "How do I correct baseline in SERS data?"

---
*To enable full AI capabilities, add your Google AI API key to the environment.*`;
}
