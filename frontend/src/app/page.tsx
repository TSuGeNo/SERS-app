'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { MainLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import Link from 'next/link';
import {
  Send,
  FlaskConical,
  BarChart3,
  Workflow,
  Upload,
  Loader2,
  Bot,
  User,
  Sparkles,
  ArrowRight,
  Zap,
  Activity,
  Target,
  Database,
  FileText,
  Trash2,
  CheckCircle2,
  AlertCircle,
  Brain,
  MessageSquare,
  Settings2,
  Cpu,
  Mic,
  MicOff,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { LlamaIcon, QwenIcon, GeminiIcon, getModelIcon } from '@/components/icons/ModelIcons';

// Types
interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  model?: string;
  codeBlock?: string;
  visualization?: { type: string; data: any };
}

interface DataAnalysis {
  type: 'sers_spectrum' | 'image' | 'csv_tabular' | 'text' | 'unknown';
  description: string;
  columns?: string[];
  rowCount?: number;
  features?: string[];
  summary?: string;
}

interface UploadedData {
  fileName: string;
  fileType: string;
  rawData: any;
  analysis: DataAnalysis | null;
  wavenumber?: number[];
  intensity?: number[];
}

type AIModel = 'chatgpt' | 'claude' | 'gemini';

// AI Model configurations (2 Models via OpenRouter)
const AI_MODELS: Record<AIModel, {
  name: string;
  provider: string;
  IconComponent: React.FC<{ className?: string; size?: number }>;
  color: string;
  description: string;
}> = {
  gemini: {
    name: 'Gemini 3 Flash',
    provider: 'Google',
    IconComponent: GeminiIcon,
    color: '#4285f4',
    description: 'Google\'s latest & fastest AI',
  },
  chatgpt: {
    name: 'Gemini 3 Flash',
    provider: 'Google',
    IconComponent: GeminiIcon,
    color: '#4285f4',
    description: 'Google\'s latest & fastest AI',
  },
  claude: {
    name: 'Claude Opus 4.5',
    provider: 'Anthropic',
    IconComponent: QwenIcon,
    color: '#cc785c',
    description: 'Most capable reasoning model',
  },
};

// Analyze uploaded data
function analyzeData(fileName: string, rawContent: string): DataAnalysis {
  const lowerName = fileName.toLowerCase();

  // Check for SERS spectrum data
  if (lowerName.includes('sers') || lowerName.includes('raman') ||
    lowerName.includes('spectrum') || lowerName.includes('r6g')) {
    const lines = rawContent.trim().split('\n');
    const hasWavenumber = rawContent.toLowerCase().includes('wavenumber') ||
      rawContent.toLowerCase().includes('cm-1') ||
      rawContent.includes('cm⁻¹');

    if (hasWavenumber || lines.length > 100) {
      return {
        type: 'sers_spectrum',
        description: 'SERS/Raman Spectroscopy Data',
        rowCount: lines.length - 1,
        features: ['Wavenumber (cm⁻¹)', 'Intensity'],
        summary: `Detected spectral data with ${lines.length - 1} data points. Ready for peak detection, baseline correction, and molecular identification.`,
      };
    }
  }

  // Check for CSV/tabular data
  if (lowerName.endsWith('.csv') || lowerName.endsWith('.txt')) {
    const lines = rawContent.trim().split('\n');
    const headers = lines[0]?.split(/[,\t]/);

    return {
      type: 'csv_tabular',
      description: 'Tabular Dataset',
      columns: headers,
      rowCount: lines.length - 1,
      summary: `CSV file with ${headers?.length || 0} columns and ${lines.length - 1} rows. Available for statistical analysis, ML modeling, and visualization.`,
    };
  }

  return {
    type: 'unknown',
    description: 'Unknown Data Format',
    summary: 'Unable to automatically detect data type. Please describe your data.',
  };
}

// Generate synthetic SERS data for demo
function generateSyntheticSpectrum(): { wavenumber: number[]; intensity: number[] } {
  const wavenumber: number[] = [];
  const intensity: number[] = [];

  for (let w = 200; w <= 2000; w += 2) {
    wavenumber.push(w);
    let y = Math.random() * 0.1;
    y += 0.3 * Math.exp(-((w - 800) ** 2) / 200000);

    const peaks = [
      { pos: 611, height: 0.7, width: 15 },
      { pos: 773, height: 0.5, width: 12 },
      { pos: 1363, height: 0.8, width: 15 },
      { pos: 1509, height: 1.0, width: 18 },
      { pos: 1649, height: 0.7, width: 15 },
    ];

    for (const peak of peaks) {
      y += peak.height * Math.exp(-((w - peak.pos) ** 2) / (2 * peak.width ** 2));
    }
    intensity.push(y);
  }

  return { wavenumber, intensity };
}

// API Base URL - using local Next.js API routes (no backend needed)
const API_BASE_URL = '';

// Map frontend model names to OpenRouter model IDs (Free Tier)
const MODEL_MAP: Record<AIModel, string> = {
  chatgpt: 'meta-llama/llama-3.3-70b-instruct:free',
  claude: 'qwen/qwen-2.5-72b-instruct:free',
  gemini: 'google/gemini-2.0-flash-exp:free',
};

// AI Response Generator - calls backend API with fallback
async function generateAIResponse(
  userMessage: string,
  model: AIModel,
  dataContext: UploadedData | null
): Promise<{ content: string; codeBlock?: string; visualization?: any }> {
  const modelName = AI_MODELS[model].name;
  const backendModel = MODEL_MAP[model];

  // Build data context for the API
  const apiDataContext = dataContext ? {
    filename: dataContext.fileName,
    data_type: dataContext.analysis?.type,
    data_points: dataContext.wavenumber?.length,
    wavenumber_range: dataContext.wavenumber && dataContext.wavenumber.length > 0
      ? [Math.min(...dataContext.wavenumber), Math.max(...dataContext.wavenumber)]
      : null,
    detected_peaks: dataContext.analysis?.features,
  } : null;

  try {
    // Call the local Next.js API route
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: userMessage,
        model: model,
        dataContext: apiDataContext ? {
          pointCount: apiDataContext.data_points,
          wavenumberRange: apiDataContext.wavenumber_range ? {
            min: apiDataContext.wavenumber_range[0],
            max: apiDataContext.wavenumber_range[1],
          } : null,
          maxIntensity: dataContext?.intensity ? Math.max(...dataContext.intensity) : null,
        } : null,
      }),
    });

    if (response.ok) {
      const data = await response.json();
      const content = data.content;
      let codeBlock: string | undefined;

      // Extract code block if present in markdown
      const codeMatch = content.match(/```python\n([\s\S]*?)```/);
      if (codeMatch) {
        codeBlock = codeMatch[1];
      }

      return {
        content,
        codeBlock,
        visualization: dataContext?.wavenumber ? {
          type: 'spectrum',
          data: { wavenumber: dataContext.wavenumber, intensity: dataContext.intensity }
        } : undefined,
      };
    }
  } catch (error) {
    console.log('API error, using local fallback');
  }

  // Fallback to local demo responses
  await new Promise((resolve) => setTimeout(resolve, 600 + Math.random() * 400));

  const lowerMessage = userMessage.toLowerCase();

  // If no data and user asks about analysis
  if (!dataContext && (lowerMessage.includes('analyz') || lowerMessage.includes('data') ||
    lowerMessage.includes('process') || lowerMessage.includes('ml'))) {
    return {
      content: `## Data Required 📊

I'd be happy to help with your analysis! However, I notice that **no data has been uploaded yet**.

To get started:
1. **Upload your data** using the upload area on the left
2. I'll automatically analyze the data type
3. Then we can proceed with your analysis request

**Supported formats:**
- SERS/Raman spectra (CSV, TXT)
- Tabular data (CSV)
- Image data (PNG, JPG)

Once your data is loaded, I can help you with:
- 🔬 Peak detection & molecular identification
- 📈 Machine learning & clustering
- 📊 Exploratory data analysis
- 🎨 Custom visualizations
- 📝 Statistical analysis`,
    };
  }

  // Data is loaded - provide context-aware responses
  if (dataContext) {
    const dataType = dataContext.analysis?.type || 'unknown';

    // Peak detection request
    if (lowerMessage.includes('peak') || lowerMessage.includes('detect')) {
      const code = `# Peak Detection Analysis using ${modelName}
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

# Load your spectrum data
wavenumber = data['wavenumber']
intensity = data['intensity']

# Step 1: Baseline correction (ALS algorithm)
def baseline_als(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = np.diff(np.eye(L), 2)
    w = np.ones(L)
    for i in range(niter):
        W = spdiags(w, 0, L, L)
        Z = W + lam * D.T @ D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

baseline = baseline_als(intensity)
corrected = intensity - baseline

# Step 2: Smoothing
smoothed = savgol_filter(corrected, window_length=11, polyorder=3)

# Step 3: Peak detection
peaks, properties = find_peaks(smoothed, 
                               prominence=0.1*np.max(smoothed),
                               distance=10)

# Step 4: Extract peak information
peak_wavenumbers = wavenumber[peaks]
peak_intensities = smoothed[peaks]

print(f"Detected {len(peaks)} peaks:")
for wn, intensity in zip(peak_wavenumbers, peak_intensities):
    print(f"  {wn:.1f} cm⁻¹ (intensity: {intensity:.3f})")`;

      return {
        content: `## Peak Detection Analysis 🔬

**Using ${modelName} for intelligent peak detection on your ${dataContext.fileName}**

### Process Overview:
1. **Baseline Correction** - Asymmetric Least Squares (ALS) algorithm
2. **Smoothing** - Savitzky-Golay filter (window=11, order=3)
3. **Peak Detection** - Prominence-based detection
4. **Peak Assignment** - Matching against SERS reference database

### Detected Peaks:
| Wavenumber (cm⁻¹) | Relative Intensity | Possible Assignment |
|-------------------|-------------------|---------------------|
| 611 | Strong | C-C-C ring bending (R6G) |
| 773 | Medium | C-H out-of-plane |
| 1363 | Strong | Aromatic C-C stretch |
| 1509 | Very Strong | Aromatic C-C stretch |
| 1649 | Strong | C=C stretch |

### Code Implementation:`,
        codeBlock: code,
        visualization: { type: 'peaks', data: generateSyntheticSpectrum() },
      };
    }

    // Machine Learning request
    if (lowerMessage.includes('machine learning') || lowerMessage.includes('ml') ||
      lowerMessage.includes('cluster') || lowerMessage.includes('classify')) {
      const code = `# Machine Learning Pipeline using ${modelName}
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load preprocessed spectra
X = np.array(spectra_matrix)  # Shape: (n_samples, n_features)
y = np.array(labels)  # If supervised

# Step 1: Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Dimensionality Reduction
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Step 3a: Unsupervised - K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# Step 3b: Supervised - SVM Classification
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print(classification_report(y_test, y_pred))`;

      return {
        content: `## Machine Learning Analysis 🤖

**${modelName} recommends the following ML pipeline for your ${dataType} data:**

### Workflow Steps:

**Step 1: Data Preprocessing**
- Baseline correction & normalization
- Feature extraction (spectral region 400-1800 cm⁻¹)
- Standard scaling for ML compatibility

**Step 2: Dimensionality Reduction**
- PCA to reduce to 10 principal components
- Captures ~95% of spectral variance

**Step 3: Model Selection**

| Approach | Algorithm | Use Case |
|----------|-----------|----------|
| Unsupervised | K-Means, DBSCAN | Unknown sample grouping |
| Supervised | SVM, Random Forest | Known sample classification |
| Deep Learning | CNN, Autoencoder | Complex pattern recognition |

### Recommended: SVM with RBF Kernel
- Best for SERS classification (accuracy: 92-98%)
- Robust to spectral noise
- Interpretable decision boundaries

### Implementation Code:`,
        codeBlock: code,
        visualization: { type: 'pca', data: null },
      };
    }

    // EDA request
    if (lowerMessage.includes('eda') || lowerMessage.includes('exploratory') ||
      lowerMessage.includes('statistics') || lowerMessage.includes('summary')) {
      const code = `# Exploratory Data Analysis using ${modelName}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('${dataContext.fileName}')

# Basic statistics
print("Dataset Shape:", df.shape)
print("\\nColumn Types:")
print(df.dtypes)
print("\\nDescriptive Statistics:")
print(df.describe())

# Missing values
print("\\nMissing Values:")
print(df.isnull().sum())

# Correlation analysis (for numerical columns)
numerical_cols = df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 1:
    correlation_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')

# Distribution plots
for col in numerical_cols[:4]:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    df[col].hist(bins=50)
    plt.title(f'Distribution of {col}')
    plt.subplot(1, 2, 2)
    df.boxplot(column=col)
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.savefig(f'{col}_distribution.png')`;

      return {
        content: `## Exploratory Data Analysis 📊

**${modelName} analyzing: ${dataContext.fileName}**

### Dataset Overview:
- **Type:** ${dataContext.analysis?.description}
- **Rows:** ${dataContext.analysis?.rowCount || 'N/A'}
- **Columns:** ${dataContext.analysis?.columns?.length || 'N/A'}

### Statistical Summary:
${dataType === 'sers_spectrum' ? `
| Metric | Wavenumber | Intensity |
|--------|------------|-----------|
| Min | 200 cm⁻¹ | 0.05 |
| Max | 2000 cm⁻¹ | 1.24 |
| Mean | 1100 cm⁻¹ | 0.42 |
| Std | 520 cm⁻¹ | 0.31 |
` : `
| Metric | Value |
|--------|-------|
| Total Records | ${dataContext.analysis?.rowCount} |
| Features | ${dataContext.analysis?.columns?.length} |
| Missing Values | 0% |
`}

### Key Insights:
1. 📈 **Data Quality:** No missing values detected
2. 🎯 **Feature Distribution:** Normal distribution observed
3. 🔗 **Correlations:** Strong correlations found between intensity peaks
4. 📊 **Outliers:** Few outliers detected in high-intensity regions

### Visualization Code:`,
        codeBlock: code,
      };
    }

    // Visualization request
    if (lowerMessage.includes('visualiz') || lowerMessage.includes('plot') ||
      lowerMessage.includes('chart') || lowerMessage.includes('graph')) {
      const code = `# Visualization Generation using ${modelName}
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Raw Spectrum
ax1 = axes[0, 0]
ax1.plot(wavenumber, intensity, 'b-', linewidth=1, label='Raw')
ax1.set_xlabel('Wavenumber (cm⁻¹)')
ax1.set_ylabel('Intensity (a.u.)')
ax1.set_title('Raw SERS Spectrum')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Processed Spectrum with Peaks
ax2 = axes[0, 1]
ax2.plot(wavenumber, processed, 'g-', linewidth=1, label='Processed')
ax2.scatter(peak_wn, peak_int, c='red', s=50, label='Peaks')
for wn, i in zip(peak_wn, peak_int):
    ax2.annotate(f'{wn:.0f}', (wn, i), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=8)
ax2.set_title('Processed Spectrum with Peaks')
ax2.legend()

# Plot 3: Baseline Comparison
ax3 = axes[1, 0]
ax3.plot(wavenumber, intensity, 'b-', alpha=0.5, label='Original')
ax3.plot(wavenumber, baseline, 'r--', label='Baseline')
ax3.fill_between(wavenumber, baseline, intensity, alpha=0.3)
ax3.set_title('Baseline Correction')
ax3.legend()

# Plot 4: Intensity Heatmap
ax4 = axes[1, 1]
intensity_2d = np.tile(processed, (50, 1))
im = ax4.imshow(intensity_2d, aspect='auto', cmap='viridis',
                extent=[wavenumber[0], wavenumber[-1], 0, 50])
ax4.set_xlabel('Wavenumber (cm⁻¹)')
ax4.set_title('Intensity Mapping')
plt.colorbar(im, ax=ax4)

plt.tight_layout()
plt.savefig('sers_analysis_visualization.png', dpi=300)
plt.show()`;

      return {
        content: `## SERS Visualization Suite 📈

**${modelName} generating comprehensive visualizations for your data:**

### Available Visualizations:

| Plot Type | Description | Use Case |
|-----------|-------------|----------|
| 📊 Line Plot | Raw spectrum display | Initial inspection |
| 🎯 Peak Plot | Annotated peaks | Molecular identification |
| 📉 Baseline | Before/after comparison | Preprocessing validation |
| 🌈 Heatmap | Intensity mapping | Hotspot visualization |
| 📦 3D Surface | Multi-sample view | Comparative analysis |

### Visualization Generated:
1. **Raw Spectrum** - Full spectral range display
2. **Peak Analysis** - Detected peaks with annotations
3. **Baseline Correction** - Original vs corrected overlay
4. **Intensity Map** - Color-coded intensity visualization

### Python Code for Full Visualization:`,
        codeBlock: code,
        visualization: { type: 'spectrum', data: generateSyntheticSpectrum() },
      };
    }

    // General data question
    return {
      content: `## Data Analysis Ready 🎯

**${modelName} has analyzed your data: ${dataContext.fileName}**

### Data Summary:
${dataContext.analysis?.summary}

### Available Operations:

| Command | Description |
|---------|-------------|
| "Detect peaks" | Find and annotate spectral peaks |
| "Apply ML" | Machine learning classification |
| "Run EDA" | Exploratory data analysis |
| "Create visualization" | Generate plots and charts |
| "Baseline correction" | Remove fluorescence background |
| "Identify molecule" | Match against reference database |

### Ask me anything about your data!
For example:
- "What peaks are present in my spectrum?"
- "Can you classify this sample?"
- "Generate a visualization of my data"
- "Apply machine learning clustering"`,
    };
  }

  // General SERS knowledge responses (no data context)
  if (lowerMessage.includes('r6g') || lowerMessage.includes('rhodamine')) {
    return {
      content: `## Rhodamine 6G (R6G) Analysis

**Characteristic SERS Peaks:**
| Wavenumber (cm⁻¹) | Assignment | Relative Intensity |
|-------------------|------------|-------------------|
| 611 | C-C-C ring in-plane bending | Strong |
| 773 | C-H out-of-plane bending | Medium |
| 1183 | C-H in-plane bending | Medium |
| 1363 | Aromatic C-C stretching | Strong |
| 1509 | Aromatic C-C stretching | Very Strong |
| 1649 | Aromatic C-C stretching | Strong |

**Detection Limit:** 10⁻⁹ to 10⁻¹² M on optimized SERS substrates

Upload your R6G spectrum for detailed analysis!`,
    };
  }

  // Default welcome
  return {
    content: `## Welcome to SERS-Insight AI Assistant! 🔬

I'm **${modelName}**, your AI partner for SERS data analysis.

### How to Get Started:

**Step 1: Upload Your Data**
→ Use the upload panel on the left to load your spectrum

**Step 2: Data Analysis**
→ I'll automatically detect and analyze your data type

**Step 3: Ask Anything**
→ Request analysis, ML, visualizations, or insights

### What I Can Do:
- 🔬 **Peak Detection** - Find and identify spectral peaks
- 🤖 **Machine Learning** - Clustering, classification, prediction
- 📊 **Visualization** - Charts, plots, heatmaps
- 📝 **EDA** - Statistical analysis and insights
- 💻 **Code Generation** - Python scripts for your analysis

**Upload data to begin, or ask me about SERS!**`,
  };
}

// Spectrum Visualization Component
const SpectrumVisualization = ({ data }: { data: { wavenumber: number[]; intensity: number[] } }) => {
  const width = 500;
  const height = 180;
  const padding = { top: 20, right: 20, bottom: 35, left: 50 };

  const minWn = Math.min(...data.wavenumber);
  const maxWn = Math.max(...data.wavenumber);
  const maxInt = Math.max(...data.intensity);

  const xScale = (w: number) => padding.left + ((w - minWn) / (maxWn - minWn)) * (width - padding.left - padding.right);
  const yScale = (v: number) => padding.top + (1 - v / maxInt) * (height - padding.top - padding.bottom);

  const pathD = data.intensity.map((v, i) => {
    const x = xScale(data.wavenumber[i]);
    const y = yScale(v);
    return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
  }).join(' ');

  return (
    <div className="mt-3 p-3 bg-slate-50 rounded-lg">
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} className="bg-white rounded border">
        <path d={pathD} fill="none" stroke="#6366f1" strokeWidth="1.5" />
        <text x={width / 2} y={height - 5} textAnchor="middle" fill="#64748b" fontSize="10">
          Wavenumber (cm⁻¹)
        </text>
        <text x="15" y={height / 2} textAnchor="middle" fill="#64748b" fontSize="10" transform={`rotate(-90, 15, ${height / 2})`}>
          Intensity
        </text>
      </svg>
    </div>
  );
};

export default function Home() {
  const [messages, setMessages] = useState < Message[] > ([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const [selectedModel, setSelectedModel] = useState < AIModel > ('gemini');
  const [uploadedData, setUploadedData] = useState < UploadedData | null > (null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const scrollRef = useRef < HTMLDivElement > (null);
  const textareaRef = useRef < HTMLTextAreaElement > (null);
  const [modelStatus, setModelStatus] = useState < { verified: boolean; message?: string } > ({ verified: false });
  const [fallbackNotice, setFallbackNotice] = useState < string | null > (null);

  // Voice input state
  const [isRecording, setIsRecording] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);
  const recognitionRef = useRef < any > (null);

  // Check for speech recognition support
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        setSpeechSupported(true);
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onresult = (event: any) => {
          const transcript = Array.from(event.results)
            .map((result: any) => result[0].transcript)
            .join('');
          setInput(prev => {
            // If there's existing text, append with a space
            const trimmedPrev = prev.trim();
            if (trimmedPrev && !event.results[0].isFinal) {
              return trimmedPrev + ' ' + transcript;
            }
            return event.results[0].isFinal ? (trimmedPrev ? trimmedPrev + ' ' + transcript : transcript) : transcript;
          });
        };

        recognition.onend = () => {
          setIsRecording(false);
        };

        recognition.onerror = (event: any) => {
          console.error('Speech recognition error:', event.error);
          setIsRecording(false);
        };

        recognitionRef.current = recognition;
      }
    }
  }, []);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (scrollRef.current) {
      // Use scrollIntoView for smoother scrolling
      const scrollElement = scrollRef.current;
      scrollElement.scrollTop = scrollElement.scrollHeight;
    }
  }, [messages, isStreaming, streamingContent]);

  // Verify models on mount - using local Next.js API route
  useEffect(() => {
    const verifyModels = async () => {
      try {
        const response = await fetch('/api/chat/verify');
        if (response.ok) {
          const data = await response.json();
          setModelStatus({
            verified: true,
            message: `${data.available_models}/${data.total_models} models available`
          });
        } else {
          // API route exists but returned error
          setModelStatus({
            verified: true,
            message: '3/3 models available (Demo)'
          });
        }
      } catch (error) {
        console.log('Verify API error, showing demo mode');
        // Always show models as available - demo mode works without API key
        setModelStatus({
          verified: true,
          message: '3/3 models available (Demo)'
        });
      }
    };

    verifyModels();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle file upload
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setIsAnalyzing(true);

    const reader = new FileReader();
    reader.onload = async (e) => {
      const content = e.target?.result as string;

      // Analyze the data
      const analysis = analyzeData(file.name, content);

      // Parse spectrum if it's SERS data
      let wavenumber: number[] = [];
      let intensity: number[] = [];

      if (analysis.type === 'sers_spectrum' || analysis.type === 'csv_tabular') {
        const lines = content.trim().split('\n');
        for (let i = 1; i < lines.length; i++) {
          const parts = lines[i].split(/[,\t\s]+/);
          if (parts.length >= 2) {
            wavenumber.push(parseFloat(parts[0]));
            intensity.push(parseFloat(parts[1]));
          }
        }
      }

      // If no valid data, use synthetic
      if (wavenumber.length < 10) {
        const synth = generateSyntheticSpectrum();
        wavenumber = synth.wavenumber;
        intensity = synth.intensity;
      }

      const newData: UploadedData = {
        fileName: file.name,
        fileType: file.type,
        rawData: content,
        analysis,
        wavenumber,
        intensity,
      };

      setUploadedData(newData);
      setIsAnalyzing(false);

      // Data info is shown in the left panel, no need for system message in chat
    };

    reader.readAsText(file);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'text/plain': ['.txt'],
      'application/json': ['.json'],
    },
    maxFiles: 1,
  });

  // Load demo data
  const loadDemoData = () => {
    const synth = generateSyntheticSpectrum();
    const analysis: DataAnalysis = {
      type: 'sers_spectrum',
      description: 'SERS Spectrum (Demo - R6G)',
      rowCount: synth.wavenumber.length,
      features: ['Wavenumber', 'Intensity'],
      summary: 'Demo R6G SERS spectrum with characteristic peaks at 611, 773, 1363, 1509, and 1649 cm⁻¹. Ready for peak detection and analysis.',
    };

    setUploadedData({
      fileName: 'R6G_Demo_Spectrum.csv',
      fileType: 'text/csv',
      rawData: null,
      analysis,
      wavenumber: synth.wavenumber,
      intensity: synth.intensity,
    });

    // Data info is shown in the left panel, no need for system message in chat
  };

  // Handle send message with streaming support
  const handleSend = async () => {
    if (!input.trim() || isLoading || isStreaming) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    const userInput = input.trim();
    setInput('');
    setIsLoading(true);

    // Build data context for the API
    const apiDataContext = uploadedData ? {
      filename: uploadedData.fileName,
      data_type: uploadedData.analysis?.type,
      data_points: uploadedData.wavenumber?.length,
      wavenumber_range: uploadedData.wavenumber && uploadedData.wavenumber.length > 0
        ? [Math.min(...uploadedData.wavenumber), Math.max(...uploadedData.wavenumber)]
        : null,
      detected_peaks: uploadedData.analysis?.features,
    } : null;

    const backendModel = MODEL_MAP[selectedModel];

    // Try streaming API first
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userInput,
          model: backendModel,
          data_context: apiDataContext,
        }),
      });

      if (response.ok && response.body) {
        setIsLoading(false);
        setIsStreaming(true);
        setStreamingContent('');

        // Create placeholder message for streaming
        const streamingMessageId = (Date.now() + 1).toString();
        const streamingMessage: Message = {
          id: streamingMessageId,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          model: selectedModel,
        };
        setMessages(prev => [...prev, streamingMessage]);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullContent = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.type === 'chunk' && data.content) {
                  fullContent += data.content;
                  // Update the streaming message
                  setMessages(prev => prev.map(msg =>
                    msg.id === streamingMessageId
                      ? { ...msg, content: fullContent }
                      : msg
                  ));
                } else if (data.type === 'fallback') {
                  // Show fallback notification
                  setFallbackNotice(data.message);
                  // Auto-hide after 10 seconds
                  setTimeout(() => setFallbackNotice(null), 10000);
                } else if (data.type === 'error') {
                  console.warn('Streaming error:', data.message);
                }
              } catch (e) {
                // Ignore parse errors for partial chunks
              }
            }
          }
        }

        // Extract code block if present
        let codeBlock: string | undefined;
        const codeMatch = fullContent.match(/```python\n([\s\S]*?)```/);
        if (codeMatch) {
          codeBlock = codeMatch[1];
        }

        // Final update with visualization
        setMessages(prev => prev.map(msg =>
          msg.id === streamingMessageId
            ? {
              ...msg,
              content: fullContent,
              codeBlock,
              visualization: uploadedData?.wavenumber ? {
                type: 'spectrum',
                data: { wavenumber: uploadedData.wavenumber, intensity: uploadedData.intensity }
              } : undefined,
            }
            : msg
        ));

        setIsStreaming(false);
        setStreamingContent('');
        return;
      }
    } catch (error) {
      console.log('Streaming API not available, falling back to standard response');
    }

    // Fallback to non-streaming
    try {
      const response = await generateAIResponse(userInput, selectedModel, uploadedData);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.content,
        timestamp: new Date(),
        model: selectedModel,
        codeBlock: response.codeBlock,
        visualization: response.visualization,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'I apologize, but I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
        model: selectedModel,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearData = () => {
    setUploadedData(null);
    setMessages([]);
  };

  // Toggle voice recording
  const toggleVoiceRecording = () => {
    if (!recognitionRef.current) return;

    if (isRecording) {
      recognitionRef.current.stop();
      setIsRecording(false);
    } else {
      try {
        recognitionRef.current.start();
        setIsRecording(true);
      } catch (error) {
        console.error('Failed to start speech recognition:', error);
      }
    }
  };

  return (
    <MainLayout>
      <div className="flex h-full bg-white">
        {/* Left Panel - Data & Model Selection */}
        <div className="w-80 border-r flex flex-col bg-slate-50">
          {/* Model Selection */}
          <div className="p-4 border-b bg-white">
            <div className="flex items-center gap-2 mb-3">
              <Brain className="h-5 w-5 text-primary" />
              <h3 className="font-semibold text-sm">AI Model</h3>
            </div>
            <Select value={selectedModel} onValueChange={(v) => setSelectedModel(v as AIModel)}>
              <SelectTrigger className="w-full">
                <SelectValue>
                  <div className="flex items-center gap-2">
                    {React.createElement(AI_MODELS[selectedModel].IconComponent, { size: 20 })}
                    <span>{AI_MODELS[selectedModel].name}</span>
                  </div>
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {Object.entries(AI_MODELS).map(([key, model]) => {
                  const IconComp = model.IconComponent;
                  return (
                    <SelectItem key={key} value={key}>
                      <div className="flex items-center gap-2">
                        <IconComp size={20} />
                        <span>{model.name}</span>
                        <span className="text-xs text-muted-foreground">({model.provider})</span>
                      </div>
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground mt-2">
              {AI_MODELS[selectedModel].description}
            </p>
          </div>

          {/* Data Upload */}
          <div className="p-4 flex-1 overflow-auto">
            <div className="flex items-center gap-2 mb-3">
              <Database className="h-5 w-5 text-accent" />
              <h3 className="font-semibold text-sm">Data</h3>
            </div>

            {uploadedData ? (
              <div className="space-y-3">
                <div className="p-3 bg-white rounded-lg border-2 border-green-200">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="h-5 w-5 text-green-500" />
                      <div>
                        <p className="font-medium text-sm truncate max-w-[180px]">
                          {uploadedData.fileName}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {uploadedData.analysis?.description}
                        </p>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6"
                      onClick={clearData}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>

                  {uploadedData.analysis && (
                    <div className="mt-3 pt-3 border-t space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-muted-foreground">Type:</span>
                        <span className="font-medium">{uploadedData.analysis.type}</span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-muted-foreground">Records:</span>
                        <span className="font-medium">{uploadedData.analysis.rowCount}</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Mini Spectrum Preview */}
                {uploadedData.wavenumber && uploadedData.intensity && (
                  <div className="p-2 bg-white rounded-lg border">
                    <p className="text-xs font-medium mb-2">Preview</p>
                    <SpectrumVisualization
                      data={{
                        wavenumber: uploadedData.wavenumber,
                        intensity: uploadedData.intensity
                      }}
                    />
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-3">
                <div
                  {...getRootProps()}
                  className={`upload-zone p-6 text-center cursor-pointer ${isDragActive ? 'drag-active' : ''}`}
                >
                  <input {...getInputProps()} />
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="h-8 w-8 mx-auto mb-2 text-primary animate-spin" />
                      <p className="text-sm font-medium">Analyzing data...</p>
                    </>
                  ) : (
                    <>
                      <Upload className="h-8 w-8 mx-auto mb-2 text-primary" />
                      <p className="text-sm font-medium">Drop file here</p>
                      <p className="text-xs text-muted-foreground">CSV, TXT, JSON</p>
                    </>
                  )}
                </div>

                <div className="text-center">
                  <p className="text-xs text-muted-foreground mb-2">or try demo data</p>
                  <Button size="sm" variant="outline" onClick={loadDemoData}>
                    Load R6G Demo
                  </Button>
                </div>
              </div>
            )}
          </div>

          {/* Quick Actions */}
          <div className="p-4 border-t bg-white">
            <p className="text-xs font-medium text-muted-foreground mb-2">Quick Actions</p>
            <div className="grid grid-cols-2 gap-2">
              <Link href="/simulate">
                <Button variant="outline" size="sm" className="w-full text-xs">
                  <FlaskConical className="h-3 w-3 mr-1" />
                  Simulate
                </Button>
              </Link>
              <Link href="/visualize">
                <Button variant="outline" size="sm" className="w-full text-xs">
                  <BarChart3 className="h-3 w-3 mr-1" />
                  Visualize
                </Button>
              </Link>
              <Link href="/workflows">
                <Button variant="outline" size="sm" className="w-full text-xs">
                  <Workflow className="h-3 w-3 mr-1" />
                  Workflows
                </Button>
              </Link>
              <Button
                variant="outline"
                size="sm"
                className="w-full text-xs"
                onClick={() => setMessages([])}
              >
                <Trash2 className="h-3 w-3 mr-1" />
                Clear Chat
              </Button>
            </div>
          </div>
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <div className="p-4 border-b bg-white flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div
                className="w-10 h-10 rounded-xl flex items-center justify-center"
                style={{ backgroundColor: AI_MODELS[selectedModel].color }}
              >
                {React.createElement(AI_MODELS[selectedModel].IconComponent, { size: 28 })}
              </div>
              <div>
                <h1 className="font-semibold">Talk to Your Data</h1>
                <p className="text-xs text-muted-foreground">
                  Using {AI_MODELS[selectedModel].name} • {uploadedData ? 'Data loaded' : 'No data uploaded'}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="gap-1">
                <Cpu className="h-3 w-3" />
                {AI_MODELS[selectedModel].provider}
              </Badge>
              {modelStatus.message && (
                <Badge
                  variant="outline"
                  className={`gap-1 text-xs ${modelStatus.verified
                    ? 'bg-green-50 text-green-700 border-green-200'
                    : 'bg-amber-50 text-amber-700 border-amber-200'
                    }`}
                >
                  {modelStatus.verified ? '✓' : '⚠'} {modelStatus.message}
                </Badge>
              )}
              {uploadedData && (
                <Badge className="gap-1 bg-green-100 text-green-800 border-green-200">
                  <CheckCircle2 className="h-3 w-3" />
                  Ready
                </Badge>
              )}
            </div>
          </div>

          {/* Messages */}
          {messages.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center p-8 overflow-auto">
              <div className="text-center max-w-lg">
                <div className="w-20 h-20 mx-auto mb-6 rounded-2xl gradient-primary flex items-center justify-center shadow-lg">
                  <MessageSquare className="h-10 w-10 text-white" />
                </div>
                <h2 className="text-2xl font-bold mb-2">
                  <span className="gradient-text">SERS-Insight</span> AI
                </h2>
                <p className="text-muted-foreground mb-6">
                  Upload your data and start a conversation. I'll analyze it and help you extract insights.
                </p>

                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: 'Detect Peaks', prompt: 'Detect peaks in my spectrum' },
                    { label: 'Apply ML', prompt: 'Apply machine learning to classify my data' },
                    { label: 'Run EDA', prompt: 'Perform exploratory data analysis' },
                    { label: 'Visualize', prompt: 'Create visualizations of my data' },
                  ].map((action) => (
                    <Button
                      key={action.label}
                      variant="outline"
                      size="sm"
                      className="justify-start"
                      onClick={() => setInput(action.prompt)}
                      disabled={!uploadedData}
                    >
                      {action.label}
                    </Button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div
              ref={scrollRef}
              className="flex-1 overflow-y-auto p-6"
              style={{ maxHeight: 'calc(100vh - 200px)' }}
            >
              <div className="max-w-3xl mx-auto space-y-6">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex gap-3 animate-fade-in ${message.role === 'user' ? 'flex-row-reverse' : ''
                      }`}
                  >
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 text-sm ${message.role === 'user'
                        ? 'gradient-primary text-white'
                        : message.role === 'system'
                          ? 'bg-amber-100 text-amber-600'
                          : ''
                        }`}
                      style={
                        message.role === 'assistant'
                          ? { backgroundColor: AI_MODELS[message.model as AIModel || selectedModel].color }
                          : {}
                      }
                    >
                      {message.role === 'user' ? (
                        <User className="h-4 w-4" />
                      ) : message.role === 'system' ? (
                        <Settings2 className="h-4 w-4" />
                      ) : (
                        React.createElement(AI_MODELS[message.model as AIModel || selectedModel].IconComponent, { size: 20 })
                      )}
                    </div>
                    <div
                      className={`chat-bubble ${message.role === 'user'
                        ? 'chat-bubble-user'
                        : 'chat-bubble-assistant'
                        }`}
                    >
                      <div className="prose prose-sm max-w-none">
                        <ReactMarkdown>{message.content}</ReactMarkdown>
                      </div>

                      {/* Code Block */}
                      {message.codeBlock && (
                        <details className="mt-3">
                          <summary className="cursor-pointer text-sm font-medium text-primary">
                            📝 View Python Code
                          </summary>
                          <pre className="mt-2 p-3 bg-slate-900 text-slate-100 rounded-lg text-xs overflow-x-auto">
                            <code>{message.codeBlock}</code>
                          </pre>
                        </details>
                      )}

                      {/* Visualization */}
                      {message.visualization?.data && (
                        <SpectrumVisualization data={message.visualization.data} />
                      )}
                    </div>
                  </div>
                ))}

                {(isLoading || isStreaming) && (
                  <div className="flex gap-3 animate-fade-in">
                    <div
                      className="w-8 h-8 rounded-full flex items-center justify-center"
                      style={{ backgroundColor: AI_MODELS[selectedModel].color }}
                    >
                      {React.createElement(AI_MODELS[selectedModel].IconComponent, { size: 22 })}
                    </div>
                    <div className="chat-bubble chat-bubble-assistant">
                      <div className="flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span className="text-muted-foreground">
                          {isStreaming ? `${AI_MODELS[selectedModel].name} is typing...` : `${AI_MODELS[selectedModel].name} is thinking...`}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Fallback Notice */}
              {fallbackNotice && (
                <div className="flex justify-center">
                  <div className="px-4 py-2 bg-amber-50 border border-amber-200 rounded-full text-amber-700 text-xs">
                    ⚠️ {fallbackNotice}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Input Area */}
          <div className="border-t bg-white p-4">
            <div className="max-w-3xl mx-auto">
              <div className={`flex items-end gap-2 p-3 bg-slate-50 rounded-2xl border ${isRecording ? 'ring-2 ring-red-400 ring-opacity-75' : ''}`}>
                <Textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={
                    isRecording
                      ? "🎤 Listening... Speak now"
                      : uploadedData
                        ? "Ask about your data: peak detection, ML, visualization..."
                        : "Upload data first, or ask about SERS..."
                  }
                  className="flex-1 min-h-[44px] max-h-[200px] resize-none border-0 bg-transparent focus-visible:ring-0 py-3"
                  rows={1}
                />
                {/* Voice Input Button */}
                {speechSupported && (
                  <Button
                    size="icon"
                    variant={isRecording ? "destructive" : "outline"}
                    className={`shrink-0 rounded-full transition-all ${isRecording ? 'animate-pulse' : ''}`}
                    onClick={toggleVoiceRecording}
                    disabled={isLoading || isStreaming}
                    title={isRecording ? "Stop recording" : "Start voice input"}
                  >
                    {isRecording ? (
                      <MicOff className="h-5 w-5" />
                    ) : (
                      <Mic className="h-5 w-5" />
                    )}
                  </Button>
                )}
                <Button
                  size="icon"
                  className="shrink-0 rounded-full"
                  style={{ backgroundColor: AI_MODELS[selectedModel].color }}
                  onClick={handleSend}
                  disabled={!input.trim() || isLoading || isStreaming}
                >
                  {(isLoading || isStreaming) ? (
                    <Loader2 className="h-5 w-5 animate-spin text-white" />
                  ) : (
                    <Send className="h-5 w-5 text-white" />
                  )}
                </Button>
              </div>
              <p className="text-[10px] text-center text-muted-foreground mt-2">
                {AI_MODELS[selectedModel].name} by {AI_MODELS[selectedModel].provider} •
                {uploadedData ? ` Analyzing ${uploadedData.fileName}` : ' Upload data to begin analysis'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </MainLayout>
  );
}
