import { create } from 'zustand';

export interface AnalysisResult {
    id: string;
    sessionId: string;
    framework: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    inputs: {
        datasetId: string;
        parameters: Record<string, unknown>;
    };
    outputs?: {
        predictions?: unknown;
        metrics?: Record<string, number>;
        visualizations?: string[];
        summary?: string;
    };
    error?: string;
    startedAt: Date;
    completedAt?: Date;
}

export interface Dataset {
    id: string;
    name: string;
    type: 'csv' | 'txt' | 'xlsx' | 'json' | 'image';
    size: number;
    uploadedAt: Date;
    metadata?: {
        columns?: string[];
        rows?: number;
        wavenumberRange?: [number, number];
        hasHeaders?: boolean;
    };
    preprocessed?: boolean;
    preprocessingConfig?: Record<string, unknown>;
}

export interface Framework {
    id: string;
    name: string;
    description: string;
    category: 'simulation' | 'detection' | 'classification' | 'unmixing' | 'custom';
    parameters: FrameworkParameter[];
    icon: string;
}

export interface FrameworkParameter {
    name: string;
    type: 'number' | 'string' | 'boolean' | 'select' | 'range';
    label: string;
    description?: string;
    defaultValue: unknown;
    options?: { label: string; value: unknown }[];
    min?: number;
    max?: number;
    step?: number;
}

interface AnalysisState {
    datasets: Dataset[];
    results: AnalysisResult[];
    frameworks: Framework[];
    selectedFramework: string | null;
    preprocessingOptions: {
        baselineCorrection: boolean;
        smoothing: boolean;
        normalization: 'none' | 'vector' | 'max' | 'minmax';
        peakDetection: boolean;
    };

    // Actions
    addDataset: (dataset: Dataset) => void;
    removeDataset: (datasetId: string) => void;
    updateDataset: (datasetId: string, updates: Partial<Dataset>) => void;
    addResult: (result: AnalysisResult) => void;
    updateResult: (resultId: string, updates: Partial<AnalysisResult>) => void;
    setSelectedFramework: (frameworkId: string | null) => void;
    setPreprocessingOptions: (options: Partial<AnalysisState['preprocessingOptions']>) => void;
    clearResults: () => void;
}

// Default available frameworks
const defaultFrameworks: Framework[] = [
    {
        id: 'lspr-simulation',
        name: 'LSPR Simulation',
        description: 'Simulate LSPR response for Ag/Au nanoparticles',
        category: 'simulation',
        icon: '🔬',
        parameters: [
            {
                name: 'material',
                type: 'select',
                label: 'Material',
                defaultValue: 'Ag',
                options: [
                    { label: 'Silver (Ag)', value: 'Ag' },
                    { label: 'Gold (Au)', value: 'Au' },
                ],
            },
            {
                name: 'nanoparticleSize',
                type: 'range',
                label: 'Nanoparticle Size (nm)',
                defaultValue: 50,
                min: 10,
                max: 200,
                step: 5,
            },
            {
                name: 'excitationWavelength',
                type: 'select',
                label: 'Excitation Wavelength',
                defaultValue: 785,
                options: [
                    { label: '532 nm', value: 532 },
                    { label: '633 nm', value: 633 },
                    { label: '785 nm', value: 785 },
                ],
            },
        ],
    },
    {
        id: 'molecule-detection',
        name: 'Molecule Detection',
        description: 'Detect and quantify known molecules (R6G, CV, etc.)',
        category: 'detection',
        icon: '🧪',
        parameters: [
            {
                name: 'targetMolecule',
                type: 'select',
                label: 'Target Molecule',
                defaultValue: 'R6G',
                options: [
                    { label: 'Rhodamine 6G (R6G)', value: 'R6G' },
                    { label: 'Crystal Violet', value: 'CV' },
                    { label: 'Nile Blue', value: 'NB' },
                    { label: 'Auto-detect', value: 'auto' },
                ],
            },
            {
                name: 'concentrationRegression',
                type: 'boolean',
                label: 'Enable Concentration Regression',
                defaultValue: true,
            },
        ],
    },
    {
        id: 'biomolecule-classification',
        name: 'Biomolecule Classification',
        description: 'Classify proteins, DNA, and other biomolecules',
        category: 'classification',
        icon: '🧬',
        parameters: [
            {
                name: 'classifier',
                type: 'select',
                label: 'Classifier',
                defaultValue: 'svm',
                options: [
                    { label: 'Support Vector Machine', value: 'svm' },
                    { label: 'Random Forest', value: 'rf' },
                    { label: 'XGBoost', value: 'xgboost' },
                ],
            },
            {
                name: 'pcaComponents',
                type: 'range',
                label: 'PCA Components',
                defaultValue: 5,
                min: 2,
                max: 20,
                step: 1,
            },
            {
                name: 'crossValidation',
                type: 'boolean',
                label: 'Enable Cross-Validation',
                defaultValue: true,
            },
        ],
    },
    {
        id: 'pathogen-detection',
        name: 'Pathogen Detection',
        description: 'Identify bacterial species using deep learning',
        category: 'classification',
        icon: '🦠',
        parameters: [
            {
                name: 'model',
                type: 'select',
                label: 'Model Architecture',
                defaultValue: 'cnn1d',
                options: [
                    { label: '1D CNN', value: 'cnn1d' },
                    { label: 'CNN + SVM Ensemble', value: 'ensemble' },
                ],
            },
            {
                name: 'dataAugmentation',
                type: 'boolean',
                label: 'Enable Data Augmentation',
                defaultValue: true,
            },
        ],
    },
    {
        id: 'spectral-unmixing',
        name: 'Spectral Unmixing',
        description: 'Separate mixed spectra into pure components',
        category: 'unmixing',
        icon: '📊',
        parameters: [
            {
                name: 'method',
                type: 'select',
                label: 'Unmixing Method',
                defaultValue: 'nmf',
                options: [
                    { label: 'Non-negative Matrix Factorization', value: 'nmf' },
                    { label: 'Independent Component Analysis', value: 'ica' },
                    { label: 'Classical Least Squares', value: 'cls' },
                ],
            },
            {
                name: 'numComponents',
                type: 'range',
                label: 'Number of Components',
                defaultValue: 3,
                min: 2,
                max: 10,
                step: 1,
            },
        ],
    },
];

export const useAnalysisStore = create<AnalysisState>((set) => ({
    datasets: [],
    results: [],
    frameworks: defaultFrameworks,
    selectedFramework: null,
    preprocessingOptions: {
        baselineCorrection: true,
        smoothing: true,
        normalization: 'vector',
        peakDetection: true,
    },

    addDataset: (dataset: Dataset) => {
        set((state) => ({
            datasets: [...state.datasets, dataset],
        }));
    },

    removeDataset: (datasetId: string) => {
        set((state) => ({
            datasets: state.datasets.filter((d) => d.id !== datasetId),
        }));
    },

    updateDataset: (datasetId: string, updates: Partial<Dataset>) => {
        set((state) => ({
            datasets: state.datasets.map((d) =>
                d.id === datasetId ? { ...d, ...updates } : d
            ),
        }));
    },

    addResult: (result: AnalysisResult) => {
        set((state) => ({
            results: [result, ...state.results],
        }));
    },

    updateResult: (resultId: string, updates: Partial<AnalysisResult>) => {
        set((state) => ({
            results: state.results.map((r) =>
                r.id === resultId ? { ...r, ...updates } : r
            ),
        }));
    },

    setSelectedFramework: (frameworkId: string | null) => {
        set({ selectedFramework: frameworkId });
    },

    setPreprocessingOptions: (options: Partial<AnalysisState['preprocessingOptions']>) => {
        set((state) => ({
            preprocessingOptions: { ...state.preprocessingOptions, ...options },
        }));
    },

    clearResults: () => {
        set({ results: [] });
    },
}));
