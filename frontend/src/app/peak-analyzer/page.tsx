'use client';

import React, { useState, useCallback, useMemo, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { MainLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Slider } from '@/components/ui/slider';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
    Activity, Upload, Download, FileText, CheckCircle2,
    Plus, Minus, Move, Target, Loader2, Info, ChevronRight,
    RotateCcw, TrendingDown, Waves, Table2, ArrowRight,
    Eye, EyeOff, X, FileSpreadsheet,
} from 'lucide-react';

// Types
interface AnchorPoint {
    id: string;
    x: number;
    y: number;
}

interface Peak {
    wavenumber: number;
    intensity: number;
    fwhm?: number;
    area?: number;
}

interface RawCSVData {
    headers: string[];
    rows: (string | number)[][];
    numericColumns: number[];
}

type AnalyzerMode = 'view' | 'add' | 'modify' | 'delete';
type InterpolationMethod = 'linear' | 'bspline' | 'polynomial';

// B-Spline basis function
function bsplineBasis(i: number, k: number, t: number, knots: number[]): number {
    if (k === 1) {
        return (knots[i] <= t && t < knots[i + 1]) ? 1 : 0;
    }
    const d1 = knots[i + k - 1] - knots[i];
    const d2 = knots[i + k] - knots[i + 1];
    let c1 = 0, c2 = 0;
    if (d1 !== 0) c1 = ((t - knots[i]) / d1) * bsplineBasis(i, k - 1, t, knots);
    if (d2 !== 0) c2 = ((knots[i + k] - t) / d2) * bsplineBasis(i + 1, k - 1, t, knots);
    return c1 + c2;
}

// B-Spline interpolation
function bsplineInterpolation(anchors: AnchorPoint[], xValues: number[]): number[] {
    if (anchors.length < 2) return xValues.map(() => 0);
    const sorted = [...anchors].sort((a, b) => a.x - b.x);
    const n = sorted.length;
    const k = Math.min(4, n);
    const knots: number[] = [];
    for (let i = 0; i < k; i++) knots.push(sorted[0].x);
    for (let i = 1; i < n - 1; i++) knots.push(sorted[i].x);
    for (let i = 0; i < k; i++) knots.push(sorted[n - 1].x);

    return xValues.map(xVal => {
        if (xVal <= sorted[0].x) return sorted[0].y;
        if (xVal >= sorted[n - 1].x) return sorted[n - 1].y;
        let yVal = 0, sumBasis = 0;
        for (let i = 0; i < n; i++) {
            const basis = bsplineBasis(i, k, xVal, knots);
            yVal += sorted[i].y * basis;
            sumBasis += basis;
        }
        return sumBasis > 0 ? yVal / sumBasis : 0;
    });
}

// Linear interpolation
function linearInterpolation(anchors: AnchorPoint[], xValues: number[]): number[] {
    if (anchors.length < 2) return xValues.map(() => 0);
    const sorted = [...anchors].sort((a, b) => a.x - b.x);
    return xValues.map(xVal => {
        if (xVal <= sorted[0].x) return sorted[0].y;
        if (xVal >= sorted[sorted.length - 1].x) return sorted[sorted.length - 1].y;
        for (let i = 0; i < sorted.length - 1; i++) {
            if (xVal >= sorted[i].x && xVal <= sorted[i + 1].x) {
                const t = (xVal - sorted[i].x) / (sorted[i + 1].x - sorted[i].x);
                return sorted[i].y + t * (sorted[i + 1].y - sorted[i].y);
            }
        }
        return 0;
    });
}

// Polynomial interpolation (simplified linear fit)
function polynomialInterpolation(anchors: AnchorPoint[], xValues: number[]): number[] {
    if (anchors.length < 2) return xValues.map(() => 0);
    const sorted = [...anchors].sort((a, b) => a.x - b.x);
    const xMean = sorted.reduce((s, p) => s + p.x, 0) / sorted.length;
    const yMean = sorted.reduce((s, p) => s + p.y, 0) / sorted.length;
    let num = 0, den = 0;
    for (const p of sorted) {
        num += (p.x - xMean) * (p.y - yMean);
        den += (p.x - xMean) ** 2;
    }
    const slope = den !== 0 ? num / den : 0;
    const intercept = yMean - slope * xMean;
    return xValues.map(xVal => intercept + slope * xVal);
}

// Peak detection
function detectPeaksAdvanced(x: number[], y: number[], prominence: number = 0.15): Peak[] {
    const peaks: Peak[] = [];
    const maxY = Math.max(...y);
    const threshold = maxY * prominence;

    for (let i = 3; i < y.length - 3; i++) {
        const isMax = y[i] > y[i - 1] && y[i] > y[i + 1] &&
            y[i] > y[i - 2] && y[i] > y[i + 2] &&
            y[i] > y[i - 3] && y[i] > y[i + 3];

        if (isMax && y[i] > threshold) {
            const halfMax = y[i] / 2;
            let leftIdx = i, rightIdx = i;
            while (leftIdx > 0 && y[leftIdx] > halfMax) leftIdx--;
            while (rightIdx < y.length - 1 && y[rightIdx] > halfMax) rightIdx++;
            const fwhm = x[rightIdx] - x[leftIdx];

            let area = 0;
            for (let j = leftIdx; j < rightIdx; j++) {
                area += (y[j] + y[j + 1]) / 2 * (x[j + 1] - x[j]);
            }

            peaks.push({ wavenumber: x[i], intensity: y[i], fwhm, area });
        }
    }
    return peaks.sort((a, b) => b.intensity - a.intensity).slice(0, 15);
}

// Auto-generate anchor points
function autoGenerateAnchors(x: number[], y: number[], count: number = 10): AnchorPoint[] {
    const anchors: AnchorPoint[] = [];
    const step = Math.floor(x.length / (count + 1));

    for (let i = 0; i <= count; i++) {
        const idx = Math.min(i * step, x.length - 1);
        const windowSize = Math.floor(step / 2);
        const start = Math.max(0, idx - windowSize);
        const end = Math.min(x.length - 1, idx + windowSize);

        let minIdx = start;
        for (let j = start; j <= end; j++) {
            if (y[j] < y[minIdx]) minIdx = j;
        }

        anchors.push({ id: `anchor-${Date.now()}-${i}`, x: x[minIdx], y: y[minIdx] });
    }
    return anchors;
}

// Generate demo spectrum
function generateDemoSpectrum(type: 'xrd' | 'sers' | 'raman'): { x: number[]; y: number[] } {
    const x: number[] = [], y: number[] = [];
    if (type === 'xrd') {
        for (let theta = 10; theta <= 80; theta += 0.1) {
            x.push(theta);
            let intensity = 100 + 50 * Math.exp(-((theta - 45) ** 2) / 1000) + Math.random() * 20;
            const peaks = [
                { pos: 21.5, height: 800, width: 0.3 }, { pos: 26.5, height: 1500, width: 0.25 },
                { pos: 36.5, height: 600, width: 0.35 }, { pos: 50.1, height: 700, width: 0.4 },
            ];
            for (const p of peaks) intensity += p.height * Math.exp(-((theta - p.pos) ** 2) / (2 * p.width ** 2));
            y.push(intensity);
        }
    } else {
        for (let w = 200; w <= 2000; w += 2) {
            x.push(w);
            let intensity = Math.random() * 0.05 + 0.2 * Math.exp(-((w - 1000) ** 2) / 300000);
            const peaks = [
                { pos: 611, height: 0.7, width: 15 }, { pos: 773, height: 0.5, width: 12 },
                { pos: 1003, height: 0.85, width: 14 }, { pos: 1363, height: 0.8, width: 15 },
                { pos: 1509, height: 1.0, width: 18 }, { pos: 1649, height: 0.65, width: 16 },
            ];
            for (const p of peaks) intensity += p.height * Math.exp(-((w - p.pos) ** 2) / (2 * p.width ** 2));
            y.push(intensity);
        }
    }
    return { x, y };
}

export default function PeakAnalyzerPage() {
    // Raw CSV data for preview
    const [rawCSVData, setRawCSVData] = useState < RawCSVData | null > (null);
    const [fileName, setFileName] = useState < string > ('');
    const [showDataPreview, setShowDataPreview] = useState(true);

    // Column selection
    const [xColumn, setXColumn] = useState < number | null > (null);
    const [yColumn, setYColumn] = useState < number | null > (null);
    const [columnsConfirmed, setColumnsConfirmed] = useState(false);

    // Processed spectrum data
    const [rawData, setRawData] = useState < { x: number[]; y: number[] } | null > (null);

    // Analyzer settings
    const [mode, setMode] = useState < AnalyzerMode > ('modify');
    const [interpolation, setInterpolation] = useState < InterpolationMethod > ('bspline');
    const [anchorPoints, setAnchorPoints] = useState < AnchorPoint[] > ([]);
    const [selectedAnchor, setSelectedAnchor] = useState < string | null > (null);
    const [step, setStep] = useState < number > (1);

    // Processing state
    const [isProcessing, setIsProcessing] = useState(false);
    const [baselineCorrectedData, setBaselineCorrectedData] = useState < { x: number[]; y: number[] } | null > (null);
    const [detectedPeaks, setDetectedPeaks] = useState < Peak[] > ([]);
    const [peakProminence, setPeakProminence] = useState(0.15);

    // SVG interaction
    const svgRef = useRef < SVGSVGElement > (null);
    const [isDragging, setIsDragging] = useState(false);

    const width = 800, height = 400;
    const padding = { top: 40, right: 30, bottom: 60, left: 70 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Parse CSV/TXT file
    const parseCSVFile = (text: string): RawCSVData => {
        const lines = text.trim().split('\n');
        const rows: (string | number)[][] = [];
        let headers: string[] = [];

        // Detect delimiter
        const firstLine = lines[0];
        const delimiter = firstLine.includes('\t') ? '\t' : firstLine.includes(',') ? ',' : /\s+/;

        // Parse all lines
        for (let i = 0; i < lines.length; i++) {
            const parts = lines[i].split(delimiter).map(p => p.trim()).filter(p => p);
            if (parts.length === 0) continue;

            // If first row contains non-numeric values, treat as headers
            if (i === 0 && parts.some(p => isNaN(parseFloat(p)))) {
                headers = parts;
            } else {
                const row = parts.map(p => {
                    const num = parseFloat(p);
                    return isNaN(num) ? p : num;
                });
                rows.push(row);
            }
        }

        // If no headers, generate column names
        if (headers.length === 0 && rows.length > 0) {
            headers = rows[0].map((_, i) => `Column ${i + 1}`);
        }

        // Identify numeric columns
        const numericColumns: number[] = [];
        if (rows.length > 0) {
            for (let col = 0; col < rows[0].length; col++) {
                const allNumeric = rows.slice(0, Math.min(10, rows.length)).every(row =>
                    row[col] !== undefined && typeof row[col] === 'number'
                );
                if (allNumeric) numericColumns.push(col);
            }
        }

        return { headers, rows, numericColumns };
    };

    // File upload handler
    const onDrop = useCallback((acceptedFiles: File[]) => {
        const file = acceptedFiles[0];
        if (!file) return;
        setFileName(file.name);
        setColumnsConfirmed(false);
        setRawData(null);
        setBaselineCorrectedData(null);
        setDetectedPeaks([]);
        setStep(1);

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const text = e.target?.result as string;
                const csvData = parseCSVFile(text);
                setRawCSVData(csvData);
                setShowDataPreview(true);

                // Auto-select first two numeric columns
                if (csvData.numericColumns.length >= 2) {
                    setXColumn(csvData.numericColumns[0]);
                    setYColumn(csvData.numericColumns[1]);
                } else if (csvData.numericColumns.length === 1) {
                    setXColumn(null);
                    setYColumn(csvData.numericColumns[0]);
                }
            } catch {
                console.error('Failed to parse file');
            }
        };
        reader.readAsText(file);
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'text/csv': ['.csv'], 'text/plain': ['.txt'] },
    });

    // Confirm column selection and extract data
    const confirmColumnSelection = () => {
        if (!rawCSVData || xColumn === null || yColumn === null) return;

        const xArr: number[] = [];
        const yArr: number[] = [];

        for (const row of rawCSVData.rows) {
            const xVal = row[xColumn];
            const yVal = row[yColumn];
            if (typeof xVal === 'number' && typeof yVal === 'number' && !isNaN(xVal) && !isNaN(yVal)) {
                xArr.push(xVal);
                yArr.push(yVal);
            }
        }

        if (xArr.length > 0) {
            setRawData({ x: xArr, y: yArr });
            setAnchorPoints(autoGenerateAnchors(xArr, yArr));
            setColumnsConfirmed(true);
            setShowDataPreview(false);
            setStep(2);
        }
    };

    // Load demo data
    const loadDemo = (type: 'xrd' | 'sers' | 'raman') => {
        const data = generateDemoSpectrum(type);
        setRawData(data);
        setFileName(`${type.toUpperCase()}_demo.csv`);
        setAnchorPoints(autoGenerateAnchors(data.x, data.y));
        setRawCSVData(null);
        setColumnsConfirmed(true);
        setShowDataPreview(false);
        setStep(2);
    };

    // Clear all data
    const clearData = () => {
        setRawCSVData(null);
        setRawData(null);
        setFileName('');
        setXColumn(null);
        setYColumn(null);
        setColumnsConfirmed(false);
        setShowDataPreview(true);
        setAnchorPoints([]);
        setBaselineCorrectedData(null);
        setDetectedPeaks([]);
        setStep(1);
    };

    // Calculate baseline
    const baseline = useMemo(() => {
        if (!rawData || anchorPoints.length < 2) return null;
        switch (interpolation) {
            case 'bspline': return bsplineInterpolation(anchorPoints, rawData.x);
            case 'polynomial': return polynomialInterpolation(anchorPoints, rawData.x);
            default: return linearInterpolation(anchorPoints, rawData.x);
        }
    }, [rawData, anchorPoints, interpolation]);

    // Scales
    const xScale = useMemo(() => {
        if (!rawData) return (v: number) => 0;
        const min = Math.min(...rawData.x), max = Math.max(...rawData.x);
        return (v: number) => padding.left + ((v - min) / (max - min)) * plotWidth;
    }, [rawData, plotWidth]);

    const yScale = useMemo(() => {
        if (!rawData) return (v: number) => 0;
        const max = Math.max(...rawData.y) * 1.1;
        return (v: number) => padding.top + (1 - v / max) * plotHeight;
    }, [rawData, plotHeight]);

    const inverseXScale = useMemo(() => {
        if (!rawData) return (px: number) => 0;
        const min = Math.min(...rawData.x), max = Math.max(...rawData.x);
        return (px: number) => min + ((px - padding.left) / plotWidth) * (max - min);
    }, [rawData, plotWidth]);

    const inverseYScale = useMemo(() => {
        if (!rawData) return (py: number) => 0;
        const max = Math.max(...rawData.y) * 1.1;
        return (py: number) => (1 - (py - padding.top) / plotHeight) * max;
    }, [rawData, plotHeight]);

    // SVG click handler
    const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
        if (!rawData || mode !== 'add') return;
        const svg = svgRef.current;
        if (!svg) return;
        const rect = svg.getBoundingClientRect();
        const px = (e.clientX - rect.left) * (width / rect.width);
        const py = (e.clientY - rect.top) * (height / rect.height);
        if (px < padding.left || px > width - padding.right) return;
        if (py < padding.top || py > height - padding.bottom) return;
        setAnchorPoints(prev => [...prev, {
            id: `anchor-${Date.now()}`,
            x: inverseXScale(px),
            y: inverseYScale(py),
        }]);
    };

    const handleAnchorMouseDown = (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (mode === 'delete') {
            setAnchorPoints(prev => prev.filter(p => p.id !== id));
        } else if (mode === 'modify') {
            setSelectedAnchor(id);
            setIsDragging(true);
        }
    };

    const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
        if (!isDragging || !selectedAnchor || mode !== 'modify') return;
        const svg = svgRef.current;
        if (!svg) return;
        const rect = svg.getBoundingClientRect();
        const px = (e.clientX - rect.left) * (width / rect.width);
        const py = (e.clientY - rect.top) * (height / rect.height);
        setAnchorPoints(prev => prev.map(p =>
            p.id === selectedAnchor ? { ...p, x: inverseXScale(px), y: inverseYScale(py) } : p
        ));
    };

    const handleMouseUp = () => { setIsDragging(false); setSelectedAnchor(null); };

    // Subtract baseline
    const subtractBaseline = async () => {
        if (!rawData || !baseline) return;
        setIsProcessing(true);
        await new Promise(r => setTimeout(r, 500));
        const correctedY = rawData.y.map((y, i) => Math.max(0, y - baseline[i]));
        setBaselineCorrectedData({ x: rawData.x, y: correctedY });
        const peaks = detectPeaksAdvanced(rawData.x, correctedY, peakProminence);
        setDetectedPeaks(peaks);
        setIsProcessing(false);
        setStep(4);
    };

    // Reset
    const resetAnalysis = () => {
        if (rawData) setAnchorPoints(autoGenerateAnchors(rawData.x, rawData.y));
        setBaselineCorrectedData(null);
        setDetectedPeaks([]);
        setStep(2);
    };

    // Export
    const exportData = () => {
        if (!baselineCorrectedData) return;
        let csv = 'X,Original_Y,Baseline,Corrected_Y\n';
        for (let i = 0; i < rawData!.x.length; i++) {
            csv += `${rawData!.x[i]},${rawData!.y[i]},${baseline![i]},${baselineCorrectedData.y[i]}\n`;
        }
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'baseline_corrected_data.csv';
        a.click();
    };

    // Paths
    const spectrumPath = useMemo(() => {
        if (!rawData) return '';
        return rawData.y.map((v, i) => {
            const x = xScale(rawData.x[i]);
            const y = yScale(v);
            return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
        }).join(' ');
    }, [rawData, xScale, yScale]);

    const baselinePath = useMemo(() => {
        if (!rawData || !baseline) return '';
        return baseline.map((v, i) => {
            const x = xScale(rawData.x[i]);
            const y = yScale(v);
            return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
        }).join(' ');
    }, [rawData, baseline, xScale, yScale]);

    return (
        <MainLayout>
            <div className="flex flex-col h-full overflow-auto bg-gradient-to-br from-slate-50 via-violet-50/30 to-white">
                {/* Header */}
                <div className="p-6 border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="p-2.5 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 shadow-lg shadow-violet-200">
                                <Waves className="h-6 w-6 text-white" />
                            </div>
                            <div>
                                <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-violet-600 to-purple-600">
                                    Peak Analyzer
                                </h1>
                                <p className="text-muted-foreground">
                                    Interactive baseline correction with column selection
                                </p>
                            </div>
                        </div>

                        {/* Step indicator */}
                        <div className="flex items-center gap-2">
                            {['Load Data', 'Configure', 'Process', 'Export'].map((label, i) => (
                                <div key={i} className="flex items-center">
                                    <div className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${step > i ? 'bg-violet-500 text-white' : step === i + 1 ? 'bg-violet-100 border-2 border-violet-500 text-violet-700' : 'bg-slate-100 text-slate-400'}`}>
                                        {label}
                                    </div>
                                    {i < 3 && <ChevronRight className={`h-4 w-4 mx-1 ${step > i + 1 ? 'text-violet-500' : 'text-slate-300'}`} />}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="flex-1 p-6">
                    <div className="grid lg:grid-cols-4 gap-6 max-w-[1700px] mx-auto">
                        {/* Left Panel - Controls */}
                        <div className="space-y-4">
                            {/* Step 1: Load Data */}
                            <Card className={`border-2 transition-all ${step === 1 ? 'border-violet-400 shadow-lg shadow-violet-100' : 'border-slate-200'}`}>
                                <CardHeader className="py-3">
                                    <CardTitle className="text-sm flex items-center gap-2">
                                        <span className={`w-6 h-6 rounded-full text-xs flex items-center justify-center font-bold ${step >= 1 ? 'bg-violet-500 text-white' : 'bg-slate-200'}`}>1</span>
                                        Load Data
                                    </CardTitle>
                                    <CardDescription className="text-xs">Upload CSV/TXT or use demo spectra</CardDescription>
                                </CardHeader>
                                <CardContent className="space-y-3">
                                    <div
                                        {...getRootProps()}
                                        className={`p-5 border-2 border-dashed rounded-xl text-center cursor-pointer transition-all
                                            ${isDragActive ? 'border-violet-500 bg-violet-50' : 'border-slate-200 hover:border-violet-300 hover:bg-violet-50/50'}`}
                                    >
                                        <input {...getInputProps()} />
                                        <Upload className="h-8 w-8 mx-auto mb-2 text-violet-500" />
                                        <p className="text-sm font-medium text-slate-700">Drop CSV/TXT file here</p>
                                        <p className="text-xs text-muted-foreground mt-1">or click to browse</p>
                                    </div>

                                    {fileName && (
                                        <div className="flex items-center gap-2 p-3 bg-gradient-to-r from-emerald-50 to-green-50 border border-emerald-200 rounded-lg">
                                            <FileSpreadsheet className="h-5 w-5 text-emerald-600" />
                                            <span className="text-sm text-emerald-800 flex-1 truncate font-medium">{fileName}</span>
                                            <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={clearData}>
                                                <X className="h-4 w-4 text-slate-400" />
                                            </Button>
                                        </div>
                                    )}

                                    <div>
                                        <p className="text-xs font-semibold mb-2 text-slate-500 uppercase tracking-wide">Demo Spectra</p>
                                        <div className="grid grid-cols-3 gap-2">
                                            <Button variant="outline" size="sm" className="text-xs h-9 hover:bg-violet-50 hover:border-violet-300" onClick={() => loadDemo('xrd')}>
                                                XRD
                                            </Button>
                                            <Button variant="outline" size="sm" className="text-xs h-9 hover:bg-violet-50 hover:border-violet-300" onClick={() => loadDemo('sers')}>
                                                SERS
                                            </Button>
                                            <Button variant="outline" size="sm" className="text-xs h-9 hover:bg-violet-50 hover:border-violet-300" onClick={() => loadDemo('raman')}>
                                                Raman
                                            </Button>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Step 2: Configure */}
                            <Card className={`border-2 transition-all ${step === 2 ? 'border-violet-400 shadow-lg shadow-violet-100' : 'border-slate-200'}`}>
                                <CardHeader className="py-3">
                                    <CardTitle className="text-sm flex items-center gap-2">
                                        <span className={`w-6 h-6 rounded-full text-xs flex items-center justify-center font-bold ${step >= 2 ? 'bg-violet-500 text-white' : 'bg-slate-200'}`}>2</span>
                                        Configure Baseline
                                    </CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    <div className="space-y-2">
                                        <Label className="text-xs font-medium">Interpolation Method</Label>
                                        <Select value={interpolation} onValueChange={(v) => setInterpolation(v as InterpolationMethod)}>
                                            <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="bspline">✨ BSpline (Recommended)</SelectItem>
                                                <SelectItem value="linear">Linear</SelectItem>
                                                <SelectItem value="polynomial">Polynomial</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>

                                    <Separator />

                                    <div className="space-y-2">
                                        <Label className="text-xs font-medium">Anchor Point Tools</Label>
                                        <div className="grid grid-cols-2 gap-2">
                                            <Button variant={mode === 'modify' ? 'default' : 'outline'} size="sm" className="h-9 gap-1.5" onClick={() => setMode('modify')}>
                                                <Move className="h-4 w-4" /> Drag
                                            </Button>
                                            <Button variant={mode === 'add' ? 'default' : 'outline'} size="sm" className="h-9 gap-1.5" onClick={() => setMode('add')}>
                                                <Plus className="h-4 w-4" /> Add
                                            </Button>
                                            <Button variant={mode === 'delete' ? 'destructive' : 'outline'} size="sm" className="h-9 gap-1.5" onClick={() => setMode('delete')}>
                                                <Minus className="h-4 w-4" /> Delete
                                            </Button>
                                            <Button variant="outline" size="sm" className="h-9 gap-1.5" onClick={resetAnalysis}>
                                                <RotateCcw className="h-4 w-4" /> Reset
                                            </Button>
                                        </div>
                                    </div>

                                    <div className="p-3 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-100">
                                        <div className="flex items-start gap-2">
                                            <Info className="h-4 w-4 text-blue-500 mt-0.5" />
                                            <p className="text-xs text-blue-700">
                                                {mode === 'add' && 'Click on the graph to add new anchor points'}
                                                {mode === 'modify' && 'Drag anchor points to adjust the baseline curve'}
                                                {mode === 'delete' && 'Click on anchor points to remove them'}
                                            </p>
                                        </div>
                                    </div>

                                    <div className="flex items-center justify-between text-sm">
                                        <span className="text-muted-foreground">Anchor Points:</span>
                                        <Badge variant="secondary" className="bg-violet-100 text-violet-700">{anchorPoints.length}</Badge>
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Step 3: Process */}
                            <Card className={`border-2 transition-all ${step === 3 ? 'border-violet-400 shadow-lg shadow-violet-100' : 'border-slate-200'}`}>
                                <CardHeader className="py-3">
                                    <CardTitle className="text-sm flex items-center gap-2">
                                        <span className={`w-6 h-6 rounded-full text-xs flex items-center justify-center font-bold ${step >= 3 ? 'bg-violet-500 text-white' : 'bg-slate-200'}`}>3</span>
                                        Process
                                    </CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-3">
                                    <div className="space-y-2">
                                        <Label className="text-xs">Peak Prominence: {(peakProminence * 100).toFixed(0)}%</Label>
                                        <Slider value={[peakProminence * 100]} onValueChange={([v]) => setPeakProminence(v / 100)} min={5} max={50} step={1} />
                                    </div>

                                    <Button className="w-full bg-gradient-to-r from-violet-500 to-purple-600 text-white shadow-lg shadow-violet-200 hover:shadow-xl hover:shadow-violet-300 transition-all" onClick={() => { setStep(3); subtractBaseline(); }} disabled={!rawData || anchorPoints.length < 2 || isProcessing}>
                                        {isProcessing ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Processing...</> : <><TrendingDown className="h-4 w-4 mr-2" />Subtract Baseline</>}
                                    </Button>
                                </CardContent>
                            </Card>

                            {/* Step 4: Export */}
                            <Card className={`border-2 transition-all ${step === 4 ? 'border-emerald-400 shadow-lg shadow-emerald-100' : 'border-slate-200'}`}>
                                <CardHeader className="py-3">
                                    <CardTitle className="text-sm flex items-center gap-2">
                                        <span className={`w-6 h-6 rounded-full text-xs flex items-center justify-center font-bold ${step >= 4 ? 'bg-emerald-500 text-white' : 'bg-slate-200'}`}>4</span>
                                        Export Results
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <Button variant="outline" className="w-full border-emerald-300 text-emerald-700 hover:bg-emerald-50" onClick={exportData} disabled={!baselineCorrectedData}>
                                        <Download className="h-4 w-4 mr-2" /> Download CSV
                                    </Button>
                                </CardContent>
                            </Card>
                        </div>

                        {/* Main Content Area */}
                        <div className="lg:col-span-3 space-y-4">
                            {/* Data Preview / Column Selection */}
                            {rawCSVData && !columnsConfirmed && (
                                <Card className="border-2 border-violet-300 shadow-lg">
                                    <CardHeader className="py-4 bg-gradient-to-r from-violet-50 to-purple-50">
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-2">
                                                <Table2 className="h-5 w-5 text-violet-600" />
                                                <CardTitle className="text-base">Data Preview & Column Selection</CardTitle>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <Badge variant="outline">{rawCSVData.rows.length} rows × {rawCSVData.headers.length} columns</Badge>
                                                <Button variant="ghost" size="sm" onClick={() => setShowDataPreview(!showDataPreview)}>
                                                    {showDataPreview ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                                                </Button>
                                            </div>
                                        </div>
                                        <CardDescription>Select the X (wavenumber) and Y (intensity) columns for analysis</CardDescription>
                                    </CardHeader>
                                    <CardContent className="p-0">
                                        {/* Column Selection Bar */}
                                        <div className="p-4 bg-slate-50 border-b flex items-center gap-6 flex-wrap">
                                            <div className="flex items-center gap-3">
                                                <Label className="text-sm font-medium whitespace-nowrap">X-Axis (Wavenumber):</Label>
                                                <Select value={xColumn?.toString() ?? ''} onValueChange={(v) => setXColumn(parseInt(v))}>
                                                    <SelectTrigger className="w-48 h-9 bg-white">
                                                        <SelectValue placeholder="Select column..." />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        {rawCSVData.numericColumns.map(col => (
                                                            <SelectItem key={col} value={col.toString()} disabled={col === yColumn}>
                                                                {rawCSVData.headers[col] || `Column ${col + 1}`}
                                                            </SelectItem>
                                                        ))}
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                            <div className="flex items-center gap-3">
                                                <Label className="text-sm font-medium whitespace-nowrap">Y-Axis (Intensity):</Label>
                                                <Select value={yColumn?.toString() ?? ''} onValueChange={(v) => setYColumn(parseInt(v))}>
                                                    <SelectTrigger className="w-48 h-9 bg-white">
                                                        <SelectValue placeholder="Select column..." />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        {rawCSVData.numericColumns.map(col => (
                                                            <SelectItem key={col} value={col.toString()} disabled={col === xColumn}>
                                                                {rawCSVData.headers[col] || `Column ${col + 1}`}
                                                            </SelectItem>
                                                        ))}
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                            <Button className="ml-auto bg-violet-600 hover:bg-violet-700" onClick={confirmColumnSelection} disabled={xColumn === null || yColumn === null}>
                                                <ArrowRight className="h-4 w-4 mr-2" /> Confirm & Continue
                                            </Button>
                                        </div>

                                        {/* Data Table */}
                                        {showDataPreview && (
                                            <ScrollArea className="h-64">
                                                <table className="w-full text-sm">
                                                    <thead className="bg-slate-100 sticky top-0">
                                                        <tr>
                                                            <th className="px-3 py-2 text-left text-xs font-semibold text-slate-500 w-12">#</th>
                                                            {rawCSVData.headers.map((header, i) => (
                                                                <th key={i} className={`px-3 py-2 text-left text-xs font-semibold cursor-pointer transition-colors ${i === xColumn ? 'bg-blue-100 text-blue-700' : i === yColumn ? 'bg-green-100 text-green-700' : 'text-slate-500 hover:bg-slate-200'}`}
                                                                    onClick={() => {
                                                                        if (!rawCSVData.numericColumns.includes(i)) return;
                                                                        if (xColumn === null) setXColumn(i);
                                                                        else if (yColumn === null && i !== xColumn) setYColumn(i);
                                                                        else if (i === xColumn) setXColumn(null);
                                                                        else if (i === yColumn) setYColumn(null);
                                                                    }}
                                                                >
                                                                    <div className="flex items-center gap-1">
                                                                        {header}
                                                                        {i === xColumn && <Badge className="ml-1 h-4 text-[10px] bg-blue-500">X</Badge>}
                                                                        {i === yColumn && <Badge className="ml-1 h-4 text-[10px] bg-green-500">Y</Badge>}
                                                                        {!rawCSVData.numericColumns.includes(i) && <span className="text-orange-500 text-[10px]">(text)</span>}
                                                                    </div>
                                                                </th>
                                                            ))}
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {rawCSVData.rows.slice(0, 50).map((row, rowIdx) => (
                                                            <tr key={rowIdx} className="border-b border-slate-100 hover:bg-slate-50">
                                                                <td className="px-3 py-1.5 text-xs text-slate-400">{rowIdx + 1}</td>
                                                                {row.map((cell, colIdx) => (
                                                                    <td key={colIdx} className={`px-3 py-1.5 text-xs font-mono ${colIdx === xColumn ? 'bg-blue-50 text-blue-700' : colIdx === yColumn ? 'bg-green-50 text-green-700' : ''}`}>
                                                                        {typeof cell === 'number' ? cell.toPrecision(6) : cell}
                                                                    </td>
                                                                ))}
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                                {rawCSVData.rows.length > 50 && (
                                                    <div className="p-3 text-center text-xs text-muted-foreground bg-slate-50">
                                                        Showing first 50 of {rawCSVData.rows.length} rows
                                                    </div>
                                                )}
                                            </ScrollArea>
                                        )}
                                    </CardContent>
                                </Card>
                            )}

                            {/* Main Graph */}
                            <Card className="border-2">
                                <CardHeader className="py-3 bg-gradient-to-r from-slate-50 to-white">
                                    <div className="flex items-center justify-between">
                                        <CardTitle className="text-base flex items-center gap-2">
                                            <Activity className="h-5 w-5 text-violet-500" />
                                            Interactive Baseline Editor
                                        </CardTitle>
                                        <div className="flex items-center gap-4 text-xs">
                                            <div className="flex items-center gap-1.5">
                                                <div className="w-4 h-0.5 bg-blue-500 rounded" />
                                                <span className="text-muted-foreground">Original</span>
                                            </div>
                                            <div className="flex items-center gap-1.5">
                                                <div className="w-4 h-0.5 bg-red-500 rounded" style={{ backgroundImage: 'repeating-linear-gradient(90deg, #ef4444, #ef4444 3px, transparent 3px, transparent 6px)' }} />
                                                <span className="text-muted-foreground">Baseline</span>
                                            </div>
                                            <div className="flex items-center gap-1.5">
                                                <div className="w-3 h-3 bg-red-500 rounded-full border-2 border-white shadow" />
                                                <span className="text-muted-foreground">Anchor</span>
                                            </div>
                                        </div>
                                    </div>
                                </CardHeader>
                                <CardContent>
                                    {rawData ? (
                                        <svg ref={svgRef} width="100%" viewBox={`0 0 ${width} ${height}`}
                                            className={`bg-white rounded-lg border cursor-${mode === 'add' ? 'crosshair' : mode === 'modify' ? 'move' : 'default'}`}
                                            onClick={handleSvgClick} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
                                            {/* Grid */}
                                            {[0.2, 0.4, 0.6, 0.8].map(v => (
                                                <line key={v} x1={padding.left} y1={padding.top + v * plotHeight} x2={width - padding.right} y2={padding.top + v * plotHeight} stroke="#f1f5f9" strokeWidth={1} />
                                            ))}
                                            {/* Spectrum */}
                                            <path d={spectrumPath} fill="none" stroke="#3b82f6" strokeWidth={2} />
                                            {/* Baseline */}
                                            {baseline && <path d={baselinePath} fill="none" stroke="#ef4444" strokeWidth={2} strokeDasharray="8,4" />}
                                            {/* Anchor points */}
                                            {anchorPoints.map(point => (
                                                <circle key={point.id} cx={xScale(point.x)} cy={yScale(point.y)} r={selectedAnchor === point.id ? 10 : 7} fill="#ef4444" stroke="white" strokeWidth={2.5}
                                                    className="cursor-pointer transition-all hover:scale-125"
                                                    onMouseDown={(e) => handleAnchorMouseDown(point.id, e)}
                                                    style={{ filter: selectedAnchor === point.id ? 'drop-shadow(0 0 6px rgba(239, 68, 68, 0.6))' : 'drop-shadow(0 2px 3px rgba(0,0,0,0.1))' }} />
                                            ))}
                                            {/* Axes */}
                                            <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="#94a3b8" strokeWidth={1.5} />
                                            <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="#94a3b8" strokeWidth={1.5} />
                                            {/* Labels */}
                                            <text x={width / 2} y={height - 15} textAnchor="middle" fill="#64748b" fontSize={12} fontWeight={500}>
                                                {rawData.x[0] < 100 ? 'Angle (2θ)' : 'Wavenumber (cm⁻¹)'}
                                            </text>
                                            <text x={20} y={height / 2} textAnchor="middle" fill="#64748b" fontSize={12} fontWeight={500} transform={`rotate(-90, 20, ${height / 2})`}>
                                                Intensity
                                            </text>
                                        </svg>
                                    ) : (
                                        <div className="h-80 flex flex-col items-center justify-center bg-gradient-to-b from-white to-slate-50 rounded-lg border-2 border-dashed border-slate-200">
                                            <Waves className="h-20 w-20 text-slate-200 mb-4" />
                                            <p className="text-lg font-medium text-muted-foreground">Load data to begin</p>
                                            <p className="text-sm text-muted-foreground/70 mt-1">Upload a CSV file or use demo spectra</p>
                                        </div>
                                    )}
                                </CardContent>
                            </Card>

                            {/* Results */}
                            {baselineCorrectedData && (
                                <div className="grid md:grid-cols-2 gap-4">
                                    <Card className="border-2 border-emerald-200 bg-gradient-to-br from-white to-emerald-50/30">
                                        <CardHeader className="py-3">
                                            <CardTitle className="text-sm flex items-center gap-2">
                                                <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                                                Baseline-Corrected Spectrum
                                            </CardTitle>
                                        </CardHeader>
                                        <CardContent>
                                            <svg width="100%" viewBox={`0 0 ${width / 2} ${height / 2}`} className="bg-white rounded-lg border">
                                                <path
                                                    d={baselineCorrectedData.y.map((v, i) => {
                                                        const xVal = padding.left / 2 + ((baselineCorrectedData.x[i] - Math.min(...baselineCorrectedData.x)) /
                                                            (Math.max(...baselineCorrectedData.x) - Math.min(...baselineCorrectedData.x))) * (width / 2 - padding.left);
                                                        const yVal = 20 + (1 - v / (Math.max(...baselineCorrectedData.y) * 1.1)) * (height / 2 - 40);
                                                        return i === 0 ? `M ${xVal} ${yVal}` : `L ${xVal} ${yVal}`;
                                                    }).join(' ')}
                                                    fill="none" stroke="#22c55e" strokeWidth={2} />
                                            </svg>
                                        </CardContent>
                                    </Card>

                                    <Card className="border-2">
                                        <CardHeader className="py-3">
                                            <CardTitle className="text-sm flex items-center gap-2">
                                                <Target className="h-5 w-5 text-violet-500" />
                                                Detected Peaks ({detectedPeaks.length})
                                            </CardTitle>
                                        </CardHeader>
                                        <CardContent>
                                            <ScrollArea className="h-40">
                                                <div className="space-y-1.5">
                                                    {detectedPeaks.map((peak, i) => (
                                                        <div key={i} className="flex items-center justify-between p-2.5 bg-gradient-to-r from-slate-50 to-white rounded-lg border text-sm">
                                                            <span className="font-mono font-semibold text-violet-700">{peak.wavenumber.toFixed(1)}</span>
                                                            <div className="flex gap-2">
                                                                <Badge variant="secondary" className="bg-violet-100 text-violet-700">I: {(peak.intensity * 100).toFixed(0)}%</Badge>
                                                                {peak.fwhm && <Badge variant="outline">FWHM: {peak.fwhm.toFixed(1)}</Badge>}
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </ScrollArea>
                                        </CardContent>
                                    </Card>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </MainLayout>
    );
}
