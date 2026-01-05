'use client';

import React, { useState, useCallback, useMemo } from 'react';
import { useDropzone } from 'react-dropzone';
import { MainLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import {
    BarChart3, Upload, Download, FileText, CheckCircle2, Plus, Layers,
    Activity, TrendingUp, Target, Loader2, Table2, ArrowRight, Eye,
    EyeOff, X, FileSpreadsheet,
} from 'lucide-react';

// Types
interface RawCSVData {
    headers: string[];
    rows: (string | number)[][];
    numericColumns: number[];
}

// Generate synthetic SERS spectrum
function generateSyntheticSpectrum(type: 'r6g' | 'bacteria' | 'protein'): { wavenumber: number[]; intensity: number[] } {
    const wavenumber: number[] = [];
    const intensity: number[] = [];

    for (let w = 200; w <= 2000; w += 2) {
        wavenumber.push(w);
        let y = Math.random() * 0.1 + 0.3 * Math.exp(-((w - 800) ** 2) / 200000);

        const peakSets: { [key: string]: { pos: number; height: number; width: number }[] } = {
            r6g: [
                { pos: 611, height: 0.7, width: 15 }, { pos: 773, height: 0.5, width: 12 },
                { pos: 1183, height: 0.4, width: 15 }, { pos: 1311, height: 0.5, width: 12 },
                { pos: 1363, height: 0.8, width: 15 }, { pos: 1509, height: 1.0, width: 18 },
                { pos: 1575, height: 0.6, width: 15 }, { pos: 1649, height: 0.7, width: 15 },
            ],
            bacteria: [
                { pos: 725, height: 0.6, width: 20 }, { pos: 780, height: 0.4, width: 18 },
                { pos: 1003, height: 0.8, width: 15 }, { pos: 1090, height: 0.5, width: 20 },
                { pos: 1245, height: 0.4, width: 25 }, { pos: 1450, height: 0.6, width: 25 },
                { pos: 1660, height: 0.7, width: 30 },
            ],
            protein: [
                { pos: 830, height: 0.3, width: 15 }, { pos: 1003, height: 0.9, width: 12 },
                { pos: 1240, height: 0.5, width: 30 }, { pos: 1340, height: 0.4, width: 20 },
                { pos: 1450, height: 0.5, width: 25 }, { pos: 1555, height: 0.4, width: 20 },
                { pos: 1655, height: 0.8, width: 35 },
            ],
        };

        for (const peak of peakSets[type] || []) {
            y += peak.height * Math.exp(-((w - peak.pos) ** 2) / (2 * peak.width ** 2));
        }
        intensity.push(y);
    }
    return { wavenumber, intensity };
}

// Baseline correction
function baselineCorrection(intensity: number[]): { corrected: number[]; baseline: number[] } {
    const windowSize = 50;
    const baseline: number[] = [];
    for (let i = 0; i < intensity.length; i++) {
        const start = Math.max(0, i - windowSize);
        const end = Math.min(intensity.length, i + windowSize);
        baseline.push(Math.min(...intensity.slice(start, end)));
    }
    const smoothed = baseline.map((_, i) => {
        const s = Math.max(0, i - 10), e = Math.min(baseline.length, i + 10);
        return baseline.slice(s, e).reduce((a, b) => a + b, 0) / (e - s);
    });
    return { corrected: intensity.map((y, i) => y - smoothed[i]), baseline: smoothed };
}

// Smoothing
function smoothSpectrum(intensity: number[], windowSize: number = 11): number[] {
    const result: number[] = [];
    const half = Math.floor(windowSize / 2);
    for (let i = 0; i < intensity.length; i++) {
        const s = Math.max(0, i - half), e = Math.min(intensity.length, i + half + 1);
        result.push(intensity.slice(s, e).reduce((a, b) => a + b, 0) / (e - s));
    }
    return result;
}

// Normalization
function normalizeSpectrum(intensity: number[], method: 'vector' | 'max' | 'minmax'): number[] {
    if (method === 'max') {
        const max = Math.max(...intensity);
        return intensity.map(y => y / max);
    } else if (method === 'minmax') {
        const min = Math.min(...intensity), max = Math.max(...intensity);
        return intensity.map(y => (y - min) / (max - min));
    } else {
        const norm = Math.sqrt(intensity.reduce((sum, y) => sum + y * y, 0));
        return intensity.map(y => y / norm);
    }
}

// Peak detection
function detectPeaks(wavenumber: number[], intensity: number[], prominence: number = 0.1): { wavenumber: number; intensity: number; assignment?: string }[] {
    const peaks: { wavenumber: number; intensity: number; assignment?: string }[] = [];
    const threshold = Math.max(...intensity) * prominence;
    const knownPeaks: { [key: number]: string } = {
        611: 'C-C-C ring (R6G)', 725: 'Adenine', 773: 'C-H (R6G)', 780: 'Cytosine',
        830: 'Tyrosine', 1003: 'Phenylalanine', 1090: 'DNA PO₂⁻', 1183: 'C-H (R6G)',
        1240: 'Amide III', 1311: 'N-H (R6G)', 1340: 'Tryptophan', 1363: 'C-C (R6G)',
        1450: 'CH₂', 1509: 'C-C (R6G)', 1555: 'Tryptophan', 1575: 'C-C (R6G)',
        1649: 'C-C (R6G)', 1655: 'Amide I', 1660: 'Amide I',
    };

    for (let i = 5; i < intensity.length - 5; i++) {
        if (intensity[i] > intensity[i - 1] && intensity[i] > intensity[i + 1] &&
            intensity[i] > intensity[i - 2] && intensity[i] > intensity[i + 2] && intensity[i] > threshold) {
            const wn = wavenumber[i];
            let assignment: string | undefined;
            for (const [k, v] of Object.entries(knownPeaks)) {
                if (Math.abs(wn - parseInt(k)) < 20) { assignment = v; break; }
            }
            peaks.push({ wavenumber: wn, intensity: intensity[i], assignment });
        }
    }
    return peaks.sort((a, b) => b.intensity - a.intensity).slice(0, 12);
}

// Derivative
function calculateDerivative(wavenumber: number[], intensity: number[]): number[] {
    const d: number[] = [0];
    for (let i = 1; i < intensity.length - 1; i++) {
        const dw = wavenumber[i + 1] - wavenumber[i - 1];
        d.push(dw !== 0 ? (intensity[i + 1] - intensity[i - 1]) / dw : 0);
    }
    d.push(0);
    return d;
}

type VisualizationType = 'raw' | 'processed' | 'baseline' | 'peaks' | 'comparison' | 'derivative' | 'intensity-map';

interface Visualization {
    id: string;
    type: VisualizationType;
    title: string;
    createdAt: Date;
}

export default function VisualizePage() {
    // Raw CSV data for preview
    const [rawCSVData, setRawCSVData] = useState < RawCSVData | null > (null);
    const [fileName, setFileName] = useState < string > ('');
    const [showDataPreview, setShowDataPreview] = useState(true);

    // Column selection
    const [xColumn, setXColumn] = useState < number | null > (null);
    const [yColumn, setYColumn] = useState < number | null > (null);
    const [columnsConfirmed, setColumnsConfirmed] = useState(false);

    // Spectrum data
    const [uploadedData, setUploadedData] = useState < { wavenumber: number[]; intensity: number[] } | null > (null);

    // Processing state
    const [processing, setProcessing] = useState({
        baseline: false,
        smoothing: false,
        normalization: 'none' as 'none' | 'vector' | 'max' | 'minmax',
        peakDetection: false,
    });

    const [activeViz, setActiveViz] = useState < VisualizationType > ('raw');
    const [visualizations, setVisualizations] = useState < Visualization[] > ([]);
    const [isCreatingViz, setIsCreatingViz] = useState(false);
    const [selectedVizType, setSelectedVizType] = useState < VisualizationType > ('processed');
    const [dialogOpen, setDialogOpen] = useState(false);

    // Parse CSV
    const parseCSVFile = (text: string): RawCSVData => {
        const lines = text.trim().split('\n');
        const rows: (string | number)[][] = [];
        let headers: string[] = [];
        const delimiter = lines[0].includes('\t') ? '\t' : lines[0].includes(',') ? ',' : /\s+/;

        for (let i = 0; i < lines.length; i++) {
            const parts = lines[i].split(delimiter).map(p => p.trim()).filter(p => p);
            if (parts.length === 0) continue;
            if (i === 0 && parts.some(p => isNaN(parseFloat(p)))) {
                headers = parts;
            } else {
                rows.push(parts.map(p => { const n = parseFloat(p); return isNaN(n) ? p : n; }));
            }
        }

        if (headers.length === 0 && rows.length > 0) {
            headers = rows[0].map((_, i) => `Column ${i + 1}`);
        }

        const numericColumns: number[] = [];
        if (rows.length > 0) {
            for (let col = 0; col < rows[0].length; col++) {
                if (rows.slice(0, 10).every(row => row[col] !== undefined && typeof row[col] === 'number')) {
                    numericColumns.push(col);
                }
            }
        }
        return { headers, rows, numericColumns };
    };

    const onDrop = useCallback((acceptedFiles: File[]) => {
        const file = acceptedFiles[0];
        if (!file) return;
        setFileName(file.name);
        setColumnsConfirmed(false);
        setUploadedData(null);
        setVisualizations([]);

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const text = e.target?.result as string;
                const csvData = parseCSVFile(text);
                setRawCSVData(csvData);
                setShowDataPreview(true);
                if (csvData.numericColumns.length >= 2) {
                    setXColumn(csvData.numericColumns[0]);
                    setYColumn(csvData.numericColumns[1]);
                }
            } catch {
                console.error('Parse error');
            }
        };
        reader.readAsText(file);
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'text/csv': ['.csv'], 'text/plain': ['.txt'] },
    });

    // Confirm column selection
    const confirmColumnSelection = () => {
        if (!rawCSVData || xColumn === null || yColumn === null) return;
        const wavenumber: number[] = [], intensity: number[] = [];
        for (const row of rawCSVData.rows) {
            const x = row[xColumn], y = row[yColumn];
            if (typeof x === 'number' && typeof y === 'number') {
                wavenumber.push(x);
                intensity.push(y);
            }
        }
        if (wavenumber.length > 0) {
            setUploadedData({ wavenumber, intensity });
            setColumnsConfirmed(true);
            setShowDataPreview(false);
        }
    };

    // Load demo
    const loadDemoData = (type: 'r6g' | 'bacteria' | 'protein') => {
        const names = { r6g: 'R6G_sample.csv', bacteria: 'Bacteria_SERS.csv', protein: 'Protein_SERS.csv' };
        setFileName(names[type]);
        setUploadedData(generateSyntheticSpectrum(type));
        setRawCSVData(null);
        setColumnsConfirmed(true);
        setShowDataPreview(false);
    };

    // Clear
    const clearData = () => {
        setRawCSVData(null);
        setUploadedData(null);
        setFileName('');
        setXColumn(null);
        setYColumn(null);
        setColumnsConfirmed(false);
        setShowDataPreview(true);
        setVisualizations([]);
    };

    // Process data
    const processedData = useMemo(() => {
        if (!uploadedData) return null;
        let intensity = [...uploadedData.intensity];
        let baseline: number[] = [];
        if (processing.baseline) {
            const result = baselineCorrection(intensity);
            intensity = result.corrected;
            baseline = result.baseline;
        }
        if (processing.smoothing) intensity = smoothSpectrum(intensity);
        if (processing.normalization !== 'none') intensity = normalizeSpectrum(intensity, processing.normalization);
        return { wavenumber: uploadedData.wavenumber, intensity, baseline };
    }, [uploadedData, processing]);

    const detectedPeaks = useMemo(() => {
        if (!processedData || !processing.peakDetection) return [];
        return detectPeaks(processedData.wavenumber, processedData.intensity);
    }, [processedData, processing.peakDetection]);

    const derivativeData = useMemo(() => {
        if (!processedData) return null;
        return calculateDerivative(processedData.wavenumber, processedData.intensity);
    }, [processedData]);

    // Create visualization
    const createVisualization = async () => {
        if (!uploadedData) return;
        setIsCreatingViz(true);
        await new Promise(r => setTimeout(r, 500));
        const titles: { [key in VisualizationType]: string } = {
            raw: 'Raw Spectrum', processed: 'Processed Spectrum', baseline: 'Baseline Correction',
            peaks: 'Peak Analysis', comparison: 'Before/After', derivative: 'Derivative Spectrum',
            'intensity-map': 'Intensity Map',
        };
        setVisualizations(prev => [{ id: Date.now().toString(), type: selectedVizType, title: titles[selectedVizType], createdAt: new Date() }, ...prev]);
        setActiveViz(selectedVizType);
        setIsCreatingViz(false);
        setDialogOpen(false);
    };

    // Spectrum SVG component
    const SpectrumSVG = ({ data, peaks = [], title, color = '#6366f1', showBaseline = false, baselineData }: {
        data: { wavenumber: number[]; intensity: number[] };
        peaks?: { wavenumber: number; intensity: number; assignment?: string }[];
        title: string;
        color?: string;
        showBaseline?: boolean;
        baselineData?: number[];
    }) => {
        const width = 600, height = 300;
        const padding = { top: 30, right: 20, bottom: 50, left: 60 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        const minWn = Math.min(...data.wavenumber), maxWn = Math.max(...data.wavenumber);
        const minInt = Math.min(...data.intensity), maxInt = Math.max(...data.intensity);
        const xScale = (w: number) => padding.left + ((w - minWn) / (maxWn - minWn)) * plotWidth;
        const yScale = (v: number) => padding.top + (1 - (v - minInt) / (maxInt - minInt)) * plotHeight;

        const pathD = data.intensity.map((v, i) => `${i === 0 ? 'M' : 'L'} ${xScale(data.wavenumber[i])} ${yScale(v)}`).join(' ');
        const baselinePathD = baselineData ? baselineData.map((v, i) => `${i === 0 ? 'M' : 'L'} ${xScale(data.wavenumber[i])} ${yScale(v)}`).join(' ') : '';

        return (
            <svg width="100%" viewBox={`0 0 ${width} ${height}`} className="bg-white rounded-lg border">
                <text x={width / 2} y={18} textAnchor="middle" fill="#0f172a" fontSize={14} fontWeight={600}>{title}</text>
                {[0.25, 0.5, 0.75].map(v => (
                    <line key={v} x1={padding.left} y1={padding.top + v * plotHeight} x2={width - padding.right} y2={padding.top + v * plotHeight} stroke="#f1f5f9" />
                ))}
                {showBaseline && baselinePathD && <path d={baselinePathD} fill="none" stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="5,5" />}
                <path d={pathD} fill="none" stroke={color} strokeWidth={2} />
                {peaks.map((peak, i) => (
                    <g key={i}>
                        <circle cx={xScale(peak.wavenumber)} cy={yScale(peak.intensity)} r={5} fill="#ef4444" stroke="white" strokeWidth={2} />
                        <text x={xScale(peak.wavenumber)} y={yScale(peak.intensity) - 10} textAnchor="middle" fill="#ef4444" fontSize={10} fontWeight={600}>{Math.round(peak.wavenumber)}</text>
                    </g>
                ))}
                <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="#94a3b8" />
                <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="#94a3b8" />
                <text x={width / 2} y={height - 10} textAnchor="middle" fill="#64748b" fontSize={11}>Wavenumber (cm⁻¹)</text>
                <text x={15} y={height / 2} textAnchor="middle" fill="#64748b" fontSize={11} transform={`rotate(-90, 15, ${height / 2})`}>Intensity (a.u.)</text>
            </svg>
        );
    };

    // Intensity Map
    const IntensityMapSVG = ({ data }: { data: { wavenumber: number[]; intensity: number[] } }) => {
        const width = 600, height = 120;
        const normalized = normalizeSpectrum(data.intensity, 'minmax');
        return (
            <svg width="100%" viewBox={`0 0 ${width} ${height}`} className="bg-white rounded-lg border">
                <text x={width / 2} y={18} textAnchor="middle" fill="#0f172a" fontSize={14} fontWeight={600}>Intensity Mapping</text>
                {normalized.map((val, i) => {
                    const x = 30 + (i / normalized.length) * (width - 60);
                    const hue = 240 - val * 240;
                    return <rect key={i} x={x} y={30} width={(width - 60) / normalized.length + 0.5} height={50} fill={`hsl(${hue}, 80%, 50%)`} />;
                })}
                <text x={30} y={100} fill="#64748b" fontSize={10}>{Math.round(data.wavenumber[0])} cm⁻¹</text>
                <text x={width - 30} y={100} textAnchor="end" fill="#64748b" fontSize={10}>{Math.round(data.wavenumber[data.wavenumber.length - 1])} cm⁻¹</text>
            </svg>
        );
    };

    const vizTypes = [
        { value: 'processed' as const, label: 'Processed Spectrum', icon: <Activity className="h-4 w-4" />, desc: 'Apply preprocessing' },
        { value: 'baseline' as const, label: 'Baseline Correction', icon: <TrendingUp className="h-4 w-4" />, desc: 'View baseline overlay' },
        { value: 'peaks' as const, label: 'Peak Analysis', icon: <Target className="h-4 w-4" />, desc: 'Detect peaks' },
        { value: 'comparison' as const, label: 'Before/After', icon: <Layers className="h-4 w-4" />, desc: 'Compare raw & processed' },
        { value: 'derivative' as const, label: 'Derivative', icon: <Activity className="h-4 w-4" />, desc: 'First derivative' },
        { value: 'intensity-map' as const, label: 'Intensity Map', icon: <BarChart3 className="h-4 w-4" />, desc: 'Heat map view' },
    ];

    return (
        <MainLayout>
            <div className="flex flex-col h-full overflow-auto bg-gradient-to-br from-slate-50 via-indigo-50/30 to-white">
                {/* Header */}
                <div className="p-6 border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="p-2.5 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 shadow-lg shadow-indigo-200">
                                <BarChart3 className="h-6 w-6 text-white" />
                            </div>
                            <div>
                                <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600">Visualization Studio</h1>
                                <p className="text-muted-foreground">Advanced SERS spectral analysis with column selection</p>
                            </div>
                        </div>

                        <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
                            <DialogTrigger asChild>
                                <Button className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg" disabled={!uploadedData}>
                                    <Plus className="h-4 w-4 mr-2" />Create Visualization
                                </Button>
                            </DialogTrigger>
                            <DialogContent className="max-w-lg">
                                <DialogHeader>
                                    <DialogTitle>Create New Visualization</DialogTitle>
                                    <DialogDescription>Select a visualization type</DialogDescription>
                                </DialogHeader>
                                <div className="grid gap-2 py-4">
                                    {vizTypes.map(viz => (
                                        <button key={viz.value} onClick={() => setSelectedVizType(viz.value)}
                                            className={`flex items-center gap-3 p-3 rounded-lg border-2 text-left transition-all ${selectedVizType === viz.value ? 'border-indigo-500 bg-indigo-50' : 'border-slate-200 hover:border-indigo-300'}`}>
                                            <div className={`p-2 rounded-lg ${selectedVizType === viz.value ? 'bg-indigo-100 text-indigo-600' : 'bg-slate-100'}`}>{viz.icon}</div>
                                            <div className="flex-1">
                                                <p className="font-medium text-sm">{viz.label}</p>
                                                <p className="text-xs text-muted-foreground">{viz.desc}</p>
                                            </div>
                                            {selectedVizType === viz.value && <CheckCircle2 className="h-5 w-5 text-indigo-500" />}
                                        </button>
                                    ))}
                                </div>
                                <div className="flex justify-end gap-2">
                                    <Button variant="outline" onClick={() => setDialogOpen(false)}>Cancel</Button>
                                    <Button className="bg-indigo-600" onClick={createVisualization} disabled={isCreatingViz}>
                                        {isCreatingViz ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Creating...</> : 'Create'}
                                    </Button>
                                </div>
                            </DialogContent>
                        </Dialog>
                    </div>
                </div>

                <div className="flex-1 p-6">
                    <div className="grid lg:grid-cols-3 gap-6 max-w-[1600px] mx-auto">
                        {/* Left Panel */}
                        <div className="space-y-4">
                            {/* Data Input */}
                            <Card className="border-2">
                                <CardHeader className="py-3">
                                    <CardTitle className="text-base flex items-center gap-2">
                                        <Upload className="h-4 w-4 text-indigo-500" />Data Input
                                    </CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    <div {...getRootProps()}
                                        className={`p-5 border-2 border-dashed rounded-xl text-center cursor-pointer transition-all ${isDragActive ? 'border-indigo-500 bg-indigo-50' : 'border-slate-200 hover:border-indigo-300'}`}>
                                        <input {...getInputProps()} />
                                        <Upload className="h-8 w-8 mx-auto mb-2 text-indigo-500" />
                                        <p className="text-sm font-medium">Drop CSV/TXT file</p>
                                        <p className="text-xs text-muted-foreground">or click to browse</p>
                                    </div>

                                    {fileName && (
                                        <div className="flex items-center gap-2 p-3 bg-gradient-to-r from-emerald-50 to-green-50 border border-emerald-200 rounded-lg">
                                            <FileSpreadsheet className="h-5 w-5 text-emerald-600" />
                                            <span className="text-sm text-emerald-800 flex-1 truncate font-medium">{fileName}</span>
                                            <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={clearData}><X className="h-4 w-4" /></Button>
                                        </div>
                                    )}

                                    <Separator />

                                    <div>
                                        <p className="text-xs font-semibold mb-2 text-slate-500 uppercase">Demo Spectra</p>
                                        <div className="grid grid-cols-3 gap-2">
                                            <Button variant="outline" size="sm" className="hover:bg-indigo-50" onClick={() => loadDemoData('r6g')}>R6G</Button>
                                            <Button variant="outline" size="sm" className="hover:bg-indigo-50" onClick={() => loadDemoData('bacteria')}>Bacteria</Button>
                                            <Button variant="outline" size="sm" className="hover:bg-indigo-50" onClick={() => loadDemoData('protein')}>Protein</Button>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Preprocessing */}
                            <Card className="border-2">
                                <CardHeader className="py-3">
                                    <CardTitle className="text-base">Preprocessing</CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-3">
                                    <div className="flex items-center justify-between">
                                        <Label className="text-sm">Baseline Correction</Label>
                                        <Button variant={processing.baseline ? 'default' : 'outline'} size="sm" onClick={() => setProcessing(p => ({ ...p, baseline: !p.baseline }))}>
                                            {processing.baseline ? 'On' : 'Off'}
                                        </Button>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <Label className="text-sm">Smoothing</Label>
                                        <Button variant={processing.smoothing ? 'default' : 'outline'} size="sm" onClick={() => setProcessing(p => ({ ...p, smoothing: !p.smoothing }))}>
                                            {processing.smoothing ? 'On' : 'Off'}
                                        </Button>
                                    </div>
                                    <div className="space-y-2">
                                        <Label className="text-sm">Normalization</Label>
                                        <Select value={processing.normalization} onValueChange={v => setProcessing(p => ({ ...p, normalization: v as typeof p.normalization }))}>
                                            <SelectTrigger><SelectValue /></SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="none">None</SelectItem>
                                                <SelectItem value="max">Max</SelectItem>
                                                <SelectItem value="vector">Vector</SelectItem>
                                                <SelectItem value="minmax">Min-Max</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <Label className="text-sm">Peak Detection</Label>
                                        <Button variant={processing.peakDetection ? 'default' : 'outline'} size="sm" onClick={() => setProcessing(p => ({ ...p, peakDetection: !p.peakDetection }))}>
                                            {processing.peakDetection ? 'On' : 'Off'}
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>

                            {/* Visualizations */}
                            {visualizations.length > 0 && (
                                <Card className="border-2">
                                    <CardHeader className="py-3"><CardTitle className="text-base">Created Visualizations</CardTitle></CardHeader>
                                    <CardContent className="space-y-2">
                                        {visualizations.map(viz => (
                                            <button key={viz.id} onClick={() => setActiveViz(viz.type)}
                                                className={`w-full flex items-center gap-2 p-2 rounded-lg text-left text-sm ${activeViz === viz.type ? 'bg-indigo-100 text-indigo-700' : 'hover:bg-slate-100'}`}>
                                                <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                                                <span className="flex-1">{viz.title}</span>
                                            </button>
                                        ))}
                                    </CardContent>
                                </Card>
                            )}
                        </div>

                        {/* Main Area */}
                        <div className="lg:col-span-2 space-y-4">
                            {/* Data Preview */}
                            {rawCSVData && !columnsConfirmed && (
                                <Card className="border-2 border-indigo-300 shadow-lg">
                                    <CardHeader className="py-4 bg-gradient-to-r from-indigo-50 to-purple-50">
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-2">
                                                <Table2 className="h-5 w-5 text-indigo-600" />
                                                <CardTitle className="text-base">Data Preview & Column Selection</CardTitle>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <Badge variant="outline">{rawCSVData.rows.length} rows × {rawCSVData.headers.length} cols</Badge>
                                                <Button variant="ghost" size="sm" onClick={() => setShowDataPreview(!showDataPreview)}>
                                                    {showDataPreview ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                                                </Button>
                                            </div>
                                        </div>
                                        <CardDescription>Select X (wavenumber) and Y (intensity) columns</CardDescription>
                                    </CardHeader>
                                    <CardContent className="p-0">
                                        <div className="p-4 bg-slate-50 border-b flex items-center gap-6 flex-wrap">
                                            <div className="flex items-center gap-3">
                                                <Label className="text-sm font-medium whitespace-nowrap">X-Axis:</Label>
                                                <Select value={xColumn?.toString() ?? ''} onValueChange={v => setXColumn(parseInt(v))}>
                                                    <SelectTrigger className="w-44 bg-white"><SelectValue placeholder="Select..." /></SelectTrigger>
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
                                                <Label className="text-sm font-medium whitespace-nowrap">Y-Axis:</Label>
                                                <Select value={yColumn?.toString() ?? ''} onValueChange={v => setYColumn(parseInt(v))}>
                                                    <SelectTrigger className="w-44 bg-white"><SelectValue placeholder="Select..." /></SelectTrigger>
                                                    <SelectContent>
                                                        {rawCSVData.numericColumns.map(col => (
                                                            <SelectItem key={col} value={col.toString()} disabled={col === xColumn}>
                                                                {rawCSVData.headers[col] || `Column ${col + 1}`}
                                                            </SelectItem>
                                                        ))}
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                            <Button className="ml-auto bg-indigo-600" onClick={confirmColumnSelection} disabled={xColumn === null || yColumn === null}>
                                                <ArrowRight className="h-4 w-4 mr-2" />Confirm & Continue
                                            </Button>
                                        </div>

                                        {showDataPreview && (
                                            <ScrollArea className="h-56">
                                                <table className="w-full text-sm">
                                                    <thead className="bg-slate-100 sticky top-0">
                                                        <tr>
                                                            <th className="px-3 py-2 text-left text-xs font-semibold text-slate-500 w-12">#</th>
                                                            {rawCSVData.headers.map((h, i) => (
                                                                <th key={i}
                                                                    className={`px-3 py-2 text-left text-xs font-semibold cursor-pointer ${i === xColumn ? 'bg-blue-100 text-blue-700' : i === yColumn ? 'bg-green-100 text-green-700' : 'text-slate-500 hover:bg-slate-200'}`}
                                                                    onClick={() => {
                                                                        if (!rawCSVData.numericColumns.includes(i)) return;
                                                                        if (xColumn === null) setXColumn(i);
                                                                        else if (yColumn === null && i !== xColumn) setYColumn(i);
                                                                        else if (i === xColumn) setXColumn(null);
                                                                        else if (i === yColumn) setYColumn(null);
                                                                    }}>
                                                                    <div className="flex items-center gap-1">
                                                                        {h}
                                                                        {i === xColumn && <Badge className="ml-1 h-4 text-[10px] bg-blue-500">X</Badge>}
                                                                        {i === yColumn && <Badge className="ml-1 h-4 text-[10px] bg-green-500">Y</Badge>}
                                                                    </div>
                                                                </th>
                                                            ))}
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {rawCSVData.rows.slice(0, 30).map((row, ri) => (
                                                            <tr key={ri} className="border-b border-slate-100 hover:bg-slate-50">
                                                                <td className="px-3 py-1.5 text-xs text-slate-400">{ri + 1}</td>
                                                                {row.map((cell, ci) => (
                                                                    <td key={ci} className={`px-3 py-1.5 text-xs font-mono ${ci === xColumn ? 'bg-blue-50' : ci === yColumn ? 'bg-green-50' : ''}`}>
                                                                        {typeof cell === 'number' ? cell.toPrecision(5) : cell}
                                                                    </td>
                                                                ))}
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                            </ScrollArea>
                                        )}
                                    </CardContent>
                                </Card>
                            )}

                            {/* Visualization */}
                            <Card className="h-full border-2">
                                <CardHeader className="py-4">
                                    <div className="flex items-center justify-between">
                                        <CardTitle className="text-base">Spectrum Visualization</CardTitle>
                                        <Button variant="outline" size="sm" disabled={!uploadedData}><Download className="h-4 w-4 mr-1" />Export</Button>
                                    </div>
                                </CardHeader>
                                <CardContent>
                                    {uploadedData ? (
                                        <Tabs value={activeViz} onValueChange={v => setActiveViz(v as VisualizationType)}>
                                            <TabsList className="mb-4 flex-wrap h-auto gap-1">
                                                <TabsTrigger value="raw">Raw</TabsTrigger>
                                                <TabsTrigger value="processed">Processed</TabsTrigger>
                                                <TabsTrigger value="baseline">Baseline</TabsTrigger>
                                                <TabsTrigger value="peaks">Peaks</TabsTrigger>
                                                <TabsTrigger value="comparison">Compare</TabsTrigger>
                                                <TabsTrigger value="derivative">Derivative</TabsTrigger>
                                                <TabsTrigger value="intensity-map">Map</TabsTrigger>
                                            </TabsList>

                                            <TabsContent value="raw"><SpectrumSVG data={uploadedData} title="Raw SERS Spectrum (Original Data)" /></TabsContent>
                                            <TabsContent value="processed">
                                                {processedData && (
                                                    <div className="space-y-4">
                                                        <SpectrumSVG
                                                            data={processedData}
                                                            peaks={processing.peakDetection ? detectedPeaks : []}
                                                            title="Processed Spectrum"
                                                            color="#06b6d4"
                                                        />
                                                        <div className="flex flex-wrap justify-center gap-2">
                                                            <Badge variant={processing.baseline ? "default" : "outline"} className={processing.baseline ? "bg-green-500" : ""}>
                                                                Baseline: {processing.baseline ? "ON" : "OFF"}
                                                            </Badge>
                                                            <Badge variant={processing.smoothing ? "default" : "outline"} className={processing.smoothing ? "bg-green-500" : ""}>
                                                                Smoothing: {processing.smoothing ? "ON" : "OFF"}
                                                            </Badge>
                                                            <Badge variant={processing.normalization !== 'none' ? "default" : "outline"} className={processing.normalization !== 'none' ? "bg-green-500" : ""}>
                                                                Normalization: {processing.normalization === 'none' ? "OFF" : processing.normalization.toUpperCase()}
                                                            </Badge>
                                                            <Badge variant={processing.peakDetection ? "default" : "outline"} className={processing.peakDetection ? "bg-green-500" : ""}>
                                                                Peaks: {processing.peakDetection ? "ON" : "OFF"}
                                                            </Badge>
                                                        </div>
                                                        {!processing.baseline && !processing.smoothing && processing.normalization === 'none' && (
                                                            <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-800 text-center">
                                                                💡 Enable preprocessing options in the left panel to see processed results
                                                            </div>
                                                        )}
                                                    </div>
                                                )}
                                            </TabsContent>
                                            <TabsContent value="baseline">{uploadedData && (
                                                <div className="space-y-4">
                                                    <SpectrumSVG
                                                        data={uploadedData}
                                                        title={processing.baseline ? "Baseline Analysis (Correction ON)" : "Baseline Analysis (Enable Baseline Correction)"}
                                                        showBaseline={true}
                                                        baselineData={processing.baseline && processedData?.baseline.length ? processedData.baseline : undefined}
                                                    />
                                                    <div className="flex justify-center gap-6 text-sm">
                                                        <div className="flex items-center gap-2"><div className="w-4 h-1 bg-indigo-500 rounded" /><span className="text-muted-foreground">Original Spectrum</span></div>
                                                        <div className="flex items-center gap-2"><div className="w-4 h-1 bg-amber-500 rounded" style={{ backgroundImage: 'repeating-linear-gradient(90deg, #f59e0b, #f59e0b 3px, transparent 3px, transparent 6px)' }} /><span className="text-muted-foreground">Estimated Baseline</span></div>
                                                    </div>
                                                    {!processing.baseline && (
                                                        <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-800 text-center">
                                                            ⚠️ Enable "Baseline Correction" in the Preprocessing panel to see the baseline
                                                        </div>
                                                    )}
                                                </div>
                                            )}</TabsContent>
                                            <TabsContent value="peaks">
                                                <div className="grid gap-4">
                                                    {processedData && <SpectrumSVG data={processedData} peaks={detectedPeaks} title="Peak Detection" />}
                                                    {detectedPeaks.length > 0 && (
                                                        <div className="p-4 bg-slate-50 rounded-lg border">
                                                            <h4 className="font-medium mb-3">Detected Peaks ({detectedPeaks.length})</h4>
                                                            <div className="grid grid-cols-3 gap-2">
                                                                {detectedPeaks.map((peak, i) => (
                                                                    <div key={i} className="p-2 bg-white rounded border">
                                                                        <div className="flex justify-between">
                                                                            <span className="font-mono text-sm font-medium">{Math.round(peak.wavenumber)} cm⁻¹</span>
                                                                            <Badge variant="outline">{(peak.intensity * 100).toFixed(0)}%</Badge>
                                                                        </div>
                                                                        {peak.assignment && <span className="text-xs text-muted-foreground">{peak.assignment}</span>}
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            </TabsContent>
                                            <TabsContent value="comparison">
                                                <div className="space-y-4">
                                                    <SpectrumSVG data={uploadedData} title="Raw" color="#94a3b8" />
                                                    {processedData && <SpectrumSVG data={processedData} title="Processed" color="#06b6d4" />}
                                                </div>
                                            </TabsContent>
                                            <TabsContent value="derivative">{processedData && derivativeData && (
                                                <SpectrumSVG data={{ wavenumber: processedData.wavenumber, intensity: derivativeData }} title="First Derivative" color="#8b5cf6" />
                                            )}</TabsContent>
                                            <TabsContent value="intensity-map">{processedData && (
                                                <div className="space-y-4">
                                                    <IntensityMapSVG data={processedData} />
                                                    <SpectrumSVG data={processedData} peaks={processing.peakDetection ? detectedPeaks : []} title="Spectrum" />
                                                </div>
                                            )}</TabsContent>
                                        </Tabs>
                                    ) : (
                                        <div className="h-80 flex flex-col items-center justify-center bg-gradient-to-b from-white to-slate-50 rounded-lg border-2 border-dashed">
                                            <BarChart3 className="h-16 w-16 text-slate-200 mb-4" />
                                            <p className="text-lg font-medium text-muted-foreground">Upload spectrum data</p>
                                            <p className="text-sm text-muted-foreground/70">Drop a CSV file or use demo spectra</p>
                                        </div>
                                    )}
                                </CardContent>
                            </Card>
                        </div>
                    </div>
                </div>
            </div>
        </MainLayout>
    );
}
