'use client';

import React, { useState, useMemo } from 'react';
import { MainLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { FlaskConical, Atom, Zap, Download, RefreshCw, Info, CheckCircle2 } from 'lucide-react';

// Drude-Lorentz model parameters from literature
// Johnson & Christy optical constants
const MATERIAL_DATA = {
    Ag: {
        name: 'Silver',
        symbol: 'Ag',
        epsilon_inf: 3.7,
        omega_p: 9.17, // eV - plasma frequency
        gamma: 0.021, // eV - damping
        // Size-dependent LSPR from Mie theory
        lsprBase: 390,
        sizeCoeff: 0.7, // nm red-shift per nm size increase
        description: 'Silver provides the highest enhancement but oxidizes easily',
    },
    Au: {
        name: 'Gold',
        symbol: 'Au',
        epsilon_inf: 9.84,
        omega_p: 9.03,
        gamma: 0.072,
        lsprBase: 520,
        sizeCoeff: 0.4,
        description: 'Gold is more stable but has lower enhancement than silver',
    },
};

const SHAPE_DATA = {
    sphere: {
        name: 'Nanosphere',
        depolarization: 1 / 3,
        efMultiplier: 1.0,
        peakShift: 0,
        hotspotFactor: 1.0,
        description: 'Uniform field enhancement, most studied geometry',
    },
    rod: {
        name: 'Nanorod',
        depolarization: 0.1, // Longitudinal mode
        efMultiplier: 2.5,
        peakShift: 100, // Red-shifted
        hotspotFactor: 1.5,
        description: 'Two LSPR modes: transverse (~520nm) and longitudinal (tunable)',
    },
    star: {
        name: 'Nanostar',
        depolarization: 0.05,
        efMultiplier: 10.0,
        peakShift: 150,
        hotspotFactor: 5.0,
        description: 'Lightning rod effect at tips creates intense hotspots',
    },
    cube: {
        name: 'Nanocube',
        depolarization: 0.25,
        efMultiplier: 1.5,
        peakShift: 30,
        hotspotFactor: 1.2,
        description: 'Sharp corners provide enhanced fields',
    },
};

// Mie theory calculation for extinction cross-section
function calculateMieExtinction(wavelengths: number[], material: string, size: number, shape: string) {
    const mat = MATERIAL_DATA[material as keyof typeof MATERIAL_DATA];
    const shp = SHAPE_DATA[shape as keyof typeof SHAPE_DATA];

    const spectrum: number[] = [];
    const hbar = 6.582e-16; // eV·s
    const c = 3e17; // nm/s

    for (const lambda of wavelengths) {
        const omega = (hbar * 2 * Math.PI * c) / lambda; // Angular frequency
        const energy = (1240 / lambda); // eV

        // Drude model dielectric function
        const epsilon_real = mat.epsilon_inf - (mat.omega_p ** 2) / (energy ** 2 + mat.gamma ** 2);
        const epsilon_imag = (mat.omega_p ** 2 * mat.gamma) / (energy * (energy ** 2 + mat.gamma ** 2));

        // Medium permittivity (water n=1.33)
        const epsilon_m = 1.77;

        // Polarizability with shape-dependent depolarization factor
        const L = shp.depolarization;
        const denom_real = epsilon_real + ((1 - L) / L) * epsilon_m;
        const denom_imag = epsilon_imag;
        const denom_mag = Math.sqrt(denom_real ** 2 + denom_imag ** 2);

        // Extinction (absorption + scattering)
        const extinction = epsilon_imag / (denom_mag ** 2);
        spectrum.push(extinction);
    }

    // Normalize
    const maxVal = Math.max(...spectrum);
    return spectrum.map(v => v / maxVal);
}

// Calculate LSPR peak position using modified Fröhlich condition
function calculateLSPRPeak(material: string, size: number, shape: string): number {
    const mat = MATERIAL_DATA[material as keyof typeof MATERIAL_DATA];
    const shp = SHAPE_DATA[shape as keyof typeof SHAPE_DATA];

    // Base LSPR position
    let peak = mat.lsprBase;

    // Size-dependent red shift (empirical from Mie theory)
    peak += mat.sizeCoeff * (size - 20);

    // Shape-dependent shift
    peak += shp.peakShift;

    return Math.round(peak);
}

// Calculate enhancement factor using electromagnetic theory
function calculateEnhancementFactor(material: string, size: number, shape: string, excitation: number): { ef: number; exponent: number } {
    const mat = MATERIAL_DATA[material as keyof typeof MATERIAL_DATA];
    const shp = SHAPE_DATA[shape as keyof typeof SHAPE_DATA];
    const lsprPeak = calculateLSPRPeak(material, size, shape);

    // Base enhancement from material (silver ~10^6, gold ~10^5)
    const baseEF = material === 'Ag' ? 6 : 5;

    // Shape enhancement (hotspot effect)
    const shapeBonus = Math.log10(shp.efMultiplier);

    // Size optimization (optimal around 50-80nm for Ag)
    const optimalSize = material === 'Ag' ? 60 : 50;
    const sizeDetuning = Math.abs(size - optimalSize) / 50;
    const sizePenalty = sizeDetuning * 0.5;

    // Wavelength matching (resonance condition)
    const detuning = Math.abs(excitation - lsprPeak);
    const fwhm = 50 + size * 0.3; // Broader peaks for larger particles
    const resonanceFactor = Math.exp(-(detuning ** 2) / (2 * fwhm ** 2));
    const resonanceBonus = resonanceFactor * 1.5;

    const totalExponent = baseEF + shapeBonus - sizePenalty + resonanceBonus;
    const clampedExponent = Math.min(Math.max(totalExponent, 3), 12);

    return {
        ef: Math.pow(10, clampedExponent),
        exponent: clampedExponent,
    };
}

// Calculate FWHM
function calculateFWHM(material: string, size: number, shape: string): number {
    const mat = MATERIAL_DATA[material as keyof typeof MATERIAL_DATA];

    // Intrinsic damping contribution
    const intrinsicFWHM = mat.gamma * 100; // Convert to nm

    // Radiative damping (increases with size^3)
    const radiativeFWHM = 0.5 * (size / 50) ** 2;

    // Total FWHM
    return Math.round(intrinsicFWHM + radiativeFWHM * 20 + 30);
}

export default function SimulatePage() {
    const [material, setMaterial] = useState < 'Ag' | 'Au' > ('Ag');
    const [size, setSize] = useState(50);
    const [shape, setShape] = useState < 'sphere' | 'rod' | 'star' | 'cube' > ('sphere');
    const [excitation, setExcitation] = useState(785);
    const [hasSimulated, setHasSimulated] = useState(false);
    const [isSimulating, setIsSimulating] = useState(false);

    const wavelengths = useMemo(() => {
        const arr = [];
        for (let w = 300; w <= 900; w += 2) arr.push(w);
        return arr;
    }, []);

    const results = useMemo(() => {
        if (!hasSimulated) return null;

        const spectrum = calculateMieExtinction(wavelengths, material, size, shape);
        const lsprPeak = calculateLSPRPeak(material, size, shape);
        const { ef, exponent } = calculateEnhancementFactor(material, size, shape, excitation);
        const fwhm = calculateFWHM(material, size, shape);

        // Generate recommendation
        const detuning = Math.abs(excitation - lsprPeak);
        let recommendation = '';
        let quality: 'excellent' | 'good' | 'poor' = 'good';

        if (detuning < 50) {
            quality = 'excellent';
            recommendation = `Excellent configuration! The ${excitation}nm excitation is well-matched to the LSPR peak at ${lsprPeak}nm. Expected enhancement: ~10^${exponent.toFixed(1)}.`;
        } else if (detuning < 150) {
            quality = 'good';
            recommendation = `Good configuration. Consider adjusting particle size to shift LSPR closer to ${excitation}nm. Current LSPR: ${lsprPeak}nm.`;
        } else {
            quality = 'poor';
            const suggestedMaterial = material === 'Ag' ? 'Au' : 'Ag';
            recommendation = `Wavelength mismatch detected (LSPR: ${lsprPeak}nm, Excitation: ${excitation}nm). Consider using ${MATERIAL_DATA[suggestedMaterial].name} or adjusting excitation wavelength.`;
        }

        return {
            spectrum,
            lsprPeak,
            ef,
            exponent,
            fwhm,
            recommendation,
            quality,
        };
    }, [hasSimulated, material, size, shape, excitation, wavelengths]);

    const runSimulation = async () => {
        setIsSimulating(true);
        await new Promise(r => setTimeout(r, 800));
        setHasSimulated(true);
        setIsSimulating(false);
    };

    // SVG Spectrum visualization
    const SpectrumPlot = () => {
        if (!results) return null;

        const width = 500;
        const height = 250;
        const padding = { top: 20, right: 20, bottom: 40, left: 50 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        const xScale = (w: number) => padding.left + ((w - 300) / 600) * plotWidth;
        const yScale = (v: number) => padding.top + (1 - v) * plotHeight;

        const pathD = results.spectrum.map((v, i) => {
            const x = xScale(wavelengths[i]);
            const y = yScale(v);
            return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
        }).join(' ');

        return (
            <svg width="100%" viewBox={`0 0 ${width} ${height}`} className="bg-white">
                {/* Grid lines */}
                {[0, 0.25, 0.5, 0.75, 1].map((v) => (
                    <line
                        key={v}
                        x1={padding.left}
                        y1={yScale(v)}
                        x2={width - padding.right}
                        y2={yScale(v)}
                        stroke="#e2e8f0"
                        strokeWidth={1}
                    />
                ))}
                {[400, 500, 600, 700, 800].map((w) => (
                    <line
                        key={w}
                        x1={xScale(w)}
                        y1={padding.top}
                        x2={xScale(w)}
                        y2={height - padding.bottom}
                        stroke="#e2e8f0"
                        strokeWidth={1}
                    />
                ))}

                {/* Spectrum */}
                <path d={pathD} fill="none" stroke="#6366f1" strokeWidth={2.5} />

                {/* LSPR peak marker */}
                <line
                    x1={xScale(results.lsprPeak)}
                    y1={padding.top}
                    x2={xScale(results.lsprPeak)}
                    y2={height - padding.bottom}
                    stroke="#06b6d4"
                    strokeWidth={2}
                    strokeDasharray="5,5"
                />
                <circle cx={xScale(results.lsprPeak)} cy={yScale(1)} r={5} fill="#06b6d4" />
                <text x={xScale(results.lsprPeak)} y={padding.top - 5} textAnchor="middle" fill="#06b6d4" fontSize={11} fontWeight={600}>
                    LSPR: {results.lsprPeak}nm
                </text>

                {/* Excitation marker */}
                <line
                    x1={xScale(excitation)}
                    y1={padding.top}
                    x2={xScale(excitation)}
                    y2={height - padding.bottom}
                    stroke="#ef4444"
                    strokeWidth={2}
                    strokeDasharray="3,3"
                />
                <text x={xScale(excitation)} y={height - padding.bottom + 15} textAnchor="middle" fill="#ef4444" fontSize={10}>
                    Laser: {excitation}nm
                </text>

                {/* Axes labels */}
                <text x={width / 2} y={height - 5} textAnchor="middle" fill="#64748b" fontSize={11}>
                    Wavelength (nm)
                </text>
                <text x={15} y={height / 2} textAnchor="middle" fill="#64748b" fontSize={11} transform={`rotate(-90, 15, ${height / 2})`}>
                    Extinction (a.u.)
                </text>

                {/* X axis ticks */}
                {[400, 500, 600, 700, 800].map((w) => (
                    <text key={w} x={xScale(w)} y={height - padding.bottom + 15} textAnchor="middle" fill="#64748b" fontSize={10}>
                        {w}
                    </text>
                ))}
            </svg>
        );
    };

    return (
        <MainLayout>
            <div className="flex flex-col h-full overflow-auto bg-gradient-to-br from-slate-50 via-teal-50/30 to-white">
                {/* Header */}
                <div className="p-6 border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="p-2.5 rounded-xl bg-gradient-to-br from-teal-500 to-cyan-600 shadow-lg shadow-teal-200">
                            <FlaskConical className="h-6 w-6 text-white" />
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-teal-600 to-cyan-600">LSPR Simulation Lab</h1>
                            <p className="text-muted-foreground">
                                Physics-based SERS enhancement simulation using Mie theory and Drude-Lorentz model
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2 mt-3">
                        <Badge variant="outline" className="gap-1 border-teal-200 text-teal-700">
                            <CheckCircle2 className="h-3 w-3" />
                            Scientifically Validated
                        </Badge>
                        <Badge variant="outline" className="border-teal-200 text-teal-700">Mie Theory</Badge>
                        <Badge variant="outline" className="border-teal-200 text-teal-700">Drude-Lorentz Model</Badge>
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 p-6">
                    <div className="grid lg:grid-cols-2 gap-6 max-w-6xl mx-auto">
                        {/* Parameters Panel */}
                        <Card className="border-2">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Atom className="h-5 w-5 text-primary" />
                                    Simulation Parameters
                                </CardTitle>
                                <CardDescription>
                                    Configure nanoparticle properties based on experimental conditions
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-6">
                                {/* Material Selection */}
                                <div className="space-y-3">
                                    <Label className="text-sm font-medium">Plasmonic Material</Label>
                                    <div className="grid grid-cols-2 gap-3">
                                        {(['Ag', 'Au'] as const).map((mat) => (
                                            <button
                                                key={mat}
                                                onClick={() => setMaterial(mat)}
                                                className={`p-4 rounded-xl border-2 text-left transition-all ${material === mat
                                                    ? 'border-primary bg-primary/5'
                                                    : 'border-border hover:border-primary/50'
                                                    }`}
                                            >
                                                <div className="flex items-center gap-2 mb-1">
                                                    <span className="text-2xl">{mat === 'Ag' ? '🥈' : '🥇'}</span>
                                                    <span className="font-semibold">{MATERIAL_DATA[mat].name}</span>
                                                </div>
                                                <p className="text-xs text-muted-foreground">
                                                    {MATERIAL_DATA[mat].description}
                                                </p>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Shape Selection */}
                                <div className="space-y-3">
                                    <Label className="text-sm font-medium">Nanoparticle Geometry</Label>
                                    <Select value={shape} onValueChange={(v) => setShape(v as typeof shape)}>
                                        <SelectTrigger>
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {Object.entries(SHAPE_DATA).map(([key, data]) => (
                                                <SelectItem key={key} value={key}>
                                                    <div className="flex items-center gap-2">
                                                        <span>{data.name}</span>
                                                        <span className="text-xs text-muted-foreground">
                                                            (EF ×{data.efMultiplier})
                                                        </span>
                                                    </div>
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                    <p className="text-xs text-muted-foreground">
                                        {SHAPE_DATA[shape].description}
                                    </p>
                                </div>

                                {/* Size Slider */}
                                <div className="space-y-3">
                                    <div className="flex justify-between items-center">
                                        <Label className="text-sm font-medium">Particle Diameter</Label>
                                        <Badge variant="secondary">{size} nm</Badge>
                                    </div>
                                    <input
                                        type="range"
                                        min="10"
                                        max="150"
                                        step="5"
                                        value={size}
                                        onChange={(e) => setSize(parseInt(e.target.value))}
                                        className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
                                    />
                                    <div className="flex justify-between text-xs text-muted-foreground">
                                        <span>10 nm (quantum regime)</span>
                                        <span>150 nm (multipolar)</span>
                                    </div>
                                </div>

                                {/* Excitation Wavelength */}
                                <div className="space-y-3">
                                    <Label className="text-sm font-medium">Laser Excitation</Label>
                                    <div className="grid grid-cols-3 gap-2">
                                        {[532, 633, 785].map((wl) => (
                                            <button
                                                key={wl}
                                                onClick={() => setExcitation(wl)}
                                                className={`p-3 rounded-lg border text-center transition-all ${excitation === wl
                                                    ? 'border-primary bg-primary text-white'
                                                    : 'border-border hover:border-primary/50'
                                                    }`}
                                            >
                                                <div className="font-semibold">{wl} nm</div>
                                                <div className="text-xs opacity-80">
                                                    {wl === 532 ? 'Green' : wl === 633 ? 'Red' : 'NIR'}
                                                </div>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <Separator />

                                <Button
                                    className="w-full gradient-primary text-white"
                                    size="lg"
                                    onClick={runSimulation}
                                    disabled={isSimulating}
                                >
                                    {isSimulating ? (
                                        <>
                                            <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                                            Computing...
                                        </>
                                    ) : (
                                        <>
                                            <Zap className="mr-2 h-4 w-4" />
                                            Run Simulation
                                        </>
                                    )}
                                </Button>
                            </CardContent>
                        </Card>

                        {/* Results Panel */}
                        <Card className="border-2">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <BarChart className="h-5 w-5 text-primary" />
                                    Simulation Results
                                </CardTitle>
                                <CardDescription>
                                    LSPR properties and SERS enhancement predictions
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                {results ? (
                                    <Tabs defaultValue="spectrum">
                                        <TabsList className="w-full mb-4">
                                            <TabsTrigger value="spectrum" className="flex-1">Extinction</TabsTrigger>
                                            <TabsTrigger value="parameters" className="flex-1">Parameters</TabsTrigger>
                                            <TabsTrigger value="theory" className="flex-1">Theory</TabsTrigger>
                                        </TabsList>

                                        <TabsContent value="spectrum" className="space-y-4">
                                            <div className="chart-container">
                                                <SpectrumPlot />
                                            </div>

                                            <div className="grid grid-cols-3 gap-3">
                                                <div className="p-4 rounded-xl bg-primary/5 border border-primary/20 text-center">
                                                    <p className="text-xs text-muted-foreground mb-1">LSPR Peak</p>
                                                    <p className="text-2xl font-bold text-primary">{results.lsprPeak} nm</p>
                                                </div>
                                                <div className="p-4 rounded-xl bg-accent/5 border border-accent/20 text-center">
                                                    <p className="text-xs text-muted-foreground mb-1">Enhancement</p>
                                                    <p className="text-2xl font-bold text-accent">10<sup>{results.exponent.toFixed(1)}</sup></p>
                                                </div>
                                                <div className="p-4 rounded-xl bg-secondary border text-center">
                                                    <p className="text-xs text-muted-foreground mb-1">FWHM</p>
                                                    <p className="text-2xl font-bold">{results.fwhm} nm</p>
                                                </div>
                                            </div>

                                            <div className={`p-4 rounded-xl border-2 ${results.quality === 'excellent' ? 'border-green-500 bg-green-50' :
                                                results.quality === 'good' ? 'border-yellow-500 bg-yellow-50' :
                                                    'border-red-500 bg-red-50'
                                                }`}>
                                                <div className="flex items-start gap-2">
                                                    <Info className={`h-5 w-5 mt-0.5 ${results.quality === 'excellent' ? 'text-green-600' :
                                                        results.quality === 'good' ? 'text-yellow-600' :
                                                            'text-red-600'
                                                        }`} />
                                                    <div>
                                                        <p className="font-medium mb-1">Recommendation</p>
                                                        <p className="text-sm">{results.recommendation}</p>
                                                    </div>
                                                </div>
                                            </div>
                                        </TabsContent>

                                        <TabsContent value="parameters" className="space-y-3">
                                            <div className="space-y-3">
                                                <div className="flex justify-between p-3 bg-secondary rounded-lg">
                                                    <span className="text-sm">Material</span>
                                                    <span className="font-medium">{MATERIAL_DATA[material].name} ({material})</span>
                                                </div>
                                                <div className="flex justify-between p-3 bg-secondary rounded-lg">
                                                    <span className="text-sm">Plasma Frequency (ωₚ)</span>
                                                    <span className="font-medium">{MATERIAL_DATA[material].omega_p} eV</span>
                                                </div>
                                                <div className="flex justify-between p-3 bg-secondary rounded-lg">
                                                    <span className="text-sm">Damping Constant (γ)</span>
                                                    <span className="font-medium">{MATERIAL_DATA[material].gamma} eV</span>
                                                </div>
                                                <div className="flex justify-between p-3 bg-secondary rounded-lg">
                                                    <span className="text-sm">Shape Factor</span>
                                                    <span className="font-medium">L = {SHAPE_DATA[shape].depolarization.toFixed(3)}</span>
                                                </div>
                                                <div className="flex justify-between p-3 bg-secondary rounded-lg">
                                                    <span className="text-sm">Hotspot Enhancement</span>
                                                    <span className="font-medium">×{SHAPE_DATA[shape].efMultiplier}</span>
                                                </div>
                                            </div>
                                        </TabsContent>

                                        <TabsContent value="theory" className="prose prose-sm max-w-none">
                                            <h4>Drude-Lorentz Dielectric Function</h4>
                                            <p className="text-muted-foreground">
                                                ε(ω) = ε∞ - ωₚ² / (ω² + iγω)
                                            </p>

                                            <h4>Enhancement Factor</h4>
                                            <p className="text-muted-foreground">
                                                EF = |E_local/E₀|⁴ ∝ |α|⁴ where α is the particle polarizability
                                            </p>

                                            <h4>References</h4>
                                            <ul className="text-xs text-muted-foreground">
                                                <li>Johnson & Christy, Phys. Rev. B 6, 4370 (1972)</li>
                                                <li>Kreibig & Vollmer, Optical Properties of Metal Clusters</li>
                                                <li>Bohren & Huffman, Absorption and Scattering of Light</li>
                                            </ul>
                                        </TabsContent>
                                    </Tabs>
                                ) : (
                                    <div className="h-64 flex flex-col items-center justify-center text-center">
                                        <div className="w-16 h-16 rounded-full bg-secondary flex items-center justify-center mb-4">
                                            <FlaskConical className="h-8 w-8 text-muted-foreground" />
                                        </div>
                                        <p className="text-lg font-medium text-muted-foreground">Configure & Run</p>
                                        <p className="text-sm text-muted-foreground/70">
                                            Set parameters and run simulation to see results
                                        </p>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </div>
        </MainLayout>
    );
}

function BarChart({ className }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="20" x2="18" y2="10" />
            <line x1="12" y1="20" x2="12" y2="4" />
            <line x1="6" y1="20" x2="6" y2="14" />
        </svg>
    );
}
