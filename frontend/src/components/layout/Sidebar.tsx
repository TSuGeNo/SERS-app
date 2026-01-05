'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    Home,
    FlaskConical,
    BarChart3,
    Workflow,
    Plus,
    ChevronLeft,
    ChevronRight,
    Settings,
    HelpCircle,
    Sparkles,
    Waves,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from '@/components/ui/tooltip';

interface SidebarProps {
    collapsed?: boolean;
    onToggle?: () => void;
    className?: string;
}

const navItems = [
    { icon: Home, label: 'Dashboard', href: '/', description: 'Home & AI Assistant' },
    { icon: FlaskConical, label: 'Simulation Lab', href: '/simulate', description: 'LSPR & Enhancement' },
    { icon: BarChart3, label: 'Visualizations', href: '/visualize', description: 'Spectral Analysis' },
    { icon: Waves, label: 'Peak Analyzer', href: '/peak-analyzer', description: 'Baseline Correction' },
    { icon: Workflow, label: 'Workflows', href: '/workflows', description: 'Scientific Pipelines' },
];

export function Sidebar({ collapsed = false, onToggle, className }: SidebarProps) {
    const pathname = usePathname();

    return (
        <TooltipProvider delayDuration={0}>
            <div
                className={cn(
                    'flex flex-col h-full bg-white text-foreground border-r border-slate-200 transition-all duration-300',
                    collapsed ? 'w-16' : 'w-64',
                    className
                )}
            >
                {/* Logo */}
                <div className="flex items-center justify-between p-4 border-b border-slate-100">
                    {!collapsed && (
                        <Link href="/" className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center shadow-md shadow-primary/20">
                                <Sparkles className="w-5 h-5 text-white" />
                            </div>
                            <div>
                                <span className="font-bold text-lg text-slate-900">SERS-Insight</span>
                                <p className="text-xs text-slate-500">Spectroscopy Platform</p>
                            </div>
                        </Link>
                    )}

                    {collapsed && (
                        <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center mx-auto shadow-md shadow-primary/20">
                            <Sparkles className="w-5 h-5 text-white" />
                        </div>
                    )}
                </div>

                {/* Toggle Button */}
                <Button
                    variant="ghost"
                    size="icon"
                    onClick={onToggle}
                    className={cn('absolute top-20 -right-3 h-6 w-6 rounded-full border bg-white shadow-md z-10 hover:bg-slate-50')}
                >
                    {collapsed ? (
                        <ChevronRight className="h-3 w-3 text-slate-600" />
                    ) : (
                        <ChevronLeft className="h-3 w-3 text-slate-600" />
                    )}
                </Button>

                {/* New Analysis Button */}
                <div className="p-4">
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Link href="/">
                                <Button
                                    className={cn(
                                        'w-full justify-start gap-2 gradient-primary text-white shadow-lg shadow-primary/20 hover:shadow-primary/30 transition-shadow',
                                        collapsed && 'justify-center px-0'
                                    )}
                                >
                                    <Plus className="h-4 w-4" />
                                    {!collapsed && 'New Analysis'}
                                </Button>
                            </Link>
                        </TooltipTrigger>
                        {collapsed && <TooltipContent side="right">New Analysis</TooltipContent>}
                    </Tooltip>
                </div>

                {/* Navigation */}
                <nav className="flex-1 px-3 py-2">
                    <p className={cn("text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 px-3", collapsed && "hidden")}>
                        Navigation
                    </p>
                    {navItems.map((item) => {
                        const isActive = pathname === item.href;
                        return (
                            <Tooltip key={item.href}>
                                <TooltipTrigger asChild>
                                    <Link
                                        href={item.href}
                                        className={cn(
                                            'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all mb-1',
                                            isActive
                                                ? 'bg-primary/10 text-primary'
                                                : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900',
                                            collapsed && 'justify-center px-0'
                                        )}
                                    >
                                        <item.icon className={cn("h-5 w-5", isActive ? "text-primary" : "text-slate-500")} />
                                        {!collapsed && (
                                            <div className="flex-1">
                                                <span>{item.label}</span>
                                            </div>
                                        )}
                                    </Link>
                                </TooltipTrigger>
                                {collapsed && (
                                    <TooltipContent side="right" className="flex flex-col">
                                        <span className="font-medium">{item.label}</span>
                                        <span className="text-xs text-slate-400">{item.description}</span>
                                    </TooltipContent>
                                )}
                            </Tooltip>
                        );
                    })}
                </nav>

                {/* SERS Illustration */}
                {!collapsed && (
                    <div className="px-4 py-2">
                        <div className="p-3 bg-slate-50 rounded-lg">
                            <svg width="100%" viewBox="0 0 180 60" className="opacity-50">
                                {/* Nanoparticle */}
                                <circle cx="30" cy="30" r="15" fill="url(#sidebar-gold)" />
                                <circle cx="50" cy="35" r="10" fill="url(#sidebar-silver)" />

                                {/* Light interaction */}
                                <line x1="70" y1="10" x2="45" y2="30" stroke="#6366f1" strokeWidth="1.5" strokeDasharray="3,2" />
                                <line x1="45" y1="30" x2="80" y2="20" stroke="#06b6d4" strokeWidth="1" opacity="0.6" />
                                <line x1="45" y1="30" x2="75" y2="40" stroke="#06b6d4" strokeWidth="1" opacity="0.6" />

                                {/* Spectrum */}
                                <path d="M100 40 Q110 30 120 35 Q130 40 140 25 Q150 10 160 30 Q170 50 175 35"
                                    stroke="#6366f1" strokeWidth="1.5" fill="none" />
                                <circle cx="140" cy="25" r="2" fill="#ef4444" />

                                <defs>
                                    <linearGradient id="sidebar-gold" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" stopColor="#fbbf24" />
                                        <stop offset="100%" stopColor="#f59e0b" />
                                    </linearGradient>
                                    <linearGradient id="sidebar-silver" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" stopColor="#94a3b8" />
                                        <stop offset="100%" stopColor="#64748b" />
                                    </linearGradient>
                                </defs>
                            </svg>
                            <p className="text-[10px] text-center text-slate-400 mt-1">AI-Powered SERS Analysis</p>
                        </div>
                    </div>
                )}

                {/* Bottom Actions */}
                <div className="mt-auto border-t border-slate-100 p-3">
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Link
                                href="/settings"
                                className={cn(
                                    'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-slate-500 hover:bg-slate-50 hover:text-slate-700 transition-all',
                                    collapsed && 'justify-center px-0'
                                )}
                            >
                                <Settings className="h-5 w-5" />
                                {!collapsed && 'Settings'}
                            </Link>
                        </TooltipTrigger>
                        {collapsed && <TooltipContent side="right">Settings</TooltipContent>}
                    </Tooltip>

                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Link
                                href="/help"
                                className={cn(
                                    'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-slate-500 hover:bg-slate-50 hover:text-slate-700 transition-all',
                                    collapsed && 'justify-center px-0'
                                )}
                            >
                                <HelpCircle className="h-5 w-5" />
                                {!collapsed && 'Documentation'}
                            </Link>
                        </TooltipTrigger>
                        {collapsed && <TooltipContent side="right">Documentation</TooltipContent>}
                    </Tooltip>
                </div>
            </div>
        </TooltipProvider>
    );
}
