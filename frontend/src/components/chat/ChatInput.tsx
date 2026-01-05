'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Sparkles, Settings2, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
    DropdownMenuSeparator,
    DropdownMenuLabel,
} from '@/components/ui/dropdown-menu';
import { FileUploadZone } from '@/components/upload/FileUploadZone';
import { useChatStore, useAnalysisStore } from '@/lib/stores';

interface ChatInputProps {
    sessionId: string;
    onSend: (message: string, files?: File[]) => void;
    placeholder?: string;
    disabled?: boolean;
    className?: string;
}

const quickActions = [
    { icon: '📊', label: 'Analyze spectrum', prompt: 'Analyze this SERS spectrum and identify peaks' },
    { icon: '🔬', label: 'Run simulation', prompt: 'Run LSPR simulation for silver nanoparticles at 785nm' },
    { icon: '🧪', label: 'Detect molecule', prompt: 'Detect R6G from this spectrum' },
    { icon: '🦠', label: 'Pathogen detection', prompt: 'Classify the bacterial species from this SERS data' },
    { icon: '📈', label: 'PCA analysis', prompt: 'Perform PCA analysis on the uploaded dataset' },
    { icon: '🎨', label: 'Visualize data', prompt: 'Create a visualization of the spectrum with peak annotations' },
];

export function ChatInput({ sessionId, onSend, placeholder, disabled, className }: ChatInputProps) {
    const [message, setMessage] = useState('');
    const [showUpload, setShowUpload] = useState(false);
    const [pendingFiles, setPendingFiles] = useState < File[] > ([]);
    const textareaRef = useRef < HTMLTextAreaElement > (null);
    const { isProcessing } = useChatStore();
    const { frameworks } = useAnalysisStore();

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
        }
    }, [message]);

    const handleSend = () => {
        if ((!message.trim() && pendingFiles.length === 0) || disabled || isProcessing) return;

        onSend(message.trim(), pendingFiles);
        setMessage('');
        setPendingFiles([]);
        setShowUpload(false);

        // Reset textarea height
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleQuickAction = (prompt: string) => {
        setMessage(prompt);
        textareaRef.current?.focus();
    };

    const handleFilesUploaded = (files: File[]) => {
        setPendingFiles((prev) => [...prev, ...files]);
    };

    return (
        <div className={cn('space-y-3', className)}>
            {/* Quick Actions */}
            <div className="flex flex-wrap gap-2 px-1">
                {quickActions.slice(0, 4).map((action) => (
                    <button
                        key={action.label}
                        onClick={() => handleQuickAction(action.prompt)}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full bg-secondary hover:bg-secondary/80 transition-colors"
                    >
                        <span>{action.icon}</span>
                        <span>{action.label}</span>
                    </button>
                ))}
                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <button className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full bg-secondary hover:bg-secondary/80 transition-colors">
                            <Sparkles className="h-3.5 w-3.5" />
                            <span>More</span>
                        </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start" className="w-56">
                        <DropdownMenuLabel>Quick Actions</DropdownMenuLabel>
                        <DropdownMenuSeparator />
                        {quickActions.slice(4).map((action) => (
                            <DropdownMenuItem
                                key={action.label}
                                onClick={() => handleQuickAction(action.prompt)}
                            >
                                <span className="mr-2">{action.icon}</span>
                                {action.label}
                            </DropdownMenuItem>
                        ))}
                    </DropdownMenuContent>
                </DropdownMenu>
            </div>

            {/* File Upload Zone */}
            {showUpload && (
                <div className="border rounded-xl p-4 bg-card">
                    <FileUploadZone
                        sessionId={sessionId}
                        onFilesUploaded={handleFilesUploaded}
                    />
                </div>
            )}

            {/* Main Input Area */}
            <div className="relative">
                <div className="flex items-end gap-2 p-3 bg-card border rounded-2xl shadow-sm focus-within:ring-2 focus-within:ring-primary/20 focus-within:border-primary/50 transition-all">
                    {/* Attachment Button */}
                    <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="h-9 w-9 shrink-0 rounded-full"
                        onClick={() => setShowUpload(!showUpload)}
                    >
                        <Paperclip className={cn('h-5 w-5', showUpload && 'text-primary')} />
                    </Button>

                    {/* Text Input */}
                    <Textarea
                        ref={textareaRef}
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder={placeholder || 'Ask anything about your SERS data...'}
                        disabled={disabled || isProcessing}
                        className="flex-1 min-h-[40px] max-h-[200px] resize-none border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 py-2.5"
                        rows={1}
                    />

                    {/* Framework Selector */}
                    <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                            <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                className="h-9 w-9 shrink-0 rounded-full"
                            >
                                <Settings2 className="h-5 w-5" />
                            </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="w-64">
                            <DropdownMenuLabel>Analysis Frameworks</DropdownMenuLabel>
                            <DropdownMenuSeparator />
                            {frameworks.map((framework) => (
                                <DropdownMenuItem
                                    key={framework.id}
                                    onClick={() => handleQuickAction(`Use ${framework.name} framework`)}
                                >
                                    <span className="mr-2">{framework.icon}</span>
                                    <div>
                                        <p className="font-medium">{framework.name}</p>
                                        <p className="text-xs text-muted-foreground">{framework.description}</p>
                                    </div>
                                </DropdownMenuItem>
                            ))}
                        </DropdownMenuContent>
                    </DropdownMenu>

                    {/* Send Button */}
                    <Button
                        type="button"
                        size="icon"
                        className="h-9 w-9 shrink-0 rounded-full gradient-primary"
                        onClick={handleSend}
                        disabled={(!message.trim() && pendingFiles.length === 0) || disabled || isProcessing}
                    >
                        {isProcessing ? (
                            <Loader2 className="h-5 w-5 animate-spin" />
                        ) : (
                            <Send className="h-5 w-5" />
                        )}
                    </Button>
                </div>

                {/* Pending Files Indicator */}
                {pendingFiles.length > 0 && (
                    <div className="absolute -top-2 right-4 px-2 py-0.5 bg-primary text-primary-foreground text-xs rounded-full">
                        {pendingFiles.length} file{pendingFiles.length > 1 ? 's' : ''} attached
                    </div>
                )}
            </div>

            {/* Disclaimer */}
            <p className="text-[10px] text-center text-muted-foreground">
                SERS-Insight may produce inaccurate results. Verify important analyses independently.
            </p>
        </div>
    );
}
