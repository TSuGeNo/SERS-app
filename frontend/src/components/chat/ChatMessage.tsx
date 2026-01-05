'use client';

import React from 'react';
import { User, Bot, FileText, Image, Copy, Check } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Message, Visualization } from '@/lib/stores';
import ReactMarkdown from 'react-markdown';

interface ChatMessageProps {
    message: Message;
    className?: string;
}

export function ChatMessage({ message, className }: ChatMessageProps) {
    const [copied, setCopied] = React.useState(false);
    const isUser = message.role === 'user';
    const isAssistant = message.role === 'assistant';

    const handleCopy = async () => {
        await navigator.clipboard.writeText(message.content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div
            className={cn(
                'flex gap-3 animate-in fade-in-0 slide-in-from-bottom-2 duration-300',
                isUser && 'flex-row-reverse',
                className
            )}
        >
            {/* Avatar */}
            <div
                className={cn(
                    'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
                    isUser ? 'gradient-primary' : 'bg-secondary'
                )}
            >
                {isUser ? (
                    <User className="h-4 w-4 text-white" />
                ) : (
                    <Bot className="h-4 w-4 text-foreground" />
                )}
            </div>

            {/* Message Content */}
            <div className={cn('flex flex-col gap-2 max-w-[85%]', isUser && 'items-end')}>
                {/* Attachments */}
                {message.attachments && message.attachments.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                        {message.attachments.map((attachment) => (
                            <div
                                key={attachment.id}
                                className="flex items-center gap-2 px-3 py-2 bg-secondary rounded-lg text-sm"
                            >
                                {attachment.type.startsWith('image/') ? (
                                    <Image className="h-4 w-4" />
                                ) : (
                                    <FileText className="h-4 w-4" />
                                )}
                                <span className="truncate max-w-[150px]">{attachment.name}</span>
                            </div>
                        ))}
                    </div>
                )}

                {/* Text Bubble */}
                <div
                    className={cn(
                        'chat-bubble',
                        isUser ? 'chat-bubble-user' : 'chat-bubble-assistant'
                    )}
                >
                    {message.isLoading ? (
                        <div className="flex items-center gap-2">
                            <div className="flex gap-1">
                                <span className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                                <span className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                                <span className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                            </div>
                            <span className="text-sm opacity-70">Analyzing...</span>
                        </div>
                    ) : (
                        <div className="prose prose-sm dark:prose-invert max-w-none">
                            <ReactMarkdown
                                components={{
                                    // eslint-disable-next-line @typescript-eslint/no-unused-vars
                                    code: ({ className, children, ...props }) => {
                                        const isInline = !className;
                                        return isInline ? (
                                            <code className="px-1.5 py-0.5 bg-black/10 dark:bg-white/10 rounded text-sm" {...props}>
                                                {children}
                                            </code>
                                        ) : (
                                            <code className={cn('block p-3 bg-black/10 dark:bg-white/10 rounded-lg overflow-x-auto', className)} {...props}>
                                                {children}
                                            </code>
                                        );
                                    },
                                    // eslint-disable-next-line @typescript-eslint/no-unused-vars
                                    p: ({ node: _node, ...props }) => <p className="mb-2 last:mb-0" {...props} />,
                                    // eslint-disable-next-line @typescript-eslint/no-unused-vars
                                    ul: ({ node: _node, ...props }) => <ul className="list-disc pl-4 mb-2" {...props} />,
                                    // eslint-disable-next-line @typescript-eslint/no-unused-vars
                                    ol: ({ node: _node, ...props }) => <ol className="list-decimal pl-4 mb-2" {...props} />,
                                    // eslint-disable-next-line @typescript-eslint/no-unused-vars
                                    li: ({ node: _node, ...props }) => <li className="mb-1" {...props} />,
                                }}
                            >
                                {message.content}
                            </ReactMarkdown>
                        </div>
                    )}
                </div>

                {/* Visualizations */}
                {message.visualizations && message.visualizations.length > 0 && (
                    <div className="grid gap-2 w-full">
                        {message.visualizations.map((viz) => (
                            <VisualizationCard key={viz.id} visualization={viz} />
                        ))}
                    </div>
                )}

                {/* Actions */}
                {isAssistant && !message.isLoading && (
                    <div className="flex items-center gap-1">
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 rounded-full"
                            onClick={handleCopy}
                        >
                            {copied ? (
                                <Check className="h-4 w-4 text-green-500" />
                            ) : (
                                <Copy className="h-4 w-4" />
                            )}
                        </Button>
                    </div>
                )}

                {/* Timestamp */}
                <span className="text-[10px] text-muted-foreground">
                    {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
            </div>
        </div>
    );
}

interface VisualizationCardProps {
    visualization: Visualization;
}

function VisualizationCard({ visualization }: VisualizationCardProps) {
    return (
        <div className="border rounded-xl p-4 bg-card">
            <div className="flex items-center justify-between mb-3">
                <h4 className="font-medium">{visualization.title}</h4>
                <span className="text-xs text-muted-foreground capitalize px-2 py-0.5 bg-secondary rounded-full">
                    {visualization.type}
                </span>
            </div>

            {/* Placeholder for actual chart - will be replaced with Plotly/Recharts */}
            <div className="aspect-[16/9] bg-gradient-to-br from-secondary to-muted rounded-lg flex items-center justify-center">
                <span className="text-muted-foreground">Chart: {visualization.type}</span>
            </div>
        </div>
    );
}
