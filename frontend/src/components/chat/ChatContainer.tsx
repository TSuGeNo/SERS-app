'use client';

import React, { useEffect, useRef } from 'react';
import { cn } from '@/lib/utils';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { useChatStore } from '@/lib/stores';
import { Sparkles } from 'lucide-react';

interface ChatContainerProps {
    className?: string;
}

// Mock AI response - will be replaced with actual API call
const generateMockResponse = async (userMessage: string): Promise<string> => {
    await new Promise((resolve) => setTimeout(resolve, 1500 + Math.random() * 1000));

    const responses: Record<string, string> = {
        analyze: `I've analyzed your SERS spectrum and found the following:

**Peak Detection Results:**
- **611 cm⁻¹**: Strong peak, characteristic of R6G
- **773 cm⁻¹**: Medium intensity, C-H out-of-plane bending
- **1363 cm⁻¹**: Strong peak, aromatic C-C stretching
- **1509 cm⁻¹**: Very strong, xanthene ring stretching

**Molecule Identification:** Rhodamine 6G (R6G) with **95.3% confidence**

**Recommendations:**
1. The signal-to-noise ratio is excellent
2. Consider running concentration regression for quantification
3. Enhancement factor estimated at ~10⁶`,

        simulation: `## LSPR Simulation Results

**Parameters:**
- Material: Silver (Ag)
- Nanoparticle Size: 50 nm (spherical)
- Excitation: 785 nm

**Results:**
| Property | Value |
|----------|-------|
| LSPR Peak | 420 nm |
| Enhancement Factor | ~10⁶ |
| FWHM | 45 nm |

**Recommendation:** Silver nanoparticles at this size show optimal enhancement for Raman excitation at 532-633 nm. For 785 nm excitation, consider using gold nanoparticles or silver nanorods.`,

        default: `I understand you want to analyze your SERS data. Here's what I can help you with:

1. **📊 Spectrum Analysis**: Upload your CSV/TXT data and I'll identify peaks
2. **🔬 LSPR Simulation**: Predict enhancement factors for Ag/Au nanoparticles
3. **🧪 Molecule Detection**: Identify R6G, Crystal Violet, and other SERS probes
4. **🦠 Pathogen Classification**: Classify bacterial species using deep learning
5. **📈 PCA/Clustering**: Dimensionality reduction and pattern recognition

Please upload your data or tell me more about what you'd like to analyze!`,
    };

    const lowerMessage = userMessage.toLowerCase();
    if (lowerMessage.includes('analyze') || lowerMessage.includes('spectrum')) {
        return responses.analyze;
    } else if (lowerMessage.includes('simulation') || lowerMessage.includes('lspr')) {
        return responses.simulation;
    }
    return responses.default;
};

export function ChatContainer({ className }: ChatContainerProps) {
    const scrollRef = useRef < HTMLDivElement > (null);
    const {
        sessions,
        currentSessionId,
        createSession,
        getCurrentSession,
        addMessage,
        updateMessage,
        setIsProcessing,
    } = useChatStore();

    // Create initial session if none exists
    useEffect(() => {
        if (sessions.length === 0) {
            createSession();
        }
    }, [sessions.length, createSession]);

    // Auto-scroll to bottom on new messages
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [getCurrentSession()?.messages]);

    const currentSession = getCurrentSession();

    const handleSendMessage = async (content: string, _files?: File[]) => {
        if (!currentSessionId) return;

        // Add user message
        addMessage(currentSessionId, {
            role: 'user',
            content,
        });

        // Add loading message
        setIsProcessing(true);
        const loadingMessageId = addMessage(currentSessionId, {
            role: 'assistant',
            content: '',
            isLoading: true,
        });

        try {
            // Generate response (mock for now)
            const response = await generateMockResponse(content);

            // Update loading message with actual response
            updateMessage(currentSessionId, loadingMessageId, {
                content: response,
                isLoading: false,
            });
        } catch (error) {
            console.error('Error generating response:', error);
            updateMessage(currentSessionId, loadingMessageId, {
                content: 'Sorry, I encountered an error processing your request. Please try again.',
                isLoading: false,
            });
        } finally {
            setIsProcessing(false);
        }
    };

    return (
        <div className={cn('flex flex-col h-full', className)}>
            {/* Messages Area */}
            <ScrollArea ref={scrollRef} className="flex-1 p-4">
                {currentSession?.messages.length === 0 ? (
                    <WelcomeScreen />
                ) : (
                    <div className="space-y-4 pb-4">
                        {currentSession?.messages.map((message) => (
                            <ChatMessage key={message.id} message={message} />
                        ))}
                    </div>
                )}
            </ScrollArea>

            {/* Input Area */}
            <div className="border-t bg-background/80 backdrop-blur-sm p-4">
                <ChatInput
                    sessionId={currentSessionId || ''}
                    onSend={handleSendMessage}
                />
            </div>
        </div>
    );
}

function WelcomeScreen() {
    return (
        <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <div className="p-4 rounded-full bg-primary/10 mb-4 animate-pulse-glow">
                <Sparkles className="h-10 w-10 text-primary" />
            </div>

            <h1 className="text-3xl font-bold mb-2">
                <span className="gradient-text">SERS-Insight</span>
            </h1>

            <p className="text-xl text-muted-foreground mb-6">
                What do you want to analyze today?
            </p>

            <div className="grid gap-3 max-w-2xl w-full">
                {[
                    {
                        icon: '📊',
                        title: 'Analyze SERS Spectra',
                        description: 'Upload your spectrum data for peak detection and molecule identification',
                    },
                    {
                        icon: '🔬',
                        title: 'Run LSPR Simulation',
                        description: 'Predict enhancement factors for different nanoparticle configurations',
                    },
                    {
                        icon: '🦠',
                        title: 'Pathogen Detection',
                        description: 'Classify bacterial species using deep learning models',
                    },
                    {
                        icon: '🎨',
                        title: 'Create Visualizations',
                        description: 'Generate publication-ready plots and figures',
                    },
                ].map((item) => (
                    <div
                        key={item.title}
                        className="flex items-start gap-3 p-4 bg-card border rounded-xl hover:border-primary/50 hover:bg-card/80 transition-all cursor-pointer group"
                    >
                        <span className="text-2xl group-hover:scale-110 transition-transform">{item.icon}</span>
                        <div className="text-left">
                            <h3 className="font-medium">{item.title}</h3>
                            <p className="text-sm text-muted-foreground">{item.description}</p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
