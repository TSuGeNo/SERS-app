'use client';

import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, X, CheckCircle2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface FileUploadZoneProps {
    sessionId?: string;
    onFilesUploaded?: (files: File[]) => void;
    acceptedFiles?: { [key: string]: string[] };
    maxFiles?: number;
}

const defaultAcceptedFiles = {
    'text/csv': ['.csv'],
    'text/plain': ['.txt'],
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    'application/json': ['.json'],
};

export function FileUploadZone({
    sessionId = 'default',
    onFilesUploaded,
    acceptedFiles = defaultAcceptedFiles,
    maxFiles = 5,
}: FileUploadZoneProps) {
    const [uploadedFiles, setUploadedFiles] = React.useState < File[] > ([]);

    const onDrop = useCallback(
        (accepted: File[]) => {
            const newFiles = [...uploadedFiles, ...accepted].slice(0, maxFiles);
            setUploadedFiles(newFiles);
            onFilesUploaded?.(newFiles);
        },
        [uploadedFiles, maxFiles, onFilesUploaded]
    );

    const removeFile = (index: number) => {
        const newFiles = uploadedFiles.filter((_, i) => i !== index);
        setUploadedFiles(newFiles);
        onFilesUploaded?.(newFiles);
    };

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: acceptedFiles,
        maxFiles: maxFiles - uploadedFiles.length,
    });

    const formatFileSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    return (
        <div className="space-y-3">
            {/* Dropzone */}
            <div
                {...getRootProps()}
                className={`upload-zone p-8 text-center cursor-pointer transition-all ${isDragActive ? 'drag-active border-primary bg-primary/5' : ''
                    }`}
            >
                <input {...getInputProps()} />
                <Upload
                    className={`h-10 w-10 mx-auto mb-3 ${isDragActive ? 'text-primary' : 'text-muted-foreground'
                        }`}
                />
                <p className="font-medium mb-1">
                    {isDragActive ? 'Drop files here' : 'Drag & drop SERS data files'}
                </p>
                <p className="text-sm text-muted-foreground">
                    Supports CSV, TXT, XLSX, JSON
                </p>
                <div className="flex flex-wrap justify-center gap-2 mt-3">
                    <Badge variant="secondary">CSV</Badge>
                    <Badge variant="secondary">TXT</Badge>
                    <Badge variant="secondary">XLSX</Badge>
                    <Badge variant="secondary">JSON</Badge>
                </div>
            </div>

            {/* Uploaded files list */}
            {uploadedFiles.length > 0 && (
                <div className="space-y-2">
                    {uploadedFiles.map((file, index) => (
                        <div
                            key={index}
                            className="flex items-center gap-3 p-3 bg-secondary rounded-lg"
                        >
                            <FileText className="h-5 w-5 text-primary shrink-0" />
                            <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium truncate">{file.name}</p>
                                <p className="text-xs text-muted-foreground">
                                    {formatFileSize(file.size)}
                                </p>
                            </div>
                            <CheckCircle2 className="h-4 w-4 text-green-500" />
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-7 w-7"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    removeFile(index);
                                }}
                            >
                                <X className="h-4 w-4" />
                            </Button>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
