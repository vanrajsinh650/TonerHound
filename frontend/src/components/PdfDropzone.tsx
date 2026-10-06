"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileText, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface PdfDropzoneProps {
  file: File | null;
  onFileChange: (file: File | null) => void;
  disabled?: boolean;
}

export function PdfDropzone({ file, onFileChange, disabled }: PdfDropzoneProps) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) {
        const f = accepted[0];
        if (f.size > 25 * 1024 * 1024) {
          alert("File must be under 25 MB.");
          return;
        }
        onFileChange(f);
      }
    },
    [onFileChange]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxFiles: 1,
    disabled,
  });

  if (file) {
    return (
      <div className="flex items-center gap-3 rounded-lg border border-gray-200 bg-gray-50 p-3">
        <FileText className="h-5 w-5 shrink-0 text-gray-500" />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-gray-900">{file.name}</p>
          <p className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
        </div>
        {!disabled && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onFileChange(null);
            }}
            className="rounded p-1 text-gray-400 hover:bg-gray-200 hover:text-gray-600"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>
    );
  }

  return (
    <div
      {...getRootProps()}
      className={cn(
        "flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed p-6 transition-colors",
        isDragActive
          ? "border-blue-400 bg-blue-50"
          : "border-gray-300 bg-gray-50 hover:border-gray-400",
        disabled && "cursor-not-allowed opacity-50"
      )}
    >
      <input {...getInputProps()} />
      <Upload className="h-8 w-8 text-gray-400" />
      <p className="text-sm text-gray-600">
        {isDragActive ? "Drop your PDF here" : "Drop a PDF or click to browse"}
      </p>
      <p className="text-xs text-gray-400">Max 25 MB</p>
    </div>
  );
}
