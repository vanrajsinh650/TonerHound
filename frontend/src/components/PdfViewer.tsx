"use client";

import { useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { BboxOverlay } from "./BboxOverlay";
import type { ResolutionResult } from "@/lib/types";

import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface PdfViewerProps {
  file: File;
  pageIndex: number;
  onPageChange: (index: number) => void;
  results: ResolutionResult[];
  selectedField: string | null;
  onSelectField: (field: string) => void;
}

export function PdfViewer({
  file,
  pageIndex,
  onPageChange,
  results,
  selectedField,
  onSelectField,
}: PdfViewerProps) {
  const [numPages, setNumPages] = useState<number>(0);
  const [containerWidth, setContainerWidth] = useState<number>(700);

  const currentPage = pageIndex + 1; // react-pdf is 1-indexed

  const visibleBoxes = results
    .filter((r) => r.page === currentPage && r.bbox !== null)
    .map((r) => ({
      field: r.field,
      bbox: r.bbox as [number, number, number, number],
      status: r.status,
    }));

  return (
    <div className="flex flex-col gap-2">
      {/* Page controls */}
      <div className="flex items-center justify-between rounded-lg border border-gray-200 bg-gray-50 px-3 py-1.5">
        <button
          onClick={() => onPageChange(Math.max(0, pageIndex - 1))}
          disabled={pageIndex === 0}
          className="rounded p-1 text-gray-500 hover:bg-gray-200 disabled:opacity-30"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        <span className="text-sm text-gray-600">
          Page {currentPage} of {numPages || "..."}
        </span>
        <button
          onClick={() => onPageChange(Math.min(numPages - 1, pageIndex + 1))}
          disabled={pageIndex >= numPages - 1}
          className="rounded p-1 text-gray-500 hover:bg-gray-200 disabled:opacity-30"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>

      {/* PDF canvas + overlay */}
      <div
        className="relative overflow-auto rounded-lg border border-gray-200 bg-white"
        ref={(el) => {
          if (el) setContainerWidth(el.clientWidth);
        }}
      >
        <Document
          file={file}
          onLoadSuccess={({ numPages: n }) => setNumPages(n)}
          loading={
            <div className="flex h-96 items-center justify-center text-sm text-gray-400">
              Loading PDF...
            </div>
          }
          error={
            <div className="flex h-96 items-center justify-center text-sm text-red-500">
              Failed to load PDF. The file may be corrupted or password-protected.
            </div>
          }
        >
          <Page
            pageNumber={currentPage}
            width={containerWidth - 2}
            renderTextLayer={false}
            renderAnnotationLayer={false}
          />
          <BboxOverlay
            boxes={visibleBoxes}
            selectedField={selectedField}
            onSelect={onSelectField}
          />
        </Document>
      </div>
    </div>
  );
}
