"use client";

import { useState, useRef, useEffect } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import {
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  Maximize2,
  ScanText,
} from "lucide-react";
import { BboxOverlay } from "./BboxOverlay";
import type { ResolutionResult, UnmappedLine } from "@/lib/types";
import { cn } from "@/lib/utils";

import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface PdfViewerProps {
  file: File;
  pageIndex: number;
  onPageChange: (index: number) => void;
  results: ResolutionResult[];
  unmappedLines?: UnmappedLine[];
  showUnmapped?: boolean;
  onToggleUnmapped?: () => void;
  selectedField: string | null;
  onSelectField: (field: string) => void;
  onSelectUnmappedText?: (text: string) => void;
}

export function PdfViewer({
  file,
  pageIndex,
  onPageChange,
  results,
  unmappedLines = [],
  showUnmapped = false,
  onToggleUnmapped,
  selectedField,
  onSelectField,
  onSelectUnmappedText,
}: PdfViewerProps) {
  const [numPages, setNumPages] = useState<number>(0);
  const [scale, setScale] = useState<number>(1.0);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState<number>(650);

  const currentPage = pageIndex + 1; // react-pdf is 1-indexed

  // Calculate container dimensions to fit comfortably on screen
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.clientWidth);
      }
    };
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  const visibleBoxes = results
    .filter((r) => r.page === currentPage && r.bbox !== null)
    .map((r) => ({
      field: r.field,
      bbox: r.bbox as [number, number, number, number],
      status: r.status,
    }));

  const currentUnmappedBoxes = unmappedLines
    .filter((u) => u.page === currentPage)
    .map((u) => ({
      id: u.id,
      bbox: u.bbox,
      text: u.text,
    }));

  const selectedResult = results.find((r) => r.field === selectedField);

  // Compute base page width: fit container width with reasonable padding
  const baseWidth = Math.max(300, Math.min(containerWidth - 40, 720));
  const renderedWidth = Math.round(baseWidth * scale);

  return (
    <div className="flex flex-col h-full min-h-0 bg-slate-900/5 rounded-xl border border-gray-200 overflow-hidden">
      {/* Top Toolbar */}
      <div className="flex items-center justify-between border-b border-gray-200 bg-white px-3 py-2 shrink-0 gap-2">
        {/* Page navigation */}
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => onPageChange(Math.max(0, pageIndex - 1))}
            disabled={pageIndex === 0}
            className="rounded p-1 text-gray-600 hover:bg-gray-100 disabled:opacity-30 disabled:hover:bg-transparent"
            title="Previous Page"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <span className="px-2 text-xs font-medium text-gray-700 min-w-[75px] text-center">
            {currentPage} / {numPages || "..."}
          </span>
          <button
            type="button"
            onClick={() => onPageChange(Math.min(numPages - 1, pageIndex + 1))}
            disabled={pageIndex >= numPages - 1}
            className="rounded p-1 text-gray-600 hover:bg-gray-100 disabled:opacity-30 disabled:hover:bg-transparent"
            title="Next Page"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>

        {/* Selected Field or Unmapped indicator */}
        <div className="hidden sm:flex items-center gap-1.5 truncate max-w-[280px]">
          {selectedField ? (
            <span className="inline-flex items-center gap-1.5 rounded-md bg-blue-50 px-2 py-0.5 text-xs text-blue-700 font-mono truncate border border-blue-200">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-pulse" />
              <span className="truncate">{selectedField}</span>
              {selectedResult?.page && (
                <span className="text-[10px] text-blue-500">(p.{selectedResult.page})</span>
              )}
            </span>
          ) : (
            <span className="text-xs text-gray-400">Select a field to locate</span>
          )}
        </div>

        {/* Controls: Unmapped toggle + Zoom */}
        <div className="flex items-center gap-1">
          {unmappedLines.length > 0 && onToggleUnmapped && (
            <button
              type="button"
              onClick={onToggleUnmapped}
              className={cn(
                "flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-colors mr-1 border",
                showUnmapped
                  ? "bg-indigo-50 border-indigo-200 text-indigo-700"
                  : "bg-white border-gray-200 text-gray-600 hover:bg-gray-50"
              )}
              title="Toggle dashed boxes for text found in PDF but omitted from JSON"
            >
              <ScanText className="h-3.5 w-3.5" />
              <span className="hidden md:inline">Unmapped ({currentUnmappedBoxes.length})</span>
            </button>
          )}

          <button
            type="button"
            onClick={() => setScale((s) => Math.max(0.6, Number((s - 0.15).toFixed(2))))}
            className="rounded p-1 text-gray-600 hover:bg-gray-100"
            title="Zoom Out"
          >
            <ZoomOut className="h-4 w-4" />
          </button>
          <span className="text-xs font-mono text-gray-500 min-w-[38px] text-center">
            {Math.round(scale * 100)}%
          </span>
          <button
            type="button"
            onClick={() => setScale((s) => Math.min(2.0, Number((s + 0.15).toFixed(2))))}
            className="rounded p-1 text-gray-600 hover:bg-gray-100"
            title="Zoom In"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => setScale(1.0)}
            className="rounded p-1 text-gray-600 hover:bg-gray-100 ml-0.5"
            title="Reset Zoom"
          >
            <Maximize2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* PDF canvas + overlay scrollable viewport */}
      <div
        ref={containerRef}
        className="flex-1 min-h-0 overflow-auto p-4 flex items-start justify-center"
      >
        <Document
          file={file}
          onLoadSuccess={({ numPages: n }) => setNumPages(n)}
          loading={
            <div className="flex h-64 items-center justify-center text-sm text-gray-400">
              Loading document...
            </div>
          }
          error={
            <div className="flex h-64 items-center justify-center text-sm text-red-500 text-center px-4">
              Failed to load PDF. Check that the file is valid.
            </div>
          }
        >
          {/* Relative wrapper ensures BboxOverlay is sized identically to the rendered Page canvas */}
          <div className="relative inline-block shadow-md rounded overflow-hidden bg-white">
            <Page
              pageNumber={currentPage}
              width={renderedWidth}
              renderTextLayer={false}
              renderAnnotationLayer={false}
            />
            <BboxOverlay
              boxes={visibleBoxes}
              unmappedBoxes={currentUnmappedBoxes}
              showUnmapped={showUnmapped}
              selectedField={selectedField}
              onSelect={onSelectField}
              onSelectUnmapped={onSelectUnmappedText}
            />
          </div>
        </Document>
      </div>
    </div>
  );
}
