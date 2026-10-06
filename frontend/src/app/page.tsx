"use client";

import { useCallback } from "react";
import dynamic from "next/dynamic";
import { useVerifyStore } from "@/store/verifyStore";
import { verifyDocument } from "@/lib/api";
import { NavBar } from "@/components/NavBar";
import { PdfDropzone } from "@/components/PdfDropzone";
import { JsonEditor } from "@/components/JsonEditor";
import { ProcessingOverlay } from "@/components/ProcessingOverlay";
import { SummaryCard } from "@/components/SummaryCard";
import { FieldList } from "@/components/FieldList";
import { FileSearch, RotateCcw } from "lucide-react";

// react-pdf uses pdfjs-dist which accesses window/document at module init.
// Must be loaded client-only to avoid SSR crash.
const PdfViewer = dynamic(
  () => import("@/components/PdfViewer").then((m) => m.PdfViewer),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-96 items-center justify-center rounded-lg border border-gray-200 bg-white text-sm text-gray-400">
        Loading PDF viewer…
      </div>
    ),
  }
);

export default function VerifyPage() {
  const store = useVerifyStore();

  const canVerify =
    store.pdfFile !== null &&
    store.extractionJson.trim() !== "" &&
    store.status === "idle";

  const isProcessing =
    store.status === "validating" ||
    store.status === "uploading" ||
    store.status === "verifying";

  const handleVerify = useCallback(async () => {
    if (!store.pdfFile || !store.extractionJson.trim()) return;

    // Validate JSON
    store.setStatus("validating");
    store.setProgress(10);

    let parsed: unknown;
    try {
      parsed = JSON.parse(store.extractionJson);
    } catch {
      store.setError("Invalid JSON. Please fix and try again.");
      return;
    }

    // Upload & verify
    store.setStatus("uploading");
    store.setProgress(30);

    try {
      store.setStatus("verifying");
      store.setProgress(60);

      const response = await verifyDocument(
        store.pdfFile,
        store.extractionJson
      );

      store.setResults(response.results, response.meta.duration_ms);

      // Jump to the first grounded field's page
      const firstGrounded = response.results.find((r) => r.is_grounded && r.page);
      if (firstGrounded && firstGrounded.page) {
        store.setPdfPageIndex(firstGrounded.page - 1);
        store.setSelectedField(firstGrounded.field);
      }
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Verification failed. Is the backend running?";
      store.setError(message);
    }
  }, [store]);

  const handleFieldSelect = useCallback(
    (field: string) => {
      store.setSelectedField(field);
      const result = store.results?.find((r) => r.field === field);
      if (result?.page) {
        store.setPdfPageIndex(result.page - 1);
      }
    },
    [store]
  );

  return (
    <div className="flex min-h-screen flex-col bg-gray-50">
      <NavBar />

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">
        {/* Two-column layout on desktop */}
        <div className="grid gap-6 lg:grid-cols-[2fr_3fr]">
          {/* Left column: Input */}
          <div className="space-y-4">
            <h2 className="text-lg font-semibold text-gray-900">
              Verify Document Grounding
            </h2>

            <PdfDropzone
              file={store.pdfFile}
              onFileChange={store.setPdfFile}
              disabled={isProcessing}
            />

            <JsonEditor
              value={store.extractionJson}
              onChange={store.setExtractionJson}
              disabled={isProcessing}
            />

            {/* Verify button */}
            <button
              onClick={handleVerify}
              disabled={!canVerify || isProcessing}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-gray-900 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <FileSearch className="h-4 w-4" />
              Verify Grounding
            </button>

            {/* Processing overlay */}
            {isProcessing && (
              <ProcessingOverlay
                status={store.status}
                progress={store.progress}
              />
            )}

            {/* Error */}
            {store.status === "error" && (
              <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                {store.errorMessage}
              </div>
            )}

            {/* Reset button (after results) */}
            {store.status === "success" && (
              <button
                onClick={store.reset}
                className="flex w-full items-center justify-center gap-2 rounded-lg border border-gray-200 bg-white px-4 py-2 text-sm text-gray-600 transition-colors hover:bg-gray-50"
              >
                <RotateCcw className="h-4 w-4" />
                New Verification
              </button>
            )}
          </div>

          {/* Right column: Results */}
          <div className="space-y-4">
            {store.status === "success" && store.results ? (
              <>
                <SummaryCard
                  results={store.results}
                  durationMs={store.durationMs}
                />

                {store.pdfFile && (
                  <PdfViewer
                    file={store.pdfFile}
                    pageIndex={store.pdfPageIndex}
                    onPageChange={store.setPdfPageIndex}
                    results={store.results}
                    selectedField={store.selectedField}
                    onSelectField={handleFieldSelect}
                  />
                )}

                <FieldList
                  results={store.results}
                  selectedField={store.selectedField}
                  onSelect={handleFieldSelect}
                />
              </>
            ) : (
              <div className="flex h-full min-h-[400px] flex-col items-center justify-center rounded-lg border-2 border-dashed border-gray-200 bg-white p-8">
                <FileSearch className="h-12 w-12 text-gray-300" />
                <p className="mt-3 text-sm text-gray-400">
                  Upload a PDF and paste extracted JSON to verify grounding
                </p>
                <p className="mt-1 text-xs text-gray-300">
                  Results will appear here with highlighted bounding boxes
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="border-t border-gray-200 bg-white px-6 py-3">
        <p className="text-center text-xs text-gray-400">
          TonerHound v0.3.1 · Deterministic evidence grounding · Zero neural networks
        </p>
      </footer>
    </div>
  );
}
