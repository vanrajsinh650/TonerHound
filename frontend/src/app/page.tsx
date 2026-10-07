"use client";

import { useCallback, useState } from "react";
import dynamic from "next/dynamic";
import { useVerifyStore } from "@/store/verifyStore";
import { verifyDocument } from "@/lib/api";
import { NavBar } from "@/components/NavBar";
import { PdfDropzone } from "@/components/PdfDropzone";
import { JsonEditor } from "@/components/JsonEditor";
import { ProcessingOverlay } from "@/components/ProcessingOverlay";
import { SummaryCard } from "@/components/SummaryCard";
import { FieldList } from "@/components/FieldList";
import { FileSearch, RotateCcw, FileText, Code2, ListOrdered, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

// react-pdf uses pdfjs-dist which accesses window/document at module init.
// Must be loaded client-only to avoid SSR crash.
const PdfViewer = dynamic(
  () => import("@/components/PdfViewer").then((m) => m.PdfViewer),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full w-full items-center justify-center rounded-xl border border-gray-200 bg-white text-sm text-gray-400">
        Loading PDF viewer…
      </div>
    ),
  }
);

export default function VerifyPage() {
  const store = useVerifyStore();
  const [activeTab, setActiveTab] = useState<"results" | "input">("results");

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

    try {
      JSON.parse(store.extractionJson);
    } catch {
      store.setError("Invalid JSON format. Please fix and try again.");
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
      setActiveTab("results");

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

  const hasResults = store.status === "success" && store.results !== null;

  return (
    <div className="flex h-screen flex-col bg-gray-50 overflow-hidden">
      <NavBar />

      <main className="flex-1 min-h-0 flex flex-col p-3 sm:p-4 overflow-hidden">
        {hasResults ? (
          /* WORKBENCH MODE: Results & PDF side-by-side (Zero window scroll) */
          <div className="flex-1 min-h-0 flex flex-col gap-3 overflow-hidden">
            {/* Top Stats & Actions Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-gray-200 bg-white px-4 py-2.5 shadow-sm shrink-0">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4 text-blue-600" />
                  <span className="text-sm font-semibold text-gray-900 truncate max-w-[200px] sm:max-w-[320px]">
                    {store.pdfFile?.name}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={store.reset}
                  className="flex items-center gap-1.5 rounded-md border border-gray-200 bg-white px-2.5 py-1 text-xs font-medium text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  <RotateCcw className="h-3 w-3 text-gray-500" />
                  New Document
                </button>
              </div>

              {/* Compact Metrics */}
              <SummaryCard
                results={store.results!}
                durationMs={store.durationMs}
                compact
              />
            </div>

            {/* Main Workbench Split View */}
            <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-12 gap-3 overflow-hidden">
              {/* Left Pane: Field List or Edit Inputs */}
              <div className="lg:col-span-5 xl:col-span-4 flex flex-col min-h-0 bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
                {/* View Switcher Tabs */}
                <div className="flex border-b border-gray-200 bg-gray-50/80 px-2 pt-2 shrink-0">
                  <button
                    type="button"
                    onClick={() => setActiveTab("results")}
                    className={cn(
                      "flex items-center gap-1.5 rounded-t-lg px-3 py-1.5 text-xs font-medium border-t border-x -mb-px transition-colors",
                      activeTab === "results"
                        ? "border-gray-200 bg-white text-gray-900 shadow-xs"
                        : "border-transparent text-gray-500 hover:text-gray-800"
                    )}
                  >
                    <ListOrdered className="h-3.5 w-3.5 text-blue-600" />
                    <span>Grounding Results</span>
                    <span className="rounded-full bg-gray-100 px-1.5 py-0.2 text-[10px] text-gray-600">
                      {store.results!.length}
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setActiveTab("input")}
                    className={cn(
                      "flex items-center gap-1.5 rounded-t-lg px-3 py-1.5 text-xs font-medium border-t border-x -mb-px transition-colors",
                      activeTab === "input"
                        ? "border-gray-200 bg-white text-gray-900 shadow-xs"
                        : "border-transparent text-gray-500 hover:text-gray-800"
                    )}
                  >
                    <Code2 className="h-3.5 w-3.5 text-gray-500" />
                    <span>Edit Query JSON</span>
                  </button>
                </div>

                {/* Tab Contents */}
                {activeTab === "results" ? (
                  <FieldList
                    results={store.results!}
                    selectedField={store.selectedField}
                    onSelect={handleFieldSelect}
                  />
                ) : (
                  <div className="flex-1 min-h-0 flex flex-col p-3 space-y-3 overflow-y-auto">
                    <JsonEditor
                      value={store.extractionJson}
                      onChange={store.setExtractionJson}
                      disabled={isProcessing}
                    />
                    <button
                      type="button"
                      onClick={handleVerify}
                      disabled={isProcessing}
                      className="flex items-center justify-center gap-2 rounded-lg bg-gray-900 px-4 py-2 text-xs font-medium text-white hover:bg-gray-800 transition-colors"
                    >
                      <FileSearch className="h-3.5 w-3.5" />
                      Re-verify Grounding
                    </button>
                  </div>
                )}
              </div>

              {/* Right Pane: PDF Viewer with bounding box overlays */}
              <div className="lg:col-span-7 xl:col-span-8 flex flex-col min-h-0 overflow-hidden">
                {store.pdfFile && (
                  <PdfViewer
                    file={store.pdfFile}
                    pageIndex={store.pdfPageIndex}
                    onPageChange={store.setPdfPageIndex}
                    results={store.results!}
                    selectedField={store.selectedField}
                    onSelectField={handleFieldSelect}
                  />
                )}
              </div>
            </div>
          </div>
        ) : (
          /* INITIAL INPUT MODE */
          <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-12 gap-4 overflow-hidden">
            {/* Left Column: Input Form */}
            <div className="lg:col-span-5 xl:col-span-5 flex flex-col justify-between bg-white rounded-xl border border-gray-200 shadow-sm p-5 overflow-y-auto min-h-0">
              <div className="space-y-4">
                <div>
                  <h2 className="text-base font-semibold text-gray-900">
                    Verify Document Grounding
                  </h2>
                  <p className="text-xs text-gray-500 mt-0.5">
                    Map extracted JSON values to exact PDF bounding boxes with zero neural networks.
                  </p>
                </div>

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
              </div>

              <div className="mt-4 space-y-2">
                {/* Verify button */}
                <button
                  type="button"
                  onClick={handleVerify}
                  disabled={!canVerify || isProcessing}
                  className="flex w-full items-center justify-center gap-2 rounded-lg bg-gray-900 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-40 shadow-xs"
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

                {/* Error Banner */}
                {store.status === "error" && (
                  <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-700">
                    {store.errorMessage}
                  </div>
                )}
              </div>
            </div>

            {/* Right Column: PDF Preview or Feature Overview */}
            <div className="lg:col-span-7 xl:col-span-7 flex flex-col min-h-0 overflow-hidden">
              {store.pdfFile ? (
                <PdfViewer
                  file={store.pdfFile}
                  pageIndex={store.pdfPageIndex}
                  onPageChange={store.setPdfPageIndex}
                  results={[]}
                  selectedField={null}
                  onSelectField={() => {}}
                />
              ) : (
                <div className="flex h-full flex-col items-center justify-center rounded-xl border-2 border-dashed border-gray-200 bg-white p-8 text-center">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-50 text-blue-600 mb-3">
                    <Sparkles className="h-6 w-6" />
                  </div>
                  <h3 className="text-sm font-semibold text-gray-900">
                    Deterministic Evidence Resolver
                  </h3>
                  <p className="mt-1 text-xs text-gray-500 max-w-sm">
                    Upload a PDF and supply extracted fields. TonerHound physically grounds evidence using exact geometry and Kuhn-Munkres matching.
                  </p>
                  <div className="mt-5 grid grid-cols-2 gap-3 text-left max-w-sm w-full">
                    <div className="rounded-lg border border-gray-100 bg-gray-50 p-2.5">
                      <p className="text-[11px] font-semibold text-gray-800">Deterministic</p>
                      <p className="text-[10px] text-gray-500">Zero hallucination from AI models</p>
                    </div>
                    <div className="rounded-lg border border-gray-100 bg-gray-50 p-2.5">
                      <p className="text-[11px] font-semibold text-gray-800">Ultra Fast</p>
                      <p className="text-[10px] text-gray-500">Sub-millisecond resolution time</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      <footer className="h-7 border-t border-gray-200 bg-white px-4 flex items-center justify-between shrink-0 text-[11px] text-gray-400">
        <span>TonerHound v0.3.1</span>
        <span>Deterministic Document Grounding Engine</span>
      </footer>
    </div>
  );
}
