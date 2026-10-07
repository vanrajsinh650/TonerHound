import { create } from "zustand";
import type { DocumentCoverage, ResolutionResult, UnmappedLine, VerifyStatus } from "@/lib/types";

interface VerifyStore {
  // Input
  pdfFile: File | null;
  extractionJson: string;

  // Process
  status: VerifyStatus;
  progress: number;
  errorMessage: string | null;

  // Results
  results: ResolutionResult[] | null;
  coverage: DocumentCoverage | null;
  selectedField: string | null;
  pdfPageIndex: number;
  durationMs: number | null;
  showUnmapped: boolean;

  // Actions
  setPdfFile: (file: File | null) => void;
  setExtractionJson: (json: string) => void;
  setStatus: (status: VerifyStatus) => void;
  setProgress: (progress: number) => void;
  setError: (message: string) => void;
  setResults: (
    results: ResolutionResult[],
    durationMs: number,
    coverage?: DocumentCoverage
  ) => void;
  setSelectedField: (field: string | null) => void;
  setPdfPageIndex: (index: number) => void;
  setShowUnmapped: (show: boolean | ((prev: boolean) => boolean)) => void;
  addUnmappedLineToExtraction: (line: UnmappedLine) => string;
  reset: () => void;
}

const initialState = {
  pdfFile: null,
  extractionJson: "",
  status: "idle" as VerifyStatus,
  progress: 0,
  errorMessage: null,
  results: null,
  coverage: null,
  selectedField: null,
  pdfPageIndex: 0,
  durationMs: null,
  showUnmapped: true,
};

export const useVerifyStore = create<VerifyStore>((set, get) => ({
  ...initialState,

  setPdfFile: (file) => set({ pdfFile: file }),
  setExtractionJson: (json) => set({ extractionJson: json }),
  setStatus: (status) => set({ status }),
  setProgress: (progress) => set({ progress }),
  setError: (message) => set({ status: "error", errorMessage: message }),
  setResults: (results, durationMs, coverage) =>
    set({
      results,
      durationMs,
      coverage: coverage || null,
      status: "success",
      progress: 100,
    }),
  setSelectedField: (field) => set({ selectedField: field }),
  setPdfPageIndex: (index) => set({ pdfPageIndex: index }),
  setShowUnmapped: (show) =>
    set((state) => ({
      showUnmapped: typeof show === "function" ? show(state.showUnmapped) : show,
    })),

  addUnmappedLineToExtraction: (line: UnmappedLine) => {
    const current = get().extractionJson;
    let data: Record<string, unknown> = {};
    try {
      if (current.trim()) {
        data = JSON.parse(current);
      }
    } catch {
      data = {};
    }

    // Generate a clean snake_case field key from the line's first 3-4 words
    const cleanWords = line.text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, "")
      .trim()
      .split(/\s+/)
      .slice(0, 4)
      .join("_");
    const key = cleanWords ? `extracted_${cleanWords}` : `field_p${line.page}`;

    // Ensure unique key
    let uniqueKey = key;
    let counter = 1;
    while (uniqueKey in data) {
      uniqueKey = `${key}_${counter++}`;
    }

    data[uniqueKey] = line.text;
    const newJson = JSON.stringify(data, null, 2);
    set({ extractionJson: newJson, selectedField: uniqueKey });
    return newJson;
  },

  reset: () => set(initialState),
}));
