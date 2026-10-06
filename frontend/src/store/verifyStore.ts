import { create } from "zustand";
import type { ResolutionResult, VerifyStatus } from "@/lib/types";

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
  selectedField: string | null;
  pdfPageIndex: number;
  durationMs: number | null;

  // Actions
  setPdfFile: (file: File | null) => void;
  setExtractionJson: (json: string) => void;
  setStatus: (status: VerifyStatus) => void;
  setProgress: (progress: number) => void;
  setError: (message: string) => void;
  setResults: (results: ResolutionResult[], durationMs: number) => void;
  setSelectedField: (field: string | null) => void;
  setPdfPageIndex: (index: number) => void;
  reset: () => void;
}

const initialState = {
  pdfFile: null,
  extractionJson: "",
  status: "idle" as VerifyStatus,
  progress: 0,
  errorMessage: null,
  results: null,
  selectedField: null,
  pdfPageIndex: 0,
  durationMs: null,
};

export const useVerifyStore = create<VerifyStore>((set) => ({
  ...initialState,

  setPdfFile: (file) => set({ pdfFile: file }),
  setExtractionJson: (json) => set({ extractionJson: json }),
  setStatus: (status) => set({ status }),
  setProgress: (progress) => set({ progress }),
  setError: (message) => set({ status: "error", errorMessage: message }),
  setResults: (results, durationMs) =>
    set({ results, durationMs, status: "success", progress: 100 }),
  setSelectedField: (field) => set({ selectedField: field }),
  setPdfPageIndex: (index) => set({ pdfPageIndex: index }),
  reset: () => set(initialState),
}));
