export type VerifyStatus =
  | "idle"
  | "validating"
  | "uploading"
  | "verifying"
  | "success"
  | "error";

export type FieldStatus = "VERIFIED" | "HALLUCINATION" | "MISMATCH";

export interface ResolutionResult {
  field: string;
  value: string;
  status: FieldStatus;
  page: number | null;
  bbox: [number, number, number, number] | null;
  is_grounded: boolean;
  matched_text?: string | null;
  confidence?: number;
}

export interface UnmappedLine {
  id: string;
  page: number;
  text: string;
  bbox: [number, number, number, number];
}

export interface DocumentCoverage {
  total_lines: number;
  mapped_lines: number;
  coverage_percent: number;
  unmapped_lines: UnmappedLine[];
}

export interface VerifyResponse {
  results: ResolutionResult[];
  meta: {
    fields_processed: number;
    duration_ms: number;
    coverage?: DocumentCoverage;
  };
}

export interface ExtractionField {
  field: string;
  value: string;
}
