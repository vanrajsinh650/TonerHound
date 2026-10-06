"use client";

import { CheckCircle2, XCircle, AlertTriangle, Clock } from "lucide-react";
import type { ResolutionResult } from "@/lib/types";

interface SummaryCardProps {
  results: ResolutionResult[];
  durationMs: number | null;
}

export function SummaryCard({ results, durationMs }: SummaryCardProps) {
  const verified = results.filter((r) => r.status === "VERIFIED").length;
  const hallucinated = results.filter((r) => r.status === "HALLUCINATION").length;
  const mismatched = results.filter((r) => r.status === "MISMATCH").length;

  return (
    <div className="grid grid-cols-4 gap-3">
      <div className="rounded-lg border border-green-200 bg-green-50 p-3">
        <div className="flex items-center gap-1.5">
          <CheckCircle2 className="h-4 w-4 text-green-600" />
          <span className="text-xs font-medium text-green-700">Verified</span>
        </div>
        <p className="mt-1 text-2xl font-bold text-green-900">{verified}</p>
      </div>
      <div className="rounded-lg border border-red-200 bg-red-50 p-3">
        <div className="flex items-center gap-1.5">
          <XCircle className="h-4 w-4 text-red-600" />
          <span className="text-xs font-medium text-red-700">Not Found</span>
        </div>
        <p className="mt-1 text-2xl font-bold text-red-900">{hallucinated}</p>
      </div>
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
        <div className="flex items-center gap-1.5">
          <AlertTriangle className="h-4 w-4 text-amber-600" />
          <span className="text-xs font-medium text-amber-700">Mismatch</span>
        </div>
        <p className="mt-1 text-2xl font-bold text-amber-900">{mismatched}</p>
      </div>
      <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
        <div className="flex items-center gap-1.5">
          <Clock className="h-4 w-4 text-gray-500" />
          <span className="text-xs font-medium text-gray-600">Time</span>
        </div>
        <p className="mt-1 text-2xl font-bold text-gray-900">
          {durationMs ? `${(durationMs / 1000).toFixed(1)}s` : "—"}
        </p>
      </div>
    </div>
  );
}
