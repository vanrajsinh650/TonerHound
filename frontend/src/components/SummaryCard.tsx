"use client";

import { CheckCircle2, XCircle, AlertTriangle, Clock, Layers } from "lucide-react";
import type { DocumentCoverage, ResolutionResult } from "@/lib/types";

interface SummaryCardProps {
  results: ResolutionResult[];
  durationMs: number | null;
  coverage?: DocumentCoverage | null;
  compact?: boolean;
}

export function SummaryCard({
  results,
  durationMs,
  coverage,
  compact = false,
}: SummaryCardProps) {
  const verified = results.filter((r) => r.status === "VERIFIED").length;
  const hallucinated = results.filter((r) => r.status === "HALLUCINATION").length;
  const mismatched = results.filter((r) => r.status === "MISMATCH").length;

  if (compact) {
    return (
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1.5 rounded-lg border border-green-200 bg-green-50 px-2.5 py-1 text-xs">
          <CheckCircle2 className="h-3.5 w-3.5 text-green-600" />
          <span className="font-semibold text-green-800">{verified}</span>
          <span className="text-green-700">Verified</span>
        </div>
        <div className="flex items-center gap-1.5 rounded-lg border border-red-200 bg-red-50 px-2.5 py-1 text-xs">
          <XCircle className="h-3.5 w-3.5 text-red-600" />
          <span className="font-semibold text-red-800">{hallucinated}</span>
          <span className="text-red-700">Not Found</span>
        </div>
        {mismatched > 0 && (
          <div className="flex items-center gap-1.5 rounded-lg border border-amber-200 bg-amber-50 px-2.5 py-1 text-xs">
            <AlertTriangle className="h-3.5 w-3.5 text-amber-600" />
            <span className="font-semibold text-amber-800">{mismatched}</span>
            <span className="text-amber-700">Mismatch</span>
          </div>
        )}
        {coverage && (
          <div
            className="flex items-center gap-1.5 rounded-lg border border-indigo-200 bg-indigo-50 px-2.5 py-1 text-xs"
            title={`${coverage.mapped_lines} of ${coverage.total_lines} lines mapped`}
          >
            <Layers className="h-3.5 w-3.5 text-indigo-600" />
            <span className="font-semibold text-indigo-800">
              {coverage.coverage_percent}%
            </span>
            <span className="text-indigo-700 hidden sm:inline">Coverage</span>
          </div>
        )}
        <div className="flex items-center gap-1.5 rounded-lg border border-gray-200 bg-white px-2.5 py-1 text-xs text-gray-600">
          <Clock className="h-3.5 w-3.5 text-gray-400" />
          <span className="font-medium text-gray-800">
            {durationMs ? `${(durationMs / 1000).toFixed(2)}s` : "—"}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-4 gap-2.5">
      <div className="rounded-lg border border-green-200 bg-green-50 p-2.5">
        <div className="flex items-center gap-1.5">
          <CheckCircle2 className="h-3.5 w-3.5 text-green-600" />
          <span className="text-[11px] font-medium text-green-700">Verified</span>
        </div>
        <p className="mt-0.5 text-xl font-bold text-green-900">{verified}</p>
      </div>
      <div className="rounded-lg border border-red-200 bg-red-50 p-2.5">
        <div className="flex items-center gap-1.5">
          <XCircle className="h-3.5 w-3.5 text-red-600" />
          <span className="text-[11px] font-medium text-red-700">Not Found</span>
        </div>
        <p className="mt-0.5 text-xl font-bold text-red-900">{hallucinated}</p>
      </div>
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-2.5">
        <div className="flex items-center gap-1.5">
          <AlertTriangle className="h-3.5 w-3.5 text-amber-600" />
          <span className="text-[11px] font-medium text-amber-700">Mismatch</span>
        </div>
        <p className="mt-0.5 text-xl font-bold text-amber-900">{mismatched}</p>
      </div>
      <div className="rounded-lg border border-gray-200 bg-gray-50 p-2.5">
        <div className="flex items-center gap-1.5">
          <Clock className="h-3.5 w-3.5 text-gray-500" />
          <span className="text-[11px] font-medium text-gray-600">Time</span>
        </div>
        <p className="mt-0.5 text-xl font-bold text-gray-900">
          {durationMs ? `${(durationMs / 1000).toFixed(1)}s` : "—"}
        </p>
      </div>
    </div>
  );
}
