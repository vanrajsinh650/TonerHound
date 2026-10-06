"use client";

import { useRef, useEffect } from "react";
import { CheckCircle2, XCircle, AlertTriangle, Eye } from "lucide-react";
import { cn, statusBgClass } from "@/lib/utils";
import type { ResolutionResult } from "@/lib/types";

interface FieldListProps {
  results: ResolutionResult[];
  selectedField: string | null;
  onSelect: (field: string) => void;
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case "VERIFIED":
      return <CheckCircle2 className="h-4 w-4 text-green-600" />;
    case "HALLUCINATION":
      return <XCircle className="h-4 w-4 text-red-600" />;
    case "MISMATCH":
      return <AlertTriangle className="h-4 w-4 text-amber-600" />;
    default:
      return null;
  }
}

export function FieldList({ results, selectedField, onSelect }: FieldListProps) {
  const selectedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (selectedRef.current) {
      selectedRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [selectedField]);

  return (
    <div className="space-y-1.5">
      <h3 className="text-sm font-medium text-gray-700">
        Fields ({results.length})
      </h3>
      <div className="max-h-[400px] space-y-1 overflow-y-auto pr-1">
        {results.map((r) => {
          const isSelected = r.field === selectedField;

          return (
            <div
              key={r.field}
              ref={isSelected ? selectedRef : null}
              onClick={() => onSelect(r.field)}
              className={cn(
                "flex cursor-pointer items-center gap-2 rounded-lg border p-2.5 transition-all",
                isSelected
                  ? "border-blue-300 bg-blue-50 ring-2 ring-blue-200"
                  : "border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50"
              )}
            >
              <StatusIcon status={r.status} />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-gray-900">
                  {r.field}
                </p>
                <p className="truncate text-xs text-gray-500">
                  {String(r.value)}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-1.5">
                <span
                  className={cn(
                    "rounded-full border px-2 py-0.5 text-[10px] font-medium",
                    statusBgClass(r.status)
                  )}
                >
                  {r.status}
                </span>
                {r.page && r.bbox && (
                  <span className="flex items-center gap-0.5 text-[10px] text-gray-400">
                    <Eye className="h-3 w-3" />
                    p.{r.page}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
