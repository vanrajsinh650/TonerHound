"use client";

import { useRef, useEffect, useState, useMemo } from "react";
import { CheckCircle2, XCircle, AlertTriangle, Eye, Search } from "lucide-react";
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
      return <CheckCircle2 className="h-4 w-4 shrink-0 text-green-600" />;
    case "HALLUCINATION":
      return <XCircle className="h-4 w-4 shrink-0 text-red-600" />;
    case "MISMATCH":
      return <AlertTriangle className="h-4 w-4 shrink-0 text-amber-600" />;
    default:
      return null;
  }
}

export function FieldList({ results, selectedField, onSelect }: FieldListProps) {
  const selectedRef = useRef<HTMLDivElement>(null);
  const [filter, setFilter] = useState<"ALL" | "VERIFIED" | "HALLUCINATION" | "MISMATCH">("ALL");
  const [search, setSearch] = useState("");

  const verifiedCount = results.filter((r) => r.status === "VERIFIED").length;
  const hallucinatedCount = results.filter((r) => r.status === "HALLUCINATION").length;
  const mismatchCount = results.filter((r) => r.status === "MISMATCH").length;

  const filteredResults = useMemo(() => {
    return results.filter((r) => {
      const matchesFilter = filter === "ALL" || r.status === filter;
      const matchesSearch =
        search.trim() === "" ||
        r.field.toLowerCase().includes(search.toLowerCase()) ||
        String(r.value).toLowerCase().includes(search.toLowerCase());
      return matchesFilter && matchesSearch;
    });
  }, [results, filter, search]);

  useEffect(() => {
    if (selectedRef.current) {
      selectedRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [selectedField]);

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header with filters and search */}
      <div className="p-3 border-b border-gray-100 space-y-2.5 shrink-0 bg-gray-50/50">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-gray-500">
            Fields ({results.length})
          </span>
          <span className="text-xs text-gray-400">
            Click to locate on PDF
          </span>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5 text-xs">
          <button
            type="button"
            onClick={() => setFilter("ALL")}
            className={cn(
              "rounded-full px-2.5 py-0.5 font-medium transition-colors shrink-0",
              filter === "ALL"
                ? "bg-gray-900 text-white"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            )}
          >
            All ({results.length})
          </button>
          <button
            type="button"
            onClick={() => setFilter("VERIFIED")}
            className={cn(
              "rounded-full px-2.5 py-0.5 font-medium transition-colors shrink-0",
              filter === "VERIFIED"
                ? "bg-green-600 text-white"
                : "bg-green-50 text-green-700 hover:bg-green-100"
            )}
          >
            Verified ({verifiedCount})
          </button>
          {hallucinatedCount > 0 && (
            <button
              type="button"
              onClick={() => setFilter("HALLUCINATION")}
              className={cn(
                "rounded-full px-2.5 py-0.5 font-medium transition-colors shrink-0",
                filter === "HALLUCINATION"
                  ? "bg-red-600 text-white"
                  : "bg-red-50 text-red-700 hover:bg-red-100"
              )}
            >
              Not Found ({hallucinatedCount})
            </button>
          )}
          {mismatchCount > 0 && (
            <button
              type="button"
              onClick={() => setFilter("MISMATCH")}
              className={cn(
                "rounded-full px-2.5 py-0.5 font-medium transition-colors shrink-0",
                filter === "MISMATCH"
                  ? "bg-amber-600 text-white"
                  : "bg-amber-50 text-amber-700 hover:bg-amber-100"
              )}
            >
              Mismatch ({mismatchCount})
            </button>
          )}
        </div>

        {/* Search if more than 4 items */}
        {results.length > 4 && (
          <div className="relative">
            <Search className="absolute left-2.5 top-2 h-3.5 w-3.5 text-gray-400" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Filter by key or value..."
              className="w-full rounded-md border border-gray-200 bg-white py-1.5 pl-8 pr-3 text-xs text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
        )}
      </div>

      {/* Scrollable list */}
      <div className="flex-1 min-h-0 overflow-y-auto p-2.5 space-y-1.5">
        {filteredResults.length === 0 ? (
          <div className="flex flex-col items-center justify-center p-6 text-center text-gray-400">
            <p className="text-xs">No fields match the selected filter</p>
          </div>
        ) : (
          filteredResults.map((r) => {
            const isSelected = r.field === selectedField;

            return (
              <div
                key={r.field}
                ref={isSelected ? selectedRef : null}
                onClick={() => onSelect(r.field)}
                className={cn(
                  "group flex cursor-pointer items-start gap-2.5 rounded-lg border p-2.5 transition-all text-left",
                  isSelected
                    ? "border-blue-500 bg-blue-50/60 ring-2 ring-blue-400/40 shadow-sm"
                    : "border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50/80"
                )}
              >
                <div className="pt-0.5">
                  <StatusIcon status={r.status} />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-1">
                    <p className="truncate text-xs font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">
                      {r.field}
                    </p>
                    <span
                      className={cn(
                        "rounded-full border px-1.5 py-0.5 text-[9px] font-semibold tracking-wide shrink-0",
                        statusBgClass(r.status)
                      )}
                    >
                      {r.status}
                    </span>
                  </div>
                  <p className="mt-0.5 line-clamp-2 font-mono text-[11px] text-gray-600 break-all">
                    {String(r.value)}
                  </p>
                  {r.page && (
                    <div className="mt-1 flex items-center gap-1 text-[10px] text-gray-500">
                      <Eye className="h-3 w-3 text-gray-400" />
                      <span>Page {r.page}</span>
                      {r.confidence && r.confidence > 0 && (
                        <>
                          <span className="text-gray-300">·</span>
                          <span>{(r.confidence * 100).toFixed(0)}% conf</span>
                        </>
                      )}
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
