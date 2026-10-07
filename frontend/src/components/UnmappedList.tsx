"use client";

import { useState, useMemo } from "react";
import { Plus, Eye, Search, AlertCircle, Check } from "lucide-react";
import type { UnmappedLine } from "@/lib/types";

interface UnmappedListProps {
  unmappedLines: UnmappedLine[];
  currentPage: number;
  onJumpToPage: (page: number) => void;
  onAddLine: (line: UnmappedLine) => void;
}

export function UnmappedList({
  unmappedLines,
  currentPage,
  onJumpToPage,
  onAddLine,
}: UnmappedListProps) {
  const [filterPageOnly, setFilterPageOnly] = useState(false);
  const [search, setSearch] = useState("");
  const [addedIds, setAddedIds] = useState<Set<string>>(new Set());

  const filteredLines = useMemo(() => {
    return unmappedLines.filter((line) => {
      const matchesPage = !filterPageOnly || line.page === currentPage;
      const matchesSearch =
        search.trim() === "" ||
        line.text.toLowerCase().includes(search.toLowerCase());
      return matchesPage && matchesSearch;
    });
  }, [unmappedLines, filterPageOnly, currentPage, search]);

  const handleAdd = (line: UnmappedLine) => {
    setAddedIds((prev) => new Set(prev).add(line.id));
    onAddLine(line);
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Explanation Banner */}
      <div className="p-3 bg-indigo-50/70 border-b border-indigo-100 shrink-0 text-xs text-indigo-900 space-y-1">
        <div className="flex items-center gap-1.5 font-semibold text-indigo-950">
          <AlertCircle className="h-3.5 w-3.5 text-indigo-600 shrink-0" />
          <span>Text Found in Document, But Omitted from JSON</span>
        </div>
        <p className="text-[11px] text-indigo-700 leading-relaxed">
          These lines exist in the PDF but were not extracted in your query. Click{" "}
          <strong className="font-semibold text-indigo-900">+ Add</strong> to append any line directly to your extraction.
        </p>
      </div>

      {/* Filter and Search Bar */}
      <div className="p-2.5 border-b border-gray-100 space-y-2 shrink-0 bg-gray-50/50">
        <div className="flex items-center justify-between text-xs">
          <span className="font-medium text-gray-700">
            {unmappedLines.length} Unmapped Line{unmappedLines.length === 1 ? "" : "s"}
          </span>
          <label className="flex items-center gap-1.5 cursor-pointer text-gray-600 select-none">
            <input
              type="checkbox"
              checked={filterPageOnly}
              onChange={(e) => setFilterPageOnly(e.target.checked)}
              className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 h-3.5 w-3.5"
            />
            <span className="text-[11px]">Only Page {currentPage}</span>
          </label>
        </div>

        {unmappedLines.length > 3 && (
          <div className="relative">
            <Search className="absolute left-2.5 top-2 h-3.5 w-3.5 text-gray-400" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search unmapped text..."
              className="w-full rounded-md border border-gray-200 bg-white py-1.5 pl-8 pr-3 text-xs text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>
        )}
      </div>

      {/* List */}
      <div className="flex-1 min-h-0 overflow-y-auto p-2.5 space-y-2">
        {filteredLines.length === 0 ? (
          <div className="flex flex-col items-center justify-center p-6 text-center text-gray-400">
            <p className="text-xs">
              {filterPageOnly
                ? `No unmapped text on Page ${currentPage}`
                : "No matching unmapped lines"}
            </p>
          </div>
        ) : (
          filteredLines.map((line) => {
            const isAdded = addedIds.has(line.id);

            return (
              <div
                key={line.id}
                className="group flex flex-col gap-1.5 rounded-lg border border-gray-200 bg-white p-2.5 hover:border-indigo-300 hover:shadow-xs transition-all text-left"
              >
                <div className="flex items-start justify-between gap-2">
                  <p className="font-mono text-xs text-gray-900 leading-snug break-all">
                    "{line.text}"
                  </p>
                </div>

                <div className="flex items-center justify-between gap-2 pt-1 border-t border-gray-50 text-[10px]">
                  <button
                    type="button"
                    onClick={() => onJumpToPage(line.page)}
                    className="flex items-center gap-1 text-gray-500 hover:text-indigo-600 font-medium"
                    title={`Go to Page ${line.page}`}
                  >
                    <Eye className="h-3 w-3 text-gray-400" />
                    <span>Page {line.page}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleAdd(line)}
                    disabled={isAdded}
                    className="flex items-center gap-1 rounded bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 px-2 py-0.5 font-medium text-indigo-700 disabled:opacity-50 disabled:bg-gray-100 disabled:text-gray-400 disabled:border-gray-200 transition-colors"
                  >
                    {isAdded ? (
                      <>
                        <Check className="h-3 w-3 text-green-600" />
                        <span>Added</span>
                      </>
                    ) : (
                      <>
                        <Plus className="h-3 w-3" />
                        <span>+ Add to JSON</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

