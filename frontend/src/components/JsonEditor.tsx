"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";

interface JsonEditorProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

const EXAMPLE_JSON = JSON.stringify(
  {
    invoice_number: "INV-2024-0847",
    total: "$8,420.50",
    due_date: "2024-03-15",
  },
  null,
  2
);

export function JsonEditor({ value, onChange, disabled }: JsonEditorProps) {
  const [error, setError] = useState<string | null>(null);

  const handleChange = (text: string) => {
    onChange(text);
    if (!text.trim()) {
      setError(null);
      return;
    }
    try {
      JSON.parse(text);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-gray-700">
          Extracted JSON
        </label>
        <button
          type="button"
          onClick={() => handleChange(EXAMPLE_JSON)}
          disabled={disabled}
          className="text-xs text-blue-600 hover:text-blue-800 disabled:opacity-50"
        >
          Load example
        </button>
      </div>
      <textarea
        value={value}
        onChange={(e) => handleChange(e.target.value)}
        disabled={disabled}
        placeholder='{\n  "field_name": "extracted value",\n  ...\n}'
        rows={10}
        spellCheck={false}
        className={cn(
          "w-full rounded-lg border bg-white p-3 font-mono text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2",
          error
            ? "border-red-300 focus:ring-red-200"
            : "border-gray-200 focus:ring-blue-200",
          disabled && "cursor-not-allowed opacity-50"
        )}
      />
      {error && (
        <p className="text-xs text-red-500">Invalid JSON: {error}</p>
      )}
    </div>
  );
}
