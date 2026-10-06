"use client";

import { CheckCircle2, Loader2 } from "lucide-react";
import type { VerifyStatus } from "@/lib/types";

interface ProcessingOverlayProps {
  status: VerifyStatus;
  progress: number;
}

const STAGES = [
  { key: "validating", label: "Validating inputs" },
  { key: "uploading", label: "Uploading PDF" },
  { key: "verifying", label: "Resolving evidence" },
] as const;

export function ProcessingOverlay({ status, progress }: ProcessingOverlayProps) {
  if (status === "idle" || status === "success" || status === "error") {
    return null;
  }

  const activeIndex = STAGES.findIndex((s) => s.key === status);

  return (
    <div className="flex flex-col items-center gap-4 rounded-lg border border-blue-200 bg-blue-50 p-6">
      <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
      <div className="w-full space-y-2">
        {STAGES.map((stage, i) => {
          const isDone = i < activeIndex;
          const isCurrent = i === activeIndex;

          return (
            <div
              key={stage.key}
              className="flex items-center gap-2 text-sm"
            >
              {isDone ? (
                <CheckCircle2 className="h-4 w-4 text-green-500" />
              ) : isCurrent ? (
                <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
              ) : (
                <div className="h-4 w-4 rounded-full border border-gray-300" />
              )}
              <span
                className={
                  isDone
                    ? "text-green-700"
                    : isCurrent
                      ? "font-medium text-blue-700"
                      : "text-gray-400"
                }
              >
                {stage.label}
              </span>
            </div>
          );
        })}
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-blue-100">
        <div
          className="h-full rounded-full bg-blue-500 transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}
