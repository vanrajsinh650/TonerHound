import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function statusColor(status: string): string {
  switch (status) {
    case "VERIFIED":
      return "#10B981";
    case "HALLUCINATION":
      return "#EF4444";
    case "MISMATCH":
      return "#F59E0B";
    default:
      return "#6B7280";
  }
}

export function statusBgClass(status: string): string {
  switch (status) {
    case "VERIFIED":
      return "bg-green-50 border-green-200 text-green-700";
    case "HALLUCINATION":
      return "bg-red-50 border-red-200 text-red-700";
    case "MISMATCH":
      return "bg-amber-50 border-amber-200 text-amber-700";
    default:
      return "bg-gray-50 border-gray-200 text-gray-700";
  }
}
