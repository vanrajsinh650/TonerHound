"use client";

import { statusColor } from "@/lib/utils";

interface BboxOverlayProps {
  boxes: Array<{
    field: string;
    bbox: [number, number, number, number];
    status: string;
  }>;
  unmappedBoxes?: Array<{
    id: string;
    bbox: [number, number, number, number];
    text: string;
  }>;
  showUnmapped?: boolean;
  selectedField: string | null;
  onSelect: (field: string) => void;
  onSelectUnmapped?: (text: string) => void;
}

export function BboxOverlay({
  boxes,
  unmappedBoxes = [],
  showUnmapped = false,
  selectedField,
  onSelect,
  onSelectUnmapped,
}: BboxOverlayProps) {
  return (
    <svg
      className="absolute inset-0 h-full w-full pointer-events-none"
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
    >
      {/* 1. Unmapped lines (faint dashed indigo boxes showing text found in PDF but not in JSON) */}
      {showUnmapped &&
        unmappedBoxes.map((u) => (
          <g key={u.id}>
            <rect
              x={u.bbox[0]}
              y={u.bbox[1]}
              width={u.bbox[2]}
              height={u.bbox[3]}
              fill="#6366F110"
              stroke="#6366F1"
              strokeWidth={0.0018}
              strokeDasharray="0.006, 0.003"
              className="pointer-events-auto cursor-help transition-opacity hover:opacity-80"
              onClick={() => onSelectUnmapped?.(u.text)}
            >
              <title>{`[Unmapped Text]: ${u.text}`}</title>
            </rect>
          </g>
        ))}

      {/* 2. Grounded Extracted Fields */}
      {boxes.map((b) => {
        const isSelected = b.field === selectedField;
        const color = statusColor(b.status);

        return (
          <g key={b.field}>
            <rect
              x={b.bbox[0]}
              y={b.bbox[1]}
              width={b.bbox[2]}
              height={b.bbox[3]}
              fill={isSelected ? `${color}25` : `${color}12`}
              stroke={color}
              strokeWidth={isSelected ? 0.0035 : 0.002}
              className="pointer-events-auto cursor-pointer"
              onClick={() => onSelect(b.field)}
            >
              <title>{`[${b.status}] ${b.field}`}</title>
              {isSelected && (
                <animate
                  attributeName="stroke-opacity"
                  values="1;0.3;1"
                  dur="0.8s"
                  repeatCount="3"
                />
              )}
            </rect>
          </g>
        );
      })}
    </svg>
  );
}
