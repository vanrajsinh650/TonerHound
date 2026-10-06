"use client";

import { cn } from "@/lib/utils";
import { statusColor } from "@/lib/utils";

interface BboxOverlayProps {
  boxes: Array<{
    field: string;
    bbox: [number, number, number, number];
    status: string;
  }>;
  selectedField: string | null;
  onSelect: (field: string) => void;
}

export function BboxOverlay({ boxes, selectedField, onSelect }: BboxOverlayProps) {
  return (
    <svg
      className="absolute inset-0 h-full w-full pointer-events-none"
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
    >
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
              fill={isSelected ? `${color}20` : `${color}10`}
              stroke={color}
              strokeWidth={isSelected ? 0.003 : 0.002}
              className="pointer-events-auto cursor-pointer"
              onClick={() => onSelect(b.field)}
            >
              {isSelected && (
                <animate
                  attributeName="stroke-opacity"
                  values="1;0.4;1"
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
