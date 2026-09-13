"""Geometry abstractions and coordinate space conversions for TonerHound.

Supported coordinate systems:
- NORMALIZED_0_1: COCO format [x, y, width, height] normalized to [0.0, 1.0].
  Used by ExtractBench.
- PDF_POINTS: Bottom-left or top-left points (72 points/inch standard PDF media box).
- NORMALIZED_0_1000: Anchorite-style integer grid [top, left, bottom, right] in 0-1000.
- PIXELS: Image pixel coordinates [x, y, width, height].
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CoordinateFrame(str, Enum):
    NORMALIZED_0_1 = "normalized_0_1"
    NORMALIZED_0_1000 = "normalized_0_1000"
    PDF_POINTS_TOP_LEFT = "pdf_points_top_left"
    PDF_POINTS_BOTTOM_LEFT = "pdf_points_bottom_left"
    PIXELS = "pixels"


@dataclass(frozen=True, slots=True)
class BBox:
    """Normalized COCO bounding box [x, y, width, height] in [0.0, 1.0].

    x: Left coordinate (0.0 to 1.0)
    y: Top coordinate (0.0 to 1.0)
    width: Width (0.0 to 1.0)
    height: Height (0.0 to 1.0)
    page: 1-indexed page number
    """

    x: float
    y: float
    width: float
    height: float
    page: int = 1

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError(f"Page must be >= 1 (1-indexed), got {self.page}")
        if self.width < 0.0 or self.height < 0.0:
            raise ValueError(f"Width and height must be >= 0.0, got ({self.width}, {self.height})")

    @property
    def x0(self) -> float:
        return self.x

    @property
    def y0(self) -> float:
        return self.y

    @property
    def x1(self) -> float:
        return self.x + self.width

    @property
    def y1(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_coco(self) -> list[float]:
        """Return [x, y, width, height] rounded for stability."""
        return [round(self.x, 6), round(self.y, 6), round(self.width, 6), round(self.height, 6)]

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return [x0, y0, x1, y1] rounded for numerical stability."""
        return (
            round(self.x, 6),
            round(self.y, 6),
            round(self.x + self.width, 6),
            round(self.y + self.height, 6),
        )

    def to_0_1000(self) -> tuple[int, int, int, int]:
        """Return (top, left, bottom, right) in 0-1000 integer space (Anchorite format)."""
        top = round(self.y * 1000)
        left = round(self.x * 1000)
        bottom = round((self.y + self.height) * 1000)
        right = round((self.x + self.width) * 1000)
        return (top, left, bottom, right)

    @classmethod
    def from_xyxy(cls, x0: float, y0: float, x1: float, y1: float, page: int = 1) -> BBox:
        left = min(x0, x1)
        top = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        return cls(x=left, y=top, width=w, height=h, page=page)

    @classmethod
    def from_pdf_points(
        cls,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        page_width: float,
        page_height: float,
        page: int = 1,
        origin_bottom_left: bool = True,
    ) -> BBox:
        """Convert PDF points to normalized [0.0, 1.0] top-left COCO bbox."""
        if page_width <= 0 or page_height <= 0:
            raise ValueError(f"Invalid page dimensions: {page_width}x{page_height}")

        norm_x0 = max(0.0, min(1.0, x0 / page_width))
        norm_x1 = max(0.0, min(1.0, x1 / page_width))

        if origin_bottom_left:
            # Standard PDF coordinates: (0,0) is bottom-left
            norm_y0 = max(0.0, min(1.0, 1.0 - (y1 / page_height)))
            norm_y1 = max(0.0, min(1.0, 1.0 - (y0 / page_height)))
        else:
            norm_y0 = max(0.0, min(1.0, y0 / page_height))
            norm_y1 = max(0.0, min(1.0, y1 / page_height))

        return cls.from_xyxy(norm_x0, norm_y0, norm_x1, norm_y1, page=page)

    def intersection(self, other: BBox) -> BBox | None:
        """Compute intersection between two bboxes on the same page."""
        if self.page != other.page:
            return None
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        return BBox.from_xyxy(ix0, iy0, ix1, iy1, page=self.page)

    def union_hull(self, other: BBox) -> BBox:
        """Compute enclosing bounding box (convex rectangular hull) of two bboxes."""
        if self.page != other.page:
            raise ValueError(f"Cannot union boxes across different pages ({self.page} vs {other.page})")
        ux0 = min(self.x0, other.x0)
        uy0 = min(self.y0, other.y0)
        ux1 = max(self.x1, other.x1)
        uy1 = max(self.y1, other.y1)
        return BBox.from_xyxy(ux0, uy0, ux1, uy1, page=self.page)

    def iou(self, other: BBox) -> float:
        """Compute exact IoU matching ExtractBench's iou_xywh."""
        if self.page != other.page:
            return 0.0
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        inter_area = inter.area
        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return 0.0
        return min(1.0, inter_area / union_area)

    def sub_bbox(self, start_char: int, end_char: int, total_chars: int) -> BBox:
        """Interpolate horizontal bounding box for a character substring within a token."""
        if total_chars <= 0 or start_char >= end_char:
            return self
        start_ratio = max(0.0, min(1.0, start_char / total_chars))
        end_ratio = max(0.0, min(1.0, end_char / total_chars))
        new_x = self.x + start_ratio * self.width
        new_w = max(0.001, (end_ratio - start_ratio) * self.width)
        return BBox(x=new_x, y=self.y, width=new_w, height=self.height, page=self.page)

    def align_to_line_height(self, target_height: float = 0.018, max_y: float = 1.0) -> BBox:
        """Normalize tight font glyph heights to standard visual text line boundaries."""
        if self.height >= target_height:
            return self
        pad = (target_height - self.height) / 2.0
        new_y = max(0.0, self.y - pad)
        new_h = min(max_y - new_y, target_height)
        return BBox(x=self.x, y=new_y, width=self.width, height=new_h, page=self.page)



def union_bbox_list(boxes: list[BBox]) -> BBox | None:
    """Compute rectangular hull for a list of boxes on the same page."""
    if not boxes:
        return None
    page = boxes[0].page
    x0 = min(b.x0 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    x1 = max(b.x1 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    return BBox.from_xyxy(x0, y0, x1, y1, page=page)
