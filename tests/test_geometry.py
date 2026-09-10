"""Tests for TonerHound geometry conversions and IoU computations."""

from tonerhound.geometry.coordinates import BBox, union_bbox_list


def test_bbox_properties():
    box = BBox(x=0.1, y=0.2, width=0.3, height=0.4, page=1)
    assert box.x0 == 0.1
    assert box.y0 == 0.2
    assert round(box.x1, 6) == 0.4
    assert round(box.y1, 6) == 0.6
    assert round(box.area, 6) == 0.12
    assert box.to_coco() == [0.1, 0.2, 0.3, 0.4]
    assert box.to_xyxy() == (0.1, 0.2, 0.4, 0.6)
    assert box.to_0_1000() == (200, 100, 600, 400)


def test_bbox_iou_identical():
    b1 = BBox(x=0.1, y=0.1, width=0.2, height=0.2, page=1)
    b2 = BBox(x=0.1, y=0.1, width=0.2, height=0.2, page=1)
    assert b1.iou(b2) == 1.0


def test_bbox_iou_disjoint():
    b1 = BBox(x=0.1, y=0.1, width=0.2, height=0.2, page=1)
    b2 = BBox(x=0.5, y=0.5, width=0.2, height=0.2, page=1)
    assert b1.iou(b2) == 0.0


def test_bbox_iou_diff_page():
    b1 = BBox(x=0.1, y=0.1, width=0.2, height=0.2, page=1)
    b2 = BBox(x=0.1, y=0.1, width=0.2, height=0.2, page=2)
    assert b1.iou(b2) == 0.0


def test_bbox_iou_partial():
    # b1: [0, 0, 2, 2] -> area 4
    # b2: [1, 0, 2, 2] -> area 4
    # intersection: [1, 0, 1, 2] -> area 2
    # union: 4 + 4 - 2 = 6
    # iou: 2 / 6 = 1/3
    b1 = BBox(x=0.0, y=0.0, width=0.2, height=0.2, page=1)
    b2 = BBox(x=0.1, y=0.0, width=0.2, height=0.2, page=1)
    expected_iou = 0.02 / (0.04 + 0.04 - 0.02)
    assert abs(b1.iou(b2) - expected_iou) < 1e-6


def test_from_pdf_points():
    # 612 x 792 standard US Letter page
    # Box at bottom left in PDF: x0=72, y0=72, x1=144, y1=144 (1 inch margin, 1 inch square)
    # In displayed top-left coordinates:
    # x: 72/612 = 0.117647
    # y: 1.0 - (144 / 792) = 0.818182
    box = BBox.from_pdf_points(
        x0=72.0,
        y0=72.0,
        x1=144.0,
        y1=144.0,
        page_width=612.0,
        page_height=792.0,
        page=1,
        origin_bottom_left=True,
    )
    assert abs(box.x - (72.0 / 612.0)) < 1e-5
    assert abs(box.y - (1.0 - 144.0 / 792.0)) < 1e-5
    assert abs(box.width - (72.0 / 612.0)) < 1e-5
    assert abs(box.height - (72.0 / 792.0)) < 1e-5


def test_union_bbox_list():
    b1 = BBox(x=0.1, y=0.1, width=0.1, height=0.1, page=1)
    b2 = BBox(x=0.2, y=0.1, width=0.2, height=0.1, page=1)
    union = union_bbox_list([b1, b2])
    assert union is not None
    assert union.x == 0.1
    assert union.y == 0.1
    assert round(union.width, 6) == 0.3
    assert round(union.height, 6) == 0.1
