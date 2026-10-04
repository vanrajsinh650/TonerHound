"""Unit tests for visual checkbox and signature provider."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from tonerhound.vision.checkbox import (
    VisualCheckboxProvider,
    detect_checkbox_with_grid_removal,
    detect_signature_region,
)


def test_detect_checkbox_synthetic() -> None:
    # Create white canvas 200x200
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    # Draw a black square checkbox at (50, 50) of size 20x20
    cv2.rectangle(img, (50, 50), (70, 70), (0, 0, 0), 2)

    bbox_hint = [0.2, 0.2, 0.2, 0.2]  # in [x, y, w, h] normalized
    res = detect_checkbox_with_grid_removal(img, bbox_hint)
    assert res is not None
    state, box = res
    assert state in ("CHECKBOX_UNCHECKED", "CHECKBOX_CHECKED")
    assert 0.2 <= box[0] <= 0.3
    assert 0.2 <= box[1] <= 0.3


def test_detect_signature_synthetic() -> None:
    # Create white canvas 200x200
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    # Draw a line with cursive stroke aspect ratio >= 1.5
    cv2.line(img, (30, 80), (120, 80), (0, 0, 0), 3)
    cv2.line(img, (40, 70), (100, 90), (0, 0, 0), 2)

    bbox_hint = [0.1, 0.3, 0.6, 0.3]
    sig_box = detect_signature_region(img, bbox_hint)
    assert sig_box is not None
    assert sig_box[2] > 0
    assert sig_box[3] > 0


def test_visual_checkbox_provider_wrapper() -> None:
    provider = VisualCheckboxProvider(dpi=150)
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = provider.detect_checkbox(img, [0.1, 0.1, 0.1, 0.1])
    assert res is None  # empty image
