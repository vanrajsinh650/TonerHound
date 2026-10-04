"""Computer vision utilities for document grounding."""

from tonerhound.vision.checkbox import (
    VisualCheckboxProvider,
    detect_checkbox_with_grid_removal,
    detect_signature_region,
)

__all__ = [
    "VisualCheckboxProvider",
    "detect_checkbox_with_grid_removal",
    "detect_signature_region",
]
