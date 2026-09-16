"""Experimental perception interface; no automatic identity commitment."""

from .video import prepare
from .vlm import VisionClient

__all__ = ["prepare", "VisionClient"]
