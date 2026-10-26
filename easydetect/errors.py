# Apache-2.0
"""Errors this package raises. Kept in one place so they are easy to catch."""

from __future__ import annotations


class EasyDetectError(Exception):
    """Base class for every error raised by easydetect."""


class ModelNotFoundError(EasyDetectError, FileNotFoundError):
    """A model name could not be resolved to weights (locally or on the mirror)."""


class DownloadError(EasyDetectError, OSError):
    """A weight download failed."""
