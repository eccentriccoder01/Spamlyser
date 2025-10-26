"""Compatibility shim for tests expecting top-level threat_analyzer.

Re-exports selected symbols from models.threat_analyzer.
"""
from models.threat_analyzer import (
    classify_threat_type,
    get_threat_specific_advice,
    THREAT_CATEGORIES,
)

__all__ = [
    "classify_threat_type",
    "get_threat_specific_advice",
    "THREAT_CATEGORIES",
]
