"""
DPeM (Dual-Process enhanced Memory) module for MaLP.

Provides:
- WorkingMemory: Buffer for newly detected information
- Short_Term_Memory: Fuzzy-match key-value store (Levenshtein distance)
- Long_Term_Memory: Semantic retrieval store (cosine similarity)
- Memory: Unified coordinator for all three memory types
"""

from .dynamic_memory import Short_Term_Memory
from .static_memory import Long_Term_Memory
from .memory import Memory, WorkingMemory

__all__ = [
    "Short_Term_Memory",
    "Long_Term_Memory",
    "Memory",
    "WorkingMemory",
]
