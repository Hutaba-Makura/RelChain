"""RelChain public API."""

from .config import MemoryConfig, RelationMode
from .pipeline import MemorySystem

__all__ = ["MemoryConfig", "MemorySystem", "RelationMode"]
__version__ = "0.1.0"
