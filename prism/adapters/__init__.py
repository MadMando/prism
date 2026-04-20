from .base import VectorAdapter
from .embedder import Embedder

__all__ = ["VectorAdapter", "Embedder"]

try:
    from .lancedb import LanceDBAdapter
    __all__.append("LanceDBAdapter")
except ImportError:
    pass

try:
    from .chroma import ChromaAdapter
    __all__.append("ChromaAdapter")
except ImportError:
    pass

try:
    from .qdrant import QdrantAdapter
    __all__.append("QdrantAdapter")
except ImportError:
    pass

try:
    from .weaviate import WeaviateAdapter
    __all__.append("WeaviateAdapter")
except ImportError:
    pass

try:
    from .pgvector import PgvectorAdapter
    __all__.append("PgvectorAdapter")
except ImportError:
    pass
