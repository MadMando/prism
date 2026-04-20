"""
PRISM — Epistemic Graph RAG with Spreading Activation
======================================================

A novel retrieval library that layers a typed epistemic graph over an existing
vector store, then uses spreading activation (Collins & Loftus, 1975) to retrieve
knowledge that is not just semantically similar but epistemically structured.

Quick start
-----------
    from prism import PRISM

    p = PRISM(
        lancedb_path = "/path/to/lancedb",
        graph_path   = "/path/to/prism_graph.json.gz",
        ollama_url   = "http://localhost:11434",
        embed_model  = "qwen3-embedding:4b",
        llm_base_url = "https://api.deepseek.com",
        llm_model    = "deepseek-chat",
        llm_api_key  = "sk-...",
    )

    # Build the epistemic graph (one-time, offline)
    p.build(max_pairs=50_000)

    # Retrieve with full epistemic structuring
    result = p.retrieve("what is data stewardship accountability")
    print(result.format_for_llm())
"""

from .prism import PRISM
from .result import EpistemicResult, EpistemicChunk, ActivationPath
from .edges import EpistemicEdgeType, EdgeValence, PROPAGATION_WEIGHTS
from .graph import EpistemicGraph
from .retriever import PRISMRetriever, Reranker
from .extractor import EpistemicExtractor
from .filter import EpistemicFilter
from .activation import SpreadingActivation, NodeActivation
from .adapters.base import VectorAdapter
from .adapters.embedder import Embedder

__version__ = "0.2.5"
__author__  = "PRISM Contributors"

__all__ = [
    # Main interface
    "PRISM",
    # Results
    "EpistemicResult",
    "EpistemicChunk",
    "ActivationPath",
    # Graph & edges
    "EpistemicGraph",
    "EpistemicEdgeType",
    "EdgeValence",
    "PROPAGATION_WEIGHTS",
    # Internals (for advanced use)
    "PRISMRetriever",
    "Reranker",
    "EpistemicExtractor",
    "EpistemicFilter",
    "SpreadingActivation",
    "NodeActivation",
    # Adapter interface
    "VectorAdapter",
    "Embedder",
]

try:
    from .adapters.lancedb import LanceDBAdapter
    __all__.append("LanceDBAdapter")
except ImportError:
    pass

try:
    from .adapters.chroma import ChromaAdapter
    __all__.append("ChromaAdapter")
except ImportError:
    pass

try:
    from .adapters.qdrant import QdrantAdapter
    __all__.append("QdrantAdapter")
except ImportError:
    pass

try:
    from .adapters.weaviate import WeaviateAdapter
    __all__.append("WeaviateAdapter")
except ImportError:
    pass

try:
    from .adapters.pgvector import PgvectorAdapter
    __all__.append("PgvectorAdapter")
except ImportError:
    pass
