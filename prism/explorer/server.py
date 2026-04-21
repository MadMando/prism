"""
prism.explorer.server
---------------------
FastAPI backend for the PRISM local explorer.

Endpoints
---------
GET  /            — serves index.html
GET  /api/graph   — full D3-ready graph JSON {nodes, links}
POST /api/retrieve — {query, top_k} → EpistemicResult as JSON
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from prism import PRISM

_prism: Optional[PRISM] = None
_static = Path(__file__).parent / "static"

app = FastAPI(title="PRISM Explorer", docs_url=None, redoc_url=None)


def _graph_to_d3(graph) -> dict:
    sub = graph._g
    degree = dict(sub.degree())
    nodes = []
    for node_id, attrs in sub.nodes(data=True):
        nodes.append({
            "id":      node_id,
            "source":  attrs.get("source", ""),
            "page":    attrs.get("page", 0),
            "section": attrs.get("section", ""),
            "preview": attrs.get("text_preview", "")[:120],
            "group":   attrs.get("source", "unknown"),
            "degree":  degree.get(node_id, 0),
        })
    links = []
    seen = set()
    for u, v, data in sub.edges(data=True):
        key = (u, v, data.get("type", ""))
        if key in seen:
            continue
        seen.add(key)
        links.append({
            "source":     u,
            "target":     v,
            "type":       data.get("type", ""),
            "weight":     round(float(data.get("weight", 0.5)), 4),
            "confidence": round(float(data.get("confidence", 1.0)), 4),
        })
    return {"nodes": nodes, "links": links}


def init_prism(
    lancedb_path: str,
    graph_path: str,
    table_name: str,
    ollama_url: str,
    embed_model: str,
    embed_api_url: Optional[str],
    embed_api_key: Optional[str],
) -> None:
    global _prism
    kwargs: dict = dict(
        graph_path=graph_path,
        lancedb_path=lancedb_path,
        table_name=table_name,
        embed_model=embed_model,
    )
    if embed_api_url:
        kwargs["embed_api_url"] = embed_api_url
        if embed_api_key:
            kwargs["embed_api_key"] = embed_api_key
    else:
        kwargs["ollama_url"] = ollama_url

    _prism = PRISM(**kwargs)
    _prism.load_graph()

    g = _prism.graph._g
    print(f"[prism-explore] Graph loaded — {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((_static / "index.html").read_text(encoding="utf-8"))


@app.get("/api/graph")
def get_graph() -> JSONResponse:
    if _prism is None or _prism.graph is None:
        raise HTTPException(503, "Graph not loaded")
    return JSONResponse(_graph_to_d3(_prism.graph))


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/api/retrieve")
def retrieve(req: RetrieveRequest) -> JSONResponse:
    if _prism is None:
        raise HTTPException(503, "Not initialized")
    result = _prism.retrieve(req.query, top_k=req.top_k)
    return JSONResponse(result.to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prism-explore",
        description="Launch the PRISM local knowledge explorer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lancedb-path",   required=True,  help="Path to your LanceDB directory")
    parser.add_argument("--graph-path",     required=True,  help="Path to prism_graph.json.gz")
    parser.add_argument("--table-name",     default="knowledge", help="LanceDB table name")
    parser.add_argument("--ollama-url",     default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--embed-model",    default="nomic-embed-text", help="Embedding model name")
    parser.add_argument("--embed-api-url",  default=None, help="OpenAI-compatible embedding API URL")
    parser.add_argument("--embed-api-key",  default=None, help="Embedding API key")
    parser.add_argument("--host",           default="127.0.0.1")
    parser.add_argument("--port",           type=int, default=7860)
    args = parser.parse_args()

    init_prism(
        lancedb_path=args.lancedb_path,
        graph_path=args.graph_path,
        table_name=args.table_name,
        ollama_url=args.ollama_url,
        embed_model=args.embed_model,
        embed_api_url=args.embed_api_url,
        embed_api_key=args.embed_api_key,
    )

    import uvicorn
    print(f"[prism-explore] Open http://{args.host}:{args.port} in your browser")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
