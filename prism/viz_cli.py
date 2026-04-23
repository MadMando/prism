"""
prism.viz_cli
-------------
prism-viz — export a PRISM epistemic graph to Gephi (GEXF) or D3 (JSON)
for visualisation and exploration.

Usage
-----
    # Export to Gephi GEXF
    prism-viz graph.json.gz --format gexf --output graph.gexf

    # Export to D3 JSON (force-directed graph)
    prism-viz graph.json.gz --format d3 --output graph.json

    # Filter: only include specific edge types
    prism-viz graph.json.gz --format d3 --edge-types supports,refutes

    # Filter: only high-confidence edges
    prism-viz graph.json.gz --format gexf --min-confidence 0.8

    # Filter: only nodes from a specific source
    prism-viz graph.json.gz --format d3 --source-filter "dmbok"

    # Cap size for large graphs (sample by degree centrality)
    prism-viz graph.json.gz --format d3 --max-nodes 500

D3 JSON format
--------------
    {
      "nodes": [{"id": "...", "source": "...", "page": 1, "section": "...",
                 "group": "source-name", "degree": 4}],
      "links": [{"source": "...", "target": "...", "type": "supports",
                 "weight": 0.85, "confidence": 0.9}]
    }

    Load in D3 with d3.forceSimulation — colour nodes by "group", scale
    link opacity by "weight".

GEXF format
-----------
    Standard Gephi GEXF — open directly in Gephi for layout and analysis.
    Edge type, weight, confidence, and rationale are exported as attributes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


def _build_subgraph(graph, edge_types=None, min_confidence=0.0, source_filter=None):
    """Return a filtered copy of the underlying networkx graph."""
    import networkx as nx

    g = graph._g
    sub = nx.MultiDiGraph()

    for node_id, attrs in g.nodes(data=True):
        src = attrs.get("source", "")
        if source_filter and source_filter.lower() not in src.lower():
            continue
        sub.add_node(node_id, **attrs)

    for u, v, data in g.edges(data=True):
        if u not in sub.nodes or v not in sub.nodes:
            continue
        etype = data.get("type", "")
        conf  = float(data.get("confidence", 1.0))
        if edge_types and etype not in edge_types:
            continue
        if conf < min_confidence:
            continue
        sub.add_edge(u, v, **data)

    return sub


def _sample_by_degree(sub, max_nodes: int):
    """Keep the top-N nodes by total degree."""
    import networkx as nx
    degree = dict(sub.degree())
    top_ids = set(sorted(degree, key=lambda n: -degree[n])[:max_nodes])
    return sub.subgraph(top_ids).copy()


def _export_gexf(sub, output_path: Path) -> None:
    import networkx as nx

    # GEXF doesn't allow "type" as an attribute name — rename to edge_type
    renamed = nx.MultiDiGraph()
    for node_id, attrs in sub.nodes(data=True):
        renamed.add_node(node_id, **attrs)
    for u, v, data in sub.edges(data=True):
        edge_attrs = {
            "edge_type":  data.get("type", ""),
            "weight":     float(data.get("weight", 0.5)),
            "confidence": float(data.get("confidence", 1.0)),
            "rationale":  str(data.get("rationale", "")),
        }
        renamed.add_edge(u, v, **edge_attrs)

    nx.write_gexf(renamed, str(output_path))
    print(f"[prism-viz] GEXF written → {output_path}")
    print(f"[prism-viz] {renamed.number_of_nodes():,} nodes, {renamed.number_of_edges():,} edges")
    print(f"[prism-viz] Open in Gephi: File → Open → {output_path}")


def _export_d3(sub, output_path: Path) -> None:
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
    for u, v, data in sub.edges(data=True):
        links.append({
            "source":     u,
            "target":     v,
            "type":       data.get("type", ""),
            "weight":     round(float(data.get("weight", 0.5)), 4),
            "confidence": round(float(data.get("confidence", 1.0)), 4),
        })

    payload = {"nodes": nodes, "links": links}

    if str(output_path) == "-":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[prism-viz] D3 JSON written → {output_path}")
        print(f"[prism-viz] {len(nodes):,} nodes, {len(links):,} links")
        print(f"[prism-viz] Load with: d3.json('{output_path}').then(data => ...)")


def _export_html(sub, output_path: Path) -> None:
    """Export a self-contained interactive HTML viewer with embedded graph data."""
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
    for u, v, data in sub.edges(data=True):
        links.append({
            "source":     u,
            "target":     v,
            "type":       data.get("type", ""),
            "weight":     round(float(data.get("weight", 0.5)), 4),
            "confidence": round(float(data.get("confidence", 1.0)), 4),
        })

    payload = json.dumps({"nodes": nodes, "links": links}, ensure_ascii=False, separators=(",", ":"))

    today      = date.today().isoformat()
    n_nodes    = len(nodes)
    n_links    = len(links)
    stats_text = f"{n_nodes:,} nodes · {n_links:,} edges · exported {today}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PRISM Graph — {today}</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0a0a0f;--surface:#111118;--border:#1e1e2e;--text:#d4d4d8;--muted:#52525b;--accent:#a78bfa}}
body{{font-family:'SF Mono','Fira Code',monospace;background:var(--bg);color:var(--text);height:100vh;display:grid;grid-template-columns:200px 1fr;grid-template-rows:44px 1fr;overflow:hidden}}
header{{grid-column:1/-1;background:#0d0d1a;border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 14px;gap:10px}}
header h1{{font-size:13px;font-weight:700;color:var(--accent);letter-spacing:.12em}}
#stats{{font-size:11px;color:var(--muted);margin-left:auto}}
#panel{{background:var(--surface);border-right:1px solid var(--border);padding:14px 12px;display:flex;flex-direction:column;gap:16px;overflow-y:auto}}
#graph-wrap{{position:relative;overflow:hidden;background:var(--bg)}}
#graph{{width:100%;height:100%;display:block}}
.sl{{font-size:9px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}}
input[type=text]{{background:#0d0d1a;border:1px solid var(--border);color:var(--text);border-radius:4px;font-family:inherit;font-size:12px;padding:5px 8px;width:100%}}
input[type=text]:focus{{outline:none;border-color:var(--accent)}}
.sr{{display:flex;align-items:center;gap:8px}}.sr input[type=range]{{flex:1;accent-color:var(--accent)}}
.sv{{font-size:11px;color:var(--muted);width:30px;text-align:right}}
#elist{{display:flex;flex-direction:column;gap:3px}}
.er{{display:flex;align-items:center;gap:7px;font-size:11px;cursor:pointer;padding:2px 0;user-select:none}}
.er input{{accent-color:var(--accent);margin:0;flex-shrink:0}}
.es{{width:10px;height:10px;border-radius:2px;flex-shrink:0}}
#tip{{position:fixed;background:rgba(13,13,26,.97);border:1px solid #2a2a4a;border-radius:5px;padding:8px 11px;font-size:11px;pointer-events:none;z-index:999;max-width:260px;display:none;line-height:1.5}}
.ts{{color:var(--accent);font-weight:600;margin-bottom:3px}}.tt{{color:#a1a1aa}}
::-webkit-scrollbar{{width:3px}}::-webkit-scrollbar-thumb{{background:#2a2a3a;border-radius:2px}}
</style>
</head>
<body>
<header>
  <h1>◈ PRISM — Epistemic Graph</h1>
  <span id="stats">{stats_text}</span>
</header>
<div id="panel">
  <div><div class="sl">Source Filter</div><input type="text" id="sf" placeholder="substring…"></div>
  <div><div class="sl">Min Confidence</div><div class="sr"><input type="range" id="cs" min="0" max="1" step="0.05" value="0"><span class="sv" id="cv">0.00</span></div></div>
  <div><div class="sl">Edge Types</div><div id="elist"></div></div>
</div>
<div id="graph-wrap"><svg id="graph"></svg></div>
<div id="tip"></div>
<script>
const EDGE_COLORS={{supports:'#4CAF50',refutes:'#F44336',supersedes:'#FF9800',derives_from:'#9C27B0',specializes:'#2196F3',contrasts_with:'#E91E63',implements:'#00BCD4',generalizes:'#8D6E63',exemplifies:'#FDD835',qualifies:'#607D8B'}};
const srcColor=d3.scaleOrdinal(d3.schemeTableau10);
let activeTypes=new Set(Object.keys(EDGE_COLORS));
const DATA={payload};

// Build edge toggles
const elist=document.getElementById('elist');
for(const[type,color]of Object.entries(EDGE_COLORS)){{
  const l=document.createElement('label');l.className='er';
  l.innerHTML=`<input type="checkbox" checked data-e="${{type}}"><span class="es" style="background:${{color}}"></span><span>${{type.replace(/_/g,' ')}}</span>`;
  elist.appendChild(l);
}}
elist.addEventListener('change',()=>{{activeTypes=new Set([...elist.querySelectorAll('input:checked')].map(e=>e.dataset.e));applyVis();}});

const slider=document.getElementById('cs');
slider.addEventListener('input',()=>{{document.getElementById('cv').textContent=parseFloat(slider.value).toFixed(2);applyVis();}});
document.getElementById('sf').addEventListener('input',applyVis);

const nid=n=>typeof n==='object'?n.id:n;

const wrap=document.getElementById('graph-wrap');
const W=wrap.clientWidth||window.innerWidth-200;
const H=wrap.clientHeight||window.innerHeight-44;
const svg=d3.select('#graph').attr('width',W).attr('height',H);
const defs=svg.append('defs');
for(const[type,color]of Object.entries(EDGE_COLORS)){{
  defs.append('marker').attr('id','a-'+type).attr('viewBox','0 -4 10 8').attr('refX',22).attr('refY',0).attr('markerWidth',7).attr('markerHeight',7).attr('orient','auto')
    .append('path').attr('d','M0,-4L10,0L0,4').attr('fill',color);
}}
const g=svg.append('g');
const nodes=DATA.nodes.map(d=>({{...d}}));
const links=DATA.links.map(d=>({{...d}}));
const r=d=>Math.max(6,6+Math.sqrt(d.degree||0)*2.8);

let lSel=g.append('g').selectAll('line').data(links).join('line')
  .attr('stroke',d=>EDGE_COLORS[d.type]||'#555')
  .attr('stroke-width',d=>Math.max(2.5,d.weight*5))
  .attr('stroke-opacity',d=>0.6+d.weight*0.4)
  .attr('marker-end',d=>'url(#a-'+d.type+')');

const tip=document.getElementById('tip');
let nSel=g.append('g').selectAll('circle').data(nodes).join('circle')
  .attr('r',r).attr('fill',d=>srcColor(d.group)).attr('stroke','#0a0a0f').attr('stroke-width',1.5).attr('cursor','pointer')
  .call(d3.drag()
    .on('start',(e,d)=>{{if(!e.active)sim.alphaTarget(0.1).restart();d.fx=d.x;d.fy=d.y;}})
    .on('drag',(e,d)=>{{d.fx=e.x;d.fy=e.y;}})
    .on('end',(e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}}))
  .on('mouseover',(e,d)=>{{tip.innerHTML='<div class="ts">'+d.source+' p.'+(d.page||'?')+(d.section?' · '+d.section:'')+'</div><div class="tt">'+(d.preview||'')+'</div>';tip.style.display='block';tip.style.left=(e.clientX+15)+'px';tip.style.top=(e.clientY-8)+'px';}})
  .on('mousemove',e=>{{tip.style.left=(e.clientX+15)+'px';tip.style.top=(e.clientY-8)+'px';}})
  .on('mouseout',()=>tip.style.display='none');

const sim=d3.forceSimulation(nodes)
  .force('link',d3.forceLink(links).id(d=>d.id).distance(100).strength(0.35))
  .force('charge',d3.forceManyBody().strength(-220))
  .force('center',d3.forceCenter(W/2,H/2))
  .force('collide',d3.forceCollide(d=>r(d)+5))
  .alpha(0.6).alphaDecay(0.02)
  .on('tick',()=>{{
    lSel.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
    nSel.attr('cx',d=>d.x).attr('cy',d=>d.y);
  }});

const zoom=d3.zoom().scaleExtent([0.03,10]).on('zoom',e=>g.attr('transform',e.transform));
svg.call(zoom);

function applyVis(){{
  const minC=parseFloat(slider.value),sf=document.getElementById('sf').value.toLowerCase();
  const vis=new Set();
  nSel.each(d=>{{if(!sf||d.source.toLowerCase().includes(sf))vis.add(d.id);}});
  nSel.style('opacity',d=>vis.has(d.id)?null:0.04);
  lSel.style('display',d=>{{const s=nid(d.source),t=nid(d.target);return(activeTypes.has(d.type)&&d.confidence>=minC&&vis.has(s)&&vis.has(t))?null:'none';}});
}}
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"[prism-viz] HTML written → {output_path}")
    print(f"[prism-viz] {n_nodes:,} nodes, {n_links:,} links")
    print(f"[prism-viz] Open in browser or view via htmlpreview.github.io")


def viz_main() -> None:
    """Entry point for `prism-viz`."""
    parser = argparse.ArgumentParser(
        prog="prism-viz",
        description="Export a PRISM epistemic graph for visualisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("graph_path",
                        help="Path to prism_graph.json.gz")
    parser.add_argument("--format", choices=["gexf", "d3", "html"], default="d3",
                        help="Output format: 'gexf' (Gephi), 'd3' (D3.js JSON), or 'html' (self-contained interactive HTML)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file path. Defaults to graph.<format>. Use '-' for stdout (d3 only).")
    parser.add_argument("--edge-types", default=None,
                        help="Comma-separated edge types to include, e.g. supports,refutes,supersedes")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Minimum edge confidence to include (0.0–1.0)")
    parser.add_argument("--source-filter", default=None,
                        help="Only include nodes whose source contains this substring")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="Cap node count by keeping top-N highest-degree nodes")

    args = parser.parse_args()

    from .graph import EpistemicGraph

    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        print(f"[prism-viz] ERROR: graph not found: {graph_path}", file=sys.stderr)
        sys.exit(1)

    g = EpistemicGraph.load(graph_path)

    edge_types = None
    if args.edge_types:
        edge_types = {e.strip() for e in args.edge_types.split(",")}

    sub = _build_subgraph(
        g,
        edge_types     = edge_types,
        min_confidence = args.min_confidence,
        source_filter  = args.source_filter,
    )

    if args.max_nodes and sub.number_of_nodes() > args.max_nodes:
        print(f"[prism-viz] sampling top {args.max_nodes} nodes by degree ...")
        sub = _sample_by_degree(sub, args.max_nodes)

    print(f"[prism-viz] exporting {sub.number_of_nodes():,} nodes, {sub.number_of_edges():,} edges ...")

    # Determine output path
    if args.output == "-":
        output_path = Path("-")
    elif args.output:
        output_path = Path(args.output)
    else:
        stem = graph_path.name.replace(".json.gz", "").replace(".json", "")
        if args.format == "gexf":
            ext = "gexf"
        elif args.format == "html":
            ext = "html"
        else:
            ext = "json"
        output_path = graph_path.parent / f"{stem}.{ext}"

    if args.format == "gexf":
        if str(output_path) == "-":
            print("[prism-viz] ERROR: GEXF cannot be written to stdout", file=sys.stderr)
            sys.exit(1)
        _export_gexf(sub, output_path)
    elif args.format == "html":
        if str(output_path) == "-":
            print("[prism-viz] ERROR: HTML cannot be written to stdout", file=sys.stderr)
            sys.exit(1)
        _export_html(sub, output_path)
    else:
        _export_d3(sub, output_path)


if __name__ == "__main__":
    viz_main()
