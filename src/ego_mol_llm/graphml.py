"""GraphML molecular-network parser (GNPS / HNSW-style)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}


@dataclass
class Node:
    id: str
    mz: float | None = None
    name: str | None = None
    smiles: str | None = None
    community_id: str | None = None
    direct_neighbor: bool = False
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def is_annotated(self) -> bool:
        n = (self.name or "").strip()
        return bool(n) and n.upper() != "NO_MATCH"


@dataclass
class Edge:
    source: str
    target: str
    cosine: float | None = None
    abs_diff_mz: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class MolecularNetwork:
    """Undirected molecular network loaded from GraphML."""

    nodes: dict[str, Node]
    edges: list[Edge]
    adjacency: dict[str, list[tuple[str, Edge]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.adjacency:
            adj: dict[str, list[tuple[str, Edge]]] = defaultdict(list)
            for e in self.edges:
                adj[e.source].append((e.target, e))
                adj[e.target].append((e.source, e))
            self.adjacency = dict(adj)

    def __len__(self) -> int:
        return len(self.nodes)

    def neighbors(self, node_id: str) -> list[tuple[Node, Edge]]:
        out: list[tuple[Node, Edge]] = []
        for nid, edge in self.adjacency.get(node_id, []):
            if nid in self.nodes:
                out.append((self.nodes[nid], edge))
        out.sort(key=lambda x: -(x[1].cosine or 0.0))
        return out

    def find_seed(
        self,
        seed_id: str | None = None,
        seed_name_contains: str | None = None,
        prefer_direct_neighbor_hub: bool = True,
    ) -> Node:
        """Resolve the query / center node."""
        if seed_id is not None:
            if seed_id not in self.nodes:
                raise KeyError(f"seed_id not found: {seed_id}")
            return self.nodes[seed_id]

        if seed_name_contains:
            hits = [
                n
                for n in self.nodes.values()
                if seed_name_contains.lower() in (n.name or "").lower()
            ]
            if not hits:
                raise KeyError(f"No node name contains: {seed_name_contains}")
            # Prefer highest degree among matches
            hits.sort(key=lambda n: len(self.adjacency.get(n.id, [])), reverse=True)
            return hits[0]

        # Heuristic: node id "0" is often the query spectrum in HNSW exports
        if "0" in self.nodes:
            return self.nodes["0"]

        if prefer_direct_neighbor_hub:
            # Pick node with most direct_neighbor=True adjacent nodes
            best = None
            best_score = -1
            for nid, node in self.nodes.items():
                score = sum(
                    1
                    for m, _ in self.neighbors(nid)
                    if m.direct_neighbor or node.direct_neighbor
                )
                deg = len(self.adjacency.get(nid, []))
                score = score * 1000 + deg
                if score > best_score:
                    best_score = score
                    best = node
            if best is not None:
                return best

        # Fallback: highest degree
        return max(self.nodes.values(), key=lambda n: len(self.adjacency.get(n.id, [])))


def _parse_float(text: str | None) -> float | None:
    if text is None:
        return None
    t = text.strip().lower()
    if t in {"", "nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_graphml(path: str | Path) -> MolecularNetwork:
    """Load a GNPS/HNSW-style GraphML molecular network."""
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    # Support default namespace and bare tags
    def findall(elem: ET.Element, tag: str) -> list[ET.Element]:
        hits = elem.findall(f"g:{tag}", GRAPHML_NS)
        if hits:
            return hits
        return elem.findall(tag)

    keys: dict[str, str] = {}
    for k in findall(root, "key"):
        kid = k.get("id")
        name = k.get("attr.name")
        if kid and name:
            keys[kid] = name

    graph = root.find("g:graph", GRAPHML_NS)
    if graph is None:
        graph = root.find("graph")
    if graph is None:
        raise ValueError("No <graph> element found in GraphML")

    nodes: dict[str, Node] = {}
    for n_el in findall(graph, "node"):
        nid = n_el.get("id")
        if not nid:
            continue
        raw: dict[str, Any] = {}
        for d in findall(n_el, "data"):
            key = d.get("key")
            if key and key in keys:
                raw[keys[key]] = d.text

        # Handle duplicated SMILES key names (string vs double) — prefer string
        smiles = raw.get("SMILES")
        if smiles is not None and str(smiles).lower() in {"nan", "none"}:
            smiles = None

        dn = str(raw.get("direct_neighbor", "0")).strip() in {"1", "true", "True", "yes"}
        nodes[nid] = Node(
            id=nid,
            mz=_parse_float(str(raw.get("PEPMASS")) if raw.get("PEPMASS") is not None else None),
            name=raw.get("name"),
            smiles=str(smiles) if smiles else None,
            community_id=str(raw["community_id"]) if raw.get("community_id") is not None else None,
            direct_neighbor=dn,
            attrs=raw,
        )

    edges: list[Edge] = []
    for e_el in findall(graph, "edge"):
        s, t = e_el.get("source"), e_el.get("target")
        if not s or not t:
            continue
        raw: dict[str, Any] = {}
        for d in findall(e_el, "data"):
            key = d.get("key")
            if key and key in keys:
                raw[keys[key]] = d.text
        edges.append(
            Edge(
                source=s,
                target=t,
                cosine=_parse_float(raw.get("cosine_score")),
                abs_diff_mz=_parse_float(raw.get("abs_diff_PEPMASS")),
                attrs=raw,
            )
        )

    return MolecularNetwork(nodes=nodes, edges=edges)


def iter_annotated(nodes: Iterable[Node]) -> list[Node]:
    return [n for n in nodes if n.is_annotated]
