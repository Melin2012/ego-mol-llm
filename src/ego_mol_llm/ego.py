"""Ego-neighborhood extraction for blind structure prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ego_mol_llm.graphml import Edge, MolecularNetwork, Node


@dataclass
class NeighborEvidence:
    node: Node
    edge: Edge
    hop: int = 1

    @property
    def cosine(self) -> float:
        return float(self.edge.cosine or 0.0)

    @property
    def delta_mz(self) -> float | None:
        if self.node.mz is None or self.edge.abs_diff_mz is not None:
            return self.edge.abs_diff_mz
        return None


@dataclass
class EgoContext:
    """Blind view of a query spectrum and its spectral neighborhood."""

    seed: Node
    seed_mz: float | None
    neighbors: list[NeighborEvidence]
    two_hop_named: list[Node] = field(default_factory=list)
    hide_seed_name: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def top_neighbors(self) -> list[NeighborEvidence]:
        return sorted(self.neighbors, key=lambda n: -n.cosine)

    def class_hints(self) -> dict[str, int]:
        """Lightweight keyword class counts from neighbor names (for logging only)."""
        keys = {
            "indole/trp": ("indole", "tryptophan", "trypt"),
            "carboline/thbc": ("carboline", "harmane", "harman", "strictosidine"),
            "peptide/aa": ("leu-", "ile-", "gly-", "pro", "peptide", "amino"),
            "unknown": ("no_match",),
        }
        counts = {k: 0 for k in keys}
        for ev in self.neighbors:
            name = (ev.node.name or "").lower()
            if not name or name == "no_match":
                counts["unknown"] += 1
                continue
            matched = False
            for lab, toks in keys.items():
                if lab == "unknown":
                    continue
                if any(t in name for t in toks):
                    counts[lab] += 1
                    matched = True
                    break
            if not matched:
                counts.setdefault("other", 0)
                counts["other"] += 1
        return counts


def build_ego(
    network: MolecularNetwork,
    seed: Node | None = None,
    seed_id: str | None = None,
    seed_name_contains: str | None = None,
    hide_seed_name: bool = True,
    max_neighbors: int = 25,
    include_two_hop: bool = True,
    max_two_hop_named: int = 20,
) -> EgoContext:
    """Build a blind ego context suitable for LLM prompting."""
    if seed is None:
        seed = network.find_seed(seed_id=seed_id, seed_name_contains=seed_name_contains)

    neigh_ev = []
    for node, edge in network.neighbors(seed.id)[:max_neighbors]:
        # Optionally redact if neighbor is clearly the same identity string as seed
        neigh_ev.append(NeighborEvidence(node=node, edge=edge, hop=1))

    two_hop: list[Node] = []
    if include_two_hop:
        seen = {seed.id} | {e.node.id for e in neigh_ev}
        scored: list[tuple[float, Node]] = []
        for ev in neigh_ev:
            for n2, e2 in network.neighbors(ev.node.id):
                if n2.id in seen:
                    continue
                if not n2.is_annotated:
                    continue
                seen.add(n2.id)
                scored.append((float(e2.cosine or 0.0), n2))
        scored.sort(key=lambda x: -x[0])
        two_hop = [n for _, n in scored[:max_two_hop_named]]

    # Blind seed copy for prompting
    blind_seed = Node(
        id=seed.id,
        mz=seed.mz,
        name=None if hide_seed_name else seed.name,
        smiles=None if hide_seed_name else seed.smiles,
        community_id=seed.community_id,
        direct_neighbor=seed.direct_neighbor,
        attrs={k: v for k, v in seed.attrs.items() if k not in {"name", "SMILES"} or not hide_seed_name},
    )

    return EgoContext(
        seed=blind_seed,
        seed_mz=seed.mz,
        neighbors=neigh_ev,
        two_hop_named=two_hop,
        hide_seed_name=hide_seed_name,
        meta={
            "true_seed_id": seed.id,
            "true_seed_name": seed.name,
            "n_nodes": len(network),
            "n_edges": len(network.edges),
            "degree": len(network.adjacency.get(seed.id, [])),
        },
    )
