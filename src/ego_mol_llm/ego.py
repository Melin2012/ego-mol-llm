"""Ego-neighborhood extraction for blind structure prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ego_mol_llm.graphml import Edge, MolecularNetwork, Node
from ego_mol_llm.validate import check_mass, canonicalize_smiles


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
        if self.edge.abs_diff_mz is not None:
            return float(self.edge.abs_diff_mz)
        return None

    def resolved_delta_mz(self, seed_mz: float | None) -> float | None:
        if self.edge.abs_diff_mz is not None:
            return float(self.edge.abs_diff_mz)
        if seed_mz is not None and self.node.mz is not None:
            return abs(float(self.node.mz) - float(seed_mz))
        return None

    def evidence_score(self, seed_mz: float | None) -> float:
        """
        Score used for ranking (manual MTCA-style reasoning):
        high cosine + near-zero Δm/z + annotation/SMILES beat distant high-cosine noise.
        """
        cos = self.cosine
        dmz = self.resolved_delta_mz(seed_mz)
        if dmz is None:
            mass_term = 0.15
            isobar = False
        elif dmz <= 0.05:
            mass_term = 1.0
            isobar = True
        elif dmz <= 0.5:
            mass_term = 0.85
            isobar = True
        elif dmz <= 2.0:
            mass_term = 0.45
            isobar = False
        elif dmz <= 20.0:
            mass_term = 0.15
            isobar = False
        else:
            # Large Δm/z edges are often spectral pollution
            mass_term = 0.05 * math.exp(-(dmz - 20.0) / 80.0)
            isobar = False

        ann = 0.15 if self.node.is_annotated else 0.0
        smi = 0.25 if self.node.smiles else 0.0
        # Boost near-isobar library hits strongly (self-match style)
        isobar_boost = 0.35 if isobar and self.node.is_annotated else 0.0
        return 0.45 * cos + 0.40 * mass_term + ann + smi + isobar_boost


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
        """Rank by evidence score (mass-aware), not raw cosine alone."""
        return sorted(
            self.neighbors,
            key=lambda n: (-n.evidence_score(self.seed_mz), -(n.cosine)),
        )

    def near_isobars(self, dmz_max: float = 0.5) -> list[NeighborEvidence]:
        out = []
        for ev in self.neighbors:
            d = ev.resolved_delta_mz(self.seed_mz)
            if d is not None and d <= dmz_max:
                out.append(ev)
        return sorted(out, key=lambda n: (-n.cosine, n.resolved_delta_mz(self.seed_mz) or 0))

    def class_hints(self) -> dict[str, int]:
        keys = {
            "indole/trp": ("indole", "tryptophan", "trypt"),
            "carboline/thbc": ("carboline", "harmane", "harman", "strictosidine"),
            "peptide/aa": ("leu-", "ile-", "gly-", "pro", "peptide", "amino"),
            "hydantoin/creatinine": ("hydantoin", "creatinine", "uracil", "xanthine"),
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

    def neighbor_structure_hypotheses(
        self,
        mass_tol_da: float = 0.05,
        dmz_max: float = 2.0,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Library SMILES among near-mass neighbors that fit the precursor mass.
        Mirrors human ego-network annotation: trust same-m/z annotated structures first.
        """
        hyps: list[dict[str, Any]] = []
        seen: set[str] = set()
        ranked = sorted(
            self.neighbors,
            key=lambda n: (-n.evidence_score(self.seed_mz), -(n.cosine)),
        )
        for ev in ranked:
            d = ev.resolved_delta_mz(self.seed_mz)
            if d is not None and d > dmz_max:
                continue
            raw_smi = ev.node.smiles
            if not raw_smi:
                continue
            can = canonicalize_smiles(raw_smi)
            if not can or can in seen:
                continue
            ok, em, err, adduct = check_mass(can, self.seed_mz, None, tol_da=mass_tol_da)
            # Also accept near-isobar library hit even if RDKit missing (ok is None)
            if ok is False:
                continue
            seen.add(can)
            conf = min(0.95, 0.55 + 0.35 * ev.cosine + (0.15 if (d is not None and d <= 0.05) else 0.0))
            if ok is True:
                conf = min(0.95, conf + 0.1)
            hyps.append(
                {
                    "smiles": can,
                    "name": ev.node.name,
                    "cosine": ev.cosine,
                    "delta_mz": d,
                    "exact_mass": em,
                    "mass_error_da": err,
                    "adduct": adduct,
                    "mass_ok": ok,
                    "confidence": conf,
                    "evidence_score": ev.evidence_score(self.seed_mz),
                    "note": "near-mass annotated neighbor",
                }
            )
            if len(hyps) >= limit:
                break
        return hyps


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

    # Take a wider pool by cosine, then re-rank by evidence score and keep top-N
    raw_pool = network.neighbors(seed.id)
    pool_cap = max(max_neighbors * 3, max_neighbors)
    neigh_ev = [NeighborEvidence(node=node, edge=edge, hop=1) for node, edge in raw_pool[:pool_cap]]
    neigh_ev.sort(key=lambda n: (-n.evidence_score(seed.mz), -(n.cosine)))
    neigh_ev = neigh_ev[:max_neighbors]

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
                # Prefer 2-hop nodes near seed mass when seed mass known
                dmz = abs(float(n2.mz) - float(seed.mz)) if (n2.mz is not None and seed.mz is not None) else 50.0
                score = float(e2.cosine or 0.0) + (0.5 if dmz <= 50 else 0.0)
                scored.append((score, n2))
        scored.sort(key=lambda x: -x[0])
        two_hop = [n for _, n in scored[:max_two_hop_named]]

    blind_seed = Node(
        id=seed.id,
        mz=seed.mz,
        name=None if hide_seed_name else seed.name,
        smiles=None if hide_seed_name else seed.smiles,
        community_id=seed.community_id,
        direct_neighbor=seed.direct_neighbor,
        attrs={k: v for k, v in seed.attrs.items() if k not in {"name", "SMILES"} or not hide_seed_name},
    )

    isobars = [
        ev
        for ev in neigh_ev
        if (ev.resolved_delta_mz(seed.mz) is not None and ev.resolved_delta_mz(seed.mz) <= 0.5)
    ]

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
            "n_near_isobars": len(isobars),
        },
    )
