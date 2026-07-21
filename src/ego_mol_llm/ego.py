"""Ego-neighborhood extraction for blind structure prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ego_mol_llm.graphml import Edge, MolecularNetwork, Node
from ego_mol_llm.validate import (
    canonicalize_smiles,
    check_mass,
    infer_multimer_adduct,
    is_multimer_adduct,
    monomer_mass_targets,
)


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

    def half_mass_delta(self, seed_mz: float | None) -> float | None:
        """
        Distance from neighbor m/z to an implied multimer-related mass target
        (e.g. monomer [M+H]+ when seed is [2M+H]+).

        Only defined for large precursors (dimers). Does not use seed m/z itself
        as a target (that bug made every near-isobar look like "half-mass").
        """
        if seed_mz is None or self.node.mz is None:
            return None
        if float(seed_mz) < 250.0:
            return None
        targets = monomer_mass_targets(float(seed_mz))
        if not targets:
            return None
        best = None
        for _label, target in targets:
            d = abs(float(self.node.mz) - target)
            if best is None or d < best:
                best = d
        return best

    def evidence_score(self, seed_mz: float | None) -> float:
        """
        Rank score: high cosine + near seed m/z OR near half/third mass (multimer)
        + annotation/SMILES.
        """
        cos = self.cosine
        dmz = self.resolved_delta_mz(seed_mz)
        hdmz = self.half_mass_delta(seed_mz)

        def _mass_term(d: float | None) -> tuple[float, bool]:
            if d is None:
                return 0.15, False
            if d <= 0.05:
                return 1.0, True
            if d <= 0.5:
                return 0.85, True
            if d <= 2.0:
                return 0.45, False
            if d <= 20.0:
                return 0.15, False
            return 0.05 * math.exp(-(d - 20.0) / 80.0), False

        m1, isobar = _mass_term(dmz)
        m2, half_iso = _mass_term(hdmz)
        # Prefer the better of same-m/z vs multimer-monomer alignment
        if m2 > m1:
            mass_term = m2
            multimer_hit = half_iso
            isobar = False
        else:
            mass_term = m1
            multimer_hit = False

        ann = 0.15 if self.node.is_annotated else 0.0
        smi = 0.25 if self.node.smiles else 0.0
        isobar_boost = 0.35 if isobar and self.node.is_annotated else 0.0
        # Strong boost: annotated SMILES at ~half mass (dimer case)
        half_boost = 0.40 if multimer_hit and self.node.is_annotated else 0.0
        if multimer_hit and self.node.smiles:
            half_boost += 0.15
        return 0.40 * cos + 0.35 * mass_term + ann + smi + isobar_boost + half_boost


@dataclass
class EgoContext:
    """Blind view of a query spectrum and its spectral neighborhood."""

    seed: Node
    seed_mz: float | None
    neighbors: list[NeighborEvidence]
    two_hop_named: list[Node] = field(default_factory=list)
    hide_seed_name: bool = True
    meta: dict[str, Any] = field(default_factory=dict)
    spectral: Any | None = None  # optional SpectralContext from mgf.py

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

    def half_mass_neighbors(self, dmz_max: float = 1.0) -> list[NeighborEvidence]:
        """Neighbors near multimer-implied monomer ion m/z."""
        out = []
        for ev in self.neighbors:
            d = ev.half_mass_delta(self.seed_mz)
            if d is not None and d <= dmz_max:
                out.append(ev)
        return sorted(
            out,
            key=lambda n: (
                n.half_mass_delta(self.seed_mz) or 99,
                -n.cosine,
            ),
        )

    def class_hints(self) -> dict[str, int]:
        keys = {
            "indole/trp": ("indole", "tryptophan", "trypt"),
            "carboline/thbc": ("carboline", "harmane", "harman", "strictosidine"),
            "bile/steroid": (
                "cholic",
                "cheno",
                "deoxy",
                "bile",
                "oxo",
                "keto",
                "cholan",
                "steroid",
                "cholest",
            ),
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
        half_dmz_max: float = 2.0,
        limit: int = 15,
        scan_all_with_smiles: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Library SMILES that fit the precursor mass as monomer *or multimer*.

        Includes:
        - near-isobar neighbors (|Δm/z| ≤ dmz_max)
        - half-mass neighbors (multimer monomers)
        - any annotated SMILES that passes check_mass (incl. [2M+H]+)
        """
        hyps: list[dict[str, Any]] = []
        seen: set[str] = set()

        # Candidate pool: high evidence neighbors + half-mass + optional full SMILES scan
        pool: list[NeighborEvidence] = []
        pool.extend(self.top_neighbors)
        pool.extend(self.half_mass_neighbors(half_dmz_max))
        if scan_all_with_smiles:
            for ev in self.neighbors:
                if ev.node.smiles:
                    pool.append(ev)

        # Dedupe by node id preserving order
        seen_ids: set[str] = set()
        ordered: list[NeighborEvidence] = []
        for ev in pool:
            if ev.node.id in seen_ids:
                continue
            seen_ids.add(ev.node.id)
            ordered.append(ev)

        # Two-hop annotated nodes (often hold library SMILES at half-mass)
        for n2 in self.two_hop_named:
            if not n2.smiles or n2.id in seen_ids:
                continue
            # Synthetic edge: cosine unknown → use 0.75 if half-mass close else 0.5
            hd = None
            if self.seed_mz is not None and n2.mz is not None:
                tgts = monomer_mass_targets(float(self.seed_mz))[:20]
                if tgts:
                    hd = min(abs(float(n2.mz) - t) for _, t in tgts)
            cos_syn = 0.85 if (hd is not None and hd <= 1.0) else 0.55
            dmz_syn = (
                abs(float(n2.mz) - float(self.seed_mz))
                if (n2.mz is not None and self.seed_mz is not None)
                else 999.0
            )
            fake_edge = Edge(
                source=self.seed.id,
                target=n2.id,
                cosine=cos_syn,
                abs_diff_mz=dmz_syn,
            )
            ordered.append(NeighborEvidence(node=n2, edge=fake_edge, hop=2))
            seen_ids.add(n2.id)

        ranked = sorted(
            ordered,
            key=lambda n: (-n.evidence_score(self.seed_mz), -(n.cosine)),
        )

        for ev in ranked:
            d = ev.resolved_delta_mz(self.seed_mz)
            hd = ev.half_mass_delta(self.seed_mz)
            near = d is not None and d <= dmz_max
            half_near = hd is not None and hd <= half_dmz_max
            raw_smi = ev.node.smiles
            if not raw_smi:
                continue
            # Skip far nodes unless they have SMILES we can mass-check
            if not near and not half_near and not scan_all_with_smiles:
                continue

            can = canonicalize_smiles(raw_smi)
            if not can or can in seen:
                continue
            # Multimer only meaningful for large precursors
            allow_multi = self.seed_mz is not None and float(self.seed_mz) >= 250.0
            ok, em, err, adduct = check_mass(
                can,
                self.seed_mz,
                None,
                tol_da=mass_tol_da,
                include_multimer=allow_multi,
                include_odd_electron=False,
            )
            # Infer [2M+H]+ etc. from neighbor ion m/z when RDKit mass unavailable
            if (
                allow_multi
                and (ok is not True or not adduct)
                and half_near
                and ev.node.mz is not None
            ):
                inf_add, inf_err = infer_multimer_adduct(
                    self.seed_mz, float(ev.node.mz), tol_da=max(mass_tol_da, 0.05)
                )
                if inf_add and (inf_err is not None and inf_err <= 0.5):
                    adduct = inf_add
                    err = inf_err if err is None else min(err, inf_err)
                    if inf_err <= mass_tol_da:
                        ok = True

            # Must pass mass or be a tight near-isobar with SMILES
            if ok is False:
                continue
            if ok is None and not (near and d is not None and d <= 0.15):
                continue
            # Reject weak spectral links with only loose mass
            if ev.cosine < 0.55 and not (near and d is not None and d <= 0.05):
                continue

            seen.add(can)
            conf = min(0.90, 0.40 + 0.40 * ev.cosine)
            if near and d is not None and d <= 0.05:
                conf = min(0.95, conf + 0.12)
            elif near and d is not None and d <= 0.15:
                conf = min(0.90, conf + 0.05)
            if half_near and allow_multi:
                conf = min(0.95, conf + 0.12)
            if ok is True and err is not None and err <= 0.01:
                conf = min(0.95, conf + 0.10)
            elif ok is True:
                conf = min(0.95, conf + 0.05)
            # Water-loss adduct is chemically common for phenols/alcohols
            if adduct and "H2O" in str(adduct):
                conf = min(0.95, conf + 0.04)
            # Prefer explicit 12-oxo / 12-keto naming (12-keto-CDCA family)
            nm = (ev.node.name or "").lower()
            if "12-oxo" in nm or "12-keto" in nm or "dihydroxy-12-oxo" in nm:
                conf = min(0.97, conf + 0.08)
            # Penalize radical-cation adducts and large dmz non-multimer
            if adduct in {"[M]+", "[M]-"}:
                conf = min(conf, 0.35)
            if d is not None and d > 5 and not is_multimer_adduct(adduct) and not (
                adduct and "H2O" in str(adduct)
            ):
                conf *= 0.5

            if is_multimer_adduct(adduct):
                note = f"mass-consistent multimer adduct {adduct}"
            elif adduct and "H2O" in str(adduct):
                note = f"mass-consistent water-loss adduct {adduct}"
            elif half_near and allow_multi:
                note = "half-mass / multimer-monomer annotated neighbor"
            elif near:
                note = "near-isobar annotated neighbor"
            else:
                note = "mass-consistent annotated neighbor"

            # Quality flag for rescue eligibility
            rescue_ok = False
            if ok is True and err is not None and err <= mass_tol_da:
                if is_multimer_adduct(adduct) and ev.cosine >= 0.65:
                    rescue_ok = True
                elif near and d is not None and d <= 0.15 and ev.cosine >= 0.70:
                    rescue_ok = True
                elif adduct and "H2O" in str(adduct) and ev.cosine >= 0.70 and err <= 0.02:
                    rescue_ok = True
                elif ev.cosine >= 0.85 and err <= 0.01:
                    rescue_ok = True
            # Without RDKit mass: only tight near-isobar + strong cosine (library self-match)
            elif ok is None and near and d is not None and d <= 0.05 and ev.cosine >= 0.80:
                rescue_ok = True
                conf = min(conf, 0.75)

            hyps.append(
                {
                    "smiles": can,
                    "name": ev.node.name,
                    "cosine": ev.cosine,
                    "delta_mz": d,
                    "half_mass_delta": hd,
                    "exact_mass": em,
                    "mass_error_da": err,
                    "adduct": adduct,
                    "mass_ok": ok if ok is not None else False,
                    "confidence": conf,
                    "evidence_score": ev.evidence_score(self.seed_mz),
                    "note": note,
                    "rescue_ok": rescue_ok,
                }
            )
            if len(hyps) >= limit:
                break

        def _rank(h: dict[str, Any]) -> tuple:
            nm = (h.get("name") or "").lower()
            oxo12 = 0 if ("12-oxo" in nm or "12-keto" in nm) else 1
            water = 0 if (h.get("adduct") and "H2O" in str(h.get("adduct"))) else 1
            odd = 0 if h.get("adduct") not in {"[M]+", "[M]-"} else 1
            return (
                0 if h.get("rescue_ok") else 1,
                0 if h.get("mass_ok") is True else 1,
                odd,
                0 if is_multimer_adduct(h.get("adduct")) else 1,
                water,
                oxo12,
                -(h.get("cosine") or 0),
                h.get("mass_error_da") if h.get("mass_error_da") is not None else 99,
                -(h.get("confidence") or 0),
            )

        hyps.sort(key=_rank)
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

    # Wider pool: cosine top + will re-rank with half-mass awareness
    raw_pool = network.neighbors(seed.id)
    pool_cap = max(max_neighbors * 4, max_neighbors)
    neigh_ev = [NeighborEvidence(node=node, edge=edge, hop=1) for node, edge in raw_pool[:pool_cap]]
    neigh_ev.sort(key=lambda n: (-n.evidence_score(seed.mz), -(n.cosine)))
    neigh_ev = neigh_ev[:max_neighbors]

    two_hop: list[Node] = []
    if include_two_hop:
        seen = {seed.id} | {e.node.id for e in neigh_ev}
        scored: list[tuple[float, Node]] = []
        # Walk 1-hop and 2-hop; also scan a wider ring of annotated SMILES near half-mass
        frontier = [seed.id] + [e.node.id for e in neigh_ev]
        for fid in frontier:
            for n2, e2 in network.neighbors(fid):
                if n2.id in seen:
                    continue
                if not n2.is_annotated and not n2.smiles:
                    continue
                seen.add(n2.id)
                dmz = (
                    abs(float(n2.mz) - float(seed.mz))
                    if (n2.mz is not None and seed.mz is not None)
                    else 50.0
                )
                half = None
                if seed.mz is not None and n2.mz is not None:
                    tgts = monomer_mass_targets(float(seed.mz))[:20]
                    if tgts:
                        half = min(abs(float(n2.mz) - t) for _, t in tgts)
                score = float(e2.cosine or 0.0) + (0.3 if dmz <= 50 else 0.0)
                if half is not None and half <= 1.0:
                    score += 1.2  # prioritize multimer monomers
                elif half is not None and half <= 2.0:
                    score += 0.7
                if n2.smiles:
                    score += 0.4
                # keyword boost for bile/oxo scaffolds common in dimer failures
                nm = (n2.name or "").lower()
                if any(k in nm for k in ("oxo", "keto", "cholan", "cholic", "bile")):
                    score += 0.25
                scored.append((score, n2))

        # Global half-mass SMILES sweep (capped) — large precursors only
        if seed.mz is not None and float(seed.mz) >= 250.0:
            tgts = monomer_mass_targets(float(seed.mz))[:20]
            if tgts:
                for nid, n2 in network.nodes.items():
                    if nid in seen or not n2.smiles:
                        continue
                    if n2.mz is None:
                        continue
                    half = min(abs(float(n2.mz) - t) for _, t in tgts)
                    if half <= 1.5:
                        seen.add(nid)
                        score = 1.5 - half + (0.3 if n2.is_annotated else 0.0)
                        scored.append((score, n2))

        scored.sort(key=lambda x: -x[0])
        two_hop = [n for _, n in scored[: max(max_two_hop_named, 40)]]

    blind_seed = Node(
        id=seed.id,
        mz=seed.mz,
        name=None if hide_seed_name else seed.name,
        smiles=None if hide_seed_name else seed.smiles,
        community_id=seed.community_id,
        direct_neighbor=seed.direct_neighbor,
        attrs={
            k: v
            for k, v in seed.attrs.items()
            if k not in {"name", "SMILES"} or not hide_seed_name
        },
    )

    isobars = [
        ev
        for ev in neigh_ev
        if (ev.resolved_delta_mz(seed.mz) is not None and ev.resolved_delta_mz(seed.mz) <= 0.5)
    ]
    half_n = [
        ev
        for ev in neigh_ev
        if (ev.half_mass_delta(seed.mz) is not None and ev.half_mass_delta(seed.mz) <= 1.0)
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
            "n_half_mass_neighbors": len(half_n),
        },
    )
