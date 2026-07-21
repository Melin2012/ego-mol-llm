"""Full analysis of PCA_anonimized bile-acid HNSW subgraph."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from ego_mol_llm.ego import build_ego
from ego_mol_llm.graphml import load_graphml
from ego_mol_llm.validate import (
    check_mass,
    formula_to_mass,
    theoretical_ion_mz,
    canonicalize_smiles,
    exact_mass_from_smiles,
)

ROOT = Path(r"C:\Users\AlexeyMelnik\Downloads\PCA_anonimized")
GRAPHML = ROOT / "Bile_Acid_HNSW_test_1.graphml"
EGO_MGF = ROOT / "Ego_MSMS.mgf"
ALL_MGF = ROOT / "subgraph_molecules_with_node_ids.mgf"


def parse_mgf(path: Path) -> dict[str, dict]:
    """Parse MGF into dict keyed by SCANS / FEATURE_ID / TITLE id."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"(?i)BEGIN IONS", text)
    out = {}
    for b in blocks[1:]:
        body = b.split("END IONS")[0]
        meta = {}
        peaks = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            if "=" in line and not line[0].isdigit():
                k, v = line.split("=", 1)
                meta[k.strip().upper()] = v.strip()
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
        # id keys
        ids = []
        for key in ("SCANS", "FEATURE_ID", "SPECTRUMID", "TITLE", "ID"):
            if key in meta:
                ids.append(meta[key])
                # also bare number
                m = re.search(r"(\d+)", meta[key])
                if m:
                    ids.append(m.group(1))
        rec = {"meta": meta, "peaks": peaks}
        for i in ids:
            out[str(i)] = rec
        # PEPMASS
        if "PEPMASS" in meta:
            try:
                rec["pepmass"] = float(meta["PEPMASS"].split()[0])
            except ValueError:
                pass
    return out


def cosine_peaks(p1, p2, tol=0.02):
    """Simple peak cosine with binning."""
    if not p1 or not p2:
        return 0.0

    def dens(peaks):
        d = defaultdict(float)
        for mz, inten in peaks:
            if inten <= 0:
                continue
            d[round(mz / tol) * tol] += inten
        # L2 norm
        norm = math.sqrt(sum(v * v for v in d.values())) or 1.0
        return {k: v / norm for k, v in d.items()}

    a, b = dens(p1), dens(p2)
    keys = set(a) | set(b)
    # match within tol by expanding
    score = 0.0
    used_b = set()
    for ka, va in a.items():
        best = None
        best_d = 1e9
        for kb, vb in b.items():
            if kb in used_b:
                continue
            dd = abs(ka - kb)
            if dd <= tol and dd < best_d:
                best_d = dd
                best = kb
        if best is not None:
            score += va * b[best]
            used_b.add(best)
    return max(0.0, min(1.0, score))


def top_peaks(peaks, n=10):
    return sorted(peaks, key=lambda x: -x[1])[:n]


def main():
    net = load_graphml(GRAPHML)
    print("nodes", len(net.nodes), "edges", len(net.edges))
    seed = net.find_seed(seed_id="0")
    print("SEED id=0")
    print("  name:", seed.name)
    print("  mz:", seed.mz)
    print("  smiles:", seed.smiles)
    print("  community:", seed.community_id)
    print("  degree:", len(net.adjacency.get("0", [])))
    print("  attrs keys:", list(seed.attrs.keys())[:20])

    # All direct neighbors with edge quality
    neigh = net.neighbors("0")
    print("\n=== DIRECT NEIGHBORS (all", len(neigh), ") ===")
    tiers = {"high": [], "mid": [], "low": []}
    for node, edge in neigh:
        cos = edge.cosine or 0
        dmz = edge.abs_diff_mz
        if dmz is None and node.mz is not None and seed.mz is not None:
            dmz = abs(node.mz - seed.mz)
        bucket = "high" if cos >= 0.7 else ("mid" if cos >= 0.5 else "low")
        tiers[bucket].append((cos, dmz, node, edge))
        print(
            f"  cos={cos:.3f} [{bucket}] dmz={dmz} mz={node.mz} "
            f"name={(node.name or '')[:100]}"
        )
        if node.smiles:
            print(f"    SMILES={node.smiles[:90]}")

    print("\nTier counts:", {k: len(v) for k, v in tiers.items()})

    # Weighted annotation vote from high/mid edges only
    print("\n=== ANNOTATION PROPAGATION (weighted) ===")
    votes = []
    for cos, dmz, node, edge in tiers["high"] + tiers["mid"]:
        w = cos if cos >= 0.7 else cos * 0.4  # downweight mid
        if cos < 0.5:
            continue
        name = node.name or "NO_MATCH"
        smi = node.smiles
        votes.append((w, cos, dmz, name, smi, node.mz))
        # mass check if SMILES
        if smi and seed.mz:
            ok, em, err, adduct = check_mass(smi, seed.mz, tol_da=0.05, include_multimer=True)
            print(
                f"  w={w:.3f} cos={cos:.3f} dmz={dmz} mz={node.mz} mass_ok={ok} "
                f"err={err} adduct={adduct}"
            )
            print(f"    {name[:110]}")
            if smi:
                print(f"    {smi[:100]}")
        else:
            print(f"  w={w:.3f} cos={cos:.3f} dmz={dmz} mz={node.mz} {name[:110]}")

    # Formula matches for seed mz
    if seed.mz:
        print("\n=== FORMULA / ADDUCT FOR SEED m/z", seed.mz, "===")
        for formula, note in [
            ("C24H40O5", "cholic acid"),
            ("C24H38O5", "oxo-dihydroxy BA (e.g. 12-keto-CDCA / 7-keto-DCA)"),
            ("C24H40O4", "CDCA / DCA"),
            ("C24H38O4", "keto monohydroxy?"),
            ("C24H36O5", "diketo hydroxy"),
            ("C24H40O3", "LCA"),
            ("C26H43NO6", "glycocholic"),
            ("C26H43NO5", "glyco-CDCA/DCA"),
            ("C26H45NO7S", "taurocholic"),
            ("C26H45NO6S", "tauro-CDCA"),
        ]:
            em = formula_to_mass(formula)
            if not em:
                continue
            for name, theo in theoretical_ion_mz(em, True):
                err = abs(seed.mz - theo)
                if err < 0.05:
                    print(f"  MATCH {formula} ({note}) {name} theo={theo:.4f} err={err:.4f}")

    # 2-hop and 3-hop
    print("\n=== 2-HOP / 3-HOP EXPANSION ===")
    hop1 = {n.id for n, _ in neigh}
    hop2 = set()
    hop3 = set()
    for nid in hop1:
        for n2, e2 in net.neighbors(nid):
            if n2.id != "0" and n2.id not in hop1:
                hop2.add(n2.id)
    for nid in hop2:
        for n3, e3 in net.neighbors(nid):
            if n3.id != "0" and n3.id not in hop1 and n3.id not in hop2:
                hop3.add(n3.id)
    print(f"hop1={len(hop1)} hop2={len(hop2)} hop3={len(hop3)}")

    def summarize_hop(ids, label):
        names = []
        formulas = Counter()
        for i in ids:
            n = net.nodes[i]
            if n.is_annotated:
                names.append((n.mz, n.name[:80] if n.name else "", n.smiles))
                low = (n.name or "").lower()
                for kw in (
                    "cholic",
                    "cheno",
                    "deoxy",
                    "litho",
                    "urso",
                    "keto",
                    "oxo",
                    "tauro",
                    "glyco",
                    "muricholic",
                    "hyodeoxy",
                ):
                    if kw in low:
                        formulas[kw] += 1
        print(f"\n{label}: annotated={len(names)}")
        print("  keyword counts:", dict(formulas.most_common(15)))
        for mz, name, smi in sorted(names, key=lambda x: x[0] or 0)[:25]:
            print(f"  mz={mz} {name}")

    summarize_hop(hop1, "HOP1")
    summarize_hop(hop2, "HOP2")
    summarize_hop(hop3, "HOP3")

    # High-cos path annotations only
    print("\n=== HIGH-COS ( >=0.7 ) NEIGHBORHOOD CHEMISTRY ===")
    for cos, dmz, node, edge in sorted(tiers["high"], reverse=True):
        print(f"  cos={cos:.3f} dmz={dmz} mz={node.mz} {(node.name or '')[:120]}")
        if node.smiles:
            ok, em, err, adduct = check_mass(
                node.smiles, seed.mz, tol_da=0.05, include_multimer=True
            )
            print(f"    mass_ok={ok} err={err} adduct={adduct} em={em}")

    # MGF
    print("\n=== MGF ===")
    ego_spec = parse_mgf(EGO_MGF)
    all_spec = parse_mgf(ALL_MGF)
    print("Ego_MSMS spectra keys sample:", list(ego_spec.keys())[:10], "n=", len(ego_spec))
    print("All MGF n=", len(all_spec), "sample keys:", list(all_spec.keys())[:15])

    # Find seed spectrum
    seed_spec = ego_spec.get("0") or all_spec.get("0")
    if not seed_spec:
        # try matching pepmass
        for k, v in list(ego_spec.items()) + list(all_spec.items()):
            pm = v.get("pepmass")
            if pm and seed.mz and abs(pm - seed.mz) < 0.02:
                seed_spec = v
                print("matched seed by pepmass key", k)
                break
    if seed_spec:
        print("SEED spectrum meta:", seed_spec["meta"])
        print("SEED top peaks:", top_peaks(seed_spec["peaks"], 15))
        print("n peaks:", len(seed_spec["peaks"]))
    else:
        print("NO SEED SPECTRUM FOUND")

    # Compare seed MS/MS to high-cos neighbors
    print("\n=== MS/MS cosine seed vs neighbors ===")
    if seed_spec:
        rows = []
        for cos, dmz, node, edge in tiers["high"] + tiers["mid"] + tiers["low"]:
            sp = all_spec.get(node.id)
            if not sp:
                continue
            c = cosine_peaks(seed_spec["peaks"], sp["peaks"], tol=0.02)
            rows.append((c, cos, dmz, node))
        rows.sort(reverse=True)
        for c, cos, dmz, node in rows[:20]:
            print(
                f"  msms_cos={c:.3f} edge_cos={cos:.3f} dmz={dmz} mz={node.mz} "
                f"{(node.name or '')[:90]}"
            )

    # Build ego via package
    ego = build_ego(net, seed_id="0", hide_seed_name=True, max_neighbors=25)
    hyps = ego.neighbor_structure_hypotheses(mass_tol_da=0.05, limit=15)
    print("\n=== PACKAGE HYPS ===")
    for h in hyps:
        print(
            f"  rescue_ok={h.get('rescue_ok')} adduct={h.get('adduct')} "
            f"err={h.get('mass_error_da')} cos={h.get('cosine')} "
            f"conf={h.get('confidence')}"
        )
        print(f"    {(h.get('name') or '')[:100]}")
        print(f"    {h.get('smiles')}")

    # Pipeline predict
    from ego_mol_llm.predict import predict_from_graphml
    from ego_mol_llm.paths import make_run_dir
    from ego_mol_llm.report import export_report

    out = make_run_dir(
        parent=Path("outputs/runs"), graphml=GRAPHML, backend="dry-run", label="pca"
    )
    res = predict_from_graphml(
        GRAPHML, backend="dry-run", seed_id="0", hide_seed_name=True, mass_tol_da=0.05
    )
    export_report(res, out)
    d = res.to_dict()
    print("\n=== DRY-RUN PIPELINE ===")
    for k in (
        "smiles",
        "name",
        "adduct",
        "confidence",
        "mass_ok",
        "source",
        "rescue_notes",
        "seed_mz",
    ):
        print(f"  {k}: {d.get(k)}")
    print("out", out)


if __name__ == "__main__":
    main()
