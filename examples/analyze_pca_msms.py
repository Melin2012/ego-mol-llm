from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

from ego_mol_llm.graphml import load_graphml

ROOT = Path(r"C:\Users\AlexeyMelnik\Downloads\PCA_anonimized")


def parse_mgf(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"(?i)BEGIN IONS", text)
    out = []
    for b in blocks[1:]:
        body = re.split(r"(?i)END IONS", b)[0]
        meta = {}
        peaks = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            if "=" in line and not (line[0].isdigit() or line.startswith(".")):
                k, v = line.split("=", 1)
                meta[k.strip().upper()] = v.strip()
            else:
                parts = re.split(r"[\s\t]+", line)
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
        rec = {"meta": meta, "peaks": peaks}
        if "PEPMASS" in meta:
            rec["pepmass"] = float(meta["PEPMASS"].split()[0])
        out.append(rec)
    return out


def cosine_peaks(p1, p2, tol=0.02):
    if not p1 or not p2:
        return 0.0

    def dens(peaks):
        d = defaultdict(float)
        for mz, inten in peaks:
            if inten <= 0:
                continue
            d[round(mz / tol) * tol] += inten
        norm = math.sqrt(sum(v * v for v in d.values())) or 1.0
        return {k: v / norm for k, v in d.items()}

    a, b = dens(p1), dens(p2)
    score = 0.0
    used = set()
    for ka, va in a.items():
        best = None
        best_d = 1e9
        for kb in b:
            if kb in used:
                continue
            dd = abs(ka - kb)
            if dd <= tol and dd < best_d:
                best_d = dd
                best = kb
        if best is not None:
            score += va * b[best]
            used.add(best)
    return max(0.0, min(1.0, score))


def diagnostic_ions(peaks, neutrals=(18.0106, 17.0265, 35.037, 147.068, 120.081, 166.086)):
    """Check for AA conjugate diagnostics relative to precursor."""
    if not peaks:
        return {}
    base = max(peaks, key=lambda x: x[1])[0]
    # also use PEPMASS if known separately
    ints = {round(m, 3): i for m, i in peaks}
    found = {}
    for mz, inten in peaks:
        # immonium / AA fragments
        if abs(mz - 120.081) < 0.02:
            found["Phe_immonium_120"] = inten
        if abs(mz - 166.086) < 0.02:
            found["Phe_related_166"] = inten
        if abs(mz - 132.102) < 0.02:
            found["Ile_Leu_immonium"] = inten
        if abs(mz - 86.097) < 0.02:
            found["Leu_Ile_immonium_86"] = inten
        if abs(mz - 72.081) < 0.02:
            found["Val_immonium"] = inten
        if abs(mz - 136.076) < 0.02:
            found["Tyr_immonium"] = inten
        if abs(mz - 159.092) < 0.02:
            found["Trp_related"] = inten
    return found


def main():
    ego_list = parse_mgf(ROOT / "Ego_MSMS.mgf")
    all_list = parse_mgf(ROOT / "subgraph_molecules_with_node_ids.mgf")
    print("Ego spectra", len(ego_list), "All spectra", len(all_list))
    seed = ego_list[0]
    print("Seed PEPMASS", seed.get("pepmass"), "npeaks", len(seed["peaks"]))
    tops = sorted(seed["peaks"], key=lambda x: -x[1])[:20]
    print("Top peaks:")
    for m, i in tops:
        print(f"  {m:.4f}\t{i:.1f}")
    print("Diagnostics:", diagnostic_ions(seed["peaks"]))

    net = load_graphml(ROOT / "Bile_Acid_HNSW_test_1.graphml")
    # index mgf by NETWORK_NODE_ID and by pepmass
    by_nid = {}
    by_mz = defaultdict(list)
    for rec in all_list:
        nid = rec["meta"].get("NETWORK_NODE_ID")
        if nid:
            by_nid[str(nid)] = rec
        if "pepmass" in rec:
            by_mz[round(rec["pepmass"], 3)].append(rec)

    print("graph nodes", len(net.nodes), "mgf with NETWORK_NODE_ID", len(by_nid))
    print("overlap node ids", len(set(net.nodes) & set(by_nid)))

    # Compare seed MSMS to each direct neighbor
    print("\n=== Seed MS/MS vs direct neighbor spectra ===")
    rows = []
    for node, edge in net.neighbors("0"):
        sp = by_nid.get(node.id)
        if not sp:
            # try pepmass match
            if node.mz:
                cands = by_mz.get(round(node.mz, 3), [])
                sp = cands[0] if cands else None
        if not sp:
            continue
        c = cosine_peaks(seed["peaks"], sp["peaks"])
        rows.append((c, edge.cosine or 0, edge.abs_diff_mz, node, sp))
    rows.sort(reverse=True)
    for c, ecos, dmz, node, sp in rows[:25]:
        print(
            f"msms={c:.3f} edge={ecos:.3f} dmz={dmz} mz={node.mz} "
            f"{(node.name or '')[:90]}"
        )
        if c > 0.5:
            print("   diag", diagnostic_ions(sp["peaks"]))

    # Characteristic BA fragment neutrals from precursor 556.363
    print("\n=== Neutral losses from 556.363 in seed spectrum ===")
    prec = 556.363
    for m, i in sorted(seed["peaks"], key=lambda x: -x[1])[:40]:
        loss = prec - m
        if 10 < loss < 200:
            label = ""
            if abs(loss - 18.01) < 0.02:
                label = "H2O"
            if abs(loss - 36.02) < 0.03:
                label = "2H2O"
            if abs(loss - 17.03) < 0.02:
                label = "NH3"
            if abs(loss - 46.01) < 0.03:
                label = "HCOOH?"
            if abs(loss - 147.07) < 0.05:
                label = "Phe?"
            if abs(loss - 165.08) < 0.05:
                label = "Phe+H2O?"
            if abs(loss - 390.28) < 0.1:
                label = "BA core?"
            if label or i > 20000:
                print(f"  frag {m:.4f} loss={loss:.4f} inten={i:.0f} {label}")

    # Isobar class vote weighted
    print("\n=== Weighted class vote (cos>=0.7) ===")
    votes = defaultdict(float)
    for node, edge in net.neighbors("0"):
        cos = edge.cosine or 0
        if cos < 0.7:
            continue
        name = (node.name or "").lower()
        if "phe-ca" in name or "phenylalano" in name:
            votes["Phe-CA"] += cos
        elif "aspart" in name:
            votes["Asp-CA"] += cos
        elif "isoleuco" in name or "ile" in name and "cholic" in name:
            votes["Ile/Leu-CA"] += cos
        elif "valine" in name:
            votes["Val-CA"] += cos
        elif "tauro" in name:
            votes["TCA"] += cos
        elif "tryptophan" in name:
            votes["Trp-BA"] += cos
        elif "muricholic" in name:
            votes["MCA-conj"] += cos
        elif "trihydroxy" in name or "cholic" in name:
            votes["other_trihydroxy_CA_conj"] += cos
        else:
            votes["other"] += cos
    for k, v in sorted(votes.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.3f}")

    # 2-hop high cos from hop1 nodes
    print("\n=== 2-hop nodes with SMILES Phe-CA and high support ===")
    hop1 = {n.id for n, _ in net.neighbors("0")}
    phe_smiles = None
    for node, edge in net.neighbors("0"):
        if node.smiles and "Phe-CA" in (node.name or ""):
            phe_smiles = node.smiles
            print("direct Phe-CA SMILES", node.smiles[:120], "mz", node.mz, "cos", edge.cosine)

    # MS/MS support for Phe: strong 120 and 166
    print("\nPhe immonium intensity ratio 120/base")
    base = max(i for _, i in seed["peaks"])
    for m, i in seed["peaks"]:
        if abs(m - 120.08) < 0.02:
            print("  120 intensity", i, "rel", i / base)
        if abs(m - 166.086) < 0.02:
            print("  166 intensity", i, "rel", i / base)


if __name__ == "__main__":
    main()
