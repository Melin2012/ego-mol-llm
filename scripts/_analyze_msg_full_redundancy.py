#!/usr/bin/env python3
"""Redundancy analysis for MSG HNSW full 11540 sealed truth."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

sealed = Path(
    r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_full11540_nist_neigh_SEALED_truth\truth_index.csv"
)
out = Path(r"C:\Users\AlexeyMelnik\HNSW_Large_Files\MSG_FULL_NEW_ANALYSIS")
out.mkdir(exist_ok=True)

rows = list(csv.DictReader(sealed.open(encoding="utf-8-sig")))
print("n", len(rows))


def mol(s):
    return Chem.MolFromSmiles(s.strip()) if s and str(s).strip() else None


def ik1(s):
    m = mol(s)
    return Chem.MolToInchiKey(m).split("-")[0].upper() if m else None


def can(s):
    m = mol(s)
    return Chem.MolToSmiles(m) if m else None


def form(s):
    m = mol(s)
    return rdMolDescriptors.CalcMolFormula(m) if m else None


for r in rows:
    smi = r.get("true_smiles") or ""
    r["_can"] = can(smi)
    r["_ik1"] = (
        ik1(smi)
        or (r.get("true_inchikey") or "").split("-")[0].upper()
        or None
    )
    r["_form"] = form(smi) or r.get("true_formula")
    try:
        r["_mz"] = float(r.get("precursor_mz") or 0)
    except Exception:
        r["_mz"] = None
    r["_fold"] = (r.get("fold") or "").lower()

smiles_c = Counter(r["_can"] for r in rows if r["_can"])
ik1_c = Counter(r["_ik1"] for r in rows if r["_ik1"])
form_c = Counter(r["_form"] for r in rows if r["_form"])
msg_c = Counter(r.get("msg_identifier") for r in rows if r.get("msg_identifier"))
mz_c = Counter(round(r["_mz"], 3) for r in rows if r["_mz"] is not None)
mz1_c = Counter(round(r["_mz"], 1) for r in rows if r["_mz"] is not None)

n = len(rows)
print(
    "unique can",
    len(smiles_c),
    "unique ik1",
    len(ik1_c),
    "unique formula",
    len(form_c),
    "unique msg_id",
    len(msg_c),
)
print("unique mz@0.003", len(mz_c), "mz@0.1", len(mz1_c))
print(f"spectra/unique_ik1 = {n / len(ik1_c):.2f}")
print(f"spectra/unique_smiles = {n / len(smiles_c):.2f}")


def mult_stats(counter: Counter, label: str):
    counts = sorted(counter.values(), reverse=True)
    print(f"\n{label}: n_unique={len(counts)}")
    print("  top10 mult:", counts[:10])
    sing = sum(1 for c in counts if c == 1)
    print(f"  singletons={sing} ({100 * sing / len(counts):.1f}%)")
    print(
        f"  mult>=2={sum(1 for c in counts if c >= 2)} "
        f">=5={sum(1 for c in counts if c >= 5)} "
        f">=10={sum(1 for c in counts if c >= 10)} "
        f">=50={sum(1 for c in counts if c >= 50)}"
    )
    for k in [1, 10, 50, 100, 500]:
        cum = sum(counts[:k])
        print(f"  top {k} cover {cum} spectra ({100 * cum / n:.1f}%)")


mult_stats(ik1_c, "IK1 multiplicity")
mult_stats(smiles_c, "canonical SMILES multiplicity")
mult_stats(form_c, "formula multiplicity")
mult_stats(mz_c, "mz@0.003 multiplicity")

print("\n=== TOP 25 molecules ===")
for ik, cnt in ik1_c.most_common(25):
    sample = next(r for r in rows if r["_ik1"] == ik)
    folds = Counter(r["_fold"] for r in rows if r["_ik1"] == ik)
    smi = (sample["_can"] or "")[:55]
    print(
        f"{cnt:5d} ({100 * cnt / n:5.2f}%)  {sample['_form']}  {ik}  "
        f"folds={dict(folds)}  {smi}"
    )

for thr in [2, 5, 10, 20]:
    covered = sum(c for c in ik1_c.values() if c >= thr)
    n_ent = sum(1 for c in ik1_c.values() if c >= thr)
    print(
        f"spectra in ik1 mult>={thr}: {covered}/{n} ({100 * covered / n:.1f}%) "
        f"across {n_ent} molecules"
    )

print("\n=== by fold ===")
for fold in ["train", "val", "test"]:
    sub = [r for r in rows if r["_fold"] == fold]
    if not sub:
        continue
    uik = len({r["_ik1"] for r in sub if r["_ik1"]})
    usm = len({r["_can"] for r in sub if r["_can"]})
    print(
        f"{fold}: n={len(sub)} unique_ik1={uik} unique_smiles={usm} "
        f"redun={len(sub) / uik:.2f}x"
    )

ik_folds: dict[str, set[str]] = defaultdict(set)
for r in rows:
    if r["_ik1"]:
        ik_folds[r["_ik1"]].add(r["_fold"])
cross = sum(1 for fset in ik_folds.values() if len(fset) > 1)
print(f"molecules appearing in >1 fold: {cross}")

print("\n=== within top5: CE/adduct diversity ===")
for ik, cnt in ik1_c.most_common(5):
    sub = [r for r in rows if r["_ik1"] == ik]
    energies = Counter(r.get("collision_energy") for r in sub)
    adducts = Counter(r.get("adduct") for r in sub)
    instruments = Counter(r.get("instrument_type") for r in sub)
    mzs = {round(r["_mz"], 4) for r in sub if r["_mz"] is not None}
    print(
        f"{ik} n={cnt} unique_mz={len(mzs)} n_adduct={len(adducts)} "
        f"n_CE={len(energies)} instr={dict(instruments)}"
    )
    print("  top CE", energies.most_common(5))
    print("  adducts", adducts.most_common(5))

mz_to_ik: dict[float, set[str]] = defaultdict(set)
for r in rows:
    if r["_mz"] is not None and r["_ik1"]:
        mz_to_ik[round(r["_mz"], 3)].add(r["_ik1"])
ambig = {mz: iks for mz, iks in mz_to_ik.items() if len(iks) > 1}
ambig_spec = sum(
    1
    for r in rows
    if r["_mz"] is not None and len(mz_to_ik[round(r["_mz"], 3)]) > 1
)
print(f"\nmz bins with >1 molecule: {len(ambig)} / {len(mz_to_ik)}")
print(f"spectra in multi-molecule mz bins: {ambig_spec}")

all_top = []
for ik, cnt in ik1_c.most_common(500):
    sample = next(r for r in rows if r["_ik1"] == ik)
    folds = Counter(r["_fold"] for r in rows if r["_ik1"] == ik)
    all_top.append(
        {
            "true_ik1": ik,
            "n_spectra": cnt,
            "pct_of_pack": round(100 * cnt / n, 3),
            "true_formula": sample["_form"],
            "true_smiles": sample["_can"] or "",
            "msg_id_example": sample.get("msg_identifier"),
            "n_train": folds.get("train", 0),
            "n_val": folds.get("val", 0),
            "n_test": folds.get("test", 0),
        }
    )
with (out / "REDUNDANCY_TOP_MOLECULES.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    w = csv.DictWriter(f, fieldnames=list(all_top[0].keys()))
    w.writeheader()
    w.writerows(all_top)

with (out / "REDUNDANCY_IK1_HISTOGRAM.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    w = csv.writer(f)
    w.writerow(["multiplicity", "n_molecules", "n_spectra"])
    hc = Counter(ik1_c.values())
    for m in sorted(hc):
        w.writerow([m, hc[m], m * hc[m]])

summary = {
    "n_spectra": n,
    "unique_canonical_smiles": len(smiles_c),
    "unique_ik1": len(ik1_c),
    "unique_formula": len(form_c),
    "unique_msg_identifier": len(msg_c),
    "unique_mz_0p001": len(mz_c),
    "spectra_per_unique_ik1": round(n / len(ik1_c), 3),
    "singleton_ik1_frac_of_mols": round(
        sum(1 for c in ik1_c.values() if c == 1) / len(ik1_c), 4
    ),
    "spectra_in_mult_ge_2": sum(c for c in ik1_c.values() if c >= 2),
    "spectra_in_mult_ge_10": sum(c for c in ik1_c.values() if c >= 10),
    "top10_ik1_cover_spectra": sum(c for _, c in ik1_c.most_common(10)),
    "top10_ik1_cover_pct": round(
        100 * sum(c for _, c in ik1_c.most_common(10)) / n, 2
    ),
    "top100_ik1_cover_pct": round(
        100 * sum(c for _, c in ik1_c.most_common(100)) / n, 2
    ),
    "molecules_in_multiple_folds": cross,
    "mz_bins_multi_molecule": len(ambig),
    "spectra_in_multi_mol_mz_bins": ambig_spec,
    "by_fold": {},
}
for fold in ["train", "val", "test"]:
    sub = [r for r in rows if r["_fold"] == fold]
    uik = len({r["_ik1"] for r in sub if r["_ik1"]})
    summary["by_fold"][fold] = {
        "n": len(sub),
        "unique_ik1": uik,
        "redundancy_factor": round(len(sub) / uik, 3) if uik else None,
    }

(out / "REDUNDANCY_SUMMARY.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

md = f"""# Redundancy analysis — MSG HNSW full 11,540 block

## Spectrum vs molecule counts
| Metric | n |
|--------|--:|
| Spectra (ego nets) | **{n}** |
| Unique IK1 | **{len(ik1_c)}** |
| Unique canonical SMILES | **{len(smiles_c)}** |
| Unique formulas | **{len(form_c)}** |
| Unique MSG IDs | **{len(msg_c)}** |
| Unique m/z (±0.001) | **{len(mz_c)}** |
| **Spectra / unique IK1** | **{n / len(ik1_c):.2f}×** |

## Concentration
| | Spectra | % pack |
|--|--:|--:|
| Top 10 molecules | {sum(c for _, c in ik1_c.most_common(10))} | {100 * sum(c for _, c in ik1_c.most_common(10)) / n:.1f}% |
| Top 100 molecules | {sum(c for _, c in ik1_c.most_common(100))} | {100 * sum(c for _, c in ik1_c.most_common(100)) / n:.1f}% |
| Mult ≥ 2 | {sum(c for c in ik1_c.values() if c >= 2)} | {100 * sum(c for c in ik1_c.values() if c >= 2) / n:.1f}% |
| Mult ≥ 10 | {sum(c for c in ik1_c.values() if c >= 10)} | {100 * sum(c for c in ik1_c.values() if c >= 10) / n:.1f}% |

Singleton molecules (1 spectrum only): {sum(1 for c in ik1_c.values() if c == 1)} / {len(ik1_c)} mols ({100 * sum(1 for c in ik1_c.values() if c == 1) / len(ik1_c):.1f}%).

## By fold
| Fold | Spectra | Unique IK1 | Redundancy |
|------|--------:|-----------:|------------|
| train | {summary['by_fold']['train']['n']} | {summary['by_fold']['train']['unique_ik1']} | {summary['by_fold']['train']['redundancy_factor']}× |
| val | {summary['by_fold']['val']['n']} | {summary['by_fold']['val']['unique_ik1']} | {summary['by_fold']['val']['redundancy_factor']}× |
| test | {summary['by_fold']['test']['n']} | {summary['by_fold']['test']['unique_ik1']} | {summary['by_fold']['test']['redundancy_factor']}× |

Molecules in >1 fold: {cross}.

## Interpretation
High **spectral** redundancy for the same structures (CE / adduct / instrument replicates). Scoring all 11,540 spectra **overweights frequent molecules**. Prefer unique-molecule metrics, test-only (n={summary['by_fold']['test']['n']}), or dedupe (~{len(ik1_c)} jobs if 1 per IK1).

Files: REDUNDANCY_TOP_MOLECULES.csv, REDUNDANCY_IK1_HISTOGRAM.csv, REDUNDANCY_SUMMARY.json
"""
(out / "REDUNDANCY_REPORT.md").write_text(md, encoding="utf-8")
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
print("wrote", out / "REDUNDANCY_REPORT.md")
