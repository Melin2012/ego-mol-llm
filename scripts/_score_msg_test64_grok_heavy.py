#!/usr/bin/env python3
"""Score Grok Heavy free-form predictions on MSG HNSW test64 pack."""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

pack = Path(r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_test64_nist_neigh_handout")
sealed = Path(r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_test64_nist_neigh_SEALED_truth")
pred_dir = pack / "predictions_grok_heavy" / "predictions"
if not pred_dir.is_dir():
    # fallback: flat
    pred_dir = pack / "predictions_grok_heavy"

out_dir = sealed
out_json = out_dir / "GROK_HEAVY_TEST64_RESULTS.json"
out_csv = out_dir / "GROK_HEAVY_TEST64_RESULTS.csv"
out_buckets = out_dir / "GROK_HEAVY_TEST64_BUCKETS.csv"


def mol(s):
    if not s or not str(s).strip():
        return None
    return Chem.MolFromSmiles(str(s).strip())


def can(s):
    m = mol(s)
    return Chem.MolToSmiles(m) if m else None


def ik1(s):
    m = mol(s)
    if not m:
        return None
    return Chem.MolToInchiKey(m).split("-")[0].upper()


def formula(s):
    m = mol(s)
    if not m:
        return None
    return rdMolDescriptors.CalcMolFormula(m)


def tani(a, b):
    ma, mb = mol(a), mol(b)
    if not ma or not mb:
        return None
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def bucket(ik_ok, exact_ok, form_ok, tt, empty):
    if empty:
        return "empty"
    if exact_ok:
        return "exact"
    if ik_ok:
        return "ik1_not_exact"
    if tt is not None and tt >= 0.7:
        return "similar_t07"
    if form_ok:
        return "formula_only"
    return "true_miss"


truth = {}
with (sealed / "truth_index.csv").open(encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        truth[r["spectrum_id"]] = r

rows = []
rat_lens = []
confs = []
ranker_phrase = 0
models = Counter()
methods = Counter()
pred_smiles_set = set()
empty_smiles = 0
short_rat = 0
invalid_smiles = 0
mol_ik1_hits = defaultdict(list)
mtime_list = []

for sid in sorted(truth.keys()):
    t = truth[sid]
    ppath = pred_dir / f"{sid}.json"
    if not ppath.exists():
        # try without nested
        ppath = pack / "predictions_grok_heavy" / f"{sid}.json"
    if not ppath.exists():
        pred = {}
        missing = True
    else:
        pred = json.loads(ppath.read_text(encoding="utf-8-sig"))
        missing = False
        mtime_list.append(ppath.stat().st_mtime)

    ps = (pred.get("smiles") or "").strip()
    ts = (t.get("true_smiles") or "").strip()
    tik_sealed = (t.get("true_inchikey") or "").split("-")[0].upper()
    tik_rdkit = ik1(ts)
    tik = tik_rdkit or tik_sealed
    pik = ik1(ps)
    if ps and not pik:
        invalid_smiles += 1

    ik_ok = bool(pik and tik and pik == tik)
    exact_ok = bool(can(ps) and can(ts) and can(ps) == can(ts))
    pf = formula(ps)
    tf = formula(ts) or (t.get("true_formula") or "").strip() or None
    form_ok = bool(pf and tf and pf == tf)
    tt = tani(ps, ts)

    rat = pred.get("rationale") or ""
    rat_lens.append(len(rat))
    if len(rat) < 100:
        short_rat += 1
    low = rat.lower()
    if any(
        x in low
        for x in (
            "pack-wide",
            "packwide",
            "smiles index",
            "ranker",
            "top of index",
            "library index of",
            "mass-transfer",
            "mz cluster",
            "m/z cluster",
            "harvested from other",
        )
    ):
        ranker_phrase += 1

    c = pred.get("confidence")
    try:
        confs.append(float(c))
    except (TypeError, ValueError):
        pass

    if not ps:
        empty_smiles += 1
    else:
        pred_smiles_set.add(can(ps) or ps)

    models[pred.get("model") or "MISSING"] += 1
    methods[pred.get("method") or "MISSING"] += 1

    if tik:
        mol_ik1_hits[tik].append(ik_ok)

    b = bucket(ik_ok, exact_ok, form_ok, tt, not bool(ps))

    rows.append(
        {
            "spectrum_id": sid,
            "fold": t.get("fold"),
            "true_formula": t.get("true_formula") or tf,
            "true_smiles": ts,
            "true_ik1": tik,
            "true_ik1_sealed": tik_sealed,
            "pred_smiles": ps,
            "pred_name": pred.get("iupac_or_common_name"),
            "pred_formula": pred.get("formula") or pf,
            "pred_ik1": pik,
            "pred_adduct": pred.get("adduct"),
            "confidence": c,
            "ik1": ik_ok,
            "exact": exact_ok,
            "formula_match": form_ok,
            "tanimoto": None if tt is None else round(tt, 4),
            "tani_ge_0.7": bool(tt is not None and tt >= 0.7),
            "tani_ge_0.85": bool(tt is not None and tt >= 0.85),
            "bucket": b,
            "empty_smiles": not bool(ps),
            "missing_pred": missing,
            "rationale_len": len(rat),
            "model": pred.get("model"),
            "method": pred.get("method"),
            "msg_identifier": t.get("msg_identifier"),
            "hnsw_index": t.get("hnsw_index"),
            "precursor_mz": t.get("precursor_mz"),
        }
    )

n = len(rows)
ik1_n = sum(1 for r in rows if r["ik1"])
exact_n = sum(1 for r in rows if r["exact"])
form_n = sum(1 for r in rows if r["formula_match"])
t07 = sum(1 for r in rows if r["tani_ge_0.7"])
t085 = sum(1 for r in rows if r["tani_ge_0.85"])
n_pred = sum(1 for r in rows if not r["empty_smiles"])
missing_n = sum(1 for r in rows if r["missing_pred"])
bucket_c = Counter(r["bucket"] for r in rows)

unique_true_ik1 = len(mol_ik1_hits)
unique_mol_ik1 = sum(1 for v in mol_ik1_hits.values() if any(v))

# unique-mol metrics: one row per true IK1 — any-hit already; also first-spectrum rate
first_ik1 = {}
for r in rows:
    tik = r["true_ik1"]
    if tik and tik not in first_ik1:
        first_ik1[tik] = r["ik1"]
unique_first_ik1 = sum(1 for v in first_ik1.values() if v)

mol_stats = []
for tik, hits in sorted(mol_ik1_hits.items(), key=lambda x: (-len(x[1]), x[0])):
    form = next((r["true_formula"] for r in rows if r["true_ik1"] == tik), "")
    mol_stats.append(
        {
            "true_ik1": tik,
            "true_formula": form,
            "n_spectra": len(hits),
            "ik1_hits": sum(hits),
            "ik1_rate": round(sum(hits) / len(hits), 4) if hits else 0,
        }
    )

span_s = None
if mtime_list:
    span_s = max(mtime_list) - min(mtime_list)

# free-form audit heuristic
ff_flags = []
if span_s is not None and span_s < 5 * 60 and n >= 50:
    ff_flags.append(
        f"file_mtime_span_short={span_s:.1f}s (~{span_s/60:.1f} min) for n={n} "
        "(may be zip write times, not generation wall time)"
    )
if ranker_phrase:
    ff_flags.append(f"ranker_phrase_files={ranker_phrase}")
if short_rat > n * 0.5:
    ff_flags.append(f"many_short_rationales={short_rat}/{n}")
if statistics.median(rat_lens) >= 200 and ranker_phrase == 0:
    freeform_verdict = "LIKELY_VALID_FREEFORM"
    freeform_note = (
        "Per-sample JSON with substantive MASS/NETWORK/MS/MS rationales; "
        "no pack-wide ranker phrases. Short mtime span alone is not disqualifying "
        "(zip timestamps / parallel batch possible)."
    )
else:
    freeform_verdict = "AUDIT_NEEDED"
    freeform_note = "Check rationale diversity and generation logs."

summary = {
    "pack": str(pack),
    "sealed": str(sealed),
    "pred_dir": str(pred_dir),
    "source_zip": r"C:\Users\AlexeyMelnik\Downloads\MSG_HNSW_test64_predictions.zip",
    "protocol": "strict_blind_freeform_fold_test_nist_neighbors",
    "arm": "grok_heavy_freeform_test64",
    "n": n,
    "n_missing_pred": missing_n,
    "n_predicted_nonempty_smiles": n_pred,
    "empty_smiles": empty_smiles,
    "invalid_pred_smiles_parse": invalid_smiles,
    "ik1": ik1_n,
    "ik1_rate": round(ik1_n / n, 4) if n else None,
    "exact": exact_n,
    "exact_rate": round(exact_n / n, 4) if n else None,
    "formula_match": form_n,
    "formula_match_rate": round(form_n / n, 4) if n else None,
    "tani_ge_0.7": t07,
    "tani_ge_0.7_rate": round(t07 / n, 4) if n else None,
    "tani_ge_0.85": t085,
    "tani_ge_0.85_rate": round(t085 / n, 4) if n else None,
    "unique_pred_smiles": len(pred_smiles_set),
    "unique_true_ik1": unique_true_ik1,
    "unique_mol_ik1_any": unique_mol_ik1,
    "unique_mol_ik1_any_rate": round(unique_mol_ik1 / unique_true_ik1, 4)
    if unique_true_ik1
    else None,
    "unique_mol_ik1_first_spectrum": unique_first_ik1,
    "buckets": dict(bucket_c),
    "freeform_validity": {
        "verdict": freeform_verdict,
        "note": freeform_note,
        "flags": ff_flags,
        "file_mtime_span_s": None if span_s is None else round(span_s, 1),
    },
    "audit": {
        "ranker_phrase_files": ranker_phrase,
        "short_rationale_lt100": short_rat,
        "rationale_len_median": int(statistics.median(rat_lens)) if rat_lens else None,
        "rationale_len_mean": round(statistics.mean(rat_lens), 1) if rat_lens else None,
        "rationale_len_min": min(rat_lens) if rat_lens else None,
        "rationale_len_max": max(rat_lens) if rat_lens else None,
        "confidence_median": round(statistics.median(confs), 4) if confs else None,
        "confidence_mean": round(statistics.mean(confs), 4) if confs else None,
        "models": dict(models),
        "methods": dict(methods),
    },
    "per_molecule": mol_stats,
    "rows": rows,
}

out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

csv_fields = [
    "spectrum_id",
    "fold",
    "bucket",
    "true_formula",
    "true_ik1",
    "pred_name",
    "pred_formula",
    "pred_ik1",
    "confidence",
    "ik1",
    "exact",
    "formula_match",
    "tanimoto",
    "tani_ge_0.7",
    "tani_ge_0.85",
    "empty_smiles",
    "rationale_len",
    "pred_smiles",
    "true_smiles",
    "pred_adduct",
    "msg_identifier",
    "hnsw_index",
    "precursor_mz",
    "model",
    "method",
]
with out_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)

with out_buckets.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["bucket", "n", "pct"],
    )
    w.writeheader()
    for b, c in bucket_c.most_common():
        w.writerow({"bucket": b, "n": c, "pct": round(100 * c / n, 1)})

print("=" * 64)
print("GROK HEAVY FREE-FORM — MSG HNSW TEST64 (no IK1 dedupe)")
print("=" * 64)
print(f"n                     {n}  (missing pred files: {missing_n})")
print(f"IK1                   {ik1_n}/{n}  ({100 * ik1_n / n:.1f}%)")
print(f"exact SMILES          {exact_n}/{n}  ({100 * exact_n / n:.1f}%)")
print(f"formula match         {form_n}/{n}  ({100 * form_n / n:.1f}%)")
print(f"Tanimoto >= 0.7       {t07}/{n}  ({100 * t07 / n:.1f}%)")
print(f"Tanimoto >= 0.85      {t085}/{n}  ({100 * t085 / n:.1f}%)")
print(f"nonempty SMILES       {n_pred}/{n}")
print(f"empty SMILES          {empty_smiles}/{n}")
print(f"invalid SMILES parse  {invalid_smiles}/{n}")
print(f"unique pred SMILES    {len(pred_smiles_set)}")
print(f"unique true IK1       {unique_true_ik1}")
print(
    f"unique-mol IK1 (any)  {unique_mol_ik1}/{unique_true_ik1}  "
    f"({100 * unique_mol_ik1 / unique_true_ik1:.1f}%)"
)
print(f"buckets               {dict(bucket_c)}")
print(f"ranker_phrase files   {ranker_phrase}")
print(f"short rationale <100  {short_rat}")
print(f"rationale len median  {summary['audit']['rationale_len_median']}")
print(f"confidence median     {summary['audit']['confidence_median']}")
print(f"models                {dict(models)}")
print(f"mtime span (s)        {summary['freeform_validity']['file_mtime_span_s']}")
print(f"free-form verdict     {freeform_verdict}")
print(f"note                  {freeform_note}")
print()
print("Misses / true_miss sample (up to 12):")
misses = [r for r in rows if r["bucket"] == "true_miss"][:12]
for r in misses:
    print(
        f"  {r['spectrum_id']}  true={r['true_ik1']} {r['true_formula']}  "
        f"pred={r['pred_ik1']} {r['pred_formula']}  T={r['tanimoto']}  "
        f"name={r['pred_name']}"
    )
print()
print(f"wrote {out_json}")
print(f"wrote {out_csv}")
print(f"wrote {out_buckets}")
