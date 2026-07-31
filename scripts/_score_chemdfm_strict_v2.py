"""Score ChemDFM-R predictions on MSG HNSW strict v2 blind pack."""
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

pack = Path(r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_blind285v2_strict_ego_msms")
sealed = Path(r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_blind285v2_strict_ego_msms_SEALED_truth")
pred_dir = pack / "predictions_chemdfm_r"
out_json = pack / "CHEMDFM_R_STRICT_V2_RESULTS.json"
out_csv = pack / "CHEMDFM_R_STRICT_V2_RESULTS.csv"
grok_json = pack / "GROK_FREEFORM_STRICT_V2_RESULTS.json"


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


truth = {}
with (sealed / "truth_index.csv").open(encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        truth[r["spectrum_id"]] = r

rows = []
rat_lens = []
confs = []
models = Counter()
methods = Counter()
pred_smiles_set = set()
empty_smiles = 0
invalid_smiles = 0
short_rat = 0
mol_ik1_hits = defaultdict(list)

for sid in sorted(truth.keys()):
    t = truth[sid]
    ppath = pred_dir / f"{sid}.json"
    if not ppath.exists():
        pred = {}
        missing = True
    else:
        pred = json.loads(ppath.read_text(encoding="utf-8-sig"))
        missing = False

    ps = (pred.get("smiles") or "").strip()
    ts = (t.get("true_smiles") or "").strip()
    tik_sealed = (t.get("true_inchikey") or "").split("-")[0].upper()
    tik = ik1(ts) or tik_sealed
    pik = ik1(ps)

    if ps and not mol(ps):
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
    methods[pred.get("method") or pred.get("source") or "MISSING"] += 1

    if tik:
        mol_ik1_hits[tik].append(ik_ok)

    rows.append(
        {
            "spectrum_id": sid,
            "true_name": t.get("true_name"),
            "true_formula": t.get("true_formula"),
            "true_smiles": ts,
            "true_ik1": tik,
            "pred_smiles": ps,
            "pred_name": pred.get("iupac_or_common_name"),
            "pred_formula": pred.get("formula") or pf,
            "pred_ik1": pik,
            "confidence": c,
            "ik1": ik_ok,
            "exact": exact_ok,
            "formula_match": form_ok,
            "tanimoto": None if tt is None else round(tt, 4),
            "tani_ge_0.7": bool(tt is not None and tt >= 0.7),
            "tani_ge_0.85": bool(tt is not None and tt >= 0.85),
            "empty_smiles": not bool(ps),
            "invalid_smiles": bool(ps and not mol(ps)),
            "missing_pred": missing,
            "rationale_len": len(rat),
            "model": pred.get("model"),
            "source": pred.get("source") or pred.get("method"),
            "msg_identifier": t.get("msg_identifier"),
            "hnsw_stem": t.get("hnsw_stem"),
        }
    )

n = len(rows)
ik1_n = sum(1 for r in rows if r["ik1"])
exact_n = sum(1 for r in rows if r["exact"])
form_n = sum(1 for r in rows if r["formula_match"])
t07 = sum(1 for r in rows if r["tani_ge_0.7"])
t085 = sum(1 for r in rows if r["tani_ge_0.85"])
n_pred = sum(1 for r in rows if not r["empty_smiles"])
unique_true_ik1 = len(mol_ik1_hits)
unique_mol_ik1 = sum(1 for v in mol_ik1_hits.values() if any(v))

mol_stats = []
for tik, hits in sorted(mol_ik1_hits.items(), key=lambda x: -len(x[1])):
    name = next((r["true_name"] for r in rows if r["true_ik1"] == tik), "")
    form = next((r["true_formula"] for r in rows if r["true_ik1"] == tik), "")
    mol_stats.append(
        {
            "true_ik1": tik,
            "true_name": name,
            "true_formula": form,
            "n_spectra": len(hits),
            "ik1_hits": sum(hits),
            "ik1_rate": round(sum(hits) / len(hits), 4) if hits else 0,
        }
    )

# contingency vs grok if available
contingency = None
if grok_json.exists():
    g = json.loads(grok_json.read_text(encoding="utf-8"))
    gmap = {r["spectrum_id"]: r for r in g.get("rows", [])}
    both = free_only = chem_only = both_miss = 0
    for r in rows:
        gr = gmap.get(r["spectrum_id"], {})
        gi = bool(gr.get("ik1"))
        ci = bool(r["ik1"])
        if gi and ci:
            both += 1
        elif gi and not ci:
            free_only += 1
        elif ci and not gi:
            chem_only += 1
        else:
            both_miss += 1
    contingency = {
        "both_ik1": both,
        "grok_only": free_only,
        "chemdfm_only": chem_only,
        "both_miss": both_miss,
        "grok_ik1": g.get("ik1"),
        "grok_ik1_rate": g.get("ik1_rate"),
    }

summary = {
    "pack": str(pack),
    "protocol": "strict_blind_freeform_v2",
    "arm": "chemdfm_r_pure_model",
    "model": "chemdfm-r-14b",
    "settings": {
        "rescue": False,
        "library": False,
        "hybrid": False,
        "seed_id": "9999999",
        "temperature": 0.2,
        "max_new_tokens": 2048,
    },
    "n": n,
    "n_predicted_nonempty_smiles": n_pred,
    "empty_smiles": empty_smiles,
    "invalid_smiles": invalid_smiles,
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
    "audit": {
        "short_rationale_lt100": short_rat,
        "rationale_len_median": int(statistics.median(rat_lens)) if rat_lens else None,
        "rationale_len_mean": round(statistics.mean(rat_lens), 1) if rat_lens else None,
        "confidence_median": round(statistics.median(confs), 4) if confs else None,
        "confidence_mean": round(statistics.mean(confs), 4) if confs else None,
        "models": dict(models),
        "methods": dict(methods),
    },
    "per_molecule": mol_stats,
    "contingency_vs_grok_v2": contingency,
    "rows": rows,
}

out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

csv_fields = [
    "spectrum_id",
    "true_name",
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
    "invalid_smiles",
    "rationale_len",
    "pred_smiles",
    "true_smiles",
    "msg_identifier",
    "hnsw_stem",
]
with out_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)

print("=" * 60)
print("ChemDFM-R pure model — STRICT V2 (MSG HNSW blind 285)")
print("=" * 60)
print(f"n                     {n}")
print(f"IK1                   {ik1_n}/{n}  ({100 * ik1_n / n:.1f}%)")
print(f"exact SMILES          {exact_n}/{n}  ({100 * exact_n / n:.1f}%)")
print(f"formula match         {form_n}/{n}  ({100 * form_n / n:.1f}%)")
print(f"Tanimoto >= 0.7       {t07}/{n}  ({100 * t07 / n:.1f}%)")
print(f"Tanimoto >= 0.85      {t085}/{n}  ({100 * t085 / n:.1f}%)")
print(f"nonempty SMILES       {n_pred}/{n}")
print(f"empty SMILES          {empty_smiles}/{n}")
print(f"invalid SMILES        {invalid_smiles}/{n}")
print(f"unique pred SMILES    {len(pred_smiles_set)}")
print(f"unique true IK1       {unique_true_ik1}")
print(
    f"unique-mol IK1 (any)  {unique_mol_ik1}/{unique_true_ik1}  "
    f"({100 * unique_mol_ik1 / unique_true_ik1:.1f}%)"
)
print(f"rationale len median  {summary['audit']['rationale_len_median']}")
print(f"confidence median     {summary['audit']['confidence_median']}")
print(f"models                {dict(models)}")
if contingency:
    print()
    print("--- contingency IK1 vs Grok v2 freeform ---")
    print(f"  both         {contingency['both_ik1']}")
    print(f"  grok only    {contingency['grok_only']}")
    print(f"  chemdfm only {contingency['chemdfm_only']}")
    print(f"  both miss    {contingency['both_miss']}")
    print(f"  grok IK1     {contingency['grok_ik1']}/285")
print()
print("Per-molecule:")
for m in mol_stats:
    print(
        f"  {m['true_ik1']}  n={m['n_spectra']:3d}  hits={m['ik1_hits']:3d}  "
        f"({100 * m['ik1_rate']:.0f}%)  {m['true_formula']}  {m['true_name']}"
    )
print()
print(f"wrote {out_json}")
print(f"wrote {out_csv}")
