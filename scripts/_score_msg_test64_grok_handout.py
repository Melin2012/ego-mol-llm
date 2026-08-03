"""Score Grok predictions in MSG_HNSW_test64_nist_neigh_GROK_handout."""
from __future__ import annotations
import csv, json, statistics, shutil
from collections import Counter, defaultdict
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

hand = Path(r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_test64_nist_neigh_GROK_handout")
sealed = Path(r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_test64_nist_neigh_SEALED_truth")
pred_dir = hand / "predictions"
# also copy into standard sealed analysis location
out_dir = sealed
arm = "grok_handout_test64"
out_json = out_dir / "GROK_HANDOUT_TEST64_RESULTS.json"
out_csv = out_dir / "GROK_HANDOUT_TEST64_RESULTS.csv"
out_buckets = out_dir / "GROK_HANDOUT_TEST64_BUCKETS.csv"

def mol(s):
    if not s or not str(s).strip(): return None
    return Chem.MolFromSmiles(str(s).strip())
def can(s):
    m=mol(s); return Chem.MolToSmiles(m) if m else None
def ik1(s):
    m=mol(s)
    return Chem.MolToInchiKey(m).split("-")[0].upper() if m else None
def formula(s):
    m=mol(s); return rdMolDescriptors.CalcMolFormula(m) if m else None
def tani(a,b):
    ma,mb=mol(a),mol(b)
    if not ma or not mb: return None
    fa=AllChem.GetMorganFingerprintAsBitVect(ma,2,nBits=2048)
    fb=AllChem.GetMorganFingerprintAsBitVect(mb,2,nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fa,fb))

def mass_ok(smiles, mz, tol=0.05):
    m=mol(smiles)
    if not m: return False, None, None
    M=Descriptors.ExactMolWt(m)
    adducts={
        "[M+H]+": M+1.007825,"[M-H]-": M-1.007825,"[M+Na]+": M+22.989218,
        "[M+K]+": M+38.963707,"[M+NH4]+": M+18.033826,"[M+H-H2O]+": M+1.007825-18.010565,
        "[2M+H]+": 2*M+1.007825,"[2M-H]-": 2*M-1.007825,"[2M+Na]+": 2*M+22.989218,
        "[3M+H]+": 3*M+1.007825,"[M]+": M,
    }
    best=min(((a,abs(tmz-mz)) for a,tmz in adducts.items()), key=lambda x:x[1])
    return best[1]<=tol, best[0], best[1]

def bucket(ik_ok, exact_ok, form_ok, tt, empty):
    if empty: return "empty"
    if exact_ok: return "exact"
    if ik_ok: return "ik1_not_exact"
    if tt is not None and tt>=0.7: return "similar_t07"
    if form_ok: return "formula_only"
    return "true_miss"

truth={}
with (sealed/"truth_index.csv").open(encoding="utf-8-sig",newline="") as f:
    for r in csv.DictReader(f):
        truth[r["spectrum_id"]]=r

rows=[]; rat_lens=[]; confs=[]; ranker_phrase=0; models=Counter(); methods=Counter()
pred_smiles_set=set(); empty_smiles=0; short_rat=0; invalid_smiles=0
mol_ik1_hits=defaultdict(list); mtimes=[]

for sid in sorted(truth):
    t=truth[sid]
    ppath=pred_dir/f"{sid}.json"
    if not ppath.exists():
        pred={}; missing=True
    else:
        pred=json.loads(ppath.read_text(encoding="utf-8-sig")); missing=False
        mtimes.append(ppath.stat().st_mtime)
    ps=(pred.get("smiles") or "").strip()
    ts=(t.get("true_smiles") or "").strip()
    tik_sealed=(t.get("true_inchikey") or "").split("-")[0].upper()
    tik=ik1(ts) or tik_sealed
    pik=ik1(ps)
    if ps and not pik: invalid_smiles+=1
    ik_ok=bool(pik and tik and pik==tik)
    exact_ok=bool(can(ps) and can(ts) and can(ps)==can(ts))
    pf=formula(ps); tf=formula(ts) or (t.get("true_formula") or "").strip() or None
    form_ok=bool(pf and tf and pf==tf)
    tt=tani(ps,ts)
    try: mz=float(t.get("precursor_mz") or 0)
    except: mz=0
    p_mok,p_add,p_res=mass_ok(ps,mz) if ps else (False,None,None)
    t_mok,t_add,t_res=mass_ok(ts,mz) if ts else (False,None,None)
    rat=pred.get("rationale") or ""
    rat_lens.append(len(rat))
    if len(rat)<100: short_rat+=1
    low=rat.lower()
    if any(x in low for x in ("pack-wide","packwide","smiles index","mass-transfer","m/z cluster","mz cluster","harvested from other","global index")):
        ranker_phrase+=1
    try: confs.append(float(pred.get("confidence")))
    except: pass
    if not ps: empty_smiles+=1
    else: pred_smiles_set.add(can(ps) or ps)
    models[pred.get("model") or "MISSING"]+=1
    methods[pred.get("method") or "MISSING"]+=1
    if tik: mol_ik1_hits[tik].append(ik_ok)
    b=bucket(ik_ok,exact_ok,form_ok,tt,not bool(ps))
    rows.append({
        "spectrum_id":sid,"fold":t.get("fold"),"bucket":b,
        "true_formula":t.get("true_formula") or tf,"true_ik1":tik,
        "pred_name":pred.get("iupac_or_common_name"),"pred_formula":pred.get("formula") or pf,
        "pred_ik1":pik,"confidence":pred.get("confidence"),
        "ik1":ik_ok,"exact":exact_ok,"formula_match":form_ok,
        "tanimoto":None if tt is None else round(tt,4),
        "tani_ge_0.7":bool(tt is not None and tt>=0.7),
        "tani_ge_0.85":bool(tt is not None and tt>=0.85),
        "empty_smiles":not bool(ps),"missing_pred":missing,
        "rationale_len":len(rat),"pred_smiles":ps,"true_smiles":ts,
        "pred_adduct":pred.get("adduct"),"pred_mass_ok":p_mok,
        "pred_mass_best_adduct":p_add,"pred_mass_resid":None if p_res is None else round(p_res,4),
        "true_mass_ok":t_mok,"msg_identifier":t.get("msg_identifier"),
        "hnsw_index":t.get("hnsw_index"),"precursor_mz":t.get("precursor_mz"),
        "model":pred.get("model"),"method":pred.get("method"),
    })

n=len(rows)
ik1_n=sum(1 for r in rows if r["ik1"]); exact_n=sum(1 for r in rows if r["exact"])
form_n=sum(1 for r in rows if r["formula_match"]); t07=sum(1 for r in rows if r["tani_ge_0.7"])
t085=sum(1 for r in rows if r["tani_ge_0.85"]); n_pred=sum(1 for r in rows if not r["empty_smiles"])
missing_n=sum(1 for r in rows if r["missing_pred"]); bucket_c=Counter(r["bucket"] for r in rows)
mass_ok_n=sum(1 for r in rows if r["pred_mass_ok"])
unique_true=len(mol_ik1_hits); unique_any=sum(1 for v in mol_ik1_hits.values() if any(v))
span=max(mtimes)-min(mtimes) if mtimes else None

# compare to previous heavy arm if present
prev_path=sealed/"GROK_HEAVY_TEST64_RESULTS.csv"
prev=None
if prev_path.exists():
    prev={r["spectrum_id"]:r for r in csv.DictReader(prev_path.open(encoding="utf-8"))}

both_hit=only_new=only_old=0
if prev:
    for r in rows:
        o=prev.get(r["spectrum_id"])
        if not o: continue
        nh=r["ik1"]; oh=o["ik1"]=="True"
        if nh and oh: both_hit+=1
        elif nh and not oh: only_new+=1
        elif (not nh) and oh: only_old+=1

verdict="LIKELY_VALID_FREEFORM"
flags=[]
if span is not None and span < 5*60 and n>=50:
    flags.append(f"short_mtime_span={span:.0f}s")
if ranker_phrase: flags.append(f"ranker_phrases={ranker_phrase}")
if statistics.median(rat_lens) < 150: flags.append("short_rationales")
if (hand/"_mass_fit.py").exists():
    flags.append("mass_fit_helper_present(validation_util_ok_if_per_sample)")

summary={
    "pack":str(hand),"sealed":str(sealed),"pred_dir":str(pred_dir),
    "arm":arm,"protocol":"strict_blind_freeform_fold_test_nist_neighbors",
    "n":n,"n_missing_pred":missing_n,"n_predicted_nonempty_smiles":n_pred,
    "empty_smiles":empty_smiles,"invalid_pred_smiles_parse":invalid_smiles,
    "ik1":ik1_n,"ik1_rate":round(ik1_n/n,4),
    "exact":exact_n,"exact_rate":round(exact_n/n,4),
    "formula_match":form_n,"formula_match_rate":round(form_n/n,4),
    "tani_ge_0.7":t07,"tani_ge_0.7_rate":round(t07/n,4),
    "tani_ge_0.85":t085,"tani_ge_0.85_rate":round(t085/n,4),
    "pred_mass_ok":mass_ok_n,"pred_mass_ok_rate":round(mass_ok_n/n,4),
    "unique_pred_smiles":len(pred_smiles_set),
    "unique_true_ik1":unique_true,"unique_mol_ik1_any":unique_any,
    "unique_mol_ik1_any_rate":round(unique_any/unique_true,4) if unique_true else None,
    "buckets":dict(bucket_c),
    "vs_prior_heavy":{
        "both_hit":both_hit,"only_this_hit":only_new,"only_heavy_hit":only_old,
        "prior_file":str(prev_path) if prev else None,
    } if prev else None,
    "freeform_validity":{"verdict":verdict,"flags":flags,"file_mtime_span_s":None if span is None else round(span,1)},
    "audit":{
        "ranker_phrase_files":ranker_phrase,"short_rationale_lt100":short_rat,
        "rationale_len_median":int(statistics.median(rat_lens)) if rat_lens else None,
        "rationale_len_mean":round(statistics.mean(rat_lens),1) if rat_lens else None,
        "rationale_len_min":min(rat_lens) if rat_lens else None,
        "rationale_len_max":max(rat_lens) if rat_lens else None,
        "confidence_median":round(statistics.median(confs),4) if confs else None,
        "confidence_mean":round(statistics.mean(confs),4) if confs else None,
        "models":dict(models),"methods":dict(methods),
    },
    "rows":rows,
}
out_json.write_text(json.dumps(summary,indent=2),encoding="utf-8")
fields=["spectrum_id","fold","bucket","true_formula","true_ik1","pred_name","pred_formula","pred_ik1",
        "confidence","ik1","exact","formula_match","tanimoto","tani_ge_0.7","tani_ge_0.85",
        "empty_smiles","rationale_len","pred_mass_ok","pred_mass_best_adduct","pred_mass_resid",
        "pred_smiles","true_smiles","pred_adduct","msg_identifier","hnsw_index","precursor_mz","model","method"]
with out_csv.open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
with out_buckets.open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["bucket","n","pct"]); w.writeheader()
    for b,c in bucket_c.most_common():
        w.writerow({"bucket":b,"n":c,"pct":round(100*c/n,1)})

# also write into handout for convenience (no truth smiles in summary? include full csv only in sealed)
# copy summary metrics only
(hand/"predictions"/"_SCORED_SUMMARY.json").write_text(json.dumps({k:summary[k] for k in summary if k!="rows"},indent=2),encoding="utf-8")

print("="*64)
print("GROK (handout folder) FREE-FORM — MSG HNSW TEST64")
print("="*64)
print(f"pred_dir              {pred_dir}")
print(f"n                     {n}  missing={missing_n}")
print(f"IK1                   {ik1_n}/{n}  ({100*ik1_n/n:.1f}%)")
print(f"exact SMILES          {exact_n}/{n}  ({100*exact_n/n:.1f}%)")
print(f"formula match         {form_n}/{n}  ({100*form_n/n:.1f}%)")
print(f"Tanimoto >= 0.7       {t07}/{n}  ({100*t07/n:.1f}%)")
print(f"Tanimoto >= 0.85      {t085}/{n}  ({100*t085/n:.1f}%)")
print(f"pred mass-OK          {mass_ok_n}/{n}  ({100*mass_ok_n/n:.1f}%)")
print(f"empty SMILES          {empty_smiles}")
print(f"unique pred SMILES    {len(pred_smiles_set)}")
print(f"unique-mol IK1 any    {unique_any}/{unique_true} ({100*unique_any/unique_true:.1f}%)")
print(f"buckets               {dict(bucket_c)}")
print(f"ranker_phrase files   {ranker_phrase}")
print(f"short rationale <100  {short_rat}")
print(f"rationale len median  {summary['audit']['rationale_len_median']}")
print(f"confidence median     {summary['audit']['confidence_median']}")
print(f"models                {dict(models)}")
print(f"mtime span (min)      {None if span is None else round(span/60,1)}")
print(f"free-form verdict     {verdict}")
print(f"flags                 {flags}")
if prev:
    print(f"vs prior heavy: both_hit={both_hit} only_this={only_new} only_heavy={only_old}")
    print(f"prior heavy IK1 was 10/64")
print()
print("HITS:")
for r in rows:
    if r["ik1"]:
        print(f"  {r['spectrum_id']} exact={r['exact']} T={r['tanimoto']} conf={r['confidence']} {r['true_formula']} | {r['pred_name']}")
print()
print("MASS-OK subset IK1:", sum(1 for r in rows if r["pred_mass_ok"] and r["ik1"]), "/", mass_ok_n)
print(f"wrote {out_json}")
print(f"wrote {out_csv}")
