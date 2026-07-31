#!/usr/bin/env python3
"""Unblinded analysis of CFM-first NO NIST predictions on holdout-40c."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt

RDLogger.DisableLog("rdApp.*")

PROTON = 1.007276


def mol(s: str | None):
    return Chem.MolFromSmiles(s or "")


def ik1(m_or_s) -> str:
    m = mol(m_or_s) if isinstance(m_or_s, str) else m_or_s
    if not m:
        return ""
    try:
        return Chem.MolToInchiKey(m).split("-")[0].upper()
    except Exception:
        return ""


def formula(m) -> str:
    if not m:
        return ""
    try:
        return rdMolDescriptors.CalcMolFormula(m)
    except Exception:
        return ""


def tani(ma, mb) -> float | None:
    if not ma or not mb:
        return None
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def adduct_fit(smi: str, mz: float, ion: str | None, tol: float = 0.05):
    m = mol(smi)
    if not m or not mz:
        return None
    mw = ExactMolWt(m)
    ion_l = (ion or "").lower()
    if ion_l.startswith("neg"):
        opts = [
            ("[M-H]-", mw - PROTON),
            ("[2M-H]-", 2 * mw - PROTON),
            ("[M-H-H2O]-", mw - PROTON - 18.010565),
        ]
    else:
        opts = [
            ("[M+H]+", mw + PROTON),
            ("[M+Na]+", mw + 22.989221),
            ("[M+NH4]+", mw + 18.033823),
            ("[M+H-H2O]+", mw + PROTON - 18.010565),
            ("[M+H-2H2O]+", mw + PROTON - 2 * 18.010565),
            ("[2M+H]+", 2 * mw + PROTON),
        ]
    best = min(opts, key=lambda x: abs(x[1] - mz))
    err = best[1] - mz
    return {"adduct": best[0], "err": err, "ok": abs(err) <= tol, "mw": mw}


def extract_network_smiles(user: str, mz: float, ion: str | None) -> list[str]:
    found: list[str] = []
    for m in re.finditer(r"SMILES=(\S+)", user or ""):
        smi = m.group(1).strip().rstrip("|")
        mm = mol(smi)
        if not mm:
            continue
        can = Chem.MolToSmiles(mm)
        fit = adduct_fit(can, mz, ion)
        if fit and fit["ok"]:
            found.append(can)
    seen: set[str] = set()
    out: list[str] = []
    for s in found:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main() -> int:
    pack = Path(r"C:\Users\AlexeyMelnik\Desktop\Blind_holdout40c_ego_msms")
    pred_dir = pack / "predictions_cfmfirst_no_nist"
    truth_p = pack.parent / "Blind_holdout40c_ego_msms_SEALED_truth" / "truth_index.csv"

    truth: dict[str, dict] = {}
    with truth_p.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            truth[(r.get("spectrum_id") or "").upper()] = r

    # seed_mz from manifest if needed
    manifest_mz: dict[str, float] = {}
    with (pack / "sample_manifest.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            sid = (r.get("spectrum_id") or "").upper()
            try:
                manifest_mz[sid] = float(r.get("seed_mz") or 0)
            except Exception:
                pass

    rows: list[dict] = []
    for sid, t in sorted(truth.items()):
        pred_p = pred_dir / f"{sid}.json"
        pred = (
            json.loads(pred_p.read_text(encoding="utf-8-sig"))
            if pred_p.is_file()
            else {}
        )
        job_p = pack / "jobs" / f"{sid}.json"
        job = json.loads(job_p.read_text(encoding="utf-8")) if job_p.is_file() else {}
        cfm_p = pack / "cfm_explain" / f"{sid}.json"
        cfm = (
            json.loads(cfm_p.read_text(encoding="utf-8")) if cfm_p.is_file() else {}
        )
        sir_p = pack / "sirius_hits" / f"{sid}.json"
        sir = (
            json.loads(sir_p.read_text(encoding="utf-8-sig"))
            if sir_p.is_file()
            else {}
        )

        true_smi = (t.get("true_smiles") or "").strip()
        true_name = t.get("true_name") or ""
        true_ik = (t.get("true_inchikey") or "").split("-")[0].upper() or ik1(true_smi)
        true_form = t.get("true_formula") or formula(mol(true_smi))
        pred_smi = (pred.get("smiles") or "").strip()
        pred_src = pred.get("source") or ""
        mz = float(job.get("seed_mz") or manifest_mz.get(sid) or 0)
        ion = job.get("seed_ion_mode") or ""

        ma, mb = mol(pred_smi), mol(true_smi)
        pik = ik1(ma)
        tt = tani(ma, mb)
        form_ok = bool(ma and mb and formula(ma) and formula(ma) == formula(mb))
        ik_ok = bool(pik and true_ik and pik == true_ik)
        exact = bool(ma and mb and Chem.MolToSmiles(ma) == Chem.MolToSmiles(mb))
        has = bool(pred_smi and ma)

        true_fit = adduct_fit(true_smi, mz, ion) if mz and true_smi else None
        net_smis = extract_network_smiles(job.get("user_prompt") or "", mz, ion) if mz else []
        true_can = Chem.MolToSmiles(mb) if mb else ""
        truth_in_network = bool(true_can and true_can in set(net_smis))

        cfm_neigh: list[str] = []
        n_trans = 0
        if cfm:
            n_trans = sum(
                1
                for p in (cfm.get("seed_peak_map") or [])
                if p.get("fragment_smiles") and p.get("source") == "transferred"
            )
            for n in cfm.get("neighbors") or []:
                mm = mol(n.get("smiles"))
                if not mm:
                    continue
                smi = Chem.MolToSmiles(mm)
                fit = adduct_fit(smi, mz, ion) if mz else None
                if fit and fit["ok"]:
                    cfm_neigh.append(smi)
        truth_in_cfm = bool(true_can and true_can in set(cfm_neigh))

        csi_hits = sir.get("hits") or []
        csi_top = None
        csi_ik = None
        csi_rank_truth = None
        for i, h in enumerate(csi_hits[:10], 1):
            hm = mol(h.get("smiles"))
            hik = ik1(hm)
            if i == 1:
                csi_top = Chem.MolToSmiles(hm) if hm else h.get("smiles")
                csi_ik = hik
            if hik and hik == true_ik and csi_rank_truth is None:
                csi_rank_truth = i

        if not has:
            outcome = "ABSTAIN"
            why: list[str] = []
            if not truth_in_network and not truth_in_cfm and csi_rank_truth is None:
                why.append(
                    "truth not in mass-OK network/CFM neighbor SMILES; CSI top list lacks truth"
                )
            elif not truth_in_network and not truth_in_cfm:
                why.append("truth not among mass-OK GraphML/CFM neighbor SMILES")
            else:
                why.append("abstain despite possible candidates")
            if true_fit and not true_fit["ok"]:
                why.append(
                    f"truth mass-awkward under common adducts "
                    f"(best err={true_fit['err']:+.4f} as {true_fit['adduct']})"
                )
            reason = "; ".join(why) if why else "no SMILES proposed"
        elif ik_ok:
            if exact:
                outcome = "MATCH_EXACT"
                reason = f"exact SMILES via source={pred_src}"
            else:
                outcome = "MATCH_IK1"
                reason = f"correct connectivity via source={pred_src}"
                if form_ok:
                    reason += " (same formula; stereo/SMILES writing differs)"
        else:
            outcome = "MISS"
            why = [f"wrong structure source={pred_src}"]
            if form_ok:
                why.append("SAME FORMULA (isomer/connectivity error)")
            if tt is not None:
                if tt >= 0.85:
                    why.append(f"very close T={tt:.2f}")
                elif tt >= 0.7:
                    why.append(f"related T={tt:.2f}")
                elif tt >= 0.5:
                    why.append(f"somewhat similar T={tt:.2f}")
                else:
                    why.append(f"distant T={tt:.2f}")
            if truth_in_network:
                why.append("TRUTH WAS IN NETWORK mass-OK SMILES but ranker preferred another")
            else:
                why.append("truth NOT in network mass-OK SMILES")
            if truth_in_cfm:
                why.append("truth WAS among CFM mass-OK neighbor SMILES")
            else:
                why.append("truth not among CFM mass-OK neighbor SMILES")
            if csi_rank_truth:
                why.append(f"truth at CSI rank {csi_rank_truth} but not selected by CFM-first arm")
            elif csi_top:
                why.append(f"CSI top IK1={csi_ik} (not truth)")
            if true_fit and not true_fit["ok"]:
                why.append(
                    f"truth mass-awkward (best residual {true_fit['err']:+.4f})"
                )
            reason = "; ".join(why)

        rows.append(
            {
                "spectrum_id": sid,
                "outcome": outcome,
                "true_name": true_name,
                "true_smiles": true_smi,
                "true_ik1": true_ik,
                "true_formula": true_form,
                "seed_mz": mz,
                "ion_mode": ion,
                "pred_smiles": pred_smi,
                "pred_ik1": pik,
                "pred_formula": formula(ma) if ma else "",
                "pred_source": pred_src,
                "pred_name": pred.get("iupac_or_common_name") or "",
                "ik1": ik_ok,
                "exact": exact,
                "formula_match": form_ok,
                "tanimoto": None if tt is None else round(tt, 4),
                "truth_mass_ok": None if not true_fit else true_fit["ok"],
                "truth_best_adduct": None if not true_fit else true_fit["adduct"],
                "truth_mass_err": None if not true_fit else round(true_fit["err"], 5),
                "truth_in_network_mass_ok": truth_in_network,
                "truth_in_cfm_mass_ok": truth_in_cfm,
                "csi_rank_of_truth": csi_rank_truth,
                "csi_top_ik1": csi_ik,
                "n_cfm_transferred_peaks": n_trans,
                "n_network_mass_ok_smiles": len(net_smis),
                "why": reason,
                "rationale": (pred.get("rationale") or "")[:400],
            }
        )

    by = Counter(r["outcome"] for r in rows)
    miss_tags: Counter[str] = Counter()
    for r in rows:
        if r["outcome"] != "MISS":
            continue
        w = r["why"]
        if "SAME FORMULA" in w:
            miss_tags["same_formula_isomer"] += 1
        if "TRUTH WAS IN NETWORK" in w:
            miss_tags["truth_in_network_ranker_wrong"] += 1
        if "truth NOT in network" in w:
            miss_tags["truth_not_in_network"] += 1
        if "truth WAS among CFM" in w:
            miss_tags["truth_in_cfm_not_picked"] += 1
        if "distant T" in w:
            miss_tags["distant_structure"] += 1
        if "related T" in w or "very close T" in w:
            miss_tags["close_but_wrong"] += 1

    def show(title: str, filt, limit: int = 60) -> None:
        subset = [r for r in rows if filt(r)]
        print(f"\n===== {title} (n={len(subset)}) =====")
        for r in subset[:limit]:
            print(f"{r['spectrum_id']}  {r['true_name'][:60]}")
            print(
                f"  true: {r['true_formula']} IK1={r['true_ik1']} "
                f"mz={r['seed_mz']:.4f} {r['ion_mode']}"
            )
            print(
                f"  pred: src={r['pred_source']} IK1={r['pred_ik1']} "
                f"form={r['pred_formula']} T={r['tanimoto']}"
            )
            print(f"  smi_true={r['true_smiles'][:80]}")
            print(f"  smi_pred={(r['pred_smiles'] or '')[:80]}")
            print(f"  net_has_truth={r['truth_in_network_mass_ok']} "
                  f"cfm_has_truth={r['truth_in_cfm_mass_ok']} "
                  f"csi_rank_truth={r['csi_rank_of_truth']} "
                  f"cfm_trans={r['n_cfm_transferred_peaks']}")
            print(f"  why: {r['why']}")
            print()

    print("ARM: CFM-first NO NIST")
    print("OUTCOMES", dict(by))
    print("IK1", sum(1 for r in rows if r["ik1"]), "/40")
    print("EXACT", sum(1 for r in rows if r["exact"]), "/40")
    print("MISS TAGS", dict(miss_tags))

    show("MATCHED — exact SMILES", lambda r: r["outcome"] == "MATCH_EXACT")
    show("MATCHED — IK1 only (not exact SMILES)", lambda r: r["outcome"] == "MATCH_IK1")
    show("MISSED — wrong structure called", lambda r: r["outcome"] == "MISS")
    show("NOT CALLED — abstain", lambda r: r["outcome"] == "ABSTAIN")

    out_json = pack / "CFMFIRST_NO_NIST_UNBLINDED_ANALYSIS.json"
    out_csv = pack / "CFMFIRST_NO_NIST_UNBLINDED_ANALYSIS.csv"
    out_json.write_text(
        json.dumps(
            {
                "arm": "CFM-first NO NIST",
                "pred_dir": str(pred_dir),
                "n": 40,
                "summary": {
                    "ik1": sum(1 for r in rows if r["ik1"]),
                    "exact": sum(1 for r in rows if r["exact"]),
                    "outcomes": dict(by),
                    "miss_tags": dict(miss_tags),
                },
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fields = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("Wrote", out_csv)
    print("Wrote", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
