#!/usr/bin/env python3
"""
Product-style ranking for holdout-40c (neighbor + seed MGF first).

Design (user-agreed):
  - Primary structure evidence: GraphML neighbors + seed MGF (mass/adduct)
  - Optional: SIRIUS CSI, CFM neighbor structures
  - Seed NIST: ONLY if high reverse-match score AND mass-OK (verify/support path).
    Low-score seed NIST is discarded entirely — never forces the ID.
  - Not closed to seed-library IDs only.
  - Extended adducts including multi-water losses (e.g. ABA [M+H-3H2O]+).

Writes predictions_product_neighbor_mgf/ and a scored comparison JSON/CSV.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt

RDLogger.DisableLog("rdApp.*")

PROTON = 1.007276
H2O = 18.010565

# High-score seed NIST only (scraped below this)
SEED_NIST_MIN_SCORE = 0.85


def mol(s: str | None):
    return Chem.MolFromSmiles(s or "")


def can_smi(s: str | None) -> str | None:
    m = mol(s)
    return Chem.MolToSmiles(m) if m else None


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


def tani(a, b) -> float | None:
    ma, mb = mol(a), mol(b)
    if not ma or not mb:
        return None
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def adduct_options(mw: float, ion: str | None) -> list[tuple[str, float]]:
    ion_l = (ion or "").lower()
    if ion_l.startswith("neg"):
        opts = [
            ("[M-H]-", mw - PROTON),
            ("[M-H-H2O]-", mw - PROTON - H2O),
            ("[M-H-2H2O]-", mw - PROTON - 2 * H2O),
            ("[2M-H]-", 2 * mw - PROTON),
            ("[M+Cl]-", mw + 34.968853 - PROTON + 1.007825),
        ]
    else:
        opts = [
            ("[M+H]+", mw + PROTON),
            ("[M+Na]+", mw + 22.989221),
            ("[M+NH4]+", mw + 18.033823),
            ("[M+H-H2O]+", mw + PROTON - H2O),
            ("[M+H-2H2O]+", mw + PROTON - 2 * H2O),
            ("[M+H-3H2O]+", mw + PROTON - 3 * H2O),  # e.g. ABA
            ("[M+H-4H2O]+", mw + PROTON - 4 * H2O),
            ("[2M+H]+", 2 * mw + PROTON),
            ("[2M+Na]+", 2 * mw + 22.989221),
        ]
    return opts


def mass_fit(smi: str, mz: float, ion: str | None, tol: float = 0.05):
    m = mol(smi)
    if not m or not mz:
        return None
    mw = ExactMolWt(m)
    opts = adduct_options(mw, ion)
    best = min(opts, key=lambda x: abs(x[1] - mz))
    err = best[1] - mz
    if abs(err) > tol:
        return None
    return {"adduct": best[0], "err": err, "mw": mw, "formula": formula(m)}


@dataclass
class Cand:
    smiles: str
    score: float
    source: str
    fit: dict
    name: str | None = None
    note: str = ""
    edge_cos: float | None = None
    msms_cos: float | None = None
    csi_conf: float | None = None
    nist_score: float | None = None
    meta: dict = field(default_factory=dict)


def parse_neighbor_lines(user: str) -> list[dict]:
    """Pull neighbor-like rows with SMILES, edge_cos, msms_cos from prompt text."""
    out = []
    for ln in (user or "").splitlines():
        if "SMILES=" not in ln:
            continue
        m = re.search(r"SMILES=(\S+)", ln)
        if not m:
            continue
        smi = can_smi(m.group(1).strip().rstrip("|"))
        if not smi:
            continue
        ec = None
        mc = None
        em = re.search(r"edge_cos=([0-9.]+)", ln)
        mm = re.search(r"msms_cos=([0-9.]+)", ln)
        if em:
            try:
                ec = float(em.group(1))
            except Exception:
                pass
        if mm:
            try:
                mc = float(mm.group(1))
            except Exception:
                pass
        nm = None
        nm_m = re.search(r"name=([^\n|]+)", ln)
        if nm_m:
            nm = nm_m.group(1).strip()[:120]
        out.append({"smiles": smi, "edge_cos": ec, "msms_cos": mc, "name": nm, "line": ln[:160]})
    return out


def collect_candidates(
    *,
    sid: str,
    job: dict,
    pack: Path,
    seed_nist_min: float = SEED_NIST_MIN_SCORE,
) -> list[Cand]:
    mz = float(job.get("seed_mz") or 0)
    ion = job.get("seed_ion_mode") or "positive"
    user = job.get("user_prompt") or ""
    cands: list[Cand] = []

    # --- 1) Neighbors from GraphML/prompt (PRIMARY) ---
    for nb in parse_neighbor_lines(user):
        # Skip pure seed-library block lines that look like NIST catalog without network
        # (still allow MASS-CONSISTENT which is often network-derived)
        fit = mass_fit(nb["smiles"], mz, ion)
        if not fit:
            continue
        ec = nb.get("edge_cos") or 0.0
        mc = nb.get("msms_cos")
        # Prefer dual-cosine; pure edge_cos-only gets lower weight
        if mc is not None:
            base = 0.35 + 0.35 * ec + 0.45 * float(mc)
        else:
            base = 0.30 + 0.40 * ec
        # penalize very low MS/MS cos when available (spurious edges)
        if mc is not None and float(mc) < 0.15:
            base *= 0.55
        cands.append(
            Cand(
                smiles=nb["smiles"],
                score=base,
                source="network_neighbor",
                fit=fit,
                name=nb.get("name"),
                note=nb.get("line") or "",
                edge_cos=ec,
                msms_cos=mc,
            )
        )

    # --- 2) CFM neighbor structures ---
    cfm_p = pack / "cfm_explain" / f"{sid}.json"
    if cfm_p.is_file():
        cfm = json.loads(cfm_p.read_text(encoding="utf-8"))
        n_trans = sum(
            1
            for p in (cfm.get("seed_peak_map") or [])
            if p.get("fragment_smiles") and p.get("source") == "transferred"
        )
        for n in cfm.get("neighbors") or []:
            smi = can_smi(n.get("smiles"))
            if not smi:
                continue
            fit = mass_fit(smi, mz, ion)
            if not fit:
                continue
            n_ann = int(n.get("n_annotated") or 0)
            n_peaks = max(1, min(int(n.get("n_peaks") or 1), 40))
            frac = n_ann / n_peaks
            base = 0.40 + 0.30 * frac + 0.05 * min(n_trans, 15) / 15.0
            cands.append(
                Cand(
                    smiles=smi,
                    score=base,
                    source="cfm_neighbor",
                    fit=fit,
                    name=n.get("name"),
                    note=f"cfm ann={n_ann}/{n_peaks} seed_trans={n_trans}",
                    meta={"n_trans": n_trans},
                )
            )

    # --- 3) SIRIUS CSI (spectral structure, not seed NIST) ---
    sir_p = pack / "sirius_hits" / f"{sid}.json"
    if sir_p.is_file():
        for h in (json.loads(sir_p.read_text(encoding="utf-8-sig")).get("hits") or [])[:8]:
            smi = can_smi(h.get("smiles"))
            if not smi:
                continue
            fit = mass_fit(smi, mz, ion)
            if not fit:
                continue
            conf = h.get("confidence")
            try:
                conf_f = float(conf)
                if conf_f != conf_f or abs(conf_f) == float("inf"):
                    conf_f = 0.2
            except Exception:
                conf_f = 0.2
            base = 0.45 + 0.45 * max(0.0, min(conf_f, 1.0))
            cands.append(
                Cand(
                    smiles=smi,
                    score=base,
                    source="sirius_csi",
                    fit=fit,
                    name=h.get("name"),
                    note=f"CSI conf={conf}",
                    csi_conf=conf_f,
                )
            )

    # --- 4) Seed NIST: HIGH SCORE ONLY (verify/support; low score scrapped) ---
    # Prefer original jobs with seed library hits (v0.2 refresh)
    seed_hits = job.get("library_hits") or []
    # also try jobs from main jobs/ if empty
    high_nist: list[dict] = []
    for h in seed_hits:
        if not isinstance(h, dict):
            continue
        sc = h.get("match_score")
        if sc is None:
            sc = h.get("score")
        try:
            sc_f = float(sc)
        except Exception:
            continue
        if sc_f < seed_nist_min:
            continue  # scrap low-score seed NIST
        smi = can_smi(h.get("smiles"))
        # resolve from inchikey if needed
        if not smi and h.get("inchikey"):
            # use cache if present
            cache_p = pack / "inchikey_smiles_cache.json"
            cache = {}
            if cache_p.is_file():
                try:
                    cache = json.loads(cache_p.read_text(encoding="utf-8"))
                except Exception:
                    cache = {}
            ik = str(h["inchikey"]).split()[0].upper()
            smi = can_smi(cache.get(ik) or "")
            if not smi:
                # try cactus once
                try:
                    import urllib.parse
                    import urllib.request

                    url = (
                        "https://cactus.nci.nih.gov/chemical/structure/InChIKey="
                        + urllib.parse.quote(ik)
                        + "/smiles"
                    )
                    with urllib.request.urlopen(url, timeout=8) as r:
                        raw = r.read().decode("utf-8", errors="replace").strip()
                    smi = can_smi(raw)
                    if smi:
                        cache[ik] = smi
                        cache_p.write_text(json.dumps(cache), encoding="utf-8")
                except Exception:
                    pass
        if not smi:
            # name-based: still record for verify later if we match SMILES by ik later
            high_nist.append({"score": sc_f, "smiles": None, "name": h.get("name"), "inchikey": h.get("inchikey"), "raw": h})
            continue
        fit = mass_fit(smi, mz, ion, tol=0.05)
        if not fit:
            # try looser for weird adducts in metadata
            fit = mass_fit(smi, mz, ion, tol=0.15)
        if not fit:
            continue
        high_nist.append({"score": sc_f, "smiles": smi, "name": h.get("name"), "inchikey": h.get("inchikey")})
        # Soft add as candidate (not exclusive) — spectrum library support for seed
        cands.append(
            Cand(
                smiles=smi,
                score=0.50 + 0.40 * sc_f,  # high score only already filtered
                source="seed_nist_high",
                fit=fit,
                name=h.get("name"),
                note=f"seed NIST high-score only sc={sc_f:.3f} (low scores scrapped)",
                nist_score=sc_f,
            )
        )

    # --- 5) Verify boost: if a neighbor/CSI candidate also has high seed NIST, boost ---
    nist_smiles = {x["smiles"] for x in high_nist if x.get("smiles")}
    nist_ik1 = set()
    for x in high_nist:
        if x.get("inchikey"):
            nist_ik1.add(str(x["inchikey"]).split("-")[0].upper())
        if x.get("smiles"):
            nist_ik1.add(ik1(x["smiles"]))

    for c in cands:
        if c.smiles in nist_smiles or ik1(c.smiles) in nist_ik1:
            c.score += 0.12
            c.note = (c.note + " | verified by high-score seed NIST").strip(" |")
            c.meta["nist_verify"] = True

    # Dedup by canonical SMILES — keep best score
    best: dict[str, Cand] = {}
    for c in cands:
        prev = best.get(c.smiles)
        if prev is None or c.score > prev.score:
            best[c.smiles] = c
    ranked = sorted(best.values(), key=lambda x: -x.score)
    return ranked


def main() -> int:
    pack = Path(r"C:\Users\AlexeyMelnik\Desktop\Blind_holdout40c_ego_msms")
    # Use original jobs with seed library hits present (for high-score NIST path)
    jobs_dir = pack / "jobs"
    out_dir = pack / "predictions_product_neighbor_mgf"
    out_dir.mkdir(exist_ok=True)

    truth: dict[str, dict] = {}
    with (pack.parent / "Blind_holdout40c_ego_msms_SEALED_truth" / "truth_index.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        for r in csv.DictReader(f):
            truth[(r.get("spectrum_id") or "").upper()] = r

    rows_out = []
    src_c = Counter()
    for row in csv.DictReader((pack / "sample_manifest.csv").open(encoding="utf-8", newline="")):
        sid = row["spectrum_id"].strip().upper()
        job_p = jobs_dir / f"{sid}.json"
        job = json.loads(job_p.read_text(encoding="utf-8"))
        ranked = collect_candidates(sid=sid, job=job, pack=pack)
        if not ranked:
            pred = {
                "smiles": "",
                "source": "abstain",
                "confidence": 0.1,
                "formula": None,
                "adduct": None,
                "name": None,
                "rationale": (
                    "Neighbor+MGF product: no mass-OK candidate from neighbors/CSI/CFM; "
                    "no high-score seed NIST either."
                ),
                "alternatives": [],
                "n_cands": 0,
            }
        else:
            top = ranked[0]
            alts = [
                {
                    "smiles": c.smiles,
                    "confidence": round(min(0.9, c.score * 0.7), 3),
                    "note": f"{c.source} score={c.score:.3f} {c.note[:80]}",
                }
                for c in ranked[1:5]
            ]
            pred = {
                "smiles": top.smiles,
                "source": top.source,
                "confidence": round(min(0.95, max(0.35, top.score)), 3),
                "formula": top.fit.get("formula"),
                "adduct": top.fit.get("adduct"),
                "name": top.name,
                "rationale": (
                    f"Neighbor+MGF product (seed NIST only if score≥{SEED_NIST_MIN_SCORE}). "
                    f"m/z={job.get('seed_mz')} {job.get('seed_ion_mode')}. "
                    f"{top.fit.get('adduct')} Δ={top.fit.get('err'):+.5f}. "
                    f"source={top.source} score={top.score:.3f}. {top.note[:200]}"
                ),
                "alternatives": alts,
                "n_cands": len(ranked),
                "top5": [
                    {"smiles": c.smiles, "source": c.source, "score": round(c.score, 3)}
                    for c in ranked[:5]
                ],
            }
        src_c[pred["source"]] += 1
        d = {
            "spectrum_id": sid,
            "smiles": pred.get("smiles") or "",
            "iupac_or_common_name": pred.get("name"),
            "formula": pred.get("formula"),
            "adduct": pred.get("adduct"),
            "confidence": pred.get("confidence"),
            "rationale": pred.get("rationale"),
            "alternatives": pred.get("alternatives") or [],
            "model": "product-neighbor-mgf-v1",
            "blind": True,
            "source": pred.get("source"),
            "product_version": "0.5-neighbor-mgf",
            "seed_nist_policy": f"high_score_only_min_{SEED_NIST_MIN_SCORE}",
            "n_cands": pred.get("n_cands"),
            "top5": pred.get("top5"),
        }
        (out_dir / f"{sid}.json").write_text(
            json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # score row
        t = truth[sid]
        true_smi = t.get("true_smiles") or ""
        ma, mb = mol(pred.get("smiles")), mol(true_smi)
        pik, tik = ik1(ma), (t.get("true_inchikey") or "").split("-")[0].upper() or ik1(mb)
        ik_ok = bool(pik and tik and pik == tik)
        exact = bool(ma and mb and Chem.MolToSmiles(ma) == Chem.MolToSmiles(mb))
        form_ok = bool(ma and mb and formula(ma) and formula(ma) == formula(mb))
        tt = tani(pred.get("smiles"), true_smi)
        rows_out.append(
            {
                "spectrum_id": sid,
                "true_name": t.get("true_name"),
                "true_smiles": true_smi,
                "true_ik1": tik,
                "pred_smiles": pred.get("smiles") or "",
                "pred_ik1": pik,
                "pred_source": pred.get("source"),
                "ik1": ik_ok,
                "exact": exact,
                "formula_match": form_ok,
                "tanimoto": None if tt is None else round(tt, 4),
                "tani_ge_0.7": bool(tt is not None and tt >= 0.7),
                "tani_ge_0.85": bool(tt is not None and tt >= 0.85),
                "n_cands": pred.get("n_cands"),
                "outcome": (
                    "MATCH_EXACT"
                    if exact
                    else "MATCH_IK1"
                    if ik_ok
                    else "ABSTAIN"
                    if not pred.get("smiles")
                    else "MISS"
                ),
            }
        )
        print(
            f"[{len(rows_out)}/40] {sid} {pred.get('source')} "
            f"ik1={ik_ok} {(pred.get('smiles') or '')[:40]}"
        )

    n = len(rows_out)
    ik = sum(1 for r in rows_out if r["ik1"])
    ex = sum(1 for r in rows_out if r["exact"])
    form_n = sum(1 for r in rows_out if r["formula_match"])
    t07 = sum(1 for r in rows_out if r["tani_ge_0.7"])
    t085 = sum(1 for r in rows_out if r["tani_ge_0.85"])
    n_pred = sum(1 for r in rows_out if r["pred_smiles"])
    summary = {
        "arm": "product_neighbor_mgf",
        "policy": {
            "primary": "network neighbors + seed MGF mass gate (extended adducts incl -3/-4 H2O)",
            "csi": True,
            "cfm_neighbors": True,
            "seed_nist": f"only if match_score>={SEED_NIST_MIN_SCORE} and mass-OK; else scrap",
            "seed_nist_role": "optional candidate + verify boost if also proposed by network/CSI",
            "closed_set_seed_library_only": False,
        },
        "n": n,
        "n_predicted": n_pred,
        "n_abstain": n - n_pred,
        "ik1": ik,
        "ik1_rate": ik / n,
        "ik1_among_pred": ik / n_pred if n_pred else None,
        "exact": ex,
        "exact_rate": ex / n,
        "formula_match": form_n,
        "formula_match_rate": form_n / n,
        "tani_ge_0.7": t07,
        "tani_ge_0.7_rate": t07 / n,
        "tani_ge_0.85": t085,
        "tani_ge_0.85_rate": t085 / n,
        "sources": dict(src_c),
    }
    (out_dir / "_batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Compare to key prior arms
    def score_dir(dname: str) -> dict:
        d = pack / dname
        if not d.is_dir():
            return {"missing": True}
        n = ik = ex = form_n = t07 = n_pred = 0
        for sid, t in truth.items():
            p = d / f"{sid}.json"
            if not p.is_file():
                continue
            pred = json.loads(p.read_text(encoding="utf-8-sig"))
            smi = (pred.get("smiles") or "").strip()
            n += 1
            if smi:
                n_pred += 1
            ma, mb = mol(smi), mol(t.get("true_smiles"))
            pik = ik1(ma)
            tik = (t.get("true_inchikey") or "").split("-")[0].upper() or ik1(mb)
            if pik and tik and pik == tik:
                ik += 1
            if ma and mb and Chem.MolToSmiles(ma) == Chem.MolToSmiles(mb):
                ex += 1
            if ma and mb and formula(ma) and formula(ma) == formula(mb):
                form_n += 1
            tt = tani(smi, t.get("true_smiles"))
            if tt is not None and tt >= 0.7:
                t07 += 1
        return {
            "n": n,
            "ik1": ik,
            "ik1_rate": ik / n if n else None,
            "exact": ex,
            "formula": form_n,
            "tani_ge_0.7": t07,
            "n_pred": n_pred,
        }

    compare = {
        "product_neighbor_mgf": summary,
        "baselines": {
            "cfmfirst_no_nist": score_dir("predictions_cfmfirst_no_nist"),
            "siriusfirst_no_nist": score_dir("predictions_siriusfirst_no_nist"),
            "network_no_nist": score_dir("predictions_network_no_nist"),
            "mgf_only_seed_nist": score_dir("predictions_mgf_only"),
            "siriusfirst_nist_neighbors": score_dir("predictions_siriusfirst_nist_neighbors"),
        },
        "rows": rows_out,
    }
    (pack / "PRODUCT_NEIGHBOR_MGF_RESULTS.json").write_text(
        json.dumps(compare, indent=2), encoding="utf-8"
    )
    with (pack / "PRODUCT_NEIGHBOR_MGF_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    print("\n=== PRODUCT NEIGHBOR+MGF SUMMARY ===")
    print(json.dumps({k: summary[k] for k in summary if k != "policy"}, indent=2))
    print("\n=== vs baselines (IK1 /40) ===")
    for k, v in compare["baselines"].items():
        if v.get("missing"):
            print(k, "MISSING")
        else:
            print(f"  {k}: IK1 {v['ik1']}/{v['n']} ({100*v['ik1_rate']:.1f}%) T>=0.7 {v['tani_ge_0.7']}/{v['n']}")
    print(f"  product_neighbor_mgf: IK1 {ik}/{n} ({100*ik/n:.1f}%) T>=0.7 {t07}/{n} sources={dict(src_c)}")
    print("Wrote", out_dir)
    print("Wrote PRODUCT_NEIGHBOR_MGF_RESULTS.json/csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
