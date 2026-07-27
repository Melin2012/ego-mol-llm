#!/usr/bin/env python3
"""Merge in-session Grok predictions with truth; write metrics + figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test\outputs\grok45_insession"),
    )
    args = ap.parse_args()
    root = args.root
    truth_path = root / "truth_index.csv"
    pred_dir = root / "predictions"
    truth = {}
    with truth_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            truth[row["spectrum_id"]] = row

    rows = []
    for p in sorted(pred_dir.glob("*.json")):
        sid = p.stem.upper()
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            rows.append({"spectrum_id": sid, "ok": False, "error": str(e)})
            continue
        t = truth.get(sid, {})
        rows.append(
            {
                "spectrum_id": sid,
                "ok": True,
                "smiles": d.get("smiles"),
                "name": d.get("iupac_or_common_name") or d.get("name"),
                "formula": d.get("formula"),
                "adduct": d.get("adduct"),
                "confidence": d.get("confidence"),
                "rationale": (d.get("rationale") or "")[:500],
                "true_name": t.get("true_name"),
                "true_smiles": t.get("true_smiles"),
                "true_inchikey": t.get("true_inchikey"),
                "true_formula": t.get("true_formula"),
                "seed_mz": t.get("seed_mz"),
                "model": d.get("model") or "grok-4.5-insession",
            }
        )

    # metrics
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs, inchi
    except Exception:
        Chem = None

    n = 0
    ik = 0
    tani_sum = 0.0
    tani_n = 0
    with_smi = 0
    for r in rows:
        if not r.get("ok"):
            continue
        n += 1
        pred = (r.get("smiles") or "").strip()
        true = (r.get("true_smiles") or "").strip()
        if pred:
            with_smi += 1
        if Chem and pred and true:
            m1, m2 = Chem.MolFromSmiles(pred), Chem.MolFromSmiles(true)
            if m1 and m2:
                try:
                    pik = (inchi.MolToInchiKey(m1) or "").split("-")[0].upper()
                    tik = (r.get("true_inchikey") or "").split("-")[0].upper()
                    if pik and tik and pik == tik:
                        ik += 1
                        r["ik_match"] = True
                    else:
                        r["ik_match"] = False
                except Exception:
                    r["ik_match"] = False
                fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
                t = DataStructs.TanimotoSimilarity(fp1, fp2)
                r["tanimoto"] = t
                tani_sum += t
                tani_n += 1

    metrics = {
        "n_predictions": len(rows),
        "n_ok": n,
        "n_with_smiles": with_smi,
        "inchikey_match": ik,
        "inchikey_match_rate": (ik / n) if n else None,
        "mean_tanimoto": (tani_sum / tani_n) if tani_n else None,
        "tanimoto_n": tani_n,
        "model": "grok-4.5-insession",
    }
    (root / "batch_summary.json").write_text(
        json.dumps({"metrics": metrics, "results": rows}, indent=2), encoding="utf-8"
    )
    fields = [
        "spectrum_id", "ok", "smiles", "name", "formula", "adduct", "confidence",
        "true_name", "true_smiles", "true_inchikey", "true_formula", "seed_mz",
        "ik_match", "tanimoto", "model", "error", "rationale",
    ]
    with (root / "batch_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (root / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {root / 'batch_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
