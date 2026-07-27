#!/usr/bin/env python3
"""
Product-path post-process: take pure-model pack predictions and apply
ego-mol-llm neighbor rescue (annotation propagation).

By design: network library SMILES + mass/MS-MS gates may replace or keep
the model structure. Writes source=model|neighbor_rescue|abstain.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ego_mol_llm.ego import build_ego
from ego_mol_llm.graphml import load_graphml
from ego_mol_llm.mgf import build_spectral_context
from ego_mol_llm.predict import refine_with_neighborhood
from ego_mol_llm.validate import ParsedPrediction, parse_model_output, validate_smiles_fields


def _pred_from_json(d: dict, precursor_mz: float | None, mass_tol: float) -> ParsedPrediction:
    """Build a ParsedPrediction from a Fable/Grok-style JSON prediction."""
    # Prefer re-parse of raw text if present; else fields
    raw = d.get("model_raw") or d.get("raw") or ""
    if isinstance(raw, str) and raw.strip() and "{" in raw:
        p = parse_model_output(raw, precursor_mz=precursor_mz, mass_tol_da=mass_tol)
        if p.smiles or p.canonical_smiles:
            return p

    smi = (d.get("smiles") or "").strip() or None
    p = ParsedPrediction(
        smiles=smi,
        name=d.get("iupac_or_common_name") or d.get("name"),
        formula=d.get("formula"),
        adduct=d.get("adduct"),
        confidence=float(d["confidence"]) if isinstance(d.get("confidence"), (int, float)) else None,
        rationale=d.get("rationale"),
        alternatives=list(d.get("alternatives") or []),
        raw_text=json.dumps(d, ensure_ascii=False),
        parse_mode="pack_json",
        source="model",
    )
    return validate_smiles_fields(p, precursor_mz, mass_tol)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--preds-in", type=str, required=True, help="subdir under pack with pure preds")
    ap.add_argument("--preds-out", type=str, required=True, help="subdir for product-path outputs")
    ap.add_argument("--max-neighbors", type=int, default=50)
    ap.add_argument("--mass-tol", type=float, default=0.05)
    ap.add_argument("--model-label", default="product")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pack = args.pack.resolve()
    src_dir = pack / args.preds_in
    out_dir = pack / args.preds_out
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pack / "sample_manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    if args.limit:
        rows = rows[: args.limit]

    print(
        f"[info] pack={pack} in={src_dir.name} out={out_dir.name} n={len(rows)}",
        flush=True,
    )

    n_ok = n_fail = n_skip = 0
    src_counts: dict[str, int] = {}
    t0 = time.perf_counter()

    for i, row in enumerate(rows, 1):
        sid = (row.get("spectrum_id") or "").strip().upper()
        if not sid:
            continue
        in_path = src_dir / f"{sid}.json"
        out_path = out_dir / f"{sid}.json"
        if not in_path.is_file():
            print(f"[{i}/{len(rows)}] SKIP missing pure pred {sid}", flush=True)
            n_skip += 1
            continue

        gpath = pack / "graphml" / f"{sid}.graphml"
        mgf = pack / "subgraph_mgfs" / f"{sid}.mgf"
        seed = pack / "seed_mgfs_blind" / f"{sid}.mgf"
        if not gpath.is_file():
            print(f"[{i}/{len(rows)}] FAIL no graphml {sid}", flush=True)
            n_fail += 1
            continue

        try:
            pure = json.loads(in_path.read_text(encoding="utf-8-sig"))
            net = load_graphml(gpath)
            ego = build_ego(
                net,
                seed_id="0",
                hide_seed_name=True,
                max_neighbors=args.max_neighbors,
                include_two_hop=True,
            )
            ego.seed.name = None
            ego.seed.smiles = None
            ego.hide_seed_name = True

            mgf_paths = [mgf] if mgf.is_file() else []
            if mgf_paths or seed.is_file():
                ego.spectral = build_spectral_context(
                    seed_id=ego.seed.id,
                    seed_mz=ego.seed_mz,
                    neighbor_ids=[e.node.id for e in ego.neighbors],
                    mgf_paths=mgf_paths,
                    seed_mgf=seed if seed.is_file() else None,
                )

            pred = _pred_from_json(pure, ego.seed_mz, args.mass_tol)
            pred, notes = refine_with_neighborhood(pred, ego, mass_tol_da=args.mass_tol)

            payload = {
                "spectrum_id": sid,
                "smiles": pred.canonical_smiles or pred.smiles,
                "iupac_or_common_name": pred.name,
                "formula": pred.formula,
                "adduct": pred.adduct or pred.matched_adduct,
                "confidence": pred.confidence,
                "rationale": pred.rationale,
                "alternatives": pred.alternatives,
                "model": args.model_label,
                "blind": True,
                "source": pred.source,
                "mass_ok": pred.mass_ok,
                "mass_error_da": pred.mass_error_da,
                "matched_adduct": pred.matched_adduct,
                "parse_mode": pred.parse_mode,
                "rescue_notes": notes,
                "msms_used": bool(ego.spectral and ego.spectral.seed),
                "pure_model_smiles": pure.get("smiles"),
                "pure_model_name": pure.get("iupac_or_common_name") or pure.get("name"),
                "product_path": True,
                "use_neighbor_rescue": True,
            }
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            src_counts[pred.source] = src_counts.get(pred.source, 0) + 1
            n_ok += 1
            print(
                f"[{i}/{len(rows)}] {sid} pure→{pred.source} "
                f"smiles={(payload.get('smiles') or '')[:36]!r}",
                flush=True,
            )
        except Exception as e:
            n_fail += 1
            out_path.write_text(
                json.dumps({"spectrum_id": sid, "error": f"{type(e).__name__}: {e}"}, indent=2),
                encoding="utf-8",
            )
            print(f"[{i}/{len(rows)}] FAIL {sid}: {type(e).__name__}: {e}", flush=True)

    summary = {
        "pack": str(pack),
        "preds_in": args.preds_in,
        "preds_out": args.preds_out,
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_fail": n_fail,
        "source_counts": src_counts,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "protocol": "hide_seed + neighbor_rescue product path (annotation propagation)",
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[done]", json.dumps(summary), flush=True)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
