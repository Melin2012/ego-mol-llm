#!/usr/bin/env python3
"""
Apply product hybrid path + offline in-silico spectrum re-ranking to pure-model
pack predictions (no LLM re-call).

Default backend is **rule** (RDKit, always offline). Use ``--backend cfmid`` when
Docker is running with ``wishartlab/cfmid`` image.

Example
-------
  python scripts/apply_insilico_rerank_to_pack.py \\
    --pack .../Blind_holdout40_ego_msms \\
    --preds-in predictions \\
    --preds-out predictions_opus_insilico_rule \\
    --backend rule
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

from ego_mol_llm.candidates import select_product_annotation
from ego_mol_llm.ego import build_ego
from ego_mol_llm.graphml import load_graphml
from ego_mol_llm.library_search import default_nist_paths, load_library_index
from ego_mol_llm.method_card import MethodCard, resolve_method_polarity
from ego_mol_llm.mgf import build_spectral_context
from ego_mol_llm.predict import refine_with_neighborhood
from ego_mol_llm.spectrum_predict import get_predictor
from ego_mol_llm.validate import ParsedPrediction, parse_model_output, validate_smiles_fields


def pred_from_json(d: dict, mz: float | None, tol: float) -> ParsedPrediction:
    raw = d.get("model_raw") or ""
    if isinstance(raw, str) and raw.strip() and "{" in raw:
        p = parse_model_output(raw, precursor_mz=mz, mass_tol_da=tol)
        if p.smiles:
            return p
    p = ParsedPrediction(
        smiles=(d.get("smiles") or None),
        name=d.get("iupac_or_common_name") or d.get("name"),
        formula=d.get("formula"),
        adduct=d.get("adduct"),
        confidence=float(d["confidence"])
        if isinstance(d.get("confidence"), (int, float))
        else None,
        rationale=d.get("rationale"),
        alternatives=list(d.get("alternatives") or []),
        parse_mode="pack_json",
        source="model",
    )
    return validate_smiles_fields(p, mz, tol)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--preds-in", required=True)
    ap.add_argument("--preds-out", required=True)
    ap.add_argument("--model-label", default="product-insilico")
    ap.add_argument("--backend", default="rule", help="auto|rule|cfmid|iceberg")
    ap.add_argument("--weight", type=float, default=0.35, help="In-silico fusion weight")
    ap.add_argument("--library-index", type=Path, default=None)
    ap.add_argument("--max-neighbors", type=int, default=50)
    ap.add_argument("--mass-tol", type=float, default=0.05)
    ap.add_argument("--no-library", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pack = args.pack.resolve()
    src = pack / args.preds_in
    out = pack / args.preds_out
    out.mkdir(parents=True, exist_ok=True)

    pred_eng = get_predictor(args.backend)
    print(
        f"[info] insilico backend={pred_eng.name} available={pred_eng.available()}",
        flush=True,
    )
    if args.backend in {"cfmid", "auto"} and not pred_eng.available():
        print(
            "[warn] requested backend not available; rule fallback may apply via auto, "
            "or use --backend rule",
            flush=True,
        )

    idx_path = args.library_index
    if idx_path is None:
        _, idx_path = default_nist_paths(_REPO)
    lib = None
    if not args.no_library and Path(idx_path).is_file():
        print(f"[info] library {idx_path}", flush=True)
        lib = load_library_index(idx_path)

    rows = list(
        csv.DictReader((pack / "sample_manifest.csv").open(encoding="utf-8", newline=""))
    )
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    t0 = time.perf_counter()
    n_ok = 0
    src_counts: dict[str, int] = {}
    cos_sum = 0.0
    cos_n = 0

    for i, row in enumerate(rows, 1):
        sid = (row.get("spectrum_id") or "").strip().upper()
        in_p = src / f"{sid}.json"
        if not in_p.is_file():
            print(f"[{i}] skip missing pure {sid}", flush=True)
            continue
        gpath = pack / "graphml" / f"{sid}.graphml"
        pure = json.loads(in_p.read_text(encoding="utf-8-sig"))
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

        mgf = pack / "subgraph_mgfs" / f"{sid}.mgf"
        seed = pack / "seed_mgfs_blind" / f"{sid}.mgf"
        ego.spectral = build_spectral_context(
            seed_id=ego.seed.id,
            seed_mz=ego.seed_mz,
            neighbor_ids=[e.node.id for e in ego.neighbors],
            mgf_paths=[mgf] if mgf.is_file() else [],
            seed_mgf=seed if seed.is_file() else None,
        )
        pol = resolve_method_polarity(
            seed_ion_mode=getattr(ego.spectral, "seed_ion_mode", None),
            method=MethodCard(polarity="both"),
        )
        method = MethodCard(
            chromatography="RP-C18",
            polarity=pol,
            ionization="ESI",
            study_id="ASTRAL_C18_holdout",
        )

        library_hits = []
        if (
            lib is not None
            and ego.spectral
            and ego.spectral.seed
            and ego.spectral.seed.peaks
        ):
            library_hits = lib.search(
                ego.spectral.seed.peaks,
                ego.seed_mz or ego.spectral.seed.pepmass,
                top_k=8,
                precursor_tol_da=max(args.mass_tol, 0.02),
                ion_mode=ego.spectral.seed_ion_mode or pol,
            )

        pred = pred_from_json(pure, ego.seed_mz, args.mass_tol)
        seed_rt = getattr(ego.spectral, "seed_rt", None) if ego.spectral else None
        neighbor_rts = list(ego.spectral.neighbor_rt.values()) if ego.spectral else []

        pred, notes, cands = select_product_annotation(
            ego,
            pred,
            library_hits=library_hits,
            method=method,
            mass_tol_da=args.mass_tol,
            seed_rt=seed_rt,
            neighbor_rts=neighbor_rts,
            use_insilico_rerank=True,
            insilico_backend=args.backend,
            insilico_weight=args.weight,
        )
        if pred.source in {"model"} and getattr(pred, "mass_ok", None) is not True:
            pred, rnotes = refine_with_neighborhood(pred, ego, mass_tol_da=args.mass_tol)
            notes.extend(rnotes)

        src_name = pred.source or "unknown"
        src_counts[src_name] = src_counts.get(src_name, 0) + 1
        best_cos = None
        for c in cands:
            if c.insilico_cosine is not None:
                best_cos = max(best_cos or 0.0, c.insilico_cosine)
        if best_cos is not None:
            cos_sum += best_cos
            cos_n += 1

        out_d = {
            "spectrum_id": sid,
            "smiles": pred.canonical_smiles or pred.smiles or "",
            "iupac_or_common_name": pred.name,
            "formula": pred.formula,
            "adduct": pred.adduct,
            "confidence": pred.confidence,
            "rationale": pred.rationale,
            "alternatives": pred.alternatives,
            "model": args.model_label,
            "blind": True,
            "source": pred.source,
            "mass_ok": pred.mass_ok,
            "mass_error_da": pred.mass_error_da,
            "product_version": "0.3-insilico",
            "insilico_backend": pred_eng.name,
            "insilico_weight": args.weight,
            "best_insilico_cosine": best_cos,
            "rescue_notes": notes,
            "hybrid_top": [c.to_dict() for c in cands[:8]],
        }
        (out / f"{sid}.json").write_text(
            json.dumps(out_d, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        n_ok += 1
        print(
            f"[{i}/{len(rows)}] {sid} → {pred.source} "
            f"insilico_top={best_cos if best_cos is not None else 'n/a'} "
            f"smiles={(out_d['smiles'] or '')[:40]!r}",
            flush=True,
        )

    summary = {
        "n_ok": n_ok,
        "source_counts": src_counts,
        "mean_best_insilico_cosine": (cos_sum / cos_n) if cos_n else None,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "backend": pred_eng.name,
        "weight": args.weight,
        "product_version": "0.3-insilico",
    }
    (out / "_insilico_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[done] {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
