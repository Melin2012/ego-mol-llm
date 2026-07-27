#!/usr/bin/env python3
"""
Run ego-mol-llm over a blind pack (GraphML + optional MGF) with an Ollama model.

Writes predictions/<spectrum_id>.json in the same schema as Fable packs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ego_mol_llm.predict import predict_from_graphml


def main() -> int:
    ap = argparse.ArgumentParser(description="Blind pack runner (Ollama / OpenAI-compatible)")
    ap.add_argument("--pack", type=Path, required=True, help="Blind pack root")
    ap.add_argument("--out-subdir", default="predictions_chemdfm_r", help="Under pack/")
    ap.add_argument("--model", default="chemdfm-r-14b")
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "ollama"))
    ap.add_argument("--max-neighbors", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--mass-tol", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--no-rescue",
        action="store_true",
        help="Pure model (use_neighbor_rescue=False). Default: rescue ON (product path).",
    )
    ap.add_argument("--no-library", action="store_true", help="Disable NIST/library search")
    ap.add_argument("--no-hybrid", action="store_true", help="Disable hybrid ranker")
    ap.add_argument("--library-index", type=Path, default=None)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if prediction JSON already exists",
    )
    args = ap.parse_args()
    if args.force:
        args.skip_existing = False

    pack = args.pack.resolve()
    manifest = pack / "sample_manifest.csv"
    if not manifest.is_file():
        print(f"[error] missing {manifest}", flush=True)
        return 1

    out_dir = pack / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    print(
        f"[info] pack={pack} n={len(rows)} model={args.model} "
        f"rescue={'OFF' if args.no_rescue else 'ON'} "
        f"library={'OFF' if args.no_library else 'ON'} "
        f"hybrid={'OFF' if args.no_hybrid else 'ON'} out={out_dir}",
        flush=True,
    )

    os.environ.setdefault("OPENAI_BASE_URL", args.base_url)
    os.environ.setdefault("OPENAI_API_KEY", args.api_key)
    os.environ.setdefault("EGO_MOL_NUM_CTX", "16384")

    n_ok = 0
    n_skip = 0
    n_fail = 0
    t0 = time.perf_counter()

    for i, row in enumerate(rows, 1):
        sid = (row.get("spectrum_id") or "").strip().upper()
        if not sid:
            continue
        pred_path = out_dir / f"{sid}.json"
        if args.skip_existing and pred_path.is_file() and pred_path.stat().st_size > 20:
            n_skip += 1
            print(f"[{i}/{len(rows)}] skip {sid}", flush=True)
            continue

        gpath = pack / "graphml" / f"{sid}.graphml"
        mgf = pack / "subgraph_mgfs" / f"{sid}.mgf"
        seed = pack / "seed_mgfs_blind" / f"{sid}.mgf"
        if not gpath.is_file():
            print(f"[{i}/{len(rows)}] FAIL missing graphml {sid}", flush=True)
            n_fail += 1
            continue

        print(f"[{i}/{len(rows)}] {sid}", flush=True)
        t1 = time.perf_counter()
        try:
            from ego_mol_llm.method_card import MethodCard

            result = predict_from_graphml(
                graphml_path=gpath,
                backend=args.backend,
                model=args.model,
                seed_id="0",
                hide_seed_name=True,
                max_neighbors=args.max_neighbors,
                include_two_hop=True,
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                mass_tol_da=args.mass_tol,
                use_neighbor_rescue=not args.no_rescue,
                mgf_paths=[mgf] if mgf.is_file() else None,
                seed_mgf=seed if seed.is_file() else None,
                use_library_search=not args.no_library,
                library_index_path=args.library_index,
                use_hybrid_ranker=not args.no_hybrid,
                method_card=MethodCard(
                    chromatography="RP-C18",
                    polarity="positive",
                    ionization="ESI",
                    study_id="ASTRAL_C18_holdout",
                ).to_dict(),
            )
            d = result.to_dict()
            payload = {
                "spectrum_id": sid,
                "smiles": d.get("smiles"),
                "iupac_or_common_name": d.get("name"),
                "formula": d.get("formula"),
                "adduct": d.get("adduct") or d.get("matched_adduct"),
                "confidence": d.get("confidence"),
                "rationale": d.get("rationale"),
                "alternatives": d.get("alternatives") or [],
                "model": args.model,
                "blind": True,
                "source": d.get("source"),
                "mass_ok": d.get("mass_ok"),
                "mass_error_da": d.get("mass_error_da"),
                "parse_mode": d.get("parse_mode"),
                "rescue_notes": d.get("rescue_notes"),
                "msms_used": d.get("msms_used"),
                "backend": d.get("backend"),
                "product_version": d.get("product_version"),
                "library_hits": d.get("library_hits"),
                "hybrid_candidates": (d.get("hybrid_candidates") or [])[:8],
                "method_card": d.get("method_card"),
                "elapsed_s": round(time.perf_counter() - t1, 2),
            }
            pred_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            # raw for audit
            (out_dir / f"{sid}.model_raw.txt").write_text(
                result.model_raw or "", encoding="utf-8"
            )
            n_ok += 1
            print(
                f"  ok source={payload.get('source')} conf={payload.get('confidence')} "
                f"smiles={(payload.get('smiles') or '')[:40]!r} t={payload['elapsed_s']}s",
                flush=True,
            )
        except Exception as e:
            n_fail += 1
            err = {
                "spectrum_id": sid,
                "error": f"{type(e).__name__}: {e}",
                "model": args.model,
                "blind": True,
            }
            pred_path.write_text(json.dumps(err, indent=2), encoding="utf-8")
            print(f"  FAIL {type(e).__name__}: {e}", flush=True)

    elapsed = time.perf_counter() - t0
    summary = {
        "pack": str(pack),
        "out": str(out_dir),
        "model": args.model,
        "rescue": not args.no_rescue,
        "n_rows": len(rows),
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_fail": n_fail,
        "elapsed_s": round(elapsed, 1),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[done]", json.dumps(summary), flush=True)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
