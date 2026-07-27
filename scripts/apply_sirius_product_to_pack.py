#!/usr/bin/env python3
"""
Apply product hybrid path with precomputed SIRIUS/CSI hits + NIST + neighbors
to pure-model pack predictions (no LLM re-call).

Expects:
  pack/sirius_hits/<SPECTRUMID>.json   from scripts/run_sirius_on_pack.py
  pack/<preds-in>/<SPECTRUMID>.json    pure model preds
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
from ego_mol_llm.sirius import SiriusHit
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


def load_sirius_hits(path: Path) -> list[SiriusHit]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    hits = data.get("hits") or data if isinstance(data, list) else data.get("hits") or []
    out: list[SiriusHit] = []
    fields = set(SiriusHit.__dataclass_fields__)  # type: ignore[attr-defined]
    for h in hits:
        if not isinstance(h, dict):
            continue
        out.append(SiriusHit(**{k: v for k, v in h.items() if k in fields}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--preds-in", required=True)
    ap.add_argument("--preds-out", required=True)
    ap.add_argument("--model-label", default="product-sirius-v03")
    ap.add_argument("--library-index", type=Path, default=None)
    ap.add_argument("--max-neighbors", type=int, default=50)
    ap.add_argument("--mass-tol", type=float, default=0.05)
    ap.add_argument("--no-library", action="store_true")
    ap.add_argument("--no-sirius", action="store_true")
    args = ap.parse_args()

    pack = args.pack.resolve()
    src = pack / args.preds_in
    out = pack / args.preds_out
    out.mkdir(parents=True, exist_ok=True)
    sirius_dir = pack / "sirius_hits"

    idx_path = args.library_index
    if idx_path is None:
        _, idx_path = default_nist_paths(_REPO)
    lib = None
    if not args.no_library and Path(idx_path).is_file():
        print(f"[info] library {idx_path}", flush=True)
        lib = load_library_index(idx_path)
        print(f"[info] n_records={lib.n_records}", flush=True)

    rows = list(
        csv.DictReader((pack / "sample_manifest.csv").open(encoding="utf-8", newline=""))
    )
    t0 = time.perf_counter()
    n_ok = 0
    src_counts: dict[str, int] = {}

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

        sirius_hits = []
        if not args.no_sirius:
            sirius_hits = load_sirius_hits(sirius_dir / f"{sid}.json")

        pred = pred_from_json(pure, ego.seed_mz, args.mass_tol)
        seed_rt = getattr(ego.spectral, "seed_rt", None) if ego.spectral else None
        neighbor_rts = (
            list(ego.spectral.neighbor_rt.values()) if ego.spectral else []
        )

        pred, notes, cands = select_product_annotation(
            ego,
            pred,
            library_hits=library_hits,
            sirius_hits=sirius_hits,
            method=method,
            mass_tol_da=args.mass_tol,
            seed_rt=seed_rt,
            neighbor_rts=neighbor_rts,
        )
        if pred.source in {"model"} and getattr(pred, "mass_ok", None) is not True:
            pred, rnotes = refine_with_neighborhood(pred, ego, mass_tol_da=args.mass_tol)
            notes.extend(rnotes)

        src_name = pred.source or "unknown"
        src_counts[src_name] = src_counts.get(src_name, 0) + 1

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
            "product_version": "0.3",
            "rescue_notes": notes,
            "n_sirius_hits": len(sirius_hits),
            "n_library_hits": len(library_hits),
            "hybrid_top": [c.to_dict() for c in cands[:5]],
        }
        (out / f"{sid}.json").write_text(
            json.dumps(out_d, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        n_ok += 1
        smi = (out_d["smiles"] or "")[:48]
        print(
            f"[{i}/{len(rows)}] {sid} → {pred.source} sirius={len(sirius_hits)} "
            f"smiles={smi!r}",
            flush=True,
        )

    summary = {
        "n_ok": n_ok,
        "source_counts": src_counts,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "product_version": "0.3",
        "sirius_dir": str(sirius_dir),
    }
    (out / "_product_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[done] {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
