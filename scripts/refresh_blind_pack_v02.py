#!/usr/bin/env python3
"""
Refresh an existing blind pack's jobs/prompts with ego-mol-llm v0.2 evidence:
  - MS/MS + RT/metadata
  - NIST library reverse search
  - method card
Does NOT change which spectrum_ids are in the pack.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ego_mol_llm.ego import build_ego
from ego_mol_llm.graphml import load_graphml
from ego_mol_llm.library_search import default_nist_paths, load_library_index
from ego_mol_llm.method_card import MethodCard
from ego_mol_llm.mgf import build_spectral_context
from ego_mol_llm.prompts import SYSTEM_PROMPT, build_messages

FABLE_EXTRA = """
TASK RULES (annotation propagation):
- Center spectrum is unlabeled; do not invent a seed library name.
- Use network neighbor names/SMILES, NIST/library hits, RT/method, and MS/MS.
- Prefer mass-consistent structures (monomer and multimer adducts).
- Prefer high dual cosine (edge + MS/MS) and high library match scores.
- Output one JSON object only with smiles, iupac_or_common_name, formula, adduct,
  confidence, rationale, alternatives.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--max-neighbors", type=int, default=50)
    ap.add_argument("--library-index", type=Path, default=None)
    ap.add_argument("--library-top-k", type=int, default=8)
    ap.add_argument("--no-library", action="store_true")
    args = ap.parse_args()

    pack = args.pack.resolve()
    manifest = pack / "sample_manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))

    idx_path = args.library_index
    if idx_path is None:
        _, idx_path = default_nist_paths(_REPO)
    lib = None
    if not args.no_library and Path(idx_path).is_file():
        print(f"[info] loading library index {idx_path}", flush=True)
        lib = load_library_index(idx_path)
        print(f"[info] library n_records={lib.n_records}", flush=True)
    else:
        print(f"[warn] no library index at {idx_path}", flush=True)

    # Study-level defaults only; polarity is overridden per seed from MGF IONMODE.
    # Holdout mixes ESI+ and ESI- — never force polarity="positive" for all rows.
    method_template = MethodCard(
        chromatography="RP-C18",
        polarity="both",
        ionization="ESI",
        gradient="aqueous_to_organic",
        study_id="ASTRAL_C18_holdout",
        notes="polarity resolved per seed from MGF IONMODE/CHARGE",
    )

    jobs_out = pack / "jobs"
    prompts_out = pack / "prompts"
    jobs_out.mkdir(exist_ok=True)
    prompts_out.mkdir(exist_ok=True)

    from ego_mol_llm.method_card import resolve_method_polarity

    n_ok = 0
    n_pos = n_neg = n_unk = 0
    for i, row in enumerate(rows, 1):
        sid = (row.get("spectrum_id") or "").strip().upper()
        gpath = pack / "graphml" / f"{sid}.graphml"
        mgf = pack / "subgraph_mgfs" / f"{sid}.mgf"
        seed = pack / "seed_mgfs_blind" / f"{sid}.mgf"
        if not gpath.is_file():
            print(f"[{i}] skip missing {sid}", flush=True)
            continue
        try:
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

            ego.spectral = build_spectral_context(
                seed_id=ego.seed.id,
                seed_mz=ego.seed_mz,
                neighbor_ids=[e.node.id for e in ego.neighbors],
                mgf_paths=[mgf] if mgf.is_file() else [],
                seed_mgf=seed if seed.is_file() else None,
            )
            seed_ion = getattr(ego.spectral, "seed_ion_mode", None) if ego.spectral else None
            pol = resolve_method_polarity(
                seed_ion_mode=seed_ion,
                method=method_template,
                default="both",
            )
            method = MethodCard(
                chromatography=method_template.chromatography,
                polarity=pol,
                ionization=method_template.ionization,
                gradient=method_template.gradient,
                study_id=method_template.study_id,
                notes=method_template.notes,
            )
            if pol == "positive":
                n_pos += 1
            elif pol == "negative":
                n_neg += 1
            else:
                n_unk += 1

            lib_hits = []
            if (
                lib is not None
                and ego.spectral
                and ego.spectral.seed
                and ego.spectral.seed.peaks
            ):
                search_mode = seed_ion if seed_ion in {"positive", "negative"} else (
                    pol if pol in {"positive", "negative"} else None
                )
                hits = lib.search(
                    ego.spectral.seed.peaks,
                    ego.seed_mz or ego.spectral.seed.pepmass,
                    top_k=args.library_top_k,
                    precursor_tol_da=0.02,
                    ion_mode=search_mode,
                )
                lib_hits = [h.to_dict() for h in hits]
            ego.meta["library_hits"] = lib_hits
            ego.meta["method_card"] = method.to_dict()
            ego.meta["seed_ion_mode"] = seed_ion
            if lib is not None:
                ego.meta["library_n_records"] = lib.n_records

            messages = build_messages(ego, extra_instructions=FABLE_EXTRA)
            system = next(
                (m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT
            )
            user = next((m["content"] for m in messages if m["role"] == "user"), "")

            job = {
                "spectrum_id": sid,
                "seed_mz": ego.seed_mz,
                "n_neighbors": len(ego.neighbors),
                "msms_used": bool(ego.spectral and ego.spectral.seed),
                "seed_rt": getattr(ego.spectral, "seed_rt", None) if ego.spectral else None,
                "seed_ion_mode": seed_ion,
                "max_neighbors": args.max_neighbors,
                "protocol": "hide_seed_neighbors_kept",
                "product_version": "0.2.1",
                "n_library_hits": len(lib_hits),
                "system_prompt": system,
                "user_prompt": user,
                "blind": True,
                "method_card": method.to_dict(),
                "library_hits": lib_hits,
                "files": {
                    "graphml": f"graphml/{sid}.graphml",
                    "network_mgf": f"subgraph_mgfs/{sid}.mgf",
                    "seed_mgf": f"seed_mgfs_blind/{sid}.mgf",
                },
            }
            (jobs_out / f"{sid}.json").write_text(
                json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            prompt_txt = (
                "### SYSTEM\n"
                + system
                + "\n\n### USER\n"
                + user
                + "\n\n### RESPONSE FORMAT\n"
                "Reply with one JSON object only.\n"
            )
            (prompts_out / f"{sid}.txt").write_text(prompt_txt, encoding="utf-8")
            n_ok += 1
            print(
                f"[{i}/{len(rows)}] {sid} pol={pol} ion={seed_ion} "
                f"lib_hits={len(lib_hits)} rt={job.get('seed_rt')}",
                flush=True,
            )
        except Exception as e:
            print(f"[{i}] FAIL {sid}: {type(e).__name__}: {e}", flush=True)

    meta_path = pack / "package_meta.json"
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "product_version": "0.2.1",
            "refreshed_v02": True,
            "polarity_resolved_per_seed": True,
            "n_polarity_positive": n_pos,
            "n_polarity_negative": n_neg,
            "n_polarity_unknown": n_unk,
            "library_index": str(idx_path) if idx_path else None,
            "n_refreshed": n_ok,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[done] refreshed {n_ok} jobs/prompts "
        f"(pol+={n_pos} pol-={n_neg} pol?={n_unk})",
        flush=True,
    )
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
