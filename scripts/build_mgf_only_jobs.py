#!/usr/bin/env python3
"""
Build MGF-only (no GraphML / no ego-network) jobs+prompts for a blind pack.

Uses seed MGF peaks + optional NIST library + method card + offline MS/MS explain.
Does NOT load neighbor annotations from GraphML.

Writes:
  pack/jobs_mgf_only/<id>.json
  pack/prompts_mgf_only/<id>.txt
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

from ego_mol_llm.ego import EgoContext
from ego_mol_llm.graphml import Node
from ego_mol_llm.library_search import default_nist_paths, load_library_index
from ego_mol_llm.method_card import MethodCard, resolve_method_polarity
from ego_mol_llm.mgf import build_spectral_context, parse_mgf
from ego_mol_llm.prompts import SYSTEM_PROMPT, build_messages

SYSTEM_MGF_ONLY = """You are an expert mass spectrometry and natural-product chemist.
You assign structure (SMILES) for an UNKNOWN precursor from **MS/MS only**
(no molecular network / no neighbor annotations).

Evidence you may use:
- experimental precursor m/z and MS/MS peaks
- optional spectral library (NIST) reverse search
- retention time / method card
- offline MS/MS explanation (losses, diagnostics)
- optional SIRIUS/CSI ranks if provided

Critical rules:
1. Do NOT invent peaks. MASS FIRST: SMILES must fit precursor m/z under a common adduct
   (including multimers when appropriate).
2. Prefer high-scoring library hits that are mass-consistent and polarity-matched.
3. Use diagnostic fragments and labeled losses when present.
4. Output one JSON object:
{
  "smiles": "<canonical SMILES of neutral monomer>",
  "iupac_or_common_name": "<string or null>",
  "formula": "<Hill formula or null>",
  "adduct": "<e.g. [M+H]+ or [M-H]->",
  "confidence": <float 0-1>,
  "rationale": "<2-5 sentences>",
  "alternatives": [{"smiles": "...", "confidence": 0.0, "note": "..."}]
}
"""

EXTRA = """
TASK RULES (MGF-only / no ego network):
- No neighbor GraphML annotations are available — do not invent network evidence.
- Use MS/MS + NIST/library + RT/method (+ optional SIRIUS if present).
- Prefer mass-consistent adducts for the reported precursor m/z.
- Output one JSON object only.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--library-index", type=Path, default=None)
    ap.add_argument("--no-library", action="store_true")
    ap.add_argument("--library-top-k", type=int, default=8)
    args = ap.parse_args()

    pack = args.pack.resolve()
    rows = list(
        csv.DictReader((pack / "sample_manifest.csv").open(encoding="utf-8", newline=""))
    )

    idx_path = args.library_index
    if idx_path is None:
        _, idx_path = default_nist_paths(_REPO)
    lib = None
    if not args.no_library and Path(idx_path).is_file():
        print(f"[info] library {idx_path}", flush=True)
        lib = load_library_index(idx_path)
        print(f"[info] n_records={lib.n_records}", flush=True)

    jobs_out = pack / "jobs_mgf_only"
    prompts_out = pack / "prompts_mgf_only"
    jobs_out.mkdir(exist_ok=True)
    prompts_out.mkdir(exist_ok=True)

    method_template = MethodCard(
        chromatography="RP-C18",
        polarity="both",
        ionization="ESI",
        gradient="aqueous_to_organic",
        study_id="ASTRAL_C18_holdout",
        notes="MGF-only ablation; polarity from seed MGF IONMODE",
    )

    n_ok = 0
    for i, row in enumerate(rows, 1):
        sid = (row.get("spectrum_id") or "").strip().upper()
        seed = pack / "seed_mgfs_blind" / f"{sid}.mgf"
        # fallback subgraph mgf (contains seed + neighbors; we only take seed spectrum)
        sub = pack / "subgraph_mgfs" / f"{sid}.mgf"
        if not seed.is_file() and not sub.is_file():
            print(f"[{i}] skip no mgf {sid}", flush=True)
            continue

        # Seed m/z from manifest or MGF
        seed_mz = None
        try:
            seed_mz = float(row.get("seed_mz") or 0) or None
        except Exception:
            seed_mz = None

        spectral = build_spectral_context(
            seed_id="0",
            seed_mz=seed_mz,
            neighbor_ids=[],
            mgf_paths=[sub] if sub.is_file() else [],
            seed_mgf=seed if seed.is_file() else None,
        )
        if spectral and spectral.seed and spectral.seed.pepmass:
            seed_mz = float(spectral.seed.pepmass)

        seed_node = Node(id="0", mz=seed_mz, name=None, smiles=None)
        ego = EgoContext(
            seed=seed_node,
            seed_mz=seed_mz,
            neighbors=[],
            two_hop_named=[],
            hide_seed_name=True,
            spectral=spectral,
        )

        seed_ion = getattr(spectral, "seed_ion_mode", None) if spectral else None
        pol = resolve_method_polarity(
            seed_ion_mode=seed_ion, method=method_template, default="both"
        )
        method = MethodCard(
            chromatography=method_template.chromatography,
            polarity=pol,
            ionization=method_template.ionization,
            gradient=method_template.gradient,
            study_id=method_template.study_id,
            notes=method_template.notes,
        )
        ego.meta["method_card"] = method.to_dict() if hasattr(method, "to_dict") else {
            "chromatography": method.chromatography,
            "polarity": method.polarity,
            "ionization": method.ionization,
            "gradient": method.gradient,
            "study_id": method.study_id,
            "notes": method.notes,
        }
        ego.meta["seed_ion_mode"] = seed_ion
        ego.meta["protocol"] = "mgf_only_no_graphml"

        library_hits = []
        if (
            lib is not None
            and spectral
            and spectral.seed
            and spectral.seed.peaks
        ):
            library_hits = lib.search(
                spectral.seed.peaks,
                seed_mz or spectral.seed.pepmass,
                top_k=args.library_top_k,
                precursor_tol_da=0.05,
                ion_mode=seed_ion or pol,
            )
            ego.meta["library_hits"] = [
                h.to_dict() if hasattr(h, "to_dict") else h for h in library_hits
            ]

        # Offline MS/MS explain (no neighbors)
        try:
            from ego_mol_llm.msms_explain import build_msms_explanation

            if spectral and spectral.seed and spectral.seed.peaks:
                exp = build_msms_explanation(
                    list(spectral.seed.peaks),
                    seed_mz,
                    neighbor_spectra=None,
                )
                ego.meta["msms_explanation"] = exp.to_dict() if hasattr(exp, "to_dict") else exp
        except Exception:
            pass

        messages = build_messages(ego, extra_instructions=EXTRA)
        # Override system for MGF-only framing
        system = SYSTEM_MGF_ONLY
        user = messages[-1]["content"] if messages else ""
        # strip accidental network sections if any empty placeholders
        user = (
            "TASK: Assign structure of the UNKNOWN precursor from MS/MS only "
            "(no ego-network / GraphML neighbors).\n\n" + user
        )

        job = {
            "spectrum_id": sid,
            "seed_mz": seed_mz,
            "n_neighbors": 0,
            "msms_used": bool(spectral and spectral.seed and spectral.seed.peaks),
            "seed_rt": getattr(spectral, "seed_rt", None) if spectral else None,
            "seed_ion_mode": seed_ion,
            "protocol": "mgf_only_no_graphml",
            "product_version": "0.2.1-mgf-only",
            "n_library_hits": len(library_hits),
            "system_prompt": system,
            "user_prompt": user,
            "blind": True,
            "method_card": ego.meta.get("method_card"),
            "library_hits": ego.meta.get("library_hits") or [],
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
            f"[{i}/{len(rows)}] {sid} mz={seed_mz} ion={seed_ion} lib={len(library_hits)}",
            flush=True,
        )

    print(f"[done] mgf-only jobs={n_ok} -> {jobs_out}", flush=True)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
