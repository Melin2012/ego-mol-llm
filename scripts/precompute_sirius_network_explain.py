#!/usr/bin/env python3
"""
SIRIUS-first ego-network fragment explanation for a blind pack.

For each sample:
  1. Load graph + seed/neighbor MGFs
  2. Top dual-cosine neighbors with SMILES → parent formula → sirius decomp
  3. Transfer fragment formulas onto seed peaks (shared m/z)
  4. Also decomp seed under top SIRIUS formula from pack/sirius_hits/ (if present)
  5. Write pack/sirius_explain/<spectrum_id>.json
  6. Optionally refresh jobs/prompts with the SIRIUS fragment block for the LLM

Example
-------
  python scripts/precompute_sirius_network_explain.py \\
    --pack .../Blind_holdout40b_ego_msms \\
    --max-neighbors 6 --inject-prompts --force
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
from ego_mol_llm.prompts import SYSTEM_PROMPT
from ego_mol_llm.sirius import find_sirius_binary
from ego_mol_llm.sirius_network_explain import (
    build_ego_sirius_fragment_explanation,
    format_sirius_fragment_explanation_for_prompt,
)


MARKER = "=== SIRIUS FRAGMENT EXPLANATION"


def load_seed_sirius_prior(hits_dir: Path, sid: str) -> tuple[str | None, str | None]:
    """Return (formula, smiles) from precomputed CSI/SIRIUS hits if available."""
    p = hits_dir / f"{sid}.json"
    if not p.is_file():
        return None, None
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return None, None
    hits = data.get("hits") or []
    formula = None
    smiles = None
    for h in hits:
        if not isinstance(h, dict):
            continue
        if not formula and h.get("formula"):
            # prefer neutral molecularFormula-like strings without charge
            f = str(h["formula"]).split()[0]
            if re_looks_like_formula(f):
                formula = f
        if not smiles and h.get("smiles"):
            smiles = h["smiles"]
        if formula and smiles:
            break
    # also meta run formula from top structure
    return formula, smiles


def re_looks_like_formula(f: str) -> bool:
    import re

    return bool(re.match(r"^[A-Z][A-Za-z0-9]*$", f or ""))


def inject_into_prompts(pack: Path, sid: str, expl: dict) -> None:
    jobs = pack / "jobs" / f"{sid}.json"
    prompts = pack / "prompts" / f"{sid}.txt"
    if not jobs.is_file():
        return
    job = json.loads(jobs.read_text(encoding="utf-8"))
    block = "\n".join(format_sirius_fragment_explanation_for_prompt(expl))
    user = job.get("user_prompt") or ""
    if MARKER in user:
        pre, _, rest = user.partition(MARKER)
        lines = rest.splitlines()
        cut = 0
        for i, ln in enumerate(lines):
            if i == 0:
                continue
            if ln.startswith("===") and "SIRIUS FRAGMENT" not in ln:
                cut = i
                break
        else:
            cut = len(lines)
        rest_keep = "\n".join(lines[cut:])
        user = pre.rstrip() + "\n" + block + "\n" + rest_keep.lstrip("\n")
    else:
        user = user.rstrip() + "\n" + block + "\n"
    # task rules nudge
    if "SIRIUS fragment FORMULAS" not in user:
        user += (
            "\nADDITIONAL (SIRIUS-first fragments):\n"
            "- Use SIRIUS fragment FORMULA block for elemental / loss reasoning.\n"
            "- Prefer structures that explain intense peaks with listed subformulas.\n"
            "- Network-transferred formulas are stronger when neighbor SMILES is mass-related.\n"
        )
    job["user_prompt"] = user
    job["sirius_explain"] = True
    job["product_version"] = "0.4-sirius-first"
    jobs.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")

    system = job.get("system_prompt") or SYSTEM_PROMPT
    prompt_txt = (
        "### SYSTEM\n"
        + system
        + "\n\n### USER\n"
        + user
        + "\n\n### RESPONSE FORMAT\n"
        "Reply with one JSON object only.\n"
    )
    prompts.write_text(prompt_txt, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--sirius-bin", type=Path, default=None)
    ap.add_argument("--max-neighbors", type=int, default=6)
    ap.add_argument("--top-peaks", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--inject-prompts", action="store_true")
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--abs-tol", type=float, default=0.02)
    args = ap.parse_args()

    pack = args.pack.resolve()
    out_dir = pack / "sirius_explain"
    out_dir.mkdir(exist_ok=True)
    hits_dir = pack / "sirius_hits"

    bin_path = find_sirius_binary(args.sirius_bin)
    print(f"[info] sirius_bin={bin_path}", flush=True)
    if bin_path is None:
        print("[error] SIRIUS CLI not found (set SIRIUS_BIN or --sirius-bin)", flush=True)
        return 2

    rows = list(
        csv.DictReader((pack / "sample_manifest.csv").open(encoding="utf-8", newline=""))
    )
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    t0 = time.perf_counter()
    n_ok = n_skip = n_fail = 0

    for i, row in enumerate(rows, 1):
        sid = (row.get("spectrum_id") or "").strip().upper()
        out_p = out_dir / f"{sid}.json"
        if out_p.is_file() and not args.force:
            print(f"[{i}/{len(rows)}] {sid} skip existing", flush=True)
            n_skip += 1
            if args.inject_prompts:
                expl = json.loads(out_p.read_text(encoding="utf-8"))
                inject_into_prompts(pack, sid, expl)
            continue

        gpath = pack / "graphml" / f"{sid}.graphml"
        if not gpath.is_file():
            print(f"[{i}] {sid} missing graphml", flush=True)
            n_fail += 1
            continue
        try:
            net = load_graphml(gpath)
            ego = build_ego(
                net,
                seed_id="0",
                hide_seed_name=True,
                max_neighbors=50,
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
            if not ego.spectral or not ego.spectral.seed or not ego.spectral.seed.peaks:
                print(f"[{i}] {sid} no seed peaks", flush=True)
                n_fail += 1
                continue

            msms = ego.spectral.neighbor_msms_cosine or {}
            npeaks = ego.spectral.neighbor_peaks or {}
            nmeta = ego.spectral.neighbor_meta or {}
            neighbors = []
            for ev in ego.neighbors:
                nid = str(ev.node.id)
                neighbors.append(
                    {
                        "id": nid,
                        "smiles": ev.node.smiles,
                        "name": ev.node.name,
                        "peaks": npeaks.get(nid) or [],
                        "mz": ev.node.mz,
                        "msms_cosine": msms.get(nid),
                        "ion_mode": (nmeta.get(nid) or {}).get("ion_mode")
                        or ego.spectral.seed_ion_mode,
                    }
                )

            seed_formula, seed_smi = load_seed_sirius_prior(hits_dir, sid)
            expl = build_ego_sirius_fragment_explanation(
                spectrum_id=sid,
                seed_peaks=list(ego.spectral.seed.peaks),
                seed_ion_mode=ego.spectral.seed_ion_mode,
                neighbors=neighbors,
                max_neighbors=args.max_neighbors,
                sirius_bin=bin_path,
                seed_formula=seed_formula,
                seed_sirius_smiles=seed_smi,
                top_peaks=args.top_peaks,
                ppm=args.ppm,
                abs_tol=args.abs_tol,
            )
            payload = expl.to_dict()
            out_p.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if args.inject_prompts:
                inject_into_prompts(pack, sid, payload)
            n_ok += 1
            n_ann = sum(1 for p in expl.seed_peak_map if p.fragment_formula)
            print(
                f"[{i}/{len(rows)}] {sid} ok neighbors={len(expl.neighbors)} "
                f"seed_formulas={n_ann}/{len(expl.seed_peak_map)} "
                f"parent={seed_formula}",
                flush=True,
            )
        except Exception as e:
            n_fail += 1
            print(f"[{i}] {sid} FAIL {type(e).__name__}: {e}", flush=True)

    summary = {
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_fail": n_fail,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "sirius_bin": str(bin_path) if bin_path else None,
        "product_version": "0.4-sirius-first",
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {summary}", flush=True)
    return 0 if n_fail == 0 or n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
