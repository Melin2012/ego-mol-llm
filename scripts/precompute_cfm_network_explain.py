#!/usr/bin/env python3
"""
CFM-ID–first ego-network fragment explanation for a blind pack.

For each sample:
  1. Load graph + seed/neighbor MGFs
  2. For top dual-cosine neighbors with SMILES: cfm-annotate (+ predict cosine)
  3. Transfer fragment labels onto seed experimental peaks (shared m/z)
  4. Write pack/cfm_explain/<spectrum_id>.json
  5. Optionally refresh jobs/prompts with the CFM block for the LLM

Requires Docker + wishartlab/cfmid (or falls back to rule-based structure match).

Example
-------
  python scripts/precompute_cfm_network_explain.py \\
    --pack .../Blind_holdout40b_ego_msms \\
    --max-neighbors 6 --limit 2 --inject-prompts
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

from ego_mol_llm.cfm_network_explain import (
    build_ego_cfm_explanation,
    docker_available,
    format_cfm_explanation_for_prompt,
)
from ego_mol_llm.ego import build_ego
from ego_mol_llm.graphml import load_graphml
from ego_mol_llm.mgf import build_spectral_context
from ego_mol_llm.prompts import SYSTEM_PROMPT, build_messages


FABLE_EXTRA = """
TASK RULES (annotation propagation + fragment-aware):
- Center is unlabeled; use neighbors, NIST, RT/method, offline MS/MS explain,
  and CFM-ID fragment annotations when present.
- Prefer mass-consistent SMILES; use transferred fragment SMILES as substructure clues.
- Output one JSON object only (smiles, adduct, confidence, rationale, alternatives).
"""


def inject_into_prompts(pack: Path, sid: str, expl: dict) -> None:
    """Append/replace CFM block in jobs + prompts for this spectrum."""
    jobs = pack / "jobs" / f"{sid}.json"
    prompts = pack / "prompts" / f"{sid}.txt"
    if not jobs.is_file():
        return
    job = json.loads(jobs.read_text(encoding="utf-8"))
    # rebuild with meta if possible is heavy; inject into user_prompt
    block = "\n".join(format_cfm_explanation_for_prompt(expl))
    user = job.get("user_prompt") or ""
    marker = "=== CFM-ID / IN-SILICO FRAGMENT EXPLANATION"
    if marker in user:
        # replace old block roughly: from marker to next === or end of spectral section
        pre, _, rest = user.partition(marker)
        # drop until next major section starting with === that is not CFM
        lines = rest.splitlines()
        cut = 0
        for i, ln in enumerate(lines):
            if i == 0:
                continue
            if ln.startswith("===") and "CFM-ID" not in ln:
                cut = i
                break
        else:
            cut = len(lines)
        rest_keep = "\n".join(lines[cut:])
        user = pre.rstrip() + "\n" + block + "\n" + rest_keep.lstrip("\n")
    else:
        # insert after MS/MS EXPLANATION or QUERY MS/MS section
        user = user.rstrip() + "\n" + block + "\n"
    job["user_prompt"] = user
    job["cfm_explain"] = True
    job["product_version"] = "0.4-cfm-first"
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
    ap.add_argument("--max-neighbors", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--no-cfm",
        action="store_true",
        help="Rule-based fallback only (no Docker CFM-ID)",
    )
    ap.add_argument(
        "--inject-prompts",
        action="store_true",
        help="Merge CFM explanation into jobs/prompts for LLM",
    )
    args = ap.parse_args()
    pack = args.pack.resolve()
    out_dir = pack / "cfm_explain"
    out_dir.mkdir(exist_ok=True)

    use_cfm = not args.no_cfm
    print(
        f"[info] docker_available={docker_available()} use_cfm={use_cfm}",
        flush=True,
    )

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

            # build neighbor dicts with peaks + smiles
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

            expl = build_ego_cfm_explanation(
                spectrum_id=sid,
                seed_peaks=list(ego.spectral.seed.peaks),
                seed_ion_mode=ego.spectral.seed_ion_mode,
                neighbors=neighbors,
                max_neighbors=args.max_neighbors,
                use_cfm=use_cfm,
            )
            payload = expl.to_dict()
            out_p.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if args.inject_prompts:
                inject_into_prompts(pack, sid, payload)
            n_ok += 1
            n_trans = sum(
                1
                for p in expl.seed_peak_map
                if p.fragment_smiles and p.source == "transferred"
            )
            print(
                f"[{i}/{len(rows)}] {sid} ok backend={expl.backend} "
                f"neighbors={len(expl.neighbors)} seed_transferred={n_trans}",
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
        "use_cfm": use_cfm,
        "docker": docker_available(),
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {summary}", flush=True)
    return 0 if n_fail == 0 or n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
