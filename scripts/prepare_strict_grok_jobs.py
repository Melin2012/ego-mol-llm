#!/usr/bin/env python3
"""Export strict-blind job prompts for in-session Grok 4.5 predictions."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Reuse strict helpers
sys.path.insert(0, str(_REPO / "scripts"))
from run_strict_blind_reasoning_batch import (  # type: ignore
    SPECTRUM_RE,
    STRICT_SYSTEM_EXTRA,
    anonymize_seed_block,
    apply_strict_blind,
    compound_stem_from_name,
    index_library_mgf,
    load_truth,
)

from ego_mol_llm.graphml import load_graphml
from ego_mol_llm.ego import build_ego
from ego_mol_llm.mgf import build_spectral_context
from ego_mol_llm.prompts import SYSTEM_PROMPT, build_messages


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test\outputs\grok45_insession_strict"),
    )
    ap.add_argument("--max-neighbors", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    data = args.data_root
    out = args.out
    jobs_dir = out / "jobs"
    seeds_dir = out / "seed_mgfs_blind"
    pred_dir = out / "predictions"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir.mkdir(exist_ok=True)
    pred_dir.mkdir(exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)

    truth = load_truth(data / "LEVEL1_ASTRAL_C18_20260708_testing2.csv")
    lib = index_library_mgf(data / "LEVEL1_ASTRAL_C18_20260708_Testing")
    gdir = data / "graohmls"
    mdir = data / "subgraph_mgfs"
    files = sorted(gdir.glob("*.graphml"))
    if args.start:
        files = files[args.start :]
    if args.limit:
        files = files[: args.limit]

    print(f"[info] preparing {len(files)} strict-blind jobs", flush=True)
    truth_rows = []
    n_err = 0
    for i, gpath in enumerate(files, 1):
        stem = gpath.stem
        sid_m = SPECTRUM_RE.search(stem)
        sid = sid_m.group(1).upper() if sid_m else stem
        t = truth.get(sid, {})
        true_smiles = (t.get("SMILES") or "").strip() or None
        true_name = (t.get("NAME") or t.get("COMPOUND") or "").strip() or None
        if not true_name:
            true_name = re.sub(r"^HNSW_", "", stem)
            true_name = re.sub(r"_AROMEC18COLGATE\d+$", "", true_name, flags=re.I)
        try:
            net = load_graphml(gpath)
            ego = build_ego(
                net,
                seed_id="0",
                hide_seed_name=True,
                max_neighbors=args.max_neighbors,
                include_two_hop=True,
            )
            blind_stats = apply_strict_blind(ego, true_smiles, true_name)
            seed_mgf = None
            if sid in lib:
                sp = seeds_dir / f"{sid}_seed_blind.mgf"
                if not sp.exists():
                    sp.write_text(anonymize_seed_block(lib[sid]), encoding="utf-8")
                seed_mgf = sp
            net_mgf = mdir / f"{stem}.mgf"
            if net_mgf.is_file() or seed_mgf:
                ego.spectral = build_spectral_context(
                    seed_id=ego.seed.id,
                    seed_mz=ego.seed_mz,
                    neighbor_ids=[e.node.id for e in ego.neighbors],
                    mgf_paths=[net_mgf] if net_mgf.is_file() else [],
                    seed_mgf=seed_mgf,
                )
            messages = build_messages(ego, extra_instructions=STRICT_SYSTEM_EXTRA)
            for m in messages:
                if true_smiles and true_smiles in m["content"]:
                    m["content"] = m["content"].replace(true_smiles, "[REDACTED_SMILES]")
                stem_n = compound_stem_from_name(true_name)
                if stem_n and stem_n in m["content"]:
                    m["content"] = m["content"].replace(stem_n, "[REDACTED]")
            user = next(m["content"] for m in messages if m["role"] == "user")
            system = SYSTEM_PROMPT + "\n" + STRICT_SYSTEM_EXTRA
            job = {
                "spectrum_id": sid,
                "stem": stem,
                "seed_mz": ego.seed_mz,
                "seed_id": ego.seed.id,
                "n_neighbors": len(ego.neighbors),
                "msms_used": bool(ego.spectral and ego.spectral.seed),
                "strict_blind_stats": blind_stats,
                "system_prompt": system,
                "user_prompt": user,
                "blind": True,
                "blind_mode": "strict",
                "model_target": "grok-4.5-insession-strict",
            }
            (jobs_dir / f"{sid}.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
            truth_rows.append({
                "spectrum_id": sid,
                "stem": stem,
                "true_name": true_name,
                "true_smiles": true_smiles or "",
                "true_inchikey": t.get("InChIKey") or t.get("INCHIKEY") or "",
                "true_formula": t.get("FORMULA") or "",
                "seed_mz": ego.seed_mz,
                "removed_neighbors": blind_stats.get("removed_neighbors"),
            })
        except Exception as e:
            n_err += 1
            print(f"  ERR {sid}: {type(e).__name__}: {e}", flush=True)
        if i % 100 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] errors={n_err}", flush=True)

    with (out / "truth_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "spectrum_id", "stem", "true_name", "true_smiles", "true_inchikey",
                "true_formula", "seed_mz", "removed_neighbors",
            ],
        )
        w.writeheader()
        w.writerows(truth_rows)

    # shard lists of 40 for parallel subagents
    pending = sorted(jobs_dir.glob("*.json"))
    shard_dir = out / "shards"
    shard_size = 40
    shards = []
    for i in range(0, len(pending), shard_size):
        chunk = pending[i : i + shard_size]
        # only pending predictions
        chunk = [p for p in chunk if not (pred_dir / p.name).exists()]
        if not chunk:
            continue
        sp = shard_dir / f"shard_{i // shard_size:03d}.txt"
        sp.write_text("\n".join(str(p) for p in chunk), encoding="utf-8")
        shards.append(str(sp))
    (out / "shard_list.txt").write_text("\n".join(shards), encoding="utf-8")
    print(f"[done] jobs={len(list(jobs_dir.glob('*.json')))} pending_shards={len(shards)} err={n_err}")
    print(f"[done] out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
