#!/usr/bin/env python3
"""
Prepare blind ego-network jobs for in-session Grok 4.5 prediction (no external API).

Writes one JSON job per spectrum under:
  <out>/jobs/<spectrum_id>.json

Truth labels go only to truth_index.csv (never inside the model-facing prompt).
"""

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

from ego_mol_llm.graphml import load_graphml
from ego_mol_llm.ego import build_ego
from ego_mol_llm.mgf import build_spectral_context
from ego_mol_llm.prompts import SYSTEM_PROMPT, build_messages

SPECTRUM_RE = re.compile(r"(AROMEC18COLGATE\d+)", re.I)
BEGIN_IONS = re.compile(r"(?i)BEGIN IONS")
END_IONS = re.compile(r"(?i)END IONS")


def index_library_mgf(library_path: Path) -> dict[str, str]:
    if not library_path.is_file():
        return {}
    text = library_path.read_text(encoding="utf-8", errors="replace")
    blocks = BEGIN_IONS.split(text)
    idx: dict[str, str] = {}
    for b in blocks[1:]:
        body = END_IONS.split(b, 1)[0]
        block = "BEGIN IONS\n" + body.strip() + "\nEND IONS\n"
        m = re.search(r"(?im)^SPECTRUMID=(.+)$", body, re.M)
        if m:
            idx[m.group(1).strip().upper()] = block
    return idx


def anonymize_seed_block(block: str) -> str:
    drop = {
        "NAME", "COMPOUND", "SMILES", "INCHIKEY", "INCHI", "FORMULA", "IUPAC",
        "TITLE", "ORIGINAL_TITLE", "CASNO", "HMDB_ID", "KEGG_ID", "PUBCHEM_ID",
        "SEQ", "SPECTRUMID",
    }
    lines = ["BEGIN IONS"]
    for line in block.splitlines():
        u = line.strip()
        if re.match(r"(?i)BEGIN IONS", u) or re.match(r"(?i)END IONS", u):
            continue
        if not u:
            continue
        if "=" in u and not (u[0].isdigit() or u.startswith(".")):
            k = u.split("=", 1)[0].strip().upper()
            if k in drop:
                continue
            if k in {"PEPMASS", "CHARGE", "MSLEVEL", "IONMODE", "ION_MODE", "RTINSECONDS"}:
                lines.append(u)
            continue
        lines.append(u)
    lines.append("END IONS")
    return "\n".join(lines) + "\n"


def load_truth(csv_path: Path) -> dict[str, dict]:
    if not csv_path.is_file():
        return {}
    out = {}
    with csv_path.open(encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("SPECTRUMID") or "").strip().upper()
            if sid:
                out[sid] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test\outputs\grok45_insession"),
    )
    ap.add_argument("--max-neighbors", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="Skip first N graphmls")
    args = ap.parse_args()

    data = args.data_root
    gdir = data / "graohmls"
    mdir = data / "subgraph_mgfs"
    truth_csv = data / "LEVEL1_ASTRAL_C18_20260708_testing2.csv"
    library = data / "LEVEL1_ASTRAL_C18_20260708_Testing"

    out = args.out
    jobs_dir = out / "jobs"
    seeds_dir = out / "seed_mgfs_blind"
    pred_dir = out / "predictions"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    truth = load_truth(truth_csv)
    print(f"[info] truth={len(truth)} indexing library…", flush=True)
    lib = index_library_mgf(library)
    print(f"[info] library spectra={len(lib)}", flush=True)

    files = sorted(gdir.glob("*.graphml"))
    if args.start:
        files = files[args.start :]
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    print(f"[info] preparing {len(files)} jobs", flush=True)

    truth_rows = []
    manifest = []
    errors = 0

    for i, gpath in enumerate(files, 1):
        stem = gpath.stem
        sid_m = SPECTRUM_RE.search(stem)
        sid = sid_m.group(1).upper() if sid_m else stem
        try:
            net = load_graphml(gpath)
            ego = build_ego(
                net,
                seed_id="0",
                hide_seed_name=True,
                max_neighbors=args.max_neighbors,
                include_two_hop=True,
            )
            seed_mgf = None
            if sid in lib:
                seed_path = seeds_dir / f"{sid}_seed_blind.mgf"
                if not seed_path.exists():
                    seed_path.write_text(anonymize_seed_block(lib[sid]), encoding="utf-8")
                seed_mgf = seed_path
            net_mgf = mdir / f"{stem}.mgf"
            mgf_paths = [net_mgf] if net_mgf.is_file() else []
            if mgf_paths or seed_mgf:
                ego.spectral = build_spectral_context(
                    seed_id=ego.seed.id,
                    seed_mz=ego.seed_mz,
                    neighbor_ids=[e.node.id for e in ego.neighbors],
                    mgf_paths=mgf_paths,
                    seed_mgf=seed_mgf,
                )
            messages = build_messages(ego)
            user = next(m["content"] for m in messages if m["role"] == "user")
            # strip any accidental true name
            true_name = ego.meta.get("true_seed_name")
            if true_name and true_name in user:
                user = user.replace(true_name, "[REDACTED_SEED]")

            job = {
                "spectrum_id": sid,
                "stem": stem,
                "seed_mz": ego.seed_mz,
                "seed_id": ego.seed.id,
                "n_neighbors": len(ego.neighbors),
                "msms_used": bool(ego.spectral and ego.spectral.seed),
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user,
                "blind": True,
                "graphml": str(gpath),
            }
            job_path = jobs_dir / f"{sid}.json"
            job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
            pred_path = pred_dir / f"{sid}.json"
            manifest.append(
                {
                    "spectrum_id": sid,
                    "stem": stem,
                    "job": str(job_path),
                    "prediction": str(pred_path),
                    "done": pred_path.is_file(),
                }
            )
            trow = truth.get(sid, {})
            truth_rows.append(
                {
                    "spectrum_id": sid,
                    "stem": stem,
                    "true_name": trow.get("NAME") or trow.get("COMPOUND") or true_name,
                    "true_smiles": trow.get("SMILES") or "",
                    "true_inchikey": trow.get("InChIKey") or trow.get("INCHIKEY") or "",
                    "true_formula": trow.get("FORMULA") or "",
                    "true_exact_mass": trow.get("EXACT_MASS") or trow.get("EXACTMASS") or "",
                    "seed_mz": ego.seed_mz,
                }
            )
        except Exception as e:
            errors += 1
            manifest.append(
                {
                    "spectrum_id": sid,
                    "stem": stem,
                    "error": f"{type(e).__name__}: {e}",
                    "done": False,
                }
            )
        if i % 100 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] errors={errors}", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (out / "truth_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "spectrum_id",
                "stem",
                "true_name",
                "true_smiles",
                "true_inchikey",
                "true_formula",
                "true_exact_mass",
                "seed_mz",
            ],
        )
        w.writeheader()
        w.writerows(truth_rows)

    pending = [m for m in manifest if not m.get("done") and not m.get("error")]
    print(f"[done] jobs={len(manifest)} pending={len(pending)} errors={errors}")
    print(f"[done] out={out}")
    return 0 if errors == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
