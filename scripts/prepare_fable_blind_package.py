#!/usr/bin/env python3
"""
Build a blinded package for external model runs (e.g. Fable).

Product protocol (not strict strip):
  - seed name/SMILES hidden
  - neighbor annotations kept (incl. library self-hits)
  - network MGF + anonymized seed MGF included
  - filenames are spectrum IDs only (no compound names)

Truth is written ONLY under sealed_truth/ (do not upload that folder).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
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

# Extra instructions: operational, schema-focused (avoid "deep reasoning" wording)
FABLE_EXTRA = """
TASK RULES:
- The center spectrum is unlabeled. Do not invent a seed library name.
- Neighbor names and SMILES are intentional network evidence; use them.
- Prefer mass-consistent structures for the precursor m/z (common adducts and multimers).
- When MS/MS peaks are present, use them to choose among mass-consistent options.
- Prefer neighbors with both high network cosine and high MS/MS cosine when available.

OUTPUT:
Return a single JSON object only, with fields:
{
  "smiles": "<neutral monomer SMILES string, not a list>",
  "iupac_or_common_name": "<string or null>",
  "formula": "<Hill formula or null>",
  "adduct": "<e.g. [M+H]+ or [2M-H]->",
  "confidence": <number 0-1>,
  "rationale": "<short justification citing m/z and neighbors>",
  "alternatives": [{"smiles": "...", "confidence": 0.0, "note": "..."}]
}
"""


def index_library_mgf(library_path: Path) -> dict[str, str]:
    if not library_path.is_file():
        return {}
    text = library_path.read_text(encoding="utf-8", errors="replace")
    idx: dict[str, str] = {}
    for b in BEGIN_IONS.split(text)[1:]:
        body = END_IONS.split(b, 1)[0]
        m = re.search(r"(?im)^SPECTRUMID=(.+)$", body, re.M)
        if m:
            idx[m.group(1).strip().upper()] = (
                "BEGIN IONS\n" + body.strip() + "\nEND IONS\n"
            )
    return idx


def anonymize_seed_block(block: str) -> str:
    drop = {
        "NAME",
        "COMPOUND",
        "SMILES",
        "INCHIKEY",
        "INCHI",
        "FORMULA",
        "IUPAC",
        "TITLE",
        "ORIGINAL_TITLE",
        "CASNO",
        "HMDB_ID",
        "KEGG_ID",
        "PUBCHEM_ID",
        "SEQ",
        "SPECTRUMID",
    }
    lines = ["BEGIN IONS"]
    for line in block.splitlines():
        u = line.strip()
        if re.match(r"(?i)BEGIN IONS", u) or re.match(r"(?i)END IONS", u) or not u:
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
    out: dict[str, dict] = {}
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
        default=Path(
            os.environ.get(
                "EGO_MOL_DATA_ROOT",
                str(Path.home() / "Desktop" / "Ego_Mol_Test_stable"),
            )
        ),
        help="Ego_Mol_Test root (or set EGO_MOL_DATA_ROOT). No machine-specific default required.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            os.environ.get(
                "EGO_MOL_FABLE_OUT",
                str(Path.home() / "Desktop" / "Fable5_blind100_ego_msms"),
            )
        ),
        help="Output pack directory (or set EGO_MOL_FABLE_OUT).",
    )
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-neighbors", type=int, default=50)
    ap.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="CSV with spectrum_id column to exclude (repeatable). Use prior packs as holdout.",
    )
    ap.add_argument(
        "--sealed-out",
        type=Path,
        default=None,
        help="Sealed truth directory (default: <out>_SEALED_truth)",
    )
    args = ap.parse_args()

    data = args.data_root
    gdir = data / "graohmls"
    mdir = data / "subgraph_mgfs"
    truth = load_truth(data / "LEVEL1_ASTRAL_C18_20260708_testing2.csv")
    lib = index_library_mgf(data / "LEVEL1_ASTRAL_C18_20260708_Testing")

    files = sorted(gdir.glob("*.graphml"))
    if not files:
        print(f"[error] no graphml in {gdir}", flush=True)
        return 1

    exclude: set[str] = set()
    for man in args.exclude_manifest or []:
        if not man.is_file():
            print(f"[warn] exclude-manifest not found: {man}", flush=True)
            continue
        with man.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                sid = (row.get("spectrum_id") or row.get("SPECTRUMID") or "").strip().upper()
                if sid:
                    exclude.add(sid)
        print(f"[info] exclude-manifest {man.name}: running total exclude={len(exclude)}", flush=True)

    if exclude:
        before = len(files)
        files = [
            p
            for p in files
            if not (
                (m := SPECTRUM_RE.search(p.stem)) and m.group(1).upper() in exclude
            )
        ]
        print(f"[info] excluded {before - len(files)} files already used; pool={len(files)}", flush=True)

    if not files:
        print("[error] no files left after exclusion", flush=True)
        return 1

    rng = random.Random(args.seed)
    k = min(args.n, len(files))
    chosen = rng.sample(files, k=k)
    chosen = sorted(chosen, key=lambda p: p.name)

    package = args.out
    sealed = args.sealed_out or Path(str(args.out) + "_SEALED_truth")
    # Fresh package dirs
    if package.exists():
        shutil.rmtree(package)
    if sealed.exists():
        shutil.rmtree(sealed)

    graphml_out = package / "graphml"
    mgf_out = package / "subgraph_mgfs"
    seed_out = package / "seed_mgfs_blind"
    prompts_out = package / "prompts"
    jobs_out = package / "jobs"
    preds_out = package / "predictions"
    for d in (graphml_out, mgf_out, seed_out, prompts_out, jobs_out, preds_out):
        d.mkdir(parents=True, exist_ok=True)
    sealed.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    truth_rows = []
    n_ok = 0
    n_err = 0

    print(f"[info] preparing {len(chosen)} blinded samples seed={args.seed}", flush=True)

    for gpath in chosen:
        sid_m = SPECTRUM_RE.search(gpath.stem)
        sid = sid_m.group(1).upper() if sid_m else gpath.stem
        t = truth.get(sid, {})
        true_smiles = (t.get("SMILES") or "").strip()
        true_name = (t.get("NAME") or t.get("COMPOUND") or "").strip()
        true_ik = (t.get("InChIKey") or t.get("INCHIKEY") or "").strip()
        true_formula = (t.get("FORMULA") or "").strip()

        try:
            net = load_graphml(gpath)
            ego = build_ego(
                net,
                seed_id="0",
                hide_seed_name=True,
                max_neighbors=args.max_neighbors,
                include_two_hop=True,
            )
            # product protocol: do NOT strip same-mol neighbors
            ego.seed.name = None
            ego.seed.smiles = None
            ego.hide_seed_name = True

            net_mgf = mdir / f"{gpath.stem}.mgf"
            seed_mgf_path = None
            if sid in lib:
                seed_mgf_path = seed_out / f"{sid}.mgf"
                seed_mgf_path.write_text(anonymize_seed_block(lib[sid]), encoding="utf-8")

            mgf_paths = [net_mgf] if net_mgf.is_file() else []
            if mgf_paths or seed_mgf_path:
                ego.spectral = build_spectral_context(
                    seed_id=ego.seed.id,
                    seed_mz=ego.seed_mz,
                    neighbor_ids=[e.node.id for e in ego.neighbors],
                    mgf_paths=mgf_paths,
                    seed_mgf=seed_mgf_path,
                )

            messages = build_messages(ego, extra_instructions=FABLE_EXTRA)
            system = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT)
            user = next((m["content"] for m in messages if m["role"] == "user"), "")

            # Copy assets with spectrum-id-only names (no compound leak in filename)
            shutil.copy2(gpath, graphml_out / f"{sid}.graphml")
            if net_mgf.is_file():
                shutil.copy2(net_mgf, mgf_out / f"{sid}.mgf")

            job = {
                "spectrum_id": sid,
                "seed_mz": ego.seed_mz,
                "n_neighbors": len(ego.neighbors),
                "msms_used": bool(ego.spectral and ego.spectral.seed),
                "max_neighbors": args.max_neighbors,
                "protocol": "hide_seed_neighbors_kept",
                "system_prompt": system,
                "user_prompt": user,
                "blind": True,
                "files": {
                    "graphml": f"graphml/{sid}.graphml",
                    "subgraph_mgf": f"subgraph_mgfs/{sid}.mgf" if net_mgf.is_file() else None,
                    "seed_mgf_blind": f"seed_mgfs_blind/{sid}.mgf" if seed_mgf_path else None,
                    "prompt": f"prompts/{sid}.txt",
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
                "Reply with one JSON object only (fields listed in SYSTEM/USER).\n"
            )
            (prompts_out / f"{sid}.txt").write_text(prompt_txt, encoding="utf-8")

            sample_rows.append(
                {
                    "spectrum_id": sid,
                    "seed_mz": ego.seed_mz,
                    "n_neighbors": len(ego.neighbors),
                    "msms_used": job["msms_used"],
                    "prompt_file": f"prompts/{sid}.txt",
                    "job_file": f"jobs/{sid}.json",
                }
            )
            truth_rows.append(
                {
                    "spectrum_id": sid,
                    "true_name": true_name,
                    "true_smiles": true_smiles,
                    "true_inchikey": true_ik,
                    "true_formula": true_formula,
                    "original_graphml_stem": gpath.stem,
                }
            )
            n_ok += 1
            if n_ok % 20 == 0:
                print(f"  prepared {n_ok}/{len(chosen)}", flush=True)
        except Exception as e:
            n_err += 1
            print(f"[warn] {sid}: {type(e).__name__}: {e}", flush=True)

    # manifest for runner
    with (package / "sample_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()) if sample_rows else ["spectrum_id"])
        w.writeheader()
        w.writerows(sample_rows)

    # sealed truth (local scoring only)
    with (sealed / "truth_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "spectrum_id",
                "true_name",
                "true_smiles",
                "true_inchikey",
                "true_formula",
                "original_graphml_stem",
            ],
        )
        w.writeheader()
        w.writerows(truth_rows)
    (sealed / "DO_NOT_UPLOAD.txt").write_text(
        "Holdout labels for local scoring only. Do not place this folder in the Fable run package.\n",
        encoding="utf-8",
    )

    # prediction template
    with (preds_out / "_TEMPLATE.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "spectrum_id": "AROMEC18COLGATE000000",
                "smiles": "",
                "iupac_or_common_name": None,
                "formula": None,
                "adduct": None,
                "confidence": 0.0,
                "rationale": "",
                "alternatives": [],
                "model": "fable-5",
                "blind": True,
            },
            f,
            indent=2,
        )

    readme = f"""# Blind MS/MS network structure assignment pack

## Purpose
Batch of {n_ok} unlabeled center spectra with molecular-network context for structure assignment.
This is a standard technical worksheet for mass spectrometry annotation.

## Protocol
- Center spectrum identity is withheld.
- Neighbor annotations (names/SMILES) are provided as network evidence.
- Subgraph MGF and anonymized seed MGF are included when available.
- Filenames are spectrum IDs only.

## Layout
- `sample_manifest.csv` — list of spectrum_id and file paths
- `graphml/` — network files (`<spectrum_id>.graphml`)
- `subgraph_mgfs/` — network MS/MS (`<spectrum_id>.mgf`)
- `seed_mgfs_blind/` — anonymized center spectrum peaks only
- `prompts/` — ready-to-run system+user prompt text
- `jobs/` — same content as JSON
- `predictions/` — write one `<spectrum_id>.json` per sample (see `_TEMPLATE.json`)

## How to run (batch)
For each row in `sample_manifest.csv`:
1. Open `prompts/<spectrum_id>.txt`.
2. Produce one JSON object as specified in the prompt.
3. Save to `predictions/<spectrum_id>.json`.

Optional: use `jobs/<spectrum_id>.json` fields `system_prompt` and `user_prompt` if you prefer structured inputs.

## Settings used to build prompts
- max_neighbors = {args.max_neighbors}
- include_two_hop = true
- random seed = {args.seed}
- n = {n_ok} (errors skipped: {n_err})

## Notes
- Keep answers as valid single SMILES strings (not lists).
- Prefer mass-consistent adducts for the reported precursor m/z.
"""
    (package / "README.md").write_text(readme, encoding="utf-8")

    # short operator card (no sealed truth)
    (package / "RUN_CARD.txt").write_text(
        "\n".join(
            [
                "Fable blind pack — operational",
                f"samples={n_ok}",
                f"max_neighbors={args.max_neighbors}",
                "mode=hide_seed_neighbors_kept",
                "write predictions/*.json from prompts/*.txt",
                "do not include any sealed_truth folder",
                "",
            ]
        ),
        encoding="utf-8",
    )

    meta = {
        "n_requested": args.n,
        "n_prepared": n_ok,
        "n_errors": n_err,
        "random_seed": args.seed,
        "max_neighbors": args.max_neighbors,
        "protocol": "hide_seed_neighbors_kept",
        "data_root": str(data),
        "package": str(package),
        "sealed_truth": str(sealed),
        "n_excluded_prior": len(exclude),
        "exclude_manifests": [str(p) for p in (args.exclude_manifest or [])],
        "pool_after_exclude": len(files),
    }
    (package / "package_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[done] package: {package}", flush=True)
    print(f"[done] sealed truth (local only): {sealed}", flush=True)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
