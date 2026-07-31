#!/usr/bin/env python3
"""
Build a multi-model blind pack for MassSpecGym test spectra that have HNSW ego nets.

Sources
-------
- GraphML: Downloads/MassSpecGym/HNSW_spectrum_<N>.graphml  (seed node id=9999999)
- Subgraph MGF: dropbox-download/HNSW_spectrum_<N>.mgf
- Seed MS/MS (blind): MassSpecGym_split/MassSpecGym_test_blind (2).mgf  SPECTRUM_ID=k
- Link: MassSpecGym_linked/blind_test_vs_HNSW_coverage.csv

Outputs
-------
  Desktop/MSG_HNSW_blind285_ego_msms/          # share with models
  Desktop/MSG_HNSW_blind285_ego_msms_SEALED_truth/  # local only
"""

from __future__ import annotations

import csv
import json
import re
import shutil
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
from ego_mol_llm.prompts import SYSTEM_PROMPT, build_messages

SEED_NODE_ID = "9999999"

MODEL_EXTRA = """
TASK RULES (MassSpecGym HNSW ego, blind seed):
- Center spectrum is unlabeled; do NOT invent peaks or a library name for the seed.
- Neighbor names/SMILES and network cosines are intentional evidence — use them.
- Prefer mass-consistent SMILES for the precursor m/z (common adducts and multimers).
- Prefer neighbors with high edge cosine AND high MS/MS cosine when both exist.
- Output one JSON object only with smiles, iupac_or_common_name, formula, adduct,
  confidence, rationale, alternatives.
"""


def parse_blind_mgf_by_spectrum_id(path: Path) -> dict[int, str]:
    """Map SPECTRUM_ID (int) -> full BEGIN/END IONS block text."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[int, str] = {}
    for body in re.split(r"(?i)BEGIN IONS", text)[1:]:
        parts = re.split(r"(?i)END IONS", body, maxsplit=1)
        block_body = parts[0]
        m = re.search(r"(?im)^SPECTRUM_ID=(\d+)\s*$", block_body)
        if not m:
            continue
        sid = int(m.group(1))
        # rewrite as PEPMASS-friendly seed MGF (ego_mol_llm uses PEPMASS)
        lines = ["BEGIN IONS"]
        pepmass = None
        for ln in block_body.splitlines():
            u = ln.strip()
            if not u:
                continue
            if u.upper().startswith("SPECTRUM_ID="):
                lines.append(f"TITLE=MSGTEST_BLIND_{sid:06d}")
                lines.append(f"SPECTRUMID=MSGTEST{sid:06d}")
                continue
            if u.upper().startswith("PRECURSOR_MZ="):
                pepmass = u.split("=", 1)[1].strip()
                lines.append(f"PEPMASS={pepmass}")
                lines.append(u)
                continue
            if u.upper().startswith("PARENT_MASS="):
                lines.append(u)
                continue
            if u.upper().startswith("COLLISION_ENERGY="):
                lines.append(u)
                continue
            if u.upper().startswith("INSTRUMENT_TYPE="):
                lines.append(u)
                continue
            if u.upper().startswith("FOLD=") or u.upper().startswith("SIMULATION_CHALLENGE="):
                continue  # not needed for models
            # peaks or other
            lines.append(u)
        # ion mode soft prior from later job; default positive often
        if not any(x.upper().startswith("IONMODE=") for x in lines):
            lines.append("IONMODE=Positive")
        lines.append("END IONS")
        out[sid] = "\n".join(lines) + "\n"
    return out


def adduct_to_ion_mode(adduct: str | None) -> str:
    a = (adduct or "").lower()
    if "-" in a and "m+" not in a.replace(" ", ""):
        # rough: [M-H]-, [M+Cl]- etc.
        if a.endswith("-") or "m-h" in a or "+cl" in a or "formate" in a:
            return "negative"
    if "[m-" in a and "]+" not in a:
        return "negative"
    return "positive"


def main() -> int:
    t0 = time.perf_counter()
    coverage = Path(
        r"C:\Users\AlexeyMelnik\Downloads\MassSpecGym_linked\blind_test_vs_HNSW_coverage.csv"
    )
    blind_mgf = Path(
        r"C:\Users\AlexeyMelnik\Downloads\MassSpecGym_split\MassSpecGym_test_blind (2).mgf"
    )
    pack = Path(r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_blind285_ego_msms")
    sealed = Path(r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_blind285_ego_msms_SEALED_truth")

    if pack.exists():
        shutil.rmtree(pack)
    if sealed.exists():
        shutil.rmtree(sealed)

    for d in (
        pack / "graphml",
        pack / "subgraph_mgfs",
        pack / "seed_mgfs_blind",
        pack / "jobs",
        pack / "prompts",
        pack / "predictions",
        sealed,
    ):
        d.mkdir(parents=True, exist_ok=True)

    print("Loading blind seed MGF index...", flush=True)
    seed_blocks = parse_blind_mgf_by_spectrum_id(blind_mgf)
    print(f"  blind spectra indexed: {len(seed_blocks)}", flush=True)

    cov = [
        r
        for r in csv.DictReader(coverage.open(encoding="utf-8"))
        if str(r.get("has_hnsw", "")).lower() in {"true", "1", "yes"}
    ]
    cov.sort(key=lambda r: int(r["blind_spectrum_id"]))
    print(f"  HNSW-linked test samples: {len(cov)}", flush=True)

    manifest_rows = []
    truth_rows = []
    n_ok = n_fail = 0
    errors = []

    for i, row in enumerate(cov, 1):
        bid = int(row["blind_spectrum_id"])
        hnsw_n = int(row["msg_file_order_index"])
        sample_id = f"MSGTEST{bid:06d}"
        gsrc = Path(row["graphml_path"])
        msrc = Path(row["subgraph_mgf_path"])
        if not gsrc.is_file() or not msrc.is_file():
            n_fail += 1
            errors.append({"sample_id": sample_id, "error": "missing graphml/mgf"})
            continue
        if bid not in seed_blocks:
            n_fail += 1
            errors.append({"sample_id": sample_id, "error": f"missing blind SPECTRUM_ID={bid}"})
            continue

        try:
            # copy assets with model-facing IDs
            gdst = pack / "graphml" / f"{sample_id}.graphml"
            mdst = pack / "subgraph_mgfs" / f"{sample_id}.mgf"
            sdst = pack / "seed_mgfs_blind" / f"{sample_id}.mgf"
            shutil.copy2(gsrc, gdst)
            shutil.copy2(msrc, mdst)
            sdst.write_text(seed_blocks[bid], encoding="utf-8")

            net = load_graphml(gdst)
            ego = build_ego(
                net,
                seed_id=SEED_NODE_ID,
                hide_seed_name=True,
                max_neighbors=50,
                include_two_hop=True,
            )
            # ensure seed hidden
            ego.seed.name = None
            ego.seed.smiles = None

            ego.spectral = build_spectral_context(
                seed_id=SEED_NODE_ID,
                seed_mz=ego.seed_mz,
                neighbor_ids=[e.node.id for e in ego.neighbors],
                mgf_paths=[mdst],
                seed_mgf=sdst,
            )

            ion = ego.spectral.seed_ion_mode if ego.spectral else None
            if not ion:
                ion = adduct_to_ion_mode(None)

            msgs = build_messages(
                ego,
                extra_instructions=MODEL_EXTRA,
            )
            # force polarity hint from adduct in truth meta later; soft positive default
            system = msgs[0]["content"] if msgs else SYSTEM_PROMPT
            user = msgs[1]["content"] if len(msgs) > 1 else ""

            # prepend sample header
            user = (
                f"SAMPLE_ID={sample_id}\n"
                f"MSG_IDENTIFIER (hidden structure; id for bookkeeping only)={row['msg_identifier']}\n"
                f"HNSW_STEM={row['hnsw_stem']}\n"
                f"BLIND_SPECTRUM_ID={bid}\n\n"
                + user
            )

            job = {
                "spectrum_id": sample_id,
                "blind_spectrum_id": bid,
                "msg_identifier": row["msg_identifier"],
                "hnsw_stem": row["hnsw_stem"],
                "msg_file_order_index": hnsw_n,
                "seed_mz": ego.seed_mz,
                "seed_ion_mode": ion or "positive",
                "n_neighbors": len(ego.neighbors),
                "msms_used": bool(ego.spectral and ego.spectral.seed and ego.spectral.seed.peaks),
                "protocol": "hide_seed_neighbors_kept",
                "seed_node_id": SEED_NODE_ID,
                "system_prompt": system,
                "user_prompt": user,
                "blind": True,
                "product_version": "msg-hnsw-blind285-v1",
            }
            (pack / "jobs" / f"{sample_id}.json").write_text(
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
            (pack / "prompts" / f"{sample_id}.txt").write_text(prompt_txt, encoding="utf-8")

            manifest_rows.append(
                {
                    "spectrum_id": sample_id,
                    "blind_spectrum_id": bid,
                    "msg_identifier": row["msg_identifier"],
                    "hnsw_stem": row["hnsw_stem"],
                    "msg_file_order_index": hnsw_n,
                    "seed_mz": ego.seed_mz,
                    "n_neighbors": len(ego.neighbors),
                    "msms_used": job["msms_used"],
                }
            )
            truth_rows.append(
                {
                    "spectrum_id": sample_id,
                    "blind_spectrum_id": bid,
                    "msg_identifier": row["msg_identifier"],
                    "hnsw_stem": row["hnsw_stem"],
                    "msg_file_order_index": hnsw_n,
                    "true_smiles": row.get("smiles") or "",
                    "true_inchikey": row.get("inchikey") or "",
                    "true_name": row.get("msg_identifier") or "",
                    "true_formula": "",
                    "precursor_mz": row.get("precursor_mz") or "",
                    "seed_mz_graphml": ego.seed_mz,
                }
            )
            n_ok += 1
        except Exception as e:
            n_fail += 1
            errors.append({"sample_id": sample_id, "error": f"{type(e).__name__}: {e}"})

        if i % 25 == 0 or i == len(cov):
            print(f"[{i}/{len(cov)}] ok={n_ok} fail={n_fail}", flush=True)

    # fill formula via rdkit if available
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdMolDescriptors

        RDLogger.DisableLog("rdApp.*")
        for t in truth_rows:
            m = Chem.MolFromSmiles(t.get("true_smiles") or "")
            if m:
                t["true_formula"] = rdMolDescriptors.CalcMolFormula(m)
                try:
                    t["true_inchikey"] = Chem.MolToInchiKey(m)
                except Exception:
                    pass
    except Exception:
        pass

    with (pack / "sample_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    with (sealed / "truth_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(truth_rows[0].keys()))
        w.writeheader()
        w.writerows(truth_rows)

    (pack / "predictions" / "_TEMPLATE.json").write_text(
        json.dumps(
            {
                "spectrum_id": "MSGTEST000000",
                "smiles": "",
                "iupac_or_common_name": None,
                "formula": None,
                "adduct": None,
                "confidence": 0.0,
                "rationale": "",
                "alternatives": [],
                "model": "your-model",
                "blind": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    meta = {
        "pack": str(pack),
        "sealed_truth": str(sealed),
        "n_requested": len(cov),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "protocol": "hide_seed_neighbors_kept",
        "seed_node_id": SEED_NODE_ID,
        "source": {
            "blind_mgf": str(blind_mgf),
            "coverage_csv": str(coverage),
            "graphml_dir": r"C:\Users\AlexeyMelnik\Downloads\MassSpecGym",
            "subgraph_mgf_dir": r"C:\Users\AlexeyMelnik\dropbox-download",
        },
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "errors": errors[:50],
        "n_errors_logged": len(errors),
    }
    (pack / "package_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (sealed / "package_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    instructions = f"""# MSG HNSW blind-285 — multi-model pack

## Meta
- **n = {n_ok}** (MassSpecGym **test** spectra that have HNSW ego GraphML + subgraph MGF)
- **protocol:** hide seed structure; neighbors kept
- **seed node:** GraphML id `{SEED_NODE_ID}`
- **NOT full MSG test** — only the 285 overlapping HNSW indices

## Paths
| | |
|--|--|
| Pack (share) | `Desktop\\MSG_HNSW_blind285_ego_msms` |
| Sealed truth (local only) | `Desktop\\MSG_HNSW_blind285_ego_msms_SEALED_truth` |
| Prompts | `prompts/<ID>.txt` |
| Jobs | `jobs/<ID>.json` |
| GraphML | `graphml/<ID>.graphml` |
| Seed MGF (blind) | `seed_mgfs_blind/<ID>.mgf` |
| Subgraph MGF | `subgraph_mgfs/<ID>.mgf` |

## IDs
`MSGTEST000001` … mapped from blind `SPECTRUM_ID` 1..285 that have HNSW  
(see `sample_manifest.csv`)

## Task (Grok / Opus / other)
For each ID in `sample_manifest.csv`:
1. Read `prompts/<ID>.txt` (or job system+user).
2. Propose **one** neutral-monomer SMILES for the unknown center.
3. Write `predictions_<model>/<ID>.json`:

```json
{{
  "spectrum_id": "<ID>",
  "smiles": "<canonical SMILES>",
  "iupac_or_common_name": "<string or null>",
  "formula": "<Hill or null>",
  "adduct": "<e.g. [M+H]+>",
  "confidence": 0.0,
  "rationale": "<2-5 sentences>",
  "alternatives": [{{"smiles": "...", "confidence": 0.0, "note": "..."}}],
  "model": "<your model name>",
  "blind": true
}}
```

## Evidence policy
- Primary: ego neighbors + seed MS/MS + network cosine / MS/MS cosine.
- Do **not** invent peaks.
- Do **not** open sealed truth until all models finish.
- Mass-gate all proposals (incl. multimers / multi-water when needed).

## Suggested folders
- `predictions_opus/`
- `predictions_grok/`

## After all models finish
Score vs `MSG_HNSW_blind285_ego_msms_SEALED_truth\\truth_index.csv`  
(metrics: IK1, exact SMILES, T≥0.7, formula match).
"""
    (pack / "INSTRUCTIONS_FOR_MODELS.md").write_text(instructions, encoding="utf-8")

    # Opus-specific paste card
    opus = f"""# Opus — MSG HNSW blind-285

## Pack
`C:\\Users\\AlexeyMelnik\\Desktop\\MSG_HNSW_blind285_ego_msms`

## Full instructions
`C:\\Users\\AlexeyMelnik\\Desktop\\MSG_HNSW_blind285_ego_msms\\INSTRUCTIONS_FOR_MODELS.md`

## Output
Write to: `predictions_opus/<MSGTEST######>.json`  
model field: `claude-opus-5-msg-hnsw-blind285`

## Rules
- n = {n_ok}; use `sample_manifest.csv` + `prompts/`
- Do NOT open `MSG_HNSW_blind285_ego_msms_SEALED_truth`
- Blind seed; neighbors kept
- When done: confirm {n_ok}/{n_ok} prediction files

## Paste
```
MassSpecGym HNSW blind-285 structure assignment.
Instructions: C:\\Users\\AlexeyMelnik\\Desktop\\MSG_HNSW_blind285_ego_msms\\INSTRUCTIONS_FOR_MODELS.md
Pack: C:\\Users\\AlexeyMelnik\\Desktop\\MSG_HNSW_blind285_ego_msms
Read prompts/<ID>.txt for every ID in sample_manifest.csv (n={n_ok}).
Write predictions_opus/<ID>.json with model="claude-opus-5-msg-hnsw-blind285".
Do NOT open Desktop\\MSG_HNSW_blind285_ego_msms_SEALED_truth.
Confirm {n_ok}/{n_ok} when finished.
```
"""
    (pack / "INSTRUCTIONS_OPUS.md").write_text(opus, encoding="utf-8")

    grok = f"""# Grok — MSG HNSW blind-285

## Pack
`C:\\Users\\AlexeyMelnik\\Desktop\\MSG_HNSW_blind285_ego_msms`

## Full instructions
`C:\\Users\\AlexeyMelnik\\Desktop\\MSG_HNSW_blind285_ego_msms\\INSTRUCTIONS_FOR_MODELS.md`

## Output
Write to: `predictions_grok/<MSGTEST######>.json`  
model field: `grok-msg-hnsw-blind285`

## Rules
- Same pack as Opus; independent blind run
- Do NOT open sealed truth; do NOT read predictions_opus/
- n = {n_ok}

## Paste (new Grok instance)
```
MassSpecGym HNSW blind-285 structure assignment.
Instructions: C:\\Users\\AlexeyMelnik\\Desktop\\MSG_HNSW_blind285_ego_msms\\INSTRUCTIONS_FOR_MODELS.md
Pack: C:\\Users\\AlexeyMelnik\\Desktop\\MSG_HNSW_blind285_ego_msms
Read prompts/<ID>.txt for every ID in sample_manifest.csv (n={n_ok}).
Write predictions_grok/<ID>.json with model="grok-msg-hnsw-blind285".
Do NOT open Desktop\\MSG_HNSW_blind285_ego_msms_SEALED_truth.
Do NOT read predictions_opus/ or any other predictions_* folder.
Confirm {n_ok}/{n_ok} when finished.
```
"""
    (pack / "INSTRUCTIONS_GROK.md").write_text(grok, encoding="utf-8")

    (pack / "RUN_CARD.txt").write_text(
        f"MSG HNSW blind pack\nsamples={n_ok}\nseed_node={SEED_NODE_ID}\n"
        f"write predictions_<model>/*.json\ndo not upload sealed_truth\n",
        encoding="utf-8",
    )
    (pack / "README.md").write_text(
        f"# MSG HNSW blind-285\n\nSee INSTRUCTIONS_FOR_MODELS.md\n\nn={n_ok}\n",
        encoding="utf-8",
    )

    if errors:
        (pack / "BUILD_ERRORS.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print("PACK", pack)
    print("SEALED", sealed)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
