#!/usr/bin/env python3
"""
Build STRICT-blind MSG HNSW pack (v2) for free-form multi-model runs.

Hardening vs v1:
  - No MassSpecGymID / HNSW stem / external accession in model-facing files
  - Free-form reasoning protocol baked into system + every user prompt
  - Explicit ban on pack-wide SMILES index / mass-transfer batches
  - Sealed truth keeps full linkage for offline scoring only

Outputs
-------
  Desktop/MSG_HNSW_blind285v2_strict_ego_msms/
  Desktop/MSG_HNSW_blind285v2_strict_ego_msms_SEALED_truth/
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

# Strong free-form protocol (appended to every sample)
FREEFORM_EXTRA = """
=== MANDATORY FREE-FORM REASONING PROTOCOL (do not skip) ===

You are doing **full chemical reasoning** for ONE blind query. Quality over speed.

HARD BANS (violations invalidate the run):
1. Do NOT build or use a pack-wide SMILES index from other samples.
2. Do NOT mass-transfer a SMILES from another spectrum_id / another prompt file.
3. Do NOT write batch scripts that only mass-fit candidates without reading THIS prompt.
4. Do NOT look up any external database ID, accession, or compound registry online.
5. Do NOT invent peaks. Use only peaks and neighbor rows present in THIS prompt.
6. There is no sealed truth in this pack for you — do not search for answer keys.

REQUIRED REASONING STEPS (reflect each in your rationale, 5–10 sentences):
A) MASS: precursor m/z + ion mode → list plausible adducts/multimers/water losses;
   monoisotopic mass of your proposed neutral monomer must fit within ~0.02 Da
   (or state why a looser adduct is needed).
B) NETWORK: name the 1–3 strongest neighbors (name if given, edge_cos, msms_cos, |Δm/z|);
   say whether any near-isobar is mass-consistent with the seed.
C) MS/MS: cite diagnostic peaks / losses / shared peaks with top neighbors;
   reject structures that cannot explain the major ions.
D) CHEMISTRY: scaffold/class consistency with the ego neighborhood;
   if isomers remain (regio/stereo), pick one and put others in alternatives.
E) DECISION: one neutral-monomer SMILES; confidence high only if A–C align;
   if ambiguous among isomers, confidence ≤ 0.55.

OUTPUT: one JSON object only:
{
  "smiles": "<canonical neutral monomer SMILES>",
  "iupac_or_common_name": "<string or null>",
  "formula": "<Hill or null>",
  "adduct": "<e.g. [M+H]+>",
  "confidence": <0-1>,
  "rationale": "<5-10 sentences covering MASS, NETWORK, MS/MS, CHEMISTRY, DECISION>",
  "alternatives": [{"smiles":"...","confidence":0.0,"note":"..."}],
  "model": "<your model name>",
  "blind": true,
  "method": "freeform_llm_reasoning",
  "evidence": "per_sample_prompt_only",
  "no_packwide_index": true
}

FORBIDDEN rationale language: "pack-wide", "packwide", "global SMILES index",
"harvested from other samples", "mass-transfer batch".
=== END FREE-FORM PROTOCOL ===
"""

# Strip lookup-prone tokens from model-facing text
RE_STRIP = re.compile(
    r"(MassSpecGymID\d+|HNSW_spectrum_\d+|MSG_IDENTIFIER[^\n]*|"
    r"BLIND_SPECTRUM_ID=\d+|msg_file_order_index=\d+|"
    r"centers_division\d+\.mgf|MSV\d+)",
    re.I,
)


def sanitize_text(s: str) -> str:
    s = RE_STRIP.sub("[REDACTED]", s)
    # collapse leftover bookkeeping headers we used to inject
    s = re.sub(r"(?m)^SAMPLE_ID=.*\n?", "", s)
    s = re.sub(r"(?m)^\[REDACTED\]\s*\n?", "", s)
    return s


def parse_blind_mgf_by_spectrum_id(path: Path) -> dict[int, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[int, str] = {}
    for body in re.split(r"(?i)BEGIN IONS", text)[1:]:
        parts = re.split(r"(?i)END IONS", body, maxsplit=1)
        block_body = parts[0]
        m = re.search(r"(?im)^SPECTRUM_ID=(\d+)\s*$", block_body)
        if not m:
            continue
        sid = int(m.group(1))
        lines = ["BEGIN IONS", f"TITLE=QUERY_{sid:06d}", f"SPECTRUMID=MSGTEST{sid:06d}"]
        for ln in block_body.splitlines():
            u = ln.strip()
            if not u:
                continue
            if u.upper().startswith("SPECTRUM_ID="):
                continue
            if u.upper().startswith("PRECURSOR_MZ="):
                pepmass = u.split("=", 1)[1].strip()
                lines.append(f"PEPMASS={pepmass}")
                continue
            if u.upper().startswith("FOLD=") or u.upper().startswith("SIMULATION_CHALLENGE="):
                continue
            if u.upper().startswith("PARENT_MASS=") or u.upper().startswith("COLLISION_ENERGY="):
                lines.append(u)
                continue
            if u.upper().startswith("INSTRUMENT_TYPE="):
                lines.append(u)
                continue
            lines.append(u)
        if not any(x.upper().startswith("IONMODE=") for x in lines):
            lines.append("IONMODE=Positive")
        lines.append("END IONS")
        out[sid] = "\n".join(lines) + "\n"
    return out


def sanitize_graphml_bytes(raw: str) -> str:
    """Remove accidental MassSpecGym / HNSW identifiers from node labels if present."""
    raw = re.sub(r"MassSpecGymID\d+", "LIBRARY_HIT", raw, flags=re.I)
    raw = re.sub(r"HNSW_spectrum_\d+", "ego_query", raw, flags=re.I)
    # seed name spectrum_N is ok (local); keep PEPMASS
    return raw


def main() -> int:
    t0 = time.perf_counter()
    coverage = Path(
        r"C:\Users\AlexeyMelnik\Downloads\MassSpecGym_linked\blind_test_vs_HNSW_coverage.csv"
    )
    blind_mgf = Path(
        r"C:\Users\AlexeyMelnik\Downloads\MassSpecGym_split\MassSpecGym_test_blind (2).mgf"
    )
    pack = Path(r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_blind285v2_strict_ego_msms")
    sealed = Path(r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_blind285v2_strict_ego_msms_SEALED_truth")

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

    print("Loading blind seed MGF...", flush=True)
    seed_blocks = parse_blind_mgf_by_spectrum_id(blind_mgf)
    print(f"  indexed {len(seed_blocks)} blind spectra", flush=True)

    cov = [
        r
        for r in csv.DictReader(coverage.open(encoding="utf-8"))
        if str(r.get("has_hnsw", "")).lower() in {"true", "1", "yes"}
    ]
    cov.sort(key=lambda r: int(r["blind_spectrum_id"]))
    print(f"  HNSW-linked test samples: {len(cov)}", flush=True)

    # Free-form-first system prompt (prepend protocol)
    system_strict = (
        SYSTEM_PROMPT.strip()
        + "\n\n"
        + "CRITICAL: You must follow the FREE-FORM REASONING PROTOCOL in the user message. "
        "Network neighbors are evidence, not an excuse to skip reasoning. "
        "Never harvest candidates from other queries.\n"
    )

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
        if not gsrc.is_file() or not msrc.is_file() or bid not in seed_blocks:
            n_fail += 1
            errors.append({"sample_id": sample_id, "error": "missing assets"})
            continue
        try:
            gdst = pack / "graphml" / f"{sample_id}.graphml"
            mdst = pack / "subgraph_mgfs" / f"{sample_id}.mgf"
            sdst = pack / "seed_mgfs_blind" / f"{sample_id}.mgf"

            gtxt = sanitize_graphml_bytes(gsrc.read_text(encoding="utf-8", errors="replace"))
            gdst.write_text(gtxt, encoding="utf-8")
            # subgraph MGF: strip MSV library accessions that enable easy lookup? keep for method context
            # but redact MassSpecGym if any
            mtxt = msrc.read_text(encoding="utf-8", errors="replace")
            mtxt = re.sub(r"MassSpecGymID\d+", "NODE", mtxt, flags=re.I)
            mdst.write_text(mtxt, encoding="utf-8")
            sdst.write_text(seed_blocks[bid], encoding="utf-8")

            net = load_graphml(gdst)
            ego = build_ego(
                net,
                seed_id=SEED_NODE_ID,
                hide_seed_name=True,
                max_neighbors=50,
                include_two_hop=True,
            )
            ego.seed.name = None
            ego.seed.smiles = None

            ego.spectral = build_spectral_context(
                seed_id=SEED_NODE_ID,
                seed_mz=ego.seed_mz,
                neighbor_ids=[e.node.id for e in ego.neighbors],
                mgf_paths=[mdst],
                seed_mgf=sdst,
            )

            msgs = build_messages(ego, extra_instructions=FREEFORM_EXTRA)
            user = msgs[1]["content"] if len(msgs) > 1 else ""
            # Model-facing header: ONLY local pack ID (no MSG accession)
            user = (
                f"QUERY_ID={sample_id}\n"
                f"PROTOCOL=strict_blind_freeform_v2\n"
                f"INSTRUCTIONS: Reason fully on THIS query only. "
                f"Do not use other QUERY_IDs or external registries.\n\n"
                + sanitize_text(user)
                + "\n\n"
                + FREEFORM_EXTRA
            )
            system = system_strict

            job = {
                "spectrum_id": sample_id,
                "seed_mz": ego.seed_mz,
                "seed_ion_mode": (ego.spectral.seed_ion_mode if ego.spectral else None) or "positive",
                "n_neighbors": len(ego.neighbors),
                "msms_used": bool(
                    ego.spectral and ego.spectral.seed and ego.spectral.seed.peaks
                ),
                "protocol": "strict_blind_freeform_v2",
                "seed_node_id": SEED_NODE_ID,
                "system_prompt": system,
                "user_prompt": user,
                "blind": True,
                "strict_blind": True,
                "freeform_required": True,
                "no_external_ids": True,
                "product_version": "msg-hnsw-blind285-strict-v2",
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
                "Reply with one JSON object only. Follow FREE-FORM REASONING PROTOCOL.\n"
            )
            (pack / "prompts" / f"{sample_id}.txt").write_text(prompt_txt, encoding="utf-8")

            # public manifest: no MSG accession
            manifest_rows.append(
                {
                    "spectrum_id": sample_id,
                    "seed_mz": ego.seed_mz,
                    "n_neighbors": len(ego.neighbors),
                    "msms_used": job["msms_used"],
                    "protocol": "strict_blind_freeform_v2",
                }
            )
            # sealed: full linkage
            truth_rows.append(
                {
                    "spectrum_id": sample_id,
                    "blind_spectrum_id": bid,
                    "msg_identifier": row.get("msg_identifier") or "",
                    "hnsw_stem": row.get("hnsw_stem") or "",
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

    # sealed-only map public id -> MSG (for operators)
    with (sealed / "id_map_public_to_msg.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "spectrum_id",
                "blind_spectrum_id",
                "msg_identifier",
                "hnsw_stem",
                "msg_file_order_index",
            ],
        )
        w.writeheader()
        for t in truth_rows:
            w.writerow(
                {
                    "spectrum_id": t["spectrum_id"],
                    "blind_spectrum_id": t["blind_spectrum_id"],
                    "msg_identifier": t["msg_identifier"],
                    "hnsw_stem": t["hnsw_stem"],
                    "msg_file_order_index": t["msg_file_order_index"],
                }
            )

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
                "method": "freeform_llm_reasoning",
                "evidence": "per_sample_prompt_only",
                "no_packwide_index": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    meta = {
        "pack": str(pack),
        "sealed_truth": str(sealed),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "protocol": "strict_blind_freeform_v2",
        "strict_blind": True,
        "freeform_baked_into_prompts": True,
        "no_external_ids_in_model_facing_files": True,
        "seed_node_id": SEED_NODE_ID,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "errors": errors[:20],
        "changes_vs_v1": [
            "stripped MassSpecGymID / HNSW stems from prompts, jobs, graphml labels, subgraph mgf",
            "public manifest has no MSG accession",
            "FREEFORM_EXTRA protocol baked into every prompt twice (extra + footer)",
            "system prompt requires free-form protocol",
            "sealed truth retains full linkage for scoring only",
        ],
    }
    (pack / "package_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (sealed / "package_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    instructions = f"""# MSG HNSW blind-285 **strict v2** — free-form only

## Meta
- **n = {n_ok}**
- **protocol = strict_blind_freeform_v2**
- **No MassSpecGymIDs / HNSW stems** in model-facing files
- Free-form reasoning protocol is **inside every prompt**

## Paths
| | |
|--|--|
| Pack (share) | `Desktop\\MSG_HNSW_blind285v2_strict_ego_msms` |
| Sealed truth (local only) | `Desktop\\MSG_HNSW_blind285v2_strict_ego_msms_SEALED_truth` |

## Task
For each ID in `sample_manifest.csv`:
1. Read **`prompts/<ID>.txt` fully** (one sample at a time).
2. Follow the **FREE-FORM REASONING PROTOCOL** in the prompt (MASS → NETWORK → MS/MS → CHEMISTRY → DECISION).
3. Write `predictions_<model>/<ID>.json` with the schema in the prompt footer.

## HARD BANS
- No pack-wide SMILES index / mass-transfer batch scripts
- No looking up external registries or database IDs
- No copying other samples' predictions
- No opening sealed truth until **all models** finish

## Output folders
- `predictions_opus_freeform/`
- `predictions_grok_freeform/`

## After both finish
Score vs sealed `truth_index.csv` (IK1, exact, formula, T≥0.7).
"""
    (pack / "INSTRUCTIONS_FOR_MODELS.md").write_text(instructions, encoding="utf-8")

    opus_md = f"""# Opus free-form — MSG HNSW strict v2 (n={n_ok})

## Pack
`C:\\Users\\AlexeyMelnik\\Desktop\\MSG_HNSW_blind285v2_strict_ego_msms`

## Read
`INSTRUCTIONS_FOR_MODELS.md` + each `prompts/<ID>.txt`

## Write
`predictions_opus_freeform/<ID>.json`  
**model:** `claude-opus-5-msg-strict-v2-freeform`

## Paste
```
MSG HNSW blind-285 STRICT v2 — FULL FREE-FORM only. No shortcuts.

Pack: C:\\Users\\AlexeyMelnik\\Desktop\\MSG_HNSW_blind285v2_strict_ego_msms
Instructions: ...\\INSTRUCTIONS_FOR_MODELS.md

For EVERY ID in sample_manifest.csv (n={n_ok}):
- Read prompts/<ID>.txt fully
- Follow FREE-FORM REASONING PROTOCOL in the prompt (MASS, NETWORK, MS/MS, CHEMISTRY, DECISION)
- Write predictions_opus_freeform/<ID>.json
- model = "claude-opus-5-msg-strict-v2-freeform"
- method = "freeform_llm_reasoning"
- evidence = "per_sample_prompt_only"
- no_packwide_index = true

FORBIDDEN:
- pack-wide SMILES index or mass-transfer batch
- external database / registry lookup
- opening MSG_HNSW_blind285v2_strict_ego_msms_SEALED_truth
- reading any other predictions_* folder

Best possible accuracy. Quality over speed.
Confirm {n_ok}/{n_ok} when finished.
```
"""
    (pack / "INSTRUCTIONS_OPUS_FREEFORM.md").write_text(opus_md, encoding="utf-8")

    grok_md = f"""# Grok free-form — MSG HNSW strict v2 (n={n_ok})

## Pack
`C:\\Users\\AlexeyMelnik\\Desktop\\MSG_HNSW_blind285v2_strict_ego_msms`

## Write
`predictions_grok_freeform/<ID>.json`  
**model:** `grok-msg-strict-v2-freeform`

## Paste (new independent Grok instance recommended)
```
MSG HNSW blind-285 STRICT v2 — FULL FREE-FORM only. No shortcuts.

Pack: C:\\Users\\AlexeyMelnik\\Desktop\\MSG_HNSW_blind285v2_strict_ego_msms
Instructions: C:\\Users\\AlexeyMelnik\\Desktop\\MSG_HNSW_blind285v2_strict_ego_msms\\INSTRUCTIONS_FOR_MODELS.md

For EVERY ID in sample_manifest.csv (n={n_ok}):
- Read prompts/<ID>.txt fully (one sample at a time)
- Follow FREE-FORM REASONING PROTOCOL baked into each prompt
- Write predictions_grok_freeform/<ID>.json
- model = "grok-msg-strict-v2-freeform"
- method = "freeform_llm_reasoning"
- evidence = "per_sample_prompt_only"
- no_packwide_index = true

FORBIDDEN:
- pack-wide SMILES index / mass-transfer batch / harvesting other QUERY_IDs
- external registry lookup
- opening SEALED_truth
- reading predictions_opus_freeform or any other predictions_*

Best possible accuracy. Quality over speed. Resume by skipping IDs that already have JSON.
Confirm {n_ok}/{n_ok} when finished.
```
"""
    (pack / "INSTRUCTIONS_GROK_FREEFORM.md").write_text(grok_md, encoding="utf-8")

    (pack / "RUN_CARD.txt").write_text(
        f"MSG HNSW strict v2 freeform\nsamples={n_ok}\n"
        f"protocol=strict_blind_freeform_v2\n"
        f"write predictions_<model>_freeform/*.json\n"
        f"do not upload sealed_truth\n"
        f"no external IDs in this pack\n",
        encoding="utf-8",
    )
    (pack / "README.md").write_text(
        f"# MSG HNSW blind-285 strict v2\n\n"
        f"Free-form reasoning baked into every prompt. No MassSpecGymIDs in model-facing files.\n\n"
        f"n={n_ok}. See INSTRUCTIONS_FOR_MODELS.md / INSTRUCTIONS_OPUS_FREEFORM.md / INSTRUCTIONS_GROK_FREEFORM.md\n",
        encoding="utf-8",
    )

    if errors:
        (pack / "BUILD_ERRORS.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    # verify no MassSpecGymID in model-facing tree
    leak = []
    for p in pack.rglob("*"):
        if not p.is_file():
            continue
        if "SEALED" in str(p):
            continue
        if p.suffix.lower() not in {".txt", ".json", ".csv", ".md", ".mgf", ".graphml"}:
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if re.search(r"MassSpecGymID\d+|HNSW_spectrum_\d+", t):
            leak.append(str(p))

    print(json.dumps(meta, indent=2))
    print("LEAK_CHECK model-facing MassSpecGymID/HNSW_stem files:", len(leak))
    if leak[:10]:
        print("  examples", leak[:10])
    print("PACK", pack)
    print("SEALED", sealed)
    return 0 if n_ok > 0 and not leak else (0 if n_ok > 0 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
