#!/usr/bin/env python3
"""
Strict-blind + reasoning-model batch for publication.

Differences vs standard hide_seed_name:
1) Seed name/SMILES never enter the prompt.
2) Neighbors / 2-hop / mass-consistent library hits that match the *true*
   seed structure (same SMILES or InChIKey first block) are REMOVED before prompting
   so the model cannot copy the answer from a library self-hit.
3) Neighbor *names* containing the true compound stem are redacted.
4) Neighbor-rescue postprocess is optional and defaults OFF for pure model score
   (annotation transfer off).
5) Designed for ChemDFM-R (reasoning) with longer max tokens.

Usage:
  set OPENAI_BASE_URL=http://127.0.0.1:11434/v1
  set OPENAI_API_KEY=ollama
  python scripts/run_strict_blind_reasoning_batch.py \\
    --model chemdfm-r-14b --workers 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ego_mol_llm.graphml import load_graphml
from ego_mol_llm.ego import build_ego, NeighborEvidence
from ego_mol_llm.mgf import build_spectral_context
from ego_mol_llm.prompts import build_messages
from ego_mol_llm.backends.factory import build_backend
from ego_mol_llm.backends.base import GenerationConfig
from ego_mol_llm.validate import parse_model_output, canonicalize_smiles
from ego_mol_llm.predict import refine_with_neighborhood, PredictionResult
from ego_mol_llm.report import export_report

SPECTRUM_RE = re.compile(r"(AROMEC18COLGATE\d+)", re.I)
BEGIN_IONS = re.compile(r"(?i)BEGIN IONS")
END_IONS = re.compile(r"(?i)END IONS")


def _slug(text: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    return re.sub(r"_+", "_", s)[:max_len] or "run"


def load_truth(csv_path: Path) -> dict[str, dict]:
    out = {}
    if not csv_path.is_file():
        return out
    with csv_path.open(encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("SPECTRUMID") or "").strip().upper()
            if sid:
                out[sid] = row
    return out


def index_library_mgf(library_path: Path) -> dict[str, str]:
    if not library_path.is_file():
        return {}
    text = library_path.read_text(encoding="utf-8", errors="replace")
    idx = {}
    for b in BEGIN_IONS.split(text)[1:]:
        body = END_IONS.split(b, 1)[0]
        m = re.search(r"(?im)^SPECTRUMID=(.+)$", body, re.M)
        if m:
            idx[m.group(1).strip().upper()] = "BEGIN IONS\n" + body.strip() + "\nEND IONS\n"
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


def inchikey_prefix(smiles: str | None) -> str:
    if not smiles:
        return ""
    try:
        from rdkit import Chem
        from rdkit.Chem import inchi

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        ik = inchi.MolToInchiKey(mol) or ""
        return ik.split("-")[0].upper()
    except Exception:
        return ""


def same_molecule(smiles_a: str | None, smiles_b: str | None, ik_true: str) -> bool:
    if not smiles_a:
        return False
    ca = canonicalize_smiles(smiles_a) or smiles_a.strip()
    if smiles_b:
        cb = canonicalize_smiles(smiles_b) or smiles_b.strip()
        if ca and cb and ca == cb:
            return True
    if ik_true:
        ik_a = inchikey_prefix(smiles_a)
        if ik_a and ik_a == ik_true:
            return True
    return False


def compound_stem_from_name(name: str | None) -> str:
    if not name:
        return ""
    # drop spectrum suffix AROMEC18...
    n = re.sub(r"_?AROMEC18COLGATE\d+$", "", name, flags=re.I)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def redact_name(name: str | None, banned_stems: list[str]) -> str | None:
    if not name:
        return name
    out = name
    for stem in banned_stems:
        if not stem or len(stem) < 4:
            continue
        # case-insensitive replace
        out = re.sub(re.escape(stem), "[REDACTED]", out, flags=re.I)
    # also strip AROME IDs that might appear
    out = re.sub(r"AROMEC18COLGATE\d+", "[SPECTRUM]", out, flags=re.I)
    return out


def apply_strict_blind(ego, true_smiles: str | None, true_name: str | None) -> dict:
    """
    Mutate ego context in place: remove same-molecule annotations and redact names.
    Returns stats for logging.
    """
    ik_true = inchikey_prefix(true_smiles) if true_smiles else ""
    banned = []
    stem = compound_stem_from_name(true_name)
    if stem:
        banned.append(stem)
        # also first token before space/underscore for multi-part names
        banned.append(stem.split()[0] if " " in stem else stem.split("_")[0])
    banned = [b for b in banned if b and len(b) >= 4]

    n_before = len(ego.neighbors)
    kept: list[NeighborEvidence] = []
    removed = 0
    for ev in ego.neighbors:
        if same_molecule(ev.node.smiles, true_smiles, ik_true):
            removed += 1
            continue
        # redact name leakage
        if ev.node.name:
            new_name = redact_name(ev.node.name, banned)
            if new_name != ev.node.name:
                # Node is a dataclass-like object — set attributes if possible
                try:
                    ev.node.name = new_name
                except Exception:
                    pass
        kept.append(ev)
    ego.neighbors = kept

    two_before = len(ego.two_hop_named or [])
    two_kept = []
    two_removed = 0
    for n in ego.two_hop_named or []:
        if same_molecule(n.smiles, true_smiles, ik_true):
            two_removed += 1
            continue
        if n.name:
            try:
                n.name = redact_name(n.name, banned)
            except Exception:
                pass
        two_kept.append(n)
    ego.two_hop_named = two_kept

    # ensure seed is blinded
    ego.seed.name = None
    ego.seed.smiles = None
    ego.hide_seed_name = True
    # do not put true name into anything the prompt builder might use
    ego.meta["strict_blind"] = True
    ego.meta["removed_same_mol_neighbors"] = removed
    ego.meta["removed_same_mol_two_hop"] = two_removed
    # strip true name from meta exposed only via reports if needed — keep for eval side channel only
    return {
        "neighbors_before": n_before,
        "neighbors_after": len(kept),
        "removed_neighbors": removed,
        "two_hop_before": two_before,
        "two_hop_after": len(two_kept),
        "removed_two_hop": two_removed,
        "true_ik_prefix": ik_true,
    }


STRICT_SYSTEM_EXTRA = """
STRICT BLIND EVALUATION RULES (override everything else if conflict):
- The query identity is withheld on purpose. You will NOT be shown the correct SMILES.
- Any library self-match for the query has been removed. Do NOT invent a structure by
  copying a name that looks like a placeholder.
- Reason from: precursor m/z + adduct logic, MS/MS peaks/losses, and remaining neighborhood
  evidence (cosine, Δm/z, non-identical scaffolds).
- Prefer multi-step chemical reasoning: mass consistency first, then fragment diagnostics,
  then scaffold analogy from *non-identical* neighbors.
- Output the required JSON block with neutral monomer SMILES and adduct.

OUTPUT FORMAT (required):
- Return ONE JSON object only (no markdown fences if possible).
- "smiles" MUST be a single SMILES string, NEVER a list/array of SMILES.
- Put other candidates only under "alternatives": [{"smiles":"...","confidence":0.0,"note":"..."}].
- "adduct" must be a single string such as [M+H]+ or [M-H]- (not a list).
- Prefer structures whose neutral mass fits the precursor m/z under a common adduct
  (including [2M+H]+ / [2M-H]- when half-mass neighbors dominate).
"""

# Intended product protocol: seed unknown, neighbors fully annotated (incl. self-hits).
HIDE_SEED_SYSTEM_EXTRA = """
EVALUATION RULES (ego-network annotation mode):
- The query (center/seed) identity is withheld: you are not given the seed name/SMILES.
- Neighbor annotations (names, SMILES, formulas, adducts) ARE available and intended as
  primary evidence — this is how real molecular networks are used.
- Prefer near-isobar neighbors with high edge cosine and/or MS/MS cosine whose library
  structure fits the precursor m/z under a common adduct (incl. multimers).
- Mass consistency first, then MS/MS diagnostics, then scaffold analogy from neighbors.
- Output the required JSON block with neutral monomer SMILES and adduct.

OUTPUT FORMAT (required):
- Return ONE JSON object only (no markdown fences if possible).
- "smiles" MUST be a single SMILES string, NEVER a list/array of SMILES.
- Put other candidates only under "alternatives": [{"smiles":"...","confidence":0.0,"note":"..."}].
- "adduct" must be a single string such as [M+H]+ or [M-H]- (not a list).
"""


def predict_one(
    *,
    graphml: Path,
    network_mgf: Path | None,
    seed_mgf: Path | None,
    true_smiles: str | None,
    true_name: str | None,
    backend: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    max_neighbors: int,
    max_new_tokens: int,
    temperature: float,
    mass_tol: float,
    use_rescue: bool,
    out_dir: Path,
    protocol: str = "strict",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "prediction.json").is_file():
        d = json.loads((out_dir / "prediction.json").read_text(encoding="utf-8"))
        d["_skipped"] = True
        return d

    protocol = (protocol or "strict").strip().lower()
    if protocol not in {"strict", "hide_seed"}:
        protocol = "strict"

    t0 = time.perf_counter()
    net = load_graphml(graphml)
    ego = build_ego(
        net,
        seed_id="0",
        hide_seed_name=True,
        max_neighbors=max_neighbors,
        include_two_hop=True,
    )

    if protocol == "strict":
        # Ablation: strip same-molecule neighbors (NOT the product default)
        blind_stats = apply_strict_blind(ego, true_smiles, true_name)
        extra = STRICT_SYSTEM_EXTRA
        scrub_true_from_prompt = True
    else:
        # Product protocol: keep all neighbor annotations (incl. library self-hits)
        n_before = len(ego.neighbors)
        two_before = len(getattr(ego, "two_hop_named", None) or [])
        ego.seed.name = None
        ego.seed.smiles = None
        ego.hide_seed_name = True
        ego.meta["strict_blind"] = False
        ego.meta["protocol"] = "hide_seed"
        blind_stats = {
            "protocol": "hide_seed",
            "neighbors_before": n_before,
            "neighbors_after": n_before,
            "removed_neighbors": 0,
            "two_hop_before": two_before,
            "two_hop_after": two_before,
            "removed_two_hop": 0,
            "true_ik_prefix": None,
        }
        extra = HIDE_SEED_SYSTEM_EXTRA
        # Do NOT scrub true SMILES/name from the prompt — they may appear on neighbors
        scrub_true_from_prompt = False

    if network_mgf or seed_mgf:
        ego.spectral = build_spectral_context(
            seed_id=ego.seed.id,
            seed_mz=ego.seed_mz,
            neighbor_ids=[e.node.id for e in ego.neighbors],
            mgf_paths=[network_mgf] if network_mgf else [],
            seed_mgf=seed_mgf,
        )

    messages = build_messages(ego, extra_instructions=extra)
    if scrub_true_from_prompt:
        # Only for strict ablation: prevent residual true identity leakage
        for m in messages:
            if true_smiles and true_smiles in m["content"]:
                m["content"] = m["content"].replace(true_smiles, "[REDACTED_SMILES]")
            if true_name:
                stem = compound_stem_from_name(true_name)
                if stem and stem in m["content"]:
                    m["content"] = m["content"].replace(stem, "[REDACTED]")

    be = build_backend(backend=backend, model=model, base_url=base_url, api_key=api_key, load_in_4bit=False)
    raw = be.generate(
        messages,
        config=GenerationConfig(temperature=temperature, max_new_tokens=max_new_tokens),
    )
    parsed = parse_model_output(raw, precursor_mz=ego.seed_mz, mass_tol_da=mass_tol)
    notes: list[str] = [f"protocol={protocol}", f"blind_stats={blind_stats}"]
    if use_rescue:
        parsed, rnotes = refine_with_neighborhood(parsed, ego, mass_tol_da=mass_tol)
        notes.extend(rnotes)
    else:
        notes.append("neighbor_rescue=OFF (pure model)")

    result = PredictionResult(
        prediction=parsed,
        ego=ego,
        model_raw=raw,
        backend=getattr(be, "name", backend),
        model_id=getattr(be, "model_id", None) or getattr(be, "model", model),
        messages=messages,
        rescue_notes=notes,
    )
    export_report(result, out_dir)
    # Evaluation-side: do not display true seed name in the report header
    md_path = out_dir / "prediction.md"
    if md_path.is_file() and true_name:
        txt = md_path.read_text(encoding="utf-8", errors="replace")
        txt = re.sub(
            r"(\*\*Hidden true name\*\*.*?: `)[^`]+(`)",
            r"\1[WITHHELD]\2",
            txt,
        )
        md_path.write_text(txt, encoding="utf-8")

    d = result.to_dict()
    d["elapsed_s"] = time.perf_counter() - t0
    d["strict_blind_stats"] = blind_stats
    d["use_neighbor_rescue"] = use_rescue
    d["model"] = model
    d["blind_mode"] = protocol
    # evaluation fields (not in model prompt)
    d["true_name"] = true_name
    d["true_smiles"] = true_smiles
    (out_dir / "job_meta.json").write_text(json.dumps({
        "strict_blind_stats": blind_stats,
        "use_neighbor_rescue": use_rescue,
        "model": model,
        "blind_mode": protocol,
        "elapsed_s": d["elapsed_s"],
        "true_name": true_name,
        "true_smiles": true_smiles,
    }, indent=2), encoding="utf-8")
    return d


def _worker_job(payload: dict) -> dict:
    """Process-pool worker: one spectrum, returns summary row."""
    try:
        d = predict_one(
            graphml=Path(payload["graphml"]),
            network_mgf=Path(payload["network_mgf"]) if payload.get("network_mgf") else None,
            seed_mgf=Path(payload["seed_mgf"]) if payload.get("seed_mgf") else None,
            true_smiles=payload.get("true_smiles"),
            true_name=payload.get("true_name"),
            backend=payload["backend"],
            model=payload["model"],
            base_url=payload.get("base_url"),
            api_key=payload.get("api_key"),
            max_neighbors=int(payload["max_neighbors"]),
            max_new_tokens=int(payload["max_new_tokens"]),
            temperature=float(payload["temperature"]),
            mass_tol=float(payload["mass_tol"]),
            use_rescue=bool(payload["use_rescue"]),
            out_dir=Path(payload["out_dir"]),
            protocol=str(payload.get("protocol") or "strict"),
        )
        return {
            "spectrum_id": payload["spectrum_id"],
            "stem": payload["stem"],
            "ok": True,
            "smiles": d.get("smiles"),
            "name": d.get("name"),
            "formula": d.get("formula"),
            "confidence": d.get("confidence"),
            "mass_ok": d.get("mass_ok"),
            "source": d.get("source"),
            "msms_used": d.get("msms_used"),
            "true_name": payload.get("true_name"),
            "true_smiles": payload.get("true_smiles"),
            "true_inchikey": payload.get("true_inchikey") or "",
            "removed_neighbors": (d.get("strict_blind_stats") or {}).get("removed_neighbors"),
            "elapsed_s": d.get("elapsed_s"),
            "out_dir": payload["out_dir"],
            "error": None,
            "skipped": d.get("_skipped", False),
        }
    except Exception as e:
        run_dir = Path(payload["out_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        err = f"{type(e).__name__}: {e}"
        (run_dir / "error.txt").write_text(err + "\n\n" + traceback.format_exc(), encoding="utf-8")
        return {
            "spectrum_id": payload["spectrum_id"],
            "stem": payload["stem"],
            "ok": False,
            "error": err,
            "true_name": payload.get("true_name"),
            "true_smiles": payload.get("true_smiles"),
            "true_inchikey": payload.get("true_inchikey") or "",
            "out_dir": payload["out_dir"],
        }


def main() -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test\outputs\pub_batch_chemdfm_r_strict"),
    )
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--model", default="chemdfm-r-14b")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "ollama"))
    ap.add_argument("--max-neighbors", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--mass-tol", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 4) // 2)))
    ap.add_argument("--use-rescue", action="store_true", help="Enable neighbor rescue (default OFF for pure model)")
    ap.add_argument(
        "--protocol",
        choices=("strict", "hide_seed"),
        default="strict",
        help="strict=strip same-mol neighbors (ablation); hide_seed=keep neighbor annotations (product default)",
    )
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    data = args.data_root
    gdir = data / "graohmls"
    mdir = data / "subgraph_mgfs"
    truth = load_truth(data / "LEVEL1_ASTRAL_C18_20260708_testing2.csv")
    lib = index_library_mgf(data / "LEVEL1_ASTRAL_C18_20260708_Testing")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    seeds_dir = out / "seed_mgfs_blind"
    seeds_dir.mkdir(exist_ok=True)
    runs = out / "runs"
    runs.mkdir(exist_ok=True)

    files = sorted(gdir.glob("*.graphml"))
    if args.start:
        files = files[args.start :]
    if args.limit:
        files = files[: args.limit]

    if args.protocol == "hide_seed":
        protocol_lines = [
            "hide seed name/SMILES only",
            "KEEP all neighbor/2-hop library annotations (incl. self-hits)",
            "anonymized seed MS/MS only (peaks+PEPMASS)",
            "neighbor_rescue OFF by default",
        ]
    else:
        protocol_lines = [
            "hide seed name/SMILES",
            "remove same-molecule neighbor and 2-hop annotations (SMILES/InChIKey)",
            "redact true compound name substrings from neighbor labels",
            "anonymized seed MS/MS only (peaks+PEPMASS)",
            "neighbor_rescue OFF by default",
        ]
    meta = {
        "model": args.model,
        "backend": args.backend,
        "blind_mode": args.protocol,
        "use_neighbor_rescue": args.use_rescue,
        "n_jobs": len(files),
        "workers": args.workers,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol_lines,
    }
    (out / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)

    # Pre-write seed MGFs (avoid races)
    for gpath in files:
        stem = gpath.stem
        sid_m = SPECTRUM_RE.search(stem)
        sid = sid_m.group(1).upper() if sid_m else stem
        if sid in lib:
            sp = seeds_dir / f"{sid}_seed_blind.mgf"
            if not sp.exists():
                sp.write_text(anonymize_seed_block(lib[sid]), encoding="utf-8")

    payloads: list[dict] = []
    for gpath in files:
        stem = gpath.stem
        sid_m = SPECTRUM_RE.search(stem)
        sid = sid_m.group(1).upper() if sid_m else stem
        t = truth.get(sid, {})
        true_smiles = (t.get("SMILES") or "").strip() or None
        true_name = (t.get("NAME") or t.get("COMPOUND") or "").strip() or None
        if not true_name:
            true_name = re.sub(r"^HNSW_", "", stem)
            true_name = re.sub(r"_AROMEC18COLGATE\d+$", "", true_name, flags=re.I)
        seed_mgf = None
        if sid in lib:
            seed_mgf = str((seeds_dir / f"{sid}_seed_blind.mgf").resolve())
        net_mgf = mdir / f"{stem}.mgf"
        run_dir = runs / _slug(stem)
        if args.skip_existing and (run_dir / "prediction.json").is_file():
            # load light row from existing prediction
            try:
                d = json.loads((run_dir / "prediction.json").read_text(encoding="utf-8"))
                payloads.append({
                    "_pre_done": {
                        "spectrum_id": sid,
                        "stem": stem,
                        "ok": True,
                        "smiles": d.get("smiles"),
                        "name": d.get("name"),
                        "formula": d.get("formula"),
                        "confidence": d.get("confidence"),
                        "mass_ok": d.get("mass_ok"),
                        "source": d.get("source"),
                        "msms_used": d.get("msms_used"),
                        "true_name": true_name,
                        "true_smiles": true_smiles,
                        "true_inchikey": t.get("InChIKey") or t.get("INCHIKEY") or "",
                        "removed_neighbors": None,
                        "elapsed_s": 0,
                        "out_dir": str(run_dir),
                        "error": None,
                        "skipped": True,
                    }
                })
                continue
            except Exception:
                pass
        payloads.append({
            "spectrum_id": sid,
            "stem": stem,
            "graphml": str(gpath.resolve()),
            "network_mgf": str(net_mgf.resolve()) if net_mgf.is_file() else None,
            "seed_mgf": seed_mgf,
            "true_smiles": true_smiles,
            "true_name": true_name,
            "true_inchikey": t.get("InChIKey") or t.get("INCHIKEY") or "",
            "backend": args.backend,
            "model": args.model,
            "base_url": args.base_url,
            "api_key": args.api_key,
            "max_neighbors": args.max_neighbors,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "mass_tol": args.mass_tol,
            "use_rescue": args.use_rescue,
            "protocol": args.protocol,
            "out_dir": str(run_dir.resolve()),
        })

    pre_done = [p["_pre_done"] for p in payloads if "_pre_done" in p]
    todo = [p for p in payloads if "_pre_done" not in p]
    rows: list[dict] = list(pre_done)
    print(f"[info] already done={len(pre_done)} todo={len(todo)} workers={args.workers}", flush=True)

    n_total = len(pre_done) + len(todo)
    done_count = len(pre_done)

    if not todo:
        _write_summary(out, rows, meta)
        print(f"[done] all {len(rows)} already complete", flush=True)
        return 0

    if args.workers <= 1:
        for i, pl in enumerate(todo, 1):
            print(f"[{done_count + i}/{n_total}] {pl['spectrum_id']}", flush=True)
            row = _worker_job(pl)
            rows.append(row)
            if not row.get("ok"):
                print(f"  FAIL {row.get('error')}", flush=True)
            if (done_count + i) % 10 == 0 or i == len(todo):
                _write_summary(out, rows, meta)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker_job, pl): pl for pl in todo}
            finished = 0
            for fut in as_completed(futs):
                finished += 1
                pl = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    row = {
                        "spectrum_id": pl["spectrum_id"],
                        "stem": pl["stem"],
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                        "out_dir": pl["out_dir"],
                    }
                rows.append(row)
                status = "skip" if row.get("skipped") else ("ok" if row.get("ok") else "FAIL")
                print(
                    f"[{done_count + finished}/{n_total}] {status} {row.get('spectrum_id')}",
                    flush=True,
                )
                if finished % 10 == 0 or finished == len(todo):
                    _write_summary(out, rows, meta)

    _write_summary(out, rows, meta)
    ok = sum(1 for r in rows if r.get("ok"))
    print(f"[done] {ok}/{len(rows)} ok -> {out}", flush=True)
    return 0


def _write_summary(out: Path, rows: list[dict], meta: dict) -> None:
    fields = [
        "spectrum_id", "stem", "ok", "skipped", "smiles", "name", "formula", "confidence",
        "mass_ok", "source", "msms_used", "true_name", "true_smiles", "true_inchikey",
        "removed_neighbors", "elapsed_s", "out_dir", "error",
    ]
    with (out / "batch_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    # metrics
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs, inchi
    except Exception:
        Chem = None
    n = sum(1 for r in rows if r.get("ok"))
    ik = 0
    tani_sum = 0.0
    tani_n = 0
    for r in rows:
        if not r.get("ok"):
            continue
        pred = (r.get("smiles") or "").strip()
        true = (r.get("true_smiles") or "").strip()
        if Chem and pred and true:
            m1, m2 = Chem.MolFromSmiles(pred), Chem.MolFromSmiles(true)
            if m1 and m2:
                try:
                    pik = (inchi.MolToInchiKey(m1) or "").split("-")[0].upper()
                    tik = (r.get("true_inchikey") or "").split("-")[0].upper()
                    if pik and tik and pik == tik:
                        ik += 1
                except Exception:
                    pass
                fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
                tani_sum += DataStructs.TanimotoSimilarity(fp1, fp2)
                tani_n += 1
    metrics = {
        "n": len(rows),
        "n_ok": n,
        "inchikey_match": ik,
        "inchikey_match_rate": (ik / n) if n else None,
        "mean_tanimoto": (tani_sum / tani_n) if tani_n else None,
        "tanimoto_n": tani_n,
        "model": meta.get("model"),
        "blind_mode": meta.get("blind_mode") or "strict",
        "use_neighbor_rescue": meta.get("use_neighbor_rescue"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "progress.json").write_text(json.dumps({
        "done": len(rows),
        "ok": n,
        "failed": len(rows) - n,
        "updated_at": metrics["updated_at"],
    }, indent=2), encoding="utf-8")
    print(f"  checkpoint ok={n}/{len(rows)} metrics={metrics}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
