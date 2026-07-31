#!/usr/bin/env python3
"""
Build strict free-form + neighbor-only NIST handout for the NEW full HNSW dump.

Source
------
  graphml:  HNSW_Large_Files/graphmls_new/graphmls/HNSW_spectrum_{219564..231103}.graphml
  mgf:      HNSW_Large_Files/MassSpecGym_subgraph_mgfs_new/MassSpecGym_subgraph_mgfs/

Seed: always node id 9999999 (NOT find_seed()).

Outputs
-------
  Desktop/MSG_HNSW_full11540_nist_neigh_handout/   (model-facing)
  Desktop/MSG_HNSW_full11540_nist_neigh_SEALED_truth/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ego_mol_llm.ego import build_ego
from ego_mol_llm.graphml import load_graphml
from ego_mol_llm.library_search import default_nist_paths, load_library_index
from ego_mol_llm.method_card import MethodCard, resolve_method_polarity
from ego_mol_llm.mgf import build_spectral_context
from ego_mol_llm.prompts import SYSTEM_PROMPT, build_messages

SEED_NODE_ID = "9999999"

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

NIST RULES (this pack):
- Seed was NOT reverse-searched against NIST (no seed library hits by design).
- Neighbor NIST hits ARE provided: treat them as independent IDs of *neighbors*,
  usable like GraphML annotations for propagation. Only promote a neighbor NIST
  structure to the seed if it is mass-consistent with the seed precursor under a
  valid adduct/multimer.

REQUIRED REASONING STEPS (reflect each in your rationale, 5–10 sentences):
A) MASS: precursor m/z + ion mode → list plausible adducts/multimers/water losses;
   monoisotopic mass of your proposed neutral monomer must fit within ~0.02 Da
   (or state why a looser adduct is needed).
B) NETWORK: name the 1–3 strongest neighbors (name if given, edge_cos, msms_cos, |Δm/z|);
   say whether any near-isobar is mass-consistent with the seed.
C) MS/MS: cite diagnostic peaks / losses / shared peaks with top neighbors;
   reject structures that cannot explain the major ions.
D) CHEMISTRY: scaffold/class consistency with the ego neighborhood + neighbor NIST;
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
  "model": "grok-msg-full11540-nist-neigh-freeform",
  "blind": true,
  "method": "freeform_llm_reasoning",
  "evidence": "per_sample_prompt_only+neighbor_nist",
  "no_packwide_index": true
}

FORBIDDEN rationale language: "pack-wide", "packwide", "global SMILES index",
"harvested from other samples", "mass-transfer batch".
=== END FREE-FORM PROTOCOL ===
"""

RE_STRIP = re.compile(
    r"(MassSpecGymID\d+|HNSW_spectrum_\d+|spectrum_\d+|"
    r"MSG_IDENTIFIER[^\n]*|MSV\d+|"
    r"BLIND_SPECTRUM_ID=\d+|msg_file_order_index=\d+)",
    re.I,
)


def sanitize_text(s: str) -> str:
    s = RE_STRIP.sub("[REDACTED]", s)
    s = re.sub(r"(?m)^SAMPLE_ID=.*\n?", "", s)
    s = re.sub(r"(?m)^\[REDACTED\]\s*\n?", "", s)
    return s


def _inchikey_to_smiles(ik: str, cache: dict[str, str | None]) -> str | None:
    ik = (ik or "").strip().upper()
    if not ik:
        return None
    if ik in cache:
        return cache[ik]
    try:
        import urllib.parse
        import urllib.request

        from rdkit import Chem

        url = (
            "https://cactus.nci.nih.gov/chemical/structure/InChIKey="
            + urllib.parse.quote(ik)
            + "/smiles"
        )
        with urllib.request.urlopen(url, timeout=8) as r:
            smi = r.read().decode("utf-8", errors="replace").strip()
        if smi and not smi.lower().startswith("<"):
            m = Chem.MolFromSmiles(smi)
            if m:
                smi = Chem.MolToSmiles(m)
                cache[ik] = smi
                return smi
    except Exception:
        pass
    cache[ik] = None
    return None


def _format_neighbor_nist_block(neigh_hits: list[dict], *, max_neighbors: int = 8) -> str:
    lines = [
        "",
        "=== NEIGHBOR NIST LIBRARY HITS (not the seed; seed was not searched) ===",
        "  These are reverse-search IDs of *neighbor* spectra only. Use as network",
        "  structure evidence. Do NOT treat them as direct IDs of the unknown center",
        "  unless mass-consistent with the seed precursor under a valid adduct.",
    ]
    if not neigh_hits:
        lines.append("  (no neighbor NIST hits)")
        return "\n".join(lines)
    for nh in neigh_hits[:max_neighbors]:
        lines.append(
            f"  • neighbor {nh.get('neighbor_name') or nh.get('neighbor_id')} "
            f"m/z={nh.get('neighbor_mz')} msms_cos={nh.get('msms_cosine')} "
            f"n_hits={len(nh.get('hits') or [])}"
        )
        for h in (nh.get("hits") or [])[:3]:
            smi = h.get("smiles") or ""
            lines.append(
                f"      NIST#{h.get('rank')} score={h.get('match_score')} "
                f"err={h.get('precursor_error_da')} "
                f"name={h.get('name')} SMILES={smi} "
                f"ik={h.get('inchikey')} adduct={h.get('precursor_type')}"
            )
    return "\n".join(lines)


def load_msg_index(path: Path) -> list[dict]:
    """0-based file-order rows matching full MassSpecGym.mgf order."""
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--graphml-dir",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\HNSW_Large_Files\graphmls_new\graphmls"),
    )
    ap.add_argument(
        "--mgf-dir",
        type=Path,
        default=Path(
            r"C:\Users\AlexeyMelnik\HNSW_Large_Files\MassSpecGym_subgraph_mgfs_new\MassSpecGym_subgraph_mgfs"
        ),
    )
    ap.add_argument(
        "--msg-index",
        type=Path,
        default=Path(
            r"C:\Users\AlexeyMelnik\Downloads\MassSpecGym_split\MassSpecGym_spectrum_index.csv"
        ),
    )
    ap.add_argument(
        "--handout",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_full11540_nist_neigh_handout"),
    )
    ap.add_argument(
        "--sealed",
        type=Path,
        default=Path(
            r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_full11540_nist_neigh_SEALED_truth"
        ),
    )
    ap.add_argument("--max-neighbors", type=int, default=50)
    ap.add_argument("--nist-neighbors", type=int, default=8)
    ap.add_argument("--library-top-k", type=int, default=5)
    ap.add_argument("--library-index", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="Skip first N stems (resume)")
    ap.add_argument("--no-nist", action="store_true")
    ap.add_argument("--no-resolve-smiles", action="store_true")
    ap.add_argument(
        "--fold-filter",
        default="",
        help="Optional: only build for fold train|val|test (empty = all)",
    )
    args = ap.parse_args()

    gdir = args.graphml_dir.resolve()
    mdir = args.mgf_dir.resolve()
    hand = args.handout.resolve()
    sealed = args.sealed.resolve()

    graphmls = sorted(gdir.glob("HNSW_spectrum_*.graphml"))
    if args.start:
        graphmls = graphmls[args.start :]
    if args.limit and args.limit > 0:
        graphmls = graphmls[: args.limit]

    print(f"[info] graphmls selected={len(graphmls)} from {gdir}", flush=True)

    msg_rows = load_msg_index(args.msg_index)
    print(f"[info] msg index rows={len(msg_rows)}", flush=True)

    lib = None
    if not args.no_nist:
        idx_path = args.library_index
        if idx_path is None:
            _, idx_path = default_nist_paths(_REPO)
        print(f"[info] loading NIST library {idx_path}", flush=True)
        lib = load_library_index(idx_path)
        print(f"[info] n_records={lib.n_records}", flush=True)

    for d in (
        hand,
        hand / "prompts",
        hand / "jobs",
        hand / "predictions",
        sealed,
    ):
        d.mkdir(parents=True, exist_ok=True)

    cache_p = hand / "inchikey_smiles_cache.json"
    cache: dict[str, str | None] = {}
    if cache_p.is_file():
        try:
            cache = json.loads(cache_p.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    method_template = MethodCard(
        chromatography="RP-C18",
        polarity="both",
        ionization="ESI",
        gradient="aqueous_to_organic",
        study_id="msg_hnsw_full11540_nist_neighbors",
        notes="NIST for neighbors only; seed not library-searched; seed_id=9999999",
    )

    public_rows: list[dict] = []
    sealed_rows: list[dict] = []
    fold_counts: dict[str, int] = {}
    n_ok = n_fail = n_skip = 0
    t0 = time.perf_counter()
    errors: list[str] = []

    for i, gpath in enumerate(graphmls, 1):
        stem = gpath.stem  # HNSW_spectrum_N
        m = re.search(r"HNSW_spectrum_(\d+)$", stem)
        if not m:
            n_fail += 1
            continue
        hnsw_index = int(m.group(1))
        if hnsw_index < 0 or hnsw_index >= len(msg_rows):
            n_fail += 1
            errors.append(f"{stem}: index out of msg range")
            continue
        truth = msg_rows[hnsw_index]
        fold = (truth.get("fold") or "").strip().lower()
        if args.fold_filter and fold != args.fold_filter.strip().lower():
            n_skip += 1
            continue
        fold_counts[fold or "?"] = fold_counts.get(fold or "?", 0) + 1

        sample_id = f"MSGFULL{hnsw_index:06d}"
        pred_exists = (hand / "prompts" / f"{sample_id}.txt").is_file()
        # allow resume: skip if prompt exists and jobs exist
        if pred_exists and (hand / "jobs" / f"{sample_id}.json").is_file():
            # still record manifest lines once at end from rebuild? skip fully
            n_skip += 1
            if i % 500 == 0:
                print(f"[{i}/{len(graphmls)}] skip existing {sample_id}", flush=True)
            # still need public/sealed for complete manifests if resuming partial
            public_rows.append(
                {
                    "spectrum_id": sample_id,
                    "hnsw_index": hnsw_index,
                    "precursor_mz": truth.get("precursor_mz"),
                    "adduct": truth.get("adduct"),
                    "instrument_type": truth.get("instrument_type"),
                    "collision_energy": truth.get("collision_energy"),
                }
            )
            sealed_rows.append(
                {
                    "spectrum_id": sample_id,
                    "hnsw_index": hnsw_index,
                    "hnsw_stem": stem,
                    "msg_identifier": truth.get("identifier"),
                    "fold": fold,
                    "true_smiles": truth.get("smiles"),
                    "true_inchikey": truth.get("inchikey"),
                    "true_formula": truth.get("formula"),
                    "precursor_mz": truth.get("precursor_mz"),
                    "adduct": truth.get("adduct"),
                    "instrument_type": truth.get("instrument_type"),
                    "collision_energy": truth.get("collision_energy"),
                    "parent_mass": truth.get("parent_mass"),
                }
            )
            continue

        mgf_path = mdir / f"{stem}.mgf"
        seed_mgf = hand / "seed_mgfs" / f"{sample_id}.mgf"
        try:
            net = load_graphml(gpath)
            if SEED_NODE_ID not in net.nodes:
                raise KeyError(f"missing seed node {SEED_NODE_ID}")
            ego = build_ego(
                net,
                seed_id=SEED_NODE_ID,
                hide_seed_name=True,
                max_neighbors=args.max_neighbors,
                include_two_hop=True,
            )
            ego.seed.name = None
            ego.seed.smiles = None
            # Seed MS/MS lives in full MSG MGF (pre-extracted); subgraph MGF has neighbors only.
            ego.spectral = build_spectral_context(
                seed_id=SEED_NODE_ID,
                seed_mz=ego.seed_mz,
                neighbor_ids=[e.node.id for e in ego.neighbors],
                mgf_paths=[mgf_path] if mgf_path.is_file() else [],
                seed_mgf=seed_mgf if seed_mgf.is_file() else None,
            )
            seed_ion = (
                getattr(ego.spectral, "seed_ion_mode", None) if ego.spectral else None
            )
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
            ego.meta["library_hits"] = []
            ego.meta["seed_nist_disabled"] = True
            ego.meta["method_card"] = method.to_dict()
            ego.meta["seed_ion_mode"] = seed_ion
            ego.meta["protocol"] = "strict_blind_freeform_full11540_nist_neighbors"
            ego.meta["seed_node_id"] = SEED_NODE_ID
            if lib is not None:
                ego.meta["library_n_records"] = lib.n_records

            neighbor_nist: list[dict] = []
            n_hits_total = 0
            if lib is not None and ego.spectral:
                msms = ego.spectral.neighbor_msms_cosine or {}
                npeaks = ego.spectral.neighbor_peaks or {}
                nmeta = ego.spectral.neighbor_meta or {}

                def nkey(ev):
                    nid = str(ev.node.id)
                    return (
                        0 if (npeaks.get(nid) or []) else 1,
                        -(float(msms.get(nid) or 0.0)),
                        -(float(ev.cosine or 0.0)),
                    )

                ranked_ev = sorted(ego.neighbors, key=nkey)[: args.nist_neighbors]
                for ev in ranked_ev:
                    nid = str(ev.node.id)
                    peaks = npeaks.get(nid) or []
                    if not peaks:
                        continue
                    ion = (nmeta.get(nid) or {}).get("ion_mode") or seed_ion or pol
                    if ion not in {"positive", "negative"}:
                        ion = None
                    hits = lib.search(
                        peaks,
                        ev.node.mz,
                        top_k=args.library_top_k,
                        precursor_tol_da=0.05,
                        ion_mode=ion,
                    )
                    hit_dicts = []
                    for h in hits:
                        d = h.to_dict()
                        if (
                            not d.get("smiles")
                            and d.get("inchikey")
                            and not args.no_resolve_smiles
                        ):
                            smi = _inchikey_to_smiles(
                                str(d["inchikey"]).split()[0], cache
                            )
                            if smi:
                                d["smiles"] = smi
                                d["smiles_resolved_from_inchikey"] = True
                        hit_dicts.append(d)
                    n_hits_total += len(hit_dicts)
                    # sanitize neighbor name for model-facing
                    nname = sanitize_text(ev.node.name or "") or nid
                    neighbor_nist.append(
                        {
                            "neighbor_id": nid,
                            "neighbor_name": nname,
                            "neighbor_smiles_graphml": ev.node.smiles,
                            "neighbor_mz": ev.node.mz,
                            "msms_cosine": msms.get(nid),
                            "edge_cosine": ev.cosine,
                            "hits": hit_dicts,
                        }
                    )
                ego.meta["neighbor_library_hits"] = neighbor_nist

            messages = build_messages(ego, extra_instructions=FREEFORM_EXTRA)
            system = next(
                (m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT
            )
            system = (
                system.rstrip()
                + "\n\nCRITICAL: Follow the FREE-FORM REASONING PROTOCOL in the user "
                "message. Neighbor NIST hits are evidence about *neighbors*, not a "
                "direct library ID of the seed. Seed NIST is disabled. "
                "Seed node id is hidden; do not invent accessions.\n"
            )
            user = next((m["content"] for m in messages if m["role"] == "user"), "")
            user = user.replace(
                "=== SPECTRAL LIBRARY SEARCH (e.g. NIST; independent of network names) ===",
                "=== SPECTRAL LIBRARY SEARCH (SEED) — DISABLED BY PROTOCOL ===\n"
                "  (seed was NOT reverse-searched against NIST)\n"
                "=== END SEED LIBRARY (empty) ===\n"
                "=== SPECTRAL LIBRARY SEARCH (e.g. NIST; independent of network names) ===",
            )
            user = user.rstrip() + "\n" + _format_neighbor_nist_block(neighbor_nist) + "\n"
            user = (
                user.rstrip()
                + "\n\n=== PROTOCOL FOOTER (re-read before answering) ===\n"
                + FREEFORM_EXTRA
                + "\n"
            )
            system = sanitize_text(system)
            user = sanitize_text(user)

            job = {
                "spectrum_id": sample_id,
                "hnsw_index": hnsw_index,
                "seed_mz": ego.seed_mz,
                "seed_ion_mode": seed_ion,
                "seed_node_id": SEED_NODE_ID,
                "n_neighbors": len(ego.neighbors),
                "msms_used": bool(ego.spectral and ego.spectral.seed),
                "protocol": "strict_blind_freeform_full11540_nist_neighbors",
                "strict_blind": True,
                "freeform_required": True,
                "n_library_hits": 0,
                "n_neighbor_nist_hits": n_hits_total,
                "n_neighbors_nist_searched": len(neighbor_nist),
                "system_prompt": system,
                "user_prompt": user,
                "blind": True,
                "seed_nist_disabled": True,
                "method_card": method.to_dict(),
                "library_hits": [],
                "neighbor_library_hits": neighbor_nist,
            }
            (hand / "jobs" / f"{sample_id}.json").write_text(
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
            (hand / "prompts" / f"{sample_id}.txt").write_text(
                prompt_txt, encoding="utf-8"
            )

            public_rows.append(
                {
                    "spectrum_id": sample_id,
                    "hnsw_index": hnsw_index,
                    "precursor_mz": truth.get("precursor_mz") or ego.seed_mz,
                    "adduct": truth.get("adduct"),
                    "instrument_type": truth.get("instrument_type"),
                    "collision_energy": truth.get("collision_energy"),
                }
            )
            sealed_rows.append(
                {
                    "spectrum_id": sample_id,
                    "hnsw_index": hnsw_index,
                    "hnsw_stem": stem,
                    "msg_identifier": truth.get("identifier"),
                    "fold": fold,
                    "true_smiles": truth.get("smiles"),
                    "true_inchikey": truth.get("inchikey"),
                    "true_formula": truth.get("formula"),
                    "precursor_mz": truth.get("precursor_mz"),
                    "adduct": truth.get("adduct"),
                    "instrument_type": truth.get("instrument_type"),
                    "collision_energy": truth.get("collision_energy"),
                    "parent_mass": truth.get("parent_mass"),
                    "seed_mz_graphml": ego.seed_mz,
                    "n_neighbor_nist_hits": n_hits_total,
                }
            )
            n_ok += 1
            if i % 25 == 0 or i == 1:
                elapsed = time.perf_counter() - t0
                rate = n_ok / elapsed if elapsed > 0 else 0
                eta = (len(graphmls) - i) / rate / 60 if rate > 0 else float("nan")
                print(
                    f"[{i}/{len(graphmls)}] {sample_id} fold={fold} "
                    f"neigh_nist={n_hits_total} ok={n_ok} fail={n_fail} "
                    f"rate={rate:.2f}/s eta_min={eta:.1f}",
                    flush=True,
                )
        except Exception as e:
            n_fail += 1
            errors.append(f"{stem}: {type(e).__name__}: {e}")
            print(f"[{i}] FAIL {stem}: {type(e).__name__}: {e}", flush=True)

    # write manifests
    pub_fields = [
        "spectrum_id",
        "hnsw_index",
        "precursor_mz",
        "adduct",
        "instrument_type",
        "collision_energy",
    ]
    with (hand / "sample_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pub_fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(public_rows, key=lambda x: int(x["hnsw_index"])):
            w.writerow(r)

    seal_fields = [
        "spectrum_id",
        "hnsw_index",
        "hnsw_stem",
        "msg_identifier",
        "fold",
        "true_smiles",
        "true_inchikey",
        "true_formula",
        "precursor_mz",
        "adduct",
        "instrument_type",
        "collision_energy",
        "parent_mass",
        "seed_mz_graphml",
        "n_neighbor_nist_hits",
    ]
    with (sealed / "truth_index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=seal_fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(sealed_rows, key=lambda x: int(x["hnsw_index"])):
            w.writerow(r)

    # fold summary among sealed
    fc = {}
    for r in sealed_rows:
        fc[r.get("fold") or "?"] = fc.get(r.get("fold") or "?", 0) + 1

    comparison = {
        "n_ego_networks_built_for": len(sealed_rows),
        "n_ego_networks_on_disk_source": 11540,
        "msg_full_spectra": 231104,
        "msg_test_spectra": 17556,
        "msg_val_spectra": 19429,
        "msg_train_spectra": 194119,
        "folds_in_this_hnsw_block": fc,
        "match_full_msg_test": False,
        "note": (
            "Ego nets are HNSW_spectrum_219564..231103 = last 11540 of full MassSpecGym.mgf "
            "file order. Official test fold has 17556 spectra. This block is almost all TRAIN "
            "(~11386 train / ~90 val / ~64 test). NOT equal to full MSG test set."
        ),
    }

    meta = {
        "pack": str(hand),
        "sealed_truth": str(sealed),
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_fail": n_fail,
        "n_manifest": len(public_rows),
        "protocol": "strict_blind_freeform_full11540_nist_neighbors",
        "seed_node_id": SEED_NODE_ID,
        "nist": not args.no_nist,
        "nist_neighbors": args.nist_neighbors,
        "max_neighbors": args.max_neighbors,
        "fold_counts": fc,
        "comparison_vs_msg_test": comparison,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "errors_head": errors[:50],
        "n_errors": len(errors),
    }
    (hand / "package_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (sealed / "package_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (hand / "COMPARE_EGO_VS_MSG_TEST.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    cache_p.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    start = f"""# START HERE — MSG HNSW full block free-form + **neighbor-only NIST**

## Scale
- **n = {len(public_rows)}** prompts (spectrum_id `MSGFULL######`)
- Protocol: strict blind free-form; **seed NIST OFF**; **neighbor NIST ON** (precomputed)
- Seed node always `9999999` (hidden in prompts)

## IMPORTANT vs official MSG test set
| Set | n |
|-----|--:|
| Official MassSpecGym **test** fold | **17556** |
| Ego networks in this handout (HNSW 219564–231103) | **{len(public_rows)}** |
| Of which fold=test | **{fc.get('test', 0)}** |
| fold=val | **{fc.get('val', 0)}** |
| fold=train | **{fc.get('train', 0)}** |

**They do NOT match.** This HNSW dump is the **last 11540 spectra of the full MGF file order**, mostly **train**, not the full test fold.

See `COMPARE_EGO_VS_MSG_TEST.json`.

## Read / write
| Path | Role |
|------|------|
| `sample_manifest.csv` | All IDs (no structures) |
| `prompts/<ID>.txt` | Full free-form + neighbor NIST |
| `jobs/<ID>.json` | Same as structured system/user |
| `predictions/` | **Write only here** |

## Task (Grok)
For every `spectrum_id` in `sample_manifest.csv`:
1. Read `prompts/<ID>.txt` fully (one sample at a time).
2. Free-form MASS → NETWORK → MS/MS → CHEMISTRY → DECISION.
3. Neighbor NIST = evidence about **neighbors** only.
4. Write `predictions/<ID>.json` with model `grok-msg-full11540-nist-neigh-freeform`.

## HARD BANS
- Pack-wide SMILES index / bulk mass-fit scripts
- Sealed truth / external registry
- Treating neighbor NIST as automatic seed ID without mass gate

## Cost note
Prior Opus free-form on long prompts ≈ 100k+ tokens/sample. Full free-form on {len(public_rows)} samples is expensive — quality over speed; resume by skipping existing prediction JSON.

## Paste
```
MSG HNSW full-block free-form + neighbor-only NIST.
Folder: this handout
Read prompts/<ID>.txt; write predictions/<ID>.json
model = grok-msg-full11540-nist-neigh-freeform
Seed NIST off; neighbor NIST on (in prompt).
No pack-wide index. One sample at a time. Resume on existing JSON.
```
"""
    (hand / "START_HERE.md").write_text(start, encoding="utf-8")
    (hand / "README.md").write_text(
        "Open START_HERE.md. Free-form + neighbor NIST handout. "
        "Not equal to full MSG test (see COMPARE_EGO_VS_MSG_TEST.json).\n",
        encoding="utf-8",
    )

    print(f"[done] {json.dumps(meta, indent=2)}", flush=True)
    print(f"[compare] {json.dumps(comparison, indent=2)}", flush=True)
    return 0 if n_fail == 0 or n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
