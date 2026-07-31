#!/usr/bin/env python3
"""
Refresh blind pack jobs/prompts with NIST used **only for neighbors**, never the seed.

Protocol
--------
- Seed (central node): NO NIST reverse search (library_hits = []).
- Neighbors: reverse-search each neighbor's experimental MS/MS against NIST;
  attach hits as structure clues for propagation (not a direct library ID of the seed).
- GraphML neighbor SMILES still kept.
- Optionally re-append precomputed CFM / SIRIUS fragment blocks from
  pack/cfm_explain and pack/sirius_explain.

Writes under pack/jobs_nist_neighbors/ and pack/prompts_nist_neighbors/ by default
(so original jobs/ are preserved). Use --in-place to overwrite jobs/prompts.
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
from ego_mol_llm.library_search import default_nist_paths, load_library_index
from ego_mol_llm.method_card import MethodCard, resolve_method_polarity
from ego_mol_llm.mgf import build_spectral_context
from ego_mol_llm.prompts import SYSTEM_PROMPT, build_messages

FABLE_EXTRA = """
TASK RULES (NIST for neighbors only — seed is NOT library-searched):
- Center spectrum is unlabeled; do NOT treat any seed NIST hit as available
  (there are none by design).
- Neighbor NIST hits are independent spectral IDs of *neighbors*, usable as
  network structure evidence like GraphML annotations.
- Prefer mass-consistent SMILES for the seed (from neighbors / optional SIRIUS/CFM).
- Output one JSON object only.
"""

# Same free-form contract as MSG strict v2 packs (quality / no pack-wide index)
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
  "model": "<your model name>",
  "blind": true,
  "method": "freeform_llm_reasoning",
  "evidence": "per_sample_prompt_only+neighbor_nist",
  "no_packwide_index": true
}

FORBIDDEN rationale language: "pack-wide", "packwide", "global SMILES index",
"harvested from other samples", "mass-transfer batch".
=== END FREE-FORM PROTOCOL ===
"""


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
        with urllib.request.urlopen(url, timeout=10) as r:
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


def _append_precomputed_blocks(user: str, pack: Path, sid: str) -> str:
    """Re-attach CFM / SIRIUS fragment blocks if JSON precomputes exist."""
    # CFM
    cfm_p = pack / "cfm_explain" / f"{sid}.json"
    if cfm_p.is_file():
        try:
            from ego_mol_llm.cfm_network_explain import format_cfm_explanation_for_prompt

            expl = json.loads(cfm_p.read_text(encoding="utf-8"))
            block = "\n".join(format_cfm_explanation_for_prompt(expl))
            marker = "=== CFM-ID / IN-SILICO FRAGMENT EXPLANATION"
            if marker in user:
                pre, _, rest = user.partition(marker)
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
                user = pre.rstrip() + "\n" + block + "\n" + "\n".join(lines[cut:]).lstrip("\n")
            else:
                user = user.rstrip() + "\n" + block + "\n"
        except Exception:
            pass
    # SIRIUS fragments
    srx_p = pack / "sirius_explain" / f"{sid}.json"
    if srx_p.is_file():
        try:
            from ego_mol_llm.sirius_network_explain import (
                format_sirius_fragment_explanation_for_prompt,
            )

            expl = json.loads(srx_p.read_text(encoding="utf-8"))
            block = "\n".join(format_sirius_fragment_explanation_for_prompt(expl))
            marker = "=== SIRIUS FRAGMENT EXPLANATION"
            if marker in user:
                pre, _, rest = user.partition(marker)
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
                user = pre.rstrip() + "\n" + block + "\n" + "\n".join(lines[cut:]).lstrip("\n")
            else:
                user = user.rstrip() + "\n" + block + "\n"
        except Exception:
            pass
    return user


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--max-neighbors", type=int, default=50)
    ap.add_argument("--nist-neighbors", type=int, default=8, help="Max neighbors to NIST-search")
    ap.add_argument("--library-top-k", type=int, default=5)
    ap.add_argument("--library-index", type=Path, default=None)
    ap.add_argument("--in-place", action="store_true", help="Overwrite jobs/ and prompts/")
    ap.add_argument("--no-resolve-smiles", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--seed-id",
        default=None,
        help="GraphML seed node id (default: package_meta.seed_node_id or 0)",
    )
    ap.add_argument(
        "--freeform",
        action="store_true",
        help="Bake free-form reasoning protocol (MSG strict / Grok-Claude style)",
    )
    ap.add_argument(
        "--study-id",
        default=None,
        help="Method card study_id (default from package_meta or nist_neighbors_only)",
    )
    args = ap.parse_args()

    pack = args.pack.resolve()
    meta: dict = {}
    meta_p = pack / "package_meta.json"
    if meta_p.is_file():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    seed_id = str(args.seed_id or meta.get("seed_node_id") or "0")
    freeform = bool(args.freeform or meta.get("freeform_baked_into_prompts"))
    study_id = args.study_id or meta.get("protocol") or "nist_neighbors_only"
    protocol = (
        "strict_blind_freeform_v2_nist_neighbors"
        if freeform
        else "nist_neighbors_only"
    )
    extra = FREEFORM_EXTRA if freeform else FABLE_EXTRA

    rows = list(
        csv.DictReader((pack / "sample_manifest.csv").open(encoding="utf-8", newline=""))
    )
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    idx_path = args.library_index
    if idx_path is None:
        _, idx_path = default_nist_paths(_REPO)
    print(f"[info] loading library {idx_path}", flush=True)
    print(
        f"[info] pack={pack} seed_id={seed_id} freeform={freeform} protocol={protocol}",
        flush=True,
    )
    lib = load_library_index(idx_path)
    print(f"[info] n_records={lib.n_records}", flush=True)

    if args.in_place:
        jobs_out = pack / "jobs"
        prompts_out = pack / "prompts"
    else:
        jobs_out = pack / "jobs_nist_neighbors"
        prompts_out = pack / "prompts_nist_neighbors"
    jobs_out.mkdir(exist_ok=True)
    prompts_out.mkdir(exist_ok=True)

    method_template = MethodCard(
        chromatography="RP-C18",
        polarity="both",
        ionization="ESI",
        gradient="aqueous_to_organic",
        study_id=study_id,
        notes="NIST for neighbors only; seed not library-searched"
        + ("; freeform protocol baked" if freeform else ""),
    )

    cache_p = pack / "inchikey_smiles_cache.json"
    cache: dict[str, str | None] = {}
    if cache_p.is_file():
        try:
            cache = json.loads(cache_p.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    n_ok = 0
    t0 = time.perf_counter()
    for i, row in enumerate(rows, 1):
        sid = (row.get("spectrum_id") or "").strip().upper()
        gpath = pack / "graphml" / f"{sid}.graphml"
        mgf = pack / "subgraph_mgfs" / f"{sid}.mgf"
        seed = pack / "seed_mgfs_blind" / f"{sid}.mgf"
        if not gpath.is_file():
            print(f"[{i}] skip missing graphml {sid}", flush=True)
            continue
        try:
            net = load_graphml(gpath)
            ego = build_ego(
                net,
                seed_id=seed_id,
                hide_seed_name=True,
                max_neighbors=args.max_neighbors,
                include_two_hop=True,
            )
            ego.seed.name = None
            ego.seed.smiles = None
            ego.spectral = build_spectral_context(
                seed_id=str(ego.seed.id),
                seed_mz=ego.seed_mz,
                neighbor_ids=[e.node.id for e in ego.neighbors],
                mgf_paths=[mgf] if mgf.is_file() else [],
                seed_mgf=seed if seed.is_file() else None,
            )
            seed_ion = getattr(ego.spectral, "seed_ion_mode", None) if ego.spectral else None
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

            # SEED: explicitly no NIST
            ego.meta["library_hits"] = []
            ego.meta["seed_nist_disabled"] = True
            ego.meta["method_card"] = method.to_dict()
            ego.meta["seed_ion_mode"] = seed_ion
            ego.meta["library_n_records"] = lib.n_records
            ego.meta["protocol"] = protocol
            ego.meta["seed_node_id"] = seed_id

            # NEIGHBORS: NIST reverse search on experimental peaks
            msms = (ego.spectral.neighbor_msms_cosine or {}) if ego.spectral else {}
            npeaks = (ego.spectral.neighbor_peaks or {}) if ego.spectral else {}
            nmeta = (ego.spectral.neighbor_meta or {}) if ego.spectral else {}

            def nkey(ev):
                nid = str(ev.node.id)
                return (
                    0 if (npeaks.get(nid) or []) else 1,
                    -(float(msms.get(nid) or 0.0)),
                    -(float(ev.cosine or 0.0)),
                )

            ranked_ev = sorted(ego.neighbors, key=nkey)[: args.nist_neighbors]
            neighbor_nist: list[dict] = []
            n_hits_total = 0
            for ev in ranked_ev:
                nid = str(ev.node.id)
                peaks = npeaks.get(nid) or []
                if not peaks:
                    continue
                ion = (nmeta.get(nid) or {}).get("ion_mode") or seed_ion or pol
                if ion not in {"positive", "negative"}:
                    ion = None
                pep = ev.node.mz
                hits = lib.search(
                    peaks,
                    pep,
                    top_k=args.library_top_k,
                    precursor_tol_da=0.05,
                    ion_mode=ion,
                )
                hit_dicts = []
                for h in hits:
                    d = h.to_dict()
                    if not d.get("smiles") and d.get("inchikey") and not args.no_resolve_smiles:
                        smi = _inchikey_to_smiles(str(d["inchikey"]).split()[0], cache)
                        if smi:
                            d["smiles"] = smi
                            d["smiles_resolved_from_inchikey"] = True
                    hit_dicts.append(d)
                n_hits_total += len(hit_dicts)
                neighbor_nist.append(
                    {
                        "neighbor_id": nid,
                        "neighbor_name": ev.node.name,
                        "neighbor_smiles_graphml": ev.node.smiles,
                        "neighbor_mz": ev.node.mz,
                        "msms_cosine": msms.get(nid),
                        "edge_cosine": ev.cosine,
                        "hits": hit_dicts,
                    }
                )

            ego.meta["neighbor_library_hits"] = neighbor_nist

            messages = build_messages(ego, extra_instructions=extra)
            system = next(
                (m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT
            )
            if freeform:
                system = (
                    system.rstrip()
                    + "\n\nCRITICAL: Follow the FREE-FORM REASONING PROTOCOL in the user "
                    "message. Neighbor NIST hits are evidence about *neighbors*, not a "
                    "direct library ID of the seed. Seed NIST is disabled.\n"
                )
            user = next((m["content"] for m in messages if m["role"] == "user"), "")
            # Ensure seed library section is empty / explicit
            user = user.replace(
                "=== SPECTRAL LIBRARY SEARCH (e.g. NIST; independent of network names) ===",
                "=== SPECTRAL LIBRARY SEARCH (SEED) — DISABLED BY PROTOCOL ===\n"
                "  (seed was NOT reverse-searched against NIST)\n"
                "=== END SEED LIBRARY (empty) ===\n"
                "=== SPECTRAL LIBRARY SEARCH (e.g. NIST; independent of network names) ===",
            )
            # inject neighbor NIST block after method / library area
            user = user.rstrip() + "\n" + _format_neighbor_nist_block(neighbor_nist) + "\n"
            user = _append_precomputed_blocks(user, pack, sid)
            if freeform:
                # Bake freeform twice (extra already in build_messages + footer)
                user = (
                    user.rstrip()
                    + "\n\n=== PROTOCOL FOOTER (re-read before answering) ===\n"
                    + FREEFORM_EXTRA
                    + "\n"
                )

            job = {
                "spectrum_id": sid,
                "seed_mz": ego.seed_mz,
                "n_neighbors": len(ego.neighbors),
                "msms_used": bool(ego.spectral and ego.spectral.seed),
                "seed_rt": getattr(ego.spectral, "seed_rt", None) if ego.spectral else None,
                "seed_ion_mode": seed_ion,
                "seed_node_id": seed_id,
                "max_neighbors": args.max_neighbors,
                "protocol": protocol,
                "product_version": "0.3-nist-neighbors-only-freeform"
                if freeform
                else "0.2.1-nist-neighbors-only",
                "strict_blind": True,
                "freeform_required": freeform,
                "n_library_hits": 0,  # seed
                "n_neighbor_nist_hits": n_hits_total,
                "n_neighbors_nist_searched": len(neighbor_nist),
                "system_prompt": system,
                "user_prompt": user,
                "blind": True,
                "method_card": method.to_dict(),
                "library_hits": [],  # seed empty
                "neighbor_library_hits": neighbor_nist,
                "seed_nist_disabled": True,
                "files": {
                    "graphml": f"graphml/{sid}.graphml",
                    "network_mgf": f"subgraph_mgfs/{sid}.mgf",
                    "seed_mgf": f"seed_mgfs_blind/{sid}.mgf",
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
                "Reply with one JSON object only.\n"
            )
            (prompts_out / f"{sid}.txt").write_text(prompt_txt, encoding="utf-8")
            n_ok += 1
            print(
                f"[{i}/{len(rows)}] {sid} seed_nist=0 neigh_nist={n_hits_total} "
                f"n_searched={len(neighbor_nist)}",
                flush=True,
            )
        except Exception as e:
            print(f"[{i}] FAIL {sid}: {type(e).__name__}: {e}", flush=True)

    cache_p.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    summary = {
        "n_ok": n_ok,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "protocol": protocol,
        "seed_id": seed_id,
        "freeform": freeform,
        "jobs_out": str(jobs_out),
        "prompts_out": str(prompts_out),
        "library_index": str(idx_path),
    }
    (jobs_out / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {summary}", flush=True)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
