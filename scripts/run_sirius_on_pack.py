#!/usr/bin/env python3
"""
Run SIRIUS (+ CSI:FingerID structure step) on every seed spectrum in a blind pack.

Writes under pack/sirius_work/<SPECTRUMID>/:
  - <id>.ms
  - <id>.sirius/          (project space, if CLI succeeds)
  - <id>_summaries/       (write-summaries output)
  - hits.json             (parsed structure/formula hits)

Also writes pack/sirius_hits/<SPECTRUMID>.json for product fusion / prompt refresh.

Requires:
  - SIRIUS 5/6 CLI on PATH, or --sirius-bin / $env:SIRIUS_BIN
  - ``sirius login`` for CSI:FingerID (academic free; commercial = Bright Giant)

Examples
--------
  # Dry-run: only write .ms files + show CLI that would run
  python scripts/run_sirius_on_pack.py --pack ... --dry-run

  # Full run (needs login)
  python scripts/run_sirius_on_pack.py --pack ... --sirius-bin "C:\\path\\to\\sirius.bat"

  # Parse already-computed project trees without re-running CLI
  python scripts/run_sirius_on_pack.py --pack ... --parse-existing-only
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

from ego_mol_llm.mgf import build_spectral_context, parse_mgf
from ego_mol_llm.sirius import find_sirius_binary, identify_spectrum


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--sirius-bin", type=Path, default=None)
    ap.add_argument("--profile", default="orbitrap", help="SIRIUS instrument profile")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--limit", type=int, default=0, help="Max spectra (0 = all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parse-existing-only", action="store_true")
    ap.add_argument(
        "--no-structure",
        action="store_true",
        help="Formula-only (skip CSI:FingerID structure step)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if hits.json already exists",
    )
    args = ap.parse_args()

    pack = args.pack.resolve()
    manifest = pack / "sample_manifest.csv"
    if not manifest.is_file():
        print(f"[error] missing {manifest}", flush=True)
        return 1

    bin_path = find_sirius_binary(args.sirius_bin)
    if not args.parse_existing_only and not args.dry_run and bin_path is None:
        print(
            "[error] SIRIUS CLI not found.\n"
            "  Install SIRIUS 5/6, then either:\n"
            "    set SIRIUS_BIN=C:\\path\\to\\sirius.bat\n"
            "    or pass --sirius-bin ...\n"
            "  Login once:  sirius login\n"
            "  Or use --dry-run / --parse-existing-only.",
            flush=True,
        )
        return 2
    print(f"[info] sirius_bin={bin_path}", flush=True)

    work_root = pack / "sirius_work"
    hits_root = pack / "sirius_hits"
    work_root.mkdir(exist_ok=True)
    hits_root.mkdir(exist_ok=True)

    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    n_ok = n_hits = n_skip = n_fail = 0
    t0 = time.perf_counter()

    for i, row in enumerate(rows, 1):
        sid = (row.get("spectrum_id") or "").strip().upper()
        if not sid:
            continue
        out_hits = hits_root / f"{sid}.json"
        if out_hits.is_file() and not args.force and not args.parse_existing_only:
            # allow re-parse of existing work even without --force when parse-only
            print(f"[{i}/{len(rows)}] {sid} skip existing hits (--force to redo)", flush=True)
            n_skip += 1
            continue

        seed_mgf = pack / "seed_mgfs_blind" / f"{sid}.mgf"
        if not seed_mgf.is_file():
            print(f"[{i}] {sid} missing seed MGF", flush=True)
            n_fail += 1
            continue

        spectra = parse_mgf(seed_mgf)
        if not spectra or not spectra[0].peaks:
            print(f"[{i}] {sid} empty spectrum", flush=True)
            n_fail += 1
            continue
        sp = spectra[0]
        # ion mode via spectral context helper
        ctx = build_spectral_context(
            seed_id="0",
            seed_mz=sp.pepmass,
            neighbor_ids=[],
            seed_mgf=seed_mgf,
        )
        ion = ctx.seed_ion_mode
        mz = float(sp.pepmass or 0.0)
        work = work_root / sid

        try:
            hits, meta = identify_spectrum(
                compound_id=sid,
                precursor_mz=mz,
                peaks=list(sp.peaks),
                work_dir=work,
                ion_mode=ion,
                sirius_bin=bin_path,
                top_k=args.top_k,
                profile=args.profile,
                no_structure=args.no_structure,
                timeout_s=args.timeout,
                dry_run=args.dry_run,
                parse_existing_only=args.parse_existing_only,
            )
        except Exception as e:
            print(f"[{i}] {sid} FAIL {type(e).__name__}: {e}", flush=True)
            n_fail += 1
            continue

        payload = {
            "spectrum_id": sid,
            "precursor_mz": mz,
            "ion_mode": ion,
            "n_hits": len(hits),
            "hits": [h.to_dict() for h in hits],
            "meta": meta,
            "product_version": "0.3",
        }
        # always write work-local copy
        (work / "hits.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not args.dry_run:
            out_hits.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        n_ok += 1
        n_hits += len(hits)
        top = hits[0].smiles if hits else None
        status = "dry_run" if args.dry_run else ("ok" if hits else "no_hits")
        print(
            f"[{i}/{len(rows)}] {sid} {status} ion={ion} n_hits={len(hits)} "
            f"top={(top or '')[:50]}",
            flush=True,
        )
        if meta.get("run") and not meta["run"].get("ok") and not args.dry_run:
            err = meta["run"].get("error") or meta["run"].get("stderr", "")[:200]
            if err:
                print(f"    run_note: {err}", flush=True)

    elapsed = time.perf_counter() - t0
    summary = {
        "pack": str(pack),
        "n_rows": len(rows),
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_fail": n_fail,
        "n_hits_total": n_hits,
        "elapsed_s": round(elapsed, 2),
        "sirius_bin": str(bin_path) if bin_path else None,
        "dry_run": args.dry_run,
        "parse_existing_only": args.parse_existing_only,
    }
    (pack / "sirius_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[done] {json.dumps(summary)}", flush=True)
    return 0 if n_fail == 0 or n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
