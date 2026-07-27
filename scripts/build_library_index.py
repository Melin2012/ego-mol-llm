#!/usr/bin/env python3
"""Build precursor-binned spectral library index (NIST MGF → pickle)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ego_mol_llm.library_search import SpectralLibraryIndex, default_nist_paths


def main() -> int:
    ap = argparse.ArgumentParser(description="Build ego-mol-llm library index from MGF")
    default_mgf, default_idx = default_nist_paths(_REPO)
    ap.add_argument("--mgf", type=Path, default=default_mgf)
    ap.add_argument("--out", type=Path, default=default_idx)
    ap.add_argument("--max-spectra", type=int, default=0, help="0 = all")
    ap.add_argument("--top-peaks", type=int, default=40)
    args = ap.parse_args()

    if not args.mgf.is_file():
        print(f"[error] MGF not found: {args.mgf}", flush=True)
        return 1

    max_sp = args.max_spectra if args.max_spectra > 0 else None
    idx = SpectralLibraryIndex.build_from_mgf(
        args.mgf,
        top_n_peaks=args.top_peaks,
        max_spectra=max_sp,
        progress_every=25_000,
    )
    out = idx.save(args.out)
    print(f"[done] wrote {out} records={idx.n_records} bins={len(idx.bins)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
