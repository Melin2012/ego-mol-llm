#!/usr/bin/env python3
"""
Split MassSpecGym.mgf by official FOLD= field (train / val / test).

Example
-------
  python scripts/split_massspecgym_mgf.py ^
    --input C:\\Users\\AlexeyMelnik\\Downloads\\MassSpecGym.mgf ^
    --out-dir C:\\Users\\AlexeyMelnik\\Downloads\\MassSpecGym_split
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path


def split_mgf(src: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    fold_counts: Counter[str] = Counter()
    writers: dict[str, object] = {}
    index_rows: list[dict] = []
    n_spec = 0
    n_peaks_total = 0
    current: list[str] = []
    meta: dict[str, str] = {}
    in_spec = False
    peak_n = 0

    def flush() -> None:
        nonlocal n_spec, n_peaks_total, current, meta, peak_n
        if not current:
            return
        fold = (meta.get("FOLD") or "").strip().lower()
        if fold not in ("train", "val", "test"):
            fold = "missing" if not fold else "other"
        fold_counts[fold] += 1
        if fold not in writers:
            path = out_dir / f"MassSpecGym_{fold}.mgf"
            writers[fold] = path.open("w", encoding="utf-8", newline="\n")
        w = writers[fold]
        for line in current:
            w.write(line if line.endswith("\n") else line + "\n")
        n_spec += 1
        n_peaks_total += peak_n
        index_rows.append(
            {
                "identifier": meta.get("IDENTIFIER") or meta.get("TITLE") or "",
                "fold": fold,
                "smiles": meta.get("SMILES") or "",
                "inchikey": meta.get("INCHIKEY") or "",
                "formula": meta.get("FORMULA") or "",
                "precursor_mz": meta.get("PRECURSOR_MZ") or meta.get("PEPMASS") or "",
                "adduct": meta.get("ADDUCT") or "",
                "instrument_type": meta.get("INSTRUMENT_TYPE") or "",
                "collision_energy": meta.get("COLLISION_ENERGY") or "",
                "parent_mass": meta.get("PARENT_MASS") or "",
                "n_peaks": peak_n,
                "simulation_challenge": meta.get("SIMULATION_CHALLENGE") or "",
            }
        )
        current = []
        meta = {}
        peak_n = 0

    with src.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.upper() == "BEGIN IONS":
                in_spec = True
                current = [line if line.endswith("\n") else line + "\n"]
                meta = {}
                peak_n = 0
                continue
            if not in_spec:
                continue
            current.append(line if line.endswith("\n") else line + "\n")
            if s.upper() == "END IONS":
                flush()
                in_spec = False
                if n_spec % 50000 == 0 and n_spec:
                    print(f"... {n_spec} spectra", flush=True)
                continue
            if "=" in s and not (s[0].isdigit() or s.startswith(".")):
                k, _, v = s.partition("=")
                meta[k.strip().upper()] = v.strip()
            else:
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        float(parts[0])
                        float(parts[1])
                        peak_n += 1
                    except ValueError:
                        pass

    if current:
        flush()

    for w in writers.values():
        w.close()

    mols_by_fold: dict[str, set[str]] = {k: set() for k in fold_counts}
    for r in index_rows:
        key = (r["inchikey"] or "").split("-")[0].upper() or (r["smiles"] or "")
        if key:
            mols_by_fold.setdefault(r["fold"], set()).add(key)

    idx_path = out_dir / "MassSpecGym_spectrum_index.csv"
    with idx_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(index_rows[0].keys()) if index_rows else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(index_rows)

    summary = {
        "source": str(src.resolve()),
        "source_bytes": src.stat().st_size,
        "out_dir": str(out_dir.resolve()),
        "n_spectra": n_spec,
        "n_peaks_total": n_peaks_total,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "spectra_by_fold": dict(fold_counts),
        "unique_inchikey_or_smiles_by_fold": {k: len(v) for k, v in mols_by_fold.items()},
        "files": {fold: str((out_dir / f"MassSpecGym_{fold}.mgf").resolve()) for fold in fold_counts},
        "index_csv": str(idx_path.resolve()),
        "split_rule": "Official MassSpecGym FOLD= field (train/val/test); no custom random split",
    }
    (out_dir / "SPLIT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="Path to MassSpecGym.mgf")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input_parent>/MassSpecGym_split)",
    )
    args = ap.parse_args()
    src = args.input.resolve()
    if not src.is_file():
        print(f"ERROR: not found: {src}")
        return 1
    out_dir = (args.out_dir or (src.parent / "MassSpecGym_split")).resolve()
    summary = split_mgf(src, out_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
