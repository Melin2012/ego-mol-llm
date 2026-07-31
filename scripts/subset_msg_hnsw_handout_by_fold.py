#!/usr/bin/env python3
"""
Subset MSG HNSW full handout by official fold (no IK1 dedupe).

Default: **test only**, keeping the original fold count (n=64 in the HNSW block).

Source
------
  Desktop/MSG_HNSW_full11540_nist_neigh_handout/
  Desktop/MSG_HNSW_full11540_nist_neigh_SEALED_truth/truth_index.csv

Outputs (default fold=test)
---------------------------
  Desktop/MSG_HNSW_test64_nist_neigh_handout/
  Desktop/MSG_HNSW_test64_nist_neigh_SEALED_truth/

Copies prompts/jobs/seed_mgfs; does not re-run NIST.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


def hnsw_idx(row: dict) -> int:
    try:
        return int(row.get("hnsw_index") or 10**12)
    except Exception:
        return 10**12


def copy_sample_files(src_hand: Path, dst_hand: Path, spectrum_id: str) -> dict[str, bool]:
    ok = {}
    for sub, ext in (("prompts", ".txt"), ("jobs", ".json"), ("seed_mgfs", ".mgf")):
        s = src_hand / sub / f"{spectrum_id}{ext}"
        d = dst_hand / sub / f"{spectrum_id}{ext}"
        if s.is_file():
            shutil.copy2(s, d)
            ok[sub] = True
        else:
            ok[sub] = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fold",
        default="test",
        choices=("train", "val", "test"),
        help="Official MSG fold to keep (default: test)",
    )
    ap.add_argument(
        "--src-handout",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_full11540_nist_neigh_handout"),
    )
    ap.add_argument(
        "--src-sealed",
        type=Path,
        default=Path(
            r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_full11540_nist_neigh_SEALED_truth"
        ),
    )
    ap.add_argument(
        "--out-handout",
        type=Path,
        default=None,
        help="Default: Desktop/MSG_HNSW_{fold}{n}_nist_neigh_handout",
    )
    ap.add_argument(
        "--out-sealed",
        type=Path,
        default=None,
        help="Default: Desktop/MSG_HNSW_{fold}{n}_nist_neigh_SEALED_truth",
    )
    args = ap.parse_args()

    src_hand = args.src_handout.resolve()
    src_sealed = args.src_sealed.resolve()
    truth_path = src_sealed / "truth_index.csv"
    if not truth_path.is_file():
        print(f"[error] missing {truth_path}", file=sys.stderr)
        return 1
    if not (src_hand / "prompts").is_dir():
        print(f"[error] missing prompts in {src_hand}", file=sys.stderr)
        return 1

    fold = args.fold.strip().lower()
    rows_all = list(csv.DictReader(truth_path.open(encoding="utf-8-sig", newline="")))
    kept = [r for r in rows_all if (r.get("fold") or "").strip().lower() == fold]
    kept.sort(key=hnsw_idx)
    n = len(kept)
    if n == 0:
        print(f"[error] no rows with fold={fold}", file=sys.stderr)
        return 1

    desk = Path(r"C:\Users\AlexeyMelnik\Desktop")
    out_hand = (args.out_handout or desk / f"MSG_HNSW_{fold}{n}_nist_neigh_handout").resolve()
    out_sealed = (
        args.out_sealed or desk / f"MSG_HNSW_{fold}{n}_nist_neigh_SEALED_truth"
    ).resolve()

    print(f"[info] fold={fold} kept={n} / parent={len(rows_all)}", flush=True)
    print(f"[info] out_handout={out_hand}", flush=True)
    print(f"[info] out_sealed={out_sealed}", flush=True)

    for d in (
        out_hand,
        out_hand / "prompts",
        out_hand / "jobs",
        out_hand / "seed_mgfs",
        out_hand / "predictions",
        out_sealed,
    ):
        d.mkdir(parents=True, exist_ok=True)

    missing = []
    for i, r in enumerate(kept, 1):
        sid = r["spectrum_id"]
        ok = copy_sample_files(src_hand, out_hand, sid)
        if not all(ok.values()):
            missing.append({"spectrum_id": sid, "ok": ok})
        if i % 20 == 0 or i == n:
            print(f"[copy] {i}/{n}", flush=True)

    # sealed truth (full rows; no structure strip — sealed only)
    fieldnames = list(rows_all[0].keys())
    for extra in ("subset_policy", "source_pack_n"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    sealed_rows = []
    for r in kept:
        rr = dict(r)
        rr["subset_policy"] = f"official_fold={fold}; no_ik1_dedupe; original_fold_count"
        rr["source_pack_n"] = str(len(rows_all))
        sealed_rows.append(rr)

    with (out_sealed / "truth_index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(sealed_rows)

    ids = [r["spectrum_id"] for r in kept]
    (out_hand / f"ids_{fold}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (out_sealed / f"ids_{fold}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (out_sealed / "selection.json").write_text(
        json.dumps(
            {
                "fold": fold,
                "n": n,
                "n_parent": len(rows_all),
                "dedupe": False,
                "policy": "all spectra with official fold; original count preserved",
                "spectrum_ids": ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    man_fields = [
        "spectrum_id",
        "hnsw_index",
        "precursor_mz",
        "adduct",
        "instrument_type",
        "collision_energy",
        "fold",
    ]
    with (out_hand / "sample_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=man_fields, extrasaction="ignore")
        w.writeheader()
        for r in kept:
            w.writerow({k: r.get(k, "") for k in man_fields})

    meta = {
        "pack": str(out_hand),
        "sealed_truth": str(out_sealed),
        "parent_pack": str(src_hand),
        "parent_sealed": str(src_sealed),
        "fold": fold,
        "n": n,
        "n_parent": len(rows_all),
        "dedupe": False,
        "protocol": f"strict_blind_freeform_fold_{fold}_nist_neighbors",
        "seed_node_id": "9999999",
        "nist": True,
        "nist_seed": False,
        "nist_neighbors": True,
        "fold_counts": {fold: n},
        "note": (
            f"Official fold={fold} only from MSG HNSW full block 11540. "
            f"No IK1 dedupe — original fold count preserved (n={n}). "
            "Not the full official MSG test set (17556)."
        ),
        "n_missing_files": len(missing),
        "missing_files_head": missing[:20],
    }
    (out_hand / "package_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_sealed / "package_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    start = f"""# START HERE — MSG HNSW **{fold} fold only** + neighbor-only NIST

## Scale
- **n = {n}** spectra — **all official `{fold}` rows** from the HNSW block (indices 219564–231103)
- **No IK1 dedupe** — original fold count preserved (parent test was 64 → here **{n}** if fold=test)
- Protocol: strict blind free-form; **seed NIST OFF**; **neighbor NIST ON** (precomputed)
- Seed node always `9999999` (hidden in prompts)

## vs official MSG / parent packs
| Set | n |
|-----|--:|
| This handout (`fold={fold}`) | **{n}** |
| Parent HNSW block (all folds) | 11540 |
| Parent train / val / test | 11386 / 90 / 64 |
| Official full MSG **test** fold | 17556 |

**This is only the `{fold}` slice of the HNSW dump**, not the entire official MSG test set.

## Read / write
| Path | Role |
|------|------|
| `sample_manifest.csv` | All IDs (no structures) |
| `ids_{fold}.txt` | Same ID list |
| `prompts/<ID>.txt` | Full free-form + neighbor NIST |
| `jobs/<ID>.json` | Structured system/user |
| `predictions/` | **Write only here** |

## Task
For every `spectrum_id` in `sample_manifest.csv`:
1. Read `prompts/<ID>.txt` fully (one sample at a time).
2. Free-form MASS → NETWORK → MS/MS → CHEMISTRY → DECISION.
3. Neighbor NIST = evidence about **neighbors** only.
4. Write `predictions/<ID>.json`.

## HARD BANS
- Pack-wide SMILES index / bulk mass-fit scripts
- Sealed truth / external registry
- Treating neighbor NIST as automatic seed ID without mass gate

## Paste
```
MSG HNSW {fold}-only free-form + neighbor-only NIST (n={n}, no dedupe).
Folder: this handout
Read prompts/<ID>.txt; write predictions/<ID>.json
Seed NIST off; neighbor NIST on. One sample at a time. No pack-wide index.
```
"""
    (out_hand / "START_HERE.md").write_text(start, encoding="utf-8")
    (out_hand / "README.md").write_text(
        start
        + "\n## Build\n```\n"
        + f"python scripts/subset_msg_hnsw_handout_by_fold.py --fold {fold}\n"
        + "```\n",
        encoding="utf-8",
    )
    (out_sealed / "README.md").write_text(
        f"""# Sealed truth — fold={fold} only (n={n})

**OFFLINE ONLY — do not ship with model-facing zip.**

- `truth_index.csv` — {n} rows, no IK1 dedupe
- `ids_{fold}.txt`
- `selection.json`

Parent: `{src_sealed}`
""",
        encoding="utf-8",
    )

    print("[done] handout", out_hand, flush=True)
    print("[done] sealed ", out_sealed, flush=True)
    print("[done] n      ", n, "fold=", fold, flush=True)
    if missing:
        print(f"[warn] missing files for {len(missing)} samples", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
