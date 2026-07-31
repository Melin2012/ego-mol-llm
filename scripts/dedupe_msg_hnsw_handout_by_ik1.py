#!/usr/bin/env python3
"""
Dedupe MSG HNSW full handout to 1 spectrum per unique IK1, stratified by fold.

Policy
------
- Group by official fold (train / val / test) independently.
- Within each fold, keep **one** spectrum per true InChIKey first block (IK1).
- Tie-break: higher n_neighbor_nist_hits, then lower hnsw_index (stable).
- Molecules do not cross folds in this HNSW block (verified); still key by fold+IK1.

Source (default)
----------------
  Desktop/MSG_HNSW_full11540_nist_neigh_handout/
  Desktop/MSG_HNSW_full11540_nist_neigh_SEALED_truth/truth_index.csv

Outputs (default)
-----------------
  Desktop/MSG_HNSW_dedup_ik1_nist_neigh_handout/
  Desktop/MSG_HNSW_dedup_ik1_nist_neigh_SEALED_truth/

Does not re-run NIST: copies existing prompts/jobs/seed_mgfs for selected IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


def ik1_of(row: dict) -> str:
    ik = (row.get("true_inchikey") or "").strip().upper()
    if ik:
        return ik.split("-")[0]
    return ""


def nist_hits(row: dict) -> int:
    try:
        return int(float(row.get("n_neighbor_nist_hits") or 0))
    except Exception:
        return 0


def hnsw_idx(row: dict) -> int:
    try:
        return int(row.get("hnsw_index") or 10**12)
    except Exception:
        return 10**12


def select_deduped(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (kept_rows sorted by hnsw_index, drop_log rows)."""
    # fold -> ik1 -> best row
    best: dict[str, dict[str, dict]] = defaultdict(dict)
    drop_log: list[dict] = []

    # process in stable order so first-seen is deterministic before tie-break
    ordered = sorted(rows, key=lambda r: (r.get("fold") or "", hnsw_idx(r), r.get("spectrum_id") or ""))

    for r in ordered:
        fold = (r.get("fold") or "?").strip().lower()
        ik = ik1_of(r)
        if not ik:
            # keep orphans as unique keys so we don't silently drop
            ik = f"__no_ik1__{r.get('spectrum_id')}"
        cur = best[fold].get(ik)
        if cur is None:
            best[fold][ik] = r
            continue
        # prefer richer neighbor NIST, then lower index
        cand_key = (nist_hits(r), -hnsw_idx(r))
        cur_key = (nist_hits(cur), -hnsw_idx(cur))
        if cand_key > cur_key:
            drop_log.append(
                {
                    "dropped_spectrum_id": cur.get("spectrum_id"),
                    "kept_spectrum_id": r.get("spectrum_id"),
                    "fold": fold,
                    "ik1": ik,
                    "reason": "replaced_by_higher_nist_hits_or_lower_index",
                    "dropped_nist": nist_hits(cur),
                    "kept_nist": nist_hits(r),
                }
            )
            best[fold][ik] = r
        else:
            drop_log.append(
                {
                    "dropped_spectrum_id": r.get("spectrum_id"),
                    "kept_spectrum_id": cur.get("spectrum_id"),
                    "fold": fold,
                    "ik1": ik,
                    "reason": "duplicate_ik1_same_fold",
                    "dropped_nist": nist_hits(r),
                    "kept_nist": nist_hits(cur),
                }
            )

    kept: list[dict] = []
    for fold in ("train", "val", "test"):
        if fold in best:
            kept.extend(best[fold].values())
    # any other folds
    for fold, m in best.items():
        if fold not in ("train", "val", "test"):
            kept.extend(m.values())

    kept.sort(key=lambda r: hnsw_idx(r))
    return kept, drop_log


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
        default=Path(r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_dedup_ik1_nist_neigh_handout"),
    )
    ap.add_argument(
        "--out-sealed",
        type=Path,
        default=Path(
            r"C:\Users\AlexeyMelnik\Desktop\MSG_HNSW_dedup_ik1_nist_neigh_SEALED_truth"
        ),
    )
    ap.add_argument(
        "--also-fold-splits",
        action="store_true",
        help="Also write train/val/test subfolders under out-handout/by_fold/",
    )
    args = ap.parse_args()

    src_hand = args.src_handout.resolve()
    src_sealed = args.src_sealed.resolve()
    out_hand = args.out_handout.resolve()
    out_sealed = args.out_sealed.resolve()

    truth_path = src_sealed / "truth_index.csv"
    if not truth_path.is_file():
        print(f"[error] missing {truth_path}", file=sys.stderr)
        return 1
    if not (src_hand / "prompts").is_dir():
        print(f"[error] missing prompts in {src_hand}", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(truth_path.open(encoding="utf-8-sig", newline="")))
    print(f"[info] source spectra={len(rows)}", flush=True)
    print(f"[info] source folds={dict(Counter((r.get('fold') or '?').lower() for r in rows))}", flush=True)

    kept, drop_log = select_deduped(rows)
    fold_counts = Counter((r.get("fold") or "?").lower() for r in kept)
    print(f"[info] kept={len(kept)} dropped={len(drop_log)}", flush=True)
    print(f"[info] kept folds={dict(fold_counts)}", flush=True)

    # prepare dirs
    for d in (
        out_hand,
        out_hand / "prompts",
        out_hand / "jobs",
        out_hand / "seed_mgfs",
        out_hand / "predictions",
        out_sealed,
    ):
        d.mkdir(parents=True, exist_ok=True)

    # copy selected files
    missing = []
    for i, r in enumerate(kept, 1):
        sid = r["spectrum_id"]
        ok = copy_sample_files(src_hand, out_hand, sid)
        if not all(ok.values()):
            missing.append({"spectrum_id": sid, "ok": ok})
        if i % 200 == 0 or i == len(kept):
            print(f"[copy] {i}/{len(kept)}", flush=True)

    # sealed truth
    fieldnames = list(rows[0].keys())
    extra = ["ik1", "dedupe_policy", "source_pack_n"]
    for e in extra:
        if e not in fieldnames:
            fieldnames.append(e)

    sealed_rows = []
    for r in kept:
        rr = dict(r)
        rr["ik1"] = ik1_of(r)
        rr["dedupe_policy"] = "1_per_ik1_per_fold;prefer_max_n_neighbor_nist_hits;then_min_hnsw_index"
        rr["source_pack_n"] = str(len(rows))
        sealed_rows.append(rr)

    with (out_sealed / "truth_index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(sealed_rows)

    # drop log (sealed only — contains no structures of dropped beyond ids)
    drop_fields = [
        "dropped_spectrum_id",
        "kept_spectrum_id",
        "fold",
        "ik1",
        "reason",
        "dropped_nist",
        "kept_nist",
    ]
    with (out_sealed / "dedupe_drop_log.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=drop_fields)
        w.writeheader()
        w.writerows(drop_log)

    # selection map
    with (out_sealed / "dedupe_selection.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_source": len(rows),
                "n_kept": len(kept),
                "n_dropped": len(drop_log),
                "folds_kept": dict(fold_counts),
                "policy": "1 spectrum per IK1 per official fold; max neighbor NIST hits; min hnsw_index",
                "kept_spectrum_ids": [r["spectrum_id"] for r in kept],
            },
            f,
            indent=2,
        )

    # model-facing manifest (no structures)
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

    # fold-only id lists (handy for stratified scoring)
    for fold in ("train", "val", "test"):
        ids = [r["spectrum_id"] for r in kept if (r.get("fold") or "").lower() == fold]
        (out_hand / f"ids_{fold}.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        (out_sealed / f"ids_{fold}.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    if args.also_fold_splits:
        for fold in ("train", "val", "test"):
            sub = out_hand / "by_fold" / fold
            for d in (sub / "prompts", sub / "jobs", sub / "seed_mgfs", sub / "predictions"):
                d.mkdir(parents=True, exist_ok=True)
            fold_rows = [r for r in kept if (r.get("fold") or "").lower() == fold]
            with (sub / "sample_manifest.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=man_fields, extrasaction="ignore")
                w.writeheader()
                for r in fold_rows:
                    w.writerow({k: r.get(k, "") for k in man_fields})
                    sid = r["spectrum_id"]
                    for part, ext in (("prompts", ".txt"), ("jobs", ".json"), ("seed_mgfs", ".mgf")):
                        s = out_hand / part / f"{sid}{ext}"
                        d = sub / part / f"{sid}{ext}"
                        if s.is_file():
                            # hardlink if possible else copy
                            try:
                                if d.exists():
                                    d.unlink()
                                d.hardlink_to(s)
                            except Exception:
                                shutil.copy2(s, d)

    # package meta
    meta = {
        "pack": str(out_hand),
        "sealed_truth": str(out_sealed),
        "parent_pack": str(src_hand),
        "parent_sealed": str(src_sealed),
        "n_source": len(rows),
        "n_kept": len(kept),
        "n_dropped": len(drop_log),
        "n_missing_files": len(missing),
        "protocol": "strict_blind_freeform_dedup_ik1_nist_neighbors",
        "dedupe": {
            "key": "true_inchikey first block (IK1)",
            "scope": "per official fold (train/val/test)",
            "tie_break": "max n_neighbor_nist_hits, then min hnsw_index",
        },
        "seed_node_id": "9999999",
        "nist": True,
        "nist_seed": False,
        "nist_neighbors": True,
        "fold_counts": dict(fold_counts),
        "fold_counts_source": dict(Counter((r.get("fold") or "?").lower() for r in rows)),
        "note": (
            "1-per-IK1 dedupe of MSG HNSW full block 11540. "
            "Official folds preserved. Not full MSG test set. "
            "Prompts reused from parent pack (neighbor NIST precomputed)."
        ),
        "missing_files_head": missing[:20],
    }
    (out_hand / "package_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_sealed / "package_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # START_HERE + README
    n = len(kept)
    start = f"""# START HERE — MSG HNSW **IK1-deduped** free-form + neighbor-only NIST

## Scale
- **n = {n}** spectra (**1 unique molecule / IK1**, stratified by official fold)
- Parent pack: full HNSW block **11,540** → deduped here
- Protocol: strict blind free-form; **seed NIST OFF**; **neighbor NIST ON** (precomputed)
- Seed node always `9999999` (hidden in prompts)

## Folds after dedupe (1 spectrum per IK1 per fold)
| Fold | n (deduped) | n (parent spectra) |
|------|------------:|-------------------:|
| train | **{fold_counts.get('train', 0)}** | 11386 |
| val | **{fold_counts.get('val', 0)}** | 90 |
| test | **{fold_counts.get('test', 0)}** | 64 |
| **total** | **{n}** | 11540 |

Still **not** the full official MSG test fold (17,556). This is a unique-molecule subsample of the HNSW dump block (indices 219564–231103).

## Selection policy
- Within each official fold, keep **one** spectrum per true InChIKey first block (IK1).
- Prefer spectrum with more neighbor NIST hits; then lower HNSW index.
- No molecule appeared in multiple folds in the parent block.

## Read / write
| Path | Role |
|------|------|
| `sample_manifest.csv` | All IDs + fold (no structures) |
| `ids_train.txt` / `ids_val.txt` / `ids_test.txt` | Fold ID lists |
| `prompts/<ID>.txt` | Full free-form + neighbor NIST |
| `jobs/<ID>.json` | Same as structured system/user |
| `predictions/` | **Write only here** |

## Task
For every `spectrum_id` in `sample_manifest.csv` (or a fold list):
1. Read `prompts/<ID>.txt` fully (one sample at a time).
2. Free-form MASS → NETWORK → MS/MS → CHEMISTRY → DECISION.
3. Neighbor NIST = evidence about **neighbors** only.
4. Write `predictions/<ID>.json`.

## HARD BANS
- Pack-wide SMILES index / bulk mass-fit scripts
- Sealed truth / external registry
- Treating neighbor NIST as automatic seed ID without mass gate

## Cost note
Deduping to ~{n} unique molecules (~6.5× fewer than 11,540) is the recommended free-form API set. Score **spectrum-level on this pack** = unique-molecule metrics for the parent block.

## Paste
```
MSG HNSW IK1-deduped free-form + neighbor-only NIST.
Folder: this handout (n={n})
Read prompts/<ID>.txt; write predictions/<ID>.json
Seed NIST off; neighbor NIST on. One sample at a time. No pack-wide index.
Optional: only ids_test.txt / ids_val.txt / ids_train.txt for stratified runs.
```
"""
    (out_hand / "START_HERE.md").write_text(start, encoding="utf-8")
    (out_hand / "README.md").write_text(
        start
        + "\n## Build\n"
        + "```\npython scripts/dedupe_msg_hnsw_handout_by_ik1.py\n```\n"
        + f"Source handout: `{src_hand}`\n",
        encoding="utf-8",
    )

    # sealed README (short)
    (out_sealed / "README.md").write_text(
        f"""# Sealed truth — IK1-deduped handout

**OFFLINE ONLY — do not ship with model-facing zip.**

- `truth_index.csv` — {n} rows (1 per IK1 per fold)
- `dedupe_drop_log.csv` — which parent spectra were dropped
- `dedupe_selection.json` — kept IDs + policy
- `ids_{{train,val,test}}.txt`

Parent: `{src_sealed}`
""",
        encoding="utf-8",
    )

    print("[done] handout", out_hand, flush=True)
    print("[done] sealed ", out_sealed, flush=True)
    print("[done] folds  ", dict(fold_counts), flush=True)
    if missing:
        print(f"[warn] missing files for {len(missing)} samples", flush=True)
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
