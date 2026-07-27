#!/usr/bin/env python3
"""Diagnose ChemDFM strict-batch failure modes."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

chem = Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test\outputs\pub_batch_chemdfm_r_strict")
rows = list(csv.DictReader((chem / "batch_summary.csv").open(encoding="utf-8")))
print("summary_n", len(rows))
print("mass_ok_true", sum(1 for r in rows if str(r.get("mass_ok")).lower() == "true"))
print("with_smiles", sum(1 for r in rows if (r.get("smiles") or "").strip()))

valid = invalid = mass_ok = mass_bad = listlike = empty = 0
confs: list[float] = []
removed: list[int] = []
n = 0
for p in (chem / "runs").glob("*/prediction.json"):
    n += 1
    d = json.loads(p.read_text(encoding="utf-8"))
    smi = d.get("smiles") or d.get("raw_smiles") or ""
    if isinstance(smi, list):
        listlike += 1
        smi = str(smi)
    smi = str(smi).strip()
    if not smi:
        empty += 1
    elif smi.startswith("[") or "','" in smi:
        listlike += 1
    if d.get("smiles_valid") is True:
        valid += 1
    else:
        invalid += 1
    if d.get("mass_ok") is True:
        mass_ok += 1
    elif d.get("mass_ok") is False:
        mass_bad += 1
    c = d.get("confidence")
    if c is not None:
        try:
            confs.append(float(c))
        except Exception:
            pass
    for note in d.get("rescue_notes") or []:
        m = re.search(r"removed_neighbors': (\d+)", note)
        if m:
            removed.append(int(m.group(1)))

print("pred_files", n)
print("smiles_valid", valid, "invalid", invalid)
print("mass_ok", mass_ok, "mass_bad", mass_bad)
print("listlike_smiles", listlike, "empty", empty)
if confs:
    print(
        "conf_mean",
        round(sum(confs) / len(confs), 3),
        "n",
        len(confs),
        "le0.2",
        sum(1 for x in confs if x <= 0.2),
    )
if removed:
    print(
        "removed_neighbors mean",
        round(sum(removed) / len(removed), 2),
        "gt0",
        sum(1 for x in removed if x > 0),
        "n",
        len(removed),
    )
