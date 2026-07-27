#!/usr/bin/env python3
"""
Publication batch: blind ego-mol-llm predictions over Ego_Mol_Test.

Pairs each GraphML with its subgraph MGF, optionally extracts an anonymized
seed spectrum from the library MGF (by SPECTRUMID), runs hide_seed_name=True,
supports resume, multi-worker shards, and writes evaluation-ready summaries.

Example (dry-run baseline, all networks)::

    python scripts/run_publication_batch.py \\
      --data-root "C:/Users/AlexeyMelnik/Downloads/Ego_Mol_Test" \\
      --backend dry-run --workers 4

Example (Ollama ChemDFM when available)::

    set OPENAI_BASE_URL=http://localhost:11434/v1
    set OPENAI_API_KEY=ollama
    python scripts/run_publication_batch.py --backend ollama -m chemdfm-v2-14b --workers 1
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root without install
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

SPECTRUM_RE = re.compile(r"(AROMEC18COLGATE\d+)", re.I)
BEGIN_IONS = re.compile(r"(?i)BEGIN IONS")
END_IONS = re.compile(r"(?i)END IONS")


@dataclass
class Job:
    stem: str
    graphml: str
    network_mgf: str | None
    spectrum_id: str | None
    true_name: str | None = None
    true_smiles: str | None = None
    true_inchikey: str | None = None
    true_formula: str | None = None
    true_exact_mass: str | None = None


def _slug(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    s = re.sub(r"_+", "_", s)
    return (s or "run")[:max_len]


def discover_jobs(data_root: Path) -> list[Job]:
    gdir = data_root / "graohmls"
    if not gdir.is_dir():
        # typo-tolerant
        for alt in ("graphmls", "graphml", "networks"):
            if (data_root / alt).is_dir():
                gdir = data_root / alt
                break
    mdir = data_root / "subgraph_mgfs"
    files = sorted(gdir.glob("*.graphml"))
    jobs: list[Job] = []
    for g in files:
        stem = g.stem
        mgf = mdir / f"{stem}.mgf"
        sid_m = SPECTRUM_RE.search(stem)
        jobs.append(
            Job(
                stem=stem,
                graphml=str(g.resolve()),
                network_mgf=str(mgf.resolve()) if mgf.is_file() else None,
                spectrum_id=sid_m.group(1).upper() if sid_m else None,
            )
        )
    return jobs


def load_ground_truth_csv(csv_path: Path) -> dict[str, dict[str, str]]:
    """Index testing2.csv by SPECTRUMID."""
    if not csv_path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("SPECTRUMID") or "").strip().upper()
            if not sid:
                continue
            out[sid] = {
                "true_name": row.get("NAME") or row.get("COMPOUND") or "",
                "true_smiles": row.get("SMILES") or "",
                "true_inchikey": row.get("InChIKey") or row.get("INCHIKEY") or "",
                "true_formula": row.get("FORMULA") or "",
                "true_exact_mass": row.get("EXACT_MASS") or row.get("EXACTMASS") or "",
                "true_adduct": row.get("ADDUCT") or row.get("PRECURSOR_TYPE") or "",
                "true_pepmass": row.get("PEPMASS") or "",
            }
    return out


def index_library_mgf(library_path: Path) -> dict[str, str]:
    """
    Map SPECTRUMID -> full BEGIN IONS ... END IONS block text.
    Streams file; holds ~7MB for this library.
    """
    if not library_path.is_file():
        return {}
    text = library_path.read_text(encoding="utf-8", errors="replace")
    blocks = BEGIN_IONS.split(text)
    idx: dict[str, str] = {}
    for b in blocks[1:]:
        parts = END_IONS.split(b, 1)
        body = parts[0]
        block = "BEGIN IONS\n" + body.strip() + "\nEND IONS\n"
        m = re.search(r"(?im)^SPECTRUMID=(.+)$", body, re.M)
        if m:
            idx[m.group(1).strip().upper()] = block
    return idx


def anonymize_seed_block(block: str) -> str:
    """Keep PEPMASS/CHARGE/peaks; strip structure-identifying metadata."""
    drop_keys = {
        "NAME",
        "COMPOUND",
        "SMILES",
        "INCHIKEY",
        "INCHI",
        "FORMULA",
        "IUPAC",
        "TITLE",
        "ORIGINAL_TITLE",
        "CASNO",
        "HMDB_ID",
        "KEGG_ID",
        "PUBCHEM_ID",
        "SEQ",
    }
    lines_out = ["BEGIN IONS"]
    in_body = False
    for line in block.splitlines():
        u = line.strip()
        if re.match(r"(?i)BEGIN IONS", u):
            in_body = True
            continue
        if re.match(r"(?i)END IONS", u):
            break
        if not u:
            continue
        if "=" in u and not (u[0].isdigit() or u.startswith(".")):
            k = u.split("=", 1)[0].strip().upper()
            if k in drop_keys:
                continue
            if k in {"PEPMASS", "CHARGE", "MSLEVEL", "IONMODE", "ION_MODE", "RTINSECONDS"}:
                lines_out.append(u)
            # drop everything else (SPECTRUMID etc.) for blind safety
            continue
        # peak line
        lines_out.append(u)
    lines_out.append("END IONS")
    return "\n".join(lines_out) + "\n"


def ensure_seed_mgf(
    spectrum_id: str | None,
    lib_index: dict[str, str],
    seed_dir: Path,
) -> str | None:
    if not spectrum_id:
        return None
    sid = spectrum_id.upper()
    block = lib_index.get(sid)
    if not block:
        return None
    seed_dir.mkdir(parents=True, exist_ok=True)
    path = seed_dir / f"{sid}_seed_blind.mgf"
    if not path.is_file():
        path.write_text(anonymize_seed_block(block), encoding="utf-8")
    return str(path.resolve())


def job_out_dir(batch_root: Path, stem: str) -> Path:
    return batch_root / "runs" / _slug(stem, 80)


def already_done(out_dir: Path) -> bool:
    return (out_dir / "prediction.json").is_file()


def _worker_run(payload: dict) -> dict:
    """Process-pool worker entry (picklable)."""
    try:
        from ego_mol_llm.predict import predict_from_graphml
        from ego_mol_llm.report import export_report
    except Exception as e:
        return {
            "stem": payload.get("stem"),
            "ok": False,
            "error": f"import: {type(e).__name__}: {e}",
        }

    stem = payload["stem"]
    out_dir = Path(payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if payload.get("skip_existing") and (out_dir / "prediction.json").is_file():
        try:
            d = json.loads((out_dir / "prediction.json").read_text(encoding="utf-8"))
            return {
                "stem": stem,
                "ok": True,
                "skipped": True,
                "out_dir": str(out_dir),
                "smiles": d.get("smiles"),
                "name": d.get("name"),
                "formula": d.get("formula"),
                "confidence": d.get("confidence"),
                "mass_ok": d.get("mass_ok"),
                "source": d.get("source"),
                "seed_mz": d.get("seed_mz"),
                "true_seed_name": d.get("true_seed_name"),
                "msms_used": d.get("msms_used"),
                "smiles_valid": d.get("smiles_valid"),
                "spectrum_id": payload.get("spectrum_id"),
                "true_name": payload.get("true_name"),
                "true_smiles": payload.get("true_smiles"),
                "true_inchikey": payload.get("true_inchikey"),
                "true_formula": payload.get("true_formula"),
                "graphml": payload.get("graphml"),
                "error": None,
            }
        except Exception:
            pass

    t0 = time.perf_counter()
    try:
        mgf_paths = [payload["network_mgf"]] if payload.get("network_mgf") else None
        result = predict_from_graphml(
            graphml_path=payload["graphml"],
            backend=payload["backend"],
            model=payload["model"],
            seed_id=payload.get("seed_id") or "0",
            hide_seed_name=True,
            max_neighbors=int(payload.get("max_neighbors") or 25),
            include_two_hop=not payload.get("no_two_hop"),
            load_in_4bit=bool(payload.get("load_in_4bit", False)),
            base_url=payload.get("base_url"),
            api_key=payload.get("api_key"),
            temperature=float(payload.get("temperature") or 0.2),
            max_new_tokens=int(payload.get("max_new_tokens") or 1024),
            mass_tol_da=float(payload.get("mass_tol") or 0.05),
            mgf_paths=mgf_paths,
            seed_mgf=payload.get("seed_mgf"),
        )
        export_report(result, out_dir)
        # lightweight figures can be slow; export_report already tries
        d = result.to_dict()
        elapsed = time.perf_counter() - t0
        meta = {
            "stem": stem,
            "spectrum_id": payload.get("spectrum_id"),
            "elapsed_s": elapsed,
            "backend": payload["backend"],
            "model": payload["model"],
            "blind": True,
            "network_mgf": payload.get("network_mgf"),
            "seed_mgf": payload.get("seed_mgf"),
            "true_name": payload.get("true_name"),
            "true_smiles": payload.get("true_smiles"),
            "true_inchikey": payload.get("true_inchikey"),
            "true_formula": payload.get("true_formula"),
            "true_exact_mass": payload.get("true_exact_mass"),
        }
        (out_dir / "job_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return {
            "stem": stem,
            "ok": True,
            "skipped": False,
            "out_dir": str(out_dir),
            "smiles": d.get("smiles"),
            "name": d.get("name"),
            "formula": d.get("formula"),
            "confidence": d.get("confidence"),
            "mass_ok": d.get("mass_ok"),
            "source": d.get("source"),
            "seed_mz": d.get("seed_mz"),
            "true_seed_name": d.get("true_seed_name"),
            "msms_used": d.get("msms_used"),
            "smiles_valid": d.get("smiles_valid"),
            "elapsed_s": elapsed,
            "spectrum_id": payload.get("spectrum_id"),
            "true_name": payload.get("true_name"),
            "true_smiles": payload.get("true_smiles"),
            "true_inchikey": payload.get("true_inchikey"),
            "true_formula": payload.get("true_formula"),
            "graphml": payload.get("graphml"),
            "error": None,
        }
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        (out_dir / "error.txt").write_text(
            err + "\n\n" + traceback.format_exc(), encoding="utf-8"
        )
        return {
            "stem": stem,
            "ok": False,
            "skipped": False,
            "out_dir": str(out_dir),
            "error": err,
            "spectrum_id": payload.get("spectrum_id"),
            "true_name": payload.get("true_name"),
            "true_smiles": payload.get("true_smiles"),
            "true_inchikey": payload.get("true_inchikey"),
            "true_formula": payload.get("true_formula"),
            "graphml": payload.get("graphml"),
            "elapsed_s": time.perf_counter() - t0,
        }


def write_summaries(batch_root: Path, rows: list[dict], meta: dict) -> None:
    batch_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "stem",
        "spectrum_id",
        "ok",
        "skipped",
        "smiles",
        "smiles_valid",
        "name",
        "formula",
        "confidence",
        "mass_ok",
        "source",
        "msms_used",
        "seed_mz",
        "true_seed_name",
        "true_name",
        "true_smiles",
        "true_inchikey",
        "true_formula",
        "elapsed_s",
        "out_dir",
        "graphml",
        "error",
    ]
    csv_path = batch_root / "batch_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    payload = {"meta": meta, "n": len(rows), "results": rows}
    (batch_root / "batch_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    ok_n = sum(1 for r in rows if r.get("ok"))
    msms_n = sum(1 for r in rows if r.get("msms_used"))
    md = [
        "# Publication batch summary",
        "",
        f"- Generated: `{meta.get('generated_at')}`",
        f"- Backend: `{meta.get('backend')}` / model `{meta.get('model')}`",
        "- Blind: **true** (seed name/SMILES withheld)",
        f"- Total: **{len(rows)}**",
        f"- OK: **{ok_n}**",
        f"- Failed: **{len(rows) - ok_n}**",
        f"- MS/MS used: **{msms_n}**",
        f"- Data root: `{meta.get('data_root')}`",
        "",
        "Ground-truth columns (`true_*`) are for **evaluation only** and were not fed to the model.",
        "",
        "| # | spectrum_id | OK | pred name | SMILES | conf | mass_ok | source | true_name |",
        "|---|-------------|----|-----------|--------|------|---------|--------|-----------|",
    ]
    for i, r in enumerate(rows, 1):
        smi = (r.get("smiles") or "")[:32]
        pn = (str(r.get("name") or "")[:28]).replace("|", "/")
        tn = (str(r.get("true_name") or "")[:28]).replace("|", "/")
        md.append(
            f"| {i} | `{r.get('spectrum_id') or ''}` | {r.get('ok')} | {pn} | `{smi}` | "
            f"{r.get('confidence')} | {r.get('mass_ok')} | {r.get('source')} | {tn} |"
        )
    (batch_root / "batch_summary.md").write_text("\n".join(md), encoding="utf-8")

    # progress checkpoint
    (batch_root / "progress.json").write_text(
        json.dumps(
            {
                "done": len(rows),
                "ok": ok_n,
                "failed": len(rows) - ok_n,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def inchikey_first_block(s: str | None) -> str:
    if not s:
        return ""
    return s.split("-")[0].strip().upper()


def _metrics_for_rows(rows: list[dict], Chem, AllChem, DataStructs) -> dict:
    """Score one row subset. Abstain: no SMILES counts as incorrect for structure rates."""
    n = 0
    n_pred = 0
    n_abstain = 0
    n_ik_prefix = 0
    n_exact_smi = 0
    tani_sum = 0.0
    tani_n = 0
    conf_bins = {"high": 0, "mid": 0, "low": 0, "none": 0}

    for r in rows:
        if not r.get("ok"):
            continue
        n += 1
        src = str(r.get("source") or "unknown").lower()
        conf = r.get("confidence")
        try:
            c = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            c = None
        if c is None:
            conf_bins["none"] += 1
        elif c >= 0.7:
            conf_bins["high"] += 1
        elif c >= 0.4:
            conf_bins["mid"] += 1
        else:
            conf_bins["low"] += 1

        pred = (r.get("smiles") or "").strip()
        true = (r.get("true_smiles") or "").strip()
        if src == "abstain" or not pred:
            n_abstain += 1 if src == "abstain" or not pred else 0
            # Abstain / empty: structure metrics count as non-match (denom = n)
            continue
        n_pred += 1
        tik = inchikey_first_block(r.get("true_inchikey"))
        pik = ""
        if Chem and pred:
            mol = Chem.MolFromSmiles(pred)
            if mol is not None:
                try:
                    from rdkit.Chem import inchi

                    pik = inchikey_first_block(inchi.MolToInchiKey(mol))
                except Exception:
                    pik = ""
        if tik and pik and pik == tik:
            n_ik_prefix += 1
        if pred and true and pred == true:
            n_exact_smi += 1
        if Chem and pred and true:
            m1 = Chem.MolFromSmiles(pred)
            m2 = Chem.MolFromSmiles(true)
            if m1 is not None and m2 is not None:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
                tani_sum += DataStructs.TanimotoSimilarity(fp1, fp2)
                tani_n += 1

    return {
        "n": n,
        "n_with_pred_smiles": n_pred,
        "n_abstain_or_empty": n_abstain,
        "inchikey_first_block_match": n_ik_prefix,
        # Denominator = all ok rows (abstain counts as miss)
        "inchikey_match_rate": (n_ik_prefix / n) if n else None,
        "exact_smiles_match": n_exact_smi,
        "mean_tanimoto_morgan2": (tani_sum / tani_n) if tani_n else None,
        "tanimoto_n": tani_n,
        "confidence_bins": conf_bins,
        "scoring_note": (
            "Abstain/empty SMILES count as incorrect for structure rates "
            "(denom = n_ok). Tanimoto mean is over pairs with both SMILES only."
        ),
    }


def compute_eval_metrics(rows: list[dict]) -> dict:
    """
    Evaluation stratified by prediction source.

    Headline for pure-model comparisons: ``by_source.model`` (and optionally
    model+empty before rescue). Product-path: overall includes neighbor_rescue.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except Exception:
        Chem = None  # type: ignore
        AllChem = None  # type: ignore
        DataStructs = None  # type: ignore

    ok_rows = [r for r in rows if r.get("ok")]
    source_counts: dict[str, int] = {}
    for r in ok_rows:
        src = str(r.get("source") or "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    overall = _metrics_for_rows(ok_rows, Chem, AllChem, DataStructs)
    by_source: dict[str, dict] = {}
    for src in sorted(source_counts.keys()):
        sub = [r for r in ok_rows if str(r.get("source") or "unknown") == src]
        by_source[src] = _metrics_for_rows(sub, Chem, AllChem, DataStructs)

    # Model-only structure accuracy among rows that stayed model (accepted SMILES)
    model_rows = [r for r in ok_rows if str(r.get("source") or "") == "model"]
    # Pure-model attempt rate: model + abstain (rescue not applied as final answer)
    pure_attempt = [
        r
        for r in ok_rows
        if str(r.get("source") or "") in {"model", "abstain"}
    ]

    return {
        "n_ok": overall["n"],
        "n_with_pred_smiles": overall["n_with_pred_smiles"],
        "inchikey_first_block_match": overall["inchikey_first_block_match"],
        "inchikey_match_rate": overall["inchikey_match_rate"],
        "exact_smiles_match": overall["exact_smiles_match"],
        "mean_tanimoto_morgan2": overall["mean_tanimoto_morgan2"],
        "tanimoto_n": overall["tanimoto_n"],
        "confidence_bins": overall["confidence_bins"],
        "source_counts": source_counts,
        "overall": overall,
        "by_source": by_source,
        "headline_model_only": _metrics_for_rows(model_rows, Chem, AllChem, DataStructs),
        "headline_pure_model_path": _metrics_for_rows(
            pure_attempt, Chem, AllChem, DataStructs
        ),
        "scoring_rules": {
            "abstain": "counts as incorrect for IK/exact rates (no predicted structure)",
            "neighbor_rescue": (
                "reported under by_source.neighbor_rescue and overall product path; "
                "NOT pure-model accuracy"
            ),
            "model": "mass-accepted model SMILES only (headline pure-model successes)",
            "recommended_reporting": (
                "Report headline_model_only or pure_model_path for LLM comparison; "
                "report overall (incl. rescue) separately as product-path metrics"
            ),
        },
    }


def plot_publication_figures(batch_root: Path, rows: list[dict], metrics: dict) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        (batch_root / "figures_error.txt").write_text(str(e), encoding="utf-8")
        return paths

    fig_dir = batch_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) confidence histogram
    confs = []
    for r in rows:
        if not r.get("ok"):
            continue
        try:
            if r.get("confidence") is not None:
                confs.append(float(r["confidence"]))
        except (TypeError, ValueError):
            pass
    if confs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(confs, bins=20, color="#0ea5e9", edgecolor="white")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_title("Blind prediction confidence")
        fig.tight_layout()
        p = fig_dir / "confidence_hist.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(str(p))

    # 2) source breakdown
    sc = metrics.get("source_counts") or {}
    if sc:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = list(sc.keys())
        vals = [sc[k] for k in labels]
        ax.bar(labels, vals, color="#1e3a5f")
        ax.set_ylabel("Count")
        ax.set_title("Prediction source")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        p = fig_dir / "source_counts.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(str(p))

    # 3) tanimoto if we can recompute list
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        tans = []
        for r in rows:
            if not r.get("ok"):
                continue
            pred = (r.get("smiles") or "").strip()
            true = (r.get("true_smiles") or "").strip()
            if not pred or not true:
                continue
            m1 = Chem.MolFromSmiles(pred)
            m2 = Chem.MolFromSmiles(true)
            if m1 is None or m2 is None:
                continue
            fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
            tans.append(DataStructs.TanimotoSimilarity(fp1, fp2))
        if tans:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(tans, bins=25, color="#22c55e", edgecolor="white")
            ax.axvline(sum(tans) / len(tans), color="#b91c1c", ls="--", label=f"mean={sum(tans)/len(tans):.3f}")
            ax.set_xlabel("Tanimoto (Morgan r=2)")
            ax.set_ylabel("Count")
            ax.set_title("Predicted vs true structure similarity")
            ax.legend()
            fig.tight_layout()
            p = fig_dir / "tanimoto_hist.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(str(p))

            # cumulative success at thresholds
            thr = [0.3, 0.5, 0.7, 0.85, 0.95, 1.0]
            rates = [sum(1 for t in tans if t >= t0) / len(tans) for t0 in thr]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(thr, rates, marker="o", color="#7c3aed")
            ax.set_xlabel("Tanimoto threshold")
            ax.set_ylabel("Fraction ≥ threshold")
            ax.set_ylim(0, 1.05)
            ax.set_title("Structure recovery curve")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = fig_dir / "tanimoto_recovery_curve.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(str(p))
    except Exception as e:
        (fig_dir / "tanimoto_error.txt").write_text(str(e), encoding="utf-8")

    # 4) ok/fail pie
    ok_n = sum(1 for r in rows if r.get("ok"))
    fail_n = len(rows) - ok_n
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [ok_n, fail_n],
        labels=[f"OK ({ok_n})", f"Fail ({fail_n})"],
        colors=["#86efac", "#fecaca"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Batch completion")
    fig.tight_layout()
    p = fig_dir / "completion_pie.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publication blind batch for Ego_Mol_Test")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Batch output root (default: data-root/outputs/pub_batch_<ts>)",
    )
    p.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Existing batch root to resume (skips finished prediction.json)",
    )
    p.add_argument("--backend", default="dry-run")
    p.add_argument("--model", "-m", default="chemdfm-8b")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--limit", type=int, default=0, help="Only first N jobs (0=all)")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--seed-id", default="0")
    p.add_argument("--max-neighbors", type=int, default=25)
    p.add_argument("--no-two-hop", action="store_true")
    p.add_argument("--mass-tol", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--no-mgf", action="store_true", help="GraphML only")
    p.add_argument("--no-seed-mgf", action="store_true", help="Do not extract library seed spectrum")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--library-mgf", type=Path, default=None)
    p.add_argument("--truth-csv", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        print(f"ERROR: data root not found: {data_root}", file=sys.stderr)
        return 2

    skip_existing = args.skip_existing and not args.no_skip_existing

    if args.resume_dir:
        batch_root = args.resume_dir.resolve()
        batch_root.mkdir(parents=True, exist_ok=True)
    elif args.out:
        batch_root = args.out.resolve()
        batch_root.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_root = data_root / "outputs" / f"pub_batch_{ts}_{_slug(args.backend, 16)}"
        batch_root.mkdir(parents=True, exist_ok=True)

    truth_csv = args.truth_csv or (data_root / "LEVEL1_ASTRAL_C18_20260708_testing2.csv")
    library_mgf = args.library_mgf or (data_root / "LEVEL1_ASTRAL_C18_20260708_Testing")
    truth = load_ground_truth_csv(truth_csv)
    print(f"[info] ground truth spectra indexed: {len(truth)} from {truth_csv.name}")

    jobs = discover_jobs(data_root)
    print(f"[info] discovered graphml jobs: {len(jobs)}")
    if args.limit and args.limit > 0:
        jobs = jobs[: args.limit]
        print(f"[info] limited to first {len(jobs)}")

    # shard
    if args.shard_count > 1:
        jobs = [j for i, j in enumerate(jobs) if i % args.shard_count == args.shard_index]
        print(
            f"[info] shard {args.shard_index}/{args.shard_count} -> {len(jobs)} jobs"
        )

    # attach truth
    for j in jobs:
        if j.spectrum_id and j.spectrum_id in truth:
            t = truth[j.spectrum_id]
            j.true_name = t.get("true_name")
            j.true_smiles = t.get("true_smiles")
            j.true_inchikey = t.get("true_inchikey")
            j.true_formula = t.get("true_formula")
            j.true_exact_mass = t.get("true_exact_mass")

    lib_index: dict[str, str] = {}
    seed_dir = batch_root / "seed_mgfs_blind"
    if not args.no_seed_mgf and not args.no_mgf:
        print(f"[info] indexing library MGF: {library_mgf}")
        lib_index = index_library_mgf(library_mgf)
        print(f"[info] library SPECTRUMID blocks: {len(lib_index)}")

    payloads: list[dict] = []
    for j in jobs:
        out_dir = job_out_dir(batch_root, j.stem)
        seed_mgf = None
        if not args.no_seed_mgf and not args.no_mgf:
            seed_mgf = ensure_seed_mgf(j.spectrum_id, lib_index, seed_dir)
        network_mgf = None if args.no_mgf else j.network_mgf
        payloads.append(
            {
                "stem": j.stem,
                "graphml": j.graphml,
                "network_mgf": network_mgf,
                "seed_mgf": seed_mgf,
                "out_dir": str(out_dir),
                "backend": args.backend,
                "model": args.model,
                "seed_id": args.seed_id,
                "max_neighbors": args.max_neighbors,
                "no_two_hop": args.no_two_hop,
                "mass_tol": args.mass_tol,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "base_url": args.base_url,
                "api_key": args.api_key,
                "load_in_4bit": False,
                "skip_existing": skip_existing,
                "spectrum_id": j.spectrum_id,
                "true_name": j.true_name,
                "true_smiles": j.true_smiles,
                "true_inchikey": j.true_inchikey,
                "true_formula": j.true_formula,
                "true_exact_mass": j.true_exact_mass,
            }
        )

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "batch_root": str(batch_root),
        "backend": args.backend,
        "model": args.model,
        "blind": True,
        "workers": args.workers,
        "n_jobs": len(payloads),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "with_network_mgf": not args.no_mgf,
        "with_seed_mgf": not args.no_seed_mgf and not args.no_mgf,
        "truth_csv": str(truth_csv),
        "library_mgf": str(library_mgf),
    }
    (batch_root / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[info] batch_root = {batch_root}")
    print(f"[info] workers = {args.workers}  backend = {args.backend}")

    rows: list[dict] = []
    t_all = time.perf_counter()
    n = len(payloads)
    if n == 0:
        print("No jobs.")
        return 1

    # Sequential if 1 worker (easier debugging); else process pool
    if args.workers <= 1:
        for i, pl in enumerate(payloads, 1):
            print(f"[{i}/{n}] {pl['stem'][:70]}", flush=True)
            rows.append(_worker_run(pl))
            if i % 25 == 0 or i == n:
                write_summaries(batch_root, rows, meta)
                ok = sum(1 for r in rows if r.get("ok"))
                print(f"  checkpoint {i}/{n} ok={ok}", flush=True)
    else:
        # Windows-friendly process pool
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker_run, pl): pl for pl in payloads}
            done = 0
            for fut in as_completed(futs):
                done += 1
                r = fut.result()
                rows.append(r)
                stem = (r.get("stem") or "")[:60]
                status = "ok" if r.get("ok") else "FAIL"
                if r.get("skipped"):
                    status = "skip"
                print(f"[{done}/{n}] {status} {stem}", flush=True)
                if done % 25 == 0 or done == n:
                    # stable order by original stem list for checkpoints
                    by_stem = {x.get("stem"): x for x in rows}
                    ordered = [by_stem[pl["stem"]] for pl in payloads if pl["stem"] in by_stem]
                    write_summaries(batch_root, ordered, meta)

    # final order
    by_stem = {x.get("stem"): x for x in rows}
    ordered = [by_stem[pl["stem"]] for pl in payloads if pl["stem"] in by_stem]
    # any missing
    for pl in payloads:
        if pl["stem"] not in by_stem:
            ordered.append({"stem": pl["stem"], "ok": False, "error": "missing result"})

    meta["elapsed_s"] = time.perf_counter() - t_all
    write_summaries(batch_root, ordered, meta)

    metrics = compute_eval_metrics(ordered)
    (batch_root / "eval_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("[info] eval_metrics:", json.dumps(metrics, indent=2))

    if not args.no_figures:
        figs = plot_publication_figures(batch_root, ordered, metrics)
        print(f"[info] figures: {len(figs)}")
        for f in figs:
            print(f"  - {f}")

    ok = sum(1 for r in ordered if r.get("ok"))
    print(f"[done] {ok}/{len(ordered)} ok in {meta['elapsed_s']:.1f}s")
    print(f"[done] summary: {batch_root / 'batch_summary.csv'}")
    return 0 if ok == len(ordered) else 0  # still 0 — partial is ok for resume


if __name__ == "__main__":
    # Windows process spawn guard
    raise SystemExit(main())
