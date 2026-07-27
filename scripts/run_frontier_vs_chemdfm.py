#!/usr/bin/env python3
"""
Publication head-to-head: Grok 4.5 (xAI frontier) vs ChemDFM (open-source).

Runs the *same* blind ego-mol jobs with two backends, then merges summaries
and evaluation metrics for tables/figures.

Examples
--------
# Grok 4.5 via xAI API (requires XAI_API_KEY)
python scripts/run_frontier_vs_chemdfm.py grok \\
  --data-root "C:/Users/AlexeyMelnik/Downloads/Ego_Mol_Test" \\
  --limit 10

# ChemDFM via Ollama
python scripts/run_frontier_vs_chemdfm.py chemdfm \\
  --data-root "C:/Users/AlexeyMelnik/Downloads/Ego_Mol_Test" \\
  --model chemdfm-v2-14b

# Merge two finished batch folders + figures
python scripts/run_frontier_vs_chemdfm.py compare \\
  --grok-dir  .../pub_batch_grok45 \\
  --chemdfm-dir .../pub_batch_chemdfm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "run_publication_batch.py"


def _env_key(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v and v.strip():
            return v.strip()
    return None


def run_batch(
    *,
    label: str,
    data_root: Path,
    out: Path,
    backend: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    workers: int,
    limit: int,
    mass_tol: float,
    max_neighbors: int,
) -> int:
    cmd = [
        sys.executable,
        str(_SCRIPT),
        "--data-root",
        str(data_root),
        "--out",
        str(out),
        "--backend",
        backend,
        "--model",
        model,
        "--workers",
        str(workers),
        "--mass-tol",
        str(mass_tol),
        "--max-neighbors",
        str(max_neighbors),
        "--skip-existing",
    ]
    if limit and limit > 0:
        cmd.extend(["--limit", str(limit)])
    if base_url:
        cmd.extend(["--base-url", base_url])
    if api_key:
        cmd.extend(["--api-key", api_key])

    env = os.environ.copy()
    if base_url:
        env["OPENAI_BASE_URL"] = base_url
    if api_key:
        env["OPENAI_API_KEY"] = api_key

    print(f"[launch] {label}")
    print(f"  backend={backend} model={model}")
    print(f"  out={out}")
    print(f"  cmd={' '.join(cmd[:8])} ...")
    out.mkdir(parents=True, exist_ok=True)
    (out / "launcher_cmd.json").write_text(
        json.dumps(
            {
                "label": label,
                "backend": backend,
                "model": model,
                "base_url": base_url,
                "workers": workers,
                "limit": limit,
                "cmd": cmd,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(cmd, cwd=str(_REPO), env=env)
    return int(proc.returncode)


def _load_summary(batch_dir: Path) -> list[dict]:
    p = batch_dir / "batch_summary.csv"
    if not p.is_file():
        raise FileNotFoundError(p)
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(x) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _truthy(x) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y"}


def compare_batches(grok_dir: Path, chemdfm_dir: Path, out_dir: Path) -> Path:
    grok = {r.get("spectrum_id") or r.get("stem"): r for r in _load_summary(grok_dir)}
    chem = {r.get("spectrum_id") or r.get("stem"): r for r in _load_summary(chemdfm_dir)}
    keys = sorted(set(grok) | set(chem))

    rows: list[dict] = []
    for k in keys:
        g = grok.get(k) or {}
        c = chem.get(k) or {}
        rows.append(
            {
                "spectrum_id": k,
                "true_name": g.get("true_name") or c.get("true_name"),
                "true_smiles": g.get("true_smiles") or c.get("true_smiles"),
                "true_inchikey": g.get("true_inchikey") or c.get("true_inchikey"),
                "true_formula": g.get("true_formula") or c.get("true_formula"),
                "grok_ok": g.get("ok"),
                "grok_smiles": g.get("smiles"),
                "grok_name": g.get("name"),
                "grok_confidence": g.get("confidence"),
                "grok_mass_ok": g.get("mass_ok"),
                "grok_source": g.get("source"),
                "chemdfm_ok": c.get("ok"),
                "chemdfm_smiles": c.get("smiles"),
                "chemdfm_name": c.get("name"),
                "chemdfm_confidence": c.get("confidence"),
                "chemdfm_mass_ok": c.get("mass_ok"),
                "chemdfm_source": c.get("source"),
                "both_have_smiles": bool((g.get("smiles") or "").strip() and (c.get("smiles") or "").strip()),
                "smiles_agree": (g.get("smiles") or "").strip() == (c.get("smiles") or "").strip()
                and bool((g.get("smiles") or "").strip()),
            }
        )

    # RDKit metrics if available
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs, inchi
    except Exception:
        Chem = None  # type: ignore

    def ik_prefix(smi: str) -> str:
        if not Chem or not smi:
            return ""
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        try:
            return (inchi.MolToInchiKey(m) or "").split("-")[0].upper()
        except Exception:
            return ""

    def tanimoto(a: str, b: str) -> float | None:
        if not Chem or not a or not b:
            return None
        m1, m2 = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
        if m1 is None or m2 is None:
            return None
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))

    for r in rows:
        true_smi = (r.get("true_smiles") or "").strip()
        true_ik = (r.get("true_inchikey") or "").split("-")[0].upper()
        gs = (r.get("grok_smiles") or "").strip()
        cs = (r.get("chemdfm_smiles") or "").strip()
        g_ik = ik_prefix(gs)
        c_ik = ik_prefix(cs)
        r["grok_ik_match"] = bool(true_ik and g_ik and true_ik == g_ik)
        r["chemdfm_ik_match"] = bool(true_ik and c_ik and true_ik == c_ik)
        r["grok_tanimoto"] = tanimoto(gs, true_smi)
        r["chemdfm_tanimoto"] = tanimoto(cs, true_smi)
        r["pair_tanimoto"] = tanimoto(gs, cs)

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    csv_path = out_dir / "grok_vs_chemdfm.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def rate(flag: str) -> float | None:
        vals = [r for r in rows if r.get("true_smiles")]
        if not vals:
            return None
        return sum(1 for r in vals if r.get(flag)) / len(vals)

    def mean_t(key: str) -> float | None:
        xs = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return (sum(xs) / len(xs)) if xs else None

    summary = {
        "n_rows": len(rows),
        "n_both_smiles": sum(1 for r in rows if r.get("both_have_smiles")),
        "n_smiles_agree": sum(1 for r in rows if r.get("smiles_agree")),
        "grok_ik_match_rate": rate("grok_ik_match"),
        "chemdfm_ik_match_rate": rate("chemdfm_ik_match"),
        "grok_mean_tanimoto": mean_t("grok_tanimoto"),
        "chemdfm_mean_tanimoto": mean_t("chemdfm_tanimoto"),
        "mean_pair_tanimoto": mean_t("pair_tanimoto"),
        "grok_dir": str(grok_dir),
        "chemdfm_dir": str(chemdfm_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # figures
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        # bar: ik match rates
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            ["Grok 4.5", "ChemDFM"],
            [
                100 * (summary["grok_ik_match_rate"] or 0),
                100 * (summary["chemdfm_ik_match_rate"] or 0),
            ],
            color=["#0ea5e9", "#1e3a5f"],
        )
        ax.set_ylabel("InChIKey first-block match (%)")
        ax.set_title("Blind structure recovery (InChIKey)")
        ax.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(fig_dir / "ik_match_bar.png", dpi=150)
        plt.close(fig)

        # tanimoto hist overlay
        gt = [r["grok_tanimoto"] for r in rows if isinstance(r.get("grok_tanimoto"), float)]
        ct = [r["chemdfm_tanimoto"] for r in rows if isinstance(r.get("chemdfm_tanimoto"), float)]
        if gt or ct:
            fig, ax = plt.subplots(figsize=(7, 4))
            if gt:
                ax.hist(gt, bins=25, alpha=0.55, label="Grok 4.5", color="#0ea5e9")
            if ct:
                ax.hist(ct, bins=25, alpha=0.55, label="ChemDFM", color="#1e3a5f")
            ax.set_xlabel("Tanimoto vs true (Morgan r=2)")
            ax.set_ylabel("Count")
            ax.set_title("Predicted vs true structure similarity")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "tanimoto_overlay.png", dpi=150)
            plt.close(fig)

        # recovery curves
        thr = [0.3, 0.5, 0.7, 0.85, 0.95, 1.0]

        def rec(xs):
            if not xs:
                return [0.0] * len(thr)
            return [sum(1 for t in xs if t >= t0) / len(xs) for t0 in thr]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(thr, rec(gt), marker="o", label="Grok 4.5", color="#0ea5e9")
        ax.plot(thr, rec(ct), marker="s", label="ChemDFM", color="#1e3a5f")
        ax.set_xlabel("Tanimoto threshold")
        ax.set_ylabel("Fraction ≥ threshold")
        ax.set_ylim(0, 1.05)
        ax.set_title("Structure recovery curve")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "recovery_curves.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        (out_dir / "figures_error.txt").write_text(str(e), encoding="utf-8")

    md = [
        "# Grok 4.5 vs ChemDFM — blind ego-network prediction",
        "",
        f"- Rows: **{summary['n_rows']}**",
        f"- Grok InChIKey match: **{summary['grok_ik_match_rate']}**",
        f"- ChemDFM InChIKey match: **{summary['chemdfm_ik_match_rate']}**",
        f"- Grok mean Tanimoto: **{summary['grok_mean_tanimoto']}**",
        f"- ChemDFM mean Tanimoto: **{summary['chemdfm_mean_tanimoto']}**",
        f"- Mean Grok↔ChemDFM Tanimoto: **{summary['mean_pair_tanimoto']}**",
        "",
        f"CSV: `{csv_path}`",
    ]
    (out_dir / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[compare] wrote {csv_path}")
    return csv_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test"),
    )
    common.add_argument("--limit", type=int, default=0)
    common.add_argument("--mass-tol", type=float, default=0.05)
    common.add_argument("--max-neighbors", type=int, default=25)

    g = sub.add_parser("grok", parents=[common], help="Run Grok 4.5 batch via xAI API")
    g.add_argument(
        "--out",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test\outputs\pub_batch_grok45"),
    )
    g.add_argument("--model", default="grok-4.5")
    g.add_argument("--base-url", default="https://api.x.ai/v1")
    g.add_argument("--api-key", default=None, help="Or set XAI_API_KEY")
    g.add_argument("--workers", type=int, default=2)

    c = sub.add_parser("chemdfm", parents=[common], help="Run ChemDFM via Ollama/OpenAI-compatible")
    c.add_argument(
        "--out",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test\outputs\pub_batch_chemdfm"),
    )
    c.add_argument("--model", default="chemdfm-v2-14b")
    c.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    c.add_argument("--api-key", default="ollama")
    c.add_argument("--workers", type=int, default=1)
    c.add_argument(
        "--backend",
        default="ollama",
        help="ollama | openai | transformers",
    )

    m = sub.add_parser("compare", help="Merge finished Grok vs ChemDFM batches")
    m.add_argument("--grok-dir", type=Path, required=True)
    m.add_argument("--chemdfm-dir", type=Path, required=True)
    m.add_argument(
        "--out",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test\outputs\pub_compare_grok_vs_chemdfm"),
    )

    both = sub.add_parser("both", parents=[common], help="Run Grok then ChemDFM then compare")
    both.add_argument(
        "--grok-out",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test\outputs\pub_batch_grok45"),
    )
    both.add_argument(
        "--chemdfm-out",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test\outputs\pub_batch_chemdfm"),
    )
    both.add_argument(
        "--compare-out",
        type=Path,
        default=Path(r"C:\Users\AlexeyMelnik\Downloads\Ego_Mol_Test\outputs\pub_compare_grok_vs_chemdfm"),
    )
    both.add_argument("--grok-model", default="grok-4.5")
    both.add_argument("--chemdfm-model", default="chemdfm-v2-14b")
    both.add_argument("--grok-workers", type=int, default=2)
    both.add_argument("--chemdfm-workers", type=int, default=1)
    both.add_argument("--chemdfm-base-url", default="http://127.0.0.1:11434/v1")
    both.add_argument("--chemdfm-backend", default="ollama")

    args = p.parse_args(argv)

    if args.cmd == "grok":
        key = args.api_key or _env_key("XAI_API_KEY", "GROK_API_KEY", "OPENAI_API_KEY")
        if not key:
            print(
                "ERROR: No API key. Set XAI_API_KEY (https://console.x.ai/) or pass --api-key.",
                file=sys.stderr,
            )
            return 2
        return run_batch(
            label="grok-4.5",
            data_root=args.data_root,
            out=args.out,
            backend="openai",
            model=args.model,
            base_url=args.base_url,
            api_key=key,
            workers=args.workers,
            limit=args.limit,
            mass_tol=args.mass_tol,
            max_neighbors=args.max_neighbors,
        )

    if args.cmd == "chemdfm":
        return run_batch(
            label="chemdfm",
            data_root=args.data_root,
            out=args.out,
            backend=args.backend,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            workers=args.workers,
            limit=args.limit,
            mass_tol=args.mass_tol,
            max_neighbors=args.max_neighbors,
        )

    if args.cmd == "compare":
        compare_batches(args.grok_dir, args.chemdfm_dir, args.out)
        return 0

    if args.cmd == "both":
        key = _env_key("XAI_API_KEY", "GROK_API_KEY", "OPENAI_API_KEY")
        if not key:
            print("ERROR: Set XAI_API_KEY for Grok 4.5.", file=sys.stderr)
            return 2
        rc1 = run_batch(
            label="grok-4.5",
            data_root=args.data_root,
            out=args.grok_out,
            backend="openai",
            model=args.grok_model,
            base_url="https://api.x.ai/v1",
            api_key=key,
            workers=args.grok_workers,
            limit=args.limit,
            mass_tol=args.mass_tol,
            max_neighbors=args.max_neighbors,
        )
        rc2 = run_batch(
            label="chemdfm",
            data_root=args.data_root,
            out=args.chemdfm_out,
            backend=args.chemdfm_backend,
            model=args.chemdfm_model,
            base_url=args.chemdfm_base_url,
            api_key="ollama",
            workers=args.chemdfm_workers,
            limit=args.limit,
            mass_tol=args.mass_tol,
            max_neighbors=args.max_neighbors,
        )
        if rc1 == 0 and rc2 == 0:
            compare_batches(args.grok_out, args.chemdfm_out, args.compare_out)
        return rc1 or rc2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
