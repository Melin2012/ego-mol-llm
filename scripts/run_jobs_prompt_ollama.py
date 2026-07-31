#!/usr/bin/env python3
"""
Run free-form predictions from prebuilt job JSONs (system_prompt + user_prompt).

Intended for NIST-neighbors-only packs where jobs_nist_neighbors/ already embeds
the evidence. Pure model generation — no rescue / library / hybrid.

Example:
  set OPENAI_BASE_URL=http://localhost:11434/v1
  set OPENAI_API_KEY=ollama
  python scripts/run_jobs_prompt_ollama.py ^
    --pack .../MSG_HNSW_blind285v2_strict_ego_msms ^
    --jobs-subdir jobs_nist_neighbors ^
    --out-subdir predictions_qwen3_freeform_nist_neigh ^
    --model qwen3:14b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ego_mol_llm.backends.base import GenerationConfig
from ego_mol_llm.backends.factory import build_backend
from ego_mol_llm.validate import parse_model_output


def _extract_json_blob(text: str) -> str | None:
    if not text:
        return None
    # strip thinking tags if present (qwen3)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.S | re.I)
    # fenced json
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return m.group(1)
    # first balanced-ish object
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--jobs-subdir", default="jobs_nist_neighbors")
    ap.add_argument("--out-subdir", default="predictions_qwen3_freeform_nist_neigh")
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "ollama"))
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()
    if args.force:
        args.skip_existing = False

    pack = args.pack.resolve()
    jobs_dir = pack / args.jobs_subdir
    out_dir = pack / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = sorted(jobs_dir.glob("MSGTEST*.json"))
    if not jobs:
        jobs = sorted(
            p for p in jobs_dir.glob("*.json") if not p.name.startswith("_")
        )
    if args.limit and args.limit > 0:
        jobs = jobs[: args.limit]

    print(
        f"[info] pack={pack} n_jobs={len(jobs)} model={args.model} "
        f"jobs={jobs_dir.name} out={out_dir.name}",
        flush=True,
    )

    os.environ.setdefault("OPENAI_BASE_URL", args.base_url)
    os.environ.setdefault("OPENAI_API_KEY", args.api_key)
    os.environ.setdefault("EGO_MOL_NUM_CTX", "32768")

    backend = build_backend(args.backend, model=args.model, base_url=args.base_url, api_key=args.api_key)
    gen = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    n_ok = n_skip = n_fail = 0
    t0 = time.perf_counter()

    for i, jp in enumerate(jobs, 1):
        job = json.loads(jp.read_text(encoding="utf-8-sig"))
        sid = (job.get("spectrum_id") or jp.stem).strip().upper()
        pred_path = out_dir / f"{sid}.json"
        if args.skip_existing and pred_path.is_file() and pred_path.stat().st_size > 20:
            n_skip += 1
            print(f"[{i}/{len(jobs)}] skip {sid}", flush=True)
            continue

        system = job.get("system_prompt") or ""
        user = job.get("user_prompt") or ""
        if not user:
            n_fail += 1
            print(f"[{i}/{len(jobs)}] FAIL {sid}: empty user_prompt", flush=True)
            continue

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        t1 = time.perf_counter()
        try:
            raw = backend.generate(messages, gen)
            if isinstance(raw, dict):
                text = raw.get("text") or raw.get("content") or json.dumps(raw)
            else:
                text = str(raw)
        except Exception as e:
            n_fail += 1
            print(f"[{i}/{len(jobs)}] FAIL {sid}: generate {type(e).__name__}: {e}", flush=True)
            pred_path.write_text(
                json.dumps(
                    {
                        "spectrum_id": sid,
                        "smiles": "",
                        "error": f"{type(e).__name__}: {e}",
                        "model": args.model,
                        "blind": True,
                        "method": "freeform_llm_reasoning",
                        "source": "model",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue

        out: dict = {
            "smiles": "",
            "iupac_or_common_name": None,
            "formula": None,
            "adduct": None,
            "confidence": None,
            "rationale": None,
            "alternatives": [],
        }
        try:
            pred = parse_model_output(text)
            smi_use = pred.canonical_smiles or pred.smiles
            if smi_use:
                out["smiles"] = smi_use
            if pred.name:
                out["iupac_or_common_name"] = pred.name
            if pred.formula:
                out["formula"] = pred.formula
            if pred.adduct or pred.matched_adduct:
                out["adduct"] = pred.adduct or pred.matched_adduct
            if pred.confidence is not None:
                out["confidence"] = pred.confidence
            if pred.rationale:
                out["rationale"] = pred.rationale
            if pred.alternatives:
                out["alternatives"] = pred.alternatives
            out["parse_mode"] = pred.parse_mode
            out["mass_ok"] = pred.mass_ok
            out["mass_error_da"] = pred.mass_error_da
        except Exception:
            blob = _extract_json_blob(text)
            if blob:
                try:
                    data = json.loads(blob)
                    if isinstance(data, dict):
                        out.update({k: data.get(k) for k in out if k in data})
                        for k in (
                            "smiles",
                            "iupac_or_common_name",
                            "formula",
                            "adduct",
                            "confidence",
                            "rationale",
                            "alternatives",
                        ):
                            if k in data:
                                out[k] = data[k]
                except Exception:
                    pass

        out["spectrum_id"] = sid
        out["model"] = args.model
        out["blind"] = True
        out["method"] = "freeform_llm_reasoning"
        out["evidence"] = "jobs_nist_neighbors_prompt"
        out["source"] = "model"
        out["no_packwide_index"] = True
        out["protocol"] = job.get("protocol")
        out["elapsed_s"] = round(time.perf_counter() - t1, 2)
        smi = (out.get("smiles") or "").strip()
        if not smi:
            out["raw_response_head"] = (text or "")[:2000]

        pred_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        n_ok += 1
        conf = out.get("confidence")
        print(
            f"[{i}/{len(jobs)}] {sid} ok conf={conf} "
            f"smiles={(smi[:60] + '…') if len(smi) > 60 else smi!r} "
            f"t={out['elapsed_s']}s",
            flush=True,
        )

    summary = {
        "pack": str(pack),
        "jobs": str(jobs_dir),
        "out": str(out_dir),
        "model": args.model,
        "n_jobs": len(jobs),
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_fail": n_fail,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {summary}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
