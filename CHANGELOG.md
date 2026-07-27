# Changelog

## 0.3.1 — In-silico spectrum re-rank (rule / CFM-ID / ICEBERG)

### Features
- **spectrum_predict.py**: pluggable offline spectrum predictors
  - 
ule — RDKit structure-aware losses + SMARTS diagnostics (default, always on)
  - cfmid — Wishart CFM-ID 4 via Docker (wishartlab/cfmid), disk-cached
  - iceberg — Coley ms-pred scaffold (when installed)
- Hybrid fusion: usion ← (1-w)·fusion + w·cosine(pred, experimental)
- select_product_annotation(..., use_insilico_rerank=True)
- Script: scripts/apply_insilico_rerank_to_pack.py
- Docs: docs/INSILICO_RERANK.md

### Notes
- Does **not** require SIRIUS. Prefer rule for throughput; CFM-ID when Docker is up.
- Conservative product policy still keeps mass-OK model SMILES.

## 0.3.0 — Offline MS/MS explanation (+ optional SIRIUS)

### Features (always-on, offline, fast)
- **`msms_explain.py`**: expanded neutral-loss table, diagnostic ions, chemistry class hints,
  shared/unique peaks vs top MS/MS-similar neighbors, Δm/z chemical guesses.
- Wired into `SpectralContext` + prompt block **MS/MS EXPLANATION (offline, fast; not SIRIUS)**.
- Runs on every sample in milliseconds — no login, no web services.

### Features (optional, expensive)
- **`sirius.py`**: SIRIUS 6 CLI (`formulas` / `structures` / `summaries`), CSI:FingerID parse.
- Hybrid `source=sirius` only when hits are provided; **not required for default product path**.
- Use SIRIUS only on hard cases / ablations, not every spectrum.

### Scripts
- `scripts/run_sirius_on_pack.py` — optional batch SIRIUS on blind packs.
- `scripts/apply_sirius_product_to_pack.py` — optional fusion with precomputed CSI hits.

### Notes
- Prefer offline MS/MS explain + NIST + neighbors for production throughput.
- CSI:FingerID needs CLI login; slow on large molecules; academic free / commercial Bright Giant.

## 0.2.0 — annotation propagation product path

### Features
- **Spectral library reverse search** (`library_search.py`): stream large MGFs (NIST2023) into a precursor-binned pickle index; reverse search by precursor window + √I cosine.
- **Method card + RT priors** (`method_card.py`): chromatography/polarity/ionization priors; relative RT compatibility scoring.
- **Hybrid candidates** (`candidates.py`): fuse NIST hits ∪ mass-consistent network neighbors ∪ LLM SMILES; product selection with provenance `source`.
- **MGF metadata in spectral context**: seed/neighbor `RTINSECONDS`, `MSV_LIB`, `FILENAME`, `CLUSTERSIZE`, ion mode → prompts and ranking.
- **Prompts**: experimental method block + NIST hit list; neighbor lines show RT/MSV when available.
- **Pipeline**: `predict_ego` / `predict_from_graphml` options `use_library_search`, `library_index_path`, `method_card`, `use_hybrid_ranker`.

### Scripts
- `scripts/build_library_index.py` — build NIST/custom MGF index.

### Notes
- Full NIST index build is offline one-time cost (~GB-scale MGF).
- Pure-model ablation: `use_hybrid_ranker=False`, `use_neighbor_rescue=False`, `use_library_search=False`.

## 0.1.x
- Initial ego-network LLM prediction, neighbor rescue, publication batch tools.
