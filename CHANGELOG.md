# Changelog

## 0.4.1 — Evaluation freeze / MSG HNSW packs (2026-07-31)

### Added
- MassSpecGym HNSW pack builders: strict blind 285 (scripts/build_msg_hnsw_blind285_strict.py), full 11540 neighbor-NIST handout (scripts/build_msg_hnsw_full_new_nist_handout.py).
- Neighbor-only NIST refresh for blind packs (scripts/refresh_pack_nist_neighbors_only.py); seed NIST disabled by protocol.
- Free-form job runner for prebuilt prompts (scripts/run_jobs_prompt_ollama.py); Ollama blind-pack runner improvements (scripts/run_blind_pack_ollama.py).
- MSG MGF train/val/test split utility (scripts/split_massspecgym_mgf.py).
- Paper freeze docs: paper/METHODS.md (updated), paper/RESULTS_BOARD.md, paper/HANDOFF_ALEXANDER.md, paper/EMAIL_DRAFT_ALEXANDER.md.
- SIRIUS network-first explain path and docs; product ranker scripts for holdout 40c/40d.
- Redundancy analysis helper: scripts/_analyze_msg_full_redundancy.py.

### Evaluation protocol notes
- Valid free-form vs bulk ranker-style returns audited and separated on results board.
- MSG HNSW linked by file-order index N; official fold labels joined offline (not in GraphML).

---

## 0.4.0 ΓÇö CFM-IDΓÇôfirst network fragment explanation

### Features
- **`cfm_network_explain.py`**: for ego neighbors with SMILES, run CFM-ID `cfm-annotate`
  (+ predict cosine); transfer fragment SMILES onto seed peaks that share m/z.
- **`scripts/precompute_cfm_network_explain.py`**: pack batch ΓåÆ `cfm_explain/*.json`,
  optional `--inject-prompts` so the LLM sees peakΓåÆchemistry before proposing SMILES.
- Prompt system text updated to use CFM fragment block as substructure evidence.
- Complements (does not replace) rule `msms_explain` and post-hoc CFM candidate re-rank.

### Docs
- `docs/CFM_NETWORK_FIRST.md`

## 0.3.1 ΓÇö In-silico spectrum re-rank (rule / CFM-ID / ICEBERG)

### Features
- **spectrum_predict.py**: pluggable offline spectrum predictors
  - 
ule ΓÇö RDKit structure-aware losses + SMARTS diagnostics (default, always on)
  - cfmid ΓÇö Wishart CFM-ID 4 via Docker (wishartlab/cfmid), disk-cached
  - iceberg ΓÇö Coley ms-pred scaffold (when installed)
- Hybrid fusion: usion ΓåÉ (1-w)┬╖fusion + w┬╖cosine(pred, experimental)
- select_product_annotation(..., use_insilico_rerank=True)
- Script: scripts/apply_insilico_rerank_to_pack.py
- Docs: docs/INSILICO_RERANK.md

### Notes
- Does **not** require SIRIUS. Prefer rule for throughput; CFM-ID when Docker is up.
- Conservative product policy still keeps mass-OK model SMILES.

## 0.3.0 ΓÇö Offline MS/MS explanation (+ optional SIRIUS)

### Features (always-on, offline, fast)
- **`msms_explain.py`**: expanded neutral-loss table, diagnostic ions, chemistry class hints,
  shared/unique peaks vs top MS/MS-similar neighbors, ╬öm/z chemical guesses.
- Wired into `SpectralContext` + prompt block **MS/MS EXPLANATION (offline, fast; not SIRIUS)**.
- Runs on every sample in milliseconds ΓÇö no login, no web services.

### Features (optional, expensive)
- **`sirius.py`**: SIRIUS 6 CLI (`formulas` / `structures` / `summaries`), CSI:FingerID parse.
- Hybrid `source=sirius` only when hits are provided; **not required for default product path**.
- Use SIRIUS only on hard cases / ablations, not every spectrum.

### Scripts
- `scripts/run_sirius_on_pack.py` ΓÇö optional batch SIRIUS on blind packs.
- `scripts/apply_sirius_product_to_pack.py` ΓÇö optional fusion with precomputed CSI hits.

### Notes
- Prefer offline MS/MS explain + NIST + neighbors for production throughput.
- CSI:FingerID needs CLI login; slow on large molecules; academic free / commercial Bright Giant.

## 0.2.0 ΓÇö annotation propagation product path

### Features
- **Spectral library reverse search** (`library_search.py`): stream large MGFs (NIST2023) into a precursor-binned pickle index; reverse search by precursor window + ΓêÜI cosine.
- **Method card + RT priors** (`method_card.py`): chromatography/polarity/ionization priors; relative RT compatibility scoring.
- **Hybrid candidates** (`candidates.py`): fuse NIST hits Γê¬ mass-consistent network neighbors Γê¬ LLM SMILES; product selection with provenance `source`.
- **MGF metadata in spectral context**: seed/neighbor `RTINSECONDS`, `MSV_LIB`, `FILENAME`, `CLUSTERSIZE`, ion mode ΓåÆ prompts and ranking.
- **Prompts**: experimental method block + NIST hit list; neighbor lines show RT/MSV when available.
- **Pipeline**: `predict_ego` / `predict_from_graphml` options `use_library_search`, `library_index_path`, `method_card`, `use_hybrid_ranker`.

### Scripts
- `scripts/build_library_index.py` ΓÇö build NIST/custom MGF index.

### Notes
- Full NIST index build is offline one-time cost (~GB-scale MGF).
- Pure-model ablation: `use_hybrid_ranker=False`, `use_neighbor_rescue=False`, `use_library_search=False`.

## 0.1.x
- Initial ego-network LLM prediction, neighbor rescue, publication batch tools.
