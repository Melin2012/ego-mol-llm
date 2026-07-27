# Changelog

## 0.3.0 — SIRIUS / CSI:FingerID spectral structure evidence

### Features
- **`sirius.py`**: write `.ms` spectra, run SIRIUS CLI (formula + structure + write-summaries), parse CSI:FingerID TSV/CSV/JSON.
- **Hybrid candidates**: `source=sirius` with CSI/confidence fusion score; product provenance `sirius_csi`.
- **Prompts**: optional SIRIUS/CSI block from `ego.meta["sirius_hits"]`.
- **`predict_ego`**: flags `use_sirius`, `sirius_bin`, `sirius_work_dir`, `sirius_hits`, `sirius_parse_existing_only`.

### Scripts
- `scripts/run_sirius_on_pack.py` — batch SIRIUS on blind-pack seed MGFs → `sirius_hits/`.
- `scripts/apply_sirius_product_to_pack.py` — post-hoc product fusion with NIST + SIRIUS + neighbors (no LLM).

### Notes
- CSI:FingerID web services: academic free; commercial → Bright Giant. Client is AGPL.
- Set `SIRIUS_BIN` or pass `--sirius-bin`; run `sirius login` once before structure search.

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
