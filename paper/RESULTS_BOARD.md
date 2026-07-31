# Results board (frozen evaluation snapshot)

*Internal board for manuscript + student benchmarks. Dates: July 2026. “Valid free-form” means sample-by-sample LLM reasoning without pack-wide mass-fit automation.*

---

## A. Product algorithm (final)

1. **Inputs:** GraphML ego (seed `9999999` for MSG HNSW), subgraph MGF (neighbors by `NETWORK_NODE_ID`), seed MGF (query MS/MS).
2. **Blind ego:** hide seed name/SMILES; rank neighbors (edge cosine + MS/MS cosine + mass geometry); offline MS/MS explanation block.
3. **Optional neighbor-only NIST:** reverse-search **neighbor** spectra vs NIST2023 MSMS; **seed NIST disabled** by protocol.
4. **Free-form LLM:** MASS → NETWORK → MS/MS → CHEMISTRY → DECISION; one JSON SMILES per spectrum; no pack-wide index.
5. **Optional product ranker / rescue / hybrid:** mass-OK neighbor vote; seed NIST only if score ≥ 0.85 (ablations).
6. **Score offline vs sealed truth:** IK1, exact SMILES, formula, Morgan-2 T ≥ 0.7 / 0.85.

**Link rule (MSG HNSW):** `HNSW_spectrum_N` ↔ full `MassSpecGym.mgf` file-order index **N** (not GraphML-internal MassSpecGymIDs). Fold labels (`train`/`val`/`test`) come from official MSG index after that join. GraphML/MGF themselves do **not** store fold.

---

## B. Holdout packs (ASTRAL C18, n = 40)

See prior product tables (40c/40d): product neighbor+MGF ranker ~60% IK1; SIRIUS-first ~55–57.5%; network ~40–43%; MGF+NIST inflated when seed library is unrestricted. Tyr *meta*/*para* library case documented in METHODS.

---

## C. MassSpecGym HNSW subset — strict blind 285

| Pack | Path (Desktop) |
|------|----------------|
| Model-facing | `MSG_HNSW_blind285v2_strict_ego_msms` |
| Sealed | `MSG_HNSW_blind285v2_strict_ego_msms_SEALED_truth` |
| n | 285 spectra / **18 unique true IK1** (high replicate rate) |
| Coverage | 285 / 17 556 official MSG **test** with HNSW in *old* index range only |

### C1. Free-form / local models (strict v2 unless noted)

| Arm | n scored | IK1 | Exact | Notes |
|-----|--------:|----:|------:|-------|
| **Grok free-form, no neighbor NIST** (strict v2) | 285 | **102 (35.8%)** | 90 | Credible free-form arm; empty SMILES 128; unique-mol any-hit 13/18 |
| **Claude Opus free-form + neighbor NIST** | **64 / 285** | **55 (85.9%)** | 52 | Partial pilot only; ~8M tokens / 64; **killed — unsustainable** |
| **Qwen3:14b free-form + neighbor NIST** | **87 / 285** | **12 (13.8%)** | 12 | Partial; max-runtime stop; 12/40 assigned = 30% IK1; slow ~6–10 min/sample |
| ChemDFM-R pure model | 285 | 3 (1.1%) | 2 | Mass-illegal hallucinations; not competitive free-form |

### C2. Invalid / ablation (do **not** cite as free-form)

| Arm | IK1 | Why invalid |
|-----|----:|-------------|
| Grok ranker-fast nist neigh | 195/285 (68.4%) | Helper scripts / bulk mass-fit |
| Grok online m/z-cluster zip | 187/285 (65.6%) | Self-admitted bulk m/z clustering |
| Grok clone of no-NIST freeform | 100/285 (35.1%) | ~10 s relabel of prior free-form |
| v1 free-form 249–252/285 | — | Leaky pack (IDs / structure); superseded by strict v2 |

### C3. Outcome buckets (online Grok m/z-cluster INVALID, for ranker upper bound)

Exact 175 · Similar 16 · Formula-only 10 · True miss 20 · Empty 64 (n=285).

---

## D. MassSpecGym HNSW full block 11 540 (new dump)

| Item | Value |
|------|------:|
| Ego nets | **11 540** (`HNSW_spectrum_219564`…`231103`) |
| Official MSG test fold size | **17 556** |
| Match full test? | **No** |
| Official folds *inside* this block | train **11 386** / val **90** / test **64** |
| Unique IK1 | **1 781** (~**6.5×** spectra per molecule) |
| Top 100 mols cover | **66.5%** of spectra |
| Seed PEPMASS vs MSG precursor | **11 540/11 540** (index-N link OK) |

| Artifact | Path |
|----------|------|
| Handout (local full) | `Desktop\MSG_HNSW_full11540_nist_neigh_handout` |
| Grok zip (prompts) | `Desktop\MSG_HNSW_full11540_nist_neigh_GROK_handout.zip` (~89 MB) |
| Sealed | `Desktop\MSG_HNSW_full11540_nist_neigh_SEALED_truth` |
| QC / redundancy | `HNSW_Large_Files\MSG_FULL_NEW_ANALYSIS\` |

### D0. IK1-deduped handout (recommended free-form set)

**Policy:** 1 spectrum per true InChIKey first block **per official fold**; prefer max neighbor NIST hits, then min HNSW index. Script: `scripts/dedupe_msg_hnsw_handout_by_ik1.py` (copies parent prompts; no NIST rebuild).

| Item | Value |
|------|------:|
| n spectra (= unique IK1) | **1 780** |
| train / val / test | **1 652 / 75 / 53** |
| Parent pack | 11 540 spectra |
| Dropped (replicates) | 9 760 |

| Artifact | Path |
|----------|------|
| Handout (deduped) | `Desktop\MSG_HNSW_dedup_ik1_nist_neigh_handout` |
| Grok zip | `Desktop\MSG_HNSW_dedup_ik1_nist_neigh_GROK_handout.zip` |
| Sealed | `Desktop\MSG_HNSW_dedup_ik1_nist_neigh_SEALED_truth` |
| Fold ID lists | `ids_train.txt` / `ids_val.txt` / `ids_test.txt` in handout |

Use this pack for **unique-molecule free-form** API runs and cost control (~6.5× fewer jobs than full 11 540). Parent full pack remains for spectrum-level / ranker ablations.

### D1. Grok return on full 11 540 (INVALID free-form)

| | |
|--|--|
| Source | `Downloads\MSG_HNSW_full11540_predictions.zip` |
| Wall | ~**1.6 min** for 11 540 → bulk automation |
| IK1 all | **1105/11540 (9.6%)** |
| Empty | 8349 (72%) |
| IK1 official test subset | **21/64 (32.8%)** |
| Use | Ranker-style ablation only — **not** free-form |

### D2. Frontier API cost order (*valid* free-form)

Rough planning (input ~8k tok/sample; solid free-form ~4k out):

| Model (API) | Full 11 540 | Deduped 1 780 |
|-------------|-------------|----------------|
| Claude Opus 5 | ~$1.5k–$5k+ | ~$0.2k–$0.8k+ |
| Claude Sonnet 5 (intro) | ~$0.6k–$2k | ~$90–$300 |
| Grok 4.5 / 4.3 | ~$0.25k–$1.2k | ~$40–$180 |
| DeepSeek V4 Flash | ~$50–$400 | ~$8–$60 |

**Recommendation:** free-form on **deduped 1 780** (or fold-stratified `ids_test.txt` first); meter a 20-sample pilot; prefer Sonnet or Grok API.

---

## E. Student TODO (benchmarks)

1. Finish **valid free-form** on frozen prompts (285 strict and/or **deduped 1 780**).
2. Deterministic **product ranker** on same packs (neighbor mass-OK ± neighbor NIST).
3. Optional SIRIUS/CFM ablations already scripted.
4. Tables: spectrum-level (full 11 540) + unique-IK1 (dedup pack); never mix invalid Grok bulk with free-form rows.
5. When new full MSG HNSW test coverage arrives, re-link and re-run under same protocol.

See `paper/HANDOFF_ALEXANDER.md`.
