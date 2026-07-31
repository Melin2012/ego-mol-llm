# Handoff: ego-mol-llm evaluation → Alexander / student

**From:** Alexey (product + packs + protocol freeze)  
**To:** Alexander (pass to student for finishing benchmarks)  
**Repo:** https://github.com/Melin2012/ego-mol-llm  
**Software:** `ego-mol-llm` (network-assisted MS/MS → free-form LLM / ranker)

---

## 1. Algorithm (product path — frozen)

```
GraphML ego + seed MGF + subgraph MGF
    → hide seed structure
    → neighbor ranking + offline MS/MS explain
    → optional NEIGHBOR-ONLY NIST (seed NIST OFF)
    → free-form LLM (MASS→NETWORK→MS/MS→CHEMISTRY→DECISION)  OR  mass-OK ranker
    → JSON SMILES
    → score vs SEALED truth (IK1 / exact / formula / T≥0.7)
```

**Critical implementation details:**

| Rule | Detail |
|------|--------|
| MSG HNSW seed node | Always **`9999999`** — do **not** use `find_seed()` hub heuristic on MSG HNSW (often wrong) |
| Link GraphML → MSG | **`HNSW_spectrum_N` ↔ file-order index N** in full `MassSpecGym.mgf` / `MassSpecGym_spectrum_index.csv` |
| Fold labels | From official MSG index **after** join — **not** inside GraphML/MGF |
| Neighbor NIST | Reverse-search neighbor peaks only; inject NIST block into prompt |
| Free-form validity | One sample at a time; ban pack-wide SMILES index / m/z-cluster bulk writers |
| Subgraph MGF | Neighbors only; seed MS/MS from full MSG (or blind seed MGF) |

---

## 2. Code map (repo)

| Area | Path |
|------|------|
| Core library | `src/ego_mol_llm/` (`ego`, `mgf`, `prompts`, `predict`, `library_search`, `validate`, …) |
| Strict 285 pack builder | `scripts/build_msg_hnsw_blind285_strict.py` |
| Neighbor-NIST refresh | `scripts/refresh_pack_nist_neighbors_only.py` |
| Full 11 540 handout builder | `scripts/build_msg_hnsw_full_new_nist_handout.py` |
| IK1 dedupe (1 per molecule per fold) | `scripts/dedupe_msg_hnsw_handout_by_ik1.py` |
| Ollama / OpenAI-compatible free-form runner | `scripts/run_jobs_prompt_ollama.py`, `scripts/run_blind_pack_ollama.py` |
| Product ranker (40c/40d style) | `scripts/run_product_neighbor_mgf_40c.py` (pattern) |
| MSG split / index | `scripts/split_massspecgym_mgf.py` |
| Methods + board | `paper/METHODS.md`, `paper/RESULTS_BOARD.md` |

Install: `pip install -e ".[api]"` (and RDKit for scoring).

---

## 3. Data locations (Alexey’s machine — copy or re-export)

| Dataset | Path |
|---------|------|
| Strict 285 pack | `Desktop\MSG_HNSW_blind285v2_strict_ego_msms` |
| Strict 285 sealed | `Desktop\MSG_HNSW_blind285v2_strict_ego_msms_SEALED_truth` |
| Full 11 540 handout | `Desktop\MSG_HNSW_full11540_nist_neigh_handout` |
| Full 11 540 sealed | `Desktop\MSG_HNSW_full11540_nist_neigh_SEALED_truth` |
| Grok prompts zip (full) | `Desktop\MSG_HNSW_full11540_nist_neigh_GROK_handout.zip` |
| **IK1-deduped handout (n=1780)** | `Desktop\MSG_HNSW_dedup_ik1_nist_neigh_handout` |
| **IK1-deduped sealed** | `Desktop\MSG_HNSW_dedup_ik1_nist_neigh_SEALED_truth` |
| **Grok zip (deduped)** | `Desktop\MSG_HNSW_dedup_ik1_nist_neigh_GROK_handout.zip` |
| Dedupe script | `scripts/dedupe_msg_hnsw_handout_by_ik1.py` |
| Raw GraphML/MGF | `HNSW_Large_Files\graphmls_new\graphmls`, `…\MassSpecGym_subgraph_mgfs_new\…` |
| MSG split / index | `Downloads\MassSpecGym_split\` |
| Link docs (old 8556) | `Downloads\MassSpecGym_linked\` |
| QC + redundancy | `HNSW_Large_Files\MSG_FULL_NEW_ANALYSIS\` |
| NIST index | `Codes\ego-mol-llm\libraries\LEVEL2_NIST2023MSMS_20240408.index.pkl` |

**Do not** ship sealed truth to external model operators.

---

## 4. Frozen numbers (see RESULTS_BOARD.md)

- **Valid free-form baseline (285 strict, no neigh NIST):** Grok IK1 **102/285 (35.8%)**.
- **Claude free-form + neigh NIST:** 64/285 only, IK1 **85.9%** interim — token cost killed full run.
- **Qwen3:14b + neigh NIST:** 87/285, IK1 **13.8%** (partial).
- **Invalid Grok bulk** on 285 and 11 540: high or mid scores but **not free-form** (seconds–minutes for full packs).
- **11 540 block ≠ official MSG test (17 556)**; official folds inside block: train/val/test = **11386/90/64**.
- **Redundancy:** 11 540 spectra → **1780** unique IK1 (~6.5×).
- **IK1-deduped pack (ready):** n=**1780** (train **1652** / val **75** / test **53**); prefer for free-form API.

---

## 5. Student checklist (finish benchmarks)

### Required
- [ ] Reproduce scoring scripts on sealed truth (IK1 / exact / formula / T≥0.7).
- [ ] Run **deterministic product ranker** on frozen 285 strict pack (and optionally 11 540).
- [ ] Complete **one valid free-form arm** on 285 with neighbor-NIST prompts (API frontier model with output caps) **or** document cost stop.
- [ ] Report **spectrum-level and unique-molecule** metrics.
- [ ] Separate tables: free-form vs ranker vs invalid bulk.

### Strongly recommended
- [ ] Free-form on **deduped pack** (`MSG_HNSW_dedup_ik1_nist_neigh_handout`, n=1780) — already 1 spectrum per IK1 per fold.
- [ ] Start with **test-only** (`ids_test.txt`, n=53) then val (75) then train (1652).
- [ ] When **new full MSG HNSW test coverage** arrives: re-link with same seed/`N` protocol; do not assume entire dump is `FOLD=test`.

### Do not
- [ ] Cite 5-minute / 1.6-minute Grok dumps as free-form.
- [ ] Use `find_seed()` instead of `9999999` on MSG HNSW.
- [ ] Put sealed SMILES in model-facing zips.

---

## 6. How to score a predictions folder

```python
# Pattern used throughout: RDKit IK1 / can / formula / Morgan2 Tanimoto
# Sealed: truth_index.csv with true_smiles, fold, msg_identifier, hnsw_index
# Preds: <spectrum_id>.json with "smiles" field
```

Existing helpers: `scripts/_score_grok_strict_v2.py`, scoring blocks in session outputs under sealed dirs (`*_RESULTS.json`).

---

## 7. Contact / ownership

| Role | Person |
|------|--------|
| Product, packs, protocol, invalid-run audit | Alexey |
| Benchmarks, student execution, final tables | Alexander → student |
| Code repo | Melin2012/ego-mol-llm |

Questions on packs/paths: Alexey. Questions on compute budget / API keys for student: Alexander.
