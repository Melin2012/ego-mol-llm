# Draft email — Alexander (forward to student)

**Subject:** ego-mol-llm — Grok free-form results on MSG HNSW **test64** (Claude running next)

---

Hi Alexander,

Quick update for you + the student. Protocol and packs are frozen; we now have a **scored free-form Grok run** on the preferred small set (**official MSG `fold=test` spectra inside our HNSW dump, n = 64, no IK1 dedupe**). **Claude free-form on the same 64 is running now** — I’ll send numbers when it finishes.

### Latest result (primary arm to cite)

**Pack:** `MSG_HNSW_test64_nist_neigh_*`  
**Model arm:** Grok free-form + **neighbor-only NIST** (seed never library-searched)  
**Validity:** valid free-form (per-sample reasoning, ~40 min pack span, mass-first; not the old bulk/m/z-cluster dumps)

| Metric | Grok free-form (test64) |
|--------|------------------------:|
| n | **64** (0 missing) |
| **Structure hit (exact SMILES or IK1)** | **20/64 (31.2%)** |
| Exact SMILES | 15/64 (23.4%) |
| Same connectivity only (IK1, not exact) | 5/64 (7.8%) |
| Same-formula isomer (not IK1) | 7/64 (10.9%) |
| Near-or-better (hit + isomer + T≥0.7) | **27/64 (42.2%)** |
| Any formula match | 27/64 (42.2%) |
| Clear distant miss (T&lt;0.3) | 35/64 (54.7%) |
| Pred mass-OK vs precursor (common adducts) | ~95% |
| Unique-molecule IK1 any-hit | 16/53 (30.2%) |

**Outcome breakdown (for tables / paper):**
- exact 15  
- same connectivity (stereo/tautomer/representation) 5  
- same-formula isomer 7  
- moderate / weak similarity (0.3–0.7 T) 2  
- distant miss 35  

Full row-level table (hits + misses + isomers, color-coded Excel):  
`Desktop\MSG_HNSW_test64_GROK_RESULTS_FULL.xlsx`  
(also under sealed: `MSG_HNSW_test64_nist_neigh_SEALED_truth\GROK_HANDOUT_TEST64_FULL_TABLE.xlsx`)

### How to read this (please pass to student)

1. **This is the hard, non-redundant slice** — 64 spectra / ~53 unique molecules. It is **not** comparable 1:1 to the old strict **285** pack (~18 unique molecules, high replicates), where Grok free-form no-NIST was ~**36% IK1** at spectrum level.  
2. **Network ceiling:** free-form here is largely “reason over the ego.” When the true structure is **absent** from neighbor SMILES, hit rate collapses; when present, retrieval is imperfect (wrong analog / isomer). Expect many **distant misses** even with good mass discipline.  
3. **Do not mix** invalid bulk Grok returns (full 11 540 or “finished in minutes” packs) into free-form tables. Those are ranker-style ablations only.  
4. **Claude on the same test64** is in progress under the same prompts/protocol — we will report the same category table (exact / IK1 / isomer / T bands / distant miss).  
5. Still **not** full official MSG test (17 556); only the **64 test-fold** spectra that fall in HNSW block 219564–231103.

### Algorithm (one line)
GraphML ego + MGF → hide seed → neighbor ranking + offline MS/MS explain → optional **neighbor-only NIST** → free-form LLM (MASS→NETWORK→MS/MS→CHEMISTRY→DECISION) → one SMILES JSON → score vs sealed truth. MSG HNSW seed node always **9999999**; link by file-order index **N**.

### Repo / docs
https://github.com/Melin2012/ego-mol-llm  

- `paper/RESULTS_BOARD.md` / `.docx` — frozen numbers  
- `paper/METHODS.md` / `.docx` — methods  
- `paper/HANDOFF_ALEXANDER.md` / `.docx` — paths + student checklist  
- Scorers: `scripts/_score_msg_test64_grok_handout.py`, `scripts/export_test64_results_table.py`

### Student asks (unchanged)
1. Keep **free-form vs ranker vs invalid bulk** in separate columns.  
2. Report spectrum-level **and** unique-molecule metrics.  
3. When Claude finishes, same scoring categories as the Excel above.  
4. Optional next: deterministic product ranker on the same 64 for a non-LLM baseline.

Happy to walk through the Excel + failure modes (true-not-in-network vs wrong isomer) on a short call.

Best,  
Alexey

---

### Attach / share locally

- Results Excel: `Desktop\MSG_HNSW_test64_GROK_RESULTS_FULL.xlsx`  
- Handout (model-facing): `Desktop\MSG_HNSW_test64_nist_neigh_handout` or `...\MSG_HNSW_test64_nist_neigh_GROK_handout`  
- Sealed (offline only): `Desktop\MSG_HNSW_test64_nist_neigh_SEALED_truth`  
- Zip for operators: `Desktop\MSG_HNSW_test64_nist_neigh_GROK_handout.zip`  

*(Do not ship sealed truth with model zips.)*
