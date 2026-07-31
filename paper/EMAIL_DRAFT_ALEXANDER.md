# Draft email — Alexander (forward to student)

**Subject:** ego-mol-llm handoff — frozen protocol, packs, results board, student benchmarks TODO

---

Hi Alexander,

Please pass this package to your student to **finish the benchmarks** for the network + free-form LLM MS/MS structure paper. Product path, blind packs, and evaluation protocol are frozen on my side; remaining work is clean scoring tables and completing free-form / ranker arms under the same rules.

### Repo
https://github.com/Melin2012/ego-mol-llm  
(latest commit on master will include methods + handoff docs + pack builders / runners)

Key docs in-repo:
- `paper/METHODS.md` — manuscript methods  
- `paper/RESULTS_BOARD.md` — frozen numbers (what is valid free-form vs invalid bulk)  
- `paper/HANDOFF_ALEXANDER.md` — algorithm, paths, student checklist  

### Algorithm (one paragraph)
Unknown seed in a spectral ego-network (GraphML + MGF) → hide seed structure → rank neighbors → optional **neighbor-only NIST** (seed never library-searched) → **free-form LLM** reasons MASS→NETWORK→MS/MS→CHEMISTRY→DECISION and outputs one SMILES JSON — scored offline vs sealed truth (IK1 / exact SMILES / formula / Tanimoto). Deterministic mass-OK **ranker** is a separate ablation. For MassSpecGym HNSW, seed node is always **9999999**; link to MSG is by **file-order index N** in `HNSW_spectrum_N` (fold labels come from the official MSG index after that join — GraphML itself has no fold).

### What is ready
1. **Strict blind 285** pack (HNSW-available MSG test subset) + sealed truth + neighbor-NIST prompts.  
2. **Full 11 540** HNSW block handout with neighbor NIST precomputed + sealed truth + Grok prompt zip (~89 MB).  
3. QC: index-N link **100%** PEPMASS match; redundancy ~**1781** unique molecules / 11 540 spectra.  
4. Local runners (Ollama) + product ranker scripts in repo.

### Important caveats (please stress to the student)
- **11 540 ≠ official MSG test (17 556).** Under official MSG folds this block is mostly **train** (11 386 / 90 / 64 train/val/test). We will re-run when fuller HNSW coverage of the true test fold is available.  
- Several **fast Grok “full pack” returns are invalid free-form** (bulk mass-fit / m/z cluster in minutes). Do **not** put them in free-form tables. Credible free-form baseline on 285: Grok **~36% IK1** without neighbor NIST. Claude pilot **~86% on 64/285** then stopped (~8M tokens). Qwen local **~14% on 87/285** partial.  
- Free-form on all 11 540 via **frontier API** is the intended completion path (Sonnet/Grok-class cost order hundreds–low thousands USD if thinking is controlled; Opus can be much higher). Deduping to 1 spectrum per IK1 (~1.8k) cuts cost ~6×.

### Student asks
1. Finish ranker + free-form tables on frozen packs (spectrum + unique-molecule metrics).  
2. Keep free-form vs ranker vs invalid bulk in **separate** columns.  
3. Coordinate API budget with you; I can help wire the batch runner if needed.

Happy to jump on a short call with you + the student to walk through folder layout and scoring.

Best,  
Alexey

---

### Paths to attach / share (local Desktop / HNSW_Large_Files)

- Code: GitHub push after commit  
- Docs: `paper/HANDOFF_ALEXANDER.md`, `RESULTS_BOARD.md`, `METHODS.md`  
- Packs: `MSG_HNSW_blind285v2_strict_ego_msms` (+ `_SEALED_truth`)  
- Full handout: `MSG_HNSW_full11540_nist_neigh_handout` (+ sealed); zip `MSG_HNSW_full11540_nist_neigh_GROK_handout.zip`  
- QC: `HNSW_Large_Files\MSG_FULL_NEW_ANALYSIS\`  

*(Sealed truth stays off public model shares.)*
