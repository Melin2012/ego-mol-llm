# Methods

*Manuscript-ready methods for ego-network LLM structure annotation and multi-model blind evaluation. Software: ego-mol-llm (product path v0.2–v0.5). Adapt journal style as needed.*

---

## Overview

We developed a **network-assisted MS/MS structure assignment** workflow in which an unlabeled precursor is annotated using its **ego neighborhood** in a spectral molecular network, optional spectral library reverse search, offline fragment explanation (rule-based and/or CFM-ID), and a **frontier chemistry LLM as a free-form reasoner** over that evidence. The product design treats the network and spectrum as **primary evidence** and the LLM as the structure-calling step—not pure de novo elucidation without network context, and not closed-set ranking alone.

Complementary ablations include deterministic **product rankers** (mass-gated neighbor/library voting), **SIRIUS/CSI:FingerID**, and **CFM-ID network fragment transfer**, used either as prompt blocks or as ranking signals.

---

## Molecular network and spectrum inputs

**GraphML.** Spectral networks were provided as GraphML (GNPS / HNSW-style). Nodes carried precursor mass (`PEPMASS`), optional library name and SMILES, community labels, and direct-neighbor flags. Edges stored MS/MS cosine similarity and absolute precursor Δ*m/z*. For MassSpecGym HNSW egos, the query was encoded as a fixed seed node (`id=9999999`) with PEPMASS equal to the query precursor; library nodes retained names/SMILES.

**MGF.** Seed and subgraph MS/MS spectra were supplied as MGF. Subgraph spectra were keyed by `NETWORK_NODE_ID` matching GraphML node ids. Seed spectra for blind packs used precursor *m/z*, ion mode, and peak lists only (structure metadata removed).

**Library reverse search (optional).** When enabled, seed spectra were reverse-searched against NIST2023 MS/MS (LEVEL2 index) with precursor mass filtering and match-score ranking. Product policy treats seed NIST as **optional verification** (high score + mass-OK may support a structure; low scores are ignored and must not force an ID). Neighbor-only NIST can be used for ablations without seed reverse search.

---

## Blind ego-context construction

For each query seed:

1. Load GraphML and resolve the seed (explicit seed id, or hub heuristic).
2. Build a one-hop (and optional two-hop) ego with **seed name/SMILES hidden** in blind mode.
3. Rank neighbors by a mass-aware evidence score combining edge cosine, near-isobar / multimer residuals, and annotation richness (default top *N* = 25–50 for prompts).
4. Attach spectral context: seed peaks, neighbor peaks, MS/MS cosine to seed, offline MS/MS explanation (diagnostics, neutral losses, shared peaks vs top neighbors).

**Mass / adduct gate.** Candidate structures must fit observed precursor *m/z* under even-electron monomer and multimer adducts, including multi-water losses (e.g. [M+H−*n*H₂O]⁺) when needed. Default monomer mass tolerance was 0.05 Da unless stated. Multimer relationships used a tight residual gate (default 0.10 Da) for self-consistent [2M…]/[3M…] hypotheses.

---

## Free-form LLM structure assignment (product path)

Frontier models received a fixed **system prompt** specifying: network-assisted annotation; mass-first rules; multimer awareness; use of MS/MS and dual cosine; optional NIST as soft evidence; required JSON schema (`smiles`, `iupac_or_common_name`, `formula`, `adduct`, `confidence`, `rationale`, `alternatives`).

User prompts contained only **that sample’s** ego evidence (query *m/z*, MS/MS, neighbors, optional CFM/SIRIUS blocks). Models were instructed to:

1. Propose one **neutral monomer** SMILES with mass-consistent adduct.
2. Prefer dual-cosine near-isobars and MS/MS-consistent scaffolds.
3. Treat NIST mid-scores as soft clues; never force low-score library IDs.
4. Cite mass, network, and MS/MS in the rationale; place close isomers in `alternatives`.

**Strict free-form protocol (evaluation arms).** For multi-model trials, prompts further required an explicit reasoning checklist (MASS → NETWORK → MS/MS → CHEMISTRY → DECISION) and banned pack-wide SMILES indices, mass-transfer batch scripts, and external registry lookup. Strict-blind packs redacted database accessions (e.g. MassSpecGymIDs) from model-facing files; sealed truth retained full linkage offline.

Models evaluated included **Claude Opus 5** and **Grok** free-form sessions writing one JSON prediction per spectrum. Deterministic **product rankers** (mass-OK neighbor SMILES ± high-score seed NIST ≥ 0.85 ± CFM neighbor structures) were scored separately as ablations, not as free-form LLM accuracy.

---

## Optional precomputes injected into prompts

**SIRIUS / CSI:FingerID.** When configured, SIRIUS (v6.x CLI) was run on seed spectra (formula → fingerprints → compound classes → structures). Top CSI hits were serialized into optional prompt blocks; multi-step chaining avoided incorrect nested `structures` invocation.

**CFM-ID network-first explanation.** For selected neighbors with known SMILES, CFM-ID (Docker `wishartlab/cfmid`) annotated experimental peaks; fragment SMILES were transferred onto seed peaks at matching *m/z*. Neighbor annotations and seed peak maps were written to `cfm_explain/` and optionally injected into jobs/prompts before LLM assignment. CFM was **not** used as a closed-set ID by itself.

---

## Blind evaluation packs

### ASTRAL / ego holdout packs (n = 40)

Randomized holdouts (e.g. holdout-40c/40d) were drawn from ASTRAL C18 networks with prior sets excluded. Each pack included GraphML, blind seed MGF, subgraph MGF, jobs/prompts, and a **sealed** truth index (local only). Polarity was resolved per seed from MGF ion mode. Multi-model folders (`predictions_*`) were scored after all models finished.

**Evidence arms (examples).** Pure network; network without seed NIST; MGF-only ± NIST; SIRIUS-first ± NIST; CFM-first; product neighbor+MGF ranker (NIST high-score verify only). Metrics: InChIKey first block (IK1), exact canonical SMILES, formula match, Morgan-2 Tanimoto (T ≥ 0.7 / 0.85).

### MassSpecGym HNSW subset (n = 285)

MassSpecGym spectra with precomputed HNSW ego GraphML and subgraph MGF were linked by **full-file index N** (`HNSW_spectrum_N` ↔ position N in `MassSpecGym.mgf` / official spectrum index). A blind pack used structure-stripped seed MGF from the MassSpecGym test blind export. Overlap with the full official test fold was partial (**285 of 17 556** test spectra in the early HNSW index range 0–8555); results are reported for this HNSW-available subset only. A **strict v2** rebuild removed MassSpecGym accessions from model-facing prompts and baked free-form reasoning requirements into every prompt for independent multi-model re-runs.

**Neighbor-only NIST (optional evaluation arm).** Neighbor MS/MS peaks were reverse-searched against NIST2023 MS/MS; hits (name, InChIKey/SMILES when available, match score) were injected as a prompt block. The **seed was never reverse-searched** in this arm. Free-form models were instructed to treat neighbor NIST as network evidence only and to promote a structure to the seed only if mass-consistent with the seed precursor.

**Free-form validity.** Valid free-form arms require sample-by-sample model consumption of each prompt (wall time and rationale diversity consistent with LLM generation). Bulk writers that mass-fit SMILES from pack prompts in minutes (or that self-report m/z clustering) were archived as **ranker-style ablations** and excluded from free-form accuracy claims.

### MassSpecGym HNSW full block (n = 11 540)

A second HNSW dump covered indices **219564–231103** (11 540 contiguous file-order positions at the end of the full MassSpecGym MGF). Linkage and seed PEPMASS checks were 100% consistent with the official index. Under **official MSG fold labels** (not present in GraphML; joined after index N), this block comprises approximately **11 386 train / 90 val / 64 test** spectra—not the full official test fold (17 556). The block is chemically redundant at the molecule level (~**1 780** unique InChIKey first blocks; ~6.5 spectra per molecule), with train phospholipids heavily over-represented. Seed MS/MS for prompts was taken from the full MSG MGF; subgraph MGF supplied neighbor peaks only. Neighbor-only NIST free-form handouts were precomputed for the full block for future API free-form runs; bulk automated returns on this pack were not counted as free-form.

**Test-only free-form handout.** The recommended API free-form set is the **official `fold=test` slice of this HNSW block** with **no IK1 dedupe**, preserving the original test count (**n = 64**). A larger optional IK1-deduped all-fold pack (n = 1 780) and the full 11 540 pack remain available for expanded free-form or spectrum-level ranker ablations.

---

## Scoring

Predictions were matched to sealed SMILES with RDKit:

- **IK1:** InChIKey first block equality (connectivity).
- **Exact:** canonical SMILES equality.
- **Formula:** RDKit molecular formula equality.
- **Tanimoto:** Morgan fingerprints, radius 2, 2048 bits.

When sealed InChIKeys disagreed with RDKit(true_smiles), scoring used RDKit-derived keys from true SMILES (and sealed keys were patched for consistency). Metrics were reported overall and, when relevant, stratified by outcome class (both hit, one-model hit, same-formula isomer miss, true miss).

---

## Case study: library *meta*-tyrosine mislabeled as tyrosine

During holdout-40c evaluation, spectrum **AROMEC18COLGATE000057** (precursor *m/z* 182.0811, positive mode, [M+H]⁺) was labeled in the sealed / LEVEL1 metadata as **“Tyrosine”** with formula C₉H₁₁NO₃. The stored SMILES, however, corresponds to **3-hydroxyphenylalanine (*meta*-tyrosine)**:

| | SMILES (canonical) | InChIKey (full) | OH position |
|--|--------------------|-----------------|-------------|
| **Sealed / library structure** | `N[C@@H](Cc1cccc(O)c1)C(=O)O` | `JZKXXXDKRQWDET-QMMMGPOBSA-N` | *meta* (3-) |
| **Standard L-tyrosine (*para*)** | `N[C@@H](Cc1ccc(O)cc1)C(=O)O` | `OUYCCCASQSFEME-QMMMGPOBSA-N` | *para* (4-) |

Network- and CFM-informed arms proposed **para-tyrosine** (IK1 `OUYCCCASQSFEME…`), matching the common biological / library meaning of the name “tyrosine,” exact mass for C₉H₁₁NO₃, and typical aromatic substitution. Relative to sealed SMILES this was scored as an **IK1 miss** with **formula match** and Tanimoto ≈ 0.63—i.e. a **regioisomer error relative to the sealed structure**, not a random false ID.

Supporting context in the LEVEL1 testing table associated this feature with PubChem/CAS entries consistent with **m-tyrosine** (e.g. CAS 587-33-7) while the display name remained “Tyrosine.” Thus the apparent model failure is better interpreted as a **library / ground-truth naming inconsistency** (*meta*-Tyr structure stored under the name of *para*-Tyr). This case motivates (i) scoring against structure (InChIKey/SMILES), not free-text names alone; (ii) reporting same-formula isomer misses separately; and (iii) expert review when models disagree with library regioisomer assignments under strong spectral and network evidence for the common isomer.

---

## Baselines and ablations

1. **Deterministic product ranker:** mass-OK network neighbor SMILES; optional CFM neighbor structures and CSI hits; seed NIST only if match score ≥ 0.85 and mass-OK; deduplicate by canonical SMILES; top score wins (or abstain).
2. **MGF-only ± NIST:** no GraphML; library reverse search only (shows NIST inflation when unrestricted).
3. **SIRIUS/CSI-first** and **CFM-first** ranking over the same packs without free-form LLM.
4. Free-form multi-model comparison on frozen prompts (Opus, Grok, optional Fable/ChemDFM).

---

## Limitations

Network LLM annotation inherits **library annotation errors** (including regioisomer mislabels as above) and **spectral isobar / OH-map confusion** (bile acids, flavonoid glycosides). Multimer and multi-water adduct gates reduce but do not eliminate chance mass hits. Deterministic rankers abstain or err when the correct structure is absent from the local mass-OK set. Free-form LLMs can still fail on hard isomers or weak egos. MassSpecGym HNSW results apply only to spectra with available ego GraphML; partial HNSW coverage must not be reported as full official-test performance. Spectrum-level accuracy on redundant packs overweights frequent molecules—unique-molecule metrics are recommended. Free-form API cost and context length limit full-pack multi-model runs; incomplete free-form pilots should be labeled as such. Expert review remains required for publication-grade IDs.

---

## Software and data availability

Implementation: **ego-mol-llm** (Apache-2.0), Python 3.10+, RDKit for validation and metrics; optional SIRIUS CLI and CFM-ID Docker. Blind packs, sealed truth, and prediction folders follow a fixed layout (`prompts/`, `jobs/`, `predictions_<model>/`, sealed `truth_index.csv`). Exact random seeds, pack versions, and model tags are recorded in pack metadata (`package_meta.json`) and prediction JSON `model` fields.
