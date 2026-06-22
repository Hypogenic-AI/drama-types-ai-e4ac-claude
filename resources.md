# Resources Catalog — *Different Kinds of Drama*

## Summary

Resources gathered to study **how LLMs represent what different people find
dramatic**, and whether the dramatic-vs-melodramatic distinction is a *threshold
on a single axis* (H1: a low-dimensional drama-feature space) or a
*reader-feature space containing drama subspaces* (H2).

- **Papers**: 33 curated → **27 PDFs downloaded** (6 abstract-only); selected
  from 471 unique results across 3 relevance-ranked searches.
- **Datasets**: **2 downloaded** (EmoBank, GoEmotions-raw) — both provide
  **per-reader / per-annotator** affective labels; 5 more documented.
- **Code**: 0 external repos cloned (none specified); 4 gathering scripts written
  + 7 recommended repos documented for the runner.

---

## Papers

Total curated: 33 · Downloaded: 27 · See `papers/README.md` and
`papers/download_manifest.json`.

Selection spans four pillars:

| Pillar | Example papers | Role |
|--------|----------------|------|
| **Representation geometry: single direction vs subspace** | Linear Representation Hypothesis (Park'23); Refusal=single direction (Arditi'24) **vs** Refusal=concept cones (Wollschläger'25); Beyond Single Concept Vector / Gaussian subspace ('24); From Directions to Regions / MFA ('26); Geometry of Harmfulness subspace ('25) | Tools + the H1↔H2 debate transposed to geometry |
| **Affect / emotion latent space** | Emotions Where Art Thou ('25, low-D emotion manifold); Do LLMs "Feel"? emotion circuits ('25); Hierarchical Emotion Organization ('25); Linear Representations of Sentiment ('23) | Drama's substrate is low-D & steerable |
| **Subjectivity / per-reader modeling** | LLMs for Subjective Language: Survey ('25); Beyond Demographics ('25); PALS / Personalized Chuckles / Personalized Hate-Speech (Kanclerz/Bielaniewicz); Multi-Perspective LLM Annotations ('26); Modeling Subjectivity in Cognitive Appraisal ('25) | Parameterize the *reader* space |
| **Narrative drama / reader response** | Human-Level Narratives? ('24, arcs/suspense/arousal); Emotional Trajectories in Stories ('24); Is the Top Still Spinning? ('25); Tears or Cheers?/CEDAR ('26) | The "drama" end + reader-group variance |

| Title | Year | File / status |
|-------|------|---------------|
| The Linear Representation Hypothesis and the Geometry of LLMs | 2023 | abstract-only |
| Beyond Single Concept Vector: Gaussian Concept Subspace | 2024 | ✅ pdf |
| The Geometry of Truth | 2023 | abstract-only |
| Linear Representations of Sentiment in LLMs | 2023 | ✅ pdf |
| Refusal in LMs Is Mediated by a Single Direction | 2024 | ✅ pdf |
| The Geometry of Refusal: Concept Cones & Representational Independence | 2025 | ✅ pdf |
| From Directions to Regions (MFA local geometry) | 2026 | abstract-only |
| The Geometry of Harmfulness through Subconcept Probing | 2025 | abstract-only |
| Extracting Latent Steering Vectors | 2022 | ✅ pdf |
| A Structural Probe for Finding Syntax | 2019 | ✅ pdf |
| Emergent Linear Representations in World Models | 2023 | ✅ pdf |
| Semantic Structure of Feature Space in LLMs | 2026 | abstract-only |
| H-Probes: Hierarchical Structures from Latent Reps | 2026 | ✅ pdf |
| Sparse Feature Circuits | 2024 | ✅ pdf |
| Linear Representations of Political Perspective | 2025 | ✅ pdf |
| The Granularity Axis (social roles) | 2026 | ✅ pdf |
| Tensor Product Representation Probes | 2026 | abstract-only |
| Emotions Where Art Thou (emotional latent space) | 2025 | ✅ pdf |
| Emergence of Hierarchical Emotion Organization | 2025 | ✅ pdf |
| Do LLMs "Feel"? Emotion Circuits | 2025 | ✅ pdf |
| LLMs for Subjective Language Understanding: A Survey | 2025 | ✅ pdf |
| Beyond Demographics (individual perceptions) | 2025 | ✅ pdf |
| PALS: Personalized Active Learning for Subjective Tasks | 2023 | ✅ pdf |
| Capturing Human Perspectives in NLP | 2023 | abstract-only |
| Multi-Perspective LLM Annotations for Subjective Tasks | 2026 | ✅ pdf |
| Modeling Subjectivity in Cognitive Appraisal | 2025 | ✅ pdf |
| What If Ground Truth Is Subjective? (personalized hate speech) | 2022 | abstract-only |
| Annotating and Training for Population Subjective Views | 2023 | ✅ pdf |
| Are LLMs Capable of Generating Human-Level Narratives? | 2024 | ✅ pdf |
| Is the Top Still Spinning? (narrative subjectivity) | 2025 | ✅ pdf |
| Tears or Cheers? / CEDAR | 2026 | ✅ pdf |
| Modeling Emotional Trajectories in Written Stories | 2024 | ✅ pdf |
| From Generalized Laughter to Personalized Chuckles | 2023 | ✅ pdf |

---

## Datasets

Total downloaded: 2. See `datasets/README.md` (full download + load instructions).
Large files are git-ignored; `samples.json` committed for each.

| Name | Source | Size | Task | Location | Why chosen |
|------|--------|------|------|----------|-----------|
| **EmoBank** | JULIELab GitHub (CC-BY-SA) | 10,062 sents; 53,055 per-reader VAD ratings | VAD regression, bi-perspective | `datasets/emobank/` | **Reader vs writer** perspectives + **per-individual** reader VAD — the cleanest fit; Arousal ≈ drama intensity |
| **GoEmotions (raw)** | HuggingFace (Apache-2.0) | 211k annotations; 58k texts; **82 raters** | 28-way multi-label, multi-annotator | `datasets/go_emotions/` | Real **inter-reader disagreement** at scale; `rater_id` → per-reader embeddings |

Documented extensions (not downloaded): WASSA story trajectories (Christ'24),
CEDAR (Dai'26), LeWiDi suite, **Emotions in Drama** (German plays, Dennerlein'23),
SEV (Wang'25). See `datasets/README.md`.

---

## Code Repositories

Total cloned: 0 (no `code_references` specified; large frameworks are pulled
selectively by the runner). See `code/README.md`.

Gathering scripts written: `curate.py`, `select_top.py`, `download_papers.py`,
`retry_known.py`. Recommended external repos: TransformerLens, representation-
engineering (RepE), refusal_direction, feature_circuits, EmoBank,
emotional_trajectories_stories.

---

## Resource gathering notes

### Search strategy
Three relevance-ranked paper-finder runs (1 fast + 2 diligent) on complementary
framings: (a) *drama/melodrama & affective representation*, (b) *representation
geometry / subspaces / probing-steering*, (c) *annotator subjectivity /
reader-perspective NLP*. 471 unique results → curated to 33 by relevance (≥3) and
a drama/affect/reader keyword filter.

### Selection criteria
Prioritized: (1) papers that make the single-direction-vs-subspace question
concrete (the geometric form of the hypothesis); (2) direct LLM-affect-geometry
evidence; (3) per-reader/subjective modeling methods + datasets; (4) narrative
drama operationalizations. Balanced foundational (2019–23) and SOTA (2024–26).

### Challenges & workarounds
- Paper-finder needed `httpx`; main "drama" query timed out in diligent mode →
  used fast mode. A stray `| head -5` truncated two searches → re-ran with full
  capture (the script auto-saves JSONL).
- Semantic Scholar rate-limiting (429) caused first-pass NO-PDFs → added a
  direct-arXiv-ID retry + a second S2 pass; reconciled the manifest against disk
  and deduped filename-length collisions. Final: 27/33 PDFs; 6 recent/paywalled
  papers remain abstract-only.

### Gaps
No paper studies "drama" or dramatic-vs-melodramatic directly — confirming the
niche. The single-axis-vs-subspace question is unsettled even for well-studied
concepts (refusal), making drama a clean adjudication case.

---

## Recommendations for experiment design

1. **Primary dataset**: EmoBank (per-reader VAD; Arousal as drama-intensity
   proxy; reader-spread as "drama vs melodrama"). **Scale-up**: GoEmotions-raw
   (82 raters → per-reader embeddings). Optionally a small controlled
   drama↔melodrama stimulus set (SEV-style synthetic rewrite).
2. **Baselines**: single-direction (diff-in-means) drama probe + ablation (H1);
   majority-vote; sociodemographic conditioning (known-weak); PCA/SAE; shuffled-
   reader and random-direction controls.
3. **H1 vs H2 tests**:
   - Estimate the **rank** of a drama subspace built from many drama subconcepts
     (Geometry-of-Harmfulness / Gaussian-Concept-Subspace style).
   - Compare **single-axis + reader-threshold** vs **reader-space ⊗ drama
     subspace** (TPR / region-MFA probes) for predicting per-reader drama labels.
   - Use **stress/distortion** metrics (Emotions-Where-Art-Thou) to test whether
     two readers' drama geometries differ only by global rescale (H1-ish) or
     anisotropically (H2).
4. **Causal validation**: steer along candidate drama axes/subspaces and reader
   directions; check held-out-reader and cross-dataset transfer.
5. **Metrics**: probe AUROC/CCC, subspace rank / participation ratio, ablation &
   steering effect sizes, per-reader gain over majority vote, representational-
   independence under intervention, cross-reader/-dataset transfer.
6. **Care**: use Park's causal inner product for geometry; treat disagreement as
   signal; difference-in-means + ablation for robust causal claims.
