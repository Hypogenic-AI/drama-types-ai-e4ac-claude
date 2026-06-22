# Planning — *Different Kinds of Drama*

## Research Question

How do large language models internally represent **what different people find
dramatic**? Specifically, is the distinction between what reader *X* finds
**dramatic** and reader *Y* finds **melodramatic** best modeled as:

- **(H1) Drama-feature space** — a low-dimensional space of *drama features*
  (intensity, stakes, excess…) in which readers differ by **thresholds / regions
  on shared axes**; in the limit, a *single* drama axis with per-reader
  thresholds; **or**
- **(H2) Reader-feature space with drama subspaces** — reader variation is a
  *direction within a multi-dimensional drama subspace* (readers reweight
  sub-features), so "drama vs melodrama" is **not** reducible to a threshold on
  one axis.

We make this concrete as a **geometry-of-representations** question, following the
single-direction-vs-subspace debate in the interpretability literature
(Arditi 2024 "refusal = 1 direction" vs Wollschläger 2025 "refusal = concept
cone"). Drama is a fresh, evaluative, reader-dependent construct on which to
adjudicate.

---

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs increasingly mediate subjective, affect-laden judgments (content moderation,
storytelling, companionship, criticism). If a model collapses "different kinds of
drama" onto a single intensity axis, it will systematically conflate *earned
drama* with *melodramatic excess* and will mis-model readers whose sensibilities
differ in **kind**, not just **degree**. Understanding the internal geometry tells
us whether reader-personalization can be a cheap threshold shift or requires
steering within a genuinely multi-dimensional subspace.

### Gap in Existing Work (from literature_review.md)
1. **No work studies "drama" or the dramatic-vs-melodramatic contrast directly.**
2. The single-axis-vs-subspace question is **unsettled even for well-studied
   concepts** (refusal). Affect is often reported as a *low-dimensional manifold*
   (Reichman 2025), but whether **reader variation** rides that manifold as a
   threshold or as a subspace rotation is **untested**.
3. Reader structure is studied **behaviorally** (per-annotator modeling,
   Orlikowski 2025) but never **joined to the model-internal affect geometry**.

### Our Novel Contribution
We test, *inside the model's activation space*, whether (a) "melodrama" is
collinear with "drama" (single axis) or off-axis (subspace); (b) the drama
subspace built from many subconcepts is genuinely low-rank; and (c) **real
readers' disagreement** (GoEmotions, 82 identified raters) is captured by a
per-reader **threshold on a shared axis** (H1) or requires a per-reader
**direction within a drama subspace** (H2) — a clean nested model comparison that
operationalizes the user's exact question.

### Experiment Justification
- **E1 (drama axis & the melodrama residual):** Establishes drama *is* linearly
  decodable and tests the sharpest form of the hypothesis — is melodrama just
  "more drama" (collinear) or a distinct direction? Necessary: it is the minimal
  single-axis-vs-subspace test.
- **E2 (subspace rank via subconcept probing):** Quantifies "low-dimensional" —
  how many independent drama directions exist (participation ratio). Necessary to
  know the *dimensionality* the reader variation could live in.
- **E3 (reader variation: threshold vs subspace — the crux):** Nested model
  comparison on real per-reader labels. Directly answers H1 vs H2. The central
  experiment.
- **E4 (causal steering, confirmatory):** Validates that the discovered
  directions are causal, not just decodable correlations.

---

## Hypothesis Decomposition

| Sub-hypothesis | Prediction if H1 (single/threshold) | Prediction if H2 (subspace) |
|---|---|---|
| SH1: melodrama vs drama geometry | melodrama collinear with drama axis; off-axis residual is unstructured/noise | melodrama has a **structured, decodable off-axis** component |
| SH2: drama subspace rank | rank ≈ 1 (one dominant axis) | rank > 1, low but multi-D (≈3–8) |
| SH3: reader disagreement | per-reader **intercept** (threshold) on a shared axis suffices; adding reader-specific *direction* gives no held-out gain | reader-specific **direction in subspace** beats threshold-only on held-out data |
| SH4: reader-space dimensionality | reader weights ≈ rank-1 (just a gain) | reader weight matrix low-rank but >1 ("reader-feature space") |

---

## Proposed Methodology

### Approach
Mechanistic / representation-geometry probing of an **open-weight instruct model**
(activations required → local model, not API), combined with **real LLM-generated
controlled stimuli** and **real human per-reader labels** (GoEmotions). Causal
validation by activation steering. All geometry uses difference-in-means
directions (robust, causal — Marks & Tegmark 2023) and standard subspace tools
(PCA, participation ratio, principal angles).

### Models
- **Primary:** `Qwen/Qwen2.5-7B-Instruct` (open, 2024 SOTA-class). Internals via
  HuggingFace hidden states (mean/last-token pooled per layer).
- **Scale ablation:** `Qwen/Qwen2.5-1.5B-Instruct` — does dimensionality grow
  with scale (Zhao 2025 hierarchy-deepens-with-scale)?
- **Stimulus generation & behavioral elicitation:** `gpt-4.1` via OpenAI API
  (real model, generates + adversarially validates the drama corpus and produces
  persona-conditioned drama ratings).

### Data
1. **Controlled drama corpus (built here):** base scenarios × registers
   {neutral, understated, dramatic, melodramatic} + per-subconcept contrast pairs
   (suspense, grief/pathos, spectacle, stakes, sincerity, excess/sentimentality,
   shock, restraint). Generated by `gpt-4.1`, then **adversarially validated** by
   a second LLM pass (label agreement filter).
2. **GoEmotions raw** (`datasets/go_emotions/raw_train.parquet`): 211k
   annotations, 58k texts, **82 identified raters** (`rater_id`). Per-rater
   "perceived intensity / drama" derived from emotion labels (arousal-weighted).
   Real inter-reader disagreement — the substrate for E3.
3. **EmoBank** (aggregate VAD; Arousal ≈ drama intensity) — auxiliary external
   validation of the drama axis (individual ratings are anonymized, so used at the
   aggregate level only).

### Experimental Steps
1. **E1 — Drama axis + melodrama residual.** Extract activations for the
   controlled corpus. Diff-in-means drama axis (neutral↔dramatic); AUROC across
   layers. Project melodramatic items on the axis; test the **off-axis residual**:
   is it decodable (drama vs melodrama classifier on residuals) above shuffled/
   random-direction controls? Principal angle between "drama" and "melodrama"
   diff-in-means directions.
2. **E2 — Subspace rank.** One diff-in-means direction per subconcept → matrix of
   K directions. **Participation ratio / explained-variance rank**; principal
   angles; cluster into drama-core vs melodrama/excess families. Compare to random
   directions (control) and to the same analysis for a non-affective concept.
3. **E3 — Reader variation (crux).** GoEmotions: per-rater intensity labels on
   shared texts. Extract text activations. Fit nested models predicting per-rater
   label:
   - **M0** shared single axis + global threshold;
   - **M1 (H1-threshold)** + per-reader **intercept**;
   - **M2 (H1-gain)** + per-reader **scalar gain** (still 1-D);
   - **M3 (H2-subspace)** D-dim drama subspace + per-reader **weight vector**
     (direction within subspace).
   Compare **held-out** (reader-stratified) log-loss / AUROC with bootstrap CIs.
   H2 supported iff M3 beats M2 beyond noise. Then measure **rank of the reader
   weight matrix W** (readers × subspace-dims) → "reader-feature space"
   dimensionality (SH4). Controls: **shuffled-reader** labels.
4. **E4 — Causal steering (confirmatory).** Add ±α·(drama axis) and
   ±α·(melodrama-residual direction) to activations of neutral prompts; have the
   model rate drama/melodrama; check the two directions move *different* axes
   (independence under intervention, Wollschläger).
5. **Persona probe (bonus).** Re-derive the drama axis under several reader
   personas (prompted). Cosine/principal-angle between persona axes: threshold
   shift (parallel axes) vs rotation (H2). Stress/distortion (Reichman 2025).

### Baselines & Controls
- Single-direction diff-in-means probe (H1 baseline) + its ablation.
- Majority/global model (M0) and reader-intercept model (M1) as the H1 foils M3
  must beat.
- **Random-direction** and **shuffled-reader** null controls (CIs must exclude).
- PCA/unsupervised rank as a structure baseline.
- Cross-layer and cross-model (1.5B vs 7B) replication.

### Evaluation Metrics
Probe AUROC / accuracy; **participation ratio & effective rank**; principal
angles / cosine; held-out log-loss & AUROC for nested models; **ΔAUROC(M3−M2)**
with bootstrap 95% CI and permutation p-value; reader-weight-matrix rank; steering
effect sizes (Cohen's d) and cross-direction independence.

### Statistical Analysis Plan
- Bootstrap (1000×) 95% CIs on all key metrics; reader-stratified cross-validation
  for E3 (held-out reader×item pairs, no leakage).
- Permutation tests (shuffled-reader / random-direction) for null distributions;
  report effect sizes, not just p. Significance α = 0.05; Bonferroni across the 4
  sub-hypotheses where multiple tests apply.
- Pre-registered decision rule: **H2 supported** iff (SH1 residual decodable >
  control) AND (SH3 ΔAUROC(M3−M2) CI excludes 0). **H1 supported** iff M1≈M2≈M3
  and melodrama residual ≈ control.

## Expected Outcomes
Literature (low-D affect manifold + persona-dependent emotion bias) predicts a
**mixed** result: a low-D drama subspace (rank > 1 but small) with reader
variation that is **more than a single threshold** — i.e., partial support for H2
within an overall low-dimensional geometry. We report whatever we find, including
nulls.

## Timeline & Milestones
1. Setup + model download + stimulus generation — 30 min.
2. E1 + E2 (activation extraction + subspace geometry) — 60 min.
3. E3 (reader nested models) — 45 min.
4. E4 + persona — 30 min.
5. Analysis, figures, REPORT.md — 45 min.
(20–30% buffer for debugging GPU/transformers API.)

## Potential Challenges
- transformers 5.x hidden-states API drift → validate on tiny model first.
- GoEmotions "drama" is a proxy (emotion intensity), not literal drama → triangulate
  with the controlled corpus.
- Reader sparsity / imbalance → restrict to raters with ≥N labels; reader-stratified CV.
- API cost/rate limits → cache all completions; cap corpus size.

## Success Criteria
A clear, statistically-supported answer to H1 vs H2 on each sub-hypothesis, with
controls passing, replicated across ≥2 layers and ≥2 model scales, documented in
REPORT.md with figures and honest limitations.
