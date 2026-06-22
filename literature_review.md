# Literature Review — *Different Kinds of Drama*

**Research question.** Large language models (LLMs) must represent what
*different people* find dramatic. Is the distinction between "dramatic" (to me)
and "melodramatic" (to you) best modeled as **thresholds on a single axis**, or
is it richer? Two competing geometric hypotheses:

- **(H1) Drama-feature space.** A low-dimensional space of *drama features*
  (intensity, stakes, sincerity-vs-excess, novelty…) in which different readers
  occupy different regions / apply different thresholds.
- **(H2) Reader-feature space with drama subspaces.** A low-dimensional space of
  *reader features* (taste, sensibility, demographic/personality priors) that
  *contains* drama-relevant subspaces — i.e., drama is read off a
  reader-conditioned subspace rather than a single shared axis.

The literature does not address "drama" per se, but three mature lines of work
supply (a) the **geometric vocabulary and tools** to formalize H1 vs H2, (b)
direct evidence about how **affect/emotion** is encoded in LLMs, and (c) the
**subjectivity / per-reader modeling** needed to make "different people" precise.

---

## 1. Research-area overview

**Linear representation hypothesis (LRH) and its limits.** A dominant finding is
that high-level concepts are encoded as *linear directions* in activation space,
recoverable by probes and manipulable by steering. But a growing wave of 2024–26
papers argues that a *single direction* is too coarse: concepts live in
**subspaces, cones, regions, or hierarchies**. This single-direction-vs-subspace
debate is *exactly* the H1/H2 question transposed onto representation geometry.
Whether "drama" is a threshold on one axis (single direction) or a
reader-conditioned subspace (multi-dimensional cone/region) is the geometric
form of the research hypothesis.

**Affect in LLMs.** Several recent papers locate emotion in a *low-dimensional,
steerable manifold* that generalizes across datasets and languages — strong prior
evidence that "drama-as-affect-intensity" would have low-dimensional structure
(supports H1's "low-dimensional" premise), while leaving open whether reader
variation is a threshold or a subspace.

**Subjective NLP.** A parallel community treats annotator disagreement as
*signal, not noise*, building **per-reader / per-perspective** models. This line
provides the reader-feature machinery for H2 and the datasets with per-reader
labels needed to test it.

---

## 2. Key papers

### A. Representation geometry — single direction vs. subspace (methodological core)

#### Park, Choe, Veitch (2023) — *The Linear Representation Hypothesis and the Geometry of LLMs*
- **Contribution**: Formalizes "linear representation" via counterfactuals;
  derives a (non-Euclidean) *causal inner product* unifying probing and steering.
- **Relevance**: The right metric for measuring drama directions; warns that
  cosine/projection are only meaningful under the correct inner product — a
  subtlety that matters when testing "single axis."

#### Tigges, Hollinsworth, Geiger, Nanda (2023) — *Linear Representations of Sentiment in LLMs*
- **Contribution**: Sentiment is *one direction* (positive↔negative), causally
  validated by ablation; introduces the "summarization motif" (affect summarized
  at punctuation/name tokens).
- **Relevance**: The strongest case that an affective construct can be a *single
  axis* — a baseline/foil for H1. If "drama" were like sentiment, a single
  direction with reader-specific thresholds (H1) would suffice.

#### Marks & Tegmark (2023) — *The Geometry of Truth*
- **Contribution**: Truth/falsehood is linearly represented; difference-in-means
  probes are as good as fancier probes and more causal; transfer across datasets.
- **Relevance**: Template methodology (visualize → probe-transfer → causal
  intervention) directly portable to drama; difference-in-means as a robust
  single-direction baseline.

#### Arditi et al. (2024) — *Refusal Is Mediated by a Single Direction*
- **Contribution**: Across 13 chat models, refusal is a *1-D subspace*; ablating
  it removes refusal, adding it induces refusal.
- **Relevance**: Canonical "it's just one axis with a threshold" result — the H1
  archetype. Methodologically shows how to *test* single-direction sufficiency
  (directional ablation).

#### Wollschläger et al. (2025) — *The Geometry of Refusal: Concept Cones and Representational Independence*
- **Contribution**: Refutes the single-direction view for refusal — finds
  **multiple independent directions and multi-dimensional concept cones**;
  introduces *representational independence* (orthogonality ≠ independence under
  intervention).
- **Relevance**: The H2 archetype and the **central methodological model for this
  project**: the exact toolkit for asking "is drama one axis or a cone/subspace?"

#### Zhao et al. (2024) — *Beyond Single Concept Vector: Gaussian Concept Subspace (GCS)*
- **Contribution**: Replaces a single concept vector with a *Gaussian subspace*;
  more faithful/robust; used for **emotion steering**.
- **Relevance**: Concrete method to represent "drama" as a distribution/subspace
  and compare it against a single-vector probe (H1 vs H2 test) — and it already
  targets emotion.

#### Shafran et al. (2026) — *From Directions to Regions: Decomposing Activations via Local Geometry*
- **Contribution**: Mixture of Factor Analyzers (MFA) models activation space as
  Gaussian **regions** with local covariance; captures nonlinear/multi-D
  structure; beats SAEs on steering.
- **Relevance**: If drama is reader-conditioned (H2), region/local-covariance
  models should outperform global directions — a clean discriminating test.

#### Shah et al. (2025) — *The Geometry of Harmfulness through Subconcept Probing*
- **Contribution**: 55 harmfulness *subconcepts* → 55 probes spanning a
  **strikingly low-rank harmfulness subspace**; dominant-direction steering
  nearly removes harm with low utility loss.
- **Relevance**: Direct template: build many drama *subconcepts* (suspense,
  grief, spectacle, sincerity, excess), probe each, and measure the **rank** of
  the resulting drama subspace — the quantitative crux of "low-dimensional."

#### Subramani, Suresh, Peters (2022) — *Extracting Latent Steering Vectors*
- **Contribution**: Steering vectors extractable from frozen LMs reproduce target
  sentences; vector arithmetic does unsupervised sentiment transfer.
- **Relevance**: Baseline technique for obtaining drama steering vectors without
  fine-tuning; distances between vectors as a similarity geometry.

#### Hewitt & Manning (2019) — *A Structural Probe for Finding Syntax*
- **Contribution**: A *linear transformation* under which L2 distance encodes
  tree distance — probing for **structure**, not a single label direction.
- **Relevance**: Methodological ancestor for probing *relational/structured*
  drama representations (e.g., a reader-by-drama bilinear structure).

#### Nanda, Lee, Wattenberg (2023) — *Emergent Linear Representations in World Models*
- **Relevance**: "My vs opponent" reframing makes a nonlinear world-model
  linear — a caution that *the right basis/contrast* can turn apparent nonlinear
  reader effects into linear ones (relevant to picking reader contrasts).

#### Lee, Viégas, Wattenberg (2026) — *Tensor Product Representation Probes Reveal Shared Structure*
- **Contribution**: A "bag of linear directions" misses relational structure;
  TPR probes recover a factorization (e.g., square ⊗ color) underlying many
  linear directions.
- **Relevance**: Suggests drama directions across readers might **factorize** as
  reader ⊗ drama-primitive — a concrete formalization of H2.

#### Kozlowski & Boutyline (2026) — *Semantic Structure of Feature Space in LLMs*
- **Contribution**: 32 semantic axes (beautiful–ugly, soft–hard…) over 360 words
  correlate with human ratings; **much variance lies on a low-D subspace**;
  steering one axis spills over to correlated axes.
- **Relevance**: Best evidence that human-like *evaluative* axes (drama is
  evaluative) form a low-D, correlated subspace — supports the "low-dimensional"
  premise and predicts cross-axis spillover for drama features.

#### Dawes et al. (2026) — *H-Probes: Extracting Hierarchical Structures*
- **Relevance**: If drama↔melodrama is *hierarchical* (melodrama as a child/excess
  variant of drama), H-probes test for low-D hierarchy-bearing subspaces.

#### Kim, Evans, Schein (2025) — *Linear Representations of Political Perspective Emerge in LLMs*
- **Contribution**: Probes predict lawmakers' DW-NOMINATE ideology from attention
  heads; same probes predict news-outlet slant — a **perspective/reader** axis.
- **Relevance**: Direct precedent for H2: a *reader/viewpoint* feature is linearly
  represented and transfers — the template for a "reader axis" that could host a
  drama subspace.

#### Qin et al. (2026) — *The Granularity Axis: Micro-to-Macro Latent Direction for Social Roles*
- **Contribution**: A single contrast axis captures role granularity (PC1, 52.6%
  variance), stable across layers, causally steerable.
- **Relevance**: Shows a *persona/role* property can be one dominant axis —
  evidence a reader-sensibility axis (H2 substrate) may be low-dimensional.

#### Marks et al. (2024) — *Sparse Feature Circuits*
- **Relevance**: SAE-feature circuits as an alternative, fine-grained unit for
  locating drama features and the reader-conditioning that gates them.

### B. Affect / emotion latent geometry in LLMs

#### Reichman, Avsian, Heck (2025) — *Emotions Where Art Thou: The Emotional Latent Space of LLMs*
- **Contribution**: Identifies a **low-dimensional emotional manifold**;
  directionally encoded, distributed across layers, generalizes to 8 datasets /
  5 languages; a "universal emotional subspace"; steerable while preserving
  semantics.
- **Relevance**: **Cornerstone prior.** Establishes that affect (the substrate of
  drama) is low-dimensional and steerable — supports H1's premise and supplies
  the manifold into which drama/reader structure must be located.
- **Method detail (from deep read of full paper, ICLR 2026)**: (1) Extract
  *clean, maximally polarized* affect directions from a **synthetic** corpus
  (neutral sentences rewritten into each emotion), then evaluate **only** on
  human-written data — a transfer protocol directly reusable for "drama"
  (rewrite neutral text into dramatic/melodramatic variants → extract directions
  → test on real reader-rated text). (2) Tests cross-dataset *geometric
  equivalence* with **stress (Stress-1/2, Sammon)** and **distortion (average,
  ℓ2, σ)** metrics on same-emotion pairwise distances — the exact toolkit to ask
  whether two *readers'* drama subspaces are the same space up to rotation/scale
  (global rescale) vs anisotropically different (genuinely different geometry).
  (3) A **~50-D synthetic emotional subspace** retains linear decodability —
  concrete evidence for "low-dimensional." (4) **ML-AURA** scores neurons as
  one-vs-all emotion detectors (AUROC>0.9 ⇒ "expert" units) — usable to localize
  drama-selective units. (5) One of their 8 eval sets is literally **"Emotions in
  Drama"** (Dennerlein et al. 2023, German plays) — an off-the-shelf drama corpus.

#### Wang et al. (2025) — *Do LLMs "Feel"? Emotion Circuits Discovery and Control*
- **Contribution**: Builds SEV (Scenario-Event-Valence) dataset; extracts
  *context-agnostic emotion directions*; identifies neurons/heads → global
  **emotion circuits**; 99.65% emotion-control accuracy.
- **Relevance**: Mechanistic template; the "context-agnostic vs context-dependent"
  split parallels "shared drama axis vs reader-dependent drama."

#### Zhao et al. (2025) — *Emergence of Hierarchical Emotion Organization in LLMs*
- **Contribution**: LLMs form **hierarchical emotion trees** (emotion-wheel-like);
  larger models → deeper hierarchies; **systematic biases across socioeconomic
  personas**.
- **Relevance**: Persona-conditioned emotion bias is direct evidence that affect
  perception is *reader/persona-dependent* (H2), and that the dependence is
  structured (hierarchy), not a flat threshold.

### C. Subjectivity, disagreement, and per-reader modeling

#### Song et al. (2025) — *LLMs for Subjective Language Understanding: A Survey*
- **Relevance**: Frames the whole subjective-NLP landscape (sentiment, emotion,
  humor, aesthetics) and its datasets — the field map for "different people,
  different judgments," and where drama/aesthetics sit.

#### Orlikowski et al. (2025) — *Beyond Demographics: Fine-tuning LLMs to Predict Individuals' Subjective Perceptions*
- **Contribution**: Training improves per-annotator prediction, but gains come
  from learning **annotator-specific behavior**, *not* sociodemographic patterns.
- **Relevance**: Cautionary, pivotal for H2: reader structure may be *idiosyncratic
  per-reader* rather than low-D demographic — informs how to parameterize the
  reader space (learned reader embeddings, not demographic features).

#### Kanclerz et al. (2023) — *PALS: Personalized Active Learning for Subjective Tasks*
#### Kanclerz et al. (2022) — *What If Ground Truth Is Subjective? Personalized Hate-Speech Detection*
#### Bielaniewicz & Kazienko (2023) — *From Generalized Laughter to Personalized Chuckles*
- **Contribution (this group)**: Per-reader ("personalized") models substantially
  beat majority-vote models on subjective tasks (humor +~35% F1); personalization
  matters more than architecture.
- **Relevance**: Empirical backbone for H2 — reader identity carries large,
  learnable variance. "Personalized chuckles" is the humor analogue of
  "different kinds of drama."

#### Mehrotra, Visokay, Gligorić (2026) — *Multi-Perspective LLM Annotations for Subjective Tasks*
- **Relevance**: Treats the *distribution over groups* as the target; adaptive
  sampling where LLM proxies are weakest — a method to estimate reader-conditioned
  drama distributions cheaply.

#### Zhou et al. (2025) — *Modeling Subjectivity in Cognitive Appraisal with LMs*
- **Relevance**: Personality + demographics improve subjectivity modeling;
  post-hoc calibration underperforms — guidance on reader features and pitfalls.

#### Mieleszczenko-Kowszewicz et al. (2023) — *Capturing Human Perspectives in NLP*
#### Alexeeva et al. (2023) — *Annotating and Training for Population Subjective Views*
- **Relevance**: Annotation design / questionnaire-grounded reader traits — useful
  if we elicit reader covariates for the reader-feature space.

### D. Narrative drama and reader response (the "drama" end)

#### Tian et al. (2024) — *Are LLMs Capable of Generating Human-Level Narratives?*
- **Contribution**: Analyzes story **arcs, turning points, arousal/valence**;
  human stories are suspenseful/arousing, LLM stories are homogeneously positive
  and low-tension; injecting discourse features improves suspense/arousal +40%.
- **Relevance**: Operationalizes *drama* (suspense, tension, arousal) at discourse
  level and gives features to correlate with internal representations.

#### Christ et al. (2024) — *Modeling Emotional Trajectories in Written Stories*
- **Contribution**: Continuous V/A **trajectories** over children's stories;
  DeBERTa + weak supervision (CCC .82 valence / .71 arousal); variance by author/
  story/section.
- **Relevance**: A drama-arc dataset+model; arousal trajectory ≈ "drama curve."

#### Subbiah et al. (2025) — *Is the Top Still Spinning? Evaluating Subjectivity in Narrative Understanding*
- **Contribution**: Narrative faithfulness is often *ambiguous*; reframes via an
  **Ambiguity Rewrite Metric**; reduces forced-binary disagreement.
- **Relevance**: Methodology for handling reader-dependent ambiguity in narrative
  — relevant to "one reader's drama is another's melodrama."

#### Dai et al. (2026) — *Tears or Cheers? Benchmarking LLMs via Culturally Elicited Distinct Affective Responses (CEDAR)*
- **Contribution**: 10,962 instances, 7 languages, 14 emotions, built to isolate
  **cross-cultural affective divergence**; models show language≠culture alignment.
- **Relevance**: A ready-made benchmark where the *same* stimulus yields
  *different* affect by reader-group — an external test bed for H1 vs H2.

---

## 3. Common methodologies (synthesis)

| Method | Papers | Use for drama |
|---|---|---|
| Linear probes (logistic / diff-in-means) | Marks&Tegmark; Tigges; Kim | Find a candidate single "drama axis" |
| Difference-in-means direction + ablation | Marks&Tegmark; Arditi | Test single-direction *sufficiency* (H1) |
| Concept **subspace** (Gaussian / PCA / cone) | Zhao GCS; Wollschläger; Shah | Test multi-D drama structure (H2) |
| **Region / local-covariance** (MFA) | Shafran 2026 | Reader-conditioned, nonlinear drama |
| Steering vectors / activation addition | Subramani; GCS; Emotions-Where-Art-Thou | Causal validation of drama directions |
| SAE features & feature circuits | Marks SFC; Wang emotion circuits | Fine-grained drama/reader features |
| Structural / TPR / H-probes | Hewitt&Manning; Lee TPR; Dawes | Reader⊗drama factorization, hierarchy |
| Per-reader (personalized) modeling | Kanclerz; Bielaniewicz; Orlikowski | Parameterize the reader space |

**Causal inner product (Park 2023)** should be used wherever cosine/projection
geometry is invoked.

## 4. Standard baselines

- **Single-direction probe** (diff-in-means) on a drama/intensity label — the H1
  baseline; its *ablation* tests sufficiency.
- **Majority-vote / aggregated-label** model — ignores reader identity; the foil
  personalization must beat (per subjective-NLP results).
- **Sociodemographic prompting/conditioning** — known-weak baseline
  (Orlikowski 2025) for reader modeling.
- **PCA / SAE of activations** — unsupervised structure baseline for estimating
  subspace rank.
- **Random-direction / shuffled-reader controls** — to show effects aren't trivial.

## 5. Evaluation metrics

- **Probe accuracy / AUROC / CCC** for drama or VAD prediction (CCC used by
  Christ 2024).
- **Subspace rank / explained variance / participation ratio** — quantifies "low
  dimensional."
- **Ablation effect size** (Δ behavior when a direction/subspace is removed) —
  sufficiency/necessity (Arditi; Shah).
- **Steering success while preserving fluency/semantics** (GCS; Emotions paper).
- **Cross-dataset / cross-reader transfer** of probes (Marks&Tegmark; Kim) —
  universality vs reader-specificity.
- **Per-reader gain over majority vote** (personalization F1/MAE) — H2 evidence.
- **Representational independence under intervention** (Wollschläger) — whether
  multiple drama directions are genuinely distinct.

## 6. Datasets in the literature

- **EmoBank** — bi-perspective (reader/writer) VAD, per-individual ratings.
  *Primary dataset here.*
- **GoEmotions (raw)** — 58k texts, 82 raters, per-rater labels. *Secondary.*
- **SEV** (Wang 2025), **CEDAR** (Dai 2026), **WASSA stories** (Christ 2024),
  **Stanford Sentiment Treebank** (Tigges), **LeWiDi** suite, personalized
  hate/humor sets (Kanclerz/Bielaniewicz) — extensions / external validation.

## 7. Gaps and opportunities

1. **No work studies "drama" directly**, nor the dramatic-vs-melodramatic
   contrast — a genuine niche.
2. **Single-axis vs subspace is unsettled even for well-studied concepts**
   (refusal: Arditi=1-D vs Wollschläger=cone). Drama is a fresh, evaluative,
   reader-dependent test case to adjudicate H1 vs H2.
3. **Reader structure is under-characterized**: Orlikowski shows it's learnable
   but *not* low-D demographic — yet whether a learned reader space *factorizes*
   with drama (TPR-style, H2) is untested.
4. **Affect manifolds are studied model-internally; reader variation is studied
   behaviorally** — the two have not been joined. This project's contribution is
   to test reader-conditioned drama geometry *inside* the model.

## 8. Recommendations for our experiment

- **Datasets**: EmoBank (per-reader VAD; Arousal≈drama intensity) as primary;
  GoEmotions-raw (82 raters) for scale; optionally a small curated
  drama↔melodrama stimulus set with multi-reader labels.
- **Design**:
  1. Probe a **single "drama/intensity" axis** (diff-in-means); test sufficiency
     via ablation/steering (H1 baseline).
  2. Build a **drama subspace** from many subconcepts (GCS / harmfulness-style)
     and measure its **rank**; test whether it beats the single axis.
  3. Add **reader conditioning**: learn per-reader embeddings (not demographics);
     test (a) reader-feature space with drama subspaces (project drama out of a
     reader space — H2) vs (b) reader-as-threshold-shift on one drama axis (H1);
     use **TPR / region (MFA) probes** to check reader⊗drama factorization.
  4. **Causal** checks: steer along candidate axes/subspaces and along reader
     directions; verify with held-out readers and cross-dataset transfer.
- **Baselines**: single-direction probe, majority-vote, sociodemographic
  conditioning, PCA/SAE, shuffled-reader control.
- **Metrics**: probe AUROC/CCC, subspace rank/participation ratio, ablation &
  steering effect sizes, per-reader gain, representational-independence tests,
  cross-reader/-dataset transfer.
- **Methodological care**: use Park's causal inner product for geometry; treat
  disagreement as signal; prefer difference-in-means and ablation for causal
  claims (robust per Marks&Tegmark).
