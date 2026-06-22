# Datasets

Data files are **not** committed to git (see `.gitignore`). Follow the download
instructions below to reproduce. Small sample files (`samples.json`) *are*
committed for reference.

These datasets were chosen because the research question — *do LLMs represent
"what different people find dramatic" via drama-feature dimensions or via
reader-feature dimensions containing drama subspaces?* — requires text with
**per-reader / per-annotator affective judgments**, so that we can separate
*item* (text) structure from *reader* structure in activation space.

---

## Dataset 1: EmoBank (primary)

The single best public fit for the hypothesis: every sentence is rated on
**Valence–Arousal–Dominance (VAD)** from **two distinct perspectives** — the
*writer's* intended emotion and the *reader's* perceived emotion — and the
per-individual reader ratings are released, so multiple readers rate the same
text.

### Overview
- **Source**: https://github.com/JULIELab/EmoBank (Buechel & Hahn, EACL 2017)
- **Size**: 10,062 sentences (combined); 53,055 individual *reader* ratings over
  10,548 text spans (~5 readers/text); matching individual *writer* ratings.
- **Format**: CSV (UTF-8).
- **Dimensions**: V, A, D on a 1–5 scale (continuous, averaged in `emobank.csv`).
- **Genres** (from `meta.tsv`): blog, essays, fiction, letters, news, travel
  guides — a genre spread useful for "dramatic vs melodramatic" contrasts.
- **License**: CC-BY-SA 4.0.

### Files downloaded (in `datasets/emobank/`)
| File | Rows | What |
|------|------|------|
| `emobank.csv` | 10,062 | combined, bi-perspective averaged VAD + text + split |
| `reader.csv` | ~10k | reader-perspective aggregated VAD |
| `writer.csv` | ~10k | writer-perspective aggregated VAD |
| `individual_reader_ratings.csv` | 53,055 | **per-reader** VAD (key for reader-subspace analysis) |
| `individual_writer_ratings.csv` | ~53k | per-writer-annotator VAD |
| `raw.csv` | ~10k | raw merged annotations |
| `meta.tsv` | 10,062 | document id, genre category/subcategory |

### Download instructions
```bash
mkdir -p datasets/emobank && cd datasets/emobank
base="https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus"
for f in emobank.csv reader.csv writer.csv individual_reader_ratings.csv \
         individual_writer_ratings.csv raw.csv meta.tsv; do
  curl -sL "$base/$f" -o "$f"
done
```

### Loading
```python
import pandas as pd
eb   = pd.read_csv("datasets/emobank/emobank.csv")        # id, split, V, A, D, text
ind  = pd.read_csv("datasets/emobank/individual_reader_ratings.csv")  # id, V, A, D (one row per reader)
meta = pd.read_csv("datasets/emobank/meta.tsv", sep="\t") # id, document, category, subcategory
```

### Notes
- `individual_reader_ratings.csv` rows share `id` with `emobank.csv`; group by
  `id` to recover the multi-reader rating distribution per text. The **spread**
  across readers (esp. on Arousal) is a natural operationalization of
  "one reader's drama is another's melodrama."
- Arousal ≈ intensity/drama; Valence ≈ positivity. A "melodrama" signature is
  plausibly high-arousal + stereotyped/low-novelty — testable against text
  features.

---

## Dataset 2: GoEmotions (raw, multi-annotator)

Large-scale corpus where each of 58k Reddit comments is labeled by multiple
raters (with `rater_id`), over 27 emotions + neutral. Provides real
**inter-reader disagreement** at scale for modeling reader-conditioned affect.

### Overview
- **Source**: https://huggingface.co/datasets/google-research-datasets/go_emotions
  (Demszky et al., ACL 2020).
- **Size (raw config)**: 211,225 annotations, **58,011 unique texts**, **82 raters**.
- **Size (simplified)**: 43,410 train / 5,426 val / 5,427 test (aggregated labels).
- **Format**: Parquet.
- **Labels**: 28 columns (27 emotions + `neutral`) as 0/1 per annotation, plus
  `rater_id`, `author`, `subreddit`.
- **License**: Apache-2.0.

### Files downloaded (in `datasets/go_emotions/`)
| File | What |
|------|------|
| `raw_train.parquet` | 211k per-rater annotations (has `rater_id`) — **key file** |
| `simplified_train/validation/test.parquet` | aggregated multi-label splits |

### Download instructions
```bash
mkdir -p datasets/go_emotions && cd datasets/go_emotions
base="https://huggingface.co/datasets/google-research-datasets/go_emotions/resolve/refs%2Fconvert%2Fparquet"
curl -sL "$base/raw/train/0000.parquet"        -o raw_train.parquet
curl -sL "$base/simplified/train/0000.parquet" -o simplified_train.parquet
curl -sL "$base/simplified/test/0000.parquet"  -o simplified_test.parquet
curl -sL "$base/simplified/validation/0000.parquet" -o simplified_validation.parquet
```
(Or `load_dataset("google-research-datasets/go_emotions", "raw")` with the
HuggingFace `datasets` library.)

### Loading
```python
import pandas as pd
raw = pd.read_parquet("datasets/go_emotions/raw_train.parquet")
# multi-reader emotion judgments for the same text:
by_text = raw.groupby("id")  # disagreement across rater_id within each text
```

### Notes
- Map the 27 emotions to an intensity/arousal proxy (e.g., grief/excitement/fear
  high-arousal) to study "drama intensity" disagreement across readers.
- `rater_id` enables learning a per-reader embedding — the substrate for testing
  whether reader differences are a single threshold shift vs a reader subspace.

---

## Optional / extension datasets (not downloaded; documented for the runner)

- **WASSA / "Modeling Emotional Trajectories in Written Stories"** (Christ et al.
  2024): children's stories with continuous V/A trajectories — for *narrative*
  drama arcs. https://github.com/lc0197/emotional_trajectories_stories
- **CEDAR** (Tears or Cheers?, 2026): 10,962 instances, 7 languages, culturally
  *distinct* affective responses — directly about reader-group variance in
  affect. Check the paper for release.
- **LeWiDi (Learning With Disagreements)** shared-task datasets (MD-Agreement,
  HS-Brexit, ConvAbuse, ArMIS): per-annotator subjective labels; a standard
  benchmark for the "disagreement is signal" framing.
- **Emotions in Drama** (Dennerlein et al. 2023): German plays (18th–19th c.)
  annotated with emotion categories — an actual *drama* corpus, used as one of
  the eight eval sets in *Emotions Where Art Thou*. Good for a literal
  drama-genre probe set.
- **SEV (Scenario-Event-Valence)** (Wang et al. 2025, *Do LLMs "Feel"?*):
  controlled scenarios eliciting comparable internal states across emotions —
  a template for building a controlled drama/melodrama stimulus set.

See `../resources.md` and `../literature_review.md` for how these map to the
experimental design.
