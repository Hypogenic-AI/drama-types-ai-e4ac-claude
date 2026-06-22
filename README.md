# Different Kinds of Drama

How do LLMs represent **what different people find dramatic**? Is the
dramatic-vs-melodramatic distinction a **threshold on a single axis** (H1), or a
**reader-specific direction in a drama subspace** (H2)? We test this *inside* the
activations of `Qwen2.5-7B/1.5B-Instruct`, using a controlled LLM-authored
drama corpus and **70 identified human raters** from GoEmotions.

## Key findings

- **Drama is multi-dimensional, not one axis.** Melodrama is **not "just more
  drama"**: the dramatic→melodramatic step becomes **~80–86° (nearly orthogonal)**
  to the drama axis in deep layers; melodrama separates on a distinct **"excess"
  direction**. (`figures/drama_excess_plane.png`)
- **The drama subspace is low-rank but >1-D:** participation ratio ~5.6–5.9 across
  8 subconcepts vs ~8.0 for random directions — a genuine *low-dimensional space*
  of drama features. (`figures/e1_e2_geometry.png`)
- **Reader variation is a THRESHOLD, not a direction (H1).** A per-reader threshold
  on one shared drama axis drives held-out AUROC 0.66→0.70; a per-reader *direction*
  in a 30-D drama subspace adds **exactly the shuffled-reader null** (z≈1, n.s.).
  Replicated across layers, K, both model scales, and a transferred subspace.
  (`figures/e3b_indomain.png`)
- **Prompted reader personas don't rotate the drama axis** (pairwise cosine ≥0.97).
  (`figures/e4_persona.png`)
- **Bottom line:** "different *kinds* of drama" are shared content directions;
  "different *people*" mostly slide a threshold along them.

See **[REPORT.md](REPORT.md)** for the full study and **[planning.md](planning.md)**
for the pre-registered design.

## Reproduce

```bash
source .venv/bin/activate                 # uv-managed env (torch 2.12+cu130, transformers 5.12)
export HF_HOME=$PWD/.hf_cache             # bf16 Qwen2.5 weights (fp16 → NaNs)

python src/go_emotions_prep.py            # build per-reader GoEmotions drama labels (70 raters)
# (controlled drama corpus already built by the 28-agent workflow → data/corpus_*.parquet)

python src/e1_e2_geometry.py  --model Qwen/Qwen2.5-7B-Instruct  --tag qwen7b    # E1/E2 geometry
python src/extract_ge.py      --model Qwen/Qwen2.5-7B-Instruct  --tag qwen7b --layers 16 21 28
python src/e3_reader.py       --model Qwen/Qwen2.5-7B-Instruct  --ge_tag qwen7b --layer 21  # E3 (transferred)
python src/e3b_indomain.py    --ge_tag qwen7b --layer 21 --K 30                 # E3b (in-domain, the crux)
python src/e4_persona.py      --model Qwen/Qwen2.5-7B-Instruct  --layer 21      # E4 persona
python src/make_figures.py                                                      # all figures
```

GPU: 1× RTX A6000 (bf16). Full pipeline ≈ 30–40 min. Seeds = 42.

## Structure
```
planning.md            # Phase 0/1: motivation, hypotheses, pre-registered design
REPORT.md              # full report with results, figures, limitations
src/                   # extraction + 4 experiments + figures
data/                  # corpus_*.parquet (stimuli), ge_*.parquet (labels), ge_acts_*.npz (activations)
results/               # per-experiment JSON + summary_all.json
figures/               # 5 figures
literature_review.md   # pre-gathered (33 papers); resources.md; papers/ datasets/
```
