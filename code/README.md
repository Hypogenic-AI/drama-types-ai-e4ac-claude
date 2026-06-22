# Code

No external baseline repositories were specified in the research topic
(`code_references` was empty). Rather than clone large interpretability
frameworks wholesale, this directory holds the **resource-gathering scripts**
written for this project and documents the key external repos the experiment
runner should pull selectively.

## Scripts in this directory

| Script | Purpose |
|--------|---------|
| `download_papers.py` | Resolve paper-finder results (arXiv URLs / Semantic Scholar CorpusID / S2 paper-hash) to open-access PDFs and download into `papers/`. Writes `papers/download_manifest.json`. Handles both JSONL and JSON-array inputs. |
| `curate.py` | Merge the three `paper_search_results/*.jsonl` files, dedupe by title, and select papers relevant to the drama hypothesis (writes `_curated.json`). |
| `select_top.py` | Whitelist-match the curated set down to the focused ~33-paper download set (`_download_set.json`). |
| `retry_known.py` | Directly fetch canonical arXiv/ACL papers that S2 rate-limiting missed. |

Run order (already executed during resource gathering):
```bash
source ../.venv/bin/activate
python curate.py          # -> paper_search_results/_curated.json
python select_top.py      # -> paper_search_results/_download_set.json
python download_papers.py ../paper_search_results/_download_set.json
python retry_known.py
```

## Recommended external repositories for the experiment runner

These are **not cloned** (keep the repo small); pull the specific module needed.

| Repo | URL | Why |
|------|-----|-----|
| TransformerLens | https://github.com/TransformerLensOrg/TransformerLens | Activation caching / hooking for probing & steering (used by Nanda-group papers). |
| repate / representation-engineering | https://github.com/andyzoujm/representation-engineering | Reading-vector / steering toolkit (RepE). |
| EmoBank | https://github.com/JULIELab/EmoBank | Primary dataset (already downloaded to `datasets/emobank/`). |
| Sparse Feature Circuits | https://github.com/saprmarks/feature_circuits | SAE feature circuits (Marks et al. 2024). |
| Gaussian Concept Subspace | (see *Beyond Single Concept Vector*, arXiv 2410.00153) | Subspace-vs-single-vector probing — central to H1 vs H2. |
| Refusal-direction | https://github.com/andyrdt/refusal_direction | Single-direction ablation template (Arditi 2024). |
| emotional_trajectories_stories | https://github.com/lc0197/emotional_trajectories_stories | Narrative V/A trajectory model & data (Christ 2024). |

## Suggested experiment dependencies (install when running)
```bash
uv add torch transformers accelerate scikit-learn numpy pandas matplotlib
uv add transformer_lens   # optional, for hooked activation extraction
```

See `../resources.md` for the full resource catalog and `../literature_review.md`
for methodology grounding.
