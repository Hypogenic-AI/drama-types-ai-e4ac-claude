"""
Prepare GoEmotions raw (82 identified raters) for per-reader 'drama'/intensity modeling.

'Drama' is operationalized as perceived emotional INTENSITY (arousal), regardless of
valence. Each GoEmotions emotion is assigned an arousal weight in [0,1] (circumplex /
NRC-VAD style). For each (text, rater) annotation we compute:
  - intensity = max arousal weight over the emotions that rater selected (0 if only neutral/none)
  - drama_label = 1 if the rater selected ANY high-arousal emotion, else 0
Reader disagreement on the SAME text => "different people find different things dramatic".

Outputs (data/):
  ge_annotations.parquet : one row per (text_id, rater_id) with intensity + drama_label
  ge_texts.parquet       : unique texts (text_id, text) to extract activations for
  ge_meta.json           : summary stats
"""
import json, os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
DATA = "datasets/go_emotions/raw_train.parquet"
OUT = "data"
os.makedirs(OUT, exist_ok=True)

# Arousal (intensity) weights per GoEmotions emotion, ~Russell circumplex / NRC-VAD.
# High arousal => high "drama" potential. Neutral excluded (0).
AROUSAL = {
    "admiration": 0.45, "amusement": 0.55, "anger": 0.95, "annoyance": 0.55,
    "approval": 0.30, "caring": 0.45, "confusion": 0.50, "curiosity": 0.50,
    "desire": 0.75, "disappointment": 0.50, "disapproval": 0.45, "disgust": 0.80,
    "embarrassment": 0.65, "excitement": 0.90, "fear": 0.95, "gratitude": 0.40,
    "grief": 0.85, "joy": 0.70, "love": 0.65, "nervousness": 0.80,
    "optimism": 0.50, "pride": 0.60, "realization": 0.40, "relief": 0.40,
    "remorse": 0.60, "sadness": 0.55, "surprise": 0.80,
}
EMOTIONS = list(AROUSAL.keys())
# "High-arousal / high-drama" emotions for the binary label (arousal >= 0.70)
HIGH_DRAMA = [e for e, a in AROUSAL.items() if a >= 0.70]
print("High-drama emotions:", HIGH_DRAMA)

df = pd.read_parquet(DATA)
print("raw shape", df.shape)
# drop very-unclear examples
df = df[~df["example_very_unclear"].astype(bool)].copy()

# per-annotation intensity + label
emo_mat = df[EMOTIONS].to_numpy(dtype=float)            # (N, 27) 0/1
arousal_vec = np.array([AROUSAL[e] for e in EMOTIONS])  # (27,)
weighted = emo_mat * arousal_vec[None, :]               # (N,27)
df["intensity"] = weighted.max(axis=1)                  # max arousal among selected
hd_idx = [EMOTIONS.index(e) for e in HIGH_DRAMA]
df["drama_label"] = (emo_mat[:, hd_idx].sum(axis=1) > 0).astype(int)
df["n_emotions"] = emo_mat.sum(axis=1).astype(int)

ann = df[["id", "text", "rater_id", "intensity", "drama_label", "n_emotions"]].rename(
    columns={"id": "text_id"})

# Keep texts annotated by >=3 raters and raters with >=150 annotations (stable per-reader fit)
tcounts = ann.groupby("text_id")["rater_id"].nunique()
keep_texts = tcounts[tcounts >= 3].index
ann = ann[ann["text_id"].isin(keep_texts)]
rcounts = ann["rater_id"].value_counts()
keep_raters = rcounts[rcounts >= 150].index
ann = ann[ann["rater_id"].isin(keep_raters)].copy()

# Recompute text keep after rater filter (still >=3 raters)
tcounts2 = ann.groupby("text_id")["rater_id"].nunique()
keep_texts2 = tcounts2[tcounts2 >= 3].index
ann = ann[ann["text_id"].isin(keep_texts2)].copy()

texts = ann[["text_id", "text"]].drop_duplicates("text_id").reset_index(drop=True)

# Disagreement diagnostics: per-text std of drama_label across raters
disagree = ann.groupby("text_id")["drama_label"].agg(["mean", "std", "count"])
frac_contested = ((disagree["mean"] > 0.15) & (disagree["mean"] < 0.85)).mean()

meta = {
    "n_annotations": int(len(ann)),
    "n_texts": int(texts.shape[0]),
    "n_raters": int(ann["rater_id"].nunique()),
    "drama_base_rate": float(ann["drama_label"].mean()),
    "mean_intensity": float(ann["intensity"].mean()),
    "frac_texts_contested_0.15_0.85": float(frac_contested),
    "high_drama_emotions": HIGH_DRAMA,
    "arousal_weights": AROUSAL,
    "annotations_per_rater": ann["rater_id"].value_counts().describe().to_dict(),
}
ann.to_parquet(f"{OUT}/ge_annotations.parquet")
texts.to_parquet(f"{OUT}/ge_texts.parquet")
with open(f"{OUT}/ge_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2, default=str))
print(f"\nSaved {len(ann)} annotations, {len(texts)} texts, {meta['n_raters']} raters.")
