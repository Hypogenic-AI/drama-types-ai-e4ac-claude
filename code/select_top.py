#!/usr/bin/env python3
"""Pick the focused download set by matching curated titles against a whitelist."""
import json, re

WHITELIST = [
    # --- Representation geometry / subspaces (methodology backbone) ---
    "The Linear Representation Hypothesis and the Geometry",
    "Beyond Single Concept Vector",
    "The Geometry of Truth",
    "Linear Representations of Sentiment",
    "Refusal in Language Models Is Mediated by a Single Direction",
    "The Geometry of Refusal in Large Language Models",
    "From Directions to Regions",
    "The Geometry of Harmfulness",
    "Extracting Latent Steering Vectors",
    "A Structural Probe for Finding Syntax",
    "Emergent Linear Representations in World Models",
    "Semantic Structure of Feature Space",
    "H-Probes: Extracting Hierarchical Structures",
    "Sparse Feature Circuits",
    "Linear Representations of Political Perspective",
    "The Granularity Axis",
    "Tensor Product Representation Probes",
    # --- Affect / emotion latent geometry ---
    "Emotions Where Art Thou",
    "Emergence of Hierarchical Emotion Organization",
    'Do LLMs "Feel"',
    "Do LLMs “Feel”",
    # --- Subjective NLP / reader / personalization ---
    "Large Language Models for Subjective Language Understanding",
    "Beyond Demographics",
    "PALS: Personalized Active Learning",
    "Capturing Human Perspectives in NLP",
    "Multi-Perspective LLM Annotations",
    "Modeling Subjectivity in Cognitive Appraisal",
    "What If Ground Truth Is Subjective",
    "Annotating and Training for Population Subjective Views",
    # --- Narrative drama / reader response ---
    "Are Large Language Models Capable of Generating Human-Level Narratives",
    "Is the Top Still Spinning",
    "Predicting subjective ratings of affect",
    "Tears or Cheers",
    "Modeling Emotional Trajectories in Written Stories",
    "From Generalized Laughter to Personalized Chuckles",
]

cur = json.load(open("paper_search_results/_curated.json"))
def norm(s): return re.sub(r"\s+", " ", s or "").strip().lower()
picked, used = [], set()
for w in WHITELIST:
    wl = norm(w)
    for r in cur:
        t = norm(r.get("title"))
        if wl in t and r.get("title") not in used:
            picked.append(r); used.add(r.get("title")); break

json.dump(picked, open("paper_search_results/_download_set.json", "w"), indent=2)
print(f"picked {len(picked)} papers")
for r in picked:
    print(f"  ({r.get('year')}) {r.get('title')}")
