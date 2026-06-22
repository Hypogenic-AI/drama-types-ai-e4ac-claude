#!/usr/bin/env python3
"""Directly download well-known arXiv papers that S2 resolution may have missed,
plus retry any NO-PDF entries in download_manifest.json via S2."""
import os, re, json, time, sys
import requests

PAPERS = os.path.join(os.path.dirname(__file__), "..", "papers")
HEAD = {"User-Agent": "Mozilla/5.0 (research resource finder)"}

# title-slug -> known arXiv id (verified canonical papers)
KNOWN = {
    "the_linear_representation_hypothesis_and_the_geometry_of_large_languag": "2311.03658",
    "the_geometry_of_truth_emergent_linear_structure_in_large_language_mod": "2310.06824",
    "linear_representations_of_sentiment_in_large_language_models": "2310.15154",
    "the_geometry_of_refusal_in_large_language_models_concept_cones_and_re": "2502.17420",
    "sparse_feature_circuits_discovering_and_editing_interpretable_causal_g": "2403.19647",
    "a_structural_probe_for_finding_syntax_in_word_representations": None,  # ACL anthology, handled below
    "emergent_linear_representations_in_world_models_of_self_supervised_seq": "2309.00941",
    "are_large_language_models_capable_of_generating_human_level_narratives": "2407.13248",
    "modeling_emotional_trajectories_in_written_stories_utilizing_transform": "2406.02251",
}
# direct non-arxiv PDF URLs
DIRECT = {
    "a_structural_probe_for_finding_syntax_in_word_representations":
        "https://aclanthology.org/N19-1419.pdf",
}


def slug(s, n=60):
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()[:n]


def dl(url, path):
    try:
        r = requests.get(url, headers=HEAD, timeout=60, allow_redirects=True)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            open(path, "wb").write(r.content)
            return True
    except Exception as e:
        print("  err", e, file=sys.stderr)
    return False


man = json.load(open(os.path.join(PAPERS, "download_manifest.json")))
by_slug = {slug(m["title"]): m for m in man}

for s, ax in KNOWN.items():
    m = by_slug.get(s)
    fpath = os.path.join(PAPERS, f"{s}.pdf")
    if os.path.exists(fpath) and os.path.getsize(fpath) > 1000:
        continue
    ok = False
    if ax:
        ok = dl(f"https://arxiv.org/pdf/{ax}.pdf", fpath)
    if not ok and s in DIRECT:
        ok = dl(DIRECT[s], fpath)
    print(f"[{'OK' if ok else 'FAIL'}] {s} ({ax})")
    if ok and m:
        m["downloaded"] = True
        m["file"] = f"{s}.pdf"
        m["source"] = f"arxiv:{ax}" if ax else "direct"

json.dump(man, open(os.path.join(PAPERS, "download_manifest.json"), "w"), indent=2)
ok = sum(1 for m in man if m["downloaded"])
print(f"\nManifest now: {ok}/{len(man)} downloaded", file=sys.stderr)
