#!/usr/bin/env python3
"""Combine the three search result files, dedupe, and select a curated set
of papers most relevant to the 'Different Kinds of Drama' hypothesis."""
import json, glob, re, sys

DRAMA_KEYS = re.compile(
    r"subjective|reader|narrative|emotion|affect|drama|melodrama|story|"
    r"sentiment|appraisal|aesthetic|poetry|feel|circuit", re.I)

files = glob.glob("paper_search_results/*.jsonl")
rows = []
for f in files:
    for line in open(f):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

# dedupe by normalized title
seen = {}
for r in rows:
    t = re.sub(r"[^a-z0-9]+", "", (r.get("title") or "").lower())
    if not t:
        continue
    # keep the record with the highest relevance
    if t not in seen or (r.get("relevance") or 0) > (seen[t].get("relevance") or 0):
        seen[t] = r
uniq = list(seen.values())

selected = []
for r in uniq:
    rel = r.get("relevance") or 0
    title = r.get("title") or ""
    abs = r.get("abstract") or ""
    keep = False
    if rel >= 3:
        keep = True
    elif rel >= 2 and DRAMA_KEYS.search(title + " " + abs):
        keep = True
    elif rel >= 1 and DRAMA_KEYS.search(title) and re.search(r"LLM|language model|GPT|transformer|neural", title + " " + abs, re.I):
        keep = True
    if keep:
        selected.append(r)

# sort: relevance desc then year desc
selected.sort(key=lambda r: (r.get("relevance") or 0, r.get("year") or 0), reverse=True)
json.dump(selected, open("paper_search_results/_curated.json", "w"), indent=2)
print(f"total unique={len(uniq)}  selected={len(selected)}", file=sys.stderr)
for r in selected:
    print(f"[{r.get('relevance')}] ({r.get('year')}) {r.get('title')}")
