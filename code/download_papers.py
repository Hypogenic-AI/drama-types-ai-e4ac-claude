#!/usr/bin/env python3
"""Resolve paper-finder results to open-access PDFs and download them.

Handles:
  - arxiv.org URLs / abs IDs  -> https://arxiv.org/pdf/{id}.pdf
  - api.semanticscholar.org/CorpusId:N -> use S2 Graph API to get openAccessPdf
Writes PDFs to papers/ and prints a manifest line per paper.
"""
import json, os, re, sys, time, glob
import requests

PAPERS = os.path.join(os.path.dirname(__file__), "..", "papers")
os.makedirs(PAPERS, exist_ok=True)
S2 = "https://api.semanticscholar.org/graph/v1/paper/"
HEAD = {"User-Agent": "Mozilla/5.0 (research resource finder)"}


def slug(s, n=60):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s[:n]


def arxiv_id_from(url):
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", url or "")
    if m:
        return m.group(1)
    return None


def corpus_id_from(url):
    m = re.search(r"Corpus[Ii][Dd]:?(\d+)", url or "")
    return m.group(1) if m else None


def s2_hash_from(url):
    m = re.search(r"semanticscholar\.org/paper/(?:[^/]+/)?([0-9a-f]{40})", url or "")
    return m.group(1) if m else None


def download(url, path):
    try:
        r = requests.get(url, headers=HEAD, timeout=60, allow_redirects=True)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            open(path, "wb").write(r.content)
            return True
        # some arxiv pdfs need .pdf appended / retry
    except Exception as e:
        print(f"   ! download error {e}", file=sys.stderr)
    return False


def resolve_s2(pid):
    """Return (pdf_url, arxiv_id) from Semantic Scholar. pid may be 'CorpusID:N' or a 40-char hash."""
    fields = "externalIds,openAccessPdf,title,year"
    for attempt in range(3):
        try:
            r = requests.get(f"{S2}{pid}", params={"fields": fields},
                             headers=HEAD, timeout=40)
            if r.status_code == 200:
                d = r.json()
                pdf = (d.get("openAccessPdf") or {}).get("url")
                ext = d.get("externalIds") or {}
                ax = ext.get("ArXiv")
                return pdf, ax
            if r.status_code == 429:
                time.sleep(3 + attempt * 3)
                continue
        except Exception as e:
            time.sleep(2)
    return None, None


def process(papers, limit=None):
    manifest = []
    seen = set()
    count = 0
    for p in papers:
        title = p.get("title", "untitled")
        key = slug(title)
        if key in seen:
            continue
        seen.add(key)
        url = p.get("url") or ""
        ax = arxiv_id_from(url)
        pdf_url = None
        cid = corpus_id_from(url)
        h = s2_hash_from(url)
        pid = (f"CorpusID:{cid}" if cid else h)
        if not ax and pid:
            pdf_url, ax2 = resolve_s2(pid)
            ax = ax or ax2
            time.sleep(1.1)  # be polite to S2
        fname = f"{key}.pdf"
        fpath = os.path.join(PAPERS, fname)
        ok = False
        src = ""
        if os.path.exists(fpath) and os.path.getsize(fpath) > 1000:
            ok = True; src = "cached"
        if not ok and ax:
            ok = download(f"https://arxiv.org/pdf/{ax}.pdf", fpath); src = f"arxiv:{ax}"
        if not ok and pdf_url:
            ok = download(pdf_url, fpath); src = "s2-oa"
        status = "OK" if ok else "NO-PDF"
        print(f"[{status}] {title[:70]} | {src}")
        manifest.append({"title": title, "year": p.get("year"), "file": fname if ok else None,
                         "downloaded": ok, "source": src, "url": url,
                         "authors": p.get("authors"), "abstract": p.get("abstract"),
                         "relevance": p.get("relevance")})
        if ok:
            count += 1
        if limit and count >= limit:
            break
    return manifest


if __name__ == "__main__":
    files = sys.argv[1:] if len(sys.argv) > 1 else glob.glob(
        os.path.join(os.path.dirname(__file__), "..", "paper_search_results", "*.jsonl"))
    allp = []
    for f in files:
        txt = open(f).read().strip()
        try:
            obj = json.loads(txt)
            if isinstance(obj, list):
                allp.extend(obj); continue
        except Exception:
            pass
        for line in txt.splitlines():
            line = line.strip()
            if line:
                try:
                    allp.append(json.loads(line))
                except Exception:
                    pass
    print(f"Loaded {len(allp)} paper records from {len(files)} files", file=sys.stderr)
    man = process(allp)
    json.dump(man, open(os.path.join(PAPERS, "download_manifest.json"), "w"), indent=2)
    ok = sum(1 for m in man if m["downloaded"])
    print(f"\nDownloaded {ok}/{len(man)} unique papers", file=sys.stderr)
