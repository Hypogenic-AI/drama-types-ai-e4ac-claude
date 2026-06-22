"""
E4 (persona probe): Does conditioning the model on a READER PERSONA shift a
threshold along a fixed drama axis (H1) or ROTATE the drama direction (H2)?

For each persona we re-derive the drama axis (dramatic-neutral diff-in-means) from
the SAME corpus read under that persona's system prompt, then compare axes
(cosine / principal angle) and where each persona places melodrama.
Also saves a 2D (drama, excess) projection of the no-persona corpus for figures.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from extract import load_model, extract, free

RNG = np.random.default_rng(42)
PERSONAS = {
    "neutral": None,
    "cynic": "You are a hardened cynic. You find most emotional displays excessive, theatrical and melodramatic, and you are rarely moved.",
    "romantic": "You are a deeply sentimental romantic. You are moved to tears easily and find intense emotional expression beautiful and profound.",
    "stoic": "You are a stoic minimalist who prizes emotional restraint and understatement, and is uncomfortable with overt displays of feeling.",
    "thrill_seeker": "You are an excitable teenager who loves high drama, intensity and excitement, and is bored by calm or restrained things.",
    "critic": "You are a discerning literary critic, attuned to subtle, earned emotion and quick to notice when feeling is overwrought or insincere.",
}

def dim_direction(X, y):
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-9)

def main(model_name="Qwen/Qwen2.5-7B-Instruct", layer=21, tag="qwen7b"):
    reg = pd.read_parquet("data/corpus_register.parquet")
    model, tok, n_layers = load_model(model_name)
    order = ["neutral", "understated", "dramatic", "melodramatic"]

    axes, proj = {}, {}
    base_acts = None
    for pname, ptext in PERSONAS.items():
        acts = extract(model, tok, reg["text"].tolist(), [layer], pooling="mean",
                       persona=ptext)[layer]
        if pname == "neutral":
            base_acts = acts
        dn = reg.register.isin(["dramatic", "neutral"])
        ax = dim_direction(acts[dn.values],
                           (reg.loc[dn, "register"] == "dramatic").astype(int).values)
        axes[pname] = ax
        nmean = acts[reg.register == "neutral"].mean(0)
        proj[pname] = {r: float(((acts[reg.register == r] - nmean) @ ax).mean()) for r in order}
    free(model)

    pnames = list(PERSONAS.keys())
    cosmat = np.array([[float(axes[a] @ axes[b]) for b in pnames] for a in pnames])
    out = {"model": model_name, "layer": layer, "personas": pnames,
           "axis_cosine_matrix": cosmat.tolist(),
           "persona_register_projection_on_own_axis": proj,
           "min_offdiag_cosine": float(cosmat[~np.eye(len(pnames), dtype=bool)].min()),
           "mean_offdiag_cosine": float(cosmat[~np.eye(len(pnames), dtype=bool)].mean())}

    # cross-projection: project each persona's melodrama onto the NEUTRAL persona axis,
    # to see if personas mostly shift threshold (axis fixed) vs need their own axis.
    out["note"] = ("High off-diagonal cosine => personas share a drama axis (threshold shift, H1). "
                   "Lower cosine => persona rotates the drama direction (H2).")
    os.makedirs("results", exist_ok=True)
    with open(f"results/e4_persona_{tag}.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("persona axis cosine matrix:")
    print("        " + " ".join(f"{p[:6]:>7}" for p in pnames))
    for i, p in enumerate(pnames):
        print(f"{p[:7]:>7} " + " ".join(f"{cosmat[i,j]:7.3f}" for j in range(len(pnames))))
    print("mean off-diag cosine:", round(out["mean_offdiag_cosine"], 3),
          "min:", round(out["min_offdiag_cosine"], 3))

    # ---- save 2D (drama, excess) projection of no-persona corpus for figure ----
    dn = reg.register.isin(["dramatic", "neutral"])
    drama_axis = dim_direction(base_acts[dn.values],
                               (reg.loc[dn, "register"] == "dramatic").astype(int).values)
    nmean = base_acts[reg.register == "neutral"].mean(0)
    dramatic_mean = base_acts[reg.register == "dramatic"].mean(0)
    melo_mean = base_acts[reg.register == "melodramatic"].mean(0)
    step = melo_mean - dramatic_mean
    excess = step - (step @ drama_axis) * drama_axis
    excess /= (np.linalg.norm(excess) + 1e-9)
    reg2 = reg.copy()
    reg2["drama_proj"] = (base_acts - nmean) @ drama_axis
    reg2["excess_proj"] = (base_acts - nmean) @ excess
    reg2[["register", "scenario", "drama_proj", "excess_proj"]].to_parquet(
        f"results/corpus_plane_{tag}_L{layer}.parquet")
    print("saved 2D plane to results/corpus_plane_%s_L%d.parquet" % (tag, layer))
    return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--tag", default="qwen7b")
    a = ap.parse_args()
    main(a.model, a.layer, a.tag)
