"""Generate all figures from results/*.json into figures/."""
import os, json, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


def load(p):
    with open(p) as f:
        return json.load(f)


def fig_e1_e2():
    files = {"Qwen2.5-7B": "results/e1_e2_qwen7b.json",
             "Qwen2.5-1.5B": "results/e1_e2_qwen1p5b.json"}
    data = {k: load(v) for k, v in files.items() if os.path.exists(v)}
    if not data:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    # (a) melodrama-step angle vs depth
    ax = axes[0]
    for name, d in data.items():
        layers = [int(l) for l in d["by_layer"]]
        frac = [d["by_layer"][str(l)]["dramatic_to_melo_angle_to_drama_axis_deg"] for l in layers]
        rel = [l / d["n_layers"] for l in layers]
        ax.plot(rel, frac, "o-", label=name)
    ax.axhline(90, ls=":", c="gray"); ax.axhline(0, ls=":", c="gray")
    ax.set_xlabel("relative depth"); ax.set_ylabel("angle (deg)")
    ax.set_title("(a) Angle of dramatic→melodramatic\nstep vs the drama axis")
    ax.text(0.02, 85, "orthogonal (off-axis)", fontsize=8, color="gray")
    ax.text(0.02, 4, "collinear (just 'more drama')", fontsize=8, color="gray")
    ax.legend(fontsize=8); ax.set_ylim(-5, 95)
    # (b) subspace participation ratio vs depth
    ax = axes[1]
    for name, d in data.items():
        layers = [int(l) for l in d["by_layer"]]
        pr = [d["by_layer"][str(l)]["subspace_participation_ratio"] for l in layers]
        rel = [l / d["n_layers"] for l in layers]
        ax.plot(rel, pr, "s-", label=name)
    rnd = list(data.values())[0]["by_layer"]
    rndpr = np.mean([v["subspace_participation_ratio_random_mean"] for v in rnd.values()])
    ax.axhline(rndpr, ls="--", c="red", label=f"random dirs (~{rndpr:.1f})")
    ax.axhline(1, ls=":", c="gray", label="single axis (=1)")
    ax.set_xlabel("relative depth"); ax.set_ylabel("participation ratio")
    ax.set_title("(b) Drama subspace dimensionality\n(8 subconcepts)")
    ax.legend(fontsize=8)
    # (c) subconcept cosine heatmap (7B last layer)
    ax = axes[2]
    d = data.get("Qwen2.5-7B", list(data.values())[0])
    L = max(int(l) for l in d["by_layer"])
    res = d["by_layer"][str(L)]
    C = np.array(res["subconcept_cos_matrix"]); names = res["subconcept_names"]
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_title(f"(c) Subconcept direction cosines\n(7B, layer {L})")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig("figures/e1_e2_geometry.png", bbox_inches="tight"); plt.close()
    print("saved figures/e1_e2_geometry.png")


def fig_plane():
    fs = glob.glob("results/corpus_plane_*.parquet")
    if not fs:
        return
    df = pd.read_parquet(sorted(fs)[0])
    fig, ax = plt.subplots(figsize=(6, 5.2))
    colors = {"neutral": "#888888", "understated": "#2c7fb8",
              "dramatic": "#41ab5d", "melodramatic": "#d7301f"}
    for r, c in colors.items():
        sub = df[df.register == r]
        ax.scatter(sub.drama_proj, sub.excess_proj, c=c, label=r, s=45,
                   edgecolor="k", linewidth=0.3, alpha=0.85)
        ax.scatter(sub.drama_proj.mean(), sub.excess_proj.mean(), c=c, s=320,
                   marker="*", edgecolor="k", linewidth=1.2)
    ax.set_xlabel("drama axis  (neutral → dramatic)")
    ax.set_ylabel("excess / melodrama axis  (off drama-axis)")
    ax.set_title("Drama vs Excess plane (Qwen2.5-7B)\nmelodrama separates on a 2nd, ~orthogonal axis")
    ax.legend(fontsize=9); ax.axhline(0, c="gray", lw=0.5); ax.axvline(0, c="gray", lw=0.5)
    plt.tight_layout(); plt.savefig("figures/drama_excess_plane.png", bbox_inches="tight"); plt.close()
    print("saved figures/drama_excess_plane.png")


def fig_e3():
    fs = sorted(glob.glob("results/e3_*.json"))
    if not fs:
        return
    d = load(fs[0])
    mods = ["M0", "M1", "M2", "M3"]
    labels = {"M0": "M0\nglobal\naxis", "M1": "M1\n+reader\nthreshold", "M2": "M2\n+reader\ngain",
              "M3": "M3\n+reader\nsubspace dir"}
    auc = [d["models"][m]["test_auroc"] for m in mods]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    bars = ax.bar(range(4), auc, color=["#bbbbbb", "#74a9cf", "#2b8cbe", "#d7301f"])
    ax.set_xticks(range(4)); ax.set_xticklabels([labels[m] for m in mods], fontsize=8)
    ax.set_ylim(min(auc) - 0.012, max(auc) + 0.01)
    ax.set_ylabel("held-out test AUROC")
    ax.set_title(f"(a) Nested reader models, transferred subspace\n(layer {d['layer']}, {d['n_readers']} readers)")
    for b, a in zip(bars, auc):
        ax.text(b.get_x() + b.get_width() / 2, a, f"{a:.4f}", ha="center", va="bottom", fontsize=8)
    # delta annotation
    ci = d["delta_M3_M2_auroc_ci"]
    ax.text(0.5, 0.02, f"ΔM3−M2 95%CI: [{ci[0]:+.4f}, {ci[2]:+.4f}]\nnull(shuffled reader): "
            f"{d['null_delta_M3_M2_mean']:+.4f}±{d['null_delta_M3_M2_std']:.4f}  z={d['delta_M3_M2_z_vs_null']:.1f}",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", fc="#fff5eb"))
    # (b) reader weight matrix heatmap
    ax = axes[1]
    W = np.array(d["reader_weight_matrix"]); names = d["feat_names"]
    # sort readers by first PC for visual structure
    Wc = W - W.mean(0)
    u, s, vt = np.linalg.svd(Wc, full_matrices=False)
    order = np.argsort(u[:, 0])
    im = ax.imshow(W[order], aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(W).max(), vmax=np.abs(W).max())
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("reader (sorted by PC1)")
    ax.set_title("(b) Per-reader weights over\ndrama subspace (M3)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    # (c) reader-space spectrum
    ax = axes[2]
    evr = d["reader_space_evr_cumsum"]
    ax.plot(range(1, len(evr) + 1), evr, "o-")
    ax.axhline(0.9, ls=":", c="gray")
    ax.set_xlabel("# reader-space dimensions")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title(f"(c) Reader-feature space spectrum\nparticipation ratio={d['reader_space_participation_ratio']:.2f} of {len(names)}")
    ax.set_ylim(0, 1.02)
    plt.tight_layout(); plt.savefig("figures/e3_reader_models.png", bbox_inches="tight"); plt.close()
    print("saved figures/e3_reader_models.png")


def fig_persona():
    fs = glob.glob("results/e4_persona_*.json")
    if not fs:
        return
    d = load(sorted(fs)[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    C = np.array(d["axis_cosine_matrix"]); names = d["personas"]
    ax = axes[0]
    im = ax.imshow(C, cmap="viridis", vmin=min(0.5, C.min()), vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    color="w" if C[i, j] < 0.9 else "k", fontsize=7)
    ax.set_title("(a) Cosine between persona-conditioned\ndrama axes")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1]
    proj = d["persona_register_projection_on_own_axis"]
    order = ["neutral", "understated", "dramatic", "melodramatic"]
    # normalize each persona to its own 'dramatic' projection (compare SHAPE, not
    # the raw scale, which differs because 'neutral' = no chat template).
    for p in names:
        v = np.array([proj[p][r] for r in order], float)
        denom = proj[p]["dramatic"] if abs(proj[p]["dramatic"]) > 1e-6 else 1.0
        ax.plot(order, v / denom, "o-", label=p)
    ax.set_ylabel("projection on own drama axis\n(normalized to 'dramatic'=1)")
    ax.set_title("(b) Register ordering per persona\n(shape, not scale): no reordering")
    ax.legend(fontsize=7); ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); plt.savefig("figures/e4_persona.png", bbox_inches="tight"); plt.close()
    print("saved figures/e4_persona.png")


def fig_e3b():
    """In-domain decomposition across layers: the H1 verdict figure."""
    fs = sorted(glob.glob("results/e3b_qwen7b_L*_K30.json"))
    if not fs:
        return
    ds = [load(f) for f in fs]
    ds = sorted(ds, key=lambda d: d["layer"])
    rungs = ["M0", "M1", "M2K", "M3"]
    rlab = {"M0": "M0\nshared\n1 axis", "M1": "M1\n+reader\nthreshold",
            "M2K": "M2K\nshared\nK-dim", "M3": "M3\nreader\nK-dim dir"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    w = 0.8 / len(ds)
    for i, d in enumerate(ds):
        auc = [d["models"][m]["test_auroc"] for m in rungs]
        ax.bar(np.arange(len(rungs)) + i * w, auc, w, label=f"layer {d['layer']}")
    ax.set_xticks(np.arange(len(rungs)) + w * (len(ds) - 1) / 2)
    ax.set_xticklabels([rlab[r] for r in rungs], fontsize=8)
    ax.set_ylabel("held-out test AUROC"); ax.set_ylim(0.62, 0.715)
    ax.set_title("(a) In-domain nested ladder (GoEmotions, K=30 PCA)\nbig jump = reader threshold; M3≈M2K")
    ax.legend(fontsize=8)
    # (b) real vs null ΔM3-M2K
    ax = axes[1]
    x = np.arange(len(ds))
    real = [d["real_delta_M3_M2K"] for d in ds]
    null = [d["null_delta_M3_M2K_mean"] for d in ds]
    nstd = [d["null_delta_M3_M2K_std"] for d in ds]
    ax.bar(x - 0.2, real, 0.4, label="real readers", color="#d7301f")
    ax.bar(x + 0.2, null, 0.4, yerr=nstd, label="shuffled-reader null", color="#999999")
    ax.set_xticks(x); ax.set_xticklabels([f"L{d['layer']}" for d in ds])
    ax.set_ylabel("ΔAUROC (M3 − M2K)")
    ax.set_title("(b) Reader-specific DIRECTION effect\nreal = null  ⇒  no genuine reader-direction (H1)")
    ax.legend(fontsize=8); ax.axhline(0, c="k", lw=0.5)
    plt.tight_layout(); plt.savefig("figures/e3b_indomain.png", bbox_inches="tight"); plt.close()
    print("saved figures/e3b_indomain.png")


if __name__ == "__main__":
    fig_e1_e2(); fig_plane(); fig_e3(); fig_e3b(); fig_persona()
    print("done figures")
