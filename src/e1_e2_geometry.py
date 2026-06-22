"""
E1: Is 'drama' a single axis, and is 'melodrama' just more drama (collinear) or a
    distinct direction (subspace)?
E2: Rank of the drama subspace built from many subconcept directions.

Outputs: results/e1_e2_<model>.json , figures/*.png
Geometry uses difference-in-means directions (robust, causal; Marks & Tegmark 2023).
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from numpy.linalg import svd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(__file__))
from extract import load_model, extract, free

RNG = np.random.default_rng(42)


def dim_direction(X, y):
    """difference-in-means unit direction (class1 - class0)."""
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-9)


def cv_auroc(X, y, n=5):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
    return cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")


def participation_ratio(vals):
    vals = np.asarray(vals, float)
    return (vals.sum() ** 2) / (np.square(vals).sum() + 1e-12)


def principal_angles(A, B):
    """principal angles (deg) between subspaces spanned by columns of A,B (orthonormalized)."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, -1, 1)
    return np.degrees(np.arccos(s))


def main(model_name, tag):
    reg = pd.read_parquet("data/corpus_register.parquet")
    sub = pd.read_parquet("data/corpus_subconcept.parquet")
    model, tok, n_layers = load_model(model_name)
    # probe layers spread across depth
    layers = sorted(set([max(1, int(f * n_layers)) for f in (0.25, 0.4, 0.5, 0.6, 0.75, 0.9)] + [n_layers]))
    print("layers:", layers)

    reg_acts = extract(model, tok, reg["text"].tolist(), layers, pooling="mean")
    sub_acts = extract(model, tok, sub["text"].tolist(), layers, pooling="mean")
    free(model)

    results = {"model": model_name, "n_layers": n_layers, "layers": layers, "by_layer": {}}
    order = ["neutral", "understated", "dramatic", "melodramatic"]
    reg["lvl"] = reg["register"].map({k: i for i, k in enumerate(order)})

    for L in layers:
        Xr = reg_acts[L]
        res = {}

        # ---- E1a: drama axis = diff-in-means(dramatic vs neutral) ----
        mask_dn = reg["register"].isin(["dramatic", "neutral"])
        Xdn = Xr[mask_dn.values]
        ydn = (reg.loc[mask_dn, "register"] == "dramatic").astype(int).values
        drama_axis = dim_direction(Xdn, ydn)
        res["drama_axis_auroc"] = float(np.mean(cv_auroc(Xdn, ydn)))

        # projection of all register items on drama axis (centered at neutral mean)
        neutral_mean = Xr[reg["register"] == "neutral"].mean(0)
        proj = (Xr - neutral_mean) @ drama_axis
        reg["_proj"] = proj
        res["mean_proj"] = {r: float(reg.loc[reg.register == r, "_proj"].mean()) for r in order}

        # ---- E1b: is melodrama collinear with drama, or off-axis? ----
        melo_axis = dim_direction(
            Xr[reg.register.isin(["melodramatic", "neutral"])],
            (reg.loc[reg.register.isin(["melodramatic", "neutral"]), "register"] == "melodramatic").astype(int).values)
        cos_drama_melo = float(drama_axis @ melo_axis)
        res["cos(drama,melo)"] = cos_drama_melo
        res["principal_angle_drama_melo_deg"] = float(principal_angles(
            drama_axis[:, None], melo_axis[:, None])[0])

        # ---- E1b-geometry: is the dramatic->melodramatic step ALONG the drama axis? ----
        # If melodrama is "just more drama", the vector (melo_mean - dramatic_mean) is
        # parallel to the drama axis. Measure the fraction of that step along the axis.
        dramatic_mean = Xr[reg.register == "dramatic"].mean(0)
        melo_mean = Xr[reg.register == "melodramatic"].mean(0)
        step = melo_mean - dramatic_mean
        step_norm = np.linalg.norm(step) + 1e-9
        along = float((step @ drama_axis))                 # signed length along drama axis
        frac_along = float(along ** 2 / step_norm ** 2)     # in [0,1]; 1 => collinear
        res["dramatic_to_melo_frac_var_along_drama_axis"] = frac_along
        res["dramatic_to_melo_angle_to_drama_axis_deg"] = float(
            np.degrees(np.arccos(np.clip(abs(along) / step_norm, 0, 1))))
        # Off-axis 'excess' direction: component of melo step orthogonal to drama axis
        excess = step - along * drama_axis
        excess /= (np.linalg.norm(excess) + 1e-9)
        res["cos(excess,drama)"] = float(excess @ drama_axis)  # ~0 by construction (sanity)
        # Do dramatic vs melodramatic differ on this off-axis direction beyond drama-axis?
        dm = reg.register.isin(["dramatic", "melodramatic"])
        ydm = (reg.loc[dm, "register"] == "melodramatic").astype(int).values
        proj_drama = (Xr[dm.values] - neutral_mean) @ drama_axis
        proj_excess = (Xr[dm.values] - neutral_mean) @ excess
        # AUROC of separating melo vs dramatic using ONLY drama-axis vs ONLY excess-axis
        res["dramatic_vs_melo_auroc_drama_axis_only"] = float(np.mean(
            cv_auroc(proj_drama.reshape(-1, 1), ydm, n=4)))
        res["dramatic_vs_melo_auroc_excess_axis_only"] = float(np.mean(
            cv_auroc(proj_excess.reshape(-1, 1), ydm, n=4)))
        res["mean_excess_proj"] = {r: float(((Xr[reg.register == r] - neutral_mean) @ excess).mean())
                                    for r in order}

        # ---- E2: subspace rank from subconcept diff-in-means directions ----
        Xs = sub_acts[L]
        dirs = []
        sc_names = []
        for sc in sorted(sub["subconcept"].unique()):
            m = sub["subconcept"] == sc
            Xsc = Xs[m.values]
            ysc = (sub.loc[m, "polarity"] == "high").astype(int).values
            dirs.append(dim_direction(Xsc, ysc))
            sc_names.append(sc)
        D = np.stack(dirs)  # (K, H) unit rows
        # singular values of the K directions => effective rank
        sv = svd(D, compute_uv=False)
        ev = sv ** 2
        res["subconcept_names"] = sc_names
        res["subspace_singular_values"] = sv.tolist()
        res["subspace_participation_ratio"] = float(participation_ratio(ev))
        res["subspace_explained_var_cumsum"] = list(np.cumsum(ev / ev.sum()))
        # control: random directions participation ratio
        ctrl_prs = []
        for _ in range(20):
            Rm = RNG.standard_normal(D.shape)
            Rm /= np.linalg.norm(Rm, axis=1, keepdims=True)
            ctrl_prs.append(participation_ratio(svd(Rm, compute_uv=False) ** 2))
        res["subspace_participation_ratio_random_mean"] = float(np.mean(ctrl_prs))
        res["subspace_participation_ratio_random_std"] = float(np.std(ctrl_prs))
        # cosine similarity matrix between subconcept directions
        res["subconcept_cos_matrix"] = (D @ D.T).tolist()
        # where does the drama axis live in the subconcept subspace? projection energy onto top-k PCs
        # PCA basis of subconcept directions
        U, S, Vt = svd(D - D.mean(0, keepdims=True), full_matrices=False)
        res["drama_axis_energy_in_subconcept_space"] = float(
            np.sum((drama_axis @ Vt[:min(5, Vt.shape[0])].T) ** 2))

        results["by_layer"][str(L)] = res
        print(f"L{L}: drama_auroc={res['drama_axis_auroc']:.3f} "
              f"cos(drama,melo)={cos_drama_melo:.3f} "
              f"melo_step_angle={res['dramatic_to_melo_angle_to_drama_axis_deg']:.0f}deg "
              f"excess_auroc={res['dramatic_vs_melo_auroc_excess_axis_only']:.2f} "
              f"subspace_PR={res['subspace_participation_ratio']:.2f}(rand {res['subspace_participation_ratio_random_mean']:.2f})")

    os.makedirs("results", exist_ok=True)
    with open(f"results/e1_e2_{tag}.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    # save projections at best layer for plotting
    reg.to_parquet(f"results/reg_proj_{tag}.parquet")
    print("saved results/e1_e2_%s.json" % tag)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tag", default="qwen7b")
    a = ap.parse_args()
    main(a.model, a.tag)
