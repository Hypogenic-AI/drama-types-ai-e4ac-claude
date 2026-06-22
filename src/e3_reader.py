"""
E3 (CRUX): Do real readers' drama judgments differ by a THRESHOLD on a single shared
drama axis (H1), or by a DIRECTION within a multi-dimensional drama subspace (H2)?

Nested logistic models predicting per-rater drama_label on GoEmotions, using a
drama subspace TRANSFERRED from the controlled corpus (cross-dataset):
  M0  global:            sigma(a*z1 + c)
  M1  H1-threshold:      per-reader intercept, shared slope on z1
  M2  H1-gain:           per-reader intercept + per-reader slope on z1 (still 1-D axis)
  M3  H2-subspace:       per-reader intercept + per-reader weight vector over D-dim drama subspace
H2 supported iff M3 beats M2 on HELD-OUT (text-disjoint) data beyond the
shuffled-reader null. Also: rank of the fitted reader weight matrix => reader-space dim.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss

sys.path.insert(0, os.path.dirname(__file__))
from extract import load_model, extract, free

RNG = np.random.default_rng(42)


def dim_direction(X, y):
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-9)


def corpus_subspace(model_name, layer):
    """Return (drama_axis, B) at a layer: B = 8 subconcept diff-in-means directions."""
    reg = pd.read_parquet("data/corpus_register.parquet")
    sub = pd.read_parquet("data/corpus_subconcept.parquet")
    model, tok, n_layers = load_model(model_name)
    racts = extract(model, tok, reg["text"].tolist(), [layer], pooling="mean")[layer]
    sacts = extract(model, tok, sub["text"].tolist(), [layer], pooling="mean")[layer]
    free(model)
    dn = reg.register.isin(["dramatic", "neutral"])
    drama_axis = dim_direction(racts[dn.values],
                               (reg.loc[dn, "register"] == "dramatic").astype(int).values)
    dirs, names = [], []
    for sc in sorted(sub["subconcept"].unique()):
        m = (sub["subconcept"] == sc).values
        dirs.append(dim_direction(sacts[m], (sub.loc[m, "polarity"] == "high").astype(int).values))
        names.append(sc)
    return drama_axis, np.stack(dirs), names


def fit_eval(design_tr, ytr, design_te, yte, C=1.0):
    clf = LogisticRegression(max_iter=3000, C=C, solver="lbfgs")
    clf.fit(design_tr, ytr)
    p = clf.predict_proba(design_te)[:, 1]
    return clf, roc_auc_score(yte, p), log_loss(yte, p, labels=[0, 1])


def make_designs(z1, Z, rid_onehot):
    """Return design matrices for M0,M1,M2,M3 given standardized features."""
    n = len(z1)
    z1 = z1.reshape(-1, 1)
    M0 = z1
    M1 = np.hstack([rid_onehot, z1])                       # per-reader intercept + shared slope
    M2 = np.hstack([rid_onehot, rid_onehot * z1])          # + per-reader slope on z1
    inter = (rid_onehot[:, :, None] * Z[:, None, :]).reshape(n, -1)  # reader x Z interactions
    M3 = np.hstack([rid_onehot, inter])                    # per-reader weight vector over Z
    return {"M0": M0, "M1": M1, "M2": M2, "M3": M3}


def main(model_name, ge_tag, layer, C=0.5, n_boot=500):
    print(f"== E3 model={model_name} layer={layer} ==")
    ann = pd.read_parquet("data/ge_annotations.parquet")
    npz = np.load(f"data/ge_acts_{ge_tag}.npz", allow_pickle=True)
    tid = npz["text_id"]
    X = npz[f"L{layer}"].astype(np.float32)
    id2row = {t: i for i, t in enumerate(tid)}
    ann = ann[ann["text_id"].isin(id2row)].copy()
    ann["row"] = ann["text_id"].map(id2row)
    y = ann["drama_label"].to_numpy()

    # transfer drama subspace from corpus
    drama_axis, B, names = corpus_subspace(model_name, layer)
    Xa = X[ann["row"].to_numpy()]
    z1 = Xa @ drama_axis                          # single shared drama axis
    Zsub = Xa @ B.T                               # 8 subconcept projections
    Z = np.hstack([z1.reshape(-1, 1), Zsub])      # D=9 drama-subspace features (z1 included => M3 nests M2)
    feat_names = ["drama_axis"] + names

    # standardize features (fit on all; split is text-disjoint so fine)
    zsc = StandardScaler().fit(z1.reshape(-1, 1)); z1s = zsc.transform(z1.reshape(-1, 1)).ravel()
    Zsc = StandardScaler().fit(Z); Zs = Zsc.transform(Z)

    # reader index
    raters = ann["rater_id"].astype("category")
    rid = raters.cat.codes.to_numpy()
    R = rid.max() + 1
    onehot = np.eye(R, dtype=np.float32)[rid]

    # text-disjoint split (primary): 80/20 by text_id
    uids = ann["text_id"].unique()
    RNG.shuffle(uids)
    cut = int(0.8 * len(uids))
    train_ids = set(uids[:cut])
    tr = ann["text_id"].isin(train_ids).to_numpy()
    te = ~tr
    print(f"annotations: {len(ann)}  train {tr.sum()} test {te.sum()}  readers {R}  base_rate {y.mean():.3f}")

    designs = make_designs(z1s, Zs, onehot)
    out = {"model": model_name, "layer": layer, "C": C, "n_readers": int(R),
           "feat_names": feat_names, "n_ann": int(len(ann)),
           "base_rate": float(y.mean()), "models": {}}
    fitted = {}
    for name, D in designs.items():
        clf, auc, ll = fit_eval(D[tr], y[tr], D[te], y[te], C=C)
        fitted[name] = (clf, D)
        out["models"][name] = {"test_auroc": float(auc), "test_logloss": float(ll),
                               "n_features": int(D.shape[1])}
        print(f"  {name}: AUROC={auc:.4f} logloss={ll:.4f} (p={D.shape[1]})")

    # bootstrap CI on test for key contrasts M3-M2, M2-M1
    pte = {}
    for name, (clf, D) in fitted.items():
        pte[name] = clf.predict_proba(D[te])[:, 1]
    yte = y[te]
    idx = np.where(te)[0]
    def boot_delta(pa, pb):
        deltas = []
        n = len(yte)
        for _ in range(n_boot):
            bi = RNG.integers(0, n, n)
            try:
                deltas.append(roc_auc_score(yte[bi], pa[bi]) - roc_auc_score(yte[bi], pb[bi]))
            except ValueError:
                pass
        return np.percentile(deltas, [2.5, 50, 97.5]).tolist()
    out["delta_M3_M2_auroc_ci"] = boot_delta(pte["M3"], pte["M2"])
    out["delta_M2_M1_auroc_ci"] = boot_delta(pte["M2"], pte["M1"])
    out["delta_M1_M0_auroc_ci"] = boot_delta(pte["M1"], pte["M0"])
    print("  ΔAUROC M3-M2 95%CI:", [f"{x:+.4f}" for x in out["delta_M3_M2_auroc_ci"]])
    print("  ΔAUROC M2-M1 95%CI:", [f"{x:+.4f}" for x in out["delta_M2_M1_auroc_ci"]])

    # ---- shuffled-reader null control: permute rater ids, refit M3 ----
    null_deltas = []
    for k in range(10):
        perm = RNG.permutation(R)
        rid_sh = perm[rid]
        oh_sh = np.eye(R, dtype=np.float32)[rid_sh]
        Dn = make_designs(z1s, Zs, oh_sh)
        _, auc3, _ = fit_eval(Dn["M3"][tr], y[tr], Dn["M3"][te], yte, C=C)
        _, auc2, _ = fit_eval(Dn["M2"][tr], y[tr], Dn["M2"][te], yte, C=C)
        null_deltas.append(auc3 - auc2)
    out["null_delta_M3_M2_mean"] = float(np.mean(null_deltas))
    out["null_delta_M3_M2_std"] = float(np.std(null_deltas))
    real_delta = out["models"]["M3"]["test_auroc"] - out["models"]["M2"]["test_auroc"]
    out["real_delta_M3_M2"] = float(real_delta)
    z = (real_delta - np.mean(null_deltas)) / (np.std(null_deltas) + 1e-9)
    out["delta_M3_M2_z_vs_null"] = float(z)
    print(f"  real ΔM3-M2={real_delta:+.4f}  null={np.mean(null_deltas):+.4f}±{np.std(null_deltas):.4f}  z={z:.2f}")

    # ---- reader-space dimensionality: rank of fitted reader weight matrix (M3) ----
    clf3 = fitted["M3"][0]
    coef = clf3.coef_.ravel()
    Dz = Z.shape[1]
    W = coef[R:].reshape(R, Dz)            # (readers x D) interaction weights
    Wc = W - W.mean(0, keepdims=True)
    sv = np.linalg.svd(Wc, compute_uv=False)
    pr = (sv.sum() ** 2) / (np.square(sv).sum() + 1e-12)
    out["reader_weight_matrix_singular_values"] = sv.tolist()
    out["reader_space_participation_ratio"] = float(pr)
    out["reader_space_evr_cumsum"] = list(np.cumsum(sv**2 / (sv**2).sum()))
    out["reader_weight_matrix"] = W.tolist()
    print(f"  reader-space participation ratio (of D={Dz}): {pr:.2f}")
    print(f"  reader-space EVR cumsum (first 4): {[f'{x:.2f}' for x in out['reader_space_evr_cumsum'][:4]]}")

    os.makedirs("results", exist_ok=True)
    with open(f"results/e3_{ge_tag}_L{layer}.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("saved results/e3_%s_L%d.json" % (ge_tag, layer))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--ge_tag", default="qwen7b")
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--C", type=float, default=0.5)
    a = ap.parse_args()
    main(a.model, a.ge_tag, a.layer, a.C)
