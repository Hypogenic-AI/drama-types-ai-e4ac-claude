"""
E3b: In-domain robustness of the H1-vs-H2 verdict.

The transferred corpus subspace (E3) may not contain the reader-discriminating
directions. Here the drama subspace is data-driven (top-K PCA of GoEmotions
activations, fit on TRAIN texts only). The shared axis is the supervised
diff-in-means direction within that space. We then ask the same question:
does a per-reader DIRECTION in the K-dim subspace (M3) beat a per-reader
threshold+gain on the single shared axis (M2) on held-out texts?

No model needed (uses cached activations).
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, log_loss

RNG = np.random.default_rng(42)

def fit_eval(Xtr, ytr, Xte, yte, C):
    clf = LogisticRegression(max_iter=3000, C=C, solver="lbfgs")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return clf, roc_auc_score(yte, p), log_loss(yte, p, labels=[0, 1])

def designs(z1, Z, oh):
    n = len(z1); z1 = z1.reshape(-1, 1)
    inter = (oh[:, :, None] * Z[:, None, :]).reshape(n, -1)
    return {"M0": z1,                                  # shared single axis
            "M1": np.hstack([oh, z1]),                 # + per-reader threshold (H1 minimal)
            "M2": np.hstack([oh, oh * z1]),            # + per-reader gain on single axis
            "M2K": np.hstack([oh, Z]),                 # shared K-dim weights + reader threshold (H1, full shared subspace)
            "M3": np.hstack([oh, inter])}              # per-reader K-dim direction (H2)

def main(ge_tag, layer, K, C=0.5):
    print(f"== E3b in-domain  layer={layer} K={K} C={C} ==")
    ann = pd.read_parquet("data/ge_annotations.parquet")
    npz = np.load(f"data/ge_acts_{ge_tag}.npz", allow_pickle=True)
    tid = npz["text_id"]; X = npz[f"L{layer}"].astype(np.float32)
    id2row = {t: i for i, t in enumerate(tid)}
    ann = ann[ann["text_id"].isin(id2row)].copy()
    ann["row"] = ann["text_id"].map(id2row)
    y = ann["drama_label"].to_numpy()

    # text-disjoint split
    uids = ann["text_id"].unique(); RNG.shuffle(uids)
    train_ids = set(uids[:int(0.8 * len(uids))])
    tr = ann["text_id"].isin(train_ids).to_numpy(); te = ~tr

    # PCA fit on TRAIN texts only (no leakage)
    train_rows = np.unique(ann.loc[tr, "row"].to_numpy())
    scaler = StandardScaler().fit(X[train_rows])
    pca = PCA(n_components=K, random_state=42).fit(scaler.transform(X[train_rows]))
    Xa = pca.transform(scaler.transform(X[ann["row"].to_numpy()]))  # (n_ann, K)
    Xa = StandardScaler().fit(Xa[tr]).transform(Xa)

    # shared supervised axis (diff-in-means on TRAIN pooled label) within PCA space
    d = Xa[tr][y[tr] == 1].mean(0) - Xa[tr][y[tr] == 0].mean(0)
    d /= (np.linalg.norm(d) + 1e-9)
    z1 = Xa @ d
    z1 = (z1 - z1[tr].mean()) / (z1[tr].std() + 1e-9)

    raters = ann["rater_id"].astype("category"); rid = raters.cat.codes.to_numpy()
    R = rid.max() + 1; oh = np.eye(R, dtype=np.float32)[rid]

    D = designs(z1, Xa, oh)
    out = {"layer": layer, "K": K, "C": C, "n_readers": int(R), "models": {}}
    fitted = {}
    for name, M in D.items():
        clf, auc, ll = fit_eval(M[tr], y[tr], M[te], y[te], C)
        fitted[name] = (clf, M)
        out["models"][name] = {"test_auroc": float(auc), "n_features": int(M.shape[1])}
        print(f"  {name}: AUROC={auc:.4f} (p={M.shape[1]})")

    yte = y[te]
    p = {n: f[0].predict_proba(f[1][te])[:, 1] for n, f in fitted.items()}
    def boot(pa, pb, nb=500):
        ds = []
        for _ in range(nb):
            bi = RNG.integers(0, len(yte), len(yte))
            try: ds.append(roc_auc_score(yte[bi], pa[bi]) - roc_auc_score(yte[bi], pb[bi]))
            except ValueError: pass
        return np.percentile(ds, [2.5, 50, 97.5]).tolist()
    out["delta_M3_M2K_ci"] = boot(p["M3"], p["M2K"])   # the TRUE reader-direction test
    out["delta_M2K_M1_ci"] = boot(p["M2K"], p["M1"])    # does shared multi-dim help beyond threshold?
    print("  ΔAUROC M3-M2K (reader-specific direction) 95%CI:", [f"{x:+.4f}" for x in out["delta_M3_M2K_ci"]])
    print("  ΔAUROC M2K-M1 (shared multi-dim) 95%CI:", [f"{x:+.4f}" for x in out["delta_M2K_M1_ci"]])

    # shuffled-reader null for M3-M2K (isolates genuine reader-specific direction)
    nd = []
    for _ in range(10):
        rid_sh = RNG.permutation(R)[rid]; oh_sh = np.eye(R, dtype=np.float32)[rid_sh]
        Dn = designs(z1, Xa, oh_sh)
        _, a3, _ = fit_eval(Dn["M3"][tr], y[tr], Dn["M3"][te], yte, C)
        _, a2k, _ = fit_eval(Dn["M2K"][tr], y[tr], Dn["M2K"][te], yte, C)
        nd.append(a3 - a2k)
    real = out["models"]["M3"]["test_auroc"] - out["models"]["M2K"]["test_auroc"]
    out["real_delta_M3_M2K"] = float(real)
    out["null_delta_M3_M2K_mean"] = float(np.mean(nd)); out["null_delta_M3_M2K_std"] = float(np.std(nd))
    out["delta_M3_M2K_z"] = float((real - np.mean(nd)) / (np.std(nd) + 1e-9))
    print(f"  real ΔM3-M2K={real:+.4f}  null(shuffled reader)={np.mean(nd):+.4f}±{np.std(nd):.4f}  z={out['delta_M3_M2K_z']:.2f}")

    os.makedirs("results", exist_ok=True)
    with open(f"results/e3b_{ge_tag}_L{layer}_K{K}.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("saved results/e3b_%s_L%d_K%d.json" % (ge_tag, layer, K))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ge_tag", default="qwen7b")
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--C", type=float, default=0.5)
    a = ap.parse_args()
    main(a.ge_tag, a.layer, a.K, a.C)
