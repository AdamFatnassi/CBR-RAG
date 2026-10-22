"""
baseline_knn.py : the k-NN reference methods (scikit-learn).

1. Standard k-NN: the k-NN view prepared by preprocessing.py (standardized numeric + one-hot categories),
   Euclidean distance, no variable weights. k, the neighbour weighting (uniform / distance) and the
   decision rule (majority / balanced) are chosen on the validation set, like for the CBR.

2. Matched k-NN: the SAME representation and weights as our CBR. Numeric columns are multiplied by
   their weight, one-hot categories by half their weight, and the Manhattan distance is used.
   Then the k-NN distance equals the CBR distance, so both find the same neighbours: any difference
   in results comes from the vote, not from the retrieval.

The case base is never modified (no revise, no retain).
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from src.cbr import PROCESSED_DIR


def load_knn_data(processed_dir=PROCESSED_DIR):
    d = np.load(Path(processed_dir) / "knn_arrays.npz")
    return {k: d[k] for k in d.files}


def _decide(p, rule, base_rate):
    return (p > 0.5).astype(int) if rule == "majority" else (p >= base_rate - 1e-12).astype(int)


def run_knn(data, k_grid, verbose=True):
    """Standard k-NN: tune k, weighting and rule on validation, refit on train + validation, predict test."""
    Xtr, ytr, Xva, yva = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    Xte, yte = data["X_test"], data["y_test"]
    nn = NearestNeighbors(n_neighbors=max(k_grid), algorithm="brute").fit(Xtr)
    dist, idx = nn.kneighbors(Xva)                 # computed once, sliced for every k
    lab, base = ytr[idx], ytr.mean()
    tuning, best = [], None
    for k in k_grid:
        d, l = dist[:, :k], lab[:, :k]
        for weighting in ("uniform", "distance"):
            if weighting == "uniform":
                p = l.mean(axis=1)
            else:  # same as scikit-learn: 1/d, exact matches dominate
                with np.errstate(divide="ignore"):
                    w = 1.0 / d
                zero = d == 0
                w[zero.any(axis=1)] = zero[zero.any(axis=1)].astype(float)
                p = (w * l).sum(axis=1) / w.sum(axis=1)
            for rule in ("majority", "balanced"):
                ba = balanced_accuracy_score(yva, _decide(p, rule, base))
                tuning.append({"k": k, "weighting": weighting, "rule": rule, "val_balanced_accuracy": ba})
                if best is None or ba > best["val_ba"] + 1e-12:
                    best = {"k": k, "weighting": weighting, "rule": rule, "val_ba": ba}
    if verbose:
        print(f"  k-NN standard: best k={best['k']} {best['weighting']} {best['rule']} -> val BA {best['val_ba']:.4f}")
    # final model: case base = train + validation
    Xcb, ycb = np.vstack([Xtr, Xva]), np.concatenate([ytr, yva])
    model = KNeighborsClassifier(n_neighbors=best["k"], weights=best["weighting"], algorithm="brute").fit(Xcb, ycb)
    test_scores = model.predict_proba(Xte)[:, 1]
    test_pred = _decide(test_scores, best["rule"], ycb.mean())
    return {**{k: best[k] for k in ("k", "weighting", "rule")}, "val_ba": best["val_ba"],
            "test_scores": test_scores, "test_pred": test_pred, "y_test": yte, "tuning": tuning}


def matched_encoding(xnum, xcat, cbr):
    """Encode clients so that Manhattan distance = the CBR distance (weights as shares summing to 1)."""
    w_num, w_cat = cbr.w_num / cbr.w_sum, cbr.w_cat / cbr.w_sum
    parts = [xnum * w_num]
    for j, c in enumerate(cbr.categorical_cols):
        n_cat = len(cbr.cat_maps[c])
        oh = np.zeros((len(xcat), n_cat))
        oh[np.arange(len(xcat)), xcat[:, j]] = 1.0
        parts.append(oh * 0.5 * w_cat[j])
    return np.hstack(parts)


def matched_knn(cbr, xnum_cases, xcat_cases, y_cases, xnum, xcat, k):
    """k-NN with the CBR representation. Returns majority predictions, balanced predictions, distances."""
    model = KNeighborsClassifier(n_neighbors=k, metric="manhattan", algorithm="brute")
    model.fit(matched_encoding(xnum_cases, xcat_cases, cbr), y_cases)
    Xq = matched_encoding(xnum, xcat, cbr)
    p = model.predict_proba(Xq)[:, 1]
    dist, _ = model.kneighbors(Xq[:500])
    return model.predict(Xq), _decide(p, "balanced", y_cases.mean()), dist
