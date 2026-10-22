"""
experiments.py : runs the whole project and saves tables/plots in results/.

    python -m src.experiments            # full run (a few minutes)
    python -m src.experiments --quick    # fast run on subsets, to check that everything works

Steps
  0. Load the prepared data (run src/preprocessing.py first).
  1. Standard k-NN baseline: tune k, weighting and rule on validation.
  2. CBR: tune the weight scheme, k and the vote rule on validation (weights computed on train only).
  3. EXPERIMENT 1 (static): CBR vs k-NN on the SAME test clients, same case base (train + validation),
     case base frozen. Includes the k-NN with the CBR representation, to show the retrieval is the same.
  4. EXPERIMENT 2 (online): test clients one by one: predict first, then revise + retain.
     Compared with the same CBR without learning, and with "retain only".
  5. Example of an explained prediction (the retrieved clients).
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.baseline_knn import load_knn_data, matched_knn, run_knn
from src.cbr import ROOT, VOTES, WEIGHT_SCHEMES, bootstrap_ba, build, evaluate, load_cbr_data, vote

RESULTS_DIR = ROOT / "results"
K_GRID = [1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 101, 151, 201]
SEED = 42


# ----------------------------------------------------------------------------
# CBR tuning (validation set only)
# ----------------------------------------------------------------------------
def tune_cbr(train, val, meta, k_grid):
    kmax = max(k_grid)
    yv = val["target"].to_numpy()
    rows = []
    for scheme in WEIGHT_SCHEMES:
        t0 = time.time()
        model = build(train, meta, scheme, kmax, "weighted_balanced").fit(train)
        xn, xc = model.encode(val)
        idx, sims = model.batch_neighbours(xn, xc, kmax)   # neighbours computed once per scheme
        lab, rel = model.y[idx], model.reliability[idx]
        for k in k_grid:
            for rule in VOTES:
                pred, _ = vote(lab[:, :k], sims[:, :k], rel[:, :k], model.base_rate, rule)
                rows.append({"weights": scheme, "k": k, "vote": rule,
                             "val_balanced_accuracy": balanced_accuracy_score(yv, pred)})
        best_s = max((r for r in rows if r["weights"] == scheme), key=lambda r: r["val_balanced_accuracy"])
        print(f"  CBR  weights={scheme:19s} best val BA = {best_s['val_balanced_accuracy']:.4f} "
              f"(k={best_s['k']}, {best_s['vote']})  [{time.time() - t0:.0f}s]")
    table = pd.DataFrame(rows)
    # best = highest validation balanced accuracy; ties -> smaller k
    best = table.sort_values(["val_balanced_accuracy", "k"], ascending=[False, True]).iloc[0].to_dict()
    best["k"] = int(best["k"])
    return best, table


# ----------------------------------------------------------------------------
# Experiment 2: online loop
# ----------------------------------------------------------------------------
def online_run(base, xn, xc, y, revise, retain):
    """Test clients one by one: PREDICT first, then (optionally) revise and retain."""
    m = base.clone()
    m.use_reliability = revise
    scores, preds = np.empty(len(y)), np.empty(len(y), dtype=int)
    for i in range(len(y)):
        idx, sims = m.neighbours(xn[i], xc[i], m.k)
        p, s = m.decide(idx, sims)                    # prediction fixed BEFORE the outcome is used
        scores[i], preds[i] = s, p
        m.revise(idx, p, y[i])                         # reliability update only if revise=True
        if retain:
            m.retain(xn[i], xc[i], y[i])
    return scores, preds, m.n


def rolling_ba(y, pred, window, step):
    xs, vals = [], []
    for end in range(window, len(y) + 1, step):
        sl = slice(end - window, end)
        vals.append(balanced_accuracy_score(y[sl], pred[sl]))
        xs.append(end)
    return np.array(xs), np.array(vals)


# ----------------------------------------------------------------------------
def main(quick=False):
    RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("Loading data ...")
    cbr_data, meta = load_cbr_data()
    knn_data = load_knn_data()
    train, val, test = cbr_data["train"], cbr_data["val"], cbr_data["test"]
    k_grid = K_GRID
    if quick:
        vi = np.sort(rng.choice(len(val), 1500, replace=False))
        val = val.iloc[vi].reset_index(drop=True)
        k_grid = [5, 21, 101]
        print("QUICK MODE: 1500 validation clients, k in", k_grid)
    y_test = test["target"].to_numpy()
    print(f"Train: {len(train)} | validation: {len(val)} | test: {len(test)}")

    # ---- 1. k-NN baseline ------------------------------------------------------
    print("\n[1/4] Standard k-NN (tuning on validation)")
    knn = run_knn(knn_data, k_grid)
    assert (knn["y_test"] == y_test).all(), "k-NN and CBR views are not aligned"

    # ---- 2. CBR tuning -----------------------------------------------------------
    print("\n[2/4] CBR (tuning weights, k and vote on validation)")
    best, tuning = tune_cbr(train, val, meta, k_grid)
    tuning.to_csv(RESULTS_DIR / "tuning_cbr.csv", index=False)
    pd.DataFrame(knn["tuning"]).to_csv(RESULTS_DIR / "tuning_knn.csv", index=False)
    print(f"  -> CBR choice: weights {best['weights']}, k={best['k']}, vote {best['vote']}")

    # ---- 3. Experiment 1 ---------------------------------------------------------
    print("\n[3/4] EXPERIMENT 1: CBR vs k-NN, case base = train + validation, frozen")
    cases = pd.concat([train, cbr_data["val"]], ignore_index=True)
    # weights computed on the TRAINING clients only, then frozen
    base = build(train, meta, best["weights"], best["k"], best["vote"]).fit(cases, extra_capacity=len(test))
    weights_used = base.weights
    xn, xc = base.encode(test)
    t0 = time.time()
    idx, sims = base.batch_neighbours(xn, xc, best["k"])
    cbr_pred, cbr_scores = vote(base.y[idx], sims, base.reliability[idx], base.base_rate, best["vote"])
    t_cbr = time.time() - t0
    xn_cb, xc_cb = base.encode(cases)
    m_major, m_bal, m_dist = matched_knn(base, xn_cb, xc_cb, cases["target"].to_numpy(), xn, xc, best["k"])
    checks = {"max_abs_diff_knn_distance_vs_cbr": float(np.abs(m_dist - (1 - sims[:500])).max()),
              "agreement_cbr_vs_matched_balanced": float((cbr_pred == m_bal).mean()),
              "cbr_seconds": t_cbr}

    results = {}
    for name, pred, sc in [("CBR (retrieve + reuse)", cbr_pred, cbr_scores),
                           ("k-NN, same representation, majority", m_major, None),
                           ("k-NN, same representation, balanced", m_bal, None),
                           ("k-NN standard (tuned)", knn["test_pred"], knn["test_scores"]),
                           ("Always 'no' (reference)", np.zeros_like(y_test), None)]:
        r = evaluate(y_test, pred, sc)
        lo, hi = bootstrap_ba(y_test, pred)
        r["BA 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
        r["predicted_yes_rate"] = float(np.mean(pred))
        results[name] = r
    exp1 = pd.DataFrame(results).T
    cols = ["balanced_accuracy", "BA 95% CI", "recall_yes", "recall_no", "precision_yes", "roc_auc", "accuracy",
            "predicted_yes_rate", "TP", "FP", "FN", "TN"]
    exp1 = exp1[[c for c in cols if c in exp1.columns]]
    exp1.to_csv(RESULTS_DIR / "experiment1.csv")
    pd.DataFrame({"row_id": test["row_id"], "target": y_test, "cbr_score": cbr_scores, "cbr_pred": cbr_pred,
                  "knn_score": knn["test_scores"], "knn_pred": knn["test_pred"]}).to_csv(
        RESULTS_DIR / "experiment1_per_client.csv", index=False)
    print(exp1.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"  retrieval check: max |k-NN distance - CBR distance| = {checks['max_abs_diff_knn_distance_vs_cbr']:.1e}")

    # ---- 4. Experiment 2 ---------------------------------------------------------
    print("\n[4/4] EXPERIMENT 2: test clients one by one (predict first, then revise + retain)")
    variants = {"CBR without learning": (False, False),
                "CBR retain only": (False, True),
                "CBR revise (reliability) + retain": (True, True)}
    window, step = 1000, 250
    fig, ax = plt.subplots(figsize=(9, 4.8))
    rows, curves, preds_by = [], [], {}
    half = len(test) // 2
    for name, (rev, ret) in variants.items():
        t0 = time.time()
        sc, pr, n_final = online_run(base, xn, xc, y_test, rev, ret)
        preds_by[name] = pr
        xs, ys = rolling_ba(y_test, pr, window, step)
        ax.plot(xs, ys, marker="o", ms=3, label=name)
        curves += [{"variant": name, "clients": int(a), "balanced_accuracy": float(b)} for a, b in zip(xs, ys)]
        static = preds_by["CBR without learning"]
        rows.append({"variant": name,
                     "BA all test clients": balanced_accuracy_score(y_test, pr),
                     "BA 2nd half of test": balanced_accuracy_score(y_test[half:], pr[half:]),
                     "decisions changed": int((pr != static).sum()),
                     "case base at the end": n_final})
        print(f"  {name:34s} done [{time.time() - t0:.0f}s]")
    xs, ys = rolling_ba(y_test, knn["test_pred"], window, step)
    ax.plot(xs, ys, ls="--", color="gray", label="k-NN standard (frozen)")
    curves += [{"variant": "k-NN standard (frozen)", "clients": int(a), "balanced_accuracy": float(b)} for a, b in zip(xs, ys)]
    rows.append({"variant": "k-NN standard (frozen)",
                 "BA all test clients": balanced_accuracy_score(y_test, knn["test_pred"]),
                 "BA 2nd half of test": balanced_accuracy_score(y_test[half:], knn["test_pred"][half:]),
                 "decisions changed": 0, "case base at the end": len(cases)})
    ax.set_xlabel("number of test clients processed")
    ax.set_ylabel(f"balanced accuracy (window of {window} clients)")
    ax.set_title("Experiment 2: quality while the case base evolves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment2.png", dpi=150)
    pd.DataFrame(curves).to_csv(RESULTS_DIR / "experiment2_curves.csv", index=False)
    exp2 = pd.DataFrame(rows).set_index("variant")
    exp2.to_csv(RESULTS_DIR / "experiment2.csv")
    print(exp2.to_string(float_format=lambda v: f"{v:.3f}"))

    with open(RESULTS_DIR / "best_parameters.json", "w", encoding="utf-8") as f:
        json.dump({"cbr": {**best, "case_base": "train + validation", "weights_values": weights_used},
                   "knn": {k: knn[k] for k in ("k", "weighting", "rule", "val_ba")},
                   "checks": checks}, f, indent=2)

    # ---- Explained prediction ----------------------------------------------------
    pick = np.where((cbr_pred == 1) & (y_test == 1))[0]
    i = int(pick[0]) if len(pick) else 0
    pred, score, table = base.explain(xn[i], xc[i], k=10, low=meta["numeric_low"], high=meta["numeric_high"])
    table.to_csv(RESULTS_DIR / "example_explanation.csv", index=False)
    print(f"\nExample: test client #{i}, score = {score:.2f}, real outcome = {'yes' if y_test[i] else 'no'}")
    show = ["rank", "similarity", "outcome", "same_categories", "poutcome", "contact", "job", "age"]
    print(table[[c for c in show if c in table.columns]].to_string(index=False))
    print(f"\nAll results saved in {RESULTS_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast run on subsets")
    main(ap.parse_args().quick)
