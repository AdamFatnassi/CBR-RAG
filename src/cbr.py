"""
cbr.py : a Case-Based Reasoning system for the Bank Marketing data.

A CASE   = one past client (problem description) + his real outcome (target: 1 = subscribed).
THE CYCLE (the 4 R's), one method each:
    retrieve : find the k most similar past clients          -> neighbours()
    reuse    : turn their outcomes into a prediction         -> decide()
    revise   : once the real outcome is known, check the prediction and correct -> revise()
    retain   : add the new client with his real outcome      -> retain()

Similarity between two clients
    distance   = sum_j w_j * d_j(a_j, b_j) / sum_j w_j          (value in [0, 1])
    similarity = 1 - distance
  d_j = |a - b| for numeric variables (scaled to [0, 1] with the training percentiles)
  d_j = 0 if same category, 1 otherwise, for categorical variables.

Weights (chosen on the validation set among three schemes)
    equal              : every variable weight 1
    grouped_economy    : equal, but the 5 economic indicators SHARE one weight (they are correlated)
    mutual_information : weight = mutual information between the variable and the outcome, computed on
                         the TRAINING clients; the 5 economic indicators share the information of the best one

Reuse (vote of the k neighbours), with pi = share of subscribers in the case base:
    majority          : yes if more than half of the neighbours subscribed
    balanced          : yes if the share of subscribers among the neighbours >= pi
                        (each neighbour counts its reliability; all reliabilities are 1 without Revise)
    weighted_balanced : same, each neighbour counting similarity * reliability
  Only ~11% of clients subscribe, so the majority vote almost never says yes; comparing with pi is the
  same as dividing each vote by the frequency of its class. No threshold is tuned.

Revise
    The prediction is compared with the real outcome; the solution kept is the real outcome.
    Optional (use_reliability=True): neighbours that voted for a wrong answer lose reliability
    (x decay), neighbours that voted correctly regain some (capped at 1). Tested as a variant in Experiment 2.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:  # notebook
    ROOT = Path.cwd()
PROCESSED_DIR = ROOT / "data" / "processed"

VOTES = ("majority", "balanced", "weighted_balanced")


def load_cbr_data(processed_dir=PROCESSED_DIR):
    processed_dir = Path(processed_dir)
    with open(processed_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    data = {k: pd.read_csv(processed_dir / f"cbr_{k}.csv") for k in ["train", "val", "test"]}
    for d in data.values():
        for c in meta["categorical_cols"]:
            d[c] = d[c].astype(str)
    return data, meta


# ----------------------------------------------------------------------------
# Variable weights
# ----------------------------------------------------------------------------
GROUPS = {
    "profile": ["age", "job", "marital", "education"],
    "credit": ["default", "housing", "loan"],
    "contact": ["contact", "month", "day_of_week"],
    "history": ["campaign", "previous", "poutcome", "pdays_group"],
    "economy": ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"],
}
WEIGHT_SCHEMES = ["equal", "grouped_economy", "mutual_information"]


def mutual_information(train, columns, categorical_cols, seed=42):
    """Mutual information between each variable and the target, on the training clients only."""
    X = train[columns].copy()
    for c in categorical_cols:
        X[c] = X[c].astype("category").cat.codes
    mi = mutual_info_classif(X, train["target"], discrete_features=[c in categorical_cols for c in columns],
                             random_state=seed)
    return dict(zip(columns, mi))


def make_weights(columns, scheme="mutual_information", train=None, categorical_cols=()):
    """Weight per variable, returned as shares that sum to 1."""
    eco = [c for c in GROUPS["economy"] if c in columns]
    if scheme == "equal":
        w = {c: 1.0 for c in columns}
    elif scheme == "grouped_economy":
        w = {c: (1.0 / len(eco) if c in eco else 1.0) for c in columns}
    elif scheme == "mutual_information":
        if train is None:
            raise ValueError("mutual_information needs the training clients")
        w = mutual_information(train, list(columns), list(categorical_cols))
        best_eco = max(w[c] for c in eco)
        for c in eco:
            w[c] = best_eco / len(eco)   # the group counts once, with the information of its best member
    else:
        raise ValueError(f"unknown weight scheme {scheme!r}")
    total = sum(w.values())
    return {c: v / total for c, v in w.items()}


# ----------------------------------------------------------------------------
# Metrics helpers (also used by baseline_knn.py and experiments.py)
# ----------------------------------------------------------------------------
def evaluate(y_true, y_pred, score=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall_yes": tp / (tp + fn) if tp + fn else 0.0,
        "recall_no": tn / (tn + fp) if tn + fp else 0.0,
        "precision_yes": tp / (tp + fp) if tp + fp else 0.0,
        "accuracy": (tp + tn) / len(y_true),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }
    if score is not None:
        out["roc_auc"] = roc_auc_score(y_true, score)
    return out


def bootstrap_ba(y_true, y_pred, n_boot=300, seed=0):
    """95% confidence interval of the balanced accuracy (resampling the test clients)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        if y_true[i].min() == y_true[i].max():
            continue
        vals.append(balanced_accuracy_score(y_true[i], y_pred[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def vote(labels, sims, rel, base_rate, rule):
    """Prediction and score for many clients. labels, sims, rel: (n_clients, k), best neighbour first.

    Returns (prediction 0/1, score). The score is the (weighted) share of subscribers among the neighbours.
    """
    labels = labels.astype(float)
    share = labels.mean(axis=1)
    if rule == "majority":
        pred = (share > 0.5).astype(int)
        tie = np.isclose(share, 0.5)
        pred[tie] = labels[tie, 0].astype(int)       # tie -> the most similar neighbour decides
        return pred, share
    if rule == "balanced":
        # each neighbour counts its reliability (all 1 unless the reliability Revise is used)
        score = (rel * labels).sum(axis=1) / rel.sum(axis=1)
        return (score >= base_rate).astype(int), score
    if rule == "weighted_balanced":
        w = np.maximum(sims, 1e-6) * rel
        score = (w * labels).sum(axis=1) / w.sum(axis=1)
        return (score >= base_rate).astype(int), score
    raise ValueError(f"unknown vote rule {rule!r}")


# ----------------------------------------------------------------------------
# The CBR system
# ----------------------------------------------------------------------------
class CBR:
    def __init__(self, numeric_cols, categorical_cols, weights=None, k=101, vote="weighted_balanced",
                 categories=None, use_reliability=False, decay=0.9, min_reliability=0.1):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        allcols = self.numeric_cols + self.categorical_cols
        self.set_weights(weights or {c: 1.0 for c in allcols})
        if vote not in VOTES:
            raise ValueError(f"vote must be one of {VOTES}")
        self.k, self.vote = k, vote
        self.use_reliability, self.decay, self.min_reliability = use_reliability, decay, min_reliability
        # category codes from the data schema (all known categories), so codes never depend on the order seen
        self.cat_maps = {c: {v: i for i, v in enumerate(categories[c])} if categories else {} for c in self.categorical_cols}
        self.n = 0

    def set_weights(self, weights):
        self.weights = weights
        self.w_num = np.array([weights[c] for c in self.numeric_cols], dtype=float)
        self.w_cat = np.array([weights[c] for c in self.categorical_cols], dtype=float)
        self.w_sum = float(self.w_num.sum() + self.w_cat.sum())
        if self.w_sum <= 0:
            raise ValueError("at least one weight must be > 0")

    # ----- building the case base -------------------------------------------
    def encode(self, df):
        """DataFrame -> (numeric matrix, categorical integer codes). Only the problem variables are read."""
        xnum = df[self.numeric_cols].to_numpy(dtype=float)
        xcat = np.empty((len(df), len(self.categorical_cols)), dtype=np.int16)
        for j, c in enumerate(self.categorical_cols):
            m = self.cat_maps[c]
            for v in pd.unique(df[c].astype(str)):
                if v not in m:
                    m[v] = len(m)
            xcat[:, j] = df[c].astype(str).map(m).to_numpy()
        return xnum, xcat

    def fit(self, df, extra_capacity=0):
        """Store clients with known outcomes as the initial case base (no learning happens here)."""
        xnum, xcat = self.encode(df)
        n = len(df)
        cap = n + extra_capacity
        self.Xnum = np.zeros((cap, len(self.numeric_cols)))
        self.Xcat = np.full((cap, len(self.categorical_cols)), -1, dtype=np.int16)
        self.y = np.zeros(cap, dtype=np.int8)
        self.reliability = np.ones(cap)
        self.Xnum[:n], self.Xcat[:n], self.y[:n] = xnum, xcat, df["target"].to_numpy()
        self.n = n
        return self

    def clone(self):
        return copy.deepcopy(self)

    @property
    def base_rate(self):
        return float(self.y[:self.n].mean())

    # ----- 1. RETRIEVE ---------------------------------------------------------
    def distances(self, xnum, xcat):
        n = self.n
        d_num = np.abs(self.Xnum[:n] - xnum) @ self.w_num
        d_cat = (self.Xcat[:n] != xcat).astype(float) @ self.w_cat
        return (d_num + d_cat) / self.w_sum

    def neighbours(self, xnum, xcat, k=None):
        """The k most similar cases of ONE client, best first. Ties: earlier case first."""
        k = min(k or self.k, self.n)
        dist = self.distances(xnum, xcat)
        part = np.argpartition(dist, k - 1)[:k]
        cand = np.flatnonzero(dist <= dist[part].max())       # every case tied at the boundary
        idx = cand[np.lexsort((cand, dist[cand]))][:k]
        return idx, 1.0 - dist[idx]

    def batch_neighbours(self, xnum, xcat, k):
        m = len(xnum)
        idx = np.empty((m, k), dtype=np.int32)
        sims = np.empty((m, k))
        for i in range(m):
            idx[i], sims[i] = self.neighbours(xnum[i], xcat[i], k)
        return idx, sims

    # ----- 2. REUSE ------------------------------------------------------------
    def decide(self, idx, sims, rule=None):
        """Prediction and score of ONE client from its neighbours."""
        pred, score = vote(self.y[idx][None, :], sims[None, :], self.reliability[idx][None, :],
                           self.base_rate, rule or self.vote)
        return int(pred[0]), float(score[0])

    def predict(self, df, k=None):
        """Scores and 0/1 predictions for a DataFrame (case base unchanged)."""
        k = k or self.k
        xnum, xcat = self.encode(df)
        idx, sims = self.batch_neighbours(xnum, xcat, k)
        pred, score = vote(self.y[idx], sims, self.reliability[idx], self.base_rate, self.vote)
        return score, pred

    # ----- 3. REVISE -----------------------------------------------------------
    def revise(self, idx, y_pred, y_true):
        """Check the prediction against the real outcome. Returns True if it was wrong.

        The solution kept for the case is always the real outcome (see retain()).
        With use_reliability=True, the neighbours are also re-weighted for later predictions."""
        wrong = y_pred != y_true
        if self.use_reliability:
            voted = self.y[idx]
            if wrong:
                bad = idx[voted == y_pred]  # neighbours that pushed toward the wrong outcome
                self.reliability[bad] = np.maximum(self.min_reliability, self.reliability[bad] * self.decay)
            else:
                good = idx[voted == y_true]
                self.reliability[good] = np.minimum(1.0, self.reliability[good] / self.decay)
        return bool(wrong)

    # ----- 4. RETAIN -----------------------------------------------------------
    def retain(self, xnum, xcat, y_true):
        """Add the new client, with his REAL outcome, to the case base."""
        if self.n >= len(self.y):
            extra = max(1000, self.n // 2)
            self.Xnum = np.vstack([self.Xnum, np.zeros((extra, self.Xnum.shape[1]))])
            self.Xcat = np.vstack([self.Xcat, np.full((extra, self.Xcat.shape[1]), -1, np.int16)])
            self.y = np.concatenate([self.y, np.zeros(extra, np.int8)])
            self.reliability = np.concatenate([self.reliability, np.ones(extra)])
        i = self.n
        self.Xnum[i], self.Xcat[i], self.y[i], self.reliability[i] = xnum, xcat, y_true, 1.0
        self.n += 1

    # ----- Explanation -----------------------------------------------------------
    def explain(self, xnum, xcat, k=None, low=None, high=None):
        """Score + table of the retrieved clients: this IS the explanation of the prediction."""
        idx, sims = self.neighbours(xnum, xcat, k)
        pred, score = self.decide(idx, sims)
        inv = {c: {v: key for key, v in m.items()} for c, m in self.cat_maps.items()}
        rows = []
        for rank, (i, s) in enumerate(zip(idx, sims), 1):
            row = {"rank": rank, "similarity": round(float(s), 3), "outcome": "yes" if self.y[i] else "no",
                   "reliability": round(float(self.reliability[i]), 2)}
            for j, c in enumerate(self.numeric_cols):
                v = float(self.Xnum[i, j])
                if low is not None:  # back to original units (capped values stay capped)
                    v = v * (high[c] - low[c]) + low[c]
                row[c] = round(v, 2)
            for j, c in enumerate(self.categorical_cols):
                row[c] = inv[c].get(int(self.Xcat[i, j]), "?")
            row["same_categories"] = int((self.Xcat[i] == xcat).sum())
            rows.append(row)
        return pred, score, pd.DataFrame(rows)


def build(train, meta, scheme, k, vote_rule, weights_train=None, **kw):
    """Convenience: a CBR system with a weight scheme (weights computed on `weights_train`)."""
    num, cat = meta["numeric_cols"], meta["categorical_cols"]
    w = make_weights(num + cat, scheme, weights_train if weights_train is not None else train, cat)
    return CBR(num, cat, w, k=k, vote=vote_rule, categories=meta["categories"], **kw)


if __name__ == "__main__":
    data, meta = load_cbr_data()
    m = build(data["train"], meta, "mutual_information", 101, "weighted_balanced").fit(data["train"])
    val = data["val"].head(1000)
    s, p = m.predict(val)
    print(f"Quick check on 1000 validation clients: balanced accuracy = {balanced_accuracy_score(val['target'], p):.3f}")
