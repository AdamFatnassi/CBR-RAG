"""Unit tests. Run from the project root (after src/preprocessing.py):
    python -m unittest discover -s tests -t .
"""
import unittest

import numpy as np
import pandas as pd

from src.baseline_knn import matched_knn
from src.cbr import CBR, build, load_cbr_data, make_weights, vote
from src.preprocessing import RAW_DIR, RAW_FILE, clean, load_raw

DATA, META = load_cbr_data()
TRAIN = DATA["train"].sample(3000, random_state=0).reset_index(drop=True)
QUERIES = DATA["val"].sample(50, random_state=0).reset_index(drop=True)
COLS = META["numeric_cols"] + META["categorical_cols"]


def system(scheme="equal", k=11, rule="balanced"):
    return build(DATA["train"], META, scheme, k, rule).fit(TRAIN, extra_capacity=10)


class TestData(unittest.TestCase):
    def test_no_duration_no_leakage_columns(self):
        self.assertNotIn("duration", COLS)
        self.assertNotIn("target", COLS)
        self.assertIn("campaign", COLS)

    def test_split_stratified_and_disjoint(self):
        rates = [DATA[k]["target"].mean() for k in DATA]
        self.assertLess(max(rates) - min(rates), 0.002)
        ids = [set(DATA[k]["row_id"]) for k in DATA]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

    def test_duplicates_removed_and_pdays_cut_is_training_median(self):
        raw = load_raw(RAW_DIR / RAW_FILE)
        self.assertEqual(len(clean(raw)), len(raw) - 12)
        train_raw = raw.set_index("row_id").loc[DATA["train"]["row_id"]]
        self.assertEqual(train_raw.loc[train_raw["pdays"] != 999, "pdays"].median(), META["pdays_cut"])

    def test_scaled_values_in_unit_interval(self):
        x = DATA["test"][META["numeric_cols"]].to_numpy()
        self.assertTrue((x >= 0).all() and (x <= 1).all())


class TestSimilarityAndWeights(unittest.TestCase):
    def test_similarity_bounds_and_identity(self):
        cbr = system()
        xn, xc = cbr.encode(TRAIN.iloc[:5])
        for i in range(5):
            idx, sims = cbr.neighbours(xn[i], xc[i], 20)
            self.assertTrue((sims >= -1e-12).all() and (sims <= 1 + 1e-12).all())
            self.assertAlmostEqual(sims[0], 1.0)

    def test_weights_sum_to_one_and_economy_shares(self):
        for s in ("equal", "grouped_economy", "mutual_information"):
            w = make_weights(COLS, s, DATA["train"], META["categorical_cols"])
            self.assertAlmostEqual(sum(w.values()), 1.0)
        w = make_weights(COLS, "mutual_information", DATA["train"], META["categorical_cols"])
        eco = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
        self.assertEqual(len({round(w[c], 12) for c in eco}), 1)    # the 5 indicators share one weight

    def test_changing_weights_changes_retrieval(self):
        w_job = {c: (1.0 if c == "job" else 0.0) for c in COLS}
        cbr = CBR(META["numeric_cols"], META["categorical_cols"], w_job, k=20, categories=META["categories"]).fit(TRAIN)
        xn, xc = cbr.encode(QUERIES.iloc[[0]])
        idx, _ = cbr.neighbours(xn[0], xc[0])
        self.assertTrue((TRAIN.loc[idx, "job"] == QUERIES.loc[0, "job"]).all())

    def test_ties_broken_by_case_order(self):
        dup = pd.concat([TRAIN.iloc[[5]]] * 3, ignore_index=True)
        cbr = system(k=3).fit(pd.concat([TRAIN, dup], ignore_index=True))
        xn, xc = cbr.encode(TRAIN.iloc[[5]])
        idx, sims = cbr.neighbours(xn[0], xc[0])
        self.assertEqual(list(idx), [5, len(TRAIN), len(TRAIN) + 1])


class TestVoteAndCycle(unittest.TestCase):
    def test_vote_rules(self):
        lab = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]); sims = np.array([[0.9] + [0.5] * 9]); rel = np.ones((1, 10))
        self.assertEqual(vote(lab, sims, rel, 0.11, "majority")[0][0], 0)
        self.assertEqual(vote(lab, sims, rel, 0.11, "balanced")[0][0], 0)            # 10% < 11%
        self.assertEqual(vote(lab, sims, rel, 0.11, "weighted_balanced")[0][0], 1)   # 0.9 / 5.4 = 16.7%

    def test_prediction_ignores_the_outcome(self):
        cbr = system()
        q = QUERIES.iloc[[2]].copy()
        s1, p1 = cbr.predict(q)
        q["target"] = 1 - q["target"]
        s2, p2 = cbr.predict(q)
        self.assertEqual(p1[0], p2[0]); self.assertEqual(s1[0], s2[0])

    def test_retain_adds_real_outcome_and_is_found_again(self):
        cbr = system(k=1)
        q = QUERIES.iloc[[4]]
        xn, xc = cbr.encode(q)
        n0, y = cbr.n, int(q["target"].iloc[0])
        idx, sims = cbr.neighbours(xn[0], xc[0])
        pred, _ = cbr.decide(idx, sims)
        cbr.revise(idx, pred, y)
        cbr.retain(xn[0], xc[0], y)
        self.assertEqual(cbr.n, n0 + 1); self.assertEqual(int(cbr.y[n0]), y)
        idx2, sims2 = cbr.neighbours(xn[0], xc[0])
        self.assertAlmostEqual(sims2[0], 1.0)

    def test_reliability_revise_only_when_enabled(self):
        cbr = system(k=11)
        idx = np.arange(11)
        cbr.revise(idx, 1, 0)
        self.assertTrue((cbr.reliability[:11] == 1).all())
        cbr.use_reliability = True
        cbr.revise(idx, 1, 0)
        voted_yes = idx[cbr.y[idx] == 1]
        self.assertTrue((cbr.reliability[voted_yes] < 1).all())


class TestMatchedKNN(unittest.TestCase):
    def test_same_distances_as_cbr(self):
        cbr = build(DATA["train"], META, "mutual_information", 15, "balanced").fit(TRAIN)
        xn_c, xc_c = cbr.encode(TRAIN)
        xn, xc = cbr.encode(QUERIES)
        _, _, dist = matched_knn(cbr, xn_c, xc_c, TRAIN["target"].to_numpy(), xn, xc, 15)
        _, sims = cbr.batch_neighbours(xn, xc, 15)
        np.testing.assert_allclose(dist, 1 - sims[:len(dist)], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
