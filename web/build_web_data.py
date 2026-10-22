"""Export the data and results needed by the web demo into web/data.json.

Run from the project root, after src/preprocessing.py and `python -m src.experiments`:
    python web/build_web_data.py

Also writes web/reference.json (Python predictions) used by web/verify_engine.js.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cbr import GROUPS, WEIGHT_SCHEMES, build, load_cbr_data, make_weights  # noqa: E402
from src.preprocessing import RAW_DIR, RAW_FILE, load_raw  # noqa: E402

RES, WEB = ROOT / "results", ROOT / "web"
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
LABELS = {"age": "Age", "job": "Job", "marital": "Marital status", "education": "Education",
          "default": "Credit in default", "housing": "Housing loan", "loan": "Personal loan",
          "contact": "Contact type", "month": "Month of the call", "day_of_week": "Day of the week",
          "campaign": "Calls in this campaign (incl. this one)", "previous": "Contacts before this campaign",
          "poutcome": "Outcome of the previous campaign", "pdays_group": "Days since last contact",
          "emp.var.rate": "Employment variation rate", "cons.price.idx": "Consumer price index",
          "cons.conf.idx": "Consumer confidence index", "euribor3m": "Euribor 3-month rate",
          "nr.employed": "Number of employees (thousands)"}


def main():
    data, meta = load_cbr_data()
    best = json.loads((RES / "best_parameters.json").read_text())
    p = best["cbr"]
    num, cat = meta["numeric_cols"], meta["categorical_cols"]
    train, val, test = data["train"], data["val"], data["test"]
    cases = pd.concat([train, val], ignore_index=True)

    raw = load_raw(RAW_DIR / RAW_FILE).set_index("row_id")     # original values (incl. pdays) by row_id
    model = build(train, meta, p["weights"], p["k"], p["vote"]).fit(cases)
    vocab = {c: list(model.cat_maps[c].keys()) for c in cat}

    def columns(df):
        r = raw.loc[df["row_id"]]
        return {"num": [r[c].astype(float).round(6).tolist() for c in num],
                "cat": [[vocab[c].index(v) for v in df[c].astype(str)] for c in cat],
                "pdays": r["pdays"].astype(int).tolist(),
                "y": df["target"].astype(int).tolist(), "row_id": df["row_id"].astype(int).tolist()}

    # economic periods (one month each), labelled with their year; subscription rate from TRAIN only
    key = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "nr.employed"]
    allp = raw.reset_index().groupby(key).agg(first=("row_id", "min"), month=("month", "first"),
                                              euribor=("euribor3m", "median")).sort_values("first").reset_index()
    train_raw = raw.loc[train["row_id"]]
    year, prev, periods = 2008, None, []
    for _, r in allp.iterrows():
        m = MONTHS.index(r["month"])
        if prev is not None and m < prev:
            year += 1
        prev = m
        tr = train_raw[(train_raw[key] == r[key].values).all(axis=1)]
        periods.append({"label": f"{r['month'].capitalize()} {year}", "month": r["month"],
                        "values": {**{c: float(r[c]) for c in key}, "euribor3m": float(r["euribor"])},
                        "train_rate": float((tr["y"] == "yes").mean()) if len(tr) else None})

    per = pd.read_csv(RES / "experiment1_per_client.csv")
    test_cols = columns(test)
    test_cols.update({"cbr_pred": per["cbr_pred"].astype(int).tolist(), "knn_pred": per["knn_pred"].astype(int).tolist()})

    weights = {s: make_weights(num + cat, s, train, cat) for s in WEIGHT_SCHEMES}
    config = {
        "num": num, "cat": cat, "low": meta["numeric_low"], "high": meta["numeric_high"],
        "min": meta["numeric_min"], "max": meta["numeric_max"], "vocab": vocab, "groups": GROUPS,
        "weights": weights, "schemes": WEIGHT_SCHEMES, "best": {k: p[k] for k in ("weights", "k", "vote", "val_balanced_accuracy")},
        "knn_best": best["knn"], "checks": best["checks"], "decay": model.decay, "min_reliability": model.min_reliability,
        "labels": LABELS, "int_cols": ["age", "campaign", "previous"], "pdays_cut": meta["pdays_cut"],
        "sizes": meta["sizes"], "case_base_size": len(cases),
        "modes": {c: train[c].mode().iloc[0] for c in cat},
        "medians": {c: float(train_raw[c].median()) for c in num},
        "train_rate": float(train["target"].mean()),
    }
    out = {
        "config": config, "cases": columns(cases), "test": test_cols, "periods": periods,
        "exp1": json.loads(pd.read_csv(RES / "experiment1.csv", index_col=0).reset_index().to_json(orient="records")),
        "exp2": json.loads(pd.read_csv(RES / "experiment2.csv", index_col=0).reset_index().to_json(orient="records")),
        "curves": pd.read_csv(RES / "experiment2_curves.csv").to_dict(orient="list"),
        "tuning_cbr": pd.read_csv(RES / "tuning_cbr.csv").round(4).to_dict(orient="list"),
        "tuning_knn": pd.read_csv(RES / "tuning_knn.csv").round(4).to_dict(orient="list"),
    }
    (WEB / "data.json").write_text(json.dumps(out, separators=(",", ":")))
    print("web/data.json", round((WEB / "data.json").stat().st_size / 1e6, 2), "MB")

    # ---------- reference for the JS check: static predictions + online runs ----------
    sample = test.iloc[:300]
    xn, xc = model.encode(sample)
    score, pred = model.predict(sample)
    ref = {"clients": columns(sample), "scores": [float(s) for s in score], "preds": [int(x) for x in pred], "online": {}}
    for name, rel in (("retain", False), ("revise_reliability", True)):
        m = build(train, meta, p["weights"], p["k"], p["vote"], use_reliability=rel).fit(cases, extra_capacity=400)
        preds = []
        for i in range(len(sample)):
            idx, sims = m.neighbours(xn[i], xc[i])
            pr, _ = m.decide(idx, sims)
            preds.append(pr)
            m.revise(idx, pr, int(sample["target"].iloc[i]))
            m.retain(xn[i], xc[i], int(sample["target"].iloc[i]))
        ref["online"][name] = {"preds": preds, "reliability_sum": float(m.reliability[:m.n].sum()), "n": int(m.n)}
    (WEB / "reference.json").write_text(json.dumps(ref))
    print("web/reference.json written")


if __name__ == "__main__":
    main()
