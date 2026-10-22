"""
preprocessing.py : shared preprocessing for the Bank Marketing project (CBR + k-NN)

Reads   : data/raw/bank-additional-full.csv
Writes  : data/processed/...   (same split, three views of it)

Run from the project root:
    python src/preprocessing.py                  # random stratified split (default)
    python src/preprocessing.py --split time     # time-based split (file is chronological)

What it does
  1. Loads the CSV (separator ';'), removes the 12 exact duplicate rows and DROPS `duration`
     (not known before the call). `campaign` is kept (known before the call).
  2. `pdays` (999 = no recorded contact) becomes a category `pdays_group`:
     'none' / '<=6 days' / '>6 days'. 6 is the median of the recorded values in the training part.
  3. Makes ONE split: train 60% (case base) / validation 20% (tuning) / test 20%, stratified, seed 42.
  4. Fits all scalers/encoders on TRAIN ONLY (no leakage), applies them to val/test.
     Numeric variables are scaled to [0, 1] with the 1st and 99th percentiles of the training
     clients and clipped: one client called 56 times does not squeeze everybody else.
  5. Saves two views of the same split, with identical rows and row order:
       - CBR view : numeric scaled to [0, 1], categories kept as text
       - k-NN view: standardized numeric + one-hot categories (numpy arrays)
  6. Saves a third view, the TEXT view (text_train/val/test.csv): one short description per client,
     built from the cleaned RAW values (not the scaled ones), same rows and same order as the
     other views. Columns: row_id, target, text. The label is NOT written inside the text.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:  # running inside a notebook
    ROOT = Path.cwd()

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_FILE = "bank-additional-full.csv"

SEED = 42
VAL_SIZE = 0.20
TEST_SIZE = 0.20
PDAYS_CUT = 6

NUMERIC = ["age", "campaign", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
           "euribor3m", "nr.employed"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact", "month",
               "day_of_week", "poutcome", "pdays_group"]

# ---- text view settings ----
TEXT_INCLUDE_MACRO = True  # False -> leave the 5 macro-economic variables out of the descriptions
DAY_NAMES = {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday", "fri": "Friday"}
EDUCATION_NAMES = {"basic.4y": "basic (4 years)", "basic.6y": "basic (6 years)",
                   "basic.9y": "basic (9 years)", "high.school": "high school",
                   "illiterate": "illiterate", "professional.course": "professional course",
                   "university.degree": "university degree", "unknown": "unknown"}
POUTCOME_NAMES = {"nonexistent": "no previous campaign", "failure": "failure", "success": "success"}
PDAYS_NAMES = {"none": "no recorded previous contact",
               "<=6 days": f"last contact {PDAYS_CUT} days or less before this campaign",
               ">6 days": f"last contact more than {PDAYS_CUT} days before this campaign"}


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put {RAW_FILE} in {RAW_DIR} (separator ';').")
    df = pd.read_csv(path, sep=";")
    df.insert(0, "row_id", np.arange(len(df)))  # original position = time order
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 0. exact duplicates (same call recorded twice, even the same duration)
    df = df.loc[~df.drop(columns="row_id").duplicated()]

    # 1. Leakage: call length is unknown before the call
    df = df.drop(columns=["duration"])

    # 2. Target
    df["target"] = (df["y"] == "yes").astype(int)
    df = df.drop(columns=["y"])

    # 3. pdays: 999 is a code ("no recorded contact"), not a number of days -> 3 groups
    df["pdays_group"] = np.where(df["pdays"] == 999, "none",
                                 np.where(df["pdays"] <= PDAYS_CUT, "<=6 days", ">6 days"))
    df = df.drop(columns=["pdays"])

    # 4. 'unknown' stays as its own category; NaN (if any) becomes 'unknown'
    for c in CATEGORICAL:
        df[c] = df[c].fillna("unknown").astype(str)
    return df.reset_index(drop=True)


def make_split(df: pd.DataFrame, mode: str):
    if mode == "time":
        # File is in chronological order: oldest -> train, newest -> test
        n = len(df)
        n_test, n_val = int(n * TEST_SIZE), int(n * VAL_SIZE)
        train, val, test = df.iloc[: n - n_val - n_test], df.iloc[n - n_val - n_test: n - n_test], df.iloc[n - n_test:]
    else:
        train_val, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["target"], random_state=SEED)
        val_frac = VAL_SIZE / (1 - TEST_SIZE)
        train, val = train_test_split(train_val, test_size=val_frac, stratify=train_val["target"], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_views(train, val, test):
    meta_cols = ["row_id", "target"]
    num_cols, cat_cols = list(NUMERIC), list(CATEGORICAL)

    # ---------------- CBR view: percentile scaling fitted on TRAIN ----------------
    low = train[num_cols].quantile(0.01)
    high = train[num_cols].quantile(0.99)
    rng = (high - low).replace(0, 1)

    def cbr_view(d):
        scaled = ((d[num_cols] - low) / rng).clip(0, 1)  # values beyond the percentiles are capped
        return pd.concat([d[meta_cols], scaled, d[cat_cols]], axis=1)

    cbr = {"train": cbr_view(train), "val": cbr_view(val), "test": cbr_view(test)}

    # ---------------- k-NN view: standardize + one-hot, fitted on TRAIN ----------------
    scaler = StandardScaler().fit(train[num_cols])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older scikit-learn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    ohe.fit(train[cat_cols])

    def knn_view(d):
        return np.hstack([scaler.transform(d[num_cols]), ohe.transform(d[cat_cols])])

    knn = {k: knn_view(d) for k, d in zip(["train", "val", "test"], [train, val, test])}
    meta = {
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "numeric_low": low.to_dict(),            # 1st percentile (train)  -> scaled value 0
        "numeric_high": high.to_dict(),          # 99th percentile (train) -> scaled value 1
        "numeric_min": train[num_cols].min().to_dict(),   # for the input forms only
        "numeric_max": train[num_cols].max().to_dict(),
        "categories": {c: sorted(pd.concat([train[c], val[c], test[c]]).unique().tolist()) for c in cat_cols},
        "knn_feature_names": list(num_cols) + list(ohe.get_feature_names_out(cat_cols)),
        "pdays_cut": PDAYS_CUT,
        "seed": SEED,
    }
    return cbr, knn, meta


def describe_client(r: dict, include_macro: bool = TEXT_INCLUDE_MACRO) -> str:
    """One client (a row of the CLEANED dataframe, raw values) -> one short description.

    Fixed template: same wording for every client. `unknown` is written explicitly.
    The target is never part of the text.
    """
    job = r["job"].replace(".", "").replace("-", " ")
    text = (
        f"Client aged {int(r['age'])}, job: {job}, marital status: {r['marital']}, "
        f"education: {EDUCATION_NAMES.get(r['education'], r['education'])}. "
        f"Credit in default: {r['default']}; housing loan: {r['housing']}; personal loan: {r['loan']}. "
        f"Contacted by {r['contact']} in {r['month'].capitalize()} on a {DAY_NAMES.get(r['day_of_week'], r['day_of_week'])}. "
        f"Contacts during this campaign: {int(r['campaign'])}. "
        f"Before this campaign: {int(r['previous'])} contacts, "
        f"previous outcome: {POUTCOME_NAMES.get(r['poutcome'], r['poutcome'])}, "
        f"{PDAYS_NAMES[r['pdays_group']]}."
    )
    if include_macro:
        text += (
            f" Economic context: employment variation rate {r['emp.var.rate']:g}, "
            f"consumer price index {r['cons.price.idx']:g}, consumer confidence index {r['cons.conf.idx']:g}, "
            f"euribor 3-month rate {r['euribor3m']:g}, number of employees {r['nr.employed']:g}."
        )
    return text


def build_text_view(train, val, test, include_macro: bool = TEXT_INCLUDE_MACRO):
    """TEXT view: row_id, target, text. Same rows and row order as the CBR / k-NN views.

    Built from the cleaned raw values, so nothing is fitted: no leakage between splits.
    """
    views = {}
    for name, d in zip(["train", "val", "test"], [train, val, test]):
        texts = [describe_client(r, include_macro) for r in d.to_dict("records")]
        views[name] = pd.DataFrame({"row_id": d["row_id"].values,
                                    "target": d["target"].values,
                                    "text": texts})
    return views


def save_text_view(text_views):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, d in text_views.items():
        d.to_csv(PROCESSED_DIR / f"text_{name}.csv", index=False)


def check_text_view(text_views, cbr):
    print("\n=== Text view checks ===")
    for k in ["train", "val", "test"]:
        t = text_views[k]
        assert len(t) == len(cbr[k]), "row mismatch between text and CBR views"
        assert (t["row_id"].values == cbr[k]["row_id"].values).all(), "row order mismatch (text vs CBR)"
        assert (t["target"].values == cbr[k]["target"].values).all(), "target mismatch (text vs CBR)"
        assert t["text"].str.len().min() > 0, "empty description"
        dup = t["text"].duplicated(keep=False).mean()
        print(f"{k:5s}: {len(t):6d} texts | mean length = {t['text'].str.len().mean():.0f} chars "
              f"| clients sharing an identical text = {dup:.1%}")
    print("Example:", text_views["train"]["text"].iloc[0])
    print("OK: text view aligned with the other views.")


def save(cbr, knn, ytr, meta, mode):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, d in cbr.items():
        d.to_csv(PROCESSED_DIR / f"cbr_{name}.csv", index=False)
    np.savez_compressed(
        PROCESSED_DIR / "knn_arrays.npz",
        X_train=knn["train"], y_train=ytr["train"], X_val=knn["val"], y_val=ytr["val"],
        X_test=knn["test"], y_test=ytr["test"],
        row_id_train=cbr["train"]["row_id"].values, row_id_val=cbr["val"]["row_id"].values,
        row_id_test=cbr["test"]["row_id"].values,
    )
    meta["split_mode"] = mode
    meta["sizes"] = {k: int(len(v)) for k, v in cbr.items()}
    with open(PROCESSED_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def sanity_checks(cbr, knn, ytr):
    print("\n=== Sanity checks ===")
    for k in ["train", "val", "test"]:
        n = len(cbr[k])
        print(f"{k:5s}: {n:6d} rows | positive rate = {cbr[k]['target'].mean():.3%} | kNN matrix = {knn[k].shape}")
        assert knn[k].shape[0] == n, "row mismatch between views"
        assert (cbr[k]["target"].values == ytr[k]).all(), "target mismatch between views"
        assert not np.isnan(knn[k]).any(), "NaN in kNN matrix"
    ids = [set(cbr[k]["row_id"]) for k in ["train", "val", "test"]]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2]), "overlap between splits"
    print("OK: no overlap between splits, views are aligned, no NaN.")


def run(split: str = "random"):
    raw_path = RAW_DIR / RAW_FILE
    print(f"Reading  : {raw_path}")
    raw = load_raw(raw_path)
    df = clean(raw)
    print(f"Cleaned  : {len(raw)} rows -> {len(df)} after removing exact duplicates; "
          f"{len(NUMERIC) + len(CATEGORICAL)} features (duration removed, pdays -> pdays_group)")
    train, val, test = make_split(df, split)
    cbr, knn, meta = build_views(train, val, test)
    ytr = {"train": train["target"].values, "val": val["target"].values, "test": test["target"].values}
    text_views = build_text_view(train, val, test)
    meta["text_view"] = {"columns": ["row_id", "target", "text"],
                         "include_macro": TEXT_INCLUDE_MACRO,
                         "files": [f"text_{k}.csv" for k in ["train", "val", "test"]]}
    save(cbr, knn, ytr, meta, split)
    save_text_view(text_views)
    sanity_checks(cbr, knn, ytr)
    check_text_view(text_views, cbr)
    print(f"\nSaved in : {PROCESSED_DIR}")
    return cbr, knn, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "time"], default="random")
    run(ap.parse_args().split)