"""
app.py : Streamlit interface for the Bank Marketing CBR project.

    pip install streamlit
    streamlit run src/app.py

Prerequisites: src/preprocessing.py has been run (data/processed exists).
For the "Results" tab and the default settings, also run `python -m src.experiments` once (results/ folder).
(A web version with the same features is in web/index.html.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so that `src` imports work from anywhere
from src.cbr import GROUPS, ROOT, VOTES, WEIGHT_SCHEMES, build, load_cbr_data  # noqa: E402

RESULTS_DIR = ROOT / "results"
INT_COLS = {"age", "campaign", "previous"}
GROUP_TITLES = {"profile": "Client profile", "credit": "Credit", "contact": "Current call",
                "history": "History with the bank", "economy": "Economic context"}
VOTE_NAMES = {"majority": "majority", "balanced": "balanced", "weighted_balanced": "similarity-weighted balanced"}

st.set_page_config(page_title="Bank Marketing - Case-Based Reasoning", layout="wide")


@st.cache_data
def load_data():
    return load_cbr_data()


@st.cache_resource
def load_base(scheme):
    """Initial case base = training + validation clients; weights computed on the training clients."""
    data, meta = load_data()
    cases = pd.concat([data["train"], data["val"]], ignore_index=True)
    return build(data["train"], meta, scheme, 151, "weighted_balanced").fit(cases, extra_capacity=5000)


def default_params():
    path = RESULTS_DIR / "best_parameters.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["cbr"]
    return {"weights": "mutual_information", "k": 151, "vote": "weighted_balanced"}


def reset_session(scheme):
    st.session_state.model = load_base(scheme).clone()
    st.session_state.scheme = scheme
    st.session_state.last = None
    st.session_state.n_added = 0


try:
    data, meta = load_data()
except FileNotFoundError:
    st.error("data/processed not found. Run `python src/preprocessing.py` first.")
    st.stop()

num_cols, cat_cols = meta["numeric_cols"], meta["categorical_cols"]
low, high = meta["numeric_low"], meta["numeric_high"]
best = default_params()

# ----------------------------------------------------------------------------
# Sidebar: CBR settings (tuned on validation)
# ----------------------------------------------------------------------------
st.sidebar.header("CBR settings")
scheme = st.sidebar.selectbox("Variable weights", WEIGHT_SCHEMES, index=WEIGHT_SCHEMES.index(best["weights"]))
k = st.sidebar.slider("k (similar clients retrieved)", 1, 301, int(best["k"]))
rule = st.sidebar.selectbox("Vote", list(VOTES), index=list(VOTES).index(best["vote"]), format_func=VOTE_NAMES.get)
use_rel = st.sidebar.checkbox("Revise also lowers the reliability of misleading neighbours", value=False)
st.sidebar.caption(f"Tuned on validation: weights {best['weights']}, k = {best['k']}, {VOTE_NAMES[best['vote']]} vote.")

if "model" not in st.session_state or st.session_state.get("scheme") != scheme:
    reset_session(scheme)
if st.sidebar.button("Reset case base"):
    reset_session(scheme)

model = st.session_state.model
model.k, model.vote, model.use_reliability = k, rule, use_rel
st.sidebar.metric("Cases in the case base", model.n)
st.sidebar.caption(f"Clients added in this session: {st.session_state.n_added} · "
                   f"share of subscribers: {model.base_rate:.1%}")

tab_predict, tab_results, tab_about = st.tabs(["Predict a client", "Results", "About CBR"])


# ----------------------------------------------------------------------------
# Tab 1: predict, explain, revise, retain
# ----------------------------------------------------------------------------
def raw_median(col):
    v = float(data["train"][col].median()) * (high[col] - low[col]) + low[col]
    return int(round(v)) if col in INT_COLS else v


def numeric_input(col, container):
    mn, mx = meta["numeric_min"][col], meta["numeric_max"][col]
    if col in INT_COLS:
        return container.number_input(col, min_value=int(mn), max_value=int(mx),
                                      value=int(np.clip(raw_median(col), mn, mx)), step=1, key=f"in_{col}")
    return container.number_input(col, min_value=float(mn), max_value=float(mx), value=float(np.clip(raw_median(col), mn, mx)),
                                  step=float((mx - mn) / 200) or 0.01, format="%.3f", key=f"in_{col}")


def categorical_input(col, container):
    options = meta["categories"][col]
    mode = data["train"][col].mode().iloc[0]
    return container.selectbox(col, options, index=options.index(mode), key=f"in_{col}")


with tab_predict:
    st.subheader("1. Describe the client (known before the call)")
    st.caption("The call duration is not asked: it is unknown before the call and was removed from the data.")
    values = {}
    with st.form("client_form"):
        for g, cols in GROUPS.items():
            with st.expander(GROUP_TITLES[g], expanded=g in ("profile", "history")):
                grid = st.columns(2)
                for i, c in enumerate(cols):
                    values[c] = numeric_input(c, grid[i % 2]) if c in num_cols else categorical_input(c, grid[i % 2])
        submitted = st.form_submit_button("Retrieve similar clients and predict")

    if submitted:
        row = {c: float(np.clip((values[c] - low[c]) / ((high[c] - low[c]) or 1), 0, 1)) for c in num_cols}
        row.update({c: values[c] for c in cat_cols})
        xn, xc = model.encode(pd.DataFrame([row]))
        idx, sims = model.neighbours(xn[0], xc[0], k)
        pred, score, table = model.explain(xn[0], xc[0], k=k, low=low, high=high)
        st.session_state.last = {"idx": idx, "xn": xn[0], "xc": xc[0], "score": score, "pred": pred,
                                 "table": table, "answered": False}

    last = st.session_state.last
    if last is not None:
        st.subheader("2. Prediction and explanation")
        c1, c2, c3 = st.columns(3)
        c1.metric("Score (share of subscribers among neighbours)", f"{last['score']:.3f}")
        c2.metric("Compared with", "0.5" if model.vote == "majority" else f"{model.base_rate:.3f} (all clients)")
        c3.metric("Prediction", "Will subscribe" if last["pred"] else "Will not subscribe")
        table = last["table"]
        n_yes = int((table["outcome"] == "yes").sum())
        st.write(f"Among the **{len(table)} most similar past clients**, **{n_yes}** subscribed and "
                 f"**{len(table) - n_yes}** did not. Only {model.base_rate:.1%} of all clients subscribe, so a share "
                 "above that means this client looks more likely than average. This is the explanation:")
        first = ["rank", "similarity", "outcome", "reliability", "same_categories"]
        st.dataframe(table[first + [c for c in table.columns if c not in first]], hide_index=True)

        st.subheader("3. Revise and retain")
        if last["answered"]:
            st.success("This client has been added to the case base with his real outcome.")
        else:
            real = st.radio("What really happened after the call?", ["Subscribed", "Did not subscribe"], horizontal=True)
            if st.button("Confirm real outcome (Revise + Retain)"):
                y_true = 1 if real == "Subscribed" else 0
                wrong = model.revise(last["idx"], last["pred"], y_true)
                model.retain(last["xn"], last["xc"], y_true)
                last["answered"] = True
                st.session_state.n_added += 1
                msg = "The prediction was wrong: the solution is corrected to the real outcome (REVISE)." if wrong else \
                      "The prediction was right (REVISE)."
                if model.use_reliability:
                    msg += " Neighbours that misled us lost reliability; those that voted correctly regained some."
                st.warning(msg) if wrong else st.success(msg)
                st.info(f"The client is now in the case base (RETAIN): {model.n} clients. Predict the same client again to see it.")


# ----------------------------------------------------------------------------
# Tab 2: results of experiments.py
# ----------------------------------------------------------------------------
with tab_results:
    exp1, exp2, img = RESULTS_DIR / "experiment1.csv", RESULTS_DIR / "experiment2.csv", RESULTS_DIR / "experiment2.png"
    if not exp1.exists():
        st.info("No results yet. Run `python -m src.experiments` first.")
    else:
        st.subheader("Experiment 1: CBR vs k-NN (same test clients, same case base, frozen)")
        st.dataframe(pd.read_csv(exp1, index_col=0))
        st.caption("Main metric: balanced accuracy. 'k-NN, same representation' finds the same neighbours as the CBR: "
                   "the difference with the majority version shows the effect of the vote. 'Always no' shows why accuracy is misleading.")
    if exp2.exists():
        st.subheader("Experiment 2: test clients one by one (predict, then revise + retain)")
        st.dataframe(pd.read_csv(exp2, index_col=0))
    if img.exists():
        st.image(str(img))


# ----------------------------------------------------------------------------
# Tab 3: about
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown("""
### The CBR cycle in this app

| Step | What happens | Where you see it |
|---|---|---|
| **Retrieve** | Find the k most similar past clients (weighted similarity) | The table of neighbours |
| **Reuse** | Vote: predict "yes" if the (similarity-weighted) share of subscribers among them is at least the share among all clients | The score and the decision |
| **Revise** | Compare with the real outcome; the solution kept is the real outcome (optionally, misleading neighbours lose reliability) | The "Confirm real outcome" button |
| **Retain** | Add the new client, with his real outcome, to the case base | The case counter in the sidebar |

**Weights:** proportional to the mutual information between each variable and the outcome, computed on the training
clients; the five correlated economic indicators share one weight. Chosen on validation against equal weights.

**Why the vote compares with ~11%:** only about 11% of clients subscribe, so a majority vote would almost always
predict "no". Comparing with the base rate is the same as dividing each vote by the frequency of its class.

**Why there is no call duration:** it is unknown before the call, so using it would be data leakage.

**RAG vs this app:** here the retrieved cases *are* the answer (their outcomes are reused directly).
In RAG, the retrieved items would be handed to a generator (an LLM) as context.
""")
