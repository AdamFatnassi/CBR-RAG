"""
augmentation.py : Part 2 (RAG), prompt construction (Retrieved info + Query + System prompt)

Prediction : computed on the K = 84 retrieved neighbors (same k as part 1)
Generation : the LLM only sees the M = 8 most similar cases among those K

Reads : results/embedding_neighbors.csv, data/processed/text_train.csv, text_test.csv
Usage : python src/augmentation.py 1592
"""
import sys
from pathlib import Path
from typing import Union

import pandas as pd

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()

from src.preprocessing import describe_client

PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

K = 84                 # neighbors used for the prediction (retrieved by embeddings)
M = 8                  # most relevant neighbors shown to the LLM
THRESHOLD = 0.113      # threshold on the share of yes (to tune; ~ base rate)

SYSTEM_PROMPT = (
    "You are a banking marketing analyst. You justify a targeting decision for a "
    "term deposit campaign, based SOLELY on the cases provided. "
    "Do not invent any information that is not in the cases. "
    "Answer in 3 to 5 sentences: the similarities with the cases that support the decision, "
    "then the differences that qualify it."
)


def decide(neighbors: pd.DataFrame, threshold: float = THRESHOLD):
    """Decision computed in code: share of 'yes' among the K neighbors compared to a threshold."""
    p = neighbors["case_target"].mean()
    return ("yes" if p >= threshold else "no"), p


def build_prompt_from_context(context: dict, m: int = M):
    """
    Builds (system_prompt, user_prompt, decision, truth) from a context dictionary.
    Supports runtime queries from the interactive Streamlit interface.
    Expected keys in context:
      - 'query_data': dict of client features (or string text)
      - 'prediction': int, bool, or str (0/1, 'yes'/'no')
      - 'score': float (share of positive neighbours)
      - 'retrieved_cases': pd.DataFrame or list of dicts representing similar cases
    """
    query_data = context.get("query_data", context)
    if isinstance(query_data, str):
        query_text = query_data
    elif isinstance(query_data, dict):
        if "text" in query_data and pd.notna(query_data["text"]):
            query_text = query_data["text"]
        else:
            query_text = describe_client(query_data)
    else:
        query_text = str(query_data)

    table = context.get("retrieved_cases")
    if isinstance(table, list):
        table = pd.DataFrame(table)
    elif table is None:
        table = pd.DataFrame()

    total_k = len(table)

    # Positive neighbours count
    if not table.empty and "outcome" in table.columns:
        n_yes = int(table["outcome"].astype(str).str.lower().isin(["yes", "1", "true"]).sum())
    elif not table.empty and "case_target" in table.columns:
        n_yes = int((table["case_target"] == 1).sum())
    else:
        n_yes = 0

    # Share / score
    if "score" in context and context["score"] is not None:
        p = float(context["score"])
    else:
        p = (n_yes / total_k) if total_k > 0 else 0.0

    # Decision
    if "prediction" in context and context["prediction"] is not None:
        pred_val = context["prediction"]
        decision = "yes" if str(pred_val).lower() in ("1", "true", "yes") or pred_val == 1 else "no"
    else:
        decision = "yes" if p >= THRESHOLD else "no"

    top_shown = table.head(m) if not table.empty else pd.DataFrame()
    cases = []
    for rank_idx, (_, r) in enumerate(top_shown.iterrows(), start=1):
        raw_outcome = r.get("outcome") if "outcome" in r else r.get("case_target", 0)
        is_sub = str(raw_outcome).lower() in ("yes", "1", "true") or raw_outcome == 1
        outcome_str = "subscribed" if is_sub else "did not subscribe"

        rank = int(r.get("rank", rank_idx))
        sim = float(r.get("similarity", 0.0))

        if "text" in r and pd.notna(r["text"]):
            case_text = r["text"]
        else:
            try:
                case_text = describe_client(r.to_dict())
            except Exception:
                case_text = ", ".join(f"{k}: {v}" for k, v in r.items()
                                      if k not in ("rank", "similarity", "outcome", "reliability", "same_categories"))

        cases.append(f"{rank}. (similarity {sim:.3f}, {outcome_str}) {case_text}")

    cases_block = "\n".join(cases) if cases else "No similar cases available."

    user = (
        f"New client: {query_text}\n\n"
        f"Among the {total_k} most similar clients: {n_yes} subscribed, "
        f"{total_k - n_yes} did not (subscription rate: {p:.1%}).\n\n"
        f"The {len(cases)} closest cases:\n{cases_block}\n\n"
        f"Proposed decision: {'target' if decision == 'yes' else 'do not target'} this client "
        f"(subscription: {decision}).\n"
        "Write the justification using only the information above."
    )

    truth = context.get("truth", None)
    return SYSTEM_PROMPT, user, decision, truth


def build_prompt(query: Union[int, dict], k: int = K, m: int = M):
    """
    Builds the prompt either from:
      - a test set row ID (query_row_id: int) using offline retrieval files
      - a runtime context dictionary (context: dict) from the Streamlit app
    """
    if isinstance(query, dict):
        return build_prompt_from_context(query, m=m)

    query_row_id = int(query)
    nb = pd.read_csv(RESULTS_DIR / "embedding_neighbors.csv")
    train = pd.read_csv(PROCESSED_DIR / "text_train.csv").rename(columns={"row_id": "case_row_id"})
    test = pd.read_csv(PROCESSED_DIR / "text_test.csv")

    filtered_test = test[test["row_id"] == query_row_id]
    
    if filtered_test.empty:
        raise IndexError(
            f"query_row_id {query_row_id} was not found in text_test.csv. "
            f"Check if {query_row_id} belongs to text_train.csv or verify your test row IDs "
            f"(available test row_ids range from {test['row_id'].min()} to {test['row_id'].max()})."
        )
    
    q = filtered_test.iloc[0]

    # K neighbors: used for the prediction
    top_all = (nb[nb["query_row_id"] == query_row_id]
               .merge(train[["case_row_id", "text"]], on="case_row_id")
               .sort_values("rank")
               .head(k))
    decision, p = decide(top_all)
    n_yes = int(top_all["case_target"].sum())

    # M most relevant among the K: shown to the LLM
    top_shown = top_all.head(m)
    cases = []
    for _, r in top_shown.iterrows():
        outcome = "subscribed" if r["case_target"] == 1 else "did not subscribe"
        cases.append(f"{int(r['rank'])}. (similarity {r['similarity']:.3f}, {outcome}) {r['text']}")

    user = (
        f"New client: {q['text']}\n\n"
        f"Among the {len(top_all)} most similar clients: {n_yes} subscribed, "
        f"{len(top_all) - n_yes} did not (subscription rate: {p:.1%}).\n\n"
        f"The {len(top_shown)} closest cases:\n" + "\n".join(cases) + "\n\n"
        f"Proposed decision: {'target' if decision == 'yes' else 'do not target'} this client "
        f"(subscription: {decision}).\n"
        "Write the justification using only the information above."
    )
    return SYSTEM_PROMPT, user, decision, int(q["target"])


if __name__ == "__main__":
    rid = int(sys.argv[1]) if len(sys.argv) > 1 else 1592
    system, user, decision, truth = build_prompt(rid)
    print("=== SYSTEM ===\n" + system)
    print("\n=== USER ===\n" + user)
    print(f"\nComputed decision: {decision} | true decision: {'yes' if truth else 'no'}")
