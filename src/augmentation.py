"""
augmentation.py : Part 2 (RAG), prompt construction (Retrieved info + Query + System prompt)

Prediction : computed on the K = 84 retrieved neighbors (same k as part 1)
Generation : the LLM only sees the M = 8 most similar cases among those K

Reads : results/embedding_neighbors.csv, data/processed/text_train.csv, text_test.csv
Usage : python src/augmentation.py 1592
"""
import sys
from pathlib import Path

import pandas as pd

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()

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


def build_prompt(query_row_id: int, k: int = K, m: int = M):
    nb = pd.read_csv(RESULTS_DIR / "embedding_neighbors.csv")
    train = pd.read_csv(PROCESSED_DIR / "text_train.csv").rename(columns={"row_id": "case_row_id"})
    test = pd.read_csv(PROCESSED_DIR / "text_test.csv")

    q = test[test["row_id"] == query_row_id].iloc[0]

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