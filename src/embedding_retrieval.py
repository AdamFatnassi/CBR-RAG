"""
embedding_retrieval.py : Partie 2 (RAG), embeddings + recherche des k voisins

Lit     : data/processed/text_train.csv (base de cas), text_test.csv (requêtes)
Ecrit   : data/processed/case_embeddings.npy
          results/embedding_neighbors.csv

Lancer depuis la racine du projet :
    python src/embedding_retrieval.py            # test complet
    python src/embedding_retrieval.py --n 200    # essai rapide sur 200 clients de test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()

PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

MODEL_NAME = "all-MiniLM-L6-v2"
K = 84            # même k que la partie 1
BATCH_SIZE = 256


def load_texts():
    train = pd.read_csv(PROCESSED_DIR / "text_train.csv")
    test = pd.read_csv(PROCESSED_DIR / "text_test.csv")
    return train, test


def build_index(train: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """Embeddings normalisés de la base de cas (calculés une fois, puis sauvegardés)."""
    path = PROCESSED_DIR / "case_embeddings.npy"
    if path.exists():
        emb = np.load(path)
        if emb.shape[0] == len(train):
            print(f"Embeddings rechargés : {emb.shape}")
            return emb
    print(f"Calcul des embeddings de {len(train)} cas ...")
    emb = model.encode(train["text"].tolist(), batch_size=BATCH_SIZE,
                       normalize_embeddings=True, show_progress_bar=True)
    np.save(path, emb)
    return emb


def retrieve(query_emb: np.ndarray, case_emb: np.ndarray, k: int = K):
    """Similarité cosinus (vecteurs normalisés = produit scalaire). Retourne indices et scores."""
    scores = case_emb @ query_emb
    top = np.argpartition(-scores, k)[:k]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]


def run(n_test=None):
    RESULTS_DIR.mkdir(exist_ok=True)
    train, test = load_texts()
    if n_test:
        test = test.head(n_test)

    model = SentenceTransformer(MODEL_NAME)
    case_emb = build_index(train, model)

    print(f"Encodage de {len(test)} clients de test ...")
    test_emb = model.encode(test["text"].tolist(), batch_size=BATCH_SIZE,
                            normalize_embeddings=True, show_progress_bar=True)

    train_ids = train["row_id"].to_numpy()
    train_y = train["target"].to_numpy()

    rows = []
    for i in range(len(test)):
        top, sc = retrieve(test_emb[i], case_emb, K)
        for rank, (j, s) in enumerate(zip(top, sc), start=1):
            rows.append({"query_row_id": int(test["row_id"].iloc[i]),
                         "query_target": int(test["target"].iloc[i]),
                         "rank": rank,
                         "case_row_id": int(train_ids[j]),
                         "case_target": int(train_y[j]),
                         "similarity": float(s)})
    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "embedding_neighbors.csv"
    out.to_csv(out_path, index=False)
    print(f"Sauvegardé : {out_path} ({len(out)} lignes)")

    # Contrôle rapide : part de 'yes' parmi les voisins vs dans la base
    vote = out.groupby("query_row_id")["case_target"].mean()
    print(f"Part de 'yes' dans les voisins : {vote.mean():.3f} (base : {train_y.mean():.3f})")

    # Exemple : premier client de test et ses 3 meilleurs voisins
    q = test.iloc[0]
    print("\n--- Exemple ---")
    print("Client :", q["text"])
    ex = out[out["query_row_id"] == q["row_id"]].head(3)
    texts = train.set_index("row_id")["text"]
    for _, r in ex.iterrows():
        print(f"  #{int(r['rank'])} sim={r['similarity']:.3f} target={int(r['case_target'])} "
              f"row_id={int(r['case_row_id'])}\n     {texts[int(r['case_row_id'])]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="nombre de clients de test (essai rapide)")
    run(ap.parse_args().n)