# What was corrected, and why

The structure of the project is the same (same files, same functions, same two experiments).
These are the methodological changes, with the reason for each.

| # | Before | Now | Why |
|---|---|---|---|
| 1 | 12 exact duplicate rows kept | Removed | The same call recorded twice (even the same duration) |
| 2 | Split 70 / 15 / 15 | 60 / 20 / 20, stratified, seed 42 | A larger validation set makes the tuning more reliable |
| 3 | Min–max scaling | 1st–99th percentile scaling of the training clients, values capped | `campaign` goes up to 56 while 99% of clients have ≤ 14: with min–max, 1 call and 2 calls looked almost identical |
| 4 | `pdays` = days (0 if never) + 0/1 flag | `pdays_group`: none / ≤6 days / >6 days | 999 is a code, not a number of days. 6 is the median of the recorded values in the training part |
| 5 | Weights set by hand per group (3 presets) | 3 schemes compared on validation: equal, economy grouped, **mutual information** computed on the training clients | The weights come from the data, and we check on validation that they beat equal weights |
| 6 | k tested up to 81, and 81 won (edge of the grid) | k from 1 to 201 | The best k must be inside the tested range: 151 now |
| 7 | Decision threshold and "power" tuned on validation | **Balanced vote**: yes if the (similarity-weighted) share of subscribers among the neighbours ≥ share among all stored clients (~11%) | Same idea (correct for the 11% imbalance) with no extra parameter. It equals dividing each vote by the frequency of its class |
| 8 | Final case base = training only | Training + validation, with every setting frozen first | Uses more cases for the final experiments, without leakage |
| 9 | k-NN compared with its standard recipe only | Also **k-NN with the same representation** | It finds exactly the same neighbours (distance difference ≈ 1e-16). That answers "isn't this just k-NN?": the difference is the vote and the rest of the cycle |
| 10 | Revise always lowered the reliability of misleading neighbours | Revise keeps the real outcome as the solution; the reliability rule is kept as an **option** and tested in Experiment 2 | It slightly hurts (0.745 vs 0.754), which is an interesting result to discuss |
| 11 | float32, ties broken arbitrarily | float64, ties broken by case order | Reproducible results |
| 12 | `app.py` imported `cbr`, the others `src.cbr` | All use `src.cbr` | `streamlit run src/app.py` works from the project root |
| 13 | No tests | `tests/test_cbr.py` (13 tests) | Leakage, cycle order, weights, ties, equivalence with k-NN |

## New results (test set, 8,236 clients)

| Method | Balanced accuracy | 95% interval |
|---|---|---|
| CBR (mutual information weights, k = 151, similarity-weighted balanced vote) | **0.755** | [0.739, 0.770] |
| k-NN, same representation, balanced vote | 0.757 | [0.741, 0.772] |
| k-NN standard, tuned on validation (k = 201, balanced rule) | 0.753 | [0.739, 0.768] |
| k-NN, same representation, majority vote | 0.585 | [0.574, 0.597] |
| Always "no" | 0.500 | |

Experiment 2 (test clients one by one):

| Variant | Balanced accuracy | Decisions changed vs no learning |
|---|---|---|
| without learning | 0.755 | 0 |
| retain only | 0.754 | 105 |
| revise (reliability) + retain | 0.745 | 310 |

## Run

```bash
pip install -r requirements.txt
python src/preprocessing.py
python -m src.experiments              # about 2 minutes
python -m unittest discover -s tests -t .
streamlit run src/app.py               # old interface (updated)
# web demo: open web/index.html (rebuild: python web/build_web_data.py && python web/build_web.py)
```
