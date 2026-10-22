# Web demo

Open `web/index.html` in any browser (double-click). It needs no install and no internet: the engine, the data and the charts are all inside the page.

## Pages

- **Overview**: the project in a few numbers.
- **Predict a new client**: fill in a client (or start from a typical client or a real test client). The prediction updates live. You see:
  - the past clients behind it (similarity, outcome, reliability);
  - where the similarity comes from, group by group;
  - what one change controlled by the bank (contact type, day, number of calls) would do.

  Then enter the real outcome: **Revise** keeps the real outcome as the solution (optionally lowering the reliability of the neighbours that misled the prediction), and **Retain** adds the client to the case base. Predict again and the client comes back first with similarity 1.000.
- **Results**: Experiment 1 (CBR vs k-NN, with 95% intervals), the validation tuning, and Experiment 2 (one by one).
- **How it works**: the four steps, the formulas and the weight presets.

## Rebuild after rerunning the experiments

From the project root:

```bash
python src/preprocessing.py
python -m src.experiments
python web/build_web_data.py     # exports data + results to web/data.json
python web/build_web.py          # writes web/index.html
node web/verify_engine.js        # optional: JS engine must agree with src/cbr.py
```

`web/engine.js` is a line-by-line JavaScript version of `src/cbr.py`. `verify_engine.js` checks it against Python on 300 test clients: identical predictions (static, retain only, and revise with reliability, including the reliability values). One score in 300 differs slightly, because two neighbours tied at position 151 are rounded differently by Python's matrix product and by JavaScript; the prediction is the same.

## Changes to the method

See `CHANGES.md` at the project root.
