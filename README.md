# Bank Marketing: Case-Based Reasoning

A small Bank Marketing project comparing an explainable Case-Based Reasoning (CBR) classifier with a k-nearest neighbors baseline. The Streamlit app demonstrates predictions, similar past clients, and online revise-and-retain learning.

## Setup

Use Python 3.9 or newer. From the project root, in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The raw dataset should be at `data/raw/bank-additional-full.csv`.

## Run

Prepare the train, validation, and test data:

```powershell
python src/preprocessing.py
```

Run the experiments (use `--quick` for a faster run), then launch the app:

```powershell
python src/experiments.py --quick
streamlit run src/app.py
```

## Project layout

- `src/` — preprocessing, CBR, k-NN baseline, experiments, and Streamlit app
- `data/raw/` — source dataset
- `data/processed/` — generated splits and model input arrays
- `results/` — experiment outputs used by the app
- `notebooks/` — exploratory data analysis
