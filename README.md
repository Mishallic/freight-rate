# Freight Rate Prediction

Predicts the posted rate (USD) of truck loads from distance, equipment, weight, pickup and
delivery coordinates, and day of week. A LightGBM model is trained on loads from January to
October 2025 and predicts 12,000 loads from November and December 2025. It is validated
with time-based folds that each predict the two months after their training data.

## Results

MAPE on clean test rows (rows with a corrupted rate removed):

| Model | Fold A: train Jan-Jun, test Jul-Aug | Fold B: train Jan-Aug, test Sep-Oct |
|---|---|---|
| Baseline: median rate per mile by distance band and equipment | 4.00% | 3.85% |
| LightGBM, final model (V1) | 2.26% | 1.91% |

On all test rows, corrupted rates included, the final model's MAPE is 4.51% and 4.42%.
Full tables are in `reports/metrics_models.csv`; the write-up is `reports/report.md`
(also as `reports/report.docx`).

## Repository layout

```
data/                        input data, unchanged
provided/                    the supplied scorer (score.py), README and requirements, unmodified
src/
  data.py                    loading, cleaning (prepare), corrupted-rate rule
  explore.py                 data facts and figures 1-4
  evaluate.py                time-based folds, metrics, baseline model
  model.py                   features and the LightGBM model
  compare.py                 baseline vs feature variants, extra checks, figure 5
  predict.py                 final model, prediction files, scorer run, sanity checks
  report_to_docx.py          converts reports/report.md to reports/report.docx
  style.py                   shared chart colors and style
tests/                       pytest tests for cleaning, folds, metrics and features
reports/                     write-up, metrics tables, figures
outputs/                     December chart predictions
validation_predictions.csv   final predictions: load_id, predicted_rate
```

## How to run

Tested with Python 3.14 on Windows. From the repository root:

```
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python src/predict.py
```

`src/predict.py` trains the final model on all labelled loads (January-October), writes
`validation_predictions.csv` and `outputs/december_predictions.csv`, runs the provided
scorer on both files, and copies the scorer's December chart to
`reports/figures/6_candidate_december.png`. To run the scorer on its own:

```
python provided/score.py --predictions validation_predictions.csv --december-predictions outputs/december_predictions.csv
```

The other scripts reproduce the analysis: `python src/explore.py` (data facts, figures 1-4),
`python src/evaluate.py` (baseline metrics), `python src/compare.py` (model comparison,
figure 5) and `python src/report_to_docx.py` (Word version of the report).

## Outputs

| File | Contents |
|---|---|
| `validation_predictions.csv` | 12,000 predictions in the template's order, columns `load_id,predicted_rate` |
| `outputs/december_predictions.csv` | the 31 December chart rows with `predicted_rate` filled in; `data/` is left unchanged |
| `scorer_results/candidate_december.png` | the scorer's December chart, created when the scorer runs (not tracked in git) |
| `reports/figures/` | charts 1-6; the report uses 2, 5 and 6 |
| `reports/metrics_*.csv` | validation metrics for the baseline, the feature variants, the settings check and the quote_signal check |
| `reports/report.md`, `reports/report.docx` | write-up of the data, validation, model and December chart |
