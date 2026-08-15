# 🍷 Wine Quality Prediction Using Machine Learning

Predicting the sensory quality score of Portuguese *Vinho Verde* white wine from
11 physicochemical lab measurements, and comparing a linear baseline against two
tree-ensemble models.

> Academic project — Statistical Learning, Indiana University (May 2025)
> Authors: Sai Pranam, Ahlad, Akhil Kumar (Group 4)

---

## Overview

Wine quality is traditionally assessed by expert sensory panels — a process that is
slow, expensive, and subjective. This project asks: **can measurable chemical
properties of a wine predict the quality score a human panel would give it?**

Using the [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
(4,898 white wine samples, 11 features, quality scored 0–10 by expert tasters), we
built and compared three models:

| Model | RMSE | R² | MAE |
|---|---|---|---|
| Multiple Linear Regression | 0.7594 | 0.2695 | 0.5897 |
| XGBoost | 0.6356 | 0.4887 | 0.4811 |
| **Random Forest** | **0.6089** | **0.5401** | **0.4447** |

**Random Forest was the best performer**, explaining ~54% of the variance in wine
quality. Across all three models, **alcohol content, volatile acidity, and
sulphates** were consistently the strongest predictors — findings that align with
established oenological knowledge and prior literature (Cortez et al., 2009).

---

## Repository structure

```
wine-quality-prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── README.md              # where the data comes from + how to fetch it
│   └── download_data.py       # pulls the CSV straight from UCI
├── python/
│   └── wine_quality_analysis.py
├── R/
│   └── wine_quality_analysis.R
├── figures/                   # generated plots land here
└── results/
    └── metrics_summary.csv    # the table above, machine-readable
```

## Dataset

- **Source:** [UCI Machine Learning Repository — Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
  (Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J., 2009. *Decision
  Support Systems*, 47(4), 547–553.)
- **Samples:** 4,898 white wine records
- **Features:** 11 continuous physicochemical measurements (fixed acidity, volatile
  acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide,
  density, pH, sulphates, alcohol)
- **Target:** `quality`, an integer score from 0–10 assigned by wine experts
  (observed range in this dataset: 3–9)
- **License:** CC BY 4.0 — free to use with attribution

The raw CSV isn't committed to this repo (standard practice — don't check in data
you can fetch). Run `python data/download_data.py` or just execute either analysis
script, which downloads it automatically on first run. See `data/README.md`.

## Methodology

1. **Preprocessing** — checked for missing values (none found), examined outliers
   via IQR (kept, as they reflect legitimate variation), applied Z-score
   normalization to all predictors, 70/30 train-test split.
2. **Models**
   - **Multiple Linear Regression** — baseline, all 11 predictors entered linearly.
   - **Random Forest Regression** — 500 trees, 3 variables considered per split.
   - **XGBoost** — `max_depth=6`, `eta=0.1`, `nrounds=100`, with cross-validation.
3. **Evaluation** — RMSE, MAE, and R² on the held-out 30% test set.

Full methodological detail and discussion of results is in the original report
(not included here — see the write-up section below if you want to add it).

## Key findings

- **Alcohol** has the strongest positive correlation with quality; **density** and
  **volatile acidity** have the strongest negative correlations.
- Linear regression captures only ~27% of the variance — wine quality is driven by
  **non-linear interactions** between chemical properties that tree-based models
  capture far better.
- Random Forest edged out XGBoost here, likely because the dataset is small/
  moderate in size and the untuned XGBoost hyperparameters left some performance
  on the table (a natural next step: grid/Bayesian search over XGBoost's
  hyperparameters).

## How to run it

### Python
```bash
cd python
pip install -r ../requirements.txt
python wine_quality_analysis.py
```

### R
```bash
cd R
Rscript wine_quality_analysis.R
```

Both scripts download the dataset automatically, run the full pipeline (EDA →
preprocessing → all three models → evaluation), print the metrics table, and
save figures (correlation heatmap, feature importance, model comparison bar
chart) to `figures/`.

## Limitations & future work

- Quality labels come from human tasters — inherently subjective, single snapshot
  in time (no vintage/aging effects captured).
- Scope limited to Portuguese white wine; doesn't generalize to reds or other
  regions without retraining.
- XGBoost hyperparameters were fixed, not tuned via grid/random/Bayesian search.
- Natural extensions: hyperparameter tuning, SHAP-based interpretability, deep
  learning / hybrid models, incorporating vineyard/terroir data.

## References

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling
  wine preferences by data mining from physicochemical properties. *Decision
  Support Systems*, 47(4), 547–553.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
  *Proceedings of the 22nd ACM SIGKDD*, 785–794.
- Gutiérrez, J., & Gutiérrez, P. A. (2021). Wine quality prediction using machine
  learning techniques. *Sensors*, 21(2), 562.

## License

MIT — see [LICENSE](LICENSE). Note the dataset itself is CC BY 4.0 from UCI.
