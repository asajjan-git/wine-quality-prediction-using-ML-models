# Data

This project uses the **white wine** subset of the UCI Wine Quality dataset:

- Dataset page: https://archive.ics.uci.edu/dataset/186/wine+quality
- Direct file: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
- License: CC BY 4.0
- Citation: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).
  Modeling wine preferences by data mining from physicochemical properties.
  *Decision Support Systems*, 47(4), 547–553.

The raw CSV is **not committed to this repo** — that's intentional. Committing
fetchable third-party data bloats repo size and can drift out of sync with the
source. Instead, fetch it once:

```bash
python download_data.py
```

This saves `winequality-white.csv` into this folder (semicolon-delimited, as
distributed by UCI). Both `python/wine_quality_analysis.py` and
`R/wine_quality_analysis.R` will also auto-download it on first run if it isn't
already here.

If you're offline, alternative mirrors that carry the same file:
- https://github.com/stedy/Machine-Learning-with-R-datasets
- Via the [`ucimlrepo`](https://pypi.org/project/ucimlrepo/) Python package:
  `pip install ucimlrepo` then `from ucimlrepo import fetch_ucirepo; d =
  fetch_ucirepo(id=186)`
