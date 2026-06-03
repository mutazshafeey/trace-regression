# TRACE: Transparent Regression with Adaptive Confidence Estimation

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Code repository for the paper:
**TRACE: Transparent Regression with Adaptive Confidence Estimation**
Submitted to *Nature Machine Intelligence*

---

## Overview

TRACE is an interpretable regression framework that operates directly on unprocessed numerical data without preprocessing or normalization. Every prediction is fully auditable through matched training instances, learned feature sensitivity exponents and a native per-prediction confidence score that quantifies individual prediction reliability without post-hoc calibration.

---

## Repository structure

```
trace-regression/
├── cv_v2.py                 # TRACE core algorithm with 5-fold cross-validation
├── benchmark_models.py      # All 10 baseline models with 5-fold cross-validation
├── wilcoxon_test.py         # Corrected resampled t-test (Nadeau and Bengio 2003)
├── pareto_mean_rank.py      # Interpretability-performance Pareto frontier figure
├── figure_audit.py          # Prediction-level audit trail figure (4 panels)
├── requirements.txt         # Python dependencies
└── data/
└── README.md                # Dataset download instructions
```
## Requirements

Python 3.8 or later. Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Datasets

All eight benchmark datasets are publicly available. No data files are included in this repository.

| Dataset | Source |
|---------|--------|
| Friedman #2, California Housing | `sklearn.datasets` |
| cpu\_activity, Kin8nm, Protein | OpenML CTR23 |
| CCPP, Airfoil, Concrete | UCI Machine Learning Repository |

See `data/README.md` for full download instructions.

---

## Reproducing results

All scripts use a generic settings block at the top. Set your dataset path and target column before running:

```python
FILE_CV    = "data/your_dataset.xlsx"  # or .csv
SHEET_CV   = "Sheet1"
TARGET_COL = "y"
```

**Run order:**

1. Run TRACE cross-validation on a dataset:
```bash
python cv_v2.py
```

2. Run all baseline models under the same protocol:
```bash
python benchmark_models.py
```

3. Run statistical significance tests (requires steps 1 and 2 in the same session):
```bash
python wilcoxon_test.py
```

4. Generate the Pareto frontier figure (standalone):
```bash
python pareto_mean_rank.py
```

5. Generate the audit trail figure (requires cv_v2.py session variables):
```bash
python figure_audit.py
```

---

## Citation

If you use this code please cite:

```bibtex
@misc{shafeey2025trace,
  author = {Shafeey, Mutaz},
  title  = {TRACE: Transparent Regression with Adaptive Confidence Estimation},
  year   = {2025},
  url    = {https://github.com/mutazshafeey/trace-regression}
}
```

---

## License

MIT License — see `LICENSE` for details.
