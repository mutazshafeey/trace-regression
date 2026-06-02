# TRACE: Transparent Regression with Adaptive Confidence Estimation

[![Paper](https://img.shields.io/badge/Nature_Machine_Intelligence-under_review-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

TRACE is an interpretable regression framework that operates directly 
on unprocessed numerical data without preprocessing or normalization. 
Every prediction is fully auditable through matched training instances, 
learned feature sensitivity exponents and a native per-prediction 
confidence score.

## Repository structure

```
trace-regression/
├── cv_v2.py                 # TRACE cross-validation and core algorithm
├── benchmark_models.py      # All 10 baseline models with 5-fold CV
├── wilcoxon_test.py         # Corrected resampled t-tests
├── pareto_mean_rank.py      # Figure 2: Pareto frontier plot
├── figure1_audit.py         # Figure 1: Prediction audit trace
├── confidence_viz.py        # Confidence stratification figures
└── requirements.txt         # Python dependencies
```
## Requirements

Python 3.8 or later. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Datasets

All eight benchmark datasets are publicly available:

- Friedman #2 and California Housing: `sklearn.datasets`
- cpu_activity, Kin8nm, Protein: [OpenML CTR23](https://openml.org)
- CCPP, Airfoil, Concrete: [UCI Machine Learning Repository](https://archive.ics.uci.edu)

## Reproducing results

Run TRACE cross-validation on a dataset:

```bash
python cv_v2.py
```

Run all baseline models:

```bash
python benchmark_models.py
```

## Citation

If you use this code please cite:
@misc{shafeey2025trace,
author = {Shafeey, Mutaz},
title  = {TRACE: Transparent Regression with Adaptive
Confidence Estimation},
year   = {2025},
url    = {https://github.com/mutazshafeey/trace-regression}
}

## License

MIT License — see [LICENSE](LICENSE) for details.
