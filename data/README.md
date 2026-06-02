# Data

Place your dataset files in this folder.

All eight benchmark datasets used in the paper are publicly available:

- **Friedman #2** and **California Housing**: generated via `sklearn.datasets`
- **cpu_activity**, **Kin8nm**, **Protein**: [OpenML CTR23](https://openml.org/search?type=benchmark&study.id=336)
- **CCPP**, **Airfoil**, **Concrete**: [UCI Machine Learning Repository](https://archive.ics.uci.edu)

Datasets are not included in this repository due to licensing.
To reproduce results, download each dataset and save it here as an Excel or CSV file,
then update the FILE_CV path in cv_v2.py accordingly.
