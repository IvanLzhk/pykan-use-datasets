# Benchmarking Kolmogorov-Arnold Networks (KAN) vs. MLP

## Overview

This repository benchmarks Kolmogorov-Arnold Networks (KAN) against traditional Multi-Layer Perceptrons (MLP) on classical tabular classification tasks. Rather than comparing a single fixed-size model of each type, the experiments sweep both architectures across **matched parameter budgets**, so that accuracy differences reflect the architecture itself rather than model capacity.

KAN replaces MLP's fixed nonlinearities on nodes with learnable B-spline activation functions on edges, which — per [Liu et al., 2024](https://arxiv.org/abs/2404.19756) — can offer better accuracy and interpretability at comparable or lower parameter counts.

## Datasets

Four datasets from the UCI Machine Learning Repository are used, three for binary classification and one for multiclass:

| Folder | Dataset | Task | Source |
|---|---|---|---|
| `classification/breast-canser/` | Breast Cancer Wisconsin | Binary | `sklearn.datasets.load_breast_cancer` |
| `classification/musk/` | Musk | Binary | UCI `id=75` via `ucimlrepo` |
| `classification/gamma/` | MAGIC Gamma Telescope | Binary (gamma vs. hadron) | UCI `id=159` via `ucimlrepo` |
| `multiclassification/pykan-orign/` | Dry Bean | Multiclass, 7 classes | UCI `id=602` via `ucimlrepo` |

## Methodology

For each dataset:
1. **Matched parameter sweep** — both KAN and MLP are trained at multiple, closely matched parameter counts (not just one architecture each), so results can be plotted as accuracy vs. parameter count.
2. **5-fold stratified cross-validation** (`StratifiedKFold(n_splits=5)`) at each parameter budget.
3. **Metrics**: accuracy, precision, recall, and F1-score (macro-averaged for the multiclass Dry Bean task), reported as mean ± standard deviation across folds.
4. **KAN configuration**: grid size 3, spline order 3 (`KAN_g=3`, `KAN_k=3`), built with [pykan](https://github.com/KindXiaoming/pykan).
5. **MLP configuration**: scikit-learn's `MLPClassifier`, with hidden-layer widths chosen so the total parameter count lines up with each KAN budget.

## Results

Accuracy (mean) at matched parameter counts, summarized from the raw per-fold results:

- **Breast Cancer**: KAN's advantage is largest at low budgets — e.g. ~85% (KAN, 272 params) vs. ~63% (MLP, 251 params); both converge toward ~97–98% as parameter count grows.
- **Musk (Version 2)** *(folder labeled "diabetes")*: KAN outperforms MLP at every tested budget, e.g. ~94.8% vs. ~95.4% at the smallest budget, widening to ~99.2% vs. ~97.3% at the largest.
- **MAGIC Gamma Telescope**: the two architectures are close throughout, with KAN slightly ahead at most budgets and both converging to ~87.5% at the highest budget tested.
- **Dry Bean (multiclass)**: KAN and MLP track closely across all budgets (~92–93% accuracy), with KAN reaching a usable accuracy at a lower parameter count and using a narrower network (2 hidden layers of width 7 vs. MLP's 3 hidden layers of width 70) to hit the same budget.

Full per-fold numbers are in the raw `KAN_<n_params>.txt` / `MLP_<n_params>.txt` files in each dataset folder, and are visualized in each `plotResultByParams.ipynb`.

## Project Structure

```
classification/
├── breast-canser/
│   ├── classification.ipynb         # Trains & evaluates KAN and MLP on Breast Cancer Wisconsin
│   ├── plotResultByParams.ipynb     # Plots accuracy vs. parameter count from the .txt results
│   ├── KAN_<n_params>.txt           # Mean/std accuracy for a KAN at this parameter budget
│   └── MLP_<n_params>.txt           # Mean/std accuracy for an MLP at this parameter budget
├── musk/
│   └── ... (same structure as above)
└── gamma/
    └── ... (same structure as above)

multiclassification/
└── pykan-orign/
    ├── multiclassification.ipynb    # Trains & evaluates KAN and MLP on Dry Bean (7-class)
    ├── plotResultByParams.ipynb
    ├── KAN_<n_params>.txt
    └── MLP_<n_params>.txt
```

## Reference

Liu, Z. et al. (2024). *KAN: Kolmogorov-Arnold Networks*. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
