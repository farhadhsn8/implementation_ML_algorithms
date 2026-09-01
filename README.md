# Machine Learning Algorithms — From Scratch

[![Python 3](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#)
[![NumPy](https://img.shields.io/badge/NumPy-Core%20dependency-013243.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Built for learning](https://img.shields.io/badge/Built%20for-learning-purple.svg)](#)

**Clean, dependency-light implementations of classic machine learning algorithms built from first principles with NumPy** — no black boxes, no `sklearn` classifiers. Every model lives in its own self-contained package with source modules, runnable experiments, tests, and documentation.

## About

This repository re-implements the foundations of modern machine learning from the math up:

- **K-Nearest Neighbors** — instance-based classification with configurable distance metrics.
- **Gaussian Naive Bayes** — probabilistic classification under conditional independence.
- **Multi-Layer Perceptron** — fully connected neural networks trained with backpropagation.
- **Radial Basis Function Networks** — kernel-style networks for classification and regression.
- **Bagging Random Forest** — ID3 / C4.5 decision trees aggregated by bootstrap voting.

Each algorithm is evaluated on a standard dataset (Iris, Salary, and two synthetic sets), with accuracy and error metrics reported alongside every run.

## Projects

| # | Project | Algorithm | Task | Dataset | Test performance |
|---|---------|-----------|------|---------|------------------|
| 1 | [K-Nearest Neighbors](implementation_KNN/) | KNN (Euclidean / Manhattan / Cosine) | Classification | Iris | 96.0 % (LOO) |
| 2 | [Gaussian Naive Bayes](implementation_GaussianNaiveBayes/) | Gaussian NB | Classification | Iris | 93.3 % |
| 3 | [Dynamic MLP](implementationDynamicMLP/) | MLP + backpropagation | Classification & regression | Iris / Salary | 96.7 % |
| 4 | [RBF Networks](implementationRBF/) | RBF classifier & regressor | Classification & regression | Q4 / Q5 | 99.0 % |
| 5 | [Bagging Random Forest](bagging_randomForest/) | ID3, C4.5, Random Forest | Classification | Iris | 93.3 % |

## Highlights

- **Truly from scratch** — decision trees, entropy, Gaussians, backpropagation and K-means are all hand-coded in NumPy.
- **Flexible MLP** — configure layers, sizes and per-layer activation functions (sigmoid, tanh, ReLU, linear).
- **Production-friendly packages** — each project has a clean `src/` layout, runnable `scripts/`, `requirements.txt` and `pytest` tests.
- **Reproducible** — deterministic results via optional random seeds.
- **Well documented** — every module is docstring-documented and each project has its own README.

## Tech stack

- **Python 3.9+**
- **NumPy / SciPy** — the computational core
- **scikit-learn** — used only for datasets and evaluation metrics (never for the algorithms)
- **Matplotlib / Seaborn** — visualization
- **Jupyter** — reference notebooks

## Repository structure

```
implementation_ML_algorithms/
├── implementation_KNN/               # K-Nearest Neighbors
├── implementation_GaussianNaiveBayes/# Gaussian Naive Bayes
├── implementationDynamicMLP/         # Multi-Layer Perceptron + backpropagation
├── implementationRBF/                # RBF networks (classifier + regressor)
├── bagging_randomForest/             # Decision trees + bagging random forest
├── GITHUB.md                         # GitHub About & topics cheat-sheet
├── README.md
└── LICENSE
```

Each project follows the same pattern:

```
project/
├── project.ipynb       # original reference notebook (untouched)
├── README.md           # project documentation
├── requirements.txt
├── scripts/            # runnable experiment scripts
├── src/                # reusable, documented source modules
└── tests/              # pytest test suite
```

## Getting started

Clone the repository and run any project:

```bash
git clone https://github.com/farhadhsn8/implementation_ML_algorithms.git
cd implementation_ML_algorithms

# Pick a project, e.g. the random forest
cd bagging_randomForest
pip install -r requirements.txt
python scripts/train_evaluate.py
```

Run the tests for every project:

```bash
cd implementation_KNN && pytest -q
cd ../implementation_GaussianNaiveBayes && pytest -q
cd ../implementationDynamicMLP && pytest -q
cd ../implementationRBF && pytest -q
cd ../bagging_randomForest && pytest -q
```

## Development

Contributions are welcome. Ideas for extending the repository:

- Add **softmax + cross-entropy** output to the MLP for improved multi-class training.
- Add **SVM** or **linear/logistic regression** from scratch.
- Benchmark each from-scratch model against its `sklearn` equivalent.

## License

Distributed under the [MIT License](LICENSE).

## Author

**Seyed Farhad Hosseini** — Machine Learning & Deep Learning engineer.
