# Gaussian Naive Bayes

> A Gaussian Naive Bayes classifier implemented from scratch with NumPy — no ML framework.

Models every feature/class pair with a univariate Gaussian distribution and classifies using maximum log-likelihood under the naive (conditional independence) assumption. Evaluated on the classic **Iris** dataset.

## Highlights

- Per-class, per-feature Gaussian parameters (mean & std) fitted from data.
- Numerically stable **log-likelihood** scoring with class priors.
- Pure NumPy + `math` — no sklearn classifier.
- Deterministic with an optional random seed.

## How it works

1. **Fit** — for each feature and each class, fit a univariate Gaussian to the training samples of that class.
2. **Predict** — for a new sample, compute the log-likelihood of each class:

   ```
   log P(class) + sum_j log P(feature_j | class)
   ```

3. **Decision** — return the class with the highest log-likelihood.

## Project structure

```
implementation_GaussianNaiveBayes/
├── implementation_GaussianNaiveBayes.ipynb   # original notebook (reference, untouched)
├── README.md
├── requirements.txt
├── scripts/
│   └── train_evaluate.py                     # end-to-end experiment runner
├── src/
│   ├── data_loader.py                        # Iris loading (one-hot labels)
│   ├── evaluation.py                         # accuracy + prediction helpers
│   ├── visualization.py                      # pair plot, confusion matrix
│   └── models/
│       └── gaussian_naive_bayes.py           # Gaussian + GaussianNaiveBayes
└── tests/
    └── test_gaussian_naive_bayes.py          # pytest smoke tests
```

## Getting started

```bash
cd implementation_GaussianNaiveBayes
pip install -r requirements.txt

# Run the full experiment (plots + printed accuracy)
python scripts/train_evaluate.py

# Run the test suite
pytest
```

## Results (Iris, 120 train / 30 test)

| Split | Accuracy |
|-------|----------|
| Train | 95.83 %  |
| Test  | 93.33 %  |

## Running the notebook

```bash
jupyter notebook implementation_GaussianNaiveBayes.ipynb
```

---

**[← Back to repository home](../README.md)**
