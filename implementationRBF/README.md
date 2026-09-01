# Radial Basis Function (RBF) Networks

> RBF networks for classification and regression, implemented from scratch with NumPy + SciPy.

Gaussian basis functions project the input into a higher-dimensional space; a single output neuron then combines the basis responses with a learnable weight vector. Trained with online gradient descent on the **Q4** (binary classification) and **Q5** (regression) datasets.

## Highlights

- **K-means clustering** discovers the Gaussian basis centers automatically.
- Gaussian width derived from the spread of the centers (or set explicitly).
- Classifier uses a **sigmoid** output; regressor uses a **linear** output.
- Optional user-provided centers and widths for full control.
- Fully reproducible with an optional random seed.

## How it works

1. **Basis centers** — K-means partitions the input space into ``k`` clusters; each cluster center becomes the center of a Gaussian.
2. **Basis layer** — every sample is projected through ``k`` Gaussians:

   ```
   phi_j(x) = exp( - (||x - c_j|| / sigma)^2 )
   ```

3. **Output** — the basis responses are combined with a weight vector plus a bias:

   ```
   y = sigmoid(phi(x) · W + b)      # classifier
   y = phi(x) · W + b               # regressor
   ```

4. **Training** — weights are updated with the delta rule via online gradient descent.

## Project structure

```
implementationRBF/
├── RBF.ipynb                  # original notebook (reference, untouched)
├── Q4.csv                     # classification dataset (used by the notebook)
├── Q5.csv                     # regression dataset (used by the notebook)
├── README.md
├── requirements.txt
├── scripts/
│   ├── train_classifier.py    # Q4 binary classification demo
│   └── train_regressor.py     # Q5 regression demo
├── src/
│   ├── kmeans.py              # K-means clustering (basis centers)
│   ├── data_loader.py         # Q4 / Q5 loading
│   ├── evaluation.py          # accuracy + MSE helpers
│   ├── visualization.py       # cluster, confusion matrix, fit plots
│   └── models/
│       ├── rbf_classifier.py  # RBFClassifier
│       └── rbf_regressor.py   # RBFRegressor
└── tests/
    └── test_rbf.py            # pytest smoke tests
```

## Getting started

```bash
cd implementationRBF
pip install -r requirements.txt

# Binary classification on Q4
python scripts/train_classifier.py

# Regression on Q5
python scripts/train_regressor.py

# Run the test suite
pytest
```

## Results

| Task        | Dataset | Model                          | Metric           | Result     |
|-------------|---------|--------------------------------|------------------|------------|
| Classification | Q4   | RBF classifier (50 Gaussians)  | Test accuracy    | 99.00 %    |
| Regression  | Q5      | RBF regressor (auto centers)   | MSE              | 0.0023     |
| Regression  | Q5      | RBF regressor (manual, std=2)  | MSE              | 0.1659     |

## Running the notebook

```bash
jupyter notebook RBF.ipynb
```

---

**[← Back to repository home](../README.md)**
