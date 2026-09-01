# K-Nearest Neighbors

> A K-Nearest Neighbors classifier implemented from scratch with NumPy + SciPy distance metrics — no ML framework.

Classifies an unknown query point by majority vote among its ``K`` closest training neighbors. Evaluated on the classic **Iris** dataset with leave-one-out cross-validation.

## Highlights

- Three distance metrics: **Euclidean**, **Manhattan**, **Cosine**.
- Configurable ``K`` (number of neighbors).
- Efficient neighbor tracking that keeps only the ``K`` best candidates.
- Pure NumPy + SciPy — no sklearn classifier.

## How it works

1. **Distance** — compute the distance between the query point and every training sample.
2. **Selection** — keep the ``K`` closest neighbors.
3. **Voting** — return the majority-vote class among the selected neighbors.

| Code | Metric     |
|------|------------|
| 0    | Euclidean  |
| 1    | Manhattan  |
| 2    | Cosine     |

## Project structure

```
implementation_KNN/
├── implementation_KNN.ipynb   # original notebook (reference, untouched)
├── README.md
├── requirements.txt
├── scripts/
│   └── train_evaluate.py      # end-to-end experiment runner
├── src/
│   ├── data_loader.py         # Iris loading / shuffling
│   ├── evaluation.py          # leave-one-out CV + accuracy
│   ├── visualization.py       # pair plot, confusion matrix, K sweep plot
│   └── models/
│       └── knn.py             # the KNN classifier (from scratch)
└── tests/
    └── test_knn.py            # pytest smoke tests
```

## Getting started

```bash
cd implementation_KNN
pip install -r requirements.txt

# Run the full experiment (plots + printed accuracy)
python scripts/train_evaluate.py

# Run the test suite
pytest
```

## Results

The notebook reports a **leave-one-out accuracy of 96.0%** with ``K=20`` using the Manhattan distance on the Iris dataset.

## Running the notebook

```bash
jupyter notebook implementation_KNN.ipynb
```

---

**[← Back to repository home](../README.md)**
