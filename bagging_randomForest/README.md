# Bagging Random Forest

> Decision trees (ID3 & C4.5) and a bagging random forest built entirely from scratch with NumPy — no ML framework.

A hand-coded implementation of **binary-split decision trees** using information-theoretic split criteria and a **Random Forest** that aggregates many bootstrapped trees by majority vote. Evaluated on the classic **Iris** dataset.

## Highlights

- **ID3 tree** — splits on the feature with maximum information gain (entropy reduction).
- **C4.5 tree** — splits on the feature with maximum gain ratio (normalised for many-valued features).
- **Random Forest** — bootstrap samples per tree, majority-vote predictions.
- Configurable **purity threshold** that controls when a node becomes a leaf.
- Fully reproducible with an optional random seed.

## How it works

1. **Split criterion** — at every node, each continuous feature is binarised at its midpoint threshold and the best feature is chosen by information gain (ID3) or gain ratio (C4.5).
2. **Stopping rule** — a node becomes a leaf once the dominant class reaches the configured purity threshold.
3. **Bagging** — each tree is grown on a bootstrap sample (sampling *with replacement*) of the training set.
4. **Voting** — the forest predicts by majority vote across all trees.

## Project structure

```
bagging_randomForest/
├── bagging_randomForest.ipynb    # original notebook (reference, untouched)
├── README.md
├── requirements.txt
├── scripts/
│   └── train_evaluate.py         # end-to-end experiment runner
├── src/
│   ├── data_loader.py            # Iris loading / splitting
│   ├── evaluation.py             # accuracy + batch prediction helpers
│   ├── visualization.py          # pair plot, confusion matrix, accuracy curves
│   └── models/
│       ├── decision_tree.py      # TreeNode, ID3Node, C45Node, DecisionTree
│       └── random_forest.py      # RandomForest
└── tests/
    └── test_random_forest.py     # pytest smoke tests
```

## Getting started

```bash
cd bagging_randomForest
pip install -r requirements.txt

# Run the full experiment (plots + printed accuracy)
python scripts/train_evaluate.py

# Run the test suite
pytest
```

## Results (Iris, 120 train / 30 test)

| Model          | Test accuracy |
|----------------|---------------|
| ID3 tree       | 90.00 %       |
| C4.5 tree      | 90.00 %       |
| Random Forest  | 93.33 %       |

## Running the notebook

The reference notebook is fully self-contained:

```bash
jupyter notebook bagging_randomForest.ipynb
```

---

**[← Back to repository home](../README.md)**
